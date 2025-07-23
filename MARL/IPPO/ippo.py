import os
import random
import time
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import cv2
import imageio
# import ale_py
from pettingzoo.atari import pong_v3
import importlib
import supersuit as ss

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-PettingZoo-PongIPPO"
    seed = 42
    env_id = "pong_v3"
    total_timesteps = 10_000_000  # Standard metric for vectorized training

    # PPO & Agent settings
    lr = 2.5e-4
    gamma = 0.99
    num_envs = 16  # Number of parallel environments
    max_steps = 128  # Steps per rollout per environment (aka num_steps)
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.1 
    clip_coeff = 0.1  # Value clipping coefficient
    ENTROPY_COEFF = 0.01
    
    VALUE_COEFF = 0.5
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"
    
    GAE = 0.95  # Generalized Advantage Estimation
    anneal_lr = True  # Whether to linearly decay the learning rate
    max_grad_norm = 0.5  # Gradient clipping value
    num_agents = 2  # Number of agents in the environment
    
    # Derived values
    @property
    def batch_size(self):
        return self.num_envs * self.max_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

# --- Preprocessing ---
TARGET_HEIGHT = 64
TARGET_WIDTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, action_space):
        super(Agent, self).__init__()
        # Shared CNN feature extractor
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )
        # Actor head
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)
        # Critic head
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_features(self, x):
        return self.network(x)

    def get_value(self, x):  
        x = x.clone()
        x = x.permute(0, 3, 1, 2)
        x[:, :4, :, :] /= 255.0
        return self.critic(self.get_features(x))

    def get_action(self, x, action=None, deterministic=False):
        # print("No eval: ", x.shape)
        x = x.clone()
        x = x.permute(0, 3, 1, 2)
        x[:, :4, :, :] /= 255.0
        
        features = self.get_features(x)
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def evaluate_get_action(self, x, action):
        # print("Eval: ", x.shape)
        x = x.clone()
        x = x.permute(0, 3, 1, 2)
        x[:, :4, :, :] /= 255.0
        features = self.get_features(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

# --- Environment Creation ---
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.reset(seed=seed)  # <--- Required to initialize np_random
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    print(env.observation_space)
    print(env.action_space)
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    # envs.is_vector_env = True
    # envs = gym.wrappers.RecordEpisodeStatistics(env)
  
    return envs

def evaluate(model, device, seed, num_eval_eps=10, record=False):
    eval_env = pong_v3.env(render_mode="rgb_array" if record else None)
    env = ss.max_observation_v0(eval_env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    eval_env = ss.agent_indicator_v0(env, type_only=False)

    for agt_idx in range(args.num_agents):
        model[agt_idx].to(device)
        model[agt_idx].eval()
    
    all_episode_rewards = {agent: [] for agent in eval_env.possible_agents}
    frames = []

    for i in tqdm(range(num_eval_eps), desc="Evaluating"):
        eval_env.reset(seed=args.seed)
        episode_rewards = {agent: 0 for agent in eval_env.possible_agents}
        for agent in eval_env.agent_iter():
            obs, reward, terminated, truncated, info = eval_env.last()
            done = terminated or truncated
            episode_rewards[agent] += reward
            if done:
                eval_env.step(None)
                continue
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
                action, _, _ = model[agt_idx].get_action(obs_tensor, deterministic=False)
            eval_env.step(action.cpu().item())
            if record:
                frames.append(eval_env.render())
        for agent, r in episode_rewards.items():
            all_episode_rewards[agent].append(r)

    eval_env.close()
    for agt_idx in range(args.num_agents):
        model[agt_idx].train()
    
    avg_return1 = np.mean(all_episode_rewards['first_0'])
    avg_return2 = np.mean(all_episode_rewards['second_0'])
    return all_episode_rewards['first_0'], all_episode_rewards['second_0'], avg_return1, avg_return2, frames
    



# --- Checkpoint Saving Function ---
def save_checkpoint(model, optimizer, update, global_step, args, run_name, save_dir="/content/checkpoints/ippo"):
    """Save model and optimizer state as a checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"actor_network_{run_name}_update{update}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': update,
        'global_step': global_step,
        'args': vars(args)
    }, model_path)
    print(f"Model checkpoint saved at {model_path}")
    
# --- Main Execution ---
if __name__ == "__main__":
    args = Config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    # 3. Create multiple parallel games
    envs = make_env(env_id=args.env_id, seed=args.seed, idx=0, run_name=run_name)
    # print(env.single_action_space)
    # envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    
    actor_network_ls = [ Agent(envs.single_action_space.n) for _ in range(args.num_agents)]
    optimizers = [optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5) for actor in actor_network_ls]
    for actor in actor_network_ls:
        actor.to(device)
        
    obs_storage = torch.zeros((args.max_steps, args.num_envs, ) + envs.observation_space.shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)

    # Episode tracking variables
    episodic_return_reward = np.zeros((args.num_envs, args.num_agents))
    episode_step_count = np.zeros((args.num_envs, args.num_agents))

    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    # next_done_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    bootstrap_value = torch.zeros(args.num_envs).to(device)
    # next_obs_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        
        frac = 1.0 - (update / num_updates)
        lr = args.lr * frac
        
        for agent_idx, actor in enumerate(actor_network_ls):
            for param_group in optimizers[agent_idx].param_groups:
                param_group['lr'] = lr

        for step in range(0, args.max_steps):
            
            global_step += args.num_envs * 2
            masks = []
            for agent_idx in range(args.num_agents):
                
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 1.0 #1d tensor
                masks.append(mask)
                
            obs_storage[step] = next_obs
            
            
            actions_per_step = torch.zeros(args.num_envs, device=device)
            
            for agent_idx in range(args.num_agents):
                
                agent_obs = next_obs[masks[agent_idx]]
                with torch.no_grad():
                    action, logprob, _ = actor_network_ls[agent_idx].get_action(agent_obs)
                    value = actor_network_ls[agent_idx].get_value(agent_obs)
                # print
                # actions_storage = actions_storage.long()
                actions_per_step = actions_per_step.long()
                actions_per_step[masks[agent_idx]] = action.long()
                # print(action.shape, value.shape)
                values_storage[step][masks[agent_idx]] = value.flatten()
                # actions_storage[step][mask[agent_idx]] = action
                logprobs_storage[step][masks[agent_idx]] = logprob
                
                
            actions_storage[step] = actions_per_step
            next_obs, reward, terminated, truncated, info = envs.step(actions_per_step.cpu().numpy())
            # print(next_obs)
            done = np.logical_or(terminated, truncated)
            # print(next_obs.shape, reward.shape, done.shape)
          
                        
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            # yy
            dones_storage[step] = next_done

        with torch.no_grad():         
        #Creating masks for GAE
            masks = []
            for agent_idx in range(args.num_agents):
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 1.0
                masks.append(mask)

            advantages = torch.zeros(( args.max_steps, args.num_envs)).to(device)

            for agent_idx in range(args.num_agents):
        # for t in reversed(range(args.max_steps)):                  
            # === Advantage Calculation & Returns 
            
                lastgae = 0
                #Bootstrap VALUE CALCULATE ONCE PER AGENT
                bootstrap_value[masks[agent_idx]] = actor_network_ls[agent_idx].get_value(next_obs[masks[agent_idx]]).squeeze()
                
                for t in reversed(range(args.max_steps)):  
               
        
                    if t == args.max_steps - 1:
                        nextnonterminal = (1.0 - next_done[masks[agent_idx]])
                        gt_next_state = bootstrap_value[masks[agent_idx]] * nextnonterminal
                    else:
                        nextnonterminal = (1.0 - dones_storage[t + 1][masks[agent_idx]])
                        gt_next_state = values_storage[t + 1][masks[agent_idx]] * nextnonterminal # If done at t, the next gt is 0

                    delta = (rewards_storage[t][masks[agent_idx]] +  args.gamma *  gt_next_state ) - values_storage[t][masks[agent_idx]]

                    advantages[t][masks[agent_idx]] = lastgae = delta + args.GAE * lastgae * nextnonterminal * args.gamma

        # Calculate advantages using the computed returns and stored values
        returns = advantages + values_storage
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === PPO Update Phase ===
        b_obs = obs_storage.reshape((-1,) +  envs.observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)

        masks = []
        # Create masks for each agent based on the next_obs
        
        # NICE IDEA BY DEEPSEEK TO JUST CHOOSE THE RANDOM BATCHES BASED ON MASKS FIRST
        for agent_idx in range(args.num_agents):
            mask = next_obs[:, 0, 0, (4 + agent_idx)] == 1.0
            masks.append(mask)
            
            # Get all indices for this agent
            agent_indices = torch.where(masks[agent_idx])[0]

            # Skip if no samples for this agent
            if len(agent_indices) == 0:
                continue
                
            # Shuffle indices for this agent
            agent_indices = agent_indices[torch.randperm(len(agent_indices))]
            
            # Minibatch update
            for epoch in range(args.PPO_EPOCHS):
                for start in range(0, len(agent_indices), args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = agent_indices[start:end]
                    
                    # Get minibatch data
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_values = b_values[mb_inds]

                    # Calculate losses
                    new_log_probs, entropy = actor_network_ls[agent_idx].evaluate_get_action(mb_obs, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_logprobs)
                    
                    # Policy loss
                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    current_values = actor_network_ls[agent_idx].get_value(mb_obs).squeeze()
                    v_loss_unclipped = (current_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(current_values - mb_values, -args.clip_coeff, args.clip_coeff)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    critic_loss = args.VALUE_COEFF * 0.5 * v_loss_max.mean()
                    
                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = policy_loss - args.ENTROPY_COEFF * entropy_loss + critic_loss

                    optimizers[agent_idx].zero_grad()
                    
                    loss.backward()
                    
                    grad_norm_dict = {}
                    total_norm = 0
                    for name, param in actor_network_ls[agent_idx].named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            if 'actor' in name or 'critic' in name:
                                grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                            else:
                                grad_norm_dict[f"gradients/shared_norm_{name}"] = param_norm.item()
                            total_norm += param_norm.item() ** 2
                    grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
                    wandb.log(grad_norm_dict)
                    
                    nn.utils.clip_grad_norm_(actor_network_ls[agent_idx].parameters(), 0.5)
                    # nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                    # actor_optimizer.step()
                    optimizers[agent_idx].step()
                    

            if args.use_wandb:
                wandb.log({ 
                    f"losses/total_loss_agent{agent_idx}": loss.item(),
                    f"losses/policy_loss_agent{agent_idx}": policy_loss.item(),
                    f"losses/value_loss_agent{agent_idx}": critic_loss.item(),
                    f"losses/entropy_agent{agent_idx}": entropy_loss.item(),
                    f"charts/learning_rate_agent{agent_idx}": optimizers[agent_idx].param_groups[0]['lr'],
                    f"charts/avg_rewards_agent{agent_idx}": np.mean(rewards_storage[:, agent_idx].cpu().numpy()),
                    f"charts/advantages_mean_agent{agent_idx}": b_advantages.mean().item(),
                    f"charts/advantages_std_agent{agent_idx}": b_advantages.std().item(),
                    f"charts/returns_mean_agent{agent_idx}": b_returns.mean().item(),
                    "global_step": global_step,
                })
                # print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
        

        if update % 50 == 0:
            rewards_player1, rewards_player2, avg_return1, avg_return2, _ = evaluate(actor_network_ls, device, run_name, num_eval_eps=5, record=False)
            # Log the average return from the evaluation
            # avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return_player1": avg_return1,
                    "eval/avg_return_player2": avg_return2,
                    "global_step": global_step,
                })
            print("Rewards from evaluation:", rewards_player1, '   ', rewards_player2)
            print(f"Evaluation at step {global_step}: Average raw return for player 1  = {avg_return1:.2f}, Average raw return for player 2  = {avg_return2:.2f}")

        # Save the model at intervals of 200 updates
        if update % 200 == 0:
            for idx, (actor, optim) in enumerate(zip(actor_network_ls, optimizers)):
                save_checkpoint(actor, optim, update, global_step, args, f"{run_name}_agent{idx}")

    if args.capture_video:
        print("Capturing final evaluation video...")
        _, _, _, _, eval_frames = evaluate(actor_network_ls, device, run_name, num_eval_eps=10, record=True)

        if len(eval_frames) > 0:
            video_path = f"final_eval_{run_name}.mp4"
            # os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                print(f"Final evaluation video saved and uploaded to WandB.")

    for idx, (actor, optim) in enumerate(zip(actor_network_ls, optimizers)):
        save_checkpoint(actor, optim, num_updates, global_step, args, f"{run_name}_agent{idx}")
    envs.close()
    if args.use_wandb:
        wandb.finish()