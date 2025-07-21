

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
from pettingzoo.atari import pong_v3

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-PettingZoo-Pong"
    seed = 42
    env_id = "pong_v3"
    total_timesteps = 15_000_000  # Standard metric for vectorized training

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
# --- Evaluation Function ---
def evaluate(model, device, seed, num_eval_eps=10, record=False):
    eval_env = pong_v3.env(render_mode="rgb_array" if record else None)
    env = ss.max_observation_v0(eval_env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    eval_env = ss.agent_indicator_v0(env, type_only=False)

    model.to(device)
    model.eval()
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
                action, _, _ = model.get_action(obs_tensor, deterministic=False)
            eval_env.step(action.cpu().item())
            if record:
                frames.append(eval_env.render())
        for agent, r in episode_rewards.items():
            all_episode_rewards[agent].append(r)

    eval_env.close()
    model.train()
    avg_return1 = np.mean(all_episode_rewards['first_0'])
    avg_return2 = np.mean(all_episode_rewards['second_0'])
    return all_episode_rewards['first_0'], all_episode_rewards['second_0'], avg_return1, avg_return2, frames
    



# --- Checkpoint Saving Function ---
def save_checkpoint(model, optimizer, update, global_step, args, run_name, save_dir="/content/checkpoints"):
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
    
    actor_network = Agent(envs.action_space.n).to(device)
    
    optimizer = optim.Adam(actor_network.parameters(), lr=args.lr, eps=1e-5)
    # critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

    obs_storage = torch.zeros((args.max_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    
    # Episode tracking variables
    episodic_return = np.zeros(args.num_envs)
    episode_step_count = np.zeros(args.num_envs)
    
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        
        frac = 1.0 - (update / num_updates)
        lr = args.lr * frac
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
      
        
        for step in range(0, args.max_steps):
            global_step += args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            # for agent in envs.agent_iter():
                
            
            # if done:
            #     envs.step(None)
            #     continue
            
            # elif agent == 'first_0':
            
            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_obs)
                value = actor_network.get_value(next_obs)
            
            values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            # print(next_obs)
            done = np.logical_or(terminated, truncated)
            
            # Update episode tracking
            episodic_return += reward
            episode_step_count += 1
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            # else:
            #     # For other agents, just sample random actions
            #     action = envs.single_action_space(agent).sample()
            #     envs.step(action)
            #     rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
                # next_obs = torch.Tensor(obs).to(device)
                # next_done = torch.Tensor(done).to(device)
                
                
            if "final_info" in info:
                for item in info["final_info"]:
                    # The item can be None if the env at that index is not done
                    if item and "episode" in item:
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        wandb.log({
                            "rollout/episodic_return": item['episode']['r'],
                            "roolout/episodic_length": item['episode']['l'],
                            "global_step": global_step
                        })
        # === Advantage Calculation & Returns (YOUR ORIGINAL LOGIC) ===
        with torch.no_grad():
            advantages = torch.zeros_like(rewards_storage).to(device)
            
            # 1. Bootstrap value: Get value of the state *after*
            bootstrap_value = actor_network.get_value(next_obs).squeeze()
            lastgae = 0.0

            for t in reversed(range(args.max_steps)):
                
                if t == args.max_steps - 1:
                    nextnonterminal = (1.0 - next_done)
                    gt_next_state = bootstrap_value * nextnonterminal
                else:
                    nextnonterminal = (1.0 - dones_storage[t + 1])
                    gt_next_state = values_storage[t + 1] * nextnonterminal # If done at t, the next gt is 0
                
                delta = (rewards_storage[t] +  args.gamma *  gt_next_state ) - values_storage[t]

                advantages[t] = lastgae = delta + args.GAE * lastgae * nextnonterminal * args.gamma

        
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

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            
        
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_log_probs, entropy = actor_network.evaluate_get_action(b_obs[mb_inds], b_actions[mb_inds])
                ratio = torch.exp(new_log_probs - b_logprobs[mb_inds])
                logratio = new_log_probs - b_logprobs[mb_inds]
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    wandb.log({"charts/approx_kl": approx_kl.item()})

                b_advantages[mb_inds] = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                
                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                current_values = actor_network.get_value(b_obs[mb_inds]).squeeze()
                
                # Value clipping
                v_loss_unclipped = (current_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    current_values - b_values[mb_inds], -args.clip_coeff, args.clip_coeff
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                critic_loss = args.VALUE_COEFF * 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                loss = policy_loss - args.ENTROPY_COEFF * entropy_loss + critic_loss

                # actor_optim.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                
                grad_norm_dict = {}
                total_norm = 0
                for name, param in actor_network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        if 'actor' in name or 'critic' in name:
                            grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                        else:
                            grad_norm_dict[f"gradients/shared_norm_{name}"] = param_norm.item()
                        total_norm += param_norm.item() ** 2
                grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
                wandb.log(grad_norm_dict)
                
                nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                # nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                # actor_optim.step()
                optimizer.step()
        
        if args.use_wandb:
            wandb.log({ 
                "losses/total_loss": loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": critic_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "charts/learning_rate": optimizer.param_groups[0]['lr'],
                "charts/episodic_return": np.mean(rewards_storage.cpu().numpy()),
                "charts/advantages_mean": b_advantages.mean().item(),
                "charts/advantages_std": b_advantages.std().item(),
                "charts/returns_mean": b_returns.mean().item(),
                "global_step": global_step,
            })
            print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
    

        if update % 50 == 0:
            rewards_player1, rewards_player2, avg_return1, avg_return2, _ = evaluate(actor_network, device, run_name, num_eval_eps=5, record=False)
            # Log the average return from the evaluation
            # avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return_player2": avg_return2,
                    "eval/avg_return_player1": avg_return1,
                    "global_step": global_step,
                })
            print("Rewards from evaluation:", rewards_player1, '   ', rewards_player2)
            print(f"Evaluation at step {global_step}: Average raw return for player 1  = {avg_return1:.2f}, Average raw return for player 2  = {avg_return2:.2f}")

        # Save the model at intervals of 200 updates
        if update % 200 == 0:
            save_checkpoint(actor_network, optimizer, update, global_step, args, run_name)


    if args.capture_video:
        print("Capturing final evaluation video...")
        _, _, _, _, eval_frames = evaluate(actor_network, device, run_name, num_eval_eps=10, record=True)

        if len(eval_frames) > 0:
            video_path = f"final_eval_{run_name}.mp4"
            # os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                print(f"Final evaluation video saved and uploaded to WandB.")

    save_checkpoint(actor_network, optimizer, num_updates, global_step, args, run_name)
    envs.close()
    if args.use_wandb:
        wandb.finish()