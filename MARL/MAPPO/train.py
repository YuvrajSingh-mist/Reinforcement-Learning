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
from pettingzoo.butterfly import cooperative_pong_v5
import importlib
import supersuit as ss

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-PettingZoo-Pong-MAPPO"
    seed = 42
    env_id = "cooperative_pong_v5"  # Environment ID
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

class Actor(nn.Module):
    def __init__(self, action_space):
        super(Actor, self).__init__()
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
        # self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_features(self, x):
        return self.network(x)

   
    def get_action(self, x, action=None, deterministic=False):

        x = x.clone()
        x = x.permute(0, 3, 1, 2)

        
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
   
        features = self.get_features(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Shared CNN feature extractor
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6 * args.num_agents, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )
        # 
        # Critic head
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_features(self, x):
        return self.network(x)

    def forward(self, x):
        # x = x.clone()
        # x = x.permute(0, 3, 1, 2)
   
        features = self.get_features(x)
        logits = self.critic(features)
        return logits

# --- Environment Creation ---
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    
    env = importlib.import_module(f"pettingzoo.butterfly.{args.env_id}").parallel_env()
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)  # <--- Required to initialize np_random
    env = ss.max_observation_v0(env, 2)
    # env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")
    # print(env.observation_space)
    # print(env.action_space)
    # envs.single_observation_space = single_observation_space
    # envs.single_action_space = single_action_space
    # envs.is_vector_env = True
    # envs = gym.wrappers.RecordEpisodeStatistics(env)
  
    return env

def reshape_obs_shape(obs):
    num_games = args.num_envs // args.num_agents # e.g., 16 // 2 = 8 games
    ma_obs_shape = (num_games, args.num_agents) + single_observation_space.shape # (8, 2, 84, 84, 6)
    ma_obs = obs.reshape(ma_obs_shape)
    return ma_obs


def evaluate(model, device, seed, num_eval_eps=10, record=False):
    eval_env = cooperative_pong_v5.env(render_mode="rgb_array" if record else None)
    env = ss.max_observation_v0(eval_env, 2)
    env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    # eval_env = ss.agent_indicator_v0(env, type_only=False)
    model = model.train()
    # for agt_idx in range(args.num_agents):
        # model[agt_idx].to(device)
        # model[agt_idx].eval()
    
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
    # for agt_idx in range(args.num_agents):
    model.train()

    avg_return1 = np.mean(all_episode_rewards['first_0'])
    avg_return2 = np.mean(all_episode_rewards['second_0'])
    return all_episode_rewards['first_0'], all_episode_rewards['second_0'], avg_return1, avg_return2, frames
    



# --- Checkpoint Saving Function ---
def save_checkpoint(model, optimizer, update, global_step, args, run_name, save_dir="/content/checkpoints/mappo"):
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
    env = make_env(env_id=args.env_id, seed=args.seed, idx=0, run_name=run_name)
    sample_agent = env.possible_agents[0]
    single_action_space = env.action_space(sample_agent)
    single_observation_space = env.observation_space(sample_agent)
    assert isinstance(single_action_space, gym.spaces.Discrete), "Action space is not Discrete"
    print(f"Single agent action space: {single_action_space} (n={single_action_space.n})")
    print(f"Single agent observation space shape: {single_observation_space.shape}")
        # envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    
    # print("Vectorizing environments...")
    # envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")
    critic_network = Critic()
    actor_network = Actor(single_action_space.n) 
    optimizer = optim.Adam(list(critic_network.parameters()) + list(actor_network.parameters()), lr=args.lr, eps=1e-5) 
    
    # critic_optimizer = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

    print("Wrapping PettingZoo env to be Gymnasium VectorEnv compliant...")
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Step D.2: Now that `vec_env` is compliant, stack it.
    # `vec_env` treats each of the 2 agents as a parallel environment.
    # To get `args.num_envs` (e.g., 16) total parallel agent streams, we need to stack
    # `args.num_envs // args.num_agents` copies.
    print(f"Stacking {args.num_envs // args.num_agents} environments to get {args.num_envs} parallel agent streams...")
    envs = ss.concat_vec_envs_v1(vec_env, 
                                num_vec_envs=args.num_envs // args.num_agents, 
                                num_cpus=0, 
                                base_class="gymnasium")

    # E. Now you can reset the environment without error!
    print("Resetting vectorized environments...")
    # This will now correctly return two values.
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    
    print(f"Successfully reset envs. Observation shape: {next_obs.shape}")
    
    # Reshape for MAPPO logic
    num_games = args.num_envs // args.num_agents # e.g., 16 // 2 = 8 games
    
    ma_obs = reshape_obs_shape(next_obs)
    print(f"Reshaped observation to multi-agent format: {ma_obs.shape}")
    
    actor_network = actor_network.to(device)

    critic_network = critic_network.to(device)
    global_state = torch.zeros((args.max_steps, num_games, ma_obs.shape[-1] * args.num_agents, ) + single_observation_space.shape).to(device)
    obs_storage = torch.zeros((args.max_steps, num_games, args.num_agents, ) + single_observation_space.shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)

    # Episode tracking variables
    # episodic_return_reward = np.zeros((args.num_envs, args.num_agents))
    # episode_step_count = np.zeros((args.num_envs, args.num_agents))

    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    # next_obs, _ = envs.reset(seed=args.seed)
    # next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    # next_done_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    bootstrap_value = torch.zeros(args.num_envs).to(device)
    # next_obs_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        
        frac = 1.0 - (update / num_updates)
        lr = args.lr * frac
        
        # for agent_idx, actor in enumerate(actor_network_ls):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #  for agent_idx, actor in enumerate(actor_network_ls):
        # for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
                
        for step in range(0, args.max_steps):
            
            global_step += args.num_envs
            masks = []
            
            for agent_idx in range(args.num_agents):
                
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 255.0 #1d tensor
                masks.append(mask)
            
            # Cant' just use RESHAPE CUS THE PARALLEL ENVS - ONE OF THEM CAN GET OVER EARLY AND PUT THAT DATA ANYWHERE IN THE BACTH
            ma_obs = torch.zeros((num_games, args.num_agents) + single_observation_space.shape).to(device)
            # print(next_obs[:, 0, 0, (4+1)]) # ma_obs = reshape_obs_shape(next_obs)

            # print(ma_obs.shape)
            # print(next_obs.shape)
            # print(masks[0])
            for agent_idx in range(args.num_agents):
                # print(next_obs[masks[agent_idx]])
                agent_obs = next_obs[masks[agent_idx]]
                ma_obs[:, agent_idx , ...] = agent_obs
            # print(ma_obs.shape)
            obs_storage[step] = ma_obs
            
            global_state = ma_obs.permute(0, 1, 4, 2, 3).reshape(num_games, -1, 84, 84)
            
            actions_per_step = torch.zeros(args.num_envs, device=device)
            
            with torch.no_grad():
                    # print("No eval: ", global_state.shape)
                    value = critic_network(global_state)
                                
            for agent_idx in range(args.num_agents):
                
                with torch.no_grad():
                    action, logprob, _ = actor_network.get_action(next_obs[masks[agent_idx]], deterministic=False)
                    # value = critic_network(global_state[step])
                actions_per_step = actions_per_step.long()
                actions_per_step[masks[agent_idx]] = action.long()
              
                values_storage[step][masks[agent_idx]] = value.flatten()
             
                logprobs_storage[step][masks[agent_idx]] = logprob
            
            # print(actions_per_step)
            next_obs, reward, terminated, truncated, info = envs.step(actions_per_step.cpu().numpy())
        
            done = np.logical_or(terminated, truncated)
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            
            next_done = torch.Tensor(done).to(device)
            dones_storage[step] = next_done

        with torch.no_grad():
            
        #Creating masks for GAE
            masks = []
            for agent_idx in range(args.num_agents):
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 255.0
                masks.append(mask)

            advantages = torch.zeros(( num_games, args.max_steps, args.num_agents)).to(device)
            ma_obs = torch.zeros((num_games, args.num_agents) + single_observation_space.shape).to(device)
            
            for agent_idx in range(args.num_agents):
                agent_obs = next_obs[masks[agent_idx]]
                ma_obs[:, agent_idx , ...] = agent_obs
            
            global_state = ma_obs.permute(0, 1, 4, 2, 3).reshape(num_games, -1, 84, 84)
            bootstrap_value = critic_network(global_state).squeeze()  
            # print(values_storage.shape, next_done.shape)
       
            done_storage_gae = dones_storage.reshape((args.max_steps, num_games, args.num_agents)) 
            values_storage_gae = values_storage.reshape((args.max_steps, num_games, args.num_agents))
            rewards_storage_gae = rewards_storage.reshape(args.max_steps, num_games, args.num_agents)
            
            for agent_idx in range(args.num_agents):
               
            # === Advantage Calculation & Returns 
                lastgae = 0
                #This is not right because we n INDEPENDENT PARALLEL ENVS and thus we need to calculate the adv value for each agent separately per game basis
                for t in reversed(range(args.max_steps)):
                    if t == args.max_steps - 1:
                        nextnonterminal = (1.0 - next_done.reshape(num_games, args.num_agents)[:, agent_idx]) #its shared env so we can just take the first agent cus cooperative game
                        # print("Next nonterminal: ", nextnonterminal)
                        gt_next_state = bootstrap_value * nextnonterminal
                    else:
                        nextnonterminal = (1.0 - done_storage_gae[t + 1, :, agent_idx])
                        gt_next_state = values_storage_gae[t + 1, :, agent_idx] * nextnonterminal # If done at t, the next gt is 0

                    delta = (rewards_storage_gae[t, :, agent_idx] +  args.gamma *  gt_next_state ) - values_storage_gae[t, :, agent_idx]
                    advantages[:, t, agent_idx] = lastgae = delta + args.GAE * lastgae * nextnonterminal * args.gamma

        # print(values_storage_gae.shape, advantages.shape, rewards_storage_gae.shape)
        returns = advantages.reshape(args.max_steps, -1) + values_storage_gae.reshape(args.max_steps, -1)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === PPO Update Phase ===
        b_obs = obs_storage.reshape((-1, args.num_agents, ) +  single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1, args.num_agents)
        b_actions = actions_storage.reshape((-1, args.num_agents) + single_action_space.shape)
        b_advantages = advantages.reshape(-1, args.num_agents)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)

        masks = []
        # Create masks for each agent based on the next_obs

        # Total number of game-steps in our batch
        game_batch_size = num_games * args.max_steps
        # Minibatch size in terms of number of game-steps
        game_minibatch_size = game_batch_size // args.num_minibatches

        b_game_inds = np.arange(game_batch_size)

        # Minibatch update
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_game_inds)
            for start in range(0, len(b_game_inds), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_game_inds[start:end]
                # nice one by gemini to do the bird-eyes' view update of the CRITIC NETOWRK
                # in this loop
                # Get minibatch data
                global_state = b_obs[mb_inds].permute(0,1, 4, 2, 3).reshape(mb_inds.shape[0], -1, 84, 84)
                
                # print("mb_obs shape: ", mb_obs.shape)
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]
                current_values = critic_network(global_state).squeeze()
                v_loss_unclipped = (current_values - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(current_values - mb_values, -args.clip_coeff, args.clip_coeff)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                critic_loss = args.VALUE_COEFF * 0.5 * v_loss_max.mean()
                
                policy_loss_total = 0.0
                entropy_loss = 0.0
                
                for agent_idx in range(args.num_agents):
                    mb_obs = b_obs[mb_inds, agent_idx, ...]
                    mb_actions = b_actions[mb_inds, agent_idx]
                    mb_logprobs = b_logprobs[mb_inds, agent_idx]
                    mb_advantages = b_advantages[mb_inds, agent_idx]


                    # Calculate losses
                    new_log_probs, entropy = actor_network.evaluate_get_action(mb_obs, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_logprobs)
                    
                    # Policy loss
                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                    policy_loss_total += policy_loss

                    # Entropy loss
                    entropy_loss += entropy.mean()


                # Average the policy loss across agents CUS ITS A COOPERATIVE GAME AND WE NEED THE AGENTS TO WORK TOGETHER LEARN TOGETHER
                
                policy_loss_total /= args.num_agents
                entropy_loss /= args.num_agents

                # Total loss
                loss = policy_loss_total - args.ENTROPY_COEFF * entropy_loss + critic_loss

                optimizer.zero_grad()
                # critic_optimizer.zero_grad()
                
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
                optimizer.step()


        if args.use_wandb:
            wandb.log({ 
                f"losses/total_loss_agent{agent_idx}": loss.item(),
                f"losses/policy_loss_agent{agent_idx}": policy_loss.item(),
                f"losses/value_loss_agent{agent_idx}": critic_loss.item(),
                f"losses/entropy_agent{agent_idx}": entropy_loss.item(),
                f"charts/learning_rate_agent{agent_idx}": optimizer.param_groups[0]['lr'],
                f"charts/avg_rewards_agent{agent_idx}": np.mean(rewards_storage[:, agent_idx].cpu().numpy()),
                f"charts/advantages_mean_agent{agent_idx}": b_advantages.mean().item(),
                f"charts/advantages_std_agent{agent_idx}": b_advantages.std().item(),
                f"charts/returns_mean_agent{agent_idx}": b_returns.mean().item(),
                "global_step": global_step,
            })
            # print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
        
        # critic_optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(critic_network.parameters(), 0.5)
        # critic_optimizer.step()
        
        if update % 50 == 0:
            rewards_player1, rewards_player2, avg_return1, avg_return2, _ = evaluate(actor_network, device, run_name, num_eval_eps=5, record=False)
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
            # for idx, (actor, optim) in enumerate(zip(actor_network, optimizer)):
            save_checkpoint(actor_network, optimizer, num_updates, global_step, args, 'actor_network')

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


    save_checkpoint(actor_network, optimizer, num_updates, global_step, args, 'actor_network')
    envs.close()
    if args.use_wandb:
        wandb.finish()