

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
import ale_py

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

gym.register_envs(ale_py)
# from vizdoom import gymnasium_wrapper # Ensure ViZDoom is registered

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-Vectorized-Atari"
    seed = 42
    env_id = "BreakoutNoFrameskip-v4"
    total_timesteps = 10_000_000  # Standard metric for vectorized training

    # PPO & Agent settings
    lr = 2.5e-4
    gamma = 0.99
    num_envs = 8  # Number of parallel environments
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
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
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
        return self.critic(self.get_features(x))

    def get_action(self, x, action=None, deterministic=False):
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
        features = self.get_features(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

# --- Environment Creation ---
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    def thunk():
        render_mode = "rgb_array" if eval_mode else None
        # Force RGB24 format for ViZDoom to avoid CRCGCB warning
        env = gym.make(env_id, render_mode=render_mode)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.AtariPreprocessing(env,
        #     frame_skip=4,  # Standard frame skip for Atari
        #     grayscale_obs=True,  # Add channel dimension for grayscale
        #     scale_obs=True,  # Scale observations to [0, 1]
        #     screen_size=(TARGET_HEIGHT, TARGET_WIDTH),  # Resize to target dimensions
        # )
        # # Use our custom wrapper for all preprocessing
        # # env = PreprocessAndFrameStack(env, height=TARGET_HEIGHT, width=TARGET_WIDTH, num_stack=4)
        # env = gym.wrappers.FrameStackObservation(env, 4)
        
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)  
        # # print(env.unwrapped.get_action_meanings())
       
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayscaleObservation(env)
        # env = gym.wrappers.FrameStackObservation(env, 4)
         # Use the all-in-one, official Atari wrapper
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84, # It assumes square images
            terminal_on_life_loss=True, # Standard for training
            grayscale_obs=True,
            scale_obs=True # We want uint8 [0, 255] for storage
        )
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # Now, stack the preprocessed frames
        # env = ClipRewardEnv(env)  # Clip rewards to [-1, 1]
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

# --- Evaluation ---
def evaluate(agent_model, device, run_name, num_eval_eps=10, record=False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, idx=0, run_name=run_name, eval_mode=True)()
    
    agent_model.to(device)
    agent_model.eval()
    returns = []
    frames = []

    for eps in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if record:
                # Get the raw frame from the original env for nice videos
                frame = eval_env.unwrapped.render()
                frames.append(frame)

            with torch.no_grad():
                # Add batch dimension and convert to tensor
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                action, _, _ = agent_model.get_action(obs_tensor, deterministic=True)
                # Convert action to scalar integer for ViZDoom
                action_scalar = action.cpu().numpy().item()
                obs, reward, terminated, truncated, info = eval_env.step(action_scalar)
                done = terminated or truncated
                episode_reward += float(reward)
                # Use raw reward from info if available
                # if "episode" in info:
                #     episode_reward = info["episode"]["r"]
          
        returns.append(episode_reward)
      
    eval_env.close()
    agent_model.train()
    return returns, frames



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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor_network = Agent(envs.single_action_space.n).to(device)
    optimizer = optim.Adam(actor_network.parameters(), lr=args.lr, eps=1e-5)
    # critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

    obs_storage = torch.zeros((args.max_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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

            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_obs)
                value = actor_network.get_value(next_obs)
            
            values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Update episode tracking
            episodic_return += reward
            episode_step_count += 1
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            if 'final_observation' in info.keys():
                for i_env, done_flag in enumerate(info['_final_observation']):
                    if done_flag:
                        print(f"\rglobal_step={global_step}, episodic_return={episodic_return[i_env]}", end='')

                        wandb.log({"charts/episodic_return": episodic_return[i_env], "global_step": global_step})
                        wandb.log({"charts/episodic_length": episode_step_count[i_env], "global_step": global_step})

                        episodic_return[i_env], episode_step_count[i_env] = 0., 0.

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
        b_obs = obs_storage.reshape((-1,) +  envs.single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + envs.single_action_space.shape)
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

                b_advantages_block = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)

                pg_loss1 = b_advantages_block * ratio
                pg_loss2 = b_advantages_block * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
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
            episodic_returns, _ = evaluate(actor_network, device, run_name, num_eval_eps=5, record=args.capture_video)
            # Log the average return from the evaluation
            avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return": avg_return,
                    "global_step": global_step,
                })
            print(f"Evaluation at step {global_step}: Average raw return = {avg_return:.2f}")

    if args.capture_video:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(actor_network, device, run_name, num_eval_eps=10, record=True)

        if len(eval_frames) > 0:
            video_path = f"videos/final_eval_{run_name}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                print(f"Final evaluation video saved and uploaded to WandB.")

    envs.close()
    if args.use_wandb:
        wandb.finish()