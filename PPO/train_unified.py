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
torch.autograd.set_detect_anomaly(True)

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-Unified-CartPole"
    seed = 42
    env_id = "CartPole-v1"
    episodes = 10000 
   
    learning_rate = 5e-4
    
    gamma = 0.99
  
    capture_video = True
    save_model = True
    upload_model = True
    clip_norm = 1.0
    clip_value = 0.2
    PPO_EPOCHS = 10
    ENTROPY_COEFF = 0.01
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"

VALUE_COEFF = 1.0

class ActorCriticNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorCriticNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_space, 64)
        self.shared_fc2 = nn.Linear(64, 64)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(64, 32)
        self.actor_out = nn.Linear(32, action_space)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(64, 32)
        self.critic_out = nn.Linear(32, 1)

    def forward(self, x):
        # Shared feature extraction
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))
        
        # Actor output (policy)
        actor_x = torch.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor_x)
        action_probs = torch.nn.functional.softmax(action_logits, dim=1)
        
        # Critic output (value)
        critic_x = torch.relu(self.critic_fc(x))
        value = self.critic_out(critic_x)
        
        return action_probs, value
    
    def get_action(self, x):
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        action = dist.sample() 
        return action, dist.log_prob(action), value
    
    def get_action_and_log_probs(self, x, action):
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, entropy, value

    def get_value(self, x):
        _, value = self.forward(x)
        return value
    
def make_env(env_id, seed, capture_video, run_name, eval_mode=False, render_mode=None):
    """Create environment with video recording"""
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    env.action_space.seed(seed)

    return env


def evaluate(model, device, run_name, num_eval_eps = 10, record = False, render_mode=None):
    
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, capture_video=True, render_mode=render_mode, run_name=run_name, eval_mode=True)
    eval_env.action_space.seed(Config.seed)
    
    model = model.to(device)
    model = model.eval()
    returns = []
    frames = []

    for eps in tqdm(range(num_eval_eps)):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:

            if(record):
                if (episode_reward > 500):
                    print("Hooray! Episode reward exceeded 500, stopping early.")
                    break
                frame = eval_env.render()
                frames.append(frame)  # Capture all frames

            with torch.no_grad():
                action, log_probs, _ = model.get_action(torch.tensor(obs, device=device).unsqueeze(0))
                obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

        returns.append(episode_reward)
      
    eval_env.close()
    model.train()
    return returns, frames

args = Config()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

 # Initialize WandB
if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
os.makedirs(f"videos/{run_name}/train", exist_ok=True)
os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
os.makedirs(f"runs/{run_name}", exist_ok=True)
writer = SummaryWriter(f"runs/{run_name}")
    
# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env(args.env_id, args.seed, args.capture_video, run_name)
# Single unified network for both actor and critic
unified_network = ActorCriticNet(env.observation_space.shape[0], env.action_space.n).to(device)

# Single optimizer for the unified network
optimizer = optim.Adam(unified_network.parameters(), lr=args.learning_rate)

unified_network.train()

start_time = time.time()

for step in tqdm(range(args.episodes)):
    obs,  _ = env.reset()
    rewards = []
    done = False
    rt = 0.0
    new_log_probs = []  # Initialize new log probabilities
    old_log_probs = []
    values = []
    states = []
    actions = []
    
    while not done:
        states.append(obs)
        action, probs, value = unified_network.get_action(torch.tensor(obs, device=device).unsqueeze(0))
        action = action.item()
        new_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        values.append(value)
        done = terminated or truncated
        old_log_probs.append(probs)
        obs = new_obs
        actions.append(action)
        
    returns = []
    rt = 0.0
    for reward in reversed(rewards):
        rt = reward +  rt * args.gamma
        returns.insert(0, rt)
    
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    values = torch.stack(values).squeeze()
    
    # Calculate advantages and detach immediately to avoid gradient issues
    advantages = (returns - values).detach()  # Calculate advantages and detach
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
    returns = returns.detach()  # Detach returns
    
    # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
    
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                "global_step": step,
                "calculated_return": returns.mean().item()
            })
    
    #Calculating the loss
    old_log_probs = torch.stack(old_log_probs).detach()  # Stack and detach log probabilities
    states_tensor = torch.tensor(states, device=device, dtype=torch.float32).detach()
    actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).detach()

    # PPO Update epochs
    for epoch in range(args.PPO_EPOCHS):
        # Get new action probabilities and values for all states at once
        new_log_probs, entropy, current_values = unified_network.get_action_and_log_probs(states_tensor, actions_tensor)
        entropy_mean = entropy.mean()  # Keep as tensor for backpropagation
        ratio = torch.exp(new_log_probs - old_log_probs)  # Calculate the ratio of new to old probabilities
        
        # Calculate surrogate loss (actor loss)
        surrogate_loss = -(torch.min(
            ratio * advantages,  # For the current policy
            torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value) * advantages  # Clipped version
        ).mean() + args.ENTROPY_COEFF * entropy_mean)  # Subtract entropy to encourage exploration
        
        # VALUE LOSS (critic loss)
        current_values = current_values.squeeze()
        critic_loss = VALUE_COEFF * torch.nn.functional.mse_loss(current_values, returns)
        
        # Combined loss
        total_loss = surrogate_loss - critic_loss
        
        # Update unified network
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(unified_network.parameters(), args.clip_norm)
        optimizer.step()
    
    # Log gradient norms for monitoring
    if args.use_wandb and step % 200 == 0:
        grad_norm_dict = {}
        total_norm = 0
        for name, param in unified_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2
        grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
        wandb.log(grad_norm_dict)

    if step % 200 == 0:
        # Log parameter statistics
        param_dict = {}
        for name, param in unified_network.named_parameters():
            param_dict[f"parameters/mean_{name}"] = param.data.mean().item()
            param_dict[f"parameters/std_{name}"] = param.data.std().item()
            param_dict[f"parameters/max_{name}"] = param.data.max().item()
            param_dict[f"parameters/min_{name}"] = param.data.min().item()
        
        # Log loss and other metrics
        wandb.log({
            "losses/critic_loss": critic_loss.item(),
            "losses/policy_loss": surrogate_loss.item(),
            "losses/total_loss": total_loss.item(),
            "step": step,
            **param_dict
        })
        print(f"Step {step}, Total Loss: {total_loss.item()}")
        print("Actor loss: ", surrogate_loss.item())
        print("Critic loss: ", critic_loss.item())
        print("Rewards:", sum(rewards))
    
    # Model evaluation & saving
    if args.save_model and step % 200 == 0:
        # Evaluate model
        episodic_returns, eval_frames = evaluate(unified_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        if args.use_wandb:
            wandb.log({
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")

# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(unified_network, device, run_name, record=True, num_eval_eps=5, render_mode='rgb_array')
  
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()
