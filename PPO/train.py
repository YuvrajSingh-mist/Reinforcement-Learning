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
    exp_name = "PPO-LunarLander-v2"
    seed = 42
    env_id = "LunarLander-v3"
    episodes = 10000
   
    # actor_learning_rate = 3e-4
    # critic_lr = 1e-3
    lr = 3e-4
    gamma = 0.99
  
    capture_video = True
    # save_model = True
    # upload_model = True
    # clip_norm = 1.0
    clip_value = 0.2
    PPO_EPOCHS = 4
    ENTROPY_COEFF = 0.01
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"
    max_steps = 512
    # final_lr = 5e-5
    # decay_iters = 0.9 * episodes
    
VALUE_COEFF = 0.5

class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, action_space)

    def forward(self, x):
        # Use sequential processing for better numerical stability
        x = self.fc1(x)
            
        x = torch.nn.functional.relu(x)  # Use Mish activation
        
        x = self.fc2(x)
      
        x = torch.nn.functional.relu(x)  # Use Mish activation
        
        x = self.fc3(x)
        
        x = torch.nn.functional.relu(x)  # Use Mish activation 889438865
        
        x = self.out(x)  # Output layer
       
        # Apply softmax with numerical stability
        x = torch.nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        
        return x
    
    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        action = dist.sample() 
        return action, dist.log_prob(action) 
    
    def get_action_and_log_probs(self, x, action):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, entropy

class CriticNet(nn.Module):
    
    def __init__(self, state_space, action_space):
        super(CriticNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.value(x)
    
    
    
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
        # episode_frames = []

        while not done:

            if(record):
                if (episode_reward > 500):
                    print("Hooray! Episode reward exceeded 500, stopping early.")
                    break
                frame = eval_env.render()
                frames.append(frame)  # Capture all frames

            with torch.no_grad():
          
                action, log_probs = model.get_action(torch.tensor(obs, device=device).unsqueeze(0))
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
            # entity=args.wandb_entity,
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
device = "cuda"



env = make_env(args.env_id, args.seed, args.capture_video, run_name)
actor_network = ActorNet(env.observation_space.shape[0], env.action_space.n).to(device)
critic_network = CriticNet(env.observation_space.shape[0], 1).to(device)

actor_optim = optim.Adam(actor_network.parameters(), lr=args.lr)
critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr)

actor_network.train()
critic_network.train()

start_time = time.time()

obs, _ = env.reset()

for step in tqdm(range(args.episodes)):
    

    # obs,  _ = env.reset()
    rewards = []
    done = False
    rt = 0.0
    new_log_probs = []  # Initialize new log probabilities
    old_log_probs = []
    values = []
    states = []
    actions = []
    dones = []
    # states.append(obs)
    for _ in range(args.max_steps):
   
  
        states.append(obs)
        action, probs = actor_network.get_action(torch.tensor(obs, device=device).unsqueeze(0))
        action = action.item()
        new_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        value = critic_network(torch.tensor(obs, device=device).unsqueeze(0))
        values.append(value)
        done = terminated or truncated
        dones.append(done)
        old_log_probs.append(probs)
        obs = new_obs
        actions.append(action)
        if done:
            obs, _ = env.reset()
        #     dones.append(True)
        # else:
        #     dones.append(False)
        # states.append(obs)
        
    returns = []
    curr_observation = obs # The state after the last step
    
    # Use this logic after the fixed-size rollout
    returns = []
    if dones[-1]: # If the rollout ended on a terminal state
        bootstrap_scalar = 0.0
    else:
        with torch.no_grad():
            bootstrap_scalar = critic_network(torch.tensor(curr_observation, device=device).unsqueeze(0)).squeeze().item()

    gt_next_state = bootstrap_scalar
    for reward_at_t, done_at_t in zip(reversed(rewards), reversed(dones)):
        if done_at_t:
            gt_next_state = 0.0
        rt = reward_at_t + args.gamma * gt_next_state
        returns.insert(0, rt)
        gt_next_state = rt


    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    values = torch.stack(values).squeeze()
    
    # Calculate advantages and detach immediately to avoid gradient issues
    advantages = (returns - values).detach()  # Calculate advantages and detach
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
    returns = returns.detach()  # Detach returns

    
    entropyls = []
    
    #Calculating the loss
    old_log_probs = torch.stack(old_log_probs).squeeze().detach()  # Stack and detach log probabilities
    states_tensor = torch.tensor(np.array(states), device=device, dtype=torch.float32).detach()
    actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).detach()
    
    # print(f"Advantages shape: {advantages.shape}, Old log probs shape: {old_log_probs.shape}, Returns shape: {returns.shape}")

    # PPO Update epochs
    for epoch in range(args.PPO_EPOCHS):
        # Get new action probabilities for all states at once
        new_log_probs, entropy = actor_network.get_action_and_log_probs(states_tensor, actions_tensor)
        # entropy_mean = entropy.mean()  # Keep as tensor for backpropagation
        ratio = torch.exp(new_log_probs - old_log_probs)  # Calculate the ratio of new to old probabilities
        
        # Calculate surrogate loss
        # print(f"Ratio shape: {ratio.shape}, Advantages shape: {advantages.shape}")
        surrogate_loss = -(torch.min(
            ratio * advantages,  # For the current policy
            torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value) * advantages  # Clipped version
        ).mean())  # Subtract entropy to encourage exploration
        
        # VALUE LOSS - Get fresh values for current policy
        current_values = critic_network(states_tensor).squeeze()  # Get current values for all states at once
        # print(f"Current values shape: {current_values.shape}, Returns shape: {returns.shape}")
        critic_loss = VALUE_COEFF * torch.nn.functional.mse_loss(current_values, returns)
        
        loss = surrogate_loss + critic_loss
            
        actor_optim.zero_grad()
        critic_optim.zero_grad()
        loss.backward()
        actor_optim.step()
        critic_optim.step()

    

    if step % 200 == 0:
    # Log parameter statistics
        param_dict = {}
        for name, param in actor_network.named_parameters():
            param_dict[f"parameters/mean_{name}"] = param.data.mean().item()
            param_dict[f"parameters/std_{name}"] = param.data.std().item()
            param_dict[f"parameters/max_{name}"] = param.data.max().item()
            param_dict[f"parameters/min_{name}"] = param.data.min().item()
        
        # Log loss and other metrics
        wandb.log({
            "losses/critic_loss": critic_loss.item(),
            "losses/policy_loss": surrogate_loss.item(),
            "step": step,
            **param_dict
        })
        print(f"Step {step}, Actor Loss: {surrogate_loss.item()}")
        print("Critic loss: ", critic_loss.item())
        print("Rewards:", sum(rewards))
    
    
    #         # ===== MODEL EVALUATION & SAVING =====
    if step % 100 == 0:

        # Evaluate model
        episodic_returns, eval_frames = evaluate(actor_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        
        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")



# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/final_{args.env_id}.mp4"
    returns, frames = evaluate(actor_network, device, run_name, record=True, num_eval_eps=5, render_mode='rgb_array')
  
    imageio.mimsave(train_video_path, frames, fps=30, codec='')
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()

