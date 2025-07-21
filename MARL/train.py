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
from pettingzoo.atari import pong_v3
import cv2

import imageio


def preprocess_observation(obs):
    """Convert observation from (H, W, C) to (C, H, W) format for PyTorch"""
    if isinstance(obs, np.ndarray):
        if len(obs.shape) == 3:  # (H, W, C)
            return np.transpose(obs, (2, 0, 1))  # Convert to (C, H, W)
        else:
            return obs
    return obs


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "A2C-PettingzooPong"
    seed = 42
    env_id = "Pong-v5"  # Use Atari Pong environment
    episodes = 2000

    lr = 1e-3
    final_lr = 5e-5
    
    decay_steps = 2000
    # actor_learning_rate = 3e-4
    # critic_learning_rate = 1e-3
    gamma = 0.99
  
    capture_video = True
    save_model = True
    upload_model = True
    
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"


class ActorNet(nn.Module):
    
    def __init__(self, state_space, action_space):
        super(ActorNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_value = nn.Linear(512, action_space)
        
        
    def forward(self, x):
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        res = torch.nn.functional.softmax(self.q_value(x), dim=-1)  # Apply softmax to get action probabilities
        # print("Action shape: ", res.shape)
        return res
    
    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        action = dist.sample() 
        return action, dist.log_prob(action) 
    

class CriticNet(nn.Module):
    
    def __init__(self, state_space, action_space):
        super(CriticNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        
        # Use the same architecture as ActorNet for image processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.value = nn.Linear(512, action_space)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)
    




# --- Configuration for Preprocessing ---
TARGET_HEIGHT = 64
TARGET_WIDTH = 64
DEVICE = "cuda:9"

# --- Albumentations Preprocessing Function ---
def preprocess_frame_albumentations(frame_numpy, target_height, target_width, device):
    """
    Preprocesses a game frame (NumPy array) using Albumentations.
    Handles both single frames and frame stacks (e.g., from frame skipping).
    Input: NumPy array, typically (H, W, C), (C, H, W), (H,W) for single frame
           or (N, H, W, C) for frame stack where N is the number of frames.
    Output: PyTorch tensor of shape (1, target_height, target_width) for single frame
            or (N, 1, target_height, target_width) for frame stack, on the specified device.
    """
    if frame_numpy is None:
        # Return a black image of the target size if input is None
        return torch.zeros((1, target_height, target_width), dtype=torch.float32, device=device)

    # 1. Ensure input is a NumPy array
    if not isinstance(frame_numpy, np.ndarray):
        frame_numpy = np.array(frame_numpy)

    # Handle frame stacking case: (N, H, W, C)
    if frame_numpy.ndim == 4:
        # Process each frame in the stack
        processed_frames = []
        for i in range(frame_numpy.shape[0]):
            single_frame = frame_numpy[i]  # Shape: (H, W, C)
            processed_frame = preprocess_single_frame_albumentations(single_frame, target_height, target_width, device)
            processed_frames.append(processed_frame.squeeze(0))  # Remove channel dim: (H, W)
        # Stack the processed frames: (N, H, W)
        return torch.stack(processed_frames)
    else:
        # Single frame case
        return preprocess_single_frame_albumentations(frame_numpy, target_height, target_width, device)

def preprocess_single_frame_albumentations(frame_numpy, target_height, target_width, device):
    """
    Preprocesses a single game frame using PyTorch operations instead of Albumentations.
    """
    # Convert to tensor
    if not isinstance(frame_numpy, np.ndarray):
        frame_numpy = np.array(frame_numpy)
    
    # Ensure the array has the right dtype
    if frame_numpy.dtype == np.object_:
        # Convert object array to float array
        frame_numpy = np.array(frame_numpy, dtype=np.float32)
    elif frame_numpy.dtype != np.float32:
        frame_numpy = frame_numpy.astype(np.float32)
    
    frame_tensor = torch.from_numpy(frame_numpy)
    
    # Handle different input shapes
    if frame_tensor.dim() == 2:  # (H, W) - already grayscale
        frame_tensor = frame_tensor.unsqueeze(0)  # (1, H, W)
    elif frame_tensor.dim() == 3:
        if frame_tensor.shape[0] in [1, 3]:  # (C, H, W)
            # Already in channel-first format
            pass
        else:  # (H, W, C)
            frame_tensor = frame_tensor.permute(2, 0, 1)  # (C, H, W)
    
    # Convert to grayscale if RGB
    if frame_tensor.shape[0] == 3:  # RGB
        # Standard RGB to grayscale conversion
        grayscale = 0.299 * frame_tensor[0] + 0.587 * frame_tensor[1] + 0.114 * frame_tensor[2]
        frame_tensor = grayscale.unsqueeze(0)  # (1, H, W)
    
    # Add batch dimension for interpolation
    frame_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # Resize using PyTorch interpolation
    frame_tensor = torch.nn.functional.interpolate(
        frame_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Normalize to [0, 1] and remove batch dimension
    frame_tensor = frame_tensor.squeeze(0) / 255.0  # (1, H, W)
    
    return frame_tensor.to(device)

    
def make_env(env_id, seed, capture_video, run_name, eval_mode=False, render_mode=None):
    """Create environment with video recording"""
    if render_mode is None:
        render_mode = "rgb_array"  # Default to rgb_array
        
    env = pong_v3.env(render_mode=render_mode, num_players=2, max_cycles=2000)
    print("Environment created. Agents:", env.possible_agents)
    print("Observation spaces:", env.observation_spaces)
    print("Action spaces:", env.action_spaces)
    
    # Set seed for reproducibility
    env.reset(seed=seed)
    
    return env


def evaluate(model, device, run_name, num_eval_eps = 10, record = False, render_mode=None):
    
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, capture_video=True, render_mode=render_mode, run_name=run_name, eval_mode=True)
    # PettingZoo environments don't need action_space.seed - seeding is done in make_env
    
    model = model.to(device)
    model = model.eval()
    returns = []
    frames = []

    for eps in tqdm(range(num_eval_eps)):
        eval_env.reset()
        # obs = preprocess_observation(obs)  # Convert to (C, H, W) format
        done = False
        episode_reward = 0.0
        # episode_frames = []

        # while not done:
        for agent in eval_env.agent_iter():
            obs, reward, terminated, truncated, _ = eval_env.last()
            done = terminated or truncated
            if done:
                eval_env.step(None)
                continue
            
            if(record):
                # if (episode_reward > 500):
                #     print("Hooray! Episode reward exceeded 500, stopping early.")
                #     break
                frame = eval_env.render()
                frames.append(frame)  # Capture all frames

            if agent == 'first_0':
                obs = preprocess_frame_albumentations(obs, TARGET_HEIGHT, TARGET_WIDTH, device)
                obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, probs = actor_network.get_action(obs)
                action = action.item()
                eval_env.step(action)  # Use eval_env here instead of env
                # new_obs, reward, terminated, truncated, info = env.step(action)
                # obs = new_obs
                
            
            else:
                eval_env.step(eval_env.action_space(agent).sample())  # Sample random action for other agents
                episode_reward += float(reward)  # Convert reward to float
                

          
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
device = DEVICE



env = make_env(args.env_id, args.seed, args.capture_video, run_name)
print(f"Observation space: {env.observation_spaces}")
print(f"Action space: {env.action_spaces}")

# Get observation shape for the first agent
first_agent = env.possible_agents[0]
if first_agent in env.observation_spaces:
    obs_space = env.observation_spaces[first_agent]
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        obs_shape = obs_space.shape
        obs_channels = obs_shape[2] if len(obs_shape) == 3 else 1
    else:
        obs_channels = 1  # Default if shape is unknown
else:
    obs_channels = 1  # Fallback

# Get action space size for the first agent
if first_agent in env.action_spaces:
    action_space = env.action_spaces[first_agent]
    if hasattr(action_space, 'n'):
        action_space_n = getattr(action_space, 'n')
    else:
        action_space_n = 6  # Default for Pong
else:
    action_space_n = 6  # Fallback

actor_network = ActorNet(1, action_space_n).to(device)
critic_network = CriticNet(1, 1).to(device)

actor_optim = optim.Adam(actor_network.parameters(), lr=args.lr)
critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr)

actor_network.train()
critic_network.train()

start_time = time.time()

# Add this function right before the main training loop
def save_models(actor_model, critic_model, run_name, step=None, is_best=False):
    """
    Save both actor and critic models
    
    Args:
        actor_model: The actor network to save
        critic_model: The critic network to save
        run_name: The name of the current run (for directory structure)
        step: Current training step (None for final model)
        is_best: Whether this is the best model so far
    """
    save_dir = f"runs/{run_name}/models"
    os.makedirs(save_dir, exist_ok=True)
    
    if step is not None:
        actor_path = f"{save_dir}/actor_step_{step}.pt"
        critic_path = f"{save_dir}/critic_step_{step}.pt"
    else:
        actor_path = f"{save_dir}/actor_final.pt"
        critic_path = f"{save_dir}/critic_final.pt"
    
    if is_best:
        actor_path = f"{save_dir}/actor_best.pt"
        critic_path = f"{save_dir}/critic_best.pt"
    
    torch.save(actor_model.state_dict(), actor_path)
    torch.save(critic_model.state_dict(), critic_path)
    
    print(f"Models saved to {save_dir}")

for step in tqdm(range(args.episodes)):
    env.reset()
    # obs = preprocess_observation(obs)  # Convert to (C, H, W) format
    # obs, reward, terminated, truncated, info = env.last()
    # done = terminated or truncated
    rewards = []
    done = False
    rt = 0.0
    
    log_probs = []
    values = []
    
     # --- Learning Rate Annealing ---
    if step < args.decay_steps:
        # Calculate the fraction of decay completed
        fraction = step / args.decay_steps
        
        # Linearly interpolate Actor LR
        current_actor_lr = args.lr - fraction * (args.lr - args.final_lr)
        for param_group in actor_optim.param_groups:
            param_group['lr'] = current_actor_lr
        
        for param_group in critic_optim.param_groups:
            
            param_group['lr'] = current_actor_lr

       
    else:
        # After decay period, keep LR at final_learning_rate
        for param_group in actor_optim.param_groups:
            param_group['lr'] = args.final_lr
        for param_group in critic_optim.param_groups:
            param_group['lr'] = args.final_lr
            
    # while not done:
    for agent in env.agent_iter():
        
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        if done:
            env.step(None)
            continue
        
        elif agent == 'first_0':
            obs = preprocess_frame_albumentations(obs, TARGET_HEIGHT, TARGET_WIDTH, device)
            obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            # print("Obs shape: ", obs.shape
            action, probs = actor_network.get_action(obs)
            action = action.item()
            env.step(action)  # This is the line that's causing the error
            # new_obs = preprocess_observation(new_obs)  # Convert to (C, H, W) format
            
            value = critic_network(obs)
            values.append(value)
           
            log_probs.append(probs)
        else:
            # Use a random action for the second agent
            env.step(env.action_space(agent).sample())
            rewards.append(float(reward))  # Convert reward to float
        # Log episode returns
        if "episode" in info:
                print(f"Step={step}, Return={info['episode']['r']}")
            
            
                # WandB logging
                if args.use_wandb:
                    wandb.log({
                        "episodic_return": info['episode']['r'],
                        "episodic_length": info['episode']['l'],
                        # "epsilon": eps_decay(step, args.exploration_fraction),
                        "global_step": step,
                        "calculated_return": returns.mean().item()
                    })
    
    assert len(rewards) == len(values) == len(log_probs), "Rewards, values, and log_probs must have the same length"
    # print(len(rewards), len(values), len(log_probs))

    min_len = min(len(rewards), len(values), len(log_probs))
    rewards = rewards[:min_len]
    values = values[:min_len]
    log_probs = log_probs[:min_len]
    # print(rewards)
    
    wandb.log({
            "rewards/mean_reward": np.mean(rewards),
            "rewards/total_reward": sum(rewards),
            # "step": step
            'lr/actor_lr': actor_optim.param_groups[0]['lr'],
            'lr/critic_lr': critic_optim.param_groups[0]['lr'],
        })
        
        
        # obs = new_obs
     
    returns = []

    
    # print(f"Rewards: {rewards}")
    rt = 0.0
    for reward in reversed(rewards):
        
        rt = reward +  rt * args.gamma
    
        returns.insert(0, rt)
    
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    values = torch.stack(values)

    advantages = (returns - values.squeeze()).detach()  # Calculate advantages


    #Calculating the loss
   
    log_probs = torch.stack(log_probs)  # Stack log probabilities
  
 
    # Calculate loss
    policy_loss = []
    for log_prob, advantage in zip(log_probs, advantages):
        policy_loss.append(-log_prob * advantage)  # Negative for gradient ascent

    #Actor loss is the negative log probability of the action taken, weighted by the advantage
    policy_loss = torch.stack(policy_loss, dim=0)
    policy_loss = policy_loss.mean()  # Mean over the batch
    actor_loss = policy_loss
    actor_optim.zero_grad()
    
   
    actor_loss.backward()
    # Clip gradients to prevent exploding gradients
    if args.use_wandb and step % 20 == 0 and step != 0:
        grad_norm_dict = {}
        total_norm = 0
        for name, param in actor_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2
        grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
        wandb.log(grad_norm_dict)
        
    torch.nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=1.0)
    actor_optim.step()
    
    #VALUE LOSS
    # Critic loss is the mean squared error between the predicted values and the returns
    critic_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)

    critic_network.zero_grad()
    critic_loss.backward()
      
    grad_norm_dict = {}
    total_norm = 0
    for name, param in critic_network.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
            total_norm += param_norm.item() ** 2
    grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
    wandb.log(grad_norm_dict)
        
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=1.0)
    critic_optim.step()
    
    
    # Log gradient norms for monitoring
   
      
        

    

    if step % 50 == 0:
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
            "losses/policy_loss": actor_loss.item(),
            "step": step,
            **param_dict
        })
        print(f"Step {step}, Actor Loss: {actor_loss.item()}")
        print("Critic loss: ", critic_loss.item())
        print("Rewards:", sum(rewards))
    
    
    #         # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % 50 == 0:
        # Evaluate model
        episodic_returns, eval_frames = evaluate(actor_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        # Save model after evaluation
        save_models(actor_network, critic_network, run_name, step=step)
        
        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")


# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(actor_network, device, run_name, record=True, num_eval_eps=5, render_mode='rgb_array')
    
    # Save final model
    save_models(actor_network, critic_network, run_name)
    
    imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()



