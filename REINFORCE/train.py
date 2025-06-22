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
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
from huggingface_hub import HfApi, upload_folder
import cv2

import imageio


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "DQN-CartPole"
    seed = 42
    env_id = "CartPole-v1"
    episodes = 100000  # Number of episodes to train
    # Training parameters
    total_timesteps = 10000
    learning_rate = 2e-4
    buffer_size = 20000 
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 50
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 10
    limit_steps = 200  # Limit steps for initial testing
    # Logging & saving
    capture_video = True
    save_model = True
    upload_model = True
    hf_entity = ""  # Your Hugging Face username
    
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"
    wandb_entity = ""  # Your WandB username/team


class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x =  self.out(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))
        # print('Output shape:', x.shape)
        x = torch.nn.functional.softmax(x, dim=1)  # Apply softmax to get probabilities
        
        # cat = torch.distributions.Categorical(x)
        return x
    
    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  # Create a categorical distribution from the probabilities
        action = dist.sample()  # Sample an action from the distribution
        return action, dist.log_prob(action) 
    

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
            # print(torch.tensor(obs, device=device).unsqueeze(0))
                action, log_probs = model.get_action(torch.tensor(obs, device=device).unsqueeze(0))
                obs, reward, terminated, truncated, _ = eval_env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

          
        returns.append(episode_reward)
        # if eps == 0:  # Save frames only for the first episode (optional)
        #     frames = episode_frames.copy()  # Avoid memory issues

    eval_env.close()
    
    # # Save video
    # if frames:
    #     os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
    #     imageio.mimsave(
    #         f"videos/{run_name}/eval/eval_video.mp4",
    #         frames,
    #         fps=30
    #     )
    model.train()
    return returns, frames

args = Config()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

 # Initialize WandB
if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
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
q_network = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(device)
q_network = q_network.to(device)
# target_net = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
# target_net.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
# eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q_network.train()
# target_net.train()


# replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)



start_time = time.time()



for step in tqdm(range(args.episodes)):
    obs,  _ = env.reset()
    rewards = []
    done = False
    rt = 0.0
    
    log_probs = []
    while not done:

        action, probs = q_network.get_action(torch.tensor(obs, device=device).unsqueeze(0))
        action = action.item()
        new_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        log_probs.append(probs)
        # if done:
        #     break
    # if done:
    #     obs, _ = env.reset()
    # else:   
    #     obs = new_obs
        
    returns = []
    # discounted _return = 0.0
    # Calculate returns
    rt = 0.0
    for reward in reversed(rewards):
        
        rt = reward +  rt * args.gamma
    
        returns.insert(0, rt)
    
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
     
     # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
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
    
    
    #Calculating the loss
    # log_probs = []
    log_probs = torch.stack(log_probs)  # Stack log probabilities
  
    
      # Calculate loss
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)  # Negative for gradient ascent
    
    # Update policy
    optimizer.zero_grad()
    loss = torch.stack(policy_loss, dim=0).mean()  # Use stack instead of cat for 0-dimensional tensors
    loss.backward()
    
    # Log gradient norms for monitoring
    if args.use_wandb and step % 200 == 0:
        grad_norm_dict = {}
        total_norm = 0
        for name, param in q_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2
        grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
        wandb.log(grad_norm_dict)
    
    # torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0) # Or 0.5
    optimizer.step()
    
    
    if step % 200 == 0:
        # Log parameter statistics
        param_dict = {}
        for name, param in q_network.named_parameters():
            param_dict[f"parameters/mean_{name}"] = param.data.mean().item()
            param_dict[f"parameters/std_{name}"] = param.data.std().item()
            param_dict[f"parameters/max_{name}"] = param.data.max().item()
            param_dict[f"parameters/min_{name}"] = param.data.min().item()
        
        # Log loss and other metrics
        wandb.log({
            "losses/policy_loss": loss.item(),
            "step": step,
            **param_dict
        })
        print(f"Step {step}, Loss: {loss.item()}")
        print("Rewards:", sum(rewards))
    
    
    #         # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % 1000 == 0:

        # Evaluate model
        episodic_returns, eval_frames = evaluate(q_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        
        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")

    
# if done:
#     obs, _ = env.reset()
# else:
#     obs = new_obs
        
# env.close()
# writer.close()

# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(q_network, device, run_name, record=True, num_eval_eps=5, render_mode='rgb_array')
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()


