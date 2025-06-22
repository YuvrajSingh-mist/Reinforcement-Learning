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


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "A2C-CartPole"
    seed = 42
    env_id = "CartPole-v1"
    episodes = 2000 
   
    learning_rate = 2e-3

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
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x =  self.out(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))
      
        x = torch.nn.functional.softmax(x, dim=1)  # Apply softmax to get probabilities
        

        return x
    
    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  
        action = dist.sample() 
        return action, dist.log_prob(action) 
    

class CriticNet(nn.Module):
    
    def __init__(self, state_space, action_space):
        super(CriticNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 16)
        self.q_value = nn.Linear(16, action_space)
        
    def forward(self, x):
        return self.q_value(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    
    
    
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
actor_network = ActorNet(env.observation_space.shape[0], env.action_space.n).to(device)
critic_network = CriticNet(env.observation_space.shape[0], 1).to(device)

actor_optim = optim.Adam(actor_network.parameters(), lr=args.learning_rate)
critic_optim = optim.Adam(critic_network.parameters(), lr=args.learning_rate)

actor_network.train()


start_time = time.time()



for step in tqdm(range(args.episodes)):
    obs,  _ = env.reset()
    rewards = []
    done = False
    rt = 0.0
    
    log_probs = []
    values = []
    while not done:

        action, probs = actor_network.get_action(torch.tensor(obs, device=device).unsqueeze(0))
        action = action.item()
        new_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        value = critic_network(torch.tensor(obs, device=device).unsqueeze(0))
        values.append(value)
        done = terminated or truncated
        log_probs.append(probs)
        obs = new_obs
     
    returns = []
 
    rt = 0.0
    for reward in reversed(rewards):
        
        rt = reward +  rt * args.gamma
    
        returns.insert(0, rt)
    
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    values = torch.stack(values)
    
    advantages = returns - values.squeeze()  # Calculate advantages
   
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
   
    log_probs = torch.stack(log_probs)  # Stack log probabilities
  
 
      # Calculate loss
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)  # Negative for gradient ascent
    
   
    #Actor loss is the negative log probability of the action taken, weighted by the advantage
    policy_loss = torch.stack(policy_loss, dim=0) * advantages  # Stack to create a tensor
    policy_loss = policy_loss.mean()  # Mean over the batch
    actor_loss = policy_loss
    actor_optim.zero_grad()
    
   
    actor_loss.backward(retain_graph=True)
    actor_optim.step()
    
    #VALUE LOSS
    # Critic loss is the mean squared error between the predicted values and the returns
    critic_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)

    critic_network.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    
    
    # Log gradient norms for monitoring
    if args.use_wandb and step % 200 == 0:
        grad_norm_dict = {}
        total_norm = 0
        for name, param in actor_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm_dict[f"gradients/norm_{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2
        grad_norm_dict["gradients/total_norm"] = total_norm ** 0.5
        wandb.log(grad_norm_dict)

    

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
            "losses/policy_loss": actor_loss.item(),
            "step": step,
            **param_dict
        })
        print(f"Step {step}, Actor Loss: {actor_loss.item()}")
        print("Critic loss: ", critic_loss.item())
        print("Rewards:", sum(rewards))
    
    
    #         # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % 1000 == 0:

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
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(actor_network, device, run_name, record=True, num_eval_eps=5, render_mode='rgb_array')
  
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()


