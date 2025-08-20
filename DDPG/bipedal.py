
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


import imageio


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "DDPG-BipedalWalker-v0"
    seed = 42
    env_id = "BipedalWalker-v3"
    
    low = -1.0
    noise_clip = 0.5
    high = 1.0  # Action space limits for BipedalWalker
    # Training parameters
    total_timesteps = 4000000
    learning_rate = 3e-4
    buffer_size = 100000
    gamma = 0.99
    tau = 0.005  # Soft update parameter for target networks
    target_network_frequency = 1
    batch_size = 256
   
    exploration_fraction = 0.1
    learning_starts = 25000
    train_frequency = 2
    
    # Logging & saving
    capture_video = True
    # save_model = True
    # upload_model = True
    hf_entity = ""  # Your Hugging Face username
    
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"
    wandb_entity = ""  # Your WandB username/team

    



class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_space)

    def forward(self, x):
        x =  torch.tanh(self.out(torch.nn.functional.mish(self.fc2(torch.nn.functional.mish(self.fc1(x))))))
        x = x * 1.0
        # x = torch.nn.functional.softmax(x, dim=1)  # Apply softmax to get probabilities
        

        return x
    

    
    
class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)  # Output a single Q-value
        self.out = nn.Linear(256, 1)
    def forward(self, state, act):
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


def make_env(env_id, seed, capture_video, run_name, eval_mode=False, render_mode=None):
    """Create environment with video recording"""
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    
    env.action_space.seed(seed)

    return env


def evaluate(model, device, run_name, num_eval_eps = 10, record = False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, capture_video=True, run_name=run_name, eval_mode=True, render_mode='rgb_array')
    eval_env.action_space.seed(Config.seed)
    
    model = model.to(device)
    model = model.eval()
    returns = []
    frames = []

    for eps in range(num_eval_eps):
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

            # with torch.no_grad():
            with torch.no_grad():
                action = model(torch.tensor(obs, device=device).unsqueeze(0))
                # action += torch.randn_like(action) * args.exploration_fraction # Add some noise for exploration
                action = torch.clip(action, args.low, args.high)  # Use args low and high
                action_numpy = action.cpu().numpy().flatten()  # Convert to numpy for environment
            obs, reward, terminated, truncated, _ = eval_env.step(action_numpy)
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


actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

q_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)


target_actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
target_q_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

target_q_network.load_state_dict(q_network.state_dict())
target_actor_net.load_state_dict(actor_net.state_dict())    


actor_optim = optim.Adam(actor_net.parameters(), lr=args.learning_rate)
q_optim = optim.Adam(q_network.parameters(), lr=args.learning_rate)

# eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q_network.train()
actor_net.train()


replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    # eps = eps_decay(step, args.exploration_fraction)
    # rnd = random.random()
    # if rnd < eps:
    #     action = env.action_space.sample()
    # else:
    # Get action from actor network
    with torch.no_grad():  # No need to track gradients for environment interactions
        action = actor_net(torch.tensor(obs, device=device).unsqueeze(0))
        action = action +  torch.clip(torch.randn_like(action) * args.exploration_fraction, -args.noise_clip, args.noise_clip)  # Add some noise for exploration
        action = torch.clip(action, args.low, args.high)
    # Convert to numpy for environment step (no gradients needed here)
    action = action.cpu().numpy().flatten()  # Convert to numpy array and flatten
    
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay_buffer.add(obs, new_obs, action, np.array(reward), np.array(done), [info])

     # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
        
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                
                "action": action,  # Use action_numpy which is already a numpy array
                "global_step": step
            })
    if step > args.learning_starts:
        
        data = replay_buffer.sample(args.batch_size)
        
       
        with torch.no_grad():
            next_actions = target_actor_net(data.next_observations)
            target_max = target_q_network(data.next_observations, next_actions)
            td_target = data.rewards + args.gamma * target_max * (1 - data.dones)
        
        old_val = q_network(data.observations, data.actions)

        q_optim.zero_grad()
        loss = nn.functional.mse_loss(old_val, td_target)
        
        loss.backward()
        q_optim.step()
        
        if step % args.train_frequency == 0:

            actor_optim.zero_grad()
        # returns = torch.tensor(returns, device=device, dtype=torch.float32)
            actions = actor_net(data.observations)
            policy_loss = -q_network(data.observations, actions).mean()  # Maximize Q-value for next actions
            policy_loss.backward()
            actor_optim.step()
        
        # Log loss and metrics every 100 steps
        if step % 100 == 0:
            if args.use_wandb:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    "Q": old_val.mean().item(),
                    # "losses/q_values": old_val.mean().item(),
                    # "step": step
                })
        
        
        # Update target network
            
        if step % args.target_network_frequency == 0:
            for q_params, target_params  in zip(q_network.parameters(), target_q_network.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)

            for actor_params, target_actor_params in zip(actor_net.parameters(), target_actor_net.parameters()):
                target_actor_params.data.copy_(args.tau * actor_params.data + (1.0 - args.tau) * target_actor_params.data)
        
            # ===== MODEL EVALUATION & SAVING =====
    if step % 500 == 0:
     
        
        # Evaluate model
        episodic_returns, eval_frames = evaluate(actor_net, device, run_name)
        avg_return = np.mean(episodic_returns)
      
        
        
        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")
       
        
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
        
# env.close()
# writer.close()

# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/BipedalWalker-v3.mp4"
    returns, frames = evaluate(actor_net, device, run_name, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
    print(f"Final training video saved to {train_video_path}")

    wandb.finish()
