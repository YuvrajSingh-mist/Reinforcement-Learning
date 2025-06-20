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
    exp_name = "DQN-FrozenLake"
    seed = 42
    env_id = "FrozenLake-v1"
    
    # Training parameters
    total_timesteps = 5000000
    learning_rate = 2.5e-4
    buffer_size = 50000 
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 500
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 4
    
    # Logging & saving
    capture_video = True
    save_model = True
    upload_model = True
    hf_entity = ""  # Your Hugging Face username
    
    # WandB settings
    use_wandb = True
    wandb_project = "cleanRL"
    wandb_entity = ""  # Your WandB username/team


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.q_value = nn.Linear(512, action_space)
    def forward(self, x):
        return self.q_value(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    
    
    
class LinearEpsilonDecay(nn.Module):
    def __init__(self, initial_eps, end_eps, total_timesteps):
        super(LinearEpsilonDecay, self).__init__()
        self.initial_eps = initial_eps
        # self.decay_factor = decay_factor
        self.total_timesteps = total_timesteps
        self.end_eps = end_eps
        
        
    def forward(self, current_timestep, decay_factor):
        slope = (self.end_eps - self.initial_eps) / (self.total_timesteps * decay_factor)
        return max(slope * current_timestep + self.initial_eps, self.end_eps)


def one_hot_encode(obs, n_states):
    """Convert integer observation to one-hot encoded vector"""
    encoded = np.zeros(n_states, dtype=np.float32)
    encoded[obs] = 1.0
    return encoded



def make_env(env_id, seed, capture_video, run_name, eval_mode=False, render_mode=None):
    """Create environment with video recording"""
    env = gym.make(env_id, render_mode=render_mode, map_name="4x4", is_slippery=True)
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

            # with torch.no_grad():
            obs = one_hot_encode(obs, 16)  # 16 is your state space size
            action = model(torch.tensor(obs, device=device).unsqueeze(0)).argmax().item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
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
os.makedirs(f"videos/{run_name}/train")
os.makedirs(f"videos/{run_name}/eval")
os.makedirs(f"runs/{run_name}")
writer = SummaryWriter(f"runs/{run_name}")
    
    # Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env = make_env(args.env_id, args.seed, args.capture_video, run_name)
# print("Obs space:", env.observation_space.n)
q_network = QNet(env.observation_space.n, env.action_space.n).to(device)
q_network = q_network.to(device)
target_net = QNet(env.observation_space.n, env.action_space.n).to(device)
target_net.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q_network.train()
target_net.train()


replay_buffer = ReplayBuffer(args.buffer_size,  gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32), env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    eps = eps_decay(step, args.exploration_fraction)
    rnd = random.random()
    obs = one_hot_encode(obs, 16)  # 16 is your state space size
    if rnd < eps:
        action = env.action_space.sample()
    else:
        action = q_network(torch.tensor(obs, device=device).unsqueeze(0)).argmax().item()
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    # obs_encoded = one_hot_encode(obs, 16)  # 16 is your state space size
    new_obs_encoded = one_hot_encode(new_obs, 16)
    
    # print(f"Step={step}, Action={action}, Reward={reward}, Obs={obs}, New Obs={new_obs_encoded}, Done={terminated or truncated}")
    done = terminated or truncated
    replay_buffer.add(
    obs,  # Reshape to (1, 16)
    new_obs_encoded,
    np.array(action),
    np.array(reward),
    np.array(done),
    [info]
)
     # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
        
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                "epsilon": eps,
                "global_step": step
            })
    if step > args.learning_starts and step % args.train_frequency == 0:
        data = replay_buffer.sample(args.batch_size)

        # Q(s t ​ ,a t ​ )←Q(s t ​ ,a t ​ )+α ​ TD target r t ​ +γ a ′ max ​ Q(s t+1 ​ ,a ′ ) ​ ​ −Q(s t ​ ,a t ​ ) ​
        # with torch.no_grad():
        target_max = target_net(data.next_observations).max(1)[0] # dim=1
        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
        
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(old_val, td_target)
        
        loss.backward()
        optimizer.step()
            
        # Log loss and metrics every 100 steps
        if step % 100 == 0:
            if args.use_wandb:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    # "losses/q_values": old_val.mean().item(),
                    # "step": step
                })
        

            
        if step % args.target_network_frequency == 0:
            for q_params, target_params  in zip(q_network.parameters(), target_net.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
        
        
            # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % 50000 == 0:

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
   
        
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
        
# env.close()
# writer.close()

# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(q_network, device, run_name, record=True, num_eval_eps=1, render_mode='rgb_array')
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()


