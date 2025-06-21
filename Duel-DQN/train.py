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
# import wandb  # Commented out as we're using only TensorBoard
from huggingface_hub import HfApi, upload_folder
import cv2
import imageio


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "DQN-CliffWalking"
    seed = 42
    env_id = "CliffWalking-v0"
    
    # Training parameters
    total_timesteps = 300000
    learning_rate = 2e-4
    buffer_size = 30000
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 50
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.4
    learning_starts = 1000
    train_frequency = 4
    max_grad_norm = 4.0  # Maximum gradient norm for gradient clipping
    
    # Logging & saving
    capture_video = True
    save_model = True
    upload_model = True
    hf_entity = ""  # Your Hugging Face username


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        
        self.features = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.values = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        self.adv = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )
        # self.q_value = nn.Linear(512, action_space)
    def forward(self, x):
        
        feat = self.features(x)
        values = self.values(feat)
        adv = self.adv(feat)
        # print(adv.shape)
        res = values + adv - adv.mean(dim=1, keepdim=True) #adding stuf --> big big grads thus normalize babyyy!
        # print(res.shape)
        return res ,  values, adv, feat
    
    
    
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


def calculate_param_norm(model):
    """Calculate the L2 norm of all parameters in a model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calculate_grad_norm(model):
    """Calculate the L2 norm of all gradients in a model."""
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
    )


def one_hot_encode(obs):
    """Convert integer observation to one-hot encoded vector"""
    encoded = np.zeros(48, dtype=np.float32)
    encoded[obs] = 1.0
    return encoded



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

            # with torch.no_grad():
            obs = one_hot_encode(obs)  # 16 is your state space size
            action, values, adv, feat = model(torch.tensor(obs, device=device).unsqueeze(0))
            action = action.argmax().item()
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

# Initialize TensorBoard writer
os.makedirs(f"videos/{run_name}/train", exist_ok=True)
os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
os.makedirs(f"runs/{run_name}", exist_ok=True)
writer = SummaryWriter(f"runs/{run_name}")
print(f"TensorBoard logs will be saved to runs/{run_name}")

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


replay_buffer = ReplayBuffer(args.buffer_size,  gym.spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32), env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    eps = eps_decay(step, args.exploration_fraction)
    rnd = random.random()
    obs = one_hot_encode(obs)  # 16 is your state space size
    if rnd < eps:
        action = env.action_space.sample()
    else:
        action, values, adv , feat = q_network(torch.tensor(obs, device=device).unsqueeze(0))
        action = action.argmax().item()
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    # obs_encoded = one_hot_encode(obs, 16)  # 16 is your state space size
    new_obs_encoded = one_hot_encode(new_obs)
    
    # print(f"Step={step}, Action={action}, Reward={reward}, Obs={obs}, New Obs={new_obs_encoded}, Done={terminated or truncated}")
    done = terminated or truncated
    replay_buffer.add(
    obs,  # Reshape to (1, 500)
    new_obs_encoded,
    np.array(action),
    np.array(reward),
    np.array(done),
    [info]
)
     # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
        
        # TensorBoard logging
        writer.add_scalar("charts/episodic_return", info['episode']['r'], step)
        writer.add_scalar("charts/episodic_length", info['episode']['l'], step)
        writer.add_scalar("charts/epsilon", eps, step)
        # Only log tensor values if they were defined (when using the policy)
        if rnd >= eps:  # Policy was used to select action
            writer.add_scalar("values/q_value", values.mean().item(), step)
            writer.add_scalar("values/q_adv", adv.mean().item(), step)
            writer.add_histogram("features/q_feat", feat.detach().cpu().numpy(), step)

    if step > args.learning_starts and step % args.train_frequency == 0:
        data = replay_buffer.sample(args.batch_size)

        # Q(s t ​ ,a t ​ )←Q(s t ​ ,a t ​ )+α ​ TD target r t ​ +γ a ′ max ​ Q(s t+1 ​ ,a ′ ) ​ ​ −Q(s t ​ ,a t ​ ) ​
        # with torch.no_grad():
        target_max, values, adv, feat = target_net(data.next_observations)
        target_max = target_max.max(1)[0] 
        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        old_val, values, adv , feat = q_network(data.observations)
        old_val = old_val.gather(1, data.actions).squeeze()

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(old_val, td_target)
        
        loss.backward()
        
        # Calculate gradient norm before clipping
        if args.max_grad_norm != 0.0:
            # Calculate gradient norm before clipping
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in q_network.parameters() if p.grad is not None]), 2
            )
            
            # Log gradient norm
            writer.add_scalar("gradients/norm_before_clip", total_norm_before.item(), step)
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=args.max_grad_norm)
            
            # Compute gradient norms after clipping
            total_norm_after = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in q_network.parameters() if p.grad is not None]), 2
            )
            
            writer.add_scalar("gradients/norm_after_clip", total_norm_after.item(), step)
            writer.add_scalar("gradients/clip_ratio", total_norm_after.item() / (total_norm_before.item() + 1e-10), step)
        
        optimizer.step()
            
        # Log loss and metrics every 10000 steps
        if step % 10000 == 0:
            writer.add_scalar("losses/td_loss", loss.item(), step)

        

            
        if step % args.target_network_frequency == 0:
            # Calculate norm of the target network parameters before update
            target_norm_before = calculate_param_norm(target_net)
            
            # Perform soft update of target network
            for q_params, target_params in zip(q_network.parameters(), target_net.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
            
            # Calculate norm of the target network parameters after update
            target_norm_after = calculate_param_norm(target_net)
            
            # Calculate change in target network parameters
            target_norm_delta = abs(target_norm_after - target_norm_before)
            
            # Log target network update statistics
            writer.add_scalar("target_network/norm_before_update", target_norm_before, step)
            writer.add_scalar("target_network/norm_after_update", target_norm_after, step)
            writer.add_scalar("target_network/norm_delta", target_norm_delta, step)
            writer.add_scalar("target_network/update_ratio", target_norm_delta / (target_norm_before + 1e-10), step)
        
        
            # ===== MODEL EVALUATION & SAVING =====
    if step != 0 and args.save_model and step % 100000 == 0:

        # Evaluate model
        episodic_returns, eval_frames = evaluate(q_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        writer.add_scalar("evaluation/avg_return", avg_return, step)
        for i, ret in enumerate(episodic_returns):
            writer.add_scalar(f"evaluation/episode_{i}_return", ret, step)
        
        print(f"Evaluation returns: {episodic_returns}")
   
        
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
        
# env.close()
# writer.close()

# Save final video
train_video_path = f"videos/final.mp4"
returns, frames = evaluate(q_network, device, run_name, record=True, num_eval_eps=1, render_mode='rgb_array')
imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
print(f"Final training video saved to {train_video_path}")

# Close TensorBoard writer
writer.close()

if args.capture_video:
    cv2.destroyAllWindows()


