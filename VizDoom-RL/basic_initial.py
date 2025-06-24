import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Important for converting to PyTorch tensor
import numpy as np
import matplotlib.pyplot as plt # For visualization
import gymnasium as gym
from vizdoom import gymnasium_wrapper # To get a ViZDoom frame
import cv2 # OpenCV is used by albumentations for many transforms
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
from tqdm import tqdm
from stable_baselines3.common.buffers import ReplayBuffer
# from torch.utils.tensorboard import SummaryWriter
import wandb
# from huggingface_hub import HfApi, upload_folder
import imageio  # For saving videos
# import cv2  # OpenCV for rendering frames

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "DQN-VisDoom-Basic-v0"
    seed = 42
    env_id = "VizdoomBasic-v0"
    
    # Training parameters
    total_timesteps = 1000000
    learning_rate = 2e-4
    buffer_size = 30000
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 50
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

    eval_iter = 10000  # Evaluate every 10,000 steps



class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.conv1 = nn.Conv2d(state_space, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_value = nn.Linear(512, action_space)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)
    
    


# --- Configuration for Preprocessing ---
TARGET_HEIGHT = 128
TARGET_WIDTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # Ensure the array has the right dtype - ALWAYS convert to float32 first
    # This fixes the "Input type (unsigned char) and bias type (float) should be the same" error
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



# Stride
# Input Size−Kernel Size+2×Padding
# ​
#  ⌋+1
    
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



def make_env(env_id, seed, capture_video, run_name, eval_mode=False):
    """Create environment with video recording"""
    env = gym.make(args.env_id, render_mode='rgb_array', frame_skip=4)
    
    # env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # return env

    # # Video recording setup
    # if capture_video:
    #     if eval_mode:
    #         # Evaluation videos
    #         video_prefix = f"videos/{run_name}/eval"
    #     else:
    #         # Training videos
    #         video_prefix = f"videos/{run_name}/train"
    #         env = gym.wrappers.RecordVideo(
    #             env, 
    #             video_prefix,
    #             episode_trigger=lambda x: x % 100 == 0  # Record every 100 episodes
    #         )
    
    env.action_space.seed(seed)

    return env


def evaluate(model, device, run_name, num_eval_eps = 3, record = False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, capture_video=True, run_name=run_name, eval_mode=True)
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
        obs = obs['screen'] if isinstance(obs, dict) and 'screen' in obs else obs
        obs = preprocess_frame_albumentations(obs, TARGET_HEIGHT, TARGET_WIDTH, device)
        obs = obs.unsqueeze(0)  # Add batch dimension
        while not done:

            if(record):
                if (episode_reward > 100):
                    print("Hooray! Episode reward exceeded 500, stopping early.")
                    break
                frame = eval_env.render()
                frames.append(frame)  # Capture all frames

            # with torch.no_grad():
            # obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action = model(obs).argmax().item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            obs = obs['screen'] if isinstance(obs, dict) and 'screen' in obs else obs
            obs = preprocess_frame_albumentations(obs, TARGET_HEIGHT, TARGET_WIDTH, device)
            obs = obs.unsqueeze(0)
            # new_obs = new_obs['screen'] if isinstance(new_obs, dict) and 'screen' in new_obs else new_obs
            # new_obs = preprocess_frame_albumentations(new_obs, TARGET_HEIGHT, TARGET_WIDTH, device)
            # obs = new_obs
          
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
# writer = SummaryWriter(f"runs/{run_name}")
    
    # Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env = make_env(args.env_id, args.seed, args.capture_video, run_name)

# Get a sample observation to determine the correct shapes
sample_obs, _ = env.reset()
if isinstance(sample_obs, dict):
    screen_shape = sample_obs['screen'].shape  # (4, 240, 320, 3)
    # After preprocessing: each frame becomes (1, 128, 128), so 4 frames -> (4, 128, 128)
    processed_shape = (4, TARGET_HEIGHT, TARGET_WIDTH)  # (4, 128, 128)
else:
    screen_shape = sample_obs.shape
    processed_shape = (1, TARGET_HEIGHT, TARGET_WIDTH)

print(f"Original screen shape: {screen_shape}")
print(f"Processed shape: {processed_shape}")

# Initialize networks with correct input shape (4 channels for frame stack)
# Determine action space size safely
try:
    # Try to get action space size from the environment
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_space_size = action_space.n
    elif hasattr(action_space, 'shape') and action_space.shape is not None:
        action_space_size = action_space.shape[0]
    else:
        # Default to 3 if we can't determine (common in ViZDoom)
        action_space_size = 3
    print(f"Using action space size: {action_space_size}")
except Exception as e:
    print(f"Error determining action space size: {e}")
    action_space_size = 3
    print(f"Falling back to default action space size: {action_space_size}")

q_network = QNet(4, action_space_size).to(device)
q_network = q_network.to(device)
target_net = QNet(4, action_space_size).to(device)
target_net.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q_network.train()
target_net.train()

# Initialize replay buffer with correct observation space
obs_space = gym.spaces.Box(low=0, high=1, shape=processed_shape, dtype=np.float32)
replay_buffer = ReplayBuffer(args.buffer_size, obs_space, env.action_space, device=device, handle_timeout_termination=False)


obs = sample_obs  # Use the sample observation from initialization
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    # Extract screen data from observation dictionary
    if isinstance(obs, dict):
        screen_data = obs['screen'] if 'screen' in obs else obs
    else:
        screen_data = obs
        
    # Ensure proper conversion to float32 to avoid type mismatch errors
    transformed_obs = preprocess_frame_albumentations(screen_data, TARGET_HEIGHT, TARGET_WIDTH, device)
    eps = eps_decay(step, args.exploration_fraction)
    rnd = random.random()
    if rnd < eps:
        action = env.action_space.sample()
    else:
        action = q_network(transformed_obs.unsqueeze(0)).argmax().item()
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Process new observation
    if isinstance(new_obs, dict):
        new_screen_data = new_obs['screen'] if 'screen' in new_obs else new_obs
    else:
        new_screen_data = new_obs
    transformed_new_obs = preprocess_frame_albumentations(new_screen_data, TARGET_HEIGHT, TARGET_WIDTH, device)
    
    # Store processed observations in replay buffer
    replay_buffer.add(
        transformed_obs.cpu().numpy(), 
        transformed_new_obs.cpu().numpy(), 
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
        
        
    
        # Update target network
            
        if step % args.target_network_frequency == 0:
            for q_params, target_params  in zip(q_network.parameters(), target_net.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
        
        
            # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % args.eval_iter == 0 and step != 0:
        # Save model
        # model_path = f"runs/{run_name}/model_{step}.pth"
        # torch.save(q_network.state_dict(), model_path)
        # print(f"Model saved to {model_path}")
        
        # Evaluate model
        episodic_returns, eval_frames = evaluate(q_network, device, run_name)
        avg_return = np.mean(episodic_returns)
        # avg_episodic_returns = np.mean(episodic_returns)
        # print(f"Evaluation returns: {episodic_returns}")
        # print(f"Average return: {avg_return:.2f}")
        
        
        if args.use_wandb:
            wandb.log({
                # "val_episodic_returns": episodic_returns,
                "val_avg_return": avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")
        # Log evaluation video to WandB
        # if args.use_wandb and eval_frames:
        #     val_video_path = f"videos/{run_name}/eval/rl-video-episode-{step}.mp4"
            
        #     imageio.mimsave(val_video_path, eval_frames, fps=30)
            
        #     eval_frames = np.array(eval_frames).transpose(0, 3, 1, 2)
        #     wandb.log({"eval_video": wandb.Video(eval_frames, fps=30)})
        
        
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
        
# env.close()
# writer.close()

# Save final video to WandB
print("Training complete. Evaluating final model...")
if args.use_wandb:
    train_video_path = f"images/final.mp4"
    returns, frames = evaluate(q_network, device, run_name, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30)
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
if args.capture_video:
    cv2.destroyAllWindows()
    