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

import ale_py

gym.register_envs(ale_py)


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "Atari"
    seed = 42
    env_id = "BreakoutNoFrameskip-v4"
    
    # Training parameters
    total_timesteps = 1000000
    learning_rate = 2.5e-4
    buffer_size = 20000
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 50
    batch_size = 256
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.3
    learning_starts = 1000
    train_frequency = 4
    
    # Logging & saving
    capture_video = True
    save_model = True
    upload_model = True
    hf_entity = ""  # Your Hugging Face username
    
    # TensorBoard settings only


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.conv1 = nn.Conv2d(state_space, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_value = nn.Linear(512, action_space)
    def forward(self, x):
        return self.q_value(self.fc2(torch.relu(self.fc1(self.flatten(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x)))))))))))
    
        
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
    env = gym.make(env_id, render_mode='rgb_array')
    
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True,  scale_obs=True)
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

        while not done:

            if(record):
                if (episode_reward > 100):
                    print("Hooray! Episode reward exceeded 500, stopping early.")
                    break
                frame = eval_env.render()
                frames.append(frame)  # Capture all frames

            # with torch.no_grad():
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

# Initialize TensorBoard writer
os.makedirs(f"videos/{run_name}/train", exist_ok=True)
os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
os.makedirs(f"runs/{run_name}", exist_ok=True)
writer = SummaryWriter(f"runs/{run_name}")
print(f"TensorBoard logs will be saved to runs/{run_name}")
os.makedirs(f"runs/{run_name}", exist_ok=True)
writer = SummaryWriter(f"runs/{run_name}")
    
    # Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



env = make_env(args.env_id, args.seed, args.capture_video, run_name)
q_network = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
q_network = q_network.to(device)
target_net = QNet(env.observation_space.shape[0], env.action_space.n).to(device)
target_net.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q_network.train()
target_net.train()


replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    eps = eps_decay(step, args.exploration_fraction)
    rnd = random.random()
    if rnd < eps:
        action = env.action_space.sample()
    else:
        action = q_network(torch.tensor(obs, device=device).unsqueeze(0)).argmax().item()
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay_buffer.add(obs, new_obs, np.array(action), np.array(reward), np.array(done), [info])

     # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
        
        # TensorBoard logging
        writer.add_scalar("charts/episodic_return", info['episode']['r'], step)
        writer.add_scalar("charts/episodic_length", info['episode']['l'], step)
        writer.add_scalar("charts/epsilon", eps, step)
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
        # if step % 100 == 0:
        writer.add_scalar("losses/td_loss", loss.item(), step)
            # writer.add_scalar("losses/q_values", old_val.mean().item(), step)
                
        
        
    # if args.capture_video:
    #     frame = env.render()                     # Render as RGB array
      # If the episode ended this step
        # if done :
        #     cv2.putText(frame, f"Episode Done!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 1, (0, 0, 255), 2, cv2.LINE_AA)
        #     if reward > 200:
        #         cv2.putText(frame, "SUCCESS!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
        #                     1, (0, 255, 0), 2, cv2.LINE_AA)
        #     else:
        #         cv2.putText(frame, "FAILED!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
        #                     1, (0, 0, 255), 2, cv2.LINE_AA)

        # # Overlay step count
        # cv2.putText(frame, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
        #             1, (255, 255, 255), 2, cv2.LINE_AA)

        # # Display the window
        # cv2.imshow("CartPole Training", frame)
        # cv2.waitKey(1)
            
       
        # Update target network
            
        if step % args.target_network_frequency == 0:
            for q_params, target_params  in zip(q_network.parameters(), target_net.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
        
        
            # ===== MODEL EVALUATION & SAVING =====
    if args.save_model and step % 100000 == 0 and step != 0:
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
        # Log evaluation metrics
        writer.add_scalar("evaluation/avg_return", avg_return, step)
        for i, ret in enumerate(episodic_returns):
            writer.add_scalar(f"evaluation/episode_{i}_return", ret, step)
        
        print(f"Evaluation returns: {episodic_returns}")
        # Save evaluation video if needed
        
        
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
        
# env.close()
# writer.close()

# Save final video to WandB
print("Training complete. Evaluating final model...")
# Save final video
train_video_path = f"videos/{run_name}/final.mp4"
returns, frames = evaluate(q_network, device, run_name, record=True)
imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
print(f"Final training video saved to {train_video_path}")

# Close TensorBoard writer
writer.close()

if args.capture_video:
    cv2.destroyAllWindows()
    