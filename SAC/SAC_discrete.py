
import os
import random
import time
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
import cv2
import ale_py
import imageio
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

gym.register_envs(ale_py)
# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "SAC-CarRacing"
    seed = 42
    env_id = "CarRacing-v3"
    alpha = 0.3
    # Training parameters
    total_timesteps = 2000000
    learning_rate = 2.5e-4
    buffer_size = 20000
    gamma = 0.99
    tau = 0.005  # Soft update parameter for target networks
    target_network_frequency = 1
    batch_size = 256
    clip = 0.5
    # exploration_fraction = 0.1
    learning_starts = 25e3
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


class PreprocessAndFrameStack(gym.ObservationWrapper):
    """
    A wrapper that extracts the 'screen' observation, resizes, grayscales,
    and then stacks frames. This simplifies the environment interaction loop.
    Note: The order of operations is important.
    """
    def __init__(self, env, height, width, num_stack):
        # 1. First, apply the stacking wrapper to the raw environment
        env = gym.wrappers.FrameStackObservation(env, num_stack)
        super().__init__(env)
        self.height = height
        self.width = width
        self.num_stack = num_stack

        # The new observation space after all transformations
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.num_stack, self.height, self.width), dtype=np.uint8
        )

    def observation(self, obs):
        # `obs` here is a LazyFrames object from FrameStack of shape (num_stack, H, W, C)
        # 1. Convert LazyFrames to a single numpy array
        # print(obs)
        stack = np.array(obs, dtype=np.uint8)
        
        # 2. Extract 'screen' if obs is a dict (for VizDoom)
        if isinstance(self.env.observation_space, gym.spaces.Dict) and 'screen' in stack[0]:
            stack = np.array([frame for frame in stack])
        else:
            stack = np.array([frame for frame in stack])
        
        # 3. Grayscale and Resize each frame in the stack
        processed_stack = []
        for frame in stack:
            if frame.ndim == 3 and frame.shape[2] == 3: # H, W, C
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            processed_stack.append(frame)
        
        # 4. Stack frames along a new channel dimension
        return np.stack(processed_stack, axis=0)



class FeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # Adjusted for 64x64 input
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)  # Normalize image    
    
class ActorNet(nn.Module):
    def __init__(self, action_space):
        super(ActorNet, self).__init__()
        self.network = FeatureExtractor((4, 84, 84))
        self.actor = nn.Linear(512, action_space)

    def forward(self, x):
        return self.actor(self.network(x))

    def get_action(self, x, action=None, deterministic=False):
        hidden = self.forward(x) # Normalize image
        # pro = self.actor(hidden)
        hidden = torch.nn.functional.softmax(hidden, dim=-1)  # Ensure probabilities sum to 1
        dist = torch.distributions.Categorical(probs=hidden)
        if action is None:
            action = dist.sample()
        if deterministic:
          action = torch.argmax(hidden, dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class QNet(nn.Module):
    def __init__(self, action_space):
        super(QNet, self).__init__()
        self.network = FeatureExtractor((4, 84, 84))
        # self.fc3 = nn.Linear(1024, 512)
        # self.reduce = nn.Linear(512, 256)  # Output a single Q-value
        self.out = nn.Linear(512, action_space)
        
    def forward(self, state):
        x = self.network(state)
        x = self.out(x)
        return x



# --- Environment Creation ---
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    def thunk():
        render_mode = "rgb_array" if eval_mode else None
        # Force RGB24 format for ViZDoom to avoid CRCGCB warning
        env = gym.make(env_id, render_mode=render_mode, continuous=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.AtariPreprocessing(env,
        #     frame_skip=4,  # Standard frame skip for Atari
        #     grayscale_obs=True,  # Add channel dimension for grayscale
        #     scale_obs=True,  # Scale observations to [0, 1]
        #     screen_size=(TARGET_HEIGHT, TARGET_WIDTH),  # Resize to target dimensions
        # )
        # # Use our custom wrapper for all preprocessing
        env = PreprocessAndFrameStack(env, height=84, width=84, num_stack=4)
        # env = gym.wrappers.FrameStackObservation(env, 4)
        
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)  
        # # print(env.unwrapped.get_action_meanings())
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayscaleObservation(env)
        # env = gym.wrappers.FrameStackObservation(env, 4)
         # Use the all-in-one, official Atari wrapper
        # env = gym.wrappers.AtariPreprocessing(
        #     env,
        #     noop_max=30,
        #     frame_skip=4,
        #     screen_size=84, # It assumes square images
        #     terminal_on_life_loss=True, # Standard for training
        #     grayscale_obs=True,
        #     scale_obs=True # We want uint8 [0, 255] for storage
        # )
        
        # Now, stack the preprocessed frames
        # env = ClipRewardEnv(env)  # Clip rewards to [-1, 1]
        # env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


def evaluate(model, device, run_name, num_eval_eps = 10, record = False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, idx=0, run_name=run_name, eval_mode=True)()
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
                action, _, entropy = model.get_action(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0), deterministic=True)
                # action += entropy
                # action = torch.clip(action, args.low, args.high)  # Use args low and high
                action_numpy = action.cpu().numpy().flatten()  # Convert to numpy for environment
            obs, reward, terminated, truncated, _ = eval_env.step(action_numpy[0])
            done = terminated or truncated
            episode_reward += float(reward)

          
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



env = make_env(args.env_id, args.seed, 0, run_name)()

# Breakout has 4 actions: NOOP, FIRE, RIGHT, LEFT
action_dim = 5

actor_net = ActorNet(action_dim).to(device)

q1_network = QNet(action_dim).to(device)
q2_network = QNet(action_dim).to(device)

# target_actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
target_q1_network = QNet(action_dim).to(device)
target_q2_network = QNet(action_dim).to(device)

target_q1_network.load_state_dict(q1_network.state_dict())
target_q2_network.load_state_dict(q2_network.state_dict())
# target_actor_net.load_state_dict(actor_net.state_dict())


actor_optim = optim.Adam(actor_net.parameters(), lr=args.learning_rate)
q_optim = optim.Adam(list(q1_network.parameters()) + list(q2_network.parameters()), lr=args.learning_rate)
# q2_optim = optim.Adam(q2_network.parameters(), lr=args.learning_rate)

# eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q1_network.train()
q2_network.train()
actor_net.train()


replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    # Get action from actor network
    
    with torch.no_grad():  # No need to track gradients for environment interactions
        action, _, entropy = actor_net.get_action(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0))

    # Convert to numpy # for environment step (no gradients needed here)
    # print(action)
    action_numpy = action.cpu().numpy().flatten()  # Convert to numpy array
    
    new_obs, reward, terminated, truncated, info = env.step(action_numpy[0])
    done = terminated or truncated
    
    # Store transition
    replay_buffer.add(obs, new_obs, action_numpy, np.array([reward]), np.array([done]), [info])

    # Log episode returns
    if "episode" in info:
        # print(f"Step={step}, Return={info['episode']['r']}")
        
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                "global_step": step
            })
    if step > args.learning_starts:
        data = replay_buffer.sample(args.batch_size)


        #why min of target q nets? well bruh thats cus the max op is being on them (kinda)! so we have to lower the overestimation of them!!
        with torch.no_grad(): #OOOO WHy Not The Target policy? cus we are using MC sampling and we can get the current up2date signal without the dog following its own tail problem since entropy scrutinizes it
            next_actions, log_pr, entropy = actor_net.get_action(data.next_observations.to(torch.float32))
            target_q1_vals = target_q1_network(data.next_observations.to(torch.float32))
            target_q2_vals = target_q2_network(data.next_observations.to(torch.float32))
            target_q1_selected = target_q1_vals.gather(1, next_actions.unsqueeze(1))
            target_q2_selected = target_q2_vals.gather(1, next_actions.unsqueeze(1)) 
            target_max = torch.min(target_q1_selected, target_q2_selected)
            soft_target = target_max - args.alpha * log_pr.unsqueeze(1) #Why no entropy/? cus dawg we using mc sampling and it is for  a SINGLE action not the whole dist
            td_target = data.rewards + args.gamma * soft_target * (1 - data.dones)

        # print(f"Actions shape: {data.actions.shape}, Observations shape: {data.observations.shape}, TD target shape: {td_target.shape}")
        old_val1 = q1_network(data.observations.to(torch.float32)).gather(1, data.actions)
        old_val2 = q2_network(data.observations.to(torch.float32)).gather(1, data.actions)
        # q1_optim.zero_grad()
        loss1 = nn.functional.mse_loss(old_val1, td_target)
        loss2 = nn.functional.mse_loss(old_val2, td_target)

        loss = loss1 + loss2
        q_optim.zero_grad()
        loss.backward()
        # q1_optim.step()
        # q2_optim.zero_grad()
        # loss2.backward()
        q_optim.step()
        
        
        if step % args.train_frequency == 0:
            
            actions, log_pr, entropy = actor_net.get_action(data.observations.to(torch.float32))
            action_values1 = q1_network(data.observations.to(torch.float32)).gather(1, actions.unsqueeze(1))
            action_values2 = q2_network(data.observations.to(torch.float32)).gather(1, actions.unsqueeze(1))
            action_values = torch.min(action_values1 , action_values2)
            loss = action_values - args.alpha * log_pr.unsqueeze(1)
            loss = -loss.mean()  # Minimize the negative log probability
            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
        
        
        # Log loss and metrics every 100 steps
        if step % 100 == 0:
            if args.use_wandb:
                wandb.log({
                    "losses/td_loss1": loss1.item(),
                    "losses/td_loss2": loss2.item(),
                    # "losses/q_values": old_val.mean().item(),
                    # "step": step
                })
        
        
        # Update target network
            
        if step % args.target_network_frequency == 0:
            for q_params, target_params  in zip(q1_network.parameters(), target_q1_network.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)

            # for actor_params, target_actor_params in zip(actor_net.parameters(), target_actor_net.parameters()):
                # target_actor_params.data.copy_(args.tau * actor_params.data + (1.0 - args.tau) * target_actor_params.data)

            for q_params, target_params in zip(q2_network.parameters(), target_q2_network.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
                
            # ===== MODEL EVALUATION & SAVING =====
        if step!=0 and step % 10000 == 0:
        
            
            # Evaluate model
            episodic_returns, eval_frames = evaluate(actor_net, device, run_name)
            avg_return = np.mean(episodic_returns)
        
            
            
            if args.use_wandb:
                wandb.log({
                    # "val_episodic_returns": episodic_returns,
                    "val_avg_return": avg_return,
                    "val_step": step
                })
        # print(f"Evaluation returns: {episodic_returns}")
        # Log evaluation video to WandB
        # if args.use_wandb and eval_frames:
        #     val_video_path = f"videos/{run_name}/eval/rl-video-episode-{step}.mp4"
            
        #     imageio.mimsave(val_video_path, eval_frames, fps=30)
            
        #     eval_frames = np.array(eval_frames).transpose(0, 3, 1, 2)
        #     wandb.log({"eval_video": wandb.Video(eval_frames, fps=30)})
        
        
    # Update observations
    obs = new_obs
        
# envs.close()
# writer.close()

# Save final video to WandB
if args.use_wandb:
    train_video_path = f"videos/SAC_{args.env_id}.mp4"
    returns, frames = evaluate(actor_net, device, run_name, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
