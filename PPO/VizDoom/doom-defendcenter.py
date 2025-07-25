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

from vizdoom import gymnasium_wrapper # Ensure ViZDoom is registered

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-Vectorized-VizDoom-DFC"
    seed = 42
    env_id = "VizdoomDefendCenter-v0"
    total_timesteps = 5_000_000 # Standard metric for vectorized training

    # PPO & Agent settings
    lr = 3e-4
    gamma = 0.99
    num_envs = 8  # Number of parallel environments
    max_steps = 128  # Steps per rollout per environment (aka num_steps)
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.2
    ENTROPY_COEFF = 0.01
    VALUE_COEFF = 0.5
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL-Atari"
    
    # Derived values
    @property
    def batch_size(self):
        return self.num_envs * self.max_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

# --- Preprocessing ---
TARGET_HEIGHT = 64
TARGET_WIDTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame_numpy):
    """Preprocesses a single game frame (or a stack of frames)."""
    if frame_numpy.ndim == 4: # (N, H, W, C) from FrameStack
        # Transpose to (N, C, H, W) for PyTorch
        frame_numpy = np.transpose(frame_numpy, (0, 3, 1, 2))
    elif frame_numpy.ndim == 3: # (H, W, C)
        frame_numpy = np.transpose(frame_numpy, (2, 0, 1))
        frame_numpy = np.expand_dims(frame_numpy, 0) # Add batch dim
    
    frame_tensor = torch.from_numpy(frame_numpy.astype(np.float32)).to(DEVICE)
    
    # This assumes input is already grayscale from a wrapper
    # Normalize to [0, 1]
    return frame_tensor / 255.0

class PreprocessAndFrameStack(gym.ObservationWrapper):
    """
    A wrapper that extracts the 'screen' observation, resizes, grayscales,
    and stacks frames while preserving game variables.
    """
    def __init__(self, env, height, width, num_stack):
        super().__init__(env)
        self.height = height
        self.width = width
        self.num_stack = num_stack
        self.frame_buffer = []

        # Get the original observation space
        orig_obs_space = env.observation_space
        
        # Define new observation space as a Dict with screen and gamevariables
        self.observation_space = gym.spaces.Dict({
            'screen': gym.spaces.Box(
                low=0, high=255, 
                shape=(self.num_stack, self.height, self.width), 
                dtype=np.uint8
            ),
            'gamevariables': orig_obs_space['gamevariables']  # Keep original game variables space
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize frame buffer with the first frame
        self.frame_buffer = []
        for _ in range(self.num_stack):
            processed_frame = self._process_frame(obs['screen'])
            self.frame_buffer.append(processed_frame)
        
        return {
            'screen': np.stack(self.frame_buffer, axis=0),
            'gamevariables': obs['gamevariables']
        }, info

    def _process_frame(self, frame):
        """Process a single frame: grayscale and resize"""
        if frame.ndim == 3 and frame.shape[2] == 3:  # RGB to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def observation(self, obs):
        # Process the new frame
        processed_frame = self._process_frame(obs['screen'])
        
        # Update frame buffer (FIFO)
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.num_stack:
            self.frame_buffer.pop(0)
        
        # Return dict with stacked frames and game variables
        return {
            'screen': np.stack(self.frame_buffer, axis=0),
            'gamevariables': obs['gamevariables']
        }

# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNet(nn.Module):
    def __init__(self, action_space):
        super(ActorNet, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)
        
          # --- 2. MLP for Game Variable Processing ---
        self.vars_processor = nn.Sequential(
            layer_init(nn.Linear(2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)), # Feature embedding for variables
            nn.ReLU()
        )
        
        # self.health_and_ammo = nn.Linear(2, 512, bias=False, device=DEVICE)
        self.combined = nn.Linear(640, 512, bias=False, device=DEVICE)
        # self.health = 
    def forward(self, x):
        return self.network(x)
        
    def get_action(self, x, health_ammo=None, action=None):
        hidden = self.forward(x / 255.0) # Normalize image
        
        if health_ammo is not None:
        #    print("sdfsefsadf: ", health_ammo.shape)
           health_ammo_layer = self.vars_processor(health_ammo)
           combined = torch.cat([hidden, health_ammo_layer], dim=-1)
           combined = self.combined(combined)
           logits = self.actor(combined)
        else:
            logits = self.actor(hidden)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        # ammo_and_health = self.health_and_ammo(health_ammo)
        # print("action: ", action.shape)
        # print("Ammo and Health shape:", ammo_and_health.shape)
        # total = torch.cat([action, ammo_and_health], dim=-1)
        # combined = self.combined(total)
        return action, log_prob, entropy

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.var_processor = nn.Sequential(
            layer_init(nn.Linear(2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)), # Feature embedding for variables
            nn.ReLU()
        )
        self.combined = nn.Linear(640, 512, bias=False, device=DEVICE)
        
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x, health_ammo=None):
        # print(x)
        hidden = self.network(x / 255.0) # Normalize image
        if health_ammo is not None:
            health_ammo_layer = self.var_processor(health_ammo)
            combined = torch.cat([hidden, health_ammo_layer], dim=-1)
            combined = self.combined(combined)
            res = self.critic(combined)
            # logits = self.actor(combined)
        else:
            res = self.critic(hidden)
            
        return res

# --- Utility Functions ---
def obs_to_tensor(obs_dict, device):
    """Convert dictionary observation to tensors"""
    return {
        'screen': torch.tensor(obs_dict['screen'], device=device, dtype=torch.float32),
        'gamevariables': torch.tensor(obs_dict['gamevariables'], device=device, dtype=torch.float32)
    }

def obs_dict_to_separated_tensors(obs_dict, device):
    """Convert dict obs to separate screen and gamevar tensors"""
    screen = torch.tensor(obs_dict['screen'], device=device, dtype=torch.uint8)
    gamevars = torch.tensor(obs_dict['gamevariables'], device=device, dtype=torch.float32)
    return screen, gamevars
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    def thunk():
        render_mode = "rgb_array" if eval_mode else None
        # Force RGB24 format for ViZDoom to avoid CRCGCB warning
        env = gym.make(env_id, render_mode=render_mode)
        # print("Game variables shape:", env.observation_space['gamevariables'].shape)
        # print("Game variables bounds:", env.observation_space['gamevariables'].low, "to", env.observation_space['gamevariables'].high)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Use our custom wrapper for all preprocessing
        env = PreprocessAndFrameStack(env, height=TARGET_HEIGHT, width=TARGET_WIDTH, num_stack=4)
        
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

# --- Evaluation ---
def evaluate(actor_model, device, run_name, num_eval_eps=10, record=False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, idx=0, run_name=run_name, eval_mode=True)()
    
    actor_model.to(device)
    actor_model.eval()
    returns = []
    frames = []

    for eps in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if record:
                # Get the raw frame from the original env for nice videos
                frame = eval_env.unwrapped.render()
                frames.append(frame)

            with torch.no_grad():
                # Convert dict observation to tensor
                obs_tensor = obs_to_tensor(obs, device)
                action, _, _ = actor_model.get_action(
                    obs_tensor['screen'].unsqueeze(0), 
                    health_ammo=obs_tensor['gamevariables'].unsqueeze(0)
                )
                # Convert action to scalar integer for ViZDoom
                action_scalar = action.cpu().numpy().item()
                obs, reward, terminated, truncated, info = eval_env.step(action_scalar)
                done = terminated or truncated
                episode_reward += float(reward)
                # Use raw reward from info if available
                # if "episode" in info:
                #     episode_reward = info["episode"]["r"]
          
        returns.append(episode_reward)
      
    eval_env.close()
    actor_model.train()
    return returns, frames

# --- Main Execution ---
if __name__ == "__main__":
    args = Config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.use_wandb:
        wandb.init(project=args.wandb_project, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create Synchronous Vectorized Environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
    )
    
    # Get environment specs from a single environment
    single_env = make_env(args.env_id, args.seed, 0, run_name)()
    action_space_n = single_env.action_space.n  # type: ignore
    screen_shape = single_env.observation_space['screen'].shape  # type: ignore
    gamevars_shape = single_env.observation_space['gamevariables'].shape  # type: ignore
    single_env.close()
    
    actor_network = ActorNet(action_space_n).to(DEVICE)
    critic_network = CriticNet().to(DEVICE)
    optimizer = optim.Adam(list(actor_network.parameters()) + list(critic_network.parameters()), lr=args.lr, eps=1e-5)

    # print(envs.single_observation_space)
    # print(envs.single_action_space)
    # Tensor Storage for dict observations
    screen_storage = torch.zeros((args.max_steps, args.num_envs) + screen_shape, dtype=torch.uint8).to(DEVICE)
    gamevars_storage = torch.zeros((args.max_steps, args.num_envs) + gamevars_shape, dtype=torch.float32).to(DEVICE)
    actions_storage = torch.zeros((args.max_steps, args.num_envs)).to(DEVICE)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(DEVICE)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(DEVICE)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(DEVICE)
    values_storage = torch.zeros((args.max_steps, args.num_envs)).to(DEVICE)
    
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_screen, next_gamevars = obs_dict_to_separated_tensors(next_obs, DEVICE)
    next_done = torch.zeros(args.num_envs).to(DEVICE)

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        # Rollout Phase
        for step in range(0, args.max_steps):
            global_step = (update - 1) * args.batch_size + step * args.num_envs
            screen_storage[step] = next_screen
            gamevars_storage[step] = next_gamevars
            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_screen, health_ammo=next_gamevars)
                value = critic_network(next_screen)
            
            values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_storage[step] = torch.tensor(reward).to(DEVICE).view(-1)
            next_screen, next_gamevars = obs_dict_to_separated_tensors(next_obs, DEVICE)
            next_done = torch.Tensor(done).to(DEVICE)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        episode_info = item["episode"]
                        writer.add_scalar("charts/episodic_return", episode_info['r'][0], global_step)
                        writer.add_scalar("charts/episodic_length", episode_info['l'][0], global_step)

         # === Advantage Calculation & Returns (YOUR ORIGINAL LOGIC) ===
        with torch.no_grad():
            returns = torch.zeros_like(rewards_storage).to(DEVICE)
            
            # 1. Bootstrap value: Get value of the state *after* the rollout ends.
            # This is your 'bootstrap_scalar'.
            bootstrap_value = critic_network(next_screen).squeeze()
            
            # 2. Initialize the return for the next state (gt_next_state).
            # For any environment that was 'done' at the end of the rollout, this is 0.
            # Otherwise, it's the bootstrapped value.
            gt_next_state = bootstrap_value * (1.0 - next_done)

            # 3. Loop backwards through all the steps of the rollout.
            # This is your 'for reward_at_t, done_at_t in zip(reversed(rewards), reversed(dones))'
            for t in reversed(range(args.max_steps)):
                # 4. Calculate the return at the current step 't'.
                # This is your 'rt = reward_at_t + args.gamma * gt_next_state'.
                rt = rewards_storage[t] + args.gamma * gt_next_state
                
                # 5. Store this return in our storage tensor. This is your 'returns.insert(0, rt)'.
                returns[t] = rt
                
                # 6. Update gt_next_state for the *previous* step in the loop (t-1).
                # The 'gt_next_state' becomes the return we just calculated.
                # However, if an episode terminated at this step 't', the value from this
                # point on is reset, so the return is just the reward.
                # The logic `if done_at_t: gt_next_state = 0` followed by `gt_next_state = rt`
                # is equivalent to `gt_next_state = rt * (1.0 - dones_storage[t]) + rewards_storage[t] * dones_storage[t]`
                # For simplicity and correctness, the standard is to use the computed return `rt`
                # and let the *next* iteration handle the `done` flag.
                # Your logic `gt_next_state = rt` is correct for the next step. We reset the propagated value
                # using the 'done' flag from the *next* state in the sequence.
                gt_next_state = returns[t] * (1.0 - dones_storage[t]) # If done at t, the next gt is 0
         # Calculate advantages using the computed returns and stored values
        advantages = returns - values_storage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten the batch - separate screen and gamevars
        b_screens = screen_storage.reshape((-1,) + screen_storage.shape[2:])
        b_gamevars = gamevars_storage.reshape((-1,) + gamevars_storage.shape[2:])
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # PPO Update Phase
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_probs, entropy = actor_network.get_action(
                    b_screens[mb_inds], 
                    b_gamevars[mb_inds],
                    b_actions[mb_inds].long() 
                    
                )
                ratio = torch.exp(new_log_probs - b_logprobs[mb_inds])

                # Policy Loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                current_values = critic_network(b_screens[mb_inds]).squeeze()
                critic_loss = args.VALUE_COEFF * torch.nn.functional.mse_loss(current_values, b_returns[mb_inds])
                
                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = policy_loss - args.ENTROPY_COEFF * entropy_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(list(actor_network.parameters()) + list(critic_network.parameters()), 0.5)
                optimizer.step()
        
        
        if update % 20 == 0:
            episodic_returns, _ = evaluate(actor_network, DEVICE, run_name, record=True, num_eval_eps=5)
            avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return": avg_return,
                    # "global_step": global_step,
                })
            print(f"Evaluation at step {global_step}: Average raw return = {avg_return:.2f}")


        # Logging
        if args.use_wandb and update % 1 == 0:
            writer.add_scalar("losses/total_loss", loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", critic_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

    # --- Final Evaluation and Video Saving ---
    if args.capture_video:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(actor_network, DEVICE, run_name, record=True, num_eval_eps=5)
        if len(eval_frames) > 0:
            video_path = f"videos/final_eval_{run_name}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})

    envs.close()
    writer.close()
    if args.use_wandb:
        wandb.finish()

