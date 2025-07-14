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
import torch.nn.functional as F
import imageio
# from vizdoom import gymnasium_wrapper

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-RND-Vectorized-CarRacing"
    seed = 42
    env_id = "CarRacing-v3"
 
    total_timesteps = 1_000_000

    # PPO & Agent settings
    lr = 3e-4
    int_gamma = 0.99
    ext_gamma = 0.99
    num_envs = 8    
    max_steps = 128
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.2
    ENTROPY_COEFF = 0.01
    VALUE_COEFF = 0.5
    EXT_COEFF = 2.0  # Weight for extrinsic advantage
    INT_COEFF = 1.0  # Weight for intrinsic advantage
    
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"
    
    # Derived values
    @property
    def batch_size(self):
        return self.num_envs * self.max_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

import matplotlib.pyplot as plt

def safe_display(obs):
    plt.imshow(obs)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.clf()


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer




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
            layer_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)  # Normalize image    
    
class ActorNet(nn.Module):
    def __init__(self, action_space):
        super(ActorNet, self).__init__()
        self.network = FeatureExtractor((4, 84, 84))
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)

    def forward(self, x):
        return torch.nn.functional.softmax(self.actor(self.network(x / 255.0)), dim=-1)

    def get_action(self, x, action=None):
        hidden = self.forward(x) # Normalize image
        # pro = self.actor(hidden)
        dist = torch.distributions.Categorical(probs=hidden)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
# ae



class CriticNet(nn.Module):
    def __init__(self, state_space):
        super(CriticNet, self).__init__()
        self.network = FeatureExtractor((4, 84, 84))
        self.value_ext = layer_init(nn.Linear(512, 1), std=1.0)
        self.value_int = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.value_ext(x), self.value_int(x)

class PredictorNet(nn.Module):
    def __init__(self, state_space):
        super(PredictorNet, self).__init__()
        
        self.network = FeatureExtractor((4, 84, 84))
        
        self.out = layer_init(nn.Linear(512, 256))

    def forward(self, x):
        x = self.network(x / 255.0)
        
        # x = self.out(x)
        return self.out(x)

class TargetNet(nn.Module):
    def __init__(self, state_space):
        super(TargetNet, self).__init__()
        self.network = FeatureExtractor((4, 84, 84))
    
        self.out = layer_init(nn.Linear(512, 256))

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.out(x)

     
     
def make_env(env_id, seed, idx, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode, continuous=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        env = PreprocessAndFrameStack(env, height=84, width=84, num_stack=4)
        # env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

def evaluate(model, device, run_name, num_eval_eps=10, record=False, render_mode=None, display=False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, idx=0, render_mode=render_mode)()
    model.eval()
    returns = []
    frames = []

    for _ in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        rewards = 0.0
        while not done:
            if record:
                frames.append(eval_env.render())
            if display:
                safe_display(obs)
            with torch.no_grad():
                action, _, _ = model.get_action(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0))
                obs, rewards_curr, terminated, truncated, info = eval_env.step(action.cpu().numpy().item())
                done = terminated or truncated
                rewards += rewards_curr
        # if "episode" in info:
            
        returns.append(rewards)
    eval_env.close()
    model.train()
    return returns, frames

# --- Main Execution ---
if __name__ == "__main__":
    args = Config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.use_wandb:
        wandb.init(project=args.wandb_project, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True)
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" 
    env = make_env(args.env_id, args.seed, 0)()
    
    # Create a vectorized environment
    print(env.observation_space)
    print(env.action_space)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i) for i in range(args.num_envs)]
    )
    obs_space_shape = envs.single_observation_space.shape
    action_space_n = envs.single_action_space.n

    actor_network = ActorNet( action_space_n).to(device)
    critic_network = CriticNet(obs_space_shape[0]).to(device)
    predictor_network = PredictorNet(obs_space_shape[0]).to(device)
    target_network = TargetNet(obs_space_shape[0]).to(device)
    
    optimizer = optim.Adam(
        list(actor_network.parameters()) + list(critic_network.parameters()) + list(predictor_network.parameters()), 
        lr=args.lr, eps=1e-5
    )

    for param in target_network.parameters():
        param.requires_grad = False

    # Tensor Storage
    obs_storage = torch.zeros((args.max_steps, args.num_envs) + obs_space_shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    intrinsic_rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    ext_values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    int_values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    
    global_step = 0
    num_updates = args.total_timesteps // args.batch_size
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        # Rollout Phase
        # env = gym.make(args.env_id, render_mode='rgb_array')
        # obs , _ = env.reset(seed=args.seed)
        # print(obs)
        for step in range(0, args.max_steps):
            global_step = (update - 1) * args.batch_size + step * args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_obs)
                ext_value, int_value = critic_network(next_obs)
            
            ext_values_storage[step] = ext_value.flatten()
            int_values_storage[step] = int_value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            next_obs_cpu = next_obs.cpu().numpy()
            
            # --- RND Intrinsic Reward Calculation ---
            with torch.no_grad():
                pred_features = predictor_network(next_obs)
                target_features = target_network(next_obs)
                intrinsic_reward = torch.pow(pred_features - target_features, 2).sum(1)

            intrinsic_rewards_storage[step] = intrinsic_reward
            # print(f"Step {step}",  {intrinsic_reward.mean().item()})

            # Step the environment
            new_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(new_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            wandb.log({
                "train/step": global_step,
                "train/intrinsic_reward": intrinsic_reward.mean().item(),
                "train/obs": next_obs.mean().item(),
                "train/extrinsic_reward": rewards_storage[step].mean().item(),
                "train/total_reward": (intrinsic_reward + rewards_storage[step]).mean().item(),
            })
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        wandb.log({"charts/episodic_return": item['episode']['r'][0], "global_step": global_step})
                        wandb.log({"charts/episodic_length": item['episode']['l'][0], "global_step": global_step})

    
        with torch.no_grad():
            # Get the bootstrapped value from the state after the last step
            bootstrap_ext_value, bootstrap_int_value = critic_network(next_obs)
            
            # Initialize tensors for returns
            ext_returns = torch.zeros_like(rewards_storage).to(device)
            int_returns = torch.zeros_like(intrinsic_rewards_storage).to(device)
            
            # Set the initial "next state" return. If an env was done, this is 0, otherwise it's the bootstrap value.
            # (1.0 - next_done) is a trick to multiply by 0 if done, and 1 if not done.
            ext_gt_next_state = bootstrap_ext_value.squeeze() * (1.0 - next_done)
            int_gt_next_state = bootstrap_int_value.squeeze() * (1.0 - next_done)

            # Loop backwards from the last step to the first
            for t in reversed(range(args.max_steps)):
                # Calculate return at step t
                rt_ext = rewards_storage[t] + args.ext_gamma * ext_gt_next_state
                rt_int = intrinsic_rewards_storage[t] + args.int_gamma * int_gt_next_state
                
                # Store the calculated returns
                ext_returns[t] = rt_ext
                int_returns[t] = rt_int
                
                # Update the "next state" for the previous step (t-1).
                # If an episode was done at step t, the return propagation is cut off (multiplied by 0).
                ext_gt_next_state = rt_ext * (1.0 - dones_storage[t])
                int_gt_next_state = rt_int * (1.0 - dones_storage[t])
            
        # Calculate advantages as the difference between returns and value function estimates
        ext_advantages = ext_returns - ext_values_storage
        int_advantages = int_returns - int_values_storage

        advantages = (args.INT_COEFF * int_advantages) + (args.EXT_COEFF * ext_advantages)
        wandb.log({
            "advantages/ext_advantages": ext_advantages.mean().item(), "advantages/int_advantages": int_advantages.mean().item()})
        # Normalize the final combined advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Combine advantages
        # advantages = args.INT_COEFF * int_advantages + args.EXT_COEFF * ext_advantages

        # Flatten the batch
        b_obs = obs_storage.reshape((-1,) + obs_space_shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)

        # PPO & RND Update Phase
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # --- PPO Policy and Value Loss ---
                _, new_log_probs, entropy = actor_network.get_action(b_obs[mb_inds], b_actions[mb_inds].long())
                ratio = torch.exp(new_log_probs - b_logprobs[mb_inds])
                
                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                ext_current_values, int_current_values = critic_network(b_obs[mb_inds])
                ext_critic_loss = F.mse_loss(ext_current_values.squeeze(), b_ext_returns[mb_inds])
                int_critic_loss = F.mse_loss(int_current_values.squeeze(), b_int_returns[mb_inds])
                critic_loss = ext_critic_loss + int_critic_loss
                
                entropy_loss = entropy.mean()

                # --- RND Predictor Loss ---
                pred_features = predictor_network(b_obs[mb_inds])
                with torch.no_grad():
                    target_features = target_network(b_obs[mb_inds])
                intrinsic_loss = F.mse_loss(pred_features, target_features)

                # Total Loss
                loss = policy_loss + args.VALUE_COEFF * critic_loss + intrinsic_loss - args.ENTROPY_COEFF * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(
                #     list(actor_network.parameters()) + list(critic_network.parameters()) + list(predictor_network.parameters()), 
                #     0.5
                # )
                optimizer.step()
        
        if args.use_wandb:
            wandb.log({
                "losses/total_loss": loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": critic_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "charts/learning_rate": optimizer.param_groups[0]['lr'],
                "charts/episodic_return": np.mean(rewards_storage.cpu().numpy()),
                "charts/advantages_mean": b_advantages.mean().item(),
                "charts/advantages_std": b_advantages.std().item(),
                "charts/ext_returns_mean": b_ext_returns.mean().item(),
                "charts/int_returns_mean": b_int_returns.mean().item(),
                "charts/ext_advantages_mean": ext_advantages.mean().item(),
                "charts/int_advantages_mean": int_advantages.mean().item(),
                # "global_step": global_step,
            })
            # print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
    
        if update % 100 == 0 and update != 0:
            episodic_returns, eval_frames = evaluate(actor_network, device, run_name, record=False, render_mode='rgb_array')
            avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return": avg_return,
                    "global_step": global_step,
                })
            print(f"Evaluation at step {global_step}: Average raw return:  {avg_return:.2f}")
        
         # --- Final Evaluation and Video Saving ---
    if args.capture_video:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(actor_network, device, run_name, record=True, render_mode='rgb_array')
        if len(eval_frames) > 0:
            video_path = f"videos/final_eval_{run_name}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30)
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                
    envs.close()
    writer.close()
    if args.use_wandb:
        wandb.finish()