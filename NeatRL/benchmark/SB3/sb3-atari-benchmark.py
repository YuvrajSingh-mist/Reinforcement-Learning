import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
import imageio
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from wandb.integration.sb3 import WandbCallback

gym.register_envs(ale_py)

# ===== CONFIGURATION (Mirrors the custom script) =====
class Config:
    # Experiment settings
    exp_name = "PPO-SB3-Atari-Benchmark"
    seed = 42
    env_id = "BoxingNoFrameskip-v4"
    total_timesteps = 10_000_000

    # PPO & Agent settings
    lr = 2.5e-4
    gamma = 0.99
    num_envs = 8  # Number of parallel environments
    n_steps = 128  # Steps per rollout per environment (max_steps in custom)
    num_minibatches = 4
    n_epochs = 4   # PPO_EPOCHS in custom
    clip_range = 0.1 # clip_value in custom
    ent_coef = 0.01  # ENTROPY_COEFF
    vf_coef = 0.5    # VALUE_COEFF
    gae_lambda = 0.95 # GAE
    max_grad_norm = 0.5
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"
    
    # Evaluation
    eval_freq_updates = 200 # Evaluate every 200 updates
    num_eval_episodes = 5

    # Derived values
    @property
    def batch_size(self):
        # In SB3, this is the minibatch size.
        return (self.num_envs * self.n_steps) // self.num_minibatches
    
    @property
    def eval_freq_steps(self):
        # Convert update frequency to step frequency
        return self.eval_freq_updates * self.n_steps

# --- Environment Creation (Mirrors the custom script) ---
def make_env(env_id, seed, idx):
    # def thunk():
    env = gym.make(env_id)
    # Use the all-in-one, official Atari wrapper from Gymnasium
    # This handles: No-op resets, frame skipping, resizing, grayscaling, life-based terminals, and reward clipping.
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=True # Keep as False to match custom script's uint8 storage, SB3 handles scaling
    )
    # Stack the preprocessed frames
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed + idx)
    env.observation_space.seed(seed + idx)
    return env
    # return thunk

# --- Custom Network Architecture (Mirrors the custom script) ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor to match the architecture of the custom PyTorch script.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # The observation space from FrameStack is (4, 84, 84)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, features_dim)),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 automatically handles the normalization of images (dividing by 255)
        return self.linear(self.cnn(observations))

# ===== SCRIPT START =====
if __name__ == '__main__':
    # --- Setup ---
    args = Config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # --- Set seeds for reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # --- Create Vectorized Environment ---
    env = make_vec_env(
        args.env_id,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv, # Use DummyVecEnv for simplicity
    )
    # --- Define Hyperparameters and Policy Architecture ---
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [],  # No extra hidden layers between extractor and heads
        "activation_fn": nn.ReLU,
    }

    # --- Create PPO Model ---
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        tensorboard_log=f"runs/{run.id}" if args.use_wandb else None,
        verbose=1,
    )

    # --- Setup Callbacks ---
    callbacks = []
    # 1. Evaluation Callback
    # Create a separate, non-vectorized env for evaluation
    eval_env = make_vec_env(args.env_id, n_envs=1, seed=args.seed, vec_env_cls=DummyVecEnv)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'models/{run.id}/' if args.use_wandb else None,
        log_path=f'models/{run.id}/' if args.use_wandb else None,
        eval_freq=max(args.eval_freq_steps // args.num_envs, 1),
        n_eval_episodes=args.num_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # 2. W&B Callback
    if args.use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=10_000, # Log gradients periodically
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # --- Train ---
    print("Policy Architecture:")
    print(model.policy)
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )


    # --- Cleanup ---
    env.close()
    eval_env.close()
    if 'video_env' in locals():
        video_env.close()
    if args.use_wandb:
        run.finish()