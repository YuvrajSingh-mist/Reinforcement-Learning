import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import argparse
import wandb
import os
import torch.nn as nn
import numpy as np

class CustomWandbCallback(BaseCallback):
    """
    A custom callback to log SB3 metrics to W&B with names matching the user's custom script.
    It also handles logging the total loss and evaluation metrics with the correct names.
    """
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.is_eval_log = False

    def _on_training_start(self) -> None:
        # Define the mapping from SB3 logger names to your desired W&B names
        self.METRIC_MAP = {
            "rollout/ep_rew_mean": "charts/episodic_return",
            "rollout/ep_len_mean": "charts/episodic_length",
            "train/policy_loss": "losses/policy_loss",
            "train/value_loss": "losses/value_loss",
            "train/ent_loss": "losses/entropy",
            "train/approx_kl": "charts/approx_kl",
            "train/learning_rate": "charts/learning_rate",
        }

    def _on_step(self) -> bool:
        # This part runs every `self.log_freq` steps during training
        log_dict = {}
        
        # 1. Log metrics from the SB3 logger with custom names
        for sb3_name, wandb_name in self.METRIC_MAP.items():
            if sb3_name in self.model.logger.get_latest_values():
                log_dict[wandb_name] = self.model.logger.get_latest_values()[sb3_name]

        # 2. Manually calculate and log the total loss to match the custom script
        if "losses/policy_loss" in log_dict and "losses/value_loss" in log_dict:
            policy_loss = log_dict["losses/policy_loss"]
            value_loss = log_dict["losses/value_loss"]
            entropy_loss = log_dict.get("losses/entropy", 0) # Use .get for safety
            
            # Recreate the total loss calculation from the custom script
            total_loss = (policy_loss -
                          self.model.ent_coef * entropy_loss +
                          self.model.vf_coef * value_loss)
            log_dict["losses/total_loss"] = total_loss

        # 3. Check for and log evaluation metrics from the EvalCallback
        # The EvalCallback stores its results in `self.parent.best_mean_reward` etc.
        # A simpler way is to check the `locals` dict provided to callbacks
        if "eval/mean_reward" in self.logger.get_latest_values():
             # The key from EvalCallback is `eval/mean_reward`, we map it to `eval/avg_return`
             log_dict["eval/avg_return"] = self.logger.get_latest_values()["eval/mean_reward"]

        # Log to wandb if we have anything to log
        if log_dict:
            wandb.log(log_dict, step=self.num_timesteps)

        return True

# ===== SCRIPT START =====
if __name__ == '__main__':
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="Total timesteps for training")
    # MATCHING num_envs from custom script
    parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments") 
    args = parser.parse_args()

    # --- W&B Initialization ---
    group_name = f"NeatRL-Benchmark-SB3-PPO-{args.env}"
    run = wandb.init(
        project="NeatRL", # MATCHING project name
        group=group_name,
        name=f"PPO-SB3-seed-{args.seed}",
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # --- MATCHING HYPERPARAMETERS ---
    # These are now set to be IDENTICAL to the custom PPO implementation's Config class
    batch_size = (args.n_envs * 1024) // 32 # num_envs * max_steps / num_minibatches
    
    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 1024,                  # max_steps
        "batch_size": batch_size,         # Calculated to match
        "n_epochs": 10,                   # PPO_EPOCHS
        "gamma": 0.99,
        "gae_lambda": 0.95,               # GAE
        "clip_range": 0.2,                # clip_value
        "ent_coef": 0.01,                 # ENTROPY_COEFF (CRITICAL FIX)
        "vf_coef": 0.5,                   # VALUE_COEFF
        "max_grad_norm": 0.5,             # Missing in custom, but standard practice
        # --- MATCHING NETWORK ARCHITECTURE (CRITICAL FIX) ---
        # A list means a SHARED network, just like the custom ActorCriticNet
        "policy_kwargs": {"net_arch": [64, 64], "activation_fn": nn.Tanh}
    }
    # Log the matched hyperparameters to W&B
    wandb.config.update(hyperparams)

    # --- Create Vectorized and NORMALIZED Environment ---
    # The `VecNormalize` wrapper handles observation and reward normalization, matching the custom script
    env = make_vec_env(
        args.env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv, # Use DummyVecEnv for simplicity
    )
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        gamma=hyperparams["gamma"],
        clip_reward=10.0 # Matches TransformReward clip
    )
    
    # --- Create a separate, non-normalized environment for evaluation ---
    # The EvalCallback will automatically use the stats from the training env
    eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed, vec_env_cls=SubprocVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, gamma=hyperparams["gamma"])

    # --- Setup Callbacks ---
    # 1. Our custom callback for logging metrics with the right names
    custom_wandb_callback = CustomWandbCallback()
    
    # 2. The EvalCallback for periodic evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'models/{run.id}/',
        log_path=f'models/{run.id}/',
        eval_freq=max((1024 * 10) // args.n_envs, 1), # Eval every 10 updates
        n_eval_episodes=30, # MATCHING num_eval_eps
        deterministic=True, # In eval, usually we take the best action, not a random one
        render=False,
    )

    # --- Create PPO Model ---
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        **hyperparams
    )
    print(model.policy)  # Print the policy architecture to verify it matches the custom script
    # --- Train ---
    # The EvalCallback needs to be passed to learn() to trigger evaluations
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[custom_wandb_callback, eval_callback] # Use a list for multiple callbacks
    )

    # --- FINAL VIDEO CAPTURE ---
    # Load the best model saved by the EvalCallback and record a video
    # print("Capturing final evaluation video with the best model...")
    # best_model_path = os.path.join(f'models/{run.id}/', 'best_model.zip')
    # if os.path.exists(best_model_path):
        # model = PPO.load(best_model_path, env=eval_env)
    
    # video_frames = []
    # Use a fresh, renderable environment for the video
    # video_env = gym.make(args.env, render_mode="rgb_array")
    
    # # Manually run episodes to collect frames
    # for _ in range(5): # Record 5 episodes for the video
    #     obs, _ = video_env.reset()
    #     done = False
    #     while not done:
    #         # Important: must manually normalize obs since video_env is raw
    #         obs = eval_env.normalize_obs(obs)
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, _, terminated, truncated, _ = video_env.step(action)
    #         done = terminated or truncated
    #         video_frames.append(video_env.render())

    # # Save and log the video
    # if video_frames:
    #     import imageio
    #     video_path = f"videos/final_eval_{run.name}.mp4"
    #     os.makedirs(os.path.dirname(video_path), exist_ok=True)
    #     imageio.mimsave(video_path, [np.array(frame) for frame in video_frames], fps=30)
    #     wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
    #     print(f"Final evaluation video saved and logged to W&B.")

    env.close()
    eval_env.close()
    # video_env.close()
    run.finish()