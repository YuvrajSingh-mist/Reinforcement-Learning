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

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO(C)-Pendulum-v1-Vectorized-OriginalReturns"
    seed = 42
    env_id = "Pendulum-v1"
    total_timesteps = 50000

    # PPO & Agent settings
    lr = 3e-4
    gamma = 0.99
    num_envs = 4
    max_steps = 256
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.2
    ENTROPY_COEFF = 0.01
    VALUE_COEFF = 0.5
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"

    @property
    def batch_size(self):
        return self.num_envs * self.max_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 16))
        self.mu = layer_init(nn.Linear(16, action_space), std=0.01)
        self.sigma = layer_init(nn.Linear(16, action_space), std=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mu = self.mu(x)
        sigma = torch.nn.functional.softplus(self.sigma(x)) + 1e-6
        return mu, sigma

    def get_action(self, x):
        mu, sigma = self.forward(x)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, log_prob, entropy

    def evaluate_get_action(self, x, act):
        mu, sigma = self.forward(x)
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(act).sum(1)
        entropy = dist.entropy().sum(1)
        return log_probs, entropy


class CriticNet(nn.Module):
    def __init__(self, state_space):
        super(CriticNet, self).__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.value = layer_init(nn.Linear(256, 1), std=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.value(x)
    
def make_env(env_id, seed, idx, capture_video, run_name, gamma, eval_mode=False, render_mode=None):
    def thunk():
        if eval_mode:
            env = gym.make(env_id, render_mode=render_mode)
        else:
            env = gym.make(env_id, render_mode=None)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

def evaluate(model, device, run_name, gamma, num_eval_eps=10, record=False, render_mode=None):
    eval_env = gym.make(Config.env_id, render_mode=render_mode)
    eval_env.action_space.seed(Config.seed)
    
    model = model.to(device)
    model.eval()
    returns = []
    frames = []

    for eps in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if record:
                frame = eval_env.render()
                frames.append(frame)

            with torch.no_grad():
                mu, _ = model.forward(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0))
                obs, reward, terminated, truncated, _ = eval_env.step(mu.cpu().numpy().flatten())
                done = terminated or truncated
                episode_reward += reward
        returns.append(episode_reward)
    
    eval_env.close()
    model.train()
    return returns, frames

# --- Main Execution ---
if __name__ == "__main__":
    args = Config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor_network = ActorNet(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    critic_network = CriticNet(envs.single_observation_space.shape[0]).to(device)
    actor_optim = optim.Adam(actor_network.parameters(), lr=args.lr, eps=1e-5)
    critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

    obs_storage = torch.zeros((args.max_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.max_steps, args.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        for step in range(0, args.max_steps):
            global_step = (update - 1) * args.batch_size + step * args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, _ = actor_network.get_action(next_obs)
                value = critic_network(next_obs)
            
            values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        episode_info = item["episode"]
                        if args.use_wandb:
                            wandb.log({
                                "charts/episodic_return": episode_info['r'][0],
                                "charts/episodic_length": episode_info['l'][0],
                                "global_step": global_step
                            })

        # === Advantage Calculation & Returns (YOUR ORIGINAL LOGIC) ===
        with torch.no_grad():
            returns = torch.zeros_like(rewards_storage).to(device)
            
            # 1. Bootstrap value: Get value of the state *after* the rollout ends.
            # This is your 'bootstrap_scalar'.
            bootstrap_value = critic_network(next_obs).squeeze()
            
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

        # === PPO Update Phase ===
        b_obs = obs_storage.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_log_probs, entropy = actor_network.evaluate_get_action(b_obs[mb_inds], b_actions[mb_inds])
                ratio = torch.exp(new_log_probs - b_logprobs[mb_inds])

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                current_values = critic_network(b_obs[mb_inds]).squeeze()
                critic_loss = args.VALUE_COEFF * torch.nn.functional.mse_loss(current_values, b_returns[mb_inds])
                
                entropy_loss = entropy.mean()
                loss = policy_loss - args.ENTROPY_COEFF * entropy_loss + critic_loss

                actor_optim.zero_grad()
                critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic_network.parameters(), 0.5)
                actor_optim.step()
                critic_optim.step()
        
        if args.use_wandb and update % 10 == 0:
            wandb.log({
                "losses/total_loss": loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": critic_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "charts/learning_rate": actor_optim.param_groups[0]['lr'],
                "global_step": global_step,
            })
            print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
    
        if update % 20 == 0:
            episodic_returns, _ = evaluate(actor_network, device, run_name, args.gamma)
            avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return": avg_return,
                    "global_step": global_step,
                })
            print(f"Evaluation at step {global_step}: Average raw return = {avg_return:.2f}")

    if args.capture_video:
        print("Capturing final evaluation video...")
        _, eval_frames = evaluate(actor_network, device, run_name, args.gamma, record=True, num_eval_eps=5, render_mode='rgb_array')
        if len(eval_frames) > 0:
            video_path = f"videos/final_eval_{run_name}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                print(f"Final evaluation video saved and uploaded to WandB.")

    envs.close()
    if args.use_wandb:
        wandb.finish()