import os
import random
import time
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import wandb

import cv2
import imageio


# from pyvirtualdisplay import Display
#     # Start a virtual display
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO(C)-Reacher-Vectorized"
    seed = 42
    env_id = "Reacher-v5"
    total_timesteps = 10_000_000

    # PPO & Agent settings
    lr = 2.5e-4
    gamma = 0.99
    num_envs = 16
    max_steps = 1024
    num_minibatches = 32
    PPO_EPOCHS = 10
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
        self.mu = layer_init(nn.Linear(16, action_space))
        self.sigma = nn.Parameter(torch.zeros(1, action_space))  # Log standard deviation

    def forward(self, x):
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        
        x_mu = torch.nn.functional.tanh(self.fc3(x))
        
        # x_logvar = torch.nn.functional.relu(self.fc3(x))
        mu = self.mu(x_mu)
        # sigma_log = torch.nn.functional.softplus(self.sigma(x_logvar))
        logvar = self.sigma.expand_as(mu)  # Use the same log variance for all actions
        return mu, logvar.exp()

    def get_action(self, x):
        mu, sigma = self.forward(x)
        # sigma = sigma.expand_as(mu)  # Ensure sigma has the same shape as mu
        # sigma = torch.exp(sigma)  # If you want to use exp to get
        #
        # print("Current mu:", mu, "Current sigma:", sigma)
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
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = torch.nn.functional.tanh(self.fc3(x))
        return self.value(x)    
    
def make_env(env_id, seed, idx, capture_video, run_name, gamma, eval_mode=False, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env) # Still useful for eval logging
        env = gym.wrappers.ClipAction(env)

        # ONLY apply normalization to training environments
        if not eval_mode:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        env.action_space.seed(seed + idx)
        return env
    return thunk



def evaluate(envs, model, device, run_name, gamma, num_eval_eps=30, record=False, render_mode=None):
    # Create a "raw" evaluation environment without the normalization wrappers
    eval_env = make_env(Config.env_id, Config.seed, 0, record, run_name, gamma, eval_mode=True, render_mode=render_mode)()
    eval_env.action_space.seed(Config.seed)
    
    model = model.to(device)
    model.eval()
    returns = []
    frames = []

    # Get the observation normalization stats from the training envs
    obs_rms = envs.get_attr("obs_rms")[0] # Get from the first environment in the vector

    for eps in tqdm(range(num_eval_eps), desc="Evaluating"):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
   
        while not done:
            if record:
                frame = eval_env.render()
                frames.append(frame)

            with torch.no_grad():
                # Manually normalize the observation before passing it to the model
                # Note: clip before normalizing, as per the original wrapper order.
                # The obs_rms.mean and obs_rms.var are numpy arrays.
                norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                
                act, _, _ = model.get_action(torch.tensor(norm_obs, device=device, dtype=torch.float32).unsqueeze(0))
                obs, reward, terminated, truncated, _ = eval_env.step(act.cpu().numpy().flatten())
                done = terminated or truncated
                
                episode_reward += reward # Log the RAW reward, not a normalized one
                
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
    # writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor_network = ActorNet(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    critic_network = CriticNet(envs.single_observation_space.shape[0]).to(device)
    optimizer = optim.Adam(list(actor_network.parameters()) + list(critic_network.parameters()), lr=args.lr, eps=1e-5)
    # critic_optim = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

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
        
        frac = 1.0 - (update - 1) / num_updates
        lr = args.lr * frac
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
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
                        wandb.log({"charts/episodic_return": item['episode']['r'], "global_step": global_step})
                        wandb.log({"charts/episodic_length": item['episode']['l'], "global_step": global_step})

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
        b_obs = obs_storage.reshape((-1,) +  envs.single_observation_space.shape)
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

                pg_loss1 = b_advantages[mb_inds] * ratio
                pg_loss2 = b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                current_values = critic_network(b_obs[mb_inds]).squeeze()
                critic_loss = args.VALUE_COEFF * torch.nn.functional.mse_loss(current_values, b_returns[mb_inds])
                
                entropy_loss = entropy.mean()
                loss = policy_loss - args.ENTROPY_COEFF * entropy_loss + critic_loss

                # actor_optim.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                
                grad_norm_dict = {}
                total_norm = 0
                for name, param in actor_network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm_dict[f"gradients/actor_norm_{name}"] = param_norm.item()
                        total_norm += param_norm.item() ** 2
                grad_norm_dict["gradients/actor_total_norm"] = total_norm ** 0.5
                wandb.log(grad_norm_dict)
                
                
                grad_norm_dict = {}
                total_norm = 0
                for name, param in critic_network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm_dict[f"gradients/critic_norm_{name}"] = param_norm.item()
                        total_norm += param_norm.item() ** 2
                grad_norm_dict["gradients/critic_total_norm"] = total_norm ** 0.5
                wandb.log(grad_norm_dict)
                nn.utils.clip_grad_norm_(list(actor_network.parameters()) + list(critic_network.parameters()), 0.5)
                # nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                # actor_optim.step()
                optimizer.step()
        
        if args.use_wandb and update % 10 == 0:
            wandb.log({
                "losses/total_loss": loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": critic_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "charts/learning_rate": optimizer.param_groups[0]['lr'],
                "charts/episodic_return": np.mean(rewards_storage.cpu().numpy()),
                "charts/advantages_mean": b_advantages.mean().item(),
                "charts/advantages_std": b_advantages.std().item(),
                "charts/returns_mean": b_returns.mean().item(),
                # "global_step": global_step,
            })
            print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
    
        if update % 100 == 0:
            episodic_returns, _ = evaluate(envs, actor_network, device, run_name, args.gamma)
            avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return": avg_return,
                    # "global_step": global_step,
                })
            print(f"Evaluation at step {global_step}: Average raw return = {avg_return:.2f}")

    if args.capture_video:
        print("Capturing final evaluation video...")
        episodic_returns, eval_frames = evaluate(envs, actor_network, device, run_name, args.gamma, record=True, num_eval_eps=10, render_mode='rgb_array')
        
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