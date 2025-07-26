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
# import ale_py
from pettingzoo.butterfly import cooperative_pong_v5
import importlib
import supersuit as ss

# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "PPO-PettingZoo-Pong-MAPPO"
    seed = 42
    env_id = "cooperative_pong_v5"  # Environment ID
    total_timesteps = 10_000_000  # Standard metric for vectorized training

    # PPO & Agent settings
    lr = 3e-4
    # Discount factors for extrinsic and intrinsic learning
    ext_gamma = 0.99  # Extrinsic discount
    int_gamma = 0.999  # Intrinsic discount
    gamma = ext_gamma  # Back-compat single gamma

    # Advantage scaling coefficients (see CarRacing RND)
    EXT_COEFF = 2.0
    INT_COEFF = 1.0
    num_envs = 16  # Number of parallel environments
    max_steps = 128  # Steps per rollout per environment (aka num_steps)
    num_minibatches = 4
    PPO_EPOCHS = 4
    clip_value = 0.2 
    clip_coeff = 0.2  # Value clipping coefficient
    ENTROPY_COEFF = 0.01
    
    VALUE_COEFF = 0.5
    
    # Logging & Saving
    capture_video = True
    use_wandb = True
    wandb_project = "cleanRL"
    
    GAE = 0.95  # Generalized Advantage Estimation
    anneal_lr = True  # Whether to linearly decay the learning rate
    max_grad_norm = 0.5  # Gradient clipping value
    num_agents = 2  # Number of agents in the environment
    
    # Derived values
    @property
    def batch_size(self):
        return self.num_envs * self.max_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches




# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardShapingWrapper(BaseParallelWrapper):
    """
    This wrapper reshapes the reward to be sparse and meaningful.
    - +1 for hitting the ball back (detecting a change in ball's x-velocity).
    - -1 for the ball going off-screen (termination).
    - 0 for all other timesteps.
    """
    def __init__(self, env):
        super().__init__(env)
        # We need to access the environment's internal state.
        self.unwrapped_env = env.unwrapped 
        self._last_ball_x_velocity = {agent: 0 for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        # Store the initial velocity of the ball
        self._last_ball_x_velocity = self.unwrapped_env.ball.velocity[0]
        return observations, infos

    def step(self, actions):
        observations, original_rewards, terminations, truncations, infos = self.env.step(actions)
        
        current_ball_x_velocity = self.unwrapped_env.ball.velocity[0]
        
        hit_detected = np.sign(current_ball_x_velocity) != np.sign(self._last_ball_x_velocity) and self._last_ball_x_velocity != 0
        self._last_ball_x_velocity = current_ball_x_velocity

        new_rewards = {}
        for agent in self.agents:
            if hit_detected:
                new_rewards[agent] = 1.0  # Big reward for hitting!
            elif terminations[agent]:
                new_rewards[agent] = -1.0 # Big penalty for losing
            else:
                new_rewards[agent] = 0.0   # No reward for waiting
        
        return observations, new_rewards, terminations, truncations, infos

class Actor(nn.Module):
    def __init__(self, action_space):
        super(Actor, self).__init__()
        # Shared CNN feature extractor
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )
        # Actor head
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)
        # Critic head
        # self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_features(self, x):
        return self.network(x)

   
    def get_action(self, x, action=None, deterministic=False):

        x = x.clone()
        x = x.permute(0, 3, 1, 2)
        # x[:, :4, :, :] /= 255.0
        
        features = self.get_features(x)
        logits = self.actor(features)
        # probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        # print("Actions: ", torch.nn.functional.softmax(logits, dim=-1))
        if deterministic:
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
            
        if not deterministic and action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def evaluate_get_action(self, x, action):
        # print("Eval: ", x.shape)
        x = x.clone()
        x = x.permute(0, 3, 1, 2)
        # x[:, :4, :, :] /= 255.0
        features = self.get_features(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
    

class Critic(nn.Module):
    """Bird-eye critic with separate extrinsic and intrinsic value heads."""
    def __init__(self):
        super(Critic, self).__init__()
        # Shared CNN feature extractor
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6 * args.num_agents, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # Adjusted for 64x64 input
            nn.ReLU(),
        )
        # 
        # Critic head
        # Two value heads
        self.value_ext = layer_init(nn.Linear(512, 1), std=1.0)
        self.value_int = layer_init(nn.Linear(512, 1), std=1.0)

    def get_features(self, x):
        return self.network(x)

    def forward(self, x):
        # x = x.clone()
        # x = x.permute(0, 3, 1, 2)
        # x[:, :4, :, :] /= 255.0
        # x[:, 6:10, :, :] /= 255.0
        # print(x.shape)
        features = self.get_features(x)
        return self.value_ext(features), self.value_int(features)

# --- RND Networks ---
class PredictorNet(nn.Module):
    def __init__(self):
        super(PredictorNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.out = layer_init(nn.Linear(512, 256))

    def forward(self, x):
        x = x.clone().permute(0, 3, 1, 2)
        feats = self.feature_extractor(x)
        return self.out(feats)

class TargetNet(nn.Module):
    def __init__(self):
        super(TargetNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.out = nn.Linear(512, 256)

    def forward(self, x):
        x = x.clone().permute(0, 3, 1, 2)
        feats = self.feature_extractor(x)
        return self.out(feats)

# --- Environment Creation ---
def make_env(env_id, seed, idx, run_name, eval_mode=False):
    
    env = importlib.import_module(f"pettingzoo.butterfly.{args.env_id}").parallel_env()
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)  # <--- Required to initialize np_random
    env = RewardShapingWrapper(env)
    env = ss.max_observation_v0(env, 2)
    # env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")
    # print(env.observation_space)
    # print(env.action_space)
    # envs.single_observation_space = single_observation_space
    # envs.single_action_space = single_action_space
    # envs.is_vector_env = True
    # envs = gym.wrappers.RecordEpisodeStatistics(env)
  
    return env

def reshape_obs_shape(obs):
    num_games = args.num_envs // args.num_agents # e.g., 16 // 2 = 8 games
    ma_obs_shape = (num_games, args.num_agents) + single_observation_space.shape # (8, 2, 84, 84, 6)
    ma_obs = obs.reshape(ma_obs_shape)
    return ma_obs

def evaluate(model, device, seed, num_eval_eps=10, record=False):
    eval_env = cooperative_pong_v5.env(render_mode="rgb_array")
    env = RewardShapingWrapper(env)
    env = ss.max_observation_v0(eval_env, 2)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    eval_env = ss.agent_indicator_v0(env, type_only=False)
    model.eval()
    
    all_episode_rewards = {agent: [] for agent in eval_env.possible_agents}
    frames = []

    for i in tqdm(range(num_eval_eps), desc="Evaluating"):
        eval_env.reset(seed=args.seed + i)
        episode_rewards = {agent: 0 for agent in eval_env.possible_agents}
        
        # Initialize CV2 window only if we're displaying
        # if not record:
        #     cv2.namedWindow('Evaluation', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('Evaluation', 640, 480)
        
        for agent in eval_env.agent_iter():
            obs, reward, terminated, truncated, info = eval_env.last()
            done = terminated or truncated
            episode_rewards[agent] += reward
            
            obs_tensor = torch.Tensor(obs).to(device) 
            obs_tensor /= 255.0
            # print(obs_tensor.shape)
            if done:
                eval_env.step(None)
                continue
            
            # mask = []
            # for agent_idx in range(args.num_agents):
            #     mask.append(obs_tensor[:, 0, 0, (4 + agent_idx)] == 1.0)

            # for agent_idx in range(args.num_agents):
            
            with torch.no_grad():

                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension


                action, logprobs, _ = model.get_action(obs_tensor, deterministic=True)

            eval_env.step(action.cpu().item())
            # print(logprobs.exp())
            # Get and process frame
            frame = eval_env.render()
            # print(f"Frame shape: {frame.shape }")
            # print(f"Frame dtype: {frame.dtype }")
            if frame is None:
                continue
                
            # Convert frame to BGR for OpenCV if needed
            if frame.ndim == 3 and frame.shape[2] == 3:  # RGB format
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame
                
            # Display or record frame
            if record:
                frames.append(frame)
            else:
                cv2.imshow('Evaluation', display_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):  # 25ms delay (~40fps)
                    break
                    

        # Store episode rewards for each agent
        for agent, r in episode_rewards.items():
            all_episode_rewards[agent].append(r)
        if not record:
            cv2.destroyAllWindows()

    eval_env.close()
    model.train()
    
    avg_return1 = np.mean(all_episode_rewards['paddle_0'])
    avg_return2 = np.mean(all_episode_rewards['paddle_1'])
    return all_episode_rewards['paddle_0'], all_episode_rewards['paddle_1'], avg_return1, avg_return2, frames


# --- Checkpoint Saving Function ---
def save_checkpoint(model, optimizer, update, global_step, args, run_name, save_dir="./checkpoints"):
    """Save model and optimizer state as a checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"actor_network_{run_name}_update{update}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': update,
        'global_step': global_step,
        'args': vars(args)
    }, model_path)
    print(f"Model checkpoint saved at {model_path}")
    
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

    # 3. Create multiple parallel games
    env = make_env(env_id=args.env_id, seed=args.seed, idx=0, run_name=run_name)
    sample_agent = env.possible_agents[0]
    single_action_space = env.action_space(sample_agent)
    single_observation_space = env.observation_space(sample_agent)
    assert isinstance(single_action_space, gym.spaces.Discrete), "Action space is not Discrete"
    print(f"Single agent action space: {single_action_space} (n={single_action_space.n})")
    print(f"Single agent observation space shape: {single_observation_space.shape}")
        # envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    
    # print("Vectorizing environments...")
    # envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")
    critic_network = Critic()
    actor_network = Actor(single_action_space.n)
    predictor_network = PredictorNet()
    target_network = TargetNet()
    # target_network.load_state_dict(predictor_network.state_dict())
    target_network.eval()

    critic_network = critic_network.to(device)
    actor_network = actor_network.to(device)
    predictor_network = predictor_network.to(device)
    target_network = target_network.to(device)
    # Include predictor params, target frozen
    optimizer = optim.Adam(
        list(critic_network.parameters()) + list(actor_network.parameters()) + list(predictor_network.parameters()),
        lr=args.lr,
        eps=1e-5,
    ) 
    
    # critic_optimizer = optim.Adam(critic_network.parameters(), lr=args.lr, eps=1e-5)

    print("Wrapping PettingZoo env to be Gymnasium VectorEnv compliant...")
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Step D.2: Now that `vec_env` is compliant, stack it.
    # `vec_env` treats each of the 2 agents as a parallel environment.
    # To get `args.num_envs` (e.g., 16) total parallel agent streams, we need to stack
    # `args.num_envs // args.num_agents` copies.
    print(f"Stacking {args.num_envs // args.num_agents} environments to get {args.num_envs} parallel agent streams...")
    envs = ss.concat_vec_envs_v1(vec_env, 
                                num_vec_envs=args.num_envs // args.num_agents, 
                                num_cpus=0, 
                                base_class="gymnasium")

    # E. Now you can reset the environment without error!
    print("Resetting vectorized environments...")
    # This will now correctly return two values.
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_obs /= 255.0
    
    print(f"Successfully reset envs. Observation shape: {next_obs.shape}")
    
    # Reshape for MAPPO logic
    num_games = args.num_envs // args.num_agents # e.g., 16 // 2 = 8 games
    
    ma_obs = reshape_obs_shape(next_obs)
    print(f"Reshaped observation to multi-agent format: {ma_obs.shape}")
    
    actor_network = actor_network.to(device)

    critic_network = critic_network.to(device)
    global_state = torch.zeros((args.max_steps, num_games, ma_obs.shape[-1] * args.num_agents, ) + single_observation_space.shape).to(device)
    obs_storage = torch.zeros((args.max_steps, num_games, args.num_agents) + single_observation_space.shape, device=device)
    next_obs_storage = torch.zeros((args.max_steps, num_games, args.num_agents) + single_observation_space.shape, device=device)
    actions_storage = torch.zeros((args.max_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs_storage = torch.zeros((args.max_steps, args.num_envs), device=device)
    ext_rewards_storage = torch.zeros((args.max_steps, args.num_envs), device=device)
    int_rewards_storage = torch.zeros((args.max_steps, args.num_envs), device=device)
    dones_storage = torch.zeros((args.max_steps, args.num_envs), device=device)
    ext_values_storage = torch.zeros((args.max_steps, args.num_envs), device=device)
    int_values_storage = torch.zeros((args.max_steps, args.num_envs), device=device)

    # Episode tracking variables
    # episodic_return_reward = np.zeros((args.num_envs, args.num_agents))
    # episode_step_count = np.zeros((args.num_envs, args.num_agents))

    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    # next_obs, _ = envs.reset(seed=args.seed)
    # next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    # next_done_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    bootstrap_value = torch.zeros(args.num_envs).to(device)
    # next_obs_vec = torch.zeros(1, args.num_envs, args.num_agents).to(device)
    for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
        
        frac = 1.0 - (update / num_updates)
        lr = args.lr * frac
        
        # for agent_idx, actor in enumerate(actor_network_ls):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #  for agent_idx, actor in enumerate(actor_network_ls):
        # for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
                
        for step in range(0, args.max_steps):
            
            global_step += args.num_envs
            masks = []
            
            for agent_idx in range(args.num_agents):
                
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 1.0 #1d tensor
                # print(mask)
                masks.append(mask)
            
            # Cant' just use RESHAPE CUS THE PARALLEL ENVS - ONE OF THEM CAN GET OVER EARLY AND PUT THAT DATA ANYWHERE IN THE BACTH
            ma_obs = torch.zeros((num_games, args.num_agents) + single_observation_space.shape).to(device)
            # print(next_obs[:, 0, 0, (4+1)]) # ma_obs = reshape_obs_shape(next_obs)
            # next_obs[:, :, :, :4] /= 255.0  # Normalize the first 4 channels
            # print(ma_obs.shape)
            # print(next_obs.shape)
        
            # print(masks[0])
            # masks = []
            for agent_idx in range(args.num_agents):
                # print(next_obs[masks[agent_idx]])
                agent_obs = next_obs[masks[agent_idx]]
                ma_obs[:, agent_idx , ...] = agent_obs
            # print(ma_obs.shape)
            obs_storage[step] = ma_obs
            # print(ma_obs.shape)
            global_state = ma_obs.permute(0, 1, 4, 2, 3).reshape(num_games, -1, 84, 84)
            
            actions_per_step = torch.zeros(args.num_envs, device=device)
            
            with torch.no_grad():
                    # print("No eval: ", global_state.shape)
                    value_ext, value_int = critic_network(global_state)
                                
            for agent_idx in range(args.num_agents):
                
                with torch.no_grad():
                    action, logprob, _ = actor_network.get_action(next_obs[masks[agent_idx]], deterministic=False)
                    # value = critic_network(global_state[step])
                actions_per_step = actions_per_step.long()
                actions_per_step[masks[agent_idx]] = action.long()
              
                ext_values_storage[step][masks[agent_idx]] = value_ext.flatten()
                int_values_storage[step][masks[agent_idx]] = value_int.flatten()
             
                logprobs_storage[step][masks[agent_idx]] = logprob
            
            actions_storage[step] = actions_per_step
            # print(actions_per_step)
            next_obs, reward, terminated, truncated, info = envs.step(actions_per_step.cpu().numpy())
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32) / 255.0
            next_ma_obs = torch.zeros((num_games, args.num_agents) + single_observation_space.shape).to(device)
            
            done = np.logical_or(terminated, truncated)
            
            for agent_idx in range(args.num_agents):
                agent_obs = next_obs[masks[agent_idx]]
                next_ma_obs[:, agent_idx , ...] = agent_obs
            
            ext_rewards_storage[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            # --- intrinsic reward ---
            with torch.no_grad(): #Use next_obs not current_obs
                pred_features = predictor_network(next_ma_obs.reshape(-1, *single_observation_space.shape))
                targ_features = target_network(next_ma_obs.reshape(-1, *single_observation_space.shape))
                int_r = (pred_features - targ_features).pow(2)
                int_r = int_r.mean(1)
                int_r = int_r.view(args.num_envs)  # align with env order
            int_rewards_storage[step] = int_r
            next_obs_storage[step] = next_ma_obs
            next_done = torch.Tensor(done).to(device)
            dones_storage[step] = next_done

        
        with torch.no_grad():
            
        #Creating masks for GAE
            # next_obs[:, :, :, :4] /= 255.0  # Normalize the first 4 channels
            masks = []
            for agent_idx in range(args.num_agents):
                mask = next_obs[:, 0, 0, (4 + agent_idx)] == 1.0
                masks.append(mask)

            ext_advantages = torch.zeros((num_games, args.max_steps, args.num_agents), device=device)
            int_advantages = torch.zeros_like(ext_advantages)
            ma_obs = torch.zeros((num_games, args.num_agents) + single_observation_space.shape).to(device)
            
            for agent_idx in range(args.num_agents):
                agent_obs = next_obs[masks[agent_idx]]
                ma_obs[:, agent_idx , ...] = agent_obs
            
            global_state = ma_obs.permute(0, 1, 4, 2, 3).reshape(num_games, -1, 84, 84)
            ext_bootstrap, int_bootstrap = critic_network(global_state)
            ext_bootstrap = ext_bootstrap.squeeze()
            int_bootstrap = int_bootstrap.squeeze()  
            # print(values_storage.shape, next_done.shape)
       
            done_storage_gae = dones_storage.reshape((args.max_steps, num_games, args.num_agents)) 
            ext_values_gae = ext_values_storage.reshape((args.max_steps, num_games, args.num_agents))
            int_values_gae = int_values_storage.reshape((args.max_steps, num_games, args.num_agents))
            ext_rewards_gae = ext_rewards_storage.reshape(args.max_steps, num_games, args.num_agents)
            int_rewards_gae = int_rewards_storage.reshape(args.max_steps, num_games, args.num_agents)
            
            for agent_idx in range(args.num_agents):
               
            # === Advantage Calculation & Returns 
                lastgae_ext = 0
                lastgae_int = 0
                #This is not right because we n INDEPENDENT PARALLEL ENVS and thus we need to calculate the adv value for each agent separately per game basis
                for t in reversed(range(args.max_steps)):
                    if t == args.max_steps - 1:
                        ext_gt_next_state = ext_bootstrap  # same value for all agents
                        int_gt_next_state = int_bootstrap
                        nextnonterminal = (1.0 - next_done.reshape(num_games, args.num_agents)[:, agent_idx]) #its shared env so we can just take the first agent cus cooperative game
                        # print("Next nonterminal: ", nextnonterminal)
                        
                    else:
                        nextnonterminal = (1.0 - done_storage_gae[t + 1, :, agent_idx])
                        ext_gt_next_state = ext_values_gae[t + 1, :, agent_idx] * nextnonterminal
                        int_gt_next_state = int_values_gae[t + 1, :, agent_idx] * nextnonterminal

                    # Extrinsic
                    delta_ext = (ext_rewards_gae[t, :, agent_idx] + args.ext_gamma * ext_gt_next_state) - ext_values_gae[t, :, agent_idx]
                    ext_advantages[:, t, agent_idx] = lastgae_ext = delta_ext + args.GAE * lastgae_ext * nextnonterminal * args.ext_gamma
                    # Intrinsic
                    delta_int = (int_rewards_gae[t, :, agent_idx] + args.int_gamma * int_gt_next_state) - int_values_gae[t, :, agent_idx]
                    int_advantages[:, t, agent_idx] = lastgae_int = delta_int + args.GAE * lastgae_int * nextnonterminal * args.int_gamma

        # print(values_storage_gae.shape, advantages.shape, rewards_storage_gae.shape)
        ext_advantages = ext_advantages.permute(1, 0, 2)
        int_advantages = int_advantages.permute(1, 0, 2)
        combined_advantages = args.EXT_COEFF * ext_advantages + args.INT_COEFF * int_advantages

        ext_returns = ext_advantages + ext_values_gae
        int_returns = int_advantages + int_values_gae
        # Combined for logging/legacy code
        combined_values_gae = args.EXT_COEFF * ext_values_gae + args.INT_COEFF * int_values_gae
        returns = combined_advantages + combined_values_gae
        combined_values_storage = args.EXT_COEFF * ext_values_storage + args.INT_COEFF * int_values_storage
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # print("\n" + "="*50)
        # print(f"DEBUG: Shapes before flattening (Update #{update})")
        # print(f"{'obs_storage:':} {str(obs_storage.shape)}")
        # print(f"{'logprobs_storage:':} {str(logprobs_storage.shape)}")
        # print(f"{'actions_storage:':} {str(actions_storage.shape)}")
        # print(f"{'advantages:':} {str(advantages.shape)}")
        # print(f"{'returns:':} {str(returns.shape)}")
        # print(f"{'values_storage:':} {str(values_storage.shape)}")
        # print("="*50 + "\n")

        # === PPO Update Phase ===
        b_obs = obs_storage.reshape((-1, args.num_agents, ) +  single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1, args.num_agents)
        b_actions = actions_storage.reshape((-1, args.num_agents) + single_action_space.shape)
        b_advantages = combined_advantages.reshape(-1, args.num_agents)
        b_ext_returns = ext_returns.reshape(-1, args.num_agents)
        b_int_returns = int_returns.reshape(-1, args.num_agents)
        b_ext_values = ext_values_storage.reshape(-1, args.num_agents)
        b_int_values = int_values_storage.reshape(-1, args.num_agents)
        b_returns = returns.reshape(-1, args.num_agents)
        b_values = combined_values_storage.reshape(-1, args.num_agents)
        b_obs_next = next_obs_storage.reshape((-1, args.num_agents, ) + single_observation_space.shape)
        # print("\n" + "="*50)
        # print("DEBUG: Shapes of flattened 'b_' tensors")
        # print(f"{'b_obs:':} {str(b_obs.shape)}")
        # print(f"{'b_logprobs:':} {str(b_logprobs.shape)}")
        # print(f"{'b_actions:':} {str(b_actions.shape)}")
        # print(f"{'b_advantages:':} {str(b_advantages.shape)}")
        # print(f"{'b_returns:':} {str(b_returns.shape)}")
        # print(f"{'b_values:':} {str(b_values.shape)}")
        # print("="*50 + "\n")
        # masks = []
        # Create masks for each agent based on the next_obs

        # Total number of game-steps in our batch
        game_batch_size = num_games * args.max_steps
        # Minibatch size in terms of number of game-steps
        game_minibatch_size = game_batch_size // args.num_minibatches

        b_game_inds = np.arange(game_batch_size)

        # print(b_game_inds.shape)
        # b_inds = np.arange(args.batch_size)
        # Minibatch update
        for epoch in range(args.PPO_EPOCHS):
            np.random.shuffle(b_game_inds)
            for start in range(0, len(b_game_inds), game_minibatch_size):
                end = start + game_minibatch_size
                mb_inds = b_game_inds[start:end]
                # nice one by gemini to do the bird-eyes' view update of the CRITIC NETOWRK
                # in this loop
                # Get minibatch data
                global_state = b_obs[mb_inds].permute(0, 1, 4, 2, 3).reshape(mb_inds.shape[0], -1, 84, 84)
                
                # print("mb_obs shape: ", mb_obs.shape)
                mb_ext_returns = b_ext_returns[mb_inds, 0]
                mb_int_returns = b_int_returns[mb_inds, 0]
                mb_ext_values = b_ext_values[mb_inds, 0]
                mb_int_values = b_int_values[mb_inds, 0]
                current_ext, current_int = critic_network(global_state)
                current_ext = current_ext.squeeze()
                current_int = current_int.squeeze()
                # print("Current values: ", current_values)
                # print("mb_returns: ", mb_returns.shape, "mb_values: ", mb_values.shape, "current_values: ", current_values.shape)
                # --- Extrinsic value loss with clipping ---
                v_ext_unclipped = (current_ext - mb_ext_returns) ** 2
                v_ext_clipped_target = mb_ext_values + torch.clamp(
                    current_ext - mb_ext_values,
                    -args.clip_coeff,
                    args.clip_coeff,
                )
                v_ext_clipped = (v_ext_clipped_target - mb_ext_returns) ** 2
                v_loss_ext = torch.max(v_ext_unclipped, v_ext_clipped)
                # --- Intrinsic value loss with clipping ---
                v_int_unclipped = (current_int - mb_int_returns) ** 2
                v_int_clipped_target = mb_int_values + torch.clamp(
                    current_int - mb_int_values,
                    -args.clip_coeff,
                    args.clip_coeff,
                )
                v_int_clipped = (v_int_clipped_target - mb_int_returns) ** 2
                v_loss_int = torch.max(v_int_unclipped, v_int_clipped)

                critic_loss = args.VALUE_COEFF * 0.5 * (v_loss_ext.mean() + v_loss_int.mean())
                # print("V loss unclipped: ", v_loss_unclipped, "V loss clipped: ", v_loss_clipped)
                policy_loss_total = 0.0
                entropy_loss = 0.0
                b_adv_norm = b_advantages[mb_inds, 0]
                b_adv_norm = (b_adv_norm - b_adv_norm.mean()) / (b_adv_norm.std() + 1e-8)
                # print("Advantages: ", b_adv_norm)
                # print(mb_inds)
                # print("b_advantages: ", b_advantages.shape)
                for agent_idx in range(args.num_agents):
                    mb_obs = b_obs[mb_inds, agent_idx, ...]
                    mb_actions = b_actions[mb_inds, agent_idx]
                    mb_logprobs = b_logprobs[mb_inds, agent_idx]
                    # mb_advantages = b_adv_norm[mb_inds, agent_idx]
                    # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Calculate losses
                    new_log_probs, entropy = actor_network.evaluate_get_action(mb_obs, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_logprobs)

                    # print(new_log_probs, '    ', mb_logprobs)
                    # Policy loss
                    wandb.log({
                        "policy/old_logprobs": mb_logprobs.mean().item(),
                        "policy/new_logprobs": new_log_probs.mean().item(),
                        "policy/ratio": ratio.mean().item(),
                        
                    })
                    pg_loss1 = b_adv_norm * ratio
                    pg_loss2 = b_adv_norm * torch.clamp(ratio, 1 - args.clip_value, 1 + args.clip_value)
                    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    # print("Policy loss: ", policy_loss)
                    # print("Entropy: ", entropy.mean())
                    policy_loss_total += policy_loss

                    # Entropy loss
                    entropy_loss += entropy.mean()


                # Average the policy loss across agents CUS ITS A COOPERATIVE GAME AND WE NEED THE AGENTS TO WORK TOGETHER LEARN TOGETHER
                
                policy_loss_total /= args.num_agents
                entropy_loss /= args.num_agents

                # Total loss
                # RND predictor loss (use agent-0 observations for simplicity)
                mb_obs_all_agents = b_obs_next[mb_inds].reshape(-1, *single_observation_space.shape)
                pred_feat = predictor_network(mb_obs_all_agents)
                targ_feat = target_network(mb_obs_all_agents)
                intrinsic_loss = (pred_feat - targ_feat).pow(2).mean(1).mean()

                loss = policy_loss_total - args.ENTROPY_COEFF * entropy_loss + critic_loss + intrinsic_loss

                optimizer.zero_grad()
                # critic_optimizer.zero_grad()
                
                loss.backward()
                
                grad_norm_dict = {}
                total_norm = 0
                for name, param in actor_network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        if 'actor' in name or 'critic' in name:
                            grad_norm_dict[f"gradients/actor/norm_{name}"] = param_norm.item()
                        else:
                            grad_norm_dict[f"gradients/actor/shared_norm_{name}"] = param_norm.item()
                        total_norm += param_norm.item() ** 2
                grad_norm_dict["gradients/actor/total_norm"] = total_norm ** 0.5
                wandb.log(grad_norm_dict)

                for name, param in critic_network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        if 'actor' in name or 'critic' in name:
                            grad_norm_dict[f"gradients/critic/norm_{name}"] = param_norm.item()
                        else:
                            grad_norm_dict[f"gradients/critic/shared_norm_{name}"] = param_norm.item()
                        total_norm += param_norm.item() ** 2
                grad_norm_dict["gradients/critic/total_norm"] = total_norm ** 0.5
                wandb.log(grad_norm_dict)
                nn.utils.clip_grad_norm_(critic_network.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                optimizer.step()


            if args.use_wandb:
                wandb.log({ 
                    f"losses/total_loss_agent{agent_idx}": loss.item(),
                    f"losses/policy_loss_agent{agent_idx}": policy_loss_total,
                    f"losses/value_loss_agent{agent_idx}": critic_loss.item(),
                    f"losses/entropy_agent{agent_idx}": entropy_loss,
                    f"charts/learning_rate_agent{agent_idx}": optimizer.param_groups[0]['lr'],
                    f"charts/avg_ext_rewards_agent{agent_idx}": ext_rewards_storage.mean().item(),
                    f"charts/avg_int_rewards_agent{agent_idx}": int_rewards_storage.mean().item(),
                    f"charts/avg_ext_value_agent{agent_idx}": ext_values_storage.mean().item(),
                    f"charts/avg_int_value_agent{agent_idx}": int_values_storage.mean().item(),
                    f"charts/advantages_mean_agent{agent_idx}": b_advantages.mean().item(),
                    f"charts/advantages_std_agent{agent_idx}": b_advantages.std().item(),
                    # f"charts/returns_mean_agent{agent_idx}": b_returns.mean().item(),
                    "global_step": global_step,
                })
                # print(f"Update {update}, Global Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}")
        
        # critic_optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(critic_network.parameters(), 0.5)
        # critic_optimizer.step()
        
        if update % 50 == 0:
            rewards_player1, rewards_player2, avg_return1, avg_return2, _ = evaluate(actor_network, device, run_name, num_eval_eps=5, record=False)
            # Log the average return from the evaluation
            # avg_return = np.mean(episodic_returns)
            
            if args.use_wandb:
                wandb.log({
                    "eval/avg_return_player1": avg_return1,
                    "eval/avg_return_player2": avg_return2,
                    "global_step": global_step,
                })
            print("Rewards from evaluation:", rewards_player1, '   ', rewards_player2)
            print(f"Evaluation at step {global_step}: Average raw return for player 1  = {avg_return1:.2f}, Average raw return for player 2  = {avg_return2:.2f}")

        # Save the model at intervals of 200 updates
        if update % 200 == 0:
            # for idx, (actor, optim) in enumerate(zip(actor_network, optimizer)):
            save_checkpoint(actor_network, optimizer, num_updates, global_step, args, 'actor_network')

    if args.capture_video:
        print("Capturing final evaluation video...")
        _, _, _, _, eval_frames = evaluate(actor_network, device, run_name, num_eval_eps=10, record=True)

        if len(eval_frames) > 0:
            video_path = f"final_eval_{run_name}.mp4"
            # os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, eval_frames, fps=30, codec='libx264')
            if args.use_wandb:
                wandb.log({"eval/final_video": wandb.Video(video_path, fps=30, format="mp4")})
                print(f"Final evaluation video saved and uploaded to WandB.")


    save_checkpoint(actor_network, optimizer, num_updates, global_step, args, 'actor_network')
    envs.close()
    if args.use_wandb:
        wandb.finish()