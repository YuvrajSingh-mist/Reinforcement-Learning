
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
import wandb


import imageio


# ===== CONFIGURATION =====
class Config:
    # Experiment settings
    exp_name = "TD3-Pendulum"
    seed = 42
    env_id = "Pendulum-v1"
    policy_noise= 0.3
    low = -2.0
    high = 2.0  # Action space limits for Pendulum
    # Training parameters
    total_timesteps = 2000000
    learning_rate = 3e-4
    buffer_size = 100000 
    gamma = 0.99
    tau = 0.005  # Soft update parameter for target networks
    target_network_frequency = 1
    batch_size = 256
    clip = 0.5
    exploration_fraction = 0.1
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

    



class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x =  torch.tanh(self.out(self.fc3(torch.nn.functional.mish(self.fc2(torch.nn.functional.mish(self.fc1(x)))))))
      
        # x = torch.nn.functional.softmax(x, dim=1)  # Apply softmax to get probabilities
        

        return x * 2.0
    
    # def get_action(self, x):
    #     action_probs = self.forward(x)
    #     dist = torch.distributions.Categorical(action_probs)  
    #     action = dist.sample() 
    #     return action, dist.log_prob(action) 
    
    
class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)  # Output a single Q-value
        self.out = nn.Linear(256, 1)
    def forward(self, state, act):
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


def make_env(env_id, seed, capture_video, run_name, eval_mode=False, render_mode=None):
    """Create environment with video recording"""
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
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


def evaluate(model, device, run_name, num_eval_eps = 10, record = False):
    eval_env = make_env(env_id=Config.env_id, seed=Config.seed, capture_video=True, run_name=run_name, eval_mode=True, render_mode='rgb_array')
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
                action = model(torch.tensor(obs, device=device).unsqueeze(0))
                action += torch.randn_like(action) * args.exploration_fraction # Add some noise for exploration
                action = torch.clip(action, args.low, args.high)  # Use args low and high
                action_numpy = action.cpu().numpy().flatten()  # Convert to numpy for environment
            obs, reward, terminated, truncated, _ = eval_env.step(action_numpy)
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



env = make_env(args.env_id, args.seed, args.capture_video, run_name)


actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

q1_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
q2_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

target_actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
target_q1_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
target_q2_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

target_q1_network.load_state_dict(q1_network.state_dict())
target_q2_network.load_state_dict(q2_network.state_dict())
target_actor_net.load_state_dict(actor_net.state_dict())


actor_optim = optim.Adam(actor_net.parameters(), lr=args.learning_rate)
q1_optim = optim.Adam(q1_network.parameters(), lr=args.learning_rate)
q2_optim = optim.Adam(q2_network.parameters(), lr=args.learning_rate)

# eps_decay = LinearEpsilonDecay(args.start_e, args.end_e, args.total_timesteps)

q1_network.train()
q2_network.train()
actor_net.train()


replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)


obs,  _ = env.reset()
start_time = time.time()



for step in tqdm(range(args.total_timesteps)):
    
    # eps = eps_decay(step, args.exploration_fraction)
    # rnd = random.random()
    # if rnd < eps:
    #     action = env.action_space.sample()
    # else:
    # Get action from actor network
    with torch.no_grad():  # No need to track gradients for environment interactions
        action = actor_net(torch.tensor(obs, device=device).unsqueeze(0))
        noise = torch.clip(torch.randn_like(action) * args.exploration_fraction, -args.clip, args.clip) # Add some noise for exploration
        action += noise
        action = torch.clip(action, args.low, args.high)
    # Convert to numpy for environment step (no gradients needed here)
    action = action.cpu().numpy().flatten()  # Convert to numpy array and flatten
    new_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay_buffer.add(obs, new_obs, action, np.array(reward), np.array(done), [info])

     # Log episode returns
    if "episode" in info:
        print(f"Step={step}, Return={info['episode']['r']}")
        
        # WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return": info['episode']['r'],
                "episodic_length": info['episode']['l'],
                "action": action,  # Use action_numpy which is already a numpy array
                "global_step": step
            })
    if step > args.learning_starts:
        data = replay_buffer.sample(args.batch_size)

        # Q(s t ​ ,a t ​ )←Q(s t ​ ,a t ​ )+α ​ TD target r t ​ +γ a ′ max ​ Q(s t+1 ​ ,a ′ ) ​ ​ −Q(s t ​ ,a t ​ ) ​
        # with torch.no_grad():
        
        # returns = []
        
        # bootstrap = 0
        
        # if data.dones.flatten().tolist()[-1] == 0:
            
        #     bootstrap = target_q_network(data.next_observations).max(1)[0]  # dim=1
            
        # gt_next_state = bootstrap
        # for reward_at_t, done_at_t in zip(reversed(data.rewards.flatten().tolist()), reversed(data.dones.flatten().tolist())):
        #     if done_at_t:
        #         gt_next_state = 0.0
        #     rt = reward_at_t + args.gamma * gt_next_state
        #     returns.insert(0, rt)
        #     gt_next_state = rt

        # actor_optim.zero_grad()
        # returns = torch.tensor(returns, device=device, dtype=torch.float32)
        # actions = actor_net(data.observations)
        # action_values = -q1_network(data.observations, actions).mean()  # Maximize Q-value for next actions
        # action_values.backward()
        # actor_optim.step()

        #why min of target q nets? well bruh thats cus the max op is being on them (kinda)! so we have to lower the overestimation of them!!
        with torch.no_grad():
            next_actions = target_actor_net(data.next_observations)
            noise = torch.clip(torch.randn_like(next_actions) * args.policy_noise, -args.clip, args.clip)  # Add some noise for exploration
            next_actions += noise
            next_actions = torch.clip(next_actions, args.low, args.high)  # Use args low and high
            target_max1 = target_q1_network(data.next_observations, next_actions)
            target_max2 = target_q2_network(data.next_observations, next_actions)
            target_max = torch.min(target_max1, target_max2)
            td_target = data.rewards + args.gamma * target_max * (1 - data.dones)

        old_val1 = q1_network(data.observations, data.actions)
        old_val2 = q2_network(data.observations, data.actions)
        q1_optim.zero_grad()
        loss1 = nn.functional.mse_loss(old_val1, td_target)
        loss2 = nn.functional.mse_loss(old_val2, td_target)

        # temp = loss1 + loss2
        # q1_optim.zero_grad()
        loss1.backward(retain_graph=True)
        q1_optim.step()
        q2_optim.zero_grad()
        loss2.backward()
        q2_optim.step()
        
        
        if step % args.train_frequency == 0:
            
            actions = actor_net(data.observations)
            action_values = -q1_network(data.observations, actions).mean()  # Maximize Q-value for next actions
            actor_optim.zero_grad()
            action_values.backward()
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
            for q_params, target_params  in zip(q1_network.parameters(), target_q1_network.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)

            for actor_params, target_actor_params in zip(actor_net.parameters(), target_actor_net.parameters()):
                target_actor_params.data.copy_(args.tau * actor_params.data + (1.0 - args.tau) * target_actor_params.data)

            for q_params, target_params in zip(q2_network.parameters(), target_q2_network.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
                
            # ===== MODEL EVALUATION & SAVING =====
    if step % 500 == 0:
        # Save model
        # model_path = f"runs/{run_name}/model_{step}.pth"
        # torch.save(q_network.state_dict(), model_path)
        # print(f"Model saved to {model_path}")
        
        # Evaluate model
        episodic_returns, eval_frames = evaluate(actor_net, device, run_name)
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
if args.use_wandb:
    train_video_path = f"videos/final.mp4"
    returns, frames = evaluate(actor_net, device, run_name, record=True)
    # if os.path.exists(train_video_path) and os.listdir(train_video_path):
        # wandb.log({"train_video": wandb.Video(f"{train_video_path}/rl-video-episode-0.mp4")})
    imageio.mimsave(train_video_path, frames, fps=30, codec='libx264')
    print(f"Final training video saved to {train_video_path}")
    wandb.finish()
