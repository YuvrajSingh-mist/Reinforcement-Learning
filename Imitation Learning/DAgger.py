import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from tqdm import tqdm
import torch
import json
import wandb
from gridworld import GridWorld
import numpy as np
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("gridworld.json", "r") as f:
    json_object = json.load(f)
    env = GridWorld(**json_object, device=device)



import pickle
expert_data = pickle.load(open("/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/RL/Imitation Learning/imitation-learning-tutorials/expert_data/ckpt0.pkl", "rb"))
all_states = expert_data["states"]
all_actions = expert_data["actions"]
timestep_lens = expert_data["timestep_lens"]
# print(timestep_lens)
# print(all_states)
# print(all_actions)

@dataclass
class Config:
    lr: float = 2.5e-4
    project_name: str = "dagger"
    run_name: str = "dagger-gridworld"
    train_every: int = 20
    data_collection_steps: int = 10
    batch_size: int = 128
    eval_every: int = 20
    max_data_size: int = 20000

STATE_SIZE = env.grid_size * env.grid_size  # 2500
ACTION_SIZE = env.action_size               # 4

class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.output = nn.Linear(STATE_SIZE, ACTION_SIZE, bias=False)

    def forward(self, state, mask=None):
        """
        Set mask (size (TIMESTEP_SIZE, BATCH_SIZE)), when the number of timestep differs.
        """
        logits = self.output(state)
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            mask = mask.expand(-1, -1, ACTION_SIZE)
            logits = logits.masked_fill(mask, 0.0)
        return logits
    

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2500, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


def sample_action(logits, greedy=True):
    """Sample an action based on the given probabilities."""
    probs = torch.softmax(logits, dim=-1)
    # print(logits.shape)
    if greedy:
        return torch.argmax(probs, dim=-1, keepdim=True)
    else:
        return torch.distributions.Categorical(probs=probs).sample()


def one_hot_encode(obs, n_states=2500):
    """Convert integer observation to one-hot encoded vector"""
    if obs.dim() == 2 and obs.shape[1] == 1:
        obs = obs.squeeze(1) 
    encoded = torch.zeros((obs.shape[0], n_states), dtype=torch.float32, device=device)
    # print(encoded.shape, obs.shape)

    encoded.scatter_(1, obs.long().unsqueeze(1), 1.0)

    # print(encoded[0])
    return encoded

def decay(beta, step, p=0.999):
    """Decay function for beta."""
    return beta * (p ** step)  # Example decay function, adjust as needed
    
class BC:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.expert_net = ActorNet()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=Config.lr)
        self.beta_at_start = 1.0
        self.beta = self.beta_at_start

        # Initialize wandb
        wandb.init(
            project=Config.project_name,
            name=Config.run_name,
            config={
                "learning_rate": Config.lr,
                "architecture": "PolicyNet",
                "dataset": "expert_trajectories"
            }
        )
        self.load_expert_data()
        
    def load_expert_data(self):
        
        ckpt = torch.load("/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/RL/Imitation Learning/imitation-learning-tutorials/expert_actor.pt")
        self.expert_net.load_state_dict(ckpt)

    def train(self,  expert_states, expert_actions, policy_network=None):

        # for num_eps in range(eps):

        self.policy_net.train()
        self.policy_net.to(device)
        
        # Convert deque to tensors if needed
        if isinstance(expert_states, deque):
            expert_states = torch.cat(list(expert_states), dim=0).to(device)
            expert_actions = torch.cat(list(expert_actions), dim=0).to(device)
        else:
            expert_states = expert_states.to(device)
            expert_actions = expert_actions.to(device)
            
        expert_states_ohe = one_hot_encode(expert_states, n_states=2500)  # Assuming 2500 possible actions
    
        # Forward pass
        logits = self.policy_net(expert_states_ohe.float())
        # sampled_Action = sample_action(predicted_actions, greedy=False)
        # sampled_Action = sampled_Action.squeeze(1)  # Remove extra dimension
        # print(predicted_actions.shape, expert_actions.shape)
        # print(logits.shape, expert_actions.shape)
        # Compute loss
        # print(logits.shape, expert_actions.shape)
        loss = F.cross_entropy(logits, expert_actions.squeeze(1).squeeze(1))

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log loss to wandb
        wandb.log({"train_loss": loss.item()})

        return loss.item()
    
    def evaluate(self, batch_size=128, n_evals_eps = 5):
        self.policy_net.eval()
        
      
        reward_out = torch.tensor(0.0).to(device)
        for _ in tqdm(range(n_evals_eps)):
            obs = env.reset(batch_size)
            done = False
            total_reward = torch.tensor(0.0).to(device)
            while not done:
                with torch.no_grad():
                    expert_states_ohe = one_hot_encode(obs, n_states=2500)
                    predicted_actions = self.policy_net(expert_states_ohe.float())
                    sampled_Action = sample_action(predicted_actions, greedy=True)
                    # print(sampled_Action.shape, obs.shape)
                    # print(sampled_Action)
                    obs, reward, terminated, truncated = env.step(sampled_Action.squeeze(1), obs.clone().detach())

                    total_reward += reward.sum().item()
                    # total_reward /= batch_size
                    done = terminated.all() or truncated.all()

            reward_out += (total_reward / batch_size)

        return reward_out.sum().item() / n_evals_eps



model = BC()

# Use deque for efficient data collection with max size to prevent memory issues
collected_states = deque(maxlen=Config.max_data_size)
collected_actions = deque(maxlen=Config.max_data_size)
model.expert_net = model.expert_net.to(device)  # Ensure expert network is on the correct device
model.policy_net = model.policy_net.to(device)  # Ensure policy network is on the correct device    
#data using expert

print("Collecting data using expert network...")
for _ in tqdm(range(10000)):

    obs = env.reset(1)
    done = False
    # collected_states.append(obs)
    while not done:
        with torch.no_grad():
            expert_states_ohe = one_hot_encode(obs, n_states=2500)
            predicted_actions = model.expert_net(expert_states_ohe.float())
            sampled_Action = sample_action(predicted_actions, greedy=True)
            # print(sampled_Action.shape, obs.shape)
            new_obs, reward, terminated, truncated = env.step(sampled_Action.squeeze(1), obs.clone().detach())

            collected_states.append(obs)
            collected_actions.append(sampled_Action)

            done = terminated.all() or truncated.all()
            obs = new_obs


print(f"Collected {len(collected_states)} states and {len(collected_actions)} actions.")

# Keep the collected data as deque to maintain maxlen constraint
# Don't convert to tensors yet - keep them as deque for memory management
new_collected_states = deque(maxlen=10000)
new_collected_actions = deque(maxlen=10000)





for i in tqdm(range(3000)):

    if i % Config.train_every == 0:
        print(f"Training at step {i} with {len(collected_states)} collected states.")
        
        for _ in tqdm(range(4), desc='Training'): 
            # Shuffle the data before batching
            combined = list(zip(collected_states, collected_actions))
            random.shuffle(combined)
            shuffled_states, shuffled_actions = zip(*combined)
    
            for j in range(0, len(shuffled_states), Config.batch_size):
                states_batch = shuffled_states[j:j + Config.batch_size]
                states_batch = torch.stack(states_batch).to(device)
                actions_batch = shuffled_actions[j:j + Config.batch_size]
                actions_batch = torch.stack(actions_batch).to(device)
                model.train(states_batch, actions_batch)
            # model.train(collected_states, collected_actions)
    
    # if i % aggregate_every == 0:
    beta = decay(model.beta_at_start, i)
    # model.beta = beta
    wandb.log({"beta": beta, "step": i})
    
    if np.random.rand() < beta:
        
        # Use expert network for data collection
        print(f"Collecting data using expert network at step {i} with beta {beta}.")
        for _ in tqdm(range(Config.data_collection_steps)):
            obs = env.reset(1)
            done = False
            
            while not done:
                with torch.no_grad():
                    expert_states_ohe = one_hot_encode(obs, n_states=2500)
                    predicted_actions = model.expert_net(expert_states_ohe.float())
                    sampled_Action = sample_action(predicted_actions, greedy=True)
                    # print(sampled_Action)
                    new_obs, reward, terminated, truncated = env.step(sampled_Action.squeeze(1), obs.clone().detach())

                    new_collected_states.append(obs)
                    new_collected_actions.append(sampled_Action)

                    done = terminated.all() or truncated.all()
                    obs = new_obs
                    
        # Update collected states and actions
        if len(new_collected_states) > 0:
            new_collected_states_tensor = torch.cat(list(new_collected_states), dim=0).to(device)
            new_collected_actions_tensor = torch.cat(list(new_collected_actions), dim=0).to(device)
            
            # Add new data to the deques (will automatically manage maxlen)
            for state, action in zip(new_collected_states_tensor, new_collected_actions_tensor):
                collected_states.append(state.unsqueeze(0))
                collected_actions.append(action.unsqueeze(0))
        
        new_collected_states.clear()
        new_collected_actions.clear()
        
        
    else:
        print(f"Collecting data using policy network at step {i} with beta {beta}.")
        # Use policy network for data collection
        for _ in tqdm(range(Config.data_collection_steps)):
            obs = env.reset(1)
            done = False
            
            while not done:
                with torch.no_grad():
                    student_states_ohe = one_hot_encode(obs, n_states=2500)
                    predicted_actions = model.policy_net(student_states_ohe.float())
                    sampled_Action = sample_action(predicted_actions, greedy=True)

                    new_obs, reward, terminated, truncated = env.step(sampled_Action.squeeze(1), obs.clone().detach())

                    expert_actions = model.expert_net(student_states_ohe.float())
                    sampled_expert_action = sample_action(expert_actions, greedy=True)
                    new_collected_states.append(obs)
                    new_collected_actions.append(sampled_expert_action)

                    done = terminated.all() or truncated.all()
                    obs = new_obs
        # Update collected states and actions
        if len(new_collected_states) > 0:
            new_collected_states_tensor = torch.cat(list(new_collected_states), dim=0).to(device)
            new_collected_actions_tensor = torch.cat(list(new_collected_actions), dim=0).to(device)
            
            # Add new data to the deques (will automatically manage maxlen)
            for state, action in zip(new_collected_states_tensor, new_collected_actions_tensor):
                collected_states.append(state.unsqueeze(0))
                collected_actions.append(action.unsqueeze(0))
        
        new_collected_states.clear()
        new_collected_actions.clear()

    if i % Config.eval_every == 0:
        rew = model.evaluate()
        wandb.log({"eval_reward": rew, "episode": i})
        print(f"Episode {i}, Eval Reward: {rew}")
    # print(f"Episode {i}, Loss: {loss}")
    # curr += length
