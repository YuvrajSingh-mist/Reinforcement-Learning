import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from tqdm import tqdm
import torch
import json
import wandb
from gridworld import GridWorld

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
    project_name: str = "behavioral-cloning"
    run_name: str = "bc-gridworld"


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2500, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def sample_action(logits, greedy=True):
    """Sample an action based on the given probabilities."""
    probs = torch.softmax(logits, dim=-1)
    # print(logits.shape)
    if greedy:
        return torch.argmax(probs, dim=-1, keepdim=True)
    else:
        return torch.multinomial(probs, num_samples=1)
    
    
def one_hot_encode(obs, n_states=2500):
    """Convert integer observation to one-hot encoded vector"""
    encoded = torch.zeros((obs.shape[0], n_states), dtype=torch.float32, device=device)
    # print(encoded.shape, obs.shape)

    encoded.scatter_(1, torch.tensor(obs).long().unsqueeze(1), 1.0)

    # print(encoded[0])
    return encoded
    
    
class BC:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=Config.lr)
        
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

    def train(self,  expert_states, expert_actions, policy_network=None):

        # for num_eps in range(eps):

        self.policy_net.train()
        self.policy_net.to(device)
        expert_states = torch.tensor(expert_states).to(device)
        expert_actions = torch.tensor(expert_actions).to(device)
        expert_states_ohe = one_hot_encode(expert_states, n_states=2500)  # Assuming 2500 possible actions
    
        # Forward pass
        logits = self.policy_net(expert_states_ohe.float())
        # sampled_Action = sample_action(predicted_actions, greedy=False)
        # sampled_Action = sampled_Action.squeeze(1)  # Remove extra dimension
        # print(predicted_actions.shape, expert_actions.shape)
        # print(logits.shape, expert_actions.shape)
        # Compute loss
        # print(sampled_Action.shape, expert_actions.shape)
        loss = F.cross_entropy(logits, expert_actions)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log loss to wandb
        wandb.log({"train_loss": loss.item()})

        return loss.item()
    
    def evaluate(self, batch_size=128, n_evals_eps = 1):
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

curr = 0
for i, length in enumerate(timestep_lens):
    expert_states = all_states[curr: curr + length]
    expert_actions = all_actions[curr: curr + length]
    loss = model.train(expert_states, expert_actions)
    
    if i % 100 == 0:
        rew = model.evaluate()
        wandb.log({"eval_reward": rew, "episode": i})
        print(f"Episode {i}, Eval Reward: {rew}")
    # print(f"Episode {i}, Loss: {loss}")
    curr += length
