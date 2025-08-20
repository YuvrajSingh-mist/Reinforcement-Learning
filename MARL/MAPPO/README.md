# Multi-Agent Proximal Policy Optimization (MAPPO)

<p align="center">
  <img src="images/simple_spread.mp4" width="400" alt="MAPPO Simple Spread Demo"/>
  <br>
  <em>MAPPO agents cooperating in Simple Spread environment</em>
</p>

## Overview

**Multi-Agent Proximal Policy Optimization (MAPPO)** is a centralized training with decentralized execution (CTDE) algorithm that extends PPO to multi-agent settings. MAPPO uses a centralized critic during training while maintaining decentralized policies for execution, making it highly effective for cooperative multi-agent tasks.

## Algorithm Theory

### Core Concept
MAPPO operates under the **Centralized Training, Decentralized Execution (CTDE)** paradigm, where agents share information during training but act independently during execution. This approach allows agents to leverage global information for better coordination while maintaining the benefits of decentralized execution.

### Key Components

#### 1. Centralized Training
- All agents share a centralized critic network
- Global state information is available during training
- Joint optimization of all agent policies

#### 2. Decentralized Execution
- Each agent has its own policy network
- Agents act based on local observations only
- No communication required during execution

#### 3. Proximal Policy Optimization
- Uses PPO's clipped objective function for stable updates
- Trust region optimization prevents large policy changes
- Entropy regularization encourages exploration

#### 4. Random Network Distillation (RND) Variants
- Intrinsic motivation for exploration
- Helps agents discover novel strategies
- Improves performance in complex environments

## Implementation Details

### Network Architecture

#### Centralized Critic
```python
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, num_agents):
        self.network = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )
```

#### Decentralized Actors
```python
class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        self.network = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(128, 64), std=0.01)
```

### Training Process

1. **Environment Interaction**
   - Multiple parallel environments (15 by default)
   - Agents interact using decentralized policies
   - Global state information is collected for critic

2. **Experience Collection**
   - Rollout length: 256 steps per environment (longer than IPPO)
   - Store local observations, actions, rewards, global states
   - Compute advantages using centralized critic

3. **Policy Updates**
   - PPO epochs: 10 (more than IPPO for better convergence)
   - Minibatch size: 3840 (15 envs × 256 steps)
   - Learning rate: 2.5e-4 with linear annealing

4. **Optimization**
   - Adam optimizer with gradient clipping (0.5)
   - Orthogonal initialization for stable training
   - Entropy coefficient: 0.02 for enhanced exploration

## Supported Environments

### 1. Simple Spread (Cooperative)
- **Environment**: `simple_spread_v3`
- **Task**: Cooperative navigation where agents must cover landmarks
- **Actions**: Discrete (5 actions per agent)
- **Observations**: Vector observations with agent positions
- **Global State**: Full environment state including all agent positions

### 2. Cooperative Pong (Butterfly)
- **Environment**: `cooperative_pong_v5`
- **Task**: Cooperative version of Pong where agents work together
- **Actions**: Discrete actions for paddle movement
- **Observations**: Image-based observations
- **Global State**: Full game state including ball and paddle positions

### 3. RND-Enhanced Environments
- **Purpose**: Improved exploration through intrinsic motivation
- **Implementation**: RND networks provide additional reward signals
- **Benefits**: Better performance in complex, sparse-reward environments

## Usage

### Installation
```bash
pip install torch pettingzoo[mpe,butterfly] supersuit wandb tqdm imageio opencv-python gymnasium
```

### Training Commands

#### Standard MAPPO (Simple Spread)
```bash
python mappo_without_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### MAPPO with RND
```bash
python mappo_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### MAPPO for Cooperative Pong
```bash
python mappo_rnd_pong.py --env_id cooperative_pong_v5 --total_timesteps 10000000
```

#### MAPPO Training Script
```bash
python train.py --env_id cooperative_pong_v5 --total_timesteps 10000000
```

### Key Hyperparameters

```python
# Training Configuration
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Parallel environments
max_steps = 256               # Rollout length (longer than IPPO)
PPO_EPOCHS = 10               # PPO update epochs (more than IPPO)
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.02          # Entropy regularization (higher than IPPO)
GAE = 0.95                    # GAE lambda parameter
total_timesteps = 20000000    # Total training steps
```

### Evaluation
```bash
# Evaluate trained model
python mappo_without_rnd.py --eval --checkpoint "checkpoint.pt"

# Interactive play
python play_ippo.py "checkpoint.pt"
```

## Technical Implementation

### File Structure
```
MAPPO/
├── mappo_without_rnd.py    # Standard MAPPO implementation
├── mappo_rnd.py           # MAPPO with RND for exploration
├── mappo_rnd_pong.py      # MAPPO with RND for cooperative Pong
├── train.py               # MAPPO training script
├── images/                # Training visualizations
│   └── simple_spread.mp4  # Demo video
└── README.md             # This file
```

### Key Classes

#### Config
Centralized configuration class containing all hyperparameters and training settings.

#### CentralizedCritic
Global value function that has access to the full environment state.

#### Actor Networks
Decentralized policy networks for each agent.

#### MAPPO Trainer
Main training loop implementing the MAPPO algorithm with centralized training.

## RND Integration

### Random Network Distillation
RND provides intrinsic motivation by measuring how "surprising" or "novel" an observation is:

```python
class RNDNetwork(nn.Module):
    def __init__(self, observation_dim):
        self.predictor = nn.Sequential(...)  # Predicts target features
        self.target = nn.Sequential(...)     # Fixed target network
```

## References

### Papers
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

### Code References
- [CleanRL MAPPO Implementation](https://github.com/vwxyzjn/cleanrl)
- [PettingZoo Multi-Agent Environments](https://pettingzoo.farama.org/)
- [SuperSuit Environment Wrappers](https://github.com/Farama-Foundation/SuperSuit)

---

## Contributing

This implementation is part of a larger MARL research project. Contributions are welcome in the form of:
- Bug reports and fixes
- Performance improvements
- New environment support
- Algorithm extensions

## License

This implementation is open source and available under the MIT License.
