# Independent Proximal Policy Optimization (IPPO)

<p align="center">
  <img src="images/pong.gif" width="400" alt="IPPO Pong Demo"/>
  <br>
  <em>IPPO agents competing in Pong environment</em>
</p>

## Overview

**Independent Proximal Policy Optimization (IPPO)** is a state-of-the-art multi-agent reinforcement learning algorithm that extends the single-agent PPO algorithm to multi-agent settings. Unlike centralized training approaches, IPPO allows each agent to learn independently while sharing observation processing capabilities.

## Algorithm Theory

### Core Concept
IPPO operates on the principle that each agent can learn an optimal policy independently while sharing a common observation encoder. This approach is particularly effective in cooperative multi-agent environments where agents need to coordinate but can benefit from independent learning.

### Key Components

#### 1. Independent Learning
- Each agent maintains its own policy network (actor) and value network (critic)
- Agents learn independently without direct policy sharing
- Shared observation processing reduces computational overhead

#### 2. Proximal Policy Optimization
- Uses PPO's clipped objective function to ensure stable policy updates
- Trust region optimization prevents large policy changes
- Entropy regularization encourages exploration

#### 3. Generalized Advantage Estimation (GAE)
- Computes advantages using GAE with λ=0.95
- Reduces variance in policy gradient estimates
- Balances bias-variance trade-off in advantage estimation

## Implementation Details

### Network Architecture

#### Shared Observation Encoder
```python
class SharedEncoder(nn.Module):
    def __init__(self, observation_dim):
        self.network = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
```

#### Agent-Specific Heads
```python
class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        # Shared feature extraction
        self.network = nn.Sequential(...)
        # Agent-specific actor head
        self.actor = layer_init(nn.Linear(64, action_dim), std=0.01)

class Critic(nn.Module):
    def __init__(self, observation_dim):
        # Shared feature extraction
        self.network = nn.Sequential(...)
        # Agent-specific critic head
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
```

### Training Process

1. **Environment Interaction**
   - Multiple parallel environments (15 by default)
   - Each agent interacts independently
   - Observations are processed through shared encoder

2. **Experience Collection**
   - Rollout length: 128 steps per environment
   - Store observations, actions, rewards, values, log probabilities
   - Compute advantages using GAE

3. **Policy Updates**
   - PPO epochs: 4
   - Minibatch size: 1920 (15 envs × 128 steps)
   - Learning rate: 2.5e-4 with linear annealing

4. **Optimization**
   - Adam optimizer with gradient clipping (0.5)
   - Orthogonal initialization for stable training
   - Entropy coefficient: 0.001 for exploration

## Supported Environments

### 1. Simple Spread (Discrete Actions)
- **Environment**: `simple_spread_v3`
- **Task**: Cooperative navigation where agents must cover landmarks
- **Actions**: Discrete (5 actions per agent)
- **Observations**: Vector observations with agent positions and landmark locations
- **Reward**: Cooperative reward based on landmark coverage

### 2. Simple Tag (Competitive)
- **Environment**: `simple_tag_v3`
- **Task**: Competitive tagging game
- **Actions**: Discrete actions for movement and tagging
- **Observations**: Vector observations with agent positions
- **Reward**: Competitive rewards for taggers and runners

### 3. Continuous Control
- **Environment**: `simple_spread_v3` (continuous variant)
- **Task**: Same cooperative navigation with continuous actions
- **Actions**: Continuous 2D movement vectors
- **Observations**: Same vector observations
- **Reward**: Same cooperative reward structure

## Usage

### Installation
```bash
pip install torch pettingzoo[mpe] supersuit wandb tqdm imageio opencv-python gymnasium
```

### Training Commands

#### Discrete Actions (Simple Spread)
```bash
python ippo_discrete.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### Continuous Actions
```bash
python ippo_continuous.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### Simple Tag Environment
```bash
python ippo_simple_tag.py --env_id simple_tag_v3 --total_timesteps 20000000
```

### Key Hyperparameters

```python
# Training Configuration
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.001         # Entropy regularization
GAE = 0.95                    # GAE lambda parameter
total_timesteps = 20000000    # Total training steps
```

### Evaluation
```bash
# Evaluate trained model
python ippo_discrete.py --eval --checkpoint "checkpoint.pt"

# Interactive play (Pong)
python play_ippo.py "checkpoint.pt"
```

## Technical Implementation

### File Structure
```
IPPO/
├── ippo_discrete.py      # Discrete action implementation
├── ippo_continuous.py    # Continuous action implementation
├── ippo_simple_tag.py    # Simple Tag environment
├── play_ippo.py         # Interactive play script
├── images/              # Training visualizations
│   ├── pong.gif        # Demo video
│   └── image.png       # Training plots
└── README.md           # This file
```

### Key Classes

#### Config
Centralized configuration class containing all hyperparameters and training settings.

#### Actor/Critic Networks
Agent-specific policy and value networks with shared observation processing.

#### IPPO Trainer
Main training loop implementing the IPPO algorithm with experience collection and policy updates.

## References

### Papers
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

### Code References
- [CleanRL IPPO Implementation](https://github.com/vwxyzjn/cleanrl)
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
