# Self-Play Multi-Agent Reinforcement Learning

<p align="center">
  <img src="../IPPO/images/pong.gif" width="400" alt="Self-Play Pong Demo"/>
  <br>
  <em>Self-play agents competing in Pong environment</em>
</p>

## Overview

**Self-Play** is a powerful training paradigm in multi-agent reinforcement learning where agents learn by competing against themselves or other agents from the same population. This approach has been instrumental in achieving superhuman performance in games like Go, Chess, and Dota 2. Our implementation focuses on competitive environments like Pong, where agents learn optimal strategies through continuous self-improvement.

## Self-Play Theory

### Core Concept
Self-play operates on the principle that an agent can improve by playing against increasingly skilled versions of itself. This creates a natural curriculum where the agent's opponent (itself) becomes progressively stronger, forcing continuous improvement.

### Key Mechanisms

#### 1. Population-Based Learning
- Multiple agents form a population
- Agents compete against each other
- Best strategies are preserved and improved

#### 2. Opponent Sampling
- Agents play against current and historical versions
- Prevents overfitting to a single opponent
- Maintains diverse strategy exploration

#### 3. Strategy Evolution
- Successful strategies are reinforced
- Novel strategies emerge through exploration
- Continuou
## Implementation Details

### Network Architecture

#### Shared Policy Network
```python
class Agent(nn.Module):
    def __init__(self, action_space):
        # Shared CNN feature extractor
        self.network = nn.Sequential(
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
        # Actor and Critic heads
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
```

#### Observation Processing
- **Input**: 6-channel observation (4-frame stack + agent indicator)
- **Preprocessing**: Grayscale, resize to 84×84, frame stacking
- **Output**: Action probabilities and state value

### Training Process

1. **Environment Setup**
   - PettingZoo Atari Pong-v3 environment
   - Two agents compete in each episode
   - Shared policy network for both agents

2. **Experience Collection**
   - 16 parallel environments
   - 128 steps per rollout
   - Store observations, actions, rewards, values

3. **Self-Play Training**
   - Agents compete against each other
   - Winner gets positive reward, loser gets negative
   - Policy updates based on competitive outcomes

4. **Opponent Management**
   - Current policy serves as opponent
   - Historical policies can be used for diversity
   - Prevents overfitting to current strategy

## Supported Environments

### 1. Pong (Atari)
- **Environment**: `pong_v3`
- **Task**: Competitive Pong game
- **Actions**: 6 discrete actions (NOOP, FIRE, RIGHT, LEFT, FIRE_RIGHT, FIRE_LEFT)
- **Observations**: 6-channel image (4-frame stack + agent indicator)
- **Reward**: +1 for winning, -1 for losing

### 2. Custom Competitive Environments
- **Extensible**: Framework supports other competitive games
- **Modular**: Easy to adapt to new environments
- **Scalable**: Supports multiple agents and teams

## Usage

### Installation
```bash
pip install torch pettingzoo[atari] supersuit wandb tqdm imageio opencv-python gymnasium
```

### Training Commands

#### Main Self-Play Training
```bash
cd MARL
python train.py --env_id pong_v3 --total_timesteps 15000000
```

#### Alternative Self-Play Driver
```bash
cd MARL/Self Play
python self_play.py --env_id pong_v3 --total_timesteps 15000000
```

### Key Hyperparameters

```python
# Self-Play Configuration
lr = 2.5e-4                    # Learning rate
num_envs = 16                  # Parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.1              # PPO clipping coefficient
ENTROPY_COEFF = 0.01          # Entropy regularization
total_timesteps = 15000000    # Total training steps
```

### Interactive Play

#### Human vs AI
```bash
python play.py "pt files/Pong-MARL.pt"
```

**Controls:**
- `W` or `↑`: Move right
- `S` or `↓`: Move left
- `F`: Fire
- `D`: Fire right
- `A`: Fire left
- `Q`: Quit

#### AI vs AI
```bash
python play.py "pt files/Pong-MARL.pt" --ai_vs_ai
```

## Results and Performance

### Training Metrics
- **Convergence**: Typically converges within 10-15M timesteps
- **Win Rate**: Agents achieve >90% win rate against random opponents
- **Strategy Evolution**: Emergence of sophisticated playing strategies

### Emergent Behaviors

#### 1. Defensive Strategies
- Agents learn to position paddles optimally
- Effective blocking of opponent shots
- Strategic use of paddle movement

#### 2. Offensive Strategies
- Agents develop sophisticated shot patterns
- Use of angles and speed variations
- Exploitation of opponent weaknesses

#### 3. Adaptive Play
- Agents adapt to opponent strategies
- Counter-strategies emerge naturally
- Continuous improvement through competition

## Advantages of Self-Play

### 1. Automatic Curriculum
- Difficulty increases naturally with agent improvement
- No manual curriculum design required
- Optimal learning progression

### 2. Strategy Discovery
- Novel strategies emerge through exploration
- Agents discover optimal play patterns
- No human expertise required

### 3. Robustness
- Agents learn to handle diverse opponents
- Strategies generalize well
- Robust to different playing styles

### 4. Scalability
- Works with any number of agents
- Easy to extend to new environments
- Minimal human intervention required

## Technical Implementation

### File Structure
```
Self Play/
├── play.py              # Interactive play script
├── self_play.py         # Self-play training driver
├── pt files/           # Saved checkpoints
│   └── Pong-MARL.pt   # Pre-trained model
└── README.md          # This file
```

### Key Classes

#### Agent
Shared policy network used by both competing agents.

#### SelfPlayTrainer
Main training loop implementing self-play with PPO updates.

#### PlayEnvironment
Interactive environment for human vs AI and AI vs AI gameplay.

## Pre-trained Models

### Pong-MARL.pt
- **Training**: 15M timesteps of self-play training
- **Performance**: >90% win rate against random opponents
- **Usage**: Ready for immediate evaluation and interactive play
- **Size**: ~19MB

### Loading Pre-trained Models
```python
import torch
agent = Agent(action_space)
checkpoint = torch.load("pt files/Pong-MARL.pt")
agent.load_state_dict(checkpoint['model_state_dict'])
```

## Comparison with Other Approaches

| Aspect | Self-Play | Supervised Learning | Imitation Learning |
|--------|-----------|-------------------|-------------------|
| Data Requirements | None | Human demonstrations | Human demonstrations |
| Strategy Discovery | Automatic | Limited | Limited |
| Scalability | High | Medium | Medium |
| Performance | Excellent | Good | Good |
| Implementation | Simple | Complex | Complex |

## Future Work

### Potential Improvements
1. **Population-Based Training**: Multiple agent populations
2. **Meta-Learning**: Fast adaptation to new opponents
3. **Hierarchical Policies**: Multi-level strategy learning
4. **Communication**: Adding explicit communication channels

### Research Directions
1. **Multi-Agent Self-Play**: Teams of agents competing
2. **Transfer Learning**: Cross-game knowledge transfer
3. **Adversarial Training**: Improving robustness
4. **Scalable Architectures**: Handling larger games

## References

### Papers
- [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)

### Code References
- [PettingZoo Atari Environments](https://pettingzoo.farama.org/environments/atari/)
- [SuperSuit Environment Wrappers](https://github.com/Farama-Foundation/SuperSuit)
- [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl)

---

## Contributing

This implementation is part of a larger MARL research project. Contributions are welcome in the form of:
- Bug reports and fixes
- Performance improvements
- New environment support
- Algorithm extensions

## License

This implementation is open source and available under the MIT License.
