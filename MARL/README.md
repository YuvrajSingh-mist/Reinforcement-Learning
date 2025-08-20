# Multi-Agent Reinforcement Learning (MARL) Research Project

<p align="center">
  <img src="IPPO/images/pong.gif" width="300" alt="IPPO Pong Demo"/>
  <img src="MAPPO/images/simple_spread.mp4" width="300" alt="MAPPO Simple Spread Demo"/>
  <br>
  <em>IPPO agents competing in Pong (left) and MAPPO agents cooperating in Simple Spread (right)</em>
</p>

## 🚀 Project Overview

This comprehensive **Multi-Agent Reinforcement Learning (MARL)** research project implements and evaluates state-of-the-art algorithms for multi-agent systems. The project features **IPPO** (Independent Proximal Policy Optimization), **MAPPO** (Multi-Agent Proximal Policy Optimization), and **Self-Play** implementations, supporting both cooperative and competitive multi-agent scenarios.

### 🎯 Key Features

- **Multiple Algorithms**: IPPO, MAPPO, and Self-Play implementations
- **Diverse Environments**: Atari, PettingZoo MPE, and Butterfly environments
- **Action Spaces**: Support for both discrete and continuous actions
- **Exploration**: RND (Random Network Distillation) integration
- **Interactive Play**: Human vs AI and AI vs AI gameplay
- **Pre-trained Models**: Ready-to-use trained agents
- **Comprehensive Documentation**: Detailed READMEs for each algorithm

---

## 📚 Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Project Structure](#project-structure)
3. [Supported Environments](#supported-environments)
4. [Quick Start Guide](#quick-start-guide)
5. [Algorithm-Specific Guides](#algorithm-specific-guides)
6. [Training Examples](#training-examples)
7. [Results and Performance](#results-and-performance)
8. [Technical Details](#technical-details)
9. [Contributing](#contributing)
10. [References](#references)

---

## 🧠 Algorithm Overview

### IPPO (Independent Proximal Policy Optimization)
**Location**: [`IPPO/`](IPPO/README.md)

IPPO extends single-agent PPO to multi-agent settings through independent learning with shared observation processing. Each agent maintains its own policy while benefiting from shared feature extraction.

**Key Features:**
- Independent learning for each agent
- Shared observation encoder
- Support for discrete and continuous actions
- Self-play capabilities for competitive environments

**Best For:** Cooperative tasks requiring independent decision-making, competitive scenarios, scalable multi-agent systems.

### MAPPO (Multi-Agent Proximal Policy Optimization)
**Location**: [`MAPPO/`](MAPPO/README.md)

MAPPO implements centralized training with decentralized execution (CTDE), using a centralized critic during training while maintaining decentralized policies for execution.

**Key Features:**
- Centralized training with decentralized execution
- Global state information during training
- RND variants for enhanced exploration
- Superior coordination in cooperative tasks

**Best For:** Cooperative multi-agent tasks, scenarios requiring coordination, complex environments with global state information.

### Self-Play
**Location**: [`Self Play/`](Self%20Play/README.md)

Self-play training where agents learn by competing against themselves or other agents from the same population, creating a natural curriculum for continuous improvement.

**Key Features:**
- Population-based learning
- Automatic curriculum generation
- Strategy evolution through competition
- Interactive human vs AI gameplay

**Best For:** Competitive environments, strategy games, scenarios requiring emergent behavior discovery.

---

## 📁 Project Structure

```
MARL/
├── README.md                 # Main project documentation (this file)
├── train.py                  # Main training script for Pong self-play
├── play_ippo.py             # Play script for trained models
│
├── IPPO/                    # Independent PPO implementations
│   ├── README.md           # Detailed IPPO documentation
│   ├── ippo_discrete.py    # Discrete action spaces (Simple Spread)
│   ├── ippo_continuous.py  # Continuous action spaces
│   ├── ippo_simple_tag.py  # Simple Tag environment
│   ├── play_ippo.py        # Interactive play script (Pong)
│   ├── images/             # Training visualizations
│   │   ├── pong.gif       # Demo video
│   │   └── image.png      # Training plots
│   └── *.mp4              # Demo videos
│
├── MAPPO/                   # Multi-Agent PPO implementations
│   ├── README.md          # Detailed MAPPO documentation
│   ├── mappo_without_rnd.py    # Standard MAPPO
│   ├── mappo_rnd.py           # MAPPO with RND for exploration
│   ├── mappo_rnd_pong.py      # MAPPO with RND for cooperative Pong
│   ├── train.py               # MAPPO training script (cooperative Pong)
│   ├── images/                # Training visualizations
│   │   └── simple_spread.mp4  # Demo video
│   └── __pycache__/
│
└── Self Play/               # Self-play utilities
    ├── README.md           # Detailed Self-Play documentation
    ├── play.py             # Watch two trained agents compete (Pong)
    ├── self_play.py        # Self-play training driver (Pong)
    └── pt files/           # Saved checkpoints
        └── Pong-MARL.pt    # Pre-trained Pong model (19MB)
```

---

## 🌍 Supported Environments

### Atari Environments
- **Pong-v3**: Classic Atari Pong with self-play capabilities
  - **Features**: Image-based observations, discrete actions, competitive gameplay
  - **Use Cases**: Self-play training, competitive scenarios

### PettingZoo MPE Environments
- **Simple Spread**: Cooperative navigation task
  - **Features**: Vector observations, discrete/continuous actions, cooperative rewards
  - **Use Cases**: IPPO and MAPPO training, coordination studies
- **Simple Tag**: Competitive tagging game
  - **Features**: Vector observations, competitive rewards
  - **Use Cases**: Competitive multi-agent scenarios

### PettingZoo Butterfly Environments
- **Cooperative Pong-v5**: Cooperative version of Pong for MAPPO
  - **Features**: Multi-agent cooperation, image-based observations
  - **Use Cases**: Cooperative training, coordination studies

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Install all dependencies
pip install torch pettingzoo[atari,mpe,butterfly] supersuit wandb tqdm imageio opencv-python gymnasium
```

### 2. Choose Your Algorithm

#### For Cooperative Tasks (IPPO)
```bash
cd MARL/IPPO
python ippo_discrete.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### For Cooperative Tasks with Coordination (MAPPO)
```bash
cd MARL/MAPPO
python mappo_without_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000
```

#### For Competitive Self-Play (Pong)
```bash
cd MARL
python train.py --env_id pong_v3 --total_timesteps 15000000
```

### 3. Interactive Play

#### Human vs AI (Pong)
```bash
cd MARL/Self Play
python play.py "pt files/Pong-MARL.pt"
```

#### AI vs AI
```bash
cd MARL/IPPO
python play_ippo.py "checkpoint.pt"
```

---

## 📖 Algorithm-Specific Guides

### [IPPO Documentation](IPPO/README.md)
- **Theory**: Independent learning with shared observation processing
- **Implementation**: Discrete, continuous, and Simple Tag variants
- **Usage**: Training commands, hyperparameters, evaluation
- **Results**: Performance metrics and emergent behaviors

### [MAPPO Documentation](MAPPO/README.md)
- **Theory**: Centralized training with decentralized execution
- **Implementation**: Standard MAPPO and RND variants
- **Usage**: Training commands, hyperparameters, evaluation
- **Results**: Coordination performance and sample efficiency

### [Self-Play Documentation](Self%20Play/README.md)
- **Theory**: Population-based learning and strategy evolution
- **Implementation**: Competitive training and interactive play
- **Usage**: Training commands, interactive controls, evaluation
- **Results**: Strategy emergence and competitive performance

---

## 🎯 Training Examples

### IPPO Training Commands
```bash
# Discrete actions (Simple Spread)
python IPPO/ippo_discrete.py --env_id simple_spread_v3 --total_timesteps 20000000

# Continuous actions
python IPPO/ippo_continuous.py --env_id simple_spread_v3 --total_timesteps 20000000

# Simple Tag environment
python IPPO/ippo_simple_tag.py --env_id simple_tag_v3 --total_timesteps 20000000
```

### MAPPO Training Commands
```bash
# Standard MAPPO (Simple Spread)
python MAPPO/mappo_without_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000

# MAPPO with RND for exploration
python MAPPO/mappo_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000

# MAPPO for cooperative Pong
python MAPPO/mappo_rnd_pong.py --env_id cooperative_pong_v5 --total_timesteps 10000000
```

### Self-Play Training Commands
```bash
# Main self-play training (Pong)
python train.py --env_id pong_v3 --total_timesteps 15000000

# Alternative self-play driver
python "Self Play/self_play.py" --env_id pong_v3 --total_timesteps 15000000
```

---

## 📊 Results and Performance

### Algorithm Comparison

| Aspect | IPPO | MAPPO | Self-Play |
|--------|------|-------|-----------|
| **Training Paradigm** | Independent | Centralized | Population-based |
| **Sample Efficiency** | High | Very High | Medium |
| **Coordination** | Good | Excellent | N/A |
| **Scalability** | High | Medium | High |
| **Implementation** | Simple | Complex | Simple |
| **Best For** | Cooperative/Competitive | Cooperative | Competitive |

### Environment-Specific Performance

#### Simple Spread (Cooperative)
- **IPPO**: Achieves 85-90% landmark coverage
- **MAPPO**: Achieves 95-98% landmark coverage
- **Convergence**: 10-20M timesteps

#### Pong (Competitive)
- **Self-Play**: >90% win rate against random opponents
- **Strategy Emergence**: Sophisticated defensive and offensive strategies
- **Convergence**: 10-15M timesteps

#### Simple Tag (Competitive)
- **IPPO**: Effective competitive strategies
- **Balance**: Maintains competitive balance between teams
- **Adaptation**: Agents adapt to opponent strategies

---

## 🔧 Technical Details

### Hyperparameters

#### IPPO Configuration
```python
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.001         # Entropy regularization
GAE = 0.95                    # GAE lambda parameter
```

#### MAPPO Configuration
```python
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Parallel environments
max_steps = 256               # Rollout length (longer than IPPO)
PPO_EPOCHS = 10               # PPO update epochs (more than IPPO)
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.02          # Entropy regularization (higher than IPPO)
GAE = 0.95                    # GAE lambda parameter
```

#### Self-Play Configuration
```python
lr = 2.5e-4                    # Learning rate
num_envs = 16                  # Parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.1              # PPO clipping coefficient
ENTROPY_COEFF = 0.01          # Entropy regularization
total_timesteps = 15000000    # Total training steps
```

### Network Architectures

#### Observation Processing
- **Atari**: Grayscale, resize to 84×84, 4-frame stack, agent indicator channel, downsampled to 64×64
- **MPE**: Direct vector observations with agent-specific processing
- **Butterfly**: Image-based observations with multi-agent coordination

#### Shared Components
- **Shared Encoder**: Convolutional tower for images, MLP for vectors
- **Agent-Specific Heads**: Separate actor and critic networks per agent
- **Optimization**: Adam with gradient clipping (0.5) + orthogonal initialization

### Pre-trained Models

#### Pong-MARL.pt
- **Location**: `Self Play/pt files/Pong-MARL.pt`
- **Training**: 15M timesteps of self-play training
- **Performance**: >90% win rate against random opponents
- **Size**: ~19MB
- **Usage**: Ready for immediate evaluation and interactive play

---

## 🎮 Interactive Features

### Human vs AI Gameplay
- **Controls**: Keyboard-based interaction
- **Visualization**: Real-time rendering with OpenCV
- **Feedback**: Immediate visual and score feedback

### AI vs AI Competition
- **Visualization**: Real-time agent competition
- **Analysis**: Strategy observation and analysis
- **Recording**: Video capture for analysis

### Evaluation Tools
- **Metrics**: Win rates, cooperation scores, efficiency measures
- **Visualization**: Training curves, performance plots
- **Comparison**: Cross-algorithm performance analysis

---

## 🔬 Research Contributions

### Novel Implementations
1. **IPPO Variants**: Discrete, continuous, and competitive implementations
2. **MAPPO with RND**: Enhanced exploration for cooperative tasks
3. **Self-Play Framework**: Comprehensive competitive training system

### Technical Innovations
1. **Shared Observation Processing**: Efficient feature extraction
2. **RND Integration**: Intrinsic motivation for exploration
3. **Interactive Play**: Human-AI interaction capabilities

### Performance Improvements
1. **Sample Efficiency**: Optimized training procedures
2. **Stability**: Robust training across environments
3. **Scalability**: Efficient multi-agent implementations

---

## 🚀 Future Work

### Algorithm Extensions
1. **Attention Mechanisms**: Improving observation processing
2. **Hierarchical Policies**: Multi-level decision making
3. **Communication Protocols**: Explicit agent communication
4. **Meta-Learning**: Fast adaptation to new environments

### Environment Support
1. **New PettingZoo Environments**: Additional multi-agent scenarios
2. **Custom Environments**: Domain-specific applications
3. **Real-world Applications**: Robotics, autonomous systems

### Research Directions
1. **Multi-Objective Optimization**: Balancing multiple objectives
2. **Transfer Learning**: Cross-environment knowledge transfer
3. **Adversarial Training**: Improving robustness
4. **Scalable Architectures**: Handling larger numbers of agents

---

## 📚 References

### Key Papers
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961)

### Libraries and Tools
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environment library
- [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) - Environment preprocessing
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Reference implementations

### WandB Reports
- [![WandB Report](https://img.shields.io/badge/WandB-Report-blue?logo=wandb)](https://api.wandb.ai/links/rentio/a74ndy24)

---

## 🤝 Contributing

This project welcomes contributions from the research community! We encourage:

### Types of Contributions
- **Bug Reports**: Help improve code quality and stability
- **Feature Requests**: Suggest new algorithms or environments
- **Performance Improvements**: Optimize training procedures
- **Documentation**: Enhance tutorials and examples
- **Research Extensions**: Implement new MARL algorithms

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive documentation
- Include performance benchmarks
- Provide usage examples

---

## 📄 License

This project is open source and available under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PettingZoo Team**: For providing excellent multi-agent environments
- **CleanRL Community**: For reference implementations and best practices
- **PyTorch Team**: For the powerful deep learning framework
- **Research Community**: For foundational papers and algorithms

---

## 📞 Contact

For questions, suggestions, or collaborations:
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Join our community discussions
- **Research**: Reach out for research collaborations

---

*This project represents a comprehensive exploration of multi-agent reinforcement learning, combining theoretical insights with practical implementations to advance the field of MARL research.*
