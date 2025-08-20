# Multi-Agent Reinforcement Learning (MARL) Implementation

This comprehensive MARL project demonstrates state-of-the-art multi-agent algorithms implemented and trained on various PettingZoo environments. The implementation includes **IPPO** (Independent Proximal Policy Optimization), **MAPPO** (Multi-Agent Proximal Policy Optimization), and **RND** (Random Network Distillation) variants, supporting both discrete and continuous action spaces, with extensive **self-play** capabilities.

<p align="center">
  <img src="https://github.com/PettingZoo-Team/PettingZoo/raw/master/imgs/pong.gif" width="300"/>
  <img src="https://github.com/PettingZoo-Team/PettingZoo/raw/master/imgs/simple_spread.gif" width="300"/>
</p>

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Supported Algorithms](#supported-algorithms)
3. [Supported Environments](#supported-environments)
4. [Self-Play Capabilities](#self-play-capabilities)
5. [Quick Start](#quick-start)
6. [Training Examples](#training-examples)
7. [Hyper-parameters](#hyper-parameters)
8. [Training Details](#training-details)
9. [Evaluation](#evaluation)
10. [Self Play](#self-play)
11. [Saving & Loading Checkpoints](#saving--loading-checkpoints)
12. [Dependencies](#dependencies)
13. [References](#references)

---

## Project Structure
```
MARL/
├── train.py                 # Main training script for Pong self-play
├── play_ippo.py            # Play script for trained models
├── IPPO/                   # Independent PPO implementations
│   ├── ippo_discrete.py    # IPPO for discrete action spaces (Simple Spread)
│   ├── ippo_continuous.py  # IPPO for continuous action spaces
│   ├── ippo_simple_tag.py  # IPPO for Simple Tag environment
│   ├── play_ippo.py        # Play script for IPPO models (Pong)
│   ├── pong.mp4            # Demo video
│   └── images/             # Training visualizations
├── MAPPO/                  # Multi-Agent PPO implementations
│   ├── mappo_without_rnd.py    # Standard MAPPO
│   ├── mappo_rnd.py           # MAPPO with RND for exploration
│   ├── mappo_rnd_pong.py      # MAPPO with RND for cooperative Pong
│   ├── train.py               # MAPPO training script (cooperative Pong)
│   └── images/                # Training visualizations
├── Self Play/              # Self-play utilities
│   ├── play.py             # Watch two trained agents compete (Pong)
│   ├── self_play.py        # Self-play training driver (Pong)
│   └── pt files/           # Saved checkpoints
│       └── Pong-MARL.pt    # Pre-trained Pong model
└── README.md               # ← you are here
```

## Supported Algorithms

### 1. IPPO (Independent Proximal Policy Optimization)
- **Location**: `IPPO/` directory
- **Variants**:
  - `ippo_discrete.py`: Discrete action spaces (Simple Spread, Simple Tag)
  - `ippo_continuous.py`: Continuous action spaces
  - `ippo_simple_tag.py`: Specialized for Simple Tag environment
  - `play_ippo.py`: Interactive play script for Pong
- **Key Features**:
  - Independent learning for each agent
  - Shared observation processing with agent-specific heads
  - Support for both discrete and continuous action spaces
  - GAE advantage estimation and PPO clipping
  - **Self-play capabilities** for competitive environments

### 2. MAPPO (Multi-Agent Proximal Policy Optimization)
- **Location**: `MAPPO/` directory
- **Variants**:
  - `mappo_without_rnd.py`: Standard MAPPO implementation
  - `mappo_rnd.py`: MAPPO with Random Network Distillation for exploration
  - `mappo_rnd_pong.py`: MAPPO with RND specifically for cooperative Pong
  - `train.py`: MAPPO training script for cooperative Pong
- **Key Features**:
  - Centralized training with decentralized execution
  - Enhanced exploration through RND variants
  - Optimized for cooperative multi-agent tasks
  - **Cooperative Pong** environment support

### 3. RND (Random Network Distillation)
- **Purpose**: Intrinsic motivation for exploration
- **Implementation**: Integrated into MAPPO variants
- **Benefits**: Helps agents explore complex environments more effectively

## Supported Environments

### Atari Environments
- **Pong-v3**: Classic Atari Pong with self-play capabilities
- **Features**: Image-based observations, discrete actions, competitive gameplay

### PettingZoo MPE Environments
- **Simple Spread**: Cooperative navigation task
- **Simple Tag**: Competitive tagging game
- **Features**: Vector observations, both discrete and continuous actions

### PettingZoo Butterfly Environments
- **Cooperative Pong-v5**: Cooperative version of Pong for MAPPO
- **Features**: Multi-agent cooperation, image-based observations

## Self-Play Capabilities

### 1. Competitive Self-Play (Pong)
- **Implementation**: `train.py` (main), `Self Play/self_play.py`
- **Environment**: PettingZoo Atari Pong-v3
- **Features**:
  - Two agents compete against each other
  - Shared policy learning
  - Automatic opponent generation
  - Real-time visualization

### 2. Interactive Play
- **Implementation**: `Self Play/play.py`, `IPPO/play_ippo.py`
- **Features**:
  - Human vs. AI gameplay
  - Keyboard controls for human player
  - Real-time rendering
  - Pre-trained model loading

### 3. Cooperative Multi-Agent (MAPPO)
- **Implementation**: `MAPPO/train.py`, `MAPPO/mappo_rnd_pong.py`
- **Environment**: Cooperative Pong-v5
- **Features**:
  - Multiple agents work together
  - Centralized training
  - RND exploration enhancement

## Quick Start

### 1. Install Dependencies
```bash
pip install torch pettingzoo[atari,mpe,butterfly] supersuit wandb tqdm imageio opencv-python gymnasium
```

### 2. Train IPPO on Simple Spread
```bash
cd MARL/IPPO
python ippo_discrete.py --env_id simple_spread_v3 --total_timesteps 10000000
```

### 3. Train MAPPO with RND
```bash
cd MARL/MAPPO
python mappo_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000
```

### 4. Train Self-Play Pong
```bash
cd MARL
python train.py --env_id pong_v3 --total_timesteps 10000000
```

### 5. Train Cooperative Pong (MAPPO)
```bash
cd MARL/MAPPO
python train.py --env_id cooperative_pong_v5 --total_timesteps 10000000
```

## Training Examples

### IPPO Training Commands
```bash
# Discrete action space (Simple Spread)
python IPPO/ippo_discrete.py --env_id simple_spread_v3 --total_timesteps 10000000

# Continuous action space
python IPPO/ippo_continuous.py --env_id simple_spread_v3 --total_timesteps 10000000

# Simple Tag environment
python IPPO/ippo_simple_tag.py --env_id simple_tag_v3 --total_timesteps 10000000
```

### MAPPO Training Commands
```bash
# Standard MAPPO (Simple Spread)
python MAPPO/mappo_without_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000

# MAPPO with RND for exploration
python MAPPO/mappo_rnd.py --env_id simple_spread_v3 --total_timesteps 20000000

# MAPPO with RND for cooperative Pong
python MAPPO/mappo_rnd_pong.py --env_id cooperative_pong_v5 --total_timesteps 10000000

# MAPPO training script for cooperative Pong
python MAPPO/train.py --env_id cooperative_pong_v5 --total_timesteps 10000000
```

### Self-Play Training Commands
```bash
# Main self-play training (Pong)
python train.py --env_id pong_v3 --total_timesteps 15000000

# Alternative self-play driver
python "Self Play/self_play.py" --env_id pong_v3 --total_timesteps 15000000
```

## Hyper-parameters

### IPPO Configuration
```python
# Key parameters in IPPO implementations
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Number of parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.001         # Entropy regularization
GAE = 0.95                    # Generalized Advantage Estimation λ
```

### MAPPO Configuration
```python
# Key parameters in MAPPO implementations
lr = 2.5e-4                    # Learning rate
num_envs = 15                  # Number of parallel environments
max_steps = 256               # Rollout length (longer for MAPPO)
PPO_EPOCHS = 10               # PPO update epochs
clip_coeff = 0.2              # PPO clipping coefficient
ENTROPY_COEFF = 0.02          # Entropy regularization
GAE = 0.95                    # Generalized Advantage Estimation λ
```

### Self-Play Configuration
```python
# Key parameters for self-play training
lr = 2.5e-4                    # Learning rate
num_envs = 16                  # Number of parallel environments
max_steps = 128               # Rollout length
PPO_EPOCHS = 4                # PPO update epochs
clip_coeff = 0.1              # PPO clipping coefficient
ENTROPY_COEFF = 0.01          # Entropy regularization
total_timesteps = 15000000    # Total training steps
```

## Training Details

### Observation Processing
- **Atari**: Grayscale, resize to 84×84, 4-frame stack, agent indicator channel, downsampled to 64×64
- **MPE**: Direct vector observations with agent-specific processing
- **Butterfly**: Image-based observations with multi-agent coordination

### Network Architecture
- **Shared Encoder**: Convolutional tower for images, MLP for vectors
- **Agent-Specific Heads**: Separate actor and critic networks per agent
- **Optimization**: Adam with gradient clipping (0.5) + orthogonal initialization

### Multi-Agent Coordination
- **IPPO**: Independent learning with shared observation processing
- **MAPPO**: Centralized training with decentralized execution
- **Self-Play**: Agents compete against each other in the same environment
- **Cooperative**: Multiple agents work together toward common goals

## Evaluation

### Evaluation Metrics
- Per-episode rewards for each agent
- Average returns across episodes
- Cooperation/competition metrics for multi-agent tasks
- Self-play win rates and ELO ratings

### Video Recording
```bash
# Enable video capture during evaluation
python IPPO/ippo_discrete.py --eval --capture_video True --checkpoint "checkpoint.pt"

# Self-play evaluation
python "Self Play/play.py" --checkpoint "Self Play/pt files/Pong-MARL.pt"
```

## Self Play

### Watch Trained Agents
```bash
# Watch two self-play agents compete (Pong)
python "Self Play/play.py" "Self Play/pt files/Pong-MARL.pt"

# Watch IPPO agents in Pong
python IPPO/play_ippo.py "IPPO/checkpoint.pt"

# Watch MAPPO agents in cooperative task
python MAPPO/play_ippo.py "MAPPO/checkpoint.pt"
```

### Interactive Play (Human vs AI)
```bash
# Play against trained Pong agent
python "Self Play/play.py" "Self Play/pt files/Pong-MARL.pt"
# Controls: W=Right, S=Left, F=Fire, D=Fire Right, A=Fire Left, Q=Quit
```

### Self-Play Training
```bash
# Continue training with self-play
python "Self Play/self_play.py" --checkpoint "checkpoint.pt"

# Train from scratch with self-play
python train.py --env_id pong_v3 --total_timesteps 15000000
```

## Saving & Loading Checkpoints

### Automatic Checkpointing
- Checkpoints saved every 200 updates
- Final checkpoint saved at training completion
- Location: `pt files/` directory

### Pre-trained Models
- `Self Play/pt files/Pong-MARL.pt`: Pre-trained Pong self-play model
- Ready for immediate evaluation and interactive play

### Manual Checkpoint Loading
```python
import torch
state_dict = torch.load("checkpoint.pt")
actor.load_state_dict(state_dict["model_state"])
optimizer.load_state_dict(state_dict["optimizer_state"])
```

## Dependencies

### Core Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- gymnasium
- pettingzoo[atari,mpe,butterfly]
- supersuit

### Optional Dependencies
- wandb (experiment tracking)
- tqdm (progress bars)
- imageio (video recording)
- opencv-python (image processing)

### Installation
```bash
pip install torch pettingzoo[atari,mpe,butterfly] supersuit wandb tqdm imageio opencv-python gymnasium
```

## References

### Papers
- [IPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [RND: Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

### Libraries
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environment library
- [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) - Environment preprocessing
- [PyTorch](https://pytorch.org/) - Deep learning framework

### WandB Reports
- [![WandB Report](https://img.shields.io/badge/WandB-Report-blue?logo=wandb)](https://api.wandb.ai/links/rentio/a74ndy24)

---

## Contributing

This project welcomes contributions! Please feel free to:
- Add new multi-agent algorithms
- Implement additional environments
- Improve documentation
- Submit bug reports and feature requests

## License

This project is open source and available under the MIT License.
