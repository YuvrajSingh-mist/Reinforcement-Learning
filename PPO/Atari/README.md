# PPO on Atari Games

This directory contains **Proximal Policy Optimization (PPO)** implementations for training agents on classic Atari games using PyTorch and Gymnasium.

## 🎮 Overview

This implementation features vectorized PPO training on Atari environments with proper preprocessing, frame stacking, and environment wrappers. The code includes both custom implementations and Stable Baselines3 benchmarks for comparison.

## 🎯 Supported Games

- **Pong** (`PongNoFrameskip-v4`) - Classic paddle game
- **Boxing** (`BoxingNoFrameskip-v4`) - Fighting game
- **Bowling** (`BowlingNoFrameskip-v4`) - Bowling simulation

## 📁 Files

- `atari-pong.py` - PPO implementation for Pong
- `boxing-atari.py` - PPO implementation for Boxing  
- `atari-bowling.py` - PPO implementation for Bowling
- `sb3-atari-benchmark.py` - Stable Baselines3 benchmark comparison
- `images/` - Contains training videos and results

## 🚀 Features

### Core PPO Implementation
- **Vectorized Training**: 8 parallel environments for efficient data collection
- **Generalized Advantage Estimation (GAE)**: λ = 0.95 for bias-variance tradeoff
- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Value Function Clipping**: Stabilizes critic training
- **Gradient Clipping**: Prevents exploding gradients (max_grad_norm = 0.5)

### Atari-Specific Features
- **Frame Preprocessing**: Grayscale conversion and resizing to 64x64
- **Frame Stacking**: 4 consecutive frames as input
- **Standard Atari Wrappers**:
  - `NoopResetEnv`: Random no-op actions at episode start
  - `MaxAndSkipEnv`: Frame skipping and max pooling
  - `EpisodicLifeEnv`: Treats life loss as episode end
  - `FireResetEnv`: Automatically fires at episode start
  - `ClipRewardEnv`: Clips rewards to [-1, 1]

### Network Architecture
```
CNN Feature Extractor:
- Conv2d(4, 32, kernel=8, stride=4) + ReLU
- Conv2d(32, 64, kernel=4, stride=2) + ReLU  
- Conv2d(64, 64, kernel=3, stride=1) + ReLU
- Flatten
- Linear(64*7*7, 512) + ReLU

Actor Head: Linear(512, action_space)
Critic Head: Linear(512, 1)
```

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2.5e-4 | Adam optimizer learning rate |
| Discount Factor | 0.99 | Reward discount factor |
| Parallel Envs | 8 | Number of vectorized environments |
| Steps per Rollout | 128 | Steps collected per environment |
| Minibatches | 4 | Number of minibatches per update |
| PPO Epochs | 4 | Training epochs per rollout |
| Clip Range | 0.1 | PPO clipping parameter |
| Entropy Coeff | 0.01 | Entropy bonus coefficient |
| Value Coeff | 0.5 | Value loss coefficient |
| GAE Lambda | 0.95 | Advantage estimation parameter |

## 🏃‍♂️ Quick Start

### Prerequisites
```bash
pip install torch gymnasium ale-py stable-baselines3 wandb opencv-python imageio
```

### Training
```bash
# Train on Pong
python atari-pong.py

# Train on Boxing
python boxing-atari.py

# Train on Bowling
python atari-bowling.py

# Run SB3 benchmark
python sb3-atari-benchmark.py
```

### Configuration
Edit the `Config` class in each file to modify hyperparameters:
- `total_timesteps`: Total training steps (default: 10M)
- `env_id`: Environment name
- `lr`: Learning rate
- `num_envs`: Number of parallel environments
- `use_wandb`: Enable Weights & Biases logging

## 📊 Results

Training videos are saved in the `images/` directory:
- `pong.mp4` - Trained Pong agent gameplay
- `boxing.mp4` - Trained Boxing agent gameplay
- `bowling.mp4` - Trained Bowling agent gameplay

## 📈 Performance

The implementation typically achieves:
- **Pong**: 15-20 average reward after 10M steps
- **Boxing**: 80-95 average reward after 10M steps
- **Bowling**: 40-60 average reward after 10M steps

## 🔧 Technical Details

### Environment Preprocessing
1. **Observation**: 210x160x3 RGB frames → 64x64 grayscale
2. **Frame Stacking**: 4 consecutive frames
3. **Reward Clipping**: Rewards clipped to [-1, 1]
4. **Life Management**: Episode ends on life loss

### Training Loop
1. **Rollout Collection**: Collect trajectories from vectorized environments
2. **Advantage Estimation**: Compute GAE advantages
3. **Policy Update**: Multiple PPO epochs with minibatch updates
4. **Value Update**: Train critic with clipped value loss

## 🎯 Key Insights

- **Vectorization**: Dramatically improves sample efficiency
- **Frame Stacking**: Provides temporal information for decision making
- **Proper Preprocessing**: Essential for stable Atari training
- **Clipping**: Prevents destructive policy updates
- **GAE**: Reduces variance in advantage estimation

## 🔍 Monitoring

- **Tensorboard**: Real-time training metrics
- **Weights & Biases**: Experiment tracking and visualization
- **Video Recording**: Periodic agent gameplay videos
- **Console Logging**: Episode rewards and training progress

## 📚 References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## 💡 Tips for Success

1. **Sufficient Training Time**: Atari games require 10M+ steps
2. **Proper Preprocessing**: Use standard Atari wrappers
3. **Stable Learning Rate**: 2.5e-4 works well for most games
4. **Monitor Training**: Watch for policy collapse or instability
5. **Vectorization**: Use multiple environments for efficiency