# Deep Deterministic Policy Gradient (DDPG)

This directory contains implementations of the Deep Deterministic Policy Gradient (DDPG) algorithm for various continuous control environments.

## Overview

DDPG is an off-policy actor-critic algorithm designed for continuous action spaces. It combines insights from both Deep Q-Networks (DQN) and policy gradient methods to learn policies in high-dimensional, continuous action spaces.

Key features of this implementation:
- Actor-Critic architecture with separate target networks
- Experience replay buffer for stable learning
- Soft target network updates using Polyak averaging
- Exploration using Ornstein-Uhlenbeck noise process
- Support for different continuous control environments

## Environments

This implementation includes support for the following environments:
- **Pendulum-v1**: A classic control problem where the goal is to balance a pendulum in an upright position.
- **BipedalWalker-v3**: A more challenging environment where a 2D biped robot must walk forward without falling.
- **HalfCheetah-v5**: A MuJoCo environment where a 2D cheetah-like robot must run forward as fast as possible.


## Configuration

Each implementation includes a `Config` class that specifies the hyperparameters for training. You can modify these parameters to experiment with different settings:

- `exp_name`: Name of the experiment
- `seed`: Random seed for reproducibility
- `env_id`: ID of the Gymnasium environment
- `total_timesteps`: Total number of training steps
- `learning_rate`: Learning rate for the optimizer
- `buffer_size`: Size of the replay buffer
- `gamma`: Discount factor
- `tau`: Soft update coefficient for target networks
- `batch_size`: Batch size for training
- `exploration_fraction`: Fraction of total timesteps for exploration
- `learning_starts`: Number of timesteps before learning starts

## Architecture

The DDPG implementation includes:

1. **Actor Network**: Determines the best action in a given state
2. **Critic Network**: Evaluates the Q-value of state-action pairs
3. **Target Networks**: Slowly updated copies of both actor and critic for stability
4. **Replay Buffer**: Stores and samples transitions for training
5. **Noise Process**: Adds exploration noise to actions

## Logging and Monitoring

Training progress is logged using:
- **TensorBoard**: Local visualization of training metrics
- **Weights & Biases (WandB)**: Cloud-based experiment tracking (optional)
- **Video Capture**: Records videos of agent performance at intervals

## Dependencies

- PyTorch
- Gymnasium
- NumPy
- Stable-Baselines3 (for the replay buffer)
- WandB (optional, for experiment tracking)
- TensorBoard
- Tqdm

## References

- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) - Original DDPG paper by Lillicrap et al.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Inspiration for code structure and implementation style
