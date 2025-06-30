# Twin Delayed Deep Deterministic Policy Gradient (TD3)

This directory contains implementations of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for various continuous control environments.

## Overview

TD3 is an advanced off-policy actor-critic algorithm designed to address the overestimation bias in DDPG. It introduces three critical improvements:

1. **Twin Critics**: Uses two Q-value networks to reduce overestimation bias through taking the minimum Q-value.
2. **Delayed Policy Updates**: Updates the policy less frequently than the critics to reduce variance.
3. **Target Policy Smoothing**: Adds noise to the target actions to make the algorithm more robust to errors.

Key features of this implementation:
- Actor-Critic architecture with twin critics
- Delayed policy updates
- Target policy smoothing regularization
- Experience replay buffer for stable learning
- Soft target network updates using Polyak averaging
- Exploration using additive noise
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
- `policy_noise`: Standard deviation of noise added to target policy
- `total_timesteps`: Total number of training steps
- `learning_rate`: Learning rate for the optimizer
- `buffer_size`: Size of the replay buffer
- `gamma`: Discount factor
- `tau`: Soft update coefficient for target networks
- `batch_size`: Batch size for training
- `clip`: Clipping range for target policy smoothing noise
- `exploration_fraction`: Fraction of total timesteps for exploration
- `learning_starts`: Number of timesteps before learning starts
- `train_frequency`: Frequency of updates to the networks

## Architecture

The TD3 implementation includes:

1. **Actor Network**: Determines the best action in a given state
2. **Twin Critic Networks**: Two separate networks that evaluate the Q-value of state-action pairs
3. **Target Networks**: Slowly updated copies of both actor and critics for stability
4. **Replay Buffer**: Stores and samples transitions for training
5. **Noise Process**: Adds exploration noise to actions during training

## Improvements Over DDPG

TD3 addresses several shortcomings of DDPG:

1. **Reducing Overestimation Bias**: By using the minimum of two critics, TD3 helps mitigate the overestimation bias that plagues many Q-learning algorithms.
2. **Stabilized Learning**: Delayed policy updates (updating the policy less frequently than the critics) help reduce variance and stabilize learning.
3. **Smoother Target Values**: Adding noise to target actions smooths the value function, making the learning process more robust to errors.

## Results

The implementation includes a video recording (`TD3_BipedalWalker.mp4`) that demonstrates the performance of the trained TD3 agent on the BipedalWalker environment.

### Training Visualizations

#### BipedalWalker Agent

![BipedalWalker Performance](images/BiPedal.png)

Here's a GIF showing the trained TD3 agent navigating the BipedalWalker environment:

![BipedalWalker Agent in Action](images/output_bipedal_walker.gif)

#### HalfCheetah Training

The following graph shows the training losses for the HalfCheetah environment:

![HalfCheetah Training Loss](images/lossHalfCheetah.png)


- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Inspiration for code structure and implementation style
