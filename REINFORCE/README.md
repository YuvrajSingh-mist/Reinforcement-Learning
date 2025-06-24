# REINFORCE Algorithm Implementation

## Overview

This repository contains an implementation of the REINFORCE algorithm (also known as Monte Carlo Policy Gradient), a foundational policy gradient method in reinforcement learning. The implementation is built with PyTorch and supports training on various Gymnasium environments, with a focus on the CartPole-v1 environment.

## Algorithm Description

REINFORCE is a policy gradient method that directly optimizes a policy without using a value function. It belongs to the class of Monte Carlo methods as it uses complete episode returns for updating the policy. The key features of this implementation include:

- **Monte Carlo Policy Gradient**: Updates policy parameters using complete episode returns
- **Policy Network**: Neural network that maps states to action probabilities
- **Return Calculation**: Computes discounted returns from rewards
- **Return Normalization**: Normalizes returns to reduce variance in updates
- **Gradient Monitoring**: Tracks parameter and gradient statistics during training

### The Algorithm Steps

1. Initialize a parameterized policy π(a|s; θ)
2. For each episode:
   - Generate a complete trajectory following the current policy
   - For each step in the trajectory:
     - Calculate the discounted return from that step onwards
   - Update policy parameters using gradient ascent:
     - θ ← θ + α ∇θ log π(at|st; θ) Gt
3. Repeat until convergence

## Implementation Details

### Network Architecture

The policy network consists of:
- Input layer matching state space dimensions
- Two hidden layers (32 nodes each) with ReLU activation
- One hidden layer (16 nodes) with ReLU activation
- Output layer matching action space dimensions with softmax activation

### Key Features

- **Policy Network**: Maps states to action probabilities
- **Stochastic Action Selection**: Uses categorical distribution for action sampling
- **Return Calculation**: Computes discounted returns for each step
- **Return Normalization**: Reduces variance in policy updates
- **Gradient and Parameter Monitoring**: Tracks training dynamics with WandB
- **Evaluation**: Periodically evaluates policy performance
- **Video Recording**: Captures agent behavior for visualization

## Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3 (for ReplayBuffer utility)
- Weights & Biases (for logging)
- TensorBoard
- tqdm, numpy, imageio, cv2

### Configuration

The `Config` class contains all hyperparameters and settings:

```python
class Config:
    # Experiment settings
    exp_name = "DQN-CartPole"  # Can be renamed to "REINFORCE-CartPole"
    seed = 42
    env_id = "CartPole-v1"
    episodes = 2000  # Number of episodes to train
    # Training parameters
    learning_rate = 2e-3
    gamma = 0.99  # Discount factor
    # Logging & saving
    capture_video = True
    save_model = True
    use_wandb = True
    wandb_project = "cleanRL"
    wandb_entity = ""  # Your WandB username/team
```


## License

This project is open source and available under the [MIT License](LICENSE).
