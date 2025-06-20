# Frozen Lake Reinforcement Learning

This project implements reinforcement learning algorithms for the Frozen Lake environment from OpenAI Gymnasium. The agent learns to navigate across a frozen lake from the start to a goal without falling into holes.

![Frozen Lake Environment](images/output.gif)

## Environment Description

**FrozenLake-v1** is a grid-world environment where:
- The agent navigates on a frozen lake from start (S) to goal (G)
- Some tiles are frozen (F) and safe to walk on
- Some tiles have holes (H) and the agent falls if it steps on them
- The ice is slippery, so the agent's movement can be stochastic

Example 4x4 map:
```
SFFF
FHFH
FFFH
HFFG
```

- **State space:** Discrete with 16 states (for 4x4 grid) or 64 states (for 8x8 grid)
- **Action space:** 4 discrete actions (LEFT, DOWN, RIGHT, UP)
- **Rewards:** +1 for reaching the goal, 0 otherwise

## Algorithms Implemented

This project includes implementations of:

1. **Q-Learning**: A model-free, off-policy algorithm using a tabular approach
2. **Deep Q-Network (DQN)**: Neural network implementation for Q-learning
3. **Double DQN**: Reducing overestimation bias with two networks

## Features

- Multiple map sizes (4x4 and 8x8)
- Option for deterministic or stochastic (slippery) environments
- Exploration vs. exploitation control with epsilon-greedy strategy
- Visualization of learned policies
- Tracking of training metrics
- Integration with TensorBoard and WandB



```python
class Config:
    # Environment settings
    env_id = "FrozenLake-v1"
    map_size = "4x4"  # or "8x8"
    is_slippery = True
    
    # Algorithm parameters
    learning_rate = 0.1  # for Q-Learning
    gamma = 0.99  # Discount factor
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    # Training parameters
    total_episodes = 10000
    max_steps = 100
    
    # For DQN
    buffer_size = 10000
    batch_size = 64
    target_update = 100
    
    # Logging
    use_wandb = True
    log_interval = 100
```

## Results

The algorithms learn efficient policies for navigating the Frozen Lake:

- **Q-Learning**: Converges to optimal policy after ~5000 episodes for 4x4 map
- **DQN**: Learns good policies but might be less sample-efficient for this simple environment
- **Double DQN**: Provides more stable learning, especially for the 8x8 map

## Visualization

The project includes tools to visualize:
- Learning curves
- Value functions
- Optimal policies
- Step-by-step agent behavior


## References

- [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Q-Learning Paper](https://link.springer.com/article/10.1007/BF00992698)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)

## License

MIT License
