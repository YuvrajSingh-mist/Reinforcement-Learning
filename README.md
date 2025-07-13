# Deep Reinforcement Learning Projects

This repository contains various implementations of deep reinforcement learning algorithms across different environments. Each subfolder represents a specific environment or implementation of reinforcement learning techniques.


## Project Structure

### Value-Based Methods
- **[DQN](/DQN)**: Deep Q-Network implementation for CartPole and LunarLander environments
- **[DQN-atari](/DQN-atari)**: DQN adapted for Atari games with convolutional networks
- **[DQN-flappy](/DQN-flappy)**: DQN implementation for FlappyBird environment
- **[DQN-Lunar](/DQN-Lunar)**: DQN specifically tuned for the Lunar Lander environment
- **[DQN-Taxi](/DQN-Taxi)**: DQN for the discrete Taxi-v3 environment
- **[DQN-FrozenLake](/DQN-FrozenLake)**: DQN implementation for the FrozenLake environment
- **[Duel-DQN](/Duel-DQN)**: Dueling DQN with separate value and advantage streams for CliffWalking
- **[Q-Learning](/Q-Learning)**: Classic tabular Q-learning implementations

### Policy-Based Methods
- **[REINFORCE](/REINFORCE)**: Monte Carlo policy gradient method for CartPole environment
- **[A2C](/A2C)**: Advantage Actor-Critic implementation for multiple environments (CartPole, FrozenLake, LunarLander)
- **[PPO](/PPO)**: Proximal Policy Optimization with clipped surrogate objective for LunarLander
- **[FlappyBird-PPO](/FlappyBird-PPO)**: PPO implementation specifically for FlappyBird environment

### Actor-Critic Methods (Continuous Control)
- **[DDPG](/DDPG)**: Deep Deterministic Policy Gradient for continuous action spaces (Pendulum, BipedalWalker)
- **[TD3](/TD3)**: Twin Delayed DDPG with twin critics and delayed policy updates
- **[SAC](/SAC)**: Soft Actor-Critic with maximum entropy reinforcement learning

### Exploration & Advanced Methods
- **[RND](/RND)**: Random Network Distillation combined with PPO for curiosity-driven exploration
- **[NeatRL](/NeatRL)**: NEAT (NeuroEvolution of Augmenting Topologies) reinforcement learning implementations

### Game-Specific Implementations
- **[Pong](/Pong)**: Classic Pong environment implementations
- **[VizDoom-RL](/VizDoom-RL)**: Reinforcement learning in VizDoom 3D environments
- **[Frozen-Lake](/Frozen-Lake)**: Specialized implementations for FrozenLake environment
- **[SimpleRLGames](/SimpleRLGames)**: Collection of simple RL game implementations

### Unity ML-Agents
- **[ml-agents](/ml-agents)**: Unity ML-Agents toolkit for training intelligent agents in Unity environments
- **[ml-agents-train](/ml-agents-train)**: Training scripts and utilities for Unity ML-Agents

## Key Features

- **Comprehensive Algorithm Coverage**: Implementations spanning value-based (DQN variants), policy-based (REINFORCE, A2C, PPO), and actor-critic methods (DDPG, TD3, SAC)
- **Multiple Environment Support**: Code for various Gymnasium/OpenAI Gym environments including discrete and continuous action spaces
- **Advanced Techniques**: Experience replay, target networks, dueling architectures, curiosity-driven exploration (RND)
- **Continuous Control**: Specialized implementations for continuous action spaces with advanced algorithms
- **Visualization & Logging**: Integration with TensorBoard and Weights & Biases (WandB) for comprehensive experiment tracking
- **Game-Specific Optimizations**: Tailored implementations for specific games and environments
- **Unity Integration**: ML-Agents support for training in Unity environments
- **Trained Models**: Saved model weights and training logs for reproducible results
- **Comprehensive Logging**: Track metrics like Q-values, advantage, episode returns, and exploration statistics

## Getting Started

### Prerequisites

```bash
# Install the core requirements
pip install torch gymnasium numpy matplotlib tqdm tensorboard wandb stable-baselines3

# For Atari environments
pip install gymnasium[atari] autorom[accept-rom-license] opencv-python

# For video recording
pip install imageio opencv-python
```

### Running Experiments

Each project directory contains its own `train.py` script. Here are examples for different algorithm categories:

```bash
# Value-Based Methods
cd DQN && python train.py              # Standard DQN
cd DQN-atari && python train.py        # DQN for Atari
cd Duel-DQN && python train.py         # Dueling DQN
cd DQN-Taxi && python train.py         # DQN for Taxi

# Policy-Based Methods
cd REINFORCE && python train.py        # Basic policy gradient
cd A2C && python train.py              # Advantage Actor-Critic
cd PPO && python train.py              # Proximal Policy Optimization

# Continuous Control (Actor-Critic)
cd DDPG && python train.py             # Deep Deterministic Policy Gradient
cd TD3 && python train.py              # Twin Delayed DDPG
cd SAC && python train.py              # Soft Actor-Critic

# Advanced Methods
cd RND && python train.py              # Curiosity-driven exploration
cd FlappyBird-PPO && python flappy.py  # Game-specific PPO

# Unity ML-Agents
cd ml-agents-train && python game.py   # Unity environment training
```

## Reinforcement Learning Concepts

This repository explores comprehensive RL concepts across different paradigms:

### Value-Based Methods
- **Deep Q-Networks (DQN)**: Neural network function approximation for Q-values
- **Experience Replay**: Store and reuse past experiences for stable learning
- **Target Networks**: Stabilize training by reducing correlation between updates
- **Dueling Networks**: Separate value and advantage estimation for better learning

### Policy-Based Methods
- **Policy Gradient (REINFORCE)**: Direct policy optimization using Monte Carlo returns
- **Actor-Critic Methods**: Combine policy gradients with value function estimation
- **Advantage Functions**: Reduce variance in policy gradient estimates
- **Proximal Policy Optimization**: Stable policy updates with clipped objectives

### Continuous Control
- **Deterministic Policy Gradients**: Handle continuous action spaces efficiently
- **Twin Critics**: Reduce overestimation bias in Q-value estimation
- **Soft Actor-Critic**: Maximum entropy reinforcement learning for robust policies
- **Noise Injection**: Exploration strategies for continuous action spaces

### Advanced Techniques
- **Curiosity-Driven Learning**: Intrinsic motivation through prediction error (RND)
- **Multi-Environment Training**: Consistent algorithms across different domains
- **Exploration vs. Exploitation**: Various strategies including epsilon-greedy and entropy bonuses
- **Unity Integration**: Real-time training in complex 3D environments

## Results

Each implementation includes trained models and performance visualizations. Check the individual project READMEs for specific results.

## Extending the Projects

Ideas for extensions and improvements:

### Algorithm Enhancements
- Implement Rainbow DQN with all improvements (prioritized replay, noisy nets, etc.)
- Add Double DQN and other DQN variants
- Implement advanced policy gradient methods (TRPO, IMPALA)
- Add multi-agent reinforcement learning (MADDPG, QMIX)

### Architecture Improvements
- Experiment with different neural network architectures (CNNs, RNNs, Transformers)
- Implement attention mechanisms for partially observable environments
- Add hierarchical reinforcement learning approaches
- Explore meta-learning and few-shot adaptation

### Environment Extensions
- Apply algorithms to custom environments and real-world problems
- Implement curriculum learning for complex environments
- Add support for partial observability and memory-based agents
- Create multi-task learning setups

### Training Enhancements
- Implement distributed training across multiple GPUs/machines
- Add hyperparameter optimization and automated tuning
- Implement model-based reinforcement learning approaches
- Add imitation learning and learning from human feedback

## References

- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper (Mnih et al.)](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## License

MIT License
