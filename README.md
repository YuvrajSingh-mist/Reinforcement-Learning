# Deep Reinforcement Learning Projects

This repository contains various implementations of deep reinforcement learning algorithms across different environments. Each subfolder represents a specific environment or implementation of reinforcement learning techniques.


## Project Structure

- **[DQN](/DQN)**: Deep Q-Network implementation for CartPole and LunarLander environments
- **[DQN-atari](/DQN-atari)**: DQN adapted for Atari games with convolutional networks
- **[DQN-flappy](/DQN-flappy)**: DQN implementation for FlappyBird environment
- **[DQN-Lunar](/DQN-Lunar)**: DQN specifically tuned for the Lunar Lander environment
- **[DQN-Taxi](/DQN-Taxi)**: DQN for the discrete Taxi-v3 environment
- **[Q-Learning](/Q-Learning)**: Classic tabular Q-learning implementations

## Key Features

- **Various Algorithms**: Implementations of DQN, Q-Learning, and their variants
- **Multiple Environments**: Code for various Gymnasium/OpenAI Gym environments
- **Visualization**: Integration with TensorBoard and Weights & Biases (WandB)
- **Trained Models**: Saved model weights and training logs
- **Comprehensive Logging**: Track metrics like Q-values, advantage, episode returns

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

Each project directory contains its own `train.py` script:

```bash
# For DQN CartPole
cd DQN
python train.py

# For Atari
cd DQN-atari
python train.py

# For Taxi
cd DQN-Taxi
python train.py
```

## Reinforcement Learning Concepts

This repository explores several key RL concepts:

- **Value-Based Methods**: DQN and Q-learning algorithms
- **Experience Replay**: Store and reuse past experiences
- **Exploration vs. Exploitation**: Epsilon-greedy strategy
- **Function Approximation**: Neural networks to approximate Q-values
- **Target Networks**: Stabilize training by reducing correlation

## Results

Each implementation includes trained models and performance visualizations. Check the individual project READMEs for specific results.

## Extending the Projects

Ideas for extensions:
- Implement additional algorithms (DDPG, PPO, SAC)
- Try different neural network architectures
- Add prioritized experience replay
- Implement multi-agent reinforcement learning
- Apply these techniques to custom environments

## References

- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper (Mnih et al.)](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## License

MIT License
