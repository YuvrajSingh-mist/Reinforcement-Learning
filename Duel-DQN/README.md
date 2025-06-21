# Dueling Deep Q-Network (Dueling DQN)

This project implements the Dueling Deep Q-Network (Dueling DQN) algorithm for the CliffWalking environment from OpenAI Gymnasium. The agent learns to navigate through a dangerous cliff area to reach a goal without falling off.

![Cliff Climbing Environment](images/output.gif)

## Algorithm Description

**Dueling DQN** is an advanced variant of the Deep Q-Network (DQN) algorithm that uses a special neural network architecture to separately estimate:

1. **Value function (V)**: The value of being in a particular state
2. **Advantage function (A)**: The advantage of taking specific actions in that state

The Q-values are then computed as: Q(s,a) = V(s) + A(s,a) - mean(A(s))

This architecture helps the agent learn which states are valuable without having to learn the effect of each action for each state, leading to more efficient learning, especially in environments with many similar-valued actions.

## Environment Description

**CliffWalking-v0** is a grid-world environment where:
- The agent must navigate from a starting position to a goal position
- There's a cliff along the bottom of the grid that the agent must avoid
- Falling off the cliff gives a large negative reward and resets the agent to the start
- Each step incurs a small negative reward to encourage the agent to find the shortest path

- **State space:** Discrete with 48 states (represented as one-hot encoded vectors)
- **Action space:** 4 discrete actions (LEFT, DOWN, RIGHT, UP)
- **Rewards:** -1 for each step, -100 for falling off the cliff, 0 for reaching the goal

## Implementation Details

```python
class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.values = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        self.adv = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )
        
    def forward(self, x):
        feat = self.features(x)
        values = self.values(feat)
        adv = self.adv(feat)
        # Q = V + A - mean(A)
        res = values + adv - adv.mean(dim=1, keepdim=True)
        return res, values, adv, feat
```

## Key Features

- **Dueling Network Architecture**: Separate estimators for state values and action advantages
- **Experience Replay**: Buffer stores transitions for off-policy learning
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Linear decay of exploration rate
- **Gradient Clipping**: Prevents unstable updates with large gradients
- **Comprehensive Logging**: Track metrics like Q-values, advantage values, and training progress
- **Model Evaluation**: Periodic evaluation with video recording

## Configuration Options

```python
class Config:
    # Experiment settings
    exp_name = "DQN-CliffWalking"
    seed = 42
    env_id = "CliffWalking-v0"
    
    # Training parameters
    total_timesteps = 300000
    learning_rate = 2e-4
    buffer_size = 30000
    gamma = 0.99
    tau = 1.0  # Target network update rate
    target_network_frequency = 50
    batch_size = 128
    start_e = 1.0  # Initial exploration rate
    end_e = 0.05   # Final exploration rate
    exploration_fraction = 0.4
    learning_starts = 1000
    train_frequency = 4
    max_grad_norm = 4.0  # Maximum gradient norm for gradient clipping
```

## Results

The Dueling DQN algorithm demonstrates several advantages:

- **Faster Learning**: Converges to better policies more quickly than standard DQN
- **More Stable Performance**: Reduced variance in learning due to the value/advantage decomposition
- **Better Policy Quality**: Finds more optimal paths by focusing on important state features

The agent successfully learns to navigate the cliff environment by taking the longer but safer path along the top of the grid, avoiding the risky cliff edge.

## Visualization

Trained agent performance is recorded as videos in the `videos/` directory, showing the learned navigation policy avoiding the cliff while reaching the goal.

