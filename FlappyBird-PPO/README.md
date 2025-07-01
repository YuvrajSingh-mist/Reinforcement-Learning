# Flappy Bird with Proximal Policy Optimization (PPO)

This directory contains an implementation of the Proximal Policy Optimization (PPO) algorithm applied to the Flappy Bird environment.

## Overview

This project demonstrates how to train an agent to play Flappy Bird using the PPO algorithm, a state-of-the-art policy gradient method in reinforcement learning. The implementation leverages the `flappy_bird_gymnasium` environment, which provides a Gym-compatible interface for the classic Flappy Bird game.

## Environment

**Flappy Bird** is a side-scrolling game where the player controls a bird, attempting to fly between columns of green pipes without hitting them. The game mechanics are simple:
- The bird automatically moves forward
- The player can make the bird "flap" to move upward
- Gravity pulls the bird downward
- The goal is to navigate through as many pipes as possible

**State Space**: The observation space consists of game state information, including:
- Bird's position and velocity
- Positions of the upcoming pipes
- Distances between the bird and pipe openings

**Action Space**: The action space is discrete with two possible actions:
- 0: Do nothing (let the bird fall)
- 1: Flap (make the bird move upward)

## Implementation

The implementation uses a PPO agent with:

- **Actor-Critic Architecture**: Separate networks for policy (actor) and value function (critic)
- **Clipped Surrogate Objective**: Prevents excessive policy updates
- **Entropy Bonus**: Encourages exploration
- **Generalized Advantage Estimation (GAE)**: For variance reduction in policy gradient estimation

## Configuration

The implementation uses a `Config` class with the following key parameters:

- `exp_name`: "PPO-Flappy" - Name of the experiment
- `env_id`: "FlappyBird-v0" - Environment ID
- `episodes`: 10000 - Number of training episodes
- `lr`: 3e-4 - Learning rate
- `gamma`: 0.99 - Discount factor
- `clip_value`: 0.2 - PPO clipping parameter
- `PPO_EPOCHS`: 4 - Number of optimization epochs per batch
- `ENTROPY_COEFF`: 0.01 - Coefficient for entropy bonus
- `max_steps`: 512 - Maximum steps per episode

## Training Process

The agent is trained through an iterative process:

1. **Interaction with Environment**: The agent collects trajectories by playing the game
2. **Advantage Calculation**: Compute advantages using Generalized Advantage Estimation
3. **Policy Update**: Update policy and value function using the PPO objective
4. **Repeat**: Continue training until the agent achieves satisfactory performance

## Results

The agent successfully learns to play Flappy Bird, navigating through pipes with increasing proficiency as training progresses. A video of the trained agent's performance is included (`final_FlappyBird-v0.mp4`).

![Flappy Bird Agent](images/output.gif)
