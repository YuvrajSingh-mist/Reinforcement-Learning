# PPO on MuJoCo Environments

This directory contains PPO implementations specifically for MuJoCo continuous control environments. These environments represent some of the most challenging continuous control tasks in reinforcement learning, requiring sophisticated policy learning to coordinate multiple joints and maintain dynamic stability.

## Overview

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for robotics and biomechanics simulation. The environments implemented here focus on locomotion tasks that require:

- Complex multi-joint coordination
- Dynamic stability and balance
- Continuous action spaces with high dimensionality
- Robust policy learning for physical simulation

## Environments

PPO has been successfully applied to several challenging MuJoCo continuous control environments:

| Environment | Description | Action Space | Observation Space | Demo | WandB Report |
|-------------|-------------|--------------|-------------------|------|--------------|
| **HalfCheetah-v5** | Quadrupedal locomotion task where the agent learns to run forward using a cheetah-like body. Requires coordination of hip, knee, and ankle joints for fast and stable running. | 6D continuous (hip, knee, ankle torques) | 17D continuous (joint positions, velocities, and orientations) | ![HalfCheetah](images/cheetah.gif) | [PPO on HalfCheetah-v5](https://wandb.ai/rentio/cleanRL/reports/PPO-C-on-Half-Cheetah--VmlldzoxMzUxNjA5Mg?accessToken=r6rxlnqyjc1lpoq1gx4w2ybt01jsrta2y8p8ycwmcnisgm60hhabbvywl3aepbpa) |
| **Humanoid-v5** | Complex humanoid control task with full body coordination. The agent learns to maintain balance and locomotion using a humanoid robot with multiple joints and degrees of freedom. Requires sophisticated control of torso, arms, and legs. | High-dimensional continuous (full body joint torques) | High-dimensional continuous (joint positions, velocities, body orientation, and contact information) | ![Humanoid](images/humanoid.gif) | [PPO on Humanoid-v5](https://wandb.ai/rentio/cleanRL/reports/PPO-Humanoid--VmlldzoxMzUzMzMwNA) |
| **Hopper-v5** | Single-leg hopping locomotion task where the agent learns to hop forward while maintaining balance. Requires precise control of thigh, leg, and foot joints to achieve stable hopping motion without falling. | 3D continuous (thigh, leg, foot torques) | 11D continuous (joint positions, velocities, and body orientation) | ![Hopper](images/hopper.gif) | [PPO on Hopper-v5](https://wandb.ai/rentio/cleanRL/reports/PPO-Hopper---VmlldzoxMzU1NDA0MQ?accessToken=oo2o366arrkrc26st173cov4n701398wca186wzx6dpxy47gadpl9zwkocbvry0r) |
| **Walker2d-v5** | Bipedal walking locomotion task where the agent learns to walk forward using two legs. Requires coordination of thigh, leg, and foot joints for both legs to achieve stable walking motion while maintaining upright posture. | 6D continuous (left/right thigh, leg, foot torques) | 17D continuous (joint positions, velocities, and body orientation) | ![Walker2d](images/walker2d.gif) | [PPO on Walker2d-v5](https://api.wandb.ai/links/rentio/sn4gsbac) |
| **Pusher-v4** | Robotic arm manipulation task where the agent learns to control a 7-DOF arm to push objects to target locations. Requires precise control of multiple joints to position and manipulate objects in 3D space while avoiding obstacles. | 7D continuous (arm joint torques) | 23D continuous (joint angles, velocities, object positions, target location) | ![Pusher](images/pusher.gif) | [PPO on Pusher-v4](https://wandb.ai/rentio/cleanRL/reports/PPO-Reacher--VmlldzoxMzU1NDA2NA?accessToken=bw14z6xzu5603mvjvha1eoqbzasq3iom5rkzg9q0kr82s57hysdx96m9fm74nz8i) |
| **Reacher-v4** | Robotic arm reaching task where the agent learns to control a 2-joint arm to reach target positions. Requires precise control of shoulder and elbow joints to position the end effector at randomly placed targets in 2D space. | 2D continuous (shoulder, elbow torques) | 11D continuous (joint angles, velocities, target position, fingertip position) | ![Reacher](images/reacher.gif) | [PPO on Reacher-v4](https://wandb.ai/rentio/cleanRL/reports/REACHER-V4-PPO--VmlldzoxMzU1NDA5NA?accessToken=jey5gs4ocoyq47foysww4ss9irq9y6s8pnpqamfre6xevadkzk20rvd8afmn04o0) |
| **Ant-v4** | Quadrupedal locomotion task where the agent learns to control a four-legged ant robot to move forward. Requires coordination of 8 joints (2 per leg) to achieve stable walking while maintaining balance and avoiding falls. | 8D continuous (hip and ankle torques for 4 legs) | 27D continuous (joint positions, velocities, and body orientation) | ![Ant](images/ant.gif) | [PPO on Ant-v4](https://wandb.ai/rentio/cleanRL/reports/PPO-Ant--VmlldzoxMzU1NDA2NA) |
| **Swimmer-v4** | Aquatic locomotion task where the agent learns to control a swimming robot to move forward through water. Requires coordination of multiple joints to generate propulsive forces and maintain directional movement in fluid environment. | 2D continuous (joint torques for swimming motion) | 8D continuous (joint angles, velocities, and body orientation) | ![Swimmer](images/swimmer.gif) | [PPO on Swimmer-v4](https://wandb.ai/rentio/cleanRL/reports/Swimmer-V4-PPO--VmlldzoxMzU1NDEyMA?accessToken=0k2daj964kvs5xdgugt05dx7ssgxlmhxo8y1n0c4uaa1o9w7yo3x0bq7vrnp7sh4) |

## Implementation Details

All MuJoCo implementations use:
- **Continuous PPO**: Adapted for continuous action spaces using Gaussian policies
- **Clipped Surrogate Objective**: For stable policy updates
- **Value Function Learning**: Separate critic network for state value estimation

### Hyperparameters
The MuJoCo environments typically require:
- Lower learning rates (1e-4 to 3e-4) for stable learning
- Longer training episodes due to environment complexity
- Careful entropy coefficient tuning for exploration vs exploitation
- Higher GAE lambda values for better value estimation

## Files Description

- `half-cheetah.py`: PPO implementation for HalfCheetah-v5 environment
- `humanoids.py`: PPO implementation for Humanoid-v5 environment
- `ant-v4.py`: PPO implementation for Ant-v4 quadrupedal locomotion
- `hopper.py`: PPO implementation for Hopper hopping locomotion
- `swimmer.py`: PPO implementation for Swimmer aquatic locomotion
- `pusher.py`: PPO implementation for Pusher manipulation task
- `reacher.py`: PPO implementation for Reacher arm reaching task

## Usage

```bash
# Train HalfCheetah
python half-cheetah.py

# Train Humanoid
python humanoids.py

# Train other MuJoCo environments
python ant-v4.py
python hopper.py
# ... etc
```

## Dependencies

- PyTorch
- Gymnasium[mujoco]
- MuJoCo (physics engine)
- NumPy
- WandB (for experiment tracking)
- TensorBoard

## Notes

- MuJoCo environments require the MuJoCo physics engine to be installed
- These environments are computationally intensive and benefit from GPU acceleration
- Training times can be significant (hours to days depending on environment and hardware)
- Hyperparameter tuning is crucial for success in these complex environments

## References

- [MuJoCo Official Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium MuJoCo Environments](https://gymnasium.farama.org/environments/mujoco/)
- [PPO Paper: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
