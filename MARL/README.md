# Multi-Agent Self-Play with PPO on Atari Pong

This mini-project demonstrates MARL algorithms, implemented and trained on the PettingZoo Atari environment **`pong_v3`** using PPO. The implementation relies on PyTorch, PettingZoo, SuperSuit, and wandb for experiment tracking.

<p align="center">
  <img src="https://github.com/PettingZoo-Team/PettingZoo/raw/master/imgs/pong.gif" width="400"/>
</p>

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Hyper-parameters](#hyper-parameters)
4. [Training Details](#training-details)
5. [IPPO](#ippo)
6. [MAPPO](#mappo)
7. [Self Play](#self-play)
8. [Evaluation](#evaluation)
9. [Saving & Loading Checkpoints](#saving--loading-checkpoints)
10. [Dependencies](#dependencies)
11. [References](#references)

---

## Project Structure
```
MARL/
├── ippo.py              # Core IPPO implementation (training & evaluation)
├── train.py             # Convenience wrapper around ippo.py
├── self_play.py         # Alternate self-play driver (optional)
├── Self Play/
│   └── play.py          # Simple script to watch two trained agents compete
├── pt files/            # Saved `.pt` checkpoints
└── README.md            # ← you are here
```

## Quick Start
1. **Install requirements** (create a new environment first):
   ```bash
   pip install -r requirements.txt   # provides torch, pettingzoo, supersuit, wandb, tqdm, imageio …
   ```

2. **Train** (16 parallel envs, 10M timesteps by default):
   ```bash
   python train.py --env_id pong_v3 --total_timesteps 10000000
   ```
   TensorBoard logs are written to `runs/`, and (optionally) wandb will sync metrics if `use_wandb=True` in `Config`.

3. **Watch evaluation** (renders to a window and saves an MP4 when `capture_video=True`):
   ```bash
   python ippo.py --eval --checkpoint "pt files/Pong-MARL.pt"
   ```

## Hyper-parameters
Key parameters live in the `Config` class inside `ippo.py`:
* `lr`: learning rate (default **2.5e-4**)
* `num_envs`: number of parallel envs (default **16**)
* `max_steps`: rollout length (**128**)
* `PPO_EPOCHS`, `num_minibatches`, `clip_coeff`, `entropy_coeff`, etc.
* `GAE`: λ for Generalised Advantage Estimation (**0.95**)
* `anneal_lr`: linearly decay LR to 0

Tweak these via **CLI args** or editing the config directly.

## Training Details
* **Observation preprocessing**: grayscale, resize to 84×84, 4-frame stack, agent indicator channel, then downsampled again to 64×64 before entering the CNN.
* **Network**: shared conv tower → 512-unit MLP. Separate **actor** logits and **critic** value head per agent.
* **Self-play**: agents interact in the same vectorised env. Rewards are stored separately and GAE/returns are computed per-agent.
* **Optimisation**: Adam with gradient clipping (0.5) + orthogonal initialisation.

## IPPO
`ippo.py` contains the core Independent Proximal Policy Optimisation implementation used in this repo.

Key features:
* Vectorised PettingZoo environments with automated preprocessing (`supersuit`).
* Shared convolutional encoder with agent-specific heads.
* GAE advantage estimation, PPO clipping, entropy regularisation.
* Linear learning-rate annealing and gradient norm logging.
* Periodic checkpointing and TensorBoard/WandB instrumentation.

* **Observation preprocessing**: grayscale, resize to 84×84, 4-frame stack, agent indicator channel, then downsampled again to 64×64 before entering the CNN.
* **Network**: shared conv tower → 512-unit MLP. Separate **actor** logits and **critic** value head per agent.
* **Self-play**: agents interact in the same vectorised env. Rewards are stored separately and GAE/returns are computed per-agent.
* **Optimisation**: Adam with gradient clipping (0.5) + orthogonal initialisation.

<p align="center">
  <img src="image.png" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/PettingZoo-Team/PettingZoo/raw/master/imgs/pong.gif" width="400"/>
</p>

### IPPO WandB Report
[![IPPO WandB Report](https://img.shields.io/badge/WandB-IPPO-blue?logo=wandb)](https://wandb.ai/rentio/cleanRL/reports/IPPO-Pong--VmlldzoxMzcxMDgwMA?accessToken=pcozaxci9gcrt0z2k43layaab7huqpjkt5goorcxwspjbupherpw0qhyu1iajlqy)

---

## MAPPO
`MAPPO` (Multi-Agent Proximal Policy Optimization) is a centralized variant of PPO for multi-agent environments. It leverages shared experience and centralized training for improved stability and performance in cooperative or competitive settings.

* Centralized critic for better credit assignment.
* Shared policy or separate policies per agent.
* Suitable for environments with many agents or complex interactions.
* Can be implemented using PettingZoo and PyTorch, similar to IPPO.

---

## Evaluation
The `evaluate` function rolls out **`num_eval_eps`** episodes deterministically (greedy actions) and logs:
* per-episode raw rewards for each agent;
* average returns (`avg_return1`, `avg_return2`).

Setting `record=True` captures frames and stitches them into an MP4 (wandb video if enabled).

## Self Play
Self-play utilities live in the `Self Play/` directory and allow you to evaluate or continue training agents against each other.

| File | Purpose |
|------|---------|
| `Self Play/play.py` | Render two trained agents (or human vs. agent) in real-time. |
| `Self Play/self_play.py` | Alternate driver script for self-play training experiments. |
| `Self Play/pt files/` | Example checkpoint `Pong-MARL.pt`. |

### Quick Demo
Run a head-to-head match:
```bash
python "Self Play/play.py" --checkpoint "pt files/Pong-MARL.pt"
```

### WandB Report
[![WandB Report](https://img.shields.io/badge/WandB-Report-blue?logo=wandb)](https://api.wandb.ai/links/rentio/a74ndy24)

The report aggregates self-play learning curves, matchup scores, and evaluation videos.
## Saving & Loading Checkpoints
`save_checkpoint` serialises the actor + optimizer state every 200 updates (and at finish) under `pt files/`.  To resume:
```python
state_dict = torch.load("pt files/Pong-MARL.pt")
actor.load_state_dict(state_dict["model_state"])
```

## Dependencies
* Python ≥ 3.9
* PyTorch ≥ 2.0
* gymnasium
* pettingzoo[atari]
* supersuit
* wandb (optional)
* tqdm, imageio, opencv-python
