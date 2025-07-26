import torch
from ippo import Actor, Config
import gymnasium as gym
from pettingzoo.atari import pong_v3
import supersuit
import argparse
import imageio

# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="Play Pong with trained IPPO agent.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to play')
    parser.add_argument('--record', action='store_true', help='Record gameplay to MP4')
    parser.add_argument('--output', type=str, default='ippo_pong_play.mp4', help='Output MP4 filename')
    return parser.parse_args()

# --- Main ---
def main():
    args = parse_args()
    # Load config and model
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    config = Config()
    actor = Actor(config).eval()
    actor.load_state_dict(state_dict["model_state"])

    env = pong_v3.env(render_mode="rgb_array")
    env = supersuit.frame_stack_v1(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.color_reduction_v0(env, mode='full')
    env = supersuit.agent_indicator_v0(env)
    env = supersuit.resize_v1(env, 64, 64)
    env.reset()

    frames = []
    for ep in range(args.episodes):
        env.reset()
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        obs = {agent: env.observe(agent) for agent in env.agents}
        while not all(terminated.values()) and not all(truncated.values()):
            actions = {}
            for agent in env.agents:
                if not terminated[agent] and not truncated[agent]:
                    obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = actor(obs_tensor, agent)
                        action = torch.argmax(logits, dim=-1).item()
                    actions[agent] = action
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            obs = next_obs
            if args.record:
                frames.append(env.render())
    if args.record and frames:
        imageio.mimsave(args.output, frames, fps=30)
        print(f"Saved gameplay to {args.output}")
    env.close()

if __name__ == "__main__":
    main()
