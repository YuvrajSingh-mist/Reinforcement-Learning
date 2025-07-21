import torch
import numpy as np
import cv2
import sys
import time
from pettingzoo.atari import pong_v3
import supersuit as ss

# Import your Agent class from the correct path
from MARL.ippo import Agent

# --- Load Model ---
def load_agent(model_path, action_space):
    agent = Agent(action_space)
    agent.load_state_dict(torch.load(model_path, map_location="cuda")['model_state_dict'])
    agent.eval()
    return agent

# --- Preprocessing ---
def preprocess_obs(obs):
    # obs: (H, W, C) or (C, H, W) -> (1, H, W, C)
    if obs.shape[-1] != 6:
        obs = np.repeat(obs, 6 // obs.shape[-1], axis=-1)
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return obs

# --- Main Play Loop ---
def play_pong(model_path):
    env = pong_v3.env(render_mode="rgb_array")
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)

    possible_agents = env.possible_agents
    ai_agent_name = possible_agents[0]  # 'first_0'
    human_agent_name = possible_agents[1]  # 'second_0'

    env.reset(seed=42)
    obs, _, _, _, _ = env.last()
    action_space = env.action_space(ai_agent_name).n
    agent = load_agent(model_path, action_space)

    done = False
    while not done:
        for agent_name in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated
            if done:
                env.step(None)
                continue
            if agent_name == ai_agent_name:
                # obs_tensor = preprocess_obs(obs).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    action, _, _ = agent.get_action(obs, deterministic=True)
                    action = action.cpu().item()
            else:
                # Human plays using keyboard
                frame = env.render()
                # Display controls on the frame
                controls = "Controls: W=Right, S=Left, F=Fire, D=Fire Right, A=Fire Left, Q=Quit"
                cv2.putText(frame, controls, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow("Pong - You are Player 2", frame)
                key = cv2.waitKey(50) & 0xFF
                
                # Map keys to actions
                if key == ord('q'):  # Quit
                    print("Quitting game...")
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord('w') or key == 82:  # Up arrow or W
                    action = 2  # Move right
                elif key == ord('s') or key == 84:  # Down arrow or S
                    action = 3  # Move left
                elif key == ord('f'):  # F key
                    action = 1  # Fire
                elif key == ord('d'):  # D key
                    action = 4  # Fire right
                elif key == ord('a'):  # A key
                    action = 5  # Fire left
                else:
                    action = 0  # NOOP
            env.step(action)
            if done:
                break
    print("Game Over!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    play_pong(model_path)
