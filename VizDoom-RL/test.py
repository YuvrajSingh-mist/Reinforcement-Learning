import gymnasium
from vizdoom import gymnasium_wrapper # THIS IS CRUCIAL FOR REGISTERING ENVS
import time
import random

# --- Configuration ---
# List of available environment IDs can be found by looking into:
# from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomScenarioEnv
# print(VizdoomScenarioEnv.metadata["render_modes"]) # to see render modes
# Or checking the vizdoom documentation for gymnasium_wrapper
ENV_ID = "VizdoomBasic-v0"
# Other examples:
# ENV_ID = "VizdoomCorridor-v0"
# ENV_ID = "VizdoomDefendTheCenter-v0"
# ENV_ID = "VizdoomDeathmatch-v0"
# ENV_ID = "VizdoomMyWayHome-v0"
# ENV_ID = "VizdoomPredictPosition-v0"
# ENV_ID = "VizdoomTakeCover-v0"
# ENV_ID = "VizdoomHealthGathering-v0" # or VizdoomHealthGatheringSupreme-v0 for harder
# ENV_ID = "VizdoomDeadlyCorridor-v0"


NUM_EPISODES = 5
RENDER_MODE = "human"  # "human" to watch, "rgb_array" for offscreen, None for no rendering (faster)
# For ViZDoom, "human" mode makes the game window visible.

# Time to sleep between actions to make it observable (in seconds)
# ViZDoom runs at 35 FPS by default, so 1/35 is one game tick.
# The `frame_skip` argument in gymnasium.make can also control this.
ACTION_SLEEP_TIME = 0.05 # s

# --- Initialize Environment ---
print(f"Attempting to create Gymnasium environment: {ENV_ID} with render_mode='{RENDER_MODE}'")

try:
    # You can pass additional arguments to the underlying ViZDoom game via `gymnasium.make`
    # For example: frame_skip=4 (action repeated for 4 game tics)
    
    
    env = gymnasium.make(ENV_ID, render_mode=RENDER_MODE, frame_skip=4)
    env = gymnasium.wrappers.FrameStackObservation(env, stack_size=4)  # Stack 4 frames for better temporal context
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)  # This wrapper records episode statistics like total reward and length
    
    # env = gymnasium.make(ENV_ID, render_mode=RENDER_MODE) # Default frame_skip is usually 1 for wrapper if not specified by scenario
except Exception as e:
    print(f"Error creating environment: {e}")
    print("Please ensure ViZDoom is installed correctly, `vizdoom.gymnasium_wrapper` was imported,")
    print("and a display is available if render_mode='human'.")
    print("On headless servers, you might need Xvfb (e.g., `xvfb-run python your_script.py`).")
    exit()

print("Environment created successfully.")
print(f"Observation space: {env.observation_space}") # Should be a Box for screen pixels
print(f"Action space: {env.action_space}")         # Should be Discrete

# The wrapper usually includes RecordEpisodeStatistics
# if hasattr(env, 'return_queue') and env.return_queue is not None:
# print("RecordEpisodeStatistics wrapper seems to be active.")

# --- Game Loop ---
for i_episode in range(NUM_EPISODES):
    print(f"\n--- Episode {i_episode + 1}/{NUM_EPISODES} ---")

    # Reset the environment to get the initial observation
    # You can pass a seed for reproducibility: observation, info = env.reset(seed=42)
    observation, info = env.reset()
    # print(f"Initial observation shape: {observation.shape if isinstance(observation, np.ndarray) else type(observation)}")

    terminated = False
    truncated = False
    episode_reward = 0
    step_count = 0

    while not (terminated or truncated):
        # env.render() # For some gym envs, render must be called each step.
                       # For ViZDoom with render_mode="human", the window is usually updated by env.step().
                       # Calling it here doesn't hurt.

        # Take a random action from the action space
        action = env.action_space.sample()

        # Perform the action in the environment
        # Returns: next_observation, reward, terminated, truncated, info
        try:
            next_observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during env.step(): {e}")
            break # Exit episode on error

        episode_reward += reward
        step_count += 1

        # print(f"  Step: {step_count}, Action: {action}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
        # print(f"  Next observation shape: {next_observation.shape if isinstance(next_observation, np.ndarray) else type(next_observation)}")

        if ACTION_SLEEP_TIME > 0 and RENDER_MODE == "human":
            time.sleep(ACTION_SLEEP_TIME)

        observation = next_observation

    print(f"Episode finished after {step_count} steps.")
    print(f"Total reward for episode #{i_episode + 1}: {episode_reward:.2f}")
    if "episode" in info: # From RecordEpisodeStatistics wrapper
        print(f"Info from wrapper: Return={info['episode']['r']:.2f}, Length={info['episode']['l']}, Time={info['episode']['t']:.2f}s")


# --- Cleanup ---
print("\nClosing environment.")
env.close()
print("Gymnasium ViZDoom test finished.")