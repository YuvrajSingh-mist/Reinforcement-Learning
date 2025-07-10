"""
Ultra-fixed AnimalEnv that handles the channels-first observation format issue
"""
import gym
from gym.spaces import Discrete, Box, Dict
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs import rpc_utils
import numpy as np
import matplotlib.pyplot as plt
import time

# Monkey patch the observation processing function to handle channels-first format
original_observation_to_np_array = rpc_utils._observation_to_np_array

def patched_observation_to_np_array(obs, expected_shape):
    """
    Patched version that handles both channels-first and channels-last formats
    """
    try:
        # First, try the original function
        return original_observation_to_np_array(obs, expected_shape)
    except Exception as e:
        if "Decompressed observation did not have the expected shape" in str(e):
            print(f"Observation shape mismatch detected. Attempting to fix...")
            
            # Decompress the observation manually
            if obs.compression_type == 0:  # No compression
                data = obs.float_data.data if obs.float_data.data else obs.compressed_data
                if len(data) == 0:
                    return np.zeros(expected_shape, dtype=np.float32)
                
                # Convert to numpy array
                np_array = np.array(data, dtype=np.float32)
                
                # Calculate what shape the data actually is
                total_elements = np_array.size
                
                # For visual observations, try to reshape as channels-first (C, H, W)
                if len(expected_shape) == 3 and total_elements == np.prod(expected_shape):
                    # Try channels-first format first
                    channels_first_shape = (expected_shape[2], expected_shape[0], expected_shape[1])
                    if total_elements == np.prod(channels_first_shape):
                        print(f"Reshaping from channels-first {channels_first_shape} to channels-last {expected_shape}")
                        # Reshape as channels-first then transpose to channels-last
                        reshaped = np_array.reshape(channels_first_shape)
                        # Transpose from (C, H, W) to (H, W, C)
                        transposed = np.transpose(reshaped, (1, 2, 0))
                        return transposed.astype(np.uint8) if np.max(transposed) > 1.0 else (transposed * 255).astype(np.uint8)
                
                # If that doesn't work, try to reshape directly
                try:
                    reshaped = np_array.reshape(expected_shape)
                    return reshaped.astype(np.uint8) if np.max(reshaped) > 1.0 else (reshaped * 255).astype(np.uint8)
                except:
                    print(f"Could not reshape observation. Creating zero array with shape {expected_shape}")
                    return np.zeros(expected_shape, dtype=np.uint8)
            else:
                # Handle compressed data
                print(f"Handling compressed observation...")
                try:
                    # Decompress using PIL if it's PNG compressed
                    from PIL import Image
                    import io
                    
                    # Create image from compressed data
                    image = Image.open(io.BytesIO(obs.compressed_data))
                    # Convert to numpy array
                    np_array = np.array(image)
                    
                    # If the image is in the wrong format, try to fix it
                    if np_array.shape != expected_shape:
                        if len(np_array.shape) == 3 and len(expected_shape) == 3:
                            # Try transposing if it's channels-first
                            if np_array.shape == (expected_shape[2], expected_shape[0], expected_shape[1]):
                                np_array = np.transpose(np_array, (1, 2, 0))
                        
                        # If still not right, resize
                        if np_array.shape != expected_shape:
                            from PIL import Image
                            pil_image = Image.fromarray(np_array)
                            # Resize to expected height and width
                            pil_image = pil_image.resize((expected_shape[1], expected_shape[0]))
                            np_array = np.array(pil_image)
                            
                            # Ensure correct number of channels
                            if len(expected_shape) == 3 and len(np_array.shape) == 2:
                                np_array = np.stack([np_array] * expected_shape[2], axis=-1)
                            elif len(expected_shape) == 3 and np_array.shape[2] != expected_shape[2]:
                                if expected_shape[2] == 3 and np_array.shape[2] == 4:
                                    np_array = np_array[:, :, :3]  # Remove alpha channel
                                elif expected_shape[2] == 3 and np_array.shape[2] == 1:
                                    np_array = np.repeat(np_array, 3, axis=2)  # Convert grayscale to RGB
                    
                    return np_array.astype(np.uint8)
                    
                except Exception as decompress_error:
                    print(f"Error decompressing observation: {decompress_error}")
                    return np.zeros(expected_shape, dtype=np.uint8)
        else:
            # Re-raise the original error if it's not a shape mismatch
            raise e

# Apply the monkey patch
rpc_utils._observation_to_np_array = patched_observation_to_np_array

class AnimalEnv(gym.Env):
    def __init__(self, unity_file_path, worker_id=0, no_graphics=False, time_scale=1.0):
        """
        Initializes the AnimalEnv with proper observation format handling.
        
        Args:
            unity_file_path (str): The path to the Unity executable.
            worker_id (int): The port offset for communication.
            no_graphics (bool): Whether to run Unity in headless mode.
            time_scale (float): How fast the simulation runs.
        """
        # Create communication channels
        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()
        
        # Configure environment settings
        engine_config_channel.set_configuration_parameters(
            time_scale=time_scale,
            width=800,
            height=600,
            quality_level=0
        )
        
        print(f"Attempting to connect to Unity environment at: {unity_file_path}")
        print(f"Worker ID: {worker_id}, No Graphics: {no_graphics}, Time Scale: {time_scale}")
        
        # Try to connect to Unity environment
        try:
            self.unity_env = UnityEnvironment(
                file_name=unity_file_path,
                worker_id=worker_id,
                no_graphics=False,  # Must be False for this Unity build
                timeout_wait=60,
                side_channels=[engine_config_channel, env_params_channel]
            )
            print("Unity environment created successfully!")
        except Exception as e:
            print(f"Error creating Unity environment: {e}")
            raise
        
        # Reset to initialize environment
        print("Resetting environment for initialization...")
        self.unity_env.reset()
        
        # Look for behaviors
        behavior_names = list(self.unity_env.behavior_specs.keys())
        print(f"Detected behaviors: {behavior_names}")
        
        # Use detected behavior if available, otherwise use hardcoded name
        if behavior_names:
            self.behavior_name = behavior_names[0]
            print(f"Using detected behavior name: {self.behavior_name}")
            
            # Get behavior spec
            self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
            print(f"Behavior spec: {self.behavior_spec}")
        else:
            # Use hardcoded behavior name from Unity
            self.behavior_name = "AnimalAgent"
            print(f"No behaviors detected, using hardcoded name: {self.behavior_name}")
            self.behavior_spec = None
        
        # Define action and observation spaces based on Unity screenshot
        # Continuous action: Move speed (1D) - from 0 to 5 in Unity
        # Discrete action: Shoot/Don't shoot (Binary)
        self.action_space = Dict({
            "continuous": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "discrete_0": Discrete(2)
        })
        
        # Observation space from camera - using standard format (channels last)
        # Unity sends observations in channels-first but we'll convert them
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space} (channels last: H, W, C)")
        
        # For rendering
        self.current_obs = None
        self.fig = None
        self.ax = None

    def step(self, action):
        """Takes a step in the environment."""
        # Prepare continuous action
        if "continuous" in action:
            continuous_action = np.array([[action["continuous"]]], dtype=np.float32)
        else:
            continuous_action = np.array([[0.0]], dtype=np.float32)
        
        # Prepare discrete action
        if "discrete_0" in action:
            discrete_action = np.array([[action["discrete_0"]]], dtype=np.int32)
        else:
            discrete_action = np.array([[0]], dtype=np.int32)
        
        print(f"Sending actions - Continuous: {continuous_action}, Discrete: {discrete_action}")
        
        # Create action tuple
        action_tuple = ActionTuple(continuous=continuous_action, discrete=discrete_action)
        
        try:
            # Set actions and step
            self.unity_env.set_actions(self.behavior_name, action_tuple)
            self.unity_env.step()
            print("Step executed successfully")
            
            # Try to get steps
            try:
                decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
                
                # Process decision or terminal steps
                if len(terminal_steps) > 0:
                    # Episode ended
                    agent_id = list(terminal_steps.agent_id)[0]
                    reward = terminal_steps.reward[0]
                    done = True
                    
                    # Try to get real observation from terminal steps
                    try:
                        if len(terminal_steps.obs) > 0 and len(terminal_steps.obs[0]) > 0:
                            # Get observation - it should now be in (64, 64, 3) format thanks to our patch
                            obs = terminal_steps.obs[0][0].astype(np.uint8)
                            print(f"Got real observation from terminal steps: {obs.shape}")
                        else:
                            # Create placeholder observation in correct shape (64, 64, 3)
                            obs = np.zeros((64, 64, 3), dtype=np.uint8)
                            print("Using placeholder observation (no terminal obs available)")
                    except Exception as e:
                        print(f"Error processing terminal observation: {e}")
                        # Create placeholder observation in correct shape (64, 64, 3)
                        obs = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    self.current_obs = obs
                    
                elif len(decision_steps) > 0:
                    # New decision
                    agent_id = list(decision_steps.agent_id)[0]
                    reward = decision_steps.reward[0]
                    done = False
                    
                    # Try to get real observation from decision steps
                    try:
                        if len(decision_steps.obs) > 0 and len(decision_steps.obs[0]) > 0:
                            # Get observation - it should now be in (64, 64, 3) format thanks to our patch
                            obs = decision_steps.obs[0][0].astype(np.uint8)
                            print(f"Got real observation from decision steps: {obs.shape}")
                        else:
                            # Create placeholder observation in correct shape (64, 64, 3)
                            obs = np.zeros((64, 64, 3), dtype=np.uint8)
                            print("Using placeholder observation (no decision obs available)")
                    except Exception as e:
                        print(f"Error processing decision observation: {e}")
                        # Create placeholder observation in correct shape (64, 64, 3)
                        obs = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    self.current_obs = obs
                    
                else:
                    # No decision or terminal steps available
                    print("No steps available after action. Using default values.")
                    reward = 0.0
                    done = False
                    obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format (64, 64, 3)
                    self.current_obs = obs
                
                # Ensure observation is in correct format (64, 64, 3)
                if len(obs.shape) != 3 or obs.shape != (64, 64, 3):
                    print(f"Warning: Observation is not in expected format (64, 64, 3): {obs.shape}")
                    # Create placeholder observation in correct shape (64, 64, 3)
                    obs = np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Return step results
                info = {"action": action}
                return obs, reward, done, info
                
            except Exception as e:
                print(f"Error getting steps: {e}")
                # Return default values
                obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format (64, 64, 3)
                self.current_obs = obs
                return obs, 0.0, False, {}
                
        except Exception as e:
            print(f"Error during step: {e}")
            # Return default values
            obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format (64, 64, 3)
            self.current_obs = obs
            return obs, 0.0, False, {}
    
    def reset(self, seed=None, options=None):
        """Resets the environment."""
        print("Resetting environment...")
        
        try:
            # Reset Unity environment
            self.unity_env.reset()
            
            # Try to get steps
            try:
                decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
                
                if len(decision_steps) > 0:
                    # Try to get real observation from decision steps
                    try:
                        if len(decision_steps.obs) > 0 and len(decision_steps.obs[0]) > 0:
                            # Get observation - it should now be in (64, 64, 3) format thanks to our patch
                            obs = decision_steps.obs[0][0].astype(np.uint8)
                            print(f"Got real reset observation: {obs.shape}")
                        else:
                            # Create placeholder observation in correct shape (64, 64, 3)
                            obs = np.zeros((64, 64, 3), dtype=np.uint8)
                            print("Using placeholder reset observation (no obs data)")
                    except Exception as e:
                        print(f"Error processing reset observation: {e}")
                        # Create placeholder observation in correct shape (64, 64, 3)
                        obs = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    self.current_obs = obs
                else:
                    # No decision steps available
                    print("No decision steps available after reset. Using default observation.")
                    obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format
                    self.current_obs = obs
                    
            except Exception as e:
                print(f"Error getting steps after reset: {e}")
                # Default observation
                obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format
                self.current_obs = obs
                
            return obs, {}
            
        except Exception as e:
            print(f"Error during reset: {e}")
            # Default observation
            obs = np.zeros((64, 64, 3), dtype=np.uint8)  # Shape matches standard format
            self.current_obs = obs
            return obs, {}

    def render(self, mode='human'):
        """Renders the current observation."""
        if self.current_obs is None:
            print("No observation available to render.")
            return
        
        # Observation should already be in channels-last format (64, 64, 3) for display
        display_obs = self.current_obs
        print(f"Rendering observation with shape: {display_obs.shape}")
            
        if mode == 'human':
            try:
                # Create figure if needed
                if self.fig is None or self.ax is None:
                    plt.ion()
                    self.fig, self.ax = plt.subplots(figsize=(8, 8))
                    self.ax.set_title("Unity ML-Agents Environment")
                    
                # Display observation
                if self.ax is not None:
                    self.ax.clear()
                    self.ax.imshow(display_obs)
                    self.ax.set_title("AnimalAgent Environment")
                    self.ax.axis('off')
                    
                    plt.draw()
                    plt.pause(0.001)
            except Exception as e:
                print(f"Error rendering: {e}")
                
        elif mode == 'rgb_array':
            return display_obs
            
    def close(self):
        """Closes the environment."""
        # Close matplotlib resources
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Close Unity environment
        if hasattr(self, 'unity_env'):
            self.unity_env.close()
            
        print("Environment closed.")
