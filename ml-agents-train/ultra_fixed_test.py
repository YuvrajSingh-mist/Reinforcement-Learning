"""
Test script for the ultra-fixed AnimalEnv environment that handles channels-first format
"""
import os
import time
import numpy as np
from ultra_fixed_game_env import AnimalEnv

def main():
    print("Ultra-Fixed AnimalEnv Test - Handles Channels-First Format")
    print("="*60)
    
    # Get executable path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    unity_executable = os.path.join(current_dir, "AnimalGame.exe")
    
    print(f"Unity executable: {unity_executable}")
    
    # Create environment
    env = AnimalEnv(
        unity_file_path=unity_executable,
        worker_id=0,
        no_graphics=False,  # Must be False for this Unity build
        time_scale=1.0
    )
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Reset complete. Observation shape: {obs.shape}")
    
    # Verify the observation shape is channels-last (64, 64, 3)
    if obs.shape == (64, 64, 3):
        print("✓ Observation shape is correct (64, 64, 3) - channels last!")
    else:
        print(f"❌ Unexpected observation shape: {obs.shape}, expected (64, 64, 3)")
    
    # Render the initial observation
    print("Rendering initial observation...")
    env.render()
    time.sleep(1)
    
    # Take a few actions
    for i in range(5):
        print(f"\nStep {i+1}")
        action = {"continuous": np.array([0.5]), "discrete_0": i % 2}
        print(f"Action: {action}")
        
        obs, reward, done, info = env.step(action)
        print(f"Step result - Reward: {reward}, Done: {done}, Obs shape: {obs.shape}")
        
        # Verify the observation shape is channels-last (64, 64, 3)
        if obs.shape == (64, 64, 3):
            print("✓ Observation shape is correct (64, 64, 3) - channels last!")
        else:
            print(f"❌ Unexpected observation shape: {obs.shape}, expected (64, 64, 3)")
        
        # Render the observation
        env.render()
        time.sleep(0.5)
        
        if done:
            print("Episode finished. Resetting...")
            obs, info = env.reset()
            print(f"Reset observation shape: {obs.shape}")
            
            # Verify the reset observation shape
            if obs.shape == (64, 64, 3):
                print("✓ Reset observation shape is correct (64, 64, 3) - channels last!")
            else:
                print(f"❌ Unexpected reset observation shape: {obs.shape}, expected (64, 64, 3)")
    
    # Close environment
    env.close()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
