#!/bin/bash

# =================================================================
#  PPO BENCHMARKING SCRIPT
# =================================================================
# This script automates running PPO benchmarks for multiple
# environments and random seeds. It runs the experiments
# sequentially to ensure fair use of resources (especially the GPU)
# and result validity.
#
# To Run:
# 1. Make the script executable:  chmod +x run_benchmark.sh
# 2. Execute it:                ./run_benchmark.sh
# =================================================================

echo "Starting PPO Benchmark Suite..."

# --- Configuration ---
# Define the MuJoCo environments to test on.
# You can easily add more environments here, e.g., "Ant-v4".
ENVS=("HalfCheetah-v5" "Hopper-v5" "Walker2d-v5")

# Define the random seeds to use for each experiment.
# Using multiple seeds is crucial for robust evaluation.
SEEDS=(137)

# Define the names of your Python training scripts.
# Make sure these filenames match your project structure.
SB3_SCRIPT="SB3/half-cheetah-sb3.py"
CUSTOM_PPO_SCRIPT="custom_ppo/half_cheetah.py" # Your custom implementation script


# --- Main Execution Loop ---
# Outer loop iterates through each environment.
for env_id in "${ENVS[@]}"; do
  
  # Inner loop iterates through each seed for the current environment.
  for seed in "${SEEDS[@]}"; do
    
    echo ""
    echo "================================================================="
    echo "  ENVIRONMENT: $env_id  |  SEED: $seed"
    echo "================================================================="
    
    # --- Run Stable Baselines 3 Benchmark ---
    echo ""
    echo "--> Launching Stable Baselines 3 PPO..."
    # Execute the SB3 script, passing the current environment and seed as arguments.
    python $SB3_SCRIPT --env "$env_id" --seed "$seed"
    echo "--> SB3 PPO run finished."
    
    
    # --- Run Your Custom PPO Benchmark ---
    echo ""
    echo "--> Launching Custom PPO..."
    # Execute your custom PPO script with the same arguments.
    # IMPORTANT: Ensure your script is set up to accept --env and --seed arguments.
    python $CUSTOM_PPO_SCRIPT --env_id "$env_id" --seed "$seed" # Assuming your script uses --env_id
    echo "--> Custom PPO run finished."

  done
done

echo ""
echo "================================================================="
echo "All benchmark runs are complete!"
echo "Check your results at: https://wandb.ai/NeatRL"
echo "================================================================="