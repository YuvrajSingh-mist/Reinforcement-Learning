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
ENVS=("HalfCheetah-v5" "Hopper-v5" "Walker2d-v5" "Ant-v5" "Humanoid-v5"
    "HumanoidStandup-v5"
    "InvertedDoublePendulum-v5"
    "InvertedPendulum-v5"
    "Pusher-v5"
    "Reacher-v5"
    "Swimmer-v5")

# Define the random seeds to use for each experiment.
# Using multiple seeds is crucial for robust evaluation.
SEEDS=(42 1337 91)

# Define the run number for this benchmark iteration.
# This helps distinguish between multiple benchmark runs.
RUN_NUMBER=8

# Define the names of your Python training scripts.
# Make sure these filenames match your project structure.
SB3_SCRIPT="SB3/half-cheetah-sb3.py"
CUSTOM_PPO_SCRIPT="Custom/half-cheetah-v2.py" # Your custom implementation script

# --- Pre-flight Checks ---
echo "Checking if required scripts exist..."

if [ ! -f "$SB3_SCRIPT" ]; then
    echo "ERROR: SB3 script not found at: $SB3_SCRIPT"
    echo "Please check the file path and ensure the script exists."
    exit 1
fi
echo "✓ Found SB3 script: $SB3_SCRIPT"

if [ ! -f "$CUSTOM_PPO_SCRIPT" ]; then
    echo "ERROR: Custom PPO script not found at: $CUSTOM_PPO_SCRIPT"
    echo "Please check the file path and ensure the script exists."
    exit 1
fi
echo "✓ Found Custom PPO script: $CUSTOM_PPO_SCRIPT"

echo "All required scripts found. Starting benchmark..."

# --- Calculate Total Experiments ---
TOTAL_ENVS=${#ENVS[@]}
TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_EXPERIMENTS=$((TOTAL_ENVS * TOTAL_SEEDS * 2))  # 2 = SB3 + Custom
echo ""
echo "📊 BENCHMARK OVERVIEW:"
echo "   • Environments: ${TOTAL_ENVS} (${ENVS[*]})"
echo "   • Seeds per env: ${TOTAL_SEEDS} (${SEEDS[*]})"
echo "   • Run number: ${RUN_NUMBER}"
echo "   • Algorithms: 2 (Custom-PPO, SB3-PPO)"
echo "   • Total experiments: ${TOTAL_EXPERIMENTS}"
echo ""

# --- Main Execution Loop ---
CURRENT_EXPERIMENT=0
ENV_COUNT=0

echo ""
echo "🔥 PHASE 1: RUNNING ALL CUSTOM-PPO EXPERIMENTS"
echo "================================================================="

# PHASE 1: Run ALL Custom PPO experiments first
for env_id in "${ENVS[@]}"; do
  ENV_COUNT=$((ENV_COUNT + 1))
  SEED_COUNT=0
  
  echo ""
  echo "🎯 CUSTOM-PPO ENVIRONMENT ${ENV_COUNT}/${TOTAL_ENVS}: ${env_id}"
  echo "================================================================="
  
  for seed in "${SEEDS[@]}"; do
    SEED_COUNT=$((SEED_COUNT + 1))
    
    echo ""
    echo "🌱 CUSTOM-PPO SEED ${SEED_COUNT}/${TOTAL_SEEDS}: ${seed}"
    echo "----------------------------------------------------------------"
    
    # --- Run Your Custom PPO Benchmark ---
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    echo ""
    echo "🚀 [${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Launching Custom-PPO..."
    echo "   Environment: ${env_id} | Seed: ${seed} | Run: ${RUN_NUMBER}"
    
    START_TIME=$(date +%s)
    python $CUSTOM_PPO_SCRIPT --env_id "$env_id" --seed "$seed" --run "$RUN_NUMBER"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "✅ [${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Custom-PPO completed in ${DURATION}s"
    
    # Progress summary
    COMPLETED_EXPERIMENTS=$CURRENT_EXPERIMENT
    REMAINING_EXPERIMENTS=$((TOTAL_EXPERIMENTS - COMPLETED_EXPERIMENTS))
    PROGRESS_PERCENT=$((COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))
    
    echo ""
    echo "📈 PROGRESS: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS} experiments (${PROGRESS_PERCENT}%) | ${REMAINING_EXPERIMENTS} remaining"

  done
  
  echo ""
  echo "🏁 CUSTOM-PPO ENVIRONMENT ${ENV_COUNT}/${TOTAL_ENVS} COMPLETE: ${env_id}"
  echo "================================================================="
done

echo ""
echo "✅ PHASE 1 COMPLETE: All Custom-PPO experiments finished!"
echo ""
echo "🔥 PHASE 2: RUNNING ALL SB3-PPO EXPERIMENTS"
echo "================================================================="

# PHASE 2: Run ALL SB3 experiments second
ENV_COUNT=0
for env_id in "${ENVS[@]}"; do
  ENV_COUNT=$((ENV_COUNT + 1))
  SEED_COUNT=0
  
  echo ""
  echo "🎯 SB3-PPO ENVIRONMENT ${ENV_COUNT}/${TOTAL_ENVS}: ${env_id}"
  echo "================================================================="
  
  for seed in "${SEEDS[@]}"; do
    SEED_COUNT=$((SEED_COUNT + 1))
    
    echo ""
    echo "🌱 SB3-PPO SEED ${SEED_COUNT}/${TOTAL_SEEDS}: ${seed}"
    echo "----------------------------------------------------------------"
    
    # --- Run Stable Baselines 3 Benchmark ---
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    echo ""
    echo "🚀 [${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] Launching SB3-PPO..."
    echo "   Environment: ${env_id} | Seed: ${seed} | Run: ${RUN_NUMBER}"
    
    START_TIME=$(date +%s)
    python $SB3_SCRIPT --env "$env_id" --seed "$seed" --run "$RUN_NUMBER"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "✅ [${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] SB3-PPO completed in ${DURATION}s"
    
    # Progress summary
    COMPLETED_EXPERIMENTS=$CURRENT_EXPERIMENT
    REMAINING_EXPERIMENTS=$((TOTAL_EXPERIMENTS - COMPLETED_EXPERIMENTS))
    PROGRESS_PERCENT=$((COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))
    
    echo ""
    echo "📈 PROGRESS: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS} experiments (${PROGRESS_PERCENT}%) | ${REMAINING_EXPERIMENTS} remaining"

  done
  
  echo ""
  echo "🏁 SB3-PPO ENVIRONMENT ${ENV_COUNT}/${TOTAL_ENVS} COMPLETE: ${env_id}"
  echo "================================================================="
done

echo ""
echo "✅ PHASE 2 COMPLETE: All SB3-PPO experiments finished!"

# Calculate total benchmark time
BENCHMARK_END_TIME=$(date +%s)
TOTAL_BENCHMARK_TIME=$((BENCHMARK_END_TIME - $(date +%s)))

echo ""
echo "🎉 SUCCESS! All benchmark runs are complete!"
echo "================================================================="
echo "📊 FINAL SUMMARY:"
echo "   • Total experiments completed: ${TOTAL_EXPERIMENTS}"
echo "   • Environments tested: ${TOTAL_ENVS} (${ENVS[*]})"
echo "   • Seeds per environment: ${TOTAL_SEEDS}"
echo "   • Algorithms compared: Custom-PPO vs SB3-PPO"
echo ""
echo "🔗 View your results at: https://wandb.ai/your-username/NeatRL"
echo "💡 Tip: Use the W&B dashboard to compare SB3 vs Custom performance!"
echo "================================================================="