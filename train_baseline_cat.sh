#!/bin/bash
# Script to train Nest-Splatting with baseline and cat modes (hybrid_levels 0-6)
# Usage: ./train_baseline_cat.sh <base_name> [scene_names] [iterations] [data_dir]
#
# Example:
#   ./train_baseline_cat.sh exp1              # Run all scenes
#   ./train_baseline_cat.sh exp1 drums,mic    # Run specific scenes
#   ./train_baseline_cat.sh exp1 all 30000    # Run all scenes with 30k iters

set -e  # Exit on error

# Default values
DEFAULT_ITERATIONS=30000
DEFAULT_DATA_DIR="/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic"
YAML_CONFIG="./configs/nerfsyn.yaml"
ALL_SCENES="chair,drums,ficus,hotdog,lego,materials,mic,ship"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_name> [scene_names] [iterations] [data_dir]"
    echo ""
    echo "Arguments:"
    echo "  base_name     Required. Base name for experiments (e.g., exp1, test)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  data_dir      Optional. Base data directory"
    echo ""
    echo "Examples:"
    echo "  $0 exp1                         # Run all 8 scenes"
    echo "  $0 exp1 drums,mic               # Run only drums and mic"
    echo "  $0 exp1 all 30000               # Run all scenes with 30k iterations"
    echo ""
    echo "Methods trained per scene:"
    echo "  - baseline"
    echo "  - cat0 through cat6 (hybrid_levels 0-6)"
    echo ""
    echo "Available scenes: ${ALL_SCENES}"
    exit 1
fi

BASE_NAME=$1
SCENE_NAMES=${2:-all}
ITERATIONS=${3:-$DEFAULT_ITERATIONS}
DATA_DIR=${4:-$DEFAULT_DATA_DIR}

# Handle "all" keyword
if [ "$SCENE_NAMES" = "all" ]; then
    SCENE_NAMES=$ALL_SCENES
fi

# Convert comma-separated list to array
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"

# Verify YAML config exists
if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting - Baseline & Cat Training"
echo "════════════════════════════════════════════════════════════════════"
echo "Base name:  $BASE_NAME"
echo "Scenes:     ${SCENES[@]}"
echo "Iterations: $ITERATIONS"
echo "Data dir:   $DATA_DIR"
echo "Config:     $YAML_CONFIG"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Global log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR
GLOBAL_LOG_FILE="${LOG_DIR}/train_${BASE_NAME}_${TIMESTAMP}.log"

echo "Logging to: $GLOBAL_LOG_FILE"
echo ""

# Function to run training
run_training() {
    local scene_name=$1
    local method=$2
    local experiment_name=$3
    local extra_args=$4
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    local scene_path="${DATA_DIR}/${scene_name}"
    
    if [ ! -d "$scene_path" ]; then
        echo "WARNING: Scene path does not exist: $scene_path - SKIPPING"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    
    # Path structure: outputs/nerf_synthetic/{scene}/{method}/{name}
    # For cat mode, train.py appends _{hybrid_levels}_levels to the name
    if [ "$method" = "cat" ]; then
        local hl_suffix=$(echo "$extra_args" | grep -oP '(?<=--hybrid_levels )\d+')
        OUTPUT_PATH="outputs/nerf_synthetic/${scene_name}/${method}/${experiment_name}_${hl_suffix}_levels"
    else
        OUTPUT_PATH="outputs/nerf_synthetic/${scene_name}/${method}/${experiment_name}"
    fi
    TEST_METRICS="${OUTPUT_PATH}/test_metrics.txt"
    
    # Check if already completed
    if [ -f "$TEST_METRICS" ]; then
        echo "════════════════════════════════════════════════════════════════════"
        echo "  [$CURRENT_RUN/$TOTAL_RUNS] SKIPPING: ${scene_name} - ${experiment_name}"
        echo "════════════════════════════════════════════════════════════════════"
        echo "Already completed: $TEST_METRICS"
        echo ""
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [$CURRENT_RUN/$TOTAL_RUNS] Training: ${scene_name} - ${experiment_name}"
    echo "════════════════════════════════════════════════════════════════════"
    
    CMD="python train.py -s $scene_path -m $experiment_name --yaml $YAML_CONFIG --eval --iterations $ITERATIONS --method $method $extra_args"
    
    echo "Command: $CMD"
    echo "Started: $(date)"
    echo ""
    
    $CMD 2>&1 | tee -a $GLOBAL_LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Completed: ${scene_name} - ${experiment_name}"
        echo "Finished: $(date)"
        echo ""
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo "✗ FAILED: ${scene_name} - ${experiment_name} (exit code: $EXIT_CODE)"
        echo "Finished: $(date)"
        echo ""
        FAILED=$((FAILED + 1))
    fi
}

# Calculate total runs: baseline + cat0-6 = 8 methods per scene
NUM_SCENES=${#SCENES[@]}
METHODS_PER_SCENE=8  # baseline + cat0-6
TOTAL_RUNS=$((NUM_SCENES * METHODS_PER_SCENE))
CURRENT_RUN=0
COMPLETED=0
SKIPPED=0
FAILED=0

echo "Total experiments: $TOTAL_RUNS (${NUM_SCENES} scenes × ${METHODS_PER_SCENE} methods)"
echo ""

# ============================================================================
# TRAINING LOOP
# ============================================================================
for scene in "${SCENES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SCENE: ${scene}"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    # 1. BASELINE MODE
    run_training "$scene" "baseline" "${BASE_NAME}_baseline" ""
    
    # 2. CAT MODE - hybrid_levels 0 to 6
    for hl in {0..6}; do
        run_training "$scene" "cat" "${BASE_NAME}_cat${hl}" "--hybrid_levels $hl"
    done
    
    echo ""
    echo "  Completed all methods for scene: ${scene}"
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════════════"
echo "Base name:       $BASE_NAME"
echo "Scenes:          ${SCENES[@]}"
echo "Total runs:      $TOTAL_RUNS"
echo "Completed:       $COMPLETED"
echo "Skipped:         $SKIPPED"
echo "Failed:          $FAILED"
echo "Log file:        $GLOBAL_LOG_FILE"
echo "════════════════════════════════════════════════════════════════════"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "⚠ WARNING: $FAILED experiments failed. Check the log file for details."
    echo ""
fi

# Generate results table
echo "Generating results tables..."
python create_results_table.py --base_name "$BASE_NAME" --scenes "${SCENES[*]// /,}"

echo ""
echo "Done! Check metrics_tables/ for results."
echo ""


