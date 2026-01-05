#!/bin/bash
# Script to train Nest-Splatting with baseline and cat modes (hybrid_levels 0-6)
# Usage: ./train_baseline_cat.sh <dataset> <base_name> [scene_names] [iterations] [extra_args]
#
# Example:
#   ./train_baseline_cat.sh nerf_synthetic exp1              # Run all nerf_synthetic scenes
#   ./train_baseline_cat.sh DTU exp1 scan24,scan37           # Run specific DTU scenes
#   ./train_baseline_cat.sh mip_360 exp1 all 30000           # Run all mip_360 scenes with 30k iters
#   ./train_baseline_cat.sh nerf_synthetic exp1 all 30000 "--disable_c2f"  # With extra args

set -e  # Exit on error

# Default values
DEFAULT_ITERATIONS=30000
BASE_DATA_DIR="/home/nilkel/Projects/data/nest_synthetic"
DTU_DATA_DIR="/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU"
MIP360_DATA_DIR="/home/nilkel/Projects/data/mip360"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <base_name> [scene_names] [iterations] [extra_args]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Required. Dataset name: nerf_synthetic, DTU, or mip_360"
    echo "  base_name     Required. Base name for experiments (e.g., exp1, test)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  extra_args    Optional. Extra arguments to pass to train.py (e.g., '--disable_c2f')"
    echo ""
    echo "Examples:"
    echo "  $0 nerf_synthetic exp1                       # Run all 8 nerf_synthetic scenes"
    echo "  $0 DTU exp1 scan24,scan37                    # Run specific DTU scenes"
    echo "  $0 mip_360 exp1 all 30000                    # Run all mip_360 scenes with 30k iters"
    echo "  $0 nerf_synthetic exp1 all 30000 \"--disable_c2f\"  # With --disable_c2f flag"
    echo "  $0 DTU exp1 scan24 30000 \"--mcmc --cap_max 300000\"  # With MCMC mode"
    echo ""
    echo "Methods trained per scene:"
    echo "  - baseline"
    echo "  - cat0 through cat6 (hybrid_levels 0-6)"
    echo ""
    echo "Datasets:"
    echo "  nerf_synthetic: chair, drums, ficus, hotdog, lego, materials, mic, ship"
    echo "  DTU: scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122"
    echo "  mip_360: bicycle, bonsai, counter, garden, kitchen, room, stump"
    exit 1
fi

DATASET=$1
BASE_NAME=$2
SCENE_NAMES=${3:-all}
ITERATIONS=${4:-$DEFAULT_ITERATIONS}
EXTRA_ARGS=${5:-""}

# Configure dataset-specific settings
case "$DATASET" in
    nerf_synthetic)
        DATA_DIR="${BASE_DATA_DIR}/nerf_synthetic"
        YAML_CONFIG="./configs/nerfsyn.yaml"
        ALL_SCENES="chair,drums,ficus,hotdog,lego,materials,mic,ship"
        DATASET_PATH="nerf_synthetic"
        RESOLUTION_ARG=""
        ;;
    DTU)
        DATA_DIR="$DTU_DATA_DIR"
        YAML_CONFIG="./configs/dtu.yaml"
        ALL_SCENES="scan24,scan37,scan40,scan55,scan63,scan65,scan69,scan83,scan97,scan105,scan106,scan110,scan114,scan118,scan122"
        DATASET_PATH="DTU"
        RESOLUTION_ARG="-r 2"  # DTU uses resolution 2
        ;;
    mip_360)
        DATA_DIR="$MIP360_DATA_DIR"
        YAML_CONFIG="./configs/360_outdoor.yaml"  # Default to outdoor, can be overridden
        ALL_SCENES="bicycle,bonsai,counter,garden,kitchen,room,stump"
        DATASET_PATH="mip_360"
        RESOLUTION_ARG="-r 2"  # mip_360 uses resolution 2
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "Available datasets: nerf_synthetic, DTU, mip_360"
        exit 1
        ;;
esac

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
echo "Dataset:     $DATASET"
echo "Base name:   $BASE_NAME"
echo "Scenes:      ${SCENES[@]}"
echo "Iterations:  $ITERATIONS"
echo "Data dir:    $DATA_DIR"
echo "Config:      $YAML_CONFIG"
if [ -n "$RESOLUTION_ARG" ]; then
echo "Resolution:  ${RESOLUTION_ARG#-r }"
fi
if [ -n "$EXTRA_ARGS" ]; then
echo "Extra args:  $EXTRA_ARGS"
fi
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
    
    # Path structure: outputs/{dataset}/{scene}/{method}/{name}
    # For cat mode, train.py appends _{hybrid_levels}_levels to the name
    if [ "$method" = "cat" ]; then
        local hl_suffix=$(echo "$extra_args" | grep -oP '(?<=--hybrid_levels )\d+')
        OUTPUT_PATH="outputs/${DATASET_PATH}/${scene_name}/${method}/${experiment_name}_${hl_suffix}_levels"
    else
        OUTPUT_PATH="outputs/${DATASET_PATH}/${scene_name}/${method}/${experiment_name}"
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
    
    CMD="python train.py -s $scene_path -m $experiment_name --yaml $YAML_CONFIG --eval --iterations $ITERATIONS $RESOLUTION_ARG --method $method $extra_args $EXTRA_ARGS"
    
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
    for hl in {5..6}; do
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
echo "Dataset:         $DATASET"
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


