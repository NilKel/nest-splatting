#!/bin/bash
# Script to train Nest-Splatting on all methods and configurations
# Usage: ./train_all_methods.sh <scene_name> <base_name> [iterations] [data_dir]
#
# Example:
#   ./train_all_methods.sh drums testall 30000

set -e  # Exit on error

# Default values
DEFAULT_ITERATIONS=30000
DEFAULT_DATA_DIR="/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic"
YAML_CONFIG="./configs/nerfsyn.yaml"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <scene_name> <base_name> [iterations] [data_dir]"
    echo ""
    echo "Arguments:"
    echo "  scene_name    Required. Name of the scene (e.g., drums, mic, lego)"
    echo "  base_name     Required. Base name for experiments (e.g., testall, exp1)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  data_dir      Optional. Base data directory (default: /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic)"
    echo ""
    echo "Example:"
    echo "  $0 drums testall"
    echo "  $0 drums testall 30000"
    echo "  $0 mic experiment1 30000 /path/to/data"
    echo ""
    echo "Output structure:"
    echo "  outputs/baseline/nerf_synthetic/<scene>/<base_name>_baseline/"
    echo "  outputs/add/nerf_synthetic/<scene>/<base_name>_add/"
    echo "  outputs/cat/nerf_synthetic/<scene>/<base_name>_cat1/"
    echo "  outputs/cat/nerf_synthetic/<scene>/<base_name>_cat2/"
    echo "  ...etc"
    exit 1
fi

SCENE_NAME=$1
BASE_NAME=$2
ITERATIONS=${3:-$DEFAULT_ITERATIONS}
DATA_DIR=${4:-$DEFAULT_DATA_DIR}

# Construct full scene path
SCENE_PATH="${DATA_DIR}/${SCENE_NAME}"

# Verify scene path exists
if [ ! -d "$SCENE_PATH" ]; then
    echo "ERROR: Scene path does not exist: $SCENE_PATH"
    exit 1
fi

# Verify YAML config exists
if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting - Train All Methods"
echo "════════════════════════════════════════════════════════════════════"
echo "Scene:      $SCENE_NAME"
echo "Base name:  $BASE_NAME"
echo "Path:       $SCENE_PATH"
echo "Iterations: $ITERATIONS"
echo "Config:     $YAML_CONFIG"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Counter for completed runs
TOTAL_RUNS=8
COMPLETED=0

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/train_all_${SCENE_NAME}_${BASE_NAME}_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Function to run training
run_training() {
    local method=$1
    local experiment_name=$2
    local extra_args=$3
    
    COMPLETED=$((COMPLETED + 1))
    
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [$COMPLETED/$TOTAL_RUNS] Training: $method - $experiment_name"
    echo "════════════════════════════════════════════════════════════════════"
    
    # Construct command
    CMD="python train.py -s $SCENE_PATH -m $experiment_name --yaml $YAML_CONFIG --eval --iterations $ITERATIONS --method $method $extra_args"
    
    echo "Command: $CMD"
    echo "Started: $(date)"
    echo ""
    
    # Run training and log output
    $CMD 2>&1 | tee -a $LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Completed successfully: $method - $experiment_name"
        echo "Finished: $(date)"
        echo ""
    else
        echo ""
        echo "✗ FAILED: $method - $experiment_name (exit code: $EXIT_CODE)"
        echo "Finished: $(date)"
        echo ""
        exit $EXIT_CODE
    fi
}

# ============================================================================
# 1. BASELINE MODE
# ============================================================================
run_training "baseline" "${BASE_NAME}_baseline" ""

# ============================================================================
# 2. ADD MODE
# ============================================================================
run_training "add" "${BASE_NAME}_add" ""

# ============================================================================
# 3. CAT MODE - hybrid_levels 1 to 6 (no c2f)
# ============================================================================
for hl in {1..6}; do
    run_training "cat" "${BASE_NAME}_cat${hl}" "--hybrid_levels $hl"
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL TRAINING RUNS COMPLETED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════"
echo "Scene:           $SCENE_NAME"
echo "Base name:       $BASE_NAME"
echo "Total runs:      $TOTAL_RUNS"
echo "Completed:       $COMPLETED"
echo "Log file:        $LOG_FILE"
echo ""
echo "Output directories:"
echo "  outputs/baseline/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_baseline/"
echo "  outputs/add/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_add/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat1/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat2/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat3/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat4/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat5/"
echo "  outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat6/"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Create summary file
SUMMARY_FILE="${LOG_DIR}/summary_${SCENE_NAME}_${BASE_NAME}_${TIMESTAMP}.txt"
echo "Creating summary file: $SUMMARY_FILE"
cat > $SUMMARY_FILE <<EOF
Nest-Splatting Training Summary
════════════════════════════════════════════════════════════════════

Scene:           $SCENE_NAME
Base Name:       $BASE_NAME
Scene Path:      $SCENE_PATH
Iterations:      $ITERATIONS
YAML Config:     $YAML_CONFIG
Timestamp:       $TIMESTAMP
Log File:        $LOG_FILE

Training Runs Completed: $COMPLETED/$TOTAL_RUNS
────────────────────────────────────────────────────────────────────

1. Baseline Mode
   Output: outputs/baseline/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_baseline/

2. Add Mode
   Output: outputs/add/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_add/

3. Cat Mode (no c2f)
   - hybrid_levels=1: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat1/
   - hybrid_levels=2: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat2/
   - hybrid_levels=3: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat3/
   - hybrid_levels=4: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat4/
   - hybrid_levels=5: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat5/
   - hybrid_levels=6: outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat6/

════════════════════════════════════════════════════════════════════

Checkpoint Files (for each output directory):
  - test_metrics.txt
  - train_metrics.txt
  - checkpoint_config.json
  - training_summary.txt
  - ngp_30000.pth
  - point_cloud/iteration_30000/point_cloud.ply
  - point_cloud/iteration_30000/point_cloud_gaussian_features.pth (add/cat only)

════════════════════════════════════════════════════════════════════
EOF

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "To analyze results, check the test_metrics.txt in each output directory:"
echo "  cat outputs/baseline/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_baseline/test_metrics.txt"
echo "  cat outputs/add/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_add/test_metrics.txt"
echo "  cat outputs/cat/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_cat*/test_metrics.txt"
echo ""
echo "Or compare all at once:"
echo "  grep 'Average PSNR' outputs/*/nerf_synthetic/${SCENE_NAME}/${BASE_NAME}_*/test_metrics.txt"
echo ""

