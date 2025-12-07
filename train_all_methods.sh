#!/bin/bash
# Script to train Nest-Splatting on all methods and configurations
# Usage: ./train_all_methods.sh <base_name> [scene_names] [iterations] [data_dir]
#
# Example:
#   ./train_all_methods.sh testall              # Run all scenes
#   ./train_all_methods.sh testall drums,mic    # Run specific scenes
#   ./train_all_methods.sh testall all 30000    # Run all scenes with 30k iters

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
    echo "  base_name     Required. Base name for experiments (e.g., testall, exp1)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "                Examples: drums,mic,lego OR all"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  data_dir      Optional. Base data directory (default: /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic)"
    echo ""
    echo "Examples:"
    echo "  $0 testall                      # Run all 8 scenes"
    echo "  $0 testall drums,mic            # Run only drums and mic"
    echo "  $0 testall all 30000            # Run all scenes with 30k iterations"
    echo "  $0 exp1 lego 30000 /path/to/data"
    echo ""
    echo "Output structure:"
    echo "  outputs/baseline/nerf_synthetic/<scene>/<base_name>_baseline/"
    echo "  outputs/add/nerf_synthetic/<scene>/<base_name>_add/"
    echo "  outputs/cat/nerf_synthetic/<scene>/<base_name>_cat0/"
    echo "  outputs/cat/nerf_synthetic/<scene>/<base_name>_cat1/"
    echo "  ...etc (cat0 through cat6)"
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
echo "  Nest-Splatting - Train All Methods & Scenes"
echo "════════════════════════════════════════════════════════════════════"
echo "Base name:  $BASE_NAME"
echo "Scenes:     ${SCENES[@]}"
echo "Iterations: $ITERATIONS"
echo "Data dir:   $DATA_DIR"
echo "Config:     $YAML_CONFIG"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Global log file for all scenes
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR
GLOBAL_LOG_FILE="${LOG_DIR}/train_all_scenes_${BASE_NAME}_${TIMESTAMP}.log"

echo "Logging to: $GLOBAL_LOG_FILE"
echo ""

# Function to run training for a single scene
run_training() {
    local scene_name=$1
    local method=$2
    local experiment_name=$3
    local extra_args=$4
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Construct scene path
    local scene_path="${DATA_DIR}/${scene_name}"
    
    # Verify scene path exists
    if [ ! -d "$scene_path" ]; then
        echo "WARNING: Scene path does not exist: $scene_path - SKIPPING"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    
    # Determine output path based on method and experiment name
    OUTPUT_PATH="outputs/${method}/nerf_synthetic/${scene_name}/${experiment_name}"
    TEST_METRICS="${OUTPUT_PATH}/test_metrics.txt"
    
    # Check if experiment already completed
    if [ -f "$TEST_METRICS" ]; then
        echo "════════════════════════════════════════════════════════════════════"
        echo "  [$CURRENT_RUN/$TOTAL_RUNS] SKIPPING: ${scene_name} - $method - $experiment_name"
        echo "════════════════════════════════════════════════════════════════════"
        echo "Output already exists with test_metrics.txt:"
        echo "  $TEST_METRICS"
        echo ""
        echo "⊘ Skipped (already completed)"
        echo "Skipped at: $(date)"
        echo ""
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [$CURRENT_RUN/$TOTAL_RUNS] Training: ${scene_name} - $method - $experiment_name"
    echo "════════════════════════════════════════════════════════════════════"
    
    # Construct command
    CMD="python train.py -s $scene_path -m $experiment_name --yaml $YAML_CONFIG --eval --iterations $ITERATIONS --method $method $extra_args"
    
    echo "Command: $CMD"
    echo "Started: $(date)"
    echo ""
    
    # Run training and log output
    $CMD 2>&1 | tee -a $GLOBAL_LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Completed successfully: ${scene_name} - $method - $experiment_name"
        echo "Finished: $(date)"
        echo ""
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo "✗ FAILED: ${scene_name} - $method - $experiment_name (exit code: $EXIT_CODE)"
        echo "Finished: $(date)"
        echo ""
        FAILED=$((FAILED + 1))
        # Don't exit - continue with other experiments
    fi
}

# Calculate total runs
NUM_SCENES=${#SCENES[@]}
METHODS_PER_SCENE=9  # baseline, add, cat0-6
TOTAL_RUNS=$((NUM_SCENES * METHODS_PER_SCENE))
CURRENT_RUN=0
COMPLETED=0
SKIPPED=0
FAILED=0

echo "Total experiments to check: $TOTAL_RUNS (${NUM_SCENES} scenes × ${METHODS_PER_SCENE} methods)"
echo ""

# ============================================================================
# LOOP THROUGH ALL SCENES AND METHODS
# ============================================================================
for scene in "${SCENES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SCENE: ${scene}"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    # 1. BASELINE MODE
    run_training "$scene" "baseline" "${BASE_NAME}_baseline" ""
    
    # 2. ADD MODE
    run_training "$scene" "add" "${BASE_NAME}_add" ""
    
    # 3. CAT MODE - hybrid_levels 0 to 6
    # Note: cat0 should be identical to baseline (0 Gaussian features, all hashgrid)
    for hl in {0..6}; do
        run_training "$scene" "cat" "${BASE_NAME}_cat${hl}" "--hybrid_levels $hl --cat_coarse2fine"
    done
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Completed all methods for scene: ${scene}"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL TRAINING RUNS COMPLETED!"
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

echo "Output directories structure:"
echo "  outputs/{method}/nerf_synthetic/{scene}/${BASE_NAME}_{method}/"
echo ""
echo "Methods per scene: baseline, add, cat0, cat1, cat2, cat3, cat4, cat5, cat6"
echo "  (cat0 = all hashgrid, should match baseline performance)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Create summary file
SUMMARY_FILE="${LOG_DIR}/summary_${BASE_NAME}_${TIMESTAMP}.txt"
echo "Creating summary file: $SUMMARY_FILE"
cat > $SUMMARY_FILE <<EOF
Nest-Splatting Multi-Scene Training Summary
════════════════════════════════════════════════════════════════════

Base Name:       $BASE_NAME
Scenes:          ${SCENES[@]}
Data Directory:  $DATA_DIR
Iterations:      $ITERATIONS
YAML Config:     $YAML_CONFIG
Timestamp:       $TIMESTAMP
Log File:        $GLOBAL_LOG_FILE

Training Results:
  Total runs:    $TOTAL_RUNS
  Completed:     $COMPLETED
  Skipped:       $SKIPPED
  Failed:        $FAILED
────────────────────────────────────────────────────────────────────

Methods per scene:
  1. Baseline Mode:  ${BASE_NAME}_baseline
  2. Add Mode:       ${BASE_NAME}_add
  3. Cat Mode:       ${BASE_NAME}_cat0 through ${BASE_NAME}_cat6 (hybrid_levels 0-6)
     - cat0: 0 Gaussian + 6 Hash (should match baseline)
     - cat1-6: N Gaussian + (6-N) Hash levels

Output Structure:
  outputs/{method}/nerf_synthetic/{scene}/${BASE_NAME}_{method}/

Scenes Processed:
EOF

# Add each scene to summary
for scene in "${SCENES[@]}"; do
    echo "  - $scene" >> $SUMMARY_FILE
done

cat >> $SUMMARY_FILE <<EOF

════════════════════════════════════════════════════════════════════

Checkpoint Files (for each output directory):
  - test_metrics.txt        (test set evaluation metrics)
  - train_metrics.txt       (training set evaluation metrics)
  - checkpoint_config.json  (configuration used)
  - ngp_30000.pth           (trained model)
  - point_cloud/iteration_30000/point_cloud.ply
  - point_cloud/iteration_30000/point_cloud_gaussian_features.pth (add/cat1-6 only)

════════════════════════════════════════════════════════════════════
EOF

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "To analyze results across all scenes:"
echo "  grep 'Average PSNR' outputs/*/nerf_synthetic/*/${BASE_NAME}_*/test_metrics.txt"
echo ""
echo "To see results for a specific scene (e.g., drums):"
echo "  grep 'Average PSNR' outputs/*/nerf_synthetic/drums/${BASE_NAME}_*/test_metrics.txt"
echo ""
echo "To compare methods across all scenes:"
echo "  for method in baseline add cat{0..6}; do echo \"=== \$method ===\"; grep 'Average PSNR' outputs/*/nerf_synthetic/*/${BASE_NAME}_\${method}/test_metrics.txt; done"
echo ""

