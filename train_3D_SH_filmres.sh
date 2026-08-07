#!/bin/bash
# Script to train Nest-Splatting with 3D_SH_filmres mode (FiLM on the residual MLP).
# Clone of train_3D_SH_res.sh with: --method 3D_SH_filmres, outputs under
# 3D_SH_filmres/, and filmres-typical defaults (--hybrid_levels 2 --aabb accutile).
# mip_360 per-scene configs: 360_indoor.yaml / 360_outdoor.yaml.
#
# Usage: ./train_3D_SH_filmres.sh <dataset> <base_name> [scene_names] [iterations] [extra_args]
#
# Example:
#   ./train_3D_SH_filmres.sh mip_360 exp1 all 35000 "--hybrid_levels 2 --film_gamma_init 1.0"
#   ./train_3D_SH_filmres.sh mip_360 exp1 garden,room 35000 "--film_act gamma_sigm_split"

set -e  # Exit on error

# Default values
DEFAULT_ITERATIONS=30000
BASE_DATA_DIR="/home/nilkel/Projects/data/nest_synthetic"
DTU_DATA_DIR="/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU"
MIP360_DATA_DIR="/home/nilkel/Projects/data/mip_360"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <base_name> [scene_names] [iterations] [extra_args]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Required. Dataset name: nerf_synthetic, DTU, mip_360, tnt, db"
    echo "  base_name     Required. Base name for experiments (e.g., exp1, test)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  extra_args    Optional. Extra arguments to pass to train.py"
    echo ""
    echo "mip_360 per-scene configs: indoor -> 360_indoor.yaml (images_2),"
    echo "                           outdoor -> 360_outdoor.yaml (images_4)"
    echo ""
    echo "Datasets:"
    echo "  nerf_synthetic: chair, drums, ficus, hotdog, lego, materials, mic, ship"
    echo "  DTU: scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122"
    echo "  mip_360: bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill"
    echo "  tnt:     train, truck                    (Tanks & Temples — full res, configs/tandt.yaml)"
    echo "  db:      drjohnson, playroom             (Deep Blending  — full res, configs/db.yaml)"
    exit 1
fi

DATASET=$1
BASE_NAME=$2
SCENE_NAMES=${3:-all}
ITERATIONS=${4:-$DEFAULT_ITERATIONS}
EXTRA_ARGS=${5:-""}

METHOD="3D_SH_filmres"

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
        # YAML_CONFIG and RESOLUTION_ARG set per-scene (indoor vs outdoor) — 2D configs
        YAML_CONFIG="PER_SCENE"
        ALL_SCENES="bicycle,bonsai,counter,garden,kitchen,room,stump,flowers,treehill"
        DATASET_PATH="mip_360"
        RESOLUTION_ARG="PER_SCENE"
        # Scene classification
        MIP360_OUTDOOR_SCENES="bicycle flowers garden stump treehill"
        MIP360_INDOOR_SCENES="room counter kitchen bonsai"
        ;;
    tnt)
        DATA_DIR="/home/nilkel/Projects/data/tnt"
        YAML_CONFIG="./configs/tandt.yaml"
        ALL_SCENES="train,truck"
        DATASET_PATH="tnt"
        RESOLUTION_ARG=""
        ;;
    db)
        DATA_DIR="/home/nilkel/Projects/data/db"
        YAML_CONFIG="./configs/db.yaml"
        ALL_SCENES="drjohnson,playroom"
        DATASET_PATH="db"
        RESOLUTION_ARG=""
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "Available datasets: nerf_synthetic, DTU, mip_360, tnt, db"
        exit 1
        ;;
esac

# Handle "all" keyword
if [ "$SCENE_NAMES" = "all" ]; then
    SCENE_NAMES=$ALL_SCENES
fi

# Convert comma-separated list to array
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"

# Verify YAML config exists (skip for PER_SCENE which is resolved at runtime)
if [ "$YAML_CONFIG" != "PER_SCENE" ] && [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting - ${METHOD} Training"
echo "════════════════════════════════════════════════════════════════════"
echo "Dataset:     $DATASET"
echo "Base name:   $BASE_NAME"
echo "Scenes:      ${SCENES[@]}"
echo "Iterations:  $ITERATIONS"
echo "Data dir:    $DATA_DIR"
if [ "$YAML_CONFIG" = "PER_SCENE" ]; then
echo "Config:      PER_SCENE (indoor: 360_indoor.yaml, outdoor: 360_outdoor.yaml)"
echo "Resolution:  PER_SCENE (indoor: images_2, outdoor: images_4)"
else
echo "Config:      $YAML_CONFIG"
if [ -n "$RESOLUTION_ARG" ]; then
echo "Resolution:  ${RESOLUTION_ARG#-r }"
fi
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
GLOBAL_LOG_FILE="${LOG_DIR}/train_${METHOD}_${BASE_NAME}_${TIMESTAMP}.log"

echo "Logging to: $GLOBAL_LOG_FILE"
echo ""

# Function to run training
run_training() {
    local scene_name=$1
    local experiment_name=$2
    local extra_args=$3

    CURRENT_RUN=$((CURRENT_RUN + 1))

    local scene_path="${DATA_DIR}/${scene_name}"

    if [ ! -d "$scene_path" ]; then
        echo "WARNING: Scene path does not exist: $scene_path - SKIPPING"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Handle per-scene config for mip_360 (indoor vs outdoor)
    local scene_yaml="$YAML_CONFIG"
    local scene_resolution="$RESOLUTION_ARG"
    if [ "$YAML_CONFIG" = "PER_SCENE" ]; then
        if echo "$MIP360_INDOOR_SCENES" | grep -qw "$scene_name"; then
            scene_yaml="./configs/360_indoor.yaml"
            scene_resolution="-i images_2"
        else
            scene_yaml="./configs/360_outdoor.yaml"
            scene_resolution="-i images_4"
        fi
    fi

    OUTPUT_PATH="outputs/${DATASET_PATH}/${scene_name}/${METHOD}/${experiment_name}"
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

    CMD="python train.py -s $scene_path -m $experiment_name --yaml $scene_yaml --eval --iterations $ITERATIONS $scene_resolution --method ${METHOD} --hybrid_levels 2 --disable_c2f --aabb accutile --warmup gaussian --kernel gaussian $extra_args $EXTRA_ARGS"

    echo "Command: $CMD"
    echo "Started: $(date)"
    echo ""

    $CMD 2>&1 | tee -a $GLOBAL_LOG_FILE

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "Completed: ${scene_name} - ${experiment_name}"
        echo "Finished: $(date)"
        echo ""
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo "FAILED: ${scene_name} - ${experiment_name} (exit code: $EXIT_CODE)"
        echo "Finished: $(date)"
        echo ""
        FAILED=$((FAILED + 1))
    fi
}

# Calculate total runs
NUM_SCENES=${#SCENES[@]}
TOTAL_RUNS=$NUM_SCENES
CURRENT_RUN=0
COMPLETED=0
SKIPPED=0
FAILED=0

echo "Total experiments: $TOTAL_RUNS (${NUM_SCENES} scenes)"
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

    run_training "$scene" "${BASE_NAME}" ""

    echo ""
    echo "  Completed scene: ${scene}"
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
    echo "WARNING: $FAILED experiments failed. Check the log file for details."
    echo ""
fi

# Generate results table
echo "Generating results tables..."
python create_results_table.py --base_name "$BASE_NAME" --scenes "${SCENES[*]// /,}"

echo ""
echo "Done! Check metrics_tables/ for results."
echo ""
