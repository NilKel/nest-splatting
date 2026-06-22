#!/bin/bash
# Script to train Nest-Splatting with --method res_3d_paired
#   (staged curriculum: mode 0→2 flip at --res_switch_iter, then 2D/3D split
#    at --res_3d_iter. Joint T cascade via diff_surfel_mixed_3d. Tex carriers
#    contribute ReLU(SV+0.5)+residual, untex EWA carriers contribute
#    ReLU(SV+0.5). See docs/RES_3D_MODES.md.)
#
# Usage: ./train_res_3d_paired.sh <dataset> <base_name> [scene_names] [iterations] [extra_args]
#
# Equivalent to:
#   python train.py -s <scene> -m <expname> --yaml <cfg> --eval --iterations N \
#       <-i imagesX> --method res_3d_paired --kernel beta_scaled \
#       --res_switch_iter 10000 --res_3d_iter 15000 \
#       --hybrid_levels 2 --disable_c2f --aabb snugbox <extra_args>
#
# Examples:
#   ./train_res_3d_paired.sh mip_360 Dec10_15L01S10k_SV_30thr_005w25gLP4lev_FRP5k10_N2F_Jac_5ksp_i2 all 35000 \
#       "--cold --fastgs --fastgs_densify_interval 500 --fastgs_densify_until 20000 \
#        --fastgs_grad_thresh 0.00015 --fastgs_grad_abs_thresh 0.0006 --fastgs_dense 0.01 \
#        --fastgs_importance_thresh 30 --fastgs_loss_thresh 0.1 --grads abs --feature SV \
#        --w_lambda 0.005 --w_lambda_gamma 25 --lowpass \
#        --freeze_hash_iter 5000 --freeze_hash_period 10"
#
# Scenes whose `outputs/<dataset>/<scene>/res_3d_paired/<base_name>/test_metrics.txt`
# already exists are SKIPPED automatically — re-runs are safe.

set -e

DEFAULT_ITERATIONS=35000
BASE_DATA_DIR="/home/nilkel/Projects/data/nest_synthetic"
DTU_DATA_DIR="/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU"
MIP360_DATA_DIR="/home/nilkel/Projects/data/mip_360"

# --- Defaults for --method res_3d_paired. Override any of these via extra_args. ---
# Stage 1 (mode 0→2 + post-blend LeakyReLU) fires at --res_switch_iter.
# Stage 2 (2D-residual / 3D-EWA-SV split) fires at --res_3d_iter.
# --kernel2 gaussian = canonical recipe (textured 2D = beta_scaled,
# untextured EWA = Gaussian ellipsoid). NOTE: prior res_3d*/res_3d_paired/
# res_3d_double runs in this repo did NOT set --kernel2 → untextured fell
# back to --kernel beta_scaled. Override here via extra_args if you want
# to reproduce that earlier behaviour (e.g. extra_args="--kernel2 beta_scaled").
METHOD_DEFAULTS="--method res_3d_paired --kernel beta_scaled --kernel2 gaussian \
                 --res_switch_iter 10000 --res_3d_iter 15000 \
                 --hybrid_levels 2 --disable_c2f --aabb snugbox"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <base_name> [scene_names] [iterations] [extra_args]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Required. nerf_synthetic, DTU, mip_360, tnt, db"
    echo "  base_name     Required. Base name for experiment (output subdir)"
    echo "  scene_names   Optional. Comma-separated or 'all' (default: all)"
    echo "  iterations    Optional. Training iterations (default: $DEFAULT_ITERATIONS)"
    echo "  extra_args    Optional. Extra args to train.py (in quotes)"
    echo ""
    echo "Baked-in defaults: $METHOD_DEFAULTS"
    echo ""
    echo "Output path: outputs/<dataset>/<scene>/res_3d_paired/<base_name>"
    echo "Note: scenes whose test_metrics.txt already exists are SKIPPED."
    exit 1
fi

DATASET=$1
BASE_NAME=$2
SCENE_NAMES=${3:-all}
ITERATIONS=${4:-$DEFAULT_ITERATIONS}
EXTRA_ARGS=${5:-""}

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
        RESOLUTION_ARG="-r 2"
        ;;
    mip_360)
        DATA_DIR="$MIP360_DATA_DIR"
        YAML_CONFIG="PER_SCENE"
        ALL_SCENES="bicycle,bonsai,counter,garden,kitchen,room,stump,flowers,treehill"
        DATASET_PATH="mip_360"
        RESOLUTION_ARG="PER_SCENE"
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
        exit 1
        ;;
esac

if [ "$SCENE_NAMES" = "all" ]; then
    SCENE_NAMES=$ALL_SCENES
fi
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"

if [ "$YAML_CONFIG" != "PER_SCENE" ] && [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"; exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting - res_3d_paired Training"
echo "════════════════════════════════════════════════════════════════════"
echo "Dataset:     $DATASET"
echo "Base name:   $BASE_NAME"
echo "Scenes:      ${SCENES[@]}"
echo "Iterations:  $ITERATIONS"
echo "Data dir:    $DATA_DIR"
if [ "$YAML_CONFIG" = "PER_SCENE" ]; then
echo "Config:      PER_SCENE (indoor: 360_indoor.yaml -i images_2 / outdoor: 360_outdoor.yaml -i images_4)"
else
echo "Config:      $YAML_CONFIG"
[ -n "$RESOLUTION_ARG" ] && echo "Resolution:  ${RESOLUTION_ARG#-r }"
fi
echo "Defaults:    $METHOD_DEFAULTS"
[ -n "$EXTRA_ARGS" ] && echo "Extra args:  $EXTRA_ARGS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR
GLOBAL_LOG_FILE="${LOG_DIR}/train_res_3d_paired_${BASE_NAME}_${TIMESTAMP}.log"
echo "Logging to: $GLOBAL_LOG_FILE"
echo ""

run_training() {
    local scene_name=$1
    local experiment_name=$2
    local extra_args=$3

    CURRENT_RUN=$((CURRENT_RUN + 1))
    local scene_path="${DATA_DIR}/${scene_name}"
    if [ ! -d "$scene_path" ]; then
        echo "WARNING: Scene path does not exist: $scene_path - SKIPPING"
        SKIPPED=$((SKIPPED + 1)); return 0
    fi

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

    OUTPUT_PATH="outputs/${DATASET_PATH}/${scene_name}/res_3d_paired/${experiment_name}"
    TEST_METRICS="${OUTPUT_PATH}/test_metrics.txt"
    if [ -f "$TEST_METRICS" ]; then
        echo "════════════════════════════════════════════════════════════════════"
        echo "  [$CURRENT_RUN/$TOTAL_RUNS] SKIPPING: ${scene_name} - ${experiment_name}"
        echo "════════════════════════════════════════════════════════════════════"
        echo "Already completed: $TEST_METRICS"; echo ""
        SKIPPED=$((SKIPPED + 1)); return 0
    fi

    echo "════════════════════════════════════════════════════════════════════"
    echo "  [$CURRENT_RUN/$TOTAL_RUNS] Training: ${scene_name} - ${experiment_name}"
    echo "════════════════════════════════════════════════════════════════════"

    CMD="python train.py -s $scene_path -m $experiment_name --yaml $scene_yaml --eval --iterations $ITERATIONS $scene_resolution $METHOD_DEFAULTS $extra_args $EXTRA_ARGS"
    echo "Command: $CMD"; echo "Started: $(date)"; echo ""

    $CMD 2>&1 | tee -a $GLOBAL_LOG_FILE
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""; echo "Completed: ${scene_name} - ${experiment_name}"; echo "Finished: $(date)"; echo ""
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""; echo "FAILED: ${scene_name} - ${experiment_name} (exit code: $EXIT_CODE)"; echo "Finished: $(date)"; echo ""
        FAILED=$((FAILED + 1))
    fi
}

NUM_SCENES=${#SCENES[@]}
TOTAL_RUNS=$NUM_SCENES
CURRENT_RUN=0; COMPLETED=0; SKIPPED=0; FAILED=0
echo "Total experiments: $TOTAL_RUNS (${NUM_SCENES} scenes)"; echo ""

for scene in "${SCENES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SCENE: ${scene}"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    run_training "$scene" "${BASE_NAME}" ""
    echo ""; echo "  Completed scene: ${scene}"; echo ""
done

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
    echo "WARNING: $FAILED experiments failed. Check the log file for details."; echo ""
fi
