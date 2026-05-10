#!/bin/bash
# Train Hybrid-Latents (Gaussian) — `--method cat` — across datasets.
# Mirrors the JSON config used for the bicycle "cat5" runs:
#   --method cat --hybrid_levels 5 --disable_c2f --kernel gaussian --aabb adr
#   --mcmc --cap_max 100000 --opacity_reg 0.001 --noise_lr 1e3
#   --bce --bce_iter 5000 --bce_lambda 0.01 --bce_threshold 0.5
#   --lambda_adaptive 0.001 --scout_lambda 0.01
#   --lambda_adaptive_cat 0.01 --adaptive_cat_anneal_start 15000 --adaptive_cat_threshold 0.9
#   --lambda_sparsity 0.005 --force_ratio 0.2 --gate_init 2.0
#   --relocation clone --temp_anneal_start 3000 --temp_anneal_end 25000
#
# Usage: ./train_cat_hybrid.sh <dataset> <base_name> [scenes] [iterations] [extra_args]
#
# Examples:
#   ./train_cat_hybrid.sh tnt     cat5_hl all 35000
#   ./train_cat_hybrid.sh db      cat5_hl all 35000
#   ./train_cat_hybrid.sh mip_360 cat5_hl all 35000
#   ./train_cat_hybrid.sh tnt     cat5_hl truck 35000 "--cap_max 200000"

set -e

# Defaults
DEFAULT_ITERATIONS=35000
NESTSYN_DIR="/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic"
DTU_DIR="/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU"
MIP360_DIR="/home/nilkel/Projects/data/mip_360"
TNT_DIR="/home/nilkel/Projects/data/tnt"
DB_DIR="/home/nilkel/Projects/data/db"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <base_name> [scenes] [iterations] [extra_args]"
    echo ""
    echo "Datasets:"
    echo "  nerf_synthetic: chair, drums, ficus, hotdog, lego, materials, mic, ship    (configs/nerfsyn.yaml)"
    echo "  DTU:            scan24..scan122                                            (configs/dtu.yaml, -r 2)"
    echo "  mip_360:        bicycle,bonsai,counter,garden,kitchen,room,stump,flowers,treehill"
    echo "                  (per-scene: 360_outdoor.yaml -i images_4 OR 360_indoor.yaml -i images_2)"
    echo "  tnt:            train, truck                  (configs/tandt.yaml, full res)"
    echo "  db:             drjohnson, playroom           (configs/db.yaml,    full res)"
    exit 1
fi

DATASET=$1
BASE_NAME=$2
SCENE_NAMES=${3:-all}
ITERATIONS=${4:-$DEFAULT_ITERATIONS}
EXTRA_ARGS=${5:-""}

case "$DATASET" in
    nerf_synthetic)
        DATA_DIR="$NESTSYN_DIR"; YAML_CONFIG="./configs/nerfsyn.yaml"
        ALL_SCENES="chair,drums,ficus,hotdog,lego,materials,mic,ship"
        DATASET_PATH="nerf_synthetic"; RESOLUTION_ARG=""
        ;;
    DTU)
        DATA_DIR="$DTU_DIR"; YAML_CONFIG="./configs/dtu.yaml"
        ALL_SCENES="scan24,scan37,scan40,scan55,scan63,scan65,scan69,scan83,scan97,scan105,scan106,scan110,scan114,scan118,scan122"
        DATASET_PATH="DTU"; RESOLUTION_ARG="-r 2"
        ;;
    mip_360)
        DATA_DIR="$MIP360_DIR"; YAML_CONFIG="PER_SCENE"
        ALL_SCENES="bicycle,bonsai,counter,garden,kitchen,room,stump,flowers,treehill"
        DATASET_PATH="mip_360"; RESOLUTION_ARG="PER_SCENE"
        MIP360_OUTDOOR_SCENES="bicycle flowers garden stump treehill"
        MIP360_INDOOR_SCENES="room counter kitchen bonsai"
        ;;
    tnt)
        DATA_DIR="$TNT_DIR"; YAML_CONFIG="./configs/tandt.yaml"
        ALL_SCENES="train,truck"
        DATASET_PATH="tnt"; RESOLUTION_ARG=""
        ;;
    db)
        DATA_DIR="$DB_DIR"; YAML_CONFIG="./configs/db.yaml"
        ALL_SCENES="drjohnson,playroom"
        DATASET_PATH="db"; RESOLUTION_ARG=""
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Available: nerf_synthetic, DTU, mip_360, tnt, db"
        exit 1
        ;;
esac

[ "$SCENE_NAMES" = "all" ] && SCENE_NAMES=$ALL_SCENES
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"

[ "$YAML_CONFIG" != "PER_SCENE" ] && [ ! -f "$YAML_CONFIG" ] && { echo "ERROR: missing $YAML_CONFIG"; exit 1; }
[ ! -d "$DATA_DIR" ] && { echo "ERROR: missing data dir $DATA_DIR"; exit 1; }

echo "════════════════════════════════════════════════════════════════════"
echo "  Hybrid-Latents (Gaussian) — --method cat"
echo "════════════════════════════════════════════════════════════════════"
echo "Dataset:    $DATASET   Base name: $BASE_NAME"
echo "Scenes:     ${SCENES[@]}"
echo "Iters:      $ITERATIONS"
echo "Extra:      $EXTRA_ARGS"
echo "════════════════════════════════════════════════════════════════════"
echo

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
GLOBAL_LOG="logs/train_cat_hybrid_${BASE_NAME}_${TIMESTAMP}.log"
echo "Log: $GLOBAL_LOG"; echo

NUM=${#SCENES[@]}; CUR=0; OK=0; SKIP=0; FAIL=0

run_training() {
    local scene=$1
    CUR=$((CUR + 1))
    local scene_path="${DATA_DIR}/${scene}"
    [ ! -d "$scene_path" ] && { echo "WARN: $scene_path missing — SKIP"; SKIP=$((SKIP + 1)); return 0; }

    local yaml="$YAML_CONFIG"; local resarg="$RESOLUTION_ARG"
    if [ "$YAML_CONFIG" = "PER_SCENE" ]; then
        if echo "$MIP360_INDOOR_SCENES" | grep -qw "$scene"; then
            yaml="./configs/360_indoor.yaml"; resarg="-i images_2"
        else
            yaml="./configs/360_outdoor.yaml"; resarg="-i images_4"
        fi
    fi

    local OUTPATH="outputs/${DATASET_PATH}/${scene}/cat/${BASE_NAME}"
    local METRICS="${OUTPATH}/test_metrics.txt"
    if [ -f "$METRICS" ]; then
        echo "[$CUR/$NUM] SKIP $scene — already done ($METRICS)"
        SKIP=$((SKIP + 1)); return 0
    fi

    echo "════════════════════════════════════════════════════════════════════"
    echo "  [$CUR/$NUM] $scene"
    echo "════════════════════════════════════════════════════════════════════"

    CMD="python train.py -s $scene_path -m $BASE_NAME --yaml $yaml --eval --iterations $ITERATIONS $resarg \
--method cat --hybrid_levels 2 --disable_c2f --kernel gaussian --aabb adr \
--mcmc_fps --cap_max 150000 --opacity_reg 0.01 --scale_reg 0.0 --noise_lr 1e3 \
--grads abs --lowpass\

$EXTRA_ARGS"

    echo "CMD: $CMD"; echo "Started: $(date)"; echo
    $CMD 2>&1 | tee -a "$GLOBAL_LOG"
    EC=${PIPESTATUS[0]}
    if [ $EC -eq 0 ]; then
        echo "OK   $scene  (finished $(date))"; OK=$((OK + 1))
    else
        echo "FAIL $scene  (exit $EC)"; FAIL=$((FAIL + 1))
    fi
}

for scene in "${SCENES[@]}"; do run_training "$scene"; echo; done

echo "════════════════════════════════════════════════════════════════════"
echo "  COMPLETE  ok=$OK  skip=$SKIP  fail=$FAIL"
echo "  Log: $GLOBAL_LOG"
echo "════════════════════════════════════════════════════════════════════"
[ $FAIL -gt 0 ] && exit 1
exit 0
