#!/bin/bash
#=============================================================================
# Worker script - runs a single training experiment
#=============================================================================

# Arguments passed via environment:
# JOB_CONFIG_FILE - path to job config file
# SLURM_ARRAY_TASK_ID - which line to read from config
# DATA_DIR, YAML_CONFIG, DATASET_PATH, RESOLUTION_ARG, ITERATIONS, EXTRA_ARGS

set -e

# mip_360 scene classification
MIP360_OUTDOOR_SCENES="bicycle flowers garden stump treehill"
MIP360_INDOOR_SCENES="room counter kitchen bonsai"

# Read the job specification from the config file
JOB_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_CONFIG_FILE")
if [ -z "$JOB_LINE" ]; then
    echo "ERROR: No job found at line $SLURM_ARRAY_TASK_ID in $JOB_CONFIG_FILE"
    exit 1
fi

# Parse job line: scene method experiment_name [extra_method_args]
read -r SCENE METHOD EXPERIMENT_NAME METHOD_ARGS <<< "$JOB_LINE"

# Handle per-scene config for mip_360
SCENE_YAML_CONFIG="$YAML_CONFIG"
SCENE_RESOLUTION_ARG="$RESOLUTION_ARG"

if [ "$DATASET_PATH" = "mip_360" ]; then
    # Check if scene is indoor or outdoor
    if echo "$MIP360_INDOOR_SCENES" | grep -qw "$SCENE"; then
        SCENE_YAML_CONFIG="./configs/360_indoor.yaml"
        SCENE_RESOLUTION_ARG="-i images_2"
    else
        # Default to outdoor for any scene in outdoor list or unknown
        SCENE_YAML_CONFIG="./configs/360_outdoor.yaml"
        SCENE_RESOLUTION_ARG="-i images_4"
    fi
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  SLURM Job: Task $SLURM_ARRAY_TASK_ID"
echo "════════════════════════════════════════════════════════════════════"
echo "Scene:       $SCENE"
echo "Method:      $METHOD"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Method args: $METHOD_ARGS"
echo "YAML config: $SCENE_YAML_CONFIG"
echo "Resolution:  $SCENE_RESOLUTION_ARG"
echo "Host:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Started:     $(date)"
echo "════════════════════════════════════════════════════════════════════"

SCENE_PATH="${DATA_DIR}/${SCENE}"

# Check if scene exists
if [ ! -d "$SCENE_PATH" ]; then
    echo "ERROR: Scene path does not exist: $SCENE_PATH"
    exit 1
fi

# Determine output path and check for completion
if [ "$METHOD" = "cat" ]; then
    HL_SUFFIX=$(echo "$METHOD_ARGS" | grep -oP '(?<=--hybrid_levels )\d+')
    OUTPUT_PATH="outputs/${DATASET_PATH}/${SCENE}/${METHOD}/${EXPERIMENT_NAME}_${HL_SUFFIX}_levels"
else
    OUTPUT_PATH="outputs/${DATASET_PATH}/${SCENE}/${METHOD}/${EXPERIMENT_NAME}"
fi
TEST_METRICS="${OUTPUT_PATH}/test_metrics.txt"

# Skip if already completed
if [ -f "$TEST_METRICS" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SKIPPING - Already completed"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Found: $TEST_METRICS"
    echo "Finished: $(date)"
    exit 0
fi

# Assign unique port per task to avoid conflicts when multiple jobs run on same node
PORT=$((6009 + SLURM_ARRAY_TASK_ID))

# Run training
CMD="python train.py -s $SCENE_PATH -m $EXPERIMENT_NAME --yaml $SCENE_YAML_CONFIG --eval --iterations $ITERATIONS $SCENE_RESOLUTION_ARG --method $METHOD --port $PORT $METHOD_ARGS $EXTRA_ARGS"

echo ""
echo "Command: $CMD"
echo ""

$CMD

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ SUCCESS"
else
    echo "  ✗ FAILED (exit code: $EXIT_CODE)"
fi
echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"

exit $EXIT_CODE
