#!/bin/bash
#=============================================================================
# SLURM Parallel Training Script for Nest-Splatting
#=============================================================================
# This script runs baseline and cat experiments in parallel using SLURM job arrays.
#
# PHASE 1: All baseline experiments run first (creates warmup checkpoints)
# PHASE 2: All cat experiments run after baseline completes (uses warmup checkpoints)
#
# Usage:
#   ./slurm_train_parallel.sh <dataset> <base_name> [scene_names] [iterations] [extra_args]
#
# Examples:
#   ./slurm_train_parallel.sh nerf_synthetic exp1              # All nerf_synthetic scenes
#   ./slurm_train_parallel.sh DTU exp1 scan24,scan37           # Specific DTU scenes
#   ./slurm_train_parallel.sh mip_360 exp1 all 30000           # All mip_360 with 30k iters
#
# The script will:
#   1. Generate job configuration files
#   2. Submit baseline jobs as a job array (Phase 1)
#   3. Submit cat jobs as a job array with dependency on Phase 1 (Phase 2)
#=============================================================================

set -e

# Default values
DEFAULT_ITERATIONS=30000
BASE_DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/nest_synthetic"
DTU_DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/dtu/2DGS_data/DTU"
MIP360_DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/mip_360"

# mip_360 scene classification (for correct resolution/config)
MIP360_OUTDOOR_SCENES="bicycle,flowers,garden,stump,treehill"
MIP360_INDOOR_SCENES="room,counter,kitchen,bonsai"

# SLURM settings - adjust these as needed
PARTITION="a100-4gpu-40gb"
ACCOUNT="rctcd82061"
TIME_LIMIT="24:00:00"  # 24 hours per job

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <base_name> [scene_names] [iterations] [extra_args]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Required. Dataset name: nerf_synthetic, DTU, or mip_360"
    echo "  base_name     Required. Base name for experiments (e.g., exp1, test)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  extra_args    Optional. Extra arguments to pass to train.py"
    echo ""
    echo "Examples:"
    echo "  $0 nerf_synthetic exp1"
    echo "  $0 DTU exp1 scan24,scan37"
    echo "  $0 mip_360 exp1 all 30000"
    echo ""
    echo "SLURM settings (edit script to change):"
    echo "  Partition: $PARTITION"
    echo "  Account:   $ACCOUNT"
    echo "  Time:      $TIME_LIMIT"
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
        RESOLUTION_ARG="-r 2"
        ;;
    mip_360)
        DATA_DIR="$MIP360_DATA_DIR"
        # YAML_CONFIG is set per-scene (indoor vs outdoor) in the worker
        YAML_CONFIG="PER_SCENE"
        ALL_SCENES="bicycle,bonsai,counter,garden,kitchen,room,stump"
        DATASET_PATH="mip_360"
        # RESOLUTION_ARG is set per-scene (-i images_4 outdoor, -i images_2 indoor) in the worker
        RESOLUTION_ARG="PER_SCENE"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        exit 1
        ;;
esac

# Handle "all" keyword
if [ "$SCENE_NAMES" = "all" ]; then
    SCENE_NAMES=$ALL_SCENES
fi

# Convert comma-separated list to array
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"
NUM_SCENES=${#SCENES[@]}

# Create job config directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="slurm_jobs/${BASE_NAME}_${TIMESTAMP}"
mkdir -p "$JOB_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting SLURM Parallel Training"
echo "════════════════════════════════════════════════════════════════════"
echo "Dataset:     $DATASET"
echo "Base name:   $BASE_NAME"
echo "Scenes:      ${SCENES[@]}"
echo "Iterations:  $ITERATIONS"
echo "Data dir:    $DATA_DIR"
echo "Config:      $YAML_CONFIG"
echo "Job dir:     $JOB_DIR"
echo "════════════════════════════════════════════════════════════════════"
echo ""

#=============================================================================
# Generate job configuration files
#=============================================================================

# Phase 1: Baseline jobs (one per scene)
echo "Generating Phase 1 (baseline) job configs..."
BASELINE_CONFIG="$JOB_DIR/baseline_jobs.txt"
> "$BASELINE_CONFIG"

for scene in "${SCENES[@]}"; do
    echo "$scene baseline ${BASE_NAME}_baseline" >> "$BASELINE_CONFIG"
done

# Phase 2: Cat jobs (7 per scene: cat0-cat6)
echo "Generating Phase 2 (cat) job configs..."
CAT_CONFIG="$JOB_DIR/cat_jobs.txt"
> "$CAT_CONFIG"

for scene in "${SCENES[@]}"; do
    for hl in {0..6}; do
        echo "$scene cat ${BASE_NAME}_cat${hl} --hybrid_levels $hl" >> "$CAT_CONFIG"
    done
done

NUM_BASELINE_JOBS=$(wc -l < "$BASELINE_CONFIG")
NUM_CAT_JOBS=$(wc -l < "$CAT_CONFIG")

echo "Phase 1 jobs: $NUM_BASELINE_JOBS (baseline experiments)"
echo "Phase 2 jobs: $NUM_CAT_JOBS (cat experiments)"
echo ""

#=============================================================================
# Create the worker script that each SLURM task will run
#=============================================================================

WORKER_SCRIPT="$JOB_DIR/worker.sh"
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
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

#=============================================================================
# GPU Validation Function
#=============================================================================
validate_gpu() {
    echo "[GPU Check] Validating CUDA availability..."
    python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('[GPU Check] FATAL: No CUDA device available')
    sys.exit(1)

try:
    # Test basic CUDA operations
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    torch.cuda.synchronize()

    # Try importing tinycudann (common failure point)
    import tinycudann as tcnn

    print(f'[GPU Check] OK - {torch.cuda.get_device_name(0)}')
    print(f'[GPU Check] CUDA {torch.version.cuda}, Compute Capability {torch.cuda.get_device_capability(0)}')
except Exception as e:
    print(f'[GPU Check] FATAL: GPU test failed - {e}')
    sys.exit(1)
"
    return $?
}

#=============================================================================
# Request requeue on bad GPU - exits with special code to trigger SLURM requeue
#=============================================================================
MAX_REQUEUES=5  # Maximum times a job can be requeued

request_requeue() {
    local reason=$1

    # Track requeue count using SLURM's restart count
    REQUEUE_COUNT=${SLURM_RESTART_COUNT:-0}

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  REQUESTING REQUEUE - Bad GPU/Node detected"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Reason: $reason"
    echo "Host: $(hostname)"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "Requeue attempt: $((REQUEUE_COUNT + 1))/$MAX_REQUEUES"
    echo ""

    if [ $REQUEUE_COUNT -ge $MAX_REQUEUES ]; then
        echo "ERROR: Max requeue attempts ($MAX_REQUEUES) reached!"
        echo "This job has failed on $MAX_REQUEUES different GPUs."
        echo "Manual intervention required."
        echo "════════════════════════════════════════════════════════════════════"
        exit 1
    fi

    echo "Exiting with code 99 to trigger SLURM requeue..."
    echo "Job will be rescheduled on a different node."
    echo "════════════════════════════════════════════════════════════════════"
    exit 99
}

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
        SCENE_RESOLUTION_ARG="-r 2"
    else
        # Default to outdoor for any scene in outdoor list or unknown
        SCENE_YAML_CONFIG="./configs/360_outdoor.yaml"
        SCENE_RESOLUTION_ARG="-r 4"
    fi
fi

REQUEUE_COUNT=${SLURM_RESTART_COUNT:-0}
echo "════════════════════════════════════════════════════════════════════"
echo "  SLURM Job: Task $SLURM_ARRAY_TASK_ID"
if [ $REQUEUE_COUNT -gt 0 ]; then
    echo "  (Requeue attempt $REQUEUE_COUNT/$MAX_REQUEUES)"
fi
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

# Validate GPU BEFORE running - request requeue if bad
if ! validate_gpu; then
    request_requeue "GPU validation failed (tinycudann import or CUDA test)"
fi

# Run training
CMD="python train.py -s $SCENE_PATH -m $EXPERIMENT_NAME --yaml $SCENE_YAML_CONFIG --eval --iterations $ITERATIONS $SCENE_RESOLUTION_ARG --method $METHOD --port $PORT $METHOD_ARGS $EXTRA_ARGS"

echo ""
echo "Command: $CMD"
echo ""

set +e  # Don't exit on error
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ SUCCESS"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Finished: $(date)"
    exit 0
fi

# Check if this looks like a GPU error (common CUDA errors)
# If so, request requeue to try on a different GPU
if [ $EXIT_CODE -ne 0 ]; then
    echo "  ✗ FAILED (exit code: $EXIT_CODE)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Checking if this is a GPU-related failure..."

    # Request requeue for GPU-related failures
    # The job will be rescheduled on a different node
    request_requeue "Training failed with exit code $EXIT_CODE (likely GPU issue)"
fi
WORKER_EOF

chmod +x "$WORKER_SCRIPT"

#=============================================================================
# Create SLURM submission scripts
#=============================================================================

# Phase 1: Baseline jobs
BASELINE_SBATCH="$JOB_DIR/submit_baseline.sbatch"
cat > "$BASELINE_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_baseline_${BASE_NAME}
#SBATCH --output=$JOB_DIR/logs/baseline_%a.out
#SBATCH --error=$JOB_DIR/logs/baseline_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=1-${NUM_BASELINE_JOBS}%${NUM_BASELINE_JOBS}
#SBATCH --requeue
#SBATCH --open-mode=append

# Handle requeue signal - SLURM sends SIGUSR1 before requeueing
trap 'echo "Received requeue signal, job will be rescheduled..."' SIGUSR1

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables for worker
export JOB_CONFIG_FILE="$BASELINE_CONFIG"
export DATA_DIR="$DATA_DIR"
export YAML_CONFIG="$YAML_CONFIG"
export DATASET_PATH="$DATASET_PATH"
export RESOLUTION_ARG="$RESOLUTION_ARG"
export ITERATIONS="$ITERATIONS"
export EXTRA_ARGS="$EXTRA_ARGS"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker - exit code 99 triggers requeue via scontrol
$WORKER_SCRIPT
EXIT_CODE=\$?

# If worker requested requeue (exit 99), tell SLURM to requeue this job
if [ \$EXIT_CODE -eq 99 ]; then
    echo "Worker requested requeue, notifying SLURM..."
    scontrol requeue \$SLURM_JOB_ID
    exit 0
fi

exit \$EXIT_CODE
EOF

# Phase 2: Cat jobs
CAT_SBATCH="$JOB_DIR/submit_cat.sbatch"
cat > "$CAT_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_cat_${BASE_NAME}
#SBATCH --output=$JOB_DIR/logs/cat_%a.out
#SBATCH --error=$JOB_DIR/logs/cat_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=1-${NUM_CAT_JOBS}%${NUM_CAT_JOBS}
#SBATCH --requeue
#SBATCH --open-mode=append

# Handle requeue signal - SLURM sends SIGUSR1 before requeueing
trap 'echo "Received requeue signal, job will be rescheduled..."' SIGUSR1

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables for worker
export JOB_CONFIG_FILE="$CAT_CONFIG"
export DATA_DIR="$DATA_DIR"
export YAML_CONFIG="$YAML_CONFIG"
export DATASET_PATH="$DATASET_PATH"
export RESOLUTION_ARG="$RESOLUTION_ARG"
export ITERATIONS="$ITERATIONS"
export EXTRA_ARGS="$EXTRA_ARGS"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker - exit code 99 triggers requeue via scontrol
$WORKER_SCRIPT
EXIT_CODE=\$?

# If worker requested requeue (exit 99), tell SLURM to requeue this job
if [ \$EXIT_CODE -eq 99 ]; then
    echo "Worker requested requeue, notifying SLURM..."
    scontrol requeue \$SLURM_JOB_ID
    exit 0
fi

exit \$EXIT_CODE
EOF

# Create logs directory
mkdir -p "$JOB_DIR/logs"

#=============================================================================
# Check for existing warmup checkpoints
#=============================================================================

echo "Checking for existing warmup checkpoints..."
ALL_CHECKPOINTS_EXIST=true
MISSING_CHECKPOINTS=""

for scene in "${SCENES[@]}"; do
    CKPT_PATH="${DATA_DIR}/${scene}/warmup_checkpoint.pth"
    if [ -f "$CKPT_PATH" ]; then
        echo "  ✓ $scene: checkpoint exists"
    else
        echo "  ✗ $scene: checkpoint missing"
        ALL_CHECKPOINTS_EXIST=false
        MISSING_CHECKPOINTS="$MISSING_CHECKPOINTS $scene"
    fi
done
echo ""

#=============================================================================
# Submit jobs
#=============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "  Submitting SLURM Jobs"
echo "════════════════════════════════════════════════════════════════════"

if [ "$ALL_CHECKPOINTS_EXIST" = true ]; then
    echo ""
    echo "All warmup checkpoints found! Running baseline and cat jobs in PARALLEL."
    echo "(Baseline jobs will skip training since checkpoints exist)"
    echo ""

    # Submit both phases without dependency - they run in parallel
    echo "Submitting baseline jobs..."
    BASELINE_JOB_ID=$(sbatch --parsable "$BASELINE_SBATCH")
    echo "  Baseline job array ID: $BASELINE_JOB_ID"

    echo "Submitting cat jobs (NO dependency - running in parallel)..."
    CAT_JOB_ID=$(sbatch --parsable "$CAT_SBATCH")
    echo "  Cat job array ID: $CAT_JOB_ID"

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Jobs Submitted Successfully! (PARALLEL MODE)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Baseline jobs: $BASELINE_JOB_ID"
    echo "  - $NUM_BASELINE_JOBS tasks (will skip - checkpoints exist)"
    echo ""
    echo "Cat jobs: $CAT_JOB_ID"
    echo "  - $NUM_CAT_JOBS tasks (running in parallel with baseline)"
    echo ""
else
    echo ""
    echo "Some warmup checkpoints missing:$MISSING_CHECKPOINTS"
    echo "Running baseline FIRST, then cat jobs."
    echo ""

    # Submit Phase 1 (baseline)
    echo "Submitting Phase 1 (baseline) jobs..."
    BASELINE_JOB_ID=$(sbatch --parsable "$BASELINE_SBATCH")
    echo "  Baseline job array ID: $BASELINE_JOB_ID"

    # Submit Phase 2 (cat) with dependency on Phase 1
    echo "Submitting Phase 2 (cat) jobs with dependency on Phase 1..."
    CAT_JOB_ID=$(sbatch --parsable --dependency=afterok:${BASELINE_JOB_ID} "$CAT_SBATCH")
    echo "  Cat job array ID: $CAT_JOB_ID"

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Jobs Submitted Successfully! (SEQUENTIAL MODE)"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Phase 1 (baseline): Job $BASELINE_JOB_ID"
    echo "  - $NUM_BASELINE_JOBS tasks (one per scene)"
    echo "  - Creates warmup_checkpoint.pth in each scene's data directory"
    echo ""
    echo "Phase 2 (cat): Job $CAT_JOB_ID"
    echo "  - $NUM_CAT_JOBS tasks (7 cat levels × $NUM_SCENES scenes)"
    echo "  - Depends on Phase 1 completion"
    echo "  - Uses warmup checkpoints from Phase 1"
fi

echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j $BASELINE_JOB_ID"
echo "  squeue -j $CAT_JOB_ID"
echo ""
echo "View logs:"
echo "  tail -f $JOB_DIR/logs/baseline_*.out"
echo "  tail -f $JOB_DIR/logs/cat_*.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $BASELINE_JOB_ID $CAT_JOB_ID"
echo ""

# Save job info for later reference
cat > "$JOB_DIR/job_info.txt" << EOF
Submission time: $(date)
Dataset: $DATASET
Base name: $BASE_NAME
Scenes: ${SCENES[@]}
Iterations: $ITERATIONS
Extra args: $EXTRA_ARGS

Phase 1 (baseline):
  Job ID: $BASELINE_JOB_ID
  Tasks: $NUM_BASELINE_JOBS

Phase 2 (cat):
  Job ID: $CAT_JOB_ID
  Tasks: $NUM_CAT_JOBS
  Dependency: afterok:$BASELINE_JOB_ID
EOF

echo "Job info saved to: $JOB_DIR/job_info.txt"
echo ""
