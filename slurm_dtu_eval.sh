#!/bin/bash
#=============================================================================
# SLURM Parallel DTU Evaluation Script
#=============================================================================
# This script runs DTU training and evaluation in parallel using SLURM job arrays.
# Consistent with scripts/dtu_eval.py but parallelized across GPUs.
#
# Usage:
#   ./slurm_dtu_eval.sh <exp_name> [scene_names] [iterations] [extra_args]
#
# Examples:
#   ./slurm_dtu_eval.sh dtu_baseline                    # All DTU scenes
#   ./slurm_dtu_eval.sh dtu_test scan24,scan37          # Specific scenes
#   ./slurm_dtu_eval.sh dtu_exp all 30000 "--some_flag" # With extra args
#
# The script will:
#   1. Submit training jobs as a job array (one per scene)
#   2. Submit rendering jobs with dependency on training
#   3. Submit a single metrics job with dependency on rendering
#=============================================================================

set -e

# DTU scenes (same as dtu_eval.py)
ALL_SCENES="scan24,scan37,scan40,scan55,scan63,scan65,scan69,scan83,scan97,scan105,scan106,scan110,scan114,scan118,scan122"

# Default values (consistent with dtu_eval.py)
DEFAULT_ITERATIONS=30000
DEFAULT_YAML="./configs/2dgs.yaml"
DATASET_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/dtu/2DGS_data/DTU"

# SLURM settings
PARTITION="a100-4gpu-40gb"
ACCOUNT="rctcd82061"
TIME_LIMIT="12:00:00"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [scene_names] [iterations] [yaml_config] [extra_args]"
    echo ""
    echo "Arguments:"
    echo "  exp_name      Required. Experiment name (output saved to ./output/<exp_name>)"
    echo "  scene_names   Optional. Comma-separated scene names or 'all' (default: all)"
    echo "  iterations    Optional. Number of training iterations (default: 30000)"
    echo "  yaml_config   Optional. YAML config file (default: ./configs/2dgs.yaml)"
    echo "  extra_args    Optional. Extra arguments to pass to train.py"
    echo ""
    echo "Examples:"
    echo "  $0 dtu_baseline"
    echo "  $0 dtu_test scan24,scan37"
    echo "  $0 dtu_exp all 30000 ./configs/nerfsyn.yaml"
    echo ""
    echo "DTU scenes: $ALL_SCENES"
    exit 1
fi

EXP_NAME=$1
SCENE_NAMES=${2:-all}
ITERATIONS=${3:-$DEFAULT_ITERATIONS}
YAML_CONFIG=${4:-$DEFAULT_YAML}
EXTRA_ARGS=${5:-""}

OUTPUT_DIR="./output/${EXP_NAME}"

# Handle "all" keyword
if [ "$SCENE_NAMES" = "all" ]; then
    SCENE_NAMES=$ALL_SCENES
fi

# Convert comma-separated list to array
IFS=',' read -ra SCENES <<< "$SCENE_NAMES"
NUM_SCENES=${#SCENES[@]}

# Create job config directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="slurm_jobs/dtu_${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$JOB_DIR/logs"

echo "════════════════════════════════════════════════════════════════════"
echo "  DTU Evaluation - SLURM Parallel"
echo "════════════════════════════════════════════════════════════════════"
echo "Experiment:  $EXP_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Scenes:      ${SCENES[@]}"
echo "Iterations:  $ITERATIONS"
echo "YAML config: $YAML_CONFIG"
echo "Dataset dir: $DATASET_DIR"
echo "Job dir:     $JOB_DIR"
echo "════════════════════════════════════════════════════════════════════"
echo ""

#=============================================================================
# Generate job configuration file
#=============================================================================

echo "Generating job configs..."
JOB_CONFIG="$JOB_DIR/scenes.txt"
> "$JOB_CONFIG"

for scene in "${SCENES[@]}"; do
    echo "$scene" >> "$JOB_CONFIG"
done

echo "Created config for $NUM_SCENES scenes"
echo ""

#=============================================================================
# Create worker scripts
#=============================================================================

# Training worker
TRAIN_WORKER="$JOB_DIR/train_worker.sh"
cat > "$TRAIN_WORKER" << 'WORKER_EOF'
#!/bin/bash
set -e

# Read scene from config file
SCENE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_CONFIG_FILE")
if [ -z "$SCENE" ]; then
    echo "ERROR: No scene found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  DTU Training: $SCENE"
echo "════════════════════════════════════════════════════════════════════"
echo "Host:     $(hostname)"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"
echo ""

SOURCE="${DATASET_DIR}/${SCENE}"
SAVE_DIR="${OUTPUT_DIR}/${SCENE}"

# Check if scene exists
if [ ! -d "$SOURCE" ]; then
    echo "ERROR: Scene path does not exist: $SOURCE"
    exit 1
fi

# Assign unique port per task
PORT=$((6009 + SLURM_ARRAY_TASK_ID))

# Training command (consistent with dtu_eval.py)
CMD="python train.py -s $SOURCE --scene_name $SCENE -m $SAVE_DIR --yaml $YAML_CONFIG --ip 127.0.0.1 --port $PORT -r 2 --eval $EXTRA_ARGS"

echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Training complete: $SCENE"
echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
WORKER_EOF
chmod +x "$TRAIN_WORKER"

# Rendering worker
RENDER_WORKER="$JOB_DIR/render_worker.sh"
cat > "$RENDER_WORKER" << 'WORKER_EOF'
#!/bin/bash
set -e

# Read scene from config file
SCENE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_CONFIG_FILE")
if [ -z "$SCENE" ]; then
    echo "ERROR: No scene found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  DTU Rendering: $SCENE"
echo "════════════════════════════════════════════════════════════════════"
echo "Host:     $(hostname)"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"
echo ""

SOURCE="${DATASET_DIR}/${SCENE}"
SAVE_DIR="${OUTPUT_DIR}/${SCENE}"

# Rendering command (consistent with dtu_eval.py)
CMD="python eval_render.py --iteration $ITERATIONS -s $SOURCE -m $SAVE_DIR --yaml $YAML_CONFIG --scene $SCENE --quiet --skip_train --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"

echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Rendering complete: $SCENE"
echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
WORKER_EOF
chmod +x "$RENDER_WORKER"

#=============================================================================
# Create SLURM submission scripts
#=============================================================================

# Training sbatch
TRAIN_SBATCH="$JOB_DIR/submit_train.sbatch"
cat > "$TRAIN_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=dtu_train_${EXP_NAME}
#SBATCH --output=$JOB_DIR/logs/train_%a.out
#SBATCH --error=$JOB_DIR/logs/train_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=1-${NUM_SCENES}

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate baseline_nest

# Export variables for worker
export JOB_CONFIG_FILE="$JOB_CONFIG"
export DATASET_DIR="$DATASET_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"
export YAML_CONFIG="$YAML_CONFIG"
export ITERATIONS="$ITERATIONS"
export EXTRA_ARGS="$EXTRA_ARGS"

# Change to project directory
cd ~/userdir/Projects/baseline_nest/nest-splatting

# Run worker
$TRAIN_WORKER
EOF

# Rendering sbatch
RENDER_SBATCH="$JOB_DIR/submit_render.sbatch"
cat > "$RENDER_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=dtu_render_${EXP_NAME}
#SBATCH --output=$JOB_DIR/logs/render_%a.out
#SBATCH --error=$JOB_DIR/logs/render_%a.err
#SBATCH --time=02:00:00
#SBATCH --array=1-${NUM_SCENES}

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate baseline_nest

# Export variables for worker
export JOB_CONFIG_FILE="$JOB_CONFIG"
export DATASET_DIR="$DATASET_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"
export YAML_CONFIG="$YAML_CONFIG"
export ITERATIONS="$ITERATIONS"

# Change to project directory
cd ~/userdir/Projects/baseline_nest/nest-splatting

# Run worker
$RENDER_WORKER
EOF

# Metrics sbatch (single job, not array)
METRICS_SBATCH="$JOB_DIR/submit_metrics.sbatch"
cat > "$METRICS_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=dtu_metrics_${EXP_NAME}
#SBATCH --output=$JOB_DIR/logs/metrics.out
#SBATCH --error=$JOB_DIR/logs/metrics.err
#SBATCH --time=01:00:00

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate baseline_nest

# Change to project directory
cd ~/userdir/Projects/baseline_nest/nest-splatting

echo "════════════════════════════════════════════════════════════════════"
echo "  DTU Metrics Evaluation"
echo "════════════════════════════════════════════════════════════════════"
echo "Output dir: $OUTPUT_DIR"
echo "Iteration:  $ITERATIONS"
echo "Started:    \$(date)"
echo ""

# Metrics command (consistent with dtu_eval.py)
python scripts/metric_eval.py $OUTPUT_DIR $ITERATIONS

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Metrics complete"
echo "════════════════════════════════════════════════════════════════════"
echo "Finished: \$(date)"
EOF

#=============================================================================
# Submit jobs with dependencies
#=============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "  Submitting SLURM Jobs"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Submit training
echo "Submitting training jobs..."
TRAIN_JOB_ID=$(sbatch --parsable "$TRAIN_SBATCH")
echo "  Training job array ID: $TRAIN_JOB_ID"

# Submit rendering with dependency on training
echo "Submitting rendering jobs (depends on training)..."
RENDER_JOB_ID=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB_ID} "$RENDER_SBATCH")
echo "  Rendering job array ID: $RENDER_JOB_ID"

# Submit metrics with dependency on rendering
echo "Submitting metrics job (depends on rendering)..."
METRICS_JOB_ID=$(sbatch --parsable --dependency=afterok:${RENDER_JOB_ID} "$METRICS_SBATCH")
echo "  Metrics job ID: $METRICS_JOB_ID"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Jobs Submitted Successfully!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Training:   $TRAIN_JOB_ID ($NUM_SCENES tasks)"
echo "Rendering:  $RENDER_JOB_ID ($NUM_SCENES tasks, depends on training)"
echo "Metrics:    $METRICS_JOB_ID (single task, depends on rendering)"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f $JOB_DIR/logs/train_*.out"
echo "  tail -f $JOB_DIR/logs/render_*.out"
echo "  tail -f $JOB_DIR/logs/metrics.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $TRAIN_JOB_ID $RENDER_JOB_ID $METRICS_JOB_ID"
echo ""

# Save job info
cat > "$JOB_DIR/job_info.txt" << EOF
Submission time: $(date)
Experiment: $EXP_NAME
Output dir: $OUTPUT_DIR
Scenes: ${SCENES[@]}
Iterations: $ITERATIONS
YAML config: $YAML_CONFIG
Extra args: $EXTRA_ARGS

Training:
  Job ID: $TRAIN_JOB_ID
  Tasks: $NUM_SCENES

Rendering:
  Job ID: $RENDER_JOB_ID
  Tasks: $NUM_SCENES
  Dependency: afterok:$TRAIN_JOB_ID

Metrics:
  Job ID: $METRICS_JOB_ID
  Dependency: afterok:$RENDER_JOB_ID
EOF

echo "Job info saved to: $JOB_DIR/job_info.txt"
