#!/bin/bash
#=============================================================================
# SLURM Benchmark Script for Beta-Splatting on DTU
#=============================================================================
# Runs beta-splatting on all 15 DTU scenes with:
#   - Black background (default, no flag needed)
#   - NeST-splatting test camera holdout (patched dataset_readers.py)
#   - Resolution -r 2 (standard for DTU)
#
# Usage:
#   sbatch slurm_benchmark_beta_dtu.sh [cap_max] [output_dir]
#
# Examples:
#   sbatch slurm_benchmark_beta_dtu.sh                    # 1M budget, ./eval_dtu
#   sbatch slurm_benchmark_beta_dtu.sh 100000             # 100k budget, ./eval_dtu_100k
#   sbatch slurm_benchmark_beta_dtu.sh 50000              # 50k budget, ./eval_dtu_50k
#   sbatch slurm_benchmark_beta_dtu.sh 100000 ./my_dir    # 100k budget, custom dir
#   sbatch --array=1-5 slurm_benchmark_beta_dtu.sh        # specific scenes
#=============================================================================

#SBATCH --partition=a100-4gpu-40gb
#SBATCH --account=rctcd82061
#SBATCH --gres=gpu:1
#SBATCH --job-name=beta_dtu
#SBATCH --output=slurm_logs/beta_dtu_%a_%j.out
#SBATCH --error=slurm_logs/beta_dtu_%a_%j.err
#SBATCH --time=12:00:00
#SBATCH --array=1-15
#SBATCH --exclude=node185,node188

set -e

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/dtu/2DGS_data/DTU"
BETA_SPLATTING_DIR="/home/z0051beu/userdir/Projects/beta-splatting"

# Accept cap_max and output_dir as positional args
CAP_MAX=${1:-1000000}
OUTPUT_DIR=${2:-"./eval_dtu_${CAP_MAX}"}

# DTU scenes (same order as NeST dtu_eval.py)
declare -A SCENES
SCENES[1]="scan24"
SCENES[2]="scan37"
SCENES[3]="scan40"
SCENES[4]="scan55"
SCENES[5]="scan63"
SCENES[6]="scan65"
SCENES[7]="scan69"
SCENES[8]="scan83"
SCENES[9]="scan97"
SCENES[10]="scan105"
SCENES[11]="scan106"
SCENES[12]="scan110"
SCENES[13]="scan114"
SCENES[14]="scan118"
SCENES[15]="scan122"


# ============================================================================
# Setup Environment
# ============================================================================
echo "=================================================================="
echo "  Beta-Splatting DTU Benchmark"
echo "=================================================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:        $(hostname)"
echo "Started:     $(date)"
echo "=================================================================="

# Load required modules
module load gcc/12.1.0
module load cuda12.1/toolkit/12.1.0

# Activate conda environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate beta_splatting

# Change to beta-splatting directory
cd "$BETA_SPLATTING_DIR"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

# ============================================================================
# Get scene configuration for this task
# ============================================================================
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$SCENE" ]; then
    echo "ERROR: Invalid task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

SOURCE_PATH="${DATA_DIR}/${SCENE}"
MODEL_PATH="${OUTPUT_DIR}/${SCENE}"

echo ""
echo "Scene:       $SCENE"
echo "Source:      $SOURCE_PATH"
echo "Output:      $MODEL_PATH"
echo "Resolution:  -r 2"
echo "Cap Max:     $CAP_MAX"
echo ""

# Check if source exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# Check if already completed
METRICS_FILE="${MODEL_PATH}/point_cloud/iteration_best/metrics.json"
if [ -f "$METRICS_FILE" ]; then
    echo "=================================================================="
    echo "  SKIPPING - Already completed"
    echo "=================================================================="
    echo "Found: $METRICS_FILE"
    cat "$METRICS_FILE"
    exit 0
fi

# ============================================================================
# Run Training with Evaluation
# ============================================================================
echo "=================================================================="
echo "  Starting Training"
echo "=================================================================="

CMD="python train.py \
    -s $SOURCE_PATH \
    -m $MODEL_PATH \
    -r 2 \
    --cap_max $CAP_MAX \
    --eval \
    --disable_viewer \
    --quiet"

echo "Command: $CMD"
echo ""

$CMD
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=================================================================="
    echo "  FAILED (exit code: $EXIT_CODE)"
    echo "=================================================================="
    exit $EXIT_CODE
fi

# ============================================================================
# Display Results
# ============================================================================
echo ""
echo "=================================================================="
echo "  Training Complete"
echo "=================================================================="

if [ -f "$METRICS_FILE" ]; then
    echo "Metrics:"
    cat "$METRICS_FILE"
else
    echo "WARNING: Metrics file not found at $METRICS_FILE"
fi

echo ""
echo "Finished: $(date)"
echo "=================================================================="
