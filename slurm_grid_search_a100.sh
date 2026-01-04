#!/bin/bash
#=============================================================================
# SLURM Grid Search Script - A100 Only Full Parallelization
#=============================================================================
# This script runs ALL grid search experiments in parallel on A100 nodes only.
# One job per experiment for maximum parallelization.
#
# Usage:
#   ./slurm_grid_search_a100.sh <base_name>
#
# Example:
#   ./slurm_grid_search_a100.sh gridsearch1
#=============================================================================

set -e

# SLURM settings - A100 partitions only
PARTITIONS=("a100-4gpu-40gb" "a100-8gpu-40gb")
ACCOUNT="rctcd82061"
TIME_LIMIT="12:00:00"  # 12 hours per experiment

# Data paths for cluster
BASE_DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/nerf_synthetic"
SCENE_PATH="${BASE_DATA_DIR}/nerf_synthetic/chair"
YAML_CONFIG="./configs/nerfsyn.yaml"

# Fixed parameters for all experiments
BASE_ARGS="--yaml ./configs/nerfsyn.yaml --iterations 30000 --method cat --eval --hybrid_levels 5 --disable_c2f --mcmc --cap_max 100000"

# Grid search parameters
OPACITY_REGS=(0.001 0.0001)
SCALE_REGS=(0.001 0.0001)
NOISE_LRS=(1e2 1e3 1e4)
BCE_LAMBDAS=(0.1 0.01 0.001)  # BCE mode (not solo)

# Kernel parameters
GENERAL_LAMBDAS=(0.01 0.001 0)
FLEX_LAMBDAS=(0.01 0.001 0)

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_name>"
    echo ""
    echo "This will run ALL experiments in parallel on A100 nodes only!"
    echo ""
    echo "Total experiments: $((${#OPACITY_REGS[@]} * ${#SCALE_REGS[@]} * ${#NOISE_LRS[@]} * ${#BCE_LAMBDAS[@]} * (${#GENERAL_LAMBDAS[@]} + ${#FLEX_LAMBDAS[@]})))"
    echo ""
    echo "Available A100 resources:"
    echo "  - a100-4gpu-40gb: 4 idle nodes (16 GPUs)"
    echo "  - a100-8gpu-40gb: 2 idle nodes (16 GPUs)" 
    echo "  Total: 32 idle A100 GPUs available!"
    echo ""
    echo "With 32 GPUs running in parallel, all experiments should complete quickly."
    exit 1
fi

BASE_NAME=$1

# Create job config directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="slurm_jobs/gridsearch_a100_${BASE_NAME}_${TIMESTAMP}"
mkdir -p "$JOB_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  Nest-Splatting Grid Search - A100 ONLY"
echo "════════════════════════════════════════════════════════════════════"
echo "Base name:   $BASE_NAME"
echo "Scene:       chair"
echo "Job dir:     $JOB_DIR"
echo "Strategy:    One job per experiment on A100s only"
echo ""

#=============================================================================
# Generate ALL experiment configurations
#=============================================================================

echo "Generating ALL experiment configs..."
EXPERIMENT_CONFIG="$JOB_DIR/all_experiments.txt"
> "$EXPERIMENT_CONFIG"

# Function to convert decimal to scientific notation for filenames
format_for_filename() {
    local value=$1
    if [[ $value == "0.001" ]]; then
        echo "1e-3"
    elif [[ $value == "0.0001" ]]; then
        echo "1e-4"
    elif [[ $value == "0.01" ]]; then
        echo "1e-2"
    elif [[ $value == "0.1" ]]; then
        echo "1e-1"
    else
        echo "$value"
    fi
}

EXPERIMENT_COUNT=0

# Generate all experiments - one line per experiment
for op_reg in "${OPACITY_REGS[@]}"; do
    for sc_reg in "${SCALE_REGS[@]}"; do
        for noise_lr in "${NOISE_LRS[@]}"; do
            for bce_lambda in "${BCE_LAMBDAS[@]}"; do
                # Format values for filename
                op_formatted=$(format_for_filename $op_reg)
                sc_formatted=$(format_for_filename $sc_reg)
                bce_formatted=$(format_for_filename $bce_lambda)
                
                # General kernel experiments
                for lambda_shape in "${GENERAL_LAMBDAS[@]}"; do
                    lambda_formatted=$(format_for_filename $lambda_shape)
                    exp_name="general_${lambda_formatted}_ns_${noise_lr}_op_${op_formatted}_sc_${sc_formatted}_b_${bce_formatted}"
                    
                    # Format: exp_name kernel kernel_param op_reg sc_reg noise_lr bce_lambda
                    echo "$exp_name general $lambda_shape $op_reg $sc_reg $noise_lr $bce_lambda" >> "$EXPERIMENT_CONFIG"
                    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                done
                
                # Flex kernel experiments  
                for lambda_flex_beta in "${FLEX_LAMBDAS[@]}"; do
                    lambda_formatted=$(format_for_filename $lambda_flex_beta)
                    exp_name="flex_${lambda_formatted}_ns_${noise_lr}_op_${op_formatted}_sc_${sc_formatted}_b_${bce_formatted}"
                    
                    # Format: exp_name kernel kernel_param op_reg sc_reg noise_lr bce_lambda
                    echo "$exp_name flex $lambda_flex_beta $op_reg $sc_reg $noise_lr $bce_lambda" >> "$EXPERIMENT_CONFIG"
                    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                done
            done
        done
    done
done

echo "Total experiments generated: $EXPERIMENT_COUNT"
echo ""

#=============================================================================
# Create the worker script for individual experiments
#=============================================================================

WORKER_SCRIPT="$JOB_DIR/experiment_worker.sh"
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
#=============================================================================
# Single Experiment Worker - runs one specific experiment
#=============================================================================

set -e

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
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    import tinycudann as tcnn
    print(f'[GPU Check] OK - {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'[GPU Check] FATAL: GPU test failed - {e}')
    sys.exit(1)
"
    return $?
}

#=============================================================================
# Request requeue on bad GPU
#=============================================================================
MAX_REQUEUES=3

request_requeue() {
    local reason=$1
    REQUEUE_COUNT=${SLURM_RESTART_COUNT:-0}

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  REQUESTING REQUEUE - Bad GPU/Node detected"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Reason: $reason"
    echo "Host: $(hostname)"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "Requeue attempt: $((REQUEUE_COUNT + 1))/$MAX_REQUEUES"

    if [ $REQUEUE_COUNT -ge $MAX_REQUEUES ]; then
        echo "ERROR: Max requeue attempts reached!"
        exit 1
    fi

    echo "Exiting with code 99 to trigger SLURM requeue..."
    exit 99
}

# Read the experiment specification
EXPERIMENT_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT_CONFIG_FILE")
if [ -z "$EXPERIMENT_LINE" ]; then
    echo "ERROR: No experiment found at line $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse experiment line: exp_name kernel kernel_param op_reg sc_reg noise_lr bce_lambda
read -r EXP_NAME KERNEL KERNEL_PARAM OPACITY_REG SCALE_REG NOISE_LR BCE_LAMBDA <<< "$EXPERIMENT_LINE"

REQUEUE_COUNT=${SLURM_RESTART_COUNT:-0}
echo "════════════════════════════════════════════════════════════════════"
echo "  SLURM Experiment: $EXP_NAME"
if [ $REQUEUE_COUNT -gt 0 ]; then
    echo "  (Requeue attempt $REQUEUE_COUNT/$MAX_REQUEUES)"
fi
echo "════════════════════════════════════════════════════════════════════"
echo "Experiment:  $EXP_NAME"
echo "Kernel:      $KERNEL"
echo "Kernel param: $KERNEL_PARAM"
echo "Opacity reg: $OPACITY_REG"
echo "Scale reg:   $SCALE_REG"
echo "Noise LR:    $NOISE_LR"
echo "BCE lambda:  $BCE_LAMBDA"
echo "Host:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Started:     $(date)"
echo "════════════════════════════════════════════════════════════════════"

# Check output path and skip if completed
OUTPUT_PATH="outputs/nerf_synthetic/chair/cat/${EXP_NAME}_5_levels"
TEST_METRICS="${OUTPUT_PATH}/test_metrics.txt"

if [ -f "$TEST_METRICS" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SKIPPING - Already completed"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Found: $TEST_METRICS"
    echo "Finished: $(date)"
    exit 0
fi

# Validate GPU before running
if ! validate_gpu; then
    request_requeue "GPU validation failed"
fi

# Unique port for this experiment
PORT=$((6009 + SLURM_ARRAY_TASK_ID))

# Build command based on kernel type
if [ "$KERNEL" = "general" ]; then
    KERNEL_ARGS="--kernel general --lambda_shape $KERNEL_PARAM"
else
    KERNEL_ARGS="--kernel flex --lambda_flex_beta $KERNEL_PARAM"
fi

CMD="python train.py -s $SCENE_PATH -m $EXP_NAME $BASE_ARGS --opacity_reg $OPACITY_REG --scale_reg $SCALE_REG --noise_lr $NOISE_LR --bce_lambda $BCE_LAMBDA $KERNEL_ARGS --port $PORT"

echo ""
echo "Command: $CMD"
echo ""

# Run the experiment
set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ SUCCESS: $EXP_NAME"
    echo "════════════════════════════════════════════════════════════════════"
    echo "Finished: $(date)"
    exit 0
else
    echo "  ✗ FAILED: $EXP_NAME (exit code: $EXIT_CODE)"
    echo "════════════════════════════════════════════════════════════════════"
    
    # Request requeue for GPU-related failures
    request_requeue "Experiment failed with exit code $EXIT_CODE"
fi
WORKER_EOF

chmod +x "$WORKER_SCRIPT"

#=============================================================================
# Create SLURM submission scripts for A100 partitions
#=============================================================================

echo "Creating submission scripts for A100 partitions..."
SUBMITTED_JOBS=()

# Split experiments between the two A100 partitions
# a100-4gpu-40gb gets first half, a100-8gpu-40gb gets second half
SPLIT_POINT=$((EXPERIMENT_COUNT / 2))

# Partition 1: a100-4gpu-40gb (experiments 1 to SPLIT_POINT)
PARTITION1="a100-4gpu-40gb"
PARTITION1_START=1
PARTITION1_END=$SPLIT_POINT
PARTITION1_COUNT=$SPLIT_POINT

echo "  $PARTITION1: experiments $PARTITION1_START-$PARTITION1_END ($PARTITION1_COUNT experiments)"

PARTITION1_SBATCH="$JOB_DIR/submit_${PARTITION1}.sbatch"
cat > "$PARTITION1_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION1
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_${BASE_NAME}_4gpu
#SBATCH --output=$JOB_DIR/logs/4gpu_%a.out
#SBATCH --error=$JOB_DIR/logs/4gpu_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=${PARTITION1_START}-${PARTITION1_END}
#SBATCH --requeue
#SBATCH --open-mode=append

# Handle requeue signal
trap 'echo "Received requeue signal, job will be rescheduled..."' SIGUSR1

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables
export EXPERIMENT_CONFIG_FILE="$EXPERIMENT_CONFIG"
export SCENE_PATH="$SCENE_PATH"
export BASE_ARGS="$BASE_ARGS"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
$WORKER_SCRIPT
EXIT_CODE=\$?

# Handle requeue requests
if [ \$EXIT_CODE -eq 99 ]; then
    echo "Worker requested requeue, notifying SLURM..."
    scontrol requeue \$SLURM_JOB_ID
    exit 0
fi

exit \$EXIT_CODE
EOF

# Partition 2: a100-8gpu-40gb (experiments SPLIT_POINT+1 to END)
PARTITION2="a100-8gpu-40gb"
PARTITION2_START=$((SPLIT_POINT + 1))
PARTITION2_END=$EXPERIMENT_COUNT
PARTITION2_COUNT=$((EXPERIMENT_COUNT - SPLIT_POINT))

echo "  $PARTITION2: experiments $PARTITION2_START-$PARTITION2_END ($PARTITION2_COUNT experiments)"

PARTITION2_SBATCH="$JOB_DIR/submit_${PARTITION2}.sbatch"
cat > "$PARTITION2_SBATCH" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION2
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_${BASE_NAME}_8gpu
#SBATCH --output=$JOB_DIR/logs/8gpu_%a.out
#SBATCH --error=$JOB_DIR/logs/8gpu_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=${PARTITION2_START}-${PARTITION2_END}
#SBATCH --requeue
#SBATCH --open-mode=append

# Handle requeue signal
trap 'echo "Received requeue signal, job will be rescheduled..."' SIGUSR1

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables
export EXPERIMENT_CONFIG_FILE="$EXPERIMENT_CONFIG"
export SCENE_PATH="$SCENE_PATH"
export BASE_ARGS="$BASE_ARGS"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
$WORKER_SCRIPT
EXIT_CODE=\$?

# Handle requeue requests
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
# Submit jobs to both A100 partitions
#=============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Submitting A100 Grid Search Jobs"
echo "════════════════════════════════════════════════════════════════════"

echo "Submitting to $PARTITION1 (experiments $PARTITION1_START-$PARTITION1_END)..."
JOB1_ID=$(sbatch --parsable "$PARTITION1_SBATCH")
SUBMITTED_JOBS+=("$JOB1_ID")
echo "  Job ID: $JOB1_ID"

echo "Submitting to $PARTITION2 (experiments $PARTITION2_START-$PARTITION2_END)..."
JOB2_ID=$(sbatch --parsable "$PARTITION2_SBATCH")
SUBMITTED_JOBS+=("$JOB2_ID")
echo "  Job ID: $JOB2_ID"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL A100 JOBS SUBMITTED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Total experiments: $EXPERIMENT_COUNT"
echo "A100-4GPU partition: $PARTITION1_COUNT experiments (Job $JOB1_ID)"
echo "A100-8GPU partition: $PARTITION2_COUNT experiments (Job $JOB2_ID)"
echo ""
echo "With 32 A100 GPUs available, experiments will run in waves."
echo "Estimated completion time: ~$(((EXPERIMENT_COUNT + 31) / 32)) waves of experiments"
echo ""

echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1_ID"
echo "  squeue -j $JOB2_ID"
echo ""

echo "View logs:"
echo "  tail -f $JOB_DIR/logs/4gpu_*.out"
echo "  tail -f $JOB_DIR/logs/8gpu_*.out"
echo ""

echo "Cancel all jobs:"
echo "  scancel $JOB1_ID $JOB2_ID"
echo ""

# Save job info
cat > "$JOB_DIR/job_info.txt" << EOF
Grid Search - A100 Only Full Parallelization
============================================
Submission time: $(date)
Base name: $BASE_NAME
Scene: chair
Strategy: One job per experiment on A100s only

Total experiments: $EXPERIMENT_COUNT
A100 partitions used: 2
Job IDs: ${SUBMITTED_JOBS[*]}

Distribution:
- a100-4gpu-40gb: experiments $PARTITION1_START-$PARTITION1_END ($PARTITION1_COUNT jobs) - Job $JOB1_ID
- a100-8gpu-40gb: experiments $PARTITION2_START-$PARTITION2_END ($PARTITION2_COUNT jobs) - Job $JOB2_ID

Parameters tested:
- Opacity reg: ${OPACITY_REGS[@]}
- Scale reg: ${SCALE_REGS[@]}
- Noise LR: ${NOISE_LRS[@]}
- BCE lambda: ${BCE_LAMBDAS[@]}
- General kernel lambdas: ${GENERAL_LAMBDAS[@]}
- Flex kernel lambdas: ${FLEX_LAMBDAS[@]}

Output naming: kernel_kernellambda_ns_noisevalue_op_opvalue_sc_scalevalue_b_bcelambda
EOF

echo "Job info saved to: $JOB_DIR/job_info.txt"
echo ""
echo "🚀 A100s are locked and loaded! Grid search initiated! 🚀"