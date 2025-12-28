#!/bin/bash
#=============================================================================
# SLURM Retry Failed Jobs Script
#=============================================================================
# This script checks a completed job directory and resubmits failed experiments.
# It uses the existing skip logic in worker.sh - completed jobs have test_metrics.txt
#
# Usage:
#   ./slurm_retry_failed.sh <job_dir> [max_retries]
#
# Examples:
#   ./slurm_retry_failed.sh slurm_jobs/exp1_20251224_073342
#   ./slurm_retry_failed.sh slurm_jobs/exp1_20251224_073342 3
#=============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job_dir> [max_retries]"
    echo ""
    echo "Arguments:"
    echo "  job_dir      Required. Path to the slurm job directory"
    echo "  max_retries  Optional. Maximum retry attempts (default: 3)"
    exit 1
fi

JOB_DIR=$1
MAX_RETRIES=${2:-3}

if [ ! -d "$JOB_DIR" ]; then
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi

# Check for required files
BASELINE_CONFIG="$JOB_DIR/baseline_jobs.txt"
CAT_CONFIG="$JOB_DIR/cat_jobs.txt"
WORKER_SCRIPT="$JOB_DIR/worker.sh"

if [ ! -f "$BASELINE_CONFIG" ] || [ ! -f "$CAT_CONFIG" ]; then
    echo "ERROR: Job config files not found in $JOB_DIR"
    exit 1
fi

# Read job info
source_info() {
    # Extract variables from the sbatch files
    PARTITION=$(grep -oP '(?<=--partition=)\S+' "$JOB_DIR/submit_baseline.sbatch" | head -1)
    ACCOUNT=$(grep -oP '(?<=--account=)\S+' "$JOB_DIR/submit_baseline.sbatch" | head -1)
    TIME_LIMIT=$(grep -oP '(?<=--time=)\S+' "$JOB_DIR/submit_baseline.sbatch" | head -1)
}

source_info

echo "════════════════════════════════════════════════════════════════════"
echo "  Retry Failed Jobs"
echo "════════════════════════════════════════════════════════════════════"
echo "Job directory: $JOB_DIR"
echo "Max retries:   $MAX_RETRIES"
echo "Partition:     $PARTITION"
echo "Account:       $ACCOUNT"
echo ""

# Function to check if a job completed successfully
check_job_completed() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        if grep -q "SUCCESS" "$log_file" || grep -q "SKIPPING" "$log_file"; then
            return 0  # Completed
        fi
    fi
    return 1  # Failed or not run
}

# Function to get failed node from log
get_failed_node() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        grep -oP '(?<=Host:        )\S+' "$log_file" | tail -1
    fi
}

# Collect failed baseline jobs
echo "Checking baseline jobs..."
FAILED_BASELINE_TASKS=""
FAILED_NODES=""
BASELINE_COUNT=$(wc -l < "$BASELINE_CONFIG")

for i in $(seq 1 $BASELINE_COUNT); do
    log_file="$JOB_DIR/logs/baseline_${i}.out"
    if ! check_job_completed "$log_file"; then
        FAILED_BASELINE_TASKS="$FAILED_BASELINE_TASKS $i"
        node=$(get_failed_node "$log_file")
        if [ -n "$node" ]; then
            FAILED_NODES="$FAILED_NODES $node"
        fi
        scene=$(sed -n "${i}p" "$BASELINE_CONFIG" | awk '{print $1}')
        echo "  [FAILED] Task $i: $scene (node: ${node:-unknown})"
    fi
done

# Collect failed cat jobs
echo ""
echo "Checking cat jobs..."
FAILED_CAT_TASKS=""
CAT_COUNT=$(wc -l < "$CAT_CONFIG")

for i in $(seq 1 $CAT_COUNT); do
    log_file="$JOB_DIR/logs/cat_${i}.out"
    if ! check_job_completed "$log_file"; then
        FAILED_CAT_TASKS="$FAILED_CAT_TASKS $i"
        node=$(get_failed_node "$log_file")
        if [ -n "$node" ]; then
            FAILED_NODES="$FAILED_NODES $node"
        fi
        job_info=$(sed -n "${i}p" "$CAT_CONFIG")
        scene=$(echo "$job_info" | awk '{print $1}')
        hl=$(echo "$job_info" | grep -oP '(?<=--hybrid_levels )\d+')
        echo "  [FAILED] Task $i: $scene cat$hl (node: ${node:-unknown})"
    fi
done

# Get unique failed nodes
UNIQUE_FAILED_NODES=$(echo $FAILED_NODES | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════════"
echo "Failed baseline tasks:${FAILED_BASELINE_TASKS:-" none"}"
echo "Failed cat tasks:${FAILED_CAT_TASKS:-" none"}"
echo "Problematic nodes: ${UNIQUE_FAILED_NODES:-none}"
echo ""

if [ -z "$FAILED_BASELINE_TASKS" ] && [ -z "$FAILED_CAT_TASKS" ]; then
    echo "All jobs completed successfully! Nothing to retry."
    exit 0
fi

# Create retry sbatch scripts
RETRY_DIR="$JOB_DIR/retry_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RETRY_DIR/logs"

# Build exclude list from failed nodes
EXCLUDE_OPTION=""
if [ -n "$UNIQUE_FAILED_NODES" ]; then
    EXCLUDE_OPTION="#SBATCH --exclude=$UNIQUE_FAILED_NODES"
    echo "Will exclude nodes: $UNIQUE_FAILED_NODES"
fi

# Resubmit failed baseline jobs
if [ -n "$FAILED_BASELINE_TASKS" ]; then
    BASELINE_ARRAY=$(echo $FAILED_BASELINE_TASKS | tr ' ' ',' | sed 's/^,//')

    cat > "$RETRY_DIR/retry_baseline.sbatch" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=retry_baseline
#SBATCH --output=$RETRY_DIR/logs/baseline_%a.out
#SBATCH --error=$RETRY_DIR/logs/baseline_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=$BASELINE_ARRAY
$EXCLUDE_OPTION

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Verify GPU before running
echo "Checking GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

# Export environment variables for worker
export JOB_CONFIG_FILE="$BASELINE_CONFIG"
$(grep "^export" "$JOB_DIR/submit_baseline.sbatch" | grep -v JOB_CONFIG_FILE)

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
$WORKER_SCRIPT
EOF

    echo ""
    echo "Submitting retry for baseline tasks: $BASELINE_ARRAY"
    RETRY_BASELINE_ID=$(sbatch --parsable "$RETRY_DIR/retry_baseline.sbatch")
    echo "  Job ID: $RETRY_BASELINE_ID"
fi

# Resubmit failed cat jobs
if [ -n "$FAILED_CAT_TASKS" ]; then
    CAT_ARRAY=$(echo $FAILED_CAT_TASKS | tr ' ' ',' | sed 's/^,//')

    # If there were baseline retries, add dependency
    DEPENDENCY_OPTION=""
    if [ -n "$RETRY_BASELINE_ID" ]; then
        DEPENDENCY_OPTION="#SBATCH --dependency=afterok:$RETRY_BASELINE_ID"
    fi

    cat > "$RETRY_DIR/retry_cat.sbatch" << EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --job-name=retry_cat
#SBATCH --output=$RETRY_DIR/logs/cat_%a.out
#SBATCH --error=$RETRY_DIR/logs/cat_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=$CAT_ARRAY
$EXCLUDE_OPTION
$DEPENDENCY_OPTION

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Verify GPU before running
echo "Checking GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

# Export environment variables for worker
export JOB_CONFIG_FILE="$CAT_CONFIG"
$(grep "^export" "$JOB_DIR/submit_cat.sbatch" | grep -v JOB_CONFIG_FILE)

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
$WORKER_SCRIPT
EOF

    echo ""
    echo "Submitting retry for cat tasks: $CAT_ARRAY"
    RETRY_CAT_ID=$(sbatch --parsable "$RETRY_DIR/retry_cat.sbatch")
    echo "  Job ID: $RETRY_CAT_ID"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Retry Jobs Submitted"
echo "════════════════════════════════════════════════════════════════════"
echo "Retry logs: $RETRY_DIR/logs/"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
