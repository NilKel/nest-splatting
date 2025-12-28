#!/bin/bash
#SBATCH --partition=a100-4gpu-40gb
#SBATCH --account=rctcd82061
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_cat_exp1
#SBATCH --output=slurm_jobs/exp1_20251224_124149/logs/cat_%a.out
#SBATCH --error=slurm_jobs/exp1_20251224_124149/logs/cat_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-49%49

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables for worker
export JOB_CONFIG_FILE="slurm_jobs/exp1_20251224_124149/cat_jobs.txt"
export DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/mip_360"
export YAML_CONFIG="PER_SCENE"
export DATASET_PATH="mip_360"
export RESOLUTION_ARG="PER_SCENE"
export ITERATIONS="30000"
export EXTRA_ARGS="--disable_c2f"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
slurm_jobs/exp1_20251224_124149/worker.sh
