#!/bin/bash
#SBATCH --partition=a100-4gpu-40gb
#SBATCH --account=rctcd82061
#SBATCH --gres=gpu:1
#SBATCH --job-name=nest_baseline_mcop001sc001ns1e3bc
#SBATCH --output=slurm_jobs/mcop001sc001ns1e3bc_20251225_064500/logs/baseline_%a.out
#SBATCH --error=slurm_jobs/mcop001sc001ns1e3bc_20251225_064500/logs/baseline_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=1-7%7

# Setup environment
source ~/userdir/miniconda3/etc/profile.d/conda.sh
conda activate nest_splatting

# Export environment variables for worker
export JOB_CONFIG_FILE="slurm_jobs/mcop001sc001ns1e3bc_20251225_064500/baseline_jobs.txt"
export DATA_DIR="/data/rg_data/aig/users/z0051beu/Projects/data/mip_360"
export YAML_CONFIG="PER_SCENE"
export DATASET_PATH="mip_360"
export RESOLUTION_ARG="PER_SCENE"
export ITERATIONS="30000"
export EXTRA_ARGS="--disable_c2f --mcmc --cap_max 100000 --opacity_reg 0.001 --scale_reg 0.001 --noise_lr 1e3 --bce"

# Change to project directory
cd /data/rg_data/aig/users/z0051beu/Projects/nest-splatting

# Run worker
slurm_jobs/mcop001sc001ns1e3bc_20251225_064500/worker.sh
