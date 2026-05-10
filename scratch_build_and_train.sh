#!/bin/bash
# Detached build + train for cat-mode shared-mem cache experiment.
# Survives terminal close: launched via nohup setsid.

set -e

LOG_DIR="/home/nilkel/Projects/nest-splatting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/build_and_train_synth_style5_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$LOG" "$LOG_DIR/build_and_train_synth_style5_latest.log"

echo "==== START $(date) ====" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# 1) Build
echo "==== BUILD diff-surfel-rasterization ====" | tee -a "$LOG"
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda run -n nest_splatting --no-capture-output python -m pip install -e . --no-build-isolation >> "$LOG" 2>&1

BUILD_RC=$?
echo "==== BUILD finished rc=$BUILD_RC at $(date) ====" | tee -a "$LOG"
if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD FAILED. Skipping training." | tee -a "$LOG"
    exit $BUILD_RC
fi

# 2) Train
echo "==== TRAIN synth_style5 (cat hybrid_levels=5, no mask/normal/dist regs) ====" | tee -a "$LOG"
cd /home/nilkel/Projects/nest-splatting
conda run -n nest_splatting --no-capture-output python train.py \
  -s /home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU/scan24 \
  -m synth_style5 \
  --yaml ./configs/dtu.yaml \
  --eval \
  --iterations 30000 \
  -r 2 \
  --method cat --hybrid_levels 5 \
  --lambda_mask 0.0 \
  --lambda_normal 0.0 \
  --lambda_dist 0.0 \
  >> "$LOG" 2>&1

TRAIN_RC=$?
echo "==== TRAIN finished rc=$TRAIN_RC at $(date) ====" | tee -a "$LOG"
exit $TRAIN_RC
