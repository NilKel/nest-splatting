#!/bin/bash
# Wait for the GPU to be idle (no train.py processes), then run
# cat hl=2 on DTU scan24 with the same flags as the hl=5 run.

set -u
cd /home/nilkel/Projects/nest-splatting

LOG_DIR="/home/nilkel/Projects/nest-splatting/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
WAIT_LOG="$LOG_DIR/cat_hl2_wait_then_run_${TS}.log"
ln -sfn "$WAIT_LOG" "$LOG_DIR/cat_hl2_latest.log"

NAME="synth_style_cat_randombg"   # train.py auto-appends _2_levels for cat hl=2
OUTDIR="outputs/DTU/scan24/cat/${NAME}_2_levels"

echo "==== WAIT-AND-RUN START $(date) ====" | tee -a "$WAIT_LOG"

# Skip if already done
if [ -f "${OUTDIR}/test_metrics.txt" ] && grep -q "Final Evaluation" "${OUTDIR}/test_metrics.txt" 2>/dev/null; then
    echo "Already done at ${OUTDIR}/test_metrics.txt — exiting." | tee -a "$WAIT_LOG"
    exit 0
fi

# Poll until no python train.py processes are running.
# (Don't gate on nvidia-smi alone — bbsplat finishing might briefly show 0 utilization
# even mid-run.)
echo "Waiting for any python train.py to finish before launching..." | tee -a "$WAIT_LOG"
while pgrep -f "python train\.py" > /dev/null 2>&1; do
    CUR=$(pgrep -af "python train\.py" | head -1)
    echo "  $(date): still running: $CUR" | tee -a "$WAIT_LOG"
    sleep 60
done
echo "GPU appears free at $(date). Launching cat hl=2." | tee -a "$WAIT_LOG"

RUN_LOG="$LOG_DIR/${NAME}2_${TS}.log"
echo "Per-run log: $RUN_LOG" | tee -a "$WAIT_LOG"

conda run -n nest_splatting --no-capture-output python train.py \
    -s /home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU/scan24 \
    -m "$NAME" \
    --yaml ./configs/dtu.yaml \
    --eval \
    --iterations 30000 \
    -r 2 \
    --lambda_mask 0.0 --lambda_normal 0.0 --lambda_dist 0.0 \
    --random_background \
    --method cat --hybrid_levels 2 \
    > "$RUN_LOG" 2>&1
RC=$?

echo "==== Run finished rc=$RC at $(date) ====" | tee -a "$WAIT_LOG"
echo "Output: $OUTDIR" | tee -a "$WAIT_LOG"
