#!/bin/bash
# A/B: baseline vs cat-hybrid-5 on DTU scan24 with --random_background.
# All reg lambdas zeroed; everything else matches the user's spec.
#   python train.py -s .../scan24 -m <name> --yaml ./configs/dtu.yaml --eval \
#       --iterations 30000 -r 2 --method <method> [--hybrid_levels 5] \
#       --lambda_mask 0.0 --lambda_normal 0.0 --lambda_dist 0.0 \
#       --random_background

set -u
cd /home/nilkel/Projects/nest-splatting

LOG_DIR="/home/nilkel/Projects/nest-splatting/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
QUEUE_LOG="$LOG_DIR/randombg_scan24_queue_${TS}.log"
ln -sfn "$QUEUE_LOG" "$LOG_DIR/randombg_scan24_queue_latest.log"

DATA="/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU/scan24"
COMMON=(--yaml ./configs/dtu.yaml --eval --iterations 30000 -r 2
        --lambda_mask 0.0 --lambda_normal 0.0 --lambda_dist 0.0
        --random_background)

echo "==== QUEUE START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Queue log: $QUEUE_LOG" | tee -a "$QUEUE_LOG"

# ---- Run 1: baseline ----
NAME1="synth_style_baseline_randombg"
RUN1_LOG="$LOG_DIR/${NAME1}_${TS}.log"
echo "" | tee -a "$QUEUE_LOG"
echo "==== RUN 1: $NAME1 (baseline) START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Per-run log: $RUN1_LOG" | tee -a "$QUEUE_LOG"
conda run -n nest_splatting --no-capture-output python train.py \
    -s "$DATA" -m "$NAME1" \
    "${COMMON[@]}" \
    --method baseline \
    > "$RUN1_LOG" 2>&1
RC1=$?
echo "==== RUN 1 finished rc=$RC1 at $(date) ====" | tee -a "$QUEUE_LOG"

# ---- Run 2: cat hybrid_levels=5 ----
NAME2="synth_style_cat_randombg"
RUN2_LOG="$LOG_DIR/${NAME2}5_${TS}.log"
echo "" | tee -a "$QUEUE_LOG"
echo "==== RUN 2: ${NAME2} (cat, hybrid_levels=5) START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Per-run log: $RUN2_LOG" | tee -a "$QUEUE_LOG"
conda run -n nest_splatting --no-capture-output python train.py \
    -s "$DATA" -m "$NAME2" \
    "${COMMON[@]}" \
    --method cat --hybrid_levels 5 \
    > "$RUN2_LOG" 2>&1
RC2=$?
echo "==== RUN 2 finished rc=$RC2 at $(date) ====" | tee -a "$QUEUE_LOG"

echo "" | tee -a "$QUEUE_LOG"
echo "==== QUEUE END $(date) ====" | tee -a "$QUEUE_LOG"
echo "Run 1 (baseline)         rc=$RC1   log=$RUN1_LOG" | tee -a "$QUEUE_LOG"
echo "Run 2 (cat hybrid_lv=5)  rc=$RC2   log=$RUN2_LOG" | tee -a "$QUEUE_LOG"
