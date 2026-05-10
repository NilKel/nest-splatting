#!/bin/bash
# Sequential queue: alllosses5 → SV_30thr...
# Run inside a detached tmux session so it survives terminal/SSH close.

set -u
cd /home/nilkel/Projects/nest-splatting

LOG_DIR="/home/nilkel/Projects/nest-splatting/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
QUEUE_LOG="$LOG_DIR/queue_${TS}.log"
ln -sfn "$QUEUE_LOG" "$LOG_DIR/queue_latest.log"

echo "==== QUEUE START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Log: $QUEUE_LOG" | tee -a "$QUEUE_LOG"

# ---- Run 1: alllosses5 (cat, hybrid=5, default DTU regs) ----
RUN1_LOG="$LOG_DIR/alllosses5_${TS}.log"
echo "" | tee -a "$QUEUE_LOG"
echo "==== RUN 1: alllosses5 START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Per-run log: $RUN1_LOG" | tee -a "$QUEUE_LOG"

conda run -n nest_splatting --no-capture-output python train.py \
    -s /home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU/scan24 \
    -m alllosses5 \
    --yaml ./configs/dtu.yaml \
    --eval \
    --iterations 30000 \
    -r 2 \
    --method cat --hybrid_levels 5 \
    > "$RUN1_LOG" 2>&1
RC1=$?
echo "==== RUN 1: alllosses5 finished rc=$RC1 at $(date) ====" | tee -a "$QUEUE_LOG"

# ---- Run 2: train_3D_SH_res.sh tnt SV_30thr... ----
RUN2_LOG="$LOG_DIR/SV_30thr_0w0gLP4lev_FRP5k10_c2f_Jac_${TS}.log"
echo "" | tee -a "$QUEUE_LOG"
echo "==== RUN 2: SV_30thr_... START $(date) ====" | tee -a "$QUEUE_LOG"
echo "Per-run log: $RUN2_LOG" | tee -a "$QUEUE_LOG"

# train_3D_SH_res.sh needs the env active for its inner python calls
conda run -n nest_splatting --no-capture-output bash ./train_3D_SH_res.sh tnt SV_30thr_0w0gLP4lev_FRP5k10_c2f_Jac all 35000 \
    "--hybrid_levels 2 --disable_c2f --aabb rect --kernel beta_scaled --cold --fastgs --fastgs_densify_interval 500 --fastgs_densify_until 20000 --fastgs_grad_thresh 0.00015 --fastgs_grad_abs_thresh 0.0006 --fastgs_dense 0.01 --fastgs_importance_thresh 30 --fastgs_loss_thresh 0.1 --grads abs --feature SV --w_lambda 0.0 --w_lambda_gamma 0 --lowpass --freeze_hash_iter 5000 --freeze_hash_period 10" \
    > "$RUN2_LOG" 2>&1
RC2=$?
echo "==== RUN 2: SV_30thr_... finished rc=$RC2 at $(date) ====" | tee -a "$QUEUE_LOG"

echo "" | tee -a "$QUEUE_LOG"
echo "==== QUEUE END $(date) ====" | tee -a "$QUEUE_LOG"
echo "Run 1 rc=$RC1   Run 2 rc=$RC2" | tee -a "$QUEUE_LOG"
