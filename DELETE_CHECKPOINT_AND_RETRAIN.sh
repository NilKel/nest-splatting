#!/bin/bash
# Script to delete corrupted checkpoint and retrain from scratch

CHECKPOINT_PATH="/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums/gaussian_init.pth"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           DELETING CORRUPTED CHECKPOINT                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint: $CHECKPOINT_PATH"
    echo "Creating backup..."
    mv "$CHECKPOINT_PATH" "${CHECKPOINT_PATH}.corrupted.backup"
    echo "✓ Checkpoint backed up to: ${CHECKPOINT_PATH}.corrupted.backup"
else
    echo "No checkpoint found at: $CHECKPOINT_PATH"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           TRAINING FROM SCRATCH                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Train 2DGS for 10,000 iterations"
echo "  2. Save checkpoint at iteration 10,000"
echo "  3. Continue with INGP training for 2,000 iterations (10,001-12,000)"
echo ""

python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m drums_fresh_train \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --eval \
  --iterations 12000

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           TRAINING COMPLETE                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "New checkpoint saved to: $CHECKPOINT_PATH"
echo "Output: drums_fresh_train"

