#!/bin/bash
# Test script for adaptive_cat mode

set -e  # Exit on error

SCENE_PATH="/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/ficus"
CONFIG="./configs/nerfsyn.yaml"
ITERATIONS=30000

echo "========================================="
echo "Testing adaptive_cat mode implementation"
echo "========================================="
echo ""

# Test 1: Basic adaptive_cat training
echo "Test 1: Basic adaptive_cat training (smooth blending)"
python train.py \
  -s "$SCENE_PATH" \
  -m test_adaptive_cat_basic \
  --yaml "$CONFIG" \
  --method adaptive_cat \
  --iterations "$ITERATIONS" \
  --eval \
  --lambda_adaptive_cat 0.01 \
  --adaptive_cat_anneal_start 15000

echo ""
echo "Test 1 complete! Check outputs/nerf_synthetic/ficus/adaptive_cat/test_adaptive_cat_basic/"
echo ""

# Test 2: adaptive_cat with binary inference
echo "Test 2: adaptive_cat with binary inference (speedup mode)"
python train.py \
  -s "$SCENE_PATH" \
  -m test_adaptive_cat_inference \
  --yaml "$CONFIG" \
  --method adaptive_cat \
  --iterations "$ITERATIONS" \
  --eval \
  --lambda_adaptive_cat 0.01 \
  --adaptive_cat_anneal_start 15000 \
  --adaptive_cat_inference

echo ""
echo "Test 2 complete! Check training_log.txt for FPS comparison"
echo ""

# Test 3: adaptive_cat with MCMC
echo "Test 3: adaptive_cat with MCMC"
python train.py \
  -s "$SCENE_PATH" \
  -m test_adaptive_cat_mcmc \
  --yaml "$CONFIG" \
  --method adaptive_cat \
  --iterations "$ITERATIONS" \
  --eval \
  --mcmc --cap_max 100000 \
  --opacity_reg 0.001 --scale_reg 0.001 --noise_lr 1e4 \
  --lambda_adaptive_cat 0.01 \
  --adaptive_cat_anneal_start 15000

echo ""
echo "Test 3 complete! Check point count in training_log.txt"
echo ""

# Test 4: Compare to cat mode baseline
echo "Test 4: cat mode baseline (for comparison)"
python train.py \
  -s "$SCENE_PATH" \
  -m test_cat5_baseline \
  --yaml "$CONFIG" \
  --method cat --hybrid_levels 5 \
  --iterations "$ITERATIONS" \
  --eval \
  --disable_c2f

echo ""
echo "Test 4 complete!"
echo ""

echo "========================================="
echo "All tests complete!"
echo "========================================="
echo ""
echo "Compare metrics in test_metrics.txt:"
echo "  - adaptive_cat basic vs cat5 (quality should match)"
echo "  - adaptive_cat inference vs basic (FPS should improve)"
echo "  - adaptive_cat MCMC (point count should be lower)"
echo ""
echo "Check tensorboard for weight evolution:"
echo "  adaptive_cat/mean_weight - should approach 0 or 1"
echo "  adaptive_cat/pct_gaussian - % of Gaussians using per-Gaussian features"
echo ""
