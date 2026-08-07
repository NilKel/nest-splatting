#!/usr/bin/env bash
# Re-bench all 9 FastGS mip-360 scenes at the CORRECT mip-360 resolutions.
#
# Why: FastGS's checkpoints were trained with `images` + `resolution=-1`, which
# in the 3DGS/FastGS convention caps the long edge at 1600 px. That is NOT the
# mip-360 evaluation convention NeST uses (images_2 indoor / images_4 outdoor).
# For outdoor scenes the mismatch is large (1600x1050 vs 1267x832 = +60% pixels).
# Only `garden` was trained with images_4 and is therefore already comparable.
#
# This script renders each FastGS checkpoint at the NeST-matching resolution so
# the FPS numbers are apples-to-apples. It does NOT retrain -- the Gaussian
# counts still reflect densification at the training resolution (see the doc).
#
# `mult` is left at FastGS's own default (0.5, arguments/__init__.py:100, used by
# their train.py) -- these are FastGS's args, not ours.
#
# Timing methodology matches NeST's CONIC bench: per-frame cuda.Event pairs,
# one sync after the loop, FPS = 1000 / mean_ms. 50 warmup / 400 timed.
set -uo pipefail

FASTGS=/home/nilkel/Projects/FastGS
OUT=/home/nilkel/Projects/nest-splatting/speed_comparison/fastgs_correct_res
mkdir -p "$OUT"

# mip-360 convention: indoor -> images_2, outdoor -> images_4
declare -A TIER=(
  [bicycle]=images_4  [flowers]=images_4  [garden]=images_4
  [stump]=images_4    [treehill]=images_4
  [bonsai]=images_2   [counter]=images_2  [kitchen]=images_2  [room]=images_2
)

cd "$FASTGS"
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
  imgs=${TIER[$scene]}
  echo ""
  echo "=================================================================="
  echo "  $scene   ($imgs)"
  echo "=================================================================="
  conda run -n fastgs --no-capture-output python bench_fps.py \
      -m "output/$scene" -i "$imgs" \
      --num_warmup 50 --num_benchmark 400 \
      2>&1 | tee "$OUT/${scene}.log" | grep -E \
      'resolution|gaussians|mult|PSNR|cuda.Event|GPU FPS'
done

echo ""
echo "=== sweep complete -> $OUT ==="
