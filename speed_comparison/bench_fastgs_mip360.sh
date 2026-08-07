#!/usr/bin/env bash
# Bench FastGS on all 9 mip360 scenes on this GPU (RTX 5090), matching
# our lean-fork sweep so numbers are directly comparable.
set -uo pipefail
FASTGS=/home/nilkel/Projects/FastGS
OUT_DIR=/home/nilkel/Projects/nest-splatting/speed_comparison/mip360_fastgs
DATA_ROOT=/home/nilkel/Projects/data/mip_360   # source path for cameras/images

mkdir -p "$OUT_DIR"

scenes=(bicycle bonsai counter flowers garden kitchen room stump treehill)

cd "$FASTGS"

for scene in "${scenes[@]}"; do
    model_path="$FASTGS/output/$scene"
    if [ ! -d "$model_path" ]; then
        echo "SKIP $scene: no FastGS checkpoint at $model_path"
        continue
    fi
    src="$DATA_ROOT/$scene"
    if [ ! -d "$src" ]; then
        echo "SKIP $scene: no source data at $src"
        continue
    fi
    echo ""
    echo "==================================================================="
    echo "  $scene"
    echo "==================================================================="
    conda run -n fastgs --no-capture-output python -u render_eval.py \
        -m "$model_path" -s "$src" \
        --num_warmup 30 --num_benchmark 300 \
        --skip_metrics 2>&1 | \
        grep -Ei "fps|Loaded|Number of points|test|elapsed|ms/frame"
    # Copy the eval_summary.json so we have per-scene numbers
    latest=$(ls -td "$model_path"/test/ours_* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        cp "$latest/eval_summary.json" "$OUT_DIR/$scene.json" 2>/dev/null || true
    fi
done

echo ""
echo "==================================================================="
echo "  DONE. Results dir: $OUT_DIR"
echo "==================================================================="
