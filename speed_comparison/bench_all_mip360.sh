#!/usr/bin/env bash
# Benchmark baseline lean vs. production on all mip360 scenes with the
# named RD_SV_30thr_005w25gLP_N2f_frz5k10 checkpoint.
set -uo pipefail
NEST=/home/nilkel/Projects/nest-splatting
BENCH="$NEST/speed_comparison/bench_lean_vs_prod.py"
OUT_DIR="$NEST/speed_comparison/mip360_lean_baseline"
RUN=RD_SV_30thr_005w25gLP_N2f_frz5k10

mkdir -p "$OUT_DIR"

scenes=(bicycle bonsai counter flowers garden kitchen room stump treehill)

for scene in "${scenes[@]}"; do
    mp="$NEST/outputs/mip_360/$scene/3D_SH_res/$RUN"
    bake="$mp/baked_atlas"
    if [ ! -f "$bake/baked.ply" ]; then
        echo "SKIP $scene: no bake"
        continue
    fi
    echo ""
    echo "==================================================================="
    echo "  $scene"
    echo "==================================================================="
    conda run -n nest_splatting --no-capture-output python -u "$BENCH" \
        --model_path "$mp" --bake_dir "$bake" \
        --num_warmup 30 --num_benchmark 300 \
        --out "$OUT_DIR/$scene.json" 2>&1 | \
        grep -E "PROD|LEAN|PSNR=|Gauss|SH-only|SH\+atlas"
done

echo ""
echo "==================================================================="
echo "  DONE. Results dir: $OUT_DIR"
echo "==================================================================="
