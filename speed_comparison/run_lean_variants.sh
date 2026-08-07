#!/usr/bin/env bash
# Rebuild the lean fork with each combination of LEAN flags and run the
# lean-vs-prod bench.  Results append into a JSON dir per variant.
set -euo pipefail

NEST=/home/nilkel/Projects/nest-splatting
LEAN_DIR="$NEST/submodules/diff_surfel_bake_render_lean"
BENCH_SCRIPT="$NEST/speed_comparison/bench_lean_vs_prod.py"
RESULTS_DIR="$NEST/speed_comparison/lean_variants"
CKPT="$NEST/outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10"
BAKE="$CKPT/baked_atlas"

mkdir -p "$RESULTS_DIR"

variants=(
    ""              # baseline (already benched but re-run for cross-check)
    "T2"
    "CTG"
    "CONIC"
    "T2,CTG"
    "T2,CONIC"
    "CTG,CONIC"
    "T2,CTG,CONIC"
)

for v in "${variants[@]}"; do
    tag="${v//,/_}"
    [ -z "$tag" ] && tag="none"
    echo ""
    echo "============================================================"
    echo "  variant: LEAN_FLAGS='$v'   tag='$tag'"
    echo "============================================================"

    # Rebuild
    cd "$LEAN_DIR"
    LEAN_FLAGS="$v" conda run -n nest_splatting python -m pip install -e . --no-build-isolation \
        > "$RESULTS_DIR/build_${tag}.log" 2>&1 || {
        echo "BUILD FAILED — see $RESULTS_DIR/build_${tag}.log"
        continue
    }

    # Bench
    cd "$NEST"
    conda run -n nest_splatting python -u "$BENCH_SCRIPT" \
        --model_path "$CKPT" --bake_dir "$BAKE" \
        --num_warmup 50 --num_benchmark 400 \
        --out "$RESULTS_DIR/bench_${tag}.json" \
        > "$RESULTS_DIR/bench_${tag}.log" 2>&1 || {
        echo "BENCH FAILED — see $RESULTS_DIR/bench_${tag}.log"
        continue
    }

    # Extract summary
    grep -E "PROD  SH-only|PROD  SH\+atlas|LEAN  SH-only|LEAN  SH\+atlas|PSNR=" "$RESULTS_DIR/bench_${tag}.log" | head -8
done

echo ""
echo "============================================================"
echo "  ALL VARIANTS DONE"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"
