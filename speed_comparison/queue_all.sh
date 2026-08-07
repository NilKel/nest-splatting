#!/usr/bin/env bash
# Full pipeline: wait for current Option A sweep to finish, then rebuild with
# exact correction, then re-sweep, then aggregate everything.
# Runs detached (nohup) so it survives Claude Code session pause.
set -uo pipefail
NEST=/home/nilkel/Projects/nest-splatting
LEAN=$NEST/submodules/diff_surfel_bake_render_lean
LOG=$NEST/speed_comparison/queue_all.log
DONE=$NEST/speed_comparison/queue_all.done

echo "=== [$(date +%H:%M:%S)] Start queue_all" > "$LOG"

# ----- Step 1: wait for Option A sweep to finish (marker: "DONE." in mip360_optA.log) -----
echo "[$(date +%H:%M:%S)] Waiting for Option A sweep to finish..." >> "$LOG"
while ! grep -q "DONE. Results dir" /tmp/mip360_optA.log 2>/dev/null; do
    sleep 15
done
echo "[$(date +%H:%M:%S)] Option A sweep finished." >> "$LOG"

# ----- Step 2: rebuild lean fork with exact rational correction -----
echo "[$(date +%H:%M:%S)] Rebuilding lean fork with correction..." >> "$LOG"
cd "$LEAN"
rm -rf build
touch cuda_rasterizer/forward.cu
LEAN_FLAGS="CONIC" conda run -n nest_splatting --no-capture-output python setup.py build_ext --inplace >> "$LOG" 2>&1
build_exit=$?
echo "[$(date +%H:%M:%S)] Build exit=$build_exit" >> "$LOG"
if [ "$build_exit" -ne 0 ]; then
    echo "BUILD FAILED — aborting" >> "$LOG"
    touch "$DONE"
    exit 1
fi

# ----- Step 3: quick sanity check on garden -----
echo "[$(date +%H:%M:%S)] Sanity-check garden..." >> "$LOG"
conda run -n nest_splatting python /home/nilkel/Projects/nest-splatting/speed_comparison/diff_lean_prod_images.py 2>&1 | \
    grep -E "Prod|Lean|Diff" >> "$LOG"

# ----- Step 4: run corrected sweep -----
echo "[$(date +%H:%M:%S)] Running corrected sweep on 9 scenes..." >> "$LOG"
OUT_DIR="$NEST/speed_comparison/mip360_lean_optA_corrected"
mkdir -p "$OUT_DIR"
scenes=(bicycle bonsai counter flowers garden kitchen room stump treehill)
for scene in "${scenes[@]}"; do
    mp="$NEST/outputs/mip_360/$scene/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10"
    bake="$mp/baked_atlas"
    [ ! -f "$bake/baked.ply" ] && { echo "SKIP $scene" >> "$LOG"; continue; }
    echo "" >> "$LOG"
    echo "=== [$(date +%H:%M:%S)] $scene ===" >> "$LOG"
    conda run -n nest_splatting --no-capture-output python -u \
        "$NEST/speed_comparison/bench_lean_vs_prod.py" \
        --model_path "$mp" --bake_dir "$bake" \
        --num_warmup 30 --num_benchmark 300 \
        --out "$OUT_DIR/$scene.json" 2>&1 | \
        grep -E "PROD|LEAN|PSNR=|Gauss|SH-only|SH\+atlas" >> "$LOG"
done

# ----- Step 5: aggregate everything -----
echo "" >> "$LOG"
echo "=== [$(date +%H:%M:%S)] Aggregating final results ===" >> "$LOG"
conda run -n nest_splatting python -c "
import json
from pathlib import Path
sp = Path('$NEST/speed_comparison')
scenes = ['bicycle','bonsai','counter','flowers','garden','kitchen','room','stump','treehill']
paths = {
    'base':  sp/'mip360_lean_baseline',
    'optA':  sp/'mip360_lean_optA',
    'corr':  sp/'mip360_lean_optA_corrected',
    'fgs':   sp/'mip360_fastgs',
}
def gfps(p, k):
    d = json.load(open(p))
    if k == 'atlas': return d['modes']['lean_sh_atlas']['fps']
    if k == 'fastgs': return d['fps']['fps']
    return None

print(f\"{'scene':<10} {'prod':>7} {'base':>7} {'optA':>7} {'corr':>7} {'FastGS':>7} | {'basePSNR':>9} {'optAPSNR':>9} {'corrPSNR':>9}\")
print('-'*100)
for s in scenes:
    b = json.load(open(paths['base'] / f'{s}.json'))
    prod = b['modes']['prod_sh_atlas']['fps']; base_fps = b['modes']['lean_sh_atlas']['fps']
    prod_psnr = b['modes']['prod_sh_atlas']['psnr']; base_psnr = b['modes']['lean_sh_atlas']['psnr']
    a = json.load(open(paths['optA'] / f'{s}.json'))
    optA_fps = a['modes']['lean_sh_atlas']['fps']; optA_psnr = a['modes']['lean_sh_atlas']['psnr']
    c = json.load(open(paths['corr'] / f'{s}.json'))
    corr_fps = c['modes']['lean_sh_atlas']['fps']; corr_psnr = c['modes']['lean_sh_atlas']['psnr']
    fgs = json.load(open(paths['fgs'] / f'{s}.json'))['fps']['fps']
    print(f'{s:<10} {prod:>7.1f} {base_fps:>7.1f} {optA_fps:>7.1f} {corr_fps:>7.1f} {fgs:>7.1f} | {base_psnr:>9.3f} {optA_psnr:>9.3f} {corr_psnr:>9.3f}')
" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== [$(date +%H:%M:%S)] queue_all DONE ===" >> "$LOG"
touch "$DONE"
