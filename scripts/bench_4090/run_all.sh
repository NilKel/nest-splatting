#!/usr/bin/env bash
# Run bench_minimal on every bundle under bundles/, write a markdown table.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUNDLES_DIR="${1:-$ROOT/bundles}"
OUT_DIR="$ROOT/bench_results"
mkdir -p "$OUT_DIR"

. "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate bench

# Reproducibility: pin clocks to base if user is root; otherwise just note GPU.
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_BENCHMARK="${NUM_BENCHMARK:-200}"

results=()
for bundle in "$BUNDLES_DIR"/*/; do
  name="$(basename "$bundle")"
  json="$OUT_DIR/${name}.json"
  echo ""
  echo "=== $name ==="
  python "$HERE/bench_minimal.py" \
      --bundle "$bundle" \
      --num_warmup "$NUM_WARMUP" --num_benchmark "$NUM_BENCHMARK" \
      --out "$json" 2>&1 | tail -12 || { echo "  FAIL on $name"; continue; }
  results+=("$json")
done

# Tabulate
PY="$(command -v python)"
"$PY" - <<PY
import json, glob, os
rows = []
for p in sorted(glob.glob("$OUT_DIR/*.json")):
    with open(p) as f:
        d = json.load(f)
    name = os.path.splitext(os.path.basename(p))[0]
    rows.append((name, d))

print()
print("# Benchmark results (RTX 4090)")
print()
print("| Scene | N (gauss) | Resolution | PSNR | SSIM | LPIPS | FPS | ms/frame |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
psnrs, ssims, lpips, fpss = [], [], [], []
for name, d in rows:
    print(f"| {name} | {d['n_gaussians']:,} | {d['resolution']} | "
          f"{d['psnr']:.2f} | {d['ssim']:.4f} | {d['lpips']:.4f} | "
          f"**{d['fps']:.1f}** | {d['ms_per_frame']:.2f} |")
    psnrs.append(d['psnr']); ssims.append(d['ssim'])
    lpips.append(d['lpips']); fpss.append(d['fps'])
if rows:
    n = len(rows)
    print(f"| **mean** |  |  | **{sum(psnrs)/n:.2f}** | "
          f"**{sum(ssims)/n:.4f}** | **{sum(lpips)/n:.4f}** | "
          f"**{sum(fpss)/n:.1f}** |  |")
PY
