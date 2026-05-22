#!/usr/bin/env bash
# Batch-bake a named experiment across every scene that has it.
#
# Usage:
#   scripts/bake_all_scenes.sh <run_name> [options]
#
# Options:
#   --dataset <name>        Top-level dataset folder under outputs/  (default: mip_360)
#   --method <name>         Filter to a single method dir            (default: any)
#   --scenes "s1 s2 ..."    Override scene list                      (default: auto-discover)
#   --max-res <N>           --max_res                                (default: 64)
#   --atlas-budget-mb <N>   --atlas_budget_mb                        (default: 8192)
#   --bake-dtype <X>        --bake_dtype                             (default: bc7)
#   --aabb-mode <N>         --aabb_mode                              (default: 5)
#   --sort-mode <N>         --sort_mode                              (default: 0)
#   --num-warmup <N>        bench warmup iters                       (default: 5)
#   --num-benchmark <N>     bench measured iters                     (default: 50)
#   --output-subdir <name>  Bake output dir under each model_path    (default: baked_atlas)
#   --log <path>            Log file                                 (default: /tmp/<run>_bake_all.log)
#   --dry-run               Print the plan, don't bake
#   --help
#
# Discovers scenes via:    outputs/<dataset>/<scene>/{<method>|*}/<run_name>
# Each scene's bake output: <model_path>/<output-subdir>/
# Aggregates per-scene benchmark_results.json → <log>.summary.md
set -u

NEST=/home/nilkel/Projects/nest-splatting
PY="conda run -n nest_splatting python"

# --- defaults ---
RUN=""
DATASET="mip_360"
METHOD=""
SCENES_OVERRIDE=""
MAX_RES=64
ATLAS_BUDGET_MB=8192
BAKE_DTYPE=bc7
AABB_MODE=5
SORT_MODE=0
NUM_WARMUP=5
NUM_BENCHMARK=50
OUTPUT_SUBDIR=baked_atlas
LOG=""
DRY_RUN=0

usage() { sed -n '2,/^set -u/{/^set -u/q; p;}' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --dataset)        DATASET="$2"; shift 2 ;;
    --method)         METHOD="$2"; shift 2 ;;
    --scenes)         SCENES_OVERRIDE="$2"; shift 2 ;;
    --max-res)        MAX_RES="$2"; shift 2 ;;
    --atlas-budget-mb) ATLAS_BUDGET_MB="$2"; shift 2 ;;
    --bake-dtype)     BAKE_DTYPE="$2"; shift 2 ;;
    --aabb-mode)      AABB_MODE="$2"; shift 2 ;;
    --sort-mode)      SORT_MODE="$2"; shift 2 ;;
    --num-warmup)     NUM_WARMUP="$2"; shift 2 ;;
    --num-benchmark)  NUM_BENCHMARK="$2"; shift 2 ;;
    --output-subdir)  OUTPUT_SUBDIR="$2"; shift 2 ;;
    --log)            LOG="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; usage 1 ;;
    *)  if [ -z "$RUN" ]; then RUN="$1"; shift; else echo "extra positional: $1" >&2; usage 1; fi ;;
  esac
done

[ -z "$RUN" ] && { echo "error: run name required" >&2; usage 1; }
[ -z "$LOG" ] && LOG="/tmp/${RUN}_bake_all.log"
DONE="${LOG%.log}.done"; SUMMARY="${LOG%.log}.summary.md"
: > "$LOG"; rm -f "$DONE" "$SUMMARY"
ts(){ date +"%H:%M:%S"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

# --- discover candidate run paths ---
DATASET_ROOT="$NEST/outputs/$DATASET"
[ -d "$DATASET_ROOT" ] || { log "FATAL: dataset root $DATASET_ROOT not found"; exit 1; }

# Pattern: outputs/<dataset>/<scene>/<method>/<run>
if [ -n "$METHOD" ]; then
  GLOB="$DATASET_ROOT/*/$METHOD/$RUN"
else
  GLOB="$DATASET_ROOT/*/*/$RUN"
fi
mapfile -t CANDIDATES < <(compgen -G "$GLOB" 2>/dev/null | sort)

if [ -n "$SCENES_OVERRIDE" ]; then
  # Filter discovered candidates to scenes the user asked for
  declare -A KEEP=()
  for s in $SCENES_OVERRIDE; do KEEP[$s]=1; done
  FILTERED=()
  for p in "${CANDIDATES[@]}"; do
    scene=$(basename "$(dirname "$(dirname "$p")")")
    [ -n "${KEEP[$scene]:-}" ] && FILTERED+=("$p")
  done
  CANDIDATES=("${FILTERED[@]}")
fi

N=${#CANDIDATES[@]}
if [ "$N" -eq 0 ]; then
  log "FATAL: no runs found matching $RUN under $DATASET_ROOT" \
      "$([ -n "$METHOD" ] && echo "(method=$METHOD)")"
  exit 1
fi

log "=== batch bake: $N runs (dataset=$DATASET, run=$RUN, method=${METHOD:-any}) ==="
log "  settings: max_res=$MAX_RES atlas_budget_mb=$ATLAS_BUDGET_MB dtype=$BAKE_DTYPE aabb=$AABB_MODE sort=$SORT_MODE"
for p in "${CANDIDATES[@]}"; do
  scene=$(basename "$(dirname "$(dirname "$p")")")
  method=$(basename "$(dirname "$p")")
  log "  - $scene/$method"
done
[ "$DRY_RUN" -eq 1 ] && { log "(dry run — exiting)"; exit 0; }

# --- bake loop ---
cd "$NEST"
i=0
declare -a SCENES_DONE=() METHODS_DONE=() PATHS_DONE=()
for MP in "${CANDIDATES[@]}"; do
  i=$((i+1))
  scene=$(basename "$(dirname "$(dirname "$MP")")")
  method=$(basename "$(dirname "$MP")")
  BAKE="$MP/$OUTPUT_SUBDIR"
  log "--- [$i/$N] $scene/$method → $BAKE ---"

  if ! compgen -G "$MP/point_cloud/iteration_*" >/dev/null; then
    log "  SKIP $scene/$method: no point_cloud/iteration_* checkpoint"
    continue
  fi
  if [ ! -f "$MP/cameras.json" ]; then
    log "  SKIP $scene/$method: missing cameras.json"
    continue
  fi

  $PY scripts/benchmark_baked.py --model_path "$MP" --output_dir "$BAKE" \
    --max_res "$MAX_RES" --atlas_budget_mb "$ATLAS_BUDGET_MB" \
    --aabb_mode "$AABB_MODE" --sort_mode "$SORT_MODE" \
    --bake_dtype "$BAKE_DTYPE" \
    --num_warmup "$NUM_WARMUP" --num_benchmark "$NUM_BENCHMARK" 2>&1 | tee -a "$LOG"

  if [ -f "$BAKE/baked.ply" ] && [ -f "$BAKE/atlas_texture.pt" ]; then
    SZ=""
    [ -f "$BAKE/atlas_texture.${BAKE_DTYPE}" ] && SZ=" ($(du -h "$BAKE/atlas_texture.${BAKE_DTYPE}" | cut -f1) ${BAKE_DTYPE} atlas)"
    log "  OK $scene/$method$SZ"
    SCENES_DONE+=("$scene"); METHODS_DONE+=("$method"); PATHS_DONE+=("$BAKE")
  else
    log "  WARN $scene/$method: bake did not produce baked.ply / atlas_texture.pt"
  fi
done

# --- aggregate benchmark_results.json → markdown summary ---
if [ "${#PATHS_DONE[@]}" -gt 0 ]; then
  log "=== aggregating ${#PATHS_DONE[@]} benchmark_results.json → $SUMMARY ==="
  RUN_TAG="$RUN" SUMMARY_PATH="$SUMMARY" \
  PATHS="$(printf '%s\n' "${PATHS_DONE[@]}")" \
  SCENES="$(printf '%s\n' "${SCENES_DONE[@]}")" \
  METHODS="$(printf '%s\n' "${METHODS_DONE[@]}")" \
  $PY - <<'PY' 2>&1 | tee -a "$LOG"
import os, json
run    = os.environ["RUN_TAG"]
paths  = os.environ["PATHS"].strip().splitlines()
scenes = os.environ["SCENES"].strip().splitlines()
methods= os.environ["METHODS"].strip().splitlines()

def fmt(x, p=2):
    try: return f"{float(x):.{p}f}"
    except Exception: return "—"

out = [
    f"# {run} — batch bake summary",
    "",
    "| scene | method | neural PSNR | baked PSNR | Δ dB | baked SSIM | baked LPIPS | baked FPS | neural FPS | speedup |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for scene, method, p in zip(scenes, methods, paths):
    j = os.path.join(p, "benchmark_results.json")
    if not os.path.exists(j):
        out.append(f"| {scene} | {method} | — | — | — | — | — | — | — | (no result) |")
        continue
    try:
        r = json.load(open(j))
    except Exception as e:
        out.append(f"| {scene} | {method} | — | — | — | — | — | — | — | (parse err: {e}) |")
        continue
    n = r.get("neural", {})
    b = r.get("baked_sh_atlas") or r.get("baked_atlas") or r.get("baked", {}) or {}
    # benchmark_baked.py writes neural keys as neural_psnr / neural_ssim /
    # neural_lpips / train_fps; baked side uses bare psnr/ssim/lpips/fps.
    nps = n.get("neural_psnr",  n.get("psnr"))
    bps = b.get("psnr")
    nfp = n.get("train_fps",    n.get("fps"))
    bfp = b.get("fps")
    nssim  = n.get("neural_ssim",  n.get("ssim"))
    nlpips = n.get("neural_lpips", n.get("lpips"))
    delta   = (bps - nps) if (nps is not None and bps is not None) else None
    speedup = (bfp / nfp) if (nfp and bfp) else None
    out.append(
        f"| {scene} | {method} | {fmt(nps)} | {fmt(bps)} | "
        f"{'—' if delta is None else f'{delta:+.2f}'} | "
        f"{fmt(b.get('ssim'),4)} | {fmt(b.get('lpips'),4)} | "
        f"{fmt(bfp,1)} | {fmt(nfp,1)} | "
        f"{'—' if speedup is None else f'{speedup:.2f}x'} |"
    )

text = "\n".join(out) + "\n"
open(os.environ["SUMMARY_PATH"], "w").write(text)
print(text)
PY
fi

touch "$DONE"
log "ALL DONE (baked ${#PATHS_DONE[@]}/$N runs; summary: $SUMMARY; log: $LOG)"
