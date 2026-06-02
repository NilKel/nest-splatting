#!/usr/bin/env bash
# video_to_dataset.sh
#
# Turn any video into a nest-splatting / 3DGS training dataset.
#
# Pipeline:
#   1. ffmpeg     : extract frames at FPS into images_raw/
#   2. COLMAP     : feature_extractor → sequential_matcher → mapper → undistort
#   3. restructure: promote undistorted/{images, sparse/0} to <out>/{images, sparse/0}
#   4. mips       : write images_2/, images_4/ via PIL LANCZOS
#   5. export     : sparse/0/points3D.ply (diagnostic)
#
# Output layout (matches nest-splatting --source convention):
#   <out>/Max.mp4              (original, preserved if copied in)
#   <out>/images/              (undistorted, full-res)
#   <out>/images_2/            (half)
#   <out>/images_4/            (quarter)
#   <out>/sparse/0/{cameras,images,points3D}.bin
#   <out>/sparse/0/points3D.ply
#   <out>/sparse_distorted/    (pre-undistort recon, kept for debug)
#
# Headless gotchas (built-in here):
#   * uses conda env's ffmpeg (system ffmpeg often missing)
#   * exports QT_QPA_PLATFORM=offscreen so COLMAP's Qt init doesn't try to open an X display
#   * passes --disable_gpu so COLMAP uses CPU SIFT (GPU SIFT needs an OpenGL
#     context — fails on headless boxes without xvfb). Set USE_GPU=1 to override
#     (requires X / xvfb / EGL-built COLMAP).
#
# Usage:
#   scripts/video_to_dataset.sh <video.mp4> <out_dir> [fps]
#
# Defaults: fps=10. Re-runs are idempotent (skips any stage whose output exists).
set -u

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input_video> <out_dir> [fps]"
  echo "  fps default 10"
  exit 1
fi

VIDEO=$1
OUT=$2
FPS=${3:-10}
COLMAP_REPO=${COLMAP_REPO:-/home/nilkel/Projects/video-3d-reconstruction-gsplat}
CONDA_ENV=${CONDA_ENV:-nest_splatting}

LOG=$OUT/pipeline.log
mkdir -p "$OUT"
echo "" > "$LOG"
ts(){ date +"%H:%M:%S"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

if [ ! -f "$VIDEO" ]; then log "FATAL: missing $VIDEO"; exit 1; fi
if [ ! -d "$COLMAP_REPO" ]; then log "FATAL: missing colmap repo $COLMAP_REPO"; exit 1; fi

# Headless COLMAP: no X server available over SSH → use the offscreen Qt platform.
export QT_QPA_PLATFORM=offscreen

# Use conda env's ffmpeg (system ffmpeg often missing on training boxes).
FFMPEG=$(conda run -n "$CONDA_ENV" which ffmpeg | tr -d '\r')
if [ -z "$FFMPEG" ]; then log "FATAL: ffmpeg not in conda env $CONDA_ENV"; exit 1; fi

log "video=$VIDEO  out=$OUT  fps=$FPS  ffmpeg=$FFMPEG"

# ---- Stage 1: extract frames ----
log "=== Stage 1: extract frames @ ${FPS} fps ==="
mkdir -p "$OUT/images_raw"
if [ -z "$(ls -A "$OUT/images_raw" 2>/dev/null)" ]; then
  "$FFMPEG" -hide_banner -loglevel warning -i "$VIDEO" -vf "fps=$FPS" "$OUT/images_raw/frame_%04d.png" 2>&1 | tee -a "$LOG"
else
  log "images_raw/ already populated — skipping extraction"
fi
NUM=$(ls "$OUT/images_raw" 2>/dev/null | wc -l)
log "extracted $NUM frames"
if [ "$NUM" -eq 0 ]; then log "FATAL: no frames extracted"; exit 1; fi

# ---- Stage 2: COLMAP SfM + undistort ----
USE_GPU=${USE_GPU:-0}
GPU_FLAG=$([ "$USE_GPU" = "1" ] && echo "--enable_gpu" || echo "--disable_gpu")
log "=== Stage 2: COLMAP SfM + undistort ($GPU_FLAG) ==="
if [ ! -d "$OUT/undistorted/sparse/0" ] && [ ! -d "$OUT/sparse/0" ]; then
  ( cd "$COLMAP_REPO" && bash ./colmap_undistorted_sfm_export.sh "$OUT/images_raw" "$OUT" $GPU_FLAG ) 2>&1 | tee -a "$LOG"
else
  log "sparse already exists — skipping COLMAP"
fi

# After the wrapper: $OUT/undistorted/{images, sparse/0/...}
if [ ! -d "$OUT/undistorted/sparse/0" ] && [ ! -d "$OUT/sparse/0" ]; then
  log "FATAL: COLMAP did not produce sparse/0 — see log"; exit 1
fi

# ---- Stage 3: restructure to nest-splatting --source layout ----
log "=== Stage 3: restructure to nest-splatting layout ==="
# Keep the pre-undistortion sparse model as sparse_distorted/ for debug.
if [ -d "$OUT/sparse" ] && [ ! -d "$OUT/sparse_distorted" ] && [ -d "$OUT/undistorted/sparse" ]; then
  mv "$OUT/sparse" "$OUT/sparse_distorted"
fi
# Promote undistorted/{images, sparse} to top level.
if [ -d "$OUT/undistorted/images" ] && [ ! -d "$OUT/images" ]; then
  mv "$OUT/undistorted/images" "$OUT/images"
fi
if [ -d "$OUT/undistorted/sparse" ] && [ ! -d "$OUT/sparse" ]; then
  mv "$OUT/undistorted/sparse" "$OUT/sparse"
fi
# Cleanup transient.
rm -rf "$OUT/undistorted" "$OUT/images_raw" "$OUT/database.db"

# ---- Stage 4: downsampled mips (images_2, images_4) ----
# NOTE: don't use `conda run python - <<EOF` — it silently swallows stdin in
# this combo (heredoc + script-from-stdin + extra argv). Resolve the conda
# env's python binary directly and invoke it.
log "=== Stage 4: image mips (images_2, images_4) ==="
PYBIN=$(conda run -n "$CONDA_ENV" which python | tr -d '\r')
"$PYBIN" - "$OUT" 2>&1 <<'PY' | tee -a "$LOG"
import os, glob, sys
from PIL import Image
OUT = sys.argv[1]
src = sorted(glob.glob(f'{OUT}/images/*'))
for k in (2, 4):
    out = f'{OUT}/images_{k}'
    os.makedirs(out, exist_ok=True)
    if len(os.listdir(out)) == len(src):
        print(f'images_{k} already populated ({len(src)} files) — skipping')
        continue
    for p in src:
        im = Image.open(p)
        w, h = im.size
        im.resize((w // k, h // k), Image.LANCZOS).save(os.path.join(out, os.path.basename(p)))
    print(f'wrote {len(src)} files to images_{k}/')
PY

# ---- Stage 5: export points3D.ply (diagnostic) ----
log "=== Stage 5: export points3D.ply ==="
if [ ! -f "$OUT/sparse/0/points3D.ply" ]; then
  colmap model_converter --input_path "$OUT/sparse/0" --output_path "$OUT/sparse/0/points3D.ply" --output_type PLY 2>&1 | tee -a "$LOG"
fi

# ---- Stage 6: summary ----
log "=== final layout ==="
ls -d "$OUT"/* 2>/dev/null | tee -a "$LOG"
log "images: $(ls "$OUT/images" 2>/dev/null | wc -l)"
NPTS=$(grep '^element vertex' "$OUT/sparse/0/points3D.ply" 2>/dev/null | awk '{print $3}')
log "sparse points: ${NPTS:-?}"
log "DONE — train with:  python train.py -s $OUT -m <outdir> --yaml configs/<scene>.yaml"
