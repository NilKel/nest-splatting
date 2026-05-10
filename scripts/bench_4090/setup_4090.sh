#!/usr/bin/env bash
# One-shot setup for the 4090 box.
# Network policy on this campus blocks repo.anaconda.com and download.pytorch.org
# (Zscaler 403). Workarounds:
#   - Miniforge installer: GitHub release (whitelisted)
#   - Conda packages: prefix.dev/conda-forge mirror (whitelisted)
#   - Pure-Python pip pkgs: pypi.org / files.pythonhosted.org (also whitelisted)
#     plus --trusted-host because Zscaler MITMs the SSL with an untrusted root.
# Idempotent — re-run safely.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=bench

PIP_TRUSTED=(--trusted-host pypi.org
             --trusted-host pypi.python.org
             --trusted-host files.pythonhosted.org)

# 1) Miniforge (conda from GitHub, no anaconda.com)  ---------------------
if [ ! -d "$HOME/miniforge3" ]; then
  echo "[setup] installing miniforge → $HOME/miniforge3"
  TMP=$(mktemp -d)
  MF_URL="https://github.com/conda-forge/miniforge/releases/download/24.7.1-2/Miniforge3-24.7.1-2-Linux-x86_64.sh"
  wget --no-check-certificate -q -O "$TMP/mf.sh" "$MF_URL"
  bash "$TMP/mf.sh" -b -p "$HOME/miniforge3"
  rm -rf "$TMP"
fi
. "$HOME/miniforge3/etc/profile.d/conda.sh"

# Pin all conda calls to the prefix.dev mirror; ban the default anaconda.com.
CHAN=(-c https://prefix.dev/conda-forge --override-channels)

# 2) bench env (python + pytorch + numpy/plyfile/pillow)  ---------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] creating env '$ENV_NAME' from prefix.dev/conda-forge"
  # cuda-version=12.1 pins the pytorch build to the cu121 variant; matches
  # 4090 driver 570 well (driver supports CUDA 12.x universally).
  conda create -n "$ENV_NAME" "${CHAN[@]}" -y \
    "python=3.10" "cuda-version=12.1" \
    "pytorch=2.4" torchvision \
    "numpy=1.26" plyfile pillow pip
fi
conda activate "$ENV_NAME"

# 3) pip-only deps (lpips, pytorch_msssim are PyPI-only)  ---------------
echo "[setup] pip-installing lpips + pytorch_msssim"
pip install --quiet "${PIP_TRUSTED[@]}" pytorch_msssim lpips

# 4) build diff_surfel_bake_render against the env's torch  -------------
SUBMOD="$ROOT/diff_surfel_bake_render"
if [ ! -d "$SUBMOD" ]; then
  echo "[setup] FATAL: $SUBMOD missing — was it rsynced?"; exit 1
fi
echo "[setup] building diff_surfel_bake_render (compute_89, ~2-3 min)"
export TORCH_CUDA_ARCH_LIST="8.9"
cd "$SUBMOD"
pip install --quiet "${PIP_TRUSTED[@]}" -e . --no-build-isolation 2>&1 | tail -3
cd "$HERE"

# 5) verify  -------------------------------------------------------------
python - <<'PY'
import torch
print(f"[verify] torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"[verify] device 0 = {torch.cuda.get_device_name(0)}")
import diff_surfel_bake_render as drr
print(f"[verify] diff_surfel_bake_render OK ({drr.__file__})")
import lpips, pytorch_msssim, plyfile, PIL  # noqa: F401
print(f"[verify] lpips, pytorch_msssim, plyfile, PIL all import OK")
PY

echo "[setup] DONE.  activate with:"
echo "    source ~/miniforge3/etc/profile.d/conda.sh && conda activate $ENV_NAME"
