"""
Bench the *pure-RVQ* submodule (submodules/diff_surfel_bake_render_rvq).

This fork has every non-RVQ branch + every toggle stripped from the kernel,
so the per-fragment cost is the minimum global-memory RVQ-bilinear decode.
Compare its FPS directly to the original BC7 baseline (run via the regular
benchmark_baked.py — different binary, same scene, apples-to-apples).

Usage:
  python scripts/bench_rvq_submodule.py \\
      --bake_dir   outputs/.../baked_atlas \\
      --model_path outputs/.../<config>  \\
      [--use_bply baked.bply]            \\
      [--num_warmup 10 --num_benchmark 30]
"""
import argparse, json, os, sys, subprocess, time
import numpy as np
import torch

# Force-use the forked submodule (alias as drop-in for prepare_gaussian_inputs / render_baked
# helpers, which live in scripts/benchmark_baked.py and the bake_render bindings).
sys.path.insert(0, os.path.dirname(__file__))
# Use the pure-RVQ submodule but expose it under the original module name so the
# existing benchmark_baked.evaluate_baked() helper picks it up transparently.
import diff_surfel_bake_render_rvq as _rvq_pkg
sys.modules['diff_surfel_bake_render'] = _rvq_pkg

# Now import benchmark_baked; its imports of diff_surfel_bake_render will resolve
# to the RVQ-only fork above.
import benchmark_baked as BB
from bench_rvq_render_lib import reorder_indices_surfel_major

# Reimport so the inner imports pick up the alias.
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from argparse import ArgumentParser


def install_rvq_atlas_via_rvq_module(bake_dir, device='cuda'):
    from diff_surfel_bake_render_rvq import set_atlas_rvq, clear_atlas_rvq
    cb = torch.load(os.path.join(bake_dir, "vq/codebooks.pt"),
                    map_location=device, weights_only=False).to(torch.float16).contiguous()
    ind_raw = torch.load(os.path.join(bake_dir, "vq/indices.pt"),
                          map_location=device, weights_only=False).long()
    block_meta = torch.load(os.path.join(bake_dir, "vq/block_meta.pt"),
                            map_location=device, weights_only=False)
    rects = torch.load(os.path.join(bake_dir, "atlas_rects.pt"),
                       map_location=device, weights_only=False)
    H = int(block_meta["atlas_HW"][0]); W = int(block_meta["atlas_HW"][1])
    B = int(block_meta["block"])
    L, K, D = cb.shape
    print(f"[RVQ-SUBMODULE] L={L} K={K} D={D} atlas {H}x{W} "
          f"({rects.shape[0]:,} surfels, {ind_raw.shape[1]:,} used blocks)")
    t0 = time.time()
    ind_sm, offsets = reorder_indices_surfel_major(ind_raw, rects, (H, W), B)
    print(f"  remap row-major → surfel-major in {time.time()-t0:.1f}s")
    meta = json.load(open(os.path.join(bake_dir, "bake_meta.json")))
    atlas_scale  = float(meta.get("atlas_scale", 1.0))
    atlas_offset = float(meta.get("atlas_offset", 0.0))
    clear_atlas_rvq()
    set_atlas_rvq(cb, ind_sm, offsets, B, atlas_scale, atlas_offset)


if __name__ == "__main__":
    # Patch benchmark_baked to install via the RVQ submodule instead of the lib helper.
    sys.modules['bench_rvq_render_lib'].install_rvq_atlas = install_rvq_atlas_via_rvq_module

    # Now defer to benchmark_baked.main() with --rvq forced.
    sys.argv += ["--rvq", "--skip_bake"]
    BB.main()
