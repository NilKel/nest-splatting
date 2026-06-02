"""Bench the *paired RVQ* submodule — L=4 → L=2 collapsed pair codebooks."""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
import diff_surfel_bake_render_rvq_paired as _pkg
sys.modules['diff_surfel_bake_render'] = _pkg

import benchmark_baked as BB
from bench_rvq_render_lib import reorder_indices_surfel_major


def install_rvq_atlas_via_paired_module(bake_dir, device='cuda'):
    from diff_surfel_bake_render_rvq_paired import set_atlas_rvq, clear_atlas_rvq
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
    print(f"[RVQ-PAIRED-SUBMODULE] L={L} K={K} D={D} atlas {H}x{W} "
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
    sys.modules['bench_rvq_render_lib'].install_rvq_atlas = install_rvq_atlas_via_paired_module
    sys.argv += ["--rvq", "--skip_bake"]
    BB.main()
