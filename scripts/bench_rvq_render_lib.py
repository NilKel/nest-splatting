"""Helpers for the end-to-end RVQ render bench.
Kept separate from bench_rvq_render.py so the install path can be reused.
"""
import os, time
import torch


def reorder_indices_surfel_major(indices_row_major, rects, atlas_HW, block_size=4):
    device = indices_row_major.device
    H, W = atlas_HW
    B = block_size
    M = rects.shape[0]

    bw = (rects[:, 2] // B).long()
    bh = (rects[:, 3] // B).long()
    surfel_offsets = torch.zeros(M + 1, dtype=torch.int64, device=device)
    surfel_offsets[1:] = torch.cumsum(bw * bh, dim=0)
    N_used = int(surfel_offsets[-1].item())
    assert N_used == indices_row_major.shape[1], (
        f"surfel-block sum {N_used} != indices N {indices_row_major.shape[1]}")

    used_mask = torch.zeros((H // B, W // B), dtype=torch.bool, device=device)
    for i in range(M):
        ww = int(rects[i, 2].item()); hh = int(rects[i, 3].item())
        if ww == 0 or hh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        used_mask[bv0:bv0 + hh // B, bu0:bu0 + ww // B] = True
    rm_id_flat = torch.cumsum(used_mask.reshape(-1).long(), dim=0) - 1
    rm_id_flat[~used_mask.reshape(-1)] = -1
    rm_id = rm_id_flat.reshape(H // B, W // B)

    perm = torch.empty(N_used, dtype=torch.int64, device=device)
    for i in range(M):
        nw = int(bw[i].item()); nh = int(bh[i].item())
        if nw == 0 or nh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        off = int(surfel_offsets[i].item())
        sub = rm_id[bv0:bv0+nh, bu0:bu0+nw].reshape(-1)
        perm[off:off+nw*nh] = sub
    indices_sm = indices_row_major[:, perm].contiguous().to(torch.uint8)
    return indices_sm, surfel_offsets


def install_rvq_atlas(bake_dir, device='cuda'):
    """Load vq/ artifacts, surfel-major-reorder, install via set_atlas_rvq."""
    from diff_surfel_bake_render import (set_atlas_rvq, clear_atlas_bc7,
                                          clear_atlas_cache)
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
    print(f"[RVQ] L={L} K={K} D={D} atlas {H}x{W} ({rects.shape[0]:,} surfels, "
          f"{ind_raw.shape[1]:,} used blocks)")

    print("[RVQ] reordering row-major → surfel-major …")
    t0 = time.time()
    ind_sm, offsets = reorder_indices_surfel_major(ind_raw, rects, (H, W), B)
    print(f"  remap done in {time.time()-t0:.1f}s")

    clear_atlas_bc7()
    clear_atlas_cache()
    # Read atlas dequant params from bake_meta so the uint8-RGBA 2D codebook
    # texture uses the same scale/offset as the atlas itself.
    import json
    meta_path = os.path.join(bake_dir, "bake_meta.json")
    atlas_scale = 1.0; atlas_offset = 0.0
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        atlas_scale  = float(m.get("atlas_scale", 1.0))
        atlas_offset = float(m.get("atlas_offset", 0.0))
    set_atlas_rvq(cb, ind_sm, offsets, B, atlas_scale, atlas_offset)
    return cb, ind_sm, offsets
