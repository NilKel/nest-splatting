"""Repack a baked atlas into a narrower layout (width <= MAX_W).

The original bake's atlas width grew past WebGPU's mobile maxTextureDimension2D
(8192) because the bake used a cudaArray with a 60k-pixel height cap. The
typeD pipeline doesn't need cudaArray, so we can re-shelf at narrow width
post-bake; height grows but the per-Gauss content is identical.

Output: <baked_atlas>/../baked_atlas_narrow/ with new atlas_texture.pt,
atlas_rects.pt, bake_meta.json. Other files (baked.ply, etc.) are symlinked
from the source dir so downstream scripts find them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_baked import shelf_pack_atlas


def repack(src_dir: Path, dst_dir: Path, max_w: int = 8192):
    meta = json.load(open(src_dir / "bake_meta.json"))
    W_old, H_old = meta["atlas_width"], meta["atlas_height"]
    print(f"[repack] source: {W_old} x {H_old}")
    if W_old <= max_w:
        print(f"[repack] already <= {max_w}, no repack needed")
        return False

    rects_old = torch.load(src_dir / "atlas_rects.pt", map_location="cpu", weights_only=False)
    N = rects_old.shape[0]
    resolutions = rects_old[:, 2:4].long()  # (w, h)

    print(f"[repack] shelf-packing {N} rects at width {max_w}...")
    rects_new, H_new, used_rows, util = shelf_pack_atlas(resolutions, atlas_width=max_w)
    print(f"[repack] new: {max_w} x {H_new} (used {used_rows} rows, util {util:.1f}%)")

    print(f"[repack] loading atlas_texture.pt ({W_old}x{H_old}x3)...")
    atlas_old = torch.load(src_dir / "atlas_texture.pt", map_location="cpu", weights_only=False)
    assert atlas_old.dtype == torch.uint8, f"expected uint8, got {atlas_old.dtype}"
    assert atlas_old.shape == (H_old, W_old, 3), f"unexpected atlas shape {atlas_old.shape}"

    # Save the small metadata first so the destination dir is usable even
    # if the big atlas_texture.pt save aborts (OOM kills, etc.).
    dst_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rects_new, dst_dir / "atlas_rects.pt")
    meta_new = dict(meta)
    meta_new["atlas_width"] = max_w
    meta_new["atlas_height"] = H_new
    meta_new["atlas_bc7_padded_w"] = max_w
    meta_new["atlas_bc7_padded_h"] = H_new
    meta_new["repacked_from"] = f"{W_old}x{H_old}"
    json.dump(meta_new, open(dst_dir / "bake_meta.json", "w"), indent=2)

    # Try GPU-accelerated copy if there's space; otherwise CPU loop. To bound
    # CPU peak memory, free `atlas_old` on CPU as soon as we've staged the
    # GPU copy.
    rects_old_np = rects_old.cpu().numpy().astype(np.int64)
    rects_new_np = rects_new.cpu().numpy().astype(np.int64)
    use_gpu = False
    if torch.cuda.is_available():
        bytes_needed = (H_old * W_old + H_new * max_w) * 3
        free, _ = torch.cuda.mem_get_info()
        if bytes_needed * 1.2 < free:
            use_gpu = True
            print(f"[repack] using GPU (free {free/1e9:.1f} GB, need {bytes_needed/1e9:.1f} GB)")

    if use_gpu:
        atlas_old_g = atlas_old.cuda()
        del atlas_old  # free 1.8GB CPU now that GPU has its own copy
        atlas_new_g = torch.zeros((H_new, max_w, 3), dtype=torch.uint8, device="cuda")
        log_every = max(1, N // 20)
        for i in range(N):
            u0o, v0o, w, h = rects_old_np[i]
            u0n, v0n, _, _ = rects_new_np[i]
            if w <= 2 or h <= 2:
                continue
            atlas_new_g[v0n:v0n+h, u0n:u0n+w] = atlas_old_g[v0o:v0o+h, u0o:u0o+w]
            if (i+1) % log_every == 0 or i == N - 1:
                print(f"[repack]   {i+1}/{N}")
        del atlas_old_g
        torch.cuda.empty_cache()
        atlas_new = atlas_new_g.cpu()
        del atlas_new_g
        torch.cuda.empty_cache()
    else:
        atlas_new = torch.zeros((H_new, max_w, 3), dtype=torch.uint8)
        log_every = max(1, N // 20)
        for i in range(N):
            u0o, v0o, w, h = rects_old_np[i]
            u0n, v0n, _, _ = rects_new_np[i]
            if w <= 2 or h <= 2:
                continue
            atlas_new[v0n:v0n+h, u0n:u0n+w] = atlas_old[v0o:v0o+h, u0o:u0o+w]
            if (i+1) % log_every == 0 or i == N - 1:
                print(f"[repack]   {i+1}/{N}")
        del atlas_old

    # torch.save of multi-GB uint8 tensors trips an iostream/zip-writer
    # error inside the script (not reproducible standalone). Stream the
    # raw bytes via numpy instead — atlas_texture.u8.bin + shape sidecar.
    # The producer reads either format.
    print(f"[repack] saving atlas_texture.u8.bin ({atlas_new.numel()/1e9:.2f} GB) via numpy tofile...")
    arr = atlas_new.numpy()
    arr.tofile(str(dst_dir / "atlas_texture.u8.bin"))
    (dst_dir / "atlas_texture.u8.shape").write_text(f"{arr.shape[0]},{arr.shape[1]},{arr.shape[2]}")

    # Symlink the rest of the bake dir so downstream scripts find baked.ply etc.
    for f in src_dir.iterdir():
        if f.name in ("atlas_texture.pt", "atlas_rects.pt", "bake_meta.json"):
            continue
        if f.name in ("atlas_texture.bc7", "scene.nat2", "scene_astc.nat2", "renders", "vq"):
            continue  # stale; will be regenerated
        link = dst_dir / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    print(f"[repack] done -> {dst_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to baked_atlas dir")
    ap.add_argument("--dst", default=None, help="dest dir (default: <src>/../baked_atlas_narrow)")
    ap.add_argument("--max-w", type=int, default=8192)
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.parent / "baked_atlas_narrow"
    repack(src, dst, args.max_w)


if __name__ == "__main__":
    main()
