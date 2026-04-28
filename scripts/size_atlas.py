"""Print atlas size for hashgrid-only vs min(view, hash) Nyquist strategies.
No baking, no rendering — just resolution selection + shelf packing.
Usage: python scripts/size_atlas.py --model_path PATH"""
import argparse
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arguments import ModelParams
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config

from scripts.benchmark_baked import (
    compute_adaptive_resolution,
    compute_view_max_footprint,
    compute_view_aware_resolution,
    shelf_pack_atlas,
)


def report(label, resolutions, atlas_width=4096):
    rects, atlas_h, used_rows, util = shelf_pack_atlas(resolutions, atlas_width=atlas_width)
    total_texels = (resolutions[:, 0].long() * resolutions[:, 1].long()).sum().item()
    raw_mb = total_texels * 3 * 2 / (1024 ** 2)            # FP16 RGB
    packed_mb = atlas_h * atlas_width * 3 * 2 / (1024 ** 2)
    print(f"  {label:<28} raw {raw_mb:>7.1f} MB   packed {packed_mb:>7.1f} MB "
          f"({atlas_width}x{atlas_h}, {util:.1f}% util)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--max_res", type=int, default=64)
    ap.add_argument("--min_res", type=int, default=4)
    ap.add_argument("--atlas_width", type=int, default=4096)
    ap.add_argument("--uv_extent", type=float, default=4.0)
    args = ap.parse_args()

    with open(os.path.join(args.model_path, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)
    train_args.model_path = args.model_path

    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    dataset = model_params.extract(train_args)

    yaml_path = os.path.join(args.model_path, "config.yaml")
    cfg = Config(yaml_path) if os.path.exists(yaml_path) else Config(train_args.yaml)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False, full_args=train_args)
    iteration = scene.loaded_iter
    gaussians.base_opacity = cfg.surfel.tg_base_alpha

    ingp = INGP(cfg, args=train_args).to('cuda')
    ingp.load_model(args.model_path, iteration)
    ingp.set_active_levels(iteration)

    # Drop dead Gaussians (mirrors benchmark_baked main loop).
    dead = (gaussians.get_opacity <= 0.005).squeeze(-1)
    if int(dead.sum()) > 0:
        keep = ~dead
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level',
                     '_gaussian_features', '_shape', '_flex_beta',
                     '_sb_params', '_sg_directions', '_sg_sharpness_sg', '_sg_rgb',
                     '_sv_sites', '_sv_colors', '_gamma', '_adaptive_features',
                     '_adaptive_cat_weight', '_adaptive_zero_weight', '_gate_logits']:
            t = getattr(gaussians, attr, None)
            if t is not None and t.numel() > 0 and t.shape[0] == keep.shape[0]:
                setattr(gaussians, attr, t[keep.to(t.device)])

    N = gaussians.get_xyz.shape[0]
    print(f"[size_atlas] {N:,} Gaussians")

    embeddings, offsets, num_levels, per_level_scale, base_resolution, _, _ = \
        ingp.hash_encoding.get_params()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    finest = base_resolution * (per_level_scale ** (num_levels - 1))
    cell_size = (voxel_max - voxel_min) / finest

    res_hash = compute_adaptive_resolution(
        gaussians.get_scaling, cell_size, uv_extent=args.uv_extent,
        max_res=args.max_res, min_res=args.min_res)

    # Uncapped hashgrid Nyquist for diagnostics — how many Gaussians WANT
    # more than max_res from the hashgrid?
    scales = gaussians.get_scaling
    n_cells = 2.0 * args.uv_extent * scales / cell_size
    nyq_uncapped = (2.0 * n_cells).clamp(min=1.0)
    res_hash_uncapped = (2.0 ** torch.ceil(torch.log2(nyq_uncapped))).int()

    print(f"[size_atlas] hashgrid finest={finest:.0f}  cell_size={cell_size:.6f}")
    print(f"[size_atlas] scale percentiles: "
          f"p50={scales.flatten().median():.4f} "
          f"p90={scales.flatten().quantile(0.9):.4f} "
          f"p99={scales.flatten().quantile(0.99):.4f} "
          f"max={scales.flatten().max():.4f}")
    above_cap = (res_hash_uncapped > args.max_res).any(dim=1).sum().item()
    print(f"[size_atlas] Gaussians wanting res_hash > max_res ({args.max_res}): "
          f"{above_cap:,} / {N:,} ({100.0*above_cap/N:.1f}%)")

    train_cameras = scene.getTrainCameras()
    print(f"[size_atlas] walking {len(train_cameras)} train views for projected footprints...")
    fp = compute_view_max_footprint(
        gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation,
        train_cameras, k_sigma=4.0)
    res_view = compute_view_aware_resolution(
        fp, max_res=args.max_res, min_res=args.min_res, nyquist_factor=2.0)
    res_min = torch.minimum(res_view, res_hash)

    view_lim = (res_view < res_hash).any(dim=1).sum().item()
    hash_lim = (res_hash <= res_view).all(dim=1).sum().item()
    print(f"[size_atlas] footprint median (px): w={fp[:,0].median():.1f} h={fp[:,1].median():.1f}  | "
          f"view-limited: {view_lim:,}  hashgrid-limited: {hash_lim:,}")

    # The user's question: are there a lot of Gaussians where uncapped_hash is
    # huge but viewing is small? Those benefit from view-aware sizing
    # only if we let res_hash exceed max_res (i.e., raise the cap or remove it).
    huge_hash = (res_hash_uncapped > args.max_res).any(dim=1)
    if int(huge_hash.sum()) > 0:
        # Among huge-hash Gaussians, how many have view_res < max_res
        # (i.e., view actually demands LESS than the cap)?
        view_smaller_than_cap = (res_view < args.max_res).any(dim=1) & huge_hash
        print(f"[size_atlas] of {int(huge_hash.sum()):,} huge-hash Gaussians, "
              f"{int(view_smaller_than_cap.sum()):,} have at least one view axis < max_res "
              f"(would benefit from view-aware sizing if cap raised).")
        # Also: how many would benefit from min(view, hash_uncapped) where the
        # VIEW dominates (view < uncapped hash)?
        view_dom_uncapped = (res_view < res_hash_uncapped).any(dim=1).sum().item()
        print(f"[size_atlas] view < UNCAPPED hash on at least one axis: "
              f"{view_dom_uncapped:,} ({100.0*view_dom_uncapped/N:.1f}%)")

    print()
    report("hashgrid-only",  res_hash, args.atlas_width)
    report("view-only",      res_view, args.atlas_width)
    report("min(view, hash)", res_min, args.atlas_width)


if __name__ == "__main__":
    main()
