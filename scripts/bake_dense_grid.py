#!/usr/bin/env python3
"""
Bake 3D_SH_res hash MLP into a dense 3D RGB residual grid.

Instead of sampling the MLP per-Gaussian on a UV map, this evaluates the
MLP at every voxel in a uniform 3D grid covering the hash grid's voxel_range.
The result is a [R, R, R, 3] FP16 tensor that can be trilinearly interpolated
at render time, bypassing the hash grid + MLP entirely.

Output:
  - baked.ply: Gaussians with original SH coefficients (unchanged)
  - dense_rgb_grid.pt: [R, R, R, 3] FP16 RGB residual grid
  - dense_grid_meta.json: metadata (resolution, voxel_range, etc.)

Usage:
    python scripts/bake_dense_grid.py --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
    python scripts/bake_dense_grid.py --model_path ... --grid_resolution 256  # smaller for testing
"""

import os
import sys
import json
import pickle
import glob
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams


def main():
    parser = ArgumentParser(description="Bake 3D_SH_res hash MLP into dense 3D RGB grid")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--grid_resolution", type=int, default=512,
                        help="Voxel grid resolution per axis (512³=805MB, 256³=100MB)")
    parser.add_argument("--output_dir", type=str, default=None)
    bake_args = parser.parse_args()

    # Load training config
    print(f"\n[BAKE] Loading config from: {bake_args.model_path}")
    with open(os.path.join(bake_args.model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = bake_args.model_path
    args.eval = True

    config_yaml_path = os.path.join(bake_args.model_path, "config.yaml")
    cfg_model = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    assert args.method == "3D_SH_res", f"Expected 3D_SH_res method, got {args.method}"

    # Auto-detect iteration
    iteration = bake_args.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(bake_args.model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
        print(f"[CONFIG] Latest iteration: {iteration}")

    # Load INGP (hash encoding + MLP)
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(bake_args.model_path, iteration)
    ingp.set_active_levels(iteration)

    # Load Gaussians (for saving baked PLY)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level']:
            tensor = getattr(gaussians, attr)
            setattr(gaussians, attr, tensor[valid_mask])
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    N = len(gaussians.get_xyz)
    print(f"[BAKE] Gaussians after pruning: {N:,}")

    # =========================================================================
    # Bake hash MLP at dense 3D grid
    # =========================================================================
    R = bake_args.grid_resolution
    voxel_range = cfg_model.encoding.hashgrid.range  # e.g., [-1.5, 1.5]
    vmin, vmax = voxel_range[0], voxel_range[1]

    grid_size_mb = R**3 * 3 * 2 / 1024 / 1024
    print(f"\n[BAKE] Dense grid: {R}³ × 3 × FP16 = {grid_size_mb:.1f} MB")
    print(f"[BAKE] Voxel range: [{vmin}, {vmax}]³")
    print(f"[BAKE] Total voxels: {R**3:,}")

    # Get MLP and hash encoding ready
    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim  # 4
    mlp_input_padded = mlp[0].weight.shape[1]  # 16
    bias_col = hash_dim  # column 4 = bias

    print(f"[BAKE] MLP: input={mlp_input_padded}D (hash={hash_dim}D + bias@col{bias_col}), "
          f"hidden=16D, output=3D (RGB residual)")

    # Allocate output grid
    dense_grid = torch.zeros(R, R, R, 3, dtype=torch.float32, device='cuda')

    # Process in Z-slices to manage memory
    # Each slice: R × R points → R² hash queries + MLP evals
    slice_pts = R * R
    print(f"[BAKE] Processing {R} Z-slices ({slice_pts:,} points each)")

    # Precompute 1D coordinates (voxel centers)
    coords_1d = (torch.arange(R, dtype=torch.float32, device='cuda') + 0.5) / R * (vmax - vmin) + vmin

    with torch.no_grad():
        for zi in range(R):
            z_val = coords_1d[zi]

            # Build [R*R, 3] grid points for this Z-slice
            yy, xx = torch.meshgrid(coords_1d, coords_1d, indexing='ij')  # [R, R]
            xyz = torch.stack([
                xx.reshape(-1),
                yy.reshape(-1),
                torch.full((slice_pts,), z_val, device='cuda'),
            ], dim=-1)  # [R*R, 3]

            # Query hash encoding
            hash_feat = ingp._encode_3D(xyz)  # [R*R, hash_dim]

            # Build MLP input: [hash(4) | 1.0 | zeros(11)] = 16D
            mlp_input = torch.zeros(slice_pts, mlp_input_padded, device='cuda')
            mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
            mlp_input[:, bias_col] = 1.0

            # Forward through MLP → take first 3 outputs (RGB residual)
            mlp_out = mlp(mlp_input)  # [R*R, 16]
            rgb_residual = mlp_out[:, :3]  # [R*R, 3]

            dense_grid[:, :, zi, :] = rgb_residual.reshape(R, R, 3)

            if (zi + 1) % 64 == 0 or zi == R - 1:
                print(f"  Z-slice {zi+1}/{R}")

    print(f"\n[BAKE] Grid stats: mean={dense_grid.mean():.6f}, "
          f"std={dense_grid.std():.6f}, "
          f"min={dense_grid.min():.6f}, max={dense_grid.max():.6f}")

    # =========================================================================
    # Save outputs
    # =========================================================================
    output_dir = bake_args.output_dir or os.path.join(bake_args.model_path, "baked_dense")
    os.makedirs(output_dir, exist_ok=True)

    # PLY: SH coefficients unchanged
    ply_path = os.path.join(output_dir, "baked.ply")
    gaussians.save_ply(ply_path)
    print(f"[BAKE] Saved baked.ply → {ply_path}")

    # Dense RGB grid: [R, R, R, 3] FP16
    grid_path = os.path.join(output_dir, "dense_rgb_grid.pt")
    torch.save(dense_grid.half().cpu(), grid_path)
    actual_mb = dense_grid.half().nelement() * 2 / 1024 / 1024
    print(f"[BAKE] Saved dense_rgb_grid.pt → {grid_path} ({actual_mb:.1f} MB)")

    # Metadata
    meta = {
        "grid_resolution": R,
        "voxel_range": [vmin, vmax],
        "num_gaussians": N,
        "iteration": iteration,
        "method": args.method,
        "kernel": getattr(args, 'kernel', 'gaussian'),
        "sh_degree": 3,
        "hash_dim": hash_dim,
        "mlp_input_padded": mlp_input_padded,
    }
    meta_path = os.path.join(output_dir, "dense_grid_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[BAKE] Saved dense_grid_meta.json → {meta_path}")

    print(f"\n[BAKE] Done! Dense grid saved to {output_dir}")


if __name__ == "__main__":
    main()
