#!/usr/bin/env python3
"""
Bake 3D_SH_res model: SH stays in PLY, hash MLP residual → 3-channel texture.

The MLP is view-independent (hash features → RGB), so we can bake it into a
simple [N, grid_size, grid_size, 3] FP16 texture. No SH decomposition needed.

Output:
  - baked.ply: Gaussians with original SH coefficients (unchanged)
  - residual_textures.pt: [N, 8, 8, 3] FP16 RGB residual textures
  - bake_meta.json: metadata

Usage:
    python scripts/bake_sh_res.py --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
"""

import os
import sys
import json
import pickle
import glob
import math
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams


def quat_to_rotcols(quats):
    """Quaternion [N, 4] (w,x,y,z) → R_col0 [N, 3], R_col1 [N, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm

    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    R0 = torch.stack([r00, r10, r20], dim=-1)  # [N, 3]
    R1 = torch.stack([r01, r11, r21], dim=-1)  # [N, 3]
    return R0, R1


def main():
    parser = ArgumentParser(description="Bake 3D_SH_res model into SH + RGB residual textures")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ss", type=int, default=2, help="Supersample factor")
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

    # Setup model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    # Load INGP (hash encoding + MLP)
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(bake_args.model_path, iteration)
    ingp.set_active_levels(iteration)

    # Load Gaussians
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
    print(f"\n[BAKE] Gaussians after pruning: {N:,}")

    # =========================================================================
    # Bake hash MLP at UV grid per Gaussian (Python, no CUDA kernel needed)
    # =========================================================================
    grid_size = bake_args.grid_size
    ss = bake_args.ss
    bake_res = grid_size * ss
    uv_extent = bake_args.uv_extent

    print(f"[BAKE] Sampling MLP at {bake_res}×{bake_res} UV grid (ss={ss}× → {grid_size}×{grid_size})")
    print(f"[BAKE] UV extent: [-{uv_extent}, +{uv_extent}]")

    # Build UV grid (texel-center convention)
    step = 2.0 * uv_extent / bake_res
    coords = torch.arange(bake_res, dtype=torch.float32, device='cuda')
    uv_1d = (coords + 0.5) * step - uv_extent  # [-3.75, -3.25, ..., 3.75] for ss=2, gs=8
    uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')  # [bake_res, bake_res]
    uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # [bake_res^2, 2]

    # Get Gaussian params
    centers = gaussians.get_xyz          # [N, 3]
    quats = gaussians.get_rotation       # [N, 4]
    scales = gaussians.get_scaling       # [N, 2]
    R0, R1 = quat_to_rotcols(quats)     # [N, 3] each

    # Chunk to stay under ~4 GB
    n_pts = bake_res * bake_res
    max_batch = 4 * (1024**3) // (3 * 4 * n_pts)  # ~bytes per Gaussian
    chunk_size = max(1, min(N, max_batch))
    n_chunks = (N + chunk_size - 1) // chunk_size

    print(f"[BAKE] Total: {N * n_pts:,} MLP evaluations"
          f"{f' ({n_chunks} chunks)' if n_chunks > 1 else ''}")

    # Output: residual at bake resolution
    residual_hi = torch.zeros(N, n_pts, 3, device='cuda')

    # Get MLP and hash encoding ready
    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim  # 4
    mlp_input_padded = mlp[0].weight.shape[1]  # 16
    bias_col = hash_dim  # column 4 = bias

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, N)
        n_batch = c_end - c_start

        # Compute world XYZ: center + u*sx*R0 + v*sy*R1
        # centers[c_start:c_end]: [n_batch, 3]
        # uv_grid: [n_pts, 2] (u, v)
        c = centers[c_start:c_end]         # [n_batch, 3]
        sx = scales[c_start:c_end, 0:1]    # [n_batch, 1]
        sy = scales[c_start:c_end, 1:2]    # [n_batch, 1]
        r0 = R0[c_start:c_end]             # [n_batch, 3]
        r1 = R1[c_start:c_end]             # [n_batch, 3]

        u_vals = uv_grid[:, 0]  # [n_pts]
        v_vals = uv_grid[:, 1]  # [n_pts]

        # xyz = center[i] + u[j]*sx[i]*R0[i] + v[j]*sy[i]*R1[i]
        # Shape: [n_batch, n_pts, 3]
        xyz = (c.unsqueeze(1)
               + u_vals.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
               + v_vals.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))

        # Flatten for hash encoding: [n_batch * n_pts, 3]
        xyz_flat = xyz.reshape(-1, 3)

        with torch.no_grad():
            # Query hash encoding
            hash_feat = ingp._encode_3D(xyz_flat)  # [n_batch*n_pts, hash_dim]

            # Build MLP input: [hash(4) | 1.0 | zeros(11)] = 16D
            mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device='cuda')
            mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
            mlp_input[:, bias_col] = 1.0

            # Forward through MLP → take first 3 outputs (RGB residual)
            mlp_out = mlp(mlp_input)  # [n_batch*n_pts, 16]
            rgb_residual = mlp_out[:, :3]  # [n_batch*n_pts, 3]

            residual_hi[c_start:c_end] = rgb_residual.reshape(n_batch, n_pts, 3)

        del xyz, xyz_flat, hash_feat, mlp_input, mlp_out, rgb_residual
        if n_chunks > 1:
            torch.cuda.empty_cache()
            if ci % 5 == 0:
                print(f"  chunk {ci+1}/{n_chunks}")

    # Box-filter downsample if ss > 1
    if ss > 1:
        # [N, bake_res*bake_res, 3] → [N, bake_res, bake_res, 3] → [N, gs, ss, gs, ss, 3] → mean
        x = residual_hi.view(N, bake_res, bake_res, 3)
        x = x.view(N, grid_size, ss, grid_size, ss, 3)
        residual_tex = x.mean(dim=(2, 4))  # [N, gs, gs, 3]
        del residual_hi
        print(f"[BAKE] Downsampled to {grid_size}×{grid_size}")
    else:
        residual_tex = residual_hi.view(N, grid_size, grid_size, 3)

    # Transpose [N, u, v, D] → [N, v, u, D] to match CUDA render kernel convention
    # Render kernel uses (v*grid_size + u) indexing, so v must be the outer (row) dim
    residual_tex = residual_tex.transpose(1, 2).contiguous()

    print(f"\n[BAKE] Residual texture stats: mean={residual_tex.mean():.6f}, "
          f"std={residual_tex.std():.6f}, "
          f"min={residual_tex.min():.6f}, max={residual_tex.max():.6f}")

    # =========================================================================
    # Save outputs
    # =========================================================================
    output_dir = bake_args.output_dir or os.path.join(bake_args.model_path, "baked")
    os.makedirs(output_dir, exist_ok=True)

    # PLY: SH coefficients are already in the Gaussians, just save as-is
    ply_path = os.path.join(output_dir, "baked.ply")
    gaussians.save_ply(ply_path)
    print(f"[BAKE] Saved baked.ply → {ply_path}")

    # Residual textures: [N, gs, gs, 3] FP16
    tex_path = os.path.join(output_dir, "residual_textures.pt")
    torch.save(residual_tex.half().cpu(), tex_path)
    tex_mb = residual_tex.half().nelement() * 2 / 1024 / 1024
    print(f"[BAKE] Saved residual_textures.pt → {tex_path} ({tex_mb:.1f} MB)")

    # Metadata
    meta = {
        "texture_mode": "shared",
        "grid_size": grid_size,
        "residual_dim": 3,
        "uv_extent": uv_extent,
        "supersample": ss,
        "num_gaussians": N,
        "iteration": iteration,
        "method": args.method,
        "kernel": getattr(args, 'kernel', 'gaussian'),
        "sh_degree": 3,
    }
    meta_path = os.path.join(output_dir, "bake_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[BAKE] Saved bake_meta.json → {meta_path}")

    print(f"\n[BAKE] Done! Baked model saved to {output_dir}")
    print(f"[BAKE] Texture size: {tex_mb:.1f} MB (vs ~268 MB for 48D SH textures)")


if __name__ == "__main__":
    main()
