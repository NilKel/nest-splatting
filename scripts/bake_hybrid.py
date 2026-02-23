#!/usr/bin/env python3
"""
Bake 3D_SH_TC model into Mean SH + Residual Textures.

Takes a trained 3D_SH_TC checkpoint and produces:
  Shared mode (--texture shared):
    - baked.ply: Gaussian with Mean SH coefficients
    - residual_textures.pt: [N, 8, 8, 48] FP16 SH residual textures (per_texel_sh - mean_sh)
  Atlas mode (--texture atlas):
    - baked.ply: Gaussian with Mean SH coefficients
    - atlas_texture.pt: [atlas_size, atlas_size, 3] FP16 packed atlas (DC residual)
    - atlas_rects.pt: [N, 4] float per-Gaussian UV rects in atlas pixel coords

Usage:
    python scripts/bake_hybrid.py --model_path outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma
    python scripts/bake_hybrid.py --model_path ... --texture atlas
"""

import os
import sys
import math
import json
import pickle
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, OptimizationParams
from diff_surfel_bake import set_mlp_weights, bake_gaussians


SH_C0 = 0.28209479177387814
ALLOWED_RES = [2, 4, 8, 16, 32, 64]


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
        return args

    raise FileNotFoundError(f"No training config found in {model_path}")


def compute_per_gaussian_resolution(scales, cell_size, uv_extent=4.0, res_mult=1.0):
    """
    Compute per-Gaussian texture resolution from Nyquist criterion vs hash cell size.

    Bake samples span [-uv_extent*s, +uv_extent*s] in world space = 2*uv_extent*max_scale.
    Hash cells across that extent = 2*uv_extent*max_scale / cell_size.
    Nyquist requires 2 samples per cell → grid_size = 2 * (2*uv_extent*max_scale / cell_size).
    res_mult scales on top of the Nyquist baseline.
    """
    max_scale = scales.max(dim=1).values  # [N]
    n_cells = 2.0 * uv_extent * max_scale / cell_size  # hash cells across Gaussian
    raw_res = 2.0 * res_mult * n_cells  # Nyquist: 2 samples per cell

    resolutions = torch.full((len(scales),), ALLOWED_RES[0], dtype=torch.int32, device=scales.device)
    for res in ALLOWED_RES:
        resolutions[raw_res > (res / 2.0)] = res

    return resolutions.clamp(min=ALLOWED_RES[0], max=ALLOWED_RES[-1])


def compute_atlas_height(resolutions, atlas_width):
    """Compute minimum atlas height needed for shelf packing."""
    res_cpu = resolutions.cpu().numpy()
    height = 0
    for sz in sorted(set(int(x) for x in res_cpu), reverse=True):
        count = int((res_cpu == sz).sum())
        per_row = atlas_width // sz
        rows = (count + per_row - 1) // per_row
        height += rows * sz
    return int(height)


def shelf_pack_atlas(resolutions, atlas_width=4096, atlas_height=None):
    """
    Pack square patches into atlas using shelf-first-fit-decreasing.

    If atlas_height is None, auto-computes the needed height.

    Returns:
        atlas_rects: [N, 4] float tensor (u0_px, v0_px, w_px, h_px)
        atlas_height: int, actual atlas height used
        utilization: float, percentage of atlas area used
    """
    N = len(resolutions)
    res_cpu = resolutions.cpu().numpy()

    # Auto-compute atlas height if not specified
    if atlas_height is None:
        needed = compute_atlas_height(resolutions, atlas_width)
        # Round up to multiple of 64 for alignment
        atlas_height = ((needed + 63) // 64) * 64
        atlas_height = max(atlas_height, 64)

    order = np.argsort(-res_cpu)  # descending by resolution

    # Shelves: list of [y_start, height, x_cursor]
    shelves = []
    rects = np.zeros((N, 4), dtype=np.float32)

    for idx in order:
        sz = int(res_cpu[idx])
        placed = False

        # Try existing shelf with matching height
        for shelf in shelves:
            if shelf[1] >= sz and shelf[2] + sz <= atlas_width:
                rects[idx] = [shelf[2], shelf[0], sz, sz]
                shelf[2] += sz
                placed = True
                break

        if not placed:
            y_start = max((s[0] + s[1] for s in shelves), default=0)
            if y_start + sz > atlas_height:
                print(f"[WARN] Atlas overflow at Gaussian {idx} (res={sz}), "
                      f"y={y_start}+{sz} > {atlas_height}")
                rects[idx] = [0, 0, 2, 2]  # fallback
                continue
            shelves.append([y_start, sz, sz])
            rects[idx] = [0, y_start, sz, sz]

    used_rows = max((s[0] + s[1] for s in shelves), default=0)
    total_area = float(np.sum(res_cpu.astype(np.int64) ** 2))
    utilization = total_area / (atlas_width * atlas_height) * 100

    return torch.from_numpy(rects).float(), atlas_height, used_rows, utilization


def compute_gaussian_weights(grid_res, uv_extent=4.0):
    """Compute G(u,v) = exp(-0.5*(u^2+v^2)) weights for the UV grid.
    Returns [grid_res*grid_res] weight tensor (normalized to sum=1).
    """
    step = 2.0 * uv_extent / grid_res
    coords = torch.arange(grid_res, dtype=torch.float32)
    uv = (coords + 0.5) * step - uv_extent  # texel-center convention
    uu, vv = torch.meshgrid(uv, uv, indexing='ij')
    rho = uu**2 + vv**2
    G = torch.exp(-0.5 * rho).reshape(-1)  # [grid_res^2]
    return (G / G.sum()).cuda()


def compute_weighted_mean(sh_grid, grid_res, uv_extent=4.0):
    """Compute G-weighted mean SH over the UV grid.
    sh_grid: [N, grid_res^2, 48]
    Returns: [N, 48]
    """
    w = compute_gaussian_weights(grid_res, uv_extent)  # [grid_res^2]
    return (sh_grid * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)


def compute_residual(sh_grid, mean_sh, grid_res):
    """Compute DC residual texture from SH grid and mean SH (3D per texel)."""
    n_group = sh_grid.shape[0]
    n_pts = grid_res * grid_res
    residual = torch.zeros(n_group, n_pts, 3, device='cuda')
    for ch in range(3):
        pred_dc = SH_C0 * sh_grid[:, :, ch * 16] + 0.5
        base_dc = SH_C0 * mean_sh[:, ch * 16:ch * 16 + 1] + 0.5
        residual[:, :, ch] = torch.clamp(pred_dc, min=0) - torch.clamp(base_dc, min=0)
    return residual.view(n_group, grid_res, grid_res, 3)


def compute_sh_residual(sh_grid, mean_sh, grid_res):
    """Compute full 48D SH residual: per_texel_sh - mean_sh."""
    # sh_grid: [n_group, n_pts, 48]
    # mean_sh: [n_group, 48]
    residual = sh_grid - mean_sh.unsqueeze(1)  # [n_group, n_pts, 48]
    return residual.view(sh_grid.shape[0], grid_res, grid_res, 48)


def box_downsample(sh_grid_hi, hi_res, lo_res, ss, C=48):
    """Box-filter downsample: average ss×ss blocks.
    sh_grid_hi: [N, hi_res*hi_res, C]  (hi_res = lo_res * ss)
    Returns: [N, lo_res*lo_res, C]
    """
    N = sh_grid_hi.shape[0]
    # Reshape to spatial grid, then fold ss×ss blocks
    x = sh_grid_hi.view(N, hi_res, hi_res, C)
    x = x.view(N, lo_res, ss, lo_res, ss, C)
    x = x.mean(dim=(2, 4))  # average over the two ss dims
    return x.reshape(N, lo_res * lo_res, C)


def main():
    parser = ArgumentParser(description="Bake 3D_SH_TC model into Mean SH + Residual Textures")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (-1 = latest)")
    parser.add_argument("--grid_size", type=int, default=8, help="UV grid resolution per side (shared mode)")
    parser.add_argument("--uv_extent", type=float, default=4.0, help="UV sampling extent (matches AABB cutoff)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: model_path/baked/)")
    parser.add_argument("--texture", choices=["shared", "atlas"], default="shared",
                        help="Texture mode: 'shared' = fixed NxN per Gaussian, 'atlas' = variable-res packed atlas")
    parser.add_argument("--atlas_size", type=int, default=4096, help="Atlas texture dimensions (atlas mode)")
    parser.add_argument("--res_mult", type=float, default=1.0, help="Multiplier on per-Gaussian atlas resolution (e.g. 2.0 = double)")
    parser.add_argument("--ss", type=int, default=2, help="Supersample factor (bake at ss× resolution, box-filter downsample)")
    bake_args = parser.parse_args()

    # Load training config
    print(f"\n[BAKE] Loading config from: {bake_args.model_path}")
    args = load_training_config(bake_args.model_path)
    args.model_path = bake_args.model_path

    # Load YAML config
    config_yaml_path = os.path.join(bake_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    assert args.method == "3D_SH_TC", f"Expected 3D_SH_TC method, got {args.method}"

    # Auto-detect iteration
    iteration = bake_args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(bake_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth checkpoints found in {bake_args.model_path}")

    # Setup model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    args.eval = True
    dataset = model_params.extract(args)

    # Load INGP
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(bake_args.model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    ingp.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        gaussians._xyz = gaussians._xyz[valid_mask]
        gaussians._features_dc = gaussians._features_dc[valid_mask]
        gaussians._features_rest = gaussians._features_rest[valid_mask]
        gaussians._opacity = gaussians._opacity[valid_mask]
        gaussians._scaling = gaussians._scaling[valid_mask]
        gaussians._rotation = gaussians._rotation[valid_mask]
        gaussians._appearance_level = gaussians._appearance_level[valid_mask]
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
            gaussians._gaussian_features = gaussians._gaussian_features[valid_mask.to(gaussians._gaussian_features.device)]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    N = len(gaussians.get_xyz)
    print(f"\n[BAKE] Gaussians after pruning: {N:,}")

    # =========================================================================
    # Step 1: Upload MLP weights
    # =========================================================================
    mlp_weights = ingp.get_fused_mlp_weights()
    assert mlp_weights is not None, "No MLP weights found (not a fused mode?)"
    W1, W2, W3 = mlp_weights
    print(f"[BAKE] MLP weights: W1={list(W1.shape)}, W2={list(W2.shape)}, W3={list(W3.shape)}")
    set_mlp_weights(W1, W2, W3, is_sh_mode=True)

    # =========================================================================
    # Step 2: Get hash grid parameters
    # =========================================================================
    hash_encoding = ingp.hash_encoding
    embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_encoding.get_params()

    hash_features = embeddings.half()
    voxel_min = ingp.voxel_range[0]
    voxel_max = ingp.voxel_range[1]
    l_scale = math.log2(per_level_scale)
    contract = ingp.contract

    total_levels = ingp.levels
    hybrid_levels = ingp.hybrid_levels
    active_hashgrid_levels = total_levels - hybrid_levels

    # Finest hashgrid level resolution
    finest_resolution = base_resolution * (per_level_scale ** (num_levels - 1))
    cell_size = (voxel_max - voxel_min) / finest_resolution

    print(f"[BAKE] Hash grid: {num_levels} levels, scale={per_level_scale:.4f}, "
          f"base={base_resolution}, range=[{voxel_min}, {voxel_max}]")
    print(f"[BAKE] Finest resolution: {finest_resolution:.0f}, cell size: {cell_size:.6f}")
    print(f"[BAKE] Level split: {hybrid_levels} Gaussian + {active_hashgrid_levels} hash")

    uv_extent = bake_args.uv_extent
    output_dir = bake_args.output_dir or os.path.join(bake_args.model_path, "baked")
    os.makedirs(output_dir, exist_ok=True)

    # Common bake args for kernel calls
    bake_kwargs = dict(
        hash_features=hash_features.contiguous(),
        level_offsets=offsets.int().contiguous(),
        voxel_min=voxel_min,
        voxel_max=voxel_max,
        l_scale=l_scale,
        Base=base_resolution,
        align_corners=align_corners,
        interp=interp_id,
        if_contract=contract,
        active_hashgrid_levels=active_hashgrid_levels,
        uv_extent=uv_extent,
    )

    if bake_args.texture == "atlas":
        # =================================================================
        # ATLAS MODE: variable-resolution per Gaussian
        # =================================================================
        atlas_size = bake_args.atlas_size

        # Compute per-Gaussian resolution
        resolutions = compute_per_gaussian_resolution(
            gaussians.get_scaling, cell_size, uv_extent=bake_args.uv_extent, res_mult=bake_args.res_mult)
        print(f"\n[ATLAS] Resolution distribution:")
        for res in ALLOWED_RES:
            count = (resolutions == res).sum().item()
            if count > 0:
                print(f"  {res:>2}×{res:<2}: {count:>7,} Gaussians")

        # Shelf-pack into atlas (auto-computes height)
        atlas_rects, atlas_height, used_rows, utilization = shelf_pack_atlas(
            resolutions, atlas_width=atlas_size)
        atlas_rects = atlas_rects.cuda()
        print(f"[ATLAS] Packed into {atlas_size}×{atlas_height}, "
              f"used {used_rows}/{atlas_height} rows, {utilization:.1f}% utilization")

        # Allocate atlas and mean SH buffer
        atlas = torch.zeros(atlas_height, atlas_size, 3, device='cuda', dtype=torch.float32)
        mean_sh = torch.zeros(N, 48, device='cuda')

        # Bake per resolution group (with supersampling)
        ss = bake_args.ss
        unique_res = resolutions.unique().sort().values
        for res_val in unique_res:
            res = res_val.item()
            bake_res = res * ss  # supersample resolution
            mask = (resolutions == res)
            group_idx = mask.nonzero(as_tuple=True)[0]
            n_group = len(group_idx)

            # Estimate memory: n_group * bake_res^2 * 48 * 4 bytes (float32)
            # Chunk to stay under ~8 GB per batch
            max_samples = 8 * (1024**3) // (48 * 4)  # ~44M samples
            chunk_size = max(1, max_samples // (bake_res * bake_res))
            n_chunks = (n_group + chunk_size - 1) // chunk_size

            print(f"[ATLAS] Baking {n_group:,} Gaussians at {bake_res}×{bake_res} "
                  f"(ss={ss}× → {res}×{res}) "
                  f"= {n_group * bake_res * bake_res:,} samples"
                  f"{f' ({n_chunks} chunks)' if n_chunks > 1 else ''}...")

            rects_group = atlas_rects[group_idx].cpu().numpy()

            for ci in range(n_chunks):
                c_start = ci * chunk_size
                c_end = min(c_start + chunk_size, n_group)
                c_idx = group_idx[c_start:c_end]

                with torch.no_grad():
                    sh_grid_hi = bake_gaussians(
                        centers=gaussians.get_xyz[c_idx].contiguous(),
                        quats=gaussians.get_rotation[c_idx].contiguous(),
                        scales=gaussians.get_scaling[c_idx].contiguous(),
                        gauss_features=gaussians.get_gaussian_features[c_idx].contiguous(),
                        appearance_levels=gaussians._appearance_level[c_idx].int().contiguous(),
                        grid_size=bake_res,
                        **bake_kwargs,
                    )

                # Box-filter downsample
                if ss > 1:
                    sh_grid = box_downsample(sh_grid_hi, bake_res, res, ss)
                    del sh_grid_hi
                else:
                    sh_grid = sh_grid_hi

                # G-weighted mean SH for this chunk
                mean_sh_chunk = compute_weighted_mean(sh_grid, res, bake_args.uv_extent)
                mean_sh[c_idx] = mean_sh_chunk

                # DC residual
                residual = compute_residual(sh_grid, mean_sh_chunk, res)

                # Write into atlas
                for i in range(c_end - c_start):
                    u0 = int(rects_group[c_start + i, 0])
                    v0 = int(rects_group[c_start + i, 1])
                    atlas[v0:v0+res, u0:u0+res, :] = residual[i]

                del sh_grid, residual, mean_sh_chunk
                torch.cuda.empty_cache()

        # Stats
        print(f"\n[BAKE] Mean SH stats: mean={mean_sh.mean():.6f}, std={mean_sh.std():.6f}")
        print(f"[BAKE] Atlas stats: mean={atlas.mean():.6f}, std={atlas.std():.6f}, "
              f"min={atlas.min():.6f}, max={atlas.max():.6f}")

        # Save atlas outputs
        mean_sh_per_channel = mean_sh.view(N, 3, 16)
        mean_sh_ply = mean_sh_per_channel.permute(0, 2, 1)
        gaussians._features_dc = nn.Parameter(mean_sh_ply[:, 0:1, :].contiguous())
        gaussians._features_rest = nn.Parameter(mean_sh_ply[:, 1:, :].contiguous())

        ply_path = os.path.join(output_dir, "baked.ply")
        gaussians.save_ply(ply_path)
        print(f"\n[BAKE] Saved baked.ply → {ply_path}")

        atlas_path = os.path.join(output_dir, "atlas_texture.pt")
        torch.save(atlas.half().cpu(), atlas_path)
        atlas_mb = atlas.half().nelement() * 2 / 1024 / 1024
        print(f"[BAKE] Saved atlas_texture.pt → {atlas_path} ({atlas_mb:.1f} MB)")

        rects_path = os.path.join(output_dir, "atlas_rects.pt")
        torch.save(atlas_rects.cpu(), rects_path)
        print(f"[BAKE] Saved atlas_rects.pt → {rects_path}")

        meta = {
            "texture_mode": "atlas",
            "atlas_width": int(atlas_size),
            "atlas_height": int(atlas_height),
            "atlas_used_rows": int(used_rows),
            "atlas_utilization": float(round(utilization, 1)),
            "uv_extent": uv_extent,
            "supersample": ss,
            "num_gaussians": N,
            "iteration": iteration,
            "method": args.method,
            "kernel": getattr(args, 'kernel', 'gaussian'),
            "sh_degree": 3,
        }

    else:
        # =================================================================
        # SHARED MODE: fixed grid_size per Gaussian
        # =================================================================
        grid_size = bake_args.grid_size
        ss = bake_args.ss
        bake_res = grid_size * ss  # supersample resolution

        # Chunk to stay under ~4 GB output tensor per batch
        max_samples = 4 * (1024**3) // (48 * 4)  # ~22M samples
        chunk_size = max(1, max_samples // (bake_res * bake_res))
        n_chunks = (N + chunk_size - 1) // chunk_size

        print(f"\n[BAKE] Baking {N:,} Gaussians × {bake_res}×{bake_res} grid "
              f"(ss={ss}× → {grid_size}×{grid_size})"
              f" = {N * bake_res * bake_res:,} samples"
              f"{f' ({n_chunks} chunks)' if n_chunks > 1 else ''}")
        print(f"[BAKE] UV extent: [-{uv_extent}, +{uv_extent}]")

        # Allocate output buffers at final (downsampled) resolution
        sh_grid = torch.zeros(N, grid_size * grid_size, 48, device='cuda')

        for ci in range(n_chunks):
            c_start = ci * chunk_size
            c_end = min(c_start + chunk_size, N)
            c_slice = slice(c_start, c_end)

            with torch.no_grad():
                sh_grid_hi = bake_gaussians(
                    centers=gaussians.get_xyz[c_slice].contiguous(),
                    quats=gaussians.get_rotation[c_slice].contiguous(),
                    scales=gaussians.get_scaling[c_slice].contiguous(),
                    gauss_features=gaussians.get_gaussian_features[c_slice].contiguous(),
                    appearance_levels=gaussians._appearance_level[c_slice].int().contiguous(),
                    grid_size=bake_res,
                    **bake_kwargs,
                )

            torch.cuda.synchronize()

            # Box-filter downsample ss×ss → 1 texel
            if ss > 1:
                sh_grid[c_slice] = box_downsample(sh_grid_hi, bake_res, grid_size, ss)
            else:
                sh_grid[c_slice] = sh_grid_hi

            del sh_grid_hi
            if n_chunks > 1:
                torch.cuda.empty_cache()
                if ci % 5 == 0:
                    print(f"  chunk {ci+1}/{n_chunks}")

        print(f"[BAKE] SH grid shape: {list(sh_grid.shape)}")

        mean_sh = compute_weighted_mean(sh_grid, grid_size, uv_extent)  # [N, 48]
        residual_tex = compute_sh_residual(sh_grid, mean_sh, grid_size)  # [N, gs, gs, 48]

        print(f"\n[BAKE] Mean SH stats: mean={mean_sh.mean():.6f}, std={mean_sh.std():.6f}")
        print(f"[BAKE] Residual tex (48D SH): mean={residual_tex.mean():.6f}, std={residual_tex.std():.6f}, "
              f"min={residual_tex.min():.6f}, max={residual_tex.max():.6f}")

        mean_sh_per_channel = mean_sh.view(N, 3, 16)
        mean_sh_ply = mean_sh_per_channel.permute(0, 2, 1)
        gaussians._features_dc = nn.Parameter(mean_sh_ply[:, 0:1, :].contiguous())
        gaussians._features_rest = nn.Parameter(mean_sh_ply[:, 1:, :].contiguous())

        ply_path = os.path.join(output_dir, "baked.ply")
        gaussians.save_ply(ply_path)
        print(f"\n[BAKE] Saved baked.ply → {ply_path}")

        tex_path = os.path.join(output_dir, "residual_textures.pt")
        torch.save(residual_tex.half().cpu(), tex_path)
        print(f"[BAKE] Saved residual_textures.pt → {tex_path} "
              f"({residual_tex.half().nelement() * 2 / 1024 / 1024:.1f} MB)")

        meta = {
            "texture_mode": "shared",
            "grid_size": grid_size,
            "residual_dim": 48,
            "uv_extent": uv_extent,
            "supersample": ss,
            "num_gaussians": N,
            "iteration": iteration,
            "method": args.method,
            "kernel": getattr(args, 'kernel', 'gaussian'),
            "sh_degree": 3,
        }

    # Save metadata
    meta_path = os.path.join(output_dir, "bake_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[BAKE] Saved bake_meta.json → {meta_path}")

    print(f"\n[BAKE] Done! Baked model saved to {output_dir}")


if __name__ == "__main__":
    main()
