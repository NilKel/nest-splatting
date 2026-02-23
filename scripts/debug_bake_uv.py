#!/usr/bin/env python3
"""
Debug script: Compare 3D_SH_res residual-only renders (SH zeroed).
Renders training (hash+MLP) vs baked texture with SH=0 to isolate residual.

Usage:
    python scripts/debug_bake_uv.py --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
"""

import os
import sys
import json
import pickle
import glob
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8


def quat_to_rotcols(quats):
    """Quaternion [N, 4] (w,x,y,z) -> R_col0 [N, 3], R_col1 [N, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z);     r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    R0 = torch.stack([r00, r10, r20], dim=-1)
    R1 = torch.stack([r01, r11, r21], dim=-1)
    return R0, R1


def load_model(model_path, iteration=-1):
    """Load 3D_SH_res model."""
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True

    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)
    assert args.method == "3D_SH_res", f"Expected 3D_SH_res, got {args.method}"

    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level']:
            setattr(gaussians, attr, getattr(gaussians, attr)[valid_mask])
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    N = len(gaussians.get_xyz)
    print(f"Loaded {N:,} Gaussians (pruned {n_dead:,}), iteration {iteration}")
    return args, cfg, gaussians, ingp, scene, pipe, iteration


def bake_residual_textures(gaussians, ingp, grid_size=8, uv_extent=4.0):
    """Bake hash+MLP residual into [N, gs, gs, 3] texture (same as bake_sh_res.py)."""
    centers = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling
    R0, R1 = quat_to_rotcols(quats)
    N = centers.shape[0]

    step = 2.0 * uv_extent / grid_size
    coords = torch.arange(grid_size, dtype=torch.float32, device='cuda')
    uv_1d = (coords + 0.5) * step - uv_extent
    uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    n_pts = grid_size * grid_size

    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]

    residual_hi = torch.zeros(N, n_pts, 3, device='cuda')
    u_vals = uv_grid[:, 0]
    v_vals = uv_grid[:, 1]

    chunk_size = max(1, min(N, 4 * (1024**3) // (3 * 4 * n_pts)))
    n_chunks = (N + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(c_start + chunk_size, N)
        n_batch = c_end - c_start
        c = centers[c_start:c_end]
        sx = scales[c_start:c_end, 0:1]
        sy = scales[c_start:c_end, 1:2]
        r0 = R0[c_start:c_end]
        r1 = R1[c_start:c_end]

        xyz = (c.unsqueeze(1)
               + u_vals.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
               + v_vals.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))
        xyz_flat = xyz.reshape(-1, 3)

        with torch.no_grad():
            hash_feat = ingp._encode_3D(xyz_flat)
            mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device='cuda')
            mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
            mlp_input[:, hash_dim] = 1.0  # bias
            mlp_out = mlp(mlp_input)
            residual_hi[c_start:c_end] = mlp_out[:, :3].reshape(n_batch, n_pts, 3)

        del xyz, xyz_flat, hash_feat, mlp_input, mlp_out

    residual_tex = residual_hi.view(N, grid_size, grid_size, 3)
    # Transpose [N, u, v, D] -> [N, v, u, D] to match CUDA v*gs+u indexing
    residual_tex = residual_tex.transpose(1, 2).contiguous()
    return residual_tex


def render_training(cam, gaussians, pipe, ingp, cfg, iteration, zero_sh=False):
    """Render with training renderer. If zero_sh, zeroes SH for residual-only."""
    from gaussian_renderer import render
    background = torch.zeros(3, device="cuda")
    beta = cfg.surfel.tg_beta

    if zero_sh:
        # Temporarily zero all SH coefficients
        orig_dc = gaussians._features_dc.data.clone()
        orig_rest = gaussians._features_rest.data.clone()
        gaussians._features_dc.data.zero_()
        gaussians._features_rest.data.zero_()

    with torch.no_grad():
        render_pkg = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                            iteration=iteration, cfg=cfg)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

    if zero_sh:
        gaussians._features_dc.data.copy_(orig_dc)
        gaussians._features_rest.data.copy_(orig_rest)

    return image


def render_baked(cam, gaussians, cfg, residual_tex, zero_sh=False):
    """Render with bake_render. If zero_sh, zeroes SH for residual-only."""
    from diff_surfel_bake_render import (
        GaussianRasterizationSettings as BakeSettings,
        GaussianRasterizer as BakeRasterizer,
    )

    background = torch.zeros(3, device="cuda")

    settings = BakeSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
        beta=cfg.surfel.tg_beta,
    )

    rasterizer = BakeRasterizer(raster_settings=settings)

    if zero_sh:
        shs = torch.zeros_like(gaussians.get_features)
    else:
        shs = gaussians.get_features

    # Kernel type
    shapes = None
    kernel_type = 0
    if hasattr(gaussians, 'kernel_type') and gaussians.kernel_type == "beta_scaled" and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape
        kernel_type = 4
    elif hasattr(gaussians, 'kernel_type') and gaussians.kernel_type == "beta" and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape
        kernel_type = 1

    # Flatten residual: [N, gs, gs, 3] -> flat FP16
    residual_flat = residual_tex.half().contiguous().view(-1)

    with torch.no_grad():
        color, radii, others = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=torch.zeros_like(gaussians.get_xyz),
            opacities=gaussians.get_opacity,
            shs=shs,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            shapes=shapes,
            kernel_type=kernel_type,
            residual_textures=residual_flat,
        )

    return torch.clamp(color, 0.0, 1.0)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--uv_extent", type=float, default=4.0)
    parser.add_argument("--num_cameras", type=int, default=3, help="Number of test cameras to render")
    eval_args = parser.parse_args()

    out_dir = os.path.join(eval_args.model_path, "comparisons")
    os.makedirs(out_dir, exist_ok=True)

    args, cfg, gaussians, ingp, scene, pipe, iteration = load_model(
        eval_args.model_path, eval_args.iteration)

    # Bake textures
    print(f"\nBaking residual textures ({eval_args.grid_size}x{eval_args.grid_size}, extent={eval_args.uv_extent})...")
    residual_tex = bake_residual_textures(gaussians, ingp,
                                          grid_size=eval_args.grid_size,
                                          uv_extent=eval_args.uv_extent)
    print(f"  Residual tex: shape={list(residual_tex.shape)}, "
          f"mean={residual_tex.mean():.6f}, std={residual_tex.std():.6f}")

    # Get test cameras
    cameras = scene.getTestCameras()
    n_cams = min(eval_args.num_cameras, len(cameras))

    gt_available = hasattr(cameras[0], 'original_image')

    for ci in range(n_cams):
        cam = cameras[ci]
        name = cam.image_name
        print(f"\n{'='*60}")
        print(f"Camera {ci}: {name} ({cam.image_width}x{cam.image_height})")
        print(f"{'='*60}")

        # 1. Full renders (SH + residual)
        img_train_full = render_training(cam, gaussians, pipe, ingp, cfg, iteration, zero_sh=False)
        img_baked_full = render_baked(cam, gaussians, cfg, residual_tex, zero_sh=False)

        # 2. Residual-only renders (SH = 0)
        img_train_res = render_training(cam, gaussians, pipe, ingp, cfg, iteration, zero_sh=True)
        img_baked_res = render_baked(cam, gaussians, cfg, residual_tex, zero_sh=True)

        # 3. SH-only render (no residual) — render baked with empty residual
        from diff_surfel_bake_render import (
            GaussianRasterizationSettings as BakeSettings,
            GaussianRasterizer as BakeRasterizer,
        )
        settings = BakeSettings(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
            prefiltered=False, debug=False, beta=cfg.surfel.tg_beta,
        )
        rasterizer = BakeRasterizer(raster_settings=settings)
        shapes = None; kernel_type = 0
        if hasattr(gaussians, 'kernel_type') and gaussians.kernel_type == "beta_scaled" and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
            shapes = gaussians.get_shape; kernel_type = 4
        with torch.no_grad():
            img_sh_only, _, _ = rasterizer(
                means3D=gaussians.get_xyz, means2D=torch.zeros_like(gaussians.get_xyz),
                opacities=gaussians.get_opacity, shs=gaussians.get_features,
                scales=gaussians.get_scaling, rotations=gaussians.get_rotation,
                shapes=shapes, kernel_type=kernel_type,
            )
        img_sh_only = torch.clamp(img_sh_only, 0.0, 1.0)

        # Compute metrics
        def psnr(a, b):
            mse = ((a - b) ** 2).mean().item()
            return -10 * math.log10(mse + 1e-10)

        diff_full = (img_train_full - img_baked_full).abs()
        diff_res = (img_train_res - img_baked_res).abs()

        print(f"\n  Full (SH+residual):")
        print(f"    train vs baked: PSNR={psnr(img_train_full, img_baked_full):.2f} dB, "
              f"mean_diff={diff_full.mean():.6f}, max_diff={diff_full.max():.4f}")

        print(f"\n  Residual-only (SH=0):")
        print(f"    train vs baked: PSNR={psnr(img_train_res, img_baked_res):.2f} dB, "
              f"mean_diff={diff_res.mean():.6f}, max_diff={diff_res.max():.4f}")

        if gt_available:
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            print(f"\n  vs GT:")
            print(f"    training:  PSNR={psnr(img_train_full, gt):.2f} dB")
            print(f"    baked:     PSNR={psnr(img_baked_full, gt):.2f} dB")
            print(f"    SH-only:   PSNR={psnr(img_sh_only, gt):.2f} dB")

        # Save all images
        prefix = f"{ci:02d}_{name}"

        def save(img, fname):
            path = os.path.join(out_dir, fname)
            save_img_u8(img.permute(1, 2, 0).cpu().numpy(), path)

        if gt_available:
            save(gt, f"{prefix}_gt.png")
        save(img_train_full, f"{prefix}_train_full.png")
        save(img_baked_full, f"{prefix}_baked_full.png")
        save(img_train_res, f"{prefix}_train_residual.png")
        save(img_baked_res, f"{prefix}_baked_residual.png")
        save(img_sh_only, f"{prefix}_sh_only.png")

        # Difference maps (amplified 10x)
        save((diff_full * 10).clamp(0, 1), f"{prefix}_diff_full_10x.png")
        save((diff_res * 10).clamp(0, 1), f"{prefix}_diff_residual_10x.png")

        print(f"\n  Saved to {out_dir}/{prefix}_*.png")

    print(f"\n{'='*60}")
    print(f"All images saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
