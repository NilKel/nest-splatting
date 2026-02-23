#!/usr/bin/env python3
"""
Single-Gaussian comparison: hash+MLP per-pixel vs baked 8x8 texture.

Picks one Gaussian, enlarges it, zeros SH, renders with:
  1. Training renderer (hash+MLP evaluated per intersection) → mlp.png
  2. Bake hash+MLP into 8x8 texture, render with bake_render → baked.png

Usage:
    python scripts/debug_single_gaussian.py \
        --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
"""

import os, sys, pickle, glob, math, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.render_utils import save_img_u8


def load_model(model_path, iteration=-1):
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)
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
    return args, cfg, gaussians, ingp, scene, pipe, iteration


def isolate_single_gaussian(gaussians, idx, scale_mult=3.0):
    for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_appearance_level']:
        tensor = getattr(gaussians, attr)
        setattr(gaussians, attr, torch.nn.Parameter(tensor[idx:idx+1].clone()))
    if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        gaussians._shape = torch.nn.Parameter(gaussians._shape[idx:idx+1].clone())
    gaussians._scaling.data += math.log(scale_mult)
    gaussians._opacity.data.fill_(10.0)
    gaussians._features_dc.data.zero_()
    gaussians._features_rest.data.zero_()
    print(f"  center: {gaussians.get_xyz[0].tolist()}")
    print(f"  scales (after {scale_mult}x): {gaussians.get_scaling[0].tolist()}")


def quat_to_rotcols(quats):
    """Quaternion [N, 4] (w,x,y,z) -> R_col0 [N, 3], R_col1 [N, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    R0 = torch.stack([r00, r10, r20], dim=-1)
    R1 = torch.stack([r01, r11, r21], dim=-1)
    return R0, R1


def bake_single_gaussian(gaussians, ingp, grid_size=8, uv_extent=4.0):
    """Bake the hash+MLP into an [1, gs, gs, 3] texture for a single Gaussian."""
    centers = gaussians.get_xyz          # [1, 3]
    quats = gaussians.get_rotation       # [1, 4]
    scales = gaussians.get_scaling       # [1, 2]
    R0, R1 = quat_to_rotcols(quats)     # [1, 3] each

    step = 2.0 * uv_extent / grid_size
    coords = torch.arange(grid_size, dtype=torch.float32, device='cuda')
    uv_1d = (coords + 0.5) * step - uv_extent
    uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # [gs^2, 2]

    c = centers[0:1]       # [1, 3]
    sx = scales[0:1, 0:1]  # [1, 1]
    sy = scales[0:1, 1:2]
    r0 = R0[0:1]           # [1, 3]
    r1 = R1[0:1]

    u_vals = uv_grid[:, 0]
    v_vals = uv_grid[:, 1]

    xyz = (c.unsqueeze(1)
           + u_vals.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
           + v_vals.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))
    xyz_flat = xyz.reshape(-1, 3)  # [gs^2, 3]

    mlp = ingp.mlp_fused
    mlp.eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_input_padded = mlp[0].weight.shape[1]
    bias_col = hash_dim

    with torch.no_grad():
        hash_feat = ingp._encode_3D(xyz_flat)
        mlp_input = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device='cuda')
        mlp_input[:, :hash_dim] = hash_feat[:, :hash_dim]
        mlp_input[:, bias_col] = 1.0
        mlp_out = mlp(mlp_input)
        rgb_residual = mlp_out[:, :3]

    # [1, gs, gs, 3] with transpose to match CUDA v*gs+u indexing
    tex = rgb_residual.reshape(1, grid_size, grid_size, 3)
    tex = tex.transpose(1, 2).contiguous()

    print(f"  Baked texture stats: mean={tex.mean():.4f}, std={tex.std():.4f}, "
          f"min={tex.min():.4f}, max={tex.max():.4f}")
    return tex


def render_baked(cam, gaussians, cfg, residual_tex):
    """Render single Gaussian with bake_render using shared 8x8 texture."""
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    bg = torch.zeros(3, device="cuda")
    N = len(gaussians.get_xyz)

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(gaussians, 'kernel_type', 'gaussian'), 0)
    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
        prefiltered=False, debug=False, beta=cfg.surfel.tg_beta,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    # SH = zero (same as training render)
    shs = torch.zeros_like(gaussians.get_features)

    # Shared texture: [N, gs*gs*dim] FP16
    residual_flat = residual_tex.half().cuda().view(N, -1).contiguous()

    with torch.no_grad():
        color, radii, _ = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
            opacities=gaussians.get_opacity,
            shs=shs,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            shapes=shapes, kernel_type=kernel_type,
            residual_textures=residual_flat,
        )
    return color.clamp(0, 1)


def save(img, path):
    save_img_u8(img.permute(1, 2, 0).cpu().numpy(), path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scale_mult", type=float, default=3.0)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--cam_idx", type=int, default=0)
    a = parser.parse_args()

    out_dir = os.path.join(a.model_path, "comparisons", "single_gaussian")
    os.makedirs(out_dir, exist_ok=True)

    args, cfg, gaussians, ingp, scene, pipe, iteration = load_model(a.model_path, a.iteration)
    N = len(gaussians.get_xyz)

    # Pick a Gaussian near the center with decent size
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    centroid = centers.mean(dim=0)
    dists = (centers - centroid).norm(dim=1)
    area = scales[:, 0] * scales[:, 1]
    score = -dists + 0.1 * area.clamp(min=1e-8).log()
    idx = score.argmax().item()

    print(f"\nSelected Gaussian {idx}/{N}")
    isolate_single_gaussian(gaussians, idx, scale_mult=a.scale_mult)

    cam = scene.getTestCameras()[a.cam_idx]
    print(f"Camera: {cam.image_name} ({int(cam.image_width)}x{int(cam.image_height)})")

    # --- 1. Training renderer (hash+MLP per pixel, SH=0) ---
    print(f"\n[1] Training renderer (hash+MLP per pixel, SH=0)...")
    with torch.no_grad():
        result = render(cam, gaussians, pipe, torch.zeros(3, device="cuda"),
                        ingp=ingp, iteration=iteration,
                        cfg=cfg, beta=cfg.surfel.tg_beta, is_training=False)
    img_mlp = result["render"].clamp(0, 1)
    save(img_mlp, os.path.join(out_dir, "mlp.png"))
    print(f"    Saved mlp.png")

    # --- 2. Bake hash+MLP into texture ---
    print(f"\n[2] Baking hash+MLP into {a.grid_size}x{a.grid_size} texture...")
    residual_tex = bake_single_gaussian(gaussians, ingp, grid_size=a.grid_size)

    # --- 3. Render with bake_render ---
    print(f"\n[3] Bake_render (shared {a.grid_size}x{a.grid_size} texture, SH=0)...")
    img_baked = render_baked(cam, gaussians, cfg, residual_tex)
    save(img_baked, os.path.join(out_dir, "baked.png"))
    print(f"    Saved baked.png")

    # --- Compare ---
    diff = (img_mlp - img_baked.cpu()).abs()
    mask = (img_mlp.sum(0) > 0) | (img_baked.cpu().sum(0) > 0)
    if mask.sum() > 0:
        mse = (diff[:, mask] ** 2).mean().item()
    else:
        mse = 0.0
    psnr = -10 * math.log10(mse + 1e-10)

    print(f"\n{'='*60}")
    print(f"  mlp.png   = training renderer (hash+MLP per pixel)")
    print(f"  baked.png = bake_render ({a.grid_size}x{a.grid_size} texture)")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Max diff:  {diff.max():.4f}")
    print(f"{'='*60}")

    save((diff * 10).clamp(0, 1), os.path.join(out_dir, "diff_mlp_vs_baked_10x.png"))

    # Also save the texture itself upscaled for inspection
    tex_vis = residual_tex[0].float().cpu()  # [gs, gs, 3]
    tex_vis = tex_vis.permute(2, 0, 1).unsqueeze(0)  # [1, 3, gs, gs]
    tex_up = torch.nn.functional.interpolate(tex_vis, size=(256, 256), mode='nearest')[0]
    # Shift to visible range: residual is small, so map [-0.1, 0.1] → [0, 1]
    tex_up = (tex_up * 5 + 0.5).clamp(0, 1)
    save(tex_up, os.path.join(out_dir, "baked_texture_5x.png"))

    print(f"\nImages saved to: {out_dir}")


if __name__ == "__main__":
    main()
