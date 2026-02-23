#!/usr/bin/env python3
"""
Debug spatial color: known world-XYZ→RGB function through the full bake chain.

Tests the FULL coordinate pipeline:
  UV grid → world XYZ (center + u*sx*R0 + v*sy*R1) → color(xyz) → texture → bake_render

Color function operates in WORLD XYZ (same space the hash encoding queries):
  R = (x - bbox_min.x) / (bbox_max.x - bbox_min.x)
  G = (y - bbox_min.y) / (bbox_max.y - bbox_min.y)
  B = (z - bbox_min.z) / (bbox_max.z - bbox_min.z)

Outputs:
  ground_truth_256.png  — color(xyz) at dense 256×256 UV grid → world XYZ → RGB
  baked_NxN_nearest.png — color(xyz) at NxN UV grid, nearest upscale
  baked_NxN_bilinear.png— color(xyz) at NxN UV grid, bilinear upscale
  rendered_baked.png    — NxN texture rendered through bake_render
"""

import os, sys, pickle, glob, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8


UV_EXTENT = 4.0


def quat_to_rotcols(quats):
    """Quaternion [N, 4] (w,x,y,z) → R_col0 [N, 3], R_col1 [N, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


def uv_grid_to_world_xyz(gaussians, grid_size, uv_extent=UV_EXTENT):
    """Build UV grid, compute world XYZ for single Gaussian. Returns [gs, gs, 3]."""
    centers = gaussians.get_xyz       # [1, 3]
    quats = gaussians.get_rotation    # [1, 4]
    scales = gaussians.get_scaling    # [1, 2]
    R0, R1 = quat_to_rotcols(quats)  # [1, 3] each

    step = 2.0 * uv_extent / grid_size
    coords = torch.arange(grid_size, dtype=torch.float32, device='cuda')
    uv_1d = (coords + 0.5) * step - uv_extent
    uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    u_flat = uu.reshape(-1)  # [gs^2]
    v_flat = vv.reshape(-1)

    c = centers[0]      # [3]
    sx = scales[0, 0]   # scalar
    sy = scales[0, 1]
    r0 = R0[0]          # [3]
    r1 = R1[0]

    # xyz = center + u*sx*R0 + v*sy*R1  (same formula as bake_sh_res.py)
    xyz = (c.unsqueeze(0)
           + u_flat.unsqueeze(-1) * (sx * r0).unsqueeze(0)
           + v_flat.unsqueeze(-1) * (sy * r1).unsqueeze(0))  # [gs^2, 3]

    return xyz.reshape(grid_size, grid_size, 3)


def spatial_color_xyz(xyz, bbox_min, bbox_max):
    """Map world XYZ → RGB. Each channel = normalized position along that axis."""
    span = bbox_max - bbox_min
    span = span.clamp(min=1e-6)
    t = (xyz - bbox_min) / span  # [*, 3] in [0, 1]
    return t.clamp(0, 1)


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
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel
    return args, cfg, gaussians, scene, pipe, iteration


def save(img, path):
    """Save [H,W,3] or [3,H,W] float tensor as PNG."""
    if img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    save_img_u8(img.clamp(0, 1).detach().cpu().numpy(), path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scale_mult", type=float, default=3.0)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=8)
    a = parser.parse_args()

    out_dir = os.path.join(a.model_path, "comparisons", "spatial_color")
    os.makedirs(out_dir, exist_ok=True)

    args, cfg, gaussians, scene, pipe, iteration = load_model(a.model_path, a.iteration)
    N = len(gaussians.get_xyz)

    # Pick Gaussian near center with decent size
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    centroid = centers.mean(dim=0)
    dists = (centers - centroid).norm(dim=1)
    area = scales[:, 0] * scales[:, 1]
    score = -dists + 0.1 * area.clamp(min=1e-8).log()
    idx = score.argmax().item()

    # Isolate single Gaussian
    for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_appearance_level']:
        tensor = getattr(gaussians, attr)
        setattr(gaussians, attr, torch.nn.Parameter(tensor[idx:idx+1].clone()))
    if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        gaussians._shape = torch.nn.Parameter(gaussians._shape[idx:idx+1].clone())
    gaussians._scaling.data += math.log(a.scale_mult)
    gaussians._opacity.data.fill_(10.0)
    gaussians._features_dc.data.zero_()
    gaussians._features_rest.data.zero_()

    print(f"Gaussian {idx}/{N}")
    print(f"  center: {gaussians.get_xyz[0].tolist()}")
    print(f"  scales: {gaussians.get_scaling[0].tolist()}")

    # --- Compute world XYZ at dense grid to get bounding box ---
    print(f"\nComputing world XYZ at 256×256 UV grid...")
    xyz_dense = uv_grid_to_world_xyz(gaussians, 256)  # [256, 256, 3]
    bbox_min = xyz_dense.reshape(-1, 3).min(dim=0).values
    bbox_max = xyz_dense.reshape(-1, 3).max(dim=0).values
    print(f"  World XYZ bbox: min={bbox_min.tolist()}")
    print(f"                  max={bbox_max.tolist()}")
    print(f"                  span={( bbox_max - bbox_min).tolist()}")

    # --- 1. Ground truth: color(xyz) at dense 256×256 UV grid ---
    print(f"\n[1] Ground truth: color(xyz) at 256×256 UV → world XYZ → RGB")
    gt_rgb = spatial_color_xyz(xyz_dense, bbox_min, bbox_max)  # [256, 256, 3]
    # Transpose to match CUDA v*gs+u convention for display
    gt_vis = gt_rgb.transpose(0, 1).contiguous()
    save(gt_vis, os.path.join(out_dir, "ground_truth_256.png"))

    # --- 2. Bake: color(xyz) at grid_size UV grid ---
    gs = a.grid_size
    print(f"[2] Baked: color(xyz) at {gs}×{gs} UV → world XYZ → RGB")
    xyz_baked = uv_grid_to_world_xyz(gaussians, gs)  # [gs, gs, 3]
    baked_rgb = spatial_color_xyz(xyz_baked, bbox_min, bbox_max)  # [gs, gs, 3]
    # Transpose for CUDA indexing
    baked = baked_rgb.transpose(0, 1).contiguous()  # [gs, gs, 3]

    # Upscale for visual comparison
    baked_chw = baked.permute(2, 0, 1).unsqueeze(0)  # [1, 3, gs, gs]
    baked_nearest = torch.nn.functional.interpolate(baked_chw, size=(256, 256), mode='nearest')[0]
    save(baked_nearest, os.path.join(out_dir, f"baked_{gs}x{gs}_nearest.png"))
    baked_bilinear = torch.nn.functional.interpolate(baked_chw, size=(256, 256), mode='bilinear', align_corners=False)[0]
    save(baked_bilinear, os.path.join(out_dir, f"baked_{gs}x{gs}_bilinear.png"))

    print(f"  Texture range: [{baked.min():.3f}, {baked.max():.3f}]")

    # Verify a few texels
    print(f"\n  Sample texels (after transpose, [row, col, :]):")
    print(f"    [0,0]: {baked[0, 0].tolist()}")
    print(f"    [0,{gs-1}]: {baked[0, gs-1].tolist()}")
    print(f"    [{gs-1},0]: {baked[gs-1, 0].tolist()}")
    print(f"    [{gs-1},{gs-1}]: {baked[gs-1, gs-1].tolist()}")

    # --- 3. Render through bake_render ---
    print(f"\n[3] Rendering with bake_render ({gs}×{gs} texture)...")
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    cam = scene.getTestCameras()[a.cam_idx]
    bg = torch.zeros(3, device="cuda")

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

    # SH=0 → base color = 0.5 per channel. Residual = desired_color - 0.5.
    shs = torch.zeros_like(gaussians.get_features)
    residual = (baked.unsqueeze(0).cuda() - 0.5).half()  # [1, gs, gs, 3]
    residual_flat = residual.view(1, -1).contiguous()

    print(f"  Residual range: [{residual.min():.3f}, {residual.max():.3f}]")

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
    img_rendered = color.clamp(0, 1)
    save(img_rendered, os.path.join(out_dir, "rendered_baked.png"))

    mask = img_rendered.sum(0) > 0.01
    n_pixels = mask.sum().item()
    print(f"  Rendered {n_pixels} non-background pixels")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Images saved to: {out_dir}")
    print(f"  ground_truth_256.png       = color(xyz) at dense 256×256 UV→XYZ")
    print(f"  baked_{gs}x{gs}_nearest.png   = color(xyz) at {gs}×{gs} UV→XYZ, nearest")
    print(f"  baked_{gs}x{gs}_bilinear.png  = color(xyz) at {gs}×{gs} UV→XYZ, bilinear")
    print(f"  rendered_baked.png         = bake_render with {gs}×{gs} texture")
    print(f"")
    print(f"Color function: R=x_norm, G=y_norm, B=z_norm (world XYZ)")
    print(f"This tests the full chain: UV grid → quat_to_rotcols → world XYZ → color")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
