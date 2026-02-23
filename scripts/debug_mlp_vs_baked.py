#!/usr/bin/env python3
"""
Single-Gaussian: vivid hash+MLP output vs baked 8x8 texture.

Randomizes MLP weights so the hash+MLP produces strong spatially-varying RGB.
Then compares:
  1. Dense MLP evaluation (256x256 UV grid) — what the MLP actually outputs
  2. Baked 8x8 texture rendered through bake_render — what baking gives you

Usage:
    python scripts/debug_mlp_vs_baked.py \
        --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
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
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel
    return args, cfg, gaussians, ingp, scene, pipe, iteration


def quat_to_rotcols(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


def eval_mlp_on_grid(gaussians, ingp, grid_size, uv_extent=4.0):
    """Evaluate hash+MLP at a grid_size x grid_size UV grid. Returns [gs, gs, 3]."""
    centers = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling
    R0, R1 = quat_to_rotcols(quats)

    step = 2.0 * uv_extent / grid_size
    coords = torch.arange(grid_size, dtype=torch.float32, device='cuda')
    uv_1d = (coords + 0.5) * step - uv_extent
    uu, vv = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    u_vals = uu.reshape(-1)
    v_vals = vv.reshape(-1)

    c = centers[0:1]
    sx = scales[0:1, 0:1]
    sy = scales[0:1, 1:2]
    r0 = R0[0:1]
    r1 = R1[0:1]

    xyz = (c + u_vals.unsqueeze(-1) * (sx * r0)
             + v_vals.unsqueeze(-1) * (sy * r1))  # [gs^2, 3]

    mlp = ingp.mlp_fused
    hash_dim = ingp.mlp_fused_hash_dim
    pad_dim = mlp[0].weight.shape[1]

    with torch.no_grad():
        h = ingp._encode_3D(xyz)
        inp = torch.zeros(xyz.shape[0], pad_dim, device='cuda')
        inp[:, :hash_dim] = h[:, :hash_dim]
        inp[:, hash_dim] = 1.0  # bias column
        out = mlp(inp)[:, :3]  # [gs^2, 3] RGB residual

    # indexing='ij': uu varies along dim0 (rows), vv along dim1 (cols)
    # reshape to [gs, gs, 3] then transpose to match CUDA v*gs+u
    return out.reshape(grid_size, grid_size, 3).transpose(0, 1).contiguous()


def save(img, path):
    if img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    save_img_u8(img.cpu().numpy(), path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scale_mult", type=float, default=3.0)
    parser.add_argument("--cam_idx", type=int, default=0)
    a = parser.parse_args()

    out_dir = os.path.join(a.model_path, "comparisons", "mlp_vs_baked")
    os.makedirs(out_dir, exist_ok=True)

    args, cfg, gaussians, ingp, scene, pipe, iteration = load_model(a.model_path, a.iteration)
    N = len(gaussians.get_xyz)

    # Pick Gaussian near center
    centers = gaussians.get_xyz.detach()
    scales = gaussians.get_scaling.detach()
    centroid = centers.mean(dim=0)
    dists = (centers - centroid).norm(dim=1)
    area = scales[:, 0] * scales[:, 1]
    score = -dists + 0.1 * area.clamp(min=1e-8).log()
    idx = score.argmax().item()

    # Isolate
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

    # --- Randomize MLP to produce vivid spatial RGB ---
    print(f"\nRandomizing MLP weights (seed=42) for vivid output...")
    mlp = ingp.mlp_fused
    torch.manual_seed(42)
    with torch.no_grad():
        for layer in mlp:
            if hasattr(layer, 'weight'):
                torch.nn.init.xavier_normal_(layer.weight, gain=2.0)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.zero_()

    # --- 1. Dense MLP evaluation (what the MLP actually produces) ---
    print(f"\n[1] Evaluating MLP on 256x256 UV grid...")
    dense = eval_mlp_on_grid(gaussians, ingp, 256)  # [256, 256, 3]
    # The residual has arbitrary range. Normalize to [0,1] for visualization.
    dmin, dmax = dense.min(), dense.max()
    print(f"    Raw range: [{dmin:.3f}, {dmax:.3f}]")
    dense_vis = (dense - dmin) / (dmax - dmin + 1e-8)
    save(dense_vis, os.path.join(out_dir, "mlp_256x256.png"))

    # --- 2. Bake at 8x8 ---
    print(f"[2] Evaluating MLP on 8x8 UV grid (= baked texture)...")
    baked_8 = eval_mlp_on_grid(gaussians, ingp, 8)  # [8, 8, 3]
    # Normalize with SAME range as dense
    baked_8_vis = (baked_8 - dmin) / (dmax - dmin + 1e-8)
    baked_8_up = torch.nn.functional.interpolate(
        baked_8_vis.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='nearest')[0]
    save(baked_8_up, os.path.join(out_dir, "baked_8x8_nearest.png"))
    baked_8_bilinear = torch.nn.functional.interpolate(
        baked_8_vis.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)[0]
    save(baked_8_bilinear, os.path.join(out_dir, "baked_8x8_bilinear.png"))

    # --- 3. Render with bake_render ---
    print(f"[3] Rendering with bake_render (8x8 shared texture)...")
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

    # Use raw (un-normalized) baked texture as residual
    shs = torch.zeros_like(gaussians.get_features)
    residual_flat = baked_8.unsqueeze(0).half().cuda().view(1, -1).contiguous()

    with torch.no_grad():
        color, _, _ = rasterizer(
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
    save(img_rendered, os.path.join(out_dir, "rendered_baked_8x8.png"))

    print(f"\n{'='*60}")
    print(f"Images saved to: {out_dir}")
    print(f"  mlp_256x256.png         = MLP evaluated at 256x256 (ground truth)")
    print(f"  baked_8x8_nearest.png   = MLP evaluated at 8x8, upscaled nearest")
    print(f"  baked_8x8_bilinear.png  = MLP evaluated at 8x8, upscaled bilinear")
    print(f"  rendered_baked_8x8.png  = bake_render output (8x8 texture on Gaussian)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
