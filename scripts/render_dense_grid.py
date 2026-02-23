#!/usr/bin/env python3
"""
Render using dense 3D RGB grid: SH base color + trilinear grid residual lookup.

Uses the diff_surfel_dense_grid_render CUDA submodule for fast inference.
The grid replaces the hash MLP entirely — trilinear interpolation at intersection xyz.

Usage:
    python scripts/render_dense_grid.py --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled
    python scripts/render_dense_grid.py --model_path ... --grid_path .../dense_rgb_grid.pt --render_frame 125
"""

import os
import sys
import json
import math
import time
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def load_training_config(model_path):
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


def render_dense_grid(viewpoint_camera, gaussians, background,
                      dense_grid=None, grid_resolution=0, grid_vmin=0.0, grid_vmax=0.0,
                      beta=0.0, kernel_type=0):
    """Render using diff_surfel_dense_grid_render: SH base + dense grid residual."""
    from diff_surfel_dense_grid_render import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        beta=beta,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D[:, :2], requires_grad=False)
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features

    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    # Flatten grid for CUDA: [R, R, R, 3] → [R*R*R*3] FP16
    grid_flat = None
    if dense_grid is not None:
        grid_flat = dense_grid.reshape(-1).contiguous()

    result = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        scales=scales,
        rotations=rotations,
        shapes=shapes,
        kernel_type=kernel_type,
        dense_grid=grid_flat,
        grid_resolution=grid_resolution,
        grid_vmin=grid_vmin,
        grid_vmax=grid_vmax,
    )

    rendered_image = result[0]
    return {"render": rendered_image}


def main():
    parser = ArgumentParser(description="Render using dense 3D RGB grid")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--grid_path", type=str, default=None,
                        help="Path to dense_rgb_grid.pt (default: model_path/baked_dense/dense_rgb_grid.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: model_path/baked_dense/renders)")
    parser.add_argument("--render_frame", type=int, default=-1,
                        help="Render single frame index (default: all test frames)")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_benchmark", type=int, default=100)
    render_args = parser.parse_args()

    # Load training config
    args = load_training_config(render_args.model_path)
    args.model_path = render_args.model_path
    args.eval = True

    config_yaml_path = os.path.join(render_args.model_path, "config.yaml")
    cfg_model = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    # Auto-detect iteration
    iteration = render_args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(render_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)

    # Load Gaussians
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)

    # Determine grid path
    baked_dir = os.path.join(render_args.model_path, "baked_dense")
    grid_path = render_args.grid_path or os.path.join(baked_dir, "dense_rgb_grid.pt")
    baked_ply = os.path.join(baked_dir, "baked.ply")

    # Load baked PLY
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    kernel_name = getattr(args, 'kernel', 'gaussian')
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = kernel_name
    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(kernel_name, 0)

    N = len(gaussians.get_xyz)
    print(f"[RENDER] Loaded {N:,} Gaussians from {baked_ply}")

    # Load dense grid
    meta_path = os.path.join(baked_dir, "dense_grid_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    grid_resolution = meta["grid_resolution"]
    voxel_range = meta["voxel_range"]
    grid_vmin, grid_vmax = voxel_range[0], voxel_range[1]

    dense_grid = torch.load(grid_path).cuda()  # [R, R, R, 3] FP16
    grid_mb = dense_grid.nelement() * 2 / 1024 / 1024
    print(f"[RENDER] Loaded dense grid: {list(dense_grid.shape)}, {grid_mb:.1f} MB")
    print(f"[RENDER] Grid resolution: {grid_resolution}³, range: [{grid_vmin}, {grid_vmax}]")

    # Load test cameras (Scene overwrites PLY, reload after)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()

    # Reload baked PLY after Scene init
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    print(f"[RENDER] Reloaded baked PLY: {N:,} Gaussians, {len(test_cameras)} test cameras")

    beta = cfg_model.surfel.tg_beta
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    output_dir = render_args.output_dir or os.path.join(baked_dir, "renders")
    os.makedirs(output_dir, exist_ok=True)

    # Determine which frames to render
    if render_args.render_frame >= 0:
        # Single frame mode: render SH-only, grid-only, and full
        frame_idx = render_args.render_frame
        if frame_idx >= len(test_cameras):
            print(f"[ERROR] Frame {frame_idx} out of range (0-{len(test_cameras)-1})")
            return
        cam = test_cameras[frame_idx]
        gt = cam.original_image[:3].cuda()

        with torch.no_grad():
            # Full: SH + grid
            result_full = render_dense_grid(
                cam, gaussians, bg_color,
                dense_grid=dense_grid, grid_resolution=grid_resolution,
                grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                beta=beta, kernel_type=kernel_type)
            img_full = result_full["render"]

            # SH only: grid zeroed
            result_sh = render_dense_grid(
                cam, gaussians, bg_color,
                dense_grid=None, grid_resolution=0,
                grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                beta=beta, kernel_type=kernel_type)
            img_sh = result_sh["render"]

            # Grid only: zero out SH, pass grid
            # Temporarily zero SH features
            orig_dc = gaussians._features_dc.data.clone()
            orig_rest = gaussians._features_rest.data.clone()
            gaussians._features_dc.data.zero_()
            gaussians._features_rest.data.zero_()

            result_grid = render_dense_grid(
                cam, gaussians, bg_color,
                dense_grid=dense_grid, grid_resolution=grid_resolution,
                grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                beta=beta, kernel_type=kernel_type)
            img_grid = result_grid["render"]

            # Restore SH
            gaussians._features_dc.data.copy_(orig_dc)
            gaussians._features_rest.data.copy_(orig_rest)

        # Save images
        prefix = f"{frame_idx}"
        for name, img in [("full", img_full), ("SH", img_sh), ("grid", img_grid)]:
            img_np = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            save_path = os.path.join(output_dir, f"{prefix}_{name}.png")
            save_img_u8(img_np, save_path)
            p = psnr(img, gt).mean().item()
            s = ssim(img, gt).item()
            print(f"  {prefix}_{name}: PSNR={p:.2f} dB, SSIM={s:.4f}")

        # Save GT too
        gt_np = gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        save_img_u8(gt_np, os.path.join(output_dir, f"{prefix}_gt.png"))

    else:
        # Full test set evaluation
        psnrs, l1s, ssims = [], [], []
        with torch.no_grad():
            for i, cam in enumerate(test_cameras):
                result = render_dense_grid(
                    cam, gaussians, bg_color,
                    dense_grid=dense_grid, grid_resolution=grid_resolution,
                    grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                    beta=beta, kernel_type=kernel_type)
                rendered = result["render"]
                gt = cam.original_image[:3].cuda()

                psnrs.append(psnr(rendered, gt).mean().item())
                l1s.append(l1_loss(rendered, gt).item())
                ssims.append(ssim(rendered, gt).item())

                img_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                save_img_u8(img_np, os.path.join(output_dir, f"{cam.image_name}.png"))

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(test_cameras)}")

        metrics = {
            "psnr": float(np.mean(psnrs)),
            "l1": float(np.mean(l1s)),
            "ssim": float(np.mean(ssims)),
        }

        # FPS benchmark
        with torch.no_grad():
            for i in range(render_args.num_warmup):
                cam = test_cameras[i % len(test_cameras)]
                _ = render_dense_grid(cam, gaussians, bg_color,
                                      dense_grid=dense_grid, grid_resolution=grid_resolution,
                                      grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                                      beta=beta, kernel_type=kernel_type)
            torch.cuda.synchronize()

            times = []
            for i in range(render_args.num_benchmark):
                cam = test_cameras[i % len(test_cameras)]
                torch.cuda.synchronize()
                t0 = time.time()
                _ = render_dense_grid(cam, gaussians, bg_color,
                                      dense_grid=dense_grid, grid_resolution=grid_resolution,
                                      grid_vmin=grid_vmin, grid_vmax=grid_vmax,
                                      beta=beta, kernel_type=kernel_type)
                torch.cuda.synchronize()
                times.append(time.time() - t0)

        metrics["fps"] = float(1.0 / np.mean(times))
        metrics["ms_per_frame"] = float(np.mean(times) * 1000)

        print(f"\n[RESULTS] PSNR: {metrics['psnr']:.2f} dB  |  SSIM: {metrics['ssim']:.4f}  |  "
              f"L1: {metrics['l1']:.4f}  |  FPS: {metrics['fps']:.1f}")

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[RENDER] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
