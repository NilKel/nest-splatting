#!/usr/bin/env python3
"""
Render hash feature magnitude maps to diagnose grey haze.

Renders the L2 magnitude of hash features at each pixel, masked by alpha.
This shows whether hash features are actually zero or contain values.

Usage:
    python scripts/render_hash_magnitude.py -m /path/to/checkpoint --iteration 35000
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from scene.gaussian_model import GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8, convert_gray_to_cmap
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args
from gaussian_renderer import render
from utils.point_utils import depths_to_points


def render_hash_magnitude(camera, gaussians, ingp, pipe, background, beta, iteration, cfg_model):
    """Render hash feature magnitude map."""

    # Render to get depth and alpha
    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipe, background,
                           ingp=ingp, beta=beta, iteration=iteration,
                           cfg=cfg_model, decompose_mode='ngp_only')

    H, W = camera.image_height, camera.image_width
    render_alpha = render_pkg['rend_alpha']
    render_depth = render_pkg['depth_median']
    ngp_rgb = render_pkg.get('ngp_rgb', None)

    # Unproject depth to 3D points
    points_3d, rays_d, rays_o = depths_to_points(camera, render_depth.detach())

    # Query hashgrid at unprojected 3D points (just the hash, no per-Gaussian features)
    hash_features = ingp(points_3D=points_3d, with_xyz=False).float()  # (H*W, feat_dim)

    # Compute feature magnitude (L2 norm)
    feature_magnitude = torch.norm(hash_features, dim=-1)  # (H*W,)
    feature_magnitude = feature_magnitude.view(H, W)  # (H, W)

    # Apply alpha mask
    render_mask = (render_alpha > 0).squeeze()
    feature_magnitude_masked = feature_magnitude * render_mask

    # Compute statistics
    valid_magnitudes = feature_magnitude_masked[render_mask]
    if valid_magnitudes.numel() > 0:
        stats = {
            'mag_mean': valid_magnitudes.mean().item(),
            'mag_max': valid_magnitudes.max().item(),
            'mag_min': valid_magnitudes.min().item(),
            'mag_std': valid_magnitudes.std().item(),
            'near_zero_pct': (valid_magnitudes < 0.01).float().mean().item() * 100,
        }
    else:
        stats = {'mag_mean': 0, 'mag_max': 0, 'mag_min': 0, 'mag_std': 0, 'near_zero_pct': 0}

    return {
        'feature_magnitude': feature_magnitude_masked,
        'ngp_rgb': ngp_rgb,
        'alpha': render_alpha,
        'stats': stats,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Render hash feature magnitude maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--yaml", type=str, default="tiny")
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--hybrid_levels", type=int, default=3)
    parser.add_argument("--kernel", type=str, default="gaussian")
    parser.add_argument("--max_images", type=int, default=10, help="Max images to render per split")
    args = get_combined_args(parser)

    exp_path = args.model_path
    iteration = args.iteration

    # Load args.json to get training configuration
    args_json_path = os.path.join(exp_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            saved_args = json.load(f)
        args.yaml = saved_args.get("yaml", args.yaml)
        args.method = saved_args.get("method", args.method)
        args.hybrid_levels = saved_args.get("hybrid_levels", args.hybrid_levels)
        args.kernel = saved_args.get("kernel", args.kernel)
        print(f"[INFO] Loaded config: method={args.method}, hybrid_levels={args.hybrid_levels}")

    # Load model config
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(exp_path, iteration)
    ingp_model.set_active_levels(iteration)
    ingp_model.eval()

    # Load scene
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.XYZ_TYPE = "UV"
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta

    if args.kernel == "beta":
        gaussians.kernel_type = "beta"
    elif args.kernel == "flex":
        gaussians.kernel_type = "flex"
    elif args.kernel == "general":
        gaussians.kernel_type = "general"

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Output directory
    output_dir = os.path.join(exp_path, f"hash_magnitude_{iteration}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Output: {output_dir}")

    all_stats = []

    # Render test cameras
    if not args.skip_test and len(scene.getTestCameras()) > 0:
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        cameras = scene.getTestCameras()[:args.max_images]
        print(f"\n[INFO] Rendering {len(cameras)} test cameras...")

        with torch.no_grad():
            for idx, cam in enumerate(tqdm(cameras)):
                results = render_hash_magnitude(cam, gaussians, ingp_model, pipe, background, beta, iteration, cfg_model)

                # Save magnitude heatmap (normalized per-image)
                mag = results['feature_magnitude'].cpu().numpy()
                mag_max = mag.max() + 1e-8
                mag_normalized = (mag / mag_max * 255).astype(np.uint8)
                # Use matplotlib colormap
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('turbo')
                mag_colored = (cmap(mag_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
                save_img_u8(mag_colored / 255.0, os.path.join(test_dir, f"{idx:03d}_magnitude.png"))

                # Save ngp_only render for comparison
                if results['ngp_rgb'] is not None:
                    ngp = torch.clamp(results['ngp_rgb'], 0, 1)
                    save_img_u8(ngp.permute(1, 2, 0).cpu().numpy(), os.path.join(test_dir, f"{idx:03d}_ngp_rgb.png"))

                all_stats.append(results['stats'])

                if idx < 3:
                    s = results['stats']
                    print(f"  [{idx}] mag: mean={s['mag_mean']:.4f}, max={s['mag_max']:.4f}, near_zero={s['near_zero_pct']:.1f}%")

    # Summary
    if all_stats:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print('='*50)
        avg_mean = np.mean([s['mag_mean'] for s in all_stats])
        avg_max = np.mean([s['mag_max'] for s in all_stats])
        avg_near_zero = np.mean([s['near_zero_pct'] for s in all_stats])
        print(f"Avg magnitude (mean): {avg_mean:.4f}")
        print(f"Avg magnitude (max):  {avg_max:.4f}")
        print(f"Avg % near zero (<0.01): {avg_near_zero:.1f}%")

        if avg_near_zero > 80:
            print(f"\n[RESULT] Hash features ARE mostly zero ({avg_near_zero:.1f}% < 0.01)")
            print(f"         Grey haze must come from MLP bias mapping zero → grey")
        else:
            print(f"\n[RESULT] Hash features are NOT zero ({avg_near_zero:.1f}% < 0.01)")
            print(f"         Grey comes from actual hash feature values")

    print(f"\n[DONE] Saved to {output_dir}")
