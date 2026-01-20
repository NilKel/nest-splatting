#!/usr/bin/env python
"""
Extract raw CAT mode features (before MLP) for a specific test view.
Outputs: gaussian_pca.png, hash_pca.png, viewdir_pca.png
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from argparse import ArgumentParser

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from train import merge_cfg_to_args
from utils.render_utils import save_img_u8
from utils.point_utils import cam2rays


def pca_to_rgb(features_flat, alpha_flat, H, W):
    """Apply PCA to reduce features to 3 channels and normalize to RGB."""
    C = features_flat.shape[1]
    if C < 3:
        # Pad to 3 channels
        padded = np.zeros((features_flat.shape[0], 3))
        padded[:, :C] = features_flat
        features_flat = padded

    # PCA to 3 components
    mean = features_flat.mean(axis=0)
    centered = features_flat - mean
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:3]
    components = eigenvectors[:, idx]
    projected = centered @ components  # (H*W, 3)

    # Normalize each component to [0, 1]
    for i in range(3):
        p_min, p_max = projected[:, i].min(), projected[:, i].max()
        if p_max - p_min > 1e-6:
            projected[:, i] = (projected[:, i] - p_min) / (p_max - p_min)

    # Apply alpha mask
    projected = projected * alpha_flat[:, None]

    # Reshape to image
    return projected.reshape(H, W, 3)


def main():
    parser = ArgumentParser(description="Extract CAT mode raw features")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--view_idx", default=50, type=int, help="Test view index to render")
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (default: model_path/feature_viz)")
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "beta", "flex"])

    # Parse command line first to get output_dir before get_combined_args overwrites
    cmdline_args = parser.parse_args()
    output_dir_cmdline = cmdline_args.output_dir

    args = get_combined_args(parser)

    iteration = args.iteration
    yaml_file = args.yaml

    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    # Load model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(args.model_path, iteration)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    gaussians.XYZ_TYPE = "UV"

    ingp_model.set_active_levels(iteration)

    # Get test camera
    test_cams = scene.getTestCameras()
    if args.view_idx >= len(test_cams):
        print(f"View index {args.view_idx} out of range (max {len(test_cams)-1})")
        return

    cam = test_cams[args.view_idx]
    print(f"Rendering view {args.view_idx}: {cam.image_name}")

    # Output directory
    output_dir = output_dir_cmdline if output_dir_cmdline else os.path.join(args.model_path, "feature_viz")
    os.makedirs(output_dir, exist_ok=True)

    # Get feature dimensions from ingp
    hybrid_levels = ingp_model.hybrid_levels
    level_dim = ingp_model.level_dim
    gaussian_dim = hybrid_levels * level_dim  # e.g., 5*4 = 20D
    total_levels = ingp_model.levels
    hash_dim = (total_levels - hybrid_levels) * level_dim  # e.g., 1*4 = 4D

    print(f"Feature split: Gaussian={gaussian_dim}D, Hash={hash_dim}D")

    with torch.no_grad():
        # Standard render to get alpha mask
        render_pkg = render(cam, gaussians, pipe, background, ingp=ingp_model,
                          beta=beta, iteration=iteration, cfg=cfg_model)
        rend_alpha = render_pkg["rend_alpha"]

        # Get the rasterized features (before MLP) by rendering with return_raw_features=True
        render_pkg_features = render(cam, gaussians, pipe, background, ingp=ingp_model,
                                     beta=beta, iteration=iteration, cfg=cfg_model,
                                     return_raw_features=True)

        # Get view directions
        H, W = cam.image_height, cam.image_width
        rays_d, rays_o = cam2rays(cam)
        ray_unit = F.normalize(rays_d, dim=-1).cpu().numpy()  # (H*W, 3)

        alpha_flat = rend_alpha.view(-1).cpu().numpy()

        # Check if raw_features is in the return dict
        if "raw_features" in render_pkg_features:
            raw_features = render_pkg_features["raw_features"]  # (C, H, W)
            C, _, _ = raw_features.shape
            print(f"Raw features shape: {raw_features.shape}")

            feat_flat = raw_features.view(C, -1).permute(1, 0).cpu().numpy()  # (H*W, C)

            # Split into Gaussian and Hash parts
            gaussian_feat = feat_flat[:, :gaussian_dim]  # (H*W, 20)
            hash_feat = feat_flat[:, gaussian_dim:]  # (H*W, 4)

            # 1. Gaussian features PCA
            gaussian_rgb = pca_to_rgb(gaussian_feat, alpha_flat, H, W)
            save_img_u8(gaussian_rgb, os.path.join(output_dir, f"view{args.view_idx}_gaussian_pca.png"))
            print(f"  - view{args.view_idx}_gaussian_pca.png (Gaussian features, {gaussian_dim}D)")

            # 2. Hash features PCA
            hash_rgb = pca_to_rgb(hash_feat, alpha_flat, H, W)
            save_img_u8(hash_rgb, os.path.join(output_dir, f"view{args.view_idx}_hash_pca.png"))
            print(f"  - view{args.view_idx}_hash_pca.png (Hash features, {hash_dim}D)")

            # 3. View direction visualization (already 3D, just normalize and mask)
            viewdir_rgb = (ray_unit + 1.0) / 2.0  # Map [-1,1] to [0,1]
            viewdir_rgb = viewdir_rgb * alpha_flat[:, None]
            viewdir_rgb = viewdir_rgb.reshape(H, W, 3)
            save_img_u8(viewdir_rgb, os.path.join(output_dir, f"view{args.view_idx}_viewdir.png"))
            print(f"  - view{args.view_idx}_viewdir.png (View directions)")

        else:
            print("Warning: raw_features not available. Need to add return_raw_features support to render().")

        print(f"\nSaved outputs to {output_dir}:")


if __name__ == "__main__":
    main()
