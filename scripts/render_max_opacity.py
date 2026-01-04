#!/usr/bin/env python3
"""
Render test views with all Gaussian opacities set to maximum (1.0).
Used to diagnose if low opacity is causing see-through intersection maps.
"""

import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.render_utils import save_img_u8, create_intersection_heatmap, create_intersection_histogram
from utils.system_utils import searchForMaxIteration
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args
import re


def parse_training_log(exp_path):
    """Parse training_log.txt to extract method and hybrid_levels."""
    log_path = os.path.join(exp_path, "training_log.txt")
    method = None
    hybrid_levels = None

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()

        method_match = re.search(r'Method:\s*(\w+)', content)
        if method_match:
            method = method_match.group(1)

        hybrid_match = re.search(r'Hybrid Levels:\s*(\d+)', content)
        if hybrid_match:
            hybrid_levels = int(hybrid_match.group(1))

    return method, hybrid_levels


if __name__ == "__main__":
    parser = ArgumentParser(description="Render with max opacity to diagnose intersection maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="tiny")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--hybrid_levels", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None,
                       help="Override beta for Gaussian sharpening (higher = more opaque throughout)")

    args = get_combined_args(parser)
    temp_args, _ = parser.parse_known_args()

    exp_path = args.model_path
    iteration = args.iteration
    yaml_file = args.yaml

    # Auto-detect method and hybrid_levels
    log_method, log_hybrid_levels = parse_training_log(exp_path)

    if temp_args.method is not None:
        args.method = temp_args.method
    elif log_method is not None:
        args.method = log_method
        print(f"[AUTO-DETECT] Method: {log_method}")
    else:
        args.method = "baseline"

    if temp_args.hybrid_levels is not None:
        args.hybrid_levels = temp_args.hybrid_levels
    elif log_hybrid_levels is not None:
        args.hybrid_levels = log_hybrid_levels
        print(f"[AUTO-DETECT] Hybrid levels: {log_hybrid_levels}")
    else:
        args.hybrid_levels = 3

    # Resolve iteration
    if iteration == -1:
        point_cloud_path = os.path.join(exp_path, "point_cloud")
        if os.path.exists(point_cloud_path):
            iteration = searchForMaxIteration(point_cloud_path)
        else:
            ngp_files = [f for f in os.listdir(exp_path) if f.startswith('ngp_') and f.endswith('.pth')]
            if ngp_files:
                iterations = [int(f.replace('ngp_', '').replace('.pth', '')) for f in ngp_files]
                iteration = max(iterations)

    print(f"\n{'='*60}")
    print(f"Rendering with MAX OPACITY (all Gaussians opacity=1.0)")
    print(f"{'='*60}")
    print(f"Model path: {exp_path}")
    print(f"Iteration: {iteration}")
    print(f"{'='*60}\n")

    # Load config
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(exp_path, iteration)

    # Load dataset and scene
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Setup background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Setup Gaussians
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta

    # Override beta if specified
    if temp_args.beta is not None:
        beta = temp_args.beta
        print(f"[OVERRIDE] Using beta = {beta} (higher = sharper Gaussians, more opaque throughout)")

    # Set active levels
    ingp_model.set_active_levels(iteration)

    # Print original opacity stats
    original_opacity = gaussians.get_opacity
    print(f"Original opacity stats:")
    print(f"  Total Gaussians: {original_opacity.shape[0]}")
    print(f"  Min: {original_opacity.min().item():.4f}")
    print(f"  Max: {original_opacity.max().item():.4f}")
    print(f"  Mean: {original_opacity.mean().item():.4f}")
    print(f"  Median: {original_opacity.median().item():.4f}")
    print(f"  base_opacity: {gaussians.base_opacity}")

    # Prune dead Gaussians (opacity <= 0.005, the MCMC threshold)
    MCMC_THRESHOLD = 0.005
    dead_mask = (gaussians.get_opacity <= MCMC_THRESHOLD).squeeze(-1)
    alive_mask = ~dead_mask
    n_dead = dead_mask.sum().item()
    n_alive = alive_mask.sum().item()
    print(f"\nPruning dead Gaussians (opacity <= {MCMC_THRESHOLD}):")
    print(f"  Dead: {n_dead}")
    print(f"  Alive: {n_alive}")

    # Keep only alive Gaussians
    with torch.no_grad():
        gaussians._xyz = torch.nn.Parameter(gaussians._xyz[alive_mask])
        gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[alive_mask])
        gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[alive_mask])
        gaussians._opacity = torch.nn.Parameter(gaussians._opacity[alive_mask])
        gaussians._scaling = torch.nn.Parameter(gaussians._scaling[alive_mask])
        gaussians._rotation = torch.nn.Parameter(gaussians._rotation[alive_mask])
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
            gaussians._gaussian_features = torch.nn.Parameter(gaussians._gaussian_features[alive_mask])
        if hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight is not None and gaussians._adaptive_cat_weight.numel() > 0:
            gaussians._adaptive_cat_weight = torch.nn.Parameter(gaussians._adaptive_cat_weight[alive_mask])
        if hasattr(gaussians, '_appearance_level') and gaussians._appearance_level is not None and gaussians._appearance_level.numel() > 0:
            gaussians._appearance_level = gaussians._appearance_level[alive_mask]

    print(f"  Remaining Gaussians: {gaussians._xyz.shape[0]}")

    # Override opacity to maximum (1.0)
    # Since get_opacity = sigmoid(_opacity) * (1 - base_opacity) + base_opacity
    # To get opacity = 1.0, we need sigmoid(_opacity) = (1.0 - base_opacity) / (1 - base_opacity) = 1.0
    # So _opacity needs to be very large (e.g., 10)
    with torch.no_grad():
        gaussians._opacity.data.fill_(10.0)  # sigmoid(10) ≈ 0.99995

    new_opacity = gaussians.get_opacity
    print(f"\nAfter pruning + max opacity:")
    print(f"  Min: {new_opacity.min().item():.6f}")
    print(f"  Max: {new_opacity.max().item():.6f}")
    print(f"  Mean: {new_opacity.mean().item():.6f}")

    # Output directories
    output_dir = os.path.join(exp_path, 'max_opacity_test')
    render_dir = os.path.join(output_dir, 'renders')
    intersection_dir = os.path.join(output_dir, 'intersections')
    histogram_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(intersection_dir, exist_ok=True)
    os.makedirs(histogram_dir, exist_ok=True)

    # Render test cameras
    cameras = scene.getTestCameras()
    print(f"\nRendering {len(cameras)} test views with max opacity...")

    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(cameras, desc="Rendering")):
            render_pkg = render(viewpoint, gaussians, pipe, background,
                              ingp=ingp_model, beta=beta, iteration=iteration, cfg=cfg_model)

            # Save rendered image
            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            save_img_u8(rendered_np, os.path.join(render_dir, f"{idx:03d}.png"))

            # Save intersection heatmap
            gaussian_num = render_pkg['gaussian_num']
            intersection_heatmap, min_count, max_count = create_intersection_heatmap(gaussian_num, max_display=200)
            save_img_u8(intersection_heatmap, os.path.join(intersection_dir, f"{idx:03d}_intersection.png"))

            # Save histogram
            histogram_img, stats = create_intersection_histogram(gaussian_num, max_display=200)
            save_img_u8(histogram_img, os.path.join(histogram_dir, f"{idx:03d}_histogram.png"))

            if idx == 0:
                print(f"\nFrame 0 intersection stats:")
                print(f"  Min: {gaussian_num.min().item():.0f}")
                print(f"  Max: {gaussian_num.max().item():.0f}")
                print(f"  Mean: {gaussian_num.mean().item():.1f}")

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*60}\n")
