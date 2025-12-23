#!/usr/bin/env python3
"""
Render Gaussian intersection count heatmaps for an already-trained scene.

Usage:
    # Render all frames for a single experiment
    python scripts/render_intersection_heatmap.py -m outputs/nerf_synthetic/chair/baseline/no2f --yaml tiny

    # Render only frame 5
    python scripts/render_intersection_heatmap.py -m outputs/nerf_synthetic/chair/baseline/no2f --yaml tiny --frame 5

    # Render only test frames
    python scripts/render_intersection_heatmap.py -m outputs/nerf_synthetic/chair/baseline/no2f --yaml tiny --skip_train

This will render intersection heatmaps and save them to final_test_intersection/
and final_train_intersection/ directories.
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

        # Extract method
        method_match = re.search(r'Method:\s*(\w+)', content)
        if method_match:
            method = method_match.group(1)

        # Extract hybrid levels
        hybrid_match = re.search(r'Hybrid Levels:\s*(\d+)', content)
        if hybrid_match:
            hybrid_levels = int(hybrid_match.group(1))

    return method, hybrid_levels


def render_intersection_heatmaps(scene, gaussians, pipe, background, ingp, beta, iteration,
                                  cfg_model, cameras, output_dir, camera_type="test", frame_idx=None):
    """Render intersection heatmaps for a set of cameras.

    Args:
        frame_idx: If specified, only render this specific frame index. Otherwise render all.
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(cameras) == 0:
        print(f"[{camera_type.upper()}] No cameras available, skipping.")
        return

    # Filter to specific frame if requested
    if frame_idx is not None:
        if frame_idx >= len(cameras):
            print(f"[{camera_type.upper()}] Frame {frame_idx} out of range (max: {len(cameras)-1}), skipping.")
            return
        cameras_to_render = [(frame_idx, cameras[frame_idx])]
        print(f"\n[{camera_type.upper()}] Rendering frame {frame_idx} only...")
    else:
        cameras_to_render = list(enumerate(cameras))
        print(f"\n[{camera_type.upper()}] Rendering intersection heatmaps for {len(cameras)} cameras...")

    with torch.no_grad():
        for idx, viewpoint in tqdm(cameras_to_render, desc=f"Rendering {camera_type}"):
            render_pkg = render(viewpoint, gaussians, pipe, background,
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)

            # Get gaussian intersection count per pixel
            gaussian_num = render_pkg['gaussian_num']  # (1, H, W)

            # Create heatmap with legend (fixed max_display=200 for consistency)
            intersection_heatmap, min_count, max_count = create_intersection_heatmap(gaussian_num, max_display=200)

            # Create histogram showing distribution
            histogram_img, stats = create_intersection_histogram(gaussian_num, max_display=200)

            # Save heatmap and histogram
            save_img_u8(intersection_heatmap, os.path.join(output_dir, f"{idx:03d}_intersection.png"))
            save_img_u8(histogram_img, os.path.join(output_dir, f"{idx:03d}_histogram.png"))

    print(f"[{camera_type.upper()}] Intersection heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render Gaussian intersection count heatmaps for trained scenes")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to load (default: latest)")
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip rendering train camera heatmaps")
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip rendering test camera heatmaps")
    parser.add_argument("--frame", type=int, default=None,
                       help="Render only this specific frame index (default: render all)")
    parser.add_argument("--yaml", type=str, default="tiny",
                       help="YAML config file name (must match training)")
    parser.add_argument("--method", type=str, default=None,
                       choices=["baseline", "cat", "adaptive", "adaptive_add", "adaptive_cat",
                               "diffuse", "specular", "diffuse_ngp", "diffuse_offset",
                               "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid"],
                       help="Rendering method (auto-detected from training_log.txt if not specified)")
    parser.add_argument("--hybrid_levels", type=int, default=None,
                       help="Number of hybrid levels (auto-detected from training_log.txt if not specified)")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    # Preserve frame/skip args that may not be in saved config
    # Re-parse just to get these specific args
    temp_args, _ = parser.parse_known_args()
    args.frame = temp_args.frame
    args.skip_train = temp_args.skip_train
    args.skip_test = temp_args.skip_test

    exp_path = args.model_path
    iteration = args.iteration
    yaml_file = args.yaml

    # Auto-detect method and hybrid_levels from training_log.txt if not specified
    log_method, log_hybrid_levels = parse_training_log(exp_path)

    # Use CLI args if specified, otherwise use auto-detected values, otherwise use defaults
    if temp_args.method is not None:
        args.method = temp_args.method
    elif log_method is not None:
        args.method = log_method
        print(f"[AUTO-DETECT] Method: {log_method} (from training_log.txt)")
    else:
        args.method = "baseline"
        print(f"[AUTO-DETECT] Method: baseline (default, no training_log.txt found)")

    if temp_args.hybrid_levels is not None:
        args.hybrid_levels = temp_args.hybrid_levels
    elif log_hybrid_levels is not None:
        args.hybrid_levels = log_hybrid_levels
        print(f"[AUTO-DETECT] Hybrid levels: {log_hybrid_levels} (from training_log.txt)")
    else:
        args.hybrid_levels = 3
        if args.method == "cat":
            print(f"[AUTO-DETECT] Hybrid levels: 3 (default for cat mode)")

    # Resolve iteration=-1 to actual max iteration
    if iteration == -1:
        point_cloud_path = os.path.join(exp_path, "point_cloud")
        if os.path.exists(point_cloud_path):
            iteration = searchForMaxIteration(point_cloud_path)
        else:
            # Try to find from ngp checkpoint files
            ngp_files = [f for f in os.listdir(exp_path) if f.startswith('ngp_') and f.endswith('.pth')]
            if ngp_files:
                iterations = [int(f.replace('ngp_', '').replace('.pth', '')) for f in ngp_files]
                iteration = max(iterations)
            else:
                raise FileNotFoundError(f"Could not find any checkpoints in {exp_path}")

    print(f"\n{'='*60}")
    print(f"Rendering Intersection Heatmaps")
    print(f"{'='*60}")
    print(f"Model path: {exp_path}")
    print(f"Iteration: {iteration}")
    print(f"Config: {yaml_file}")
    print(f"Method: {args.method}")
    if args.method == "cat":
        print(f"Hybrid levels: {args.hybrid_levels}")
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

    # Set active levels
    active_levels = ingp_model.set_active_levels(iteration)

    # Determine output directories
    test_output_dir = os.path.join(scene.model_path, 'final_test_intersection')
    train_output_dir = os.path.join(scene.model_path, 'final_train_intersection')

    # Render test cameras
    if not args.skip_test and len(scene.getTestCameras()) > 0:
        render_intersection_heatmaps(
            scene, gaussians, pipe, background, ingp_model, beta, iteration,
            cfg_model, scene.getTestCameras(), test_output_dir, "test",
            frame_idx=args.frame
        )
    elif args.skip_test:
        print("[TEST] Skipped (--skip_test flag)")
    else:
        print("[TEST] No test cameras available")

    # Render train cameras
    if not args.skip_train:
        render_intersection_heatmaps(
            scene, gaussians, pipe, background, ingp_model, beta, iteration,
            cfg_model, scene.getTrainCameras(), train_output_dir, "train",
            frame_idx=args.frame
        )
    else:
        print("[TRAIN] Skipped (--skip_train flag)")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")
