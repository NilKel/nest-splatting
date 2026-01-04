#!/usr/bin/env python3
"""
Render flex beta heatmap for a trained model checkpoint.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.render_utils import save_img_u8, create_flex_beta_heatmap
from hash_encoder.config import Config
from hash_encoder.modules import INGP
from tqdm import tqdm

def render_flex_beta(args):
    # Load config
    cfg_model = Config(args.yaml)

    # Setup
    device = torch.device("cuda")
    background = torch.tensor([1, 1, 1] if cfg_model.model.white_background else [0, 0, 0],
                              dtype=torch.float32, device=device)

    # Initialize Gaussian model
    gaussians = GaussianModel(cfg_model.model.sh_degree)

    # Load checkpoint
    checkpoint_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(checkpoint_path):
        # Try to find the latest checkpoint
        pc_dir = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(pc_dir):
            iterations = [int(d.split("_")[1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")]
            if iterations:
                args.iteration = max(iterations)
                checkpoint_path = os.path.join(pc_dir, f"iteration_{args.iteration}", "point_cloud.ply")
                print(f"Using latest checkpoint at iteration {args.iteration}")

    print(f"Loading checkpoint: {checkpoint_path}")
    gaussians.load_ply(checkpoint_path, args=args)

    # Set kernel type to flex
    gaussians.kernel_type = "flex"

    # Check if flex_beta exists, if not initialize it
    if not hasattr(gaussians, '_flex_beta') or gaussians._flex_beta.numel() == 0:
        print("No flex_beta found in checkpoint, initializing from scratch")
        n_gaussians = len(gaussians.get_xyz)
        flex_beta_init = torch.full((n_gaussians, 1), 5.0, device="cuda").float()
        gaussians._flex_beta = nn.Parameter(flex_beta_init.requires_grad_(False))

    print(f"Loaded {len(gaussians.get_xyz)} Gaussians")
    print(f"Flex beta stats: mean={gaussians.get_flex_beta.mean().item():.3f}, "
          f"min={gaussians.get_flex_beta.min().item():.3f}, "
          f"max={gaussians.get_flex_beta.max().item():.3f}")

    # Load scene for cameras
    from arguments import ModelParams, PipelineParams
    from argparse import Namespace

    model_args = Namespace(
        source_path=args.source_path,
        model_path=args.model_path,
        images="images",
        resolution=1,
        white_background=cfg_model.model.white_background,
        data_device="cuda",
        eval=True,
        load_allres=False,
        sh_degree=cfg_model.model.sh_degree,
    )

    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)

    # Pipeline args
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0,
        debug=False,
    )

    # Create output directory
    output_dir = os.path.join(args.model_path, "flex_beta_heatmap")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to: {output_dir}")

    # Render test cameras
    cameras = scene.getTestCameras() if args.eval else scene.getTrainCameras()
    print(f"Rendering {len(cameras)} {'test' if args.eval else 'train'} views")

    for idx, viewpoint in enumerate(tqdm(cameras, desc="Rendering flex beta")):
        # Get per-Gaussian flex beta and expand to RGB
        flex_beta_vals = gaussians.get_flex_beta  # (N, 1)
        flex_beta_color = flex_beta_vals.expand(-1, 3)  # (N, 3)

        # Render with flex_beta as override_color
        render_pkg = render(viewpoint, gaussians, pipe, background,
                           ingp=None, beta=0.0, iteration=args.iteration, cfg=cfg_model,
                           override_color=flex_beta_color)

        flex_beta_map = render_pkg["render"][0:1]  # Take first channel
        render_alpha = render_pkg["rend_alpha"]  # (1, H, W)

        # Create heatmap
        flex_beta_heatmap, min_beta, max_beta = create_flex_beta_heatmap(
            flex_beta_map, render_alpha,
            min_display=args.min_display,
            max_display=args.max_display
        )

        # Save
        save_img_u8(flex_beta_heatmap, os.path.join(output_dir, f"{idx:03d}_flex_beta.png"))

    print(f"Done! Saved {len(cameras)} heatmaps to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render flex beta heatmap for trained model")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to trained model output directory")
    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="Path to source data directory")
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml",
                        help="Config YAML file")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Iteration to load (default: 30000)")
    parser.add_argument("--eval", action="store_true",
                        help="Render test cameras (default: train cameras)")
    parser.add_argument("--min_display", type=float, default=0.0,
                        help="Min beta for colormap (default: 0.0)")
    parser.add_argument("--max_display", type=float, default=10.0,
                        help="Max beta for colormap (default: 10.0)")
    parser.add_argument("--kernel", type=str, default="flex",
                        help="Kernel type (for checkpoint loading)")

    args = parser.parse_args()

    render_flex_beta(args)
