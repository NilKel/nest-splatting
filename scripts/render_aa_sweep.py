#!/usr/bin/env python3
"""
Render a single frame with different AA (anti-aliasing) values to visualize distance-based attenuation.

Usage:
    # By frame index (0-indexed)
    python scripts/render_aa_sweep.py --model_path outputs/mip_360/bicycle/cat/model_5_levels --frame 0

    # By image name pattern
    python scripts/render_aa_sweep.py --model_path outputs/mip_360/bicycle/cat/model_5_levels --name "_DSC8679"

    # Custom AA range (default: 1.0 to 1e-8, factor 0.1)
    python scripts/render_aa_sweep.py --model_path outputs/mip_360/bicycle/cat/model_5_levels --frame 0 --aa_start 2.0 --aa_end 1e-6 --aa_factor 0.1

Output:
    Creates aa_sweep/ directory in model_path with renders at each AA value.
"""

import os
import sys
import json
import pickle
import glob
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_max_iteration(model_path):
    """Find the maximum iteration from saved checkpoints."""
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if os.path.exists(point_cloud_dir):
        folders = [f for f in os.listdir(point_cloud_dir) if f.startswith("iteration_")]
        if folders:
            iters = [int(f.split("_")[1]) for f in folders]
            return max(iters)

    # Fallback: look for ngp_*.pth files
    ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
    if ngp_files:
        iters = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        return max(iters)

    return None

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
        return args

    raise FileNotFoundError(f"No training config found in {model_path}")


def save_image(tensor, path):
    """Save a tensor (C, H, W) as an image."""
    img = tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = ArgumentParser(description="Render single frame with AA sweep")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--frame", type=int, default=None,
                       help="Frame index (0-indexed) from test set")
    parser.add_argument("--name", type=str, default=None,
                       help="Substring to match in camera image_name")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--aa_start", type=float, default=1.0,
                       help="Starting AA value (highest)")
    parser.add_argument("--aa_end", type=float, default=1e-8,
                       help="Ending AA value (lowest)")
    parser.add_argument("--aa_factor", type=float, default=0.1,
                       help="Multiply AA by this factor each step")
    parser.add_argument("--aa_threshold", type=float, default=0.01,
                       help="AA threshold for hash skip")
    parser.add_argument("--train", action="store_true",
                       help="Use train cameras instead of test")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/aa_sweep)")

    eval_args = parser.parse_args()

    if eval_args.frame is None and eval_args.name is None:
        parser.error("Must specify either --frame or --name")

    # Load training config
    print(f"\n[AA SWEEP] Loading model: {eval_args.model_path}")
    args = load_training_config(eval_args.model_path)
    args.model_path = eval_args.model_path

    # Verify it's cat mode
    if args.method != "cat":
        print(f"[WARNING] Model was trained with method={args.method}, AA only affects cat mode")

    # Load YAML config
    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    # Resolve iteration (-1 means latest)
    iteration = eval_args.iteration
    if iteration == -1:
        iteration = find_max_iteration(eval_args.model_path)
        if iteration is None:
            print("[ERROR] Could not find any saved checkpoints")
            sys.exit(1)
        print(f"[AA SWEEP] Using iteration: {iteration}")

    # Create parser for ModelParams/PipelineParams
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(eval_args.model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        gaussians._xyz = gaussians._xyz[valid_mask]
        gaussians._features_dc = gaussians._features_dc[valid_mask]
        gaussians._features_rest = gaussians._features_rest[valid_mask]
        gaussians._opacity = gaussians._opacity[valid_mask]
        gaussians._scaling = gaussians._scaling[valid_mask]
        gaussians._rotation = gaussians._rotation[valid_mask]
        gaussians._appearance_level = gaussians._appearance_level[valid_mask]
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
            gaussians._gaussian_features = gaussians._gaussian_features[valid_mask.to(gaussians._gaussian_features.device)]
        if hasattr(gaussians, '_adaptive_features') and gaussians._adaptive_features is not None and gaussians._adaptive_features.numel() > 0:
            gaussians._adaptive_features = gaussians._adaptive_features[valid_mask.to(gaussians._adaptive_features.device)]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    print(f"[AA SWEEP] Gaussians: {len(gaussians.get_xyz):,}")

    # Get cameras
    cameras = scene.getTrainCameras() if eval_args.train else scene.getTestCameras()
    cameras = sorted(cameras, key=lambda c: c.image_name)
    camera_type = "train" if eval_args.train else "test"
    print(f"[AA SWEEP] {len(cameras)} {camera_type} cameras")

    # Find target camera
    target_cam = None
    if eval_args.frame is not None:
        if eval_args.frame < 0 or eval_args.frame >= len(cameras):
            print(f"[ERROR] Frame {eval_args.frame} out of range [0, {len(cameras)-1}]")
            sys.exit(1)
        target_cam = cameras[eval_args.frame]
    else:
        for cam in cameras:
            if eval_args.name in cam.image_name:
                target_cam = cam
                break
        if target_cam is None:
            print(f"[ERROR] No camera found matching '{eval_args.name}'")
            print("Available cameras:")
            for i, cam in enumerate(cameras[:10]):
                print(f"  {i}: {cam.image_name}")
            if len(cameras) > 10:
                print(f"  ... ({len(cameras) - 10} more)")
            sys.exit(1)

    print(f"[AA SWEEP] Target camera: {target_cam.image_name}")

    # Setup output
    output_dir = eval_args.output_dir or os.path.join(eval_args.model_path, "aa_sweep")
    os.makedirs(output_dir, exist_ok=True)

    # Background
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Generate AA values: start, start*factor, start*factor^2, ... until <= end
    aa_values = []
    aa = eval_args.aa_start
    while aa >= eval_args.aa_end:
        aa_values.append(aa)
        aa *= eval_args.aa_factor

    # Also include 0.0 (disabled) for reference
    aa_values.append(0.0)

    print(f"[AA SWEEP] Rendering {len(aa_values)} frames with AA values:")
    for aa in aa_values:
        print(f"  {aa:.2e}")

    # Render with each AA value
    with torch.no_grad():
        for i, aa_val in enumerate(aa_values):
            render_pkg = render(
                target_cam, gaussians, pipe, background,
                ingp=ingp_model, beta=beta, iteration=iteration, cfg=cfg_model,
                aa=aa_val, aa_threshold=eval_args.aa_threshold
            )

            rendered = render_pkg["render"]

            # Save with descriptive filename
            if aa_val == 0.0:
                filename = f"{i:02d}_aa_0_disabled.png"
            else:
                filename = f"{i:02d}_aa_{aa_val:.2e}.png"

            save_path = os.path.join(output_dir, filename)
            save_image(rendered, save_path)
            print(f"  Saved: {filename}")

    # Save ground truth for comparison
    gt = target_cam.original_image.to("cuda")
    save_image(gt, os.path.join(output_dir, "gt.png"))
    print(f"  Saved: gt.png")

    # Save metadata
    metadata = {
        'model_path': eval_args.model_path,
        'camera_name': target_cam.image_name,
        'camera_type': camera_type,
        'aa_values': aa_values,
        'aa_threshold': eval_args.aa_threshold,
        'method': args.method,
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[AA SWEEP] Done! Output saved to: {output_dir}")
    print(f"[AA SWEEP] Compare images to see how AA affects distant Gaussians:")
    print(f"  - Higher AA (1.0): More attenuation of high-freq hash at distance")
    print(f"  - Lower AA (1e-8): Less attenuation, more detail preserved")
    print(f"  - AA = 0: Disabled, original behavior")


if __name__ == "__main__":
    main()
