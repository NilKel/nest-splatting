#!/usr/bin/env python3
"""
Create a minimal single-Gaussian checkpoint for MLP gradient debugging.

This extracts one Gaussian with highest opacity from a trained checkpoint
and saves it as a new checkpoint for fast, reproducible testing.
"""

import os
import sys
import glob
import pickle
import shutil
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams

# Source checkpoint (trained 3D_direct model)
SOURCE_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"

# Output path for single-Gaussian checkpoint
OUTPUT_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/single_gaussian_test"


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


def create_single_gaussian_checkpoint(source_path=SOURCE_PATH, output_path=OUTPUT_PATH, gaussian_idx=None):
    """
    Create a single-Gaussian checkpoint from a trained model.

    Args:
        source_path: Path to source trained model
        output_path: Path to save single-Gaussian checkpoint
        gaussian_idx: Index of Gaussian to keep (None = highest opacity)
    """
    print("=" * 80)
    print("Creating Single-Gaussian Test Checkpoint")
    print("=" * 80)
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")

    # Load config
    args = load_training_config(source_path)
    args.model_path = source_path
    args.eval = True

    config_yaml_path = os.path.join(source_path, "config.yaml")
    cfg_model = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    # Find iteration
    ngp_files = glob.glob(os.path.join(source_path, "ngp_*.pth"))
    iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
    iteration = max(iterations)
    print(f"  Iteration: {iteration}")

    # Setup params
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"

    print(f"\n[INFO] Loaded {len(gaussians.get_xyz)} Gaussians")

    # Select single Gaussian
    if gaussian_idx is None:
        # Pick Gaussian with highest opacity
        opacities = gaussians.get_opacity.squeeze()
        gaussian_idx = opacities.argmax().item()
        print(f"[INFO] Selected Gaussian {gaussian_idx} (highest opacity: {opacities[gaussian_idx].item():.4f})")
    else:
        print(f"[INFO] Using specified Gaussian index: {gaussian_idx}")

    # Create mask for single Gaussian
    mask = torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device="cuda")
    mask[gaussian_idx] = True

    # Print original Gaussian info
    print(f"\n[GAUSSIAN {gaussian_idx}]")
    print(f"  xyz: {gaussians._xyz[gaussian_idx].tolist()}")
    print(f"  opacity: {gaussians._opacity[gaussian_idx].item():.4f}")
    print(f"  scaling: {gaussians._scaling[gaussian_idx].tolist()}")
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        print(f"  features[0:5]: {gaussians._gaussian_features[gaussian_idx, :5].tolist()}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Copy config files
    for fname in ["args.pkl", "args.json", "config.yaml", "cfg_args", "cameras.json"]:
        src = os.path.join(source_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_path)

    # Create point cloud directory
    pc_dir = os.path.join(output_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(pc_dir, exist_ok=True)

    # Save single Gaussian as PLY
    # We need to filter and save
    gaussians._xyz = nn.Parameter(gaussians._xyz[mask])
    gaussians._features_dc = nn.Parameter(gaussians._features_dc[mask])
    gaussians._features_rest = nn.Parameter(gaussians._features_rest[mask])
    gaussians._scaling = nn.Parameter(gaussians._scaling[mask])
    gaussians._rotation = nn.Parameter(gaussians._rotation[mask])
    gaussians._opacity = nn.Parameter(gaussians._opacity[mask])
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        gaussians._gaussian_features = nn.Parameter(gaussians._gaussian_features[mask])
    if hasattr(gaussians, '_appearance_level') and gaussians._appearance_level is not None:
        gaussians._appearance_level = gaussians._appearance_level[mask]

    gaussians.save_ply(os.path.join(pc_dir, "point_cloud.ply"))
    print(f"\n[SAVED] point_cloud.ply with 1 Gaussian")

    # Load and save INGP model (unchanged)
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(source_path, iteration)

    # Save INGP checkpoint
    ingp_model.save_model(output_path, iteration)
    print(f"[SAVED] ngp_{iteration}.pth")

    # Create a summary file
    summary_path = os.path.join(output_path, "single_gaussian_info.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Single Gaussian Test Checkpoint\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Source: {source_path}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Gaussian index (from source): {gaussian_idx}\n\n")
        f.write(f"Gaussian properties:\n")
        f.write(f"  xyz: {gaussians._xyz[0].tolist()}\n")
        f.write(f"  opacity: {gaussians._opacity[0].item():.4f}\n")
        f.write(f"  scaling: {gaussians._scaling[0].tolist()}\n")
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
            f.write(f"  features (first 5): {gaussians._gaussian_features[0, :5].tolist()}\n")

    print(f"[SAVED] single_gaussian_info.txt")
    print(f"\n[DONE] Single-Gaussian checkpoint created at: {output_path}")
    print(f"\nUse with:")
    print(f"  python scripts/test_3d_lean_vs_3d_direct.py --model_path {output_path}")

    return output_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_path", type=str, default=SOURCE_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--gaussian_idx", type=int, default=None, help="Specific Gaussian index (default: highest opacity)")
    args = parser.parse_args()

    create_single_gaussian_checkpoint(
        source_path=args.source_path,
        output_path=args.output_path,
        gaussian_idx=args.gaussian_idx
    )
