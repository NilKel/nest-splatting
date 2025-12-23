#!/usr/bin/env python3
"""
Batch render Gaussian intersection count heatmaps for all experiments in a directory.

This script finds all trained experiment folders within a given directory and
runs the intersection heatmap renderer on each of them.

Usage:
    # Process all experiments under a directory
    python scripts/render_intersection_heatmap_batch.py --exp_dir outputs/nerf_synthetic --yaml tiny

    # Process all experiments, render only frame 0
    python scripts/render_intersection_heatmap_batch.py --exp_dir outputs/nerf_synthetic --yaml tiny --frame 0

    # Skip train frames, only render test
    python scripts/render_intersection_heatmap_batch.py --exp_dir outputs/nerf_synthetic --yaml tiny --skip_train

The script will recursively search for folders containing 'point_cloud' subdirectories
(indicating trained experiments) and process each one.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_experiment_folders(root_dir):
    """Find all experiment folders containing trained models.

    An experiment folder is identified by having:
    - A 'point_cloud' subdirectory (contains saved Gaussians)
    - OR an 'ngp_*.pth' file (contains saved INGP model)
    """
    experiments = []
    root_path = Path(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirpath = Path(dirpath)

        # Check for point_cloud directory
        has_point_cloud = 'point_cloud' in dirnames

        # Check for ngp checkpoint
        has_ngp_checkpoint = any(f.startswith('ngp_') and f.endswith('.pth') for f in filenames)

        if has_point_cloud or has_ngp_checkpoint:
            experiments.append(str(dirpath))

    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description="Batch render intersection heatmaps for all experiments in a directory"
    )
    parser.add_argument("--exp_dir", type=str, required=True,
                       help="Root directory containing experiment folders")
    parser.add_argument("--yaml", type=str, default="tiny",
                       help="YAML config file name (must match training)")
    parser.add_argument("--method", type=str, default=None,
                       choices=["baseline", "cat", "adaptive", "adaptive_add", "adaptive_cat",
                               "diffuse", "specular", "diffuse_ngp", "diffuse_offset",
                               "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid"],
                       help="Rendering method (auto-detected from training_log.txt if not specified)")
    parser.add_argument("--hybrid_levels", type=int, default=None,
                       help="Number of hybrid levels (auto-detected from training_log.txt if not specified)")
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to load (default: latest)")
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip rendering train camera heatmaps")
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip rendering test camera heatmaps")
    parser.add_argument("--frame", type=int, default=None,
                       help="Render only this specific frame index (default: render all)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print experiments that would be processed without running")

    args = parser.parse_args()

    # Find all experiment folders
    print(f"\n{'='*60}")
    print(f"Searching for experiments in: {args.exp_dir}")
    print(f"{'='*60}\n")

    experiments = find_experiment_folders(args.exp_dir)

    if not experiments:
        print(f"No experiment folders found in {args.exp_dir}")
        print("An experiment folder should contain a 'point_cloud' directory or 'ngp_*.pth' checkpoint.")
        return

    print(f"Found {len(experiments)} experiment(s):\n")
    for exp in experiments:
        print(f"  - {exp}")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above experiments.")
        return

    print(f"\n{'='*60}")
    print("Starting batch processing...")
    print(f"{'='*60}\n")

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_script = os.path.join(script_dir, "render_intersection_heatmap.py")

    successful = []
    failed = []

    for i, exp_path in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Processing: {exp_path}")
        print("-" * 60)

        # Build command
        cmd = [
            sys.executable, render_script,
            "-m", exp_path,
            "--yaml", args.yaml,
            "--iteration", str(args.iteration),
        ]

        # Only pass method/hybrid_levels if explicitly specified (otherwise auto-detect)
        if args.method is not None:
            cmd.extend(["--method", args.method])
        if args.hybrid_levels is not None:
            cmd.extend(["--hybrid_levels", str(args.hybrid_levels)])

        if args.skip_train:
            cmd.append("--skip_train")
        if args.skip_test:
            cmd.append("--skip_test")
        if args.frame is not None:
            cmd.extend(["--frame", str(args.frame)])

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            successful.append(exp_path)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to process {exp_path}")
            failed.append((exp_path, str(e)))
        except Exception as e:
            print(f"[ERROR] Unexpected error processing {exp_path}: {e}")
            failed.append((exp_path, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("Batch Processing Complete")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)}/{len(experiments)}")
    print(f"  Failed:     {len(failed)}/{len(experiments)}")

    if failed:
        print(f"\nFailed experiments:")
        for exp_path, error in failed:
            print(f"  - {exp_path}")
            print(f"    Error: {error}")

    print()


if __name__ == "__main__":
    main()
