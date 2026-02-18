#!/usr/bin/env python3
"""
Re-render intersection maps for specific frames with custom max_display.

Usage:
    python scripts/rerender_intersection.py --model_paths outputs/nerf_synthetic/chair/cat/fixedscaled* --frame r_50 --max_display 100
"""

import os
import sys
import glob
import pickle
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from argparse import Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8, create_intersection_heatmap, create_intersection_histogram


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args

    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        return Namespace(**args_dict)

    raise FileNotFoundError(f"No training config found in {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                       help="Paths to model directories (supports glob patterns)")
    parser.add_argument("--frame", type=str, default="r_50",
                       help="Frame name to render (e.g., r_50)")
    parser.add_argument("--max_display", type=int, default=100,
                       help="Max value for intersection colormap (default 100)")
    parser.add_argument("--output_suffix", type=str, default="_max100",
                       help="Suffix for output filename")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")

    args = parser.parse_args()

    # Expand glob patterns
    model_paths = []
    for pattern in args.model_paths:
        expanded = glob.glob(pattern)
        if expanded:
            model_paths.extend(expanded)
        else:
            model_paths.append(pattern)

    model_paths = sorted(set(model_paths))
    print(f"Processing {len(model_paths)} model(s)...")

    for model_path in model_paths:
        if not os.path.isdir(model_path):
            print(f"Skipping {model_path} (not a directory)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_path}")
        print(f"{'='*60}")

        try:
            # Load training config
            train_args = load_training_config(model_path)
            train_args.model_path = model_path
            train_args.eval = True

            # Load YAML config
            config_yaml_path = os.path.join(model_path, "config.yaml")
            if os.path.exists(config_yaml_path):
                cfg_model = Config(config_yaml_path)
            else:
                cfg_model = Config(train_args.yaml)

            # Find iteration
            iteration = args.iteration
            if iteration == -1:
                ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
                if ngp_files:
                    iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
                    iteration = max(iterations)
                else:
                    print(f"  No checkpoints found, skipping")
                    continue

            print(f"  Loading iteration {iteration}")

            # Setup models
            temp_parser = argparse.ArgumentParser()
            model_params = ModelParams(temp_parser, sentinel=True)
            pipeline_params = PipelineParams(temp_parser)

            dataset = model_params.extract(train_args)
            pipe = pipeline_params.extract(train_args)

            # Load INGP
            ingp_model = INGP(cfg_model, args=train_args).to('cuda')
            ingp_model.load_model(model_path, iteration)

            # Load Gaussians
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
            gaussians.XYZ_TYPE = "UV"
            ingp_model.set_active_levels(iteration)

            if hasattr(train_args, 'kernel'):
                gaussians.kernel_type = train_args.kernel

            # Find the target camera
            test_cameras = scene.getTestCameras()
            target_cam = None
            for cam in test_cameras:
                if cam.image_name == args.frame:
                    target_cam = cam
                    break

            if target_cam is None:
                print(f"  Frame '{args.frame}' not found in test cameras, trying train cameras...")
                train_cameras = scene.getTrainCameras()
                for cam in train_cameras:
                    if cam.image_name == args.frame:
                        target_cam = cam
                        break

            if target_cam is None:
                print(f"  Frame '{args.frame}' not found, skipping")
                continue

            print(f"  Rendering frame: {target_cam.image_name}")

            # Render
            background = torch.zeros(3, device="cuda")
            beta = cfg_model.surfel.tg_beta

            with torch.no_grad():
                render_pkg = render(target_cam, gaussians, pipe, background,
                                   ingp=ingp_model, beta=beta,
                                   iteration=iteration, cfg=cfg_model)

            # Create intersection heatmap with custom max_display
            gaussian_num = render_pkg['gaussian_num']
            intersection_heatmap, min_count, max_count = create_intersection_heatmap(
                gaussian_num, max_display=args.max_display
            )
            histogram_img, stats = create_intersection_histogram(
                gaussian_num, max_display=args.max_display
            )

            print(f"  Intersection stats: min={min_count}, max={max_count}, mean={stats['mean']:.1f}")

            # Save to intersection output dir
            intersection_dir = os.path.join(model_path, "final_test_intersection")
            os.makedirs(intersection_dir, exist_ok=True)

            # Find existing index for this frame
            existing_files = glob.glob(os.path.join(intersection_dir, f"*_{args.frame}_intersection.png"))
            if existing_files:
                # Extract index from existing file
                basename = os.path.basename(existing_files[0])
                idx = int(basename.split('_')[0])
            else:
                idx = 0

            output_name = f"{idx:03d}_{args.frame}_intersection{args.output_suffix}.png"
            hist_name = f"{idx:03d}_{args.frame}_histogram{args.output_suffix}.png"

            save_img_u8(intersection_heatmap, os.path.join(intersection_dir, output_name))
            save_img_u8(histogram_img, os.path.join(intersection_dir, hist_name))

            print(f"  Saved: {output_name}")
            print(f"  Saved: {hist_name}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone!")


if __name__ == "__main__":
    main()
