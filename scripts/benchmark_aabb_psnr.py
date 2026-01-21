#!/usr/bin/env python3
"""
Benchmark AABB modes - measure both FPS and PSNR.

Usage:
    python scripts/benchmark_aabb_psnr.py --model_path outputs/nerf_synthetic/chair/cat/run_name_5_levels
"""

import torch
import time
import sys
import os
import pickle
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.loss_utils import ssim
from utils.image_utils import psnr


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args
    raise FileNotFoundError(f"No args.pkl found in {model_path}")


def evaluate_mode(gaussians, scene, pipe, ingp, cfg_model, iteration, aabb_mode=0):
    """Evaluate FPS and PSNR for an AABB mode over all test images."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        return None

    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    psnrs = []
    ssims = []
    times_list = []

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = render(test_cameras[0], gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, aabb_mode=aabb_mode)
        torch.cuda.synchronize()

        # Evaluate all test images
        for viewpoint in test_cameras:
            torch.cuda.synchronize()
            t0 = time.time()
            render_pkg = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                               iteration=iteration, cfg=cfg_model, aabb_mode=aabb_mode)
            torch.cuda.synchronize()
            times_list.append(time.time() - t0)

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = viewpoint.original_image[0:3, :, :].cuda()

            psnrs.append(psnr(image, gt).mean().item())
            ssims.append(ssim(image, gt).item())

    mean_psnr = sum(psnrs) / len(psnrs)
    mean_ssim = sum(ssims) / len(ssims)
    mean_time = sum(times_list) / len(times_list)
    fps = 1.0 / mean_time

    return {
        'fps': fps,
        'ms': mean_time * 1000,
        'psnr': mean_psnr,
        'ssim': mean_ssim,
        'num_images': len(test_cameras)
    }


def main():
    parser = ArgumentParser(description="Benchmark AABB modes - FPS and PSNR")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to load (-1 for latest)")

    eval_args = parser.parse_args()

    # Load training config from checkpoint
    print(f"\n[BENCHMARK] Loading config from: {eval_args.model_path}")
    args = load_training_config(eval_args.model_path)

    # Override model_path to the actual directory
    args.model_path = eval_args.model_path

    # Force eval=True to load test cameras
    args.eval = True

    # Load YAML config from checkpoint directory
    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
        print(f"[CONFIG] Loaded config from {config_yaml_path}")
    else:
        raise FileNotFoundError(f"No config.yaml found in {eval_args.model_path}")

    args.cfg = cfg_model

    # Auto-detect iteration if not specified
    iteration = eval_args.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth checkpoints found in {eval_args.model_path}")

    print(f"\n[BENCHMARK] Loading model from: {args.model_path}")

    # Create parser for ModelParams/PipelineParams extraction
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(eval_args.model_path, iteration)

    dataset, pipe = model_params.extract(args), pipeline_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    # Set kernel type from saved args
    kernel_type = getattr(args, 'kernel', 'gaussian')
    if kernel_type != "gaussian":
        gaussians.kernel_type = kernel_type
        print(f"[BENCHMARK] Kernel type set to: {kernel_type}")

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

    num_gaussians = len(gaussians.get_xyz)
    print(f"[BENCHMARK] Gaussians after pruning: {num_gaussians:,}")

    test_cameras = scene.getTestCameras()
    if len(test_cameras) > 0:
        H, W = test_cameras[0].image_height, test_cameras[0].image_width
        print(f"[BENCHMARK] Resolution: {W}x{H}")
        print(f"[BENCHMARK] Test images: {len(test_cameras)}")

    aabb_modes = {
        0: "2dgs (square, 4σ)",
        1: "adr_only (square, AdR)",
        2: "rect (rectangular, 4σ)",
        3: "adr (rectangular, AdR)",
        4: "beta (fixed r=1 cutoff)",
    }

    results = {}

    for mode, name in aabb_modes.items():
        print(f"\n[BENCHMARK] Testing mode {mode}: {name}...")
        result = evaluate_mode(gaussians, scene, pipe, ingp_model, cfg_model, iteration, aabb_mode=mode)
        if result:
            results[mode] = result
            results[mode]['name'] = name
            print(f"  FPS: {result['fps']:.2f}, PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}")

    # Summary table
    print("\n" + "="*90)
    print("AABB MODE BENCHMARK (Full Test Set - FPS + Quality)")
    print("="*90)
    print(f"Model: {args.model_path}")
    print(f"Gaussians: {num_gaussians:,}")
    print(f"Kernel: {kernel_type}")
    print(f"Test images: {len(test_cameras)}")

    print(f"\n{'Mode':<5} {'Name':<25} {'FPS':>8} {'ms':>8} {'PSNR':>8} {'SSIM':>8} {'Speedup':>10}")
    print("-"*85)

    baseline_fps = results[0]['fps'] if 0 in results else 1.0

    for mode in sorted(results.keys()):
        r = results[mode]
        speedup = (r['fps'] - baseline_fps) / baseline_fps * 100
        speedup_str = f"{speedup:+.1f}%" if mode != 0 else "baseline"
        print(f"{mode:<5} {r['name']:<25} {r['fps']:>8.2f} {r['ms']:>8.2f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {speedup_str:>10}")

    print("="*85)

    # Save results to file
    output_file = os.path.join(args.model_path, "aabb_benchmark.txt")
    with open(output_file, 'w') as f:
        f.write("="*85 + "\n")
        f.write("AABB MODE BENCHMARK (Full Test Set - FPS + Quality)\n")
        f.write("="*85 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Gaussians: {num_gaussians:,}\n")
        f.write(f"Kernel: {kernel_type}\n")
        f.write(f"Test images: {len(test_cameras)}\n")
        f.write(f"\n{'Mode':<5} {'Name':<25} {'FPS':>8} {'ms':>8} {'PSNR':>8} {'SSIM':>8} {'Speedup':>10}\n")
        f.write("-"*85 + "\n")

        for mode in sorted(results.keys()):
            r = results[mode]
            speedup = (r['fps'] - baseline_fps) / baseline_fps * 100
            speedup_str = f"{speedup:+.1f}%" if mode != 0 else "baseline"
            f.write(f"{mode:<5} {r['name']:<25} {r['fps']:>8.2f} {r['ms']:>8.2f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {speedup_str:>10}\n")

        f.write("="*85 + "\n")

    print(f"\n[BENCHMARK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
