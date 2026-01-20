#!/usr/bin/env python3
"""
Benchmark AABB modes - measure both FPS and PSNR.
"""

import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.loss_utils import ssim
from utils.image_utils import psnr
import torchvision


def merge_cfg_to_args(args, cfg_model):
    args.cfg = cfg_model


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
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "beta", "flex", "general"])

    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    print(f"\n[BENCHMARK] Loading model from: {args.model_path}")

    ingp_model = INGP(cfg_model, args=args).to('cuda')
    iteration = args.iteration
    ingp_model.load_model(args.model_path, iteration)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    if args.kernel != "gaussian":
        gaussians.kernel_type = args.kernel
        print(f"[BENCHMARK] Kernel type set to: {args.kernel}")

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
    print(f"Kernel: {args.kernel}")
    print(f"Test images: {len(test_cameras)}")

    print(f"\n{'Mode':<5} {'Name':<25} {'FPS':>8} {'ms':>8} {'PSNR':>8} {'SSIM':>8} {'Speedup':>10}")
    print("-"*90)
    
    baseline_fps = results[0]['fps'] if 0 in results else 1.0
    baseline_psnr = results[0]['psnr'] if 0 in results else 0.0
    
    for mode in sorted(results.keys()):
        r = results[mode]
        speedup = (r['fps'] - baseline_fps) / baseline_fps * 100
        speedup_str = f"{speedup:+.1f}%" if mode != 0 else "baseline"
        psnr_diff = r['psnr'] - baseline_psnr
        psnr_str = f"{r['psnr']:.2f}" if mode == 0 else f"{r['psnr']:.2f} ({psnr_diff:+.2f})"
        print(f"{mode:<5} {r['name']:<25} {r['fps']:>8.2f} {r['ms']:>8.2f} {psnr_str:>8} {r['ssim']:>8.4f} {speedup_str:>10}")

    print("="*90)


if __name__ == "__main__":
    main()
