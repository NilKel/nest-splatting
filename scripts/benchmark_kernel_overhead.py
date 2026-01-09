#!/usr/bin/env python3
"""
Benchmark kernel overhead by fixing the number of evaluations per pixel.

By capping max_intersections (which now caps BEFORE kernel computation),
we ensure both Gaussian and General kernels do the same number of powf/exp calls.
This isolates the kernel computation cost from tile workload differences.
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


def merge_cfg_to_args(args, cfg_model):
    args.cfg = cfg_model


def benchmark_fps(gaussians, scene, pipe, ingp, cfg_model, iteration,
                  max_intersections=0, num_warmup=10, num_iters=100):
    """Benchmark FPS with fixed max evaluations."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        return None

    viewpoint = test_cameras[0]
    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model,
                      max_intersections=max_intersections)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model,
                      max_intersections=max_intersections)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    mean_time = sum(times) / len(times)
    fps = 1.0 / mean_time
    return fps, mean_time * 1000


def main():
    parser = ArgumentParser(description="Benchmark kernel overhead with fixed evaluations")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations for FPS benchmark")

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

    # Test different max_intersections values
    max_ints_to_test = [0, 1, 5, 10, 20, 50, 100]
    kernel_types = ["gaussian", "general"]

    results = {}

    for kernel in kernel_types:
        print(f"\n[BENCHMARK] Testing kernel: {kernel}")
        gaussians.kernel_type = kernel
        results[kernel] = {}

        for max_ints in max_ints_to_test:
            print(f"  max_intersections={max_ints}...", end=" ", flush=True)
            fps_data = benchmark_fps(gaussians, scene, pipe, ingp_model, cfg_model, iteration,
                                    max_intersections=max_ints, num_iters=args.num_iters)
            if fps_data:
                fps, ms = fps_data
                results[kernel][max_ints] = (fps, ms)
                print(f"FPS: {fps:.1f}, ms: {ms:.2f}")
            else:
                print("FAILED")

    # Summary table
    print("\n" + "="*90)
    print("KERNEL OVERHEAD BENCHMARK (Fixed Evaluations per Pixel)")
    print("="*90)
    print(f"Model: {args.model_path}")
    print(f"Gaussians: {num_gaussians:,}")

    # Header
    print(f"\n{'Max Evals':<12}", end="")
    for kernel in kernel_types:
        print(f"  {kernel:<12} {'ms':>8}", end="")
    print("  Overhead")
    print("-" * 90)

    for max_ints in max_ints_to_test:
        label = "unlimited" if max_ints == 0 else str(max_ints)
        print(f"{label:<12}", end="")

        fps_vals = []
        ms_vals = []
        for kernel in kernel_types:
            if max_ints in results[kernel]:
                fps, ms = results[kernel][max_ints]
                fps_vals.append(fps)
                ms_vals.append(ms)
                print(f"  {fps:>8.1f} FPS {ms:>8.2f}", end="")
            else:
                fps_vals.append(None)
                ms_vals.append(None)
                print(f"  {'N/A':>8} {'N/A':>8}", end="")

        # Calculate overhead (general vs gaussian)
        if len(ms_vals) == 2 and ms_vals[0] is not None and ms_vals[1] is not None:
            overhead_ms = ms_vals[1] - ms_vals[0]
            overhead_pct = (ms_vals[1] - ms_vals[0]) / ms_vals[0] * 100
            print(f"  {overhead_ms:+.2f}ms ({overhead_pct:+.1f}%)")
        else:
            print()

    print("="*90)

    # Analysis
    print("\n[ANALYSIS]")
    print("If overhead % is constant across max_evals values:")
    print("  → powf overhead is per-evaluation (compute bound)")
    print("If overhead % decreases with fewer max_evals:")
    print("  → overhead is elsewhere (memory/tile bound)")
    print("If overhead % is ~0 for all values:")
    print("  → No kernel overhead, bottleneck is preprocessing/MLP")


if __name__ == "__main__":
    main()
