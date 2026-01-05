#!/usr/bin/env python
"""
Benchmark renderer FPS with different effective intersection caps.

By setting opacity to max (0.99), transmittance drops below threshold after 1 intersection.
This allows benchmarking the overhead of multiple intersections without CUDA changes.
"""

import torch
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser


def merge_cfg_to_args(args, cfg_model):
    """Merge config file settings into args."""
    args.cfg = cfg_model
    if hasattr(cfg_model, 'ingp_stage'):
        if hasattr(cfg_model.ingp_stage, 'XYZ_TYPE'):
            pass  # Keep XYZ_TYPE in cfg


def benchmark_fps(gaussians, scene, pipe, ingp, cfg_model, iteration, num_iters=100, opacity_override=None, max_intersections=0):
    """Benchmark render FPS, optionally overriding opacity or capping intersections."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras available!")
        return 0, 0

    viewpoint = test_cameras[0]
    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Save original opacity if we're overriding
    original_opacity = None
    if opacity_override is not None:
        original_opacity = gaussians._opacity.data.clone()
        # Set opacity to override value (in logit space since get_opacity applies sigmoid)
        # sigmoid(x) = opacity_override => x = log(opacity_override / (1 - opacity_override))
        if opacity_override >= 0.99:
            logit_val = 10.0  # Large value -> sigmoid close to 1
        else:
            logit_val = torch.log(torch.tensor(opacity_override / (1 - opacity_override)))
        gaussians._opacity.data.fill_(logit_val)

    with torch.no_grad():
        # Warm-up render
        _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                  iteration=iteration, cfg=cfg_model, max_intersections=max_intersections)
        torch.cuda.synchronize()

        # Time multiple renders
        start_time = time.time()
        for _ in range(num_iters):
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, max_intersections=max_intersections)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        fps = num_iters / elapsed
        ms_per_frame = (elapsed / num_iters) * 1000

    # Restore original opacity if we overrode it
    if original_opacity is not None:
        gaussians._opacity.data = original_opacity

    return fps, ms_per_frame


def get_intersection_stats(gaussians, scene, pipe, ingp, cfg_model, iteration, opacity_override=None, max_intersections=0):
    """Get intersection statistics from a render."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        return None

    viewpoint = test_cameras[0]
    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Save original opacity if we're overriding
    original_opacity = None
    if opacity_override is not None:
        original_opacity = gaussians._opacity.data.clone()
        if opacity_override >= 0.99:
            logit_val = 10.0
        else:
            logit_val = torch.log(torch.tensor(opacity_override / (1 - opacity_override)))
        gaussians._opacity.data.fill_(logit_val)

    with torch.no_grad():
        render_pkg = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                           iteration=iteration, cfg=cfg_model, max_intersections=max_intersections)

        if 'gaussian_num' in render_pkg:
            gaussian_num = render_pkg['gaussian_num']  # (1, H, W)
            stats = {
                'mean': gaussian_num.mean().item(),
                'median': gaussian_num.median().item(),
                'max': gaussian_num.max().item(),
                'min': gaussian_num.min().item(),
                'std': gaussian_num.std().item(),
            }
        else:
            stats = None

    # Restore original opacity
    if original_opacity is not None:
        gaussians._opacity.data = original_opacity

    return stats


def main():
    parser = ArgumentParser(description="Benchmark intersection overhead")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations for timing")
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "beta", "flex", "general"],
                        help="Kernel type: gaussian (default 2DGS), beta, flex, or general")

    args = get_combined_args(parser)

    # Load config
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    # Load model
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

    # Override kernel type if specified (PLY loading defaults to "beta" if shape column exists)
    if args.kernel != "gaussian":
        gaussians.kernel_type = args.kernel
        print(f"[BENCHMARK] Kernel type set to: {args.kernel}")

    num_gaussians_total = len(gaussians.get_xyz)
    print(f"[BENCHMARK] Total Gaussians in PLY: {num_gaussians_total:,}")

    # Prune dead Gaussians (same as training script does before FPS measurement)
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    n_alive = num_gaussians_total - n_dead
    print(f"[BENCHMARK] Dead (opacity <= 0.005): {n_dead:,}")
    print(f"[BENCHMARK] Alive (opacity > 0.005): {n_alive:,}")

    if n_dead > 0:
        # Manual pruning without optimizer (for eval mode)
        valid_mask = ~dead_mask
        gaussians._xyz = gaussians._xyz[valid_mask]
        gaussians._features_dc = gaussians._features_dc[valid_mask]
        gaussians._features_rest = gaussians._features_rest[valid_mask]
        gaussians._opacity = gaussians._opacity[valid_mask]
        gaussians._scaling = gaussians._scaling[valid_mask]
        gaussians._rotation = gaussians._rotation[valid_mask]
        gaussians._appearance_level = gaussians._appearance_level[valid_mask]
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
            # Ensure mask is on the same device as the tensor
            mask_device = valid_mask.to(gaussians._gaussian_features.device)
            gaussians._gaussian_features = gaussians._gaussian_features[mask_device]
        if hasattr(gaussians, '_adaptive_features') and gaussians._adaptive_features is not None and gaussians._adaptive_features.numel() > 0:
            mask_device = valid_mask.to(gaussians._adaptive_features.device)
            gaussians._adaptive_features = gaussians._adaptive_features[mask_device]
        print(f"[BENCHMARK] Pruned {n_dead:,} dead Gaussians")

    num_gaussians = len(gaussians.get_xyz)
    print(f"[BENCHMARK] Gaussians after pruning: {num_gaussians:,}")

    # Get test camera resolution
    test_cameras = scene.getTestCameras()
    if len(test_cameras) > 0:
        H, W = test_cameras[0].image_height, test_cameras[0].image_width
        print(f"[BENCHMARK] Resolution: {W}x{H}")

    # Get intersection stats with normal opacity
    print("\n[BENCHMARK] Getting intersection statistics (normal opacity)...")
    stats = get_intersection_stats(gaussians, scene, pipe, ingp_model, cfg_model, iteration)
    if stats:
        print(f"  Mean intersections: {stats['mean']:.2f}")
        print(f"  Median intersections: {stats['median']:.2f}")
        print(f"  Max intersections: {stats['max']:.0f}")
        print(f"  Std intersections: {stats['std']:.2f}")

    # Benchmark with different max_intersections caps
    caps_to_test = [0, 1, 20, 50, 100]  # 0 means no limit
    results = {}

    for cap in caps_to_test:
        cap_label = "unlimited" if cap == 0 else str(cap)
        print(f"\n[BENCHMARK] Testing max_intersections={cap_label}...")

        # Get intersection stats with this cap
        stats_cap = get_intersection_stats(gaussians, scene, pipe, ingp_model, cfg_model, iteration, max_intersections=cap)
        if stats_cap:
            print(f"  Actual mean intersections: {stats_cap['mean']:.2f}")
            print(f"  Actual max intersections: {stats_cap['max']:.0f}")

        # Benchmark FPS
        fps, ms = benchmark_fps(gaussians, scene, pipe, ingp_model, cfg_model, iteration,
                               num_iters=args.num_iters, max_intersections=cap)
        print(f"  FPS: {fps:.2f} ({ms:.2f} ms/frame)")
        results[cap] = {'fps': fps, 'ms': ms, 'stats': stats_cap}

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Gaussians: {num_gaussians:,}")
    if stats:
        print(f"Mean intersections (unlimited): {stats['mean']:.2f}")
        print(f"Max intersections (unlimited): {stats['max']:.0f}")

    print(f"\n{'Cap':<12} {'FPS':>10} {'ms/frame':>12} {'Mean Ints':>12} {'Max Ints':>10}")
    print("-" * 60)
    for cap in caps_to_test:
        cap_label = "unlimited" if cap == 0 else str(cap)
        r = results[cap]
        mean_ints = r['stats']['mean'] if r['stats'] else 0
        max_ints = r['stats']['max'] if r['stats'] else 0
        print(f"{cap_label:<12} {r['fps']:>10.2f} {r['ms']:>12.2f} {mean_ints:>12.2f} {max_ints:>10.0f}")

    # Speedup relative to unlimited
    if results[0]['fps'] > 0:
        print(f"\nSpeedups relative to unlimited:")
        for cap in caps_to_test[1:]:
            speedup = results[cap]['fps'] / results[0]['fps']
            print(f"  max_intersections={cap}: {speedup:.2f}x")
    print("="*60)


if __name__ == "__main__":
    main()
