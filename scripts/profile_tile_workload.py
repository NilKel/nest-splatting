#!/usr/bin/env python3
"""
Profile the Gaussian-Tile workload to detect "Ghost Computation".

If the model has few Gaussians but high tile pairs, the bounding boxes are too large.
This is the "efficiency ratio" - how much work is actually useful vs wasted.
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


def benchmark_fps(gaussians, scene, pipe, ingp, cfg_model, iteration, aabb_mode=0, num_warmup=10, num_iters=100):
    """Benchmark FPS for a given AABB mode."""
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
                      iteration=iteration, cfg=cfg_model, aabb_mode=aabb_mode)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, aabb_mode=aabb_mode)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    mean_time = sum(times) / len(times)
    fps = 1.0 / mean_time
    return fps, mean_time * 1000  # Return FPS and ms per frame


def profile_tile_workload(gaussians, scene, pipe, ingp, cfg_model, iteration, aabb_mode=0):
    """Profile tile workload and efficiency."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras available!")
        return None

    viewpoint = test_cameras[0]
    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta
    H, W = viewpoint.image_height, viewpoint.image_width

    with torch.no_grad():
        # Render and get radii
        render_pkg = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                           iteration=iteration, cfg=cfg_model, aabb_mode=aabb_mode)

        radii = render_pkg.get('radii', None)
        if radii is None:
            print("No radii returned from render!")
            return None

        # Filter out Gaussians with radius 0 (culled)
        valid_mask = radii > 0
        valid_radii = radii[valid_mask].float()

        # Calculate tiles per Gaussian
        # Diameter in pixels / tile size (16), squared for area
        TILE_SIZE = 16
        tiles_per_gaussian = ((valid_radii * 2.0) / TILE_SIZE).pow(2)

        # Get intersection statistics if available
        gaussian_num = render_pkg.get('gaussian_num', None)
        if gaussian_num is not None:
            mean_intersections = gaussian_num.mean().item()
            max_intersections = gaussian_num.max().item()
            total_visible_hits = gaussian_num.sum().item()
        else:
            mean_intersections = None
            max_intersections = None
            total_visible_hits = None

        stats = {
            'num_gaussians': len(gaussians.get_xyz),
            'num_visible': valid_mask.sum().item(),
            'mean_radius_pixels': valid_radii.mean().item(),
            'max_radius_pixels': valid_radii.max().item(),
            'mean_tiles_per_gaussian': tiles_per_gaussian.mean().item(),
            'max_tiles_per_gaussian': tiles_per_gaussian.max().item(),
            'total_tile_pairs': tiles_per_gaussian.sum().item(),
            'total_pixels': H * W,
            'mean_intersections': mean_intersections,
            'max_intersections': max_intersections,
            'total_visible_hits': total_visible_hits,
        }

        # Calculate efficiency ratio if we have intersection data
        if total_visible_hits is not None:
            # Total work done = tile_pairs * pixels_per_tile (approx)
            # But more directly: total_tile_pairs represents the sorting/binning work
            # The intersection count represents actual useful work
            stats['efficiency_ratio'] = total_visible_hits / stats['total_tile_pairs'] if stats['total_tile_pairs'] > 0 else 0

    return stats


def main():
    parser = ArgumentParser(description="Profile Gaussian-Tile workload")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "beta", "flex", "general"])
    parser.add_argument("--compare_aabb", action="store_true", help="Compare all AABB modes")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations for FPS benchmark")

    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    print(f"\n[WORKLOAD] Loading model from: {args.model_path}")

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
        print(f"[WORKLOAD] Kernel type set to: {args.kernel}")

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
    print(f"[WORKLOAD] Gaussians after pruning: {num_gaussians:,}")

    test_cameras = scene.getTestCameras()
    if len(test_cameras) > 0:
        H, W = test_cameras[0].image_height, test_cameras[0].image_width
        print(f"[WORKLOAD] Resolution: {W}x{H}")

    aabb_mode_names = {0: "square", 1: "AdR", 2: "rect", 3: "AdR+rect"}

    if args.compare_aabb:
        modes_to_test = [0, 1, 2, 3]
        results = {}
        fps_results = {}

        for mode in modes_to_test:
            print(f"\n[WORKLOAD] Testing aabb_mode={mode} ({aabb_mode_names[mode]})...")
            stats = profile_tile_workload(gaussians, scene, pipe, ingp_model, cfg_model, iteration, aabb_mode=mode)
            if stats:
                results[mode] = stats

            # Benchmark FPS
            print(f"[WORKLOAD] Benchmarking FPS ({args.num_iters} iterations)...")
            fps_data = benchmark_fps(gaussians, scene, pipe, ingp_model, cfg_model, iteration,
                                     aabb_mode=mode, num_iters=args.num_iters)
            if fps_data:
                fps_results[mode] = fps_data

        # Summary table
        print("\n" + "="*100)
        print("TILE WORKLOAD + FPS COMPARISON")
        print("="*100)
        print(f"Model: {args.model_path}")
        print(f"Gaussians: {num_gaussians:,}")
        print(f"Kernel: {args.kernel}")

        print(f"\n{'Mode':<12} {'Visible':<10} {'Mean Radius':<12} {'Total Tiles':<15} {'Mean Ints':<10} {'FPS':>8} {'ms/frame':>10}")
        print("-" * 100)
        for mode in modes_to_test:
            if mode in results:
                s = results[mode]
                mean_ints = f"{s['mean_intersections']:.2f}" if s['mean_intersections'] else "N/A"
                if mode in fps_results:
                    fps, ms = fps_results[mode]
                    print(f"{mode} ({aabb_mode_names[mode]:<8}) {s['num_visible']:<10} {s['mean_radius_pixels']:<12.1f} {s['total_tile_pairs']:<15,.0f} {mean_ints:<10} {fps:>8.1f} {ms:>10.2f}")
                else:
                    print(f"{mode} ({aabb_mode_names[mode]:<8}) {s['num_visible']:<10} {s['mean_radius_pixels']:<12.1f} {s['total_tile_pairs']:<15,.0f} {mean_ints:<10} {'N/A':>8} {'N/A':>10}")

        # Calculate potential speedup
        if 0 in results and 1 in results:
            baseline_tiles = results[0]['total_tile_pairs']
            adr_tiles = results[1]['total_tile_pairs']
            reduction = (baseline_tiles - adr_tiles) / baseline_tiles * 100
            print(f"\nTile reduction (mode 0 → 1): {reduction:.1f}%")

        if 0 in fps_results and 1 in fps_results:
            baseline_fps = fps_results[0][0]
            adr_fps = fps_results[1][0]
            speedup = (adr_fps - baseline_fps) / baseline_fps * 100
            print(f"FPS speedup (mode 0 → 1): {speedup:+.1f}%")

        print("="*100)

    else:
        # Single mode
        stats = profile_tile_workload(gaussians, scene, pipe, ingp_model, cfg_model, iteration, aabb_mode=0)

        # Benchmark FPS
        print(f"\n[WORKLOAD] Benchmarking FPS ({args.num_iters} iterations)...")
        fps_data = benchmark_fps(gaussians, scene, pipe, ingp_model, cfg_model, iteration,
                                 aabb_mode=0, num_iters=args.num_iters)

        if stats:
            print("\n" + "="*60)
            print("TILE WORKLOAD PROFILE")
            print("="*60)
            print(f"Model: {args.model_path}")
            print(f"Kernel: {args.kernel}")
            print(f"\nGaussians (total): {stats['num_gaussians']:,}")
            print(f"Gaussians (visible): {stats['num_visible']:,}")
            print(f"\nMean radius (pixels): {stats['mean_radius_pixels']:.1f}")
            print(f"Max radius (pixels): {stats['max_radius_pixels']:.1f}")
            print(f"\nMean tiles per Gaussian: {stats['mean_tiles_per_gaussian']:.1f}")
            print(f"Max tiles per Gaussian: {stats['max_tiles_per_gaussian']:.1f}")
            print(f"\nTotal Gaussian-Tile pairs: {stats['total_tile_pairs']:,.0f}")
            print(f"Total pixels: {stats['total_pixels']:,}")

            if stats['mean_intersections'] is not None:
                print(f"\nMean intersections per pixel: {stats['mean_intersections']:.2f}")
                print(f"Max intersections per pixel: {stats['max_intersections']:.0f}")
                print(f"Total visible hits: {stats['total_visible_hits']:,.0f}")

                if stats.get('efficiency_ratio'):
                    print(f"\nEfficiency ratio: {stats['efficiency_ratio']:.2%}")
                    print("  (Higher = less wasted computation)")

            if fps_data:
                fps, ms = fps_data
                print(f"\nFPS: {fps:.1f}")
                print(f"ms/frame: {ms:.2f}")

            print("="*60)


if __name__ == "__main__":
    main()
