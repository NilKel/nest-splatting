#!/usr/bin/env python3
"""
Benchmark hash query overhead by comparing:
1. Normal mode: 5 hybrid levels + 1 hash level (20D + 4D = 24D)
2. No-hash mode: 6 hybrid levels + 0 hash levels (24D, all per-Gaussian)

The no-hash mode pads per-Gaussian features to 24D and disables hash queries entirely.
"""

import os
import sys
import pickle
import json
import time
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.point_utils import cam2rays
from utils.general_utils import build_rotation, build_H


def load_model(model_path, iteration=-1):
    """Load model from checkpoint."""
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

    # Load config
    args_json_path = os.path.join(model_path, 'args.json')
    args_pkl_path = os.path.join(model_path, 'args.pkl')

    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
    elif os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
    else:
        raise FileNotFoundError(f"No args.json or args.pkl found in {model_path}")

    args.model_path = model_path
    args.eval = True

    cfg_path = os.path.join(model_path, 'config.yaml')
    cfg = Config(cfg_path) if os.path.exists(cfg_path) else Config(args.yaml)

    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(model_path, iteration)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = 'UV'
    ingp.set_active_levels(iteration)
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    return gaussians, scene, ingp, cfg, pipe, args, iteration


def prune_dead_gaussians(gaussians, threshold=0.005):
    """Remove Gaussians with opacity below threshold."""
    dead_mask = (gaussians.get_opacity <= threshold).squeeze(-1)
    n_dead = dead_mask.sum().item()

    if n_dead > 0:
        valid = ~dead_mask
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation', '_appearance_level']:
            if hasattr(gaussians, attr):
                setattr(gaussians, attr, getattr(gaussians, attr)[valid])
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
            gaussians._gaussian_features = gaussians._gaussian_features[valid]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid]

    return n_dead


def benchmark_fps(gaussians, cameras, pipe, ingp, cfg, iteration, background,
                  n_passes=5, cache=None, aabb_mode='2dgs'):
    """Benchmark FPS over multiple passes."""
    beta = cfg.surfel.tg_beta

    total_time = 0
    total_frames = 0

    with torch.no_grad():
        for _ in range(n_passes):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for cam in cameras:
                _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                          iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                          fast_inference=True, cache=cache)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_frames += len(cameras)

    fps = total_frames / total_time
    ms = total_time / total_frames * 1000
    return fps, ms


def benchmark_components(gaussians, cam, pipe, ingp, cfg, iteration, background,
                         n_iters=50, cache=None, aabb_mode='2dgs'):
    """Profile individual components."""
    beta = cfg.surfel.tg_beta
    H, W = cam.image_height, cam.image_width

    results = {}

    # 1. Full render (fast_inference=True)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                      fast_inference=True, cache=cache)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['total_ms'] = sum(times) / len(times) * 1000

    # 2. cam2rays + normalize
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rays_d, rays_o = cam2rays(cam)
            ray_unit = F.normalize(rays_d, dim=-1).float()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['cam2rays_ms'] = sum(times) / len(times) * 1000

    # 3. MLP decode only
    feat_dim = ingp.levels * ingp.level_dim  # 24 for 6 levels * 4 dim
    dummy_features = torch.randn(H*W, feat_dim, device='cuda')
    dummy_dirs = F.normalize(torch.randn(H*W, 3, device='cuda'), dim=-1)

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = ingp.rgb_decode(dummy_features, dummy_dirs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['mlp_ms'] = sum(times) / len(times) * 1000

    # Compute rasterizer time (total - cam2rays - mlp)
    results['rasterizer_ms'] = results['total_ms'] - results['cam2rays_ms'] - results['mlp_ms']

    # Percentages
    results['cam2rays_pct'] = 100 * results['cam2rays_ms'] / results['total_ms']
    results['mlp_pct'] = 100 * results['mlp_ms'] / results['total_ms']
    results['rasterizer_pct'] = 100 * results['rasterizer_ms'] / results['total_ms']

    return results


def main():
    parser = ArgumentParser(description="Benchmark hash query overhead")
    parser.add_argument("--model_path", type=str,
                        default="outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral005decayadrrec_5_levels")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_passes", type=int, default=5)
    parser.add_argument("--prune_threshold", type=float, default=0.005)
    args = parser.parse_args()

    print("="*70)
    print("HASH QUERY OVERHEAD BENCHMARK")
    print("="*70)

    # Load model
    gaussians, scene, ingp, cfg, pipe, train_args, iteration = load_model(args.model_path)
    background = torch.zeros(3, device='cuda')

    print(f"\nModel: {args.model_path}")
    print(f"Iteration: {iteration}")
    print(f"Original config: hybrid_levels={ingp.hybrid_levels}, hashgrid_levels={ingp.hashgrid_levels}")
    print(f"Gaussians (before pruning): {len(gaussians.get_xyz):,}")

    # Prune dead Gaussians
    n_pruned = prune_dead_gaussians(gaussians, args.prune_threshold)
    print(f"Pruned {n_pruned:,} dead Gaussians")
    n_gaussians = len(gaussians.get_xyz)
    print(f"Gaussians (after pruning): {n_gaussians:,}")

    aabb_mode = getattr(train_args, 'aabb', '2dgs')
    print(f"AABB mode: {aabb_mode}")

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    print(f"Cameras: {len(all_cams)}")
    if len(all_cams) > 0:
        print(f"Image size: {all_cams[0].image_width}x{all_cams[0].image_height}")

    # Setup cache
    cache = RenderCache()
    cache.get_screenspace_points(n_gaussians)
    cache.homotrans = gaussians.get_homotrans()

    # Save original state
    original_gaussian_features = gaussians._gaussian_features.clone()
    original_hybrid_levels = ingp.hybrid_levels
    original_hashgrid_disabled = ingp.hashgrid_disabled
    original_hashgrid_levels = ingp.hashgrid_levels

    print("\n" + "="*70)
    print("MODE 1: WITH HASH (5 hybrid + 1 hash = 24D)")
    print("="*70)

    # Warmup
    print(f"Warmup ({args.n_warmup} frames)...")
    with torch.no_grad():
        for i in range(args.n_warmup):
            cam = all_cams[i % len(all_cams)]
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=cfg.surfel.tg_beta,
                      iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                      fast_inference=True, cache=cache)
    torch.cuda.synchronize()

    # Benchmark with hash
    fps_with_hash, ms_with_hash = benchmark_fps(
        gaussians, all_cams, pipe, ingp, cfg, iteration, background,
        n_passes=args.n_passes, cache=cache, aabb_mode=aabb_mode)
    print(f"FPS: {fps_with_hash:6.1f}  ({ms_with_hash:.2f} ms/frame)")

    # Component breakdown with hash
    components_with_hash = benchmark_components(
        gaussians, all_cams[0], pipe, ingp, cfg, iteration, background,
        n_iters=50, cache=cache, aabb_mode=aabb_mode)
    print(f"\nComponent breakdown:")
    print(f"  Total:      {components_with_hash['total_ms']:6.2f} ms")
    print(f"  Rasterizer: {components_with_hash['rasterizer_ms']:6.2f} ms ({components_with_hash['rasterizer_pct']:.1f}%)")
    print(f"  MLP:        {components_with_hash['mlp_ms']:6.2f} ms ({components_with_hash['mlp_pct']:.1f}%)")
    print(f"  cam2rays:   {components_with_hash['cam2rays_ms']:6.2f} ms ({components_with_hash['cam2rays_pct']:.1f}%)")

    print("\n" + "="*70)
    print("MODE 2: NO HASH (6 hybrid + 0 hash = 24D, all per-Gaussian)")
    print("="*70)

    # Modify ingp to disable hash
    ingp.hybrid_levels = 6  # All levels are per-Gaussian
    ingp.hashgrid_levels = 0
    ingp.hashgrid_disabled = True

    # Pad per-Gaussian features from 20D to 24D (just zeros for the extra 4D)
    # Original: (N, 20), Padded: (N, 24)
    padded_features = torch.zeros(n_gaussians, 24, device='cuda')
    padded_features[:, :20] = original_gaussian_features
    gaussians._gaussian_features = padded_features

    print(f"Padded per-Gaussian features: {original_gaussian_features.shape} -> {padded_features.shape}")
    print(f"ingp.hybrid_levels = {ingp.hybrid_levels}")
    print(f"ingp.hashgrid_disabled = {ingp.hashgrid_disabled}")

    # Warmup
    print(f"Warmup ({args.n_warmup} frames)...")
    with torch.no_grad():
        for i in range(args.n_warmup):
            cam = all_cams[i % len(all_cams)]
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=cfg.surfel.tg_beta,
                      iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                      fast_inference=True, cache=cache)
    torch.cuda.synchronize()

    # Benchmark without hash
    fps_no_hash, ms_no_hash = benchmark_fps(
        gaussians, all_cams, pipe, ingp, cfg, iteration, background,
        n_passes=args.n_passes, cache=cache, aabb_mode=aabb_mode)
    print(f"FPS: {fps_no_hash:6.1f}  ({ms_no_hash:.2f} ms/frame)")

    # Component breakdown without hash
    components_no_hash = benchmark_components(
        gaussians, all_cams[0], pipe, ingp, cfg, iteration, background,
        n_iters=50, cache=cache, aabb_mode=aabb_mode)
    print(f"\nComponent breakdown:")
    print(f"  Total:      {components_no_hash['total_ms']:6.2f} ms")
    print(f"  Rasterizer: {components_no_hash['rasterizer_ms']:6.2f} ms ({components_no_hash['rasterizer_pct']:.1f}%)")
    print(f"  MLP:        {components_no_hash['mlp_ms']:6.2f} ms ({components_no_hash['mlp_pct']:.1f}%)")
    print(f"  cam2rays:   {components_no_hash['cam2rays_ms']:6.2f} ms ({components_no_hash['cam2rays_pct']:.1f}%)")

    # Restore original state
    gaussians._gaussian_features = original_gaussian_features
    ingp.hybrid_levels = original_hybrid_levels
    ingp.hashgrid_disabled = original_hashgrid_disabled
    ingp.hashgrid_levels = original_hashgrid_levels

    # Summary
    hash_overhead_ms = ms_with_hash - ms_no_hash
    hash_overhead_pct = 100 * hash_overhead_ms / ms_with_hash if ms_with_hash > 0 else 0
    speedup = fps_no_hash / fps_with_hash if fps_with_hash > 0 else 0

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"With hash (5+1):    {fps_with_hash:6.1f} FPS  ({ms_with_hash:.2f} ms/frame)")
    print(f"Without hash (6+0): {fps_no_hash:6.1f} FPS  ({ms_no_hash:.2f} ms/frame)")
    print(f"")
    print(f"Total overhead: {hash_overhead_ms:.2f} ms/frame ({hash_overhead_pct:.1f}% of total)")
    print(f"Speedup without hash: {speedup:.2f}x")
    print(f"")
    raster_overhead = components_with_hash['rasterizer_ms'] - components_no_hash['rasterizer_ms']
    print(f"Rasterizer breakdown:")
    print(f"  With hash:    {components_with_hash['rasterizer_ms']:.2f} ms")
    print(f"  Without hash: {components_no_hash['rasterizer_ms']:.2f} ms")
    print(f"  Hash query:   {raster_overhead:.2f} ms")


if __name__ == '__main__':
    main()
