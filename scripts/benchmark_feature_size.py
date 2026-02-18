#!/usr/bin/env python3
"""
Benchmark effect of MLP hidden dimension on rendering performance.

Tests different MLP hidden sizes (16, 32, 64, 256) with the original hash configuration
(5 hybrid levels + 1 hash level = 24D features).
"""

import os
import sys
import pickle
import json
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.point_utils import cam2rays


def load_model(model_path, iteration=-1):
    """Load model from checkpoint."""
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

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


def create_mlp(input_dim, hidden_dim=256, output_dim=3):
    """Create an MLP matching the original config (2 layers, 256 hidden)."""
    # Original config: num_layers=2, hidden_dim=256, weight_norm=True
    # Architecture: input -> hidden -> output (2 layer = 1 hidden layer)
    layers = [
        nn.utils.weight_norm(nn.Linear(input_dim + 3, hidden_dim)),  # +3 for view direction
        nn.ReLU(inplace=True),
        nn.utils.weight_norm(nn.Linear(hidden_dim, output_dim)),
        nn.Sigmoid()
    ]
    return nn.Sequential(*layers).cuda()


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

    # 3. MLP decode - measure the actual MLP being used
    feat_dim = ingp.levels * ingp.level_dim
    dummy_features = torch.randn(H*W, feat_dim, device='cuda')
    dummy_dirs = F.normalize(torch.randn(H*W, 3, device='cuda'), dim=-1)

    # Warmup MLP before timing
    with torch.no_grad():
        for _ in range(10):
            _ = ingp.rgb_decode(dummy_features, dummy_dirs)
    torch.cuda.synchronize()

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

    return results


def main():
    parser = ArgumentParser(description="Benchmark effect of feature dimension on rasterizer")
    parser.add_argument("--model_path", type=str,
                        default="outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral005decayadrrec_5_levels")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_passes", type=int, default=5)
    parser.add_argument("--prune_threshold", type=float, default=0.005)
    args = parser.parse_args()

    print("="*70)
    print("FEATURE DIMENSION BENCHMARK")
    print("="*70)

    # Load model
    gaussians, scene, ingp, cfg, pipe, train_args, iteration = load_model(args.model_path)
    background = torch.zeros(3, device='cuda')

    print(f"\nModel: {args.model_path}")
    print(f"Iteration: {iteration}")
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
    original_levels = ingp.levels
    original_hybrid_levels = ingp.hybrid_levels
    original_hashgrid_disabled = ingp.hashgrid_disabled
    original_hashgrid_levels = ingp.hashgrid_levels
    original_gaussian_features = gaussians._gaussian_features.clone() if gaussians._gaussian_features is not None else None
    original_rgb_decode = ingp.rgb_decode

    # Test grid: per-Gaussian feature sizes x MLP hidden sizes
    # Keep 1 hash level (4D), vary hybrid_levels from 1 to 5
    level_dim = ingp.level_dim  # 4
    hybrid_configs = [1, 2, 3, 4, 5]  # hybrid levels (+ 1 hash level each)
    hidden_sizes = [16, 32, 64, 256]

    results = []

    print(f"\nBenchmarking: hybrid_levels x MLP_hidden (with 1 hash level)")
    print(f"Original config: {original_hybrid_levels} hybrid + {original_hashgrid_levels} hash")

    for hybrid_levels in hybrid_configs:
        total_levels = hybrid_levels + 1  # +1 for hash level
        feat_dim = total_levels * level_dim
        pg_dim = hybrid_levels * level_dim  # per-Gaussian dimension

        for hidden_dim in hidden_sizes:
            print("\n" + "="*70)
            print(f"Testing: {hybrid_levels} hybrid + 1 hash = {feat_dim}D, MLP hidden={hidden_dim}")
            print("="*70)

            # Configure for this test
            ingp.levels = total_levels
            ingp.hybrid_levels = hybrid_levels
            ingp.hashgrid_levels = 1
            ingp.hashgrid_disabled = False

            # Create per-Gaussian features of the right size
            gaussians._gaussian_features = torch.randn(n_gaussians, pg_dim, device='cuda') * 0.1

            # Create a new MLP with the specified hidden size
            new_mlp = create_mlp(feat_dim, hidden_dim=hidden_dim, output_dim=3)

            # Replace rgb_decode with our new MLP
            def make_rgb_decode(mlp):
                def rgb_decode(features, view_dirs):
                    x = torch.cat([features, view_dirs], dim=-1)
                    return mlp(x)
                return rgb_decode

            ingp.rgb_decode = make_rgb_decode(new_mlp)

            # Warmup
            print(f"Warmup ({args.n_warmup} frames)...")
            with torch.no_grad():
                for i in range(args.n_warmup):
                    cam = all_cams[i % len(all_cams)]
                    _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=cfg.surfel.tg_beta,
                              iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                              fast_inference=True, cache=cache)
            torch.cuda.synchronize()

            # Benchmark FPS
            fps, ms = benchmark_fps(
                gaussians, all_cams, pipe, ingp, cfg, iteration, background,
                n_passes=args.n_passes, cache=cache, aabb_mode=aabb_mode)
            print(f"FPS: {fps:6.1f}  ({ms:.2f} ms/frame)")

            # Component breakdown
            components = benchmark_components(
                gaussians, all_cams[0], pipe, ingp, cfg, iteration, background,
                n_iters=50, cache=cache, aabb_mode=aabb_mode)
            print(f"  Rasterizer: {components['rasterizer_ms']:6.2f} ms")
            print(f"  MLP:        {components['mlp_ms']:6.2f} ms")

            results.append({
                'hybrid_levels': hybrid_levels,
                'feat_dim': feat_dim,
                'pg_dim': pg_dim,
                'hidden_dim': hidden_dim,
                'fps': fps,
                'ms': ms,
                'rasterizer_ms': components['rasterizer_ms'],
                'mlp_ms': components['mlp_ms'],
            })

    # Restore original state
    ingp.levels = original_levels
    ingp.hybrid_levels = original_hybrid_levels
    ingp.hashgrid_disabled = original_hashgrid_disabled
    ingp.hashgrid_levels = original_hashgrid_levels
    ingp.rgb_decode = original_rgb_decode
    if original_gaussian_features is not None:
        gaussians._gaussian_features = original_gaussian_features

    # Summary table - Full results
    print("\n" + "="*90)
    print("FULL RESULTS: Per-Gaussian Features x MLP Hidden Size (all with 1 hash level)")
    print("="*90)
    print(f"{'Hybrid':<8} {'FeatDim':<10} {'Hidden':<10} {'FPS':<10} {'Total ms':<12} {'Raster ms':<12} {'MLP ms':<10}")
    print("-"*82)
    for r in results:
        print(f"{r['hybrid_levels']:<8} {r['feat_dim']:<10} {r['hidden_dim']:<10} {r['fps']:<10.1f} {r['ms']:<12.2f} {r['rasterizer_ms']:<12.2f} {r['mlp_ms']:<10.2f}")

    # FPS Grid summary
    print("\n" + "="*70)
    print("FPS GRID: Hybrid Levels (rows) x Hidden Dim (cols)")
    print("="*70)
    header = "HybLvl"
    print(f"{header:<10}", end="")
    for h in hidden_sizes:
        print(f"{h:<12}", end="")
    print()
    print("-"*58)

    for hl in hybrid_configs:
        print(f"{hl:<10}", end="")
        for hd in hidden_sizes:
            # Find matching result
            for r in results:
                if r['hybrid_levels'] == hl and r['hidden_dim'] == hd:
                    print(f"{r['fps']:<12.1f}", end="")
                    break
        print()

    # Rasterizer time grid
    print("\n" + "="*70)
    print("RASTERIZER MS GRID: Hybrid Levels (rows) x Hidden Dim (cols)")
    print("="*70)
    print(f"{header:<10}", end="")
    for h in hidden_sizes:
        print(f"{h:<12}", end="")
    print()
    print("-"*58)

    for hl in hybrid_configs:
        print(f"{hl:<10}", end="")
        for hd in hidden_sizes:
            for r in results:
                if r['hybrid_levels'] == hl and r['hidden_dim'] == hd:
                    print(f"{r['rasterizer_ms']:<12.2f}", end="")
                    break
        print()


if __name__ == '__main__':
    main()
