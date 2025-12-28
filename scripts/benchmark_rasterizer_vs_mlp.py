#!/usr/bin/env python
"""
Benchmark to separate rasterizer time from MLP time.

Measures:
1. Full render (rasterizer + MLP)
2. Rasterizer only (skip_mlp=True)
3. MLP only (on random features)
"""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from utils.point_utils import cam2rays
import torch.nn.functional as torch_F


def benchmark_fps(gaussians, viewpoint, pipe, ingp, cfg_model, iteration, num_iters=100, skip_mlp=False, max_intersections=0, force_no_hash_cuda=False):
    """Benchmark render FPS."""
    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    with torch.no_grad():
        # Warm-up
        _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                  iteration=iteration, cfg=cfg_model, skip_mlp=skip_mlp, max_intersections=max_intersections,
                  force_no_hash_cuda=force_no_hash_cuda)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iters):
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, skip_mlp=skip_mlp, max_intersections=max_intersections,
                      force_no_hash_cuda=force_no_hash_cuda)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    return num_iters / elapsed, (elapsed / num_iters) * 1000


def benchmark_mlp_only(ingp, viewpoint, num_iters=100):
    """Benchmark just the MLP decode on random features."""
    H, W = viewpoint.image_height, viewpoint.image_width

    # Get actual feature dimension from the INGP model
    feat_dim = ingp.feat_dim  # This is the actual MLP input feature dim

    # Random features (simulating rasterizer output)
    features = torch.randn(H * W, feat_dim, device="cuda")

    # View directions
    rays_d, rays_o = cam2rays(viewpoint)
    ray_unit = torch_F.normalize(rays_d, dim=-1).float()

    with torch.no_grad():
        # Warm-up
        _ = ingp.rgb_decode(features, ray_unit)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iters):
            _ = ingp.rgb_decode(features, ray_unit)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    return num_iters / elapsed, (elapsed / num_iters) * 1000, feat_dim


def main():
    parser = ArgumentParser(description="Benchmark rasterizer vs MLP")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=100)

    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    args.cfg = cfg_model

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
            mask_device = valid_mask.to(gaussians._gaussian_features.device)
            gaussians._gaussian_features = gaussians._gaussian_features[mask_device]

    num_gaussians = len(gaussians.get_xyz)
    print(f"[BENCHMARK] Gaussians: {num_gaussians:,}")

    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras!")
        return

    viewpoint = test_cameras[0]
    H, W = viewpoint.image_height, viewpoint.image_width
    print(f"[BENCHMARK] Resolution: {W}x{H}")

    # Check mode
    is_cat_mode = hasattr(ingp_model, 'is_cat_mode') and ingp_model.is_cat_mode
    print(f"[BENCHMARK] Mode: {'cat' if is_cat_mode else 'baseline'}")

    print(f"\n[BENCHMARK] Running {args.num_iters} iterations each...")

    # 1. Full render (hash_in_CUDA=True, with MLP)
    print("\n  Full render (hash_in_CUDA + MLP)...")
    full_fps, full_ms = benchmark_fps(gaussians, viewpoint, pipe, ingp_model, cfg_model, iteration, args.num_iters, skip_mlp=False)
    print(f"    {full_fps:.2f} FPS ({full_ms:.2f} ms/frame)")

    # 2. Hash rasterizer, skip MLP, unlimited intersections
    print("\n  hash_in_CUDA=True, skip_mlp, unlimited intersections...")
    hash_fps, hash_ms = benchmark_fps(gaussians, viewpoint, pipe, ingp_model, cfg_model, iteration, args.num_iters, skip_mlp=True, force_no_hash_cuda=False)
    print(f"    {hash_fps:.2f} FPS ({hash_ms:.2f} ms/frame)")

    # 3. Hash rasterizer, skip MLP, max_intersections=1
    print("\n  hash_in_CUDA=True, skip_mlp, max_intersections=1...")
    hash1_fps, hash1_ms = benchmark_fps(gaussians, viewpoint, pipe, ingp_model, cfg_model, iteration, args.num_iters, skip_mlp=True, max_intersections=1, force_no_hash_cuda=False)
    print(f"    {hash1_fps:.2f} FPS ({hash1_ms:.2f} ms/frame)")

    # 4. Plain 2DGS rasterizer (no hash), unlimited intersections
    print("\n  hash_in_CUDA=False (plain 2DGS), unlimited intersections...")
    plain_fps, plain_ms = benchmark_fps(gaussians, viewpoint, pipe, ingp_model, cfg_model, iteration, args.num_iters, skip_mlp=True, force_no_hash_cuda=True)
    print(f"    {plain_fps:.2f} FPS ({plain_ms:.2f} ms/frame)")

    # 5. Plain 2DGS rasterizer (no hash), max_intersections=1
    print("\n  hash_in_CUDA=False (plain 2DGS), max_intersections=1...")
    plain1_fps, plain1_ms = benchmark_fps(gaussians, viewpoint, pipe, ingp_model, cfg_model, iteration, args.num_iters, skip_mlp=True, max_intersections=1, force_no_hash_cuda=True)
    print(f"    {plain1_fps:.2f} FPS ({plain1_ms:.2f} ms/frame)")

    # 6. MLP only
    print("\n  MLP only (random features)...")
    mlp_fps, mlp_ms, actual_feat_dim = benchmark_mlp_only(ingp_model, viewpoint, args.num_iters)
    print(f"    {mlp_fps:.2f} FPS ({mlp_ms:.2f} ms/frame)")
    print(f"    (MLP input: {actual_feat_dim}D features)")

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Gaussians: {num_gaussians:,}, Resolution: {W}x{H}")
    print(f"MLP input: {actual_feat_dim}D features")
    print()
    print(f"{'Component':<45} {'FPS':>10} {'ms/frame':>12}")
    print("-" * 70)
    print(f"{'Full render (hash + MLP)':<45} {full_fps:>10.2f} {full_ms:>12.2f}")
    print(f"{'Hash rasterizer (unlimited ints)':<45} {hash_fps:>10.2f} {hash_ms:>12.2f}")
    print(f"{'Hash rasterizer (max_ints=1)':<45} {hash1_fps:>10.2f} {hash1_ms:>12.2f}")
    print(f"{'Plain 2DGS (unlimited ints)':<45} {plain_fps:>10.2f} {plain_ms:>12.2f}")
    print(f"{'Plain 2DGS (max_ints=1)':<45} {plain1_fps:>10.2f} {plain1_ms:>12.2f}")
    print(f"{'MLP only':<45} {mlp_fps:>10.2f} {mlp_ms:>12.2f}")
    print()

    # Analysis
    hash_overhead = hash_ms - plain_ms
    hash1_overhead = hash1_ms - plain1_ms
    print(f"Hash grid query overhead (per intersection work):")
    print(f"  Unlimited ints: {hash_overhead:.2f} ms (hash={hash_ms:.2f} - plain={plain_ms:.2f})")
    print(f"  Max 1 int:      {hash1_overhead:.2f} ms (hash={hash1_ms:.2f} - plain={plain1_ms:.2f})")
    print()
    print(f"Intersection processing time (hash rasterizer):")
    print(f"  {hash_ms - hash1_ms:.2f} ms ({(hash_ms - hash1_ms)/hash_ms*100:.1f}% of hash rasterizer)")
    print()
    print(f"Intersection processing time (plain 2DGS):")
    print(f"  {plain_ms - plain1_ms:.2f} ms ({(plain_ms - plain1_ms)/plain_ms*100:.1f}% of plain rasterizer)")
    print("="*70)


if __name__ == "__main__":
    main()
