#!/usr/bin/env python3
"""
Full render pipeline benchmark with component breakdown.

Usage:
    python scripts/benchmark_full.py --model_path outputs/mip_360/bicycle/cat/...
"""

import os
import sys
import pickle
import json
import time
import glob
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.point_utils import cam2rays
from utils.general_utils import build_rotation, build_H
from utils.loss_utils import ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace


def load_model(model_path, iteration=-1):
    """Load model and return all components."""
    # Find iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

    # Load config - try args.json first, fall back to args.pkl
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

    # Setup
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP
    ingp = INGP(cfg, args=args).cuda()
    ingp.load_model(model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
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
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_appearance_level']:
            if hasattr(gaussians, attr):
                setattr(gaussians, attr, getattr(gaussians, attr)[valid])

        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
            gaussians._gaussian_features = gaussians._gaussian_features[valid]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid]

    return n_dead


def warmup(gaussians, cameras, pipe, ingp, cfg, iteration, background,
           n_warmup=20, fast_inference=True, cache=None, aabb_mode='2dgs'):
    """Run warmup frames (not counted in benchmark)."""
    beta = cfg.surfel.tg_beta

    with torch.no_grad():
        for i in range(n_warmup):
            cam = cameras[i % len(cameras)]
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                      fast_inference=fast_inference, cache=cache)
    torch.cuda.synchronize()


def benchmark_fps(gaussians, cameras, pipe, ingp, cfg, iteration, background,
                  n_passes=3, fast_inference=True, cache=None, aabb_mode='2dgs'):
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
                          fast_inference=fast_inference, cache=cache)
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
    feat_dim = 24
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

    # 4. Preprocess (build_rotation + build_H) - reference, not in fast path with cache
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rots = build_rotation(gaussians._rotation)
            _ = build_H(rots, gaussians.get_scaling, gaussians.get_xyz)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['preprocess_ms'] = sum(times) / len(times) * 1000

    # Estimate rasterizer (total - cam2rays - mlp)
    results['rasterizer_ms'] = results['total_ms'] - results['cam2rays_ms'] - results['mlp_ms']

    # Percentages
    results['cam2rays_pct'] = 100 * results['cam2rays_ms'] / results['total_ms']
    results['mlp_pct'] = 100 * results['mlp_ms'] / results['total_ms']
    results['rasterizer_pct'] = 100 * results['rasterizer_ms'] / results['total_ms']

    return results


def verify_quality(gaussians, cameras, pipe, ingp, cfg, iteration, background, cache=None, aabb_mode='2dgs'):
    """Verify fast_inference produces same quality as regular render."""
    beta = cfg.surfel.tg_beta

    psnr_ref_list = []
    psnr_fast_list = []
    max_diff_list = []

    with torch.no_grad():
        for cam in cameras:
            gt = cam.original_image.cuda().clamp(0, 1)

            # Reference (fast_inference=False)
            result_ref = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                               iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                               fast_inference=False)
            img_ref = result_ref['render'].clamp(0, 1)

            # Fast (fast_inference=True)
            result_fast = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                iteration=iteration, cfg=cfg, aabb_mode=aabb_mode,
                                fast_inference=True, cache=cache)
            img_fast = result_fast['render'].clamp(0, 1)

            # PSNR vs GT
            psnr_ref_list.append(psnr(img_ref, gt).mean().item())
            psnr_fast_list.append(psnr(img_fast, gt).mean().item())

            # Max diff between fast and ref
            max_diff_list.append((img_ref - img_fast).abs().max().item())

    return {
        'psnr_ref': sum(psnr_ref_list) / len(psnr_ref_list),
        'psnr_fast': sum(psnr_fast_list) / len(psnr_fast_list),
        'max_diff': max(max_diff_list),
        'mean_max_diff': sum(max_diff_list) / len(max_diff_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--n_warmup', type=int, default=20, help='Warmup frames (not counted)')
    parser.add_argument('--n_passes', type=int, default=3, help='Benchmark passes over all cameras')
    parser.add_argument('--n_profile_iters', type=int, default=50, help='Iterations for component profiling')
    parser.add_argument('--skip_quality', action='store_true', help='Skip quality verification')
    parser.add_argument('--prune_threshold', type=float, default=0.005, help='Opacity threshold for pruning')
    args = parser.parse_args()

    print("="*70)
    print("LOADING MODEL")
    print("="*70)

    gaussians, scene, ingp, cfg, pipe, train_args, iteration = load_model(
        args.model_path, args.iteration)

    print(f"Model: {args.model_path}")
    print(f"Iteration: {iteration}")
    print(f"Method: {train_args.method}")
    print(f"Kernel: {getattr(train_args, 'kernel', 'gaussian')}")
    print(f"Gaussians (before pruning): {len(gaussians.get_xyz):,}")

    # Prune dead Gaussians
    n_pruned = prune_dead_gaussians(gaussians, args.prune_threshold)
    print(f"Pruned {n_pruned:,} dead Gaussians (opacity <= {args.prune_threshold})")
    print(f"Gaussians (after pruning): {len(gaussians.get_xyz):,}")

    # Get aabb_mode from args (same as training)
    aabb_mode = getattr(train_args, 'aabb', '2dgs')
    print(f"AABB mode: {aabb_mode}")

    # Get cameras
    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    test_cams = scene.getTestCameras()
    print(f"Total cameras: {len(all_cams)} (train+test)")
    print(f"Test cameras: {len(test_cams)}")

    if len(all_cams) > 0:
        print(f"Image size: {all_cams[0].image_width}x{all_cams[0].image_height}")

    background = torch.zeros(3, device='cuda')

    # Setup cache
    cache = RenderCache()
    cache.get_screenspace_points(len(gaussians.get_xyz))
    cache.homotrans = gaussians.get_homotrans()

    print("\n" + "="*70)
    print(f"WARMUP ({args.n_warmup} frames, not counted)")
    print("="*70)
    warmup(gaussians, all_cams, pipe, ingp, cfg, iteration, background,
           n_warmup=args.n_warmup, fast_inference=True, cache=cache, aabb_mode=aabb_mode)
    print("Warmup complete.")

    print("\n" + "="*70)
    print(f"FPS BENCHMARK ({args.n_passes} passes x {len(all_cams)} cameras)")
    print("="*70)

    fps_fast_cache, ms_fast_cache = benchmark_fps(
        gaussians, all_cams, pipe, ingp, cfg, iteration, background,
        n_passes=args.n_passes, fast_inference=True, cache=cache, aabb_mode=aabb_mode)
    print(f"fast_inference=True, cache=True:  {fps_fast_cache:6.1f} FPS  ({ms_fast_cache:.2f} ms/frame)")

    fps_fast_nocache, ms_fast_nocache = benchmark_fps(
        gaussians, all_cams, pipe, ingp, cfg, iteration, background,
        n_passes=args.n_passes, fast_inference=True, cache=None, aabb_mode=aabb_mode)
    print(f"fast_inference=True, cache=False: {fps_fast_nocache:6.1f} FPS  ({ms_fast_nocache:.2f} ms/frame)")

    fps_orig, ms_orig = benchmark_fps(
        gaussians, all_cams, pipe, ingp, cfg, iteration, background,
        n_passes=args.n_passes, fast_inference=False, cache=None, aabb_mode=aabb_mode)
    print(f"fast_inference=False (original):  {fps_orig:6.1f} FPS  ({ms_orig:.2f} ms/frame)")

    print(f"\nSpeedup from cache: {fps_fast_cache/fps_fast_nocache:.2f}x ({ms_fast_nocache - ms_fast_cache:.2f} ms saved)")
    print(f"Speedup from fast_inference: {fps_fast_nocache/fps_orig:.2f}x ({ms_orig - ms_fast_nocache:.2f} ms saved)")
    print(f"Total speedup: {fps_fast_cache/fps_orig:.2f}x ({ms_orig - ms_fast_cache:.2f} ms saved)")

    print("\n" + "="*70)
    print(f"COMPONENT BREAKDOWN ({args.n_profile_iters} iterations)")
    print("="*70)

    components = benchmark_components(
        gaussians, all_cams[0], pipe, ingp, cfg, iteration, background,
        n_iters=args.n_profile_iters, cache=cache, aabb_mode=aabb_mode)

    print(f"Total render:     {components['total_ms']:6.2f} ms")
    print(f"  Rasterizer:     {components['rasterizer_ms']:6.2f} ms ({components['rasterizer_pct']:.1f}%)")
    print(f"  MLP decode:     {components['mlp_ms']:6.2f} ms ({components['mlp_pct']:.1f}%)")
    print(f"  cam2rays:       {components['cam2rays_ms']:6.2f} ms ({components['cam2rays_pct']:.1f}%)")
    print(f"  Preprocess*:    {components['preprocess_ms']:6.2f} ms (cached, not in total)")

    if not args.skip_quality:
        print("\n" + "="*70)
        print(f"QUALITY VERIFICATION ({len(test_cams)} test images)")
        print("="*70)

        quality = verify_quality(
            gaussians, test_cams, pipe, ingp, cfg, iteration, background, cache=cache, aabb_mode=aabb_mode)

        print(f"PSNR (fast_inference=False): {quality['psnr_ref']:.2f} dB")
        print(f"PSNR (fast_inference=True):  {quality['psnr_fast']:.2f} dB")
        print(f"PSNR difference: {abs(quality['psnr_ref'] - quality['psnr_fast']):.4f} dB")
        print(f"Max pixel diff: {quality['max_diff']:.6f}")
        print(f"Mean max diff: {quality['mean_max_diff']:.6f}")

        if quality['max_diff'] < 0.01:
            print("Status: PASS (outputs match)")
        else:
            print("Status: FAIL (outputs differ)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Gaussians: {len(gaussians.get_xyz):,}")
    print(f"Best FPS: {fps_fast_cache:.1f} (fast_inference + cache)")
    print(f"Best ms/frame: {ms_fast_cache:.2f}")
    if not args.skip_quality:
        print(f"PSNR: {quality['psnr_fast']:.2f} dB")


if __name__ == '__main__':
    main()
