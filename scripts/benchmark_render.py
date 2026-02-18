#!/usr/bin/env python3
"""
Comprehensive render benchmarking script.

Profiles individual components and measures quality metrics.

Usage:
    python scripts/benchmark_render.py --model_path outputs/mip_360/bicycle/cat/...
    python scripts/benchmark_render.py --model_path outputs/mip_360/bicycle/cat/... --profile_only
    python scripts/benchmark_render.py --model_path outputs/mip_360/bicycle/cat/... --metrics_only
"""

import os
import sys
import json
import pickle
import time
import glob
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.point_utils import cam2rays
from utils.general_utils import build_rotation, build_H
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


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


def profile_components(gaussians, cam, pipe, ingp, cfg, iteration, background, n_iters=50):
    """Profile individual render pipeline components."""
    beta = cfg.surfel.tg_beta
    H, W = cam.image_height, cam.image_width

    results = {}

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=iteration, cfg=cfg, fast_inference=False)
    torch.cuda.synchronize()

    # 1. Full render (no fast_inference)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=iteration, cfg=cfg, fast_inference=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['full_render_ms'] = sum(times) / len(times) * 1000

    # 2. Fast render (fast_inference=True)
    cache = RenderCache()
    cache.cache_homotrans(gaussians.get_rotation, gaussians.get_scaling, gaussians.get_xyz)

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=iteration, cfg=cfg, fast_inference=True, cache=cache)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['fast_render_ms'] = sum(times) / len(times) * 1000

    # 3. Preprocess (build_rotation + build_H)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rots = build_rotation(gaussians._rotation)
            homotrans = build_H(rots, gaussians.get_scaling, gaussians.get_xyz)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['preprocess_ms'] = sum(times) / len(times) * 1000

    # 4. cam2rays + normalize
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

    # 5. MLP decode only
    feat_dim = 24
    dummy_features = torch.randn(H*W, feat_dim, device='cuda')
    dummy_dirs = F.normalize(torch.randn(H*W, 3, device='cuda'), dim=-1)

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rgb = ingp.rgb_decode(dummy_features, dummy_dirs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    results['mlp_decode_ms'] = sum(times) / len(times) * 1000

    # Estimate rasterizer time
    results['rasterizer_est_ms'] = results['fast_render_ms'] - results['cam2rays_ms'] - results['mlp_decode_ms']

    # Compute FPS
    results['full_fps'] = 1000 / results['full_render_ms']
    results['fast_fps'] = 1000 / results['fast_render_ms']
    results['speedup'] = results['full_render_ms'] / results['fast_render_ms']

    # Percentages
    total = results['fast_render_ms']
    results['cam2rays_pct'] = 100 * results['cam2rays_ms'] / total
    results['mlp_decode_pct'] = 100 * results['mlp_decode_ms'] / total
    results['rasterizer_pct'] = 100 * results['rasterizer_est_ms'] / total

    results['num_pixels'] = H * W
    results['image_size'] = f"{W}x{H}"

    return results


def benchmark_fps_multi_camera(gaussians, cameras, pipe, ingp, cfg, iteration, background,
                               fast_inference=False, cache=None, n_iters=100):
    """Benchmark FPS across multiple cameras."""
    beta = cfg.surfel.tg_beta

    # Warmup
    with torch.no_grad():
        for i in range(10):
            cam = cameras[i % len(cameras)]
            render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=iteration, cfg=cfg, fast_inference=fast_inference, cache=cache)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(n_iters):
            cam = cameras[i % len(cameras)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=iteration, cfg=cfg, fast_inference=fast_inference, cache=cache)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_ms = sum(times) / len(times) * 1000
    fps = 1000 / mean_ms
    return fps, mean_ms


def compute_quality_metrics(gaussians, cameras, pipe, ingp, cfg, iteration, background,
                           fast_inference=False, cache=None):
    """Compute PSNR, SSIM, LPIPS on test set."""
    beta = cfg.surfel.tg_beta

    psnr_list = []
    ssim_list = []
    lpips_list = []

    with torch.no_grad():
        for cam in cameras:
            result = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                           iteration=iteration, cfg=cfg, fast_inference=fast_inference, cache=cache)

            rendered = result['render'].clamp(0, 1)
            gt = cam.original_image.cuda().clamp(0, 1)

            psnr_val = psnr(rendered, gt).mean().item()
            ssim_val = ssim(rendered, gt).mean().item()
            lpips_val = lpips(rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)

    return {
        'psnr': sum(psnr_list) / len(psnr_list),
        'ssim': sum(ssim_list) / len(ssim_list),
        'lpips': sum(lpips_list) / len(lpips_list),
        'per_image_psnr': psnr_list,
        'per_image_ssim': ssim_list,
        'per_image_lpips': lpips_list,
    }


def verify_fast_inference(gaussians, cam, pipe, ingp, cfg, iteration, background, cache):
    """Verify fast_inference produces same output as regular render."""
    beta = cfg.surfel.tg_beta

    with torch.no_grad():
        result_ref = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                           iteration=iteration, cfg=cfg, fast_inference=False)
        result_fast = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                            iteration=iteration, cfg=cfg, fast_inference=True, cache=cache)

    img_ref = result_ref['render'].clamp(0, 1)
    img_fast = result_fast['render'].clamp(0, 1)

    diff = (img_ref - img_fast).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    mse = torch.mean((img_ref - img_fast) ** 2)
    psnr_between = -10 * torch.log10(mse).item() if mse > 0 else float('inf')

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'psnr_between': psnr_between,
        'identical': max_diff < 0.01,  # Allow small floating point differences
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive render benchmarking")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--profile_only', action='store_true', help='Only run component profiling')
    parser.add_argument('--metrics_only', action='store_true', help='Only compute quality metrics')
    parser.add_argument('--n_profile_iters', type=int, default=50, help='Iterations for profiling')
    parser.add_argument('--n_fps_iters', type=int, default=100, help='Iterations for FPS benchmark')
    args = parser.parse_args()

    # Find iteration
    if args.iteration == -1:
        ngp_files = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            args.iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

    print(f"Loading model from {args.model_path}, iteration {args.iteration}")

    # Load training config
    train_args = load_training_config(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    # Load YAML config
    config_yaml_path = os.path.join(args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg = Config(config_yaml_path)
    else:
        cfg = Config(train_args.yaml)

    # Setup models
    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(train_args)
    pipe = pipeline_params.extract(train_args)

    # Load models
    ingp = INGP(cfg, args=train_args).cuda()
    ingp.load_model(args.model_path, args.iteration)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp.set_active_levels(args.iteration)

    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel

    # Setup
    test_cameras = scene.getTestCameras()
    background = torch.zeros(3, device='cuda')

    print(f"\n{'='*70}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Method: {train_args.method}")
    print(f"Kernel: {getattr(train_args, 'kernel', 'gaussian')}")
    print(f"Gaussians: {len(gaussians.get_xyz):,}")
    print(f"Test cameras: {len(test_cameras)}")
    if len(test_cameras) > 0:
        print(f"Image size: {test_cameras[0].image_width}x{test_cameras[0].image_height}")

    # Setup cache for fast_inference
    cache = RenderCache()
    cache.cache_homotrans(gaussians.get_rotation, gaussians.get_scaling, gaussians.get_xyz)

    results = {
        'model_path': args.model_path,
        'iteration': args.iteration,
        'method': train_args.method,
        'kernel': getattr(train_args, 'kernel', 'gaussian'),
        'num_gaussians': len(gaussians.get_xyz),
        'num_test_cameras': len(test_cameras),
    }

    if not args.metrics_only:
        # Verify fast_inference correctness
        print(f"\n{'='*70}")
        print("FAST_INFERENCE VERIFICATION")
        print(f"{'='*70}")

        verify = verify_fast_inference(gaussians, test_cameras[0], pipe, ingp, cfg,
                                       args.iteration, background, cache)

        print(f"Max pixel diff: {verify['max_diff']:.6f}")
        print(f"Mean pixel diff: {verify['mean_diff']:.8f}")
        print(f"PSNR between outputs: {verify['psnr_between']:.2f} dB")
        print(f"Status: {'PASS' if verify['identical'] else 'FAIL'}")
        results['fast_inference_verification'] = verify

        # Component profiling
        print(f"\n{'='*70}")
        print(f"COMPONENT PROFILING ({args.n_profile_iters} iterations)")
        print(f"{'='*70}")

        profile = profile_components(gaussians, test_cameras[0], pipe, ingp, cfg,
                                    args.iteration, background, n_iters=args.n_profile_iters)

        print(f"\nRender times:")
        print(f"  Full render (fast_inference=False): {profile['full_render_ms']:.2f} ms ({profile['full_fps']:.1f} FPS)")
        print(f"  Fast render (fast_inference=True):  {profile['fast_render_ms']:.2f} ms ({profile['fast_fps']:.1f} FPS)")
        print(f"  Speedup: {profile['speedup']:.2f}x")

        print(f"\nComponent breakdown (fast_inference path):")
        print(f"  Rasterizer + other: {profile['rasterizer_est_ms']:.2f} ms ({profile['rasterizer_pct']:.1f}%)")
        print(f"  MLP decode:         {profile['mlp_decode_ms']:.2f} ms ({profile['mlp_decode_pct']:.1f}%)")
        print(f"  cam2rays+normalize: {profile['cam2rays_ms']:.2f} ms ({profile['cam2rays_pct']:.1f}%)")
        print(f"  Preprocess (ref):   {profile['preprocess_ms']:.2f} ms (cached)")

        results['profiling'] = profile

        # Multi-camera FPS benchmark
        print(f"\n{'='*70}")
        print(f"FPS BENCHMARK ({args.n_fps_iters} iterations across {len(test_cameras)} cameras)")
        print(f"{'='*70}")

        fps_full, ms_full = benchmark_fps_multi_camera(
            gaussians, test_cameras, pipe, ingp, cfg, args.iteration, background,
            fast_inference=False, n_iters=args.n_fps_iters)

        fps_fast, ms_fast = benchmark_fps_multi_camera(
            gaussians, test_cameras, pipe, ingp, cfg, args.iteration, background,
            fast_inference=True, cache=cache, n_iters=args.n_fps_iters)

        print(f"Full render: {fps_full:.1f} FPS ({ms_full:.2f} ms/frame)")
        print(f"Fast render: {fps_fast:.1f} FPS ({ms_fast:.2f} ms/frame)")
        print(f"Speedup: {fps_full/fps_fast if fps_fast > fps_full else fps_fast/fps_full:.2f}x")

        results['fps_benchmark'] = {
            'full_fps': fps_full,
            'full_ms': ms_full,
            'fast_fps': fps_fast,
            'fast_ms': ms_fast,
        }

    if not args.profile_only:
        # Quality metrics
        print(f"\n{'='*70}")
        print(f"QUALITY METRICS ({len(test_cameras)} test images)")
        print(f"{'='*70}")

        print("\nComputing metrics with fast_inference=False (reference)...")
        metrics_ref = compute_quality_metrics(
            gaussians, test_cameras, pipe, ingp, cfg, args.iteration, background,
            fast_inference=False)

        print("\nComputing metrics with fast_inference=True...")
        metrics_fast = compute_quality_metrics(
            gaussians, test_cameras, pipe, ingp, cfg, args.iteration, background,
            fast_inference=True, cache=cache)

        print(f"\nResults (fast_inference=False / fast_inference=True):")
        print(f"  PSNR:  {metrics_ref['psnr']:.2f} / {metrics_fast['psnr']:.2f} dB (diff: {abs(metrics_ref['psnr']-metrics_fast['psnr']):.4f})")
        print(f"  SSIM:  {metrics_ref['ssim']:.4f} / {metrics_fast['ssim']:.4f} (diff: {abs(metrics_ref['ssim']-metrics_fast['ssim']):.6f})")
        print(f"  LPIPS: {metrics_ref['lpips']:.4f} / {metrics_fast['lpips']:.4f} (diff: {abs(metrics_ref['lpips']-metrics_fast['lpips']):.6f})")

        results['quality_metrics'] = {
            'reference': metrics_ref,
            'fast_inference': metrics_fast,
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Gaussians: {results['num_gaussians']:,}")
    if 'fps_benchmark' in results:
        print(f"FPS (fast_inference): {results['fps_benchmark']['fast_fps']:.1f}")
    if 'quality_metrics' in results:
        print(f"PSNR: {results['quality_metrics']['reference']['psnr']:.2f} dB")
        print(f"SSIM: {results['quality_metrics']['reference']['ssim']:.4f}")
        print(f"LPIPS: {results['quality_metrics']['reference']['lpips']:.4f}")

    # Save results
    results_path = os.path.join(args.model_path, "benchmark_results.json")
    with open(results_path, 'w') as f:
        # Convert non-serializable types
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float):
                return float(obj)
            elif isinstance(obj, int):
                return int(obj)
            else:
                return obj
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
