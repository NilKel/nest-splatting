#!/usr/bin/env python3
"""
Simple profiling of the render pipeline by timing the actual render function
and its internal components.
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene import Scene
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from train import merge_cfg_to_args


def profile_render(viewpoint, gaussians, pipe, bg, ingp, cfg, iteration, num_iters=100):
    """Profile the full render and MLP separately."""

    beta = cfg.surfel.tg_beta
    H, W = viewpoint.image_height, viewpoint.image_width

    times = {
        'full_render': [],
        'render_no_mlp': [],
        'mlp_only': [],
    }

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg)
        torch.cuda.synchronize()

        # Profile full render
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.time()
            result = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                          iteration=iteration, cfg=cfg)
            torch.cuda.synchronize()
            times['full_render'].append(time.time() - t0)

        # Profile render with skip_mlp
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.time()
            result = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                          iteration=iteration, cfg=cfg, skip_mlp=True)
            torch.cuda.synchronize()
            times['render_no_mlp'].append(time.time() - t0)

        # Compute MLP time
        full_mean = sum(times['full_render']) / len(times['full_render'])
        no_mlp_mean = sum(times['render_no_mlp']) / len(times['render_no_mlp'])
        times['mlp_only'] = [full_mean - no_mlp_mean] * num_iters

    return times


def profile_mlp_detailed(viewpoint, gaussians, pipe, bg, ingp, cfg, iteration, num_iters=100):
    """Profile MLP and related operations in detail."""

    beta = cfg.surfel.tg_beta
    H, W = viewpoint.image_height, viewpoint.image_width

    times = {
        'rays_compute': [],
        'tensor_reshape': [],
        'mlp_forward': [],
        'output_reshape': [],
    }

    with torch.no_grad():
        # First do a render to get the feature tensor
        result = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                       iteration=iteration, cfg=cfg, skip_mlp=True)

        # Get the raw rasterizer output (features)
        # We need to manually get the intermediate feature tensor
        # For now, let's create a dummy feature tensor of the right size
        feat_dim = ingp.active_levels * ingp.level_dim
        features = torch.randn(H * W, feat_dim, device='cuda')

        # Warm-up
        import math
        tanfovx = math.tan(viewpoint.FoVx * 0.5)
        tanfovy = math.tan(viewpoint.FoVy * 0.5)

        for _ in range(10):
            # Rays direction
            i, j = torch.meshgrid(
                torch.arange(W, device='cuda'),
                torch.arange(H, device='cuda'),
                indexing='xy'
            )
            focal_x = W / (2 * tanfovx)
            focal_y = H / (2 * tanfovy)
            dirs = torch.stack([
                (i - W / 2) / focal_x,
                (j - H / 2) / focal_y,
                torch.ones_like(i)
            ], dim=-1)
            c2w = viewpoint.world_view_transform.T.inverse()
            rays_d = (dirs @ c2w[:3, :3].T).reshape(-1, 3)
            rays_dir = torch.nn.functional.normalize(rays_d, dim=-1)

            # MLP
            rgb = ingp.rgb_decode(features, rays_dir)

        torch.cuda.synchronize()

        # Profile each component
        for _ in range(num_iters):
            # Rays direction compute
            torch.cuda.synchronize()
            t0 = time.time()

            i, j = torch.meshgrid(
                torch.arange(W, device='cuda'),
                torch.arange(H, device='cuda'),
                indexing='xy'
            )
            focal_x = W / (2 * tanfovx)
            focal_y = H / (2 * tanfovy)
            dirs = torch.stack([
                (i - W / 2) / focal_x,
                (j - H / 2) / focal_y,
                torch.ones_like(i)
            ], dim=-1)
            c2w = viewpoint.world_view_transform.T.inverse()
            rays_d = (dirs @ c2w[:3, :3].T).reshape(-1, 3)
            rays_dir = torch.nn.functional.normalize(rays_d, dim=-1)

            torch.cuda.synchronize()
            times['rays_compute'].append(time.time() - t0)

            # MLP forward
            torch.cuda.synchronize()
            t0 = time.time()

            rgb = ingp.rgb_decode(features, rays_dir)

            torch.cuda.synchronize()
            times['mlp_forward'].append(time.time() - t0)

            # Output reshape
            torch.cuda.synchronize()
            t0 = time.time()

            rgb_out = rgb.view(H, W, -1).permute(2, 0, 1)

            torch.cuda.synchronize()
            times['output_reshape'].append(time.time() - t0)

    return times


def main():
    parser = ArgumentParser(description="Simple profile of render pipeline")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "beta", "flex", "general"])

    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    print(f"\n[PROFILE] Loading model from: {args.model_path}")

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
        print(f"[PROFILE] Kernel type set to: {args.kernel}")

    num_gaussians = len(gaussians.get_xyz)
    print(f"[PROFILE] Gaussians: {num_gaussians:,}")

    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras!")
        return

    viewpoint = test_cameras[0]
    H, W = viewpoint.image_height, viewpoint.image_width
    print(f"[PROFILE] Resolution: {W}x{H}")

    bg = torch.zeros(3, device="cuda")

    # Profile render
    print(f"\n[PROFILE] Profiling render ({args.num_iters} iterations)...")
    times = profile_render(viewpoint, gaussians, pipe, bg, ingp_model, cfg_model, iteration, args.num_iters)

    full_mean = sum(times['full_render']) / len(times['full_render']) * 1000
    no_mlp_mean = sum(times['render_no_mlp']) / len(times['render_no_mlp']) * 1000
    mlp_mean = full_mean - no_mlp_mean

    print("\n" + "="*60)
    print("RENDER PIPELINE BREAKDOWN")
    print("="*60)
    print(f"\n{'Component':<30} {'Time (ms)':>12} {'% Total':>10}")
    print("-"*60)
    print(f"{'Full render':<30} {full_mean:>12.3f} {'100.0%':>10}")
    print(f"{'  Rasterizer + post-process':<30} {no_mlp_mean:>12.3f} {no_mlp_mean/full_mean*100:>9.1f}%")
    print(f"{'  MLP decode (computed)':<30} {mlp_mean:>12.3f} {mlp_mean/full_mean*100:>9.1f}%")
    print("-"*60)
    print(f"{'FPS (full render)':<30} {1000/full_mean:>12.2f}")
    print(f"{'FPS (skip MLP)':<30} {1000/no_mlp_mean:>12.2f}")

    # Profile MLP in detail
    print(f"\n[PROFILE] Profiling MLP details ({args.num_iters} iterations)...")
    mlp_times = profile_mlp_detailed(viewpoint, gaussians, pipe, bg, ingp_model, cfg_model, iteration, args.num_iters)

    rays_mean = sum(mlp_times['rays_compute']) / len(mlp_times['rays_compute']) * 1000
    mlp_fwd_mean = sum(mlp_times['mlp_forward']) / len(mlp_times['mlp_forward']) * 1000
    reshape_mean = sum(mlp_times['output_reshape']) / len(mlp_times['output_reshape']) * 1000

    print("\n" + "="*60)
    print("MLP DETAILED BREAKDOWN")
    print("="*60)
    print(f"\n{'Operation':<30} {'Time (ms)':>12}")
    print("-"*60)
    print(f"{'Rays direction compute':<30} {rays_mean:>12.3f}")
    print(f"{'MLP forward pass':<30} {mlp_fwd_mean:>12.3f}")
    print(f"{'Output reshape':<30} {reshape_mean:>12.3f}")
    print("-"*60)
    print(f"{'Total MLP overhead':<30} {rays_mean + mlp_fwd_mean + reshape_mean:>12.3f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    raster_pct = no_mlp_mean / full_mean * 100
    mlp_pct = mlp_mean / full_mean * 100

    print(f"\nRasterizer dominates: {raster_pct:.1f}% of render time")
    print(f"MLP decode: {mlp_pct:.1f}% of render time")
    print(f"  - Rays compute: {rays_mean:.3f} ms")
    print(f"  - MLP forward: {mlp_fwd_mean:.3f} ms")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
