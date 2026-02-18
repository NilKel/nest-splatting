#!/usr/bin/env python3
"""
Profile rendering pipeline to benchmark MLP, rasterizer, and full render time.

Usage:
    python scripts/profile_render.py --model_path outputs/mip_360/treehill/cat/...
    python scripts/profile_render.py --model_path outputs/mip_360/treehill/cat/... --detailed
"""

import os
import sys
import json
import pickle
import time
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
import glob
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


class RenderProfiler:
    """Profiler that wraps render components to measure timing."""

    def __init__(self, gaussians, pipe, ingp, cfg_model, iteration, skybox=None, background_mode="none"):
        self.gaussians = gaussians
        self.pipe = pipe
        self.ingp = ingp
        self.cfg_model = cfg_model
        self.iteration = iteration
        self.skybox = skybox
        self.background_mode = background_mode
        self.beta = cfg_model.surfel.tg_beta

        # Timing storage
        self.timings = {
            'preprocess': [],
            'mlp_hash': [],
            'rasterizer': [],
            'postprocess': [],
            'total': [],
        }

    def profile_render(self, camera, background, detailed=False):
        """
        Profile a single render call, measuring time for each component.

        Returns dict with timing breakdown.
        """
        torch.cuda.synchronize()
        total_start = time.perf_counter()

        pc = self.gaussians
        cfg = self.cfg_model
        ingp = self.ingp
        pipe = self.pipe
        beta = self.beta
        iteration = self.iteration

        # ============ PREPROCESS ============
        torch.cuda.synchronize()
        preprocess_start = time.perf_counter()

        XYZ_TYPE = cfg.ingp_stage.XYZ_TYPE
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda")

        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation

        torch.cuda.synchronize()
        preprocess_time = time.perf_counter() - preprocess_start

        # ============ MLP / HASH ENCODING ============
        torch.cuda.synchronize()
        mlp_start = time.perf_counter()

        # Determine if we use hash_in_CUDA or Python-side hash query
        hash_in_CUDA = True
        if ingp is None:
            hash_in_CUDA = False
        elif iteration < cfg.ingp_stage.switch_iter:
            hash_in_CUDA = False

        is_cat_mode = hash_in_CUDA and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode
        hybrid_levels = ingp.hybrid_levels if is_cat_mode else 0

        # Setup hash parameters
        if hash_in_CUDA:
            output_dim = ingp.levels * ingp.level_dim
            shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")

            if hasattr(ingp, 'hashgrid_disabled') and ingp.hashgrid_disabled:
                features = torch.zeros((1, ingp.level_dim), device="cuda")
                offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
                gridrange = ingp.gridrange
                per_level_scale = 1
                base_resolution = 0
                align_corners = False
                interpolation = 0
                levels = (ingp.levels << 16) | (0 << 8) | ingp.hybrid_levels
            else:
                features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation = ingp.hash_encoding.get_params()
                gridrange = ingp.gridrange

                if is_cat_mode:
                    hash_levels = ingp.levels - hybrid_levels
                    levels = (ingp.levels << 16) | (hash_levels << 8) | hybrid_levels
        else:
            # No hash in CUDA - setup defaults
            shape_dims = torch.tensor([0, 0, 3], dtype=torch.int32, device="cuda")
            features = None
            offsets = None
            gridrange = None
            levels = 0
            per_level_scale = 1
            base_resolution = 0
            align_corners = False
            interpolation = 0

        torch.cuda.synchronize()
        mlp_time = time.perf_counter() - mlp_start

        # ============ RASTERIZER ============
        torch.cuda.synchronize()
        raster_start = time.perf_counter()

        # Cat mode: get per-Gaussian features
        colors_precomp = None
        shs = None
        render_mode = 0

        if is_cat_mode:
            gaussian_features = pc.get_gaussian_features
            colors_precomp = gaussian_features
            render_mode = 1

            decompose_flag = 0
            levels = (decompose_flag << 24) | levels

            gs_dim = hybrid_levels * ingp.level_dim
            hs_dim = (ingp.levels - hybrid_levels) * ingp.level_dim
            os_dim = ingp.levels * ingp.level_dim
            shape_dims = torch.tensor([gs_dim, hs_dim, os_dim], dtype=torch.int32, device="cuda")
        else:
            shs = pc.get_features

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=pc.active_sh_degree if not is_cat_mode else 0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            beta=beta,
            if_contract=False,
            record_transmittance=False,
            max_intersections=0,
            detach_hash_grad=False,
        )

        hashgrid_settings = HashGridSettings(
            L=levels,
            S=math.log2(per_level_scale) if per_level_scale > 0 else 0,
            H=base_resolution,
            align_corners=align_corners,
            interpolation=interpolation,
            shape_dims=shape_dims,
            aa=0.0,
            aa_threshold=0.01
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

        # Get kernel params
        shapes = None
        kernel_type = 0
        if hasattr(pc, 'kernel_type') and pc.kernel_type == "beta" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
            shapes = pc.get_shape
            kernel_type = 1
        elif hasattr(pc, 'kernel_type') and pc.kernel_type == "beta_scaled" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
            shapes = pc.get_shape
            kernel_type = 4

        # Run rasterizer
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            homotrans=None,
            ap_level=None,
            cov3D_precomp=None,
            features=features,
            offsets=offsets,
            gridrange=gridrange,
            render_mode=render_mode,
            shapes=shapes,
            kernel_type=kernel_type,
            aabb_mode=0,
        )

        torch.cuda.synchronize()
        raster_time = time.perf_counter() - raster_start

        # ============ POSTPROCESS ============
        torch.cuda.synchronize()
        postprocess_start = time.perf_counter()

        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        torch.cuda.synchronize()
        postprocess_time = time.perf_counter() - postprocess_start

        torch.cuda.synchronize()
        total_time = time.perf_counter() - total_start

        return {
            'preprocess': preprocess_time * 1000,
            'mlp_hash': mlp_time * 1000,
            'rasterizer': raster_time * 1000,
            'postprocess': postprocess_time * 1000,
            'total': total_time * 1000,
            'rendered_image': rendered_image,
        }

    def benchmark(self, cameras, num_warmup=10, num_iters=100, detailed=False):
        """Run benchmark over multiple cameras."""
        background = torch.zeros(3, device="cuda")

        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        with torch.no_grad():
            for i in range(num_warmup):
                cam = cameras[i % len(cameras)]
                _ = self.profile_render(cam, background, detailed=detailed)

        # Benchmark
        print(f"Benchmarking ({num_iters} iterations)...")
        results = {k: [] for k in ['preprocess', 'mlp_hash', 'rasterizer', 'postprocess', 'total']}

        with torch.no_grad():
            for i in range(num_iters):
                cam = cameras[i % len(cameras)]
                timings = self.profile_render(cam, background, detailed=detailed)
                for k, v in timings.items():
                    if k != 'rendered_image':
                        results[k].append(v)

        return results


def main():
    parser = ArgumentParser(description="Profile rendering pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-iteration timings")
    parser.add_argument("--compare_full_render", action="store_true", help="Also run the full render() for comparison")

    args = parser.parse_args()

    # Find iteration
    if args.iteration == -1:
        ngp_files = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            args.iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

    print(f"\n{'='*60}")
    print(f"RENDER PIPELINE PROFILER")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")

    # Load model
    train_args = load_training_config(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    config_yaml_path = os.path.join(args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(train_args.yaml)

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(train_args)
    pipe = pipeline_params.extract(train_args)

    # Load INGP
    ingp_model = INGP(cfg_model, args=train_args).to('cuda')
    ingp_model.load_model(args.model_path, args.iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(args.iteration)

    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel

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
            gaussians._gaussian_features = gaussians._gaussian_features[valid_mask]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask]

    num_gaussians = len(gaussians.get_xyz)
    print(f"Gaussians: {num_gaussians:,}")

    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        cameras = scene.getTrainCameras()
    print(f"Cameras: {len(cameras)}")

    if len(cameras) > 0:
        H, W = cameras[0].image_height, cameras[0].image_width
        print(f"Resolution: {W}x{H}")

    # Run profiler
    profiler = RenderProfiler(gaussians, pipe, ingp_model, cfg_model, args.iteration)
    results = profiler.benchmark(cameras, num_warmup=args.num_warmup, num_iters=args.num_iters, detailed=args.detailed)

    # Print results
    print(f"\n{'='*60}")
    print(f"PROFILING RESULTS (ms per frame)")
    print(f"{'='*60}")

    import numpy as np
    for stage, times in results.items():
        times = np.array(times)
        print(f"{stage:15s}: mean={times.mean():7.2f}  std={times.std():6.2f}  min={times.min():6.2f}  max={times.max():6.2f}")

    # Compute breakdown
    total_mean = np.mean(results['total'])
    print(f"\n{'='*60}")
    print(f"BREAKDOWN (% of total)")
    print(f"{'='*60}")
    for stage in ['preprocess', 'mlp_hash', 'rasterizer', 'postprocess']:
        pct = np.mean(results[stage]) / total_mean * 100
        print(f"{stage:15s}: {pct:5.1f}%")

    fps = 1000.0 / total_mean
    print(f"\n{'='*60}")
    print(f"FPS: {fps:.1f}")
    print(f"{'='*60}")

    # Optionally compare with full render()
    if args.compare_full_render:
        print(f"\nComparing with full render() function...")
        background = torch.zeros(3, device="cuda")
        beta = cfg_model.surfel.tg_beta
        background_mode = getattr(train_args, 'background', 'none')
        if background_mode is None:
            background_mode = 'none'

        # Warmup
        with torch.no_grad():
            for i in range(args.num_warmup):
                cam = cameras[i % len(cameras)]
                _ = render(cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
                          iteration=args.iteration, cfg=cfg_model, background_mode=background_mode)
                torch.cuda.synchronize()

        # Benchmark
        full_times = []
        with torch.no_grad():
            for i in range(args.num_iters):
                cam = cameras[i % len(cameras)]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = render(cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
                          iteration=args.iteration, cfg=cfg_model, background_mode=background_mode)
                torch.cuda.synchronize()
                full_times.append((time.perf_counter() - t0) * 1000)

        full_times = np.array(full_times)
        print(f"Full render(): mean={full_times.mean():.2f}ms  FPS={1000/full_times.mean():.1f}")


if __name__ == "__main__":
    main()
