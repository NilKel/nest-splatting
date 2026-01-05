#!/usr/bin/env python3
"""
Detailed profiling of the render pipeline to identify bottlenecks.
Breaks down time spent in each stage: preprocessing, rasterizer, post-processing, MLP.
"""

import torch
import time
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
from scene.gaussian_model import GaussianModel
from scene import Scene
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from train import merge_cfg_to_args
from utils.point_utils import depth_to_normal


def profile_render_detailed(viewpoint_camera, pc, pipe, bg_color, ingp, cfg, iteration, num_iters=100):
    """Profile each stage of the render pipeline separately."""

    XYZ_TYPE = cfg.ingp_stage.XYZ_TYPE
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    # Timing accumulators
    times = {
        'preprocessing': [],
        'rasterizer_setup': [],
        'rasterizer_call': [],
        'post_raster_extract': [],
        'post_raster_normal': [],
        'post_raster_depth': [],
        'rays_dir_compute': [],
        'tensor_reshape': [],
        'mlp_decode': [],
        'mask_apply': [],
        'total': [],
    }

    beta = cfg.surfel.tg_beta

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            # Full render for warm-up
            screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            means3D = pc.get_xyz
            means2D = screenspace_points
            opacity = pc.get_opacity
            scales = pc.get_scaling
            rotations = pc.get_rotation

        torch.cuda.synchronize()

        # Profiling loop
        for _ in range(num_iters):
            torch.cuda.synchronize()
            total_start = time.time()

            # ============ PREPROCESSING ============
            torch.cuda.synchronize()
            t0 = time.time()

            screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            means3D = pc.get_xyz
            means2D = screenspace_points
            opacity = pc.get_opacity
            scales = pc.get_scaling
            rotations = pc.get_rotation
            shs = pc.get_features

            # CAT mode setup
            is_cat_mode = ingp is not None and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode
            feat_dim = ingp.active_levels * ingp.level_dim if ingp else 0

            # Get per-Gaussian features
            gaussian_features = None
            if hasattr(pc, '_gaussian_features') and pc._gaussian_features is not None and pc._gaussian_features.numel() > 0:
                gaussian_features = pc._gaussian_features

            # Hashgrid settings
            features_table = None
            offsets = None
            gridrange = None
            levels = 0
            per_level_scale = 1
            base_resolution = 0
            align_corners = False
            interpolation = 0

            if ingp is not None and hasattr(ingp, 'hash_encoding'):
                features_table, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation = ingp.hash_encoding.get_params()
                gridrange = ingp.gridrange
                levels = ingp.active_levels

            # Shape dims for CAT mode
            if is_cat_mode and gaussian_features is not None:
                gf_dim = gaussian_features.shape[1]
                hash_dim = ingp.hashgrid_levels * ingp.level_dim if hasattr(ingp, 'hashgrid_levels') else feat_dim - gf_dim
                shape_dims = torch.tensor([gf_dim, hash_dim, gf_dim + hash_dim], dtype=torch.int32, device="cuda")
            else:
                shape_dims = torch.tensor([0, feat_dim, feat_dim], dtype=torch.int32, device="cuda")

            # Kernel setup
            shapes = None
            kernel_type = 0
            if hasattr(pc, 'kernel_type') and pc.kernel_type == "general" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
                shapes = pc.get_shape
                kernel_type = 3
            elif hasattr(pc, 'kernel_type') and pc.kernel_type == "beta" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
                shapes = pc.get_shape
                kernel_type = 1

            torch.cuda.synchronize()
            times['preprocessing'].append(time.time() - t0)

            # ============ RASTERIZER SETUP ============
            torch.cuda.synchronize()
            t0 = time.time()

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=0,  # CAT mode uses sh_degree=0
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False,
                beta=beta,
                if_contract=False,
                record_transmittance=False,
                max_intersections=0,
            )

            hashgrid_settings = HashGridSettings(
                L=levels,
                S=math.log2(per_level_scale) if per_level_scale > 0 else 0,
                H=base_resolution,
                align_corners=align_corners,
                interpolation=interpolation,
                shape_dims=shape_dims
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

            torch.cuda.synchronize()
            times['rasterizer_setup'].append(time.time() - t0)

            # ============ RASTERIZER CALL ============
            torch.cuda.synchronize()
            t0 = time.time()

            # Prepare features for CAT mode
            if is_cat_mode and gaussian_features is not None:
                colors_precomp = gaussian_features
            else:
                colors_precomp = None

            rendered_image, radii, allmap, transmittance_avg, num_covered_pixels = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs if colors_precomp is None else None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                homotrans=None,
                ap_level=None,
                cov3D_precomp=None,
                features=features_table,
                offsets=offsets,
                gridrange=gridrange,
                render_mode=6 if is_cat_mode else 0,  # 6 = cat mode
                shapes=shapes,
                kernel_type=kernel_type,
                aabb_mode=0,
            )

            torch.cuda.synchronize()
            times['rasterizer_call'].append(time.time() - t0)

            # ============ POST-RASTER: EXTRACT MAPS ============
            torch.cuda.synchronize()
            t0 = time.time()

            render_alpha = allmap[1:2]
            render_normal = allmap[2:5]
            render_depth_median = allmap[5:6]
            render_depth_expected = allmap[0:1]
            render_dist = allmap[6:7]
            render_gs_nums = allmap[7:8]

            torch.cuda.synchronize()
            times['post_raster_extract'].append(time.time() - t0)

            # ============ POST-RASTER: NORMAL TRANSFORM ============
            torch.cuda.synchronize()
            t0 = time.time()

            render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

            torch.cuda.synchronize()
            times['post_raster_normal'].append(time.time() - t0)

            # ============ POST-RASTER: DEPTH PROCESSING ============
            torch.cuda.synchronize()
            t0 = time.time()

            render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
            surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2,0,1)
            surf_normal = surf_normal * (render_alpha).detach()

            torch.cuda.synchronize()
            times['post_raster_depth'].append(time.time() - t0)

            # ============ RAYS DIRECTION COMPUTE ============
            torch.cuda.synchronize()
            t0 = time.time()

            # Compute rays direction for MLP (view-dependent)
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
            c2w = viewpoint_camera.world_view_transform.T.inverse()
            rays_d = (dirs @ c2w[:3, :3].T).reshape(-1, 3)
            rays_dir = torch.nn.functional.normalize(rays_d, dim=-1)

            torch.cuda.synchronize()
            times['rays_dir_compute'].append(time.time() - t0)

            # ============ TENSOR RESHAPE ============
            torch.cuda.synchronize()
            t0 = time.time()

            render_mask = (render_alpha > 0)
            feat_dim_actual = rendered_image.shape[0]
            features_flat = rendered_image.view(feat_dim_actual, -1).permute(1, 0)  # (H*W, feat_dim)

            torch.cuda.synchronize()
            times['tensor_reshape'].append(time.time() - t0)

            # ============ MLP DECODE ============
            torch.cuda.synchronize()
            t0 = time.time()

            rgb = ingp.rgb_decode(features_flat, rays_dir)  # (H*W, 3)

            torch.cuda.synchronize()
            times['mlp_decode'].append(time.time() - t0)

            # ============ MASK APPLY & RESHAPE ============
            torch.cuda.synchronize()
            t0 = time.time()

            rgb = rgb.view(H, W, -1).permute(2, 0, 1)  # (3, H, W)
            rgb = rgb * render_mask

            torch.cuda.synchronize()
            times['mask_apply'].append(time.time() - t0)

            # ============ TOTAL ============
            torch.cuda.synchronize()
            times['total'].append(time.time() - total_start)

    return times


def main():
    parser = ArgumentParser(description="Profile render pipeline stages")
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

    # Override kernel type if specified
    if args.kernel != "gaussian":
        gaussians.kernel_type = args.kernel
        print(f"[PROFILE] Kernel type set to: {args.kernel}")

    num_gaussians = len(gaussians.get_xyz)
    print(f"[PROFILE] Gaussians: {num_gaussians:,}")
    print(f"[PROFILE] Resolution: {scene.getTestCameras()[0].image_width}x{scene.getTestCameras()[0].image_height}")

    # Get test camera
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras!")
        return

    viewpoint = test_cameras[0]
    bg = torch.zeros(3, device="cuda")

    print(f"\n[PROFILE] Running {args.num_iters} iterations...")
    times = profile_render_detailed(viewpoint, gaussians, pipe, bg, ingp_model, cfg_model, iteration, args.num_iters)

    # Compute statistics
    print("\n" + "="*70)
    print("RENDER PIPELINE PROFILING RESULTS")
    print("="*70)

    total_mean = sum(times['total']) / len(times['total']) * 1000

    print(f"\n{'Stage':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'% of Total':>12}")
    print("-"*70)

    for stage, stage_times in times.items():
        if stage == 'total':
            continue
        mean_ms = sum(stage_times) / len(stage_times) * 1000
        std_ms = (sum((t*1000 - mean_ms)**2 for t in stage_times) / len(stage_times)) ** 0.5
        pct = (mean_ms / total_mean) * 100
        print(f"{stage:<25} {mean_ms:>12.3f} {std_ms:>12.3f} {pct:>11.1f}%")

    print("-"*70)
    total_std = (sum((t*1000 - total_mean)**2 for t in times['total']) / len(times['total'])) ** 0.5
    print(f"{'TOTAL':<25} {total_mean:>12.3f} {total_std:>12.3f} {'100.0%':>12}")

    fps = 1000 / total_mean
    print(f"\n{'FPS':<25} {fps:>12.2f}")

    # Group summary
    print("\n" + "="*70)
    print("GROUPED SUMMARY")
    print("="*70)

    preprocess = sum(times['preprocessing']) / len(times['preprocessing']) * 1000
    raster_setup = sum(times['rasterizer_setup']) / len(times['rasterizer_setup']) * 1000
    raster_call = sum(times['rasterizer_call']) / len(times['rasterizer_call']) * 1000
    post_extract = sum(times['post_raster_extract']) / len(times['post_raster_extract']) * 1000
    post_normal = sum(times['post_raster_normal']) / len(times['post_raster_normal']) * 1000
    post_depth = sum(times['post_raster_depth']) / len(times['post_raster_depth']) * 1000
    rays_compute = sum(times['rays_dir_compute']) / len(times['rays_dir_compute']) * 1000
    tensor_reshape = sum(times['tensor_reshape']) / len(times['tensor_reshape']) * 1000
    mlp = sum(times['mlp_decode']) / len(times['mlp_decode']) * 1000
    mask = sum(times['mask_apply']) / len(times['mask_apply']) * 1000

    raster_total = raster_setup + raster_call
    post_total = post_extract + post_normal + post_depth
    mlp_overhead = rays_compute + tensor_reshape + mask

    print(f"\n{'Component':<30} {'Time (ms)':>12} {'% of Total':>12}")
    print("-"*60)
    print(f"{'Preprocessing (Python)':<30} {preprocess:>12.3f} {preprocess/total_mean*100:>11.1f}%")
    print(f"{'Rasterizer (CUDA)':<30} {raster_total:>12.3f} {raster_total/total_mean*100:>11.1f}%")
    print(f"  - Setup                      {raster_setup:>12.3f}")
    print(f"  - Kernel call                {raster_call:>12.3f}")
    print(f"{'Post-raster processing':<30} {post_total:>12.3f} {post_total/total_mean*100:>11.1f}%")
    print(f"  - Extract maps               {post_extract:>12.3f}")
    print(f"  - Normal transform           {post_normal:>12.3f}")
    print(f"  - Depth/surface normal       {post_depth:>12.3f}")
    print(f"{'MLP decode':<30} {mlp:>12.3f} {mlp/total_mean*100:>11.1f}%")
    print(f"{'MLP overhead (rays, reshape)':<30} {mlp_overhead:>12.3f} {mlp_overhead/total_mean*100:>11.1f}%")
    print(f"  - Rays direction             {rays_compute:>12.3f}")
    print(f"  - Tensor reshape             {tensor_reshape:>12.3f}")
    print(f"  - Mask apply                 {mask:>12.3f}")
    print("-"*60)
    print(f"{'TOTAL':<30} {total_mean:>12.3f} {'100.0%':>12}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
