#!/usr/bin/env python3
"""
Benchmark baseline 2DGS rasterizer (render_mode=0) FPS.

Loads the trained point cloud (geometry + opacity) but uses random SH values,
since we only care about rendering speed, not visual quality.

Usage:
    python scripts/circle/benchmark_baseline.py \
        --model_path outputs/.../model --source_path data/.../scene \
        --radius_scale 1.3 --tilt 0.15 --num_views 120
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser, Namespace
from scripts.circle.circle_cam import (
    load_training_config, estimate_scene_params, make_circle_camera, compute_orbit_params,
)
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from hash_encoder.config import Config
import glob as glob_mod
import pickle


def main():
    parser = ArgumentParser(description="Benchmark baseline 2DGS (render_mode=0) FPS")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_views", type=int, default=120)
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--tilt", type=float, default=0.0)
    eval_args = parser.parse_args()

    # Load training config
    args = load_training_config(eval_args.model_path)
    args.model_path = eval_args.model_path
    args.eval = True
    if eval_args.source_path:
        args.source_path = eval_args.source_path

    # Find iteration
    iteration = eval_args.iteration
    if iteration == -1:
        ngp_files = glob_mod.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)

    # Load gaussians (geometry + opacity only)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Randomize SH so we don't need INGP
    with torch.no_grad():
        gaussians._features_dc = torch.randn_like(gaussians._features_dc) * 0.1
        gaussians._features_rest = torch.randn_like(gaussians._features_rest) * 0.01

    # Force baseline mode: no appearance levels, no gaussian features
    gaussians._appearance_level = None
    gaussians._gaussian_features = None
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    if dead_mask.sum().item() > 0:
        valid_mask = ~dead_mask
        gaussians._xyz = gaussians._xyz[valid_mask]
        gaussians._features_dc = gaussians._features_dc[valid_mask]
        gaussians._features_rest = gaussians._features_rest[valid_mask]
        gaussians._opacity = gaussians._opacity[valid_mask]
        gaussians._scaling = gaussians._scaling[valid_mask]
        gaussians._rotation = gaussians._rotation[valid_mask]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask]

    aabb_mode = getattr(args, 'aabb', '2dgs')
    print(f"[MODEL] {len(gaussians.get_xyz):,} Gaussians, kernel={gaussians.kernel_type}, aabb={aabb_mode}")

    # Compute orbit
    orbit = compute_orbit_params(scene, radius_scale=eval_args.radius_scale,
                                  elevation=eval_args.elevation, tilt=eval_args.tilt)
    background = torch.zeros(3, device="cuda")

    # Warmup
    warmup_cam = make_circle_camera(0.0, orbit['orbit_center'], orbit['orbit_radius'],
                                     orbit['up'], orbit['look_target'], orbit['ref_camera'])
    with torch.no_grad():
        render(warmup_cam, gaussians, pipe, background, ingp=None,
               beta=cfg.surfel.tg_beta, iteration=iteration, cfg=cfg,
               aabb_mode=aabb_mode)
    torch.cuda.synchronize()

    # Benchmark
    render_times = []
    with torch.no_grad():
        for i in range(eval_args.num_views):
            progress = i / eval_args.num_views
            cam = make_circle_camera(progress, orbit['orbit_center'], orbit['orbit_radius'],
                                      orbit['up'], orbit['look_target'], orbit['ref_camera'])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            render(cam, gaussians, pipe, background, ingp=None,
                   beta=cfg.surfel.tg_beta, iteration=iteration, cfg=cfg,
                   aabb_mode=aabb_mode)
            torch.cuda.synchronize()
            render_times.append(time.perf_counter() - t0)

    avg_ms = np.mean(render_times) * 1000
    avg_fps = 1000.0 / avg_ms
    print(f"[BENCHMARK] Baseline 2DGS: {avg_ms:.1f} ms ({avg_fps:.1f} FPS) over {len(render_times)} frames")


if __name__ == "__main__":
    main()
