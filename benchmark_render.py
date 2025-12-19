#!/usr/bin/env python3
#
# Benchmarking script for rendering pipeline
# Based on eval_render.py but adds detailed timing measurements
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8
from train import merge_cfg_to_args
import time
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Benchmarking script for rendering pipeline")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--yaml", type=str, default="tiny")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "cat", "adaptive", "adaptive_add", "diffuse", "specular", 
                                "diffuse_ngp", "diffuse_offset", "hybrid_SH", "hybrid_SH_raw", 
                                "hybrid_SH_post", "residual_hybrid"],
                        help="Rendering method (must match training)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of coarse levels (cat mode only)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations before timing")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations to benchmark")
    parser.add_argument("--skip_save", action="store_true",
                        help="Skip saving images (faster benchmarking)")
    args = get_combined_args(parser)

    exp_path = args.model_path
    iteration = args.iteration
    yaml_file = args.yaml

    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    # Load model
    print(f"Loading model from: {exp_path}, iteration: {iteration}")
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(exp_path, iteration)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    resolution = dataset.resolution
    print(f'Test resolution: {resolution}')
    
    args.cfg = cfg_model

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    gaussians.XYZ_TYPE = "UV"
    active_levels = ingp_model.set_active_levels(iteration)

    # Get test cameras
    viewpoint_stack = scene.getTestCameras().copy()
    if len(viewpoint_stack) == 0:
        print("No test cameras found, using train cameras")
        viewpoint_stack = scene.getTrainCameras().copy()
    
    num_cameras = len(viewpoint_stack)
    print(f"Benchmarking {num_cameras} cameras")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print(f"Method: {args.method}")
    if args.method == "cat":
        print(f"Hybrid levels: {args.hybrid_levels}")
    print()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i in range(args.warmup):
            cam = viewpoint_stack[i % num_cameras]
            render_pkg = render(cam, gaussians, pipe, background, ingp=ingp_model,
                              beta=beta, iteration=iteration, cfg=cfg_model)
            torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    timings = defaultdict(list)
    
    # Create CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for iter_idx in tqdm(range(args.iterations), desc="Benchmarking"):
            cam_idx = iter_idx % num_cameras
            cam = viewpoint_stack[cam_idx]
            
            # Full render timing
            torch.cuda.synchronize()
            start_event.record()
            render_pkg = render(cam, gaussians, pipe, background, ingp=ingp_model,
                              beta=beta, iteration=iteration, cfg=cfg_model)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)
            timings['total'].append(elapsed_ms)
            
            # Save image if not skipping
            if not args.skip_save and iter_idx < 5:  # Save first 5 for verification
                image = render_pkg["render"]
                test_dir = os.path.join(scene.model_path, 'test', f"ours_{iteration}")
                os.makedirs(test_dir, exist_ok=True)
                renders_dir = os.path.join(test_dir, "renders")
                os.makedirs(renders_dir, exist_ok=True)
                img_name = os.path.join(renders_dir, f"bench_{iter_idx:03d}.png")
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), img_name)

    # Compute statistics
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    for key, values in timings.items():
        values_ms = np.array(values)
        mean_ms = np.mean(values_ms)
        std_ms = np.std(values_ms)
        min_ms = np.min(values_ms)
        max_ms = np.max(values_ms)
        median_ms = np.median(values_ms)
        
        print(f"\n{key.upper()} Timing:")
        print(f"  Mean:   {mean_ms:.3f} ms")
        print(f"  Median: {median_ms:.3f} ms")
        print(f"  Std:    {std_ms:.3f} ms")
        print(f"  Min:    {min_ms:.3f} ms")
        print(f"  Max:    {max_ms:.3f} ms")
        print(f"  FPS:    {1000.0/mean_ms:.2f} frames/sec")
    
    # Per-camera statistics
    print(f"\n{'='*70}")
    print("PER-CAMERA STATISTICS")
    print(f"{'='*70}")
    camera_timings = defaultdict(list)
    for iter_idx in range(args.iterations):
        cam_idx = iter_idx % num_cameras
        camera_timings[cam_idx].append(timings['total'][iter_idx])
    
    for cam_idx in sorted(camera_timings.keys()):
        cam_times = np.array(camera_timings[cam_idx])
        cam_name = viewpoint_stack[cam_idx].image_name if hasattr(viewpoint_stack[cam_idx], 'image_name') else f"cam_{cam_idx}"
        print(f"  Camera {cam_idx:3d} ({cam_name:20s}): "
              f"mean={np.mean(cam_times):6.3f}ms, "
              f"min={np.min(cam_times):6.3f}ms, "
              f"max={np.max(cam_times):6.3f}ms")
    
    print(f"\n{'='*70}")
    print(f"Total cameras benchmarked: {num_cameras}")
    print(f"Total iterations: {args.iterations}")
    print(f"Average time per frame: {np.mean(timings['total']):.3f} ms")
    print(f"Average FPS: {1000.0/np.mean(timings['total']):.2f}")
    print(f"{'='*70}\n")
