#!/usr/bin/env python3
"""
Detailed rendering pipeline profiling with fine-grained timing.
Profiles each component: hashgrid query, rasterization, MLP, background addition.
"""

import torch
import time
import os
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as torch_F

from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from train import merge_cfg_to_args
from utils.point_utils import cam2rays


def detailed_render_profiling(scene, gaussians, pipe, background, ingp_model, beta, iteration, cfg_model, 
                               cameras, num_warmup=10, num_benchmark=100):
    """
    Profile each component of the rendering pipeline separately.
    """
    
    if len(cameras) == 0:
        print("No cameras available for profiling")
        return None
    
    viewpoint = cameras[0]
    H, W = viewpoint.image_height, viewpoint.image_width
    
    print(f"\n{'='*70}")
    print(f"DETAILED RENDERING PROFILING")
    print(f"{'='*70}")
    print(f"Resolution: {W}x{H}")
    print(f"Number of Gaussians: {len(gaussians.get_xyz):,}")
    
    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = render(viewpoint, gaussians, pipe, background, ingp=ingp_model,
                      beta=beta, iteration=iteration, cfg=cfg_model)
            torch.cuda.synchronize()
    
    results = {}
    
    # 1. Profile hashgrid encoding (if applicable)
    if ingp_model is not None and hasattr(ingp_model, 'hash_encoding') and ingp_model.hash_encoding is not None:
        print(f"\n1. Profiling hashgrid encoding...")
        xyz = gaussians.get_xyz
        
        times = []
        for _ in tqdm(range(num_benchmark), desc="Hash encoding"):
            torch.cuda.synchronize()
            start = time.time()
            _ = ingp_model._encode_3D(xyz)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        results['hash_encoding'] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'median_ms': np.median(times) * 1000,
        }
        print(f"   Mean: {results['hash_encoding']['mean_ms']:.2f} ms (±{results['hash_encoding']['std_ms']:.2f} ms)")
    
    # 2. Profile rasterization (CUDA kernel) - this includes alpha blending
    print(f"\n2. Profiling rasterization + alpha blending...")
    
    # We'll render and time just the rasterizer call
    # This is tricky because we need to isolate just the rasterization
    # Let's do a full render and subtract the MLP time
    
    full_times = []
    for _ in tqdm(range(num_benchmark), desc="Full render"):
        torch.cuda.synchronize()
        start = time.time()
        render_pkg = render(viewpoint, gaussians, pipe, background, ingp=ingp_model,
                           beta=beta, iteration=iteration, cfg=cfg_model)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        full_times.append(elapsed)
    
    full_times = np.array(full_times)
    results['full_render'] = {
        'mean_ms': np.mean(full_times) * 1000,
        'std_ms': np.std(full_times) * 1000,
        'median_ms': np.median(full_times) * 1000,
    }
    print(f"   Full render mean: {results['full_render']['mean_ms']:.2f} ms")
    
    # 3. Profile MLP decoding
    if ingp_model is not None and ingp_model.mlp_rgb is not None:
        print(f"\n3. Profiling MLP decoding...")
        
        feat_dim = ingp_model.feat_dim
        num_pixels = H * W
        
        # Get realistic features from a render
        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, background, ingp=ingp_model,
                               beta=beta, iteration=iteration, cfg=cfg_model)
        
        # Get view directions
        rays_d, rays_o = cam2rays(viewpoint)
        ray_unit = torch_F.normalize(rays_d, dim=-1).float().detach()
        
        # Create dummy features (realistic size)
        dummy_features = torch.randn(num_pixels, feat_dim, device='cuda')
        
        times = []
        for _ in tqdm(range(num_benchmark), desc="MLP decoding"):
            torch.cuda.synchronize()
            start = time.time()
            _ = ingp_model.rgb_decode(dummy_features, ray_unit)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        results['mlp_decoding'] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'median_ms': np.median(times) * 1000,
        }
        print(f"   Mean: {results['mlp_decoding']['mean_ms']:.2f} ms (±{results['mlp_decoding']['std_ms']:.2f} ms)")
    
    # 4. Profile background addition (simple tensor operation)
    print(f"\n4. Profiling background addition...")
    
    # Create dummy alpha and image
    dummy_alpha = torch.rand(1, H, W, device='cuda')
    dummy_image = torch.rand(3, H, W, device='cuda')
    bg = background.unsqueeze(-1).unsqueeze(-1)
    
    times = []
    for _ in tqdm(range(num_benchmark), desc="Background addition"):
        torch.cuda.synchronize()
        start = time.time()
        _ = dummy_image + (1.0 - dummy_alpha) * bg
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    results['background_addition'] = {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'median_ms': np.median(times) * 1000,
    }
    print(f"   Mean: {results['background_addition']['mean_ms']:.2f} ms (±{results['background_addition']['std_ms']:.2f} ms)")
    
    # Calculate rasterization time (full - mlp - bg)
    raster_time = results['full_render']['mean_ms']
    if 'mlp_decoding' in results:
        raster_time -= results['mlp_decoding']['mean_ms']
    if 'background_addition' in results:
        raster_time -= results['background_addition']['mean_ms']
    
    results['rasterization_alpha_blend'] = {
        'mean_ms': raster_time,
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TIMING BREAKDOWN SUMMARY")
    print(f"{'='*70}")
    
    total_accounted = 0
    
    if 'hash_encoding' in results:
        pct = (results['hash_encoding']['mean_ms'] / results['full_render']['mean_ms']) * 100
        print(f"1. Hashgrid encoding:        {results['hash_encoding']['mean_ms']:6.2f} ms  ({pct:5.1f}%)")
        total_accounted += results['hash_encoding']['mean_ms']
    
    if 'rasterization_alpha_blend' in results:
        pct = (results['rasterization_alpha_blend']['mean_ms'] / results['full_render']['mean_ms']) * 100
        print(f"2. Rasterization + Alpha:    {results['rasterization_alpha_blend']['mean_ms']:6.2f} ms  ({pct:5.1f}%)")
        total_accounted += results['rasterization_alpha_blend']['mean_ms']
    
    if 'mlp_decoding' in results:
        pct = (results['mlp_decoding']['mean_ms'] / results['full_render']['mean_ms']) * 100
        print(f"3. MLP decoding:             {results['mlp_decoding']['mean_ms']:6.2f} ms  ({pct:5.1f}%)")
        total_accounted += results['mlp_decoding']['mean_ms']
    
    if 'background_addition' in results:
        pct = (results['background_addition']['mean_ms'] / results['full_render']['mean_ms']) * 100
        print(f"4. Background addition:      {results['background_addition']['mean_ms']:6.2f} ms  ({pct:5.1f}%)")
        total_accounted += results['background_addition']['mean_ms']
    
    overhead = results['full_render']['mean_ms'] - total_accounted
    overhead_pct = (overhead / results['full_render']['mean_ms']) * 100
    
    print(f"{'-'*70}")
    print(f"   Total accounted:          {total_accounted:6.2f} ms")
    print(f"   Overhead/other:           {overhead:6.2f} ms  ({overhead_pct:5.1f}%)")
    print(f"   Full render:              {results['full_render']['mean_ms']:6.2f} ms  (100.0%)")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Detailed rendering profiling")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="tiny")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "cat", "adaptive", "adaptive_add", "diffuse", 
                                "specular", "diffuse_ngp", "diffuse_offset"],
                        help="Rendering method (must match training)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--num_benchmark", type=int, default=100,
                        help="Number of benchmark iterations")
    
    args = get_combined_args(parser)
    
    print(f"\n{'='*70}")
    print(f"LOADING MODEL")
    print(f"{'='*70}")
    print(f"Model path: {args.model_path}")
    print(f"Method: {args.method}")
    
    # Load config
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)
    
    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(args.model_path, args.iteration)
    
    # Load dataset and scene
    dataset, pipe = model.extract(args), pipeline.extract(args)
    args.cfg = cfg_model
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    gaussians.XYZ_TYPE = "UV"
    active_levels = ingp_model.set_active_levels(args.iteration)
    
    print(f"Model loaded!")
    print(f"Gaussians: {len(gaussians.get_xyz):,}")
    
    # Run detailed profiling
    results = detailed_render_profiling(
        scene, gaussians, pipe, background, ingp_model,
        beta, args.iteration, cfg_model,
        cameras=scene.getTestCameras(),
        num_warmup=args.num_warmup,
        num_benchmark=args.num_benchmark
    )
    
    # Save results
    output_path = os.path.join(args.model_path, 'detailed_benchmark.txt')
    with open(output_path, 'w') as f:
        f.write("Detailed Rendering Profiling Results\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Method: {args.method}\n")
        if args.method == "cat":
            f.write(f"Hybrid levels: {args.hybrid_levels}\n")
        f.write(f"Gaussians: {len(gaussians.get_xyz):,}\n\n")
        
        f.write("Timing Breakdown\n")
        f.write("-"*70 + "\n")
        
        if 'hash_encoding' in results:
            f.write(f"Hashgrid encoding:    {results['hash_encoding']['mean_ms']:.2f} ms\n")
        if 'rasterization_alpha_blend' in results:
            f.write(f"Rasterization+Alpha:  {results['rasterization_alpha_blend']['mean_ms']:.2f} ms\n")
        if 'mlp_decoding' in results:
            f.write(f"MLP decoding:         {results['mlp_decoding']['mean_ms']:.2f} ms\n")
        if 'background_addition' in results:
            f.write(f"Background addition:  {results['background_addition']['mean_ms']:.2f} ms\n")
        
        f.write(f"\nFull render:          {results['full_render']['mean_ms']:.2f} ms\n")
        f.write(f"FPS:                  {1000.0/results['full_render']['mean_ms']:.2f}\n")
    
    print(f"\nResults saved to: {output_path}")






