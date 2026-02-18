#!/usr/bin/env python3
"""
Simple benchmark - no component timing, minimal sync calls.
Just measures end-to-end FPS.
"""

import os
import sys
import pickle
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser


def main():
    model_path = 'outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral005decayadrrec_5_levels'

    # Load model
    with open(f'{model_path}/args.pkl', 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    cfg_model = Config(f'{model_path}/config.yaml')

    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(model_path, 35000)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=35000, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = 'UV'
    ingp.set_active_levels(35000)
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    if dead_mask.sum() > 0:
        valid = ~dead_mask
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation', '_appearance_level']:
            setattr(gaussians, attr, getattr(gaussians, attr)[valid])
        if gaussians._gaussian_features is not None:
            gaussians._gaussian_features = gaussians._gaussian_features[valid]
        if gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid]

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    background = torch.zeros(3, device='cuda')
    beta = cfg_model.surfel.tg_beta

    # Setup cache
    cache = RenderCache()
    cache.get_screenspace_points(len(gaussians.get_xyz))
    cache.homotrans = gaussians.get_homotrans()

    print("="*60)
    print(f"Gaussians: {len(gaussians.get_xyz):,}")
    print(f"Cameras: {len(all_cams)}")
    print(f"Image size: {all_cams[0].image_width}x{all_cams[0].image_height}")
    print("="*60)

    # Warmup (50 frames)
    print("\nWarmup (50 frames)...")
    with torch.no_grad():
        for i in range(50):
            cam = all_cams[i % len(all_cams)]
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                      iteration=35000, cfg=cfg_model, fast_inference=True, cache=cache)
    torch.cuda.synchronize()

    # Verify output shape
    result = render(all_cams[0], gaussians, pipe, background, ingp=ingp, beta=beta,
                   iteration=35000, cfg=cfg_model, fast_inference=True, cache=cache)
    print(f"Output shape: {result['render'].shape}")

    # Benchmark
    n_passes = 5
    print(f"\nBenchmark ({n_passes} passes x {len(all_cams)} cameras):")
    print("-"*60)

    with torch.no_grad():
        for p in range(n_passes):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for cam in all_cams:
                _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                          iteration=35000, cfg=cfg_model, fast_inference=True, cache=cache)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            fps = len(all_cams) / elapsed
            ms = elapsed / len(all_cams) * 1000
            print(f"  Pass {p+1}: {fps:6.1f} FPS  ({ms:.2f} ms/frame)")

    print("="*60)


if __name__ == '__main__':
    main()
