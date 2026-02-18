#!/usr/bin/env python3
"""
Inline benchmark script (from user's one-liner).
Compare with benchmark_full.py to find discrepancies.
"""

import os
import sys
import pickle
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render, RenderCache
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.point_utils import cam2rays
from argparse import ArgumentParser


def main():
    model_path = 'outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral005decayadrrec_5_levels'

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
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
            gaussians._gaussian_features = gaussians._gaussian_features[valid]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid]

    print(f'Gaussians: {len(gaussians.get_xyz):,}')

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    background = torch.zeros(3, device='cuda')
    beta = cfg_model.surfel.tg_beta
    aabb_mode = getattr(args, 'aabb', '2dgs')

    # Create cache and pre-compute
    cache = RenderCache()
    cache.get_screenspace_points(len(gaussians.get_xyz))
    cache.homotrans = gaussians.get_homotrans()

    def benchmark(use_cache, n_passes=3):
        with torch.no_grad():
            # Warmup
            for i in range(20):
                cam = all_cams[i % len(all_cams)]
                _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                          iteration=35000, cfg=cfg_model, aabb_mode=aabb_mode,
                          fast_inference=True, cache=cache if use_cache else None)
            torch.cuda.synchronize()

            # Benchmark multiple passes
            total_time = 0
            total_frames = 0
            for _ in range(n_passes):
                torch.cuda.synchronize()
                start = time.time()  # NOTE: uses time.time(), not time.perf_counter()
                for cam in all_cams:
                    _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                              iteration=35000, cfg=cfg_model, aabb_mode=aabb_mode,
                              fast_inference=True, cache=cache if use_cache else None)
                torch.cuda.synchronize()
                total_time += time.time() - start
                total_frames += len(all_cams)

        fps = total_frames / total_time
        ms = total_time / total_frames * 1000
        return fps, ms

    print()
    print('Benchmarking (3 passes over 194 cameras each)...')
    fps_no_cache, ms_no_cache = benchmark(use_cache=False)
    print(f'fast_inference=True, no cache:   {fps_no_cache:6.1f} FPS  ({ms_no_cache:.2f} ms/frame)')

    fps_cache, ms_cache = benchmark(use_cache=True)
    print(f'fast_inference=True, with cache: {fps_cache:6.1f} FPS  ({ms_cache:.2f} ms/frame)')

    print(f'')
    print(f'Speedup from cache: {fps_cache/fps_no_cache:.2f}x ({ms_no_cache - ms_cache:.2f} ms saved)')
    print(f'')

    # Compare to original (no fast_inference)
    print('Comparing to original mode...')
    def benchmark_original():
        with torch.no_grad():
            for i in range(20):
                cam = all_cams[i % len(all_cams)]
                _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                          iteration=35000, cfg=cfg_model, aabb_mode=aabb_mode, fast_inference=False)
            torch.cuda.synchronize()

            start = time.time()
            for cam in all_cams:
                _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                          iteration=35000, cfg=cfg_model, aabb_mode=aabb_mode, fast_inference=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        return len(all_cams) / elapsed, elapsed / len(all_cams) * 1000

    fps_orig, ms_orig = benchmark_original()
    print(f'Original (fast_inference=False): {fps_orig:6.1f} FPS  ({ms_orig:.2f} ms/frame)')
    print(f'')
    print(f'Total speedup (original -> cached): {fps_cache/fps_orig:.2f}x ({ms_orig - ms_cache:.2f} ms saved)')


if __name__ == '__main__':
    main()
