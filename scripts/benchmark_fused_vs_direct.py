#!/usr/bin/env python3
"""
Benchmark: 3D_direct (Python MLP) vs 3D_direct_fused (CUDA MLP)
Measures forward pass only.
"""

import os
import sys
import pickle
import json
import time
import glob
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
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
    ingp.set_active_levels(iteration)  # Initialize active_levels

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"

    return gaussians, scene, ingp, cfg, pipe, args, iteration


def benchmark_mode(mode_name, gaussians, ingp, cameras, bg, pipe, cfg, num_warmup=5, num_passes=3):
    """Benchmark a specific mode over all test cameras."""

    # Store original mode
    orig_3d_direct = ingp.is_3D_direct_mode
    orig_3d_direct_fused = ingp.is_3D_direct_fused_mode
    orig_cat = ingp.is_cat_mode

    # Configure mode
    orig_lean = getattr(ingp, 'is_3D_direct_lean_mode', False)
    if mode_name == "3D_direct":
        ingp.is_3D_direct_mode = True
        ingp.is_3D_direct_fused_mode = False
        ingp.is_3D_direct_lean_mode = False
        ingp.is_cat_mode = False
    elif mode_name == "3D_direct_fused":
        ingp.is_3D_direct_mode = False
        ingp.is_3D_direct_fused_mode = True
        ingp.is_3D_direct_lean_mode = True  # Use lean library
        ingp.is_cat_mode = False
    elif mode_name == "cat":
        ingp.is_3D_direct_mode = False
        ingp.is_3D_direct_fused_mode = False
        ingp.is_3D_direct_lean_mode = False
        ingp.is_cat_mode = True

    print(f"\n=== Benchmarking {mode_name} ({len(cameras)} cameras, {num_passes} passes) ===")

    # Warmup with first camera
    print(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for i in range(num_warmup):
            try:
                result = render(cameras[0], gaussians, pipe, bg, cfg=cfg, ingp=ingp, iteration=50000)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warmup {i} failed: {e}")
                import traceback; traceback.print_exc()
                ingp.is_3D_direct_mode = orig_3d_direct
                ingp.is_3D_direct_fused_mode = orig_3d_direct_fused
                ingp.is_3D_direct_lean_mode = orig_lean
                ingp.is_cat_mode = orig_cat
                return None, None

    # Benchmark over all cameras, multiple passes
    print(f"Benchmarking...")
    times = []

    with torch.no_grad():
        for p in range(num_passes):
            for cam in cameras:
                torch.cuda.synchronize()
                start = time.perf_counter()

                result = render(cam, gaussians, pipe, bg, cfg=cfg, ingp=ingp, iteration=50000)

                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000.0 / avg_time

    print(f"Time: {avg_time:.2f} ms (min={min_time:.2f}, max={max_time:.2f})")
    print(f"FPS: {fps:.1f}  ({len(times)} frames)")

    # Restore original mode
    ingp.is_3D_direct_mode = orig_3d_direct
    ingp.is_3D_direct_fused_mode = orig_3d_direct_fused
    ingp.is_3D_direct_lean_mode = orig_lean
    ingp.is_cat_mode = orig_cat

    return avg_time, fps


def main():
    model_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"

    print("Loading model...")
    gaussians, scene, ingp, cfg, pipe, args, iteration = load_model(model_path)

    print(f"Loaded: method={args.method}, hybrid_levels={args.hybrid_levels}")
    print(f"Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"INGP: levels={ingp.levels}, hybrid_levels={ingp.hybrid_levels}")

    # Build mlp_fused from mlp_3D_direct weights (for fused mode benchmarking)
    if ingp.mlp_fused is None and ingp.mlp_3D_direct is not None:
        from torch import nn
        mlp_input_dim = 40  # gauss(20) + hash(4) + view(16)
        ingp.mlp_fused = nn.Sequential(
            nn.Linear(mlp_input_dim + 1, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 3, bias=False),
        ).cuda()
        # Copy weights from mlp_3D_direct (biased) to mlp_fused (bias-free)
        W1_src = ingp.mlp_3D_direct[0].weight.data  # [32, 40]
        b1_src = ingp.mlp_3D_direct[0].bias.data     # [32]
        W2_src = ingp.mlp_3D_direct[2].weight.data   # [32, 32]
        W3_src = ingp.mlp_3D_direct[4].weight.data   # [3, 32]
        # W1_fused: [32, 41] where col 40 = bias (implicit bias via input padding)
        W1_fused = torch.zeros(32, 41, device='cuda')
        W1_fused[:, :40] = W1_src
        W1_fused[:, 40] = b1_src
        ingp.mlp_fused[0].weight.data.copy_(W1_fused)
        ingp.mlp_fused[2].weight.data.copy_(W2_src)
        ingp.mlp_fused[4].weight.data.copy_(W3_src)
        print(f"[BENCH] Built mlp_fused from mlp_3D_direct weights (bias folded into W1 col 40)")

    # Get all test cameras
    cameras = scene.getTestCameras()
    print(f"Test cameras: {len(cameras)}, Resolution: {cameras[0].image_width}x{cameras[0].image_height}")

    # Background
    bg = torch.ones(3, device="cuda")

    results = {}

    # Benchmark 3D_direct (Python MLP)
    try:
        t, fps = benchmark_mode("3D_direct", gaussians, ingp, cameras, bg, pipe, cfg)
        if t: results["3D_direct"] = (t, fps)
    except Exception as e:
        print(f"3D_direct failed: {e}")
        import traceback; traceback.print_exc()

    # Benchmark 3D_direct_fused (CUDA MLP)
    try:
        t, fps = benchmark_mode("3D_direct_fused", gaussians, ingp, cameras, bg, pipe, cfg)
        if t: results["3D_direct_fused"] = (t, fps)
    except Exception as e:
        print(f"3D_direct_fused failed: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (Forward Pass Only, avg over all test cameras)")
    print("="*60)
    for mode, (t, fps) in results.items():
        print(f"{mode:20s}: {t:7.2f} ms  ({fps:6.1f} FPS)")

    if "3D_direct" in results and "3D_direct_fused" in results:
        speedup = results["3D_direct"][0] / results["3D_direct_fused"][0]
        print(f"\nFused speedup vs Python: {speedup:.2f}x")


if __name__ == "__main__":
    main()
