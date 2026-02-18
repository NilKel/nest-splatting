#!/usr/bin/env python3
"""
Profile the backward pass of 3D_direct_fused (lean) mode.

Measures:
1. Python-level timing with torch.cuda.Event for macro components
2. clock64() instrumentation for intra-kernel phase breakdown

Usage:
    conda run -n nest_splatting python scripts/profile_backward.py \
        -m outputs/nerf_synthetic/chair/cat/Betascaled064_2mlp_5_levels \
        --yaml configs/nerfsyn.yaml --method cat --hybrid_levels 5 \
        --kernel beta --iteration 30000
"""

import torch
import time
import math
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

# Import lean library profiling functions
try:
    from diff_surfel_3D import reset_backward_profile, read_backward_profile
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False


def get_gpu_clock_mhz():
    """Get GPU SM clock frequency for converting cycles to time."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return None


def profile_backward(viewpoint, gaussians, pipe, bg, ingp, cfg, iteration,
                     num_warmup=10, num_iters=50):
    """Profile backward pass with CUDA event timing and clock64 instrumentation."""

    beta = cfg.surfel.tg_beta
    gt_image = viewpoint.original_image.cuda()

    # Create CUDA events
    events = {}
    for name in ['fwd_start', 'fwd_end', 'loss_end', 'bwd_end']:
        events[name] = torch.cuda.Event(enable_timing=True)

    # Warmup
    print(f"  Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        result = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                       iteration=iteration, cfg=cfg)
        image = result["render"]
        loss = torch.nn.functional.l1_loss(image, gt_image)
        loss.backward()
        # Zero grads
        pass  # Skip grad zeroing - doesn't affect timing
        pass  # Skip INGP grad zeroing
    torch.cuda.synchronize()

    # Profiling
    times = {
        'forward': [],
        'loss': [],
        'backward': [],
        'total': [],
    }

    print(f"  Profiling ({num_iters} iters)...")
    if HAS_PROFILING:
        reset_backward_profile()

    for i in range(num_iters):
        events['fwd_start'].record()

        result = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                       iteration=iteration, cfg=cfg)
        image = result["render"]

        events['fwd_end'].record()

        loss = torch.nn.functional.l1_loss(image, gt_image)

        events['loss_end'].record()

        loss.backward()

        events['bwd_end'].record()
        torch.cuda.synchronize()

        times['forward'].append(events['fwd_start'].elapsed_time(events['fwd_end']))
        times['loss'].append(events['fwd_end'].elapsed_time(events['loss_end']))
        times['backward'].append(events['loss_end'].elapsed_time(events['bwd_end']))
        times['total'].append(events['fwd_start'].elapsed_time(events['bwd_end']))

        # Zero grads for next iter
        pass  # Skip grad zeroing - doesn't affect timing
        pass  # Skip INGP grad zeroing

    # Read clock64 profiling data
    profile_data = None
    if HAS_PROFILING:
        cycles, counts = read_backward_profile()
        profile_data = {
            'cycles': cycles.numpy(),
            'counts': counts.numpy(),
        }

    return times, profile_data


def print_results(times, profile_data, gpu_clock_mhz):
    """Print formatted profiling results."""
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0

    fwd = mean(times['forward'])
    loss_t = mean(times['loss'])
    bwd = mean(times['backward'])
    total = mean(times['total'])

    print("\n" + "=" * 70)
    print("MACRO TIMING (torch.cuda.Event, averaged over iterations)")
    print("=" * 70)
    print(f"\n{'Component':<35} {'Time (ms)':>12} {'% Total':>10}")
    print("-" * 60)
    print(f"{'Forward pass':<35} {fwd:>12.3f} {fwd/total*100:>9.1f}%")
    print(f"{'Loss computation':<35} {loss_t:>12.3f} {loss_t/total*100:>9.1f}%")
    print(f"{'Backward pass':<35} {bwd:>12.3f} {bwd/total*100:>9.1f}%")
    print("-" * 60)
    print(f"{'Total':<35} {total:>12.3f} {'100.0%':>10}")
    print(f"{'FPS (fwd only)':<35} {1000/fwd:>12.1f}")
    print(f"{'it/s (fwd+bwd)':<35} {1000/total:>12.1f}")

    if profile_data is not None:
        cycles = profile_data['cycles']
        counts = profile_data['counts']
        n_iters = len(times['forward'])

        n_gaussians = int(counts[0])
        n_skipped = int(counts[1])
        n_tiles = int(counts[2])
        n_intersections = int(counts[3])

        phase_a = int(cycles[0])  # MLP fwd + GEMM L3 + dL_dz2
        phase_b = int(cycles[1])  # GEMM L2 + dL_dz1
        phase_c = int(cycles[2])  # GEMM L1 + dL_dinput + grads
        flush = int(cycles[3])
        total_cycles = int(cycles[4])

        print("\n" + "=" * 70)
        print("INTRA-KERNEL PROFILING (clock64, summed across all blocks)")
        print("=" * 70)

        print(f"\n{'Metric':<40} {'Value':>15}")
        print("-" * 60)
        print(f"{'Gaussians processed (with participation)':<40} {n_gaussians//n_iters:>15,}/iter")
        print(f"{'Gaussians skipped (ballot)':<40} {n_skipped//n_iters:>15,}/iter")
        print(f"{'Skip rate':<40} {n_skipped/(n_gaussians+n_skipped)*100 if (n_gaussians+n_skipped) > 0 else 0:>14.1f}%")
        print(f"{'Tiles processed':<40} {n_tiles//n_iters:>15,}/iter")
        print(f"{'Total intersections':<40} {n_intersections//n_iters:>15,}/iter")
        if n_tiles > 0:
            print(f"{'Avg Gaussians/tile':<40} {(n_gaussians+n_skipped)/n_tiles:>15.1f}")
        if n_gaussians > 0:
            print(f"{'Avg intersections/Gaussian':<40} {n_intersections/n_gaussians:>15.1f}")

        print(f"\n{'Phase':<40} {'Cycles':>15} {'%':>8}")
        print("-" * 65)

        if total_cycles > 0:
            print(f"{'A: MLP fwd + GEMM L3 + dL_dz2':<40} {phase_a:>15,} {phase_a/total_cycles*100:>7.1f}%")
            print(f"{'B: GEMM L2 + dL_dz1':<40} {phase_b:>15,} {phase_b/total_cycles*100:>7.1f}%")
            print(f"{'C: GEMM L1 + dL_dinput + grads':<40} {phase_c:>15,} {phase_c/total_cycles*100:>7.1f}%")
            print(f"{'D: Tile flush (atomicAdd to global)':<40} {flush:>15,} {flush/total_cycles*100 if total_cycles > 0 else 0:>7.1f}%")
            print("-" * 65)
            print(f"{'Total per-Gaussian':<40} {total_cycles:>15,} {'100.0%':>8}")

        # Convert to ms if we know the clock
        if gpu_clock_mhz and total_cycles > 0:
            # cycles are summed across blocks, so divide by n_tiles to get per-tile
            # then divide by clock to get seconds
            cycles_per_tile = total_cycles / n_tiles if n_tiles > 0 else 0
            ms_per_tile = cycles_per_tile / (gpu_clock_mhz * 1e3)  # MHz * 1e3 = cycles/ms

            print(f"\n{'GPU SM clock':<40} {gpu_clock_mhz:>12} MHz")
            print(f"{'Per-tile Gaussian processing':<40} {ms_per_tile:>11.3f} ms")

            # Estimate total kernel time from cycles
            # All tiles run in parallel on SMs. With ~3000 tiles and ~80 SMs,
            # each SM handles ~37 tiles sequentially
            import subprocess
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                    capture_output=True, text=True
                )
            except:
                pass

        # Per-Gaussian breakdown
        if n_gaussians > 0:
            print(f"\n{'Per-Gaussian (avg cycles/block)':<40}")
            print("-" * 60)
            cpg = total_cycles / n_gaussians
            print(f"{'  Phase A (MLP fwd + GEMM L3 + dL_dz2)':<40} {phase_a/n_gaussians:>12.0f} cyc")
            print(f"{'  Phase B (GEMM L2 + dL_dz1)':<40} {phase_b/n_gaussians:>12.0f} cyc")
            print(f"{'  Phase C (GEMM L1 + grads)':<40} {phase_c/n_gaussians:>12.0f} cyc")
            print(f"{'  Total':<40} {cpg:>12.0f} cyc")

            if gpu_clock_mhz:
                us_per_gaussian = cpg / (gpu_clock_mhz)  # cycles / (MHz) = microseconds
                print(f"{'  Total':<40} {us_per_gaussian:>11.2f} µs")


def main():
    parser = ArgumentParser(description="Profile backward pass")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, default="./configs/nerfsyn.yaml")
    parser.add_argument("--method", type=str, default="cat")
    parser.add_argument("--hybrid_levels", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--kernel", type=str, default="gaussian",
                       choices=["gaussian", "beta", "flex", "general"])

    args = get_combined_args(parser)
    cfg = Config(args.yaml)
    merge_cfg_to_args(args, cfg)

    print(f"\n[PROFILE] Loading model from: {args.model_path}")

    ingp = INGP(cfg, args=args).to('cuda')
    iteration = args.iteration
    try:
        ingp.load_model(args.model_path, iteration)
    except RuntimeError as e:
        print(f"[PROFILE] WARNING: Could not load INGP weights ({e})")
        print(f"[PROFILE] Using random MLP weights (timing is still valid)")
        # Load hash encoding only if possible
        import glob
        ngp_files = glob.glob(os.path.join(args.model_path, f"ngp_{iteration}.pth"))
        if ngp_files:
            ckpt = torch.load(ngp_files[0], map_location='cuda')
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            # Load only hash encoding weights (skip MLP)
            filtered = {k: v for k, v in state.items()
                       if 'hash_encoding' in k and k in ingp.state_dict()
                       and v.shape == ingp.state_dict()[k].shape}
            if filtered:
                ingp.load_state_dict(filtered, strict=False)
                print(f"[PROFILE] Loaded {len(filtered)} hash encoding parameters")

    dataset, pipe = model.extract(args), pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp.set_active_levels(iteration)

    if args.kernel != "gaussian":
        gaussians.kernel_type = args.kernel

    num_gaussians = len(gaussians.get_xyz)
    H, W = scene.getTestCameras()[0].image_height, scene.getTestCameras()[0].image_width
    print(f"[PROFILE] Gaussians: {num_gaussians:,}, Resolution: {W}x{H}")
    print(f"[PROFILE] Method: {args.method}, Kernel: {args.kernel}")
    print(f"[PROFILE] Profiling available: {HAS_PROFILING}")

    viewpoint = scene.getTestCameras()[0]
    bg = torch.zeros(3, device="cuda")

    gpu_clock = get_gpu_clock_mhz()
    print(f"[PROFILE] GPU SM clock: {gpu_clock} MHz")

    # Profile
    times, profile_data = profile_backward(
        viewpoint, gaussians, pipe, bg, ingp, cfg, iteration,
        num_iters=args.num_iters
    )

    print_results(times, profile_data, gpu_clock)


if __name__ == "__main__":
    main()
