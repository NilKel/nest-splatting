#!/usr/bin/env python3
"""
Per-surfel BLUR-SPLIT dominance stats over ALL training views.

Replays the --blur_split accumulation from train.py (~L2733-2750) on a trained
checkpoint: per training view, render_pkg['max_contrib_idx'] (int32 [H,W],
per-pixel id of the max-weight contributor, -1 invalid) is bincounted into a
per-Gaussian dominated-pixel area; a surfel is FLAGGED at threshold t if
area > H*W/t in ANY single training view. We accumulate the per-surfel MAX
normalized area (area / (H*W)) over views, then report flag counts for a set
of --blur_thresh values.

Only the diff_surfel_3D_sh_res family populates max_contrib_idx, so the render
must go through the 3D_SH_res mode-5 path (this run is --method 3D_SH_res).
The INGP weights only affect color, not the max-weight contributor, but we
load the trained ngp checkpoint anyway for fidelity.

Usage:
    conda run -n nest_splatting python scripts/blur_thresh_stats.py \
        --model_path outputs/mip_360/garden/3D_SH_res/BUG2D_SV_30thr_005w25gLP_N2F_Jac \
        --thresholds 2000 3000 5000
"""

import os
import sys
import json
import glob
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams


def load_training_config(model_path):
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            return pickle.load(f)
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            return Namespace(**json.load(f))
    raise FileNotFoundError(f"No args.pkl/args.json in {model_path}")


def main():
    parser = ArgumentParser(description="blur_split dominance stats over training views")
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[2000.0, 3000.0, 5000.0])
    cli = parser.parse_args()

    model_path = cli.model_path
    args = load_training_config(model_path)
    args.model_path = model_path
    args.eval = True  # keep the train/test split used in training

    # YAML config from the checkpoint dir (falls back to args.yaml).
    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg_model = Config(config_yaml_path if os.path.exists(config_yaml_path) else args.yaml)

    # Auto-detect iteration from ngp_*.pth
    iteration = cli.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        if not ngp_files:
            raise FileNotFoundError(f"No ngp_*.pth in {model_path}")
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                        for f in ngp_files)
    print(f"[BLUR] model={model_path}  iteration={iteration}  method={args.method}  "
          f"kernel={getattr(args, 'kernel', 'gaussian')}  feature={getattr(args, 'feature', 'sh')}")

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset, pipe = model_params.extract(args), pipeline_params.extract(args)

    # Mirror the train.py dataset annotations Scene/GaussianModel may read.
    dataset.method = args.method
    dataset.is_gestex = getattr(args, 'is_gestex', False)
    dataset.hybrid_levels = getattr(args, 'hybrid_levels', 3)
    dataset.decompose_mode = getattr(args, 'decompose_mode', None)

    # INGP (max_contrib_idx only depends on geometry/opacity, but load the
    # trained weights anyway — cheap and keeps the render faithful).
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    # --feature SV: load_ply keeps SV mode only if feature_mode is pre-set.
    gaussians.feature_mode = getattr(args, 'feature', 'sh')
    gaussians.kernel_type = getattr(args, 'kernel', 'gaussian')
    gaussians.kernel_type2 = getattr(args, 'kernel2', None)
    gaussians.is_gestex = getattr(args, 'is_gestex', False)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                  full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"

    N = gaussians.get_xyz.shape[0]
    train_cameras = scene.getTrainCameras()
    print(f"[BLUR] N={N:,} surfels   {len(train_cameras)} training views")

    bg = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    max_area = torch.zeros(N, dtype=torch.long, device="cuda")     # max pixels over views
    max_ratio = torch.zeros(N, dtype=torch.float64, device="cuda")  # max area/(H*W) over views
    hw_set = set()

    with torch.no_grad():
        for vi, cam in enumerate(train_cameras):
            render_pkg = render(cam, gaussians, pipe, bg, ingp=ingp, beta=beta,
                                iteration=iteration, cfg=cfg_model,
                                use_xyz_mode=getattr(args, 'use_xyz_mode', False),
                                decompose_mode=dataset.decompose_mode,
                                is_training=True,
                                aabb_mode=getattr(args, 'aabb', '2dgs'),
                                aa=getattr(args, 'aa', 0.0),
                                aa_threshold=getattr(args, 'aa_threshold', 0.01),
                                max_intersections_per_pixel=getattr(args, 'max_intersections_per_pixel', 32),
                                lowpass=getattr(args, 'lowpass', False),
                                pixel_center=getattr(args, 'pixel_center', False),
                                antialiasing=getattr(args, 'antialiasing', 0.0),
                                sv_metric=getattr(args, 'sv_metric', 'l2'))
            midx = render_pkg.get("max_contrib_idx", None)
            assert midx is not None and midx.numel() > 0, \
                "max_contrib_idx not populated — render did not take the 3D_SH_res path"
            H, W = midx.shape[-2], midx.shape[-1]
            hw_set.add((H, W))
            idx_flat = midx.long().reshape(-1)
            valid = idx_flat >= 0
            if valid.any():
                ids = idx_flat[valid]
                ids = ids[ids < N]
                area = torch.bincount(ids, minlength=N)
                max_area = torch.maximum(max_area, area)
                max_ratio = torch.maximum(max_ratio, area.double() / float(H * W))
            if (vi + 1) % 20 == 0 or vi == len(train_cameras) - 1:
                print(f"  [{vi+1}/{len(train_cameras)}] {W}x{H}  "
                      f"running max area={int(max_area.max())}px", flush=True)

    hw_desc = ", ".join(f"{w}x{h} ({h*w:,}px)" for (h, w) in sorted(hw_set))
    hw = next(iter(hw_set))
    HW = hw[0] * hw[1]

    ma = max_area.float()
    never = int((max_area == 0).sum())
    q = torch.quantile(ma.cpu(), torch.tensor([0.5, 0.9, 0.99]))

    print("\n" + "=" * 78)
    print("BLUR-SPLIT DOMINANCE STATS (max dominated area over ALL training views)")
    print("=" * 78)
    print(f"Model:       {model_path}")
    print(f"Iteration:   {iteration}")
    print(f"Surfels:     {N:,}")
    print(f"Train views: {len(train_cameras)}   resolution(s): {hw_desc}")
    print(f"\nMax-area distribution (pixels, per-surfel max over views):")
    print(f"  p50={q[0]:.1f}  p90={q[1]:.1f}  p99={q[2]:.1f}  max={int(max_area.max()):,}")
    print(f"  never dominant anywhere (max_area == 0): {never:,} ({100.0*never/N:.2f}%)")
    print(f"\n{'blur_thresh':>12} {'pixel cutoff H*W/t':>20} {'flagged':>12} {'% of N':>9}")
    print("-" * 58)
    for t in cli.thresholds:
        cutoff = HW / t
        flagged = int((max_ratio > (1.0 / t)).sum())
        print(f"{t:>12.0f} {cutoff:>20.1f} {flagged:>12,} {100.0*flagged/N:>8.3f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
