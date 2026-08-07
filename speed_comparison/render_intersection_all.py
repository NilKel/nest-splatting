#!/usr/bin/env python3
"""Render intersection heatmaps + raw counts for every test view of a
nest-splatting checkpoint, matching FastGS's `intersection_maps/` output
convention (max_display=200, cmap=turbo, per-view stats).

Usage:
    conda run -n nest_splatting python speed_comparison/render_intersection_all.py \\
        --model_path outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10 \\
        --out_dir speed_comparison/nest_splatting_neural
"""
import argparse, glob, json, os, pickle, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8, create_intersection_heatmap


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    ajson = model_path / "args.json"
    if ajson.exists():
        return Namespace(**json.load(open(ajson)))
    raise FileNotFoundError(model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--out_dir",    type=Path, required=True)
    ap.add_argument("--max_display", type=int, default=200)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "intersection_maps").mkdir(exist_ok=True)
    (args.out_dir / "intersection_raw").mkdir(exist_ok=True)

    train_args = load_train_args(args.model_path)
    train_args.model_path = str(args.model_path)
    train_args.eval = True
    cfg_yaml = args.model_path / "config.yaml"
    cfg_model = Config(str(cfg_yaml)) if cfg_yaml.exists() else Config(train_args.yaml)

    ngp_files = glob.glob(str(args.model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files)

    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(train_args)
    pipe = pipeline_params.extract(train_args)

    ingp_model = INGP(cfg_model, args=train_args).to("cuda")
    ingp_model.load_model(str(args.model_path), iteration)
    ingp_model.set_active_levels(iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel

    test_cameras = scene.getTestCameras()
    print(f"[intersect] {len(test_cameras)} test views, {gaussians.get_xyz.shape[0]:,} Gauss")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0

    per_view = []
    global_max = 0
    global_sum = 0.0
    global_n   = 0

    with torch.no_grad():
        for i, cam in enumerate(test_cameras):
            pkg = render(cam, gaussians, pipe, bg, beta=beta, iteration=iteration,
                         cfg=cfg_model, ingp=ingp_model)
            gaussian_num = pkg['gaussian_num']   # (1, H, W) soft contributor count

            # Raw counts as npy (matches FastGS `intersection_raw/*.npy`).
            raw = gaussian_num.squeeze(0).cpu().numpy()
            np.save(args.out_dir / "intersection_raw" / f"{i:05d}.npy", raw)

            # Heatmap turbo max_display=200 (matches FastGS).
            heatmap, min_c, max_c = create_intersection_heatmap(gaussian_num,
                                                                  max_display=args.max_display)
            save_img_u8(heatmap, str(args.out_dir / "intersection_maps" / f"{i:05d}.png"))

            # Stats.
            arr = raw.flatten()
            m = float(arr.max())
            mean = float(arr.mean())
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            per_view.append({"max": m, "mean": mean, "p50": p50, "p95": p95, "p99": p99})
            global_max = max(global_max, m)
            global_sum += arr.sum()
            global_n   += arr.size

            if i < 3 or (i + 1) == len(test_cameras):
                print(f"  [{i:03d}] max={m:6.1f} mean={mean:6.2f} p95={p95:5.1f} p99={p99:5.1f}")

    global_mean = float(global_sum / global_n) if global_n else 0.0
    global_max = float(global_max)

    summary = {
        "global_max":  global_max,
        "global_mean": global_mean,
        "max_display": args.max_display,
        "cmap": "turbo",
        "num_points": int(gaussians.get_xyz.shape[0]),
        "resolution": f"{test_cameras[0].image_width}x{test_cameras[0].image_height}",
        "per_view": per_view,
    }
    with open(args.out_dir / "intersection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[intersect] Global mean contributors/pixel: {global_mean:.2f}")
    print(f"[intersect] Global max contributors/pixel:  {global_max:.0f}")
    print(f"[intersect] Saved → {args.out_dir}")


if __name__ == "__main__":
    main()
