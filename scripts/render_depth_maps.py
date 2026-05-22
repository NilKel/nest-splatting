#!/usr/bin/env python3
"""
render_depth_maps.py
====================

Dump the depth/alpha/normal maps used by TSDF mesh extraction, from N test
views. Useful to diagnose why the resulting mesh looks janky: if depth is
noisy/inconsistent across views, marching cubes will produce a noisy surface.

Per view, saves under <out_subdir>/depth_maps/:
  - rgb/<NNN>.png             - the textured-only RGB render (target for TSDF)
  - alpha/<NNN>.png           - rend_alpha (mesh is integrated where this is high)
  - surf_depth/<NNN>.png      - the depth fed to TSDF (turbo colormap, normalized)
  - depth_expected/<NNN>.png  - alpha-weighted mean depth
  - depth_median/<NNN>.png    - depth of the median-weight contributor
  - normal/<NNN>.png          - rendered normal map (encoded as RGB)
  - depth_raw/<NNN>.npy       - raw FP32 depth (for offline analysis)
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.system_utils import searchForMaxIteration
from utils.render_utils import save_img_u8, convert_gray_to_cmap
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args

from bake_mesh_texture import install_setter_mirror, fold_train_args, slice_to_textured


def main():
    parser = ArgumentParser(description="Render depth maps from N test views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_subdir_dm", default="depth_maps", type=str)
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--split", default="test", choices=["test", "train"], type=str)
    parser.add_argument("--no_slice", action="store_true",
                        help="Don't slice to textured-only — render with full mixed_3d model")
    parser.add_argument("--depth_max", default=-1.0, type=float,
                        help="Manual max-depth for colormap normalization. -1 = auto (95th pct)")
    args = get_combined_args(parser)
    args = fold_train_args(args, args.model_path)

    exp_path = args.model_path
    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(exp_path, "point_cloud"))

    install_setter_mirror(getattr(args, "method", None))

    yaml_file = getattr(args, "yaml", None) or "tiny"
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    print(f"[DM] Model: {exp_path}")
    print(f"[DM] Iter:  {iteration}")

    ingp = INGP(cfg_model, args=args).to("cuda")
    ingp.load_model(exp_path, iteration)
    ingp.set_active_levels(iteration)
    if args.method in ("mixed", "mixed_3d"):
        from diff_surfel_3D_sh_res import (
            set_residual_mode, set_activation_bias, set_compact_mult)
        sh_b, res_b = getattr(args, "activation_bias", [0.5, 0.0])
        set_activation_bias(sh_bias=float(sh_b), res_bias=float(res_b))
        set_residual_mode(2)
        set_compact_mult(float(getattr(args, "fastgs_mult", 0.5)))

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    gaussians.feature_mode = getattr(args, "feature", "sh")
    gaussians._sv_training_flag = False

    if not args.no_slice:
        n_tex = int(gaussians._is_textured.sum().item())
        print(f"[DM] Slicing to textured-only: {n_tex:,} of {gaussians.get_xyz.shape[0]:,}")
        slice_to_textured(gaussians)
    else:
        print(f"[DM] Full model: {gaussians.get_xyz.shape[0]:,}")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    if args.num_views > 0:
        cams = list(cams)[: args.num_views]
    print(f"[DM] Rendering {len(cams)} {args.split} views (no_slice={args.no_slice})")

    out_root = os.path.join(exp_path, args.out_subdir_dm)
    for sub in ("rgb", "alpha", "surf_depth", "depth_expected", "depth_median",
                "normal", "depth_raw"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    print(f"[DM] Output: {out_root}")

    # First pass: collect depths to auto-determine a colormap max if --depth_max < 0.
    print("[DM] Pass 1: gather depth ranges…")
    surf_depths, exp_depths, med_depths = [], [], []
    rgbs, alphas, normals = [], [], []
    with torch.no_grad():
        for cam in tqdm(cams, desc="render"):
            pkg = render(cam, gaussians, pipe, bg, ingp=ingp,
                         beta=beta_cfg, iteration=iteration, cfg=cfg_model,
                         skybox=None, background_mode="none", bg_hashgrid=None)
            rgbs.append(torch.clamp(pkg["render"], 0.0, 1.0).cpu())
            alphas.append(pkg["rend_alpha"].cpu())
            surf_depths.append(pkg["surf_depth"].cpu())
            exp_depths.append(pkg.get("depth_expected", pkg["surf_depth"]).cpu()
                              if "depth_expected" in pkg else None)
            med_depths.append(pkg.get("depth_median", pkg["surf_depth"]).cpu()
                              if "depth_median" in pkg else None)
            normals.append(pkg.get("rend_normal",
                                    pkg.get("surf_normal", None)))
            if normals[-1] is not None:
                normals[-1] = normals[-1].detach().cpu()

    if args.depth_max < 0:
        all_depths = torch.cat([d.flatten() for d in surf_depths])
        nonzero = all_depths[all_depths > 0]
        depth_max = float(nonzero.quantile(0.95).item()) if nonzero.numel() > 0 else 10.0
    else:
        depth_max = args.depth_max
    print(f"[DM] Colormap normalization: depth_max = {depth_max:.3f}")

    print("[DM] Saving PNGs…")
    for i, cam in enumerate(tqdm(cams, desc="save")):
        name = f"{i:03d}"
        rgb = rgbs[i].permute(1, 2, 0).numpy()
        save_img_u8(rgb, os.path.join(out_root, "rgb", f"{name}.png"))

        a = alphas[i].squeeze(0).clamp(0, 1).numpy()
        save_img_u8(np.stack([a]*3, axis=-1), os.path.join(out_root, "alpha", f"{name}.png"))

        d = surf_depths[i].squeeze().numpy()
        np.save(os.path.join(out_root, "depth_raw", f"{name}.npy"), d.astype(np.float32))
        d_norm = np.clip(d / max(1e-6, depth_max), 0, 1)
        save_img_u8(convert_gray_to_cmap(d_norm, map_mode='turbo', revert=False),
                    os.path.join(out_root, "surf_depth", f"{name}.png"))

        if exp_depths[i] is not None:
            de = exp_depths[i].squeeze().numpy()
            de_norm = np.clip(de / max(1e-6, depth_max), 0, 1)
            save_img_u8(convert_gray_to_cmap(de_norm, map_mode='turbo', revert=False),
                        os.path.join(out_root, "depth_expected", f"{name}.png"))
        if med_depths[i] is not None:
            dm = med_depths[i].squeeze().numpy()
            dm_norm = np.clip(dm / max(1e-6, depth_max), 0, 1)
            save_img_u8(convert_gray_to_cmap(dm_norm, map_mode='turbo', revert=False),
                        os.path.join(out_root, "depth_median", f"{name}.png"))
        if normals[i] is not None:
            n = normals[i].squeeze()
            n_vis = (n * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy()
            save_img_u8(n_vis, os.path.join(out_root, "normal", f"{name}.png"))

        valid_pct = float((d > 0).mean()) * 100.0
        print(f"  view {i:03d}: depth min={d.min():.3f} max={d.max():.3f} "
              f"valid={valid_pct:.1f}%  alpha mean={a.mean():.3f}")

    print(f"[DM] Done.")


if __name__ == "__main__":
    main()
