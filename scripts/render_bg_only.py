#!/usr/bin/env python3
"""
Render just the background hashgrid output (no Gaussians), or FG-only (no BG).

This script loads a checkpoint with a trained background hashgrid and renders
the background features through the MLP decoder for each camera view.

Usage:
    python scripts/render_bg_only.py -m /path/to/checkpoint --iteration 35000
    python scripts/render_bg_only.py -m /path/to/checkpoint --iteration 35000 --fg_only
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.background import SphereHashGridBackground
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.point_utils import cam2rays
from utils.render_utils import save_img_u8
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args
from gaussian_renderer import render


def render_bg_only(camera, bg_hashgrid, ingp):
    """Render just the background hashgrid features through the MLP."""
    H, W = camera.image_height, camera.image_width

    # Get ray directions for each pixel
    rays_d, rays_o = cam2rays(camera)
    ray_unit = F.normalize(rays_d, dim=-1).float()  # (H*W, 3)

    # Query background hashgrid with camera position for position-aware sphere intersection
    ray_origins_bg = rays_o.unsqueeze(0).expand(ray_unit.shape[0], -1)  # (H*W, 3)
    bg_features = bg_hashgrid(ray_unit, ray_origins_bg)  # (H*W, feat_dim)

    # Decode through MLP
    bg_rgb = ingp.rgb_decode(bg_features, ray_unit)  # (H*W, 3)
    bg_rgb = bg_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)

    return torch.clamp(bg_rgb, 0.0, 1.0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render background hashgrid only or FG only")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--yaml", type=str, default="tiny")
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--hybrid_levels", type=int, default=3)
    parser.add_argument("--kernel", type=str, default="gaussian")
    parser.add_argument("--fg_only", action="store_true",
                        help="Render FG only (Gaussians + main hashgrid, no BG hashgrid)")
    args = get_combined_args(parser)

    exp_path = args.model_path
    iteration = args.iteration

    # Load args.json to get training configuration
    args_json_path = os.path.join(exp_path, "args.json")
    saved_args = {}
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            saved_args = json.load(f)
        # Override relevant args from saved config
        args.yaml = saved_args.get("yaml", args.yaml)
        args.method = saved_args.get("method", args.method)
        args.hybrid_levels = saved_args.get("hybrid_levels", args.hybrid_levels)
        args.kernel = saved_args.get("kernel", args.kernel)
        print(f"[INFO] Loaded config from {args_json_path}")
        print(f"[INFO] method={args.method}, hybrid_levels={args.hybrid_levels}, yaml={args.yaml}")

    # Load model config
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    # Load INGP model (for MLP decoder)
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(exp_path, iteration)
    ingp_model.set_active_levels(iteration)  # Initialize active_levels
    ingp_model.eval()

    # Get feature dimensions from INGP
    total_levels = cfg_model.encoding.levels
    level_dim = cfg_model.encoding.hashgrid.dim
    feat_dim = total_levels * level_dim

    # Load scene for cameras and Gaussians
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Set up Gaussians
    gaussians.XYZ_TYPE = "UV"
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta

    # Set kernel type
    if args.kernel == "beta":
        gaussians.kernel_type = "beta"
    elif args.kernel == "flex":
        gaussians.kernel_type = "flex"

    bg_color = [0, 0, 0]  # Black background for FG-only
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.fg_only:
        # FG-only mode: render Gaussians + main hashgrid, no BG hashgrid
        output_dir = os.path.join(exp_path, f"fg_only_{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[INFO] Rendering FG-only (no BG hashgrid)")

        # Render test cameras
        if not args.skip_test and len(scene.getTestCameras()) > 0:
            test_output_dir = os.path.join(output_dir, "test")
            os.makedirs(test_output_dir, exist_ok=True)

            print(f"\n[INFO] Rendering {len(scene.getTestCameras())} test cameras (FG only)...")
            with torch.no_grad():
                for idx, cam in enumerate(tqdm(scene.getTestCameras())):
                    # Render with Gaussians + main hashgrid, but NO bg_hashgrid
                    render_pkg = render(cam, gaussians, pipe, background,
                                       ingp=ingp_model, beta=beta, iteration=iteration,
                                       cfg=cfg_model, bg_hashgrid=None)

                    fg_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["rend_alpha"]

                    # Save FG render
                    img_name = f"{idx:03d}_{cam.image_name}_fg.png"
                    save_img_u8(fg_rgb.permute(1, 2, 0).cpu().numpy(),
                               os.path.join(test_output_dir, img_name))

                    # Save alpha
                    alpha_name = f"{idx:03d}_{cam.image_name}_alpha.png"
                    alpha_rgb = alpha.repeat(3, 1, 1)
                    save_img_u8(alpha_rgb.permute(1, 2, 0).cpu().numpy(),
                               os.path.join(test_output_dir, alpha_name))

            print(f"[INFO] Saved FG-only test renders to {test_output_dir}")

        # Render train cameras
        if not args.skip_train:
            train_output_dir = os.path.join(output_dir, "train")
            os.makedirs(train_output_dir, exist_ok=True)

            print(f"\n[INFO] Rendering {len(scene.getTrainCameras())} train cameras (FG only)...")
            with torch.no_grad():
                for idx, cam in enumerate(tqdm(scene.getTrainCameras())):
                    render_pkg = render(cam, gaussians, pipe, background,
                                       ingp=ingp_model, beta=beta, iteration=iteration,
                                       cfg=cfg_model, bg_hashgrid=None)

                    fg_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    img_name = f"{idx:03d}_{cam.image_name}_fg.png"
                    save_img_u8(fg_rgb.permute(1, 2, 0).cpu().numpy(),
                               os.path.join(train_output_dir, img_name))

            print(f"[INFO] Saved FG-only train renders to {train_output_dir}")

        print(f"\n[DONE] FG-only renders saved to {output_dir}")

    else:
        # BG-only mode: render just the background hashgrid
        bg_hashgrid_path = os.path.join(exp_path, f"bg_hashgrid_{iteration}.pth")
        if not os.path.exists(bg_hashgrid_path):
            print(f"[ERROR] Background hashgrid checkpoint not found: {bg_hashgrid_path}")
            print(f"[ERROR] This checkpoint may not have been trained with --background hashgrid")
            sys.exit(1)

        # Create background hashgrid with matching dimensions
        bg_hashgrid = SphereHashGridBackground(
            num_levels=total_levels,
            level_dim=level_dim,
            log2_hashmap_size=saved_args.get("bg_hashgrid_size", 19),
            base_resolution=16,
            desired_resolution=saved_args.get("bg_hashgrid_res", 512),
            sphere_radius=saved_args.get("bg_hashgrid_radius", 500.0),
        ).cuda()
        bg_hashgrid.load_model(exp_path, iteration)
        bg_hashgrid.eval()

        print(f"[INFO] Loaded background hashgrid from {bg_hashgrid_path}")
        print(f"[INFO] Feature dim: {feat_dim} ({total_levels} levels x {level_dim} dim)")

        # Create output directory
        bg_output_dir = os.path.join(exp_path, f"bg_only_{iteration}")
        os.makedirs(bg_output_dir, exist_ok=True)

        # Render test cameras
        if not args.skip_test and len(scene.getTestCameras()) > 0:
            test_output_dir = os.path.join(bg_output_dir, "test")
            os.makedirs(test_output_dir, exist_ok=True)

            print(f"\n[INFO] Rendering {len(scene.getTestCameras())} test cameras (BG only)...")
            with torch.no_grad():
                for idx, cam in enumerate(tqdm(scene.getTestCameras())):
                    bg_rgb = render_bg_only(cam, bg_hashgrid, ingp_model)

                    # Save
                    img_name = f"{idx:03d}_{cam.image_name}_bg.png"
                    save_img_u8(bg_rgb.permute(1, 2, 0).cpu().numpy(),
                               os.path.join(test_output_dir, img_name))

            print(f"[INFO] Saved test background renders to {test_output_dir}")

        # Render train cameras
        if not args.skip_train:
            train_output_dir = os.path.join(bg_output_dir, "train")
            os.makedirs(train_output_dir, exist_ok=True)

            print(f"\n[INFO] Rendering {len(scene.getTrainCameras())} train cameras (BG only)...")
            with torch.no_grad():
                for idx, cam in enumerate(tqdm(scene.getTrainCameras())):
                    bg_rgb = render_bg_only(cam, bg_hashgrid, ingp_model)

                    # Save
                    img_name = f"{idx:03d}_{cam.image_name}_bg.png"
                    save_img_u8(bg_rgb.permute(1, 2, 0).cpu().numpy(),
                               os.path.join(train_output_dir, img_name))

            print(f"[INFO] Saved train background renders to {train_output_dir}")

        print(f"\n[DONE] Background renders saved to {bg_output_dir}")
