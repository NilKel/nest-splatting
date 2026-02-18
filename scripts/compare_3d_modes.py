#!/usr/bin/env python3
"""
Compare forward pass outputs of 3D_direct vs 3D_direct_fused modes.

Usage:
    python scripts/compare_3d_modes.py --model_path outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8
"""

import os
import sys
import json
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
        return args

    raise FileNotFoundError(f"No training config found in {model_path}")


def main():
    parser = ArgumentParser(description="Compare 3D_direct vs 3D_direct_fused forward pass")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained 3D_direct model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--camera_idx", type=int, default=0,
                       help="Test camera index to render")
    parser.add_argument("--save_images", action="store_true",
                       help="Save rendered images to disk")

    eval_args = parser.parse_args()

    # Load training config
    print(f"\n[COMPARE] Loading config from: {eval_args.model_path}")
    args = load_training_config(eval_args.model_path)
    args.model_path = eval_args.model_path
    args.eval = True

    # Load YAML config
    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    print(f"[CONFIG] Original method: {args.method}")
    print(f"[CONFIG] Hybrid levels: {getattr(args, 'hybrid_levels', 'N/A')}")

    # Find iteration
    iteration = eval_args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth checkpoints found")

    # Setup model params
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load TWO separate INGP models - one for each mode
    # This ensures both mlp_3D_direct and mlp_fused are properly initialized

    # Model 1: 3D_direct mode
    args_direct = Namespace(**vars(args))
    args_direct.method = "3D_direct"
    ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
    ingp_direct.load_model(eval_args.model_path, iteration)

    # Model 2: 3D_direct_lean mode (uses lean library with viewdirs_enc)
    args_fused = Namespace(**vars(args))
    args_fused.method = "3D_direct_lean"
    ingp_fused = INGP(cfg_model, args=args_fused).to('cuda')
    ingp_fused.load_model(eval_args.model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_direct.set_active_levels(iteration)
    ingp_fused.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Get test camera
    cameras = scene.getTestCameras()
    if eval_args.camera_idx >= len(cameras):
        print(f"[ERROR] Camera index {eval_args.camera_idx} out of range (0-{len(cameras)-1})")
        return

    cam = cameras[eval_args.camera_idx]
    print(f"\n[COMPARE] Using test camera {eval_args.camera_idx}: {cam.image_name}")
    print(f"[COMPARE] Resolution: {cam.image_width}x{cam.image_height}")
    print(f"[COMPARE] Num Gaussians: {len(gaussians.get_xyz):,}")

    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # ========== RENDER WITH 3D_direct ==========
    print("\n" + "="*60)
    print("RENDERING WITH 3D_direct")
    print("="*60)

    with torch.no_grad():
        render_pkg_direct = render(cam, gaussians, pipe, background, ingp=ingp_direct,
                                   beta=beta, iteration=iteration, cfg=cfg_model)
        img_direct = render_pkg_direct["render"].clone()

    print(f"[3D_direct] Output shape: {img_direct.shape}")
    print(f"[3D_direct] Output range: [{img_direct.min():.4f}, {img_direct.max():.4f}]")
    print(f"[3D_direct] Output mean: {img_direct.mean():.4f}")

    # ========== RENDER WITH 3D_direct_lean ==========
    print("\n" + "="*60)
    print("RENDERING WITH 3D_direct_lean")
    print("="*60)

    with torch.no_grad():
        render_pkg_fused = render(cam, gaussians, pipe, background, ingp=ingp_fused,
                                  beta=beta, iteration=iteration, cfg=cfg_model)
        img_fused = render_pkg_fused["render"].clone()

    print(f"[3D_direct_lean] Output shape: {img_fused.shape}")
    print(f"[3D_direct_lean] Output range: [{img_fused.min():.4f}, {img_fused.max():.4f}]")
    print(f"[3D_direct_lean] Output mean: {img_fused.mean():.4f}")

    # ========== COMPARE ==========
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    diff = (img_direct - img_fused).abs()

    print(f"Absolute difference:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Max:  {diff.max():.6f}")
    print(f"  Min:  {diff.min():.6f}")
    print(f"  Std:  {diff.std():.6f}")

    # Per-channel comparison
    for c, name in enumerate(['R', 'G', 'B']):
        ch_diff = diff[c]
        print(f"  {name} channel - mean: {ch_diff.mean():.6f}, max: {ch_diff.max():.6f}")

    # Check if they're close
    rtol = 1e-4
    atol = 1e-4
    is_close = torch.allclose(img_direct, img_fused, rtol=rtol, atol=atol)
    print(f"\ntorch.allclose (rtol={rtol}, atol={atol}): {is_close}")

    # MSE and PSNR between modes
    mse = ((img_direct - img_fused) ** 2).mean()
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"MSE between modes: {mse:.8f}")
        print(f"PSNR between modes: {psnr:.2f} dB")
    else:
        print("MSE: 0 (identical)")

    # Ground truth comparison
    gt = cam.original_image.to("cuda")
    mse_direct_gt = ((img_direct - gt) ** 2).mean()
    mse_fused_gt = ((img_fused - gt) ** 2).mean()
    psnr_direct = 10 * torch.log10(1.0 / mse_direct_gt) if mse_direct_gt > 0 else float('inf')
    psnr_fused = 10 * torch.log10(1.0 / mse_fused_gt) if mse_fused_gt > 0 else float('inf')

    print(f"\nPSNR vs GT:")
    print(f"  3D_direct:      {psnr_direct:.2f} dB")
    print(f"  3D_direct_lean: {psnr_fused:.2f} dB")

    # Save images if requested
    if eval_args.save_images:
        output_dir = os.path.join(eval_args.model_path, "mode_comparison")
        os.makedirs(output_dir, exist_ok=True)

        img_direct_np = img_direct.permute(1, 2, 0).cpu().numpy()
        img_fused_np = img_fused.permute(1, 2, 0).cpu().numpy()
        gt_np = gt.permute(1, 2, 0).cpu().numpy()
        diff_np = diff.permute(1, 2, 0).cpu().numpy()

        # Scale diff for visibility
        diff_scaled = np.clip(diff_np * 10, 0, 1)  # 10x amplification

        save_img_u8(img_direct_np, os.path.join(output_dir, "3D_direct.png"))
        save_img_u8(img_fused_np, os.path.join(output_dir, "3D_direct_fused.png"))
        save_img_u8(gt_np, os.path.join(output_dir, "ground_truth.png"))
        save_img_u8(diff_scaled, os.path.join(output_dir, "diff_10x.png"))

        print(f"\n[SAVE] Images saved to {output_dir}/")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
