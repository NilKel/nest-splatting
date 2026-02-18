#!/usr/bin/env python3
"""
Test 3D_direct_fused / 3D_direct_lean forward pass only.

Requires only the lean library (diff_surfel_3D), not the main rasterizer.

Usage:
    USE_LEAN_RASTERIZER=1 python scripts/test_fused_lean.py --model_path outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8
"""

import os
import sys
import json
import pickle
import torch
import numpy as np

# Force lean rasterizer mode
os.environ['USE_LEAN_RASTERIZER'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render, LEAN_RASTERIZER_AVAILABLE
from hash_encoder.modules import INGP
from hash_encoder.config import Config
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
    parser = ArgumentParser(description="Test 3D_direct_fused/lean forward pass")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--camera_idx", type=int, default=0,
                       help="Test camera index to render")
    parser.add_argument("--save_images", action="store_true",
                       help="Save rendered images to disk")
    parser.add_argument("--method", type=str, default="3D_direct_lean",
                       choices=["3D_direct_fused", "3D_direct_lean"],
                       help="Mode to test")

    eval_args = parser.parse_args()

    print(f"\n[TEST] Using lean rasterizer: {LEAN_RASTERIZER_AVAILABLE}")
    if not LEAN_RASTERIZER_AVAILABLE:
        print("[ERROR] Lean rasterizer (diff_surfel_3D) not available!")
        return

    # Load training config
    print(f"\n[TEST] Loading config from: {eval_args.model_path}")
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
    print(f"[CONFIG] Testing method: {eval_args.method}")
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
    from arguments import ModelParams, PipelineParams
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Create INGP model with fused/lean method
    args_fused = Namespace(**vars(args))
    args_fused.method = eval_args.method
    ingp = INGP(cfg_model, args=args_fused).to('cuda')
    ingp.load_model(eval_args.model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Get test camera
    cameras = scene.getTestCameras()
    if eval_args.camera_idx >= len(cameras):
        print(f"[ERROR] Camera index {eval_args.camera_idx} out of range (0-{len(cameras)-1})")
        return

    cam = cameras[eval_args.camera_idx]
    print(f"\n[TEST] Using test camera {eval_args.camera_idx}: {cam.image_name}")
    print(f"[TEST] Resolution: {cam.image_width}x{cam.image_height}")
    print(f"[TEST] Num Gaussians: {len(gaussians.get_xyz):,}")

    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # ========== RENDER WITH FUSED/LEAN ==========
    print("\n" + "="*60)
    print(f"RENDERING WITH {eval_args.method}")
    print("="*60)

    with torch.no_grad():
        render_pkg = render(cam, gaussians, pipe, background, ingp=ingp,
                           beta=beta, iteration=iteration, cfg=cfg_model)
        img = render_pkg["render"].clone()

    print(f"[{eval_args.method}] Output shape: {img.shape}")
    print(f"[{eval_args.method}] Output range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"[{eval_args.method}] Output mean: {img.mean():.4f}")

    # Ground truth comparison
    gt = cam.original_image.to("cuda")
    mse = ((img - gt) ** 2).mean()
    psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else float('inf')

    print(f"\n[RESULT] PSNR vs GT: {psnr:.2f} dB")
    print(f"[RESULT] MSE vs GT: {mse:.6f}")

    # Quick sanity checks
    print("\n[SANITY CHECKS]")
    print(f"  - Image has valid values: {not torch.isnan(img).any() and not torch.isinf(img).any()}")
    print(f"  - Image in [0,1] range: {img.min() >= 0 and img.max() <= 1}")
    print(f"  - Image not all zeros: {img.sum() > 0}")
    print(f"  - Image not all ones: {img.mean() < 0.99}")

    # Save images if requested
    if eval_args.save_images:
        output_dir = os.path.join(eval_args.model_path, "lean_test")
        os.makedirs(output_dir, exist_ok=True)

        img_np = img.permute(1, 2, 0).cpu().numpy()
        gt_np = gt.permute(1, 2, 0).cpu().numpy()
        diff_np = (img - gt).abs().permute(1, 2, 0).cpu().numpy()
        diff_scaled = np.clip(diff_np * 10, 0, 1)

        save_img_u8(img_np, os.path.join(output_dir, f"{eval_args.method}.png"))
        save_img_u8(gt_np, os.path.join(output_dir, "ground_truth.png"))
        save_img_u8(diff_scaled, os.path.join(output_dir, "diff_10x.png"))

        print(f"\n[SAVE] Images saved to {output_dir}/")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
