"""
Sanity test: Compare 3D_SH_res (with zeroed MLP/hash) vs 2dgs (no ingp).
Both should produce identical output since the MLP residual is zero.

Usage:
    conda run -n nest_splatting python scripts/test_sh_res_vs_2dgs.py \
        --model_path outputs/mip_360/bonsai/3D_SH_res/35kiterfps5H01bc001op0sc1e3nsGS_10kfx_05wodr10_01thr
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
from argparse import ArgumentParser, Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from utils.render_utils import save_img_u8


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No args.pkl found in {model_path}")


def zero_ingp(ingp):
    """Zero out hash encoding and MLP weights so residual = 0."""
    with torch.no_grad():
        if hasattr(ingp, 'hash_encoding') and ingp.hash_encoding is not None:
            ingp.hash_encoding.embeddings.zero_()
            print(f"  Zeroed hash_encoding: {ingp.hash_encoding.embeddings.shape}")
        if hasattr(ingp, 'mlp_fused') and ingp.mlp_fused is not None:
            for m in ingp.mlp_fused:
                if hasattr(m, 'weight'):
                    m.weight.zero_()
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.zero_()
            print(f"  Zeroed mlp_fused")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    eval_args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Load training config (same pattern as eval_from_checkpoint.py)
    print(f"\n[LOAD] Loading config from: {eval_args.model_path}")
    args = load_training_config(eval_args.model_path)
    args.model_path = eval_args.model_path
    args.eval = True

    # Find iteration
    iteration = eval_args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
        else:
            # Fall back to point cloud dirs
            pc_dirs = glob.glob(os.path.join(eval_args.model_path, "point_cloud/iteration_*"))
            iterations = [int(os.path.basename(d).replace("iteration_", "")) for d in pc_dirs]
            iteration = max(iterations)
    print(f"[LOAD] Using iteration: {iteration}")

    # Load YAML config
    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    cfg_model = Config(config_yaml_path)

    # Extract params using parser (same as eval_from_checkpoint.py)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP
    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(eval_args.model_path, iteration)
    ingp.set_active_levels(iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Background
    bg_color = torch.ones(3, device="cuda") if dataset.white_background else torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    test_cams = scene.getTestCameras()
    n_views = min(eval_args.num_views, len(test_cams))

    print("=" * 70)
    print("  3D_SH_res (zero residual) vs 2dgs Forward Comparison")
    print("=" * 70)
    print(f"  Gaussians: {len(gaussians.get_xyz)}")
    print(f"  SH degree: {gaussians.active_sh_degree}")
    print(f"  Test views: {n_views}")

    # Zero out hash+MLP
    print("\n[MODE A] Zeroing hash+MLP for SH-only rendering via 3D_SH_res rasterizer...")
    zero_ingp(ingp)

    print("[MODE B] Same Gaussians, ingp=None (pure 2dgs path, main rasterizer)...\n")

    print("-" * 90)
    print(f"{'View':>6} | {'A (SH_res+zero)':>16} | {'B (2dgs)':>16} | {'Diff (L1)':>12} | {'Max Diff':>10} | {'PSNR_A':>8} | {'PSNR_B':>8}")
    print("-" * 90)

    all_diffs = []
    for i in range(n_views):
        cam = test_cams[i]
        gt = cam.original_image.cuda()

        # Mode A: 3D_SH_res with zeroed residual
        pkg_a = render(cam, gaussians, pipe, bg_color, ingp=ingp,
                       beta=beta, iteration=iteration, cfg=cfg_model, is_training=False)
        img_a = torch.clamp(pkg_a['render'], 0.0, 1.0)
        alpha_a = pkg_a['rend_alpha']

        # Mode B: same Gaussians, no ingp (pure SH via main diff_surfel_rasterization)
        pkg_b = render(cam, gaussians, pipe, bg_color, ingp=None,
                       beta=beta, iteration=iteration, cfg=cfg_model, is_training=False)
        img_b = torch.clamp(pkg_b['render'], 0.0, 1.0)
        alpha_b = pkg_b['rend_alpha']

        # Compare
        diff = (img_a - img_b).abs()
        l1_diff = diff.mean().item()
        max_diff = diff.max().item()
        psnr_a = psnr(img_a, gt).mean().item()
        psnr_b = psnr(img_b, gt).mean().item()
        alpha_diff = (alpha_a - alpha_b).abs().max().item()

        all_diffs.append(l1_diff)
        print(f"{i:>6} | {img_a.mean().item():>16.6f} | {img_b.mean().item():>16.6f} | {l1_diff:>12.8f} | {max_diff:>10.6f} | {psnr_a:>8.2f} | {psnr_b:>8.2f}")

        if i == 0:
            out_dir = os.path.join(eval_args.model_path, 'sanity_test')
            os.makedirs(out_dir, exist_ok=True)
            save_img_u8(img_a.permute(1, 2, 0).cpu().numpy(), os.path.join(out_dir, 'view0_sh_res_zero.png'))
            save_img_u8(img_b.permute(1, 2, 0).cpu().numpy(), os.path.join(out_dir, 'view0_2dgs.png'))
            save_img_u8((diff * 10).clamp(0, 1).permute(1, 2, 0).cpu().numpy(), os.path.join(out_dir, 'view0_diff_10x.png'))
            print(f"\n  Saved images to {out_dir}/")
            print(f"  Alpha diff (view 0): max={alpha_diff:.8f}\n")

    print("-" * 90)
    mean_diff = np.mean(all_diffs)
    print(f"{'Mean':>6} | {'':>16} | {'':>16} | {mean_diff:>12.8f}")

    if mean_diff < 1e-5:
        print("\nMATCH: Forward pass is identical (diff < 1e-5)")
    elif mean_diff < 1e-3:
        print(f"\nCLOSE: Small difference (mean L1 = {mean_diff:.6f})")
    else:
        print(f"\nMISMATCH: Significant difference (mean L1 = {mean_diff:.6f})")

    # ===== Optional: Backward pass comparison =====
    if eval_args.backward:
        print("\n" + "=" * 70)
        print("  Backward Pass Comparison (unit gradient)")
        print("=" * 70)

        torch.set_grad_enabled(True)
        cam = test_cams[0]

        # Mode A backward
        gaussians._xyz.requires_grad_(True)
        pkg_a = render(cam, gaussians, pipe, bg_color, ingp=ingp,
                       beta=beta, iteration=iteration, cfg=cfg_model, is_training=False)
        img_a = pkg_a['render']
        unit_grad = torch.ones_like(img_a)
        img_a.backward(unit_grad)
        grad_xyz_a = gaussians._xyz.grad.clone()
        gaussians._xyz.grad = None

        # Mode B backward
        pkg_b = render(cam, gaussians, pipe, bg_color, ingp=None,
                       beta=beta, iteration=iteration, cfg=cfg_model, is_training=False)
        img_b = pkg_b['render']
        img_b.backward(unit_grad)
        grad_xyz_b = gaussians._xyz.grad.clone()

        grad_diff = (grad_xyz_a - grad_xyz_b).abs()
        print(f"  xyz grad diff: mean={grad_diff.mean().item():.8f}, max={grad_diff.max().item():.8f}")
        print(f"  xyz grad A: norm={grad_xyz_a.norm().item():.6f}")
        print(f"  xyz grad B: norm={grad_xyz_b.norm().item():.6f}")

        cos_sim = torch.nn.functional.cosine_similarity(
            grad_xyz_a.flatten().unsqueeze(0),
            grad_xyz_b.flatten().unsqueeze(0)).item()
        print(f"  Cosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    main()
