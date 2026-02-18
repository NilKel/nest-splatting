#!/usr/bin/env python3
"""
Compare backward pass gradients between cat mode and lean (3D_direct_lean) mode.

This tests:
1. Geometry gradients: opacity, position, rotation, scale
2. Per-Gaussian feature gradients
3. MLP weight gradients (cat=tcnn, lean=CUDA)

Run with: conda run -n nest_splatting python scripts/test_backward_comparison.py
"""

import os
import sys
import torch
import numpy as np
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def cos_sim(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def rel_error(a, b):
    """Compute relative error."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    diff = (a_flat - b_flat).abs()
    scale = (a_flat.abs() + b_flat.abs()).clamp(min=1e-8) / 2
    return (diff / scale).mean().item()


def load_model_and_camera(model_path: str, iteration: int = 30000):
    """Load trained model and create a test camera."""

    # Load checkpoint
    ckpt_path = f'{model_path}/ngp_{iteration}.pth'
    if not os.path.exists(ckpt_path):
        # Try chkpnt path
        ckpt_path = f'{model_path}/chkpnt_{iteration}.pth'

    print(f"Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cuda')

    # Create GaussianModel - extract config from checkpoint
    model_state = ckpt.get('model_state_dict', ckpt)

    # Infer gaussian_dim from checkpoint
    if 'gaussian_colors' in model_state:
        gaussian_dim = model_state['gaussian_colors'].shape[-1]
    elif '_features_dc' in model_state:
        gaussian_dim = 3  # SH mode
    else:
        gaussian_dim = 20  # Default hybrid levels * 4

    gaussians = GaussianModel(sh_degree=3, gaussian_dim=gaussian_dim)

    # Load state dict
    gaussians.load_state_dict(model_state, strict=False)
    gaussians.cuda()

    # Load INGP if available
    ingp = None
    if 'ingp_state_dict' in ckpt:
        # Need to create INGP - get config from checkpoint
        from hash_encoder.modules import INGP

        ingp_state = ckpt['ingp_state_dict']
        # Infer architecture from state dict
        hash_features_shape = ingp_state.get('hash_encoding.embeddings',
                                              ingp_state.get('hash_encoding.params', torch.empty(0))).shape

        ingp = INGP(
            num_levels=6,
            level_dim=4,
            per_level_scale=2.0,
            base_resolution=16,
            hash_log2_size=19,
            mlp_hidden_dim=32,
            mlp_output_dim=3,
            input_dim=40,
            output_type='rgb'  # or 'sh'
        )
        ingp.load_state_dict(ingp_state, strict=False)
        ingp.cuda()

    # Create test camera
    W, H = 400, 400
    fov_x = 0.8  # ~46 degrees
    fov_y = fov_x * H / W

    # Camera at origin looking at z
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # Move back

    proj_matrix = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y
    ).T.cuda()

    world_view = getWorld2View2(R, T)
    full_proj = (torch.tensor(world_view).cuda() @ proj_matrix).T

    camera = Camera(
        colmap_id=0, R=R, T=T,
        FoVx=fov_x, FoVy=fov_y,
        image=torch.zeros(3, H, W),
        gt_alpha_mask=None,
        image_name="test",
        uid=0,
        data_device='cuda'
    )

    return gaussians, ingp, camera


def run_backward_comparison(model_path: str):
    """Run backward pass comparison between cat and lean modes."""

    print("=" * 60)
    print("BACKWARD PASS COMPARISON: CAT vs LEAN")
    print("=" * 60)

    # Load model
    gaussians, ingp, camera = load_model_and_camera(model_path)

    if ingp is None:
        print("ERROR: No INGP found in checkpoint. Need a cat/3D mode model.")
        return

    print(f"\nLoaded {gaussians._xyz.shape[0]} Gaussians")
    print(f"Gaussian features: {gaussians.gaussian_colors.shape}")

    # Create args for both modes
    base_args = Namespace(
        sh_degree=3,
        white_background=False,
        debug=False,
        scale_invariant=False,
        max_abs_split_scale=0.3,
        soft_beta=False,
    )

    bg_color = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    # =========================================================================
    # TEST CAT MODE BACKWARD
    # =========================================================================
    print("\n" + "=" * 60)
    print("CAT MODE BACKWARD")
    print("=" * 60)

    args_cat = Namespace(**vars(base_args), method='cat')

    # Zero all grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians.gaussian_colors.grad = None

    ingp.hash_encoding.embeddings.grad = None
    for param in ingp.mlp_3D_direct.parameters():
        param.grad = None

    # Forward
    render_pkg_cat = render(camera, gaussians, args_cat, bg_color, ingp=ingp)
    image_cat = render_pkg_cat['render']

    print(f"Rendered image: {image_cat.shape}, range=[{image_cat.min():.3f}, {image_cat.max():.3f}]")

    # Backward with grad=1
    loss_cat = image_cat.sum()
    loss_cat.backward()

    # Store cat gradients
    grad_cat = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scaling': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotation': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
        'features': gaussians.gaussian_colors.grad.clone() if gaussians.gaussian_colors.grad is not None else None,
        'hash': ingp.hash_encoding.embeddings.grad.clone() if ingp.hash_encoding.embeddings.grad is not None else None,
    }

    # Get MLP gradients for cat mode (tcnn)
    if hasattr(ingp, 'mlp_3D_direct') and ingp.mlp_3D_direct.params.grad is not None:
        grad_cat['mlp_params'] = ingp.mlp_3D_direct.params.grad.clone()

    print("\nCat mode gradients:")
    for name, grad in grad_cat.items():
        if grad is not None:
            print(f"  {name}: mean={grad.abs().mean():.6f}, max={grad.abs().max():.6f}, "
                  f"nonzero={(grad.abs() > 1e-8).sum().item()}/{grad.numel()}")
        else:
            print(f"  {name}: None")

    # =========================================================================
    # TEST LEAN MODE BACKWARD
    # =========================================================================
    print("\n" + "=" * 60)
    print("LEAN MODE (3D_direct_lean) BACKWARD")
    print("=" * 60)

    args_lean = Namespace(**vars(base_args), method='3D_direct_lean')

    # Zero all grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians.gaussian_colors.grad = None

    ingp.hash_encoding.embeddings.grad = None
    if hasattr(ingp, 'mlp_fused'):
        for param in ingp.mlp_fused.parameters():
            param.grad = None

    # Make sure MLP weights are synced
    ingp.sync_mlp_weights_to_fused()

    # Forward
    render_pkg_lean = render(camera, gaussians, args_lean, bg_color, ingp=ingp)
    image_lean = render_pkg_lean['render']

    print(f"Rendered image: {image_lean.shape}, range=[{image_lean.min():.3f}, {image_lean.max():.3f}]")

    # Compare forward pass images first
    image_diff = (image_cat.detach() - image_lean.detach()).abs()
    print(f"\nForward pass diff: mean={image_diff.mean():.6f}, max={image_diff.max():.6f}")

    # Backward with grad=1
    loss_lean = image_lean.sum()
    loss_lean.backward()

    # Store lean gradients
    grad_lean = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scaling': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotation': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
        'features': gaussians.gaussian_colors.grad.clone() if gaussians.gaussian_colors.grad is not None else None,
        'hash': ingp.hash_encoding.embeddings.grad.clone() if ingp.hash_encoding.embeddings.grad is not None else None,
    }

    # Get MLP gradients for lean mode (fused CUDA)
    if hasattr(ingp, 'mlp_fused'):
        # Gradients should be in mlp_fused parameters after backward
        mlp_grads = []
        for param in ingp.mlp_fused.parameters():
            if param.grad is not None:
                mlp_grads.append(param.grad.flatten())
        if mlp_grads:
            grad_lean['mlp_params'] = torch.cat(mlp_grads)

    print("\nLean mode gradients:")
    for name, grad in grad_lean.items():
        if grad is not None:
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            print(f"  {name}: mean={grad.abs().mean():.6f}, max={grad.abs().max():.6f}, "
                  f"nonzero={(grad.abs() > 1e-8).sum().item()}/{grad.numel()}, "
                  f"NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  {name}: None")

    # =========================================================================
    # COMPARE GRADIENTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("GRADIENT COMPARISON (CAT vs LEAN)")
    print("=" * 60)

    print("\n{:<12} {:>10} {:>12} {:>10} {:>10}".format(
        "Gradient", "cos_sim", "rel_error", "cat_norm", "lean_norm"))
    print("-" * 60)

    for name in ['xyz', 'scaling', 'rotation', 'opacity', 'features', 'hash']:
        g_cat = grad_cat.get(name)
        g_lean = grad_lean.get(name)

        if g_cat is None or g_lean is None:
            print(f"{name:<12} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
            continue

        cs = cos_sim(g_cat, g_lean)
        re = rel_error(g_cat, g_lean)
        cat_norm = g_cat.norm().item()
        lean_norm = g_lean.norm().item()

        # Check for issues
        status = "OK" if cs > 0.99 else "CHECK" if cs > 0.9 else "FAIL"

        print(f"{name:<12} {cs:>10.4f} {re:>12.6f} {cat_norm:>10.4f} {lean_norm:>10.4f}  {status}")

    # MLP gradient comparison (if available)
    if 'mlp_params' in grad_cat and 'mlp_params' in grad_lean:
        print("\n" + "=" * 60)
        print("MLP GRADIENT COMPARISON")
        print("=" * 60)

        g_mlp_cat = grad_cat['mlp_params'].float()
        g_mlp_lean = grad_lean['mlp_params'].float()

        print(f"Cat MLP grad shape: {g_mlp_cat.shape}")
        print(f"Lean MLP grad shape: {g_mlp_lean.shape}")

        # They may have different layouts - compare what we can
        min_len = min(len(g_mlp_cat), len(g_mlp_lean))
        if min_len > 0:
            cs = cos_sim(g_mlp_cat[:min_len], g_mlp_lean[:min_len])
            re = rel_error(g_mlp_cat[:min_len], g_mlp_lean[:min_len])
            print(f"MLP grad cos_sim: {cs:.4f}")
            print(f"MLP grad rel_error: {re:.6f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    # Default model path
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Trying alternate paths...")

        # Try cat mode model
        alt_paths = [
            'outputs/nerf_synthetic/chair/cat/test_cat',
            'outputs/nerf_synthetic/lego/cat/test_cat',
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            print("No model found. Run training first.")
            return

    run_backward_comparison(model_path)


if __name__ == "__main__":
    main()
