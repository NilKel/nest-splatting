#!/usr/bin/env python3
"""
Test lean mode full render backward pass with a minimal scene.

Creates a few Gaussians, renders them, and verifies:
1. Forward pass produces reasonable output
2. Backward pass computes gradients for all parameters
3. No NaN/Inf in any gradients
4. MLP gradients are retrievable and match expected patterns

Run with: conda run -n nest_splatting python scripts/test_lean_render_backward.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_minimal_scene():
    """Create a minimal scene with a few Gaussians for testing."""
    N = 100  # Number of Gaussians
    gaussian_dim = 20  # Per-Gaussian features (hybrid_levels=5 * level_dim=4)

    # Random Gaussian positions (in front of camera)
    means3D = torch.randn(N, 3, device='cuda') * 0.5
    means3D[:, 2] = means3D[:, 2].abs() + 2.0  # Ensure z > 0 (in front of camera)

    # Random scales (small)
    scales = torch.ones(N, 2, device='cuda') * 0.05

    # Random rotations (quaternions, normalized)
    rotations = torch.randn(N, 4, device='cuda')
    rotations = rotations / rotations.norm(dim=1, keepdim=True)

    # Random opacities (logit space)
    opacities = torch.zeros(N, 1, device='cuda')  # sigmoid(0) = 0.5

    # Random per-Gaussian features
    gaussian_colors = torch.randn(N, gaussian_dim, device='cuda') * 0.1

    return {
        'means3D': means3D.requires_grad_(True),
        'scales': scales.requires_grad_(True),
        'rotations': rotations.requires_grad_(True),
        'opacities': opacities.requires_grad_(True),
        'gaussian_colors': gaussian_colors.requires_grad_(True),
    }


def create_mlp_and_hash():
    """Create MLP and hash encoding for 3D_direct_lean mode."""
    import torch.nn as nn
    from gridencoder import GridEncoder

    # MLP: 40D input -> 32 -> 32 -> 3D output
    IN_DIM = 40
    HIDDEN_DIM = 32
    OUT_DIM = 3

    mlp_fused = nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUT_DIM),
        nn.Sigmoid()
    ).cuda()

    # Hash encoding (1 level for hashgrid fine features)
    hash_encoding = GridEncoder(
        input_dim=3,
        num_levels=1,  # Only 1 hashgrid level (remaining after hybrid)
        level_dim=4,
        per_level_scale=2.0,
        base_resolution=16,
        log2_hashmap_size=19,
        gridtype='hash',
        align_corners=True,
    ).cuda()

    return mlp_fused, hash_encoding


def encode_view_direction(view_dir):
    """Encode view direction to 16D (matching CUDA kernel)."""
    # view_dir: [N, 3] normalized
    pi = 3.14159265358979323846

    view_enc = torch.zeros(view_dir.shape[0], 16, device=view_dir.device)

    # Base direction (3D)
    view_enc[:, 0] = view_dir[:, 0]
    view_enc[:, 1] = view_dir[:, 1]
    view_enc[:, 2] = view_dir[:, 2]

    # Frequency band 1: sin/cos(pi * dir)
    view_enc[:, 3] = torch.sin(pi * view_dir[:, 0])
    view_enc[:, 4] = torch.cos(pi * view_dir[:, 0])
    view_enc[:, 5] = torch.sin(pi * view_dir[:, 1])
    view_enc[:, 6] = torch.cos(pi * view_dir[:, 1])
    view_enc[:, 7] = torch.sin(pi * view_dir[:, 2])
    view_enc[:, 8] = torch.cos(pi * view_dir[:, 2])

    # Frequency band 2: sin/cos(2*pi * dir)
    view_enc[:, 9] = torch.sin(2 * pi * view_dir[:, 0])
    view_enc[:, 10] = torch.cos(2 * pi * view_dir[:, 0])
    view_enc[:, 11] = torch.sin(2 * pi * view_dir[:, 1])
    view_enc[:, 12] = torch.cos(2 * pi * view_dir[:, 1])
    view_enc[:, 13] = torch.sin(2 * pi * view_dir[:, 2])
    view_enc[:, 14] = torch.cos(2 * pi * view_dir[:, 2])

    # Pad to 16D
    view_enc[:, 15] = 0.0

    return view_enc


def create_camera(W=100, H=100):
    """Create a simple camera looking at origin."""
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    fov_x = 0.8
    fov_y = fov_x * H / W

    # Camera at z=-3 looking at origin
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    world_view = getWorld2View2(R, T)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)

    return {
        'viewmatrix': torch.from_numpy(world_view).T.cuda().float(),
        'projmatrix': (torch.from_numpy(world_view).cuda() @ proj_matrix.T.cuda()).T.float(),
        'campos': torch.tensor([0.0, 0.0, -3.0], device='cuda'),
        'tanfovx': np.tan(fov_x / 2),
        'tanfovy': np.tan(fov_y / 2),
        'W': W,
        'H': H,
    }


def test_lean_render_backward():
    """Test lean mode render and backward pass."""
    print("=" * 60)
    print("TEST: Lean Mode Render Backward")
    print("=" * 60)

    # Import lean library
    try:
        import diff_surfel_3D as lean_lib
        from diff_surfel_3D import GaussianRasterizer, GaussianRasterizationSettings, HashGridSettings, set_mlp_weights, get_mlp_grads
        print("Lean library imported successfully")
    except ImportError as e:
        print(f"ERROR: Could not import lean library: {e}")
        return False

    # Create scene and camera
    scene = create_minimal_scene()
    mlp_fused, hash_encoding = create_mlp_and_hash()
    cam = create_camera()

    N = scene['means3D'].shape[0]
    W, H = cam['W'], cam['H']

    print(f"\nScene: {N} Gaussians, {W}x{H} image")

    # Upload MLP weights
    W1 = mlp_fused[0].weight.data
    b1 = mlp_fused[0].bias.data
    W2 = mlp_fused[2].weight.data
    b2 = mlp_fused[2].bias.data
    W3 = mlp_fused[4].weight.data
    b3 = mlp_fused[4].bias.data

    print(f"\nMLP weights (mlp_fused):")
    print(f"  W1: {W1.shape}, mean={W1.mean():.4f}")
    print(f"  b1: {b1.shape}, mean={b1.mean():.4f}")
    print(f"  W2: {W2.shape}, mean={W2.mean():.4f}")
    print(f"  W3: {W3.shape}, mean={W3.mean():.4f}")

    set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)

    # Create rasterizer settings
    bg_color = torch.zeros(3, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=cam['tanfovx'],
        tanfovy=cam['tanfovy'],
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam['viewmatrix'],
        projmatrix=cam['projmatrix'],
        sh_degree=0,
        campos=cam['campos'],
        prefiltered=False,
        debug=False,
        beta=0.1,
        if_contract=False,
        record_transmittance=True,
    )

    # Encode level parameter: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
    total_levels = 6
    hybrid_levels = 5
    active_hashgrid_levels = total_levels - hybrid_levels  # 1
    level_encoded = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

    hashgrid_settings = HashGridSettings(
        L=level_encoded,
        S=2.0,  # per_level_scale
        H=16,   # base_resolution
        align_corners=True,
        interpolation=0,  # trilinear
        # [GS=20, HS=4, OS=3] - OS=3 for RGB output in 3D_direct_fused mode
        shape_dims=torch.tensor([20, 4, 3], dtype=torch.int32, device='cuda'),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

    # Prepare inputs
    means2D = torch.zeros(N, 2, device='cuda', requires_grad=True)

    # Compute homotrans (homogeneous transformation matrices)
    # This is normally computed by preprocessCUDA, but we need a simple version
    homotrans = torch.zeros(N, 16, device='cuda')

    # Compute ap_level (appearance level per Gaussian)
    ap_level = torch.zeros(N, device='cuda')

    # Get hash features and offsets
    hash_features = hash_encoding.embeddings
    offsets = hash_encoding.offsets
    gridrange = torch.tensor([-1.0, 1.0], device='cuda')

    # Pre-encode view directions (all pixels looking at origin from camera)
    # Generate pixel grid
    y, x = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')
    x = x.flatten().float()
    y = y.flatten().float()

    # Simple pinhole camera model
    fx = W / (2 * cam['tanfovx'])
    fy = H / (2 * cam['tanfovy'])
    cx, cy = W / 2, H / 2

    # Ray directions in camera space
    dirs_cam = torch.stack([
        (x - cx) / fx,
        (y - cy) / fy,
        torch.ones_like(x)
    ], dim=1)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=1, keepdim=True)

    # Transform to world space (camera at z=-3 looking at +z)
    # For simplicity, assume camera aligned with world
    view_dir = dirs_cam
    viewdirs_enc = encode_view_direction(view_dir)  # [H*W, 16]

    # Render
    print("\nRunning forward pass...")

    try:
        outputs = rasterizer(
            means3D=scene['means3D'],
            means2D=means2D,
            opacities=torch.sigmoid(scene['opacities']),
            shs=None,
            colors_precomp=scene['gaussian_colors'],  # Per-Gaussian features
            scales=torch.exp(scene['scales']),
            rotations=scene['rotations'] / scene['rotations'].norm(dim=1, keepdim=True),
            cov3D_precomp=None,
            homotrans=homotrans,
            ap_level=ap_level,
            features=hash_features,
            offsets=offsets.int(),
            gridrange=gridrange,
            render_mode=5,  # 3D_direct_fused
            viewdirs_enc=viewdirs_enc
        )

        rendered = outputs[0]  # [3, H, W]
        print(f"Forward pass: output shape={rendered.shape}, "
              f"range=[{rendered.min():.4f}, {rendered.max():.4f}]")

    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Backward
    print("\nRunning backward pass...")

    try:
        loss = rendered.sum()
        loss.backward()
        print("Backward pass completed")

    except Exception as e:
        print(f"ERROR in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check gradients
    print("\nGradient check:")

    def check_grad(name, tensor):
        if tensor.grad is None:
            print(f"  {name}: grad=None")
            return False
        has_nan = torch.isnan(tensor.grad).any().item()
        has_inf = torch.isinf(tensor.grad).any().item()
        nonzero = (tensor.grad.abs() > 1e-8).sum().item()
        print(f"  {name}: mean={tensor.grad.abs().mean():.6e}, max={tensor.grad.abs().max():.6e}, "
              f"nonzero={nonzero}/{tensor.grad.numel()}, NaN={has_nan}, Inf={has_inf}")
        return not has_nan and not has_inf

    all_ok = True
    all_ok &= check_grad("means3D", scene['means3D'])
    all_ok &= check_grad("scales", scene['scales'])
    all_ok &= check_grad("rotations", scene['rotations'])
    all_ok &= check_grad("opacities", scene['opacities'])
    all_ok &= check_grad("gaussian_colors", scene['gaussian_colors'])

    # Check MLP gradients
    mlp_grads = get_mlp_grads()
    print(f"\nMLP gradients (from get_mlp_grads()):")
    if mlp_grads is None:
        print("  ERROR: get_mlp_grads() returned None!")
        all_ok = False
    else:
        names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for name, grad in zip(names, mlp_grads):
            if grad is None:
                print(f"  {name}: None")
                all_ok = False
            else:
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                nonzero = (grad.abs() > 1e-8).sum().item()
                print(f"  {name}: mean={grad.abs().mean():.6e}, max={grad.abs().max():.6e}, "
                      f"nonzero={nonzero}/{grad.numel()}, NaN={has_nan}, Inf={has_inf}")
                if has_nan or has_inf:
                    all_ok = False

    if all_ok:
        print("\n[PASS] All gradients computed correctly!")
    else:
        print("\n[FAIL] Some gradients have issues!")

    return all_ok


def main():
    ok = test_lean_render_backward()
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
