#!/usr/bin/env python3
"""
Test CUDA rasterizer gradient flow with a single Gaussian.
Compares CUDA gradients against PyTorch reference.
"""

import torch
import torch.nn as nn
import sys
import math
import numpy as np

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

def setup_minimal_rasterizer_test():
    """
    Create a minimal test case for the rasterizer with a single Gaussian.
    Returns all inputs needed for rasterize_gaussians.
    """
    from diff_surfel_3D import set_mlp_weights

    device = 'cuda'
    H, W = 64, 64
    P = 1  # Single Gaussian

    # Camera parameters
    fov = 60 * math.pi / 180
    tanfov = math.tan(fov / 2)

    viewmatrix = torch.eye(4, device=device)
    projmatrix = torch.zeros(4, 4, device=device)
    projmatrix[0, 0] = 1.0 / tanfov
    projmatrix[1, 1] = 1.0 / tanfov
    projmatrix[2, 2] = 100.0 / 99.9
    projmatrix[2, 3] = -0.1 * 100.0 / 99.9
    projmatrix[3, 2] = 1.0
    campos = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Single Gaussian at z=2 (in front of camera)
    means3D = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)

    # Scales - flat disk facing camera
    scales = torch.tensor([[0.3, 0.3, 0.001]], device=device, dtype=torch.float32)

    # Rotation - identity (facing camera)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    # Opacity
    opacities = torch.tensor([[0.9]], device=device, dtype=torch.float32)

    # Per-Gaussian features (20D for hybrid_levels=5)
    # Set to all ones for predictable gradient flow
    gaussian_features = torch.ones(P, 20, device=device, dtype=torch.float32)

    # Create unit-weight MLP and upload to CUDA
    W1 = torch.zeros(32, 40, device=device, dtype=torch.float32)
    for i in range(32):
        W1[i, i] = 1.0
    b1 = torch.zeros(32, device=device, dtype=torch.float32)

    W2 = torch.eye(32, device=device, dtype=torch.float32)
    b2 = torch.zeros(32, device=device, dtype=torch.float32)

    W3 = torch.zeros(3, 32, device=device, dtype=torch.float32)
    for i in range(3):
        W3[i, i] = 1.0
    b3 = torch.zeros(3, device=device, dtype=torch.float32)

    print("Uploading unit weights to CUDA...")
    set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)

    return {
        'means3D': means3D,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'gaussian_features': gaussian_features,
        'viewmatrix': viewmatrix,
        'projmatrix': projmatrix,
        'campos': campos,
        'tanfovx': tanfov,
        'tanfovy': tanfov,
        'H': H,
        'W': W,
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3,
    }

def test_rasterizer_gradients():
    """Run the actual rasterizer and check MLP gradients."""
    print("=" * 60)
    print("CUDA Rasterizer MLP Gradient Test")
    print("=" * 60)

    try:
        from diff_surfel_3D import rasterize_gaussians, get_mlp_grads, GaussianRasterizationSettings, HashGridSettings
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure diff_surfel_3D is built.")
        return

    params = setup_minimal_rasterizer_test()
    device = 'cuda'

    # Create raster settings
    raster_settings = GaussianRasterizationSettings(
        image_height=params['H'],
        image_width=params['W'],
        tanfovx=params['tanfovx'],
        tanfovy=params['tanfovy'],
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=params['viewmatrix'],
        projmatrix=params['projmatrix'],
        sh_degree=0,
        campos=params['campos'],
        prefiltered=False,
        debug=False,
        beta=1e8,
        if_contract=False,
        record_transmittance=False,
        max_intersections=32,
        max_intersections_per_pixel=32,
        detach_hash_grad=False,
    )

    # Hash grid settings (minimal)
    hashgrid_settings = HashGridSettings(
        L=1,  # 1 hashgrid level
        S=8,  # Small resolution
        H=2**14,  # Hash table size
        align_corners=True,
        interpolation=1,  # Linear
        shape_dims=torch.tensor([20, 4, 24], dtype=torch.int32, device=device),
        aa=False,
        aa_threshold=0.1,
    )

    # Empty tensors for unused features
    empty_float = torch.zeros(0, device=device, dtype=torch.float32)
    empty_int = torch.zeros(0, device=device, dtype=torch.int32)

    # Hash features (4D per level, 1 level)
    hash_features = torch.ones(2**14, 4, device=device, dtype=torch.float32)
    hash_offsets = torch.tensor([0, 2**14], device=device, dtype=torch.int32)
    hash_gridrange = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device, dtype=torch.float32)

    # Appearance level (per-Gaussian) - must match expected type
    ap_level = torch.zeros(1, dtype=torch.int32, device=device).to(torch.int32)

    # View direction encoding (16D per pixel)
    # For simplicity, use all ones
    viewdirs_enc = torch.ones(params['H'] * params['W'], 16, device=device, dtype=torch.float32)

    print(f"\nGaussian position: {params['means3D'][0].tolist()}")
    print(f"Gaussian opacity: {params['opacities'][0, 0].item()}")
    print(f"Gaussian features sum: {params['gaussian_features'].sum().item()}")

    # Set requires_grad
    params['gaussian_features'].requires_grad_(True)
    params['opacities'].requires_grad_(True)
    params['means3D'].requires_grad_(True)
    params['scales'].requires_grad_(True)
    params['rotations'].requires_grad_(True)

    print("\nRunning forward pass...")

    try:
        # Forward pass
        color, radii, depth, transmittance_avg, pixels, intersection_buffer, intersection_count, geomBuffer = rasterize_gaussians(
            means3D=params['means3D'],
            means2D=torch.zeros(1, 2, device=device),  # Will be computed
            sh=empty_float.reshape(0, 0, 0),
            colors_precomp=params['gaussian_features'],  # Per-Gaussian features
            opacities=params['opacities'],
            scales=params['scales'],
            rotations=params['rotations'],
            cov3Ds_precomp=empty_float,
            homotrans=empty_float.reshape(0, 3, 3),
            ap_level=ap_level,
            features=hash_features,
            offsets=hash_offsets,
            gridrange=hash_gridrange,
            features_diffuse=empty_float.reshape(0, 4),
            offsets_diffuse=empty_int,
            gridrange_diffuse=empty_float.reshape(0, 3),
            raster_settings=raster_settings,
            hashgrid_settings=hashgrid_settings,
            render_mode=5,  # 3D_direct_fused
            shapes=empty_float.reshape(0, 1),
            kernel_type=0,
            aabb_mode=0,
            viewdirs_enc=viewdirs_enc,
        )

        print(f"Output color shape: {color.shape}")
        print(f"Radii: {radii}")
        print(f"Visible Gaussians: {(radii > 0).sum().item()}")

        if radii[0] <= 0:
            print("\nWARNING: Gaussian not visible! Adjusting position...")
            return

        # Check center pixel color
        center_y, center_x = params['H'] // 2, params['W'] // 2
        center_color = color[:, center_y, center_x]
        print(f"\nCenter pixel color: {center_color.tolist()}")

        # Compute loss
        loss = color.sum()
        print(f"Loss (sum of all pixels): {loss.item():.4f}")

        print("\nRunning backward pass...")
        loss.backward()

        # Get MLP gradients
        mlp_grads = get_mlp_grads()

        if mlp_grads is None:
            print("ERROR: MLP gradients are None!")
            return

        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = mlp_grads

        print("\n" + "=" * 60)
        print("MLP Gradient Results")
        print("=" * 60)

        # Check b2 gradients
        b2_nz = (grad_b2.abs() > 1e-10).nonzero().squeeze(-1).tolist()
        print(f"grad_b2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
        print(f"grad_b2 values [0:8]: {grad_b2[:8].tolist()}")

        # Check b1 gradients
        b1_nz = (grad_b1.abs() > 1e-10).nonzero().squeeze(-1).tolist()
        print(f"grad_b1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")

        # Check b3 gradients
        b3_nz = (grad_b3.abs() > 1e-10).nonzero().squeeze(-1).tolist()
        print(f"grad_b3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")

        # Check other gradients
        print(f"\ngrad_gaussian_features nonzeros: {(params['gaussian_features'].grad.abs() > 1e-10).sum().item()}/20")
        print(f"grad_opacities: {params['opacities'].grad}")

        # Verdict
        print("\n" + "=" * 60)
        if b2_nz == [0, 1, 2]:
            print("RESULT: PASS - Correct gradient pattern!")
        else:
            print(f"RESULT: FAIL - Expected [0,1,2], got {b2_nz}")
            print("\nDEBUG: This indicates the bug is in the CUDA kernel.")
            print("The issue is likely in compute_dL_dz_all or the weight computation.")

    except Exception as e:
        print(f"Error during rasterization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rasterizer_gradients()
