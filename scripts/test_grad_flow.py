#!/usr/bin/env python3
"""Test gradient flow for features and hashgrid in 3D_direct_fused mode.

This test runs a single forward-backward pass through the full pipeline
and checks that gradients flow to:
1. Per-Gaussian features
2. Hash grid features
3. MLP weights
"""

import torch
import sys
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

def test_gradient_flow_with_training():
    """Test gradients using the actual train.py pipeline."""
    import subprocess

    # Run 1 iteration of training with 3D_direct_fused mode
    cmd = [
        "conda", "run", "-n", "nest_splatting",
        "python", "train.py",
        "--config", "configs/nerfsyn.yaml",
        "--method", "3D_direct_fused",
        "--hybrid_levels", "5",
        "--test_iterations", "-1",  # No testing
        "--save_iterations", "0",   # No saving
        "--iteration", "1",         # Just 1 iteration
        "--quiet",
    ]

    print("Running 1 training iteration with gradient checks...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


def test_gradient_flow_simple():
    """Simple test that creates minimal pipeline to check gradients."""
    from gridencoder import GridEncoder
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings,
        set_mlp_weights
    )

    device = torch.device('cuda')
    torch.manual_seed(42)

    # Setup parameters
    H, W = 64, 64  # Smaller for faster test
    N = 50  # Number of Gaussians

    # Hash grid encoder (matches INGP config)
    n_levels = 6
    level_dim = 4
    base_resolution = 16
    per_level_scale = 1.5
    log2_hashmap_size = 19

    # Create hash encoder
    encoder = GridEncoder(
        num_levels=n_levels,
        level_dim=level_dim,
        per_level_scale=per_level_scale,
        base_resolution=base_resolution,
        log2_hashmap_size=log2_hashmap_size,
    ).cuda()

    # Per-Gaussian features (hybrid_levels=5 means 5*4=20D from Gaussians)
    hybrid_levels = 5
    per_gaussian_dim = hybrid_levels * level_dim
    gaussian_features = torch.randn(N, per_gaussian_dim, device=device, requires_grad=True)

    # Hash features (remaining 1 level = 4D from hash)
    hash_levels = n_levels - hybrid_levels
    hash_dim = hash_levels * level_dim

    print(f"=== Configuration ===")
    print(f"Gaussian features: {per_gaussian_dim}D (hybrid_levels={hybrid_levels})")
    print(f"Hash features: {hash_dim}D (hash_levels={hash_levels})")
    print(f"Total: {per_gaussian_dim + hash_dim}D")

    # Get level offsets from encoder
    offsets = encoder.offsets.clone()
    # Pad to 17
    if offsets.shape[0] < 17:
        padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=device)
        padded_offsets[:offsets.shape[0]] = offsets
        offsets = padded_offsets

    gridrange = torch.tensor([-1.0, 1.0], device=device)

    # Gaussian parameters
    means3D = torch.randn(N, 3, device=device, requires_grad=True) * 0.3
    scales_init = torch.ones(N, 2, device=device) * 0.05
    scales = scales_init.clone().requires_grad_(True)
    rotations_init = torch.zeros(N, 4, device=device)
    rotations_init[:, 0] = 1.0  # Identity quaternions
    rotations = rotations_init.clone().requires_grad_(True)
    opacities = torch.ones(N, 1, device=device, requires_grad=True) * 0.9

    # Camera setup - proper OpenGL-style projection
    # Camera at z=3 looking at origin
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[2, 3] = 3.0  # Camera position in view space

    # Proper perspective projection
    fov = 1.0  # ~57 degrees
    tanfov = torch.tan(torch.tensor(fov / 2))
    near = 0.1
    far = 100.0

    # OpenGL perspective projection
    projmatrix = torch.zeros(4, 4, device=device)
    projmatrix[0, 0] = 1.0 / (tanfov * (W / H))  # aspect-corrected
    projmatrix[1, 1] = 1.0 / tanfov
    projmatrix[2, 2] = -(far + near) / (far - near)
    projmatrix[2, 3] = -2.0 * far * near / (far - near)
    projmatrix[3, 2] = -1.0

    full_proj = projmatrix @ viewmatrix

    # Camera position in world space
    campos = torch.tensor([0, 0, 3], device=device, dtype=torch.float32)

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfov.item(),
        tanfovy=tanfov.item(),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix.T.contiguous(),
        projmatrix=full_proj.T.contiguous(),
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        beta=0.1,
        if_contract=False,
        record_transmittance=False,
        max_intersections=0,
        detach_hash_grad=False,
        max_intersections_per_pixel=0,
    )

    # Hash grid settings
    hash_settings = HashGridSettings(
        L=n_levels,
        S=per_level_scale,
        H=base_resolution,
        align_corners=False,
        interpolation=1,  # Linear
        shape_dims=torch.tensor([per_gaussian_dim, hash_dim, 3], dtype=torch.int32, device=device),
        aa=0.0,
        aa_threshold=0.01,
    )

    # Create rasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hash_settings)

    # MLP weights (40D input -> 32 -> 32 -> 3D output)
    # Input: gaussian(20D) + hash(4D) + view_enc(16D) = 40D
    mlp_W1 = torch.randn(32, 40, device=device, requires_grad=True) * 0.1
    mlp_b1 = torch.zeros(32, device=device, requires_grad=True)
    mlp_W2 = torch.randn(32, 32, device=device, requires_grad=True) * 0.1
    mlp_b2 = torch.zeros(32, device=device, requires_grad=True)
    mlp_W3 = torch.randn(3, 32, device=device, requires_grad=True) * 0.1
    mlp_b3 = torch.zeros(3, device=device, requires_grad=True)

    # Set MLP weights in CUDA
    set_mlp_weights(mlp_W1, mlp_b1, mlp_W2, mlp_b2, mlp_W3, mlp_b3)

    # Encode level for render_mode=5: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
    total_levels = n_levels
    active_hashgrid_levels = hash_levels
    encoded_level = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

    print(f"\n=== Level Encoding ===")
    print(f"Encoded level: {encoded_level}")
    print(f"  total_levels: {total_levels}")
    print(f"  active_hashgrid_levels: {active_hashgrid_levels}")
    print(f"  hybrid_levels: {hybrid_levels}")

    # Screenspace points
    screenspace_points = torch.zeros(N, 3, device=device, requires_grad=True)

    # Hash features from encoder
    hash_features = encoder.embeddings  # This is the parameter we want gradients for

    print(f"\n=== Hash Features ===")
    print(f"hash_features shape: {hash_features.shape}")
    print(f"hash_features requires_grad: {hash_features.requires_grad}")
    print(f"offsets: {offsets[:n_levels+1].tolist()}")

    try:
        # Forward pass with render_mode=5 (3D_direct_fused)
        render_mode = 5

        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=gaussian_features,  # Per-Gaussian features (20D)
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            homotrans=None,
            ap_level=None,
            features=hash_features,  # Hash grid features (2D: total_entries x level_dim)
            offsets=offsets,
            gridrange=gridrange,
            features_diffuse=None,
            offsets_diffuse=None,
            gridrange_diffuse=None,
            render_mode=render_mode,
            shapes=None,
            kernel_type=0,
            aabb_mode=False,
        )

        print(f"\n=== Forward Pass ===")
        print(f"Rendered image shape: {rendered_image.shape}")
        print(f"Rendered image min/max: {rendered_image.min():.4f} / {rendered_image.max():.4f}")
        print(f"Radii nonzero: {(radii > 0).sum().item()}/{radii.shape[0]}")

        # Compute loss
        loss = rendered_image.sum()
        print(f"Loss: {loss.item():.4f}")

        # Backward
        loss.backward()

        # Check gradients
        print("\n=== Gradient Check ===")

        def check_grad(name, tensor):
            if tensor.grad is not None:
                grad_norm = tensor.grad.norm().item()
                grad_nonzero = (tensor.grad.abs() > 1e-10).sum().item()
                status = "[PASS]" if grad_norm > 1e-10 else "[FAIL]"
                print(f"{status} {name}.grad: norm={grad_norm:.8f}, nonzero={grad_nonzero}/{tensor.numel()}")
                return grad_norm > 1e-10
            else:
                print(f"[FAIL] {name}.grad is None!")
                return False

        # Per-Gaussian features
        check_grad("gaussian_features", gaussian_features)

        # Hash grid features (from GridEncoder)
        check_grad("hash_features (encoder.embeddings)", encoder.embeddings)

        # MLP weights - check stored grads from CUDA
        from diff_surfel_rasterization import _RasterizeGaussians
        if hasattr(_RasterizeGaussians, '_last_mlp_grads'):
            mlp_grads = _RasterizeGaussians._last_mlp_grads
            print(f"\nMLP grads from CUDA kernel:")
            for name, grad in zip(["W1", "b1", "W2", "b2", "W3", "b3"], mlp_grads):
                if grad is not None:
                    norm = grad.norm().item()
                    status = "[PASS]" if norm > 1e-10 else "[FAIL]"
                    print(f"  {status} dL_d{name}: norm={norm:.8f}, shape={grad.shape}")
                else:
                    print(f"  [FAIL] dL_d{name}: None")
        else:
            print("\n[WARN] No MLP grads stored (_last_mlp_grads not found)")

        # Geometry gradients
        print("\n=== Geometry Gradients ===")
        check_grad("means3D", means3D)
        check_grad("scales", scales)
        check_grad("rotations", rotations)
        check_grad("opacities", opacities)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradient_flow_simple()
