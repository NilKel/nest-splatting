#!/usr/bin/env python3
"""
End-to-end gradient test with a single Gaussian through the full rasterization pipeline.
Tests whether transmittance/weight computation is correct in the MLP backward path.
"""

import torch
import torch.nn as nn
import sys
import math

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

def create_single_gaussian_scene():
    """Create a minimal scene with a single Gaussian at the center."""
    device = 'cuda'

    # Single Gaussian at origin, facing camera
    means3D = torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True)  # In front of camera

    # Scales (small surfel)
    scales = torch.tensor([[0.1, 0.1, 0.001]], device=device, requires_grad=True)  # Flat disk

    # Rotation (identity quaternion - facing camera)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)

    # Opacity
    opacities = torch.tensor([[0.9]], device=device, requires_grad=True)

    # Per-Gaussian features (20D for hybrid_levels=5)
    # Use unit-like features: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
    gaussian_features = torch.zeros(1, 20, device=device, requires_grad=True)
    with torch.no_grad():
        for i in range(5):  # 5 levels
            gaussian_features[0, i*4] = 1.0  # First element of each level = 1

    return {
        'means3D': means3D,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'gaussian_features': gaussian_features,
    }

def create_camera(H=64, W=64):
    """Create a simple camera looking at origin."""
    device = 'cuda'

    # Camera at z=-2, looking at origin
    campos = torch.tensor([0.0, 0.0, 0.0], device=device)

    # View matrix (identity - camera at origin looking down +Z)
    viewmatrix = torch.eye(4, device=device)

    # Projection matrix (simple perspective)
    fov = 60 * math.pi / 180
    tanfov = math.tan(fov / 2)

    near = 0.1
    far = 100.0

    projmatrix = torch.zeros(4, 4, device=device)
    projmatrix[0, 0] = 1.0 / tanfov
    projmatrix[1, 1] = 1.0 / tanfov
    projmatrix[2, 2] = far / (far - near)
    projmatrix[2, 3] = -far * near / (far - near)
    projmatrix[3, 2] = 1.0

    return {
        'campos': campos,
        'viewmatrix': viewmatrix,
        'projmatrix': projmatrix,
        'tanfovx': tanfov,
        'tanfovy': tanfov,
        'H': H,
        'W': W,
    }

def test_manual_forward_backward():
    """
    Test MLP gradient flow manually without the full rasterizer.
    Simulates what should happen with a single Gaussian at a single pixel.
    """
    print("=" * 60)
    print("Manual Forward/Backward Test (Single Gaussian, Single Pixel)")
    print("=" * 60)

    device = 'cuda'

    # Create unit-weight MLP
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    ).to(device)

    # Set unit weights
    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        mlp[2].weight.copy_(torch.eye(32))

        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    # Simulate single Gaussian contribution
    # Input: [gauss_feat(20D) | hash_feat(4D) | view_enc(16D)] = 40D
    gauss_feat = torch.ones(1, 20, device=device, requires_grad=True)
    hash_feat = torch.ones(1, 4, device=device, requires_grad=True)
    view_enc = torch.ones(1, 16, device=device, requires_grad=True)

    mlp_input = torch.cat([gauss_feat, hash_feat, view_enc], dim=1)

    # Forward: MLP -> RGB
    rgb = mlp(mlp_input)  # [1, 3]

    # Simulate rasterization: pixel_color = alpha * T * rgb
    # For single Gaussian: alpha = opacity * G, T = 1 (no previous Gaussians)
    opacity = 0.9
    G = 1.0  # Assume Gaussian weight = 1 at pixel center
    alpha = min(0.99, opacity * G)
    T = 1.0  # Transmittance before this Gaussian
    weight = alpha * T

    pixel_color = weight * rgb  # [1, 3]

    # Loss: sum of pixel colors
    loss = pixel_color.sum()

    print(f"RGB output: {rgb[0].tolist()}")
    print(f"Weight (alpha * T): {weight:.4f}")
    print(f"Pixel color: {pixel_color[0].tolist()}")
    print(f"Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Check gradients
    print(f"\n--- Gradients ---")
    print(f"grad_gauss_feat nonzeros: {(gauss_feat.grad.abs() > 1e-10).sum().item()}/20")
    print(f"grad_gauss_feat[0:5]: {gauss_feat.grad[0, :5].tolist()}")

    # Check MLP bias gradients
    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\ndL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")
    print(f"dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")

    # Print actual values
    print(f"\ndL_db2 values [0:5]: {b2_grad[:5].tolist()}")
    print(f"dL_db3 values: {b3_grad.tolist()}")

    # Expected: with weight=0.9, dL_dpixel = [1,1,1] * 0.9 = [0.9, 0.9, 0.9]
    # Then sigmoid derivative, then through unit weight MLP
    expected_dL_dout = weight * torch.ones(3, device=device)
    sig = rgb[0]
    dL_dz3 = expected_dL_dout * sig * (1 - sig)
    print(f"\nExpected dL_dz3 (before backprop): {dL_dz3.tolist()}")

    return b2_nz == [0, 1, 2]

def test_with_varied_weights():
    """Test with different weight values to see gradient scaling."""
    print("\n" + "=" * 60)
    print("Varied Weight Test")
    print("=" * 60)

    device = 'cuda'

    weights_to_test = [1.0, 0.5, 0.1, 0.01]

    for w in weights_to_test:
        # Create unit-weight MLP
        mlp = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        ).to(device)

        with torch.no_grad():
            mlp[0].weight.zero_()
            mlp[0].bias.zero_()
            for i in range(32):
                mlp[0].weight[i, i] = 1.0

            mlp[2].weight.zero_()
            mlp[2].bias.zero_()
            mlp[2].weight.copy_(torch.eye(32))

            mlp[4].weight.zero_()
            mlp[4].bias.zero_()
            for i in range(3):
                mlp[4].weight[i, i] = 1.0

        mlp_input = torch.ones(1, 40, device=device, requires_grad=True)
        rgb = mlp(mlp_input)

        # Simulate weighted contribution
        pixel_color = w * rgb
        loss = pixel_color.sum()
        loss.backward()

        b2_grad = mlp[2].bias.grad
        b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

        print(f"Weight={w:.2f}: dL_db2[0]={b2_grad[0].item():.6f}, nonzeros at {b2_nz}")

def test_multiple_gaussians_accumulation():
    """Test gradient accumulation from multiple Gaussians at one pixel."""
    print("\n" + "=" * 60)
    print("Multiple Gaussians Accumulation Test")
    print("=" * 60)

    device = 'cuda'

    # Simulate 4 Gaussians contributing to same pixel
    # Each has different alpha and transmittance
    alphas = [0.5, 0.3, 0.2, 0.1]

    # Compute transmittance chain: T[i] = prod(1 - alpha[j]) for j < i
    T = [1.0]
    for a in alphas[:-1]:
        T.append(T[-1] * (1 - a))

    weights = [a * t for a, t in zip(alphas, T)]
    print(f"Alphas: {alphas}")
    print(f"Transmittances: {T}")
    print(f"Weights (alpha*T): {weights}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Create unit-weight MLP
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    ).to(device)

    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        mlp[2].weight.copy_(torch.eye(32))

        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    # Each Gaussian has same input (all ones)
    mlp_inputs = [torch.ones(1, 40, device=device, requires_grad=True) for _ in range(4)]

    # Forward: each Gaussian produces RGB, weighted sum gives pixel color
    pixel_color = torch.zeros(1, 3, device=device)
    rgbs = []
    for i, (inp, w) in enumerate(zip(mlp_inputs, weights)):
        rgb = mlp(inp)
        rgbs.append(rgb)
        pixel_color = pixel_color + w * rgb

    loss = pixel_color.sum()
    loss.backward()

    # Check gradients
    b2_grad = mlp[2].bias.grad
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\nPixel color: {pixel_color[0].tolist()}")
    print(f"Loss: {loss.item():.4f}")
    print(f"\ndL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"dL_db2[0:5]: {b2_grad[:5].tolist()}")

    # Expected: sum of weights * sigmoid_derivative
    total_weight = sum(weights)
    sig = rgbs[0][0, 0].item()  # All same with unit weights
    expected_grad_per_output = total_weight * sig * (1 - sig)
    print(f"\nExpected dL_db2[0] (approx): {expected_grad_per_output:.6f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Single Gaussian End-to-End Gradient Test")
    print("=" * 60 + "\n")

    # Test 1: Manual forward/backward with single Gaussian
    pass1 = test_manual_forward_backward()

    # Test 2: Different weight values
    test_with_varied_weights()

    # Test 3: Multiple Gaussians
    test_multiple_gaussians_accumulation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Manual test: {'PASS' if pass1 else 'FAIL'}")
    print("\nNote: All tests use PyTorch reference. The CUDA rasterizer")
    print("should produce equivalent gradients if implemented correctly.")
