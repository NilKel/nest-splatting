#!/usr/bin/env python3
"""
Minimal test for MLP backward pass with unit weights.
Tests only the gradient flow, not the full render pipeline.
"""

import torch
import torch.nn as nn

def test_unit_mlp_backward():
    """Test PyTorch MLP backward with unit weights to establish baseline."""
    print("=" * 60)
    print("PYTORCH MLP BACKWARD WITH UNIT WEIGHTS")
    print("=" * 60)

    # Build unit MLP: 40 -> 32 -> 32 -> 3
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    )

    # Set unit weights
    with torch.no_grad():
        # Layer 1: identity for first 32 inputs
        mlp[0].weight.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0
        mlp[0].bias.zero_()

        # Layer 2: identity
        mlp[2].weight.zero_()
        for i in range(32):
            mlp[2].weight[i, i] = 1.0
        mlp[2].bias.zero_()

        # Layer 3: identity for first 3
        mlp[4].weight.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0
        mlp[4].bias.zero_()

    # Create input batch
    batch_size = 256
    x = torch.randn(batch_size, 40, requires_grad=True)

    # Forward
    y = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Backward with unit gradient
    loss = y.mean()
    loss.backward()

    print("\nMLP gradient analysis:")
    for name, param in mlp.named_parameters():
        if param.grad is not None:
            nonzero = (param.grad.abs() > 1e-10).sum().item()
            print(f"  {name}: norm={param.grad.norm():.6f}, nonzero={nonzero}/{param.grad.numel()}")

    # Expected with unit weights:
    # - Layer 4 (output) bias: 3 nonzeros (all 3 output neurons)
    # - Layer 4 (output) weight: 3*32=96 theoretically, but only 3 actually (diagonal)
    # - Layer 2 bias: 3 nonzeros (only neurons 0,1,2 get gradient through unit W3)
    # - Layer 2 weight: only 3 diagonal elements get gradient
    # - Layer 0 bias: 3 nonzeros (only neurons 0,1,2 get gradient)
    # - Layer 0 weight: only 3*40=120 get gradient (3 output neurons × 40 inputs each)

    print("\n" + "=" * 60)
    print("EXPECTED NONZEROS WITH UNIT WEIGHTS:")
    print("=" * 60)
    print("  mlp.4.bias (b3): 3/3 (all 3 outputs)")
    print("  mlp.4.weight (W3): 3/96 (only diagonal: [0,0], [1,1], [2,2])")
    print("  mlp.2.bias (b2): 3/32 (only neurons 0,1,2)")
    print("  mlp.2.weight (W2): 3/1024 (only diagonal for neurons 0,1,2)")
    print("  mlp.0.bias (b1): 3/32 (only neurons 0,1,2)")
    print("  mlp.0.weight (W1): depends on which inputs are positive")

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    b3_nonzero = (mlp[4].bias.grad.abs() > 1e-10).sum().item()
    b2_nonzero = (mlp[2].bias.grad.abs() > 1e-10).sum().item()
    b1_nonzero = (mlp[0].bias.grad.abs() > 1e-10).sum().item()

    print(f"  b3 nonzeros: {b3_nonzero} (expected 3) {'PASS' if b3_nonzero == 3 else 'FAIL'}")
    print(f"  b2 nonzeros: {b2_nonzero} (expected 3) {'PASS' if b2_nonzero == 3 else 'FAIL'}")
    print(f"  b1 nonzeros: {b1_nonzero} (expected 3) {'PASS' if b1_nonzero == 3 else 'FAIL'}")

    # Check which neurons have gradient
    print("\n  Which neurons have bias gradients?")
    b3_active = (mlp[4].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    b2_active = (mlp[2].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    b1_active = (mlp[0].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    print(f"    b3 (layer 4): {b3_active}")
    print(f"    b2 (layer 2): {b2_active}")
    print(f"    b1 (layer 0): {b1_active}")


if __name__ == "__main__":
    test_unit_mlp_backward()
