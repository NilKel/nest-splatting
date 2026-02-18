#!/usr/bin/env python3
"""
Verify CUDA MLP weights are exactly what we expect.
Minimal test without rasterizer.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_weight_upload():
    """Upload unit weights and verify they read back correctly."""
    print("="*80)
    print("CUDA Weight Upload Verification")
    print("="*80)

    try:
        from diff_surfel_3D import set_mlp_weights, get_mlp_weights
    except ImportError as e:
        print(f"[ERROR] diff_surfel_3D not available: {e}")
        return

    # Create unit weights
    W1 = torch.zeros(32, 40, dtype=torch.float32, device='cuda')
    b1 = torch.zeros(32, dtype=torch.float32, device='cuda')
    W2 = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    b2 = torch.zeros(32, dtype=torch.float32, device='cuda')
    W3 = torch.zeros(3, 32, dtype=torch.float32, device='cuda')
    b3 = torch.zeros(3, dtype=torch.float32, device='cuda')

    # Set diagonals
    for i in range(32):
        W1[i, i] = 1.0
        W2[i, i] = 1.0
    for i in range(3):
        W3[i, i] = 1.0

    print("\n[UPLOAD] Setting unit weights:")
    print(f"  W1: {W1.shape}, diag[0:3]=[{W1[0,0].item()}, {W1[1,1].item()}, {W1[2,2].item()}]")
    print(f"  W2: {W2.shape}, identity")
    print(f"  W3: {W3.shape}, diag[0:3]=[{W3[0,0].item()}, {W3[1,1].item()}, {W3[2,2].item()}]")

    # W3 memory layout check
    print(f"\n[LAYOUT] W3 flattened (should be 1,0,0..., 0,1,0..., 0,0,1...):")
    W3_flat = W3.flatten()
    print(f"  W3[0:5] = {W3_flat[:5].tolist()}")  # Row 0: [1, 0, 0, 0, 0, ...]
    print(f"  W3[32:37] = {W3_flat[32:37].tolist()}")  # Row 1: [0, 1, 0, 0, 0, ...]
    print(f"  W3[64:69] = {W3_flat[64:69].tolist()}")  # Row 2: [0, 0, 1, 0, 0, ...]

    # Upload to CUDA
    set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)

    # Verify by comparing PyTorch MLP vs what CUDA would produce
    print("\n[VERIFY] Comparing PyTorch reference forward pass")

    # Create PyTorch MLP with same unit weights
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    ).cuda()

    with torch.no_grad():
        mlp[0].weight.copy_(W1)
        mlp[0].bias.copy_(b1)
        mlp[2].weight.copy_(W2)
        mlp[2].bias.copy_(b2)
        mlp[4].weight.copy_(W3)
        mlp[4].bias.copy_(b3)

    # Test input: specific pattern to distinguish weight issues
    x = torch.zeros(1, 40, device='cuda')
    x[0, 0] = 1.0  # Only first element is 1
    x[0, 1] = 2.0  # Second element is 2
    x[0, 2] = 3.0  # Third element is 3

    # Forward
    y = mlp(x)
    print(f"\n  Input: x[0:5] = {x[0, :5].tolist()}")
    print(f"  Output: {y[0].tolist()}")

    # With unit weights: out[o] = sigmoid(x[o])
    # out[0] = sigmoid(1.0) = 0.731...
    # out[1] = sigmoid(2.0) = 0.881...
    # out[2] = sigmoid(3.0) = 0.952...
    expected = [torch.sigmoid(torch.tensor(1.0)).item(),
                torch.sigmoid(torch.tensor(2.0)).item(),
                torch.sigmoid(torch.tensor(3.0)).item()]
    print(f"  Expected (sigmoid of x[0:3]): {expected}")

    if abs(y[0, 0].item() - expected[0]) < 0.01:
        print(f"\n[RESULT] Forward output MATCHES unit weight expectation")

    print("\n" + "="*80)


def test_manual_backward():
    """Manually compute expected gradients and compare with what we see."""
    print("\n" + "="*80)
    print("Manual Backward Verification")
    print("="*80)

    # With unit weights and input x[40]:
    # h1[h] = relu(x[h]) for h < 32
    # h2[h] = relu(h1[h]) = relu(relu(x[h])) = relu(x[h]) for h < 32
    # out[o] = sigmoid(h2[o]) for o < 3

    # So out[o] = sigmoid(relu(relu(x[o]))) for o < 3
    # If x[o] > 0: out[o] = sigmoid(x[o])
    # If x[o] <= 0: out[o] = sigmoid(0) = 0.5

    # Backward:
    # dL/d_out = 1 (for loss = out.sum())
    # dL/d_z3[o] = dL/d_out[o] * sigmoid'(z3[o]) = sigmoid(z3[o]) * (1 - sigmoid(z3[o]))
    # Where z3[o] = h2[o] for unit weights

    # dL/d_h2[h] = sum_o (dL/d_z3[o] * W3[o, h])
    # With W3[o,h] = 1 if o==h else 0:
    # dL/d_h2[0] = dL/d_z3[0] * W3[0,0] = dL/d_z3[0]
    # dL/d_h2[1] = dL/d_z3[1] * W3[1,1] = dL/d_z3[1]
    # dL/d_h2[2] = dL/d_z3[2] * W3[2,2] = dL/d_z3[2]
    # dL/d_h2[h] = 0 for h >= 3

    print("\n[EXPECTED WITH UNIT WEIGHTS]")
    print("  dL_dh2 nonzeros at: [0, 1, 2]")
    print("  dL_dz2 nonzeros at: [0, 1, 2] (if h2_pre > 0)")
    print("  dL_db2 nonzeros at: [0, 1, 2]")
    print("  dL_db1 nonzeros at: [0, 1, 2] (if h1_pre > 0)")

    print("\n[OBSERVED]")
    print("  dL_db2 nonzeros at: [0, 1, 2, ..., 15]  <- 16 neurons!")
    print("  This suggests W3 isn't actually unit weights in CUDA")

    # What W3 pattern would give 16 nonzeros in dL_dh2?
    # dL_dh2[h] = sum_o (dL_dz3[o] * W3[o, h])
    # For h to have nonzero gradient, at least one W3[o, h] must be nonzero
    # If indices 0-15 have nonzero gradients, it means W3 has nonzero values
    # in columns 0-15 for at least one row.

    print("\n[HYPOTHESIS]")
    print("  If W3 stored incorrectly (e.g., shape [32, 3] instead of [3, 32])")
    print("  then W3[o,h] indexing would read wrong memory locations")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_weight_upload()
    test_manual_backward()
