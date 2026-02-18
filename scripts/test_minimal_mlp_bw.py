#!/usr/bin/env python3
"""
Minimal test: Single-pixel MLP backward with unit weights.
This isolates the gradient computation from the full rendering pipeline.
"""

import torch
import torch.nn as nn


def test_pytorch_mlp_backward():
    """PyTorch reference for MLP backward with unit weights."""
    print("=" * 60)
    print("PyTorch MLP Backward Reference (Unit Weights)")
    print("=" * 60)

    # Unit MLP: 40 -> 32 -> 32 -> 3
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    )

    # Set unit weights (identity-like for first N outputs)
    with torch.no_grad():
        # W1[i, i] = 1 for i < 32 (maps input[i] to h1[i])
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        # W2 = identity 32x32
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        mlp[2].weight.copy_(torch.eye(32))

        # W3[i, i] = 1 for i < 3 (maps h2[i] to out[i])
        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    print("\nWeight shapes:")
    print(f"  W1: {mlp[0].weight.shape} (out_features, in_features)")
    print(f"  W2: {mlp[2].weight.shape}")
    print(f"  W3: {mlp[4].weight.shape}")

    # Single sample input (simulating one pixel's MLP input)
    x = torch.ones(1, 40, requires_grad=True)  # 1 sample, 40D input

    # Forward
    y = mlp(x)
    print(f"\nInput: {x[0, :5].tolist()}...")
    print(f"Output: {y[0].tolist()}")

    # Backward with grad_out = 1 for each output
    loss = y.sum()
    loss.backward()

    # Check gradient patterns
    print("\n--- Gradient Analysis ---")

    # W3 gradient: dL_dW3 = dL_dout^T @ h2
    # With unit W3, dL_dout flows directly to dL_dh2[0:3]
    W3_grad = mlp[4].weight.grad
    b3_grad = mlp[4].bias.grad
    print(f"\nW3.grad shape: {W3_grad.shape}")
    print(f"W3.grad nonzero: {(W3_grad.abs() > 1e-10).sum().item()}/{W3_grad.numel()}")
    print(f"b3.grad: {b3_grad.tolist()}")

    # dL_dh2 = W3^T @ dL_dout
    # With unit W3, only dL_dh2[0], dL_dh2[1], dL_dh2[2] are nonzero
    # Then dL_dz2 = dL_dh2 * relu'(h2_pre) = dL_dh2 (since h2_pre > 0)

    # W2 gradient: dL_dW2 = dL_dz2^T @ h1
    # Only dL_dz2[0:3] are nonzero, so:
    # dL_dW2[i, j] = dL_dz2[i] * h1[j] => only rows 0,1,2 are nonzero
    W2_grad = mlp[2].weight.grad
    b2_grad = mlp[2].bias.grad
    print(f"\nW2.grad shape: {W2_grad.shape}")
    print(f"W2.grad nonzero rows: {(W2_grad.abs().sum(dim=1) > 1e-10).sum().item()}")
    b2_nz_idx = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    print(f"b2.grad nonzero: {len(b2_nz_idx)}/32 at indices {b2_nz_idx}")

    # dL_dh1 = W2^T @ dL_dz2
    # With unit W2, dL_dh1[i] = dL_dz2[i], so only dL_dh1[0:3] are nonzero

    # W1 gradient: dL_dW1 = dL_dz1^T @ input
    # Only dL_dz1[0:3] are nonzero (after relu gate, which passes since h1_pre > 0)
    W1_grad = mlp[0].weight.grad
    b1_grad = mlp[0].bias.grad
    print(f"\nW1.grad shape: {W1_grad.shape}")
    print(f"W1.grad nonzero rows: {(W1_grad.abs().sum(dim=1) > 1e-10).sum().item()}")
    b1_nz_idx = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    print(f"b1.grad nonzero: {len(b1_nz_idx)}/32 at indices {b1_nz_idx}")

    print("\n--- Expected Pattern ---")
    print("With unit weights, gradient only flows through diagonal:")
    print("  dL_dout[0,1,2] -> dL_dh2[0,1,2] -> dL_dz2[0,1,2] -> dL_dh1[0,1,2] -> dL_dz1[0,1,2]")
    print("  b3: 3/3 nonzeros at [0,1,2]")
    print("  b2: 3/32 nonzeros at [0,1,2]")
    print("  b1: 3/32 nonzeros at [0,1,2]")

    return b1_grad, b2_grad, b3_grad


if __name__ == "__main__":
    b1, b2, b3 = test_pytorch_mlp_backward()

    print("\n" + "=" * 60)
    print("CUDA MLP Backward Bug Hypothesis")
    print("=" * 60)
    print("\nIf CUDA shows 16/32 nonzeros instead of 3/32, possible causes:")
    print("1. Weight indexing bug (wrong row/column order)")
    print("2. ReLU gate not being applied correctly")
    print("3. Non-participating threads contributing garbage")
    print("4. Shared memory layout mismatch")
    print("\nDEBUG: Check if W3^T @ dL_dz3 produces [dL_dz3[0], dL_dz3[1], dL_dz3[2], 0, 0, ...]")
