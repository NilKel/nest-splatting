#!/usr/bin/env python3
"""
Debug CUDA MLP gradient computation by comparing against PyTorch step by step.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pytorch_reference():
    """Test PyTorch MLP with unit weights - establishes the reference."""
    print("="*80)
    print("PyTorch Reference Test")
    print("="*80)

    # Create MLP: 40 -> 32 -> 32 -> 3
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    ).cuda()

    # Set unit weights
    with torch.no_grad():
        # W1: diagonal for first 32
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        # W2: identity
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        for i in range(32):
            mlp[2].weight[i, i] = 1.0

        # W3: diagonal for first 3
        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    # Test input: all ones
    x = torch.ones(1, 40, device='cuda', requires_grad=True)

    # Forward
    h1_pre = mlp[0](x)  # Linear only
    h1_post = torch.relu(h1_pre)
    h2_pre = mlp[2](h1_post)
    h2_post = torch.relu(h2_pre)
    z3 = mlp[4](h2_post)  # Linear only (before sigmoid)
    output = torch.sigmoid(z3)

    print(f"\n[FORWARD]")
    print(f"  Input: {x[0, :5].tolist()} ... (all 1.0)")
    print(f"  h1_pre[0:5]: {h1_pre[0, :5].tolist()}")
    print(f"  h1_post[0:5]: {h1_post[0, :5].tolist()}")
    print(f"  h2_pre[0:5]: {h2_pre[0, :5].tolist()}")
    print(f"  h2_post[0:5]: {h2_post[0, :5].tolist()}")
    print(f"  z3: {z3[0].tolist()}")
    print(f"  output: {output[0].tolist()}")

    # Backward with dL_dout = [1, 1, 1]
    output.sum().backward()

    # Get gradients
    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\n[BACKWARD]")
    print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")
    print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")

    # Compute expected values manually
    # dL_dz3 = dL_dout * sigmoid' = 1 * output * (1 - output)
    dL_dz3 = output * (1 - output)  # [0.197, 0.197, 0.197] for sigmoid(1)
    print(f"\n[MANUAL CHECK]")
    print(f"  dL_dz3 (sigmoid derivative): {dL_dz3[0].tolist()}")

    # dL_dh2 = dL_dz3 @ W3^T for identity-like W3
    # With W3[i,i]=1 for i<3: dL_dh2[0:3] = dL_dz3, dL_dh2[3:] = 0
    W3 = mlp[4].weight.data  # [3, 32]
    dL_dh2 = dL_dz3 @ W3  # [1, 32]
    print(f"  dL_dh2[0:5]: {dL_dh2[0, :5].tolist()}")
    print(f"  dL_dh2[3:8]: {dL_dh2[0, 3:8].tolist()} (should be ~0)")

    # dL_dz2 = dL_dh2 * relu'(h2_pre)
    dL_dz2 = dL_dh2 * (h2_pre > 0).float()
    print(f"  dL_dz2[0:5]: {dL_dz2[0, :5].tolist()}")

    print(f"\n[RESULT] Expected: dL_db2 nonzeros at [0, 1, 2]")
    if b2_nz == [0, 1, 2]:
        print(f"  PASS")
    else:
        print(f"  GOT: {b2_nz}")


def test_with_varied_input():
    """Test with varied input values (simulating actual features)."""
    print("\n" + "="*80)
    print("PyTorch Test with Varied Input")
    print("="*80)

    # Create MLP: 40 -> 32 -> 32 -> 3
    mlp = nn.Sequential(
        nn.Linear(40, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Sigmoid()
    ).cuda()

    # Set unit weights
    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        for i in range(32):
            mlp[2].weight[i, i] = 1.0

        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    # Test input: random values (simulating actual features)
    torch.manual_seed(42)
    x = torch.randn(256, 40, device='cuda', requires_grad=True)

    print(f"\n[INPUT] Random input (256 samples)")
    print(f"  x[0, :5]: {x[0, :5].tolist()}")
    print(f"  x[0, 32:37]: {x[0, 32:37].tolist()}")

    # Forward
    y = mlp(x)
    loss = y.sum()
    loss.backward()

    # Get gradients
    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\n[BACKWARD]")
    print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")
    print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")

    # Check if any input dimensions [0:3] are consistently negative
    # which would kill gradient through ReLU
    neg_counts = (x[:, :5] < 0).sum(dim=0)
    print(f"\n[ANALYSIS]")
    print(f"  Negative counts in x[:, 0:5]: {neg_counts.tolist()} / 256")

    # Check h1_pre negativity
    h1_pre = mlp[0](x)  # [256, 32]
    neg_h1 = (h1_pre[:, :5] <= 0).sum(dim=0)
    print(f"  h1_pre <= 0 counts for [0:5]: {neg_h1.tolist()} / 256")

    print(f"\n[RESULT] Expected: dL_db2 nonzeros at [0, 1, 2]")
    if b2_nz == [0, 1, 2]:
        print(f"  PASS")
    else:
        print(f"  Note: With random input, some samples may have h1[0:3] <= 0")
        print(f"  This would kill gradient for those samples, but aggregate should still show [0,1,2]")


def test_cuda_weights_verification():
    """Verify CUDA weights are uploaded correctly."""
    print("\n" + "="*80)
    print("CUDA Weight Verification")
    print("="*80)

    try:
        from diff_surfel_3D import set_mlp_weights, get_mlp_grads

        # Create unit weights (same as standalone test)
        W1 = torch.zeros(32, 40, dtype=torch.float32, device='cuda')
        for i in range(32):
            W1[i, i] = 1.0
        b1 = torch.zeros(32, dtype=torch.float32, device='cuda')

        W2 = torch.eye(32, dtype=torch.float32, device='cuda')
        b2 = torch.zeros(32, dtype=torch.float32, device='cuda')

        W3 = torch.zeros(3, 32, dtype=torch.float32, device='cuda')
        for i in range(3):
            W3[i, i] = 1.0
        b3 = torch.zeros(3, dtype=torch.float32, device='cuda')

        print(f"\n[WEIGHTS TO UPLOAD]")
        print(f"  W1 diagonal: W1[0,0]={W1[0,0]}, W1[1,1]={W1[1,1]}, W1[2,2]={W1[2,2]}")
        print(f"  W3 diagonal: W3[0,0]={W3[0,0]}, W3[1,1]={W3[1,1]}, W3[2,2]={W3[2,2]}")
        print(f"  W3[0,3]={W3[0,3]}, W3[1,3]={W3[1,3]}, W3[2,3]={W3[2,3]} (should be 0)")

        # Print raw memory layout
        print(f"\n[MEMORY LAYOUT]")
        print(f"  W3.flatten()[0:10]: {W3.flatten()[:10].tolist()}")
        print(f"  W3.flatten()[32:42]: {W3.flatten()[32:42].tolist()}")  # Row 1
        print(f"  W3.flatten()[64:74]: {W3.flatten()[64:74].tolist()}")  # Row 2

        # Upload
        print(f"\n[UPLOADING TO CUDA]")
        set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)
        print(f"  Done!")

    except ImportError as e:
        print(f"[ERROR] diff_surfel_3D not available: {e}")


if __name__ == "__main__":
    test_pytorch_reference()
    test_with_varied_input()
    test_cuda_weights_verification()
