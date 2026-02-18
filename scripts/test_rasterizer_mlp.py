#!/usr/bin/env python3
"""
Test the actual rasterizer's MLP backward with unit weights.
This uses the full rasterizer code path to identify where the bug is.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

def test_with_unit_weights():
    """Test rasterizer MLP with unit weights to trace gradient flow."""
    from diff_surfel_3D import set_mlp_weights, get_mlp_grads

    # Create unit weights (same structure as standalone test)
    # W1: [32, 40] - diagonal for first 32 columns
    W1 = torch.zeros(32, 40, dtype=torch.float32, device='cuda')
    for i in range(32):
        W1[i, i] = 1.0
    b1 = torch.zeros(32, dtype=torch.float32, device='cuda')

    # W2: [32, 32] - identity
    W2 = torch.eye(32, dtype=torch.float32, device='cuda')
    b2 = torch.zeros(32, dtype=torch.float32, device='cuda')

    # W3: [3, 32] - diagonal for first 3 rows
    W3 = torch.zeros(3, 32, dtype=torch.float32, device='cuda')
    for i in range(3):
        W3[i, i] = 1.0
    b3 = torch.zeros(3, dtype=torch.float32, device='cuda')

    print("Unit weights setup:")
    print(f"  W1[0,0]={W1[0,0].item()}, W1[1,1]={W1[1,1].item()}, W1[2,2]={W1[2,2].item()}")
    print(f"  W2[0,0]={W2[0,0].item()}, W2[1,1]={W2[1,1].item()}")
    print(f"  W3[0,0]={W3[0,0].item()}, W3[1,1]={W3[1,1].item()}, W3[2,2]={W3[2,2].item()}")

    # Upload weights to CUDA
    print("\nUploading weights to CUDA...")
    set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)

    # Verify upload
    print("\nWeights uploaded. Now we need to run a forward+backward pass...")
    print("(This requires setting up the full rasterizer which is complex)")
    print("")
    print("For now, let's verify the weights were uploaded correctly by checking the debug output.")
    print("The [setMlpWeights] VERIFY line should show:")
    print("  For unit weights: mlp_W1[0..3]=[1.0, 0.0, 0.0, 0.0]")
    print("")
    print("If the verify output shows different values, the weights aren't being uploaded correctly.")

def test_mlp_directly():
    """Test the MLP computation directly (bypassing rasterizer)."""
    print("\n=== Direct MLP Test ===")

    # Create unit weight MLP
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
        mlp[2].weight.copy_(torch.eye(32))

        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0

    # Run forward + backward
    x = torch.ones(256, 40, device='cuda', requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()

    # Check bias gradients
    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"PyTorch reference:")
    print(f"  Output[0] = {y[0].tolist()}")
    print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")
    print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")
    print(f"  Expected: all at [0, 1, 2]")

    # Print weight shapes for comparison with CUDA
    print(f"\nWeight shapes (for CUDA comparison):")
    print(f"  W1: {mlp[0].weight.shape} (should match CUDA access pattern W1[h*40+i])")
    print(f"  W2: {mlp[2].weight.shape}")
    print(f"  W3: {mlp[4].weight.shape}")

    # Print actual b2_grad values
    print(f"\nb2_grad values (first 8): {b2_grad[:8].tolist()}")

    return b2_nz == [0, 1, 2]

if __name__ == "__main__":
    print("=== Testing Rasterizer MLP ===\n")

    # First, test with PyTorch to confirm expected behavior
    pytorch_ok = test_mlp_directly()

    print("\n" + "="*60)
    if pytorch_ok:
        print("PyTorch reference: PASS")
    else:
        print("PyTorch reference: FAIL (unexpected)")

    print("\n" + "="*60)
    print("\nNow testing CUDA weight upload...")
    test_with_unit_weights()
