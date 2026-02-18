#!/usr/bin/env python3
"""
Debug MLP gradient computation with detailed output.
Compare PyTorch vs what CUDA should produce given the same input.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def debug_mlp_backward():
    """Debug MLP backward with unit weights step by step."""
    print("="*80)
    print("DEBUG: MLP Backward Step by Step")
    print("="*80)

    # Create unit weights
    W1 = torch.zeros(32, 40, device='cuda')
    b1 = torch.zeros(32, device='cuda')
    W2 = torch.eye(32, device='cuda')
    b2 = torch.zeros(32, device='cuda')
    W3 = torch.zeros(3, 32, device='cuda')
    b3 = torch.zeros(3, device='cuda')

    for i in range(32):
        W1[i, i] = 1.0
    for i in range(3):
        W3[i, i] = 1.0

    print("\n[WEIGHTS]")
    print(f"  W1: diagonal identity (32x40)")
    print(f"  W2: identity (32x32)")
    print(f"  W3: {W3[:, :5].tolist()}")
    print(f"  W3 memory layout [0:5]: {W3.flatten()[:5].tolist()}")
    print(f"  W3 memory layout [32:37]: {W3.flatten()[32:37].tolist()}")
    print(f"  W3 memory layout [64:69]: {W3.flatten()[64:69].tolist()}")

    # Create input: positive values
    x = torch.ones(1, 40, device='cuda') * 0.5  # All 0.5

    print(f"\n[INPUT]")
    print(f"  x: all 0.5 (shape {x.shape})")

    # Manual forward
    h1_pre = x @ W1.T + b1
    h1_post = torch.relu(h1_pre)
    h2_pre = h1_post @ W2.T + b2
    h2_post = torch.relu(h2_pre)
    z3 = h2_post @ W3.T + b3
    output = torch.sigmoid(z3)

    print(f"\n[FORWARD]")
    print(f"  h1_pre[0:5]: {h1_pre[0, :5].tolist()}")
    print(f"  h1_post[0:5]: {h1_post[0, :5].tolist()}")
    print(f"  h2_pre[0:5]: {h2_pre[0, :5].tolist()}")
    print(f"  h2_post[0:5]: {h2_post[0, :5].tolist()}")
    print(f"  z3: {z3[0].tolist()}")
    print(f"  output: {output[0].tolist()}")

    # Count positive values
    h1_pre_pos = (h1_pre > 0).sum().item()
    h2_pre_pos = (h2_pre > 0).sum().item()
    print(f"\n  h1_pre positive: {h1_pre_pos}/32")
    print(f"  h2_pre positive: {h2_pre_pos}/32")

    # Manual backward (loss = output.sum())
    dL_dout = torch.ones_like(output)

    # Through sigmoid
    dL_dz3 = dL_dout * output * (1 - output)
    print(f"\n[BACKWARD]")
    print(f"  dL_dz3: {dL_dz3[0].tolist()}")

    # Layer 3 backward: dL_dh2 = dL_dz3 @ W3
    dL_dh2 = dL_dz3 @ W3
    print(f"  dL_dh2[0:5]: {dL_dh2[0, :5].tolist()}")
    print(f"  dL_dh2[3:8]: {dL_dh2[0, 3:8].tolist()} (should be 0)")

    # ReLU backward: dL_dz2 = dL_dh2 * (h2_pre > 0)
    dL_dz2 = dL_dh2 * (h2_pre > 0).float()
    print(f"  dL_dz2[0:5]: {dL_dz2[0, :5].tolist()}")

    # Layer 2 backward: dL_dh1 = dL_dz2 @ W2
    dL_dh1 = dL_dz2 @ W2
    print(f"  dL_dh1[0:5]: {dL_dh1[0, :5].tolist()}")

    # ReLU backward: dL_dz1 = dL_dh1 * (h1_pre > 0)
    dL_dz1 = dL_dh1 * (h1_pre > 0).float()
    print(f"  dL_dz1[0:5]: {dL_dz1[0, :5].tolist()}")

    # Bias gradients are just dL_dz summed over batch
    dL_db1 = dL_dz1.sum(dim=0)
    dL_db2 = dL_dz2.sum(dim=0)
    dL_db3 = dL_dz3.sum(dim=0)

    b1_nz = (dL_db1.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (dL_db2.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (dL_db3.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\n[BIAS GRADIENTS]")
    print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")
    print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")
    print(f"\n  Expected: all at [0, 1, 2]")

    # Verify with PyTorch autograd
    print("\n" + "="*80)
    print("VERIFY: PyTorch autograd")
    print("="*80)

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

    x2 = torch.ones(1, 40, device='cuda', requires_grad=True) * 0.5
    y2 = mlp(x2)
    loss = y2.sum()
    loss.backward()

    pt_b1_nz = (mlp[0].bias.grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    pt_b2_nz = (mlp[2].bias.grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    pt_b3_nz = (mlp[4].bias.grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"  PyTorch dL_db1 nonzeros: {len(pt_b1_nz)}/32 at {pt_b1_nz}")
    print(f"  PyTorch dL_db2 nonzeros: {len(pt_b2_nz)}/32 at {pt_b2_nz}")
    print(f"  PyTorch dL_db3 nonzeros: {len(pt_b3_nz)}/3 at {pt_b3_nz}")

    print("\n" + "="*80)


def test_w3_indexing():
    """Verify W3 indexing matches expectation."""
    print("\n" + "="*80)
    print("DEBUG: W3 Indexing Verification")
    print("="*80)

    W3 = torch.zeros(3, 32, device='cuda')
    for i in range(3):
        W3[i, i] = 1.0

    # CUDA indexes as W3[o * 32 + h]
    # For unit weights: W3[0,0]=1, W3[1,1]=1, W3[2,2]=1

    print("\n[W3 Memory Layout]")
    flat = W3.flatten()
    print(f"  Index 0 (o=0, h=0): {flat[0].item()} (expected 1)")
    print(f"  Index 1 (o=0, h=1): {flat[1].item()} (expected 0)")
    print(f"  Index 33 (o=1, h=1): {flat[33].item()} (expected 1)")
    print(f"  Index 66 (o=2, h=2): {flat[66].item()} (expected 1)")

    # Verify indexing formula
    for o in range(3):
        for h in range(3):
            idx = o * 32 + h
            val = flat[idx].item()
            expected = 1.0 if o == h else 0.0
            match = "✓" if abs(val - expected) < 0.01 else "✗"
            print(f"  W3[{o}][{h}] = flat[{idx}] = {val} (expected {expected}) {match}")

    print("\n" + "="*80)


if __name__ == "__main__":
    debug_mlp_backward()
    test_w3_indexing()
