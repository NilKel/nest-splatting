#!/usr/bin/env python3
"""
Compare custom CUDA MLP implementation vs PyTorch MLP.
Tests with both unit weights and random weights/inputs.
"""

import torch
import torch.nn as nn
import numpy as np


def create_pytorch_mlp(in_dim=40, hidden_dim=32, out_dim=3):
    """Create a PyTorch MLP with identical architecture."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.Sigmoid()
    )


def set_unit_weights(mlp):
    """Set unit weights in PyTorch MLP."""
    with torch.no_grad():
        # W1: [32, 40] - diagonal for first 32
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(32):
            mlp[0].weight[i, i] = 1.0

        # W2: [32, 32] - identity
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        mlp[2].weight.copy_(torch.eye(32))

        # W3: [3, 32] - diagonal for first 3
        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(3):
            mlp[4].weight[i, i] = 1.0


def test_unit_weights():
    """Test with unit weights - gradient should only flow through [0,1,2]."""
    print("=" * 60)
    print("Test 1: Unit Weights (input=1)")
    print("=" * 60)

    mlp = create_pytorch_mlp().cuda()
    set_unit_weights(mlp)

    x = torch.ones(256, 40, device='cuda', requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()

    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"Output[0]: {y[0].tolist()}")
    print(f"dL_db3: {len(b3_nz)}/3 at {b3_nz}")
    print(f"dL_db2: {len(b2_nz)}/32 at {b2_nz}")
    print(f"dL_db1: {len(b1_nz)}/32 at {b1_nz}")
    print(f"Expected: all at [0,1,2]")

    return b2_nz == [0, 1, 2]


def test_random_weights():
    """Test with random weights - all bias gradients should be nonzero."""
    print("\n" + "=" * 60)
    print("Test 2: Random Weights (random input)")
    print("=" * 60)

    torch.manual_seed(42)
    mlp = create_pytorch_mlp().cuda()

    # Random input
    x = torch.randn(256, 40, device='cuda', requires_grad=True)

    y = mlp(x)
    loss = y.sum()
    loss.backward()

    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).sum().item()
    b2_nz = (b2_grad.abs() > 1e-10).sum().item()
    b3_nz = (b3_grad.abs() > 1e-10).sum().item()

    print(f"Output[0]: {y[0].tolist()}")
    print(f"dL_db3: {b3_nz}/3 nonzeros")
    print(f"dL_db2: {b2_nz}/32 nonzeros")
    print(f"dL_db1: {b1_nz}/32 nonzeros")
    print(f"Expected: most/all nonzero with random weights")

    # Print actual values for comparison with CUDA
    print(f"\ndL_db2 values (for comparison):")
    print(f"  {b2_grad[:8].tolist()}")
    print(f"  {b2_grad[8:16].tolist()}")

    return {
        'output': y[0].detach().cpu().numpy(),
        'b1_grad': b1_grad.cpu().numpy(),
        'b2_grad': b2_grad.cpu().numpy(),
        'b3_grad': b3_grad.cpu().numpy(),
        'W1': mlp[0].weight.detach().cpu().numpy(),
        'b1': mlp[0].bias.detach().cpu().numpy(),
        'W2': mlp[2].weight.detach().cpu().numpy(),
        'b2': mlp[2].bias.detach().cpu().numpy(),
        'W3': mlp[4].weight.detach().cpu().numpy(),
        'b3': mlp[4].bias.detach().cpu().numpy(),
        'input': x.detach().cpu().numpy(),
    }


def save_test_data(data, filename='mlp_test_data.npz'):
    """Save test data for comparison with CUDA."""
    np.savez(filename,
             input=data['input'],
             W1=data['W1'], b1=data['b1'],
             W2=data['W2'], b2=data['b2'],
             W3=data['W3'], b3=data['b3'],
             output=data['output'],
             b1_grad=data['b1_grad'],
             b2_grad=data['b2_grad'],
             b3_grad=data['b3_grad'])
    print(f"\nSaved test data to {filename}")


if __name__ == "__main__":
    pass1 = test_unit_weights()
    data = test_random_weights()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Unit weights test: {'PASS' if pass1 else 'FAIL'}")

    # Save data for CUDA comparison
    save_test_data(data, '/home/nilkel/Projects/nest-splatting/scripts/test_mlp_cuda/mlp_test_data.npz')
