#!/usr/bin/env python3
"""
Simple test: Compare PyTorch MLP grads vs CUDA MLP grads with unit matrices.
"""
import torch
import torch.nn as nn

def test_pytorch_unit_mlp():
    """PyTorch reference: unit MLP backward with grad_out=1"""
    print("=" * 60)
    print("PyTorch Unit MLP Backward")
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

    # Input: batch of positive values (so ReLU doesn't clip)
    x = torch.ones(256, 40, requires_grad=True)  # All 1s

    # Forward
    y = mlp(x)
    print(f"Output shape: {y.shape}")
    print(f"Output[0]: {y[0].tolist()}")

    # Backward with grad_out = 1
    loss = y.sum()
    loss.backward()

    # Check gradients
    print(f"\nW1.grad nonzero: {(mlp[0].weight.grad.abs() > 1e-10).sum().item()}/{mlp[0].weight.grad.numel()}")
    print(f"b1.grad nonzero: {(mlp[0].bias.grad.abs() > 1e-10).sum().item()}/{mlp[0].bias.grad.numel()}")
    b1_nz = (mlp[0].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    print(f"b1.grad nonzero indices: {b1_nz}")

    print(f"\nW2.grad nonzero: {(mlp[2].weight.grad.abs() > 1e-10).sum().item()}/{mlp[2].weight.grad.numel()}")
    print(f"b2.grad nonzero: {(mlp[2].bias.grad.abs() > 1e-10).sum().item()}/{mlp[2].bias.grad.numel()}")
    b2_nz = (mlp[2].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    print(f"b2.grad nonzero indices: {b2_nz}")

    print(f"\nW3.grad nonzero: {(mlp[4].weight.grad.abs() > 1e-10).sum().item()}/{mlp[4].weight.grad.numel()}")
    print(f"b3.grad nonzero: {(mlp[4].bias.grad.abs() > 1e-10).sum().item()}/{mlp[4].bias.grad.numel()}")
    b3_nz = (mlp[4].bias.grad.abs() > 1e-10).nonzero().squeeze().tolist()
    print(f"b3.grad nonzero indices: {b3_nz}")

    # Print actual values
    print(f"\nb1.grad values: {mlp[0].bias.grad[:5].tolist()}")
    print(f"b2.grad values: {mlp[2].bias.grad[:5].tolist()}")
    print(f"b3.grad values: {mlp[4].bias.grad.tolist()}")


if __name__ == "__main__":
    test_pytorch_unit_mlp()
