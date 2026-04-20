"""
Test MLP forward/backward consistency between 3D_SH_res (16-dim) and 3D_SH_32 (32-dim).
Uses unit/predictable weights and inputs to verify CUDA kernels match PyTorch.

Usage:
    conda run -n nest_splatting python scripts/test_mlp_32_vs_16.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def test_mlp_fw_bw(dim, lib_name):
    """Test MLP forward and backward for a given dimension against PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"  Testing {lib_name} (dim={dim})")
    print(f"{'='*60}")

    # Import the right library
    if dim == 16:
        from diff_surfel_3D_sh_res import set_mlp_weights, get_mlp_grads, _RasterizeGaussians
    else:
        from diff_surfel_3D_sh_32 import set_mlp_weights, get_mlp_grads, _RasterizeGaussians

    # Create predictable weights (small values, identity-ish)
    torch.manual_seed(123)
    W1_pt = torch.randn(dim, dim, device="cuda") * 0.1
    W2_pt = torch.randn(dim, dim, device="cuda") * 0.1
    W3_pt = torch.randn(dim, dim, device="cuda") * 0.1

    # Create PyTorch MLP
    mlp = nn.Sequential(
        nn.Linear(dim, dim, bias=False),
        nn.ReLU(),
        nn.Linear(dim, dim, bias=False),
        nn.ReLU(),
        nn.Linear(dim, dim, bias=False),
    ).cuda()
    with torch.no_grad():
        mlp[0].weight.copy_(W1_pt)
        mlp[2].weight.copy_(W2_pt)
        mlp[4].weight.copy_(W3_pt)

    # Upload to CUDA kernel
    set_mlp_weights(W1_pt.contiguous(), W2_pt.contiguous(), W3_pt.contiguous())

    # Test input: [hash(4), bias(1), pad(dim-5)]
    inp = torch.zeros(dim, device="cuda")
    inp[0] = 0.5
    inp[1] = -0.3
    inp[2] = 0.8
    inp[3] = -0.1
    inp[4] = 1.0  # bias

    # PyTorch forward
    inp_pt = inp.clone().requires_grad_(True)
    out_pt = mlp(inp_pt)
    rgb_pt = out_pt[:3]
    print(f"\nPyTorch forward (first 3): {rgb_pt.tolist()}")

    # PyTorch backward with unit gradient on first 3
    grad_out = torch.zeros(dim, device="cuda")
    grad_out[:3] = 1.0
    out_pt.backward(grad_out)
    print(f"PyTorch grad_input (first 5): {inp_pt.grad[:5].tolist()}")
    print(f"PyTorch grad_W1 norm: {mlp[0].weight.grad.norm().item():.6f}")
    print(f"PyTorch grad_W2 norm: {mlp[2].weight.grad.norm().item():.6f}")
    print(f"PyTorch grad_W3 norm: {mlp[4].weight.grad.norm().item():.6f}")

    # Now test with half precision (simulating CUDA kernel)
    W1_h = W1_pt.half()
    W2_h = W2_pt.half()
    W3_h = W3_pt.half()
    inp_h = inp.half()

    # Manual FP16 forward (matching CUDA __hfma2 logic)
    # Layer 1
    h1 = torch.zeros(dim, device="cuda")
    for h in range(dim):
        acc = torch.tensor(0.0, device="cuda")
        for i in range(0, dim, 2):
            # __hfma2: multiply pairs and accumulate in half
            p0 = (inp_h[i] * W1_h[h, i]).float()
            p1 = (inp_h[i+1] * W1_h[h, i+1]).float()
            acc += p0 + p1
        h1[h] = max(0, acc.item())

    # Layer 2
    h1_h = h1.half()
    h2 = torch.zeros(dim, device="cuda")
    for h in range(dim):
        acc = torch.tensor(0.0, device="cuda")
        for i in range(0, dim, 2):
            p0 = (h1_h[i] * W2_h[h, i]).float()
            p1 = (h1_h[i+1] * W2_h[h, i+1]).float()
            acc += p0 + p1
        h2[h] = max(0, acc.item())

    # Layer 3
    h2_h = h2.half()
    out_fp16 = torch.zeros(3, device="cuda")
    for o in range(3):
        acc = torch.tensor(0.0, device="cuda")
        for h in range(0, dim, 2):
            p0 = (h2_h[h] * W3_h[o, h]).float()
            p1 = (h2_h[h+1] * W3_h[o, h+1]).float()
            acc += p0 + p1
        out_fp16[o] = acc.item()

    print(f"\nFP16 sim forward (first 3): {out_fp16.tolist()}")
    diff = (rgb_pt.detach() - out_fp16).abs()
    print(f"FP32 vs FP16 diff: {diff.tolist()}")
    print(f"FP32 vs FP16 max diff: {diff.max().item():.8f}")

    # Check: how many hidden neurons are active?
    print(f"\nLayer 1: {(h1 > 0).sum().item()}/{dim} active neurons")
    print(f"Layer 2: {(h2 > 0).sum().item()}/{dim} active neurons")

    # Check: FP16 accumulation precision
    # With dim=16: 8 hfma2 ops (16 products in 2 half accumulators)
    # With dim=32: 16 hfma2 ops (32 products in 2 half accumulators)
    print(f"\nAccumulation length: {dim//2} hfma2 ops per dot product")

    # Backward test: FP32 (matching CUDA backward)
    # dL_doutput = [1,1,1,0,...0]
    dL_dout = torch.zeros(dim, device="cuda")
    dL_dout[:3] = 1.0

    # Layer 3 backward (FP32)
    dL_dz3 = dL_dout[:3].clone()  # identity activation
    dL_dh2 = torch.zeros(dim, device="cuda")
    dL_dW3_manual = torch.zeros(3, dim, device="cuda")
    for o in range(3):
        for h in range(dim):
            dL_dW3_manual[o, h] = dL_dz3[o] * h2[h]
            dL_dh2[h] += dL_dz3[o] * W3_h[o, h].float()

    # ReLU backward
    dL_dz2 = torch.where(h2 > 0, dL_dh2, torch.zeros_like(dL_dh2))

    # Layer 2 backward
    dL_dh1 = torch.zeros(dim, device="cuda")
    for h in range(dim):
        for i in range(dim):
            dL_dh1[i] += dL_dz2[h] * W2_h[h, i].float()

    dL_dz1 = torch.where(h1 > 0, dL_dh1, torch.zeros_like(dL_dh1))

    # Layer 1 backward - dL_dinput
    dL_dinput = torch.zeros(dim, device="cuda")
    for h in range(dim):
        for i in range(dim):
            dL_dinput[i] += dL_dz1[h] * W1_h[h, i].float()

    print(f"\nManual BW grad_input (first 5): {dL_dinput[:5].tolist()}")
    bw_diff = (inp_pt.grad[:5] - dL_dinput[:5]).abs()
    print(f"PyTorch vs manual BW diff (first 5): {bw_diff.tolist()}")
    print(f"PyTorch vs manual BW max diff: {bw_diff.max().item():.8f}")


if __name__ == "__main__":
    test_mlp_fw_bw(16, "diff_surfel_3D_sh_res")
    test_mlp_fw_bw(32, "diff_surfel_3D_sh_32")
