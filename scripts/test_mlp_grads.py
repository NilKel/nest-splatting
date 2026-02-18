#!/usr/bin/env python3
"""
Test MLP gradient computation: compare tcnn backward vs CUDA lean backward.

This tests the gradient flow through the MLP to catch any discrepancies
that could cause training divergence or NaNs.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytorch_mlp_forward_with_grad(x, W1, b1, W2, b2, W3, b3):
    """PyTorch MLP forward pass with gradient tracking."""
    # Layer 1: [B, 40] @ [40, 32] -> [B, 32]
    h1_pre = x @ W1.T + b1
    h1 = torch.relu(h1_pre)
    # Layer 2: [B, 32] @ [32, 32] -> [B, 32]
    h2_pre = h1 @ W2.T + b2
    h2 = torch.relu(h2_pre)
    # Layer 3: [B, 32] @ [32, 3] -> [B, 3]
    out_pre = h2 @ W3.T + b3
    out = torch.sigmoid(out_pre)
    return out


def main():
    print("="*60)
    print("MLP GRADIENT TEST: PyTorch vs CUDA lean")
    print("="*60)

    # Load trained weights from checkpoint
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    ckpt = torch.load(f'{model_path}/ngp_30000.pth', map_location='cpu')

    # Extract tcnn weights
    tcnn_params = ckpt['model_state_dict']['mlp_3D_direct.params'].float()
    print(f"tcnn params shape: {tcnn_params.shape}")

    # Extract weights with implicit bias (matching our new extraction)
    in_dim = 40
    hidden_dim = 32
    out_dim = 3
    in_dim_padded = 48
    out_dim_padded = 16

    offset = 0
    # Layer 1: [hidden, in_padded] -> extract [hidden, in] and implicit bias
    w1_size = in_dim_padded * hidden_dim
    W1_full = tcnn_params[offset:offset+w1_size].view(hidden_dim, in_dim_padded)
    W1 = W1_full[:, :in_dim].contiguous().cuda()
    b1 = W1_full[:, in_dim:].sum(dim=1).contiguous().cuda()  # Implicit bias!
    offset += w1_size

    # Layer 2: [hidden, hidden]
    w2_size = hidden_dim * hidden_dim
    W2 = tcnn_params[offset:offset+w2_size].view(hidden_dim, hidden_dim).contiguous().cuda()
    b2 = torch.zeros(hidden_dim, device='cuda')  # No padding = no implicit bias
    offset += w2_size

    # Layer 3: [out_padded, hidden] -> extract [out, hidden]
    w3_size = hidden_dim * out_dim_padded
    W3 = tcnn_params[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :].contiguous().cuda()
    b3 = torch.zeros(out_dim, device='cuda')  # No bias for output

    print(f"\nExtracted weights:")
    print(f"  W1: {W1.shape}, b1: {b1.shape} (implicit bias from tcnn padding)")
    print(f"  W2: {W2.shape}, b2: {b2.shape} (zeros)")
    print(f"  W3: {W3.shape}, b3: {b3.shape} (zeros)")
    print(f"  b1 mean: {b1.mean():.4f}")

    # Test 1: Forward pass verification
    print("\n" + "="*60)
    print("TEST 1: Forward pass verification")
    print("="*60)

    x = torch.randn(100, 40, device='cuda')

    # PyTorch forward
    out_pytorch = pytorch_mlp_forward_with_grad(x, W1, b1, W2, b2, W3, b3)

    # tcnn forward (need to reload the model)
    import tinycudann as tcnn
    tcnn_net = tcnn.Network(
        n_input_dims=40,
        n_output_dims=3,
        network_config={
            'otype': 'MLP',
            'activation': 'ReLU',
            'output_activation': 'None',
            'n_neurons': 32,
            'n_hidden_layers': 2,
        }
    )
    with torch.no_grad():
        tcnn_net.params.data.copy_(tcnn_params.cuda())

    out_tcnn_raw = tcnn_net(x.half()).float()
    out_tcnn = torch.sigmoid(out_tcnn_raw)

    diff = (out_pytorch - out_tcnn).abs()
    print(f"Forward MAE: {diff.mean():.6f}, Max: {diff.max():.6f}")

    # Test 2: Gradient comparison (loss = output.sum())
    print("\n" + "="*60)
    print("TEST 2: Gradient comparison (loss = output.sum())")
    print("="*60)

    # PyTorch gradients
    W1_pt = W1.clone().requires_grad_(True)
    b1_pt = b1.clone().requires_grad_(True)
    W2_pt = W2.clone().requires_grad_(True)
    b2_pt = b2.clone().requires_grad_(True)
    W3_pt = W3.clone().requires_grad_(True)
    b3_pt = b3.clone().requires_grad_(True)

    x_pt = x.clone()
    out_pt = pytorch_mlp_forward_with_grad(x_pt, W1_pt, b1_pt, W2_pt, b2_pt, W3_pt, b3_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    print(f"\nPyTorch gradients:")
    print(f"  dL/dW1: mean={W1_pt.grad.abs().mean():.6f}, max={W1_pt.grad.abs().max():.6f}")
    print(f"  dL/db1: mean={b1_pt.grad.abs().mean():.6f}, max={b1_pt.grad.abs().max():.6f}")
    print(f"  dL/dW2: mean={W2_pt.grad.abs().mean():.6f}, max={W2_pt.grad.abs().max():.6f}")
    print(f"  dL/db2: mean={b2_pt.grad.abs().mean():.6f}, max={b2_pt.grad.abs().max():.6f}")
    print(f"  dL/dW3: mean={W3_pt.grad.abs().mean():.6f}, max={W3_pt.grad.abs().max():.6f}")
    print(f"  dL/db3: mean={b3_pt.grad.abs().mean():.6f}, max={b3_pt.grad.abs().max():.6f}")

    # tcnn gradients
    tcnn_net.params.grad = None
    out_tcnn_grad = tcnn_net(x.half()).float()
    out_tcnn_sig = torch.sigmoid(out_tcnn_grad)
    loss_tcnn = out_tcnn_sig.sum()
    loss_tcnn.backward()

    # Extract tcnn gradients (same layout as weights)
    tcnn_grad = tcnn_net.params.grad.float()
    offset = 0

    # Layer 1 gradients: [hidden, in_padded] - extract [hidden, in] only
    grad_W1_full = tcnn_grad[offset:offset+w1_size].view(hidden_dim, in_dim_padded)
    grad_W1_tcnn = grad_W1_full[:, :in_dim]
    grad_b1_tcnn = grad_W1_full[:, in_dim:].sum(dim=1)  # Gradient of implicit bias
    offset += w1_size

    # Layer 2 gradients
    grad_W2_tcnn = tcnn_grad[offset:offset+w2_size].view(hidden_dim, hidden_dim)
    offset += w2_size

    # Layer 3 gradients
    grad_W3_tcnn = tcnn_grad[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :]

    print(f"\ntcnn gradients:")
    print(f"  dL/dW1: mean={grad_W1_tcnn.abs().mean():.6f}, max={grad_W1_tcnn.abs().max():.6f}")
    print(f"  dL/db1 (implicit): mean={grad_b1_tcnn.abs().mean():.6f}, max={grad_b1_tcnn.abs().max():.6f}")
    print(f"  dL/dW2: mean={grad_W2_tcnn.abs().mean():.6f}, max={grad_W2_tcnn.abs().max():.6f}")
    print(f"  dL/dW3: mean={grad_W3_tcnn.abs().mean():.6f}, max={grad_W3_tcnn.abs().max():.6f}")

    # Compare
    print(f"\nGradient comparison (PyTorch vs tcnn):")
    w1_diff = (W1_pt.grad - grad_W1_tcnn.cuda()).abs()
    b1_diff = (b1_pt.grad - grad_b1_tcnn.cuda()).abs()
    w2_diff = (W2_pt.grad - grad_W2_tcnn.cuda()).abs()
    w3_diff = (W3_pt.grad - grad_W3_tcnn.cuda()).abs()

    print(f"  dL/dW1 diff: mean={w1_diff.mean():.6f}, max={w1_diff.max():.6f}")
    print(f"  dL/db1 diff: mean={b1_diff.mean():.6f}, max={b1_diff.max():.6f}")
    print(f"  dL/dW2 diff: mean={w2_diff.mean():.6f}, max={w2_diff.max():.6f}")
    print(f"  dL/dW3 diff: mean={w3_diff.mean():.6f}, max={w3_diff.max():.6f}")

    # Cosine similarity
    def cos_sim(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    print(f"\nCosine similarity:")
    print(f"  dL/dW1: {cos_sim(W1_pt.grad, grad_W1_tcnn.cuda()):.6f}")
    print(f"  dL/db1: {cos_sim(b1_pt.grad, grad_b1_tcnn.cuda()):.6f}")
    print(f"  dL/dW2: {cos_sim(W2_pt.grad, grad_W2_tcnn.cuda()):.6f}")
    print(f"  dL/dW3: {cos_sim(W3_pt.grad, grad_W3_tcnn.cuda()):.6f}")

    # Test 3: Check for NaN-prone inputs
    print("\n" + "="*60)
    print("TEST 3: NaN-prone input patterns")
    print("="*60)

    nan_tests = [
        ("zeros", torch.zeros(10, 40, device='cuda')),
        ("ones", torch.ones(10, 40, device='cuda')),
        ("large", torch.ones(10, 40, device='cuda') * 100),
        ("small", torch.ones(10, 40, device='cuda') * 0.001),
        ("negative", -torch.ones(10, 40, device='cuda')),
    ]

    for name, test_x in nan_tests:
        # Forward
        out = pytorch_mlp_forward_with_grad(test_x, W1, b1, W2, b2, W3, b3)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"  {name}: output range=[{out.min():.4f}, {out.max():.4f}], NaN={has_nan}, Inf={has_inf}")

        # Check gradient
        W1_test = W1.clone().requires_grad_(True)
        out_test = pytorch_mlp_forward_with_grad(test_x, W1_test, b1, W2, b2, W3, b3)
        loss_test = out_test.sum()
        loss_test.backward()
        grad_nan = torch.isnan(W1_test.grad).any().item()
        grad_inf = torch.isinf(W1_test.grad).any().item()
        print(f"          grad_W1: NaN={grad_nan}, Inf={grad_inf}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
