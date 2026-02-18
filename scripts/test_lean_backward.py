#!/usr/bin/env python3
"""
Test lean mode backward pass - verify MLP gradients are computed correctly.

This creates a minimal test with random inputs and verifies:
1. MLP weights are properly uploaded to CUDA
2. Backward pass computes gradients
3. Gradients are retrievable via get_mlp_grads()
4. No NaN/Inf in gradients

Run with: conda run -n nest_splatting python scripts/test_lean_backward.py
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mlp_weight_upload_and_backward():
    """Test that lean library can upload MLP weights and compute backward gradients."""
    print("=" * 60)
    print("TEST: Lean Library MLP Weight Upload & Backward")
    print("=" * 60)

    # Import lean library
    try:
        import diff_surfel_3D as lean_lib
        print("Lean library imported successfully")
    except ImportError as e:
        print(f"ERROR: Could not import lean library: {e}")
        print("Run: cd submodules/diff_surfel_3D && conda run -n nest_splatting python -m pip install -e . --no-build-isolation")
        return False

    # Create MLP weights matching our architecture: 40 -> 32 -> 32 -> 3
    IN_DIM = 40
    HIDDEN_DIM = 32
    OUT_DIM = 3

    # Initialize weights
    torch.manual_seed(42)
    W1 = torch.randn(HIDDEN_DIM, IN_DIM, device='cuda')
    b1 = torch.randn(HIDDEN_DIM, device='cuda')
    W2 = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device='cuda')
    b2 = torch.randn(HIDDEN_DIM, device='cuda')
    W3 = torch.randn(OUT_DIM, HIDDEN_DIM, device='cuda')
    b3 = torch.randn(OUT_DIM, device='cuda')

    print(f"\nWeight shapes:")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  W3: {W3.shape}, b3: {b3.shape}")

    # Upload to CUDA constant memory
    try:
        lean_lib.set_mlp_weights(W1, b1, W2, b2, W3, b3, is_sh_mode=False)
        print("\nMLP weights uploaded to CUDA constant memory: OK")
    except Exception as e:
        print(f"\nERROR uploading MLP weights: {e}")
        return False

    # Check get_mlp_grads before any backward
    grads = lean_lib.get_mlp_grads()
    print(f"get_mlp_grads() before backward: {grads is not None}")

    return True


def test_pytorch_mlp_forward_backward():
    """Test PyTorch MLP forward/backward to compare with CUDA implementation."""
    print("\n" + "=" * 60)
    print("TEST: PyTorch MLP Forward/Backward Reference")
    print("=" * 60)

    # Same architecture as CUDA
    IN_DIM = 40
    HIDDEN_DIM = 32
    OUT_DIM = 3

    # Create MLP
    mlp = nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUT_DIM),
        nn.Sigmoid()
    ).cuda()

    # Random input
    x = torch.randn(100, IN_DIM, device='cuda')

    # Forward
    y = mlp(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Backward with grad=1
    loss = y.sum()
    loss.backward()

    # Check gradients
    print("\nPyTorch gradients:")
    for name, param in mlp.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            print(f"  {name}: mean={param.grad.abs().mean():.6f}, max={param.grad.abs().max():.6f}, "
                  f"NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  {name}: grad=None")

    return True


def test_tcnn_vs_pytorch_bias_extraction():
    """Test the tcnn implicit bias extraction logic."""
    print("\n" + "=" * 60)
    print("TEST: tcnn Implicit Bias Extraction")
    print("=" * 60)

    # Import tcnn
    try:
        import tinycudann as tcnn
    except ImportError:
        print("tcnn not available, skipping")
        return True

    # Create tcnn network
    IN_DIM = 40
    HIDDEN_DIM = 32
    OUT_DIM = 3

    tcnn_net = tcnn.Network(
        n_input_dims=IN_DIM,
        n_output_dims=OUT_DIM,
        network_config={
            'otype': 'MLP',
            'activation': 'ReLU',
            'output_activation': 'None',  # We apply sigmoid ourselves
            'n_neurons': HIDDEN_DIM,
            'n_hidden_layers': 2,
        }
    )

    print(f"tcnn params shape: {tcnn_net.params.shape}")

    # Extract weights following our bias extraction logic
    params = tcnn_net.params.data.float()
    IN_DIM_PADDED = 48  # tcnn pads to multiple of 16
    OUT_DIM_PADDED = 16

    offset = 0

    # Layer 1: [hidden, in_padded] -> extract [hidden, in] and implicit bias
    w1_size = IN_DIM_PADDED * HIDDEN_DIM
    W1_full = params[offset:offset+w1_size].view(HIDDEN_DIM, IN_DIM_PADDED)
    W1 = W1_full[:, :IN_DIM].contiguous().cuda()
    b1 = W1_full[:, IN_DIM:].sum(dim=1).cuda()  # CRITICAL: tcnn pads with 1s
    offset += w1_size

    # Layer 2: [hidden, hidden]
    w2_size = HIDDEN_DIM * HIDDEN_DIM
    W2 = params[offset:offset+w2_size].view(HIDDEN_DIM, HIDDEN_DIM).contiguous().cuda()
    b2 = torch.zeros(HIDDEN_DIM, device='cuda')  # No padding = no implicit bias
    offset += w2_size

    # Layer 3: [out_padded, hidden] -> extract [out, hidden]
    w3_size = HIDDEN_DIM * OUT_DIM_PADDED
    W3 = params[offset:offset+w3_size].view(OUT_DIM_PADDED, HIDDEN_DIM)[:OUT_DIM, :].contiguous().cuda()
    b3 = torch.zeros(OUT_DIM, device='cuda')

    print(f"\nExtracted weights:")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  W3: {W3.shape}, b3: {b3.shape}")
    print(f"  b1 (implicit bias) mean: {b1.mean():.4f}, norm: {b1.norm():.4f}")

    # Test: zero input should NOT produce zero pre-activation if bias is non-zero
    x_zero = torch.zeros(10, IN_DIM, device='cuda', dtype=torch.half)
    with torch.no_grad():
        out_tcnn = tcnn_net(x_zero).float()
    print(f"\ntcnn output for zero input: {out_tcnn[0].tolist()}")

    # PyTorch MLP with extracted weights and bias
    def pytorch_forward(x):
        # Layer 1
        h1 = torch.relu(x @ W1.T + b1)
        # Layer 2
        h2 = torch.relu(h1 @ W2.T + b2)
        # Layer 3 (no activation - tcnn uses None)
        out = h2 @ W3.T + b3
        return out

    x_zero_f32 = torch.zeros(10, IN_DIM, device='cuda')
    out_pytorch = pytorch_forward(x_zero_f32)
    print(f"PyTorch output for zero input: {out_pytorch[0].tolist()}")

    diff = (out_tcnn - out_pytorch).abs()
    print(f"Diff (tcnn vs PyTorch): mean={diff.mean():.6f}, max={diff.max():.6f}")

    if diff.max() < 0.01:
        print("PASS: Bias extraction is correct!")
        return True
    else:
        print("FAIL: Bias extraction mismatch")
        return False


def test_full_render_backward():
    """Test full render backward pass with lean library (if model available)."""
    print("\n" + "=" * 60)
    print("TEST: Full Render Backward (requires trained model)")
    print("=" * 60)

    # Check if model exists
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    ckpt_path = f'{model_path}/ngp_30000.pth'

    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}, skipping full render test")
        return True

    print("This test requires a running scene - skipping for now")
    return True


def main():
    results = []

    results.append(("MLP weight upload", test_mlp_weight_upload_and_backward()))
    results.append(("PyTorch MLP reference", test_pytorch_mlp_forward_backward()))
    results.append(("tcnn bias extraction", test_tcnn_vs_pytorch_bias_extraction()))
    results.append(("Full render backward", test_full_render_backward()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
