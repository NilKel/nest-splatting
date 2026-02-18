#!/usr/bin/env python3
"""
Minimal test: Pass unit tensor through both MLPs and compare outputs.

Tests:
1. tcnn MLP (mlp_3D_direct) - used by 3D_direct mode
2. CUDA MLP (mlp_fused) - used by 3D_direct_lean mode
3. PyTorch reference MLP using extracted weights
"""

import os
import sys
import json
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from hash_encoder.modules import INGP
from hash_encoder.config import Config


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


def pytorch_mlp_forward(x, W1, W2, W3, apply_sigmoid=True):
    """Reference PyTorch MLP forward pass.

    Architecture: 40D -> 32D (ReLU) -> 32D (ReLU) -> 3D (sigmoid)
    """
    # Layer 1: [B, 40] @ [40, 32] -> [B, 32]
    h1 = torch.relu(x @ W1.T)
    # Layer 2: [B, 32] @ [32, 32] -> [B, 32]
    h2 = torch.relu(h1 @ W2.T)
    # Layer 3: [B, 32] @ [32, 3] -> [B, 3]
    out = h2 @ W3.T
    if apply_sigmoid:
        out = torch.sigmoid(out)
    return out, h1, h2


def main():
    parser = ArgumentParser(description="Test MLP outputs with unit tensor")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained 3D_direct model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")
    args = parser.parse_args()

    # Load training config
    print(f"\n[TEST] Loading config from: {args.model_path}")
    train_args = load_training_config(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    # Load YAML config
    config_yaml_path = os.path.join(args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(train_args.yaml)

    # Find iteration
    iteration = args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth checkpoints found")

    # Load TWO separate INGP models
    print("\n" + "="*60)
    print("LOADING MLPs")
    print("="*60)

    # Model 1: 3D_direct mode (uses tcnn)
    args_direct = Namespace(**vars(train_args))
    args_direct.method = "3D_direct"
    ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
    ingp_direct.load_model(args.model_path, iteration)
    print(f"[3D_direct] MLP type: {type(ingp_direct.mlp_3D_direct)}")

    # Model 2: 3D_direct_lean mode (uses mlp_fused)
    args_lean = Namespace(**vars(train_args))
    args_lean.method = "3D_direct_lean"
    ingp_lean = INGP(cfg_model, args=args_lean).to('cuda')
    ingp_lean.load_model(args.model_path, iteration)
    print(f"[3D_direct_lean] MLP type: {type(ingp_lean.mlp_fused)}")

    # Extract weights from lean MLP (PyTorch Linear layers)
    W1_lean = ingp_lean.mlp_fused[0].weight.data.clone()  # [32, 40]
    W2_lean = ingp_lean.mlp_fused[2].weight.data.clone()  # [32, 32]
    W3_lean = ingp_lean.mlp_fused[4].weight.data.clone()  # [3, 32]

    print(f"\n[LEAN] Weight shapes: W1={W1_lean.shape}, W2={W2_lean.shape}, W3={W3_lean.shape}")

    # Create test inputs
    print("\n" + "="*60)
    print("TEST 1: Unit tensor (all ones)")
    print("="*60)

    test_inputs = [
        ("unit (ones)", torch.ones(1, 40, device='cuda')),
        ("zeros", torch.zeros(1, 40, device='cuda')),
        ("random", torch.randn(1, 40, device='cuda')),
        ("range 0-1", torch.linspace(0, 1, 40, device='cuda').unsqueeze(0)),
    ]

    for name, x in test_inputs:
        print(f"\n--- Input: {name} ---")
        print(f"Input shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")

        # tcnn forward (fp16 internally) - NOTE: tcnn has output_activation="None"
        with torch.no_grad():
            out_tcnn_raw = ingp_direct.mlp_3D_direct(x.half()).float()
            out_tcnn_sigmoid = torch.sigmoid(out_tcnn_raw)

        # PyTorch forward using lean weights (WITHOUT sigmoid to match tcnn)
        with torch.no_grad():
            out_pytorch_raw, h1, h2 = pytorch_mlp_forward(x, W1_lean, W2_lean, W3_lean, apply_sigmoid=False)
            out_pytorch_sigmoid = torch.sigmoid(out_pytorch_raw)

        # Compare raw outputs (pre-sigmoid)
        diff_raw = (out_tcnn_raw - out_pytorch_raw).abs()
        # Compare sigmoid outputs
        diff_sigmoid = (out_tcnn_sigmoid - out_pytorch_sigmoid).abs()

        print(f"tcnn raw:       {out_tcnn_raw.squeeze().cpu().numpy()}")
        print(f"PyTorch raw:    {out_pytorch_raw.squeeze().cpu().numpy()}")
        print(f"Raw diff:       {diff_raw.squeeze().cpu().numpy()}")
        print(f"Raw MAE: {diff_raw.mean():.6f}, Max: {diff_raw.max():.6f}")
        print(f"tcnn sigmoid:   {out_tcnn_sigmoid.squeeze().cpu().numpy()}")
        print(f"PyTorch sigmoid:{out_pytorch_sigmoid.squeeze().cpu().numpy()}")
        print(f"Sigmoid MAE: {diff_sigmoid.mean():.6f}, Max: {diff_sigmoid.max():.6f}")

    # Test 2: Check weight ordering hypotheses
    print("\n" + "="*60)
    print("TEST 2: Weight extraction - trying REVERSE order [W3, W2, W1]")
    print("="*60)

    # Get tcnn weights
    tcnn_params = ingp_direct.mlp_3D_direct.params.data.float()
    print(f"tcnn total params: {tcnn_params.shape[0]}")

    in_dim = 40
    hidden_dim = 32
    out_dim = 3
    in_dim_padded = ((in_dim + 15) // 16) * 16  # 48
    out_dim_padded = ((out_dim + 15) // 16) * 16  # 16

    # Try REVERSE order: [W3, W2, W1] (output layer first)
    offset = 0
    # Layer 3 first: [out_padded, hidden] = [16, 32]
    w3_size = out_dim_padded * hidden_dim
    W3_rev = tcnn_params[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :].clone()
    offset += w3_size

    # Layer 2: [hidden, hidden] = [32, 32]
    w2_size = hidden_dim * hidden_dim
    W2_rev = tcnn_params[offset:offset+w2_size].view(hidden_dim, hidden_dim).clone()
    offset += w2_size

    # Layer 1 last: [hidden, in_padded] = [32, 48]
    w1_size = hidden_dim * in_dim_padded
    W1_rev = tcnn_params[offset:offset+w1_size].view(hidden_dim, in_dim_padded)[:, :in_dim].clone()

    print(f"Reverse order: W3{list(W3_rev.shape)}, W2{list(W2_rev.shape)}, W1{list(W1_rev.shape)}")

    # Test with reverse-order weights
    x_test = torch.randn(100, 40, device='cuda')
    with torch.no_grad():
        out_tcnn_test = ingp_direct.mlp_3D_direct(x_test.half()).float()
        out_rev, _, _ = pytorch_mlp_forward(x_test, W1_rev, W2_rev, W3_rev, apply_sigmoid=False)

    diff_rev = (out_tcnn_test - out_rev).abs()
    print(f"Reverse order MAE: {diff_rev.mean():.6f}, Max: {diff_rev.max():.6f}")

    # Test zeros with reverse order
    x_zero = torch.zeros(1, 40, device='cuda')
    with torch.no_grad():
        out_tcnn_zero = ingp_direct.mlp_3D_direct(x_zero.half()).float()
        out_rev_zero, _, _ = pytorch_mlp_forward(x_zero, W1_rev, W2_rev, W3_rev, apply_sigmoid=False)

    print(f"Zeros - tcnn: {out_tcnn_zero.squeeze().cpu().numpy()}")
    print(f"Zeros - reverse: {out_rev_zero.squeeze().cpu().numpy()}")
    print(f"Zeros diff: {(out_tcnn_zero - out_rev_zero).abs().squeeze().cpu().numpy()}")

    # Original order comparison
    print("\n--- Original order [W1, W2, W3] comparison ---")
    offset = 0
    w1_size = in_dim_padded * hidden_dim
    W1_orig = tcnn_params[offset:offset+w1_size].view(hidden_dim, in_dim_padded)[:, :in_dim].clone()
    offset += w1_size
    w2_size = hidden_dim * hidden_dim
    W2_orig = tcnn_params[offset:offset+w2_size].view(hidden_dim, hidden_dim).clone()
    offset += w2_size
    w3_size = hidden_dim * out_dim_padded
    W3_orig = tcnn_params[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :].clone()

    with torch.no_grad():
        out_orig, _, _ = pytorch_mlp_forward(x_test, W1_orig, W2_orig, W3_orig, apply_sigmoid=False)
        out_orig_zero, _, _ = pytorch_mlp_forward(x_zero, W1_orig, W2_orig, W3_orig, apply_sigmoid=False)

    diff_orig = (out_tcnn_test - out_orig).abs()
    print(f"Original order MAE: {diff_orig.mean():.6f}, Max: {diff_orig.max():.6f}")
    print(f"Zeros - original: {out_orig_zero.squeeze().cpu().numpy()}")

    # Check if what we loaded into mlp_fused matches
    print("\n--- Comparing loaded mlp_fused weights ---")
    print(f"W1_lean matches W1_orig: {torch.allclose(W1_lean.cpu(), W1_orig.cpu())}")
    print(f"W2_lean matches W2_orig: {torch.allclose(W2_lean.cpu(), W2_orig.cpu())}")
    print(f"W3_lean matches W3_orig: {torch.allclose(W3_lean.cpu(), W3_orig.cpu())}")

    # Test 3: Batch test with realistic input distribution
    print("\n" + "="*60)
    print("TEST 3: Batch test (1000 random samples)")
    print("="*60)

    x_batch = torch.randn(1000, 40, device='cuda')

    with torch.no_grad():
        out_tcnn_batch = ingp_direct.mlp_3D_direct(x_batch.half()).float()
        out_pytorch_batch, _, _ = pytorch_mlp_forward(x_batch, W1_lean, W2_lean, W3_lean)

    diff_batch = (out_tcnn_batch - out_pytorch_batch).abs()
    print(f"MAE: {diff_batch.mean():.6f}")
    print(f"Max: {diff_batch.max():.6f}")
    print(f"Std: {diff_batch.std():.6f}")

    # Per-channel analysis
    for c in range(3):
        print(f"  Channel {c}: MAE={diff_batch[:, c].mean():.6f}, Max={diff_batch[:, c].max():.6f}")

    # Check if the error is from fp16 precision
    print("\n" + "="*60)
    print("TEST 4: fp16 precision analysis")
    print("="*60)

    x_test = torch.randn(100, 40, device='cuda')

    # Full fp32 computation
    with torch.no_grad():
        out_fp32, _, _ = pytorch_mlp_forward(x_test, W1_lean, W2_lean, W3_lean)

    # fp16 computation then convert to fp32
    with torch.no_grad():
        x_half = x_test.half()
        W1_half = W1_lean.half()
        W2_half = W2_lean.half()
        W3_half = W3_lean.half()
        out_fp16, _, _ = pytorch_mlp_forward(x_half, W1_half, W2_half, W3_half)
        out_fp16 = out_fp16.float()

    diff_precision = (out_fp32 - out_fp16).abs()
    print(f"fp32 vs fp16 (PyTorch): MAE={diff_precision.mean():.6f}, Max={diff_precision.max():.6f}")

    diff_tcnn_fp32 = (out_tcnn_batch[:100] - out_fp32).abs()
    print(f"tcnn vs fp32 PyTorch: MAE={diff_tcnn_fp32.mean():.6f}, Max={diff_tcnn_fp32.max():.6f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if diff_batch.mean() < 0.01:
        print("[OK] MLP outputs match within acceptable tolerance (MAE < 0.01)")
        print("     Difference is likely due to fp16 precision in tcnn")
    else:
        print("[WARNING] MLP outputs differ significantly!")
        print("     Need to investigate weight extraction or activation differences")


if __name__ == "__main__":
    main()
