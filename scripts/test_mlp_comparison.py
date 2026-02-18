#!/usr/bin/env python3
"""
Compare CUDA MLP (3D_direct_fused) against PyTorch MLP in identical settings.

This test:
1. Loads Gaussians from a checkpoint
2. Runs rendering through both MLP implementations with identical weights
3. Compares forward outputs and backward gradients
"""

import os
import sys
import json
import pickle
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams

MODEL_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


def create_pytorch_mlp(input_dim=40, hidden_dim=32, output_dim=3):
    """Create a PyTorch MLP matching the CUDA MLP architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid()
    ).cuda()


def set_unit_weights_pytorch(mlp, input_dim=40, hidden_dim=32, output_dim=3):
    """Set unit/diagonal weights on PyTorch MLP."""
    with torch.no_grad():
        # Layer 0: input_dim -> hidden_dim (diagonal for first hidden_dim inputs)
        mlp[0].weight.zero_()
        mlp[0].bias.zero_()
        for i in range(min(hidden_dim, input_dim)):
            mlp[0].weight[i, i] = 1.0

        # Layer 2: hidden_dim -> hidden_dim (identity)
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        for i in range(hidden_dim):
            mlp[2].weight[i, i] = 1.0

        # Layer 4: hidden_dim -> output_dim (diagonal for first output_dim)
        mlp[4].weight.zero_()
        mlp[4].bias.zero_()
        for i in range(output_dim):
            mlp[4].weight[i, i] = 1.0


def test_mlp_standalone():
    """Test MLP forward/backward with synthetic data (no rasterizer)."""
    print("="*80)
    print("STANDALONE MLP COMPARISON (No Rasterizer)")
    print("="*80)

    # Create identical inputs
    batch_size = 256
    input_dim = 40
    hidden_dim = 32
    output_dim = 3

    # Random input with gradients
    x = torch.randn(batch_size, input_dim, device='cuda', requires_grad=True)

    # Create PyTorch MLP
    pytorch_mlp = create_pytorch_mlp(input_dim, hidden_dim, output_dim)
    set_unit_weights_pytorch(pytorch_mlp, input_dim, hidden_dim, output_dim)

    # Forward through PyTorch
    pytorch_mlp.zero_grad()
    y_pytorch = pytorch_mlp(x)
    loss_pytorch = y_pytorch.sum()
    loss_pytorch.backward()

    # Get PyTorch gradients
    pytorch_b1_grad = pytorch_mlp[0].bias.grad.clone()
    pytorch_b2_grad = pytorch_mlp[2].bias.grad.clone()
    pytorch_b3_grad = pytorch_mlp[4].bias.grad.clone()

    print(f"\n[PYTORCH MLP] Forward output[0]: {y_pytorch[0].tolist()}")
    print(f"[PYTORCH MLP] Loss: {loss_pytorch.item():.4f}")

    # Check bias gradient sparsity
    b1_nz = (pytorch_b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (pytorch_b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (pytorch_b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\n[PYTORCH GRADS] b1 nonzeros: {len(b1_nz)}/{hidden_dim} at {b1_nz}")
    print(f"[PYTORCH GRADS] b2 nonzeros: {len(b2_nz)}/{hidden_dim} at {b2_nz}")
    print(f"[PYTORCH GRADS] b3 nonzeros: {len(b3_nz)}/{output_dim} at {b3_nz}")
    print(f"[PYTORCH GRADS] Expected: all at [0, 1, 2]")

    # CUDA MLP needs to go through rasterizer - can't test standalone
    print(f"\n[NOTE] CUDA MLP requires rasterizer path - see test_with_rasterizer()")

    print("\n" + "="*80)


def test_with_rasterizer():
    """Test MLP through actual rasterizer pipeline."""
    print("="*80)
    print("RASTERIZER MLP COMPARISON")
    print("="*80)

    # Load config and setup
    args = load_training_config(MODEL_PATH)
    args.model_path = MODEL_PATH
    args.eval = True

    config_yaml_path = os.path.join(MODEL_PATH, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    # Find iteration
    import glob
    ngp_files = glob.glob(os.path.join(MODEL_PATH, "ngp_*.pth"))
    iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
    iteration = max(iterations)

    # Setup params
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP and Gaussians
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(MODEL_PATH, iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    cameras = scene.getTestCameras()
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Keep a few Gaussians for faster testing
    num_keep = 100
    print(f"\n[SETUP] Keeping {num_keep} Gaussians for testing")

    # Keep Gaussians with highest opacity
    opacities = gaussians.get_opacity.squeeze()
    _, top_indices = torch.topk(opacities, min(num_keep, len(opacities)))
    mask = torch.zeros(len(opacities), dtype=torch.bool, device="cuda")
    mask[top_indices] = True

    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._appearance_level = gaussians._appearance_level[mask]
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        gaussians._gaussian_features = gaussians._gaussian_features[mask.to(gaussians._gaussian_features.device)]

    # Make them bigger to ensure visibility
    gaussians._scaling = gaussians._scaling + 1.0

    cam = cameras[0]

    # Store original method
    original_method = args.method

    # ============================================
    # Test: 3D_direct_fused mode (CUDA MLP with unit weights)
    # ============================================
    print(f"\n--- 3D_direct_fused mode (CUDA MLP with unit weights) ---")

    try:
        from diff_surfel_3D import set_mlp_weights, get_mlp_grads

        # Create PyTorch MLP with same architecture for reference
        hidden_dim = 32
        pytorch_mlp = create_pytorch_mlp(40, hidden_dim, 3)
        set_unit_weights_pytorch(pytorch_mlp, 40, hidden_dim, 3)

        # Create mlp_fused if it doesn't exist (model loaded from 3D_direct checkpoint)
        if ingp_model.mlp_fused is None:
            print(f"[SETUP] Creating mlp_fused (not in loaded checkpoint)")
            ingp_model.mlp_fused = create_pytorch_mlp(40, hidden_dim, 3)

        # IMPORTANT: Set unit weights on ingp_model.mlp_fused directly!
        # The render() function calls ingp.get_fused_mlp_weights() which extracts
        # weights from mlp_fused and uploads them. If we call set_mlp_weights()
        # directly, render() will overwrite our weights with mlp_fused's weights.
        print(f"[SETUP] Setting unit weights on ingp_model.mlp_fused")

        with torch.no_grad():
            # Layer 0: 40 -> 32 (diagonal for first 32 inputs)
            ingp_model.mlp_fused[0].weight.zero_()
            ingp_model.mlp_fused[0].bias.zero_()
            for i in range(min(32, 40)):
                ingp_model.mlp_fused[0].weight[i, i] = 1.0

            # Layer 2: 32 -> 32 (identity)
            ingp_model.mlp_fused[2].weight.zero_()
            ingp_model.mlp_fused[2].bias.zero_()
            for i in range(32):
                ingp_model.mlp_fused[2].weight[i, i] = 1.0

            # Layer 4: 32 -> 3 (diagonal for first 3)
            ingp_model.mlp_fused[4].weight.zero_()
            ingp_model.mlp_fused[4].bias.zero_()
            for i in range(3):
                ingp_model.mlp_fused[4].weight[i, i] = 1.0

        # Verify weights
        W1 = ingp_model.mlp_fused[0].weight.data
        W3 = ingp_model.mlp_fused[4].weight.data
        print(f"  W1[0,0]={W1[0,0].item():.1f}, W1[1,1]={W1[1,1].item():.1f}, W1[2,2]={W1[2,2].item():.1f}")
        print(f"  W2 diagonal: identity")
        print(f"  W3[0,0]={W3[0,0].item():.1f}, W3[1,1]={W3[1,1].item():.1f}, W3[2,2]={W3[2,2].item():.1f}")

        # Force 3D_direct_fused mode by setting the flag directly on ingp_model
        # (args.method is only checked at INGP construction time)
        ingp_model.is_3D_direct_mode = False
        ingp_model.is_3D_direct_fused_mode = True
        args.method = "3D_direct_fused"
        print(f"[SETUP] Set is_3D_direct_fused_mode=True on INGP model")

        # Also set the lean mode flag so it uses diff_surfel_3D
        ingp_model.is_3D_direct_lean_mode = True

        # Render with fused mode (use fast_inference to skip post-processing)
        render_pkg_fused = render(cam, gaussians, pipe, background, ingp=ingp_model,
                                 beta=beta, iteration=iteration, cfg=cfg_model,
                                 fast_inference=True)

        rendered_fused = render_pkg_fused["render"]

        # Use mean loss to avoid gradient explosion
        visible_mask = rendered_fused.abs() > 0
        if visible_mask.any():
            loss_fused = rendered_fused[visible_mask].mean()
        else:
            loss_fused = rendered_fused.mean()

        print(f"\n[FORWARD] Rendered mean: {rendered_fused.mean().item():.6f}")
        print(f"[FORWARD] Rendered max: {rendered_fused.max().item():.6f}")
        print(f"[FORWARD] Loss (mean visible): {loss_fused.item():.6f}")

        loss_fused.backward()

        # Get CUDA MLP gradients - returns tuple: (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3)
        grads = get_mlp_grads()

        if grads is None:
            print(f"\n[ERROR] get_mlp_grads() returned None!")
        else:
            print(f"\n[DEBUG] grads type: {type(grads)}, len: {len(grads)}")
            dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3 = grads

            print(f"[DEBUG] dL_db2 type: {type(dL_db2)}, shape: {dL_db2.shape if hasattr(dL_db2, 'shape') else 'N/A'}")

            b1_nz = (dL_db1.abs() > 1e-10).nonzero().squeeze(-1).tolist()
            b2_nz = (dL_db2.abs() > 1e-10).nonzero().squeeze(-1).tolist()
            b3_nz = (dL_db3.abs() > 1e-10).nonzero().squeeze(-1).tolist()

            print(f"\n[CUDA GRADS] dL_db1 nonzeros: {len(b1_nz)}/{len(dL_db1)} at {b1_nz}")
            print(f"[CUDA GRADS] dL_db2 nonzeros: {len(b2_nz)}/{len(dL_db2)} at {b2_nz}")
            print(f"[CUDA GRADS] dL_db3 nonzeros: {len(b3_nz)}/{len(dL_db3)} at {b3_nz}")
            print(f"[EXPECTED] For unit weights: [0, 1, 2]")

            # Print actual values
            print(f"\n[VALUES] dL_db2[0:5]: {dL_db2[:5].tolist()}")

            if b2_nz == [0, 1, 2]:
                print(f"\n*** CUDA MLP: PASS - Correct gradient sparsity ***")
            elif len(b2_nz) == 16:
                print(f"\n*** CUDA MLP: FAIL - 16/32 bug detected ***")
            else:
                print(f"\n*** CUDA MLP: {len(b2_nz)}/32 nonzeros (unexpected) ***")

    except ImportError as e:
        print(f"[ERROR] diff_surfel_3D not available: {e}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    # Restore method
    args.method = original_method

    print("\n" + "="*80)


def test_pytorch_mlp_only():
    """Quick test of PyTorch MLP with unit weights."""
    print("="*80)
    print("PYTORCH MLP UNIT TEST")
    print("="*80)

    # Create MLP
    mlp = create_pytorch_mlp(40, 32, 3)
    set_unit_weights_pytorch(mlp, 40, 32, 3)

    # Create test input (simulating blended features + view encoding)
    # All ones input, 256 samples
    x = torch.ones(256, 40, device='cuda', requires_grad=True)

    # Forward
    y = mlp(x)
    loss = y.sum()

    print(f"[FORWARD] Output[0]: {y[0].tolist()}")
    print(f"[FORWARD] Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Check gradients
    b1_grad = mlp[0].bias.grad
    b2_grad = mlp[2].bias.grad
    b3_grad = mlp[4].bias.grad

    b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
    b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()

    print(f"\n[GRADIENTS]")
    print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz}")
    print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz}")
    print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")
    print(f"  Expected: all at [0, 1, 2]")

    print(f"\n[VALUES]")
    print(f"  dL_db2 at [0,1,2]: {b2_grad[:3].tolist()}")
    print(f"  dL_db2 at [3,4,5]: {b2_grad[3:6].tolist()}")

    if b2_nz == [0, 1, 2]:
        print(f"\n*** PASS ***")
    else:
        print(f"\n*** UNEXPECTED: {b2_nz} ***")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Test 1: PyTorch MLP only (sanity check)
    test_pytorch_mlp_only()

    # Test 2: Standalone comparison (no rasterizer)
    print("\n\n")
    test_mlp_standalone()

    # Test 3: Through rasterizer
    print("\n\n")
    test_with_rasterizer()
