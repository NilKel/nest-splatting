#!/usr/bin/env python3
"""
Test transmittance chain with unit MLP matrices.

With identity/unit matrices, the gradient flow simplifies:
- Forward: input → output (identity mapping)
- Backward: dL/doutput → dL/dinput (same values)

This isolates the transmittance chain from MLP complexity.
"""

import os
import sys
import torch
import torch.nn.functional as F
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.config import Config
from hash_encoder.modules import INGP


def merge_cfg_to_args(args, cfg):
    """Merge config into args."""
    for section in ['training_cfg', 'settings', 'loss']:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    setattr(args, k, v)


def create_unit_mlp_weights(in_dim=40, hidden_dim=32, out_dim=3):
    """
    Create unit/identity-like MLP weights.

    For a 3-layer MLP: in(40) -> hidden1(32) -> hidden2(32) -> out(3)

    We can't have true identity (dims don't match), but we can:
    - W1: First 32 inputs pass through unchanged
    - W2: Identity 32x32
    - W3: First 3 hidden units pass through unchanged
    - All biases = 0

    This gives: output[i] = ReLU(ReLU(input[i])) for i < 3
    Since input is features (usually positive after hash), ReLU is identity.
    """
    # Layer 1: 40 -> 32 (take first 32 inputs)
    W1 = torch.zeros(hidden_dim, in_dim)
    for i in range(hidden_dim):
        W1[i, i] = 1.0  # W1[i,i] = 1 for i < 32
    b1 = torch.zeros(hidden_dim)

    # Layer 2: 32 -> 32 (identity)
    W2 = torch.eye(hidden_dim)
    b2 = torch.zeros(hidden_dim)

    # Layer 3: 32 -> 3 (take first 3)
    W3 = torch.zeros(out_dim, hidden_dim)
    for i in range(out_dim):
        W3[i, i] = 1.0  # W3[i,i] = 1 for i < 3
    b3 = torch.zeros(out_dim)

    return W1.cuda(), b1.cuda(), W2.cuda(), b2.cuda(), W3.cuda(), b3.cuda()


def main():
    # Model path
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    yaml_path = 'configs/nerfsyn.yaml'
    iteration = 30000

    if not os.path.exists(f'{model_path}/ngp_{iteration}.pth'):
        print(f"Model not found: {model_path}/ngp_{iteration}.pth")
        return

    print("=" * 60)
    print("UNIT MLP TRANSMITTANCE TEST")
    print("=" * 60)

    # Load config
    cfg_model = Config(yaml_path)

    # Base args
    source_path = '/home/nilkel/Projects/nest-splatting/data/nerf_synthetic/chair'
    base_args = Namespace(
        sh_degree=3,
        source_path=source_path,
        model_path=model_path,
        images='images',
        resolution=-1,
        white_background=True,
        data_device='cuda',
        eval=True,
        debug=False,
        scale_invariant=False,
        max_abs_split_scale=0.3,
        soft_beta=False,
        hybrid_levels=5,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        antialiasing=False,
    )
    merge_cfg_to_args(base_args, cfg_model)
    base_args.source_path = source_path

    # Load scene
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(base_args, gaussians, load_iteration=iteration, shuffle=False)
    camera = scene.getTrainCameras()[0]

    print(f"Loaded scene with {gaussians._xyz.shape[0]} Gaussians")

    # Lean mode
    args_lean = Namespace(**vars(base_args))
    args_lean.method = '3D_direct_lean'

    ingp_lean = INGP(cfg_model, args=args_lean).cuda()
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)

    # Create unit MLP weights and SET THEM IN THE MODEL
    # (render() calls ingp.get_fused_mlp_weights() internally)
    print("\nCreating unit MLP weights...")
    W1, b1, W2, b2, W3, b3 = create_unit_mlp_weights()
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  W3: {W3.shape}, b3: {b3.shape}")

    # Set weights directly in the INGP model's mlp_fused
    # This way render() will upload them via get_fused_mlp_weights()
    with torch.no_grad():
        ingp_lean.mlp_fused[0].weight.copy_(W1)  # Linear1
        ingp_lean.mlp_fused[0].bias.copy_(b1)
        ingp_lean.mlp_fused[2].weight.copy_(W2)  # Linear2
        ingp_lean.mlp_fused[2].bias.copy_(b2)
        ingp_lean.mlp_fused[4].weight.copy_(W3)  # Linear3
        ingp_lean.mlp_fused[4].bias.copy_(b3)
    print("  Set unit weights in ingp.mlp_fused (will be uploaded by render())")

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_lean.hash_encoding.embeddings.grad = None

    # Forward + backward
    print("\n" + "=" * 60)
    print("FORWARD PASS")
    print("=" * 60)

    # Verify unit weights are set before render
    print(f"  W1[0,0:3] before render: {ingp_lean.mlp_fused[0].weight[0, :3].tolist()}")

    render_pkg = render(camera, gaussians, args_lean, bg_color, ingp=ingp_lean, cfg=cfg_model)

    # Synchronize CUDA and flush
    torch.cuda.synchronize()
    print(f"  Render complete, synchronizing...")
    image = render_pkg['render']

    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"Render pkg keys: {list(render_pkg.keys())}")

    # Check if output makes sense
    # With unit MLP: output = sigmoid(input[0:3])
    # input[0:3] = per-gaussian features[0:3]
    # So image should be sigmoid of accumulated weighted features

    print("\n" + "=" * 60)
    print("BACKWARD PASS (loss = image.mean())")
    print("=" * 60)
    loss = image.mean()  # Use mean for normalized gradients
    loss.backward()

    print("\nGeometry Gradient Statistics:")
    def check_grad(name, tensor):
        if tensor.grad is None:
            print(f"  {name}: None")
            return None
        g = tensor.grad
        has_nan = torch.isnan(g).any().item()
        has_inf = torch.isinf(g).any().item()
        nonzero = (g.abs() > 1e-10).sum().item()
        print(f"  {name}: norm={g.norm():.4f}, mean={g.abs().mean():.6e}, max={g.abs().max():.6e}, "
              f"nonzero={nonzero}/{g.numel()}, NaN={has_nan}, Inf={has_inf}")
        return g

    grad_xyz = check_grad("xyz", gaussians._xyz)
    grad_scaling = check_grad("scaling", gaussians._scaling)
    grad_rotation = check_grad("rotation", gaussians._rotation)
    grad_opacity = check_grad("opacity", gaussians._opacity)
    grad_features = check_grad("features", gaussians._gaussian_features)
    grad_hash = check_grad("hash", ingp_lean.hash_encoding.embeddings)

    # Check MLP gradients
    print("\nMLP Gradients (should be simple with unit weights):")
    import diff_surfel_3D
    mlp_grads = diff_surfel_3D.get_mlp_grads()
    if mlp_grads is not None:
        names = ['dL_dW1', 'dL_db1', 'dL_dW2', 'dL_db2', 'dL_dW3', 'dL_db3']
        for name, grad in zip(names, mlp_grads):
            if grad is not None:
                has_nan = torch.isnan(grad).any().item()
                nonzero = (grad.abs() > 1e-10).sum().item()
                print(f"  {name}: norm={grad.norm():.4f}, mean={grad.abs().mean():.6e}, "
                      f"nonzero={nonzero}/{grad.numel()}, NaN={has_nan}")
                # Print which indices are nonzero for bias gradients
                if 'db' in name:
                    nonzero_idx = (grad.abs() > 1e-10).nonzero().squeeze().tolist()
                    print(f"    -> nonzero indices: {nonzero_idx}")
            else:
                print(f"  {name}: None")
    else:
        print("  get_mlp_grads() returned None")

    # Analyze the gradient flow
    print("\n" + "=" * 60)
    print("TRANSMITTANCE CHAIN ANALYSIS")
    print("=" * 60)

    # With unit MLP:
    # Forward: rgb[i] = sigmoid(feat[i]) for i < 3
    # Backward: dL/dfeat[i] = dL/drgb[i] * sigmoid'(feat[i])
    #
    # The transmittance chain should give:
    # dL/dxyz comes from how xyz affects the intersection position
    # dL/dscale comes from how scale affects the surfel size
    # dL/drotation comes from how rotation affects the surfel orientation
    # dL/dopacity comes from the alpha blending weights

    if grad_features is not None:
        # Check that only first 3 feature channels get gradients (with unit MLP)
        feat_grad_per_channel = grad_features.abs().mean(dim=0)
        print(f"\nFeature gradient per channel (first 10):")
        for i in range(min(10, len(feat_grad_per_channel))):
            print(f"  Channel {i}: {feat_grad_per_channel[i]:.6e}")

        # With unit MLP, only channels 0,1,2 should have non-zero gradients
        active_channels = (feat_grad_per_channel > 1e-10).sum().item()
        print(f"\nActive feature channels: {active_channels}/20")
        if active_channels <= 3:
            print("  PASS: Only first 3 channels have gradients (expected for unit MLP)")
        else:
            print("  NOTE: More than 3 channels have gradients")
            print("        This could indicate hash features are contributing")

    # Check opacity gradient magnitude vs others
    if grad_opacity is not None and grad_xyz is not None:
        ratio = grad_opacity.norm() / grad_xyz.norm()
        print(f"\nOpacity/XYZ gradient ratio: {ratio:.6f}")
        print("  (This indicates relative contribution of alpha vs geometry)")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
