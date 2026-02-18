#!/usr/bin/env python3
"""
Test MLP gradient flow with a single Gaussian from a trained checkpoint.

This tests the full rasterization pipeline to see if bias gradients
have the correct sparsity pattern (3/32 nonzeros at indices [0,1,2]
for unit weights, or the actual trained gradient pattern).
"""

import os
import sys
import json
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams

# Use the checkpoint path
MODEL_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
        return args

    raise FileNotFoundError(f"No training config found in {model_path}")


def keep_single_gaussian(gaussians, idx=0):
    """Keep only a single Gaussian at the specified index."""
    mask = torch.zeros(len(gaussians.get_xyz), dtype=torch.bool, device="cuda")
    mask[idx] = True

    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._appearance_level = gaussians._appearance_level[mask]

    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
        gaussians._gaussian_features = gaussians._gaussian_features[mask.to(gaussians._gaussian_features.device)]
    if hasattr(gaussians, '_adaptive_features') and gaussians._adaptive_features is not None and gaussians._adaptive_features.numel() > 0:
        gaussians._adaptive_features = gaussians._adaptive_features[mask.to(gaussians._adaptive_features.device)]
    if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        gaussians._shape = gaussians._shape[mask.to(gaussians._shape.device)]

    return gaussians


def find_visible_gaussian(gaussians, cameras, pipe, ingp, cfg_model, iteration, background):
    """Find a Gaussian that's visible in at least one camera."""
    beta = cfg_model.surfel.tg_beta
    num_gaussians = len(gaussians.get_xyz)

    # Try each Gaussian
    for idx in range(min(100, num_gaussians)):  # Check first 100
        # Create a temp copy with just this gaussian
        xyz = gaussians.get_xyz[idx:idx+1]
        opacity = gaussians.get_opacity[idx:idx+1]

        if opacity.item() < 0.1:
            continue

        # Check if it's visible in any camera
        for cam in cameras[:5]:  # Check first 5 cameras
            cam_pos = cam.camera_center
            # Simple distance check
            dist = torch.norm(xyz - cam_pos).item()
            if dist < 10.0:  # Within reasonable distance
                print(f"[SEARCH] Found Gaussian {idx} with opacity={opacity.item():.3f}, dist={dist:.2f}")
                return idx

    # If no good match, just return one with high opacity
    opacities = gaussians.get_opacity.squeeze()
    best_idx = torch.argmax(opacities).item()
    print(f"[SEARCH] Using Gaussian {best_idx} with highest opacity={opacities[best_idx].item():.3f}")
    return best_idx


def test_gradient_flow():
    """Test gradient flow through a single Gaussian."""
    print("="*80)
    print("SINGLE GAUSSIAN GRADIENT TEST")
    print("="*80)

    # Load training config
    print(f"\n[CONFIG] Loading from: {MODEL_PATH}")
    args = load_training_config(MODEL_PATH)
    args.model_path = MODEL_PATH
    args.eval = True

    # Load YAML config
    config_yaml_path = os.path.join(MODEL_PATH, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    print(f"[CONFIG] Method: {args.method}, Kernel: {args.kernel}")

    # Find iteration
    import glob
    ngp_files = glob.glob(os.path.join(MODEL_PATH, "ngp_*.pth"))
    if ngp_files:
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
        print(f"[CONFIG] Using iteration: {iteration}")
    else:
        raise FileNotFoundError("No checkpoints found")

    # Setup model/pipeline params
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(MODEL_PATH, iteration)

    # Load all Gaussians first
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    num_total = len(gaussians.get_xyz)
    print(f"[SETUP] Total Gaussians loaded: {num_total:,}")

    # Get cameras
    cameras = scene.getTestCameras()
    print(f"[SETUP] Test cameras: {len(cameras)}")

    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Find a good Gaussian to keep
    good_idx = find_visible_gaussian(gaussians, cameras, pipe, ingp_model, cfg_model, iteration, background)

    # Keep only that Gaussian
    print(f"\n[TEST] Keeping only Gaussian {good_idx}")
    keep_single_gaussian(gaussians, good_idx)
    print(f"[TEST] Now have {len(gaussians.get_xyz)} Gaussian(s)")

    # Print Gaussian properties
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    print(f"[GAUSS] Position: {xyz[0].tolist()}")
    print(f"[GAUSS] Opacity: {opacity[0].item():.4f}")
    print(f"[GAUSS] Scale: {scaling[0].tolist()}")

    # Get the right MLP based on method
    if hasattr(ingp_model, 'mlp_3D_direct') and ingp_model.mlp_3D_direct is not None:
        mlp = ingp_model.mlp_3D_direct
        mlp_name = "mlp_3D_direct"
    elif hasattr(ingp_model, 'mlp_rgb') and ingp_model.mlp_rgb is not None:
        mlp = ingp_model.mlp_rgb
        mlp_name = "mlp_rgb"
    else:
        raise RuntimeError("No MLP found!")

    print(f"[SETUP] Using MLP: {mlp_name}")

    # Enable gradients on MLP
    for param in mlp.parameters():
        param.requires_grad_(True)

    # Zero gradients
    mlp.zero_grad()

    # Make Gaussian bigger so it's definitely visible
    print(f"\n[TEST] Enlarging Gaussian scale to ensure visibility...")
    gaussians._scaling = gaussians._scaling + 2.0  # Make it bigger

    # Just use first camera
    best_cam = cameras[0]
    print(f"[TEST] Using camera {best_cam.image_name}")

    # Now do forward + backward with gradients
    mlp.zero_grad()

    render_pkg = render(best_cam, gaussians, pipe, background, ingp=ingp_model,
                       beta=beta, iteration=iteration, cfg=cfg_model)

    print(f"\n[DEBUG] render_pkg keys: {list(render_pkg.keys())}")

    rendered = render_pkg["render"]

    print(f"\n[FORWARD] Rendered image shape: {rendered.shape}")
    print(f"[FORWARD] Rendered mean RGB: {rendered.mean(dim=[1,2]).tolist()}")
    print(f"[FORWARD] Rendered max RGB: {rendered.max().item():.4f}")

    if "render_alpha" in render_pkg:
        render_alpha = render_pkg["render_alpha"]
        print(f"[FORWARD] Alpha sum: {render_alpha.sum().item():.4f}")
    elif "rend_alpha" in render_pkg:
        render_alpha = render_pkg["rend_alpha"]
        print(f"[FORWARD] Alpha sum: {render_alpha.sum().item():.4f}")
    else:
        print(f"[FORWARD] render_alpha not in output")

    # Check for NaN/Inf in forward
    if torch.isnan(rendered).any():
        print(f"[WARN] NaN in rendered image!")
    if torch.isinf(rendered).any():
        print(f"[WARN] Inf in rendered image!")

    # Backward with unit loss on RGB (only visible pixels)
    # Use mean instead of sum to avoid gradient explosion
    visible_mask = rendered.abs() > 0
    if visible_mask.any():
        loss = rendered[visible_mask].mean()
    else:
        loss = rendered.mean()
    print(f"[BACKWARD] Loss (mean of visible RGB): {loss.item():.6f}")

    loss.backward()

    # Check for NaN/Inf in gradients
    if hasattr(mlp, 'params') and mlp.params.grad is not None:
        grad = mlp.params.grad
        if torch.isnan(grad).any():
            print(f"[WARN] NaN in gradients!")
            nan_count = torch.isnan(grad).sum().item()
            print(f"  NaN count: {nan_count}/{len(grad)}")
        if torch.isinf(grad).any():
            print(f"[WARN] Inf in gradients!")
            inf_count = torch.isinf(grad).sum().item()
            print(f"  Inf count: {inf_count}/{len(grad)}")

    # Check MLP gradients
    # For tcnn networks, we need to check params.grad directly
    print(f"\n[GRADIENTS] MLP type: {type(mlp)}")

    if hasattr(mlp, 'params'):
        # tcnn Network - check params directly
        if mlp.params.grad is not None:
            grad = mlp.params.grad
            nz_mask = grad.abs() > 1e-10
            nz_count = nz_mask.sum().item()
            total = len(grad)
            print(f"[GRADIENTS] params grad: {nz_count}/{total} nonzeros")
            print(f"  grad norm: {grad.norm().item():.6f}")
            print(f"  grad mean: {grad.mean().item():.6f}")
            print(f"  grad max: {grad.max().item():.6f}")
            print(f"  grad min: {grad.min().item():.6f}")

            # For tcnn, weights are packed: W1 | b1 | W2 | b2 | W3 | b3
            # Dimensions depend on architecture
            # Let's decode based on architecture: 40->256->256->3 (from cat mode config)
            print(f"\n[ANALYSIS] Decoding tcnn weight layout:")
            print(f"  Total params: {total}")

            # tcnn packs weights in row-major, biases after each layer
            # Architecture: 40D -> 256 -> 256 -> 3D
            # W1: 40*256=10240, b1: 256
            # W2: 256*256=65536, b2: 256
            # W3: 256*3=768, b3: 3
            # But tcnn pads to 16 multiples and may not have biases

            # Try to find bias gradients by looking at the gradient pattern
            # First, let's just print which indices have nonzero grads
            nz_indices = nz_mask.nonzero().squeeze(-1)
            if len(nz_indices) > 0:
                print(f"  First 20 nonzero indices: {nz_indices[:20].tolist()}")
                print(f"  Last 20 nonzero indices: {nz_indices[-20:].tolist()}")
        else:
            print(f"[GRADIENTS] params grad is None!")
    else:
        # Regular PyTorch module
        print(f"[GRADIENTS] MLP bias gradients:")
        for i, layer in enumerate(mlp):
            if hasattr(layer, 'bias') and layer.bias is not None:
                if layer.bias.grad is not None:
                    grad = layer.bias.grad
                    nz_mask = grad.abs() > 1e-10
                    nz_indices = nz_mask.nonzero().squeeze(-1).tolist()
                    nz_count = nz_mask.sum().item()
                    total = len(grad)

                    print(f"  Layer {i} bias: {nz_count}/{total} nonzeros at {nz_indices[:10]}...")
                    if nz_count > 0:
                        nz_vals = grad[nz_mask].tolist()[:5]
                        print(f"    Values: {nz_vals}...")
                else:
                    print(f"  Layer {i} bias: grad is None")

    print("\n" + "="*80)


def test_with_unit_weights():
    """Test with unit weights to verify gradient sparsity."""
    print("="*80)
    print("UNIT WEIGHTS GRADIENT TEST")
    print("="*80)

    # Load training config
    print(f"\n[CONFIG] Loading from: {MODEL_PATH}")
    args = load_training_config(MODEL_PATH)
    args.model_path = MODEL_PATH
    args.eval = True

    # Load YAML config
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

    # Setup model/pipeline params
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(MODEL_PATH, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Get cameras
    cameras = scene.getTestCameras()
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Keep first Gaussian with high opacity
    opacities = gaussians.get_opacity.squeeze()
    good_idx = torch.argmax(opacities).item()
    keep_single_gaussian(gaussians, good_idx)

    # Make Gaussian bigger so it's definitely visible
    gaussians._scaling = gaussians._scaling + 1.0

    print(f"[SETUP] Using Gaussian with opacity={opacities[good_idx].item():.3f}")

    # Set unit weights on MLP
    print(f"\n[SETUP] Setting MLP to unit weights...")
    with torch.no_grad():
        mlp = ingp_model.mlp
        # Assuming structure: Linear(40,32), ReLU, Linear(32,32), ReLU, Linear(32,3), Sigmoid

        # Layer 0: 40D -> 32D (take first 32 inputs)
        if hasattr(mlp[0], 'weight'):
            mlp[0].weight.zero_()
            mlp[0].bias.zero_()
            for i in range(min(32, mlp[0].weight.shape[0])):
                if i < mlp[0].weight.shape[1]:
                    mlp[0].weight[i, i] = 1.0
            print(f"  Layer 0: {mlp[0].weight.shape} - diagonal unit")

        # Layer 2: 32D -> 32D (identity)
        if hasattr(mlp[2], 'weight'):
            mlp[2].weight.zero_()
            mlp[2].bias.zero_()
            for i in range(min(mlp[2].weight.shape[0], mlp[2].weight.shape[1])):
                mlp[2].weight[i, i] = 1.0
            print(f"  Layer 2: {mlp[2].weight.shape} - identity")

        # Layer 4: 32D -> 3D (take first 3 hidden)
        if hasattr(mlp[4], 'weight'):
            mlp[4].weight.zero_()
            mlp[4].bias.zero_()
            for i in range(min(3, mlp[4].weight.shape[0])):
                if i < mlp[4].weight.shape[1]:
                    mlp[4].weight[i, i] = 1.0
            print(f"  Layer 4: {mlp[4].weight.shape} - diagonal unit")

    # Enable gradients
    for param in ingp_model.mlp.parameters():
        param.requires_grad_(True)
    ingp_model.mlp.zero_grad()

    # Upload to CUDA if using fused mode
    if args.method in ["3D_direct_fused", "3D_fused"]:
        from diff_surfel_3D import set_mlp_weights
        set_mlp_weights(
            mlp[0].weight, mlp[0].bias,
            mlp[2].weight, mlp[2].bias,
            mlp[4].weight, mlp[4].bias,
            is_sh_mode=False
        )
        print(f"[SETUP] Uploaded unit weights to CUDA constant memory")

    # Find a camera where Gaussian is visible
    best_cam = cameras[0]

    # Render
    render_pkg = render(best_cam, gaussians, pipe, background, ingp=ingp_model,
                       beta=beta, iteration=iteration, cfg=cfg_model)

    rendered = render_pkg["render"]
    render_alpha = render_pkg["render_alpha"]

    print(f"\n[FORWARD] Alpha sum: {render_alpha.sum().item():.4f}")
    print(f"[FORWARD] Rendered mean: {rendered.mean(dim=[1,2]).tolist()}")

    # Backward
    loss = rendered.sum()
    loss.backward()

    # Check b2 gradient
    print(f"\n[GRADIENTS] b2 analysis with unit weights:")

    if args.method in ["3D_direct_fused", "3D_fused"]:
        # Get gradients from CUDA
        from diff_surfel_3D import get_mlp_grads
        grads = get_mlp_grads()
        dL_db2 = grads['dL_db2']
    else:
        # Get from PyTorch
        dL_db2 = mlp[2].bias.grad

    if dL_db2 is not None:
        nz_mask = dL_db2.abs() > 1e-10
        nz_indices = nz_mask.nonzero().squeeze(-1).tolist()

        print(f"  dL_db2 nonzeros: {len(nz_indices)}/32")
        print(f"  Nonzero indices: {nz_indices}")
        print(f"  Expected: [0, 1, 2]")

        if nz_indices == [0, 1, 2]:
            print(f"\n  *** PASS ***")
        elif len(nz_indices) == 16:
            print(f"\n  *** FAIL: 16/32 bug ***")
        else:
            print(f"\n  *** {len(nz_indices)}/32 ***")
    else:
        print(f"  No gradient!")

    print("\n" + "="*80)


if __name__ == "__main__":
    # First test with trained weights
    test_gradient_flow()

    # Skip unit weights test for now since tcnn doesn't allow easy weight modification
    # print("\n\n")
    # test_with_unit_weights()
