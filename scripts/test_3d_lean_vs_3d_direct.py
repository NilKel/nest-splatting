#!/usr/bin/env python3
"""
Compare 3D_lean (custom CUDA MLP) against 3D_direct (tcnn MLP).
Run the same scene with both modes and compare outputs/gradients.

REFERENCE TEST for debugging 3D_lean/3D_direct_fused backward pass.
See docs/DEBUG_3D_LEAN_VS_3D_DIRECT.md for full documentation.

Usage:
    python scripts/test_3d_lean_vs_3d_direct.py
    python scripts/test_3d_lean_vs_3d_direct.py --no_unit_weights
    python scripts/test_3d_lean_vs_3d_direct.py --num_gaussians 1000
"""

import os
import sys
import glob
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

# Default model path - 3D_direct trained checkpoint
MODEL_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    # Try to load args.pkl (exact reproduction)
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    # Fallback to args.json
    import json
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
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


def set_unit_weights(mlp, input_dim=40, hidden_dim=32, output_dim=3):
    """
    Set unit/diagonal weights on PyTorch MLP for gradient testing.

    With these weights:
    - h1[i] = relu(x[i]) for i < 32
    - h2[i] = relu(h1[i]) = relu(relu(x[i]))
    - out[i] = sigmoid(h2[i]) for i < 3

    Gradients should only flow through indices 0, 1, 2.
    """
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


def compare_3d_modes(model_path=MODEL_PATH, use_unit_weights=True, num_gaussians=None, use_fp16=False):
    """
    Compare 3D_direct (tcnn) vs 3D_lean/fp16 (custom CUDA MLP).

    Args:
        model_path: Path to trained 3D_direct checkpoint
        use_unit_weights: If True, use unit/diagonal weights for gradient testing
        num_gaussians: Number of Gaussians to keep (for faster testing)
        use_fp16: If True, use diff_surfel_3D_16 (FP16 weights) instead of diff_surfel_3D
    """
    print("=" * 80)
    print("COMPARING: 3D_direct (tcnn) vs 3D_lean (CUDA MLP)")
    print("=" * 80)
    print(f"  Model path: {model_path}")
    print(f"  Unit weights: {use_unit_weights}")
    print(f"  Num Gaussians: {num_gaussians}")

    # Load training config (following eval_from_checkpoint.py pattern)
    print(f"\n[LOAD] Loading config from: {model_path}")
    args = load_training_config(model_path)

    # Override model_path to the actual directory
    args.model_path = model_path
    args.eval = True  # Force eval=True to load test cameras

    # Load YAML config
    config_yaml_path = os.path.join(model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
        print(f"[CONFIG] Loaded config from {config_yaml_path}")
    else:
        cfg_model = Config(args.yaml)
        print(f"[CONFIG] Loaded config from {args.yaml}")

    # Print key training parameters
    print(f"\n[CONFIG] Training parameters:")
    print(f"  Method: {args.method}")
    print(f"  Hybrid levels: {getattr(args, 'hybrid_levels', 'N/A')}")

    # Find iteration
    iteration = -1
    ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
    if ngp_files:
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
        print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
    else:
        raise FileNotFoundError(f"No ngp_*.pth checkpoints found in {model_path}")

    # Create parser for ModelParams/PipelineParams extraction
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"

    # Set kernel type
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Optionally keep subset of Gaussians for faster testing
    if num_gaussians is not None and num_gaussians < len(gaussians.get_xyz):
        print(f"\n[SETUP] Keeping {num_gaussians} Gaussians with highest opacity")
        opacities = gaussians.get_opacity.squeeze()
        _, top_indices = torch.topk(opacities, min(num_gaussians, len(opacities)))
        mask = torch.zeros(len(opacities), dtype=torch.bool, device="cuda")
        mask[top_indices] = True

        gaussians._xyz = nn.Parameter(gaussians._xyz[mask])
        gaussians._features_dc = nn.Parameter(gaussians._features_dc[mask])
        gaussians._features_rest = nn.Parameter(gaussians._features_rest[mask])
        gaussians._scaling = nn.Parameter(gaussians._scaling[mask])
        gaussians._rotation = nn.Parameter(gaussians._rotation[mask])
        gaussians._opacity = nn.Parameter(gaussians._opacity[mask])
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
            gaussians._gaussian_features = nn.Parameter(gaussians._gaussian_features[mask])
        if hasattr(gaussians, '_appearance_level') and gaussians._appearance_level is not None:
            gaussians._appearance_level = nn.Parameter(gaussians._appearance_level[mask])
    else:
        print(f"\n[SETUP] Using all {len(gaussians.get_xyz)} Gaussians")

    cameras = scene.getTestCameras()
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    if use_unit_weights:
        print("\n[WEIGHTS] Using unit weights for gradient testing:")
        print(f"  W1: diagonal identity (32x40)")
        print(f"  W2: identity (32x32)")
        print(f"  W3: diagonal [1,1,1] in first 3 cols")
        print(f"  Expected gradient sparsity: dL_db1, dL_db2, dL_db3 at [0,1,2] only")

    # ========== TEST 1: 3D_direct (tcnn) ==========
    print("\n" + "=" * 80)
    print("TEST 1: 3D_direct mode (tcnn)")
    print("=" * 80)

    # Create and load INGP for 3D_direct
    ingp_direct = INGP(cfg_model, args=args).to('cuda')
    ingp_direct.load_model(model_path, iteration)
    ingp_direct.set_active_levels(iteration)

    args_3d_direct = Namespace(**vars(args))
    args_3d_direct.method = "3D_direct"

    # Apply unit weights if requested - MUST be done for BOTH modes for fair comparison
    if use_unit_weights and hasattr(ingp_direct, 'mlp_3D_direct') and ingp_direct.mlp_3D_direct is not None:
        print("[SETUP] Setting unit weights on ingp_direct.mlp_3D_direct...")
        set_unit_weights(ingp_direct.mlp_3D_direct, 40, 32, 3)
        print(f"  W3[0,:5]: {ingp_direct.mlp_3D_direct[4].weight[0,:5].tolist()}")
        print(f"  W3[1,:5]: {ingp_direct.mlp_3D_direct[4].weight[1,:5].tolist()}")
        print(f"  W3[2,:5]: {ingp_direct.mlp_3D_direct[4].weight[2,:5].tolist()}")

    print("[INFO] Running 3D_direct render...")

    try:
        # Reset gradients
        gaussians._xyz.requires_grad_(True)
        gaussians._xyz.grad = None
        for param in ingp_direct.parameters():
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()

        result_3d_direct = render(
            cameras[0], gaussians, args_3d_direct, background,
            ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model
        )
        render_3d_direct = result_3d_direct["render"].clone()

        print(f"  Output shape: {render_3d_direct.shape}")
        print(f"  Output range: [{render_3d_direct.min().item():.4f}, {render_3d_direct.max().item():.4f}]")
        print(f"  Output mean: {render_3d_direct.mean().item():.4f}")

        # DEBUG: Manually compute gradients for first intersection to compare with CUDA
        # Get intersection data from render result
        if 'intersection_buffer' in result_3d_direct:
            ib = result_3d_direct['intersection_buffer']
            valid_mask = ib[:, 0] >= 0  # gaussian_id >= 0 means valid
            valid_ib = ib[valid_mask]
            if len(valid_ib) > 0:
                # First intersection
                first = valid_ib[0]
                w_first = first[1].item()  # weight = alpha * T
                print(f"\n  [DEBUG PYTORCH] First intersection:")
                print(f"    weight (w): {w_first:.6f}")
                # For loss.sum(), dL/d(pixel) = 1.0
                # dL/d(rgb) = dL/d(pixel) * w = 1.0 * w = w
                print(f"    dL_drgb (1.0 * w): [{w_first:.6f}, {w_first:.6f}, {w_first:.6f}]")

        # Backward
        loss_3d_direct = render_3d_direct.sum()
        loss_3d_direct.backward()

        grad_xyz_3d_direct = gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None
        print(f"  grad_xyz mean: {grad_xyz_3d_direct.abs().mean().item() if grad_xyz_3d_direct is not None else 'None'}")

        # Check hash gradients
        if ingp_direct.hash_encoding.embeddings.grad is not None:
            hash_grad = ingp_direct.hash_encoding.embeddings.grad
            hash_nz = (hash_grad.abs() > 1e-10).sum().item()
            print(f"  hash_grad nonzero: {hash_nz}")
        else:
            print(f"  hash_grad: None")

        # Check MLP gradients for 3D_direct (PyTorch)
        grads_3d_direct = {}
        if hasattr(ingp_direct, 'mlp_3D_direct') and ingp_direct.mlp_3D_direct is not None:
            # PyTorch Sequential MLP - check individual layer gradients
            mlp = ingp_direct.mlp_3D_direct
            if mlp[0].weight.grad is not None:
                grads_3d_direct['W1'] = mlp[0].weight.grad.clone()
                grads_3d_direct['W2'] = mlp[2].weight.grad.clone()
                grads_3d_direct['W3'] = mlp[4].weight.grad.clone()
                grads_3d_direct['b1'] = mlp[0].bias.grad.clone()
                grads_3d_direct['b2'] = mlp[2].bias.grad.clone()
                grads_3d_direct['b3'] = mlp[4].bias.grad.clone()
                print(f"  mlp_3D_direct W1 grad: shape={grads_3d_direct['W1'].shape}, mean={grads_3d_direct['W1'].abs().mean().item():.6f}")
                print(f"  mlp_3D_direct W2 grad: shape={grads_3d_direct['W2'].shape}, mean={grads_3d_direct['W2'].abs().mean().item():.6f}")
                print(f"  mlp_3D_direct W3 grad: shape={grads_3d_direct['W3'].shape}, mean={grads_3d_direct['W3'].abs().mean().item():.6f}")
                print(f"  mlp_3D_direct b1 grad: mean={grads_3d_direct['b1'].abs().mean().item():.6f}")
                print(f"  mlp_3D_direct b2 grad: mean={grads_3d_direct['b2'].abs().mean().item():.6f}")
                print(f"  mlp_3D_direct b3 grad: mean={grads_3d_direct['b3'].abs().mean().item():.6f}")

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        render_3d_direct = None
        grad_xyz_3d_direct = None

    # ========== TEST 2: 3D_lean (custom CUDA MLP) ==========
    lean_method = "3D_direct_fp16" if use_fp16 else "3D_direct_lean"
    lean_label = "3D_fp16 (FP16 CUDA MLP)" if use_fp16 else "3D_lean (CUDA MLP)"
    print("\n" + "=" * 80)
    print(f"TEST 2: {lean_label}")
    print("=" * 80)

    # Create args for 3D_direct_lean/fp16 - must set method BEFORE creating INGP
    # so that INGP constructor sets the right mode flag
    args_3d_lean = Namespace(**vars(args))
    args_3d_lean.method = lean_method

    # Create new INGP for 3D_lean with the right args
    ingp_lean = INGP(cfg_model, args=args_3d_lean).to('cuda')
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)

    print(f"[SETUP] INGP created with method={args_3d_lean.method}")
    print(f"[SETUP] ingp_lean.is_3D_direct_fused_mode={ingp_lean.is_3D_direct_fused_mode}")
    print(f"[SETUP] ingp_lean.is_3D_direct_lean_mode={ingp_lean.is_3D_direct_lean_mode}")
    print(f"[SETUP] ingp_lean.is_3D_direct_fp16_mode={getattr(ingp_lean, 'is_3D_direct_fp16_mode', False)}")
    print(f"[SETUP] ingp_lean.hybrid_levels={ingp_lean.hybrid_levels}")
    print(f"[SETUP] ingp_lean.levels={ingp_lean.levels}")

    # Create mlp_fused for lean mode
    hidden_dim = 32
    if ingp_lean.mlp_fused is None:
        print("[SETUP] Creating mlp_fused...")
        ingp_lean.mlp_fused = create_pytorch_mlp(40, hidden_dim, 3)

    if use_unit_weights:
        print("[SETUP] Setting unit weights on mlp_fused...")
        set_unit_weights(ingp_lean.mlp_fused, 40, hidden_dim, 3)

    print(f"  mlp_fused[4].weight (W3):")
    print(f"    W3[0,:5]: {ingp_lean.mlp_fused[4].weight[0,:5].tolist()}")
    print(f"    W3[1,:5]: {ingp_lean.mlp_fused[4].weight[1,:5].tolist()}")
    print(f"    W3[2,:5]: {ingp_lean.mlp_fused[4].weight[2,:5].tolist()}")

    print("[INFO] Running 3D_lean render...")

    try:
        # Reset gradients
        gaussians._xyz.grad = None
        for param in ingp_lean.parameters():
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()
        for param in ingp_lean.mlp_fused.parameters():
            param.requires_grad_(True)
            param.grad = None

        result_3d_lean = render(
            cameras[0], gaussians, args_3d_lean, background,
            ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model
        )
        render_3d_lean = result_3d_lean["render"].clone()

        print(f"  Output shape: {render_3d_lean.shape}")
        print(f"  Output range: [{render_3d_lean.min().item():.4f}, {render_3d_lean.max().item():.4f}]")
        print(f"  Output mean: {render_3d_lean.mean().item():.4f}")

        # Backward
        loss_3d_lean = render_3d_lean.sum()
        loss_3d_lean.backward()

        grad_xyz_3d_lean = gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None
        print(f"  grad_xyz mean: {grad_xyz_3d_lean.abs().mean().item() if grad_xyz_3d_lean is not None else 'None'}")

        # CRITICAL: Retrieve MLP gradients from CUDA and copy to PyTorch tensors
        # This is done in train.py after backward() - see lines 1080-1110
        if use_fp16:
            from diff_surfel_3D_16 import get_mlp_grads
        else:
            from diff_surfel_3D import get_mlp_grads
        mlp_grads = get_mlp_grads()
        print(f"\n  [CUDA GRADS] get_mlp_grads() returned: {mlp_grads is not None}")
        if mlp_grads is not None:
            if use_fp16:
                # FP16 mode: bias-free, returns 3 values (W1, W2, W3)
                grad_W1, grad_W2, grad_W3 = mlp_grads
                grad_b1, grad_b2, grad_b3 = None, None, None
            else:
                grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = mlp_grads
            print(f"    grad_W1: {grad_W1.shape if grad_W1 is not None else 'None'}, nonzero={((grad_W1.abs() > 1e-10).sum().item() if grad_W1 is not None else 0)}")
            print(f"    grad_b1: {grad_b1.shape if grad_b1 is not None else 'None'}, nonzero={((grad_b1.abs() > 1e-10).sum().item() if grad_b1 is not None else 0)}")
            print(f"    grad_W2: {grad_W2.shape if grad_W2 is not None else 'None'}, nonzero={((grad_W2.abs() > 1e-10).sum().item() if grad_W2 is not None else 0)}")
            print(f"    grad_b2: {grad_b2.shape if grad_b2 is not None else 'None'}, nonzero={((grad_b2.abs() > 1e-10).sum().item() if grad_b2 is not None else 0)}")
            print(f"    grad_W3: {grad_W3.shape if grad_W3 is not None else 'None'}, nonzero={((grad_W3.abs() > 1e-10).sum().item() if grad_W3 is not None else 0)}")
            print(f"    grad_b3: {grad_b3.shape if grad_b3 is not None else 'None'}, nonzero={((grad_b3.abs() > 1e-10).sum().item() if grad_b3 is not None else 0)}")

            # Copy to PyTorch tensors (as train.py does)
            mlp = ingp_lean.mlp_fused
            if grad_W1 is not None:
                mlp[0].weight.grad = grad_W1.clone()
            if grad_b1 is not None and hasattr(mlp[0], 'bias') and mlp[0].bias is not None:
                mlp[0].bias.grad = grad_b1.clone()
            if grad_W2 is not None:
                mlp[2].weight.grad = grad_W2.clone()
            if grad_b2 is not None and hasattr(mlp[2], 'bias') and mlp[2].bias is not None:
                mlp[2].bias.grad = grad_b2.clone()
            if grad_W3 is not None:
                mlp[4].weight.grad = grad_W3.clone()
            if grad_b3 is not None and hasattr(mlp[4], 'bias') and mlp[4].bias is not None:
                mlp[4].bias.grad = grad_b3.clone()
        else:
            print("    [WARNING] get_mlp_grads() returned None!")

        # Check MLP bias gradients (only for biased MLPs, not FP16 bias-free)
        b1_grad = ingp_lean.mlp_fused[0].bias.grad if hasattr(ingp_lean.mlp_fused[0], 'bias') and ingp_lean.mlp_fused[0].bias is not None else None
        b2_grad = ingp_lean.mlp_fused[2].bias.grad if hasattr(ingp_lean.mlp_fused[2], 'bias') and ingp_lean.mlp_fused[2].bias is not None else None
        b3_grad = ingp_lean.mlp_fused[4].bias.grad if hasattr(ingp_lean.mlp_fused[4], 'bias') and ingp_lean.mlp_fused[4].bias is not None else None

        if b1_grad is not None:
            b1_nz = (b1_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
            if isinstance(b1_nz, int):
                b1_nz = [b1_nz]
            print(f"  dL_db1 nonzeros: {len(b1_nz)}/32 at {b1_nz[:10]}{'...' if len(b1_nz) > 10 else ''}")
        else:
            print(f"  dL_db1: None")

        if b2_grad is not None:
            b2_nz = (b2_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
            if isinstance(b2_nz, int):
                b2_nz = [b2_nz]
            print(f"  dL_db2 nonzeros: {len(b2_nz)}/32 at {b2_nz[:10]}{'...' if len(b2_nz) > 10 else ''}")
        else:
            print(f"  dL_db2: None")

        if b3_grad is not None:
            b3_nz = (b3_grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
            if isinstance(b3_nz, int):
                b3_nz = [b3_nz]
            print(f"  dL_db3 nonzeros: {len(b3_nz)}/3 at {b3_nz}")
        else:
            print(f"  dL_db3: None")

        if use_unit_weights:
            print(f"\n  EXPECTED (unit weights): dL_db1, dL_db2, dL_db3 all at [0, 1, 2] only")
            print(f"  NOTE: This assumes ALL inputs are POSITIVE. Negative inputs get zeroed by ReLU!")

            # Only diagnose as bug if using unit weights AND inputs should be positive
            if b2_grad is not None and len(b2_nz) > 3:
                print(f"\n  [INFO] dL_db2 has {len(b2_nz)} nonzeros - this depends on which inputs are positive")
                print(f"    With trained features that may be negative, more neurons can be active")

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        render_3d_lean = None
        grad_xyz_3d_lean = None

    # ========== SAVE IMAGES ==========
    import torchvision
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if render_3d_direct is not None:
        torchvision.utils.save_image(render_3d_direct, os.path.join(project_root, "fw_3d_direct.png"))
        print(f"  Saved: {os.path.join(project_root, 'fw_3d_direct.png')}")
    if render_3d_lean is not None:
        torchvision.utils.save_image(render_3d_lean, os.path.join(project_root, "fw_3d_lean.png"))
        print(f"  Saved: {os.path.join(project_root, 'fw_3d_lean.png')}")
    if render_3d_direct is not None and render_3d_lean is not None:
        diff_img = (render_3d_direct - render_3d_lean).abs()
        # Scale diff by 10x for visibility
        torchvision.utils.save_image(diff_img * 10, os.path.join(project_root, "fw_diff_10x.png"))
        print(f"  Saved: {os.path.join(project_root, 'fw_diff_10x.png')}")

    # Also save GT if available
    gt = cameras[0].original_image.to("cuda")
    if gt is not None:
        torchvision.utils.save_image(gt, os.path.join(project_root, "fw_gt.png"))
        print(f"  Saved: {os.path.join(project_root, 'fw_gt.png')}")

    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if render_3d_direct is not None and render_3d_lean is not None:
        diff = (render_3d_direct - render_3d_lean).abs()
        print(f"  Render diff (abs): mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

        if diff.max().item() < 0.01:
            print("  [OK] Forward pass outputs SIMILAR (within 0.01)")
        else:
            print("  [MISMATCH] Forward pass outputs differ significantly!")
            # Find where they differ most
            max_idx = diff.argmax()
            c = max_idx // (diff.shape[1] * diff.shape[2])
            rem = max_idx % (diff.shape[1] * diff.shape[2])
            h = rem // diff.shape[2]
            w = rem % diff.shape[2]
            print(f"    Max diff at channel={c.item()}, pixel=({h.item()},{w.item()})")
            print(f"    3D_direct: {render_3d_direct[:, h, w].tolist()}")
            print(f"    3D_lean: {render_3d_lean[:, h, w].tolist()}")

    if grad_xyz_3d_direct is not None and grad_xyz_3d_lean is not None:
        grad_diff = (grad_xyz_3d_direct - grad_xyz_3d_lean).abs()
        print(f"  grad_xyz diff (abs): mean={grad_diff.mean().item():.6f}, max={grad_diff.max().item():.6f}")

    # Compare MLP gradients between 3D_direct and 3D_lean
    if grads_3d_direct and ingp_lean.mlp_fused[0].weight.grad is not None:
        print("\n  MLP Gradient Comparison (3D_direct vs 3D_lean):")
        lean_mlp = ingp_lean.mlp_fused
        bias_free = not (hasattr(lean_mlp[0], 'bias') and lean_mlp[0].bias is not None)
        for name, ref_grad in grads_3d_direct.items():
            if name == 'W1':
                lean_grad = lean_mlp[0].weight.grad
                if bias_free and lean_grad is not None:
                    # FP16 W1 is [32, 41], ref is [32, 40] — compare first 40 cols
                    lean_grad = lean_grad[:, :40]
            elif name == 'b1':
                if bias_free:
                    # Implicit bias grad is column 40 of W1
                    lean_grad = lean_mlp[0].weight.grad[:, 40] if lean_mlp[0].weight.grad is not None else None
                else:
                    lean_grad = lean_mlp[0].bias.grad if hasattr(lean_mlp[0], 'bias') and lean_mlp[0].bias is not None else None
            elif name == 'W2':
                lean_grad = lean_mlp[2].weight.grad
            elif name == 'W3':
                lean_grad = lean_mlp[4].weight.grad
            elif name == 'b2':
                lean_grad = lean_mlp[2].bias.grad if hasattr(lean_mlp[2], 'bias') and lean_mlp[2].bias is not None else None
            elif name == 'b3':
                lean_grad = lean_mlp[4].bias.grad if hasattr(lean_mlp[4], 'bias') and lean_mlp[4].bias is not None else None
            else:
                continue

            if lean_grad is not None:
                diff = (ref_grad - lean_grad).abs()
                cos_sim = torch.nn.functional.cosine_similarity(
                    ref_grad.flatten().unsqueeze(0),
                    lean_grad.flatten().unsqueeze(0)
                ).item()
                print(f"    {name}: diff mean={diff.mean().item():.6f}, max={diff.max().item():.6f}, cos_sim={cos_sim:.6f}")
                print(f"      ref mag={ref_grad.abs().mean().item():.6f}, lean mag={lean_grad.abs().mean().item():.6f}")
                if diff.max().item() > 0.1 or cos_sim < 0.99:
                    print(f"      [MISMATCH] Significant gradient difference!")
            elif name in ('b2', 'b3') and bias_free:
                print(f"    {name}: N/A (bias-free mode, no separate bias gradient)")
            else:
                print(f"    {name}: lean_grad is None")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--no_unit_weights", action="store_true", help="Use trained weights instead of unit weights")
    parser.add_argument("--num_gaussians", type=int, default=None, help="Limit Gaussians (None = use all)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 library (diff_surfel_3D_16)")
    cli_args = parser.parse_args()

    compare_3d_modes(
        model_path=cli_args.model_path,
        use_unit_weights=False,  # Always use trained weights
        num_gaussians=cli_args.num_gaussians,
        use_fp16=cli_args.fp16
    )
