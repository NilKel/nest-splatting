#!/usr/bin/env python3
"""
Full gradient comparison: 3D_direct vs 3D_direct_lean

Compares ALL gradients:
1. MLP weights (W1, W2, W3, b1, b2, b3)
2. Per-Gaussian features (gaussian_features)
3. Hash table features (hash_encoding.embeddings)
4. Gaussian geometry (xyz, scales, rotations, opacity)
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

MODEL_PATH = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"


def load_training_config(model_path):
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    raise FileNotFoundError(f"No training config found in {model_path}")


def compare_tensors(name, t1, t2, rtol=1e-3, atol=1e-5):
    """Compare two tensors and print detailed comparison."""
    if t1 is None and t2 is None:
        print(f"  {name}: Both None")
        return True
    if t1 is None or t2 is None:
        print(f"  {name}: [FAIL] One is None: t1={t1 is not None}, t2={t2 is not None}")
        return False

    if t1.shape != t2.shape:
        print(f"  {name}: [FAIL] Shape mismatch: {t1.shape} vs {t2.shape}")
        return False

    diff = (t1 - t2).abs()
    t1_mag = t1.abs().mean().item()
    t2_mag = t2.abs().mean().item()

    # Cosine similarity
    if t1.numel() > 0 and t1.abs().sum() > 1e-10 and t2.abs().sum() > 1e-10:
        cos_sim = torch.nn.functional.cosine_similarity(
            t1.flatten().unsqueeze(0).float(),
            t2.flatten().unsqueeze(0).float()
        ).item()
    else:
        cos_sim = 0.0

    # Relative error
    rel_err = diff.mean().item() / max(t1_mag, t2_mag, 1e-10)

    status = "[OK]" if cos_sim > 0.999 and rel_err < 0.05 else "[MISMATCH]"

    print(f"  {name}: {status}")
    print(f"      shape: {list(t1.shape)}")
    print(f"      3D_direct mean: {t1_mag:.6f}, 3D_lean mean: {t2_mag:.6f}")
    print(f"      diff: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")
    print(f"      cos_sim: {cos_sim:.6f}, rel_err: {rel_err:.4f}")

    return cos_sim > 0.999


def run_comparison(model_path=MODEL_PATH, num_gaussians=None):
    print("=" * 80)
    print("FULL GRADIENT COMPARISON: 3D_direct vs 3D_direct_lean")
    print("=" * 80)

    # Load config
    args = load_training_config(model_path)
    args.model_path = model_path
    args.eval = True

    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg_model = Config(config_yaml_path)

    # Find iteration
    ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
    iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
    iteration = max(iterations)
    print(f"Using iteration: {iteration}")

    # Setup
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
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Optionally limit Gaussians
    if num_gaussians is not None and num_gaussians < len(gaussians.get_xyz):
        print(f"Keeping top {num_gaussians} Gaussians by opacity")
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

    cameras = scene.getTestCameras()
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # ========== 3D_direct ==========
    print("\n" + "=" * 80)
    print("RENDERING: 3D_direct")
    print("=" * 80)

    args_direct = Namespace(**vars(args))
    args_direct.method = "3D_direct"

    ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
    ingp_direct.load_model(model_path, iteration)
    ingp_direct.set_active_levels(iteration)

    # Enable gradients
    gaussians._xyz.requires_grad_(True)
    gaussians._scaling.requires_grad_(True)
    gaussians._rotation.requires_grad_(True)
    gaussians._opacity.requires_grad_(True)
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        gaussians._gaussian_features.requires_grad_(True)

    # Zero all grads
    for p in [gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]:
        p.grad = None
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        gaussians._gaussian_features.grad = None
    for param in ingp_direct.parameters():
        param.requires_grad_(True)
        param.grad = None

    result_direct = render(
        cameras[0], gaussians, args_direct, background,
        ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model
    )
    render_direct = result_direct["render"]
    print(f"Render mean: {render_direct.mean().item():.6f}")

    loss_direct = render_direct.sum()
    loss_direct.backward()

    # Collect gradients
    grads_direct = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scales': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotations': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
    }

    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        grads_direct['gaussian_features'] = gaussians._gaussian_features.grad.clone() if gaussians._gaussian_features.grad is not None else None

    if hasattr(ingp_direct, 'hash_encoding') and ingp_direct.hash_encoding.embeddings.grad is not None:
        grads_direct['hash_embeddings'] = ingp_direct.hash_encoding.embeddings.grad.clone()
    else:
        grads_direct['hash_embeddings'] = None

    if hasattr(ingp_direct, 'mlp_3D_direct') and ingp_direct.mlp_3D_direct is not None:
        mlp = ingp_direct.mlp_3D_direct
        grads_direct['W1'] = mlp[0].weight.grad.clone() if mlp[0].weight.grad is not None else None
        grads_direct['b1'] = mlp[0].bias.grad.clone() if mlp[0].bias.grad is not None else None
        grads_direct['W2'] = mlp[2].weight.grad.clone() if mlp[2].weight.grad is not None else None
        grads_direct['b2'] = mlp[2].bias.grad.clone() if mlp[2].bias.grad is not None else None
        grads_direct['W3'] = mlp[4].weight.grad.clone() if mlp[4].weight.grad is not None else None
        grads_direct['b3'] = mlp[4].bias.grad.clone() if mlp[4].bias.grad is not None else None

    # ========== 3D_direct_lean ==========
    print("\n" + "=" * 80)
    print("RENDERING: 3D_direct_lean")
    print("=" * 80)

    args_lean = Namespace(**vars(args))
    args_lean.method = "3D_direct_lean"

    ingp_lean = INGP(cfg_model, args=args_lean).to('cuda')
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)

    # Ensure mlp_fused exists
    if ingp_lean.mlp_fused is None:
        ingp_lean.mlp_fused = nn.Sequential(
            nn.Linear(40, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Sigmoid()
        ).cuda()

    # Zero all grads
    for p in [gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]:
        p.grad = None
    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        gaussians._gaussian_features.grad = None
    for param in ingp_lean.parameters():
        param.requires_grad_(True)
        param.grad = None
    for param in ingp_lean.mlp_fused.parameters():
        param.requires_grad_(True)
        param.grad = None

    result_lean = render(
        cameras[0], gaussians, args_lean, background,
        ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model
    )
    render_lean = result_lean["render"]
    print(f"Render mean: {render_lean.mean().item():.6f}")

    # Debug: check if geomBuffer contains meaningful data
    if 'geomBuffer' in result_lean:
        geom = result_lean['geomBuffer']
        print(f"  geomBuffer: type={type(geom)}, size={geom.numel() if hasattr(geom, 'numel') else 'N/A'}")

    loss_lean = render_lean.sum()

    # Debug: Check what gets gradient
    if 'allmap' in result_lean:
        allmap = result_lean.get('allmap')
        if allmap is not None:
            print(f"  allmap shape: {allmap.shape}, requires_grad: {allmap.requires_grad}")

    loss_lean.backward()

    # Retrieve CUDA MLP grads
    from diff_surfel_3D import get_mlp_grads
    mlp_grads = get_mlp_grads()
    if mlp_grads is not None:
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = mlp_grads
        ingp_lean.mlp_fused[0].weight.grad = grad_W1.clone() if grad_W1 is not None else None
        ingp_lean.mlp_fused[0].bias.grad = grad_b1.clone() if grad_b1 is not None else None
        ingp_lean.mlp_fused[2].weight.grad = grad_W2.clone() if grad_W2 is not None else None
        ingp_lean.mlp_fused[2].bias.grad = grad_b2.clone() if grad_b2 is not None else None
        ingp_lean.mlp_fused[4].weight.grad = grad_W3.clone() if grad_W3 is not None else None
        ingp_lean.mlp_fused[4].bias.grad = grad_b3.clone() if grad_b3 is not None else None

    # Collect gradients
    grads_lean = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scales': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotations': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
    }

    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None:
        grads_lean['gaussian_features'] = gaussians._gaussian_features.grad.clone() if gaussians._gaussian_features.grad is not None else None

    if hasattr(ingp_lean, 'hash_encoding') and ingp_lean.hash_encoding.embeddings.grad is not None:
        grads_lean['hash_embeddings'] = ingp_lean.hash_encoding.embeddings.grad.clone()
    else:
        grads_lean['hash_embeddings'] = None

    mlp = ingp_lean.mlp_fused
    grads_lean['W1'] = mlp[0].weight.grad.clone() if mlp[0].weight.grad is not None else None
    grads_lean['b1'] = mlp[0].bias.grad.clone() if mlp[0].bias.grad is not None else None
    grads_lean['W2'] = mlp[2].weight.grad.clone() if mlp[2].weight.grad is not None else None
    grads_lean['b2'] = mlp[2].bias.grad.clone() if mlp[2].bias.grad is not None else None
    grads_lean['W3'] = mlp[4].weight.grad.clone() if mlp[4].weight.grad is not None else None
    grads_lean['b3'] = mlp[4].bias.grad.clone() if mlp[4].bias.grad is not None else None

    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON")
    print("=" * 80)

    print("\n--- Forward Pass ---")
    diff = (render_direct - render_lean).abs()
    print(f"  Render diff: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

    print("\n--- MLP Gradients ---")
    for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        compare_tensors(name, grads_direct.get(name), grads_lean.get(name))

    print("\n--- Feature Gradients ---")
    compare_tensors('gaussian_features', grads_direct.get('gaussian_features'), grads_lean.get('gaussian_features'))
    compare_tensors('hash_embeddings', grads_direct.get('hash_embeddings'), grads_lean.get('hash_embeddings'))

    print("\n--- Geometry Gradients ---")
    compare_tensors('xyz', grads_direct.get('xyz'), grads_lean.get('xyz'))
    compare_tensors('scales', grads_direct.get('scales'), grads_lean.get('scales'))
    compare_tensors('rotations', grads_direct.get('rotations'), grads_lean.get('rotations'))
    compare_tensors('opacity', grads_direct.get('opacity'), grads_lean.get('opacity'))

    # Show statistics of individual gradient tensors
    print("\n--- Detailed Gradient Stats ---")
    for name, g in [('xyz_direct', grads_direct.get('xyz')),
                    ('xyz_lean', grads_lean.get('xyz')),
                    ('scales_direct', grads_direct.get('scales')),
                    ('scales_lean', grads_lean.get('scales')),
                    ('opacity_direct', grads_direct.get('opacity')),
                    ('opacity_lean', grads_lean.get('opacity'))]:
        if g is not None:
            nonzero = (g.abs() > 1e-10).sum().item()
            total = g.numel()
            print(f"  {name}: mean={g.abs().mean().item():.6f}, nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--num_gaussians", type=int, default=None)
    cli_args = parser.parse_args()

    run_comparison(model_path=cli_args.model_path, num_gaussians=cli_args.num_gaussians)
