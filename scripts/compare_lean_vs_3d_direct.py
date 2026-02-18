#!/usr/bin/env python3
"""
Compare gradients between 3D_direct (Python-based) and 3D_direct_lean (fused CUDA).
Both should produce identical gradients since they compute the same thing.
"""

import os
import sys
import torch
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.config import Config
from hash_encoder.modules import INGP


def merge_cfg_to_args(args, cfg):
    for section in ['training_cfg', 'settings', 'loss']:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    setattr(args, k, v)


def cos_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def main():
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    yaml_path = 'configs/nerfsyn.yaml'
    iteration = 30000

    if not os.path.exists(f'{model_path}/ngp_{iteration}.pth'):
        print(f"Model not found")
        return

    print("=" * 70)
    print("GRADIENT COMPARISON: 3D_direct vs 3D_direct_lean")
    print("=" * 70)

    cfg_model = Config(yaml_path)
    source_path = '/home/nilkel/Projects/nest-splatting/data/nerf_synthetic/chair'

    base_args = Namespace(
        sh_degree=3, source_path=source_path, model_path=model_path,
        images='images', resolution=-1, white_background=True,
        data_device='cuda', eval=True, debug=False, scale_invariant=False,
        max_abs_split_scale=0.3, soft_beta=False, hybrid_levels=5,
        compute_cov3D_python=False, convert_SHs_python=False, antialiasing=False,
    )
    merge_cfg_to_args(base_args, cfg_model)
    base_args.source_path = source_path

    # Load model
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(base_args, gaussians, load_iteration=iteration, shuffle=False)
    camera = scene.getTrainCameras()[0]

    print(f"Loaded {gaussians._xyz.shape[0]} Gaussians")
    print(f"Features shape: {gaussians._gaussian_features.shape}")

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    # =========================================================================
    # 3D_direct mode (Python-based)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3D_direct MODE (Python MLP)")
    print("=" * 70)

    args_3d = Namespace(**vars(base_args))
    args_3d.method = '3D_direct'

    ingp_3d = INGP(cfg_model, args=args_3d).cuda()
    ingp_3d.load_model(model_path, iteration)
    ingp_3d.set_active_levels(iteration)

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_3d.hash_encoding.embeddings.grad = None
    if ingp_3d.mlp_3D_direct is not None:
        for p in ingp_3d.mlp_3D_direct.parameters():
            p.grad = None

    # Forward + backward
    render_pkg = render(camera, gaussians, args_3d, bg_color, ingp=ingp_3d, cfg=cfg_model)
    image_3d = render_pkg['render']
    loss = image_3d.sum()
    loss.backward()

    print(f"Image range: [{image_3d.min():.4f}, {image_3d.max():.4f}]")

    # Store gradients
    grad_3d = {
        'xyz': gaussians._xyz.grad.clone(),
        'scaling': gaussians._scaling.grad.clone(),
        'rotation': gaussians._rotation.grad.clone(),
        'opacity': gaussians._opacity.grad.clone(),
        'features': gaussians._gaussian_features.grad.clone(),
        'hash': ingp_3d.hash_encoding.embeddings.grad.clone(),
    }

    # MLP grads (from PyTorch/tcnn)
    mlp_grads_3d = []
    if ingp_3d.mlp_3D_direct is not None:
        # tcnn-style: all params in one tensor
        if hasattr(ingp_3d.mlp_3D_direct, 'params') and ingp_3d.mlp_3D_direct.params.grad is not None:
            mlp_grads_3d.append(('all_params', ingp_3d.mlp_3D_direct.params.grad.clone()))
        else:
            print("  No MLP grad for 3D_direct (tcnn)")

    # =========================================================================
    # 3D_direct_lean mode (fused CUDA)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3D_direct_lean MODE (Fused CUDA)")
    print("=" * 70)

    args_lean = Namespace(**vars(base_args))
    args_lean.method = '3D_direct_lean'

    ingp_lean = INGP(cfg_model, args=args_lean).cuda()
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_lean.hash_encoding.embeddings.grad = None

    # Forward + backward
    render_pkg = render(camera, gaussians, args_lean, bg_color, ingp=ingp_lean, cfg=cfg_model)
    image_lean = render_pkg['render']
    loss = image_lean.sum()
    loss.backward()

    print(f"Image range: [{image_lean.min():.4f}, {image_lean.max():.4f}]")

    # Store gradients
    grad_lean = {
        'xyz': gaussians._xyz.grad.clone(),
        'scaling': gaussians._scaling.grad.clone(),
        'rotation': gaussians._rotation.grad.clone(),
        'opacity': gaussians._opacity.grad.clone(),
        'features': gaussians._gaussian_features.grad.clone(),
        'hash': ingp_lean.hash_encoding.embeddings.grad.clone(),
    }

    # MLP grads (from CUDA)
    import diff_surfel_3D
    cuda_mlp_grads = diff_surfel_3D.get_mlp_grads()
    mlp_grads_lean = []
    if cuda_mlp_grads is not None:
        names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for name, grad in zip(names, cuda_mlp_grads):
            mlp_grads_lean.append((name, grad))

    # =========================================================================
    # Compare images
    # =========================================================================
    print("\n" + "=" * 70)
    print("IMAGE COMPARISON")
    print("=" * 70)
    img_diff = (image_3d.detach() - image_lean.detach()).abs()
    print(f"Max pixel diff: {img_diff.max():.6f}")
    print(f"Mean pixel diff: {img_diff.mean():.6f}")

    # =========================================================================
    # Compare geometry gradients
    # =========================================================================
    print("\n" + "=" * 70)
    print("GEOMETRY GRADIENT COMPARISON")
    print("=" * 70)

    print(f"\n{'Gradient':<12} {'cos_sim':>10} {'3D_norm':>12} {'lean_norm':>12} {'diff_norm':>12}")
    print("-" * 60)

    for name in ['xyz', 'scaling', 'rotation', 'opacity', 'features', 'hash']:
        g1 = grad_3d[name]
        g2 = grad_lean[name]
        cs = cos_sim(g1, g2)
        diff = (g1 - g2).abs()
        print(f"{name:<12} {cs:>10.6f} {g1.norm():>12.4f} {g2.norm():>12.4f} {diff.max():>12.6f}")

    # =========================================================================
    # Compare MLP gradients
    # =========================================================================
    print("\n" + "=" * 70)
    print("MLP GRADIENT COMPARISON")
    print("=" * 70)

    print("\n3D_direct MLP grads (tcnn):")
    for name, grad in mlp_grads_3d:
        if grad is not None:
            nz = (grad.abs() > 1e-10).sum().item()
            has_nan = torch.isnan(grad).any().item()
            print(f"  {name}: norm={grad.norm():.4f}, nonzero={nz}/{grad.numel()}, NaN={has_nan}")
        else:
            print(f"  {name}: None")

    print("\n3D_direct_lean MLP grads (fused CUDA):")
    for name, grad in mlp_grads_lean:
        if grad is not None:
            nz = (grad.abs() > 1e-10).sum().item()
            has_nan = torch.isnan(grad).any().item()
            print(f"  {name}: norm={grad.norm():.4f}, nonzero={nz}/{grad.numel()}, NaN={has_nan}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
