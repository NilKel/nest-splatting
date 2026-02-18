#!/usr/bin/env python3
"""
Simple test: Load checkpoint, render with lean mode, backward, check gradients.
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
    """Merge config into args."""
    for section in ['training_cfg', 'settings', 'loss']:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    setattr(args, k, v)


def main():
    # Model path
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    yaml_path = 'configs/nerfsyn.yaml'
    iteration = 30000

    if not os.path.exists(f'{model_path}/ngp_{iteration}.pth'):
        print(f"Model not found: {model_path}/ngp_{iteration}.pth")
        return

    print("=" * 60)
    print("LEAN MODE GRADIENT TEST")
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

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_lean.hash_encoding.embeddings.grad = None

    # Forward + backward
    print("\nRunning forward pass...")
    render_pkg = render(camera, gaussians, args_lean, bg_color, ingp=ingp_lean, cfg=cfg_model)
    image = render_pkg['render']
    print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")

    print("\nRunning backward pass (loss = image.sum())...")
    loss = image.sum()
    loss.backward()

    print("\nGradient statistics:")
    def check_grad(name, tensor):
        if tensor.grad is None:
            print(f"  {name}: None")
            return
        g = tensor.grad
        has_nan = torch.isnan(g).any().item()
        has_inf = torch.isinf(g).any().item()
        nonzero = (g.abs() > 1e-10).sum().item()
        print(f"  {name}: norm={g.norm():.4f}, mean={g.abs().mean():.6e}, max={g.abs().max():.6e}, "
              f"nonzero={nonzero}/{g.numel()}, NaN={has_nan}, Inf={has_inf}")

    check_grad("xyz", gaussians._xyz)
    check_grad("scaling", gaussians._scaling)
    check_grad("rotation", gaussians._rotation)
    check_grad("opacity", gaussians._opacity)
    check_grad("features", gaussians._gaussian_features)
    check_grad("hash", ingp_lean.hash_encoding.embeddings)

    # Check MLP gradients
    print("\nMLP gradients:")
    from diff_surfel_3D import get_mlp_grads
    mlp_grads = get_mlp_grads()
    if mlp_grads is not None:
        names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for name, grad in zip(names, mlp_grads):
            if grad is not None:
                has_nan = torch.isnan(grad).any().item()
                nonzero = (grad.abs() > 1e-10).sum().item()
                print(f"  {name}: norm={grad.norm():.4f}, mean={grad.abs().mean():.6e}, "
                      f"nonzero={nonzero}/{grad.numel()}, NaN={has_nan}")
            else:
                print(f"  {name}: None")
    else:
        print("  get_mlp_grads() returned None")


if __name__ == "__main__":
    main()
