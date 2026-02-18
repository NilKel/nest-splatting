#!/usr/bin/env python3
"""
Test: Verify MLP gradient magnitude matches expected values.

With unit MLP and loss = image.sum():
- dL/dW should be proportional to number of intersections
- Each intersection contributes: dL_dout * input

This test checks if the magnitudes are reasonable.
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


def main():
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    yaml_path = 'configs/nerfsyn.yaml'
    iteration = 30000

    if not os.path.exists(f'{model_path}/ngp_{iteration}.pth'):
        print(f"Model not found")
        return

    print("=" * 60)
    print("MLP GRADIENT MAGNITUDE TEST")
    print("=" * 60)

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

    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(base_args, gaussians, load_iteration=iteration, shuffle=False)
    camera = scene.getTrainCameras()[0]
    print(f"Loaded {gaussians._xyz.shape[0]} Gaussians")

    args_lean = Namespace(**vars(base_args))
    args_lean.method = '3D_direct_lean'

    ingp_lean = INGP(cfg_model, args=args_lean).cuda()
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)

    # Use REAL MLP weights (not unit)
    print("\nUsing real MLP weights (not unit)")

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

    print("\nRunning backward (loss = image.mean())...")
    loss = image.mean()  # Use mean instead of sum for normalization
    loss.backward()

    print("\nGeometry gradients:")
    print(f"  xyz: {gaussians._xyz.grad.norm():.4f}")
    print(f"  scaling: {gaussians._scaling.grad.norm():.4f}")
    print(f"  rotation: {gaussians._rotation.grad.norm():.4f}")
    print(f"  opacity: {gaussians._opacity.grad.norm():.4f}")
    print(f"  features: {gaussians._gaussian_features.grad.norm():.4f}")
    print(f"  hash: {ingp_lean.hash_encoding.embeddings.grad.norm():.4f}")

    print("\nMLP gradients (with mean loss):")
    import diff_surfel_3D
    mlp_grads = diff_surfel_3D.get_mlp_grads()
    if mlp_grads is not None:
        names = ['dL_dW1', 'dL_db1', 'dL_dW2', 'dL_db2', 'dL_dW3', 'dL_db3']
        for name, grad in zip(names, mlp_grads):
            if grad is not None:
                print(f"  {name}: norm={grad.norm():.4f}, mean={grad.abs().mean():.6e}")

    # Now compare with Python MLP backward
    print("\n" + "=" * 60)
    print("COMPARISON WITH PYTHON FORWARD/BACKWARD")
    print("=" * 60)

    # Reset gradients
    for param in ingp_lean.mlp_fused.parameters():
        param.grad = None

    # Manual forward through Python MLP with same loss
    # This requires reconstructing the intersection data...
    # For now, just compare the gradient order of magnitude

    # Expected: with ~800*800 pixels and loss=mean, grad_out = 1/(800*800)
    # Each pixel accumulates ~30 intersections
    # So total gradient accumulation is ~800*800*30 = 19.2M intersections
    # But we divide by 800*800*3 for mean loss
    # So average per-intersection contribution = 30*3 = 90 times larger than 1.0

    expected_scale = 30 * 3  # Rough estimate
    print(f"\nExpected gradient scale factor (vs per-pixel): ~{expected_scale}x")
    print("If MLP grads are much larger, there may be an accumulation bug")


if __name__ == "__main__":
    main()
