#!/usr/bin/env python3
"""
Compare cat mode vs lean mode gradients by rendering the same view and backwarding image.sum().
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


def cos_sim(a, b):
    if a is None or b is None:
        return float('nan')
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if a_flat.norm() < 1e-8 or b_flat.norm() < 1e-8:
        return 0.0
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main():
    # Model path
    model_path = 'outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8'
    yaml_path = 'configs/nerfsyn.yaml'
    iteration = 30000

    if not os.path.exists(f'{model_path}/ngp_{iteration}.pth'):
        print(f"Model not found: {model_path}/ngp_{iteration}.pth")
        return

    print("=" * 60)
    print("GRADIENT COMPARISON: CAT vs LEAN")
    print("=" * 60)

    # Load config
    cfg_model = Config(yaml_path)

    # Base args - use absolute path
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
        # Pipeline args needed by render()
        compute_cov3D_python=False,
        convert_SHs_python=False,
        antialiasing=False,
    )
    merge_cfg_to_args(base_args, cfg_model)
    # Reset source_path after merge (config may override it incorrectly)
    base_args.source_path = source_path

    # Load scene (just for camera)
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(base_args, gaussians, load_iteration=iteration, shuffle=False)
    camera = scene.getTrainCameras()[0]

    print(f"Loaded scene with {gaussians._xyz.shape[0]} Gaussians")
    print(f"Camera: {camera.image_width}x{camera.image_height}")

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    # =========================================================================
    # 3D_DIRECT MODE (reference)
    # =========================================================================
    print("\n" + "=" * 60)
    print("3D_DIRECT MODE (reference)")
    print("=" * 60)

    args_direct = Namespace(**vars(base_args))
    args_direct.method = '3D_direct'

    ingp_direct = INGP(cfg_model, args=args_direct).cuda()
    ingp_direct.load_model(model_path, iteration)
    ingp_direct.set_active_levels(iteration)  # Initialize active_levels

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_direct.hash_encoding.embeddings.grad = None

    # Forward + backward
    render_pkg_direct = render(camera, gaussians, args_direct, bg_color, ingp=ingp_direct, cfg=cfg_model)
    image_direct = render_pkg_direct['render']
    loss_direct = image_direct.sum()
    loss_direct.backward()

    print(f"Image range: [{image_direct.min():.4f}, {image_direct.max():.4f}]")

    grad_direct = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scaling': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotation': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
        'features': gaussians._gaussian_features.grad.clone() if gaussians._gaussian_features.grad is not None else None,
        'hash': ingp_direct.hash_encoding.embeddings.grad.clone() if ingp_direct.hash_encoding.embeddings.grad is not None else None,
    }

    for name, g in grad_direct.items():
        if g is not None:
            print(f"  {name}: norm={g.norm():.4f}, mean={g.abs().mean():.6f}")
        else:
            print(f"  {name}: None")

    # =========================================================================
    # LEAN MODE
    # =========================================================================
    print("\n" + "=" * 60)
    print("LEAN MODE (3D_direct_lean)")
    print("=" * 60)

    args_lean = Namespace(**vars(base_args))
    args_lean.method = '3D_direct_lean'

    ingp_lean = INGP(cfg_model, args=args_lean).cuda()
    ingp_lean.load_model(model_path, iteration)
    ingp_lean.set_active_levels(iteration)  # Initialize active_levels

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp_lean.hash_encoding.embeddings.grad = None

    # Forward + backward
    render_pkg_lean = render(camera, gaussians, args_lean, bg_color, ingp=ingp_lean, cfg=cfg_model)
    image_lean = render_pkg_lean['render']
    loss_lean = image_lean.sum()
    loss_lean.backward()

    print(f"Image range: [{image_lean.min():.4f}, {image_lean.max():.4f}]")

    grad_lean = {
        'xyz': gaussians._xyz.grad.clone() if gaussians._xyz.grad is not None else None,
        'scaling': gaussians._scaling.grad.clone() if gaussians._scaling.grad is not None else None,
        'rotation': gaussians._rotation.grad.clone() if gaussians._rotation.grad is not None else None,
        'opacity': gaussians._opacity.grad.clone() if gaussians._opacity.grad is not None else None,
        'features': gaussians._gaussian_features.grad.clone() if gaussians._gaussian_features.grad is not None else None,
        'hash': ingp_lean.hash_encoding.embeddings.grad.clone() if ingp_lean.hash_encoding.embeddings.grad is not None else None,
    }

    for name, g in grad_lean.items():
        if g is not None:
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            print(f"  {name}: norm={g.norm():.4f}, mean={g.abs().mean():.6f}, NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  {name}: None")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("GRADIENT COMPARISON: 3D_direct vs lean")
    print("=" * 60)

    print(f"\n{'Gradient':<12} {'cos_sim':>10} {'direct_norm':>12} {'lean_norm':>12} {'Status':>8}")
    print("-" * 60)

    for name in ['xyz', 'scaling', 'rotation', 'opacity', 'features', 'hash']:
        g_direct = grad_direct.get(name)
        g_lean = grad_lean.get(name)

        cs = cos_sim(g_direct, g_lean)
        direct_norm = g_direct.norm().item() if g_direct is not None else 0
        lean_norm = g_lean.norm().item() if g_lean is not None else 0

        status = "OK" if cs > 0.99 else "CHECK" if cs > 0.9 else "FAIL" if cs < 0.5 else "LOW"
        print(f"{name:<12} {cs:>10.4f} {direct_norm:>12.4f} {lean_norm:>12.4f} {status:>8}")

    # Forward image comparison
    print("\n" + "=" * 60)
    print("FORWARD IMAGE COMPARISON")
    print("=" * 60)
    img_diff = (image_direct.detach() - image_lean.detach()).abs()
    print(f"Image diff: mean={img_diff.mean():.6f}, max={img_diff.max():.6f}")
    psnr_diff = -10 * torch.log10(img_diff.mean() + 1e-8)
    print(f"PSNR between modes: {psnr_diff:.2f} dB")


if __name__ == "__main__":
    main()
