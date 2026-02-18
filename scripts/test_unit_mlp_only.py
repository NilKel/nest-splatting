#!/usr/bin/env python3
"""
Simple test: ONLY unit MLP backward (no real weights first).
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

    print("=" * 70)
    print("UNIT MLP ONLY TEST")
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

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    args_lean = Namespace(**vars(base_args))
    args_lean.method = '3D_direct_lean'

    ingp = INGP(cfg_model, args=args_lean).cuda()
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    # Create unit weights BEFORE any render
    print("\nSetting unit MLP weights...")
    W1_unit = torch.zeros(32, 40, device='cuda')
    b1_unit = torch.zeros(32, device='cuda')
    W2_unit = torch.eye(32, device='cuda')
    b2_unit = torch.zeros(32, device='cuda')
    W3_unit = torch.zeros(3, 32, device='cuda')
    b3_unit = torch.zeros(3, device='cuda')

    for i in range(32):
        W1_unit[i, i] = 1.0
    for i in range(3):
        W3_unit[i, i] = 1.0

    with torch.no_grad():
        ingp.mlp_fused[0].weight.copy_(W1_unit)
        ingp.mlp_fused[0].bias.copy_(b1_unit)
        ingp.mlp_fused[2].weight.copy_(W2_unit)
        ingp.mlp_fused[2].bias.copy_(b2_unit)
        ingp.mlp_fused[4].weight.copy_(W3_unit)
        ingp.mlp_fused[4].bias.copy_(b3_unit)

    print(f"  W1[0,0:4]: {ingp.mlp_fused[0].weight[0, :4].tolist()}")
    print(f"  W2[0,0:4]: {ingp.mlp_fused[2].weight[0, :4].tolist()}")
    print(f"  W3[0,0:4]: {ingp.mlp_fused[4].weight[0, :4].tolist()}")

    # Zero grads
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp.hash_encoding.embeddings.grad = None

    # Forward + backward with unit MLP
    print("\nRunning forward + backward with unit MLP...")
    render_pkg = render(camera, gaussians, args_lean, bg_color, ingp=ingp, cfg=cfg_model)
    image = render_pkg['render']

    print(f"Image: [{image.min():.4f}, {image.max():.4f}]")

    loss = image.sum()
    loss.backward()

    # Synchronize and flush CUDA printf
    torch.cuda.synchronize()

    # Get CUDA MLP grads
    import diff_surfel_3D
    cuda_mlp_grads = diff_surfel_3D.get_mlp_grads()

    print("\nCUDA MLP gradients (unit weights):")
    names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    for name, grad in zip(names, cuda_mlp_grads):
        if grad is not None:
            nz = (grad.abs() > 1e-10).sum().item()
            has_nan = torch.isnan(grad).any().item()
            print(f"  {name}: norm={grad.norm():.4f}, nonzero={nz}/{grad.numel()}, NaN={has_nan}")
            if 'b' in name:
                nz_idx = (grad.abs() > 1e-10).nonzero().squeeze(-1).tolist()
                print(f"       nonzero indices: {nz_idx}")

    print("\nExpected with unit MLP:")
    print("  b3: 3/3 nonzeros at [0,1,2]")
    print("  b2: 3/32 nonzeros at [0,1,2]")
    print("  b1: 3/32 nonzeros at [0,1,2]")

    # Analysis
    b3_nz = (cuda_mlp_grads[5].abs() > 1e-10).sum().item()
    b2_nz = (cuda_mlp_grads[3].abs() > 1e-10).sum().item()
    b1_nz = (cuda_mlp_grads[1].abs() > 1e-10).sum().item()

    print(f"\nResult: b3={b3_nz}/3, b2={b2_nz}/32, b1={b1_nz}/32")


if __name__ == "__main__":
    main()
