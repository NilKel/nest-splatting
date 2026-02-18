#!/usr/bin/env python3
"""Only test unit MLP - no real weights first."""

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

    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    args = Namespace(**vars(base_args))
    args.method = '3D_direct_lean'

    ingp = INGP(cfg_model, args=args).cuda()
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    # Set unit weights IMMEDIATELY
    print("Setting unit MLP weights...")
    with torch.no_grad():
        ingp.mlp_fused[0].weight.zero_()
        ingp.mlp_fused[0].bias.zero_()
        for i in range(32):
            ingp.mlp_fused[0].weight[i, i] = 1.0

        ingp.mlp_fused[2].weight.zero_()
        ingp.mlp_fused[2].bias.zero_()
        ingp.mlp_fused[2].weight.copy_(torch.eye(32))

        ingp.mlp_fused[4].weight.zero_()
        ingp.mlp_fused[4].bias.zero_()
        for i in range(3):
            ingp.mlp_fused[4].weight[i, i] = 1.0

    print(f"W3 diagonal: {ingp.mlp_fused[4].weight[0,0]}, {ingp.mlp_fused[4].weight[1,1]}, {ingp.mlp_fused[4].weight[2,2]}")
    print(f"W3 off-diag: {ingp.mlp_fused[4].weight[0,1]}, {ingp.mlp_fused[4].weight[1,0]}")

    # Render
    print("\nRendering with unit MLP...")
    gaussians._xyz.grad = None
    gaussians._scaling.grad = None
    gaussians._rotation.grad = None
    gaussians._opacity.grad = None
    gaussians._gaussian_features.grad = None
    ingp.hash_encoding.embeddings.grad = None

    render_pkg = render(camera, gaussians, args, bg_color, ingp=ingp, cfg=cfg_model)
    image = render_pkg['render']
    loss = image.sum()
    loss.backward()

    torch.cuda.synchronize()

    import diff_surfel_3D
    grads = diff_surfel_3D.get_mlp_grads()
    names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    print("\nCUDA MLP grads:")
    for name, g in zip(names, grads):
        if g is not None:
            nz = (g.abs() > 1e-10).sum().item()
            print(f"  {name}: nonzero={nz}/{g.numel()}")
            if 'b' in name:
                idx = (g.abs() > 1e-10).nonzero().squeeze(-1).tolist()
                print(f"       indices: {idx}")


if __name__ == "__main__":
    main()
