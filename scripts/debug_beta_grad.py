#!/usr/bin/env python3
"""
Test gradient flow for beta/beta_scaled kernels.
Loads a trained model, renders one view, backprops, checks gradients.
"""
import os, sys, glob, pickle, torch, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from torch import nn

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"

with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
cfg_model = Config(os.path.join(model_path, "config.yaml"))

iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))

temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
test_cameras = scene.getTestCameras()

# Setup INGP for neural rendering
ingp = INGP(cfg_model, args=args).to('cuda')
ingp.load_model(model_path, iteration)
ingp.set_active_levels(iteration)

from gaussian_renderer import render
pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.0)
bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

cam = test_cameras[0]
gt = cam.original_image[:3].cuda()

beta = cfg_model.surfel.tg_beta

# Test each kernel type
for kernel_name, kernel_int in [("gaussian", 0), ("beta", 1), ("beta_scaled", 4)]:
    print(f"\n{'='*60}")
    print(f"Testing kernel: {kernel_name} (type={kernel_int})")
    print(f"{'='*60}")

    # Reset shape parameter
    N = len(gaussians.get_xyz)
    if kernel_int in [1, 4]:
        # Init shape: sigmoid(0.847)*5 ≈ 3.0
        shape_init = torch.full((N, 1), 0.847, device="cuda")
        gaussians._shape = nn.Parameter(shape_init.clone().requires_grad_(True))
        gaussians.kernel_type = kernel_name
        print(f"  _shape: shape={gaussians._shape.shape}, requires_grad={gaussians._shape.requires_grad}")
        print(f"  get_shape mean: {gaussians.get_shape.mean().item():.4f}")
    else:
        gaussians._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        gaussians.kernel_type = "gaussian"

    # Zero any existing grads
    for p in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._scaling, gaussians._rotation, gaussians._opacity]:
        if p.grad is not None:
            p.grad.zero_()
    if gaussians._shape.numel() > 0 and gaussians._shape.grad is not None:
        gaussians._shape.grad.zero_()

    # Render
    render_pkg = render(cam, gaussians, pipe, bg_color,
                        ingp=ingp, cfg=cfg_model, iteration=iteration,
                        beta=beta, aabb_mode="2dgs")
    img = render_pkg["render"]

    # L2 loss
    loss = ((img - gt) ** 2).mean()
    print(f"  Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check gradients
    print(f"\n  --- Gradient Check ---")
    for name, param in [("_xyz", gaussians._xyz),
                        ("_features_dc", gaussians._features_dc),
                        ("_scaling", gaussians._scaling),
                        ("_rotation", gaussians._rotation),
                        ("_opacity", gaussians._opacity)]:
        if param.grad is not None:
            g = param.grad
            print(f"  {name:20s}: grad_norm={g.norm().item():.6f}, nonzero={(g.abs() > 1e-10).sum().item()}/{g.numel()}")
        else:
            print(f"  {name:20s}: grad=None")

    if gaussians._shape.numel() > 0:
        if gaussians._shape.grad is not None:
            g = gaussians._shape.grad
            nz = (g.abs() > 1e-10).sum().item()
            print(f"  {'_shape':20s}: grad_norm={g.norm().item():.6f}, nonzero={nz}/{g.numel()}, "
                  f"mean={g.mean().item():.8f}, max={g.abs().max().item():.8f}")
            if nz == 0:
                print(f"  *** WARNING: _shape gradient is ALL ZEROS! ***")
        else:
            print(f"  {'_shape':20s}: grad=None *** MISSING GRADIENT! ***")
    else:
        print(f"  {'_shape':20s}: empty (not used for kernel_type={kernel_int})")

    # Check if shapes was in the render graph
    if kernel_int in [1, 4]:
        shapes_val = gaussians.get_shape
        print(f"\n  shapes (get_shape) is leaf: {shapes_val.is_leaf}")
        print(f"  shapes requires_grad: {shapes_val.requires_grad}")
        print(f"  shapes grad_fn: {shapes_val.grad_fn}")
        print(f"  _shape is leaf: {gaussians._shape.is_leaf}")
        print(f"  _shape requires_grad: {gaussians._shape.requires_grad}")
