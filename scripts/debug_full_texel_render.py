#!/usr/bin/env python3
"""
Test: Render with FULL per-texel SH textures (zero base SH + full texel texture).
This bypasses the mean+residual decomposition to verify the texture data is correct.

If this gives ~35 dB: texture is correct, issue is in mean+residual recombination.
If this gives ~30 dB: 8x8 texture resolution is insufficient.
"""
import os, sys, math, glob, pickle, torch, json
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
baked_dir = os.path.join(model_path, "baked")

# Load training config
with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
cfg_model = Config(os.path.join(model_path, "config.yaml"))

# Load baked PLY
temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
gaussians.load_ply(os.path.join(baked_dir, "baked.ply"))
gaussians.active_sh_degree = 3
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
N = len(gaussians.get_xyz)

# Load textures
residual_tex = torch.load(os.path.join(baked_dir, "residual_textures.pt")).cuda().float()
# [N, 8, 8, 48] FP16 residual

# Reconstruct mean SH from PLY
dc = gaussians._features_dc.data  # [N, 1, 3]
rest = gaussians._features_rest.data  # [N, 15, 3]
mean_sh_ply = torch.cat([dc, rest], dim=1)  # [N, 16, 3]
mean_sh = mean_sh_ply.permute(0, 2, 1).reshape(N, 48)  # → [N, 48] channel-first

# Reconstruct full per-texel SH
full_sh = residual_tex.view(N, 64, 48) + mean_sh.unsqueeze(1)  # [N, 64, 48]
full_sh_tex = full_sh.view(N, 8, 8, 48)

print(f"Full texel SH: mean={full_sh.mean():.4f}, std={full_sh.std():.4f}")
print(f"Residual: mean={residual_tex.mean():.4f}, std={residual_tex.std():.4f}")

# Load test cameras
iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
test_cameras = scene.getTestCameras()

# CRITICAL: Scene() overwrites baked PLY with training PLY. Reload baked.
gaussians.load_ply(os.path.join(baked_dir, "baked.ply"))
gaussians.active_sh_degree = 3
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
N = len(gaussians.get_xyz)
# Reconstruct mean SH again after reload
dc = gaussians._features_dc.data
rest = gaussians._features_rest.data
mean_sh_ply = torch.cat([dc, rest], dim=1)
mean_sh = mean_sh_ply.permute(0, 2, 1).reshape(N, 48)
print(f"Reloaded baked PLY. {len(test_cameras)} test cameras, {N} Gaussians")

# Rendering imports
from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

kernel_name = getattr(args, 'kernel', 'gaussian')
kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
kernel_type = kernel_map.get(kernel_name, 0)
beta = cfg_model.surfel.tg_beta if hasattr(cfg_model.surfel, 'tg_beta') else 0.0
bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')

def render_test(gaussians, test_cams, residual_textures, label):
    psnrs_list = []
    with torch.no_grad():
        for cam in test_cams:
            tanfovx = math.tan(cam.FoVx * 0.5)
            tanfovy = math.tan(cam.FoVy * 0.5)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam.image_height),
                image_width=int(cam.image_width),
                tanfovx=tanfovx, tanfovy=tanfovy,
                bg=bg_color, scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=cam.camera_center,
                prefiltered=False, debug=False, beta=beta,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            result = rasterizer(
                means3D=gaussians.get_xyz,
                means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
                opacities=gaussians.get_opacity,
                shs=gaussians.get_features,
                scales=gaussians.get_scaling,
                rotations=gaussians.get_rotation,
                kernel_type=kernel_type,
                residual_textures=residual_textures,
            )
            rendered = result[0].clamp(0, 1)
            gt = cam.original_image[:3].cuda()
            p = psnr(rendered, gt).mean().item()
            psnrs_list.append(p)
    avg_psnr = np.mean(psnrs_list)
    print(f"  {label}: PSNR={avg_psnr:.2f} dB (per-view: {[f'{p:.2f}' for p in psnrs_list[:5]]}...)")
    return avg_psnr

# Test 1: SH only (no residual)
print("\n--- Test 1: SH only (mean SH from PLY) ---")
render_test(gaussians, test_cameras, None, "SH only")

# Test 2: SH + residual (current approach)
print("\n--- Test 2: SH + 48D residual ---")
residual_flat = residual_tex.half().view(N, -1).contiguous()
render_test(gaussians, test_cameras, residual_flat, "SH + residual")

# Test 3: Zero base SH + full texel SH texture
print("\n--- Test 3: Zero SH + full texel texture ---")
# Zero out the PLY's SH
with torch.no_grad():
    saved_dc = gaussians._features_dc.data.clone()
    saved_rest = gaussians._features_rest.data.clone()
    gaussians._features_dc.data.zero_()
    gaussians._features_rest.data.zero_()

full_tex_flat = full_sh_tex.half().view(N, -1).contiguous()
render_test(gaussians, test_cameras, full_tex_flat, "Zero SH + full texel")

# Restore
with torch.no_grad():
    gaussians._features_dc.data.copy_(saved_dc)
    gaussians._features_rest.data.copy_(saved_rest)

# Test 4: Full texel SH as texture + mean SH base (double-counts, should be BAD)
# This tests if the texture is being applied at all
print("\n--- Test 4: Mean SH + full texel (double-count, should be wrong) ---")
render_test(gaussians, test_cameras, full_tex_flat, "SH + full texel (2x)")
