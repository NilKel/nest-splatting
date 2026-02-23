#!/usr/bin/env python3
"""
Render test frame 125 from a 3D_SH_TC model in multiple modes:
  1. 125_train.png   — full neural render (per-Gaussian + hash features)
  2. 125_gauss.png   — hash features zeroed (only per-Gaussian contribution)
  3. 125_hash.png    — per-Gaussian features zeroed (only hash contribution)
  4. 125_baked.png   — baked SH + 48D residual texture render
  5. 125_sh_only.png — baked SH only (no residual)
  6. 125_gt.png      — ground truth
"""
import os, sys, math, glob, pickle, torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from torch import nn

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("model_path", nargs="?", default="outputs/nerf_synthetic/chair/3D_SH_TC/betscaled")
_parser.add_argument("--view", type=int, default=125)
_parser.add_argument("--subdir", type=str, default="debugtex")
_parser.add_argument("--skip_bake", action="store_true", help="Skip re-baking, use existing baked/ files")
_parser.add_argument("--ss", type=int, default=None, help="Supersample factor passed to bake_hybrid.py")
_cli_args = _parser.parse_args()

model_path = _cli_args.model_path
VIEW_IDX = _cli_args.view

out_dir = os.path.join(model_path, _cli_args.subdir)
os.makedirs(out_dir, exist_ok=True)

# Load config
with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
cfg_model = Config(os.path.join(model_path, "config.yaml"))

iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))

# Setup Gaussians and Scene
temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
test_cameras = scene.getTestCameras()
print(f"{len(test_cameras)} test views, using view {VIEW_IDX}")

# Setup INGP
ingp = INGP(cfg_model, args=args).to('cuda')
ingp.load_model(model_path, iteration)
ingp.set_active_levels(iteration)

from gaussian_renderer import render
pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.0)
bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
beta = cfg_model.surfel.tg_beta

cam = test_cameras[VIEW_IDX]
gt = cam.original_image[:3].cuda()

def save_image(tensor, path):
    img = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"  Saved {path}")

def compute_psnr(rendered, gt):
    from utils.image_utils import psnr
    return psnr(rendered.clamp(0, 1), gt).mean().item()

# =====================================================================
# 0. Ground truth
# =====================================================================
save_image(gt, os.path.join(out_dir, "125_gt.png"))

# =====================================================================
# 1. Full neural render
# =====================================================================
print("\n1. Full neural render (train mode)")
with torch.no_grad():
    render_pkg = render(cam, gaussians, pipe, bg_color,
                        ingp=ingp, cfg=cfg_model, iteration=iteration, beta=beta)
    img_train = render_pkg["render"]
    p = compute_psnr(img_train, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_train, os.path.join(out_dir, "125_train.png"))

# =====================================================================
# 2. Gaussian features only (zero hash)
# =====================================================================
print("\n2. Per-Gaussian features only (hash zeroed)")
with torch.no_grad():
    # Save original hash embeddings
    hash_enc = ingp.hash_encoding
    embeddings_orig = hash_enc.embeddings.data.clone()
    # Zero hash
    hash_enc.embeddings.data.zero_()
    render_pkg = render(cam, gaussians, pipe, bg_color,
                        ingp=ingp, cfg=cfg_model, iteration=iteration, beta=beta)
    img_gauss = render_pkg["render"]
    p = compute_psnr(img_gauss, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_gauss, os.path.join(out_dir, "125_gauss.png"))
    # Restore
    hash_enc.embeddings.data.copy_(embeddings_orig)

# =====================================================================
# 3. Hash features only (zero per-Gaussian features)
# =====================================================================
print("\n3. Hash features only (per-Gaussian zeroed)")
with torch.no_grad():
    # Save original per-Gaussian features
    gauss_feats_orig = gaussians._gaussian_features.data.clone()
    # Zero per-Gaussian
    gaussians._gaussian_features.data.zero_()
    render_pkg = render(cam, gaussians, pipe, bg_color,
                        ingp=ingp, cfg=cfg_model, iteration=iteration, beta=beta)
    img_hash = render_pkg["render"]
    p = compute_psnr(img_hash, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_hash, os.path.join(out_dir, "125_hash.png"))
    # Restore
    gaussians._gaussian_features.data.copy_(gauss_feats_orig)

# =====================================================================
# 4 & 5. Bake and render
# =====================================================================
del ingp  # free VRAM

if not _cli_args.skip_bake:
    print("\n4. Baking model...")
    import subprocess
    bake_cmd = [
        sys.executable, "scripts/bake_hybrid.py",
        "--model_path", model_path,
        "--texture", "shared",
    ]
    if _cli_args.ss is not None:
        bake_cmd += ["--ss", str(_cli_args.ss)]
    result = subprocess.run(bake_cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print(f"  Bake failed:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print("  Bake complete")
else:
    print("\n4. Skipping bake (--skip_bake), using existing baked/ files")

# Load baked PLY
baked_dir = os.path.join(model_path, "baked")
baked_ply = os.path.join(baked_dir, "baked.ply")
gaussians.load_ply(baked_ply)
gaussians.active_sh_degree = 3
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
N = len(gaussians.get_xyz)
print(f"  Baked PLY loaded: {N} Gaussians")

kernel_name = getattr(args, 'kernel', 'gaussian')
kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
kernel_type = kernel_map.get(kernel_name, 0)
print(f"  Kernel: {kernel_name} (type={kernel_type})")

from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

def render_baked_view(cam, gaussians, residual_textures=None):
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
        prefiltered=False, debug=False, beta=beta,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    result = rasterizer(
        means3D=gaussians.get_xyz, means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
        opacities=gaussians.get_opacity, shs=gaussians.get_features,
        scales=gaussians.get_scaling, rotations=gaussians.get_rotation,
        kernel_type=kernel_type, shapes=gaussians.get_shape,
        residual_textures=residual_textures,
    )
    return result[0]

# Load residual textures
tex_path = os.path.join(baked_dir, "residual_textures.pt")
residual_tex = torch.load(tex_path).cuda()
residual_flat = residual_tex.half().view(N, -1).contiguous()
print(f"  Residual textures: {residual_tex.shape}")

# 4a. SH + 48D residual
print("\n4a. Baked SH + 48D residual")
with torch.no_grad():
    img_baked = render_baked_view(cam, gaussians, residual_textures=residual_flat)
    p = compute_psnr(img_baked, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_baked, os.path.join(out_dir, "125_baked.png"))

# 4b. SH only
print("\n4b. Baked SH only (no residual)")
with torch.no_grad():
    img_sh = render_baked_view(cam, gaussians)
    p = compute_psnr(img_sh, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_sh, os.path.join(out_dir, "125_sh_only.png"))

# 4c. Residual texture only (zero base SH)
print("\n4c. Residual texture only (base SH zeroed)")
with torch.no_grad():
    dc_orig = gaussians._features_dc.data.clone()
    rest_orig = gaussians._features_rest.data.clone()
    gaussians._features_dc.data.zero_()
    gaussians._features_rest.data.zero_()
    img_tex = render_baked_view(cam, gaussians, residual_textures=residual_flat)
    p = compute_psnr(img_tex, gt)
    print(f"  PSNR: {p:.2f} dB")
    save_image(img_tex, os.path.join(out_dir, "125_tex_only.png"))
    gaussians._features_dc.data.copy_(dc_orig)
    gaussians._features_rest.data.copy_(rest_orig)

print(f"\nAll images saved to {out_dir}")
