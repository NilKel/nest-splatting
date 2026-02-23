#!/usr/bin/env python3
"""
Benchmark all baked residual rendering modes on ALL test views.
Reports per-mode: average PSNR, FPS (warmup + timed iterations).
Also benchmarks the training neural renderer for reference.
"""
import os, sys, math, glob, pickle, time, torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from utils.image_utils import psnr

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
baked_dir = os.path.join(model_path, "baked")

WARMUP = 5
TIMED = 50
SAVE_VIEW = 125
out_dir = os.path.join(model_path, "debugtex")
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
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
test_cameras = scene.getTestCameras()
print(f"{len(test_cameras)} test views")

beta = cfg_model.surfel.tg_beta
kernel_name = getattr(args, 'kernel', 'gaussian')
kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
kernel_type = kernel_map.get(kernel_name, 0)
bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

def compute_psnr(rendered, gt):
    return psnr(rendered.clamp(0, 1), gt).mean().item()

def save_image(tensor, path):
    img = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

# =====================================================================
# 1. Neural renderer (training) — PSNR + FPS reference
# =====================================================================
print("\n" + "=" * 60)
print("1. Neural renderer (training)")
print("=" * 60)
ingp = INGP(cfg_model, args=args).to('cuda')
ingp.load_model(model_path, iteration)
ingp.set_active_levels(iteration)

from gaussian_renderer import render
pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.0)

# PSNR over all test views
neural_psnrs = []
with torch.no_grad():
    for i, cam in enumerate(test_cameras):
        gt = cam.original_image[:3].cuda()
        render_pkg = render(cam, gaussians, pipe, bg_color,
                            ingp=ingp, cfg=cfg_model, iteration=iteration)
        img = render_pkg["render"]
        p = compute_psnr(img, gt)
        neural_psnrs.append(p)
        if i == SAVE_VIEW:
            save_image(gt, os.path.join(out_dir, "0_gt.png"))
            save_image(img, os.path.join(out_dir, "1_neural.png"))
neural_avg = np.mean(neural_psnrs)
print(f"  Avg PSNR: {neural_avg:.2f} dB")

# FPS: use first camera repeatedly
fps_cam = test_cameras[0]
with torch.no_grad():
    for _ in range(WARMUP):
        render(fps_cam, gaussians, pipe, bg_color, ingp=ingp, cfg=cfg_model, iteration=iteration)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMED):
        render(fps_cam, gaussians, pipe, bg_color, ingp=ingp, cfg=cfg_model, iteration=iteration)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
neural_fps = TIMED / (t1 - t0)
print(f"  FPS: {neural_fps:.1f}")

# Free neural resources
del ingp
torch.cuda.empty_cache()

# =====================================================================
# Load baked PLY for all baked modes
# =====================================================================
baked_ply = os.path.join(baked_dir, "baked.ply")
gaussians.load_ply(baked_ply)
gaussians.active_sh_degree = 3
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
N = len(gaussians.get_xyz)
print(f"\nBaked PLY loaded: {N} Gaussians")

from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

def render_baked_view(cam, gaussians, residual_textures=None,
                      atlas_texture=None, atlas_rects=None, atlas_width=0):
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
        atlas_texture=atlas_texture, atlas_rects=atlas_rects, atlas_width=atlas_width,
    )
    return result[0]

def benchmark_mode(name, render_kwargs, save_filename=None):
    """Run all test views for PSNR, then time FPS on first camera."""
    psnrs = []
    with torch.no_grad():
        for i, cam in enumerate(test_cameras):
            gt = cam.original_image[:3].cuda()
            img = render_baked_view(cam, gaussians, **render_kwargs)
            psnrs.append(compute_psnr(img, gt))
            if i == SAVE_VIEW and save_filename:
                save_image(img, os.path.join(out_dir, save_filename))

    # FPS
    fps_cam = test_cameras[0]
    with torch.no_grad():
        for _ in range(WARMUP):
            render_baked_view(fps_cam, gaussians, **render_kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            render_baked_view(fps_cam, gaussians, **render_kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    fps = TIMED / (t1 - t0)
    avg_psnr = np.mean(psnrs)
    print(f"  Avg PSNR: {avg_psnr:.2f} dB  |  FPS: {fps:.1f}")
    return avg_psnr, fps

results = {}
results["Neural MLP"] = (neural_avg, neural_fps)

# =====================================================================
# 2. SH only (no residual)
# =====================================================================
print("\n" + "=" * 60)
print("2. SH only (mean SH)")
print("=" * 60)
p, f = benchmark_mode("SH only", {}, "2_sh_only.png")
results["SH only"] = (p, f)

# =====================================================================
# 3. SH + 48D SH residual (shared 8×8)
# =====================================================================
tex_path = os.path.join(baked_dir, "residual_textures.pt")
residual_tex = torch.load(tex_path).cuda()
residual_dim = residual_tex.shape[-1] if residual_tex.dim() >= 3 else 3

if residual_dim == 48:
    print("\n" + "=" * 60)
    print("3. SH + 48D residual (shared 8×8)")
    print("=" * 60)
    residual_flat_48 = residual_tex.half().view(N, -1).contiguous()
    p, f = benchmark_mode("SH+48D shared", {"residual_textures": residual_flat_48}, "3_sh_48d_shared.png")
    results["SH + 48D shared"] = (p, f)

    # Derive 3D DC residual from 48D SH
    SH_C0 = 0.28209479177387814
    full_sh = residual_tex.float().view(N, 64, 48)
    dc_ply = gaussians._features_dc.data
    rest_ply = gaussians._features_rest.data
    mean_sh_ply = torch.cat([dc_ply, rest_ply], dim=1)
    mean_sh = mean_sh_ply.permute(0, 2, 1).reshape(N, 48)
    full_sh = full_sh + mean_sh.unsqueeze(1)

    dc_residual = torch.zeros(N, 64, 3, device='cuda')
    for ch in range(3):
        texel_dc = SH_C0 * full_sh[:, :, ch * 16] + 0.5
        base_dc = SH_C0 * mean_sh[:, ch * 16] + 0.5
        dc_residual[:, :, ch] = torch.clamp(texel_dc, min=0) - torch.clamp(base_dc.unsqueeze(1), min=0)
    dc_flat = dc_residual.view(N, 8, 8, 3).half().view(N, -1).contiguous()
else:
    dc_flat = residual_tex.half().view(N, -1).contiguous()

# =====================================================================
# 4. SH + 3D DC residual (shared 8×8)
# =====================================================================
print("\n" + "=" * 60)
print("4. SH + 3D DC residual (shared 8×8)")
print("=" * 60)
p, f = benchmark_mode("SH+3D shared", {"residual_textures": dc_flat}, "4_sh_3d_shared.png")
results["SH + 3D DC shared"] = (p, f)

# =====================================================================
# 5. SH + 3D DC residual (atlas)
# =====================================================================
atlas_tex_path = os.path.join(baked_dir, "atlas_texture.pt")
atlas_rects_path = os.path.join(baked_dir, "atlas_rects.pt")
if os.path.exists(atlas_tex_path) and os.path.exists(atlas_rects_path):
    print("\n" + "=" * 60)
    print("5. SH + 3D DC residual (atlas)")
    print("=" * 60)
    atlas_tex = torch.load(atlas_tex_path).cuda()
    atlas_w = atlas_tex.shape[1]
    atlas_flat = atlas_tex.reshape(-1).contiguous()
    atlas_rects = torch.load(atlas_rects_path).cuda().contiguous()
    print(f"  Atlas: {atlas_tex.shape[0]}x{atlas_w}, rects: {list(atlas_rects.shape)}")
    p, f = benchmark_mode("SH+3D atlas", {
        "atlas_texture": atlas_flat,
        "atlas_rects": atlas_rects,
        "atlas_width": atlas_w,
    }, "5_sh_3d_atlas.png")
    results["SH + 3D DC atlas"] = (p, f)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print(f"BENCHMARK SUMMARY — {len(test_cameras)} test views, FPS on {TIMED} iters")
print("=" * 60)
print(f"  {'Mode':<25s} {'PSNR (dB)':>10s} {'FPS':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*8}")
for name in ["Neural MLP", "SH + 48D shared", "SH + 3D DC shared", "SH + 3D DC atlas", "SH only"]:
    if name in results:
        psnr_val, fps_val = results[name]
        print(f"  {name:<25s} {psnr_val:>10.2f} {fps_val:>8.1f}")
