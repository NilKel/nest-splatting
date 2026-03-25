#!/usr/bin/env python3
"""
Render baked model: Mean SH + Residual Textures (shared or atlas mode).

Loads baked.ply (with Mean SH) and residual textures, renders using
the diff_surfel_bake_render submodule (forward-only, no backward).

Usage:
    python scripts/render_baked.py --model_path outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma
    python scripts/render_baked.py --model_path ... --texture atlas
"""

import os
import sys
import json
import math
import time
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        return Namespace(**args_dict)
    raise FileNotFoundError(f"No training config found in {model_path}")


def render_baked(viewpoint_camera, gaussians, pipe, background,
                 residual_textures=None, beta=0.0, kernel_type=0,
                 atlas_texture=None, atlas_rects=None, atlas_width=0,
                 aabb_mode=3):
    """Render using diff_surfel_bake_render submodule (SH + residual textures)."""
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        beta=beta,
        aabb_mode=aabb_mode,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D[:, :2], requires_grad=False)
    opacity = gaussians.get_opacity

    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features

    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    color, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        scales=scales,
        rotations=rotations,
        shapes=shapes,
        kernel_type=kernel_type,
        residual_textures=residual_textures,
        atlas_texture=atlas_texture,
        atlas_rects=atlas_rects,
        atlas_width=atlas_width,
    )

    return {"render": color}


def evaluate_mode(test_cameras, gaussians, bg_color, beta, kernel_type,
                  residual_textures, save_dir, num_warmup, num_benchmark,
                  atlas_texture=None, atlas_rects=None, atlas_width=0,
                  aabb_mode=3):
    """Render all test views, compute metrics, benchmark FPS. Save images to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    psnrs, l1s, ssims = [], [], []
    with torch.no_grad():
        for cam in test_cameras:
            result = render_baked(cam, gaussians, None, bg_color,
                                 residual_textures=residual_textures,
                                 beta=beta, kernel_type=kernel_type,
                                 atlas_texture=atlas_texture,
                                 atlas_rects=atlas_rects,
                                 atlas_width=atlas_width,
                                 aabb_mode=aabb_mode)
            rendered = result["render"]
            gt = cam.original_image[:3].cuda()

            psnrs.append(psnr(rendered, gt).mean().item())
            l1s.append(l1_loss(rendered, gt).item())
            ssims.append(ssim(rendered, gt).item())

            img_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            save_img_u8(img_np, os.path.join(save_dir, f"{cam.image_name}.png"))

    # FPS benchmark
    with torch.no_grad():
        for i in range(num_warmup):
            cam = test_cameras[i % len(test_cameras)]
            _ = render_baked(cam, gaussians, None, bg_color,
                            residual_textures=residual_textures,
                            beta=beta, kernel_type=kernel_type,
                            atlas_texture=atlas_texture,
                            atlas_rects=atlas_rects,
                            atlas_width=atlas_width)
        torch.cuda.synchronize()

        times = []
        for i in range(num_benchmark):
            cam = test_cameras[i % len(test_cameras)]
            torch.cuda.synchronize()
            t0 = time.time()
            _ = render_baked(cam, gaussians, None, bg_color,
                            residual_textures=residual_textures,
                            beta=beta, kernel_type=kernel_type,
                            atlas_texture=atlas_texture,
                            atlas_rects=atlas_rects,
                            atlas_width=atlas_width)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    fps = 1.0 / np.mean(times)
    return {
        "psnr": float(np.mean(psnrs)),
        "l1": float(np.mean(l1s)),
        "ssim": float(np.mean(ssims)),
        "fps": float(fps),
        "ms_per_frame": float(np.mean(times) * 1000),
    }


def main():
    parser = ArgumentParser(description="Render baked model")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (-1 = latest)")
    parser.add_argument("--baked_dir", type=str, default=None, help="Baked output directory (default: model_path/baked/)")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_benchmark", type=int, default=100)
    parser.add_argument("--texture", choices=["shared", "atlas", "auto"], default="auto",
                        help="Texture mode: auto-detects from bake_meta.json")
    render_args = parser.parse_args()

    # Load training config
    args = load_training_config(render_args.model_path)
    args.model_path = render_args.model_path
    args.eval = True

    config_yaml_path = os.path.join(render_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    # Auto-detect iteration
    iteration = render_args.iteration
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(render_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)

    # Setup model from baked PLY
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)

    baked_dir = render_args.baked_dir or os.path.join(render_args.model_path, "baked_atlas")
    baked_ply = os.path.join(baked_dir, "baked.ply")

    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    kernel_name = getattr(args, 'kernel', 'gaussian')
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = kernel_name
    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(kernel_name, 0)

    N = len(gaussians.get_xyz)
    print(f"[RENDER] Loaded {N:,} Gaussians from {baked_ply}")

    # Auto-detect texture mode from metadata
    meta_path = os.path.join(baked_dir, "bake_meta.json")
    texture_mode = render_args.texture
    if texture_mode == "auto" and os.path.exists(meta_path):
        with open(meta_path) as f:
            bake_meta = json.load(f)
        texture_mode = bake_meta.get("texture_mode", "shared")
        print(f"[RENDER] Auto-detected texture mode: {texture_mode}")
    elif texture_mode == "auto":
        texture_mode = "shared"

    # Load textures based on mode
    residual_textures = None
    atlas_texture = None
    atlas_rects = None
    atlas_width = 0

    if texture_mode == "atlas":
        atlas_tex_path = os.path.join(baked_dir, "atlas_texture.pt")
        atlas_rects_path = os.path.join(baked_dir, "atlas_rects.pt")
        if os.path.exists(atlas_tex_path) and os.path.exists(atlas_rects_path):
            atlas_tex = torch.load(atlas_tex_path).cuda()  # [H, W, 3] half
            atlas_width = atlas_tex.shape[1]
            atlas_texture = atlas_tex.reshape(-1).contiguous()  # [H*W*3] half flat
            atlas_rects = torch.load(atlas_rects_path).cuda().contiguous()  # [N, 4] float
            print(f"[RENDER] Loaded atlas: {atlas_tex.shape[0]}x{atlas_tex.shape[1]}, "
                  f"rects: {list(atlas_rects.shape)}")
            atlas_mb = atlas_tex.nelement() * 2 / 1024 / 1024
            print(f"[RENDER] Atlas memory: {atlas_mb:.1f} MB")
        else:
            print(f"[RENDER] Atlas files not found, falling back to shared mode")
            texture_mode = "shared"

    if texture_mode == "shared":
        tex_path = os.path.join(baked_dir, "residual_textures.pt")
        if os.path.exists(tex_path):
            residual_tex = torch.load(tex_path).cuda()
            residual_textures = residual_tex.view(N, -1).contiguous()  # [N, gs*gs*dim] FP16
            residual_dim = residual_tex.shape[-1] if residual_tex.dim() >= 3 else 3
            print(f"[RENDER] Loaded shared textures: {list(residual_tex.shape)}, "
                  f"residual_dim={residual_dim}, total per Gaussian={residual_textures.shape[1]}")
        else:
            print(f"[RENDER] No residual_textures.pt found, SH-only mode")

    # Load test cameras (Scene overwrites gaussians' PLY, so we reload baked PLY after)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()

    # CRITICAL: Scene() just overwrote the baked PLY with the training PLY.
    # Reload baked PLY to restore the baked mean SH.
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    print(f"[RENDER] Reloaded baked PLY after Scene init: {N:,} Gaussians")
    print(f"[RENDER] {len(test_cameras)} test cameras")

    beta = cfg_model.surfel.tg_beta
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Map aabb string to int
    aabb_str = getattr(args, 'aabb', 'rect')
    aabb_map = {'square': 0, 'adr_only': 1, 'rect': 2, 'adr': 3, 'adr_rect': 3}
    aabb_mode = aabb_map.get(aabb_str, 2)
    print(f"[RENDER] aabb_mode={aabb_mode} (from '{aabb_str}')")

    render_dir = os.path.join(baked_dir, "renders")
    all_metrics = {}

    # --- SH only ---
    sh_dir = os.path.join(render_dir, "sh_only")
    print(f"\n[RENDER] Rendering SH only -> {sh_dir}")
    sh_metrics = evaluate_mode(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        residual_textures=None,
        save_dir=sh_dir,
        num_warmup=render_args.num_warmup,
        num_benchmark=render_args.num_benchmark,
        aabb_mode=aabb_mode,
    )
    all_metrics["sh_only"] = sh_metrics
    print(f"  PSNR: {sh_metrics['psnr']:.2f} dB  |  SSIM: {sh_metrics['ssim']:.4f}  |  "
          f"L1: {sh_metrics['l1']:.4f}  |  FPS: {sh_metrics['fps']:.1f}")

    # --- SH + Residual ---
    has_residual = residual_textures is not None or atlas_texture is not None
    if has_residual:
        mode_name = f"sh_{texture_mode}"
        res_dir = os.path.join(render_dir, mode_name)
        print(f"\n[RENDER] Rendering SH + Residual ({texture_mode}) -> {res_dir}")
        res_metrics = evaluate_mode(
            test_cameras, gaussians, bg_color, beta, kernel_type,
            residual_textures=residual_textures,
            save_dir=res_dir,
            num_warmup=render_args.num_warmup,
            num_benchmark=render_args.num_benchmark,
            atlas_texture=atlas_texture,
            atlas_rects=atlas_rects,
            atlas_width=atlas_width,
            aabb_mode=aabb_mode,
        )
        all_metrics[mode_name] = res_metrics
        print(f"  PSNR: {res_metrics['psnr']:.2f} dB  |  SSIM: {res_metrics['ssim']:.4f}  |  "
              f"L1: {res_metrics['l1']:.4f}  |  FPS: {res_metrics['fps']:.1f}")

    # Save metrics JSON
    metrics_path = os.path.join(render_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[RENDER] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
