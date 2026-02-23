#!/usr/bin/env python3
"""
One-view comparison: training renderer (hash+MLP) vs baked renderer (texture).

Renders a single test camera with both pipelines and saves:
  - training.png: the per-pixel hash+MLP output
  - baked.png: the baked texture lookup output
  - diff_10x.png: |training - baked| * 10
  - gt.png: ground truth

Usage:
    python scripts/compare_one_view.py \
        --model_path outputs/nerf_synthetic/chair/3D_SH_res/betscaled \
        --cam_idx 0
"""

import os, sys, pickle, glob, math, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.render_utils import save_img_u8
from utils.image_utils import psnr


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--cam_idx", type=int, default=0)
    a = parser.parse_args()

    out_dir = os.path.join(a.model_path, "comparisons", "one_view")
    os.makedirs(out_dir, exist_ok=True)

    # --- Load model ---
    with open(os.path.join(a.model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = a.model_path
    args.eval = True
    config_yaml_path = os.path.join(a.model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    if a.iteration == -1:
        ngp_files = glob.glob(os.path.join(a.model_path, "ngp_*.pth"))
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        a.iteration = max(iterations)

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(a.model_path, a.iteration)
    ingp.set_active_levels(a.iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=a.iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    cam = scene.getTestCameras()[a.cam_idx]
    gt = cam.original_image[:3].cuda()
    bg = torch.zeros(3, device="cuda")

    # --- 1. Training renderer (hash+MLP per pixel) ---
    print(f"\n[1] Training renderer (hash+MLP)...")
    with torch.no_grad():
        result = render(cam, gaussians, pipe, bg, ingp=ingp, iteration=a.iteration,
                        cfg=cfg, beta=cfg.surfel.tg_beta, is_training=False)
    img_train = result["render"].clamp(0, 1)
    psnr_train = psnr(img_train, gt).mean().item()
    print(f"    PSNR vs GT: {psnr_train:.2f} dB")

    # --- 2. Baked renderer (texture lookup) ---
    print(f"[2] Baked renderer (texture lookup)...")
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    baked_dir = os.path.join(a.model_path, "baked")
    baked_ply = os.path.join(baked_dir, "baked.ply")
    meta_path = os.path.join(baked_dir, "bake_meta.json")

    # Load baked Gaussians
    baked_gs = GaussianModel(dataset.sh_degree)
    baked_gs.load_ply(baked_ply)
    baked_gs.active_sh_degree = 3
    baked_gs.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, 'kernel'):
        baked_gs.kernel_type = args.kernel

    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(args, 'kernel', 'gaussian'), 0)

    # Load residual textures
    with open(meta_path) as f:
        bake_meta = json.load(f)
    tex_mode = bake_meta.get("texture_mode", "shared")

    residual_textures = None
    atlas_texture = None
    atlas_rects = None
    atlas_width = 0
    N = len(baked_gs.get_xyz)

    if tex_mode == "shared":
        tex = torch.load(os.path.join(baked_dir, "residual_textures.pt")).cuda()
        residual_textures = tex.view(N, -1).contiguous()
        print(f"    Shared textures: {list(tex.shape)}")
    elif tex_mode == "atlas":
        atlas_tex = torch.load(os.path.join(baked_dir, "atlas_texture.pt")).cuda()
        atlas_width = atlas_tex.shape[1]
        atlas_texture = atlas_tex.reshape(-1).contiguous()
        atlas_rects = torch.load(os.path.join(baked_dir, "atlas_rects.pt")).cuda().contiguous()
        print(f"    Atlas: {atlas_tex.shape[0]}x{atlas_tex.shape[1]}")

    shapes = None
    if kernel_type > 0 and hasattr(baked_gs, '_shape') and baked_gs._shape is not None and baked_gs._shape.numel() > 0:
        shapes = baked_gs.get_shape

    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=baked_gs.active_sh_degree, campos=cam.camera_center,
        prefiltered=False, debug=False, beta=cfg.surfel.tg_beta,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    with torch.no_grad():
        color, radii, _ = rasterizer(
            means3D=baked_gs.get_xyz,
            means2D=torch.zeros_like(baked_gs.get_xyz[:, :2]),
            opacities=baked_gs.get_opacity,
            shs=baked_gs.get_features,
            scales=baked_gs.get_scaling,
            rotations=baked_gs.get_rotation,
            shapes=shapes, kernel_type=kernel_type,
            residual_textures=residual_textures,
            atlas_texture=atlas_texture,
            atlas_rects=atlas_rects,
            atlas_width=atlas_width,
        )
    img_baked = color.clamp(0, 1)
    psnr_baked = psnr(img_baked, gt).mean().item()
    print(f"    PSNR vs GT: {psnr_baked:.2f} dB")

    # --- 3. SH-only (no residual) ---
    print(f"[3] SH-only (no residual)...")
    with torch.no_grad():
        color_sh, _, _ = rasterizer(
            means3D=baked_gs.get_xyz,
            means2D=torch.zeros_like(baked_gs.get_xyz[:, :2]),
            opacities=baked_gs.get_opacity,
            shs=baked_gs.get_features,
            scales=baked_gs.get_scaling,
            rotations=baked_gs.get_rotation,
            shapes=shapes, kernel_type=kernel_type,
        )
    img_sh = color_sh.clamp(0, 1)
    psnr_sh = psnr(img_sh, gt).mean().item()
    print(f"    PSNR vs GT: {psnr_sh:.2f} dB")

    # --- Compare ---
    diff = (img_train - img_baked).abs()
    psnr_tb = psnr(img_train, img_baked).mean().item()

    print(f"\n{'='*60}")
    print(f"Camera: {cam.image_name}")
    print(f"{'='*60}")
    print(f"  Training (MLP)  vs GT:  {psnr_train:.2f} dB")
    print(f"  Baked (texture) vs GT:  {psnr_baked:.2f} dB")
    print(f"  SH-only         vs GT:  {psnr_sh:.2f} dB")
    print(f"  Training vs Baked:      {psnr_tb:.2f} dB")
    print(f"  Residual adds:  training +{psnr_train - psnr_sh:.2f} dB, baked +{psnr_baked - psnr_sh:.2f} dB")

    # Save
    def save(img, path):
        save_img_u8(img.permute(1, 2, 0).cpu().numpy(), path)

    save(gt, os.path.join(out_dir, "gt.png"))
    save(img_train, os.path.join(out_dir, "training.png"))
    save(img_baked, os.path.join(out_dir, "baked.png"))
    save(img_sh, os.path.join(out_dir, "sh_only.png"))
    save((diff * 10).clamp(0, 1), os.path.join(out_dir, "diff_train_vs_baked_10x.png"))
    save((img_train - gt).abs().mul(10).clamp(0, 1), os.path.join(out_dir, "diff_train_vs_gt_10x.png"))
    save((img_baked - gt).abs().mul(10).clamp(0, 1), os.path.join(out_dir, "diff_baked_vs_gt_10x.png"))

    print(f"\nImages saved to: {out_dir}")


if __name__ == "__main__":
    main()
