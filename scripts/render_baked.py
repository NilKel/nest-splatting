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


def _make_sv_eval(gaussians):
    """Per-frame SV → fake-SH-DC closure.

    Mirrors gaussian_renderer/_build_fake_shs_from_SV: evaluates the
    Spherical-Voronoi softmax mix in torch and packs the result into the
    SH-DC slot so the existing CUDA `computeColorFromSH` reproduces
    `relu(feat + sh_bias)` via `clamp(SH_C0 · fake_dc + sh_bias, 0)`.
    Option A — no CUDA changes; per-frame torch overhead ~1 ms/view on a 5090.
    """
    from gaussian_renderer import eval_voronoi_sv_feat
    SH_C0 = 0.28209479177387814

    sv_sites = gaussians._sv_sites
    sv_colors = gaussians._sv_colors
    sv_tau_raw = getattr(gaussians, '_sv_tau', None)
    sv_tau = (torch.exp(sv_tau_raw)
              if sv_tau_raw is not None and sv_tau_raw.numel() > 0
              else None)
    sv_dc_param = getattr(gaussians, '_sv_dc', None)
    sv_dc = (sv_dc_param
             if sv_dc_param is not None and sv_dc_param.numel() > 0
             else None)
    sv_mask = getattr(gaussians, '_sv_mask', None)
    apply_mask = (
        sv_mask is not None
        and (not getattr(gaussians, '_sv_training_flag', True))
        and sv_mask.shape[0] == sv_sites.shape[0]
    )
    sites_mask_tensor = sv_mask if apply_mask else None

    means3D = gaussians.get_xyz
    N = means3D.shape[0]
    M = (gaussians.active_sh_degree + 1) ** 2

    def _eval(viewpoint_camera):
        cam_center = viewpoint_camera.camera_center.to(means3D.device)
        view_dirs = means3D - cam_center.unsqueeze(0)
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        feat = eval_voronoi_sv_feat(
            sv_sites, sv_colors, view_dirs,
            sv_tau=sv_tau, sites_mask=sites_mask_tensor)
        if sv_dc is not None:
            feat = feat + sv_dc

        fake = torch.zeros((N, M, 3), dtype=feat.dtype, device=feat.device)
        fake[:, 0, :] = feat / SH_C0
        return fake.contiguous()

    return _eval


def render_baked(viewpoint_camera, gaussians, pipe, background,
                 residual_textures=None, beta=0.0, kernel_type=0,
                 atlas_texture=None, atlas_rects=None, atlas_width=0,
                 aabb_mode=3,
                 sb_params=None, sb_number=0,
                 sv_eval=None, final_relu=False):
    """Render using diff_surfel_bake_render submodule (SH + residual textures).

    `sv_eval`: optional `(viewpoint_camera) -> fake_shs` callback. When set
    (i.e. trained model used --feature SV), the returned tensor replaces
    `gaussians.get_features` for this frame so the existing SH path produces
    the SV softmax color via the SH-DC slot.
    """
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
    opacity = gaussians.get_opacity

    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features if sv_eval is None else sv_eval(viewpoint_camera)

    shapes = None
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    color, _ = rasterizer(
        means3D=means3D,
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
        sb_params=sb_params,
        sb_number=sb_number,
    )

    # `--method mixed` (residual_mode==2): per-pixel ReLU on the FINAL blended
    # color (the kernel emits signed per-Gauss color), matching training.
    if final_relu:
        color = torch.relu(color)

    return {"render": color}


def evaluate_mode(test_cameras, gaussians, bg_color, beta, kernel_type,
                  residual_textures, save_dir, num_warmup, num_benchmark,
                  atlas_texture=None, atlas_rects=None, atlas_width=0,
                  aabb_mode=3,
                  sb_params=None, sb_number=0,
                  sv_eval=None, final_relu=False):
    """Render all test views, compute metrics, benchmark FPS. Save images to save_dir.

    `sv_eval`: optional callback `(camera) -> fake_shs` for --feature SV models;
    bypasses the stored SH and injects per-frame Voronoi color via SH-DC slot.
    """
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
                                 aabb_mode=aabb_mode,
                                 sb_params=sb_params,
                                 sb_number=sb_number,
                                 sv_eval=sv_eval, final_relu=final_relu)
            rendered = result["render"]
            gt = cam.original_image[:3].cuda()

            psnrs.append(psnr(rendered, gt).mean().item())
            l1s.append(l1_loss(rendered, gt).item())
            ssims.append(ssim(rendered, gt).item())

            img_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            save_img_u8(img_np, os.path.join(save_dir, f"{cam.image_name}.png"))

    # FPS benchmark — CUDA event timing, no Python wall-clock noise.
    with torch.no_grad():
        for i in range(num_warmup):
            cam = test_cameras[i % len(test_cameras)]
            _ = render_baked(cam, gaussians, None, bg_color,
                            residual_textures=residual_textures,
                            beta=beta, kernel_type=kernel_type,
                            atlas_texture=atlas_texture,
                            atlas_rects=atlas_rects,
                            atlas_width=atlas_width,
                            aabb_mode=aabb_mode,
                            sb_params=sb_params,
                            sb_number=sb_number,
                            sv_eval=sv_eval, final_relu=final_relu)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        for i in range(num_benchmark):
            cam = test_cameras[i % len(test_cameras)]
            starts[i].record()
            _ = render_baked(cam, gaussians, None, bg_color,
                            residual_textures=residual_textures,
                            beta=beta, kernel_type=kernel_type,
                            atlas_texture=atlas_texture,
                            atlas_rects=atlas_rects,
                            atlas_width=atlas_width,
                            aabb_mode=aabb_mode,
                            sb_params=sb_params,
                            sb_number=sb_number,
                            sv_eval=sv_eval, final_relu=final_relu)
            ends[i].record()
        torch.cuda.synchronize()
        times_ms = [starts[i].elapsed_time(ends[i]) for i in range(num_benchmark)]
        mean_ms = float(np.mean(times_ms))

    fps = 1000.0 / mean_ms
    return {
        "psnr": float(np.mean(psnrs)),
        "l1": float(np.mean(l1s)),
        "ssim": float(np.mean(ssims)),
        "fps": float(fps),
        "ms_per_frame": mean_ms,
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
    gaussians.kernel_type2 = getattr(args, 'kernel2', None)
    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(kernel_name, 0)

    N = len(gaussians.get_xyz)
    print(f"[RENDER] Loaded {N:,} Gaussians from {baked_ply}")

    # Auto-detect texture mode from metadata, also pull activation biases,
    # Compact Box mult, feature mode, and SB lobe count.
    meta_path = os.path.join(baked_dir, "bake_meta.json")
    bake_meta = {}
    texture_mode = render_args.texture
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            bake_meta = json.load(f)
        if texture_mode == "auto":
            texture_mode = bake_meta.get("texture_mode", "shared")
            print(f"[RENDER] Auto-detected texture mode: {texture_mode}")
    elif texture_mode == "auto":
        texture_mode = "shared"

    # Call the CUDA device-global setters so the baked kernel matches training.
    from diff_surfel_bake_render import (set_activation_bias, set_compact_mult,
                                         set_residual_mode, set_untex_kernel)
    _sh_bias = float(bake_meta.get("sh_bias", getattr(args, 'activation_bias', [0.5, 0.0])[0]))
    _res_bias = float(bake_meta.get("res_bias", getattr(args, 'activation_bias', [0.5, 0.0])[1]))
    _compact_mult = float(bake_meta.get("compact_mult", 1.0))
    _residual_mode = int(bake_meta.get("residual_mode", 0))
    set_activation_bias(_sh_bias, _res_bias)
    set_compact_mult(_compact_mult)
    set_residual_mode(_residual_mode)
    # `--method mixed_3d --kernel2` override: bake_meta records the training-time
    # `args.kernel2`; if absent (older checkpoint / non-mixed_3d) fall back to -1
    # which disables the override and uses kernel_type for the untextured EWA half.
    _kmap2 = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4, 'nexel': 5}
    _k2_str = bake_meta.get("kernel2", None) or getattr(args, 'kernel2', None)
    _untex_kt = _kmap2.get(_k2_str, -1) if _k2_str else -1
    set_untex_kernel(_untex_kt)
    print(f"[RENDER] set_activation_bias(sh={_sh_bias}, res={_res_bias})  "
          f"set_compact_mult({_compact_mult})  set_residual_mode({_residual_mode})  "
          f"set_untex_kernel({_untex_kt}{' = '+_k2_str if _k2_str else ''})")

    # Load textures based on mode
    residual_textures = None
    atlas_texture = None
    atlas_rects = None
    atlas_width = 0

    if texture_mode == "atlas":
        atlas_tex_path = os.path.join(baked_dir, "atlas_texture.pt")
        atlas_rects_path = os.path.join(baked_dir, "atlas_rects.pt")
        if os.path.exists(atlas_tex_path) and os.path.exists(atlas_rects_path):
            atlas_tex = torch.load(atlas_tex_path).cuda()
            # uint8 RGBA on disk (new default) → dequantize back to FP16 RGB for
            # the CUDA kernel, which still expects Half. scale/offset come from
            # bake_meta.json and must survive the round-trip intact.
            if atlas_tex.dtype == torch.uint8:
                atlas_scale_meta  = float(bake_meta.get("atlas_scale",  1.0))
                atlas_offset_meta = float(bake_meta.get("atlas_offset", 0.0))
                if atlas_tex.shape[-1] == 4:
                    atlas_rgb = atlas_tex[..., :3]
                else:
                    atlas_rgb = atlas_tex
                atlas_tex = (atlas_rgb.to(torch.float32) / 255.0 * atlas_scale_meta
                             + atlas_offset_meta).to(torch.float16).contiguous()
                print(f"[RENDER] Dequantized uint8 atlas → FP16 "
                      f"(scale={atlas_scale_meta:.4f}, offset={atlas_offset_meta:.4f})")
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

    # Spherical-Beta params (loaded if the training feature_mode was 'beta').
    sb_params = None
    sb_number = int(bake_meta.get("sb_number", 0))
    sb_file = bake_meta.get("sb_params_file", None)
    if sb_number > 0 and sb_file is not None:
        sb_path = os.path.join(baked_dir, sb_file)
        if os.path.exists(sb_path):
            sb_tensor = torch.load(sb_path).float().cuda().contiguous()
            sb_params = sb_tensor.reshape(-1).contiguous()  # [N * K * 6] flat
            print(f"[RENDER] Loaded SB params: shape={list(sb_tensor.shape)}, "
                  f"K={sb_number} lobes, feature=beta")
        else:
            print(f"[RENDER] bake_meta referenced {sb_file} but file missing; SB disabled.")
            sb_number = 0

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

    # --feature SV: route per-frame fake-SH-DC eval through render_baked.
    # bake_meta records `feature_mode='SV'` + `sv_number`; load_ply has just
    # restored _sv_sites/_sv_colors/_sv_tau/_sv_dc onto the model.
    _feat_mode_render = bake_meta.get("feature_mode", "sh")
    gaussians.feature_mode = _feat_mode_render
    sv_eval_cb = None
    if _feat_mode_render == "SV":
        sv_K = int(bake_meta.get("sv_number", 0))
        if (not hasattr(gaussians, '_sv_sites')) or gaussians._sv_sites.numel() == 0:
            raise RuntimeError(
                f"bake_meta says feature_mode='SV' (K={sv_K}) but baked PLY "
                f"didn't restore _sv_sites — re-bake with the SV-aware save_ply.")
        print(f"[RENDER] feature_mode=SV: K={sv_K} (per-frame torch fake-SH-DC eval)")
        sv_eval_cb = _make_sv_eval(gaussians)

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
        sb_params=sb_params,
        sb_number=sb_number,
        sv_eval=sv_eval_cb,
        final_relu=(_residual_mode == 2),
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
            sb_params=sb_params,
            sb_number=sb_number,
            sv_eval=sv_eval_cb,
            final_relu=(_residual_mode == 2),
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
