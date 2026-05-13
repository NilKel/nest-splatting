#!/usr/bin/env python3
"""Render a custom camera trajectory through the CUDA baked renderer → MP4.

Loads a baked model (PLY + atlas + bake_meta.json) and a cameras.json (same
format as the training cameras.json — e.g. the output of
generate_circle_cameras.py) and renders each view via the
`diff_surfel_bake_render` submodule. Frames are saved as PNGs and assembled
into an MP4 via ffmpeg.

Usage:
    python scripts/render_baked_trajectory.py \\
        --model_path outputs/mip_360/bicycle/.../JBT3_..._3k \\
        --cameras    outputs/mip_360/bicycle/.../cameras_circle.json \\
        --output     /tmp/bicycle_circle.mp4 \\
        --fps 30

If --output ends in a directory, only PNGs are written (skip ffmpeg).
"""
import os
import sys
import json
import math
import pickle
import shutil
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from hash_encoder.config import Config  # noqa: E402
from arguments import ModelParams  # noqa: E402
from utils.render_utils import save_img_u8  # noqa: E402
from utils.graphics_utils import focal2fov  # noqa: E402
from scripts.render_baked import render_baked, load_training_config, _make_sv_eval  # noqa: E402


def build_camera(entry: dict, idx: int) -> Camera:
    """Convert one cameras.json entry into a Camera with a black dummy image."""
    pos = np.array(entry["position"], dtype=np.float64)
    R = np.array(entry["rotation"], dtype=np.float64)  # c2w, cols = [right, up, forward]
    T = -R.T @ pos  # w2c translation
    W, H = int(entry["width"]), int(entry["height"])
    fx, fy = float(entry["fx"]), float(entry["fy"])
    fovx = focal2fov(fx, W)
    fovy = focal2fov(fy, H)
    dummy = torch.zeros(3, H, W)
    return Camera(
        colmap_id=idx,
        R=R, T=T,
        FoVx=fovx, FoVy=fovy,
        image=dummy, gt_alpha_mask=None,
        image_name=entry.get("img_name", f"view_{idx:04d}"),
        uid=idx,
    )


def main():
    p = ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--cameras", required=True,
                   help="Custom cameras.json (e.g. from generate_circle_cameras.py)")
    p.add_argument("--output", required=True,
                   help="Output MP4 path. If it ends in '/', writes PNGs only.")
    p.add_argument("--baked_dir", default=None,
                   help="Bake directory (default: <model_path>/baked_atlas)")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=18, help="ffmpeg x264 quality (lower=better)")
    p.add_argument("--keep_pngs", action="store_true",
                   help="Don't delete the intermediate PNG directory after encoding.")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Render at 1/downscale resolution (e.g. 2 for half-size).")
    p.add_argument("--untextured", action="store_true",
                   help="Skip the MLP-residual atlas — render baked SH only "
                        "(shows the per-Gaussian mean color, no high-frequency detail).")
    args = p.parse_args()

    # ---- Mirror render_baked.py's setup ----------------------------------------
    train_args = load_training_config(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    cfg_yaml = os.path.join(args.model_path, "config.yaml")
    cfg_model = Config(cfg_yaml if os.path.exists(cfg_yaml) else train_args.yaml)

    iteration = args.iteration
    if iteration == -1:
        import glob as _g
        ngp_files = _g.glob(os.path.join(args.model_path, "ngp_*.pth"))
        if ngp_files:
            iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                            for f in ngp_files)

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(train_args)
    gaussians = GaussianModel(dataset.sh_degree)

    baked_dir = args.baked_dir or os.path.join(args.model_path, "baked_atlas")
    baked_ply = os.path.join(baked_dir, "baked.ply")
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha

    kernel_name = getattr(train_args, "kernel", "gaussian")
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = kernel_name
    kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(kernel_name, 0)

    print(f"[render] {len(gaussians.get_xyz):,} Gaussians from {baked_ply}")

    # ---- Atlas + meta ----------------------------------------------------------
    meta_path = os.path.join(baked_dir, "bake_meta.json")
    bake_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            bake_meta = json.load(f)
    texture_mode = bake_meta.get("texture_mode", "atlas")

    from diff_surfel_bake_render import set_activation_bias, set_compact_mult, set_residual_mode
    sh_bias  = float(bake_meta.get("sh_bias",  getattr(train_args, "activation_bias", [0.5, 0.0])[0]))
    res_bias = float(bake_meta.get("res_bias", getattr(train_args, "activation_bias", [0.5, 0.0])[1]))
    set_activation_bias(sh_bias, res_bias)
    set_compact_mult(float(bake_meta.get("compact_mult", 1.0)))
    set_residual_mode(int(bake_meta.get("residual_mode", 0)))

    residual_textures = None
    atlas_texture = None
    atlas_rects = None
    atlas_width = 0
    if texture_mode == "atlas":
        atlas_tex_path = os.path.join(baked_dir, "atlas_texture.pt")
        atlas_rects_path = os.path.join(baked_dir, "atlas_rects.pt")
        if os.path.exists(atlas_tex_path) and os.path.exists(atlas_rects_path):
            atlas_tex = torch.load(atlas_tex_path).cuda()
            if atlas_tex.dtype == torch.uint8:
                atlas_scale  = float(bake_meta.get("atlas_scale",  1.0))
                atlas_offset = float(bake_meta.get("atlas_offset", 0.0))
                rgb = atlas_tex[..., :3] if atlas_tex.shape[-1] == 4 else atlas_tex
                atlas_tex = (rgb.to(torch.float32) / 255.0 * atlas_scale
                             + atlas_offset).to(torch.float16).contiguous()
            atlas_width = atlas_tex.shape[1]
            atlas_texture = atlas_tex.reshape(-1).contiguous()
            atlas_rects = torch.load(atlas_rects_path).cuda().contiguous()
            print(f"[render] atlas {atlas_tex.shape}, rects {list(atlas_rects.shape)}")

    if texture_mode == "shared":
        tex_path = os.path.join(baked_dir, "residual_textures.pt")
        if os.path.exists(tex_path):
            t = torch.load(tex_path).cuda()
            residual_textures = t.view(len(gaussians.get_xyz), -1).contiguous()

    if args.untextured:
        residual_textures = None
        atlas_texture = None
        atlas_rects = None
        atlas_width = 0
        print("[render] --untextured: dropping atlas, rendering SH-only baked color")

    # Spherical-Beta lobes
    sb_params, sb_number = None, int(bake_meta.get("sb_number", 0))
    sb_file = bake_meta.get("sb_params_file")
    if sb_number > 0 and sb_file:
        sb_path = os.path.join(baked_dir, sb_file)
        if os.path.exists(sb_path):
            sb_tensor = torch.load(sb_path).float().cuda().contiguous()
            sb_params = sb_tensor.reshape(-1).contiguous()
            print(f"[render] SB params K={sb_number} loaded")

    # SV (per-frame torch fake-DC injection)
    sv_eval_cb = None
    if bake_meta.get("feature_mode", "sh") == "SV":
        gaussians.feature_mode = "SV"
        sv_eval_cb = _make_sv_eval(gaussians)
        print("[render] feature_mode=SV (per-frame torch fake-DC)")

    aabb_str = getattr(train_args, "aabb", "rect")
    aabb_map = {"square": 0, "adr_only": 1, "rect": 2, "adr": 3, "adr_rect": 3}
    aabb_mode = aabb_map.get(aabb_str, 2)

    beta = cfg_model.surfel.tg_beta
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # ---- Trajectory cameras ----------------------------------------------------
    with open(args.cameras) as f:
        cam_entries = json.load(f)
    if args.downscale != 1.0:
        s = float(args.downscale)
        for e in cam_entries:
            e["width"]  = int(round(e["width"]  / s))
            e["height"] = int(round(e["height"] / s))
            e["fx"]     = e["fx"] / s
            e["fy"]     = e["fy"] / s
    cameras = [build_camera(e, i) for i, e in enumerate(cam_entries)]
    print(f"[render] {len(cameras)} trajectory cameras @ {cameras[0].image_width}x{cameras[0].image_height}")

    # ---- Render loop -----------------------------------------------------------
    out_is_dir = args.output.endswith(os.sep) or os.path.isdir(args.output)
    if out_is_dir:
        png_dir = args.output.rstrip(os.sep)
        os.makedirs(png_dir, exist_ok=True)
    else:
        png_dir = str(Path(args.output).with_suffix("")) + "_frames"
        os.makedirs(png_dir, exist_ok=True)

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            out = render_baked(
                cam, gaussians, None, bg,
                residual_textures=residual_textures,
                beta=beta, kernel_type=kernel_type,
                atlas_texture=atlas_texture, atlas_rects=atlas_rects,
                atlas_width=atlas_width, aabb_mode=aabb_mode,
                sb_params=sb_params, sb_number=sb_number,
                sv_eval=sv_eval_cb,
            )
            img = out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            save_img_u8(img, os.path.join(png_dir, f"{i:05d}.png"))
            if (i + 1) % 10 == 0 or (i + 1) == len(cameras):
                print(f"  {i+1}/{len(cameras)}")

    if out_is_dir:
        print(f"[render] wrote {len(cameras)} PNGs to {png_dir}")
        return

    # ---- ffmpeg ----------------------------------------------------------------
    if shutil.which("ffmpeg") is None:
        print("[render] ffmpeg not on PATH — PNGs in", png_dir)
        return
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", os.path.join(png_dir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(args.crf), "-preset", "slow",
        # libx264 needs even dims for yuv420p — pad if odd.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        args.output,
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[render] {args.output}")
    if not args.keep_pngs:
        shutil.rmtree(png_dir)


if __name__ == "__main__":
    main()
