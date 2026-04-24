#!/usr/bin/env python3
"""Compare training-time neural renderer vs baked renderer on unclamped raw tensors.

For a given checkpoint + baked atlas, renders each test view twice per mode:
  SH-only mode:
    neural: MLP weights zeroed, SB disabled  → output = ReLU(clamp(SH + sh_bias, 0) + 0 + res_bias)
    baked : atlas/residual disabled,  SB off → same formula (identical path)
  Texture-only mode:
    neural: sh_bias = -999 (SH → 0), SB off  → output = ReLU(0 + MLP_residual + res_bias)
    baked : sh_bias = -999 (SH → 0), SB off  → output = ReLU(0 + atlas_residual + res_bias)

SB is explicitly disabled on both sides so the comparison isolates:
  (a) SH path: should match to numerical noise.
  (b) Residual representation: neural MLP vs baked atlas = bake fidelity.

The pre-alpha-compositing ReLU is INSIDE the per-Gaussian loop in both kernels, but the
accumulated pixel value is Σ wᵢ·[unclamped non-negative term]. We save the raw FP32 output
tensor (shape [3, H, W], non-negative but UNBOUNDED — not clamped to [0, 1]) so diffs
show true unclamped error. PNGs saved separately are gamma/clipped only for visual inspection.

Usage:
    python scripts/compare_bake_vs_neural.py --model_path outputs/.../SBfastgs... \
        --num_views 4
"""

import os, sys, json, pickle, time, math, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8


def load_args(model_path):
    with open(os.path.join(model_path, "args.pkl"), "rb") as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    return args


def render_neural(viewpoint_camera, gaussians, ingp, bg, pipe, args, cfg,
                  decompose_mode, disable_sb=True):
    """Full neural render via gaussian_renderer. decompose_mode ∈ {None, 'sh_only', 'tex_only'}.
    If disable_sb, temporarily zero _sb_params so SB doesn't contaminate the isolated path."""
    from gaussian_renderer import render

    sb_backup = None
    if disable_sb and hasattr(gaussians, "_sb_params") and gaussians._sb_params.numel() > 0:
        sb_backup = gaussians._sb_params.detach().clone()
        with torch.no_grad():
            gaussians._sb_params.zero_()

    try:
        pkg = render(viewpoint_camera, gaussians, pipe, bg, ingp=ingp,
                     beta=cfg.surfel.tg_beta, cfg=cfg,
                     decompose_mode=decompose_mode,
                     aabb_mode=getattr(args, "aabb", "rect"),
                     is_training=False)
    finally:
        if sb_backup is not None:
            with torch.no_grad():
                gaussians._sb_params.copy_(sb_backup)

    return pkg["render"]


def render_bake(viewpoint_camera, gaussians, bg, kernel_type, aabb_mode,
                atlas_texture=None, atlas_rects=None, atlas_width=0,
                sb_params=None, sb_number=0):
    """Forward-only baked render. Pass atlas_texture=None for SH-only, or a tensor for SH+atlas."""
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    rs = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False,
        beta=0.0, aabb_mode=aabb_mode,
    )
    rasterizer = GaussianRasterizer(raster_settings=rs)

    shapes = None
    if kernel_type > 0 and hasattr(gaussians, "_shape") and gaussians._shape.numel() > 0:
        shapes = gaussians.get_shape

    color, _ = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=torch.zeros_like(gaussians.get_xyz[:, :2], requires_grad=False),
        opacities=gaussians.get_opacity,
        shs=gaussians.get_features,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shapes=shapes,
        kernel_type=kernel_type,
        residual_textures=None,
        atlas_texture=atlas_texture,
        atlas_rects=atlas_rects,
        atlas_width=atlas_width,
        sb_params=sb_params,
        sb_number=sb_number,
    )
    return color


def set_bias(sh_bias, res_bias):
    from diff_surfel_bake_render import set_activation_bias as _bake_set
    from diff_surfel_3D_sh_res import set_activation_bias as _train_set
    _bake_set(float(sh_bias), float(res_bias))
    _train_set(float(sh_bias), float(res_bias))


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    return {
        "l1":      float(d.mean().item()),
        "l2":      float(((a - b) ** 2).mean().sqrt().item()),
        "max":     float(d.max().item()),
        "rel_l1":  float((d.mean() / (a.abs().mean() + 1e-8)).item()),
        "a_mean":  float(a.mean().item()),
        "a_max":   float(a.max().item()),
        "b_mean":  float(b.mean().item()),
        "b_max":   float(b.max().item()),
    }


def load_atlas_for_cuda(baked_dir):
    """Load atlas from disk; dequantize uint8→FP16 if needed (mirrors render_baked.py)."""
    meta_path = os.path.join(baked_dir, "bake_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    atlas = torch.load(os.path.join(baked_dir, "atlas_texture.pt"),
                       weights_only=False, map_location="cuda")
    if atlas.dtype == torch.uint8:
        s = float(meta.get("atlas_scale", 1.0))
        o = float(meta.get("atlas_offset", 0.0))
        rgb = atlas[..., :3] if atlas.shape[-1] == 4 else atlas
        atlas = (rgb.to(torch.float32) / 255.0 * s + o).to(torch.float16).contiguous()
    atlas_width = atlas.shape[1]
    atlas_flat = atlas.reshape(-1).contiguous()
    rects = torch.load(os.path.join(baked_dir, "atlas_rects.pt"),
                       weights_only=False, map_location="cuda").contiguous()
    return atlas_flat, rects, atlas_width, meta


def main():
    p = ArgumentParser(description="Neural-render vs baked-render FP32 diff report")
    p.add_argument("--model_path", required=True)
    p.add_argument("--baked_dir", default=None)
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--output_dir", default="/tmp/bake_compare")
    p.add_argument("--iteration", type=int, default=-1)
    cmp_args = p.parse_args()

    args = load_args(cmp_args.model_path)

    cfg = Config(os.path.join(cmp_args.model_path, "config.yaml")
                 if os.path.exists(os.path.join(cmp_args.model_path, "config.yaml"))
                 else args.yaml)

    iteration = cmp_args.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(cmp_args.model_path, "ngp_*.pth"))
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                        for f in ngp_files)

    ingp = INGP(cfg, args=args).to("cuda")
    ingp.load_model(cmp_args.model_path, iteration)
    ingp.set_active_levels(iteration)

    temp_parser = ArgumentParser()
    dataset = ModelParams(temp_parser, sentinel=True).extract(args)

    # Build a simple pipe-config object with the fields gaussian_renderer expects.
    class _Pipe: pass
    pipe = _Pipe()
    pipe.debug = False
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.depth_ratio = float(getattr(args, "depth_ratio", 0.0))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(args, "kernel"):
        gaussians.kernel_type = args.kernel

    kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(args, "kernel", "gaussian"), 0)

    # AABB mode — must match what the training renderer derives from args.aabb,
    # otherwise neural + baked use different cutoff formulas for identical Gaussians.
    # See gaussian_renderer/__init__.py ~L1704 for the string→int mapping.
    _aabb_str = getattr(args, "aabb", "2dgs")
    _aabb_map = {
        "2dgs": 0, "adr_only": 1, "rect": 2,
        "adr": 3, "adr_rect": 3, "adrrect": 3,
        "beta": 4,
    }
    aabb_mode = _aabb_map.get(_aabb_str, 0)

    # Activation biases from training config.
    _act = getattr(args, "activation_bias", [0.5, 0.0])
    sh_bias_train = float(_act[0]); res_bias_train = float(_act[1])

    # Apply the fastgs compact_mult on both kernels.
    if getattr(args, "fastgs", False):
        from diff_surfel_bake_render import set_compact_mult as _bake_cmult
        from diff_surfel_3D_sh_res import set_compact_mult as _train_cmult
        _bake_cmult(float(getattr(args, "fastgs_mult", 1.0)))
        _train_cmult(float(getattr(args, "fastgs_mult", 1.0)))

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Load baked atlas (dequantize uint8 if needed).
    baked_dir = cmp_args.baked_dir or os.path.join(cmp_args.model_path, "baked_atlas")
    atlas_flat, rects, atlas_w, bake_meta = load_atlas_for_cuda(baked_dir)

    test_cams = scene.getTestCameras()[:cmp_args.num_views]
    print(f"[COMPARE] Comparing across {len(test_cams)} test views")
    print(f"[COMPARE] sh_bias={sh_bias_train}, res_bias={res_bias_train}, "
          f"aabb_mode={aabb_mode}, kernel={kernel_type}")

    os.makedirs(cmp_args.output_dir, exist_ok=True)

    agg = {"sh_only": [], "tex_only": []}

    for i, cam in enumerate(test_cams):
        name = cam.image_name

        # --- SH-only (normal bias, SB disabled on both sides) ---
        set_bias(sh_bias_train, res_bias_train)
        with torch.no_grad():
            img_n_sh = render_neural(cam, gaussians, ingp, bg, pipe, args, cfg,
                                     decompose_mode="sh_only", disable_sb=True)
            img_b_sh = render_bake(cam, gaussians, bg, kernel_type, aabb_mode,
                                   atlas_texture=None, atlas_rects=None, atlas_width=0,
                                   sb_params=None, sb_number=0)
        stats_sh = diff_stats(img_n_sh, img_b_sh)
        agg["sh_only"].append(stats_sh)

        # --- Texture-only, UNCLAMPED via shift-and-subtract trick ---
        # CUDA kernels apply `max(feat + res_bias, 0)` per Gaussian before alpha
        # compositing, so a naive tex_only render still clips negative residuals.
        # To recover Σ wᵢ·residual_i without the ReLU clipping, render twice with
        # a large positive res_bias=SHIFT:
        #   pass_A = Σ wᵢ · max(residual_i + SHIFT, 0)   (atlas/MLP on)
        #   pass_B = Σ wᵢ · max(0 + SHIFT, 0) = SHIFT · Σ wᵢ   (atlas/MLP off)
        # Pick SHIFT > |min(residual)| so max(·, 0) never triggers → linear.
        # Then tex_unclamped = pass_A - pass_B = Σ wᵢ·residual_i.
        SHIFT = 5.0  # covers MLP range ~[-2.4, 1.6] and atlas range [-0.17, 0.22].
        set_bias(-999.0, SHIFT)

        # Temporarily disable SB for neural tex_only.
        sb_backup = None
        if hasattr(gaussians, "_sb_params") and gaussians._sb_params.numel() > 0:
            sb_backup = gaussians._sb_params.detach().clone()
            with torch.no_grad():
                gaussians._sb_params.zero_()
        try:
            with torch.no_grad():
                # pass_A: residual + SHIFT
                img_n_tex_A = render_neural(cam, gaussians, ingp, bg, pipe, args, cfg,
                                            decompose_mode="tex_only", disable_sb=False)
                img_b_tex_A = render_bake(cam, gaussians, bg, kernel_type, aabb_mode,
                                          atlas_texture=atlas_flat, atlas_rects=rects,
                                          atlas_width=atlas_w,
                                          sb_params=None, sb_number=0)
                # pass_B: no residual, pure SHIFT · Σwᵢ (atlas/MLP disabled).
                # Neural "sh_only" with sh_bias=-999 & res_bias=SHIFT gives 0 + 0 + SHIFT.
                set_bias(-999.0, SHIFT)  # ensure bias still set for sh_only call below
                img_n_shift = render_neural(cam, gaussians, ingp, bg, pipe, args, cfg,
                                            decompose_mode="sh_only", disable_sb=False)
                img_b_shift = render_bake(cam, gaussians, bg, kernel_type, aabb_mode,
                                          atlas_texture=None, atlas_rects=None, atlas_width=0,
                                          sb_params=None, sb_number=0)
        finally:
            if sb_backup is not None:
                with torch.no_grad():
                    gaussians._sb_params.copy_(sb_backup)

        # Recover unclamped Σ wᵢ · residual_i.
        img_n_tex = img_n_tex_A - img_n_shift
        img_b_tex = img_b_tex_A - img_b_shift
        stats_tex = diff_stats(img_n_tex, img_b_tex)
        agg["tex_only"].append(stats_tex)

        # Save raw FP32 tensors for this view.
        view_dir = os.path.join(cmp_args.output_dir, name)
        os.makedirs(view_dir, exist_ok=True)
        np.save(os.path.join(view_dir, "neural_sh_only.npy"),  img_n_sh.cpu().numpy())
        np.save(os.path.join(view_dir, "baked_sh_only.npy"),   img_b_sh.cpu().numpy())
        np.save(os.path.join(view_dir, "neural_tex_only.npy"), img_n_tex.cpu().numpy())
        np.save(os.path.join(view_dir, "baked_tex_only.npy"),  img_b_tex.cpu().numpy())

        # Visualization PNGs. Raw values can exceed 1.0 — normalize each image by its
        # own max for display (preserves relative structure; absolute scale in NPY).
        def norm_png(t):
            arr = t.detach().cpu().float().permute(1, 2, 0).clamp(min=0).numpy()
            m = max(arr.max(), 1e-6)
            return (arr / m).clip(0, 1)
        save_img_u8(norm_png(img_n_sh),  os.path.join(view_dir, "neural_sh.png"))
        save_img_u8(norm_png(img_b_sh),  os.path.join(view_dir, "baked_sh.png"))
        save_img_u8(norm_png(img_n_tex), os.path.join(view_dir, "neural_tex.png"))
        save_img_u8(norm_png(img_b_tex), os.path.join(view_dir, "baked_tex.png"))

        print(f"[{name}]")
        print(f"  SH-only : L1={stats_sh['l1']:.5f}  L2={stats_sh['l2']:.5f}  "
              f"max={stats_sh['max']:.4f}  rel_L1={stats_sh['rel_l1']*100:.2f}%  "
              f"(neural_mean={stats_sh['a_mean']:.4f}, baked_mean={stats_sh['b_mean']:.4f})")
        print(f"  tex-only: L1={stats_tex['l1']:.5f}  L2={stats_tex['l2']:.5f}  "
              f"max={stats_tex['max']:.4f}  rel_L1={stats_tex['rel_l1']*100:.2f}%  "
              f"(neural_mean={stats_tex['a_mean']:.4f}, baked_mean={stats_tex['b_mean']:.4f})")

    # Restore original biases.
    set_bias(sh_bias_train, res_bias_train)

    # Aggregate summary.
    print("\n=== Summary over {} views ===".format(len(test_cams)))
    for mode in ["sh_only", "tex_only"]:
        if not agg[mode]:
            continue
        l1s   = [s["l1"]  for s in agg[mode]]
        l2s   = [s["l2"]  for s in agg[mode]]
        maxs  = [s["max"] for s in agg[mode]]
        rels  = [s["rel_l1"] for s in agg[mode]]
        print(f"{mode:10s}: L1={np.mean(l1s):.5f}±{np.std(l1s):.5f}  "
              f"L2={np.mean(l2s):.5f}  max={np.mean(maxs):.4f}  "
              f"rel_L1={np.mean(rels)*100:.2f}%")

    print(f"\nRaw tensors + PNGs saved to: {cmp_args.output_dir}")


if __name__ == "__main__":
    main()
