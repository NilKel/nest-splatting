"""Single-view FP32 vs FP16 quantization test for SV / SB per-Gaussian params.

Renders one test view from a baked model with the CUDA bake renderer twice:
  (a) FP32 reference — params as-trained.
  (b) FP16 quantized — `(_sv_sites, _sv_tau, _sv_colors)` for SV models or
      `_sb_params` for beta models cast to FP16 then back to FP32 (this
      simulates storing those tensors at half precision; the rasterizer
      itself still expects FP32 pointers).

Saves both PNGs side-by-side + a diff heatmap and prints PSNR/SSIM/LPIPS
between the two.

Usage:
    python scripts/test_fp16_quant.py \\
        --model_path outputs/mip_360/bonsai/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac \\
        --out_dir /tmp/fp16_test/bonsai_sv \\
        --view 0
"""
import argparse, json, math, os, pickle, sys
from argparse import Namespace
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from hash_encoder.config import Config


def _load_args(model_path):
    p = os.path.join(model_path, "args.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    p = os.path.join(model_path, "args.json")
    if os.path.exists(p):
        with open(p) as f:
            return Namespace(**json.load(f))
    raise FileNotFoundError(f"No args.pkl/args.json in {model_path}")


def _render(gaussians, cam, bake_meta, bc7_bytes, atlas_rects, sb_params, sb_number,
            sv_state, kernel_type, beta=0.0, aabb_mode=5, sort_mode=0):
    from diff_surfel_bake_render import (
        get_rasterizer, set_atlas_bc7, clear_atlas_bc7,
        set_atlas_use_uint8, set_use_atlas_tex_object, clear_atlas_cache,
        set_activation_bias, set_compact_mult, set_residual_mode,
    )
    set_use_atlas_tex_object(True)
    set_atlas_use_uint8(True)
    clear_atlas_cache()
    clear_atlas_bc7()
    bc7_t = torch.frombuffer(bytearray(bc7_bytes), dtype=torch.uint8).cuda()
    set_atlas_bc7(bc7_t,
                  int(bake_meta["atlas_bc7_padded_w"]),
                  int(bake_meta["atlas_bc7_padded_h"]),
                  float(bake_meta.get("atlas_offset", 0.0)),
                  float(bake_meta.get("atlas_scale",  1.0)))
    set_activation_bias(float(bake_meta.get("sh_bias", 0.5)),
                        float(bake_meta.get("res_bias", 0.0)))
    set_compact_mult(float(bake_meta.get("compact_mult", 1.0)))
    set_residual_mode(int(bake_meta.get("residual_mode", 0)))

    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    rasterizer = get_rasterizer(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.zeros(3, device="cuda"),
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        campos=cam.camera_center,
        sh_degree=gaussians.active_sh_degree, beta=beta,
        aabb_mode=aabb_mode, sort_mode=sort_mode,
    )

    if sv_state is not None:
        v_sites, v_tau, v_colors, v_K = (sv_state["sites"], sv_state["tau"],
                                          sv_state["colors"], sv_state["K"])
    else:
        v_sites = v_tau = v_colors = None; v_K = 0

    atlas_tex_dummy = torch.zeros(1, 1, 3, dtype=torch.float16, device="cuda")
    color, _ = rasterizer(
        means3D=gaussians.get_xyz.contiguous(),
        opacities=gaussians.get_opacity.contiguous(),
        shs=gaussians.get_features.contiguous(),
        scales=gaussians.get_scaling.contiguous(),
        rotations=gaussians.get_rotation.contiguous(),
        shapes=(gaussians.get_shape.contiguous() if (kernel_type > 0
                  and gaussians._shape.numel() > 0) else None),
        kernel_type=kernel_type,
        atlas_texture=atlas_tex_dummy,
        atlas_rects=atlas_rects,
        atlas_width=int(bake_meta["atlas_bc7_padded_w"]),
        sb_params=sb_params, sb_number=sb_number,
        voronoi_sites=v_sites, voronoi_tau=v_tau,
        voronoi_colors=v_colors, voronoi_K=v_K,
    )
    return color.clamp(0, 1).contiguous()


def _make_sv_state(g, fp16=False):
    if g._sv_sites.numel() == 0:
        return None
    K = g._sv_sites.shape[1]
    sites = torch.nn.functional.normalize(g._sv_sites, dim=-1)
    tau   = torch.exp(g._sv_tau) if g._sv_tau.numel() > 0 else torch.norm(g._sv_sites, dim=-1)
    sv_dc = getattr(g, "_sv_dc", None)
    colors = ((g._sv_colors + sv_dc.unsqueeze(1))
              if sv_dc is not None and sv_dc.numel() > 0
              else g._sv_colors)
    if fp16:
        # Round-trip through FP16 to simulate half-precision storage.
        sites  = sites.half().float()
        tau    = tau.half().float()
        colors = colors.half().float()
    return {"sites": sites.contiguous(),
            "tau":   tau.contiguous(),
            "colors": colors.contiguous(),
            "K": int(K)}


def _make_sb_params(g, fp16=False):
    if not hasattr(g, "_sb_params") or g._sb_params.numel() == 0:
        return None, 0
    sb = g._sb_params
    if fp16:
        sb = sb.half().float()
    return sb.reshape(-1).contiguous(), int(g._sb_params.shape[1])


def _psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--iteration", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_args = _load_args(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True
    cfg_yaml = os.path.join(args.model_path, "config.yaml")
    if not os.path.exists(cfg_yaml):
        cfg_yaml = train_args.yaml
    cfg = Config(cfg_yaml)

    bake_dir = os.path.join(args.model_path, "baked_atlas")
    bc7 = os.path.join(bake_dir, "atlas_texture.bc7")
    if not os.path.exists(bc7):
        raise FileNotFoundError(f"No baked atlas at {bc7} — bake first.")

    iteration = args.iteration
    if iteration == -1:
        import glob
        plys = glob.glob(os.path.join(args.model_path, "point_cloud", "iteration_*"))
        iteration = max(int(os.path.basename(p).split("_")[1]) for p in plys)

    gaussians = GaussianModel(train_args.sh_degree)
    gaussians.kernel_type = getattr(train_args, "kernel", "gaussian")
    temp = argparse.ArgumentParser()
    mp = ModelParams(temp, sentinel=True)
    dataset = mp.extract(train_args)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cams = scene.getTestCameras()
    cam = test_cams[args.view]
    print(f"[TEST] using test view {args.view}/{len(test_cams)}: {cam.image_name}  "
          f"{cam.image_width}x{cam.image_height}")

    # Reload baked PLY (Scene() ctor loads training PLY).
    gaussians.load_ply(os.path.join(bake_dir, "baked.ply"))
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha

    with open(os.path.join(bake_dir, "bake_meta.json")) as f:
        bake_meta = json.load(f)
    with open(bc7, "rb") as f:
        bc7_bytes = f.read()
    atlas_rects = torch.load(os.path.join(bake_dir, "atlas_rects.pt")).cuda().contiguous()

    kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(train_args, "kernel", "gaussian"), 0)

    feat = bake_meta.get("feature_mode", "sh")
    print(f"[TEST] feature_mode={feat}  kernel_type={kernel_type}")

    # --- FP32 reference ---
    sv32 = _make_sv_state(gaussians, fp16=False) if feat == "SV" else None
    sb32, sb_n = _make_sb_params(gaussians, fp16=False)
    print("[TEST] rendering FP32 reference...")
    img32 = _render(gaussians, cam, bake_meta, bc7_bytes, atlas_rects,
                    sb32, sb_n, sv32, kernel_type)

    # --- FP16 quantized ---
    sv16 = _make_sv_state(gaussians, fp16=True) if feat == "SV" else None
    sb16 = sb32; sb_n16 = sb_n
    if feat == "beta":
        sb16, sb_n16 = _make_sb_params(gaussians, fp16=True)
    print("[TEST] rendering FP16-quantized...")
    img16 = _render(gaussians, cam, bake_meta, bc7_bytes, atlas_rects,
                    sb16, sb_n16, sv16, kernel_type)

    # --- Diff metrics ---
    psnr = _psnr(img32, img16).item()
    diff = (img32 - img16).abs()
    print(f"\n=== Result ({feat}) ===")
    print(f"  PSNR(fp16 vs fp32):  {psnr:.2f} dB")
    print(f"  max abs diff:         {diff.max().item():.6f}")
    print(f"  mean abs diff:        {diff.mean().item():.6f}")

    # SSIM
    try:
        from pytorch_msssim import ssim as _ssim
        s = _ssim(img32.unsqueeze(0), img16.unsqueeze(0),
                  data_range=1.0, size_average=True).item()
        print(f"  SSIM(fp16 vs fp32):  {s:.6f}")
    except ImportError:
        print("  SSIM: skipped (pytorch_msssim not installed)")

    # LPIPS via lpipsPyTorch (kept consistent with the bench)
    try:
        from lpipsPyTorch import lpips as _lpips
        lp = _lpips(img32.unsqueeze(0), img16.unsqueeze(0), net_type="vgg").item()
        print(f"  LPIPS(fp16 vs fp32): {lp:.6f}")
    except ImportError:
        print("  LPIPS: skipped (lpipsPyTorch not on path)")

    # Save outputs
    def _save(t, name):
        Image.fromarray(
            (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(args.out_dir, name))
    _save(img32, "fp32.png")
    _save(img16, "fp16.png")
    # Amplified diff (8x) for visual inspection
    _save((diff * 8.0).clamp(0, 1), "diff_x8.png")
    print(f"\n  saved fp32.png, fp16.png, diff_x8.png → {args.out_dir}")


if __name__ == "__main__":
    main()
