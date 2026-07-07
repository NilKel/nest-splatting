#!/usr/bin/env python
"""
Sweep the beta_scaled compact-footprint multiplier (`set_beta_mult`) x the
Gaussian low-pass toggle (`set_drop_lowpass`) on an EXISTING baked atlas, and
report FPS speedup + quality (PSNR/SSIM/LPIPS) for every combo.

This is the bake-side analogue of FastGS's Compact Box `mult`: it scales the
beta footprint cutoff (baseline 4-sigma; beta support ends at 3-sigma, so
mult=0.75 is the lossless point) to shrink the binning box -> fewer
Gaussian-tile pairs -> faster. `--drop_lowpass` removes the Gaussian low-pass
(the alpha max-pool `max(alpha_beta, alpha_lp)` and the `filter_r` screen
extension of the footprint), which is what lets the mult bite fully.

Timing methodology is reused verbatim from benchmark_baked.evaluate_baked
(10 warmup + 100 cuda.Event-timed renders, cycling the test views).

Only valid for `--aabb accutile/snugbox` (mode 5) or `rect` (mode 2) + a
beta kernel -- that's the `else` branch `cutoff = 4*d_beta_mult` was wired into.
"""
import os
import sys
import glob
import json
import pickle
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_baked as bb
from diff_surfel_bake_render import (
    set_activation_bias, set_compact_mult, set_residual_mode, set_untex_kernel,
    set_use_atlas_tex_object, set_atlas_use_uint8, clear_atlas_cache,
    set_atlas_bc7, set_beta_mult, set_drop_lowpass)

from argparse import ArgumentParser as _AP


def load_baked(model_path, iteration=-1):
    """Replicate benchmark_baked.main()'s --skip_bake loader for one baked atlas."""
    with open(os.path.join(model_path, "args.pkl"), "rb") as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True

    cfg_yaml = os.path.join(model_path, "config.yaml")
    cfg = bb.Config(cfg_yaml) if os.path.exists(cfg_yaml) else bb.Config(args.yaml)

    if iteration == -1:
        its = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
               for f in glob.glob(os.path.join(model_path, "ngp_*.pth"))]
        iteration = max(its)

    output_dir = os.path.join(model_path, "baked_atlas")
    with open(os.path.join(output_dir, "bake_meta.json")) as f:
        meta = json.load(f)

    # --- Gaussians + dataset ---
    dataset = bb.ModelParams(_AP(), sentinel=True).extract(args)
    gaussians = bb.GaussianModel(dataset.sh_degree)

    kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(args, "kernel", "gaussian"), 0)
    assert kernel_type in (1, 4), (
        f"beta_mult only affects beta kernels (got kernel={getattr(args,'kernel',None)}).")

    baked_ply = os.path.join(output_dir, "baked.ply")
    gaussians.load_ply(baked_ply)

    # --- Atlas: BC7 fast path (room is BC7) ---
    set_use_atlas_tex_object(True)
    set_atlas_use_uint8(False)
    clear_atlas_cache()
    bc7_file = meta.get("atlas_bc7_file")
    assert bc7_file is not None, "this sweep assumes a BC7 atlas (extend for FP16/uint8 if needed)"
    bc7_bytes = open(os.path.join(output_dir, bc7_file), "rb").read()
    bc7_tensor = torch.frombuffer(bytearray(bc7_bytes), dtype=torch.uint8).cuda()
    set_atlas_bc7(bc7_tensor, int(meta["atlas_bc7_padded_w"]), int(meta["atlas_bc7_padded_h"]),
                  float(meta.get("atlas_offset", 0.0)), float(meta.get("atlas_scale", 1.0)))
    atlas_tex = torch.zeros(1, 1, 3, dtype=torch.float16, device="cuda")  # placeholder
    atlas_width = atlas_tex.shape[1]
    atlas_texture = atlas_tex.reshape(-1).contiguous()
    atlas_rects = torch.load(os.path.join(output_dir, "atlas_rects.pt")).cuda().contiguous()

    # --- device globals from bake_meta ---
    set_activation_bias(float(meta.get("sh_bias", 0.5)), float(meta.get("res_bias", 0.0)))
    set_compact_mult(float(meta.get("compact_mult", 1.0)))   # Gaussian-branch only; inert for beta
    residual_mode = int(meta.get("residual_mode", 0))
    set_residual_mode(residual_mode)
    set_untex_kernel(-1)

    # --- Scene loads cameras AND overwrites the PLY -> reload after (known pitfall) ---
    scene = bb.Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha

    # --- SV color path ---
    gaussians.feature_mode = meta.get("feature_mode", "sh")
    sv_state = bb._make_sv_state(gaussians) if gaussians.feature_mode == "SV" else None

    beta = cfg.surfel.tg_beta
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    return dict(test_cameras=test_cameras, gaussians=gaussians, bg=bg, beta=beta,
                kernel_type=kernel_type, atlas_texture=atlas_texture, atlas_rects=atlas_rects,
                atlas_width=atlas_width, sv_state=sv_state, residual_mode=residual_mode,
                meta=meta, n_gauss=int(gaussians.get_xyz.shape[0]),
                resolution=f"{int(test_cameras[0].image_width)}x{int(test_cameras[0].image_height)}",
                n_test=len(test_cameras))


def run_one(S, beta_mult, drop_lowpass, aabb_mode, sort_mode, nw, nb):
    set_beta_mult(beta_mult)
    set_drop_lowpass(drop_lowpass)
    return bb.evaluate_baked(
        S["test_cameras"], S["gaussians"], S["bg"], S["beta"], S["kernel_type"],
        atlas_texture=S["atlas_texture"], atlas_rects=S["atlas_rects"], atlas_width=S["atlas_width"],
        num_warmup=nw, num_benchmark=nb, save_dir=None,
        aabb_mode=aabb_mode, sort_mode=sort_mode,
        sb_params=None, sb_number=0, sv_state=S["sv_state"],
        final_relu=(S["residual_mode"] == 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--aabb_mode", type=int, default=5,
                    help="5=accutile/snugbox (matches --aabb accutile), 2=rect. Must be the "
                         "else-branch mode so beta_mult is live.")
    ap.add_argument("--sort_mode", type=int, default=0)
    ap.add_argument("--num_warmup", type=int, default=10)
    ap.add_argument("--num_benchmark", type=int, default=100)
    ap.add_argument("--mults", type=float, nargs="+",
                    default=[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25])
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    print(f"[LOAD] {args.model_path}")
    S = load_baked(args.model_path)
    print(f"[LOAD] {S['n_gauss']:,} Gaussians, {S['n_test']} test views, {S['resolution']}, "
          f"kernel_type={S['kernel_type']}, feature_mode={S['meta'].get('feature_mode')}")
    print(f"[LOAD] aabb_mode={args.aabb_mode}, sort_mode={args.sort_mode}  "
          f"(baseline = mult=1.0 / lowpass=ON)\n")

    rows = []  # (lowpass_on, mult, res)
    for lowpass_on in (True, False):
        for m in args.mults:
            res = run_one(S, m, not lowpass_on, args.aabb_mode, args.sort_mode,
                          args.num_warmup, args.num_benchmark)
            rows.append((lowpass_on, m, res))

    # reset globals to neutral so we don't leave the device in a swept state
    set_beta_mult(1.0)
    set_drop_lowpass(False)

    base = next(r for (lp, m, r) in rows if lp and abs(m - 1.0) < 1e-9)
    bfps, bpsnr, bssim, blpips = base["fps"], base["psnr"], base["ssim"], base["lpips"]

    def table(lowpass_on):
        title = "LOWPASS ON  (alpha max-pool + filter_r extension kept)" if lowpass_on \
                else "LOWPASS OFF (pure beta: no max-pool, no filter extension)"
        print("=" * 92)
        print(f"  {title}")
        print("-" * 92)
        print(f"  {'mult':>5} {'cutoff':>7} {'FPS':>8} {'speedup':>8} "
              f"{'ms':>7} {'PSNR':>8} {'dPSNR':>7} {'SSIM':>8} {'LPIPS':>8}")
        for (lp, m, r) in rows:
            if lp != lowpass_on:
                continue
            print(f"  {m:>5.3f} {4.0*m:>6.2f}s {r['fps']:>8.1f} {r['fps']/bfps:>7.2f}x "
                  f"{r['ms_per_frame']:>7.3f} {r['psnr']:>8.3f} {r['psnr']-bpsnr:>+7.3f} "
                  f"{r['ssim']:>8.4f} {r['lpips']:>8.4f}")
        print()

    print("\n" + "#" * 92)
    print(f"  BETA-MULT x LOWPASS SWEEP  (baseline mult=1.0/lowpass=ON: "
          f"{bfps:.1f} FPS, {bpsnr:.3f} dB, SSIM {bssim:.4f}, LPIPS {blpips:.4f})")
    print(f"  beta support ends at 3.00s -> mult=0.75 is the lossless footprint floor")
    print("#" * 92 + "\n")
    table(True)
    table(False)

    out = {
        "model_path": args.model_path, "n_gauss": S["n_gauss"], "resolution": S["resolution"],
        "n_test": S["n_test"], "aabb_mode": args.aabb_mode, "sort_mode": args.sort_mode,
        "baseline": {"fps": bfps, "psnr": bpsnr, "ssim": bssim, "lpips": blpips},
        "sweep": [
            {"lowpass": lp, "mult": m, "cutoff_sigma": 4.0 * m,
             "fps": r["fps"], "ms": r["ms_per_frame"], "speedup": r["fps"] / bfps,
             "psnr": r["psnr"], "ssim": r["ssim"], "lpips": r["lpips"]}
            for (lp, m, r) in rows
        ],
    }
    out_json = args.out_json or os.path.join(
        args.model_path, "baked_atlas", "beta_mult_lowpass_sweep.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] {out_json}")


if __name__ == "__main__":
    main()
