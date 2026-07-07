#!/usr/bin/env python
"""
A/B bench: mode-5 beta footprint = fixed 4σ (baseline) vs OPACITY-AWARE max(r_beta, r_lp)
(the 1/255 iso, fed to AccuTile) on an existing baked atlas. Confirms the tighter ellipse
drops FPS-cost at no PSNR cost. Reuses sweep_beta_mult_lowpass.load_baked + the
benchmark_baked.evaluate_baked timing (10 warmup + 100 cuda.Event renders).
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_baked as bb
from sweep_beta_mult_lowpass import load_baked
from diff_surfel_bake_render import set_beta_mult, set_drop_lowpass, set_opacity_aware_beta


def run(S, opacity_aware, aabb_mode, sort_mode, nw, nb):
    set_beta_mult(1.0)
    set_drop_lowpass(False)
    set_opacity_aware_beta(opacity_aware)
    res = bb.evaluate_baked(
        S["test_cameras"], S["gaussians"], S["bg"], S["beta"], S["kernel_type"],
        atlas_texture=S["atlas_texture"], atlas_rects=S["atlas_rects"], atlas_width=S["atlas_width"],
        num_warmup=nw, num_benchmark=nb, save_dir=None,
        aabb_mode=aabb_mode, sort_mode=sort_mode,
        sb_params=None, sb_number=0, sv_state=S["sv_state"],
        final_relu=(S["residual_mode"] == 2))
    set_opacity_aware_beta(False)  # reset
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--aabb_mode", type=int, default=5)
    ap.add_argument("--sort_mode", type=int, default=0)
    ap.add_argument("--num_warmup", type=int, default=10)
    ap.add_argument("--num_benchmark", type=int, default=100)
    args = ap.parse_args()

    print(f"[LOAD] {args.model_path}")
    S = load_baked(args.model_path)
    print(f"[LOAD] {S['n_gauss']:,} Gaussians, {S['n_test']} test views, {S['resolution']}, "
          f"kernel_type={S['kernel_type']}, aabb_mode={args.aabb_mode}\n")

    base = run(S, False, args.aabb_mode, args.sort_mode, args.num_warmup, args.num_benchmark)
    oab  = run(S, True,  args.aabb_mode, args.sort_mode, args.num_warmup, args.num_benchmark)

    print("\n" + "=" * 86)
    print("  AccuTile beta cutoff: fixed 4σ  vs  opacity-aware max(r_beta, r_lp)")
    print("-" * 86)
    print(f"  {'cutoff':<22} {'FPS':>9} {'ms':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"  {'fixed 4σ (baseline)':<22} {base['fps']:>9.1f} {base['ms_per_frame']:>8.3f} "
          f"{base['psnr']:>8.3f} {base['ssim']:>8.4f} {base['lpips']:>8.4f}")
    print(f"  {'opacity-aware':<22} {oab['fps']:>9.1f} {oab['ms_per_frame']:>8.3f} "
          f"{oab['psnr']:>8.3f} {oab['ssim']:>8.4f} {oab['lpips']:>8.4f}")
    print("-" * 86)
    print(f"  Δ : FPS {oab['fps']/base['fps']:.3f}×  ({oab['fps']-base['fps']:+.0f}),  "
          f"PSNR {oab['psnr']-base['psnr']:+.4f},  SSIM {oab['ssim']-base['ssim']:+.5f},  "
          f"LPIPS {oab['lpips']-base['lpips']:+.5f}")
    print("=" * 86 + "\n")

    out = {"model_path": args.model_path, "aabb_mode": args.aabb_mode,
           "baseline_fixed4sigma": base, "opacity_aware": oab}
    p = os.path.join(args.model_path, "baked_atlas", "opacity_aware_beta_bench.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] {p}")


if __name__ == "__main__":
    main()
