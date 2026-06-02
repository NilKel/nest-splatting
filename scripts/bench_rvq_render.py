"""
End-to-end render benchmark with RVQ atlas decode wired into the CUDA
bake-render kernel.

Replays `benchmark_baked.py --skip_bake` for the same scene but installs
the RVQ atlas (codebooks + indices) via `set_atlas_rvq()` instead of the
BC7 / uint8 / FP16 path. Reports PSNR + SSIM + LPIPS + FPS so the row
is directly comparable to the BC7 baseline.

Also supports `--use_bply` to load the 8-bit-quantized .bply PLY for the
combined "RVQply" measurement.

Usage:
  python scripts/bench_rvq_render.py \
      --model_path outputs/.../<config> \
      --bake_dir outputs/.../<config>/baked_atlas \
      [--use_bply baked.bply] [--num_benchmark 100]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
import yaml
from omegaconf import OmegaConf

# RVQ install helpers
sys.path.insert(0, os.path.dirname(__file__))
from bench_rvq_render_lib import install_rvq_atlas


# Reuse benchmark_baked.py's evaluate_baked + prepare_gaussian_inputs.
import benchmark_baked as BB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--use_bply", type=str, default=None,
                   help="Optional path to a .bply (compressed PLY) to use "
                        "instead of baked.ply. Loaded via decompress_baked_bply.py "
                        "into a temporary roundtrip .ply.")
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--num_benchmark", type=int, default=100)
    p.add_argument("--aabb_mode", type=int, default=3)
    p.add_argument("--sort_mode", type=int, default=0)
    p.add_argument("--save_dir", default=None,
                   help="Optional: dir to write per-camera rendered PNGs into.")
    args = p.parse_args()

    # ---- Build dataset/Scene args via ModelParams (mirrors benchmark_baked) ----
    cfg_args_path = os.path.join(args.model_path, "cfg_args")
    saved_args_namespace = None
    if os.path.exists(cfg_args_path):
        with open(cfg_args_path) as f:
            saved = f.read().strip()
        ns = eval(saved, {"Namespace": type("Namespace", (), {"__init__": lambda self, **k: self.__dict__.update(k)})})
        saved_args_namespace = ns

    # Build a minimal arg namespace ModelParams expects.
    class _A: pass
    args_ns = _A()
    for k, v in vars(saved_args_namespace).items():
        setattr(args_ns, k, v)
    args_ns.model_path = args.model_path
    args_ns.eval = True

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args_ns)

    # cfg for tg_base_alpha (per benchmark_baked.py)
    cfg = OmegaConf.load("configs/mip_360.yaml") if os.path.exists("configs/mip_360.yaml") else \
          OmegaConf.create({"surfel": {"tg_base_alpha": 0.5}})

    gaussians = GaussianModel(dataset.sh_degree)
    if args.use_bply is not None:
        rt_ply = os.path.join(args.bake_dir, "baked_roundtrip_for_bench.ply")
        print(f"[BPLY] decompressing {args.use_bply} → {rt_ply}")
        import subprocess
        subprocess.run([sys.executable, "scripts/decompress_baked_bply.py",
                        "--input", args.use_bply, "--output", rt_ply],
                       check=True)
        ply_to_load = rt_ply
    else:
        ply_to_load = os.path.join(args.bake_dir, "baked.ply")

    gaussians.load_ply(ply_to_load)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    if hasattr(saved_args_namespace, 'kernel'):
        gaussians.kernel_type = saved_args_namespace.kernel
    gaussians.kernel_type2 = getattr(saved_args_namespace, 'kernel2', None)
    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(getattr(saved_args_namespace, 'kernel', 'gaussian'), 0)

    # ---- Atlas: install RVQ via the new CUDA entry ---------------------------
    from diff_surfel_bake_render import (set_atlas_use_uint8,
                                          set_use_atlas_tex_object,
                                          clear_atlas_cache, clear_atlas_bc7)
    set_use_atlas_tex_object(True)
    set_atlas_use_uint8(True)
    clear_atlas_cache()
    clear_atlas_bc7()

    bake_meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    from diff_surfel_bake_render import (set_activation_bias, set_compact_mult,
                                          set_residual_mode, set_untex_kernel)
    set_activation_bias(float(bake_meta.get("sh_bias", 0.5)),
                        float(bake_meta.get("res_bias", 0.0)))
    set_compact_mult(float(bake_meta.get("compact_mult", 1.0)))
    set_residual_mode(int(bake_meta.get("residual_mode", 0)))
    _kmap2 = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    _k2 = bake_meta.get("kernel2", None)
    set_untex_kernel(_kmap2.get(_k2, -1) if _k2 else -1)

    install_rvq_atlas(args.bake_dir, device='cuda')

    # ---- Load atlas_rects (the renderer still needs this for surfel coords) --
    atlas_rects_gpu = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                                  map_location='cuda', weights_only=False).contiguous()
    atlas_width = 0   # RVQ path doesn't use this
    # We need to pass *some* atlas_texture tensor through evaluate_baked to keep
    # PyTorch's dtype checks happy; the kernel ignores it when d_rvq_codebooks
    # is set. Use a 1×1×3 FP16 placeholder.
    placeholder_atlas = torch.zeros(1, 1, 3, dtype=torch.float16, device='cuda')

    # ---- Scene + test cameras (Scene constructor reloads PLY → reload baked) ---
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    gaussians.load_ply(ply_to_load)
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    feat_mode = bake_meta.get("feature_mode", "sh")
    gaussians.feature_mode = feat_mode
    sv_state = None
    if feat_mode == "SV":
        from gaussian_renderer import _make_sv_state
        sv_state = _make_sv_state(gaussians)
    print(f"[BENCH-RVQ] {len(test_cameras)} test cameras, "
          f"PLY: {ply_to_load}, atlas: RVQ ({'BPLY' if args.use_bply else 'orig PLY'})")

    # ---- Run the benchmark ----------------------------------------------------
    bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32, device='cuda')
    final_relu = (int(bake_meta.get("residual_mode", 0)) == 2)
    beta = 2.0  # standard; unused unless --feature beta

    print(f"[BENCH-RVQ] running evaluate_baked() — {args.num_warmup} warmup, "
          f"{args.num_benchmark} timed frames")
    result = BB.evaluate_baked(
        test_cameras, gaussians, bg_color, beta, kernel_type,
        atlas_texture=placeholder_atlas, atlas_rects=atlas_rects_gpu,
        atlas_width=atlas_width,
        num_warmup=args.num_warmup, num_benchmark=args.num_benchmark,
        save_dir=args.save_dir,
        aabb_mode=args.aabb_mode, sort_mode=args.sort_mode,
        sb_params=None, sb_number=0,
        sv_state=sv_state, final_relu=final_relu,
    )
    print("\n======================================================================")
    print("  RVQ-atlas RESULTS")
    print("======================================================================")
    print(f"  Mode: {'8-bit PLY + RVQ atlas' if args.use_bply else 'FP32 PLY + RVQ atlas'}")
    print(f"  PSNR  : {result['psnr']:.2f} dB")
    print(f"  SSIM  : {result['ssim']:.4f}")
    print(f"  LPIPS : {result['lpips']:.4f}")
    print(f"  FPS   : {result['fps']:.2f}  ({result['ms_per_frame']:.3f} ms/frame)")
    print("======================================================================")


if __name__ == "__main__":
    main()
