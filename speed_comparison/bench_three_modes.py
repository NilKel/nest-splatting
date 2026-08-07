#!/usr/bin/env python3
"""Benchmark the baked garden checkpoint across three render modes to isolate
the cost of each pipeline stage.

  Mode A — SV + Atlas         : full colour + atlas residual sample     (=production)
  Mode B — SV only            : full colour, no atlas sample            (skip texture fetch)
  Mode C — Passthrough        : NO colour eval, no atlas                (skip SV softmax entirely)

Deltas answer:
  A - B  = per-fragment atlas sample + composition cost
  B - C  = per-Gauss SV softmax cost
  C      = residual pipeline (surfel rasterisation + geometry + sort + blend)

Usage:
    conda run -n nest_splatting python speed_comparison/bench_three_modes.py \\
        --model_path outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10 \\
        --bake_dir  outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10/baked_atlas \\
        --num_warmup 20 --num_benchmark 200 \\
        --out speed_comparison/baked_bench.json
"""
import argparse, glob, json, math, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from argparse import Namespace


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    ajson = model_path / "args.json"
    if ajson.exists():
        return Namespace(**json.load(open(ajson)))
    raise FileNotFoundError(f"no args.pkl or args.json in {model_path}")


def bench_mode(name: str, cameras, rasterize_fn, num_warmup: int, num_bench: int):
    """rasterize_fn(cam) -> tensor image.  Uses CUDA events for wall-clock GPU
    time and returns mean/std ms + FPS."""
    # Warmup
    for _ in range(num_warmup):
        for cam in cameras[: min(5, len(cameras))]:
            _ = rasterize_fn(cam)
    torch.cuda.synchronize()

    # Benchmark: rotate through the test cameras num_bench times each.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    cam = cameras[0]
    for i in range(num_bench):
        starts[i].record()
        _ = rasterize_fn(cam)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(num_bench)])
    mean_ms = float(times_ms.mean())
    std_ms  = float(times_ms.std())
    fps     = 1000.0 / mean_ms if mean_ms > 0 else 0
    print(f"  {name:<20} {mean_ms:.3f} ± {std_ms:.3f} ms   {fps:>8.1f} FPS")
    return {"mode": name, "ms_mean": mean_ms, "ms_std": std_ms, "fps": fps, "num_bench": num_bench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--bake_dir",   type=Path, required=True,
                    help="dir containing baked.ply + atlas_texture.bc7 + bake_meta.json + atlas_rects.pt")
    ap.add_argument("--num_warmup", type=int, default=20)
    ap.add_argument("--num_benchmark", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model_path = args.model_path
    bake_dir   = args.bake_dir
    assert model_path.exists() and bake_dir.exists()

    # Load training config so Scene() can find data.
    train_args = load_train_args(model_path)
    train_args.model_path = str(model_path)
    train_args.eval = True

    # Load YAML config.
    cfg_yaml = model_path / "config.yaml"
    cfg_model = Config(str(cfg_yaml)) if cfg_yaml.exists() else Config(train_args.yaml)

    # Latest iteration (used for scene load).
    ngp_files = glob.glob(str(model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files)

    # Load the BAKED PLY (has SV columns).  Bypass the train.py args resolver.
    gaussians = GaussianModel(train_args.sh_degree)
    baked_ply = bake_dir / "baked.ply"
    gaussians.load_ply(str(baked_ply), args=train_args)
    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    # Freeze SV site mask (inference convention).
    if hasattr(gaussians, "update_sites_mask"):
        try:
            gaussians._sv_training_flag = False
            gaussians.update_sites_mask()
        except Exception:
            pass

    # Load Scene for cameras (uses train_args.source_path from args.pkl).
    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(train_args)
    scene  = Scene(dataset, GaussianModel(train_args.sh_degree),
                   load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    print(f"[bench] {len(test_cameras)} test cameras, {gaussians.get_xyz.shape[0]:,} Gauss")

    # Atlas + rects.
    from diff_surfel_bake_render import prepare_gaussian_inputs
    kernel_map = {"gaussian": 0, "beta": 1, "adaptive_beta": 2, "adaptive_gaussian": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(train_args, "kernel", "gaussian"), 0)
    gaussian_pkg = prepare_gaussian_inputs(gaussians, sh_degree=train_args.sh_degree,
                                           kernel_type=kernel_type)
    n_gauss = gaussian_pkg['means3D'].shape[0]

    # Load bake meta + atlas texture + rects.
    meta = json.load(open(bake_dir / "bake_meta.json"))
    atlas_width = int(meta.get("atlas_width", 0))
    from PIL import Image
    # bc7 atlas — load raw bytes as uint8 tensor? Actually the bake stores it
    # as compressed bytes; the fp16 tensor at atlas_texture.pt is what the
    # baked renderer wants.
    atlas_pt = bake_dir / "atlas_texture.pt"
    atlas_texture = torch.load(str(atlas_pt), map_location="cuda") if atlas_pt.exists() else None
    if atlas_texture is not None and atlas_texture.dtype != torch.float16:
        atlas_texture = atlas_texture.half()
    atlas_rects_gpu = torch.load(str(bake_dir / "atlas_rects.pt"), map_location="cuda")

    # SV state.
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians)

    # Passthrough colour: constant per-Gauss RGB so the rasteriser skips
    # SH/voronoi entirely and just does geometry + sort + blend.
    passthrough_colors = torch.ones((n_gauss, 3), dtype=torch.float32, device="cuda") * 0.5

    from diff_surfel_bake_render import get_rasterizer

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0

    aabb_mode = 3   # matches production bake
    sort_mode = 0

    def _rasterizer_for(cam):
        return get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)

    # Mode A: SV + Atlas
    def render_sv_atlas(cam):
        r = _rasterizer_for(cam)
        v_sites, v_tau, v_colors, v_K = (sv_state['sites'], sv_state['tau'],
                                          sv_state['colors'], sv_state['K'])
        color, _ = r(
            means3D=gaussian_pkg['means3D'],
            opacities=gaussian_pkg['opacities'],
            shs=gaussian_pkg['shs'],
            scales=gaussian_pkg['scales'],
            rotations=gaussian_pkg['rotations'],
            shapes=gaussian_pkg['shapes'],
            kernel_type=gaussian_pkg['kernel_type'],
            atlas_texture=atlas_texture,
            atlas_rects=atlas_rects_gpu,
            atlas_width=atlas_width,
            voronoi_sites=v_sites, voronoi_tau=v_tau,
            voronoi_colors=v_colors, voronoi_K=v_K,
        )
        return color

    # Mode B: SV only, no atlas
    def render_sv_only(cam):
        r = _rasterizer_for(cam)
        v_sites, v_tau, v_colors, v_K = (sv_state['sites'], sv_state['tau'],
                                          sv_state['colors'], sv_state['K'])
        color, _ = r(
            means3D=gaussian_pkg['means3D'],
            opacities=gaussian_pkg['opacities'],
            shs=gaussian_pkg['shs'],
            scales=gaussian_pkg['scales'],
            rotations=gaussian_pkg['rotations'],
            shapes=gaussian_pkg['shapes'],
            kernel_type=gaussian_pkg['kernel_type'],
            voronoi_sites=v_sites, voronoi_tau=v_tau,
            voronoi_colors=v_colors, voronoi_K=v_K,
        )
        return color

    # Mode C: passthrough — colors_precomp, no SV, no atlas.
    def render_passthrough(cam):
        r = _rasterizer_for(cam)
        color, _ = r(
            means3D=gaussian_pkg['means3D'],
            opacities=gaussian_pkg['opacities'],
            colors_precomp=passthrough_colors,
            scales=gaussian_pkg['scales'],
            rotations=gaussian_pkg['rotations'],
            shapes=gaussian_pkg['shapes'],
            kernel_type=gaussian_pkg['kernel_type'],
        )
        return color

    print(f"\n[bench] warmup={args.num_warmup}  bench={args.num_benchmark}")
    print(f"  {'Mode':<20} {'ms/frame':<20}  {'FPS':>8}")
    print(f"  {'-'*20} {'-'*20}  {'-'*8}")
    a = bench_mode("A: SV+atlas",   test_cameras, render_sv_atlas,   args.num_warmup, args.num_benchmark)
    b = bench_mode("B: SV only",    test_cameras, render_sv_only,    args.num_warmup, args.num_benchmark)
    c = bench_mode("C: passthrough", test_cameras, render_passthrough, args.num_warmup, args.num_benchmark)

    # Deltas
    dAB_ms = a["ms_mean"] - b["ms_mean"]   # atlas cost
    dBC_ms = b["ms_mean"] - c["ms_mean"]   # SV cost
    c_ms   = c["ms_mean"]                   # residual pipeline
    print(f"\n[bench] Cost decomposition:")
    print(f"  Atlas sample & composition  (A - B): {dAB_ms:.3f} ms/frame ({dAB_ms/a['ms_mean']*100:.1f}% of A)")
    print(f"  SV softmax colour eval      (B - C): {dBC_ms:.3f} ms/frame ({dBC_ms/a['ms_mean']*100:.1f}% of A)")
    print(f"  Residual pipeline (surfel raster + geometry + sort + blend): {c_ms:.3f} ms/frame ({c_ms/a['ms_mean']*100:.1f}% of A)")

    result = {
        "model_path": str(model_path),
        "bake_dir":   str(bake_dir),
        "num_gaussians": int(n_gauss),
        "resolution":    f"{test_cameras[0].image_width}x{test_cameras[0].image_height}",
        "num_test_cams": len(test_cameras),
        "modes": {"sv_atlas": a, "sv_only": b, "passthrough": c},
        "decomposition_ms": {
            "atlas_sample":      dAB_ms,
            "sv_softmax_eval":   dBC_ms,
            "residual_pipeline": c_ms,
            "total_A":           a["ms_mean"],
        },
        "decomposition_frac_of_A": {
            "atlas_sample":      dAB_ms / a["ms_mean"],
            "sv_softmax_eval":   dBC_ms / a["ms_mean"],
            "residual_pipeline": c_ms   / a["ms_mean"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\n[bench] Saved → {args.out}")


if __name__ == "__main__":
    main()
