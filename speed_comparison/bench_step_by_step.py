#!/usr/bin/env python3
"""Step-by-step investigation of what's expensive in the baked renderer.

All runs are UNTEXTURED (SV only, no atlas) so we're measuring surfel raster
+ geometry + sort + blend + per-fragment kernel eval.

  Baseline   — production settings (kernel=beta_scaled, opacity-aware AdR
              cutoff, aabb_mode=3, sort_mode=0, beta_mult=1.0)
  Flat       — override per-Gauss shape=0 → base^0=1 uniformly (top-hat
              inside the disc).  Isolates the powf() cost in the beta kernel
              path.  All the geometry + sort + blend + tile setup is identical.
  Sort=1     — same as Baseline but sort_mode=1 (our two-stage sort).  Note:
              FastGS's own sort is a SINGLE 64-bit sort (same as our sort_mode=0);
              their PLY→viewer path uses ONE cub::DeviceRadixSort::SortPairs call
              on a (tile_id<<32 | depth) key.  Our sort_mode=1 is a two-stage
              radix (visible→depth, then instances→tile with stable order),
              NOT what FastGS does.
  BetaMult   — beta_mult=0.5 → cutoff halved (approx 2σ instead of 4σ).  This
              is the closest analog to FastGS's compact_mult=0.5 that works
              in our beta_scaled kernel path (compact_mult itself only applies
              to the Gaussian kernel + aabb_mode 1/3 branch in our CUDA).
"""
import argparse, glob, json, math, os, pickle, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import Namespace

from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    ajson = model_path / "args.json"
    if ajson.exists():
        return Namespace(**json.load(open(ajson)))
    raise FileNotFoundError(model_path)


def bench_mode(name: str, cameras, rasterize_fn, num_warmup: int, num_bench: int):
    for _ in range(num_warmup):
        for cam in cameras[: min(5, len(cameras))]:
            _ = rasterize_fn(cam)
    torch.cuda.synchronize()

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
    print(f"  {name:<30} {mean_ms:.3f} ± {std_ms:.3f} ms   {fps:>8.1f} FPS")
    return {"mode": name, "ms_mean": mean_ms, "ms_std": std_ms, "fps": fps, "num_bench": num_bench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--bake_dir",   type=Path, required=True)
    ap.add_argument("--num_warmup", type=int, default=20)
    ap.add_argument("--num_benchmark", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train_args = load_train_args(args.model_path)
    train_args.model_path = str(args.model_path)
    train_args.eval = True
    cfg_yaml = args.model_path / "config.yaml"
    cfg_model = Config(str(cfg_yaml)) if cfg_yaml.exists() else Config(train_args.yaml)
    ngp_files = glob.glob(str(args.model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files)

    # Load model
    gaussians = GaussianModel(train_args.sh_degree)
    gaussians.load_ply(str(args.bake_dir / "baked.ply"), args=train_args)
    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(gaussians, "update_sites_mask"):
        try:
            gaussians._sv_training_flag = False
            gaussians.update_sites_mask()
        except Exception:
            pass

    # Load cameras
    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    _ = PipelineParams(temp_parser)
    dataset = model_params.extract(train_args)
    scene  = Scene(dataset, GaussianModel(train_args.sh_degree),
                   load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    print(f"[bench] {len(test_cameras)} test cameras, {gaussians.get_xyz.shape[0]:,} Gauss")

    from diff_surfel_bake_render import (prepare_gaussian_inputs, get_rasterizer,
                                          set_beta_mult, set_drop_lowpass,
                                          set_opacity_aware_beta, set_compact_mult)
    kernel_map = {"gaussian": 0, "beta": 1, "adaptive_beta": 2,
                  "adaptive_gaussian": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(train_args, "kernel", "gaussian"), 0)
    gaussian_pkg = prepare_gaussian_inputs(gaussians, sh_degree=train_args.sh_degree,
                                           kernel_type=kernel_type)

    # SV state (mirrors production path).
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians)

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0
    aabb_mode = 3
    sort_mode = 0

    def _make_r(cam, aabb, sort_m):
        return get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb, sort_mode=sort_m)

    # Snapshot original shapes so we can restore + swap in flat variant.
    orig_shapes = gaussian_pkg['shapes']
    zero_shapes = torch.zeros_like(orig_shapes) if orig_shapes is not None else None

    def _rasterize(cam, shapes_override, sort_m, aabb_m):
        r = _make_r(cam, aabb_m, sort_m)
        v_sites, v_tau, v_colors, v_K = (sv_state['sites'], sv_state['tau'],
                                          sv_state['colors'], sv_state['K'])
        color, _ = r(
            means3D=gaussian_pkg['means3D'],
            opacities=gaussian_pkg['opacities'],
            shs=gaussian_pkg['shs'],
            scales=gaussian_pkg['scales'],
            rotations=gaussian_pkg['rotations'],
            shapes=shapes_override,
            kernel_type=gaussian_pkg['kernel_type'],
            voronoi_sites=v_sites, voronoi_tau=v_tau,
            voronoi_colors=v_colors, voronoi_K=v_K,
        )
        return color

    # -- variant knob-flippers (state lives in CUDA device globals) --
    def _reset_knobs():
        set_beta_mult(1.0)
        set_drop_lowpass(False)
        set_opacity_aware_beta(True)
        set_compact_mult(1.0)

    print(f"\n[bench] warmup={args.num_warmup}  bench={args.num_benchmark}\n")
    print(f"  {'Mode':<30} {'ms/frame':<20}  {'FPS':>8}")
    print(f"  {'-'*30} {'-'*20}  {'-'*8}")

    _reset_knobs()
    base = bench_mode("baseline (SV only)",
                       test_cameras,
                       lambda c: _rasterize(c, orig_shapes, 0, aabb_mode),
                       args.num_warmup, args.num_benchmark)

    _reset_knobs()
    flat = bench_mode("flat kernel (shape=0)",
                       test_cameras,
                       lambda c: _rasterize(c, zero_shapes, 0, aabb_mode),
                       args.num_warmup, args.num_benchmark)

    _reset_knobs()
    sort1 = bench_mode("sort_mode=1 (two-stage)",
                       test_cameras,
                       lambda c: _rasterize(c, orig_shapes, 1, aabb_mode),
                       args.num_warmup, args.num_benchmark)

    _reset_knobs()
    set_beta_mult(0.5)          # halves the beta cutoff → smaller AABB / tile count
    beta_half = bench_mode("beta_mult=0.5 (half cutoff)",
                            test_cameras,
                            lambda c: _rasterize(c, orig_shapes, 0, aabb_mode),
                            args.num_warmup, args.num_benchmark)
    _reset_knobs()

    _reset_knobs()
    set_beta_mult(0.5); set_drop_lowpass(True); set_opacity_aware_beta(False)
    combo = bench_mode("beta_mult=0.5 + drop_lp + fixed4",
                       test_cameras,
                       lambda c: _rasterize(c, orig_shapes, 0, aabb_mode),
                       args.num_warmup, args.num_benchmark)
    _reset_knobs()

    # aabb_mode=3 + beta + shapes uses a dedicated first branch that
    # ignores d_beta_mult, so beta_mult=0.5 above was a no-op for
    # production settings.  Route around it: aabb_mode=2 (rect, NO AdR)
    # hits the d_opacity_aware_beta branch, which DOES multiply the
    # cutoff by d_beta_mult.  These two variants let us see the tighter-
    # AABB effect properly:
    _reset_knobs()
    aabb2_ref = bench_mode("aabb=2 baseline (opacity-aware, mult=1.0)",
                            test_cameras,
                            lambda c: _rasterize(c, orig_shapes, 0, 2),
                            args.num_warmup, args.num_benchmark)

    _reset_knobs(); set_beta_mult(0.5)
    aabb2_half = bench_mode("aabb=2 + beta_mult=0.5 (tight)",
                             test_cameras,
                             lambda c: _rasterize(c, orig_shapes, 0, 2),
                             args.num_warmup, args.num_benchmark)
    _reset_knobs()

    # --- FastGS-style: swap kernel to plain unbounded Gaussian and apply
    # compact_mult on the tail cutoff.  Route through aabb=3 with a
    # kernel_type override to 0 (Gaussian) and shapes=None so the CUDA
    # hits the `use_adr && !is_beta_kernel` branch that consults
    # d_compact_mult.
    def _rasterize_gauss(cam, sort_m, aabb_m):
        r = _make_r(cam, aabb_m, sort_m)
        v_sites, v_tau, v_colors, v_K = (sv_state['sites'], sv_state['tau'],
                                          sv_state['colors'], sv_state['K'])
        color, _ = r(
            means3D=gaussian_pkg['means3D'],
            opacities=gaussian_pkg['opacities'],
            shs=gaussian_pkg['shs'],
            scales=gaussian_pkg['scales'],
            rotations=gaussian_pkg['rotations'],
            shapes=None,               # skip beta path
            kernel_type=0,             # plain Gaussian
            voronoi_sites=v_sites, voronoi_tau=v_tau,
            voronoi_colors=v_colors, voronoi_K=v_K,
        )
        return color

    _reset_knobs()
    gauss_ref = bench_mode("kernel=Gaussian, compact_mult=1.0",
                            test_cameras,
                            lambda c: _rasterize_gauss(c, 0, 3),
                            args.num_warmup, args.num_benchmark)

    _reset_knobs(); set_compact_mult(0.5)
    gauss_half = bench_mode("kernel=Gaussian, compact_mult=0.5 (FastGS)",
                             test_cameras,
                             lambda c: _rasterize_gauss(c, 0, 3),
                             args.num_warmup, args.num_benchmark)
    _reset_knobs()

    _reset_knobs(); set_compact_mult(0.25)
    gauss_quarter = bench_mode("kernel=Gaussian, compact_mult=0.25",
                                test_cameras,
                                lambda c: _rasterize_gauss(c, 0, 3),
                                args.num_warmup, args.num_benchmark)
    _reset_knobs()

    # -----------------------------------------------------------------
    # LEAN rasterizer variants: same math, but the diff_surfel_bake_render_lean
    # fork STRIPS collected_shapes / collected_is_textured / collected_ewa_conic
    # + the atlas/SB/mixed_3d/beta code paths inside the render kernel.
    # Shared-memory footprint: 22 KB/block -> ~16 KB/block.  If SM occupancy
    # is the bottleneck, this should be measurably faster than the equivalent
    # production kernel with the same inputs.
    # -----------------------------------------------------------------
    from diff_surfel_bake_render_lean import (
        get_rasterizer as get_rasterizer_lean,
        set_compact_mult as set_compact_mult_lean,
        prepare_gaussian_inputs as prepare_gaussian_inputs_lean,
    )
    # Lean fork insists on its own prepared inputs (independent scratch caches
    # on its own device globals).  Use the same underlying tensors.
    gaussian_pkg_lean = prepare_gaussian_inputs_lean(
        gaussians, sh_degree=train_args.sh_degree, kernel_type=0)  # force Gaussian

    def _make_r_lean(cam, aabb, sort_m):
        return get_rasterizer_lean(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb, sort_mode=sort_m)

    def _rasterize_lean(cam, sort_m, aabb_m):
        r = _make_r_lean(cam, aabb_m, sort_m)
        v_sites, v_tau, v_colors, v_K = (sv_state['sites'], sv_state['tau'],
                                          sv_state['colors'], sv_state['K'])
        color, _ = r(
            means3D=gaussian_pkg_lean['means3D'],
            opacities=gaussian_pkg_lean['opacities'],
            shs=gaussian_pkg_lean['shs'],
            scales=gaussian_pkg_lean['scales'],
            rotations=gaussian_pkg_lean['rotations'],
            shapes=None,               # force Gaussian
            kernel_type=0,
            voronoi_sites=v_sites, voronoi_tau=v_tau,
            voronoi_colors=v_colors, voronoi_K=v_K,
        )
        return color

    # Compare vs the same-args production Gaussian ref (kernel=Gauss,
    # compact_mult=1.0, aabb=3) -> gauss_ref above.
    set_compact_mult_lean(1.0)
    lean_ref = bench_mode("LEAN kernel=Gaussian, compact_mult=1.0",
                          test_cameras,
                          lambda c: _rasterize_lean(c, 0, 3),
                          args.num_warmup, args.num_benchmark)

    set_compact_mult_lean(0.5)
    lean_half = bench_mode("LEAN kernel=Gaussian, compact_mult=0.5",
                           test_cameras,
                           lambda c: _rasterize_lean(c, 0, 3),
                           args.num_warmup, args.num_benchmark)
    set_compact_mult_lean(1.0)

    # Deltas vs baseline.
    def _delta(x, ref):
        return x["ms_mean"] - ref["ms_mean"], (ref["ms_mean"] - x["ms_mean"]) / ref["ms_mean"] * 100

    print(f"\n[bench] Deltas vs baseline (aabb=3, production):")
    for m in [flat, sort1, beta_half, combo]:
        d_ms, pct_saved = _delta(m, base)
        print(f"  {m['mode']:<38} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% frame time)")

    print(f"\n[bench] Deltas vs aabb=2 baseline (opacity-aware, mult=1.0):")
    d_ms, pct_saved = _delta(aabb2_half, aabb2_ref)
    print(f"  {aabb2_half['mode']:<38} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% frame time)")

    print(f"\n[bench] FastGS-style path (unbounded Gaussian + compact_mult):")
    d_ms, pct_saved = _delta(gauss_half, gauss_ref)
    print(f"  {gauss_half['mode']:<45} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% vs gauss ref)")
    d_ms, pct_saved = _delta(gauss_quarter, gauss_ref)
    print(f"  {gauss_quarter['mode']:<45} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% vs gauss ref)")
    d_ms_v_base, pct_v_base = _delta(gauss_half, base)
    print(f"  {gauss_half['mode']:<45} Δ = {d_ms_v_base:+.3f} ms   ({-pct_v_base:+.1f}% vs production baseline)")

    print(f"\n[bench] LEAN vs PROD (same Gaussian kernel + same compact_mult):")
    d_ms, pct_saved = _delta(lean_ref, gauss_ref)
    print(f"  {lean_ref['mode']:<45} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% vs gauss ref)")
    d_ms, pct_saved = _delta(lean_half, {'ms_mean': gauss_half['ms_mean']})
    print(f"  {lean_half['mode']:<45} Δ = {d_ms:+.3f} ms   ({-pct_saved:+.1f}% vs gauss_half)")

    result = {
        "model_path":    str(args.model_path),
        "bake_dir":      str(args.bake_dir),
        "num_gaussians": int(gaussian_pkg['means3D'].shape[0]),
        "resolution":    f"{test_cameras[0].image_width}x{test_cameras[0].image_height}",
        "num_test_cams": len(test_cameras),
        "modes": {
            "baseline":       base,
            "flat":           flat,
            "sort1":          sort1,
            "beta_half":      beta_half,
            "combo":          combo,
            "aabb2_ref":      aabb2_ref,
            "aabb2_half":     aabb2_half,
            "gauss_ref":      gauss_ref,
            "gauss_half":     gauss_half,
            "gauss_quarter":  gauss_quarter,
            "lean_ref":       lean_ref,
            "lean_half":      lean_half,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\n[bench] Saved → {args.out}")


if __name__ == "__main__":
    main()
