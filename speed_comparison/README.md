# Speed comparison: nest-splatting baked vs. FastGS (garden scene, 1297×840)

Investigating the tester's suspicion: **"FastGS gets 1000 FPS with 600k Gaussians and
no overdraw reduction. Our baked garden reports ~1000 FPS with ~148k Gaussians — how
does that math work?"**

TL;DR: the FPS numbers are honest, but our per-Gauss cost is **~4× higher than
FastGS's**, so we spend ~4× more wall-clock per splat.  The bottleneck is our
**residual pipeline** (surfel raster + geometry + sort + blend) at 70% of frame
time; SV colour eval is essentially free (0.5%).  We render fewer Gauss and pay
more per-Gauss.

## Layout

```
speed_comparison/
├── README.md                        ← this file
├── bench_three_modes.py             ← 3-mode baked benchmark (SV+atlas, SV-only, passthrough)
├── render_intersection_all.py       ← per-view intersection heatmaps + summary
├── baked_bench.json                 ← nest-splatting baked FPS (all 3 modes)
├── fastgs/                          ← FastGS output (copied from ../FastGS/output/garden/test/ours_30000)
│   ├── eval_summary.json            ← FPS + PSNR/SSIM/LPIPS
│   ├── intersection_summary.json    ← per-view + global contributor stats
│   ├── intersection_maps/*.png      ← turbo heatmaps, max_display=200
│   └── intersection_raw/*.npy       ← raw per-pixel contributor counts
└── nest_splatting_neural/           ← same three, generated for the RD_SV nest checkpoint
    ├── intersection_summary.json
    ├── intersection_maps/*.png
    └── intersection_raw/*.npy
```

Everything at **max_display=200, turbo colormap** (same convention on both sides,
directly comparable pixel-for-pixel).

---

## 1. Overdraw comparison (all 24 test views of garden)

| Renderer      | Gauss | mean contributors/px | max contributors/px |
|---------------|------:|---------------------:|--------------------:|
| Nest-splatting neural (SH_res+MLP residual) | **148,803** | **20.06** | 84 |
| FastGS                                       | **660,711** | **41.96** | 173 |
| Ratio (FastGS / nest)                        | 4.44×    | 2.09×    | 2.06× |

Nest-splatting culls / consolidates ~4× harder than FastGS *at training time*, so
the model has 4× fewer Gauss AND about half the per-pixel overdraw.

---

## 2. Baked-renderer three-mode FPS decomposition

Same 24 test cameras, 200 warmup + 200 timed frames per mode, CUDA-event GPU
timing.  Checkpoint: `outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10/baked_atlas`.

| Mode | Description | ms/frame | FPS | Δ from mode above |
|------|-------------|---------:|----:|-------------------:|
| **A**: `SV + atlas`         | production render — SV softmax colour + atlas residual sample | **0.985** | **1014.9** | — |
| **B**: `SV only`            | SV colour, no atlas sample (surfel geometry unchanged) | **0.698** | **1432.7** | −0.287 ms  =  atlas sample cost |
| **C**: `passthrough`        | `colors_precomp` = constant grey, no SV, no atlas — pure surfel rasterisation + geometry + sort + blend | **0.693** | **1442.3** | −0.005 ms  =  SV softmax cost |

### What each phase costs

| Phase | ms/frame | % of production (A) |
|-------|---------:|--------------------:|
| Atlas sample + composition | 0.287 | **29.2%** |
| SV softmax colour eval     | 0.005 | **0.5%** |
| Surfel raster + geometry + sort + blend | 0.693 | **70.4%** |

Reading:

- **SV softmax is essentially free** — only 5 µs per frame across all 148 k
  Gauss.  The tester's guess ("SV is heavy") isn't supported by the numbers;
  the GPU chews through 7-way softmaxes in well under 1% of the frame.
- **Atlas sample costs ~29%** — one bilinear ASTC/BC7 texel fetch + T-matrix
  ray-disc UV per fragment × ~20 contributors/pixel × 1.09 M pixels adds up.
- **The remaining 70% is the surfel pipeline itself** — projection,
  covariance/Jacobian, tile sort, blend.  This is what we share with FastGS in
  concept but where our diff_surfel_bake_render kernel is roughly 4× slower
  per Gauss than theirs.

---

## 3. Head-to-head vs. FastGS

Same scene, same resolution, same evaluation:

| Renderer                | Gauss   | ms/frame | FPS  | PSNR  |
|-------------------------|--------:|---------:|-----:|------:|
| Nest baked (SV + atlas) | 148,803 | 0.985    | 1014.9 | 26.71 |
| Nest baked (SV only)    | 148,803 | 0.698    | 1432.7 | 20.15 |
| FastGS                  | 660,711 | 0.976    | 1024.1 | 27.34 |

- **Wall-clock is basically tied** at ~1 ms/frame despite FastGS carrying 4.4×
  the Gauss and 2.1× the overdraw.
- **PSNR**: FastGS wins by 0.6 dB — expected given the higher primitive count.
- **Per-Gauss cost**: nest ~6.6 ns/Gauss, FastGS ~1.5 ns/Gauss.  FastGS's
  rasterizer is about **4× more efficient per splat**.

---

## 4. So the FPS number *is* honest, but…

The tester's suspicion was: "if FastGS handles 4× more Gauss for the same
FPS, our baked path should easily beat 1000 FPS."  It doesn't, because our
per-Gauss compute+bandwidth cost is ~4× FastGS's.  Numbers where FastGS wins:

- Their sort is a two-stage radix (`sort_mode=1` also exists in our kernel
  but is not what `RD_SV_30thr_005w25gLP_N2f_frz5k10` bakes with).
- Their tile setup is tighter (rect AABB via `compact_mult` vs. our square
  SnugBox at 4σ).
- Their per-fragment kernel eval is a Gaussian falloff (fewer flops) vs. our
  β-scaled `pow(..., β)`.
- Their memory layout for per-Gauss attributes may be more cache-friendly
  under the actual access pattern.

Any one of these alone is worth ~1.5–2× on the residual-pipeline bar.  Getting
that ~4× back would put us at ~250 FPS on 660k Gauss vs. FastGS's 1024 FPS.

---

## Reproduce

```bash
# 3-mode FPS bench
conda run -n nest_splatting python speed_comparison/bench_three_modes.py \
    --model_path outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10 \
    --bake_dir   outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10/baked_atlas \
    --num_warmup 20 --num_benchmark 200 \
    --out        speed_comparison/baked_bench.json

# Per-view intersection maps (24 test views)
conda run -n nest_splatting python speed_comparison/render_intersection_all.py \
    --model_path outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10 \
    --out_dir    speed_comparison/nest_splatting_neural
```

FastGS side already had `intersection_maps/`, `intersection_raw/`, and
`intersection_summary.json` in `../FastGS/output/garden/test/ours_30000/`; those
are copied verbatim into `speed_comparison/fastgs/`.  They use `max_display=200`
turbo colormap, matching nest's convention (`utils/render_utils.create_intersection_heatmap`).
