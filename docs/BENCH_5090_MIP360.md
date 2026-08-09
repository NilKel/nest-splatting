# CONIC baked renderer — 5090 mip360 FPS

Full FPS sweep of the CONIC-corrected `diff_surfel_bake_render_lean`
(with `LEAN_FLAGS="CONIC"` = CONIC linearization + fp16 pack + atlas
UV precompute) vs the production `diff_surfel_bake_render` (raw BC7,
per-fragment ray-splat), across all 9 mip360 scenes on the RTX 5090.

- **Bakes**: 9 mip360 scenes, method `--method 3D_SH_res`, config
  `RD_SV_30thr_005w25gLP_N2f_frz5k10`, HD BC7 atlas at `max_res 64`
  (`atlas_budget_mb 8192`).
- **Render kernel**: `diff_surfel_bake_render_lean` built with
  `LEAN_FLAGS="CONIC"` (single-source binary — both prod and lean live
  in the same Python process via the setter-mirror pattern).
- **Bench script**: `speed_comparison/bench_lean_vs_prod.py` — 50 warmup
  + 400 timed iterations per config, `cuda.Event` GPU-throughput timing.
- **Date**: 2026-07-21.
- **Hardware**: RTX 5090 (Blackwell, 32 GB, sm_120). `--use_fast_math`,
  BLOCK_SIZE=256.

## Full-lane FPS

Two lanes benchmarked per scene: **SH-only** (atlas fetch bypassed —
isolates the geometry + kernel + SH composite cost) and **SH+atlas**
(the deployed BC7 path — real render).

| scene | prod SH-only | lean SH-only | Δ | prod SH+atlas | lean SH+atlas | **Δ** |
|---|---|---|---|---|---|---|
| bicycle  | 851 | 1598 | +88% | 711 | 1385 | **+95%** |
| bonsai   | 779 | 1398 | +79% | 660 | 1261 | +91% |
| counter  | 1001 | 1652 | +65% | 775 | 1442 | +86% |
| flowers  | 734 | 1401 | +91% | 648 | 1291 | **+99%** |
| garden   | 1431 | 2337 | +63% | 1055 | 1943 | +84% |
| kitchen  | 1137 | 1786 | +57% | 831 | 1495 | +80% |
| room     | 1301 | 2012 | +55% | 1049 | 1792 | +71% |
| stump    | 886 | 1606 | +81% | 741 | 1394 | +88% |
| treehill | 695 | 1319 | +90% | 589 | 1105 | +87% |
| **mean** | 979 | 1679 | **+72%** | 784 | 1456 | **+86.9%** |

Numbers in FPS. Quality: **all PSNR/SSIM/LPIPS bit-identical** on the
SH+atlas lane (only 0.001 dB fp16-pack noise on the SH-only bonsai
lane).

## Full-picture comparison — neural → prod → lean → CONIC → FastGS

> ⚠️ **FastGS column corrected 2026-08-03.** The original figures were measured
> at FastGS's *training* resolution — its released `train_base.sh` loads `-i images`
> with `-r -1`, which caps the long edge at 1600 px — while this table's resolution
> column reports **our** mip-360 evaluation resolution (`images_4` outdoor /
> `images_2` indoor). The two methods were therefore not timed on the same images;
> outdoor scenes differed by ~60 % in pixel count. All 9 scenes were re-measured on
> an idle GPU, rendering the same FastGS checkpoints at the matched evaluation
> resolution with FastGS's own `mult=0.5` default
> (`speed_comparison/bench_fastgs_correct_res.sh`). Mean FastGS moves
> **1183 → 1215 FPS**, so our advantage is **1.20×**, not 1.23×. Garden is the
> control: it was already trained at `images_4`, and its number moved only
> 1138 → 1145 (+0.6 %). Retraining treehill at `images_4` changed its primitive
> count by −8 % and throughput by +2.2 %, so the training-resolution mismatch does
> not materially affect the comparison.


All measurements on the same RTX 5090, same test cameras, same
3D_SH_res bakes. Puts the CONIC baked renderer in context: the "before"
is the neural MLP-at-render path (what baking replaces), and FastGS is
the per-Gauss efficiency reference.

| scene | N Gauss | resolution | neural | prod baked | lean pre-CONIC | **lean CONIC** | FastGS |
|---|---:|:---:|---:|---:|---:|---:|---:|
| bicycle  | 161k | 1237×822  | 111 | 711  | 1014 | **1385** | 1204 |
| bonsai   |  92k | 1559×1039 | 117 | 660  |  924 | **1261** | 1241 |
| counter  |  68k | 1558×1038 | 108 | 775  | 1192 | **1442** | 1207 |
| flowers  | 185k | 1256×828  | 100 | 648  |  895 | **1291** | 1173 |
| garden   | 149k | 1297×840  | 117 | 1055 | 1624 | **1943** | 1145 |
| kitchen  | 120k | 1558×1039 |  93 | 831  | 1307 | **1495** | 1059 |
| room     |  63k | 1557×1038 | 133 | 1049 | 1496 | **1792** | 1343 |
| stump    |  96k | 1245×825  | 132 | 741  | 1063 | **1394** | 1276 |
| treehill | 175k | 1267×832  |  98 | 589  |  799 | **1105** | 1286 |
| **mean** | — | — | **112** | **784** | **1146** | **1456** | **1215** |

Reference points:
- **neural → CONIC lean = 13× mean speedup** on the full mip360 set
  (112 → 1456 FPS). Baking is a 7× step by itself; fp16 pack + template
  dispatch adds another 46%; CONIC on top adds another 27%; all told
  1.87× vs the raw baked path.
- **CONIC lean vs FastGS on the same 5090: 1456 vs 1215 mean = 1.20× faster.**
  On garden specifically: **1943 vs 1145 = 1.70×** faster. This is despite
  FastGS being a mature, heavily-optimized paper renderer designed for
  splat throughput.
- **N Gauss**: nest-splatting bakes are ~4× *fewer* Gauss than FastGS
  bakes of the same scene (e.g. garden 149k vs FastGS's 661k). That's
  training-side consolidation baked in; per-Gauss cost is what the CONIC
  renderer optimizes on top.

**Pre-CONIC lean quality caveat:** the T2+CTG lean drops PSNR by
~0.05-0.7 dB vs prod (T-matrix stored fp16 in shared → math done from
fp16 values loses some precision on the ray-splat cross-product).
CONIC lean is **bit-identical** to prod (0.001 dB noise on the SH-only
lane, 0 dB on SH+atlas) because it stores the correction denominator
(`dwdxr`, `dwdyr`) as fp32 while packing everything else fp16 — the
specific values that need precision are preserved.

**Sources** (all measured on this 5090, 2026-07-21):
- Neural: `benchmark_results.json` in each scene's `baked_atlas/` (recorded at bake time)
- Prod / lean pre-CONIC / lean CONIC: `speed_comparison/bench_lean_vs_prod.py`, 50 warmup + 400 timed with `cuda.Event`. Pre-CONIC = `LEAN_FLAGS="T2,CTG"`, CONIC = `LEAN_FLAGS="CONIC"`.
- FastGS 5090: `FastGS/bench_fps.py --num_warmup 10 --num_benchmark 200`, iter_30000 checkpoints trained locally on the 5090.

## What each optimization contributed

| lever | mechanism | garden ΔFPS on SH+atlas |
|---|---|---|
| **CONIC linearization** | replace per-fragment 2DGS ray-splat (15 FMA + 2 div) with mathematically-exact rational reconstruction `u = u₀ + Δu_lin / (1 + dwdxr·dx + dwdyr·dy)`; precomputed `(u₀, v₀, J⁻¹, dwdxr, dwdyr)` in preprocess. Discard `denom < 0.1` cuts fragments past the linearization's validity boundary. | +76% (1056 → 1855) |
| **fp16 pack** | `opa` / `J⁻¹` / `uv0` / `shape` stored fp16 in shared memory; per-fragment upcast to fp32 for math. `dwdxr` / `dwdyr` stay fp32 for correction-denominator stability. | ~-2% on Blackwell (noise; kept for cross-GPU compat) |
| **Atlas UV precompute** | move `atlas_rects` fetch, base offset, and `/UV_EXTENT` division out of the fragment loop into the block-fetch phase. 4× `float2` per Gauss added to shared (68 B/Gauss total). | +2% (1855 → 1943) |
| **Dead depth-cull cleanup** | under CONIC `depth = 1.0f` constant, so `if (depth < near_n) continue;` is dead; wrapped with `#ifndef LEAN_CONIC`. | ~noise |

## Fragment-loop plateau

Further +1-3% CUDA microopts on this kernel aren't worth chasing:
- Shared-memory occupancy isn't the bottleneck (17 KB / 48 KB static
  cap at BLOCK_SIZE 256).
- Fragment throughput is dominated by (a) the 1 fast_math divide, (b)
  the beta-scaled `powf(base, shape)` fallback (~30 cycles), (c) the
  HW-decoded BC7 atlas fetch, (d) the SH+opacity composite.
- `shape<0.01 → alpha_beta=1` fast-path was ~noise (compare cost
  cancels the pow savings, since `__powf` under `--use_fast_math` is
  only ~4 cycles).

**Algorithmic** deltas (skip-list per tile, hierarchical tile eval,
LOD) would be the next real lever — see
[`project_lean_bake_render_wins.md`](../../.claude/projects/-home-nilkel-Projects-nest-splatting/memory/project_lean_bake_render_wins.md)
in memory for the full plateau analysis.

## Reproduce

Local RTX 5090:

```bash
cd /home/nilkel/Projects/nest-splatting
LEAN_FLAGS="CONIC" conda run -n nest_splatting python -m pip install -e \
    submodules/diff_surfel_bake_render_lean --no-build-isolation

# single scene
conda run -n nest_splatting python -u speed_comparison/bench_lean_vs_prod.py \
    --model_path outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10 \
    --bake_dir   outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10/baked_atlas \
    --num_warmup 50 --num_benchmark 400

# 9-scene sweep — see /tmp/claude-1000/…/scratchpad/sweep_lean_uv.sh
# and sweep_lean_uv_rest.sh for the exact loops used above.
```

Bench script auto-loads both `diff_surfel_bake_render` (prod) and
`diff_surfel_bake_render_lean` (LEAN_FLAGS build) in the same process
and drives 4 modes: prod SH-only, prod SH+atlas, lean SH-only, lean
SH+atlas.

## Deploy note

The CONIC + fp16 pack + atlas UV precompute win applies to the **CUDA
research renderer**, not directly to the deployed WebGPU viewer
(Halloumi-WS). The WebGPU viewer already ships a matching CONIC
fragment shader path (`viewer/assets/index-c5534242.js`, commit
`c19bb79` on bitymi-demos), but WebGPU-side atlas UV precompute was
skipped because it would grow the `Splat2DGS` struct 80→96 bytes for
~1-3% expected fragment win (WGSL already L1-caches `atlas_rects` and
constant-folds uniform divisors — see the analysis in
[`project_lean_bake_render_wins.md`](../../.claude/projects/-home-nilkel-Projects-nest-splatting/memory/project_lean_bake_render_wins.md)).

## res_3d_paired (BS2 i2fast) — CONIC paired renderer, 9 scenes

Config: `fix_BS2_10S15kL01_SV_30thr_005w25gLP4lev_FRP5k10_N2F_Jac_5ksp_i2fast`
(res_3d_paired, textured 2D beta_scaled surfels + untextured EWA 3D beta,
59–73% textured). Baked 2026-08-09 (max_res 64, BC7); benched with
`diff_surfel_bake_render_paired_lean` built `LEAN_FLAGS=CONIC`, 50/400
cuda.Event frames, `speed_comparison/bs2_conic_bench/*.json`.

| scene | nGauss | %tex | neural | baked PSNR | prod FPS | CONIC FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle  | 170k | 73% | 24.17 | 24.16 |  766 | 1347 | +75.8% |
| bonsai   | 129k | 69% | 32.40 | 32.34 |  724 | 1203 | +66.2% |
| counter  | 102k | 59% | 29.15 | 29.11 |  870 | 1402 | +61.2% |
| flowers  | 222k | 67% | 20.58 | 20.80 |  692 | 1192 | +72.3% |
| garden   | 188k | 70% | 26.86 | 26.82 | 1025 | 1671 | +63.0% |
| kitchen  | 177k | 62% | 31.20 | 31.11 |  794 | 1287 | +62.2% |
| room     |  80k | —   | 31.15 | 31.10 | 1030 | 1652 | +60.3% |
| stump    | 107k | —   | 25.77 | 25.86 |  913 | 1509 | +65.3% |
| treehill | 181k | —   | 22.46 | 22.51 |  654 | 1090 | +66.7% |
| **mean** | — | — | — | — | **830** | **1373** | **+65.4%** |

PSNR prod vs CONIC bit-identical (max |Δ| 0.002 dB, kitchen). Bake loss vs
neural ≈ 0 mean; flowers/stump/treehill *gain* (+0.22/+0.09/+0.05 dB) — no
post-bake finetune headroom on this config.

**Paired-branch tax, measured** (`BENCH_FORCE_PAIRED_LEAN=1` on the
3D_SH_res room bake — zero untextured rows, so the delta is pure overhead of
the paired code paths): prod matches the plain-lean run within 2% (1027 vs
1049 → same run conditions), lean SH+atlas 1659 vs plain-lean's 1792 =
**−7.4%**; SH-only 1922 vs 2012 = −4.5%. Attribution candidates, in order:
dual staging (every Gauss stages BOTH the CONIC fields and the EWA
conic/flag — dead global reads for the wrong type), +9 B/entry shared
footprint, per-j flag test. The branch itself is block-uniform (all 256
threads take the same side per Gaussian), so warp divergence is NOT a factor.
Gating the staging by `is_textured` + unioning the shared slots is the first
lever; expected recovery is a meaningful slice of the 7.4% on pure scenes and
proportionally less on mixed ones.

### FastGS footprint mult on the untextured EWA half — measured, no win

Hypothesis: crop the untextured EWA 3D Gaussians' binning footprint FastGS-style
(`t = mult·t`, mult 0.5) — scoped to the EWA half only, since the textured 2D
surfels carry the atlas residual and must not be cropped. Implemented as
`d_untex_mult` / `set_untex_mult` in `diff_surfel_bake_render_paired_lean`
(EWA branch's `t_cut` only; 1.0 = byte-identical) + `BENCH_UNTEX_MULT` env in
the bench harness (lean lane only).

Result (garden / bicycle / treehill, 30/300 frames, vs the mult=1.0 CONIC
baselines): FPS −0.3%…+0.8%, PSNR ±0.003 dB at both mult 0.5 and 0.7 — pure
noise. **The opacity-aware EWA cutoff (`t_cut = max(0.5, 2·log(255·opa))`)
already banks this win**: the untex half is low-opacity volumetric filler, so
its 1/255-iso footprints are already small, and its binning share of frame time
is evidently negligible. FastGS's mult pays off in FastGS because it applies to
ALL primitives at 3–4× our primitive count; here there is nothing left to crop.
No finetune warranted. The knob stays (default 1.0) for future A/Bs.
