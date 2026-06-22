# 5090 Baked-Atlas Bench Results

Companion to [`4090_BENCH_RESULTS.md`](4090_BENCH_RESULTS.md). The
**exhaustive per-config 5090 sweep** (all 6 SB/SV configs, full per-scene
tables, the fused-path validations) lives in
[`MIP360_BENCH_RESULTS.md`](MIP360_BENCH_RESULTS.md) — this file is the
hardware-headed summary that mirrors the 4090 doc's layout for direct
side-by-side reading. Numbers here are baked SH+atlas (`baked_sh_atlas`
field of `benchmark_results.json`); the neural column is the training-time
renderer.

**Hardware**: NVIDIA GeForce RTX 5090 (Blackwell / sm_120), 32 GB VRAM,
nilkel-Workstation (10.176.128.124), torch 2.11 / CUDA 12.8 build of
`diff_surfel_bake_render`, BC7 atlas + (torch or fused) SV path.

**Methodology** (identical to the 4090 setup so the two files compare
1:1): `torch.cuda.Event(enable_timing=True)` start/end + explicit
`cuda.synchronize`, `--num_warmup 10 --num_benchmark 200`, cycling
through `getTestCameras()`. No display layer (no vsync) → this is the
SMERF / Duckworth-2023 "min over k redraws" floor, not wall-clock.
LPIPS via the in-tree `lpipsPyTorch` (VGG net), bit-identical to the
4090's LPIPS path.

**Pipeline**: unlike the 4090 (which has no nest-splatting source and is
fed pre-built bundles over rsync), the 5090 runs the **full pipeline
natively** — `scripts/benchmark_baked.py` bakes the atlas and benchmarks
in one process on the same machine. No bundle step.

**Atlas / runtime config**: `--max_res 64 --atlas_budget_mb 8192
--aabb_mode 5 --bake_dtype bc7 --sort_mode 0`. Identical to the 4090 sweep.

## `SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` (production / paper config)

The head-to-head config: `--feature SV` (Spherical Voronoi, K=7
sites/Gaussian), error-weighted overdraw regularizer, c2f, Jacobian
contraction, freeze-hash 5k/10. Same config the 4090 file headlines.
Table uses the **torch fake-SH-DC** SV path for apples-to-apples with the
SB configs; the fused-CUDA path is ~+7 % faster at bit-identical output
(see below). Resolution per scene follows the mip-360 convention
(outdoor `images_4` ≈ 1237–1297 px wide, indoor `images_2` ≈ 1557–1559).

### Mip-NeRF 360

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle  | 23.84 | 0.6679 | 0.3007 | 722.2 | 549 | 23.87 / 97.3 |
| bonsai   | 31.94 | 0.9290 | 0.2064 | 576.4 | 284 | 32.11 / 93.5 |
| counter  | 28.98 | 0.8875 | 0.2214 | 863.6 | 185 | 29.00 / 93.0 |
| flowers  | 20.56 | 0.5370 | 0.3516 | 559.2 | 685 | 20.39 / 83.0 |
| garden   | 26.65 | 0.8137 | 0.1732 | 1014.5 | 494 | 26.60 / 103.1 |
| kitchen  | 30.62 | 0.9043 | 0.1559 | 697.7 | 215 | 30.65 / 76.3 |
| room     | 30.40 | 0.9022 | 0.2402 | 1037.9 | 225 | 30.31 / 112.7 |
| stump    | 25.62 | 0.7169 | 0.2763 | 689.1 | 333 | 25.59 / 107.9 |
| treehill | 22.34 | 0.5855 | 0.3494 | 560.6 | 736 | 22.28 / 76.8 |
| **MEAN** | **26.77** | **0.7715** | **0.2528** | **746.8** | **412** | **26.76 / 93.7** |

## `PerGS2_SV_30thr_005w25gLP4lev_FRP5k10_C2F_Jac_5ksp` (`--method mixed_3d`, 5k texsplit)

First mip-360 entry using the `--method mixed_3d` textured/untextured manifold
split (`--texsplit 5000`). Same `--feature SV` + error-weighted overdraw + c2f
+ Jacobian contraction + freeze-hash recipe as the production config above —
but at iter 5000 each surfel is duplicated into a textured copy
(2DGS ray-splat + hash+MLP residual + `--kernel beta_scaled`) and an
untextured copy (**3D EWA ellipsoid** via `computeCov3D`/`computeCov2D` from a
learnable `_scaling_z`, SV-only baseline, `--kernel2 gaussian`). After the
split, training drives the textured half toward diffuse manifolds and the
untextured half toward view-dependent / specular volume — atlas residual lives
only on the textured rows; untextured rows are folded into the bake's
skip-texture set (zero atlas rect) and rendered with the EWA-aware bake
kernel. The "% textured" column shows the share of Gaussians that carry an
atlas residual.

Bench methodology note: this run used the `bake_all_scenes.sh` default
`--num_warmup 5 --num_benchmark 50` instead of the production
10/200 — FPS numbers are ~1–3 % noisier but PSNR/SSIM/LPIPS are unaffected.

**Updated numbers** post the **EWA AccuTile** patch
([`diff_surfel_bake_render/cuda_rasterizer/forward.cu:617`](../submodules/diff_surfel_bake_render/cuda_rasterizer/forward.cu#L617)):
untextured 3D ellipsoids now use the same `duplicateToTilesTouched`
ellipse-tight binning the textured 2DGS surfels already used (square AABB +
3σ-of-maxλ → per-axis tight rect + opacity-aware Mahalanobis cutoff +
AccuTile). Bit-equivalent on most scenes; +0.4–0.5 dB on kitchen/room where
the looser binning was over-compositing low-α specular contributors.

### Mip-NeRF 360

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS | % textured |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle  | 24.26 | 0.6921 | 0.2862 | 1137.6 | 303 | 24.36 / 100.7 | 34% |
| bonsai   | 32.83 | 0.9389 | 0.2015 |  926.2 | 182 | 32.89 / 103.1 | 35% |
| counter  | 29.57 | 0.9023 | 0.2092 | 1144.0 |  87 | 29.59 / 111.3 | 30% |
| flowers  | 20.61 | 0.5566 | 0.3418 |  951.4 | 462 | 20.55 /  94.0 | 50% |
| garden   | 26.65 | 0.8264 | 0.1587 | 1424.6 | 261 | 26.70 / 111.5 | 34% |
| kitchen  | 31.50 | 0.9214 | 0.1353 |  919.3 | 101 | 31.51 /  94.3 | 28% |
| room     | 31.43 | 0.9106 | 0.2316 | 1463.3 | 115 | 31.46 / 127.0 | 34% |
| stump    | 25.73 | 0.7302 | 0.2625 |  973.6 | 235 | 25.73 / 116.0 | 48% |
| treehill | 22.43 | 0.6039 | 0.3496 |  919.6 | 462 | 22.41 /  96.0 | 47% |
| **MEAN** | **27.23** | **0.7869** | **0.2418** | **1095.5** | **245** | **27.24 / 106.0** | **38%** |

Three findings worth flagging:

- **PSNR is now at neural parity** (27.23 vs 27.24 — −0.01 dB mean). The
  EWA-AccuTile patch lifted kitchen +0.43 dB and room +0.52 dB by no longer
  over-binning low-α specular contributors into tiles whose pixels would be
  α-floor-culled anyway. The other 7 scenes are bit-equivalent (±0.02 dB FP
  noise from the tile-reordering accumulation).
- **+55 % FPS over the previous mixed_3d binning** (706 → 1095.5 mean).
  Outdoor scenes with broad EWA tile coverage gained most: bicycle +74 %,
  treehill +65 %, room +65 %; even the dense indoor scenes are +41–47 %.
- **Atlas is ~40 % smaller** (245 MB mean vs 412 MB for the matched
  `SV_30thr…c2f_Jac` row below). Only the ~38 % textured Gaussians carry
  atlas storage; the 62 % untextured EWA Gaussians cost zero atlas
  (skip-texture, zero rect).

## Cross-config means (all 9 mip-360 scenes)

The full per-scene tables for every config are in
[`MIP360_BENCH_RESULTS.md`](MIP360_BENCH_RESULTS.md); this is the rollup.

| config | baked PSNR | baked FPS | atlas (MB) | neural PSNR | neural FPS | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `SB_10thr_0w0gLP4levno2f_FRP5k10`        | 26.98 | 315.8 | 665 | 27.03 | 54.1 | **5.83×** |
| `SB_10thr_005w25gLP4levno2f_FRP5k10`     | 26.53 | 630.9 | 611 | 26.57 | 82.4 | **7.66×** |
| `SB_10thr_01w25g4levN2F_FRP5k10`         | 26.25 | 757.7 | 559 | 26.29 | 92.4 | **8.20×** |
| `SB_5thr_01w25g4levN2F_FRP5k10`          | 26.29 | 730.7 | 656 | 26.34 | 88.4 | **8.26×** |
| `SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` | 26.77 | 746.8 | 412 | 26.76 | 93.7 | **7.97×** |
| `SB_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` | 26.47 | 771.7 | 412 | 26.51 | 92.0 | **8.39×** |
| `PerGS2_SV_30thr_005w25gLP4lev_FRP5k10_C2F_Jac_5ksp` (`mixed_3d` + EWA-AccuTile) | **27.23** | **1095.5** | **245** | 27.24 | 106.0 | **10.34×** |

Baked inference is **~6–8× faster than the neural renderer** at ≤0.05 dB
PSNR cost for the SV/SB rows; the 30 %-prune `_c2f_Jac` configs give the
smallest atlases (412 MB mean) at the top of the FPS band. The new
`mixed_3d` row (with EWA-AccuTile) is now **strictly Pareto-dominant** over
every other config in the table — **smallest atlas (245 MB), highest FPS
(1095.5), highest PSNR (27.23 = neural parity)** — because EWA-AccuTile
tightened the untextured-half tile coverage by ~55 % at no quality cost, and
the bake-quant loss only ever hit textured rows in this mode.

## `res_3d_paired` vs `3D_SH_res` (9-scene per-config split)

Same scene set, five configs side by side: the two `3D_SH_res` regularization
points (`0w0` = no w-reg, `005w25` = production w_lambda=0.005, γ=25) against
the three `res_3d_paired` variants from the 35k-iter `_5ksp_i2*` runs.

- **G2**: `--kernel beta_scaled --kernel2 gaussian` — textured 2DGS beta_scaled +
  untextured 3D EWA Gaussian.
- **BS2**: `--kernel beta_scaled --kernel2 beta_scaled` — both halves use the
  compact-support beta_scaled cutoff.
- **BS2_fast**: same config as BS2, "fast" trained variant — slightly fewer
  Gaussians (~16 % fewer than BS2 mean).

Bake config for all five: `--max_res 64 --bake_dtype bc7 --atlas_budget_mb 8192
--num_benchmark 100`. Render path: aabb_mode 3 (rect+AdR+SnugBox), sort_mode 0,
FMA-fused EWA Mahalanobis.

### Baked FPS

| Scene | 0w0 | 005w25 | G2 | BS2 | BS2_fast |
|---|---:|---:|---:|---:|---:|
| bicycle  | 510.3 |  999.0 | 696.6 |  798.7 |  847.5 |
| bonsai   | 311.4 |  576.4 | 536.8 |  580.5 |  634.1 |
| counter  | 523.5 |  863.6 | 727.5 |  919.3 |  908.0 |
| flowers  | 279.6 |  559.2 | 544.7 |  634.7 |  692.6 |
| garden   | 533.9 | 1111.7 | 848.4 | 1029.7 | 1053.9 |
| kitchen  | 451.4 |  697.7 | 544.1 |  667.7 |  727.4 |
| room     | 611.1 |  886.1 | 958.9 | 1160.0 | 1186.9 |
| stump    | 360.8 |  689.1 | 629.4 |  683.4 |  858.8 |
| treehill | 267.7 |  560.6 | 566.1 |  630.1 |  695.5 |
| **mean** | **427.7** | **771.5** | **672.5** | **789.3** | **845.0** |

### Baked PSNR (dB)

| Scene | 0w0 | 005w25 | G2 | BS2 | BS2_fast |
|---|---:|---:|---:|---:|---:|
| bicycle  | 24.14 | 23.84 | 24.14 | 23.94 | 23.96 |
| bonsai   | 32.84 | 31.94 | 33.02 | 32.60 | 32.35 |
| counter  | 29.41 | 28.98 | 29.55 | 29.22 | 29.12 |
| flowers  | 20.68 | 20.56 | 20.74 | 20.65 | 20.78 |
| garden   | 26.98 | 26.65 | 27.02 | 26.79 | 26.76 |
| kitchen  | 31.55 | 30.62 | 31.68 | 31.28 | 31.06 |
| room     | 31.06 | 30.37 | 31.51 | 31.10 | 31.19 |
| stump    | 25.83 | 25.62 | 25.74 | 25.75 | 25.78 |
| treehill | 22.36 | 22.34 | 22.41 | 22.40 | 22.54 |
| **mean** | **27.21** | **26.77** | **27.31** | **27.08** | **27.06** |

**Reading the rollup:**
- **PSNR order:** G2 (27.31) > 0w0 (27.21) > BS2 (27.08) ≈ BS2_fast (27.06) > 005w25 (26.77).
- **FPS order:** BS2_fast (845.0) > BS2 (789.3) > 005w25 (771.5) > G2 (672.5) > 0w0 (427.7).
- `0w0` is the quality king of the 3D_SH_res baselines but slowest — no overdraw
  penalty lets surfels grow large → high per-pixel contributor count.
- `005w25` is the fastest 3D_SH_res config but pays ~0.4 dB vs `0w0` for the
  surfel-footprint compression.
- **G2 strictly beats both 3D_SH_res variants on PSNR** while sitting between
  them on FPS — adding the untex EWA half buys back the dB that w-reg gave up,
  at no per-Gauss capacity cost.
- **BS2 / BS2_fast** trade ~0.25 dB vs G2 for big FPS gains (beta_scaled's hard
  Mahalanobis cutoff at the untex half culls more pixels per Gauss). BS2_fast
  adds another +7.0 % FPS over BS2 essentially free (−0.02 dB mean).

## 5090-specific renderer-path findings

These optimizations were measured on the 5090 (`--skip_bake`, render path
only, bit-equivalent output verified):

- **Fused-CUDA SV vs torch fake-SH-DC** (production config): porting
  `computeColorFromVoronoi` into `preprocessCUDA` skips the per-frame
  autograd graph + fake-SH roundtrip → **+7 % mean FPS** (761.8 → 815.7
  over 9 scenes, peak +10 % on garden), PSNR identical to 4 decimals
  (cross-render max-abs 4.4e-4, FP16 noise). On a 5090 the rasterizer is
  already fast enough that the torch slack is small — the lift is larger
  on slower GPUs.
- **Fused-SB preprocess** (SB configs): moving `eval_sb` out of the
  per-pixel accumulation loop into `preprocessCUDA` (SB view-dir is
  per-Gaussian, was being recomputed ~100× per Gaussian) → **+10.95 %
  mean FPS** (range +7.6 % to +14.3 %), zero quality cost.

## 5090 vs 4090

The direct cross-hardware comparison table (005w25g, fused SV, lpipsPyTorch
on both) is maintained in
[`4090_BENCH_RESULTS.md` § 4090 vs 5090](4090_BENCH_RESULTS.md#4090-vs-5090-where-direct-comparison-exists).
Summary: PSNR/SSIM within ±0.12 dB / ±0.0014 (BC7 hardware-decode LSB drift
between Ada and Blackwell texture units), LPIPS bit-identical, and the 4090
runs at **0.81–0.91× the 5090's FPS** — roughly tracking the memory-
bandwidth ratio (5090 ≈ 1.79 TB/s vs 4090 ≈ 1.0 TB/s).

## Reproducibility / where the data lives

- **Native re-run (single scene)** on the 5090:
  ```bash
  conda run -n nest_splatting python scripts/benchmark_baked.py \
    --model_path outputs/mip_360/<scene>/3D_SH_res/<config> \
    --output_dir  outputs/mip_360/<scene>/3D_SH_res/<config>/baked_atlas \
    --max_res 64 --atlas_budget_mb 8192 --aabb_mode 5 --sort_mode 0 \
    --bake_dtype bc7 --num_warmup 10 --num_benchmark 200
  # add --skip_bake to reuse an existing baked_atlas (render-only A/B)
  ```
- **Per-scene JSON**: `<model_path>/baked_atlas/benchmark_results.json`
  (`baked_sh_atlas` / `baked_sh_only` / `neural` keys: psnr, ssim, lpips,
  fps, ms_per_frame).
- **Full per-config 5090 sweep + fused-path validation tables**:
  [`MIP360_BENCH_RESULTS.md`](MIP360_BENCH_RESULTS.md).
- **Neural-renderer (non-baked) FPS** for trained checkpoints
  (baseline-vs-cat, nexels, FastGS): [`FPS_BENCH_RESULTS.md`](FPS_BENCH_RESULTS.md).
