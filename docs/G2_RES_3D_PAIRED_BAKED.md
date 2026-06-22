# `G2_10S15kL01` res_3d_paired — Baked Benchmark (mip_360, 5090)

Bake config: `--max_res 64 --bake_dtype bc7 --atlas_budget_mb 8192 --num_benchmark 100`.
All runs trained at 35k iters with `--method res_3d_paired --kernel beta_scaled --kernel2 gaussian --res_switch_iter 10000 --res_3d_iter 15000 --hybrid_levels 2 --aabb snugbox --feature SV --cold --fastgs --lowpass`.

The textured half (beta_scaled 2DGS surfels with hash+MLP residual) and untextured half (3D EWA ellipsoids, SV-only, FastGS-verbatim `computeCov3D`/`computeCov2D`) render in a single pass through `diff_surfel_bake_render_paired`.

## Results

| Scene | N (tex/untex) | Neural FPS | Baked FPS | Speedup | Neural PSNR | Baked PSNR | ΔPSNR | Neural SSIM | Baked SSIM | Neural LPIPS | Baked LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 280,464 (146,338/132,623) | 87.3 | 669.2 | 7.7× | 24.23 | 24.14 | −0.09 | 0.697 | 0.685 | 0.270 | 0.284 |
| bonsai | 200,915 (99,863/100,422) | 91.2 | 517.9 | 5.7× | 33.07 | 33.02 | −0.05 | 0.941 | 0.940 | 0.188 | 0.200 |
| counter | 147,679 (59,932/87,446) | 106.8 | 696.9 | 6.5× | 29.57 | 29.55 | −0.02 | 0.903 | 0.902 | 0.200 | 0.210 |
| flowers | 357,302 (191,792/164,327) | 78.7 | 526.6 | 6.7× | 20.59 | 20.74 | +0.15 | 0.551 | 0.556 | 0.331 | 0.339 |
| garden | 287,978 (127,318/160,325) | 100.4 | 827.7 | 8.2× | 27.06 | 27.02 | −0.04 | 0.840 | 0.828 | 0.136 | 0.154 |
| kitchen | 273,719 (113,073/160,300) | 84.8 | 516.7 | 6.1× | 31.72 | 31.68 | −0.04 | 0.922 | 0.921 | 0.128 | 0.139 |
| room | 131,243 (59,715/70,772) | 120.8 | 922.7 | 7.6× | 31.55 | 31.51 | −0.04 | 0.914 | 0.911 | 0.216 | 0.232 |
| stump | 170,614 (88,729/81,604) | 105.7 | 615.0 | 5.8× | 25.78 | 25.74 | −0.04 | 0.731 | 0.730 | 0.253 | 0.261 |
| treehill | 339,292 (179,438/158,417) | 84.2 | 543.0 | 6.5× | 22.43 | 22.41 | −0.02 | 0.603 | 0.597 | 0.325 | 0.342 |
| **mean** | — | **95.5** | **648.4** | **6.8×** | **27.33** | **27.31** | **−0.02** | **0.789** | **0.785** | **0.227** | **0.240** |

## Notes

- Roughly 50/50 textured/untex split — the EWA half adds capacity at near-zero atlas cost (untex Gausses fold into the skip-texture set, zero atlas rect, SV-only colour at render time).
- Bake quality loss is **−0.02 dB mean** vs neural renderer; flowers actually *gains* 0.15 dB (likely because the BC7 quant happens to favour the heavy-tail residual distribution in that scene).
- 1557×1038 render @ 5090, 100-frame mean over the test cameras.
- `--sort_mode 0` (legacy 64-bit single sort), `--aabb_mode 3` defaults — see audit notes below for the next round of perf work.

## Optimization pass (Jun 2026)

After auditing `diff_surfel_bake_render_paired` against FastGS, three changes
were applied to the bake-render kernel (no re-bake required — same baked
assets, new render-time code path):

1. **AccuTile SnugBox extended to `aabb_mode=3` (AdR mode)** —
   `forward.cu:640`. Previously SnugBox was gated to modes 2 and 5 (fixed-
   cutoff rect). Now AdR's opacity-aware r_beta cutoff feeds into
   `compute_conic_from_transmat()` the same way the fixed 4σ does, and
   AccuTile then enumerates only the tiles where the AdR-shrunken ellipse
   actually intersects. The two tightening effects compound.
2. **FMA-fused EWA Mahalanobis** at `forward.cu:921` — rewrote
   `me = ax² + 2bxy + cy²` as a single `fmaf(fmaf(2b, y, ax), x, cy²)` chain.
   Marginal but free.
3. **`--sort_mode 0` is the right default for this workload** — the
   FastGS-style two-stage sort (32-bit depth on n_visible + 32-bit tile on
   n_instances) was tested but regresses ~7% on this paired EWA+textured
   workload. The EWA half has high tile-touch redundancy (3D ellipsoids
   cover more tiles per Gauss than 2DGS discs), so the two-stage scheme's
   extra prefix-sum + duplicate-emit overhead exceeds its sorting savings.
   Default stays at legacy 64-bit single sort.

### Baseline vs optimized (same baked assets)

| Scene | N | Baseline FPS | Optimized FPS | ΔFPS | PSNR (both) |
|---|---|---|---|---|---|
| bicycle | 280,464 | 669.2 | 696.6 | **+4.1%** | 24.14 / 24.14 |
| bonsai | 200,915 | 517.9 | 536.8 | **+3.7%** | 33.02 / 33.02 |
| counter | 147,679 | 696.9 | 727.5 | **+4.4%** | 29.55 / 29.55 |
| flowers | 357,302 | 526.6 | 544.7 | **+3.4%** | 20.74 / 20.74 |
| garden | 287,978 | 827.7 | 848.4 | **+2.5%** | 27.02 / 27.02 |
| kitchen | 273,719 | 516.7 | 544.1 | **+5.3%** | 31.68 / 31.68 |
| room | 131,243 | 922.7 | 958.9 | **+3.9%** | 31.51 / 31.51 |
| stump | 170,614 | 615.0 | 629.4 | **+2.3%** | 25.74 / 25.74 |
| treehill | 339,292 | 543.0 | 566.1 | **+4.3%** | 22.41 / 22.41 |
| **mean** | — | **648.4** | **672.5** | **+3.7%** | **27.31 / 27.31** |

PSNR/SSIM/LPIPS bit-identical to 2 decimal places — pure binning optimisation,
no quality difference.

### Counter scene — isolation experiments

To split which change drives the gain (counter scene, vs baseline 696.9 FPS):

| Config | FPS | Δ |
|---|---|---|
| baseline (AdR, sort=0, no FMA) | 696.9 | — |
| + FMA only | 693.5 | −0.5% (noise) |
| + sort=1 (FastGS two-stage), + FMA | 648.5 | **−7%** (sort regression) |
| + AdR+SnugBox combined, + sort=1, + FMA | 675.3 | −3% |
| **+ AdR+SnugBox combined, + sort=0, + FMA** (shipping) | **727.2** | **+4.3%** |

So the win is **AdR + AccuTile SnugBox combined** (~4-5% on average); FMA is
neutral; sort=1 is a regression for this workload. EWA-side perf was already
near-optimal (audit confirmed opacity-aware Mahalanobis + per-axis tight
rect AABB + AccuTile SnugBox already wired at `forward.cu:688, 696-697, 706`).
