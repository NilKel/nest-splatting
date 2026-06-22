# G2 vs BS2 — Baked Benchmark Comparison (mip_360, res_3d_paired, 5090)

Both runs are `res_3d_paired` at 35k iters with identical training flags
*except* `--kernel2` (the kernel applied to the untextured 3D EWA half):

| Run | `--kernel`  (textured 2D surfels) | `--kernel2` (untextured 3D EWA ellipsoids) |
|---|---|---|
| **G2** | `beta_scaled` | `gaussian` |
| **BS2** | `beta_scaled` | `beta_scaled` |

The textured half is bit-identical between runs; only the EWA half's
fragment falloff (and tile-binning support) differs.

Bake config: `--max_res 64 --bake_dtype bc7 --atlas_budget_mb 8192 --num_benchmark 100`.
Render path: optimized bake-render (AdR + AccuTile SnugBox + FMA-fused EWA me,
`sort_mode=0`).

## Results

| Scene | N (G2/BS2) | G2 FPS | BS2 FPS | ΔFPS | G2 PSNR | BS2 PSNR | ΔPSNR | G2 SSIM | BS2 SSIM | G2 LPIPS | BS2 LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 280,464 / 220,132 | 696.6 | 798.7 | **+14.7%** | 24.14 | 23.94 | −0.20 | 0.685 | 0.674 | 0.284 | 0.298 |
| bonsai | 200,915 / 166,750 | 536.8 | 580.5 | **+8.2%** | 33.02 | 32.60 | −0.42 | 0.940 | 0.935 | 0.200 | 0.203 |
| counter | 147,679 / 121,591 | 727.5 | 919.3 | **+26.4%** | 29.55 | 29.22 | −0.32 | 0.902 | 0.893 | 0.210 | 0.217 |
| flowers | 357,302 / 296,180 | 544.7 | 634.7 | **+16.5%** | 20.74 | 20.65 | −0.09 | 0.556 | 0.548 | 0.339 | 0.343 |
| garden | 287,978 / 208,331 | 848.4 | 1029.7 | **+21.4%** | 27.02 | 26.79 | −0.22 | 0.828 | 0.818 | 0.154 | 0.165 |
| kitchen | 273,719 / 216,467 | 544.1 | 667.7 | **+22.7%** | 31.68 | 31.28 | −0.39 | 0.921 | 0.912 | 0.139 | 0.147 |
| room | 131,243 / 103,493 | 958.9 | 1160.0 | **+21.0%** | 31.51 | 31.10 | −0.41 | 0.911 | 0.905 | 0.232 | 0.237 |
| stump | 170,614 / 130,698 | 629.4 | 683.4 | **+8.6%** | 25.74 | 25.75 | +0.01 | 0.730 | 0.725 | 0.261 | 0.268 |
| treehill | 339,292 / 264,685 | 566.1 | 630.1 | **+11.3%** | 22.41 | 22.40 | −0.01 | 0.597 | 0.590 | 0.342 | 0.347 |
| **mean** | — | **672.5** | **789.3** | **+17.4%** | **27.31** | **27.08** | **−0.23** | **0.785** | **0.778** | **0.240** | **0.247** |

## Reading the result

- **BS2 is +17.4% FPS on average** (range: +8% bonsai/stump → +26% counter).
  The compact-support beta_scaled kernel culls pixels at Mahalanobis `m ≥ k²=9`
  for the untextured EWA half, dropping a lot of fragment work vs the
  Gaussian's `exp(−m/2)` falloff which only goes to zero asymptotically.
- **BS2 PSNR is −0.23 dB on average** with two outlier sets:
  - Indoor scenes (room, counter, kitchen, bonsai) lose **−0.32 to −0.42 dB** —
    the beta cutoff visibly clips soft falloff at object silhouettes.
  - Outdoor scenes (stump, treehill, flowers) are **≈ 0 dB** — the EWA
    untextured Gausses there are smaller/sharper to begin with, so the
    compact-support cut barely changes the integrated colour.
- **N (Gaussian count) is consistently lower in BS2** (15–25% fewer Gausses):
  beta_scaled's compact support during *training* causes fewer surfels to be
  retained / cloned by the densifier, since pixels beyond the cutoff don't
  contribute gradients. That's a secondary speedup driver — fewer Gausses,
  fewer instances, less sort work.

## Trade-off summary

| If you care about | Pick |
|---|---|
| Maximum FPS, willing to trade ~0.2-0.4 dB indoors | **BS2** |
| Best PSNR/SSIM (especially indoor), willing to give up ~17% FPS | **G2** |
| Mobile WebGPU viewer perf (fragment-bound on TBDR) | **BS2** likely the right default |

The mip_360 cards on the bitymi website currently ship G2 (default Mixed
toggle). Worth A/B testing BS2 as the mobile default since fragment work
dominates on mobile GPUs.
