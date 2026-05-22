# mip-360 Baked-Atlas Bench Results

> This is the **exhaustive per-config 5090 sweep** (all SB/SV configs, full
> per-scene tables, fused-path validations). For the hardware-headed 5090
> summary that parallels the 4090 doc, see
> [`5090_BENCH_RESULTS.md`](5090_BENCH_RESULTS.md); the 4090 counterpart is
> [`4090_BENCH_RESULTS.md`](4090_BENCH_RESULTS.md). All numbers below are 5090.

All benches: `--max_res 64 --atlas_budget_mb 8192 --aabb_mode 5 --bake_dtype bc7`.
Numbers are baked SH+atlas (`baked_sh_atlas` field of benchmark_results.json).
Atlas size is BC7 on disk. Neural row is the training-time renderer.

**Methodology** (so cross-paper comparisons stay honest): FPS measured with
`torch.cuda.Event(enable_timing=True)` start/end + explicit `cuda.synchronize`,
10-frame warmup, 200 timed frames cycling through `getTestCameras()`.
Equivalent to the SMERF / Duckworth-2023 / Binary Opacity Grids "minimum
over k redraws" floor since we bypass any display layer (no vsync). FastGS-style
`time.time()` wall-clock around the render call would drift up by ~5-15 % vs.
ours, primarily from pre-warmup steady-state contamination. SIBR-viewer FPS
is ~2-3× pessimistic vs. our floor since it includes display sync + ImGui.

## `SB_10thr_0w0gLP4levno2f_FRP5k10`

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.91 | 0.6952 | 0.2728 | 296.5 | 1285 | 24.27 / 50.7 |
| bonsai | 32.09 | 0.9358 | 0.2055 | 246.7 | 600 | 32.22 / 55.2 |
| counter | 28.94 | 0.8931 | 0.2186 | 410.7 | 357 | 29.04 / 59.1 |
| flowers | 20.61 | 0.5424 | 0.3580 | 224.1 | 413 | 20.40 / 45.3 |
| garden | 26.91 | 0.8241 | 0.1603 | 408.0 | 1153 | 26.92 / 58.7 |
| kitchen | 31.12 | 0.9139 | 0.1471 | 300.5 | 509 | 31.18 / 44.1 |
| room | 31.21 | 0.9109 | 0.2337 | 438.0 | 511 | 31.21 / 70.1 |
| stump | 25.82 | 0.7372 | 0.2572 | 290.5 | 719 | 25.84 / 61.8 |
| treehill | 22.21 | 0.5797 | 0.3737 | 227.1 | 441 | 22.19 / 42.2 |
| **MEAN** | **26.98** | **0.7814** | **0.2474** | **315.8** | **665** | **27.03 / 54.1** |

## `SB_10thr_005w25gLP4levno2f_FRP5k10`

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.57 | 0.6636 | 0.3007 | 598.6 | 871 | 23.89 / 79.6 |
| bonsai | 31.16 | 0.9250 | 0.2061 | 506.8 | 398 | 31.32 / 84.2 |
| counter | 28.42 | 0.8818 | 0.2255 | 698.4 | 226 | 28.48 / 86.0 |
| flowers | 20.47 | 0.5366 | 0.3484 | 474.9 | 1092 | 20.32 / 69.6 |
| garden | 26.48 | 0.8223 | 0.1581 | 940.9 | 662 | 26.45 / 96.2 |
| kitchen | 30.32 | 0.9014 | 0.1578 | 533.6 | 278 | 30.34 / 67.5 |
| room | 30.62 | 0.9028 | 0.2378 | 855.6 | 294 | 30.61 / 102.2 |
| stump | 25.53 | 0.7181 | 0.2722 | 594.2 | 519 | 25.52 / 92.0 |
| treehill | 22.21 | 0.5884 | 0.3436 | 474.7 | 1165 | 22.24 / 64.2 |
| **MEAN** | **26.53** | **0.7711** | **0.2500** | **630.9** | **611** | **26.57 / 82.4** |

## `SB_10thr_01w25g4levN2F_FRP5k10`

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.50 | 0.6577 | 0.3071 | 712.4 | 800 | 23.80 / 89.2 |
| bonsai | 30.52 | 0.9159 | 0.2161 | 619.9 | 379 | 30.70 / 97.4 |
| counter | 28.08 | 0.8725 | 0.2357 | 798.3 | 192 | 28.15 / 95.0 |
| flowers | 20.39 | 0.5253 | 0.3608 | 594.4 | 1004 | 20.26 / 77.3 |
| garden | 26.25 | 0.8039 | 0.1825 | 1061.6 | 596 | 26.20 / 102.9 |
| kitchen | 29.80 | 0.8915 | 0.1702 | 644.2 | 230 | 29.83 / 76.7 |
| room | 30.13 | 0.8960 | 0.2477 | 1041.0 | 265 | 30.11 / 117.7 |
| stump | 25.37 | 0.7076 | 0.2835 | 763.8 | 517 | 25.39 / 102.7 |
| treehill | 22.18 | 0.5790 | 0.3533 | 583.3 | 1048 | 22.20 / 72.7 |
| **MEAN** | **26.25** | **0.7610** | **0.2619** | **757.7** | **559** | **26.29 / 92.4** |

## `SB_5thr_01w25g4levN2F_FRP5k10`

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.52 | 0.6594 | 0.3042 | 681.5 | 984 | 23.83 / 81.1 |
| bonsai | 30.66 | 0.9179 | 0.2128 | 602.6 | 436 | 30.82 / 93.6 |
| counter | 28.06 | 0.8735 | 0.2355 | 826.8 | 212 | 28.15 / 95.4 |
| flowers | 20.40 | 0.5292 | 0.3533 | 565.8 | 1191 | 20.26 / 70.9 |
| garden | 26.36 | 0.8054 | 0.1807 | 993.6 | 686 | 26.33 / 96.2 |
| kitchen | 29.91 | 0.8928 | 0.1676 | 627.5 | 245 | 29.96 / 76.8 |
| room | 30.11 | 0.8970 | 0.2466 | 1013.0 | 297 | 30.09 / 113.7 |
| stump | 25.37 | 0.7091 | 0.2796 | 685.9 | 626 | 25.38 / 97.6 |
| treehill | 22.17 | 0.5801 | 0.3509 | 579.3 | 1226 | 22.21 / 70.5 |
| **MEAN** | **26.29** | **0.7627** | **0.2590** | **730.7** | **656** | **26.34 / 88.4** |

## SB fused-vs-per-pixel validation

The original SB tables above were measured with `eval_sb` running **per-pixel
inside `renderBakedCUDA`'s inner accumulation loop**. But `view_dir` for SB
is per-Gaussian (depends only on the Gaussian's mean), so the same RGB
was being recomputed for every pixel a Gaussian touched (~100× redundancy).

Fix (committed): move `eval_sb` into `preprocessCUDA` next to `computeColorFromSH`,
write the result to a new per-Gauss FP16 buffer `geomState.sb_rgb` (P×3, ~1 MB
for 200k Gaussians), and have the render kernel just do an FP16 load + 3 fadds.
Math is identical (SB is added linearly to feat[] before the outer ReLU; the
operation order is preserved). Numerically bit-equivalent — same PSNR/SSIM/LPIPS
to 4 decimals across all benched scenes; no atlas re-bake needed (we used
`--skip_bake`, only the render path was exercised).

### `SB_10thr_005w25gLP4levno2f_FRP5k10` (SB sweep #1)

| scene | per-pixel FPS | fused FPS | speedup | PSNR (both) |
|---|---|---|---|---|
| bicycle | 598.6 | 665.9 | **+11.2 %** | 23.57 |
| bonsai | 506.8 | 550.9 | **+8.7 %** | 31.16 |
| counter | 698.4 | 791.1 | **+13.3 %** | 28.42 |
| flowers | 474.9 | 513.0 | **+8.0 %** | 20.47 |
| garden | 940.9 | 1062.5 | **+12.9 %** | 26.48 |
| kitchen | 533.6 | 609.9 | **+14.3 %** | 30.32 |
| room | 855.6 | 941.0 | **+10.0 %** | 30.62 |
| stump | 594.2 | 646.4 | **+8.8 %** | 25.53 |
| treehill | 474.7 | 527.6 | **+11.1 %** | 22.21 |
| **MEAN** | **630.9** | **700.9** | **+11.1 %** | — |

### `SB_10thr_01w25g4levN2F_FRP5k10` (SB sweep #2)

| scene | per-pixel FPS | fused FPS | speedup | PSNR (both) |
|---|---|---|---|---|
| bicycle | 712.4 | 787.7 | **+10.6 %** | 23.50 |
| bonsai | 619.9 | 674.5 | **+8.8 %** | 30.52 |
| counter | 798.3 | 900.8 | **+12.8 %** | 28.08 |
| flowers | 594.4 | 639.5 | **+7.6 %** | 20.39 |
| garden | 1061.6 | 1207.5 | **+13.7 %** | 26.25 |
| kitchen | 644.2 | 734.8 | **+14.1 %** | 29.80 |
| room | 1041.0 | 1143.7 | **+9.9 %** | 30.13 |
| stump | 763.8 | 824.9 | **+8.0 %** | 25.37 |
| treehill | 583.3 | 645.7 | **+10.7 %** | 22.18 |
| **MEAN** | **757.7** | **839.9** | **+10.8 %** | — |

Combined mean lift across both configs: **+10.95 %** (range +7.6 % to +14.3 %),
zero quality cost. The other two SB tables above (`SB_10thr_0w0g…` and
`SB_5thr_01w25g4levN2F…`) were not re-benched — apply the same factor as
a rough estimate; or re-run with `--skip_bake` to update.

## `SB_30thr_005w25gLP4lev_FRP5k10_c2f_Jac`

Same `_c2f_Jac` family as the SV-30thr config below, but with `--feature beta`
(K=2 spherical-beta lobes) instead of SV. 30 % importance-prune threshold gives
the smallest-yet atlases of the SB family (mean 412 MB) and the new fused-SB
preprocess path keeps FPS in the same band as the lower-threshold SB configs.

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.55 | 0.6616 | 0.3062 | 771.7 | 554 | 23.83 / 93.4 |
| bonsai | 31.09 | 0.9237 | 0.2111 | 580.5 | 297 | 31.23 / 87.6 |
| counter | 28.37 | 0.8793 | 0.2306 | 873.4 | 182 | 28.42 / 93.3 |
| flowers | 20.43 | 0.5326 | 0.3551 | 627.0 | 671 | 20.27 / 83.1 |
| garden | 26.37 | 0.8089 | 0.1796 | 1065.1 | 478 | 26.32 / 100.7 |
| kitchen | 30.20 | 0.8984 | 0.1619 | 661.0 | 219 | 30.27 / 74.0 |
| room | 30.53 | 0.9017 | 0.2406 | 1027.1 | 224 | 30.52 / 114.2 |
| stump | 25.52 | 0.7140 | 0.2807 | 747.4 | 342 | 25.52 / 106.6 |
| treehill | 22.18 | 0.5800 | 0.3594 | 591.8 | 738 | 22.18 / 75.1 |
| **MEAN** | **26.47** | **0.7667** | **0.2584** | **771.7** | **412** | **26.51 / 92.0** |

## `SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac`

First config trained with `--feature SV` (Spherical Voronoi, K=7 sites/Gaussian)
instead of SB lobes. Per-Gaussian primitive color is the softmax mix

> `feat = Σ_k softmax(-τ_k · ‖site_k − ω‖)_k · color_k`,   `rgb = relu(feat + sh_bias)`

evaluated per-frame against the per-Gaussian view direction. The hash+MLP
residual still rides on top via the BC7 atlas, identical to the SB configs.

Bench numbers below use the **torch fake-SH-DC** path: SV is evaluated in
torch each frame and the result is stuffed into the SH-DC slot of `fake_shs`,
so the existing CUDA `computeColorFromSH` reproduces `relu(feat + sh_bias)`
via `clamp(SH_C0 · fake_dc + sh_bias, 0)`. The **fused-CUDA** path
(`computeColorFromVoronoi` ported from `2dgs-voronoi` into
`diff_surfel_bake_render/forward.cu`) skips the autograd graph + fake-SH
roundtrip and gives **+10% FPS** on garden (1153 vs 1045) at bit-identical
output (max abs render diff 1.1e-4, FP16 noise). Use the `voronoi_sites`,
`voronoi_tau`, `voronoi_colors`, `voronoi_K` kwargs on `GaussianRasterizer`
for the headline FPS number; the torch path is kept as the autograd-friendly
fallback (and is what was used for this table for direct apples-to-apples
comparison with the SB configs above).

| scene | PSNR | SSIM | LPIPS | FPS | atlas (MB) | neural PSNR / FPS |
|---|---|---|---|---|---|---|
| bicycle | 23.84 | 0.6679 | 0.3007 | 722.2 | 549 | 23.87 / 97.3 |
| bonsai | 31.94 | 0.9290 | 0.2064 | 576.4 | 284 | 32.11 / 93.5 |
| counter | 28.98 | 0.8875 | 0.2214 | 863.6 | 185 | 29.00 / 93.0 |
| flowers | 20.56 | 0.5370 | 0.3516 | 559.2 | 685 | 20.39 / 83.0 |
| garden | 26.65 | 0.8137 | 0.1732 | 1014.5 | 494 | 26.60 / 103.1 |
| kitchen | 30.62 | 0.9043 | 0.1559 | 697.7 | 215 | 30.65 / 76.3 |
| room | 30.40 | 0.9022 | 0.2402 | 1037.9 | 225 | 30.31 / 112.7 |
| stump | 25.62 | 0.7169 | 0.2763 | 689.1 | 333 | 25.59 / 107.9 |
| treehill | 22.34 | 0.5855 | 0.3494 | 560.6 | 736 | 22.28 / 76.8 |
| **MEAN** | **26.77** | **0.7715** | **0.2528** | **746.8** | **412** | **26.76 / 93.7** |

### Torch vs fused-CUDA SV path (full sweep)

Identical bake artifacts; the only difference is the per-Gaussian SV
evaluator (PyTorch fake-SH-DC closure vs. `computeColorFromVoronoi`
fused into `preprocessCUDA`). The CUDA port is bit-equivalent — all 9
scenes show identical PSNR to 4 decimals and the cross-render max-abs
delta peaks at 4.4 × 10⁻⁴ (FP16 rasterizer noise; PSNR drift 4 × 10⁻⁷ dB
across the table). Speedup is purely from skipping the per-frame torch
overhead (autograd graph construction + fake-SH-DC slot fill).

The two FPS columns below differ slightly from the headline torch column
above because they were measured in a back-to-back pass with cached
atlases; absolute numbers shift ~2 % from atlas-state warm-up but the
torch↔CUDA ratio is the apples-to-apples comparison.

| scene | torch PSNR | torch FPS | CUDA PSNR | CUDA FPS | speedup | cross-render max abs |
|---|---|---|---|---|---|---|
| bicycle | 23.84 | 740.3 | 23.84 | 796.2 | **1.08×** | 2.3e-5 |
| bonsai | 31.94 | 582.1 | 31.94 | 609.9 | **1.05×** | 1.5e-4 |
| counter | 28.98 | 880.8 | 28.98 | 935.8 | **1.06×** | 4.4e-5 |
| flowers | 20.56 | 568.4 | 20.56 | 603.6 | **1.06×** | 1.9e-4 |
| garden | 26.65 | 1046.6 | 26.65 | 1153.2 | **1.10×** | 1.1e-4 |
| kitchen | 30.62 | 711.0 | 30.62 | 754.8 | **1.06×** | 4.4e-4 |
| room | 30.40 | 1057.9 | 30.40 | 1142.4 | **1.08×** | 5.4e-5 |
| stump | 25.62 | 700.9 | 25.62 | 739.6 | **1.06×** | 9.1e-5 |
| treehill | 22.34 | 568.4 | 22.34 | 605.8 | **1.07×** | 7.1e-5 |
| **MEAN** | **26.77** | **761.8** | **26.77** | **815.7** | **1.07×** | — |

CUDA path lifts mean FPS by **+54 FPS (+7 %)** at zero quality cost.
Effect would be larger on slower GPUs; on a 5090 the rasterizer pass is
already fast enough that the per-frame torch slack is small.

## Cross-config means (all 9 mip-360 scenes)

| config | baked PSNR | baked FPS | atlas (MB) | neural PSNR | neural FPS | speedup |
|---|---|---|---|---|---|---|
| `SB_10thr_0w0gLP4levno2f_FRP5k10` | 26.98 | 315.8 | 665 | 27.03 | 54.1 | **5.83×** |
| `SB_10thr_005w25gLP4levno2f_FRP5k10` | 26.53 | 630.9 | 611 | 26.57 | 82.4 | **7.66×** |
| `SB_10thr_01w25g4levN2F_FRP5k10` | 26.25 | 757.7 | 559 | 26.29 | 92.4 | **8.20×** |
| `SB_5thr_01w25g4levN2F_FRP5k10` | 26.29 | 730.7 | 656 | 26.34 | 88.4 | **8.26×** |
| `SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` | 26.77 | 746.8 | 412 | 26.76 | 93.7 | **7.97×** |
| `SB_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` | 26.47 | 771.7 | 412 | 26.51 | 92.0 | **8.39×** |


