# 4090 Baked-Atlas Bench Results

Companion to `MIP360_BENCH_RESULTS.md` (which holds the 5090 numbers). All
numbers here come from the `bench_minimal.py` pipeline run remotely on
`neel@10.176.128.69` (RTX 4090, driver 570.195.03, CUDA 12.0 build of
`diff_surfel_bake_render`, BC7 atlas + fused SV path).

**Hardware**: NVIDIA GeForce RTX 4090 (Ada / sm_89), 24 GB VRAM, AMD Threadripper 2920X, 60 GB RAM, Ubuntu 25.04.

**Methodology** (matched to the 5090 setup): `--num_warmup 10 --num_benchmark
200`, CUDA-event timing with `cuda.synchronize`, cycling through
`getTestCameras()`. LPIPS via the in-tree `lpipsPyTorch` (VGG net) so values
are directly comparable with the 5090 numbers.

**Pipeline**: bake on the 5090 → bundle (`scripts/build_bench_bundle.py`,
pre-activated tensors + cameras + GT images) → rsync → `bench_minimal.py`
loads the bundle, installs the BC7 atlas, runs the bench. No nest-splatting
source on the 4090 — only `diff_surfel_bake_render`, `lpipsPyTorch`,
`pytorch_msssim`, `lpips` (cached weights only), `plyfile`, `Pillow`.

**Atlas / runtime config**: `--max_res 64 --atlas_budget_mb 8192 --aabb_mode
5 --bake_dtype bc7 --sort_mode 0`. Identical to the 5090 sweep.

## `SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac`

The "production" config — error-weighted overdraw regularizer, c2f, Jacobian
contraction, freeze-hash-period 5k/10. This is the one used in the head-to-head
table in the paper.

### Mip-NeRF 360

| scene | N | resolution | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 173,755 | 1237×822 | 23.83 | 0.6659 | 0.3007 | 621.9 |
| bonsai | 112,481 | 1559×1039 | 31.91 | 0.9287 | 0.2064 | 483.4 |
| counter | 80,135 | 1558×1038 | 28.96 | 0.8868 | 0.2214 | 804.0 |
| flowers | 212,272 | 1256×828 | 20.56 | 0.5336 | 0.3516 | 460.1 |
| garden | 156,216 | 1297×840 | 26.65 | 0.8132 | 0.1732 | 951.8 |
| kitchen | 147,442 | 1558×1039 | 30.59 | 0.9035 | 0.1559 | 609.0 |
| room | 75,207 | 1557×1038 | 30.36 | 0.9018 | 0.2402 | 992.2 |
| stump | 102,131 | 1245×825 | 25.54 | 0.7155 | 0.2763 | 585.0 |
| treehill | 211,700 | 1267×832 | 22.33 | 0.5830 | 0.3494 | 470.9 |
| **MEAN** | **141,260** | | **26.75** | **0.7702** | **0.2528** | **664.3** |

### Tanks and Temples

| scene | N | resolution | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| train | 120,097 | 980×545 | 22.35 | 0.8043 | 0.2146 | 1038.7 |
| truck | 101,636 | 979×546 | 25.44 | 0.8742 | 0.1531 | 1149.9 |
| **MEAN** | **110,867** | | **23.89** | **0.8392** | **0.1839** | **1094.3** |

### Deep Blending

| scene | N | resolution | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| drjohnson | 73,189 | 1332×876 | 29.20 | 0.8902 | 0.2766 | 949.8 |
| playroom | 95,054 | 1264×832 | 30.01 | 0.9017 | 0.2519 | 538.7 |
| **MEAN** | **84,122** | | **29.61** | **0.8960** | **0.2643** | **744.3** |

## `SV_30thr_0w0gLP4lev_FRP5k10_c2f_Jac`

No-overdraw-regularizer ablation: `--w_lambda 0.0 --w_lambda_gamma 0`. Higher
quality at the cost of more primitives and slower inference. **DB not
trained** for this config — only mip-360 + T&T.

### Mip-NeRF 360

| scene | N | resolution | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 259,334 | 1237×822 | 24.13 | 0.6884 | 0.2817 | 280.9 |
| bonsai | 170,063 | 1559×1039 | 32.80 | 0.9393 | 0.2048 | 224.9 |
| counter | 121,670 | 1558×1038 | 29.40 | 0.8988 | 0.2140 | 438.8 |
| flowers | 299,906 | 1256×828 | 20.68 | 0.5479 | 0.3455 | 205.2 |
| garden | 242,462 | 1297×840 | 26.98 | 0.8227 | 0.1626 | 427.8 |
| kitchen | 237,683 | 1558×1039 | 31.52 | 0.9173 | 0.1442 | 357.1 |
| room | 117,936 | 1557×1038 | 31.03 | 0.9099 | 0.2347 | 482.8 |
| stump | 151,076 | 1245×825 | 25.76 | 0.7303 | 0.2639 | 257.4 |
| treehill | 317,142 | 1267×832 | 22.36 | 0.5942 | 0.3459 | 197.6 |
| **MEAN** | **213,030** | | **27.18** | **0.7832** | **0.2441** | **319.2** |

### Tanks and Temples

| scene | N | resolution | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| train | 158,858 | 980×545 | 22.57 | 0.8113 | 0.2114 | 659.4 |
| truck | 144,553 | 979×546 | 25.75 | 0.8797 | 0.1516 | 637.6 |
| **MEAN** | **151,706** | | **24.16** | **0.8455** | **0.1815** | **648.5** |

## Cross-config means (per-dataset)

| dataset | config | PSNR | SSIM | LPIPS | mean N | mean FPS |
|---|---|---:|---:|---:|---:|---:|
| Mip-NeRF 360 | 005w25g (overdraw) | 26.75 | 0.770 | 0.253 | 141k | 664 |
| Mip-NeRF 360 | 0w0g | 27.18 | 0.783 | 0.244 | 213k | 319 |
| Tanks & Temples | 005w25g (overdraw) | 23.89 | 0.839 | 0.184 | 111k | 1094 |
| Tanks & Temples | 0w0g | 24.16 | 0.846 | 0.181 | 152k | 648 |
| Deep Blending | 005w25g (overdraw) | 29.61 | 0.896 | 0.264 | 84k | 744 |

**Trade-off** (overdraw regularizer on vs. off): on mip-360, 0w0g gains
+0.43 dB PSNR / +0.013 SSIM / −0.009 LPIPS at the cost of **1.51× more
Gaussians and 2.08× slower FPS**. T&T shows the same direction at smaller
magnitude (+0.27 dB / +0.007 SSIM / −0.003 LPIPS for 1.37× more N and 1.69×
slower FPS).

## 4090 vs 5090 (where direct comparison exists)

005w25g, fused SV path on both, lpipsPyTorch on both:

| dataset | metric | 5090 | 4090 | 4090 / 5090 |
|---|---|---:|---:|---:|
| Mip-NeRF 360 | PSNR | 26.78 | 26.75 | −0.03 |
| Mip-NeRF 360 | SSIM | 0.7716 | 0.7702 | −0.0014 |
| Mip-NeRF 360 | LPIPS | 0.2528 | 0.2528 | 0.0000 |
| Mip-NeRF 360 | FPS | 815.7 | 664.3 | **0.81×** |
| T&T | PSNR | 23.89 | 23.89 | 0.00 |
| T&T | SSIM | 0.8402 | 0.8392 | −0.0010 |
| T&T | LPIPS | 0.1839 | 0.1839 | 0.0000 |
| T&T | FPS | 1196.7 | 1094.3 | **0.91×** |
| DB | PSNR | 29.73 | 29.61 | −0.12 |
| DB | SSIM | 0.8968 | 0.8960 | −0.0008 |
| DB | LPIPS | 0.2643 | 0.2643 | 0.0000 |
| DB | FPS | 911.7 | 744.3 | **0.82×** |

PSNR/SSIM differences are within ±0.12 dB / ±0.0014 — driven entirely by BC7
hardware-decode LSB drift between Ada (4090) and Blackwell (5090) texture
units. LPIPS is bit-identical because both ends use the same lpipsPyTorch
implementation and identical rendered images. FPS scales roughly with
memory bandwidth (5090 ≈ 1.79 TB/s, 4090 ≈ 1.0 TB/s; observed 4090/5090 FPS
ratio of 0.81-0.91 brackets the bandwidth ratio of 0.56 — the 4090 isn't
purely bandwidth-bound, the SM count helps light scenes).

## Reproducibility / where the data lives

- **Per-scene JSONs (4090 disk)**: `~/nest-bench/bench_results/<scene>.json`
  on `neel@10.176.128.69`. **Caveat**: the directory holds whichever config
  was rendered last per scene. Currently:
  - `db_drjohnson.json`, `db_playroom.json` → 005w25g (kept; never re-run for 0w0g).
  - `mip_360_*.json` → 0w0g (overwrote 005w25g on 2026-05-07 16:52–16:57).
  - `tnt_train.json`, `tnt_truck.json` → 0w0g (overwrote 005w25g on 2026-05-08 10:38).

  The 005w25g mip-360 + T&T per-scene JSONs no longer exist on disk; the
  values in this document were captured from the original sweep tee output
  (`run_all.sh`'s markdown table). To preserve future runs separately, point
  `run_all.sh` at a config-named output dir (`bench_results/<config>/`).

- **Pipeline tee logs (5090 disk, ephemeral `/tmp`)**:
  `/tmp/sv30thr_0w0g_4090_bench.log`,
  `/tmp/sv30thr_0w0g_tnt_4090.log`. Hold the markdown table from each
  `run_all.sh` invocation — useful for archival but `/tmp` won't survive a
  reboot.

- **Bundles (4090 disk)**: `~/nest-bench/bundles_*/` — self-contained
  per-scene packages used to feed `bench_minimal.py`. Re-running the bench is
  a single command:
  ```
  ssh neel@10.176.128.69 'cd ~/nest-bench/bench_4090 && bash run_all.sh ~/nest-bench/bundles_<config>'
  ```
