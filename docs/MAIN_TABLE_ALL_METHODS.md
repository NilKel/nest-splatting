# Main table (draft) — all rows self-run

FPS, #primitives and resolution measured on one **RTX 5090**, idle, test cameras, 50 warmup + 400 timed frames, `torch.cuda.Event`, sort and preprocess included. Resolution: mip-360 outdoor `images_4`, indoor `images_2`; T&T and Deep Blending native (already below the 1600 px cap).

Quality for the sweep-timed baselines is recomputed from saved renders by one shared eval script; `--` marks cells not yet folded in.

*BITYMI = ours (`--method 3D_SH_res`). NeST = the codebase; as a row it is `--method baseline`.*

**Two rows added 2026-08-19.**

*FastGS (Big)* is FastGS's own `train_big.sh` recipe — `--densification_interval 100`
(vs 500 in base) plus per-scene tightened `--grad_abs_thresh` — i.e. their
higher-quality/slower configuration. Trained here on all 13 scenes and benchmarked
with FastGS's own `bench_fps.py` (50 warm-up + 400 timed frames, per-frame
`cuda.Event`) at `mult 0.5`, their released default. As with the base row, mip-360 is
re-rendered at `images_4`/`images_2` so both methods are timed on identical images;
**primitive counts therefore reflect densification at FastGS's training resolution**
(`-i images`, long edge capped at 1600 px). FPS at `mult 1.0` is 8–14% lower across
the board and is recorded per-scene in
`speed_comparison/fastgs_big_2026-08/`. Size is `--` because these checkpoints were
not packaged for deployment.

*BITYMI (Gaussian kernel)* is our own pipeline trained with `--kernel gaussian`
instead of the production `beta_scaled`, otherwise identical flags, then baked and
atlas-finetuned the same way. It is an **ablation, not the shipped configuration**:
it gains ~+0.5 dB PSNR and ~0.012 LPIPS but runs at roughly **half** the frame rate,
because a Gaussian has no compact support — α never reaches zero, so AccuTile emits
looser ellipses and each surfel spawns far more fragments. `beta_scaled` remains the
default for exactly this reason.

## Mip-NeRF 360

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS | scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3DGS | 27.54 | 0.8190 | 0.2149 | 2,732,349 | 646 | **211.3** | 9 |
| 2DGS | 26.83 | 0.7989 | 0.2520 | 2,128,093 | 495 | **130.7** | 9 |
| Beta-Splatting | 28.07 | 0.8311 | 0.1904 | 3,111,111 | 356 | **109.9** | 9 |
| Beta-Splatting (200K) | 26.66 | 0.7725 | 0.2942 | 205,331 | 23 | **468.5** | 9 |
| BBSplat | 26.80 | 0.7874 | 0.2355 | 237,778 | 176 | **48.2** | 9 |
| Nexels (40K) | 25.99 | 0.7619 | 0.2411 | 39,977 | 138 | **118.5** | 9 |
| Nexels (100K) | 26.55 | 0.7764 | 0.2245 | 99,934 | 282 | **100.0** | 9 |
| Nexels (400K) | 27.21 | 0.8018 | 0.2057 | 399,771 | 230 | **78.1** | 9 |
| Speedy-Splat | 26.91 | 0.7883 | 0.2875 | 315,961 | 75 | **1258.5** | 9 |
| FastGS | 27.44 | 0.7967 | 0.2589 | 394,189 | 98 | **1208.6** | 9 |
| FastGS (Big) | 27.80 | 0.8191 | 0.2160 | 1,161,438 | -- | **889.7** | 9 |
| Content-Aware Texturing | 26.84 | 0.7886 | 0.2340 | 155,756 | 436 | **136.6** | 9 |
| Textured Gaussians | 25.86 | 0.7336 | 0.2857 | 100,000 | 3,837 | **28.3** | 9 |
| NeST (--method baseline) | 26.39 | 0.7734 | 0.2257 | 961,163 | 224 | **23.7** | 9 |
| BITYMI (ours) | 27.23 | 0.7892 | 0.2114 | 123,051 | 445 | **1538.1** | 9 |
| BITYMI (Gaussian kernel) | 27.88 | 0.8070 | 0.1981 | 248,341 | 839 | **705.5** | 9 |

## Tanks & Temples

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS | scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3DGS | 23.74 | 0.8574 | 0.1692 | 1,574,592 | 372 | **259.0** | 2 |
| 2DGS | 23.15 | 0.8367 | 0.2118 | 851,362 | 198 | **230.2** | 2 |
| Beta-Splatting | 24.71 | 0.8735 | 0.1434 | 1,750,000 | 200 | **189.7** | 2 |
| Beta-Splatting (200K) | 23.38 | 0.8348 | 0.2144 | 200,000 | 23 | **523.5** | 2 |
| BBSplat | 23.70 | 0.8570 | 0.1501 | 300,000 | 226 | **80.0** | 2 |
| Nexels (40K) | 22.25 | 0.7942 | 0.2113 | 39,979 | 138 | **204.0** | 2 |
| Nexels (100K) | 22.92 | 0.8244 | 0.1752 | 99,932 | 282 | **180.6** | 2 |
| Nexels (400K) | 23.62 | 0.8449 | 0.1576 | 399,767 | 230 | **136.9** | 2 |
| Speedy-Splat | 23.44 | 0.8252 | 0.2399 | 181,817 | 43 | **1458.9** | 2 |
| FastGS | 24.00 | 0.8404 | 0.2093 | 242,133 | 60 | **1243.4** | 2 |
| FastGS (Big) | 24.36 | 0.8572 | 0.1758 | 543,997 | -- | **959.3** | 2 |
| Content-Aware Texturing | 23.37 | 0.8387 | 0.1973 | 133,880 | 127 | **269.5** | 2 |
| Textured Gaussians | 22.70 | 0.8058 | 0.2161 | 100,000 | 3,837 | **29.7** | 2 |
| NeST (--method baseline) | 22.54 | 0.8178 | 0.1866 | 378,448 | 90 | **48.4** | 2 |
| BITYMI (ours) | 24.10 | 0.8507 | 0.1447 | 86,746 | 325 | **2217.2** | 2 |
| BITYMI (Gaussian kernel) | 24.50 | 0.8598 | 0.1392 | 149,681 | 539 | **1161.4** | 2 |

## Deep Blending

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS | scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3DGS | 29.71 | 0.9106 | 0.2375 | 2,471,124 | 584 | **215.6** | 2 |
| 2DGS | 29.47 | 0.9071 | 0.2566 | 1,508,720 | 351 | **156.5** | 2 |
| Beta-Splatting | 29.44 | 0.9090 | 0.2372 | 3,000,000 | 343 | **136.2** | 2 |
| Beta-Splatting (200K) | 29.64 | 0.9024 | 0.2756 | 200,000 | 23 | **519.6** | 2 |
| BBSplat | 29.34 | 0.9050 | 0.2570 | 160,000 | 111 | **49.4** | 2 |
| Nexels (40K) | 29.34 | 0.8981 | 0.2306 | 39,958 | 138 | **105.7** | 2 |
| Nexels (100K) | 30.00 | 0.9075 | 0.2097 | 99,843 | 282 | **85.9** | 2 |
| Nexels (400K) | 30.41 | 0.9138 | 0.2046 | 399,188 | 230 | **56.3** | 2 |
| Speedy-Splat | 29.62 | 0.9074 | 0.2678 | 250,990 | 59 | **1501.0** | 2 |
| FastGS | 30.02 | 0.9045 | 0.2651 | 216,748 | 54 | **1450.7** | 2 |
| FastGS (Big) | 30.31 | 0.9115 | 0.2437 | 649,603 | -- | **1275.8** | 2 |
| Content-Aware Texturing | 29.99 | 0.9141 | 0.2410 | 185,094 | 252 | **146.9** | 2 |
| Textured Gaussians | 29.04 | 0.8913 | 0.2652 | 100,000 | 3,837 | **30.0** | 2 |
| NeST (--method baseline) | 28.86 | 0.9021 | 0.2273 | 478,000 | 113 | **37.7** | 2 |
| BITYMI (ours) | 29.87 | 0.8902 | 0.2176 | 66,622 | 259 | **2013.9** | 2 |
| BITYMI (Gaussian kernel) | 30.28 | 0.9000 | 0.2095 | 133,095 | 498 | **961.0** | 2 |

**Size** is the payload as each method stores it, so the column mixes compressed and uncompressed formats and is not a like-for-like codec comparison. BITYMI ships a BC7-compressed atlas (its uncompressed 1.6 GB `atlas_texture.pt` intermediate is excluded, as it is not deployed); Content-Aware Texturing and Textured Gaussians store fp32 textures with no compressed variant in their pipelines — Textured Gaussians' 3.8 GB is `textures[N,50,50,4]` in fp32, genuine model weight, not optimizer state.

‡ quality averaged over only the scenes shown in `q n/N` — **not** a dataset mean, and not comparable to the complete rows. FPS and #prims still cover all N.

## Coverage

| method | mip360 | T&T | DB | quality |
|---|---:|---:|---:|---|
| 3DGS | 9/9 | 2/2 | 2/2 | yes |
| 2DGS | 9/9 | 2/2 | 2/2 | yes |
| Beta-Splatting | 9/9 | 2/2 | 2/2 | yes |
| Beta-Splatting (200K) | 9/9 | 2/2 | 2/2 | yes |
| BBSplat | 9/9 | 2/2 | 2/2 | yes |
| Nexels (40K) | 9/9 | 2/2 | 2/2 | yes |
| Nexels (100K) | 9/9 | 2/2 | 2/2 | yes |
| Nexels (400K) | 9/9 | 2/2 | 2/2 | yes |
| Speedy-Splat | 9/9 | 2/2 | 2/2 | yes |
| FastGS | 9/9 | 2/2 | 2/2 | yes |
| FastGS (Big) | 9/9 | 2/2 | 2/2 | yes |
| Content-Aware Texturing | 9/9 | 2/2 | 2/2 | yes |
| GStex | 0/9 | 0/2 | 0/2 | not run |
| Textured Gaussians | 9/9 | 2/2 | 2/2 | yes |
| NeST (--method baseline) | 9/9 | 2/2 | 2/2 | yes |
| BITYMI (ours) | 9/9 | 2/2 | 2/2 | yes |
| BITYMI (Gaussian kernel) | 9/9 | 2/2 | 2/2 | yes |

Folded 858 cluster-computed quality/size values.

---

# Per-scene results

Every method, every scene. `--` means not measured (not a zero). FPS is RTX 5090; quality is the shared eval; Size is the full render payload, not just the PLY.

## Mip-NeRF 360

### mip_360/bicycle

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 25.26 | 0.7681 | 0.2081 | 4,868,679 | 1,152 | 111.0 |
| 2DGS | 24.72 | 0.7333 | 0.2695 | 4,572,264 | 1,064 | 91.2 |
| Beta-Splatting | 25.25 | 0.7840 | 0.1747 | 6,000,000 | 687 | 58.4 |
| Beta-Splatting (200K) | 23.93 | 0.6761 | 0.3477 | 200,000 | 23 | 453.6 |
| BBSplat | 23.97 | 0.6886 | 0.2818 | 300,000 | 215 | 31.1 |
| Nexels (40K) | 23.45 | 0.6580 | 0.2883 | 39,973 | 138 | 140.7 |
| Nexels (100K) | 23.88 | 0.6808 | 0.2670 | 99,926 | 282 | 117.0 |
| Nexels (400K) | 24.74 | 0.7324 | 0.2274 | 399,824 | 230 | 90.7 |
| Speedy-Splat | 24.86 | 0.7246 | 0.3039 | 611,126 | 145 | 908.4 |
| FastGS | 24.61 | 0.7237 | 0.2870 | 539,624 | 134 | 1211.8 |
| FastGS (Big) | 24.99 | 0.7655 | 0.2198 | 1,564,886 | -- | 836.6 |
| Content-Aware Texturing | 23.84 | 0.7070 | 0.2523 | 105,286 | 703 | 156.5 |
| Textured Gaussians | 22.75 | 0.6005 | 0.3448 | 100,000 | 3,837 | 30.0 |
| NeST (--method baseline) | 24.36 | 0.7284 | 0.2364 | 2,093,136 | 487 | 14.7 |
| BITYMI (ours) | 24.41 | 0.7132 | 0.2383 | 160,097 | 620 | 1519.7 |
| BITYMI (Gaussian kernel) | 24.77 | 0.7363 | 0.2163 | 338,883 | 1,162 | 646.6 |

### mip_360/flowers

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 21.53 | 0.6082 | 0.3356 | 2,912,795 | 689 | 223.7 |
| 2DGS | 21.17 | 0.5790 | 0.3724 | 2,255,217 | 525 | 141.4 |
| Beta-Splatting | 21.75 | 0.6353 | 0.2985 | 3,000,000 | 343 | 101.7 |
| Beta-Splatting (200K) | 20.61 | 0.5314 | 0.4360 | 200,000 | 23 | 488.6 |
| BBSplat | 20.44 | 0.5331 | 0.3892 | 300,000 | 209 | 27.1 |
| Nexels (40K) | 19.81 | 0.4962 | 0.3646 | 39,977 | 138 | 137.8 |
| Nexels (100K) | 20.26 | 0.5280 | 0.3327 | 99,950 | 282 | 119.2 |
| Nexels (400K) | 21.17 | 0.5802 | 0.2972 | 399,894 | 230 | 93.3 |
| Speedy-Splat | 21.42 | 0.5837 | 0.3938 | 363,763 | 86 | 1196.5 |
| FastGS | 21.29 | 0.5716 | 0.3880 | 489,951 | 122 | 1166.3 |
| FastGS (Big) | 21.63 | 0.6136 | 0.3225 | 1,132,320 | -- | 872.9 |
| Content-Aware Texturing | 20.07 | 0.5130 | 0.3783 | 100,094 | 593 | 155.7 |
| Textured Gaussians | 19.66 | 0.4734 | 0.4125 | 100,000 | 3,837 | 28.3 |
| NeST (--method baseline) | 19.42 | 0.5104 | 0.3548 | 1,441,901 | 336 | 15.9 |
| BITYMI (ours) | 20.78 | 0.5581 | 0.3202 | 184,511 | 728 | 1300.9 |
| BITYMI (Gaussian kernel) | 21.17 | 0.5894 | 0.2996 | 378,264 | 1,368 | 472.0 |

### mip_360/garden

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 27.46 | 0.8691 | 0.1064 | 4,144,122 | 980 | 145.3 |
| 2DGS | 26.70 | 0.8441 | 0.1466 | 3,002,184 | 699 | 106.5 |
| Beta-Splatting | 27.63 | 0.8708 | 0.1020 | 5,000,000 | 572 | 79.1 |
| Beta-Splatting (200K) | 25.54 | 0.7716 | 0.2714 | 200,000 | 23 | 569.2 |
| BBSplat | 26.87 | 0.8389 | 0.1367 | 300,000 | 245 | 57.1 |
| Nexels (40K) | 25.63 | 0.7971 | 0.1631 | 39,986 | 138 | 143.0 |
| Nexels (100K) | 26.02 | 0.8076 | 0.1534 | 99,959 | 282 | 120.6 |
| Nexels (400K) | 26.96 | 0.8430 | 0.1236 | 399,900 | 230 | 96.4 |
| Speedy-Splat | 26.92 | 0.8354 | 0.1826 | 534,500 | 126 | 1006.0 |
| FastGS | 27.32 | 0.8475 | 0.1574 | 660,711 | 164 | 1146.1 |
| FastGS (Big) | 27.23 | 0.8669 | 0.1061 | 2,628,735 | -- | 584.5 |
| Content-Aware Texturing | 26.45 | 0.8306 | 0.1454 | 117,044 | 456 | 220.6 |
| Textured Gaussians | 24.81 | 0.7360 | 0.2357 | 100,000 | 3,837 | 30.0 |
| NeST (--method baseline) | 26.59 | 0.8430 | 0.1291 | 1,518,233 | 353 | 17.2 |
| BITYMI (ours) | 27.02 | 0.8373 | 0.1288 | 148,658 | 576 | 1954.3 |
| BITYMI (Gaussian kernel) | 27.49 | 0.8522 | 0.1159 | 294,102 | 1,073 | 882.7 |

### mip_360/stump

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 26.67 | 0.7738 | 0.2139 | 4,276,742 | 1,012 | 166.9 |
| 2DGS | 26.17 | 0.7572 | 0.2590 | 2,963,468 | 690 | 98.9 |
| Beta-Splatting | 26.82 | 0.7964 | 0.1845 | 4,500,000 | 515 | 71.2 |
| Beta-Splatting (200K) | 25.15 | 0.7056 | 0.3288 | 200,000 | 23 | 490.1 |
| BBSplat | 24.39 | 0.6869 | 0.3226 | 300,000 | 206 | 26.1 |
| Nexels (40K) | 25.10 | 0.6913 | 0.2828 | 39,984 | 138 | 141.8 |
| Nexels (100K) | 25.52 | 0.7138 | 0.2682 | 99,970 | 282 | 120.7 |
| Nexels (400K) | 26.58 | 0.7650 | 0.2165 | 399,931 | 230 | 93.0 |
| Speedy-Splat | 26.57 | 0.7705 | 0.2595 | 502,928 | 119 | 1061.9 |
| FastGS | 26.47 | 0.7528 | 0.2721 | 392,348 | 97 | 1262.6 |
| FastGS (Big) | 27.01 | 0.7854 | 0.2161 | 1,050,669 | -- | 970.5 |
| Content-Aware Texturing | 25.55 | 0.7330 | 0.2576 | 126,781 | 877 | 136.9 |
| Textured Gaussians | 23.73 | 0.6252 | 0.3586 | 100,000 | 3,837 | 29.2 |
| NeST (--method baseline) | 24.52 | 0.6765 | 0.2886 | 797,235 | 186 | 21.1 |
| BITYMI (ours) | 25.66 | 0.7225 | 0.2632 | 96,365 | 378 | 1433.3 |
| BITYMI (Gaussian kernel) | 26.25 | 0.7518 | 0.2336 | 178,732 | 641 | 612.8 |

### mip_360/treehill

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 22.60 | 0.6438 | 0.3252 | 3,304,087 | 781 | 184.0 |
| 2DGS | 22.30 | 0.6202 | 0.3745 | 3,227,871 | 751 | 107.7 |
| Beta-Splatting | 22.65 | 0.6464 | 0.2855 | 3,500,000 | 401 | 98.8 |
| Beta-Splatting (200K) | 22.56 | 0.6026 | 0.4302 | 200,000 | 23 | 473.6 |
| BBSplat | 22.36 | 0.6127 | 0.3519 | 300,000 | 212 | 30.1 |
| Nexels (40K) | 22.45 | 0.6077 | 0.3096 | 39,967 | 138 | 128.3 |
| Nexels (100K) | 22.62 | 0.6033 | 0.3029 | 99,937 | 282 | 106.2 |
| Nexels (400K) | 22.93 | 0.6305 | 0.2852 | 399,838 | 230 | 78.2 |
| Speedy-Splat | 22.51 | 0.5947 | 0.4446 | 366,009 | 87 | 1369.4 |
| FastGS | 22.83 | 0.6057 | 0.4141 | 394,335 | 98 | 1266.5 |
| FastGS (Big) | 22.85 | 0.6273 | 0.3626 | 1,002,068 | -- | 997.4 |
| Content-Aware Texturing | 22.82 | 0.6341 | 0.3003 | 102,248 | 625 | 157.0 |
| Textured Gaussians | 22.20 | 0.5605 | 0.3955 | 100,000 | 3,837 | 28.4 |
| NeST (--method baseline) | 21.09 | 0.5652 | 0.3240 | 1,416,800 | 330 | 16.1 |
| BITYMI (ours) | 22.55 | 0.6015 | 0.2939 | 174,744 | 727 | 1319.8 |
| BITYMI (Gaussian kernel) | 22.53 | 0.6146 | 0.2915 | 367,958 | 1,389 | 556.3 |

### mip_360/bonsai

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 32.28 | 0.9444 | 0.2029 | 1,078,318 | 255 | 357.1 |
| 2DGS | 31.29 | 0.9328 | 0.2264 | 800,426 | 186 | 131.6 |
| Beta-Splatting | 33.65 | 0.9541 | 0.1813 | 1,500,000 | 172 | 154.3 |
| Beta-Splatting (200K) | 31.60 | 0.9348 | 0.2243 | 206,613 | 24 | 477.8 |
| BBSplat | 31.98 | 0.9486 | 0.1683 | 160,000 | 124 | 69.0 |
| Nexels (40K) | 29.91 | 0.9150 | 0.1995 | 39,967 | 138 | 85.9 |
| Nexels (100K) | 30.88 | 0.9257 | 0.1836 | 99,926 | 282 | 70.5 |
| Nexels (400K) | 30.73 | 0.9121 | 0.2163 | 399,721 | 230 | 49.9 |
| Speedy-Splat | 31.03 | 0.9214 | 0.2506 | 131,005 | 31 | 1478.3 |
| FastGS | 32.05 | 0.9368 | 0.2138 | 275,719 | 68 | 1229.3 |
| FastGS (Big) | 32.81 | 0.9470 | 0.1882 | 847,329 | -- | 897.6 |
| Content-Aware Texturing | 31.37 | 0.9353 | 0.2093 | 231,923 | 145 | 83.5 |
| Textured Gaussians | 30.84 | 0.9228 | 0.2186 | 100,000 | 3,837 | 27.5 |
| NeST (--method baseline) | 31.91 | 0.9268 | 0.1754 | 336,335 | 78 | 31.1 |
| BITYMI (ours) | 32.53 | 0.9364 | 0.1669 | 92,281 | 286 | 1330.0 |
| BITYMI (Gaussian kernel) | 33.73 | 0.9471 | 0.1632 | 173,710 | 527 | 659.7 |

### mip_360/counter

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 29.08 | 0.9108 | 0.1989 | 1,079,163 | 255 | 257.9 |
| 2DGS | 28.11 | 0.8943 | 0.2300 | 645,085 | 150 | 181.4 |
| Beta-Splatting | 30.14 | 0.9238 | 0.1734 | 1,500,000 | 172 | 128.7 |
| Beta-Splatting (200K) | 28.89 | 0.9011 | 0.2192 | 200,000 | 23 | 417.6 |
| BBSplat | 28.74 | 0.9138 | 0.1743 | 160,000 | 124 | 65.7 |
| Nexels (40K) | 27.57 | 0.8815 | 0.2059 | 39,993 | 138 | 95.6 |
| Nexels (100K) | 28.20 | 0.8948 | 0.1874 | 99,954 | 282 | 81.7 |
| Nexels (400K) | 28.90 | 0.9062 | 0.1759 | 399,853 | 230 | 66.1 |
| Speedy-Splat | 28.18 | 0.8717 | 0.2730 | 98,696 | 23 | 1413.2 |
| FastGS | 29.06 | 0.8992 | 0.2199 | 208,171 | 52 | 1198.9 |
| FastGS (Big) | 29.46 | 0.9105 | 0.1958 | 470,685 | -- | 1030.8 |
| Content-Aware Texturing | 28.72 | 0.9020 | 0.2088 | 197,383 | 167 | 95.7 |
| Textured Gaussians | 28.03 | 0.8797 | 0.2177 | 100,000 | 3,837 | 26.2 |
| NeST (--method baseline) | 28.32 | 0.8861 | 0.2009 | 281,078 | 65 | 30.9 |
| BITYMI (ours) | 29.29 | 0.8994 | 0.1819 | 68,394 | 208 | 1594.2 |
| BITYMI (Gaussian kernel) | 30.01 | 0.9128 | 0.1709 | 128,313 | 387 | 830.4 |

### mip_360/kitchen

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 31.40 | 0.9297 | 0.1255 | 1,613,698 | 382 | 211.0 |
| 2DGS | 30.34 | 0.9186 | 0.1464 | 825,141 | 192 | 153.0 |
| Beta-Splatting | 32.23 | 0.9338 | 0.1219 | 1,500,000 | 172 | 141.0 |
| Beta-Splatting (200K) | 30.17 | 0.9125 | 0.1582 | 241,367 | 28 | 411.7 |
| BBSplat | 31.17 | 0.9318 | 0.1175 | 160,000 | 130 | 73.5 |
| Nexels (40K) | 29.63 | 0.9030 | 0.1469 | 39,985 | 138 | 98.5 |
| Nexels (100K) | 30.51 | 0.9154 | 0.1311 | 99,954 | 282 | 85.7 |
| Nexels (400K) | 31.18 | 0.9228 | 0.1229 | 399,886 | 230 | 73.8 |
| Speedy-Splat | 30.03 | 0.8936 | 0.2011 | 116,506 | 28 | 1413.5 |
| FastGS | 31.64 | 0.9218 | 0.1375 | 379,671 | 94 | 1055.2 |
| FastGS (Big) | 32.27 | 0.9328 | 0.1173 | 1,180,094 | -- | 700.3 |
| Content-Aware Texturing | 30.94 | 0.9187 | 0.1435 | 246,228 | 156 | 89.6 |
| Textured Gaussians | 29.85 | 0.8977 | 0.1654 | 100,000 | 3,837 | 26.9 |
| NeST (--method baseline) | 30.29 | 0.9159 | 0.1257 | 425,441 | 99 | 33.2 |
| BITYMI (ours) | 31.39 | 0.9181 | 0.1224 | 119,462 | 253 | 1384.1 |
| BITYMI (Gaussian kernel) | 32.46 | 0.9305 | 0.1110 | 242,266 | 521 | 736.2 |

### mip_360/room

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 31.59 | 0.9232 | 0.2176 | 1,313,536 | 311 | 244.5 |
| 2DGS | 30.68 | 0.9104 | 0.2432 | 861,181 | 200 | 164.7 |
| Beta-Splatting | 32.53 | 0.9354 | 0.1916 | 1,500,000 | 172 | 155.6 |
| Beta-Splatting (200K) | 31.53 | 0.9169 | 0.2317 | 200,000 | 23 | 434.0 |
| BBSplat | 31.31 | 0.9323 | 0.1776 | 160,000 | 118 | 54.4 |
| Nexels (40K) | 30.37 | 0.9073 | 0.2093 | 39,958 | 138 | 94.5 |
| Nexels (100K) | 31.07 | 0.9178 | 0.1943 | 99,833 | 282 | 78.6 |
| Nexels (400K) | 31.67 | 0.9244 | 0.1860 | 399,094 | 230 | 61.5 |
| Speedy-Splat | 30.65 | 0.8988 | 0.2779 | 119,120 | 28 | 1479.1 |
| FastGS | 31.70 | 0.9116 | 0.2405 | 207,167 | 51 | 1340.5 |
| FastGS (Big) | 31.91 | 0.9226 | 0.2159 | 576,163 | -- | 1116.5 |
| Content-Aware Texturing | 31.76 | 0.9239 | 0.2104 | 174,817 | 204 | 133.4 |
| Textured Gaussians | 30.86 | 0.9063 | 0.2228 | 100,000 | 3,837 | 27.8 |
| NeST (--method baseline) | 30.98 | 0.9080 | 0.1963 | 340,305 | 79 | 33.4 |
| BITYMI (ours) | 31.48 | 0.9164 | 0.1866 | 62,947 | 228 | 2006.7 |
| BITYMI (Gaussian kernel) | 32.54 | 0.9280 | 0.1807 | 132,841 | 485 | 952.7 |

## Tanks & Temples

### tnt/train

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 22.03 | 0.8254 | 0.1962 | 1,092,362 | 258 | 273.4 |
| 2DGS | 21.20 | 0.7954 | 0.2512 | 558,094 | 130 | 240.7 |
| Beta-Splatting | 22.96 | 0.8440 | 0.1796 | 1,000,000 | 114 | 253.9 |
| Beta-Splatting (200K) | 21.73 | 0.8024 | 0.2459 | 200,000 | 23 | 534.3 |
| BBSplat | 22.05 | 0.8297 | 0.1758 | 300,000 | 227 | 86.2 |
| Nexels (40K) | 20.71 | 0.7531 | 0.2579 | 39,978 | 138 | 204.8 |
| Nexels (100K) | 21.11 | 0.7849 | 0.2201 | 99,921 | 282 | 180.6 |
| Nexels (400K) | 21.70 | 0.8074 | 0.2014 | 399,745 | 230 | 138.7 |
| Speedy-Splat | 21.68 | 0.7773 | 0.2903 | 106,825 | 25 | 1508.2 |
| FastGS | 22.31 | 0.8054 | 0.2419 | 231,398 | 57 | 1138.9 |
| FastGS (Big) | 22.68 | 0.8255 | 0.2109 | 461,408 | -- | 928.0 |
| Content-Aware Texturing | 21.33 | 0.7976 | 0.2449 | 146,926 | 106 | 297.4 |
| Textured Gaussians | 20.57 | 0.7553 | 0.2564 | 100,000 | 3,837 | 29.6 |
| NeST (--method baseline) | 20.54 | 0.7730 | 0.2302 | 258,456 | 61 | 52.8 |
| BITYMI (ours) | 22.59 | 0.8233 | 0.1724 | 88,080 | 324 | 2035.2 |
| BITYMI (Gaussian kernel) | 22.95 | 0.8330 | 0.1689 | 145,773 | 507 | 1113.5 |

### tnt/truck

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 25.45 | 0.8893 | 0.1421 | 2,056,823 | 486 | 244.5 |
| 2DGS | 25.10 | 0.8781 | 0.1724 | 1,144,630 | 266 | 219.7 |
| Beta-Splatting | 26.46 | 0.9030 | 0.1073 | 2,500,000 | 286 | 125.5 |
| Beta-Splatting (200K) | 25.02 | 0.8671 | 0.1828 | 200,000 | 23 | 512.6 |
| BBSplat | 25.36 | 0.8843 | 0.1244 | 300,000 | 224 | 73.8 |
| Nexels (40K) | 23.79 | 0.8354 | 0.1647 | 39,980 | 138 | 203.2 |
| Nexels (100K) | 24.74 | 0.8640 | 0.1304 | 99,943 | 282 | 180.6 |
| Nexels (400K) | 25.54 | 0.8825 | 0.1138 | 399,789 | 230 | 135.2 |
| Speedy-Splat | 25.19 | 0.8732 | 0.1896 | 256,809 | 61 | 1409.6 |
| FastGS | 25.70 | 0.8755 | 0.1766 | 252,868 | 63 | 1347.9 |
| FastGS (Big) | 26.05 | 0.8888 | 0.1407 | 626,587 | -- | 990.6 |
| Content-Aware Texturing | 25.41 | 0.8799 | 0.1496 | 120,834 | 148 | 241.5 |
| Textured Gaussians | 24.82 | 0.8563 | 0.1758 | 100,000 | 3,837 | 29.8 |
| NeST (--method baseline) | 24.55 | 0.8626 | 0.1430 | 498,441 | 118 | 44.0 |
| BITYMI (ours) | 25.61 | 0.8781 | 0.1169 | 85,413 | 326 | 2399.2 |
| BITYMI (Gaussian kernel) | 26.05 | 0.8866 | 0.1095 | 153,590 | 571 | 1209.2 |

## Deep Blending

### db/drjohnson

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 29.49 | 0.9074 | 0.2357 | 3,102,137 | 734 | 164.6 |
| 2DGS | 28.87 | 0.9027 | 0.2562 | 1,809,771 | 421 | 154.1 |
| Beta-Splatting | 28.65 | 0.8989 | 0.2438 | 3,500,000 | 401 | 113.9 |
| Beta-Splatting (200K) | 29.07 | 0.8958 | 0.2796 | 200,000 | 23 | 516.1 |
| BBSplat | 28.83 | 0.8983 | 0.2638 | 160,000 | 119 | 65.8 |
| Nexels (40K) | 29.11 | 0.8991 | 0.2345 | 39,949 | 138 | 107.0 |
| Nexels (100K) | 29.76 | 0.9083 | 0.2149 | 99,836 | 282 | 87.3 |
| Nexels (400K) | 30.02 | 0.9125 | 0.2105 | 399,288 | 230 | 58.7 |
| Speedy-Splat | 29.13 | 0.9024 | 0.2661 | 314,320 | 74 | 1390.1 |
| FastGS | 29.59 | 0.8996 | 0.2697 | 252,067 | 63 | 1403.2 |
| FastGS (Big) | 29.77 | 0.9078 | 0.2471 | 711,769 | -- | 1231.5 |
| Content-Aware Texturing | 29.62 | 0.9101 | 0.2469 | 232,863 | 255 | 161.1 |
| Textured Gaussians | 28.40 | 0.8845 | 0.2695 | 100,000 | 3,837 | 30.4 |
| NeST (--method baseline) | 29.27 | 0.9026 | 0.2363 | 615,908 | 146 | 28.1 |
| BITYMI (ours) | 29.50 | 0.8898 | 0.2302 | 66,087 | 247 | 2174.8 |
| BITYMI (Gaussian kernel) | 29.85 | 0.8981 | 0.2203 | 142,098 | 502 | 1051.7 |

### db/playroom

| method | PSNR | SSIM | LPIPS | #Prims | Size (MB) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 3DGS | 29.94 | 0.9139 | 0.2393 | 1,840,110 | 435 | 266.6 |
| 2DGS | 30.06 | 0.9116 | 0.2569 | 1,207,669 | 281 | 158.9 |
| Beta-Splatting | 30.24 | 0.9192 | 0.2305 | 2,500,000 | 286 | 158.5 |
| Beta-Splatting (200K) | 30.20 | 0.9090 | 0.2716 | 200,000 | 23 | 523.1 |
| BBSplat | 29.85 | 0.9118 | 0.2502 | 160,000 | 104 | 33.1 |
| Nexels (40K) | 29.57 | 0.8972 | 0.2267 | 39,967 | 138 | 104.4 |
| Nexels (100K) | 30.24 | 0.9067 | 0.2044 | 99,850 | 282 | 84.5 |
| Nexels (400K) | 30.80 | 0.9152 | 0.1987 | 399,088 | 230 | 53.8 |
| Speedy-Splat | 30.10 | 0.9124 | 0.2694 | 187,659 | 44 | 1611.8 |
| FastGS | 30.44 | 0.9094 | 0.2605 | 181,430 | 45 | 1498.3 |
| FastGS (Big) | 30.85 | 0.9152 | 0.2404 | 587,438 | -- | 1320.0 |
| Content-Aware Texturing | 30.36 | 0.9180 | 0.2350 | 137,324 | 248 | 132.8 |
| Textured Gaussians | 29.69 | 0.8981 | 0.2608 | 100,000 | 3,837 | 29.6 |
| NeST (--method baseline) | 28.44 | 0.9017 | 0.2183 | 340,093 | 80 | 47.2 |
| BITYMI (ours) | 30.24 | 0.8907 | 0.2051 | 67,156 | 270 | 1853.1 |
| BITYMI (Gaussian kernel) | 30.71 | 0.9019 | 0.1988 | 124,092 | 494 | 870.2 |

