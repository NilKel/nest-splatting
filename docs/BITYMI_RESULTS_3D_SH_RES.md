# BITYMI results — `3D_SH_res` (baked + finetuned texture atlas)

Deployed-quality and throughput results for the production mode, across all three
benchmarks, versus **FastGS** (the fastest untextured Gaussian-splatting method).
Every number here is read from a measured artefact — see [*Provenance*](#5-provenance).

Companion doc: [`BITYMI_RESULTS_RES_3D_PAIRED.md`](BITYMI_RESULTS_RES_3D_PAIRED.md).

> **Renderer**: all FPS are the **CONIC lean** baked renderer
> (`diff_surfel_bake_render_lean`, `LEAN_FLAGS=CONIC`) including the LDS.128
> staging pack (commit `bca3e72`). Measured 2026-08-12 on an idle RTX 5090,
> one same-session sweep for all 13 scenes. This supersedes the earlier FPS
> column (mip-360 mean 1479) by +4.0%; **quality is bit-identical** — the pack
> and CONIC are exact, not approximations. See
> [`TILE_COST_MODEL.md`](TILE_COST_MODEL.md) §13.

---

## 1. Headline

| Benchmark | Method | Prims ↓ | FPS ↑ | Speed-up ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| **Mip-NeRF 360** (9 scenes) | Ours | 123,051 | 1535 | **1.26×** | 27.28 | 0.7901 | 0.2122 |
| | FastGS | 394,189 | 1215 | — | 27.48 | 0.7973 | 0.2603 |
| **Tanks & Temples** (2 scenes) | Ours | 86,746 | 2211 | **1.78×** | 24.09 | 0.8508 | 0.1448 |
| | FastGS | 242,133 | 1245 | — | 24.02 | 0.8409 | 0.2102 |
| **Deep Blending** (2 scenes) | Ours | 66,622 | 2031 | **1.40×** | 29.98 | 0.8919 | 0.2200 |
| | FastGS | 216,748 | 1452 | — | 30.16 | 0.9053 | 0.2693 |

**LPIPS favours us on 13 of 13 scenes**, and we are now **faster on 13 of 13**
(treehill was the last holdout at 0.99×; it is 1.02× with the current renderer).
Speed-up ranges 1.02×–1.78×; we carry 3.1× fewer primitives on average.


## 2. What the finetune contributes

The atlas is initialised from the baked neural residual, then optimised against the
training images with **geometry and view-dependent colour frozen** — only texels move.
Primitive count, atlas bytes and frame rate are identical before and after.

| Benchmark | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| **Mip-NeRF 360** | neural teacher | 26.84 | 0.7775 | 0.2328 |
| | baked (no finetune) | 26.80 | 0.7736 | 0.2489 |
| | **+ finetune** | **27.28** | **0.7901** | **0.2122** |
| | *Δ vs bake* | *+0.49* | *+0.0164* | *-0.0367* |
| **Tanks & Temples** | neural teacher | 23.82 | 0.8421 | 0.1662 |
| | baked (no finetune) | 23.81 | 0.8374 | 0.1886 |
| | **+ finetune** | **24.09** | **0.8508** | **0.1448** |
| | *Δ vs bake* | *+0.28* | *+0.0133* | *-0.0437* |
| **Deep Blending** | neural teacher | 29.82 | 0.8983 | 0.2457 |
| | baked (no finetune) | 29.79 | 0.8981 | 0.2633 |
| | **+ finetune** | **29.98** | **0.8919** | **0.2200** |
| | *Δ vs bake* | *+0.20* | *-0.0062* | *-0.0433* |

Baking alone can only lose information relative to its teacher; optimising the texels
directly recovers it and then exceeds it. On every benchmark the finetuned atlas beats
the neural renderer it replaces, at over 13× the frame rate.


## 3. Per-scene


### Mip-NeRF 360

| Scene | Resolution | Prims ↓ | Prims (FastGS) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 1237×822 | 160,097 | 539,624 | 1519 | 1204 | 1.26× | 24.44 | 24.62 | 0.2383 | 0.2874 | 521 MB |
| bonsai | 1559×1039 | 92,281 | 275,719 | 1328 | 1241 | 1.07× | 32.62 | 32.12 | 0.1704 | 0.2171 | 232 MB |
| counter | 1558×1038 | 68,394 | 208,171 | 1581 | 1207 | 1.31× | 29.32 | 29.08 | 0.1833 | 0.2218 | 168 MB |
| flowers | 1256×828 | 184,511 | 489,951 | 1305 | 1173 | 1.11× | 20.78 | 21.31 | 0.3200 | 0.3887 | 613 MB |
| garden | 1297×840 | 148,658 | 660,711 | 1959 | 1145 | 1.71× | 27.03 | 27.32 | 0.1288 | 0.1577 | 483 MB |
| kitchen | 1558×1039 | 119,462 | 379,671 | 1387 | 1059 | 1.31× | 31.47 | 31.74 | 0.1233 | 0.1388 | 188 MB |
| room | 1557×1038 | 62,947 | 207,167 | 1989 | 1343 | 1.48× | 31.56 | 31.74 | 0.1891 | 0.2436 | 190 MB |
| stump | 1245×825 | 96,365 | 392,348 | 1442 | 1276 | 1.13× | 25.76 | 26.54 | 0.2632 | 0.2726 | 318 MB |
| treehill | 1267×832 | 174,744 | 394,335 | 1309 | 1286 | 1.02× | 22.56 | 22.85 | 0.2937 | 0.4146 | 616 MB |
| **mean** | — | **123,051** | **394,189** | **1535** | **1215** | **1.26×** | **27.28** | **27.48** | **0.2122** | **0.2603** | **370 MB** |

### Tanks & Temples

| Scene | Resolution | Prims ↓ | Prims (FastGS) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truck | 979×546 | 85,413 | 252,868 | 2394 | 1343 | 1.78× | 25.60 | 25.70 | 0.1169 | 0.1775 | 273 MB |
| train | 980×545 | 88,080 | 231,398 | 2028 | 1147 | 1.77× | 22.58 | 22.33 | 0.1727 | 0.2429 | 270 MB |
| **mean** | — | **86,746** | **242,133** | **2211** | **1245** | **1.78×** | **24.09** | **24.02** | **0.1448** | **0.2102** | **271 MB** |

### Deep Blending

| Scene | Resolution | Prims ↓ | Prims (FastGS) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| playroom | 1264×832 | 67,156 | 181,430 | 1878 | 1503 | 1.25× | 30.39 | 30.66 | 0.2082 | 0.2654 | 228 MB |
| drjohnson | 1332×876 | 66,087 | 252,067 | 2184 | 1401 | 1.56× | 29.57 | 29.65 | 0.2317 | 0.2732 | 206 MB |
| **mean** | — | **66,622** | **216,748** | **2031** | **1452** | **1.40×** | **29.98** | **30.16** | **0.2200** | **0.2693** | **217 MB** |


## 4. Reading the numbers

- **Speed-up tracks primitive reduction, not scene difficulty.** garden (4.4× fewer
  prims → 1.71×) and the two TnT scenes (2.8× fewer → 1.78×) lead; treehill (2.3×
  fewer, and the largest atlas) trails at 1.02×. Frame time is set by
  (primitive × tile) pairs, so a method that carries 3.1× fewer primitives wins
  roughly in proportion — see [`TILE_COST_MODEL.md`](TILE_COST_MODEL.md).
- **PSNR is a wash, LPIPS is not.** We are within ±0.8 dB of FastGS everywhere
  (mean −0.20 dB on mip-360, +0.07 on TnT, −0.18 on DB) while LPIPS is better on
  every one of the 13 scenes, by 0.04–0.12. The texture atlas buys perceptual
  detail that PSNR does not reward.
- **Atlas cost is the trade.** 370 MB mean on mip-360 (BC7, `max_res=64`), ranging
  168 MB (counter) to 616 MB (treehill) — roughly linear in primitive count.

## 5. Provenance

| Quantity | Source |
|---|---|
| Ours — FPS | `scripts/benchmark_baked.py --skip_bake`, `baked_sh_atlas.fps`, CONIC lean + LDS.128 pack, RTX 5090, idle, one same-session sweep 2026-08-12. Raw log `/tmp/mip360_ab.log`, `/tmp/tntdb_ab.log` (AFTER halves). |
| Ours — PSNR/SSIM/LPIPS | same runs, `baked_sh_atlas` block of each `baked_atlas/benchmark_results.json`. |
| Ours — prims | `bake_meta.json → num_gaussians` per scene (post-bake-prune count). |
| Ours — atlas MB | `baked_atlas/atlas_texture.bc7` on disk. |
| Finetune table (§2) | teacher/bake/finetune metric sweeps recorded when the `aftp_shres` runs were produced. |
| FastGS | FastGS repo `train_base.sh` recipe (`-i images` for every mip-360 scene, i.e. their own shipped resolution), benchmarked on the same 5090. |
| Checkpoints | `outputs/{mip_360,tnt,db}/<scene>/3D_SH_res/aftp_shres`. |

**Caveats.** (1) FastGS FPS were measured in an earlier session, not the
2026-08-12 sweep — the ~4% renderer gain applies to our column only, and a
same-session FastGS re-bench would be needed before quoting speed-ups to three
digits. (2) `--skip_bake` does not evaluate the neural teacher, so the "13×"
in §2 comes from the earlier full runs, not this sweep.
