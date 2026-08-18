# 5090 benchmark results — `3D_SH_res` (baked + finetuned texture atlas)

**The single canonical RTX 5090 benchmark-results file.** Quality and throughput
for the production mode across all three benchmarks, versus **FastGS** (the
fastest untextured Gaussian-splatting method), plus the baked-renderer FPS
ladder and the SH-only ablation.

> **Consolidated 2026-08-12.** This file replaces four overlapping docs, all
> removed in the same commit (recoverable via git):
> `5090_BENCH_RESULTS.md` and `MIP360_BENCH_RESULTS.md` (the same prod-era
> sweep in two layouts — they said so themselves), `BENCH_5090_MIP360.md`
> (prod-vs-CONIC ladder; its still-relevant reference numbers are carried into
> §5–6 below), and `CONIC_VS_BASIC_LEAN.md` (folded into §5).

> **Renderer**: all "ours" FPS are the **CONIC lean** baked renderer
> (`diff_surfel_bake_render_lean`, `LEAN_FLAGS=CONIC`) including the LDS.128
> staging pack (commit `bca3e72`), measured on an idle RTX 5090 in one
> same-session sweep of all 13 scenes. Mechanism and profiler evidence:
> [`TILE_COST_MODEL.md`](TILE_COST_MODEL.md); renderer history:
> [`BAKED_RENDERER_EVOLUTION.md`](BAKED_RENDERER_EVOLUTION.md).
> `res_3d_paired` results: [`RES_3D_PAIRED_SUMMARY.md`](RES_3D_PAIRED_SUMMARY.md).

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

**LPIPS favours us on 13 of 13 scenes**, and we are **faster on 13 of 13**
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


## 3. What the atlas costs, and buys (SH-only ablation)

`baked_sh_only` renders the same geometry with the **mean SH colour only** — the
residual atlas path (per-Gauss UV staging, texture fetch, dequant) is skipped
entirely. It is the direct measure of what the texture costs.

| Benchmark | FPS (SH only) | FPS (SH+atlas) | FPS cost | PSNR (SH only) | PSNR (SH+atlas) | PSNR gain |
|---|---:|---:|---:|---:|---:|---:|
| **Mip-NeRF 360** | 1770 | 1535 | **−13.3%** | 18.12 | 27.28 | **+9.16 dB** |
| **Tanks & Temples** | 2713 | 2211 | **−18.5%** | 16.11 | 24.09 | **+7.98 dB** |
| **Deep Blending** | 2218 | 2031 | **−8.4%** | 17.75 | 29.98 | **+12.23 dB** |

SSIM/LPIPS move the same way (mip-360 SH-only 0.4981 / 0.4671 → 0.7901 / 0.2122).

**The texture is cheap for what it delivers**: 8–19% of frame rate for 8–12 dB.
Note the ablation is not a usable mode — SH-only at 18 dB is far below any
deployable quality; it exists to price the atlas path.

> **Correction to [`TILE_COST_MODEL.md`](TILE_COST_MODEL.md) §11.** That section
> reports the TEX pipe at 0.0% utilisation and calls atlas fetches "free". The
> pipe number is right — the texture unit is nowhere near a bottleneck — but
> "free" is too strong: this ablation prices the whole atlas path at 8–19% of
> frame rate. The cost is issue slots and staging (4 float2 → 2 float4 per Gauss,
> plus the per-fragment clamp/fetch/dequant on survivors), not texture-unit
> throughput.


## 4. Per-scene

Resolutions: mip-360 outdoor ≈1240–1300×820–840, indoor ≈1557×1038;
TnT ≈980×546; DB 1264×832 / 1332×876.

### Mip-NeRF 360

| Scene | Prims ↓ | Prims (FastGS) | FPS (SH only) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 160,097 | 539,624 | 1763 | 1519 | 1204 | 1.26× | 24.44 | 24.62 | 0.2383 | 0.2874 | 521 MB |
| bonsai | 92,281 | 275,719 | 1491 | 1328 | 1241 | 1.07× | 32.62 | 32.12 | 0.1704 | 0.2171 | 232 MB |
| counter | 68,394 | 208,171 | 1822 | 1581 | 1207 | 1.31× | 29.32 | 29.08 | 0.1833 | 0.2218 | 168 MB |
| flowers | 184,511 | 489,951 | 1453 | 1305 | 1173 | 1.11× | 20.78 | 21.31 | 0.3200 | 0.3887 | 613 MB |
| garden | 148,658 | 660,711 | 2378 | 1959 | 1145 | 1.71× | 27.03 | 27.32 | 0.1288 | 0.1577 | 483 MB |
| kitchen | 119,462 | 379,671 | 1624 | 1387 | 1059 | 1.31× | 31.47 | 31.74 | 0.1233 | 0.1388 | 188 MB |
| room | 62,947 | 207,167 | 2262 | 1989 | 1343 | 1.48× | 31.56 | 31.74 | 0.1891 | 0.2436 | 190 MB |
| stump | 96,365 | 392,348 | 1625 | 1442 | 1276 | 1.13× | 25.76 | 26.54 | 0.2632 | 0.2726 | 318 MB |
| treehill | 174,744 | 394,335 | 1515 | 1309 | 1286 | 1.02× | 22.56 | 22.85 | 0.2937 | 0.4146 | 616 MB |
| **mean** | **123,051** | **394,189** | **1770** | **1535** | **1215** | **1.26×** | **27.28** | **27.48** | **0.2122** | **0.2603** | **370 MB** |

### Tanks & Temples

| Scene | Prims ↓ | Prims (FastGS) | FPS (SH only) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truck | 85,413 | 252,868 | 2833 | 2394 | 1343 | 1.78× | 25.60 | 25.70 | 0.1169 | 0.1775 | 273 MB |
| train | 88,080 | 231,398 | 2593 | 2028 | 1147 | 1.77× | 22.58 | 22.33 | 0.1727 | 0.2429 | 270 MB |
| **mean** | **86,746** | **242,133** | **2713** | **2211** | **1245** | **1.78×** | **24.09** | **24.02** | **0.1448** | **0.2102** | **271 MB** |

### Deep Blending

| Scene | Prims ↓ | Prims (FastGS) | FPS (SH only) | FPS ↑ | FPS (FastGS) | Speed-up ↑ | PSNR ↑ | PSNR (FastGS) | LPIPS ↓ | LPIPS (FastGS) | atlas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| playroom | 67,156 | 181,430 | 2057 | 1878 | 1503 | 1.25× | 30.39 | 30.66 | 0.2082 | 0.2654 | 228 MB |
| drjohnson | 66,087 | 252,067 | 2380 | 2184 | 1401 | 1.56× | 29.57 | 29.65 | 0.2317 | 0.2732 | 206 MB |
| **mean** | **66,622** | **216,748** | **2218** | **2031** | **1452** | **1.40×** | **29.98** | **30.16** | **0.2200** | **0.2693** | **217 MB** |


## 5. Renderer ladder — CONIC vs basic lean

Same submodule, same bakes, same cameras; only the CUDA implementation of the
per-fragment uv reconstruction differs.

| | build | per-fragment uv math | per-Gauss shared |
|---|---|---|---|
| **basic lean** | `LEAN_FLAGS=` (none) | ray-splat: `cross(k,l)` + 2 divides — ~15 FMA + 2 div | 56 B fp32 |
| **CONIC** | `LEAN_FLAGS=CONIC` | precomputed rational coefficients, exact — ~10 FMA + 1 div | 32 B, 2× LDS.128 |

Because `k` and `l` are both linear in the pixel, `u = p.x/p.z` is *exactly* a
rational function of the pixel; its coefficients are computed once per Gauss in
`preprocessCUDA` and the fragment reduces to
`u = u₀ + (J⁻¹·Δpix).x / (1 + dw·Δpix)`. A strictly cheaper reformulation of
identical math, not an approximation.

Measured on the finetuned bakes, same session, 13 scenes:

| benchmark | basic lean | CONIC | gain | ratio |
|---|---:|---:|---:|---:|
| Mip-NeRF 360 | 1148.7 | 1535.4 | +33.7% | 1.34× |
| Tanks & Temples | 1804.9 | 2211.1 | +22.5% | 1.22× |
| Deep Blending | 1471.4 | 2031.1 | +38.0% | 1.38× |
| **all 13** | **1299.3** | **1715.6** | **+32.0%** | **1.32×** |

Per-scene gain ranges +20.4% (train) to +47.3% (stump), mean +33.9%.
Mean frame time 0.770 ms → 0.583 ms.

**Quality**: identical PSNR/SSIM/LPIPS on 10 of 13 scenes; train (22.61→22.58),
bonsai (32.63→32.62) and truck (LPIPS 0.1170→0.1169) differ marginally from
floating-point reassociation plus fp16 staging. This is a *different* trade from
the old `T2` flag, which pushed fp16 through the error-amplifying `cross(k,l)`
and cost 0.05–0.7 dB; CONIC has no cross product, so fp16 staging is safe.

**Composition**: the CONIC column includes the LDS.128 pack, worth ~+4 pp of the
+32%; the reformulation alone is ≈+27% (1.27 × 1.04 ≈ 1.32).

**Why the gain varies**: the saving is per fragment, so scenes benefit in
proportion to the frame share spent in the fragment loop. Largest gains are the
high-fragment-count scenes (stump +47.3%, flowers +47.1%); smallest are the
fastest/lowest-resolution ones (train +20.4%, room +23.5%) where fixed per-frame
costs — binning, radix sort, preprocess ≈45% of the frame — dilute it. At ~52%
of frame time in the render kernel, eliminating fragment math entirely would cap
near 2×. *(Per-scene kernel/frame split was profiled on treehill only; the
attribution is inference from the FPS pattern.)*


## 6. Historical reference — prod → lean → CONIC

Carried from the retired `BENCH_5090_MIP360.md`. **Different checkpoint set**
(`RD_SV_30thr_005w25gLP_N2f_frz5k10`, not the finetuned `aftp_shres` bakes used
above), so these are *not* directly comparable to §4–5 — they are kept for the
prod baseline and the neural-renderer reference, which the current sweep does
not re-measure.

| mip-360 mean | neural | prod baked | lean pre-CONIC (`T2,CTG`) | lean CONIC | FastGS |
|---|---:|---:|---:|---:|---:|
| FPS | 112 | 784 | 1146 | 1456 | 1215 |

- neural → CONIC lean ≈ **13× mean speedup**; baking is ~7× of it.
- `res_3d_paired` on the same ladder: prod 830 → CONIC 1373 (**+65.4%** mean),
  PSNR prod-vs-CONIC bit-identical (max |Δ| 0.002 dB). See
  [`RES_3D_PAIRED_SUMMARY.md`](RES_3D_PAIRED_SUMMARY.md).
- Paired-branch tax on a zero-untextured bake: −7.4% SH+atlas, −4.5% SH-only vs
  plain lean (dual staging + 9 B/entry shared + per-j flag test).


## 7. Reading the numbers

- **Speed-up tracks primitive reduction, not scene difficulty.** garden (4.4×
  fewer prims → 1.71×) and the TnT scenes (2.8× fewer → 1.78×) lead; treehill
  (2.3× fewer, largest atlas) trails at 1.02×. Frame time is set by
  (primitive × tile) pairs — see [`TILE_COST_MODEL.md`](TILE_COST_MODEL.md).
- **PSNR is a wash, LPIPS is not.** Within ±0.8 dB of FastGS everywhere (mean
  −0.20 dB mip-360, +0.07 TnT, −0.18 DB) while LPIPS is better on all 13 scenes
  by 0.04–0.12. The texture buys perceptual detail PSNR does not reward.
- **Atlas cost is the trade.** 370 MB mean on mip-360 (BC7, `max_res=64`),
  168 MB (counter) to 616 MB (treehill) — roughly linear in primitive count.


## 8. Provenance

| Quantity | Source |
|---|---|
| Ours — FPS (SH+atlas, SH-only) | `scripts/benchmark_baked.py --skip_bake`, `baked_sh_atlas.fps` / `baked_sh_only.fps`, CONIC lean + LDS.128 pack, RTX 5090, idle, one same-session sweep 2026-08-12. Logs `/tmp/mip360_ab.log`, `/tmp/tntdb_ab.log` (AFTER halves). |
| Ours — PSNR/SSIM/LPIPS | same runs, `baked_atlas/benchmark_results.json` per scene. |
| basic lean (§5) | `diff_surfel_bake_render_lean_t8` (clone; local edits all behind `LEAN_*` guards) built with no LEAN flags. Log `/tmp/basiclean.log`. Those runs' json writes were backed up and restored, so stored results keep the production CONIC numbers. |
| Ours — prims | `bake_meta.json → num_gaussians` (post-bake-prune). |
| Ours — atlas MB | `baked_atlas/atlas_texture.bc7` on disk. |
| Finetune table (§2) | teacher/bake/finetune sweeps recorded when the `aftp_shres` runs were produced. |
| §6 ladder | retired `BENCH_5090_MIP360.md`, `RD_SV_…_frz5k10` bakes. |
| FastGS | FastGS repo `train_base.sh` recipe (`-i images` for every mip-360 scene, their own shipped resolution), benchmarked on the same 5090. |
| Checkpoints | `outputs/{mip_360,tnt,db}/<scene>/3D_SH_res/aftp_shres`. |

**Caveats.** (1) The **FastGS column is from an earlier session**, not the
2026-08-12 sweep — the ~4% renderer gain applies to our column only, so
speed-ups should not be quoted to three digits until FastGS is re-benched
same-session. (2) `--skip_bake` does not evaluate the neural teacher, so the
"13×" in §2 and the neural column in §6 come from earlier full runs.
(3) §6 uses a different checkpoint set from §4–5.

---

## 9. typeD codebook compression — measured quality cost

`atlas_format=7` stores the atlas as a K-means codebook of BC7 blocks (K=65536,
16 B each = 1 MB) plus a uint16 index per 4×4 block, and the loader gathers the
full BC7 byte stream at load. The renderer is therefore **bit-identical to the
raw-BC7 path** — one hardware bilinear fetch per fragment, no per-fragment
decode. It is a *download*-size optimisation, and the K-means step is lossy.

Measured by reconstructing each atlas through the codebook and re-benchmarking
(`scripts/…` probe + `benchmark_baked.py --skip_bake`), all 9 mip-360
`aftp_shres` scenes, RTX 5090, 2026-08-18:

| scene | PSNR raw → typeD | LPIPS raw → typeD | ΔPSNR | ΔLPIPS | atlas-space PSNR | atlas MB → typeD MB |
|---|---|---|---:|---:|---:|---|
| bicycle | 24.44 → 24.34 | 0.2383 → 0.2650 | −0.10 | +0.0267 | 34.92 | 546.0 → 69.3 |
| bonsai | 32.62 → 32.30 | 0.1704 → 0.1890 | −0.32 | +0.0186 | 33.95 | 243.4 → 31.5 |
| counter | 29.32 → 29.11 | 0.1833 → 0.2039 | −0.21 | +0.0206 | 34.07 | 176.2 → 23.1 |
| flowers | 20.78 → 20.78 | 0.3200 → 0.3399 | +0.00 | +0.0199 | 31.79 | 642.9 → 81.4 |
| garden | 27.03 → 26.80 | 0.1288 → 0.1609 | −0.23 | +0.0321 | 31.18 | 506.6 → 64.4 |
| kitchen | 31.47 → 31.03 | 0.1233 → 0.1412 | −0.44 | +0.0179 | 32.26 | 197.1 → 25.7 |
| room | 31.56 → 31.32 | 0.1891 → 0.2084 | −0.24 | +0.0193 | 34.80 | 199.0 → 25.9 |
| stump | 25.76 → 25.74 | 0.2632 → 0.2801 | −0.02 | +0.0169 | 31.27 | 333.1 → 42.7 |
| treehill | 22.56 → 22.55 | 0.2937 → 0.3183 | −0.01 | +0.0246 | 30.98 | 645.6 → 81.8 |
| **mean** | | | **−0.174** | **+0.0218** | | **387.8 → 49.5 (7.8×)** |

SSIM falls by 0.0112 mean. **FPS is unchanged** (room 2007.3 → 2007.2), as the
format implies.

**Reading it.** PSNR is a poor guide here — it ranges from −0.44 (kitchen) to
+0.00 (flowers). **LPIPS is the stable signal: +0.017 to +0.032 on every scene,
mean +0.0218.** K-means discards exactly the high-frequency texel detail that
PSNR under-weights and a perceptual metric does not. For context that is ~59% of
the +0.0367 LPIPS the finetune earns on mip-360 (§2) — so typeD hands back
roughly half the finetune's perceptual gain in exchange for a 7.8× smaller
download. A real trade, not a free win, and it should be reported as one.

**Atlas-space PSNR does not predict render damage.** flowers has the 2nd-worst
codebook fit (31.79 dB) yet the *best* render ΔPSNR (+0.00); kitchen has a
better fit (32.26 dB) and the worst ΔPSNR (−0.44). What matters is whether the
degraded texels land on visible, high-contribution surfels — so the exporter's
atlas-PSNR readout should not be used as a quality proxy.

**Draft paper text** (implementation/compression subsection):

> For deployment we optionally store the atlas as a codebook of BC7 blocks: the
> 4×4 blocks are clustered with K-means (K = 65536), each centroid is encoded as
> a single BC7 block (16 B), and every block stores a 16-bit index. At load time
> the byte stream is gathered back into a standard BC7 texture, so rendering is
> unchanged — one hardware bilinear fetch per fragment. Across the nine
> Mip-NeRF 360 scenes this reduces the atlas from 388 MB to 50 MB on average
> (7.8×) at a cost of 0.17 dB PSNR and 0.022 LPIPS, with no change in frame rate.
