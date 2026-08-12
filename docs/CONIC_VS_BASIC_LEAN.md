# CONIC vs basic lean — measured FPS gain across 13 scenes

What the CONIC reformulation of the baked fragment loop buys over the plain
lean renderer, measured same-session on all three benchmarks.

**Headline: +32.0% mean frame rate (1.32×), bit-exact output on 10 of 13 scenes
and within 0.03 dB on the other three.**

Companion docs: [`BAKED_RENDERER_EVOLUTION.md`](BAKED_RENDERER_EVOLUTION.md) —
the prod → lean → CONIC history and the algebra in full;
[`TILE_COST_MODEL.md`](TILE_COST_MODEL.md) — why frame time tracks
(primitive × tile) pairs, and the profiler evidence behind these numbers.

---

## 1. What is being compared

Both columns are the **same submodule** (`diff_surfel_bake_render_lean`), same
forward path (2DGS ray-splat over a surfel + BC7 atlas fetch), same bakes, same
test cameras. Only the CUDA implementation of the per-fragment uv reconstruction
differs.

| | build | per-fragment uv math | per-Gauss shared |
|---|---|---|---|
| **basic lean** | `LEAN_FLAGS=` (none) | ray-splat: `k`,`l`, `cross(k,l)`, 2 divides — ~15 FMA + 2 div | 56 B fp32 (Tu,Tv,Tw, normal_opa, shape) |
| **CONIC** | `LEAN_FLAGS=CONIC` | precomputed rational coefficients, exact reconstruction — ~10 FMA + 1 div | 32 B (u₀,v₀,J⁻¹,opa,shape fp16 + dw fp32), 2× LDS.128 |

The insight (derived in `BAKED_RENDERER_EVOLUTION.md` §3): because `k` and `l`
are both linear in the pixel, `u = p.x/p.z` is *exactly* a rational function of
`(pix.x, pix.y)`. Its coefficients can be computed once per Gauss in
`preprocessCUDA` and the per-fragment work reduced to

```
u = u₀ + (J⁻¹·Δpix).x / (1 + dw·Δpix)      # exact, not a linearization
```

This is a **strictly cheaper reformulation of identical math**, not an
approximation — which is why quality is preserved (§3).

> The CONIC column here is current production, i.e. CONIC **plus** the LDS.128
> staging pack (`TILE_COST_MODEL.md` §13). The pack contributes ~+4 pp of the
> +32%; the conic reformulation alone is ≈ +27%. They compound: 1.27 × 1.04 ≈ 1.32.

## 2. FPS

RTX 5090, idle, `--skip_bake`, finetuned `aftp_shres` bakes, 2026-08-12.

### Mip-NeRF 360

| scene | basic lean | CONIC | gain | ratio |
|---|---:|---:|---:|---:|
| bicycle | 1087.5 | 1518.8 | +39.7% | 1.40× |
| bonsai | 948.2 | 1328.0 | +40.1% | 1.40× |
| counter | 1247.5 | 1580.7 | +26.7% | 1.27× |
| flowers | 887.2 | 1304.8 | +47.1% | 1.47× |
| garden | 1561.2 | 1959.2 | +25.5% | 1.26× |
| kitchen | 1093.6 | 1387.0 | +26.8% | 1.27× |
| room | 1610.0 | 1989.1 | +23.5% | 1.24× |
| stump | 979.2 | 1442.0 | +47.3% | 1.47× |
| treehill | 924.1 | 1308.8 | +41.6% | 1.42× |
| **mean** | **1148.7** | **1535.4** | **+33.7%** | **1.34×** |

### Tanks & Temples

| scene | basic lean | CONIC | gain | ratio |
|---|---:|---:|---:|---:|
| truck | 1924.9 | 2394.1 | +24.4% | 1.24× |
| train | 1684.9 | 2028.0 | +20.4% | 1.20× |
| **mean** | **1804.9** | **2211.1** | **+22.5%** | **1.22×** |

### Deep Blending

| scene | basic lean | CONIC | gain | ratio |
|---|---:|---:|---:|---:|
| drjohnson | 1636.8 | 2183.9 | +33.4% | 1.33× |
| playroom | 1306.0 | 1878.2 | +43.8% | 1.44× |
| **mean** | **1471.4** | **2031.1** | **+38.0%** | **1.38×** |

### All 13

| | basic lean | CONIC | gain |
|---|---:|---:|---:|
| mean FPS | 1299.3 | 1715.6 | **+32.0%** (1.32×) |
| mean frame time | 0.770 ms | 0.583 ms | **−0.187 ms** |
| per-scene gain | — | — | +20.4% … +47.3%, mean +33.9% |

## 3. Quality

The reformulation is exact, and the measured metrics agree:

- **10 of 13 scenes: identical** PSNR, SSIM and LPIPS to 4 decimals.
- **3 scenes differ marginally** — train (22.61 → 22.58 PSNR), bonsai
  (32.63 → 32.62), truck (LPIPS 0.1170 → 0.1169).

Those residuals are floating-point reassociation (the rational form evaluates
the same quantity in a different order, and `u₀`/`J⁻¹` are staged as fp16), not
a modelling difference. Note this is a *different* trade-off from the older
`T2` fp16 flag, which put fp16 inputs through the error-amplifying
`cross(k,l)` and cost 0.05–0.7 dB; CONIC has no cross product, so fp16 staging
is safe here.

## 4. Why the gain varies by scene

The CONIC saving is **per fragment**, so a scene benefits in proportion to how
much of its frame is spent in the fragment loop.

- **Largest gains** — stump (+47.3%), flowers (+47.1%), playroom (+43.8%),
  treehill (+41.6%): high fragment counts, where per-fragment math dominates.
- **Smallest gains** — train (+20.4%), room (+23.5%), truck (+24.4%),
  garden (+25.5%): the fastest / lowest-resolution scenes, where fixed
  per-frame costs (binning, radix sort, preprocess, launch overhead ≈ 45% of
  the frame per `TILE_COST_MODEL.md` §11) are a larger share and dilute a
  fragment-loop win.

This is the same dilution effect seen with the LDS.128 pack, and it bounds any
future fragment-level optimization: at ~52% of frame time in the render kernel,
even eliminating fragment math entirely would cap out near 2×.

*Caveat: the per-scene kernel-vs-frame split was measured on treehill only
(§11); the attribution above is inference from the FPS pattern, not a
per-scene profile.*

## 5. Provenance

| | |
|---|---|
| basic lean | `diff_surfel_bake_render_lean_t8` (clone of the shipped lean; all local edits sit behind `LEAN_*` guards) built with **no** LEAN flags, routed in via `/tmp/strip_bench.py`. Raw log `/tmp/basiclean.log`. |
| CONIC | shipped `diff_surfel_bake_render_lean`, `LEAN_FLAGS=CONIC`, commit `bca3e72` (includes LDS.128 pack). Raw logs `/tmp/mip360_ab.log`, `/tmp/tntdb_ab.log` (AFTER halves). |
| both | `scripts/benchmark_baked.py --skip_bake`, `baked_sh_atlas` row, same bakes, same cameras, idle GPU, all 26 runs on 2026-08-12. |
| jsons | the basic-lean runs' `benchmark_results.json` writes were backed up and restored, so stored results retain the production CONIC numbers. |

Relation to [`BAKED_RENDERER_EVOLUTION.md`](BAKED_RENDERER_EVOLUTION.md): that
doc's table (prod 784 → lean `T2,CTG` 1146 → CONIC 1456) was measured on a
**different, earlier checkpoint set** and against the `T2,CTG` build rather than
the no-flag build used here. The two are not directly comparable — this doc
isolates CONIC against plain lean on the current finetuned bakes.
