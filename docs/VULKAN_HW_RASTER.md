# Vulkan hardware rasterizer for baked checkpoints (`vk_raster/`)

A standalone Vulkan port of the CONIC baked renderer (`diff_surfel_bake_render_lean`,
`LEAN_FLAGS=CONIC`) that replaces the tile-walking CUDA software rasterizer with the
GPU's fixed-function rasterizer and ROP blending. Same bakes, same math, same test
cameras; quality matches to the reported precision. Built to answer one question:
**is our software tile rasterizer faster than the hardware one for this workload?**

**Answer: no — the HW path is 1.4–1.7× faster at identical quality** (room 3368 vs
1986 FPS, garden 2733 vs 1962 FPS, same session, RTX 5090). Per-scene table in
§5 (13-scene sweep appended when complete).

---

## 1. What it is

```
vk_raster/
  main.cpp                 headless Vulkan 1.3 app: load bundle -> preprocess -> sort -> draw
  shaders/preprocess.comp  port of preprocessCUDA (CONIC branch, SV color, aabb_mode 3)
  shaders/radix_*.comp     4-pass 8-bit LSD radix sort over (depth key, index), per PRIMITIVE
  shaders/sort_args.comp   visible count -> indirect dispatch/draw args
  shaders/splat.vert       clipped octagon per sorted Gaussian (fan, <= 10 verts)
  shaders/splat.frag       port of renderBakedCUDA's inner-loop body for ONE contributor
  build.sh                 glslc (conda env) + g++
scripts/export_vk_bundle.py   baked checkpoint + test cams -> flat bundle (see §2)
```

Per frame, all on the GPU and timestamped per stage:

1. **preprocess** (compute, 1 thread/Gauss): frustum cull, `compute_transmat`, AdR cutoff,
   `compute_aabb`, SnugBox conic, SV softmax colour, the CONIC `(u₀,v₀,J⁻¹,dw)`
   precompute — a line-by-line port of the CUDA kernel — plus a depth key. Visible
   Gaussians are **compacted** into the sort list with an atomic counter.
2. **sort**: radix sort of the M visible keys. Because HW draws each primitive once,
   this sorts **primitives** (room: 26 K), not (primitive × tile) instances (room: 806 K
   in the SW path) — 30× fewer keys.
3. **draw**: one `vkCmdDrawIndirect` of M instances × 10 vertices (triangle fan). The
   vertex shader builds a convex polygon that circumscribes the surfel's exact screen
   support; the fragment shader is the CUDA inner-loop body (rational uv reconstruction,
   beta/gaussian kernel with low-pass max-pool, atlas fetch, `max(0, feat+res_bias)`).
   Blending implements the front-to-back "under" operator with destination alpha as
   transmittance:  `color: src·DST_ALPHA + dst`,  `alpha: dst·(1 − src_alpha)`.

Output is an offscreen RGBA16F target (see §3.2); `C + T·bg` is composited at readback.
PSNR is computed exactly as `benchmark_baked` does (per-channel, mean, no clamp).

## 2. Parity with the CUDA renderer

The exporter reproduces the CUDA renderer's inputs byte-for-byte: activated params via
the same `get_*` accessors, SV state via a verbatim copy of `_make_sv_state`, atlas UV
precompute with the fetch-block formulas, view/proj matrices in the memory order the
kernel indexes. The BC7 atlas (up to 54 k rows) exceeds `maxImageDimension2D = 32768`,
so it is uploaded as a **2D array whose layer cuts fall on shelf boundaries** — no rect
straddles a cut, and bilinear taps never leave a rect, so sampling is unchanged.

| scene | CUDA CONIC | Vulkan HW |
|---|---:|---:|
| room PSNR | 31.56 | 31.559 |
| garden PSNR | 27.03 | 27.033 |

The first run matched (31.558) before any tuning — every subsequent optimisation was
checked against this number. Differences vs the CUDA path that are provably lossless:
no `T < 1e-4` early termination (contributions past that point are < 1e-4·colour),
fp32 `u₀/J⁻¹` where CUDA stages fp16, RGBA16F accumulation (measured: no PSNR change).

## 3. What it took — four things that each looked like "HW raster is slow"

Every step below was measured with pipeline-statistics queries (fragment invocations)
plus per-cull-reason atomic counters in a diagnostic shader build (`--stats`).

### 3.1 RGBA32F blending is the wall (681 → 1346 FPS)

First run: 681 FPS, 90 % of the frame in raster. Switching the target from RGBA32F to
**RGBA16F** alone gave 2.7× on the raster stage — fp32 ROP blending runs at a fraction
of the fp16 rate. PSNR unchanged.

### 3.2 Two thirds of fragments were in the mirror region

221 M fragment invocations/frame on room vs 28 M survivors. The cull-reason counters
put **66.8 % at `denom < 0.1`**. The SnugBox conic is the exact iso of
`cx²+cy² < k²·cz²`, but that inequality also holds where `cz` flips sign — the region
where the ray meets the surfel plane *behind* the camera. For grazing floor/wall
surfels that is most of the ellipse. The CUDA path culls it per fragment (the
"foggy discontinuity" guard) and AccuTile walks those tiles too. It is a single
half-plane `dw·d ≥ −0.9`, so the vertex shader Sutherland–Hodgman-clips the octagon
by it (a fixed 8-corner strip turned into a self-intersecting bow-tie when the cut was
deep — hence the fan). Invocations: **221 M → 69 M**, denom culls → 0.

### 3.3 0.6 % of surfels made 50 % of the fragments

Of the remaining 69 M, 34 M came from **161 of 26,037 visible surfels** (room, cam 0):
the ones whose conic is not an ellipse. `compute_aabb` returns garbage extents for them
(median 755 px, max 765,574 px), so the fallback bound was a screen-covering circle —
and **every one of those fragments was culled** (33,979,421 of 33,983,555). These are
also the frames that were 2.5 ms instead of 0.35. Fix: when the wide conic is
hyperbolic, use the `k=3` conic if elliptic (the support is inside `ellipse(3)`, so
this is exact); cull only when both fail. Invocations **69 M → 34.5 M, 82.5 % survive**.
PSNR unchanged on both scenes.

Also for `beta_scaled`: the CUDA loop tests `ρ ≥ 9` *before* the low-pass term, so the
2.35 px low-pass disc I was adding to the bound was pure waste — the support is
`ellipse(3) ∩ (ellipse(r_beta) ∪ disc(r_lp))`.

### 3.4 Clock ramp, not rendering, made the tail

Per-frame fence waits let the GPU idle and down-clock between frames (preprocess on the
first frames after a host sync: 0.29 ms vs 0.015 steady, 15×). Recording all timed
frames into **one submission** (dynamic uniform offsets, a query slot per frame) and
warming for 300 frames removed the bimodality entirely (p95 within 20 % of median).
The CUDA bench enqueues its 400 renders without syncing, so this is the like-for-like
protocol; `[FPS-DIST]` now prints median/p95 there too.

Two smaller items: **fetch-by-id** (pass only the Gaussian index to the fragment shader
and read its 96 B record from the SSBO) beat seven flat `vec4` varyings, and the sort's
per-block latency is best at 1024 elements/block for ~60 K keys and 2048 for ~150 K.

## 4. Where the time goes now

| stage (room, ms) | Vulkan HW | CUDA CONIC (nsys, §11 of TILE_COST_MODEL) |
|---|---:|---:|
| preprocess | 0.015 | ~0.03 |
| sort (+binning) | 0.095 | ~0.14 (806 K instance keys) |
| raster | 0.187 | ~0.26 (tile walk) |
| launch gaps | 0 (one command buffer) | ~0.07 |
| **frame** | **0.297** | **0.503** |

The HW raster stage is faster than the software tile walk on 34 M fragments because it
generates fragments only inside the clipped support polygon (82 % survive) where the
tile walk evaluates every pixel of every touched 16×16 tile (SW: 149 M evaluations,
17 % survive — `TILE_COST_MODEL.md` §10). The **sort is now the largest fixed cost**
(32–39 %); it is a straightforward 4×8-bit LSD implementation and has obvious headroom
(3 passes over the top 24 bits, or a onesweep).

## 5. Results (same session, RTX 5090, warm, 400 timed frames cycling the test set)

| scene | N | CUDA CONIC ms / FPS | Vulkan HW ms / FPS | speed-up | PSNR CUDA / VK |
|---|---:|---:|---:|---:|---|
| room | 62,947 | 0.503 / 1986 | **0.297 / 3368** | **1.70×** | 31.56 / 31.559 |
| garden | 148,658 | 0.510 / 1962 | **0.366 / 2733** | **1.39×** | 27.03 / 27.033 |

(13-scene sweep: `vk_raster/results_2026-08-25.csv`, appended below when complete.)

## 6. Running it

```bash
# once per checkpoint
conda run -n nest_splatting python scripts/export_vk_bundle.py --model_path outputs/mip_360/room/3D_SH_res/aftp_shres
# build (glslc from the nest_splatting env)
./vk_raster/build.sh
# bench (flags = the production configuration)
./vk_raster/vk_raster outputs/mip_360/room/3D_SH_res/aftp_shres/vk_bundle --lp 1 --pad 0.1 --byid --fp16 --elems 4
#   --stats     per-cull-reason fragment counters (slow: atomics), --bench 0 --warmup 0
#   --percam    per-camera and slow-frame report
#   --dump x.ppm  first test view; --dumpprep x.bin  the preprocess buffer
```

Scope: `--feature SV`, kernels `beta_scaled` / `gaussian`, `residual_mode 0`,
`aabb_mode 3` — i.e. the production 3D_SH_res bake. `res_3d_paired` (untextured EWA
half) is not ported.
