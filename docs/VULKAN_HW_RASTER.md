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

## 5. Results — all 13 scenes (RTX 5090, one session, `aftp_shres` bakes)

Protocol, identical for both renderers: warm GPU, 400 timed frames cycling the scene's
test cameras, GPU time only (Vulkan timestamp queries in one batched submission /
`cuda.Event` pairs in `benchmark_baked.py`), same `baked.ply` + BC7 atlas, `--byid --fp16
--lp 1 --pad 0.1`. Quality is the mean over the test set of the *dumped* HW renders
scored by `scripts/eval_vk_renders.py` with the repo's own PSNR/SSIM/LPIPS functions
(`--dumpall`), against the same GT the CUDA numbers use. Raw data:
`vk_raster/results_2026-08-25.csv` (timing) + `vk_raster/metrics_2026-08-25.csv`.

| Scene | test cams | CUDA CONIC ms / FPS | Vulkan HW ms / FPS | speed-up | PSNR CUDA / VK | SSIM CUDA / VK | LPIPS CUDA / VK | VK pre / sort / raster ms |
|---|---:|---:|---:|---:|---|---|---|---|
| mip_360/bicycle | 25 | 0.6603 / 1514 | **0.3554 / 2814** | **1.86×** | 24.44 / 24.44 | 0.7137 / 0.7137 | 0.2383 / 0.2382 | 0.0348 / 0.1399 / 0.1806 |
| mip_360/bonsai | 37 | 0.7586 / 1318 | **0.3794 / 2636** | **2.00×** | 32.62 / 32.62 | 0.9380 / 0.9380 | 0.1704 / 0.1701 | 0.0258 / 0.1165 / 0.2371 |
| mip_360/counter | 30 | 0.6299 / 1588 | **0.4519 / 2213** | **1.39×** | 29.32 / 29.32 | 0.9010 / 0.9009 | 0.1833 / 0.1832 | 0.0218 / 0.1145 / 0.3156 |
| mip_360/flowers | 22 | 0.7759 / 1289 | **0.4477 / 2234** | **1.73×** | 20.78 / 20.78 | 0.5578 / 0.5577 | 0.3200 / 0.3199 | 0.0424 / 0.1425 / 0.2628 |
| mip_360/garden | 24 | 0.5077 / 1970 | **0.3693 / 2708** | **1.37×** | 27.03 / 27.03 | 0.8376 / 0.8376 | 0.1288 / 0.1288 | 0.0395 / 0.1460 / 0.1838 |
| mip_360/kitchen | 35 | 0.7188 / 1391 | **0.5846 / 1710** | **1.23×** | 31.47 / 31.47 | 0.9190 / 0.9189 | 0.1233 / 0.1232 | 0.0422 / 0.1592 / 0.3832 |
| mip_360/room | 39 | 0.5001 / 2000 | **0.3085 / 3242** | **1.62×** | 31.56 / 31.56 | 0.9193 / 0.9193 | 0.1891 / 0.1890 | 0.0261 / 0.0950 / 0.1874 |
| mip_360/stump | 16 | 0.6950 / 1439 | **0.3115 / 3210** | **2.23×** | 25.76 / 25.76 | 0.7228 / 0.7228 | 0.2632 / 0.2632 | 0.0246 / 0.1197 / 0.1672 |
| mip_360/treehill | 18 | 0.7585 / 1318 | **0.4077 / 2453** | **1.86×** | 22.56 / 22.56 | 0.6015 / 0.6015 | 0.2937 / 0.2937 | 0.0308 / 0.1374 / 0.2394 |
| tnt/train | 38 | 0.4902 / 2040 | **0.3645 / 2744** | **1.34×** | 22.58 / 22.53 | 0.8232 / 0.8226 | 0.1727 / 0.1734 | 0.0279 / 0.1230 / 0.2136 |
| tnt/truck | 32 | 0.4176 / 2395 | **0.3067 / 3260** | **1.36×** | 25.60 / 25.56 | 0.8784 / 0.8781 | 0.1169 / 0.1171 | 0.0244 / 0.1162 / 0.1662 |
| db/drjohnson | 33 | 0.4598 / 2175 | **0.2344 / 4265** | **1.96×** | 29.57 / 29.59 | 0.8910 / 0.8910 | 0.2317 / 0.2314 | 0.0124 / 0.0862 / 0.1359 |
| db/playroom | 29 | 0.5367 / 1863 | **0.2267 / 4410** | **2.37×** | 30.39 / 30.41 | 0.8928 / 0.8926 | 0.2082 / 0.2080 | 0.0123 / 0.0818 / 0.1327 |
| **mean (13)** | | 0.608 / 1715 | **0.365 / 2915** | **1.72×** | 27.21 / 27.20 | 0.8151 / 0.8150 | 0.2030 / 0.2030 | 0.028 / 0.121 / 0.216 |

* **Quality is parity**: max |ΔPSNR| 0.05 dB, |ΔSSIM| 0.0006, |ΔLPIPS| 0.0007 across the
  13 scenes — the differences are fp16 blending + the octagon's slightly different
  fragment coverage, not a change of math.
* **Speed-up 1.23–2.37×, mean 1.72×**, largest where the CUDA tile walk is slowest
  (bonsai, stump, playroom — many small primitives per tile); smallest on kitchen, where
  the 3.5 M-fragment raster stage dominates for both.
* Inside the Vulkan frame the raster is 50–70 % and the **sort 25–40 %**; preprocess is
  ≤ 10 %. The sort is the obvious next target (3 passes over 24 bits, or a onesweep).

## 6. Implementation details

**Bundle** (`scripts/export_vk_bundle.py` → `<ckpt>/vk_bundle/`): `meta.txt` (key=value),
raw little-endian fp32 arrays `means[N,3] scales[N,2] rots[N,4] opac[N] shapes[N]
sv_sites[N,K,3] sv_tau[N,K] sv_colors[N,K,3] atlas_params[N,12]`, `atlas.bc7` (symlink
to the bake's file), `cams.bin` (count; per cam `W H view[16] proj[16] campos[3] tanfx
tanfy` + GT as u8 `[3,H,W]` — exact, since the loader's images are u8/255). Matrices are
written in the contiguous torch order, which is exactly the `matrix[0..15]` indexing the
CUDA kernel uses (`transformPoint4x3`: `m[0]x+m[4]y+m[8]z+m[12]`).

**Atlas params** (per Gauss, 3×vec4): `a0 = (u0−½+w/2, v0ℓ−½+h/2, w/2E, h/2E)`,
`a1 = (u0, v0ℓ, u0+w−1.001, v0ℓ+h−1.001)`, `a2.x = layer` — the CUDA fetch block's
precompute with `v` in layer-local coordinates; `a0.z == 0` is the "no atlas rect"
sentinel. Layer cuts: greedy largest multiple-of-4 row ≤ `cut+16384` that no rect
straddles (`inside[]` mask). Sampler: linear, clamp-to-edge, normalized coords
(unnormalized coords are illegal on array views), `(au+½)/W_atlas, (av+½)/layerH`.

**Uniform** (std140 `vec4 v[16]`, indexed as flat floats, one 256 B slot per frame,
`UNIFORM_BUFFER_DYNAMIC`): 0–15 view, 16–31 proj, 32–34 campos, 35 W, 36 H, 37–38
tanfov, 39 N, 40 K, 41 kernel_type, 42 sh_bias, 43 res_bias, 44 compact_mult,
45 opacity_aware_beta, 46 beta_mult, 47 drop_lowpass, 48 scale_mod, 49 atlas_scale,
50 atlas_offset, 51 W_atlas, 52 layerH, 53 nLayers, 54 pad, 55 lpmode.

**Preprocess record** (8×vec4 = 128 B per Gauss, std430 SSBO):
`P0=(xy, u₀, v₀)  P1=J⁻¹  P2=(dwdxr, dwdyr, opa, shape)  P3=(rgb, valid)
P4=tight conic (A,B,E,t)  P5=(filter_r, rx, ry, r_lp_px)  P6=ellipse(3) conic  P7.x=bound level`.
`compute_transmat` is ported by expanding the glm products explicitly:
`T = Aᵀ·W2N·N2P` with `A` rows `(L0,0),(L1,0),(p,1)`, `W2N(r,c)=proj[4r+c]`,
`N2P` the `W/2,(W−1)/2` pixel map; `Tu/Tv/Tw` are the *columns* of the result (glm
`T[0..2]`). The quaternion is consumed as the CUDA code does — `(quat.x, .y, .z, .w)`
read as `(w, x, y, z)`.

**Sort** (`radix_hist / radix_scan / radix_scatter`, 4 passes × 8 bits, LSD):
keys = `floatBitsToUint(p_view.z)` (positive → monotonic); block = 256 threads ×
ELEMS (`--elems`, 4 for ~60 K keys, 8 for ~150 K). Histogram: shared atomics → global
`hist[digit·numBlocks + block]` (digit-major). Scan: one 1024-thread workgroup,
subgroup exclusive scans + 32 partials, chunked. Scatter: block-local **stable** sort by
8 successive 1-bit splits (per-thread counts → workgroup scan → stable positions), then
`dst = hist[digit,block] + (pos − digitStart[digit])`. Only visible primitives are in the
list: preprocess `atomicAdd`s a slot; `sort_args.comp` turns the count into
`vkCmdDispatchIndirect` / `vkCmdDrawIndirect` arguments, so nothing on the host knows M.

**Bound polygon** (`splat.vert`): support function of the ellipse `{d : dᵀMd ≤ t}` is
`h(n) = √(t·nᵀM⁻¹n)`; eight supports at 45° give the circumscribing octagon (corners =
adjacent edge-line intersections). For `beta_scaled` the frag tests `ρ ≥ 9` *before*
the low-pass term, so `h = min(h_ell3, max(h_ell_rβ, r_lp))`; for `gaussian`
`h = max(h_ell_iso, r_lp)`. The rational's validity half-plane `(−dw)·d ≤ 0.9` is then
applied by Sutherland–Hodgman (≤ 9 vertices, emitted as a 10-vertex triangle fan). Pad
0.1 px: pixel `i` is sampled at index coordinate `i` in both paths (vertex NDC =
`(px+½)/W·2−1`, fragment `floor(gl_FragCoord)`), so only fp edge cases need slack.
Bound level: 2 = SW conic elliptic, 1 = only the k=3 conic elliptic (exact: support ⊆
ellipse(3)), 0 = degenerate → not drawn.

**Fragment** (`splat.frag`): the CUDA inner loop verbatim — `dx = pix − xy`, rational
`u = u₀ + (J⁻¹·Δ)/(1+dw·Δ)` with `denom < 0.1 → discard`, `ρ2d = 2|d|²`, beta:
`ρ ≥ 9 → discard`, `α = min(.99, opa·max((1−ρ/9)^shape, e^{−ρ2d/2}))`; gaussian:
`α = min(.99, opa·e^{−ρ/2})`; `α < 1/255 → discard`; atlas fetch with the clamped
affine UV; `feat = max(0, feat + res_bias)`; output `(feat·α, α)`. Blend:
`src·DST_ALPHA + dst·ONE` / `dst·(1−src_α)` with the target cleared to `(0,0,0,1)`, i.e.
`C += T·α·feat`, `T *= 1−α` — the front-to-back "under" operator with T in dst alpha.
`FETCH_BY_ID` passes only the Gaussian index and reads P0–P3 + atlas params in the
fragment shader (faster than 7 flat vec4 varyings on NVIDIA, and it removed a bimodal
stall). `STATS` builds add per-cull-reason atomic counters.

**Timing**: `VK_QUERY_TYPE_TIMESTAMP` ×4 per frame slot (start / after preprocess /
after sort / after draw), all `warmup+bench` frames recorded into one command buffer
and one `vkQueueSubmit`; a `SHADER_READ → SHADER_WRITE` barrier between frames because
frame i+1's preprocess rewrites what frame i's draw read. Quality pass runs per frame
with a readback (RGBA32F or RGBA16F decoded on the host). Requires the discrete GPU
with `pipelineStatisticsQuery`, `dynamicRendering` (Vulkan 1.3 core), subgroup
arithmetic; no window, no validation layers.

## 7. Running it

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
