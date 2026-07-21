# Baked renderer evolution — prod → lean → CONIC

Chronological account of the three baked-render submodules for 2DGS
surfels + BC7 atlas residual, and what changed at each step. Numbers
throughout are from [`BENCH_5090_MIP360.md`](BENCH_5090_MIP360.md)
(same RTX 5090, same test cameras, same bakes).

| stage | submodule | LEAN_FLAGS | mean FPS (SH+atlas) | vs prod | quality |
|---|---|---|---:|---:|---|
| 1. Prod            | `diff_surfel_bake_render`      | — | 784  | 1.00× | reference |
| 2. Lean pre-CONIC  | `diff_surfel_bake_render_lean` | `T2,CTG` | 1146 | 1.46× | −0.05 to −0.7 dB PSNR |
| 3. Lean CONIC      | `diff_surfel_bake_render_lean` | `CONIC` | 1456 | 1.86× | **bit-identical** |

Same forward path (2DGS ray-splat over surfel + optional BC7 atlas
fetch); only the CUDA implementation of that fragment differs.

---

## 1. The prod renderer (`diff_surfel_bake_render`)

Forward-only fork of the training rasterizer, specialized for the baked
pipeline: SH base color is precomputed per-Gauss in `preprocessCUDA`,
residual comes from a BC7 texture atlas fetched per pixel, no
hashgrid/MLP, no backward.

Per-pixel-per-Gauss fragment inner loop:

```c
// From shared (14 f32 = 56 B/Gauss):
float3 Tu, Tv, Tw;                  // transmat rows (3+3+3 f32)
float4 normal_opacity;              // 4 f32
float  shape;                       // 1 f32

// Ray-splat: solve where pixel ray hits the surfel disc, in local uv.
float3 k = pix.x * Tw - Tu;         // 3 sub
float3 l = pix.y * Tw - Tv;         // 3 sub
float3 p = cross(k, l);              // 6 mul + 3 sub
if (p.z == 0) continue;
s.x = p.x / p.z;                    // 2 fp32 divides
s.y = p.y / p.z;
rho3d = s.x*s.x + s.y*s.y;
depth = s.x*Tw.x + s.y*Tw.y + Tw.z;
```

**Costs, per Gauss, per pixel:**
- ~15 FMAs + 2 fp32 divides
- 56 B shared-memory traffic (transmat + normal_opa + shape)
- Full-fp32 pack limits how many warps coexist per SM

None of this depends on scene complexity — it's the fixed per-contributor
cost. On a typical mip360 tile, each pixel gets touched by ~20-40 Gauss,
so this cost multiplies fast.

At the fragment loop's exit come the standard tasks the CONIC change
does *not* touch:
- kernel eval (beta-scaled `pow(base, shape)` + gaussian `exp(-rho2d/2)`)
- alpha threshold + `test_T` cull
- SH base color load + BC7 atlas fetch + activation + composite

---

## 2. The lean pre-CONIC renderer (`LEAN_FLAGS="T2,CTG"`)

Same submodule (`diff_surfel_bake_render_lean`) but built with two
LEAN flags. Both are quality-preserving in intent but T2 has a small
precision cost in practice.

### T2 = fp16 shared pack

Store the per-Gauss shared-cached data as fp16 instead of fp32:
- Tu, Tv, Tw → 9 fp16 = 18 B (was 36 B)
- normal + opacity → 4 fp16 = 8 B (was 16 B)
- shape → 1 fp16 = 2 B (was 4 B)

Per-Gauss shared drops **56 B → 28 B** (2× smaller). Per-fragment we
upcast to fp32 for math via `__half2float`. Two upstream effects:

- **Shared bandwidth halves.** In the tile-cooperative fetch loop where
  all 256 threads read one Gauss's data via broadcast, we now move half
  the bytes.
- **Occupancy rises.** Less shared budget per block means the SM can
  keep more warps in flight, hiding latency on the divide + memory
  stalls better.

**Quality cost:** the ray-splat cross-product done with fp16 inputs has
~11 bits of mantissa vs fp32's 24. The cross-product amplifies small
errors (Tw is projection-heavy → mixed magnitudes → cancellation), so
`p = cross(k, l)` picks up 0.05-0.7 dB of noise depending on scene.
Visually invisible but bit-different.

### CTG = compile-time template dispatch

The prod kernel branches at runtime on `kernel_type`, `has_atlas`, and
other flags inside the fragment loop. CTG hoists these into
`<template T>` parameters resolved at compile time. The compiler dead-
strips branches you don't take, shrinks the instruction cache footprint,
and eliminates dynamic branch predicts.

Purely a scheduling win — same math, cleaner PTX.

### Result

**Fragment math is unchanged**: 15 FMAs + 2 divides, still the ray-splat
formula. The 46% speedup comes from *scheduling* (less shared traffic,
better occupancy, no branch misses) rather than doing less work per
fragment.

---

## 3. The CONIC renderer (`LEAN_FLAGS="CONIC"`)

Same submodule again, different flag. This one changes the math.

### The insight

The pixel-to-surfel-uv mapping in the ray-splat is
`u = p.x / p.z, v = p.y / p.z` where `p = cross(k, l)` and both `k, l`
are **linear in the pixel**. Since `p.x` (numerator) is a sum of
products of two pixel-linear vectors, and `p.z` (denominator) is
similarly linear (the pix.x·pix.y cross terms cancel), **`u` is exactly
a rational function of `(pix.x, pix.y)`**:

```
u(pix) = (a₀ + a₁·pix.x + a₂·pix.y) / (b₀ + b₁·pix.x + b₂·pix.y)
```

That means we can *precompute* the rational-function coefficients per
Gauss once in `preprocessCUDA`, then reconstruct u, v per pixel from a
much cheaper form — one divide instead of two, no cross product.

### The rearrangement

The standard rational form isn't the cheapest layout. It's algebraically
identical to:

```
u = u₀ + Δu_lin / (1 + dwdxr·Δpix.x + dwdyr·Δpix.y)
v = v₀ + Δv_lin / (1 + dwdxr·Δpix.x + dwdyr·Δpix.y)

where:
  u₀, v₀    = uv at the AABB-center pixel (evaluate ray-splat once)
  Δu_lin    = J⁻¹[0,0]·Δpix.x + J⁻¹[0,1]·Δpix.y    (linear in pixel)
  Δv_lin    = J⁻¹[1,0]·Δpix.x + J⁻¹[1,1]·Δpix.y
  dwdxr,dwdyr = ∂p.z/∂pix.x / p.z, ∂p.z/∂pix.y / p.z   (correction gradients)
  Δpix      = pix − pix_AABB_center
```

`J⁻¹` is the inverse Jacobian of the ray-splat mapping `(pix.x, pix.y) →
(u, v)` at the AABB center — i.e. it says "if you nudge the pixel a bit
from center, how does uv change to first order?" The `1/denom` term is
the correction that keeps the reconstruction *exact* (not just linear)
for larger pixel offsets — it accounts for the fact that a rational
function isn't its own linearization.

Store per-Gauss: `(u₀, v₀, J⁻¹[4], dwdxr, dwdyr)` = 8 floats.

### Per-fragment cost

```c
float dx = pix.x - center_pix.x;
float dy = pix.y - center_pix.y;
float du_lin = J.x*dx + J.y*dy;           // 2 FMA
float dv_lin = J.z*dx + J.w*dy;           // 2 FMA
float denom  = 1.0f + dwdxr*dx + dwdyr*dy;  // 2 FMA
if (denom < 0.1) continue;                  // cull past linearization boundary
float inv_d  = 1.0f / denom;                // 1 divide
float u = u₀ + du_lin * inv_d;              // 2 FMA
float v = v₀ + dv_lin * inv_d;
rho3d = u*u + v*v;
```

**~10 FMAs + 1 divide** vs prod's 15 FMAs + 2 divides. Fewer
instructions AND one fewer divide per fragment.

The `if (denom < 0.1) discard` handles the one degenerate case: the
linearization is only mathematically valid where the denominator stays
positive (i.e. where the surfel is actually "in front" of the pixel in
the perspective sense). Past that boundary the formula's sign flips
and would render smeared colors — the original ray-splat naturally
avoided this because it computed p.z fresh per pixel and could just
check `p.z == 0`.

### The precision trick that keeps it bit-identical

fp16 pack from T2 is applied here too, but selectively:

- `u₀, v₀, J⁻¹, opa, shape` → fp16 in shared (small dynamic range,
  fp16 is fine)
- `dwdxr, dwdyr` → **stay fp32** (these are ratios of z-derivatives to z,
  can span 4-5 orders of magnitude, and any error here corrupts the
  correction denominator across the whole splat)

Per-Gauss shared is now **36 B** (fp16 pack on most fields + 8 B fp32 for
the correction gradients).

Result: **bit-identical rendering vs prod** — 0.001 dB SH-only, 0 dB
SH+atlas. The T2 pack that lost 0.05-0.7 dB when applied to the raw
ray-splat loses nothing when applied to CONIC's rearranged form,
because the precision-sensitive terms are guarded.

### Why it's faster than pre-CONIC despite similar per-Gauss shared budget

Two reasons:
1. **Fewer FMAs per fragment.** 10 vs 15 (~33% less arithmetic).
2. **One divide instead of two.** Blackwell fast_math div is ~4 cycles;
   halving that is a real saving at fragment-loop scale.

Both compound with the shared-bandwidth and occupancy wins from the
fp16 pack that pre-CONIC already had.

---

## Does CONIC also speed up training?

**No — it's a render-only optimization.** Two reasons:

1. **CONIC is forward-only.** The reformulation is a rearrangement of
   the forward ray-splat. Training needs the backward pass — dL/dTu,
   dL/dTv, dL/dTw, dL/dscale, dL/drot — flowing through whatever form
   the forward uses. That backward has never been derived or
   implemented for CONIC. All training rasterizers
   (`diff_surfel_3D_sh_res`, the `_mixed*` family, everything with
   grads) still use the standard cross-product ray-splat.

2. **Training's forward-fragment cost isn't the bottleneck.** During
   training, forward is followed by ~3-5× backward cost, plus the
   fused MLP + hashgrid eval (MODE 5 collab GEMM) which dominates
   per-Gauss compute. A 20% forward-fragment speedup would move
   training wall-clock by ~3-5%. Not worth deriving/verifying/testing
   a new backward.

CONIC lives specifically in `diff_surfel_bake_render_lean`: forward-
only, no hash, no MLP, no residual, no backward — the setting where
the ray-splat *is* nearly 100% of frame time.

---

## Where things live

- Prod submodule: [`submodules/diff_surfel_bake_render`](../submodules/diff_surfel_bake_render/)
- Lean submodule (same source, LEAN_FLAGS switches paths):
  [`submodules/diff_surfel_bake_render_lean`](../submodules/diff_surfel_bake_render_lean/)
- LEAN_FLAGS listed in [`submodules/diff_surfel_bake_render_lean/setup.py`](../submodules/diff_surfel_bake_render_lean/setup.py):
  `T2, CTG, CONIC, DEBUG, FP16_UVJ, FP16_OPASHAPE`
- Full FPS numbers + reproduction commands: [`BENCH_5090_MIP360.md`](BENCH_5090_MIP360.md)
- Plateau analysis for what's *not* worth chasing further:
  `memory/project_lean_bake_render_wins.md`
- Deployed WebGPU viewer already ships a matching CONIC fragment shader
  (Halloumi-WS, bitymi-demos commit `c19bb79`).
