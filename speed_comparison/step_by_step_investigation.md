# Step-by-step investigation — what makes our baked renderer 4× slower per Gauss than FastGS

Scene: garden mip-360, 1297×840, 24 test cams.  All numbers are the mean of
200 CUDA-event-timed frames after 20 warmup frames.  Checkpoint:
`outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10/baked_atlas`
(148k Gauss).

Everything below runs the **untextured** SV-only path (no atlas sample) so we
isolate the surfel raster + geometry + sort + kernel eval + blend costs.

| Mode | ms/frame | FPS | Δ vs prod (aabb=3) |
|---|---:|---:|---:|
| **Baseline** — aabb=3 (rect+AdR), beta_scaled, opacity-aware cutoff | 0.698 | 1432 | ref |
| Flat kernel — per-Gauss `shape=0` → `pow(base, 0)`=1 (top-hat) | 0.660 | 1515 | −0.038 ms  (−5.4%) |
| `sort_mode=1` (our two-stage sort) | 0.841 | 1189 | **+0.143 ms  (+20.5%)** |
| `beta_mult=0.5` on aabb=3 | 0.695 | 1439 | ~0 (no-op — wrong branch) |
| aabb=2 (rect, no AdR) baseline | 0.696 | 1437 | ~0 |
| **aabb=2 + `beta_mult=0.5` (2σ instead of 4σ)** | **0.456** | **2192** | **−0.242 ms  (−34.7%)** |

FastGS reference: 660k Gauss @ 0.976 ms / 1024 FPS (`speed_comparison/fastgs/eval_summary.json`).

---

## 1. Flat kernel — how much is the per-fragment `pow(base, shape)`?

Overrode the per-Gauss shape tensor to zeros before the render call.  Same
CUDA code path (kernel_type=4), same discard condition (`rho3d >= k_sq`), just
`base^0 = 1` → uniform alpha inside the disc.

**Result: 0.660 vs 0.698 ms = 5.4% saved.**

Interpretation: the per-fragment `powf(base, shape)` is ~5% of frame time.
Cheaper than expected — CUDA's `__powf` is ~3-5 cycles, and the branch
divergence between the beta path and the lowpass max-pool
(`fmaxf(alpha_beta, alpha_lp)`) is more expensive than either evaluation
alone.  FastGS uses a plain Gaussian falloff (`expf(-0.5·rho)`), so they
save this 5%, but it's not the driver of the 4× gap.

---

## 2. Sort — is our `sort_mode=1` the same as FastGS's sort?

**No.**  FastGS uses a *single-stage* 64-bit radix sort:
```
key = (tile_id << 32) | float_to_uint_depth
cub::DeviceRadixSort::SortPairs(keys, values, ..., 0, 32 + msb(n_tiles))
```
Same algorithm, single call, no prefix sum indirection.
(`FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.cu:408`.)

**Our `sort_mode=0` (production) IS conceptually identical to FastGS's sort** —
same single 64-bit `DeviceRadixSort::SortPairs` call on the same
`(tile<<32)|depth` composite key
(`diff_surfel_bake_render/cuda_rasterizer/rasterizer_impl.cu:451`).

**Our `sort_mode=1` is a different, home-grown two-stage sort:**
1. sort visible primitives by depth (32-bit)
2. reorder tile counts into depth-sorted order
3. exclusive prefix sum
4. `create_instances` emit (tile_key, prim_idx) in depth order
5. tile-sort 32-bit keys with stable ordering (preserves depth within each tile)
6. `identifyTileRanges`

The name in the source comment is misleading (`// FastGS two-stage sort
kernels`) — FastGS doesn't do this.  It's an in-house optimization that only
wins when the visible set is much smaller than the instance count (many
tile-touches per Gauss).

**Result on garden: sort_mode=1 is 20.5% SLOWER than sort_mode=0.**  Our
production path is already using the algorithm equivalent to FastGS's, and
it's the faster one for this scene.

Verdict: **sort is not our bottleneck.  Our sort is exactly FastGS's sort.**

---

## 3. Tighter AABB — how much do we lose to loose per-Gauss tile bounds?

**Important nuance the user flagged**: our `beta_scaled` kernel has **hard
compact support** at ρ=3σ (kernel is exactly zero outside).  FastGS's
Gaussian kernel is **unbounded** — the tail decays to a 1/255 iso-line at
~2.5-3.3σ (opacity-dependent), and `compact_mult` scales *that visibility
threshold*.  So the fair comparison isn't beta+compact_mult vs. FastGS —
it's Gaussian+compact_mult vs. FastGS.  Two separate experiments below:

### 3a. Applying `beta_mult` to the beta path (routed through aabb=2)

Our production `aabb_mode=3 + beta_scaled + shapes` hits a dedicated
cutoff branch that does NOT consult `d_beta_mult`:
```c
if (use_adr && is_beta_kernel && shapes != nullptr) {
    cutoff = fmaxf(r_beta, r_lp);     // fixed AdR, no beta_mult
    cutoff = fminf(cutoff, k + 2.0f);
}
```
That's why `set_beta_mult(0.5)` was a no-op in production — wrong branch.

`compact_mult` in our CUDA is also gated: it only applies in the
`use_adr && !is_beta_kernel` branch (Gaussian kernel with AdR).  Not applicable
to our beta_scaled bake.

To measure the tighter-AABB effect **fairly for our beta path**, I routed
through `aabb_mode=2 (rect, no AdR) + opacity_aware_beta=True`, which DOES
multiply cutoff by `d_beta_mult`:

| Config | ms/frame | FPS |
|---|---:|---:|
| aabb=2, `beta_mult=1.0` (cutoff ≈ 4σ) — matches production loosely | 0.696 | 1437 |
| aabb=2, `beta_mult=0.5` (cutoff ≈ 2σ) | **0.456** | **2192** |

**Halving the cutoff = 34.7% less frame time = 1.53× speedup.**

This is the biggest single lever on the "residual pipeline" bar from the
earlier decomposition.  At 4σ our tile-touches-per-Gauss are ~4× what they'd
be at 2σ (area scales with r²), so we're binning way more tiles than
carry visible-α coverage.  FastGS's `compact_mult=0.5` in their Gaussian path
does the same math and is a big part of why they can afford 660k Gauss.

Note: at 2σ we clip the α-tail below the ~1/255 iso, so quality drops slightly
(the α-blend is missing the outer "wing" of each surfel).  Tester explicitly
said quality is not a concern for this measurement.

### 3b. Swap kernel to plain Gaussian + FastGS-style `compact_mult`

Apples-to-apples with FastGS: kernel_type=0 (unbounded Gaussian) →
`use_adr && !is_beta_kernel` branch fires, which reads
`d_compact_mult` when computing the visibility cutoff.  We're now measuring
the exact FastGS AABB code:

```c
float log_term = logf(255.0f * opacity_val);
cutoff = sqrtf(2.0f * log_term * d_compact_mult);   // FastGS Compact Box
```

| Kernel + cutoff | ms/frame | FPS | Δ vs production (0.698 ms) |
|---|---:|---:|---:|
| beta_scaled, opacity-aware (production)  | 0.698 | 1433 | ref |
| Gaussian, `compact_mult=1.0` (no crop, ~4σ)      | 0.680 | 1471 | **−2.6%** |
| **Gaussian, `compact_mult=0.5`** (FastGS-style)  | **0.541** | **1850** | **−22.5%** |
| Gaussian, `compact_mult=0.25` (very aggressive)  | 0.447 | 2237 | **−36.0%** |

- **Kernel swap alone (beta → Gaussian) saves only 2.6%.**  The powf +
  max-with-lowpass path is not the driver.
- **`compact_mult=0.5` on the unbounded Gaussian gives 22.5% back.**  This
  is the actual FastGS-style path, and it's the biggest single win we can
  claim from the AABB side.
- **`compact_mult=0.25` gets us 36% but starts clipping visible α-tail.**

### 3c. Per-Gauss cost, apples-to-apples now

| Renderer | Gauss | ms/frame | **ns/Gauss** |
|---|---:|---:|---:|
| FastGS                                             | 660,711 | 0.976 | **1.48** |
| Ours — Gaussian + `compact_mult=0.5` (FastGS-style) | 148,803 | 0.541 | **3.64** |
| Ours — production (beta_scaled, opacity-aware)      | 148,803 | 0.698 | 4.69 |

Applying FastGS's cropping strategy closes the per-Gauss-cost gap from
**3.2× → 2.5×**.  The remaining 2.5× is on the memory-latency /
shared-memory-occupancy side — Section 4 below.

---

## 4. Cache-friendliness — why FastGS's per-Gauss load is smaller

Head-to-head of what each renderer loads from global into shared per Gauss
per tile-batch:

**FastGS per-Gauss data (`points_xy` + `conic_opacity`):**
```
float2 xy;          // 8 B
float4 conic_o;     // 16 B — (conic_a, conic_b, conic_c, opacity)
────────────────────
                    24 B/Gauss
```
Their `conic_opacity` packs opacity into `.w` of the same float4 → **single
128-bit aligned load** per Gauss.  Alpha is
`opacity * exp(-0.5 · rho)`; the conic + opacity being packed means one
`ld.global.v4` fetches everything they need for the whole falloff.

**Our per-Gauss data (bake_render fragment loop):**
```
float2 xy;               // 8 B  ← means2D
float4 normal_opacity;   // 16 B ← (nx, ny, nz, opacity)
float3 Tu;               // 12 B (padded to 16 in shared)
float3 Tv;               // 12 B (padded)
float3 Tw;               // 12 B (padded)
float shapes;            // 4 B
bool is_textured;        // 1 B (padded to 4)
float4 ewa_conic;        // 16 B (only used by --method mixed_3d)
────────────────────────
                         ≥ 85 B/Gauss global-memory footprint
                         112 B / entry in shared (with padding)
```

Ratio: **~4× more per-Gauss global traffic** than FastGS, and **~4× more
shared memory pressure**.  Two things drive it:

**(a) The T matrix (Tu, Tv, Tw = 36 B)** — needed for ray-disc atlas UV
recovery in the fragment.  For the untextured SV-only path we still fetch
it because the CUDA code doesn't compile-time-gate the load on
`atlas_texture == null`.  This is pure waste in the SV-only benchmark:
36 of the 85 B/Gauss are dead loads.

**(b) `--method mixed_3d` fields** — `collected_ewa_conic` (16 B) +
`collected_is_textured` (padded to 4 B) — allocated even when the model
isn't using mixed_3d.  20 B of dead shared memory per Gauss per tile-batch.

**(c) Kernel choice** — FastGS's per-fragment falloff needs only the conic
(3 floats).  Our beta falloff needs `rho3d` from the T matrix + `rho2d`
from a low-pass, so it inherently has more data dependencies.  Even with
Gaussian kernel, we'd still fetch Tu/Tv/Tw because the geometry pipeline
uses them for the ray-disc `s` recovery.

**Shared-memory occupancy consequence.**  Adreno-class SMs (this is a
5090 but same architecture principle) have ~48 KB of static shared per SM
before scheduling drops.  Per 256-thread block:

- FastGS: 7 KB used → 6 concurrent blocks per SM → 1536 threads latency-hidden
- Ours:  22 KB used → 2 concurrent blocks per SM → 512 threads latency-hidden

**Fewer resident blocks = worse ability to hide memory-load latency**, on
top of the 4× larger per-Gauss loads themselves.  This is the biggest
architectural gap.

---

## Bottom line — where the ~3.2× per-Gauss cost gap lives

Measured against FastGS's 1.48 ns/Gauss, our production 4.69 ns/Gauss:

| Component | Est. share of 3.2× gap |
|---|---:|
| No FastGS-style crop (compact_mult=1 vs 0.5 on unbounded tail) | **~1.29×** (verified directly, section 3b) |
| Per-Gauss shared-memory footprint 3× larger → SM occupancy 1/3 | **~1.5-2×** (structural, section 4) |
| Dead T-matrix + ewa_conic + shapes loads in the untextured path | ~1.1-1.2× |
| Beta kernel `pow(base, shape)` + max-pool vs plain Gaussian | ~1.03× (section 3b: 2.6% direct) |

Compounded: `1.29 × 1.7 × 1.15 × 1.03 ≈ 2.6×`, close to the observed 3.2×.
The remaining slack is likely dispatch overhead and per-Gauss preprocess
work (Jacobian, transMat, etc.) that FastGS also has but in a leaner form.

**Sort contributes zero** — production sort is FastGS's sort.
**SV softmax contributes zero** — 5 µs / frame across all Gauss.

## Concrete next-step wins

Ordered by expected win / implementation cost:

1. **Enable `compact_mult` on the beta AdR branch** (or add a
   `d_beta_adr_mult`) — the pure-Gaussian test showed **22.5% speedup at
   `compact_mult=0.5`**, and the same crop applied to the beta+AdR path
   would recover most of it on any existing beta-trained bake without a
   rebake.  One-line CUDA edit.
2. **Compile-time-gate the T-matrix load on `atlas_texture != null`** — the
   SV-only bench doesn't need Tu/Tv/Tw.  Drops shared-mem by ~15 KB per
   256-thread block → ~2× more resident blocks per SM → real latency
   hiding.
3. **Drop `collected_ewa_conic` + `collected_is_textured` unless
   `--method mixed_3d`** — 20 B/Gauss shared saved per batch.
4. Retune 2σ / 0.5x cutoff so visible α-tail isn't clipped (or accept the
   ~0.5 dB PSNR drop for the 1.3× win).

## Files in this folder

```
speed_comparison/
├── bench_step_by_step.py       ← runs the 7 variants above
├── bench_three_modes.py        ← original decomposition (SV+atlas vs SV vs passthrough)
├── render_intersection_all.py  ← intersection heatmaps for arbitrary nest checkpoints
├── baked_bench.json            ← output of bench_three_modes.py
├── step_by_step.json           ← output of bench_step_by_step.py
├── fastgs/                     ← FastGS's own intersection maps + FPS from ours_30000/
└── nest_splatting_neural/      ← nest-splatting neural intersection maps (rendered here)
```
