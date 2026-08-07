# Proberes WebGPU Renderer — Design

**Status:** design, not yet implemented. Written for review before building.
**Source checkpoint:** `/mnt/nilkel_hdd/outputs/mip_360/room/proberes/room8k_p32`
**Reference:** [`PROBERES_PIPELINE.md`](PROBERES_PIPELINE.md) (training-side),
[`FORWARD_PIPELINE_OVERVIEW.md`](FORWARD_PIPELINE_OVERVIEW.md) (baked pipeline).

---

## 1. What changes versus the baked renderer

The baked path gives every surfel its **own rect** in a big atlas, and the
fragment shader maps the surfel's local `(u,v)` into that rect with an
axis-aligned affine — a scale and an offset per axis:

```
au = u0_px + (s.x + E)/(2E) · w_px      // baked: diagonal-only, per-surfel rect
av = v0_px + (s.y + E)/(2E) · h_px
```

Proberes replaces that with **one shared texture** plus a per-surfel
**general affine** (a "probe"), so surfels can sample *anywhere* in the shared
image, at any rotation, scale, or shear:

```
tx = A00·s.x + A01·s.y + t0            // proberes: full 2×2 + translation
ty = A10·s.x + A11·s.y + t1
```

Same number of texture fetches per fragment (one). The difference is 2 extra
multiplies and 2 extra adds, and the removal of the per-surfel rect concept.

| | baked (typeD) | proberes |
|---|---|---|
| texture | per-surfel rects packed into 4096×48576, 7 layers | one shared 8192×8192, 1 layer |
| per-surfel data | rect `(u0, v0, w, h)` — 4 floats | probe `(A00, A01, A10, A11, t0, t1)` — 6 floats |
| mapping | diagonal affine (scale + offset) | general affine (rotation + shear + scale + offset) |
| fragment cost | 2 fmadd + 1 tex | 2×2 matvec (4 mul, 2 add) + 1 tex |
| layer striping | required (exceeds `maxTextureDimension2D`) | **not needed** — 8192 ≤ 8192 limit |

---

## 2. Ground truth — the CUDA convention we must match

From `submodules/diff_surfel_3D_sh_res_probe/cuda_rasterizer/forward.cu`
(case 5, flag `0x1000`):

```cuda
const float2 uv = (rho3d <= rho2d) ? s : float2{0.0f, 0.0f};
const float* pr = hash_features_diffuse + gauss_id_p * 6;
const float tx = pr[0] * uv.x + pr[1] * uv.y + pr[4];
const float ty = pr[2] * uv.x + pr[3] * uv.y + pr[5];
probe_tex_sample(gridrange_diffuse, th, tw, tx, ty, residual_p);
```

Three things to carry over exactly:

**(a) Probe layout is `[A00, A01, A10, A11, t0, t1]`** — row-major 2×2, then
translation. Note WGSL's `mat2x2` constructor is **column-major**, so the
matrix must be built as `mat2x2f(A00, A10, A01, A11)`, not in listed order.
This is the single most likely place to introduce a silent transpose bug.

**(b) The low-pass branch forces `uv = (0,0)`.** When the screen-space
low-pass kernel wins (`rho3d > rho2d` — small/distant/edge-on surfels), the
sample point collapses to the **probe centre**, not the ray-splat point. This
cannot be folded into a precomputed affine; it is a per-fragment conditional.
Omitting it makes distant geometry sample essentially random texels.

**(c) Sampling is texel-centre bilinear with clamp-to-edge** — from
`probe_tex_setup` in `auxiliary.h`:

```cuda
const float x = tx - 0.5f;  const float xf = floorf(x);
smp.fx = x - xf;
smp.x0 = min(max((int)xf,     0), tw - 1);
smp.x1 = min(max((int)xf + 1, 0), tw - 1);
```

This is *exactly* what WebGPU hardware bilinear does given normalised
coordinates `u = tx / tw`, with `filterMode: linear` and
`addressMode: clamp-to-edge`. So we get it free from the sampler — no manual
gather. Feeding `tx/tw` is the whole conversion.

**(d) Activation cascade.** `room8k_p32` was trained without
`--probe_distill_dir`, so `d_residual_mode == 0` — the stacked form:

```
feat = ReLU( ReLU(SH + sh_bias) + residual + res_bias )
```

with `sh_bias = 0.5`, `res_bias = 0.0`. This is the *same* cascade the baked
path already implements, so no shader change is needed here — but it must be
verified rather than assumed, because mode 2 (deferred, per-pixel ReLU) is the
form used by the `res_switch` family and would be wrong here.

---

## 3. Data — measured from the checkpoint

```
probe_head.fixed_probes    (63164, 6)      float32
probe_field.pixels         (8192, 8192, 3) float32
```

63,164 probes matches room's Gaussian count.

**Probe coefficient ranges** (why they can't be fp16):

| col | meaning | min | max | mean |
|---|---|---:|---:|---:|
| 0 | A00 | −279.9 | 317.4 | 0.64 |
| 1 | A01 | −612.9 | 323.2 | 5.99 |
| 2 | A10 | −656.2 | 158.5 | −6.25 |
| 3 | A11 | −198.6 | 1414.5 | 0.29 |
| 4 | t0 | 161.7 | 8031.6 | 4034.7 |
| 5 | t1 | 178.8 | 8030.2 | 4562.2 |

The translation columns span the full 8192 texture and need sub-texel
precision. fp16 has ~11 bits of mantissa; at magnitude 8000 the spacing is
~4 texels — visibly wrong. **Probes must ship as fp32** (or be pre-divided by
`tex_res` to land in [0,1], where fp16 spacing is ~0.0005 → 4 texels, still
too coarse). Ship fp32: 63,164 × 6 × 4 B = **1.5 MB**, negligible.

**Texture statistics:**

```
range   [-2.7527, +4.6599]      mean -0.0341   std 0.1272
nonzero 46,902,836 / 67,108,864 = 69.89%
quantiles  0.1% = -0.719   1% = -0.435   50% = 0.000   99% = +0.308   99.9% = +0.675
```

---

## 4. The quantization risk (must measure, not assume)

Full range spans 7.41 but 99.8% of the mass lies within [−0.719, +0.675], a
span of 1.39. Quantizing uint8 over the **full** range gives:

```
step = 7.41 / 255 = 0.0291     vs signal std 0.127
→ only ~4.4 quantization levels per standard deviation
```

That is the failure mode recorded in `project_bake_quant_clamp_dull` — heavy
tails drag the range out, the bulk gets crushed into a handful of codes, and
renders come out flat with only a small PSNR delta to show for it.

**Measured — and the percentile clamp LOSES.** Use full min/max:

| range | step | RMSE | clipped |
|---|---:|---:|---:|
| **full min/max** | 0.02907 | **0.00854** | **0.000%** |
| P99.9 clamp | 0.00551 | 0.01187 | 0.199% |
| P99.5 clamp | 0.00362 | 0.02286 | 0.992% |
| P99.0 clamp | 0.00293 | 0.02985 | 1.980% |

The clamp makes the bulk 5.3× finer but the 0.2% of clipped texels carry
errors up to ~4.0 (they live out at −2.75 / +4.66), and that tail dominates
the total error. Net: clamping is 1.4× *worse* in RMSE and introduces
artifacts the unclamped version does not have.

**The `bake_quant_clamp_dull` failure mode does not apply here.** That was
specifically a *clamped* range (mean ± 6σ) crushing heavy tails. Full min/max
clips nothing, so there is no dull-intensity bias to induce.

The "~4.4 levels per std" concern from the paragraph above is real but largely
absorbed by BC7, which allocates endpoints **per 4×4 block** and so adapts to
local dynamic range far better than a global uniform quantizer. The global
range mainly sets the worst case, not the typical one.

Still validate end-to-end against `final_test_renders/` after encoding — the
argument above is about the quantizer alone, not the quantizer∘BC7 composition.

---

## 5. Bundle size

Actual shipped room bundles, for comparison:

```
room_astc.bitymi        172.9 MB     ← the real ASTC baseline
room_quant.bitymi       258.7 MB
room_classic.bitymi     258.7 MB
room_quant_astc.bitymi  258.7 MB
room_untex.bitymi        32.3 MB     ← no atlas at all
```

| format | bytes/px | atlas | + BPLY + probes | vs room_astc (172.9 MB) |
|---|---:|---:|---:|---:|
| fp32 (checkpoint) | 12 | 768 MiB | — | — |
| fp16 | 6 | 384 MiB | ~399 MB | 2.3× larger |
| uint8 RGB | 3 | 192 MiB | ~207 MB | 1.2× larger |
| **BC7 / ASTC 4×4** | 1 | **64 MiB** | **~78 MB** | **2.2× smaller** |
| ASTC 6×6 | 0.44 | 28 MiB | ~42 MB | 4.1× smaller |
| ASTC 8×8 | 0.25 | 16 MiB | ~30 MB | 5.8× smaller |

**Proberes at BC7/ASTC-4×4 is a size win, not a cost** — ~78 MB against the
172.9 MB ASTC bundle actually shipping today. One shared 8192² texture with a
general affine per surfel is simply a more compact parameterisation than
per-surfel rects, even before any codebook trick.

typeD's K=65536 codebook does not transfer here (a single shared texture has
far less block-level redundancy than thousands of small per-surfel rects), but
it does not need to: plain BC7 already beats the shipped baseline. The
ASTC 6×6/8×8 rows stay in the table as headroom if we ever want a lite tier,
**not** as a contingency the design depends on.

Note also that fp16 for the shared texture is not the catastrophe §4 implies —
~399 MB is heavy but only 2.3× the current bundle, so if quantization proves
lossy there is a fallback rather than a dead end.

---

## 6. Implementation plan

### 6.1 Export — `scripts/export_proberes_bundle.py` (new)

1. Load `ngp_15000.pth` → `probe_head.fixed_probes [N,6]`,
   `probe_field.pixels [8192,8192,3]`.
2. Load `point_cloud/iteration_15000/point_cloud.ply` → geometry + SV
   (existing path, unchanged).
3. Quantize the atlas to uint8 with a P99.9 clamp; record
   `atlas_scale` / `atlas_offset` into the NAT2 header (slots 12/13, already
   exist).
4. **Pre-divide probes by `tex_res`** so the shader needs no normalisation:
   `A/tex_res`, `t/tex_res`. Emit fp32, stride 6.
5. BC7-encode (existing path) and ASTC-encode (`scripts/encode_astc.py`).
6. Write NAT2 with a new `atlas_format` code.

### 6.2 NAT2 container

Existing codes run 0–8 (`export_textures_bin.py:62-77`). Add:

```
ATLAS_FORMAT_PROBE_BC7  = 9    # single-layer BC7 + [N,6] fp32 probes
ATLAS_FORMAT_PROBE_ASTC = 10   # single-layer ASTC 4×4 + [N,6] fp32 probes
```

Header reuse, no new fields needed:
- `atlas_width/height` = 8192 / 8192
- `n_layers` = 1, `layer_h` = 8192 (no striping — this is the point)
- `num_rects` = N, but the rects block carries **6 floats per surfel, not 4**.
  The format code disambiguates the stride.
- `atlas_scale` / `atlas_offset` = dequant pair, already in slots 12/13.

### 6.3 Shaders (~20 lines)

`preprocess_2dgs.wgsl` — when `probe_mode != 0`, read
`atlas_rects[gid*6 .. +5]` and write A/t straight through instead of deriving
from `(u0, v0, w_span, h_span)`.

`Splat2DGS` is 96 B with `uv_base_x/y`, `uv_scale_x/y`, `layer`, `_pad`.
Because proberes is **always single-layer**, `layer` is always 0 — so `layer`
and `_pad` can carry the two off-diagonal terms. **Zero struct growth, zero
bandwidth change:**

```
uv_base_x/y  ← t0/tex_res, t1/tex_res
uv_scale_x/y ← A00/tex_res, A11/tex_res
layer→skew_x ← A01/tex_res
_pad →skew_y ← A10/tex_res
```

`render_2dgs.wgsl` — alias those two slots, and replace the fragment's affine:

```wgsl
// low-pass branch: collapse to probe centre (matches CUDA case-5)
let uv_eff = select(vec2<f32>(0.0), s, rho3d <= rho2d);
let uv = in.uv_base + mat2x2<f32>(in.uv_scale.x, in.uv_skew.y,   // col 0
                                  in.uv_skew.x,  in.uv_scale.y)  // col 1
                     * uv_eff;
let rgba = textureSampleLevel(atlas, atlas_samp, uv, 0, 0.0);
```

Both `rho3d` (line 217) and `rho2d` (line 221) already exist in the fragment,
so the low-pass conditional costs one `select`.

Gate everything on a `probe_mode` flag in `TexParams` (32 B, has room).

### 6.4 Loader

`Nat2Parser.ts` — recognise formats 9/10, parse the rects block at stride 6.
`ply-loader.ts` — upload a single-layer 8192² texture (`bc7-rgba-unorm` or
`astc-4x4-unorm`), set `probe_mode = 1`.

### 6.5 Validation — before touching the phone

Render one test view in the browser and diff against
`room8k_p32/final_test_renders/`. This is what catches the likely bugs, all of
which are invisible on a phone:

- **column-major transpose** in the `mat2x2` constructor (§2a)
- **missing low-pass branch** → distant geometry samples wrong texels (§2b)
- **texel-centre off-by-half** → uniform half-texel blur
- **v-flip** → vertically mirrored residual
- **wrong activation mode** → residual added post-ReLU instead of pre (§2d)

Only after a numerical match on desktop does the mobile ASTC card make sense.

---

## 7. Traps when reconstructing the config

**`args.json` reports `probe_patch_px 12.0` — this is inert.** That flag is the
from-scratch `ProbeHead3D` footprint and does nothing under `--probe_init_dir`.
The real footprint came from the bake's `--patch_px 32`. Anyone reading
`args.json` to reconstruct the run will get this wrong.

**Cropping saves less than the bake coverage suggests.** The bake covers 38.7%
of the texture, but the *trained* atlas measures 69.89% nonzero (§3) —
finetuning spread values into neighbouring texels. So a crop-to-occupancy pass
recovers far less than the bake number implies, and is not worth building.

## 8. Resolved questions

1. ~~Is the 2× bundle size acceptable?~~ **Resolved — the premise was wrong.**
   Proberes at BC7 is 2.2× *smaller* than the shipped `room_astc.bitymi`
   (§5). Size is a win; the ASTC 6×6/8×8 contingency is not needed.
2. **Quantization headroom** — still the gate to measure first. If P99.9 uint8
   costs more than ~0.2 dB, fall back to fp16 (~399 MB, 2.3× the current
   bundle — heavy but survivable, not a dead end).
3. ~~Does `--probe_no_field` matter at inference?~~ **Resolved:**
   `probe_no_field=True`, so `probe_field.pixels` is the learned leaf
   parameter, not a cached MLP evaluation. Shipping only `pixels` is correct.
