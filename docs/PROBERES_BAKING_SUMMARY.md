# Residual baking: shelf-packed per-surfel atlas vs shared probe atlas

End-to-end summary of the two ways we bake the 3D_SH_res neural residual
(3D hashgrid + fused MLP) into static textures, written 2026-08-06 after the
full mip-360 validation. Covers the legacy shelf-packed pipeline, the proberes
pipeline (oct and φ placement), the finetune, and the deployment renderers.

Related docs: [`PROBERES_PIPELINE.md`](PROBERES_PIPELINE.md) (training-side
detail), [`BAKED_RENDERING.md`](BAKED_RENDERING.md) (legacy pipeline detail),
[`BAKED_RENDERER_EVOLUTION.md`](BAKED_RENDERER_EVOLUTION.md) (prod → lean →
CONIC), [`BENCH_5090_MIP360.md`](BENCH_5090_MIP360.md) (FPS baselines),
[`PROBERES_WEBGPU_RENDERER.md`](PROBERES_WEBGPU_RENDERER.md) (viewer port design).

---

## 0. What is being baked

Training-time color per fragment (`--method 3D_SH_res`, mode 0):

```
color = ReLU( ReLU(SV + 0.5) + residual )
residual = MLP(hash(x))          ← 3D hashgrid + fused 16-wide MLP, per fragment
```

Baking replaces `MLP(hash(x))` with a texture fetch so inference needs no
hashgrid and no MLP. Both pipelines keep geometry + SV untouched; they differ
only in **where the residual texels live and how a fragment addresses them**.

---

## 1. Legacy pipeline — per-surfel rects, shelf-packed (`benchmark_baked.py`)

Every surfel owns a **private rectangle** of texels.

1. **Size each surfel** (`compute_adaptive_resolution`): per-axis Nyquist vs
   the finest hash level — `res = next_pow2(2 · 8σ·s / cell_size)`, clamped to
   `[--min_res, --max_res]` (production `--max_res 64`; `large_gauss_cap`
   optionally squeezes the big-blob bucket). Anisotropic surfels get e.g. 64×8.
2. **Shelf-pack** (`shelf_pack_atlas`): first-fit-decreasing by `res_y` into a
   4096-wide atlas; rows grow as needed (room: 4096×48576). ~99% packing
   utilization.
3. **Evaluate the teacher** at every texel: `xyz = center + u·axis_u + v·axis_v`
   over the rect's uv grid (span ±4σ) → `MLP(hash(xyz))` → write. Sampling
   res == storage res, so this is exact up to the Nyquist clamp.
4. **Quantize + compress**: empirical min/max u8 (full-range — mean±6σ clipped
   tails and caused the dull-render bug), then BC7 (1 B/texel).
   `bake_meta.json` carries `atlas_scale/offset` + all render flags.
5. **No finetune.** The bake is the deliverable.

**Fragment addressing** (diagonal affine + clamp into the rect):

```
au = clamp(u0 + (s.x+E)/(2E)·w_span, rect)     2 FMA + 4 clamps + 1 BC7 fetch
```

**Properties**: exact per-surfel content (no sharing, no collisions); storage
scales with N·Nyquist — mip-360 mean **370 MB** (168–616); quality capped at
the teacher minus quantization (mean −0.04 dB, LPIPS **worse** than teacher:
0.2489 vs 0.2328 — the bake can only lose information).

---

## 2. Proberes pipeline — ONE shared texture + per-surfel affine probes

Every surfel gets **6 floats** (a probe): `tc = A·uv + t` maps its local
ray-splat uv (σ units) into one shared `tex_res²` image; the fragment does one
bilinear fetch there. Storage is **decoupled from N and from Nyquist** — the
atlas is deliberately sub-Nyquist and collisions (surfels sharing texels) are
managed, not forbidden.

### Stage A — probe placement (two options)

**oct (analytic, the default winner).** `oct_probes()`, closed-form, no
training:

```
d    = normalize(center_i − scene_centroid)      # direction on S²
t    = oct_encode(d)·0.96·tex_res + margin       # octahedral unwrap → position
ρ_u  = (patch_px/6)·su/s_med                     # per-axis extent ∝ surfel scale
ρ_v  = (patch_px/6)·sv/s_med                     #   (anisotropic: +0.30 dB vs isotropic)
A    = R(−φ_gauge)·diag(ρ_u, ρ_v)                # gauge angle aligns neighbours
```

**The octahedral map itself** (`_oct_encode` in `hash_encoder/probe_modules.py`)
is the standard sphere→square parameterization from environment-map literature:
inflate the unit sphere onto the unit **octahedron** (L1 ball) by dividing by
the L1 norm, then unfold the octahedron flat:

```
p = d / (|dx|+|dy|+|dz|)                  # radial projection onto the octahedron
upper hemisphere (z ≥ 0):  (ox,oy) = (px, py)            # top 4 faces drop to the square's centre diamond
lower hemisphere (z < 0):  ox = (1−|py|)·sign(px)        # bottom 4 faces FOLD OUTWARD
                           oy = (1−|px|)·sign(py)        #   into the square's corners
result·0.5 + 0.5 → [0,1]²
```

Geometrically: the upper hemisphere occupies the centre diamond of the square,
and the lower hemisphere's four faces are unfolded into the four corner
triangles. Every point of the square is used (unlike Texture-GS's cubemap
cross, which wastes the 4 corner tiles = 50% of its rectangle), the map is
bijective, roughly equal-area, and continuous everywhere except the square's
outer boundary — which is the seam where the −z pole is cut open. Neighbouring
directions land in neighbouring texels (away from that seam), which is what
makes nearby coplanar surfels share atlas neighbourhoods.

This is Texture-GS's spherical-domain idea with two substitutions: octahedral
unwrap instead of their cubemap cross (full square utilization), and the fixed
radial projection `x ↦ (x−centroid)/‖·‖` instead of their learned φ (which is
what makes it training-free — and what discards depth, see the limitation
below).

Coverage is guaranteed by topology (oct map is bijective on the square).
Because footprint and Nyquist are both linear in surfel scale, ONE global
`patch_px` sets every surfel to the same fraction of its own Nyquist
(`patch_px = 12·s_med/cell_size` = 100%). Limitation: purely angular — depth
along a radial ray is conflated (why brain failed; benign on mip-360).

**φ (learned, Texture-GS style — measured worse, kept for reference).**
`probe_uv_field.py train`: 8000 steps of φ:R³→[0,1]² (smooth MLP, no posenc)
+ hash-encoded φ⁻¹, losses = 3D/2D cycle-consistency + Chamfer + Jacobian-area
(`l_area` targets `patch_px`, so φ must be retrained per (tex_res, patch_px)).
Probes read off φ: `t = φ(c)·R`, `A = J_φ(c)·[axis_u|axis_v]·R` (first-order
Taylor — Texture-GS's trick). Verdict across chair + room at 2k and 8k: ties
or loses to oct everywhere (worst −0.30 dB at 8k), coverage collapses harder
as the atlas grows (12.3% vs oct's 29.4% at 8k/p16), `--w_cov` is harmful and
seed-noisy. **The φ stage is droppable.**

### Stage B — scatter bake (`probe_uv_field.py scatter`)

Teacher content written **through the same probes the renderer reads with**
(read/write consistency by construction — the φ⁻¹ bake variant gave cosine
0.06–0.36 and is abandoned):

```
for each surfel, for uv on a grid over ±3σ:
    val = teacher_residual(center + u·axis_u + v·axis_v)
    splat val into T[A·uv + t]      weight = bilinear · exp(−½|uv|²) · opacity · dA
T = num/den                          # colliding surfels AVERAGE
```

Grid = 24×24 uniform (`--scatter_grid 24`, the recommended setting) or
per-surfel Nyquist (`--scatter_grid 0` — better bake cosine, but measured
−0.2 dB after finetune: the prefilter removes the atlas gradient noise that
steers learnable probes apart).

### Stage C — finetune (`train.py --method proberes`)

15k iterations of plain **L1+SSIM against real GT** (no teacher involvement):

```
--probe_init_dir <bake> --probe_no_field --probe_learn_lr 0.01
--densify_until_iter 0 --probe_pixel_decay 0 --probe_tex_res <R>
+ the checkpoint's exact render flags (kernel, aabb, feature, lowpass, -i/-r)
```

Gradient flows into the atlas texels (main), the probes (`--probe_learn_lr`,
+0.1–0.4 dB, lets collided probes migrate apart), and optionally geometry/SV
(freezing either is a no-op ±0.01 dB; do NOT freeze geometry alone while SV
trains — bicycle measured −0.20 dB for that asymmetric combo).

Key facts established about this stage:
- It is **not** distillation. The result drifts AWAY from the teacher
  (blend-match 0.0111 → 0.0078) while images improve; any loss term pulling
  toward the teacher costs quality monotonically (λ=1 → −6.7 dB at its convex
  optimum). The bake is only an initialization.
- The student can EXCEED the teacher where collisions are low: the atlas has
  ~24× the teacher's residual parameters (room +0.53 dB, chair +0.19 dB).
- Per-scene quality tracks collision rate (corr +0.83 LPIPS, −0.59 PSNR with
  log writes/texel). Sizing rule: `patch_px = min(32, scene Nyquist)` —
  kitchen at 194% of Nyquist was the one bad scene (−0.70 dB) and capping to
  p16 recovered +0.38. Footprint past ~60% of Nyquist loses even LPIPS
  (room p55). Footprint drives LPIPS (~64% of the gain), collision relief
  drives PSNR/SSIM.

### Stage D — deployment (`diff_surfel_probe_render` + `benchmark_probe.py`)

Clone of the lean-CONIC baked renderer; `atlas_rects` reinterpreted as the
[N,6] probe array. Fragment: `4 FMA + 1 BC7 fetch` (vs baked's
`2 FMA + 4 clamps + fetch`) + the CONIC ray-splat shared with lean. Two traps,
both fixed and validated per-view (64–66 dB match vs the training renderer):
NO `+0.5` texel offset (tex2D linear already matches `probe_tex_setup`'s
`x = tx−0.5`; the baked path's +0.5 cancels its own −0.5 in `auv_base`), and
warm the lazily-created texture object before the first metric frame.
Low-pass branch required: `uv = (rho3d<=rho2d) ? s : (0,0)`.

Atlas ships as u8+BC7 with **P99.9 percentile clamp** (NOT full min/max):
−0.002…−0.042 dB across all 9 scenes, step/σ ≈ 0.04, 64 MB at 8192².
Probes ship fp32 (fp16 ULP at 8192 is 8 texels) — 1.5 MB.

---

## 3. Where things stand (mip-360, 9 scenes, RTX 5090)

| | PSNR | LPIPS | FPS | storage |
|---|---|---|---|---|
| neural teacher | 26.84 | 0.2328 | 112 | — |
| shelf-packed BC7 + lean CONIC | 26.80 | 0.2489 | **1456** | 370 MB avg |
| **probe BC7 + CONIC** | 26.77 | **0.2270** | 1366 | **64 MB flat** |

Probe vs shelf: PSNR/SSIM even, **LPIPS 9% better** (beats the teacher on
8/9 scenes — the explicit texture holds high-frequency detail the hash+MLP
can't), **5.8× less storage** (and flat in N, vs shelf's N·Nyquist scaling),
0.94× FPS — attributed by SH-only lanes to the finetuned geometry being ~5%
denser, NOT the kernel (atlas-fetch cost measured identical, 11.0% vs 11.2%).

**Winning recipe**: oct anisotropic probes → `--patch_px min(32, Nyquist)` →
grid-24 scatter → 15k GT finetune with `--probe_learn_lr 0.01` → P99.9 u8 →
BC7. No φ, no distillation, no freezing.

**Known limits / open levers**: within-scene allocation is still
size-proportional, nearly uncorrelated with residual detail (bicycle: far
band holds 87% of texels for 33% of variation → background artifacts at p32;
room: texel-richest quartile has 91% of texels, 29% of variation) —
rate–distortion allocation (`D_i(n)` curves under a budget) is the next lever.
Volumetric/nested scenes break oct's angular placement (brain). The WebGPU
port design is written (`PROBERES_WEBGPU_RENDERER.md`), with the same two
sampling traps to avoid.
