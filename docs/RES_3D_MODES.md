# Res-3D Mode Family — Staged Curriculum + 2D/3D Surfel Split

This document covers the four related training modes that share the staged
mode-0 → mode-2 curriculum and (optionally) the textured-2D / untextured-EWA
split. They sit on top of `3D_SH_res` (the per-Gaussian SH + tiny hash-MLP
residual architecture) and address two specific issues during training:

1. **Activation discontinuity at the mode flip** (mode 0 → mode 2): the
   per-Gauss outer ReLU clamps gradient in early iters but the post-blend
   per-pixel ReLU passes richer signal later. Switching mid-training trades
   the strengths of both.
2. **Geometric capacity collapse for high-frequency residual**: a single
   2DGS surfel has a thin disc-shaped support, which is good for residual
   detail but limits coverage of smooth view-dependent SV colour. Splitting
   each surfel into a 2D residual-carrier (for hash+MLP detail) and a 3D
   EWA-ellipsoid SV-carrier (for soft SV colour) lets each half specialise.

These modes also share the renderer-side post-blend `LeakyReLU` composition
that lets signed per-Gauss residual subtract from accumulated SV — see the
`--lru α` flag.

---

## Mode summary table

| Mode | Cascade | Tex carrier `feat` | Untex carrier | Kernel post-split |
|---|---|---|---|---|
| `3D_SH_res` (baseline) | single | `ReLU(SV+0.5) + residual` | n/a (no split) | `diff_surfel_3D_sh_res` |
| `res_switch` | single | `ReLU(SV+0.5) + residual` | n/a (no split) | `diff_surfel_3D_sh_res` |
| `res_3d` | **DUAL** | `residual` only (bias gate ON) | EWA SV-only | `diff_surfel_res_3d` (single-pass) |
| `res_3d_paired` | JOINT | `ReLU(SV+0.5) + residual` | EWA SV-only | `diff_surfel_mixed_3d` |
| `res_3d_double` | **DUAL** | `ReLU(SV+0.5) + residual` | EWA SV-only | `diff_surfel_res_3d` (gate OFF) |
| `mixed_3d` | JOINT | `ReLU(SV+0.5) + residual` | EWA SV-only | `diff_surfel_mixed_3d` |

- **Cascade** — single (`Σ T·α·feat` over all surfels), JOINT (one shared T for
  both halves), or DUAL (two independent T_tex / T_sv).
- **Bias gate ON** = kernel forces `sh_color = 0` for textured carriers →
  pure residual contribution. **OFF** = textured carriers contribute
  `ReLU(SV + sh_bias) + residual` (full per-Gauss capacity).
- Untex EWA carriers always render as a 3D ellipsoid (FastGS-verbatim
  `computeCov2D`, conic in screen-space) and contribute `ReLU(SV + sh_bias)`
  only — no hash query, no MLP.

---

## The staged curriculum: `--res_switch_iter` and `--res_3d_iter`

All four modes share two iter-gated events:

### Stage 1 — `--res_switch_iter` (default 10000)

Applies to: `res_switch`, `res_3d`, `res_3d_paired`, `res_3d_double`.

At this iteration the residual activation flips from **mode 0** to **mode 2**:

- **Mode 0** (pre-flip): kernel computes `feat = ReLU(ReLU(SV+0.5) + residual + res_bias)`.
  The outer ReLU clamps per-Gauss → strong locally-clamped supervision that
  helps multi-view geometry converge faster early on.
- **Mode 2** (post-flip): kernel computes `feat = ReLU(SV+0.5) + residual + res_bias`
  (signed, no per-Gauss outer ReLU). The renderer applies a per-pixel
  ReLU (or LeakyReLU with slope `--lru α`) **after the alpha blend** —
  signed residual gradient flows back to every contributor, giving the
  richer all-Gauss signal that fits fine detail better.

With `--lru > 0` on both phases the transition is a smooth knee instead of
a discontinuous jump (the leaky-grad slope at the clamp matches across the
flip).

### Stage 2 — `--res_3d_iter` (default 10000)

Applies to: `res_3d`, `res_3d_paired`, `res_3d_double`.

At this iteration every live Gauss **duplicates** into a paired set:

- **First N rows** — `_is_textured = True`. The "2D residual carrier".
  Stays a 2DGS surfel. Renders through the textured branch (hash query +
  MLP residual via the `--kernel` you trained with).
- **Second N rows** — `_is_textured = False`. The "3D EWA SV-carrier".
  Gets a learnable `_scaling_z` (3rd ellipsoid axis, initialised flat
  to `log(0.05) + min(log sx, log sy)`). Renders through the EWA branch:
  FastGS-verbatim `computeCov3D`/`computeCov2D` → screen-space conic, no
  hash/MLP, SV-only colour.

Adam is rebuilt for the doubled tensor. Densify accumulators are zeroed.

**Setting the two flags to the same value** (default `10000 == 10000`)
fires both stages on the same iteration — the original one-shot behavior.
**Setting `res_switch_iter < res_3d_iter`** (e.g. `10000 / 15000`) gives a
mid-training window of mode-2 single-cascade before the split, which lets
the SV colour stabilise on a fixed surfel set before doubling the parameter
count.

---

## `--method res_switch`

The simplest variant: **just the mode 0 → 2 flip**, no split.

```bash
--method res_switch \
--res_switch_iter 10000 \
--lru 0.01 \
--activation_bias 0.5 0.0
```

- Single cascade throughout. Kernel: `diff_surfel_3D_sh_res`.
- Pre-flip: per-Gauss outer ReLU + LRU (`feat = leaky_relu(ReLU(SV+0.5) + residual + res_bias, α)`).
- Post-flip: per-Gauss outer ReLU removed in kernel; renderer applies
  `LeakyReLU(image, α)` after the alpha blend.
- Auto-defaults `--lru` to 0.01 so the slope at the clamp matches across
  the flip.

Use when you want the mode-2 richer-gradient phase but **don't** need the
2D/3D split.

---

## `--method res_3d`

Two-stage curriculum with **dual T cascade** post-split. Pure separation
between tex (residual only) and untex (SV only) carriers.

```bash
--method res_3d \
--res_switch_iter 10000 --res_3d_iter 15000 \
--lru 0.01 \
--activation_bias 0.5 0.0
```

- Stage 1: flips to mode 2 + post-blend LeakyReLU.
- Stage 2: duplicates surfels. **Zeros SV/SH on tex rows** and **freezes
  their SV gradients** so tex carriers contribute residual only.
- Post-split renderer dispatches through `diff_surfel_res_3d` — a fork of
  `diff_surfel_mixed_3d` with a **single-pass dual-cascade** kernel:
  - Per-pixel, two independent transmittances `T_tex_aux` (decayed only by
    tex Gauss) and `T_sv_aux` (decayed only by untex Gauss) are tracked
    alongside the joint `T` used for depth/dist/normal aux outputs.
  - Color accumulators `C_tex_aux` (residual contribution) and `C_sv_aux`
    (untex SV contribution) develop independently — a wall of opaque tex
    carriers doesn't occlude untex Gauss behind it, and vice versa.
  - The kernel emits `out_color = C_sv_aux + C_tex_aux` (signed sum); the
    Python renderer applies `LeakyReLU(out_color, α)`.
  - Backward routes per-Gauss color / α grads through the cascade matching
    each Gauss's `is_textured` flag, using per-cascade reverse-T recovery
    and per-cascade `accum_rec`.
- Hard-wired per-Gauss bias gate in the `diff_surfel_res_3d` kernel forces
  `sh_color = 0` for textured carriers (matches the SV-zeroing at split).

Use when you want the cleanest "tex = high-frequency residual / untex =
smooth low-frequency SV" decomposition with independent cascades.

---

## `--method res_3d_paired`

Two-stage curriculum with **JOINT T cascade** post-split, tex carriers keep
both SV and residual (full mixed_3d-style per-Gauss capacity).

```bash
--method res_3d_paired \
--res_switch_iter 10000 --res_3d_iter 15000 \
--texsplit_tex_frac 0.5 \
--lru 0.01 \
--activation_bias 0.5 0.0
```

- Stage 1: flips to mode 2 + post-blend LeakyReLU.
- Stage 2: duplicates surfels. **Keeps SV/SH on tex rows** intact (no
  zeroing, no grad freeze). Opacity scaled at split via `--texsplit_tex_frac`:
  tex copy gets `α × tex_frac`, untex copy gets `α × (1 − tex_frac)`.
- Post-split renderer dispatches through `diff_surfel_mixed_3d` (joint
  cascade). Tex carriers contribute `T·α · (ReLU(SV+0.5) + residual)`;
  untex EWA carriers contribute `T·α · ReLU(SV+0.5)` to the same shared
  accumulator `C`. Image = `LeakyReLU(C, α)`.
- Mechanically equivalent to `--method mixed_3d --texsplit res_3d_iter`
  plus the mode-0 → mode-2 curriculum.

Use when you want the smoothness of a joint cascade (which keeps tex and
untex in lockstep visibility-wise) while still gaining the 3D EWA SV-carrier
half and the staged curriculum.

**Why joint instead of dual?** Empirically, the per-Gauss gradient through a
joint cascade is smaller (`T_joint` ≤ both `T_tex`/`T_sv`), which gentler
updates around a shared optimum. The dual cascade gives each half stronger
gradients but the optimization landscape has more equivalent minima
(`(C_tex, C_sv) = (target − x, x)` for any x sums to the same target), so
the optimizer can drift between them.

---

## `--method res_3d_double`

Two-stage curriculum with **DUAL T cascade** (like `res_3d`) but tex
carriers keep both SV and residual (like `res_3d_paired`).

```bash
--method res_3d_double \
--res_switch_iter 10000 --res_3d_iter 15000 \
--lru 0.01 \
--activation_bias 0.5 0.0
```

- Stage 1: flips to mode 2 + post-blend LeakyReLU.
- Stage 2: duplicates surfels. **Keeps SV/SH on tex rows** intact (same as
  `res_3d_paired`). Original α on both copies (independent cascades — no
  scaling needed mathematically).
- Post-split renderer dispatches through `diff_surfel_res_3d` (same kernel
  as plain `res_3d`).
- **At stage 2 the kernel-side bias gate is explicitly flipped OFF** via
  `set_textured_bias_gate(0)` — tex carriers contribute
  `ReLU(SV+0.5) + residual` (no `0` forcing on `sh_color`).
- Same dual-cascade T tracking and per-Gauss gradient routing as `res_3d`.

Use when you want the cross-set non-interference of the dual cascade
("textured wall doesn't occlude untex EWA Gaussians behind it") combined
with full mixed_3d-style per-Gauss capacity on the textured half.

---

## Flag reference

| Flag | Applies to | Default | Effect |
|---|---|---|---|
| `--res_switch_iter N` | `res_switch`, `res_3d`, `res_3d_paired`, `res_3d_double` | 10000 | Iteration of mode 0 → 2 flip + post-blend LeakyReLU enable |
| `--res_3d_iter N` | `res_3d`, `res_3d_paired`, `res_3d_double` | 10000 | Iteration of the 2D/3D surfel split |
| `--texsplit_tex_frac F` | `res_3d_paired`, `mixed_3d --texsplit` | 0.5 | Tex copy keeps `α × F`; untex copy keeps `α × (1 − F)`. Joint-cascade modes only. |
| `--lru α` | all | 0.0 | LeakyReLU slope. Auto-defaulted to 0.01 for res_switch/res_3d/res_3d_paired/res_3d_double. Applies pre-flip in the kernel and post-flip in Python. |
| `--activation_bias SH RES` | all | `0.5 0.0` | `sh_bias` (added inside `ReLU(SH + sh_bias)`) and `res_bias` (added to residual) |

---

## Composition formulas (per-pixel after all surfels)

### Single cascade (`res_switch` post-flip, baseline `3D_SH_res_sep`)

```
C = Σ_i T_i · α_i · (ReLU(SV_i + sh_bias) + residual_i + res_bias)
image = LeakyReLU(C, α=lru_slope)
```

### Joint cascade (`res_3d_paired`, `mixed_3d`)

```
For each Gauss j:
  if tex_j:   feat_j = ReLU(SV_j + sh_bias) + residual_j + res_bias
  else:       feat_j = ReLU(SV_j + sh_bias)              # untex EWA, no residual
  C   += T · α_j · feat_j                                 # shared accumulator
  T   *= (1 − α_j)                                        # shared transmittance
image = LeakyReLU(C, α=lru_slope)
```

### Dual cascade (`res_3d`, `res_3d_double`)

```
T_tex_aux = 1, T_sv_aux = 1, C_tex_aux = 0, C_sv_aux = 0

For each Gauss j in depth-sorted order:
  if tex_j:
    if res_3d (gate ON):   feat_j = residual_j + res_bias
    if res_3d_double (off): feat_j = ReLU(SV_j+sh_bias) + residual_j + res_bias
    C_tex_aux += T_tex_aux · α_j · feat_j
    T_tex_aux *= (1 − α_j)
  else (untex EWA):
    feat_j     = ReLU(SV_j + sh_bias)
    C_sv_aux  += T_sv_aux · α_j · feat_j
    T_sv_aux  *= (1 − α_j)

out_color = C_sv_aux + C_tex_aux
image     = LeakyReLU(out_color, α=lru_slope)
```

The dual cascade's pixel loop terminates only when **both** `T_tex_aux` and
`T_sv_aux` have saturated (< 1e-4). Joint `T` still tracks every Gauss for
depth / dist / normal aux outputs — those are joint-cascade quantities by
design.

---

## Backward gradient routing

For modes with `is_textured` populated (post-split `res_3d`, `res_3d_paired`,
`res_3d_double`):

- **Joint cascade** (`res_3d_paired`): standard `mixed_3d` backward — per-pixel
  reverse-T recovery on the single shared cascade, per-Gauss color / α / geom
  grads computed against it.
- **Dual cascade** (`res_3d`, `res_3d_double`): each cascade is reverse-
  recovered independently. Per Gauss `j` going in reverse:
  - If `tex_j`: `T_tex_back /= (1 − α_j)`, update `accum_rec_tex[ch]`, compute
    `dL/dfeat_j = α_j · T_tex_back · dL/dC[ch]`, α-grad uses
    `(feat_j − accum_rec_tex_post) · T_tex_back · dL/dC[ch]`.
  - If `!tex_j`: same with `T_sv_back`, `accum_rec_sv`.
- **Joint `T` reverse cascade is still tracked** for depth / dist / normal /
  mask α-grad contributions (those are joint outputs); the dual cascade only
  governs the color α-grad and the color partial gradients.
- For both flavours, `_features_dc` / `_sv_*` / `_features_rest` grads only
  flow on rows where they're not frozen. `res_3d` freezes them on tex rows
  post-split; `res_3d_paired` and `res_3d_double` keep them active.

---

## Baked rendering support

All four modes can be baked through `scripts/benchmark_baked.py`. The script
auto-detects `--method` and configures:

- `residual_mode = 2` in `bake_meta.json` (deferred per-pixel ReLU).
- `bake_meta.mixed_textured` / `mixed_untextured` counts (when `_is_textured`
  is populated post-split).
- Untextured surfels are folded into the bake-time **skip-texture** set
  (zero atlas rect) — at render time the bake kernel skips the atlas lookup
  for them and renders them SV-only at no atlas cost.
- Untextured EWA rows are rendered as 3D ellipsoids via the bake-render's
  EWA-aware preprocess (mirrors `diff_surfel_mixed_3d`'s training-time path).

The bake submodules used at render time:

| Mode | Bake render submodule |
|---|---|
| `res_switch`, `res_3d` | `diff_surfel_bake_render` |
| `res_3d_double` | `diff_surfel_bake_render` |
| `res_3d_paired` | `diff_surfel_bake_render_paired` (clone of `diff_surfel_bake_render`, identical behaviour) |

`benchmark_baked.py` aliases `diff_surfel_bake_render` → the paired build via
`sys.modules` when `args.method == "res_3d_paired"`, so all imports inside
the script transparently resolve to the paired build.

Example bake + benchmark on a 35k-iter `res_3d_paired` model at the
`mip_360/room` scene:

```
Gaussians: 102,226 (post bake-time prune)

Mode                       PSNR    SSIM   LPIPS    FPS
Neural renderer           31.28  0.9095  0.2183  120.81
Baked (SH only)           20.79  0.7446  0.3805 1289.4
Baked (SH + atlas)        31.22  0.9050  0.2371 1086.5

Bake quality loss:  −0.06 dB
Bake speedup:        9.0×
Atlas: 4096 × 46400 BC7, 181.2 MB
```

See `BAKED_RENDERING.md` for full bake pipeline details (atlas formats,
AABB modes, importance prune, BC7 / FP16 / uint8 dtypes).

---

## Choosing between modes

- **Need fastest convergence on smooth scenes, no behavioral departures from
  3D_SH_res:** `res_switch`. Just gets you the mode-2 phase without the
  parameter-doubling shock of the split.
- **Need clean separation between high-frequency residual and smooth SV
  colour, willing to accept ~halved per-Gauss capacity:** `res_3d`. Tex
  carriers carry residual only — strong specialisation, but each Gauss has
  less expressive power.
- **Want full per-Gauss capacity in the tex half AND smoothness from a joint
  cascade (closest to `mixed_3d` with a curriculum):** `res_3d_paired`.
  Empirically the most stable.
- **Want cross-set occlusion-independence (tex walls don't shadow untex)
  AND full per-Gauss capacity:** `res_3d_double`. The dual cascade gives
  each half independent T; capacity matches `res_3d_paired` per Gauss.

When trying these for the first time, start with the defaults
(`res_switch_iter=res_3d_iter=10000`) so both stages fire on the same iter
— that's the original one-shot behaviour. Then experiment with delaying
stage 2 (`--res_3d_iter 15000`) to give mode-2 single cascade some breathing
room before the parameter doubling.
