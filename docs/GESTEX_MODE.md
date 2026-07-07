# `--method GEStex` — GES-style bi-scale with a baked, fine-tunable texture atlas

> ⚠️ **SUPERSEDED — see [`GESTEX_PIPELINE.md`](GESTEX_PIPELINE.md) for the current pipeline.**
> This document describes the earlier design (discrete `surfel_opac` 1→30→60→90→255 ramp,
> bake-centric flow). The current pipeline replaced that with the **Triangle Splatting+
> hardening** — a rising opacity **floor** (opacity ≤ 1, which fixed the surfel *bloat*), a
> `beta_scaled` **β-ceiling anneal** to flat-top discs, **two-phase pruning with growth left
> on**, a mid-harden shrink, local→global LRU at 10k, SV colour on both surfels and 3DGS, and
> a `--ges_no_bake` (live hash/MLP) option. Read `GESTEX_PIPELINE.md` first; the sections
> below remain accurate only for the atlas/opacity *mechanics* they describe.

> **FINAL ARCHITECTURE (implemented + verified).** The joint stage renders with the
> **GES sort-free 2-pass** (`diff_surfel_gestex_joint_s` surfel z-buffer + `diff_surfel_gestex_joint_g`
> additive Gaussians), composited in Python. The flow is the clean GES-faithful one:
> **no 10k prune** (all surfels kept through hardening), and at 20k a single all-views pass
> does the **occlusion cull + error-map Gaussian init**, then bake + switch to sort-free +
> **standard densify/prune on the Gaussians**. Both kernels are FD-gradcheck-verified
> (joint_s: 6/6 atlas + 4/4 SV; joint_g: 5/5 SH). Tests: `scripts/test_joint_s.py`,
> `scripts/test_joint_g.py`. The res_3d_paired-cascade path (below, historical) remains as
> a fallback when the joint kernels aren't built. The one deferred refinement is the
> approx-z-buffer α-blend for 10k–20k (frontmost-first); it needs gating the *shared*
> `diff_surfel_3D_sh_res` kernel, so it's left out — near-opaque surfels make the standard
> tile-blend a good stand-in there.
>
> **Final schedule:** 0–10k normal 3D_SH_res → 10k local→global LRU + harden `w`
> (30→60@18k→90@19k) + freeze w + densify off (**no prune**) → 20k occlusion-cull + error-map
> Gaussian init + bake atlas + `w=255` + freeze surfel geom/opacity + **sort-free 2-pass** +
> re-enable standard densify (opacity-reset off; surfels opacity-pinned to 0.99 so they're
> densify-inert) → 20k+ fine-tune atlas + surfel SV + Gaussians.
>
> **New sort-free files:** `submodules/diff_surfel_gestex_joint_s` (atlas+SV z-buffer,
> `set_gestex_atlas` device-global, backward scatters to SV colors_precomp + atlas texels; no
> geometry VJP — surfels frozen), `submodules/diff_surfel_gestex_joint_g` (verbatim GES
> additive-depth-test). Renderer: `_render_gestex_joint()` gated on `ingp.is_gestex_sortfree`.



GEStex adapts the GES paper ("When Gaussian Meets Surfel", `../GES`) onto the nest
hash+MLP+SV residual pipeline. A scene is a **bi-scale** representation: opaque **2D
textured surfels** carry coarse geometry + appearance, and a few **3D Gaussians**
supplement fine detail. Our twist over vanilla GES/BITYMI: keep the hashgrid+MLP
residual for the early phases, then **bake it into an explicit per-surfel RGB texture
atlas at the joint transition and fine-tune that atlas** (dropping the hashgrid/MLP cost
for the final training segment).

## Three-stage schedule

GEStex is aliased to `res_switch` internally for phases 0–20k (so it inherits every
3D_SH_res-family gate + the mode 0→2 flip machinery), with a separate `args.is_gestex`
flag driving the GEStex-specific events. `ingp.is_gestex_joint` flips at the joint
transition and the renderer intercepts before the res_switch path.

| Stage | Iters (default) | Primitives | Rasterizer | Residual | Opacity |
|---|---|---|---|---|---|
| **explore** | 0 – `ges_phase1_iter` (10k) | 2D surfels (`_is_textured=True`) | `diff_surfel_3D_sh_res`, kernel=gaussian, mode 0 | hash+MLP | `w·exp(-r²/2)`, trainable |
| **harden** | 10k – `ges_joint_iter` (20k) | 2D surfels, near-opaque | same, mode 2 (flip) + LRU | hash+MLP | `w` frozen, ramp 30→60→90 |
| **joint** | `ges_joint_iter` (20k) – end | textured surfels (frozen geom) **+** untextured 3D Gaussians | **`diff_surfel_gestex`** | **baked RGB atlas** (surfels) / none (Gaussians) | surfels `w=255`; Gaussians trainable |

### Gated events (train.py)
- **`ges_phase1_iter` (10k)** — prune surfels with activated opacity `< ges_prune_w_thresh`
  (0.8), **saving their world positions** as spawn seeds; ramp `surfel_opac`→30; freeze
  the opacity LR; disable FastGS densification; flip residual mode 0→2 (post-blend LRU, via
  the aliased `res_switch` machinery).
- **`ges_occlusion_iter` (15k)** — GES occlusion cull: accumulate per-surfel frontmost-pixel
  counts across ALL train views (uses `render_pkg['max_contrib_idx']`), take the max over
  views, prune surfels below `ges_occlusion_thresh` (16; auto→4 if `synthetic` in path).
- **`ges_opac60_iter` / `ges_opac90_iter` (18k/19k)** — ramp `surfel_opac`→60→90.
- **`ges_joint_iter` (20k) — bake & splat:**
  1. `surfel_opac`→255 (opaque discs).
  2. **Bake** the hashgrid+MLP residual at each surfel's R×R UV lattice → `_tex_atlas`
     `[N,R,R,3]` (unbounded RGB, view-independent), a new trainable optimizer leaf.
     (`gaussian_renderer.ges_bake_atlas`, recipe byte-faithful to `benchmark_baked.py`.)
  3. **Spawn** untextured 3D Gaussians at the saved seeds (`_is_textured=False`, real
     `_scaling_z`); every per-Gauss tensor grows (spawned rows inherit from a random
     surviving surfel then geometry/opacity/atlas are overridden).
  4. Rebuild the Gaussian optimizer; freeze the hashgrid/MLP (INGP optimizer LR→0);
     install the setter-mirror to `diff_surfel_gestex` + re-fire MLP-weights/mode/bias/lru.
  5. Route rendering through `diff_surfel_gestex`.
- **`> ges_joint_iter`** — every `ges_gs_prune_interval` (500) prune untextured Gaussians
  with `α < ges_gs_prune_thresh` (0.02); surfels are kept. Surfel geometry
  (xyz/scale/rot/opacity) grad-masked to 0 each step; atlas + surfel SV (SH) +
  Gaussians (pos/scale/rot/opacity/SH) train.

## The `diff_surfel_gestex` rasterizer (joint stage)

A clone of `diff_surfel_res_3d_paired` (textured 2D surfels + untextured EWA 3D
Gaussians, joint cascade, full backward), with the **textured residual swapped from
hash+MLP to a bilinear lookup into the baked per-surfel RGB atlas.**

- **Atlas wiring** mirrors the device-global MLP-weight pattern (`set_mlp_weights`/
  `get_mlp_grads`): `set_gestex_atlas(atlas, atlas_grad, R, uv_extent)` points the
  forward-read at the atlas values and the backward-scatter at a grad buffer (Python owns
  it; the renderer stashes it on `pc._ges_atlas_grad` and train.py assigns it to
  `_tex_atlas.grad` post-`backward()`). `clear_gestex_atlas()` disables.
- **Forward** (textured, atlas active): `residual = bilinear(atlas[gid], s.x, s.y)` at the
  ray-disc UV — **hash query + MLP skipped entirely.** `feat = act(ReLU(SV+0.5) + residual)`.
- **Backward**: `dL/dresidual` scattered into the 4 atlas texels (bilinear weights) via
  `atomicAdd`; **`mlp_backward` + hash-backward skipped** (no MLP weights read). Surfel SV
  (SH) still gets its gradient; surfel geometry is frozen Python-side.
- **Forced scalar backward** when the atlas is active (`h_gestex_atlas_active`): the
  collab-GEMM backward (enabled internally via the `0x100` bit) has no atlas hooks and the
  atlas case has no MLP GEMM, so we disable it and use the scalar path where the hooks live.
- Untextured Gaussians render as EWA 3D ellipsoids (unchanged from `res_3d_paired`).

> Note: the joint stage currently uses the res_3d_paired **joint alpha-blend cascade**
> (opaque surfels ≈ frontmost dominant), not the GES sort-free 2-pass z-buffer. That is a
> valid, fully-differentiable training path; the sort-free 2-pass is an inference-speed
> optimization (future work).

## Flags
`--method GEStex` plus: `--ges_phase1_iter` (10000), `--ges_occlusion_iter` (15000),
`--ges_occlusion_thresh` (16), `--ges_opac60_iter` (18000), `--ges_opac90_iter` (19000),
`--ges_joint_iter` (20000), `--ges_prune_w_thresh` (0.8), `--ges_atlas_res` (8),
`--ges_gs_prune_interval` (500), `--ges_gs_prune_thresh` (0.02), `--ges_s_weight` (1.0),
`--ges_gs_add_start/end/num` (error-map spawn window, num=0 = off). GEStex force-overrides
`--kernel` → `gaussian` (GES falloff) and defaults `--lru` to 0.01. Composes with
`--feature SV`, `--hybrid_levels`, `--disable_c2f false`, `--cold`, `--fastgs`,
`--aabb accutile`, `--random_background`.

Example (himalaya, adapt iters for a full run):
```bash
python train.py -s .../brain/... -m gestex_run --yaml ./configs/himalaya_2d.yaml --eval \
  --iterations 35000 --method GEStex --hybrid_levels 2 --disable_c2f false --aabb accutile \
  --feature SV --cold --fastgs --fastgs_densify_until 20000 --grads abs \
  --w_lambda 0.005 --w_lambda_gamma 50 --lowpass --random_background
```

## Model state (`scene/gaussian_model.py`)
- `_tex_atlas` `[N,R,R,3]` — trainable atlas leaf (created at joint). Threaded through
  optimizer / densify / prune / PLY (`tex_atlas_*` columns) like `_scaling_z`.
- Reuses `_is_textured` (True=surfel, False=Gaussian) and `_scaling_z` (untextured EWA axis).
- `_opacity` = GES `w` (sigmoid activation kept; `surfel_opac` global multiplier applied to
  textured rows in the renderer). Helpers: `ges_prune_low_opacity`, `ges_enter_joint_stage`,
  `ges_prune_gaussians`, `ges_mod_depth`.

## Tests
`scripts/test_gestex_units.py [textured|untextured|mixed|gradcheck]` — tiny hand-built
surfel sets through `render()`, fw+bw per case, plus a finite-difference **gradcheck of
the atlas bilinear backward** (verified 6/6 texels within 0.4% of FD). One case per process
(a CUDA illegal access poisons the context). Debug env vars: `GESTEX_NOATLAS=1` (disable
atlas), `GESTEX_USE_PAIRED=1` (route joint through res_3d_paired), `GES_ATLAS_FILL=<v>`
(diagnostic constant atlas).

## Pitfalls (all bitten during bring-up)
- **gestex MLP weights**: the module has its OWN device-global MLP weights; without the
  setter-mirror (installed at the joint transition) they're NULL and the backward null-reads
  them. The atlas path skips the MLP so it doesn't strictly need them, but the mirror keeps
  the module consistent.
- **collab vs scalar backward**: atlas hooks are scalar-only; the collab GEMM must be forced
  off when the atlas is active or the atlas gets no gradient.
- **spawn must grow EVERY per-Gauss tensor** (incl. `_sv_*`) — else `--feature SV` size-
  mismatches at the next render.
- **`ges_bake_atlas` must deep-copy the MLP** (`nn.Module.half()` is in-place; mutating
  `ingp.mlp_fused` breaks the renderer's later `set_mlp_weights`).
