# `--method GEStex` — Current Pipeline Reference

> **Authoritative current-state doc.** This supersedes the design-history in
> [`GESTEX_MODE.md`](GESTEX_MODE.md) (which describes the earlier `surfel_opac`-ramp /
> bake-centric flow). GEStex is a GES-style **bi-scale** representation: **flat, opaque
> 2D surfels** (coarse geometry + appearance) **+ 3D Gaussians in front** (fine detail),
> rendered **sort-free** in two passes. Appearance per surfel = **SV (spherical-Voronoi,
> view-dependent) base + a hash/MLP residual** that is either kept live or baked into a
> per-surfel RGB atlas. The hardening strategy is adapted from **Triangle Splatting+**
> (rising opacity floor + shape anneal + two-phase pruning), which removed the surfel
> "bloat" that the earlier `surfel_opac>1` multiplier caused.

---

## 1. Three-stage schedule

| Stage | Iters (default) | Primitives | Rasterizer | Opacity | Kernel shape | LRU |
|---|---|---|---|---|---|---|
| **Explore** | 0 – `ges_phase1_iter` (10k) | 2D surfels | `diff_surfel_3D_sh_res_harden` (res_switch), mode 0 | free (optimized), **reset ON** | β free (~4, soft) | **local** per-Gauss |
| **Harden** | 10k – `ges_joint_iter` (20k) | 2D surfels → flat+opaque | same clone, mode 2; **first-intersection sort @`ges_first_int_iter` (15k)** | **rising floor** `O_t + (1−O_t)·sigmoid(w)` ≤ 1, trainable, **reset OFF** | **β ceiling** anneal → ~0.1 (flat-top) | **global** post-blend |
| **Joint (sort-free)** | 20k – end | frozen opaque surfels **+** untextured 3D Gaussians | `diff_surfel_gestex_joint_s` + `_joint_g` | surfels floor≈0.99 (frozen); Gaussians trainable | β frozen (flat) | global, **after mixing** |

GEStex is aliased to `res_switch` for phases 0–20k (inherits all 3D_SH_res-family gates +
the mode 0→2 flip); `args.is_gestex` drives the GEStex-specific events, and
`ingp.is_gestex_joint` (flipped at 20k) routes the sort-free renderer.

**CUDA isolation rule (0–20k rasterizer).** The explore+harden phase does NOT run on the
shared `diff_surfel_3D_sh_res` — it runs on an **isolated clone**,
`submodules/diff_surfel_3D_sh_res_harden`, so GEStex-specific rasterizer changes never touch
the shared main path (used by plain `3D_SH_res` + the baked pipeline). The renderer routes
GEStex 0–20k there via `_is_gestex_harden(ingp)` (→ `_rmod` + `_sh_res_setter_mod`), and
train.py points `_SHRES_SETTER_MOD` at it (so the per-render `set_mlp_weights`, all the
`set_*` device-globals, AND the 20k joint-transition MLP-weight/grad handoff all read/write
the module that actually trained the MLP). The clone name **keeps the `diff_surfel_3D_sh_res`
prefix on purpose** — the renderer gates kwargs on module-name substrings, so a name
containing `diff_surfel_gestex`/`_mixed`/`_res_3d` would wrongly get `is_textured`/`scaling_z`
passed → forward TypeError.

**Frontmost-first promotion (@ `ges_first_int_iter`, default 15k) — GES-paper-exact.** Fixes
the surfel *smearing* seen in the surfel-only harden: the 2DGS tile blend orders each tile's
splats by **Gaussian-center depth** (`depths[idx] = p_view.z`), so a *tilted* surfel whose
center is nearer but whose *ray intersection* is farther wrongly blends in front and blots out
the textured parts of surfels behind it. From 15k the clone applies the GES §render scheme:
after the (unchanged) center-depth tile sort, **each pixel's frontmost surfel — by exact
per-pixel ray-disc intersection depth — is promoted to blend FIRST; "the blending order of the
other surfels is not adjusted."** Converges exactly to the joint stage's `joint_s` z-buffer as
surfels go opaque → consistent train/render across the 20k transition.
- **Mechanism — GES-LITERAL multi-pass** (byte-faithful to `ges_rasterization_surfel`,
  verified against their code). Forward = 3 sweeps over the tile range through the
  UNCHANGED loop body: pass 0 = **full-range frontmost scan** (min exact intersection
  depth, **no transmittance early-exit** — catches frontmost surfels buried deep in the
  sort order); pass 1 = F ONLY, blends **first** at `T=1`; pass 2 = everyone except F, in
  tile order, early-exit on the promoted T. Every kernel type / lowpass / AA / hash+MLP
  path works for F by construction (same code). Stores `fm_pos` (F's 1-based contributor
  index) per pixel.
- **Backward — GES's deferred-`minGeo` pattern** via a 2-phase reverse walk through the
  UNCHANGED per-contributor code: phase 0 = all non-F (skip F: no T division/folds — the
  un-divided `(1−αF)` promotes everything in front automatically); phase 1 = F alone at
  the walk's end, where `T/(1−αF)=1` and the walk-final recurrences are exactly the non-F
  composites. Handles `F > last_contributor` (F past pass-2 termination) like GES's
  exemption. Collab-GEMM restricted to phase 0; F's MLP backward runs the scalar branch.
  **Verified**: collab ≡ scalar on every gradient (rel ≤ 1e-4) promotion on+off; FD
  gradcheck ON matches the OFF control (`scripts/test_frontmost_collab.py`).
- Aux channels (depth/normal/dist) blend in **promoted order** (consistent with color;
  GES's surfel kernel has no aux at all — ours flows F's aux grads through the same code).
- **Toggle:** `set_frontmost_first` (fwd+bwd device globals); renderer flips it per-render
  from `ingp.frontmost_on` (`--ges_frontmost_iter`, default 18k — near-opaque regime only,
  matching the paper's `w≥30` precondition; `-1` disables). Off ⇒ byte-identical.
- **Tile-depth sort** (`--ges_first_int_iter`, default 10k; independent + composable):
  key each (Gauss,tile) on the intersection depth at the tile-center ray
  (`duplicateKeysWithTileDepth` behind `set_tile_depth_sort`). **Exact alpha blending at
  any opacity** — the right mechanism for the translucent early-harden. Rect-only → the
  renderer drops the AccuTile cull (`aabb_mode 5 → 3`) while active.

---

## 2. Opacity model — rising floor (the bloat fix)

**Root cause of the old bloat:** the geometry gradient is `dL_dG = opacity · dL_dalpha`
(`backward.cu`), i.e. **∝ opacity**. The old `surfel_opac` multiplier drove opacity to
**255**, amplifying the scale/position gradient ~255× and (with the residual free to paint
overspill background-coloured) letting surfels balloon while the texture hid them.

**Fix (TS+):** keep opacity **≤ 1** with a rising floor:
```
surfel_opacity = O_t + (1 − O_t) · sigmoid(w)          # ∈ [O_t, 1], never > 1
```
- `O_t = pc.ges_opac_floor`, ramped **0 → `--ges_opac_floor_max` (0.99)** linearly over
  `[phase1, joint)`, then pinned at the joint.
- **Opacity stays trainable** (optimized within `[O_t, 1]`) — *not* frozen.
- **Opacity reset is turned OFF at 10k** (a reset knocking opacity to ~0.01 fights the floor).
- Applied to surfel rows only in both the main renderer and `_render_gestex_joint`
  (`joint_s` opacities); Gaussians keep their trained opacity.

`get_opacity` itself is unchanged (`sigmoid`); the floor is applied in the renderer.

---

## 3. Kernel / shape model — `beta_scaled` + β ceiling anneal

- `--kernel beta_scaled` (allowed for GEStex alongside `gaussian`/`beta`; anything else is
  forced to `gaussian`). Compact support: `alpha_beta = (1 − ρ3d/9)^β`, culled at `ρ3d ≥ 9`
  (**hard 3σ boundary** — footprint can't expand with opacity, unlike a Gaussian tail).
- `β = sigmoid(_shape)·5 ∈ [0,5]`: **β→0 = flat-top disc**, β~4 = Gaussian-like, β=5 = peaked.
- **β ceiling anneal** (harden): β is capped at a ceiling annealed from its 10k value down to
  `--ges_beta_end` (0.1); `_shape.data.clamp_(max=logit(β_max/5))` each iter. The optimizer is
  free *below* the ceiling → near-flat-top opaque discs by the joint.
- When beta-family is active, `--kernel2` is set to `gaussian` so the untextured 3D Gaussians
  render as **Gaussian EWA ellipsoids** (the cascade decodes `kernel2` via `render_mode` bits).

---

## 4. Colour model — SV base + residual/atlas (both primitives)

`--feature SV` (spherical Voronoi, view-dependent). Per-primitive colour:
```
surfel:   C_S = LRU_base(SV + texture)      texture = live hash/MLP residual OR baked atlas
gaussian: C_G = SV                          (untextured EWA, additive)
```
- **Both surfels and the untextured Gaussians use SV.** In `_render_gestex_joint`,
  `_build_fake_shs_from_SV` produces `(fake_shs, sv_rgb)`; surfels get `sv_rgb` as
  `colors_precomp`, and the Gaussians get **`fake_shs[gm]`** as their `joint_g` SH so their
  `_sv_*` params receive gradient. *(Bug fixed: `joint_g` previously used raw `get_features`,
  leaving the Gaussians' SV rendering-inert → they never learned colour.)*
- **Double-bias fixed:** `sv_rgb` is already `relu(feat + 0.5)`; the old surfel path added a
  second `+0.5` → a ≥0.5 brightness floor (washed-out) and a 20k colour jump. Removed.

---

## 5. LRU (leaky-ReLU) sites

- **0 → 10k:** LOCAL per-Gauss LRU (mode 0) — `feat = LRU(SV + residual)` clamped per surfel
  before blending.
- **10k → end:** GLOBAL (mode 2). Flip iter = `--ges_global_lru_iter` (default `−1 → phase1
  = 10k`), so the whole harden trains under the global activation the joint uses (one
  transition at the explore→harden boundary; no mid-harden shock).
- **Joint composite:** the LRU is applied **once, AFTER** mixing surfels + sort-free 3DGS —
  NOT per-pass:
  ```python
  final = (C_S · s_w + C_G) / (s_w + W_G)      # mix hardened surfels + 3DGS
  final = leaky_relu(final, lru)                # single post-mix rectification
  ```
- `--lru` auto-defaults to 0.01.

---

## 6. Pruning & densification schedule

| Phase | Surfel growth | Surfel pruning | Gaussian growth/prune |
|---|---|---|---|
| Explore (0–10k) | full clone/split | standard + opacity reset | — |
| Harden (10k–20k) | **still ON** (refills holes) | **two-phase** (below) | — |
| Joint (20k+) | **STOPPED** (`ges_freeze_textured_densify`) | frozen (pinned) | standard densify/prune + periodic `T·o` prune |

**Two-phase surfel pruning (harden):**
1. **Early hard opacity prune** — once at `phase1 + ges_hard_prune_offset` (500), remove
   `get_opacity < ges_prune_w_thresh` (**0.2**, was 0.8) *while the floor is still low* so
   opacity is a meaningful junk signal. Gentle → no holes.
2. **Periodic occlusion (`T·o`-proxy) cull** — every `ges_surfel_prune_interval` (2500), an
   all-views pass counts each surfel's frontmost-pixel hits (`max_contrib_idx`) and prunes
   those below `ges_occlusion_thresh` (16; auto→4 on synthetic). Removes occluded/redundant
   surfels as they harden.

Densification stays on through harden so any hole raises local error → clone/split refills it
(the earlier disaster was 0.8-threshold prune **with growth off** → permanent holes).

**Textured-growth freeze at 20k:** `densify_and_{clone,split}` AND their candidate mask with
`~_is_textured` when `ges_freeze_textured_densify` is set — closing the gap where
screenspace-grad-driven clone would keep growing *frozen* surfels (their means2D grad isn't
masked). Only the untextured 3D Gaussians grow post-20k.

**Mid-harden shrink:** at `ges_shrink_iter` (15k), surfel scale `×ges_shrink_factor` (0.75,
log-space `+= log(factor)`) so the still-trainable geometry re-fits the right size under
(near-)full opacity over the remaining harden iters.

---

## 7. Joint transition (20k) — occlusion cull + spawn + bake/no-bake

One all-views pass at `ges_joint_iter`:
1. **Occlusion cull** surfels (frontmost-count `< thresh`).
2. **Error-map 3DGS spawn:** per view, `err = Σ(render − gt)²`; importance-sample pixels by
   the error CDF (`searchsorted(cumsum(err/Σ), rand)`); backproject via `depths_to_points`;
   spawn **untextured EWA 3D Gaussians** at those seeds (`_is_textured=False`, real
   `_scaling_z`, opacity 0.1, SH inherited from a random surviving surfel). Count =
   `--ges_gs_add_num` or `max(10000, Nsurf//4)`.
3. **Bake** the hash/MLP residual → per-surfel RGB atlas `_tex_atlas [N,R,R,3]`
   (`--ges_atlas_res` 8), a trainable leaf — **OR `--ges_no_bake`**: skip the atlas, keep
   the **live hash/MLP** as the textured residual (routed through the `diff_surfel_gestex`
   cascade; INGP keeps training, `get_mlp_grads` pulled from the cascade module).
4. Freeze surfel geometry (grad-mask xyz/scale/rot/opacity), stop textured densify, re-enable
   standard densify/prune **on the Gaussians**, disable opacity reset, pin surfel opacity.

---

## 8. Sort-free render (`_render_gestex_joint`, gated on `ingp.is_gestex_sortfree`)

- **Pass 1 — `diff_surfel_gestex_joint_s`** (surfel z-buffer): frontmost opaque surfel per
  pixel, `C_S = colors_precomp(SV) + bilinear(atlas)`; outputs `D_S` (depth threshold) and,
  now, the **frontmost surfel view-space normal** (see §9). Opacity = the rising-floor value.
- **Pass 2 — `diff_surfel_gestex_joint_g`** (additive 3DGS): untextured EWA Gaussians, SH =
  `fake_shs[gm]` (SV), **depth-tested `depth ≥ D_S`**, → `C_G`, `W_G`, per-Gauss `max_contrib`.
- **Composite:** `final = LRU((C_S·s_w + C_G)/(s_w + W_G))`, `s_w = --ges_s_weight` (1.0).

The cascade path (`diff_surfel_gestex`, res_3d_paired-based) is used for `--ges_no_bake` and
as a fallback when the joint kernels aren't built; it renders textured surfels via live
hash/MLP (atlas off) + untextured EWA.

---

## 9. Rasterizer submodules & key CUDA

| Submodule | Role |
|---|---|
| `diff_surfel_3D_sh_res` | explore + harden (res_switch alias); `dL_dG = opacity·dL_dalpha` (the ∝opacity gradient). |
| `diff_surfel_gestex_joint_s` | surfel z-buffer + atlas. `out_others` now **8 channels**: `[0]`depth, `[1]`depth+modDepth, `[2]`frontmost idx, `[3]`u, `[4]`v, **`[5..7]` frontmost view-space normal**. Atlas via `set_gestex_atlas`; backward scatters into atlas texels + `colors_precomp` (no geometry VJP — surfels frozen). |
| `diff_surfel_gestex_joint_g` | additive depth-tested SH Gaussians (GES-verbatim). |
| `diff_surfel_gestex` | cascade (textured hash/MLP + untextured EWA) for `--ges_no_bake` / fallback. Has `d_gestex_frontmost` flag plumbing (frontmost-first two-pass **paused/unbuilt**). |

`joint_s` gradcheck: **6/6 atlas + 4/4 SV** (still passing after the 8-channel change; the
normal is forward-only, backward untouched).

---

## 10. Flags

**Schedule:** `--ges_phase1_iter` (10000), `--ges_joint_iter` (20000),
`--ges_global_lru_iter` (−1 → phase1), `--ges_shrink_iter` (15000),
`--ges_first_int_iter` (15000; −1 disables) — iter to switch the harden to
first-intersection (tile-depth) sort on the `diff_surfel_3D_sh_res_harden` clone.

**Opacity/shape:** `--ges_opac_floor_max` (0.99), `--ges_beta_end` (0.1),
`--ges_shrink_factor` (0.75).

**Pruning:** `--ges_prune_w_thresh` (0.2), `--ges_hard_prune_offset` (500),
`--ges_surfel_prune_interval` (2500), `--ges_occlusion_thresh` (16 / 4 synthetic),
`--ges_gs_prune_interval` (500), `--ges_gs_prune_thresh` (0.02).

**Appearance/bake:** `--ges_no_bake`, `--ges_atlas_res` (8), `--ges_gs_add_start/end/num`,
`--ges_s_weight` (1.0).

**Legacy (unused by the TS+ hardening):** `--ges_opac60_iter`, `--ges_opac90_iter`.

Composes with: `--feature SV`, `--kernel beta_scaled`, `--hybrid_levels`, `--cold`,
`--disable_c2f false`, `--aabb accutile`, `--random_background`. GEStex forces the kernel to a
beta-family/gaussian falloff and defaults `--lru` to 0.01.

---

## 11. training_output & diagnostics

**Joint-stage decomposition** (`ingp.is_gestex_joint`), per periodic save:
- `{iter}.png` / `{iter}_full.png` — full composite
- `{iter}_surfel_only.png` — C_S (SV + texture)
- `{iter}_surfel_sh_only.png` — surfel SV base (atlas zeroed) *("SH only" = the per-surfel
  feature, SV here)*
- `{iter}_surfel_texture_only.png` — texture, clamped `[0,1]`
- `{iter}_surfel_texture_signed.png` — texture signed (gray=0, bright=+, dark=−subtractive)
- `{iter}_gaussian_sh_only.png` — untextured 3DGS pass (C_G)
- `{iter}_normal.png` — real (`rend_normal` = frontmost surfel view-space normal rotated
  view→world; `surf_normal` = `depth_to_normal`)

During harden (pre-20k) the standard `sh_only`/`tex_only`/`sh_plus_tex` decomposition applies.

**Transition diagnostics** at 20k → `<model>/ges_debug/`: `A_before_{full,SVonly,residual}` (harden,
live hash+MLP) and `B_after_{full,SVonly,residual}` + `C_after_noatlas` (joint) + baked-atlas
value stats. **`scripts/ges_decompose.py`** dumps a labelled per-view decomposition montage
from any checkpoint.

---

## 12. Key insights / fixes this pipeline encodes

- **Bloat = opacity>1 amplifying `dL_dscale ∝ opacity`.** Rising floor (opacity ≤ 1) removes
  the driver at the root; `beta_scaled` compact support bounds the footprint; local→global LRU
  and pruning are secondary. (β=90 vs 255 barely changes the *saturation* radius — that's
  logarithmic — but the *gradient magnitude* is linear in opacity, so the ≤1 cap is what matters.)
- **Gentle, correctly-timed pruning + growth-on = no holes** (0.2 opacity while floor low, then
  `T·o`; never the 0.8-with-growth-off disaster).
- **Appearance can hide bad geometry:** a strong hash/MLP+SV residual paints correct colour over
  misaligned/oversized discs, so geometry needs explicit pressure — hence the rising floor,
  β anneal, `T·o` pruning, mid-harden shrink, and (user-tuned) `lambda_normal`.
- **Normal consistency** is active during harden (`lambda_normal`, `lambda_dist` nonzero ⇒
  `skip_aux_normal_dist` stays off); the joint stage now emits real normals for viz.

## 13. Deferred / not yet wired

- **Frontmost-first approximate z-buffer for harden** (GES `ges_rasterization_surfel` two-pass):
  `d_gestex_frontmost` flag + setter-mirror scaffolding exist in `diff_surfel_gestex` but the
  two-pass forward/backward is **not implemented/built**; harden currently blends tile-order.
- **Ongoing error-map re-spawn** (`--ges_gs_add_*`) window is a flag but not looped.
- Baked path is EWA/atlas-verified; `--ges_no_bake` (live hash/MLP cascade) is the current
  default for iterating on the surfel/texture/Gaussian interplay.

---

## 14. Example command

```bash
python train.py -s /path/to/nerf_synthetic/chair/ -m chair_GEStex \
  --yaml ./configs/nerfsyn.yaml --iterations 35000 --method GEStex --eval \
  --hybrid_levels 2 --disable_c2f false --aabb accutile --feature SV --cold \
  --kernel beta_scaled --ges_no_bake
```
Explore→harden→sort-free with: local→global LRU @10k, opacity floor 0→0.99 + β ceiling →0.1
over 10k–20k, hard prune @10.5k + periodic occlusion culls, shrink @15k, 3DGS spawn + freeze
@20k, live hash/MLP textures (no bake).
