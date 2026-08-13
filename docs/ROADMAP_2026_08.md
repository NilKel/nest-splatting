# Roadmap — August 2026 (proberes · anti-popping · benchmarks)

Written 2026-08 as a compaction-survival handoff. Captures everything in flight
from the proberes bring-up + six-paper anti-popping review session.
Companion docs: [`PROBERES_WEBGPU_RENDERER.md`](PROBERES_WEBGPU_RENDERER.md),
[`FORWARD_PIPELINE_OVERVIEW.md`](FORWARD_PIPELINE_OVERVIEW.md),
[`QUEST_LOCAL_VIEWER_PLAN.md`](QUEST_LOCAL_VIEWER_PLAN.md).

---

## 0. Status snapshot (what is live)

**bitymi-demos deploys (all pushed, viewer bundle `index-2ca91777.js`):**

| card | bundle | notes |
|---|---|---|
| 🎯 Room · proberes | `mip_360/room_p32_oct.bitymi` (72 MB) | shared 8192² BC7 probe texture, NAT2 format 9 |
| 📦 Room · packed | `mip_360/room_packed[_astc].bitymi` (32 MB) | same checkpoint, shelf-packed typeD — the A/B partner |
| 🎯 Bicycle · proberes | `mip_360/bicycle_p32_oct.bitymi` (85 MB) | 161k probes, 94.7% atlas occupancy — locality stress test |
| Felix (personal) | `personal/felix[_astc].bitymi` (63 MB) | 25.94 dB @ 353 FPS baked |
| heart-slider | `heart-slider/` | camera-synced split-slider, clip1 vs clip2 |

**Key fixes landed this session (bitymi-demos commits):**
- proberes black screen root cause: exporter wrote `atlas_scale = span/255`;
  WebGPU unorm samples are [0,1] so scale must be the **full span**. Fixed in
  `scripts/export_proberes_bundle.py` (committed to nest-splatting too).
- Two real latent bugs found en route: detached-ArrayBuffer view in
  `Nat2Parser` probe branch; `TexParams` silently grown to 48 B by a
  `vec3<f32>` pad vs `buildStubAtlas`'s 32 B buffer.
- `ht=1` black seams: fp16 flush — `exp(-20)` < fp16 subnormal min; exponent
  clamp moved 20 → 10 (`render_2dgs.wgsl`).
- HT audit vs the HTGS paper (commit `b540768`): everything verified correct
  (Eq. 17 algebra, T_tail product form, mode-2 suffix factorization,
  back-to-front OVER order, identity-index sort skip, resize handling)
  EXCEPT one real bug, fixed: low-pass-dominated fragments keyed the HT core
  on a plane-**extrapolated** depth instead of CUDA's center-depth fallback
  (`depth = (rho3d <= rho2d) ? plane : Tw.z`) → thin structures vanished/
  shimmered under ht=1/2. Invisible in ht=0 (zv unused there).
- Probe affine restored to zero-storage-read varying path (A01/A10 ride
  `layer`/`_pad` via bitcast; single-layer so `layer ≡ 0`).

**proberes format facts (don't re-derive):**
- NAT2 `atlas_format` 9 = probe+BC7, 10 = probe+ASTC; rects block stride 6
  `[A00,A01,A10,A11,t0,t1]`, pre-divided by tex_res by the exporter.
- Probes must be fp32 (t-columns span 8192; fp16 ULP там = 8 texels).
- Quantization: FULL min/max wins over percentile clamp (measured: rmse
  0.00854 vs 0.01187@P99.9 — the clipped tail dominates). Room span 7.41,
  bicycle span 10.48 (rmse 0.01238 — 45% coarser, watch quality).
- CUDA low-pass branch samples the probe CENTRE (`uv=(0,0)` when rho3d>rho2d)
  — per-fragment select, NOT foldable into the affine.
- WGSL `mat2x2` is column-major: `mat2x2f(A00, A10, A01, A11)`.

**Benchmark corrections (measured, docs NOT yet updated — see §4):**
- FastGS was benched at wrong resolutions (their `train_base.sh` uses
  `-i images` → 1600-cap; only garden was images_4). Corrected sweep at NeST
  resolutions, FastGS's own `mult=0.5`, idle GPU:
  mean 1215 FPS (was 1183 in docs) → NeST advantage 1.23× → **1.20×**, 8/9.
- We did NOT train FastGS wrong — `train_base.sh` is their shipped recipe.
  Retrained treehill at images_4 anyway: 362,714 G (−8%), 22.875 dB (+0.03),
  1314 FPS → NeST/FastGS on treehill 0.86× → **0.84×**. Conditional
  "retrain all 9" evaluated FALSE; optional reviewer-proofing retrain
  (~15 min total, ~107 s/scene) offered but not decided.
- Overdraw re-measured at matched res + mult=0.5: FastGS mean 47.28 → 37.75
  → ratio vs NeST 2.26× → **1.84×** (treehill: 27.7 vs 38.4 = 1.39×).
- Checkpoint: `FastGS/output/treehill_images4/`. Sweep outputs:
  `speed_comparison/fastgs_correct_res/*.log`,
  `speed_comparison/fastgs_overdraw_correct_res/*.json`.

---

## 1. P0 — validations owed (cheap, do first)

1. **Numerical proberes validation (the skipped §6.5 step).** Diff a browser
   render of `room_p32_oct` against
   `/mnt/nilkel_hdd/outputs/mip_360/room/proberes/room8k_p32/final_test_renders/`.
   The bundle *renders* now, but transpose/v-flip/half-texel bugs are
   invisible to eyeballing. No headless WebGPU on the workstation → needs a
   manual screenshot at a known camera, or add a `?camera_idx=` exact-pose
   param + canvas download button to the viewer.
2. **Re-judge ht=1/ht=2 after the two HT fixes** (fp16 clamp + low-pass depth
   key). Both prior complaints (black seams, "ht=2 looks different but can't
   tell how") had implementation-bug components. Bicycle thin structures are
   the acid test. If ht=1 is now clean, several P1 items get cheaper.
3. **Proberes perf follow-ups:** (a) check the packed-vs-probe FPS gap with
   textures OFF (if untex also slower → it's the 217-surfel/geometry delta,
   not the atlas); (b) if atlas-on gap persists on bicycle >> room, locality
   confirmed → try `--probe_tex_res 4096` retrain (¼ working set).

---

## 2. P1 — anti-popping experiment ladder (from the six-paper review)

Papers reviewed: HTGS, StopThePop-adjacent, GRay, DP-GES, 3DGUT, Power Foam,
EVER, SortFreeGS. Conclusions worth keeping:

- **2D surfels are exempt from EVER's "sorting isn't enough" theorem** — ray
  ∩ flat disc is a point, not an interval; exact per-pixel ordering IS exact
  rendering for us. Popping is purely an ordering artifact.
- Ray tracing (GRay/3DGRT) is the wrong regime: cost tracks bounding-volume
  hits ∝ primitive SIZE; we have few LARGE splats (anti-DI). 3DGRT sparse-init
  (0.11M, our profile) = 68–96 FPS on 4090 vs our ~900–1000.
- 3DGUT's anti-pop ingredient is just a k-buffer (its unsorted variant pops);
  UT itself is for distorted cameras — file under XR lens-space, not popping.
- WSR/SortFreeGS leakage = relational visibility approximated by unary
  weights; our flat-top beta_scaled kernels + near-opaque bakes are unusually
  WSR-friendly (no Gaussian-tail silhouette halo).
- HTGS "in full" = K=16 in-raster insertion sort: impossible in WebGPU (no
  ROV), economics inverted on CUDA (our sort is ~free at 123k prims; we'd
  forfeit early-out). Its own hardware-viewer suggestion = depth peeling =
  DP-GES architecture.

Ladder (stop when popping is acceptably gone):

1. **[done-ish]** ht=1 with fixes — re-judge (P0.2).
2. **τ_K winner-id fix** (only if faint-splat core hijack visible): core pass
   writes winner gauss_id to a second target; tail tests "not winner" instead
   of strictly-behind. Also fixes the coplanar-tie-drop edge case (matters
   for texsplit mixed bundles).
3. **DP-GES 3-layer peeling behind `?peel=1`** — best mobile path for opaque
   scenes: 3 peel passes + order-independent tail, deletes the radix sort,
   plain raster, no atomics. 2 layers leak, 3 suffice, 4 wastes 40% (their
   ablation). A/B vs ht=1 on room_packed.
4. **WSR mode in viewer** (~day): delete sort dispatches, additive blend +
   normalization subpass, weights = kernel α × per-surfel SV-opacity.
   16-bit accumulation (fp16 lesson already paid). Mobile FPS win even if
   quality needs the P2 field.
5. **K=16 core in CUDA `renderBakedCUDA`** — ONLY for the semi-transparent
   medical scenes (brain/heart), where peeling fails and K=1 is weakest.
   Serial per-pixel loop makes it a contained change; needs per-pixel core
   ids stored for backward if ever trained (~65 MB @ 1M px).
6. **Training-side (independent):** GRay scale decay (×0.999875/iter) +
   accumulated-weight pruning. Double duty: shrinks popping magnitude AND is
   the treehill-bloat fix predicted to lift it 1100 → 1500–1800 FPS
   (`speed_comparison/MIP360_BOTTLENECK_ANALYSIS.md` §6 — note its "drop 3.7%
   clears FastGS" claim is now false vs corrected 1286→1314 FPS baselines).

---

## 3. P2 — transmittance-field distillation (the novel one, publishable)

User's idea + distillation recipe. Core insight: for a static scene,
visibility of a surface point is a deterministic function `V(x, ω)`; a
**per-texel directional field** is its correct parameterization (unlike
per-splat unary weights — WSR's failure). Classical precedent: PRT / baked SH
visibility. Only a textured-surfel pipeline can store this cheaply — per-splat
methods have nowhere to put it.

Free lunch inventory: the atlas alpha channel is shipped as constant 255
today (`rgba[...,3]=255` in the exporter) — a per-texel scalar costs zero
storage, zero bandwidth, one shader line. SV machinery reusable as the
directional basis (SortFreeGS uses SH-opacity the same way). CUDA
atlas-backward already exists (probe `dL_dtex`, GEStex atlas grads).

Recipe:
1. Instrument the sorted CUDA forward to dump `(surfel, texel uv, ω, α·T)`
   over training views — T is already computed per fragment.
2. Fit factorized `ŵ(x, ω) = a_tex(x) · uᵢ(ω)` (alpha channel × per-surfel
   SV-opacity) to the dumped α·T.
3. Sort-free render `C = Σ ŵᵢcᵢ` ≈ exact sorted blending; leakage becomes
   field approximation error, not operator error.
4. **Composite variant (predicted winner):** keep ht=1's exact K=1 core,
   replace the heuristic `exp(−k·Δz)` tail weight with the distilled field.
   Sort-free, exact frontmost, learned everything-behind, zero new fetches.

Known limit: angular sharpness (K=7 SV sites are smooth) → soft occlusion at
grazing edges; the K=1 core covers the sharpest term geometrically.

Finetuning verdicts (from the same discussion): finetune > from-scratch for
both HT and WSR; use the `res_switch`-style operator-flip machinery; HT
finetune is a small calibration (operator ≈ identical for opaque bakes); WSR
finetune = freeze geometry, learn vᵢ + SV-opacity.

---

## 4. P3 — write corrected benchmarks into the docs

Stale docs carrying pre-correction numbers:
- `docs/BITYMI_RESULTS_3D_SH_RES.md` — FastGS column (1183 → 1215 corrected; mean
  advantage 1.23× → 1.20×; treehill 1221 → 1286, and 1314 for the
  images_4-retrained checkpoint).
- `speed_comparison/MIP360_BOTTLENECK_ANALYSIS.md` — §1 table + §6's
  "axis ≤ 0.5 filter clears FastGS" conclusion (now false).
- `speed_comparison/intersection_comparison/stats_table.md` — FastGS overdraw
  measured at wrong res AND script-default mult=1.0 (their real default is
  0.5); corrected mean 37.75, ratio 1.84×.

Also decide: reviewer-proofing retrain of all 9 FastGS scenes at images_4
(recommended: yes; ~15 min; expect FastGS mean +2–3%, NeST still 8/9).

---

## 5. P4 — GEStex rebuild, DP-GES-informed (bigger arc)

User's observed failure (hardened textured surfels lose detail) = GES's
documented structural flaw; DP-GES §1 describes it verbatim. Three causes:
blend-encoded detail collapse, winner-take-all gradients, Eulerian texture
misalignment during hardening.

Plan (user's curriculum + DP-GES amendments):
- **Diagnostic first** (1 hr): decomposition renders of the failed GEStex run
  — which cause dominates? If texture capacity, no curriculum fixes it alone.
- **Phase A:** opaque UNTEXTURED base from a converged 3D_SH_res checkpoint
  (`--init_ply`, never from scratch — concavity-filling is an exploration
  risk, not a preservation risk). Soft-rim anneal (DP-GES `α=min(1,w·G)`,
  w≈30) instead of fully-hard discs; scale reg (same lever as treehill).
- **Phase B:** texture on frozen geometry, 3-layer peeled transmittance
  weighting + their `Lt`; teacher-depth supervision from the semi-transparent
  model as belt-and-suspenders against bridging.
- **Phase C (per scene):** sort-free detail Gaussians with DP-GES
  transmittance modulation (replaces GEStex's LRU mix — restores the gradient
  path their Fig. 11 shows is worth ~0.3 dB).
- **Move the occlusion cull to after hardening** — our 15k `T·o` cull deletes
  exactly the understudy layers DP-GES keeps alive; plausible contributor to
  the observed detail loss.
- Endpoint: sort-free, pop-free base layer; typeD/proberes/CONIC/viewer stack
  preserved. Medical volumetric scenes stay on the sorted/HT path (opaque
  base is the wrong tool there — DP-GES's own stated limitation).

Long-term research thread (parked): Power Foam's pop-free-by-theorem
partition + our textures = surfels as dipole faces of a bounded power
diagram. Paper-scale.

---

## 6. Backlog / small items

- `bicycle_packed` A/B partner card (offered, not requested).
- proberes for more scenes once room/bicycle validate.
- HT known deviations documented in commit `b540768`: no τ_K, coplanar-tie
  drop, 0.99 cap (consistent with CUDA), late-Z ⇒ HT shades ~2× fragments.
- Quest/Mac: `Halloumi-Stream` M3 (real splat renderer in `mac/src/render.rs`)
  is the streaming path; `QUEST_LOCAL_VIEWER_PLAN.md` has the full map.
- Prof reply material lives in `FORWARD_PIPELINE_OVERVIEW.md` (oriented
  texture domain §7, CUDA-vs-pixel-shader texture units §8, atlas cost §9).
- treehill_images4 FastGS checkpoint kept at `FastGS/output/treehill_images4`.
