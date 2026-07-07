# GES frontmost-first 2-pass — implementation + debug handoff

Self-contained reference for debugging the GES-style frontmost-first promotion in
`submodules/diff_surfel_3D_sh_res_harden`. Written for an agent with no prior context.

## What is being implemented, and why

`--method GEStex` (see `docs/GESTEX_PIPELINE.md`) hardens flat 2D textured surfels
toward opacity 1 between iters 10k–20k (a rising opacity **floor**
`O_t + (1−O_t)·sigmoid(w)`, `O_t: 0→0.99`). The standard 2DGS tile blend orders each
tile's surfels by **Gaussian-center depth**; a tilted surfel whose center is nearer but
whose actual ray intersection is farther wrongly blends in front (smearing). The GES
paper ("When Gaussian Meets Surfel", reference code `/home/nilkel/Projects/GES`) fixes
this during their surfel stage (`submodules/ges_rasterization_surfel`, always-on 10k–20k,
where their `w≥30` makes surfels near-opaque):

> after the tile-based sorting, for each pixel we compute the accurate pixel-level
> depths of the surfels covering it. The surfel with the minimum depth is selected for
> the FIRST blending computation, while the blending order of the other surfels is not
> adjusted.

We replicate this **byte-faithfully** in our harden rasterizer. Enabled from
`--ges_frontmost_iter` (default 15000; −1 = off) via `ingp.frontmost_on` →
renderer (`gaussian_renderer/__init__.py`, search `frontmost_on`) →
`set_frontmost_first(bool)` → device globals `d_frontmost_first` (forward.cu) and
`d_frontmost_first_bw` (backward.cu). **Off ⇒ byte-identical to the base module**
(`submodules/diff_surfel_3D_sh_res`, which must NEVER be edited — clone-isolation rule).

## The implementation (all in `submodules/diff_surfel_3D_sh_res_harden/`)

Design principle: NO duplicated math. Both kernels re-run their UNCHANGED loop bodies
in extra sweeps, with tiny per-pass gates. The clone's whole delta vs the base module is
this feature (+ the off-by-default tile-depth sort, `set_tile_depth_sort`, independent):
`diff -u` against `submodules/diff_surfel_3D_sh_res/cuda_rasterizer/` shows every line.

### Forward — `cuda_rasterizer/forward.cu`, kernel `renderCUDAsurfelForward`

A pass loop wraps the entire rounds/batch loop:
`for (int fm_pass = (d_frontmost_first != 0 ? 0 : 2); fm_pass < 3; fm_pass++)`,
with per-pass reinit of `toDo/contributor/done` (T, C, and aux accumulators
intentionally carry across passes 1→2). Per-pixel state: `fm_depth` (init 1e30),
`fm_contrib` (1-based contributor index of the frontmost F; 0 = none).

- **Pass 0** (frontmost scan): gate placed AFTER the `alpha < 1/255` check —
  `if (fm_pass == 0) { if (depth < fm_depth) {fm_depth=depth; fm_contrib=contributor;}
  continue; }`. Full range, NO transmittance early-exit, no side effects. `depth` is the
  exact per-pixel ray-disc intersection (`(s.x*Tw.x + s.y*Tw.y) + Tw.z`, low-pass
  fallback `Tw.z` when rho2d wins) — same formula GES pass 1 uses.
- **Pass 1** (F only): `if (fm_pass == 1 && contributor != fm_contrib) continue;` at the
  body top. F runs the unchanged body at `T=1` → blends FIRST; after it `T = 1−αF`.
  **CRITICAL (was the NaN bug):** pass 1 must NOT publish `last_contributor` — the tail
  of the blend body has `if (fm_pass != 1) last_contributor = contributor;`.
- **Pass 2** (the rest): `if (fm_pass == 2 && fm_contrib > 0 && contributor == fm_contrib)
  continue;`. Normal blending on the promoted T; early exit `test_T < 1e-4` unchanged.
- Output: `fm_pos[pix_id] = fm_contrib` (an `ImageState` uint32 buffer;
  `rasterizer_impl.{h,cu}` — allocated next to `n_contrib`. A float3 `c_nf` buffer is
  also allocated but **dead** — leftover from a superseded design; ignore).
- Aux channels (depth/normal/dist/median) blend in promoted order (GES's kernel has no
  aux at all; ours flows them through the same unchanged code).

### Backward — `cuda_rasterizer/backward.cu`, kernel `renderCUDAsurfelBackward`

GES's deferred-`minGeo` pattern (their backward.cu:255–450) as a 2-phase reverse walk:
`for (int fm_phase = 0; fm_phase < (d_frontmost_first_bw != 0 ? 2 : 1); fm_phase++)`
around the rounds loop, `fm_contrib` loaded per pixel from `fm_pos`.

- **Phase 0** (all non-F): unchanged walk, plus (a) std-branch loop-top skips F
  (`contributor == fm_contrib−1`, 0-based after the `contributor--`); (b) collab-GEMM
  branch and the tile-level batch early-out are gated to `fm_phase == 0`; (c) collab sets
  `participates = false` for F. Skipping F means its `T = T/(1−α)` division and all
  `accum_*`/`last_*` folds never happen — the un-divided `(1−αF)` factor automatically
  applies the promotion scaling to every surfel in front of F.
- **Phase 1** (F only): re-inits `toDo/contributor/done`, re-walks all batches; std-branch
  loop-top in phase 1: `if (fm_contrib == 0 || contributor != fm_contrib−1) {underflow
  guard; continue;}` — NOTE it skips the `contributor >= last_contributor` gate, because
  F may legitimately exceed `last_contributor` (F always blends in pass 1 even when
  pass 2 terminates early / blends nothing). F then runs the UNCHANGED per-contributor
  code at the walk's end, where `T/(1−αF)` = 1 exactly and the walk-final `accum_rec`
  (+ aux recurrences) are the full non-F composites. F's MLP backward runs the scalar
  branch (collab is phase-0-only); weight grads land in the same global `dL_dmlp_W*`.

### Invariants any debugger should assert

1. **Blended-set consistency**: the set of contributors whose `(1−α)` the backward
   divides out of T (phase 0 non-F, then F in phase 1) must EXACTLY equal the set the
   forward blended (pass 1 F + pass 2 survivors). After phase 0, `T == (1−αF)`
   (± FP); after F's division, `T == 1`. Any mismatch compounds ×100 per near-opaque
   contributor (α clamps at 0.99 → 1−α = 0.01) → inf → NaN.
2. `n_contrib[pix]` = last PASS-2 blended index (0 if pass 2 blended nothing) — never F's.
3. `fm_contrib` 1-based (forward `contributor++` at body top); backward std compares
   `fm_contrib−1` post-decrement; collab compares `current_contributor = contributor−j−1`.
4. Promotion OFF ⇒ single pass 2, `fm_contrib = 0`, all gates no-ops ⇒ byte-identical.
5. `alpha` recomputation in backward must bit-match the forward (same gates:
   p.z, near_n, power/support, `alpha ≥ 1/255`) — the walk relies on re-derivation.

## Bugs found so far (fixed — regression context)

1. **Frozen MLP** (routing, train.py): `get_mlp_grads` read from the base module while
   rendering through the clone → MLP silently froze. Fix: gestex branch in the dispatch
   + setter mirror. Symptom: quality loss from iter ~500.
2. **`c_nf` factor** (superseded design): the pre-literal implementation stored a non-F
   composite with a wrong `(1−αF)` factor → biased `dL/dαF`. Gone (literal rewrite).
3. **β edge-gradient pole** (band-aid, kept): `beta_scaled` flat-top `dG/dρ ∝
   base^(β−1)` diverges for β<1; clamped `fmaxf(base, 1e-3)` in `dG_factor` (2 sites).
4. **THE NaN** (the big one): pass 1 published `last_contributor = fm_contrib`; when
   near-opaque alphas (exactly 0.99) made pass 2's FIRST candidate trip `test_T < 1e-4`
   (0.01·0.01 < 1e-4), pass 2 blended NOTHING and `n_contrib` claimed `[0, fm_contrib)`
   as blended → backward phase 0 divided T for phantom contributors → ×100 each → inf →
   NaN in geometry/SH grads first (SV-colored pixels), loss NaN a few hundred iters
   later. Fired at 18k–19.5k as the opacity floor saturated. Fix: the `if (fm_pass != 1)`
   guard (forward.cu, blend-body tail).

## Post-fix session findings (2026-07-06 debug session)

1. **`nan_dump_19404` is a STALE-BINARY artifact, not a second bug.** The bug-4 fix
   was edited into forward.cu at 18:57:08; the rebuild's `.so` landed at **19:01:25**;
   the run that produced nan_dump_19404 started at **19:01:09** (args.json mtime) and
   imported the PRE-fix `.so` (pip swaps the `.so` in only at the very end of the
   build). Its signature is identical to 19002 (fused-MLP weights NaN, hash/geometry
   finite). Lesson: after a CUDA fix, check `.so` mtime > run start before trusting
   the rerun.
2. **The verification tooling toggled the WRONG FLAG.** The renderer has two
   independent hooks: `ingp.first_int_sort` → `set_tile_depth_sort` (tile-depth SORT)
   and `ingp.frontmost_on` → `set_frontmost_first` (the 2-pass PROMOTION).
   `test_frontmost_collab.py` set `first_int_sort` — every "promotion" leg actually
   tested the sort with the promotion OFF. Fixed (now sets `frontmost_on`). Any
   pre-fix "promotion verified" claim below refers to the sort, not the promotion.
3. With the correct flag, on the fixed build, all of the following PASS:
   collab==scalar equivalence (both promo states, images bit-identical, grads rel
   ≤ ~1e-4); the near-opaque stress case (below); FD gradcheck noise identical
   promotion-ON vs OFF.
4. GStex (`../GStex`) has NO 2-pass frontmost mechanism (single pass, median-depth
   `depth_mode==3`) — the only reference is GES. Our pass-0 scan was re-verified
   gate-for-gate against GES pass 1 (same gate set incl. low-pass `Tw.z` depth
   fallback, no transmittance early-exit; GES's extra `G < 1/255` gate is redundant
   for our opa ≤ 1).

## Verification tooling

- `scripts/test_frontmost_collab.py` — collab-GEMM vs forced-scalar backward
  equivalence through the REAL `render()` mode-5 pipeline, promotion on/off (4 legs,
  separate processes — `DISABLE_COLLABORATIVE_GEMM` is read once per process):
  `python scripts/test_frontmost_collab.py run {on|off} {collab|scalar}` (from repo
  root!), then `... compare`. All grads must match to rel ≤ ~1e-4; images bit-identical.
- `scripts/test_frontmost_nearopaque.py` — the previously-missing near-opaque case:
  α at the 0.99 clamp (`_opacity=8`), `beta_scaled` flat-top (β≈0.10, the dG-pole
  regime), tilted overlapping surfels with interleaved depths (F deep in tile order),
  plus an isolated surfel (pass-2-blends-nothing / `n_contrib==0` pixels). Same
  run/compare CLI as above + a `gradcheck {on|off}` mode (judge ON against the OFF
  control — flat-top edge FD noise is large but must MATCH between the two).
- Level-0 FD gradcheck (colors_precomp, no hash/MLP): scratchpad `gradcheck_fm.py`
  (session-local; recreate from `test_frontmost_collab.py`'s scene if lost). Judge
  promotion-ON against the OFF control (FD noise at splat edges is ~0.05 relmax).
- **NaN tripwire** (train.py, after `total_loss.backward()`, GEStex ≥15k): aborts on the
  first non-finite loss OR per-tensor grad, dumping `nan_dump_<iter>/{point_cloud.ply,
  state.pt}` into the run dir. An actual dump from the fixed bug exists at
  `outputs/nerf_synthetic/chair/GEStex/chair_GEStex_2pass15k2lprect/nan_dump_19002/` —
  replayable single-step repro state.
- Repro command (NaN window was 18k–19.5k):
  `python train.py -s /home/nilkel/Projects/data/nerf_synthetic/chair -m <out> --yaml
  ./configs/nerfsyn.yaml --iterations 35000 --method GEStex --eval --hybrid_levels 2
  --disable_c2f false --aabb rect --lowpass --feature SV --cold --kernel beta_scaled
  --ges_no_bake`
- Rebuild after any `.cu/.h` edit:
  `conda run -n nest_splatting python -m pip install -e
  submodules/diff_surfel_3D_sh_res_harden --no-build-isolation` (2–5 min; background it).

## Reference implementations to diff against

- GES forward two-pass: `/home/nilkel/Projects/GES/submodules/ges_rasterization_surfel/cuda_rasterizer/forward.cu:319–448`
  (pass 1 min-depth scan — note NO transmittance gate, `alpha<1/255 || G<1/255` only;
  then F blended first; pass 2 `if(contributor == closeGeoIdx) continue;`).
- GES backward deferred-F: same submodule `backward.cu:205–450` (`minGeoIdx` from
  `n_contrib[pix + H*W]`; walk exempts F from the last_contributor gate, stashes its
  data, processes it after the loop with `T/(1−αF)` and walk-final `accum_rec`).
- Our base (promotion-free) module: `submodules/diff_surfel_3D_sh_res` — the clone must
  be byte-identical to it when the feature is off.

## Environment / schedule context a debugger needs

- Harden regime when bugs bite: opacity floor 0.8–0.99 (α clamps at exactly
  `min(0.99f, ...)` → 1−α = 0.01), `beta_scaled` β annealing down (`--ges_beta_end`),
  mask loss active (`dL_daccum` ≠ 0 via ALPHA_OFFSET), optionally `--lowpass` (0x400:
  rho2d→transMat backward), `--aabb rect` (mode 2: fixed 4σ, lossless; accutile mode 5's
  ellipse cull lacks pixel-space low-pass dilation — a known, separate approximation).
- The WMMA collab FORWARD is disabled in this fork (`if (false && ...)`) — forward is
  always scalar; the backward has BOTH std-scalar and collab-GEMM (0x100) paths and every
  fix must land in both (historic pitfall).
- train.py per-iter driver: search `ges_frontmost_iter` / `first_int_sort` (~line 2604).
