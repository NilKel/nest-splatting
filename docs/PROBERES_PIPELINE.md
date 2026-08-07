# PROBERES — probe-mapped shared-texture residual (`--method proberes`)

**Status (2026-07-30):** teacher-baked pipeline validated on chair — 36.8 dB train-view
(held-out `--eval` run in flight). From-scratch training works but underperforms; the
teacher-baked path is the one to use.

## 1. What this is

3D_SH_res replaces its per-fragment **hash+MLP residual** with a lookup into **ONE shared
2D texture image** via per-surfel affine probes:

```
fragment:  ray-splat intersection -> uv (surfel sigma units)
           tc = A_i · uv + t_i                    (probe_i = [A00,A01,A10,A11,tx,ty], texture px)
           residual = bilinear(tex[R,R,3], tc)    (signed, unbounded)
           color = ReLU( ReLU(SV + 0.5) + residual )     # mode-0 cascade, unchanged
```

Motivation: the baked per-Gaussian atlas is huge (638 MB BC7 on mip360); a shared 2048²
image is ~12 MB u8 + 6 floats/surfel. At inference this is the baked-renderer cost
profile (no hash, no MLP: one affine + one bilinear per fragment).

This is structurally **Texture-GS** (Xu et al. 2024) applied to our 2DGS stack: their
per-Gaussian `(phi(mu), Jacobian)` Taylor probes == our `[N,6]` affine probes; their
texture == our image. Key lesson adopted from them: **placement first, content second,
and content must be written through the same map it is read with.**

## 2. The winning pipeline (teacher-baked)

Requires a trained 3D_SH_res-family checkpoint (geometry + SV + hash/MLP teacher).

```bash
# Stage 1 — OPTIONAL. Analytic anisotropic oct probes match the best learned phi
#           (35.22 vs 35.22/35.26) with NO training, so prefer skipping straight to
#           stage 2 with --oct. Keep phi only if you need a learned parameterization.
#           Use --w_cov 0: the coverage loss is monotonically HARMFUL (see §4).
python scripts/probe_uv_field.py train -m <ckpt> --steps 8000 --tex_res 2048 --w_cov 0 --out_tag uvD

# Stage 2 — SCATTER bake: teacher residuals written into the atlas THROUGH THE PROBES
#           (never use the phi_inv bake for content — see §4). Sample grid is
#           Nyquist-sized per surfel by default; add --oct for analytic placement.
python scripts/probe_uv_field.py scatter -m <ckpt> --tex_res 2048 --out_tag uvD

# Gate — atlas@probe-centers vs teacher@surfel-centers; proceed only if cosine is high
python scripts/verify_uv_bake.py -m <ckpt> --out_tag uvD

# Stage 3 — finetune the IMAGE by render loss against real GT
python train.py --method proberes -s <scene> -m <run> --yaml <cfg> --cold --eval --iterations 15000 \
  --init_ply <ckpt>/point_cloud/iteration_N/point_cloud.ply \
  --probe_init_dir <ckpt>/uvD \
  --probe_no_field --densify_until_iter 0 --probe_pixel_decay 0 --probe_tex_res 2048 \
  <ALL of the checkpoint's render flags: --feature SV --kernel beta_scaled --lowpass --aabb ... --hybrid_levels N --disable_c2f>
```

With `--probe_no_field` the texture **is** the pixel image (plain `[R,R,3]` parameter);
nothing is evaluated per iteration — `dL_dtex` flows straight into the leaf. The 2D
hash+MLP field exists only for from-scratch runs (§6).

### Hard constraints (each cost a debugging session)

- **`--densify_until_iter 0` is mandatory** with `--probe_init_dir`: probes are fixed per
  surfel; N changes trigger a nearest-center remap that points probes at content baked
  for other surfels (sweep: 13.6 dB vs 36.8 with densify off).
- **Mirror the checkpoint's render flags.** `--init_ply` is an *initializer*: tensors come
  from the PLY (incl. beta `shape` and `sv_site_*/sv_col_*` — but SV columns load only if
  `feature_mode` is 'SV' *before* `load_ply`, i.e. pass `--feature SV`); the model
  *configuration* comes from your CLI. Wrong kernel = shrunken faint surfels ("fragmented").
- **No `--probe_nosh_lambda` with a baked init** (−2.5 dB): it demands ReLU(residual)
  alone reproduce full GT on top of a complete SV base → over-exposure. It is the right
  tool ONLY from scratch (it produced the first real textures in that regime).
- Freezing SV (`--freeze_sh`), geometry, or everything (`--probe_tex_only`) makes **no
  measurable difference** (36.82–36.84 across the sweep) — don't bother outside ablations.

## 3. Files / components

| piece | where |
|---|---|
| CUDA rasterizer | `submodules/diff_surfel_3D_sh_res_probe` — isolated clone of `diff_surfel_3D_sh_res`; probe branch under `render_mode = 5 \| 0x1000`; probes/tex/dims ride the unused `features_diffuse`/`gridrange_diffuse`/`offsets_diffuse` slots; grads via `dL_dfeatures_diffuse` (=dL/dprobes) + new `dL_dtex` return; collab-GEMM + backward smem-MLP preload force-skipped (null MLP → IMA otherwise); `dL_duv = Aᵀ·dL_dtc` joins the existing `dL_ds` geometry chain |
| Python modules | `hash_encoder/probe_modules.py` — `ProbeHead3D` (from-scratch probe predictor; oct base + gauge-cancelled θ + metric scale; running `log_smed`), `ProbeTexField2D` (pixels + optional 2D hash+MLP field, `_SparseBake` recompute backward) |
| phi stage + bakes | `scripts/probe_uv_field.py` (train / bake / scatter / `--oct` / `--out_tag`; per-1k-step render checks of test view 0: tex / sv / full) |
| acceptance gate | `scripts/verify_uv_bake.py` — cosine(atlas@probe-centers, teacher@surfel-centers) |
| FD gradcheck | `scripts/test_proberes_units.py` — probe/tex/uv/geometry chains vs central FD (kink-bracketed); geometry FD needs probes FROZEN |
| dumps | every `save_interval`: `{it}_sv/_probe_tex/_probe_tex_abs/_probe_atlas/_probe_atlas_probes.png` + `[PROBE]` patch-size stats; every 1k: `{it}_testview0(.png/_tex.png)`. Judge runs by these, **not** `{it}.png` (training buffer with bg compositing) |

Found & fixed along the way: the **shared base rasterizer under-scales beta/beta_scaled
positional gradients by exactly 2×** (missing d(rho3d)/ds factor in `dG_factor`, both std
and collab paths). Fixed **only in the probe clone**; base + all forks still carry it —
see memory `beta-kernel-half-positional-grad-bug`.

## 4. Why scatter-bake (the critical design fact)

`T[p] = teacher(phi_inv(p))` requires `phi_inv(phi(c)) ≈ c` to **sub-surfel-σ** accuracy,
because the residual decorrelates within 1–2σ. No achievable phi meets that (3d-cycle
0.03 ≈ 3–4σ). Measured atlas-vs-teacher cosine: phi_inv bake **0.06–0.36** (garbage →
over-exposed renders); **scatter through the probes 0.70–0.74** (read = write map, cycle
error irrelevant to values). The 2D-cycle loss is nearly irrelevant to render quality; the
3D cycle only matters for the (abandoned) phi_inv bake.

Two independent error sources remain, measured on chair (`verify_uv_bake.py` cosine):

| bake | atlas coverage | cosine (grid 24) | cosine (Nyquist) |
|---|---|---|---|
| phi, `--w_cov 3.0` (`uvD_cov3`) | 12.9% | 0.7046 | 0.7170 |
| analytic oct (`uvOCT`) | 39–42% | 0.7254 | **0.7446** |

1. **Collision averaging** (dominant). Probes cover 4–42% of the atlas, so texels are
   shared — ~50 surfels averaged per used texel at phi's 12.9%, ~6 at oct's 40%. Atlas
   std at probe centers is 0.080 vs the teacher's 0.139. The lever is `--oct` placement;
   **`--w_cov` is NOT a lever — it is monotonically harmful** (matched sweep, steps 8000 /
   seed 0 / grid-24 bake / learnable probes):

   | `--w_cov` | 0 | 1 | 2 | 3 |
   |---|---|---|---|---|
   | bake cosine | 0.6832 | 0.6939 | 0.6522 | 0.6296 |
   | test PSNR | **35.22** | 35.18 | 34.93 | 34.89 |

   The old `--w_cov 3.0` recommendation came from comparing `lc_learn` (34.91) to
   `eval_phi_learn` (35.26) — which differed in **steps AND seed**, not just `w_cov`.
   Retrained at matched settings, `w_cov 0` reaches 35.22 and the effect inverts. Seed
   alone is worth ~0.3 dB here, so treat any single-seed phi comparison with suspicion.
2. **Sampling aliasing** (secondary, now fixed). The scatter grid must resolve the finest
   hash level over the surfel's ±3σ span; chair needs p50=25.2, p90=67.5, p99=122.8
   samples/axis, so the old fixed `grid=24` band-limited only the smaller half of surfels.
   Now Nyquist-sized per surfel (§5). The fix helps oct ~1.5× more than phi, exactly
   because oct's weaker collisions leave room for it to show.

3. **Probe anisotropy** (fixed 2026-07-31). `oct_probes` used a similarity transform
   `A = rho·R` with one isotropic rho from `sqrt(su*sv)`, but 64% of chair surfels have
   aspect ratio > 2 and 34% > 4. Now `A = R(-phi_g)·diag(rho_u, rho_v)` with `rho_u ∝ su`,
   `rho_v ∝ sv` (`--oct_iso` reverts). **Total texel demand is unchanged** —
   `sum (p·su/s)(p·sv/s) == sum (p·sqrt(su·sv)/s)^2` — so this is a free shape fix:

   | oct probes | fixed | learn |
   |---|---|---|
   | isotropic | 34.79 | 34.92 |
   | **anisotropic** | 34.87 | **35.22** |

   phi probes always had this (`A = J_phi·[axis_u|axis_v]`); only the oct path discarded it.

**Analytic oct now matches learned phi** (35.22 vs 35.22/35.26), both above the 35.07
teacher — so the phi stage is optional. **Bake cosine is a WITHIN-family metric only**: it
tracks PSNR across `--w_cov` values (table above) but inverts across families (oct vs phi,
Nyquist vs grid-24, aniso vs iso all have the better-cosine bake losing). Judge any
cross-family change by a finetuned `--eval` run, never by the acceptance gate alone.

## 5. Flags (training)

`--probe_init_dir` (baked atlas+fixed probes) · `--probe_no_field` (texture = pixel image)
· `--probe_tex_res` 2048 · `--probe_pixel_decay` (set **0** with baked init; default 1e-4
counters Adam eps=1e-15 random-walk on sparse texel grads) · `--probe_pixel_lr_scale` 0.1
· `--probe_bake_interval` (field re-eval cadence when the field is on) ·
`--probe_tex_only` / `--probe_freeze_head` (ablations; freeze_head is a no-op with
init_dir) · `--probe_nosh_lambda` (from-scratch only) · from-scratch extras:
`--probe_patch_px` 12 (packing break-even ≈ tex_res/√N), `--probe_tex_levels/base/hidden`,
`--probe_field_lr_scale` (≤2–3; 5 collapses geometry via dL/duv), `--probe_abs_placement`,
`--probe_smed_freeze_iter`, `--probe_c2f_interval` (off under repo-default `--disable_c2f`).

### phi / scatter stage (`scripts/probe_uv_field.py`)

`--out_tag` (lets variants coexist under `<ckpt>/`) · `--steps` · `--tex_res` 2048 ·
`--patch_px` 12 · `--seed` · `--w_cov` (coverage; high seed variance — see §4) · `--oct`
(analytic octahedral placement, phi bypassed) · **`--scatter_grid`** (0 = default =
per-surfel Nyquist vs the finest hash level; `>0` forces a uniform grid — the pre-2026-07-31
behavior was 24) · `--scatter_grid_max` 128 / `--scatter_grid_min` 8 (Nyquist clamp; 128
leaves ~1.4% of chair surfels clamped, ~148M teacher samples).

The adaptive grid **requires** the per-sample area weight `dA = 36/(gx·gy)` applied in
`scatter_bake` — without it a 128×128 surfel outvotes an 8×8 one 256:1 wherever they
collide, purely from sampling density. It is a constant (hence a no-op) in the uniform path.
Sampling density changes only the **prefilter**, never stored detail: samples still land in
a `patch_px`-sized footprint of the shared `tex_res²` atlas.

## 6. From-scratch mode (historical / fallback)

Without `--probe_init_dir`: probes come from `ProbeHead3D` (learned, analytic oct base),
texture from `pixels + hash2D field`. It trains (FD-verified) but textures emerge weakly:
SV+densification absorbs the error first, and per-surfel private patches split each
surface point's gradient across ~8 atlas locations (no pooling). `--probe_nosh_lambda 1`
forces textures to appear, at a noise cost. Every bring-up lesson (init pitfalls: exact-zero
last layers block upstream grads; stale `s_med` collapsing patches to ~1 texel; pixel
random-walk; LR blow-ups) is recorded in memory `proberes-probe-texture-mode`.

## 7. Open items

- **Coverage/collisions** — spread probes past ~9% texel utilization (stronger `--w_cov`,
  oct placement, or higher `--tex_res` as slack).
- **Held-out numbers** — `sweep_shfrozen_eval` in flight; compare against
  `checkp30k/test_metrics.txt` (same geometry/SV, teacher vs probe-texture residual).
- **CONIC fast path** — compose ray-splat and probe into ONE per-surfel screen→texture
  affine in preprocess; low-pass via cov2D dilation (+eps2d·I) instead of min(rho3d,rho2d);
  backward = EWA VJP port. Design: memory `proberes-conic-fast-path-design`.
- **Probe refinement** — IMPLEMENTED: `--probe_learn_lr 0.005-0.02` makes the loaded
  probes learnable (dL/dprobes refines placement; colliding surfels can migrate apart).
  Root cause of the bake-time smear: phi degenerates toward a planar projection, so
  surfaces occluding each other along the projection axis share texels and average.
