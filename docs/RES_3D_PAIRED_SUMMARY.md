# res_3d_paired: local→global ReLU curriculum + textured/untextured pairing

Companion doc to [`PROBERES_BAKING_SUMMARY.md`](PROBERES_BAKING_SUMMARY.md),
focused on the `--method res_3d_paired` training mode: what the two staged
transitions do (per-Gauss "local" ReLU → post-blend "global" ReLU, then the
2D-textured / 3D-untextured split), how the two surfel populations are
combined in one alpha-blend, and where the results stand. The full res_3d
family reference (res_switch / res_3d / res_3d_double, dual-cascade variants)
is [`RES_3D_MODES.md`](RES_3D_MODES.md).

---

## 0. The two problems it solves

Baseline `3D_SH_res` renders every surfel as a flat 2DGS disc whose colour is
`ReLU( ReLU(SV+0.5) + residual )` — the hash+MLP residual rides on a
per-surfel SV base, and the **outer ReLU is applied per surfel, before
blending**. Two limitations show up mid-training:

1. **The per-surfel clamp starves the residual of gradient.** Wherever a
   surfel's pre-activation goes negative, its gradient is cut at that surfel
   — even if the *blended pixel* is too bright and wants a negative
   contribution. Signed subtraction across surfels is impossible: no surfel
   can darken what another over-brightened.
2. **One primitive shape has to serve two jobs.** Thin discs are ideal
   residual carriers (well-defined uv → crisp texture detail) but poor at
   smooth volumetric matter (foliage, distant background), where a soft 3D
   blob with plain SV colour is both cheaper and better behaved.

res_3d_paired addresses (1) with a mid-training **activation flip** and (2)
with a mid-training **population split**, on separate iterations so the model
absorbs one shock at a time.

---

## 1. Stage 0 (iters 0 → `--res_switch_iter`): local ReLU

Ordinary `3D_SH_res` mode-0 training:

```
feat_i = LeakyReLU( ReLU(SV_i + 0.5) + residual_i, α=0.01 )   # per surfel, BEFORE blend
C      = Σ_i T_i · α_i · feat_i
image  = C
```

The per-surfel clamp acts as a strong local regulariser early on — each
surfel's colour is forced non-negative(ish), which keeps multi-view geometry
optimisation stable while densification is still moving mass around. The
`--lru 0.01` leaky slope (auto-defaulted for this family) keeps a trickle of
gradient through clamped sites.

## 2. Stage 1 (`--res_switch_iter`, canonical 10000): local → global ReLU

One kernel-state flip (`set_residual_mode(2)`), no tensor changes:

```
feat_i = ReLU(SV_i + 0.5) + residual_i          # SIGNED — no per-surfel clamp
C      = Σ_i T_i · α_i · feat_i
image  = LeakyReLU(C, α=0.01)                    # ONE clamp, per pixel, AFTER blend
```

The clamp moves from each surfel to the blended pixel ("global" ReLU). Two
consequences:

- **Signed residuals become expressive**: a surfel can now *subtract* from
  the accumulated colour, so high-frequency detail that needs darkening
  (shadow edges, dark texture on bright SV) stops fighting the activation.
- **Richer gradient**: `dL/dfeat_i = T_i·α_i · LeakyReLU'(C)` flows to
  *every* contributor of the pixel; under mode 0 the per-surfel clamp gated
  each contributor independently.

Matching the leaky slope on both sides of the flip (`--lru 0.01` in the
kernel pre-flip, in Python post-flip) makes the transition a smooth knee —
without it the loss jumps discontinuously at the flip. Mechanically the flip
is: `set_residual_mode(2)` on the CUDA module + `ingp.is_mixed_deferred_relu_mode
= True` (renderer applies the Python-side post-blend LeakyReLU) +
`args._residual_mode = 2` (recorded in `bake_meta.json` so the baked renderer
reproduces the same composition).

## 3. Stage 2 (`--res_3d_iter`, canonical 15000): the paired split

Every live surfel **duplicates** into a specialised pair (N → 2N rows):

| | textured copy | untextured copy |
|---|---|---|
| `_is_textured` | True | False |
| geometry | 2DGS flat disc (unchanged) | 3D **EWA ellipsoid** — gains learnable `_scaling_z`, init flat: `log(0.05)+min(log sx, log sy)` |
| kernel | run's `--kernel` (beta_scaled ray-splat) | FastGS-verbatim `computeCov3D`/`computeCov2D` → screen conic (`--kernel2` can override its falloff) |
| colour | `ReLU(SV+0.5) + residual` (hash+MLP) | `ReLU(SV+0.5)` only — **no hash, no MLP** |
| opacity | `α · texsplit_tex_frac` (default 0.5) | `α · (1 − texsplit_tex_frac)` |

Both copies keep the full trained SV — nothing is zeroed (that's what
separates `res_3d_paired` from `res_3d`, whose textured half is residual-only).
The opacity split means the pair initially composites to approximately the
pre-split appearance, then the two halves specialise under gradient: discs
keep the texture detail, ellipsoids inflate along their new z-axis wherever
soft volumetric SV matter is the better explanation. Adam is rebuilt for the
doubled tensors; densification inherits the flag (textured parents → textured
children).

### How the two populations combine: ONE joint cascade

Both halves blend into the **same** transmittance/accumulator — a single
depth-sorted alpha-blend where only the per-surfel `feat` differs:

```
For each Gauss j in depth order:                 # one shared sort, one shared T
  α_j from its own kernel (ray-splat disc  |  EWA conic)
  feat_j = ReLU(SV_j+0.5) + residual_j     if textured
         = ReLU(SV_j+0.5)                  if untextured
  C += T · α_j · feat_j
  T *= (1 − α_j)
image = LeakyReLU(C, 0.01)                       # stage-1 global ReLU, unchanged
```

So the split does NOT introduce a second render pass or a second cascade —
occlusion between the halves is physically consistent (a disc can hide an
ellipsoid and vice versa). The sibling mode `res_3d_double` instead runs two
independent transmittances (tex walls don't shadow untex matter); paired's
joint cascade measured the most stable — the joint T gives gentler per-Gauss
gradients around a shared optimum, and the dual cascade's extra freedom
(`C_tex + C_sv = target` has a continuum of splits) lets the optimiser drift.

Kernel: post-split rendering dispatches to `diff_surfel_mixed_3d`, whose
per-Gauss `is_textured` branch is block-uniform (all 256 threads take the same
side per Gaussian — no warp divergence). Backward is the standard mixed_3d
one: reverse-T recovery on the single shared cascade; the untextured EWA VJP
(conic→cov2D→cov3D→scale/`scaling_z`/quat) is the FD-gradcheck-verified FastGS
port; hash/MLP gradients only ever see textured fragments.

---

## 4. Canonical run

```bash
python train.py --method res_3d_paired -s <scene> --yaml configs/360_outdoor.yaml \
  --eval -i images_4 --iterations 35000 \
  --res_switch_iter 10000 --res_3d_iter 15000 --texsplit_tex_frac 0.5 \
  --kernel beta_scaled --feature SV --lowpass --hybrid_levels 2 --disable_c2f \
  --aabb snugbox --contribution_thresh 30 --w_weight_reg 0.05 --w_weight_gamma 25
# --lru auto-defaults to 0.01 for this family
```

(Production config name: `fix_BS2_10S15kL01_SV_30thr_005w25gLP4lev_FRP5k10_N2F_Jac_5ksp_i2fast`
— "10S15k" = the 10k flip / 15k split stagger, "L01" = lru 0.01.)

The stagger matters: 5k iterations of mode-2 single-cascade between flip and
split lets SV re-equilibrate to the signed-residual regime on a fixed surfel
set before the parameter count doubles.

---

## 5. Results (mip-360, 9 scenes, RTX 5090)

Trained 35k with the config above, baked to a shelf-packed BC7 atlas
(`max_res 64`), rendered with the CONIC paired lean renderer
(`diff_surfel_bake_render_paired_lean`, `LEAN_FLAGS=CONIC`). Full bench:
[`BENCH_5090_MIP360.md`](BENCH_5090_MIP360.md) §res_3d_paired.

| scene | nGauss | %tex | neural PSNR | baked PSNR | CONIC FPS |
|---|---:|---:|---:|---:|---:|
| bicycle  | 170k | 73% | 24.17 | 24.16 | 1347 |
| bonsai   | 129k | 69% | 32.40 | 32.34 | 1203 |
| counter  | 102k | 59% | 29.15 | 29.11 | 1402 |
| flowers  | 222k | 67% | 20.58 | 20.80 | 1192 |
| garden   | 188k | 70% | 26.86 | 26.82 | 1671 |
| kitchen  | 177k | 62% | 31.20 | 31.11 | 1287 |
| room     |  80k | —   | 31.15 | 31.10 | 1652 |
| stump    | 107k | —   | 25.77 | 25.86 | 1509 |
| treehill | 181k | —   | 22.46 | 22.51 | 1090 |
| **mean** | — | — | **27.08** | **27.09** | **1373** |

Takeaways:

- **+0.24 dB mean over the 3D_SH_res baseline config** (27.08 vs 26.84
  neural) — the split pays for itself; 59–73% of primitives stay textured,
  the rest migrate to the cheap EWA half.
- **Bake is lossless on PSNR at this config** (mean Δ = +0.01 dB; flowers /
  stump / treehill actually *gain* at bake time). LPIPS still degrades on
  bake (bicycle 0.2678 → 0.2893) — the u8+BC7 quantization and Nyquist clamp
  cost perceptual detail even when PSNR holds.
- **1373 FPS mean baked** (+65.4% over the prod paired renderer). The paired
  code paths cost −7.4% vs the pure-2DGS lean renderer on an all-textured
  scene (dual staging of CONIC + EWA fields per Gauss); acceptable, first
  lever identified if it ever matters.

### Head-to-head vs 3D_SH_res × CONIC (same 5090, same harness)

3D_SH_res column = the `RD_SV_30thr_005w25gLP_N2f_frz5k10` bakes through
`diff_surfel_bake_render_lean` `LEAN_FLAGS=CONIC`
([BENCH_5090_MIP360.md](BENCH_5090_MIP360.md) full-picture table); paired
column = the table above. Both are baked SH+atlas lanes, 50/400 cuda.Event
frames.

| scene | 3DSH nG | paired nG | 3DSH PSNR | paired PSNR | ΔdB | 3DSH FPS | paired FPS | ΔFPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle  | 161k | 170k | 23.91 | 24.16 | **+0.25** | 1385 | 1347 | −2.7% |
| bonsai   |  92k | 129k | 31.81 | 32.34 | **+0.53** | 1261 | 1203 | −4.6% |
| counter  |  68k | 102k | 28.90 | 29.11 | **+0.21** | 1442 | 1402 | −2.8% |
| flowers  | 185k | 222k | 20.62 | 20.80 | **+0.18** | 1291 | 1192 | −7.7% |
| garden   | 149k | 188k | 26.71 | 26.82 | **+0.11** | 1943 | 1671 | −14.0% |
| kitchen  | 120k | 177k | 30.65 | 31.11 | **+0.46** | 1495 | 1287 | −13.9% |
| room     |  63k |  80k | 30.73 | 31.10 | **+0.37** | 1792 | 1652 | −7.8% |
| stump    |  96k | 107k | 25.54 | 25.86 | **+0.32** | 1394 | 1509 | **+8.2%** |
| treehill | 175k | 181k | 22.29 | 22.51 | **+0.22** | 1105 | 1090 | −1.4% |
| **mean** | **123k** | **151k** | **26.80** | **27.09** | **+0.29** | **1456** | **1373** | **−5.7%** |

Reading:

- **Paired wins PSNR on 9/9 scenes, +0.29 dB mean**, carrying +22% more
  primitives (the untextured EWA half is nearly free capacity — no atlas
  bytes, no hash/MLP at train time, SV-only colour).
- **The FPS cost is −5.7% mean**, and it is *not* uniform: it tracks the
  primitive-count gap, not the paired-branch tax. Garden/kitchen pay
  −14% carrying +26/+48% Gauss; treehill pays −1.4% at +3% Gauss. Stump
  actually *wins* FPS (+8.2%) despite +11% Gauss — its 3D_SH_res bake is
  the bloated-surfel outlier (mean axis 0.21), and the paired split
  migrates exactly those fat low-detail surfels to the compact-support
  EWA half.
- **Per-dB cost**: +0.29 dB for −83 FPS at ~1400 — cheaper than any
  training-side quality lever measured on the 3D_SH_res family (e.g. the
  `0w0` no-overdraw-reg config buys +0.13 dB for −344 FPS vs `005w25`).
- Treehill remains the family's weakest scene in both representations —
  the split does not fix surfel bloat, it just prices it differently.

### Post-bake atlas finetune (`--atlas_finetune`)

The proberes finding — a bake can only lose information, but GT-finetuning
the baked texels recovers it and can pass the teacher on LPIPS — transplanted
onto this mode's shelf atlas via `submodules/diff_surfel_atlas_ft` (the
differentiable shelf-fetch rasterizer). Bicycle, atlas-only (all other params
frozen, `--atlas_finetune_only`, atlas lr 1e-3), test-set trajectory through
the training renderer:

| | PSNR | SSIM | LPIPS(L) |
|---|---|---|---|
| teacher (neural, 35k) | 24.17 | 0.6963 | 0.2678 |
| baked atlas, no finetune (init) | 23.72 | 0.6693 | 0.2970 |
| + atlas-only finetune @2k (**peak**) | **24.14** | **0.7060** | **0.2446** |
| + atlas-only finetune @5k (overshoot) | 23.95 | 0.6904 | 0.2521 |

LPIPS passes the teacher within 2k iterations while PSNR closes to −0.03 dB
of it — same crossover signature as proberes (the explicit texture holds
high-frequency detail the hash+MLP can't).

**The finetune must be SHORT — it peaks ≈2k and then regresses.** Past the
peak, test metrics decay while train loss keeps falling: with eps=1e-15 Adam
every touched texel takes a full lr-sized sign-step regardless of gradient
magnitude, so once the coherent signal is absorbed, the sparse noisy
gradients random-walk 416M texels into high-frequency atlas noise (the exact
pathology `probe_pixel_lr_scale`/`probe_pixel_decay` exist for in proberes —
the shelf atlas has ~30× more texels at far lower per-texel sampling density,
so it shows up much faster). Recipe: ~2–2.5k iters, eval every 500, keep the
best save.

Pitfalls established during bring-up: do NOT let SV train alongside the atlas
in mode 2 (SV↔signed-residual degeneracy + eps=1e-15 Adam sign-steps smeared
everything: −1.5 dB in 1k iters); match the teacher's
`--contribution_thresh`; and stop early per the above. Recipe and further
pitfalls in `.claude/rules/nest-splatting.md` § `--atlas_finetune`.

Deployment metrics through the CONIC paired renderer (finetuned atlas
re-quantized u8 → BC7 via `scripts/export_atlas_ft.py`, benched with
`benchmark_baked.py --skip_bake` + the lean CONIC lane): *pending — being
measured now, table to be filled in.*
