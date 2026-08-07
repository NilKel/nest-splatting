# WSR distillation — sort-free rendering with learned per-surfel occlusion

Status: stage 1 (per-surfel scalar occlusion) validated on room8k_p32, 2026-08-05.

## 0. Results (room8k_p32, test set, sorted baseline 31.14 dB)

| render | test PSNR |
|---|---|
| sorted through the wsr clone (sanity) | 31.14 dB (exact reproduction) |
| WSR zero-shot, occ = 1 (naive weighted mean) | 21.02 dB |
| WSR zero-shot, distilled occ (no training) | 23.92 dB |
| WSR finetune 5k (`room_wsr5k`) | **30.47 dB** |
| composite finetune 5k (`room_htc5k`, §3b) | **30.53 dB** |
| gated-WSR finetune 5k (`room_wsrg5k`, §3d, tau 0.05, hard gate v1) | 30.50 dB |
| soft-gated-WSR finetune 5k (`room_wsrgs5k`, §3d, tau 0.05, v2) | 30.55 dB |
| mean-depth-gated WSR 5k (`room_wsrd5k`, §3e, margin 0.15) | **30.62 dB** |

The gated finetunes additionally kill ungated WSR's off-hull see-through
(novel-view stress, dolly 1.2 m: tabletop solid) with no sort and no z-buffer
core. v2 (the deployed `room_wsrg` card, `?wsr=2`) replaces v1's hard
piecewise-constant gate — whose bin edges are camera-attached iso-depth
planes and produced rippling lines tracking camera zoom/rotation — with a
continuous interpolated gate (details in §3d).

PSNR-wise the composite is only +0.06 over pure WSR — but the win is
LOCALIZED, exactly where PSNR barely looks: the opaque-occluder see-through
(carpet through the coffee table, translucent subwoofer/TV in
`room_wsr5k/final_test_renders/000_DSCF4667`) is visually GONE in
`room_htc5k` (front term is per-frame exact). Deployed cards: `room_wsr`
(`?wsr=1`) and `room_htc` (`?ht=3`), same 73 MB stride-7 bundle format.

Sorted-equivalence of the clone is bit-exact; FD gradcheck of the WSR chains
(occ, opacity/alpha, SV, probes, texture) passes at rel ≤1e-4 (the flagged
rot/f_dc sites reproduce identically on the BASE probe module — pre-existing
FD artifacts: fp16 per-Gauss color staircase + quat kinks).
Distilled occ stats on room: mean 0.337, p10 0.044, p90 0.62 — deep overdraw,
which is why the zero-shot view-averaged scalar alone leaves 7 dB on the table
and the finetune (occ + opacity + SV + probes + texture, geometry frozen)
recovers most of it.
Roadmap context: `ROADMAP_2026_08.md` §2 item 4 (WSR viewer mode) + §3
(transmittance-field distillation). Stage 2 (per-texel directional occlusion in
the atlas alpha channel) builds on this.

## 1. The operator

Sorted alpha blending (what we train and ship today) is replaced by the
order-independent weighted-sum composite (SortFreeGS/WBOIT family):

```
P    = Π_i (1 − α_i)                    (coverage product — order-independent)
w_i  = α_i · occ_i                      (occ_i = learned per-surfel occlusion)
C̄    = Σ w_i c_i / Σ w_i
out  = (1 − P) · C̄                      (premultiplied, bg composited in Python
                                         via the alpha map A = 1 − P, unchanged)
```

**Exactness identity.** `Σ_i α_i T_i = 1 − P` telescopes, so if `occ_i` equaled
the true per-fragment transmittance `T_i`, the WSR composite would reproduce
sorted blending *exactly*. A per-surfel scalar can only store a view-averaged
`T̄_i`, so WSR ≈ sorted + view-variance error — that error is what the finetune
absorbs (into occ, opacity, SV color, probes, texture).

**Distill init.** `occ_i ← Σ(α·T) / Σα` over all training-view fragments — the
visibility-weighted mean transmittance. Both sums are dumped by one sorted
forward pass per training view with `record_transmittance=True` (the WSR clone
repurposes `cover_pixel` to accumulate Σα instead of a pixel count; `trans_avg`
holds Σ(α·T) as before). This starts the finetune at (nearly) the sorted
solution instead of at uniform occ=1.

Parameterization: raw logit tensor, `occ = sigmoid(raw)`, init
`raw = logit(clamp(occ_init, 1e-3, 1−1e-3))`.

## 2. CUDA: `submodules/diff_surfel_3D_sh_res_probe_wsr`

Isolated clone of `diff_surfel_3D_sh_res_probe` (CUDA isolation rule — the probe
module is itself live for proberes training). Byte-identical when
`set_wsr(0)` (the default). Under `set_wsr(1, occ, occ_grad, aux)`:

**Forward** (scalar surfel path; the WMMA forward is compile-disabled and the
probe flag doesn't use it):
- early-out at `T < 1e-4` REMOVED — the sort-free operator includes every
  fragment, so training must too. Occluded-but-visible-elsewhere surfels are
  exactly the leakage the finetune has to learn to suppress; early-out would
  hide it from the loss (train/deploy mismatch).
- `w = α·occ[id]`, `den += w`; `T` keeps its meaning as the coverage product.
- write-out: `out_color = (1−T)·C/den`; `C̄` and `den` stored in the `aux`
  buffer `[4,H,W]` for the backward. Aux depth/normal maps get the same
  `(1−P)/den` normalization so their alpha-map-relative semantics survive.

**Backward** (std path only; the collab-GEMM path is force-skipped under BOTH
the probe flag and WSR — it has no hooks for either): fully order-independent.
Per fragment, with per-pixel constants `A = 1−P`, `den`, `C̄`:

```
dL/dfeat_i,ch = g_ch · (A/den) · w_i          — same shape as the existing
                                                dchannel_dcolor, so ALL
                                                downstream SV/probe/texture
                                                gradient chains are reused
                                                verbatim
dL/dw_i   = (A/den) · Σ_ch g_ch (feat_i − C̄)
dL/dα_i   = occ_i·dL/dw_i + (Σ_ch g_ch C̄_ch + dL_dalphamap) · P/(1−α_i)
dL/docc_i = α_i·dL/dw_i                        — atomicAdd into occ_grad
```

The T-recurrence, `accum_rec` behind-color walk, and the depth / normal /
distortion alpha recurrences are bypassed (their lambdas are 0 in WSR finetunes
and geometry is frozen). `--detach_res_shape_grad` is not supported under WSR.
Residual-skip gates use `w_gate = α·occ` in both directions (forward can't know
`den` mid-loop), matching exactly; all thresholds are 0 in the target runs
anyway.

**Plumbing** is the device-global-pointer pattern (cf. `set_mlp_weights`,
gestex atlas): `set_wsr(mode, occ, occ_grad, aux)` installs raw pointers into
both TUs. Re-call before every render while active; tensors must stay alive
through backward; occ grads bypass autograd — the Python side chains the
sigmoid derivative manually and assigns `raw.grad` before `optimizer.step()`.

## 3. Finetune protocol (stage 1)

Vehicle: `room8k_p32` (proberes, 15k, sorted test PSNR **31.14 dB**) — the
checkpoint behind the deployed `room_p32_oct` / `room_packed` cards, so the
WebGPU A/B is apples-to-apples. Popping is an ordering artifact (2D surfels are
exempt from EVER's theorem), so a WSR operator that holds PSNR ≈ kills popping
by construction.

1. Load checkpoint (PLY + `ngp_15000.pth` incl. probes + texture).
2. Distill dump: one sorted epoch with `record_transmittance` → occ init.
3. Zero-shot WSR eval (no training) — measures how far view-averaged occ alone
   gets; this is the go/no-go signal for the whole approach.
4. Finetune ~5k iters, WSR operator on train AND eval renders:
   trainable = occ, opacity, SV colors, probes, texture; frozen = xyz,
   scaling, rotation (LR = 0); no densification; lambdas: dist/normal 0.
5. Report WSR test PSNR vs 31.14 sorted; then export + viewer WSR mode
   (delete sort dispatches; additive accumulation + coverage product +
   normalize subpass — the ht_composite scaffolding already covers most of it).

## 3b. Stage 1.5 — `--wsr_composite` (ht=1 core + occ tail, wsr_mode 2)

Motivation (user's diagnosis, confirmed): a baked occlusion — scalar OR
directional — is a far-field approximation of a *ray* quantity `V(camera→x)`;
it cannot react to the camera crossing an occluder, and it fails hardest at
sharp opaque occluders (the observed see-through). A z-buffer test against the
actual fragments IS the per-frame-exact relational operator, and it is not a
sort. So: keep ht=1's exact per-pixel frontmost fragment, use the learned occ
only for everything behind it:

```
F   = argmin_i depth_i          (exact ray-splat intersection depth,
                                 center-depth fallback — same key as the
                                 viewer's z-buffer core)
out = α_F·c_F + (1−α_F)·(1−P_t)·(Σ_tail w c / Σ_tail w),  w = α·occ
A   = α_F + (1−α_F)·(1−P_t),    P_t = Π_tail(1−α)
```

Opaque-occluder bleed-through dies by construction (α_F ≈ 1 ⇒ tail ≈ 0);
the field approximation only touches genuinely translucent second layers.

CUDA (`wsr_mode 2`, same clone): the forward accumulates ALL fragments as in
mode 1 while tracking the depth-argmin, then removes the winner algebraically
at write-out (num−w_F·c_F, den−w_F, P_t = P/(1−α_F)) — still order-independent.
aux grows to [8,H,W] (+P_t, α_F, front gauss id). Backward: the front fragment
gets `dL/dc_F = g·α_F`, `dL/dα_F = Σg·(c_F − (1−P_t)C̄) + dL_dA·P_t`, no occ
grad; tail fragments get the mode-1 forms with `norm = (1−α_F)(1−P_t)/den_t`
and coverage `(1−α_F)P_t/(1−α)`. Gradcheck: occ/opacity exact (~1e-5 — their
FD doesn't move depths, so the argmin is stable); xyz/rot FD flags are the
argmin flipping under geometric perturbation (true operator discontinuity,
fixed-assignment analytic gradient is correct a.e.; geometry is frozen anyway).

Viewer: `?ht=3` ("occ tail") — ht=1's frame graph with `fs_tail_occ` (weight =
occ from the stride-7 records) instead of the `exp(−k·Δz)` heuristic. Needs a
probe_mode-2 (WSR) bundle; costs the depth pre-pass but still no sort.

## 3c. Novel-view stress test — RETRACTED first verdict + the eval bug (2026-08-05)

**The first version of this section concluded "the composite leaks off-hull —
operator limitation". That verdict was WRONG, caused by an eval bug**, and is
retracted:

- **The bug**: finetunes renumber iterations to 1..5000; standalone eval
  scripts rebuilt `Config` from the yaml (`ingp_stage.switch_iter = 10_000`)
  and passed `iteration=5000` → the renderer's
  `iteration < switch_iter ⇒ hash_in_CUDA=False` demotion silently rendered
  **mode 0 (SV-only, NO probe texture)** — ~9 dB low, blurry, with the
  operator comparison running on a degraded reconstruction. train.py avoids
  this by zeroing switch_iter under `--cold` (train.py ~9192); the scripts
  now mirror it. (Diagnosed by dumping rasterizer-input fingerprints in both
  contexts: train render_mode 4101 = 5|0x1000, loader 1024 = mode 0.)
- **Corrected result** (`novelview_stress_fixed`, room_htc5k, dolly +0.8 m):
  the COMPOSITE keeps the coffee table SOLID — no couch/carpet bleed —
  matching sorted. The composite does NOT leak at these off-hull poses.
  Pure ungated WSR still leaks there (tabletop transparency), as expected.
- With the fix the loader reproduces training exactly (room_wsr5k evals
  30.474 dB == training's 30.4739; p32 loader-vs-saved-render 52.9 dB).
  The finetuned checkpoints were never corrupt.

The interactive-viewer transparency report that motivated this section came
from free-camera poses likely far beyond dolly 1.2 m; §3d's gate targets
exactly that regime.

## 3d. `--wsr_gate_tau` — 2-pass transmittance-saturation gate (2026-08-05)

The idea, restated depth-relationally: blend coverage with the checkpoint
opacities; once the accumulated opacity IN FRONT OF a fragment's depth has
saturated, gate the fragment entirely. Sorted CUDA gets this free from the
early-out; sort-free it must not reference processing order, so:

```
prepass:  bin_b += log(1−α)  over ALL fragments   (b = log-depth bin;
                                                   16 bins, zmin 0.2, zmax 120)
gate:     fragment in bin b participates iff exp(Σ_{b'<b} bin_b') ≥ tau
```

Same-bin fragments never gate each other (bin-width leakage floor, ~1.33×
relative depth per bin); the frontmost occupied bin always survives. The gate
is a step function of DEPTH → order-independent, frame-coherent, and
identically computable by a WebGPU 2-pass (no sort, no z-buffer core).

- **CUDA** (`diff_surfel_3D_sh_res_probe_wsr`): `set_wsr_gate(tau, bins,
  zmin, zmax, tbin)` device globals; `wsrGatePrepassCUDA` (alpha-only walk
  launched from `FORWARD::render` before the main kernel) writes the
  exclusive-prefix T per bin to `tbin [16,H,W]`; forward AND backward fully
  discard gated fragments (no grads — gate treated as constant, correct
  a.e.). Renderer arms it from `ingp.wsr_gate_tau` (`--wsr_gate_tau`, 0=off).
  Gradcheck with the gate armed: occ/opacity exact (~1e-5); xyz/rot FD flags
  are gate-flips under geometric perturbation (frozen in finetunes anyway).
- **Zero-shot on room_wsr5k** (occ trained WITHOUT the gate): capture-view
  cost tau 0.01/0.05/0.1 → −0.26/−0.39/−0.54 dB; at dolly +0.8 the tabletop
  transparency is GONE (solid, crisp) at tau 0.05. The gate is the per-frame
  relational cut that baked occ cannot express.
- **Viewer** (`?wsr=2`, Halloumi-WS): pass 1 `fs_wsr_bins` — 16 bins as
  4×rgba16float additive MRT (32 B/sample = the default
  maxColorAttachmentBytesPerSample budget), lean `shade_alpha()` (no atlas
  fetch); pass 2 `fs_wsr_gated` — textureLoads the 4 bin vec4s, exclusive
  prefix over strictly-nearer bins, `discard` below WSR_GATE_TAU, then the
  normal fs_wsr accumulate; same `fs_wsr_composite`. Bin constants hardcoded
  to match CUDA — keep WSR_GATE_LO/INVR (render_2dgs.wgsl) in sync with
  setWsrGate's zmin/zmax.
- **Finetune**: `room_wsrg5k` (v1) / `room_wsrgs5k` (v2) = room_wsr5k's exact
  command + `--wsr_gate_tau 0.05` — occ/opacity/SV/probes/texture adapt WITH
  the gate in the loss loop, so train == deploy including the gate.

**v2 — soft continuous gate (the ripple fix, 2026-08-06).** The v1 hard gate
(discard when the EXCLUSIVE-prefix bin transmittance < τ) produced "rippling
lines that move with the camera" in the interactive viewer: bin edges are
iso-view-depth planes RIGIDLY ATTACHED TO THE CAMERA, so any zoom/rotation
sweeps them through the scene and whole bands of fragments flip gate state at
once (confirmed by dolly-pair diffs: patchy area-flips on the tabletop, absent
ungated). v2 makes the gate C0-continuous:

- tbin stores exclusive cumulative OPTICAL DEPTH (not exp); the gate value is
  the linear interp of that prefix evaluated **ONE BIN NEARER** than the
  fragment (`d_gate = max(d·bins − 1, 0)`). One bin back, co-surfel mass
  (hard-binned at the fragment's own bin) never counts — interpolating at the
  fragment's own position mixes in its OWN bin's mass and a one-bin surface
  stack partially gates ITSELF (first attempt: 13 dB, catastrophic) — and the
  lerp endpoints meet exactly at bin crossings, so the value is continuous
  under any camera motion. Cost: occluders must be ≥1 bin (~1.33× relative
  depth) nearer before gating starts (guard band).
- the gate is a smoothstep of T over [0.5τ, 1.5τ] FOLDED INTO THE OCC WEIGHT
  (w = α·occ·s), never a hard discard; the coverage product keeps raw α. The
  backward scales occ_v and the occ-grad by the same s; ∂s/∂α cross-terms are
  dropped (gate-as-constant policy — the opacity FD gradcheck flags ~15%
  under-estimates from exactly this, expected).
- verification: dolly-pair diff (0.80 vs 0.83 m) shows the v1 area-flips GONE
  (edge-motion only, matching ungated); dolly 1.2 stays solid; capture-view
  PSNR 30.55 (best in family). Viewer + CUDA updated in lockstep
  (fs_wsr_gated one-bin-nearer interp + smoothstep; no discard).

## 3e. `--wsr_dgate_margin` — mean-depth-gated WSR (?wsr=3, "D-gate", 2026-08-06)

User's insight, and it's the correct factorization: the SATURATION DEPTH is an
INTEGRAL of opacity along the ray — invariant (to first order) to the local
order swaps that make sorted COLOR compositing pop. So the sort is not the
problem; using its order for color is. Keep the sort, use it ONLY for the
scalar per-pixel visibility anchor, and blend color order-independently:

```
pre-pass (sorted): D̄ = Σ(α·T·d)/Σ(α·T)   (alpha-weighted mean depth)
                   A  = Σ(α·T) = 1 − T_final          (saturation)
render (OI):       w = α·occ·s
                   s = 1 − g(A)·smoothstep(D̄(1+m/2), D̄(1+3m/2), d)
                   g = smoothstep(0.6, 0.9, A)   (never gate behind an
                                                  unsaturated/translucent mix)
```

- **No bins ⇒ no ripple**: D̄ is continuous in camera motion; nothing is
  camera-attached and piecewise-constant. And the margin is RELATIVE depth
  (m = 0.15 ⇒ fade over [1.075·D̄, 1.225·D̄]) — far tighter than the binned
  gate's 33–77% bin quantum, so closer occluder/background pairs separate.
- **Viewer** (`?wsr=3`): the blend unit computes the anchor for free — draw
  in radix-sorted order with premultiplied over blending, src = (zv·α, α):
  the target accumulates exactly (Σ αT·zv, Σ αT) (`fs_wsr_depth`, cheap
  alpha-only fragments, one rgba16f target). `fs_wsr_gated_d` then reads
  (D̄, A) and does the OI accumulate. Pays the radix sort (unlike ?wsr=2) —
  the honest trade for the exact anchor — but sorted compositing of COLOR
  never happens, so popping stays structurally impossible.
- **CUDA**: `wsrDGatePrepassCUDA` (sorted walk with exact early-out at
  T<1e-4) fills dbuf [2,H,W]; occ-fold identical to §3d v2 (coverage keeps
  raw α; occ grad scaled by s; ∂s/∂α and ∂D̄/∂α dropped — gate-as-constant).
  `--wsr_dgate_margin` (0 = off), mutually exclusive with `--wsr_gate_tau`.
- **Results** (`room_wsrd5k`, margin 0.15): **30.62 dB** — best of the WSR
  family (ungated 30.47, binned soft gate 30.55, composite 30.53). Dolly 1.2
  solid; dolly-pair diff shows no band structure (matches ungated). Zero-shot
  margins on §3d weights: 0.1/−0.55, 0.15/−0.21, 0.25/+0.12 dB.
- Deployed as the `room_wsrd` card (`?wsr=3`, viewer index-287d5570; timing
  bars for the wsr modes added in index-f747f061 — Sort slot = radix for
  ?wsr=3, bins pre-pass for ?wsr=2, zero-length stamp for ?wsr=1).

**All-scenes rollout (2026-08-06).** The same recipe (5k D-gate finetune,
margin 0.15, from each scene's proberes checkpoint — `oct8k_p32_free` bases,
`bike8k_p32` for bicycle — BC7 stride-7 bundles) batch-deployed for the full
mip-360 set to the HF subfolder `mip_360_wsrd/`, one `?wsr=3` card each:

| scene | sorted base | D-gate WSR | Δ |
|---|---|---|---|
| room (`room_wsrd5k`) | 31.14 | 30.62 | −0.52 |
| bicycle | 23.97 | 23.44 | −0.53 |
| bonsai | 32.00 | 30.19 | −1.81 |
| counter | 28.90 | 28.23 | −0.67 |
| flowers | 20.48 | 19.88 | −0.60 |
| garden | 26.34 | 25.94 | −0.40 |
| kitchen | 30.11 | 29.00 | −1.11 |
| stump | 25.58 | 25.04 | −0.54 |
| treehill | 22.22 | 21.81 | −0.41 |

Pattern: outdoor scenes pay ~0.4–0.6 dB for popping-free rendering; the
high-fidelity indoors (bonsai −1.8, kitchen −1.1) pay the most — their sharp
specular/high-contrast content is where the order-independent tail mean is
weakest. Per-scene finetunes took 10–18 min each (batch driver:
one-off scratchpad script; per-scene logs alongside it).

## 4. Known limits / expectations

- Per-surfel scalar occ cannot express view-dependent occlusion (K=7 SV basis
  or the per-texel alpha field are stage 1.5/2). Expect residual leakage at
  depth-order-flipping viewpoints; flat-top beta_scaled kernels + near-opaque
  bake keep the halo channel small (SortFreeGS analysis).
- Training cost: no early-out ⇒ full-overdraw fragment lists in both passes,
  and the scalar (non-collab) backward. Fine for a 5k finetune.
- The occ ships per-surfel (1 byte in the bundle later); the atlas alpha
  channel stays free for stage 2.
