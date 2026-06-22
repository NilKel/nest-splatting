# `--deform` — Per-Surfel Time Deformation with Canonical-Space Hash

Models the small movements a subject makes during capture. Each surfel is stored
in a **canonical** pose; for each frame (timestep) an MLP produces a per-surfel
position delta that moves the surfel to its time-t pose for rasterization, while
the hashgrid is always sampled at the **canonical** position. Training factors
each frame's motion out into the deformation field; at inference you set the
deformation to zero to get a single sharp **static** reconstruction.

## Design (and why it needs a CUDA fork)

The standard `diff_surfel_3D_sh_res` rasterizer uses **one** position per surfel
for both the geometry (projection / SV color) and the in-kernel hash query — the
hash point is reconstructed inside the kernel from the means you pass in, so the
two cannot differ. To fit frame *t* the geometry must be at the deformed position;
querying the hash there too would smear the canonical texture as the subject moves.

So the deform path uses a fork, **`diff_surfel_deform`**, that decouples the two:

- **Geometry / SV / alpha / depth / normal**: the **deformed** means (`_xyz + Δpos`)
  passed to the rasterizer as usual.
- **Hash query**: the **canonical** `_xyz`, supplied out-of-band via a device-global
  pointer (`set_canonical_xyz`, the same setter idiom as `set_residual_mode`).

The kernel change is surgical because `collected_pk` (the surfel center) is used
*only* in the hash-xyz reconstruction. Forward: when the canonical pointer is set,
load `collected_pk` from it instead of the deformed transmat (forward.cu). Backward:
same load, plus drop the one center-from-hash gradient term
(`acc_dL_dhomoMat[6..8] += dL_dxyz`) — the canonical centers are detached on the
deform path (geometry still optimizes `_xyz` via the deformed means; the tangent
`dL_du/dL_dv` terms stay, so the deformed geometry keeps its hash UV gradient).
Null pointer ⇒ byte-identical to `diff_surfel_3D_sh_res`.

Because the canonical query reconstructs as `canonical_center + s·SuTu + s·SvTv` and
the deformation is **position-only**, `Δpos` cancels out of the canonical point and
no inverse transform is needed; the rotation/scale tangents are unchanged so the
canonical hash point is exact.

## Components

- **Per-surfel latent** `_deform_latent` `[N, deform_dim]` (zero-init = identity).
  A genuine per-Gauss tensor on `GaussianModel`, threaded through every
  clone/split/prune/PLY/capture path exactly like `_scaling_z` (so it rides fastgs
  densification and the res_3d_paired duplication). Lives in `gaussians.optimizer`
  as its own param group (`deform_latent`, LR `--deform_latent_lr`).
- **Deform MLP** (`scene/deform_model.py`): `MLP(latent ⊕ fourier(t)) → (Δpos[3], Δrot[4])`,
  zero-init output head (identity at iter 0). Its own Adam (`--deform_mlp_lr`).
  Scalar normalized frame time `t∈[0,1]` from sorted `image_name` index (Fourier
  encoded → temporal smoothness). Position-only v1 uses Δpos; Δrot is computed but
  not applied (kept for a future rotation-deform variant).
- **Routing** (`train.py`): when `--deform`, the renderer's `_sh_res_rasterizer`
  and `sys.modules['diff_surfel_3D_sh_res']` are swapped to `diff_surfel_deform`
  (the module-swap idiom), so the rasterizer + MLP-weight/bias/residual-mode setters
  all target the fork. `set_canonical_xyz(pc.get_xyz.detach())` is called before each
  training render and cleared (null) after backward (so eval/debug renders — Δ=0,
  means3D already canonical — fall back to means3D ⇒ correct canonical hash).

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--deform` | off | Enable per-surfel time deformation + canonical-space hash. |
| `--deform_dim` | 8 | Per-surfel latent dimension. |
| `--deform_latent_lr` | 1.6e-4 | Adam LR for the per-surfel latent. |
| `--deform_mlp_lr` | 1e-3 | Adam LR for the deform MLP. |
| `--deform_width` / `--deform_depth` | 128 / 4 | MLP hidden width / layers. |
| `--deform_time_freqs` | 6 | Fourier frequencies for the scalar time encoding. |
| `--deform_warmup` | 3000 | Iters before deformation starts (identity before this; let the canonical geometry form first — important under `--cold`). |
| `--deform_reg` | 0.0 | L2 penalty on Δ to keep the per-frame motion small. |

## Usage

`--deform` is currently wired for **`--method 3D_SH_res`** (the deform rasterizer is
a fork of `diff_surfel_3D_sh_res`). Example on the Max scene:

```bash
python train.py -s data/personal/Max -m <outdir> \
  --yaml ./configs/max.yaml --iterations 35000 \
  --method 3D_SH_res --hybrid_levels 2 --disable_c2f --aabb rect --kernel beta_scaled \
  -i images_2 --cold --fastgs --fastgs_densify_interval 500 --fastgs_densify_until 20000 \
  --feature SV --lowpass \
  --deform --deform_warmup 3000 --deform_reg 1e-4
```

Composes with `--3rgs` (deform per-surfel, then per-camera rigid pose). Monitor via
the `[DEFORM iter=N]` line (`d_xyz` mean/max and the latent magnitude).

**Inference / static recon**: render with the deformation disabled (Δ=0) → all
frames collapse to the canonical pose; the hash is already canonical, so the static
reconstruction stays sharp. (The trained PLY is the canonical model.)

## Limitations (v1)

- **`--method 3D_SH_res` only.** With `res_3d_paired` (which routes through the
  `mixed_3d` rasterizer) the deformation still moves the geometry, but the hash is
  sampled at the deformed position (no canonical decoupling) — that would need an
  equivalent fork of `diff_surfel_mixed_3d`. No crash; just not canonical.
- **Position-only.** Rotation deformation is computed by the MLP but not applied
  (applying it would require passing canonical tangents to the kernel for an exact
  canonical hash).
- **Temporal opacity is static.** A per-frame opacity window (UBS/7DGS-style fade
  in/out) is intentionally not included — for small jitter it invites flicker
  overfitting. Add later behind a flag with a wide-`σ_t` prior if disocclusions appear.
- **Canonical-position gradient detached.** The hash does not push the canonical
  surfel center (that term is dropped); centers are optimized by the geometry path.
  A full redirect (`dL_d_canonical_xyz` as a real grad output) is a future upgrade.

## Verified

- Math/forward: zero-init ⇒ identity; null canonical pointer ⇒ byte-identical to
  `diff_surfel_3D_sh_res`.
- End-to-end on `data/personal/Max` (`3D_SH_res` + `--feature SV` + `--fastgs` +
  `--cold`, 700 iters): forward + backward + fastgs densification run clean; the
  `_deform_latent` survives clone/split and serializes (`deform_0..7` in the PLY).
- Gradient/optimizer check (iter 105): latent `in_gauss_optimizer=True`,
  `grad_norm≈6e-6`, `in_adam_state=True`; MLP `params_with_grad=10/10`,
  `grad_norm≈0.82`, all in `deform_optimizer`.
