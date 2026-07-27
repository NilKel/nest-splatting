# `--method film` — Feature-wise Linear Modulation (FiLM)

> **Two FiLM modes.** `--method film` (below) is the **cat-family** variant:
> blend-then-PyTorch-MLP, FiLM-modulated 24D hash. `--method 3D_SH_filmres`
> (see the section at the bottom) applies the **same FiLM idea to `3D_SH_res`**:
> per-Gauss SH base + a CUDA-fused residual MLP whose ≤16D hash input is
> FiLM-modulated (`residual = MLP(γ·H + β)`). Both reuse the packed
> `_film_params` storage and the `--film_gamma_init` / `--film_beta_init` flags.

## Concept

Instead of *concatenating* a per-Gauss feature with the spatial hash grid (what
`--method cat` does), each surfel **conditions** the hash grid via a FiLM layer
(Perez et al., 2018, *"FiLM: Visual Reasoning with a General Conditioning Layer"*).
Each primitive stores an explicit scale `γ` and bias `β` that modulate the spatial
texture decoder — the surfels act as material identifiers conditioning a continuous
radiance field.

```
f_i        = γ_i · H(x_i) + β_i           # per-Gauss FiLM modulation (γ scalar, β 24D)
pixel_feat = Σ_i (T_i · α_i) · f_i        # 2DGS alpha-blend of the modulated features
RGB        = MLP(pixel_feat, view_dir)    # one screen-space MLP per pixel (PyTorch)
```

- `H(x_i)` — the **24D hash feature** (6 levels × 4D), queried per Gauss at its
  ray-surfel intersection `x_i` (same machinery as cat's case-1 path).
- `γ_i` — per-Gauss **scalar** scale `[N,1]`.
- `β_i` — per-Gauss **24D bias** `[N,24]`.

This is the cat family's *blend-then-decode* structure (linear blend, then **one**
MLP per pixel), but the per-Gauss conditioning is FiLM (`γ·H+β`) rather than a
concatenated coarse feature. Contrast with `3D_SH_res` which is *decode-then-blend*
(MLP per Gauss, then blend).

## Design / defaults

- **`γ` init 1.0, used directly** (`f = γ·H + β`). **`β` init 0.** → training starts
  at pure hash (`f = 1·H + 0 = H`), identity modulation.
  - **Configurable via `--film_gamma_init G` (default 1.0) and `--film_beta_init B`
    (default 0.0).** Both default to the identity init above. To shift appearance
    "lifting" from the hashgrid toward `β`, run with **`--film_gamma_init 0.1
    --film_beta_init 1.0`**: γ=0.1 attenuates `γ·H` 10× at init (the hash starts via
    `kaiming_uniform_`, std ≈ 0.707), and β=1.0 makes the per-surfel bias the dominant
    term from the start. (Applied at all four `_film_params` init sites — fresh
    `create_from_pcd`, `load_ply` fallback, and the two train.py warmup/init_ply re-inits.)
  - *Caveat:* this only biases the **start**. The hash table trains at `feat_lr=2e-2`
    while `_film_params` trains at `feature_lr=0.0025` (8× slower), so the hash can grow
    back to dominate. If the init alone doesn't hold, slow the hash with
    **`--hash_lr_scale 0.1`** (works for film today, no code).
  - *Future ablation (not yet implemented):* reparametrize as `f = (1+γ)·H + β` with
    `γ` init 0 + weight-decay, so regularization pulls modulation toward "off".
- **MLP**: identical to cat — input = 24D blended FiLM feature **+ 16D view encoding**
  → view-dependent RGB → sigmoid. Runs in **PyTorch** (`ingp.rgb_decode`), not CUDA-fused.
  FiLM has no SV/SH base color; view-dependence comes solely from this view input.
- **`hybrid_levels = 0`** — all 24 dims come from the hash; there are no per-Gauss
  concatenated features. C2F follows `--disable_c2f` (default ON → all 6 hash levels
  active from iter 1, like cat).

## CUDA submodule: `diff_surfel_film`

A fork of `submodules/diff-surfel-rasterization` (the cat rasterizer). The only kernel
change is in the **render_mode-1 (case 1)** path:

- **Forward** (`forward.cu`): after the existing case-1 hash query assembles `feat[]`,
  if `film_gamma != nullptr` it applies `feat[ch] = γ_i·feat[ch] + β_i[ch]` per Gauss
  (read by global Gauss id). `film_gamma == nullptr` ⇒ byte-identical to cat.
- **Backward** (`backward.cu`): with `grad_feat[ch] = w·dL/dpix[ch]` (`w = α·T`):
  - hash-table + xyz gradient are **γ-scaled** (`grad_feat_hashgrid · γ` into
    `query_feature<true>` — one call covers `dL/dhash = γ·w·dL/df` and the γ-scaled
    geometry grad);
  - `dL/dβ_i[ch] = grad_feat[ch]`; `dL/dγ_i = Σ_ch H[ch]·grad_feat[ch]` (un-scaled);
  - `feat` is reconstructed as the **modulated** `γ·H+β` before the α-recurrence so
    `dL/dα` uses the right per-Gauss color.

`γ`/`β` are threaded as two dedicated tensors (`film_gamma` `[N,1]`, `film_beta`
`[N,24]`) through the autograd Function fwd/bwd, with grad outputs
`dL_dfilm_gamma`/`dL_dfilm_beta`. Empty tensors ⇒ nullptr ⇒ plain cat behaviour.

**Build** (CUDA, 2–5 min — always background):
```bash
cd submodules/diff_surfel_film && conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

## Per-Gauss storage: `_film_params` `[N, 25]`

γ and β are **packed into a single per-Gauss tensor** `GaussianModel._film_params`
(`[:, 0:1]` = γ, `[:, 1:25]` = β) so they ride the exact same lifecycle as
`_gaussian_features` (init, optimizer, densify clone/split/prune, PLY round-trip)
through one code path — minimizing desync risk. The renderer slices it into
`pc.get_film_gamma` / `pc.get_film_beta` (autograd views) for the rasterizer.

- **LR**: γ and β share `feature_lr` (one optimizer group). (Separate LRs would
  require two tensors; deferred — feature_lr is a reasonable default for both.)
- **PLY**: stored as `film_0 … film_24`; `load_ply` falls back to γ=1/β=0 init for a
  film run when the columns are absent.
- **Init**: `create_from_pcd` (fresh), `load_ply` (eval/resume), and the train.py
  post-warmup / `--init_ply` paths all (re)create `_film_params` at the current point
  count with γ=1/β=0.

## Method-switch coverage

FiLM is wired as a **cat-family** mode (blend-then-PyTorch-MLP), NOT the
`3D_SH_res`/`3D_SH_cat`/`3D_SH_32` CUDA-MLP family:

- `train.py`: in `--method` choices, `skip_methods` (skip SH base-color setup),
  `_freeze_methods` (+ `set_skip_mlp_grad` dispatch → `diff_surfel_film`), the
  post-warmup feature-init, and `--init_ply` re-init. **Excluded** from the
  3D_SH_res-only CUDA-setter lists (`set_contrib_thresh`, overdraw/weight reg,
  depth_sort, etc.) — those live in `diff_surfel_3D_sh_res`, not `diff_surfel_film`.
- `hash_encoder/modules.py`: `is_film_mode` flag, generic `else` MLP (`feat_dim =
  levels·dim = 24`), baseline `else` full-hashgrid build, an explicit `set_active_levels`
  branch (respects `disable_c2f`), and cat-like warm-up `optim_gaussian`.
- `gaussian_renderer/__init__.py`: guarded `diff_surfel_film` import, `is_film_mode`
  detection, a dedicated setup block (`render_mode=1`, hybrid=0, `shape_dims=[0,24,24]`,
  empty `colors_precomp`), dispatch branch, and `film_gamma`/`film_beta` kwargs. The
  24-channel output flows through the standard cat `rgb_decode` path.

## Usage

```bash
conda run -n nest_splatting python train.py -s <scene> -m <out> --method film [--disable_c2f true]
```

Apples-to-apples baseline: compare against `--method cat` with the same hash config
(FiLM conditioning vs. concatenation).

## Limitations / out of scope

- **MCMC / minimc depth-reinit / consolidate-primitives**: `_film_params` is threaded
  through standard + FastGS densify/clone/split/prune and the MCMC/minimc densify
  *postfix* calls, but NOT the deep relocation/`reinitial_from_depth`/consolidate
  tensor-rebuild paths. Use standard or FastGS densification with FiLM (like cat).
- **Baked rendering** (`diff_surfel_bake` / `diff_surfel_bake_render`): not yet
  supported for FiLM.
- **`.pth` checkpoint resume**: `_film_params` is not in the `capture()` tuple (to keep
  the length-based format detection intact); resume via `load_ply` instead.

---

# `--method 3D_SH_filmres` — FiLM on the 3D_SH_res residual MLP

## Concept

`3D_SH_res` renders `color = act( ReLU(SH + sh_bias) + MLP(H(x)) + res_bias )`, where the
**residual MLP is fused in-kernel** (view-independent, 16D hash input → 3D RGB) and the
per-Gauss SH provides the view-dependent base. `3D_SH_filmres` conditions that residual
MLP per surfel:

```
residual = MLP( γ_i · H(x_i) + β_i )          # γ scalar, β vector, per surfel
color    = act( ReLU(SH_i + sh_bias) + residual + res_bias )   # SH base + cascade UNCHANGED
```

Only the **hash feature feeding the residual MLP** is FiLM-modulated; the SH base color and
the activation cascade are untouched. Contrast with `--method film` (cat-family,
blend-then-PyTorch-MLP, view-dependent 24D hash): `3D_SH_filmres` is *decode-then-blend*
(MLP per Gauss in CUDA, then 2DGS blend) and keeps the per-Gauss view-dependent SH.

## Dimensions / storage

- The fused MLP input is `TC_INPUT_DIM = 16`, so the hash (hence β used) is **≤16D**:
  `hash_dim = active_hashgrid_levels × l_dim` (e.g. 4D at `--hybrid_levels 5`, 16D at
  `--hybrid_levels 2`). γ is a per-Gauss scalar.
- **Reuses the packed `_film_params [N,25]`** (col 0 = γ, cols 1:25 = β). The kernel reads
  γ and the first `hash_dim` of the 24 β channels (β **stride is 24**, the storage width);
  channels beyond `hash_dim` are unused. So `--method film` and `--method 3D_SH_filmres`
  share the same model storage, PLY round-trip, optimizer group (`feature_lr`), and the
  `--film_gamma_init` / `--film_beta_init` flags.

## CUDA submodule: `diff_surfel_3D_sh_filmres`

A fork of `submodules/diff_surfel_3D_sh_res`. The only kernel change is in the **scalar
case-5 path** (the in-kernel fused MLP):
- **Forward**: `mlp_input[i] = γ·hash_feat[i] + β[i]` before the MLP GEMM (`film_gamma ==
  nullptr` ⇒ byte-identical to 3D_SH_res).
- **Backward**: after `dL/d(mlp_input)` (`dL_dinput_full`) is computed, `dL/dβ[i] =
  dL_dinput[i]`, `dL/dγ = Σ_i hash[i]·dL_dinput[i]`, then `dL_dinput_full` is **γ-scaled in
  place** so the existing hash backward (`dL_dhash[i] = dL_dinput_full[i]` → `query_feature
  <true>`) automatically yields `dL/dH = γ·dL_dinput` and the γ-scaled xyz/geometry grad.
- **Both backward paths are FiLM-patched.** The backward enables a tensor-core
  collaborative-GEMM path by default; FiLM is implemented in **both** it and the scalar
  fallback (forward modulation, `my_hash_raw` persisted for `dL/dγ`, the `dL/dβ`/`dL/dγ`
  atomics, and the in-place γ-scale of `dL_dinput`). The collab `my_dL_dinput` was also
  bumped `[12]→[16]` (it previously capped 4-level hashes). Verified: collab and scalar
  give **identical** γ/β grads (≈1e-6 FP diff). *History:* the first cut only patched the
  scalar path and force-disabled collab — which left γ/β grad = 0 (silently frozen at init)
  while hash/MLP still trained; that's the symptom to watch for if collab ever regresses.
  In the collab path `global_id` is block-uniform (one Gauss per GEMM batch), so the
  `dL/dfilm_*` atomics correctly sum a Gauss's grad over all its pixels in the block.
- `film_gamma`/`film_beta` are threaded as dedicated per-Gauss tensors (+
  `dL_dfilm_gamma`/`dL_dfilm_beta` grad outputs) through the same signature chain as
  `colors`. Build like any submodule.
- **Perf: γ/β are staged in shared memory as FP16** in both the forward render and the
  backward render kernels (mirrors the `collected_colors[C*BLOCK_SIZE]` pattern in
  `diff_surfel_3D_16`/`mixed_3d`): one global read per Gauss per tile-batch instead of one
  per contributing pixel, freeing L1 for the hot hash-table reads. FP16 (vs FP32) halves
  the footprint — `collected_film_beta` is `__half[16*BLOCK_SIZE]` = 8 KB — which is what
  lets the **backward** fit it under the 48 KB static-shared cap alongside the collab-GEMM
  tiles (FP32 overflowed at ~54 KB). γ/β math is still FP32 (`__half2float` on read);
  the model params + gradients stay FP32. Verified: FP16-staged grads match the FP32-math
  path to ~1e-6 (within FP16 rounding), consistent with the already-FP16 hash/MLP.

## Wiring

`3D_SH_filmres` is a **member of the `is_3D_SH_res_mode` family** (so it inherits the fused
MLP build, SH base, residual modes, render_mode 5, and all train.py 3D_SH_res-family gates).
A dedicated `ingp.is_3D_SH_filmres_mode` flag drives:
- renderer dispatch to `diff_surfel_3D_sh_filmres` and the `film_gamma`/`film_beta` kwargs;
- **module-local device-global routing**: the setters (`set_mlp_weights`, `set_residual_mode`,
  `set_activation_bias`, `set_lru_slope`, …) and `get_mlp_grads` are per-module, so for
  filmres they target the filmres fork (renderer `_sh_res_setter_mod(ingp)`, train.py
  `_SHRES_SETTER_MOD`). **This is essential** — `set_mlp_weights` (upload) and `get_mlp_grads`
  (apply) on the wrong module would silently break the MLP.

## FiLM activation (`--film_act`)

A raw γ can go negative (flipping the sign of the hash feature), and the offset β can
go negative too. `--film_act` selects activations applied **independently** to γ and β
**before** they modulate the hash — `mlp_input = γ_act(γ)·H + β_act(β)`:
- `identity` (default): γ and β both raw (byte-identical to no flag).
- `gamma_relu`:    `max(0, γ)` → γ ≥ 0; **β raw**.
- `beta_relu`:     **γ raw**; `max(0, β)` → β ≥ 0 (offset can only add).
- `gamma_sigmoid`: `σ(γ)` → γ ∈ (0, 1); **β raw**.
- `beta_sigmoid`:  **γ raw**; `σ(β)` → β ∈ (0, 1).
- `double_relu`:   ReLU on **both** → γ ≥ 0 and β ≥ 0.
- `double_sigmoid`: σ on **both** → γ ∈ (0,1) and β ∈ (0,1).
- `gamma_sigm_split` (mode 7): `gamma_sigmoid` with a **separate γ per hash level** —
  `mlp_input[i] = σ(γ_l)·H[i] + β[i]`, `l = i/l_dim` (capped at 3); **β raw**. γ_0 is the
  classic γ (col 0 of `_film_params`); γ_1..3 ride in the UNUSED β cols 21..23
  (`_film_params` cols 22..24 — the fused-MLP input only uses β[0:hash_dim≤16]), so their
  gradients flow through the existing `dL_dfilm_gamma`/`dL_dfilm_beta` outputs with zero
  new plumbing. All four levels init at `--film_gamma_init`; `--lock_gamma` composes
  (locks ALL levels to X, all γ grads frozen); `--film_freeze_beta_iter` spares cols
  22:25 so the per-level γs keep training during the β freeze. The `[FILM iter=…]`
  diagnostic prints per-level `σ(γ_l)` means + per-level grad norms (a level with grad
  exactly 0 = the frozen-γ symptom). FD-gradcheck-verified via
  `scripts/test_filmres_sigm_split.py` (γ_0..γ_3 + β vs central FD, collab == scalar).

Implemented as the device-global `d_film_gamma_act` (modes 0–7; mirrors `d_lru_slope`): a
`set_film_gamma_act(mode)` setter patches fwd + bwd, called once at startup via the
`_SHRES_SETTER_MOD` routing. Separate `film_gamma_apply`/`film_beta_apply` (+ `_grad`)
device fns gate per param: **γ** relu for modes {1,5}, sigmoid for {3,6,7}; **β** relu for
{2,5}, sigmoid for {4,6}. Mode 7 additionally selects the per-level raw γ at every input
assembly / grad site (`film_gamma_raw_lvl`; the extra γs are staged in shared FP16
`collected_film_beta` slots 16..18, stage widened 16→19). The backward chain-rules
`dL/dγ` by `γ_act'(γ)` (scaling the hash gradient by `γ_act(γ)`) and `dL/dβ` by
`β_act'(β)`. Any mode reduces to `identity` on the affected param wherever that param is
already in the activation's identity region (all γ>0 for γ-relu, etc.). **3D_SH_filmres
only** (the cat-family `--method film` still uses raw γ/β — not yet wired there).

## Usage

```bash
... --method 3D_SH_filmres                                  # identity init (γ=1, β=0), raw γ/β
... --method 3D_SH_filmres --film_gamma_init 0.1 --film_beta_init 1.0   # bias toward β
... --method 3D_SH_filmres --film_act gamma_relu            # γ >= 0, β raw
... --method 3D_SH_filmres --film_act beta_relu             # γ raw, β >= 0
... --method 3D_SH_filmres --film_act gamma_sigmoid         # γ in (0,1), β raw
... --method 3D_SH_filmres --film_act gamma_sigm_split      # σ(γ_l) PER HASH LEVEL (4 γs), β raw
... --method 3D_SH_filmres --film_act double_relu           # γ >= 0 AND β >= 0
... --method 3D_SH_filmres --film_freeze_beta_iter 2000     # hold β at init for first 2k iters (γ trains)
... --method 3D_SH_filmres --lock_gamma 1.0                 # pin γ_eff=1 (no scale), pure additive latent H + β
```

## `--lock_gamma X` (pin the FiLM scale)

`--lock_gamma X` forces `γ_eff = X` (a constant, **bypassing both the stored per-Gauss γ
and its `--film_act` activation**) and **freezes γ's gradient**, leaving only the β shift:
`mlp_input = X·H + β_act(β)`. Device-global `d_film_lock_gamma` (default `-1e30` = off;
`film_gamma_apply` returns `X`, `film_gamma_apply_grad` returns 0); train.py also pins the
stored γ column to `X` so the saved PLY/diagnostics reflect it.

The motivating use is **`--lock_gamma 1.0`**: it removes the sigmoid-γ handicap (σ(γ)<1
permanently attenuates the hash vs. plain 3D_SH_res) and makes the model a *pure additive
latent on the full-strength hash*. With `--film_beta_init 0` it is **byte-identical to
3D_SH_res at init**, so any quality gap is attributable purely to what β learns — the clean
A/B test for "does the additive latent help at all." 3D_SH_filmres only.
The `[FILM iter=...]` diagnostic and the `training_output/{iter}_film_*` decomposition
renders fire for `3D_SH_filmres` too.

## Limitations

- MODE-5 collaborative GEMM stays disabled (scalar path only) — same as base 3D_SH_res today.
- Baked-rendering pipeline support is not wired for `3D_SH_filmres`.
