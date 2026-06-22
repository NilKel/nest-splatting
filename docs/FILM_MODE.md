# `--method film` — Feature-wise Linear Modulation (FiLM)

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
