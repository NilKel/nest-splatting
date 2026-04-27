# NEST-Splatting Project Rules

## Environment
- **Conda environment**: `nest_splatting`
- Always use `conda run -n nest_splatting` for Python commands

## Build Commands

**CRITICAL: ALWAYS use `run_in_background: true` for ALL CUDA builds** — they take 2-5 minutes.

| Submodule | Path | Purpose |
|---|---|---|
| `diff-surfel-rasterization` | `submodules/diff-surfel-rasterization` | Original 2DGS rasterizer (modes 0/1) |
| `diff_surfel_3D_sh_res` | `submodules/diff_surfel_3D_sh_res` | **Main training rasterizer** for 3D_SH_res |
| `diff_surfel_bake` | `submodules/diff_surfel_bake` | Bakes MLP residual into per-Gaussian SH atlas |
| `diff_surfel_bake_render` | `submodules/diff_surfel_bake_render` | Forward-only renderer for baked atlas |
| `gridencoder` | `gridencoder` | Python-side hash encoding (not used at inference) |

Build any of them with:
```bash
cd <submodule-path> && conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

**Rebuild triggers**: `.cu`, `.cuh`, `.h` → rebuild. `.py` → no rebuild (editable install).

**Troubleshooting**:
- "externally-managed-environment" → use `python -m pip` (not `pip`)
- "No module named 'torch'" during build → add `--no-build-isolation`

## Architecture: 3D_SH_res

The primary rendering mode. **Color formula**:
```
color = clamp(SH + sh_bias, 0) + clamp(MLP_residual + res_bias, 0)
```
Defaults: `sh_bias=0.5`, `res_bias=0.0` (configurable at runtime via `set_activation_bias(sh, res)` — no rebuild needed). SH is unbounded; the residual rides on top to add high-frequency detail. Both components are evaluated in CUDA inside `diff_surfel_3D_sh_res`.

**Components**:
- **SH** (Lagrangian, attached to Gaussians): handles low-frequency color + view-dependence
- **Hashgrid + MLP residual** (Eulerian, fixed grid): handles high-frequency spatial detail
- View encoding: 16D (SH degree 4, from tcnn `_encode_view`)

**Hybrid levels** (`--hybrid_levels N`): split between per-Gaussian features and hash levels, with 6 total levels at 4D each:
- `hybrid_levels=5` → 5 per-Gauss (20D) + 1 hash (4D, finest)
- `hybrid_levels=2` → 2 per-Gauss (8D) + 4 hash (16D)

Hash levels are selected **finest-first** from `[128, 203, 322, 512]`. Hash table: 2^19 entries × 4D, voxel range `[-1.5, 1.5]`.

**Coarse-to-fine** (multi-level hash only): coarsest hash level always on, one finer added every 2000 iters. Single-level (hybrid=5) is always fully active.

## 3D_SH_cat (alternative)

`MLP(sum(w_i * f_i))` — accumulate features then evaluate MLP once per pixel. Render mode 1 in `diff-surfel-rasterization`. Same hybrid_levels semantics. Less expressive than 3D_SH_res; kept for comparison.

## Baked Rendering Pipeline

Goal: bake the MLP residual into static per-Gaussian SH textures for fast inference.

**Pipeline**:
1. **Bake** (`diff_surfel_bake`): evaluate MLP at 8×8 UV grid per Gaussian → 48D SH residual per texel
2. **Render** (`diff_surfel_bake_render`): forward-only 2DGS rasterizer, samples atlas via texture lookup
3. **Format**: mean SH stored in PLY; 48D residual textures `[N, 8, 8, 48]` FP16 in atlas

**Scripts**:
- `scripts/benchmark_baked.py` — neural vs baked-SH-only vs baked-SH+atlas timings + PSNR/SSIM/LPIPS
- `scripts/render_baked.py` — save baked renders to disk (no benchmarking)

**Reference quality** (chair scene):
| Mode | PSNR | SSIM |
|---|---|---|
| Neural renderer (training) | 34.66 | 0.9844 |
| SH + 48D residual (baked) | 34.01 | 0.9814 |
| SH only (baked mean) | 33.23 | 0.9757 |

**Pitfalls** (all bitten in past sessions):
- **Scene() overwrites baked PLY**: `Scene()` constructor loads training PLY. Always reload baked PLY after `Scene()` init.
- **UV conventions**: bake kernel must use texel-center `(i+0.5)*step - extent`, NOT endpoint-inclusive `/(N-1)`.
- **SH layout**: MLP outputs channel-first `[R0..R15, G0..G15, B0..B15]`; PLY stores interleaved `[N, 16, 3]`.
- **Default budget clamps resolution**: `--atlas_budget_mb 2048` (default) silently clamps `max_res` from 128 down to 16 on large scenes. Bump to 8192 for full-res bakes (`--max_res 64 --atlas_budget_mb 8192` is the path-2 standard).

## Key Files

- `gaussian_renderer/__init__.py` — `render()` dispatch
- `hash_encoder/modules.py` — INGP class (hash + MLPs), tcnn weight loading
- `scene/gaussian_model.py` — `GaussianModel`
- `train.py` — training loop
- `submodules/diff_surfel_3D_sh_res/cuda_rasterizer/{forward,backward}.cu` — main CUDA path
- `submodules/diff_surfel_bake_render/cuda_rasterizer/forward.cu` — bake render forward

## CLI Flags

### Activation & Bias
- `--activation_bias SH RES` — defaults `0.5 0.0`
- `--freeze_sh` — permanently zero SH LR
- `--sh_freeze_iter N` — freeze SH LR for first N iters
- `--freeze_mlp` — freeze MLP weights

### Hash/MLP LRs
- `--res_lr_scale X` — scale hash + MLP LR
- `--hash_lr_scale X` — scale hash LR only (stacks)
- `--res_warmup N` — disable hash/MLP residual for first N iters

### Regularization
- `--overdraw_reg λ` — sigmoid-based overdraw penalty (CUDA grad)
- `--w_overdraw_reg λ` (`--w_overdraw_gamma γ`) — error-guided overdraw
- `--weight_reg λ` — weight-squared consolidation, penalizes `1−sum(w²)` (CUDA grad)
- `--w_weight_reg λ` (`--w_weight_gamma γ`) — error-guided weight-squared
- `--w_normal λ` (`--w_normal_gamma γ`) — error-guided normal consistency
- `--contribution_thresh T` — skip hash query when `α·T < T`
- `--count_thresh N` — skip hash query past N contributors per pixel

**Error-guided pattern**: `w(r) = exp(-γ * MSE(r).detach())`. Critical: `.detach()` the error, otherwise the optimizer sabotages rendering quality to lower reg loss. Pick γ so `exp(-γ * avg_MSE) ≈ 0.5` mid-training.

### MCMC
- `--mcmc_fps`, `--cap_max N`, `--noise_lr X` (default 5e5)
- `--mcmc_depth_reinit N`, `--reinit_interval N`, `--reinit_end N`
- `--opacity_reg λ` (auto-set to 0.01 under `--mcmc`), `--scale_reg λ`

### Training schedules
- `--mini` — Mini-Splatting v2: aggressive clone (every 250) + densification (every 100), depth reinit at 2000, simp1 at 3000 (keep 60%), simp2 at 8000 (keep top 99%).
- `--mini1` — Mini-Splatting v1: repeated depth reinit (5k/10k/15k), longer densification (500→15000), simp1 at 15000 (keep 50%), simp2 at 20000. Auto-enables `--mini`.
- `--minimc` — Sweep-based RJ-MCMC. Requires `--method 3D_SH_res`. Two events on separate intervals: depth reinit (every `--minimc_reinit_interval`, in-place via `reinitial_alive_inplace`) and sweep+cull+clone (every `--minimc_relocate_interval`). Tensor size is fixed; alive/dead split moves inside `cap_max`. See `gaussian_model.py` for `minimc_sweep_and_relocate`.
- `--minispa` — chains mini v2 aggressive clone + silhouette-aware depth reinit at 2000 → GSpa Phase-2 ADMM (3000→25000) → hard prune.

### Other
- `--init_ply PATH` — initialize Gaussians from external PLY
- `--cold` — skip 2DGS warmup, train from scratch with hash-in-CUDA from iter 1
- `--bce_solo_adaptive --bce_iter N` — opacity binarization in last N iters

## Hybrid Architecture Insights

**Lagrangian vs Eulerian clash**: any operation that destroys/recreates Gaussians (depth reinit, aggressive prune) breaks the hash's spatial correspondence, while SH just relearns locally. Mini-Splatting depth reinit is designed for pure 3DGS; under hybrid, the hash loses learned spatial features. Use higher reinit opacity (0.5 vs 0.1) to keep hash gradients alive during structural changes.

**Hash gradient magnitude**: `dL/dfeat ∝ T_i · α_i`. Tiny by default (~1e-5 norm for 2M params). Adam normalizes per-parameter so updates are still ~lr-sized. With 1 hash level (4D), the hash has limited capacity — MLP does most of the work.

**MCMC noise on 2DGS surfels**: standard MCMC builds 3D covariance with `[scale_x, scale_y, 1]`. For tiny surfels (`scale=0.01`), normal noise=1 is 100× larger than in-plane noise → blasts surfels off the surface. Use `max(scale_x, scale_y)` for normal axis, or 0 for pure in-plane jitter.

**SH +0.5 bias note**: `diff_surfel_3D_sh_res` keeps the +0.5 bias (configurable at runtime). Removing it forces SH to start at black, giving hash equal footing — but typically converges slower without explicit upside.

## `out_index` / `max_contrib_idx`

Per-pixel max-contributor Gaussian id, written by `diff_surfel_3D_sh_res` only.
- CUDA: in `forward.cu`, tracked alongside `depth_max_contributor` via `if (w > max_w)`. Both WMMA and scalar paths populate.
- Python: `_RasterizeGaussians.forward` returns a 9-tuple, exposed as `render_pkg['max_contrib_idx']`. Other rasterizers return `None`.
- Consumers: `mini_depth_reinit` SH transfer (looks up `src_features_dc/_rest[max_idx]` per sampled pixel; falls back to `RGB2SH(GT_pixel)` when invalid), debug visualization (`_colorize_max_contrib_idx`).

## Known Issues & Pitfalls

- **`_appearance_level = 0` silently disables hash**: `hashgrid.h` does `max_level = min(ap_level, L)`. `ap_level=0` → 0 hash levels queried, hash features all zero, hash grads exactly zero. `create_from_pcd` correctly sets `ap_level=24` (sentinel for "all levels active"). Any code that creates/reinitializes Gaussians (`reinitial_from_depth`, prune+realloc) MUST set `_appearance_level=24`.

- **MODE 5 vs standard backward**: `diff_surfel_3D_sh_res` has separate code for the collaborative-GEMM path (mode 5) and the standard backward. When adding kernel features (e.g. beta kernel), patch BOTH — alpha computation, `dL_dshapes`, `dG_factor` (rho3d), `dG_factor_2d` (rho2d). Past silent bug: zero gradients for beta kernel under mode 5.

- **tcnn weight layout**: tcnn pads input dim to multiples of 16 (`40→48`); padding columns are filled with 1s, giving an implicit bias `b1 = sum(W1[:, 40:48], dim=1)`. tcnn has no explicit biases → zero `b2`, `b3`. Handled by `_copy_tcnn_to_pytorch_mlp()` in `hash_encoder/modules.py`.

- **Renderer level encoding**: `gaussian_renderer/__init__.py` uses `ingp.hashgrid_levels` (not `ingp.levels`) for `(total << 16) | (active << 8) | hybrid`. Wrong field → CUDA kernel queries non-existent levels → zeros/garbage.

- **Don't reset opacity under `--mini`/`--mini1`**: MSv2 never resets opacity; doing so destabilizes the importance-prune pipeline.

## Config

`configs/nerfsyn.yaml`: `lambda_normal=0`, `lambda_dist=0`, `lambda_mask=0.1`, `mask_iter=10k`. `tg_beta`, `tg_base_alpha` for surfel parameters.
