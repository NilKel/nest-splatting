# NEST-Splatting Project Rules

## Environment

- **Conda environment**: `nest_splatting`
- Always use `conda run -n nest_splatting` for Python commands

## Build Commands

**CRITICAL: ALWAYS use `run_in_background: true` for ALL CUDA builds** - they take 2-5 minutes. NEVER run builds in the foreground.

### Diff Surfel Rasterization (main CUDA rasterizer)
```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

### Diff Surfel 3D (lean library for 3D_direct_fused — FASTER BUILD)
```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff_surfel_3D
conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```
Stripped-down rasterizer with only render_mode 0, 3, and 5. Use for faster iteration on 3D_direct_fused.

### Grid Encoder
```bash
cd /home/nilkel/Projects/nest-splatting/gridencoder
conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

### Rebuild Triggers
- `.cu`, `.cuh`, `.h` files → **requires rebuild**
- `.py` files → **no rebuild needed** (editable install)

### Troubleshooting
- "externally-managed-environment": Use `python -m pip` not `pip`
- "No module named 'torch'" during build: Add `--no-build-isolation`

## Rendering Modes

| Mode | render_mode | Hash Location | MLP Location | Per-Gaussian Features |
|------|-------------|---------------|-------------|----------------------|
| baseline | 0 | CUDA | Python (tcnn) | None |
| cat | 1 | CUDA | Python (tcnn) | hybrid_levels × 4D |
| 3D_direct | 3 | Python (gridencoder) | Python (PyTorch MLP) | hybrid_levels × 4D |
| 3D_direct_fused (lean) | 5 | CUDA (query_feature) | CUDA (constant memory) | hybrid_levels × 4D |

### Mode Paradigms
- **cat**: `MLP(sum(w_i * f_i))` — accumulate features, one MLP eval per pixel
- **3D_direct / 3D_direct_fused**: `sum(w_i * MLP(f_i))` — MLP eval per intersection, accumulate RGB

## Architecture: 3D_direct_fused (lean)

### MLP
- Input: `[gauss_feat(20D) | hash_feat(4D) | view_enc(16D)]` = 40D
- Hidden: 32 neurons × 2 layers (ReLU)
- Output: 3D RGB (sigmoid)
- Weights stored in CUDA global memory, uploaded via `set_mlp_weights()`
- Weight layout: PyTorch `[out, in]` = CUDA row-major `W[h * in_dim + i]` — NO transpose needed

### Forward Pass (forward.cu case 5)
1. Compute xyz intersection point from surfel parametrization
2. Get per-Gaussian features from `colors_precomp` (20D coarse)
3. Query hash at xyz via `query_feature` (4D fine)
4. Load pre-encoded per-pixel view direction (16D, from `viewdirs_enc` tensor)
5. Concatenate → MLP → sigmoid → RGB
6. Accumulate: `pixel += alpha * T * rgb`

### Backward Pass (backward.cu case 5, collaborative GEMM)
1. Recompute MLP forward to get h1_pre, h1_post, h2_pre, h2_post, output
2. `dL_dout = dL_dpixel * w` where `w = alpha * T`
3. `compute_dL_dz_all`: sigmoid' → dL_dz3 → W3 backprop → dL_dz2 → W2 backprop → dL_dz1
4. `collaborative_mlp_backward_all`: sub-tile GEMM (2×128 batches) for weight gradients
   - Shared memory: load dL_dz and activations, threads cooperate on matmul
   - Tile accumulators in `__shared__`, flushed to global with atomicAdd at end
5. `mlp_backward_input_only`: dL_dinput for feature/hash/geometry gradients
6. Hash backward via `query_feature<true>` → dL_dxyz → dL_duv → geometry

### Gradient Flow (3D_direct mode)
- RGB gradients: `IntersectionOpacityGrad` custom autograd
- Mask loss gradients: Native CUDA backward (render_alpha path)
- Key kernels: `backward_from_weight_grad_kernel`, `transMat_to_scale_rot_grad_kernel`

### MLP Weight Loading (tcnn checkpoint → PyTorch)
- tcnn pads input dim to multiples of 16: `40 → 48`
- Padding columns filled with 1s → implicit bias: `b1 = sum(W1[:, 40:48], dim=1)`
- tcnn has NO explicit biases → zero b2, b3
- `_copy_tcnn_to_pytorch_mlp()` handles this for `mlp_fused`
- `_copy_tcnn_to_pytorch_mlp_3D_direct()` handles this for `mlp_3D_direct`

## Key Files

### Python
- `gaussian_renderer/__init__.py` — Main `render()` function, mode dispatch
- `hash_encoder/modules.py` — INGP (hash encoding + MLPs), weight loading
- `scene/gaussian_model.py` — GaussianModel class

### CUDA (diff_surfel_3D — lean library)
- `forward.cu` — Forward rendering kernel (case 5 = 3D_direct_fused)
- `backward.cu` — Backward kernel (case 5 = collaborative GEMM path)
- `modes/mode_3d_direct_fused.cu` — MLP helpers: `mlp_forward_inline`, `compute_dL_dz_all`, `collaborative_mlp_backward_all`
- `hashgrid.h` — `query_feature` for CUDA hash encoding
- `rasterize_points.cu` — Python/CUDA bindings (`set_mlp_weights`, `get_mlp_grads`)
- `diff_surfel_3D/__init__.py` — Python interface

### Test Scripts
- `scripts/test_3d_lean_vs_3d_direct.py` — Compare 3D_direct vs 3D_lean gradients

## Feature Dimensions

With `--hybrid_levels 5` and 6 total levels:
- Per-Gaussian: 5 × 4D = 20D (coarse)
- Hashgrid: 1 × 4D = 4D (fine, resolution 512)
- View encoding: 16D (SH degree 4, from tcnn `_encode_view`)
- MLP input: 40D total
- Hash table: 2^19 entries × 4D, voxel_range [-1.5, 1.5]

## Config Reference

`configs/nerfsyn.yaml`:
- lambda_normal=0, lambda_dist=0, lambda_mask=0.1, mask_iter=10k
- tg_beta, tg_base_alpha for surfel parameters

## CLI Flags Reference (3D_SH_res / 3D_SH_cat / 3D_SH_32)

### Activation & Bias
- `--activation_bias SH_BIAS RES_BIAS` — Set SH and residual biases. `color = ReLU(ReLU(SH+sh_bias) + residual+res_bias)`. Default: `0.5 0.0`
- `--freeze_sh` — Permanently zero SH LR (f_dc and f_rest)
- `--sh_freeze_iter N` — Freeze SH LR for first N iterations, then unfreeze
- `--freeze_mlp` — Freeze MLP weights (random init or from `--freeze_mlp_from`)

### Hash/MLP Learning Rates
- `--res_lr_scale X` — Scale both hash encoding and MLP LR by X (e.g., 0.1 = 10x lower)
- `--hash_lr_scale X` — Scale hash encoding LR only (stacks with `--res_lr_scale`)
- `--res_warmup N` — Disable hash/MLP residual for first N iterations (SH-only warmup)

### Overdraw / Sparsity Regularization
- `--overdraw_reg λ` — Sigmoid-based overdraw penalty (CUDA gradient). Penalizes per-pixel contributor count
- `--w_overdraw_reg λ` — Error-guided overdraw (Python). Relaxes where MSE is high. `--w_overdraw_gamma γ`
- `--weight_reg λ` — Weight-squared consolidation (CUDA gradient). Penalizes `1-sum(w²)`. Forces single opaque surface per pixel. Gradient only affects opacity, not geometry.
- `--w_weight_reg λ` — Error-guided weight-squared (dynamic CUDA lambda). `--w_weight_gamma γ`. Lambda = `λ * exp(-γ * mean_MSE)` updated each iteration.
- `--w_normal λ` — Error-guided normal consistency. `--w_normal_gamma γ`
- `--contribution_thresh T` — Skip hash query when `w = alpha*T < T` (saves compute on tail Gaussians)
- `--count_thresh N` — Skip hash query after N contributing Gaussians per pixel

### MCMC
- `--mcmc_fps` — MCMC with FPS subsampling
- `--cap_max N` — Max Gaussian count
- `--noise_lr X` — SGLD noise magnitude (default 5e5)
- `--mcmc_depth_reinit N` — One-shot depth reinit at iteration N (0 = disabled)
- `--reinit_interval N` — Repeat depth reinit every N iters (0 = once only)
- `--reinit_end N` — Stop repeating reinit after iteration N (-1 = total_iter - 15000)
- `--opacity_reg λ` — L1 opacity regularization (auto-enabled to 0.01 for MCMC)
- `--scale_reg λ` — L1 scale regularization

### Mini-Splatting v2 (`--mini`)
- `--mini` — Enable MSv2 mode (importance pruning, depth reinit, aggressive clone)
- `--mini_depth_reinit_iter N` — Depth reinit iteration (default 2000)
- `--mini_simp_iter1 N` — First simplification (default 3000)
- `--mini_simp_iter2 N` — Second simplification (default 8000)
- `--mini_densify_until N` — Stop densification (default 3000)
- `--mini_clone_interval N` — Aggressive clone interval (default 250)
- `--mini_sampling_factor F` — Simp1 keep fraction (default 0.6)
- `--mini_imp_metric indoor|outdoor` — Importance metric
- `--mini_late_prune_interval N` — Post-simp2 low-opacity prune cadence (default 500; 0 = off)
- `--mini_late_prune_thresh X` — Post-simp2 prune threshold (default 0.005)
- `--mini_warmup` — Camera warmup (0.5× resolution during densification phase)

### Mini-Splatting v1 (`--mini1`)
- `--mini1` — Enable v1 schedule: repeated depth reinit, longer densification, later simplifications. Auto-enables `--mini` internally (reuses v2 machinery with overridden defaults).
- `--mini1_depth_reinit_interval N` — Interval between repeated depth reinits (default 5000)
- `--mini1_depth_reinit_until N` — Stop repeated reinits at this iter (default 15000)
- Auto-overrides (unless explicitly set by user): `mini_depth_reinit_iter=5000`, `mini_simp_iter1=15000`, `mini_simp_iter2=20000`, `mini_densify_until=15000`, `mini_sampling_factor=0.5`
- **Aggressive clone**: now enabled for both `--mini` and `--mini1` (used to be gated off for mini1). Under mini1 it's automatically skipped on every repeated-reinit iter.

### MiniSpa — Mini v2 aggressive clone + silhouette reinit → GSpa ADMM (`--minispa`)
Auto-enables `--mini` and `--gspa` and chains their phases:
- 0 → `minispa_reinit_iter` (2000): mini v2 aggressive cloning every `mini_clone_interval` iters. Standard densify_and_prune is disabled (mini does its own growth).
- `minispa_reinit_iter`: one depth reinit using the loose silhouette-aware scale cap (distance transform of depth/alpha edges → per-pixel safe-radius in world units, clamps the NN-distance scale init). SH is transferred from the old max-contributor per pixel via `max_contrib_idx`.
- `minispa_reinit_iter` → `minispa_admm_start` (3000): settle phase, nothing structural.
- `minispa_admm_start` → `minispa_admm_stop` (25000): GSpa Phase-2 ADMM penalty active. z/u update every `gspa_interval` iters (default 50). Tensor size is flat.
- `minispa_admm_stop`: GSpa hard prune to target count.
- `minispa_admm_stop` → end: pure Adam refinement.

Key flags:
- `--minispa` — enable the mode
- `--minispa_reinit_iter N` — silhouette reinit iter (default 2000)
- `--minispa_admm_start N` — GSpa ADMM start (default 3000)
- `--minispa_admm_stop N` — GSpa ADMM stop + final hard prune (default 25000)
- `--gspa_target_count N` — final Gaussian count target (auto-splits keep ratio across Phase 1 and 2; under minispa Phase 1 is skipped so the full keep ratio lands on ADMM)
- `--gspa_rho X` — ADMM penalty strength (default 5e-4)
- Mini simp1/simp2 and GSpa Phase 1 importance-prune are both gated off under `--minispa`.

Implementation notes:
- `mini_depth_reinit(..., compute_safe_radius=True)` computes `safe_radius_world = dist_px * depth / focal` per sampled pixel and adds `'safe_radius'` to the returned dict. Boundary mask = depth discontinuity (`rel_depth_thresh=0.05`) OR alpha edge (`alpha < 0.01`). Normal-crease edges are disabled by default (blows up on curved objects).
- `reinitial_from_depth` now clamps `nn_dist = min(nn_dist, safe_radius)` when the key is present — on top of the existing `min(nn_dist, footprint_scale_cap × pixel_footprint)` cap.
- `gspa_simp_iter` under minispa is parked at `minispa_reinit_iter` purely as the SH-degree-increase anchor; the Phase-1 dispatch at that iter is gated off with `and not args.minispa`.
- The LR shift `update_learning_rate(iter - gspa_simp_iter + 5000)` is gated off under minispa so the reinit's own `reset_xyz_lr_schedule` isn't double-shifted.

### MiniMC — Sweep-based RJ-MCMC (`--minimc`)
New (2026-04) closed-loop design. Deprecated v1 3-stage schedule is kept behind `if False:` in [train.py](train.py) for reference.
- `--minimc` — Enable MiniMC. Auto-enables `--mcmc` for SGLD noise + dead-pool semantics. Requires `--method 3D_SH_res` (needs `max_contrib_idx` from the rasterizer's `out_index` slot).
- `--minimc_start_iter N` — Don't fire cull/relocate/reinit before this iter (default 500)
- `--minimc_relocate_interval N` — Sweep + cull + clone fires every N iters (default 100)
- `--minimc_reinit_interval N` — Depth reinit fires every N iters (default 0 = disabled). Preempts relocate on coincident iters.
- `--minimc_reinit_until N` — Stop depth reinits at this iter (default 20000)
- `--minimc_relocate_until N` — Stop relocate at this iter (default 25000)
- `--minimc_dead_thresh X` — Opacity ≤ this is "dead" (default 0.005, same as vanilla MCMC)
- `--minimc_no_single_view_cull` — Don't cull Gaussians with `count_vis ≤ 1`
- `--minimc_low_imp_cdf F` — CDF keep-threshold for low-importance cull. Keeps top-X cumulative importance mass, culls the rest (default 0.999). Also doubles as the clone-candidate pool threshold.
- Deprecated aliases (silently remapped): `--minimc_cull_interval → --minimc_reinit_interval`, `--minimc_cull_until → --minimc_reinit_until`, plus `--minimc_reinit_iter`, `--minimc_simp_iter1/2`, `--minimc_sampling_factor` (all ignored now)

### Initialization
- `--init_ply PATH` — Initialize Gaussians from external PLY instead of dataset point cloud
- `--cold` — Skip 2DGS warmup, train from scratch with hash_in_CUDA from iter 1

### BCE / Opacity Binarization
- `--bce_solo_adaptive` — Binary cross-entropy on opacity in last `--bce_iter` iterations
- `--bce_iter N` — Duration of BCE phase before end of training

## Known Issues Fixed

1. **Post-10k divergence**: `diff_surfel_rasterization/__init__.py` was zeroing `grad_opacities` in render_mode=3. Fixed by removing ALL geometry gradient zeroing.

2. **Normal gradient path**: `transMat_to_scale_rot_grad_kernel` needs `dL_dnormal3D` + viewmatrix.

3. **Hash gradient flow**: `dL_duv` connects to ray-disk Jacobian.

4. **tcnn weight layout**: Pads input to multiples of 16, padding columns = 1 (implicit bias). NO explicit biases.

5. **Missing geometry gradients in 3D_direct_fused backward**: Was only writing `dL_dtransMat[6..8]` (Tw). Fixed by adding full Tu, Tv, Tw computation from cat mode reference.

## 3D_SH_res Activation Combos (Experimental Log)

SH_color = `computeColorFromSH()` = `clamp(SH_eval + sh_bias, 0)` (configurable via `set_activation_bias`).
MLP residual = identity output (unbounded, no sigmoid on MLP output layer).

| # | Forward activation | Backward | Quality (chair) | Notes |
|---|---|---|---|---|
| 1 | `ReLU(SH) + residual` (unbounded residual) | identity for res, clamp for SH | baseline | can go negative from residual |
| 2 | `sigmoid(ReLU(SH) + residual)` | sigmoid derivative | slightly worse than #1 | double-activation squashes range |
| 3 | `ReLU(ReLU(SH) + residual)` | ReLU derivative on sum | doesn't work with MCMC | outer ReLU can zero SH+res |
| 4 | `sigmoid(raw_SH + residual)` | sigmoid derivative, no SH clamp | **tested** | no ReLU anywhere |
| 5 | `ReLU(SH+b) + ReLU(residual+b)` | decoupled ReLU | **testing** | negative res zeroed, can't cancel SH |
| 6 | `ReLU(ReLU(SH+b) + residual + b)` | outer ReLU on sum | **testing** | residual CAN be subtractive |

Standard 3DGS baseline (no residual): `clamp(SH_eval + 0.5, 0)` — ReLU activation with 0.5 bias.

### Bias Tuning Results (configurable via `set_activation_bias`)
- Biases now configurable at runtime: `set_activation_bias(sh_bias, res_bias)` — no CUDA rebuild needed
- Setting SH bias to 0 makes hashgrid much more active but reconstruction becomes foggy
- Increasing hash/MLP bias doesn't fix the fogginess — the issue is SH can't provide base color without bias
- The decoupled `ReLU(SH+b) + ReLU(residual+b)` method works well but MCMC has convergence issues
- SH bias=0.5 (standard 3DGS) is most stable for SH; residual bias controls hash contribution floor

### SH Bias Removal (3D_SH_res only)
Standard 3DGS uses `color = clamp(SH_eval + 0.5, 0)` — the +0.5 bias means zero-initialized SH produces gray (0.5).
In `diff_surfel_3D_sh_res`, we removed the +0.5 bias so SH starts at black (0). Rationale:
- SH handles low-frequency color + view-dependence
- Hash MLP residual handles high-frequency spatial detail + base brightness
- With bias, SH "starts ahead" at gray and captures most detail before hash can contribute
- Without bias, SH and hash start equal, forcing better division of labor
- Changed in: `forward.cu` lines 116 (`computeColorFromSH`) and 130-132 (`eval_sh_inline`)
- Backward unaffected (bias is constant, zero gradient)
- DC init in `reinitial_from_depth`: `f_dc = 0` → black start (was gray with bias)

## Current Status: 3D_direct_fused Backward

### What Works
- Forward pass: mean diff 0.0004 vs 3D_direct (most pixels match)
- b3 gradient: cos_sim=0.999 (correct direction, ~2x magnitude off)
- Geometry gradients: full Tu, Tv, Tw path implemented

### Known Issues
- **Forward mismatch at specific pixels**: max diff 0.67 at edge pixels. Likely hash encoding boundary differences between Python `gridencoder` and CUDA `query_feature`
- **W1/W2/b1/b2 gradients ~1000x too large in CUDA**: Root cause unclear. b3 is close, suggesting dL_dz3 is correct, but backprop through hidden layers amplifies error
- **W3 gradient direction off** (cos_sim=0.15): Despite b3 being correct, W3 = dL_dz3.T @ h2 is wrong, implying h2 values differ (from forward mismatch)
- **b3 magnitude ~2x off**: 15433 (PyTorch) vs 8336 (CUDA)

### Root Cause Hypothesis
The forward pass pixel mismatches cause cascading gradient errors. If MLP inputs differ at some pixels (due to hash encoding differences), then h1/h2 differ, and weight gradients diverge. Fix the forward pass first, then re-evaluate backward.

## Known Issues Fixed (Appearance Level / Hash Query)

6. **`_appearance_level = 0` silently disables hash encoding**: In `hashgrid.h`, `max_level = min(appearance_level, L)`. If `ap_level=0`, zero hash levels are queried, hash features are all zeros, and hash gradients are exactly zero. This is a silent failure — no error, just dead hash parameters. `create_from_pcd` correctly sets `ap_level=24` (sentinel for "all levels active"). Any code that creates or reinitializes Gaussians (e.g., `reinitial_from_depth`, pruning+realloc) MUST set `_appearance_level` to 24, not 0. Symptom: `[BW_DBG] grad_features: nonzero=0` and `[HASH_DBG] hash_encoding: grad norm=0.000000`.

## Hybrid Architecture Insights (SH + Hashgrid)

### Lagrangian vs Eulerian Feature Clash
- **SH** = Lagrangian (attached to moving Gaussians, moves with them)
- **Hashgrid** = Eulerian (fixed spatial grid, doesn't move)
- Any operation that destroys/recreates Gaussians (depth reinit, aggressive pruning) breaks the hash's spatial correspondence while SH just relearns locally
- Mini-Splatting v2's depth reinit is designed for pure 3DGS — applying it to hybrid architectures causes the hash to lose all learned spatial features

### Hash Gradient Magnitude
- Hash encoding gradients in 3D_SH_res are inherently tiny (~1e-5 norm for 2M params)
- This is normal: gradients flow through 3 MLP layers + trilinear interpolation + atomicAdd scattering
- Adam normalizes per-parameter, so tiny but consistent gradients still produce ~lr-sized updates
- Hash values drift (std grows over training) but convergence is slow relative to SH/MLP
- With only 1 hash level (4D), the hash has limited capacity — MLP does most of the work

### Opacity and Hash Gradient Coupling
- Hash gradient magnitude is proportional to blending weight: `dL/dfeat ∝ T_i * alpha_i`
- Low opacity (e.g., 0.1 after depth reinit) → starved hash gradients → hash can't keep up with geometry changes
- MSv2 uses opacity 0.1 for reinit (fine for pure 3DGS), but for hybrid architectures higher opacity (0.5) gives stronger hash gradients
- Opacity LR: MSv2 halves to 0.025 for stability during aggressive structural changes

### MCMC Noise Injection for 2DGS Surfels
- Standard MCMC builds 3D covariance with `[scale_x, scale_y, 1]` for normal axis
- For tiny surfels (scale=0.01), normal noise=1 is 100x larger than in-plane noise → blasts surfels off their surface
- Fix: use `max(scale_x, scale_y)` for normal axis noise, or 0 for pure in-plane jitter
- `build_scaling_rotation` takes `[scale_x, scale_y, normal_scale]` × rotation

### SH +0.5 Bias Problem
- Standard 3DGS: `color = clamp(SH_eval + 0.5, 0)` — gray at init, SH "starts ahead"
- In hybrid mode, SH quickly captures most appearance because it starts at gray while hash starts at zero
- Removing the +0.5 bias forces SH to start at black, giving hash equal footing
- Without bias: SH must learn positive DC to produce any color, hash residual adds on top
- Trade-off: may need more iterations for SH to converge on base color

### Weighted Regularization (Error-Guided Relaxation)
- Pattern: `w(r) = exp(-gamma * MSE(r).detach())` per pixel
- Where RGB error is high (complex regions): regularization relaxed → network free to use complex geometry
- Where RGB error is low (well-learned): regularization at full strength → enforce smoothness/sparsity
- CRITICAL: `.detach()` the error — without it, optimizer sabotages rendering quality to lower reg loss
- Applied to: overdraw regularization (`--w_overdraw_reg`), normal consistency (`--w_normal`)
- Gamma tuning: choose so `exp(-gamma * avg_MSE) ≈ 0.5` mid-training

## Mini-Splatting v2 Integration

### Timeline (default hyperparameters)
| Iter | Event |
|------|-------|
| 0 | SH locked at degree 0 |
| 500-2750 | Aggressive clone every 250 + densification every 100 |
| 2000 | Pre-reinit prune (top 99%) → depth reinit |
| 3000 | Simp1 (importance sampling, keep 60%) + densification stops + SH unlocks |
| 4000-6000 | SH degree increases (1→2→3) |
| 8000 | Simp2 (intersection preserving, keep top 99%) |
| 8001+ | Pure optimization |

### Key Implementation Details
- Importance computed by full eval loop over all training views (not per-frame tracking)
- `transmittance_avg` in diff_surfel_3D_sh_res IS `accum_weights` (sum of alpha*T per Gaussian)
- `factor_culling = count_vis / (count_rad + 0.1)` scales densification gradients
- No opacity reset when `--mini` is active (MSv2 never resets opacity)
- Densify and aggressive clone both skip at `depth_reinit_iter`
- `mask_blur`: Gaussians covering > H×W/5000 pixels get force-split

### Differences from Original MSv2
- No per-view `_culling` mask (we render all Gaussians every time)
- No SparseGaussianAdam (using standard Adam)
- No camera warmup (0.5x+1x resolution mixing)
- Hash/MLP resets at depth reinit (not in original — they're pure 3DGS)

## Mini-Splatting v1 (`--mini1`)

v1 fundamentally different from v2: repeated depth reinit + long densification, no simp1/simp2 concentration. Implemented as a variant of `--mini` that auto-enables `args.mini = True` and overrides schedule defaults. All v2 machinery (`mini_culling_with_clone`, `mini_intersection_preserving`, etc.) is reused — the only behavioral differences are scheduling.

### Timeline (default hyperparameters, 30k iters)
| Iter | Event |
|------|-------|
| 0 → 500 | Warmup (no densification) |
| 500 → 15000 | Standard `densify_and_prune` every 100 iters + aggressive clone every 250 iters |
| 5000 / 10000 / 15000 | Repeated depth reinit (resets xyz LR schedule, transfers SH via max-contributor id) |
| 15000 | Simp1 (importance sampling, keep 50%) + SH unlocks |
| 16000 → 18000 | SH degree increases (1→2→3) |
| 20000 | Simp2 (intersection preserving, keep top 99%) |
| 20001+ | Pure optimization |

### Key Differences vs `--mini` (v2)
- Depth reinit repeats every 5k iters instead of one-shot at 2k
- Densification window 500→15000 instead of 500→3000
- Simplifications pushed to 15k/20k instead of 3k/8k
- `sampling_factor = 0.5` instead of 0.6
- **No aggressive cloning** — uses regular `densify_and_prune` for growth, repeated depth reinit for surface quality
- xyz LR schedule is reset on every reinit via `reset_xyz_lr_schedule(iteration)` — each fresh cohort starts with `position_lr_init` instead of being born into late-decay

### `reset_xyz_lr_schedule(iteration)` (in gaussian_model.py)
Sets `self.xyz_lr_offset = iteration`. The `update_learning_rate(iter)` method now evaluates the exponential scheduler at `iter - xyz_lr_offset` instead of the raw global iteration. This means every depth reinit effectively restarts the xyz LR decay window.

## MiniMC — Zero-Waste RJ-MCMC (`--minimc`)

Replaces the old 3-stage `--minimc` schedule (depth reinit + simp1 + simp2) with a **sweep-driven** closed loop. The old code is gated behind `if False:` in train.py for reference.

**Requires**: `--mcmc` (auto-enabled) for SGLD noise, `--method 3D_SH_res` (needs `max_contrib_idx` from the rasterizer's `out_index`).

### Two events on separate intervals
**(A) Depth reinit** — every `--minimc_reinit_interval` iters:
- Same path as `--mini1`: sweep all training views, grab `depth_max_contributor` + `max_contrib_idx` per pixel, unproject + SH-transfer.
- **In-place update** via new `reinitial_alive_inplace(merged, alive_indices)`: writes only to alive slots, leaves dead pool untouched. Total tensor size is invariant.
- Budget capped at `N_alive` (not `cap_max`). Per-view integer truncation may undershoot slightly — unrefreshed alive slots just stay as-is.
- Resets Adam moments and xyz LR schedule for the refreshed slots only.
- Preempts the sweep+relocate on coincident iters.

**(B) Sweep + clone** — every `--minimc_relocate_interval` iters (when reinit isn't firing):
1. **Full-view sweep** → per-Gaussian `imp` (sum `α·T`) + `count_vis` (per-view top-99% membership). No error accumulation (benched).
2. **Cull** (snap opacity → dead pool) via three criteria ORed:
   - `imp == 0` (no contribution)
   - `count_vis ≤ 1` (single-view) — toggleable via `--minimc_no_single_view_cull`
   - `~_cdf_mask(imp, thres=low_imp_cdf)` (bottom cumulative mass) — default keep top 99.9%
3. **Clone candidate pool**: post-cull alive ∩ top-CDF(imp, `low_imp_cdf`) — matches mini v2's `mini_culling_with_clone` candidate set.
4. **Clone count** = `min(n_candidates, n_dead)` — bounded by the dead pool on one side and the candidate pool on the other. No sampling with replacement, no per-donor cap.
5. **Donor selection**: deterministic **top-K by importance** (`torch.topk`, largest first).
6. **50/50 opacity split**: donor `α_d → α_d / 2`, clone `α_d / 2`. All other fields (xyz, rot, scale, SH, kernel shape, etc.) copied raw from the donor.
7. **Adam reset** for both donor and clone slots. SGLD separates the co-located pair over subsequent steps.

### Fixed-size tensor invariant
The `_xyz` tensor is **never resized** once training starts. Cull only writes opacities, clone only writes into existing dead slots, reinit only writes into existing alive slots. The `cap_max` budget is sticky from the moment `--mcmc` allocates it — the alive/dead split moves around inside a fixed pool.

**Why this matters**: vanilla `--mcmc` respawns every dead Gaussian per cycle (via `relocate_gs(dead_mask)`) and grows the tensor by 5% per cycle (via `add_new_gs(cap_max)`), so the alive count saturates at `cap_max` and stays there. `--minimc` has **neither** — the vanilla path is gated off at [train.py:1785](train.py#L1785) when `args.minimc` is set, and minimc's own sweep+clone path only replaces `min(n_candidates, n_dead)` slots, not all of them. So **natural sparsification happens** — if the top-99.9% candidate set is smaller than the dead pool, the leftover dead slots stay dead.

### Logging
Every sweep fire prints:
```
[MINIMC] iter N: alive=K/T | cull: zero=A single_view=B low_imp=C total=N_culled | dead_in=L candidates=Q -> clones=X natural_dead_left=Y (50/50 split)
```
- `alive/T` — alive / total tensor size
- `cull:` breakdown of the three cull reasons
- `dead_in` — dead pool size entering the clone step
- `candidates` — post-cull alive ∩ top-CDF size
- `clones` — `min(candidates, dead_in)`
- `natural_dead_left` — leftover dead slots that stayed dead this cycle

Every reinit fire prints:
```
[MINIMC] Depth reinit at iter N: N_before -> N_after Gaussians
```
and saves `{iter}_pre_reinit_*` / `{iter}_post_reinit_*` debug images (RGB + alpha + depth_max_contributor + colorized max_contrib_idx) to `{model_path}/training_output/`.

### Benched code paths (kept in gaussian_model.py under `-- BENCHED` divider)
- `accumulate_minimc_error(max_contrib_idx, image, gt_image)` — per-step photometric error scatter_add onto per-Gaussian accumulator. Replaced by the full-view sweep.
- `minimc_relocate_by_error(dead_mask, max_per_donor)` — multinomial-sampled donors ∝ `error_acc`, per-donor cap, variable opacity split. Replaced by the deterministic top-K + 50/50 split above.
- `minimc_cull_losers_to_dead_pool(dead_thresh)` — pixel-ownership-based cull (count_max_wins == 0). Replaced by three-criteria importance-based cull inside the sweep.
- Re-enable by swapping the method call in train.py's `--minimc` dispatch block (~train.py:2089) and adding back the `accumulate_minimc_error` call before `total_loss.backward()`.

### `reinitial_alive_inplace(reinit_data, alive_indices, new_opacity_value=0.8)`
New method in `GaussianModel`. Replicates the field-init logic of `reinitial_from_depth` but writes in-place to a specified subset of the tensor. Used exclusively by `--minimc` to refresh alive Gaussians without destroying the dead pool.

Key details:
- Writes `min(M, K)` slots where `M = len(reinit_data['xyz'])` and `K = len(alive_indices)`. No padding/duplication of reinit data — if `M < K`, trailing alive slots stay as they were.
- Writes: `_xyz`, `_features_dc`/`_features_rest` (via sh_dc/sh_rest transfer or RGB2SH fallback), `_scaling` (NN distance), `_rotation` (from normals), `_opacity` (fixed `new_opacity_value`, default 0.8), `_appearance_level = 24`, `_shape = 1.386` (β≈4) for beta kernels.
- Zeros per-Gaussian learnable banks at alive slots: `_gaussian_features`, `_gamma`, `_adaptive_features`, `_adaptive_cat_weight`, `_adaptive_zero_weight`, `_gate_logits`.
- Resets Adam moments **only** for alive indices (dead pool Adam state preserved).
- Zeros gradient accumulators (`xyz_gradient_accum`, `feat_gradient_accum`, `denom`, `max_radii2D`, `minimc_error_accum`, `minimc_win_count`) at alive indices.

### `mini_depth_reinit` new `total_count_override` parameter
Added an optional `total_count_override` kwarg. When set, replaces `N_total = len(self._xyz)` in the per-view sample budget calculation. Used by `--minimc` to cap reinit point count at `N_alive` without shrinking the underlying tensor. `--mini`/`--mini1` callers pass no override, keeping the old behavior.

## Max-contributor infrastructure (`out_index` / `max_contrib_idx`)

Added for the SH-transfer + minimc error-routing work.

### CUDA side (`diff_surfel_3D_sh_res/cuda_rasterizer/forward.cu`)
Inside `renderCUDAsurfelForward` (the main kernel), per pixel:
- Track `float max_w`, `float max_depth`, `int max_idx = -1` alongside the existing max-contributor depth logic.
- On every gate-passed blend (`w = α·T > max_w`), update `max_idx = collected_id[j]`.
- At the pixel finalize block, write `out_index[pix_id] = max_idx` (repurposing the pre-existing `out_index` int32 `[H, W]` tensor that was allocated but never written).
- Covers both the WMMA path (~line 1316) and the scalar path (~line 1570) — both needed the same fix.
- **Consistent with `depth_max_contributor`**: they're written in the same `if (w > max_w)` block so the depth and id always refer to the same Gaussian.

### Python plumbing
- `_RasterizeGaussians.forward` in `submodules/diff_surfel_3D_sh_res/.../__init__.py` returns a 9-element tuple with `out_index` appended.
- `backward` signature accepts `grad_out_index` as the 9th positional grad arg (ignored — int tensor has no gradient).
- `gaussian_renderer/__init__.py` unpacks the 9-tuple (`len == 9` branch), exposes as `render_pkg['max_contrib_idx']`.
- Other rasterizers that don't populate `out_index` return `None` for this key → downstream code must `.get()` and guard.

### Consumers
1. **Depth reinit SH transfer** — `mini_depth_reinit` looks up `src_features_dc[max_idx]`/`src_features_rest[max_idx]` for each sampled pixel and stamps the old Gaussian's full SH onto the new reinit point. Falls back to `RGB2SH(GT_pixel)` when the max-contributor index is invalid (-1 = no contributor).
2. **Benched minimc error routing** — `accumulate_minimc_error` used to `scatter_add_` per-pixel L1 error onto `minimc_error_accum[max_idx]` for donor sampling. Now benched.
3. **Debug visualization** — `_colorize_max_contrib_idx()` in train.py hashes ids into stable RGB (3 large primes) for visual inspection. Saved pre/post each depth reinit under `{iter}_pre_reinit_maxcontrib_id.png` / `{iter}_post_reinit_maxcontrib_id.png`.

**Only `diff_surfel_3D_sh_res` populates `out_index`.** Other rasterizers (`diff-surfel-rasterization`, `diff_surfel_3D`, etc.) return `None`. To use `--minimc` or SH transfer with other methods, the CUDA patch must be ported.

## `inverse_sigmoid` opacity snap constant (used by cull paths)
Dead-pool "snap" uses `inverse_sigmoid(dead_thresh * 0.5)` — half the threshold, deep enough to guarantee the cull target is below `dead_thresh` even after small numerical drift. See `minimc_cull_losers_to_dead_pool` (benched) and `minimc_sweep_and_relocate`'s Step 2 cull loop.

## MCMC `relocate_gs` semantics reminder
For the avoidance of confusion: vanilla `--mcmc` **respawns all dead Gaussians per cycle** (every Gaussian with opacity ≤ 0.005 gets overwritten in one call) AND adds up to 5% growth via `add_new_gs(cap_max)`. Once `total == cap_max`, growth stops and the tensor just recycles dead into clones every 100 iters. Alive count saturates at `cap_max` quickly.

`--minimc` does **neither** of these:
- `relocate_gs + add_new_gs` are gated off when `args.minimc` is set (see [train.py:1785](train.py#L1785)).
- Only `min(n_candidates, n_dead)` slots are cloned per cycle via `minimc_sweep_and_relocate`. Leftover dead stay dead → natural sparsification.
- Tensor size is fixed from initialization (never grows, never shrinks).

## GSpa (`--gspa`) — GaussianSpa ADMM Sparsification

Two-phase pipeline matching the GaussianSpa (CVPR 2025) reference:
- **Phase 1** (`--gspa_simp_iter`, default 15000): importance-based pre-pruning. Sweep all views, accumulate per-Gaussian `sum(α·T)`, weighted-random sample top fraction, reinit survivors.
- **Phase 2** (`--gspa_start_iter` to `--gspa_stop_iter`): ADMM with Half-Quadratic Splitting. L2 penalty `0.5 * ρ * ‖α − z + u‖²` drives bottom `gspa_ratio` fraction's opacities toward zero. z/u update every `gspa_interval` iters. Count stays flat — ADMM is a pre-conditioner, not a pruner.
- **Hard prune** at `gspa_stop_iter`: delete bottom fraction by opacity.
- `--gspa_target_count N`: auto-splits keep ratio evenly across both phases: `keep_per_phase = sqrt(target/current)`.
- Reference uses `--densify_until_iter 15000` to stop densification before ADMM. Our default is 25000 — set it lower for faithful GSpa behavior.

## Hash Level Architecture (3D_SH_res / 3D_SH_cat)

### Hybrid Levels and Hash Grid
`--hybrid_levels` controls the split between per-Gaussian features (coarse, Lagrangian) and hash grid levels (fine, Eulerian). With `levels=6` in config and `dim=4`:
- `hybrid_levels=5` → 5 per-Gaussian levels (20D) + 1 hash level (4D)
- `hybrid_levels=2` → 2 per-Gaussian levels (8D) + 4 hash levels (16D = TC_INPUT_DIM cap)

### Finest-First Hash Resolution (2026-04)
Hash grid levels are selected from a 4-level reference progression spanning the full `r_min → r_max` range, taking the **finest** `hashgrid_levels` levels:
- Reference 4-level: `[128, 203, 322, 512]` (for min_logres=7, max_logres=9)
- hybrid=5 → 1 hash: `[512]`
- hybrid=4 → 2 hash: `[322, 512]`
- hybrid=3 → 3 hash: `[203, 322, 512]`
- hybrid=2 → 4 hash: `[128, 203, 322, 512]`

Adding per-Gaussian levels peels off from the coarsest hash levels; the finest are always retained.

### Coarse-to-Fine for Multi-Level Hash
For `3D_SH_res` and `3D_SH_cat` with >1 hash level: coarsest hash level always on, one finer level added every 2000 iters. With 1 hash level, always fully active (no C2F). Other fused modes (3D_direct_lean, etc.) remain all-levels-active.

### Renderer Level Encoding Fix
`gaussian_renderer/__init__.py` uses `ingp.hashgrid_levels` (not `ingp.levels`) for the levels encoding `(total << 16) | (active << 8) | hybrid`. Previously used config total (6) even when hash grid only had 1 level → CUDA kernel queried non-existent levels → zeros/garbage.

## Low-Pass Filter Gradient (2026-04, mode 5/6 only)

Reference 2DGS removed the low-pass filter gradient for fewer Gaussians at the cost of PSNR. We brought it back:

### What was missing
In the rho2d (low-pass) backward branch, only `dL_dmean2D` (densification signal) was computed. Two gradient paths were blocked:
1. **Alpha gradient → transMat (Tw)**: the projected center `xy ≈ Tw[:2]/Tw.z` affects alpha through the low-pass filter, but the backward only wrote to `dL_dmean2D`, not `dL_dtransMat`.
2. **Depth gradient through s → Tu, Tv, Tw**: `depth = s.x*Tw.x + s.y*Tw.y + Tw.z` depends on intersection coords `s`, which depend on all three transMat columns. The backward only propagated `dL_dz → Tw.z` directly, skipping the `dL_dz → s → Tu, Tv, Tw` chain.

### Fix (backward.cu, mode 5 WMMA + scalar paths)
The rho2d `else` branch now computes:
1. `dL_dxy` from the low-pass filter gradient → `dL_dTw` via perspective division Jacobian
2. `dL_ds = {dL_dz * Tw.x, dL_dz * Tw.y}` → full `dL_dp → dL_dk, dL_dl → dL_dTu, dL_dTv, dL_dTw` chain (same as rho3d but without the alpha-through-s term)
3. All 9 elements of `dL_dtransMat` now receive gradients in both branches

## AbsGS — Cancellation-Free Densification (2026-04)

Standard 2DGS accumulates `dL_dmean2D.xy` for densification decisions, but per-pixel contributions can cancel (positive from one side, negative from the other). AbsGS stores `fabs()` of per-pixel contributions in extra channels for a cancellation-free split signal.

### Implementation
- `dL_dmean2D` expanded from `float3` → `float4` (channels z/w = abs signal)
- **rho3d branch**: `atomicAdd(&dL_dmean2D[].z, fabsf(dL_dTu.z))` and `.w` for `fabsf(dL_dTv.z)`
- **rho2d branch**: `atomicAdd(&dL_dmean2D[].z, fabsf(dL_dG * dG_ddelx))` and `.w` for `.y`
- **Preprocess backward**: adds `fabsf(dL_dtransMat[2/5])` (AABB contribution) to z/w, scales by `depth * W/H`
- `screenspace_points` expanded to `{N, 4}` in Python
- `add_densification_stats` reads `grad[:, 2:4]` (abs channels) when shape >= 4, falls back to full norm for older rasterizers

### `--detach_hash_grad` Semantics
Zeros `dL_dxyz` from the hash backward in CUDA. Effect: hash spatial derivatives can't push Gaussian xyz/scale/rotation. Hash table features and MLP weights still update normally. Hash `dL_dxyz` does NOT flow to opacity (opacity gradient comes from `dL_dalpha * G(u,v)` which is independent of the hash backward).

## Warp-Reduced Per-Gaussian Atomics + Tile-Level Early-Out (2026-04, mode 5/6 GEMM path)

Atomic-contention optimization for the backward pass in `diff_surfel_3D_sh_res`. Only touches the collaborative-GEMM path (`render_mode & 0x100`); the non-GEMM fallback in the same kernel is unchanged. Motivated by the FastGS / Taming-3DGS per-splat refactor, but applied *without* flipping the parallelization scheme — it stays pixel-parallel so the existing tile-level MLP weight GEMM (sub-tile matmul in shared memory, one atomic flush per tile per W element) keeps its locality advantage.

### What we had before
- Kernel is pixel-parallel (1 thread = 1 pixel, 256 threads/tile) — confirmed per-pixel in `renderCUDAsurfelBackward` ([backward.cu:669](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L669)).
- GEMM path iterates synchronously over Gaussians in a tile batch ([backward.cu:1024](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L1024)). All 256 threads process the same Gaussian `j` together.
- Each participating pixel fired `atomicAdd(&dL_d*[global_id…], val)` on every per-Gaussian field. With up to 256 participating threads all hitting the same address, this was the dominant contention path. Fields hit per Gaussian per tile: `dL_dopacity` (1), `dL_dcolors` (3), `dL_dnormal3D` (3), `dL_dtransMat` (9), `dL_dmean2D` (4 incl. AbsGS abs), `dL_dhomoMat` (9 when homotrans), `dL_dshapes` (1–2 depending on kernel). ~30 reducible atomic slots × 256 threads = ~7.5K atomics per Gaussian per tile.
- Ballot-based per-Gaussian skip existed (`__syncthreads_count(participates)`, [backward.cu:1036](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L1036)) to avoid entering the inner loop body when no pixel contributed, but SMEM loads happened regardless of whether the whole *batch* was relevant.

### What changed
1. **Warp-reduced per-Gaussian atomics** ([backward.cu:~1583-1655](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L1583)). Inside the synchronized j-loop body, every per-Gaussian `atomicAdd` now writes to a register-local `acc_*` accumulator (initialized to 0 at the top of the j iteration). After `if (participates)` closes, all 32 lanes of each warp call `cg::reduce(warp, acc_*, cg::plus<float>())` to sum across the warp, and only lane 0 fires `atomicAdd` to global memory. BLOCK_SIZE=256 → NUM_WARPS=8 → **256 atomics per field per Gaussian drops to 8**. Non-participating threads contribute 0 to the reduction (branch-free). All lanes reach the reduction site because the ballot `continue` happens at the top of the j iteration, not inside.

2. **Tile-level `last_contributor` max early-out** ([backward.cu:~955-985](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L955)). Once at kernel start, compute `tile_max_last_contrib = max over pixels of last_contributor` via shared-memory `atomicMax` (256 SMEM atomics, one-time). In the main rounds loop, before the SMEM prefetch, check whether the *lowest* Gaussian-contributor-index in the upcoming batch (`contributor - effective_toDo`) is already ≥ `tile_max_last_contrib`; if so, **skip the entire batch**: no SMEM load, no j-loop body, no profiling noise. Equivalent to FastGS's `max_contrib[tile_id]` bucket-skip, scoped to our tile/batch layout.

### What is NOT changed
- **MLP weight gradients** (`dL_dmlp_W1/W2/W3`): still routed through the collaborative WMMA GEMMs (`wmma_gemm_layer1/2/3`) accumulating into shared `tile_dL_dW*`, flushed once per tile via `flush_tile_mlp_grads` at [backward.cu:~2623](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu#L2623). This produces one atomic per W element per tile — strictly better than per-splat would, since there are many more splats than tiles. **Do not move these into the warp reduction.**
- **Hash-table gradients** (`dL_dfeatures`, the 2^19 × 4D table scattered inside `query_feature<true>`): different pixels hit different hash cells, so warp-reduction doesn't apply. Left as pixel-level atomicAdds.
- **Non-GEMM path** in the same kernel (`render_mode & 0x100 == 0`): untouched. Old per-pixel atomicAdds preserved.
- **Preprocess kernels** (`transMat_to_scale_rot_grad_kernel`, etc.): already per-Gaussian, no contention.

### Atomic count math (per Gaussian per tile, GEMM path, 256-pixel tile, rho3d branch representative)
| Field | Slots | Before (atomics) | After (atomics) |
|---|---|---|---|
| dL_dopacity | 1 | 256 | 8 |
| dL_dcolors | 3 | 768 | 24 |
| dL_dnormal3D | 3 | 768 | 24 |
| dL_dtransMat | 9 | 2304 | 72 |
| dL_dmean2D (abs channels) | 2 | 512 | 16 |
| dL_dhomoMat | 9 | 2304 | 72 |
| dL_dshapes | 1 | 256 | 8 |
| **Total** |  | **~7.2K** | **~224** |

~32× reduction, and collisions on each address drop from "up to 32 lanes in the same warp step" to "one lane per warp" (8 warps spread across SM schedules).

### Correctness notes
- `cg::reduce` is a warp-level collective — requires all 32 lanes in the warp to arrive. The j-loop body is entered by all 256 threads (the ballot `continue` happens *before* the body), so this is safe. Divergent branches inside (`rho3d <= rho2d`, `render_mode & 0x400` lowpass, `kernel_type == 5`) write to *different slots* of the same accumulator array; unwritten slots stay 0 and contribute 0 to their reduction — no duplication or loss.
- `dL_dhomoMat` and `dL_dshapes` can be nullptr for some kernel configs; reduction code guards on those pointers before issuing the atomic (only lane 0 does the guarded atomic, to avoid warp divergence during the reduce call itself — the reduce happens unconditionally).
- `dL_dmean2D` is `float4` (AbsGS); .x/.y come from rho2d branch only, .z/.w are the abs accumulators written by both branches. Same accumulator array handles both.
- Numerical differences from reordered float addition are ~1e-6 relative (float-associativity).

### Files touched
- [backward.cu](submodules/diff_surfel_3D_sh_res/cuda_rasterizer/backward.cu) — `renderCUDAsurfelBackward` only. `auto warp = cg::tiled_partition<32>(block)` at kernel top; `s_tile_max_last_contrib` SMEM + compute just before the rounds loop; batch early-out inside the rounds loop before SMEM prefetch; accumulator decls at the top of the j-body; atomicAdds → `acc_*` writes throughout the j-body; warp-reduce + lane-0 atomicAdd block just after the `if (participates)` closes, before the profiling sync.
- No header / signature changes. No new kernel launch params. No Python-side changes.
