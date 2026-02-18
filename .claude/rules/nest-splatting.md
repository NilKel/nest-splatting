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

## Known Issues Fixed

1. **Post-10k divergence**: `diff_surfel_rasterization/__init__.py` was zeroing `grad_opacities` in render_mode=3. Fixed by removing ALL geometry gradient zeroing.

2. **Normal gradient path**: `transMat_to_scale_rot_grad_kernel` needs `dL_dnormal3D` + viewmatrix.

3. **Hash gradient flow**: `dL_duv` connects to ray-disk Jacobian.

4. **tcnn weight layout**: Pads input to multiples of 16, padding columns = 1 (implicit bias). NO explicit biases.

5. **Missing geometry gradients in 3D_direct_fused backward**: Was only writing `dL_dtransMat[6..8]` (Tw). Fixed by adding full Tu, Tv, Tw computation from cat mode reference.

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
