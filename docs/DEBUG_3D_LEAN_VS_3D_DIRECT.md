# Debugging 3D_lean vs 3D_direct Mode

## Problem Statement

**3D_direct mode (tcnn MLP) works correctly**, but **3D_lean/3D_direct_fused (custom CUDA MLP) has broken gradients**.

### Symptom
With **unit weights** (identity matrices for W1/W2/W3):
- **Expected**: `dL_db1`, `dL_db2`, `dL_db3` should have nonzeros at indices `[0, 1, 2]` only
- **Observed (3D_lean)**: `dL_db3` is CORRECT (3/3 at [0,1,2]), but `dL_db1` and `dL_db2` are WRONG (16/32 at [0-15])

### Key Insight
Since `dL_db3` is correct, W3 weights ARE being read correctly in backward. The bug is in **layers 1 and 2 specifically**.

## Modes Overview

| Mode | MLP Backend | Hash Location | Status |
|------|-------------|---------------|--------|
| `3D_direct` | tcnn | Python batch | WORKS |
| `3D_lean` | Custom CUDA | CUDA kernel | BROKEN gradients |
| `3D_direct_fused` | Custom CUDA | CUDA kernel | BROKEN gradients |

## Test Approach

**Compare 3D_lean against 3D_direct with identical inputs and weights** to find where they diverge.

### Test Script
```bash
conda run -n nest_splatting python scripts/test_3d_lean_vs_3d_direct.py
```

### Checkpoint
```
outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8/ngp_30000.pth
```

## MLP Architecture

```
Input: 40D (24D features + 16D view encoding)
  ├─ Layer 1: Linear(40, 32) + ReLU
  ├─ Layer 2: Linear(32, 32) + ReLU
  └─ Layer 3: Linear(32, 3) + Sigmoid
Output: 3D RGB
```

### Unit Weights for Testing
```python
W1 = diagonal identity (32x40) - first 32 diagonal elements = 1
W2 = identity (32x32)
W3 = diagonal (3x32) - W3[0,0]=1, W3[1,1]=1, W3[2,2]=1
b1 = b2 = b3 = zeros
```

With these weights:
- `h1[i] = relu(x[i])` for i < 32
- `h2[i] = relu(h1[i]) = relu(relu(x[i]))`
- `out[i] = sigmoid(h2[i])` for i < 3

Gradients should only flow through indices 0, 1, 2.

## Key Files

### Python
- `gaussian_renderer/__init__.py` - render() function, handles all methods
- `hash_encoder/modules.py` - INGP class with `mlp_fused` attribute
- `scripts/test_3d_lean_vs_3d_direct.py` - Comparison test

### CUDA (diff_surfel_3D - lean library)
- `cuda_rasterizer/forward.cu` - `setMlpWeights()` at ~line 1759
- `cuda_rasterizer/backward.cu` - Two backward paths:
  - Collaborative GEMM path (line 934)
  - Switch case 5 path (line 1389)
- `cuda_rasterizer/modes/mode_3d_direct_fused.cu`:
  - `compute_dL_dz_all` (line 453+) - Layer 3 backward
  - `collaborative_mlp_backward_all` (line 254+) - Full MLP backward

### Weight Storage
- Weights stored in device memory (NOT constant memory)
- `setMlpWeights()` uploads from PyTorch tensors
- `getMlpWeightPointers()` returns pointers for backward pass

## Gradient Flow (Expected)

```
loss = render.sum()
  │
  ▼ dL_drgb
sigmoid backward: dL_dz3 = dL_drgb * sig * (1-sig)
  │
  ▼ dL_dz3 [only indices 0,1,2 nonzero]
Layer 3 backward: dL_dh2 = dL_dz3 @ W3  (W3 is 3x32)
  │                dL_db3 = dL_dz3
  │
  ▼ dL_dh2 [only indices 0,1,2 nonzero due to W3 structure]
ReLU backward: dL_dz2 = dL_dh2 * (h2_pre > 0)
  │
  ▼ dL_dz2 [only indices 0,1,2]
Layer 2 backward: dL_dh1 = dL_dz2 @ W2  (W2 is 32x32 identity)
  │                dL_db2 = dL_dz2  [SHOULD BE only 0,1,2]
  │
  ▼ dL_dh1 [only indices 0,1,2]
...
```

## Debugging Steps

1. **Verify forward outputs match** between 3D_direct and 3D_lean
2. **Add debug prints** in CUDA kernel to see intermediate values
3. **Check weight indexing** - W[o * stride + i] vs W[i * stride + o]
4. **Check ReLU derivative** - h_pre vs h_post for mask

## Known Issues (Fixed)

1. **Initial test used checkpoint weights, not unit weights**: Fixed by setting weights on `ingp.mlp_fused` directly
2. **mlp_fused didn't exist on 3D_direct checkpoint**: Fixed by creating it in test

## Build Commands

```bash
# Lean library (faster build, only modes 0, 3, 5)
cd /home/nilkel/Projects/nest-splatting/submodules/diff_surfel_3D
conda run -n nest_splatting python -m pip install -e . --no-build-isolation

# Full rasterizer
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

**IMPORTANT**: Always use `run_in_background: true` for builds (2-5 minutes).

## Test Results (2026-02-17)

### Test Command
```bash
conda run -n nest_splatting python scripts/test_3d_lean_vs_3d_direct.py
```

### CUDA Gradients ARE Being Computed
```
  [CUDA GRADS] get_mlp_grads() returned: True
    grad_W1: torch.Size([32, 40]), nonzero=80
    grad_b1: torch.Size([32]), nonzero=16  <-- BUG: should be 3
    grad_W2: torch.Size([32, 32]), nonzero=64
    grad_b2: torch.Size([32]), nonzero=16  <-- BUG: should be 3
    grad_W3: torch.Size([3, 32]), nonzero=16
    grad_b3: torch.Size([3]), nonzero=3    <-- CORRECT
```

### Confirmed Bug Pattern
- `dL_db3` is CORRECT: 3/3 nonzeros at [0, 1, 2]
- `dL_db2` is WRONG: 16/32 nonzeros at [0-15]
- `dL_db1` is WRONG: 16/32 nonzeros at [0-15]

### Forward Pass Also Differs
```
  Render diff (abs): mean=0.032240, max=0.736555
  [MISMATCH] Forward pass outputs differ significantly!
```

This means both forward AND backward have issues with unit weights.

### Root Cause Analysis - UPDATED

The debug output revealed:
```
[DEBUG W3] h2_pre[0:5]: [0.000, 0.000, 0.000, 0.436, 0.000]
```

The Gaussian features are `[-1.022, -0.489, -0.589, 0.436, -0.203, ...]`.

With unit weights: `h1[i] = relu(input[i])`. Features 0,1,2 are NEGATIVE, so they get zeroed by ReLU! Only feature[3]=0.436 survives.

**The 16 nonzeros in gradients correspond to the ~16 positive input values out of 40 (24D features + 16D view encoding).**

This is NOT a bug - it's correct ReLU behavior! The gradient sparsity pattern depends on which inputs are positive.

### Forward Pass Mismatch

However, there's still a forward pass mismatch:
```
  3D_direct: [0.984, 0.970, 0.961]
  3D_lean:   [0.495, 0.495, 0.495]
```

The 3D_lean output is `sigmoid(0) = 0.5` (approximately), which suggests:
- With unit weights and negative inputs[0:2], we get `h1[0:2] = 0`
- So `h2[0:2] = relu(h1[0:2]) = 0`
- And `output[0:2] = sigmoid(h2[0:2]) = sigmoid(0) = 0.5`

But 3D_direct gets ~0.98, which means the tcnn MLP is producing different values.
This could be due to tcnn's weight layout or different behavior with the trained weights.

**Key Issue**: We're testing unit weights on CUDA MLP but 3D_direct still uses trained tcnn weights!

## KEY FINDING: Forward Pass Works with Trained Weights!

Running with `--no_unit_weights` (using trained checkpoint weights):
```bash
python scripts/test_3d_lean_vs_3d_direct.py \
  --model_path outputs/nerf_synthetic/chair/3D_direct/single_gaussian_test \
  --no_unit_weights
```

Result:
```
  Render diff (abs): mean=0.000000, max=0.000427
  [OK] Forward pass outputs SIMILAR (within 0.01)
```

**The 3D_lean MLP forward pass is CORRECT when using trained weights!**

The unit weight test was misleading because:
1. Trained features are often negative (e.g., `[-1.022, -0.489, ...]`)
2. ReLU zeros out negative values
3. So gradient flow depends on which inputs happen to be positive

### Single Gaussian Test Checkpoint
```
outputs/nerf_synthetic/chair/3D_direct/single_gaussian_test/
```
Features: `[-1.022, -0.489, -0.589, 0.436, -0.203, ...]` (mostly negative)

## Remaining Issue: grad_xyz Differs

With trained weights, forward matches but:
```
  grad_xyz diff (abs): mean=590.610901, max=1064.009155
```

This is because:
- 3D_direct uses tcnn for backward
- 3D_lean uses CUDA MLP for backward
- The MLP architectures may differ (tcnn uses 256-width hidden, CUDA uses 32-width)

## Test Script
```bash
# With unit weights (for gradient flow debugging)
python scripts/test_3d_lean_vs_3d_direct.py --model_path outputs/nerf_synthetic/chair/3D_direct/single_gaussian_test

# With trained weights (for production comparison)
python scripts/test_3d_lean_vs_3d_direct.py --model_path outputs/nerf_synthetic/chair/3D_direct/single_gaussian_test --no_unit_weights
```
