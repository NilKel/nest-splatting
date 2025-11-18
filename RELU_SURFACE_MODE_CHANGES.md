# ReLU Activation for Surface Mode - Implementation Summary

## Overview

Added ReLU activation after the negated dot product in surface mode computation. This forces the vector potentials and Gaussian normals to face opposite directions for features to be active.

## Mathematical Change

**Before:**
```
scalar_feature = -dot(vector_potential, normal) + baseline
```

**After:**
```
scalar_feature = ReLU(-dot(vector_potential, normal) + baseline)
             = max(0, -dot(vector_potential, normal) + baseline)
```

**Effect:** Only features where vectors and normals point in opposite directions (negative dot product becomes positive after negation) will contribute. This encourages the learned vector field to align anti-parallel with surface normals.

---

## Changes Made

### 1. Forward Pass - Case 12 (Surface Mode with Optional Baseline)

**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`

**Line 771:** Added ReLU activation
```cuda
// Before:
feat[scalar_start + i] = -dot_prod + feat_baseline[lv * 4 + i];

// After:
// ReLU activation forces vectors and normals to face opposite directions
feat[scalar_start + i] = fmaxf(0.0f, -dot_prod + feat_baseline[lv * 4 + i]);
```

### 2. Forward Pass - Case 15 (Surface RGB Mode)

**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`

**Line 804:** Added ReLU activation
```cuda
// Before:
feat[out_start + i] = -dot_prod;

// After:
// ReLU activation forces vectors and normals to face opposite directions
feat[out_start + i] = fmaxf(0.0f, -dot_prod);
```

### 3. Backward Pass - Case 12 (Surface Mode)

**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`

**Lines 767-822:** Updated gradient computation to handle ReLU

Key changes:
- Query baseline features to recompute forward values (lines 782-795)
- Recompute forward values in gradient loop (lines 803-808)
- Apply ReLU mask to gradients: zero if forward_val <= 0 (line 811)

```cuda
// Recompute forward value to check ReLU activation
float dot_prod = 0.0f;
for(int j = 0; j < 3; j++){
    dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
}
float forward_val = -dot_prod + feat_baseline[lv * 4 + i];

// ReLU gradient: zero if forward_val <= 0
float dL_dscalar = (forward_val > 0.0f) ? grad_feat[scalar_start + i] : 0.0f;
```

**Lines 837-851:** Updated baseline gradient computation with ReLU mask

```cuda
// Recompute forward value for ReLU mask
float forward_val = -dot_prod + feat_baseline[lv * 4 + c];
float relu_mask = (forward_val > 0.0f) ? 1.0f : 0.0f;

grad_baseline[lv * 4 + c] = w * grad_feat[lv * 4 + c] * relu_mask;
```

### 4. Backward Pass - Case 15 (Surface RGB Mode)

**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`

**Lines 871-921:** Updated gradient computation to handle ReLU

Key changes:
- Recompute forward values in gradient loop (lines 897-902)
- Apply ReLU mask to gradients (line 905)

```cuda
// Recompute forward value to check ReLU activation
float dot_prod = 0.0f;
for(int j = 0; j < 3; j++){
    dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
}
float forward_val = -dot_prod;  // No baseline in case 15

// ReLU gradient: zero if forward_val <= 0
float dL_dscalar = (forward_val > 0.0f) ? grad_feat[out_start + i] : 0.0f;
```

---

## Gradient Flow

The ReLU introduces a piecewise gradient:

```
dL/d(forward_val) = {
    dL/dscalar  if forward_val > 0
    0           if forward_val ≤ 0
}
```

This affects:
1. **Vector potential gradients:** Only updated when vectors point opposite to normals
2. **Normal gradients:** Only influenced by active (non-zero) features
3. **Baseline gradients (case 12):** Also masked by ReLU activation

---

## Expected Training Behavior

### Positive Effects:
- **Clearer feature semantics:** Forces learning of consistent vector field orientation relative to surface normals
- **Better surface modeling:** Encourages features to represent surface-aligned properties
- **Reduced noise:** Inactive (zero) features don't contribute gradients

### Potential Issues to Watch:
- **Slower convergence initially:** More features may be inactive early in training
- **Dead features:** If many features consistently fall on the inactive side, they won't train
- **Solution:** Proper initialization and learning rate scheduling

---

## Compilation Status

✅ **Successfully compiled** (2025-10-19)
- No compilation errors
- Only standard warnings (unused variables, format strings)
- Extension installed to: `/home/nilkel/miniconda3/envs/nest_splatting/lib/python3.10/site-packages/diff_surfel_rasterization/`

---

## Testing Recommendations

1. **Visual inspection:**
   - Check if surfaces look smoother/cleaner
   - Verify view-dependent effects still work

2. **Quantitative metrics:**
   - Compare PSNR/SSIM before and after
   - Monitor training loss curves

3. **Feature analysis:**
   - Visualize which features are active (non-zero after ReLU)
   - Check if vector field aligns with surface normals

4. **Gradient checks:**
   - Enable `torch.autograd.set_detect_anomaly(True)`
   - Verify gradients are finite and flow correctly

---

## Usage

The ReLU activation is automatically applied when using:
- `method='surface'` (case 12)
- `method='surface_rgb'` (case 15)

No configuration changes needed - just retrain your model with the recompiled extension.

---

## Related: Normal Normalization

**Also added:** Normal vectors are now normalized to unit length by default for consistent dot product magnitudes.

**Toggle:** See `NORMAL_NORMALIZATION_TOGGLE.md` for how to easily enable/disable this feature.

To disable normalization, comment out this line in both `forward.cu` (~line 119) and `backward.cu` (~line 1246):
```cuda
// #define NORMALIZE_SURFACE_NORMALS
```

Both features (ReLU + normalization) work together - the normalized normals are used in the dot product, and then ReLU is applied to the result.

---

## Related Files

- Implementation plan for dual hashgrid: `SURFACE_RGB_DUAL_HASHGRID_IMPLEMENTATION_PLAN.md`
- Forward kernel: `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`
- Backward kernel: `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`

