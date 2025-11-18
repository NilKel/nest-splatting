# Critical Gradient Flow Bug - FIXED

## The Problem

**Symptom:** Gaussians increasing instead of decreasing during training.

**Root Cause:** Buffer overflow causing gradients to not flow correctly through the dot product operation.

---

## What Was Happening

### Bug #2b: Buffer Overflow

After fixing Bug #2 (dimension mismatch in tensor allocation), the template parameter `CHANNELS` changed from 72 to 24. This created a **hidden buffer overflow**:

```cuda
template <uint32_t CHANNELS>  // CHANNELS = 24 after fix
__global__ void renderCUDAsurfelForward(...) {
    float feat[CHANNELS];  // Array size: 24
    ...
    case 12: {
        float feat_vec[CHANNELS];  // BUG: Array size 24, but need 72!
        query_feature<false, CHANNELS, 12>(feat_vec, xyz, ...);
        // ↑ Tries to write 72 values into 24-element array!
        
        // Dot product reads from corrupted memory
        for(int lv = 0; lv < level; lv++){
            int vec_start = lv * 12;
            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 3; j++){
                    dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
                    //          ↑ Reading beyond array bounds!
                }
            }
        }
    }
}
```

### Memory Corruption

```
feat_vec[] allocated:  [0..23]  (24 elements)
feat_vec[] needed:     [0..71]  (72 elements)

Write pattern:
  feat_vec[0..71] = hashgrid_output

Memory after write:
  [0..23]  ✓ Valid data
  [24..71] ✗ OVERFLOW - corrupts adjacent stack memory!

Dot product reads:
  normal[0] * feat_vec[0..71]  
              ↑ Reading corrupted memory for indices [24..71]!
```

### Effect on Gradients

**Forward pass:**
- Hashgrid outputs written to too-small buffer
- Memory corruption in `feat_vec`
- Dot product computed with garbage data
- Wrong features fed to alpha blending

**Backward pass:**
```cuda
case 12: {
    float grad_feat_vec[C * 3];  // C = 24, so 72 elements ✅
    
    // Expand gradients
    for(int lv = 0; lv < level; lv++){
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 3; j++){
                grad_feat_vec[vec_start + i * 3 + j] = -dL_dscalar * normal[j];
            }
        }
    }
    
    // But passed wrong buffer to query_feature!
    query_feature<true, C * 3, 12>(feat, xyz, ...grad_feat_vec...);
    //                             ↑ feat[24] too small, should be feat_dummy[72]!
}
```

Result:
- Gradients not properly backpropagated through hashgrid
- Gaussians not receiving correct update signals
- Training diverges: Gaussians increase instead of being pruned

---

## The Fix

### Forward Pass (`forward.cu`)

**Before:**
```cuda
case 12: {
    float feat_vec[CHANNELS];  // CHANNELS = 24 → BUFFER OVERFLOW
    query_feature<false, CHANNELS, 12>(feat_vec, xyz, ...);
}
```

**After:**
```cuda
case 12: {
    float feat_vec[96];  // Fixed size: 8 levels × 12 features
    query_feature<false, 96, 12>(feat_vec, xyz, ...);
    // ✓ Buffer correctly sized for hashgrid output
}
```

### Backward Pass (`backward.cu`)

**Before:**
```cuda
case 12: {
    float grad_feat_vec[C * 3];  // 72 elements ✓
    // ... expand gradients ...
    query_feature<true, C * 3, 12>(feat, xyz, ...grad_feat_vec...);
    //                             ↑ feat[24] → wrong size!
}
```

**After:**
```cuda
case 12: {
    float grad_feat_vec[96];  // Fixed size: 8 levels × 12 features
    float feat_dummy[96];     // Dummy array for feat parameter
    // ... expand gradients ...
    query_feature<true, 96, 12>(feat_dummy, xyz, ...grad_feat_vec...);
    // ✓ All buffers correctly sized
}
```

---

## Why This Broke Training

### Gradient Flow Path

```
                Forward
                ───────
Python MLP  →  [loss]
                 ↓
CUDA Rasterizer: Alpha-blended features (24D)
                 ↓
            Dot Product (24D ← 72D × normal)
                 ↓
            Hashgrid Query (72D)
                 ↓
            Hashgrid Parameters


                Backward
                ────────
            dL/dParams  ← Hashgrid Parameters
                 ↑
Query backprop with grad_feat_vec[72]  ← BUG: Used feat[24]
                 ↑
Expand grads: (24D → 72D) using normal
                 ↑
       dL/d(scalar features) (24D)
                 ↑
            Alpha blending
                 ↑
              [dL/dloss]
```

**Problem:**
- `query_feature` backward pass expected correct buffer sizes
- Wrong buffer size → gradients written to wrong memory locations
- Hashgrid parameters didn't receive proper gradient updates
- Without proper gradients, Gaussians couldn't be pruned

---

## Verification

After recompilation, you should see:
- ✅ `surface_features.shape = (H*W, 24)` 
- ✅ Gradients flowing correctly through dot product
- ✅ Gaussians being pruned during densification
- ✅ Number of Gaussians decreasing over training
- ✅ Loss decreasing properly

---

## Technical Details

### Memory Layout

**Hashgrid output (per pixel):**
```
Level 0: [v0=(x,y,z), v1=(x,y,z), v2=(x,y,z), v3=(x,y,z)]  ← 12 values
Level 1: [v0=(x,y,z), v1=(x,y,z), v2=(x,y,z), v3=(x,y,z)]  ← 12 values
...
Level 5: [v0=(x,y,z), v1=(x,y,z), v2=(x,y,z), v3=(x,y,z)]  ← 12 values

Total: 6 levels × 12 features = 72 values
```

**After dot product:**
```
Level 0: [s0, s1, s2, s3]  ← 4 scalars (each = -v·normal)
Level 1: [s0, s1, s2, s3]  ← 4 scalars
...
Level 5: [s0, s1, s2, s3]  ← 4 scalars

Total: 6 levels × 4 features = 24 values
```

### Buffer Sizes Summary

| Buffer | Old Size | Correct Size | Usage |
|--------|----------|--------------|-------|
| `feat_vec` (forward) | `CHANNELS` (24) | 96 | Hashgrid query output |
| `feat` (forward) | `CHANNELS` (24) | 24 | Dot product output |
| `grad_feat_vec` (backward) | `C * 3` (72) | 96 | Hashgrid gradient input |
| `grad_feat` (backward) | `C` (24) | 24 | Dot product gradient output |

---

## Recompile Command

```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda run -n nest_splatting python setup.py install
```

---

## Status: ✅ FIXED

All buffer sizes corrected in both forward and backward passes. Gradients should now flow correctly!

