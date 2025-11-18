# Surface RGB Mode: Dual Hashgrid Implementation Plan

## Overview

This document outlines the implementation of a new architecture for `surface_rgb` mode that uses **two separate hashgrids** instead of a single combined hashgrid.

### Architecture Goals

**Old Architecture (case 15):**
- Single hashgrid with 15 features/level (12 vectors + 3 RGB)
- Both queried at intersection point
- Everything alpha-blended together

**New Architecture:**
1. **Diffuse RGB Hashgrid:** 3 features/level (RGB), queried at Gaussian center
   - Features from all levels are **SUMMED** → single RGB per Gaussian
   - Stored in shared memory (one query per Gaussian)
   - Alpha-blended like standard Gaussian splatting: `C_diffuse = Σ(alpha_i * T_i * rgb_i)`

2. **View-Dependent Feature Hashgrid:** 12 features/level (vectors), queried at intersection
   - Dot product with normal → 4 scalars/level
   - Features from all levels are **CONCATENATED** → 24D per sample
   - Alpha-blended: `C_view = Σ(alpha_i * T_i * feat_i)`
   - Passed to MLP: `rgb_view = MLP(C_view + view_encoding)`

3. **Final Output:**
   ```
   final_rgb = C_diffuse + rgb_view
   ```

---

## ✅ Completed Changes

### 1. Python INGP Module (`hash_encoder/modules.py`)

**Changes:**
- Modified `__init__` to detect `surface_rgb` mode and create two hashgrids
- Added `build_encoding_diffuse()` method for diffuse RGB hashgrid (3 features/level)
- Added `build_encoding_view_features()` method for view features (12 features/level)
- Updated `training_setup()` to register both hashgrids' parameters with optimizer

**Key Code:**
```python
elif method == 'surface_rgb':
    # Build diffuse RGB hashgrid (3 features per level, summed to 3 total)
    cfg_encoding_diffuse = cfg_model.encoding
    cfg_encoding_diffuse.hashgrid.dim = 3
    self.build_encoding_diffuse(cfg_encoding_diffuse)
    
    # Build view-dependent hashgrid (12 features per level, dot product to 4, concatenated to 24)
    cfg_encoding_view = cfg_model.encoding
    cfg_encoding_view.hashgrid.dim = original_dim * 3  # 12 features per level
    self.build_encoding_view_features(cfg_encoding_view)
    
    self.level_dim = original_dim  # 4 per level (after dot product)
    self.feat_dim = cfg_model.encoding.levels * self.level_dim  # 24D for MLP
```

**Attributes Created:**
- `self.hash_encoding_diffuse`: Diffuse RGB hashgrid
- `self.hash_encoding_view_features`: View-dependent feature hashgrid
- `self.gridrange_diffuse`, `self.gridrange_view`: Spatial ranges
- `self.diffuse_rgb_dim = 3`: Output dim for diffuse (summed)
- `self.view_feat_dim = 24`: Output dim for view features (concatenated)

### 2. Python Render Function (`gaussian_renderer/__init__.py`)

**Changes:**
- Lines 126-167: Added logic to extract parameters from both hashgrids when `ingp.method == 'surface_rgb'`
- Lines 213-215: Pass diffuse hashgrid parameters to rasterizer

**Key Code:**
```python
if ingp.method == 'surface_rgb':
    # Two separate hashgrids
    features_diffuse, offsets_diffuse, levels_diffuse, per_level_scale_diffuse, base_resolution_diffuse, align_corners_diffuse, interpolation_diffuse \
        = ingp.hash_encoding_diffuse.get_params()
    gridrange_diffuse = ingp.gridrange_diffuse
    
    features_view, offsets_view, levels_view, per_level_scale_view, base_resolution_view, align_corners_view, interpolation_view \
        = ingp.hash_encoding_view_features.get_params()
    gridrange_view = ingp.gridrange_view
    
    # Use view hashgrid settings for primary hashgrid settings
    features = features_view
    offsets = offsets_view
    gridrange = gridrange_view
    # ... etc
```

### 3. Python Rasterizer Wrapper (`diff_surfel_rasterization/__init__.py`)

**Changes:**
- Lines 269-273: Added diffuse hashgrid parameters to `forward()` signature
- Lines 308-314: Initialize diffuse parameters to empty tensors if None
- Lines 333-335: Pass diffuse parameters to C++ binding

### 4. C++ Binding (`rasterize_points.cu`)

**Changes:**
- Lines 73-75: Added diffuse hashgrid parameters to function signature
- Lines 106-108: Added CHECK_INPUT for diffuse parameters
- Lines 110-139: Added logic to detect `surface_rgb` mode and calculate output dimensions:
  ```cpp
  bool is_surface_rgb = (features_diffuse.numel() > 0 && offsets_diffuse.numel() > 0);
  if(is_surface_rgb){
      // Surface RGB mode with two separate hashgrids
      D_diffuse = features_diffuse.size(1);  // Should be 3 (RGB)
      C = (offsets.size(0) - 1) * 4;  // 6 levels × 4 = 24D view features
  }
  ```
- Lines 210-213: Pass diffuse parameters to CUDA forward kernel

---

## 🚧 TODO: CUDA Kernel Implementation

### Overview of Required Changes

The CUDA kernels need a **new execution path** for `surface_rgb` mode with dual hashgrids. This is case `D=12` with `D_diffuse=3`.

### Files to Modify

1. **`cuda_rasterizer/rasterizer.h`** - Update forward/backward signatures
2. **`cuda_rasterizer/rasterizer_impl.h`** - Update Rasterizer class methods
3. **`cuda_rasterizer/rasterizer_impl.cu`** - Update wrapper functions
4. **`cuda_rasterizer/forward.cu`** - Implement dual hashgrid forward pass
5. **`cuda_rasterizer/backward.cu`** - Implement dual hashgrid backward pass

---

## Step-by-Step Implementation Plan

### STEP 1: Update Forward Function Signature

**File:** `cuda_rasterizer/rasterizer.h`

**Current signature (approximate line 50-80):**
```cpp
int forward(
    // ... existing parameters ...
    const float* hash_features,
    const int* hash_offsets,
    const float* hash_gridrange,
    // ... rest of parameters ...
    float beta);
```

**New signature:**
```cpp
int forward(
    // ... existing parameters ...
    const float* hash_features,
    const int* hash_offsets,
    const float* hash_gridrange,
    // ... rest of parameters ...
    float beta,
    // NEW: Diffuse hashgrid parameters
    const uint32_t D_diffuse,
    const float* hash_features_diffuse,
    const int* hash_offsets_diffuse,
    const float* hash_gridrange_diffuse);
```

### STEP 2: Update Rasterizer Implementation Wrappers

**File:** `cuda_rasterizer/rasterizer_impl.cu`

Update the `forward()` function (around line 200-300) to:
1. Accept new diffuse hashgrid parameters
2. Pass them to the CUDA kernel launch

**File:** `cuda_rasterizer/rasterizer_impl.h`

Update the `Rasterizer::forward()` method signature in the class declaration.

### STEP 3: Implement CUDA Forward Kernel for Dual Hashgrid

**File:** `cuda_rasterizer/forward.cu`

This is the most complex change. You need to modify the `renderCUDAsurfelForward` kernel.

#### 3.1: Preprocess Phase - Query Diffuse RGB per Gaussian

**Location:** Around line 280-350 in `forward.cu` (before the main rendering loop)

**Goal:** Query diffuse RGB hashgrid once per Gaussian, sum across levels, store in shared memory

**Implementation:**
```cuda
// Add to shared memory declaration (around line 270)
extern __shared__ char shared_data[];
float3* collected_diffuse_rgb = (float3*)(shared_data + existing_offset);  // After other shared memory

// In preprocessing phase (before pixel loop, around line 530)
// Thread 0 queries diffuse RGB for this Gaussian
if(threadIdx.x == 0 && threadIdx.y == 0 && D_diffuse == 3) {
    const float3 pk = collected_pk[0];  // Gaussian center for this block
    
    // Query diffuse hashgrid at Gaussian center
    float rgb_levels[16 * 3];  // Max 16 levels × 3 RGB
    query_feature<false, 16 * 3, 3>(
        rgb_levels, 
        pk,  // Query at Gaussian center
        voxel_min_diffuse, 
        voxel_max_diffuse, 
        collec_offsets_diffuse,
        appearance_level, 
        hash_features_diffuse, 
        level, 
        l_scale_diffuse, 
        Base_diffuse, 
        align_corners_diffuse, 
        interp_diffuse, 
        contract, 
        debug
    );
    
    // SUM across all levels to get single RGB
    float rgb_sum[3] = {0.0f, 0.0f, 0.0f};
    for(int lv = 0; lv < level; lv++){
        rgb_sum[0] += rgb_levels[lv * 3 + 0];
        rgb_sum[1] += rgb_levels[lv * 3 + 1];
        rgb_sum[2] += rgb_levels[lv * 3 + 2];
    }
    
    // Store in shared memory for all threads to access
    collected_diffuse_rgb[0] = make_float3(rgb_sum[0], rgb_sum[1], rgb_sum[2]);
}
__syncthreads();  // Ensure all threads see the diffuse RGB
```

#### 3.2: Per-Pixel Loop - Process View Features and Alpha Blend

**Location:** Around line 594-750 in the main rendering loop

**Current `case 12` logic:** Query 12 features, dot product, alpha blend

**New logic for `D=12, D_diffuse=3`:**

```cuda
// Detect dual hashgrid mode
if(D == 12 && D_diffuse == 3) {
    // DUAL HASHGRID MODE: Query view features at intersection
    
    // 1. Query view-dependent hashgrid at 3D intersection point
    const int vec_buffer_size = level * 12;
    float feat_vec[16 * 12];  // Max 16 levels × 12 features
    
    query_feature<false, 16 * 12, 12>(
        feat_vec, 
        xyz,  // Query at intersection point
        voxel_min, 
        voxel_max, 
        collec_offsets,
        appearance_level, 
        hash_features, 
        level, 
        l_scale, 
        Base, 
        align_corners, 
        interp, 
        contract, 
        debug
    );
    
    // 2. Dot product with Gaussian normal (concatenate results)
    for(int lv = 0; lv < level; lv++){
        int vec_start = lv * 12;
        int scalar_start = lv * 4;
        
        for(int i = 0; i < 4; i++){
            float dot_prod = 0.0f;
            for(int j = 0; j < 3; j++){
                dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
            }
            feat[scalar_start + i] = -dot_prod;
        }
    }
    
    // 3. Alpha-blend view features
    for (int ch = 0; ch < CHANNELS; ch++)  // CHANNELS = 24
        C[ch] += feat[ch] * w;
    
    // 4. Alpha-blend diffuse RGB (from shared memory)
    float3 diffuse_rgb = collected_diffuse_rgb[j];  // j is Gaussian index in block
    C_diffuse[0] += diffuse_rgb.x * w;
    C_diffuse[1] += diffuse_rgb.y * w;
    C_diffuse[2] += diffuse_rgb.z * w;
}
```

#### 3.3: Output Both Feature Sets

**Location:** Around line 750-800 (after rendering loop)

```cuda
// Write view features to output (for MLP)
if (inside) {
    for (int ch = 0; ch < CHANNELS; ch++)
        out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    
    // Write diffuse RGB to separate output channel (or auxiliary buffer)
    // Option 1: Use out_others for diffuse RGB
    out_others[pix_id + DIFFUSE_R_OFFSET * H * W] = C_diffuse[0];
    out_others[pix_id + DIFFUSE_G_OFFSET * H * W] = C_diffuse[1];
    out_others[pix_id + DIFFUSE_B_OFFSET * H * W] = C_diffuse[2];
}
```

### STEP 4: Update Backward Kernel

**File:** `cuda_rasterizer/backward.cu`

The backward pass needs to:
1. Backprop gradients through view features (same as before)
2. Backprop gradients through diffuse RGB (new)

#### 4.1: Gradient w.r.t. View Features

Keep existing logic for `case 12` but ensure it only affects view hashgrid.

#### 4.2: Gradient w.r.t. Diffuse RGB

**New code needed (around line 820-900):**

```cuda
if(D == 12 && D_diffuse == 3) {
    // Gradient from diffuse RGB (comes from dL/d(final_rgb) in Python)
    float dL_drgb_diffuse[3] = {
        dL_dpixels[pix_id + H * W * 0],  // R channel
        dL_dpixels[pix_id + H * W * 1],  // G channel
        dL_dpixels[pix_id + H * W * 2]   // B channel
    };
    
    // Backprop to diffuse hashgrid
    // dL/d(diffuse_rgb) = dL/d(final_rgb) (no MLP in between)
    float grad_diffuse[3] = {
        alpha * T * dL_drgb_diffuse[0],
        alpha * T * dL_drgb_diffuse[1],
        alpha * T * dL_drgb_diffuse[2]
    };
    
    // Query diffuse hashgrid features (for gradient computation)
    float rgb_levels[16 * 3];
    query_feature<false, 16 * 3, 3>(
        rgb_levels, 
        pk,  // Gaussian center
        voxel_min_diffuse, 
        voxel_max_diffuse, 
        collec_offsets_diffuse,
        appearance_level, 
        hash_features_diffuse, 
        level, 
        l_scale_diffuse, 
        Base_diffuse, 
        align_corners_diffuse, 
        interp_diffuse, 
        contract, 
        debug
    );
    
    // Expand gradient to all levels (gradient is same for all levels due to sum)
    float grad_rgb_levels[16 * 3];
    for(int lv = 0; lv < level; lv++){
        for(int c = 0; c < 3; c++){
            grad_rgb_levels[lv * 3 + c] = grad_diffuse[c];
        }
    }
    
    // Backprop through hashgrid interpolation
    float dummy[16 * 3];
    query_feature<true, 16 * 3, 3>(
        dummy, 
        pk, 
        voxel_min_diffuse, 
        voxel_max_diffuse, 
        collec_offsets_diffuse,
        appearance_level, 
        hash_features_diffuse, 
        level, 
        l_scale_diffuse, 
        Base_diffuse, 
        align_corners_diffuse, 
        interp_diffuse, 
        contract, 
        debug, 
        grad_rgb_levels,  // Gradient input
        dL_dfeatures_diffuse,  // Gradient output to hashgrid
        nullptr  // No position gradient needed
    );
}
```

### STEP 5: Update Backward Function Signature

**File:** `cuda_rasterizer/rasterizer.h`

Add diffuse hashgrid parameters to `backward()` signature (similar to forward).

**File:** `cuda_rasterizer/rasterizer_impl.cu`

Update backward wrapper to pass diffuse parameters.

**File:** `rasterize_points.cu` (C++ binding)

Around line 200-380, update `RasterizeGaussiansBackwardCUDA()`:
1. Add diffuse hashgrid parameters to function signature
2. Create `dL_dfeatures_diffuse` gradient tensor
3. Pass to CUDA backward kernel
4. Return gradient tensor

### STEP 6: Python Side - Combine Outputs

**File:** `gaussian_renderer/__init__.py`

Around line 217-280, after rasterizer returns:

```python
if ingp.method == 'surface_rgb':
    # rendered_image contains view features (24D)
    # Extract diffuse RGB from allmap
    diffuse_rgb = allmap[DIFFUSE_OFFSET:DIFFUSE_OFFSET+3]  # Shape: (3, H, W)
    
    # Pass view features to MLP
    H, W = rendered_image.shape[1], rendered_image.shape[2]
    view_features = rendered_image.view(24, -1).permute(1, 0)  # (H*W, 24)
    
    if ingp.view_dep:
        rays_d, rays_o = cam2rays(viewpoint_camera)
        rays_d = rays_d.reshape(H, W, 3)
        rays_d_flat = rays_d.reshape(-1, 3)  # (H*W, 3)
        
        # MLP produces view-dependent RGB
        rgb_view = ingp.rgb_decode(view_features, rays_d_flat)  # (H*W, 3)
        rgb_view = rgb_view.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    else:
        rgb_view = ingp.mlp_rgb(view_features).reshape(H, W, 3).permute(2, 0, 1)
        rgb_view = torch.sigmoid(rgb_view)
    
    # Combine: final = diffuse + view-dependent
    image = diffuse_rgb + rgb_view
    image = torch.clamp(image, 0.0, 1.0)
else:
    # Existing logic for other modes
    image = rendered_image
```

---

## Key Implementation Details

### Memory Layout

**Shared Memory Structure:**
```cuda
extern __shared__ char shared_data[];
// Existing allocations (xyz, conic, opacity, etc.)
float3* collected_diffuse_rgb = (float3*)(shared_data + offset_after_existing);
```

**Shared memory size calculation** (in preprocessing):
```cpp
// Add to shared memory size calculation (rasterizer_impl.cu)
size_t shared_mem_size = existing_size;
if(D == 12 && D_diffuse == 3) {
    shared_mem_size += BLOCK_SIZE * sizeof(float3);  // For diffuse RGB
}
```

### Output Tensor Dimensions

For `surface_rgb` mode with dual hashgrids:
- **Primary output** (`out_color`): `(24, H, W)` - view features only
- **Auxiliary output** (`out_others`): Extend to include 3 channels for diffuse RGB
  - Add `DIFFUSE_R_OFFSET`, `DIFFUSE_G_OFFSET`, `DIFFUSE_B_OFFSET` to config

Update `out_dim` calculation in `rasterize_points.cu` (line 134):
```cpp
int out_dim = 3+3+1+1+3+3;  // Original outputs
if(is_surface_rgb) {
    out_dim += 3;  // Add 3 for diffuse RGB
}
```

### Hashgrid Query Parameters

**View features hashgrid:**
- Query at: `xyz` (3D intersection point)
- Features: 12 per level
- Processing: Dot product with normal → 4 per level → Concatenate

**Diffuse RGB hashgrid:**
- Query at: `pk` (Gaussian center)
- Features: 3 per level (RGB)
- Processing: Sum across levels → Single RGB triplet

### Alpha Blending

**View features:** Standard alpha compositing
```cuda
for (int ch = 0; ch < 24; ch++)
    C[ch] += feat[ch] * w;  // w = alpha * T
```

**Diffuse RGB:** Standard alpha compositing (same as Gaussian splatting)
```cuda
C_diffuse[0] += rgb.x * w;
C_diffuse[1] += rgb.y * w;
C_diffuse[2] += rgb.z * w;
```

---

## Testing Strategy

### 1. Unit Tests

**Test diffuse RGB query:**
```python
# In Python, manually query diffuse hashgrid
gaussian_centers = pc.get_xyz  # (N, 3)
with torch.no_grad():
    # Query all Gaussians
    rgb_per_gaussian = []
    for center in gaussian_centers[:10]:  # Test first 10
        features = ingp.hash_encoding_diffuse(center.unsqueeze(0))  # (1, 18) = 6 levels × 3
        features = features.reshape(6, 3)  # (levels, RGB)
        rgb = features.sum(dim=0)  # Sum across levels
        rgb_per_gaussian.append(rgb)
    print("Per-Gaussian diffuse RGB:", torch.stack(rgb_per_gaussian))
```

**Test view feature query:**
```python
# Query at intersection points
intersection_points = torch.randn(10, 3).cuda()  # Sample points
normals = torch.randn(10, 3).cuda()
normals = normals / normals.norm(dim=-1, keepdim=True)

with torch.no_grad():
    features = ingp.hash_encoding_view_features(intersection_points)  # (10, 72) = 6×12
    features = features.reshape(10, 6, 12)  # (N, levels, 12)
    
    # Dot product per level
    for lv in range(6):
        feat_lv = features[:, lv, :].reshape(10, 4, 3)  # (N, 4, 3)
        scalars = -(feat_lv * normals.unsqueeze(1)).sum(dim=-1)  # (N, 4)
        print(f"Level {lv} scalars shape:", scalars.shape)
```

### 2. Integration Tests

**Simple scene:**
1. Train with a single Gaussian on a colored sphere
2. Verify diffuse RGB matches sphere color
3. Verify view-dependent effects (specular highlights) are learned

**Gradient check:**
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Forward + backward pass
loss = compute_loss(rendered_image, gt_image)
loss.backward()

# Check gradients exist and are finite
assert ingp.hash_encoding_diffuse.parameters()[0].grad is not None
assert torch.isfinite(ingp.hash_encoding_diffuse.parameters()[0].grad).all()
assert ingp.hash_encoding_view_features.parameters()[0].grad is not None
assert torch.isfinite(ingp.hash_encoding_view_features.parameters()[0].grad).all()
```

### 3. Visual Tests

**Checkpoints:**
- Diffuse-only render (set MLP output to zero): Should show base colors
- View-only render (set diffuse to zero): Should show specular/view effects only
- Combined render: Should show full appearance

---

## Potential Issues & Solutions

### Issue 1: Shared Memory Overflow

**Symptom:** CUDA kernel crashes or returns garbage

**Solution:**
- Check shared memory limit: `cudaDeviceGetAttribute(... cudaDevAttrMaxSharedMemoryPerBlock ...)`
- Reduce `BLOCK_SIZE` if needed
- Or: Query diffuse RGB in global memory instead of shared (slower but safer)

### Issue 2: Gradient Vanishing for Diffuse RGB

**Symptom:** Diffuse hashgrid doesn't train, remains at initialization

**Solution:**
- Check gradient flow: Print `dL_dfeatures_diffuse` in backward pass
- Verify sum operation is differentiable (it should be)
- Check learning rate for diffuse hashgrid (may need separate LR)

### Issue 3: Output Dimensions Mismatch

**Symptom:** Python errors about tensor shape mismatch

**Solution:**
- Double-check `C` calculation in `rasterize_points.cu` (line 123)
- Verify MLP input dimension matches view feature output (24D)
- Check `out_others` has space for diffuse RGB

### Issue 4: Color Saturation

**Symptom:** Diffuse RGB is too bright (>1.0) or clips

**Solution:**
- Initialize diffuse hashgrid with smaller values
- Add sigmoid or tanh activation after sum:
  ```cuda
  rgb_sum[c] = tanhf(rgb_sum[c]);  // or sigmoid
  ```
- Or apply activation in Python before adding to MLP output

---

## Configuration

### For Training with Dual Hashgrid Mode

In your config YAML, you can optionally add:

```yaml
encoding:
  type: hashgrid
  levels: 6
  hashgrid:
      min_logres: 9
      max_logres: 11
      dict_size: 23
      dim: 4  # Base dimension (will be 3x for view features)
      range: [-10, 10]
  
  # Optional: Separate settings for diffuse hashgrid
  hashgrid_diffuse:
      dict_size: 20  # Can be smaller since it's just RGB
      # Other params inherited from main hashgrid
```

### Python Side Enable

In `train.py`, pass `method='surface_rgb'` to INGP:

```python
ingp_model = INGP(cfg_model, method='surface_rgb')  # Instead of 'surface'
```

---

## Compilation

After making CUDA changes:

```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda activate nest_splatting
python setup.py install
```

**Expect warnings about:**
- Unused variables (normal for template code)
- Float conversions (acceptable)

**Watch for errors about:**
- Undefined symbols
- Template instantiation failures
- Shared memory size errors

---

## References

### Key Files and Line Numbers

| File | Lines | Description |
|------|-------|-------------|
| `hash_encoder/modules.py` | 49-100 | INGP `__init__` with dual hashgrid creation |
| `hash_encoder/modules.py` | 195-258 | `build_encoding_diffuse()` and `build_encoding_view_features()` |
| `gaussian_renderer/__init__.py` | 134-167 | Extract parameters from both hashgrids |
| `gaussian_renderer/__init__.py` | 217-280 | **TODO:** Combine diffuse + MLP output |
| `rasterize_points.cu` | 110-139 | Detect `surface_rgb` mode, calculate `C` |
| `forward.cu` | 721-756 | Existing `case 12` logic (modify for dual hashgrid) |
| `forward.cu` | 530-550 | **TODO:** Preprocess phase for diffuse RGB |
| `backward.cu` | 753-820 | Existing `case 12` backward (extend for diffuse) |

### Key Concepts

1. **Gaussian center (`pk`)**: Where diffuse RGB is queried
2. **Intersection point (`xyz`)**: Where view features are queried
3. **Alpha blending weight (`w`)**: `alpha * T` where `alpha = opacity * exp(gaussian_eval)`
4. **Shared memory**: Fast on-chip memory for data reuse within a thread block
5. **Template parameters**: CUDA compile-time constants for buffer sizes

---

## Success Criteria

✅ **Functional:**
- Training runs without CUDA errors
- Both hashgrids receive gradients
- Final RGB looks plausible

✅ **Quality:**
- Diffuse RGB captures base appearance
- MLP output captures view-dependent effects (specular, reflections)
- Combined output matches or exceeds old surface_rgb quality

✅ **Performance:**
- Training speed comparable to old surface_rgb (±10%)
- Memory usage acceptable (check with `nvidia-smi`)

---

## Contact & Questions

When implementing, focus on:
1. **Forward pass first** - Get rendering working
2. **Test with gradients disabled** - Verify output looks reasonable
3. **Implement backward** - Add gradient computation
4. **End-to-end test** - Full training loop

Good luck! This is a substantial refactoring but the architecture is cleaner and more efficient.

