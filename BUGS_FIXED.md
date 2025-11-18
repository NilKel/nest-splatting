# NeST Surface Mode - Bugs Fixed

## Summary

The surface mode was **architecturally correct** but had **6 critical bugs** preventing it from working. The main issue was a **dimension mismatch** where the output tensor was allocated with 72 channels (6 levels × 12 features) but the dot product correctly reduced it to 24 channels (6 levels × 4 features), leaving 48 garbage values that corrupted training.

---

## Architecture Verification ✅

### What Happens in CUDA (Correct!)

1. **3D Sample Points** (lines 682-692 in `forward.cu`)
   ```cuda
   float3 xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
                 s.x * sutu.y + s.y * svtv.y + pk.y,
                 s.x * sutu.z + s.y * svtv.z + pk.z};
   ```
   - Computes ray-Gaussian intersection points
   - NOT just Gaussian centers!

2. **Query Hashgrid** (line 724)
   ```cuda
   query_feature<false, CHANNELS, 12>(feat_vec, xyz, ...)
   ```
   - Queries at 3D intersection point `xyz`
   - Gets 12 vector features (4 base features × 3D vectors)

3. **Dot Product with Normal** (lines 721-743)
   ```cuda
   for(int i = 0; i < 4; i++){
       float dot_prod = 0.0f;
       for(int j = 0; j < 3; j++){
           dot_prod += feat_vec[vec_start + i * 3 + j] * normal[j];
       }
       feat[scalar_start + i] = -dot_prod;
   }
   ```
   - Reduces 12D vectors → 4D scalars **BEFORE alpha blending**

4. **Alpha Blending** (line 783)
   ```cuda
   for (int ch = 0; ch < CHANNELS; ch++)
       C[ch] += feat[ch] * w;
   ```
   - Alpha blends the scalar features

---

## Bug #1: `ingp` Always None When Loading Checkpoint 🐛
**Location:** `train.py` lines 246-256

**Problem:**
```python
elif iteration == cfg_model.ingp_stage.initialize:
    ingp = None  # ← BUG: Always None even with checkpoint!
```

When loading from checkpoint:
- `first_iter = cfg_model.ingp_stage.initialize - 1` (e.g., 9999)
- First training iteration: `iteration = 10000`
- Code caught `iteration == cfg_model.ingp_stage.initialize`
- Set `ingp = None` even though `ingp_model` was already initialized

**Fix:**
```python
elif iteration == cfg_model.ingp_stage.initialize:
    if loaded_pretrained:
        ingp = ingp_model  # ← Use INGP for checkpoint!
    else:
        ingp = None  # Only for training from scratch
```

---

## Bug #2: 🔴 **CRITICAL** - Dimension Mismatch in CUDA
**Location:** `rasterize_points.cu` lines 103-110 (forward) and 261-271 (backward)

**Problem:**
```cpp
D = features.size(1);        // D = 12 (hashgrid features per level)
C = (offsets.size(0) - 1) * D;  // C = 6 * 12 = 72 ← BUG!

// Allocates 72-channel output tensor
torch::Tensor out_color = torch::full({C, H, W}, 0.0, float_opts);

// Kernel dispatched with CHANNELS=72
renderCUDAsurfelForward<72><<<grid, block>>>(...);
```

**Inside kernel:**
```cuda
switch(l_dim) {  // l_dim = 12
    case 12:
        // Dot product reduces to 4 features per level
        feat[0..23] = dot_product_results;  // Only 24 values!
        break;
}

// BUG: Writes all 72 channels, but only 24 are valid!
for (int ch = 0; ch < CHANNELS; ch++)  // CHANNELS = 72
    C[ch] += feat[ch] * w;  // feat[24..71] are GARBAGE!
```

**Result:** 
- Output tensor: `(72, H, W)` but only first `(24, H, W)` are valid
- Python received `surface_features.shape = (H*W, 72)` 
- MLP expected `(H*W, 24)` → dimension mismatch error!

**Fix Part 1 - Tensor Allocation:**
```cpp
D = features.size(1);  // D = 12

// Calculate output dimension based on mode
uint32_t effective_D = D;
if(D == 12) {
    effective_D = 4;  // Surface mode: vectors → scalars
} else if(D == 15) {
    effective_D = 7;  // Surface RGB: 4 scalars + 3 RGB
}

C = (offsets.size(0) - 1) * effective_D;  // C = 6 * 4 = 24 ✅
```

---

## Bug #2b: 🔴 **CRITICAL** - Buffer Overflow in Forward/Backward
**Location:** `forward.cu` lines 723-724 and `backward.cu` lines 758-759

**Problem:**
The template parameter `CHANNELS` is used for BOTH the hashgrid query buffer AND the output buffer, but they have different sizes after the dimension fix!

**Forward:**
```cuda
float feat_vec[CHANNELS];  // BUG: CHANNELS = 24, but hashgrid outputs 72!
query_feature<false, CHANNELS, 12>(feat_vec, ...);  // Buffer overflow!
```

**Backward:**
```cuda
float grad_feat_vec[C * 3];  // C = 24, so C*3 = 72 ✅ (accidentally correct)
// But relies on C being template parameter
```

**Result:**
- Forward: `feat_vec` too small (24 instead of 72) → **BUFFER OVERFLOW**
- Backward: Gradients **not flowing correctly** to hashgrid
- Training: Gaussians increasing instead of being pruned!

**Fix:**
```cuda
// Forward - use fixed-size buffer for hashgrid output
float feat_vec[96];  // Max: 8 levels × 12 features = 96
query_feature<false, 96, 12>(feat_vec, xyz, ...);

// Backward - use fixed-size buffer for hashgrid gradients  
float grad_feat_vec[96];  // Max: 8 levels × 12 features = 96
float feat_dummy[96];  // Dummy array for feat (not used in backward)
query_feature<true, 96, 12>(feat_dummy, xyz, ...grad_feat_vec...);
```

---

## Bug #3: Incorrect Dimension Tracking in INGP
**Location:** `hash_encoder/modules.py` line 152

**Problem:**
```python
def __init__(self, cfg_model, method='surface'):
    if method == 'surface':
        self.level_dim = original_dim  # 4 ✅
    
    self.build_encoding(cfg_model.encoding)

def build_encoding(self, cfg_encoding):
    features_per_level = cfg_encoding.hashgrid.dim  # 12
    self.level_dim = features_per_level  # ← BUG: Overwrites to 12!
```

**Fix:**
```python
def build_encoding(self, cfg_encoding):
    features_per_level = cfg_encoding.hashgrid.dim  # 12
    # Don't overwrite level_dim - it's already correct!
    self.hashgrid_level_dim = features_per_level  # Store separately
```

---

## Bug #4: Dummy Rays Instead of Real Camera Rays
**Location:** `gaussian_renderer/__init__.py` lines 233-241

**Problem:**
```python
# Create dummy rays pointing in -Z
rays_o = torch.zeros((H, W, 3), device='cuda')
rays_d = torch.zeros((H, W, 3), device='cuda')
rays_d[..., 2] = -1.0  # ← BUG: All pixels same direction!
```

This made view-dependent rendering completely wrong.

**Fix:**
```python
# Compute proper camera rays
rays_d, rays_o = cam2rays(viewpoint_camera)
rays_d = rays_d.reshape(H, W, 3)
ray_unit = torch_F.normalize(rays_d, dim=-1).float().detach()
```

---

## Bug #5: Normal Transformation Disabled
**Location:** `gaussian_renderer/__init__.py` lines 187-196

**Problem:**
```python
render_normal = allmap[2:5]  # In view space
# Transformation commented out! ← BUG
# render_normal_transformed = torch.matmul(render_normal_hwc, world_view_rot)
```

Normals stayed in view space, making regularization loss incorrect.

**Fix:**
```python
render_normal = allmap[2:5]
render_normal = render_normal.contiguous()
world_view_rot = viewpoint_camera.world_view_transform[:3,:3].T.contiguous()
H, W = render_normal.shape[1], render_normal.shape[2]
render_normal_hwc = render_normal.permute(1,2,0).reshape(-1, 3)
render_normal_transformed = torch.matmul(render_normal_hwc, world_view_rot)
render_normal = render_normal_transformed.reshape(H, W, 3).permute(2,0,1)
```

---

## Bug #6: Surface Normal Regularization Disabled
**Location:** `gaussian_renderer/__init__.py` lines 216-219

**Problem:**
```python
# surf_normal = depth_to_normal(viewpoint_camera, surf_depth)  # Commented out
surf_normal = torch.zeros_like(render_normal)  # ← BUG: Dummy zeros!
```

**Fix:**
```python
surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
surf_normal = surf_normal.permute(2,0,1)
surf_normal = surf_normal * (render_alpha).detach()
```

---

## Bug #7: Debug Breakpoints Left in Code
**Location:** `gaussian_renderer/__init__.py` lines 184, 232, 330

**Problem:** Three `breakpoint()` calls halted execution.

**Fix:** Removed all breakpoints.

---

## Data Flow Verification

### Hashgrid Query
- **Python → CUDA:** 
  - `features.size(1) = 12` (surface mode)
  - Passed as `D = 12` to CUDA
- **CUDA Kernel:**
  - `l_dim = 12` → dispatches to `case 12:`
  - Queries hashgrid with `query_feature<false, CHANNELS, 12>`
- **Query Location:** 
  - At `xyz` (ray-Gaussian intersection)
  - NOT at `pk` (Gaussian center) ✅

### Dimension Reduction
- **Hashgrid Output:** 12D vectors per level
- **Dot Product:** 12D → 4D scalars per level
- **6 Levels:** 4 × 6 = 24D total
- **Alpha Blending:** (24, H, W) → accumulated per pixel
- **MLP Input:** (H*W, 24) → RGB output

---

## Test Results

After fixes:
- ✅ `surface_features.shape = (H*W, 24)` (was 72)
- ✅ MLP input dimension matches: 24 + 16 (view) = 40
- ✅ No more dimension errors
- ✅ INGP properly initialized from checkpoint
- ✅ Real camera rays used for view-dependent rendering
- ✅ Normal and depth regularization working

---

## Commands to Recompile

```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
conda run -n nest_splatting python setup.py install
```

---

## Files Modified

1. `train.py` - Fixed checkpoint loading logic
2. `gaussian_renderer/__init__.py` - Restored proper ray/normal computation, removed breakpoints
3. `hash_encoder/modules.py` - Fixed dimension tracking
4. `submodules/diff-surfel-rasterization/rasterize_points.cu` - **Fixed critical dimension bug**
5. CUDA rasterizer recompiled with fixes

---

## Conclusion

The surface mode implementation was **architecturally sound**:
- ✅ Queries hashgrid at 3D intersection points
- ✅ Dot products with normals BEFORE alpha blending
- ✅ Correct order of operations

The bugs were all in:
1. **Tensor allocation** (wrong output size)
2. **Python wrapper logic** (checkpoint loading, dummy workarounds)
3. **Dimension tracking** (overwriting correct values)

All bugs are now fixed and the rasterizer is recompiled. The surface mode should work correctly now! 🎉


