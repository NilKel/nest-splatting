# Normal Normalization Toggle

## Quick Toggle

**To enable/disable normal normalization in surface mode:**

### Enable (Default - Recommended)
```cuda
#define NORMALIZE_SURFACE_NORMALS
```

### Disable
Comment out the define:
```cuda
// #define NORMALIZE_SURFACE_NORMALS
```

---

## Files to Edit

You only need to change **ONE LINE** in **TWO FILES**:

### 1. Forward Pass
**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`  
**Line:** ~119

```cuda
// TOGGLE NORMAL NORMALIZATION: Comment/uncomment the line below
#define NORMALIZE_SURFACE_NORMALS  // ← Comment this line to disable
```

### 2. Backward Pass
**File:** `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`  
**Line:** ~1246

```cuda
// TOGGLE NORMAL NORMALIZATION GRADIENT: Must match forward pass
#define NORMALIZE_SURFACE_NORMALS  // ← Comment this line to disable
```

⚠️ **Important:** The forward and backward passes **MUST match**. If you disable it in forward, disable it in backward too!

---

## After Making Changes

Recompile the CUDA extension:
```bash
cd submodules/diff-surfel-rasterization
conda activate nest_splatting
python setup.py install
```

---

## What Does This Do?

### With Normalization (Enabled)
- Normals are **unit vectors** (length = 1)
- Dot products have consistent magnitude regardless of Gaussian scale
- **Recommended** for most cases - cleaner, more predictable behavior
- Formula: `feat = ReLU(-dot(vec, normalize(normal)) + baseline)`

### Without Normalization (Disabled)
- Normals are **scaled by Gaussian dimensions**
- Larger Gaussians have larger normal magnitudes
- Dot products scale with Gaussian size
- May be useful if you want features to depend on surface area
- Formula: `feat = ReLU(-dot(vec, normal_unnormalized) + baseline)`

---

## Mathematical Details

### Forward Pass (with normalization)
```cuda
n = R * S * [0, 0, 1]  // Normal from rotation and scale
n_normalized = n / |n|  // Unit length
feat = ReLU(-dot(vec, n_normalized))
```

### Backward Pass (with normalization)
The gradient backprops through the normalization:
```cuda
// Chain rule for normalization: df/dn_unnorm = (df/dn_norm - dot(df/dn_norm, n_norm) * n_norm) / |n_unnorm|
dL/dn_unnorm = (dL/dn_norm - dot(dL/dn_norm, n_norm) * n_norm) / |n_unnorm|
```

This ensures gradients flow correctly to the Gaussian rotation and scale parameters.

---

## Current Status

✅ **Currently ENABLED** (as of last compile)

Both forward and backward passes have normalization enabled by default.

---

## Related Features

This normalization is independent of:
- ReLU activation (also recently added)
- Beta annealing
- Dual hashgrid architecture

All these features work together seamlessly.


