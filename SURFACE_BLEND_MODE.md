# Surface Blend Mode

## Overview

`surface_blend` is a new rendering mode that **alpha-blends vector potentials first, then applies dot product with blended normals in Python**.

### Key Difference from Surface Mode

| Mode | Dot Product Location | When |
|------|---------------------|------|
| **surface** | Per-Gaussian in CUDA | Before alpha blending |
| **surface_blend** | Per-pixel in Python | After alpha blending |

### Formula

**Surface mode:**
```
For each Gaussian i:
  scalar_i = ReLU(-dot(vector_i, normal_i) + baseline_i)
Final = Σ(scalar_i × alpha_i × T_i)  # Alpha blend scalars
```

**Surface blend mode:**
```
Blended_vectors = Σ(vector_i × alpha_i × T_i)   # Alpha blend vectors first
Blended_normals = Σ(normal_i × alpha_i × T_i)   # Alpha blend normals
scalar = -dot(Blended_vectors, Blended_normals)  # Dot product AFTER blending
Final = MLP(scalar)
```

---

## Architecture

### 1. CUDA Forward Pass (case 13)

**File:** `cuda_rasterizer/forward.cu` line ~790

```cuda
case 13: {
    // Query 13 features per level (13 is a marker, only use first 12)
    query_feature<false, 16 * 13, 13>(feat_vec, xyz, ...);
    
    // Copy first 12 features (vectors), ignore 13th
    for(int lv = 0; lv < level; lv++){
        for(int i = 0; i < 12; i++){
            feat[lv * 12 + i] = feat_vec[lv * 13 + i];
        }
    }
    // Output: level × 12 dims (vectors, no dot product yet)
}
```

**These 12D vectors per level are alpha-blended like colors.**

### 2. Python Dot Product

**File:** `gaussian_renderer/__init__.py` line ~303

```python
elif ingp.method == 'surface_blend':
    # rendered_image: (levels × 12, H, W) - blended vectors
    # render_normal: (3, H, W) - blended normals
    
    # Reshape to (H*W, levels, 4, 3)
    blended_vectors = rendered_image.view(levels, 12, H, W)
    blended_vectors = blended_vectors.reshape(H * W, levels, 4, 3)
    
    # Get blended normals: (H*W, 3)
    blended_normals = render_normal.reshape(H * W, 3)
    blended_normals = F.normalize(blended_normals, dim=-1)
    
    # Dot product: -<vectors, normals>
    scalar_features = -torch.einsum('nlfc,nc->nlf', blended_vectors, blended_normals)
    # Output: (H*W, levels × 4)
    
    # Pass to MLP
    rgb = ingp.rgb_decode(scalar_features, rays_dir)
```

### 3. Backward Pass (case 13)

**File:** `cuda_rasterizer/backward.cu` line ~870

```cuda
case 13: {
    // Gradients from Python dot product come through grad_feat (level × 12)
    // Expand back to 13 dimensions for hashgrid query
    float grad_feat_vec[16 * 13];
    for(int lv = 0; lv < level; lv++){
        for(int i = 0; i < 12; i++){
            grad_feat_vec[lv * 13 + i] = grad_feat[lv * 12 + i];
        }
        grad_feat_vec[lv * 13 + 12] = 0.0f;  // 13th dim gets zero gradient
    }
    
    // Backprop through hashgrid
    query_feature<true, 16 * 13, 13>(..., grad_feat_vec, dL_dfeatures, dL_dxyz);
}
```

---

## Usage

### Configuration

In `hash_encoder/modules.py`:

```python
ingp_model = INGP(cfg_model, method='surface_blend')
```

The hashgrid will be configured with 13 features per level (where 13 is a marker):
- 12 actual features (4 base × 3 for vectors)
- 1 marker feature (unused, helps distinguish from surface mode)

### Training

Use the training script with `--method surface_blend`:

```bash
python train.py -s data/nerf_synthetic/hotdog \
  -m outputs/surface_blend_test \
  --config configs/nerfsyn.yaml \
  --method surface_blend \
  --ingp
```

### Evaluation

Use the eval script with the same method:

```bash
python eval_render.py -m outputs/surface_blend_test \
  --yaml nerfsyn \
  --method surface_blend \
  --skip_mesh
```

---

## Why Use Surface Blend?

### Advantages

1. **Smooth blending**: Vectors blend smoothly before dot product
   - Surface mode: Sharp transitions between Gaussians (dot product per-Gaussian)
   - Surface blend: Smooth interpolation in vector space

2. **Consistent with volume rendering**: Matches standard alpha compositing
   - Features blend like colors/densities

3. **Python flexibility**: Dot product in Python is differentiable and flexible
   - Easy to modify or experiment with different operations

### Disadvantages

1. **Slightly slower**: Extra Python computation for dot product
   - CUDA does less work, but Python overhead

2. **Less local control**: Dot product operates on blended quantities
   - Can't enforce per-Gaussian constraints (like ReLU)

---

## Technical Details

### Why D=13 as a Marker?

The hashgrid dimension signals the mode to the CUDA kernel:
- `D=12`: Surface mode (dot product in CUDA)
- `D=13`: Surface blend mode (dot product in Python)  
- `D=15`: Surface RGB mode

The 13th feature is allocated but never used - it's just a marker to distinguish the modes.

### Output Dimensions

| Level Dim (D) | Output per Level | Total (6 levels) | Notes |
|---------------|------------------|------------------|-------|
| 12 (surface) | 4 scalars | 24 | Dot product in CUDA |
| 13 (surface_blend) | 12 vectors | 72 | Alpha-blended, dot in Python → 24 |
| 15 (surface_rgb) | 7 (4+3) | 42 | Scalars + RGB |

After Python dot product, surface_blend produces 24D features (same as surface mode).

### Gradient Flow

```
Loss
 ↓
MLP output (RGB)
 ↓
Scalar features (after dot product in Python)
 ↓  ↓
Blended vectors ← Blended normals
 ↓                 ↓
Alpha blending    Alpha blending
 ↓                 ↓
Per-Gaussian      Per-Gaussian normals
vectors           (from rotations)
 ↓
Hashgrid interpolation
 ↓
Hashgrid parameters
```

All parts are differentiable!

---

## Comparison Table

| Feature | Baseline | Surface | Surface Blend | Surface RGB |
|---------|----------|---------|---------------|-------------|
| Hashgrid features/level | 4 | 12 | 13 (use 12) | 15 |
| Dot product location | N/A | CUDA (per-Gaussian) | Python (per-pixel) | CUDA |
| ReLU activation | No | Yes | No | Yes |
| Alpha blend | Scalars | Scalars | **Vectors** | Scalars+RGB |
| Output/level (CUDA) | 4 | 4 | **12** | 7 |
| Output/level (Python) | 4 | 4 | 4 | 7 |
| Best for | Simple | Sharp features | Smooth features | Appearance |

---

## Example Output

With 6 levels and base dimension 4:

**CUDA output shape:** `(72, H, W)` = 6 levels × 12 vectors

**After Python processing:** `(H*W, 24)` = 6 levels × 4 scalars

**Final RGB:** `(3, H, W)`

---

## Files Modified

1. `cuda_rasterizer/forward.cu` - Added case 13 for forward pass
2. `cuda_rasterizer/backward.cu` - Added case 13 for backward pass
3. `rasterize_points.cu` - Recognize D=13 and set output dims
4. `hash_encoder/modules.py` - Added surface_blend mode initialization
5. `gaussian_renderer/__init__.py` - Python dot product implementation

---

## Status

✅ **Implemented and Compiled** (2025-10-20)
- Forward pass: Alpha blend vectors
- Backward pass: Gradients through hashgrid  
- Python: Dot product with blended normals
- Ready for training!

---

## Notes

- No ReLU by design (user requested negated dot product only)
- Normals are normalized before dot product
- Compatible with all existing regularization losses
- Can be combined with view-dependent MLP

