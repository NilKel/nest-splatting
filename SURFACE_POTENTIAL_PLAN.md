# Surface Potential Method Implementation Plan

## Branch Setup ✅

- **Branch**: `surface_potential`
- **Remote**: https://github.com/NilKel/nest-splatting.git
- **Status**: Initial commit pushed

## CLI Argument Added ✅

### Usage:
```bash
# Baseline mode (default NeST behavior)
python train.py -s <scene> -m <output> --yaml ./configs/nerfsyn.yaml --method baseline

# Surface potential mode (your new method)
python train.py -s <scene> -m <output> --yaml ./configs/nerfsyn.yaml --method surface
```

### Implementation Location:
- **File**: `train.py` line 470-471
- **Variable**: `args.method` (accessible throughout training)

## Next Steps: Implementing Surface Potential Mode

Based on your vector potential field idea and the codebase analysis:

### 1. Modify Hash Grid Output Shape
**Location**: `hash_encoder/modules.py`

Current output: `(N, F)` where F = levels × features_per_level (e.g., 6 × 4 = 24)

Target output: `(N, F, 3)` for vector potentials

**Changes needed**:
- Line 59: Update `self.feat_dim` calculation to account for 3D vectors
- Line 61: Modify MLP input dimension
- Lines 111-142: Update `build_encoding()` to reshape features

### 2. Implement Dot Product in CUDA Kernel
**Location**: `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`

**Current code** (lines 700-726):
```cuda
// Query hash grid features
query_feature<false, CHANNELS, 4>(feat, xyz, ...);

// Alpha blend features
for (int ch = 0; ch < CHANNELS; ch++)
    C[ch] += feat[ch] * w;
```

**Target code** (lines 700-730):
```cuda
// Query hash grid features → now returns (F, 3) vector potentials
query_feature<false, CHANNELS, 4>(feat_vectors, xyz, ...);

// Get Gaussian normal (already computed at line 114)
float3 normal = {collected_normal[j].x, 
                 collected_normal[j].y, 
                 collected_normal[j].z};

// Compute dot product: surface_feature = -Φ · n
for (int ch = 0; ch < CHANNELS/3; ch++) {
    float dot_product = -(feat_vectors[ch*3 + 0] * normal.x +
                          feat_vectors[ch*3 + 1] * normal.y +
                          feat_vectors[ch*3 + 2] * normal.z);
    
    // Alpha blend the dot product result
    C[ch] += dot_product * w;
}
```

### 3. Condition on args.method
**Location**: `train.py` line 56

Add method parameter to rendering:
```python
render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                   ingp=ingp, beta=beta, iteration=iteration, 
                   cfg=cfg_model, 
                   method=args.method)  # ← Pass method here
```

Then in `gaussian_renderer/__init__.py`:
```python
def render(viewpoint_camera, pc, pipe, bg_color, ..., method="baseline"):
    # ...
    if method == "surface":
        # Use surface potential rendering
        pass
    else:
        # Use baseline NeST rendering
        pass
```

### 4. Key Considerations

#### Normal Consistency Loss
- **Already implemented** at line 195 in `train.py`:
  ```python
  normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
  normal_loss = lambda_normal * (normal_error).mean()
  ```
- Default weight: `lambda_normal = 0.05` (from config)
- This ensures your normals will be clean for the dot product!

#### Feature Dimensions
- Current: 6 levels × 4 features/level = 24 features
- Surface mode: 6 levels × 4 features/level → reshape to (8, 3) vector potentials
  - Or: Use 24 features as 8 × 3D vectors
  - Alternative: Change config to 8 levels × 3 features/level = 24 features = 8 vectors

#### MLP Decoder
- Input will be (8 features, 3D vector) → need to handle variable shape
- May need conditional logic based on method

### 5. Testing Strategy

1. **Verify baseline still works**:
   ```bash
   python train.py -s /path/to/drums -m ./output/test_baseline \
     --yaml ./configs/nerfsyn.yaml --method baseline --iterations 1000
   ```

2. **Test surface mode**:
   ```bash
   python train.py -s /path/to/drums -m ./output/test_surface \
     --yaml ./configs/nerfsyn.yaml --method surface --iterations 1000
   ```

3. **Compare outputs**:
   - Visual quality
   - PSNR/SSIM metrics
   - Number of Gaussians
   - Training speed

## Files to Modify

### High Priority:
1. ✅ `train.py` - Add CLI argument (DONE)
2. `hash_encoder/modules.py` - Modify hash grid output shape
3. `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu` - Implement dot product
4. `gaussian_renderer/__init__.py` - Add method parameter to render()

### Medium Priority:
5. `hash_encoder/modules.py` - Update MLP input handling
6. `configs/nerfsyn.yaml` - Create surface-specific config (optional)

### Low Priority:
7. `eval_render.py` - Add method parameter for evaluation
8. `scripts/nerfsyn_eval.py` - Add method parameter to batch script

## References

- **Hash Grid Query**: Line 708-723 in `forward.cu`
- **Normal Computation**: Line 114 in `forward.cu`
- **Alpha Blending**: Line 725-726 in `forward.cu`
- **Normal Loss**: Line 195-196 in `train.py`
- **Hash Encoder**: `hash_encoder/modules.py` lines 111-142
- **MLP Decoder**: `hash_encoder/modules.py` lines 153-165

## Branch Workflow

```bash
# Make changes
git add <files>
git commit -m "Descriptive message"

# Push to fork
git push origin surface_potential

# When ready, create PR on GitHub
```

