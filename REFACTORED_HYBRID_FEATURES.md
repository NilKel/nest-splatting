# Hybrid Features Refactoring - Complete

## Summary

Refactored `hybrid_features` mode to properly split features between per-Gaussian (coarse) and hashgrid (fine) components while maintaining a constant total output dimension.

## Key Design Principles

### Feature Distribution
- **Per-Gaussian features**: `hybrid_levels × D` (coarse levels, no spatial hashing)
- **Hashgrid features**: `(total_levels - hybrid_levels) × D` (fine levels, spatially hashed)
- **Total output**: `total_levels × D` (always constant, e.g., 24D for 6 levels)

### Rationale
✅ **Coarse features** = learned per-Gaussian (good for global appearance)  
✅ **Fine features** = hashgrid (good for spatial detail)

### Edge Cases
- `--hybrid_levels 0`: Pure baseline (0D per-Gaussian + 24D hashgrid)
- `--hybrid_levels 6`: Pure 2DGS (24D per-Gaussian + 0D hashgrid)

## Files Modified

### 1. `hash_encoder/modules.py` (lines 151-210)
**Changes**:
- Calculate `hashgrid_levels = total_levels - hybrid_levels`
- Build hashgrid with `hashgrid_levels` levels (finest N levels)
- Store `self.total_levels` and `self.hashgrid_levels` for dimension calculations
- Handle edge case: `hybrid_levels == total_levels` (pure 2DGS, no hashgrid)

**Key code**:
```python
per_gaussian_dim = hybrid_levels * original_dim
hashgrid_levels = original_levels - hybrid_levels
hashgrid_dim = hashgrid_levels * original_dim
total_dim = original_levels * original_dim  # Always constant
```

### 2. `gaussian_renderer/__init__.py` (lines 119-126, 181-193)
**Changes**:
- Set `feat_dim = ingp.total_levels * ingp.level_dim` (always constant)
- Encode both `total_levels` and `hybrid_levels` into `levels` parameter:
  ```python
  levels = (ingp.total_levels << 16) | ingp.hybrid_levels
  ```
- Example: `(6 << 16) | 3 = 393219` for `total_levels=6, hybrid_levels=3`

### 3. `cuda_rasterizer/forward.cu` (lines 824-860)
**Changes**:
- Decode `levels` parameter:
  ```cuda
  const int total_levels = level >> 16;  // High 16 bits
  const int hybrid_levels = level & 0xFFFF;  // Low 16 bits
  const int hashgrid_levels = total_levels - hybrid_levels;
  ```
- Query hashgrid with `hashgrid_levels` (skip if 0)
- Concatenate: `[per_gaussian (hybrid_levels×D) | hashgrid (hashgrid_levels×D)]`

### 4. `cuda_rasterizer/backward.cu` (lines 900-934)
**Changes**:
- Same decoding logic as forward pass
- Split gradients:
  - First `hybrid_levels×D` → per-Gaussian features
  - Last `hashgrid_levels×D` → hashgrid
- Skip hashgrid backprop if `hashgrid_levels == 0`

### 5. `rasterize_points.cu` (lines 157-163, 372-378, 408-421)
**Changes**:
- **Forward buffer allocation**:
  ```cpp
  uint32_t total_levels = Level >> 16;
  C = total_levels * D;  // e.g., 6 × 4 = 24D
  ```
- **Backward `dL_dcolors` allocation**:
  ```cpp
  if (render_mode == 6) {
      uint32_t hybrid_levels = Level & 0xFFFF;
      colors_dim = hybrid_levels * D;  // e.g., 3 × 4 = 12D
  }
  ```

## Usage Examples

```bash
# Pure baseline (no per-Gaussian features)
python train.py ... --method hybrid_features --hybrid_levels 0

# Half-and-half (default)
python train.py ... --method hybrid_features --hybrid_levels 3

# Pure 2DGS (no hashgrid)
python train.py ... --method hybrid_features --hybrid_levels 6
```

## Dimension Table

| Mode | hybrid_levels | Per-Gaussian | Hashgrid | Total | Equivalent |
|------|---------------|--------------|----------|-------|------------|
| Pure baseline | 0 | 0D | 24D (6 levels) | 24D | Baseline |
| Hybrid (coarse) | 2 | 8D | 16D (4 levels) | 24D | - |
| **Hybrid (default)** | **3** | **12D** | **12D (3 levels)** | **24D** | **-** |
| Hybrid (fine) | 4 | 16D | 8D (2 levels) | 24D | - |
| Pure 2DGS | 6 | 24D | 0D | 24D | 2DGS |

## Testing Status

✅ All Python files updated  
✅ All CUDA files updated  
✅ Buffer allocation corrected  
✅ CUDA extension compiled successfully  
⏳ **Ready for training tests**

## Next Steps

1. Test `--hybrid_levels 0` (should match baseline performance)
2. Test `--hybrid_levels 3` (default, balanced split)
3. Test `--hybrid_levels 6` (should match 2DGS performance)
4. Compare rendering quality across different splits

## Implementation Notes

- **Encoding scheme**: Using bitwise packing `(total << 16) | hybrid` to pass both values through single `levels` parameter
- **Backward compatibility**: Not maintained with old checkpoints (feature dimensions changed)
- **Hashgrid resolution**: Always uses finest N levels for optimal spatial detail
- **Per-Gaussian resolution**: Represents coarsest N levels (global appearance)

