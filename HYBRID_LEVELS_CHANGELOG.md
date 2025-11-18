# Hybrid Levels CLI Argument - Changelog

## Summary

Added `--hybrid_levels N` CLI argument to make `hybrid_features` mode configurable. Now you can control how many of the finest hashgrid levels to use, scaling from 1 to 6 levels.

## What Changed

### 1. CLI Argument
- **File**: `train.py`
- **Added**: `--hybrid_levels` argument (default: 3)
- **Usage**: `python train.py ... --method hybrid_features --hybrid_levels N`

### 2. INGP Module
- **File**: `hash_encoder/modules.py`
- **Constructor**: Added `hybrid_levels` parameter
- **Hashgrid init**: Dynamically creates N-level hashgrid (finest N levels)
- **Feature dim**: Automatically computes `feat_dim = 2×N×D`

### 3. Gaussian Model
- **File**: `scene/gaussian_model.py`
- **Per-Gaussian features**: Dimension scales with `hybrid_levels` (N×D)
- **Initialization**: Automatically uses correct dimension from args

### 4. Renderer
- **File**: `gaussian_renderer/__init__.py`
- **Output dimension**: `feat_dim = 2×ingp.hybrid_levels×ingp.level_dim`
- **Buffer allocation**: `levels = 2×ingp.hybrid_levels`

### 5. CUDA Kernels
- **Files**: `cuda_rasterizer/forward.cu`, `cuda_rasterizer/backward.cu`
- **Dynamic levels**: `actual_hash_levels = level / 2`
- **Variable dimensions**: All loops use computed `per_gaussian_dim = actual_hash_levels × l_dim`

### 6. Buffer Allocation
- **File**: `rasterize_points.cu`
- **Forward**: `C = Level × D`
- **Backward**: `colors_dim = C / 2`

### 7. Documentation
- **Created**: `HYBRID_LEVELS_CLI.md` - Comprehensive usage guide
- **Updated**: `HYBRID_FEATURES_IMPLEMENTATION.md` - Added variable N documentation

## Backward Compatibility

✅ **Fully backward compatible!**
- Default `--hybrid_levels 3` maintains original behavior
- Existing code works without changes
- Checkpoints with 12D features load correctly

## Example Usage

```bash
# Default (same as before): 3 levels, 24D total
python train.py ... --method hybrid_features

# Lightweight: 1 level, 8D total
python train.py ... --method hybrid_features --hybrid_levels 1

# Medium: 2 levels, 16D total
python train.py ... --method hybrid_features --hybrid_levels 2

# Heavyweight: 4 levels, 32D total
python train.py ... --method hybrid_features --hybrid_levels 4

# Maximum: 6 levels, 48D total (all hashgrid levels)
python train.py ... --method hybrid_features --hybrid_levels 6
```

## Testing

- [x] CLI argument appears in `--help`
- [x] CUDA compilation successful
- [x] Variable dimensions handled correctly in Python
- [x] Backward compatibility maintained (default N=3)

## Files Modified

1. `train.py` - Added CLI argument, pass to INGP
2. `hash_encoder/modules.py` - Variable hashgrid initialization
3. `scene/gaussian_model.py` - Variable per-Gaussian feature dimension
4. `gaussian_renderer/__init__.py` - Dynamic output dimension
5. `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu` - Variable N handling
6. `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu` - Variable N gradients
7. `submodules/diff-surfel-rasterization/rasterize_points.cu` - Dynamic buffer allocation

## Documentation Created

1. `HYBRID_LEVELS_CLI.md` - Comprehensive usage guide with examples
2. `HYBRID_LEVELS_CHANGELOG.md` - This file
3. Updated `HYBRID_FEATURES_IMPLEMENTATION.md` - Added variable N details
