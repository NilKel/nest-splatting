# Hybrid Features Mode Implementation

## Overview

The `hybrid_features` mode combines:
- **N×D per-Gaussian features** (similar to original 2DGS, but configurable)
- **N×D interpolated hashgrid features** (from the finest N levels of a hashgrid)
- **Total: 2×N×D output features** (default: 24D with N=3, D=4)

## Usage

### Basic (Default: 3 levels, 24D total)
```bash
python train.py -s /path/to/data -m output_dir \
  --yaml ./configs/nerfsyn.yaml \
  --eval --iterations 13000 \
  --method hybrid_features
```

### With Custom Levels
```bash
python train.py -s /path/to/data -m output_dir \
  --yaml ./configs/nerfsyn.yaml \
  --method hybrid_features \
  --hybrid_levels 4  # Use 4 levels: 32D total (16D per-Gaussian + 16D hashgrid)
```

See `HYBRID_LEVELS_CLI.md` for detailed examples with different `--hybrid_levels` values.

## Implementation Details

### 1. Hashgrid Configuration
- Uses N levels (configurable via `--hybrid_levels`, default N=3)
- Only the **finest N levels** are queried in CUDA
- Each level contributes D=4 features → N levels × 4D = N×4D hashgrid features

### 2. Per-Gaussian Features
- Each Gaussian has a learnable N×D feature vector (default: 12D for N=3)
- Stored in `gaussians._gaussian_features` tensor
- Dimension automatically scales with `--hybrid_levels`
- Initialized automatically when loading checkpoints (if not present)

### 3. Feature Concatenation (in CUDA)
The CUDA kernel (`render_mode=6`) concatenates:
```
Output[2×N×D] = [per_gaussian_features[N×D] | hashgrid_features[N×D]]
```
For default N=3, D=4:
```
Output[24D] = [per_gaussian_features[12D] | hashgrid_features[12D]]
```

### 4. Key Code Locations

#### Python Side
- **`gaussian_renderer/__init__.py`**: 
  - Lines 122-125: Setup for hybrid_features
  - Lines 85-87: Force `hash_in_CUDA=True` to bypass warmup
  - Line 185: Set `levels=6` for 24D buffer allocation
  - Line 252: Set `render_mode=6`

#### CUDA Side
- **`rasterize_points.cu`**:
  - Lines 161-164: Force `C=24` for forward pass
  - Lines 375-378: Force `C=24` for backward pass
  
- **`cuda_rasterizer/forward.cu`**:
  - Lines 824-852: `case 6` implements hybrid concatenation

#### Model
- **`train.py`**:
  - Lines 100-107: Initialize `_gaussian_features` on checkpoint load

## Building After Changes

**ALWAYS use the rebuild script:**

```bash
cd submodules/diff-surfel-rasterization
conda run -n nest_splatting ./rebuild.sh
```

This ensures the compiled extension is properly installed to site-packages.

## Troubleshooting

### Issue: Output is 12D instead of 24D
- **Cause**: Python is loading old .so from site-packages
- **Fix**: Run `./rebuild.sh` instead of `python setup.py build_ext --inplace`

### Issue: "Unsupported channel count: 12"
- **Cause**: `render_mode` not correctly set or `C` calculation wrong
- **Fix**: Check `render_mode=6` is set and `C=24` is forced in both forward/backward

## Key Fixes Applied

1. **Fixed `C` calculation**: Added `else if(render_mode == 6)` checks in BOTH the `Level > 0` blocks (lines 161 and 375 of `rasterize_points.cu`)
2. **Fixed gradient allocation**: Set `colors_dim = 12` for `render_mode == 6` in backward pass (line 410 of `rasterize_points.cu`)
3. **Fixed Python path issue**: Created `rebuild.sh` to ensure compiled extension is installed to site-packages
4. **Fixed warmup bypass**: Force `hash_in_CUDA=True` for hybrid_features mode
5. **Fixed buffer allocation**: Set `levels=6` to allocate 24D output buffer
6. **Fixed feature passing**: Pass 12D `colors_precomp` directly without padding
7. **Fixed pruning**: Added `_gaussian_features` update in `prune_points()` function (line 355 of `gaussian_model.py`)

## Architecture Flow

```
Training:
├── Load checkpoint → Initialize _gaussian_features[N, 12D]
├── For each iteration:
│   ├── Forward pass:
│   │   ├── Python: Pass 12D per-Gaussian features to CUDA
│   │   ├── CUDA: Query hashgrid at intersection → 12D
│   │   ├── CUDA: Concatenate → 24D output
│   │   └── Python: Decode 24D → RGB via MLP
│   └── Backward pass:
│       ├── Gradient flows back through decoder
│       ├── Split 24D gradient into [12D | 12D]
│       ├── First 12D → per-Gaussian features
│       └── Last 12D → hashgrid
└── Save checkpoint with updated _gaussian_features
```

## Configuration Files

Make sure your YAML config has:
```yaml
ingp_stage:
  switch_iter: 1000  # Warmup iterations (hybrid_features bypasses this)
  n_levels: 6        # Total hashgrid levels (only finest 3 used in CUDA)
  feature_per_level: 4
```

## Notes

- The hashgrid is initialized to half the normal size (3 levels instead of 6)
- Per-Gaussian features are optimized together with other Gaussian parameters
- The mode is designed for fine-grained appearance modeling with per-Gaussian control

