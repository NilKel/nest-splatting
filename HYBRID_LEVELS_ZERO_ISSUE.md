# Issue with `hybrid_levels=0` (Pure Baseline Mode)

## Problem

When running with `--hybrid_levels 0` (which should behave as pure baseline with only hashgrid features), the code crashes with:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## Root Cause

The issue is multi-fold:

1. **Python Side**: When `hybrid_levels=0`, `_gaussian_features` is created with shape `[N, 0]` (0D per Gaussian).

2. **CUDA Forward Pass**: In `forward.cu` line 849-854, the code tries to access `features[gauss_id * per_gaussian_dim]` even when `per_gaussian_dim=0`. With an empty `features` tensor, this causes illegal memory access.

3. **CUDA Backward Pass**: Similar issue in `backward.cu` line 923-926 where it tries to atomicAdd to `dL_dcolors` even when `per_gaussian_dim=0`.

## Solutions Attempted

### Fix 1: Added `if (hybrid_levels > 0)` guards in CUDA ✅
- Forward pass (line 848-855): Only copy per-Gaussian features if `hybrid_levels > 0`
- Backward pass (line 923-927): Only backprop to per-Gaussian features if `hybrid_levels > 0`

### Fix 2: Python side passes empty tensor ✅
- `gaussian_renderer/__init__.py` line 126-128: Create empty tensor `torch.empty((N, 0), device='cuda')` for `hybrid_levels=0`

## Status

The fix has been applied to:
- ✅ `cuda_rasterizer/forward.cu`
- ✅ `cuda_rasterizer/backward.cu`  
- ✅ `gaussian_renderer/__init__.py`
- ✅ **CUDA extension rebuilt**

## Testing

After rebuilding, test with:
```bash
python train.py -s /path/to/data -m output \
  --yaml ./configs/nerfsyn.yaml --eval --iterations 10100 \
  --method hybrid_features --hybrid_levels 0
```

This should behave identically to the baseline method (pure hashgrid, no per-Gaussian features).

## Expected Behavior for Different `hybrid_levels`

- **`hybrid_levels=0`**: Pure baseline (24D from hashgrid only)
- **`hybrid_levels=3`**: 12D per-Gaussian + 12D hashgrid = 24D total
- **`hybrid_levels=6`**: Pure 2DGS (24D per-Gaussian only, no hashgrid)




