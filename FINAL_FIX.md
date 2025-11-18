# Final Fix: hybrid_levels=0 Bug

## The Bug

When `hybrid_levels=0`, we were setting `render_mode=1` (thinking this was "baseline mode").

**BUT**: In the CUDA code (`forward.cu`), there is **NO case 1**! The switch statement goes:
- `case 0`: 2DGS and baseline (handles l_dim=4)
- `case 4`: baseline_double
- `case 5`: baseline_blend_double
- `case 6`: hybrid_features
- `case 12`: surface_rgb

Setting `render_mode=1` caused it to hit the **default case**, printing "FW unsupported level dim : 4"

## The Fix

**Use `render_mode=0` for BOTH `hybrid_levels=0` and `hybrid_levels=total_levels`**

Both cases should use CUDA's `case 0` which handles baseline mode perfectly:

```python
if ingp.hybrid_levels == 0:
    # Pure baseline: use render_mode 0
    render_mode = 0
elif ingp.hybrid_levels == ingp.total_levels:
    # Pure per-Gaussian: use render_mode 0  
    render_mode = 0
else:
    # True hybrid: use render_mode 6
    render_mode = 6
```

## Why This Works

**CUDA `case 0`** handles:
- Standard 2DGS with `colors_precomp`
- Baseline mode with hashgrid querying (when `level > 0`)
- Various l_dim values: 2, 4, 8, 12

So for:
- `hybrid_levels=0`: `render_mode=0`, `level=6`, `colors_precomp=None` → queries hashgrid (baseline)
- `hybrid_levels=6`: `render_mode=0`, `level=0`, `colors_precomp=24D` → uses per-Gaussian (2DGS)

Both work with the same CUDA case!

## Files Modified

1. ✅ `gaussian_renderer/__init__.py`: Changed `render_mode=1` to `render_mode=0` for `hybrid_levels=0`

## Testing

```bash
# Pure baseline (hybrid_levels=0)
python train.py -s /path/to/data -m output \
  --yaml ./configs/nerfsyn.yaml --eval --iterations 10100 \
  --method hybrid_features --hybrid_levels 0

# True hybrid (hybrid_levels=1-5)
python train.py -s /path/to/data -m output \
  --yaml ./configs/nerfsyn.yaml --eval --iterations 10100 \
  --method hybrid_features --hybrid_levels 3

# Pure 2DGS (hybrid_levels=6)
python train.py -s /path/to/data -m output \
  --yaml ./configs/nerfsyn.yaml --eval --iterations 10100 \
  --method hybrid_features --hybrid_levels 6
```

All three cases should now work perfectly!




