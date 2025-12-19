# Bug Fixes - December 18, 2024

## Bug 1: residual_hybrid Output Dimension Mismatch ✅

**Issue**: `residual_hybrid` mode was outputting 64D instead of 4D for hashgrid features, causing MLP input dimension error.

**Root Cause**: Missing `C` dimension calculation for `render_mode=11` in `rasterize_points.cu`.

**Symptoms**:
```
Unsupported channel count: 64
feat.shape: torch.Size([640000, 80])  # Expected: torch.Size([640000, 20])
```

**Fix** (`submodules/diff-surfel-rasterization/rasterize_points.cu`):
```cpp
else if(render_mode == 11) {
    // residual_hybrid mode: dual output buffers (SH RGB + hashgrid features)
    // Output dimension: hashgrid features only (hashgrid_levels × D)
    // Level encoding: (total_levels << 16) | (hashgrid_levels << 8) | hybrid_levels
    uint32_t hashgrid_levels = (Level >> 8) & 0xFF;  // Extract bits 8-15
    C = hashgrid_levels * D;  // e.g., 1 × 4 = 4D (hashgrid features only)
}
```

**Status**: Fixed and compiled ✅

---

## Bug 2: adaptive_cat Optimizer Loading Error ✅

**Issue**: `ValueError: loaded state dict has a different number of parameter groups` when loading warmup checkpoint.

**Root Cause**: New parameters (`_gaussian_features`, `_adaptive_cat_weight`) don't exist in warmup checkpoint optimizer state.

**Symptoms**:
```
ValueError: loaded state dict has a different number of parameter groups
```

**Fix** (`train.py` line 280):
```python
# Load optimizer state from warmup checkpoint
# For cat/adaptive/adaptive_cat/diffuse mode, new params won't be in saved state - they train from scratch
if args.method not in ["cat", "adaptive", "adaptive_cat", "diffuse"]:
    gaussians.optimizer.load_state_dict(ckpt['optimizer_state'])
```

**Status**: Fixed ✅

---

## Bug 3: Coarse-to-Fine Not Disabled for cat Mode ✅

**Issue**: `--disable_c2f` flag was ignored for `cat` mode, C2F was still enabled.

**Root Cause**: C2F override logic didn't check the `--disable_c2f` flag, only checked specific mode names.

**Symptoms**:
```
C2F disabled: all levels active from start  # From train.py
...
If coarse2fine : True  # From modules.py - WRONG!
```

**Fix** (`hash_encoder/modules.py` lines 389-404):
```python
self.level_mask = cfg_encoding.coarse2fine.enabled
# Override C2F if disable_c2f flag is set or for special modes
disable_c2f = args is not None and hasattr(args, 'disable_c2f') and args.disable_c2f
# Diffuse_ngp/diffuse_offset/hybrid_SH/hybrid_SH_raw/hybrid_SH_post/adaptive_cat: override C2F to disabled
if disable_c2f or self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode or self.is_hybrid_sh_mode or self.is_hybrid_sh_raw_mode or self.is_hybrid_sh_post_mode or self.is_adaptive_cat_mode:
    if disable_c2f:
        print(f'If coarse2fine : False (disabled by --disable_c2f flag)')
    else:
        print(f'If coarse2fine : False (disabled for diffuse_ngp/diffuse_offset/hybrid_SH/adaptive_cat modes)')
    self.level_mask = False
    self.init_active_level = 1
    self.step = 1000  # Dummy value
else:
    print(f'If coarse2fine : {self.level_mask}')
    if self.level_mask:
        self.init_active_level = cfg_encoding.coarse2fine.init_active_level
        self.step = cfg_encoding.coarse2fine.step
```

**Status**: Fixed ✅

---

## Bug 4: NameError for 'args' in Renderer ✅

**Issue**: `NameError: name 'args' is not defined` in `gaussian_renderer/__init__.py`.

**Root Cause**: Renderer function doesn't receive `args` parameter, only `cfg`. Attempted to check `args.adaptive_cat_inference` flag directly.

**Symptoms**:
```
NameError: name 'args' is not defined
  File "/home/nilkel/Projects/nest-splatting/gaussian_renderer/__init__.py", line 248, in render
    use_inference_mode = hasattr(args, 'adaptive_cat_inference') and args.adaptive_cat_inference
```

**Fix** (2 files):

1. **Store flag in INGP object** (`hash_encoder/modules.py` line 71):
```python
# Store args for adaptive_cat mode configuration (cat with learnable binary blend weights)
self.is_adaptive_cat_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_cat"
self.adaptive_cat_inference = args is not None and hasattr(args, 'adaptive_cat_inference') and args.adaptive_cat_inference
```

2. **Read from INGP object** (`gaussian_renderer/__init__.py` line 248):
```python
# Inference flag: 1 = binary mode (skip intersection or skip Gaussian), 0 = training mode (smooth blend)
# Check if inference flag is set in ingp object (stored during initialization)
use_inference_mode = hasattr(ingp, 'adaptive_cat_inference') and ingp.adaptive_cat_inference
```

**Status**: Fixed ✅

---

## Testing Status

### residual_hybrid mode
- ✅ Code compiles
- ✅ Output dimension correct (4D hashgrid features)
- 🔄 Full training test pending

### adaptive_cat mode
- ✅ Code compiles
- ✅ Optimizer loading works
- ✅ Inference flag accessible
- 🔄 Full training test pending

### cat mode with --disable_c2f
- ✅ Code compiles
- ✅ C2F correctly disabled
- 🔄 Full training test pending

---

## Files Modified

1. `submodules/diff-surfel-rasterization/rasterize_points.cu` - Added render_mode=11 dimension calculation
2. `train.py` - Added "adaptive_cat" to optimizer skip list
3. `hash_encoder/modules.py` - Fixed C2F override logic, added adaptive_cat_inference storage
4. `gaussian_renderer/__init__.py` - Fixed args reference to use ingp.adaptive_cat_inference

---

## Next Steps

1. **Test residual_hybrid mode**:
   ```bash
   python train.py -s <data_path> -m test_residual_hybrid \
     --yaml ./configs/nerfsyn.yaml \
     --method residual_hybrid --hybrid_levels 5 --disable_c2f \
     --iterations 30000 --eval
   ```

2. **Test adaptive_cat mode**:
   ```bash
   python train.py -s <data_path> -m test_adaptive_cat \
     --yaml ./configs/nerfsyn.yaml \
     --method adaptive_cat \
     --iterations 30000 --eval
   ```

3. **Test cat mode with C2F disabled**:
   ```bash
   python train.py -s <data_path> -m test_cat_noc2f \
     --yaml ./configs/nerfsyn.yaml \
     --method cat --hybrid_levels 5 --disable_c2f \
     --iterations 30000 --eval
   ```

All modes should now train successfully without errors!
