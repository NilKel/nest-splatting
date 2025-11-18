# Fix for Hybrid Features "Garbage Output" Issue

## Problem

When implementing configurable `--hybrid_levels`, the output was producing garbage instead of reasonable images. This was worse than the original fixed 3-level implementation.

## Root Cause

The hashgrid initialization in `hash_encoder/modules.py` was incorrectly calculating the starting resolution for the hashgrid. The bug was in using `ceil(log2(start_res))` which rounds the resolution and breaks the per-level scale continuity.

### Example of the Bug

For `hybrid_levels=1` with original grid `[128, 169, 223, 294, 388, 512]`:
- **Intended**: Hashgrid should start at 169 (level 1 of original grid)
- **Actual**: `ceil(log2(169)) = 8` → `2^8 = 256` (wrong!)
- **Result**: Hashgrid uses `[256, 304, 361, 428, 512]` instead of `[169, 223, 294, 388, 512]`

This mismatch meant the hashgrid resolutions didn't align with what the CUDA code expected, causing mismatched feature queries and garbage output.

## Solution

### 1. Fixed Resolution Calculation (`hash_encoder/modules.py` lines 181-205)

Instead of using log2 rounding, we now:
1. Calculate the **exact** starting resolution: `start_res = original_base * (original_scale ** hybrid_levels)`
2. Calculate the **exact** per-level scale for the hashgrid: `hashgrid_scale = (finest / start_res) ** (1 / (hashgrid_levels - 1))`
3. Pass these values directly to the hashgrid initialization

```python
# Calculate exact resolutions to match the original grid's finest levels
start_level = hybrid_levels
original_scale = (original_finest / original_base) ** (1 / (original_levels - 1))
start_res = original_base * (original_scale ** start_level)

# Calculate hashgrid scale
hashgrid_scale = (original_finest / start_res) ** (1 / (hashgrid_levels - 1))

# Store for hashgrid init
cfg_encoding_hybrid.hashgrid.base_resolution = int(np.round(start_res))
cfg_encoding_hybrid.hashgrid.per_level_scale = float(hashgrid_scale)
cfg_encoding_hybrid.hashgrid.finest_resolution = original_finest
```

### 2. Updated `build_encoding` to Use Explicit Values (`hash_encoder/modules.py` lines 275-286)

Modified `build_encoding` to check for and use explicit `base_resolution` and `per_level_scale` when provided:

```python
# Check if explicit base_resolution and per_level_scale are provided (hybrid mode)
if hasattr(cfg_encoding.hashgrid, 'base_resolution') and hasattr(cfg_encoding.hashgrid, 'per_level_scale'):
    # Use explicit values (for hybrid_features mode)
    r_min = cfg_encoding.hashgrid.base_resolution
    r_max = cfg_encoding.hashgrid.finest_resolution
    self.growth_rate = cfg_encoding.hashgrid.per_level_scale
else:
    # Calculate from log2 resolutions (default behavior)
    l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
    r_min, r_max = 2 ** l_min, 2 ** l_max
    num_levels = cfg_encoding.levels
    self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
```

### 3. Fixed CUDA `level_offsets` Copying Bug (`cuda_rasterizer/forward.cu` & `backward.cu`)

The CUDA code was using the **encoded** `level` value (e.g., 393217) as a loop bound when copying `level_offsets`, causing array overflow and memory corruption. Fixed by decoding first:

```cuda
int actual_levels = level;
if(render_mode == 6){
    // Extract hashgrid_levels from encoded value
    int total_levels = level >> 16;
    int hybrid_levels = level & 0xFFFF;
    actual_levels = total_levels - hybrid_levels;  // hashgrid levels (e.g., 5)
}
for(int l = 0; l <= actual_levels; l++) collec_offsets[l] = level_offsets[l];
```

## Verification

The resolution calculation now produces exact matches:

```
Original grid resolutions:
  Level 0: 128.00
  Level 1: 168.90
  Level 2: 222.86
  Level 3: 294.07
  Level 4: 388.02
  Level 5: 512.00

Hashgrid for hybrid_levels=1:
  Level 0: 168.90  ← matches original level 1
  Level 1: 222.86  ← matches original level 2
  Level 2: 294.07  ← matches original level 3
  Level 3: 388.02  ← matches original level 4
  Level 4: 512.00  ← matches original level 5
```

## Testing

Run with:
```bash
python train.py -s /path/to/data -m output_name \
  --yaml ./configs/nerfsyn.yaml --eval --iterations 13000 \
  --method hybrid_features --hybrid_levels 1
```

The output should now be high-quality and comparable to or better than the original baseline.

## Files Modified

1. `hash_encoder/modules.py`:
   - Lines 181-205: Fixed hashgrid resolution calculation
   - Lines 275-286: Updated `build_encoding` to use explicit parameters

2. `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`:
   - Lines 566-582: Fixed `level_offsets` copying with proper decoding

3. `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`:
   - Lines 572-588: Fixed `level_offsets` copying with proper decoding

## Rebuild Required

After these changes, rebuild the CUDA extension:
```bash
cd submodules/diff-surfel-rasterization
./rebuild.sh
```




