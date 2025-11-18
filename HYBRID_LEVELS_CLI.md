# Hybrid Levels CLI Argument

## Overview

The `--hybrid_levels N` argument controls the number of finest hashgrid levels used in `hybrid_features` mode.

## Usage

```bash
python train.py -s /path/to/data -m output_dir \
  --yaml ./configs/nerfsyn.yaml \
  --method hybrid_features \
  --hybrid_levels N
```

## How It Works

For `--hybrid_levels N`:
- **Hashgrid**: Uses the N **finest** levels from the original 6-level configuration
- **Per-Gaussian features**: N×D dimensions (stored per Gaussian)
- **Hashgrid features**: N×D dimensions (from finest N levels)
- **Total output**: 2×N×D dimensions (concatenated)

Where D = feature dimension per level (typically 4).

## Examples

### hybrid_levels=1 (Minimal)
```bash
python train.py ... --method hybrid_features --hybrid_levels 1
```
- Per-Gaussian: 1×4 = **4D**
- Hashgrid: 1×4 = **4D** (only finest level 5)
- Total output: **8D**

### hybrid_levels=2
```bash
python train.py ... --method hybrid_features --hybrid_levels 2
```
- Per-Gaussian: 2×4 = **8D**
- Hashgrid: 2×4 = **8D** (levels 4-5)
- Total output: **16D**

### hybrid_levels=3 (Default)
```bash
python train.py ... --method hybrid_features --hybrid_levels 3
# or just
python train.py ... --method hybrid_features  # defaults to 3
```
- Per-Gaussian: 3×4 = **12D**
- Hashgrid: 3×4 = **12D** (levels 3-5)
- Total output: **24D**

### hybrid_levels=4
```bash
python train.py ... --method hybrid_features --hybrid_levels 4
```
- Per-Gaussian: 4×4 = **16D**
- Hashgrid: 4×4 = **16D** (levels 2-5)
- Total output: **32D**

### hybrid_levels=5
```bash
python train.py ... --method hybrid_features --hybrid_levels 5
```
- Per-Gaussian: 5×4 = **20D**
- Hashgrid: 5×4 = **20D** (levels 1-5)
- Total output: **40D**

### hybrid_levels=6 (Maximum)
```bash
python train.py ... --method hybrid_features --hybrid_levels 6
```
- Per-Gaussian: 6×4 = **24D**
- Hashgrid: 6×4 = **24D** (all 6 levels)
- Total output: **48D**

## Hashgrid Level Selection

For a standard 6-level hashgrid configuration (base=128, finest=512):

| hybrid_levels | Hashgrid Levels Used | Resolutions | Per-Gaussian Dim | Hashgrid Dim | Total Dim |
|---------------|----------------------|-------------|------------------|--------------|-----------|
| 1 | Level 5 only | [512] | 4D | 4D | **8D** |
| 2 | Levels 4-5 | [389, 512] | 8D | 8D | **16D** |
| 3 | Levels 3-5 | [295, 389, 512] | 12D | 12D | **24D** |
| 4 | Levels 2-5 | [224, 295, 389, 512] | 16D | 16D | **32D** |
| 5 | Levels 1-5 | [170, 224, 295, 389, 512] | 20D | 20D | **40D** |
| 6 | Levels 0-5 (all) | [128, 170, 224, 295, 389, 512] | 24D | 24D | **48D** |

## Implementation Details

### What Changes with `--hybrid_levels N`:

1. **Hashgrid initialization** (`hash_encoder/modules.py`):
   - Creates N-level hashgrid starting from level (6-N) of original configuration
   - Calculates starting resolution based on original 6-level spacing

2. **Per-Gaussian features** (`scene/gaussian_model.py`):
   - `_gaussian_features` tensor initialized to N×D dimensions
   - Stored per Gaussian, optimized during training

3. **Python renderer** (`gaussian_renderer/__init__.py`):
   - `feat_dim = 2×N×D` for decoder input
   - `levels = 2×N` passed to CUDA for buffer allocation

4. **CUDA kernels** (`cuda_rasterizer/forward.cu`, `backward.cu`):
   - Dynamically computes `actual_hash_levels = level / 2`
   - Concatenates N×D per-Gaussian + N×D hashgrid features
   - Splits gradients correctly in backward pass

5. **Buffer allocation** (`rasterize_points.cu`):
   - Forward: `C = Level * D` (e.g., 6×4=24D for N=3)
   - Backward: `colors_dim = C / 2` (gradient buffer for per-Gaussian features)

## Memory Considerations

Per-Gaussian features add memory overhead:
- **N=1**: +4 bytes/Gaussian
- **N=2**: +8 bytes/Gaussian
- **N=3**: +12 bytes/Gaussian (default)
- **N=4**: +16 bytes/Gaussian
- **N=5**: +20 bytes/Gaussian
- **N=6**: +24 bytes/Gaussian

For 100K Gaussians:
- N=1: ~400 KB
- N=3: ~1.2 MB (default)
- N=6: ~2.4 MB

## When to Use Different Values

- **`--hybrid_levels 1-2`**: Lightweight, less expressiveness
- **`--hybrid_levels 3`**: **Recommended default**, good balance
- **`--hybrid_levels 4-5`**: More expressiveness, higher memory
- **`--hybrid_levels 6`**: Maximum control, uses all levels (effectively doubles baseline features)

## Notes

- The hashgrid always uses the **finest N levels** from the original 6-level configuration
- Per-Gaussian features are **not hashed** - they're stored directly per Gaussian
- Both forward and backward passes correctly handle variable N
- Densification/pruning automatically resizes per-Gaussian features


