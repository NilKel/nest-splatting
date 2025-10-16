# Changelog - Surface Potential Branch

## [Unreleased] - 2025-10-16

### Added

#### Final Test Rendering Feature ✅
- **New CLI argument**: `--test_render_stride` (default: 25)
- Automatically renders test views at end of training
- Saves concatenated images: GT | Rendered | Normals (side-by-side)
- Also saves individual components separately
- **Computes and reports PSNR, SSIM, and L1 metrics**
- **Saves per-image and average metrics to `metrics.txt`**
- Location: `train.py` lines 354-462
- Output directory: `<model_path>/final_test_renders/`

**Usage:**
```bash
python train.py ... --test_render_stride 25  # Every 25th test image
python train.py ... --test_render_stride 5   # Every 5th test image
python train.py ... --test_render_stride 1   # All test images
```

**Output files:**
- `*_concat.png` - GT | Rendered | Normal (side-by-side)
- `*_gt.png`, `*_render.png`, `*_normal.png` - Individual components
- `metrics.txt` - PSNR, SSIM, L1 metrics (per-image and average)

#### Method Selection
- **New CLI argument**: `--method {baseline,surface}` (default: baseline)
- Prepares infrastructure for surface potential implementation
- Location: `train.py` lines 535-536
- Currently both modes behave identically (baseline)

**Usage:**
```bash
python train.py ... --method baseline  # Original NeST
python train.py ... --method surface   # Surface potential (to be implemented)
```

### Fixed

#### CLI Iterations Override
- `--iterations` argument now correctly overrides config file value
- Previously, config's `training_cfg.iterations` would always win
- Fix: Check if CLI argument was explicitly set before config merge
- Location: `train.py` lines 490-510 (merge_cfg_to_args function)

#### RTX 5090 Installation Support
- Added `README_RTX5090.md` with SM120 installation instructions
- Added `environment_rtx5090.yml` for conda environment
- Added `install_rtx5090.sh` automated installation script
- Fixed submodule compilation issues:
  - `diff-surfel-rasterization`: Fixed GLM include paths, added `<cstdint>`
  - `simple-knn`: Added missing `<cfloat>` header

#### Dataset Setup
- Added `scripts/generate_initial_pointclouds.py` 
- Generates random initial point clouds for NeRF synthetic scenes
- 100,000 points per scene within ±1.5 bounds
- Required since NeRF synthetic doesn't include initial geometry

### Documentation

- **ANSWERS_TO_QUESTIONS.md**: Architecture analysis Q&A
- **INSTALLATION_SUMMARY.md**: RTX 5090 setup guide
- **SURFACE_POTENTIAL_PLAN.md**: Implementation roadmap
- **TEST_COMMANDS.md**: Testing commands
- **CHANGELOG.md**: This file

### Changed

- Updated `scripts/nerfsyn_eval.py` dataset path
- Remote URL updated to fork: `https://github.com/NilKel/nest-splatting.git`

## Architecture Findings

### Hash Grid Encoder
- **Backend**: Custom CUDA `GridEncoder` (NOT tiny-cuda-nn)
- **Location**: `gridencoder/src/gridencoder.cu`
- **Output**: (N, F) where F = levels × features_per_level

### MLP Decoder
- **Backend**: tiny-cuda-nn's fused MLP ✅
- **Location**: `hash_encoder/modules.py` lines 153-165
- **Input**: Hash features + view direction (Spherical Harmonics)

### Rendering Pipeline
1. Ray-Gaussian intersection → 3D sample point
2. Query hash grid at sample point → features
3. Evaluate Gaussian at sample point → density weight
4. Alpha-blend features using Gaussian weights
5. Pass blended features to MLP → RGB

### Normal Consistency Loss
- **Active**: Yes, default weight 0.05
- **Location**: `train.py` line 195-196
- Ensures clean normals for dot product operation

## Next Steps

### Surface Potential Implementation

1. **Modify Hash Grid Output** → `(N, F, 3)` vector potentials
   - File: `hash_encoder/modules.py`
   - Update feature dimension calculation
   - Reshape output to include 3D vectors

2. **Implement Dot Product in CUDA**
   - File: `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`
   - Location: Lines 700-730 (feature accumulation)
   - Operation: `surface_feature = -Φ · n`
   - Already have: 3D sample points, Gaussian normals, features

3. **Add Method Conditional**
   - File: `gaussian_renderer/__init__.py`
   - Pass `args.method` to render function
   - Branch based on baseline vs. surface

4. **Test and Compare**
   - Visual quality (use final_test_renders!)
   - Metrics (PSNR, SSIM, LPIPS)
   - Number of Gaussians
   - Training speed

## Git Information

- **Branch**: `surface_potential`
- **Remote**: https://github.com/NilKel/nest-splatting.git
- **Commits**: 3 commits (as of 2025-10-16)

## Testing

Quick test command:
```bash
conda activate nest_splatting
python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --eval \
  --iterations 100 \
  --test_render_stride 5
```

Check output: `./output/test/final_test_renders/`

