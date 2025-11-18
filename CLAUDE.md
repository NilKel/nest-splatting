# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Neural Shell Texture Splatting (NeST)** - ICCV 2025

This is a 3D scene reconstruction and rendering system that decouples geometry and texture in Gaussian splatting. It uses **2D Gaussian splats** for geometry representation and **multi-level instant hash tables** for texture encoding, enabling richer texture rendering with fewer Gaussian points.

The codebase is built on:
- **2D Gaussian Splatting** (2DGS) for geometry
- **instant-ngp** for texture encoding via hash grids
- Custom CUDA kernels for ray-splat intersection and rasterization

## Architecture

### Core Pipeline (Two-Stage Training)

1. **Stage 1: 2DGS Training (iterations 0-10000)**
   - Trains 2D Gaussian splats using standard 3DGS approach
   - Outputs geometry representation (Gaussians) saved to `{dataset}/gaussian_init.pth`
   - Generates alpha masks for subsequent INGP training

2. **Stage 2: INGP Training (iterations 10000-30000)**
   - Freezes or refines Gaussian geometry
   - Trains instant-ngp hash grid for texture features
   - Performs deferred rendering: rasterization → feature map → MLP decoder → RGB

### Rendering Methods (`--method` flag)

The codebase supports multiple rendering approaches:

- **`baseline`**: Standard NeST rendering (ray-splat intersection + hash grid features)
- **`surface`**: Surface potential mode using 3 separate hash grids (for gradient computation)
- **`surface_blend`**: Alpha-blend feature vectors before dot product
- **`surface_rgb`**: Surface potential + diffuse RGB from additional hash grid

**CRITICAL**: The same `--method` must be used for both training (`train.py`) and evaluation (`eval_render.py`).

### Key Components

- **`gaussian_renderer/__init__.py`**: Core rendering loop, orchestrates rasterization and INGP queries
- **`hash_encoder/modules.py`**: INGP implementation with hash grid encoding and MLP decoder
- **`scene/gaussian_model.py`**: 2D Gaussian splat representation and densification logic
- **`submodules/diff-surfel-rasterization/`**: Custom CUDA rasterizer for 2D surfels with ray-splat intersection
- **`train.py`**: Two-stage training orchestration with checkpoint management
- **`eval_render.py`**: Rendering and mesh extraction for trained models

## Development Commands

### Environment Setup

```bash
# Clone with submodules
git clone https://github.com/zhangxin-cg/nest-splatting.git --recursive
cd nest-splatting

# Create conda environment
conda env create -f environment.yml
conda activate nest_splatting

# Install tiny-cuda-nn (required for hash grid encoding)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Training

```bash
# Train with default baseline method
python train.py -s /path/to/scene -m output_name --yaml ./configs/nerfsyn.yaml

# Train with surface potential method
python train.py -s /path/to/scene -m output_name --yaml ./configs/nerfsyn.yaml --method surface

# Training with custom iterations (must be > 10000 for INGP to activate)
python train.py -s /path/to/scene -m output_name --yaml ./configs/nerfsyn.yaml --iterations 30000

# Resume from checkpoint
python train.py -s /path/to/scene -m output_name --yaml ./configs/nerfsyn.yaml --start_checkpoint /path/to/chkpnt.pth
```

**Output structure**: `outputs/{method}/{dataset}/{scene}/{run_name}/`

### Evaluation & Rendering

```bash
# Render test images and extract mesh
python eval_render.py -m /path/to/model --yaml ./configs/nerfsyn.yaml --method baseline --iteration 30000

# Skip mesh extraction (faster)
python eval_render.py -m /path/to/model --yaml ./configs/nerfsyn.yaml --method surface --skip_mesh

# Unbounded mesh extraction (for outdoor scenes)
python eval_render.py -m /path/to/model --yaml ./configs/nerfsyn.yaml --unbounded --mesh_res 1024
```

### Batch Evaluation Scripts

```bash
# NeRF Synthetic dataset
python scripts/nerfsyn_eval.py --yaml ./configs/nerfsyn.yaml

# MipNeRF360 dataset
python scripts/360_eval.py

# DTU dataset
python scripts/dtu_eval.py --yaml ./configs/dtu.yaml
```

**Remember**: Update `dataset_dir` in scripts before running.

### Building CUDA Extensions

When modifying CUDA code in `submodules/` or `gridencoder/`:

```bash
# Rebuild diff-surfel-rasterization
pip install ./submodules/diff-surfel-rasterization

# Rebuild simple-knn
pip install ./submodules/simple-knn

# Rebuild gridencoder
pip install ./gridencoder
```

## Configuration System

### YAML Configs (`configs/*.yaml`)

Configurations use a hierarchical structure with sections:

- **`training_cfg`**: Iteration counts, densification parameters, learning rates
- **`settings`**: Flags like `if_ingp`, `white_background`, `gs_alpha`
- **`loss`**: Loss weights and activation iterations for regularization
- **`ingp_stage`**: Phase transition points (e.g., `initialize: 10000`)
- **`encoding`**: Hash grid parameters (levels, resolution, dictionary size)
- **`rgb`**: MLP architecture for RGB decoder
- **`surfel`**: Beta and opacity parameters for 2D surfel rendering

**Key parameters**:
- `ingp_stage.initialize`: Iteration when 2DGS training ends and INGP begins (default: 10000)
- `training_cfg.iterations`: Total training iterations (default: 30000)
- `encoding.levels`: Number of hash grid levels (default: 6 for baseline, 3x6=18 for surface modes)

### CLI Override

Command-line args override YAML values:
```bash
python train.py -s /path/to/scene --iterations 50000  # Overrides YAML iterations
```

## Checkpoint Management

### 2DGS Checkpoint (`gaussian_init.pth`)

Saved at iteration 10000 to `{dataset_path}/gaussian_init.pth`. Contains:
- Gaussian parameters: xyz, features, scaling, rotation, opacity
- Spatial learning rate scale
- Active SH degree

**Checkpoint loading behavior**:
- If `gaussian_init.pth` exists: Skip 2DGS training (iterations 0-10000), load Gaussians, start INGP training at iteration 10000
- If missing: Train 2DGS from scratch, save checkpoint at iteration 10000, continue with INGP

**To retrain from scratch**: Delete `{dataset_path}/gaussian_init.pth`

### Full Training Checkpoints

Saved via `--checkpoint_iterations` or `--save_iterations`:
- Format: `chkpnt{iteration}.pth` or `point_cloud/iteration_{iteration}/`
- Contains full training state (Gaussians + optimizer + INGP model)

## Working with Rendering Methods

### Adding a New Rendering Method

1. **Add method to choices** in `train.py` and `eval_render.py`:
   ```python
   parser.add_argument("--method", choices=["baseline", "surface", "your_method"])
   ```

2. **Implement rendering logic** in `hash_encoder/modules.py`:
   - Add conditional logic in `INGP.__init__()` to initialize required hash grids
   - Modify `INGP.forward()` to handle new rendering mode

3. **Update CUDA rasterizer** if needed:
   - Modify `submodules/diff-surfel-rasterization/` for kernel changes
   - Rebuild: `pip install ./submodules/diff-surfel-rasterization`

4. **Update config**: Add method-specific parameters to YAML configs if needed

### Surface Potential Mode Details

The `surface` method uses **3 separate hash grids** (18 levels total) to enable proper gradient computation for surface potential fields. This is different from the baseline single hash grid (6 levels).

Key differences in `hash_encoder/modules.py`:
- `method="surface"`: Creates 3 `feat_encoders` for x, y, z components
- `method="baseline"`: Creates 1 `feat_encoder` for RGB features

## Common Workflows

### Comparing Baseline vs Surface Methods

Use `train_surface_comparison.sh`:
```bash
./train_surface_comparison.sh /path/to/scene 15000
```

This script:
1. Trains 2DGS once (saves to `gaussian_init.pth`)
2. Trains baseline method with INGP
3. Trains surface method with INGP (reuses same Gaussians)
4. Compares PSNR/SSIM metrics

### Custom Dataset Preparation

1. Process with COLMAP (same as 3DGS):
   ```bash
   # Use COLMAP to generate sparse reconstruction
   # Expected structure: images/ + sparse/0/ (cameras, images, points3D)
   ```

2. Create YAML config:
   - Set `range` for scene bounding box
   - Set `contract: True` for outdoor/unbounded scenes
   - Adjust `resolution` for downsampling

3. Train:
   ```bash
   python train.py -s /path/to/colmap_output -m experiment_name --yaml ./configs/custom.yaml
   ```

## CUDA Development Notes

### Rasterization Pipeline

The custom rasterizer in `submodules/diff-surfel-rasterization/` performs:
1. **Forward pass** (`forward.cu`): Ray-splat intersection, writes UV coords and features to tiles
2. **Backward pass** (`backward.cu`): Gradient computation for Gaussians and hash grid features

Key files:
- `rasterize_points.cu/h`: Main entry points
- `cuda_rasterizer/forward.cu`: Intersection and tile-based rasterization
- `cuda_rasterizer/backward.cu`: Gradient propagation

### Hash Grid Integration

Hash grid queries happen in CUDA during rasterization:
- Intersection points (UV coords) computed in `forward.cu`
- Features queried from hash grid using `gridencoder` library
- Feature map assembled and returned to Python
- MLP decoder processes feature map → RGB image

### Debugging CUDA Issues

```bash
# Check for NaN/Inf in tensors
python train.py ... --detect_anomaly

# Rebuild with debug symbols
cd submodules/diff-surfel-rasterization
TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --force-reinstall

# View CUDA memory usage
# (automatic during training via torch.cuda.max_memory_allocated())
```

## Testing & Metrics

### Final Test Rendering

At the end of training, `train.py` automatically renders test images with stride:
```bash
# Default: render every 25th test image
python train.py ... --test_render_stride 25

# Render all test images
python train.py ... --test_render_stride 1
```

Outputs:
- `final_test_renders/test_{idx}_{name}_concat.png`: GT | Rendered | Normal
- `final_test_renders/metrics.txt`: Per-image and average PSNR/SSIM/L1

### Evaluation Metrics

Computed in `scripts/metric_eval.py` and saved to `{output_dir}/results.json`:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

## Git Workflow

Current branch: `surface_potential`
Main branch: `main`

Modified files focus on surface potential rendering:
- `hash_encoder/modules.py`: 3-grid implementation
- `train.py`: Checkpoint loading and method routing
- `gaussian_renderer/__init__.py`: Surface rendering integration
- `submodules/diff-surfel-rasterization/`: CUDA kernel updates

When creating PRs:
```bash
# Create PR to main branch
gh pr create --base main --title "Add surface potential rendering"
```

## Environment & Hardware

Tested configuration:
- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA RTX 4090 (also supports RTX 5090 - see `install_rtx5090.sh`)
- **CUDA**: 11.8
- **Python**: 3.10
- **PyTorch**: 2.1.0

## Important Notes

- **Method consistency**: Always use the same `--method` for training and evaluation
- **Iteration requirements**: Training must run > 10000 iterations for INGP to activate
- **Checkpoint compatibility**: Checkpoints are method-agnostic for 2DGS phase, but INGP checkpoints are method-specific
- **Memory management**: CUDA cache is cleared periodically; expect ~10-12GB VRAM usage for typical scenes
- **YAML precedence**: CLI args override YAML configs (except when config explicitly manages args)
