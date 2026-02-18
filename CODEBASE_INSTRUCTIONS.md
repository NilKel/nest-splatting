# Nest Splatting Codebase Instructions

## Overview

**Nest Splatting** (ICCV 2025) decouples geometry and texture representation in Gaussian splatting:
- **Geometry**: 2D Gaussian splats (efficient primitives)
- **Texture**: Multi-level instant hash table (rich detail encoding)

The core idea: Fewer Gaussians with richer texture representation enables more efficient rendering while preserving visual quality.

## Build Instructions

### Prerequisites
- CUDA toolkit (compatible with your PyTorch version)
- conda environment with PyTorch installed

### Building CUDA Extensions

Always use `python -m pip` within your conda environment to avoid system Python conflicts:

```bash
conda activate nest_splatting
```

#### 1. Grid Encoder
```bash
cd /home/nilkel/Projects/nest-splatting/gridencoder
python -m pip install -e . --no-build-isolation
```

#### 2. Diff Surfel Rasterization
```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
python -m pip install -e . --no-build-isolation
```

#### 3. Simple KNN (if needed)
```bash
cd /home/nilkel/Projects/nest-splatting/submodules/simple-knn
python -m pip install -e . --no-build-isolation
```

### Rebuilding After Code Changes

If you modify CUDA files (`.cu`, `.cuh`, `.h`), you must rebuild:
```bash
cd /home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization
python -m pip install -e . --no-build-isolation
```

Python files (`.py`) don't require rebuilding since they're installed in editable mode (`-e`).

### Troubleshooting

**"externally-managed-environment" error:**
Use `python -m pip` instead of `pip` directly, or use `$CONDA_PREFIX/bin/pip`.

**"No module named 'torch'" during build:**
Add `--no-build-isolation` flag to use the current environment's PyTorch.

**JIT compilation taking too long:**
The grid encoder uses JIT compilation by default. Installing with pip pre-compiles it.

---

## Project Structure

```
nest-splatting/
├── train.py                          # Main training script
├── gaussian_renderer/
│   └── __init__.py                  # Rendering pipeline (render function)
├── scene/
│   ├── gaussian_model.py            # Gaussian representation (GaussianModel class)
│   ├── colmap_loader.py             # Scene loading
│   └── cameras.py                   # Camera utilities
├── hash_encoder/
│   ├── modules.py                   # INGP network (hash encoding + MLPs)
│   ├── config.py                    # Configuration
│   └── __init__.py
├── gridencoder/                      # Grid encoding implementation
├── submodules/
│   └── diff-surfel-rasterization/   # Custom CUDA rasterizer
│       ├── cuda_rasterizer/
│       │   ├── forward.cu           # Forward rendering kernels
│       │   ├── backward.cu          # Backward gradient kernels
│       │   ├── hashgrid.h           # Hash grid query functions
│       │   ├── rasterizer.h/.cu     # Rasterizer API
│       │   └── auxiliary.h          # Helper functions
│       ├── rasterize_points.cu      # Python/CUDA bindings
│       └── diff_surfel_rasterization/
│           └── __init__.py          # Python interface
├── arguments/
│   └── __init__.py                  # CLI argument definitions
└── utils/
    ├── nerf_utils.py                # MLP utilities
    ├── sh_utils.py                  # Spherical harmonics
    ├── point_utils.py               # Point operations
    └── general_utils.py             # Common utilities
```

---

## Three Rendering Modes

### 1. Baseline Mode (`--method baseline`)

**Purpose**: Query hashgrid directly in CUDA rasterizer.

**Pipeline**:
1. CUDA rasterizer computes ray-disk intersections
2. At each intersection, queries spatial hashgrid at 3D position
3. Alpha-blends features per pixel → `[24, H, W]` feature map
4. Python MLP decodes features → RGB

**Key characteristics**:
- `hash_in_CUDA = True`
- `render_mode = 0`
- All levels in hashgrid (6 levels × 4 dim = 24D)
- Coarse-to-fine (C2F) scheduling enabled

**Feature flow**:
```
3D intersection → hash query (CUDA) → blend features → MLP decode (Python) → RGB
```

---

### 2. Cat Mode (`--method cat`)

**Purpose**: Concatenate per-Gaussian features with hashgrid features.

**Architecture**:
```
Per-Gaussian Features: hybrid_levels × level_dim    [coarse representation]
        +
Hashgrid Features:   (total_levels - hybrid_levels) × level_dim  [fine details]
        =
Output Features:     total_levels × level_dim      [combined]
```

**Example** (with `--hybrid_levels 5`):
- Per-Gaussian: 5 levels × 4 dim = 20D (stored per Gaussian)
- Hashgrid: 1 level × 4 dim = 4D (finest level only)
- Total output: 24D

**Pipeline**:
1. CUDA rasterizer computes ray-disk intersections
2. At each intersection:
   - Gathers per-Gaussian features by ID
   - Queries hashgrid at 3D position (finest levels only)
   - Concatenates: `[per_gaussian | hash]`
3. Alpha-blends concatenated features per pixel
4. Python MLP decodes → RGB

**Key characteristics**:
- `hash_in_CUDA = True`
- `render_mode = 1`
- `level` encoded as: `(total << 16) | (active_hashgrid << 8) | hybrid_levels`
- Per-Gaussian features always active, hashgrid progressively added (C2F)
- Gradient flow: All geometry gradients from CUDA rasterizer backward

**Feature flow**:
```
3D intersection → [gather per-gaussian | hash query] (CUDA) → blend features → MLP decode (Python) → RGB
```

---

### 3. 3D_direct Mode (`--method 3D_direct`)

**Purpose**: Ray-Gaussian intersection buffer with direct RGB output via MLP.

**Architecture**:
- Same feature split as cat mode (hybrid_levels for coarse, hashgrid for fine)
- But MLP takes view direction encoding and outputs RGB directly (no SH)

**Pipeline**:
1. CUDA rasterizer computes ray-disk intersections
2. Outputs **intersection buffer** instead of blended features:
   ```
   [gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G]
   ```
3. Python post-processing (batched):
   - Recomputes 3D position from `(s_x, s_y)` + Gaussian parameters
   - Queries hashgrid at 3D position (in Python)
   - Gathers per-Gaussian features
   - Concatenates: `[per_gaussian | hash | view_encoding]`
   - MLP → direct RGB
4. Weighted accumulation in Python

**Key characteristics**:
- `hash_in_CUDA = False` (hash query in Python)
- `render_mode = 3`
- Uses `IntersectionOpacityGrad` custom autograd for geometry gradients
- `backward_from_weight_grad` CUDA kernel for gradient computation
- Scale/rotation detached in xyz recomputation (single gradient path)

**Feature flow**:
```
3D intersection → intersection buffer (CUDA) → xyz recompute (Python) → hash query (Python) → MLP (Python) → RGB
```

---

## Key Configuration Parameters

### `--hybrid_levels N`

Controls the split between per-Gaussian and hashgrid features:
- `N=0`: All features in hashgrid (like baseline)
- `N=5`: 5 coarse levels per-Gaussian, 1 fine level in hashgrid
- `N=6`: All per-Gaussian, no hashgrid

### `--disable_c2f`

Disables coarse-to-fine scheduling. When enabled (or for non-baseline methods), all levels are active from the start.

### `--method {baseline, cat, 3D_direct, ...}`

Selects the rendering mode.

---

## Comparison Table

| Aspect | Baseline | Cat | 3D_direct |
|--------|----------|-----|-----------|
| **Hash Location** | CUDA (3D query) | CUDA (3D query) | Python (3D query) |
| **Per-Gaussian Features** | None | `hybrid_levels × dim` | `hybrid_levels × dim` |
| **render_mode** | 0 | 1 | 3 |
| **Feature Blending** | CUDA | CUDA | Python |
| **View Direction** | MLP input | MLP input | MLP input |
| **MLP Output** | RGB | RGB | RGB |
| **Gradient Path** | CUDA backward | CUDA backward | Custom autograd |
| **Coarse-to-Fine** | All hashgrid | Per-Gaussian always + Hash progressive | All active |

---

## Key Files for Understanding

### Python

1. **`gaussian_renderer/__init__.py`**: Main `render()` function
   - Lines 367-741: Mode detection and setup
   - Lines 844-1114: 3D/3D_direct post-processing

2. **`hash_encoder/modules.py`**: INGP class
   - Lines 47-211: Initialization and MLP setup
   - Lines 748-835: `set_active_levels()` for C2F scheduling

3. **`scene/gaussian_model.py`**: GaussianModel class
   - Lines 353-459: `create_from_pcd()` initialization
   - Lines 805-883: `densification_postfix()` for cloning

### CUDA

1. **`forward.cu`**: Forward rendering
   - Lines 1286-1588: `switch(render_mode)` for different modes
   - Case 1 (lines 1305-1421): Cat mode
   - Case 3 (lines 1531-1588): 3D mode intersection buffer

2. **`backward.cu`**: Backward gradients
   - Lines 1862-2015: `backward_from_weight_grad` for 3D mode

---

## Gradient Flow Differences

### Cat Mode
```
dL/dRGB
    ↓ (Python MLP backward)
dL/dfeatures [24, H, W]
    ↓ (CUDA rasterizer backward)
├── dL/d(per_gaussian_features)
├── dL/d(hash_features) → dL/d(hash_table)
├── dL/d(opacity)
├── dL/d(scale)
├── dL/d(rotation)
└── dL/d(position) via screenspace_points
```

### 3D_direct Mode (Detailed)

#### Forward Pass

**Step 1: CUDA Rasterizer (render_mode=3)**
```
For each tile/pixel (front-to-back sorted Gaussians):
  For each Gaussian:
    1. Compute ray-disk intersection → (s_x, s_y) disk coordinates
    2. Compute rho3d = s_x² + s_y² (disk distance)
    3. Compute rho2d = FilterInvSquare * (d.x² + d.y²) (screen distance)
    4. If rho3d <= rho2d: use disk intersection (rho_flag=1)
       Else: use Gaussian center (rho_flag=0)
    5. Compute G = kernel(rho), alpha = opacity * G, weight = alpha * T
    6. Update T = T * (1 - alpha) for next Gaussian
    7. Write to intersection buffer: [gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G]
  Store transMat in geomBuffer for backward
```

**Step 2: Python Post-processing**
```python
# IntersectionOpacityGrad.forward():
xyz = s_x * scale_x * R[:,0] + s_y * scale_y * R[:,1] + mean  # when rho_flag=1
xyz = mean  # when rho_flag=0
# Returns (weights_with_grad, xyz_with_grad, ...)

# For each batch:
b_hash = ingp._encode_3D(xyz)                    # Hash encode positions
b_gauss = gaussian_features[gaussian_ids]        # Gather per-Gaussian features
b_viewdir = normalize(xyz - camera_center)       # View direction
b_viewdir_enc = ingp._encode_view(b_viewdir)     # Encode view direction
b_combined = [b_gauss | b_hash | b_viewdir_enc]  # Concatenate
b_rgb = sigmoid(mlp_3D_direct(b_combined))       # MLP → RGB
blended_rgb += b_rgb * weights                   # Weighted accumulation

# Final compositing:
rgb = blended_rgb + (1 - total_weight) * bg_color
```

#### Backward Pass

**Step 1: PyTorch autograd (MLP/hash chain)**
```
Loss → dL/d(rgb) → scatter backward → dL/d(b_weighted_rgb)
  → dL/d(b_rgb) and dL/d(weights)
  → MLP backward → dL/d(b_combined)
  → Split to: dL/d(b_gauss), dL/d(b_hash), dL/d(b_viewdir_enc)
  → Hash backward → dL/d(xyz) = grad_xyz
```

**Step 2: IntersectionOpacityGrad.backward()**
```
Receives: grad_weight (from weighted accumulation) and grad_xyz (from hash/MLP)

A) Compute dL_duv from grad_xyz (INDIRECT path input):
   dL_duv_x = (grad_xyz · (scale_x * R[:,0])) * rho_flag
   dL_duv_y = (grad_xyz · (scale_y * R[:,1])) * rho_flag

B) CUDA kernel backward_from_weight_grad:
   - Iterates back-to-front per pixel
   - Transmittance chain: dL_dalpha = (dL_dw - last_dL_dT) * T
   - dL_dopacity = G * dL_dalpha
   - INDIRECT: dL_ds = dL_dG * (-G) * s + dL_duv  ← includes hash gradient!
   - Ray-disk Jacobian: dL_ds → dL_dtransMat

C) Python converts dL_dtransMat → dL_dscale, dL_drotation (INDIRECT path):
   dL_dscale_x = dL_dTu · R[:,0]
   dL_dscale_y = dL_dTv · R[:,1]
   dL_dR[:,0] = scale_x * dL_dTu
   dL_dR[:,1] = scale_y * dL_dTv

D) Python DIRECT gradients from grad_xyz:
   dL/d(scale_x) += (grad_xyz · R[:,0]) * s_x * rho_flag  (scatter_add)
   dL/d(scale_y) += (grad_xyz · R[:,1]) * s_y * rho_flag
   dL/d(R[:,0]) += grad_xyz * s_x * scale_x * rho_flag
   dL/d(R[:,1]) += grad_xyz * s_y * scale_y * rho_flag

E) Means gradient:
   dL/d(means3D) = scatter_add(grad_xyz)
```

#### Two Gradient Paths for Geometry

**INDIRECT path** (through ray-disk Jacobian):
```
dL/dxyz → dL/ds (via d(xyz)/d(s)) → ray-disk Jacobian → dL/dtransMat → dL/d(scale,rotation)
```
This captures: "when transMat changes, s changes, which changes xyz"

**DIRECT path** (chain rule):
```
dL/dxyz → dL/d(scale,rotation) directly
```
This captures: "when transMat changes, xyz changes directly (with s held constant)"

Both paths are needed because `xyz = s_x * Tu + s_y * Tv + mean` depends on transMat both directly (Tu, Tv) and indirectly (s depends on transMat via ray-disk intersection).

#### Comparison with Cat Mode

| Aspect | Cat Mode | 3D_direct Mode |
|--------|----------|----------------|
| Feature blending | CUDA | Python |
| Hash query | CUDA | Python |
| Geometry gradients | All in CUDA backward | Split: CUDA kernel + Python |
| dL_duv handling | CUDA backward directly | Python computes, passes to CUDA |
| DIRECT gradient | In CUDA backward | In Python backward |

---

## Common Training Commands

```bash
# Baseline mode
python train.py -s <scene_path> --method baseline

# Cat mode with 5 hybrid levels
python train.py -s <scene_path> --method cat --hybrid_levels 5

# 3D_direct mode
python train.py -s <scene_path> --method 3D_direct --hybrid_levels 5

# With specific iterations
python train.py -s <scene_path> --method cat --hybrid_levels 5 --iterations 30000
```

---

## Debugging Tips

1. **Feature dimension mismatch**: Check `shape_dims` tensor `[GS, HS, OS]`
2. **C2F not working**: Check `set_active_levels()` and `active_hashgrid_levels`
3. **Gradient issues in 3D_direct**: Check `IntersectionOpacityGrad.backward()`
4. **Memory issues**: Reduce `max_intersections_per_pixel` (default 32)
5. **Slow training**: Check batch size in 3D mode processing (default 5M intersections)

---

## Decomposed Rendering (Debugging)

For cat and 3D_direct modes, use `decompose_mode` parameter to isolate feature contributions:

```python
# Gaussian features only (hash zeroed)
render(..., decompose_mode='gaussian_only')

# Hash features only (Gaussian zeroed)
render(..., decompose_mode='ngp_only')
```

**Training output**: At each `save_interval`, three images are saved:
- `{iteration}.png` - Full render
- `{iteration}_gaussian.png` - Gaussian features only
- `{iteration}_hash.png` - Hash features only

This helps diagnose whether divergence is in Gaussian features or hash features.

---

## Intersection Buffer Format (3D_direct mode)

The CUDA rasterizer outputs a buffer with **12 floats** per intersection:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `gaussian_id` | Which Gaussian (as int, reinterpreted) |
| 1 | `weight` | Blending weight = alpha * T |
| 2 | `pixel_id` | Which pixel (as int, reinterpreted) |
| 3 | `xyz.x` | World-space intersection point x (for hash query) |
| 4 | `xyz.y` | World-space intersection point y |
| 5 | `xyz.z` | World-space intersection point z |
| 6 | `s_x` | Disk coordinate x (for backward gradient computation) |
| 7 | `s_y` | Disk coordinate y (for backward gradient computation) |
| 8 | `rho_flag` | 1.0 = disk intersection, 0.0 = center |
| 9 | `alpha` | opacity * G (for gradient computation) |
| 10 | `T` | Transmittance BEFORE this intersection |
| 11 | `G` | Kernel value (for dL/dopacity = G * dL/dalpha) |

The buffer uses padded layout: each pixel gets `max_intersections_per_pixel` slots.

**Key**: The xyz coordinates are precomputed by CUDA using the same formula as CAT mode:
```cuda
if (rho3d <= rho2d) {  // disk intersection
    xyz = pk + s.x * SuTu + s.y * SvTv;  // SuTu = R[:, 0] * scale_x
} else {  // center fallback
    xyz = pk;  // Gaussian center
}
```

This ensures **exact forward feature matching** between CAT mode and 3D_direct mode for hash queries.

---

## Gradient Matching Test Script

**IMPORTANT: Run this script to verify gradient correctness after any changes to the backward pass!**

### Quick Run Command

```bash
conda run -n nest_splatting python /tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_intersection_opacity_grad.py 2>&1 | tail -100
```

### Full Path
`/tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_intersection_opacity_grad.py`

### Purpose
Verifies gradient matching between CAT mode and 3D_direct mode for hash-enabled rendering.

### Current Status

> **See [MEMORY.md](~/.claude/projects/-home-nilkel-Projects-nest-splatting/memory/MEMORY.md) for current focus and known issues.**

All RGB/feature path gradients match perfectly (cos_sim=1.0). Tests use `loss = feat.sum()`.

### Key Insight: Two Gradient Paths

The xyz/scale/rot gradients have two sources:
1. **Hash path** (autograd): loss → hash_feat → xyz → center/scale/rot
2. **Weight path** (CUDA): loss → weight → alpha → G → transMat → center/scale/rot

In 3D_direct mode:
- Hash path: Computed via Python autograd, passed via `dL_dhomoMat`
- Weight path: Computed via CUDA `backward_from_weight_grad` + `transmat_to_scale_rot_grad`

### Test Flow

```python
# CAT MODE
result_cat = rasterizer(render_mode=1)  # CUDA computes everything
loss_cat.backward()  # CUDA backward

# 3D_direct MODE
result_3d = rasterizer(render_mode=3)  # Get intersection buffer

# Compute xyz from Gaussian parameters (for autograd chain)
b_xyz = rho_flag * xyz_disk + (1-rho_flag) * b_center

# Hash encoding (autograd connected)
b_hash = ingp._encode_3D(b_xyz)
loss_3d.backward()  # b_xyz.grad has hash gradient

# Compute dL_dhomoMat from b_xyz.grad
dL_dhomoMat[:, 6:9] = scatter_add(grad_xyz)  # mean gradient
dL_dhomoMat[:, 0:3] = scatter_add(grad_xyz * s_x)  # col0 gradient
dL_dhomoMat[:, 3:6] = scatter_add(grad_xyz * s_y)  # col1 gradient

# CUDA backward kernels
grad_opacity, dL_dtransMat, dL_dmean2D = backward_from_weight_grad(...)
grad_scale, grad_rot, grad_means3D = transmat_to_scale_rot_grad(
    dL_dtransMat, dL_dhomoMat, dL_dmean2D, ...
)
```

### Running the Tests

```bash
# Test 1: CUDA kernels directly (verifies backward_from_weight_grad + transmat_to_scale_rot_grad)
conda run -n nest_splatting python /tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_intersection_opacity_grad.py 2>&1 | tail -100

# Test 2: IntersectionOpacityGrad via autograd (verifies the actual renderer code path)
conda run -n nest_splatting python /tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_intersection_opacity_grad_autograd.py 2>&1 | tail -50

# Test 3: Transmittance chain with multiple overlapping Gaussians (verifies alpha blending)
conda run -n nest_splatting python /tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_transmittance_chain.py 2>&1 | tail -60

# Test 4: Hash indexing (verifies trilinear interpolation and hash table gradient distribution)
conda run -n nest_splatting python /tmp/claude-1000/-home-nilkel-Projects-nest-splatting/760f7453-9090-4e31-afd1-81bb309cb1ec/scratchpad/test_hash_indexing.py 2>&1 | tail -60
```

### Test 4: Hash Indexing Details

Verifies that hash table gradients match between CAT and 3D_direct modes:
- **Hash indices**: Correct mapping from 3D positions to hash table indices
- **Trilinear interpolation**: Gradients distributed to 8 corner vertices correctly
- **Gradient aggregation**: Multiple lookups at same index aggregate correctly

Expected output:
```
Hash gradient cos_sim: 1.000000 - PERFECT
Forward feature match: 1.000000
HASH INDEXING: CORRECT
```
