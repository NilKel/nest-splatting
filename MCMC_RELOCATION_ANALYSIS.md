# MCMC Relocation Analysis: Normals and Orientation

## Question
When using MCMC relocation with the training command:
```bash
python train.py -s /path/to/ficus -m no2fmcmc1e3ns001sc001oprune \
    --yaml ./configs/nerfsyn.yaml --iterations 30000 --method cat --eval \
    --hybrid_levels 5 --disable_c2f --mcmc --cap_max 100000 \
    --opacity_reg 0.001 --scale_reg 0.001 --noise_lr 1e3
```

**Do relocated Gaussians maintain similar normals or orientation to their source Gaussians?**

## Answer: **YES, they maintain IDENTICAL orientation/rotation**

## How MCMC Relocation Works

### 1. Relocation Process (`relocate_gs`)

When a Gaussian is "dead" (opacity ≤ 0.005), it gets relocated:

```python
# From scene/gaussian_model.py, lines 836-902

def relocate_gs(self, dead_mask):
    # 1. Find dead and alive Gaussians
    dead_indices = dead_mask.nonzero()
    alive_indices = (~dead_mask).nonzero()
    
    # 2. Sample alive Gaussians based on opacity (probability weighted)
    probs = self.get_opacity[alive_indices]
    reinit_idx = sample_from_alives(probs, num=len(dead_indices))
    
    # 3. Get parameters from sampled Gaussians
    new_xyz = self._xyz[reinit_idx]
    new_rotation = self._rotation[reinit_idx]  # ← COPIED DIRECTLY
    new_features_dc = self._features_dc[reinit_idx]
    new_features_rest = self._features_rest[reinit_idx]
    # ... etc
    
    # 4. Update opacity and scale using MCMC kernel
    new_opacity, new_scaling = compute_relocation_cuda(
        opacity_old=self.get_opacity[reinit_idx],
        scale_old=self.get_scaling[reinit_idx],
        N=ratio  # How many children this Gaussian produces
    )
    
    # 5. Replace dead Gaussians with copied parameters
    self._xyz.data[dead_indices] = new_xyz
    self._rotation.data[dead_indices] = new_rotation  # ← EXACT COPY
    self._opacity.data[dead_indices] = new_opacity    # ← ADJUSTED
    self._scaling.data[dead_indices] = new_scaling    # ← ADJUSTED
```

### 2. What Gets Copied vs. Adjusted

| Parameter | Treatment | Reason |
|-----------|-----------|--------|
| **Position (xyz)** | **Copied exactly** | Dead Gaussian moves to same location as source |
| **Rotation (quaternion)** | **Copied exactly** | Maintains same orientation/normal |
| **Color (SH coefficients)** | **Copied exactly** | Maintains same appearance |
| **Per-Gaussian features** | **Copied exactly** | Maintains same learned features |
| **Opacity** | **Adjusted by MCMC kernel** | Split to preserve total contribution |
| **Scale (2D)** | **Adjusted by MCMC kernel** | Split to preserve total contribution |

### 3. The MCMC Relocation Kernel

The relocation kernel (from `utils/reloc_utils.py`) implements Equation (9) from the paper:

```python
def compute_relocation_cuda(opacity_old, scale_old, N):
    """
    When a Gaussian is split into N children:
    - New opacity: α_new = 1 - (1 - α_old)^(1/N)
    - New scale: s_new = s_old / sqrt(N)
    
    This preserves the total "mass" contribution to the rendered image.
    """
```

**Key insight**: Only opacity and scale are adjusted. Rotation is NOT touched.

## Implications for Normals/Orientation

### ✅ What This Means

1. **Relocated Gaussians have IDENTICAL rotation/orientation** to their source
   - The rotation quaternion is copied exactly
   - The normal direction (derived from rotation) is identical

2. **Multiple dead Gaussians can be relocated to the same source**
   - They will all have the same position, rotation, and features
   - Only their opacity and scale differ (adjusted by MCMC kernel)
   - This creates "clones" at the same location with same orientation

3. **SGLD Noise will eventually move them apart**
   - After relocation, SGLD noise (controlled by `--noise_lr 1e3`) is injected
   - This noise is scaled by the covariance matrix (rotation + scale)
   - Gaussians will drift apart along their principal axes

### Example Scenario

```
Before relocation:
  Gaussian A (alive): pos=[1,2,3], rotation=[0.7,0,0,0.7], opacity=0.8, scale=[0.1,0.05]
  Gaussian B (dead):  pos=[5,6,7], rotation=[0.5,0.5,0.5,0.5], opacity=0.003, scale=[0.2,0.1]

After relocating B to A:
  Gaussian A (source): pos=[1,2,3], rotation=[0.7,0,0,0.7], opacity=0.6, scale=[0.07,0.035]
  Gaussian B (relocated): pos=[1,2,3], rotation=[0.7,0,0,0.7], opacity=0.6, scale=[0.07,0.035]
                          ↑ SAME POSITION AND ROTATION AS A!

After SGLD noise injection (next iteration):
  Gaussian A: pos=[1.01,2.02,2.99], rotation=[0.7,0,0,0.7], ...
  Gaussian B: pos=[0.98,1.97,3.01], rotation=[0.7,0,0,0.7], ...
              ↑ Drifted apart due to noise, but rotation still similar
```

## Why This Design Makes Sense

### 1. Preserves Local Surface Structure
- Alive Gaussians have learned good orientations for the surface
- Relocated Gaussians inherit this knowledge
- Maintains surface normal consistency

### 2. Efficient Exploration
- Starting with good orientation reduces search space
- SGLD noise can refine position while keeping orientation
- Faster convergence than random initialization

### 3. MCMC Theory
- The relocation kernel preserves the "birth-death" balance
- Only opacity and scale need adjustment to maintain equilibrium
- Rotation is part of the "state" that gets inherited

## Impact of Your Hyperparameters

```bash
--noise_lr 1e3  # SGLD noise learning rate
--opacity_reg 0.001  # L1 regularization on opacity
--scale_reg 0.001  # L1 regularization on scale
```

### `noise_lr 1e3` (Low noise)
- **Effect**: Relocated Gaussians stay close to source position
- Lower noise → slower exploration → more similar orientations persist longer
- Higher noise (e.g., 1e5) → faster exploration → orientations can diverge more quickly

### `opacity_reg 0.001` and `scale_reg 0.001` (Low regularization)
- **Effect**: Encourages sparse representation (low opacity, small scale)
- More Gaussians become "dead" → more relocation events
- More relocation → more Gaussians with similar orientations (clones)

## Visualization of Relocation

```
Time t=0: Initial Gaussians
  ●  ●  ●  ●  ●  (various orientations)

Time t=5000: Some Gaussians die
  ●  ●  ✗  ●  ✗  (✗ = dead, opacity ≤ 0.005)

Time t=5001: Relocation happens
  ●  ●  ●  ●  ●  (dead ones moved to alive positions)
     ↑     ↑     (these two now have SAME position & rotation)

Time t=5002: SGLD noise applied
  ●  ● ● ●  ●  (clones drift apart slightly)
     ↑ ↑ ↑     (still similar orientations)

Time t=10000: After many iterations
  ●  ● ● ●  ●  (positions diverged, orientations may still be similar)
```

## Checking Orientation Similarity in Your Model

You can verify this by analyzing your trained model:

```python
import torch
from scene import Scene, GaussianModel

# Load model
gaussians = GaussianModel(sh_degree=3)
scene = Scene(dataset, gaussians, load_iteration=30000)

# Get rotations (quaternions)
rotations = gaussians.get_rotation  # (N, 4)

# Compute pairwise quaternion distances (dot product)
# Similar orientations have dot product close to ±1
dots = rotations @ rotations.T  # (N, N)
similar_pairs = (dots.abs() > 0.99).sum() - len(rotations)  # Exclude diagonal

print(f"Number of Gaussian pairs with similar orientation: {similar_pairs // 2}")
print(f"Percentage: {100 * similar_pairs / (len(rotations) * (len(rotations) - 1)):.2f}%")
```

## Summary

**Yes, relocated Gaussians maintain identical normals/orientation to their source Gaussians.**

This is by design in the MCMC relocation algorithm:
- ✅ Rotation (and thus normal) is **copied exactly** from the source Gaussian
- ✅ Position is **copied exactly** from the source Gaussian
- ✅ Color and features are **copied exactly** from the source Gaussian
- ⚙️ Only opacity and scale are **adjusted** by the MCMC kernel
- 🎲 SGLD noise gradually moves clones apart over time

This design preserves local surface structure and accelerates convergence by inheriting good orientations from alive Gaussians.





