# Nest-Splatting Checkpoint Information

## 1. Configuration Storage

### ✅ Where is the config saved?

The exact configuration used for training is saved in **`cfg_args`** file in the output directory:

```
outputs/[method]/[dataset]/[scene]/[experiment_name]/cfg_args
```

**Example:**
```
outputs/add/nerf_synthetic/mic/retryall/cfg_args
```

**Contents:**
- CLI arguments (source_path, model_path, resolution, eval, etc.)
- Stored as a Python `Namespace` object representation
- **Note:** The YAML config path is referenced but not saved in full

**What's saved:**
```python
Namespace(
    sh_degree=3,
    source_path='/path/to/data',
    model_path='outputs/add/nerf_synthetic/mic/retryall',
    images='images',
    resolution=1,
    white_background=False,
    ...
)
```

**What's NOT saved in cfg_args:**
- Full YAML config (you need to reference the original YAML file)
- Method-specific parameters (method, hybrid_levels, cat_coarse2fine)
- Training hyperparameters from YAML

---

## 2. Checkpoint Contents

### ✅ What's saved in the checkpoints?

Checkpoints are saved at **iteration milestones** (e.g., 30000):

#### **A. Gaussian Parameters** (`point_cloud/iteration_30000/point_cloud.ply`)

**For ALL methods (baseline/add/cat):**
- ✅ Gaussian positions (`xyz`)
- ✅ Gaussian normals (placeholder, usually zeros)
- ✅ Spherical Harmonics features (`f_dc`, `f_rest`)
- ✅ Opacity
- ✅ Scaling
- ✅ Rotation

**Per-Gaussian Features (add & cat modes):**
- ❌ **NOT saved in PLY file**
- ⚠️ **Lost if you only save PLY!**
- These are stored in `_gaussian_features` parameter but NOT serialized to PLY

#### **B. INGP Model** (`ngp_30000.pth`)

**For ALL methods:**
- ✅ Hash encoder parameters (`hash_encoding.embeddings`, `hash_encoding.offsets`)
- ✅ View direction encoder (`encoder_dir.params`)
- ✅ RGB MLP weights (`mlp_rgb.params`)

**Contents:**
```python
{
    'model_state_dict': {
        'encoder_dir.params': Tensor,           # SH encoder weights
        'hash_encoding.embeddings': Tensor,     # Hashgrid table
        'hash_encoding.offsets': Tensor,        # Level offsets
        'mlp_rgb.params': Tensor,               # MLP weights
    }
}
```

---

## 3. What's Missing? ⚠️

### **Critical Issue: Per-Gaussian Features Not Saved!**

For **add** and **cat** modes, the per-Gaussian features (`_gaussian_features`) are:
- ✅ Trained and optimized during training
- ❌ **NOT saved** in `point_cloud.ply`
- ❌ **NOT saved** in `ngp_*.pth`

**Impact:**
- If you reload a checkpoint, per-Gaussian features will be **re-initialized to zeros**
- This breaks add/cat mode rendering unless you add explicit save/load

**Solution Needed:**
You should add explicit save/load for `_gaussian_features` in:
1. `scene/gaussian_model.py` → `save_ply()` / `load_ply()`
2. OR save separately as `gaussian_features_30000.pth`

---

## 4. Rendering Script

### ✅ New Script: `render_from_checkpoint.py`

**Features:**
- ✅ Renders test images from any checkpoint folder
- ✅ Configurable stride (default: 1 = all images)
- ✅ **Keeps image indices consistent** (renders 0, stride, 2*stride, ...)
- ✅ Auto-detects method, iteration, YAML config
- ✅ Computes PSNR, SSIM, L1 metrics
- ✅ Saves metrics to `metrics.txt`

**Usage:**
```bash
# Render all test images (stride=1)
python render_from_checkpoint.py --checkpoint_dir outputs/add/nerf_synthetic/drums/retryall

# Render every 25th image
python render_from_checkpoint.py --checkpoint_dir outputs/add/nerf_synthetic/drums/retryall --stride 25

# Specify method and iteration explicitly
python render_from_checkpoint.py \
    --checkpoint_dir outputs/baseline/nerf_synthetic/mic/test \
    --stride 10 \
    --method baseline \
    --iteration 30000
```

**Output:**
```
outputs/add/nerf_synthetic/drums/retryall/rerender_stride_1/
├── 000_gt.png       # Consistent indices
├── 000_render.png
├── 001_gt.png       # (or 025, 050, etc. if stride > 1)
├── 001_render.png
├── ...
└── metrics.txt      # PSNR, SSIM, L1
```

**Why indices are consistent:**
- Uses `range(0, len(test_cameras), stride)`
- Always renders camera 0, then stride, then 2*stride, etc.
- Different from training where stride sampling might vary

---

## 5. Checkpoint Directory Structure

```
outputs/[method]/[dataset]/[scene]/[experiment_name]/
├── cfg_args                              # Saved CLI arguments
├── cameras.json                          # Camera parameters
├── ngp_30000.pth                         # INGP checkpoint (hash + MLP)
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply               # Gaussian parameters (NO per-Gaussian features!)
├── final_test_renders/                   # Final test renders (stride=25)
│   ├── 000_gt.png
│   ├── 000_render.png
│   ├── 025_gt.png
│   ├── 025_render.png
│   └── metrics.txt
└── training_output/                      # Training visualizations
    ├── 10000.png
    ├── 10000_gt.png
    └── ...
```

---

## 6. Summary

| Question | Answer |
|----------|--------|
| **1. Config saved?** | ✅ Yes, in `cfg_args` (Namespace repr) |
| **2a. Baseline checkpoint?** | ✅ Gaussians (PLY) + Hashgrid + MLP (pth) |
| **2b. Add/Cat checkpoint?** | ⚠️ Same as baseline, but **per-Gaussian features NOT saved!** |
| **3. Rendering script?** | ✅ Created `render_from_checkpoint.py` with consistent indices & configurable stride |

---

## 7. Recommended Fix for Per-Gaussian Features

Add this to `scene/gaussian_model.py`:

```python
def save_ply(self, path):
    # ... existing code ...
    
    # NEW: Save per-Gaussian features separately
    if hasattr(self, '_gaussian_features') and self._gaussian_features.numel() > 0:
        features_path = path.replace('.ply', '_gaussian_features.pth')
        torch.save(self._gaussian_features, features_path)
        print(f"Saved per-Gaussian features to {features_path}")

def load_ply(self, path, args=None):
    # ... existing code ...
    
    # NEW: Load per-Gaussian features if available
    features_path = path.replace('.ply', '_gaussian_features.pth')
    if os.path.exists(features_path):
        self._gaussian_features = nn.Parameter(
            torch.load(features_path, map_location='cuda').requires_grad_(True)
        )
        print(f"Loaded per-Gaussian features from {features_path}")
    else:
        # Initialize with zeros if not found (for backward compatibility)
        if args and hasattr(args, 'method') and args.method in ['add', 'cat']:
            gaussian_feat_dim = 24  # Or infer from args
            self._gaussian_features = nn.Parameter(
                torch.zeros((self.get_xyz.shape[0], gaussian_feat_dim), device='cuda').requires_grad_(True)
            )
```

This ensures add/cat modes can be properly resumed from checkpoints!



