# CLI Methods Quick Reference

## Available Methods

The `--method` argument is now available in training and evaluation scripts:

| Method | Description | CLI Value |
|--------|-------------|-----------|
| **Baseline** | Standard NeST (default) | `--method baseline` |
| **Surface** | Surface potential (dot product per-Gaussian in CUDA) | `--method surface` |
| **Surface Blend** | Alpha-blend vectors, then dot product in Python | `--method surface_blend` |
| **Surface RGB** | Surface potential + diffuse RGB (dual hashgrid) | `--method surface_rgb` |

## Training Examples

### Baseline Mode (Default)
```bash
python train.py -s data/nerf_synthetic/hotdog \
  -m outputs/baseline_test \
  --config configs/nerfsyn.yaml \
  --ingp
```

### Surface Mode
```bash
python train.py -s data/nerf_synthetic/hotdog \
  -m outputs/surface_test \
  --config configs/nerfsyn.yaml \
  --method surface \
  --ingp
```

### Surface Blend Mode (NEW!)
```bash
python train.py -s data/nerf_synthetic/hotdog \
  -m outputs/surface_blend_test \
  --config configs/nerfsyn.yaml \
  --method surface_blend \
  --ingp
```

### Surface RGB Mode
```bash
python train.py -s data/nerf_synthetic/hotdog \
  -m outputs/surface_rgb_test \
  --config configs/nerfsyn.yaml \
  --method surface_rgb \
  --ingp
```

## Evaluation Examples

**Important:** Always use the same `--method` for evaluation as was used for training!

### Evaluate Baseline Model
```bash
python eval_render.py -m outputs/baseline_test \
  --yaml nerfsyn \
  --method baseline \
  --skip_mesh
```

### Evaluate Surface Model
```bash
python eval_render.py -m outputs/surface_test \
  --yaml nerfsyn \
  --method surface \
  --skip_mesh
```

### Evaluate Surface Blend Model
```bash
python eval_render.py -m outputs/surface_blend_test \
  --yaml nerfsyn \
  --method surface_blend \
  --skip_mesh
```

### Evaluate Surface RGB Model
```bash
python eval_render.py -m outputs/surface_rgb_test \
  --yaml nerfsyn \
  --method surface_rgb \
  --skip_mesh
```

## Other Useful Flags

### Training Flags
- `--ingp`: Enable INGP (required for surface modes)
- `--config configs/nerfsyn.yaml`: Specify config file
- `--port 6010`: Specify different port for GUI (avoid conflicts)
- `--detect_anomaly`: Enable gradient anomaly detection (debug)

### Evaluation Flags
- `--skip_train`: Skip rendering training views
- `--skip_test`: Skip rendering test views
- `--skip_mesh`: Skip mesh extraction
- `--iteration 30000`: Evaluate specific iteration (default: -1 = latest)

## Method Defaults

If `--method` is not specified:
- **Training:** Defaults to `baseline`
- **Evaluation:** Defaults to `baseline`

⚠️ **Always specify the method explicitly to avoid confusion!**

## Files Updated

- ✅ `train.py` - Added `surface_blend` to choices
- ✅ `eval_render.py` - Added `--method` argument with all choices
- ✅ `hash_encoder/modules.py` - Supports all 4 methods
- ✅ `gaussian_renderer/__init__.py` - Rendering logic for all methods
- ✅ CUDA kernels - Compiled with all method support


