# Per-Gaussian Features Fix

## Problem
Per-Gaussian features in `hybrid_features` mode were:
1. All zeros (not being passed from Python to CUDA)
2. Had `requires_grad=False` (gradients not flowing)

## Root Causes

### Issue 1: Variable Shadowing in CUDA
In `forward.cu` case 6, a local `feat` array was being declared that shadowed the outer scope's `feat` array:
```cuda
float feat[CHANNELS] = {0};  // This created a NEW local variable
```

This meant the per-Gaussian features were being copied into a local array that was never used, and the outer `feat` array remained uninitialized.

### Issue 2: Incorrect Property Access in Python
In `gaussian_renderer/__init__.py`, the code was checking `pc.get_gaussian_features is not None` which checks if the METHOD exists (always True), not if the tensor exists.

Should have been: `pc.get_gaussian_features() is not None` (call the method).

### Issue 3: Tensor Initialization in Constructor
In `scene/gaussian_model.py`, `_gaussian_features` was initialized as:
```python
self._gaussian_features = torch.empty(0)  # NOT a Parameter, requires_grad=False
```

This caused issues when `training_setup()` was called before the proper Parameter was assigned, because the optimizer would see a non-trainable tensor.

## Fixes Applied

### 1. Fixed CUDA Variable Shadowing (`forward.cu`)
Changed:
```cuda
case 6: {
    float feat[CHANNELS] = {0};  // BAD: shadows outer feat
    ...
}
```

To:
```cuda
case 6: {
    // Initialize outer feat array (don't shadow it!)
    for(int i = 0; i < CHANNELS; i++) feat[i] = 0.0f;
    ...
}
```

### 2. Fixed Python Method Calls (`gaussian_renderer/__init__.py`)
Changed:
```python
if pc.get_gaussian_features is not None:
    colors_precomp = pc.get_gaussian_features
```

To:
```python
if pc.get_gaussian_features() is not None:
    colors_precomp = pc.get_gaussian_features()
```

### 3. Fixed Constructor Initialization (`scene/gaussian_model.py`)
Changed:
```python
self._gaussian_features = torch.empty(0)
```

To:
```python
self._gaussian_features = None
```

## Verification
After these fixes:
- Per-Gaussian features should have non-zero values
- `colors_precomp.requires_grad` should be `True`
- Gradients should flow through per-Gaussian features during backprop

## Files Modified
1. `/home/nilkel/Projects/nest-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`
2. `/home/nilkel/Projects/nest-splatting/gaussian_renderer/__init__.py`
3. `/home/nilkel/Projects/nest-splatting/scene/gaussian_model.py`


