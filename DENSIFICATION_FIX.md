# Densification/Pruning Fix for `hybrid_levels=0`

## Problem

When `hybrid_levels=0`, `_gaussian_features` is `None` (baseline mode doesn't use per-Gaussian features).

But densification code was trying to:
```python
new_gaussian_features = self._gaussian_features[selected_pts_mask]  # CRASH! None object
```

## Solution

Added `None` checks in all densification/pruning operations:

### 1. `densify_and_clone()` (line ~469)
```python
# Only clone gaussian_features if they exist
new_gaussian_features = self._gaussian_features[selected_pts_mask] if self._gaussian_features is not None else None
```

### 2. `densify_and_split()` (line ~447)
```python
# Only split gaussian_features if they exist
new_gaussian_features = self._gaussian_features[selected_pts_mask].repeat(N, 1) if self._gaussian_features is not None else None
```

### 3. `densification_postfix()` (line ~291)
```python
d = {...}

# Only add gaussian_features if they exist
if new_gaussian_features is not None:
    d["gaussian_features"] = new_gaussian_features

# Later when unpacking:
if "gaussian_features" in optimizable_tensors:
    self._gaussian_features = optimizable_tensors["gaussian_features"]
```

### 4. `prune_points()` (line ~355)
```python
# Only update gaussian_features if they exist
if "gaussian_features" in optimizable_tensors:
    self._gaussian_features = optimizable_tensors["gaussian_features"]
```

## Result

Now `hybrid_levels=0` can safely:
- Clone Gaussians (densification)
- Split Gaussians (densification)
- Prune Gaussians (pruning)

Without trying to access `_gaussian_features` when it's `None`.




