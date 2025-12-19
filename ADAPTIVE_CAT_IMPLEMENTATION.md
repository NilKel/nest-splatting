# Adaptive Cat Implementation Plan

## Overview
Implement `--method adaptive_cat` that combines per-Gaussian features and hashgrid with learnable binary weights. Each Gaussian has `total_levels * per_level_dim` features (e.g., 6×4=24D) and a single scalar blend weight. The last level features are blended during training, but at inference, binary decisions skip either the hashgrid query OR the 3D intersection computation for maximum speedup.

## Key Design Decisions (from user requirements)
- **Weight initialization**: 0.0 → sigmoid(0) = 0.5 (equal blend initially)
- **Regularization**: Entropy-based to push weights toward 0 or 1, annealed from iteration 15000
- **Inference threshold**: 0.5 (weight > 0.5 uses Gaussian only, else hashgrid only)
- **Weight scope**: Single scalar per Gaussian (not per-level)
- **Critical optimization**: When weight indicates Gaussian-only, skip 3D intersection computation entirely and use shared memory path (like hash_in_cuda=False). When weight indicates hashgrid-only, skip Gaussian features and only compute 3D intersection for hashgrid query.

## Architecture

### Training Mode (Smooth Blending)
```
For each Gaussian:
  - First (total_levels - 1) levels: Use per-Gaussian features directly (20D for 5 levels)
  - Last level (4D): Blend = weight * gaussian_feat + (1-weight) * hash_feat
    - Requires 3D intersection computation to query hashgrid
  - Output: 24D features → MLP → RGB
```

### Inference Mode (Binary Decision)
```
If weight > 0.5 (Gaussian-dominated):
  - Use ALL 24D per-Gaussian features directly
  - Skip 3D intersection computation entirely
  - Use shared memory path (fast, like original 2DGS)
  
Else (Hashgrid-dominated):
  - Compute 3D intersection
  - Query hashgrid for last level (4D)
  - Use first (total_levels - 1) levels from per-Gaussian (20D)
  - Concatenate: 20D + 4D = 24D
  - Output: 24D features → MLP → RGB
```

## Implementation Steps

### 1. Add Command-Line Arguments (`train.py`)

```python
parser.add_argument("--method", type=str, default="baseline",
                    choices=["baseline", "cat", "adaptive", "adaptive_add", "adaptive_cat", ...],
                    help="... 'adaptive_cat' (cat with learnable binary blend weights)")
parser.add_argument("--lambda_adaptive_cat", type=float, default=0.01,
                    help="Entropy regularization weight for adaptive_cat binarization")
parser.add_argument("--adaptive_cat_anneal_start", type=int, default=15000,
                    help="Iteration to start annealing adaptive_cat regularization")
parser.add_argument("--adaptive_cat_inference", action="store_true",
                    help="Use binary decisions at inference (skip intersection or skip Gaussian features)")
```

### 2. Initialize Gaussian Parameters (`scene/gaussian_model.py`)

In `create_from_pcd()`:

```python
if hasattr(args, 'method') and args.method == "adaptive_cat":
    per_level_dim = 4
    num_levels = 6  # total_levels from config
    self._gaussian_feat_dim = num_levels * per_level_dim  # 24D
    
    # Initialize features to small random values
    gaussian_feats = torch.randn((N, self._gaussian_feat_dim), device="cuda") * 0.01
    self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
    
    # Initialize blend weight to 0.0 (sigmoid(0) = 0.5)
    blend_weight = torch.zeros((N, 1), device="cuda")
    self._adaptive_cat_weight = nn.Parameter(blend_weight.requires_grad_(True))
```

Add to `training_setup()`:

```python
if self._gaussian_feat_dim > 0 and hasattr(self, '_adaptive_cat_weight'):
    l.append({'params': [self._adaptive_cat_weight], 'lr': training_args.opacity_lr, "name": "adaptive_cat_weight"})
```

### 3. Configure Hashgrid (`hash_encoder/modules.py`)

Create single-level hashgrid at finest resolution:

```python
self.is_adaptive_cat_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_cat"

if self.is_adaptive_cat_mode:
    # Use only finest level for blending
    finest_resolution = all_resolutions[-1]
    self.hashgrid_levels = 1
    self.hashgrid_disabled = False
    
    config = SimpleNamespace(
        device="cuda",
        otype="HashGrid",
        n_levels=1,
        n_features_per_level=cfg_encoding.hashgrid.dim,  # 4D
        log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
        base_resolution=finest_resolution,
        finest_resolution=finest_resolution,
        init_mode='uniform',
        per_level_scale=1.0,
        range=self.voxel_range,
    )
    
    self.hash_encoding = register_GridEncoder(config)
    self.resolutions = [finest_resolution]
    
    # Disable C2F for adaptive_cat
    self.level_mask = False
    self.init_active_level = 1
    self.active_hashgrid_levels = 1
```

### 4. Renderer Integration (`gaussian_renderer/__init__.py`)

Add render_mode=11 for adaptive_cat:

```python
is_adaptive_cat_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_cat_mode') and ingp.is_adaptive_cat_mode

if is_adaptive_cat_mode and pc._gaussian_feat_dim > 0:
    # Pass per-Gaussian features (24D) + blend weight (1D) + inference_flag (1D)
    gaussian_features = pc.get_gaussian_features  # (N, 24)
    blend_weight = torch.sigmoid(pc._adaptive_cat_weight)  # (N, 1)
    
    # Inference flag: 0 = training (smooth blend), 1 = inference (binary)
    inference_flag = torch.ones((len(blend_weight), 1), device="cuda") if args.adaptive_cat_inference else torch.zeros((len(blend_weight), 1), device="cuda")
    
    colors_precomp = torch.cat([gaussian_features, blend_weight, inference_flag], dim=1)  # (N, 26)
    shs = None
    render_mode = 11  # adaptive_cat mode
    
    # Encode: (total_levels << 16) | (1 << 8) for single hashgrid level
    levels = (ingp.levels << 16) | (1 << 8)
```

### 5. CUDA Forward Pass (`submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu`)

Add case 11 with binary decision logic:

```cuda
case 11: {
    // adaptive_cat mode: blend last level during training, binary decision at inference
    // rgb buffer: [N, 26] = [24D features | 1D weight | 1D inference_flag]
    
    const int total_levels = (level >> 16) & 0xFF;  // e.g., 6
    const int per_level_dim = l_dim;  // 4
    const int total_dim = total_levels * per_level_dim;  // 24
    
    int gauss_id = collected_id[j];
    const float* gauss_feat = &rgb[gauss_id * (total_dim + 2)];
    const float weight = gauss_feat[total_dim];  // Blend weight (sigmoid output)
    const float inference_flag = gauss_feat[total_dim + 1];  // 0=training, 1=inference
    
    // Binary decision at inference
    bool use_gaussian_only = (inference_flag > 0.5f) && (weight > 0.5f);
    bool use_hashgrid_only = (inference_flag > 0.5f) && (weight <= 0.5f);
    
    if (use_gaussian_only) {
        // FAST PATH: Use all per-Gaussian features, skip 3D intersection
        // This is like the original 2DGS path (no hashgrid query needed)
        for(int i = 0; i < total_dim; i++) {
            feat[i] = gauss_feat[i];
        }
    }
    else if (use_hashgrid_only) {
        // HASHGRID PATH: Use first (total_levels-1) from Gaussian, last level from hashgrid
        // Copy first (total_levels - 1) levels directly
        for(int i = 0; i < (total_levels - 1) * per_level_dim; i++) {
            feat[i] = gauss_feat[i];
        }
        
        // Query hashgrid for last level (requires 3D intersection - already computed)
        float hash_feat[4];
        query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, 
                                   collec_offsets, appearance_level, hash_features, 
                                   1, l_scale, Base, align_corners, interp, contract, debug);
        
        // Use ONLY hashgrid for last level (weight is near 0)
        const int last_level_start = (total_levels - 1) * per_level_dim;
        for(int i = 0; i < per_level_dim; i++) {
            feat[last_level_start + i] = hash_feat[i];
        }
    }
    else {
        // TRAINING PATH: Smooth blending
        // Copy first (total_levels - 1) levels directly
        for(int i = 0; i < (total_levels - 1) * per_level_dim; i++) {
            feat[i] = gauss_feat[i];
        }
        
        // Query hashgrid for last level
        float hash_feat[4];
        query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, 
                                   collec_offsets, appearance_level, hash_features, 
                                   1, l_scale, Base, align_corners, interp, contract, debug);
        
        // Blend last level: weight * gaussian + (1-weight) * hash
        const int last_level_start = (total_levels - 1) * per_level_dim;
        for(int i = 0; i < per_level_dim; i++) {
            feat[last_level_start + i] = weight * gauss_feat[last_level_start + i] 
                                        + (1.0f - weight) * hash_feat[i];
        }
    }
    
    break;
}
```

**Note**: The 3D intersection computation happens BEFORE the switch statement, so for `use_gaussian_only` path, we're still computing it unnecessarily. To truly skip it, we need to add an early check before the intersection computation.

### 5b. Optimize 3D Intersection Skipping (Advanced)

Before the intersection computation in the per-pixel loop, add:

```cuda
// Early check for adaptive_cat Gaussian-only path
if (render_mode == 11) {
    int gauss_id = collected_id[j];
    const float weight = rgb[gauss_id * 26 + 24];  // Extract weight
    const float inference_flag = rgb[gauss_id * 26 + 25];  // Extract inference flag
    
    if (inference_flag > 0.5f && weight > 0.5f) {
        // Use Gaussian-only path: copy features from shared memory and skip intersection
        float feat[CHANNELS] = {0};
        for(int i = 0; i < 24; i++) {
            feat[i] = rgb[gauss_id * 26 + i];
        }
        
        // Continue with alpha blending (skip intersection computation entirely)
        // Use 2D Gaussian approximation for alpha
        float2 d = {xy.x - pixf.x, xy.y - pixf.y};
        float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
        float power = -0.5f * rho2d;
        if (power > 0.0f) continue;
        
        float G = exp(power);
        if(beta > 0.0) G = (1.0 + beta) * G / (1.0 + beta * G);
        float alpha = min(0.99f, opa * G);
        if (alpha < 1.0f / 255.0f) continue;
        
        // Alpha blend features
        for(int ch = 0; ch < CHANNELS; ch++) {
            C[ch] += feat[ch] * alpha * T;
        }
        T *= (1.0f - alpha);
        
        // Skip the rest of the loop (no 3D intersection needed)
        continue;
    }
}

// Normal path: compute 3D intersection
const float2 xy = collected_xy[j];
const float3 Tu = collected_Tu[j];
// ... rest of intersection computation
```

### 6. CUDA Backward Pass (`submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu`)

Add case 11 with gradient routing:

```cuda
case 11: {
    // Gradients for adaptive_cat blending
    const int total_levels = (level >> 16) & 0xFF;
    const int per_level_dim = l_dim;
    const int total_dim = total_levels * per_level_dim;
    
    int gauss_id = collected_id[j];
    const float weight = collected_colors[gauss_id * (total_dim + 2) + total_dim];
    const float inference_flag = collected_colors[gauss_id * (total_dim + 2) + total_dim + 1];
    
    bool use_gaussian_only = (inference_flag > 0.5f) && (weight > 0.5f);
    bool use_hashgrid_only = (inference_flag > 0.5f) && (weight <= 0.5f);
    
    if (use_gaussian_only) {
        // Gaussian-only: all gradients go to per-Gaussian features
        for(int i = 0; i < total_dim; i++) {
            atomicAdd(&dL_dcolors[gauss_id * (total_dim + 2) + i], dL_dfeat[i]);
        }
        // No gradient for weight (frozen at inference)
    }
    else if (use_hashgrid_only) {
        // Hashgrid-only: first levels to Gaussian, last level to hashgrid
        for(int i = 0; i < (total_levels - 1) * per_level_dim; i++) {
            atomicAdd(&dL_dcolors[gauss_id * (total_dim + 2) + i], dL_dfeat[i]);
        }
        
        // Gradient for hashgrid (last level)
        float dL_dhash[4];
        const int last_start = (total_levels - 1) * per_level_dim;
        for(int i = 0; i < per_level_dim; i++) {
            dL_dhash[i] = dL_dfeat[last_start + i];
        }
        query_feature_backward<false, 4, 4>(dL_dhash, xyz, voxel_min, voxel_max, 
                                            collec_offsets, appearance_level, hash_features, 
                                            1, l_scale, Base, align_corners, interp, contract, debug, 
                                            dL_dfeatures, dL_dxyz);
    }
    else {
        // Training: smooth blending with weight gradients
        // Gradient for per-Gaussian features (first total_levels-1 direct, last level weighted)
        for(int i = 0; i < (total_levels - 1) * per_level_dim; i++) {
            atomicAdd(&dL_dcolors[gauss_id * (total_dim + 2) + i], dL_dfeat[i]);
        }
        
        // Query hashgrid for last level
        float hash_feat[4];
        query_feature<false, 4, 4>(hash_feat, xyz, voxel_min, voxel_max, 
                                   collec_offsets, appearance_level, hash_features, 
                                   1, l_scale, Base, align_corners, interp, contract, debug);
        
        // Gradient for last level Gaussian features (scaled by weight)
        const int last_start = (total_levels - 1) * per_level_dim;
        for(int i = 0; i < per_level_dim; i++) {
            atomicAdd(&dL_dcolors[gauss_id * (total_dim + 2) + last_start + i], 
                      weight * dL_dfeat[last_start + i]);
        }
        
        // Gradient for blend weight
        float dL_dweight = 0.0f;
        const float* gauss_last = &collected_colors[gauss_id * (total_dim + 2) + last_start];
        for(int i = 0; i < per_level_dim; i++) {
            dL_dweight += dL_dfeat[last_start + i] * (gauss_last[i] - hash_feat[i]);
        }
        atomicAdd(&dL_dcolors[gauss_id * (total_dim + 2) + total_dim], dL_dweight);
        
        // Gradient for hashgrid (scaled by 1-weight)
        float dL_dhash[4];
        for(int i = 0; i < per_level_dim; i++) {
            dL_dhash[i] = (1.0f - weight) * dL_dfeat[last_start + i];
        }
        query_feature_backward<false, 4, 4>(dL_dhash, xyz, voxel_min, voxel_max, 
                                            collec_offsets, appearance_level, hash_features, 
                                            1, l_scale, Base, align_corners, interp, contract, debug, 
                                            dL_dfeatures, dL_dxyz);
    }
    
    break;
}
```

### 7. Training Loss (`train.py`)

Add entropy regularization with annealing:

```python
# Adaptive_cat: entropy regularization to force binary weights
adaptive_cat_reg_loss = torch.tensor(0.0, device="cuda")
if args.method == "adaptive_cat" and gaussians._gaussian_feat_dim > 0:
    # Compute annealing factor
    if iteration >= args.adaptive_cat_anneal_start:
        progress = (iteration - args.adaptive_cat_anneal_start) / (opt.iterations - args.adaptive_cat_anneal_start)
        anneal_factor = min(1.0, progress)  # Linear ramp from 0 to 1
    else:
        anneal_factor = 0.0
    
    # Entropy regularization: -w*log(w) - (1-w)*log(1-w)
    # Penalizes 0.5, encourages 0 or 1
    weight = torch.sigmoid(gaussians._adaptive_cat_weight)
    eps = 1e-7
    entropy = -(weight * torch.log(weight + eps) + (1 - weight) * torch.log(1 - weight + eps))
    adaptive_cat_reg_loss = args.lambda_adaptive_cat * anneal_factor * entropy.mean()

total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_cat_reg_loss + ...
```

### 8. Densification/MCMC Support (`scene/gaussian_model.py`)

Handle adaptive_cat_weight in all operations:

```python
# In densify_and_split()
new_adaptive_cat_weight = None
if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
    new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask].repeat(N, 1)

# In densify_and_clone()
new_adaptive_cat_weight = None
if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
    new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask]

# In densification_postfix()
if new_adaptive_cat_weight is not None:
    d["adaptive_cat_weight"] = new_adaptive_cat_weight

# In prune_points()
if "adaptive_cat_weight" in optimizable_tensors:
    self._adaptive_cat_weight = optimizable_tensors["adaptive_cat_weight"]

# In cat_tensors_to_optimizer()
if group["name"] not in tensors_dict:
    optimizable_tensors[group["name"]] = group["params"][0]
    continue

# In MCMC methods (_mcmc_update_params, relocate_gs, add_new_gs)
adaptive_cat_weight = None
if self._adaptive_cat_weight.numel() > 0:
    adaptive_cat_weight = self._adaptive_cat_weight[idxs]
# Return and use in densification_postfix
```

### 9. Checkpoint Support (`scene/gaussian_model.py`)

```python
# In capture()
return (
    ...,
    self._adaptive_cat_weight if hasattr(self, '_adaptive_cat_weight') else torch.empty(0),
)

# In restore()
if len(model_args) == 20:  # New format with adaptive_cat
    (..., self._adaptive_cat_weight) = model_args
else:
    # Old format compatibility
    self._adaptive_cat_weight = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
```

### 10. Warmup Checkpoint Loading (`train.py`)

```python
elif args.method == "adaptive_cat":
    num_levels = cfg_model.encoding.levels
    per_level_dim = cfg_model.encoding.hashgrid.dim
    gaussians._gaussian_feat_dim = num_levels * per_level_dim
    
    # Initialize features
    gaussian_feats = torch.randn((len(gaussians.get_xyz), gaussians._gaussian_feat_dim), 
                                 device="cuda").float() * 0.01
    gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
    
    # Initialize blend weight to 0.0
    blend_weight = torch.zeros((len(gaussians.get_xyz), 1), device="cuda").float()
    gaussians._adaptive_cat_weight = nn.Parameter(blend_weight.requires_grad_(True))
    
    print(f"[ADAPTIVE_CAT MODE] Initialized {len(gaussians.get_xyz)} Gaussians")
    print(f"[ADAPTIVE_CAT MODE] Per-Gaussian features: {gaussians._gaussian_feat_dim}D")
    print(f"[ADAPTIVE_CAT MODE] Blend weight: 1D per Gaussian")
```

### 11. Logging (`train.py`)

```python
# In progress bar
if args.method == "adaptive_cat":
    mean_weight = torch.sigmoid(gaussians._adaptive_cat_weight).mean().item()
    pct_gaussian = (torch.sigmoid(gaussians._adaptive_cat_weight) > 0.5).float().mean().item() * 100
    loss_dict["W"] = f"{mean_weight:.2f}"
    loss_dict["G%"] = f"{pct_gaussian:.0f}"
    if args.mcmc:
        loss_dict["AdR"] = f"{adaptive_cat_reg_loss.item():.{5}f}"

# In tensorboard
if tb_writer and args.method == "adaptive_cat":
    tb_writer.add_scalar('adaptive_cat/mean_weight', mean_weight, iteration)
    tb_writer.add_scalar('adaptive_cat/pct_gaussian', pct_gaussian, iteration)
    tb_writer.add_scalar('adaptive_cat/reg_loss', adaptive_cat_reg_loss.item(), iteration)
```

### 12. Update rasterize_points.cu

Handle colors_precomp size for render_mode=11:

```cuda
// Around line 90
if (colors.numel() != 0 && render_mode != 4 && render_mode != 6 && render_mode != 11) 
    C = colors.size(1);
else if (render_mode == 11) 
    C = 26;  // 24D features + 1D weight + 1D inference_flag
```

## Testing Strategy

1. **Sanity check**: Run with `--method adaptive_cat --hybrid_levels 5` on ficus
2. **Verify blending**: Check that weights start at ~0.5 and converge to 0 or 1
3. **Compare to cat mode**: Ensure quality matches `--method cat --hybrid_levels 5`
4. **Inference speedup**: Measure FPS with `--adaptive_cat_inference` flag
5. **Weight distribution**: Visualize which Gaussians use which representation
6. **MCMC compatibility**: Test with `--mcmc` flag

## Expected Behavior

- **Early training (iter < 15000)**: Weights near 0.5, smooth blending, no regularization
- **Mid training (15000-25000)**: Entropy regularization ramps up, weights start diverging
- **Late training (> 25000)**: Weights converge to ~0 or ~1, clear binary decisions
- **Inference**: 
  - Gaussians with weight > 0.5: Skip 3D intersection, use shared memory (fast!)
  - Gaussians with weight ≤ 0.5: Compute intersection, query hashgrid (accurate)

## Files to Modify

1. `train.py` - Arguments, loss, warmup init, logging
2. `scene/gaussian_model.py` - Parameters, densification, checkpoints
3. `hash_encoder/modules.py` - Single-level hashgrid config
4. `gaussian_renderer/__init__.py` - render_mode=11 setup
5. `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu` - Blending logic
6. `submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu` - Gradients
7. `submodules/diff-surfel-rasterization/rasterize_points.cu` - colors_precomp size

## Compilation

After modifying CUDA files:

```bash
cd submodules/diff-surfel-rasterization
pip install -e .
```


