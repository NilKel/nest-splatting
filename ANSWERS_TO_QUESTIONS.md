# Answers to Your Questions

## 1) Why is it still 30k iterations? ✅ FIXED

**Problem:** The YAML config file's `training_cfg.iterations: 30_000` was overriding the CLI `--iterations` argument.

**Fix:** Modified `train.py` to check if `--iterations` was explicitly set via CLI and preserve it instead of letting the config override it.

Now this will work correctly:
```bash
python train.py -s /path/to/scene -m ./output --yaml ./configs/nerfsyn.yaml --eval --iterations 5000
```

---

## 2) Hash Encoder and MLP Architecture

### Hash Encoder: **Custom GridEncoder** (NOT tiny-cuda-nn)
- **Location:** `gridencoder/` directory
- **Backend:** Custom CUDA implementation in `gridencoder/src/gridencoder.cu`
- **Registration:** Line 38-45 in `hash_encoder/modules.py`:
  ```python
  def register_GridEncoder(cfg_encoding):
      return GridEncoder(
          num_levels = cfg_encoding.n_levels,
          level_dim = cfg_encoding.n_features_per_level,
          per_level_scale = cfg_encoding.per_level_scale,
          base_resolution = cfg_encoding.base_resolution,
          log2_hashmap_size = cfg_encoding.log2_hashmap_size)
  ```

### MLP Decoder: **YES, uses tiny-cuda-nn's fused MLP!**
- **Location:** Lines 153-165 in `hash_encoder/modules.py`
- **Implementation:**
  ```python
  def build_mlp(self, cfg_rgb, input_dim, output_dim = 3):
      cfg_mlp = cfg_rgb.mlp
      return tcnn.Network(  # ← tiny-cuda-nn's fused MLP!
          n_input_dims=input_dim,
          n_output_dims=output_dim,
          network_config={
              "otype": "MLP",
              "activation": "ReLU",
              "output_activation": "None",
              "n_neurons": cfg_mlp.hidden_dim,
              "n_hidden_layers": cfg_mlp.num_layers,
          },
      )
  ```

**So the architecture is:**
1. Custom CUDA hash grid encoder (not tiny-cuda-nn)
2. tiny-cuda-nn fused MLP decoder
3. Spherical Harmonics encoding for view directions (tiny-cuda-nn)

---

## 3) Do we get sample points in 3D space to query the hashgrid and alpha blend?

**YES!** Here's exactly how it works:

### Step-by-step process:

1. **Per-pixel ray-Gaussian intersection** (lines 682-692 in `forward.cu`):
   ```cuda
   const float3 pk = collected_pk[j];  // Gaussian center
   float3 xyz;
   
   // If inside 2D Gaussian footprint, compute 3D intersection point
   if(rho3d <= rho2d){
       const float3 sutu = collected_SuTu[j];
       const float3 svtv = collected_SvTv[j];
       // Compute actual 3D intersection point on Gaussian surface
       xyz = {s.x * sutu.x + s.y * svtv.x + pk.x,
              s.x * sutu.y + s.y * svtv.y + pk.y,
              s.x * sutu.z + s.y * svtv.z + pk.z};
   }
   else xyz = pk;  // Fallback to Gaussian center
   ```

2. **Query hash grid at intersection point** (lines 708-723):
   ```cuda
   float feat[CHANNELS];
   query_feature<false, CHANNELS, 4>(
       feat,           // output features
       xyz,            // 3D query point
       voxel_min, voxel_max, 
       collec_offsets,
       appearance_level, 
       hash_features, 
       ...
   );
   ```

3. **Alpha-blend the features** (lines 725-726):
   ```cuda
   for (int ch = 0; ch < CHANNELS; ch++)
       C[ch] += feat[ch] * w;  // w is the Gaussian weight (alpha)
   ```

4. **Pass blended features to MLP** (line 241 in `gaussian_renderer/__init__.py`):
   ```python
   rendered_image = ingp.rgb_decode(
       rendered_image.view(feat_dim, -1).permute(1, 0), 
       rays_dir
   )
   ```

---

## 4) Is the density at a 3D sample point the density of the Gaussian directly or is the Gaussian evaluated at that point?

**The Gaussian IS evaluated at that point!**

### The weight `w` is computed as:

From lines 627-655 in `forward.cu`:

```cuda
// Evaluate 2D Gaussian at pixel location
float power = -0.5f * (sigma.x * s.x * s.x + sigma.y * s.y * s.y) 
              - sigma.z * s.x * s.y;

if (power > 0.0f)
    continue;

// Compute Gaussian weight
float alpha = min(0.99f, opa * exp(power));  // ← Gaussian evaluation!

if (alpha < 1.0f / 255.0f)
    continue;

float test_T = T * (1 - alpha);
if (test_T < 0.0001f) {
    done = true;
    continue;
}

// Weight for alpha blending
float w = alpha * T;  // ← This is the blending weight
```

**So:**
- `w = alpha * T` where:
  - `alpha = opacity * exp(-0.5 * Mahalanobis_distance²)` ← **Gaussian evaluated at intersection point**
  - `T` = transmittance (accumulated transparency along ray)
  
**This is NOT just the Gaussian's stored density/opacity!** It's the actual Gaussian function evaluated at the 3D intersection point, modulated by the learned opacity parameter.

---

## Summary

| Component | Implementation |
|-----------|----------------|
| Hash Grid | Custom CUDA GridEncoder |
| MLP Decoder | tiny-cuda-nn fused MLP |
| View Encoding | tiny-cuda-nn Spherical Harmonics |
| Sampling | Per-pixel ray-Gaussian intersection points |
| Density | Gaussian evaluated at intersection point |
| Blending | Alpha compositing with evaluated weights |

The key insight: **The 3D intersection points are where the Gaussian is evaluated, then those points query the hash grid, and features are alpha-blended using the Gaussian's evaluated density.**
