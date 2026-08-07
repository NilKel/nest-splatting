# Baked Forward Rendering Pipeline — Sort to Textured Composite

Reference for the deployed baked renderer (`diff_surfel_bake_render_lean`,
CONIC variant). Written to answer two specific questions:

1. **Is the per-Gaussian texture stored in an axis-aligned bounding box in
   texture space, or packed into an oriented bounding box together with a
   rotation matrix?** → Oriented, in the surfel's own eigenbasis, and no
   rotation matrix needs storing. § 7.
2. **Do texture lookups in a CUDA kernel pay a penalty versus a pixel
   shader?** → Not in this implementation; we already issue hardware `tex`
   instructions with fixed-function BC7 decode and bilinear filtering. § 8.

Measured costs are in § 9.

---

## 1. What is stored per Gaussian

Geometry and appearance are Lagrangian (attached to each surfel):

| quantity | shape | notes |
|---|---|---|
| center | `[N,3]` fp32 | world-space μ |
| rotation | `[N,4]` fp32 | quaternion (w,x,y,z) |
| scales | `[N,2]` fp32 | `(sx, sy)` — 2D surfel, no third axis |
| opacity | `[N,1]` | pre-activation |
| shape β | `[N,1]` | beta-kernel exponent, `--kernel beta_scaled` |
| SV colour | `[N,K,·]` | K=7 Spherical-Voronoi sites + colours (view-dependent base) |
| atlas rect | `[N,4]` fp32 | `(u0_px, v0_px, w_px, h_px)` — **offset and size only** |

The residual texture is Eulerian at training time (a hash grid + MLP
evaluated at world position), but baking collapses it to an explicit
per-surfel RGB texture. The atlas is one large 2D image; `atlas_rects`
says where each surfel's tile lives inside it.

Note what is *absent* from `atlas_rects`: any orientation. That is the
crux of § 7.

---

## 2. Stage 1 — Preprocess (one thread per Gaussian)

`preprocessCUDA` in `forward.cu`. Per surfel:

**a. Build the tangent frame and the ray-splat transform.**

```cuda
glm::mat3 R = quat_to_rotmat(rot);        // Gaussian's own rotation
glm::mat3 S = scale_to_mat(scale, mod);   // diag(sx, sy, 1)
glm::mat3 L = R * S;                      // L[0] = sx·R_col0, L[1] = sy·R_col1

glm::mat3x4 splat2world = glm::mat3x4(
    glm::vec4(L[0], 0.0),
    glm::vec4(L[1], 0.0),
    glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
);
T = glm::transpose(splat2world) * world2ndc * ndc2pix;
```

`T`'s three rows become `Tu, Tv, Tw`. This composite maps homogeneous
pixel coordinates to the surfel's **local `(u,v)` parametric frame**,
where `u = ±1` is `±1σ` along the first principal axis and `v = ±1` is
`±1σ` along the second. The frame is the Gaussian's own eigenbasis by
construction — `L[0]` and `L[1]` are the rotated, scaled principal axes.

**b. Evaluate the view-dependent base colour.** Spherical Voronoi with
K=7 sites, evaluated once per Gaussian here rather than per pixel in the
render loop (a 10–25% saving; view direction is per-Gaussian, not
per-pixel). Result is written to `geomState.rgb` as fp16.

**c. Project to a screen-space ellipse and compute the tile footprint.**
The disk projects to an ellipse; `aabb_mode 5` (production default) takes
the SnugBox tight bounding box and then runs AccuTile, which emits only
the tiles the ellipse actually intersects rather than every tile in the
bounding rectangle. Falls back to the rectangular AABB on a degenerate
conic (near-edge-on surfels), so coverage is never lost.

**d. (CONIC path) Cache the per-Gaussian reconstruction coefficients.**
See § 6 — instead of re-deriving `(u,v)` from a cross product at every
fragment, preprocess stores `(u₀, v₀, J⁻¹, ∂w/∂x, ∂w/∂y)` per Gaussian.

Output: `tiles_touched[i]` per Gaussian.

---

## 3. Stage 2 — Prefix sum and key expansion

```
InclusiveSum(tiles_touched) → point_offsets
num_rendered = point_offsets[P-1]
```

`num_rendered` is the total number of (Gaussian, tile) instances. Then
`duplicateWithKeys` writes one 64-bit key per instance:

```cuda
uint64_t key = y * grid.x + x;          // tile id → high 32 bits
key <<= 32;
key |= *((uint32_t*)&depths[idx]);      // float bits → low 32 bits
```

Packing the depth's raw IEEE-754 bit pattern into the low word works
because all depths are positive after near-plane culling, and positive
floats compare identically as unsigned integers. So one integer sort
orders the whole buffer tile-major, depth-minor.

---

## 4. Stage 3 — Radix sort

```cuda
cub::DeviceRadixSort::SortPairs(
    ..., point_list_keys_unsorted, point_list_keys,
         point_list_unsorted,      point_list,
    num_rendered, 0, 32 + bit);        // bit = ceil(log2(num_tiles))
```

Only the low `32 + log2(tiles)` bits are sorted — the high bits are
known-zero, so those passes are skipped. This is the single global sort;
there is no per-tile sort afterwards.

## 5. Stage 4 — Tile ranges

`identifyTileRanges` scans the sorted key array for tile-id boundaries and
writes `ranges[tile] = (start, end)` into the image state. Each tile now
has a contiguous, depth-ordered slice of the instance list.

---

## 6. Stage 5 — Render (one 16×16 thread block per tile)

`renderBakedCUDA`, `__launch_bounds__(256)`. One thread per pixel.

**Batched cooperative staging.** The block walks its tile's range in
batches of 256. Each thread loads exactly one Gaussian's data from global
memory into shared:

```cuda
__shared__ int    collected_id[256];
__shared__ float2 collected_xy[256];
__shared__ float4 collected_normal_opacity[256];
__shared__ float3 collected_Tu[256], collected_Tv[256], collected_Tw[256];
__shared__ float  collected_shapes[256];
```

so 256 global reads serve 256×256 = 65,536 (pixel, Gaussian) evaluations.
This amortization is the main reason a tile-compute rasterizer competes
with fixed-function hardware here, and it has no equivalent in a fragment
shader, where each fragment fetches its own primitive data.

**Per-fragment: recover the local `(u,v)`.**

Baseline ray-splat — intersect the camera ray with the surfel's tangent
plane, in the tangent frame directly:

```cuda
float3 k = pix.x * Tw - Tu;
float3 l = pix.y * Tw - Tv;
float3 p = cross(k, l);
float2 s = { p.x / p.z, p.y / p.z };     // local (u,v), units of σ
```

CONIC variant (deployed) — algebraically identical, but hoists the
Gaussian-constant part to preprocess:

```cuda
float du_lin = J.x*dx + J.y*dy;
float dv_lin = J.z*dx + J.w*dy;
float denom  = 1.0f + dw.x*dx + dw.y*dy;
if (denom < 0.1f) continue;              // past validity boundary
float u = uv0.x + du_lin / denom;
float v = uv0.y + dv_lin / denom;
```

This is **exact, not a linearization**: `u = p.x/p.z` with `p.x` and
`p.z` each affine in pixel coordinates, so `u` is exactly a rational
function of the pixel offset. Storing `(u₀, v₀, J⁻¹)` as fp16 while
keeping the correction denominator `(dw.x, dw.y)` in fp32 preserves
bit-identical output versus the cross-product path — verified across the
mip360 set (0.001 dB noise on the SH-only lane, 0 dB on SH+atlas).

**Kernel evaluation.** `ρ = min(ρ_3d, ρ_2d)` — the surfel's own falloff
max-pooled against a screen-space low-pass, which keeps sub-pixel surfels
from aliasing. Then the restricted beta kernel:

```
α = min(0.99, opacity · max(0, 1 − ρ/k²)^β),   k² = 9 for beta_scaled
```

Compact support: `ρ ≥ k²` contributes nothing.

**Texture fetch.** Local `(u,v)` → atlas pixel, then one hardware sample:

```cuda
float au = u0_px + (s.x + UV_EXTENT) / (2.0f*UV_EXTENT) * u_span - 0.5f;
float av = v0_px + (s.y + UV_EXTENT) / (2.0f*UV_EXTENT) * v_span - 0.5f;
au = clamp(au, u0_px, u0_px + u_span - 1.001f);
av = clamp(av, v0_px, v0_px + v_span - 1.001f);

float4 rgba = tex2D<float4>(atlas_tex_obj, au + 0.5f, av + 0.5f);
feat[ch] += rgba[ch] * atlas_scale + atlas_offset;
```

`UV_EXTENT = 4.0`, so the texture covers `s ∈ [−4σ, +4σ]²`. The map from
local `(u,v)` to atlas texel is a pure affine rescale — a shift and a
scale per axis, nothing more. That is the whole point of § 7.

The clamp keeps bilinear taps inside the surfel's own rect so neighbouring
rects never bleed across.

**Composite.**

```cuda
float w = alpha * T;
for (ch) C[ch] += feat[ch] * w;
T = test_T;                              // T *= (1 - alpha)
```

Front-to-back with two early exits: fragments with `α < 1/255` are
skipped, and a pixel terminates once `T < 0.0001`. The whole block exits
its batch loop when `__syncthreads_count(done) == 256`. Neither early-out
is expressible in fixed-function ROP blending, which must process every
fragment that reaches it.

Final: `out_color = C + T · background`.

---

## 7. Texture parameterization — oriented, with no stored rotation

**The texture lives in the surfel's own oriented eigenbasis.** There is no
axis-aligned bounding box in world space anywhere in the pipeline, and no
per-texture rotation matrix is stored.

The bake kernel makes this explicit. For texel `(ui, vi)` of a
`grid_size × grid_size` tile:

```cuda
float step = 2.0f * uv_extent / (float)grid_size;
float u = ((float)ui + 0.5f) * step - uv_extent;      // texel-center convention
float v = ((float)vi + 0.5f) * step - uv_extent;

// R columns (local tangent frame), from the Gaussian's own quaternion
float r00 = 1-2*(qy*qy+qz*qz), r10 = 2*(qx*qy+qw*qz), r20 = 2*(qx*qz-qw*qy);
float r01 = 2*(qx*qy-qw*qz),   r11 = 1-2*(qx*qx+qz*qz), r21 = 2*(qy*qz+qw*qx);

// Local → world: xyz = center + u·sx·R_col0 + v·sy·R_col1
xyz.x = centers[g*3+0] + u*sx*r00 + v*sy*r01;
xyz.y = centers[g*3+1] + u*sx*r10 + v*sy*r11;
xyz.z = centers[g*3+2] + u*sx*r20 + v*sy*r21;
```

`xyz = μ + u·sx·R₀ + v·sy·R₁` is the parametric form of an oriented
rectangle in the tangent plane. The residual field (hash grid + MLP) is
sampled at those world positions and the result written to the texel.

At render time the inverse is the affine map in § 6. The rotation never
appears there because it was already applied when the surfel's `(u,v)`
frame was constructed in preprocess — `T = transpose(splat2world) ·
world2ndc · ndc2pix` folds `R` into the same matrix that produces the
ray-splat. The renderer recovers `(u,v)` in the oriented frame *directly*;
it never works in a world-axis-aligned space that would need rotating out.

Three consequences:

**Storage.** `atlas_rects` is 4 floats per Gaussian: `(u0_px, v0_px,
w_px, h_px)`. Orientation is inherited from the quaternion the Gaussian
already stores as geometry. A world-AABB scheme would need an explicit
rotation *and* would waste texels on the empty corners of a rotated
ellipse.

**Anisotropy.** Resolution is allocated per axis, each axis sized against
its own scale by a Nyquist criterion versus the finest hash-grid cell:

```python
n_cells         = 2.0 * uv_extent * scales / cell_size   # [N,2]
nyquist_samples = 2.0 * n_cells                          # [N,2]
resolutions     = 2 ** ceil(log2(nyquist_samples))
resolutions     = resolutions.clamp(min_res, max_res)
```

So a surfel with `sx ≫ sy` gets a rect like 64×8 rather than a wasteful
64×64. On treehill (174,744 surfels), 14.8% of rects are non-square, and
both orientations appear — 8,309 at 32×64 and 6,338 at 64×32 — because
each tracks its own surfel's aspect ratio:

| rect | count | share |
|---|---:|---:|
| 64×64 | 147,984 | 84.7% |
| 32×64 | 8,309 | 4.8% |
| 64×32 | 6,338 | 3.6% |
| 16×64 | 3,613 | 2.1% |
| 8×64 | 2,393 | 1.4% |
| 64×16 | 2,279 | 1.3% |
| 64×8 | 967 | 0.6% |

**Sampling quality.** Because texel axes align with the surfel's principal
axes, the texel grid is isotropic *in the space the residual actually
varies in*. A world-AABB parameterization would shear the sampling
lattice relative to the ellipse and lose effective resolution along the
minor axis.

---

## 8. Texture lookup cost — CUDA kernel versus pixel shader

A common claim is that CUDA texture reads go through load/store units and
therefore miss the dedicated texture cache, hardware decompression, and
filtering that a pixel shader gets, costing 2–4×. That describes reading
*linear memory* (where `__ldg()` and the read-only data cache apply). It
does not describe this implementation.

The atlas is bound as a `cudaTextureObject_t` over a `cudaArray`:

```cuda
cudaChannelFormatDesc desc = cudaCreateChannelDesc(
    8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7);
cudaMallocArray(&g_bc7_array, &desc, W, H);
...
td.filterMode      = cudaFilterModeLinear;
td.readMode        = cudaReadModeNormalizedFloat;
td.addressMode[0]  = cudaAddressModeClamp;
cudaCreateTextureObject(&g_bc7_tex, &res, &td, nullptr);
```

`tex2D<float4>()` against that object issues a texture instruction through
the same texture units and texture cache hierarchy a fragment shader uses,
with:

- **fixed-function BC7 decompression** (`cudaChannelFormatKindUnsignedBlockCompressed7`),
- **hardware bilinear filtering** (`cudaFilterModeLinear`),
- **hardware clamp addressing**.

None of that is emulated. The performance gap the claim describes is
therefore not available to recover — we are already on that path.

The one genuine fragment-shader-only advantage is quad-based screen-space
derivatives driving automatic mip level selection. We do not use mipmaps:
each surfel's texture is Nyquist-sized at bake time for exactly the
frequency content it carries (§ 7), so there is no LOD chain to select
from. The `min(ρ_3d, ρ_2d)` max-pool handles the sub-pixel aliasing case
that mips would otherwise address.

Moving to hardware rasterization would also *give up* two things this
pipeline depends on:

- **Front-to-back early termination.** A pixel stops once `T < 0.0001`.
  ROP blending processes every fragment delivered to it; there is no
  equivalent cutoff.
- **Cooperative shared-memory staging.** 256 global loads amortized over
  65,536 fragment evaluations per batch. Per-fragment shading refetches
  per fragment.

Empirically the CUDA tile renderer is not leaving hardware throughput on
the table: on an RTX 5090 across the nine mip360 scenes it averages 1456
FPS against FastGS's 1183 (1.23×), and 1943 versus 1138 on garden
(1.71×) — FastGS being a mature, throughput-tuned splat rasterizer.

---

## 9. Measured cost of the texture stage

Adding the residual atlas to the SH/SV base costs **15.1% of frame time
on average**, RTX 5090, CONIC lean renderer:

| scene | SH-only (ms) | SH+atlas (ms) | Δ ms | Δ % | Δ FPS |
|---|---:|---:|---:|---:|---:|
| bicycle | 0.626 | 0.722 | 0.096 | 15.4% | 213 |
| bonsai | 0.715 | 0.793 | 0.078 | 10.9% | 137 |
| counter | 0.605 | 0.693 | 0.088 | 14.6% | 210 |
| flowers | 0.714 | 0.775 | 0.061 | 8.5% | 110 |
| garden | 0.428 | 0.515 | 0.087 | 20.3% | 394 |
| kitchen | 0.560 | 0.669 | 0.109 | 19.5% | 291 |
| room | 0.497 | 0.558 | 0.061 | 12.3% | 220 |
| stump | 0.623 | 0.717 | 0.095 | 15.2% | 212 |
| treehill | 0.758 | 0.905 | 0.147 | 19.4% | 214 |
| **mean** | | | | **15.1%** | |

The FPS column is worth reading with care. Because FPS is `1/t`, the
apparent gap is dominated by how fast the scene already is, not by the
cost of the texture stage: garden shows the largest FPS drop (394) for
the *second smallest* time cost (0.087 ms), while treehill costs 70% more
time (0.147 ms) but shows a 45% smaller FPS gap. Frame time is the
meaningful unit.

What that 0.147 ms buys on treehill: PSNR 13.81 → 22.29 dB (+8.5 dB),
LPIPS 0.592 → 0.342. The baked-with-atlas result (22.29 dB) matches the
neural renderer it replaces (22.38 dB) to within 0.1 dB while running
roughly 11× faster.

Per-fragment the texture stage is four floats of `atlas_rects`, an affine
`(u,v)` → texel map, and one `tex2D`. It sits in the innermost blend loop,
so it is paid once per (pixel, contributing Gaussian) pair.

---

## 10. Source map

| stage | file | symbol |
|---|---|---|
| pipeline driver | `rasterizer_impl.cu` | `Rasterizer::forward` |
| preprocess | `forward.cu` | `preprocessCUDA` |
| tangent frame | `forward.cu` | `compute_transmat` |
| key expansion | `rasterizer_impl.cu` | `duplicateWithKeys` |
| sort | `rasterizer_impl.cu` | `cub::DeviceRadixSort::SortPairs` |
| tile ranges | `rasterizer_impl.cu` | `identifyTileRanges` |
| render + composite | `forward.cu` | `renderBakedCUDA` |
| BC7 texture binding | `rasterize_points.cu` | `SetAtlasBC7CUDA` |
| bake (texel → world) | `diff_surfel_bake/cuda_rasterizer/forward.cu` | ~line 2073 |
| resolution allocation | `scripts/benchmark_baked.py` | `compute_adaptive_resolution` |

Deployed variant is `submodules/diff_surfel_bake_render_lean` built with
`LEAN_FLAGS=CONIC`. The unoptimized reference is
`submodules/diff_surfel_bake_render`; the two are bit-identical in output.
