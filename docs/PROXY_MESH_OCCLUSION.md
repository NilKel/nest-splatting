# Proxy-Mesh Occlusion Rendering

Z-cull of surfels against a per-scene **proxy occluder mesh** — a triangle mesh
fused at the scene's *opaque frontier* (the depth where cumulative transmittance
saturates). Used in three places:

1. **Training / finetune** — CUDA per-pixel occluder (`set_occluder_depth`) +
   per-Gauss pruning behind the mesh (`speed_comparison/finetune_mesh_cull.py`).
2. **Baked benchmarking** — deployment-config FPS/quality bench
   (`speed_comparison/bench_bfc_meshcull.py`, `diff_surfel_bake_render_lean_occ`).
3. **WebGPU viewer (Halloumi-WS)** — a depth-only mesh pre-pass feeding the
   surfel-cull compute pass (`mesh_depth.wgsl` → `surfel_cull.wgsl` bit 3).

**Why**: WebGPU has no per-pixel early termination — every sorted surfel behind
an opaque wall still costs sort keys and fragment work. Culling surfels behind
the frontier *before* the radix sort removes 51–64 % of the per-frame surfel
load on the brain scene with no visible loss (mesh built in LAST_DEPTH_MODE —
see § Losslessness). It also enables backface-cull–style deployments where the
model was *finetuned under the same cull*, so the render matches training.

**Deployed example** (brain, live on bitymi-demos): checkpoint
`…/3D_SH_res/nse_SV_..._noisetrain_brutebf5k02_r2bs` (`_pruned` derivative
baked), bundle `himalaya_brain/brain_brutebf_r2_bfc_meshcull.bitymi`, mesh
`bitymi-demos/test-data/brain_mesh_bfc_r2.ply`, card URL params
`mesh_cull=1&mesh_normal_margin=-0.012&bfc=1&bfc_cos=0.2`.

---

## 1. Building the mesh

### 1.1 TSDF fusion of depth frontiers — `speed_comparison/build_proxy_mesh.py`

Renders the neural model at every camera and TSDF-fuses a per-view **depth
frontier** map (Open3D `ScalableTSDFVolume` via `utils/mesh_utils.py`'s
`GaussianExtractor`), then marching-cubes the volume:

```bash
conda run -n nest_splatting python speed_comparison/build_proxy_mesh.py \
    --model_path <ckpt> --out_dir speed_comparison/proxy_meshes/<scene> \
    [--use_train_views] [--depth_ratio 1.0] [--push_back 1.0] [--voxel 0.004]
```

- **Depth source** = the rasterizer's median-depth aux channel
  (`out_others[5:6]`). Two frontier definitions:
  - default: the **T=0.5 crossing** (median depth). Simple, but culling at a
    T=X crossing discards X of the remaining energy *by definition* — visible
    dimming behind fuzzy surfaces.
  - **`LAST_DEPTH_MODE`** (compile-time `#ifdef` in the rasterizer's
    `forward.cu`): the median slot instead carries the **deepest surviving
    contributor** depth (the T-saturation frontier). Fusing this is the
    **lossless** recipe — nothing that still contributes is behind the mesh.
- `--voxel`: TSDF voxel size. Fine voxels trace fuzzy noise into ragged,
  fragmented meshes; the brain result that shipped used a **coarse 0.012**
  voxel → smooth 601 K-triangle web-ready mesh, 51–64 % cull.
- `--depth_ratio`: 1.0 = pure median/last depth; 0 blends expected depth
  (2DGS unbounded-scene convention).
- `--push_back`: multiplicative fudge pushing the fused surface slightly
  behind the frontier so a later erosion step has room.

### 1.2 Erode + smooth — `speed_comparison/shrink_smooth_mesh.py`

```bash
conda run -n nest_splatting python speed_comparison/shrink_smooth_mesh.py \
    --input …/proxy_mesh_cleaned.ply --output … --shrink 0.03 --smooth 20
```

Shifts every vertex **inward** along −normal by `--shrink` metres (erosion —
depth from any external camera grows, occlusion happens later, safer), then
Taubin-smooths (edge-preserving, no further shrink). Front-face "cheating"
fuzz that picks up background color stays un-occluded.

---

## 2. Training-side cull — `speed_comparison/finetune_mesh_cull.py`

Finetunes a checkpoint **under the cull it will be deployed with**, so the
model redistributes energy that the cull removes:

- **`MeshDepthBaker`**: Open3D raycasting of the (optionally normal-offset)
  mesh at every training camera → per-view fp32 `[H*W]` cam-Z depth maps
  (non-hits = +inf), precomputed once.
- **Per-Gauss prune**: Gaussians whose depth exceeds the mesh depth in *all*
  training views are behind the occluder → pruned. (This is the "prune behind
  the mesh first" step used before baking the brain checkpoint.)
- **Per-pixel CUDA cull** (`--per_pixel_cull`): installs the view's depth map
  via the rasterizer's **`set_occluder_depth(map)`** device-global
  (`FORWARD::setOccluderDepth` / `clearOccluderDepth`, mirrored in BACKWARD so
  the backward's contributor set matches the forward). Fragments with
  `depth > occluder[pix]` are dropped inside the render kernel — the exact
  semantics the viewer applies at inference.
- **`--mesh_normal_margin M`**: pushes each vertex along its **outward**
  normal by `M` metres before raycasting. Sign convention (same everywhere):
  **positive = outward** (mesh inflates, silhouette grows, cull more
  aggressive on the front face), **negative = inward** (mesh deflates, front
  cull more forgiving). The brain finetune used `-0.03`; the deployed viewer
  card uses `-0.012`.
- FastGS densify/prune during the finetune is mesh-cull-aware (decisions based
  on the actual post-cull render).

## 3. Baked benchmark — `speed_comparison/bench_bfc_meshcull.py`

Benches the LEAN CONIC baked renderer under the deployment config, following
`scripts/benchmark_baked.py` methodology (one loop cycling all test views,
cuda-event timing). Variants: `A_plain` (no culls), `B_bfc` (CUDA backface
cull only), `C_bfc_meshocc` (BFC + per-view occluder depth map; the per-frame
`set_occluder_depth` pointer swap is inside the timed loop). Uses the
`diff_surfel_bake_render_lean_occ` submodule. Quality = per-frame
PSNR/SSIM/LPIPS vs GT per variant.

---

## 4. WebGPU viewer runtime (Halloumi-WS)

### 4.1 Loading

- **Bundle chunk**: `.bitymi` chunk `KIND_MESH = 6` carries the mesh; parsed in
  `splat-app.ts`, loaded by `src/mesh-loader.ts` (which computes per-vertex
  outward normals CPU-side: area-weighted face-normal accumulation,
  normalized once at load).
- **Standalone**: `?mesh_url=<ply>` bootstraps the cull path without re-baking
  the bundle (iteration / A-B testing).

### 4.2 Depth pre-pass — `src/shaders/mesh_depth.wgsl`

One draw call per frame rasterizes the mesh **depth-only** (vertex-only
pipeline — no fragment stage, zero color attachments, `depth32float`
attachment at canvas resolution; fixed-function early-Z writes interpolated
NDC-z). The vertex shader applies the runtime normal margin *geometrically*:

```wgsl
let pos_world_pushed = position + mesh_cull_params.normal_margin_m * normal;
```

This superseded an older view-space Z shift (`pos_view.z += margin`), which
never extended the *silhouette* — pixels just outside the mesh boundary stayed
at far-plane so boundary surfels behind the mesh escaped the cull. Pushing
along normals grows/shrinks the silhouette itself, which is the correct fix.
The margin lives in a tiny group-1 uniform, dialable at runtime via Tweakpane.

### 4.3 Cull consumption — `src/shaders/surfel_cull.wgsl` (accel bit 3, `MESH_CULL`)

The surfel-cull compute pass (which runs **before** the radix sort — culled
surfels never enter the sort, which is where most of the mobile frame time
goes) samples the depth texture at each surfel's projected pixel and rejects
surfels behind the mesh. Robustness layers, each fixing an observed popping
mode:

- **Range check**: view-clip lets surfels project up to ±1.2× outside the
  frame; out-of-range sample → *keep* the surfel (never false-cull on a
  texture-edge load).
- **Footprint-aware 5-tap sampling**: center + ±r on each axis (r = projected
  surfel radius in px, clamped [1, 64]), combined **most-permissively** — any
  far-plane tap (no mesh at that pixel) ⇒ keep outright; otherwise compare
  against the **deepest** tap. A single center point-sample flickers two ways
  under sub-pixel camera motion (mesh-silhouette texels flip finite↔far, and a
  wide surfel's center samples one texel while its splat spans dozens); no
  depth-band fade can bridge a discontinuity in the *sampled* value.
- **Scale slack + metric fade band** (anti-popping on the depth compare):
  a surfel gets `max(sx, sy) · MESH_SLACK_K` (K = 2.0) of extra depth
  allowance (big discs whose centers sit behind the wall but whose rims are
  visible survive), then opacity ramps 1→0 over `MESH_FADE_BAND_M` = 0.012 m
  beyond the slack instead of a step — stateless and view-continuous. Metres
  → NDC-z via the projective slope `dz/dz_ndc = −p11/z²`. Cull fires only
  when faded opacity ≤ 1/255.
- **Debug modes**: accel bit 7 = silhouette-only cull (fires on any mesh hit —
  separates sample-pixel bugs from depth-compare bugs), bit 8 = inverted
  compare, `?mesh_debug=1` draws the mesh as a translucent green overlay
  (`mesh_overlay.wgsl`) to sanity-check transforms.

### 4.4 URL parameters

| Param | Default | Meaning |
|---|---|---|
| `mesh_url` | — | standalone proxy-mesh PLY (else bundle `KIND_MESH` chunk) |
| `mesh_cull` | `0` | `1`/`true` enables the Z-cull (off by default so a bundle with a mesh chunk doesn't silently change rendering; auto-disabled when no mesh loaded) |
| `mesh_normal_margin` | `0.0` | runtime normal-push in metres (alias: legacy `mesh_margin`); match the CUDA finetune's `--mesh_normal_margin` |
| `mesh_debug` | `0` | translucent green mesh overlay |
| `mesh_sample_mode` | — | sampling-mode override (debug) |

---

## 5. Losslessness & pitfalls

- **Culling at a T=X crossing discards X of the remaining energy by
  definition.** A mesh fused at the T=0.5 median frontier dims everything
  behind fuzzy surfaces by up to half. The lossless recipe is
  **LAST_DEPTH_MODE (deepest surviving contributor) + coarse TSDF voxel
  (0.012)**; verified on brain at 51–64 % cull with the 601 K-tri mesh.
- **Fine TSDF voxels on fuzzy scenes** produce ragged, fragmented meshes with
  ragged saturation depth — go coarse first.
- **Sign conventions**: viewer/CUDA `mesh_normal_margin` positive = outward.
  `shrink_smooth_mesh.py --shrink` positive = **inward** erosion (it's a
  build-time bake of a negative margin). Don't apply both without accounting.
- **Train-vs-deploy margin mismatch**: the finetune bakes the model to a
  specific offset mesh; the viewer margin should match what the finetune used
  (brain: trained at −0.03, shipped card runs −0.012 — chosen empirically as
  the least-lossy runtime value for that mesh).
- The mesh depth pre-pass is one cheap vertex-only draw; the win is upstream
  of the **sort** (fewer keys), not in fragment work alone.

## Related docs

- [`HALLOUMI_WS_VIEWER.md`](HALLOUMI_WS_VIEWER.md) — full viewer pipeline the
  cull pass sits in (surfel_cull → preprocess → sort → tile_raster).
- [`BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md) — bundle/chunk format & upload.
- [`DEPLOY_DEMO.md`](DEPLOY_DEMO.md) — single-scene deploy procedure.
