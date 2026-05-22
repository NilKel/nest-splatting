# Halloumi-WS Viewer ("WebSplatter")

The **TypeScript / WebGPU** 2DGS viewer at [`/home/nilkel/Projects/Halloumi-WS`](../../Halloumi-WS). Rewrite of the upstream WebSplatter codebase tailored to our `.bitymi` bundle format with SV/SB color paths, BC7 + ASTC atlas decode, and a 2DGS-only render pipeline.

> **Deployment status**: NOT currently deployed to the `bitymi-demos` site — that still serves the older Rust/wasm `Halloumi-web-splat` viewer. See [§ Deploying to bitymi-demos](#deploying-to-bitymi-demos) at the end. Day-to-day editing happens here; switching the live demos to this viewer is a small swap.

---

## 1. What it is

A standalone Vite + TypeScript app that:
- Loads `.bitymi` bundles (or raw 2DGS `.ply` files) from disk or a URL.
- Parses out the PLY chunk, optional `cameras.json`, optional `scene.nat2` atlas.
- Uploads Gaussians as a 32-byte / surfel buffer to GPU.
- Optionally uploads the atlas as a BC7 or ASTC-4×4 `texture_2d_array` (one layer per 8192-row stripe, to fit Adreno/Mali limits).
- Per frame: surfel-cull → preprocess → radix sort → tile rasterize → display.
- All shaders are WGSL; pipeline is entirely compute-pass driven (no vertex/fragment except final blit).

Built from upstream `WebSplatter` (Wang et al., anonymous repo) plus our additions.

---

## 2. Repo layout

```
Halloumi-WS/
├── src/
│   ├── main.ts                   entry point (mounts Vite app)
│   ├── splat-app.ts              app glue: bundle parse, scene setup, render loop,
│   │                             camera presets, UI panel, orbit pivot logic
│   ├── camera.ts                 Camera class (position, rotation mat4, focal)
│   ├── camera-control.ts         orbit / pan / wheel-zoom controller, ray-bbox
│   │                             pivot init, lookAt re-orientation
│   ├── gaussian-renderer.ts      GPU pipeline: surfel_cull, preprocess, radix sort,
│   │                             indirect dispatch, tile raster, display blit
│   ├── ply-loader.ts             parses 2DGS PLY → Surfel buffer; uploads atlas
│   ├── render-settings.ts        uniform layout + setters (sh_bias, scaling, OAC/SPR/BFC)
│   ├── reorder-types.ts          radix-sort pipeline factory
│   ├── shaders/                  WGSL — see § 3 for per-shader purpose
│   ├── utils/loaders/ply/        PLY parsers (INRIA V1 + SurfelPlyParser) + Nat2Parser
│   ├── utils/simple-console.ts   UI loading spinner / status box
│   └── workers/ply-worker.ts     off-thread PLY parsing (keeps main thread responsive)
├── public/                       static assets served by Vite
├── package.json                  scripts: dev, build, preview
└── README.md                     upstream README (mostly intact)
```

### 3. Shaders

| File | Role |
|---|---|
| `surfel_cull.wgsl` | Per-Gauss frustum + backface + opacity cull; unpacks `scale_rot` (3× u32 = 6× f16) into world-space scale + quat; appends survivors to a compact index list. Reads accel flags from `render_settings` (OAC, SPR, BFC). |
| `gaussian_calculate_indirect_dispatch.wgsl` | Sizes the radix-sort and tile-raster dispatches from the live surfel count post-cull. |
| `preprocess_2dgs.wgsl` | Computes screen-space conic / AABB, depth, and color for each surviving surfel. SV path: softmax mix over K voronoi sites; SB path: spherical-beta lobes; SH-only path: degree-3 SH eval. Writes per-surfel records + depth-keys for sort. |
| `gaussian_sort_local_histogram.wgsl`, `gaussian_sort_blelloch.wgsl`, `gaussian_sort_scatter.wgsl` | 4-pass radix sort over 32-bit depth keys. Apple Silicon-safe variant (no cross-workgroup deadlock). |
| `render_2dgs.wgsl` | Tile-based α-compositing of the sorted surfel records. One workgroup per 16×16 tile. |
| `display.wgsl` | Final blit of the tile-raster output into the swap-chain texture. |

---

## 4. Build + run

```bash
cd /home/nilkel/Projects/Halloumi-WS
npm install          # one-time
npm run dev          # vite dev server, http://localhost:5173, instant HMR
npm run build        # tsc strict-check + vite build → dist/
npm run preview      # serve dist/ for sanity
```

**Type-check only** (no bundle): `npx tsc --noEmit`. The build will fail if TS errors are present (`build` runs `tsc` then `vite build`).

---

## 5. Bundle loader (BITYMI)

`splat-app.ts::parseBitymi` reads the magic-prefixed container our `pack_bitymi.py` produces:

```
[magic "BITYMI01" 8B] [n_chunks u32] [TOC: n × (kind u32, off u64, size u64)] [chunks ...]

KIND_PLY     = 0
KIND_NPZ     = 1
KIND_CAMERAS = 2
KIND_NAT2    = 3
KIND_NATL    = 4
```

Each chunk is sliced into its own `ArrayBuffer` so it can be transferred across the worker boundary without aliasing the parent. The atlas (NAT2) is decoded via `Nat2Parser.ts`, which reads the 64-byte header (format, dims, dequant scale/offset, layer cuts) and the rects + atlas payload. See [`docs/BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md) for the NAT2 v2 byte layout.

When `atlasBuffer == null` the viewer renders SH-only (no residual texture) — useful for sanity checks.

---

## 6. Surfel buffer layout (CPU and GPU)

32 bytes per Gauss; shared between CPU and GPU exactly:

| offset (B) | field | type | notes |
|---:|---|---|---|
| 0–11 | `position` | 3× f32 | world-space xyz |
| 12–15 | `opacity_shape` | u32 | packed: low 16 = opacity (f16), high 16 = β kernel shape (f16) |
| 16–19 | `scale_rot[0]` | u32 | packed: low 16 = scale_x (f16), high 16 = scale_y (f16) |
| 20–23 | `scale_rot[1]` | u32 | packed: low 16 = rot_w (f16), high 16 = rot_x (f16) |
| 24–27 | `scale_rot[2]` | u32 | packed: low 16 = rot_y (f16), high 16 = rot_z (f16) |
| 28–31 | `_pad` | u32 | reserved |

The CPU-side `pc.surfel_data: Float32Array` is exactly this buffer reinterpreted as f32 (8 floats / stride). For picking we view the same `ArrayBuffer` as `Uint32Array` to read the packed u32s — see `splat-app.ts::pickGaussAt`.

**Quaternion → world-space normal** (2DGS surface is the local xy-plane → normal = R column 2):
```
n.x = 2*(q.x*q.z + q.w*q.y)
n.y = 2*(q.y*q.z − q.w*q.x)
n.z = 1 − 2*(q.x² + q.y²)
```
(Matches `quat_to_rotmat` in `surfel_cull.wgsl`.)

---

## 7. Camera control + orbit pivot

`camera-control.ts` manages an orbit camera: single-finger / left-drag rotates around `this.center`, two-finger / right-drag pans, wheel + pinch zoom. The interesting bit is **where `this.center` lives** — getting it right per-view is what makes orbit feel grounded vs. drifty.

### 7.1 How the pivot is set

Three entry points, each in `splat-app.ts`:

| Trigger | Path | What runs |
|---|---|---|
| Initial load | bottom of bundle-load block | `control.resetToCamera()` → `pickAtCenter()` |
| Preset switch (instant) | `setPreset(i, animated=false)` | `camera.set_preset()` → `control.resetToCamera()` → `pickAtCenter()` |
| Preset switch (animated) | `setPreset(i, animated=true)` → `startTransition()` → tick reaches `t≥1` | `transition = null` → `control.resetToCamera()` → `pickAtCenter()` |
| Reset button | onClick | `camera.set_preset(cameras[0])` → `control.resetToCamera()` → `pickAtCenter()` |
| Double-tap / click | `pickAndPivot(x, y)` | `pickGaussAt(x, y)` → `control.setOrbitPivot(hit)` (reorients camera to look at hit) |

Both pieces matter:
- `control.resetToCamera()` puts the pivot on the camera's forward ray at the midpoint of the ray-bbox intersection. This is a *stable but coarse* pivot — drift-free across views but not anchored to actual visible surface.
- `pickAtCenter()` refines by ray-casting through the canvas centre, finding the nearest surfel, and calling `control.setOrbitDepth(t)` to slide the pivot along the forward ray to the surface depth. **Does not re-orient the camera** — preserves the preset's framing exactly.

### 7.2 Ray-disk intersection in `pickGaussAt`

The legacy implementation filtered surfels by perpendicular distance from the ray to the surfel center, then returned the *center*. `pickAtCenter` then projected that center onto the camera's forward axis to get a depth `t`. For tilted surfels the center sits perpendicular-off the ray, so the projected depth drifts from the actual ray-surface intersection by `perp_offset · tan(angle_to_normal)`. Across views this manifested as the orbit pivot jittering off the surface.

Current `pickGaussAt` (`splat-app.ts`):
1. Same coarse "near the ray" filter (perpendicular distance ≤ 0.5% scene radius).
2. For each candidate: unpack quat from `scale_rot[1..2]` (half-to-float in JS via `halfToFloat`), build the world-space normal, compute the **exact ray-plane intersection** `t = ((P − O)·n) / (D·n)`.
3. Falls back to the center projection only when the plane is near-parallel to the ray (`|D·n| < 1e-6`).
4. Returns `origin + t * dir` — a point **on the ray and on the visible surfel's tangent plane**.

This stays consistent across view changes because the result lives on the actual surface, not on a tilted disk's center.

---

## 8. Render settings + accel toggles

Bound in group 1 of every pipeline. Mutable at runtime via the Tweakpane UI (top-right):

| Flag | Effect (read by `surfel_cull.wgsl`) |
|---|---|
| `OAC` (Opacity-Aware Culling) | Drop surfels whose opacity falls below a screen-space pixel-coverage threshold. |
| `SPR` (Splatting Pixel Reorder) | Reorder surfel records by tile coverage prior to raster to improve locality. |
| `BFC` (Backface Cull) | Drop surfels whose normal points away from camera (`(pos − cam)·n > 0`). |
| `sh_bias` | Additive bias on the SH eval before clamp (default `0.5`). |
| `res_bias` | Additive bias on the residual sample after dequant (default `0.0`). |
| Surfel scale | Multiplier on f16-unpacked scale_xy. Useful for shrinking surfels for debugging. |

Atlas toggle is separate (a button in the panel). When OFF, the renderer uses SH-only; bake-residual atlas is skipped entirely.

---

## 9. Differences vs. upstream WebSplatter

| Area | Halloumi-WS (ours) | Upstream WebSplatter |
|---|---|---|
| Primitive | 2DGS surfels only | 3DGS (full 3D Gaussian) |
| Bundle format | BITYMI (PLY + cameras + NAT2 atlas) | raw `.ply` + adjacent `cameras.json` |
| Atlas decode | BC7 + ASTC 4×4 `texture_2d_array` w/ layer cuts | none |
| Color path | SV (Spherical Voronoi mix), SB (Spherical Beta), or SH degree 3 | SH only |
| Sort | Apple-Silicon-safe radix (`gaussian_sort_*`) | upstream sort variant |
| Culling | OAC + SPR + BFC accel toggles | OAC + SPR only |
| Camera presets | Smooth cubic-eased lerp+slerp animations | hard jumps |
| Orbit pivot | Ray-disk intersection per preset switch | upstream lacks this |
| Render output | Tile-raster compute → final display blit | similar tile raster |

---

## 10. Pitfalls

- **Atlas layering** (corrected 2026-05-15 — the old "6-layer hard cap" claim was wrong): WebGPU on Adreno/Mali exposes `max_texture_dimension_2d = 8192`, so the NAT2 packer slices the atlas into ≤8192-row stripes (one `texture_2d_array` layer each; viewer reads `n_layers` + `layer_cuts` from the NAT2 header and binary-searches per pixel — see `Nat2Parser.ts` / Rust `pointcloud.rs`). There is **no hard layer-count limit in the viewer**: the only constraints are per-layer height ≤ `max_texture_dimension_2d` (8192) and total layers ≤ `maxTextureArrayLayers` (WebGPU guaranteed minimum **256**). "6 layers" was a *typical bake-config output*, never a viewer ceiling — `brain_blursplit` ships a working 7-layer (6784×55424) atlas. The real cost of more layers is **download size**, not a render failure. So `atlas_height` up to `8192 × 256` ≈ 2M rows is technically renderable; bandwidth bites long before the texture-array limit does.
- **BC7 vs. ASTC capability**: `device.features.has('texture-compression-bc')` for BC7, `'texture-compression-astc'` for ASTC. Desktop browsers usually have BC7; Android Adreno/Mali have ASTC only. The bundle format determines what's loaded — ship both `_astc` and BC7 variants for full coverage (see `BITYMI_BUNDLES.md`).
- **Half-precision quaternion drift**: f16 round-trip can knock `|q|` away from 1 by a few percent. `pickGaussAt` renormalizes defensively; `surfel_cull.wgsl::quat_to_rotmat` also does `q * inverseSqrt(dot(q,q))`. Don't assume f16-packed quats are unit.
- **`cameras.json` missing**: bundle loaders return `camerasBuffer = null` if absent. App falls back to a bbox-default camera pose (`bbox_center − r/2 · (1,1,1)`, looking back at origin). Without cameras, the "Next / Prev view" buttons are hidden.
- **Preset transition doesn't run `pickAtCenter` mid-animation**: the orbit pivot stays stale until the transition reaches `t = 1`, then both `resetToCamera()` and `pickAtCenter()` run in one frame. If you cut a transition short by dragging, the controller cancels and uses whatever pivot the last update wrote — usually the bbox-ray pivot.
- **`surfel_data` is Float32Array, but reading the packed u32 fields requires a `Uint32Array` view of the same buffer**. Don't allocate a fresh Uint32Array each call — share one across the loop. (Current `pickGaussAt` allocates once per pick — fine for click events, not per-frame.)

---

## 11. Adding new features — quick map

| You want to … | File to edit |
|---|---|
| Add a UI toggle | `splat-app.ts` (Tweakpane), `render-settings.ts` (uniform layout), `surfel_cull.wgsl` (read flag) |
| Add a per-Gauss attribute | `ply-loader.ts` (parser), `surfel_cull.wgsl` + `preprocess_2dgs.wgsl` (consume) |
| Add a new color model | `preprocess_2dgs.wgsl` (eval), bundle format spec, `ply-loader.ts` (load params) |
| Tweak orbit feel | `camera-control.ts` |
| Change pivot picking | `splat-app.ts::pickGaussAt` and `pickAtCenter` |
| Add a new bundle chunk kind | `splat-app.ts::parseBitymi` + new parser in `utils/loaders/` |

---

## 12. Deploying to bitymi-demos

The `bitymi-demos/viewer/` directory currently ships `web_splats.js` + `web_splats_bg.wasm` from the **Rust** `Halloumi-web-splat` build (compiled 2026-05-06). To switch to this TS viewer:

1. `cd /home/nilkel/Projects/Halloumi-WS && npm run build` → produces `dist/`
2. `rm -rf /home/nilkel/Projects/bitymi-demos/viewer && cp -r dist /home/nilkel/Projects/bitymi-demos/viewer`
3. Verify `viewer/index.html` exposes a `?bundle=` query-string parameter that `splat-app.ts` already reads.
4. `cd /home/nilkel/Projects/bitymi-demos && git add viewer && git commit -m "viewer: switch to Halloumi-WS TS build" && git push`

GitHub Pages will redeploy in ~1–2 min. Test across BC7 (desktop) and ASTC (Android phone) before nuking the Rust artifact.

---

## See also

- [`docs/BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md) — bake → NAT2 → `.bitymi` → HF upload pipeline
- [`docs/BAKED_RENDERING.md`](BAKED_RENDERING.md) — atlas baking math + BC7 compression strategy
- Upstream Halloumi-WS README (`/home/nilkel/Projects/Halloumi-WS/README.md`) for the original WebSplatter paper context
