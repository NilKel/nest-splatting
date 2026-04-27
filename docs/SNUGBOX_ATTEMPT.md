# SnugBox / AccuTile port to 2DGS bake_render — attempted, reverted

## Goal

Port FastGS / Speedy-Splat's SnugBox + AccuTile ellipse-tight tile binning
into `submodules/diff_surfel_bake_render` to replace the rectangular AABB
tile binning. Expected to reduce (gauss, tile) pair count by ~30–40% for
2DGS scenes (anisotropic surfels benefit even when the kernel is bounded
beta_scaled — the gain comes from anisotropy, not unboundedness).

## Status: reverted

PSNR collapsed to 5.83 in both attempt branches. Reverted to keep the
working `compute_aabb` + rectangular AABB binning. All the OTHER
optimizations (FP16 `rgb`, persistent output buffers, CUDA Event timing,
dead-write removal, `.contiguous()` cleanup, atlas-only render path)
remain in place — they're orthogonal to tile binning.

## What was implemented

### `auxiliary.h`

Added `compute_conic_from_transmat`, `computeEllipseIntersection`,
`processTiles`, `duplicateToTilesTouched` (last three ported almost verbatim
from `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`
of FastGS / Speedy-Splat).

### `forward.cu` `preprocessCUDA`

After `compute_transmat`, call `compute_conic_from_transmat(T, cutoff, …)`
to produce the screen-space ellipse conic `(A, B, E, t, p)`. Then call
`duplicateToTilesTouched(…, nullptr, nullptr)` to count the tiles the
ellipse crosses (count-only mode). Store `(A, B, E, t)` in
`geomState.conic_t` (new `float4*` field) so `duplicateWithKeys` can re-run
the same scan without recomputing the conic.

### `rasterizer_impl.cu` `duplicateWithKeys`

Replaced the rect-AABB `for y for x` enumeration with a single call to
`duplicateToTilesTouched(…, real_keys, real_values)` keyed on the per-Gaussian
`conic_t`.

## Why it didn't work

The conic derivation. The 2DGS `transMat T` does not satisfy
`T·(s.x, s.y, 1) = (px·w, py·w, w)` (surfel→pixel) — it's used by the
render kernel via the cross-product trick:

```
k = px·Tw − Tu
l = py·Tw − Tv
s = cross(k, l) / cross(k, l).z
```

Expanding `cross(k, l)` componentwise (the px·py terms cancel — disk
projects to a conic, not a quartic):

```
cross(k, l).x = (Tv×Tw).x · px + (Tw×Tu).x · py + (Tu×Tv).x
cross(k, l).y = (Tv×Tw).y · px + (Tw×Tu).y · py + (Tu×Tv).y
cross(k, l).z = (Tv×Tw).z · px + (Tw×Tu).z · py + (Tu×Tv).z
```

With `n0 = Tv×Tw`, `n1 = Tw×Tu`, `n2 = Tu×Tv`, the disk
`s.x² + s.y² ≤ k²` becomes `cross.x² + cross.y² − k²·cross.z² ≤ 0`,
quadratic in `(px, py)`. The `px²` coefficient is

```
A = n0.x² + n0.y² − k²·n0.z²
```

NOT `A = n0.x² + n1.x² − k²·n2.x²` (which is what I wrote first by
analogy with the rows-of-`adj(T)` derivation — wrong because
the cross-product expansion groups by component, not by vector).

### Two bug branches we hit

1. **Initial `glm::inverse(T)` formulation** — gave A = -8e9 for
   typical kitchen surfels (wrong sign because T isn't the surfel→pixel
   map I assumed); rejected nearly all surfels → PSNR 5.83 (black bg).

2. **Cross-product formulation with grouping bug**
   (`A = n0.x² + n1.x² − k²·n2.x²`) — same negative A for the same reason,
   same PSNR 5.83. This was a transposition of the correct
   per-component grouping.

3. **Cross-product formulation with correct grouping**
   (`A = n0.x² + n0.y² − k²·n0.z²`) — gives positive A for typical surfels
   (verified A = 2988 for kitchen idx=0 vs the buggy A = -8.46e9).
   But: hangs / OOMs the rasterizer because the resulting bbox extents
   for some surfels disagree with `compute_aabb`'s conventions. Need
   careful comparison vs `compute_aabb` on a synthetic scene to debug.

## What's likely still off

`compute_aabb` returns `extent.y = 7.5` for kitchen idx=0 (the surfel
the project center used in debugging). My conic gives `extent.y = 12.4`
(verified empirically — pixels at `dy = 12` are inside the disk). So
my SnugBox bbox is **bigger** than what the OLD code uses. For some
surfels this difference may blow up (bbox covers most of the screen,
n_tiles ≈ grid_total per Gaussian, total binningBuffer overflows).

## Why beta_scaled doesn't change the picture

AccuTile is geometry-only: it scan-line-walks an ellipse boundary in
screen space. The kernel choice (Gaussian vs bounded beta) only affects
the `cutoff` value we pull back through T to define the surfel-space disk.
For `beta_scaled`, `cutoff = max(r_beta, r_lp)` clamped to `k+2 = 5`. Same
ellipse, same algorithm. So adapting AccuTile to bounded beta is trivial
once the conic-from-T derivation is correct.

## To resume later

1. Build a unit test: synthetic surfel with known transMat T and cutoff,
   compute the projected ellipse via `cross(k, l)/cross.z` ground-truth,
   compare bbox to `compute_conic_from_transmat`.
2. Reconcile with `compute_aabb`'s bbox (which works in production).
3. Once conic is right, rerun on kitchen — should give PSNR == baseline
   and ~30% faster preprocess + sort.

## Other speedups that DID land (and stay)

- FP16-packed `geomState.rgb` (halves the inner-loop fetch bandwidth)
- Persistent `out_color`, `radii`, scratch buffers (no per-frame alloc)
- Empty-tensor singletons cached in Python wrapper
- `prepare_gaussian_inputs()` snapshots post-activation tensors once per scene
- `get_rasterizer()` caches the `nn.Module` wrapper per (H, W, sh_degree)
- CUDA Event timing (replaces wall-clock `time.time()`)
- Dropped `final_T` / `n_contrib` / `out_others` / `clamped` writes
- Dropped `residual_textures` (shared 8×8 path) and 48D-SH residual code
  paths (only the atlas path is supported now)
- Atlas-mode hardware-bilinear texture path is the default; software
  fallback only when atlas exceeds the 65k cudaArray 2D dimension

Net measured kitchen result: PSNR 31.12 (matches neural −0.06 dB), FPS
~308 vs prior ~280 (+10%).
