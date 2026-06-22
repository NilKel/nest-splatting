# Baked Rendering — Atlas Pipeline

End-to-end baked-rendering pipeline for `3D_SH_res` models: a static SH PLY plus a
per-Gaussian residual atlas, sampled via hardware bilinear at inference. **6–9×
faster than the neural renderer at near-identical quality.**

> **Method coverage**: `3D_SH_res`, `3D_SH_res_sep`, `res_switch`, `res_3d`,
> `res_3d_paired`, `res_3d_double`, `mixed_3d`, `mixed_3d_sep`, `3D_SH_cat`.
> For the staged-curriculum / 2D-3D-split modes see [`RES_3D_MODES.md`](RES_3D_MODES.md).
> `--method res_3d_paired` is routed through a dedicated bake-render submodule
> `diff_surfel_bake_render_paired` (functionally a clone of
> `diff_surfel_bake_render`; built separately so paired-specific kernel
> tweaks don't touch the shared submodule). `benchmark_baked.py` does this
> aliasing automatically via `sys.modules` when it detects
> `args.method == "res_3d_paired"` in `args.pkl`.

## Key entry point

```bash
conda run -n nest_splatting python scripts/benchmark_baked.py \
    --model_path <model_path> \
    --max_res 64 --atlas_budget_mb 8192 \
    --aabb_mode 5 --sort_mode 0 \
    --bake_dtype bc7
```

Outputs land in `<model_path>/baked_atlas/` (or `--output_dir`):
`baked.ply`, `atlas_texture.bc7`, `atlas_rects.pt`, `bake_meta.json`,
optional `sb_params.pt`.

## Atlas dtypes (`--bake_dtype`)

| Mode | Bake-time CPU RAM | Disk + VRAM | Quality | Use when |
|---|---|---|---|---|
| `fp16` (legacy) | 1× (≈ atlas_h·atlas_w·6 B) | 1× | reference | back-compat with old artifacts |
| `uint8` | **½×** | ½× | bit-identical to fp16 | dev iteration, no encode step |
| `bc7` | ½× (RAM still uint8) | **⅙×** | bit-identical to fp16 | **production / deployment default** |

Implementation: `bc7` first builds a uint8 atlas (per-chunk quantization with a
pre-sampled ±6σ range, see `precompute_atlas_quant_range`) then encodes the
whole atlas to BC7 via `bc7encoder` (vendored richgel999/bc7enc, 8 cores
parallel, ~5 ms/MB). Saved as `atlas_texture.bc7` next to the legacy
`atlas_texture.pt` (uint8). Runtime: a CUDA `cudaArray` with
`cudaChannelFormatKindUnsignedBlockCompressed7` + hardware-bilinear `tex2D`.

## Headline results (mip-360, 18 scene/config pairs, `--max_res 64 --atlas_budget_mb 8192`)

|  | PSNR | SSIM | LPIPS | FPS | Atlas (MB, mean) |
|---|---|---|---|---|---|
| Neural (pre-bake) | 26.81 | 0.7811 | 0.2307 | 67.2 | — |
| Baked FP16 (prior) | 26.75 | 0.7764 | 0.2485 | 430.4 | ~3 GB / scene |
| **Baked BC7 (current)** | **26.76** | **0.7762** | **0.2487** | **472.8** | **638** |

- **BC7 vs FP16**: PSNR Δ +0.01 dB, SSIM Δ −0.0002, LPIPS Δ +0.0002 — bit-identical
  within numerical noise. **+10% FPS** (smaller working set → better L2 hit rate).
  **5–6× smaller storage / VRAM** (e.g., bicycle 0w0g: 7.7 GB FP16 → 1.28 GB BC7).
- **Bake quality cost vs neural**: PSNR Δ −0.05 dB, LPIPS Δ +0.018 — imperceptible.
- **Speedup vs neural**: **7.04×**.

Atlas size range across the 18 pairs (BC7): 226 MB (counter 005w25) →
1285 MB (bicycle 0w0g).

## AABB modes (`--aabb_mode`)

`5` is the default and uses **SnugBox + AccuTile** (FastGS-style ellipse-tight
tile binning). `2` is the legacy rect AABB. Mode 5 gives **+5–14% FPS** at
bit-identical PSNR across all 18 mip-360 scene/configs vs mode 2. See the
no-FMA-fusion fix in `auxiliary.h:processTiles` for why count and emit phases
must use `__fmul_rn`/`__fadd_rn` (else: silent count↔emit drift → OOB writes).

## Sort modes (`--sort_mode`)

| Mode | Description | When to use |
|---|---|---|
| `0` (default) | Legacy 64-bit composite sort `(tile<<32 | depth_bits)` | Our 18 scenes (200–500k Gaussians) |
| `1` | FastGS two-stage: 32-bit depth sort on n_visible + 32-bit tile sort on n_instances; stable to preserve depth within tile | Multi-million Gaussian scenes where sort cost dominates |

For our workload, mode 1 is **−15 to −23% slower** because compaction overhead
(atomic counters, scattered reads in `apply_depth_ordering`, +5 kernel launches,
+2 sync points) outweighs the cheaper 2-stage sort. The flag is wired through
for future testing on bigger scenes.

## Atlas-width auto-grow

`shelf_pack_atlas` auto-grows `atlas_width` to keep `atlas_height ≤ 60000`
(under CUDA's 65536 cudaArray 2D dim limit). Required for full-Nyquist BC7 on
dense outdoor scenes (bicycle/treehill/garden 0w0g would otherwise OOM at
~328k atlas rows with default `--atlas_width 4096`).

## Resolution sizing

- `--max_res 64` (default): per-Gaussian atlas resolution snapped to power-of-2
  in `[min_res=4, max_res=64]`.
- `compute_adaptive_resolution`: hashgrid Nyquist (encoding bandwidth).
  Surfels with scale > 0.1 saturate at 64 — they're undersampled at the cap. See
  `--max_res 128` for the next axis of improvement (+0.05–0.2 dB PSNR, ~2× atlas size).
- `--view_aware_res`: walks all training views, computes max projected (w_px, h_px)
  per Gaussian, takes `min(view_nyquist, hash_nyquist)` per axis. Modest savings
  (~2% on bicycle, 0% on kitchen) — most surfels saturate at the cap either way.

## Importance-based pruning / skip-texture (optional)

Reuses GSpa's `compute_importance_scores` (alpha·T accumulated over all train
views) to drop or strip the atlas of low-contributors:

| Flag | Behavior |
|---|---|
| `--bake_prune_low_contrib N` | drop bottom N fraction of Gaussians |
| `--bake_skip_texture_low_contrib M` | mark bottom M as zero-rect (SH-only) |
| `--bake_prune_thresh T` | drop Gaussians with importance ≤ T (absolute) |
| `--bake_skip_texture_thresh T` | zero-rect Gaussians with importance ≤ T |
| `--bake_budget_mode importance` | spend atlas budget on highest-importance first; tail gets reduced res / skip-texture |

Importance scores are typically heavy-tailed: median ~860, max ~3.7e5 for
counter — thresholds need to be at the percentile of the actual distribution,
not nominal values like 0.001. Use `--bake_prune_thresh 0.0001` to catch
literally-never-visible surfels (~0.1% of Gaussians).

## Pitfalls (all bitten)

- **`Scene()` overwrites baked PLY**: the constructor loads training PLY.
  Always `gaussians.load_ply(baked_ply)` after `Scene()` init.
- **UV conventions**: bake kernel uses texel-center `(i+0.5)*step − extent`,
  NOT endpoint-inclusive `/(N−1)`.
- **SH layout**: MLP outputs channel-first `[R0..R15, G0..G15, B0..B15]`; PLY
  stores interleaved `[N, 16, 3]`.
- **FMA fusion in AccuTile**: `processTiles`/`computeEllipseIntersection` MUST
  use `__fmul_rn`/`__fadd_rn`/`__fdiv_rn` to keep count and emit phases
  bit-identical. Otherwise silent OOB → `cudaErrorIllegalAddress`.
- **cudaArray 2D dim cap**: 65536 in either dim. Auto-grow atlas_width handles
  this; without it, BC7 cudaMallocArray fails on dense outdoor scenes and
  corrupts the CUDA context for subsequent torch ops.
- **bake_meta carries the dequant range**: `atlas_offset`, `atlas_scale` written
  during uint8/BC7 bake. The runtime reads these instead of recomputing from
  the FP16 atlas.

## Open improvements

1. **`--max_res 128`**: would close most of the LPIPS gap to neural for outdoor
   scenes with large Gaussians (scale > 0.1). Atlas cost ~2–4× current, still
   feasible with BC7. (See `docs/BAKED_RENDERING_128.md` for results.)
2. **True per-chunk BC7 streaming**: current path holds a uint8 atlas in CPU
   RAM, then bulk-encodes. Phase 3 would emit BC7 blocks per-Gaussian during
   bake. Requires 4-aligned shelf packing.
3. **VQ palettization** (per-Gaussian patch codebook): theoretically deeper
   compression but residual textures aren't tile-repetitive, so likely beaten
   by BC7 at much higher engineering cost. See conversation history.
