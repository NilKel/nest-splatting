# RVQ shader-side atlas decode — feasibility benchmark

The "RVQply" deployment format (see [`VQ_BAKE.md`](VQ_BAKE.md) §9 +
[`ATLAS_DECOMPOSITION_AND_CLUSTERING.md`](ATLAS_DECOMPOSITION_AND_CLUSTERING.md)
§4.5) ships a baked scene as `8-bit PLY + RVQ codebook + indices`,
3.7× smaller bundle vs current BC7 (65 MB vs 240 MB for the room scene).

The question this doc answers: **can the GPU shader decode RVQ
per-fragment fast enough to keep the existing 60+ fps real-time
budget?**

## TL;DR

**Yes.** On a 5090 the per-fragment CUDA cost for RVQ-bilinear
(L=4 K=256, the deployed config) is **0.62 ms / 1920×1080 frame
= 1 615 fps headroom**. That's ~1.8× the cost of a no-cache-helper
FP16-atlas bilinear baseline, comfortably above 60 fps.

GPU memory at decode-time drops from **225 MB (BC7 atlas)** to
**58 MB (RVQ buffers)** — same scene, 3.9× less VRAM.

The corresponding WGSL fragment-shader port should keep these
characteristics: the codebook (96 KB at L=4 K=256) fits in any
modern GPU's L1 cache and is shared across all fragments; only the
indices stream sees DRAM bandwidth, and it's far smaller than the
BC7 atlas.

## Methodology

`scripts/bench_rvq_shader_decode_cuda.py` JIT-compiles
`scripts/_rvq_decode_cuda.cu` and benchmarks three CUDA kernels on
2 073 600 synthetic fragments (one Full-HD frame). Fragment
positions are randomly sampled from used surfel rects, weighted by
surfel area so the distribution matches a real frame.

The three kernels:

1. **`baseline_bilinear`** — 4-tap bilinear from a pre-dequantized
   FP16 atlas tensor (1.35 GB on GPU). Proxy for the production BC7
   path *without* the hardware texture-cache assist; real BC7-tex2D
   in the rasterizer is ~3–5× faster than this on a 5090 (cache
   hits + dedicated sampler unit).

2. **`rvq_decode_nearest`** — RVQ with nearest-block sample. For
   each fragment: compute its surfel-local UV, derive a block-id
   from `surfel_offsets[g] + bv_local·(w/B) + bu_local`, do L
   codebook lookups, sum.

3. **`rvq_decode_bilinear`** — same but 4 block lookups per
   fragment (the 4 taps of a true bilinear, each with its own
   block-id derivation + L codebook reads).

Codebook stored FP16 (96 KB total at L=4 K=256). Indices uint8
since K ≤ 256.

### One-time CPU prep (load-time, not per-frame)

The K-means RVQ producer (`scripts/bake_cluster_blocks_residual.py`
+ `scripts/vq_bake.py`) writes indices in **atlas-row-major order**
(the iteration order of `used.nonzero()`). The shader wants
**surfel-major order** so it can compute `block_id` from `(g,
local_uv)` without a (bv, bu) → block_id LUT (which would be ~56 MB
at room scale — same size as the indices, doubling GPU memory).

`bench_rvq_shader_decode_cuda.py` does the row-major →
surfel-major permutation at load time on the CPU (one-time ~13 s
for the room scene; trivially parallelisable). Production should
move this reorder into the producer (`bake_cluster_blocks_residual.py`)
so loaded artifacts are already in shader-friendly order.

## Results (5090, room scene, L=4 K=256, 2.07 M frag/frame)

| variant | ms/frame | fps | vs baseline |
|---|---:|---:|---:|
| Baseline FP16 atlas bilinear | 0.338 | 2 962 | 1.00× |
| **RVQ nearest (4 codeword reads)** | **0.209** | **4 778** | 0.62× (faster) |
| **RVQ bilinear (16 codeword reads)** | 0.619 | 1 615 | 1.83× |

Sanity check: RVQ-bilinear vs baseline-bilinear PSNR = **43.76 dB**
on the synthetic fragment set, matching the atlas-fidelity
measurement (42.49 dB on the full atlas, the slight difference
from the sample distribution).

GPU memory comparison:
| component | BC7 path | RVQ path |
|---|---:|---:|
| Codebooks (FP16) | — | 96 KB |
| Indices (uint8) | — | 56.2 MB |
| Surfel offsets (int64) | — | 587 KB |
| Rects (int32) | 1.2 MB | 1.2 MB |
| Atlas texture | 225 MB | — |
| **Total** | **226.2 MB** | **58.1 MB** |

## Why RVQ-nearest is *faster* than the baseline

Counter-intuitive at first. Three reasons:

1. **Codebook fits in L1.** 96 KB at L=4 K=256 sits in every modern
   GPU's L1 data cache (sized 128 KB on a 5090 SM). The codebook is
   shared across all fragments — first ~100 fragments load it into
   cache, the next millions hit cache exclusively. The baseline FP16
   atlas (1.35 GB) does NOT fit in any cache; every fragment does a
   DRAM round-trip.

2. **Single lookup per stage.** RVQ-nearest is 4 codeword reads per
   fragment (12 bytes per codeword). Bilinear baseline is 4 texel
   reads per fragment (6 bytes per texel × 4 = 24 bytes). With the
   codebook cache-resident, RVQ wins.

3. **Pre-dequantization done at compile-time.** Baseline pays a
   `__half2float` per channel per fragment. RVQ pays the same
   conversion but on cache-hot codewords. Same FLOP cost; the
   cache-hit savings dominate.

For RVQ-**bilinear**, we pay 4× the codebook reads (one per
sub-pixel tap), and now we DO start hitting bandwidth limits. The
1.83× slowdown vs baseline is reasonable.

## When the production BC7 path will still be faster

The numbers above use a **no-cache-helper** FP16 baseline (just a
strided gather from a large FP16 tensor). The production BC7 path
uses hardware `tex2D<float4>` which:

- Has a dedicated texture-cache unit, not just L1.
- Reads BC7 blocks (16 bytes each, 4×4 texels) — natural locality
  matches the access pattern.
- Decodes BC7 in fixed-function hardware (one cycle, no software
  reconstruction).

Realistic estimate: production BC7-tex2D is 3–5× faster than our
no-cache-helper FP16 baseline. So RVQ-bilinear is ~3.5–9× slower
than real BC7 at the *fragment level*.

That sounds bad until you note: the room scene with BC7 currently
runs at 1 193 fps test bench (measured at
`bench_vq_atlas.log`). RVQ-bilinear's per-fragment slowdown means
the same scene would land at roughly 1 193 / 5 = ~240 fps. Still
4× the 60 fps target, well above any real workload.

For mobile GPUs the trade flips: BC7 isn't supported on Adreno/Mali
(they use ASTC), and the 225 MB atlas doesn't fit on mid-range
phones. RVQ's 58 MB footprint and shader-only decode is the
deployable form.

## What this means for the WGSL viewer port

The WGSL fragment shader needs:

1. **Two new storage buffers** bound to the existing atlas
   resource group: `codebooks` (FP16) and `indices` (uint8 packed
   into u32 — WGSL doesn't have u8 storage).

2. **A new function `sample_atlas_rvq(g_id, uv) -> vec3<f32>`**
   that does the per-fragment work above. The existing
   `sample_atlas_bc7` keeps working for old bundles.

3. **A format flag in the atlas-resource builder** — when NAT2
   reports `atlas_format == 4`, allocate the RVQ buffers and bind
   the RVQ variant of the sample function. The existing BC7/ASTC
   paths are untouched.

4. **At load time**: do the row-major → surfel-major reorder
   (or have the producer do it). Build the `surfel_offsets` table.

5. **No new GPU texture** — RVQ atlas lives entirely in storage
   buffers. The atlas-texture bind in the shader becomes a
   conditional: if BC7/ASTC bundle, use the texture sampler; if
   RVQ bundle, use the buffer-decode function.

WGSL doesn't have u8 storage types, so `indices` becomes
`array<u32>` with 4 indices packed per u32. The decode is `u8 =
(indices[i/4] >> ((i%4)*8)) & 0xFF`. Trivial.

## File index

- `scripts/_rvq_decode_cuda.cu` — the CUDA kernel itself (~150
  lines). Mirror of the WGSL we'll write.
- `scripts/bench_rvq_shader_decode_cuda.py` — JIT compile + bench
  driver.
- `scripts/bench_rvq_shader_decode.py` — earlier PyTorch-only
  reference; slower due to Python overhead but useful for sanity
  cross-checks of the row-major→surfel-major remap.
- `docs/VQ_BAKE.md` — the producer side (codebook training + bake).
- `docs/ATLAS_DECOMPOSITION_AND_CLUSTERING.md` §4.5 — RVQ design,
  storage analysis, choice of L=4 K=256.
