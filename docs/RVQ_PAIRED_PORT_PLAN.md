# RVQ-paired deployment — CUDA results + WGSL port plan

> # ⚠️ SUPERSEDED — historical reference only
>
> **RVQ was replaced by typeD (BC7-codebook, `atlas_format=7`) in production
> because RVQ's per-fragment SW decode was unusably slow on TBDR mobile GPUs**
> (Adreno / Mali / Apple / PowerVR). Confirmed 2026-07-21 by shipping
> `garden_sh_res_rvq.bitymi` and hitting single-digit FPS on a Snapdragon
> phone while the same bake as typeD ran at normal FPS.
>
> **The RVQ code path was removed from the WebGPU viewer on 2026-07-23**
> (`render_2dgs.wgsl`, `Nat2Parser.ts`, `gaussian-renderer.ts` — no
> `atlas_format == 5` branch survives). The producer-side flag
> `scripts/export_textures_bin.py --rvq-paired` still exists but nothing
> consumes its output; do **NOT** ship bundles produced with it.
>
> This document is kept because the CUDA benchmark numbers, the on-disk
> layout, and the WGSL decode sketch may be useful reference for any
> future compressed-atlas format that wants a shader-decode path. But the
> plan itself is dead — do **NOT** treat any step below as an actionable
> deployment recipe.
>
> **Current production path** — see [`DEPLOY_DEMO.md`](DEPLOY_DEMO.md) and
> [`BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md) (`--bc7-codebook` / typeD).

## Summary of CUDA submodule experiments

Four submodules, each isolating one variant of the RVQ decode path. Bench
config: room scene (75 207 Gausses, 1557×1038 test cameras, 30 frames, 10
warmup, 5090). All metrics are test-set means.

| variant | submodule path | PSNR | LPIPS | FPS | atlas storage |
|---|---|---:|---:|---:|---:|
| BC7 baseline | `submodules/diff_surfel_bake_render` | **30.40** | 0.2402 | **1 152** | 225 MB |
| RVQ global-mem (L=4) | `submodules/diff_surfel_bake_render_rvq` | 30.37 | 0.2454 | 452 | 58 MB |
| RVQ 2D-tex codebook (L=4) | `submodules/diff_surfel_bake_render_rvq_tex` | 30.37 | 0.2454 | 609 | 58 MB |
| **RVQ paired (L=2)** | `submodules/diff_surfel_bake_render_rvq_paired` | **30.37** | 0.2453 | **967** | 64 MB |

**Production target**: RVQ-paired. 1.19× slower than BC7 with 3.5× less GPU
memory and bit-equivalent PSNR. Same bundle size on disk as the L=4 form
(the 8 MB pair codebook replaces 96 KB of L=4 codebooks; indices unchanged).

## RVQ-paired data structures (the bundle format)

### Pair codebook texture
- **2D image, 1024 × 2048 pixels, rgba8unorm** (8 MB).
- Codeword (c_hi, c_lo) of pair `p ∈ {0, 1}` lives at:
  - `u = c_lo * 4 + intra_u`
  - `v = p * 1024 + c_hi * 4 + intra_v`
- where `intra_u, intra_v ∈ [0, 3]`.
- **Sampler**: `cudaFilterModePoint` (CUDA) / `nearest` (WGSL). Hardware-
  bilinear was *slower* due to warp divergence between same-block and
  cross-block paths; software 4-tap is faster.

### Packed indices
- **uint32 per used block**, stored as `array<u32>` of length `N_used`.
- Packing:
  ```
  packed = (pair1_idx << 16) | pair0_idx
  pair_p_idx = (code_stage_2p << 8) | code_stage_2p+1
  ```
- For block `b` of surfel `g`:
  `block_id = surfel_offsets[g] + bv_local * (w/B) + bu_local`

### Per-surfel block offsets
- **int64 array**, length `n_gauss + 1`, cumulative used-block count.
- Pre-computed at load time (one CPU pass over `atlas_rects`).

### Per-pair dequant
- 4 floats: `pair_scale[2]`, `pair_offset[2]`.
- Computed at bake time: per-pair-codebook min/max over the precomputed
  sum codebook (NOT inherited from the atlas's scale/offset).

## Per-fragment decode (CUDA — to be ported to WGSL)

```cuda
// Per atlas-sample point (au, av) of surfel g:
int local_u = (int)floorf(au) - u0i;
int local_v = (int)floorf(av) - v0i;
local_u = clamp(local_u, 0, u_span - 1);
local_v = clamp(local_v, 0, v_span - 1);
int bu = local_u / 4;  int bv = local_v / 4;
int intra_u = local_u - bu * 4;
int intra_v = local_v - bv * 4;
long bid = surfel_offsets[g] + bv * (u_span / 4) + bu;

uint32_t packed = paired_indices[bid];                       // 1 dependent load
uint32_t p0 = packed & 0xFFFFu;
uint32_t p1 = packed >> 16;
int c0_hi = p0 >> 8;  int c0_lo = p0 & 0xFF;
int c1_hi = p1 >> 8;  int c1_lo = p1 & 0xFF;

float4 rgba0 = tex2D<float4>(pair_tex,
    c0_lo * 4 + intra_u + 0.5,
    c0_hi * 4 + intra_v + 0.5);                              // tex2D, point sampling
float4 rgba1 = tex2D<float4>(pair_tex,
    c1_lo * 4 + intra_u + 0.5,
    1024 + c1_hi * 4 + intra_v + 0.5);

float r = rgba0.x * pair_scale[0] + pair_offset[0]
        + rgba1.x * pair_scale[1] + pair_offset[1];
// same for g, b
```

This is invoked 4× per fragment for the bilinear taps, then the fu/fv
weights combine the 4 corner samples in software (cross-codeword bilinear
must happen in software because adjacent codewords aren't spatially
related in the codebook image).

## WGSL port — the actionable plan

### 1. NAT2 format extension (producer)

Add `atlas_format = 5` (RVQ_PAIRED) to `scripts/export_textures_bin.py`.
Payload layout after the existing 64-byte NAT2 header + layer_cuts +
rects:

```
[4 B] sub-magic "RVQP"
[4 B] version  (u32 = 1)
[4 B] K_orig   (u32, typically 256)
[4 B] B        (u32, block size in pixels = 4)
[4 B] N_used   (u32, used blocks total)
[16 B] pair_scale[2], pair_offset[2]  (f32 × 4)
[K*B × 2*K*B × 4 B] codebook image (uint8 RGBA, row-major)
[N_used × 4 B]      packed indices (uint32)
```

Note: this can also be **derived at load time from the existing RVQ
format** (vq/codebooks.pt + vq/indices.pt) — the producer would precompute
the pair codebook by summing pairs of stages, then quantize to uint8 +
compute min/max. So shipping the L=4 form and converting at load time is
equally viable. Pick whichever is simpler for the bundle pipeline.

### 2. TS bundle parser (`Halloumi-WS/src/utils/loaders/ply/Nat2Parser.ts`)

Add the format=5 branch to the existing `parseNat2()`:

```ts
export const ATLAS_FORMAT_RVQ_PAIRED = 5;

// In the type:
interface Nat2RVQPaired {
    format: 5;
    K_orig: number;             // 256
    B: number;                  // 4
    N_used: number;
    pair_scale: Float32Array;   // length 2
    pair_offset: Float32Array;  // length 2
    codebook_bytes: Uint8Array; // (K*B) × (2*K*B) × 4 = 8 MB
    packed_indices: Uint32Array;// length N_used
}
```

Cleanly slice each section from the NAT2 buffer; no decode needed at
parse time (everything ships to GPU as-is).

### 3. GPU resource builder (`ply-loader.ts:buildAtlasResources`)

Add a new branch when `atlas.format === 5`:

```ts
// Codebook 2D texture (rgba8unorm, nearest sampler)
const cb_tex = device.createTexture({
    label: 'rvq paired codebook',
    size: { width: K_orig * B, height: 2 * K_orig * B },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
device.queue.writeTexture(
    { texture: cb_tex },
    codebook_bytes,
    { bytesPerRow: K_orig * B * 4, rowsPerImage: 2 * K_orig * B },
    { width: K_orig * B, height: 2 * K_orig * B },
);

// Packed indices storage buffer
const idx_buf = device.createBuffer({
    label: 'rvq packed indices',
    size: N_used * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(idx_buf, 0, packed_indices.buffer);

// surfel_offsets: precompute from rects at load time
// (cumulative count of (w/B)*(h/B) per surfel)
const surfel_offsets = new BigUint64Array(num_rects + 1);
let acc = 0n;
for (let i = 0; i < num_rects; i++) {
    const w = rects_raw[i*4 + 2], h = rects_raw[i*4 + 3];
    const blocks = (w / B) * (h / B);
    surfel_offsets[i] = acc;
    acc += BigInt(blocks);
}
surfel_offsets[num_rects] = acc;
const off_buf = device.createBuffer({ ... GPUBufferUsage.STORAGE | COPY_DST ... });
device.queue.writeBuffer(off_buf, 0, surfel_offsets.buffer);

// New uniform: pair_scale[2], pair_offset[2] (16 bytes total)
const pair_dq = new ArrayBuffer(16);
new Float32Array(pair_dq)[0] = pair_scale[0]; ...
```

WGSL doesn't have native u64, so `surfel_offsets` becomes `array<vec2<u32>>`
(lo, hi). Same data, just stored as pairs. Or use a single u32 if N_used
fits — at 14.7 M it fits in u32 fine; use u32 directly.

### 4. WGSL shader (`render_2dgs.wgsl`)

Add atlas-sample branch for `atlas_format == 5`:

```wgsl
@group(2) @binding(4) var<storage, read> rvq_packed_indices : array<u32>;
@group(2) @binding(5) var<storage, read> rvq_surfel_offsets : array<u32>;
@group(2) @binding(6) var                rvq_codebook       : texture_2d<f32>;
@group(2) @binding(7) var                rvq_cb_samp        : sampler;

// In tex_params struct (or a new uniform): pair_scale[2], pair_offset[2].

fn sample_rvq_paired(g: u32, au: f32, av: f32, u0: f32, v0: f32,
                     w_span: f32, h_span: f32) -> vec3<f32> {
    let B: i32 = 4;
    let local_u = clamp(i32(floor(au)) - i32(u0), 0, i32(w_span) - 1);
    let local_v = clamp(i32(floor(av)) - i32(v0), 0, i32(h_span) - 1);
    let bu = local_u / B;
    let bv = local_v / B;
    let intra_u = local_u - bu * B;
    let intra_v = local_v - bv * B;
    let bw_g = i32(w_span) / B;
    let bid = rvq_surfel_offsets[g] + u32(bv) * u32(bw_g) + u32(bu);

    let packed = rvq_packed_indices[bid];
    let p0 = packed & 0xFFFFu;
    let p1 = packed >> 16u;
    let c0_hi = i32(p0 >> 8u);  let c0_lo = i32(p0 & 0xFFu);
    let c1_hi = i32(p1 >> 8u);  let c1_lo = i32(p1 & 0xFFu);

    let K_B = K_ORIG * B;            // 1024 at K=256 (compile-time const)
    let uv0 = vec2<f32>(
        (f32(c0_lo * B + intra_u) + 0.5) / f32(W_TEX),
        (f32(c0_hi * B + intra_v) + 0.5) / f32(H_TEX));
    let uv1 = vec2<f32>(
        (f32(c1_lo * B + intra_u) + 0.5) / f32(W_TEX),
        (f32(K_B + c1_hi * B + intra_v) + 0.5) / f32(H_TEX));

    let rgba0 = textureSampleLevel(rvq_codebook, rvq_cb_samp, uv0, 0.0);
    let rgba1 = textureSampleLevel(rvq_codebook, rvq_cb_samp, uv1, 0.0);

    let s0 = pair_dq.scale.x;  let o0 = pair_dq.offset.x;
    let s1 = pair_dq.scale.y;  let o1 = pair_dq.offset.y;
    return rgba0.rgb * s0 + vec3<f32>(o0)
         + rgba1.rgb * s1 + vec3<f32>(o1);
}
```

Then in the fragment shader's atlas-residual block (currently lines
~238–256 of `render_2dgs.wgsl`):

```wgsl
if tex_params.atlas_format == ATLAS_FORMAT_RVQ_PAIRED {
    // Software 4-tap bilinear over the rvq paired decode (cross-codeword
    // taps live in unrelated codewords, so hw bilinear within the codebook
    // texture is wrong; bilinear must be in software at the atlas level).
    let au0 = floor(au); let av0 = floor(av);
    let fu = au - au0; let fv = av - av0;
    let c00 = sample_rvq_paired(gauss_id, au0,     av0,     u0, v0_local, w_span, h_span);
    let c01 = sample_rvq_paired(gauss_id, au0+1.0, av0,     u0, v0_local, w_span, h_span);
    let c10 = sample_rvq_paired(gauss_id, au0,     av0+1.0, u0, v0_local, w_span, h_span);
    let c11 = sample_rvq_paired(gauss_id, au0+1.0, av0+1.0, u0, v0_local, w_span, h_span);
    let top = mix(c00, c01, fu);
    let bot = mix(c10, c11, fu);
    residual = mix(top, bot, fv);
} else if tex_params.atlas_format == ATLAS_FORMAT_BC7 || ATLAS_FORMAT_ASTC {
    // existing path
    let rgba = textureSampleLevel(atlas, atlas_samp, uv, layer, 0.0);
    residual = rgba.rgb * atlas_scale + vec3<f32>(atlas_offset);
}
```

`K_ORIG`, `W_TEX`, `H_TEX`, `B` are compile-time constants (overrides at
pipeline-create time, or written into the shader source via template).

### 5. Bind group layout updates

Existing `@group(2)` has 4 bindings (atlas, atlas_rects, atlas_samp, tex_params).
Add 4 more for the RVQ path. The whole group needs the same number of
bindings for all bundles, even when only one path is active — use stub
empty buffers for the inactive bindings.

### 6. Deploy

- Add `room_rvqply.bitymi` to `bitymi-demos/scenes/mip_360/` (already
  built from the producer-side work earlier — file exists at
  `bitymi-demos/scenes/mip_360/room_rvqply.bitymi` (65.6 MB).
- Re-encode it with the paired atlas_format=5 once the WGSL viewer is
  ready and tested.
- Add an `<a class="demo">` card to `bitymi-demos/index.html`:
  ```html
  <a class="demo" data-scene="room"
     data-hd-mb="66"
     href="viewer/index.html?bundle=https://huggingface.co/datasets/Nilkel/bitymi-demos/resolve/main/mip_360/room_rvqply.bitymi">
    Room (RVQply) — 66 MB
  </a>
  ```
- Build the TS viewer (`cd /home/nilkel/Projects/Halloumi-WS && npm run build`)
  and copy `dist/` into `/home/nilkel/Projects/bitymi-demos/viewer/`.
- Commit + push bitymi-demos repo.
- Upload `room_rvqply.bitymi` to Hugging Face dataset
  `Nilkel/bitymi-demos`.

## Notes for the porter

- **Bind-group consistency**: WebGPU requires all bindings in a group to
  exist at pipeline-create time, even if unused. Either use stub buffers
  for the BC7 path when binding the RVQ path's stuff, or maintain two
  pipelines (one per atlas format).
- **u64 surfel_offsets**: at N_used ≤ 16 M (14.7 M for room), `u32` is
  sufficient. Keep the buffer as `array<u32>`.
- **K_orig assumption**: code hard-codes 256. If we ever want different K,
  thread it through as a uniform. For now compile-time is fine.
- **Storage buffer size limits**: WebGPU default max storage buffer is
  128 MB. 56 MB indices fits with margin.
- **Bilinear policy**: software 4-tap at the atlas level, point sampler
  on the codebook texture. The CUDA bench confirmed hw-bilinear within the
  codebook is *slower* (warp divergence + sampler tax).

## File index

- `submodules/diff_surfel_bake_render_rvq_paired/` — production-shape
  CUDA reference; WGSL port follows its data layout exactly.
- `scripts/bench_rvq_paired_submodule.py` — bench harness (used to
  confirm 967 fps).
- `bitymi-demos/scenes/mip_360/room_rvqply.bitymi` — 65.6 MB bundle
  (currently in the L=4 RVQ format; to be re-emitted as atlas_format=5
  paired form once the WGSL viewer is ready).
- `docs/RVQ_SHADER_DECODE.md` — earlier analysis of the L=4 form.
- `docs/VQ_BAKE.md` — Phase 1 codebook-training design.
- `docs/ATLAS_DECOMPOSITION_AND_CLUSTERING.md` — RVQ K-means analysis.
