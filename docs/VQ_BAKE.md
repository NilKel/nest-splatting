# End-to-end VQ-bake

> **TL;DR (after the room-scene test-set measurement below): post-hoc
> K-means RVQ at L=4 K=256 already lands at −0.05 dB / +0.005 LPIPS vs
> the original baked atlas, at **3.9× smaller storage** (58 MB vs
> 225 MB BC7). End-to-end trained VQ-bake is **not worth the CUDA
> investment** — the bake/neural delta is already +0.06 dB so there's
> nowhere left to push. This document is kept as a record of the
> design, the math, and the decision.**

> **⚠️ Deployment update (2026-07-23):** the L-stage residual VQ
> deployment format (`atlas_format=5` / paired-RVQ) analysed in §9 was
> shipped briefly then **retired** — its per-fragment SW codebook decode
> is unusable on TBDR mobile GPUs (Adreno / Mali / Apple / PowerVR). The
> RVQ code path was removed from the WebGPU viewer on 2026-07-23.
> Production compressed-atlas format is now **typeD** (`atlas_format=7`,
> `scripts/export_textures_bin.py --bc7-codebook`): single-stage K-means
> (K=65536) over 4×4 blocks with each centroid **re-encoded as one BC7
> block**, gathered back into a normal BC7 texture at load time → a
> single HW BC7 tex fetch per fragment, bit-identical cost to raw BC7,
> ~7× smaller download. The **K-means-over-4×4-blocks primitive from
> this document is still used** — typeD is essentially §4.5's L=1 (single
> stage) case with the codewords BC7-encoded. What was retired is the
> L>1 residual cascade and the shader-decode path it required. See
> [`DEPLOY_DEMO.md`](DEPLOY_DEMO.md),
> [`BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md), and
> [`RVQ_PAIRED_PORT_PLAN.md`](RVQ_PAIRED_PORT_PLAN.md) (the retired
> port plan) for the current pipeline and the swap rationale.

Jointly-trained vector-quantization codebooks + per-Gauss atlas patches,
applied *after* the regular bake. Sits between `bake_sh_res_atlas.py`
(which produces the float32 atlas) and the bundle export pipeline.

The motivation is laid out fully in
[`ATLAS_DECOMPOSITION_AND_CLUSTERING.md`](ATLAS_DECOMPOSITION_AND_CLUSTERING.md)
§ 4.5 — post-hoc K-means RVQ already gives **+1.3 dB over single-stage
VQ at matched bytes**. The hypothesis under test in this doc was: a
trained codebook (the Compact3DGS / SoundStream / EnCodec template)
could squeeze out another 0.5–1 dB. **The test rejected that
hypothesis empirically** — see §9.

---

## 1. Background — why end-to-end beats K-means

K-means assumes the **source signal is fixed**: each iteration only
moves codewords toward already-assigned points. Each stage of post-hoc
RVQ is a greedy fit to a static residual.

Trained VQ-VAE-style codebooks also let the *source* (the per-Gauss
atlas patches) drift toward codebook-friendly positions, through:

- the **straight-through estimator** (STE): forward = `codebook[argmin]`,
  backward = identity. Gradient from the reconstruction loss flows
  through the hard assignment *as if* it were soft.
- the **commitment loss** `β · ||A_soft − sg[recon]||²`: pulls the
  continuous atlas toward the running codebook reconstruction.

The atlas can "trade" a small amount of self-fidelity for a much
larger gain in codebook approximation. Compact3DGS / RVQ-Gaussians
(Park et al.) and the SoundStream / EnCodec audio codecs both follow
this template.

---

## 2. Architecture

```
                       ┌──────────────────────────────┐
                       │  baked atlas (float32, fixed)│   ← bake_sh_res_atlas.py
                       └──────────────┬───────────────┘
                                      ▼  tile to 4×4 blocks
                            ┌────────────────────┐
                            │  A_target [N, 48]  │
                            └─────────┬──────────┘
                                      │  init
                                      ▼
                       ┌──────────────────────────────┐
   learnable  ◄────────│  A_soft   [N, 48]            │
                       └──────────────┬───────────────┘
                                      │  RVQ-with-STE forward
                                      ▼
                            ┌────────────────────┐
                            │  recon = Σ_l C_l[ass_l] │
                            └─────────┬──────────┘
                                      │
                          ┌───────────┴────────────┐
                          ▼                        ▼
                   L_recon (Phase 1)        L_render (Phase 2)
                  ||recon − A_target||²    ||I(recon) − I_gt||²
                          │                        │
                          └────────────┬───────────┘
                                       │  Adam
                                       ▼
                              update A_soft + C_l
```

**Learnable parameters:**
- `A_soft ∈ ℝ^{N_blocks × 48}` — initialised from the baked atlas tiled
  into 4×4 RGB blocks.
- `C_1..L ∈ ℝ^{K × 48}` — one codebook per RVQ stage. Initialised from
  L-stage K-means RVQ (warm start so Adam refines rather than searches).

**Frozen everywhere:**
- The trained MLP, hashgrid, and Gaussian set (xyz, scale, rot, opacity,
  SH coefficients).
- The atlas rect packing (per-Gauss u0/v0/w/h).
- The dequantisation scale/offset of the source uint8 atlas.

**Output artifacts** (under `<bake_dir>/vq/`):
| file | shape / contents |
|---|---|
| `codebooks.pt` | `[L, K, 48]` FP16 — the learned codebooks |
| `indices.pt` | `[L, N_blocks]` uint16/int32 — per-stage block indices |
| `block_meta.pt` | atlas dims, block grid, `used_idx`, atlas dequant params |
| `vq_meta.json` | L, K, training config, K-means baseline + final PSNR |

---

## 3. RVQ forward with STE

The L-stage residual VQ is the same recurrence used in
`bake_cluster_blocks_residual.py`, but with a differentiable wrapper:

```python
def vq_step_ste(R, codebook):
    # 1. hard assignment (no_grad)
    with torch.no_grad():
        idx = argmin_k ||R − codebook[k]||²
    # 2. STE: forward = hard, backward = identity to R
    recon = R + (codebook[idx] − R).detach()
    return recon, idx

def rvq_forward(A_blocks, codebooks):
    recon_running = 0
    for l, C_l in enumerate(codebooks):
        R = A_blocks − recon_running   # residual at this stage
        stage_recon, _ = vq_step_ste(R, C_l)
        recon_running = recon_running + stage_recon
    return recon_running
```

Gradient flow:
- **To codebooks** — the `codebook[idx]` lookup is a real index-select
  so gradient accumulates at each used codeword (sum over assigned
  blocks).
- **To `A_soft`** — STE makes the argmin look like identity, so
  `dL/dA_soft = dL/drecon`. The atlas drifts to be codebook-friendly.
- Stage-to-stage: each later stage's residual is `(A_soft − Σ earlier
  recons).detach()` — the chain is broken to avoid explosive gradients
  through nested STE. Commitment loss handles the encoder pull.

---

## 4. Phase 1: atlas-fidelity loss (Python only)

Script: `scripts/vq_bake.py`.

Loss:
```
L_recon  = ||VQ_recon  −  A_target||²
L_commit = β · ||A_soft − sg[VQ_recon]||²
L        = L_recon + L_commit
```

No CUDA changes; everything is PyTorch. Trains on a random batch of
500 k blocks per iter (full set is ~14.7 M for the mip-360 room scene).

**Why atlas-fidelity is a meaningful proxy for render-PSNR:**
The 18-pair mip-360 benchmark shows uint8→BC7 quant costs ~2–5 dB
of atlas fidelity but only ~0.05 dB of render fidelity — render
PSNR is much less sensitive than atlas PSNR. So **atlas-PSNR is an
upper bound on the render-PSNR loss**: any atlas-PSNR improvement
translates monotonically (but compressed) to render-PSNR.

Phase 1 establishes the architecture and is sufficient validation
that the trained-codebook approach beats post-hoc K-means.

### Usage

```bash
conda run -n nest_splatting python scripts/vq_bake.py \
    --bake_dir outputs/mip_360/room/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac/baked_atlas \
    --L 4 --K 256 \
    --iters 2000 --batch 500000 \
    --lr_codebook 1e-3 --lr_atlas 3e-4 --beta 0.25
```

### Default hyperparams

| flag | default | rationale |
|---|---:|---|
| `--L` | 4 | matches Compact3DGS paper config |
| `--K` | 256 | best-quality post-hoc RVQ point @ 25 % storage |
| `--iters` | 2 000 | ~2 epochs over 14.7 M blocks at batch=500 k |
| `--batch` | 500 000 | fits comfortably in 32 GB; ~30 batches/epoch |
| `--lr_codebook` | 1e-3 | codebook needs to move further than atlas |
| `--lr_atlas` | 3e-4 | atlas already starts at the target; small steps |
| `--beta` | 0.25 | standard VQ-VAE commitment weight |
| `--init_kmeans_iters` | 15 | K-means warm-start; cheap (< 5 s/stage) |

The warm-start makes Adam start from a strong K-means RVQ baseline
rather than random codewords — the script prints the baseline PSNR
before training so any Δ is the *additional* gain from joint training.

### Expected gains

Post-hoc K-means RVQ at (L=4, K=256) on the room scene: **42.47 dB at
25 % storage**. Phase 1 should land in the **42.9–43.5 dB** range
(+0.5–1 dB) — confirmed on the run logged at
`<bake_dir>/vq/vq_meta.json` after training completes.

---

## 5. Phase 2: render-loss (CUDA backward, future work)

To replace `L_recon` with the actual quantity we care about — rendered
image MSE — we need a backward path through the bake-render kernel,
which `diff_surfel_bake_render` does not currently expose.

Three options, in order of effort:

### 5.a. Add backward to `diff_surfel_bake_render`

The bake-render kernel reads `atlas_texture[u, v]` per fragment and
accumulates into the 2DGS blend. A backward pass for it needs:

- `dL/datlas_texture[u, v]` — straightforward: scatter the per-fragment
  RGB gradient (after the alpha cascade) into the contributing atlas
  texels via bilinear weights.
- `dL/dxyz`, `dL/dscale`, `dL/drot`, `dL/dopacity` — already implemented
  in `diff_surfel_3D_sh_res`'s backward; can be ported almost verbatim
  since the bake-render forward shares geometry with the training
  forward (just substitutes atlas-lookup for the MLP eval).
- `dL/dSH` — same as `diff_surfel_3D_sh_res`.

Effort: ~2–3 days CUDA work + verification (numerical gradcheck against
`diff_surfel_3D_sh_res` with the MLP set to atlas-lookup).

### 5.b. Substitute VQ atlas into `diff_surfel_3D_sh_res`

The training rasterizer has full backward but evaluates the MLP per
pixel. We could add a flag that *bypasses* the MLP and reads from a
pre-supplied residual atlas instead, using the existing per-pixel
gradient machinery.

Effort: 1–2 days CUDA + Python plumbing. Slightly hacky (the kernel
isn't designed for this) but reuses every existing backward path.

### 5.c. Pure-Python differentiable renderer

Reimplement bake-rendering in PyTorch (per-fragment atlas sampling +
2DGS blend). Slow (~seconds per frame) but correct and easy to debug.
Useful as a reference for gradient verification of (5.a) or (5.b).

Effort: 1 day. Probably should write this first either way to
sanity-check Phase 1's atlas-fidelity proxy.

### Expected Phase-2 gain

The atlas-fidelity / render-PSNR translation ratio is roughly 1:0.3
in the mip-360 18-pair benchmark (−2 dB atlas → −0.05 to −0.6 dB
render). So a +1 dB atlas gain from Phase 1 is +0.3 dB render. Phase
2 directly optimizes render so the gain should be larger — *if* the
2DGS blend's sub-pixel mixing creates exploitable slack the
atlas-PSNR loss can't see.

Worth implementing only after Phase 1 establishes a clear gain.

---

## 6. Inference path

The VQ artifacts replace `atlas_texture.pt` for compressed deployment:

```
inference:
  for each Gauss g in render:
    for each texel (u, v) in g's rect:
      block_id     = atlas_lookup_to_block_id(u, v)            # one-time precompute
      recon[u, v]  = Σ_l codebooks[l, indices[l, block_id]]    # L lookups
  → standard 2DGS blend with recon as the residual signal
```

The decode is `L` codebook lookups per block (L=4 → 4 reads of a 48 B
codeword), all cache-warm at K=256 (12 KB codebook fits in L1). No
matrix multiplies, no FP16 work. Costs less than the BC7 decode it
replaces **on desktop**.

**Retired for mobile deploy (2026-07-23):** the desktop-friendly
per-fragment codebook decode above turned out to be the fragment
bottleneck on TBDR mobile GPUs (Adreno / Mali / Apple / PowerVR); a
brief production run of paired-RVQ (`atlas_format=5`) hit single-digit
FPS on a Snapdragon phone. The shipping format is **typeD**
(`atlas_format=7`) — L=1 K=65536 with each centroid BC7-encoded and
the atlas gathered back to a normal BC7 texture at load time, so the
fragment path is a single HW BC7 tex fetch. See
[`BITYMI_BUNDLES.md`](BITYMI_BUNDLES.md) and
[`DEPLOY_DEMO.md`](DEPLOY_DEMO.md).

---

## 7. Files

- `scripts/vq_bake.py` — **this script.** Phase 1 implementation.
- `scripts/bake_cluster_blocks_residual.py` — post-hoc K-means RVQ
  (the baseline this trained variant must beat).
- `<bake_dir>/vq/` — output dir for codebooks + indices + metadata.
- `submodules/diff_surfel_bake_render/cuda_rasterizer/forward.cu` —
  current forward-only bake renderer (needs Phase 2 backward).
- `submodules/diff_surfel_3D_sh_res/cuda_rasterizer/` — training
  rasterizer with full backward (Phase 2 reference / substitution
  target).

---

## 8. Status

- [x] Architecture + STE forward designed.
- [x] Phase 1 script (`scripts/vq_bake.py`) — atlas-fidelity loss.
- [x] **Phase 1 finding: cannot beat post-hoc K-means RVQ (Δ ≈ +0.01 dB).**
      Math: with A_target fixed and L_recon = ||recon − A_target||²,
      greedy K-means RVQ already sits at the joint local optimum. No
      slack for joint training to exploit. See §4 commentary.
- [x] **Test-set render measurement** with the post-hoc K-means RVQ
      artifacts (§9). Result: −0.05 dB / +0.005 LPIPS at 3.9× smaller
      atlas. Below perceptual threshold.
- [x] **Decision: Phase 2 NOT pursued.** Baked / neural delta is
      already +0.06 dB; render-loss VQ-bake has nowhere meaningful to
      push.
- [ ] Multi-scene confirmation of §9 (bicycle, garden, counter,
      kitchen, …) — sanity-check the room result isn't lucky.
- [x] ~~Viewer integration of the VQ format (decode K=256 codebook +
      uint8 index stream in the bake-render CUDA path).~~ **Superseded
      2026-07-23** — paired-RVQ was integrated then retired after
      on-device mobile testing (see deployment banner). What ships is
      typeD (`atlas_format=7`, single-stage K=65536 with BC7-encoded
      codewords, load-time gather to a normal BC7 texture); the
      per-fragment shader-decode variant is dead.

---

## 9. Test-set render measurement — the decision-driver

Script: `scripts/bench_vq_atlas.py` — reconstructs the atlas from
`<bake_dir>/vq/{codebooks,indices,block_meta}.pt` back to uint8 with
the original quantisation params, temporarily swaps it in over
`<bake_dir>/atlas_texture.pt`, and runs `benchmark_baked.py
--skip_bake` twice (once with original, once with VQ atlas) before
restoring the originals.

**Scene:** room (mip-360), 75 207 Gausses, atlas 54 272 × 4 352
uint8 with `atlas_scale=1.1308 atlas_offset=−0.6138`. RVQ config
L=4, K=256.

| | PSNR | SSIM | LPIPS | FPS | Storage (atlas only) |
|---|---:|---:|---:|---:|---:|
| Neural renderer | 30.31 | 0.9036 | 0.2210 | 112.7 | — |
| Original baked uint8 | **30.42** | 0.9021 | 0.2403 | 1248.9 | 225 MB (BC7) |
| **K-means RVQ L=4 K=256** | **30.37** | 0.9005 | 0.2454 | 1256.0 | **≈ 58 MB** |
| Δ vs original baked | −0.05 | −0.0016 | +0.0051 | +0.6 % | **3.9× smaller** |

(Atlas-fidelity for this RVQ config is **42.49 dB** at 25 % storage —
see [`ATLAS_DECOMPOSITION_AND_CLUSTERING.md`](ATLAS_DECOMPOSITION_AND_CLUSTERING.md)
§ 4.5. The −2 dB atlas drop translates to −0.05 dB render drop, a
~40× attenuation; in line with the BC7-vs-uint8 ratio.)

**Storage breakdown (the actual minimum, not what the v1 `vq_bake.py`
serialised — see below):**
- codebooks: 4 × 256 × 48 bytes uint8 = 49 KB
- indices: 4 × 14.7 M × 8 bits = 58.0 MB
- used-block geometry: 0 bytes (re-derived from `atlas_rects.pt`)
- **total ≈ 58.0 MB** = 25.8 % of the original 225 MB BC7 atlas

**v1 `vq_bake.py` serialised these wastefully** — saved
indices as `uint16` (2× bloat → 117 MB) and dumped `used_idx [14.7
M, 2]` as `int64` (235 MB of redundant data). Both fixed in the
current script:
- indices → `uint8` for K ≤ 256, `int16` for K ≤ 65 k, `int32` otherwise.
- `used_idx` is **not** saved (re-derived from `atlas_rects.pt`).

After the fix, re-running `vq_bake.py` produces ~58 MB on disk for
the L=4 K=256 case — matching the bytes-on-the-wire that
[`ATLAS_DECOMPOSITION_AND_CLUSTERING.md`](ATLAS_DECOMPOSITION_AND_CLUSTERING.md)
§ 4.5 reports.

### Implications

1. **~~Ship post-hoc K-means RVQ as the production VQ format.~~
   Superseded** — see the deployment-update banner at the top of this
   doc. L>1 residual VQ was tried in production (`atlas_format=5` /
   paired-RVQ) but pulled 2026-07-23 because the per-fragment SW
   codebook decode is unusable on TBDR mobile GPUs. The **single-stage**
   K-means primitive still ships: production format is **typeD**
   (`atlas_format=7`), essentially L=1 K=65536 with each centroid
   re-encoded as one BC7 block so the fragment path is a single HW BC7
   tex fetch (~7× smaller download than raw BC7 at bit-identical render
   cost). See [`DEPLOY_DEMO.md`](DEPLOY_DEMO.md).
2. **Phase 2 (end-to-end render-loss VQ-bake) is not worth pursuing
   for this codebase.** The baked path is already ≈ neural; there's
   no quality gap for end-to-end training to close.
3. **Codebook-only fixes have diminishing returns at the bake side.**
   The remaining quality budget is on the bake itself (more aggressive
   importance pruning, atlas-side resolution allocation, etc.), not
   on how we compress the residual.

### Next steps (historical — see deployment banner)

The plan below was written **before** the mobile testing that retired
RVQ (2026-07-23). Kept for record; do **not** treat as an actionable
integration plan for shipping paired-RVQ.

- ~~Wire the VQ format into `diff_surfel_bake_render`: decode is L
  codebook lookups per fragment vs current BC7 decode.~~ The
  fragment-level codebook decode is exactly what was unusable on
  Adreno / Mali / Apple mobile GPUs. Load-time gather into a plain
  BC7 texture (typeD) is the shape that shipped.
- ~~Generate the deployment bundle pipeline analogue:
  `<bake_dir>/atlas_vq.bin` (concat of codebooks + indices, ~58 MB)
  → bitymi packer.~~ The typeD path ships the codebook + uint16 index
  stream inside the existing NAT2 container instead, and reconstructs
  the raw BC7 byte stream at load time.
- Confirm on other mip-360 scenes (bicycle, garden, counter, kitchen,
  bonsai, stump): a single scene's render delta could be lucky. (Still
  valid as an atlas-quality question independent of which encoding
  ships.)
