#!/usr/bin/env python3
"""Export a baked scene to the NAT2 binary format consumed by the Halloumi WebGPU viewer.

NAT2 format (current):
  [magic "NAT2": 4 bytes]
  [header: 64 bytes, 16 × 4B words]
    0: atlas_width   (u32)
    1: atlas_height  (u32)
    2: channels      (u32)   — 3 for FP16_RGB, 4 for UINT8_RGBA / BC7
    3: kernel_type   (u32)   — 0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled
    4: num_rects     (u32)   — must equal # Gaussians in the PLY
    5: uv_extent     (f32)   — bake UV extent (default 4.0)
    6: sb_number     (u32)   — # SB lobes per Gaussian (0 = SB disabled)
    7: atlas_format  (u32)   — 0=FP16_RGB, 1=UINT8_RGBA, 2=BC7 (texture_2d_array)
    8: sh_bias       (f32)   — CUDA d_sh_bias
    9: res_bias      (f32)   — CUDA d_res_bias
   10: compact_mult  (f32)   — CUDA d_compact_mult (FastGS Compact Box)
   11: layer_h       (u32)   — BC7 only: per-layer texture height in pixels
   12: atlas_scale   (f32)   — UINT8/BC7 dequant multiplier (unused for FP16)
   13: atlas_offset  (f32)   — UINT8/BC7 dequant offset
   14: n_layers      (u32)   — BC7 only: number of layers in the texture_2d_array
   15: _pad2         (u32)
  [if BC7] layer_cuts: (n_layers + 1) × u32   — pixel-row boundaries [0, c_1, ..., atlas_height]
  [rects: num_rects * 4 f32]                  — (u0_px, v0_px, w_px, h_px) per Gaussian
                                                 (v0_px is GLOBAL atlas y; loader splits by layer)
  [atlas payload]
    FP16_RGB   → atlas_height * atlas_width * 3 * 2 B
    UINT8_RGBA → atlas_height * atlas_width * 4     B
    BC7        → concatenated per-layer BC7 byte stream:
                   layer i bytes = ((cuts[i+1] - cuts[i]) / 4) * (atlas_width / 4) * 16
  [if sb_number > 0] sb_params: num_rects * sb_number * 6 f32

Legacy NATL (v1) output can be requested with --legacy-natl for compatibility
with old viewers. Legacy atlas is always FP16 RGB, no SB, no bake scalars.

Reads from baked_dir:
  atlas_texture.pt    — required: [H, W, C] tensor (FP16 for fp16_rgb, uint8 for uint8_rgba)
  atlas_rects.pt      — required: [N, 4] float32
  sb_params.pt        — optional: [N, K, 6] float32
  bake_meta.json      — optional: kernel, uv_extent, sh_bias, res_bias, compact_mult,
                        atlas_format (str), atlas_scale, atlas_offset, sb_number

Usage:
  python export_textures_bin.py <baked_dir> [output.nat2]
  python export_textures_bin.py <baked_dir> --legacy-natl out.natl
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


KERNEL_MAP = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}

ATLAS_FORMAT_FP16_RGB   = 0
ATLAS_FORMAT_UINT8_RGBA = 1
ATLAS_FORMAT_BC7        = 2
ATLAS_FORMAT_RVQ        = 4   # post-hoc RVQ codebook + indices (decoded → uint8 atlas)
ATLAS_FORMAT_RVQ_PAIRED = 5   # paired-RVQ (L=4 → L=2, K²=65536), uint8 codebook 2D texture
                              # — direct fragment-shader decode, no atlas reconstruction

# texture_2d_array layer height for BC7 atlases. Capped at 8192 because
# Android Adreno/Mali typically expose max_texture_dimension_2d=8192 to WebGPU,
# while iOS Safari and desktop adapters allow 16384. 8192 covers both at the
# cost of ~2× layer count.
LAYER_H_BC7 = 8192


def _load_meta(meta_path: Path):
    """Defaults match CUDA's d_sh_bias / d_res_bias / d_compact_mult defaults.

    For BC7 bakes, `atlas_dtype` in bake_meta.json determines the format —
    `atlas_format` is the v1 string field, kept for back-compat.
    """
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    # Resolve format precedence: explicit `atlas_format` string > `atlas_dtype` > fp16.
    fmt_str = meta.get("atlas_format")
    if fmt_str is None:
        dtype = str(meta.get("atlas_dtype", "")).lower()
        fmt_str = {"bc7": "bc7", "uint8": "uint8_rgba",
                   "fp16": "fp16_rgb"}.get(dtype, "fp16_rgb")
    fmt_str = str(fmt_str).lower()

    return {
        "kernel":        meta.get("kernel", "gaussian"),
        "uv_extent":     float(meta.get("uv_extent", 4.0)),
        "sh_bias":       float(meta.get("sh_bias", 0.5)),
        "res_bias":      float(meta.get("res_bias", 0.0)),
        "compact_mult":  float(meta.get("compact_mult", 1.0)),
        "atlas_format":  fmt_str,
        "atlas_scale":   float(meta.get("atlas_scale", 1.0)),
        "atlas_offset":  float(meta.get("atlas_offset", 0.0)),
        "sb_number":     int(meta.get("sb_number", 0)),
        # BC7-specific (only set when atlas_dtype == "bc7"):
        "atlas_bc7_padded_h": int(meta.get("atlas_bc7_padded_h", 0)),
        "atlas_bc7_padded_w": int(meta.get("atlas_bc7_padded_w", 0)),
    }


def _find_layer_cuts(rects_np, atlas_h, max_layer_h=LAYER_H_BC7, align=4):
    """Find vertical cut positions in [0, atlas_h] such that:
      - no rect spans a cut (every rect lies fully within one layer)
      - each (cuts[i+1] - cuts[i]) <= max_layer_h
      - cuts are 4-aligned (BC7 block boundaries)
    Returns list [0, c_1, ..., atlas_h] of length n_layers + 1.
    """
    import bisect
    if atlas_h <= max_layer_h:
        return [0, int(atlas_h)]

    opens  = sorted(rects_np[:, 1].astype(int).tolist())
    closes = sorted((rects_np[:, 1] + rects_np[:, 3]).astype(int).tolist())

    def is_safe(Y):
        # active rect count at Y == 0  ⇔  no rect spans Y
        return bisect.bisect_left(opens, Y) == bisect.bisect_right(closes, Y)

    cuts = [0]
    while cuts[-1] + max_layer_h < atlas_h:
        target = ((cuts[-1] + max_layer_h) // align) * align
        Y = target
        while Y > cuts[-1] and not is_safe(Y):
            Y -= align
        if Y <= cuts[-1]:
            raise RuntimeError(
                f"No safe layer cut between {cuts[-1]} and {target} "
                f"(re-bake with --max_layer_h <= {max_layer_h} to fix)."
            )
        cuts.append(int(Y))
    cuts.append(int(atlas_h))
    return cuts


def _export_rvq(baked: Path, output_path: str | None, kernel_type: int,
                meta: dict, atlas_rects_np: np.ndarray, N: int,
                legacy_natl: bool) -> None:
    """RVQ path — reads vq/{codebooks,indices,block_meta}.pt and writes a
    NAT2 with atlas_format=4. The viewer reconstructs a uint8 atlas at load
    time by summing L codeword lookups per 4×4 block.

    NAT2 RVQ payload layout (after header + layer_cuts + rects):
        [4 B]  sub-magic "RVQ\\0"
        [4 B]  version (u32 = 1)
        [4 B]  L (num RVQ stages, u32)
        [4 B]  K (codewords per stage, u32)
        [4 B]  block_size in pixels (u32, e.g. 4)
        [4 B]  num_used_blocks (u32, sanity)
        [4 B]  bytes_per_index (1 / 2 / 4; depends on K)
        [4 B]  reserved
        [L × K × block_size² × 3 × 2 B]  codebooks (FP16, atlas-residual float space)
        [num_used_blocks × L × bytes_per_index]  indices, stage-major
                  (i.e. all stage-0 indices first, then stage-1, etc.)

    The viewer reconstructs the atlas in JS:
        for each used block i (in atlas-row-major order, derived from rects):
            float[48] block = sum_{l=0..L-1} codebooks[l][indices[l*N+i]]
            for each pixel p in the 4×4 block:
                u8 = clamp((block[p] - atlas_offset) / atlas_scale * 255, 0, 255)
            paste 4×4 into the per-layer uint8 atlas at the block's position
    """
    if legacy_natl:
        raise SystemExit("--legacy-natl is FP16 RGB only; not compatible with RVQ.")

    vq_dir = baked / "vq"
    cb_path = vq_dir / "codebooks.pt"
    ix_path = vq_dir / "indices.pt"
    bm_path = vq_dir / "block_meta.pt"
    for p in (cb_path, ix_path, bm_path):
        if not p.exists():
            raise SystemExit(f"RVQ export needs {p} (run vq_bake.py or "
                             f"bake_cluster_blocks_residual.py first).")

    codebooks = torch.load(cb_path, map_location="cpu", weights_only=False).to(torch.float16).contiguous()
    indices = torch.load(ix_path, map_location="cpu", weights_only=False)
    block_meta = torch.load(bm_path, map_location="cpu", weights_only=False)

    L = int(codebooks.shape[0])
    K = int(codebooks.shape[1])
    D = int(codebooks.shape[2])
    B = int(block_meta["block"])
    if D != B * B * 3:
        raise SystemExit(f"codebook dim {D} != block_size²·3 = {B*B*3}")
    n_used_blocks = int(block_meta["n_used_blocks"])
    if indices.shape != (L, n_used_blocks):
        raise SystemExit(
            f"indices shape {tuple(indices.shape)} != ({L}, {n_used_blocks})")
    if K <= 256:
        bytes_per_index = 1; idx_dtype = np.uint8
    elif K <= 65536:
        bytes_per_index = 2; idx_dtype = np.uint16
    else:
        bytes_per_index = 4; idx_dtype = np.uint32

    # Atlas dims: prefer bake_meta atlas_bc7_padded_h/w (set when bake was BC7),
    # else fall back to atlas_texture.pt shape.
    H = int(meta["atlas_bc7_padded_h"])
    W = int(meta["atlas_bc7_padded_w"])
    if H == 0 or W == 0:
        atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
        H, W = int(atlas_tex.shape[0]), int(atlas_tex.shape[1])
        del atlas_tex
    cuts = _find_layer_cuts(atlas_rects_np, H, LAYER_H_BC7, align=4)
    n_layers = len(cuts) - 1

    sb_number, sb_bytes = _load_sb(baked, N)

    if output_path is None:
        output_path = str(baked / "scene.nat2")

    # Top-level NAT2 header (same 64-byte layout as BC7/uint8 paths)
    header = struct.pack(
        "<IIIIIfIIfffIffII",
        W, H, 4, kernel_type,
        N, meta["uv_extent"], sb_number, ATLAS_FORMAT_RVQ,
        meta["sh_bias"], meta["res_bias"], meta["compact_mult"], LAYER_H_BC7,
        meta["atlas_scale"], meta["atlas_offset"], n_layers, 0,
    )
    assert len(header) == 64

    # RVQ sub-header
    rvq_subheader = struct.pack(
        "<4sIIIIIII",
        b"RVQ\x00",
        1,                           # version
        L,
        K,
        B,
        n_used_blocks,
        bytes_per_index,
        0,                           # reserved
    )
    assert len(rvq_subheader) == 32

    codebooks_bytes = codebooks.numpy().tobytes()                          # [L, K, D] FP16
    indices_bytes = indices.numpy().astype(idx_dtype).tobytes()            # [L, N] stage-major

    with open(output_path, "wb") as f:
        f.write(b"NAT2")
        f.write(header)
        f.write(struct.pack(f"<{n_layers + 1}I", *cuts))
        f.write(atlas_rects_np.tobytes())
        f.write(rvq_subheader)
        f.write(codebooks_bytes)
        f.write(indices_bytes)
        f.write(sb_bytes)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[NAT2 RVQ] {W}x{H}, {n_layers} layers, {N} rects, L={L}, K={K}, B={B}, "
          f"n_used={n_used_blocks:,}, idx_bytes={bytes_per_index}, "
          f"codebook={len(codebooks_bytes)/1024:.1f} KB, "
          f"indices={len(indices_bytes)/1024/1024:.2f} MB "
          f"→ {output_path} ({size_mb:.2f} MB)")


def _reorder_indices_surfel_major(indices_row_major: np.ndarray,
                                   atlas_rects_np: np.ndarray,
                                   atlas_HW: tuple, block_size: int = 4):
    """Row-major (raster of used blocks across the atlas image) → surfel-major
    (per-Gauss block range, indexed by surfel_offsets). Matches the CUDA
    bench's reorder_indices_surfel_major; we re-implement in numpy so the
    producer doesn't import torch tensors. Returns (indices_sm[L,N_used],
    surfel_offsets[N_rects+1, uint32]).
    """
    L, N = indices_row_major.shape
    H, W = atlas_HW
    B = block_size
    M = atlas_rects_np.shape[0]

    bw = (atlas_rects_np[:, 2].astype(np.int64) // B)
    bh = (atlas_rects_np[:, 3].astype(np.int64) // B)
    blocks_per_surfel = bw * bh
    surfel_offsets = np.zeros(M + 1, dtype=np.int64)
    surfel_offsets[1:] = np.cumsum(blocks_per_surfel)
    N_used = int(surfel_offsets[-1])
    assert N_used == N, f"surfel-block sum {N_used} != indices N {N}"

    # Used-block-id grid (row-major over (H/B, W/B) of *used* blocks).
    used_mask = np.zeros((H // B, W // B), dtype=bool)
    u0v = (atlas_rects_np[:, 0].astype(np.int64) // B)
    v0v = (atlas_rects_np[:, 1].astype(np.int64) // B)
    for i in range(M):
        ww = int(atlas_rects_np[i, 2]); hh = int(atlas_rects_np[i, 3])
        if ww == 0 or hh == 0:
            continue
        used_mask[v0v[i]:v0v[i] + hh // B, u0v[i]:u0v[i] + ww // B] = True
    rm_flat = used_mask.reshape(-1).astype(np.int64)
    rm_id_flat = np.cumsum(rm_flat) - 1
    rm_id_flat[~used_mask.reshape(-1)] = -1
    rm_id = rm_id_flat.reshape(H // B, W // B)

    perm = np.empty(N_used, dtype=np.int64)
    for i in range(M):
        nw = int(bw[i]); nh = int(bh[i])
        if nw == 0 or nh == 0:
            continue
        off = int(surfel_offsets[i])
        sub = rm_id[v0v[i]:v0v[i] + nh, u0v[i]:u0v[i] + nw].reshape(-1)
        perm[off:off + nw * nh] = sub

    indices_sm = indices_row_major[:, perm]
    return indices_sm, surfel_offsets.astype(np.uint32)


def _export_rvq_paired(baked: Path, output_path: str | None, kernel_type: int,
                        meta: dict, atlas_rects_np: np.ndarray, N: int,
                        legacy_natl: bool) -> None:
    """Paired-RVQ path — collapses L=4 RVQ stages into L=2 pair codebooks
    (K²=65536 entries each, K=256). Emits NAT2 atlas_format=5 with a uint8
    2D codebook image and surfel-major uint32 packed indices. The viewer
    decodes directly in the fragment shader (no atlas reconstruction).

    See diff_surfel_bake_render_rvq_paired/rasterize_points.cu for the
    canonical encoder — this is a host-side port of that logic.

    NAT2 atlas_format=5 payload (after header + layer_cuts + rects):
        [4 B]  sub-magic "RVQP"
        [4 B]  version (u32 = 1)
        [4 B]  K_orig    (u32, codewords per ORIGINAL stage, typically 256)
        [4 B]  B         (u32, block size in pixels, typically 4)
        [4 B]  num_rects (u32, sanity)
        [4 B]  N_used    (u32, total used blocks across all surfels)
        [4 × 4 B]  pair_scale[2], pair_offset[2]   (f32 × 4)
        [(K_orig*B) × (2*K_orig*B) × 4 B]  codebook image (RGBA8 row-major, alpha=0)
        [N_used × 4 B]       packed indices (uint32, surfel-major)
        [(N_rects + 1) × 4 B] surfel_offsets (uint32 cumulative block count)

    Packing:
        packed = (pair1_idx << 16) | pair0_idx
        pair0_idx = (idx_stage_0 << 8) | idx_stage_1   (pair 0 combines stages 0+1)
        pair1_idx = (idx_stage_2 << 8) | idx_stage_3   (pair 1 combines stages 2+3)
    Codebook tile layout (B=4): at (c_lo*B + iu, p*(K_orig*B) + c_hi*B + iv),
    RGB carries the 3 components of voxel `iv*B + iu` of the summed codeword
    (c_hi=idx_hi, c_lo=idx_lo); alpha unused.
    """
    if legacy_natl:
        raise SystemExit("--legacy-natl is FP16 RGB only; not compatible with RVQ paired.")

    vq_dir = baked / "vq"
    cb_path = vq_dir / "codebooks.pt"
    ix_path = vq_dir / "indices.pt"
    bm_path = vq_dir / "block_meta.pt"
    for p in (cb_path, ix_path, bm_path):
        if not p.exists():
            raise SystemExit(f"RVQ-paired export needs {p}")

    codebooks = torch.load(cb_path, map_location="cpu", weights_only=False).to(torch.float32).contiguous().numpy()
    indices = torch.load(ix_path, map_location="cpu", weights_only=False).numpy()
    block_meta = torch.load(bm_path, map_location="cpu", weights_only=False)

    L, K, D = codebooks.shape
    B = int(block_meta["block"])
    if L != 4 or B != 4:
        raise SystemExit(f"paired-RVQ requires L=4, B=4 (got L={L}, B={B}). "
                         "Re-bake VQ with the canonical config.")
    if D != B * B * 3:
        raise SystemExit(f"codebook dim {D} != block_size²·3 = {B*B*3}")
    n_used_blocks = int(block_meta["n_used_blocks"])
    if indices.shape != (L, n_used_blocks):
        raise SystemExit(
            f"indices shape {tuple(indices.shape)} != ({L}, {n_used_blocks})")
    if K > 256:
        raise SystemExit(f"paired-RVQ requires K ≤ 256 (got K={K}); pair index "
                         "packs to (idx_hi<<8)|idx_lo in a u16.")

    # Atlas dims (for the row-major → surfel-major index reorder).
    H = int(meta["atlas_bc7_padded_h"])
    W = int(meta["atlas_bc7_padded_w"])
    if H == 0 or W == 0:
        atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
        H, W = int(atlas_tex.shape[0]), int(atlas_tex.shape[1])
        del atlas_tex

    # Build pair codebooks: pair[p][k_hi, k_lo, d] = cb[2p][k_hi, d] + cb[2p+1][k_lo, d].
    #   pair_cb shape: [2, K, K, D]
    pair_cb = np.empty((2, K, K, D), dtype=np.float32)
    for p in range(2):
        cb_hi = codebooks[2 * p    ]    # [K, D]
        cb_lo = codebooks[2 * p + 1]    # [K, D]
        pair_cb[p] = cb_hi[:, None, :] + cb_lo[None, :, :]

    # Per-pair quantization range (use empirical min/max).
    pair_scale  = np.zeros(2, dtype=np.float32)
    pair_offset = np.zeros(2, dtype=np.float32)
    for p in range(2):
        mn = float(pair_cb[p].min()); mx = float(pair_cb[p].max())
        pair_scale[p]  = (mx - mn) if (mx - mn) > 1e-8 else 1.0
        pair_offset[p] = mn

    # Build the codebook image: width = K*B, height = 2*K*B, 4 channels (RGBA8).
    Wp = K * B
    Hp = 2 * K * B
    # pair_cb is laid out (2, K, K, B*B*3) = (2, K_hi, K_lo, voxel, channel).
    # Reshape to (2, K_hi, K_lo, B, B, 3) → (2, K_hi, B, K_lo, B, 3) → (2*K_hi*B, K_lo*B, 3).
    pair_cb_voxel = pair_cb.reshape(2, K, K, B, B, 3)              # (p, c_hi, c_lo, iv, iu, ch)
    pair_image_rgb = pair_cb_voxel.transpose(0, 1, 3, 2, 4, 5).reshape(Hp, Wp, 3)

    # Quantize per pair (scan rows by pair index).
    pair_image_rgba = np.zeros((Hp, Wp, 4), dtype=np.uint8)
    for p in range(2):
        y0 = p * K * B; y1 = (p + 1) * K * B
        slab = pair_image_rgb[y0:y1]               # (K*B, K*B, 3) float32
        q = (slab - pair_offset[p]) / pair_scale[p] * 255.0 + 0.5
        np.clip(q, 0.0, 255.0, out=q)
        pair_image_rgba[y0:y1, :, :3] = q.astype(np.uint8)
    # alpha stays 0 (unused)

    # Reorder row-major indices → surfel-major.
    indices_sm, surfel_offsets = _reorder_indices_surfel_major(
        indices, atlas_rects_np, (H, W), B)

    # Pack to uint32 per block: pair0 = (idx_0<<8)|idx_1; pair1 = (idx_2<<8)|idx_3;
    #                          packed = (pair1<<16) | pair0.
    idx0 = indices_sm[0].astype(np.uint16)
    idx1 = indices_sm[1].astype(np.uint16)
    idx2 = indices_sm[2].astype(np.uint16)
    idx3 = indices_sm[3].astype(np.uint16)
    pair0 = (idx0 << 8) | idx1
    pair1 = (idx2 << 8) | idx3
    packed = (pair1.astype(np.uint32) << 16) | pair0.astype(np.uint32)

    cuts = _find_layer_cuts(atlas_rects_np, H, LAYER_H_BC7, align=4)
    n_layers = len(cuts) - 1

    sb_number, sb_bytes = _load_sb(baked, N)

    if output_path is None:
        output_path = str(baked / "scene.nat2")

    # NAT2 top-level header. layer_h and atlas dims kept for parser
    # consistency (rects carry global v0; layer_cuts split them into local).
    # n_layers is set to layer_cuts.size-1 just like the BC7 path.
    header = struct.pack(
        "<IIIIIfIIfffIffII",
        W, H, 4, kernel_type,
        N, meta["uv_extent"], sb_number, ATLAS_FORMAT_RVQ_PAIRED,
        meta["sh_bias"], meta["res_bias"], meta["compact_mult"], LAYER_H_BC7,
        meta["atlas_scale"], meta["atlas_offset"], n_layers, 0,
    )
    assert len(header) == 64

    # RVQ-paired sub-header
    rvqp_subheader = struct.pack(
        "<4sIIIII",
        b"RVQP",
        1,                           # version
        K,                           # K_orig
        B,                           # block size in pixels
        N,                           # num_rects (sanity)
        int(packed.size),            # N_used
    )
    assert len(rvqp_subheader) == 24
    pair_dequant_bytes = pair_scale.tobytes() + pair_offset.tobytes()  # 16 B
    assert len(pair_dequant_bytes) == 16

    cb_image_bytes = pair_image_rgba.tobytes()
    packed_bytes   = packed.astype(np.uint32).tobytes()
    offsets_bytes  = surfel_offsets.astype(np.uint32).tobytes()

    with open(output_path, "wb") as f:
        f.write(b"NAT2")
        f.write(header)
        f.write(struct.pack(f"<{n_layers + 1}I", *cuts))
        f.write(atlas_rects_np.tobytes())
        f.write(rvqp_subheader)
        f.write(pair_dequant_bytes)
        f.write(cb_image_bytes)
        f.write(packed_bytes)
        f.write(offsets_bytes)
        f.write(sb_bytes)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[NAT2 RVQ-PAIRED] {W}x{H}, {n_layers} layers, {N} rects, "
          f"K_orig={K}, B={B}, N_used={packed.size:,}, "
          f"codebook image {Wp}x{Hp} rgba8 ({len(cb_image_bytes)/1024/1024:.1f} MB), "
          f"packed indices ({len(packed_bytes)/1024/1024:.1f} MB), "
          f"surfel_offsets ({len(offsets_bytes)/1024/1024:.2f} MB), "
          f"pair_scale={pair_scale.tolist()}, pair_offset={pair_offset.tolist()} "
          f"→ {output_path} ({size_mb:.1f} MB)")


def _load_sb(baked: Path, N: int) -> tuple[int, bytes]:
    sb_path = baked / "sb_params.pt"
    if not sb_path.exists():
        return 0, b""
    sb_tensor = torch.load(sb_path, map_location="cpu").to(torch.float32).contiguous()
    if sb_tensor.ndim != 3 or sb_tensor.shape[0] != N or sb_tensor.shape[2] != 6:
        raise SystemExit(
            f"sb_params.pt has shape {list(sb_tensor.shape)}; expected [N={N}, K, 6]"
        )
    return int(sb_tensor.shape[1]), sb_tensor.numpy().tobytes()


def _export_bc7(baked: Path, output_path: str | None, kernel_type: int,
                meta: dict, atlas_rects_np: np.ndarray, N: int,
                legacy_natl: bool) -> None:
    """BC7 path — reads atlas_texture.bc7 directly, slices into texture_2d_array
    layers, emits NAT2 with atlas_format=2.

    The BC7 byte stream is row-major in 4×4 blocks. We slice on 4-aligned pixel
    rows that don't intersect any rect; each layer ends up <= LAYER_H_BC7
    pixels tall. The shader gets per-rect (u0, v0_local, w, h) plus a layer
    index computed at load time from the global v0.
    """
    if legacy_natl:
        raise SystemExit("--legacy-natl is FP16 RGB only; not compatible with BC7.")

    bc7_path = baked / "atlas_texture.bc7"
    if not bc7_path.exists():
        raise SystemExit(f"BC7 export requires {bc7_path} but it does not exist.")
    bc7_bytes = bc7_path.read_bytes()

    H = meta["atlas_bc7_padded_h"]
    W = meta["atlas_bc7_padded_w"]
    if H == 0 or W == 0:
        raise SystemExit(
            "bake_meta.json missing atlas_bc7_padded_h/w (re-run benchmark_baked "
            "with --bake_dtype bc7 to regenerate)."
        )
    if H % 4 != 0 or W % 4 != 0:
        raise SystemExit(f"BC7 atlas dims must be 4-aligned, got {W}x{H}")

    expected_bytes = (H // 4) * (W // 4) * 16
    if len(bc7_bytes) != expected_bytes:
        raise SystemExit(
            f"BC7 file size {len(bc7_bytes)} != expected {expected_bytes} for {W}x{H}"
        )

    cuts = _find_layer_cuts(atlas_rects_np, H, LAYER_H_BC7, align=4)
    n_layers = len(cuts) - 1

    # Slice BC7 byte stream by block-row range. blocks_per_row is constant
    # (full atlas_width); only the row count changes per layer.
    blocks_per_row = W // 4
    bytes_per_block_row = blocks_per_row * 16
    layer_blobs = []
    for i in range(n_layers):
        b0 = (cuts[i]   // 4) * bytes_per_block_row
        b1 = (cuts[i+1] // 4) * bytes_per_block_row
        layer_blobs.append(bc7_bytes[b0:b1])

    sb_number, sb_bytes = _load_sb(baked, N)
    if sb_number == 0 and meta["sb_number"] > 0:
        print(f"[NAT2 BC7] WARNING: meta says sb_number={meta['sb_number']} but "
              f"sb_params.pt not found; exporting with SB disabled")

    if output_path is None:
        output_path = str(baked / "scene.nat2")

    header = struct.pack(
        "<IIIIIfIIfffIffII",
        W, H, 4, kernel_type,
        N, meta["uv_extent"], sb_number, ATLAS_FORMAT_BC7,
        meta["sh_bias"], meta["res_bias"], meta["compact_mult"], LAYER_H_BC7,
        meta["atlas_scale"], meta["atlas_offset"], n_layers, 0,
    )
    assert len(header) == 64

    with open(output_path, "wb") as f:
        f.write(b"NAT2")
        f.write(header)
        f.write(struct.pack(f"<{n_layers + 1}I", *cuts))
        f.write(atlas_rects_np.tobytes())
        for blob in layer_blobs:
            f.write(blob)
        f.write(sb_bytes)

    size_mb = Path(output_path).stat().st_size / 1e6
    layer_heights = [cuts[i+1] - cuts[i] for i in range(n_layers)]
    print(f"[NAT2 BC7] {W}x{H}, {n_layers} layers (heights {layer_heights}, "
          f"texture layer dim {LAYER_H_BC7}), {N} rects, kernel={kernel_type}, "
          f"sb_number={sb_number}, sh_bias={meta['sh_bias']}, "
          f"res_bias={meta['res_bias']}, compact_mult={meta['compact_mult']}, "
          f"atlas_scale={meta['atlas_scale']}, atlas_offset={meta['atlas_offset']} "
          f"→ {output_path} ({size_mb:.1f} MB)")


def export_nat2(baked_dir: str, output_path: str | None = None,
                kernel_type: int | None = None, legacy_natl: bool = False,
                rvq: bool = False, rvq_paired: bool = False):
    baked = Path(baked_dir)

    atlas_rects = torch.load(baked / "atlas_rects.pt", map_location="cpu")
    assert atlas_rects.ndim == 2 and atlas_rects.shape[1] == 4, \
        f"Expected rects [N, 4], got {atlas_rects.shape}"
    N = atlas_rects.shape[0]
    atlas_rects_np = atlas_rects.to(torch.float32).contiguous().numpy()

    meta = _load_meta(baked / "bake_meta.json")
    if kernel_type is None:
        kernel_type = KERNEL_MAP.get(meta["kernel"], 0)

    atlas_format_str = meta["atlas_format"]

    # ---- RVQ-PAIRED path: format=5, fragment-shader-side decode ----
    if rvq_paired:
        return _export_rvq_paired(baked, output_path, kernel_type, meta,
                                   atlas_rects_np, N, legacy_natl)

    # ---- RVQ path: vq/{codebooks,indices,block_meta}.pt are the source ----
    if rvq:
        return _export_rvq(baked, output_path, kernel_type, meta,
                           atlas_rects_np, N, legacy_natl)

    # ---- BC7 path: atlas_texture.bc7 is the source of truth ----
    if atlas_format_str == "bc7":
        return _export_bc7(baked, output_path, kernel_type, meta,
                           atlas_rects_np, N, legacy_natl)

    # ---- FP16 / UINT8 path: atlas_texture.pt is the source ----
    atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
    assert atlas_tex.ndim == 3, f"Expected [H, W, C], got {atlas_tex.shape}"
    H, W, C = atlas_tex.shape

    if atlas_format_str == "fp16_rgb":
        atlas_format = ATLAS_FORMAT_FP16_RGB
    elif atlas_format_str == "uint8_rgba":
        atlas_format = ATLAS_FORMAT_UINT8_RGBA
    else:
        atlas_format = (ATLAS_FORMAT_UINT8_RGBA if atlas_tex.dtype == torch.uint8
                        else ATLAS_FORMAT_FP16_RGB)

    if atlas_format == ATLAS_FORMAT_UINT8_RGBA:
        if atlas_tex.dtype != torch.uint8:
            raise SystemExit(
                f"atlas_format=uint8_rgba but atlas_texture.pt dtype is {atlas_tex.dtype}; "
                "re-bake with the uint8 quantized path."
            )
        if C == 3:
            # Pad to RGBA (alpha channel unused) so wgpu's unpack4x8unorm can read one u32/texel.
            alpha = torch.zeros((H, W, 1), dtype=torch.uint8)
            atlas_bytes = torch.cat([atlas_tex, alpha], dim=-1).contiguous().numpy()
            c_out = 4
        elif C == 4:
            atlas_bytes = atlas_tex.contiguous().numpy()
            c_out = 4
        else:
            raise SystemExit(f"UINT8 atlas must be C=3 or C=4, got C={C}")
    else:
        # FP16 RGB legacy path.
        atlas_bytes = atlas_tex.to(torch.float16).contiguous().numpy()
        c_out = C

    # SB params (optional).
    sb_number, sb_bytes = _load_sb(baked, N)
    if sb_number == 0 and meta["sb_number"] > 0:
        print(f"[NAT2] WARNING: meta says sb_number={meta['sb_number']} but "
              f"sb_params.pt not found; exporting with SB disabled")

    if legacy_natl:
        # NATL (v1) for old viewers: FP16 RGB only, no SB, no bake scalars.
        if atlas_format != ATLAS_FORMAT_FP16_RGB:
            raise SystemExit("--legacy-natl requires FP16 RGB atlas; bake with that format")
        if output_path is None:
            output_path = str(baked / "scene.natl")
        with open(output_path, "wb") as f:
            f.write(b"NATL")
            f.write(struct.pack("<IIIIIfI", W, H, c_out, kernel_type, N, meta["uv_extent"], 0))
            f.write(atlas_rects_np.tobytes())
            f.write(atlas_bytes.tobytes())
        size_mb = Path(output_path).stat().st_size / 1e6
        print(f"[NATL v1] {W}x{H}x{c_out} fp16_rgb + {N} rects, kernel={kernel_type} "
              f"→ {output_path} ({size_mb:.1f} MB)")
        return

    # NAT2 (v2).
    if output_path is None:
        output_path = str(baked / "scene.nat2")

    header = struct.pack(
        "<IIIIIfIIfffIffII",
        W, H, c_out, kernel_type,
        N, meta["uv_extent"], sb_number, atlas_format,
        meta["sh_bias"], meta["res_bias"], meta["compact_mult"], 0,
        meta["atlas_scale"], meta["atlas_offset"], 0, 0,
    )
    assert len(header) == 64, f"Header size is {len(header)} (expected 64)"

    with open(output_path, "wb") as f:
        f.write(b"NAT2")
        f.write(header)
        f.write(atlas_rects_np.tobytes())
        f.write(atlas_bytes.tobytes())
        f.write(sb_bytes)

    size_mb = Path(output_path).stat().st_size / 1e6
    fmt_str = "uint8_rgba" if atlas_format == ATLAS_FORMAT_UINT8_RGBA else "fp16_rgb"
    print(f"[NAT2] {W}x{H}x{c_out} {fmt_str} + {N} rects, kernel={kernel_type}, "
          f"sb_number={sb_number}, sh_bias={meta['sh_bias']}, res_bias={meta['res_bias']}, "
          f"compact_mult={meta['compact_mult']}, atlas_scale={meta['atlas_scale']}, "
          f"atlas_offset={meta['atlas_offset']} → {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export baked atlas → NAT2 (or legacy NATL)")
    parser.add_argument("baked_dir", help="Path to baked_atlas/ directory")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output path (default: baked_dir/scene.nat2 or scene.natl)")
    parser.add_argument("--kernel-type", type=int, default=None,
                        help="Override kernel type int (0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled)")
    parser.add_argument("--legacy-natl", action="store_true",
                        help="Emit NATL v1 (FP16 RGB, no SB, no bake scalars) for old viewers")
    parser.add_argument("--rvq", action="store_true",
                        help="Emit NAT2 with atlas_format=4 (RVQ codebooks + indices, "
                             "read from baked_dir/vq/{codebooks,indices,block_meta}.pt)")
    parser.add_argument("--rvq-paired", action="store_true",
                        help="Emit NAT2 with atlas_format=5 (paired-RVQ: L=4→L=2 collapsed "
                             "codebooks, K²=65536, uint8 2D texture; surfel-major uint32 "
                             "packed indices — for fragment-shader-side decode in the WGSL viewer)")
    args = parser.parse_args()
    export_nat2(args.baked_dir, args.output, args.kernel_type,
                args.legacy_natl, args.rvq, args.rvq_paired)
