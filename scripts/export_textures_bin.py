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

# texture_2d_array layer height for BC7 atlases. wgpu's typical
# max_texture_dimension_2d is 16384 on Apple Silicon Metal and most desktop
# adapters; tall packed atlases are sliced into n layers of this height.
LAYER_H_BC7 = 16384


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
                kernel_type: int | None = None, legacy_natl: bool = False):
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
    args = parser.parse_args()
    export_nat2(args.baked_dir, args.output, args.kernel_type, args.legacy_natl)
