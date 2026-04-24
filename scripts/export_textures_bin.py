#!/usr/bin/env python3
"""Export a baked scene to the NAT2 binary format consumed by the Halloumi WebGPU viewer.

NAT2 format (current):
  [magic "NAT2": 4 bytes]
  [header: 64 bytes, 16 × 4B words]
    0: atlas_width   (u32)
    1: atlas_height  (u32)
    2: channels      (u32)   — 3 for FP16_RGB, 4 for UINT8_RGBA
    3: kernel_type   (u32)   — 0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled
    4: num_rects     (u32)   — must equal # Gaussians in the PLY
    5: uv_extent     (f32)   — bake UV extent (default 4.0)
    6: sb_number     (u32)   — # SB lobes per Gaussian (0 = SB disabled)
    7: atlas_format  (u32)   — 0=FP16_RGB, 1=UINT8_RGBA
    8: sh_bias       (f32)   — CUDA d_sh_bias
    9: res_bias      (f32)   — CUDA d_res_bias
   10: compact_mult  (f32)   — CUDA d_compact_mult (FastGS Compact Box)
   11: _pad0         (u32)
   12: atlas_scale   (f32)   — UINT8 dequant multiplier (unused for FP16)
   13: atlas_offset  (f32)   — UINT8 dequant offset
   14: _pad1         (u32)
   15: _pad2         (u32)
  [rects: num_rects * 4 f32]           — (u0_px, v0_px, w_px, h_px) per Gaussian
  [atlas: atlas_height * atlas_width * channels bytes]
    FP16_RGB   → C=3, 2 bytes/channel
    UINT8_RGBA → C=4, 1 byte/channel  (A unused, matches cudaTextureObject uchar4 layout)
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

ATLAS_FORMAT_FP16_RGB = 0
ATLAS_FORMAT_UINT8_RGBA = 1


def _load_meta(meta_path: Path):
    """Defaults match CUDA's d_sh_bias / d_res_bias / d_compact_mult defaults."""
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return {
        "kernel":        meta.get("kernel", "gaussian"),
        "uv_extent":     float(meta.get("uv_extent", 4.0)),
        "sh_bias":       float(meta.get("sh_bias", 0.5)),
        "res_bias":      float(meta.get("res_bias", 0.0)),
        "compact_mult":  float(meta.get("compact_mult", 1.0)),
        "atlas_format":  str(meta.get("atlas_format", "fp16_rgb")).lower(),
        "atlas_scale":   float(meta.get("atlas_scale", 1.0)),
        "atlas_offset":  float(meta.get("atlas_offset", 0.0)),
        "sb_number":     int(meta.get("sb_number", 0)),
    }


def export_nat2(baked_dir: str, output_path: str | None = None,
                kernel_type: int | None = None, legacy_natl: bool = False):
    baked = Path(baked_dir)

    atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
    atlas_rects = torch.load(baked / "atlas_rects.pt", map_location="cpu")
    assert atlas_tex.ndim == 3, f"Expected [H, W, C], got {atlas_tex.shape}"
    assert atlas_rects.ndim == 2 and atlas_rects.shape[1] == 4, \
        f"Expected rects [N, 4], got {atlas_rects.shape}"

    H, W, C = atlas_tex.shape
    N = atlas_rects.shape[0]

    meta = _load_meta(baked / "bake_meta.json")
    if kernel_type is None:
        kernel_type = KERNEL_MAP.get(meta["kernel"], 0)

    # Resolve atlas format: bake_meta.json wins. Fall back to tensor dtype heuristic:
    # uint8 tensor → uint8_rgba, otherwise fp16_rgb.
    atlas_format_str = meta["atlas_format"]
    if atlas_format_str == "fp16_rgb":
        atlas_format = ATLAS_FORMAT_FP16_RGB
    elif atlas_format_str == "uint8_rgba":
        atlas_format = ATLAS_FORMAT_UINT8_RGBA
    else:
        atlas_format = (ATLAS_FORMAT_UINT8_RGBA if atlas_tex.dtype == torch.uint8
                        else ATLAS_FORMAT_FP16_RGB)

    atlas_rects_np = atlas_rects.to(torch.float32).contiguous().numpy()

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
    sb_path = baked / "sb_params.pt"
    sb_number = 0
    sb_bytes = b""
    if sb_path.exists():
        sb_tensor = torch.load(sb_path, map_location="cpu").to(torch.float32).contiguous()
        if sb_tensor.ndim != 3 or sb_tensor.shape[0] != N or sb_tensor.shape[2] != 6:
            raise SystemExit(
                f"sb_params.pt has shape {list(sb_tensor.shape)}; expected [N={N}, K, 6]"
            )
        sb_number = int(sb_tensor.shape[1])
        sb_bytes = sb_tensor.numpy().tobytes()
    elif meta["sb_number"] > 0:
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
