#!/usr/bin/env python3
"""Export baked atlas textures to NATL binary format for WebGPU renderer.

NATL format:
  [magic: 4 bytes "NATL"]
  [atlas_width: u32 LE]
  [atlas_height: u32 LE]
  [channels: u32 LE]        (3 for DC)
  [kernel_type: u32 LE]     (0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled)
  [num_rects: u32 LE]
  [uv_extent: f32 LE]       (typically 4.0)
  [_pad: u32 LE]            (padding for 32-byte header alignment)
  [rects: num_rects * 4 * 4 bytes (f32 LE)]  — [u0_px, v0_px, w_px, h_px] per Gaussian
  [atlas_data: atlas_height * atlas_width * channels * 2 bytes (FP16 LE)]

Usage:
  python export_textures_bin.py <baked_dir> [output.natl] [--kernel-type 0]

Reads atlas_texture.pt, atlas_rects.pt, and bake_meta.json from baked_dir.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


def export_atlas(baked_dir: str, output_path: str | None = None, kernel_type: int | None = None):
    baked = Path(baked_dir)

    atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
    atlas_rects = torch.load(baked / "atlas_rects.pt", map_location="cpu")

    assert atlas_tex.ndim == 3, f"Expected [H, W, C], got {atlas_tex.shape}"
    assert atlas_rects.ndim == 2 and atlas_rects.shape[1] == 4, f"Expected [N, 4], got {atlas_rects.shape}"

    H, W, C = atlas_tex.shape
    N = atlas_rects.shape[0]

    # Read metadata
    meta_path = baked / "bake_meta.json"
    uv_extent = 4.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        uv_extent = meta.get("uv_extent", 4.0)
        if kernel_type is None:
            kernel_map = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
            kernel_type = kernel_map.get(meta.get("kernel", "gaussian"), 0)
        print(f"  meta: kernel={meta.get('kernel')}, uv_extent={uv_extent}, "
              f"res_dist={meta.get('resolution_distribution')}")

    if kernel_type is None:
        kernel_type = 0

    atlas_tex = atlas_tex.to(torch.float16).contiguous()
    atlas_rects = atlas_rects.to(torch.float32).contiguous()

    if output_path is None:
        output_path = str(baked / "scene.natl")

    with open(output_path, "wb") as f:
        # 32-byte header
        f.write(b"NATL")
        f.write(struct.pack("<IIIIIfI", W, H, C, kernel_type, N, uv_extent, 0))
        # Rects: [N, 4] as f32
        f.write(atlas_rects.numpy().tobytes())
        # Atlas: [H, W, C] as FP16
        f.write(atlas_tex.numpy().tobytes())

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Exported atlas {W}x{H}x{C} + {N} rects (kernel_type={kernel_type}) "
          f"to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export baked atlas to NATL binary format")
    parser.add_argument("baked_dir", help="Path to baked_atlas/ directory")
    parser.add_argument("output", nargs="?", help="Output path (default: baked_dir/scene.natl)")
    parser.add_argument("--kernel-type", type=int, default=None,
                        help="Override kernel type: 0=Gaussian, 1=Beta, 4=BetaScaled")
    args = parser.parse_args()
    export_atlas(args.baked_dir, args.output, args.kernel_type)
