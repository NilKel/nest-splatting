#!/usr/bin/env python3
"""Encode a baked uint8 RGBA atlas to ASTC 4×4 and pack into a NAT2.

Companion to scripts/export_textures_bin.py. Produces an ASTC-flavored
NAT2 (atlas_format=3) that the Halloumi-WS viewer can load on Adreno/Mali
phones (which lack BC7 but support ASTC). Same on-disk layout as the BC7
NAT2 — both are 16-byte 4×4 blocks; only the GPU texture format differs.

Uses the `astc-encoder-py` Python binding so no system astcenc install is
needed (`pip install astc-encoder-py`).

Usage:
    encode_astc.py <baked_dir>
        [--quality fastest|fast|medium|thorough|verythorough|exhaustive]
        [--output path/to/scene_astc.nat2]
        [--threads N]
"""
import argparse
import bisect
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch


ATLAS_FORMAT_ASTC_4x4 = 3
LAYER_H_ASTC = 8192      # match BC7's per-layer cap (Adreno/Mali typically expose 8192)

KERNEL_MAP = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}


def find_layer_cuts(rects_np, atlas_h, max_layer_h=LAYER_H_ASTC, align=4):
    """Vertical cut positions (4-aligned) such that no rect spans a cut and
    each layer is ≤ max_layer_h tall. Mirrors the BC7 helper in
    export_textures_bin.py.
    """
    if atlas_h <= max_layer_h:
        return [0, int(atlas_h)]
    opens  = sorted(rects_np[:, 1].astype(int).tolist())
    closes = sorted((rects_np[:, 1] + rects_np[:, 3]).astype(int).tolist())
    def is_safe(Y):
        return bisect.bisect_left(opens, Y) == bisect.bisect_right(closes, Y)
    cuts = [0]
    while cuts[-1] + max_layer_h < atlas_h:
        target = ((cuts[-1] + max_layer_h) // align) * align
        Y = target
        while Y > cuts[-1] and not is_safe(Y):
            Y -= align
        if Y <= cuts[-1]:
            raise RuntimeError(
                f"No safe layer cut between {cuts[-1]} and {target}; "
                f"reduce --max_layer_h to refit."
            )
        cuts.append(int(Y))
    cuts.append(int(atlas_h))
    return cuts


def encode_astc_4x4(uint8_rgba: np.ndarray, quality: str, threads: int) -> bytes:
    """`uint8_rgba` must be [H, W, 4] uint8 with H, W multiples of 4.
    Returns raw ASTC block stream (16 bytes per 4×4 block, no file header)."""
    try:
        from astc_encoder import (
            ASTCConfig, ASTCContext, ASTCImage,
            ASTCProfile, ASTCType, ASTCSwizzle, ASTCQualityPreset,
            ASTCSwizzleComponentSelector as Sel,
        )
    except ImportError:
        raise SystemExit(
            "astc-encoder-py not installed. Run: "
            "conda run -n nest_splatting python -m pip install astc-encoder-py"
        )

    H, W, C = uint8_rgba.shape
    assert C == 4 and H % 4 == 0 and W % 4 == 0, f"got {uint8_rgba.shape}"

    quality_map = {
        "fastest":      ASTCQualityPreset.FASTEST,
        "fast":         ASTCQualityPreset.FAST,
        "medium":       ASTCQualityPreset.MEDIUM,
        "thorough":     ASTCQualityPreset.THOROUGH,
        "verythorough": ASTCQualityPreset.VERYTHOROUGH,
        "exhaustive":   ASTCQualityPreset.EXHAUSTIVE,
    }
    cfg = ASTCConfig(ASTCProfile.LDR, 4, 4, 1, quality_map[quality])
    ctx = ASTCContext(cfg, max(1, threads))

    src = ASTCImage(ASTCType.U8, W, H, 1, uint8_rgba.tobytes())
    sw  = ASTCSwizzle(Sel.R, Sel.G, Sel.B, Sel.A)

    print(f"[ASTC] encoding {W}x{H} (4x4, {quality}, {threads} threads) ...")
    blocks = bytes(ctx.compress(src, sw))

    expected = (H // 4) * (W // 4) * 16
    if len(blocks) != expected:
        raise SystemExit(f"ASTC block stream {len(blocks)} != expected {expected}")
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baked_dir")
    ap.add_argument("--quality",
                    choices=["fastest", "fast", "medium", "thorough", "verythorough", "exhaustive"],
                    default="medium")
    ap.add_argument("--output", default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    baked = Path(args.baked_dir)

    atlas_tex = torch.load(baked / "atlas_texture.pt", map_location="cpu")
    if atlas_tex.dtype != torch.uint8:
        raise SystemExit(
            f"atlas_texture.pt dtype={atlas_tex.dtype}; need uint8. "
            f"Re-bake with --bake_dtype uint8 (or bc7 — that path saves uint8 too)."
        )
    H, W, C = atlas_tex.shape

    Hp = ((H + 3) // 4) * 4
    Wp = ((W + 3) // 4) * 4
    rgba = np.zeros((Hp, Wp, 4), dtype=np.uint8)
    rgba[:H, :W, :3] = atlas_tex.numpy()[..., :3]
    rgba[..., 3] = 255

    blocks = encode_astc_4x4(rgba, args.quality, args.threads)
    blocks_mb = len(blocks) / (1024 ** 2)
    print(f"[ASTC] encoded → {blocks_mb:.1f} MB ({len(blocks)/(H*W):.2f} B/texel; "
          f"raw uint8 was {atlas_tex.nelement()/(1024**2):.1f} MB)")

    rects = torch.load(baked / "atlas_rects.pt", map_location="cpu").to(torch.float32).contiguous()
    rects_np = rects.numpy()
    N = rects_np.shape[0]

    meta_path = baked / "bake_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    kernel_type  = KERNEL_MAP.get(str(meta.get("kernel", "gaussian")).lower(), 0)
    uv_extent    = float(meta.get("uv_extent", 4.0))
    sh_bias      = float(meta.get("sh_bias", 0.5))
    res_bias     = float(meta.get("res_bias", 0.0))
    compact_mult = float(meta.get("compact_mult", 1.0))
    atlas_scale  = float(meta.get("atlas_scale", 1.0))
    atlas_offset = float(meta.get("atlas_offset", 0.0))

    cuts = find_layer_cuts(rects_np, Hp, LAYER_H_ASTC, align=4)
    n_layers = len(cuts) - 1
    blocks_per_row = Wp // 4
    bytes_per_block_row = blocks_per_row * 16
    layer_blobs = [
        blocks[(cuts[i]   // 4) * bytes_per_block_row :
               (cuts[i+1] // 4) * bytes_per_block_row]
        for i in range(n_layers)
    ]
    layer_heights = [cuts[i + 1] - cuts[i] for i in range(n_layers)]

    out = args.output or str(baked / "scene_astc.nat2")

    header = struct.pack(
        "<IIIIIfIIfffIffII",
        Wp, Hp, 4, kernel_type,
        N, uv_extent, 0, ATLAS_FORMAT_ASTC_4x4,
        sh_bias, res_bias, compact_mult, LAYER_H_ASTC,
        atlas_scale, atlas_offset, n_layers, 0,
    )
    assert len(header) == 64

    with open(out, "wb") as f:
        f.write(b"NAT2")
        f.write(header)
        f.write(struct.pack(f"<{n_layers + 1}I", *cuts))
        f.write(rects_np.tobytes())
        for blob in layer_blobs:
            f.write(blob)

    size_mb = Path(out).stat().st_size / 1e6
    print(f"[NAT2 ASTC] {Wp}x{Hp}, {n_layers} layers (heights {layer_heights}, "
          f"texture layer dim {LAYER_H_ASTC}), {N} rects, kernel={kernel_type}, "
          f"sh_bias={sh_bias}, res_bias={res_bias}, atlas_scale={atlas_scale}, "
          f"atlas_offset={atlas_offset} → {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
