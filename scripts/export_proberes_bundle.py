#!/usr/bin/env python3
"""Export a `--method proberes` checkpoint to a NAT2 bundle for the WebGPU viewer.

Design + rationale: docs/PROBERES_WEBGPU_RENDERER.md

Proberes replaces the baked pipeline's per-surfel atlas rects with ONE shared
texture plus a per-surfel general affine ("probe"):

    tx = A00*s.x + A01*s.y + t0
    ty = A10*s.x + A11*s.y + t1
    residual = bilinear(shared_tex, tx, ty)

vs the baked path's diagonal-only `au = u0 + (s.x+E)/(2E)*w`. Same single
texture fetch per fragment; the mapping just gains rotation/shear.

Key conventions carried over from the CUDA ground truth
(`diff_surfel_3D_sh_res_probe/cuda_rasterizer/forward.cu` case 5, flag 0x1000):

  * probe layout is ROW-major [A00, A01, A10, A11, t0, t1]. WGSL's mat2x2
    constructor is COLUMN-major, so the shader must build
    mat2x2f(A00, A10, A01, A11). Handled shader-side, not here.
  * the low-pass branch forces uv = (0,0) (samples the probe CENTRE) whenever
    rho3d > rho2d. Per-fragment conditional; cannot be folded into the affine.
  * sampling is texel-centre bilinear with clamp-to-edge, which HW linear
    filtering reproduces exactly given normalised uv = (A*s + t)/tex_res.
    We therefore PRE-DIVIDE the probes by tex_res here so the shader does no
    normalisation.

Probes ship as fp32. They cannot be fp16: the translation columns span the
full 8192 texture, where fp16's ULP is 8 texels.

Quantisation of the shared texture uses FULL min/max, not a percentile clamp.
Measured on room8k_p32: full-range RMSE 0.00854 / 0% clipped, vs P99.9 clamp
RMSE 0.01187 / 0.2% clipped. The clamp makes the bulk finer but the clipped
tail (values out to -2.75/+4.66) dominates the error. See doc section 4.

Usage:
  python scripts/export_proberes_bundle.py <model_path> <out.nat2> \
      [--ckpt ngp_15000.pth] [--format bc7|astc] [--quality medium]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# NAT2 atlas_format codes. 0-8 are defined in export_textures_bin.py; proberes
# claims 9/10. The code also selects the rects-block stride (6, not 4).
ATLAS_FORMAT_PROBE_BC7 = 9
ATLAS_FORMAT_PROBE_ASTC = 10

KERNEL_MAP = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}


def load_probe_state(ckpt_path: Path):
    """Pull the two proberes tensors out of the training checkpoint.

    `probe_field.pixels` is the learned leaf parameter (the run used
    --probe_no_field, so it is NOT a cached evaluation of probe_field.mlp) —
    shipping only `pixels` is correct and the enc/mlp weights are dead here.
    """
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    if "probe_head.fixed_probes" not in sd:
        raise SystemExit(f"{ckpt_path} has no probe_head.fixed_probes — not a proberes run?")
    probes = sd["probe_head.fixed_probes"].float().numpy()      # [N, 6]
    pixels = sd["probe_field.pixels"].float().numpy()            # [Ht, Wt, 3]
    return probes, pixels


def quantise_atlas(pixels: np.ndarray):
    """float32 [H,W,3] -> uint8 RGBA + (scale, offset) with dequant
        value = code * scale + offset

    Full min/max, no clamping. See module docstring / doc section 4 for why the
    percentile clamp was measured and rejected.
    """
    lo = float(pixels.min())
    hi = float(pixels.max())
    step = (hi - lo) / 255.0                 # per-CODE step, for building u8 codes
    codes = np.clip(np.round((pixels - lo) / step), 0, 255).astype(np.uint8)

    # atlas_scale is the dequant multiplier the SHADER applies, and WebGPU
    # returns *-unorm texture samples already normalised to [0, 1] -- NOT raw
    # 0..255 codes. So the multiplier is the full SPAN, not the per-code step.
    # (Using the step here makes every residual ~= atlas_offset, i.e. a large
    # negative constant, and max(0, SV + residual) renders pure black. That is
    # exactly what the first proberes bundle did.) The baked path already does
    # it this way -- cf. "BC7 atlas installed: scale=1.9398 offset=-0.9661",
    # where 1.9398 is the range span.
    scale = hi - lo

    deq = codes.astype(np.float32) / 255.0 * scale + lo
    err = pixels - deq
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"[quant] range [{lo:+.4f}, {hi:+.4f}]  span(scale) {scale:.5f}  step {step:.5f}  rmse {rmse:.5f}")

    H, W, _ = codes.shape
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = codes
    rgba[..., 3] = 255
    return rgba, scale, lo


def build_mip_pyramid(pixels: np.ndarray, max_levels: int = 0):
    """float32 [H,W,3] -> list of float32 levels, L0 first, 2x2 box-filtered.

    Filtering happens on the FLOAT atlas, before quantisation, so coarse levels
    are true averages rather than averages-of-codes. All levels are later
    quantised with ONE shared (scale, offset) so the shader keeps a single
    dequant pair regardless of which level it samples.

    Why mips at all: bilinear alone handles magnification, not MINIFICATION. A
    surfel with a 74-texel footprint drawn 5 px wide steps ~15 texels per pixel
    -> shimmer under camera motion. The renderer's low-pass branch is only a
    binary 2-level approximation (full-res bilinear, or collapse to the probe
    centre) and pops at the switch; a real chain makes that transition smooth.
    """
    levels = [pixels]
    H, W, _ = pixels.shape
    while min(H, W) > 4 and (max_levels == 0 or len(levels) < max_levels):
        cur = levels[-1]
        H, W = cur.shape[0] // 2, cur.shape[1] // 2
        levels.append(cur[:2 * H, :2 * W].reshape(H, 2, W, 2, 3).mean(axis=(1, 3)))
    return levels


def quantise_level(level: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Quantise one mip level with the pyramid-wide (scale, offset)."""
    step = scale / 255.0
    codes = np.clip(np.round((level - offset) / step), 0, 255).astype(np.uint8)
    H, W, _ = codes.shape
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = codes
    rgba[..., 3] = 255
    return rgba


def encode_bc7(rgba: np.ndarray) -> bytes:
    try:
        import bc7encoder
    except ImportError:
        raise SystemExit("bc7encoder not installed (build from submodules/bc7enc_lib).")
    H, W, _ = rgba.shape
    print(f"[bc7] encoding {W}x{H} -> {(W // 4) * (H // 4) * 16 / 2**20:.1f} MiB ...")
    return bytes(bc7encoder.encode_image_rgba(rgba, uber_level=1, perceptual=False))


def encode_astc(rgba: np.ndarray, block: int = 4, quality: str = "medium") -> bytes:
    try:
        from astc_encoder import (
            ASTCConfig, ASTCContext, ASTCImage,
            ASTCProfile, ASTCType, ASTCSwizzle, ASTCQualityPreset,
            ASTCSwizzleComponentSelector as Sel,
        )
    except ImportError:
        raise SystemExit("astc-encoder-py not installed (pip install astc-encoder-py).")
    import os as _os
    qmap = {
        "fastest": ASTCQualityPreset.FASTEST, "fast": ASTCQualityPreset.FAST,
        "medium": ASTCQualityPreset.MEDIUM, "thorough": ASTCQualityPreset.THOROUGH,
    }
    H, W, _ = rgba.shape
    print(f"[astc] encoding {W}x{H} block {block}x{block} ({quality}) -> "
          f"{(W // block) * (H // block) * 16 / 2**20:.1f} MiB ...")
    cfg = ASTCConfig(ASTCProfile.LDR, block, block, 1, qmap[quality])
    ctx = ASTCContext(cfg, max(1, _os.cpu_count() or 4))
    src = ASTCImage(ASTCType.U8, W, H, 1, rgba.tobytes())
    sw = ASTCSwizzle(Sel.R, Sel.G, Sel.B, Sel.A)
    return bytes(ctx.compress(src, sw))


def write_nat2(out_path: Path, probes_norm: np.ndarray, payload: bytes,
               Ht: int, Wt: int, atlas_format: int, kernel_type: int,
               uv_extent: float, sh_bias: float, res_bias: float,
               atlas_scale: float, atlas_offset: float, flags: int = 0):
    """NAT2 with the standard 64-byte header. Reuses existing slots:

      n_layers = 1, layer_h = Ht      -- single layer; 8192 <= maxTextureDimension2D
                                         so proberes needs NO striping (unlike the
                                         baked 4096x48576 atlas)
      num_rects = N                   -- but the rects block is stride 6, not 4.
                                         atlas_format disambiguates.
      atlas_scale/offset (slots 12/13) -- dequant pair, already in the format
      flags (the former _pad u32)      -- bit 0: WSR — rects stride is 7, the
                                         7th float is the per-surfel occlusion
                                         (sigmoid-activated, in [0,1]).
                                         bits 8-15: mip_count (0/1 = no chain).
                                         Payload is then levels concatenated
                                         L0..Ln, each 16 B per 4x4 block.
    """
    N = probes_norm.shape[0]
    stride = probes_norm.shape[1]
    assert stride == (7 if (flags & 1) else 6), (stride, flags)
    hdr = struct.pack(
        "<IIIIIfII fffI ffII",
        Wt, Ht, 4, kernel_type,
        N, uv_extent, 0, atlas_format,
        sh_bias, res_bias, 1.0, Ht,          # compact_mult=1.0, layer_h
        atlas_scale, atlas_offset, 1, flags, # n_layers=1, flags (ex-_pad)
    )
    assert len(hdr) == 64, len(hdr)

    with open(out_path, "wb") as f:
        f.write(b"NAT2")
        f.write(hdr)
        # layer_cuts for the single layer: [0, Ht]
        f.write(struct.pack("<II", 0, Ht))
        # probes, stride 6 (or 7 with WSR occ) fp32, affine ALREADY divided by tex_res
        f.write(probes_norm.astype(np.float32).tobytes())
        f.write(payload)

    mb = out_path.stat().st_size / 2**20
    print(f"[nat2] {out_path}  {mb:.1f} MiB  ({Wt}x{Ht}, 1 layer, {N:,} probes stride {stride}, "
          f"format {atlas_format}, flags {flags})")


def main():
    ap = argparse.ArgumentParser(description="Export a proberes checkpoint to NAT2")
    ap.add_argument("model_path", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--ckpt", default=None, help="checkpoint filename (default: newest ngp_*.pth)")
    ap.add_argument("--format", choices=["bc7", "astc"], default="bc7")
    ap.add_argument("--astc-block", type=int, default=4)
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--mips", action="store_true",
                    help="build a mip chain (2x2 box filter on the FLOAT atlas, shared "
                         "dequant pair). +33%% payload; enables trilinear minification "
                         "in the viewer and smooths the low-pass pop.")
    ap.add_argument("--mip-levels", type=int, default=0,
                    help="cap the chain length (0 = down to 4x4)")
    ap.add_argument("--kernel", default="beta_scaled", choices=list(KERNEL_MAP))
    ap.add_argument("--uv-extent", type=float, default=4.0)
    ap.add_argument("--sh-bias", type=float, default=0.5)
    ap.add_argument("--res-bias", type=float, default=0.0)
    ap.add_argument("--wsr-ply", type=Path, default=None,
                    help="WSR finetune PLY (point_cloud.ply with the wsr_occ column). "
                         "Appends sigmoid(wsr_occ) as a 7th float per probe record "
                         "and sets header flag bit 0 (viewer ?wsr=1 mode).")
    args = ap.parse_args()

    ckpt = (args.model_path / args.ckpt) if args.ckpt else \
        max(args.model_path.glob("ngp_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"[load] {ckpt}")
    probes, pixels = load_probe_state(ckpt)
    Ht, Wt, _ = pixels.shape
    print(f"[load] probes {probes.shape}  atlas {pixels.shape}  "
          f"nonzero {100 * (np.abs(pixels).sum(-1) > 1e-6).mean():.2f}%")

    # Pre-divide by tex_res so the shader does uv = base + M*s with no normalisation.
    # Columns are [A00, A01, A10, A11, t0, t1]; x-terms scale by Wt, y-terms by Ht.
    probes_norm = probes.copy()
    probes_norm[:, [0, 1, 4]] /= float(Wt)   # A00, A01, t0  -> x / width
    probes_norm[:, [2, 3, 5]] /= float(Ht)   # A10, A11, t1  -> y / height

    flags = 0
    if args.wsr_ply is not None:
        from plyfile import PlyData
        ply = PlyData.read(str(args.wsr_ply))
        el = ply.elements[0]
        names = [p.name for p in el.properties]
        if "wsr_occ" not in names:
            raise SystemExit(f"{args.wsr_ply} has no wsr_occ column — not a WSR finetune PLY?")
        occ_logit = np.asarray(el["wsr_occ"], dtype=np.float32)
        if occ_logit.shape[0] != probes_norm.shape[0]:
            raise SystemExit(f"wsr_occ count {occ_logit.shape[0]} != probes {probes_norm.shape[0]}")
        occ = 1.0 / (1.0 + np.exp(-occ_logit))
        probes_norm = np.concatenate([probes_norm, occ[:, None]], axis=1)  # stride 7
        flags |= 1
        print(f"[wsr] occ appended: mean {occ.mean():.4f}  p10 {np.percentile(occ,10):.4f} "
              f"p90 {np.percentile(occ,90):.4f}")

    rgba, scale, offset = quantise_atlas(pixels)
    enc = (lambda r: encode_bc7(r)) if args.format == "bc7" else \
          (lambda r: encode_astc(r, args.astc_block, args.quality))
    fmt = ATLAS_FORMAT_PROBE_BC7 if args.format == "bc7" else ATLAS_FORMAT_PROBE_ASTC

    if args.mips:
        levels = build_mip_pyramid(pixels, args.mip_levels)
        del pixels
        chunks = [enc(rgba)]                       # L0 reuses the already-quantised codes
        del rgba
        for lvl in levels[1:]:
            chunks.append(enc(quantise_level(lvl, scale, offset)))
        payload = b"".join(chunks)
        mip_count = len(chunks)
        print(f"[mip] {mip_count} levels, "
              f"{' + '.join(f'{len(c)/2**20:.1f}' for c in chunks)} MiB "
              f"= {len(payload)/2**20:.1f} MiB "
              f"(+{100*(len(payload)/len(chunks[0])-1):.0f}% vs L0 alone)")
        flags |= (mip_count & 0xFF) << 8
    else:
        del pixels
        payload = enc(rgba)
        del rgba

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_nat2(args.output, probes_norm, payload, Ht, Wt, fmt,
               KERNEL_MAP[args.kernel], args.uv_extent,
               args.sh_bias, args.res_bias, scale, offset, flags=flags)


if __name__ == "__main__":
    main()
