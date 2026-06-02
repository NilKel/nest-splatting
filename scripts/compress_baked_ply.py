"""
PTQ-only 8-bit Min-Max compression for baked.ply → compact .bply format.

For each PLY vertex column we store either:
- FP16  (xyz: position needs sub-mm precision per the Compact3DGS paper)
- u8 with per-column [min, max] (everything else; ~4× shrink vs FP32)

Columns we drop entirely:
- normals (nx, ny, nz): recomputable from rot per fragment, viewer doesn't need them.
- ap_level: always 24 (sentinel "all levels active"); restored at load time.

Resulting `.bply` is fully self-describing (column names + dtypes in header)
so the same loader handles any scene's bake (different SH degrees, different
SV/SB K, etc.).

Binary layout (little-endian):

    [4]  magic = "BPLY"
    [4]  version = 1 (u32)
    [4]  num_gauss (u32)
    [4]  num_columns (u32)
    [4]  ap_level_default (f32) — restored to dropped ap_level column
    [16] reserved
    for each column (16 bytes each, packed):
        [16]  name (ascii, null-padded)
        [1]   dtype: 1=fp16, 2=u8
        [3]   _pad
        [4]   data_offset (u32 — byte offset from end-of-header to column data)
        [4]   data_size_bytes (u32)
        [4]   min (f32)
        [4]   max (f32)
    [variable] payload — each column's array contiguous, in declaration order

Decoder: read header, mmap or slice payload; dequant u8 → float via
    f = u/255 * (max - min) + min
and fp16 via standard half→float.

For the room scene:
  N=75,207; ~88 stored columns
  Size: 75,207 × (3 fp16 + ~85 u8) + header ≈ 7.6 MB (vs 32 MB FP32 PLY)
"""
import argparse, os, struct, sys, time
import numpy as np
from plyfile import PlyData


MAGIC = b"BPLY"
VERSION = 1

# Columns to drop entirely.
DROP_COLUMNS = {"nx", "ny", "nz", "ap_level"}
# Columns kept as FP16 (high-precision required, paper-style).
FP16_COLUMNS = {"x", "y", "z"}


def quantize_u8(arr_f32):
    mn = float(arr_f32.min())
    mx = float(arr_f32.max())
    if mx <= mn:
        # degenerate column — all same value
        u = np.zeros(arr_f32.shape, dtype=np.uint8)
    else:
        u = np.clip(np.round((arr_f32 - mn) / (mx - mn) * 255.0), 0, 255).astype(np.uint8)
    return u, mn, mx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to baked.ply")
    p.add_argument("--output", required=True, help="Path to write .bply")
    p.add_argument("--ap_level_default", type=float, default=24.0,
                   help="Constant restored to ap_level column at load time.")
    args = p.parse_args()

    print(f"[LOAD] {args.input}")
    t0 = time.time()
    ply = PlyData.read(args.input)
    v = ply["vertex"]
    n = v.count
    print(f"  {n:,} Gausses, {len(v.properties)} columns")

    columns = []                                                       # (name, data_u8_or_fp16_bytes, dtype, mn, mx)
    total_bytes_data = 0
    drop_log = []
    for prop in v.properties:
        name = prop.name
        if name in DROP_COLUMNS:
            drop_log.append(name); continue
        col = np.asarray(v[name], dtype=np.float32)
        if name in FP16_COLUMNS:
            data = col.astype(np.float16).tobytes()
            mn = float(col.min()); mx = float(col.max())
            dtype = 1
        else:
            u, mn, mx = quantize_u8(col)
            data = u.tobytes()
            dtype = 2
        columns.append((name, data, dtype, mn, mx))
        total_bytes_data += len(data)

    print(f"  kept {len(columns)} columns, dropped {len(drop_log)} ({', '.join(drop_log)})")

    # Build the header
    HEADER_FIXED = 4 + 4 + 4 + 4 + 4 + 16                              # magic, version, num_gauss, num_columns, ap_level_default, reserved
    COL_DESC_SIZE = 16 + 1 + 3 + 4 + 4 + 4 + 4                         # 36 bytes per column descriptor
    header_size = HEADER_FIXED + COL_DESC_SIZE * len(columns)
    print(f"[PACK] header = {header_size:,} B, data = {total_bytes_data/1024/1024:.2f} MB")

    # Compute offsets (relative to start of payload — i.e. byte 0 of data section)
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", n)
    out += struct.pack("<I", len(columns))
    out += struct.pack("<f", float(args.ap_level_default))
    out += b"\x00" * 16                                                # reserved
    offset = 0
    payload_parts = []
    for name, data, dtype, mn, mx in columns:
        name_bytes = name.encode("ascii")
        if len(name_bytes) > 16:
            raise ValueError(f"column name '{name}' > 16 bytes; bump COL_DESC_SIZE")
        name_padded = name_bytes + b"\x00" * (16 - len(name_bytes))
        out += name_padded
        out += struct.pack("<B", dtype)
        out += b"\x00" * 3                                             # pad
        out += struct.pack("<I", offset)
        out += struct.pack("<I", len(data))
        out += struct.pack("<f", mn)
        out += struct.pack("<f", mx)
        payload_parts.append(data)
        offset += len(data)

    out += b"".join(payload_parts)

    print(f"[WRITE] {args.output}")
    with open(args.output, "wb") as f:
        f.write(out)
    total_mb = len(out) / 1024 / 1024
    orig_mb = os.path.getsize(args.input) / 1024 / 1024
    print(f"  written {total_mb:.2f} MB  (vs {orig_mb:.2f} MB original = {total_mb/orig_mb*100:.1f}%)")
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
