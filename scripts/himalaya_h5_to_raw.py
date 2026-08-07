"""Export a downsampled HiP-CT h5 volume to raw + MetaImage (.mhd) for volume renderers.

Writes x-fastest uint16 little-endian, which is what essentially every volume
renderer expects, plus a .mhd header so the result is directly loadable by ITK/
3D Slicer/ParaView if the target renderer wants something else and a conversion
step is needed.
"""
import argparse
import json
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="input .h5 with a 'volume' dataset")
    ap.add_argument("--out-prefix", required=True, help="output path prefix (no extension)")
    ap.add_argument("--float32", action="store_true",
                    help="also write a normalized [0,1] float32 raw")
    args = ap.parse_args()

    with h5py.File(args.src, "r") as f:
        d = f["volume"]
        vol = d[:]
        vox_um = float(d.attrs.get("voxel_size_um", 0.0))

    nz, ny, nx = vol.shape
    vox_mm = vox_um / 1000.0
    lo, hi = int(vol.min()), int(vol.max())
    print(f"volume {nz} x {ny} x {nx} (z,y,x)  uint16  range [{lo}, {hi}]  "
          f"voxel {vox_um} um")

    raw_path = args.out_prefix + ".raw"
    vol.astype("<u2").tofile(raw_path)
    print(f"wrote {raw_path} ({os.path.getsize(raw_path)/1e6:.0f} MB, uint16 LE, x fastest)")

    mhd_path = args.out_prefix + ".mhd"
    with open(mhd_path, "w") as fh:
        fh.write(
            "ObjectType = Image\nNDims = 3\nBinaryData = True\n"
            "BinaryDataByteOrderMSB = False\nCompressedData = False\n"
            "TransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = 0 0 0\n"
            "CenterOfRotation = 0 0 0\nAnatomicalOrientation = RAI\n"
            f"ElementSpacing = {vox_mm:.6f} {vox_mm:.6f} {vox_mm:.6f}\n"
            f"DimSize = {nx} {ny} {nz}\n"
            "ElementType = MET_USHORT\n"
            f"ElementDataFile = {os.path.basename(raw_path)}\n"
        )
    print(f"wrote {mhd_path}")

    if args.float32:
        f32 = ((vol.astype(np.float32) - lo) / max(hi - lo, 1)).astype("<f4")
        p = args.out_prefix + "_f32.raw"
        f32.tofile(p)
        print(f"wrote {p} ({os.path.getsize(p)/1e6:.0f} MB, float32 normalized to [0,1])")

    meta = {
        "dims_xyz": [nx, ny, nz],
        "voxel_size_um": vox_um,
        "voxel_size_mm": vox_mm,
        "dtype": "uint16_le",
        "axis_order": "x fastest, then y, then z",
        "value_range": [lo, hi],
        "extent_mm": [round(nx * vox_mm, 3), round(ny * vox_mm, 3), round(nz * vox_mm, 3)],
        "source_h5": os.path.abspath(args.src),
    }
    with open(args.out_prefix + "_info.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
