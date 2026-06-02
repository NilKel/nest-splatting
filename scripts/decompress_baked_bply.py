"""
Decoder for the .bply 8-bit PTQ format produced by compress_baked_ply.py.
Used for round-trip testing — restores the original PLY column layout
(dequantized values + dropped columns re-added as constants/zeros) so
existing render scripts work unchanged.
"""
import argparse, os, struct, sys, time
import numpy as np
from plyfile import PlyData, PlyElement


MAGIC = b"BPLY"


def load_bply(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == MAGIC, f"not a BPLY file: {path}"
    version = struct.unpack_from("<I", buf, 4)[0]
    assert version == 1, f"unsupported BPLY version {version}"
    n_gauss = struct.unpack_from("<I", buf, 8)[0]
    n_cols  = struct.unpack_from("<I", buf, 12)[0]
    ap_level_default = struct.unpack_from("<f", buf, 16)[0]
    # 16-byte reserved at offset 20..36
    header_end = 36 + n_cols * 36
    columns = []
    p = 36
    for _ in range(n_cols):
        name = buf[p:p+16].rstrip(b"\x00").decode("ascii"); p += 16
        dtype = buf[p]; p += 1
        p += 3                                                          # pad
        data_off = struct.unpack_from("<I", buf, p)[0]; p += 4
        data_size = struct.unpack_from("<I", buf, p)[0]; p += 4
        mn = struct.unpack_from("<f", buf, p)[0]; p += 4
        mx = struct.unpack_from("<f", buf, p)[0]; p += 4
        columns.append({"name": name, "dtype": dtype, "data_off": data_off,
                        "data_size": data_size, "min": mn, "max": mx})
    payload = buf[header_end:]
    return n_gauss, columns, payload, ap_level_default


def dequant_column(col, payload, n_gauss):
    data_bytes = payload[col["data_off"]:col["data_off"]+col["data_size"]]
    if col["dtype"] == 1:                                               # fp16
        arr = np.frombuffer(data_bytes, dtype=np.float16).astype(np.float32)
    elif col["dtype"] == 2:                                             # u8 dequant
        u = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32)
        arr = u / 255.0 * (col["max"] - col["min"]) + col["min"]
    else:
        raise ValueError(f"unknown dtype {col['dtype']} for {col['name']}")
    assert arr.shape[0] == n_gauss, f"col {col['name']} bad len {arr.shape[0]} vs {n_gauss}"
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to .bply")
    p.add_argument("--output", required=True, help="Path to write reconstructed .ply")
    p.add_argument("--add_normals", action="store_true", default=True,
                   help="Pad nx/ny/nz columns (zero) so the resulting .ply matches "
                        "the standard baked.ply schema.")
    args = p.parse_args()

    n_gauss, columns, payload, ap_level_default = load_bply(args.input)
    print(f"[LOAD] {args.input}: {n_gauss:,} Gausses, {len(columns)} columns")

    # Re-derive arrays.
    data = {}
    for col in columns:
        data[col["name"]] = dequant_column(col, payload, n_gauss)
    if args.add_normals:
        for nm in ("nx", "ny", "nz"):
            if nm not in data:
                data[nm] = np.zeros(n_gauss, dtype=np.float32)
    if "ap_level" not in data:
        data["ap_level"] = np.full(n_gauss, ap_level_default, dtype=np.float32)

    # PLY schema order (must match what existing loaders expect).
    desired_order = ["x", "y", "z", "nx", "ny", "nz",
                     "f_dc_0", "f_dc_1", "f_dc_2"]
    desired_order += [f"f_rest_{i}" for i in range(45)]
    desired_order += ["opacity", "scale_0", "scale_1",
                      "rot_0", "rot_1", "rot_2", "rot_3"]
    if "shape" in data:
        desired_order.append("shape")
    # SV columns: emit any sv_* in numeric order.
    sv_cols = sorted([k for k in data.keys() if k.startswith("sv_")],
                     key=lambda s: (s.split("_")[0:2], int(s.rsplit("_",1)[-1]) if s.rsplit("_",1)[-1].isdigit() else 0))
    desired_order += sv_cols
    if "ap_level" in data:
        desired_order.append("ap_level")
    # Add any other unknowns to the end.
    extras = [k for k in data.keys() if k not in desired_order]
    desired_order += extras

    # Filter to only existing columns and build the structured array.
    final_cols = [k for k in desired_order if k in data]
    dtype = [(k, "f4") for k in final_cols]
    structured = np.empty(n_gauss, dtype=dtype)
    for k in final_cols:
        structured[k] = data[k].astype(np.float32)

    el = PlyElement.describe(structured, "vertex")
    PlyData([el], byte_order="<").write(args.output)
    out_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[WRITE] {args.output}  ({out_mb:.2f} MB, {len(final_cols)} columns)")


if __name__ == "__main__":
    main()
