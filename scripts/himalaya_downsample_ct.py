"""Build a downsampled uint16 volume from a HiP-CT raw JP2 slice stack on LSDF.

The raw stacks are ~800 GB uncompressed (8200 slices of 7604x6620 uint16), which
neither fits locally nor transfers in reasonable time over the ~9 MB/s sshfs link.
Reading every Nth slice and block-mean reducing in-plane by N gives an isotropic
1/N volume while transferring only 1/N of the bytes.

In-plane is averaged (proper anti-aliasing); z is subsampled, since averaging in z
would require fetching every slice and defeat the point.
"""
import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def block_mean_2d(a, f):
    """Block-average `a` by factor f, cropping the ragged edge."""
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def load_one(args):
    path, factor = args
    with Image.open(path) as im:
        a = np.asarray(im)
    return block_mean_2d(a.astype(np.float32), factor).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of .jp2 slices")
    ap.add_argument("--out", required=True, help="output .h5")
    ap.add_argument("--factor", type=int, default=16)
    ap.add_argument("--voxel-um", type=float, default=8.518)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".jp2"))
    if not files:
        sys.exit(f"no .jp2 files in {args.src}")
    sel = [os.path.join(args.src, f) for f in files[:: args.factor]]
    print(f"{len(files)} slices -> {len(sel)} at every {args.factor}", flush=True)

    # Probe one slice for the output shape.
    probe = load_one((sel[0], args.factor))
    ny, nx = probe.shape
    nz = len(sel)
    nbytes = nz * ny * nx * 2
    print(f"output {nz} x {ny} x {nx} uint16 = {nbytes / 1e6:.0f} MB", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with h5py.File(args.out, "w") as h5:
        dset = h5.create_dataset(
            "volume", shape=(nz, ny, nx), dtype=np.uint16,
            chunks=(1, min(ny, 256), min(nx, 256)), compression="lzf",
        )
        vox = args.voxel_um * args.factor
        dset.attrs["voxel_size_um"] = vox
        dset.attrs["downsample_factor"] = args.factor
        dset.attrs["source_dir"] = args.src
        dset.attrs["note"] = "in-plane block-mean; z subsampled"

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, plane in enumerate(ex.map(load_one, ((p, args.factor) for p in sel), chunksize=1)):
                dset[i] = plane
                if i % 25 == 0 or i == nz - 1:
                    el = time.time() - t0
                    rate = (i + 1) / el
                    print(f"  {i + 1}/{nz}  {el:6.1f}s  {rate:.2f} slice/s  "
                          f"eta {(nz - i - 1) / max(rate, 1e-9):6.0f}s", flush=True)

    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
