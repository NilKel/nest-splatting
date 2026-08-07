"""Sanity-check a volume by dumping its three central orthogonal slices as one PNG.

This is NOT a volume rendering -- no ray marching, transfer function, or lighting.
It just extracts the middle plane along each axis and contrast-stretches it, which is
enough to confirm a converted volume is right-way-up, correctly shaped, and not garbage.
"""
import argparse

import h5py
import numpy as np
from PIL import Image


def stretch(a, lo_pct=0.5, hi_pct=99.5):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".h5 with a 'volume' dataset")
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--gap", type=int, default=10, help="pixel gap between panels")
    args = ap.parse_args()

    with h5py.File(args.src, "r") as f:
        v = f["volume"][:]
    nz, ny, nx = v.shape

    panels = [stretch(v[nz // 2]),        # axial   (y, x)
              stretch(v[:, ny // 2, :]),  # coronal (z, x)
              stretch(v[:, :, nx // 2])]  # sagittal(z, y)

    H = max(p.shape[0] for p in panels)
    W = sum(p.shape[1] for p in panels) + args.gap * (len(panels) - 1)
    canvas = np.zeros((H, W), np.float32)
    x = 0
    for p in panels:
        canvas[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + args.gap

    Image.fromarray((canvas * 255).astype(np.uint8)).save(args.out)
    print(f"volume {nz} x {ny} x {nx} (z,y,x)")
    print(f"panels axial/coronal/sagittal: {[p.shape for p in panels]}")
    print(f"percentiles (1/50/99): {np.percentile(v, [1, 50, 99]).round(0)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
