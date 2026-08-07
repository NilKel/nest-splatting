"""Convert a raw Cinematic-Renderer 3DGS export into the layout nest-splatting trains on.

CR (via cr2nerf.py -3dgs) emits `images/`, `transforms.json` and `pointcloud.ply` in the
volume's own world coordinates. `Scene()` takes the Blender branch when
`transforms_train.json` exists, and that branch additionally requires `points3d.ply`.

Three things have to happen:
  1. Normalize world coords so the scene fits the hash grid's [-1.5, 1.5] range --
     `x_new = scale * (x_old - center)`, center = point-cloud bbox midpoint and
     scale chosen so the largest half-extent lands on --half-extent (default 1.4).
     Applied to the point cloud AND to camera translations (rotations are unchanged).
  2. Strip the image extension from `file_path`: readCamerasFromTransforms does
     `frame["file_path"] + extension`, so a path that already ends in .png becomes
     `.png.png` and the load fails.
  3. Split train/test by --test-every (matches the llffhold=8 convention used elsewhere).

The written norm_info.json is what you need to map trained Gaussians back into the
volume's coordinate frame.
"""
import argparse
import json
import os

import numpy as np
from plyfile import PlyData, PlyElement

IMG_EXTS = (".png", ".jpg", ".jpeg", ".exr")


def strip_ext(p):
    for e in IMG_EXTS:
        if p.lower().endswith(e):
            return p[: -len(e)]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with images/, transforms.json, pointcloud.ply")
    ap.add_argument("--half-extent", type=float, default=1.4)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--ply-in", default="pointcloud.ply")
    args = ap.parse_args()

    tf_path = os.path.join(args.src, "transforms.json")
    ply_in = os.path.join(args.src, args.ply_in)
    for p in (tf_path, ply_in):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    v = PlyData.read(ply_in)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    has_rgb = all(c in v.data.dtype.names for c in ("red", "green", "blue"))
    rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
           if has_rgb else np.full((len(xyz), 3), 128, np.uint8))

    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2.0
    scale = args.half_extent / np.max((hi - lo) / 2.0)
    print(f"point cloud: {len(xyz)} pts  bbox {lo.round(3)} .. {hi.round(3)}")
    print(f"center {center.round(6)}  scale {scale:.9f}")

    xyz_n = scale * (xyz - center)
    print(f"normalized bbox {xyz_n.min(0).round(3)} .. {xyz_n.max(0).round(3)}")

    out_ply = os.path.join(args.src, "points3d.ply")
    arr = np.empty(len(xyz_n), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                      ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz_n[:, 0], xyz_n[:, 1], xyz_n[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(out_ply)
    print(f"wrote {out_ply}")

    with open(tf_path) as fh:
        tf = json.load(fh)
    frames = tf["frames"]

    n_fixed = 0
    for f in frames:
        c2w = np.array(f["transform_matrix"], dtype=np.float64)
        c2w[:3, 3] = scale * (c2w[:3, 3] - center)
        f["transform_matrix"] = c2w.tolist()
        s = strip_ext(f["file_path"])
        if s != f["file_path"]:
            n_fixed += 1
        f["file_path"] = s
    pos = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
    print(f"{len(frames)} frames, stripped extension on {n_fixed}")
    print(f"normalized cam radius: mean {np.linalg.norm(pos,axis=1).mean():.3f} "
          f"min {np.linalg.norm(pos,axis=1).min():.3f} max {np.linalg.norm(pos,axis=1).max():.3f}")

    tf["ply_file_path"] = "points3d.ply"
    test = [f for i, f in enumerate(frames) if i % args.test_every == 0]
    train = [f for i, f in enumerate(frames) if i % args.test_every != 0]

    for name, fr in (("transforms.json", frames),
                     ("transforms_train.json", train),
                     ("transforms_test.json", test)):
        d = dict(tf)
        d["frames"] = fr
        with open(os.path.join(args.src, name), "w") as fh:
            json.dump(d, fh)
        print(f"wrote {name}: {len(fr)} frames")

    with open(os.path.join(args.src, "norm_info.json"), "w") as fh:
        json.dump({"center": center.tolist(), "scale": float(scale),
                   "note": f"x_new = scale*(x_old - center); "
                           f"half-extent {args.half_extent} to fit hash range [-1.5,1.5]"},
                  fh, indent=1)
    print("wrote norm_info.json")


if __name__ == "__main__":
    main()
