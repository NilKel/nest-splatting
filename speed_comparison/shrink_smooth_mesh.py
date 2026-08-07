#!/usr/bin/env python3
"""Shrink + smooth a proxy occlusion mesh so it sits BEHIND the visible
surface — leaves the front-face Gauss ('cheating' fuzzy surface Gauss
that pick up background color) un-occluded on Z-cull.

Pipeline:
  1. Compute vertex normals (outward, from TSDF marching cubes).
  2. Shift each vertex along -normal by `--shrink` metres (erosion —
     the surface moves INTO the object body → depth from any external
     camera grows → occlusion happens later → less content occluded).
  3. Apply Taubin smoothing (edge-preserving, no further shrink) so
     the shifted mesh isn't bumpy.

Usage:
    conda run -n nest_splatting python speed_comparison/shrink_smooth_mesh.py \
        --input  speed_comparison/proxy_meshes/bonsai/proxy_mesh_cleaned.ply \
        --output speed_comparison/proxy_meshes/bonsai_median_shrunk/proxy_mesh_cleaned.ply \
        --shrink 0.03 --smooth 20
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shrink", type=float, default=0.03,
                    help="Erosion distance in metres. Larger = mesh pulls "
                         "further behind the visible surface = safer.")
    ap.add_argument("--smooth", type=int, default=20,
                    help="Taubin smoothing iterations. Edge-preserving; won't "
                         "shrink further.")
    ap.add_argument("--taubin_lambda", type=float, default=0.5)
    ap.add_argument("--taubin_mu", type=float, default=-0.53)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(args.input))
    print(f"[shrink] input  {args.input.name}: {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris")
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    norms = np.asarray(mesh.vertex_normals)

    # Erode along −normal (into the object). Normals from marching cubes
    # point OUTWARD from surface, so subtracting shifts INWARD.
    if args.shrink != 0.0:
        verts_new = verts - args.shrink * norms
        mesh.vertices = o3d.utility.Vector3dVector(verts_new)
        print(f"[shrink] eroded vertices by {args.shrink*1000:.1f} mm along -normal")

    # Smooth. Taubin lambda>0 shrinks, mu<0 expands — combined preserves volume.
    if args.smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth,
                                          lambda_filter=args.taubin_lambda,
                                          mu=args.taubin_mu)
        print(f"[shrink] Taubin smoothed {args.smooth} iters "
              f"(λ={args.taubin_lambda}, μ={args.taubin_mu})")

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    print(f"[shrink] wrote {args.output}: {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris")


if __name__ == "__main__":
    main()
