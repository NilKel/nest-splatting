"""
Hole-fill + reorient a proxy mesh so the WebGPU / CUDA mesh-cull depth
attachment doesn't get inf-holes in the silhouette (the source of
view-dependent popping we're chasing).

Pipeline:
  1. Load (Open3D) + basic cleanup: dedup verts/tris, drop degenerate/
     unreferenced, drop non-manifold edges.
  2. Trimesh `fill_holes()` (triangulate the boundary loops of open edges).
  3. Trimesh `fix_normals()` + `fix_winding()` (make CCW-outward consistent
     — required for our normal-offset shader to push each vertex outward,
     not inward, and for backface culling to be correct if we enable it).
  4. Optional smoothing (`--smooth_iters N`) — a couple Laplacian iters to
     hide the triangulation seam where holes were filled. Off by default.
  5. Save PLY (binary_little_endian float32 x/y/z + face uint indices).

Reports before/after stats and how many holes were filled. Idempotent —
running it on an already-watertight mesh is a no-op.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def _stats(mesh_o3d: o3d.geometry.TriangleMesh, label: str) -> None:
    v = np.asarray(mesh_o3d.vertices)
    f = np.asarray(mesh_o3d.triangles)
    print(f"  [{label:<18s}] {len(v):>7,} verts   {len(f):>8,} tris   "
          f"watertight={mesh_o3d.is_watertight()}   "
          f"edge_manifold={mesh_o3d.is_edge_manifold()}")


def _to_trimesh(m: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(m.vertices),
                           faces=np.asarray(m.triangles),
                           process=False)


def _from_trimesh(m: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, dtype=np.int32)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source proxy mesh (PLY).")
    p.add_argument("--output", required=True, help="Where to write the repaired mesh (PLY).")
    p.add_argument("--smooth_iters", type=int, default=0,
                   help="Optional Laplacian smoothing iterations on filled regions "
                        "(applied to ALL verts — small values like 1-3 hide fill "
                        "seams without noticeably shrinking the mesh). Default 0.")
    args = p.parse_args()

    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(args.input)
    print(f"Loaded {args.input}")
    _stats(mesh, "loaded")

    # -------- Basic cleanup (Open3D) --------
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # Non-manifold edges (edges shared by >2 tris) confuse hole-fill and
    # winding-fix; drop them. This can open new small holes which fill_holes
    # then patches.
    mesh.remove_non_manifold_edges()
    _stats(mesh, "after cleanup")

    # -------- Watertight hole fill (PyMeshFix — Marco Attene's MeshFix) --------
    # Trimesh's fill_holes uses ear-clipping per boundary loop; misses
    # non-trivial topology (e.g. holes with islands) and returned filled_ok=
    # False on our first pass with only +4K triangles closed. PyMeshFix wraps
    # a robust triangulation + hole-filling pipeline that GUARANTEES a
    # watertight, single-connected-component, manifold output. Slower but
    # actually finishes the job.
    try:
        import pymeshfix
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.triangles, dtype=np.int32)
        mf = pymeshfix.MeshFix(v, f)
        # joincomp=True: merge disconnected components into one.
        # remove_smallest_components=False: keep everything so we don't
        # accidentally drop the object because floaters throw off the
        # largest-component heuristic.
        mf.repair(joincomp=True, remove_smallest_components=False)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mf.points, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(mf.faces, dtype=np.int32)))
        _stats(mesh, "after pymeshfix")
    except ImportError:
        # Fallback to trimesh (partial fill) — better than nothing.
        tm = _to_trimesh(mesh)
        before = len(tm.faces)
        ok = trimesh.repair.fill_holes(tm)
        print(f"  [trimesh fallback]   filled_ok={ok}  tris {before:,} → "
              f"{len(tm.faces):,} (+{len(tm.faces) - before:,})")
        mesh = _from_trimesh(tm)

    # Reorient winding so all normals point OUTWARD consistently. Critical
    # for:
    #   * the vertex-normal offset shader (mesh_depth.wgsl in Halloumi-WS)
    #     — pushes each vertex along ITS outward normal; inconsistent
    #     winding = some pushes go inward, silhouette warps.
    #   * enabling cullMode: 'back' safely in the mesh pipelines later.
    tm = _to_trimesh(mesh)
    trimesh.repair.fix_winding(tm)
    trimesh.repair.fix_normals(tm)
    mesh = _from_trimesh(tm)
    _stats(mesh, "after wind-fix")

    # -------- Optional smoothing --------
    if args.smooth_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(args.smooth_iters))
        _stats(mesh, f"after smooth({args.smooth_iters})")

    # Recompute + normalize vertex normals in-place. Downstream code that
    # doesn't recompute them (or uses vertex normals directly from PLY) will
    # get sane outward-pointing unit vectors.
    mesh.compute_vertex_normals(normalized=True)

    # -------- Save --------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(args.output, mesh,
                                    write_ascii=False,
                                    write_vertex_normals=True)
    print(f"\nSaved → {args.output}  (ok={ok})")
    print(f"Total time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
