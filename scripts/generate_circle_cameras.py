#!/usr/bin/env python3
"""Generate a circular camera trajectory that follows the training cameras.

Fits a plane (PCA) and a circle (least-squares) to the training camera
positions, then emits N evenly-spaced cameras around the circle, all looking
at the bundle-adjusted scene center. Output is a cameras.json with the same
format as the training cameras.json — drop-in for the WebGPU viewer's preset
switcher, or as input to scripts/render_trajectory.py for a baked MP4.

Convention matches the training cameras.json from this repo:
  c2w columns = [right, up, forward], right-handed (det = +1), where
    forward = (look_at - cam_pos)   (camera looks toward scene)
    up      = plane normal          (PCA-fitted from training cams)
    right   = up x forward
The json stores this c2w row-by-row.

Usage:
    python scripts/generate_circle_cameras.py \\
        --cameras outputs/.../cameras.json \\
        --output  outputs/.../cameras_circle.json \\
        --num_views 60
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def fit_plane(positions: np.ndarray, visual_up: np.ndarray | None = None):
    """PCA fit a plane. Returns (centroid, normal, u_axis, v_axis).
    `normal` is the direction of least variance, aligned to `visual_up` if
    provided. World convention differs by dataset (COLMAP is Y-down, others
    Y-up), so a static heuristic on `normal[1]` is unreliable — pass the
    camera-derived visual-up direction instead."""
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u_axis = vh[0]
    v_axis = vh[1]
    normal = vh[2]
    if visual_up is not None and np.dot(normal, visual_up) < 0:
        normal = -normal
        v_axis = -v_axis  # keep right-handed
    return centroid, normal, u_axis, v_axis


def fit_circle_2d(points_2d: np.ndarray):
    """Algebraic least-squares circle fit.
    Solves (x-a)^2 + (y-b)^2 = r^2 linearized as
        -2*a*x - 2*b*y + k = -(x^2 + y^2),  with k = a^2 + b^2 - r^2.
    Returns (center_xy, radius)."""
    x, y = points_2d[:, 0], points_2d[:, 1]
    A = np.column_stack([-2 * x, -2 * y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, c, k = sol
    r2 = a * a + c * c - k
    return np.array([a, c]), math.sqrt(max(r2, 1e-12))


def intersect_camera_rays(positions: np.ndarray, forwards: np.ndarray) -> np.ndarray:
    """Point closest to all training-camera forward rays.
    min_q sum_i ||(I - d_i d_i^T)(q - p_i)||^2.
    Solution: q = [sum(I - d d^T)]^{-1} [sum(I - d d^T) p].
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, forwards):
        d = d / np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", required=True, type=str,
                    help="Input training cameras.json")
    ap.add_argument("--output", required=True, type=str,
                    help="Output cameras.json")
    ap.add_argument("--num_views", type=int, default=60)
    ap.add_argument("--radius_scale", type=float, default=1.0,
                    help="Scale fitted radius (1.0 = LS fit). Smaller = tighter orbit.")
    ap.add_argument("--elevation_scale", type=float, default=1.0,
                    help="Scale the circle's distance from the look-at plane along the normal.")
    ap.add_argument("--elevation_offset", type=float, default=0.0,
                    help="Add this many units of radius to the circle's height along the plane "
                         "normal. Positive = lift the cameras (look down more); negative = drop.")
    ap.add_argument("--tilt_down_deg", type=float, default=0.0,
                    help="Tilt the camera's forward axis down by this many degrees by lowering "
                         "the look-at point by radius*tan(deg) along the plane normal. Works "
                         "regardless of the look-at's fitted height.")
    ap.add_argument("--start_phase", type=float, default=0.0,
                    help="Starting angle in turns [0,1).")
    ap.add_argument("--direction", choices=["ccw", "cw"], default="ccw",
                    help="Orbit direction when viewed from the +normal side.")
    ap.add_argument("--align_first", action="store_true",
                    help="Set start_phase so view 0 is closest to the first training camera.")
    ap.add_argument("--look_at_mode", choices=["rays", "centroid"], default="rays",
                    help="rays: intersect training-camera rays (best for in-distribution). "
                         "centroid: just use the mean position projected onto plane.")
    args = ap.parse_args()

    with open(args.cameras) as f:
        cams = json.load(f)
    if not cams:
        sys.exit("[error] input cameras.json is empty")

    positions = np.array([c["position"] for c in cams], dtype=np.float64)
    # Forward = camera's local +Z axis in world coords. In the json's row-major
    # c2w, col 2 in math sense is (M[0,2], M[1,2], M[2,2]).
    forwards = np.array([
        [c["rotation"][0][2], c["rotation"][1][2], c["rotation"][2][2]]
        for c in cams
    ], dtype=np.float64)
    # Camera-local +Y axis in world coords (c2w col 1). COLMAP convention
    # has local +Y = image-down, so visual-up = -col1. Averaging across
    # training cameras gives a robust visual-up estimate that doesn't depend
    # on the world's Y sign convention (Y-up vs Y-down datasets).
    local_ys = np.array([
        [c["rotation"][0][1], c["rotation"][1][1], c["rotation"][2][1]]
        for c in cams
    ], dtype=np.float64)
    visual_up = -local_ys.mean(axis=0)
    visual_up = visual_up / np.linalg.norm(visual_up)

    centroid, normal, u, v = fit_plane(positions, visual_up=visual_up)

    # Project training cams into the plane basis (u, v) and fit a circle.
    pts2 = np.stack([(positions - centroid) @ u, (positions - centroid) @ v], axis=1)
    c2, radius = fit_circle_2d(pts2)
    circle_center = centroid + c2[0] * u + c2[1] * v

    if args.look_at_mode == "rays":
        look_at = intersect_camera_rays(positions, forwards)
    else:
        look_at = centroid

    # Optionally re-elevate the circle along the plane normal relative to look_at.
    if args.elevation_scale != 1.0:
        proj = (circle_center - look_at) @ normal
        circle_center = look_at + normal * proj * args.elevation_scale

    radius *= args.radius_scale

    if args.elevation_offset != 0.0:
        circle_center = circle_center + normal * (radius * args.elevation_offset)

    if args.tilt_down_deg != 0.0:
        # Drop the look-at along the (visual) down direction so each camera's
        # forward tilts down by tilt_down_deg. `normal` now points visual-up
        # (aligned to mean training-camera up), so visual-down = -normal.
        # drop ≈ radius * tan(deg) for a camera at orbit radius from look_at.
        drop = radius * math.tan(math.radians(args.tilt_down_deg))
        look_at = look_at - normal * drop

    print(f"[fit] {len(cams)} training cameras")
    print(f"[fit] plane normal = {normal.round(4).tolist()}")
    print(f"[fit] circle center = {circle_center.round(4).tolist()}, radius = {radius:.4f}")
    print(f"[fit] look-at = {look_at.round(4).tolist()}")
    res2 = np.linalg.norm(pts2 - c2, axis=1) - radius / args.radius_scale
    print(f"[fit] residual: mean |r-fit|/r = {np.abs(res2).mean() / (radius / args.radius_scale):.3%}, "
          f"max = {np.abs(res2).max() / (radius / args.radius_scale):.3%}")

    if args.align_first:
        # Angle of the first training cam in the (u, v) plane (relative to circle center).
        rel = positions[0] - circle_center
        theta0 = math.atan2(rel @ v, rel @ u)
        args.start_phase = (theta0 / (2 * math.pi)) % 1.0
        print(f"[align] aligned start_phase to first training cam: {args.start_phase:.4f}")

    ref = cams[0]
    W, H = ref["width"], ref["height"]
    fx, fy = ref["fx"], ref["fy"]
    sign = 1.0 if args.direction == "ccw" else -1.0

    out_cams = []
    for i in range(args.num_views):
        theta = (args.start_phase + i / args.num_views * sign) * 2 * math.pi
        pos = circle_center + radius * (math.cos(theta) * u + math.sin(theta) * v)

        forward = look_at - pos
        forward /= np.linalg.norm(forward)
        # Training c2w convention (COLMAP-style): col 0 = image-right,
        # col 1 = image-DOWN, col 2 = forward. `normal` points visual-up, so
        # image-down = -normal. Build the right axis from the right-hand rule
        # right x down = forward  ⇒  right = down x forward = forward x normal.
        right = np.cross(forward, normal)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = u.copy()
        else:
            right /= rn
        # Recompute image-down so it's exactly perpendicular to (right, fwd).
        cam_down = np.cross(forward, right)
        cam_down /= np.linalg.norm(cam_down)

        # c2w columns = [right, image-down, forward]
        R = np.stack([right, cam_down, forward], axis=1)

        out_cams.append({
            "id": i,
            "img_name": f"circle_{i:04d}",
            "width": int(W),
            "height": int(H),
            "position": pos.tolist(),
            "rotation": R.tolist(),
            "fy": float(fy),
            "fx": float(fx),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_cams, f, indent=2)
    print(f"[write] {args.output} ({len(out_cams)} cameras)")


if __name__ == "__main__":
    main()
