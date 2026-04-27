#!/usr/bin/env python3
"""
Single-Gaussian, single-camera test for SnugBox / AccuTile math in 2DGS.

We pick one camera + one Gaussian from kitchen/SB_10thr_005w25..., then
*from Python* compute:

  1. transMat T (matches CUDA `compute_transmat`)
  2. (s.x, s.y) at any pixel via the cross-product trick (matches CUDA render kernel)
  3. Empirical projected ellipse bbox by sampling pixels around the projected center
  4. compute_aabb's (point_image, extent) — the 2DGS reference
  5. Cross-product conic (A, B, E, t, p) — the SnugBox candidate

Then we cross-check:
  - Conic center p == compute_aabb's center (== projected origin in nice cases)
  - Conic bbox half-extents (h_x, h_y from Eq 15+16) == empirical max-|dx|, max-|dy|
  - compute_aabb's extent vs both

ALL coords are 2D screen pixel space (verified). T's columns are vec3s
in homogeneous pixel coords.
"""
import os, sys, math, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from arguments import ModelParams


# ------------------------------------------------------------ Math helpers

def quat_to_rotmat(q):
    """q = (w, x, y, z) → 3x3 rotation matrix (column convention matches GLM)."""
    w, x, y, z = q
    n = (w*w + x*x + y*y + z*z) ** 0.5
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def compute_transmat_python(p_orig, scale_xy, quat, viewmatrix, projmatrix, W, H):
    """Replicates CUDA compute_transmat in NumPy. T's columns Tu, Tv, Tw are
    3-vecs in homogeneous pixel coords. T·(s.x, s.y, 1) ~ pixel-h."""
    R = quat_to_rotmat(quat)
    S = np.diag([scale_xy[0], scale_xy[1], 1.0])
    L = R @ S  # 3x3, columns = scaled basis vectors

    # splat2world: 3x4 (3 cols of vec4). col0 = (L[:, 0], 0), col1 = (L[:, 1], 0), col2 = (p_orig, 1)
    splat2world = np.zeros((4, 3), dtype=np.float64)  # numpy row-major: 4 rows × 3 cols
    splat2world[:3, 0] = L[:, 0]
    splat2world[:3, 1] = L[:, 1]
    splat2world[:3, 2] = p_orig
    splat2world[3, 2] = 1.0

    # world2ndc: 4x4. CUDA stores it column-major from projmatrix[i].
    # projmatrix is 16 floats in column-major order (per the CUDA constructor).
    # In Python we receive viewpoint_camera.full_proj_transform which is the
    # PyTorch tensor — it's already 4x4 in row-major numpy (just transposed
    # because PyTorch stores as [row][col]). The CUDA code reads it as cm.
    world2ndc = np.array(projmatrix, dtype=np.float64).reshape(4, 4)
    # viewpoint_camera.full_proj_transform from PyTorch is stored as W2V·P
    # already; we just need to transpose to match CUDA's column-major load.
    # Actually: CUDA does:
    #   glm::mat4 world2ndc = glm::mat4(
    #       projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
    #       projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
    #       ...
    #   );
    # GLM mat4 ctor is column-by-column. So world2ndc[col=0] = (P[0],P[4],P[8],P[12])
    # = first ROW of the row-major matrix. So GLM column-major holds the math
    # transpose of the row-major source. PyTorch's .full_proj_transform is
    # already meant to be column-major in this convention (i.e., when flattened
    # row-major, it gives the transpose of the math matrix).
    # Net: numpy receives it in PyTorch row-major, and that IS the math
    # transpose. To use as math matrix, transpose once.
    world2ndc = world2ndc.T  # now math row-major

    # ndc2pix: 3 cols × 4 rows (mat3x4). CUDA constructs each col as vec4.
    #   col0 = (W/2, 0, 0, (W-1)/2)
    #   col1 = (0, H/2, 0, (H-1)/2)
    #   col2 = (0, 0, 0, 1)
    # As a 4×3 numerical matrix:
    #   [W/2,    0,    0]
    #   [0,      H/2,  0]
    #   [0,      0,    0]
    #   [(W-1)/2, (H-1)/2, 1]
    ndc2pix = np.array([
        [W/2.0, 0.0,    0.0],
        [0.0,   H/2.0,  0.0],
        [0.0,   0.0,    0.0],
        [(W - 1)/2.0, (H - 1)/2.0, 1.0],
    ], dtype=np.float64)

    # CUDA computes: T = transpose(splat2world) * world2ndc * ndc2pix.
    # splat2world has GLM dim mat3x4 (3 cols × 4 rows numerically). Transpose
    # gives mat4x3 (4 cols × 3 rows). Multiplying:
    #   mat4x3 * mat4   = mat4x3 (still 4 cols × 3 rows)
    #   mat4x3 * mat3x4 = mat3 (3×3)
    # In numpy with rows×cols arrays, splat2world is shape (4,3); transpose gives (3,4).
    # (3,4) @ (4,4) = (3,4). (3,4) @ (4,3) = (3,3).
    splat2world_T = splat2world.T  # shape (3, 4)
    T_math = splat2world_T @ world2ndc @ ndc2pix  # shape (3, 3) — math row-major
    # T's columns (in the CUDA / GLM column-major sense) are T_math's columns.
    return T_math, R, L


def cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ], dtype=a.dtype)


def surfel_coord_at_pixel(T, px, py):
    """Replicates CUDA render kernel: cross(k, l)/cross.z."""
    Tu = T[:, 0]; Tv = T[:, 1]; Tw = T[:, 2]
    k = px * Tw - Tu
    l = py * Tw - Tv
    p = cross(k, l)
    if abs(p[2]) < 1e-12:
        return None
    return p[0] / p[2], p[1] / p[2]


# ------------------------------------------------------------ The two methods

def compute_aabb_python(T, cutoff):
    """Python port of CUDA compute_aabb. Returns (p, extent_xy, d, h0)."""
    Tu = T[:, 0]; Tv = T[:, 1]; Tw = T[:, 2]
    k_sq = cutoff * cutoff
    t_vec = np.array([k_sq, k_sq, -1.0])
    d = float(t_vec @ (Tw * Tw))
    if d == 0:
        return None
    f = t_vec / d
    p = np.array([
        float(f @ (Tu * Tw)),
        float(f @ (Tv * Tw)),
    ])
    h0 = p * p - np.array([
        float(f @ (Tu * Tu)),
        float(f @ (Tv * Tv)),
    ])
    h = np.sqrt(np.maximum(h0, 1e-4))
    return p, h, d, h0


def conic_from_T(T, cutoff):
    """SnugBox candidate: conic in screen-pixel coords from cross-product trick.

    cross(k, l) = px·n0 + py·n1 + n2  where n0=Tv×Tw, n1=Tw×Tu, n2=Tu×Tv.
    Disk: cross.x² + cross.y² − k²·cross.z² ≤ 0 → quadratic in (px, py).

    Returns (A, B, E, D, F, G, p, t) where the conic is
        A·px² + 2B·px·py + E·py² + 2D·px + 2F·py + G ≤ 0
    and centered form
        A·dx² + 2B·dx·dy + E·dy² ≤ t,  dx = px - p.x, dy = py - p.y, t = -(D·p.x + F·p.y + G).
    """
    Tu = T[:, 0]; Tv = T[:, 1]; Tw = T[:, 2]
    k_sq = cutoff * cutoff
    n0 = cross(Tv, Tw)
    n1 = cross(Tw, Tu)
    n2 = cross(Tu, Tv)
    A = n0[0]**2 + n0[1]**2 - k_sq * n0[2]**2
    B = n0[0]*n1[0] + n0[1]*n1[1] - k_sq * n0[2]*n1[2]
    E = n1[0]**2 + n1[1]**2 - k_sq * n1[2]**2
    D = n0[0]*n2[0] + n0[1]*n2[1] - k_sq * n0[2]*n2[2]
    F = n1[0]*n2[0] + n1[1]*n2[1] - k_sq * n1[2]*n2[2]
    G = n2[0]**2 + n2[1]**2 - k_sq * n2[2]**2
    det = A * E - B * B
    if det <= 0 or A <= 0 or E <= 0:
        return None
    p_x = (B * F - E * D) / det
    p_y = (B * D - A * F) / det
    # Compute t = -Q(p) using the CROSS-PRODUCT form, not the polynomial
    # form. The polynomial `D·p.x + F·p.y + G` involves huge cancellations
    # between O(1e9) numbers — float64 loses 7+ digits and can come out
    # 2× wrong for elongated ellipses. The cross-product evaluation at p
    # has all O(1) numbers, so cancellation is benign.
    cx_p = p_x * n0[0] + p_y * n1[0] + n2[0]
    cy_p = p_x * n0[1] + p_y * n1[1] + n2[1]
    cz_p = p_x * n0[2] + p_y * n1[2] + n2[2]
    t = -(cx_p*cx_p + cy_p*cy_p - k_sq * cz_p*cz_p)
    return dict(A=A, B=B, E=E, D=D, F=F, G=G, p=(p_x, p_y), t=t,
                disc=B*B - A*E, n0=n0, n1=n1, n2=n2,
                cross_at_p=(cx_p, cy_p, cz_p))


def conic_bbox_extent(c):
    """Half-extent of the axis-aligned bbox of `A·dx²+2B·dx·dy+E·dy²=t` (paper Eq 16)."""
    A, B, E, t, disc = c['A'], c['B'], c['E'], c['t'], c['disc']
    # Half-extent in x: at points where ∂Q/∂y=0 (left/right tangent vertical).
    #   dx² = t·E / (-disc)
    # Half-extent in y: at points where ∂Q/∂x=0.
    #   dy² = t·A / (-disc)
    if disc >= 0 or t <= 0:
        return None
    h_x = math.sqrt(t * E / (-disc))
    h_y = math.sqrt(t * A / (-disc))
    return h_x, h_y


# =============================================================================
# AccuTile (FastGS / Speedy-Splat port)
# =============================================================================

def compute_ellipse_intersection(A, B, E, disc, t, p, isY, coord):
    """Solve the ellipse Q_centered = t for the perpendicular coord.

    isY=False, coord=x: fix x, return the two y values where the ellipse crosses.
    isY=True,  coord=y: fix y, return the two x values where the ellipse crosses.

    Mirrors FastGS's computeEllipseIntersection (auxiliary.h) but with
    (A, B, E) instead of FastGS's (con_o.x, con_o.y, con_o.z).
    """
    p_u = p[1] if isY else p[0]
    p_v = p[0] if isY else p[1]
    coeff = A if isY else E
    h = coord - p_u
    radicand = disc * h * h + t * coeff
    sqrt_term = math.sqrt(max(radicand, 0.0))
    return (
        (-B * h - sqrt_term) / coeff + p_v,
        (-B * h + sqrt_term) / coeff + p_v,
    )


def snugbox_bbox(c, BLOCK_X, BLOCK_Y, grid_x, grid_y):
    """SnugBox: tight rect AABB of the projected ellipse, snapped to tile grid.

    Returns (rect_min, rect_max, bbox_min, bbox_max, bbox_argmin, bbox_argmax, isY)
    in pixel-space (rect in tile-space).
    """
    A, B, E, t, p, disc = c['A'], c['B'], c['E'], c['t'], c['p'], c['disc']
    if A <= 0 or E <= 0 or disc >= 0 or t <= 0:
        return None

    # Eq 16: at the ellipse's xmin/xmax point, dy = ±sqrt(-B²t/(disc·E)) · sign(B)
    # paired with dx = ±h_x. Same for ymin/ymax with A↔E swapped.
    x_term_sq = -(B * B * t) / (disc * A)
    y_term_sq = -(B * B * t) / (disc * E)
    if x_term_sq < 0 or y_term_sq < 0:
        return None
    x_term = math.sqrt(x_term_sq)
    y_term = math.sqrt(y_term_sq)
    # Sign convention from FastGS:
    if B >= 0:
        x_term, y_term = -x_term, -y_term

    # bbox_argmin/argmax: (y at xmin point, x at ymin point) and similar for max.
    bbox_argmin = (p[1] - y_term, p[0] - x_term)
    bbox_argmax = (p[1] + y_term, p[0] + x_term)

    # bbox_min, bbox_max: (xmin, ymin) and (xmax, ymax) of the projected ellipse.
    bbox_min = (
        compute_ellipse_intersection(A, B, E, disc, t, p, True, bbox_argmin[0])[0],
        compute_ellipse_intersection(A, B, E, disc, t, p, False, bbox_argmin[1])[0],
    )
    bbox_max = (
        compute_ellipse_intersection(A, B, E, disc, t, p, True, bbox_argmax[0])[1],
        compute_ellipse_intersection(A, B, E, disc, t, p, False, bbox_argmax[1])[1],
    )

    rect_min = (
        max(0, min(grid_x, int(bbox_min[0] / BLOCK_X))),
        max(0, min(grid_y, int(bbox_min[1] / BLOCK_Y))),
    )
    rect_max = (
        max(0, min(grid_x, int(bbox_max[0] / BLOCK_X) + 1)),
        max(0, min(grid_y, int(bbox_max[1] / BLOCK_Y) + 1)),
    )
    if (rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0:
        return None

    isY = (rect_max[1] - rect_min[1]) < (rect_max[0] - rect_min[0])
    return dict(rect_min=rect_min, rect_max=rect_max,
                bbox_min=bbox_min, bbox_max=bbox_max,
                bbox_argmin=bbox_argmin, bbox_argmax=bbox_argmax,
                isY=isY, x_term=x_term, y_term=y_term)


def accutile_walk(c, sb, BLOCK_X, BLOCK_Y, emit=False):
    """AccuTile: scan-line walk over the ellipse, returning the set of tiles
    it actually crosses (not just rect AABB). Returns set of (tile_x, tile_y).
    """
    A, B, E, t, p, disc = c['A'], c['B'], c['E'], c['t'], c['p'], c['disc']
    rect_min, rect_max = sb['rect_min'], sb['rect_max']
    bbox_min, bbox_max = list(sb['bbox_min']), list(sb['bbox_max'])
    bbox_argmin, bbox_argmax = list(sb['bbox_argmin']), list(sb['bbox_argmax'])
    isY = sb['isY']

    # If isY, we walk along Y and find x-extent for each y-slice.
    # FastGS swaps coordinates so the "u" axis is the iteration axis.
    BLOCK_U = BLOCK_Y if isY else BLOCK_X
    BLOCK_V = BLOCK_X if isY else BLOCK_Y

    if isY:
        rect_min = (rect_min[1], rect_min[0])
        rect_max = (rect_max[1], rect_max[0])
        bbox_min = [bbox_min[1], bbox_min[0]]
        bbox_max = [bbox_max[1], bbox_max[0]]
        bbox_argmin = [bbox_argmin[1], bbox_argmin[0]]
        bbox_argmax = [bbox_argmax[1], bbox_argmax[0]]

    tiles = set()

    intersect_max_line = (bbox_max[1], bbox_min[1])  # sentinel
    min_line = rect_min[0] * BLOCK_U
    if bbox_min[0] <= min_line:
        intersect_min_line = compute_ellipse_intersection(
            A, B, E, disc, t, p, isY, rect_min[0] * BLOCK_U)
    else:
        intersect_min_line = intersect_max_line

    for u in range(rect_min[0], rect_max[0]):
        max_line = min_line + BLOCK_U
        if max_line <= bbox_max[0]:
            intersect_max_line = compute_ellipse_intersection(
                A, B, E, disc, t, p, isY, max_line)

        if min_line <= bbox_argmin[1] < max_line:
            ellipse_min = bbox_min[1]
        else:
            ellipse_min = min(intersect_min_line[0], intersect_max_line[0])

        if min_line <= bbox_argmax[1] < max_line:
            ellipse_max = bbox_max[1]
        else:
            ellipse_max = max(intersect_min_line[1], intersect_max_line[1])

        min_tile_v = max(rect_min[1], min(rect_max[1], int(ellipse_min / BLOCK_V)))
        max_tile_v = min(rect_max[1], max(rect_min[1], int(ellipse_max / BLOCK_V) + 1))

        for v in range(min_tile_v, max_tile_v):
            if isY:
                tiles.add((v, u))
            else:
                tiles.add((u, v))

        intersect_min_line = intersect_max_line
        min_line = max_line

    return tiles


# =============================================================================
# Ground truth: brute-force enumerate tiles by sub-pixel sampling
# =============================================================================

def brute_force_tiles(T, cutoff, BLOCK_X, BLOCK_Y, grid_x, grid_y,
                      samples_per_pixel=2, rect_min=None, rect_max=None):
    """Vectorized: sample sub-pixel positions; mark a tile 'touched' if ANY
    sample inside it has |s|² ≤ cutoff². Restricted to `rect_min..rect_max`
    tiles (passed in tile coords) to avoid scanning the entire image.

    Returns a set of (tile_x, tile_y).
    """
    if rect_min is None:
        rect_min = (0, 0)
    if rect_max is None:
        rect_max = (grid_x, grid_y)
    k_sq = cutoff * cutoff
    Tu, Tv, Tw = T[:, 0], T[:, 1], T[:, 2]
    tiles = set()
    step = 1.0 / samples_per_pixel
    for ty in range(rect_min[1], rect_max[1]):
        for tx in range(rect_min[0], rect_max[0]):
            x0 = tx * BLOCK_X
            y0 = ty * BLOCK_Y
            # Vectorized check: build a grid of sample coords, evaluate all at once.
            xs = np.arange(0.0, BLOCK_X, step) + x0
            ys = np.arange(0.0, BLOCK_Y, step) + y0
            PX, PY = np.meshgrid(xs, ys, indexing='xy')
            kx = PX * Tw[0] - Tu[0]
            ky = PX * Tw[1] - Tu[1]
            kz = PX * Tw[2] - Tu[2]
            lx = PY * Tw[0] - Tv[0]
            ly = PY * Tw[1] - Tv[1]
            lz = PY * Tw[2] - Tv[2]
            cx = ky*lz - kz*ly
            cy = kz*lx - kx*lz
            cz = kx*ly - ky*lx
            mask = np.abs(cz) > 1e-12
            sxsq_ssq = np.zeros_like(cz)
            sxsq_ssq[mask] = (cx[mask]**2 + cy[mask]**2) / (cz[mask]**2)
            if np.any(mask & (sxsq_ssq <= k_sq)):
                tiles.add((tx, ty))
    return tiles


# ------------------------------------------------------------ Main test

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=
        "/home/nilkel/Projects/nest-splatting/outputs/mip_360/kitchen/3D_SH_res/SB_10thr_005w25gLP4levno2f_FRP5k10")
    parser.add_argument("--gauss_idx", type=int, default=10000,
                        help="Single index (used if --gauss_indices not given)")
    parser.add_argument("--gauss_indices", type=str, default=None,
                        help="Comma-separated list of indices to test in one run")
    parser.add_argument("--quick", action="store_true",
                        help="Compact output: only the bbox/AccuTile comparison")
    parser.add_argument("--cam_idx", type=int, default=0,
                        help="Index of test camera (0..N)")
    parser.add_argument("--cutoff", type=float, default=3.5,
                        help="Surfel-space cutoff (k where disk = s.x²+s.y² ≤ k²)")
    args_cli = parser.parse_args()

    # Load args + scene
    with open(os.path.join(args_cli.model_path, "args.pkl"), 'rb') as f:
        train_args = pickle.load(f)
    train_args.model_path = args_cli.model_path
    train_args.eval = True

    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(train_args)

    iteration = max(int(p.split('_')[-1].rstrip('.pth'))
                    for p in os.listdir(args_cli.model_path)
                    if p.startswith('ngp_') and p.endswith('.pth'))

    gaussians = GaussianModel(dataset.sh_degree)
    baked_ply = os.path.join(args_cli.model_path, "baked_atlas/baked.ply")
    gaussians.load_ply(baked_ply)
    gaussians.active_sh_degree = 3

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cams = scene.getTestCameras()
    cam = test_cams[args_cli.cam_idx]
    W, H = int(cam.image_width), int(cam.image_height)
    viewmatrix = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
    projmatrix = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)

    # Determine indices to test
    if args_cli.gauss_indices:
        indices = [int(x) for x in args_cli.gauss_indices.split(',')]
    else:
        indices = [args_cli.gauss_idx]

    # Quick batch: SnugBox + AccuTile vs Rect AABB vs brute-force, no per-Gaussian verbose output.
    if args_cli.quick:
        BLOCK_X, BLOCK_Y = 16, 16
        grid_x = (W + BLOCK_X - 1) // BLOCK_X
        grid_y = (H + BLOCK_Y - 1) // BLOCK_Y
        cutoff = args_cli.cutoff
        print(f"# scene={os.path.basename(args_cli.model_path)} cam={args_cli.cam_idx} cutoff={cutoff}")
        print(f"# image {W}×{H}  grid {grid_x}×{grid_y}")
        print(f"  {'idx':>8} {'rect_n':>8} {'accu_n':>8} {'bf_n':>8} {'missed':>8} {'extra':>8} {'reduction':>10}")
        for idx in indices:
            p_orig = gaussians.get_xyz[idx].detach().cpu().numpy().astype(np.float64)
            scale_xy = gaussians.get_scaling[idx].detach().cpu().numpy().astype(np.float64)
            quat = gaussians.get_rotation[idx].detach().cpu().numpy().astype(np.float64)
            T, _, _ = compute_transmat_python(p_orig, scale_xy, quat, viewmatrix, projmatrix, W, H)
            c = conic_from_T(T, cutoff)
            if c is None:
                print(f"  {idx:>8} {'(degenerate conic)':>40}")
                continue
            sb = snugbox_bbox(c, BLOCK_X, BLOCK_Y, grid_x, grid_y)
            if sb is None:
                print(f"  {idx:>8} {'(empty bbox after clamp)':>40}")
                continue
            accu = accutile_walk(c, sb, BLOCK_X, BLOCK_Y)
            rect = set((tx, ty)
                       for tx in range(sb['rect_min'][0], sb['rect_max'][0])
                       for ty in range(sb['rect_min'][1], sb['rect_max'][1]))
            bf = brute_force_tiles(T, cutoff, BLOCK_X, BLOCK_Y, grid_x, grid_y,
                                   samples_per_pixel=4,
                                   rect_min=sb['rect_min'], rect_max=sb['rect_max'])
            n_rect, n_accu, n_bf = len(rect), len(accu), len(bf)
            missed = len(bf - accu)
            extra = len(accu - bf)
            red = (1.0 - n_accu / n_rect) * 100 if n_rect else 0.0
            print(f"  {idx:>8} {n_rect:>8} {n_accu:>8} {n_bf:>8} {missed:>8} {extra:>8} {red:>9.1f}%")
        return

    # Verbose single-Gaussian path:
    idx = indices[0]
    p_orig = gaussians.get_xyz[idx].detach().cpu().numpy().astype(np.float64)
    scale_xy = gaussians.get_scaling[idx].detach().cpu().numpy().astype(np.float64)
    quat = gaussians.get_rotation[idx].detach().cpu().numpy().astype(np.float64)

    print(f"=== Test setup ===")
    print(f"  scene:  {args_cli.model_path}")
    print(f"  cam:    idx={args_cli.cam_idx}  W×H = {W}×{H}")
    print(f"  Gauss:  idx={idx}")
    print(f"          xyz   = {p_orig}")
    print(f"          scale = {scale_xy}")
    print(f"          quat  = {quat}")
    print(f"  cutoff: {args_cli.cutoff}")

    # Get viewmatrix and projmatrix from camera (PyTorch tensors).
    viewmatrix = cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
    projmatrix = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)

    print(f"\n=== Camera matrices ===")
    print(f"  world_view_transform (4x4) =\n{viewmatrix}")
    print(f"  full_proj_transform  (4x4) =\n{projmatrix}")

    # Build T
    T, R_world, L = compute_transmat_python(p_orig, scale_xy, quat, viewmatrix, projmatrix, W, H)
    Tu, Tv, Tw = T[:, 0], T[:, 1], T[:, 2]
    print(f"\n=== transMat T (3×3, columns = Tu, Tv, Tw) ===")
    print(T)
    print(f"  Tu = {Tu}")
    print(f"  Tv = {Tv}")
    print(f"  Tw = {Tw}")

    # Surfel origin projects to:
    if abs(Tw[2]) > 1e-12:
        proj_origin = (Tw[0]/Tw[2], Tw[1]/Tw[2])
    else:
        proj_origin = (None, None)
    print(f"  Projected surfel origin (Tw.xy/Tw.z) = {proj_origin}")

    # Verify cross-product trick at a few points: surfel origin should give s≈0.
    if proj_origin[0] is not None:
        s = surfel_coord_at_pixel(T, proj_origin[0], proj_origin[1])
        print(f"  s at projected origin = {s} (should be ≈(0,0))")

    # ----------------------------------------------------------------
    # 1. compute_aabb result
    cutoff = args_cli.cutoff
    res = compute_aabb_python(T, cutoff)
    if res is None:
        print("  compute_aabb: degenerate (d=0)")
        return
    (aabb_p, aabb_h, aabb_d, aabb_h0) = res
    print(f"\n=== compute_aabb ===")
    print(f"  d        = {aabb_d:.4e}")
    print(f"  p        = ({aabb_p[0]:.3f}, {aabb_p[1]:.3f})")
    print(f"  h0       = ({aabb_h0[0]:.4e}, {aabb_h0[1]:.4e})")
    print(f"  extent   = ({aabb_h[0]:.3f}, {aabb_h[1]:.3f})")

    # ----------------------------------------------------------------
    # 2. Cross-product conic
    c = conic_from_T(T, cutoff)
    print(f"\n=== Cross-product conic (SnugBox candidate) ===")
    if c is None:
        print("  Conic invalid (det≤0 or A,E≤0)")
        return
    print(f"  A    = {c['A']:.4e}")
    print(f"  B    = {c['B']:.4e}")
    print(f"  E    = {c['E']:.4e}")
    print(f"  D    = {c['D']:.4e}")
    print(f"  F    = {c['F']:.4e}")
    print(f"  G    = {c['G']:.4e}")
    print(f"  disc = {c['disc']:.4e}")
    print(f"  p    = ({c['p'][0]:.6f}, {c['p'][1]:.6f})")
    print(f"  t    = {c['t']:.6e}")
    # Verify t breakdown:
    Dp = c['D'] * c['p'][0]
    Fp = c['F'] * c['p'][1]
    print(f"     D·p.x = {Dp:+.6e}")
    print(f"     F·p.y = {Fp:+.6e}")
    print(f"     G     = {c['G']:+.6e}")
    print(f"     D·p.x + F·p.y + G = {Dp + Fp + c['G']:+.6e}")
    print(f"     -(D·p.x + F·p.y + G) = {-(Dp + Fp + c['G']):+.6e}    [<-- this is t]")

    bbox_h = conic_bbox_extent(c)
    if bbox_h is None:
        print("  conic bbox: invalid")
        return
    h_x, h_y = bbox_h
    print(f"  bbox half-extents (Eq 16):  h_x = {h_x:.3f}   h_y = {h_y:.3f}")
    print(f"  n0 = Tv×Tw = {c['n0']}")
    print(f"  n1 = Tw×Tu = {c['n1']}")
    print(f"  n2 = Tu×Tv = {c['n2']}")
    # cross-product polynomial coefficients (cross.x, cross.y, cross.z as functions of (px, py)):
    n0, n1, n2 = c['n0'], c['n1'], c['n2']
    print(f"  cross.x(px, py) = {n0[0]:+.4e}·px + {n1[0]:+.4e}·py + {n2[0]:+.4e}")
    print(f"  cross.y(px, py) = {n0[1]:+.4e}·px + {n1[1]:+.4e}·py + {n2[1]:+.4e}")
    print(f"  cross.z(px, py) = {n0[2]:+.4e}·px + {n1[2]:+.4e}·py + {n2[2]:+.4e}")
    # Also dump cross.x at the projected center directly via T to verify.
    cx_at_p = float(c['p'][0]*n0[0] + c['p'][1]*n1[0] + n2[0])
    cy_at_p = float(c['p'][0]*n0[1] + c['p'][1]*n1[1] + n2[1])
    cz_at_p = float(c['p'][0]*n0[2] + c['p'][1]*n1[2] + n2[2])
    print(f"  cross at conic center p: ({cx_at_p:.4f}, {cy_at_p:.4f}, {cz_at_p:.4f})")

    # ----------------------------------------------------------------
    # 3. Empirical bbox: sample pixels around projected center, check inside disk
    print(f"\n=== Empirical bbox by pixel sampling ===")
    cx, cy = c['p']
    # Sample a ±2× range to find the actual bbox via disk-membership.
    span = max(h_x, h_y) * 2.0 + 30
    pxs = np.arange(int(cx - span), int(cx + span) + 1)
    pys = np.arange(int(cy - span), int(cy + span) + 1)
    Pys, Pxs = np.meshgrid(pys, pxs, indexing='ij')
    inside = np.zeros_like(Pxs, dtype=bool)
    k_sq = cutoff * cutoff
    for i in range(Pxs.shape[0]):
        for j in range(Pxs.shape[1]):
            s = surfel_coord_at_pixel(T, Pxs[i, j], Pys[i, j])
            if s is not None and (s[0]*s[0] + s[1]*s[1]) <= k_sq:
                inside[i, j] = True
    if not inside.any():
        print("  No pixels inside disk in sampling range — surfel is too far from camera or culled.")
        return
    in_pxs = Pxs[inside]
    in_pys = Pys[inside]
    emp_x_min, emp_x_max = in_pxs.min(), in_pxs.max()
    emp_y_min, emp_y_max = in_pys.min(), in_pys.max()
    emp_cx = (emp_x_min + emp_x_max) / 2.0
    emp_cy = (emp_y_min + emp_y_max) / 2.0
    emp_h_x = (emp_x_max - emp_x_min) / 2.0
    emp_h_y = (emp_y_max - emp_y_min) / 2.0
    print(f"  Inside pixels: {inside.sum()} (out of {Pxs.size})")
    print(f"  Bounding pixels: x ∈ [{emp_x_min}, {emp_x_max}], y ∈ [{emp_y_min}, {emp_y_max}]")
    print(f"  Bbox center      ({emp_cx:.2f}, {emp_cy:.2f})")
    print(f"  Bbox half-extent ({emp_h_x:.2f}, {emp_h_y:.2f})")

    # ----------------------------------------------------------------
    # 4. Verification: at the four ellipse extreme points (max-x, min-x,
    # max-y, min-y), Q should equal t (the cutoff). For a rotated ellipse,
    # these points are NOT (p.x ± h_x, p.y) etc. — they're at:
    #   max-x point: (p.x + h_x, p.y - B·h_x/E)
    #   max-y point: (p.x - B·h_y/A, p.y + h_y)
    # (and mirrors).
    A_, B_, E_, t_, p_ = c['A'], c['B'], c['E'], c['t'], c['p']
    dy_at_max_x = -B_ * h_x / E_
    dx_at_max_y = -B_ * h_y / A_
    print(f"\n=== Conic vs cross-product Q-value sanity check ===")
    print(f"  At ellipse extremes (Q should ≈ t = {t_:.4f}):")
    test_pts = [(p_[0],         p_[1],         "center (Q=-t expected? no, Q=0 but centered = -G+stuff)"),
                (p_[0] + h_x,   p_[1] + dy_at_max_x, "max-x point"),
                (p_[0] - h_x,   p_[1] - dy_at_max_x, "min-x point"),
                (p_[0] + dx_at_max_y, p_[1] + h_y,   "max-y point"),
                (p_[0] - dx_at_max_y, p_[1] - h_y,   "min-y point"),
                (p_[0] + 0.5*h_x, p_[1] + 0.5*dy_at_max_x, "halfway to max-x")]
    Tu_arr, Tv_arr, Tw_arr = T[:, 0], T[:, 1], T[:, 2]
    for (px, py, label) in test_pts:
        s = surfel_coord_at_pixel(T, px, py)
        k = px*Tw_arr - Tu_arr; l = py*Tw_arr - Tv_arr
        cr = cross(k, l)
        Q_cross = cr[0]**2 + cr[1]**2 - k_sq * cr[2]**2
        # Full Q_conic (should equal Q_cross by construction).
        Q_full = (c['A']*px*px + 2*c['B']*px*py + c['E']*py*py
                  + 2*c['D']*px + 2*c['F']*py + c['G'])
        # Centered Q: A·dx² + 2B·dx·dy + E·dy².
        dx = px - p_[0]; dy = py - p_[1]
        Q_centered = A_*dx*dx + 2*B_*dx*dy + E_*dy*dy
        s_disc = (s[0]**2 + s[1]**2) if s is not None else float('nan')
        print(f"  ({px:9.3f}, {py:9.3f}) [{label:20}]")
        print(f"     Q_full={Q_full:+.6e}  Q_cross={Q_cross:+.6e}  ratio={(Q_full/Q_cross) if abs(Q_cross)>1e-12 else float('nan'):.6f}")
        print(f"     Q_centered={Q_centered:+.6e}  |s|^2={s_disc:.6f}")

    # ----------------------------------------------------------------
    # 4b. SnugBox bbox + AccuTile vs brute-force ground truth.
    BLOCK_X = 16
    BLOCK_Y = 16
    grid_x = (W + BLOCK_X - 1) // BLOCK_X
    grid_y = (H + BLOCK_Y - 1) // BLOCK_Y
    sb = snugbox_bbox(c, BLOCK_X, BLOCK_Y, grid_x, grid_y)
    if sb is None:
        print("\n=== SnugBox: bbox is empty/invalid ===")
    else:
        accu_tiles = accutile_walk(c, sb, BLOCK_X, BLOCK_Y)
        # Rect-AABB tiles (everything in the bbox rect):
        rect_tiles = set(
            (tx, ty)
            for tx in range(sb['rect_min'][0], sb['rect_max'][0])
            for ty in range(sb['rect_min'][1], sb['rect_max'][1])
        )
        # Brute force ground truth (restricted to rect to keep it fast).
        bf_tiles = brute_force_tiles(T, cutoff, BLOCK_X, BLOCK_Y, grid_x, grid_y,
                                     samples_per_pixel=4,
                                     rect_min=sb['rect_min'], rect_max=sb['rect_max'])

        print(f"\n=== SnugBox / AccuTile / brute-force comparison ===")
        print(f"  Image grid:    {grid_x}×{grid_y} tiles ({grid_x*grid_y} total)")
        print(f"  Rect AABB:     {len(rect_tiles)} tiles  (rect=[{sb['rect_min']}, {sb['rect_max']}))")
        print(f"  AccuTile:      {len(accu_tiles)} tiles")
        print(f"  Brute force:   {len(bf_tiles)} tiles")
        # AccuTile should ⊇ brute-force (every truly-touched tile must be in AccuTile),
        # and AccuTile should ⊆ rect (it's a refinement within the bbox).
        only_in_bf = bf_tiles - accu_tiles
        only_in_accu = accu_tiles - bf_tiles
        print(f"  Tiles in BF but NOT in AccuTile (missed): {len(only_in_bf)}")
        if only_in_bf:
            print(f"    examples: {sorted(only_in_bf)[:5]}")
        print(f"  Tiles in AccuTile but NOT in BF (extra):  {len(only_in_accu)}")
        if only_in_accu:
            print(f"    examples: {sorted(only_in_accu)[:5]}")
        not_in_rect = bf_tiles - rect_tiles
        print(f"  Tiles in BF but NOT in Rect (rect missed): {len(not_in_rect)}")

    # ----------------------------------------------------------------
    # 5. Comparison
    print(f"\n=== Side-by-side ===")
    print(f"  {'method':<24} {'p.x':>10} {'p.y':>10} {'h_x':>10} {'h_y':>10}")
    print(f"  {'-'*64}")
    print(f"  {'compute_aabb':<24} {aabb_p[0]:10.3f} {aabb_p[1]:10.3f} {aabb_h[0]:10.3f} {aabb_h[1]:10.3f}")
    print(f"  {'cross-product conic':<24} {c['p'][0]:10.3f} {c['p'][1]:10.3f} {h_x:10.3f} {h_y:10.3f}")
    print(f"  {'empirical (samples)':<24} {emp_cx:10.3f} {emp_cy:10.3f} {emp_h_x:10.3f} {emp_h_y:10.3f}")


if __name__ == "__main__":
    main()
