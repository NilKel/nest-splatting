#!/usr/bin/env python
"""
Generate a synthetic multi-view "watermelon clipping" dataset in NeRF-synthetic
(Gaussian-Splatting friendly) format -- WITHOUT Blender.

The scene is a solid unit sphere ("watermelon") sliced by a half-space (the
"clip plane"). Rendered with an analytic GPU ray-marcher in PyTorch:

  * Exterior shell  -> shiny dark-green rind with longitudinal stripes
                       (Blinn-Phong specular, view-dependent highlight).
  * Cut face        -> red interior with a volumetric subsurface-scattering
                       look: depth-weighted flesh colour, a thin-slab rim glow,
                       and high-frequency 3D Worley "seeds".

The seeds + flesh use *object-space* (= world-space, object sits at the origin)
coordinates, so the internal structure is STATIC as the clip plane moves -- the
cut reveals a consistent solid, exactly like slicing a real watermelon.

Output layout (matches nerf_synthetic/lego etc.):

    <out>/
        train/r_0.png ... r_{Ntrain-1}.png     (800x800 RGBA, transparent bg)
        test/r_0.png ...
        val/r_0.png ...
        transforms_train.json
        transforms_test.json
        transforms_val.json
        points3d.ply            (random in-sphere init cloud for 3DGS)
        metadata.json

Each frame entry carries the standard `transform_matrix` (4x4 camera-to-world,
OpenGL convention) plus an extra `clip_plane` field: the world-space plane
[a, b, c, d] with a*x + b*y + c*z + d = 0, where the KEEP half-space is
a*x + b*y + c*z + d <= 0 (i.e. n . x <= s, n=[a,b,c], d=-s). Standard NeRF
loaders ignore the extra key; clip-conditioned models can read it.

Run:
    conda run -n nest_splatting python scripts/generate_watermelon_dataset.py \
        --out data/clipping/watermelon

Everything is vectorised on the GPU and chunked over pixel rows, so a full
800x800, 2x supersampled, 300-frame run takes a few minutes on a 5090.
"""

import argparse
import json
import math
import os

import numpy as np
import torch


# ----------------------------------------------------------------------------- #
# Math helpers
# ----------------------------------------------------------------------------- #
def normalize(v, dim=-1, eps=1e-8):
    return v / (v.norm(dim=dim, keepdim=True) + eps)


def smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_uniform_sphere(rng, n):
    """n uniform directions on the unit sphere (numpy, shape (n,3))."""
    z = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def look_at_c2w(cam_pos, world_up=(0.0, 0.0, 1.0)):
    """Camera-to-world 4x4 (OpenGL: +x right, +y up, -z forward) looking at origin."""
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    zc = cam_pos / (np.linalg.norm(cam_pos) + 1e-12)          # camera +Z (backward)
    up = np.asarray(world_up, dtype=np.float64)
    if abs(float(np.dot(zc, up))) > 0.999:                    # looking near-vertical
        up = np.array([0.0, 1.0, 0.0])
    xc = np.cross(up, zc)
    xc = xc / (np.linalg.norm(xc) + 1e-12)                    # camera +X (right)
    yc = np.cross(zc, xc)                                     # camera +Y (up)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = xc
    c2w[:3, 1] = yc
    c2w[:3, 2] = zc
    c2w[:3, 3] = cam_pos
    return c2w


# ----------------------------------------------------------------------------- #
# 3D Worley / cellular noise (for watermelon seeds) -- hash-based, O(27) cells.
# ----------------------------------------------------------------------------- #
def _hash3(ix, iy, iz):
    """Deterministic per-cell jitter in [0,1)^3 given integer cell coords."""
    def frac(x):
        return x - torch.floor(x)
    # three independent fract(sin(dot)) hashes
    h1 = frac(torch.sin(ix * 127.1 + iy * 311.7 + iz * 74.7) * 43758.5453)
    h2 = frac(torch.sin(ix * 269.5 + iy * 183.3 + iz * 246.1) * 43758.5453)
    h3 = frac(torch.sin(ix * 113.5 + iy * 271.9 + iz * 124.6) * 43758.5453)
    return h1, h2, h3


def worley_f1(pos, cell=0.13):
    """F1 (distance to nearest jittered feature point) of a cellular noise.

    pos: (...,3) world/object coords. Returns (...,) nearest-feature distance.
    One feature point per cell of size `cell`; checks the 27 neighbouring cells.
    """
    p = pos / cell
    base = torch.floor(p)
    f = p - base
    best = torch.full(p.shape[:-1], 1e9, device=pos.device, dtype=pos.dtype)
    rng = (-1.0, 0.0, 1.0)
    for ox in rng:
        for oy in rng:
            for oz in rng:
                ci = base[..., 0] + ox
                cj = base[..., 1] + oy
                ck = base[..., 2] + oz
                hx, hy, hz = _hash3(ci, cj, ck)
                # feature point offset within the neighbour cell
                dx = (ox + hx) - f[..., 0]
                dy = (oy + hy) - f[..., 1]
                dz = (oz + hz) - f[..., 2]
                d = torch.sqrt(dx * dx + dy * dy + dz * dz)
                best = torch.minimum(best, d)
    return best * cell  # back to world units


# ----------------------------------------------------------------------------- #
# Shading
# ----------------------------------------------------------------------------- #
class Shader:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.dev = device
        self.L = normalize(torch.tensor(cfg.light_dir, dtype=torch.float32, device=device))
        self.r0 = cfg.radius

    # --- exterior rind ---------------------------------------------------- #
    def shade_exterior(self, p, view):
        """p: (K,3) surface points (object coords). view: (K,3) dir to camera."""
        N = normalize(p)
        diff = torch.clamp((N * self.L).sum(-1), 0.0)
        half = normalize(self.L + view)
        spec = torch.clamp((N * half).sum(-1), 0.0) ** self.cfg.shininess

        # longitudinal watermelon stripes (object space => static under cut)
        lon = torch.atan2(p[..., 1], p[..., 0])
        stripe = 0.5 + 0.5 * torch.cos(lon * self.cfg.n_stripes)
        stripe = smoothstep(0.35, 0.65, stripe)                 # crisp dark stripes
        # fine mottling along the stripe
        mottle = 0.85 + 0.15 * torch.cos(lon * self.cfg.n_stripes * 7.0 +
                                         p[..., 2] * 9.0)
        dark = torch.tensor([0.015, 0.16, 0.05], device=self.dev)
        light = torch.tensor([0.05, 0.42, 0.12], device=self.dev)
        rind = dark[None] + (light - dark)[None] * (stripe * mottle)[..., None]

        amb = self.cfg.ambient
        shaded = rind * (amb + (1.0 - amb) * diff[..., None])
        shaded = shaded + (self.cfg.spec_strength * spec)[..., None]
        return torch.clamp(shaded, 0.0, 1.0)

    # --- interior cut face (subsurface look) ------------------------------ #
    def shade_interior(self, o, d, t0, t1, n_world, view):
        """Volumetric march from t0 (cut face) to t1 (back of solid)."""
        cfg = self.cfg
        K = t0.shape[0]
        thick = torch.clamp(t1 - t0, min=0.0)
        M = cfg.march_steps

        acc = torch.zeros(K, 3, device=self.dev)
        wsum = torch.zeros(K, device=self.dev)
        # step centres in [t0, t1]
        for k in range(M):
            frac = (k + 0.5) / M
            t = t0 + frac * thick
            x = o + t[..., None] * d                              # object coords
            rad = x.norm(dim=-1)

            # flesh colour by radius: pale centre -> deep red -> pale rind
            deep = torch.tensor([0.85, 0.10, 0.16], device=self.dev)
            pale_center = torch.tensor([0.95, 0.55, 0.55], device=self.dev)
            white_rind = torch.tensor([0.80, 0.85, 0.62], device=self.dev)
            col = deep[None].expand(K, 3).clone()
            # toward centre -> pinker
            tc = smoothstep(0.30, 0.05, rad)[..., None]
            col = col + (pale_center - deep)[None] * tc
            # toward rind -> whitish-green flesh
            tr = smoothstep(0.86, 0.99, rad)[..., None]
            col = col * (1.0 - tr) + white_rind[None] * tr

            # seeds: Worley F1, only in the red flesh band
            f1 = worley_f1(x, cell=cfg.seed_cell)
            seed_mask = (f1 < cfg.seed_radius).float()
            band = ((rad > 0.28) & (rad < 0.86)).float()
            seed_mask = seed_mask * band
            seed_col = torch.tensor([0.06, 0.03, 0.03], device=self.dev)
            col = col * (1.0 - seed_mask[..., None]) + seed_col[None] * seed_mask[..., None]

            # depth weighting => near the cut face dominates (volumetric falloff)
            depth = t - t0
            w = torch.exp(-cfg.sigma_a * depth)
            acc = acc + w[..., None] * col
            wsum = wsum + w

        flesh = acc / (wsum[..., None] + 1e-6)

        # lighting on the flat cut face (outward normal = +n, toward camera)
        diff = torch.clamp((n_world * self.L).sum(-1), 0.0)
        amb = cfg.ambient
        lit = flesh * (amb + (1.0 - amb) * diff[..., None])

        # subsurface rim glow: thin slabs transmit more light -> warm glow
        glow = cfg.sss_strength * torch.exp(-cfg.sigma_sss * thick)
        tint = torch.tensor([1.0, 0.45, 0.45], device=self.dev)
        lit = lit + glow[..., None] * tint[None]

        # faint wet sheen on the cut surface
        half = normalize(self.L + view)
        sheen = torch.clamp((n_world * half).sum(-1), 0.0) ** 40.0
        lit = lit + (0.10 * sheen)[..., None]
        return torch.clamp(lit, 0.0, 1.0)


# ----------------------------------------------------------------------------- #
# Renderer
# ----------------------------------------------------------------------------- #
def render_frame(cfg, shader, c2w, plane, device):
    """Render one RGBA frame. c2w: (4,4) np. plane: (n(3,), s) world keep n.x<=s."""
    W = cfg.res * cfg.ss
    H = W
    fx = 0.5 * W / math.tan(0.5 * cfg.camera_angle_x)

    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    R = c2w_t[:3, :3]
    cam = c2w_t[:3, 3]
    n_world = torch.tensor(plane[0], dtype=torch.float32, device=device)
    s = float(plane[1])

    no = float((n_world * cam).sum())            # n . o  (camera, scalar)
    r0 = cfg.radius

    # pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_ndc = (xs + 0.5 - W * 0.5) / fx
    y_ndc = -(ys + 0.5 - H * 0.5) / fx
    dirs_cam = torch.stack([x_ndc, y_ndc, -torch.ones_like(x_ndc)], dim=-1)  # (H,W,3)
    dirs_world = normalize(dirs_cam @ R.T)                                    # (H,W,3)

    out = torch.zeros(H, W, 4, device=device)

    rows = cfg.row_chunk
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        d = dirs_world[y0:y1].reshape(-1, 3)        # (P,3)
        P = d.shape[0]
        o = cam[None].expand(P, 3)

        # ray-sphere
        b = (o * d).sum(-1)
        c = (o * o).sum(-1) - r0 * r0
        disc = b * b - c
        hit_sphere = disc > 0.0
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        t_near = -b - sq
        t_far = -b + sq

        # ray-halfspace (keep n.x <= s)
        nd = (d * n_world).sum(-1)
        eps = 1e-7
        tp = (s - no) / torch.where(nd.abs() < eps, torch.full_like(nd, eps), nd)

        pos_nd = nd > eps
        neg_nd = nd < -eps
        zero_nd = ~(pos_nd | neg_nd)

        t_lo = t_near.clone()
        t_hi = t_far.clone()
        # nd>0: keep upper-bounded by tp (entry = sphere)
        t_hi = torch.where(pos_nd, torch.minimum(t_far, tp), t_hi)
        # nd<0: keep lower-bounded by tp (entry = plane if tp>t_near)
        t_lo = torch.where(neg_nd, torch.maximum(t_near, tp), t_lo)
        # nd==0: keep all iff n.o<=s, else nothing
        keep_zero = no <= s
        if not keep_zero:
            t_hi = torch.where(zero_nd, t_lo - 1.0, t_hi)   # force empty

        valid = hit_sphere & (t_lo <= t_hi) & (t_hi > 0.0) & (t_lo > 0.0)
        plane_entry = valid & neg_nd & (tp > t_near)
        sphere_entry = valid & (~plane_entry)

        rgb = torch.zeros(P, 3, device=device)
        view = -d

        if sphere_entry.any():
            idx = sphere_entry.nonzero(as_tuple=True)[0]
            t0 = t_lo[idx]
            p = o[idx] + t0[..., None] * d[idx]
            rgb[idx] = shader.shade_exterior(p, view[idx])

        if plane_entry.any():
            idx = plane_entry.nonzero(as_tuple=True)[0]
            t0 = t_lo[idx]
            t1 = t_hi[idx]
            nw = n_world[None].expand(idx.shape[0], 3)
            rgb[idx] = shader.shade_interior(o[idx], d[idx], t0, t1, nw, view[idx])

        alpha = valid.float()
        frame = torch.cat([rgb, alpha[..., None]], dim=-1).reshape(y1 - y0, W, 4)
        out[y0:y1] = frame

    # supersample downsample (premultiplied to avoid dark edge halos)
    if cfg.ss > 1:
        ss = cfg.ss
        out = out.reshape(cfg.res, ss, cfg.res, ss, 4).permute(0, 2, 1, 3, 4)
        out = out.reshape(cfg.res, cfg.res, ss * ss, 4)
        a = out[..., 3:4]
        rgb_sum = (out[..., :3] * a).sum(dim=2)
        a_sum = a.sum(dim=2)
        rgb = rgb_sum / (a_sum + 1e-6)
        alpha = a_sum.squeeze(-1) / (ss * ss)
        out = torch.cat([rgb, alpha[..., None]], dim=-1)

    return torch.clamp(out, 0.0, 1.0)


# ----------------------------------------------------------------------------- #
# Clip-plane sampling
# ----------------------------------------------------------------------------- #
def sample_clip_plane(cfg, rng):
    """Returns (n(3,), s) so keep half-space is n.x <= s, plane n.x = s cuts sphere."""
    if cfg.clip_mode == "none":
        # plane far outside the sphere on the +X side -> nothing removed
        return np.array([1.0, 0.0, 0.0]), cfg.radius * 10.0
    if cfg.clip_mode == "fixed":
        n = np.array(cfg.fixed_normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        return n, cfg.fixed_offset
    # per-frame random
    n = sample_uniform_sphere(rng, 1)[0]
    s = rng.uniform(cfg.clip_smin, cfg.clip_smax) * cfg.radius
    return n, s


def plane_abcd(n, s):
    """World plane [a,b,c,d] with a x+b y+c z+d=0; keep side n.x+d<=0 (d=-s)."""
    return [float(n[0]), float(n[1]), float(n[2]), float(-s)]


# ----------------------------------------------------------------------------- #
# Split generation
# ----------------------------------------------------------------------------- #
def gen_split(cfg, shader, device, split, n_frames, rng, save_png):
    from PIL import Image
    split_dir = os.path.join(cfg.out, split)
    os.makedirs(split_dir, exist_ok=True)

    dirs = sample_uniform_sphere(rng, n_frames)
    radii = rng.uniform(cfg.cam_rmin, cfg.cam_rmax, size=n_frames)

    frames = []
    for i in range(n_frames):
        cam_pos = dirs[i] * radii[i]
        c2w = look_at_c2w(cam_pos)
        n, s = sample_clip_plane(cfg, rng)

        img = render_frame(cfg, shader, c2w, (n, s), device)   # (res,res,4) float
        if save_png:
            arr = (img.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
            Image.fromarray(arr, mode="RGBA").save(
                os.path.join(split_dir, f"r_{i}.png"))

        frames.append({
            "file_path": f"./{split}/r_{i}",
            "rotation": 0.0,
            "transform_matrix": c2w.tolist(),
            "clip_plane": plane_abcd(n, s),
        })
        if (i + 1) % 25 == 0 or i == n_frames - 1:
            print(f"  [{split}] {i + 1}/{n_frames}", flush=True)

    meta = {"camera_angle_x": cfg.camera_angle_x, "frames": frames}
    with open(os.path.join(cfg.out, f"transforms_{split}.json"), "w") as f:
        json.dump(meta, f, indent=4)
    return frames


def write_points3d(cfg, rng):
    """Random in-sphere point cloud (radius-coloured) for 3DGS initialisation."""
    n = cfg.n_init_points
    # rejection-free: uniform in ball via direction * U^(1/3)
    dirs = sample_uniform_sphere(rng, n)
    r = (rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)) * cfg.radius
    pts = dirs * r[:, None]
    rad = np.linalg.norm(pts, axis=1) / cfg.radius
    # green near surface, red inside
    green = np.array([0.05, 0.42, 0.12])
    red = np.array([0.85, 0.12, 0.18])
    t = np.clip((rad - 0.85) / 0.15, 0.0, 1.0)[:, None]
    col = (red * (1 - t) + green * t)
    rgb = (np.clip(col, 0, 1) * 255).astype(np.uint8)

    path = os.path.join(cfg.out, "points3d.ply")
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    return path


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--ss", type=int, default=2, help="supersample factor (AA)")
    ap.add_argument("--row_chunk", type=int, default=256,
                    help="pixel rows per GPU chunk (lower if OOM)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_png", action="store_true", help="JSON only (debug)")

    # scene / camera
    ap.add_argument("--radius", type=float, default=1.0, help="watermelon radius")
    ap.add_argument("--cam_rmin", type=float, default=3.6)
    ap.add_argument("--cam_rmax", type=float, default=4.4)
    ap.add_argument("--fov", type=float, default=0.6911112,
                    help="horizontal FOV (radians); stored as camera_angle_x")

    # clip plane
    ap.add_argument("--clip_mode", choices=["per_frame", "fixed", "none"],
                    default="per_frame")
    ap.add_argument("--clip_smin", type=float, default=-0.5,
                    help="min signed offset (x radius) of the keep boundary")
    ap.add_argument("--clip_smax", type=float, default=0.7)
    ap.add_argument("--fixed_normal", type=float, nargs=3, default=[1.0, 0.3, 0.2])
    ap.add_argument("--fixed_offset", type=float, default=0.0)

    # appearance
    ap.add_argument("--light_dir", type=float, nargs=3, default=[1.0, 1.0, 1.2])
    ap.add_argument("--ambient", type=float, default=0.18)
    ap.add_argument("--shininess", type=float, default=60.0)
    ap.add_argument("--spec_strength", type=float, default=0.45)
    ap.add_argument("--n_stripes", type=float, default=10.0)
    ap.add_argument("--march_steps", type=int, default=20)
    ap.add_argument("--sigma_a", type=float, default=3.5, help="flesh extinction")
    ap.add_argument("--sigma_sss", type=float, default=2.2, help="rim-glow falloff")
    ap.add_argument("--sss_strength", type=float, default=0.55)
    ap.add_argument("--seed_cell", type=float, default=0.11)
    ap.add_argument("--seed_radius", type=float, default=0.028)
    ap.add_argument("--n_init_points", type=int, default=100000)

    cfg = ap.parse_args()
    cfg.camera_angle_x = cfg.fov
    os.makedirs(cfg.out, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | out: {cfg.out}")
    print(f"Res {cfg.res} x{cfg.ss}SS | clip_mode={cfg.clip_mode}")

    rng = np.random.default_rng(cfg.seed)
    shader = Shader(cfg, device)
    save_png = not cfg.no_png

    splits = [("train", cfg.n_train), ("test", cfg.n_test), ("val", cfg.n_val)]
    for split, n in splits:
        if n <= 0:
            continue
        print(f"Rendering split '{split}' ({n} frames)...")
        gen_split(cfg, shader, device, split, n, rng, save_png)

    ply = write_points3d(cfg, rng)
    print(f"Wrote {ply}")

    with open(os.path.join(cfg.out, "metadata.json"), "w") as f:
        json.dump({
            "description": "Procedural watermelon sliced by a per-frame clip plane.",
            "renderer": "analytic GPU ray-marcher (PyTorch), no Blender",
            "camera_angle_x": cfg.camera_angle_x,
            "resolution": cfg.res,
            "supersample": cfg.ss,
            "radius": cfg.radius,
            "clip_mode": cfg.clip_mode,
            "clip_offset_range_x_radius": [cfg.clip_smin, cfg.clip_smax],
            "clip_plane_convention": "n.x + d <= 0 is the KEPT half (d=-s); [a,b,c,d] world",
            "n_train": cfg.n_train, "n_test": cfg.n_test, "n_val": cfg.n_val,
            "seed": cfg.seed,
        }, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()
