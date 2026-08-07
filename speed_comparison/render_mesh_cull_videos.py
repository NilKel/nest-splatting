"""
Render two CUDA-reference videos of the mesh-cull finetune, for A/B against
the WebGPU viewer's recording of the same trajectory:

  1. Circular orbit around the object (fitted plane through training cams).
  2. Smooth Slerp interpolation through every train + test view in order.

Both use the same per-Gauss mesh-cull pipeline as `render_all_views_cull.py`
(the "working CUDA renderer" — override_opacity from a keep-mask computed
against the per-view mesh depth map). Frames go into a temp dir then get
piped to ffmpeg for h264 MP4.
"""
from __future__ import annotations
import os, sys, math, argparse, subprocess, tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio

# Reuse load_model / MeshDepthBaker / gauss_keep / depth-to-viridis /
# to_u8 from the sibling per-view renderer — same CUDA path, same cull
# semantics, same rasterizer flags.
from speed_comparison.render_all_views_cull import (
    load_model, MeshDepthBaker, gauss_keep, to_u8,
)
from gaussian_renderer import render
from scene.cameras import Camera


# -------- Camera synthesis helpers --------
def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Quaternion Slerp. q's are [w, x, y, z], unit-norm."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / (np.linalg.norm(r) + 1e-12)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    a = math.sin((1.0 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rot matrix -> quat [w, x, y, z]. Standard Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ])


def _make_cam(template: Camera, R_wc: np.ndarray, t_cw: np.ndarray,
              uid: int, name: str) -> Camera:
    """Build a fresh Camera object by copying a template's intrinsics and
    injecting a new pose. `R_wc` = 3x3 world->cam rotation (COLMAP convention:
    identical to the R stored on Camera). `t_cw` = 3-vec translation as stored
    on Camera. GT image is a black CPU placeholder — inference render never
    reads it (only pose + intrinsics matter), and pinning on CUDA blows out
    VRAM at ~24 MB/cam for 4K images × hundreds of frames (bit us on the
    sequence-video first attempt with 793 cams → OOM). world_view_transform
    and projection_matrix are unconditionally on CUDA, so render is unaffected."""
    black = torch.zeros(3, template.image_height, template.image_width, dtype=torch.float32)
    return Camera(
        colmap_id=uid,
        R=R_wc, T=t_cw,
        FoVx=float(template.FoVx), FoVy=float(template.FoVy),
        image=black, gt_alpha_mask=None,
        image_name=name, uid=uid,
        data_device="cpu",
    )


def _as_np(x):
    """Camera.R/T can be either numpy arrays or torch tensors (device may be
    CUDA) depending on how the Scene was loaded; normalize to CPU numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cam_world_pos(cam: Camera) -> np.ndarray:
    """Extract world-space camera position from a Camera object. Note the
    training Camera stores R (world->cam) and T (translation applied AFTER R
    to a world point, giving cam-space); so world-position = -R^T @ T."""
    R = _as_np(cam.R).astype(np.float64)
    T = _as_np(cam.T).astype(np.float64)
    return -R.T @ T


def _look_at(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray):
    """Build a COLMAP-style (R_wc, T_cw) pair such that the resulting camera
    sits at `cam_pos` and looks at `target` with world-up `up`. COLMAP cam
    convention: +Z looks INTO the scene (not -Z like OpenGL), +Y is DOWN.
    Returns R (3x3 world->cam) and T (cam-space translation)."""
    fwd = target - cam_pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    # Right = up x fwd, then re-orthogonalize up.
    right = np.cross(up, fwd)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # cam_pos aligned with up axis — pick a safe fallback.
        right = np.cross(np.array([1.0, 0.0, 0.0]), fwd)
        rn = np.linalg.norm(right) + 1e-12
    right = right / rn
    new_up = np.cross(fwd, right)  # already unit
    # COLMAP world-to-cam rotation: rows are [right, down, forward] in world.
    # We want +y_cam = DOWN in image, so flip up → down.
    R_wc = np.stack([right, -new_up, fwd], axis=0)   # 3x3
    T_cw = -R_wc @ cam_pos
    return R_wc.astype(np.float32), T_cw.astype(np.float32)


def make_circular_cams(train_cams, test_cams, num_frames: int) -> list:
    """Orbit spec (user-directed):
      * center     = mean of ALL cam positions (train + test)
      * start pose = first test camera position (so frame 0 = a known reference)
      * axis       = PCA minimum-variance direction of the cam positions —
                     for a turntable capture, this is the turntable rotation
                     axis (all cams lie ~perpendicular to it)
      * radius     = |p0 − center| projected onto the orbit plane
      * height     = signed offset of p0 along the orbit axis (holds the
                     starting camera's latitude constant across the sweep)

    Each frame looks AT the center with the orbit axis as world-up. That
    fixes the earlier "zoomed in / off-center / wrong axis" symptoms — those
    came from centering the orbit on the cam-centroid AND using in-plane
    variance for radius, which for a hemisphere collapses to a tiny fraction
    of the true camera→object distance."""
    all_cams = list(train_cams) + list(test_cams)
    positions = np.stack([_cam_world_pos(c) for c in all_cams], axis=0)
    center = positions.mean(axis=0)

    # PCA to find the minimum-variance axis of the cam cloud.
    vecs = positions - center
    _, _, Vt = np.linalg.svd(vecs, full_matrices=False)
    axis = Vt[-1]
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    # Starting pose = first test cam. Project its offset from center onto
    # the plane perpendicular to `axis` to get the in-plane radial vector.
    p0 = _cam_world_pos(test_cams[0])
    rel = p0 - center
    height = float(np.dot(rel, axis))           # keep this latitude
    rel_in_plane = rel - height * axis
    r = float(np.linalg.norm(rel_in_plane))
    if r < 1e-6:
        raise RuntimeError("First test cam lies on the orbit axis — cannot orbit.")
    e1 = rel_in_plane / r                       # radial start direction
    e2 = np.cross(axis, e1)                     # tangential (right-hand rule)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    print(f"[orbit] center = {center}")
    print(f"[orbit] axis   = {axis}  (min-variance direction of cam cloud)")
    print(f"[orbit] radius = {r:.4f}  (first-test-cam distance from center, in orbit plane)")
    print(f"[orbit] height = {height:+.4f}  (first-test-cam latitude offset along axis)")

    template = test_cams[0]
    out = []
    for i in range(num_frames):
        theta = 2 * math.pi * (i / num_frames)
        cam_pos = center + height * axis + r * (math.cos(theta) * e1 + math.sin(theta) * e2)
        R, T = _look_at(cam_pos, center, axis)
        out.append(_make_cam(template, R, T, uid=1000 + i, name=f"orbit_{i:04d}"))
    return out


def iter_sequence_cams(all_cams: list, frames_per_gap: int):
    """Generator that Slerps between consecutive cameras in `all_cams` and
    yields one Camera at a time. Lazy so we never hold >1 cam simultaneously
    — building the full list upfront blew ~19 GB of RAM for 793 frames on the
    brain scene (each Camera pins a H×W×3 CPU placeholder image)."""
    if len(all_cams) < 2:
        yield from all_cams
        return
    template = all_cams[0]
    quats = [_R_to_quat(_as_np(c.R).astype(np.float64)) for c in all_cams]
    poses = [_cam_world_pos(c) for c in all_cams]
    frame_uid = 2000
    for i in range(len(all_cams) - 1):
        q0, q1 = quats[i], quats[i + 1]
        p0, p1 = poses[i], poses[i + 1]
        gap = frames_per_gap
        end = gap + 1 if i == len(all_cams) - 2 else gap
        for k in range(end):
            t = k / gap
            q = _slerp(q0, q1, t)
            p = p0 + t * (p1 - p0)
            R_wc = _quat_to_R(q).astype(np.float64)
            T_cw = (-R_wc @ p).astype(np.float32)
            yield _make_cam(template, R_wc.astype(np.float32), T_cw,
                            uid=frame_uid, name=f"seq_{frame_uid:05d}")
            frame_uid += 1


def sequence_total_frames(num_cams: int, frames_per_gap: int) -> int:
    """Total frames iter_sequence_cams will yield for `num_cams` inputs."""
    if num_cams < 2:
        return num_cams
    return 1 + (num_cams - 1) * frames_per_gap


# -------- Frame rendering --------
def render_trajectory(cam_iter, total: int, out_dir: Path, gaussians, cfg, pipe,
                      ingp, beta_kern, baker: MeshDepthBaker, iteration: int,
                      mesh_margin: float, log_prefix: str):
    """Iterate cameras (accept generator OR list), render each with mesh
    cull applied, save PNGs into out_dir. Streaming — the previous list-based
    version blew RAM on the 793-frame sequence pass by holding all Camera
    objects (each with a full-res placeholder image) at once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    for idx, cam in enumerate(cam_iter):
        mesh_z = baker.cam_depth(cam, mesh_margin)
        keep = gauss_keep(gaussians.get_xyz, cam, mesh_z)
        mask = keep.to(gaussians.get_xyz.dtype).view(-1, 1)
        override_op = gaussians.get_opacity * mask
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                         iteration=iteration, cfg=cfg, ingp=ingp,
                         is_training=False, lowpass=True,
                         override_opacity=override_op)
        img = pkg["render"].clamp(0, 1)
        imageio.imwrite(str(out_dir / f"frame_{idx:05d}.png"), to_u8(img))
        # Explicitly drop the just-consumed Camera so the CPU placeholder image
        # gets GC'd promptly, and clear CUDA cache so long trajectories don't
        # accumulate fragmentation.
        del cam, mesh_z, keep, mask, override_op, pkg, img
        if idx % 20 == 0:
            print(f"  {log_prefix} frame {idx+1}/{total}")
    torch.cuda.empty_cache()


def frames_to_mp4(frame_dir: Path, out_mp4: Path, fps: int):
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        # Even H/W ensures libx264 stops complaining about odd dimensions.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    print(f"  -> {out_mp4}")
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", default="/home/nilkel/Projects/bitymi-demos/test-data/brain_mesh.ply")
    p.add_argument("--mesh_margin", type=float, default=0.03,
                   help="Legacy per-pixel depth bump.")
    p.add_argument("--mesh_normal_margin", type=float, default=0.0,
                   help="Geometric normal-inflation (preferred). See MeshDepthBaker.")
    p.add_argument("--out_dir", default=None,
                   help="Default: <model_path>/trajectory_videos/")
    p.add_argument("--orbit_frames", type=int, default=240,
                   help="Frames in the circular orbit video.")
    p.add_argument("--seq_gap", type=int, default=8,
                   help="Slerp frames between consecutive train+test views.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip_orbit", action="store_true")
    p.add_argument("--skip_sequence", action="store_true")
    args = p.parse_args()

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)
    print(f"[traj] iter={args.iteration}")
    out_root = Path(args.out_dir) if args.out_dir else Path(args.model_path) / "trajectory_videos"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model + mesh.
    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)
    train_cams = list(scene.getTrainCameras())
    test_cams  = list(scene.getTestCameras())
    print(f"[traj] {len(train_cams)} train + {len(test_cams)} test cams")
    baker = MeshDepthBaker(args.mesh_ply,
                           inflate_margin_normal=float(args.mesh_normal_margin))

    # ---- Video 1: circular orbit ----
    if not args.skip_orbit:
        if len(test_cams) == 0:
            raise SystemExit("No test cameras — orbit spec requires first test cam as start pose.")
        print(f"\n[traj] === orbit: {args.orbit_frames} frames ===")
        cams = make_circular_cams(train_cams, test_cams, args.orbit_frames)
        frame_dir = out_root / "orbit_frames"
        render_trajectory(cams, len(cams), frame_dir, gaussians, cfg, pipe, ingp,
                          beta_kern, baker, args.iteration, args.mesh_margin, "orbit")
        frames_to_mp4(frame_dir, out_root / "orbit.mp4", args.fps)

    # ---- Video 2: smooth pass through train+test views ----
    if not args.skip_sequence:
        all_cams = list(train_cams) + list(test_cams)
        total = sequence_total_frames(len(all_cams), args.seq_gap)
        print(f"\n[traj] === sequence: {len(all_cams)} views, "
              f"{args.seq_gap} frames/gap → {total} frames ===")
        frame_dir = out_root / "sequence_frames"
        # Streamed generator — one Camera in memory at a time. Building the
        # full list held ~19 GB RAM for 793 frames (H×W×3 CPU placeholder
        # per cam) and got OOM-killed.
        render_trajectory(iter_sequence_cams(all_cams, args.seq_gap), total,
                          frame_dir, gaussians, cfg, pipe, ingp, beta_kern,
                          baker, args.iteration, args.mesh_margin, "seq")
        frames_to_mp4(frame_dir, out_root / "sequence.mp4", args.fps)

    print(f"\n[traj] done. Videos in {out_root}/")


if __name__ == "__main__":
    main()
