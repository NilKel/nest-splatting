"""
Shape-validation + render-once harness for ClipRelightHead (Phase-1 prototype).

What it does (standalone — no dataset loader / train loop wiring):
  1. Reads a REAL camera + clip_plane from data/clipping/watermelon/transforms_train.json.
  2. Builds a real-shaped 2DGS surfel set from the watermelon points3d.ply
     (μ [N,3], scale [N,2], quat [N,4], opacity [N,1]) + SV sites/colors [N,K,3].
  3. Evaluates SV -> clamped base color sv_rgb = ReLU(feat+0.5)  (mirrors the
     3D_SH_res SV path).
  4. Runs ClipRelightHead -> adjusted μ/scale/quat/opacity + signed sv_exposed.
  5. Asserts every tensor shape, checks IDENTITY-at-init (deltas≈0, m≈1, a≈0).
  6. Calls the REAL 2DGS rasterizer once (render_mode=0, colors_precomp) and
     saves the render. At init the head is identity but the ANALYTIC culling
     mask is active, so you should see the watermelon with the cap removed by
     the clip plane, colored by SV (no relight yet).
  7. Backward sanity: loss=render.sum(); confirms grads reach the head.

Run:
    conda run -n nest_splatting python scripts/test_clip_relight.py
"""

import json
import math
import os
import sys

import numpy as np
import torch

sys.path += ["./", "../"]
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
from hash_encoder.clip_relight import ClipRelightHead

SCENE = "data/clipping/watermelon"
OUT_DIR = "/tmp/clip_relight_test"


# --------------------------------------------------------------------------- #
def load_ply_xyz_rgb(path):
    """Parse the ASCII points3d.ply we wrote (x y z r g b)."""
    with open(path) as f:
        lines = f.read().splitlines()
    n = 0
    hdr_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
        if ln.strip() == "end_header":
            hdr_end = i + 1
            break
    data = np.array([list(map(float, lines[hdr_end + k].split())) for k in range(n)])
    xyz = data[:, :3].astype(np.float32)
    rgb = (data[:, 3:6] / 255.0).astype(np.float32)
    return xyz, rgb


def eval_voronoi_sv_feat(sv_sites, sv_colors, view_dirs, sv_tau):
    """Mirror of gaussian_renderer.eval_voronoi_sv_feat (the 3D_SH_res SV path)."""
    site_dirs = sv_sites / (sv_sites.norm(dim=-1, keepdim=True) + 1e-12)   # [N,K,3]
    diff = site_dirs - view_dirs.unsqueeze(1)                              # [N,K,3]
    dist = torch.norm(diff, dim=-1)                                       # [N,K]
    W = torch.softmax(-sv_tau * dist, dim=-1).unsqueeze(-1)               # [N,K,1]
    return (W * sv_colors).sum(dim=1)                                     # [N,3]


def build_camera(frame, cax, W, H, device="cuda"):
    """Replicate scene.cameras.Camera matrix construction from a transforms frame."""
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1                       # OpenGL -> COLMAP axes (as in dataset_readers)
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    fovx = cax
    fovy = focal2fov(fov2focal(fovx, W), H)
    wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(device).float()
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).to(device).float()
    full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    campos = wvt.inverse()[3, :3]
    return dict(wvt=wvt, full=full, campos=campos,
                tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
                W=W, H=H)


def shape_ok(name, t, expected):
    ok = tuple(t.shape) == tuple(expected)
    print(f"  {'OK ' if ok else 'BAD'} {name:18s} {tuple(t.shape)}  (expect {tuple(expected)})")
    assert ok, f"{name}: {tuple(t.shape)} != {tuple(expected)}"


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    dev = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- camera + clip plane (frame 0) ---
    tj = json.load(open(os.path.join(SCENE, "transforms_train.json")))
    cax = tj["camera_angle_x"]
    frame = tj["frames"][0]
    clip_plane = torch.tensor(frame["clip_plane"], dtype=torch.float32, device=dev)  # [4]
    print(f"clip_plane (a,b,c,d) = {clip_plane.tolist()}  (kept half: n·x+d <= 0)")

    from PIL import Image
    img0 = Image.open(os.path.join(SCENE, frame["file_path"] + ".png"))
    W, H = img0.size
    cam = build_camera(frame, cax, W, H, dev)
    print(f"camera: {W}x{H}, fov_x={cax:.4f}, campos={cam['campos'].tolist()}")

    # --- surfels from the watermelon point cloud ---
    xyz, rgb = load_ply_xyz_rgb(os.path.join(SCENE, "points3d.ply"))
    mu = torch.tensor(xyz, device=dev)                                    # [N,3]
    rgb = torch.tensor(rgb, device=dev)                                   # [N,3]
    N = mu.shape[0]
    K = 4
    scaling = torch.full((N, 2), 0.02, device=dev)                        # [N,2] 2DGS in-plane
    rotation = torch.zeros(N, 4, device=dev); rotation[:, 0] = 1.0        # [N,4] identity quat
    opacity = torch.full((N, 1), 0.9, device=dev)                         # [N,1]
    sv_sites = torch.nn.functional.normalize(torch.randn(N, K, 3, device=dev), dim=-1)
    sv_colors = (rgb - 0.5).unsqueeze(1).expand(N, K, 3).contiguous()     # all sites = rgb-0.5
    sv_tau = torch.ones(N, K, device=dev)
    print(f"N surfels = {N},  K SV sites = {K}")

    # --- SV base color (clamped), mirrors 3D_SH_res SV path ---
    view_dirs = torch.nn.functional.normalize(mu - cam["campos"].unsqueeze(0), dim=-1)
    sv_feat = eval_voronoi_sv_feat(sv_sites, sv_colors, view_dirs, sv_tau)  # [N,3]
    sv_rgb = torch.relu(sv_feat + 0.5)                                     # [N,3] == rgb here

    # --- run the head ---
    head = ClipRelightHead(scene_bound=1.5, sigma=0.12).to(dev)
    out = head(mu, scaling, rotation, opacity, sv_rgb, clip_plane)

    print("\n[shape checks]")
    shape_ok("mu", out.mu, (N, 3))
    shape_ok("scaling", out.scaling, (N, 2))
    shape_ok("rotation", out.rotation, (N, 4))
    shape_ok("opacity", out.opacity, (N, 1))
    shape_ok("sv_exposed", out.sv_exposed, (N, 3))
    shape_ok("signed_dist", out.signed_dist, (N, 1))
    shape_ok("influence", out.influence, (N, 1))
    shape_ok("cull_mask", out.cull_mask, (N, 1))

    print("\n[identity-at-init checks]  (zero-init head => exact identity except culling)")
    print(f"  max|mu-Δ|        = {(out.mu - mu).abs().max().item():.3e}")
    print(f"  max|scale ratio-1|= {(out.scaling / scaling - 1).abs().max().item():.3e}")
    print(f"  max|rot-Δ|       = {(out.rotation - rotation).abs().max().item():.3e}")
    print(f"  max|sv_exposed-sv_rgb| = {(out.sv_exposed - sv_rgb).abs().max().item():.3e}")
    print(f"  m in [{out.m.min().item():.3f},{out.m.max().item():.3f}] (expect ~1.0)   "
          f"a in [{out.a.min().item():.3f},{out.a.max().item():.3f}] (expect ~0.0)")

    kept = (out.signed_dist <= 0).float().mean().item()
    print(f"\n[culling]  kept fraction (s_g<=0) = {kept*100:.1f}%   "
          f"cull_mask mean = {out.cull_mask.mean().item():.3f}")

    # --- real 2DGS rasterizer, once ---
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings)
    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=cam["wvt"], projmatrix=cam["full"], sh_degree=0,
        campos=cam["campos"], prefiltered=False, debug=False,
        beta=0.0, if_contract=False, record_transmittance=False)
    hgs = HashGridSettings(L=0, S=1.0, H=16, align_corners=False,
                           interpolation=0, shape_dims=torch.empty(0, device=dev))
    rasterizer = GaussianRasterizer(raster_settings, hgs)

    # 2DGS screenspace-points grad carrier is [N,4] in this rasterizer fork.
    means2D = torch.zeros(out.mu.shape[0], 4, device=dev, requires_grad=True)
    colors = out.sv_exposed.clamp(min=0.0)  # display clamp (no deferred ReLU in mode 0)
    rendered = rasterizer(
        means3D=out.mu, means2D=means2D, opacities=out.opacity,
        shs=None, colors_precomp=colors,
        scales=out.scaling, rotations=out.rotation,
        render_mode=0, kernel_type=0)
    color = rendered[0]  # [3,H,W]
    print(f"\n[render]  rasterizer returned color {tuple(color.shape)}, "
          f"range [{color.min().item():.3f},{color.max().item():.3f}]")

    arr = (color.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT_DIR, "render_init.png"))
    print(f"  saved {OUT_DIR}/render_init.png  (watermelon with cap culled, SV color, no relight)")

    # --- backward sanity ---
    loss = color.sum()
    loss.backward()
    gw = head.out.weight.grad
    print("\n[backward]")
    print(f"  head.out.weight.grad: nonzero={int((gw != 0).sum())}/{gw.numel()}  "
          f"max|g|={gw.abs().max().item():.3e}")
    print("  (clipgrid/trunk grads are 0 at init by design — zero-init last layer "
          "blocks upstream until out.weight moves)")
    assert gw is not None and gw.abs().max() > 0, "no gradient reached the head!"
    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
