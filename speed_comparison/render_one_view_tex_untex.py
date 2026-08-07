"""
Render ONE test view under 4 configurations (LEAN CONIC baked, mesh cull applied):
  UNTEX_noCull   — SV base only (no atlas)
  UNTEX_cull     — SV base only + mesh occluder
  TEX_noCull     — SV + BC7 atlas residual (full)
  TEX_cull       — SV + BC7 atlas residual + mesh occluder

Uses diff_surfel_bake_render_lean_occ. `atlas_texture` arg is a NULL/non-null
gate — None = SH-only, dummy fp16 tensor = atlas-on.
"""
from __future__ import annotations
import os, sys, argparse, pickle, math, json, glob
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
import open3d as o3d

import diff_surfel_bake_render as _prod_mod
import diff_surfel_bake_render_lean_occ as _lean_occ_mod

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from hash_encoder.config import Config


class MeshDepthBaker:
    def __init__(self, mesh_path: str):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def cam_depth(self, cam, margin: float) -> torch.Tensor:
        H, W = cam.image_height, cam.image_width
        fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
        fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        W2C = cam.world_view_transform.detach().cpu().numpy().T
        C2W = np.linalg.inv(W2C)
        js, is_ = np.meshgrid(np.arange(W), np.arange(H))
        xs = (js - cx) / fx
        ys = (is_ - cy) / fy
        dirs_cam = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
        R = C2W[:3, :3].astype(np.float32); t = C2W[:3, 3].astype(np.float32)
        dirs_w = dirs_cam @ R.T
        dirs_w /= np.linalg.norm(dirs_w, axis=-1, keepdims=True)
        origins = np.broadcast_to(t, dirs_w.shape).copy()
        rays = o3d.core.Tensor(
            np.concatenate([origins.reshape(-1, 3), dirs_w.reshape(-1, 3)], axis=1),
            dtype=o3d.core.Dtype.Float32)
        t_hit = self.scene.cast_rays(rays)['t_hit'].numpy().reshape(H, W)
        unnorm = np.sqrt(xs * xs + ys * ys + 1.0).astype(np.float32)
        depth = (t_hit / unnorm).astype(np.float32)
        hit = np.isfinite(depth)
        depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    return Namespace(**json.load(open(model_path / "args.json")))


def to_u8(img):
    return (img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--mesh_ply", required=True)
    ap.add_argument("--mesh_margin", type=float, default=0.03)
    ap.add_argument("--view", type=int, default=0, help="Test cam index.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    mp = Path(args.model_path)
    train_args = load_train_args(mp)
    train_args.model_path = str(mp)
    train_args.eval = True
    cfg = Config(str(mp / "config.yaml"))
    it = args.iteration
    if it < 0:
        ngps = glob.glob(str(mp / "ngp_*.pth"))
        it = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                 for f in ngps)
    print(f"[tex_untex] iter={it}")

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)

    baked_dir = mp / "baked_atlas"
    bake_meta = json.load(open(baked_dir / "bake_meta.json"))
    atlas_rects = torch.load(baked_dir / "atlas_rects.pt", weights_only=False)
    if isinstance(atlas_rects, dict):
        atlas_rects = atlas_rects["rects"]
    atlas_rects = atlas_rects.cuda().float()
    bc7_bytes = (baked_dir / "atlas_texture.bc7").read_bytes()
    bc7_W = int(bake_meta["atlas_width"])
    bc7_H = int(bake_meta["atlas_height"])
    atlas_offset = float(bake_meta["atlas_offset"])
    atlas_scale = float(bake_meta["atlas_scale"])
    bc7_tensor = torch.from_numpy(np.frombuffer(bc7_bytes, dtype=np.uint8).copy()).cuda()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False,
                  full_args=train_args)
    baked_ply = baked_dir / "baked.ply"
    if baked_ply.exists():
        gaussians.load_ply(str(baked_ply), args=train_args)
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel
    gaussians.feature_mode = getattr(train_args, "feature", "sh")
    gaussians._sv_training_flag = False
    if hasattr(gaussians, "update_sites_mask"):
        gaussians.update_sites_mask()

    from scripts.benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians) or {'sites': None, 'tau': None,
                                              'colors': None, 'K': 0}

    lean = _lean_occ_mod
    _KMAP = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3,
             'beta_scaled': 4, 'nexel': 5}
    kernel_str = getattr(train_args, "kernel", "gaussian")
    _kt = _KMAP.get(kernel_str, 0)
    _compact_mult = float(getattr(train_args, "fastgs_mult", 1.0)) \
        if getattr(train_args, "fastgs", False) else 1.0
    _residual_mode = int(bake_meta.get("residual_mode", 0))

    pkg = lean.prepare_gaussian_inputs(gaussians,
                                       sh_degree=train_args.sh_degree,
                                       kernel_type=_kt)
    lean.set_activation_bias(float(bake_meta.get("sh_bias", 0.5)),
                             float(bake_meta.get("res_bias", 0.0)))
    lean.set_compact_mult(_compact_mult)
    lean.set_beta_mult(_compact_mult)
    lean.set_residual_mode(_residual_mode)
    lean.set_atlas_bc7(bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale)

    aabb_mode = int(getattr(train_args, "_aabb_mode", 5))
    sort_mode = int(getattr(train_args, "_sort_mode", 0))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg.surfel.tg_beta) if hasattr(cfg.surfel, "tg_beta") else 0.0

    # atlas_texture is a NULL/non-null gate; None = SH-only, dummy = atlas-on
    atlas_gate = torch.zeros(3, dtype=torch.float16, device="cuda")
    orig_shapes = pkg.get("shapes", None)

    def _rasterize(cam, with_atlas: bool):
        r = lean.get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)
        color, _ = r(
            means3D=pkg['means3D'], opacities=pkg['opacities'],
            shs=pkg['shs'], scales=pkg['scales'], rotations=pkg['rotations'],
            shapes=orig_shapes, kernel_type=pkg['kernel_type'],
            atlas_texture=atlas_gate if with_atlas else None,
            atlas_rects=atlas_rects if with_atlas else None,
            atlas_width=bc7_W,
            voronoi_sites=sv_state['sites'], voronoi_tau=sv_state['tau'],
            voronoi_colors=sv_state['colors'], voronoi_K=sv_state['K'],
            is_textured=pkg.get('is_textured'), scaling_z=pkg.get('scaling_z'))
        return color

    test_cams = scene.getTestCameras()
    cam = test_cams[args.view]
    print(f"[tex_untex] rendering view {args.view} ({cam.image_name})")

    baker = MeshDepthBaker(args.mesh_ply)
    mesh_z = baker.cam_depth(cam, args.mesh_margin)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for tag, with_atlas, install in [
            ("UNTEX_noCull", False, False),
            ("UNTEX_cull",   False, True),
            ("TEX_noCull",   True,  False),
            ("TEX_cull",     True,  True),
        ]:
            if install:
                lean.set_occluder_depth(mesh_z)
            else:
                lean.clear_occluder_depth()
            img = _rasterize(cam, with_atlas).clamp(0, 1)
            u8 = to_u8(img)
            out = os.path.join(args.out_dir,
                               f"view{args.view:02d}_{cam.image_name}_{tag}.png")
            imageio.imwrite(out, u8)
            print(f"  {tag:<15s} → {out}")
        lean.clear_occluder_depth()
        # GT for reference
        gt = cam.original_image[:3].cuda().clamp(0, 1)
        imageio.imwrite(os.path.join(args.out_dir,
                                     f"view{args.view:02d}_{cam.image_name}_GT.png"),
                        to_u8(gt))


if __name__ == "__main__":
    main()
