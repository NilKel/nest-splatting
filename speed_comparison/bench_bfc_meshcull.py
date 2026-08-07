"""
Bench the LEAN CONIC baked renderer under the BFC + offset-meshcull deployment
config, following scripts/benchmark_baked.py's methodology: ONE render loop
cycling through all test views (view-switch cost included), `--num_warmup`
untimed frames then `--num_benchmark` cuda-event-timed frames, FPS = mean over
the cycling loop. Per-view mesh depth maps are raycast ONCE up front from the
(normal-offset) mesh; the backface cull runs INSIDE the CUDA preprocess
(set_backface_cull — quat 3rd-axis normal, centroid-oriented, one dot product)
so frames pay the true deployment cost, not a Python mask.

Variants:
  A_plain        — no culls
  B_bfc          — CUDA backface cull only (training semantics)
  C_bfc_meshocc  — BFC + per-view occluder depth map (deployment config);
                   the per-frame set_occluder_depth pointer swap is INSIDE
                   the timed loop, as a real renderer would pay it.

Quality: one render per view per variant vs GT → per-frame PSNR/SSIM/LPIPS
+ aggregates. PNGs via --save_png_dir, JSON via --out_json.
"""
from __future__ import annotations
import os, sys, json, glob, math, pickle, argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

import diff_surfel_bake_render as _prod_mod                # noqa: F401 (aliasing order)
import diff_surfel_bake_render_lean_occ as _lean_occ_mod

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from hash_encoder.config import Config
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    return Namespace(**json.load(open(model_path / "args.json")))


class MeshDepthBaker:
    """Raycast cam-Z depth from a (optionally normal-offset) proxy mesh."""

    def __init__(self, mesh_path: str, inflate_margin_normal: float = 0.0):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if inflate_margin_normal != 0.0:
            mesh.compute_vertex_normals(normalized=True)
            V = np.asarray(mesh.vertices)
            N = np.asarray(mesh.vertex_normals)
            mesh.vertices = o3d.utility.Vector3dVector(V + float(inflate_margin_normal) * N)
            print(f"[baker] offset {V.shape[0]:,} verts by {inflate_margin_normal:+.4f} m along normals")
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
        R = C2W[:3, :3].astype(np.float32)
        t = C2W[:3, 3].astype(np.float32)
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
        if margin != 0.0:
            depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.0)
    p.add_argument("--mesh_normal_margin", type=float, default=-0.006)
    p.add_argument("--bfc_cos", type=float, default=0.2)
    # benchmark_baked.py parity: 10 warmup + 100 timed frames, cycling views.
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--num_benchmark", type=int, default=100)
    p.add_argument("--save_png_dir", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    mp = Path(args.model_path)
    train_args = load_train_args(mp)
    train_args.model_path = str(mp)
    train_args.eval = True
    cfg = Config(str(mp / "config.yaml")) if (mp / "config.yaml").exists() \
        else Config(train_args.yaml)
    it = args.iteration
    if it < 0:
        ngps = glob.glob(str(mp / "ngp_*.pth"))
        it = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                 for f in ngps)
    print(f"[bench_bfc] iter={it}  bfc_cos={args.bfc_cos}  "
          f"normal_margin={args.mesh_normal_margin}  depth_margin={args.mesh_margin}")

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    PipelineParams(tp).extract(train_args)

    baked_dir = mp / "baked_atlas"
    bake_meta = json.load(open(baked_dir / "bake_meta.json"))
    atlas_rects = torch.load(baked_dir / "atlas_rects.pt", weights_only=False)
    if isinstance(atlas_rects, dict):
        atlas_rects = atlas_rects["rects"]
    atlas_rects = atlas_rects.cuda().float()
    bc7_bytes = (baked_dir / "atlas_texture.bc7").read_bytes()
    bc7_W = int(bake_meta["atlas_width"])
    bc7_H = int(bake_meta["atlas_height"])
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
    print(f"[bench_bfc] baked N={gaussians.get_xyz.shape[0]:,}")

    from scripts.benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians) or {'sites': None, 'tau': None,
                                             'colors': None, 'K': 0}

    lean = _lean_occ_mod
    kernel_type = getattr(train_args, "kernel", "gaussian")
    _KMAP = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3,
             'beta_scaled': 4, 'nexel': 5}
    _kt = _KMAP.get(kernel_type, 0)
    _compact_mult = float(getattr(train_args, "fastgs_mult", 1.0)) \
        if getattr(train_args, "fastgs", False) else 1.0

    pkg = lean.prepare_gaussian_inputs(gaussians, sh_degree=train_args.sh_degree,
                                       kernel_type=_kt)
    lean.set_activation_bias(float(bake_meta.get("sh_bias", 0.5)),
                             float(bake_meta.get("res_bias", 0.0)))
    lean.set_compact_mult(_compact_mult)
    lean.set_beta_mult(_compact_mult)
    lean.set_residual_mode(int(bake_meta.get("residual_mode", 0)))
    lean.set_atlas_bc7(bc7_tensor, bc7_W, bc7_H,
                       float(bake_meta["atlas_offset"]), float(bake_meta["atlas_scale"]))
    atlas_gate = torch.zeros(3, dtype=torch.float16, device="cuda")

    aabb_mode = int(getattr(train_args, "_aabb_mode", 5))
    sort_mode = int(getattr(train_args, "_sort_mode", 0))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg.surfel.tg_beta) if hasattr(cfg.surfel, "tg_beta") else 0.0

    def _make_r(cam):
        return lean.get_rasterizer(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)

    orig_shapes = pkg.get("shapes", None)

    def rasterize(cam):
        r = _make_r(cam)
        color, _ = r(
            means3D=pkg['means3D'], opacities=pkg['opacities'], shs=pkg['shs'],
            scales=pkg['scales'], rotations=pkg['rotations'], shapes=orig_shapes,
            kernel_type=pkg['kernel_type'], atlas_texture=atlas_gate,
            atlas_rects=atlas_rects, atlas_width=bc7_W,
            voronoi_sites=sv_state['sites'], voronoi_tau=sv_state['tau'],
            voronoi_colors=sv_state['colors'], voronoi_K=sv_state['K'],
            is_textured=pkg.get('is_textured'), scaling_z=pkg.get('scaling_z'))
        return color

    # ---- One-time setup: offset mesh → per-view depth maps; BFC centroid ----
    baker = MeshDepthBaker(args.mesh_ply,
                           inflate_margin_normal=float(args.mesh_normal_margin))
    test_cams = list(scene.getTestCameras())
    print(f"[bench_bfc] raycasting {len(test_cams)} per-view depth maps once ...")
    mesh_zs = [baker.cam_depth(c, args.mesh_margin) for c in test_cams]
    centroid = gaussians.get_xyz.detach().mean(dim=0).tolist()

    VAR = ["A_plain", "B_bfc", "C_bfc_meshocc"]

    def arm(variant: str):
        """Install the variant's device-global cull state."""
        if variant == "A_plain":
            lean.clear_backface_cull()
            lean.clear_occluder_depth()
        elif variant == "B_bfc":
            lean.set_backface_cull(args.bfc_cos, centroid)
            lean.clear_occluder_depth()
        else:
            lean.set_backface_cull(args.bfc_cos, centroid)
            # occluder swaps per frame inside the loop

    # ---- FPS: benchmark_baked-style cycling loop per variant ----
    print(f"[bench_bfc] FPS loop: {args.num_warmup} warmup + "
          f"{args.num_benchmark} timed frames, cycling {len(test_cams)} views\n")
    fps_agg = {}
    for v in VAR:
        arm(v)

        def frame(i):
            cam = test_cams[i % len(test_cams)]
            if v == "C_bfc_meshocc":
                # Real renderers pay a per-frame occluder update; include the
                # pointer-swap cost inside the timed region.
                lean.set_occluder_depth(mesh_zs[i % len(test_cams)])
            return rasterize(cam)

        for i in range(args.num_warmup):
            frame(i)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_benchmark)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_benchmark)]
        for i in range(args.num_benchmark):
            starts[i].record()
            frame(i)
            ends[i].record()
        torch.cuda.synchronize()
        ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(args.num_benchmark)])
        fps_agg[v] = {"mean_ms": float(ms.mean()), "std_ms": float(ms.std()),
                      "fps": float(1000.0 / ms.mean())}
        print(f"  FPS {v:<15s} {ms.mean():7.3f} ± {ms.std():5.3f} ms  → {1000.0/ms.mean():8.1f} fps")
    lean.clear_backface_cull()
    lean.clear_occluder_depth()

    # ---- Quality: one render per view per variant vs GT ----
    print(f"\n[bench_bfc] per-frame quality across {len(test_cams)} test views")
    if args.save_png_dir:
        import imageio.v2 as imageio
        for v in VAR:
            (Path(args.save_png_dir) / v).mkdir(parents=True, exist_ok=True)

    hdr = (f"  {'view':>4} | " + " | ".join(
        f"{v}: {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}" for v in VAR))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    per_view = []
    for idx, cam in enumerate(test_cams):
        gt = cam.original_image[:3].cuda().clamp(0, 1)
        row = {"view": idx, "name": cam.image_name}
        for v in VAR:
            arm(v)
            if v == "C_bfc_meshocc":
                lean.set_occluder_depth(mesh_zs[idx])
            with torch.no_grad():
                img = rasterize(cam).clamp(0, 1)
                row[v] = {
                    "psnr": float(psnr_fn(img, gt).mean().item()),
                    "ssim": float(ssim(img, gt).mean().item()),
                    "lpips": float(lpips(img.unsqueeze(0), gt.unsqueeze(0),
                                         net_type='vgg').item()),
                }
            if args.save_png_dir:
                import imageio.v2 as imageio
                u8 = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                      ).clip(0, 255).astype(np.uint8)
                imageio.imwrite(str(Path(args.save_png_dir) / v / f"{cam.image_name}.png"), u8)
        lean.clear_backface_cull()
        lean.clear_occluder_depth()
        print("  " + f"{idx:>4d} | " + " | ".join(
            f"{row[v]['psnr']:>6.2f} {row[v]['ssim']:>6.4f} {row[v]['lpips']:>6.4f}"
            for v in VAR))
        per_view.append(row)

    print()
    agg = {}
    for v in VAR:
        agg[v] = {
            **fps_agg[v],
            "psnr": float(np.mean([r[v]['psnr'] for r in per_view])),
            "ssim": float(np.mean([r[v]['ssim'] for r in per_view])),
            "lpips": float(np.mean([r[v]['lpips'] for r in per_view])),
        }
        print(f"  TOTAL {v:<15s} ms={agg[v]['mean_ms']:7.3f}  fps={agg[v]['fps']:8.1f}  "
              f"PSNR={agg[v]['psnr']:6.2f}  SSIM={agg[v]['ssim']:6.4f}  "
              f"LPIPS={agg[v]['lpips']:6.4f}")
    print(f"\n  speedup C vs A: {agg['A_plain']['mean_ms']/agg['C_bfc_meshocc']['mean_ms']:.2f}x   "
          f"PSNR delta C-A: {agg['C_bfc_meshocc']['psnr']-agg['A_plain']['psnr']:+.2f} dB")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"per_view": per_view, "aggregate": agg,
                       "config": vars(args)}, f, indent=2)
        print(f"  json → {args.out_json}")


if __name__ == "__main__":
    main()
