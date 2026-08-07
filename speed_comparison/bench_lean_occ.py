"""
Bench the LEAN CONIC baked renderer FPS with and without the per-Gauss
proxy-mesh Z-cull (installed via set_occluder_depth per view).

Compares:
  A. LEAN CONIC baseline           — no occluder (current fastest baseline)
  B. LEAN CONIC + occluder mesh    — preprocess drops surfels behind mesh
                                     (fewer surfels reach sort/render)

Also reports per-view cull rate (% of surfels killed by the occluder) so we
can see whether the cull is doing meaningful work at each viewpoint.

Uses the same load path as bench_lean_vs_prod.py to get the baked BC7 atlas
installed on the LEAN_OCC module.
"""
from __future__ import annotations
import os, sys, json, glob, math, pickle, argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

# Import BEFORE scene to avoid sys.modules aliasing surprises.
import diff_surfel_bake_render as _prod_mod                # for prepare_gaussian_inputs
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


def bench(render_fn, cam, num_warmup, num_bench):
    """Time render_fn(cam) via cuda events; returns dict with ms + fps."""
    for _ in range(num_warmup):
        _ = render_fn(cam)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    for i in range(num_bench):
        starts[i].record()
        _ = render_fn(cam)
        ends[i].record()
    torch.cuda.synchronize()
    ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(num_bench)])
    return {"mean_ms": float(ms.mean()), "std_ms": float(ms.std()),
            "fps": float(1000.0 / ms.mean())}


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
        depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Baked model dir (needs args.pkl, config.yaml, ngp_*.pth, "
                        "baked_atlas/{atlas_texture.bc7, atlas_rects.pt, bake_meta.json, "
                        "baked.ply}).")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--mesh_ply", required=True)
    p.add_argument("--mesh_margin", type=float, default=0.03)
    p.add_argument("--num_warmup", type=int, default=100)
    p.add_argument("--num_bench", type=int, default=300)
    p.add_argument("--n_views", type=int, default=6,
                   help="Number of test views to bench (each with fresh cuda-events).")
    p.add_argument("--save_png_dir", type=str, default=None,
                   help="If set, render EVERY test view (both with and without "
                        "occluder) as PNGs into this dir. Skips FPS bench when set.")
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
    print(f"[bench_lean_occ] iter={it}")

    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    pipe = PipelineParams(tp).extract(train_args)

    # Load the BAKED point cloud + atlas (mirrors bench_lean_vs_prod.py).
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

    # Load scene + baked PLY (Scene will read the training PLY; we swap to
    # baked.ply below like benchmark_baked does).
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

    print(f"[bench_lean_occ] baked N={gaussians.get_xyz.shape[0]:,}")

    # SV state for the fused voronoi path.
    from scripts.benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians) or {'sites': None, 'tau': None,
                                              'colors': None, 'K': 0}

    # LEAN_OCC setup.
    lean = _lean_occ_mod
    kernel_type = getattr(train_args, "kernel", "gaussian")
    _KMAP = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3,
             'beta_scaled': 4, 'nexel': 5}
    _kt = _KMAP.get(kernel_type, 0)
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

    # atlas_texture is a NULL/non-null gate in the LEAN CONIC kernel — not the
    # actual atlas data (that comes from set_atlas_bc7). Pass a dummy fp16
    # tensor so the atlas-read branch is enabled; None here silently degrades
    # to SH-only, which matches benchmark_baked's baked_sh_only mode.
    atlas_gate = torch.zeros(3, dtype=torch.float16, device="cuda")

    aabb_mode = int(getattr(train_args, "_aabb_mode", 5))
    sort_mode = int(getattr(train_args, "_sort_mode", 0))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg.surfel.tg_beta) if hasattr(cfg.surfel, "tg_beta") else 0.0

    def _make_r(cam):
        return lean.get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)

    orig_shapes = pkg.get("shapes", None)

    def rasterize(cam):
        r = _make_r(cam)
        color, _ = r(
            means3D=pkg['means3D'],
            opacities=pkg['opacities'],
            shs=pkg['shs'],
            scales=pkg['scales'],
            rotations=pkg['rotations'],
            shapes=orig_shapes,
            kernel_type=pkg['kernel_type'],
            atlas_texture=atlas_gate,
            atlas_rects=atlas_rects,
            atlas_width=bc7_W,
            voronoi_sites=sv_state['sites'],
            voronoi_tau=sv_state['tau'],
            voronoi_colors=sv_state['colors'],
            voronoi_K=sv_state['K'],
            is_textured=pkg.get('is_textured'),
            scaling_z=pkg.get('scaling_z'),
        )
        return color

    baker = MeshDepthBaker(args.mesh_ply)
    test_cams = scene.getTestCameras()
    n = min(args.n_views, len(test_cams))
    picked = list(range(n))
    print(f"[bench_lean_occ] benching {n} test views ({args.num_warmup} warmup + "
          f"{args.num_bench} timed each)\n")

    header = f"  {'view':>4} {'A_noOccl ms':>13} {'A_fps':>8}   " \
             f"{'B_occl ms':>11} {'B_fps':>8}   {'speedup':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for slot in picked:
        cam = test_cams[slot]

        # Compute per-view mesh depth on GPU.
        mesh_z = baker.cam_depth(cam, args.mesh_margin)

        # --- Variant A: no occluder ---
        lean.clear_occluder_depth()
        A = bench(rasterize, cam, args.num_warmup, args.num_bench)

        # --- Variant B: mesh occluder installed ---
        lean.set_occluder_depth(mesh_z)
        B = bench(rasterize, cam, args.num_warmup, args.num_bench)
        lean.clear_occluder_depth()

        speedup = A['mean_ms'] / B['mean_ms']
        print(f"  {slot:>4d} {A['mean_ms']:>13.3f} {A['fps']:>8.1f}   "
              f"{B['mean_ms']:>11.3f} {B['fps']:>8.1f}   {speedup:>7.2f}x")
        results.append({"view": slot, "A": A, "B": B, "speedup": speedup})

    # Aggregate.
    mean_A_ms = np.mean([r['A']['mean_ms'] for r in results])
    mean_B_ms = np.mean([r['B']['mean_ms'] for r in results])
    print()
    print(f"  {'mean':>4} {mean_A_ms:>13.3f} {1000/mean_A_ms:>8.1f}   "
          f"{mean_B_ms:>11.3f} {1000/mean_B_ms:>8.1f}   "
          f"{mean_A_ms/mean_B_ms:>7.2f}x")

    # ------- Quality on ALL test views (both configs) -------
    print(f"\n[quality] PSNR / SSIM / LPIPS across {len(test_cams)} test cams")
    header = f"  {'config':<20} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _eval(cams, install_occluder: bool):
        psnrs, ssims, lps = [], [], []
        with torch.no_grad():
            for c in cams:
                if install_occluder:
                    mz = baker.cam_depth(c, args.mesh_margin)
                    lean.set_occluder_depth(mz)
                else:
                    lean.clear_occluder_depth()
                img = rasterize(c).clamp(0, 1)
                gt = c.original_image[:3].cuda().clamp(0, 1)
                psnrs.append(float(psnr_fn(img, gt).mean().item()))
                ssims.append(float(ssim(img, gt).mean().item()))
                lps.append(float(lpips(img.unsqueeze(0), gt.unsqueeze(0),
                                       net_type='vgg').item()))
        lean.clear_occluder_depth()
        return (float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lps)))

    p_a, s_a, l_a = _eval(test_cams, install_occluder=False)
    p_b, s_b, l_b = _eval(test_cams, install_occluder=True)
    print(f"  {'A no-occluder':<20} {p_a:>10.3f} {s_a:>8.4f} {l_a:>8.4f}")
    print(f"  {'B with mesh occ.':<20} {p_b:>10.3f} {s_b:>8.4f} {l_b:>8.4f}")
    print(f"  {'delta (B-A)':<20} {p_b-p_a:>+10.3f} {s_b-s_a:>+8.4f} {l_b-l_a:>+8.4f}")

    # ------- Save per-view PNGs (mirrors benchmark_baked's sh_atlas/) -------
    if args.save_png_dir:
        import imageio.v2 as imageio
        out_no = Path(args.save_png_dir) / "no_cull"
        out_yes = Path(args.save_png_dir) / "with_mesh_cull"
        out_no.mkdir(parents=True, exist_ok=True)
        out_yes.mkdir(parents=True, exist_ok=True)
        print(f"\n[save_png] writing {len(test_cams)} test views to "
              f"{args.save_png_dir}/{{no_cull,with_mesh_cull}}/ ...")

        with torch.no_grad():
            for c in test_cams:
                # no occluder
                lean.clear_occluder_depth()
                im = rasterize(c).clamp(0, 1)
                u8 = (im.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                      ).clip(0, 255).astype(np.uint8)
                imageio.imwrite(str(out_no / f"{c.image_name}.png"), u8)
                # with mesh cull
                mz = baker.cam_depth(c, args.mesh_margin)
                lean.set_occluder_depth(mz)
                im = rasterize(c).clamp(0, 1)
                u8 = (im.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                      ).clip(0, 255).astype(np.uint8)
                imageio.imwrite(str(out_yes / f"{c.image_name}.png"), u8)
        lean.clear_occluder_depth()
        print(f"[save_png] done.")


if __name__ == "__main__":
    main()
