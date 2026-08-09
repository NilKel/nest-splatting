#!/usr/bin/env python3
"""4-mode benchmark: {production, lean} × {SH-only, SH+atlas} on a baked
RD_SV checkpoint. Tests the impact of the lean-fork's shared-memory strip
independently of the atlas fetch.

Modes:
  A: production rasterizer, SH-only  (no atlas)
  B: production rasterizer, SH+atlas (BC7)
  C: lean rasterizer,       SH-only
  D: lean rasterizer,       SH+atlas (BC7)

Each mode runs 200 warmup + 400 timed frames using CUDA-event timing on
the FIRST test camera (same protocol as bench_step_by_step.py).
"""

import os, sys, json, glob, math, argparse, pickle
from argparse import Namespace
from pathlib import Path

# Import the bake_render modules BEFORE scene / benchmark_baked to prevent
# sys.modules aliasing (res_3d_paired mode reassigns diff_surfel_bake_render).
import diff_surfel_bake_render as _prod_mod
import diff_surfel_bake_render_lean as _lean_mod
# res_3d_paired uses a dedicated lean module that ports the paired branches
# (is_textured + ewa_conic + mode-2 signed passthrough) back into the CONIC
# fast path. Imported eagerly so sys.modules aliasing can't confuse it.
try:
    import diff_surfel_bake_render_paired_lean as _lean_paired_mod
except ImportError:
    _lean_paired_mod = None

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from hash_encoder.config import Config
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def eval_quality(rasterize_fn, test_cameras, tag):
    """Render each test camera and compute PSNR/SSIM/LPIPS vs original_image."""
    psnrs, ssims, lps = [], [], []
    with torch.no_grad():
        for cam in test_cameras:
            img = rasterize_fn(cam).clamp(0, 1)
            gt = cam.original_image[:3].cuda()
            psnrs.append(psnr(img, gt).mean().item())
            ssims.append(ssim(img, gt).item())
            lps.append(lpips(img.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item())
    mp, ms, ml = float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lps))
    print(f"  {tag:<45} PSNR={mp:6.3f}  SSIM={ms:6.4f}  LPIPS={ml:6.4f}")
    return {"psnr": mp, "ssim": ms, "lpips": ml}


def load_train_args(model_path: Path) -> Namespace:
    apkl = model_path / "args.pkl"
    if apkl.exists():
        return pickle.load(open(apkl, "rb"))
    ajson = model_path / "args.json"
    return Namespace(**json.load(open(ajson)))


def bench(rasterize_fn, cam, num_warmup, num_bench, tag):
    for _ in range(num_warmup):
        _ = rasterize_fn(cam)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_bench)]
    for i in range(num_bench):
        starts[i].record()
        _ = rasterize_fn(cam)
        ends[i].record()
    torch.cuda.synchronize()
    ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(num_bench)])
    mean, std = float(ms.mean()), float(ms.std())
    fps = 1000.0 / mean if mean > 0 else 0
    print(f"  {tag:<45} {mean:.4f} ± {std:.4f} ms   {fps:>8.1f} FPS")
    return {"tag": tag, "ms_mean": mean, "ms_std": std, "fps": fps}


def load_bc7(bake_dir):
    """Load BC7 atlas bytes + metadata."""
    bc7_path = os.path.join(bake_dir, "atlas_texture.bc7")
    meta_path = os.path.join(bake_dir, "bake_meta.json")
    meta = json.load(open(meta_path))
    bc7_W = int(meta["atlas_width"])
    bc7_H = int(meta["atlas_height"])
    offset = float(meta.get("atlas_offset", 0.0))
    scale  = float(meta.get("atlas_scale", 1.0))
    bytes_tensor = torch.from_numpy(np.fromfile(bc7_path, dtype=np.uint8)).cuda()
    return bytes_tensor, bc7_W, bc7_H, offset, scale, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--bake_dir",   type=Path, required=True)
    ap.add_argument("--num_warmup", type=int, default=50)
    ap.add_argument("--num_benchmark", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_axis", type=float, default=None,
                    help="Drop Gauss whose max(get_scaling axis) exceeds this. "
                         "For the bloat A/B test — keeps geometry pruned but same bake otherwise.")
    args = ap.parse_args()

    train_args = load_train_args(args.model_path)
    train_args.model_path = str(args.model_path)
    train_args.eval = True
    cfg_yaml = args.model_path / "config.yaml"
    cfg_model = Config(str(cfg_yaml)) if cfg_yaml.exists() else Config(train_args.yaml)
    ngp_files = glob.glob(str(args.model_path / "ngp_*.pth"))
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                    for f in ngp_files)

    # Load baked Gaussians (SV state included).
    gaussians = GaussianModel(train_args.sh_degree)
    gaussians.load_ply(str(args.bake_dir / "baked.ply"), args=train_args)
    if hasattr(train_args, "kernel"):
        gaussians.kernel_type = train_args.kernel
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    if hasattr(gaussians, "update_sites_mask"):
        try:
            gaussians._sv_training_flag = False
            gaussians.update_sites_mask()
        except Exception:
            pass

    # Load cameras.
    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    _ = PipelineParams(temp_parser)
    dataset = model_params.extract(train_args)
    scene = Scene(dataset, GaussianModel(train_args.sh_degree),
                  load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    print(f"[bench] {len(test_cameras)} test cameras, {gaussians.get_xyz.shape[0]:,} Gauss")
    cam = test_cameras[0]

    # Load atlas rects + BC7 payload.
    atlas_rects = torch.load(args.bake_dir / "atlas_rects.pt").cuda()

    # --max_axis bloat filter: drop Gauss whose largest activated in-plane
    # axis exceeds the threshold. Filters BOTH the model tensors and the
    # per-Gauss atlas rects so their indexing stays aligned. The atlas
    # texture itself is not resliced — the filtered Gauss just don't get
    # rendered; their atlas rects go unread. Same bake, fewer contributors.
    if args.max_axis is not None:
        with torch.no_grad():
            axes = gaussians.get_scaling.max(dim=-1).values  # [N] activated in-plane max
            keep = (axes <= args.max_axis).cpu()
            N0 = int(keep.numel())
            Nkeep = int(keep.sum())
            print(f"[bench] --max_axis {args.max_axis} → keep {Nkeep:,}/{N0:,} ({100*Nkeep/N0:.2f}%)")
            for name in ("_xyz", "_features_dc", "_features_rest", "_scaling",
                         "_rotation", "_opacity", "_shape",
                         "_sv_sites", "_sv_taus", "_sv_colors", "_sv_dc",
                         "_film_params", "_scaling_z", "_appearance_level",
                         "_is_textured"):
                if hasattr(gaussians, name):
                    t = getattr(gaussians, name)
                    if torch.is_tensor(t) and t.numel() > 0 and t.shape[0] == N0:
                        setattr(gaussians, name, t[keep.to(t.device)])
            if hasattr(gaussians, "update_sites_mask"):
                try:    gaussians.update_sites_mask()
                except: pass
            atlas_rects = atlas_rects[keep.to(atlas_rects.device)]
        print(f"[bench] post-filter: {gaussians.get_xyz.shape[0]:,} Gauss, "
              f"atlas_rects {atlas_rects.shape}")
    bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale, bake_meta = load_bc7(args.bake_dir)
    # The BC7 fast path in rasterize_points.cu requires atlas_texture_ptr != 0
    # to activate (uses it as an "atlas is present" signal, even though the BC7
    # cudaArray supersedes the fp16 tensor content). The kernel never READS
    # this tensor under BC7 — it only null-checks the pointer. Pass a tiny
    # fp16 placeholder so the .data_ptr<at::Half>() cast in the binding
    # succeeds. The stored atlas_texture.pt is uint8 shape [H,W,3] and would
    # fail the fp16 dtype cast.
    atlas_texture_fp16 = torch.zeros(3, dtype=torch.float16, device="cuda")
    _compact_mult_train = float(bake_meta.get("compact_mult", 1.0))
    _residual_mode = int(bake_meta.get("residual_mode", 0))
    # kernel2 (untextured EWA half in mixed_3d / res_3d_paired): overrides kernel for
    # untextured Gauss in the CUDA render. Stored in bake_meta as a string.
    _kernel2_str = bake_meta.get("kernel2", None)
    _kernel_map = {"gaussian": 0, "beta": 1, "adaptive_beta": 2,
                   "adaptive_gaussian": 3, "beta_scaled": 4}
    _untex_kernel = _kernel_map.get(_kernel2_str, -1) if _kernel2_str else -1
    # Paired-mode signature: residual_mode==2 means the CUDA kernel emits SIGNED
    # per-Gauss features (no per-Gauss outer ReLU) and Python applies the outer
    # activation AFTER alpha blending. Matches diff_surfel_mixed_3d training
    # forward, which uses F.leaky_relu(_, 0.01) — NOT plain torch.relu — because
    # res_3d_paired defaults `--lru 0.01` (see docs/RES_3D_MODES.md § Stage 1).
    _final_leaky = (_residual_mode == 2)
    _final_lru_slope = float(getattr(train_args, "lru", 0.01)) if _final_leaky else 0.0
    print(f"[bench] atlas={bc7_W}x{bc7_H} scale={atlas_scale:.4f} offset={atlas_offset:.4f}")
    print(f"[bench] bake compact_mult={_compact_mult_train} residual_mode={_residual_mode} "
          f"kernel2={_kernel2_str} untex_kernel={_untex_kernel} "
          f"final_leaky={_final_leaky} lru={_final_lru_slope}")

    # Method-based LEAN module selection: 3D_SH_res family uses plain _lean;
    # res_3d_paired / mixed_3d family uses _lean_paired (has is_textured +
    # ewa_conic branches + mode-2 dispatch in renderBakedCUDA). Everything
    # else falls back to plain _lean.
    _method = getattr(train_args, "method", "3D_SH_res")
    # BENCH_FORCE_PAIRED_LEAN=1 forces the paired_lean module for the LEAN side
    # regardless of method — used to measure the paired-branch overhead on
    # non-paired scenes (should be near-zero if the null-sentinel guards work).
    _force_paired = bool(int(os.environ.get("BENCH_FORCE_PAIRED_LEAN", "0")))
    _use_paired_lean = (_force_paired or _method in ("res_3d_paired", "mixed_3d")) \
                       and _lean_paired_mod is not None
    if _use_paired_lean:
        print(f"[bench] --method {_method}: LEAN routed through diff_surfel_bake_render_paired_lean")
    elif _method in ("res_3d_paired", "mixed_3d"):
        print(f"[bench] --method {_method}: paired_lean not built; falling back to plain _lean "
              f"(will render paired scenes incorrectly)")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_val = float(cfg_model.surfel.tg_beta) if hasattr(cfg_model.surfel, "tg_beta") else 0.0
    aabb_mode = 5   # SnugBox + AccuTile — production baked default
    sort_mode = 0

    kernel_map = {"gaussian": 0, "beta": 1, "adaptive_beta": 2,
                  "adaptive_gaussian": 3, "beta_scaled": 4}
    kernel_type = kernel_map.get(getattr(train_args, "kernel", "gaussian"), 0)

    # SV state.
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from benchmark_baked import _make_sv_state
    sv_state = _make_sv_state(gaussians)

    # =================================================================
    # Production rasterizer setup.
    # =================================================================
    import diff_surfel_bake_render as prod
    pkg_prod = prod.prepare_gaussian_inputs(gaussians,
                                             sh_degree=train_args.sh_degree,
                                             kernel_type=kernel_type)
    prod.set_activation_bias(0.5, 0.0)
    prod.set_compact_mult(_compact_mult_train)
    prod.set_beta_mult(_compact_mult_train)   # in case kernel is beta
    prod.set_residual_mode(_residual_mode)
    if hasattr(prod, "set_untex_kernel"):
        prod.set_untex_kernel(_untex_kernel)
    # Install BC7 atlas for the textured production runs. We control atlas
    # visibility per call by passing atlas_rects=None (SH-only) or the real
    # rects (SH+atlas) — set_atlas_bc7 keeps the texture object live.
    prod.set_atlas_bc7(bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale)

    def _make_r_prod(cam, atlas):
        return prod.get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)

    orig_shapes = pkg_prod.get("shapes", None)

    def rasterize_prod(cam, with_atlas):
        r = _make_r_prod(cam, with_atlas)
        color, _ = r(
            means3D=pkg_prod['means3D'],
            opacities=pkg_prod['opacities'],
            shs=pkg_prod['shs'],
            scales=pkg_prod['scales'],
            rotations=pkg_prod['rotations'],
            shapes=orig_shapes,
            kernel_type=pkg_prod['kernel_type'],
            atlas_texture=atlas_texture_fp16 if with_atlas else None,
            atlas_rects=atlas_rects if with_atlas else None,
            atlas_width=bc7_W,
            voronoi_sites=sv_state['sites'],
            voronoi_tau=sv_state['tau'],
            voronoi_colors=sv_state['colors'],
            voronoi_K=sv_state['K'],
            # res_3d_paired / mixed_3d: is_textured folds untex → skip-texture;
            # scaling_z triggers the untextured EWA-3D-ellipsoid branch.
            is_textured=pkg_prod.get('is_textured'),
            scaling_z=pkg_prod.get('scaling_z'),
        )
        # Paired-mode: kernel emits signed color; ReLU on final blended image
        # (see benchmark_baked.py:_render_baked_frame_with_final_relu).
        if _final_leaky:
            color = torch.nn.functional.leaky_relu(color, negative_slope=_final_lru_slope)
        return color

    # =================================================================
    # Lean rasterizer setup.
    # =================================================================
    lean = _lean_paired_mod if _use_paired_lean else _lean_mod
    pkg_lean = lean.prepare_gaussian_inputs(gaussians,
                                             sh_degree=train_args.sh_degree,
                                             kernel_type=kernel_type)
    lean.set_activation_bias(0.5, 0.0)
    lean.set_compact_mult(_compact_mult_train)
    lean.set_beta_mult(_compact_mult_train)
    # BENCH_UNTEX_MULT: inference-only FastGS footprint mult on the UNTEXTURED
    # EWA 3D half ONLY (lean lane; prod stays the reference). The textured 2D
    # surfels carry the atlas residual, so their footprint is never cropped
    # here. Absent -> byte-identical.
    import os as _os
    _untex_mult = _os.environ.get("BENCH_UNTEX_MULT")
    if _untex_mult and hasattr(lean, "set_untex_mult"):
        lean.set_untex_mult(float(_untex_mult))
        print(f"[bench] BENCH_UNTEX_MULT={_untex_mult} (untextured EWA half only, lean lane)")
    lean.set_residual_mode(_residual_mode)
    if hasattr(lean, "set_untex_kernel"):
        lean.set_untex_kernel(_untex_kernel)
    lean.set_atlas_bc7(bc7_tensor, bc7_W, bc7_H, atlas_offset, atlas_scale)

    def _make_r_lean(cam):
        return lean.get_rasterizer(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5),
            tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform, campos=cam.camera_center,
            sh_degree=train_args.sh_degree, beta=beta_val,
            aabb_mode=aabb_mode, sort_mode=sort_mode)

    orig_shapes_lean = pkg_lean.get("shapes", None)

    def rasterize_lean(cam, with_atlas):
        r = _make_r_lean(cam)
        color, _ = r(
            means3D=pkg_lean['means3D'],
            opacities=pkg_lean['opacities'],
            shs=pkg_lean['shs'],
            scales=pkg_lean['scales'],
            rotations=pkg_lean['rotations'],
            shapes=orig_shapes_lean,
            kernel_type=pkg_lean['kernel_type'],
            atlas_texture=atlas_texture_fp16 if with_atlas else None,
            atlas_rects=atlas_rects if with_atlas else None,
            atlas_width=bc7_W,
            voronoi_sites=sv_state['sites'],
            voronoi_tau=sv_state['tau'],
            voronoi_colors=sv_state['colors'],
            voronoi_K=sv_state['K'],
            is_textured=pkg_lean.get('is_textured'),
            scaling_z=pkg_lean.get('scaling_z'),
        )
        if _final_leaky:
            color = torch.nn.functional.leaky_relu(color, negative_slope=_final_lru_slope)
        return color

    # =================================================================
    # Bench all four modes.
    # =================================================================
    print(f"\n[bench] warmup={args.num_warmup} bench={args.num_benchmark}\n")
    print(f"  {'Mode':<45} {'ms/frame':<20}  {'FPS':>8}")
    print(f"  {'-'*45} {'-'*20}  {'-'*8}")

    A = bench(lambda c: rasterize_prod(c, with_atlas=False), cam,
              args.num_warmup, args.num_benchmark, "PROD  SH-only  (no atlas)")
    B = bench(lambda c: rasterize_prod(c, with_atlas=True), cam,
              args.num_warmup, args.num_benchmark, "PROD  SH+atlas (BC7)")
    C = bench(lambda c: rasterize_lean(c, with_atlas=False), cam,
              args.num_warmup, args.num_benchmark, "LEAN  SH-only  (no atlas)")
    D = bench(lambda c: rasterize_lean(c, with_atlas=True), cam,
              args.num_warmup, args.num_benchmark, "LEAN  SH+atlas (BC7)")

    print(f"\n[quality] PSNR / SSIM / LPIPS across {len(test_cameras)} test cams")
    print(f"  {'-'*45} {'-'*40}")
    qA = eval_quality(lambda c: rasterize_prod(c, with_atlas=False), test_cameras, "PROD  SH-only")
    qB = eval_quality(lambda c: rasterize_prod(c, with_atlas=True),  test_cameras, "PROD  SH+atlas")
    qC = eval_quality(lambda c: rasterize_lean(c, with_atlas=False), test_cameras, "LEAN  SH-only")
    qD = eval_quality(lambda c: rasterize_lean(c, with_atlas=True),  test_cameras, "LEAN  SH+atlas")
    A.update(qA); B.update(qB); C.update(qC); D.update(qD)

    def pct(a, b): return (a['ms_mean'] - b['ms_mean']) / b['ms_mean'] * 100

    print(f"\n[bench] Lean vs Prod:")
    print(f"  SH-only :  {C['ms_mean']:.4f} vs {A['ms_mean']:.4f} ms  → Δ = {pct(C, A):+.1f}%  ({A['fps']:.0f} → {C['fps']:.0f} FPS)")
    print(f"  SH+atlas:  {D['ms_mean']:.4f} vs {B['ms_mean']:.4f} ms  → Δ = {pct(D, B):+.1f}%  ({B['fps']:.0f} → {D['fps']:.0f} FPS)")

    result = {
        "model_path": str(args.model_path),
        "bake_dir": str(args.bake_dir),
        "num_gaussians": int(pkg_prod['means3D'].shape[0]),
        "resolution": f"{cam.image_width}x{cam.image_height}",
        "compact_mult_bake": _compact_mult_train,
        "residual_mode": _residual_mode,
        "modes": {"prod_sh_only": A, "prod_sh_atlas": B,
                  "lean_sh_only": C, "lean_sh_atlas": D},
        "deltas": {"sh_only_pct": pct(C, A), "sh_atlas_pct": pct(D, B)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\n[bench] Saved → {args.out}")


if __name__ == "__main__":
    main()
