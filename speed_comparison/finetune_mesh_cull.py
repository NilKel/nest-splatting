"""
finetune_mesh_cull.py — fine-tune a trained 3D_SH_res (surfel + hash+MLP) model
under a proxy occluder mesh (per-Gauss cull).

Loads a training checkpoint, precomputes per-view mesh depth (Open3D raycast),
and per iteration culls Gaussians whose centre projects behind the mesh at
that pixel. The cull acts through override_opacity so gradients naturally
zero on culled rows for that view; unoccluded rows get the usual L1+SSIM
photometric gradient.

Fine-tunes: _xyz, _scaling, _rotation, _opacity, all SV heads,
hashgrid + MLP. Skips densification / cloning / pruning entirely — surfel
count is fixed to the loaded PLY.
"""
from __future__ import annotations
import os, sys, pickle, argparse, time, math
from pathlib import Path

# Repo root on sys.path so `from scene import ...` works no matter the cwd.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render, set_default_activation_bias
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.loss_utils import l1_loss, ssim


# ---------------------------------------------------------------------------
# Mesh depth precompute (Open3D)
# ---------------------------------------------------------------------------
class MeshDepthBaker:
    """Wraps an Open3D RaycastingScene, produces per-cam cam-Z depth maps.

    `inflate_margin_normal` (default 0.0): if non-zero, geometrically inflate
    the mesh ONCE at load by pushing each vertex along its area-weighted
    OUTWARD vertex normal by this many metres. Silhouette grows so boundary
    surfels behind the mesh get finite mesh_z (fixes the "boundary surfels
    wrongly escape cull" issue of the per-pixel `margin` approach). Stacks
    with per-call `margin` if both are set.
    """

    def __init__(self, mesh_path: str, inflate_margin_normal: float = 0.0):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if inflate_margin_normal != 0.0:
            mesh.compute_vertex_normals(normalized=True)
            V = np.asarray(mesh.vertices)
            N = np.asarray(mesh.vertex_normals)
            mesh.vertices = o3d.utility.Vector3dVector(
                V + float(inflate_margin_normal) * N)
            print(f"[MeshDepthBaker] inflated {V.shape[0]} vertices along "
                  f"vertex normals by {inflate_margin_normal:+.4f} m")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def cam_depth(self, cam, margin: float) -> torch.Tensor:
        """Raycast mesh from `cam` → CUDA fp32 [H, W] cam-Z depth. Non-hit
        pixels are +inf. `margin` added to finite hits (breathing room)."""
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
            np.concatenate([origins.reshape(-1, 3),
                            dirs_w.reshape(-1, 3)], axis=1),
            dtype=o3d.core.Dtype.Float32)
        t_hit = self.scene.cast_rays(rays)['t_hit'].numpy().reshape(H, W)
        # Convert Euclidean t_hit to cam-Z: t_hit is along the unit-normalized
        # world ray; per-pixel dir_cam had z=1, so scaling by |dir_cam| flips
        # back to cam-Z (see test_proxy_cull_perpixel.py for the derivation).
        unnorm = np.sqrt(xs * xs + ys * ys + 1.0).astype(np.float32)
        depth = (t_hit / unnorm).astype(np.float32)
        hit = np.isfinite(depth)
        if margin != 0.0:
            depth[hit] += float(margin)
        return torch.from_numpy(depth).contiguous().cuda()

    def cam_intrinsics(self, cam):
        H, W = cam.image_height, cam.image_width
        fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
        fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        return H, W, fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Per-Gauss occlusion mask
# ---------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def _null_ctx():
    yield


class CompactedGaussians:
    """Context manager that rebinds per-Gauss tensors to their bool-masked
    ``[keep]`` views for the duration of the ``with`` block, then restores.

    Effect: preprocess + sort + tile-assign + render all run on N_survivors
    rows instead of N_total (the CUDA preprocess already opacity-culls at
    <1/255, so this mainly cuts tile-level GEMM work in the collab-GEMM
    render + backward paths). Autograd handles scatter-back through the
    bool index — dL/d(_xyz_c) scatters into dL/d(_xyz.grad) at survivor
    rows; the optimizer sees the same Parameter objects it did at setup.

    INCOMPATIBLE with the FastGS densify path — that path relies on
    ``xyz_gradient_accum`` / ``max_radii2D`` being sized to N, and rewriting
    those to a compacted view breaks their scatter semantics. Guard the
    ``with`` block on your fastgs flag.
    """

    # Per-Gauss tensors we know about. Absent (empty) tensors are skipped.
    _ATTRS = (
        "_xyz",
        "_features_dc", "_features_rest",
        "_opacity",
        "_scaling", "_rotation",
        "_shape", "_flex_beta",
        "_sv_sites", "_sv_colors", "_sv_dc", "_sv_tau",
        "_appearance_level",
        "_film_params",
        "_gaussian_features",
        "_scaling_z", "_is_textured",
        "_sv_mask",             # bool buffer, no grad
    )

    def __init__(self, gaussians, keep_mask: torch.Tensor):
        self.g = gaussians
        # Ensure device match with tensors.
        self.k = keep_mask.to(gaussians._xyz.device)
        self._orig = {}

    def __enter__(self):
        # Snapshot originals + rebind to indexed views. Skip anything that's
        # empty or size-mismatched (avoids blowing up on optional tensors).
        n = int(self.k.shape[0])
        for name in self._ATTRS:
            t = getattr(self.g, name, None)
            if t is None:
                continue
            if not torch.is_tensor(t):
                continue
            if t.numel() == 0 or t.shape[0] != n:
                continue
            self._orig[name] = t
            setattr(self.g, name, t[self.k])
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore originals — Parameter identity preserved, so optimizer state
        # + Parameter references in ingp / gaussians.optimizer are untouched.
        for name, t in self._orig.items():
            setattr(self.g, name, t)
        self._orig.clear()
        return False


@torch.no_grad()
def gauss_occ_mask(centers_w: torch.Tensor,
                   cam,
                   mesh_depth: torch.Tensor) -> torch.Tensor:
    """centers_w [N, 3] world-frame Gauss centres → bool mask [N]
    where True = keep (in front of mesh or mesh missed at that pixel).

    - Behind camera → keep (the rasterizer will cull them anyway).
    - Off-frame → keep (mesh silhouette doesn't govern pixels we can't see).
    - Mesh missed (+inf depth at that pixel) → keep.
    - Gauss depth > mesh depth → cull (behind the occluder).
    """
    H, W = cam.image_height, cam.image_width
    fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    cx, cy = W / 2.0, H / 2.0

    W2C = cam.world_view_transform.to(centers_w.device).T   # [4, 4]
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    cam_xyz = centers_w @ R.T + t                            # [N, 3]
    z = cam_xyz[:, 2]
    keep = torch.ones(centers_w.shape[0], dtype=torch.bool,
                      device=centers_w.device)

    infront = z > 1e-4                                       # camera space z>0
    px = (cam_xyz[:, 0] / z.clamp(min=1e-4)) * fx + cx
    py = (cam_xyz[:, 1] / z.clamp(min=1e-4)) * fy + cy
    inbounds = infront & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    if inbounds.any():
        px_i = px[inbounds].long().clamp(0, W - 1)
        py_i = py[inbounds].long().clamp(0, H - 1)
        mesh_z = mesh_depth[py_i, px_i]
        gauss_z = z[inbounds]
        # cull only when mesh actually hit (finite) AND gauss is behind
        cull = torch.isfinite(mesh_z) & (gauss_z > mesh_z)
        idx = torch.nonzero(inbounds, as_tuple=False).squeeze(-1)
        keep[idx[cull]] = False
    return keep


# ---------------------------------------------------------------------------
# Model loader (mirrors the pattern that hits recorded test PSNR)
# ---------------------------------------------------------------------------
def load_checkpoint(model_path: str, iteration: int):
    with open(os.path.join(model_path, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)
    train_args.model_path = model_path
    train_args.eval = True

    cfg_path = os.path.join(model_path, "config.yaml")
    cfg = Config(cfg_path)

    # CUDA setters that mirror the training-time state.
    from diff_surfel_3D_sh_res import set_activation_bias, set_residual_mode
    ab = getattr(train_args, 'activation_bias', [0.5, 0.0])
    sh_bias, res_bias = float(ab[0]), float(ab[1])
    set_activation_bias(sh_bias=sh_bias, res_bias=res_bias)
    residual_mode = int(getattr(train_args, '_residual_mode', 0))
    set_residual_mode(residual_mode)
    set_default_activation_bias(sh_bias, res_bias)

    # INGP (hash + MLP)
    ingp = INGP(cfg, args=train_args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    # Gaussians
    tp = argparse.ArgumentParser()
    dataset = ModelParams(tp, sentinel=True).extract(train_args)
    pipe = PipelineParams(tp).extract(train_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                  full_args=train_args)

    # Wire attributes that load_ply doesn't touch but the renderer needs.
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = 'UV'
    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel
    if hasattr(train_args, 'kernel2'):
        gaussians.kernel_type2 = getattr(train_args, 'kernel2', None)
    # CRITICAL: load_ply leaves feature_mode="sh" — the SV path is only taken
    # when feature_mode=="SV". Without this the render silently uses empty SH
    # → ~19 dB gap vs recorded test PSNR (bitten once already).
    gaussians.feature_mode = getattr(train_args, 'feature', 'sh')
    # Training-flag TRUE so eval_voronoi_sv skips baking the mask into sites
    # (we WANT the SV heads to keep learning during fine-tune).
    gaussians._sv_training_flag = True
    if hasattr(gaussians, 'update_sites_mask'):
        gaussians.update_sites_mask()

    beta_kern = float(cfg.surfel.tg_beta) if hasattr(cfg.surfel, 'tg_beta') else 0.0
    return train_args, cfg, ingp, gaussians, scene, pipe, dataset, beta_kern


# ---------------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Directory with args.pkl, config.yaml, ngp_XX.pth, point_cloud/iteration_XX/")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Checkpoint iteration to start from (-1 = latest).")
    p.add_argument("--mesh_ply", required=True, help="Proxy occlusion mesh (PLY).")
    p.add_argument("--mesh_margin", type=float, default=0.03,
                   help="Per-pixel depth margin added to hit pixels. Legacy mode "
                        "— extends breathing room ONLY where the mesh already has "
                        "a hit; pixels just outside the silhouette get no margin "
                        "→ boundary surfels wrongly culled. Prefer --mesh_normal_margin.")
    p.add_argument("--mesh_normal_margin", type=float, default=0.0,
                   help="Geometrically INFLATE the proxy mesh by pushing each vertex "
                        "along its area-weighted normal by this many metres (done once "
                        "at load). The silhouette grows too, so surfels near the "
                        "original boundary land INSIDE the inflated mesh and get a "
                        "finite mesh_z (correct cull behavior). Recommended replacement "
                        "for --mesh_margin. Stacks with --mesh_margin if you want both.")
    p.add_argument("--out_dir", required=True, help="Output model dir.")
    p.add_argument("--finetune_iters", type=int, default=5000,
                   help="Number of fine-tune iterations to run.")
    p.add_argument("--save_every", type=int, default=1000,
                   help="Save PLY + INGP every N iters.")
    p.add_argument("--lambda_dssim", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="Multiplier applied to all learning rates (constant, no scheduler).")
    p.add_argument("--xyz_lr", type=float, default=-1.0,
                   help="Override xyz LR. Default = position_lr_final * spatial_lr_scale (matches "
                        "the natural end-of-schedule value). Set to a positive number to override.")
    p.add_argument("--freeze_hash_mlp", action="store_true",
                   help="Freeze hash+MLP — only fine-tune surfels.")
    # Per-pixel Z-cull (CUDA-level). Rejects a fragment at pixel P if its
    # ray-splat depth exceeds the mesh depth at P — even for surfels whose
    # centers are in front (per-Gauss cull only handles whole-surfel cases).
    p.add_argument("--per_pixel_cull", action="store_true",
                   help="Enable per-fragment mesh Z-cull inside the CUDA rasterizer "
                        "(installs the occluder via set_occluder_depth per view). "
                        "Composable with the per-Gauss opacity mask.")
    p.add_argument("--no_per_gauss_cull", action="store_true",
                   help="Disable the per-Gauss opacity mask. Use with --per_pixel_cull "
                        "if you want CUDA cull only (the pixel-level version subsumes "
                        "the Gauss-level one at slightly higher cost).")
    p.add_argument("--random_background", action="store_true",
                   help="Match train.py's --random_background: post-render, blend a "
                        "random RGB (changes every 100 iters) into both the rendered "
                        "image AND the GT via rend_alpha. Required for volumetric / "
                        "semi-transparent scenes (brain, micro-CT); the residual "
                        "won't form correctly against a fixed black bg. Auto-enabled "
                        "when the loaded checkpoint's args.pkl has random_background=True.")
    # Complement to --random_background. Solid random-bg pressure is uniform, so
    # a semi-transparent surfel can partially blend into it without a huge
    # penalty; the surfels covering the mesh silhouette end up not fully opaque
    # (the "surfels not fully hiding the bg" symptom). Injecting a SPATIALLY
    # VARYING noise pattern in the mesh-silhouette region raises the pressure:
    # to hide blocky noise, the surfel must be actually opaque, not just the
    # right average color. Stacks with --random_background: solid random RGB
    # goes outside the mesh silhouette, blocky noise goes INSIDE it.
    p.add_argument("--random_mesh", action="store_true",
                   help="Composite a per-iter random noise pattern (upsampled from a "
                        "coarse grid) into the mesh-silhouette region on top of the "
                        "rend_alpha blend, in both rendered image and GT. Forces "
                        "surfels covering the mesh to be opaque enough to hide the "
                        "spatial variance. Stacks with --random_background (solid RGB "
                        "outside the silhouette, noise inside). No-op without a mesh.")
    p.add_argument("--random_mesh_grid", type=int, default=32,
                   help="Noise-grid resolution for --random_mesh. Grid = G means G×G "
                        "blocks upsampled (nearest) to full frame — for 1080p at G=32 "
                        "each block is ~33×34 px. Larger G → finer/harder noise, "
                        "smaller G → coarser/easier. Default 32.")
    # Regularizers (parity with train.py). Defaults -1 → auto-read from the
    # loaded cfg / args.pkl. Explicit 0 disables. All active by default at
    # iter 35000+ since normal_iter=7k / dist_iter=3k are already past.
    p.add_argument("--lambda_normal", type=float, default=-1.0,
                   help="Normal-consistency reg (rend_normal vs surf_normal). "
                        "Default: cfg_model.loss.lambda_normal (0.0005 in himalaya_2d).")
    p.add_argument("--lambda_dist", type=float, default=-1.0,
                   help="2DGS depth-distortion reg (rend_dist). "
                        "Default: cfg_model.loss.lambda_dist (1000 in himalaya_2d).")
    p.add_argument("--lambda_mask", type=float, default=-1.0,
                   help="Alpha-mask loss weight (rend_alpha vs gt_alpha_mask). "
                        "Default: cfg_model.loss.lambda_mask (0 = disabled).")
    p.add_argument("--mask_dssim", type=float, default=-1.0,
                   help="SSIM mix inside mask_error (0=pure L1). "
                        "Default: cfg_model.loss.mask_dssim (0 in himalaya_2d).")
    p.add_argument("--w_lambda", type=float, default=-1.0,
                   help="Error-guided shape reg (beta_scaled kernel only): "
                        "λ·exp(-γ·MSE)·mean(β). Default: train_args.w_lambda.")
    p.add_argument("--w_lambda_gamma", type=float, default=-1.0,
                   help="γ for --w_lambda's exp(-γ·MSE) weighting.")

    p.add_argument("--fast_compact", action="store_true",
                   help="Speedup: temporarily rebind per-Gauss tensors to their "
                        "bool-masked [keep] views before render — preprocess/sort/"
                        "tile-assign/collab-GEMM all run on N_survivors rows instead "
                        "of N_total. Autograd scatters grads back through the mask, "
                        "optimizer sees the same Parameter objects. INCOMPATIBLE "
                        "with --fastgs (rebinding _xyz breaks the densify accumulator "
                        "size contract). Composes with --per_pixel_cull.")
    # FastGS densification (paper-style importance/pruning) — off by default.
    p.add_argument("--fastgs", action="store_true",
                   help="Enable FastGS densify+prune during finetune (mesh-cull-aware).")
    p.add_argument("--fastgs_densify_interval", type=int, default=500,
                   help="Densify every N iters when --fastgs.")
    p.add_argument("--fastgs_densify_until_frac", type=float, default=0.8,
                   help="Fraction of finetune_iters after which densify stops.")
    p.add_argument("--fastgs_num_views", type=int, default=10,
                   help="Sampled views for score computation each interval.")
    args = p.parse_args()

    if args.fast_compact and args.fastgs:
        raise SystemExit(
            "--fast_compact is incompatible with --fastgs. Compaction rebinds "
            "_xyz to a K-length view; add_densification_stats + max_radii2D "
            "assume the original N-length arrays. Pick one.")

    # ---- Load ----
    train_args, cfg, ingp, gaussians, scene, pipe, dataset, beta_kern = \
        load_checkpoint(args.model_path, args.iteration)
    start_iter = scene.loaded_iter
    print(f"[FT] loaded iter={start_iter}  N={gaussians.get_xyz.shape[0]:,}  "
          f"feature_mode={gaussians.feature_mode}  kernel={gaussians.kernel_type}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Optimizers (surfel + hash/MLP) ----
    # spatial_lr_scale is only set by create_from_pcd; load_ply leaves it at 0
    # → training_setup would build xyz_lr = position_lr_init * 0 = 0. Copy it
    # from scene.cameras_extent (the identity used at fresh-training time).
    gaussians.spatial_lr_scale = float(scene.cameras_extent)

    # 1) Scale the LR knobs on train_args BEFORE training_setup so param groups
    #    that read them directly (opacity/scaling/rotation/feature, sv_colors,
    #    sv_dc, sites_lr, sv_tau_lr) start at the scaled values.
    if args.lr_scale != 1.0:
        for k in ["position_lr_init", "position_lr_final",
                  "opacity_lr", "scaling_lr", "rotation_lr", "feature_lr",
                  "sites_lr", "sites_lr_final", "sv_tau_lr",
                  "sv_color_lr", "sv_dc_lr"]:
            if hasattr(train_args, k):
                v = getattr(train_args, k)
                if isinstance(v, (int, float)):
                    setattr(train_args, k, v * args.lr_scale)
    gaussians.training_setup(train_args)
    ingp.training_setup(cfg.optim)

    # 2) Force xyz + SV heads to sensible CONSTANT LRs (no scheduler is called
    #    during finetune). Under `--feature SV`, training_setup initializes
    #    sv_sites and sv_tau at lr=0 (warmup), and their cosine scheduler is
    #    entirely inside [warmup, 35000-sv_freeze_last]. Without an explicit
    #    override, sv_sites and sv_tau would sit at lr=0 for the whole finetune.
    _default_xyz_lr = (float(getattr(train_args, 'position_lr_final', 1.6e-6))
                       * float(gaussians.spatial_lr_scale))
    _xyz_lr = args.xyz_lr if args.xyz_lr > 0 else _default_xyz_lr
    _sites_lr = float(getattr(train_args, 'sites_lr', 2e-3))
    _tau_lr = float(getattr(train_args, 'sv_tau_lr', 6e-3))
    for g in gaussians.optimizer.param_groups:
        name = g.get('name')
        if name == 'xyz':
            g['lr'] = _xyz_lr
        elif name == 'sv_sites':
            g['lr'] = _sites_lr
        elif name == 'sv_tau':
            g['lr'] = _tau_lr
        # Freeze SH DC/rest — SV mode replaces them, renderer doesn't read them.
        elif name in ('f_dc', 'f_rest', 'ap_level'):
            g['lr'] = 0.0

    # 3) Report the effective LRs so the user can sanity-check.
    print(f"[FT] gaussians.optimizer: {len(gaussians.optimizer.param_groups)} groups "
          f"(lr_scale={args.lr_scale})")
    for g in gaussians.optimizer.param_groups:
        if float(g['lr']) > 0:
            print(f"       {g.get('name'):<20s} lr={float(g['lr']):.3e}")
    print(f"[FT] ingp.optimizer:      {len(ingp.optimizer.param_groups)} groups")
    for g in ingp.optimizer.param_groups:
        print(f"       {g.get('name'):<20s} lr={float(g['lr']):.3e}")
    if args.freeze_hash_mlp:
        for g in ingp.optimizer.param_groups:
            g['lr'] = 0.0
        print("[FT] hash+MLP frozen (lr=0 for all INGP groups).")

    # ---- Precompute per-view mesh depth ----
    print("[FT] precomputing per-view mesh depths ...")
    baker = MeshDepthBaker(args.mesh_ply,
                           inflate_margin_normal=float(args.mesh_normal_margin))
    train_cams = scene.getTrainCameras().copy()
    mesh_depths = []
    t0 = time.time()
    for cam in tqdm(train_cams, desc="raycast"):
        mesh_depths.append(baker.cam_depth(cam, args.mesh_margin))
    print(f"[FT] mesh depths done ({len(train_cams)} views, "
          f"{time.time()-t0:.1f}s)")

    # ---- Training loop ----
    bg = torch.zeros(3, dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(args.seed)
    order = list(range(len(train_cams)))

    # Random-background auto-detect: if the loaded checkpoint's args.pkl had
    # random_background=True, mirror it unless the caller explicitly forced it
    # via the CLI flag.
    use_random_bg = args.random_background or bool(getattr(train_args, 'random_background', False))

    # ------- Regularizer resolution (parity with train.py) -------
    # -1 = use the checkpoint's original value (cfg.loss.* or args.pkl).
    def _resolve(cli_val, cfg_getter, arg_fallback=None, default=0.0):
        if cli_val >= 0:
            return float(cli_val)
        try:
            v = cfg_getter()
            if v is not None:
                return float(v)
        except Exception:
            pass
        if arg_fallback is not None:
            return float(getattr(train_args, arg_fallback, default))
        return float(default)

    lam_normal = _resolve(args.lambda_normal,
                          lambda: cfg.loss.lambda_normal, default=0.0)
    lam_dist   = _resolve(args.lambda_dist,
                          lambda: cfg.loss.lambda_dist, default=0.0)
    lam_mask   = _resolve(args.lambda_mask,
                          lambda: getattr(cfg.loss, 'lambda_mask', 0.0), default=0.0)
    mask_dssim = _resolve(args.mask_dssim,
                          lambda: getattr(cfg.loss, 'mask_dssim', 0.0), default=0.0)
    w_lam      = _resolve(args.w_lambda, lambda: None,
                          arg_fallback='w_lambda', default=0.0)
    w_lam_gam  = _resolve(args.w_lambda_gamma, lambda: None,
                          arg_fallback='w_lambda_gamma', default=50.0)
    kernel_str = getattr(train_args, 'kernel', 'gaussian')
    w_lam_active = (w_lam > 0.0 and kernel_str in ('beta', 'beta_scaled', 'general')
                    and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0)

    print(f"[FT] regularizers: "
          f"lambda_normal={lam_normal}  lambda_dist={lam_dist}  "
          f"lambda_mask={lam_mask}  mask_dssim={mask_dssim}  "
          f"w_lambda={w_lam} (γ={w_lam_gam}, active={w_lam_active}, "
          f"kernel={kernel_str})")
    if args.random_mesh:
        print(f"[FT] random-mesh noise compositing ENABLED "
              f"(grid={args.random_mesh_grid} — {args.random_mesh_grid}x{args.random_mesh_grid} "
              f"blocky noise pattern refreshed every iter, painted into the mesh "
              f"silhouette region on top of the rend_alpha blend)")
    if use_random_bg:
        print(f"[FT] random-background compositing ENABLED "
              f"(train_args.random_background={getattr(train_args, 'random_background', False)}, "
              f"cli={args.random_background})")

    # FastGS densify schedule (both bounds in LOCAL iters; global iter passed only
    # to render for hash active-level ramp).
    if args.fastgs:
        fastgs_until_local = int(args.fastgs_densify_until_frac * args.finetune_iters)
        # Pull thresholds from the training args pickle so we mirror what the
        # checkpoint was trained under.
        fastgs_grad_thresh = float(getattr(train_args, 'fastgs_grad_thresh', 0.00015))
        fastgs_grad_abs = float(getattr(train_args, 'fastgs_grad_abs_thresh', 0.0006))
        fastgs_dense = float(getattr(train_args, 'fastgs_dense', 0.01))
        fastgs_imp_thresh = float(getattr(train_args, 'fastgs_importance_thresh', 30))
        fastgs_loss_thresh = float(getattr(train_args, 'fastgs_loss_thresh', 0.1))
        fastgs_lam_dssim = float(getattr(train_args, 'fastgs_lambda_dssim', args.lambda_dssim))
        fastgs_prune_budget = float(getattr(train_args, 'fastgs_prune_budget_frac', 0.5))
        fastgs_opacity_clamp = float(getattr(train_args, 'fastgs_opacity_clamp', 0.8))
        opacity_cull = float(getattr(train_args, 'opacity_cull', 0.005))
        print(f"[FT] FastGS ENABLED: densify every {args.fastgs_densify_interval} iters "
              f"until local iter {fastgs_until_local}. grad>={fastgs_grad_thresh}, "
              f"grad_abs>={fastgs_grad_abs}, dense={fastgs_dense}, "
              f"imp>={fastgs_imp_thresh}, K={args.fastgs_num_views} views.")

    # id(cam) → index-in-train_cams (mesh_depths). Cameras are the same
    # objects Scene loaded; sampling_cameras only reorders them (with .pop),
    # so identity lookup is stable across densify events.
    cam2idx = {id(c): i for i, c in enumerate(train_cams)}

    def make_render_fn(current_iter: int):
        """Render fn used by compute_gaussian_score_fastgs — mesh-cull-aware so
        densify/prune decisions are based on the actual (post-cull) rendered
        image, matching what the training loop sees. Installs the CUDA
        occluder per view when --per_pixel_cull is set."""
        def _fn(v, metric_map=None):
            _idx = cam2idx[id(v)]
            _mz = mesh_depths[_idx]
            with torch.no_grad():
                _keep = gauss_occ_mask(gaussians.get_xyz, v, _mz)
                if args.no_per_gauss_cull:
                    _override = None
                else:
                    _override = gaussians.get_opacity * _keep.to(
                        gaussians.get_xyz.dtype).view(-1, 1)
            if args.per_pixel_cull:
                from diff_surfel_3D_sh_res import set_occluder_depth
                set_occluder_depth(_mz.contiguous() if not _mz.is_contiguous() else _mz)
            return render(v, gaussians, pipe, bg, beta=beta_kern,
                          iteration=current_iter, cfg=cfg, ingp=ingp,
                          record_transmittance=False, is_training=False,
                          lowpass=True, override_opacity=_override,
                          metric_map=metric_map)
        return _fn

    ema_loss = None
    prog = tqdm(range(1, args.finetune_iters + 1), desc="finetune")
    # Iteration for LR schedules / hash active-level ramp: keep the checkpoint's
    # native iter count so nothing regresses schedules that fired mid-training.
    for it_local in prog:
        it_global = start_iter + it_local
        if it_local % len(order) == 1:
            rng.shuffle(order)
        cam_idx = order[(it_local - 1) % len(order)]
        cam = train_cams[cam_idx]
        gt = cam.original_image[:3].cuda()
        mesh_z = mesh_depths[cam_idx]

        # Per-Gauss cull mask (same math whether we use it for opacity override
        # or for --fast_compact).
        centers = gaussians.get_xyz
        keep = gauss_occ_mask(centers.detach(), cam, mesh_z)   # [N] bool

        # Path A (default): keep N surfels + zero-opacity culled ones. CUDA
        #                   preprocess early-culls at opacity<1/255, so this is
        #                   already reasonably efficient; downside is the tile-
        #                   level collab-GEMM still enumerates full N per tile.
        # Path B (--fast_compact): rebind per-Gauss tensors to their [keep]
        #                   views; preprocess/sort/render all run on K rows
        #                   only. Bigger wall-clock cut, especially at high
        #                   cull rates. Not usable with --fastgs.
        if args.fast_compact:
            override_opacity = None                            # cull via compaction
        elif args.no_per_gauss_cull:
            override_opacity = None
        else:
            mask = keep.to(centers.dtype).view(-1, 1)          # [N,1] {0,1}
            override_opacity = gaussians.get_opacity * mask

        # Per-pixel Z-cull: install CUDA occluder for this view. Rasterizer
        # drops fragments (fwd + bwd) whose depth exceeds mesh_z[pix]. Kept
        # alive here in Python across the render call (setter is a raw pointer).
        if args.per_pixel_cull:
            from diff_surfel_3D_sh_res import set_occluder_depth
            _mz_for_cuda = mesh_z if mesh_z.is_contiguous() else mesh_z.contiguous()
            set_occluder_depth(_mz_for_cuda)

        # Wrap the render+backward in the compact context if requested. The
        # context restores original tensor bindings in __exit__ even on error,
        # so an exception in render can't leave `gaussians` in a bad state.
        _compact_cm = CompactedGaussians(gaussians, keep) if args.fast_compact \
            else _null_ctx()
        with _compact_cm:
            pkg = render(cam, gaussians, pipe, bg, beta=beta_kern,
                         iteration=it_global, cfg=cfg, ingp=ingp,
                         lowpass=True, override_opacity=override_opacity)
            img = pkg['render']
            visibility_filter = pkg['visibility_filter']
            radii = pkg['radii']
            viewspace_points = pkg['viewspace_points']

        # Random-background compositing (matches train.py). The neural render
        # itself uses fixed black bg; POST-render, we blend a shared random
        # RGB into BOTH the rendered image AND the GT via rend_alpha. Bg
        # changes every 100 iters (`torch.manual_seed(it_global // 100)` for
        # reproducibility across a window, matching train.py). Required for
        # volumetric / --random_background-trained scenes; without it those
        # scenes collapse to opacity/color degeneracy.
        #
        # `--random_mesh` layers on top: within the mesh silhouette we
        # overwrite the solid random RGB with a per-iter blocky noise pattern
        # (grid G×G, nearest-upsampled to H×W). Solid bg outside stays intact.
        # Higher spatial-variance pressure than random_bg alone; surfels can't
        # blend into an average color when there IS no single average color.
        if use_random_bg or (args.random_mesh and len(mesh_depths) > 0):
            H, W = img.shape[1], img.shape[2]
            torch.manual_seed(it_global // 100)
            random_bg_color = torch.rand(3, 1, 1, device="cuda")
            random_bg = random_bg_color.expand(3, H, W)
            if args.random_mesh and len(mesh_depths) > 0:
                # Fresh noise EACH iter (unlike solid bg which reuses across
                # 100-iter windows) — spatial variance is what does the work
                # here, so we don't need slow decorrelation to prevent surfel
                # over-fitting to a specific pattern. Same seed source so runs
                # are still reproducible.
                torch.manual_seed(it_global * 977 + 31)  # decorrelated stream
                G = max(1, int(args.random_mesh_grid))
                noise_lo = torch.rand(1, 3, G, G, device="cuda")
                # Nearest upsample: blocky pattern, sharp edges — hardest for
                # a semi-transparent surfel to average away.
                noise_hi = torch.nn.functional.interpolate(
                    noise_lo, size=(H, W), mode="nearest").squeeze(0)
                # Silhouette mask from the CACHED mesh_z for this cam.
                # torch.isfinite → 1 where mesh covers the pixel, 0 elsewhere.
                m_hit = torch.isfinite(mesh_z).to(torch.float32).unsqueeze(0)  # [1, H, W]
                bg_composite = m_hit * noise_hi + (1.0 - m_hit) * random_bg
            else:
                bg_composite = random_bg
            rend_alpha = pkg["rend_alpha"]
            img = img + (1.0 - rend_alpha) * bg_composite
            # GT alpha: prefer the loaded mask, else infer from any-nonzero.
            gt_alpha = (cam.gt_alpha_mask.cuda().float()
                        if getattr(cam, "gt_alpha_mask", None) is not None
                        else (gt != 0).any(dim=0, keepdim=True).float())
            gt = gt + (1.0 - gt_alpha) * bg_composite

        # Photometric loss (train.py's default L1 + SSIM mix).
        Ll1 = l1_loss(img, gt)
        loss = (1.0 - args.lambda_dssim) * Ll1 \
               + args.lambda_dssim * (1.0 - ssim(img, gt))

        # ------- Geometry / mask / shape regularizers (parity with train.py) -------
        # Iters 35k+ so normal_iter (7k) and dist_iter (3k) gates are past;
        # we don't re-check them here for simplicity.
        if lam_normal > 0.0 or lam_dist > 0.0:
            rend_normal = pkg['rend_normal']
            surf_normal = pkg['surf_normal']
            rend_dist_ = pkg['rend_dist']
            normal_error = (1.0 - (rend_normal * surf_normal).sum(dim=0))[None]
            if lam_normal > 0.0:
                loss = loss + lam_normal * normal_error.mean()
            if lam_dist > 0.0:
                loss = loss + lam_dist * rend_dist_.mean()

        if lam_mask > 0.0:
            rend_alpha = pkg['rend_alpha']
            gt_alpha = (cam.gt_alpha_mask.cuda().float()
                        if getattr(cam, 'gt_alpha_mask', None) is not None
                        else (cam.original_image[:3].cuda() != 0).any(
                            dim=0, keepdim=True).float())
            mask_err = l1_loss(gt_alpha, rend_alpha).mean()
            if mask_dssim > 0.0:
                mask_err = ((1.0 - mask_dssim) * mask_err
                            + mask_dssim * (1.0 - ssim(rend_alpha.unsqueeze(0),
                                                       gt_alpha.unsqueeze(0))))
            loss = loss + lam_mask * mask_err

        # Error-guided shape reg on beta_scaled kernel (mirrors train.py w_lambda).
        if w_lam_active:
            with torch.no_grad():
                mse_pp = ((img - gt) ** 2).mean(dim=0, keepdim=True)
                w_r = torch.exp(-w_lam_gam * mse_pp).mean()
            loss = loss + w_lam * w_r * gaussians.get_shape.mean()

        loss.backward()

        # ------- Sync CUDA-side MLP weight grads into ingp.mlp_fused -------
        # For 3D_SH_res, the fused MLP weights are stored on the CUDA side (in
        # d_mlp_W1/W2/W3) and grads accumulate into a separate CUDA buffer.
        # Without this pull, ingp.mlp_fused.parameters().grad stays None →
        # the INGP optimizer step is a no-op for the MLP.
        if not args.freeze_hash_mlp:
            from diff_surfel_3D_sh_res import get_mlp_grads
            _mlp_grads = get_mlp_grads()
            if _mlp_grads is not None and getattr(ingp, 'mlp_fused', None) is not None:
                _gW1, _gW2, _gW3 = _mlp_grads
                _mlp = ingp.mlp_fused
                if _mlp[0].weight.grad is None:
                    _mlp[0].weight.grad = _gW1.clone()
                else:
                    _mlp[0].weight.grad += _gW1
                if _mlp[2].weight.grad is None:
                    _mlp[2].weight.grad = _gW2.clone()
                else:
                    _mlp[2].weight.grad += _gW2
                if _mlp[4].weight.grad is None:
                    _mlp[4].weight.grad = _gW3.clone()
                else:
                    _mlp[4].weight.grad += _gW3

        # ------- One-shot grad audit @ iter 100 -------
        # Confirms every param we intend to train received a non-zero gradient
        # this backward. A near-zero norm on any row here means that param
        # group is silently frozen (has bitten us before with sv_sites/sv_tau
        # under the SV cosine LR scheduler leaving them at lr=0).
        if it_local == 100:
            def _norm(t):
                if t is None or getattr(t, 'grad', None) is None or t.numel() == 0:
                    return "grad=<None>"
                return f"|grad|={t.grad.norm().item():.3e}"
            def _mod_norm(m):
                if m is None:
                    return "<None>"
                total = 0.0
                for p in m.parameters():
                    if p.grad is not None:
                        total += p.grad.norm().item() ** 2
                return f"|grad|={total ** 0.5:.3e}"
            print(f"\n[GRAD @ it_local=100]")
            print(f"  position   (_xyz)       {_norm(gaussians._xyz)}")
            print(f"  scale      (_scaling)   {_norm(gaussians._scaling)}")
            print(f"  rotation   (_rotation)  {_norm(gaussians._rotation)}")
            print(f"  opacity    (_opacity)   {_norm(gaussians._opacity)}")
            print(f"  SV sites   (_sv_sites)  {_norm(gaussians._sv_sites)}")
            print(f"  SV colors  (_sv_colors) {_norm(gaussians._sv_colors)}")
            print(f"  SV tau     (_sv_tau)    {_norm(getattr(gaussians, '_sv_tau', None))}")
            print(f"  hash grid  (hash_encoding) "
                  f"{_mod_norm(getattr(ingp, 'hash_encoding', None))}")
            print(f"  MLP fused  (mlp_fused)  "
                  f"{_mod_norm(getattr(ingp, 'mlp_fused', None))}")

        # ------- FastGS gradient stats (visibility + viewspace grad) -------
        if args.fastgs and it_local < fastgs_until_local:
            with torch.no_grad():
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter].float())
                gaussians.add_densification_stats(
                    viewspace_points, visibility_filter, pixels=None)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        if not args.freeze_hash_mlp:
            ingp.optimizer.step()
            ingp.optimizer.zero_grad(set_to_none=True)

        # ------- FastGS periodic densify+prune -------
        if (args.fastgs
                and it_local > 0
                and it_local < fastgs_until_local
                and it_local % args.fastgs_densify_interval == 0):
            from utils.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
            # sampling_cameras .pop()s from the list it receives — pass a shallow
            # copy so train_cams (and the mesh_depths alignment) stays intact.
            _sampled = sampling_cameras(list(train_cams),
                                        num_cams=args.fastgs_num_views)
            _rfn = make_render_fn(it_global)
            imp, prn = compute_gaussian_score_fastgs(
                _sampled, gaussians, _rfn,
                loss_thresh=fastgs_loss_thresh,
                lambda_dssim=fastgs_lam_dssim,
                densify=True)
            stats = gaussians.densify_and_prune_fastgs(
                min_opacity=opacity_cull,
                extent=scene.cameras_extent,
                max_screen_size=None,
                importance_score=imp,
                pruning_score=prn,
                grad_thresh=fastgs_grad_thresh,
                grad_abs_thresh=fastgs_grad_abs,
                dense=fastgs_dense,
                importance_thresh=fastgs_imp_thresh,
                prune_budget_frac=fastgs_prune_budget,
                extra_split_mask=None,
                opacity_clamp=fastgs_opacity_clamp)
            if gaussians.feature_mode == "SV":
                gaussians.update_sites_mask()
            tqdm.write(f"[FASTGS] it_local={it_local}: cloned={stats['cloned']} "
                       f"split={stats['split_parents']} N={gaussians.get_xyz.shape[0]:,}")

        with torch.no_grad():
            l = float(loss.item())
            ema_loss = l if ema_loss is None else 0.4 * l + 0.6 * ema_loss
            # Match train.py's cadence: refresh every 10 iters.
            if it_local % 10 == 0:
                cull_pct = 100.0 * (1 - keep.float().mean().item())
                prog.set_postfix_str(
                    f"loss={ema_loss:.4f}  cull={cull_pct:4.1f}%  "
                    f"N={gaussians.get_xyz.shape[0]//1000}k")

        # Periodic preview: render test view #0 under the SAME cull settings and
        # save as PNG so training progress is visible on disk. No PLY or ngp.pth
        # here — those come at the very end (below).
        if it_local % args.save_every == 0 and it_local > 0:
            save_iter = start_iter + it_local
            test_cams = scene.getTestCameras()
            if len(test_cams) > 0:
                cam1 = test_cams[0]
                with torch.no_grad():
                    _mz = baker.cam_depth(cam1, args.mesh_margin)
                    _keep = gauss_occ_mask(gaussians.get_xyz, cam1, _mz)
                    if args.no_per_gauss_cull:
                        _override = None
                    else:
                        _override = gaussians.get_opacity * _keep.to(
                            gaussians.get_xyz.dtype).view(-1, 1)
                    if args.per_pixel_cull:
                        from diff_surfel_3D_sh_res import set_occluder_depth
                        set_occluder_depth(_mz.contiguous() if not _mz.is_contiguous() else _mz)
                    _pkg = render(cam1, gaussians, pipe, bg, beta=beta_kern,
                                  iteration=it_global, cfg=cfg, ingp=ingp,
                                  is_training=False,
                                  lowpass=True, override_opacity=_override)
                    _img = _pkg['render'].clamp(0, 1)
                    _gt = cam1.original_image[:3].cuda().clamp(0, 1)
                    _mse = ((_img - _gt) ** 2).mean().item()
                    _psnr = -10 * np.log10(_mse) if _mse > 0 else float('inf')
                _u8 = (_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                       ).clip(0, 255).astype(np.uint8)
                _png = os.path.join(args.out_dir,
                                    f"test_view1_iter_{save_iter:06d}.png")
                import imageio.v2 as imageio
                imageio.imwrite(_png, _u8)
                tqdm.write(f"[FT] iter={save_iter}  PSNR(view1)={_psnr:.2f} dB  → {_png}")

                # Second preview: what the LOSS actually saw this iter — the
                # same rendered image but with random_bg / random_mesh noise
                # composited into rend_alpha, matching the training-loop
                # compositing block. Useful for "is the noise actually being
                # applied" gut-check: if this preview looks clean too, either
                # the flags weren't picked up or the mesh silhouette is empty
                # for this cam. Refetch the same seeds used in-loop so pattern
                # matches the actual per-iter noise (approximately — the loop
                # advances the seed once per iter so this is one iter behind).
                if use_random_bg or (args.random_mesh and len(mesh_depths) > 0):
                    with torch.no_grad():
                        _rend_alpha = _pkg['rend_alpha']
                        H, W = _img.shape[1], _img.shape[2]
                        torch.manual_seed(it_global // 100)
                        _bg_col = torch.rand(3, 1, 1, device="cuda")
                        _bg = _bg_col.expand(3, H, W)
                        if args.random_mesh and len(mesh_depths) > 0:
                            torch.manual_seed(it_global * 977 + 31)
                            G = max(1, int(args.random_mesh_grid))
                            _noise_lo = torch.rand(1, 3, G, G, device="cuda")
                            _noise_hi = torch.nn.functional.interpolate(
                                _noise_lo, size=(H, W), mode="nearest").squeeze(0)
                            _m_hit = torch.isfinite(_mz).to(torch.float32).unsqueeze(0)
                            _bg_c = _m_hit * _noise_hi + (1.0 - _m_hit) * _bg
                        else:
                            _bg_c = _bg
                        _img_noised = _img + (1.0 - _rend_alpha) * _bg_c
                        _img_noised = _img_noised.clamp(0, 1)
                    _u8_n = (_img_noised.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                             ).clip(0, 255).astype(np.uint8)
                    _png_n = os.path.join(args.out_dir,
                                          f"test_view1_iter_{save_iter:06d}_noised.png")
                    imageio.imwrite(_png_n, _u8_n)
                    tqdm.write(f"[FT] iter={save_iter}  noised preview → {_png_n}")

    # Clear the CUDA occluder so downstream renders (save-time preview,
    # subsequent scripts sharing this process) revert to the default no-op path.
    if args.per_pixel_cull:
        from diff_surfel_3D_sh_res import clear_occluder_depth
        clear_occluder_depth()

    # Final checkpoint: write PLY + ngp.pth once at the end.
    final_iter = start_iter + args.finetune_iters
    print(f"\n[FT] saving final checkpoint iter={final_iter} → {args.out_dir}")
    orig_mp = scene.model_path
    scene.model_path = args.out_dir
    try:
        scene.save(final_iter)
        if not args.freeze_hash_mlp:
            ingp.save_model(args.out_dir, final_iter)
        else:
            src = os.path.join(args.model_path, f"ngp_{start_iter}.pth")
            dst = os.path.join(args.out_dir, f"ngp_{final_iter}.pth")
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
    finally:
        scene.model_path = orig_mp
    print(f"[FT] done. output at {args.out_dir}")


if __name__ == "__main__":
    main()
