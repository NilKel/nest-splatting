#!/usr/bin/env python3
"""
bake_mesh_texture.py
====================

Inverse-render a per-vertex RGBA texture on a TSDF-fused mesh of a
`--method mixed_3d` checkpoint's textured surfels, so that the simpler
"SV-only surfels + mesh RGBA" representation reproduces the original
textured-half render.

Pipeline
--------
1. Load checkpoint, slice to textured-only (drop EWA untextured half).
2. TSDF-fuse the textured-half renders → mesh_textured.ply.
3. Build a "mesh-Gaussian sheet": one flat-disk Gaussian per mesh vertex,
   xyz/scale/rotation FROZEN to the mesh geometry, opacity + RGB LEARNABLE.
4. Pre-cache per-view targets (full textured-only render WITH residual) and
   per-view "SV-only" backgrounds (textured surfels with MLP zeroed).
5. Optimize mesh per-vertex RGB (signed) + opacity by L1+SSIM photometric
   loss between `relu(α_mesh · rgb_mesh + (1-α_mesh) · surf_sv_rgb)` and the
   cached target.
6. Save mesh + per-vertex signed FP32 RGBA as PLY.

Key design choices
------------------
* **Mesh RGB is signed** (stored FP32) so it can both ADD and SUBTRACT from
  the SV background — matches the unbounded property of the original
  signed residual.
* **Two-pass composite** (`α·mesh + (1-α)·surfels`) rather than a single
  rasterizer pass because the existing `render()` forces
  `colors_precomp=None` on the mixed_3d path. Mesh and surfels are
  collocated (same surface), so alpha-over is a good approximation.
* **Per-pixel ReLU** applied AFTER compositing — same as mixed_3d's
  inference path. Lets the signed mesh contribution be clipped at zero.
"""

import os
import sys
import math
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import open3d as o3d
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.system_utils import searchForMaxIteration
from utils.loss_utils import l1_loss, ssim
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def install_setter_mirror(method: str):
    """Mirror diff_surfel_3D_sh_res setters onto diff_surfel_mixed[_3d] so
    set_mlp_weights / set_residual_mode / etc. propagate to the kernel that
    actually runs. MUST be called before the first render() invocation."""
    if method not in ("mixed", "mixed_3d"):
        return
    import diff_surfel_3D_sh_res as _ds_orig
    if method == "mixed_3d":
        import diff_surfel_mixed_3d as _ds_mirror
    else:
        import diff_surfel_mixed as _ds_mirror
    _SETTERS = (
        'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
        'set_overdraw_lambda', 'set_weight_reg_lambda',
        'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
        'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
        'set_depth_sort',
    )
    for name in _SETTERS:
        if not hasattr(_ds_orig, name) or not hasattr(_ds_mirror, name):
            continue
        of, mf = getattr(_ds_orig, name), getattr(_ds_mirror, name)

        def _make_wrap(of_, mf_):
            def _wrapped(*a, **k):
                of_(*a, **k)
                mf_(*a, **k)
            return _wrapped
        setattr(_ds_orig, name, _make_wrap(of, mf))


def fold_train_args(args, exp_path: str):
    """Fold args.pkl saved at training time onto our argparse Namespace so
    method/kernel/kernel2/feature/yaml/etc. survive get_combined_args (which
    only reads cfg_args, not args.pkl)."""
    pkl = os.path.join(exp_path, "args.pkl")
    if not os.path.exists(pkl):
        return args
    with open(pkl, "rb") as f:
        train_args = pickle.load(f)
    for k, v in vars(train_args).items():
        if not hasattr(args, k) or getattr(args, k, None) is None:
            setattr(args, k, v)
    for k in ("method", "kernel", "kernel2", "feature", "hybrid_levels",
              "disable_c2f", "aabb", "yaml", "lowpass", "texsplit",
              "activation_bias", "fastgs_mult", "sv_metric"):
        if hasattr(train_args, k):
            setattr(args, k, getattr(train_args, k))
    return args


def slice_to_textured(gaussians: GaussianModel):
    """In-memory slice: keep only textured surfels. Walks every tensor
    attribute whose leading dim == N and slices by `_is_textured`."""
    mask = gaussians._is_textured
    N = gaussians.get_xyz.shape[0]
    saved = {}
    for f in sorted(vars(gaussians).keys()):
        t = getattr(gaussians, f)
        if not torch.is_tensor(t) or t.numel() == 0 or t.shape[0] != N:
            continue
        saved[f] = t
        sliced = t[mask].detach().clone()
        if isinstance(t, nn.Parameter):
            setattr(gaussians, f, nn.Parameter(sliced.requires_grad_(t.requires_grad)))
        else:
            setattr(gaussians, f, sliced)
    return saved


# ---------------------------------------------------------------------------
# Mesh → Gaussian sheet
# ---------------------------------------------------------------------------
def normal_to_quat(normals: torch.Tensor) -> torch.Tensor:
    """Per-vertex quaternion (w, x, y, z) rotating z=(0,0,1) → normal.
    The 2DGS rasterizer expects this convention."""
    N = normals.shape[0]
    dev = normals.device
    z = torch.zeros(N, 3, device=dev); z[:, 2] = 1.0
    dots = (z * normals).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    angles = torch.acos(dots)
    axes = torch.cross(z, normals, dim=-1)
    axes_n = axes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axes = axes / axes_n
    half = angles / 2
    qw = torch.cos(half).unsqueeze(-1)
    qv = axes * torch.sin(half).unsqueeze(-1)
    quats = torch.cat([qw, qv], dim=-1)   # (w, x, y, z)
    # Degenerate: parallel (dot≈1) → identity quat; antiparallel (dot≈-1) →
    # 180° rotation about any orthogonal axis. The acos+cross above already
    # produces the right quat for parallel; for antiparallel we'd want to
    # pick a perpendicular axis explicitly. Bonsai meshes shouldn't have
    # vertex normals pointing -z, so leave the simple case for now.
    parallel = dots > 1 - 1e-5
    quats[parallel] = torch.tensor([1., 0., 0., 0.], device=dev)
    return quats


def build_mesh_gaussian_model(mesh, sh_degree: int, disk_scale: float):
    """Construct a frozen-geometry GaussianModel from an Open3D triangle mesh.
    xyz/scaling/rotation are non-leaf tensors (not optimized); opacity +
    features_dc are nn.Parameter (learnable). Higher-order SH frozen at zero."""
    mesh.compute_vertex_normals()
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    normals = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float32, device="cuda")
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    M = verts.shape[0]

    mg = GaussianModel(sh_degree)
    # Frozen geometry — we hold these as plain tensors AND as Parameter(requires_grad=False)
    # so calls like `mg.get_xyz` (which expects nn.Parameter underneath) still work.
    mg._xyz = nn.Parameter(verts, requires_grad=False)
    log_s = float(np.log(disk_scale))
    mg._scaling = nn.Parameter(torch.full((M, 2), log_s, device="cuda"), requires_grad=False)
    mg._rotation = nn.Parameter(normal_to_quat(normals), requires_grad=False)

    # Learnable: opacity (raw, post-sigmoid via get_opacity) and per-vertex DC.
    # We override colors via `override_color` in the render call, so _features_dc
    # is just a placeholder — keep it learnable so SH backward path works.
    mg._opacity = nn.Parameter(torch.zeros((M, 1), device="cuda"), requires_grad=True)
    nfeat = (sh_degree + 1) ** 2
    mg._features_dc = nn.Parameter(torch.zeros((M, 1, 3), device="cuda"), requires_grad=False)
    mg._features_rest = nn.Parameter(torch.zeros((M, nfeat - 1, 3), device="cuda"), requires_grad=False)

    mg.base_opacity = 0.0
    mg.max_sh_degree = sh_degree
    mg.active_sh_degree = 0
    mg.kernel_type = "gaussian"
    mg.feature_mode = "sh"
    mg._appearance_level = torch.zeros((M, 1), dtype=torch.float32, device="cuda")
    return mg, verts, normals


# ---------------------------------------------------------------------------
# PLY writer (vertex RGBA float32 + face indices)
# ---------------------------------------------------------------------------
def save_mesh_rgba(mesh, mesh_rgb_param: torch.Tensor,
                   mesh_opacity_param: torch.Tensor, out_path: str,
                   write_uint8_companion: bool = True):
    """Save the mesh with per-vertex signed FP32 RGB + FP32 alpha. Optionally
    also write a uint8 companion (clipped to [0,1]) for viewers that don't
    understand FP32 vertex colors (MeshLab, most browsers)."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    rgb = mesh_rgb_param.detach().cpu().numpy().astype(np.float32)              # signed
    alpha = torch.sigmoid(mesh_opacity_param.squeeze(-1)).detach().cpu().numpy().astype(np.float32)
    V = verts.shape[0]

    vertex_data = np.zeros(V, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),
        ('alpha', 'f4'),
    ])
    vertex_data['x'] = verts[:, 0].astype(np.float32)
    vertex_data['y'] = verts[:, 1].astype(np.float32)
    vertex_data['z'] = verts[:, 2].astype(np.float32)
    vertex_data['red']   = rgb[:, 0]
    vertex_data['green'] = rgb[:, 1]
    vertex_data['blue']  = rgb[:, 2]
    vertex_data['alpha'] = alpha

    face_data = np.zeros(tris.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    face_data['vertex_indices'] = tris.astype(np.int32)

    PlyData([
        PlyElement.describe(vertex_data, 'vertex'),
        PlyElement.describe(face_data, 'face'),
    ]).write(out_path)

    if write_uint8_companion:
        rgb_u8 = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb_u8 * 255.0 + 0.5).astype(np.uint8)
        alpha_u8 = (alpha * 255.0 + 0.5).astype(np.uint8)
        vertex_u8 = np.zeros(V, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
        ])
        vertex_u8['x'] = verts[:, 0].astype(np.float32)
        vertex_u8['y'] = verts[:, 1].astype(np.float32)
        vertex_u8['z'] = verts[:, 2].astype(np.float32)
        vertex_u8['red']   = rgb_u8[:, 0]
        vertex_u8['green'] = rgb_u8[:, 1]
        vertex_u8['blue']  = rgb_u8[:, 2]
        vertex_u8['alpha'] = alpha_u8
        PlyData([
            PlyElement.describe(vertex_u8, 'vertex'),
            PlyElement.describe(face_data, 'face'),
        ]).write(out_path.replace('.ply', '_u8.ply'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Bake mesh RGBA from mixed_3d textured layer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_subdir", default="mesh_bake", type=str)

    # Mesh extraction (TSDF + post-process). Mirrors 2DGS's eval_render.py:
    #   --unbounded → extract_mesh_unbounded (MERF contraction, for mip-360)
    #   default     → extract_mesh_bounded (object-centric, DTU/NeRF-Syn)
    parser.add_argument("--mesh_res", default=1024, type=int)
    parser.add_argument("--voxel_size", default=-1.0, type=float)
    parser.add_argument("--depth_trunc", default=-1.0, type=float,
                        help="Hard depth_trunc for bounded TSDF. -1 = auto "
                             "= extractor.radius * --depth_trunc_mult.")
    parser.add_argument("--depth_trunc_mult", default=5.0, type=float,
                        help="Multiplier on extractor.radius for the auto "
                             "depth_trunc. 2DGS default was 2.0; bumped to 5.0 "
                             "to cover indoor mip-360 scenes where far walls "
                             "are well beyond 2× camera-to-center distance.")
    parser.add_argument("--sdf_trunc", default=-1.0, type=float)
    parser.add_argument("--num_cluster", default=200, type=int)
    parser.add_argument("--unbounded", action="store_true",
                        help="Use extract_mesh_unbounded (MERF contraction, "
                             "for mip-360-style unbounded scenes). Default "
                             "(extract_mesh_bounded) clips at depth_trunc and "
                             "drops far geometry (walls/background).")
    parser.add_argument("--mesh_depth_ratio", default=1.0, type=float,
                        help="pipe.depth_ratio used for TSDF source depth. "
                             "0.0 = alpha-weighted mean (2DGS unbounded default; "
                             "noisy for semi-transparent textured surfels). "
                             "1.0 = median depth (sharp, recommended for "
                             "mostly-opaque content with thin foliage).")
    parser.add_argument("--mesh_alpha_thresh", default=0.5, type=float,
                        help="Zero out depth at pixels where rend_alpha is below "
                             "this threshold before TSDF integration. Suppresses "
                             "spurious geometry from low-confidence pixels (where "
                             "depth_expected = total/tiny_alpha is extrapolated). "
                             "0 disables.")

    # Mesh-Gaussian sheet
    parser.add_argument("--mesh_disk_scale", default=0.7, type=float,
                        help="Disk scale as a fraction of mean face edge")

    # Optimization
    parser.add_argument("--lr_rgb", default=2e-2, type=float)
    parser.add_argument("--lr_op",  default=5e-3, type=float)
    parser.add_argument("--opt_iters", default=3000, type=int)
    parser.add_argument("--lambda_l1_rgb", default=1e-3, type=float,
                        help="L1 sparsity on mesh_rgb — encourage the mesh layer "
                             "to be nonzero only where it has to correct SV.")
    parser.add_argument("--lambda_ssim", default=0.2, type=float)

    # Bookkeeping
    parser.add_argument("--limit_views", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip TSDF mesh extraction; load existing "
                             "mesh_textured.ply from out_subdir")
    parser.add_argument("--save_every", default=500, type=int)
    parser.add_argument("--log_every",  default=50,  type=int)
    parser.add_argument("--cache_targets_to_ram", action="store_true",
                        help="Keep target tensors on GPU instead of CPU "
                             "(faster but RAM-hungry for many views).")

    args = get_combined_args(parser)
    args = fold_train_args(args, args.model_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_path = args.model_path
    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(exp_path, "point_cloud"))

    # Must precede any `from diff_surfel_3D_sh_res import set_*` inside the renderer.
    install_setter_mirror(getattr(args, "method", None))

    yaml_file = getattr(args, "yaml", None) or "tiny"
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    print(f"[BAKE] Model:     {exp_path}")
    print(f"[BAKE] Iter:      {iteration}")
    print(f"[BAKE] Method:    {getattr(args, 'method', '?')}")
    print(f"[BAKE] Kernel:    {getattr(args, 'kernel', '?')} / kernel2={getattr(args, 'kernel2', None)}")
    print(f"[BAKE] Feature:   {getattr(args, 'feature', '?')}")

    # INGP + CUDA globals
    ingp = INGP(cfg_model, args=args).to("cuda")
    ingp.load_model(exp_path, iteration)
    ingp.set_active_levels(iteration)
    if args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
        from diff_surfel_3D_sh_res import (
            set_residual_mode, set_activation_bias, set_compact_mult)
        sh_b, res_b = getattr(args, "activation_bias", [0.5, 0.0])
        set_activation_bias(sh_bias=float(sh_b), res_bias=float(res_b))
        # Use the saved-args residual_mode so old (mode-2) and new (mode-0)
        # checkpoints both render with the same activation pipeline they were
        # trained under. Default to 0 (post-mode-0-changes convention).
        _rm = int(getattr(args, "_residual_mode", 0))
        set_residual_mode(_rm)
        set_compact_mult(float(getattr(args, "fastgs_mult", 0.5)))
        # Mirror onto the renderer's Python-side per-pixel ReLU gate. Mode 2 →
        # signed per-Gauss feat, needs the deferred clamp; modes 0/1 → per-Gauss
        # is already non-negative, no per-pixel clamp.
        if ingp is not None:
            ingp.is_mixed_deferred_relu_mode = (_rm == 2)
        print(f"[BAKE] CUDA globals: sh_bias={sh_b} res_bias={res_b} "
              f"residual_mode={_rm} (deferred_relu={_rm==2}) "
              f"compact_mult={getattr(args, 'fastgs_mult', 0.5)}")

    # Scene
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    gaussians.feature_mode = getattr(args, "feature", "sh")
    gaussians._sv_training_flag = False
    print(f"[BAKE] feature_mode = {gaussians.feature_mode}")

    N_total = gaussians.get_xyz.shape[0]
    n_tex = int(gaussians._is_textured.sum().item())
    print(f"[BAKE] Loaded: {N_total:,} surfels  (textured {n_tex:,})")

    slice_to_textured(gaussians)
    print(f"[BAKE] After slice: {gaussians.get_xyz.shape[0]:,} (all textured)")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    bg_zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    out_root = os.path.join(exp_path, args.out_subdir)
    os.makedirs(out_root, exist_ok=True)
    print(f"[BAKE] Output:    {out_root}")

    # -----------------------------------------------------------------------
    # 1. Mesh extraction (or load)
    # -----------------------------------------------------------------------
    mesh_path = os.path.join(out_root, "mesh_textured.ply")
    train_cams = scene.getTrainCameras().copy()
    if args.limit_views > 0:
        train_cams = train_cams[:args.limit_views]

    if args.skip_extract and os.path.exists(mesh_path):
        print(f"[BAKE] Loading existing mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    else:
        print(f"[BAKE] TSDF-fusing {len(train_cams)} textured-only renders "
              f"({'unbounded' if args.unbounded else 'bounded'} mode, "
              f"depth_ratio={args.mesh_depth_ratio})…")
        # Set the depth_ratio that GaussianExtractor.reconstruction will use
        # when reading render_pkg['surf_depth'] (which is the
        # depth_expected/median blend). 1.0 = median (sharp, recommended for
        # semi-transparent textured surfels like leaves / weave); 0.0 = expected
        # (2DGS unbounded default, but noisy here).
        _orig_dr = getattr(pipe, "depth_ratio", 0.0)
        pipe.depth_ratio = float(args.mesh_depth_ratio)
        extractor = GaussianExtractor(
            render, gaussians, pipe, bg,
            ingp=ingp, beta=beta_cfg, iteration=iteration, cfg=cfg_model)
        extractor.reconstruction(train_cams)
        pipe.depth_ratio = _orig_dr   # restore (paranoia — render() below uses ingp path)

        # Alpha-gate the cached depth maps: zero out depths at pixels where the
        # textured-half alpha is below a threshold. extract_mesh_bounded already
        # does this via `mask_backgrond` IF the dataset has gt_alpha_mask, but
        # mip-360 has none → without this, TSDF integrates extrapolated
        # `depth_expected/tiny_alpha = huge_value` garbage and marching cubes
        # produces spurious fragmented surfaces. (Note: applied unconditionally
        # for our textured-only fusion since we WANT to exclude low-confidence
        # pixels, not just background.)
        if args.mesh_alpha_thresh > 0 and extractor.alphamaps:
            n_zeroed = 0
            n_total = 0
            for i in range(len(extractor.depthmaps)):
                alpha = extractor.alphamaps[i]
                bad = (alpha < args.mesh_alpha_thresh)
                n_zeroed += int(bad.sum().item())
                n_total += int(bad.numel())
                extractor.depthmaps[i][bad] = 0
            print(f"[BAKE] Alpha-gate (thr={args.mesh_alpha_thresh}): "
                  f"zeroed {n_zeroed:,}/{n_total:,} pixels "
                  f"({100.0*n_zeroed/max(1,n_total):.1f}%)")

        if args.unbounded:
            mesh = extractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            depth_trunc = (extractor.radius * float(args.depth_trunc_mult)) \
                          if args.depth_trunc < 0 else args.depth_trunc
            print(f"[BAKE] extractor.radius={extractor.radius:.3f}  "
                  f"depth_trunc={depth_trunc:.3f}  (= radius × {depth_trunc/max(extractor.radius,1e-6):.2f})")
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = extractor.extract_mesh_bounded(
                voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        mesh = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[BAKE] Mesh saved → {mesh_path}")
    print(f"[BAKE] Mesh: V={len(mesh.vertices):,} F={len(mesh.triangles):,}")

    # -----------------------------------------------------------------------
    # 2. Build mesh-Gaussian sheet
    # -----------------------------------------------------------------------
    verts_np = np.asarray(mesh.vertices)
    tris_np = np.asarray(mesh.triangles)
    if len(tris_np) > 0:
        e0 = np.linalg.norm(verts_np[tris_np[:, 0]] - verts_np[tris_np[:, 1]], axis=1)
        e1 = np.linalg.norm(verts_np[tris_np[:, 1]] - verts_np[tris_np[:, 2]], axis=1)
        e2 = np.linalg.norm(verts_np[tris_np[:, 2]] - verts_np[tris_np[:, 0]], axis=1)
        mean_edge = float(np.mean(np.concatenate([e0, e1, e2])))
    else:
        bbox = mesh.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(np.asarray(bbox.get_extent())))
        mean_edge = diag / float(np.sqrt(max(1, len(verts_np))))
    disk_scale = args.mesh_disk_scale * mean_edge
    print(f"[BAKE] Mean face edge: {mean_edge:.4f}, disk scale: {disk_scale:.4f}")

    mesh_gm, _, _ = build_mesh_gaussian_model(mesh, sh_degree=0, disk_scale=disk_scale)
    M = mesh_gm.get_xyz.shape[0]
    print(f"[BAKE] Mesh-Gaussians: {M:,}")

    # Learnable per-vertex SIGNED RGB. Initialize at 0 (no contribution).
    mesh_rgb_param = nn.Parameter(torch.zeros((M, 3), device="cuda"))
    mesh_opacity_param = mesh_gm._opacity   # already a learnable nn.Parameter

    # -----------------------------------------------------------------------
    # 3. Pre-cache per-view targets (full textured-only render WITH residual)
    #    and SV-only backgrounds (textured surfels, MLP zeroed).
    # -----------------------------------------------------------------------
    print(f"[BAKE] Caching {len(train_cams)} target + SV-only renders…")
    targets, surf_rgbs = [], []
    with torch.no_grad():
        for cam in tqdm(train_cams, desc="cache"):
            tgt_pkg = render(cam, gaussians, pipe, bg, ingp=ingp,
                             beta=beta_cfg, iteration=iteration, cfg=cfg_model,
                             skybox=None, background_mode="none", bg_hashgrid=None)
            target = torch.clamp(tgt_pkg["render"], 0.0, 1.0).detach().clone()
            sv_pkg = render(cam, gaussians, pipe, bg, ingp=ingp,
                            beta=beta_cfg, iteration=iteration, cfg=cfg_model,
                            skybox=None, background_mode="none", bg_hashgrid=None,
                            decompose_mode='sh_only')
            surf_rgb = sv_pkg["render"].detach().clone()    # NOT clamped — can exceed [0,1]
            if not args.cache_targets_to_ram:
                target = target.cpu(); surf_rgb = surf_rgb.cpu()
            targets.append(target)
            surf_rgbs.append(surf_rgb)
    print(f"[BAKE] Targets cached on {'GPU' if args.cache_targets_to_ram else 'CPU'}.")

    # -----------------------------------------------------------------------
    # 4. Optimization loop
    # -----------------------------------------------------------------------
    optimizer = Adam([
        {"params": [mesh_rgb_param],     "lr": args.lr_rgb, "name": "mesh_rgb"},
        {"params": [mesh_opacity_param], "lr": args.lr_op,  "name": "mesh_op"},
    ])

    print(f"[BAKE] Optimizing {args.opt_iters} iters…")
    log_l1, log_ss, log_reg = [], [], []
    pbar = tqdm(range(args.opt_iters), desc="opt")
    for it in pbar:
        idx = random.randint(0, len(train_cams) - 1)
        cam = train_cams[idx]
        target = targets[idx].cuda(non_blocking=True) if not args.cache_targets_to_ram else targets[idx]
        surf_rgb = surf_rgbs[idx].cuda(non_blocking=True) if not args.cache_targets_to_ram else surf_rgbs[idx]

        # --- Render mesh layer with signed override_color. Without ingp →
        # baseline rasterizer → colors_precomp = override_color respected.
        mesh_pkg = render(
            cam, mesh_gm, pipe, bg_zero,
            ingp=None, beta=0.0, iteration=0, cfg=cfg_model,
            override_color=mesh_rgb_param,
            skybox=None, background_mode="none", bg_hashgrid=None,
            is_training=False,
        )
        mesh_rgb_rendered = mesh_pkg["render"]              # (3, H, W) — signed
        mesh_alpha = mesh_pkg.get("rend_alpha", None)        # (1, H, W) ∈ [0, 1]
        if mesh_alpha is None:
            # Fallback: derive alpha from rendered intensity (shouldn't normally hit).
            mesh_alpha = (mesh_rgb_rendered.abs().sum(0, keepdim=True) > 0).float()

        # --- Composite: alpha-over with mesh in front (mesh + surfels are
        # collocated at the surface, so depth-order is approximate).
        pred = mesh_alpha * mesh_rgb_rendered + (1.0 - mesh_alpha) * surf_rgb
        pred = torch.relu(pred)                              # per-pixel ReLU
        pred = torch.clamp(pred, 0.0, 1.0)                   # match target's clamp

        # --- Loss
        l1 = l1_loss(pred, target)
        ss = 1.0 - ssim(pred.unsqueeze(0), target.unsqueeze(0))
        reg = mesh_rgb_param.abs().mean()
        loss = l1 + args.lambda_ssim * ss + args.lambda_l1_rgb * reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        log_l1.append(l1.item())
        log_ss.append(ss.item())
        log_reg.append(reg.item())

        if (it + 1) % args.log_every == 0:
            n = min(args.log_every, len(log_l1))
            pbar.set_postfix(
                l1=f"{np.mean(log_l1[-n:]):.4f}",
                ssim_loss=f"{np.mean(log_ss[-n:]):.4f}",
                reg=f"{np.mean(log_reg[-n:]):.4f}",
                mrgb_rng=f"[{mesh_rgb_param.min().item():+.2f},{mesh_rgb_param.max().item():+.2f}]",
                malpha_mean=f"{torch.sigmoid(mesh_opacity_param).mean().item():.2f}",
            )

        if (it + 1) % args.save_every == 0 or (it + 1) == args.opt_iters:
            ckpt_path = os.path.join(out_root, f"mesh_rgba_it{it+1:05d}.ply")
            save_mesh_rgba(mesh, mesh_rgb_param, mesh_opacity_param, ckpt_path)

    final_path = os.path.join(out_root, "mesh_rgba_final.ply")
    save_mesh_rgba(mesh, mesh_rgb_param, mesh_opacity_param, final_path)
    print(f"[BAKE] Saved final → {final_path}")
    print(f"[BAKE]   (uint8 companion at {final_path.replace('.ply', '_u8.ply')})")

    # Validation: full-set PSNR of pred vs target on the train set.
    print(f"[BAKE] Validation pass over {len(train_cams)} train views…")
    psnrs = []
    with torch.no_grad():
        for idx, cam in enumerate(train_cams):
            target = targets[idx].cuda() if not args.cache_targets_to_ram else targets[idx]
            surf_rgb = surf_rgbs[idx].cuda() if not args.cache_targets_to_ram else surf_rgbs[idx]
            mesh_pkg = render(
                cam, mesh_gm, pipe, bg_zero,
                ingp=None, beta=0.0, iteration=0, cfg=cfg_model,
                override_color=mesh_rgb_param,
                skybox=None, background_mode="none", bg_hashgrid=None,
                is_training=False,
            )
            mesh_rgb_rendered = mesh_pkg["render"]
            mesh_alpha = mesh_pkg.get("rend_alpha", None)
            if mesh_alpha is None:
                mesh_alpha = (mesh_rgb_rendered.abs().sum(0, keepdim=True) > 0).float()
            pred = mesh_alpha * mesh_rgb_rendered + (1.0 - mesh_alpha) * surf_rgb
            pred = torch.relu(pred).clamp(0.0, 1.0)
            mse = ((pred - target) ** 2).mean().item()
            psnrs.append(-10.0 * math.log10(mse + 1e-12))
    print(f"[BAKE] Train PSNR (pred vs textured-target): {np.mean(psnrs):.2f} dB "
          f"(median {np.median(psnrs):.2f}, min {np.min(psnrs):.2f})")


if __name__ == "__main__":
    main()
