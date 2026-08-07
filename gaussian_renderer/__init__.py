#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math

# Import rasterizer - use lean library if available, fall back to main
# The lean library (diff_surfel_3D) is a stripped-down version for faster builds
import os
_USE_LEAN_ONLY = os.environ.get('USE_LEAN_RASTERIZER', '0') == '1'

try:
    import diff_surfel_3D as _lean_rasterizer
    from diff_surfel_3D import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
    LEAN_RASTERIZER_AVAILABLE = True
    _main_rasterizer = None
except ImportError:
    _lean_rasterizer = None
    LEAN_RASTERIZER_AVAILABLE = False

# FP16 lean library (diff_surfel_3D_16) — FP16 weights + FP16 GEMM shared memory
try:
    import diff_surfel_3D_16 as _fp16_rasterizer
    FP16_RASTERIZER_AVAILABLE = True
except ImportError:
    _fp16_rasterizer = None
    FP16_RASTERIZER_AVAILABLE = False

# TC lean library (diff_surfel_3D_tc) — Tensor Core WMMA for MLP
try:
    import diff_surfel_3D_tc as _tc_rasterizer
    TC_RASTERIZER_AVAILABLE = True
except ImportError:
    _tc_rasterizer = None
    TC_RASTERIZER_AVAILABLE = False

# SH TC library (diff_surfel_3D_sh) — TC WMMA MLP → SH coefficients
try:
    import diff_surfel_3D_sh as _sh_tc_rasterizer
    SH_TC_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_tc_rasterizer = None
    SH_TC_RASTERIZER_AVAILABLE = False

# SH+residual library (diff_surfel_3D_sh_res) — per-Gaussian SH + tiny hash MLP residual
try:
    import diff_surfel_3D_sh_res as _sh_res_rasterizer
    SH_RES_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_res_rasterizer = None
    SH_RES_RASTERIZER_AVAILABLE = False

# 3D_SH_filmres: 3D_SH_res fork that FiLM-conditions the residual MLP's hash input.
# Same kernel family as diff_surfel_3D_sh_res; its device-global setters (set_mlp_weights,
# set_activation_bias, set_residual_mode, ...) are module-local, so for filmres they MUST
# target this module, not diff_surfel_3D_sh_res. Use _sh_res_setter_mod(ingp) to resolve.
try:
    import diff_surfel_3D_sh_filmres as _sh_filmres_rasterizer
    SH_FILMRES_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_filmres_rasterizer = None
    SH_FILMRES_RASTERIZER_AVAILABLE = False

# diff_surfel_3D_sh_res_harden: an ISOLATED clone of diff_surfel_3D_sh_res used
# ONLY for the GEStex explore+harden phase (0-20k, before the joint transition).
# It is byte-identical to the shared rasterizer EXCEPT it carries the first-
# intersection (tile-depth) sort (set_tile_depth_sort). GEStex-specific rasterizer
# changes go HERE, never in the shared diff_surfel_3D_sh_res (see
# docs/GESTEX_PIPELINE.md "CUDA isolation rule"). The name deliberately KEEPS the
# `diff_surfel_3D_sh_res` prefix so the renderer's substring gates below
# (metric_map ON; is_textured / scaling_z / film OFF) resolve exactly as they do
# for the base — and does NOT contain `diff_surfel_gestex` / `_mixed` / `_res_3d`,
# which would wrongly trip those gates. Its device-globals are module-local, so
# both the per-render set_mlp_weights (via _sh_res_setter_mod) AND train.py's
# _SHRES_SETTER_MOD resolve to this module for a GEStex harden.
try:
    import diff_surfel_3D_sh_res_harden as _gestex_harden_rasterizer
    GESTEX_HARDEN_RASTERIZER_AVAILABLE = True
except ImportError:
    _gestex_harden_rasterizer = None
    GESTEX_HARDEN_RASTERIZER_AVAILABLE = False

# diff_surfel_3D_sh_res_densfix: an ISOLATED clone of diff_surfel_3D_sh_res for
# `--densfix` (--method 3D_SH_res only). Byte-identical to the base EXCEPT the
# backward optionally excludes the hash-query-point term from the AbsGS
# densification proxy (set_exclude_hash_from_densify) — surfels still reposition
# on the full SV+hash gradient, but densification drops the hash-inflated boost.
# Name keeps the `diff_surfel_3D_sh_res` prefix (substring gates resolve as base)
# and avoids gestex/mixed/res_3d/film substrings.
try:
    import diff_surfel_3D_sh_res_densfix as _densfix_rasterizer
    DENSFIX_RASTERIZER_AVAILABLE = True
except ImportError:
    _densfix_rasterizer = None
    DENSFIX_RASTERIZER_AVAILABLE = False

# diff_surfel_3D_sh_res_trunc: an ISOLATED clone of diff_surfel_3D_sh_res for
# `--trunc` (--method 3D_SH_res only). Byte-identical to the base EXCEPT a
# settable POST-blend truncation exit threshold (set_exit_T, default 1e-4):
# at e.g. 0.5 the forward walk stops once T drops below 0.5 — the crossing
# fragment still blends, so an opaque terminator can drive T→0 and kill the
# Python-side (1−rend_alpha)·noise term (opacity-cliff / base-plate training).
# Name keeps the `diff_surfel_3D_sh_res` prefix (substring gates resolve as
# base) and avoids gestex/mixed/res_3d/film substrings.
try:
    import diff_surfel_3D_sh_res_trunc as _trunc_rasterizer
    TRUNC_RASTERIZER_AVAILABLE = True
except ImportError:
    _trunc_rasterizer = None
    TRUNC_RASTERIZER_AVAILABLE = False

# diff_surfel_3D_sh_res_probe: an ISOLATED clone of diff_surfel_3D_sh_res for
# `--method proberes`. The residual is a bilinear fetch from ONE shared texture
# image via per-surfel affine probes (texcoord = A·uv + t) instead of the 3D
# hash + fused MLP. Signaled by render_mode = 5 | 0x1000; probes [N,6] ride the
# features_diffuse kwarg, the texture [R,R,3] rides gridrange_diffuse, dims
# {Ht,Wt} ride offsets_diffuse — the kernel returns dL/dprobes + dL/dtex through
# those autograd slots. Name keeps the `diff_surfel_3D_sh_res` prefix so the
# substring gates below resolve as base.
try:
    import diff_surfel_3D_sh_res_probe as _sh_res_probe_rasterizer
    SH_RES_PROBE_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_res_probe_rasterizer = None
    SH_RES_PROBE_RASTERIZER_AVAILABLE = False

# diff_surfel_3D_sh_res_probe_wsr: isolated clone of the probe rasterizer with
# the WSR sort-free weighted-sum composite (`--wsr`, docs/WSR_DISTILL.md).
# Byte-identical to the probe clone under set_wsr(0); its record_transmittance
# accumulators dump the distill targets (Σα, Σ(α·T)).
try:
    import diff_surfel_3D_sh_res_probe_wsr as _sh_res_probe_wsr_rasterizer
    SH_RES_PROBE_WSR_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_res_probe_wsr_rasterizer = None
    SH_RES_PROBE_WSR_RASTERIZER_AVAILABLE = False

def _is_proberes(ingp):
    """True for a --method proberes run: render + setters route to the isolated
    diff_surfel_3D_sh_res_probe clone (or its _wsr clone under --wsr)."""
    return (ingp is not None and getattr(ingp, 'is_proberes_mode', False)
            and (SH_RES_PROBE_RASTERIZER_AVAILABLE
                 or _is_proberes_wsr(ingp)))

def _is_proberes_wsr(ingp):
    """True for a --wsr proberes run: routes to diff_surfel_3D_sh_res_probe_wsr."""
    return (ingp is not None and getattr(ingp, 'is_proberes_mode', False)
            and getattr(ingp, 'is_wsr_mode', False)
            and SH_RES_PROBE_WSR_RASTERIZER_AVAILABLE)

def _is_gestex_harden(ingp):
    """True for a GEStex run in its explore+harden phase (0-20k), i.e. rendering
    through the isolated diff_surfel_3D_sh_res_harden clone rather than the shared
    3D_SH_res rasterizer. False once the joint stage is entered."""
    return (ingp is not None and getattr(ingp, 'is_gestex_mode', False)
            and not getattr(ingp, 'is_gestex_joint', False)
            and GESTEX_HARDEN_RASTERIZER_AVAILABLE)

def _is_densfix(ingp):
    """True for a `--densfix` run (--method 3D_SH_res): render + setters route to
    the isolated diff_surfel_3D_sh_res_densfix clone."""
    return (ingp is not None and getattr(ingp, 'is_densfix_mode', False)
            and DENSFIX_RASTERIZER_AVAILABLE)

def _is_trunc(ingp):
    """True for a `--trunc` run (--method 3D_SH_res): render + setters route to
    the isolated diff_surfel_3D_sh_res_trunc clone (settable exit_T)."""
    return (ingp is not None and getattr(ingp, 'is_trunc_mode', False)
            and TRUNC_RASTERIZER_AVAILABLE)

def _sh_res_setter_mod(ingp):
    """Module whose device-global setters back the active 3D_SH_res-family render.
    For --method 3D_SH_filmres that's diff_surfel_3D_sh_filmres; for a GEStex
    explore/harden it's the isolated diff_surfel_3D_sh_res_harden clone; for
    --densfix the diff_surfel_3D_sh_res_densfix clone; otherwise the base."""
    if _is_gestex_harden(ingp):
        return _gestex_harden_rasterizer
    if (ingp is not None and getattr(ingp, 'is_3D_SH_filmres_mode', False)
            and SH_FILMRES_RASTERIZER_AVAILABLE):
        return _sh_filmres_rasterizer
    if _is_densfix(ingp):
        return _densfix_rasterizer
    if _is_trunc(ingp):
        return _trunc_rasterizer
    if _is_proberes_wsr(ingp):
        return _sh_res_probe_wsr_rasterizer
    if _is_proberes(ingp):
        return _sh_res_probe_rasterizer
    return _sh_res_rasterizer

# `--method mixed` library — fork of diff_surfel_3D_sh_res that will host
# the diffuse-textured / specular-untextured per-Gauss kernel branch.
# Currently identical to the 3D_SH_res renderer (CUDA branch is phase 2).
try:
    import diff_surfel_mixed as _mixed_rasterizer
    MIXED_RASTERIZER_AVAILABLE = True
except ImportError:
    _mixed_rasterizer = None
    MIXED_RASTERIZER_AVAILABLE = False

# `--method mixed_3d` library — like diff_surfel_mixed but the untextured half
# renders as 3D ellipsoids (beta-splatting EWA, restricted beta_scaled kernel).
try:
    import diff_surfel_mixed_3d as _mixed_3d_rasterizer
    MIXED_3D_RASTERIZER_AVAILABLE = True
except ImportError:
    _mixed_3d_rasterizer = None
    MIXED_3D_RASTERIZER_AVAILABLE = False

# `--method res_3d` SINGLE-PASS library — fork of diff_surfel_mixed_3d with
# single-pass dual-cascade forward + backward. Forward emits out_color =
# C_sv_aux + C_tex_aux (signed sum); backward maintains independent T_sv /
# T_tex reverse cascades and routes color/alpha grads per-Gauss via
# is_textured. Renderer composes LRU(out_color) for the final image.
try:
    import diff_surfel_res_3d as _res_3d_rasterizer
    RES_3D_RASTERIZER_AVAILABLE = True
except ImportError:
    _res_3d_rasterizer = None
    RES_3D_RASTERIZER_AVAILABLE = False

# `--method res_3d_paired` SLIM library — a clone of diff_surfel_mixed_3d
# with `out_others` trimmed from 18 channels to 5 (DEPTH + ALPHA + NORMAL).
# At 4K image resolution this saves ~870 MB of per-pixel forward output and
# the matching backward gradient tensor. Functionally identical otherwise.
# The renderer routes res_3d_paired post-split through this when available.
try:
    import diff_surfel_res_3d_paired as _res_3d_paired_rasterizer
    RES_3D_PAIRED_RASTERIZER_AVAILABLE = True
except ImportError:
    _res_3d_paired_rasterizer = None
    RES_3D_PAIRED_RASTERIZER_AVAILABLE = False

# `--method film`: FiLM (Feature-wise Linear Modulation) rasterizer. Fork of
# diff_surfel_rasterization (cat mode) with per-Gauss gamma/beta modulating the
# hashgrid feature (f = gamma*H + beta) before blend. MLP runs in PyTorch (cat-family).
try:
    import diff_surfel_film as _film_rasterizer
    FILM_RASTERIZER_AVAILABLE = True
except ImportError:
    _film_rasterizer = None
    FILM_RASTERIZER_AVAILABLE = False

# SH+residual 32-dim library (diff_surfel_3D_sh_32) — same as sh_res but 32-dim hidden MLP
try:
    import diff_surfel_3D_sh_32 as _sh_32_rasterizer
    SH_32_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_32_rasterizer = None
    SH_32_RASTERIZER_AVAILABLE = False

# 3D_SH_concat (diff_surfel_3D_sh_concat) — 32-dim MLP, input = concat[surfel latent(16) | hash(16)]
try:
    import diff_surfel_3D_sh_concat as _sh_concat_rasterizer
    SH_CONCAT_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_concat_rasterizer = None
    SH_CONCAT_RASTERIZER_AVAILABLE = False

# `--method GEStex` joint-stage rasterizer — a clone of diff_surfel_res_3d_paired
# (textured 2D surfels + untextured EWA 3D "Gaussians", joint cascade, full backward)
# with the textured residual swapped from hash+MLP to an explicit per-surfel RGB
# texture-atlas bilinear lookup (set_gestex_atlas). Built on demand.
try:
    import diff_surfel_gestex as _gestex_rasterizer
    GESTEX_RASTERIZER_AVAILABLE = True
except ImportError:
    _gestex_rasterizer = None
    GESTEX_RASTERIZER_AVAILABLE = False
# `--method GEStex` SORT-FREE 2-pass kernels (full-hardening joint stage):
#   joint_s = surfel z-buffer (frontmost + SV + atlas); joint_g = additive depth-tested Gaussians.
try:
    import diff_surfel_gestex_joint_s as _gestex_joint_s
    import diff_surfel_gestex_joint_g as _gestex_joint_g
    GESTEX_JOINT_S_AVAILABLE = True
    GESTEX_JOINT_G_AVAILABLE = True
except ImportError:
    _gestex_joint_s = None
    _gestex_joint_g = None
    GESTEX_JOINT_S_AVAILABLE = False
    GESTEX_JOINT_G_AVAILABLE = False


@torch.no_grad()
def ges_bake_atlas(ingp, pc, R, uv_extent=4.0):
    """`--method GEStex` bake: evaluate the trained hashgrid+fused-MLP residual at each
    surfel's R×R UV lattice and return an unbounded RGB atlas [N, R, R, 3] (u-major,
    v-minor; view-independent). Recipe is byte-faithful to scripts/benchmark_baked.py's
    bake_atlas (FP16 MLP math, texel-center UV, no bias column). Only textured (surfel)
    rows are baked; the caller has not yet spawned the untextured 3D Gaussians."""
    from utils.general_utils import build_rotation
    device = pc.get_xyz.device
    centers = pc.get_xyz
    N = centers.shape[0]
    Rot = build_rotation(pc.get_rotation)          # [N,3,3]
    r0 = Rot[:, :, 0]                               # tangent u axis
    r1 = Rot[:, :, 1]                               # tangent v axis
    scales = pc.get_scaling                         # [N,>=2]
    sx = scales[:, 0:1]
    sy = scales[:, 1:2]

    # Deep-copy the MLP to FP16 for the bake — nn.Module.half() is IN-PLACE, and
    # mutating ingp.mlp_fused would break the renderer's later set_mlp_weights (which
    # expects FP32 weights). The copy matches the training kernel's __half2 math.
    import copy
    mlp = copy.deepcopy(ingp.mlp_fused).half().eval()
    hash_dim = int(ingp.mlp_fused_hash_dim)
    mlp_input_padded = int(mlp[0].weight.shape[1])

    step = 2.0 * uv_extent / R
    uv = (torch.arange(R, dtype=torch.float32, device=device) + 0.5) * step - uv_extent
    uu, vv = torch.meshgrid(uv, uv, indexing='ij')  # [R,R] (u-major, v-minor)
    u_flat = uu.reshape(-1)                          # [R*R]
    v_flat = vv.reshape(-1)
    n_pts = R * R

    atlas = torch.zeros(N, R, R, 3, dtype=torch.float32, device=device)
    # Chunk over surfels to bound memory.
    bytes_per = 168 * n_pts
    chunk = max(1, int((4 * (1024 ** 3)) // max(bytes_per, 1)))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        c = centers[s:e]                             # [b,3]
        xyz = (c.unsqueeze(1)
               + u_flat.view(1, -1, 1) * (sx[s:e].unsqueeze(1) * r0[s:e].unsqueeze(1))
               + v_flat.view(1, -1, 1) * (sy[s:e].unsqueeze(1) * r1[s:e].unsqueeze(1)))  # [b,R*R,3]
        xyz_flat = xyz.reshape(-1, 3)
        hash_feat = ingp._encode_3D(xyz_flat)
        mlp_in = torch.zeros(xyz_flat.shape[0], mlp_input_padded, device=device, dtype=torch.float16)
        mlp_in[:, :hash_dim] = hash_feat[:, :hash_dim].to(torch.float16)
        rgb = mlp(mlp_in)[:, :3].float()             # [b*R*R,3], residual (view-independent)
        atlas[s:e] = rgb.reshape(e - s, R, R, 3)
    return atlas

# Import main rasterizer if lean-only mode not set
if not _USE_LEAN_ONLY:
    try:
        import diff_surfel_rasterization as _main_rasterizer
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
    except ImportError:
        if not LEAN_RASTERIZER_AVAILABLE:
            raise ImportError("Neither diff_surfel_rasterization nor diff_surfel_3D could be imported. Build one of them.")
        _main_rasterizer = None
else:
    _main_rasterizer = None

from scene.gaussian_model import GaussianModel
from scene.camera_pose_opt import quaternion_multiply as _pose_quaternion_multiply
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal, save_points, depths_to_points, cam2rays
import torch.nn.functional as torch_F
import time
# from hash_encoder.FeatureBlend import FeatureBlend
from utils.general_utils import MEM_PRINT


class STERelu(torch.autograd.Function):
    """`--ste`: SIGN-AWARE straight-through ReLU for `--method mixed[_3d]_sep`.

    Forward: standard ReLU (identical to `torch.relu`).

    Backward: gradient passes where
      - `x > 0`  (the normal "non-clamped" case), OR
      - `x ≤ 0` AND `grad_out < 0`  (loss wants this pixel HIGHER → pushing
        the residual up will release the clamp and reduce loss).

    The asymmetric gate eliminates the "wasted gradient pushing parameters
    deeper negative at clamped pixels" pathology of naive STE — at a clamped
    pixel where `grad_out ≥ 0` the gate is 0, since pushing residual further
    negative wouldn't change the forward (still clamped) but would drift
    parameters with no loss feedback.

    Used in mode 2 (`mixed_sep` / `mixed_3d_sep`) where the per-pixel ReLU
    after blend is what's killing gradient. Mode-0's per-Gauss STE has a
    parallel CUDA-side implementation (`d_ste_relu` in backward.cu).
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # Pass where forward was unclamped (x > 0), OR at clamped pixels where
        # the gradient would push toward release (grad_out < 0).
        gate = (x > 0) | (grad_out < 0)
        return grad_out * gate.to(grad_out.dtype)


# --feature beta: spherical-beta directional color function.
# Per-Gaussian view-dependent RGB:
#     C(v) = sum_i softplus(rgb_i) * max(0, dot(mu_i, v))^(4 * exp(beta))
# where (theta_i, phi_i) -> mu_i is one of K lobe directions and `beta` is a shared
# per-Gaussian sharpness. Returns [N, 3] in natural RGB space (NOT SH-scaled).
# See beta-splatting reference: submodules/gsplat/cuda/csrc/spherical_beta.cuh.
_SH_C0 = 0.28209479177387814


def eval_sb(sb_params: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
    """Spherical beta directional color evaluation. Matches beta-splatting reference:
       contrib = sum_k softplus_steep(rgb_k) · dot^(4·exp(beta_k))
    with per-primitive beta at index 5 and dot clamped via `dot > 0`.

    Args:
        sb_params:  [N, K, 6]  per-primitive (r, g, b, theta, phi, beta_per_primitive).
        view_dirs:  [N, 3]     unit view direction per Gaussian (camera→Gaussian center).

    Returns:
        [N, 3] RGB contribution from the K primitive lobes.
    """
    import math
    # Steep softplus: beta=10*ln(2)≈6.93, matches reference
    rgb   = torch.nn.functional.softplus(sb_params[..., 0:3], beta=10.0 * math.log(2.0))  # [N, K, 3]
    theta = sb_params[..., 3]                                             # [N, K]
    phi   = sb_params[..., 4]                                             # [N, K]
    beta_per_primitive = sb_params[..., 5]                                # [N, K]

    sin_t = torch.sin(theta)
    mu = torch.stack([
        sin_t * torch.cos(phi),
        sin_t * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)                                                            # [N, K, 3]

    # Reference CUDA: betaTerm = dot > 0 ? dot^(4·exp(β)) : 0
    dot = (mu * view_dirs.unsqueeze(1)).sum(dim=-1)                        # [N, K]
    expo = 4.0 * torch.exp(beta_per_primitive)                             # [N, K] per-primitive
    term = torch.where(dot > 0, dot.clamp_min(1e-12).pow(expo), torch.zeros_like(dot))
    contrib = (term.unsqueeze(-1) * rgb).sum(dim=1)                        # [N, 3]
    return contrib


def eval_sg(sg_directions: torch.Tensor, sg_sharpness: torch.Tensor,
            sg_rgb: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
    """MEGS-2 Spherical Gaussian directional color evaluation.

    color = Σ_k softplus(sg_rgb[k]) · exp(|λ_k| · (cos θ_k − 1))

    `softplus` on rgb keeps per-lobe contributions non-negative so the downstream
    SH clamp (`max(color, 0)` in the rasterizer) never zeros real SG output.

    Args:
        sg_directions: [N, K, 3] raw direction vectors (will be normalized).
        sg_sharpness:  [N, K, 1] raw sharpness (|·| activation).
        sg_rgb:        [N, K, 3] raw per-axis RGB (softplus'd at eval).
        view_dirs:     [N, 3]    unit view direction per Gaussian.

    Returns:
        [N, 3] additive RGB contribution, always >= 0.
    """
    d = sg_directions / (sg_directions.norm(dim=-1, keepdim=True) + 1e-8)  # [N, K, 3]
    cos_theta = (d * view_dirs.unsqueeze(1)).sum(dim=-1)                    # [N, K]
    lam = torch.abs(sg_sharpness.squeeze(-1))                               # [N, K]
    scale = torch.exp(lam * (cos_theta - 1.0))                              # [N, K]
    rgb_pos = torch.nn.functional.softplus(sg_rgb)                          # [N, K, 3] > 0
    contrib = (scale.unsqueeze(-1) * rgb_pos).sum(dim=1)                    # [N, 3]
    return contrib


def eval_voronoi(sv_sites: torch.Tensor, sv_colors: torch.Tensor,
                 view_dirs: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """Spherical Voronoi (radiance) directional color evaluation.

    Two metrics supported (both match sphericalvoronoi/radiance):
      - 'l2'     : logits = -τ_k · ||site_k_unit − ω||    (L2 distance, default)
      - 'cosine' : logits = s_k · ω                       (paper formulation,
                   unconstrained dot product; τ is implicit in ||s_k||).

    Args:
        sv_sites:   [N, K, 3] raw direction vectors (magnitude = τ).
        sv_colors:  [N, K, 3] per-site RGB.
        view_dirs:  [N, 3]    unit view direction per Gaussian.
        metric:     'l2' or 'cosine'.
    """
    if metric == "cosine":
        # Paper-clean: logits_k = s_k · ω  (no normalization, no sqrt).
        logits = (sv_sites * view_dirs.unsqueeze(1)).sum(dim=-1)            # [N, K]
    else:
        tau = torch.norm(sv_sites, dim=-1)                                  # [N, K]
        site_dirs = sv_sites / (tau.unsqueeze(-1) + 1e-8)                   # [N, K, 3]
        diff = site_dirs - view_dirs.unsqueeze(1)                           # [N, K, 3]
        dist = torch.norm(diff, dim=-1)                                     # [N, K]
        logits = -tau * dist                                                # [N, K]
    W = torch.softmax(logits, dim=-1).unsqueeze(-1)                         # [N, K, 1]
    V = (W * sv_colors).sum(dim=1)                                          # [N, 3]
    return torch.clamp_min(V, 0.0)


def eval_voronoi_sv_feat(sv_sites: torch.Tensor, sv_colors: torch.Tensor,
                         view_dirs: torch.Tensor,
                         sv_tau: torch.Tensor = None,
                         sites_mask: torch.Tensor = None) -> torch.Tensor:
    """Spherical Voronoi softmax-weighted feature (pre-bias, pre-clamp).

    Returns the raw `sum_k W_k · colors_k` so callers choose how to apply the
    `+0.5` bias and clamp:
      • `relu(feat + 0.5)` for the `colors_precomp` path (vanilla 2DGS).
      • `feat / SH_C0` plopped into the fake SH-DC slot so the rasterizer's
        `clamp(SH·DC + sh_bias=0.5, 0)` reproduces the same formula.

    `sv_tau` (optional, post-activation, e.g. `exp(_sv_tau)`) decouples site
    direction from softmax sharpness — matches 2dgs-voronoi `tau_mode='param'`.
    When None, falls back to radiance-paper convention `tau = ||sites||`.
    """
    site_dirs = sv_sites / (sv_sites.norm(dim=-1, keepdim=True) + 1e-12)    # [N, K, 3]
    tau = sv_tau if sv_tau is not None else torch.norm(sv_sites, dim=-1)    # [N, K]
    if sites_mask is not None:
        far = torch.full_like(site_dirs, 1e8)
        site_dirs = torch.where(sites_mask.unsqueeze(-1), site_dirs, far)
    diff = site_dirs - view_dirs.unsqueeze(1)                               # [N, K, 3]
    dist = torch.norm(diff, dim=-1)                                         # [N, K]
    logits = -tau * dist                                                    # [N, K]
    W = torch.softmax(logits, dim=-1).unsqueeze(-1)                         # [N, K, 1]
    feat = (W * sv_colors).sum(dim=1)                                       # [N, 3]
    return feat


def eval_voronoi_sv(sv_sites: torch.Tensor, sv_colors: torch.Tensor,
                    view_dirs: torch.Tensor,
                    sv_tau: torch.Tensor = None,
                    sites_mask: torch.Tensor = None,
                    sv_dc: torch.Tensor = None) -> torch.Tensor:
    """SV RGB, post-bias, post-clamp. Mirrors 2dgs-voronoi: `relu(feat + 0.5)`.

    Init convention: `_sv_colors = pcd_rgb - 0.5` so initial render = pcd_rgb.
    Legacy `sv_dc` argument adds a per-Gaussian DC to feat before bias+clamp.
    """
    feat = eval_voronoi_sv_feat(sv_sites, sv_colors, view_dirs,
                                 sv_tau=sv_tau, sites_mask=sites_mask)
    if sv_dc is not None:
        feat = feat + sv_dc
    return torch.nn.functional.relu(feat + 0.5)


def _build_fake_shs_from_voronoi(pc, view_dirs, max_sh_degree, sh_bias: float = 0.5,
                                  metric: str = "l2"):
    """Hybrid SH-DC + SV. SH degree-0 (DC) carries the view-independent base color;
    SV adds a view-dependent refinement on top. Higher-order SH stays zero.

    Rasterizer computes `color = clamp(SH_C0 * fake_dc + sh_bias, 0)` + (any
    higher-order SH, but fake_rest=0 so none). To inject SV additively:
       fake_dc = real_dc + SV / SH_C0
    so final = SH_C0 * real_dc + sh_bias + SV = pcd_RGB_baseline + view-dep SV."""
    sv_rgb = eval_voronoi(pc._sv_sites, pc._sv_colors, view_dirs, metric=metric)
    real_shs = pc.get_features  # [N, M, 3]
    fake = real_shs.new_zeros(real_shs.shape)
    fake[:, 0, :] = real_shs[:, 0, :] + sv_rgb / _SH_C0  # DC + SV/SH_C0
    # fake[:, 1:, :] stays zero — higher-order SH disabled
    return fake


def _build_fake_shs_from_SV(pc, view_dirs, sv_lru=0.0):
    """SV via fake-SH injection (2dgs-voronoi formulation: `relu(feat + 0.5)`).

    Plumbs SV into both rendering paths so the same `relu(feat + 0.5)` comes
    out regardless of which one the dispatch picks:

      • `colors_precomp` path (`override_color = sv_rgb`):
        sv_rgb is the post-bias, post-clamp RGB; rasterizer renders it directly.

      • SH-DC slot path (3D_SH_res / 3D_SH_cat / baseline SH):
        fake_dc = feat / SH_C0 (PRE-bias, NO clamp). The rasterizer evaluates
        `clamp(SH_C0 · fake_dc + sh_bias, 0) = clamp(feat + sh_bias, 0)`. With
        the default `--activation_bias 0.5 0.0`, that reproduces 2dgs-voronoi's
        `relu(feat + 0.5)` exactly.

    Real `_features_dc` is intentionally ignored. Higher-order SH stays zero.
    Legacy `_sv_dc` (--sv_dc opt-in) is added to feat before bias+clamp.
    """
    _sv_mask = getattr(pc, '_sv_mask', None)
    apply_mask = (
        _sv_mask is not None
        and (not getattr(pc, '_sv_training_flag', True))
        and _sv_mask.shape[0] == pc._sv_sites.shape[0]
    )
    _sv_dc = getattr(pc, '_sv_dc', None)
    if _sv_dc is not None and _sv_dc.numel() == 0:
        _sv_dc = None
    _sv_tau_raw = getattr(pc, '_sv_tau', None)
    if _sv_tau_raw is not None and _sv_tau_raw.numel() == 0:
        _sv_tau = None
    elif _sv_tau_raw is not None:
        _sv_tau = torch.exp(_sv_tau_raw)
    else:
        _sv_tau = None
    feat = eval_voronoi_sv_feat(
        pc._sv_sites, pc._sv_colors, view_dirs,
        sv_tau=_sv_tau,
        sites_mask=(_sv_mask if apply_mask else None),
    )
    if _sv_dc is not None:
        feat = feat + _sv_dc
    # `--sv_lru α`: leaky inner ReLU on the SV base, relu(feat+0.5). α=0 ⇒ F.leaky_relu is
    # exactly relu (byte-identical default). α>0 lets the SV base recover when feat+0.5<0 —
    # autograd carries the leaky gradient here into _sv_sites/_sv_colors/_sv_tau, so the base
    # keeps learning instead of dying (the inner-ReLU analog of what --lru does for the outer).
    sv_rgb = torch.nn.functional.leaky_relu(feat + 0.5, negative_slope=sv_lru)
    real_shs = pc.get_features                                              # [N, M, 3]
    fake = real_shs.new_zeros(real_shs.shape)
    fake[:, 0, :] = feat / _SH_C0  # pre-bias slot — rasterizer adds sh_bias
    return fake, sv_rgb


def _build_fake_shs_from_sg(pc, view_dirs, max_sh_degree, sh_bias: float = 0.5):
    """Add the SG contribution on top of the real SH, matching MEGS-2:
    `color = clamp(SH_eval + sh_bias + SG_sum, 0)`.
    """
    sg_rgb_out = eval_sg(pc._sg_directions, pc._sg_sharpness_sg, pc._sg_rgb, view_dirs)
    real_shs = pc.get_features
    fake = real_shs.clone()
    fake[:, 0, :] = real_shs[:, 0, :] + sg_rgb_out / _SH_C0
    return fake


def _build_fake_shs_from_sb(pc, view_dirs, max_sh_degree, sh_bias: float = 0.5):
    """Add the spherical-beta contribution on top of the real SH, matching
    beta-splatting's: `color = clamp(SH_eval + sh_bias + SB_sum, 0)`.

    Rasterizer computes `SH_C0 * fake_dc + sh_bias + higher_SH + ...`.
    Real SH computes: `SH_C0 * dc + sh_bias + higher_SH`.
    To add SB contribution: fake_dc = dc + SB_sum / SH_C0.
    The higher-order SH terms carry through unchanged.
    """
    beta_rgb = eval_sb(pc._sb_params, view_dirs)                           # [N, 3]
    real_shs = pc.get_features                                              # [N, M, 3]
    fake = real_shs.clone()
    # real_shs[:,0,:] is the DC coefficient; the rasterizer multiplies it by SH_C0.
    # Adding beta_rgb / SH_C0 to DC injects beta_rgb into the final color additively.
    fake[:, 0, :] = real_shs[:, 0, :] + beta_rgb / _SH_C0
    return fake

# One-time verification flags for render modes
_3D_DIRECT_FUSED_VERIFIED = False
_ACTIVATION_BIAS = [0.5, 0.0]  # [sh_bias, res_bias] — set from train.py via set_default_activation_bias()


class IntersectionOpacityGrad(torch.autograd.Function):
    """
    Custom autograd Function to connect intersection weight gradients back to Gaussian
    opacity, scale, rotation, AND position (through the exact geometry gradient path via transMat).

    This provides a SINGLE clean gradient path for geometry parameters, similar to how
    cat mode gets all geometry gradients through the CUDA rasterizer backward pass.

    CRITICAL DESIGN: This function also computes xyz for all intersections and returns it.
    This allows grad_xyz (from hash/MLP backward) to flow into this function's backward,
    where we compute dL_duv and pass it to the CUDA kernel - matching cat mode's gradient path.

    In 3D/3D_direct mode, the blending weight is: weight = alpha * T = opacity * G * T
    where G is the kernel value, T is transmittance, and alpha = opacity * G.

    IMPORTANT: This function must be called ONCE for ALL intersections (not per-batch)
    to correctly compute the transmittance chain gradient effect.

    The gradient computation uses the unified backward_from_weight_grad kernel that:
    1. Computes transmittance chain gradient: dL_dalpha = (dL_dweight - last_dL_dT) * T
    2. Computes dL_dopacity = G * dL_dalpha
    3. Computes dL_dtransMat via ray-disk intersection backprop with dL_duv from hash/xyz path
    4. Computes dL_dmean2D for position gradients (screen-space position feedback)
    5. NEW: Uses transmat_to_scale_rot_grad CUDA kernel for proper coordinate conversion

    Forward: Returns (weight_values, xyz, scale, rotation_quaternions, screenspace_points)
    Backward: Computes dL/dopacity, dL/dscale, dL/drotation, dL/dmean3D, and dL/dmean2D

    Args:
        opacity: [N] per-Gaussian opacity values
        scale: [N, 2] per-Gaussian scale values
        rotation_quaternions: [N, 4] quaternion rotation parameters
        rotation_matrices: [N, 3, 3] rotation matrices for xyz computation
        screenspace_points: [N, 3] screen-space points (requires_grad=True) for position gradients
        means3D: [N, 3] Gaussian centers
        projmatrix: [4, 4] projection matrix for proper coordinate conversion
        geomBuffer: Raw geometry buffer from CUDA forward (contains transMat)
        weight_values: [M] blending weights (alpha * T) from CUDA
        T_values: [M] transmittance values from CUDA buffer
        G_values: [M] kernel values from CUDA buffer
        alpha_values: [M] alpha values from CUDA buffer
        s_x: [M] intersection s.x coordinates
        s_y: [M] intersection s.y coordinates
        rho_flag: [M] 1.0=disk intersection, 0.0=center intersection
        gaussian_ids: [M] Gaussian indices for each intersection
        pixel_ids: [M] pixel indices for each intersection (must be sorted!)
        W, H: image dimensions
    """
    @staticmethod
    def forward(ctx, opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weight_values,
                T_values, G_values, alpha_values, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H):
        # Compute xyz for ALL intersections (so grad_xyz flows into backward)
        # Using the gathered parameters for the intersections
        b_scales = scale[gaussian_ids]  # [M, 2]
        b_R = rotation_matrices[gaussian_ids]  # [M, 3, 3]
        b_means = means3D[gaussian_ids]  # [M, 3]

        # Disk intersection formula: xyz = s_x * scale_x * R[:,0] + s_y * scale_y * R[:,1] + mean
        xyz_disk = (s_x[:, None] * b_scales[:, 0:1] * b_R[:, :, 0] +
                    s_y[:, None] * b_scales[:, 1:2] * b_R[:, :, 1] +
                    b_means)  # [M, 3]

        # Blend based on rho_flag: disk intersection vs Gaussian center
        xyz = rho_flag[:, None] * xyz_disk + (1.0 - rho_flag[:, None]) * b_means  # [M, 3]

        # Save for backward - note geomBuffer is not a tensor we save directly
        ctx.save_for_backward(opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
                              projmatrix, viewmatrix, gaussian_ids, pixel_ids, T_values, G_values, alpha_values,
                              s_x, s_y, rho_flag)
        ctx.geomBuffer = geomBuffer  # Store as attribute (raw buffer)
        ctx.n_gaussians = opacity.shape[0]
        ctx.W = W
        ctx.H = H

        # Return weights, xyz, scale, rotation_quaternions, and screenspace_points
        # xyz is returned so that grad_xyz flows back here from the hash/MLP chain
        return weight_values, xyz, scale, rotation_quaternions, screenspace_points

    @staticmethod
    def backward(ctx, grad_weight, grad_xyz, grad_scale_passthrough, grad_rotation_passthrough, grad_screenspace_passthrough):
        opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D, \
            projmatrix, viewmatrix, gaussian_ids, pixel_ids, T_values, G_values, alpha_values, \
            s_x, s_y, rho_flag = ctx.saved_tensors

        M = grad_weight.shape[0]
        N = ctx.n_gaussians
        W, H = ctx.W, ctx.H
        device = grad_weight.device
        dtype = grad_weight.dtype

        if M == 0:
            grad_opacity = torch.zeros(N, device=device, dtype=dtype)
            grad_scale = torch.zeros(N, 2, device=device, dtype=dtype)
            grad_rotation = torch.zeros(N, 4, device=device, dtype=dtype)  # quaternion gradients
            grad_screenspace = torch.zeros(N, 3, device=device, dtype=dtype)
            grad_means3D = torch.zeros(N, 3, device=device, dtype=dtype)
            if grad_scale_passthrough is not None:
                grad_scale = grad_scale + grad_scale_passthrough
            if grad_rotation_passthrough is not None:
                grad_rotation = grad_rotation + grad_rotation_passthrough
            if grad_screenspace_passthrough is not None:
                grad_screenspace = grad_screenspace + grad_screenspace_passthrough
            return grad_opacity, grad_scale, grad_rotation, None, grad_screenspace, grad_means3D, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        # =======================================================================
        # Compute dL_duv from grad_xyz (xyz gradient contribution to ray-disk Jacobian)
        #
        # This matches CAT mode backward.cu lines 1240-1241 where dL_duv is added
        # to dL_ds BEFORE the ray-disk Jacobian is applied:
        #   dL_ds.x += dL_duv.x;
        #   dL_ds.y += dL_duv.y;
        #
        # The chain rule for xyz = s_x * scale_x * R[:,0] + s_y * scale_y * R[:,1] + mean:
        #   dL/d(s_x) = dot(dL/d(xyz), scale_x * R[:,0]) = dot(dL_dxyz, SuTu)
        #   dL/d(s_y) = dot(dL/d(xyz), scale_y * R[:,1]) = dot(dL_dxyz, SvTv)
        #
        # This is CRITICAL for matching CAT mode gradients - dL_duv flows through
        # the ray-disk Jacobian to contribute to transMat gradients.
        # =======================================================================
        if grad_xyz is not None:
            # Gather per-intersection parameters
            b_scales = scale[gaussian_ids]  # [M, 2]
            b_R = rotation_matrices[gaussian_ids]  # [M, 3, 3]

            # Only disk intersections contribute (center intersections have s ≈ 0)
            grad_xyz_masked = grad_xyz * rho_flag[:, None]  # [M, 3]

            # SuTu = scale_x * R[:,0], SvTv = scale_y * R[:,1]
            SuTu = b_scales[:, 0:1] * b_R[:, :, 0]  # [M, 3]
            SvTv = b_scales[:, 1:2] * b_R[:, :, 1]  # [M, 3]

            # dL_duv = dot(dL_dxyz, SuTu/SvTv)
            dL_duv_x = (grad_xyz_masked * SuTu).sum(dim=1)  # [M]
            dL_duv_y = (grad_xyz_masked * SvTv).sum(dim=1)  # [M]
        else:
            dL_duv_x = torch.zeros(M, device=device, dtype=dtype)
            dL_duv_y = torch.zeros(M, device=device, dtype=dtype)

        # Compute pixel boundaries for CUDA kernel
        pixel_change = torch.zeros(M + 1, dtype=torch.bool, device=device)
        pixel_change[0] = True
        pixel_change[-1] = True
        if M > 1:
            pixel_change[1:-1] = pixel_ids[1:] != pixel_ids[:-1]
        pixel_starts = torch.nonzero(pixel_change, as_tuple=True)[0].int()

        # Call unified CUDA kernel that reads transMat from geomBuffer
        # Now includes dL_duv from hash/xyz gradient path!
        # Returns: (dL_dopacity [N], dL_dtransMat [N, 9], dL_dmean2D [N, 2])
        if FP16_RASTERIZER_AVAILABLE and _fp16_rasterizer is not None:
            from diff_surfel_3D_16 import backward_from_weight_grad, transmat_to_scale_rot_grad
        elif LEAN_RASTERIZER_AVAILABLE and _main_rasterizer is None:
            from diff_surfel_3D import backward_from_weight_grad, transmat_to_scale_rot_grad
        else:
            from diff_surfel_rasterization import backward_from_weight_grad, transmat_to_scale_rot_grad

        # Compute per-intersection opacity for the kernel
        opacity_per_int = opacity[gaussian_ids]  # [M]

        grad_opacity, dL_dtransMat, dL_dmean2D = backward_from_weight_grad(
            ctx.geomBuffer, N,
            grad_weight.contiguous(),
            gaussian_ids.int().contiguous(),
            pixel_ids.int().contiguous(),
            pixel_starts.contiguous(),
            T_values.contiguous(),
            G_values.contiguous(),
            alpha_values.contiguous(),
            opacity_per_int.contiguous(),
            s_x.contiguous(),
            s_y.contiguous(),
            rho_flag.contiguous(),
            dL_duv_x.contiguous(),  # dL_duv from hash/xyz path
            dL_duv_y.contiguous(),  # dL_duv from hash/xyz path
            W, H
        )

        # =======================================================================
        # PART 1: INDIRECT gradient (via ray-disk Jacobian from CUDA kernel)
        # This converts screen-space dL_dtransMat to world-space scale/rotation gradients
        # using the projection matrix transformation: dL_dM = P * transpose(dL_dT) + dL_dhomoMat
        #
        # CRITICAL: We also include the xyz gradient contribution (dL_dhomoMat) here,
        # matching CAT mode's backward.cu lines 1186-1197 and 1428-1430.
        # This is the missing piece that caused gradient direction issues!
        # =======================================================================

        # Compute dL_dhomoMat from per-intersection grad_xyz and s values
        # Layout: [col0.xyz, col1.xyz, col2.xyz] where:
        #   col0 = sum over intersections of (dL_dxyz * s_x) - for scale_x direction
        #   col1 = sum over intersections of (dL_dxyz * s_y) - for scale_y direction
        #   col2 = sum over intersections of (dL_dxyz) - for mean position
        dL_dhomoMat = torch.zeros(N, 9, device=device, dtype=dtype)
        if grad_xyz is not None:
            # For col0 and col1: only disk intersections contribute (rho_flag=1)
            # because center intersections have s_x ≈ 0, s_y ≈ 0
            grad_xyz_disk = grad_xyz * rho_flag[:, None]  # [M, 3]

            # Column 0: += grad_xyz * s_x (for scale_x direction) - disk only
            dL_dhomoMat[:, 0:3].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz_disk * s_x[:, None]
            )
            # Column 1: += grad_xyz * s_y (for scale_y direction) - disk only
            dL_dhomoMat[:, 3:6].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz_disk * s_y[:, None]
            )
            # Column 2: += grad_xyz (for mean position) - ALL intersections!
            # Center intersections have xyz = mean, so they contribute to mean gradient
            dL_dhomoMat[:, 6:9].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz  # No masking for mean gradient!
            )

        # Extract transMat from geomBuffer for the t_vec formula (needed for dL_dmean2D contribution)
        if FP16_RASTERIZER_AVAILABLE and _fp16_rasterizer is not None:
            from diff_surfel_3D_16 import get_transmat_from_geombuffer
        elif LEAN_RASTERIZER_AVAILABLE and _main_rasterizer is None:
            from diff_surfel_3D import get_transmat_from_geombuffer
        else:
            from diff_surfel_rasterization import get_transmat_from_geombuffer
        transMat_precomp = get_transmat_from_geombuffer(ctx.geomBuffer, N)

        # dL_dnormal3D: Normal gradient from depth/normal loss
        # In 3D_direct mode, this is typically zero since normal loss gradients flow through native backward
        # We pass empty tensor for now; can be extended to capture normal gradients if needed
        dL_dnormal3D = torch.empty(0, device=device, dtype=dtype)

        grad_scale_indirect, grad_rotation_indirect, grad_means3D_indirect = transmat_to_scale_rot_grad(
            dL_dtransMat.contiguous(),
            dL_dhomoMat.contiguous(),
            dL_dmean2D.contiguous(),       # 2D mean gradient from backward_from_weight_grad
            dL_dnormal3D,                  # Normal gradient (empty for now)
            means3D.contiguous(),          # World-space positions
            transMat_precomp.contiguous(), # Forward pass transMat for t_vec formula
            scale.contiguous(),
            rotation_quaternions.contiguous(),
            projmatrix.contiguous(),
            viewmatrix.contiguous(),       # View matrix for normal gradient transform
            W, H  # Image dimensions for ndc2pix transformation
        )

        # =======================================================================
        # GRADIENT SCALING FOR 3D MODE
        #
        # With dL_dhomoMat now included in the CUDA kernel, the INDIRECT path
        # computes both:
        #   1. Kernel shape gradient contribution: P * transpose(dL_dT)
        #   2. XYZ gradient contribution: dL_dhomoMat (what DIRECT path computed before)
        #
        # This matches CAT mode's backward.cu where:
        #   dL_dM = P * transpose(dL_dT) + dL_dhomoMat  (line 1428-1430)
        #
        # IMPORTANT: The 3D MLP (xyz -> features) produces larger gradients than
        # CAT mode's 2D hash (uv -> features) because:
        #   1. 3D coordinates have larger spatial extent than 2D screen coords
        #   2. The gradient chains differently through 3D vs 2D networks
        #
        # Based on empirical comparison with CAT mode:
        #   - Scale gradients are ~3-5x larger
        #   - Rotation gradients are ~5-8x larger
        # We use a single scaling factor to bring them in line.
        # =======================================================================

        # NO SCALING FACTORS - gradients should be correct as computed
        # The test shows cosine similarity ~0.99 with autograd
        # If training is unstable, adjust learning rate instead
        grad_scale = grad_scale_indirect

        # =======================================================================
        # Rotation gradients: CUDA kernel returns dL/d(normalized_q)
        # We return this directly - PyTorch will chain through F.normalize
        # (which creates rotation_quaternions from _rotation) automatically.
        # DO NOT apply F.normalize chain rule here - that would double-apply it!
        # =======================================================================
        grad_rotation = grad_rotation_indirect

        # Convert dL_dmean2D [N, 2] to dL_dscreenspace [N, 3] (third component is 0)
        # This provides the screen-space position gradient that tells Gaussians where to move
        grad_screenspace = torch.zeros(N, 3, device=device, dtype=dtype)
        grad_screenspace[:, :2] = dL_dmean2D  # [N, 2] -> [N, 3] with z=0

        # Mean position gradients: from CUDA kernel's dL_dhomoMat column 2 (already computed!)
        # This is sum over intersections of grad_xyz, which equals d(xyz)/d(mean)
        grad_means3D = grad_means3D_indirect

        # Passthrough gradients (should be None/zero in normal operation)
        if grad_scale_passthrough is not None:
            grad_scale = grad_scale + grad_scale_passthrough
        if grad_rotation_passthrough is not None:
            grad_rotation = grad_rotation + grad_rotation_passthrough
        if grad_screenspace_passthrough is not None:
            grad_screenspace = grad_screenspace + grad_screenspace_passthrough

        # Return order: opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
        #               projmatrix, viewmatrix, geomBuffer, weight_values, T_values, G_values, alpha_values,
        #               s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
        return grad_opacity, grad_scale, grad_rotation, None, grad_screenspace, grad_means3D, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def _quat_to_rotmat_vjp_batch(quats, dL_dR0, dL_dR1):
    """
    Convert rotation matrix column gradients to quaternion gradients.
    This is a batch-friendly Python implementation of quat_to_rotmat_vjp.

    IMPORTANT: Unlike the CUDA implementation, this CORRECTLY chains through the
    normalization gradient. The CUDA quat_to_rotmat_vjp computes gradients w.r.t.
    the normalized quaternion but doesn't chain back to the raw quaternion.
    This is technically incorrect but native cat mode works because the INDIRECT
    path (through CUDA) dominates and the gradients are approximately correct.

    For the DIRECT path in 3D_direct mode, we need correct gradients, so we
    include the normalization chain rule here.

    Args:
        quats: [N, 4] quaternions (w, x, y, z ordering as stored in the model)
        dL_dR0: [N, 3] gradient for first column of rotation matrix
        dL_dR1: [N, 3] gradient for second column of rotation matrix

    Returns:
        dL_dquat: [N, 4] quaternion gradients (w.r.t. raw/unnormalized quaternions)
    """
    # Normalize quaternions - same as build_rotation and CUDA quat_to_rotmat
    norm = torch.sqrt((quats ** 2).sum(dim=-1, keepdim=True))
    q = quats / (norm + 1e-8)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build full dL_dR from column gradients (column 2 is zeros)
    # v_R is column-major: v_R[col][row]
    # v_R[0] = dL_dR0, v_R[1] = dL_dR1, v_R[2] = 0
    v_R00, v_R01, v_R02 = dL_dR0[:, 0], dL_dR1[:, 0], torch.zeros_like(w)
    v_R10, v_R11, v_R12 = dL_dR0[:, 1], dL_dR1[:, 1], torch.zeros_like(w)
    v_R20, v_R21, v_R22 = dL_dR0[:, 2], dL_dR1[:, 2], torch.zeros_like(w)

    # quat_to_rotmat_vjp formulas from auxiliary.h
    # These compute dL/d(normalized_q), i.e., gradients w.r.t. normalized quaternion
    # In CUDA, v_R[col][row], so v_R[1][2] maps to Python v_R21, v_R[2][1] to v_R12, etc.

    # v_quat.x (w gradient) - CUDA: x*(v_R[1][2]-v_R[2][1]) + y*(v_R[2][0]-v_R[0][2]) + z*(v_R[0][1]-v_R[1][0])
    v_w = 2.0 * (x * (v_R21 - v_R12) + y * (v_R02 - v_R20) + z * (v_R10 - v_R01))

    # v_quat.y (x gradient) - CUDA: -2x*(v_R[1][1]+v_R[2][2]) + y*(v_R[0][1]+v_R[1][0]) + z*(v_R[0][2]+v_R[2][0]) + w*(v_R[1][2]-v_R[2][1])
    v_x = 2.0 * (-2.0 * x * (v_R11 + v_R22) + y * (v_R10 + v_R01) + z * (v_R20 + v_R02) + w * (v_R21 - v_R12))

    # v_quat.z (y gradient) - CUDA: x*(v_R[0][1]+v_R[1][0]) - 2y*(v_R[0][0]+v_R[2][2]) + z*(v_R[1][2]+v_R[2][1]) + w*(v_R[2][0]-v_R[0][2])
    v_y = 2.0 * (x * (v_R10 + v_R01) - 2.0 * y * (v_R00 + v_R22) + z * (v_R21 + v_R12) + w * (v_R02 - v_R20))

    # v_quat.w (z gradient) - CUDA: x*(v_R[0][2]+v_R[2][0]) + y*(v_R[1][2]+v_R[2][1]) - 2z*(v_R[0][0]+v_R[1][1]) + w*(v_R[0][1]-v_R[1][0])
    v_z = 2.0 * (x * (v_R20 + v_R02) + y * (v_R21 + v_R12) - 2.0 * z * (v_R00 + v_R11) + w * (v_R10 - v_R01))

    # dL/d(normalized_q) - gradients w.r.t. unit quaternion
    dL_dq = torch.stack([v_w, v_x, v_y, v_z], dim=-1)

    # Chain through the normalization: q = quats / ||quats||
    # Jacobian: dq/d(quats) = (I - q q^T) / ||quats||
    # dL/d(quats) = (dL/dq - q * (dL/dq · q)) / ||quats||
    q_dot_grad = (q * dL_dq).sum(dim=-1, keepdim=True)
    dL_dquats = (dL_dq - q * q_dot_grad) / (norm + 1e-8)

    return dL_dquats


class RenderCache:
    """Cache for pre-allocated tensors to avoid repeated allocations during inference."""

    def __init__(self):
        self.screenspace_points = None
        self.rotation_matrices = None
        self.homotrans = None
        self.shape_dims = {}  # Cache shape_dims tensors by key
        self._num_gaussians = 0

    def get_screenspace_points(self, num_gaussians, device='cuda'):
        if self.screenspace_points is None or self.screenspace_points.shape[0] != num_gaussians:
            self.screenspace_points = torch.zeros(num_gaussians, 4, dtype=torch.float32, device=device)
            self._num_gaussians = num_gaussians
        return self.screenspace_points

    def get_shape_dims(self, key, values, device='cuda'):
        """Get or create shape_dims tensor. Key is tuple of (gs_dim, hs_dim, os_dim)."""
        if key not in self.shape_dims:
            self.shape_dims[key] = torch.tensor(values, dtype=torch.int32, device=device)
        return self.shape_dims[key]

    def cache_rotation_matrices(self, rotations_quat):
        """Pre-compute and cache rotation matrices from quaternions."""
        from utils.general_utils import build_rotation
        self.rotation_matrices = build_rotation(rotations_quat)
        return self.rotation_matrices

    def cache_homotrans(self, rotations_quat, scales, xyzs):
        """Pre-compute and cache homogeneous transform matrices from quaternions."""
        from utils.general_utils import build_rotation, build_H
        rots = build_rotation(rotations_quat)
        self.homotrans = build_H(rots, scales, xyzs)
        return self.homotrans


def set_default_activation_bias(sh_bias, res_bias):
    """Set the default activation biases used for decompose restore."""
    global _ACTIVATION_BIAS
    _ACTIVATION_BIAS = [sh_bias, res_bias]


def _render_gestex_joint(viewpoint_camera, pc, bg_color, lru_slope=0.01, decompose_mode=None):
    """`--method GEStex` full-hardening SORT-FREE 2-pass render:
      Pass 1 (joint_s): textured surfels as a frontmost z-buffer, C_S = LRU(SV + atlas(uv)),
                        + depth threshold D_S = minDepth + mod_depth.
      Pass 2 (joint_g): untextured 3D Gaussians (SH), additive with `depth >= D_S` discard,
                        -> C_G, W_G, per-Gauss max_contrib.
      Composite: LRU((C_S*s_w + C_G)/(s_w + W_G)).
    Surfel geometry+opacity are frozen (no geom VJP in joint_s); atlas + surfel SV + Gaussians
    train. Atlas grad flows via a subset device-global buffer; train.py scatters it back to
    pc._tex_atlas.grad[surfel_mask]."""
    import diff_surfel_gestex_joint_s as _js
    import diff_surfel_gestex_joint_g as _jg
    dev = pc.get_xyz.device
    sm = pc._is_textured           # surfel (textured) mask
    gm = ~sm                       # gaussian (untextured) mask
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    wvt = viewpoint_camera.world_view_transform
    fpt = viewpoint_camera.full_proj_transform
    campos = viewpoint_camera.camera_center
    deg = pc.active_sh_degree

    means = pc.get_xyz
    opac = pc.get_opacity          # sigmoid (N,1)
    scal = pc.get_scaling          # (N,2) for the 2DGS surfel model
    rot = pc.get_rotation

    # Decomposition for diagnostics (mirrors the 3D_SH_res sh_only/tex_only split):
    #   'sh_only'  -> surfel color = SV base only          (atlas residual zeroed)
    #   'tex_only' -> surfel color = atlas residual only   (SV base zeroed)
    # Affects ONLY the textured surfel pass (C_S); the untextured Gaussian pass (C_G)
    # is unchanged so the tex/untex split stays visible. Return dict also exposes the
    # separate C_S / C_G / W_G so callers can inspect each component directly.
    _zero_atlas = (decompose_mode == 'sh_only')
    _zero_sv    = (decompose_mode in ('tex_only', 'tex_only_raw'))

    # ---- per-primitive view-dependent SV colour ----
    # BOTH the surfels (colors_precomp for joint_s) and the untextured Gaussians (fake-SH
    # for joint_g) must use the SV colour, else the Gaussians' _sv_* params get NO gradient
    # (previously joint_g used raw get_features → SV never optimized → Gaussians learned no
    # colour). We build the fake-SH once and index it per set.
    dirs = means - campos.unsqueeze(0)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    fm = getattr(pc, 'feature_mode', 'sh')
    _g_shs = None
    if fm == 'SV' and getattr(pc, '_sv_sites', torch.empty(0)).numel() > 0:
        fake_shs, sv_rgb = _build_fake_shs_from_SV(pc, dirs, sv_lru=float(getattr(ingp, 'sv_lru_slope', 0.0)))
        # sv_rgb is ALREADY relu(feat + 0.5) (post-bias, post-clamp) — render it directly.
        # The previous `+ _ACTIVATION_BIAS[0]` double-biased it → a >=0.5 brightness floor
        # (washed-out surfels) and a colour jump vs the harden-phase base. Dropped.
        surfel_colors = sv_rgb[sm].contiguous()
        _g_shs = fake_shs[gm].contiguous()   # Gaussians: SV via fake-SH so _sv_* gets gradient
    else:
        shs_view = pc.get_features.transpose(1, 2).view(means.shape[0], 3, -1)
        surfel_colors = torch.clamp_min(eval_sh(deg, shs_view, dirs) + _ACTIVATION_BIAS[0], 0.0)[sm].contiguous()

    if _zero_sv:
        surfel_colors = torch.zeros_like(surfel_colors)   # tex_only: atlas residual alone

    # ---- Pass 1: joint_s (surfel z-buffer + atlas) ----
    # TS+ rising opacity floor: surfel opacity = O_t + (1-O_t)*get_opacity, <= 1 (no
    # ∝opacity>1 amplification). O_t == 0 (default/pre-harden) => opacity unchanged.
    _floor = float(getattr(pc, 'ges_opac_floor', 0.0))
    _opac_s = (_floor + (1.0 - _floor) * opac[sm]) if _floor > 0.0 else opac[sm]
    s_scales = scal[sm][:, :2].contiguous()
    mod_depth = (5.0 * s_scales.mean(-1, keepdim=True)).contiguous()
    atlas_s = pc._tex_atlas[sm].contiguous()
    if _zero_atlas:
        atlas_s = torch.zeros_like(atlas_s)               # sh_only: SV base alone
    atlas_grad = torch.zeros_like(atlas_s)
    _js.set_gestex_atlas(atlas_s, atlas_grad, int(pc.ges_atlas_res), 4.0)
    pc._ges_atlas_grad = atlas_grad          # train.py scatters -> _tex_atlas.grad[sm]
    pc._ges_surfel_mask = sm.clone()
    st_s = _js.GaussianRasterizationSettings(H, W, tanfovx, tanfovy, bg_color, 1.0,
                                             wvt, fpt, deg, campos, False, False)
    C_S, radii_s, others_s = _js.GaussianRasterizer(st_s)(
        means3D=means[sm].contiguous(), opacities=_opac_s.contiguous(),
        colors_precomp=surfel_colors, mod_depth=mod_depth,
        scales=s_scales, rotations=rot[sm].contiguous())
    D_S = others_s[1:2].contiguous()         # [1,H,W] depth threshold

    # ---- Pass 2: joint_g (additive Gaussians, SH, depth-tested vs D_S) ----
    ng = int(gm.sum().item())
    # full-size screenspace so standard densification reads viewspace_points.grad
    screenspace = torch.zeros((means.shape[0], 3), dtype=means.dtype, device=dev, requires_grad=True) + 0
    try: screenspace.retain_grad()
    except Exception: pass
    maxc = torch.zeros((means.shape[0], 1), dtype=means.dtype, device=dev, requires_grad=True) + 0
    try: maxc.retain_grad()
    except Exception: pass
    radii_g = None
    if ng > 0:
        sz = pc.get_scaling_z[gm] if getattr(pc, '_scaling_z', torch.empty(0)).numel() > 0 else scal[gm][:, :1]
        g_scales = torch.cat([scal[gm][:, :2], sz], dim=1).contiguous()
        st_g = _jg.GaussianRasterizationSettings(H, W, tanfovx, tanfovy, bg_color, 1.0,
                                                 wvt, fpt, deg, campos, False, False)
        C_G, radii_g, W_G = _jg.GaussianRasterizer(st_g)(
            means3D=means[gm].contiguous(), means2D=screenspace[gm], max_contrib_ret=maxc[gm],
            opacities=opac[gm].contiguous(), depth_map=D_S,
            shs=(_g_shs if _g_shs is not None else pc.get_features[gm].contiguous()),
            scales=g_scales, rotations=rot[gm].contiguous())
    else:
        C_G = torch.zeros_like(C_S)
        W_G = torch.zeros((1, H, W), device=dev)

    # ---- sort-free composite ----
    # LeakyReLU is applied ONCE, AFTER compositing the sort-free 3DGS onto the surfels
    # (single post-composite site) — NOT per-pass. So negative C_S / C_G survive into the
    # blend and only the combined result is rectified.
    s_w = float(getattr(pc, 'ges_s_weight', 1.0))
    final = (C_S * s_w + C_G) / (s_w + W_G + 1e-8)
    final = torch.nn.functional.leaky_relu(final, lru_slope)

    N = means.shape[0]
    radii = torch.zeros(N, device=dev, dtype=torch.int32)
    if radii_s is not None: radii[sm] = radii_s.to(torch.int32)
    if radii_g is not None: radii[gm] = radii_g.to(torch.int32)
    pc._ges_maxc = maxc                        # per-Gauss max contribution (for prune)

    # Aux maps for the training loop. rend_alpha = foreground coverage (opaque surfel
    # frontmost, or Gaussian weight) — used by --random_background. Surfels are opaque
    # z-buffer discs so their coverage is ~binary; W_G adds Gaussian-only pixels.
    surfel_cov = (others_s[2:3] >= 0).float()
    rend_alpha = torch.clamp(surfel_cov + W_G, 0.0, 1.0)
    surf_depth = others_s[0:1]
    # --- Auxiliary maps for training_output / eval viz ---
    # rend_normal: frontmost surfel view-space normal (joint_s out_others[5:8]), rotated
    # view->world to match the res_switch convention (allmap @ W2V[:3,:3].T). Falls back to
    # zeros if joint_s wasn't rebuilt with the 8-channel out_others.
    if others_s.shape[0] >= 8:
        _vn = others_s[5:8]                                   # [3,H,W] view-space surfel normal
        rend_normal = (_vn.permute(1, 2, 0) @ wvt[:3, :3].T).permute(2, 0, 1).contiguous()
    else:
        rend_normal = torch.zeros(3, H, W, device=dev)
    # surf_normal: geometric normal from the rendered (frontmost-surfel) depth — the
    # normal-consistency target; comparing it to rend_normal reveals surfel misalignment.
    try:
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1).contiguous()
    except Exception:
        surf_normal = torch.zeros(3, H, W, device=dev)
    return {
        "render": final,
        "surfel_render": C_S,                # C_S: textured-surfel pass (SV + atlas), post-LRU
        "gaussian_render": C_G,              # C_G: untextured 3D-Gaussian additive pass, post-LRU
        "gaussian_weight": W_G,              # W_G: [1,H,W] summed Gaussian alpha (untex coverage)
        "surfel_coverage": (others_s[2:3] >= 0).float(),  # [1,H,W] frontmost-surfel hit mask
        "viewspace_points": screenspace,
        "visibility_filter": radii > 0,
        "radii": radii,
        "max_contrib_idx": None,
        "rend_alpha": rend_alpha,
        "rend_normal": rend_normal,
        "surf_normal": surf_normal,
        "surf_depth": surf_depth,
        # Opaque z-buffer surfels concentrate to one depth → ~zero distortion by design.
        "rend_dist": torch.zeros(1, H, W, device=dev),
        # Depth maps for eval/save (render_final_images). GEStex's coarse depth is the
        # frontmost-surfel view-space depth; median/max-contrib reuse it (opaque discs).
        "depth_expected": surf_depth,
        "depth_median": surf_depth,
        "depth_max_contributor": surf_depth,
        "gaussian_num": (W_G > 0.01).float(),   # untex-Gaussian coverage (final-save heatmap)
    }


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ingp = None,
    beta = 0, iteration = None, cfg = None, record_transmittance = False, use_xyz_mode = False, decompose_mode = None, max_intersections = 0,
    skip_mlp = False, force_no_hash_cuda = False, temperature = 1.0, force_ratio = 0.2, no_gumbel = False, dropout_lambda = 0.0, is_training = True,
    aabb_mode = "2dgs", aa = 0.0, aa_threshold = 0.01, skybox = None, background_mode = "none", bg_hashgrid = None, detach_hash_grad = False,
    return_raw_features = False, fast_inference = False, cache = None, max_intersections_per_pixel = 32, lowpass = False, pixel_center = False, antialiasing = 0.0, sv_metric = "l2",
    metric_map = None, pose_correction = None, deform = None, override_opacity = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    decompose_mode: None (normal), 'gaussian_only' (zero hashgrid features), 'ngp_only' (zero per-Gaussian features)
                   Only used in cat mode for visualization/debugging.
    fast_inference: If True, skip expensive post-processing (normal computation, etc.) for faster inference.
    cache: Optional RenderCache object to reuse pre-allocated tensors and cached computations.
    max_intersections: Max ray-Gaussian intersections per pixel. 0 means no limit.
    skip_mlp: If True, skip MLP decode and return zeros for RGB (for benchmarking).
    force_no_hash_cuda: If True, disable hash_in_CUDA (use plain 2DGS rasterizer, no hash query).
    dropout_lambda: Hash dropout rate for cat_dropout mode (0.2 = 20% of Gaussians don't query hash during training).
    is_training: Whether we're in training mode (affects dropout behavior).
    """

    render_start_time = time.time()

    XYZ_TYPE = cfg.ingp_stage.XYZ_TYPE
    assert(XYZ_TYPE == "UV" or XYZ_TYPE == "DEPTH")

    # Create zero tensor for screen-space points
    if fast_inference and cache is not None:
        # Use cached pre-allocated tensor (no gradients needed for inference)
        screenspace_points = cache.get_screenspace_points(pc.get_xyz.shape[0])
        screenspace_points.zero_()  # Reset to zeros
    else:
        # Training mode: need gradients
        screenspace_points = torch.zeros(pc.get_xyz.shape[0], 4, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    # --deform: per-surfel canonical->time-t deformation (Δposition), applied
    # BEFORE the --3rgs rigid correction below and before the CUDA hash query
    # (which is reconstructed from these means). deform = (d_xyz [N,3], d_rot
    # [N,4]); None ⇒ no deformation. Grad flows to the per-surfel latent + the
    # deform MLP via dL/dmeans3D.
    if deform is not None:
        means3D = means3D + deform[0]
    # --3rgs camera pose refinement: apply the per-camera differentiable rigid
    # transform M = C2W·inv(Td)·W2V to the Gaussian centers (and, below, to the
    # surfel rotations). Rendering the transformed cloud through the ORIGINAL
    # camera matrices == rendering the un-transformed cloud through the
    # delta-corrected camera, so the pose gradient flows to the per-camera delta
    # via dL/dmeans3D (the kernel has no viewmatrix grad). Injected here — before
    # the SV/beta/hash view-dir prep below — so the whole forward is consistent.
    # pose_correction = (M_rot [3,3], M_t [3], q_M [4]); None ⇒ identity (no-op).
    if pose_correction is not None:
        _pc_Mrot, _pc_Mt, _pc_qM = pose_correction
        means3D = means3D @ _pc_Mrot.transpose(0, 1) + _pc_Mt
    means2D = screenspace_points
    opacity = pc.get_opacity
    # Finetune-with-occluder-mesh hook: multiply by a per-Gauss [N,1] mask so
    # occluded Gaussians contribute nothing this frame (autograd zeroes their
    # per-view grad). Set to `pc.get_opacity * mask` from the caller; None ⇒
    # untouched, byte-identical to before.
    if override_opacity is not None:
        opacity = override_opacity

    # `--method GEStex`: TS+-style rising opacity FLOOR to harden surfels into near-opaque
    # flat discs while keeping opacity <= 1 (so the ∝opacity geometry-gradient amplification
    # that bloats surfels never appears). surfel_opac_eff = O_t + (1-O_t)*get_opacity, with
    # O_t = pc.ges_opac_floor ramped 0 -> ~0.99 over the harden window. The optimizer keeps
    # per-surfel freedom within [O_t, 1]. O_t == 0 (default) => opacity unchanged.
    #  - harden phase (10k-20k): ALL rows are surfels → floor everything.
    #  - joint stage (>= --ges_joint_iter): floor ONLY textured (surfel) rows; the spawned
    #    untextured 3D Gaussians keep their trained (differentiable) opacity.
    _floor = float(getattr(pc, 'ges_opac_floor', 0.0))
    if getattr(pc, 'is_gestex', False) and _floor > 0.0:
        _gj = (ingp is not None and getattr(ingp, 'is_gestex_joint', False))
        _surf_op = _floor + (1.0 - _floor) * opacity   # in [O_t, 1], <= 1
        if _gj and hasattr(pc, '_is_textured') and pc._is_textured.numel() == opacity.shape[0]:
            opacity = torch.where(pc._is_textured.view(-1, 1), _surf_op, opacity)
        else:
            opacity = _surf_op

    # `--method GEStex` SORT-FREE joint stage: intercept before the (aliased res_switch /
    # cascade) rasterizer path. Opt-in via ingp.is_gestex_sortfree so the verified cascade
    # path stays the default until this is validated end-to-end.
    if (ingp is not None and getattr(ingp, 'is_gestex_joint', False)
            and getattr(ingp, 'is_gestex_sortfree', False)
            and GESTEX_JOINT_S_AVAILABLE and GESTEX_JOINT_G_AVAILABLE
            and getattr(pc, '_tex_atlas', None) is not None and pc._tex_atlas.numel() > 0):
        _lru = float(getattr(ingp, 'lru_slope', 0.0)) or 0.01
        return _render_gestex_joint(viewpoint_camera, pc, bg_color, lru_slope=_lru,
                                    decompose_mode=decompose_mode)

    # --feature beta / --feature sg: compute fake SH tensor from the directional
    # lobes once per render. Stashed on pc._beta_fake_shs so dispatch blocks below
    # can read it back in place of pc.get_features.
    pc._beta_fake_shs = None
    _fm = getattr(pc, 'feature_mode', 'sh')
    if _fm == "beta" and pc._sb_params.numel() > 0:
        with torch.no_grad():
            _cam_center = viewpoint_camera.camera_center.to(means3D.device)
        _dirs = means3D - _cam_center.unsqueeze(0)
        _dirs = _dirs / (_dirs.norm(dim=-1, keepdim=True) + 1e-8)
        _sh_bias_local = _ACTIVATION_BIAS[0]
        pc._beta_fake_shs = _build_fake_shs_from_sb(pc, _dirs, pc.max_sh_degree,
                                                    sh_bias=_sh_bias_local)
    elif _fm == "sg" and pc._sg_directions.numel() > 0:
        with torch.no_grad():
            _cam_center = viewpoint_camera.camera_center.to(means3D.device)
        _dirs = means3D - _cam_center.unsqueeze(0)
        _dirs = _dirs / (_dirs.norm(dim=-1, keepdim=True) + 1e-8)
        _sh_bias_local = _ACTIVATION_BIAS[0]
        pc._beta_fake_shs = _build_fake_shs_from_sg(pc, _dirs, pc.max_sh_degree,
                                                     sh_bias=_sh_bias_local)
    elif _fm == "voronoi" and pc._sv_sites.numel() > 0:
        with torch.no_grad():
            _cam_center = viewpoint_camera.camera_center.to(means3D.device)
        _dirs = means3D - _cam_center.unsqueeze(0)
        _dirs = _dirs / (_dirs.norm(dim=-1, keepdim=True) + 1e-8)
        _sh_bias_local = _ACTIVATION_BIAS[0]
        pc._beta_fake_shs = _build_fake_shs_from_voronoi(pc, _dirs, pc.max_sh_degree,
                                                          sh_bias=_sh_bias_local,
                                                          metric=sv_metric)

    # --feature SV: reference-faithful Spherical Voronoi (sphericalvoronoi/radiance).
    # Wires SV into both rasterizer color paths so it works regardless of method:
    #   • SH path (3D_SH_res / 3D_SH_cat / baseline-with-SH / ...): bake SV into
    #     fake_shs DC slot via `_build_fake_shs_from_SV` so SH_eval(view) outputs
    #     sv_rgb. Real `_features_dc` is ignored. Visible in `sh_only` decompose.
    #   • Vanilla colors_precomp path (no ingp): set `override_color = sv_rgb`
    #     so the rasterizer uses RGB directly (no bias, no SH). This is the
    #     closest reference match.
    # The `_sites_mask` (if populated) is applied at eval only, mirroring the
    # reference's `_nn_sites[~mask] = 1e8`. NB: under 3D_SH_* the rasterizer
    # still adds sh_bias and hash MLP residual on top of SV — pass
    # `--activation_bias 0.0 0.0` for the cleanest match.
    if _fm == "SV" and pc._sv_sites.numel() > 0:
        with torch.no_grad():
            _cam_center = viewpoint_camera.camera_center.to(means3D.device)
        _dirs_sv = means3D - _cam_center.unsqueeze(0)
        _dirs_sv = _dirs_sv / (_dirs_sv.norm(dim=-1, keepdim=True) + 1e-8)
        # `--sv_lru α`: leaky INNER ReLU on the SV base (relu(feat+0.5)) for the main
        # 3D_SH_res-family render path (3D_SH_res, 3D_SH_filmres, mixed[_3d], …). α=0 ⇒
        # F.leaky_relu is exactly ReLU ⇒ byte-identical. getattr handles ingp=None → 0.0.
        pc._beta_fake_shs, _sv_rgb = _build_fake_shs_from_SV(
            pc, _dirs_sv, sv_lru=float(getattr(ingp, 'sv_lru_slope', 0.0)))
        # `--method res_3d` post-split: the 2D residual-carriers must contribute
        # ZERO SV. We already zeroed the SV/SH params on those rows at split
        # AND mask `colors_precomp` to 0 for them at the rasterizer call, but
        # `_build_fake_shs_from_SV` may apply biases / softmax-normalisation
        # that lift a 0-param row's output above 0 (e.g. an SV-DC bias). Zero
        # `_sv_rgb` and `pc._beta_fake_shs` for tex rows here so any code path
        # that uses these tensors (SH fallback, decompose dump, etc.) sees a
        # rigorous zero for the residual-carrier half.
        # IMPORTANT: gate on `(~_is_textured).any()` (split has fired = at
        # least one untex row exists), NOT on `.any()` — `_is_textured` is
        # initialised to all-True at startup, so `.any()` is trivially true
        # pre-split and would zero _sv_rgb / _beta_fake_shs for EVERY Gauss,
        # starving the forward of SV signal (and consequently the backward
        # of SV gradients) until the split fires at --res_3d_iter.
        # SKIP for `--method res_3d_paired` AND `--method res_3d_double`:
        # both keep SV on tex carriers (full per-Gauss capacity), so don't
        # zero anything.
        # `--method GEStex` ALSO keeps SV on its textured surfels (surfel colour = SV +
        # texture) — it is NOT a res_3d residual-carrier. Without this, GEStex's surfel SV
        # gets zeroed once untextured Gaussians exist (post-20k spawn), producing a gray
        # sh_only and a degraded joint render.
        _paired_keeps_sv = (ingp is not None
                             and (getattr(ingp, 'is_res_3d_paired_mode', False)
                                  or getattr(ingp, 'is_res_3d_double_mode', False)
                                  or getattr(ingp, 'is_gestex_mode', False)))
        _paired_keeps_sv = _paired_keeps_sv or bool(getattr(pc, 'is_gestex', False))
        if (not _paired_keeps_sv
                and hasattr(pc, '_is_textured') and pc._is_textured.numel() == _sv_rgb.shape[0]
                and bool((~pc._is_textured).any())):
            with torch.no_grad():
                _tex_mask = pc._is_textured
                _sv_rgb = _sv_rgb.clone()
                _sv_rgb[_tex_mask] = 0.0
                if pc._beta_fake_shs is not None:
                    pc._beta_fake_shs = pc._beta_fake_shs.clone()
                    pc._beta_fake_shs[_tex_mask] = 0.0
        override_color = _sv_rgb

    def _effective_shs():
        """Return directional-lobe-derived fake SH if --feature beta/sg, else real SH."""
        if pc._beta_fake_shs is not None:
            return pc._beta_fake_shs
        return pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        # --deform: per-surfel canonical->time-t rotation delta (quaternion
        # residual, add-then-renormalize; Deformable-3DGS). Applied before the
        # --3rgs rotation below. Grad flows to the latent + MLP via dL/drotations.
        # deform[1] is None for position-only deform (v1 canonical-hash path: with
        # the rotation un-deformed, the canonical hash reconstruction — canonical
        # center + deformed tangents — is exact, since tangents are unchanged).
        if deform is not None and deform[1] is not None:
            rotations = torch_F.normalize(rotations + deform[1], dim=-1)
        # --3rgs: rotate each surfel's orientation by the camera's pose-delta
        # rotation (q_M ⊗ q), so the splat normals move with the camera — needed
        # for the cloud-transform to be *exactly* equivalent to moving the camera
        # (means alone only handles the disk centers). Grad flows to the delta via
        # dL/drotations. q_M ≈ identity (deltas are tiny), so this is a no-op at
        # init. The compute_cov3D_python path above is left to means-only (it is
        # not on the default training path).
        if pose_correction is not None:
            rotations = _pose_quaternion_multiply(pose_correction[2], rotations)

    # --- clip_relight: per-Gauss deform + relight pre-pass (on top of 3D_SH_res) ---
    # Run the clip head over ALL surfels: override geometry/opacity and inject the
    # exposed SV as the per-Gauss SH base. The invariant texgrid (hash+MLP) is added
    # downstream in CUDA, untouched (clip plane = reveal + reshade, not a texture edit).
    # Requires --feature SV (override_color holds the clamped sv_rgb at this point).
    if (ingp is not None and getattr(ingp, 'is_clip_relight_mode', False)
            and getattr(ingp, 'clip_head', None) is not None
            and override_color is not None
            and scales is not None and rotations is not None):
        _clip_plane = getattr(viewpoint_camera, 'clip_plane', None)
        if _clip_plane is not None:
            _cr = ingp.clip_head(means3D, scales, rotations, opacity,
                                 override_color, _clip_plane.to(means3D.device))
            means3D = _cr.mu
            scales = _cr.scaling
            rotations = _cr.rotation
            opacity = _cr.opacity
            # Inject signed sv_exposed as the per-Gauss SH base. CUDA (mode 2) computes
            #   base = ReLU(SH_C0·DC + sh_bias); set DC = (sv_exposed - sh_bias)/SH_C0
            # (higher orders 0) so the pre-ReLU base == sv_exposed (Phase-1 clamped base),
            # then the invariant texgrid residual is added and the per-pixel ReLU deferred.
            _shb = float(_ACTIVATION_BIAS[0])
            _fake = torch.zeros_like(pc._beta_fake_shs)
            _fake[:, 0, :] = (_cr.sv_exposed - _shb) / _SH_C0
            pc._beta_fake_shs = _fake
            override_color = _cr.sv_exposed
        elif not getattr(render, '_clip_relight_warned', False):
            print("[CLIP_RELIGHT] WARNING: viewpoint_camera.clip_plane is None — "
                  "running as plain 3D_SH_res (no cull/relight). Is this a clip dataset?")
            render._clip_relight_warned = True

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    
    # Diffuse mode: use 0-degree SH (just DC component = diffuse RGB), no hashgrid
    is_diffuse_mode = ingp is not None and hasattr(ingp, 'is_diffuse_mode') and ingp.is_diffuse_mode
    # Specular mode: full 2DGS with SH (view-dependent), no hashgrid
    is_specular_mode = ingp is not None and hasattr(ingp, 'is_specular_mode') and ingp.is_specular_mode
    # Diffuse_ngp mode: diffuse SH + hashgrid on unprojected depth
    is_diffuse_ngp_mode = ingp is not None and hasattr(ingp, 'is_diffuse_ngp_mode') and ingp.is_diffuse_ngp_mode
    # Diffuse_offset mode: diffuse SH as xyz offset for hashgrid query
    is_diffuse_offset_mode = ingp is not None and hasattr(ingp, 'is_diffuse_offset_mode') and ingp.is_diffuse_offset_mode
    # 3D mode: intersection buffer output, SH blending in PyTorch
    is_3D_mode = ingp is not None and hasattr(ingp, 'is_3D_mode') and ingp.is_3D_mode
    # 3D_direct mode: intersection buffer output, direct RGB via MLP (like cat mode's 2D MLP)
    is_3D_direct_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_mode') and ingp.is_3D_direct_mode
    # 3D_direct_fused mode: fused in-kernel MLP (no intersection buffer, hash+MLP in CUDA)
    is_3D_direct_fused_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_fused_mode') and ingp.is_3D_direct_fused_mode
    # 3D_direct_lean mode: same as 3D_direct_fused but uses lean rasterizer library (faster builds)
    is_3D_direct_lean_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_lean_mode') and ingp.is_3D_direct_lean_mode
    # 3D_direct_fp16 mode: FP16 weights + FP16 GEMM shared memory (diff_surfel_3D_16)
    is_3D_direct_fp16_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_fp16_mode') and ingp.is_3D_direct_fp16_mode
    # 3D_direct_TC mode: Tensor Core WMMA for MLP (diff_surfel_3D_tc)
    is_3D_direct_tc_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_tc_mode') and ingp.is_3D_direct_tc_mode
    # 3D_SH_TC mode: TC WMMA MLP → 48D SH coefs, viewdir eval in kernel (diff_surfel_3D_sh)
    is_3D_direct_sh_tc_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_sh_tc_mode') and ingp.is_3D_direct_sh_tc_mode
    # 3D_SH_res mode: per-Gaussian SH + tiny hash MLP residual (diff_surfel_3D_sh_res)
    is_3D_SH_res_mode = ingp is not None and hasattr(ingp, 'is_3D_SH_res_mode') and ingp.is_3D_SH_res_mode
    # Default for the mixed[_3d]-only flags; only assigned for real inside the
    # 3D_SH_res rasterizer-dispatch branch below. Without this default, calling
    # render(ingp=None) (e.g. baseline auxiliary passes) hits an
    # UnboundLocalError at the `if _is_mixed_3d:` site near line ~1821.
    _is_mixed_3d = False
    _is_mixed = False
    # 3D_SH_cat mode: per-Gaussian SH + hash+DC MLP residual (diff_surfel_3D_sh_res)
    is_3D_SH_cat_mode = ingp is not None and hasattr(ingp, 'is_3D_SH_cat_mode') and ingp.is_3D_SH_cat_mode
    # 3D_SH_32 mode: per-Gaussian SH + 32-dim hash MLP residual (diff_surfel_3D_sh_32)
    is_3D_SH_32_mode = ingp is not None and hasattr(ingp, 'is_3D_SH_32_mode') and ingp.is_3D_SH_32_mode
    # 3D_SH_concat mode: 32-dim MLP, input = concat[surfel latent(16) | hash(16)] (diff_surfel_3D_sh_concat)
    is_3D_SH_concat_mode = ingp is not None and hasattr(ingp, 'is_3D_SH_concat_mode') and ingp.is_3D_SH_concat_mode
    # Treat lean/fp16/tc/sh_tc/sh_res/sh_cat/sh_32/sh_concat mode same as fused mode for rendering logic
    if is_3D_direct_lean_mode or is_3D_direct_fp16_mode or is_3D_direct_tc_mode or is_3D_direct_sh_tc_mode or is_3D_SH_res_mode or is_3D_SH_cat_mode or is_3D_SH_32_mode or is_3D_SH_concat_mode:
        is_3D_direct_fused_mode = True

    hash_in_CUDA = True
    try:
        if ingp is None:
            hash_in_CUDA = False
        if iteration < cfg.ingp_stage.switch_iter:
            hash_in_CUDA = False
        # Diffuse/Specular mode: never use hash_in_CUDA (no hashgrid)
        # Diffuse_ngp/diffuse_offset: also don't use hash_in_CUDA (we query hashgrid in Python on unprojected depth)
        # 3D/3D_direct mode: hash query happens in PyTorch after getting intersection buffer
        # 3D_direct_fused mode: hash query happens in CUDA kernel (hash_in_CUDA = True)
        if is_diffuse_mode or is_specular_mode or is_diffuse_ngp_mode or is_diffuse_offset_mode or is_3D_mode or is_3D_direct_mode:
            hash_in_CUDA = False
        # Force disable for benchmarking
        if force_no_hash_cuda:
            hash_in_CUDA = False
    except:
        pass


    # 3D/3D_direct mode: configure max_intersections_per_pixel (other setup done after hash_in_CUDA block)
    # 3D_direct_fused mode: no intersection buffer needed (fused in-kernel)
    if not is_3D_mode and not is_3D_direct_mode:
        max_intersections_per_pixel = 0  # Disabled for non-3D modes

    if ingp is not None and hash_in_CUDA == False and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode and not is_3D_mode and not is_3D_direct_mode and not is_3D_direct_fused_mode and _fm != "SV":
        ### warm-up
        override_color = ingp(points_3D = means3D, with_xyz = False).float()
        feat_dim = ingp.active_levels * ingp.level_dim
        # Set shape_dims for warmup phase (baseline mode, no hashgrid in CUDA)
        output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim
        shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
    
    if override_color is None:
        if pipe.convert_SHs_python:
            _feat_src = _effective_shs()
            shs_view = _feat_src.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(_feat_src.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = _effective_shs()
    else:
        colors_precomp = override_color

    homotrans = None
    features = None
    offsets = None
    gridrange = None 
    levels = base_resolution = interpolation = 0
    per_level_scale = 1
    align_corners = False
    ap_level = None
    contract = False

    # Cat mode detection and setup
    is_cat_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode

    # FiLM mode detection (per-Gauss gamma/beta modulate the hashgrid feature; cat-family)
    is_film_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_film_mode') and ingp.is_film_mode

    # Adaptive_zero mode detection (cat-like features + weighted hash, zeros when weight=0)
    is_adaptive_zero_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_zero_mode') and ingp.is_adaptive_zero_mode

    # Adaptive_gate mode detection (VQ-AD style gating: soft→STE→hard)
    is_adaptive_gate_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_gate_mode') and ingp.is_adaptive_gate_mode

    # Cat_dropout mode detection (cat mode with hash dropout during training)
    is_cat_dropout_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_cat_dropout_mode') and ingp.is_cat_dropout_mode

    hybrid_levels = ingp.hybrid_levels if (is_cat_mode or is_adaptive_zero_mode or is_adaptive_gate_mode or is_cat_dropout_mode or is_3D_direct_fused_mode) else 0
    
    render_mode = 0  # 0 = baseline, 1 = cat, 2 = adaptive_zero, 3 = 3D
    viewdirs_enc = None  # Pre-encoded view directions for 3D_direct_fused mode
    # proberes: (probes [N,6], tex [R,R,3], dims int32 [2]) — filled in the
    # 3D_SH_res body when _is_proberes(ingp); threaded via the *_diffuse kwargs.
    _probe_tensors = None
    # NOTE: 3D mode (render_mode=3) is set after hash_in_CUDA block to avoid being overwritten

    # Initialize shape_dims tensor [GS, HS, OS] - will be updated per mode
    # GS = Gaussian shape, HS = Hash shape, OS = Output shape
    # Default for SH rendering (no hashgrid): GS=0, HS=0, OS=3 (RGB)
    # This will be overwritten for warmup mode (line 111) or hash_in_CUDA modes (below)
    shape_dims = torch.tensor([0, 0, 3], dtype=torch.int32, device="cuda")

    if hash_in_CUDA:
        # Initialize shape_dims for hash_in_CUDA modes
        output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim
        shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
        # Check if hashgrid is disabled (cat mode with hybrid_levels == total_levels)
        if hasattr(ingp, 'hashgrid_disabled') and ingp.hashgrid_disabled:
            # No hashgrid - pure per-Gaussian mode
            features = torch.zeros((1, ingp.level_dim), device="cuda")
            offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
            gridrange = ingp.gridrange
            per_level_scale = 1
            base_resolution = 0
            align_corners = False
            interpolation = 0
            # Encode: total_levels in upper bits, 0 hashgrid levels, hybrid_levels in lower bits
            levels = (ingp.levels << 16) | (0 << 8) | ingp.hybrid_levels
        else:
            # Normal hashgrid mode (baseline or cat with hashgrid)
            features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
                = ingp.hash_encoding.get_params()
            gridrange = ingp.gridrange
            levels = ingp.active_levels

            # FP16 hash features: convert embeddings to half precision
            # Saves per-frame FP32→FP16 conversion in rasterize_points.cu
            # Autograd tracks .half() so gradients flow back to the FP32 parameter
            if is_3D_direct_fp16_mode or is_3D_direct_tc_mode or is_3D_direct_sh_tc_mode:
                features = features.half()

        # Use cached homotrans if available (for fast inference)
        if fast_inference and cache is not None and cache.homotrans is not None:
            homotrans = cache.homotrans
        else:
            homotrans = pc.get_homotrans()
        ap_level = pc.get_appearance_level
        contract = ingp.contract

        # Baseline mode: set shape_dims for hashgrid-only rendering
        # NOTE: We keep SH for preprocessing (it evaluates to 3-channel RGB in geomState.rgb)
        # but the rendering kernel will ignore it when level > 0 and query hashgrid instead
        # shape_dims tells CUDA to allocate 24-channel output buffer for hashgrid features
        if not is_cat_mode:
            # Baseline with hashgrid: GS=0 (no per-Gaussian), HS=output_dim, OS=output_dim
            # Use total levels (not active_levels) - buffer is always full size, C2F masks unused channels
            output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim (e.g., 6*4 = 24D)
            shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
        
        # Cat mode: only activate if hybrid_levels > 0
        # When hybrid_levels == 0, behave identically to baseline
        if is_cat_mode and hybrid_levels > 0:
            # Set per-Gaussian features as colors_precomp
            gaussian_features = pc.get_gaussian_features
            shs = None

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
            # Uses active_hashgrid_levels for C2F (progressively enables hashgrid levels)
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            # This is needed because CUDA copies offsets based on max possible levels
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            colors_precomp = gaussian_features
            render_mode = 1

            # Cat mode: G + H = O
            gaussian_dim = hybrid_levels * ingp.level_dim  # e.g., 5*4 = 20
            hash_dim = active_hashgrid_levels * ingp.level_dim  # e.g., 1*4 = 4
            output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24
            if fast_inference and cache is not None:
                shape_dims = cache.get_shape_dims((gaussian_dim, hash_dim, output_dim), [gaussian_dim, hash_dim, output_dim])
            else:
                shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # FiLM mode: f = gamma*H + beta per Gauss, then blend + screen-space MLP
        # (cat-family blend-then-PyTorch-MLP). hybrid_levels=0 → all 24D come from
        # the hashgrid; per-Gauss gamma/beta are passed as separate kernel tensors.
        elif is_film_mode:
            # Keep SH for preprocessing exactly like BASELINE (shs = _effective_shs(),
            # colors_precomp = None set above): the preprocess needs a color source to
            # build geomState.rgb — leaving both empty makes computeColorFromSH read an
            # empty sh tensor (illegal access). The render kernel's case-1 path queries
            # the hash + applies FiLM and ignores geomState.rgb, so the SH is unused at
            # render time (just like baseline's hash render).
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            # Cat-style level encoding with hybrid=0 → kernel takes the case-1 path
            # and queries all active hash levels (C2F respected) per Gauss.
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0

            # Pad offsets to 17 (CUDA copies up to 16 levels + 1)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            render_mode = 1
            output_dim = total_levels * ingp.level_dim  # 6*4 = 24
            shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_SH_cat mode: per-Gaussian SH + hash+DC_SH MLP residual
        # Same as 3D_SH_res but MLP input = [hash(4) | DC_SH(3) | bias(1)]
        # DC SH (unactivated) gives per-Gaussian identity to the MLP
        elif is_3D_SH_cat_mode:
            from diff_surfel_3D_sh_res import set_mlp_weights

            # Full SH for view-dependent base color (evaluated in CUDA preprocessing).
            # Under --feature beta/sg/voronoi/SV, _effective_shs() injects the directional
            # lobe term into the FAKE-DC slot so the rasterizer's SH eval reproduces the
            # paper formula: `SH_eval(real_dc + lobe_rgb/SH_C0 + higher_SH) = SH_eval(real)+lobe`.
            shs = _effective_shs()

            # MLP DC identity input: must be the VIEW-INDEPENDENT real DC SH evaluated to
            # RGB. The MLP residual rides on top of (SH_eval + lobe). Reading from
            # `_effective_shs()[:,0,:]` here would leak the (view-dependent) lobe into the
            # MLP's identity input and make the residual itself view-dependent — defeating
            # the intended decomposition (lobe = view-dep, MLP = view-indep spatial residual).
            # Use real `_features_dc` directly so the MLP sees pure view-independent identity.
            #
            # NOTE on the .detach():
            #   _features_dc gets gradient via TWO valid paths through the rasterizer:
            #     (a) `shs = _effective_shs()` → CUDA SH-eval backward → dL_dshs → grad_sh
            #         → autograd through `_effective_shs()` clone → _features_dc
            #     (b) `colors_precomp = SH_C0·_features_dc + 0.5` → grad_colors_precomp
            #         → autograd through this expression → _features_dc
            #   The CUDA cat backward routes BOTH the SH-color gradient AND the MLP's
            #   dc_sh-slot gradient into the SAME dL_dcolors buffer (which the SH backward
            #   then turns into dL_dshs[0,:] = SH_C0·dL_dcolors). If we left this expression
            #   differentiable, _features_dc would receive every contribution twice (once
            #   via SH backward → grad_sh, once via grad_colors_precomp). We .detach() so
            #   only the SH path feeds _features_dc — and that single path now carries
            #   BOTH the SH-color and the MLP-DC-slot contributions.
            SH_C0 = 0.28209479177387814
            dc_sh = SH_C0 * pc._features_dc.detach().squeeze(1) + 0.5  # [N, 3], view-indep, detached
            colors_precomp = dc_sh.contiguous()

            # Decompose mode for 3D_SH_cat:
            #   'sh_only': zero MLP weights, set res_bias=-999 → ReLU(0-999)=0
            #   'tex_only': set sh_bias=-999 → ReLU(SH-999)=0
            _zero_mlp_weights = False
            if decompose_mode == 'sh_only':
                _zero_mlp_weights = True
                _restore_bias = True
                set_activation_bias = _sh_res_setter_mod(ingp).set_activation_bias
                set_activation_bias(sh_bias=_ACTIVATION_BIAS[0], res_bias=0.0)  # sh_only: zero MLP weights handle it
            elif decompose_mode in ('tex_only', 'tex_only_raw'):
                _restore_bias = True
                set_activation_bias = _sh_res_setter_mod(ingp).set_activation_bias
                set_activation_bias(sh_bias=-999.0, res_bias=_ACTIVATION_BIAS[1])  # tex_only: kill SH

            # Hash grid setup: use actual hashgrid_levels (not config total),
            # since hybrid_levels may have reduced the hash grid size.
            total_levels = ingp.hashgrid_levels
            active_hashgrid_levels = min(
                ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0,
                total_levels)

            # Encode levels: (total << 16) | (active_hashgrid << 8) | hybrid=0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0

            # Pad offsets
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Upload MLP weights (zero for sh_only decomposition)
            if _zero_mlp_weights:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3))
            else:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(W1, W2, W3)

            render_mode = 6  # 3D_SH_cat: hash+DC MLP
            if ingp.freeze_mlp:
                render_mode |= 0x200  # bit 9: skip MLP weight gradients in CUDA backward

            # One-time verification
            global _3D_DIRECT_FUSED_VERIFIED
            if not _3D_DIRECT_FUSED_VERIFIED:
                hash_dim = active_hashgrid_levels * ingp.level_dim
                _pad = 16 - hash_dim - 3 - 1  # MLP input is 16D total
                print(f"[3D_SH_CAT] render_mode={render_mode}, "
                      f"SH=degree-3 (48 params), "
                      f"hash={active_hashgrid_levels}×{ingp.level_dim}={hash_dim}D, "
                      f"MLP input=[hash({hash_dim})|DC_SH(3)|bias(1)|pad({_pad})]=16D, "
                      f"MLP=16→16→16→3 residual")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Dimensions: no per-Gaussian features, hash only
            gaussian_dim = 0
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = 3  # RGB output
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_SH_32 mode: per-Gaussian SH + 32-dim hash MLP residual (diff_surfel_3D_sh_32)
        # Same as 3D_SH_res but with 32-dim hidden MLP
        elif is_3D_SH_32_mode or is_3D_SH_concat_mode:
            # Route setters to the concat fork when --method 3D_SH_concat (else plain 3D_SH_32).
            _setter_mod = _sh_concat_rasterizer if is_3D_SH_concat_mode else _sh_32_rasterizer
            set_mlp_weights = _setter_mod.set_mlp_weights

            # Use standard SH coefficients (NOT per-Gaussian features)
            shs = _effective_shs()
            colors_precomp = None

            # Decompose mode for 3D_SH_32 / 3D_SH_concat:
            #   'sh_only'      : zero MLP weights        → SV/SH base only.
            #   'tex_only'     : sh_bias=-999 (kill SH)  → MLP residual only (full input).
            #   'concat_latent': kill SH + zero hash     → ReLU(MLP([latent(16)|0]))   (latent's residual)
            #   'concat_hash'  : kill SH + zero latent   → ReLU(MLP([0|hash(16)]))      (hash's residual)
            _zero_mlp_weights = False
            _restore_bias = False
            _concat_zero_hash = False     # 'concat_latent': force active hash levels → 0
            _concat_zero_latent = False   # 'concat_hash': suppress the film_beta latent kwarg
            if decompose_mode == 'sh_only':
                _zero_mlp_weights = True
                _restore_bias = True
                set_activation_bias = _setter_mod.set_activation_bias
                set_activation_bias(sh_bias=_ACTIVATION_BIAS[0], res_bias=0.0)  # sh_only: zero MLP weights handle it
            elif decompose_mode in ('tex_only', 'tex_only_raw'):
                _restore_bias = True
                set_activation_bias = _setter_mod.set_activation_bias
                set_activation_bias(sh_bias=-999.0, res_bias=_ACTIVATION_BIAS[1])  # tex_only: kill SH
            elif decompose_mode == 'concat_latent':
                _restore_bias = True
                _concat_zero_hash = True
                set_activation_bias = _setter_mod.set_activation_bias
                set_activation_bias(sh_bias=-999.0, res_bias=0.0)  # kill SH; hash zeroed via active=0 below
            elif decompose_mode == 'concat_hash':
                _restore_bias = True
                _concat_zero_latent = True
                set_activation_bias = _setter_mod.set_activation_bias
                set_activation_bias(sh_bias=-999.0, res_bias=0.0)  # kill SH; latent zeroed via film_beta suppression

            # Hash grid setup: use actual hashgrid_levels (not config total),
            # since hybrid_levels may have reduced the hash grid size.
            total_levels = ingp.hashgrid_levels
            active_hashgrid_levels = min(
                ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0,
                total_levels)

            # concat_latent decompose: zero the hash half (query 0 hash levels)
            # → mlp_input = [latent(16) | 0], residual reflects the latent alone.
            if _concat_zero_hash:
                active_hashgrid_levels = 0

            # Encode levels: (total << 16) | (active_hashgrid << 8) | hybrid=0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0

            # Pad offsets
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Upload MLP weights (zero for sh_only decomposition)
            if _zero_mlp_weights:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3))
            else:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(W1, W2, W3)

            render_mode = 5  # Fused in-kernel MLP
            if ingp.freeze_mlp:
                render_mode |= 0x200  # bit 9: skip MLP weight gradients in CUDA backward

            # One-time verification
            if not _3D_DIRECT_FUSED_VERIFIED:
                hash_dim = active_hashgrid_levels * ingp.level_dim
                _tag = "3D_SH_CONCAT" if is_3D_SH_concat_mode else "3D_SH_32"
                _inp = f"input=[latent(16)|hash({hash_dim}D)]" if is_3D_SH_concat_mode else f"hash={active_hashgrid_levels}×{ingp.level_dim}={hash_dim}D"
                print(f"[{_tag}] render_mode={render_mode}, "
                      f"SH=degree-3 (48 params), "
                      f"{_inp}, "
                      f"MLP=32→32→32→3 residual")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Dimensions: no per-Gaussian features, hash only
            gaussian_dim = 0
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = 3  # RGB output
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_SH_res mode: per-Gaussian SH + tiny hash MLP residual
        # SH handles per-Gaussian view-dependent appearance (evaluated in CUDA preprocessing)
        # Hash MLP adds view-independent spatial correction per-intersection
        # No per-Gaussian features needed (hybrid_levels=0)
        elif is_3D_SH_res_mode:
            # filmres routes its module-local device globals (incl. set_mlp_weights) to
            # diff_surfel_3D_sh_filmres; plain 3D_SH_res-family stays on the base module.
            set_mlp_weights = _sh_res_setter_mod(ingp).set_mlp_weights

            # Use standard SH coefficients (NOT per-Gaussian features)
            shs = _effective_shs()
            colors_precomp = None

            # Decompose mode for 3D_SH_res:
            #   'sh_only': zero MLP weights, set res_bias=-999 → ReLU(0-999)=0
            #   'tex_only': set sh_bias=-999 → ReLU(SH-999)=0 for any normal SH values
            _zero_mlp_weights = False
            _restore_bias = False
            if decompose_mode == 'sh_only':
                _zero_mlp_weights = True
                _restore_bias = True
                set_activation_bias = _sh_res_setter_mod(ingp).set_activation_bias
                set_activation_bias(sh_bias=_ACTIVATION_BIAS[0], res_bias=0.0)  # sh_only: zero MLP weights handle it
            elif decompose_mode in ('tex_only', 'tex_only_raw'):
                _restore_bias = True
                set_activation_bias = _sh_res_setter_mod(ingp).set_activation_bias
                set_activation_bias(sh_bias=-999.0, res_bias=_ACTIVATION_BIAS[1])  # tex_only: kill SH

            # Hash grid setup: use actual hashgrid_levels (not config total),
            # since hybrid_levels may have reduced the hash grid size.
            total_levels = ingp.hashgrid_levels
            active_hashgrid_levels = min(
                ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0,
                total_levels)

            # PROBERES: silence the 3D scene hash — the kernel probe branch
            # (flag 0x1000, set below) replaces the hash+MLP residual entirely.
            if _is_proberes(ingp):
                active_hashgrid_levels = 0
            
            # Encode levels: (total << 16) | (active_hashgrid << 8) | hybrid=0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0

            # Pad offsets
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Upload MLP weights (zero for sh_only decomposition).
            # proberes: no in-kernel MLP — skip the upload entirely.
            if _is_proberes(ingp):
                pass
            elif _zero_mlp_weights:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3))
            else:
                mlp_weights = ingp.get_fused_mlp_weights()
                if mlp_weights is not None:
                    W1, W2, W3 = mlp_weights
                    set_mlp_weights(W1, W2, W3)

            render_mode = 5  # Fused in-kernel MLP
            if ingp.freeze_mlp:
                render_mode |= 0x200  # bit 9: skip MLP weight gradients in CUDA backward
            
            if _is_proberes(ingp):
                # PROBERES: flag the kernel probe branch and build the per-render
                # probe/texture tensors. Both stay graph-connected: dL/dprobes
                # flows through ProbeHead3D into xyz/rotation/scale (positional
                # gradients) and dL/dtex through the bake into ProbeTexField2D.
                render_mode |= 0x1000
                # --res_warmup: while the family's hashgrid_disabled flag is up,
                # withhold the probe/texture tensors — the kernel's null-pointer
                # guard renders residual = 0 (SV-only), matching res-warmup
                # semantics for the other family members. Same trick implements
                # decompose_mode='sh_only' (the base's zero-MLP-weights route
                # doesn't exist here). 'tex_only' needs no probe-side handling:
                # the sh_bias=-999 set above zeroes the SV base as usual.
                if not ingp.hashgrid_disabled and decompose_mode != 'sh_only':
                    _pr = ingp.probe_head(pc.get_xyz, pc.get_rotation, pc.get_scaling)
                    _tex = (ingp.probe_field.bake_cached(iteration)
                            if is_training else ingp.probe_field.bake(sparse_bw=False))
                    _dims = torch.tensor([_tex.shape[0], _tex.shape[1]],
                                         dtype=torch.int32, device="cuda")
                    _probe_tensors = (_pr.contiguous(), _tex.contiguous(), _dims)

            # One-time verification
            if not _3D_DIRECT_FUSED_VERIFIED:
                hash_dim = active_hashgrid_levels * ingp.level_dim
                print(f"[3D_SH_RES] render_mode={render_mode}, "
                      f"SH=degree-3 (48 params), "
                      f"hash={active_hashgrid_levels}×{ingp.level_dim}={hash_dim}D, "
                      f"MLP=16→16→16→3 residual")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Set Nexels-style anti-aliasing (hash-grid down-weighting per level).
            # Only call when enabled — skip entirely if off so rasterizer builds without
            # the symbol still work.
            if antialiasing > 0.0:
                try:
                    _set_aa = _sh_res_setter_mod(ingp).set_anti_alias
                    _focal = max(viewpoint_camera.image_width / (2.0 * math.tan(viewpoint_camera.FoVx / 2.0)),
                                 viewpoint_camera.image_height / (2.0 * math.tan(viewpoint_camera.FoVy / 2.0)))
                    _set_aa(antialiasing, _focal)
                except (ImportError, AttributeError):
                    pass  # AA not compiled in this build — fall back to no AA

            # Dimensions: no per-Gaussian features, hash only
            gaussian_dim = 0
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = 3  # RGB output
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_direct_fused mode: fused in-kernel MLP (hash+MLP in CUDA)
        # Like cat mode but MLP runs in-kernel and outputs RGB directly
        # hybrid_levels are per-Gaussian (coarse), remaining levels are hashgrid (fine)
        elif is_3D_direct_fused_mode and ingp.hybrid_levels > 0:
            # Import set_mlp_weights from the appropriate rasterizer
            if is_3D_direct_sh_tc_mode and SH_TC_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_sh import set_mlp_weights
            elif is_3D_direct_tc_mode and TC_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_tc import set_mlp_weights
            elif is_3D_direct_fp16_mode and FP16_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_16 import set_mlp_weights
            elif is_3D_direct_lean_mode and LEAN_RASTERIZER_AVAILABLE:
                from diff_surfel_3D import set_mlp_weights
            else:
                from diff_surfel_rasterization import set_mlp_weights

            # Same setup as cat mode: per-Gaussian features + hashgrid
            gaussian_features = pc.get_gaussian_features
            shs = None

            # Level split: hybrid_levels are Gaussian, rest are hashgrid
            # E.g., total=6, hybrid=3 means 3 Gaussian levels + 3 hashgrid levels
            total_levels = ingp.levels
            hybrid_levels_fused = ingp.hybrid_levels
            hashgrid_levels = total_levels - hybrid_levels_fused  # Fine levels from hash

            # 3D_direct_fused has no C2F - MLP expects fixed input dimensions
            # Use all hashgrid levels (not active_hashgrid_levels which may be limited by C2F)
            active_hashgrid_levels = hashgrid_levels if not ingp.hashgrid_disabled else 0

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_fused

            # Pad offsets (same as cat mode)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            colors_precomp = gaussian_features

            # Upload MLP weights to CUDA constant memory
            # NOTE: MLP gradients are computed in CUDA but not yet wired back to PyTorch
            # The mlp_fused parameters won't receive gradients until that's implemented
            mlp_weights = ingp.get_fused_mlp_weights()
            if mlp_weights is not None:
                W1, W2, W3 = mlp_weights
                set_mlp_weights(W1, W2, W3, is_sh_mode=is_3D_direct_sh_tc_mode)

            render_mode = 5  # 3D_direct_fused: fused in-kernel MLP

            # One-time verification that we're using the fused mode
            if not _3D_DIRECT_FUSED_VERIFIED:
                print(f"[3D_DIRECT_FUSED] render_mode={render_mode}, "
                      f"gauss={hybrid_levels_fused}×{ingp.level_dim}={hybrid_levels_fused * ingp.level_dim}D, "
                      f"hash={active_hashgrid_levels}×{ingp.level_dim}={active_hashgrid_levels * ingp.level_dim}D")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Dimensions for CUDA kernel
            gaussian_dim = hybrid_levels_fused * ingp.level_dim  # Coarse: e.g., 5*4 = 20
            hash_dim = active_hashgrid_levels * ingp.level_dim   # Fine: e.g., 1*4 = 4
            output_dim = 3  # RGB output from fused MLP
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

            # View direction handling depends on mode:
            # - SH_TC: no view encoding needed (CUDA kernel computes raw viewdir per-intersection)
            # - Other fused modes: pre-encode view directions for MLP input
            if not is_3D_direct_sh_tc_mode:
                rays_d, _ = cam2rays(viewpoint_camera)  # [H*W, 3]
                ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # [H*W, 3] normalized
                viewdirs_enc = ingp._encode_view(ray_unit).float().contiguous()  # [H*W, 16]

        # Cat_dropout mode: cat mode with hash dropout during training
        # Uses mode 14 (adaptive_zero kernel) with hardcoded weights
        # Training: weight=1 for (1-dropout_lambda)% of Gaussians, weight=0 for dropout_lambda%
        # Inference: weight=1 for all (identical to cat mode)
        elif is_cat_dropout_mode and hybrid_levels > 0:
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)
            shs = None

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            N = gaussian_features.shape[0]

            if is_training and dropout_lambda > 0:
                # Training with dropout: randomly mask some Gaussians to not use hash
                # weight=1 means use hash, weight=0 means use zeros for fine levels
                dropout_mask = (torch.rand(N, 1, device="cuda") >= dropout_lambda).float()
            else:
                # Inference or no dropout: all weights = 1 (use hash for all)
                dropout_mask = torch.ones(N, 1, device="cuda")

            # Concatenate: [coarse_features | weight]
            colors_precomp = torch.cat([gaussian_features, dropout_mask], dim=1)

            # Use mode 14 (adaptive_zero kernel) - training mode (no inference flag)
            render_mode = 2

            # Shape dims: GS = coarse, HS = fine hash, OS = total
            gaussian_dim = hybrid_levels * ingp.level_dim
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = total_levels * ingp.level_dim
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # Adaptive_zero mode: cat-like features + weighted hash (zeros when weight=0)
        elif is_adaptive_zero_mode and hybrid_levels > 0:
            # Same feature layout as cat mode: per-Gaussian for coarse, hash for fine
            # But with a weight that controls whether to query hash or use zeros
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)

            # Apply temperature scaling to sigmoid (higher temp = sharper sigmoid)
            blend_weight = torch.sigmoid(pc._adaptive_zero_weight * temperature)  # (N, 1)

            # Determine if we're in inference mode
            use_inference_mode = (hasattr(ingp, 'adaptive_zero_inference') and ingp.adaptive_zero_inference) or (decompose_mode is not None)

            # Decompose mode: override features/opacity for visualization
            opacity_override = None
            disable_hash_query = False

            if decompose_mode == 'gaussian_only':
                # Show only Gaussians with weight < 0.5 (they use zeros for fine levels)
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) >= 0.5] = 0.0
            elif decompose_mode == 'hybrid_gaussian_only':
                # Show Gaussians with weight >= 0.5, but mask out hashgrid
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) < 0.5] = 0.0
                disable_hash_query = True  # Will set active_hashgrid_levels = 0
            elif decompose_mode == 'hybrid_hash_only':
                # Show Gaussians with weight >= 0.5, but mask out gaussian features
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) < 0.5] = 0.0
                gaussian_features = torch.zeros_like(gaussian_features)

            if opacity_override is not None:
                opacity = opacity_override

            # Concatenate: [coarse_features | weight]
            colors_precomp = torch.cat([gaussian_features, blend_weight], dim=1)
            shs = None

            # Use mode 14 for adaptive_zero
            inference_flag = 1 if use_inference_mode else 0
            render_mode = 2 | (inference_flag << 8)

            # Level encoding same as cat: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            if disable_hash_query:
                active_hashgrid_levels = 0  # Disable hash query for hybrid_gaussian_only mode
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Shape dims: GS = coarse features + weight, HS = fine hash, OS = total output
            gaussian_dim = hybrid_levels * ingp.level_dim  # e.g., 5*4 = 20D (coarse only)
            hash_dim = active_hashgrid_levels * ingp.level_dim  # e.g., 1*4 = 4D
            output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24D
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # Adaptive_gate mode: Gumbel-STE with forced training for binary hash selection
        # Always binary masking (0 or 1) to prevent scale compensation artifacts
        # Uses same CUDA kernel as adaptive_zero (mode 14)
        elif is_adaptive_gate_mode and hybrid_levels > 0:
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)
            gate_logits = pc._gate_logits  # (N, 1)

            use_inference_mode = hasattr(ingp, 'adaptive_gate_inference') and ingp.adaptive_gate_inference

            if use_inference_mode or (decompose_mode is not None):
                # INFERENCE: Hard threshold on probability (prob > 0.5 → use hash)
                effective_mask = (torch.sigmoid(gate_logits) > 0.5).float()
            else:
                # TRAINING: STE with optional Gumbel noise + Forced Training

                if no_gumbel:
                    # Deterministic STE: no Gumbel noise, just hard threshold with soft gradients
                    soft_gate = torch.sigmoid(gate_logits / temperature)
                    hard_gate = (soft_gate > 0.5).float() - soft_gate.detach() + soft_gate
                else:
                    # Gumbel-STE: stochastic exploration
                    # 1. Gumbel noise for stochastic exploration
                    uniform = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
                    gumbel_noise = -torch.log(-torch.log(uniform))
                    noisy_logits = (gate_logits + gumbel_noise) / temperature

                    # 2. STE: hard forward, soft backward
                    soft_gate = torch.sigmoid(noisy_logits)
                    hard_gate = (soft_gate > 0.5).float() - soft_gate.detach() + soft_gate

                # 3. Forced training: force some Gaussians to use hash (detached, not learnable)
                force_mask = (torch.rand_like(gate_logits) < force_ratio).float().detach()

                # 4. Combine: max ensures binary output (0 or 1)
                effective_mask = torch.max(hard_gate, force_mask)

            # Handle decompose modes for visualization
            opacity_override = None
            disable_hash_query = False

            if decompose_mode == 'gaussian_only':
                # Force all gates closed (Gaussian-only rendering)
                effective_mask = torch.zeros_like(effective_mask)
            elif decompose_mode == 'ngp_only':
                # Force all gates open (hash for all)
                effective_mask = torch.ones_like(effective_mask)
            elif decompose_mode == 'gate_closed':
                # Show only Gaussians with gate closed (not using hash)
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) > 0.5] = 0.0  # Hide gate-open Gaussians
                effective_mask = torch.zeros_like(effective_mask)  # All use Gaussian-only
            elif decompose_mode == 'gate_open':
                # Show only Gaussians with gate open (using hash)
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0  # Hide gate-closed Gaussians
                effective_mask = torch.ones_like(effective_mask)  # All use hash
            elif decompose_mode == 'hybrid_gaussian_only':
                # Show Gaussians with gate open, but mask out hashgrid
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0
                disable_hash_query = True
            elif decompose_mode == 'hybrid_hash_only':
                # Show Gaussians with gate open, but mask out gaussian features
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0
                gaussian_features = torch.zeros_like(gaussian_features)

            if opacity_override is not None:
                opacity = opacity_override

            # Concatenate: [coarse_features | mask]
            colors_precomp = torch.cat([gaussian_features, effective_mask], dim=1)
            shs = None

            # Use mode 14 (same CUDA kernel as adaptive_zero)
            inference_flag = 1 if (use_inference_mode or decompose_mode is not None) else 0
            render_mode = 2 | (inference_flag << 8)

            # Level encoding same as cat: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            if disable_hash_query:
                active_hashgrid_levels = 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Shape dims: GS = coarse features + mask, HS = fine hash, OS = total output
            gaussian_dim = hybrid_levels * ingp.level_dim
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = total_levels * ingp.level_dim
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

    # 3D mode: Configure AFTER the reset block (like cat mode configures inside hash_in_CUDA)
    # This follows cat mode's pattern exactly: set shs=None, colors_precomp=gaussian_features
    if is_3D_mode and ingp is not None:
        # Per-Gaussian features as colors_precomp (like cat mode at line 550-566)
        gaussian_features = pc.get_gaussian_features
        shs = None  # Like cat mode: no SH, use colors_precomp instead
        colors_precomp = gaussian_features  # Like cat mode

        # Homotrans for intersection computation (like cat mode at line 532)
        homotrans = pc.get_homotrans()

        # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
        # IMPORTANT: levels must be non-zero to enter the render_mode switch in CUDA
        total_levels = ingp.levels
        active_hashgrid_levels = 0  # 3D mode: hashgrid encoding in PyTorch, not CUDA
        hybrid_levels_3D = ingp.hybrid_levels
        levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_3D

        # Pad offsets to 17 elements (like cat mode at line 561-564)
        offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
        if offsets.shape[0] < 17:
            padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
            padded_offsets[:offsets.shape[0]] = offsets
            offsets = padded_offsets

        # Hashgrid params (dummy for 3D mode - hash query happens in PyTorch)
        features = torch.zeros((1, 4), device="cuda")  # Dummy hash features
        gridrange = ingp.gridrange if hasattr(ingp, 'gridrange') else torch.tensor([-2.0, 2.0], device="cuda")
        per_level_scale = ingp.growth_rate if hasattr(ingp, 'growth_rate') else 1.0
        base_resolution = ingp.resolutions[0] if hasattr(ingp, 'resolutions') and len(ingp.resolutions) > 0 else 16
        align_corners = False
        interpolation = 0
        contract = ingp.contract if hasattr(ingp, 'contract') else False

        # render_mode = 3 for intersection buffer output
        render_mode = 3

        # shape_dims like cat mode: G + H = O (line 570-576)
        gaussian_dim = hybrid_levels_3D * ingp.level_dim  # e.g., 5*4 = 20
        hash_dim = active_hashgrid_levels * ingp.level_dim  # 0 for 3D mode
        output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24
        shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

    # 3D_direct mode: Same rasterizer setup as 3D mode, but uses direct RGB output instead of SH
    if is_3D_direct_mode and ingp is not None:
        gaussian_features = pc.get_gaussian_features
        shs = None
        colors_precomp = gaussian_features

        homotrans = pc.get_homotrans()

        total_levels = ingp.levels
        active_hashgrid_levels = 0  # Hash query in PyTorch
        hybrid_levels_3D = ingp.hybrid_levels
        levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_3D

        offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
        if offsets.shape[0] < 17:
            padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
            padded_offsets[:offsets.shape[0]] = offsets
            offsets = padded_offsets

        features = torch.zeros((1, 4), device="cuda")
        gridrange = ingp.gridrange if hasattr(ingp, 'gridrange') else torch.tensor([-2.0, 2.0], device="cuda")
        per_level_scale = ingp.growth_rate if hasattr(ingp, 'growth_rate') else 1.0
        base_resolution = ingp.resolutions[0] if hasattr(ingp, 'resolutions') and len(ingp.resolutions) > 0 else 16
        align_corners = False
        interpolation = 0
        contract = ingp.contract if hasattr(ingp, 'contract') else False

        render_mode = 3  # Same rasterizer as 3D mode (intersection buffer output)

        gaussian_dim = hybrid_levels_3D * ingp.level_dim
        hash_dim = active_hashgrid_levels * ingp.level_dim
        output_dim = total_levels * ingp.level_dim
        shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

    # For diffuse/diffuse_ngp/diffuse_offset mode, use sh_degree=0 (only DC component)
    # For specular mode, use full active_sh_degree.
    # `--feature beta` also uses only the DC slot — directional component
    # comes from SB lobes, not SH. _features_rest is allocated as [N, 0, 3] in
    # that mode, so passing sh_degree=3 would read past the end of the tensor.
    _is_beta_feature = getattr(pc, 'feature_mode', 'sh') == 'beta'
    sh_degree_to_use = 0 if (is_diffuse_mode or is_diffuse_ngp_mode or is_diffuse_offset_mode or _is_beta_feature) else pc.active_sh_degree
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree_to_use,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        beta=beta,
        if_contract = contract,
        record_transmittance = record_transmittance,
        max_intersections = max_intersections,
        detach_hash_grad = detach_hash_grad,
        max_intersections_per_pixel = max_intersections_per_pixel,
        # pipe.debug
    )

    hashgrid_settings = HashGridSettings(
        L = levels,
        S = math.log2(per_level_scale),
        H = base_resolution,
        align_corners = align_corners,
        interpolation = interpolation,
        shape_dims = shape_dims,
        aa = aa,
        aa_threshold = aa_threshold
    )

    # Use SH_RES, SH_TC, TC, FP16, or lean rasterizer for fused modes
    if is_3D_SH_concat_mode and SH_CONCAT_RASTERIZER_AVAILABLE:
        rasterizer = _sh_concat_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_SH_32_mode and SH_32_RASTERIZER_AVAILABLE:
        rasterizer = _sh_32_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif (is_3D_SH_res_mode or is_3D_SH_cat_mode) and SH_RES_RASTERIZER_AVAILABLE:
        # `--method mixed`: route through diff_surfel_mixed (the fork that has the
        # per-Gauss textured/untextured CUDA branch). For all other 3D_SH_res / 3D_SH_cat
        # runs use the original rasterizer (bit-identical to before).
        # `--method res_3d`: ONLY route through diff_surfel_mixed_3d AFTER the
        # split fires (post-split needs the EWA-ellipsoid path for the 3D
        # SV-only carrier). Pre-split res_3d uses the same rasterizer as
        # 3D_SH_res (diff_surfel_3D_sh_res) so it's bit-identical to
        # `--method 3D_SH_res --lru α` until the split — no risk of crossing
        # mixed_3d's setup paths for a configuration it wasn't initialised for.
        # `--method res_3d` SINGLE-PASS path (preferred when available): one
        # forward+backward call to diff_surfel_res_3d. Forward emits the
        # signed dual-cascade sum; renderer applies LRU/relu after. Backward
        # routes per-Gauss color/alpha grads through dual T cascades.
        # `--method res_3d` AND `--method res_3d_double` both route through
        # the single-pass dual-cascade kernel `diff_surfel_res_3d`. The two
        # methods differ only in the kernel's `d_textured_bias_gate` value
        # (set at stage 2 by train.py): res_3d keeps it at 1 (tex sh_color
        # forced to 0), res_3d_double sets it to 0 (tex keeps SV color).
        _is_res_3d_single_pass = (ingp is not None
                                   and ((getattr(ingp, 'is_res_3d_mode', False)
                                          and getattr(ingp, 'is_res_3d_post_split', False))
                                        or (getattr(ingp, 'is_res_3d_double_mode', False)
                                            and getattr(ingp, 'is_res_3d_double_post_split', False)))
                                   and RES_3D_RASTERIZER_AVAILABLE)
        # `--method res_3d_paired` post-split: SHARED-T joint cascade via the
        # mixed_3d kernel (single render, no two-render dispatch, no dual cascade).
        # Opacity scaling at split (texsplit_tex_frac) is what differentiates
        # the two halves' contributions.
        _is_res_3d_paired = (ingp is not None
                              and getattr(ingp, 'is_res_3d_paired_mode', False)
                              and getattr(ingp, 'is_res_3d_paired_post_split', False))
        # `--method GEStex` joint stage: behaves like res_3d_paired post-split (joint
        # cascade: textured 2D surfels + untextured EWA 3D Gaussians), but routes to the
        # diff_surfel_gestex clone whose textured residual is the baked atlas lookup.
        _is_gestex_joint = (ingp is not None
                            and getattr(ingp, 'is_gestex_joint', False)
                            and GESTEX_RASTERIZER_AVAILABLE)
        _is_mixed_3d = ingp is not None and (
            getattr(ingp, 'is_mixed_3d_mode', False)
            # res_3d falls back to mixed_3d two-render path only if
            # diff_surfel_res_3d isn't built.
            or (getattr(ingp, 'is_res_3d_mode', False)
                and getattr(ingp, 'is_res_3d_post_split', False)
                and not _is_res_3d_single_pass)
            or _is_res_3d_paired
            or _is_gestex_joint   # EWA + is_textured + scaling_z plumbing
        ) and (MIXED_3D_RASTERIZER_AVAILABLE or _is_gestex_joint)
        _is_mixed = ingp is not None and getattr(ingp, 'is_mixed_mode', False) and MIXED_RASTERIZER_AVAILABLE
        # res_3d_paired routes through its dedicated SLIM submodule when built,
        # falling back to plain mixed_3d if not. Both kernels are functionally
        # identical for the per-Gauss math; the paired clone just trims the
        # 18-channel out_others (kept) to 5 (DEPTH+ALPHA+NORMAL) — at 4K image
        # resolution this saves ~870 MB of per-pixel forward output + matching
        # backward gradient. The Python allmap reads below are size-aware so
        # the trimmed channels return None.
        _use_paired_slim = (_is_res_3d_paired and RES_3D_PAIRED_RASTERIZER_AVAILABLE)
        # `--method 3D_SH_filmres`: 3D_SH_res with FiLM-conditioned residual MLP — its own fork.
        _is_filmres = (ingp is not None and getattr(ingp, 'is_3D_SH_filmres_mode', False)
                       and SH_FILMRES_RASTERIZER_AVAILABLE)
        if _is_gestex_joint and os.environ.get('GESTEX_USE_PAIRED') == '1' and RES_3D_PAIRED_RASTERIZER_AVAILABLE:
            _rmod = _res_3d_paired_rasterizer   # isolation: verified module, same config, no atlas
        elif _is_gestex_joint:
            _rmod = _gestex_rasterizer
        elif _is_filmres:
            _rmod = _sh_filmres_rasterizer
        elif _is_res_3d_single_pass:
            _rmod = _res_3d_rasterizer
        elif _use_paired_slim:
            _rmod = _res_3d_paired_rasterizer
        elif _is_mixed_3d:
            _rmod = _mixed_3d_rasterizer
        elif _is_mixed:
            _rmod = _mixed_rasterizer
        elif _is_gestex_harden(ingp):
            # GEStex explore+harden (0-20k, pre-joint): the isolated clone (=sh_res +
            # first-intersection sort). Keeps GEStex rasterizer changes out of the
            # shared diff_surfel_3D_sh_res. Byte-identical to sh_res until first-int
            # sort flips on at --ges_first_int_iter (15k).
            _rmod = _gestex_harden_rasterizer
        elif _is_densfix(ingp):
            # `--densfix` (--method 3D_SH_res): isolated clone that excludes the
            # hash-query-point term from the AbsGS densify proxy. Byte-identical to
            # sh_res unless set_exclude_hash_from_densify(1) is installed.
            _rmod = _densfix_rasterizer
        elif _is_trunc(ingp):
            # `--trunc` (--method 3D_SH_res): isolated clone with the settable
            # POST-blend truncation exit threshold (set_exit_T). Byte-identical
            # to sh_res at the default 1e-4.
            _rmod = _trunc_rasterizer
        elif _is_proberes_wsr(ingp):
            # `--wsr` proberes: WSR clone (sort-free weighted-sum composite,
            # docs/WSR_DISTILL.md). Sorted-identical under set_wsr(0).
            _rmod = _sh_res_probe_wsr_rasterizer
            _H = int(viewpoint_camera.image_height)
            _W = int(viewpoint_camera.image_width)
            if getattr(ingp, 'wsr_sorted', False):
                # Distill-dump / debug: render SORTED through the wsr clone
                # (needed for its Σα / Σ(α·T) record_transmittance semantics).
                _rmod.set_wsr(0)
                pc._wsr_render_state = None
            else:
                assert hasattr(pc, '_wsr_occ') and pc._wsr_occ.numel() > 0, \
                    "--wsr render needs pc._wsr_occ (load a PLY with args.wsr set)"
                # Detached activation: occ grads arrive via the device-global
                # accumulator; train.py chains the sigmoid derivative manually.
                _occ_act = torch.sigmoid(pc._wsr_occ.detach()).view(-1).contiguous()
                _occ_grad = torch.zeros_like(_occ_act)
                # aux is [8,H,W]: mode 1 uses slots 0..3 (C̄, den); mode 2
                # (--wsr_composite, ht=1-style front + occ tail) adds P_t, α_F
                # and the front gauss id in slots 4..6.
                _wsr_aux = torch.zeros((8, _H, _W), dtype=torch.float32, device="cuda")
                _wsr_m = 2 if getattr(ingp, 'is_wsr_composite', False) else 1
                _rmod.set_wsr(_wsr_m, _occ_act, _occ_grad, _wsr_aux)
                # Transmittance gate (--wsr_gate_tau > 0): arm the depth-binned
                # pre-pass. tbin must outlive the backward (stashed below).
                # Constants (bins=16, zmin=0.2, zmax=120) are mirrored in the
                # viewer's WGSL uniforms — change together or train≠deploy.
                _gate_tau = float(getattr(ingp, 'wsr_gate_tau', 0.0) or 0.0)
                _dgate_m = float(getattr(ingp, 'wsr_dgate_margin', 0.0) or 0.0)
                if _gate_tau > 0.0:
                    _wsr_tbin = torch.empty((16, _H, _W), dtype=torch.float32, device="cuda")
                    _rmod.set_wsr_gate(_gate_tau, 16, 0.2, 120.0, _wsr_tbin)
                    _rmod.set_wsr_dgate(0.0)
                elif _dgate_m > 0.0:
                    # Mean-depth gate (?wsr=3): sorted pre-pass fills (D̄, A).
                    _wsr_tbin = torch.empty((2, _H, _W), dtype=torch.float32, device="cuda")
                    _rmod.set_wsr_dgate(_dgate_m, _wsr_tbin)
                    _rmod.set_wsr_gate(0.0)
                else:
                    _wsr_tbin = None
                    _rmod.set_wsr_gate(0.0)
                    _rmod.set_wsr_dgate(0.0)
                # Keep alive through backward + let train.py read the grads.
                # NOTE: any additional render between this forward and its
                # backward would clobber the device-global pointers.
                pc._wsr_render_state = (_occ_act, _occ_grad, _wsr_aux, _wsr_tbin)
        elif _is_proberes(ingp):
            # `--method proberes`: isolated clone whose case-5 residual is a
            # probe-mapped bilinear fetch from the shared texture image
            # (render_mode 5 | 0x1000). No in-kernel MLP, collab-GEMM forced off.
            _rmod = _sh_res_probe_rasterizer
        else:
            _rmod = _sh_res_rasterizer
        # `--method GEStex` joint stage: point the textured residual at the baked atlas
        # (bilinear at ray-disc UV) instead of hash+MLP, and allocate the grad buffer the
        # backward atomic-scatters into (train.py assigns it to _tex_atlas.grad post-backward).
        if _is_gestex_joint and getattr(pc, '_tex_atlas', None) is not None and pc._tex_atlas.numel() > 0 \
                and os.environ.get('GESTEX_NOATLAS') != '1':
            _atlas = pc._tex_atlas.contiguous()
            _atlas_grad = torch.zeros_like(_atlas)
            pc._ges_atlas_grad = _atlas_grad   # retrieved by train.py after loss.backward()
            _R = int(getattr(pc, 'ges_atlas_res', _atlas.shape[1]))
            _rmod.set_gestex_atlas(_atlas, _atlas_grad, _R, 4.0)
        elif _is_gestex_joint and _gestex_rasterizer is not None:
            _gestex_rasterizer.clear_gestex_atlas()   # isolation / no-atlas fallback
        rasterizer = _rmod.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_sh_tc_mode and SH_TC_RASTERIZER_AVAILABLE:
        rasterizer = _sh_tc_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_tc_mode and TC_RASTERIZER_AVAILABLE:
        rasterizer = _tc_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_fp16_mode and FP16_RASTERIZER_AVAILABLE:
        rasterizer = _fp16_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_lean_mode and LEAN_RASTERIZER_AVAILABLE:
        rasterizer = _lean_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_film_mode and FILM_RASTERIZER_AVAILABLE:
        # `--method film`: FiLM rasterizer fork (per-Gauss gamma/beta modulate the hash).
        rasterizer = _film_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

    # Get shape parameter for beta/general kernel (if using beta or general kernel)
    # Get flex_beta parameter for flex kernel (if using flex kernel)
    shapes = None
    kernel_type = 0  # 0=gaussian, 1=beta, 2=flex, 3=general, 4=beta_scaled
    if hasattr(pc, 'kernel_type') and pc.kernel_type == "beta" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        shapes = pc.get_shape
        kernel_type = 1
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "beta_scaled" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        # Beta kernel scaled to match 3σ Gaussian extent (r ∈ [0,3] instead of [0,1])
        shapes = pc.get_shape
        kernel_type = 4
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "flex" and hasattr(pc, '_flex_beta') and pc._flex_beta.numel() > 0:
        # For flex kernel, pass per-Gaussian beta via the shapes parameter
        # The CUDA kernel will interpret this as beta instead of shape based on kernel_type=2
        shapes = pc.get_flex_beta
        kernel_type = 2
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "general" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        # For general kernel (Isotropic Generalized Gaussian), pass beta via shapes parameter
        # Beta in range [2.0, 8.0]: 2.0=standard Gaussian, 8.0=super-Gaussian (box)
        shapes = pc.get_shape
        kernel_type = 3
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "nexel" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        # Nexel kernel: per-axis gamma exponents [N, 2] (gamma_x, gamma_y).
        # G = exp(-0.5 * (pow(s_x²+eps, gamma_x) + pow(s_y²+eps, gamma_y)))
        # gamma = exp(raw) + 1, range [1, inf). gamma=1 is standard Gaussian.
        shapes = pc.get_shape  # [N, 2] activated gamma values
        kernel_type = 5

    # Convert aabb_mode string to int:
    # 0 = square AABB, fixed 4σ cutoff (2DGS default)
    # 1 = square AABB, AdR cutoff (adaptive) - use "adr_only" for this
    # 2 = rectangular AABB, fixed 4σ cutoff
    # 3 = rectangular AABB, AdR cutoff (full optimization) - use "adr" for this
    # 4 = beta kernel: fixed r=1 cutoff (compact support)
    # 5 = AdR + rectangular AABB + AccuTile ellipse cull (SnugBox; fastest)
    if isinstance(aabb_mode, int):
        aabb_mode_int = aabb_mode
    elif aabb_mode == "adr":
        aabb_mode_int = 3  # Full optimization: AdR + rectangular AABB
    elif aabb_mode == "adr_only":
        aabb_mode_int = 1  # AdR cutoff only (square AABB)
    elif aabb_mode == "rect":
        aabb_mode_int = 2
    elif aabb_mode == "adr_rect":
        aabb_mode_int = 3  # Same as "adr"
    elif aabb_mode == "beta":
        aabb_mode_int = 4  # Beta kernel: fixed r=1 cutoff
    elif aabb_mode == "accutile" or aabb_mode == "adrrect_accu" or aabb_mode == "snugbox":
        aabb_mode_int = 5  # AdR + rect AABB + AccuTile ellipse cull
    else:
        aabb_mode_int = 0  # "2dgs" or default

    # First-intersection handling (GEStex harden) — two INDEPENDENT mechanisms
    # on the isolated diff_surfel_3D_sh_res_harden clone, driven per-iter by
    # train.py:
    #  - `ingp.first_int_sort` (--ges_first_int_iter, 10k): TILE-DEPTH SORT —
    #    key each (Gauss,tile) on the intersection depth at the tile-center ray
    #    (set_tile_depth_sort). Exact alpha blending at any opacity, correct
    #    ordering for tilted surfels. Rect-only → drop the AccuTile ellipse
    #    cull (mode 5 → 3) while active.
    #  - `ingp.frontmost_on` (--ges_frontmost_iter, 18k): GES-literal
    #    FRONTMOST-FIRST 2-pass — per-pixel frontmost (full-range scan) blends
    #    first (set_frontmost_first, fwd+bwd). Near-opaque regime only; does
    #    not touch binning. Composable with the tile sort.
    # Both off (default) ⇒ byte-identical to the shared rasterizer.
    _first_int_sort = bool(getattr(ingp, 'first_int_sort', False)) if ingp is not None else False
    _frontmost_on = bool(getattr(ingp, 'frontmost_on', False)) if ingp is not None else False
    _fis_mod = _sh_res_setter_mod(ingp)
    if _fis_mod is not None and hasattr(_fis_mod, 'set_frontmost_first'):
        _fis_mod.set_frontmost_first(_frontmost_on)
    if _first_int_sort and aabb_mode_int == 5:
        aabb_mode_int = 3
    if _fis_mod is not None and hasattr(_fis_mod, 'set_tile_depth_sort'):
        _fis_mod.set_tile_depth_sort(_first_int_sort)

    # Bit 10: enable low-pass filter backward gradient (rho2d → transMat)
    if lowpass:
        render_mode |= 0x400
    # Bit 11: pixel-center convention (pixf = pix + 0.5, ndc2pix offset = W/2)
    if pixel_center:
        render_mode |= 0x800
    # Bits [16..19]: `--method mixed_3d` `--kernel2` — kernel for the UNTEXTURED
    # EWA half, overriding `kernel_type` for that half only. Encoded as
    # (kernel_type2 + 1); a zero nibble means "unset" → untextured use
    # kernel_type (byte-identical to before). mixed_3d-gated; render_mode is
    # already threaded to every render kernel (fwd + bwd std/MODE-5), so no
    # signature changes are needed. The textured half always uses kernel_type.
    if _is_mixed_3d:
        _k2s = getattr(pc, 'kernel_type2', None)
        if _k2s is not None:
            _KMAP2 = {'gaussian': 0, 'beta': 1, 'flex': 2,
                      'general': 3, 'beta_scaled': 4, 'nexel': 5}
            _kt2 = _KMAP2.get(_k2s, -1)
            if _kt2 >= 0:
                render_mode |= ((_kt2 + 1) << 16)

    # Build rasterizer kwargs
    rasterizer_kwargs = dict(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        homotrans = homotrans,
        ap_level = ap_level,
        cov3D_precomp = cov3D_precomp,
        features = features,
        offsets = offsets,
        gridrange = gridrange,
        render_mode = render_mode,
        shapes = shapes,
        kernel_type = kernel_type,
        aabb_mode = aabb_mode_int,
    )
    # PROBERES: probes [N,6] / texture [R,R,3] / dims {Ht,Wt} ride the (otherwise
    # unused) dual-hashgrid kwargs — the probe kernel reads them under flag 0x1000
    # and returns dL/dprobes + dL/dtex through the same autograd slots.
    if _probe_tensors is not None:
        rasterizer_kwargs['features_diffuse'] = _probe_tensors[0]
        rasterizer_kwargs['gridrange_diffuse'] = _probe_tensors[1]
        rasterizer_kwargs['offsets_diffuse'] = _probe_tensors[2]
    # Other rasterizers (lean, fp16, etc.) still accept viewdirs_enc
    if viewdirs_enc is not None and not isinstance(rasterizer, GaussianRasterizer):
        rasterizer_kwargs['viewdirs_enc'] = viewdirs_enc
    # FastGS: only diff_surfel_3D_sh_res / diff_surfel_mixed accept metric_map.
    # NOTE: diff_surfel_3D_sh_filmres is a res-family clone whose name does NOT contain
    # the 'diff_surfel_3D_sh_res' substring (film·res, not _sh_res), so it must be listed
    # explicitly — otherwise metric_map is never passed → FastGS importance is all-zero →
    # every clone/split is gated off → densification silently disabled for filmres.
    _rasterizer_mod = getattr(type(rasterizer), '__module__', '') or ''
    if metric_map is not None and (
            'diff_surfel_3D_sh_res' in _rasterizer_mod
            or 'diff_surfel_3D_sh_filmres' in _rasterizer_mod
            or 'diff_surfel_mixed' in _rasterizer_mod):
        rasterizer_kwargs['metric_map'] = metric_map
    # `--method mixed[_3d]` / `--method res_3d`: pass per-Gauss textured/
    # untextured bool to the kernel. (`diff_surfel_mixed` substring also matches
    # `diff_surfel_mixed_3d`; res_3d is a separate fork.)
    _kernel_takes_is_textured = (
        'diff_surfel_mixed' in _rasterizer_mod
        or 'diff_surfel_res_3d' in _rasterizer_mod
        or 'diff_surfel_gestex' in _rasterizer_mod   # `--method GEStex` joint stage
    )
    if _kernel_takes_is_textured and hasattr(pc, '_is_textured') \
            and pc._is_textured.numel() > 0 and pc._is_textured.shape[0] == pc.get_xyz.shape[0]:
        rasterizer_kwargs['is_textured'] = pc._is_textured
    # EWA-3D-ellipsoid untextured: untextured surfels render as 3D ellipsoids —
    # pass the learnable 3rd axis as the ACTIVATED scale (exp), consistent with
    # `scales` (= pc.get_scaling, also activated). Both `mixed_3d` and `res_3d`
    # use this path.
    _kernel_takes_scaling_z = (
        'diff_surfel_mixed_3d' in _rasterizer_mod
        or 'diff_surfel_res_3d' in _rasterizer_mod
        or 'diff_surfel_gestex' in _rasterizer_mod   # `--method GEStex` joint stage
    )
    if _kernel_takes_scaling_z and hasattr(pc, '_scaling_z') \
            and pc._scaling_z.numel() > 0 and pc._scaling_z.shape[0] == pc.get_xyz.shape[0]:
        rasterizer_kwargs['scaling_z'] = pc.get_scaling_z

    # `--method film`: pass per-Gauss FiLM scale (gamma [N,1]) + bias (beta [N,24])
    # to the FiLM rasterizer. Empty / absent → kernel falls back to plain cat behaviour.
    if is_film_mode and 'diff_surfel_film' in _rasterizer_mod \
            and hasattr(pc, '_film_params') and pc._film_params.numel() > 0 \
            and pc._film_params.shape[0] == pc.get_xyz.shape[0]:
        rasterizer_kwargs['film_gamma'] = pc.get_film_gamma
        rasterizer_kwargs['film_beta'] = pc.get_film_beta

    # `--method 3D_SH_filmres`: pass per-Gauss gamma/beta to the filmres fork. Beta is the
    # full [N,24] slice; the kernel reads the first hash_dim<=16 channels with stride 24.
    if getattr(ingp, 'is_3D_SH_filmres_mode', False) and 'diff_surfel_3D_sh_filmres' in _rasterizer_mod \
            and hasattr(pc, '_film_params') and pc._film_params.numel() > 0 \
            and pc._film_params.shape[0] == pc.get_xyz.shape[0]:
        rasterizer_kwargs['film_gamma'] = pc.get_film_gamma
        rasterizer_kwargs['film_beta'] = pc.get_film_beta

    # `--method 3D_SH_concat`: pass the per-Gauss surfel latent to the concat fork. The kernel
    # reads cols 0..15 of the [N,24] beta slice as the latent half of [latent(16) | hash(16)].
    if getattr(ingp, 'is_3D_SH_concat_mode', False) and 'diff_surfel_3D_sh_concat' in _rasterizer_mod \
            and hasattr(pc, '_film_params') and pc._film_params.numel() > 0 \
            and pc._film_params.shape[0] == pc.get_xyz.shape[0] \
            and not locals().get('_concat_zero_latent', False):   # concat_hash decompose: zero latent
        rasterizer_kwargs['film_beta'] = pc.get_film_beta

    # `--method res_3d` post-split dispatch (TWO renders, mathematically
    # equivalent to a single-pass dual-cascade kernel — each render visits
    # one primitive subset, masked by opacity, so its T cascade is independent
    # of the other half). Render A: only 2D residual-carriers contribute
    # (3D opacity → 0). Render B: only 3D EWA SV-carriers contribute (2D
    # opacity → 0). Composed: image = C_sv + LRU(C_tex, α). The SV baseline
    # is also zeroed for 2D rows in render A so the textured contribution is
    # residual-only (mode 2 → feat = colors_precomp + residual + res_bias,
    # which equals residual + res_bias when colors_precomp = 0).
    _is_res_3d_post_split = (ingp is not None
                              and (getattr(ingp, 'is_res_3d_post_split', False)
                                   or getattr(ingp, 'is_res_3d_double_post_split', False))
                              and hasattr(pc, '_is_textured')
                              and pc._is_textured.numel() == pc.get_xyz.shape[0])
    _use_res_3d_single_pass = (_is_res_3d_post_split
                                and 'diff_surfel_res_3d' in _rasterizer_mod)
    if _use_res_3d_single_pass:
        # SINGLE-PASS dual-cascade kernel. Forward emits out_color =
        # C_sv_aux + C_tex_aux (signed sum); apply LRU/relu here, backward
        # routes per-Gauss color grads through dual T cascades.
        rasterizer_output = rasterizer(**rasterizer_kwargs)
        _image_signed = rasterizer_output[0]
        _alpha_lru = float(getattr(ingp, 'lru_slope', 0.0))
        if _alpha_lru > 0.0:
            _image_composed = torch.nn.functional.leaky_relu(_image_signed, negative_slope=_alpha_lru)
        else:
            _image_composed = torch.relu(_image_signed)
        rasterizer_output = (_image_composed,) + tuple(rasterizer_output[1:])
    elif _is_res_3d_post_split:
        # Two-render fallback when diff_surfel_res_3d isn't built. Each render
        # visits one primitive subset, masked by opacity, so its T cascade
        # is independent of the other half (mathematically equivalent to
        # single-pass but ~2× kernel cost).
        _orig_opacity = rasterizer_kwargs['opacities']                       # [N, 1]
        _orig_colors_precomp = rasterizer_kwargs['colors_precomp']           # [N, 3] or empty
        _tex_mask_f = pc._is_textured.float().unsqueeze(-1)                  # [N, 1]
        _untex_mask_f = 1.0 - _tex_mask_f                                    # [N, 1]

        # --- Render A: only 2D residual-carriers contribute (image_tex) ---
        # Mask 3D opacity to 0; zero SV baseline for 2D rows (so feat is
        # residual-only in mode 2). Bypass autograd on the gate by multiplying
        # (mask is constant 0/1).
        rasterizer_kwargs['opacities'] = _orig_opacity * _tex_mask_f
        if _orig_colors_precomp is not None and _orig_colors_precomp.numel() > 0 \
                and _orig_colors_precomp.shape[0] == pc.get_xyz.shape[0]:
            rasterizer_kwargs['colors_precomp'] = _orig_colors_precomp * _untex_mask_f
        rasterizer_output_tex = rasterizer(**rasterizer_kwargs)
        _image_tex = rasterizer_output_tex[0]

        # --- Render B: only 3D EWA SV-carriers contribute (image_sv) ---
        rasterizer_kwargs['opacities'] = _orig_opacity * _untex_mask_f
        rasterizer_kwargs['colors_precomp'] = _orig_colors_precomp  # SV intact
        rasterizer_output = rasterizer(**rasterizer_kwargs)
        _image_sv = rasterizer_output[0]

        # Compose: image = LRU(C_sv + C_tex, α). The LRU is on the SUM so that
        # signed C_tex (residual cascade) can SUBTRACT from C_sv (SV cascade)
        # to darken overbright SV regions — recovering the per-Gauss "residual
        # modulates SV" capability that pre-split mode 0 had, but at the
        # per-pixel post-blend site instead of per-Gauss. The LRU's negative
        # slope α controls how much the sum can leak below 0.
        _alpha_lru = float(getattr(ingp, 'lru_slope', 0.0))
        if _alpha_lru > 0.0:
            _image_composed = torch.nn.functional.leaky_relu(_image_sv + _image_tex, negative_slope=_alpha_lru)
        else:
            _image_composed = torch.relu(_image_sv + _image_tex)

        # Re-pack rasterizer_output with the composed image in slot 0. We
        # keep render B's geometry/depth/normal buffers for downstream regs
        # (densification uses SV-side radii primarily; depth/normal from the
        # SV render reflects the EWA primitives' geometry). The 2D-side
        # info is available via rasterizer_output_tex for future use.
        rasterizer_output = (_image_composed,) + tuple(rasterizer_output[1:])

        # Restore for safety (caller doesn't reuse, but explicit is better).
        rasterizer_kwargs['opacities'] = _orig_opacity
        rasterizer_kwargs['colors_precomp'] = _orig_colors_precomp
    else:
        rasterizer_output = rasterizer(**rasterizer_kwargs)
    # Main rasterizer returns 7 values (with max_weight, accum_weights); other rasterizers (lean, fp16, etc.) return 8 (with intersection_buffer, intersection_count, geomBuffer)
    # diff_surfel_3D_sh_res additionally appends out_index (max-contrib id) and
    # metric_counts (FastGS per-Gaussian counter) → 10 values.
    max_weight_buf = None
    accum_weights_buf = None
    max_contrib_idx = None
    metric_counts = None
    # `--l2` (mixed_3d only): the rasterizer surfaces a second image-output
    # slot (`rendered_image_untex`) — a clone of `rendered_image` but a
    # separate autograd node so the backward can receive two upstream image
    # gradients (one per loss) and route per-Gauss inside CUDA.
    rendered_image_untex = None
    if len(rasterizer_output) == 11:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer, max_contrib_idx, metric_counts, rendered_image_untex = rasterizer_output
    elif len(rasterizer_output) == 10:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer, max_contrib_idx, metric_counts = rasterizer_output
    elif len(rasterizer_output) == 9:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer, max_contrib_idx = rasterizer_output
    elif len(rasterizer_output) == 8:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer = rasterizer_output
    elif len(rasterizer_output) == 7:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, max_weight_buf, accum_weights_buf = rasterizer_output
        intersection_buffer = None
        intersection_count = None
        geomBuffer = None
    elif len(rasterizer_output) == 6:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, geomBuffer = rasterizer_output
        intersection_buffer = None
        intersection_count = None
    else:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels = rasterizer_output
        intersection_buffer = None
        intersection_count = None
        geomBuffer = None

    # `--method mixed`: per-pixel ReLU on the final blended color. The CUDA kernel
    # (residual_mode=2) leaves the per-Gauss feat signed (ReLU(SV)+residual, no
    # per-Gauss clamp), so the blended pixel can go negative — clamp it here.
    # PyTorch autograd handles the per-pixel ReLU gate for the backward pass, so
    # dL_dpixel arrives at the rasterizer already correctly gated.
    # Pre-clamp blended image. For `--method mixed[_3d]` the per-Gauss feat is
    # signed (ReLU(SV)+residual), so the blended pixel can go negative; the
    # final per-pixel ReLU below hides those negative regions. `render_raw`
    # keeps the signed pre-ReLU image so decomposition viz can show the abs /
    # signed residual magnitude (negative texture that the clamp would erase).
    rendered_image_raw = rendered_image
    # Per-pixel ReLU after blend: only applied for the deferred-clamp variants
    # (`mixed_sep` / `mixed_3d_sep`, residual_mode=2). Bare `mixed`/`mixed_3d`
    # use mode 0 (per-Gauss outer ReLU before the blend) — the blended pixel is
    # already non-negative, so the relu would be a no-op and is skipped to
    # match 3D_SH_res byte-for-byte.
    # `--ste`: when enabled AND in deferred-clamp mode, use sign-aware
    # straight-through ReLU instead of the plain torch.relu — gradient also
    # flows at clamped pixels where grad_out < 0 (release-clamp direction),
    # rescuing the "many texture queries miss gradient at clamped pixels" case.
    # 'tex_only_raw' returns the SIGNED blended residual sum(w_i * residual_i)
    # WITHOUT the mode-2 per-pixel clamp — that sum is linear in the residual and
    # is the distillation target/prediction (--probe_distill_dir). Plain 'tex_only'
    # keeps the clamp so existing --decomp supervision is byte-identical.
    if (ingp is not None and getattr(ingp, 'is_mixed_deferred_relu_mode', False)
            and decompose_mode != 'tex_only_raw'):
        _lru_alpha = float(getattr(ingp, 'lru_slope', 0.0))
        if getattr(ingp, 'is_ste_relu', False):
            rendered_image = STERelu.apply(rendered_image)
        elif _lru_alpha != 0.0:
            # `--lru` α: leaky-ReLU at the per-pixel after-blend clamp (mode 2).
            # autograd handles forward + backward naturally.
            rendered_image = torch.nn.functional.leaky_relu(rendered_image, negative_slope=_lru_alpha)
        else:
            rendered_image = torch.relu(rendered_image)

    # 3D mode: Process intersection buffer through PyTorch pipeline
    # Recompute xyz from s_x,s_y → hash encode → gather features → MLP → SH → blend → eval
    if is_3D_mode and ingp is not None and intersection_buffer is not None:
        from utils.general_utils import build_rotation
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

        # intersection_buffer: [H*W * max_per_pixel, 9]
        # Format: (gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G)
        # T and G are stored directly from CUDA for correct gradient computation
        # intersection_count: [H*W] - number of valid intersections per pixel

        # Create mask for valid intersections (non-padded entries)
        counts = intersection_count  # [H*W]
        total_slots = intersection_buffer.shape[0]  # H*W * max_per_pixel

        # Build valid mask by checking if each slot index < count for its pixel
        slot_indices = torch.arange(total_slots, device="cuda")
        local_indices = slot_indices % max_intersections_per_pixel
        pixel_ids_from_slot = slot_indices // max_intersections_per_pixel
        valid_mask = local_indices < counts[pixel_ids_from_slot]

        # Extract valid intersections
        valid_buffer = intersection_buffer[valid_mask]  # [M, 9] where M = total valid
        total_M = valid_buffer.shape[0]

        if total_M > 0:
            # Unpack intersection data (12-field layout)
            # [gaussian_id, weight, pixel_id, xyz.x, xyz.y, xyz.z, s_x, s_y, rho_flag, alpha, T, G]
            gaussian_ids = valid_buffer[:, 0].view(torch.int32).long()  # [M]
            weights_raw = valid_buffer[:, 1].contiguous().float()       # [M] weight = alpha * T
            pixel_ids = valid_buffer[:, 2].view(torch.int32).long()     # [M]
            xyz_x = valid_buffer[:, 3].contiguous().float()             # [M] xyz.x from CUDA
            xyz_y = valid_buffer[:, 4].contiguous().float()             # [M] xyz.y from CUDA
            xyz_z = valid_buffer[:, 5].contiguous().float()             # [M] xyz.z from CUDA
            s_x = valid_buffer[:, 6].contiguous().float()               # [M] disk coord for backward
            s_y = valid_buffer[:, 7].contiguous().float()               # [M] disk coord for backward
            rho_flag = valid_buffer[:, 8].contiguous().float()          # [M] 1.0=disk, 0.0=center
            alpha_all = valid_buffer[:, 9].contiguous().float()         # [M] alpha = opacity * G
            T_all = valid_buffer[:, 10].contiguous().float()            # [M] transmittance (from buffer)
            G_all = valid_buffer[:, 11].contiguous().float()            # [M] kernel value (from buffer)

            # Get Gaussian parameters (needed for xyz recomputation)
            means3D = pc.get_xyz           # [N, 3]
            scales = pc.get_scaling        # [N, 2] - activated
            # CRITICAL: Must use get_rotation (normalized), not _rotation (raw)!
            # The CUDA transMat_to_scale_rot_grad kernel expects normalized quaternions
            # to match native CAT mode's preprocessCUDA which normalizes before computing R.
            rotations = pc.get_rotation    # [N, 4] - normalized quaternions
            opacity_activated = pc.get_opacity.squeeze(-1)  # [N]
            gaussian_features = pc.get_gaussian_features    # [N, gauss_dim]

            # Precompute rotation matrices for all Gaussians
            rotation_matrices = build_rotation(rotations)  # [N, 3, 3]

            # Get camera center for view direction computation
            camera_center = viewpoint_camera.camera_center  # [3]

            # IMPORTANT: Connect weights to opacity, scale, rotation, AND position gradient ONCE for ALL intersections
            # This ensures the transmittance chain effect is computed correctly across all pixels
            # Uses unified backward_from_weight_grad kernel that reads transMat from geomBuffer
            # NOW ALSO computes xyz so that grad_xyz flows back from hash/MLP chain (matching cat mode's dL_duv path)
            # Returns (weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad)
            # rotation_with_grad is now quaternion gradients [N, 4] for proper coordinate conversion
            projmatrix = viewpoint_camera.full_proj_transform  # [4, 4] for proper coordinate conversion
            viewmatrix = viewpoint_camera.world_view_transform  # [4, 4] for normal gradient transform
            weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad = IntersectionOpacityGrad.apply(
                opacity_activated, scales, rotations, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weights_raw, T_all, G_all, alpha_all, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
            )

            # Initialize accumulators for batched processing
            # NOTE: Blend RGB, not SH! SH evaluation must happen per-intersection.
            blended_rgb = torch.zeros(H * W, 3, device="cuda", dtype=torch.float32)
            total_weight = torch.zeros(H * W, device="cuda", dtype=torch.float32)

            # Process MLP/hash in batches to avoid OOM
            # The gradient connection is already set up above for ALL intersections
            batch_size = 5000000  # Tune based on VRAM (matched to 3D_direct)
            for start_idx in range(0, total_M, batch_size):
                end_idx = min(start_idx + batch_size, total_M)

                # Slice batch
                b_gids = gaussian_ids[start_idx:end_idx]
                b_weights = weights_with_grad[start_idx:end_idx]  # Use weights with grad from single call
                b_pids = pixel_ids[start_idx:end_idx]

                # Use xyz computed in IntersectionOpacityGrad (with gradients for hash/MLP chain)
                # This allows grad_xyz to flow back through the hash → MLP → SH → RGB → loss path
                # IntersectionOpacityGrad.backward will capture grad_xyz and compute dL_duv for the CUDA kernel
                b_xyz = xyz_with_grad[start_idx:end_idx]  # [B, 3] - has gradient for hash/MLP chain

                # Hash encoding (batched tcnn query)
                # If detach_hash_grad is set, detach xyz before hash query to prevent
                # hash gradients from flowing to position (matches CUDA behavior in cat mode)
                if ingp.hash_encoding is not None:
                    b_xyz_for_hash = b_xyz.detach() if detach_hash_grad else b_xyz
                    b_hash = ingp._encode_3D(b_xyz_for_hash).float()  # [B, hash_dim]
                else:
                    b_hash = torch.zeros(end_idx - start_idx, 0, device="cuda", dtype=torch.float32)

                # Per-Gaussian features
                b_gauss = gaussian_features[b_gids].float()  # [B, gauss_dim]

                # Apply decompose_mode for debugging visualization (3D mode)
                if decompose_mode == 'gaussian_only':
                    b_hash = torch.zeros_like(b_hash)  # Zero out hash features
                elif decompose_mode == 'ngp_only':
                    b_gauss = torch.zeros_like(b_gauss)  # Zero out Gaussian features

                # Concatenate and MLP → SH coefficients
                if b_hash.shape[1] > 0:
                    b_combined = torch.cat([b_gauss, b_hash], dim=-1)
                else:
                    b_combined = b_gauss
                b_sh = ingp.mlp_3D(b_combined).float().view(-1, 3, 16)  # [B, 3, 16]

                # Compute view direction: from xyz to camera (matches CUDA convention)
                b_viewdir = torch_F.normalize(b_xyz - camera_center[None, :], dim=-1)  # [B, 3]

                # Evaluate SH per intersection to get RGB (correct order: eval first, blend after)
                b_rgb = eval_sh(3, b_sh, b_viewdir)  # [B, 3]

                # Add SH DC offset and clamp per-intersection (matches CUDA behavior)
                b_rgb = torch.clamp(b_rgb + 0.5, 0.0, None)  # [B, 3]

                # Weighted accumulation of RGB (not SH!)
                b_weights_f32 = b_weights.float()
                b_weighted_rgb = b_rgb * b_weights_f32[:, None]
                blended_rgb.scatter_add_(0, b_pids[:, None].expand_as(b_weighted_rgb), b_weighted_rgb)
                total_weight.scatter_add_(0, b_pids, b_weights_f32)

            # Alpha compositing with background (no +0.5 needed since we added it per-intersection)
            T_final = 1.0 - total_weight
            rgb = blended_rgb + T_final[:, None] * bg_color[None, :].float()
            rgb = torch.clamp(rgb, 0.0, 1.0)

            rendered_image = rgb.view(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        else:
            # No valid intersections - return background
            rendered_image = bg_color.unsqueeze(-1).unsqueeze(-1).expand(3, H, W)

    # 3D_direct mode: Same intersection buffer processing, but direct RGB output (like cat mode's 2D MLP)
    # Uses view encoding instead of SH for view-dependent effects
    if is_3D_direct_mode and ingp is not None and intersection_buffer is not None:
        from utils.general_utils import build_rotation
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

        # intersection_buffer: [H*W * max_per_pixel, 9]
        # Format: (gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G)
        # T and G are stored directly from CUDA for correct gradient computation
        # intersection_count: [H*W] - number of valid intersections per pixel

        # Create mask for valid intersections (non-padded entries)
        counts = intersection_count  # [H*W]
        total_slots = intersection_buffer.shape[0]  # H*W * max_per_pixel

        # Build valid mask by checking if each slot index < count for its pixel
        slot_indices = torch.arange(total_slots, device="cuda")
        local_indices = slot_indices % max_intersections_per_pixel
        pixel_ids_from_slot = slot_indices // max_intersections_per_pixel
        valid_mask = local_indices < counts[pixel_ids_from_slot]

        # Extract valid intersections
        valid_buffer = intersection_buffer[valid_mask]  # [M, 9] where M = total valid
        total_M = valid_buffer.shape[0]

        if total_M > 0:
            # Unpack intersection data (12-field layout)
            # [gaussian_id, weight, pixel_id, xyz.x, xyz.y, xyz.z, s_x, s_y, rho_flag, alpha, T, G]
            gaussian_ids = valid_buffer[:, 0].view(torch.int32).long()  # [M]
            weights_raw = valid_buffer[:, 1].contiguous().float()       # [M] weight = alpha * T
            pixel_ids = valid_buffer[:, 2].view(torch.int32).long()     # [M]
            xyz_x = valid_buffer[:, 3].contiguous().float()             # [M] xyz.x from CUDA
            xyz_y = valid_buffer[:, 4].contiguous().float()             # [M] xyz.y from CUDA
            xyz_z = valid_buffer[:, 5].contiguous().float()             # [M] xyz.z from CUDA
            s_x = valid_buffer[:, 6].contiguous().float()               # [M] disk coord for backward
            s_y = valid_buffer[:, 7].contiguous().float()               # [M] disk coord for backward
            rho_flag = valid_buffer[:, 8].contiguous().float()          # [M] 1.0=disk, 0.0=center
            alpha_all = valid_buffer[:, 9].contiguous().float()         # [M] alpha = opacity * G
            T_all = valid_buffer[:, 10].contiguous().float()            # [M] transmittance (from buffer)
            G_all = valid_buffer[:, 11].contiguous().float()            # [M] kernel value (from buffer)

            # Get Gaussian parameters (needed for xyz recomputation)
            means3D = pc.get_xyz           # [N, 3]
            scales = pc.get_scaling        # [N, 2] - activated
            # CRITICAL: Must use get_rotation (normalized), not _rotation (raw)!
            # The CUDA transMat_to_scale_rot_grad kernel expects normalized quaternions
            # to match native CAT mode's preprocessCUDA which normalizes before computing R.
            rotations = pc.get_rotation    # [N, 4] - normalized quaternions
            opacity_activated = pc.get_opacity.squeeze(-1)  # [N]
            gaussian_features = pc.get_gaussian_features    # [N, gauss_dim]

            # Precompute rotation matrices for all Gaussians
            rotation_matrices = build_rotation(rotations)  # [N, 3, 3]

            # Get camera center for view direction computation
            camera_center = viewpoint_camera.camera_center  # [3]

            # IMPORTANT: Connect weights to opacity, scale, rotation, AND position gradient ONCE for ALL intersections
            # This ensures the transmittance chain effect is computed correctly across all pixels
            # Uses unified backward_from_weight_grad kernel that reads transMat from geomBuffer
            # NOW ALSO computes xyz so that grad_xyz flows back from hash/MLP chain (matching cat mode's dL_duv path)
            # Returns (weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad)
            # rotation_with_grad is now quaternion gradients [N, 4] for proper coordinate conversion
            projmatrix = viewpoint_camera.full_proj_transform  # [4, 4] for proper coordinate conversion
            viewmatrix = viewpoint_camera.world_view_transform  # [4, 4] for normal gradient transform
            weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad = IntersectionOpacityGrad.apply(
                opacity_activated, scales, rotations, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weights_raw, T_all, G_all, alpha_all, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
            )

            # Initialize accumulators for batched processing
            blended_rgb = torch.zeros(H * W, 3, device="cuda", dtype=torch.float32)
            total_weight = torch.zeros(H * W, device="cuda", dtype=torch.float32)

            # Compute per-pixel ray directions ONCE (same as CAT mode)
            # This ensures view encoding is identical to CAT mode: per-pixel, no xyz gradient
            rays_d, _ = cam2rays(viewpoint_camera)  # [H*W, 3]
            ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # [H*W, 3] normalized

            # Process MLP/hash in batches to avoid OOM
            # The gradient connection is already set up above for ALL intersections
            batch_size = 5000000  # Tune based on VRAM
            for start_idx in range(0, total_M, batch_size):
                end_idx = min(start_idx + batch_size, total_M)

                # Slice batch
                b_gids = gaussian_ids[start_idx:end_idx]
                b_weights = weights_with_grad[start_idx:end_idx]  # Use weights with grad from single call
                b_pids = pixel_ids[start_idx:end_idx]

                # Use xyz computed in IntersectionOpacityGrad (with gradients for hash/MLP chain)
                # This allows grad_xyz to flow back through the hash → MLP → RGB → loss path
                # IntersectionOpacityGrad.backward will capture grad_xyz and compute dL_duv for the CUDA kernel
                b_xyz = xyz_with_grad[start_idx:end_idx]  # [B, 3] - has gradient for hash/MLP chain

                # Hash encoding (batched tcnn query)
                # If detach_hash_grad is set, detach xyz before hash query to prevent
                # hash gradients from flowing to position (matches CUDA behavior in cat mode)
                if ingp.hash_encoding is not None:
                    b_xyz_for_hash = b_xyz.detach() if detach_hash_grad else b_xyz
                    b_hash = ingp._encode_3D(b_xyz_for_hash).float()  # [B, hash_dim]
                else:
                    b_hash = torch.zeros(end_idx - start_idx, 0, device="cuda", dtype=torch.float32)

                # Per-Gaussian features
                b_gauss = gaussian_features[b_gids].float()  # [B, gauss_dim]

                # View direction: use per-pixel ray_unit (same as CAT mode)
                # Index by pixel_id to get the ray direction for each intersection
                # This is per-PIXEL (same for all intersections at a pixel), no gradient to xyz
                b_viewdir = ray_unit[b_pids]  # [B, 3]

                # Encode view direction (same as cat mode's 2D MLP)
                b_viewdir_enc = ingp._encode_view(b_viewdir)  # [B, view_enc_dim]

                # Apply decompose_mode for debugging visualization (3D_direct mode)
                if decompose_mode == 'gaussian_only':
                    b_hash = torch.zeros_like(b_hash)  # Zero out hash features
                elif decompose_mode == 'ngp_only':
                    b_gauss = torch.zeros_like(b_gauss)  # Zero out Gaussian features

                # Concatenate: [per-Gaussian features | hash features | view encoding]
                if b_hash.shape[1] > 0:
                    b_combined = torch.cat([b_gauss, b_hash, b_viewdir_enc], dim=-1)
                else:
                    b_combined = torch.cat([b_gauss, b_viewdir_enc], dim=-1)

                # Query MLP for RGB directly (no SH step)
                b_rgb = torch.sigmoid(ingp.mlp_3D_direct(b_combined).float())  # [B, 3]

                # Weighted accumulation of RGB
                b_weights_f32 = b_weights.float()
                b_weighted_rgb = b_rgb * b_weights_f32[:, None]
                blended_rgb.scatter_add_(0, b_pids[:, None].expand_as(b_weighted_rgb), b_weighted_rgb)
                total_weight.scatter_add_(0, b_pids, b_weights_f32)

            # Alpha compositing with background
            T_final = 1.0 - total_weight
            rgb = blended_rgb + T_final[:, None] * bg_color[None, :].float()
            rgb = torch.clamp(rgb, 0.0, 1.0)

            rendered_image = rgb.view(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        else:
            # No valid intersections - return background
            rendered_image = bg_color.unsqueeze(-1).unsqueeze(-1).expand(3, H, W)

    # additional regularizations
    render_alpha = allmap[1:2]

    # Fast inference mode: skip expensive post-processing (normals, etc.)
    # But still do MLP decode if needed (cat mode returns features, not RGB)
    if fast_inference:
        # Check if MLP decode is needed (rendered_image has >3 channels = features)
        if rendered_image.shape[0] > 3 and ingp is not None:
            H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
            feat_dim = rendered_image.shape[0]
            rays_d, _ = cam2rays(viewpoint_camera)
            ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # Normalize ray directions
            fg_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)
            rendered_image = ingp.rgb_decode(fg_features, ray_unit)  # (H*W, 3)
            rendered_image = rendered_image.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
            # Apply render_mask to zero out pixels with no Gaussian coverage
            render_mask = (render_alpha > 0)
            rendered_image = rendered_image * render_mask

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "render_alpha": render_alpha,
            "render_depth": allmap[0:1],  # expected depth (unnormalized)
        }

    # `--skip_aux_normal_dist` (pipe flag): skip the expensive Python-side
    # normal/dist intermediates when no consumer (lambda_normal, w_normal,
    # lambda_dist) is active. At 4K image res this avoids ~600 MB per call
    # for dx, dy, cross product, and the 3-ch viewmatrix rotate.
    _skip_aux = bool(getattr(pipe, 'skip_aux_normal_dist', False))

    # Sizes for the optional zero placeholders below.
    _N = allmap.shape[0]
    _H, _W = allmap.shape[1], allmap.shape[2]
    _zero_hw = torch.zeros(1, _H, _W, device=allmap.device, dtype=allmap.dtype)

    # get normal map. transform normal from view space to world space.
    if _skip_aux:
        render_normal = torch.zeros(3, _H, _W, device=allmap.device, dtype=allmap.dtype)
    else:
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    # get median depth map. `--method res_3d_paired` SLIM out_others has only
    # 5 channels (DEPTH+ALPHA+NORMAL) — every read below is size-gated and
    # falls back to a zero tensor of matching [1, H, W] shape when the channel
    # is absent. Downstream consumers (mini reinit, --lambda_dist, etc.) must
    # not require these for res_3d_paired runs with this build.
    render_depth_median = allmap[5:6] if _N >= 6 else _zero_hw
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7] if _N >= 7 else _zero_hw

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    if _skip_aux:
        surf_normal = torch.zeros(3, _H, _W, device=allmap.device, dtype=allmap.dtype)
    else:
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

    # surf_normal = render_normal

    # get contributed gaussians per pixel
    render_gs_nums = allmap[7:8] if _N >= 8 else _zero_hw

    # get overdraw map (soft contributor count from sigmoid relaxation)
    render_overdraw = allmap[14:15] if _N >= 15 else _zero_hw

    # get max-contributor depth (intersection depth of Gaussian with highest alpha*T per pixel)
    render_depth_max_contributor = allmap[15:16] if _N >= 16 else _zero_hw

    # get w² sum (sum of squared weights per pixel, for weight_reg loss)
    render_w_square = allmap[16:17] if _N >= 17 else _zero_hw

    # per-pixel sum of w_i * beta_i (--w_lambda_perpix shape reg). Only meaningful
    # for beta-supporting kernels (beta / beta_scaled / general / flex). For scalar
    # Gaussian kernels, shape values are 0 → beta_sum is 0 everywhere → no-op.
    render_beta_sum = allmap[17:18] if allmap.shape[0] >= 18 else None

    # Diffuse_ngp mode: unproject median depth, query hashgrid, add to diffuse RGB
    gaussian_rgb_diffuse_ngp = None
    ngp_rgb_diffuse_ngp = None
    if is_diffuse_ngp_mode and ingp is not None:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        
        # Store Gaussian RGB (diffuse SH) before adding NGP contribution
        render_mask = (render_alpha > 0)
        gaussian_rgb_diffuse_ngp = rendered_image.clone()
        
        # Unproject median depth to 3D points (detached - no gradient through depth)
        points_3d, rays_d, rays_o = depths_to_points(viewpoint_camera, render_depth_median.detach())
        # points_3d: (H*W, 3), rays_d: (H*W, 3)
        
        # Query hashgrid at unprojected 3D points
        hash_features = ingp(points_3D=points_3d, with_xyz=False).float()  # (H*W, feat_dim)
        
        # Get view direction for MLP
        ray_unit = torch_F.normalize(rays_d, dim=-1).float()
        
        # Decode through MLP to get view-dependent RGB
        ngp_rgb = ingp.rgb_decode(hash_features, ray_unit)  # (H*W, 3)
        ngp_rgb = ngp_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        
        # Apply alpha mask
        ngp_rgb = ngp_rgb * render_mask
        
        # Store NGP RGB separately
        ngp_rgb_diffuse_ngp = ngp_rgb.clone()
        
        # Add NGP contribution to diffuse SH RGB
        rendered_image = rendered_image + ngp_rgb
    
    # Diffuse_offset mode: use rendered diffuse SH as xyz offset, query hashgrid at offset position
    # Implements "Scout and Squad" strategy for clean gradient flow
    scout_loss_data = None
    if is_diffuse_offset_mode and ingp is not None:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        render_mask = (render_alpha > 0)
        
        # rendered_image contains the diffuse SH output (3, H, W) - this is the offset (Delta_P)
        # SH to RGB conversion is: rgb = sh * 0.28209 + 0.5, so we need to subtract 0.5 to center at 0
        # Reshape to (H*W, 3) for adding to xyz
        offset_3d_raw = rendered_image.permute(1, 2, 0).reshape(-1, 3) - 0.5  # (H*W, 3), centered at 0
        # Clamp offset to [-0.1, 0.1] to prevent large displacements
        offset_3d = torch.clamp(offset_3d_raw, -0.1, 0.1)
        
        # Store the offset for visualization (unclamped, but centered)
        gaussian_rgb_diffuse_ngp = rendered_image.clone() - 0.5
        
        if use_xyz_mode:
            # XYZ MODE: Use rasterized xyz directly from allmap[8:11]
            
            # P_base: rasterized Gaussian positions (alpha-blended)
            render_xyz = allmap[8:11]  # (3, H, W)
            
            # For RGB loss: detach P_base so gradients only flow through offset
            points_3d_base_detached = render_xyz.permute(1, 2, 0).reshape(-1, 3).detach()  # (H*W, 3)
            
            # P_query = P_base.detach() + Delta_P (offset has grads for RGB loss)
            points_3d_query = points_3d_base_detached + offset_3d  # (H*W, 3)
            
            # For scout loss: DON'T detach P_base - we need gradients to move Gaussians
            # scout_loss = MSE(P_base, P_target) where P_target = (P_base + offset).detach()
            # This pulls Gaussians toward where the offset found the surface
            points_3d_base_with_grad = render_xyz.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3), has grads
            points_3d_target = points_3d_query.detach()  # (H*W, 3), detached
            
            scout_loss_data = {
                'points_base': points_3d_base_with_grad,  # Has gradients for Gaussian positions
                'points_target': points_3d_target,  # Detached target
                'render_mask': render_mask,  # Only compute loss where alpha > 0
            }
            
            # Get view direction from camera
            rays_d, rays_o = cam2rays(viewpoint_camera)
        else:
            # DEPTH MODE: Unproject median depth to 3D points
            points_3d, rays_d, rays_o = depths_to_points(viewpoint_camera, render_depth_median.detach())
            # points_3d: (H*W, 3), rays_d: (H*W, 3)
            
            # Add offset to unprojected xyz (offset has gradients, points_3d is detached)
            points_3d_query = points_3d.detach() + offset_3d  # (H*W, 3)
        
        # Query hashgrid at query points
        hash_features = ingp(points_3D=points_3d_query, with_xyz=False).float()  # (H*W, feat_dim)
        
        # Get view direction for MLP
        ray_unit = torch_F.normalize(rays_d, dim=-1).float()
        
        # Decode through MLP to get final RGB
        final_rgb = ingp.rgb_decode(hash_features, ray_unit)  # (H*W, 3)
        final_rgb = final_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        
        # Apply alpha mask
        final_rgb = final_rgb * render_mask
        
        # Store NGP RGB separately (this is the final output in diffuse_offset mode)
        ngp_rgb_diffuse_ngp = final_rgb.clone()
        
        # Final RGB is purely from hashgrid MLP (not additive like diffuse_ngp)
        rendered_image = final_rgb
    
    # Save raw features before MLP decode (if requested)
    raw_features_saved = None
    if return_raw_features and hash_in_CUDA:
        raw_features_saved = rendered_image.clone()  # (feat_dim, H, W)

    # Diffuse/Specular mode: no MLP decoding needed, rendered_image is already RGB from SH
    # Baseline mode: decode hashgrid features through MLP
    # ONLY when hash_in_CUDA is True (not during warmup SH rendering)
    elif ingp is not None and hash_in_CUDA and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode and not is_3D_direct_fused_mode:
        # Hashgrid rendering active:
        rays_d, rays_o = cam2rays(viewpoint_camera)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        ray_unit = torch_F.normalize(rays_d, dim=-1).float().detach()
        normals = render_normal.view(3, -1).permute(1, 0)
        normals_unit = torch_F.normalize(normals, dim=-1).float().detach()

        d_dot_n = torch.sum(ray_unit * normals_unit, dim=-1, keepdim=True)
        ray_out = ray_unit - 2.0 * d_dot_n * normals_unit
        
        ray_out_norm = torch.norm(ray_out, dim = -1)

        feat_dim = rendered_image.shape[0]

        render_mask = (render_alpha > 0)

        rays_dir = ray_unit
        try:
            if cfg.settings.dir_out:
                rays_dir = ray_out
        except:
            pass

        ray_map = rays_dir.view(H, W, -1).permute(2, 0, 1).abs()
        ray_map = ray_map * render_mask

        feature_vis = rendered_image[:3].detach().abs()
        
        # Cat mode decomposition: mask out features for visualization
        if decompose_mode is not None and is_cat_mode and hybrid_levels > 0:
            per_level_dim = ingp.level_dim
            gaussian_feat_dim = hybrid_levels * per_level_dim
            hashgrid_feat_dim = (ingp.levels - hybrid_levels) * per_level_dim
            
            if decompose_mode == 'gaussian_only':
                # Zero out hashgrid features (last hashgrid_feat_dim channels)
                rendered_image[gaussian_feat_dim:, :, :] = 0
            elif decompose_mode == 'ngp_only':
                # Zero out per-Gaussian features (first gaussian_feat_dim channels)
                rendered_image[:gaussian_feat_dim, :, :] = 0
        
        if skip_mlp:
            # Skip MLP decode, just return zeros (for benchmarking rasterizer only)
            rendered_image = torch.zeros(3, H, W, device="cuda")
        elif background_mode == "skybox_sparse":
            # SPARSE MLP: Only decode pixels with Gaussian coverage
            # This avoids running MLP on sky/background pixels that will be replaced by skybox
            valid_mask = render_mask.squeeze()  # (H, W) bool
            n_valid = valid_mask.sum().item()

            if n_valid > 0:
                # Gather valid features and directions
                features_flat = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)
                valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]  # (D,) indices
                valid_features = features_flat[valid_indices]  # (D, F)
                valid_dirs = rays_dir[valid_indices]  # (D, 3)

                # MLP decode only valid pixels
                valid_rgb = ingp.rgb_decode(valid_features, valid_dirs)  # (D, 3)

                # Scatter back to full image
                rendered_image = torch.zeros(H * W, 3, device="cuda")
                rendered_image[valid_indices] = valid_rgb
                rendered_image = rendered_image.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
            else:
                # No valid pixels, return zeros
                rendered_image = torch.zeros(3, H, W, device="cuda")
        else:
            # DENSE MLP: Decode all pixels (standard path)
            fg_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)

            # Background hashgrid compositing
            if bg_hashgrid is not None:
                # Query background hashgrid at ray directions with camera position
                ray_unit_bg = torch_F.normalize(rays_dir, dim=-1).float()  # (H*W, 3)
                # rays_o is (3,) camera position, expand to (H*W, 3)
                ray_origins_bg = rays_o.unsqueeze(0).expand(ray_unit_bg.shape[0], -1)
                bg_features = bg_hashgrid(ray_unit_bg, ray_origins_bg)  # (H*W, F)

                # Get alpha for compositing (H*W, 1)
                alpha_flat = render_alpha.view(-1, 1)  # (H*W, 1)

                if background_mode == "hashgrid_sep":
                    # SEPARATE DECODE: Decode FG and BG separately, composite in RGB space
                    # This prevents BG from learning subtractive features
                    fg_rgb = ingp.rgb_decode(fg_features, rays_dir)  # (H*W, 3)
                    bg_rgb = ingp.rgb_decode(bg_features, rays_dir)  # (H*W, 3)
                    # Composite in RGB space: FG is already alpha-premultiplied from rasterizer
                    # Use alpha for compositing, NOT render_mask (BG fills empty regions)
                    rendered_image = fg_rgb + (1.0 - alpha_flat) * bg_rgb
                elif background_mode == "hashgrid_relu":
                    # RELU: Apply ReLU to BG features to prevent subtractive features
                    bg_features = torch_F.relu(bg_features)
                    composite_features = fg_features + (1.0 - alpha_flat) * bg_features
                    rendered_image = ingp.rgb_decode(composite_features, rays_dir)
                else:
                    # STANDARD: Composite features before MLP decode (original behavior)
                    composite_features = fg_features + (1.0 - alpha_flat) * bg_features
                    rendered_image = ingp.rgb_decode(composite_features, rays_dir)

                # BG hashgrid handles empty regions via (1-alpha) compositing
                # Do NOT apply render_mask here - it would zero out BG contribution
                rendered_image = rendered_image.view(H, W, -1).permute(2, 0, 1)
            else:
                # Standard path without background hashgrid
                rendered_image = ingp.rgb_decode(fg_features, rays_dir)
                rendered_image = rendered_image.view(H, W, -1).permute(2, 0, 1)
                rendered_image = rendered_image * render_mask

        vis_appearance_level = allmap[11:14] if allmap.shape[0] >= 14 else torch.zeros(
            3, allmap.shape[1], allmap.shape[2], device=allmap.device, dtype=allmap.dtype)

    # Background compositing: skybox or solid color (legacy path for skybox texture)
    # Initialize FG/BG outputs for visualization
    rendered_image_fg = None
    skybox_rgb = None

    # Skip background compositing for 3D/3D_direct modes - they already added background in Python
    if not is_3D_mode and not is_3D_direct_mode:
        if skybox is not None:
            # Compute ray directions if not already available
            H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
            rays_d, rays_o = cam2rays(viewpoint_camera)
            ray_unit_skybox = torch_F.normalize(rays_d, dim=-1).float()

            # Query skybox for background colors
            skybox_rgb = skybox(ray_unit_skybox)  # (H*W, 3)
            skybox_rgb = skybox_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)

            # Save foreground before compositing (for visualization)
            rendered_image_fg = rendered_image.clone()

            # Composite: foreground + (1-alpha) * skybox
            rendered_image = rendered_image + (1.0 - render_alpha) * skybox_rgb
        elif bg_color.sum() > 0:
            rendered_image = rendered_image + (1.0 - render_alpha) * bg_color.unsqueeze(-1).unsqueeze(-1)

    # Restore activation biases after decomposition render. MUST target the same module the
    # decompose set the bias on (concat set it on the concat module, else the SH stays killed).
    if '_restore_bias' in dir() and _restore_bias:
        if is_3D_SH_concat_mode:
            from diff_surfel_3D_sh_concat import set_activation_bias as _restore_set_activation_bias
        elif is_3D_SH_32_mode:
            from diff_surfel_3D_sh_32 import set_activation_bias as _restore_set_activation_bias
        else:
            _restore_set_activation_bias = _sh_res_setter_mod(ingp).set_activation_bias
        _restore_set_activation_bias(sh_bias=_ACTIVATION_BIAS[0], res_bias=_ACTIVATION_BIAS[1])

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "render_raw": rendered_image_raw,  # pre-ReLU (signed) — mixed[_3d] decomposition viz
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    if record_transmittance:
        rets.update({
            'transmittance_avg': transmittance_avg,
            'cover_pixels': num_covered_pixels,
        })
        if max_weight_buf is not None:
            rets['max_weight'] = max_weight_buf
        if accum_weights_buf is not None:
            rets['accum_weights'] = accum_weights_buf

    # Unbiased Depth: convergence-loss per-pixel scalar (only present when the
    # diff_surfel_3D_sh_res_unbiased rasterizer is active — its allmap has 19
    # channels, channel 18 = CONVERGE_OFFSET; the legacy rasterizer has 18).
    if allmap.shape[0] >= 19:
        rets['converge'] = allmap[18:19]

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'gaussian_num' : render_gs_nums,
            'depth_expected': render_depth_expected,
            'depth_median': render_depth_median,
            'depth_max_contributor': render_depth_max_contributor,
            'render_w_square': render_w_square,
            'render_overdraw': render_overdraw,
            'render_beta_sum': render_beta_sum,
            # int32 [H, W] per-pixel id of the max-weight Gaussian (-1 if none).
            # Used by the mini depth-reinit SH-transfer path. Only populated by
            # rasterizers that thread out_index through (currently diff_surfel_3D_sh_res).
            'max_contrib_idx': max_contrib_idx,
            # int32 [P] FastGS VCD/VCP per-Gaussian counter: number of high-error
            # pixels (metric_map==1) this Gaussian contributed to above alpha=1/255.
            # Only populated by diff_surfel_3D_sh_res when metric_map was provided.
            'metric_counts': metric_counts,
            # `--l2` (mixed_3d): second image output (clone of `render`, separate
            # autograd node). Wire L2(image_untex, gt) here so the rasterizer's
            # backward receives two upstream image gradients and routes them
            # per-Gauss inside CUDA (textured Gauss → L1+SSIM grad, untextured
            # → L2 grad). None when --l2 is off or when not using mixed_3d.
            'render_untex': rendered_image_untex,
    })
    
    # Add diffuse_ngp mode separate RGB outputs
    if gaussian_rgb_diffuse_ngp is not None:
        rets['gaussian_rgb'] = gaussian_rgb_diffuse_ngp
    if ngp_rgb_diffuse_ngp is not None:
        rets['ngp_rgb'] = ngp_rgb_diffuse_ngp
    
    # Add scout loss data for diffuse_offset xyz mode
    if scout_loss_data is not None:
        rets['scout_loss_data'] = scout_loss_data

    # Add skybox FG/BG outputs for visualization
    if rendered_image_fg is not None:
        rets['render_fg'] = rendered_image_fg
    if skybox_rgb is not None:
        rets['render_bg'] = skybox_rgb

    # Add raw features (before MLP decode) if requested
    if raw_features_saved is not None:
        rets['raw_features'] = raw_features_saved

    return rets
