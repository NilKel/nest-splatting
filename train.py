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

import os
import json
import math
import sys

# === Unbiased Depth: pre-import sys.modules swap ===
# When --unbiased is set on the command line, transparently substitute the
# diff_surfel_3D_sh_res_unbiased rasterizer for diff_surfel_3D_sh_res so every
# downstream `import diff_surfel_3D_sh_res` (gaussian_renderer's top-level
# import + many inline imports inside render()) resolves to the unbiased fork.
# Done BEFORE importing gaussian_renderer so the binding takes effect.
if "--unbiased" in sys.argv:
    try:
        import diff_surfel_3D_sh_res_unbiased as _unb_rast
        sys.modules['diff_surfel_3D_sh_res'] = _unb_rast
        sys.modules['diff_surfel_3D_sh_res._C'] = _unb_rast._C
        print("[UNBIASED] Substituted diff_surfel_3D_sh_res_unbiased for diff_surfel_3D_sh_res")
    except ImportError as _e:
        raise ImportError(
            "--unbiased requested but diff_surfel_3D_sh_res_unbiased not installed. "
            "Build it via: cd submodules/diff_surfel_3D_sh_res_unbiased && "
            "python -m pip install -e . --no-build-isolation"
        ) from _e

import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, ssim_map

# --lpips_w: lazily-built frozen LPIPS backbone for the perceptual training loss.
_LPIPS_NET = None
from optimizing_spa import OptimizingSpa
from gaussian_renderer import render, network_gui
import traceback
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_scaling_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def write_gpu_failure(model_path, error_msg):
    """Write GPU failure to marker file for SLURM retry.

    This file is checked by the SLURM worker script after training completes.
    If it contains content, the worker will submit a retry job on a different node.
    """
    failure_file = os.path.join(model_path, ".gpu_failure")
    try:
        with open(failure_file, "w") as f:
            f.write(f"Time: {datetime.datetime.now()}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        print(f"\n[GPU FAILURE] Written to {failure_file}")
    except Exception as e:
        print(f"\n[GPU FAILURE] Could not write failure file: {e}")
        print(f"Original error: {error_msg}")


def is_gpu_error(error):
    """Check if an exception is a GPU-related error that warrants retry."""
    error_str = str(error).lower()
    gpu_error_patterns = [
        "cuda",
        "out of memory",
        "illegal memory access",
        "device-side assert",
        "nccl",
        "cublas",
        "cudnn",
        "gpu",
        "tinycudann",
        "compute capability",
    ]
    return any(pattern in error_str for pattern in gpu_error_patterns)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from hash_encoder.modules import INGP
from hash_encoder.config import Config
from scene.background import LearnableSkybox, SphereHashGridBackground
from utils.render_utils import save_img_u8, convert_gray_to_cmap, create_intersection_heatmap, create_intersection_histogram, create_flex_beta_heatmap
from utils.mesh_reinit import tsdf_mesh_reinit
from utils.point_utils import cam2rays
from utils.render_utils import gsnum_trans_color
import open3d as o3d
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt


def _colorize_max_contrib_idx(max_idx_map):
    """Turn a per-pixel max-contributor Gaussian id map into an [H, W, 3] uint8-ready
    float array in [0, 1] via a hash-based cyclic colormap. Pixels with no
    contributor (id < 0) render as black.

    Args:
        max_idx_map: torch tensor of shape [1, H, W] or [H, W], integer dtype.
    Returns:
        numpy array [H, W, 3] in [0, 1] suitable for save_img_u8.
    """
    arr = max_idx_map.squeeze().detach().cpu().numpy().astype(np.int64)  # [H, W]
    H, W = arr.shape
    invalid = arr < 0
    # Hash each id to a stable color via three multiplicatively-mixed primes.
    # Clamp the negative/invalid slots before the modulo hash to keep the integer
    # math defined; we'll black them out afterwards.
    hashed = np.clip(arr, 0, None)
    r = ((hashed * 2654435761) & 0xFFFFFF) / 0xFFFFFF
    g = ((hashed * 40503 + 31) & 0xFFFFFF) / 0xFFFFFF
    b = ((hashed * 1442695040888963407 + 11) & 0xFFFFFF) / 0xFFFFFF
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    if invalid.any():
        rgb[invalid] = 0.0
    return rgb


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):

    training_start_time = time.time()

    # Auto-enable `pipe.skip_aux_normal_dist` when no consumer (normal-
    # consistency or depth-distortion regulariser) is active for the run.
    # The renderer then short-circuits the per-call `depth_to_normal` +
    # render_normal view→world rotate + the loss-side normal_error build.
    # At 4K image resolution this avoids ~600 MB of intermediate tensors
    # per iter, on top of the ~1.7 GB saved by the SLIM out_others kernel.
    if getattr(pipe, 'keep_aux_normal_dist', False):
        # User opt-out: force aux (normals/depth-distortion) rendering ON, overriding
        # both any --skip_aux_normal_dist and the auto-enable below.
        pipe.skip_aux_normal_dist = False
        print("[PIPE] `keep_aux_normal_dist` set — aux normals/depth rendering forced ON.")
    elif not getattr(pipe, 'skip_aux_normal_dist', False):
        _no_normal_dist = (
            getattr(opt, 'lambda_normal', 0.0) == 0.0
            and getattr(opt, 'lambda_dist', 0.0) == 0.0
            and getattr(args, 'w_normal', 0.0) == 0.0
        )
        if _no_normal_dist:
            pipe.skip_aux_normal_dist = True
            print("[PIPE] Auto-enabled `skip_aux_normal_dist` (no normal/dist "
                  "regs active for this run — saves ~600 MB per render at 4K). "
                  "Pass --keep_aux_normal_dist to keep them ON.")

    # 3D_SH_filmres: the 3D_SH_res device-global setters (set_residual_mode,
    # set_activation_bias, set_lru_slope, ...) are module-local, so for filmres they
    # must target the filmres fork. `_SHRES_SETTER_MOD.set_X(...)` calls below resolve
    # here once (method is fixed for the run); plain 3D_SH_res-family uses the base.
    import diff_surfel_3D_sh_res as _SHRES_BASE_MOD
    try:
        import diff_surfel_3D_sh_filmres as _SHRES_FILM_MOD
    except ImportError:
        _SHRES_FILM_MOD = None
    # GEStex explore+harden (0-20k) renders through an ISOLATED clone of
    # diff_surfel_3D_sh_res — diff_surfel_3D_sh_res_harden — that carries the
    # first-intersection sort (see docs/GESTEX_PIPELINE.md "CUDA isolation rule").
    # Its device-globals (MLP weights, residual mode, biases, LRU, tile-depth sort)
    # are module-local, so every _SHRES_SETTER_MOD.set_X call must target it — this
    # ALSO makes the 20k joint-transition MLP-weight/grad handoff read from the
    # module that actually trained the MLP during harden.
    try:
        import diff_surfel_3D_sh_res_harden as _SHRES_GESTEX_HARDEN_MOD
    except ImportError:
        _SHRES_GESTEX_HARDEN_MOD = None
    # `--densfix` (--method 3D_SH_res): isolated clone that excludes the hash-query-
    # point term from the densification proxy. Module-local device-globals, so every
    # _SHRES_SETTER_MOD.set_X (incl. set_mlp_weights / get_mlp_grads) must target it.
    try:
        import diff_surfel_3D_sh_res_densfix as _SHRES_DENSFIX_MOD
    except ImportError:
        _SHRES_DENSFIX_MOD = None
    # `--trunc` (--method 3D_SH_res): isolated clone with the settable POST-blend
    # truncation exit threshold (set_exit_T). Module-local device-globals, so every
    # _SHRES_SETTER_MOD.set_X (incl. set_mlp_weights / get_mlp_grads) must target it.
    try:
        import diff_surfel_3D_sh_res_trunc as _SHRES_TRUNC_MOD
    except ImportError:
        _SHRES_TRUNC_MOD = None
    # `--method proberes`: isolated clone whose residual is a probe-mapped bilinear
    # fetch from a shared texture image (no in-kernel MLP). Module-local device-
    # globals (set_residual_mode, set_activation_bias, thresholds, ...) must target it.
    try:
        import diff_surfel_3D_sh_res_probe as _SHRES_PROBE_MOD
    except ImportError:
        _SHRES_PROBE_MOD = None
    # `--wsr` (proberes only): the WSR sort-free clone. Must win over the plain
    # probe module so every module-local setter targets the module that renders.
    try:
        import diff_surfel_3D_sh_res_probe_wsr as _SHRES_PROBE_WSR_MOD
    except ImportError:
        _SHRES_PROBE_WSR_MOD = None
    if getattr(args, 'is_gestex', False) and _SHRES_GESTEX_HARDEN_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_GESTEX_HARDEN_MOD
    elif getattr(args, 'method', None) == "3D_SH_filmres" and _SHRES_FILM_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_FILM_MOD
    elif getattr(args, 'densfix', False) and getattr(args, 'method', None) == "3D_SH_res" \
            and _SHRES_DENSFIX_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_DENSFIX_MOD
    elif (getattr(args, 'trunc', False) or getattr(args, 'gap_noise', False)) \
            and getattr(args, 'method', None) == "3D_SH_res" \
            and _SHRES_TRUNC_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_TRUNC_MOD
    elif getattr(args, 'method', None) == "proberes" \
            and (getattr(args, 'wsr', False) or getattr(args, 'wsr_composite', False)) \
            and _SHRES_PROBE_WSR_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_PROBE_WSR_MOD
    elif getattr(args, 'method', None) == "proberes" and _SHRES_PROBE_MOD is not None:
        _SHRES_SETTER_MOD = _SHRES_PROBE_MOD
    else:
        _SHRES_SETTER_MOD = _SHRES_BASE_MOD

    # 3D_SH_add: same architecture as 3D_SH_res, only the outer activation differs.
    # Set the residual-mode flag now so `--decomp` validation accepts it. The
    # actual `args.method = "3D_SH_res"` alias happens AFTER
    # prepare_output_and_logger so the run lands in its own `3D_SH_add/` folder.
    # 0 = 3D_SH_res (outer per-Gauss ReLU on ReLU(SV)+residual),
    # 1 = 3D_SH_add (separate ReLUs: ReLU(SV) + ReLU(residual)),
    # 2 = mixed_*_sep (signed per-Gauss residual, deferred per-pixel ReLU in Python).
    # `--method mixed[_3d]` now uses mode 0 (per-Gauss outer ReLU) so the textured
    # half is byte-equivalent to 3D_SH_res; the untextured half adds ReLU(SV) into
    # the same blend (no per-pixel deferred clamp). Pass `--method mixed_3d_sep`
    # (added separately) to opt into the mode-2 deferred-clamp variant.
    if args.method == "3D_SH_add":
        args._residual_mode = 1
    elif args.method in ("mixed_sep", "mixed_3d_sep", "3D_SH_res_sep", "clip_relight"):
        args._residual_mode = 2
    else:
        # 3D_SH_res, mixed, mixed_3d, res_switch (starts mode 0, flips to 2 at
        # --res_switch_iter), baseline, others → mode 0.
        args._residual_mode = 0

    # `--method GEStex`: GES-style sort-free bi-scale. Phases 0-20k are behaviorally
    # `res_switch` (mode 0->2 flip + --lru), so we record a separate `args.is_gestex`
    # flag now and ALIAS `args.method = "res_switch"` after prepare_output_and_logger
    # (mirrors the 3D_SH_add alias). This inherits every 3D_SH_res-family gate + the
    # res_switch flip machinery for free; GEStex-specific events (opacity ramp,
    # occlusion cull, bake, spawn, joint dispatch) are added as separate blocks gated
    # on `args.is_gestex`. The joint stage (>= --ges_joint_iter) flips
    # `ingp.is_gestex_joint = True` and the renderer intercepts BEFORE the res_switch
    # path. See docs/GESTEX_MODE.md.
    args.is_gestex = (args.method == "GEStex")
    if args.is_gestex:
        args._residual_mode = 0
        # LOCAL (per-Gauss, mode 0) vs GLOBAL (post-blend, mode 2) LRU: the mode 0->2
        # flip is the res_switch flip. Keeping LOCAL LRU through hardening clamps
        # LRU(SH+residual) PER SURFEL before blending, so the residual can't go deeply
        # negative to "subtract-and-hide" bloated geometry (which global/mode-2 allows).
        # Default the flip to 5k — BEFORE the harden phase. Local per-Gauss LRU (mode 0)
        # clamps LRU(SV+residual) per surfel, which CHOKES a residual that needs to subtract;
        # flipping to global (mode 2, signed residual + post-blend LRU) early lets the texture
        # express before the surfels harden opaque. Override with --ges_global_lru_iter.
        _glru = int(getattr(args, 'ges_global_lru_iter', -1) or -1)
        if _glru <= 0:
            _glru = 5000
        args.res_switch_iter = _glru
        # --ges_local_lru: NEVER flip to global/mode-2. Stays pure 3D_SH_res semantics
        # (mode 0: per-surfel outer ReLU/LRU clamps ReLU(SV+0.5)+residual BEFORE the
        # blend) through explore+harden. A/B for the "signed textures pass through
        # opaque surfels → top layer never forced to self-correct" hypothesis: in
        # mode 2 a hardened frontmost surfel can carry a wrong signed color that the
        # post-blend clamp hides; mode 0 forces each surfel's own color to be valid.
        if getattr(args, 'ges_local_lru', False):
            args.res_switch_iter = -1          # flip block requires > 0 → never fires
            print(f"[GEStex] --ges_local_lru: mode 0->2 flip DISABLED; per-surfel outer "
                  f"ReLU{'' if float(getattr(args, 'lru', 0.0)) == 0.0 else f' (LRU α={args.lru})'} "
                  f"for the whole run (3D_SH_res semantics).")
        # Post-blend LRU (three-site leaky-ReLU in the joint stage) needs a nonzero slope.
        elif float(getattr(args, 'lru', 0.0)) == 0.0:
            args.lru = 0.01
            print(f"[GEStex] --lru auto-defaulted to 0.01 (post-blend + composite LRU).")
        # Kernel: GES uses a plain Gaussian falloff min(1, w*exp(-r^2/2)) (infinite tails).
        # We also allow beta-family kernels: beta_scaled gives HARD compact support at 3σ
        # (rho3d>=9 culled), so the per-surfel footprint can't expand as opacity ramps —
        # a direct lever against the opacity-tail bloat during hardening. Any other kernel
        # falls back to gaussian.
        if getattr(args, 'kernel', 'gaussian') not in ('gaussian', 'beta', 'beta_scaled'):
            print(f"[GEStex] overriding --kernel {args.kernel} -> gaussian (GES falloff).")
            args.kernel = 'gaussian'
        if args.kernel != 'gaussian':
            # Compact-support textured surfels; keep the (post-20k) untextured 3D Gaussians
            # on a Gaussian EWA falloff — canonical `--kernel beta_scaled --kernel2 gaussian`.
            args.kernel2 = 'gaussian'
            print(f"[GEStex] --kernel {args.kernel} (compact-support surfels; footprint capped at "
                  f"3σ regardless of opacity). Untextured Gaussians use Gaussian EWA (--kernel2 gaussian).")
        elif getattr(args, 'kernel2', None) is not None:
            args.kernel2 = None
        _fis = int(getattr(args, 'ges_first_int_iter', -1))
        _fmi = int(getattr(args, 'ges_frontmost_iter', -1))
        print(f"[GEStex] schedule: harden@{args.ges_phase1_iter} (freeze w, ramp opac), "
              f"LOCAL->GLOBAL LRU (mode 0->2) @"
              f"{'OFF (--ges_local_lru)' if args.res_switch_iter <= 0 else args.res_switch_iter}, "
              f"tile-depth-sort@{_fis if _fis >= 0 else 'off'}, "
              f"frontmost-first@{_fmi if _fmi >= 0 else 'off'}, "
              f"occlusion-cull@{args.ges_occlusion_iter} (n_thr={args.ges_occlusion_thresh}), "
              f"bake+splat@{args.ges_joint_iter}.")

    # `--method res_switch`: two-phase curriculum (3D_SH_res → 3D_SH_res_sep at
    # --res_switch_iter). Default `--lru` to 0.01 so both phases share leaky-
    # gradient at the outer clamp and the switch is a smooth knee instead of a
    # discontinuous jump. User can still pass an explicit `--lru` to override.
    if args.method == "res_switch" and float(getattr(args, 'lru', 0.0)) == 0.0:
        args.lru = 0.01
        print(f"[RES_SWITCH] --lru auto-defaulted to 0.01 for smooth mode-0 → mode-2 transition "
              f"at iter {getattr(args, 'res_switch_iter', 10000)}. Override with --lru <α>.")

    # `--method res_3d`: two-phase split curriculum. Pre-`--res_3d_iter` behaves
    # like 3D_SH_res + --lru (single-cascade, per-Gauss outer LRU). At the split
    # iteration each Gauss duplicates → (2D residual-carrier + 3D EWA SV-carrier);
    # rendering switches to a two-pass scheme where each pass masks one half's
    # opacity so the two T cascades are independent. Default `--lru` to 0.01 so
    # the pre-split phase has the LRU gradient that maps cleanly onto the post-
    # split per-pixel LRU on C_tex.
    if args.method in ("res_3d", "res_3d_paired", "res_3d_double") and float(getattr(args, 'lru', 0.0)) == 0.0:
        args.lru = 0.01
        print(f"[{args.method.upper()}] --lru auto-defaulted to 0.01 for the post-blend LRU "
              f"(split fires at iter {getattr(args, 'res_3d_iter', 10000)}). Override with --lru <α>.")

    # `--method mixed`: install setter mirrors so all existing
    # `from diff_surfel_3D_sh_res import set_X` call sites also update the
    # `diff_surfel_mixed` device globals. This avoids touching every call site.
    # Lazy `from X import Y` re-reads the module attribute each call, so the
    # monkey-patch propagates as long as we patch BEFORE the first import.
    if args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
        try:
            import diff_surfel_3D_sh_res as _ds_orig
            if args.method in ("mixed_3d", "mixed_3d_sep"):
                import diff_surfel_mixed_3d as _ds_mirror
            else:
                import diff_surfel_mixed as _ds_mirror
            _MIRRORED_SETTERS = (
                'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                'set_overdraw_lambda', 'set_weight_reg_lambda',
                'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                'set_depth_sort', 'set_ste_relu', 'set_lru_slope',
            )
            for _name in _MIRRORED_SETTERS:
                if not hasattr(_ds_orig, _name) or not hasattr(_ds_mirror, _name):
                    continue
                _of = getattr(_ds_orig, _name)
                _mf = getattr(_ds_mirror, _name)
                def _make_mirror(of, mf):
                    def _wrapped(*a, **k):
                        of(*a, **k); mf(*a, **k)
                    return _wrapped
                setattr(_ds_orig, _name, _make_mirror(_of, _mf))
            print("[MIXED] mirrored diff_surfel_3D_sh_res setters → diff_surfel_mixed")
        except ImportError as _e:
            print(f"[MIXED] WARNING: could not install setter mirror: {_e}")

    # `--method GEStex`: same mirror trick for the explore+harden clone. GEStex 0-20k
    # renders through diff_surfel_3D_sh_res_harden (isolated first-intersection-sort
    # clone), whose device globals are module-local. _SHRES_SETTER_MOD already targets
    # the clone, but several call sites do a direct
    # `from diff_surfel_3D_sh_res import set_X` (set_opacity_thresh, set_dropout,
    # set_activation_bias re-fires, ...) — mirror them so those writes reach the clone
    # too. NOTE getters (get_mlp_grads) can NOT be mirrored — they're routed explicitly
    # at their call sites.
    if getattr(args, 'is_gestex', False):
        try:
            import diff_surfel_3D_sh_res as _ds_orig
            import diff_surfel_3D_sh_res_harden as _ds_mirror
            _MIRRORED_SETTERS = (
                'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                'set_overdraw_lambda', 'set_weight_reg_lambda',
                'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                'set_depth_sort', 'set_ste_relu', 'set_lru_slope',
                'set_opacity_thresh', 'set_dropout', 'set_detach_res_shape_grad',
                'set_converge_threshold',
            )
            for _name in _MIRRORED_SETTERS:
                if not hasattr(_ds_orig, _name) or not hasattr(_ds_mirror, _name):
                    continue
                _of = getattr(_ds_orig, _name)
                _mf = getattr(_ds_mirror, _name)
                def _make_mirror_ges(of, mf):
                    def _wrapped(*a, **k):
                        of(*a, **k); mf(*a, **k)
                    return _wrapped
                setattr(_ds_orig, _name, _make_mirror_ges(_of, _mf))
            print("[GEStex] mirrored diff_surfel_3D_sh_res setters → diff_surfel_3D_sh_res_harden")
        except ImportError as _e:
            print(f"[GEStex] WARNING: could not install harden setter mirror: {_e} — "
                  f"falling back to the shared rasterizer would silently freeze the MLP; "
                  f"build submodules/diff_surfel_3D_sh_res_harden.")

    # --decomp: only the diff_surfel_3D_sh_res rasterizer exposes the sh_only /
    # tex_only decompose_mode paths needed to split the supervision.
    if getattr(args, 'decomp', False) and args.method not in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "3D_SH_add", "proberes"):
        raise RuntimeError(
            f"--decomp requires --method 3D_SH_res or 3D_SH_add; got --method {args.method}. "
            f"Other rasterizers don't expose the sh_only/tex_only decompose path."
        )
    # --blurprog: uses the same gt_low cache as --decomp but only needs main-loop
    # render — no decompose_mode dependency. Gated to 3D_SH_res here only because
    # the cache plumbing (Scene.__init__) is tied to the same code path. Could be
    # relaxed if needed.
    if getattr(args, 'blurprog', False) and args.method not in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "3D_SH_add"):
        raise RuntimeError(
            f"--blurprog requires --method 3D_SH_res or 3D_SH_add; got --method {args.method}."
        )

    # Pass mini flag to OptimizationParams so training_setup can pick SparseGaussianAdam
    opt.mini = getattr(args, 'mini', False)
    # `--film_latent_lr`: bridge onto opt so training_setup sees it (opt = op.extract(args)
    # only carries OptimizationParams fields). -1 = fall back to feature_lr.
    opt.film_latent_lr = getattr(args, 'film_latent_lr', -1.0)

    # Final-iter eval is handled by render_final_images at the end of training,
    # so don't ALSO include it here (would double-eval).
    # First-iter debug eval removed.
    # Periodic eval every 5k from 25k onwards so you can pick a good stop point.
    testing_iterations += list(range(25_000, opt.iterations, 5_000))
    saving_iterations += [opt.iterations]

    # --patience: dense periodic test eval for convergence diagnostics.
    # Inject extra evals every `patience_eval_interval` iters from
    # `patience_start_iter` onwards so the existing training_report path
    # picks them up (writes one row to test_metrics.txt per eval).
    if args.patience > 0:
        _patience_iters = list(range(args.patience_start_iter,
                                     opt.iterations + 1,
                                     args.patience_eval_interval))
        testing_iterations = sorted(set(testing_iterations + _patience_iters))
        print(f"[PATIENCE] Convergence study: test eval every "
              f"{args.patience_eval_interval} iters from {args.patience_start_iter} "
              f"to {opt.iterations}. Early-stop after {args.patience} consecutive "
              f"non-improvements (Δ < {args.patience_min_delta} dB). "
              f"Total dense evals: {len(_patience_iters)}.")

    # Patience tracking state (only consumed when args.patience > 0).
    _patience_best_psnr = -float('inf')
    _patience_best_iter = 0
    _patience_no_improve = 0
    _patience_evals_seen = 0

    test_psnr = []
    train_psnr = []
    iter_list = []
    optimizing_spa = None

    scene_name = args.scene_name
    tb_writer = prepare_output_and_logger(dataset, scene_name, args.yaml, args)
    args.model_path = dataset.model_path

    # === --finetune_from: resolve the three artifacts of a finished run ===
    # Runs here (before BOTH --init_ply sites, cold and normal) so the PLY simply
    # rides the existing --init_ply path. The hash/MLP and ISP are stashed on args
    # and applied later, at the points where those objects exist.
    args._finetune_ngp = None
    args._finetune_ppisp = None
    if getattr(args, "finetune_from", None):
        _ft_dir = args.finetune_from
        _pc_root = os.path.join(_ft_dir, "point_cloud")
        if not os.path.isdir(_pc_root):
            raise RuntimeError(f"--finetune_from: no point_cloud/ under {_ft_dir}")
        if args.finetune_iter > 0:
            _ft_it = args.finetune_iter
        else:
            _its = [int(d.split("_")[1]) for d in os.listdir(_pc_root)
                    if d.startswith("iteration_") and d.split("_")[1].isdigit()]
            if not _its:
                raise RuntimeError(f"--finetune_from: no iteration_* under {_pc_root}")
            _ft_it = max(_its)
        _ply = os.path.join(_pc_root, f"iteration_{_ft_it}", "point_cloud.ply")
        if not os.path.isfile(_ply):
            raise RuntimeError(f"--finetune_from: missing {_ply}")
        args.init_ply = _ply
        print("\n" + "=" * 70)
        print(f"  FINETUNE FROM {_ft_dir} @ iteration {_ft_it}")
        print("=" * 70)
        print(f"  surfels+SV+SH : {_ply}")
        _ngp = os.path.join(_ft_dir, f"ngp_{_ft_it}.pth")
        if os.path.isfile(_ngp):
            args._finetune_ngp = (_ft_dir, _ft_it)
            print(f"  hash+MLP      : {_ngp}")
        else:
            # Loud, not fatal: without this the residual restarts from random and
            # the reloaded SH base is left explaining detail it was never fit for.
            print(f"  hash+MLP      : *** MISSING {_ngp} — residual restarts from scratch ***")
        _pp = os.path.join(_pc_root, f"iteration_{_ft_it}", "ppisp.pt")
        if os.path.isfile(_pp):
            args._finetune_ppisp = _pp
            print(f"  ISP           : {_pp}"
                  + ("" if getattr(args, "ppisp", False) else "  (found, but --ppisp is OFF — will be IGNORED)"))
        print(f"  optimizer state: not restored (fresh Adam + fresh LR schedule)")
        print("=" * 70 + "\n")
        if getattr(args, "ppisp", False) and args._finetune_ppisp is None:
            print("[FINETUNE] WARNING: --ppisp is on but the source run saved no ppisp.pt. "
                  "The ISP will start at identity while the Gaussians are already canonical "
                  "— they will re-absorb exposure/vignetting. Verify the source run used --ppisp.")

    # 3D_SH_add → alias to 3D_SH_res for the rest of training. We had to wait
    # until after prepare_output_and_logger so the run gets its own
    # `outputs/.../3D_SH_add/<run>` folder; downstream code only knows about
    # 3D_SH_res. The activation switch is plumbed via set_residual_mode(1)
    # further below (gated on args._residual_mode == 1).
    if args.method == "3D_SH_add":
        args.method = "3D_SH_res"
        print("[3D_SH_add] activation = ReLU(SH+sh_bias) + ReLU(residual+res_bias) (separate ReLUs)")

    # `--method GEStex` → alias to res_switch for phases 0-20k. The run folder was
    # already named by prepare_output_and_logger (user -m path). Downstream 3D_SH_res-
    # family gates + the res_switch flip now all apply. `args.is_gestex` (set earlier)
    # drives the GEStex-specific schedule + joint dispatch. We stash the joint-stage
    # config on args before the alias so it survives.
    if getattr(args, 'is_gestex', False):
        args.method = "res_switch"
        print("[GEStex] aliased method -> res_switch for phases 0-20k "
              "(GEStex-specific schedule/joint dispatch gated on args.is_gestex).")

    # Pass method, hybrid_levels, and decompose_mode to dataset for use in Scene/GaussianModel
    dataset.method = args.method
    dataset.is_gestex = getattr(args, 'is_gestex', False)
    dataset.hybrid_levels = args.hybrid_levels if hasattr(args, 'hybrid_levels') else 3
    dataset.decompose_mode = args.decompose_mode if hasattr(args, 'decompose_mode') else None

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)

    # --wsr_composite implies --wsr (same machinery, operator mode 2).
    if getattr(args, 'wsr_composite', False):
        args.wsr = True

    # --wsr validation + occ LR handoff (read by training_setup's wsr_occ group).
    if getattr(args, 'wsr', False):
        assert args.method == "proberes", "--wsr currently requires --method proberes"
        assert int(getattr(opt, 'densify_until_iter', 0)) == 0, \
            "--wsr requires --densify_until_iter 0 (densification would desync _wsr_occ)"
        gaussians.wsr_occ_lr = float(args.wsr_occ_lr)

    # `--start_resolution N` progressive curriculum: load Scene at start_resolution
    # + start_data_device now; remember the FINAL `-r` / `--data_device` for the
    # mid-train reload triggered at `--freeze_hash_iter`.
    _progressive_res_active = (getattr(dataset, 'start_resolution', 0) > 0
                                and dataset.start_resolution != dataset.resolution
                                and args.freeze_hash_iter > 0)
    if _progressive_res_active:
        _final_resolution = int(dataset.resolution)
        _final_data_device = str(dataset.data_device)
        dataset.resolution = int(dataset.start_resolution)
        dataset.data_device = str(dataset.start_data_device)
        # `args` is the same Namespace object as `dataset` for these fields
        # in most cases — re-sync just to be safe.
        if hasattr(args, 'resolution'):
            args.resolution = dataset.resolution
        if hasattr(args, 'data_device'):
            args.data_device = dataset.data_device
        print(f"[PROGRESSIVE-RES] iter 0 → {dataset.start_resolution}x downsample, "
              f"GT on {dataset.start_data_device}. "
              f"At iter {args.freeze_hash_iter} will reload at {_final_resolution}x + "
              f"{_final_data_device}.")
    else:
        _final_resolution = None
        _final_data_device = None

    # Set kernel type for beta kernel support
    gaussians.kernel_type = args.kernel
    # `--method mixed_3d`: optional separate kernel for the untextured EWA half.
    # None → untextured uses gaussians.kernel_type (unchanged behavior).
    gaussians.kernel_type2 = getattr(args, 'kernel2', None)
    # Set densification gradient mode (vanilla = signed, abs = AbsGS)
    gaussians.use_absgs = (args.grads == "abs")

    # `--method GEStex`: tag the model + init the GES surfel-opacity multiplier.
    # `surfel_opac` (global float) is applied to surfel opacity in the renderer during
    # phases 10k-20k and in the joint-stage surfel pass; ramps 1 -> 30 -> 60 -> 90 -> 255.
    gaussians.is_gestex = getattr(args, 'is_gestex', False)
    gaussians.surfel_opac = 1.0
    gaussians.ges_s_weight = float(getattr(args, 'ges_s_weight', 1.0))
    gaussians.ges_atlas_res = int(getattr(args, 'ges_atlas_res', 8))

    # Check for warmup checkpoint in data directory
    # If --warmup tag is specified, use warmup_checkpoint_{tag}.pth
    if args.warmup:
        warmup_checkpoint_path = os.path.join(dataset.source_path, f"warmup_checkpoint_{args.warmup}.pth")
    else:
        warmup_checkpoint_path = os.path.join(dataset.source_path, "warmup_checkpoint.pth")
    loaded_from_warmup = False

    # Shared-resume checkpoint path (only used when --share_ckpt_iter > 0).
    # Captures Gaussians + their optimizer + densif state + INGP model + INGP
    # optimizer + gs_alpha_masks at iteration N. Sweeps that vary downstream-only
    # knobs (e.g., --lambda_converge after iter N) load from this snapshot
    # instead of re-running iters 0 → N.
    share_ckpt_path = None
    if args.share_ckpt_iter > 0:
        if args.share_ckpt_tag:
            _share_tag = args.share_ckpt_tag
        else:
            _share_tag = f"{args.method}_{args.kernel}_h{args.hybrid_levels}_iter{args.share_ckpt_iter}"
        share_ckpt_path = os.path.join(dataset.source_path, f"shared_ckpt_{_share_tag}.pth")
    shared_ckpt_data = None  # populated on load; INGP state applied later, post-INGP-creation
    loaded_from_shared = False
    
    # Shared-resume checkpoint takes precedence over --cold IF the file already
    # exists — explicit opt-in via --share_ckpt_iter signals "I want sweep-mode
    # resumption". First run with --cold + --share_ckpt_iter still trains from
    # scratch (file doesn't exist yet) and saves at iter N; subsequent runs hit
    # the elif below and resume.
    _shared_ckpt_overrides_cold = (
        args.cold and share_ckpt_path is not None
        and os.path.exists(share_ckpt_path) and not args.scratch
    )
    if _shared_ckpt_overrides_cold:
        print("\n" + "="*70)
        print("  --cold OVERRIDDEN by existing shared-resume checkpoint")
        print("="*70)
        print(f"  Shared ckpt found at: {share_ckpt_path}")
        print(f"  Falling through to share-ckpt load (cold start skipped).")
        print("="*70 + "\n")

    # Cold start mode: skip all checkpoint loading
    if args.cold and not _shared_ckpt_overrides_cold:
        print("\n" + "="*70)
        print("  COLD START MODE")
        print("="*70)
        print("  Skipping 2DGS warmup phase and all checkpoint loading")
        print("  Training Nest representation from scratch with hash_in_CUDA=True")
        print("="*70 + "\n")
        mini_res_scales = [1.0, 0.5] if (args.mini and args.mini_warmup) else [1.0]
        scene = Scene(dataset, gaussians, resolution_scales=mini_res_scales,
                      mcmc_fps=(args.mcmc_fps and not args.init_ply),
                      cap_max=args.cap_max, full_args=args)

        # Override initialization with external PLY (same logic as else branch)
        if args.init_ply:
            print(f"\n[INIT_PLY] Loading Gaussians from: {args.init_ply}")
            gaussians.load_ply(args.init_ply, args=args)
            n_gs = len(gaussians.get_xyz)
            print(f"[INIT_PLY] Loaded {n_gs} Gaussians, SH degree={gaussians.active_sh_degree}")
            if args.mcmc or args.mcmc_fps or args.mcmc_deficit:
                print(f"[INIT_PLY] cap_max: {args.cap_max} → {max(args.cap_max, n_gs)}")
                args.cap_max = max(args.cap_max, n_gs)
            gaussians._appearance_level = nn.Parameter(
                torch.ones(n_gs, 1, device="cuda") * 24, requires_grad=False)
            if hasattr(args, 'method') and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
                gaussians._gaussian_feat_dim = 0
                gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            gaussians.max_radii2D = torch.zeros(n_gs, device="cuda")
            print(f"[INIT_PLY] Using loaded geometry + SH as initialization\n")

        # Initialize flex kernel per-Gaussian beta parameter (if using flex kernel)
        if args.kernel == "flex":
            n_gaussians = len(gaussians.get_xyz)
            flex_beta_init_val = 5.0  # softplus(5) ≈ 5.007, starts sharp/hard
            flex_beta_init = torch.full((n_gaussians, 1), flex_beta_init_val, device="cuda").float()
            gaussians._flex_beta = nn.Parameter(flex_beta_init.requires_grad_(True))
            init_beta_val = torch.nn.functional.softplus(torch.tensor(flex_beta_init_val)).item()
            print(f"[FLEX KERNEL] Initialized {n_gaussians} Gaussians with per-Gaussian beta")
            print(f"[FLEX KERNEL] Initial beta value: {init_beta_val:.4f} (0=standard Gaussian, higher=sharper)")
        elif args.kernel == "general":
            n_gaussians = len(gaussians.get_xyz)
            shape_init_val = -10.0  # sigmoid(-10) ≈ 0 -> beta = 0*6+2 = 2.0 (standard Gaussian)
            shape_init = torch.full((n_gaussians, 1), shape_init_val, device="cuda").float()
            gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
            init_shape_val = (torch.sigmoid(torch.tensor(shape_init_val)) * 6.0 + 2.0).item()
            print(f"[GENERAL KERNEL] Initialized {n_gaussians} Gaussians with shape parameter")
            print(f"[GENERAL KERNEL] Initial beta value: {init_shape_val:.3f} (2=Gaussian, 8=super-Gaussian/box)")

        gaussians.training_setup(opt)
    elif checkpoint:
        # User-specified checkpoint takes priority
        scene = Scene(dataset, gaussians, mcmc_fps=args.mcmc_fps, cap_max=args.cap_max, full_args=args)

        # Initialize flex kernel per-Gaussian beta parameter (if using flex kernel)
        if args.kernel == "flex":
            n_gaussians = len(gaussians.get_xyz)
            flex_beta_init_val = 5.0  # softplus(5) ≈ 5.007, starts sharp/hard
            flex_beta_init = torch.full((n_gaussians, 1), flex_beta_init_val, device="cuda").float()
            gaussians._flex_beta = nn.Parameter(flex_beta_init.requires_grad_(True))
            init_beta_val = torch.nn.functional.softplus(torch.tensor(flex_beta_init_val)).item()
            print(f"[FLEX KERNEL] Initialized {n_gaussians} Gaussians with per-Gaussian beta")
            print(f"[FLEX KERNEL] Initial beta value: {init_beta_val:.4f} (0=standard Gaussian, higher=sharper)")
        elif args.kernel == "general":
            n_gaussians = len(gaussians.get_xyz)
            shape_init_val = -10.0  # sigmoid(-10) ≈ 0 -> beta = 0*6+2 = 2.0 (standard Gaussian)
            shape_init = torch.full((n_gaussians, 1), shape_init_val, device="cuda").float()
            gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
            init_shape_val = (torch.sigmoid(torch.tensor(shape_init_val)) * 6.0 + 2.0).item()
            print(f"[GENERAL KERNEL] Initialized {n_gaussians} Gaussians with shape parameter")
            print(f"[GENERAL KERNEL] Initial beta value: {init_shape_val:.3f} (2=Gaussian, 8=super-Gaussian/box)")

        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    elif share_ckpt_path is not None and os.path.exists(share_ckpt_path) and not args.scratch:
        # === LOAD SHARED-RESUME CHECKPOINT (takes precedence over warmup ckpt) ===
        # Restores everything: Gaussian params + their optimizer + densif accumulators
        # + gs_alpha_masks + (later, after INGP creation) INGP weights + INGP optimizer.
        print("\n" + "="*70)
        print("  LOADING SHARED-RESUME CHECKPOINT")
        print("="*70)
        print(f"  Path: {share_ckpt_path}")

        shared_ckpt_data = torch.load(share_ckpt_path, map_location='cpu', weights_only=False)
        sckpt = shared_ckpt_data
        print(f"  Saved at iteration: {sckpt['iteration']}")
        print(f"  Number of Gaussians: {sckpt['n_gaussians']}")

        # --- Gaussian parameters ---
        gaussians.active_sh_degree = sckpt['active_sh_degree']
        gaussians._xyz = nn.Parameter(sckpt['xyz'].cuda().requires_grad_(True))
        gaussians._features_dc = nn.Parameter(sckpt['features_dc'].cuda().requires_grad_(True))
        gaussians._features_rest = nn.Parameter(sckpt['features_rest'].cuda().requires_grad_(True))
        gaussians._scaling = nn.Parameter(sckpt['scaling'].cuda().requires_grad_(True))
        gaussians._rotation = nn.Parameter(sckpt['rotation'].cuda().requires_grad_(True))
        gaussians._opacity = nn.Parameter(sckpt['opacity'].cuda().requires_grad_(True))
        gaussians._appearance_level = nn.Parameter(sckpt['appearance_level'].cuda().requires_grad_(True))
        gaussians.max_radii2D = sckpt['max_radii2D'].cuda()
        gaussians.spatial_lr_scale = sckpt['spatial_lr_scale']
        # Method/kernel-specific params (saved only when present at save time)
        if 'shape' in sckpt:
            gaussians._shape = nn.Parameter(sckpt['shape'].cuda().requires_grad_(True))
        if 'flex_beta' in sckpt:
            gaussians._flex_beta = nn.Parameter(sckpt['flex_beta'].cuda().requires_grad_(True))
        if 'gaussian_features' in sckpt and sckpt['gaussian_features'].numel() > 0:
            gaussians._gaussian_features = nn.Parameter(sckpt['gaussian_features'].cuda().requires_grad_(True))
            gaussians._gaussian_feat_dim = sckpt.get('gaussian_feat_dim', sckpt['gaussian_features'].shape[1])

        # Skip the from-scratch Scene init below — Gaussians already populated
        gaussians._loaded_from_checkpoint = True
        scene = Scene(dataset, gaussians, mcmc_fps=args.mcmc_fps, cap_max=args.cap_max, full_args=args)

        # Optimizer + densif state restore
        gaussians.training_setup(opt)
        if 'gaussians_optimizer_state' in sckpt:
            try:
                ckpt_groups = len(sckpt['gaussians_optimizer_state']['param_groups'])
                cur_groups = len(gaussians.optimizer.param_groups)
                if ckpt_groups == cur_groups:
                    gaussians.optimizer.load_state_dict(sckpt['gaussians_optimizer_state'])
                    for state in gaussians.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                    print(f"  Gaussians optimizer state: restored ({ckpt_groups} param groups)")
                else:
                    print(f"  [WARN] Optimizer state mismatch: ckpt has {ckpt_groups} groups, "
                          f"current has {cur_groups}. Skipping optimizer state load.")
            except Exception as _e:
                print(f"  [WARN] Failed to restore Gaussians optimizer: {_e}")
        if 'xyz_gradient_accum' in sckpt:
            gaussians.xyz_gradient_accum = sckpt['xyz_gradient_accum'].cuda()
            gaussians.denom = sckpt['denom'].cuda()
            if 'feat_gradient_accum' in sckpt:
                gaussians.feat_gradient_accum = sckpt['feat_gradient_accum'].cuda()
            print(f"  Densification accumulators: restored")
        # gs_alpha_masks
        gs_alpha_masks = sckpt.get('gs_alpha_masks', {})
        for cam in scene.getTrainCameras():
            if cam.image_name in gs_alpha_masks:
                cam.gs_alpha_mask = gs_alpha_masks[cam.image_name].cpu().float()

        first_iter = sckpt['iteration']
        loaded_from_shared = True
        # Suppress the warmup-ckpt save branch (we have a more complete snapshot)
        loaded_from_warmup = True
        print(f"  GS alpha masks loaded: {len(gs_alpha_masks)}")
        print(f"  Resuming from iteration {first_iter + 1}")
        print(f"  (INGP weights + optimizer will be applied after INGP construction)")
        print("="*70 + "\n")
    elif cfg_model.settings.if_ingp and args.method != "2dgs" and os.path.exists(warmup_checkpoint_path) and not args.scratch:
        # Load warmup checkpoint - skip 2DGS phase
        print("\n" + "="*70)
        print("  LOADING 2DGS WARMUP CHECKPOINT")
        print("="*70)
        print(f"  Path: {warmup_checkpoint_path}")
        
        ckpt = torch.load(warmup_checkpoint_path, map_location='cpu', weights_only=False)
        print(f"  Saved at iteration: {ckpt['iteration']}")
        print(f"  Number of Gaussians: {ckpt['n_gaussians']}")
        
        # Load Gaussian parameters
        gaussians.active_sh_degree = ckpt['active_sh_degree']
        gaussians._xyz = nn.Parameter(ckpt['xyz'].cuda().requires_grad_(True))
        gaussians._features_dc = nn.Parameter(ckpt['features_dc'].cuda().requires_grad_(True))
        gaussians._features_rest = nn.Parameter(ckpt['features_rest'].cuda().requires_grad_(True))
        gaussians._scaling = nn.Parameter(ckpt['scaling'].cuda().requires_grad_(True))
        gaussians._rotation = nn.Parameter(ckpt['rotation'].cuda().requires_grad_(True))
        gaussians._opacity = nn.Parameter(ckpt['opacity'].cuda().requires_grad_(True))
        gaussians._appearance_level = nn.Parameter(ckpt['appearance_level'].cuda().requires_grad_(True))
        gaussians.max_radii2D = ckpt['max_radii2D'].cuda()
        gaussians.spatial_lr_scale = ckpt['spatial_lr_scale']

        # Apply FPS subsampling to loaded Gaussians if mcmc_fps is enabled
        if args.mcmc_fps:
            n_loaded = len(gaussians._xyz)
            if args.cap_max > 0 and n_loaded > args.cap_max:
                print(f"  [FPS] Loaded Gaussians: {n_loaded}, cap_max: {args.cap_max}")
                print(f"  [FPS] Subsampling loaded Gaussians using FPS...")

                # Use FPS to select indices
                from utils.point_cloud_utils import farthest_point_subsample
                xyz_np = gaussians._xyz.detach().cpu().numpy()
                selected_indices = farthest_point_subsample(xyz_np, args.cap_max)
                selected_indices = torch.tensor(selected_indices, device="cuda")

                # Subsample all Gaussian parameters
                gaussians._xyz = nn.Parameter(gaussians._xyz[selected_indices].requires_grad_(True))
                gaussians._features_dc = nn.Parameter(gaussians._features_dc[selected_indices].requires_grad_(True))
                gaussians._features_rest = nn.Parameter(gaussians._features_rest[selected_indices].requires_grad_(True))
                gaussians._scaling = nn.Parameter(gaussians._scaling[selected_indices].requires_grad_(True))
                gaussians._rotation = nn.Parameter(gaussians._rotation[selected_indices].requires_grad_(True))
                gaussians._opacity = nn.Parameter(gaussians._opacity[selected_indices].requires_grad_(True))
                gaussians._appearance_level = nn.Parameter(gaussians._appearance_level[selected_indices].requires_grad_(True))
                gaussians.max_radii2D = gaussians.max_radii2D[selected_indices]

                # Reset gradient accumulators for new size (will be properly initialized in training_setup)
                n_new = len(gaussians._xyz)
                gaussians.xyz_gradient_accum = torch.zeros((n_new, 1), device="cuda")
                gaussians.feat_gradient_accum = torch.zeros((n_new, 1), device="cuda")
                gaussians.denom = torch.zeros((n_new, 1), device="cuda")

                print(f"  [FPS] Subsampled to {n_new} Gaussians")
            else:
                # Warmup has FEWER points than the requested cap_max. Clamp cap_max
                # DOWN to the loaded warmup count so MCMC doesn't spend the whole run
                # densifying up to an oversized cap (which insanely slows indoor /
                # low-point scenes). The warmup point count becomes the effective cap.
                if args.cap_max <= 0 or n_loaded < args.cap_max:
                    print(f"  [FPS] loaded {n_loaded} < cap_max {args.cap_max}: clamping cap_max -> {n_loaded} "
                          f"(warmup count becomes cap_max; no over-densification)")
                    args.cap_max = n_loaded
                else:
                    print(f"  [FPS] Using all {n_loaded} loaded Gaussians (== cap_max {args.cap_max})")

        # Create scene (won't reinitialize Gaussians)
        gaussians._loaded_from_checkpoint = True
        scene = Scene(dataset, gaussians, mcmc_fps=args.mcmc_fps, cap_max=args.cap_max, full_args=args)

        # Initialize per-Gaussian features for cat/cat_dropout/3D_direct/3D_direct_fused mode (trained from scratch after warmup)
        if args.method in ["cat", "cat_dropout", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC"] and args.hybrid_levels > 0:
            per_level_dim = 4  # From config encoding.hashgrid.dim
            gaussians._gaussian_feat_dim = args.hybrid_levels * per_level_dim
            gaussian_feats = torch.zeros((len(gaussians.get_xyz), gaussians._gaussian_feat_dim), device="cuda").float()
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
        else:
            gaussians._gaussian_feat_dim = 0
            gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

        # --method film: (re)initialize packed FiLM params for the loaded point count.
        # Hash is inactive during 2DGS warmup so this is the fresh start. Defaults
        # gamma=1/beta=0 (identity); --film_gamma_init/--film_beta_init bias toward beta.
        if args.method in ("film", "3D_SH_filmres", "3D_SH_concat"):
            N = len(gaussians.get_xyz)
            g_init = getattr(args, 'film_gamma_init', 1.0); b_init = getattr(args, 'film_beta_init', 0.0)
            film_init = torch.cat([g_init * torch.ones((N, 1), device="cuda"), b_init * torch.ones((N, 24), device="cuda")], dim=1).float()
            # gamma_sigm_split: cols 22:25 hold the per-level gammas 1..3 — init them like col 0
            # so all 4 levels start identically.
            if getattr(args, 'film_act', 'identity') == 'gamma_sigm_split':
                film_init[:, 22:25] = g_init
            gaussians._film_params = nn.Parameter(film_init.requires_grad_(True))
        
        # Initialize adaptive mode parameters (trained from scratch after warmup)
        if args.method == "adaptive":
            # Use total_levels from config (same as hashgrid)
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gaussians._adaptive_feat_dim = num_levels * per_level_dim
            gaussians._adaptive_num_levels = num_levels
            
            # Initialize gamma to -1.0 (favors hashgrid initially)
            gamma_init = -1.0 * torch.ones((gaussians.get_xyz.shape[0], 1), device="cuda").float()
            gaussians._gamma = nn.Parameter(gamma_init.requires_grad_(True))
            
            # Initialize adaptive features to small random values
            adaptive_feats = torch.randn((gaussians.get_xyz.shape[0], gaussians._adaptive_feat_dim), device="cuda").float() * 0.01
            gaussians._adaptive_features = nn.Parameter(adaptive_feats.requires_grad_(True))
        elif args.method == "adaptive_add":
            # adaptive_add mode: per-Gaussian features + weight for blending with hashgrid
            # Use total_levels from config (same as hashgrid)
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gaussians._adaptive_feat_dim = num_levels * per_level_dim
            gaussians._adaptive_num_levels = num_levels
            
            # Initialize gamma (blend weight) to 0.0 (sigmoid(0) = 0.5, equal blend)
            gamma_init = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda").float()
            gaussians._gamma = nn.Parameter(gamma_init.requires_grad_(True))
            
            # Initialize adaptive features to small random values
            adaptive_feats = torch.randn((gaussians.get_xyz.shape[0], gaussians._adaptive_feat_dim), device="cuda").float() * 0.01
            gaussians._adaptive_features = nn.Parameter(adaptive_feats.requires_grad_(True))
            
            print(f"[ADAPTIVE_ADD MODE] Initialized {len(gaussians.get_xyz)} Gaussians")
            print(f"[ADAPTIVE_ADD MODE] Per-Gaussian features: {gaussians._adaptive_feat_dim}D")
            print(f"[ADAPTIVE_ADD MODE] Blend weight (gamma): 1D per Gaussian")
        elif args.method == "adaptive_cat":
            # adaptive_cat mode: per-Gaussian features (total_levels × D) + blend weight
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gaussians._gaussian_feat_dim = num_levels * per_level_dim

            # Initialize per-Gaussian features to small random values
            gaussian_feats = torch.randn((len(gaussians.get_xyz), gaussians._gaussian_feat_dim), device="cuda").float() * 0.01
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            # Initialize blend weight to 0.0 (sigmoid(0) = 0.5, equal blend initially)
            blend_weight = torch.zeros((len(gaussians.get_xyz), 1), device="cuda").float()
            gaussians._adaptive_cat_weight = nn.Parameter(blend_weight.requires_grad_(True))

            print(f"[ADAPTIVE_CAT MODE] Initialized {len(gaussians.get_xyz)} Gaussians")
            print(f"[ADAPTIVE_CAT MODE] Per-Gaussian features: {gaussians._gaussian_feat_dim}D")
            print(f"[ADAPTIVE_CAT MODE] Blend weight: 1D per Gaussian (starts at 0.5)")
        elif args.method == "adaptive_zero":
            # adaptive_zero mode: cat-like features (hybrid_levels × D) + blend weight for hash
            num_levels = cfg_model.encoding.levels
            hybrid_levels = args.hybrid_levels  # From CLI, like cat mode
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gaussians._gaussian_feat_dim = hybrid_levels * per_level_dim  # Same as cat mode

            # Initialize per-Gaussian features (coarse levels only)
            gaussian_feats = torch.randn((len(gaussians.get_xyz), gaussians._gaussian_feat_dim), device="cuda").float() * 0.01
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            # Initialize weight to 0.0 (sigmoid(0) = 0.5)
            # weight=0 → zeros for fine levels, weight=1 → query hash
            weight = torch.zeros((len(gaussians.get_xyz), 1), device="cuda").float()
            gaussians._adaptive_zero_weight = nn.Parameter(weight.requires_grad_(True))

            print(f"[ADAPTIVE_ZERO MODE] Initialized {len(gaussians.get_xyz)} Gaussians")
            print(f"[ADAPTIVE_ZERO MODE] Per-Gaussian features (coarse): {gaussians._gaussian_feat_dim}D")
            print(f"[ADAPTIVE_ZERO MODE] Hash weight: 1D per Gaussian (w=0→zeros, w=1→hash)")
        elif args.method == "adaptive_gate":
            # adaptive_gate mode: Gumbel-STE with forced training for binary hash selection
            # Always binary masking (0 or 1) to prevent scale compensation artifacts
            hybrid_levels = args.hybrid_levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gaussians._gaussian_feat_dim = hybrid_levels * per_level_dim

            # Initialize per-Gaussian features (coarse levels)
            gaussian_feats = torch.randn((len(gaussians.get_xyz), gaussians._gaussian_feat_dim), device="cuda").float() * 0.01
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            # Initialize gate logits to negative value for sparse start
            # sigmoid(-2.0) ≈ 0.12, sigmoid(-3.0) ≈ 0.05
            # Start sparse: mostly Gaussian-only, only turn on hash where needed
            gate_logits = torch.full((len(gaussians.get_xyz), 1), args.gate_init, device="cuda").float()
            gaussians._gate_logits = nn.Parameter(gate_logits.requires_grad_(True))

            print(f"[ADAPTIVE_GATE] Initialized {len(gaussians.get_xyz)} Gaussians with sparse gating")
            print(f"[ADAPTIVE_GATE] Per-Gaussian features (coarse): {gaussians._gaussian_feat_dim}D")
            print(f"[ADAPTIVE_GATE] Gate init: {args.gate_init} (sigmoid={torch.sigmoid(torch.tensor(args.gate_init)).item():.2f})")
            print(f"[ADAPTIVE_GATE] Force ratio: {args.force_ratio} ({args.force_ratio*100:.0f}% forced hash during training)")
        elif args.method == "3D":
            # 3D mode: per-Gaussian features for intersection-based SH rendering
            # Uses cat-style split: hybrid_levels for per-Gaussian (coarse), rest for hashgrid (fine)
            # Combined with hash features → MLP → SH coefficients
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gauss_feat_dim = args.hybrid_levels * per_level_dim  # Coarse levels as per-Gaussian features
            gaussians._gaussian_feat_dim = gauss_feat_dim

            # Initialize per-Gaussian features to zeros (matching CAT mode and 3D_direct)
            n_gaussians = len(gaussians.get_xyz)
            gaussian_feats = torch.zeros((n_gaussians, gauss_feat_dim), device="cuda").float()
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            # Store config for renderer
            gaussians._3D_mode = True

            print(f"[3D MODE] Initialized {n_gaussians} Gaussians (cat-style)")
            print(f"[3D MODE] Total levels: {num_levels}, Hybrid levels: {args.hybrid_levels}")
            print(f"[3D MODE] Per-Gaussian features (coarse): {args.hybrid_levels} × {per_level_dim} = {gauss_feat_dim}D")
            print(f"[3D MODE] Hashgrid features (fine): {num_levels - args.hybrid_levels} × {per_level_dim} = {(num_levels - args.hybrid_levels) * per_level_dim}D")
            print(f"[3D MODE] Max intersections per pixel: {args.max_intersections_per_pixel}")

        elif args.method == "3D_direct":
            # 3D_direct mode: per-Gaussian features for intersection-based direct RGB rendering
            # Uses cat-style split: hybrid_levels for per-Gaussian (coarse), rest for hashgrid (fine)
            # Combined with hash features + view encoding → MLP → RGB (no SH step)
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gauss_feat_dim = args.hybrid_levels * per_level_dim  # Coarse levels as per-Gaussian features
            gaussians._gaussian_feat_dim = gauss_feat_dim

            # Initialize per-Gaussian features to zeros (matching CAT mode)
            n_gaussians = len(gaussians.get_xyz)
            gaussian_feats = torch.zeros((n_gaussians, gauss_feat_dim), device="cuda").float()
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            # Store config for renderer
            gaussians._3D_direct_mode = True

            print(f"[3D_DIRECT MODE] Initialized {n_gaussians} Gaussians (cat-style)")
            print(f"[3D_DIRECT MODE] Total levels: {num_levels}, Hybrid levels: {args.hybrid_levels}")
            print(f"[3D_DIRECT MODE] Per-Gaussian features (coarse): {args.hybrid_levels} × {per_level_dim} = {gauss_feat_dim}D")
            print(f"[3D_DIRECT MODE] Hashgrid features (fine): {num_levels - args.hybrid_levels} × {per_level_dim} = {(num_levels - args.hybrid_levels) * per_level_dim}D")
            print(f"[3D_DIRECT MODE] Max intersections per pixel: {args.max_intersections_per_pixel}")

        elif args.method in ("3D_SH_32", "3D_SH_concat"):
            # 3D_SH_32: per-Gaussian SH + 32-dim hash MLP residual (input = [hash|pad])
            # 3D_SH_concat: same 32-dim MLP but input = concat[surfel latent(16) | hash(16)]
            gaussians._gaussian_feat_dim = 0
            gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            n_gaussians = len(gaussians.get_xyz)
            _mode_tag = "3D_SH_CONCAT" if args.method == "3D_SH_concat" else "3D_SH_32"
            print(f"[{_mode_tag} MODE] Initialized {n_gaussians} Gaussians")
            print(f"[{_mode_tag} MODE] Per-Gaussian: standard SH (degree-3, 48 params)")
            print(f"[{_mode_tag} MODE] Hash levels: {num_levels}, {per_level_dim}D per level")
            print(f"[{_mode_tag} MODE] MLP: 32D → 32D → 32D → 3D (RGB residual, identity)"
                  + (" | input = [latent(16)|hash(16)]" if args.method == "3D_SH_concat" else ""))

        elif args.method in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
            # 3D_SH_res / mixed: per-Gaussian SH + tiny hash MLP residual
            # No per-Gaussian features needed — standard SH handles per-Gaussian appearance
            # SH is kept from warmup checkpoint (or point cloud init) — not reinitialized
            gaussians._gaussian_feat_dim = 0
            gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            n_gaussians = len(gaussians.get_xyz)
            print(f"[3D_SH_RES MODE] Initialized {n_gaussians} Gaussians")
            print(f"[3D_SH_RES MODE] Per-Gaussian: standard SH (degree-3, 48 params)")
            print(f"[3D_SH_RES MODE] Hash levels: {num_levels}, {per_level_dim}D per level")
            print(f"[3D_SH_RES MODE] MLP: 16D → 16D → 16D → 3D (RGB residual, identity)")

        elif args.method == "3D_SH_cat":
            # 3D_SH_cat: per-Gaussian SH + hash+DC MLP residual
            # Same as 3D_SH_res but MLP input includes DC SH for per-Gaussian identity
            # SH is kept from warmup checkpoint (or point cloud init) — not reinitialized
            gaussians._gaussian_feat_dim = 0
            gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            n_gaussians = len(gaussians.get_xyz)
            print(f"[3D_SH_CAT MODE] Initialized {n_gaussians} Gaussians")
            print(f"[3D_SH_CAT MODE] Per-Gaussian: standard SH (degree-3, 48 params)")
            print(f"[3D_SH_CAT MODE] Hash levels: {num_levels}, {per_level_dim}D per level")
            print(f"[3D_SH_CAT MODE] MLP input: [hash(4)|DC_SH(3)|bias(1)]=8D → 16D → 16D → 3D (RGB residual)")

        elif args.method in ["3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC"]:
            # 3D_direct_fused/lean/SH_TC mode: fused in-kernel MLP rendering
            # Like cat mode but MLP runs inside CUDA kernel, outputs RGB directly
            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            gauss_feat_dim = args.hybrid_levels * per_level_dim  # Coarse levels as per-Gaussian features
            gaussians._gaussian_feat_dim = gauss_feat_dim

            # Initialize per-Gaussian features to zeros (matching CAT mode)
            n_gaussians = len(gaussians.get_xyz)
            gaussian_feats = torch.zeros((n_gaussians, gauss_feat_dim), device="cuda").float()
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

            print(f"[3D_DIRECT_FUSED MODE] Initialized {n_gaussians} Gaussians (cat-style)")
            print(f"[3D_DIRECT_FUSED MODE] Total levels: {num_levels}, Hybrid levels: {args.hybrid_levels}")
            print(f"[3D_DIRECT_FUSED MODE] Per-Gaussian features (coarse): {args.hybrid_levels} × {per_level_dim} = {gauss_feat_dim}D")
            print(f"[3D_DIRECT_FUSED MODE] Hashgrid features (fine): {num_levels - args.hybrid_levels} × {per_level_dim} = {(num_levels - args.hybrid_levels) * per_level_dim}D")

        # Initialize beta kernel shape parameter (if using beta or beta_scaled kernel)
        if args.kernel in ["beta", "beta_scaled"]:
            # Initialize _shape such that sigmoid(_shape) * 4 + 0.001 starts close to 4.0 (soft Gaussian-like)
            # sigmoid(5.0) ≈ 0.993 -> 0.993 * 4 + 0.001 ≈ 3.97
            n_gaussians = len(gaussians.get_xyz)
            shape_init_val = 5.0  # Results in shape ≈ 3.97
            shape_init = torch.full((n_gaussians, 1), shape_init_val, device="cuda").float()
            gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
            init_shape_val = (torch.sigmoid(torch.tensor(shape_init_val)) * 4.0 + 0.001).item()
            kernel_name = "BETA" if args.kernel == "beta" else "BETA_SCALED"
            print(f"[{kernel_name} KERNEL] Initialized {n_gaussians} Gaussians with shape parameter")
            print(f"[{kernel_name} KERNEL] Shape tensor: {gaussians._shape.shape}, numel={gaussians._shape.numel()}")
            print(f"[{kernel_name} KERNEL] Initial shape value: {init_shape_val:.3f} (will be pushed toward 0 by regularization)")
        elif args.kernel == "flex":
            # Initialize _flex_beta such that softplus(_flex_beta) starts at 0 (standard Gaussian)
            # softplus(x) = log(1 + exp(x)), so softplus(-5) ≈ 0.007, softplus(0) ≈ 0.693
            # We want to start at 0 (standard Gaussian), so use large negative value
            n_gaussians = len(gaussians.get_xyz)
            flex_beta_init_val = 5.0  # softplus(5) ≈ 5.007, starts sharp/hard
            flex_beta_init = torch.full((n_gaussians, 1), flex_beta_init_val, device="cuda").float()
            gaussians._flex_beta = nn.Parameter(flex_beta_init.requires_grad_(True))
            init_beta_val = torch.nn.functional.softplus(torch.tensor(flex_beta_init_val)).item()
            print(f"[FLEX KERNEL] Initialized {n_gaussians} Gaussians with per-Gaussian beta")
            print(f"[FLEX KERNEL] _flex_beta tensor: {gaussians._flex_beta.shape}, numel={gaussians._flex_beta.numel()}")
            print(f"[FLEX KERNEL] Initial beta value: {init_beta_val:.4f} (0=standard Gaussian, higher=sharper)")
        elif args.kernel == "general":
            # Initialize _shape such that sigmoid(_shape) * 6.0 + 2.0 starts at 2.0 (standard Gaussian)
            # sigmoid(-10) ≈ 0 -> 0 * 6 + 2 = 2.0
            n_gaussians = len(gaussians.get_xyz)
            shape_init_val = -10.0  # Results in beta ≈ 2.0 (standard Gaussian)
            shape_init = torch.full((n_gaussians, 1), shape_init_val, device="cuda").float()
            gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
            init_shape_val = (torch.sigmoid(torch.tensor(shape_init_val)) * 6.0 + 2.0).item()
            print(f"[GENERAL KERNEL] Initialized {n_gaussians} Gaussians with shape parameter")
            print(f"[GENERAL KERNEL] _shape tensor: {gaussians._shape.shape}, numel={gaussians._shape.numel()}")
            print(f"[GENERAL KERNEL] Initial beta value: {init_shape_val:.3f} (2=Gaussian, 8=super-Gaussian/box)")

        # Set relocation mode for adaptive weights (clone from source or reset to 0)
        if args.method in ["adaptive_cat", "adaptive_zero", "adaptive_gate"]:
            gaussians._relocation_mode = args.relocation
            print(f"[RELOCATION MODE] {args.relocation} - new Gaussians will {'copy weights from source' if args.relocation == 'clone' else 'reset weights to 0 (sigmoid=0.5)'}")
        else:
            gaussians._adaptive_feat_dim = 0
            gaussians._adaptive_num_levels = 0
            gaussians._gamma = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            gaussians._adaptive_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        
        # Initialize diffuse mode: reinitialize SH from scratch with degree 0 only
        if args.method == "diffuse":
            gaussians._diffuse_mode = True
            gaussians._specular_mode = False
            gaussians._diffuse_ngp_mode = False
            gaussians._diffuse_offset_mode = False
            n_gaussians = len(gaussians.get_xyz)
            
            # Reinitialize features_dc to zeros (will be optimized)
            # SH DC: rgb = sh * 0.28209 + 0.5, so sh=0 gives gray (0.5, 0.5, 0.5)
            features_dc = torch.zeros((n_gaussians, 1, 3), device="cuda").float()
            gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
            
            # Zero out features_rest (not used with degree 0, but keep for compatibility)
            features_rest = torch.zeros((n_gaussians, 15, 3), device="cuda").float()
            gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(False))
            
            # Set active_sh_degree to 0
            gaussians.active_sh_degree = 0
            
            print(f"[DIFFUSE MODE] Initialized {n_gaussians} Gaussians with fresh SH degree 0")
            print(f"[DIFFUSE MODE] features_dc: {gaussians._features_dc.shape} (trainable)")
            print(f"[DIFFUSE MODE] features_rest: {gaussians._features_rest.shape} (frozen)")
        
        # Initialize specular mode: reinitialize full SH from scratch (2DGS style)
        elif args.method == "specular":
            gaussians._diffuse_mode = False
            gaussians._specular_mode = True
            gaussians._diffuse_ngp_mode = False
            gaussians._diffuse_offset_mode = False
            n_gaussians = len(gaussians.get_xyz)
            
            # Reinitialize features_dc to zeros
            features_dc = torch.zeros((n_gaussians, 1, 3), device="cuda").float()
            gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
            
            # Reinitialize features_rest to zeros (will be trained)
            features_rest = torch.zeros((n_gaussians, 15, 3), device="cuda").float()
            gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
            
            # Start with active_sh_degree = 0, will increase during training
            gaussians.active_sh_degree = 0
            
            print(f"[SPECULAR MODE] Initialized {n_gaussians} Gaussians with fresh SH (max degree 3)")
            print(f"[SPECULAR MODE] features_dc: {gaussians._features_dc.shape} (trainable)")
            print(f"[SPECULAR MODE] features_rest: {gaussians._features_rest.shape} (trainable)")
        
        # Initialize diffuse_ngp mode: diffuse SH + hashgrid on unprojected depth
        elif args.method == "diffuse_ngp":
            gaussians._diffuse_mode = False
            gaussians._specular_mode = False
            gaussians._diffuse_ngp_mode = True
            gaussians._diffuse_offset_mode = False
            n_gaussians = len(gaussians.get_xyz)
            
            # Reinitialize features_dc to zeros (diffuse component)
            features_dc = torch.zeros((n_gaussians, 1, 3), device="cuda").float()
            gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
            
            # Zero out features_rest (not used with degree 0)
            features_rest = torch.zeros((n_gaussians, 15, 3), device="cuda").float()
            gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(False))
            
            # Set active_sh_degree to 0 (diffuse only)
            gaussians.active_sh_degree = 0
            
            print(f"[DIFFUSE_NGP MODE] Initialized {n_gaussians} Gaussians with fresh SH degree 0")
            print(f"[DIFFUSE_NGP MODE] features_dc: {gaussians._features_dc.shape} (trainable)")
            print(f"[DIFFUSE_NGP MODE] Hashgrid will be queried on unprojected expected depth")
        
        # Initialize diffuse_offset mode: diffuse SH as xyz offset for hashgrid query
        elif args.method == "diffuse_offset":
            gaussians._diffuse_mode = False
            gaussians._specular_mode = False
            gaussians._diffuse_ngp_mode = False
            gaussians._diffuse_offset_mode = True
            n_gaussians = len(gaussians.get_xyz)
            
            # Reinitialize features_dc to zeros (will be used as xyz offset)
            features_dc = torch.zeros((n_gaussians, 1, 3), device="cuda").float()
            gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
            
            # Zero out features_rest (not used with degree 0)
            features_rest = torch.zeros((n_gaussians, 15, 3), device="cuda").float()
            gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(False))
            
            # Set active_sh_degree to 0 (diffuse only)
            gaussians.active_sh_degree = 0
            
            print(f"[DIFFUSE_OFFSET MODE] Initialized {n_gaussians} Gaussians with zero offsets")
            print(f"[DIFFUSE_OFFSET MODE] features_dc: {gaussians._features_dc.shape} (trainable, used as xyz offset)")
            print(f"[DIFFUSE_OFFSET MODE] Hashgrid queried at unprojected_xyz + rendered_offset")
        else:
            gaussians._diffuse_mode = False
            gaussians._specular_mode = False
            gaussians._diffuse_ngp_mode = False
            gaussians._diffuse_offset_mode = False
        
        # Setup optimizer (gaussian_features will be added to param groups if present)
        gaussians.training_setup(opt)
        
        # Load optimizer state from warmup checkpoint
        # Skip if method/kernel adds new params not in saved state (they train from scratch)
        # Also skip if param group counts don't match (checkpoint from different config)
        skip_methods = ["cat", "film", "adaptive", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "3D_SH_filmres"]
        # If FPS subsampling shrank the Gaussian set, the saved optimizer state is
        # sized for the full warmup set and no longer matches the (smaller) params.
        # Skip the restore and let the freshly-built optimizer start with empty Adam
        # state (correct for the relocated/subsampled MCMC set). Without this guard
        # the param-group COUNT matches but the per-tensor sizes don't → optimizer.step
        # crashes in _foreach_lerp_ (2.4M vs cap_max). cat-family methods already skip.
        fps_was_applied = args.mcmc_fps and len(gaussians._xyz) < ckpt['xyz'].shape[0]
        if args.method not in skip_methods and args.kernel == "gaussian" and not fps_was_applied:
            ckpt_groups = len(ckpt['optimizer_state']['param_groups'])
            cur_groups = len(gaussians.optimizer.param_groups)
            if ckpt_groups == cur_groups:
                gaussians.optimizer.load_state_dict(ckpt['optimizer_state'])
            else:
                print(f"  [WARN] Optimizer state mismatch: checkpoint has {ckpt_groups} param groups, current model has {cur_groups}. Skipping optimizer state load.")
        
        # Move optimizer state to GPU
        for state in gaussians.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        # Restore densification state (critical for identical behavior)
        # Skip if FPS subsampling was applied - the sizes won't match
        fps_was_applied = args.mcmc_fps and len(gaussians._xyz) < ckpt['xyz'].shape[0]
        if 'xyz_gradient_accum' in ckpt and not fps_was_applied:
            gaussians.xyz_gradient_accum = ckpt['xyz_gradient_accum'].cuda()
            gaussians.denom = ckpt['denom'].cuda()
            if 'feat_gradient_accum' in ckpt:
                gaussians.feat_gradient_accum = ckpt['feat_gradient_accum'].cuda()
        
        # Load gs_alpha masks for all cameras
        gs_alpha_masks = ckpt['gs_alpha_masks']
        for cam in scene.getTrainCameras():
            if cam.image_name in gs_alpha_masks:
                cam.gs_alpha_mask = gs_alpha_masks[cam.image_name].cpu().float()
        
        # Resume from after the warmup iteration
        first_iter = ckpt['iteration']
        loaded_from_warmup = True

        print(f"  GS Alpha masks loaded: {len(gs_alpha_masks)}")
        print(f"  Densification state: {'restored' if 'xyz_gradient_accum' in ckpt else 'reset (old checkpoint)'}")
        print(f"  Resuming from iteration {first_iter + 1}")

        print("="*70 + "\n")
    else:
        # Normal initialization - train from scratch
        mini_res_scales = [1.0, 0.5] if (args.mini and args.mini_warmup) else [1.0]
        # Skip mcmc_fps subsampling when init_ply is provided (we'll load our own points)
        scene = Scene(dataset, gaussians, resolution_scales=mini_res_scales,
                      mcmc_fps=(args.mcmc_fps and not args.init_ply),
                      cap_max=args.cap_max, full_args=args)

        # Override initialization with external PLY
        if args.init_ply:
            print(f"\n[INIT_PLY] Loading Gaussians from: {args.init_ply}")
            gaussians.load_ply(args.init_ply, args=args)
            n_gs = len(gaussians.get_xyz)
            print(f"[INIT_PLY] Loaded {n_gs} Gaussians, SH degree={gaussians.active_sh_degree}")

            # Override cap_max to match loaded PLY if user didn't set a specific value
            if args.mcmc or args.mcmc_fps or args.mcmc_deficit:
                print(f"[INIT_PLY] cap_max: {args.cap_max} → {max(args.cap_max, n_gs)} (at least loaded PLY count)")
                args.cap_max = max(args.cap_max, n_gs)

            # Ensure ap_level is 24
            gaussians._appearance_level = nn.Parameter(
                torch.ones(n_gs, 1, device="cuda") * 24, requires_grad=False)

            # Re-initialize per-Gaussian features for the target method
            if hasattr(args, 'method') and args.method in ["film", "3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
                gaussians._gaussian_feat_dim = 0
                gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
                # FiLM: ensure packed gamma/beta exist at the loaded point count (load_ply
                # restores from PLY when present; this covers the no-film-columns case).
                # Defaults gamma=1/beta=0; --film_gamma_init/--film_beta_init bias toward beta.
                if args.method in ("film", "3D_SH_filmres", "3D_SH_concat") and (not hasattr(gaussians, '_film_params') or gaussians._film_params.numel() == 0 or gaussians._film_params.shape[0] != n_gs):
                    g_init = getattr(args, 'film_gamma_init', 1.0); b_init = getattr(args, 'film_beta_init', 0.0)
                    film_init = torch.cat([g_init * torch.ones((n_gs, 1), device="cuda"), b_init * torch.ones((n_gs, 24), device="cuda")], dim=1).float()
                    # gamma_sigm_split: per-level gammas 1..3 live in cols 22:25 — init like col 0.
                    if getattr(args, 'film_act', 'identity') == 'gamma_sigm_split':
                        film_init[:, 22:25] = g_init
                    gaussians._film_params = nn.Parameter(film_init.requires_grad_(True))
            elif hasattr(args, 'method') and args.method in ["cat"] and hasattr(args, 'hybrid_levels'):
                per_level_dim = 4
                gaussians._gaussian_feat_dim = args.hybrid_levels * per_level_dim
                if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features.numel() > 0 and gaussians._gaussian_features.shape[1] == gaussians._gaussian_feat_dim:
                    print(f"[INIT_PLY] Keeping existing per-Gaussian features ({gaussians._gaussian_feat_dim}D)")
                else:
                    gaussians._gaussian_features = nn.Parameter(
                        torch.randn(n_gs, gaussians._gaussian_feat_dim, device="cuda") * 0.01)
                    print(f"[INIT_PLY] Re-initialized per-Gaussian features ({gaussians._gaussian_feat_dim}D)")

            # Reset accumulators for new Gaussian count
            gaussians.max_radii2D = torch.zeros(n_gs, device="cuda")

            print(f"[INIT_PLY] Using loaded geometry + SH as initialization\n")

        # Initialize flex kernel per-Gaussian beta parameter (if using flex kernel)
        if args.kernel == "flex":
            n_gaussians = len(gaussians.get_xyz)
            flex_beta_init_val = 5.0  # softplus(5) ≈ 5.007, starts sharp/hard
            flex_beta_init = torch.full((n_gaussians, 1), flex_beta_init_val, device="cuda").float()
            gaussians._flex_beta = nn.Parameter(flex_beta_init.requires_grad_(True))
            init_beta_val = torch.nn.functional.softplus(torch.tensor(flex_beta_init_val)).item()
            print(f"[FLEX KERNEL] Initialized {n_gaussians} Gaussians with per-Gaussian beta")
            print(f"[FLEX KERNEL] Initial beta value: {init_beta_val:.4f} (0=standard Gaussian, higher=sharper)")
        elif args.kernel == "general":
            n_gaussians = len(gaussians.get_xyz)
            shape_init_val = -10.0  # sigmoid(-10) ≈ 0 -> beta = 0*6+2 = 2.0 (standard Gaussian)
            shape_init = torch.full((n_gaussians, 1), shape_init_val, device="cuda").float()
            gaussians._shape = nn.Parameter(shape_init.requires_grad_(True))
            init_shape_val = (torch.sigmoid(torch.tensor(shape_init_val)) * 6.0 + 2.0).item()
            print(f"[GENERAL KERNEL] Initialized {n_gaussians} Gaussians with shape parameter")
            print(f"[GENERAL KERNEL] Initial beta value: {init_shape_val:.3f} (2=Gaussian, 8=super-Gaussian/box)")

        gaussians.training_setup(opt)
        if args.scratch:
            print(f"\n[INFO] --scratch flag: ignoring warmup checkpoint, training from scratch")
            print(f"[INFO] Will train 2DGS for {cfg_model.ingp_stage.initialize} iterations, then save checkpoint.\n")
        elif cfg_model.settings.if_ingp:
            print(f"\n[INFO] No warmup checkpoint found at {warmup_checkpoint_path}")
            print(f"[INFO] Will train 2DGS for {cfg_model.ingp_stage.initialize} iterations, then save checkpoint.\n")

    # mcmc_fps: auto-set cap_max ONLY if the user didn't specify one.
    # Previously this also clamped cap_max down to num_init when num_init < cap_max,
    # which conflicted with the new FPS-halving path where cap_max is intentionally
    # larger than the subsampled init count (MCMC grows back to cap_max).
    if args.mcmc_fps:
        n_current = len(gaussians.get_xyz)
        if args.cap_max <= 0:
            print(f"[mcmc_fps] --cap_max unset; auto-setting to current Gaussian count: {n_current}")
            args.cap_max = n_current

    surfel_cfg = cfg_model.surfel

    # Override tg_beta if --beta is specified
    if args.beta is not None:
        surfel_cfg.tg_beta = args.beta
        print(f"[OVERRIDE] tg_beta set to {args.beta} (from --beta argument)")

    gaussians.base_opacity = surfel_cfg.base_opacity
    beta = surfel_cfg.base_beta
    print(f'base opacity {surfel_cfg.base_opacity}, base beta {beta}, target beta {surfel_cfg.tg_beta}')

    if not os.path.exists(os.path.join(scene.model_path, "training_output")):
        os.mkdir(os.path.join(scene.model_path, "training_output"))

    # Initialise test_metrics.txt with a periodic-eval header. training_report appends
    # one row per periodic test eval; render_final_images appends the final block.
    test_metrics_path = os.path.join(scene.model_path, 'test_metrics.txt')
    with open(test_metrics_path, 'w') as f:
        f.write("Periodic Test Eval (during training, full test set)\n")
        f.write("LPIPS(L) = legacy 3DGS-ecosystem convention (no rescale, ~18% lower than canonical)\n")
        f.write("LPIPS(C) = canonical Zhang spec ([-1,1] rescaled before VGG)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Iter':<10}{'PSNR(dB)':<10}{'SSIM':<9}{'LPIPS(L)':<11}{'LPIPS(C)':<11}{'L1':<12}{'Points':<12}\n")
        f.write("-" * 75 + "\n")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # For diffuse_ngp/diffuse_offset: prepare alternating backgrounds to prevent RGB hiding
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    use_alternating_bg = args.method in ["diffuse_ngp", "diffuse_offset"]
    
    # Random background mode: use black BG until 10k iters, then random uniform background until 20k, then black again
    use_random_bg = args.random_background
    random_bg_start_iter = 0
    random_bg_end_iter = 14000
    if use_random_bg:
        print(f"Using black background until iteration {random_bg_start_iter}, then random uniform background until {random_bg_end_iter}, then black background (eval will use black background)")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_converge_for_log = 0.0
    ema_converge_raw_for_log = 0.0  # raw Converge.mean() before lambda scaling
    ema_mask_for_log = 0.0
    ema_mcmc_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    mini_sh_unfreeze_iter = 0  # Set by depth reinit to temporarily freeze SH
    mini_last_visibility = None  # MSv2: visibility mask for SparseGaussianAdam
    minimc_noise_disabled_until = 0  # MiniMC: disable MCMC noise after depth reinit

    ingp_model = None
    if cfg_model.settings.if_ingp and args.method != "2dgs":
        ingp_model = INGP(cfg_model, args=args).to('cuda')

        # --finetune_from: restore the trained hash table + residual MLP. Same
        # ordering constraint as the shared-ckpt path below — INGP.__init__ has to
        # build the tables from cfg_model first, then we overwrite the weights.
        # Weights only: the Adam moments are intentionally dropped.
        if getattr(args, "_finetune_ngp", None) is not None:
            _ft_dir, _ft_it = args._finetune_ngp
            ingp_model.load_model(_ft_dir, _ft_it)
            print(f"[FINETUNE] Restored hash+MLP from ngp_{_ft_it}.pth")

        # Restore INGP state from shared-resume checkpoint (if loaded earlier).
        # INGP construction has to happen first because its __init__ builds the
        # hash table + MLPs using cfg_model — we then overwrite with the saved
        # state_dict and Adam moments.
        if loaded_from_shared and shared_ckpt_data is not None:
            if 'ingp_state_dict' in shared_ckpt_data:
                try:
                    ingp_model.load_state_dict(shared_ckpt_data['ingp_state_dict'], strict=False)
                    print(f"[SHARED CKPT] INGP weights restored ({len(shared_ckpt_data['ingp_state_dict'])} keys)")
                except Exception as _e:
                    print(f"[SHARED CKPT] WARN: INGP state_dict load failed: {_e}")
            if 'ingp_optimizer_state' in shared_ckpt_data and hasattr(ingp_model, 'optimizer'):
                try:
                    ingp_model.optimizer.load_state_dict(shared_ckpt_data['ingp_optimizer_state'])
                    for state in ingp_model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                    print(f"[SHARED CKPT] INGP optimizer state restored")
                except Exception as _e:
                    print(f"[SHARED CKPT] WARN: INGP optimizer load failed: {_e}")
            # Drop the buffer once applied — large tensors don't need to linger.
            shared_ckpt_data = None

    # ---- Footprint mult (--fastgs_mult): ONE knob for BOTH kernel families ----
    # Tightens the per-primitive tile box DURING TRAINING so primitives adapt to it
    # (the FastGS trained-in-mult idea). Dispatched by --kernel:
    #   • beta (beta/beta_scaled) → set_beta_mult  → use_beta_cutoff branch (cutoff×mult,
    #     baseline ~3.3σ for beta_scaled; support ends at 3σ ≈ 0.9·baseline).
    #   • gaussian (else)         → set_compact_mult → use_adr_cutoff branch
    #     (opacity-aware cutoff = sqrt(2·log(255·α)·mult), FastGS Compact Box).
    # DEFAULT 1.0 = OFF (no change). Fires under --fastgs OR standalone (any 3D_SH_res-
    # family method). Only the base diff_surfel_3D_sh_res rasterizer has the setters; forks
    # (filmres/concat) don't — fail loudly there instead of silently no-op'ing.
    _FOOTPRINT_METHODS = ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d",
                          "res_3d_paired", "res_3d_double", "mixed", "mixed_3d",
                          "mixed_sep", "mixed_3d_sep")
    if (args.fastgs or args.fastgs_mult != 1.0) and args.method in _FOOTPRINT_METHODS:
        _is_beta = getattr(args, "kernel", "gaussian") in ("beta", "beta_scaled")
        _setter_name = "set_beta_mult" if _is_beta else "set_compact_mult"
        _setter = getattr(_SHRES_SETTER_MOD, _setter_name, None)
        if _setter is None:
            raise SystemExit(
                f"--fastgs_mult requires the 3D_SH_res rasterizer ({_setter_name}); "
                f"method={getattr(args, 'method', None)} routes to a fork without it.")
        _setter(args.fastgs_mult)
        # Auto-pick an AABB mode whose cutoff branch the mult actually scales when the
        # caller left the default. NOTE: use "adr" (→ mode 3), NOT "adrrect" — the renderer
        # maps "adrrect" to the else-default (mode 0 / 2dgs), so the cutoff would never run.
        if args.aabb == "2dgs":
            args.aabb = "snugbox" if _is_beta else "adr"
            print(f"[FOOTPRINT_MULT] Auto-enabled --aabb {args.aabb} so the cutoff runs.")
        # Warn if the (kernel, aabb) combo lands on a branch the mult doesn't scale.
        # beta → scaled only on use_beta_cutoff (mode 4 beta / mode 5 snugbox|accutile);
        # gaussian → scaled only on use_adr_cutoff (mode 1 adr_only / 3 adr / 5 snugbox|accutile).
        _supported = ({"beta", "accutile", "snugbox"} if _is_beta
                      else {"adr_only", "adr", "accutile", "snugbox"})
        if args.aabb not in _supported:
            print(f"[FOOTPRINT_MULT] WARNING: --fastgs_mult={args.fastgs_mult} has NO EFFECT "
                  f"with --kernel {getattr(args,'kernel','gaussian')} + --aabb {args.aabb} "
                  f"(needs --aabb in {sorted(_supported)}).")
        _branch = ("beta use_beta_cutoff (cutoff = mult·max(k·1.1, r_lp_typical))" if _is_beta
                   else "gaussian use_adr_cutoff (cutoff = sqrt(2·log(255·α)·mult))")
        _tag = "FastGS" if args.fastgs else "standalone"
        print(f"[FOOTPRINT_MULT] {_tag}: --fastgs_mult={args.fastgs_mult} → {_branch}; "
              f"primitives adapt to the tighter box (1.0 = off).")

    # Set hash query transmittance threshold (skip hash+MLP when T < threshold)
    if args.contribution_thresh > 0.0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_contrib_thresh
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_contrib_thresh
        else:
            set_contrib_thresh = _SHRES_SETTER_MOD.set_contrib_thresh
        set_contrib_thresh(args.contribution_thresh)
        print(f"[CONTRIB_THRESH] Skipping hash query when w = T*alpha < {args.contribution_thresh}")

    if args.count_thresh > 0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_count_thresh
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_count_thresh
        else:
            set_count_thresh = _SHRES_SETTER_MOD.set_count_thresh
        set_count_thresh(args.count_thresh)
        print(f"[COUNT_THRESH] Skipping hash query after {args.count_thresh} contributing Gaussians per pixel")

    # `--densfix` (--method 3D_SH_res): install the device-global that makes the
    # AbsGS densification proxy exclude the hash-query-point term. _SHRES_SETTER_MOD
    # already resolves to the diff_surfel_3D_sh_res_densfix clone for this run.
    if getattr(args, 'densfix', False) and args.method == "3D_SH_res":
        if _SHRES_DENSFIX_MOD is None or not hasattr(_SHRES_SETTER_MOD, 'set_exclude_hash_from_densify'):
            raise RuntimeError("--densfix requires the diff_surfel_3D_sh_res_densfix module "
                               "(build it: cd submodules/diff_surfel_3D_sh_res_densfix && "
                               "pip install -e . --no-build-isolation)")
        _SHRES_SETTER_MOD.set_exclude_hash_from_densify(1)
        print("[DENSFIX] Excluding the hashgrid query-point term from the densification "
              "gradient (transMat/mean3D optimizer still uses the full SV+hash gradient).")

    # `--gap_noise` (--method 3D_SH_res): macro-gap truncation — rays end in
    # the first inter-manifold void and the remaining transmittance carries
    # the (1−rend_alpha)·noise composite. Requires the trunc clone.
    if getattr(args, 'gap_noise', False) and args.method == "3D_SH_res":
        if _SHRES_TRUNC_MOD is None or not hasattr(_SHRES_SETTER_MOD, 'set_gap_trunc'):
            raise RuntimeError("--gap_noise requires the diff_surfel_3D_sh_res_trunc module "
                               "(build it: cd submodules/diff_surfel_3D_sh_res_trunc && "
                               "pip install -e . --no-build-isolation)")
        _SHRES_SETTER_MOD.set_gap_trunc(0)
        print(f"[GAP_NOISE] v2 mass automaton armed: from iter {args.gap_noise_after}, "
              f"rays end in the first void longer than {args.gap_noise_thresh} holding "
              f"< {args.gap_noise_void_mass} opacity mass (margin {args.gap_noise_margin}, "
              f"arm_T {args.gap_noise_arm_T}, T_lo {args.gap_noise_T_lo}); remaining T "
              f"carries the noise composite.")
        if args.gap_noise_sat_T > 0:
            print(f"[GAP_NOISE] SATURATION GATE ON (sat_T {args.gap_noise_sat_T}): the wall "
                  f"only commits on rays whose full march saturates (final T < sat_T); "
                  f"background blends / semi-transparent floaters over bg take no pressure.")
        else:
            print("[GAP_NOISE][WARN] sat_T <= 0 — legacy IMMEDIATE wall (fires on all armed "
                  "rays, including background blends; known to bloat silhouette floaters).")
        if not getattr(args, 'random_mesh', False):
            print("[GAP_NOISE][WARN] --random_mesh is OFF — voids will composite plain "
                  "random background only; blocky noise gives much stronger pressure.")

    # `--trunc` (--method 3D_SH_res): validate the clone and announce the exit_T
    # ramp. The per-iteration set_exit_T call lives in the training loop; here we
    # just fail loudly if the module is missing and pin the default (1e-4 = off,
    # byte-identical to the base rasterizer) so pre-ramp iterations are exact.
    if getattr(args, 'trunc', False) and args.method == "3D_SH_res":
        if _SHRES_TRUNC_MOD is None or not hasattr(_SHRES_SETTER_MOD, 'set_exit_T'):
            raise RuntimeError("--trunc requires the diff_surfel_3D_sh_res_trunc module "
                               "(build it: cd submodules/diff_surfel_3D_sh_res_trunc && "
                               "pip install -e . --no-build-isolation)")
        _SHRES_SETTER_MOD.set_exit_T(1e-4)
        print(f"[TRUNC] Post-blend truncation exit armed: exit_T ramps 1e-4 → "
              f"{args.trunc_exit_T} over iters [{args.trunc_ramp_start}, {args.trunc_ramp_end}]. "
              f"Remaining T carries the --random_mesh noise composite (enable it!).")
        if not getattr(args, 'random_mesh', False):
            print("[TRUNC][WARN] --random_mesh is OFF — truncated transmittance will "
                  "composite plain random background only; the opacity-cliff pressure "
                  "is much weaker without the blocky noise term.")

    # Opacity threshold: skip hash query when the per-pixel opacity contribution
    # alpha = opa*kernel_val (the queried beta_scaled/Gaussian kernel × opacity) is below
    # threshold. Occlusion-independent (unlike contribution_thresh's w = T*alpha) → drops the
    # residual on the soft *tails of fuzzy surfels*. Scoped to 3D_SH_res (the method that
    # renders through diff_surfel_3D_sh_res, where the setter lives). mixed/mixed_3d/filmres/
    # concat/32 render through their own forks and would no-op — extend those if needed.
    if getattr(args, 'opacity_thresh', 0.0) > 0.0 and args.method == "3D_SH_res":
        from diff_surfel_3D_sh_res import set_opacity_thresh
        set_opacity_thresh(args.opacity_thresh)
        print(f"[OPACITY_THRESH] Skipping hash query when alpha = opa*kernel_val < {args.opacity_thresh}")

    # Texture-query dropout: enabled flag + setter resolved once here; the rate/seed is set
    # per-iteration around the main render+backward in the training loop (set before render,
    # reset to 0 right after backward) so eval/test/debug renders never drop queries.
    _use_tex_dropout = (getattr(args, 'texture_dropout', 0.0) > 0.0 and args.method == "3D_SH_res")
    if _use_tex_dropout:
        from diff_surfel_3D_sh_res import set_dropout as _set_tex_dropout
        print(f"[TEXTURE_DROPOUT] Dropping {args.texture_dropout*100:.0f}% of texture queries per iter "
              f"(per-Gauss, unscaled, training only)")
    # --texture_dropout_bw: BACKWARD-ONLY texture dropout. The forward renders the
    # full image (all textures); the dropout device-global is armed AFTER the forward,
    # so only the backward kernel's skip_hash gate sees it → the dropped Gaussians'
    # hash/MLP GRADIENTS are skipped while the rendered image (and thus the loss) is
    # exact. A gradient-sparsification regularizer, not an image perturbation.
    _use_tex_dropout_bw = (getattr(args, 'texture_dropout_bw', 0.0) > 0.0
                           and args.method == "3D_SH_res")
    if _use_tex_dropout_bw:
        if _use_tex_dropout:
            raise ValueError("--texture_dropout and --texture_dropout_bw are mutually exclusive.")
        from diff_surfel_3D_sh_res import set_dropout as _set_tex_dropout
        print(f"[TEXTURE_DROPOUT_BW] Dropping {args.texture_dropout_bw*100:.0f}% of texture "
              f"GRADIENTS per iter (backward-only; forward image exact)")

    if args.overdraw_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_overdraw_lambda
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_overdraw_lambda
        else:
            set_overdraw_lambda = _SHRES_SETTER_MOD.set_overdraw_lambda
        set_overdraw_lambda(args.overdraw_reg)
        print(f"[OVERDRAW_REG] Overdraw regularization lambda = {args.overdraw_reg}")

    if args.weight_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_weight_reg_lambda
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_weight_reg_lambda
        else:
            set_weight_reg_lambda = _SHRES_SETTER_MOD.set_weight_reg_lambda
        set_weight_reg_lambda(args.weight_reg)
        print(f"[WEIGHT_REG] Weight-squared regularization lambda = {args.weight_reg} (CUDA gradient)")

    if args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_activation_bias
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_activation_bias
        else:
            set_activation_bias = _SHRES_SETTER_MOD.set_activation_bias
        _sh_bias, _res_bias = args.activation_bias
        set_activation_bias(sh_bias=_sh_bias, res_bias=_res_bias)
        from gaussian_renderer import set_default_activation_bias
        set_default_activation_bias(_sh_bias, _res_bias)
        print(f"[ACTIVATION_BIAS] SH bias={_sh_bias}, residual bias={_res_bias}")

        # Flip the residual activation mode in the kernel device-global. Default 0.
        #   1 = 3D_SH_add (separate ReLUs), 2 = mixed (signed residual, per-pixel ReLU in Python).
        # For --method mixed the setter call is mirrored onto diff_surfel_mixed by the
        # monkey-patch installed at startup.
        _rm = getattr(args, '_residual_mode', 0)
        if _rm in (1, 2) and args.method in (
                "3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight"):
            set_residual_mode = _SHRES_SETTER_MOD.set_residual_mode
            set_residual_mode(_rm)
            _desc = ("3D_SH_add: separate outer ReLUs" if _rm == 1 else
                     "mixed_*_sep: signed residual, per-pixel ReLU in Python")
            print(f"[RESIDUAL_MODE] mode={_rm} ({_desc})")
        # No print for mode 0 — that's the default and matches 3D_SH_res.

        # `--ste`: SIGN-AWARE straight-through. Two paths, same gate logic:
        #   - Mode 0 (3D_SH_res / mixed / mixed_3d): per-Gauss outer ReLU
        #     gradient gates in CUDA (d_ste_relu). The mirror propagates to
        #     diff_surfel_mixed_3d for mixed-family methods.
        #   - Mode 2 (mixed_sep / mixed_3d_sep): per-pixel ReLU after blend,
        #     switched torch.relu → STERelu in the Python renderer via
        #     ingp.is_ste_relu (set in INGP.__init__).
        # Same flag enables both; the relevant path is selected by mode.
        if getattr(args, 'ste', False) and float(getattr(args, 'lru', 0.0)) != 0.0:
            raise RuntimeError("--ste and --lru are mutually exclusive: both modify the "
                               "outer per-Gauss ReLU at the same call sites. Pick one.")
        if getattr(args, 'ste', False):
            set_ste_relu = _SHRES_SETTER_MOD.set_ste_relu
            set_ste_relu(1)
            _rm_now = getattr(args, '_residual_mode', 0)
            _ste_site = ("per-pixel ReLU after blend (renderer torch.relu→STERelu)"
                         if _rm_now == 2 else
                         "per-Gauss outer ReLU in CUDA backward (d_ste_relu)")
            print(f"[STE_RELU] enabled (sign-aware) — gradient pass at clamped "
                  f"pixels gated to dL/dpixel < 0 (release-clamp direction). "
                  f"Active at: {_ste_site}.")

        # `--detach_res_shape_grad`: backward-only. Drives the per-Gauss
        # alpha/shape gradient from the SV (SH base) color only — detaches the
        # MLP residual from surfel-shape gradients (residual still moves
        # surfels via the hash-query xyz path + opacity). Forward unchanged.
        # Isolates "should high-freq residual reshape surfels, or should
        # geometry follow low-freq SV?". Default off → byte-identical.
        if getattr(args, 'detach_res_shape_grad', False):
            set_detach_res_shape_grad = _SHRES_SETTER_MOD.set_detach_res_shape_grad
            set_detach_res_shape_grad(1)
            print("[DETACH_RES_SHAPE_GRAD] enabled — surfel shape gradient "
                  "driven by SV only; MLP residual detached from shape "
                  "(still drives position via hash xyz + opacity).")

        # `--lru`: leaky-ReLU at the outer activation. Mirrors --ste's call
        # sites — CUDA d_lru_slope for the per-Gauss clamp (modes 0/1/cat),
        # Python F.leaky_relu for the per-pixel after-blend clamp (mode 2,
        # sep methods, surfaced as ingp.lru_slope below).
        _lru = float(getattr(args, 'lru', 0.0))
        if _lru != 0.0:
            set_lru_slope = _SHRES_SETTER_MOD.set_lru_slope
            set_lru_slope(_lru)
            _rm_now = getattr(args, '_residual_mode', 0)
            _lru_site = ("per-pixel LeakyReLU after blend (renderer)"
                         if _rm_now == 2 else
                         "per-Gauss outer LeakyReLU in CUDA fwd+bw (d_lru_slope)")
            print(f"[LRU] enabled α={_lru} — leaky-ReLU at the outer activation. "
                  f"Active at: {_lru_site}.")

        # --sv_lru: leaky INNER ReLU on the SV base (relu(SV+sh_bias)). Read by INGP as
        # ingp.sv_lru_slope and applied Python-side in the renderer's _build_fake_shs_from_SV
        # (F.leaky_relu). Independent of --lru; no CUDA rebuild for --feature SV.
        _sv_lru = float(getattr(args, 'sv_lru', 0.0))
        if _sv_lru != 0.0:
            print(f"[SV_LRU] enabled α={_sv_lru} — leaky INNER ReLU on the SV base "
                  f"(relu(SV+sh_bias)); Python-side for --feature SV. Keeps the base "
                  f"learnable when SV+bias goes negative (--lru only covers the outer site).")

        # --film_act (3D_SH_filmres): independent activation on the per-surfel FiLM gamma/beta.
        if args.method == "3D_SH_filmres":
            _fga = {"identity": 0, "gamma_relu": 1, "beta_relu": 2, "gamma_sigmoid": 3,
                    "beta_sigmoid": 4, "double_relu": 5, "double_sigmoid": 6,
                    "gamma_sigm_split": 7}[getattr(args, 'film_act', 'identity')]
            _SHRES_SETTER_MOD.set_film_gamma_act(_fga)
            if _fga == 7:
                print(f"[FILM] activation = gamma_sigm_split (d_film_gamma_act=7); "
                      f"mlp_input[i] = sigmoid(gamma_l)*H[i] + beta[i], l = i//l_dim; "
                      f"gamma_0 = col 0, gamma_1..3 = _film_params cols 22..24 (beta raw)")
            else:
                print(f"[FILM] activation = {args.film_act} (d_film_gamma_act={_fga}); "
                      f"mlp_input = gamma_act(gamma)*H + beta_act(beta)")
            # --lock_gamma X: pin gamma_eff = X (bypass gamma + its activation, freeze gamma grad).
            if getattr(args, 'lock_gamma', None) is not None:
                _SHRES_SETTER_MOD.set_film_lock_gamma(float(args.lock_gamma))
                # Pin the stored gamma to X too, so saved PLY / diagnostics reflect the lock.
                # gamma_sigm_split: the lock applies to ALL levels (apply() returns X for any
                # raw gamma), so pin the per-level gammas (cols 22:25) as well.
                if hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0:
                    with torch.no_grad():
                        gaussians._film_params[:, 0] = float(args.lock_gamma)
                        if _fga == 7:
                            gaussians._film_params[:, 22:25] = float(args.lock_gamma)
                print(f"[FILM] gamma LOCKED to {float(args.lock_gamma)} (bypasses --film_act, "
                      f"gamma grad frozen); mlp_input = {float(args.lock_gamma)}*H + beta_act(beta)")

    if args.depth_sort and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
        if args.method == "3D_SH_concat":
            from diff_surfel_3D_sh_concat import set_depth_sort
        elif args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_depth_sort
        else:
            set_depth_sort = _SHRES_SETTER_MOD.set_depth_sort
        set_depth_sort(True)
        print(f"[DEPTH_SORT] Using separated depth sort")

    # Unbiased Depth: set the per-pair depth-difference cutoff to scene_radius / 4
    # (paper formulation). The hardcoded 1.0 default is fine for DTU/T&T-scale
    # objects but clips most pairs in mip-360-scale outdoor scenes (extent ≈ 5).
    # `set_converge_threshold` resolves to the unbiased fork via the sys.modules
    # swap at the top of this file when --unbiased is on.
    if args.unbiased:
        try:
            set_converge_threshold = _SHRES_SETTER_MOD.set_converge_threshold
            _converge_thresh = float(scene.cameras_extent) / 4.0
            set_converge_threshold(_converge_thresh)
            print(f"[UNBIASED] CONVERGE_THRESHOLD = scene.cameras_extent / 4 = {_converge_thresh:.4f}")
        except (ImportError, AttributeError) as _e:
            print(f"[UNBIASED] WARNING: set_converge_threshold not available ({_e}); "
                  f"falling back to compiled default 1.0")

    if args.aa_2dgs > 0.0 and args.method in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
        set_aa_kernel_size = _SHRES_SETTER_MOD.set_aa_kernel_size
        set_aa_kernel_size(args.aa_2dgs)
        print(f"[AA-2DGS] Jacobian mip-filter kernel σ = {args.aa_2dgs} (3D_SH_res standard Gaussian path)")
    elif args.aa_2dgs > 0.0:
        raise RuntimeError(f"--aa_2dgs is only supported for --method 3D_SH_res, got {args.method}")

    # `--l2` / `--l1`: mixed_3d-only per-Gauss loss split (mutually exclusive).
    # Validate at setup so a misuse fails loudly with a clear message rather
    # than silently routing every pixel through L1+SSIM (which would look like
    # "the flag did nothing").
    if getattr(args, 'l2', False) and getattr(args, 'l1', False):
        raise RuntimeError("--l1 and --l2 are mutually exclusive: both reroute the untex Gauss "
                           "leg, just to different loss functions. Pick one.")
    if getattr(args, 'l2', False) or getattr(args, 'l1', False):
        _untex_loss_name = "l1" if getattr(args, 'l1', False) else "l2"
        if args.method not in ("mixed_3d", "mixed_3d_sep"):
            raise RuntimeError(
                f"--{_untex_loss_name} only applies to --method mixed_3d or mixed_3d_sep, "
                f"got {args.method}. It splits the photometric loss with per-Gauss routing in the "
                f"rasterizer backward (textured → L1+SSIM, untextured → {_untex_loss_name.upper()}), "
                f"which needs the per-Gauss _is_textured flag those methods carry.")
        print(f"[LOSS-SPLIT] Photometric loss: TEXTURED Gauss params receive L1+SSIM gradient, "
              f"UNTEXTURED Gauss params receive {_untex_loss_name.upper()} gradient. Routing "
              f"happens per-Gauss inside the rasterizer backward (one forward, one backward "
              f"through the kernel).")

    # Initialize learnable skybox for background modeling
    skybox = None
    bg_hashgrid = None
    background_mode = args.background  # "none", "skybox_dense", "skybox_sparse", "hashgrid", "hashgrid_relu", "hashgrid_sep"
    if background_mode in ["skybox_dense", "skybox_sparse"]:
        skybox = LearnableSkybox(
            resolution_h=args.skybox_res,
            resolution_w=args.skybox_res * 2
        ).cuda()
        skybox.training_setup(lr=args.skybox_lr)
        sparse_str = " (sparse MLP)" if background_mode == "skybox_sparse" else " (dense MLP)"
        print(f"[SKYBOX] Initialized {args.skybox_res}x{args.skybox_res * 2} learnable skybox{sparse_str} (lr={args.skybox_lr})")
        print(f"[SKYBOX] texture.requires_grad={skybox.texture.requires_grad}, device={skybox.texture.device}, shape={skybox.texture.shape}")
        print(f"[SKYBOX] optimizer param groups: {len(skybox.optimizer.param_groups)}, params: {sum(p.numel() for g in skybox.optimizer.param_groups for p in g['params'])}")
    elif background_mode in ["hashgrid", "hashgrid_relu", "hashgrid_sep"]:
        # Get num_levels and level_dim from main method's config to match feature dimensions
        total_levels = cfg_model.encoding.levels
        level_dim = cfg_model.encoding.hashgrid.dim

        # Use CLI args or defaults matching main method
        bg_levels = args.bg_hashgrid_levels if args.bg_hashgrid_levels is not None else total_levels
        bg_dim = args.bg_hashgrid_dim if args.bg_hashgrid_dim is not None else level_dim

        # Validate that output dimensions match
        main_feat_dim = total_levels * level_dim
        bg_feat_dim = bg_levels * bg_dim
        if bg_feat_dim != main_feat_dim:
            print(f"[BG_HASHGRID] WARNING: Feature dimension mismatch!")
            print(f"[BG_HASHGRID]   Main method: {total_levels} levels × {level_dim} dim = {main_feat_dim}D")
            print(f"[BG_HASHGRID]   BG hashgrid: {bg_levels} levels × {bg_dim} dim = {bg_feat_dim}D")
            print(f"[BG_HASHGRID]   Adjusting BG levels to match main method...")
            bg_levels = total_levels
            bg_dim = level_dim

        bg_hashgrid = SphereHashGridBackground(
            num_levels=bg_levels,
            level_dim=bg_dim,
            log2_hashmap_size=args.bg_hashgrid_size,
            base_resolution=16,
            desired_resolution=args.bg_hashgrid_res,
            sphere_radius=args.bg_hashgrid_radius,
        ).cuda()
        bg_hashgrid.training_setup(lr=args.bg_hashgrid_lr)
        bg_start_iter = max(args.bg_hashgrid_start_iter, cfg_model.ingp_stage.switch_iter) if args.bg_hashgrid_start_iter > 0 else cfg_model.ingp_stage.switch_iter
        mode_str = {"hashgrid": "feature composite", "hashgrid_relu": "ReLU features", "hashgrid_sep": "separate RGB decode"}[background_mode]
        print(f"[BG_HASHGRID] Mode: {background_mode} ({mode_str})")
        print(f"[BG_HASHGRID] Feature dim: {bg_hashgrid.output_dim}D (matches main method's {main_feat_dim}D)")
        print(f"[BG_HASHGRID] Will activate at iteration {bg_start_iter} (switch_iter={cfg_model.ingp_stage.switch_iter})")

    opacity_reset_protect = cfg_model.training_cfg.opacity_reset_protect
    if_pixel_densify_enhance = cfg_model.settings.pixel_densify_enhance

    # Freeze beta kernel shape parameter if --freeze_beta is specified
    if args.freeze_beta is not None and args.kernel in ["beta", "beta_scaled"] and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
        # Convert target beta to raw _shape value: beta = sigmoid(_shape) * 5.0
        # So _shape = logit(beta / 5.0) = log(beta / (5.0 - beta))
        target_beta = args.freeze_beta
        if target_beta <= 0 or target_beta >= 5.0:
            raise ValueError(f"--freeze_beta must be in range (0, 5), got {target_beta}")
        raw_shape = math.log(target_beta / (5.0 - target_beta))
        gaussians._shape.data.fill_(raw_shape)
        gaussians._shape.requires_grad_(False)
        # Store frozen value for densification/MCMC to use
        gaussians._frozen_beta_raw = raw_shape
        # Remove from optimizer to prevent cat_tensors_to_optimizer from re-enabling grad
        gaussians.optimizer.param_groups = [g for g in gaussians.optimizer.param_groups if g.get("name") != "shape"]
        actual_beta = torch.sigmoid(torch.tensor(raw_shape)).item() * 5.0
        print(f"[BETA KERNEL] Shape frozen at β={actual_beta:.3f} (raw={raw_shape:.3f}, requires_grad=False)")

    if args.sh_freeze_iter > 0:
        print(f"[SH_FREEZE] SH parameters (f_dc, f_rest) frozen for first {args.sh_freeze_iter} iterations")
    if args.freeze_prim > 0:
        print(f"[FREEZE_PRIM] Per-Gaussian appearance (SH/SB/SG/SV) frozen for first "
              f"{args.freeze_prim} iterations. Geometry, opacity, shape, hashgrid + MLP "
              f"all train normally during the warmup.")

    # === --3rgs: per-camera pose refinement setup ===
    # Build a separate Adam over a per-camera 9D pose delta (decoupled from the
    # Gaussian optimizer, so densification and res_3d_paired splits never touch
    # it). Keyed by image_name → contiguous index, which is scale-agnostic (a
    # downsampled camera carries the same extrinsics).
    pose_opt = None
    pose_optimizer = None
    pose_name_to_idx = {}
    if getattr(args, "pose_refine", False):
        from scene.camera_pose_opt import CameraPoseOpt
        _train_cams = scene.getTrainCameras()
        pose_name_to_idx = {c.image_name: i for i, c in enumerate(_train_cams)}
        pose_opt = CameraPoseOpt(len(pose_name_to_idx)).cuda()
        pose_optimizer = torch.optim.Adam(pose_opt.parameters(),
                                          lr=args.pose_refine_lr,
                                          weight_decay=args.pose_refine_reg, eps=1e-15)
        if args.pose_refine_until < 0:
            args.pose_refine_until = opt.iterations
        print(f"[3RGS] Camera pose refinement ON: {len(pose_name_to_idx)} train cams, "
              f"lr={args.pose_refine_lr:g}, warmup={args.pose_refine_warmup}, "
              f"step until={args.pose_refine_until}, reg={args.pose_refine_reg:g}.")

    def _pose_correction_for(cam):
        """(M_rot, M_t, q_M) for `cam` once warmup has passed, else None (identity)."""
        if pose_opt is None or iteration < args.pose_refine_warmup:
            return None
        idx = pose_name_to_idx.get(cam.image_name)
        if idx is None:
            return None
        return pose_opt.correction(cam.world_view_transform, idx)

    # === --ppisp: photometric (ISP) compensation setup ===
    # NVIDIA PPISP (Deutsch et al. 2026, ../ppisp) — a differentiable ISP layer
    # applied to the RENDERED image before the photometric loss, jointly
    # optimized with the Gaussians. It absorbs the per-image nuisances a
    # handheld/phone capture carries (auto-exposure drift, white-balance drift,
    # lens vignetting) so the radiance field stops explaining them with floaters
    # and view-dependent SH abuse.
    #
    # Chain (identity at init, verified to 1e-5 vs the repo's torch reference):
    #   rgb *= 2^exposure[frame]                        per-frame,  1 scalar
    #   rgb *= vignetting_falloff(uv)                   per-camera, 3x5 (radial poly + center)
    #   rgb  = chromaticity_homography(rgb)             per-frame,  8 latents (intensity-preserving)
    #   rgb  = crf(clamp(rgb, 0, 1))                    per-camera, 3x4 (toe/shoulder/gamma)
    #
    # Mapping onto our data: one physical lens per scene ⇒ num_cameras=1
    # (vignetting + CRF are scene-global), num_frames = #train images, keyed by
    # image_name → contiguous index (scale-agnostic, same idiom as --3rgs).
    #
    # Two deliberate deviations from the library defaults, both about what the
    # BAKED/exported scene should look like:
    #   * CRF frozen at identity unless --ppisp_crf. Our captures come from a
    #     single ISP whose tone curve is already in the GT and which we WANT
    #     reproduced by the splats; letting it train would leave the splats in
    #     pre-CRF space and they'd render wrong in the viewer (which has no CRF).
    #   * controller off unless --ppisp_controller. It's only needed to predict
    #     per-frame corrections for held-out views; with it off, novel views get
    #     zero exposure/color offset = the canonical appearance, which is both
    #     the fair eval and exactly what we want to bake.
    ppisp = None
    ppisp_optimizers = []
    ppisp_schedulers = []
    ppisp_name_to_idx = {}
    if getattr(args, "ppisp", False):
        from ppisp import PPISP, PPISPConfig
        _train_cams = scene.getTrainCameras()
        ppisp_name_to_idx = {c.image_name: i for i, c in enumerate(_train_cams)}
        _ppisp_cfg = PPISPConfig(
            use_controller=bool(args.ppisp_controller),
            controller_distillation=bool(args.ppisp_controller),
            ppisp_lr=args.ppisp_lr,
            scheduler_base_lr=args.ppisp_lr,
            scheduler_decay_max_steps=opt.iterations,
        )
        ppisp = PPISP(num_cameras=1, num_frames=len(ppisp_name_to_idx),
                      config=_ppisp_cfg)
        if not args.ppisp_crf:
            ppisp.crf_params.requires_grad_(False)   # stays at its identity init

        # --finetune_from: restore the trained ISP. This is NOT optional bookkeeping
        # — the reloaded Gaussians are the CANONICAL scene (they satisfy
        # PPISP(render) ≈ GT, not render ≈ GT). Resuming with an identity ISP makes
        # the loss compare the canonical render against raw GT, and the Gaussians
        # re-absorb the vignetting and per-frame exposure the ISP had explained away
        # — fast, because those are large low-frequency errors.
        if getattr(args, "_finetune_ppisp", None) is not None:
            _pck = torch.load(args._finetune_ppisp, weights_only=False)
            # The mode flags change the chain itself; a mismatch is a discontinuity.
            if bool(_pck.get("crf_trained", False)) != bool(args.ppisp_crf):
                raise RuntimeError(
                    f"--finetune_from: source run had ppisp_crf={_pck.get('crf_trained')} "
                    f"but this run has --ppisp_crf={bool(args.ppisp_crf)}. The CRF stage "
                    f"differs, so the reloaded Gaussians would be in the wrong space.")
            if bool(_pck.get("no_camera", False)) != bool(args.ppisp_no_camera):
                raise RuntimeError(
                    f"--finetune_from: source run had ppisp_no_camera={_pck.get('no_camera')} "
                    f"but this run has --ppisp_no_camera={bool(args.ppisp_no_camera)}. The "
                    f"per-camera stages differ, so the reloaded Gaussians would be in the "
                    f"wrong space.")
            _src = PPISP.from_state_dict(_pck["state_dict"], _ppisp_cfg)
            _src_map = _pck["name_to_idx"]
            # Re-key by image_name, NOT by index: a different --eval split, -i
            # resolution or image subset shifts the frame ordering. Frames the source
            # run never saw keep their identity init (= canonical), which is correct.
            _hit = 0
            with torch.no_grad():
                ppisp.vignetting_params.copy_(_src.vignetting_params)   # per-camera, order-free
                ppisp.crf_params.copy_(_src.crf_params)
                for _name, _new_i in ppisp_name_to_idx.items():
                    _old_i = _src_map.get(_name)
                    if _old_i is None:
                        continue
                    ppisp.exposure_params[_new_i] = _src.exposure_params[_old_i]
                    ppisp.color_params[_new_i] = _src.color_params[_old_i]
                    _hit += 1
                if args.ppisp_controller and len(ppisp.controllers) and len(_src.controllers):
                    ppisp.controllers.load_state_dict(_src.controllers.state_dict())
            _miss = len(ppisp_name_to_idx) - _hit
            print(f"[FINETUNE] Restored ISP: {_hit}/{len(ppisp_name_to_idx)} frames matched by name"
                  + (f", {_miss} new frames start at identity" if _miss else "")
                  + f" | exposure {ppisp.exposure_params.min().item():+.3f}"
                  f"..{ppisp.exposure_params.max().item():+.3f} stops carried over.")
            if _hit == 0:
                print("[FINETUNE] WARNING: ZERO frames matched by image_name. The ISP is "
                      "effectively at identity for every frame — check that both runs use "
                      "the same dataset.")

        ppisp_optimizers = ppisp.create_optimizers()
        ppisp_schedulers = ppisp.create_schedulers(ppisp_optimizers, opt.iterations)
        print(f"[PPISP] Photometric compensation ON: {len(ppisp_name_to_idx)} train frames, "
              f"1 camera, lr={args.ppisp_lr:g}, "
              f"exposure+color{'' if args.ppisp_no_camera else '+vignetting'}"
              f"{'+CRF(trained)' if (args.ppisp_crf and not args.ppisp_no_camera) else ''}, "
              f"controller={'on' if args.ppisp_controller else 'off'}, "
              f"camera_path={'off' if args.ppisp_no_camera else 'on'}.")
        if not args.ppisp_no_camera:
            print(f"[PPISP] Camera path active ⇒ the kernel clamps the render to [0,1] "
                  f"before the CRF, which ZEROES the gradient of any pixel >1. "
                  f"Compensating with an over-range penalty w={args.ppisp_overflow_w:g} "
                  f"(--ppisp_overflow_w 0 to disable, --ppisp_no_camera to drop the clamp entirely).")

    def _ppisp_frame_idx(cam):
        """Train-frame index for `cam`, or -1 for a held-out / unknown camera."""
        if ppisp is None:
            return -1
        return ppisp_name_to_idx.get(cam.image_name, -1)

    def _ppisp_apply(img, cam, frame_idx=None):
        """Apply the ISP to a [3,H,W] render. Returns it unchanged when --ppisp is off.

        camera_idx=None disables the per-camera stages (vignetting + CRF) AND the
        [0,1] clamp that rides with them; camera_idx=0 enables the full chain.
        """
        if ppisp is None:
            return img
        fi = _ppisp_frame_idx(cam) if frame_idx is None else frame_idx
        return ppisp(
            img.permute(1, 2, 0).contiguous(),
            camera_idx=None if args.ppisp_no_camera else 0,
            frame_idx=fi,
        ).permute(2, 0, 1)

    # === --deform: per-surfel time-dependent deformation setup ===
    # Per-surfel latent lives on the GaussianModel (rides densification); the MLP
    # + its Adam live here. Per-frame scalar time t∈[0,1] from sorted image names.
    deform_model = None
    deform_optimizer = None
    deform_raster_mod = None
    deform_name_to_time = {}
    if getattr(args, "deform", False):
        from scene.deform_model import DeformModel
        # Route the 3D_SH_res color path through diff_surfel_deform (a fork that
        # samples the hashgrid at CANONICAL xyz while geometry/SV use the deformed
        # means). Swap the renderer's bound module AND sys.modules so the rasterizer,
        # set_mlp_weights, set_activation_bias and set_residual_mode all target the
        # fork's device globals consistently (the codebase's module-swap idiom).
        import sys as _sys, importlib as _il, gaussian_renderer as _gr
        import diff_surfel_deform as _deform_mod
        _gr._sh_res_rasterizer = _deform_mod
        _sys.modules['diff_surfel_3D_sh_res'] = _deform_mod
        deform_raster_mod = _deform_mod
        print("[DEFORM] Routed 3D_SH_res path → diff_surfel_deform (canonical-space hash).")
        _names_sorted = sorted({c.image_name for c in scene.getTrainCameras()})
        _T = len(_names_sorted)
        deform_name_to_time = {nm: (i / (_T - 1) if _T > 1 else 0.0)
                               for i, nm in enumerate(_names_sorted)}
        gaussians.init_deform_latent(args.deform_dim, args.deform_latent_lr)
        # Add the per-surfel latent to the Gaussian optimizer (preserve existing
        # state). training_setup() re-adds it on later rebuilds (res_3d_paired split).
        if "deform_latent" not in [g.get("name") for g in gaussians.optimizer.param_groups]:
            gaussians.optimizer.add_param_group(
                {'params': [gaussians._deform_latent],
                 'lr': args.deform_latent_lr, 'name': 'deform_latent'})
        deform_model = DeformModel(args.deform_dim, width=args.deform_width,
                                   depth=args.deform_depth,
                                   num_time_freqs=args.deform_time_freqs).cuda()
        deform_optimizer = torch.optim.Adam(deform_model.parameters(),
                                            lr=args.deform_mlp_lr, eps=1e-15)
        print(f"[DEFORM] Per-surfel deformation ON: {_T} frames, latent_dim={args.deform_dim}, "
              f"MLP {args.deform_width}x{args.deform_depth}, latent_lr={args.deform_latent_lr:g}, "
              f"mlp_lr={args.deform_mlp_lr:g}, warmup={args.deform_warmup}, reg={args.deform_reg:g}.")

    def _deform_for(cam):
        """(d_xyz, d_rot) for `cam` once warmup has passed, else None (identity).
        Heals a latent/Gauss count desync (e.g. an unwired reinit) by zero-init."""
        if deform_model is None or iteration < args.deform_warmup:
            return None
        N = gaussians.get_xyz.shape[0]
        if gaussians._deform_latent.shape[0] != N:
            tqdm.write(f"[DEFORM] latent/Gauss count desync "
                       f"({gaussians._deform_latent.shape[0]} vs {N}) — reinitializing latent to identity.")
            gaussians.init_deform_latent(args.deform_dim, args.deform_latent_lr)
            for g in gaussians.optimizer.param_groups:
                if g.get("name") == "deform_latent":
                    g['params'][0] = gaussians._deform_latent
            return None
        t = deform_name_to_time.get(cam.image_name, 0.0)
        d_xyz, d_rot = deform_model(gaussians._deform_latent, t)
        # Position-only v1: rotation un-deformed so the canonical hash (canonical
        # center + deformed tangents) is exact (tangents unchanged). Return None for
        # the rotation slot → render() skips the rotation deform.
        return (d_xyz, None)

    # --wsr distill init: one SORTED pass over all training views through the
    # WSR clone (its record_transmittance accumulates Σα in cover_pixels and
    # Σ(α·T) in trans_avg — semantics differ from the base module). occ_init =
    # Σ(α·T)/Σα = visibility-weighted mean transmittance; with occ ≈ T the WSR
    # composite coincides with sorted blending (docs/WSR_DISTILL.md §1), so the
    # finetune starts at (nearly) the sorted solution.
    if getattr(args, 'wsr', False) and not getattr(args, 'wsr_no_distill_init', False) \
            and gaussians._wsr_occ.numel() > 0:
        _N_wsr = gaussians.get_xyz.shape[0]
        _wsr_num = torch.zeros(_N_wsr, device="cuda")
        _wsr_den = torch.zeros(_N_wsr, device="cuda")
        # The training loop normally calls set_active_levels per iteration; this
        # runs before the loop, so arm the levels here (c2f is disabled for
        # proberes — the value just needs to be past all schedules).
        ingp_model.set_active_levels(max(first_iter, 1))
        ingp_model.wsr_sorted = True
        with torch.no_grad():
            _cams = scene.getTrainCameras()
            for _cam in tqdm(_cams, desc="[WSR] distill dump (sorted)"):
                _pk = render(_cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
                             iteration=first_iter, cfg=cfg_model, record_transmittance=True,
                             is_training=False, aabb_mode=args.aabb, lowpass=args.lowpass,
                             pixel_center=args.pixel_center, antialiasing=args.antialiasing,
                             max_intersections_per_pixel=args.max_intersections_per_pixel)
                _wsr_num += _pk['transmittance_avg'].view(-1)
                _wsr_den += _pk['cover_pixels'].view(-1)
        ingp_model.wsr_sorted = False
        _occ0 = (_wsr_num / _wsr_den.clamp_min(1e-6)).clamp(1e-3, 1.0 - 1e-3)
        _unseen = _wsr_den < 1e-6
        _occ0[_unseen] = 0.5  # never rendered → neutral occ, let opt decide
        gaussians._wsr_occ.data = torch.log(_occ0 / (1.0 - _occ0)).view(-1, 1)
        print(f"[WSR] occ distill-init over {len(_cams)} views: "
              f"mean {_occ0.mean():.4f}  p10 {_occ0.quantile(0.1):.4f}  "
              f"p90 {_occ0.quantile(0.9):.4f}  unseen {_unseen.sum().item()}")

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # `--start_resolution` progressive curriculum: swap from the cheap
        # start-res / GPU phase to the expensive final-res / CPU phase at the
        # iter the periodic hash freeze takes over (= freeze_hash_iter). Hash
        # backward then fires 1-in-`--freeze_hash_period` iters, so per-iter
        # cost is dominated by the cheap-iter side again — affordable to pay
        # the 4K rasterizer cost.
        if (_progressive_res_active
                and iteration == args.freeze_hash_iter
                and _final_resolution is not None):
            _msg = reload_cameras_at_resolution(
                scene, args, _final_resolution, _final_data_device)
            print(_msg)
            # Rebuild viewpoint_stack — old references are invalidated.
            try:
                viewpoint_stack = None
            except NameError:
                pass
            _progressive_res_active = False  # one-shot

        # LR schedule: reset to "iteration 5000" after GSPA Phase 1 pruning.
        # Under --minispa, Phase 1 is skipped (silhouette reinit serves that role)
        # and the reinit itself calls reset_xyz_lr_schedule — don't double-shift.
        if args.gspa and iteration >= args.gspa_simp_iter and not args.minispa:
            gaussians.update_learning_rate(iteration - args.gspa_simp_iter + 5000)
        else:
            gaussians.update_learning_rate(iteration)

        # --wsr geometry freeze: must run AFTER update_learning_rate, which
        # rewrites the xyz LR from its scheduler every step.
        if getattr(args, 'wsr', False) and not getattr(args, 'wsr_unfreeze_geom', False):
            for _g in gaussians.optimizer.param_groups:
                if _g["name"] in ("xyz", "scaling", "rotation"):
                    _g['lr'] = 0.0

        if args.nexelparam and ingp_model is not None and hasattr(ingp_model, 'update_nexel_lr'):
            ingp_model.update_nexel_lr(iteration)

        # `--trunc`: per-iteration exit_T ramp (1e-4 → --trunc_exit_T over
        # [--trunc_ramp_start, --trunc_ramp_end]). Linear lerp; before the ramp
        # the threshold stays at 1e-4 (byte-identical to base), after it stays
        # pinned at the target. One 4-byte device-global write per iteration.
        # `--gap_noise`: activate the macro-gap wall once past --gap_noise_after.
        # One 16-byte device-global write per iteration (mirrors the exit_T ramp).
        if getattr(args, 'gap_noise', False) and args.method == "3D_SH_res":
            _gap_on = 1 if iteration >= int(args.gap_noise_after) else 0
            _SHRES_SETTER_MOD.set_gap_trunc(_gap_on,
                                            thresh=args.gap_noise_thresh,
                                            margin=args.gap_noise_margin,
                                            void_mass=args.gap_noise_void_mass,
                                            arm_T=args.gap_noise_arm_T,
                                            T_lo=args.gap_noise_T_lo,
                                            sat_T=args.gap_noise_sat_T)
            if iteration == int(args.gap_noise_after):
                print(f"[GAP_NOISE iter={iteration}] macro-gap truncation ACTIVE "
                      f"(thresh={args.gap_noise_thresh}, margin={args.gap_noise_margin})")

        if getattr(args, 'trunc', False) and args.method == "3D_SH_res":
            _t0, _t1 = int(args.trunc_ramp_start), int(args.trunc_ramp_end)
            if iteration <= _t0:
                _exit_T = 1e-4
            elif iteration >= _t1:
                _exit_T = float(args.trunc_exit_T)
            else:
                _frac = (iteration - _t0) / max(1, _t1 - _t0)
                _exit_T = 1e-4 + (float(args.trunc_exit_T) - 1e-4) * _frac
            _SHRES_SETTER_MOD.set_exit_T(_exit_T)
            if iteration % 1000 == 0:
                print(f"[TRUNC iter={iteration}] exit_T = {_exit_T:.4f}")

        # Freeze/unfreeze SH learning rates
        if args.sh_freeze_iter > 0:
            if iteration <= args.sh_freeze_iter:
                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] in ["f_dc", "f_rest"]:
                        if iteration == 1:
                            param_group["_saved_lr"] = param_group["lr"]
                        param_group["lr"] = 0.0
            elif iteration == args.sh_freeze_iter + 1:
                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] in ["f_dc", "f_rest"] and "_saved_lr" in param_group:
                        param_group["lr"] = param_group["_saved_lr"]
                        print(f"\n[SH_UNFREEZE] Unfreezing {param_group['name']} at iter {iteration}, lr={param_group['lr']:.6f}")

        # --freeze_prim: freeze the per-Gaussian directional appearance groups
        # (SH + SB + SG + SV) for the first N iters. Geometry (xyz, scale,
        # rotation), opacity, kernel shape, and the hash/MLP path all keep
        # training normally, so the hashgrid residual has a chance to fit the
        # scene's appearance alone before the primitive color lobes engage.
        # sv_sites is scheduled — unfreeze lets update_learning_rate take over.
        # Also drops the SH activation bias (clamp(SH + sh_bias) term) to 0
        # during the freeze window so SH's color contribution is a true zero,
        # then restores it to the --activation_bias configured value afterward.
        if args.freeze_prim > 0:
            _FP_APPEARANCE = {
                "f_dc", "f_rest",                                 # SH
                "sb_params",                                      # Spherical Beta
                "sg_directions", "sg_sharpness", "sg_rgb",        # Spherical Gaussian
                "sv_sites", "sv_colors",                          # Spherical Voronoi
            }
            _FP_SCHEDULED = {"sv_sites"}
            _FP_HAS_BIAS = args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]
            if iteration <= args.freeze_prim:
                for pg in gaussians.optimizer.param_groups:
                    name = pg.get("name")
                    if name not in _FP_APPEARANCE:
                        continue
                    if iteration == 1 and name not in _FP_SCHEDULED:
                        pg["_saved_lr_prim"] = pg["lr"]
                    pg["lr"] = 0.0
                if iteration == 1 and _FP_HAS_BIAS:
                    # Override the startup-time bias setter: zero SH bias so SH
                    # contribution = ReLU(SH + 0) — effectively nothing while SH
                    # weights are frozen (SH DC typically inits to ~0 anyway).
                    if args.method == "3D_SH_concat":
                        from diff_surfel_3D_sh_concat import set_activation_bias as _sab
                    elif args.method == "3D_SH_32":
                        from diff_surfel_3D_sh_32 import set_activation_bias as _sab
                    else:
                        from diff_surfel_3D_sh_res import set_activation_bias as _sab
                    from gaussian_renderer import set_default_activation_bias as _sdab
                    _cfg_sh, _cfg_res = args.activation_bias
                    _sab(sh_bias=0.0, res_bias=_cfg_res)
                    _sdab(0.0, _cfg_res)
                    tqdm.write(f"[FREEZE_PRIM] SH bias 0.0 during freeze window "
                               f"(configured={_cfg_sh}, restored at iter {args.freeze_prim + 1})")
            elif iteration == args.freeze_prim + 1:
                for pg in gaussians.optimizer.param_groups:
                    name = pg.get("name")
                    if name not in _FP_APPEARANCE or name in _FP_SCHEDULED:
                        continue
                    if "_saved_lr_prim" in pg:
                        pg["lr"] = pg["_saved_lr_prim"]
                if _FP_HAS_BIAS:
                    if args.method == "3D_SH_concat":
                        from diff_surfel_3D_sh_concat import set_activation_bias as _sab
                    elif args.method == "3D_SH_32":
                        from diff_surfel_3D_sh_32 import set_activation_bias as _sab
                    else:
                        from diff_surfel_3D_sh_res import set_activation_bias as _sab
                    from gaussian_renderer import set_default_activation_bias as _sdab
                    _cfg_sh, _cfg_res = args.activation_bias
                    _sab(sh_bias=_cfg_sh, res_bias=_cfg_res)
                    _sdab(_cfg_sh, _cfg_res)
                    print(f"[FREEZE_PRIM] SH bias restored to {_cfg_sh}")
                print(f"\n[FREEZE_PRIM] Unfrozen SH/SB/SG/SV appearance groups at iter {iteration}")

        opacity_reset_interval = opt.opacity_reset_interval
        densification_interval = opt.densification_interval
        if ingp_model is None:
            ingp = None
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif args.cold:
            # Cold start: enable ingp from the start, skip warmup phase
            # Use warmup phase densification settings (same as during initialize phase)
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif iteration <= cfg_model.ingp_stage.initialize:
            ingp = None
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif iteration <= cfg_model.ingp_stage.switch_iter:
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        else:
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.ingp_densify_threshold
            densification_interval = cfg_model.training_cfg.ingp_densification_interval
            opacity_reset_interval = cfg_model.training_cfg.ingp_opacity_reset_interval
            appearance_update_threshold = 0.0
        
        optim_gaussian = True
        optim_ngp = False
        active_levels = None

        if ingp is not None:
            # 3D_SH_res warmup: disable hash/MLP for first N iterations
            if args.res_warmup > 0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat", "proberes"] and iteration < args.res_warmup:
                ingp.hashgrid_disabled = True
                optim_ngp = False
                optim_gaussian = True
            else:
                if args.res_warmup > 0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat", "proberes"] and iteration == args.res_warmup:
                    ingp.hashgrid_disabled = False
                    tqdm.write(f"[3D_SH_RES] Enabling hash/MLP residual at iteration {iteration}")

                active_levels = ingp.set_active_levels(iteration)
                optim_ngp = True
                optim_gaussian = ingp.optim_gaussian
                # --probe_tex_only: pin ALL Gaussian params — only the INGP
                # (texture field + pixels [+ head unless frozen]) trains. Safe
                # to pair with aggressive probe_field_lr_scale: no geometry
                # feedback loop exists.
                if getattr(args, 'probe_tex_only', False) and getattr(ingp, 'is_proberes_mode', False):
                    optim_gaussian = False

            # Periodic hashgrid freeze. After `freeze_hash_iter`, train hash on 1 iter
            # out of every `freeze_hash_period`. CUDA backward runs the hash query
            # forward-only on skip iters (no hash/xyz gradient propagation).
            #   3D_SH_res → freezes hash + MLP together (MLP is part of the residual
            #               path, no shared decoder concern). Skip optimizer.step too.
            #   cat       → MLP is a SHARED DECODER for both per-Gaussian and hash
            #               features. Freezing it would starve per-Gauss features of
            #               useful gradients. So skip only the hash gradient (CUDA-side)
            #               and keep the INGP optimizer.step running so the MLP updates.
            #               Hash table sees ~zero gradient, Adam moments decay slightly,
            #               weights effectively don't move.
            _freeze_methods = ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_filmres", "cat", "film", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep")
            if (args.freeze_hash_iter > 0 and iteration >= args.freeze_hash_iter
                    and args.method in _freeze_methods):
                period = max(1, int(args.freeze_hash_period))
                should_train = (iteration % period) == 0
                desired_skip = not should_train
                # Flip the CUDA flag only on state transitions (saves a 1-thread
                # kernel launch every iter).
                current_skip = getattr(ingp, '_skip_mlp_grad', None)
                if current_skip != desired_skip:
                    try:
                        if args.method in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_filmres", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
                            set_skip_mlp_grad = _SHRES_SETTER_MOD.set_skip_mlp_grad
                        elif args.method == "film":
                            from diff_surfel_film import set_skip_mlp_grad
                        else:  # cat
                            from diff_surfel_rasterization import set_skip_mlp_grad
                        set_skip_mlp_grad(desired_skip)
                    except (ImportError, AttributeError):
                        tqdm.write("[FREEZE_HASH] WARN: set_skip_mlp_grad not available; rebuild?")
                    ingp._skip_mlp_grad = desired_skip
                    if current_skip is None:
                        if args.method == "cat":
                            tqdm.write(f"[FREEZE_HASH] Periodic hash freeze active at iter "
                                       f"{iteration} (cat: hash 1 of {period} iters, MLP "
                                       f"trains every iter)")
                        else:
                            tqdm.write(f"[FREEZE_HASH] Periodic freeze active at iter "
                                       f"{iteration} (train hash+MLP 1 of every {period} iters)")
                # 3D_SH_res: skip whole INGP optimizer on freeze iters (freezes MLP too).
                # cat: keep INGP optimizer stepping so the shared MLP decoder updates;
                # CUDA-side zero gradient already freezes the hash table.
                if args.method in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_filmres", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
                    optim_ngp = should_train
                    # 3D_SH_filmres: the FiLM latent (gaussians-side `film_params` group) modulates
                    # the hash, so freeze it WITH the hash. set_skip_mlp_grad already zeros its grad;
                    # pin LR to 0 on freeze iters so leftover Adam momentum can't drift it either
                    # (restore feature_lr on train iters). The hash/MLP are frozen via optim_ngp.
                    if args.method == "3D_SH_filmres":
                        for _pg in gaussians.optimizer.param_groups:
                            if _pg.get("name") == "film_params":
                                _pg["lr"] = opt.feature_lr if should_train else 0.0
                # else: leave optim_ngp at its default (True) for cat.

            if iteration % surfel_cfg.update_interval == 0 and optim_gaussian \
                and beta < surfel_cfg.tg_beta and active_levels == cfg_model.encoding.levels:
                
                update_times = (surfel_cfg.update_interations / surfel_cfg.update_interval)
                gaussians.base_opacity += surfel_cfg.tg_base_alpha / update_times
                beta += surfel_cfg.tg_beta / update_times

        # Unfreeze SH after depth reinit grace period
        if mini_sh_unfreeze_iter > 0 and iteration == mini_sh_unfreeze_iter:
            for pg in gaussians.optimizer.param_groups:
                if pg["name"] == "f_dc":
                    pg["lr"] = opt.feature_lr
                elif pg["name"] == "f_rest":
                    pg["lr"] = opt.feature_lr / 20.0
            mini_sh_unfreeze_iter = 0
            tqdm.write(f"[MINI] Unfroze SH at iter {iteration}")

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # When GSPA is active, delay SH increases until after Phase 1 (importance pruning)
        # When sh_freeze_iter is set, delay SH increases until after unfreeze
        # When --mini is active, lock SH at degree 0 until simp1 (matches MSv2 paper)
        sh_base_iter = max(args.sh_freeze_iter, args.freeze_prim, args.gspa_simp_iter if args.gspa else 0)
        if iteration > sh_base_iter and (iteration - sh_base_iter) % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # Mini warmup: alternate between 0.5x and 1.0x resolution until simp1
        if args.mini and args.mini_warmup and iteration < args.mini_simp_iter1:
            warmup_scale = [1.0, 0.5][iteration % 2]
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras(scale=warmup_scale).copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        record_transmittance = if_pixel_densify_enhance & (iteration >= opt.pixel_densify_from_iter) & (iteration < opt.densify_until_iter)
        
        # Alternate background color every 10 iterations for diffuse_ngp/diffuse_offset
        # This prevents Gaussians from hiding things with RGB instead of opacity
        if use_random_bg:
            # Random background: pass black to renderer, apply random per-pixel bg afterward
            current_bg = black_bg
        elif use_alternating_bg:
            current_bg = white_bg if (iteration // 10) % 2 == 0 else black_bg
        else:
            current_bg = background

        # Compute temperature for adaptive modes (sigmoid sharpening)
        if args.temp_end > args.temp_start:
            if iteration < args.temp_anneal_start:
                temperature = args.temp_start
            elif iteration >= args.temp_anneal_end:
                temperature = args.temp_end
            else:
                progress = (iteration - args.temp_anneal_start) / (args.temp_anneal_end - args.temp_anneal_start)
                temperature = args.temp_start + progress * (args.temp_end - args.temp_start)
        else:
            temperature = 1.0

        # Only use skybox/bg_hashgrid after their respective start iterations
        active_skybox = skybox if (skybox is not None and iteration >= cfg_model.ingp_stage.switch_iter) else None
        # BG hashgrid can start later than main hashgrid to let FG train first
        # bg_start_iter is at least switch_iter (need hashgrid features to decode)
        bg_start_iter = max(args.bg_hashgrid_start_iter, cfg_model.ingp_stage.switch_iter)
        active_bg_hashgrid = bg_hashgrid if (bg_hashgrid is not None and iteration >= bg_start_iter) else None

        # Debug: verify SH is zero on first iteration for 3D_SH_res
        if iteration == first_iter + 1 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
            dc_norm = gaussians._features_dc.data.abs().max().item()
            # `_features_rest` is shape [N, 0, 3] under --feature beta (we drop
            # higher-order SH there). max() on an empty tensor needs special-casing.
            rest_norm = (gaussians._features_rest.data.abs().max().item()
                         if gaussians._features_rest.numel() > 0 else 0.0)
            print(f"[DEBUG] First iter SH check: DC max={dc_norm:.6f}, REST max={rest_norm:.6f}, "
                  f"active_sh_degree={gaussians.active_sh_degree}")

        # Timing: forward pass
        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_fwd_start = time.time()

        # `--method res_3d` / `--method res_switch` STAGE 1: mode-0 → mode-2
        # flip + post-blend per-pixel LRU. Fires at `--res_switch_iter`
        # (default 10000) for BOTH methods. Pre-stage-1 the kernel applies
        # the outer ReLU per-Gauss in CUDA (mode 0); post-stage-1 it leaves
        # the per-Gauss sum signed (mode 2) and the renderer's deferred
        # per-pixel LRU clamps the blended pixel. With --lru > 0 the two
        # phases share a leaky-gradient slope at the clamp so the transition
        # is a smooth knee rather than a discontinuous jump.
        #
        # For `--method res_3d` this is STAGE 1 of a two-stage curriculum;
        # STAGE 2 (the tex/untex split) fires later at `--res_3d_iter`. If
        # both flags share the default 10000, the two stages fire on the
        # same iteration — current behavior preserved. Set res_switch_iter
        # earlier than res_3d_iter (e.g. 10000 / 15000) to space them out.
        # --probe_distill_dir forces the SAME mode-0 -> mode-2 flip at iter 1: the
        # distillation target is the teacher's SIGNED blended residual, which is only
        # linear in the atlas texels once the per-Gauss outer ReLU is gone.
        _probe_distill_flip = (args.method == "proberes"
                               and getattr(args, 'probe_distill_dir', None)
                               and iteration == 1)
        if _probe_distill_flip or (
                args.method in ("res_switch", "res_3d", "res_3d_paired", "res_3d_double")
                and args.res_switch_iter > 0
                and iteration == args.res_switch_iter):
            print(f"[STAGE1 iter={iteration}] mode 0 → mode 2 (per-Gauss outer "
                  f"ReLU → per-pixel ReLU after blend). "
                  f"LRU α={getattr(args, 'lru', 0.0)} active on the post-blend "
                  f"image. (method={args.method})")
            # 1) CUDA: kernel forward + backward now treat the per-Gauss sum
            #    as signed (no outer ReLU). Setter mirrors propagate to mixed/
            #    mixed_3d if those are loaded (no-op for plain res_switch).
            set_residual_mode = _SHRES_SETTER_MOD.set_residual_mode
            set_residual_mode(2)
            args._residual_mode = 2
            # 2) Python renderer: enable the post-blend per-pixel ReLU/LRU/STE
            #    dispatch by flipping the INGP flag the renderer reads.
            if ingp is not None:
                ingp.is_mixed_deferred_relu_mode = True

        # ======================= --method GEStex schedule =======================
        # GEStex-specific gated events (method is aliased to res_switch, so the
        # mode 0->2 flip above already fired at --ges_phase1_iter). All events run
        # inside torch.no_grad() blocks in the main loop's grad-disabled section is
        # not guaranteed here, so wrap explicitly.
        if getattr(args, 'is_gestex', False):
            with torch.no_grad():
                # All-views frontmost-count occlusion cull (T·o-proxy pruning). A surfel never
                # the max-weight (frontmost) contributor at >= _thr pixels across all training
                # views is occluded/redundant → prune. Reused for the periodic harden cull.
                def _ges_occlusion_prune(_thr):
                    _cams = scene.getTrainCameras()
                    _N = gaussians.get_xyz.shape[0]
                    _cover = torch.zeros(_N, device="cuda", dtype=torch.long)
                    _got = False
                    for _cam in _cams:
                        _pk = render(_cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                     iteration=iteration, cfg=cfg_model, lowpass=args.lowpass, is_training=False)
                        _mci = _pk.get('max_contrib_idx', None)
                        if _mci is not None:
                            _got = True
                            _m = _mci.reshape(-1).long(); _v = (_m >= 0) & (_m < _N)
                            _cover = torch.maximum(_cover, torch.bincount(_m[_v], minlength=_N))
                    if not _got:
                        return 0, _N
                    _pm = (_cover < _thr); _nc = int(_pm.sum().item())
                    if 0 < _nc < _N:
                        gaussians.prune_points(_pm)
                    return _nc, _N

                # --- Harden onset (10k): TS+-style. Opacity stays TRAINABLE (optimized within
                #     the rising floor); opacity RESET is turned OFF (a reset knocking opacity to
                #     ~0.01 fights the floor); densify/prune stay ON (refill + cull); record
                #     β_start for the ceiling anneal. NO aggressive prune here. ---
                if iteration == args.ges_phase1_iter:
                    opt.opacity_reset_interval = 10 ** 9   # reset OFF (fights the rising floor)
                    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                        gaussians._ges_beta_start = float(gaussians.get_shape.max().item())
                    print(f"[GEStex iter={iteration}] HARDEN start ({gaussians.get_xyz.shape[0]} surfels): "
                          f"opacity TRAINABLE within rising floor 0->{args.ges_opac_floor_max:.2f}, reset OFF, "
                          f"β ceiling {getattr(gaussians,'_ges_beta_start',0.0):.2f}->{args.ges_beta_end:.2f}, "
                          f"prune+grow ON over {args.ges_phase1_iter}->{args.ges_joint_iter}.")
                # --- TS+ rising opacity FLOOR + β CEILING over [phase1, joint). Opacity =
                #     O_t + (1-O_t)*get_opacity stays <= 1 (no ∝opacity>1 bloat); β capped at a
                #     falling ceiling → flat-top discs by the joint. Optimizer free within both. ---
                # Opacity floor ramp: starts at --ges_floor_start_iter (default -1 =
                # phase1/10k) and reaches ges_opac_floor_max at joint (20k). Starting
                # earlier (e.g. 5000) gives a LONGER, gentler ramp — less popping shock
                # when the ordering promotion + occlusion culls kick in mid-harden.
                _fstart = int(getattr(args, 'ges_floor_start_iter', -1))
                if _fstart < 0:
                    _fstart = args.ges_phase1_iter
                if _fstart <= iteration < args.ges_joint_iter:
                    _tf = (iteration - _fstart) / max(1, args.ges_joint_iter - _fstart)
                    gaussians.ges_opac_floor = float(args.ges_opac_floor_max) * _tf
                if args.ges_phase1_iter <= iteration < args.ges_joint_iter:
                    _t = (iteration - args.ges_phase1_iter) / max(1, args.ges_joint_iter - args.ges_phase1_iter)
                    # β ceiling: cap get_shape (=sigmoid(_shape)*5) at β_max(t), annealed
                    # β_start -> ges_beta_end. Clamp _shape's UPPER bound; optimizer free below.
                    if args.kernel in ("beta", "beta_scaled") and hasattr(gaussians, '_shape') \
                            and gaussians._shape.numel() > 0 \
                            and not (int(getattr(args, 'ges_lock_beta_iter', -1)) >= 0
                                     and iteration >= int(args.ges_lock_beta_iter)):
                        _bs = float(getattr(gaussians, '_ges_beta_start', 4.0))
                        _bmax = _bs * (1.0 - _t) + float(args.ges_beta_end) * _t
                        _frac = min(max(_bmax / 5.0, 1e-4), 1.0 - 1e-4)
                        _raw_ceil = math.log(_frac / (1.0 - _frac))
                        gaussians._shape.data.clamp_(max=_raw_ceil)
                # --ges_lock_beta_iter: HARD-LOCK all betas to --ges_lock_beta_val from
                # iter N (fully flat discs A/B). Pinned EVERY iter (overrides optimizer
                # steps and the ceiling anneal, which is bypassed above when locked);
                # applies through the joint stage too. -1 = off.
                _lockb_it = int(getattr(args, 'ges_lock_beta_iter', -1))
                if _lockb_it >= 0 and iteration >= _lockb_it \
                        and args.kernel in ("beta", "beta_scaled") \
                        and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                    _bv = min(max(float(args.ges_lock_beta_val) / 5.0, 1e-4), 1.0 - 1e-4)
                    gaussians._shape.data.fill_(math.log(_bv / (1.0 - _bv)))
                    if iteration == _lockb_it:
                        print(f"[GEStex iter={iteration}] β LOCKED to "
                              f"{float(args.ges_lock_beta_val):.3f} (fully-flat A/B; pinned per-iter).")
                # --- Two-phase pruning (TS+): (1) one gentle hard opacity cut early, while the
                #     floor is low so get_opacity is a meaningful junk signal; (2) periodic
                #     occlusion (T·o-proxy) culls thereafter. Densification stays ON to refill. ---
                if iteration == args.ges_phase1_iter + int(args.ges_hard_prune_offset):
                    _om = (gaussians.get_opacity.flatten() < float(args.ges_prune_w_thresh))
                    _no = int(_om.sum().item()); _Nt = gaussians.get_xyz.shape[0]
                    if 0 < _no < _Nt:
                        gaussians.prune_points(_om)
                    print(f"[GEStex iter={iteration}] HARD opacity prune (<{args.ges_prune_w_thresh}): "
                          f"removed {_no}/{_Nt} surfels.")
                if (args.ges_surfel_prune_interval > 0
                        and args.ges_phase1_iter < iteration < args.ges_joint_iter
                        and iteration % args.ges_surfel_prune_interval == 0):
                    _othr = args.ges_occlusion_thresh
                    if 'synthetic' in str(getattr(dataset, 'source_path', '')).lower():
                        _othr = min(_othr, 4)
                    _nc, _Nt = _ges_occlusion_prune(_othr)
                    print(f"[GEStex iter={iteration}] occlusion cull (<{_othr} frontmost px): "
                          f"removed {_nc}/{_Nt} surfels.")
                # --- Mid-harden SHRINK: once the surfels are ~half-hardened, the SV tends to
                #     overshoot and the (now near-opaque) discs sit at too-large a scale. Shrink
                #     surfel scale to --ges_shrink_factor (0.75) at --ges_shrink_iter so the
                #     still-trainable geometry + mask/RGB loss re-fit the right size under full
                #     opacity over the remaining harden iters. Log-space: += log(factor). ---
                if iteration == args.ges_shrink_iter and float(args.ges_shrink_factor) < 1.0:
                    _ln = math.log(max(float(args.ges_shrink_factor), 1e-6))
                    if hasattr(gaussians, '_is_textured') and \
                            gaussians._is_textured.numel() == gaussians._scaling.shape[0]:
                        gaussians._scaling.data[gaussians._is_textured] += _ln
                    else:
                        gaussians._scaling.data += _ln
                    print(f"[GEStex iter={iteration}] SHRINK surfels to "
                          f"{float(args.ges_shrink_factor):.0%} scale (re-fit under full opacity).")
                # --- Bake & splat (20k): ONE pass over all views computes surfel occlusion
                #     coverage + error-map Gaussian-init positions; then prune occluded
                #     surfels, bake atlas, spawn Gaussians, switch to sort-free, and re-enable
                #     standard densify/prune (surfels frozen + opacity-pinned → densify-inert). ---
                if iteration == args.ges_joint_iter and not getattr(ingp, 'is_gestex_joint', False):
                    # --- DIAGNOSTIC: localize the "mangled at 20k" issue. Saves renders of a
                    #     fixed train cam into <model>/ges_debug/. Captured BEFORE any prune/
                    #     densify so densification is ruled out. A = harden (live hash+MLP+SV,
                    #     tile-order). B (below) = first joint sort-free render (atlas). C (below)
                    #     = joint with the atlas ZEROED (SV-only). If A clean & C clean & B
                    #     mangled → the baked atlas VALUES are wrong. If C also mangled → the
                    #     Gaussian composite/spawn is the culprit, not the atlas. ---
                    def _ges_dbg_save(_tag, _zero_atlas=False, _decompose=None):
                        try:
                            import os as _os
                            from torchvision.utils import save_image as _si
                            _dd = os.path.join(scene.model_path, 'ges_debug'); _os.makedirs(_dd, exist_ok=True)
                            _cam = scene.getTrainCameras()[0]
                            _bak = None
                            if _zero_atlas and getattr(gaussians, '_tex_atlas', None) is not None \
                                    and gaussians._tex_atlas.numel() > 0:
                                _bak = gaussians._tex_atlas.data.clone(); gaussians._tex_atlas.data.zero_()
                            with torch.no_grad():
                                _pk = render(_cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                             iteration=iteration, cfg=cfg_model, lowpass=args.lowpass,
                                             is_training=False, decompose_mode=_decompose)
                            _si(_pk['render'].clamp(0, 1), os.path.join(_dd, f'{_tag}.png'))
                            if _bak is not None: gaussians._tex_atlas.data.copy_(_bak)
                            print(f"[GEStex DIAG] saved {_tag}.png (render range "
                                  f"[{_pk['render'].min().item():.3f},{_pk['render'].max().item():.3f}])")
                        except Exception as _e:
                            print(f"[GEStex DIAG] {_tag} failed: {_e}")
                            import traceback as _tb; _tb.print_exc()
                            if _zero_atlas and _bak is not None: gaussians._tex_atlas.data.copy_(_bak)
                    try:
                        from torchvision.utils import save_image as _si0
                        _dd0 = os.path.join(scene.model_path, 'ges_debug'); os.makedirs(_dd0, exist_ok=True)
                        _si0(scene.getTrainCameras()[0].original_image.cuda().clamp(0, 1),
                             os.path.join(_dd0, 'GT.png'))
                    except Exception:
                        pass
                    # BEFORE the 20k transition (harden state, LIVE hash+MLP+SV, tile-order blend):
                    # full + SV-base-only + residual-only. If SV_only is already noise and the
                    # residual is already subtracting here, the pathology is in HARDEN, not the bake.
                    _ges_dbg_save(f'A_before_full_{iteration}')
                    _ges_dbg_save(f'A_before_SVonly_{iteration}', _decompose='sh_only')
                    _ges_dbg_save(f'A_before_residual_{iteration}', _decompose='tex_only')
                    gaussians.ges_opac_floor = float(args.ges_opac_floor_max)   # fully opaque surfels
                    try:
                        from gaussian_renderer import ges_bake_atlas
                        from utils.point_utils import depths_to_points
                        from utils.general_utils import inverse_sigmoid as _isig
                        _cams = scene.getTrainCameras()
                        _thr = args.ges_occlusion_thresh
                        if 'synthetic' in str(getattr(dataset, 'source_path', '')).lower():
                            _thr = min(_thr, 4)
                        Nsurf = gaussians.get_xyz.shape[0]
                        cover = torch.zeros(Nsurf, device="cuda", dtype=torch.long)
                        _n_total = args.ges_gs_add_num if args.ges_gs_add_num > 0 else max(10000, Nsurf // 4)
                        _n_per = max(1, _n_total // max(1, len(_cams)))
                        _spawn, _have_mci = [], False
                        for _cam in _cams:
                            _pkg = render(_cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                          iteration=iteration, cfg=cfg_model, lowpass=args.lowpass, is_training=False)
                            _img = _pkg['render']
                            _mci = _pkg.get('max_contrib_idx', None)
                            _depth = _pkg.get('depth_expected', _pkg.get('surf_depth', None))
                            if _mci is not None:
                                _have_mci = True
                                _m = _mci.reshape(-1).long(); _v = (_m >= 0) & (_m < Nsurf)
                                cover = torch.maximum(cover, torch.bincount(_m[_v], minlength=Nsurf))
                            if _depth is not None:
                                _gt = _cam.original_image.cuda()
                                _err = ((_img - _gt) ** 2).sum(0).reshape(-1)
                                _s = _err.sum()
                                if _s > 0:
                                    _cdf = torch.cumsum(_err / _s, 0)
                                    _idx = torch.searchsorted(_cdf, torch.rand(_n_per, device='cuda')).clamp(max=_err.numel() - 1)
                                    _pts, _, _ = depths_to_points(_cam, _depth.reshape(int(_cam.image_height), int(_cam.image_width)))
                                    _dv = _depth.reshape(-1)[_idx] > 1e-6
                                    _spawn.append(_pts[_idx][_dv])
                        if _have_mci:
                            _pm = (cover < _thr); _nc = int(_pm.sum().item())
                            if 0 < _nc < Nsurf:
                                gaussians.prune_points(_pm)
                            print(f"[GEStex iter={iteration}] OCCLUSION CULL: pruned {_nc}/{Nsurf} surfels (<{_thr} px).")
                        seeds = torch.cat(_spawn, 0) if len(_spawn) > 0 else torch.empty(0, 3, device='cuda')
                        print(f"[GEStex iter={iteration}] error-map init: {seeds.shape[0]} Gaussian seeds from {len(_cams)} views.")
                        # --ges_no_bake: keep the hashgrid+MLP as the texture source through
                        # the joint stage (route via the cascade, atlas OFF) instead of baking
                        # into a static atlas. Isolates the surfel/texture/Gaussian interplay
                        # from the bake. INGP keeps training below.
                        _no_bake = bool(getattr(args, 'ges_no_bake', False))
                        if _no_bake:
                            atlas = None
                            print(f"[GEStex iter={iteration}] --ges_no_bake: SKIP atlas bake; "
                                  f"textured surfels keep LIVE hash+MLP via the cascade.")
                        else:
                            atlas = ges_bake_atlas(ingp, gaussians, int(args.ges_atlas_res))
                        # pin surfel get_opacity high so standard opacity-prune never removes
                        # them (render opacity = get_opacity*surfel_opac stays opaque; surfel
                        # opacity grad masked to 0 each step → frozen at 0.99).
                        with torch.no_grad():
                            gaussians._opacity.data[:] = _isig(torch.full_like(gaussians._opacity.data, 0.99))
                        gaussians.ges_enter_joint_stage(atlas, seeds, opt, bake_atlas=not _no_bake)
                        ingp.is_gestex_joint = True
                        # Stop TEXTURED surfel growth: from here densify_and_{clone,split} only
                        # grow the untextured 3D Gaussians (surfels are frozen + baked). The
                        # standard clone/split is otherwise screenspace-grad-driven and would
                        # keep cloning frozen surfels.
                        gaussians.ges_freeze_textured_densify = True
                        # re-enable STANDARD densify/prune for 20k+ Gaussians; disable opacity
                        # reset (would knock surfels off their pinned 0.99).
                        opt.densify_until_iter = int(opt.iterations)
                        if hasattr(args, 'fastgs_densify_until'):
                            args.fastgs_densify_until = int(opt.iterations)
                        opt.opacity_reset_interval = 10 ** 9
                        # Use the sort-free 2-pass renderer at full hardening when the
                        # joint kernels are built (falls back to the cascade otherwise).
                        # --ges_no_bake forces the cascade (sort-free is atlas-only; with no
                        # atlas the cascade renders textured surfels via live hash+MLP).
                        if _no_bake:
                            ingp.is_gestex_sortfree = False
                            print("[GEStex] --ges_no_bake → joint via cascade (diff_surfel_gestex), "
                                  "atlas OFF → LIVE hash+MLP textures.")
                        else:
                            try:
                                import diff_surfel_gestex_joint_s, diff_surfel_gestex_joint_g  # noqa
                                ingp.is_gestex_sortfree = True
                                print("[GEStex] sort-free 2-pass rendering ENABLED (joint_s + joint_g).")
                            except ImportError:
                                ingp.is_gestex_sortfree = False
                                print("[GEStex] joint kernels not built → using res_3d_paired cascade fallback.")
                        # CRITICAL: the gestex kernel has its OWN device-global MLP weights /
                        # activation-bias / residual-mode / lru-slope. GEStex is aliased to
                        # res_switch, so no setter mirror was installed → gestex's d_mlp_W1..3
                        # are NULL and its backward null-reads them (illegal access). Install
                        # the mirror now (like the res_3d split does for mixed_3d) and re-fire
                        # the setters so gestex gets the current MLP weights + mode 2 + bias + lru.
                        try:
                            import diff_surfel_3D_sh_res as _ds_orig_g
                            import diff_surfel_gestex as _ds_gestex
                            _MIRROR_G = (
                                'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                                'set_overdraw_lambda', 'set_weight_reg_lambda',
                                'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                                'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                                'set_depth_sort', 'set_ste_relu', 'set_lru_slope',
                            )
                            for _nm in _MIRROR_G:
                                if not hasattr(_ds_orig_g, _nm) or not hasattr(_ds_gestex, _nm):
                                    continue
                                _of_g = getattr(_ds_orig_g, _nm); _mf_g = getattr(_ds_gestex, _nm)
                                def _mk_g(of, mf):
                                    def _w(*a, **k):
                                        of(*a, **k); mf(*a, **k)
                                    return _w
                                setattr(_ds_orig_g, _nm, _mk_g(_of_g, _mf_g))
                            # Re-fire current state onto gestex (weights + mode/bias/lru).
                            _mw = ingp.get_fused_mlp_weights()
                            if _mw is not None:
                                _ds_gestex.set_mlp_weights(_mw[0].contiguous(), _mw[1].contiguous(), _mw[2].contiguous())
                            # Mirror the CURRENT residual mode (2 after the res_switch flip;
                            # 0 when --ges_local_lru kept 3D_SH_res semantics all the way).
                            _ds_gestex.set_residual_mode(int(getattr(args, '_residual_mode', 2)))
                            _ab_g = getattr(args, 'activation_bias', [0.5, 0.0])
                            _ds_gestex.set_activation_bias(float(_ab_g[0]), float(_ab_g[1]))
                            _ds_gestex.set_lru_slope(float(getattr(args, 'lru', 0.0)))
                            print("[GEStex] installed setter mirror diff_surfel_3D_sh_res → "
                                  "diff_surfel_gestex + re-fired MLP weights / mode 2 / bias / lru.")
                        except Exception as _me:
                            print(f"[GEStex] WARNING: setter mirror to diff_surfel_gestex failed: {_me}")
                        # NOTE: we intentionally do NOT set ingp.hashgrid_disabled = True.
                        # The kernel's collab-GEMM backward mishandles active_hashgrid_levels==0
                        # (illegal access). Instead we keep the hashgrid active: the forward
                        # OVERWRITES the MLP residual with the baked atlas lookup, and the
                        # backward scatters dL/dresidual into the atlas. The hash/MLP still
                        # compute gradients but they're discarded (INGP optimizer LR = 0 below).
                        # stop training the hash/MLP (INGP optimizer) — atlas is the leaf now.
                        # --ges_no_bake: KEEP the INGP training (hash+MLP is still the texture).
                        if (not _no_bake) and ingp is not None and getattr(ingp, 'optimizer', None) is not None:
                            for _g in ingp.optimizer.param_groups:
                                _g['lr'] = 0.0
                        if _no_bake:
                            print(f"[GEStex iter={iteration}] SPLAT complete (NO BAKE) → cascade render, "
                                  f"hash+MLP textures still training.")
                        else:
                            print(f"[GEStex iter={iteration}] BAKE & SPLAT complete → sort-free 2-pass render.")
                        # DIAGNOSTIC B/C: first joint render (sort-free), BEFORE any post-20k
                        # densify/prune. B = with baked atlas; C = atlas zeroed (SV-only). Also
                        # log atlas value stats to spot a broken bake (huge/NaN residuals).
                        try:
                            _at = gaussians._tex_atlas[gaussians._is_textured]
                            print(f"[GEStex DIAG] baked atlas stats: shape={tuple(gaussians._tex_atlas.shape)} "
                                  f"surfel_rows={_at.shape[0]} mean={_at.mean().item():.4f} std={_at.std().item():.4f} "
                                  f"min={_at.min().item():.4f} max={_at.max().item():.4f} "
                                  f"abs>1={( _at.abs()>1).float().mean().item()*100:.1f}%")
                        except Exception:
                            pass
                        # AFTER the transition (joint stage, BAKED atlas): full + SV-base-only
                        # + atlas-residual-only. Compare against the A_before_* trio: if
                        # A_before ≈ B_after the bake faithfully reproduced the harden state.
                        _ges_dbg_save(f'B_after_full_{iteration}', _zero_atlas=False)
                        _ges_dbg_save(f'B_after_SVonly_{iteration}', _decompose='sh_only')
                        _ges_dbg_save(f'B_after_residual_{iteration}', _decompose='tex_only')
                        _ges_dbg_save(f'C_after_noatlas_{iteration}', _zero_atlas=True)
                    except Exception as _e:
                        print(f"[GEStex] WARNING: joint transition failed ({_e}); "
                              f"continuing in harden-phase rendering.")
                        import traceback; traceback.print_exc()

                # --- Joint stage: periodic prune of low-opacity untextured 3D Gaussians. ---
                # Surfels (textured, frozen) are kept; only the spawned Gaussians are culled
                # by alpha, matching GES's "standard periodic pruning for the 3D Gaussians".
                if (getattr(ingp, 'is_gestex_joint', False)
                        and iteration > args.ges_joint_iter
                        and args.ges_gs_prune_interval > 0
                        and iteration % args.ges_gs_prune_interval == 0
                        and hasattr(gaussians, '_is_textured') and gaussians._is_textured.numel() > 0):
                    _act = gaussians.get_opacity.flatten()
                    _untex = ~gaussians._is_textured
                    _prune = torch.logical_and(_act < float(args.ges_gs_prune_thresh), _untex)
                    _n = int(_prune.sum().item())
                    if 0 < _n < gaussians.get_xyz.shape[0]:
                        gaussians.prune_points(_prune)
                        tqdm.write(f"[GEStex iter={iteration}] pruned {_n} low-α untextured Gaussians "
                                   f"(α<{args.ges_gs_prune_thresh}); now {gaussians.get_xyz.shape[0]} total.")
        # ========================================================================

        # `--method res_3d` STAGE 2: SPLIT event. At --res_3d_iter (default
        # 10000) each Gauss duplicates → 2D residual-carrier + 3D EWA SV-
        # carrier. The renderer then dispatches to the SINGLE-PASS dual-
        # cascade kernel (diff_surfel_res_3d) which composes the final image
        # as LRU(C_sv + C_tex) inside the kernel + Python-side LRU. Adam is
        # rebuilt for the doubled tensor. Stage 1's mode flip must have
        # already happened — if res_switch_iter == res_3d_iter (defaults),
        # stage 1 just fired this same iteration above; if res_switch_iter
        # is earlier, the flip was applied at an earlier iter.
        if (args.method in ("res_3d", "res_3d_paired", "res_3d_double")
                and args.res_3d_iter > 0
                and iteration == args.res_3d_iter):
            _n_before = gaussians.get_xyz.shape[0]
            # Variant flags:
            #  - `_paired` (`--method res_3d_paired`): joint T cascade via
            #    `diff_surfel_mixed_3d`. Opacity scaled by --texsplit_tex_frac.
            #    Tex carriers keep SV (full mixed_3d capacity).
            #  - `_double` (`--method res_3d_double`): dual T cascade via
            #    `diff_surfel_res_3d`. Tex carriers keep SV too (bias gate
            #    flipped to 0 below). Original α on both copies (independent
            #    cascades — no scaling needed).
            #  - else (`--method res_3d`): dual T cascade, tex residual ONLY
            #    (bias gate stays at default 1).
            _paired = (args.method == "res_3d_paired")
            _double = (args.method == "res_3d_double")
            if _paired:
                _tex_frac = float(getattr(args, "texsplit_tex_frac", 0.5))
                _untex_frac_arg = float(getattr(args, "texsplit_untex_frac", -1.0))
                _untex_frac = (1.0 - _tex_frac) if _untex_frac_arg < 0.0 else _untex_frac_arg
                print(f"[STAGE2 iter={iteration}] SPLIT (paired): duplicating {_n_before} surfels → "
                      f"{_n_before} 2D-residual-carriers (α×{_tex_frac:.3f}) + "
                      f"{_n_before} 3D-EWA-SV-carriers (α×{_untex_frac:.3f}). "
                      f"Joint-T cascade (shared accumulator), LRU α={getattr(args, 'lru', 0.0)} on post-blend image.")
            elif _double:
                _tex_frac = None
                _untex_frac = None
                print(f"[STAGE2 iter={iteration}] SPLIT (double): duplicating {_n_before} surfels → "
                      f"{_n_before} 2D tex carriers (SV + residual) + {_n_before} 3D EWA SV carriers. "
                      f"DUAL T cascade, tex bias gate OFF (full per-Gauss capacity), "
                      f"LRU α={getattr(args, 'lru', 0.0)} on post-blend image.")
            else:
                _tex_frac = None
                _untex_frac = None
                print(f"[STAGE2 iter={iteration}] SPLIT: duplicating {_n_before} surfels → "
                      f"{_n_before} 2D-residual-carriers + {_n_before} 3D-EWA-SV-carriers. "
                      f"LRU α={getattr(args, 'lru', 0.0)} active on the post-blend C_tex.")
            # 1) Duplicate the live surfel set with the residual / SV split.
            # `keep_tex_sv=True` for paired AND double: tex carriers KEEP their
            # SV/SH params. `keep_tex_sv=False` (plain res_3d): SV/SH zeroed
            # so tex contributes residual only.
            with torch.no_grad():
                gaussians.split_at_res_3d(
                    tex_opacity_scale=_tex_frac,
                    untex_opacity_scale=_untex_frac,
                    keep_tex_sv=(_paired or _double))
            # 2) Rebuild Adam state for the doubled tensor (fresh per-row
            #    momentum, since per-row paste is fragile across this shape change).
            gaussians.training_setup(opt)
            torch.cuda.empty_cache()
            # 3) CUDA setup on diff_surfel_mixed_3d (was untouched pre-split
            #    because the renderer routes res_3d to diff_surfel_3D_sh_res
            #    until the split). Install the full setter mirror so any
            #    future `from diff_surfel_3D_sh_res import set_X` call (e.g.
            #    contrib_thresh, count_thresh, weight_reg_lambda, aa_kernel,
            #    skip_mlp_grad, depth_sort, overdraw_lambda, anti_alias,
            #    compact_mult, ste_relu, lru_slope, residual_mode,
            #    activation_bias, ...) ALSO updates the diff_surfel_mixed_3d
            #    device globals. Match exactly the mirror that --method
            #    mixed_3d / mixed_3d_sep installs at startup.
            import diff_surfel_3D_sh_res as _ds_orig
            import diff_surfel_mixed_3d as _ds_mirror
            # `--method res_3d` SINGLE-PASS dual-cascade submodule (optional —
            # falls back to the two-render mixed_3d path if not built).
            try:
                import diff_surfel_res_3d as _ds_mirror_res3d
            except ImportError:
                _ds_mirror_res3d = None
            # `--method res_3d_paired` SLIM submodule (clone of mixed_3d with
            # out_others trimmed to 5 channels). Mirror setters into it too so
            # the kernel device-globals (sh_bias, residual_mode, lru_slope, ...)
            # track diff_surfel_3D_sh_res.
            try:
                import diff_surfel_res_3d_paired as _ds_mirror_res3d_paired
            except ImportError:
                _ds_mirror_res3d_paired = None
            _MIRRORED_SETTERS = (
                'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                'set_overdraw_lambda', 'set_weight_reg_lambda',
                'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                'set_depth_sort', 'set_ste_relu', 'set_lru_slope',
            )
            def _make_mirror(of, mf, mf2, mf3):
                def _wrapped(*a, **k):
                    of(*a, **k); mf(*a, **k)
                    if mf2 is not None:
                        mf2(*a, **k)
                    if mf3 is not None:
                        mf3(*a, **k)
                return _wrapped
            for _name in _MIRRORED_SETTERS:
                if not hasattr(_ds_orig, _name) or not hasattr(_ds_mirror, _name):
                    continue
                _of = getattr(_ds_orig, _name)
                _mf = getattr(_ds_mirror, _name)
                _mf2 = getattr(_ds_mirror_res3d, _name, None) if _ds_mirror_res3d is not None else None
                _mf3 = getattr(_ds_mirror_res3d_paired, _name, None) if _ds_mirror_res3d_paired is not None else None
                setattr(_ds_orig, _name, _make_mirror(_of, _mf, _mf2, _mf3))
            print("[STAGE2] installed setter mirror diff_surfel_3D_sh_res "
                  "→ diff_surfel_mixed_3d"
                  + (" + diff_surfel_res_3d" if _ds_mirror_res3d is not None else "")
                  + (" + diff_surfel_res_3d_paired" if _ds_mirror_res3d_paired is not None else "")
                  + ".")
            # Now re-fire EVERY setter that was called during startup so the
            # mixed_3d submodule catches up to current state. The simplest
            # safe way: re-apply the few we know are user-tunable here.
            # set_residual_mode(2) is idempotent — if stage 1 already fired
            # it's a no-op, but if stage 1 == stage 2 it's needed.
            from diff_surfel_3D_sh_res import (set_residual_mode, set_lru_slope,
                                                 set_activation_bias, set_anti_alias,
                                                 set_compact_mult, set_aa_kernel_size,
                                                 set_contrib_thresh, set_count_thresh,
                                                 set_overdraw_lambda, set_weight_reg_lambda)
            set_residual_mode(2)
            args._residual_mode = 2
            if float(getattr(args, 'lru', 0.0)) != 0.0:
                set_lru_slope(float(args.lru))
            _ab = getattr(args, 'activation_bias', [0.5, 0.0])
            set_activation_bias(float(_ab[0]), float(_ab[1]))
            # Re-apply other startup-tunables (each setter is a noop if its
            # arg matches the current device global, so re-applying is free).
            if hasattr(args, 'contribution_thresh'):
                set_contrib_thresh(float(args.contribution_thresh))
            if hasattr(args, 'count_thresh'):
                set_count_thresh(int(args.count_thresh))
            if hasattr(args, 'overdraw_reg'):
                set_overdraw_lambda(float(args.overdraw_reg))
            if hasattr(args, 'weight_reg'):
                set_weight_reg_lambda(float(args.weight_reg))
            if hasattr(args, 'aa_2dgs'):
                set_aa_kernel_size(float(args.aa_2dgs))
            # 4) Renderer: flip post-split flag. For `--method res_3d` this
            #    triggers the dual-cascade single-pass dispatch (or two-render
            #    fallback). For `--method res_3d_paired` this triggers joint-T
            #    routing through mixed_3d + ALSO turns on the per-Gauss bias
            #    gate so textured carriers contribute residual only (no +0.5
            #    floor) inside the mixed_3d kernel.
            if ingp is not None:
                if _paired:
                    ingp.is_res_3d_paired_post_split = True
                elif _double:
                    ingp.is_res_3d_double_post_split = True
                else:
                    ingp.is_res_3d_post_split = True
            # Bias gate notes:
            #  - `--method res_3d_paired` (mixed_3d kernel): kernel default 0 →
            #    do NOT call set_textured_bias_gate(1). Tex carriers contribute
            #    ReLU(SV+0.5) + residual via the joint cascade.
            #  - `--method res_3d_double` (diff_surfel_res_3d kernel): kernel
            #    default 1 (matches plain `res_3d` byte-identical). For double
            #    we want tex to keep SV → explicitly call
            #    `set_textured_bias_gate(0)` to flip the device global OFF.
            #  - `--method res_3d`: leave default 1 → tex sh_color = 0.
            if _double:
                try:
                    from diff_surfel_res_3d import set_textured_bias_gate as _set_gate
                    _set_gate(0)
                    print("[STAGE2] set_textured_bias_gate(0) on diff_surfel_res_3d "
                          "(tex carriers keep SV + residual).")
                except (ImportError, AttributeError) as _e:
                    print(f"[STAGE2] WARNING: set_textured_bias_gate failed: {_e}")
            # NOTE: --lru continues to apply at the new (mode-2) site; the
            # renderer's deferred-ReLU dispatch picks F.leaky_relu when
            # ingp.lru_slope > 0 (which `set_lru_slope(args.lru)` already set
            # at startup — unchanged here).

        # `--method mixed` one-shot textured/untextured split event. Fires once at
        # iteration == args.texsplit (>0). Pure duplication of the live surfel set
        # — textured copy + untextured copy (no depth reinit; that shocked a
        # partially-converged scene). Adam is rebuilt for the doubled tensor.
        if args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep") and args.texsplit > 0 and iteration == args.texsplit:
            _n_before = gaussians.get_xyz.shape[0]
            print(f"[TEXSPLIT] iter={iteration}: duplicating {_n_before} surfels → "
                  f"{_n_before} textured + {_n_before} untextured")
            with torch.no_grad():
                # mixed_3d: untextured surfels are 3D ellipsoids → create the
                # learnable 3rd axis. Plain mixed stays 2D (no _scaling_z) so its
                # PLY/optimizer footprint is unchanged.
                _tex_frac = float(getattr(args, "texsplit_tex_frac", 0.5))
                _untex_frac = float(getattr(args, "texsplit_untex_frac", -1.0))
                if _untex_frac < 0.0:
                    _untex_frac = 1.0 - _tex_frac
                gaussians.split_at_texsplit(
                    make_scaling_z=(args.method in ("mixed_3d", "mixed_3d_sep")),
                    tex_opacity_scale=_tex_frac,
                    untex_opacity_scale=_untex_frac)
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()

        _pose_corr = _pose_correction_for(viewpoint_cam)
        _deform = _deform_for(viewpoint_cam)
        # --deform: point the in-kernel hash query at the current CANONICAL centers
        # (un-deformed _xyz). means3D passed to the rasterizer is the DEFORMED set
        # (render adds _deform[0]); geometry/SV use deformed, hash uses canonical.
        # Refreshed every iter (positions move / densify reallocates _xyz). The
        # detached view shares storage with the persistent _xyz Parameter, so the
        # device pointer stays valid through this step's forward AND backward.
        _canon_hold = None
        if deform_raster_mod is not None:
            _canon_hold = gaussians.get_xyz.detach()
            deform_raster_mod.set_canonical_xyz(_canon_hold)
        # Texture-query dropout: arm for this iteration (seed = iteration so the dropped set
        # rotates). Stays active through the forward AND total_loss.backward() below; any
        # --decomp renders this iter share the same seed → consistent. Reset to 0 after backward.
        if _use_tex_dropout:
            _set_tex_dropout(args.texture_dropout, iteration)
        # First-intersection handling (GEStex harden): two INDEPENDENT, composable
        # mechanisms on the diff_surfel_3D_sh_res_harden clone —
        #  1. tile-depth SORT from --ges_first_int_iter (10k): exact alpha blending
        #     at any opacity, ordered by intersection depth at the tile-center ray.
        #     Valid for the translucent early-harden.
        #  2. frontmost-first PROMOTION from --ges_frontmost_iter (18k): GES-paper
        #     scheme, only valid once surfels are near-opaque (floor ~0.8+);
        #     converges to the joint stage's z-buffer.
        # Set per-iter (robust to checkpoint resume); the renderer reads
        # ingp.first_int_sort / ingp.frontmost_on.
        if ingp is not None and args.is_gestex:
            _fis_it = int(getattr(args, 'ges_first_int_iter', -1))
            _fm_it = int(getattr(args, 'ges_frontmost_iter', -1))
            ingp.first_int_sort = (_fis_it >= 0 and iteration >= _fis_it)
            ingp.frontmost_on = (_fm_it >= 0 and iteration >= _fm_it)
        # --backface_cull: per-view surfel backface culling via override_opacity
        # (no CUDA changes — same per-Gauss mechanism as the mesh cull).
        # Disc normal = 3rd rotation axis, sign-disambiguated OUTWARD via the
        # cloud centroid (2DGS surfel normals have arbitrary sign; centroid
        # orientation is exact for star-convex shapes and good enough
        # elsewhere given the cos threshold). Surfels clearly facing AWAY
        # from the camera are zero-opacitied for this view — so opposite-
        # shell surfels cannot render, cannot silence noise pressure, and
        # cannot pollute last-fragment depths with view-inconsistent tails
        # (the source of TSDF holes in the cull-mesh pipeline). Culled
        # surfels simply get no gradient from this view; training adapts by
        # representing each surface with properly-oriented surfels.
        _bfc_override = None
        if (getattr(args, 'backface_cull', False)
                and iteration >= int(getattr(args, 'backface_cull_after', 5000))):
            from utils.general_utils import build_rotation
            # Schedulable strictness: with --backface_cull_anneal > 0 the cos
            # threshold anneals linearly from --backface_cull_cos_start (mild —
            # only near-anti-facing surfels culled) down to --backface_cull_cos
            # (strict) over the anneal window. Lets the representation reorient
            # gradually instead of losing a big surfel population in one iter
            # (divergence risk). anneal == 0 → brute constant threshold.
            _bfc_after = int(getattr(args, 'backface_cull_after', 5000))
            _cos_thr = float(getattr(args, 'backface_cull_cos', 0.2))
            _bfc_anneal = int(getattr(args, 'backface_cull_anneal', 0))
            if _bfc_anneal > 0:
                _t01 = min(1.0, max(0.0, (iteration - _bfc_after) / float(_bfc_anneal)))
                _cos_start = float(getattr(args, 'backface_cull_cos_start', 0.9))
                _cos_thr = _cos_start + (_cos_thr - _cos_start) * _t01
            with torch.no_grad():
                _xyz = gaussians.get_xyz.detach()
                _nrm = build_rotation(gaussians.get_rotation.detach())[:, :, 2]  # [N,3] disc normal
                _ctr = _xyz.mean(dim=0, keepdim=True)
                _sgn = torch.sign((_nrm * (_xyz - _ctr)).sum(dim=1, keepdim=True))
                _sgn = torch.where(_sgn == 0, torch.ones_like(_sgn), _sgn)
                _out_n = _nrm * _sgn                                            # outward-oriented
                _vdir = _xyz - viewpoint_cam.camera_center.cuda().view(1, 3)
                _vdir = _vdir / (_vdir.norm(dim=1, keepdim=True) + 1e-12)
                _facing = (_vdir * _out_n).sum(dim=1)
                # SMOOTH fade over ±BFC_FADE_BAND around the threshold —
                # byte-matches the WebGPU viewer's smoothstep fade
                # (surfel_cull.wgsl BFC_FADE_BAND) so training and deployed
                # inference share the exact transition behavior. A binary
                # cull here + smooth fade at render = mismatch in the band;
                # and the binary version pops silhouette surfels per-view
                # during training, which the smooth version avoids.
                _BFC_FADE_BAND = 0.08
                _t = torch.clamp((_facing - (_cos_thr - _BFC_FADE_BAND))
                                 / (2.0 * _BFC_FADE_BAND), 0.0, 1.0)
                _fade = 1.0 - _t * _t * (3.0 - 2.0 * _t)   # smoothstep
                _bfc_keep = _fade.to(gaussians.get_opacity.dtype).view(-1, 1)
                if iteration % 1000 == 0:
                    _culled = (_fade < 0.5).float().mean().item()
                    tqdm.write(f"[BFC iter={iteration}] cos_thr={_cos_thr:.3f}  "
                               f"culled(fade<0.5)={100.0 * _culled:.1f}% "
                               f"of {_fade.numel():,} surfels (this view, smooth fade)")
            _bfc_override = gaussians.get_opacity * _bfc_keep   # grads flow ∝ fade

        def _do_train_render():
            return render(viewpoint_cam, gaussians, pipe, current_bg, ingp = ingp,
                beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance,
                use_xyz_mode = args.use_xyz_mode, decompose_mode = dataset.decompose_mode,
                temperature = temperature, force_ratio = args.force_ratio, no_gumbel = args.no_gumbel,
                dropout_lambda = args.dropout_lambda, is_training = True, aabb_mode = args.aabb,
                aa = args.aa, aa_threshold = args.aa_threshold, skybox = active_skybox,
                background_mode = background_mode, bg_hashgrid = active_bg_hashgrid,
                detach_hash_grad = args.detach_hash_grad, max_intersections_per_pixel = args.max_intersections_per_pixel,
                lowpass = args.lowpass, pixel_center = args.pixel_center,
                antialiasing = args.antialiasing, sv_metric = args.sv_metric,
                pose_correction = _pose_corr, deform = _deform,
                override_opacity = _bfc_override)

        # --random_mesh_depth: DEPTH-LOCATED noise wall (the NGS-faithful
        # variant; see project memory noise-pressure-cull-mesh-findings).
        # Two-pass: (1) no_grad render → this iter's own median-depth map;
        # (2) install it (+eps) as the CUDA per-pixel occluder so fragments
        # BEHIND the median are dropped in fwd+bwd, then render for the loss.
        # The composite block below then adds noise on the WALLED render's
        # (1 − rend_alpha) — i.e. noise weighted by the transmittance AT the
        # wall, not after the full ray. Deep fuzz is excluded from the loss
        # term entirely, so the only gradient escape is closing frontier
        # cracks — unlike --random_mesh (back-plate), whose T_end weighting
        # lets any-depth saturation silence the noise (ragged-tail loophole).
        # The wall tracks the live median each iter → no static-mesh shock,
        # no hard opacity zeroing → none of the finetune's pop-off artifacts.
        _wall_mask = None
        _use_depth_wall = (getattr(args, 'random_mesh_depth', False)
                           and iteration >= int(getattr(args, 'random_mesh_after', 5000)))
        if _use_depth_wall:
            from diff_surfel_3D_sh_res import set_occluder_depth, clear_occluder_depth
            # Wall eps, optionally ANNEALED: start deep behind the median
            # (mild pressure — only far fuzz is excluded from the loss) and
            # tighten linearly to --random_mesh_wall_eps over
            # --random_mesh_wall_anneal iters. Avoids the convergence shock of
            # slamming a tight wall onto a half-converged model (the loss
            # suddenly demands the front-of-median prefix explain ALL of GT).
            _eps = float(getattr(args, 'random_mesh_wall_eps', 0.01))
            _anneal = int(getattr(args, 'random_mesh_wall_anneal', 0))
            if _anneal > 0:
                _t01 = min(1.0, max(0.0, (iteration - int(getattr(args, 'random_mesh_after', 5000))) / float(_anneal)))
                _eps_start = float(getattr(args, 'random_mesh_wall_eps_start', 0.15))
                _eps = _eps_start + (_eps - _eps_start) * _t01
            with torch.no_grad():
                _pre = _do_train_render()
                _med = _pre['depth_median'].squeeze(0).float()   # [H,W] kernel-depth units; 0 = no contributor
                _wall = torch.where(
                    _med > 0,
                    _med + _eps,
                    torch.full_like(_med, float('inf')))         # background: no wall
                del _pre
            _wall = _wall.contiguous()
            set_occluder_depth(_wall)
            render_pkg = _do_train_render()
            # NOTE: do NOT clear the occluder here. The backward must run with
            # the same wall as the forward (see the clear after
            # total_loss.backward() below); `_wall` stays referenced in this
            # scope so the device-global pointer never dangles.
            _wall_mask = torch.isfinite(_wall).to(torch.float32).unsqueeze(0)  # [1,H,W]
        else:
            render_pkg = _do_train_render()
        # --deform: optional L2 reg keeping the per-frame deformation minimal.
        deform_reg_loss = torch.tensor(0.0, device="cuda")
        if _deform is not None and args.deform_reg > 0.0:
            _dreg = _deform[0].pow(2).mean()
            if _deform[1] is not None:
                _dreg = _dreg + _deform[1].pow(2).mean()
            deform_reg_loss = args.deform_reg * _dreg

        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_fwd_end = time.time()

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # --ppisp: apply the learned ISP to the rendered radiance. Placed HERE,
        # before the random-background composite below, on purpose: the composite
        # adds the same raw bg to both `image` and `gt_image`, so keeping the ISP
        # upstream of it means the ISP never sees (and never tries to explain) the
        # synthetic background. Everything downstream — error_img, the L1/SSIM/
        # LPIPS losses, the error-guided reg weights, the debug dumps — then
        # operates on the ISP-corrected image, which is what GT is compared to.
        ppisp_overflow_loss = torch.tensor(0.0, device="cuda")
        if ppisp is not None:
            if args.ppisp_overflow_w > 0 and not args.ppisp_no_camera:
                # The camera path clamps to [0,1] inside the kernel, so pixels
                # rendering above 1 get exactly zero gradient and can never come
                # back down. Restore an explicit downward push for them.
                ppisp_overflow_loss = args.ppisp_overflow_w * \
                    (image - 1.0).clamp(min=0.0).mean()
            image = _ppisp_apply(image, viewpoint_cam)
            if render_pkg.get('render_untex', None) is not None:
                render_pkg['render_untex'] = _ppisp_apply(
                    render_pkg['render_untex'], viewpoint_cam)

        # --blur_split (mini-splatting2): accumulate per-Gaussian dominance count
        # as a "blurry / oversized" flag. Resized/reset after every densify event.
        if args.blur_split:
            _N_now = gaussians.get_xyz.shape[0]
            if (not hasattr(gaussians, "_blur_split_mask")
                    or gaussians._blur_split_mask is None
                    or gaussians._blur_split_mask.shape[0] != _N_now):
                gaussians._blur_split_mask = torch.zeros(_N_now, dtype=torch.bool, device="cuda")
            _max_idx = render_pkg.get("max_contrib_idx", None)
            if _max_idx is not None and _max_idx.numel() > 0:
                _idx_flat = _max_idx.long().reshape(-1)
                _valid = _idx_flat >= 0
                if _valid.any():
                    _ids = _idx_flat[_valid]
                    _ids = _ids[_ids < _N_now]  # safety: ignore stale ids past current N
                    _area = torch.bincount(_ids, minlength=_N_now)
                    _h, _w = image.shape[-2], image.shape[-1]
                    gaussians._blur_split_mask |= (_area > (_h * _w) / args.blur_thresh)

        # MSv2 SparseGaussianAdam: track visibility for sparse optimizer step
        if args.mini:
            mini_last_visibility = radii > 0

        gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)

        # Apply random background for unbiased opacity training
        # Skip if skybox is active - skybox already provides the background
        if use_random_bg and iteration >= random_bg_start_iter and active_skybox is None:
            H, W = image.shape[1], image.shape[2]
            # Generate a single random RGB value for the entire background
            # Changes every 100 iterations
            torch.manual_seed(iteration // 100)
            random_bg_color = torch.rand(3, 1, 1, device="cuda")
            random_bg = random_bg_color.expand(3, H, W)
            rend_alpha = render_pkg["rend_alpha"]

            # In 3D/3D_direct mode, detach rend_alpha to prevent double counting of geometry gradients.
            # Geometry grads flow through IntersectionOpacityGrad; using (1-rend_alpha) would create
            # a second gradient path through native backward, causing gradients to be ~3x too large.
            if args.method in ["3D", "3D_direct"]:
                rend_alpha = rend_alpha.detach()

            # --random_mesh: fire once iter >= --random_mesh_after. Composite
            # BLOCKY noise into the region where rend_alpha ≥ 0.5 (T=0.5
            # median-depth crossing has been reached, i.e., an opaque-ish
            # frontier exists at that pixel). Solid random_bg stays outside
            # that region. Same noise is folded into both the render and the
            # GT below via `bg_composite`; semi-transparent leading-frontier
            # surfels then take a |(1 - rend_alpha) · 0.5 · noise|-scale
            # gradient penalty until they saturate opacity → opaque manifold
            # at the median depth. No pre-computed mesh required — the mask
            # is derived from the live render.
            _noise_on = ((getattr(args, 'random_mesh', False) or _use_depth_wall)
                         and iteration >= int(getattr(args, 'random_mesh_after', 5000)))
            if _noise_on:
                torch.manual_seed(iteration * 977 + 31)   # decorrelated stream from bg
                G = max(1, int(getattr(args, 'random_mesh_grid', 32)))
                noise_lo = torch.rand(1, 3, G, G, device="cuda")
                noise_hi = torch.nn.functional.interpolate(
                    noise_lo, size=(H, W), mode="nearest").squeeze(0)
                # Noise-region mask (non-differentiable gate; opacity gradient
                # flows via (1 − rend_alpha) in the composite line below):
                #  * depth-wall mode: pixels where a median wall exists — and
                #    rend_alpha here is the WALLED render's alpha, so the noise
                #    term is T_at_wall · noise (depth-located pressure).
                #  * back-plate mode: live rend_alpha ≥ 0.5 heuristic.
                if _wall_mask is not None:
                    m_hit = _wall_mask
                else:
                    m_hit = (rend_alpha.detach() >= 0.5).to(torch.float32)
                    if m_hit.dim() == 2:
                        m_hit = m_hit.unsqueeze(0)
                bg_composite = m_hit * noise_hi + (1.0 - m_hit) * random_bg
            else:
                bg_composite = random_bg

            # Apply composited background to rendered image
            image = image + (1.0 - rend_alpha) * bg_composite

            # Apply same composited background to GT image
            gt_alpha_for_bg = viewpoint_cam.gt_alpha_mask.cuda().float() if cfg_model.settings.gt_alpha else (gt_image != 0).any(dim=0, keepdim=True).float()
            gt_image = gt_image + (1.0 - gt_alpha_for_bg) * bg_composite

            # Periodic dump of the noise-composited pair into training_output/
            # — the EXACT tensors the photometric loss compares this iter, so
            # the injected noise is visible on disk (the regular debug renders
            # are separate no_grad renders without compositing and never show
            # it). `{iter}_noised_render.png` = walled/back-plate render +
            # noise; `{iter}_noised_gt.png` = GT + same noise outside its
            # alpha. Cheap (2 PNGs / interval); gated on noise being active.
            if _noise_on and int(getattr(args, 'noise_debug_interval', 1000)) > 0 \
                    and iteration % int(getattr(args, 'noise_debug_interval', 1000)) == 0:
                _nod = os.path.join(scene.model_path, 'training_output')
                os.makedirs(_nod, exist_ok=True)
                save_img_u8(image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                            os.path.join(_nod, f'{iteration}_noised_render.png'))
                save_img_u8(gt_image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                            os.path.join(_nod, f'{iteration}_noised_gt.png'))
        # Apply same background to GT image for consistent loss computation
        elif use_alternating_bg:
            gt_alpha_for_bg = viewpoint_cam.gt_alpha_mask.cuda().float() if cfg_model.settings.gt_alpha else (gt_image != 0).any(dim=0, keepdim=True).float()
            gt_image = gt_image + (1.0 - gt_alpha_for_bg) * current_bg.unsqueeze(-1).unsqueeze(-1)
            
            # Debug: save images at bg transition iterations to verify alternating works
            debug_iters = [first_iter, first_iter + 9, first_iter + 10, first_iter + 19, first_iter + 20]
            if iteration in debug_iters:
                output_path = os.path.join(scene.model_path, 'training_output')
                bg_str = "white" if current_bg[0] > 0.5 else "black"
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), 
                           os.path.join(output_path, f'debug_iter{iteration}_render_{bg_str}.png'))
                save_img_u8(gt_image.permute(1,2,0).detach().cpu().numpy(), 
                           os.path.join(output_path, f'debug_iter{iteration}_gt_{bg_str}.png'))

        error_img = torch.abs(gt_image - image)

        if cfg_model.settings.gt_alpha :
            if viewpoint_cam.gt_alpha_mask is None:
                print(f"[ERROR] gt_alpha=True but gt_alpha_mask is None for {viewpoint_cam.image_name}!")
                gt_alpha = (gt_image != 0).any(dim=0, keepdim=True).float()
            else:
                gt_alpha = viewpoint_cam.gt_alpha_mask.cuda().float()
        else:
            gt_alpha = (gt_image != 0).any(dim=0, keepdim=True).float()
        
        try:
            if cfg_model.settings.gs_alpha and ingp is not None:
                gt_alpha = viewpoint_cam.gs_alpha_mask.cuda().float()
        except:
            if not loaded_from_warmup and not args.cold:
                print(f"Error! no gs alpha for {viewpoint_cam.image_name} .")
            pass

        rend_alpha = render_pkg['rend_alpha']
        
        # --blurprog: curriculum — interpolate target from gt_low (heavy structure)
        # to gt (fully sharp) across the first `blurprog_until` iters. After that,
        # target is pure gt (same as no curriculum).
        if args.blurprog:
            gt_low_cam = getattr(viewpoint_cam, 'gt_low', None)
            if gt_low_cam is None:
                raise RuntimeError(
                    f"--blurprog: camera {viewpoint_cam.image_name} has no gt_low. "
                    f"Scene loader should have populated it.")
            t_prog = min(1.0, float(iteration) / max(1, args.blurprog_until))
            if args.blurprog_schedule == "cosine":
                _alpha = 0.5 * (1.0 - np.cos(np.pi * t_prog))
            else:
                _alpha = t_prog  # linear
            gt_image = (1.0 - _alpha) * gt_low_cam + _alpha * gt_image
            if iteration % 500 == 0:
                tqdm.write(f"[BLURPROG iter={iteration}] t={t_prog:.3f}, α={_alpha:.3f} "
                           f"(α=0 → gt_low, α=1 → gt)")

        # `--l2` / `--l1` (mixed_3d[_sep] only): per-Gauss loss routing via the
        # rasterizer's dual image-output. `image` and `image_untex` are
        # numerically identical (both = the full blended render) but they
        # are SEPARATE autograd nodes in the rasterizer Function's graph.
        # The Function's backward receives two upstream image gradients and
        # routes them per-Gauss inside the CUDA kernel based on the
        # `is_textured` flag — textured Gauss accumulate gradient from
        # `dL/d image_tex` (L1+SSIM), untextured Gauss accumulate from
        # `dL/d image_untex` (L2 or L1 depending on flag). Pre-`--texsplit`
        # (no untextured rows) or when neither flag is set → kernel reverts
        # to single-loss behavior, byte-identical to default.
        _split_untex_loss = (
            'l2' if getattr(args, 'l2', False)
            else 'l1' if getattr(args, 'l1', False)
            else None
        )
        _split_active = (_split_untex_loss is not None
                         and args.method in ("mixed_3d", "mixed_3d_sep")
                         and render_pkg.get('render_untex', None) is not None
                         and hasattr(gaussians, '_is_textured')
                         and gaussians._is_textured.numel() == gaussians.get_xyz.shape[0]
                         and bool((~gaussians._is_textured).any().item()))
        if _split_active:
            image_untex = render_pkg['render_untex']
            Ll1 = l1_loss(image, gt_image)
            if _split_untex_loss == 'l2':
                L_untex = ((image_untex - gt_image) ** 2).mean()
            else:  # 'l1'
                L_untex = (image_untex - gt_image).abs().mean()
            loss = ((1.0 - opt.lambda_dssim) * Ll1
                    + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    + L_untex)
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ---- proberes residual distillation (--probe_distill_dir) ----------------
        # Match the probe/atlas blended residual to the TEACHER's, per train view:
        #   R = sum_i T_i*alpha_i*res_i   (mode 2 => signed, no per-Gauss ReLU)
        # Rendered with 'tex_only_raw' (sh_bias=-999 kills SV, per-pixel clamp
        # bypassed). Unlike the bake — which resolves a contested texel with blind
        # geometric weights — this weights every surfel by its ACTUAL T*alpha
        # visibility across the training views, so collisions resolve toward what is
        # actually seen.
        if getattr(args, 'probe_distill_dir', None) and args.method == "proberes":
            global _DISTILL_TGT
            if '_DISTILL_TGT' not in globals() or _DISTILL_TGT is None:
                _dp = os.path.join(args.probe_distill_dir, 'targets.pt')
                _blob = torch.load(_dp, map_location='cpu')
                _DISTILL_TGT = _blob['targets']
                print(f"[DISTILL] loaded {len(_DISTILL_TGT)} teacher residual targets "
                      f"from {_dp} (residual_mode={_blob.get('residual_mode')})")
            _tgt = _DISTILL_TGT.get(viewpoint_cam.image_name)
            if _tgt is not None:
                _rres = render(viewpoint_cam, gaussians, pipe, current_bg,
                               decompose_mode='tex_only_raw',
                               ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                               use_xyz_mode=args.use_xyz_mode, is_training=True,
                               aabb_mode=args.aabb, aa=args.aa, aa_threshold=args.aa_threshold,
                               lowpass=args.lowpass, pixel_center=args.pixel_center,
                               antialiasing=args.antialiasing, sv_metric=args.sv_metric,
                               max_intersections_per_pixel=args.max_intersections_per_pixel,
                               )['render']
                L_distill = (_rres - _tgt.to(_rres.device, torch.float32)).abs().mean()
                if args.probe_distill_only:
                    loss = L_distill
                else:
                    _ld = float(args.probe_distill_lambda)
                    loss = (1.0 - _ld) * loss + _ld * L_distill
                if iteration % int(os.environ.get('PROBERES_DISTILL_LOG_EVERY', 500)) == 0:
                    print(f"[DISTILL iter={iteration}] L_distill={L_distill.item():.5f} "
                          f"R_probe std={_rres.std().item():.4f} "
                          f"R_teach std={_tgt.float().std().item():.4f} "
                          f"({'distill-only' if args.probe_distill_only else f'joint lam={args.probe_distill_lambda}'})")

        # --lpips_w: perceptual loss on the final rendered image (any method; in
        # GEStex it covers both the pre-20k cascade and the post-20k sort-free
        # composite since both land in render_pkg['render']). Backbone frozen;
        # normalize=True maps [0,1] -> [-1,1] inside the lpips module.
        lpips_train_loss = torch.tensor(0.0, device="cuda")
        if args.lpips_w > 0 and iteration >= args.lpips_start_iter:
            global _LPIPS_NET
            if _LPIPS_NET is None:
                import lpips as _lpips_mod
                _LPIPS_NET = _lpips_mod.LPIPS(net=args.lpips_net).cuda().eval()
                _LPIPS_NET.requires_grad_(False)
                print(f"[LPIPS] training loss ON: net={args.lpips_net} w={args.lpips_w} "
                      f"from iter {args.lpips_start_iter}")
            lpips_train_loss = args.lpips_w * _LPIPS_NET(
                image.unsqueeze(0), gt_image.unsqueeze(0), normalize=True).mean()

        # --decomp: supervise the structure branch against the guided-filter low-freq
        # image, and the hashgrid-MLP residual against the high-freq residual.
        #   default (--decomp):       L1(sh_only,  gt_low) + L1(tex_only, gt_high)
        #                             → 2 extra renders per iter (sh_only, tex_only)
        #   --decomp_comb:            L1(full,     gt_low) + L1(tex_only, gt_high)
        #                             → 1 extra render per iter (tex_only); reuses the
        #                               main-loop `image`. SH and hashgrid jointly fit
        #                               the blur; residual pulls hashgrid toward detail.
        decomp_sh_loss = torch.tensor(0.0, device="cuda")
        decomp_tex_loss = torch.tensor(0.0, device="cuda")
        if args.decomp and (iteration % max(1, args.decomp_interval) == 0):
            gt_low_cam = getattr(viewpoint_cam, 'gt_low', None)
            if gt_low_cam is None:
                raise RuntimeError(
                    f"--decomp: camera {viewpoint_cam.image_name} has no gt_low. "
                    f"Scene loader should have populated it.")
            gt_low_t = gt_low_cam
            gt_high_t = (gt_image - gt_low_t).clamp(-1.0, 1.0)

            _decomp_render_kwargs = dict(
                ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                record_transmittance=record_transmittance,
                use_xyz_mode=args.use_xyz_mode,
                temperature=temperature, force_ratio=args.force_ratio,
                no_gumbel=args.no_gumbel, dropout_lambda=args.dropout_lambda,
                is_training=True, aabb_mode=args.aabb,
                aa=args.aa, aa_threshold=args.aa_threshold,
                skybox=active_skybox, background_mode=background_mode,
                bg_hashgrid=active_bg_hashgrid,
                detach_hash_grad=args.detach_hash_grad,
                max_intersections_per_pixel=args.max_intersections_per_pixel,
                lowpass=args.lowpass, pixel_center=args.pixel_center,
                antialiasing=args.antialiasing, sv_metric=args.sv_metric,
                pose_correction=_pose_corr, deform=_deform)

            # Structure target: either the sh_only render (default) or the full
            # render already computed above (--decomp_comb).
            if args.decomp_comb:
                structure_render = image
            else:
                sh_pkg = render(viewpoint_cam, gaussians, pipe, current_bg,
                                decompose_mode='sh_only', **_decomp_render_kwargs)
                structure_render = sh_pkg['render']

            tex_pkg = render(viewpoint_cam, gaussians, pipe, current_bg,
                             decompose_mode='tex_only', **_decomp_render_kwargs)
            tex_only = tex_pkg['render']

            decomp_sh_loss = args.decomp_lambda_sh * l1_loss(structure_render, gt_low_t)
            decomp_tex_loss = args.decomp_lambda_tex * l1_loss(tex_only, gt_high_t)

            if iteration % 500 == 0:
                _tag = "full_vs_low" if args.decomp_comb else "sh_vs_low"
                tqdm.write(f"[DECOMP iter={iteration}] {_tag}={decomp_sh_loss.item():.6f}, "
                           f"tex_vs_high={decomp_tex_loss.item():.6f}, main_l1={Ll1.item():.6f}")

        # regularization
        lambda_normal = opt.lambda_normal if iteration > cfg_model.loss.normal_iter else 0.0
        lambda_dist = opt.lambda_dist if iteration > cfg_model.loss.dist_iter else 0.0
        lambda_mask = opt.lambda_mask if iteration > cfg_model.loss.mask_iter else 0.0

        # Unbiased Depth: replace 2DGS depth distortion with convergence loss.
        # Paper sets λ_dist=0 in the unbiased configuration (the two regularizers
        # would otherwise compete on the same depth signal with opposite biases).
        if args.unbiased:
            lambda_dist = 0.0

        # `--reg_after_texsplit`: in mixed/mixed_3d with --texsplit > 0, defer
        # normal+dist regs until AFTER the split fires. Useful when you want
        # the textured half to settle on its diffuse colour first and only
        # then snap its geometry. Default off → no behavior change.
        if (getattr(args, 'reg_after_texsplit', False) and
                args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep") and
                getattr(args, 'texsplit', -1) > 0 and iteration < args.texsplit):
            lambda_normal = 0.0
            lambda_dist = 0.0
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        # When no normal/dist regulariser is active and pipe.skip_aux_normal_dist
        # is on, all three of the above are placeholder zero tensors emitted by
        # the renderer (it skipped the expensive Python `depth_to_normal` +
        # render_normal view→world rotate). We mirror by skipping the
        # per-pixel normal_error / dist_loss intermediates here too.
        _skip_aux_loss = (bool(getattr(pipe, 'skip_aux_normal_dist', False))
                          and args.w_normal == 0.0
                          and lambda_normal == 0.0
                          and lambda_dist == 0.0)

        pixels = None
        if record_transmittance:
            pixels = render_pkg["cover_pixels"]
            transmittance_avg = render_pkg["transmittance_avg"]

        scales = gaussians.get_scaling
        alpha = gaussians.get_opacity
        
        mask_error = l1_loss(gt_alpha, rend_alpha).mean()
        if lambda_mask > 0 and opt.mask_dssim > 0:
            mask_error = ((1.0 - opt.mask_dssim) * mask_error
                          + opt.mask_dssim * (1.0 - ssim(rend_alpha.unsqueeze(0),
                                                         gt_alpha.unsqueeze(0))))
        mask_loss = lambda_mask * mask_error
        if lambda_mask > 0 and (iteration <= 5 or iteration % 500 == 0):
            tqdm.write(f"[ALPHA iter={iteration}] mask_error={mask_error.item():.4f} "
                       f"(λ={lambda_mask}, ssim_mix={opt.mask_dssim}, "
                       f"gt_alpha mean={gt_alpha.mean().item():.3f}, "
                       f"rend_alpha mean={rend_alpha.mean().item():.3f})")

        if _skip_aux_loss:
            normal_error = None
        else:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

        # `--method mixed[_3d]`: mask normal+dist regs to BETA-SURFEL pixels
        # (i.e. the textured 2DGS half). Untextured EWA Gaussians have a
        # constant +z normal (planar disc inside their projected ellipsoid)
        # and zero distortion contribution by design — including them in the
        # loss pulls geometry toward an artificial flat plane. We pick the
        # per-pixel max-contributor and gate on _is_textured[that idx].
        # Reduction uses sum/active_pixels so the loss magnitude is the
        # MEAN over textured-dominant pixels, not the full image (otherwise
        # the loss would silently scale with mask coverage).
        _tex_pix_mask = None
        if (args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep") and
                hasattr(gaussians, '_is_textured') and gaussians._is_textured.numel() > 0):
            mci = render_pkg.get('max_contrib_idx', None)
            if mci is not None and mci.numel() > 0:
                mci_safe = mci.clamp(min=0).long()
                _tex_pix_mask = (gaussians._is_textured[mci_safe] & (mci >= 0)).float()
        def _masked_mean(per_pix):
            """Mean over textured-dominant pixels, or full image if no mask."""
            if _tex_pix_mask is None:
                return per_pix.mean()
            n_active = _tex_pix_mask.sum().clamp(min=1.0)
            return (per_pix * _tex_pix_mask).sum() / n_active

        if _skip_aux_loss:
            normal_loss = torch.tensor(0.0, device="cuda")
            dist_loss = torch.tensor(0.0, device="cuda")
        else:
            if args.w_normal > 0.0 and iteration > cfg_model.loss.normal_iter:
                # Weighted normal consistency: relax where RGB error is high
                mse_per_pixel_n = ((image - gt_image) ** 2).mean(dim=0, keepdim=True).detach()
                w_n = torch.exp(-args.w_normal_gamma * mse_per_pixel_n)
                normal_loss = args.w_normal * _masked_mean(w_n * normal_error)
            else:
                normal_loss = lambda_normal * _masked_mean(normal_error)
            dist_loss = lambda_dist * _masked_mean(rend_dist)

        # Adaptive mode: regularization to encourage per-Gaussian features
        adaptive_reg_loss = torch.tensor(0.0, device="cuda")
        if args.method == "adaptive" and gaussians._adaptive_feat_dim > 0 and ingp is not None:
            # Update temperature (exponential decay)
            gaussians.update_temperature(iteration, opt.iterations)
            
            # Penalize hashgrid usage: mean(1 - mask) where mask=1 means use per-Gaussian
            if 'adaptive_mask' in render_pkg:
                mask = render_pkg['adaptive_mask']
                adaptive_reg_loss = args.lambda_adaptive * (1.0 - mask).mean()

        # Scout loss for diffuse_offset xyz mode: move Gaussians toward offset target
        # This is the "Squad follows Scout" geometry loss
        scout_loss = torch.tensor(0.0, device="cuda")
        if args.method == "diffuse_offset" and args.use_xyz_mode and 'scout_loss_data' in render_pkg:
            scout_data = render_pkg['scout_loss_data']
            points_base = scout_data['points_base']  # (H*W, 3) - has gradients
            points_target = scout_data['points_target']  # (H*W, 3) - detached
            scout_mask = scout_data['render_mask'].view(-1)  # (H*W,)
            
            # MSE loss only where alpha > 0
            diff = (points_base - points_target) ** 2  # (H*W, 3)
            diff_masked = diff[scout_mask.bool()]  # Only valid pixels
            if diff_masked.numel() > 0:
                scout_loss = args.scout_lambda * diff_masked.mean()

        # MCMC regularization losses - encourage sparsity in opacity and scale
        # Regularize ACTIVATED values (after sigmoid/exp) - following 3dgrut MCMC implementation
        mcmc_opacity_reg = torch.tensor(0.0, device="cuda")
        mcmc_scale_reg = torch.tensor(0.0, device="cuda")
        bce_start_iter = opt.iterations - args.bce_iter
        bce_phase_active = args.bce and iteration > bce_start_iter
        # Adaptive BCE: compute opacity statistic at BCE start and use as decision boundary
        if (args.bce_solo_adaptive or args.bce_adaptive) and iteration == bce_start_iter + 1:
            with torch.no_grad():
                opacities = gaussians.get_opacity.squeeze()
                if args.bce_adaptive_stat == "mean":
                    threshold = opacities.mean().item()
                else:
                    threshold = opacities.median().item()
            args._bce_adaptive_threshold = threshold
            print(f"\n[BCE_ADAPTIVE] Setting threshold to {args.bce_adaptive_stat} opacity: {threshold:.4f}")
        # If --bce_solo, skip opacity/scale regularization during BCE phase to let BCE work alone
        if not (args.bce_solo and bce_phase_active):
            if args.opacity_reg > 0:
                mcmc_opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            if args.scale_reg > 0:
                mcmc_scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        # --feature voronoi / SV: L1 loss on SV colors (sparsification) to match
        # sphericalvoronoi/radiance. Reference applies it ONLY inside the
        # densification window: `densify_from_iter < iter < densify_until_iter`
        # (radiance/train.py:134-139). Outside that window the L1 is silent so
        # colors can grow back if needed.
        # NB: in the reference, l_l1 is 1e-5 for blender/db/tandt and 0 for
        # indoor/outdoor — pass --sv_l1 0 when training on indoor/outdoor scenes.
        sv_l1_loss = torch.tensor(0.0, device="cuda")
        if args.feature in ("voronoi", "SV") and args.sv_l1 > 0 and gaussians._sv_colors.numel() > 0:
            in_densify_window = opt.densify_from_iter < iteration < opt.densify_until_iter
            if in_densify_window:
                sv_l1_loss = args.sv_l1 * gaussians._sv_colors.abs().sum(dim=-1).mean()

        # Adaptive_cat entropy regularization - encourage binary blend weights (0 or 1)
        adaptive_cat_reg_loss = torch.tensor(0.0, device="cuda")
        if args.method == "adaptive_cat" and hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight.numel() > 0:
            # Compute annealing factor (ramps from 0 to 1 starting at anneal_start iteration)
            if iteration >= args.adaptive_cat_anneal_start:
                progress = (iteration - args.adaptive_cat_anneal_start) / (opt.iterations - args.adaptive_cat_anneal_start)
                anneal_factor = min(1.0, progress)  # Linear ramp from 0 to 1
            else:
                anneal_factor = 0.0
            
            # Entropy regularization: -w*log(w) - (1-w)*log(1-w)
            # This penalizes weights near 0.5 and encourages weights near 0 or 1
            weight = torch.sigmoid(gaussians._adaptive_cat_weight)
            eps = 1e-7
            entropy = -(weight * torch.log(weight + eps) + (1 - weight) * torch.log(1 - weight + eps))
            adaptive_cat_reg_loss = args.lambda_adaptive_cat * anneal_factor * entropy.mean()

        # Adaptive_zero entropy regularization - encourage binary hash weights (0 or 1)
        adaptive_zero_reg_loss = torch.tensor(0.0, device="cuda")
        if args.method == "adaptive_zero" and hasattr(gaussians, '_adaptive_zero_weight') and gaussians._adaptive_zero_weight.numel() > 0:
            # Compute annealing factor (ramps from 0 to 1 starting at anneal_start iteration)
            if iteration >= args.adaptive_zero_anneal_start:
                progress = (iteration - args.adaptive_zero_anneal_start) / (opt.iterations - args.adaptive_zero_anneal_start)
                anneal_factor = min(1.0, progress)  # Linear ramp from 0 to 1
            else:
                anneal_factor = 0.0

            # BCE regularization with configurable threshold
            # Loss: -(t * log(w) + (1-t) * log(1-w))
            # Minimized when w = t, so pushes weights away from threshold t
            # t=0.5: symmetric push to 0 or 1
            # t=0.1: asymmetric, strongly pushes w<0.1 toward 0, w>0.1 toward 1
            # Use temperature-scaled sigmoid to match rendering
            weight = torch.sigmoid(gaussians._adaptive_zero_weight * temperature)
            eps = 1e-7
            t = args.bce_threshold
            bce = -(t * torch.log(weight + eps) + (1 - t) * torch.log(1 - weight + eps))
            bce_loss = args.lambda_adaptive_zero * anneal_factor * bce.mean()

            # Hash bias regularization: push weights toward 1 (favor hash queries)
            # L1 on (1 - weight) penalizes weights near 0, encouraging hash usage
            hash_bias_loss = args.hash_lambda * anneal_factor * (1 - weight).mean()

            # Parabola regularization: w*(1-w), max at 0.5, zero at 0 or 1
            parabola_loss = torch.tensor(0.0, device="cuda")
            if args.lambda_parabola > 0:
                parabola_loss = args.lambda_parabola * anneal_factor * (weight * (1 - weight)).mean()

            adaptive_zero_reg_loss = bce_loss + hash_bias_loss + parabola_loss

        # Adaptive_gate sparsity regularization
        # Penalize gate probability to encourage sparse hash usage (gates stay closed by default)
        adaptive_gate_reg_loss = torch.tensor(0.0, device="cuda")
        if args.method == "adaptive_gate" and hasattr(gaussians, '_gate_logits') and gaussians._gate_logits.numel() > 0:
            # Sparsity loss: penalize gate probability (not the mask)
            # This encourages gates to stay closed, only open where needed for quality
            gate_prob = torch.sigmoid(gaussians._gate_logits)
            adaptive_gate_reg_loss = args.lambda_sparsity * gate_prob.mean()

        # BCE opacity regularization - encourage binary opacity (0 or 1) to reduce foggy Gaussians
        # Applied only in the last bce_iter iterations
        bce_opacity_loss = torch.tensor(0.0, device="cuda")
        if args.bce and iteration > bce_start_iter:
            # Get activated opacity (after sigmoid, in [0, 1])
            opacity = gaussians.get_opacity.squeeze()  # (N,)
            eps = 1e-7
            # Adaptive threshold: use mean opacity computed at BCE start as decision boundary
            # Standard (t=0.5): entropy H(p) = -p*log(p) - (1-p)*log(1-p), pushes toward 0 or 1
            # Adaptive: t = mean opacity, used as decision boundary (not target!)
            #   opacity > t → target=1 (push up), opacity < t → target=0 (push down)
            t = getattr(args, '_bce_adaptive_threshold', 0.5)
            if t == 0.5:
                # Standard entropy: symmetric, self-targeting
                bce = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps))
            else:
                # Adaptive: hard-assign targets based on threshold
                target = (opacity > t).float().detach()
                bce = -(target * torch.log(opacity + eps) + (1 - target) * torch.log(1 - opacity + eps))
            bce_opacity_loss = args.bce_lambda * bce.mean()

        # Beta kernel shape regularization - encourage shapes toward 0 (hard flat disks)
        # Shape in range [0.001, 4.001]: low = hard disk, high = soft Gaussian cloud
        # Applied only in the last shape_iter iterations (or always if shape_iter == 0)
        shape_reg_loss = torch.tensor(0.0, device="cuda")
        shape_phase_active = args.shape_iter == 0 or iteration > (opt.iterations - args.shape_iter)
        # --w_lambda: error-guided version of lambda_shape. Mirrors --w_normal's logic.
        # Weight = exp(-γ · mean_per_pixel_MSE): low loss → full penalty (push flat),
        # high loss → relax (let the kernel stay soft where detail needs it).
        # Overrides the static lambda_shape path when active.
        _w_lambda_active = (args.w_lambda > 0.0 and shape_phase_active
                            and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0
                            and args.kernel in ["beta", "beta_scaled", "general"])
        if _w_lambda_active:
            mse_per_pixel_s = ((image - gt_image) ** 2).mean(dim=0, keepdim=True).detach()
            w_s = torch.exp(-args.w_lambda_gamma * mse_per_pixel_s).mean()
            # --method mixed/mixed_3d: mask β to TEXTURED surfels only. The
            # untextured half renders with --kernel2 (typically gaussian) and
            # doesn't read _shape at render time, so its β entries have no
            # photometric counterforce — without this mask the regularizer
            # would drag those (rendering-inert) values toward 0 every step.
            _shape_vals = gaussians.get_shape
            if (args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep") and hasattr(gaussians, '_is_textured')
                    and gaussians._is_textured.numel() == _shape_vals.numel()):
                _shape_vals = _shape_vals[gaussians._is_textured]
            if _shape_vals.numel() == 0:
                shape_reg_loss = torch.tensor(0.0, device="cuda")
            elif args.kernel in ["beta", "beta_scaled"]:
                # Push β toward 0 (flat disks), same direction as lambda_shape.
                shape_reg_loss = args.w_lambda * w_s * _shape_vals.mean()
            else:  # general kernel: push β toward 8 (flat/super-Gaussian box).
                shape_reg_loss = args.w_lambda * w_s * (8.0 - _shape_vals).mean()
        elif args.kernel in ["beta", "beta_scaled"] and args.lambda_shape > 0 and shape_phase_active and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
            # L1 penalty on shape values - pushes toward 0 (hard disks)
            shape_reg_loss = args.lambda_shape * gaussians.get_shape.mean()

        # --w_lambda_perpix: per-pixel error-guided shape reg.
        # Forward rasterizer emits beta_sum[pix] = Σ_i w_i · β_i (BETA_SUM_OFFSET).
        # Loss: λ · mean(w_r · beta_sum) where w_r = exp(-γ · MSE).
        # CUDA backward injects DIRECT dL/dβ_i += λ · w_r · w_i / N  (no α path).
        # Composes additively with --w_lambda (scalar) and --lambda_shape (static).
        w_lambda_perpix_loss = torch.tensor(0.0, device="cuda")
        if (args.w_lambda_perpix > 0.0 and shape_phase_active
                and args.kernel in ["beta", "beta_scaled", "general", "flex"]):
            beta_sum_map = render_pkg.get('render_beta_sum')
            if beta_sum_map is not None and beta_sum_map.numel() > 0:
                mse_per_pixel_pp = ((image - gt_image) ** 2).mean(dim=0, keepdim=True).detach()  # [1, H, W]
                w_r_pp = torch.exp(-args.w_lambda_perpix_gamma * mse_per_pixel_pp)
                w_lambda_perpix_loss = args.w_lambda_perpix * (w_r_pp * beta_sum_map).mean()

        # Flex kernel beta regularization - prevent runaway sharpening
        # Beta in range [0, inf): 0 = standard Gaussian, higher = sharper
        # Positive lambda pushes toward 0 (softer), negative pushes toward infinity (harder)
        flex_beta_reg_loss = torch.tensor(0.0, device="cuda")
        if args.kernel == "flex" and args.lambda_flex_beta != 0 and hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0:
            # L1 penalty on beta values - sign determines direction
            flex_beta_reg_loss = args.lambda_flex_beta * gaussians.get_flex_beta.mean()

        # General kernel beta regularization - push toward high beta (super-Gaussian/box)
        # Beta in range [2.0, 8.0]: 2.0 = standard Gaussian, 8.0 = super-Gaussian (box)
        # Positive lambda pushes toward 8 (hard), negative lambda pushes toward 2 (soft)
        # Modes: basic (constant), decay (linear decay), scaled (by RGB loss), scaled_decay (both)
        # Applied only in the last shape_iter iterations (or always if shape_iter == 0)
        general_beta_reg_loss = torch.tensor(0.0, device="cuda")
        # Skip static general-beta reg when --w_lambda is driving the penalty (above).
        if args.kernel == "general" and args.lambda_shape != 0 and shape_phase_active and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0 and not _w_lambda_active:
            effective_lambda = args.lambda_shape

            # Apply decay if requested
            if args.genreg in ["decay", "scaled_decay"]:
                decay_factor = max(0.0, 1.0 - iteration / opt.iterations)
                effective_lambda = effective_lambda * decay_factor

            # Apply RGB loss scaling if requested
            if args.genreg in ["scaled", "scaled_decay"]:
                loss_scale = loss.detach().clamp(min=1e-4)
                effective_lambda = effective_lambda * loss_scale

            # L1 penalty on (8 - β) - positive lambda pushes toward 8 (hard)
            general_beta_reg_loss = effective_lambda * (8.0 - gaussians.get_shape).mean()

        # L1 regularization on hashgrid embeddings - encourage sparsity to remove grey haze
        l1_hash_loss = torch.tensor(0.0, device="cuda")
        if args.l1_hash > 0 and ingp is not None and hasattr(ingp, 'hash_encoding') and ingp.hash_encoding is not None:
            l1_hash_loss = args.l1_hash * torch.abs(ingp.hash_encoding.embeddings).mean()

        # L1 regularization on higher-order SH (features_rest) — keep SH low-frequency,
        # force hashgrid to learn high-frequency spatial detail
        l1_sh_rest_loss = torch.tensor(0.0, device="cuda")
        if args.l1_sh_rest > 0:
            l1_sh_rest_loss = args.l1_sh_rest * torch.abs(gaussians._features_rest).mean()

        # GaussianSpa ADMM sparsification loss (only on z/u update iterations, matching paper)
        gspa_loss = torch.tensor(0.0, device="cuda")
        if args.gspa and optimizing_spa is not None and iteration > args.gspa_start_iter and iteration <= args.gspa_stop_iter and iteration % args.gspa_interval == 0:
            gspa_loss = optimizing_spa.compute_spa_loss()

        # Weighted overdraw regularization: error-guided relaxation
        # w(r) = exp(-gamma * MSE(r)), penalty relaxed where RGB error is high
        w_overdraw_loss = torch.tensor(0.0, device="cuda")
        if args.w_overdraw_reg > 0.0:
            od_map = render_pkg.get('render_overdraw')
            if od_map is not None and od_map.numel() > 0:
                # Per-pixel MSE (detached — no gradients through the weight)
                mse_per_pixel = ((image - gt_image) ** 2).mean(dim=0, keepdim=True).detach()  # [1, H, W]
                w_r = torch.exp(-args.w_overdraw_gamma * mse_per_pixel)  # [1, H, W]
                w_overdraw_loss = args.w_overdraw_reg * (w_r * od_map).mean()

        # Weight-squared regularization: penalize (1 - sum(w_i^2)) per pixel
        # --weight_reg: CUDA backward handles gradient with fixed lambda
        # --w_weight_reg: dynamically adjusts CUDA lambda based on mean reconstruction error
        #   High error → low lambda (relax regularization), low error → full lambda
        if args.w_weight_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat"]:
            avg_mse = ((image - gt_image) ** 2).mean().detach().item()
            effective_lambda = args.w_weight_reg * float(np.exp(-args.w_weight_gamma * avg_mse))
            if args.method == "3D_SH_concat":
                from diff_surfel_3D_sh_concat import set_weight_reg_lambda
            elif args.method == "3D_SH_32":
                from diff_surfel_3D_sh_32 import set_weight_reg_lambda
            else:
                set_weight_reg_lambda = _SHRES_SETTER_MOD.set_weight_reg_lambda
            set_weight_reg_lambda(effective_lambda)

        # Unbiased Depth: convergence loss (Peng et al.). Only added when --unbiased
        # is on AND iteration > unbiased_iter (paper default: 10000). The 'converge'
        # tensor is exposed by gaussian_renderer when allmap has the 19th channel
        # (i.e., when the unbiased rasterizer is in use).
        converge_loss = torch.tensor(0.0, device="cuda")
        if args.unbiased and iteration > args.unbiased_iter:
            converge_t = render_pkg.get('converge')
            if converge_t is not None:
                converge_loss = args.lambda_converge * converge_t.mean()

        # --trunc_cliff_reg: per-pixel cliff-sharpness penalty under truncation.
        # T_final = 1 - rend_alpha is the transmittance AT the truncation
        # barrier; penalizing it over CROSSED pixels (rend_alpha >= 0.5, which
        # under an active exit_T <= 0.5 identifies exactly the rays that hit
        # the barrier) drives the crossing fragment's opacity toward 1 — the
        # deterministic, color-free version of the back-plate noise pressure.
        # Side benefit: pushes T_final away from the threshold, widening every
        # ray's margin to the inclusion-flip discontinuity. Mask is detached
        # (non-differentiable gate, same idiom as --random_mesh's m_hit).
        trunc_cliff_loss = torch.tensor(0.0, device="cuda")
        if (getattr(args, 'trunc', False) and getattr(args, 'trunc_cliff_reg', 0.0) > 0.0
                and iteration > int(args.trunc_ramp_start)):
            _ra = render_pkg['rend_alpha']
            _m = (_ra.detach() >= 0.5).float()
            if _m.sum() > 0:
                trunc_cliff_loss = args.trunc_cliff_reg * (((1.0 - _ra) * _m).sum() / _m.sum())

        # loss
        # --probe_nosh_lambda (Texture-GS Eq.16 noSH loss, adapted): render with
        # the SV base killed (tex_only) and supervise directly against FULL GT.
        # Forces the texture to be a primary appearance carrier instead of a
        # scraps-residual — the loss-based answer to the SV race (their ablation:
        # dropping it collapses texture-alone quality 27.6->25.1 as appearance
        # leaks back into per-Gaussian attrs). One extra render per iter.
        probe_nosh_loss = 0.0
        if (getattr(args, 'probe_nosh_lambda', 0.0) > 0.0
                and getattr(ingp, 'is_proberes_mode', False)
                and not getattr(ingp, 'hashgrid_disabled', False)):
            _nosh_pkg = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                               beta=beta, iteration=iteration, cfg=cfg_model,
                               decompose_mode='tex_only', is_training=True,
                               lowpass=args.lowpass,
                               max_intersections_per_pixel=args.max_intersections_per_pixel)
            _nosh_img = _nosh_pkg['render']
            probe_nosh_loss = args.probe_nosh_lambda * (
                0.8 * l1_loss(_nosh_img, gt_image) + 0.2 * (1.0 - ssim(_nosh_img, gt_image)))

        # --ppisp: the module's own physically-motivated priors (exposure mean ≈ 0
        # to break the SH↔exposure ambiguity, vignetting center near the optical
        # center, cross-channel similarity, non-positive falloff, color mean ≈ 0).
        # Without the mean terms the ISP and the radiance field drift together.
        ppisp_reg_loss = torch.tensor(0.0, device="cuda")
        if ppisp is not None:
            ppisp_reg_loss = ppisp.get_regularization_loss()

        total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_reg_loss + scout_loss + mcmc_opacity_reg + mcmc_scale_reg + adaptive_cat_reg_loss + adaptive_zero_reg_loss + adaptive_gate_reg_loss + bce_opacity_loss + shape_reg_loss + flex_beta_reg_loss + general_beta_reg_loss + l1_hash_loss + l1_sh_rest_loss + gspa_loss + w_overdraw_loss + sv_l1_loss + decomp_sh_loss + decomp_tex_loss + converge_loss + w_lambda_perpix_loss + deform_reg_loss + lpips_train_loss + trunc_cliff_loss + probe_nosh_loss + ppisp_reg_loss + ppisp_overflow_loss

        # --minimc per-step error accumulation: BENCHED.
        # Replaced by the full-view sweep inside `minimc_sweep_and_relocate`,
        # which is dispatched once per `--minimc_relocate_interval`.

        # DEBUG: print loss components before backward
        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_bwd_start = time.time()

        # --texture_dropout_bw: arm the dropout AFTER the forward (image already
        # rendered full) so ONLY the backward kernel sees it. Seed = iteration
        # (dropped set rotates). Disarmed right after backward, with the fwd variant.
        if _use_tex_dropout_bw:
            _set_tex_dropout(args.texture_dropout_bw, iteration)

        total_loss.backward()

        # --wsr: occlusion grads arrive OUTSIDE autograd (device-global
        # accumulator filled by the WSR backward). Chain the sigmoid derivative
        # and assign .grad NOW, before any further render this iteration can
        # clobber pc._wsr_render_state / the device-global pointers.
        if getattr(args, 'wsr', False):
            _wsr_st = getattr(gaussians, '_wsr_render_state', None)
            if _wsr_st is not None and gaussians._wsr_occ.numel() > 0:
                _occ_act, _occ_grad = _wsr_st[0], _wsr_st[1]
                _g_raw = (_occ_grad * _occ_act * (1.0 - _occ_act)).view(-1, 1)
                if gaussians._wsr_occ.grad is None:
                    gaussians._wsr_occ.grad = _g_raw.clone()
                else:
                    gaussians._wsr_occ.grad += _g_raw
                gaussians._wsr_render_state = None

        # --random_mesh_depth: clear the per-pixel occluder ONLY NOW — after
        # backward. The rasterizer backward re-walks the fragment list to
        # reconstruct the alpha/T recurrence and MUST see the same wall the
        # forward rendered with; clearing between fwd and bwd makes the
        # recurrence inconsistent → garbage gradients (hash grad ~1e19,
        # NaN MLP — bitten at iter 5000 on the first depthnoise run).
        # Clearing here keeps downstream renders this iter (fastgs scoring,
        # debug/eval, densify metric maps) occluder-free.
        if _use_depth_wall:
            from diff_surfel_3D_sh_res import clear_occluder_depth
            clear_occluder_depth()

        # NaN TRIPWIRE (GEStex frontmost debugging): catch the FIRST bad iteration —
        # a non-finite loss or a non-finite per-Gauss gradient — save the model state
        # for offline single-step postmortem, print the culprit tensors, and abort.
        if args.is_gestex and iteration >= 15000:
            _nan_hit = not torch.isfinite(total_loss)
            _bad = []
            if not _nan_hit:
                for _nm, _p in (('xyz', gaussians._xyz), ('opacity', gaussians._opacity),
                                ('scaling', gaussians._scaling), ('rotation', gaussians._rotation),
                                ('f_dc', gaussians._features_dc), ('shape', getattr(gaussians, '_shape', None))):
                    if _p is not None and _p.grad is not None and not torch.isfinite(_p.grad).all():
                        _bad.append(_nm)
                _nan_hit = len(_bad) > 0
            if _nan_hit:
                _dump = os.path.join(args.model_path, f"nan_dump_{iteration}")
                os.makedirs(_dump, exist_ok=True)
                gaussians.save_ply(os.path.join(_dump, "point_cloud.ply"))
                torch.save({'iteration': iteration,
                            'viewpoint': getattr(viewpoint_cam, 'image_name', '?'),
                            'bad_tensors': _bad,
                            'loss': float(total_loss.detach().cpu()) if torch.isfinite(total_loss) else float('nan'),
                            'ingp': (ingp.state_dict() if ingp is not None else None)},
                           os.path.join(_dump, "state.pt"))
                _msg = (f"[NaN TRIPWIRE] iter={iteration} view={getattr(viewpoint_cam,'image_name','?')} "
                        f"loss_finite={torch.isfinite(total_loss).item()} bad_grads={_bad} -> dumped {_dump}")
                tqdm.write(_msg)
                raise RuntimeError(_msg)

        # Texture-query dropout: disarm now that this iteration's forward+backward consumed the
        # mask, so every subsequent render (eval/test PSNR, debug, save, network GUI) is full-query.
        if _use_tex_dropout or _use_tex_dropout_bw:
            _set_tex_dropout(0.0, 0)

        # --deform: clear the canonical-xyz device pointer now that this step's
        # forward+backward are done. Prevents a stale pointer (if densification
        # reallocates _xyz before the next training render) from being read by any
        # intervening eval/debug render. Eval renders pass deform=None ⇒ means3D is
        # already canonical, and null falls back to means3D ⇒ correct canonical hash.
        if deform_raster_mod is not None:
            deform_raster_mod.set_canonical_xyz(None)

        # `--method res_3d`: freeze the SV/SH params on the 2D residual-carrier
        # rows (where `_is_textured = True`). Belt-and-suspenders alongside the
        # value zeroing at split-time + the renderer's `colors_precomp` mask:
        # ensures no gradient leak through some yet-unforeseen code path can
        # drift the 2D side's SV away from 0. The 3D-carrier rows (untextured)
        # keep training normally — same `nn.Parameter` storage, just per-row
        # grad mask after backward.
        # IMPORTANT: gate on `(~_is_textured).any()` (= "at least one untex
        # row exists" = split has fired), NOT on `.any()` — `_is_textured` is
        # initialized to all-True at startup, so `.any()` is trivially true
        # pre-split and would freeze SV gradients from iter 1, starving the
        # SV from ever learning before the split.
        # `--method res_3d` ONLY (NOT res_3d_paired): freeze SV/SH grads on
        # textured carriers post-split so they stay residual-only. Paired
        # mode WANTS SV to train on tex rows (full mixed_3d-style capacity).
        if args.method == "res_3d" \
                and hasattr(gaussians, '_is_textured') \
                and gaussians._is_textured.numel() == gaussians.get_xyz.shape[0] \
                and bool((~gaussians._is_textured).any()):
            with torch.no_grad():
                _tex_idx = gaussians._is_textured
                for _sv_name in ('_sv_sites', '_sv_colors', '_sv_tau', '_sv_dc',
                                  '_features_dc', '_features_rest'):
                    _sv = getattr(gaussians, _sv_name, None)
                    if _sv is None or not hasattr(_sv, 'grad') or _sv.grad is None:
                        continue
                    # Broadcast tex mask across the trailing dims of the grad.
                    _m = _tex_idx
                    while _m.dim() < _sv.grad.dim():
                        _m = _m.unsqueeze(-1)
                    _sv.grad[_m.expand_as(_sv.grad)] = 0.0

        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_bwd_end = time.time()

        # Apply MLP gradients for 3D_direct_fused mode
        # MLP weights are in CUDA constant memory, gradients computed in CUDA backward
        # Skip when freeze_mlp is active (no weight gradients computed)
        if ingp is not None and hasattr(ingp, 'is_3D_direct_fused_mode') and ingp.is_3D_direct_fused_mode and not ingp.freeze_mlp \
                and not getattr(ingp, 'is_proberes_mode', False):  # proberes: no in-kernel MLP; probe/texture grads flow via autograd
            # Import from appropriate library based on mode
            if getattr(ingp, 'is_gestex_joint', False) and not getattr(ingp, 'is_gestex_sortfree', False):
                # `--ges_no_bake` joint stage: the cascade (diff_surfel_gestex) renders
                # textured surfels via live hash+MLP, so its backward writes the MLP weight
                # grads into the gestex module's buffer (NOT diff_surfel_3D_sh_res's). Pull
                # from there so the hash/MLP keeps training. (Sort-free/bake path has LR=0
                # and never hits this — is_gestex_sortfree=True there.)
                from diff_surfel_gestex import get_mlp_grads
            elif getattr(ingp, 'is_gestex_mode', False) and not getattr(ingp, 'is_gestex_joint', False):
                # GEStex explore+harden (0-20k): renders through the ISOLATED
                # diff_surfel_3D_sh_res_harden clone (first-intersection sort), so the MLP
                # weight grads live in THAT module's buffer. Reading the base module here
                # silently freezes the MLP at init (getters can't be mirror-patched).
                from diff_surfel_3D_sh_res_harden import get_mlp_grads
            elif hasattr(ingp, 'is_3D_SH_filmres_mode') and ingp.is_3D_SH_filmres_mode:
                # `--method 3D_SH_filmres`: MLP weight grads live in the filmres fork's
                # autograd buffer (must precede the is_3D_SH_res_mode branch — filmres is
                # also is_3D_SH_res_mode=True).
                from diff_surfel_3D_sh_filmres import get_mlp_grads
            elif hasattr(ingp, 'is_3D_SH_concat_mode') and ingp.is_3D_SH_concat_mode:
                # `--method 3D_SH_concat`: MLP weight grads live in the concat fork's buffer.
                from diff_surfel_3D_sh_concat import get_mlp_grads
            elif hasattr(ingp, 'is_3D_SH_32_mode') and ingp.is_3D_SH_32_mode:
                from diff_surfel_3D_sh_32 import get_mlp_grads
            elif hasattr(ingp, 'is_mixed_3d_mode') and ingp.is_mixed_3d_mode:
                # `--method mixed_3d`: grads live in the mixed_3d module's buffer.
                from diff_surfel_mixed_3d import get_mlp_grads
            elif hasattr(ingp, 'is_mixed_mode') and ingp.is_mixed_mode:
                # `--method mixed`: backward writes MLP grads into the mixed module's
                # autograd Function buffer, NOT diff_surfel_3D_sh_res's. The setter
                # mirror only covers weight uploads, not grad pulls — read directly.
                from diff_surfel_mixed import get_mlp_grads
            elif (hasattr(ingp, 'is_3D_SH_res_mode') and ingp.is_3D_SH_res_mode) or \
               (hasattr(ingp, 'is_3D_SH_cat_mode') and ingp.is_3D_SH_cat_mode):
                from diff_surfel_3D_sh_res import get_mlp_grads
            elif hasattr(ingp, 'is_3D_direct_sh_tc_mode') and ingp.is_3D_direct_sh_tc_mode:
                from diff_surfel_3D_sh import get_mlp_grads
            elif hasattr(ingp, 'is_3D_direct_tc_mode') and ingp.is_3D_direct_tc_mode:
                from diff_surfel_3D_tc import get_mlp_grads
            elif hasattr(ingp, 'is_3D_direct_fp16_mode') and ingp.is_3D_direct_fp16_mode:
                from diff_surfel_3D_16 import get_mlp_grads
            elif hasattr(ingp, 'is_3D_direct_lean_mode') and ingp.is_3D_direct_lean_mode:
                from diff_surfel_3D import get_mlp_grads
            else:
                from diff_surfel_rasterization import get_mlp_grads
            mlp_grads = get_mlp_grads()
            if mlp_grads is not None:
                grad_W1, grad_W2, grad_W3 = mlp_grads
                # Trim WMMA-padded gradients back to PyTorch MLP dimensions
                # 3D_SH_res: all [16,16] — no trimming needed (PyTorch MLP matches CUDA)
                # 3D_SH_32: all [32,32] — no trimming needed (PyTorch MLP matches CUDA)
                if hasattr(ingp, 'is_3D_direct_sh_tc_mode') and ingp.is_3D_direct_sh_tc_mode:
                    grad_W1 = grad_W1[:, :25]  # [32, 32] -> [32, 25]
                    # W3 is [48, 32] — no trimming needed (all 48 SH coefficients used)
                elif hasattr(ingp, 'is_3D_direct_tc_mode') and ingp.is_3D_direct_tc_mode:
                    grad_W1 = grad_W1[:, :41]  # [32, 48] -> [32, 41]
                    grad_W3 = grad_W3[:3, :]   # [16, 32] -> [3, 32]
                mlp = ingp.mlp_fused
                # Accumulate gradients (in case there are multiple backward passes)
                # MLP layout: Linear(41, 32, bias=False), Linear(32, 32, bias=False), Linear(32, 3, bias=False)
                if mlp[0].weight.grad is None:
                    mlp[0].weight.grad = grad_W1.clone()
                else:
                    mlp[0].weight.grad += grad_W1
                if mlp[2].weight.grad is None:
                    mlp[2].weight.grad = grad_W2.clone()
                else:
                    mlp[2].weight.grad += grad_W2
                if mlp[4].weight.grad is None:
                    mlp[4].weight.grad = grad_W3.clone()
                else:
                    mlp[4].weight.grad += grad_W3


        # Total variation regularization on hashgrid - penalizes uniform regions while preserving edges
        # Must be called after backward() and before optimizer.step() as it directly modifies gradients
        if args.tv_hash > 0 and ingp is not None and hasattr(ingp, 'hash_encoding') and ingp.hash_encoding is not None:
            ingp.hash_encoding.grad_total_variation(weight=args.tv_hash)


        iter_end.record()
        
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_for_log
            # Unbiased Depth: track convergence loss + raw Converge.mean() so we can
            # diagnose magnitude (paper λ=7 is for vanilla 2DGS Gaussian + DTU/T&T scale;
            # beta-scaled + mip-360 may need a different λ).
            if args.unbiased:
                ema_converge_for_log = 0.4 * converge_loss.item() + 0.6 * ema_converge_for_log
                _converge_t_log = render_pkg.get('converge')
                if _converge_t_log is not None:
                    ema_converge_raw_for_log = 0.4 * _converge_t_log.mean().item() + 0.6 * ema_converge_raw_for_log
            
            # Track MCMC regularization losses
            if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
                mcmc_total = mcmc_opacity_reg.item() + mcmc_scale_reg.item()
                ema_mcmc_loss_for_log = 0.4 * mcmc_total + 0.6 * ema_mcmc_loss_for_log

            if iteration % 10 == 0:
                # For MCMC / MiniMC, show alive Gaussians (opacity > 0.005) instead of total
                if getattr(args, 'is_gestex', False):
                    # `--method GEStex`: show textured surfels vs untextured 3D Gaussians.
                    _n = len(gaussians.get_xyz)
                    _it = getattr(gaussians, '_is_textured', None)
                    if _it is not None and _it.numel() == _n and _n > 0:
                        _ntex = int(_it.sum().item())
                        points_str = f"tex={_ntex}/gs={_n - _ntex}"
                    else:
                        points_str = f"tex={_n}/gs=0"
                elif args.mcmc or args.mcmc_deficit or args.mcmc_fps or args.minimc:
                    n_alive = (gaussians.get_opacity > 0.005).sum().item()
                    points_str = f"{int(n_alive)}/{len(gaussians.get_xyz)}"
                elif args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "res_3d", "res_3d_paired", "res_3d_double"):
                    # Total points / % textured (rest are untextured: for
                    # mixed_* simple-2DGS, for res_3d the 3D EWA SV-carriers).
                    _n = len(gaussians.get_xyz)
                    _it = getattr(gaussians, '_is_textured', None)
                    if _it is not None and _it.numel() == _n and _n > 0:
                        _pct = 100.0 * float(_it.sum().item()) / _n
                        points_str = f"{_n}/{_pct:.0f}%tex"
                    else:
                        points_str = f"{_n}"  # pre-split (all textured)
                else:
                    points_str = f"{len(gaussians.get_xyz)}"

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": points_str,
                }
                # Add opacity/scale regularization to progress bar if active
                if args.opacity_reg > 0:
                    loss_dict["OpR"] = f"{mcmc_opacity_reg.item():.{5}f}"
                if args.scale_reg > 0:
                    loss_dict["ScR"] = f"{mcmc_scale_reg.item():.{5}f}"
                # Add BCE phase indicator to progress bar
                if bce_phase_active:
                    t = getattr(args, '_bce_adaptive_threshold', 0.5)
                    loss_dict["BCE"] = f"t={t:.2f}"
                # Add adaptive_cat metrics to progress bar
                if args.method == "adaptive_cat" and hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight.numel() > 0:
                    weights = torch.sigmoid(gaussians._adaptive_cat_weight)
                    pct_high = (weights > 0.9).float().mean().item() * 100  # Gaussian-dominant
                    pct_low = (weights < 0.1).float().mean().item() * 100   # Hash-dominant
                    loss_dict["G>0.9"] = f"{pct_high:.0f}%"
                    loss_dict["H<0.1"] = f"{pct_low:.0f}%"
                    if adaptive_cat_reg_loss.item() > 0:
                        loss_dict["AdR"] = f"{adaptive_cat_reg_loss.item():.{5}f}"
                # Add adaptive_zero metrics to progress bar
                if args.method == "adaptive_zero" and hasattr(gaussians, '_adaptive_zero_weight') and gaussians._adaptive_zero_weight.numel() > 0:
                    weights = torch.sigmoid(gaussians._adaptive_zero_weight)
                    pct_zero = (weights < 0.1).float().mean().item() * 100  # Using zeros (fast)
                    pct_hash = (weights > 0.9).float().mean().item() * 100  # Using hash (slow)
                    loss_dict["Z<0.1"] = f"{pct_zero:.0f}%"
                    loss_dict["H>0.9"] = f"{pct_hash:.0f}%"
                    if adaptive_zero_reg_loss.item() > 0:
                        loss_dict["AzR"] = f"{adaptive_zero_reg_loss.item():.{5}f}"
                # Add adaptive_gate metrics to progress bar
                if args.method == "adaptive_gate" and hasattr(gaussians, '_gate_logits') and gaussians._gate_logits.numel() > 0:
                    gate_prob = torch.sigmoid(gaussians._gate_logits)
                    pct_open = (gate_prob > 0.5).float().mean().item() * 100  # Using hash (gate open)
                    avg_prob = gate_prob.mean().item()
                    loss_dict["Gate"] = f"{pct_open:.0f}%"
                    loss_dict["AvgP"] = f"{avg_prob:.2f}"
                    if adaptive_gate_reg_loss.item() > 0:
                        loss_dict["Spr"] = f"{adaptive_gate_reg_loss.item():.{5}f}"
                # Add BCE opacity loss to progress bar if active
                if args.bce and bce_opacity_loss.item() > 0:
                    t = getattr(args, '_bce_adaptive_threshold', 0.5)
                    loss_dict["BCE"] = f"{bce_opacity_loss.item():.5f}(t={t:.2f})"
                # Add beta kernel shape stats to progress bar (always show when using beta/beta_scaled kernel)
                if args.kernel in ["beta", "beta_scaled"]:
                    loss_dict["ShR"] = f"{shape_reg_loss.item():.5f}"
                    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                        shape_vals = gaussians.get_shape
                        loss_dict["Shp"] = f"{shape_vals.mean().item():.2f}"
                # Add flex kernel beta stats to progress bar
                if args.kernel == "flex":
                    loss_dict["FxR"] = f"{flex_beta_reg_loss.item():.5f}"
                    if hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0:
                        beta_vals = gaussians.get_flex_beta
                        loss_dict["Fxβ"] = f"{beta_vals.mean().item():.2f}"
                # Add general kernel beta stats to progress bar
                if args.w_normal > 0.0 and normal_loss.item() > 0:
                    loss_dict["wN"] = f"{normal_loss.item():.5f}"
                if args.w_lambda > 0.0 and shape_reg_loss.item() != 0.0:
                    loss_dict["wλ"] = f"{shape_reg_loss.item():.5f}"
                if args.overdraw_reg > 0.0:
                    od_map = render_pkg.get('render_overdraw')
                    if od_map is not None:
                        loss_dict["OD"] = f"{od_map.mean().item():.1f}"
                if args.w_overdraw_reg > 0.0:
                    loss_dict["wOD"] = f"{w_overdraw_loss.item():.5f}"
                if args.w_lambda_perpix > 0.0:
                    loss_dict["wλp"] = f"{w_lambda_perpix_loss.item():.5f}"
                if args.w_weight_reg > 0.0:
                    avg_mse_disp = ((image - gt_image) ** 2).mean().item()
                    eff_lambda = args.w_weight_reg * float(np.exp(-args.w_weight_gamma * avg_mse_disp))
                    loss_dict["wwR"] = f"{eff_lambda:.4f}"
                if args.kernel == "general":
                    loss_dict["GnR"] = f"{general_beta_reg_loss.item():.5f}"
                    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                        beta_vals = gaussians.get_shape
                        loss_dict["Gnβ"] = f"{beta_vals.mean().item():.2f}"
                if args.gspa and optimizing_spa is not None and iteration > args.gspa_start_iter and iteration <= args.gspa_stop_iter:
                    loss_dict["GSPA"] = f"{gspa_loss.item():.5f}"
                # Unbiased: show λ·Converge.mean() and the raw Converge.mean() (post-iter > unbiased_iter)
                if args.unbiased:
                    loss_dict["Cv"] = f"{ema_converge_for_log:.4f}"
                    loss_dict["Cv_raw"] = f"{ema_converge_raw_for_log:.6f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/rgb_loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/mask_loss', ema_mask_for_log, iteration)
                if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
                    tb_writer.add_scalar('train_loss_patches/mcmc_reg_loss', ema_mcmc_loss_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/mcmc_opacity_reg', mcmc_opacity_reg.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/mcmc_scale_reg', mcmc_scale_reg.item(), iteration)
                if args.method == "adaptive_cat" and hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight.numel() > 0:
                    weights = torch.sigmoid(gaussians._adaptive_cat_weight)
                    mean_weight = weights.mean().item()
                    pct_high = (weights > 0.9).float().mean().item() * 100  # Gaussian-dominant
                    pct_low = (weights < 0.1).float().mean().item() * 100   # Hash-dominant
                    tb_writer.add_scalar('adaptive_cat/mean_weight', mean_weight, iteration)
                    tb_writer.add_scalar('adaptive_cat/pct_gaussian_above_0.9', pct_high, iteration)
                    tb_writer.add_scalar('adaptive_cat/pct_hash_below_0.1', pct_low, iteration)
                    tb_writer.add_scalar('adaptive_cat/reg_loss', adaptive_cat_reg_loss.item(), iteration)
                if args.bce:
                    tb_writer.add_scalar('train_loss_patches/bce_opacity_loss', bce_opacity_loss.item(), iteration)
                # Log beta kernel shape stats
                if args.kernel in ["beta", "beta_scaled"] and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                    shape_vals = gaussians.get_shape
                    tb_writer.add_scalar('beta_kernel/shape_mean', shape_vals.mean().item(), iteration)
                    tb_writer.add_scalar('beta_kernel/shape_min', shape_vals.min().item(), iteration)
                    tb_writer.add_scalar('beta_kernel/shape_max', shape_vals.max().item(), iteration)
                    tb_writer.add_scalar('beta_kernel/shape_reg_loss', shape_reg_loss.item(), iteration)
                    # Track shape distribution: percent of Gaussians with hard disk shape (< 0.5)
                    pct_hard = (shape_vals < 0.5).float().mean().item() * 100
                    tb_writer.add_scalar('beta_kernel/pct_hard_disk', pct_hard, iteration)
                # Log flex kernel beta stats
                if args.kernel == "flex" and hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0:
                    beta_vals = gaussians.get_flex_beta
                    tb_writer.add_scalar('flex_kernel/beta_mean', beta_vals.mean().item(), iteration)
                    tb_writer.add_scalar('flex_kernel/beta_min', beta_vals.min().item(), iteration)
                    tb_writer.add_scalar('flex_kernel/beta_max', beta_vals.max().item(), iteration)
                    tb_writer.add_scalar('flex_kernel/beta_reg_loss', flex_beta_reg_loss.item(), iteration)
                    # Track beta distribution: percent with high sharpening (beta > 1)
                    pct_sharp = (beta_vals > 1.0).float().mean().item() * 100
                    tb_writer.add_scalar('flex_kernel/pct_sharp_beta_gt_1', pct_sharp, iteration)
                # Log general kernel beta stats
                if args.kernel == "general" and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                    beta_vals = gaussians.get_shape  # beta in [2.0, 8.0]
                    tb_writer.add_scalar('general_kernel/beta_mean', beta_vals.mean().item(), iteration)
                    tb_writer.add_scalar('general_kernel/beta_min', beta_vals.min().item(), iteration)
                    tb_writer.add_scalar('general_kernel/beta_max', beta_vals.max().item(), iteration)
                    tb_writer.add_scalar('general_kernel/beta_reg_loss', general_beta_reg_loss.item(), iteration)
                    # Track beta distribution: percent with high beta (super-Gaussian, > 5.0)
                    pct_super = (beta_vals > 5.0).float().mean().item() * 100
                    pct_gaussian = (beta_vals < 3.0).float().mean().item() * 100
                    tb_writer.add_scalar('general_kernel/pct_super_gaussian_gt_5', pct_super, iteration)
                    tb_writer.add_scalar('general_kernel/pct_gaussian_lt_3', pct_gaussian, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), \
                ingp_model=ingp, beta = beta, args = args, cfg_model = cfg_model, test_psnr = test_psnr, train_psnr = train_psnr, iter_list = iter_list, skybox_model = skybox,
                background_mode = background_mode, bg_hashgrid_model = bg_hashgrid,
                ppisp_apply = (_ppisp_apply if ppisp is not None else None))

            # --patience: convergence early-stop. Only triggers on iters in the
            # patience eval grid (>= patience_start_iter, on the eval interval).
            # test_psnr was just appended by training_report if iteration was
            # in testing_iterations. We compare the latest entry to the best so far.
            if (args.patience > 0
                    and iteration >= args.patience_start_iter
                    and iteration in testing_iterations
                    and len(test_psnr) > _patience_evals_seen):
                _patience_evals_seen = len(test_psnr)
                latest = test_psnr[-1]
                if latest > _patience_best_psnr + args.patience_min_delta:
                    _patience_best_psnr = latest
                    _patience_best_iter = iteration
                    _patience_no_improve = 0
                    print(f"[PATIENCE] iter {iteration}: new best PSNR {latest:.4f} dB (counter reset)")
                else:
                    _patience_no_improve += 1
                    print(f"[PATIENCE] iter {iteration}: PSNR {latest:.4f} dB "
                          f"(no improvement, counter={_patience_no_improve}/{args.patience}; "
                          f"best={_patience_best_psnr:.4f} @ iter {_patience_best_iter})")
                    if _patience_no_improve >= args.patience:
                        print(f"[PATIENCE] Early stop at iter {iteration}: "
                              f"{args.patience} consecutive evals without improvement. "
                              f"Best PSNR {_patience_best_psnr:.4f} dB at iter {_patience_best_iter}. "
                              f"All evals are logged in test_metrics.txt.")
                        break

            # Print beta kernel stats every 1000 iterations
            if args.kernel in ["beta", "beta_scaled"] and iteration % 1000 == 0 and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                shape_vals = gaussians.get_shape
                shape_mean = shape_vals.mean().item()
                shape_std = shape_vals.std().item()
                shape_min = shape_vals.min().item()
                shape_max = shape_vals.max().item()
                pct_hard = (shape_vals < 0.5).float().mean().item() * 100
                pct_soft = (shape_vals > 2.0).float().mean().item() * 100
                has_grad = gaussians._shape.requires_grad
                kernel_name = "Beta" if args.kernel == "beta" else "Beta Scaled"
                print(f"\n[ITER {iteration}] {kernel_name} Kernel Stats: shape={shape_mean:.3f}±{shape_std:.3f} (min={shape_min:.3f}, max={shape_max:.3f}) [requires_grad={has_grad}]")
                print(f"  Hard disks (<0.5): {pct_hard:.1f}% | Soft clouds (>2.0): {pct_soft:.1f}% | Reg loss: {shape_reg_loss.item():.6f}")

            # Print flex kernel stats every 1000 iterations
            if args.kernel == "flex" and iteration % 1000 == 0 and hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0:
                beta_vals = gaussians.get_flex_beta
                beta_mean = beta_vals.mean().item()
                beta_std = beta_vals.std().item()
                beta_min = beta_vals.min().item()
                beta_max = beta_vals.max().item()
                pct_standard = (beta_vals < 0.1).float().mean().item() * 100  # Nearly standard Gaussian
                pct_sharp = (beta_vals > 1.0).float().mean().item() * 100     # Significantly sharpened
                print(f"\n[ITER {iteration}] Flex Kernel Stats: beta={beta_mean:.3f}±{beta_std:.3f} (min={beta_min:.3f}, max={beta_max:.3f})")
                print(f"  Standard (<0.1): {pct_standard:.1f}% | Sharp (>1.0): {pct_sharp:.1f}% | Reg loss: {flex_beta_reg_loss.item():.2e}")

            # Print general kernel stats every 1000 iterations
            if args.kernel == "general" and iteration % 1000 == 0 and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                beta_vals = gaussians.get_shape  # beta in [2.0, 8.0]
                beta_mean = beta_vals.mean().item()
                beta_std = beta_vals.std().item()
                beta_min = beta_vals.min().item()
                beta_max = beta_vals.max().item()
                pct_gaussian = (beta_vals < 3.0).float().mean().item() * 100  # Near standard Gaussian (β≈2)
                pct_super = (beta_vals > 5.0).float().mean().item() * 100     # Super-Gaussian (β>5)
                print(f"\n[ITER {iteration}] General Kernel Stats: β={beta_mean:.3f}±{beta_std:.3f} (min={beta_min:.3f}, max={beta_max:.3f})")
                print(f"  Gaussian (<3): {pct_gaussian:.1f}% | Super-Gaussian (>5): {pct_super:.1f}% | Reg loss: {general_beta_reg_loss.item():.2e}")

            if (iteration in saving_iterations):
                # For MCMC, skip saving at final iteration - will save after pruning dead Gaussians
                is_final_iter = (iteration == opt.iterations)
                is_mcmc = (args.mcmc or args.mcmc_deficit or args.mcmc_fps)
                if is_mcmc and is_final_iter:
                    print("\n[ITER {}] Skipping save (MCMC: will save after pruning dead Gaussians)".format(iteration))
                else:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                # --3rgs: export the refined camera poses next to the PLY.
                if pose_opt is not None:
                    from scene.camera_pose_opt import export_refined_poses
                    _pose_out = os.path.join(scene.model_path,
                                             f"point_cloud/iteration_{iteration}/refined_poses")
                    _stats = export_refined_poses(pose_opt, scene.getTrainCameras(),
                                                  pose_name_to_idx, _pose_out)
                    print(f"[3RGS] Exported refined poses for {_stats['n']} cams → {_pose_out} "
                          f"(|Δt| mean={_stats.get('trans_mean', 0):.2e}/max={_stats.get('trans_max', 0):.2e}, "
                          f"|Δrot6d| mean={_stats.get('rot6d_mean', 0):.2e}/max={_stats.get('rot6d_max', 0):.2e})")
                # --ppisp: save the ISP next to the PLY. Tiny (a few KB). Kept
                # OUT of the PLY on purpose — the exported splats are the
                # ISP-free ("canonical appearance") scene, which is what the
                # bake / .bitymi viewer should render; this file is only needed
                # to reproduce a specific training frame's appearance.
                if ppisp is not None:
                    _ppisp_out = os.path.join(
                        scene.model_path, f"point_cloud/iteration_{iteration}/ppisp.pt")
                    os.makedirs(os.path.dirname(_ppisp_out), exist_ok=True)
                    torch.save({"state_dict": ppisp.state_dict(),
                                "name_to_idx": ppisp_name_to_idx,
                                "crf_trained": bool(args.ppisp_crf),
                                "no_camera": bool(args.ppisp_no_camera)}, _ppisp_out)
                    with torch.no_grad():
                        _e = ppisp.exposure_params
                    print(f"[PPISP] Saved ISP → {_ppisp_out} "
                          f"(exposure range {_e.min().item():+.3f}..{_e.max().item():+.3f} stops)")
                if ingp is not None:
                    ingp.save_model(scene.model_path, iteration)
                if skybox is not None:
                    skybox.save_model(scene.model_path, iteration)
                if bg_hashgrid is not None:
                    bg_hashgrid.save_model(scene.model_path, iteration)

            # MCMC depth reinit: replacement of all Gaussians from depth maps
            # Triggers at mcmc_depth_reinit, then every reinit_interval until reinit_end
            _reinit_end = args.reinit_end if args.reinit_end >= 0 else max(0, opt.iterations - 15000)
            _do_reinit = False
            if args.mcmc_depth_reinit > 0 and (args.mcmc or args.mcmc_deficit or args.mcmc_fps):
                if iteration == args.mcmc_depth_reinit:
                    _do_reinit = True
                elif args.reinit_interval > 0 and iteration > args.mcmc_depth_reinit and iteration <= _reinit_end:
                    if (iteration - args.mcmc_depth_reinit) % args.reinit_interval == 0:
                        _do_reinit = True
            if _do_reinit:
                n_before = len(gaussians.get_xyz)
                torch.cuda.empty_cache()

                # Save pre-reinit renders (depth + RGB + alpha) for first training view
                output_path = os.path.join(scene.model_path, 'training_output')
                os.makedirs(output_path, exist_ok=True)
                with torch.no_grad():
                    dbg_cam = scene.getTrainCameras()[0]
                    dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                     iteration=iteration, cfg=cfg_model, ingp=ingp,
                                     record_transmittance=False, is_training=False)
                    dbg_img = torch.clamp(dbg_pkg['render'], 0.0, 1.0)
                    dbg_depth_mean = dbg_pkg['depth_expected']
                    dbg_depth_median = dbg_pkg['depth_median']
                    dbg_depth_max = dbg_pkg['depth_max_contributor']
                    dbg_alpha = dbg_pkg['rend_alpha']
                    save_img_u8(dbg_img.permute(1, 2, 0).cpu().numpy(),
                                os.path.join(output_path, f'{iteration}_pre_reinit_rgb.png'))
                    save_img_u8(dbg_alpha.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                os.path.join(output_path, f'{iteration}_pre_reinit_alpha.png'))
                    depth_np = dbg_depth_mean.squeeze().cpu().numpy()
                    save_img_u8(convert_gray_to_cmap(depth_np, map_mode='turbo', revert=False),
                                os.path.join(output_path, f'{iteration}_pre_reinit_depth_mean.png'))
                    depth_med_np = dbg_depth_median.squeeze().cpu().numpy()
                    save_img_u8(convert_gray_to_cmap(depth_med_np, map_mode='turbo', revert=False),
                                os.path.join(output_path, f'{iteration}_pre_reinit_depth_median.png'))
                    depth_max_np = dbg_depth_max.squeeze().cpu().numpy()
                    save_img_u8(convert_gray_to_cmap(depth_max_np, map_mode='turbo', revert=False),
                                os.path.join(output_path, f'{iteration}_pre_reinit_depth_maxcontrib.png'))
                    del dbg_pkg

                views = scene.getTrainCameras()
                all_reinit_data = []
                N_total = len(gaussians.get_xyz)
                for v in views:
                    with torch.no_grad():
                        rpkg = render(v, gaussians, pipe, background, beta=beta,
                                      iteration=iteration, cfg=cfg_model, ingp=ingp,
                                      record_transmittance=False, is_training=False)
                        gt_img = v.original_image.cuda()
                        # Use max-contributor depth if available, fall back to median
                        reinit_depth = rpkg.get('depth_max_contributor', None)
                        if reinit_depth is None or reinit_depth.numel() == 0:
                            reinit_depth = rpkg['depth_median']
                        data = gaussians.mini_depth_reinit(
                            [reinit_depth.detach()],
                            [rpkg['rend_alpha'].detach()],
                            [v],
                            gt_images=[gt_img],
                            normal_maps=[rpkg['rend_normal'].detach()],
                            num_total_views=len(views))
                        if data is not None:
                            all_reinit_data.append({k: t.cpu() for k, t in data.items()})
                        del rpkg
                    torch.cuda.empty_cache()
                if all_reinit_data:
                    merged = {
                        'xyz': torch.cat([d['xyz'] for d in all_reinit_data], dim=0).cuda(),
                        'colors': torch.cat([d['colors'] for d in all_reinit_data], dim=0).cuda() if 'colors' in all_reinit_data[0] else None,
                        'normals': torch.cat([d['normals'] for d in all_reinit_data], dim=0).cuda() if 'normals' in all_reinit_data[0] else None,
                    }
                    gaussians.reinitial_from_depth(merged)
                    gaussians.training_setup(opt)

                    # Reset INGP optimizer (stale Adam momentum from old Gaussian layout)
                    if ingp is not None:
                        ingp.training_setup(cfg_model.optim)
                        tqdm.write(f"[MCMC] Reset INGP optimizer after depth reinit")

                    # Save post-reinit renders
                    with torch.no_grad():
                        dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                         iteration=iteration, cfg=cfg_model, ingp=ingp,
                                         record_transmittance=False, is_training=False)
                        dbg_img = torch.clamp(dbg_pkg['render'], 0.0, 1.0)
                        dbg_depth_mean = dbg_pkg['depth_expected']
                        dbg_depth_median = dbg_pkg['depth_median']
                        dbg_alpha = dbg_pkg['rend_alpha']
                        save_img_u8(dbg_img.permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_post_reinit_rgb.png'))
                        save_img_u8(dbg_alpha.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_post_reinit_alpha.png'))
                        depth_np = dbg_depth_mean.squeeze().cpu().numpy()
                        save_img_u8(convert_gray_to_cmap(depth_np, map_mode='turbo', revert=False),
                                    os.path.join(output_path, f'{iteration}_post_reinit_depth_mean.png'))
                        depth_med_np = dbg_depth_median.squeeze().cpu().numpy()
                        save_img_u8(convert_gray_to_cmap(depth_med_np, map_mode='turbo', revert=False),
                                    os.path.join(output_path, f'{iteration}_post_reinit_depth_median.png'))
                        del dbg_pkg

                    torch.cuda.empty_cache()
                    tqdm.write(f"[MCMC] Depth reinit at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} Gaussians")

            # Densification / MCMC Relocation.
            # When --minimc is active, the closed-loop RJ-MCMC pipeline (below) owns
            # all relocation and birth — skip the vanilla opacity-proportional path here.
            if iteration < opt.densify_until_iter and optim_gaussian and not args.minimc:
                if args.mcmc or args.mcmc_deficit or args.mcmc_fps or args.mcmc_fps:
                    # MCMC mode: relocate dead Gaussians and add new ones
                    if args.cap_max <= 0:
                        raise ValueError("--cap_max must be specified and positive when using --mcmc, --mcmc_deficit, or --mcmc_fps mode")

                    if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                        # Find dead Gaussians (very low opacity)
                        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                        current_count = len(gaussians.get_xyz)

                        if args.mcmc_deficit and current_count > args.cap_max:
                            # DEFICIT MODE: Delete dead Gaussians until we reach cap_max
                            n_dead = dead_mask.sum().item()
                            if n_dead > 0:
                                gaussians.prune_points(dead_mask)
                            # No add_new_gs while in deficit - just delete
                        else:
                            # Normal MCMC (also used by mcmc_fps): relocate dead Gaussians + add new ones
                            gaussians.relocate_gs(dead_mask=dead_mask, probs_mode=args.mcmc_sample)
                            gaussians.add_new_gs(cap_max=args.cap_max, probs_mode=args.mcmc_sample)
                else:
                    # Traditional densification
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels = pixels)

                    prune_tag = (iteration % opacity_reset_interval >= opacity_reset_protect * densification_interval)
                    # Diffuse_offset mode: pause pruning for first 3k iterations after initialize
                    if args.method == "diffuse_offset" and iteration < cfg_model.ingp_stage.initialize + 3000:
                        prune_tag = False
                    # Skip densification on any iter that hosts a depth reinit.
                    # --mini (v2): only the single mini_depth_reinit_iter.
                    # --mini1 (v1): every repeated reinit iter within the window.
                    _is_reinit_iter = (args.mini and iteration == args.mini_depth_reinit_iter)
                    if args.mini1:
                        if (iteration > args.mini_depth_reinit_iter
                                and iteration <= args.mini1_depth_reinit_until
                                and args.mini1_depth_reinit_interval > 0
                                and (iteration - args.mini_depth_reinit_iter) % args.mini1_depth_reinit_interval == 0):
                            _is_reinit_iter = True
                    # Gating rules:
                    # --mini (v2): aggressive clone every 250 iters does the work; disable.
                    # --mini1 (v1): enabled — the pre-reinit aggressive-clone only fires
                    #               3x total, so standard densify_and_prune carries the
                    #               bulk of growth between reinits.
                    # everything else: enabled.
                    # --minispa: use standard 3DGS densify_and_prune for growth
                    # (aggressive cloning is disabled via mini_simp_iter1 in argparse).
                    _disable_densify = args.mini and not args.mini1 and not args.minispa
                    # --minispa: no densification once ADMM has started (matches GSpa).
                    # Growth is from 500 → minispa_admm_start only.
                    _in_admm = (args.minispa and iteration >= args.minispa_admm_start)

                    # --fastgs: replaces standard densify_and_prune with VCD/VCP.
                    # Fires every fastgs_densify_interval iters (paper: 500) until
                    # fastgs_densify_until (paper: 15000). Uses K random views.
                    if (args.fastgs
                            and iteration > opt.densify_from_iter
                            and iteration < args.fastgs_densify_until
                            and iteration % args.fastgs_densify_interval == 0
                            and not _is_reinit_iter):
                        from utils.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
                        _vp_stack = scene.getTrainCameras().copy()
                        _camlist = sampling_cameras(_vp_stack, num_cams=args.fastgs_num_views)
                        def _fastgs_render_fn(v, metric_map=None):
                            return render(v, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False,
                                          metric_map=metric_map)
                        importance_score, pruning_score = compute_gaussian_score_fastgs(
                            _camlist, gaussians, _fastgs_render_fn,
                            loss_thresh=args.fastgs_loss_thresh,
                            lambda_dssim=args.fastgs_lambda_dssim,
                            densify=True,
                        )
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        _blur_mask_for_densify = (gaussians._blur_split_mask
                                                  if args.blur_split
                                                  and hasattr(gaussians, "_blur_split_mask")
                                                  else None)
                        _stats = gaussians.densify_and_prune_fastgs(
                            min_opacity=opt.opacity_cull,
                            extent=scene.cameras_extent,
                            max_screen_size=size_threshold,
                            importance_score=importance_score,
                            pruning_score=pruning_score,
                            grad_thresh=args.fastgs_grad_thresh,
                            grad_abs_thresh=args.fastgs_grad_abs_thresh,
                            dense=args.fastgs_dense,
                            importance_thresh=args.fastgs_importance_thresh,
                            prune_budget_frac=args.fastgs_prune_budget_frac,
                            extra_split_mask=_blur_mask_for_densify,
                            opacity_clamp=args.fastgs_opacity_clamp,
                        )
                        # Reset blur-split accumulator (size changed via clone/split/prune).
                        if args.blur_split:
                            gaussians._blur_split_mask = torch.zeros(
                                gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                        # NOTE: no training_setup() — densification_postfix +
                        # prune_points already preserve Adam state via
                        # cat_tensors_to_optimizer / _prune_optimizer.
                        tqdm.write(
                            f"[FASTGS] Densify+prune at iter {iteration}: "
                            f"cloned={_stats['cloned']}, split_parents={_stats['split_parents']}, "
                            f"N={len(gaussians.get_xyz)}")
                    elif (not args.fastgs
                            and iteration > opt.densify_from_iter
                            and iteration % densification_interval == 0
                            and not _is_reinit_iter
                            and not _disable_densify
                            and not _in_admm):
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        gaussians.densify_and_prune(densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, \
                        appearance_update_threshold, active_levels, densify_tag = (iteration < opt.densify_until_iter), prune_tag = prune_tag)

                        # --feature SV: refresh per-site eval mask after densify
                        # (matches reference's `update_sites_mask()` cadence — called
                        # inside the densify cycle of radiance/train.py:181).
                        if args.feature == "SV":
                            gaussians.update_sites_mask()
                    
                    if not args.mini and not args.fastgs and (iteration % opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                        if iteration <= cfg_model.training_cfg.reset_until_iter:
                            gaussians.reset_opacity()

                    # FastGS post-densify aggressive pruning (paper: every 3000 iters after 15k).
                    if (args.fastgs
                            and iteration >= args.fastgs_densify_until
                            and iteration < args.fastgs_final_prune_until
                            and iteration % args.fastgs_final_prune_interval == 0):
                        from utils.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
                        _vp_stack = scene.getTrainCameras().copy()
                        _camlist = sampling_cameras(_vp_stack, num_cams=args.fastgs_num_views)
                        def _fastgs_prune_render_fn(v, metric_map=None):
                            return render(v, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False,
                                          metric_map=metric_map)
                        _, pruning_score = compute_gaussian_score_fastgs(
                            _camlist, gaussians, _fastgs_prune_render_fn,
                            loss_thresh=args.fastgs_loss_thresh,
                            lambda_dssim=args.fastgs_lambda_dssim,
                            densify=False,
                        )
                        n_before = len(gaussians.get_xyz)
                        n_pruned = gaussians.final_prune_fastgs(
                            min_opacity=args.fastgs_final_min_opacity,
                            pruning_score=pruning_score,
                            score_thresh=args.fastgs_final_score_thresh,
                        )
                        # NOTE: no training_setup() — prune_points preserves
                        # Adam state via _prune_optimizer.
                        tqdm.write(
                            f"[FASTGS] Final-prune at iter {iteration}: "
                            f"{n_before} -> {len(gaussians.get_xyz)} "
                            f"(pruned {n_pruned})")

            # --merge: mode-agnostic geometric consolidation. Placed at the top
            # scope so it runs regardless of --mcmc / --mini / traditional path.
            # Fires AFTER any density changes above, BEFORE mini/gspa blocks below.
            if (args.merge
                    and iteration > 0
                    and iteration % args.merge_interval == 0
                    and iteration <= args.merge_until):
                _n_before_merge = len(gaussians.get_xyz)
                _n_merged = gaussians.consolidate_primitives(
                    tau_dist=(args.merge_tau_dist if args.merge_tau_dist > 0 else None),
                    tau_normal=args.merge_tau_normal,
                    tau_planar=(args.merge_tau_planar if args.merge_tau_planar > 0 else None),
                    tau_feat=args.merge_tau_feat,
                    verbose=False,  # we print via tqdm.write below
                )
                tqdm.write(
                    f"[MERGE] iter {iteration}: {_n_before_merge} -> "
                    f"{len(gaussians.get_xyz)} Gaussians "
                    f"(merged {_n_merged} pairs)"
                )

            # GaussianSpa two-phase sparsification pipeline
            if args.gspa:
                # Phase 1: Importance-based pre-pruning at simp_iter.
                # Sweep all training views, accumulate per-Gaussian importance
                # (sum of α·T blending weights), sample (1 - p1) fraction
                # weighted by importance, reinit the survivors.
                # Under --minispa the silhouette-aware depth reinit serves as
                # Phase 1, so skip the importance-prune block entirely.
                if iteration == args.gspa_simp_iter and not args.minispa:
                    _n_before_p1 = len(gaussians.get_xyz)

                    # If --gspa_target_count > 0, auto-compute p1 AND p2 so the
                    # expected final count after BOTH phases equals the target.
                    # Split the keep-ratio evenly across both phases:
                    #   keep_total = target / current
                    #   keep_per_phase = sqrt(keep_total)
                    #   p1 = p2 = 1 - keep_per_phase
                    if args.gspa_target_count > 0:
                        keep_total = args.gspa_target_count / max(_n_before_p1, 1)
                        keep_total = min(max(keep_total, 1e-6), 1.0)
                        keep_per_phase = keep_total ** 0.5
                        _p1 = max(0.0, min(0.99, 1.0 - keep_per_phase))
                        _p2 = _p1
                        args.gspa_prune_ratio1 = _p1
                        args.gspa_ratio = _p2
                        print(f"[GSPA] --gspa_target_count={args.gspa_target_count} → "
                              f"auto prune_ratio1={_p1:.3f}, ratio2={_p2:.3f} "
                              f"(current {_n_before_p1} → expected ~{args.gspa_target_count})")
                    else:
                        _p1 = args.gspa_prune_ratio1

                    print(f"[GSPA] Phase 1: computing importance scores over all train views...")
                    # Wrap the training `render` with the flags compute_importance_scores
                    # expects (record_transmittance=True so transmittance_avg / cover_pixels
                    # are populated per Gaussian).
                    def _gspa_render_fn(view):
                        return render(view, gaussians, pipe, background, beta=beta,
                                      iteration=iteration, cfg=cfg_model, ingp=ingp,
                                      record_transmittance=True, is_training=False)
                    imp_score = OptimizingSpa.compute_importance_scores(
                        gaussians, scene, _gspa_render_fn, pipe, background,
                        imp_metric=args.gspa_imp_metric,
                    )
                    OptimizingSpa.importance_prune(
                        gaussians, imp_score, prune_ratio=_p1, scene=scene)
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    print(f"[GSPA] Phase 1 done: {_n_before_p1} → {len(gaussians.get_xyz)} Gaussians")

                # Phase 2: ADMM sparsification (start_iter to stop_iter)
                if iteration == args.gspa_start_iter:
                    optimizing_spa = OptimizingSpa(
                        gaussians, rho=args.gspa_rho, prune_ratio=args.gspa_ratio)
                    optimizing_spa.update_z_u(update_u=False)  # Initial z-only update
                    n = len(gaussians.get_xyz)
                    print(f"\n[GSPA] ADMM started: {n} Gaussians, "
                          f"rho={args.gspa_rho}, ratio={args.gspa_ratio}, "
                          f"interval={args.gspa_interval}, stop={args.gspa_stop_iter}")
                elif optimizing_spa is not None and iteration > args.gspa_start_iter and iteration <= args.gspa_stop_iter:
                    optimizing_spa.handle_densification_change()
                    if iteration % args.gspa_interval == 0:
                        optimizing_spa.update_z_u()
                        # --minispa: progressive hard prune during ADMM.
                        # Each z/u update, delete a batch of the lowest-opacity
                        # Gaussians so the count descends smoothly toward
                        # --gspa_target_count by gspa_stop_iter, instead of all
                        # landing in one big drop. ADMM has been pushing the
                        # doomed set's opacities down, so the bottom slice is
                        # mostly already-faded Gaussians.
                        if args.minispa and args.gspa_target_count > 0 and iteration < args.gspa_stop_iter:
                            n_cur = len(gaussians.get_xyz)
                            if n_cur > args.gspa_target_count:
                                remaining_intervals = max(1, (args.gspa_stop_iter - iteration) // max(args.gspa_interval, 1))
                                to_prune = (n_cur - args.gspa_target_count) // remaining_intervals
                                if to_prune > 0:
                                    with torch.no_grad():
                                        op = gaussians.get_opacity.squeeze(-1)
                                        _, low_idx = torch.topk(op, k=int(to_prune), largest=False)
                                        prune_mask = torch.zeros(n_cur, dtype=torch.bool, device=op.device)
                                        prune_mask[low_idx] = True
                                    gaussians.prune_points(prune_mask)
                                    tqdm.write(f"[MINISPA] ADMM prune at iter {iteration}: "
                                               f"{n_cur} -> {len(gaussians.get_xyz)} "
                                               f"(target {args.gspa_target_count}, "
                                               f"{remaining_intervals} intervals left)")
                if iteration == args.gspa_stop_iter and optimizing_spa is not None:
                    # Final cleanup prune in case the progressive schedule
                    # didn't exactly hit the target.
                    if args.minispa and args.gspa_target_count > 0:
                        n_cur = len(gaussians.get_xyz)
                        n_extra = n_cur - args.gspa_target_count
                        if n_extra > 0:
                            with torch.no_grad():
                                op = gaussians.get_opacity.squeeze(-1)
                                _, low_idx = torch.topk(op, k=int(n_extra), largest=False)
                                prune_mask = torch.zeros(n_cur, dtype=torch.bool, device=op.device)
                                prune_mask[low_idx] = True
                            gaussians.prune_points(prune_mask)
                            tqdm.write(f"[MINISPA] ADMM final prune at iter {iteration}: "
                                       f"{n_cur} -> {len(gaussians.get_xyz)} (target {args.gspa_target_count})")
                    else:
                        optimizing_spa.prune()
                    optimizing_spa = None

            # Mini-Splatting v2: scheduled events (depth reinit, importance pruning)
            if args.mini and optim_gaussian:
                # Decide whether this iter triggers a depth reinit.
                # --mini (v2): fires exactly at mini_depth_reinit_iter.
                # --mini1 (v1): fires at mini_depth_reinit_iter AND repeatedly on interval
                #               until mini1_depth_reinit_until.
                _reinit_trigger = (iteration == args.mini_depth_reinit_iter)
                # --minispa: pre-ADMM depth reinit at minispa_reinit_iter (default 2000).
                # Under --minispa, mini_depth_reinit_iter is parked at 10^9 so the first
                # line above never fires; this clause restores a real depth reinit at 2000.
                if args.minispa and iteration == args.minispa_reinit_iter:
                    _reinit_trigger = True
                if args.mini1 and not _reinit_trigger:
                    if (iteration > args.mini_depth_reinit_iter
                            and iteration <= args.mini1_depth_reinit_until
                            and args.mini1_depth_reinit_interval > 0
                            and (iteration - args.mini_depth_reinit_iter) % args.mini1_depth_reinit_interval == 0):
                        _reinit_trigger = True
                # --minispa: periodic reinit during ADMM phase.
                # Fires at minispa_admm_start + k·interval for k=1,2,... as long as
                # we're strictly before minispa_admm_stop (no reinit at the final prune iter).
                if (args.minispa and not _reinit_trigger
                        and args.minispa_reinit_interval > 0
                        and iteration > args.minispa_admm_start
                        and iteration < args.minispa_admm_stop
                        and (iteration - args.minispa_admm_start) % args.minispa_reinit_interval == 0):
                    _reinit_trigger = True

                # Depth reinitialization: intersection-preserving prune then snap to surfaces
                if _reinit_trigger:
                    n_before = len(gaussians.get_xyz)

                    # Save pre-reinit debug renders
                    output_path = os.path.join(scene.model_path, 'training_output')
                    os.makedirs(output_path, exist_ok=True)
                    with torch.no_grad():
                        dbg_cam = scene.getTrainCameras()[0]
                        dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                         iteration=iteration, cfg=cfg_model, ingp=ingp,
                                         record_transmittance=False, is_training=False)
                        dbg_img = torch.clamp(dbg_pkg['render'], 0.0, 1.0)
                        dbg_alpha = dbg_pkg['rend_alpha']
                        save_img_u8(dbg_img.permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_pre_reinit_rgb.png'))
                        save_img_u8(dbg_alpha.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_pre_reinit_alpha.png'))
                        for dname, dkey in [('depth_mean', 'depth_expected'), ('depth_median', 'depth_median'), ('depth_maxcontrib', 'depth_max_contributor')]:
                            d = dbg_pkg.get(dkey)
                            if d is not None and d.numel() > 0:
                                save_img_u8(convert_gray_to_cmap(d.squeeze().cpu().numpy(), map_mode='turbo', revert=False),
                                            os.path.join(output_path, f'{iteration}_pre_reinit_{dname}.png'))
                        # Max-contributor *id* colormap (per-pixel dominant Gaussian id).
                        mci_pre = dbg_pkg.get('max_contrib_idx', None)
                        if mci_pre is not None and mci_pre.numel() > 0:
                            save_img_u8(_colorize_max_contrib_idx(mci_pre),
                                        os.path.join(output_path, f'{iteration}_pre_reinit_maxcontrib_id.png'))
                        del dbg_pkg

                    # Pre-reinit: prune low-importance Gaussians (matches MSv2 interesction_preserving)
                    tqdm.write(f"[MINI] Pre-reinit pruning: rendering all views for importance...")
                    n_pre_pruned = gaussians.mini_intersection_preserving(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric=args.mini_imp_metric)
                    if n_pre_pruned > 0:
                        gaussians.training_setup(opt)
                    tqdm.write(f"[MINI] Pre-reinit prune: {n_before} -> {len(gaussians.get_xyz)} Gaussians")
                    torch.cuda.empty_cache()

                    # --minispa_mesh: TSDF-fuse depth maps into a mesh, area-sample uniformly.
                    # Bypasses the per-view pixel-sampling loop below so the reinit is not
                    # biased toward scene regions with more camera coverage.
                    did_mesh_reinit = False
                    if args.minispa and args.minispa_mesh:
                        def _mesh_render_fn(v):
                            return render(v, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False)
                        # Always sample exactly gspa_target_count points from the mesh.
                        # Fall back to current count only if target isn't set.
                        if args.gspa_target_count > 0:
                            _mesh_cap = args.gspa_target_count
                        else:
                            _mesh_cap = len(gaussians.get_xyz)
                        _voxel = args.minispa_mesh_voxel if args.minispa_mesh_voxel > 0 else None
                        tqdm.write(f"[MINISPA-MESH] TSDF fusing {len(scene.getTrainCameras())} views, "
                                   f"target {_mesh_cap} points"
                                   + (f", voxel={_voxel:.4f}" if _voxel is not None else ", voxel=auto")
                                   + (", poisson-disk" if args.minispa_mesh_poisson else ", uniform"))
                        merged = tsdf_mesh_reinit(
                            scene, _mesh_render_fn, target_count=_mesh_cap,
                            voxel_size=_voxel,
                            use_poisson_disk=args.minispa_mesh_poisson,
                        )
                        if merged is not None:
                            merged['scale_factor'] = args.minispa_mesh_scale_factor
                            merged['init_opacity'] = args.minispa_mesh_opacity
                            tqdm.write(f"[MINISPA-MESH] Mesh sampled {merged['xyz'].shape[0]} points "
                                       f"(colors + normals from TSDF, "
                                       f"scale×{args.minispa_mesh_scale_factor}, "
                                       f"opacity={args.minispa_mesh_opacity}).")
                            gaussians.reinitial_from_depth(merged)
                            gaussians.training_setup(opt)
                            gaussians.reset_xyz_lr_schedule(iteration)
                            torch.cuda.empty_cache()
                            did_mesh_reinit = True

                            # Post-reinit debug renders (mirror the pixel-path debug block below).
                            output_path = os.path.join(scene.model_path, 'training_output')
                            os.makedirs(output_path, exist_ok=True)
                            with torch.no_grad():
                                dbg_cam = scene.getTrainCameras()[0]
                                dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                                 iteration=iteration, cfg=cfg_model, ingp=ingp,
                                                 record_transmittance=False, is_training=False)
                                dbg_img = torch.clamp(dbg_pkg['render'], 0.0, 1.0)
                                dbg_alpha = dbg_pkg['rend_alpha']
                                dbg_depth = dbg_pkg.get('depth_max_contributor', None)
                                save_img_u8(dbg_img.permute(1, 2, 0).cpu().numpy(),
                                            os.path.join(output_path, f'{iteration}_post_reinit_rgb.png'))
                                save_img_u8(dbg_alpha.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                            os.path.join(output_path, f'{iteration}_post_reinit_alpha.png'))
                                mci_post = dbg_pkg.get('max_contrib_idx', None)
                                if mci_post is not None and mci_post.numel() > 0:
                                    save_img_u8(_colorize_max_contrib_idx(mci_post),
                                                os.path.join(output_path, f'{iteration}_post_reinit_maxcontrib_id.png'))
                                if dbg_depth is not None and dbg_depth.numel() > 0:
                                    depth_np = dbg_depth.squeeze().cpu().numpy()
                                    save_img_u8(convert_gray_to_cmap(depth_np, map_mode='turbo', revert=False),
                                                os.path.join(output_path, f'{iteration}_post_reinit_depth_maxcontrib.png'))
                                del dbg_pkg
                        else:
                            tqdm.write("[MINISPA-MESH] TSDF fusion produced an empty mesh — falling back to pixel-sampling path.")

                    if not did_mesh_reinit:
                        views = scene.getTrainCameras()
                        all_reinit_data = []
                        N_total = len(gaussians.get_xyz)
                        # Snapshot the OLD SH so we can transfer it into the new (reinit) Gaussians
                        # by max-contributor id sampled per pixel during the per-view render below.
                        old_features_dc = gaussians._features_dc.detach().clone()      # [N, 1, 3]
                        old_features_rest = gaussians._features_rest.detach().clone()  # [N, K, 3]
                        for v in views:
                            with torch.no_grad():
                                rpkg = render(v, gaussians, pipe, background, beta=beta,
                                              iteration=iteration, cfg=cfg_model, ingp=ingp,
                                              record_transmittance=False, is_training=False)
                                gt_img = v.original_image.cuda()
                                # Use max-contributor depth if available, fall back to median
                                reinit_depth = rpkg.get('depth_max_contributor', None)
                                if reinit_depth is None or reinit_depth.numel() == 0:
                                    reinit_depth = rpkg['depth_median']
                                max_idx_map = rpkg.get('max_contrib_idx', None)
                                # --minispa: cap reinit point count at gspa_target_count
                                # (or current point count, whichever is smaller) so reinits
                                # never produce more points than the final target.
                                _minispa_cap = None
                                if args.minispa and args.gspa_target_count > 0:
                                    _minispa_cap = min(len(gaussians.get_xyz), args.gspa_target_count)
                                data = gaussians.mini_depth_reinit(
                                    [reinit_depth.detach()],
                                    [rpkg['rend_alpha'].detach()],
                                    [v],
                                    gt_images=[gt_img],
                                    normal_maps=[rpkg['rend_normal'].detach()],
                                    num_total_views=len(views),
                                    max_idx_maps=[max_idx_map.detach()] if max_idx_map is not None else None,
                                    src_features_dc=old_features_dc,
                                    src_features_rest=old_features_rest,
                                    compute_safe_radius=args.minispa,
                                    total_count_override=_minispa_cap)
                                if data is not None:
                                    all_reinit_data.append({k: t.cpu() for k, t in data.items()})
                                del rpkg
                            torch.cuda.empty_cache()
                        del old_features_dc, old_features_rest
                        if all_reinit_data:
                            merged = {
                                'xyz': torch.cat([d['xyz'] for d in all_reinit_data], dim=0).cuda(),
                                'colors': torch.cat([d['colors'] for d in all_reinit_data], dim=0).cuda() if 'colors' in all_reinit_data[0] else None,
                                'normals': torch.cat([d['normals'] for d in all_reinit_data], dim=0).cuda() if 'normals' in all_reinit_data[0] else None,
                                'sh_dc': torch.cat([d['sh_dc'] for d in all_reinit_data], dim=0).cuda() if 'sh_dc' in all_reinit_data[0] else None,
                                'sh_rest': torch.cat([d['sh_rest'] for d in all_reinit_data], dim=0).cuda() if 'sh_rest' in all_reinit_data[0] else None,
                                'pixel_footprint': torch.cat([d['pixel_footprint'] for d in all_reinit_data], dim=0).cuda() if 'pixel_footprint' in all_reinit_data[0] else None,
                                'safe_radius': torch.cat([d['safe_radius'] for d in all_reinit_data], dim=0).cuda() if 'safe_radius' in all_reinit_data[0] else None,
                            }
                            gaussians.reinitial_from_depth(merged)
                            gaussians.training_setup(opt)
                            # Reset the xyz exponential schedule so the fresh cohort starts
                            # with position_lr_init instead of the decayed late-training LR.
                            gaussians.reset_xyz_lr_schedule(iteration)

                            # Save post-reinit debug renders
                            output_path = os.path.join(scene.model_path, 'training_output')
                            os.makedirs(output_path, exist_ok=True)
                            with torch.no_grad():
                                dbg_cam = scene.getTrainCameras()[0]
                                dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                                 iteration=iteration, cfg=cfg_model, ingp=ingp,
                                                 record_transmittance=False, is_training=False)
                                dbg_img = torch.clamp(dbg_pkg['render'], 0.0, 1.0)
                                dbg_alpha = dbg_pkg['rend_alpha']
                                dbg_depth = dbg_pkg['depth_max_contributor']
                                save_img_u8(dbg_img.permute(1, 2, 0).cpu().numpy(),
                                            os.path.join(output_path, f'{iteration}_post_reinit_rgb.png'))
                                save_img_u8(dbg_alpha.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                            os.path.join(output_path, f'{iteration}_post_reinit_alpha.png'))
                                # Max-contributor id colormap for the NEW (post-reinit) point set.
                                mci_post = dbg_pkg.get('max_contrib_idx', None)
                                if mci_post is not None and mci_post.numel() > 0:
                                    save_img_u8(_colorize_max_contrib_idx(mci_post),
                                                os.path.join(output_path, f'{iteration}_post_reinit_maxcontrib_id.png'))
                                depth_np = dbg_depth.squeeze().cpu().numpy()
                                save_img_u8(convert_gray_to_cmap(depth_np, map_mode='turbo', revert=False),
                                            os.path.join(output_path, f'{iteration}_post_reinit_depth_maxcontrib.png'))
                                del dbg_pkg

                            torch.cuda.empty_cache()
                            tqdm.write(f"[MINI] Depth reinit at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} Gaussians")

                # Aggressive cloning cadence: --mini v2 only.
                # --mini1 (v1) does NOT use this periodic cadence — it runs
                # aggressive clone exactly once right before each depth reinit
                # (wired inside the reinit dispatch above).
                # Skip on depth_reinit_iter — avoids running clone + reinit in the same step.
                _is_mini1_reinit = False
                if args.mini1:
                    if (iteration > args.mini_depth_reinit_iter
                            and iteration <= args.mini1_depth_reinit_until
                            and args.mini1_depth_reinit_interval > 0
                            and (iteration - args.mini_depth_reinit_iter) % args.mini1_depth_reinit_interval == 0):
                        _is_mini1_reinit = True
                if (not args.mini1
                        and iteration >= 500 and iteration < args.mini_simp_iter1
                        and iteration % args.mini_clone_interval == 0
                        and iteration != args.mini_depth_reinit_iter
                        and not _is_mini1_reinit):
                    n_before = len(gaussians.get_xyz)
                    tqdm.write(f"[MINI] Aggressive clone: rendering all views...")
                    _n_pruned, _n_cloned = gaussians.mini_culling_with_clone(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric=args.mini_imp_metric)
                    if _n_pruned > 0 or _n_cloned > 0:
                        gaussians.training_setup(opt)
                    tqdm.write(
                        f"[MINI] Aggressive clone at iter {iteration}: "
                        f"{n_before} -> {len(gaussians.get_xyz)} Gaussians "
                        f"(pruned={_n_pruned}, cloned={_n_cloned})"
                    )

                # First simplification: importance-weighted sampling (keep ~60%)
                # --minispa skips simp1/simp2 — GSpa ADMM handles sparsification.
                if iteration == args.mini_simp_iter1 and not args.minispa:
                    n_before = len(gaussians.get_xyz)
                    tqdm.write(f"[MINI] Simp1: rendering all views for importance scores...")
                    n_pruned = gaussians.mini_intersection_sampling(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric=args.mini_imp_metric,
                        sampling_factor=args.mini_sampling_factor)
                    if n_pruned > 0:
                        gaussians.training_setup(opt)
                    tqdm.write(f"[MINI] Simplification 1 at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} Gaussians")

                # Second simplification: intersection-preserving (keep top 99%)
                if iteration == args.mini_simp_iter2 and not args.minispa:
                    n_before = len(gaussians.get_xyz)
                    tqdm.write(f"[MINI] Simp2: rendering all views for importance scores...")
                    n_pruned = gaussians.mini_intersection_preserving(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric=args.mini_imp_metric)
                    if n_pruned > 0:
                        gaussians.training_setup(opt)
                    tqdm.write(f"[MINI] Simplification 2 at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} Gaussians")

                # Late low-opacity prune: keep the point count honest after simp2,
                # so opacity-driven regs (--overdraw_reg / --w_overdraw_reg) actually delete dead Gaussians.
                if (args.mini_late_prune_interval > 0
                        and iteration > args.mini_simp_iter2
                        and iteration % args.mini_late_prune_interval == 0):
                    with torch.no_grad():
                        opacity = gaussians.get_opacity.squeeze(-1)
                        prune_mask = opacity < args.mini_late_prune_thresh
                        n_prune = int(prune_mask.sum().item())
                        if n_prune > 0:
                            n_before = len(gaussians.get_xyz)
                            gaussians.prune_points(prune_mask)
                            tqdm.write(f"[MINI] Late prune at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} (opacity < {args.mini_late_prune_thresh})")

            # MiniMC: MCMC + importance pruning with adaptive budget reduction
            # ============================================================
            # --minimc: sweep-based RJ-MCMC + periodic non-contributor cull.
            #
            # Two events:
            #
            # (A) NON-CONTRIBUTOR CULL (every --minimc_reinit_interval iters):
            #     sweep all training views, find every Gaussian that was the
            #     max-weight contributor for ≥1 pixel anywhere ("winners"), then
            #     despawn every OTHER alive Gaussian into the MCMC dead pool by
            #     snapping opacity below dead_thresh. Winners are preserved
            #     exactly (no reinit, no opacity change). Non-contributors
            #     become dead and get re-cloned by (B) on subsequent iters.
            #     Tensor size is invariant. Preempts (B) on coincident iters.
            #
            # (B) SWEEP + CLONE (every --minimc_relocate_interval iters):
            #     importance + visibility sweep, cull by three criteria, clone
            #     top-K alive candidates into dead slots with 50/50 split.
            # ============================================================
            if args.minimc and (args.mcmc or args.mcmc_fps or args.mcmc_deficit):
                _minimc_reinit_fires = (
                    args.minimc_reinit_interval > 0
                    and iteration >= args.minimc_start_iter
                    and iteration <= args.minimc_reinit_until
                    and iteration % args.minimc_reinit_interval == 0
                )
                # --- (A) NON-CONTRIBUTOR CULL ---
                if _minimc_reinit_fires:
                    _mc_out_dir = os.path.join(scene.model_path, 'training_output')
                    os.makedirs(_mc_out_dir, exist_ok=True)
                    # Pre-cull debug renders (RGB + depth_maxcontrib + maxcontrib_id)
                    with torch.no_grad():
                        _dbg_cam_cull = scene.getTrainCameras()[0]
                        _dbg_pre = render(_dbg_cam_cull, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False)
                        save_img_u8(torch.clamp(_dbg_pre['render'], 0, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_pre_reinit_rgb.png'))
                        save_img_u8(_dbg_pre['rend_alpha'].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_pre_reinit_alpha.png'))
                        _d_pre = _dbg_pre.get('depth_max_contributor', None)
                        if _d_pre is not None and _d_pre.numel() > 0:
                            save_img_u8(convert_gray_to_cmap(_d_pre.squeeze().cpu().numpy(),
                                                             map_mode='turbo', revert=False),
                                        os.path.join(_mc_out_dir, f'{iteration}_pre_reinit_depth_maxcontrib.png'))
                        _mci_pre = _dbg_pre.get('max_contrib_idx', None)
                        if _mci_pre is not None and _mci_pre.numel() > 0:
                            save_img_u8(_colorize_max_contrib_idx(_mci_pre),
                                        os.path.join(_mc_out_dir, f'{iteration}_pre_reinit_maxcontrib_id.png'))
                        del _dbg_pre
                    torch.cuda.empty_cache()

                    # Non-contributor cull: union of per-pixel max contributors across
                    # all training views is kept exactly; every OTHER alive Gaussian
                    # is despawned into the MCMC dead pool. Tensor size unchanged.
                    _stats = gaussians.minimc_despawn_non_contributors(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        dead_thresh=args.minimc_dead_thresh,
                    )
                    tqdm.write(
                        f"[MINIMC] Non-contributor cull at iter {iteration}: "
                        f"alive={_stats['n_alive_before']}/{_stats['n_total']} "
                        f"winners={_stats['n_winners']} despawned={_stats['n_despawned']} "
                        f"(dead pool += {_stats['n_despawned']})"
                    )

                    # Post-cull debug renders
                    with torch.no_grad():
                        _dbg_post = render(_dbg_cam_cull, gaussians, pipe, background, beta=beta,
                                           iteration=iteration, cfg=cfg_model, ingp=ingp,
                                           record_transmittance=False, is_training=False)
                        save_img_u8(torch.clamp(_dbg_post['render'], 0, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_post_reinit_rgb.png'))
                        save_img_u8(_dbg_post['rend_alpha'].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_post_reinit_alpha.png'))
                        _d_post = _dbg_post.get('depth_max_contributor', None)
                        if _d_post is not None and _d_post.numel() > 0:
                            save_img_u8(convert_gray_to_cmap(_d_post.squeeze().cpu().numpy(),
                                                             map_mode='turbo', revert=False),
                                        os.path.join(_mc_out_dir, f'{iteration}_post_reinit_depth_maxcontrib.png'))
                        _mci_post = _dbg_post.get('max_contrib_idx', None)
                        if _mci_post is not None and _mci_post.numel() > 0:
                            save_img_u8(_colorize_max_contrib_idx(_mci_post),
                                        os.path.join(_mc_out_dir, f'{iteration}_post_reinit_maxcontrib_id.png'))
                        del _dbg_post
                    torch.cuda.empty_cache()

                # --- (B) SWEEP + RELOCATE ---
                # On coincident iters, runs AFTER the reinit cull so the sweep
                # observes the cull's new dead pool and immediately refills it.
                if (iteration >= args.minimc_start_iter
                        and iteration <= args.minimc_relocate_until
                        and iteration % args.minimc_relocate_interval == 0):
                    # Pre-relocate debug render (one fixed train view) -> training_output/.
                    _mc_out_dir = os.path.join(scene.model_path, 'training_output')
                    os.makedirs(_mc_out_dir, exist_ok=True)
                    # Debug-image cadence: only save full image set on reinit cadence (cheap
                    # every-sweep RGB pre/post otherwise).
                    _save_full_imgs = (args.minimc_reinit_interval > 0
                                       and iteration % args.minimc_reinit_interval == 0)
                    with torch.no_grad():
                        _dbg_cam = scene.getTrainCameras()[0]
                        _dbg_pre = render(_dbg_cam, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False)
                        _img_pre = torch.clamp(_dbg_pre['render'], 0.0, 1.0)
                        save_img_u8(_img_pre.permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_pre_minimc_rgb.png'))
                        if _save_full_imgs:
                            _d_pre = _dbg_pre.get('depth_max_contributor', None)
                            if _d_pre is not None and _d_pre.numel() > 0:
                                save_img_u8(
                                    convert_gray_to_cmap(_d_pre.squeeze().cpu().numpy(),
                                                         map_mode='turbo', revert=False),
                                    os.path.join(_mc_out_dir,
                                                 f'{iteration}_pre_minimc_depth_maxcontrib.png'))
                            _mci_pre = _dbg_pre.get('max_contrib_idx', None)
                            if _mci_pre is not None and _mci_pre.numel() > 0:
                                save_img_u8(_colorize_max_contrib_idx(_mci_pre),
                                            os.path.join(_mc_out_dir,
                                                         f'{iteration}_pre_minimc_maxcontrib_id.png'))
                        del _dbg_pre

                    _stats = gaussians.minimc_sweep_and_relocate(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric="indoor",
                        dead_thresh=args.minimc_dead_thresh,
                        cull_single_view=(not args.minimc_no_single_view_cull),
                        low_imp_cdf_thres=args.minimc_low_imp_cdf,
                        growth_frac=args.minimc_growth_frac,
                    )
                    tqdm.write(
                        f"[MINIMC] iter {iteration}: "
                        f"alive={_stats['n_alive_before']}/{_stats['n_total']} | "
                        f"cull: zero={_stats['n_culled_zero']} "
                        f"single_view={_stats['n_culled_single_view']} "
                        f"low_imp={_stats['n_culled_low_imp']} "
                        f"total={_stats['n_culled_total']} | "
                        f"dead_in={_stats['n_dead_in']} "
                        f"candidates={_stats['n_candidates']} "
                        f"-> clones={_stats['n_clones']} "
                        f"natural_dead_left={_stats['n_natural_dead_left']} "
                        f"(growth={args.minimc_growth_frac*100:.1f}%, vol-preserving split)"
                    )

                    # Post-relocate debug render (same view).
                    with torch.no_grad():
                        _dbg_post = render(_dbg_cam, gaussians, pipe, background, beta=beta,
                                           iteration=iteration, cfg=cfg_model, ingp=ingp,
                                           record_transmittance=False, is_training=False)
                        _img_post = torch.clamp(_dbg_post['render'], 0.0, 1.0)
                        save_img_u8(_img_post.permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(_mc_out_dir, f'{iteration}_post_minimc_rgb.png'))
                        if _save_full_imgs:
                            _d_post = _dbg_post.get('depth_max_contributor', None)
                            if _d_post is not None and _d_post.numel() > 0:
                                save_img_u8(
                                    convert_gray_to_cmap(_d_post.squeeze().cpu().numpy(),
                                                         map_mode='turbo', revert=False),
                                    os.path.join(_mc_out_dir,
                                                 f'{iteration}_post_minimc_depth_maxcontrib.png'))
                            _mci_post = _dbg_post.get('max_contrib_idx', None)
                            if _mci_post is not None and _mci_post.numel() > 0:
                                save_img_u8(_colorize_max_contrib_idx(_mci_post),
                                            os.path.join(_mc_out_dir,
                                                         f'{iteration}_post_minimc_maxcontrib_id.png'))
                        del _dbg_post

            # ============================================================
            # Legacy --minimc schedule (deprecated, disabled, kept for reference).
            # The old 3-stage code used to sit here; replaced by the new loop above.
            # ============================================================
            if False and args.minimc and (args.mcmc or args.mcmc_fps or args.mcmc_deficit):
                # Depth reinitialization
                if args.minimc_reinit_iter > 0 and iteration == args.minimc_reinit_iter:
                    n_before = len(gaussians.get_xyz)
                    # Save pre-reinit debug renders
                    output_path = os.path.join(scene.model_path, 'training_output')
                    os.makedirs(output_path, exist_ok=True)
                    with torch.no_grad():
                        dbg_cam = scene.getTrainCameras()[0]
                        dbg_pkg = render(dbg_cam, gaussians, pipe, background, beta=beta,
                                         iteration=iteration, cfg=cfg_model, ingp=ingp,
                                         record_transmittance=False, is_training=False)
                        save_img_u8(torch.clamp(dbg_pkg['render'], 0, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_pre_reinit_rgb.png'))
                        save_img_u8(dbg_pkg['rend_alpha'].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_pre_reinit_alpha.png'))
                        for dname, dkey in [('depth_mean', 'depth_expected'), ('depth_median', 'depth_median'), ('depth_maxcontrib', 'depth_max_contributor')]:
                            d = dbg_pkg.get(dkey)
                            if d is not None and d.numel() > 0:
                                save_img_u8(convert_gray_to_cmap(d.squeeze().cpu().numpy(), map_mode='turbo', revert=False),
                                            os.path.join(output_path, f'{iteration}_pre_reinit_{dname}.png'))
                        del dbg_pkg
                    torch.cuda.empty_cache()

                    # Depth reinit
                    views = scene.getTrainCameras()
                    all_reinit_data = []
                    for v in views:
                        with torch.no_grad():
                            rpkg = render(v, gaussians, pipe, background, beta=beta,
                                          iteration=iteration, cfg=cfg_model, ingp=ingp,
                                          record_transmittance=False, is_training=False)
                            gt_img = v.original_image.cuda()
                            reinit_depth = rpkg.get('depth_max_contributor', None)
                            if reinit_depth is None or reinit_depth.numel() == 0:
                                reinit_depth = rpkg['depth_median']
                            data = gaussians.mini_depth_reinit(
                                [reinit_depth.detach()], [rpkg['rend_alpha'].detach()], [v],
                                gt_images=[gt_img], normal_maps=[rpkg['rend_normal'].detach()],
                                num_total_views=len(views))
                            if data is not None:
                                all_reinit_data.append({k: t.cpu() for k, t in data.items()})
                            del rpkg
                        torch.cuda.empty_cache()
                    if all_reinit_data:
                        merged = {
                            'xyz': torch.cat([d['xyz'] for d in all_reinit_data], dim=0).cuda(),
                            'colors': torch.cat([d['colors'] for d in all_reinit_data], dim=0).cuda() if 'colors' in all_reinit_data[0] else None,
                            'normals': torch.cat([d['normals'] for d in all_reinit_data], dim=0).cuda() if 'normals' in all_reinit_data[0] else None,
                        }
                        gaussians.reinitial_from_depth(merged)
                        gaussians.training_setup(opt)
                        if ingp is not None:
                            ingp.training_setup(cfg_model.optim)
                            tqdm.write(f"[MINIMC] Reset INGP optimizer after depth reinit")
                        # Disable MCMC noise for 2k iterations
                        minimc_noise_disabled_until = iteration + 2000
                        tqdm.write(f"[MINIMC] Depth reinit at iter {iteration}: {n_before} -> {len(gaussians.get_xyz)} Gaussians")
                        tqdm.write(f"[MINIMC] MCMC noise disabled until iter {minimc_noise_disabled_until}")

                # First simplification: importance sampling, reduce budget by K/2
                if iteration == args.minimc_simp_iter1:
                    n_before = len(gaussians.get_xyz)
                    tqdm.write(f"[MINIMC] Simp1: rendering all views for importance scores...")
                    n_pruned = gaussians.mini_intersection_sampling(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric="indoor", sampling_factor=args.minimc_sampling_factor)
                    if n_pruned > 0:
                        gaussians.training_setup(opt)
                        # Reduce budget by K/2 (half the pruned count)
                        budget_reduction = n_pruned // 2
                        old_cap = args.cap_max
                        args.cap_max = max(len(gaussians.get_xyz), args.cap_max - budget_reduction)
                        tqdm.write(f"[MINIMC] Simp1: {n_before} -> {len(gaussians.get_xyz)} Gaussians (pruned {n_pruned})")
                        tqdm.write(f"[MINIMC] Budget: {old_cap} -> {args.cap_max} (reduced by {budget_reduction})")

                # Second simplification: intersection preserving, reduce budget by L fully
                if iteration == args.minimc_simp_iter2:
                    n_before = len(gaussians.get_xyz)
                    tqdm.write(f"[MINIMC] Simp2: rendering all views for importance scores...")
                    n_pruned = gaussians.mini_intersection_preserving(
                        scene, render, pipe, background, beta=beta,
                        iteration=iteration, cfg=cfg_model, ingp=ingp,
                        imp_metric="indoor")
                    if n_pruned > 0:
                        gaussians.training_setup(opt)
                        # Reduce budget by full L
                        old_cap = args.cap_max
                        args.cap_max = max(len(gaussians.get_xyz), args.cap_max - n_pruned)
                        tqdm.write(f"[MINIMC] Simp2: {n_before} -> {len(gaussians.get_xyz)} Gaussians (pruned {n_pruned})")
                        tqdm.write(f"[MINIMC] Budget: {old_cap} -> {args.cap_max} (reduced by {n_pruned})")

            # Morton z-order sort for cache locality
            if args.morton_interval > 0 and iteration % args.morton_interval == 0 and iteration <= args.morton_end_iter:
                gaussians.morton_sort()
                if iteration % args.morton_interval == 0:
                    tqdm.write(f"[Morton] Sorted {len(gaussians.get_xyz)} Gaussians at iter {iteration}")

            # Optimizer step
            if iteration < opt.iterations:

                if iteration % 500 == 0:
                    torch.cuda.synchronize()
                    _t_opt_start = time.time()

                # Debug: ap_level and hashgrid stats before optimizer step
                if (iteration % 500 == 0 or iteration == first_iter):
                    ap = gaussians._appearance_level.data.squeeze()
                    n_zero_ap = (ap == 0).sum().item()
                    n_total = ap.shape[0]
                    if n_zero_ap > 0:
                        tqdm.write(f"[AP_LEVEL iter={iteration}] WARNING: {n_zero_ap}/{n_total} Gaussians have ap_level=0 (dead hash)")
                    else:
                        tqdm.write(f"[AP_LEVEL iter={iteration}] OK: all {n_total} Gaussians have ap_level>0 (mean={ap.mean().item():.1f})")
                if (iteration % 500 == 0 or iteration == first_iter) and ingp is not None and hasattr(ingp, 'hash_encoding') and ingp.hash_encoding is not None and hasattr(ingp, 'current_optimizer'):
                    with torch.no_grad():
                        for pg in ingp.current_optimizer.param_groups:
                            name = pg.get('name', '?')
                            p = pg['params'][0]
                            v = p.data
                            tqdm.write(f"[HASH_DBG iter={iteration}] {name}: shape={list(v.shape)} "
                                       f"val(mean={v.mean().item():.6f}, std={v.std().item():.6f}, "
                                       f"min={v.min().item():.6f}, max={v.max().item():.6f})")
                            if p.grad is not None:
                                g = p.grad
                                tqdm.write(f"[HASH_DBG iter={iteration}] {name}: "
                                           f"grad(mean={g.mean().item():.6f}, std={g.std().item():.6f}, "
                                           f"min={g.min().item():.6f}, max={g.max().item():.6f}, "
                                           f"norm={g.norm().item():.6f})")
                            else:
                                tqdm.write(f"[HASH_DBG iter={iteration}] {name}: grad=None")

                # --film_freeze_beta_iter: freeze the per-surfel FiLM beta (offset) for the
                # first N iters of this run by zeroing its gradient before the step (gamma,
                # col 0, still trains). beta lives in _film_params[:, 1:]. Lets the hash/MLP
                # + gamma settle against a fixed bias before beta starts refining.
                if args.method in ("film", "3D_SH_filmres", "3D_SH_concat") \
                        and getattr(args, 'film_freeze_beta_iter', 0) > 0 \
                        and (iteration - first_iter) < args.film_freeze_beta_iter \
                        and hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0 \
                        and gaussians._film_params.grad is not None:
                    if getattr(args, 'film_act', 'identity') == 'gamma_sigm_split':
                        # cols 22:25 are the per-level gammas 1..3 — keep them training
                        # (mirrors "gamma, col 0, keeps training"); freeze only true beta.
                        gaussians._film_params.grad[:, 1:22] = 0.0
                    else:
                        gaussians._film_params.grad[:, 1:] = 0.0

                # --film_freeze_gamma_iter: hold the FiLM gammas WIDE OPEN (sigmoid ~1)
                # for the first N iters — pin raw gamma to --film_freeze_gamma_raw
                # (default 4.0 -> sigmoid 0.982) + zero its grads, so the hash trains
                # ungated (no per-surfel level-killing during the early LR race, no
                # early sigmoid saturation lock-in); gammas then train from the open
                # state. Covers col 0 always, cols 22:25 too for gamma_sigm_split.
                # Pin + grad-zero every iter => truly frozen (no Adam momentum drift).
                if args.method in ("film", "3D_SH_filmres") \
                        and int(getattr(args, 'film_freeze_gamma_iter', 0)) > 0 \
                        and (iteration - first_iter) < int(args.film_freeze_gamma_iter) \
                        and hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0:
                    _graw = float(getattr(args, 'film_freeze_gamma_raw', 4.0))
                    _gsplit = getattr(args, 'film_act', 'identity') == 'gamma_sigm_split'
                    with torch.no_grad():
                        gaussians._film_params.data[:, 0] = _graw
                        if _gsplit:
                            gaussians._film_params.data[:, 22:25] = _graw
                    if gaussians._film_params.grad is not None:
                        gaussians._film_params.grad[:, 0] = 0.0
                        if _gsplit:
                            gaussians._film_params.grad[:, 22:25] = 0.0

                # --film_beta_active_dims N (film/3D_SH_filmres): restrict the FiLM-beta latent to one
                # frequency end. beta[i] modulates hash channel i, and the 16 channels are 4 hash
                # levels × 4D (feat[0:4]=coarsest .. feat[12:16]=finest).
                #   N > 0: keep the BOTTOM N dims (low-freq) active, lock beta[N:16]   (high-freq).
                #   N < 0: keep the TOP |N| dims (high-freq) active, lock beta[0:16-|N|] (low-freq).
                #   |N| >= 16 (incl. default 16): all active, no lock.
                # Locked dims are pinned to 0 + grad-zeroed every iter (truly frozen, no momentum drift).
                _bad = int(getattr(args, 'film_beta_active_dims', 16))
                if args.method in ("film", "3D_SH_filmres") and -16 < _bad < 16 \
                        and hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0:
                    _blo, _bhi = (1 + _bad, 17) if _bad >= 0 else (1, 17 + _bad)
                    with torch.no_grad():
                        gaussians._film_params.data[:, _blo:_bhi] = 0.0
                    if gaussians._film_params.grad is not None:
                        gaussians._film_params.grad[:, _blo:_bhi] = 0.0

                # `--method GEStex` joint stage: (1) route the atlas gradient (produced
                # by the rasterizer backward into the buffer the renderer set) onto the
                # _tex_atlas leaf's .grad (mirrors the get_mlp_grads pattern), and (2)
                # freeze surfel geometry — zero the grad of xyz/scaling/rotation/opacity
                # on textured (surfel) rows so only the spawned 3D Gaussians move. The
                # atlas + surfel SV (SH) + Gaussians stay trainable.
                if getattr(args, 'is_gestex', False) and ingp is not None and getattr(ingp, 'is_gestex_joint', False):
                    _ag = getattr(gaussians, '_ges_atlas_grad', None)
                    if _ag is not None and hasattr(gaussians, '_tex_atlas') and gaussians._tex_atlas.numel() > 0:
                        # Use the CURRENT is_textured (matches _tex_atlas size) — densify may
                        # have appended Gaussian rows between the render and here. Surfels are
                        # the first rows and never pruned (gated to ~is_textured + opacity-pinned),
                        # so the surfel-subset grad _ag aligns with the True rows. Guard on sizes
                        # so a rare surfel-count change degrades gracefully instead of crashing.
                        _cur_sf = getattr(gaussians, '_is_textured', None)
                        _at = gaussians._tex_atlas
                        if (_cur_sf is not None and _cur_sf.numel() == _at.shape[0]
                                and _ag.shape[0] == int(_cur_sf.sum().item())
                                and tuple(_ag.shape[1:]) == tuple(_at.shape[1:])):
                            _full = torch.zeros_like(_at)
                            _full[_cur_sf] = _ag
                            _at.grad = _full
                        elif _at.numel() == _ag.numel():
                            _at.grad = _ag   # cascade path: full-size atlas grad
                    if hasattr(gaussians, '_is_textured') and gaussians._is_textured.numel() > 0:
                        _texrows = gaussians._is_textured
                        for _p in (gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity):
                            if _p.grad is not None and _p.grad.shape[0] == _texrows.shape[0]:
                                _p.grad[_texrows] = 0.0
                        if getattr(gaussians, '_scaling_z', None) is not None \
                                and gaussians._scaling_z.numel() > 0 and gaussians._scaling_z.grad is not None \
                                and gaussians._scaling_z.grad.shape[0] == _texrows.shape[0]:
                            gaussians._scaling_z.grad[_texrows] = 0.0

                # `--method GEStex`: every 1k iters, print param-value norm + grad norm for
                # each optimizable group so you can see what is actually training in each
                # phase. Pre-bake: hashgrid + MLP + SH + geometry train. Post-bake (joint):
                # hashgrid + MLP are FROZEN + BYPASSED (grad None/0 by design); atlas + SH +
                # 3D-Gaussian geometry/opacity train. Printed here (grads populated, pre-step).
                if getattr(args, 'is_gestex', False) and iteration % 1000 == 0:
                    def _pn(t):
                        return float(t.detach().norm().item()) if (t is not None and t.numel() > 0) else 0.0
                    def _gn(t):
                        return float(t.grad.detach().norm().item()) if (t is not None and getattr(t, 'grad', None) is not None) else float('nan')
                    _joint = bool(getattr(ingp, 'is_gestex_joint', False))
                    _hp = _hg = 0.0
                    if ingp is not None and getattr(ingp, 'hash_encoding', None) is not None:
                        for _pp in ingp.hash_encoding.parameters():
                            _hp += _pn(_pp) ** 2
                            _hg += (_pn(_pp.grad) if _pp.grad is not None else 0.0) ** 2
                        _hp, _hg = _hp ** 0.5, _hg ** 0.5
                    _mlp = getattr(ingp, 'mlp_fused', None) if ingp is not None else None
                    _mp = _mg = 0.0
                    if _mlp is not None:
                        for _li in (0, 2, 4):
                            _w = _mlp[_li].weight
                            _mp += _pn(_w) ** 2
                            _mg += (_pn(_w.grad) if _w.grad is not None else 0.0) ** 2
                        _mp, _mg = _mp ** 0.5, _mg ** 0.5
                    _atl = getattr(gaussians, '_tex_atlas', None)
                    tqdm.write(
                        f"[GEStex iter={iteration}] phase={'JOINT' if _joint else 'pre-bake'} | "
                        f"hash: |w|={_hp:.4f} |grad|={_hg:.3e} | "
                        f"mlp: |w|={_mp:.4f} |grad|={_mg:.3e} | "
                        f"atlas: |w|={_pn(_atl):.4f} |grad|={_gn(_atl):.3e} | "
                        f"SH(f_dc): |w|={_pn(gaussians._features_dc):.3f} |grad|={_gn(gaussians._features_dc):.3e} | "
                        f"xyz|grad|={_gn(gaussians._xyz):.3e} scale|grad|={_gn(gaussians._scaling):.3e} "
                        f"opac|grad|={_gn(gaussians._opacity):.3e}")

                if args.mini and mini_last_visibility is not None:
                    gaussians.optimizer.step(visibility=mini_last_visibility, N=radii.shape[0])
                else:
                    gaussians.optimizer.step()

                # --method film: confirm per-surfel gamma/beta are actually training.
                # Printed AFTER step() / BEFORE zero_grad so we see both the updated
                # values and the gradient that drove them. gamma init 1.0, beta init 0.0.
                # Iterations are RELATIVE to first_iter so it fires on the first 500/1000
                # steps of this run (works whether fresh or resumed from a warmup ckpt).
                if args.method in ("film", "3D_SH_filmres", "3D_SH_concat") and (iteration - first_iter) in (0, 50, 100, 250, 500, 1000) \
                        and hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0:
                    with torch.no_grad():
                        fp = gaussians._film_params
                        g = fp[:, 0]      # gamma [N]
                        b = fp[:, 1:]     # beta  [N, 24]
                        gr = fp.grad
                        gnorm = float(gr.norm()) if gr is not None else float('nan')
                        # gamma_sigm_split: per-level gamma stats (sigmoid-activated means).
                        # gamma_0 = col 0, gamma_1..3 = cols 22..24. Per-level grad norms
                        # expose the frozen-gamma symptom (a level with grad exactly 0).
                        _split_stats = ""
                        if getattr(args, 'film_act', 'identity') == 'gamma_sigm_split':
                            _glvls = [fp[:, 0]] + [fp[:, 22 + k] for k in range(3)]
                            _sig_means = " ".join(
                                f"l{k}={torch.sigmoid(_g).mean().item():.4f}" for k, _g in enumerate(_glvls))
                            if gr is not None:
                                _gr_lvls = [gr[:, 0]] + [gr[:, 22 + k] for k in range(3)]
                                _gr_norms = " ".join(
                                    f"l{k}={float(_gg.norm()):.2e}" for k, _gg in enumerate(_gr_lvls))
                            else:
                                _gr_norms = "n/a"
                            _split_stats = (f" | split sig(gamma_l).mean: {_sig_means} | "
                                            f"gamma_l.grad_norm: {_gr_norms}")
                        tqdm.write(
                            f"[FILM iter={iteration}] "
                            f"gamma(init1): mean={g.mean().item():.5f} std={g.std().item():.5f} "
                            f"min={g.min().item():.4f} max={g.max().item():.4f} | "
                            f"beta(init0): mean={b.mean().item():.5f} std={b.std().item():.5f} "
                            f"absmax={b.abs().max().item():.4f} | "
                            f"film_params.grad_norm={gnorm:.3e} | "
                            f"moved(|g-1|>1e-4)={(g.sub(1.0).abs()>1e-4).float().mean().item():.3f} "
                            f"moved(|b|>1e-4)={(b.abs()>1e-4).float().mean().item():.3f}"
                            + _split_stats)

                gaussians.optimizer.zero_grad(set_to_none = True)

                # --3rgs: step the per-camera pose delta. Gated to the active
                # window [warmup, until]; the grad was produced by the rendered
                # photometric loss above (routed through means3D/rotations).
                if (pose_optimizer is not None
                        and args.pose_refine_warmup <= iteration <= args.pose_refine_until):
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
                    if iteration % 500 == 0:
                        with torch.no_grad():
                            _pw = pose_opt.embeds.weight
                            _dt = _pw[:, :3].norm(dim=1)
                            _dr = _pw[:, 3:].norm(dim=1)
                        tqdm.write(f"[3RGS iter={iteration}] pose delta | "
                                   f"trans(mean={_dt.mean():.2e} max={_dt.max():.2e}) "
                                   f"rot6d(mean={_dr.mean():.2e} max={_dr.max():.2e})")

                # --ppisp: step the ISP params (+ controller). Its own Adam and
                # LR schedule (linear warmup → exp decay), decoupled from the
                # Gaussian optimizer so densification never touches it. The
                # scheduler also drives the controller activation check inside
                # PPISP.forward(), so it must be stepped every iteration.
                if ppisp is not None:
                    for _o in ppisp_optimizers:
                        _o.step()
                        _o.zero_grad(set_to_none=True)
                    for _s in ppisp_schedulers:
                        _s.step()
                    if iteration % 500 == 0:
                        with torch.no_grad():
                            _e = ppisp.exposure_params
                            _c = ppisp.color_params
                            _v = ppisp.vignetting_params
                        tqdm.write(
                            f"[PPISP iter={iteration}] "
                            f"exposure(stops): mean={_e.mean().item():+.4f} "
                            f"std={_e.std().item():.4f} "
                            f"min={_e.min().item():+.3f} max={_e.max().item():+.3f} | "
                            f"color |c|max={_c.abs().max().item():.4f} | "
                            f"vig alpha={_v[0, :, 2:].mean(0).tolist()} "
                            f"center=({_v[0, :, 0].mean().item():+.4f},{_v[0, :, 1].mean().item():+.4f})")

                # --deform: step the deformation MLP (the per-surfel latent is in
                # gaussians.optimizer, already stepped above). Gated to warmup→end;
                # before warmup deform is None ⇒ no grad ⇒ nothing to step.
                if deform_optimizer is not None and iteration >= args.deform_warmup:
                    deform_optimizer.step()
                    deform_optimizer.zero_grad(set_to_none=True)
                    if iteration % 500 == 0 and _deform is not None:
                        with torch.no_grad():
                            _dx = _deform[0].norm(dim=1)
                            _dq_s = (f"d_rot(mean={_deform[1].norm(dim=1).mean():.2e})"
                                     if _deform[1] is not None else "d_rot(off)")
                        tqdm.write(f"[DEFORM iter={iteration}] frame deform | "
                                   f"d_xyz(mean={_dx.mean():.2e} max={_dx.max():.2e}) {_dq_s} "
                                   f"latent|.|={gaussians._deform_latent.abs().mean():.2e}")

                if optim_ngp:
                    ingp.current_optimizer.step()
                    ingp.current_optimizer.zero_grad(set_to_none = True)
                    # proberes: decoupled per-step decay on the pixel image.
                    # Rarely-supervised texels (sparse, noisy dL/dtex through the
                    # eps=1e-15 Adam) random-walk to large values without this;
                    # decay pulls them back to 0 while supervised texels re-earn
                    # their content each step. Equilibrium noise std is roughly
                    # lr_px * sqrt(1/(2*decay)).
                    if getattr(ingp, 'is_proberes_mode', False)                             and getattr(ingp.probe_field, 'pixels', None) is not None                             and args.probe_pixel_decay > 0.0:
                        with torch.no_grad():
                            ingp.probe_field.pixels.mul_(1.0 - args.probe_pixel_decay)

                if iteration % 500 == 0:
                    torch.cuda.synchronize()
                    _t_opt_end = time.time()
                    # Per-pixel contributor stats
                    gs_num = render_pkg.get("gaussian_num", None)
                    gs_stats = ""
                    if gs_num is not None:
                        gs_map = gs_num.squeeze()
                        gs_stats = (f" | contrib: max={gs_map.max().item():.0f}, "
                                    f"mean={gs_map.mean().item():.1f}, "
                                    f"median={gs_map.median().item():.0f}")
                    od_stats = ""
                    od_map = render_pkg.get("render_overdraw", None)
                    if od_map is not None and od_map.numel() > 0:
                        od = od_map.squeeze()
                        if od.numel() > 0 and od.max().item() > 0:
                            od_stats = (f" | soft_contrib: mean={od.mean().item():.1f}, "
                                        f"median={od.median().item():.0f}")
                    tqdm.write(f"[TIMING {iteration}] fwd={(_t_fwd_end-_t_fwd_start)*1000:.1f}ms, "
                              f"bwd={(_t_bwd_end-_t_bwd_start)*1000:.1f}ms, "
                              f"opt={(_t_opt_end-_t_opt_start)*1000:.1f}ms, "
                              f"total={(_t_opt_end-_t_fwd_start)*1000:.1f}ms"
                              f"{gs_stats}{od_stats}")

                # --method proberes: render TEST view 0 (full + tex_only) so texture
                # readback is directly inspectable:
                #   training_output/{it}_testview0.png / _testview0_tex.png
                # Cadence follows --save_interval (5000) rather than a hardcoded 1000 —
                # at 8192^2 these dumps are large and 15 per run is wasteful. Set
                # PROBERES_TESTVIEW_EVERY to override (0 disables entirely).
                _tv_every = int(os.environ.get('PROBERES_TESTVIEW_EVERY',
                                               getattr(args, 'save_interval', 5000) or 5000))
                if (getattr(ingp, 'is_proberes_mode', False) and _tv_every > 0
                        and iteration % _tv_every == 0
                        and not getattr(ingp, 'hashgrid_disabled', False)):
                    try:
                        _tv_cams = scene.getTestCameras()
                        _tv_cam = _tv_cams[0] if _tv_cams else scene.getTrainCameras()[0]
                        _tv_out = os.path.join(scene.model_path, 'training_output')
                        os.makedirs(_tv_out, exist_ok=True)
                        with torch.no_grad():
                            _tv_k = dict(ingp=ingp, beta=beta, iteration=iteration,
                                         cfg=cfg_model, is_training=False,
                                         lowpass=args.lowpass)
                            _tv_full = render(_tv_cam, gaussians, pipe,
                                              torch.zeros(3, device="cuda"), **_tv_k)['render']
                            save_img_u8(_tv_full.clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                                        os.path.join(_tv_out, f'{iteration}_testview0.png'))
                            _tv_tex = render(_tv_cam, gaussians, pipe,
                                             torch.zeros(3, device="cuda"),
                                             decompose_mode='tex_only', **_tv_k)['render']
                            save_img_u8(_tv_tex.clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                                        os.path.join(_tv_out, f'{iteration}_testview0_tex.png'))
                    except Exception as _tv_e:
                        tqdm.write(f"[PROBERES] testview0 dump failed: {_tv_e}")

                # Skybox optimizer step (only after switch_iter when skybox is active)
                if skybox is not None and iteration >= cfg_model.ingp_stage.switch_iter:
                    # Debug: check if gradients exist
                    if iteration % 1000 == 0:
                        grad = skybox.texture.grad
                        if grad is not None:
                            print(f"[SKYBOX] iter {iteration}: grad norm = {grad.norm().item():.6f}, texture range = [{skybox.texture.min().item():.4f}, {skybox.texture.max().item():.4f}]")
                        else:
                            print(f"[SKYBOX] iter {iteration}: NO GRADIENT!")
                    skybox.optimizer.step()
                    skybox.optimizer.zero_grad(set_to_none=True)

                # Background hashgrid optimizer step (only after bg_start_iter)
                bg_start_iter = max(args.bg_hashgrid_start_iter, cfg_model.ingp_stage.switch_iter)
                if bg_hashgrid is not None and iteration >= bg_start_iter:
                    # Debug: check if gradients exist
                    if iteration % 1000 == 0 or iteration == bg_start_iter:
                        grad = bg_hashgrid.hash_encoding.embeddings.grad
                        if grad is not None:
                            print(f"[BG_HASHGRID] iter {iteration}: grad norm = {grad.norm().item():.6f}")
                        else:
                            print(f"[BG_HASHGRID] iter {iteration}: NO GRADIENT!")
                        if iteration == bg_start_iter:
                            print(f"[BG_HASHGRID] Starting BG hashgrid training at iteration {bg_start_iter}")
                    bg_hashgrid.optimizer.step()
                    bg_hashgrid.optimizer.zero_grad(set_to_none=True)

                # MCMC: SGLD noise injection after optimizer step
                # MiniMC: disable noise for 2k iters after depth reinit
                if (args.mcmc or args.mcmc_deficit or args.mcmc_fps) and iteration >= minimc_noise_disabled_until:
                    # Get current xyz learning rate
                    xyz_lr = gaussians.optimizer.param_groups[0]['lr']

                    # Build covariance from scale and rotation. For 2DGS surfels
                    # the third (normal) axis has no stored scale; we use
                    # min(sx, sy) so SGLD noise can drift the Gaussian slightly
                    # off the tangent plane without overshooting the surface.
                    scale_xy = gaussians.get_scaling                                  # [N, 2]
                    scale_n = scale_xy.min(dim=-1, keepdim=True).values               # [N, 1]
                    L = build_scaling_rotation(
                        torch.cat([scale_xy, scale_n], dim=-1),
                        gaussians.get_rotation
                    )
                    actual_covariance = L @ L.transpose(1, 2)
                    
                    # Sigmoid function for opacity-based noise scaling
                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    # Generate noise scaled by opacity (low opacity = more noise)
                    noise = torch.randn_like(gaussians._xyz) * op_sigmoid(1 - gaussians.get_opacity) * args.noise_lr * xyz_lr
                    # Transform noise by covariance
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    # Add noise to positions
                    gaussians._xyz.data.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():      
            cam_uid = viewpoint_cam.uid  
            from utils.image_utils import colormap
            
            save_interval = cfg_model.settings.save_interval
            _save_this_iter = (iteration % save_interval == 0) or iteration == first_iter
            if _save_this_iter:

                output_path = os.path.join(scene.model_path, 'training_output')

                # Aux maps (normals + depth) for the training_output viz. When no
                # normal/dist regulariser is active the run auto-enables
                # `pipe.skip_aux_normal_dist`, so the training render_pkg carries
                # ZEROED normals (and the loop never saved depth here). Re-render this
                # one view with aux forced ON (save-interval only → negligible cost)
                # so the maps below are populated. If aux is already on (keep_aux or a
                # normal reg active), reuse the training render_pkg — no re-render.
                _aux_pkg = render_pkg
                if bool(getattr(pipe, 'skip_aux_normal_dist', False)):
                    _saved_skip = pipe.skip_aux_normal_dist
                    pipe.skip_aux_normal_dist = False
                    try:
                        _aux_pkg = render(viewpoint_cam, gaussians, pipe, current_bg,
                            ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                            is_training=False,
                            max_intersections_per_pixel=args.max_intersections_per_pixel)
                    finally:
                        pipe.skip_aux_normal_dist = _saved_skip

                img_name = os.path.join(output_path,  str(iteration) + '.png')
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), img_name)

                gt_name = os.path.join(output_path,  str(iteration) + '_gt.png')
                save_img_u8(gt_image.permute(1,2,0).detach().cpu().numpy(), gt_name)

                normal_name = os.path.join(output_path,  str(iteration) + '_normal.png')
                _viz_normal = _aux_pkg.get('rend_normal', rend_normal)
                save_img_u8(_viz_normal.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, normal_name)

                # Depth maps (expected / median / max-contributor) + foreground alpha.
                _viz_depth_exp = _aux_pkg.get('depth_expected', _aux_pkg.get('surf_depth', None))
                if _viz_depth_exp is not None:
                    save_img_u8(convert_gray_to_cmap(_viz_depth_exp.squeeze().detach().cpu().numpy(), map_mode='turbo', revert=False),
                                os.path.join(output_path, str(iteration) + '_depth_mean.png'))
                _viz_depth_med = _aux_pkg.get('depth_median', None)
                if _viz_depth_med is not None and _viz_depth_med.numel() > 0:
                    save_img_u8(convert_gray_to_cmap(_viz_depth_med.squeeze().detach().cpu().numpy(), map_mode='turbo', revert=False),
                                os.path.join(output_path, str(iteration) + '_depth_median.png'))
                _viz_depth_max = _aux_pkg.get('depth_max_contributor', None)
                if _viz_depth_max is not None and _viz_depth_max.numel() > 0:
                    save_img_u8(convert_gray_to_cmap(_viz_depth_max.squeeze().detach().cpu().numpy(), map_mode='turbo', revert=False),
                                os.path.join(output_path, str(iteration) + '_depth_maxcontrib.png'))
                _viz_alpha = _aux_pkg.get('rend_alpha', None)
                if _viz_alpha is not None:
                    save_img_u8(_viz_alpha.repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy(),
                                os.path.join(output_path, str(iteration) + '_alpha.png'))

                ### error image from superGS
                error_img = error_img.mean(axis=0)
                color_map = convert_gray_to_cmap(error_img.detach().cpu(), map_mode = 'jet', revert = False, vmax = 1)
                error_name = os.path.join(output_path,  str(iteration) + '_diff.png')
                save_img_u8(color_map, error_name)

                # Save decomposed renders for 3D_direct and cat modes (gaussian-only and hash-only)
                if args.method in ["3D_direct", "cat"] and ingp is not None:
                    # Gaussian-only render (hash features zeroed)
                    render_pkg_gauss = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                        beta=beta, iteration=iteration, cfg=cfg_model, decompose_mode='gaussian_only',
                        is_training=False, max_intersections_per_pixel=args.max_intersections_per_pixel)
                    gauss_image = torch.clamp(render_pkg_gauss["render"], 0.0, 1.0)
                    gauss_name = os.path.join(output_path, str(iteration) + '_gaussian.png')
                    save_img_u8(gauss_image.permute(1,2,0).detach().cpu().numpy(), gauss_name)

                    # Hash-only render (gaussian features zeroed)
                    render_pkg_hash = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                        beta=beta, iteration=iteration, cfg=cfg_model, decompose_mode='ngp_only',
                        is_training=False, max_intersections_per_pixel=args.max_intersections_per_pixel)
                    hash_image = torch.clamp(render_pkg_hash["render"], 0.0, 1.0)
                    hash_name = os.path.join(output_path, str(iteration) + '_hash.png')
                    save_img_u8(hash_image.permute(1,2,0).detach().cpu().numpy(), hash_name)

                # FiLM decomposition (full render is {iteration}.png above).
                # 3D_SH_filmres (SH base + CUDA residual MLP(gamma*H + beta)):
                #   _film_beta_only : zero hash (H=0) + kill SH => ReLU(MLP(beta))     [robust to gamma lock/act]
                #   _film_gamma_hash: beta cols = 0 + kill SH   => ReLU(MLP(gamma*H))
                #   _sv             : zero MLP residual         => ReLU(SH base)
                # NOTE: zeroing the HASH (not the gamma column) is what makes _beta_only correct under
                # --lock_gamma / --film_act sigmoid, where gamma_eff ignores/transforms the stored gamma.
                # Cat-family 'film' has no SH base (MLP output IS the color) → keep plain column-zeroing.
                if args.method in ("film", "3D_SH_filmres") and ingp is not None \
                        and hasattr(gaussians, '_film_params') and gaussians._film_params.numel() > 0:
                    _film_saved = gaussians._film_params.data.clone()
                    _is_filmres = (args.method == "3D_SH_filmres")
                    _frk = dict(ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                is_training=False,
                                max_intersections_per_pixel=args.max_intersections_per_pixel)
                    _km = {'decompose_mode': 'tex_only'} if _is_filmres else {}  # filmres: kill the SH base
                    # filmres: force gamma_eff = 0 via the device-global lock for the beta-only render.
                    # This zeros gamma*H WITHOUT zeroing the hash levels (the FiLM modulation loop is
                    # bounded by hash_dim, so active=0 would also drop beta → all-black). The lock
                    # overrides --lock_gamma AND --film_act, so beta-only is correct in every config.
                    _filmres_lock = None
                    _set_lru = None  # filmres: lru_slope=1 → identity outer activation → SIGNED residual
                    _lock_restore = float(args.lock_gamma) if getattr(args, 'lock_gamma', None) is not None else -1e30
                    _lru_restore = float(getattr(args, 'lru', 0.0) or 0.0)
                    if _is_filmres:
                        try:
                            from diff_surfel_3D_sh_filmres import set_film_lock_gamma as _filmres_lock
                            from diff_surfel_3D_sh_filmres import set_lru_slope as _set_lru
                        except Exception:
                            pass
                    # Save a residual decomp at the CURRENT lock/beta state: a normal post-ReLU image
                    # (negatives clamped to 0) AND, for filmres, an _abs image of the SIGNED residual
                    # (rendered with lru=1 so the outer activation is identity → the tex residual can
                    # go negative, |.| reveals it).
                    def _save_resid(_suffix):
                        _rp = render(viewpoint_cam, gaussians, pipe, current_bg, **_frk, **_km)
                        save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + _suffix + '.png'))
                        if _is_filmres and _set_lru is not None:
                            _set_lru(1.0)
                            try:
                                _rp = render(viewpoint_cam, gaussians, pipe, current_bg, **_frk, **_km)
                                save_img_u8(torch.clamp(_rp["render"].abs(), 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + _suffix + '_abs.png'))
                            finally:
                                _set_lru(_lru_restore)
                    try:
                        # --- beta only: residual = MLP(beta) (gamma_eff = 0) ---
                        if _is_filmres and _filmres_lock is not None:
                            _filmres_lock(0.0)                       # gamma_eff = 0 → mlp_input = beta
                        elif not _is_filmres:
                            gaussians._film_params.data[:, 0] = 0.0  # cat film: gamma = 0
                        _save_resid('_film_beta_only')
                        # --- gamma*H only: residual = MLP(gamma*H) (beta = 0, gamma restored) ---
                        if _is_filmres and _filmres_lock is not None:
                            _filmres_lock(_lock_restore)             # restore trained/locked gamma
                        gaussians._film_params.data.copy_(_film_saved)
                        gaussians._film_params.data[:, 1:] = 0.0     # beta = 0
                        _save_resid('_film_gamma_hash')
                        # --- SH/SV base only (filmres) ---
                        if _is_filmres:
                            gaussians._film_params.data.copy_(_film_saved)
                            _rp = render(viewpoint_cam, gaussians, pipe, current_bg, **_frk, decompose_mode='sh_only')
                            save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                        os.path.join(output_path, str(iteration) + '_sv.png'))
                    finally:
                        gaussians._film_params.data.copy_(_film_saved)
                        if _is_filmres and _filmres_lock is not None:
                            _filmres_lock(_lock_restore)             # ensure lock restored for training
                        if _is_filmres and _set_lru is not None:
                            _set_lru(_lru_restore)                   # ensure lru restored for training

                # --method 3D_SH_concat: full breakdown of the decode (full render is {iteration}.png).
                #   _sv     : SV/SH base only          (zero MLP residual)
                #   _latent : ReLU(MLP([latent(16)|0]))  (kill SH + zero hash → the surfel latent alone)
                #   _hash   : ReLU(MLP([0|hash(16)]))    (kill SH + zero latent → the hashgrid alone)
                # The MLP is nonlinear so _latent + _hash != full residual, but each isolates one input.
                # For the residual halves (_latent/_hash) we ALSO save an _abs image of the SIGNED
                # residual (rendered with lru=1 → identity outer activation), since the tex residual
                # can be negative and the normal post-ReLU clamps it to 0.
                if args.method == "3D_SH_concat" and ingp is not None:
                    _crk = dict(ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                is_training=False,
                                max_intersections_per_pixel=args.max_intersections_per_pixel)
                    _csl = None
                    try:
                        from diff_surfel_3D_sh_concat import set_lru_slope as _csl
                    except Exception:
                        _csl = None
                    # SV base — always >= 0, no abs version needed.
                    _rp = render(viewpoint_cam, gaussians, pipe, current_bg, decompose_mode='sh_only', **_crk)
                    save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                os.path.join(output_path, str(iteration) + '_sv.png'))
                    # Residual halves: normal (post-ReLU) + abs (signed).
                    for _dm, _suffix in (('concat_latent', '_latent'), ('concat_hash', '_hash')):
                        _rp = render(viewpoint_cam, gaussians, pipe, current_bg, decompose_mode=_dm, **_crk)
                        save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + _suffix + '.png'))
                        if _csl is not None:
                            _csl(1.0)
                            try:
                                _rp = render(viewpoint_cam, gaussians, pipe, current_bg, decompose_mode=_dm, **_crk)
                                save_img_u8(torch.clamp(_rp["render"].abs(), 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + _suffix + '_abs.png'))
                            finally:
                                _csl(0.0)

                # --method proberes: decomposition (full render is {iteration}.png above).
                #   _sv            : SV/SH base only (probe tensors withheld -> residual = 0)
                #   _probe_tex     : ReLU(residual) alone (sh_bias=-999 kills the SV base)
                #   _probe_tex_abs : |residual| blended exactly - the residual is LINEAR in
                #       the field's last Linear (and the pixel image), so negating them in
                #       place negates it, and ReLU(r) + ReLU(-r) = |r| per contribution.
                #   _probe_atlas   : the shared texture image itself (T + 0.5, clamped).
                if getattr(ingp, 'is_proberes_mode', False) and not getattr(ingp, 'hashgrid_disabled', False):
                    with torch.no_grad():
                        _prk = dict(ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                    is_training=False,
                                    max_intersections_per_pixel=args.max_intersections_per_pixel)
                        _rp = render(viewpoint_cam, gaussians, pipe, current_bg,
                                     decompose_mode='sh_only', **_prk)
                        save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + '_sv.png'))
                        _rp = render(viewpoint_cam, gaussians, pipe, current_bg,
                                     decompose_mode='tex_only', **_prk)
                        _probe_texp = _rp["render"]
                        save_img_u8(torch.clamp(_probe_texp, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + '_probe_tex.png'))
                        _pf_W = ingp.probe_field.mlp[-1]
                        _pf_px = getattr(ingp.probe_field, 'pixels', None)
                        _pf_W.weight.data.neg_(); _pf_W.bias.data.neg_()
                        if _pf_px is not None:
                            _pf_px.data.neg_()
                        try:
                            _probe_texn = render(viewpoint_cam, gaussians, pipe, current_bg,
                                                 decompose_mode='tex_only', **_prk)["render"]
                        finally:
                            _pf_W.weight.data.neg_(); _pf_W.bias.data.neg_()
                            if _pf_px is not None:
                                _pf_px.data.neg_()
                        save_img_u8(torch.clamp(_probe_texp + _probe_texn, 0.0, 1.0)
                                    .permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + '_probe_tex_abs.png'))
                        _probe_atlas = (ingp.probe_field.bake(sparse_bw=False) + 0.5).clamp(0.0, 1.0)
                        save_img_u8(_probe_atlas.cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + '_probe_atlas.png'))
                        # Probe patch-size stats: rho = sqrt(|det A|) texels/sigma,
                        # patch = 6*rho texels per +-3sigma. THE health metric for
                        # "are surfels actually sampling texture variation".
                        _pp = ingp.probe_head(gaussians.get_xyz, gaussians.get_rotation,
                                              gaussians.get_scaling)
                        _rho = (_pp[:, 0] * _pp[:, 3] - _pp[:, 1] * _pp[:, 2]).abs().sqrt()
                        _patch = 6.0 * _rho
                        _q = torch.quantile(_patch, torch.tensor([0.1, 0.5, 0.9], device=_patch.device))
                        tqdm.write(f"[PROBE iter={iteration}] patch texels per +-3sigma: "
                                   f"p10={_q[0]:.2f} p50={_q[1]:.2f} p90={_q[2]:.2f} "
                                   f"(log_smed={float(ingp.probe_head.log_smed):.3f}, "
                                   f"frozen={ingp.probe_head.smed_frozen})")
                        # Probe-rectangle overlay: each surfel's +-3sigma uv square
                        # mapped through its affine -> rotated rect in texture px,
                        # outlined over the atlas in a per-surfel pseudo-random color.
                        _R = ingp.probe_field.tex_res
                        _N = _pp.shape[0]
                        _A = _pp[:, :4].view(_N, 2, 2)
                        _t = _pp[:, 4:6]
                        _cor = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]],
                                            device=_pp.device)
                        _c = torch.einsum('nij,cj->nci', _A, _cor) + _t[:, None, :]   # [N,4,2]
                        _nx = _c.roll(-1, dims=1)
                        _sf = torch.linspace(0.0, 1.0, 8, device=_pp.device)
                        _pts = (_c[:, :, None, :] + (_nx - _c)[:, :, None, :] * _sf[None, None, :, None])
                        _idx = torch.arange(_N, device=_pp.device, dtype=torch.float32)
                        _col = torch.stack([(_idx * 0.7548).frac(), (_idx * 0.5698).frac(),
                                            (_idx * 0.3213).frac()], dim=-1) * 0.7 + 0.3   # [N,3]
                        _col = _col[:, None, None, :].expand(-1, 4, 8, -1).reshape(-1, 3)
                        _pts = _pts.reshape(-1, 2)
                        _xs = _pts[:, 0].round().long()
                        _ys = _pts[:, 1].round().long()
                        _ok = (_xs >= 0) & (_xs < _R) & (_ys >= 0) & (_ys < _R)
                        _ov = _probe_atlas.clone()
                        _ov[_ys[_ok], _xs[_ok]] = _col[_ok]
                        save_img_u8(_ov.cpu().numpy(),
                                    os.path.join(output_path, str(iteration) + '_probe_atlas_probes.png'))
                        del _probe_texp, _probe_texn, _probe_atlas, _pp, _ov

                # Save contributor heatmap every 5k iterations
                if iteration % 5000 == 0:
                    gs_num = render_pkg.get("gaussian_num", None)
                    if gs_num is not None:
                        heatmap, min_c, max_c = create_intersection_heatmap(gs_num, max_display=100)
                        heatmap_name = os.path.join(output_path, str(iteration) + '_contributors.png')
                        save_img_u8(heatmap, heatmap_name)

                # `--method mixed` / `--method res_3d`: three-way decomposition
                # — textured-spherical (SV/SH baseline of the textured half),
                # textured-residual (MLP residual of the textured half),
                # untextured-spherical (SV baseline of the untextured half).
                # Per-half isolation via a temporary opacity mask (untextured/
                # textured opacity → ~0), restored in a finally block.
                # For `--method res_3d` with the per-Gauss bias gate enabled,
                # tex_sp will be ~zero (textured carriers force sh_color = 0),
                # tex_res shows the pure residual, and untex_sp shows the EWA
                # SV image.
                if args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "res_3d", "res_3d_paired", "res_3d_double") and ingp is not None and not getattr(ingp, 'hashgrid_disabled', False):
                    with torch.no_grad():
                        _is_tex = getattr(gaussians, '_is_textured', None)
                        _Np = gaussians.get_xyz.shape[0]
                        if _is_tex is not None and _is_tex.numel() == _Np:
                            _saved_op = gaussians._opacity.data.clone()
                            _OFF = -50.0  # sigmoid(-50) ≈ 0 → invisible
                            _mrk = {'beta': beta, 'iteration': iteration, 'cfg': cfg_model,
                                    'is_training': False}
                            try:
                                # tex_sp: textured half only, residual zeroed (sh_only) → ReLU(SV)
                                gaussians._opacity.data.copy_(_saved_op)
                                gaussians._opacity.data[~_is_tex] = _OFF
                                _rp = render(viewpoint_cam, gaussians, pipe, current_bg,
                                             ingp=ingp, decompose_mode='sh_only', **_mrk)
                                save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0)
                                            .permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + '_tex_sp.png'))

                                # tex_res: textured half only, SV killed (tex_only) → residual.
                                # The MLP residual is SIGNED (residual_mode 2); the per-pixel
                                # final ReLU in render() clamps negative residual to 0, so the
                                # plain `tex_res` view hides every subtractive (negative) texel.
                                # Use the pre-ReLU `render_raw` to also dump abs() (magnitude,
                                # incl. negatives) and a signed view (gray 0.5 = zero,
                                # bright = additive, dark = subtractive), mirroring the
                                # 3D_SH_res decomposition.
                                gaussians._opacity.data.copy_(_saved_op)
                                gaussians._opacity.data[~_is_tex] = _OFF
                                _rp = render(viewpoint_cam, gaussians, pipe, current_bg,
                                             ingp=ingp, decompose_mode='tex_only', **_mrk)
                                save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0)
                                            .permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + '_tex_res.png'))
                                _tex_res_raw = _rp.get("render_raw", _rp["render"])
                                save_img_u8(torch.clamp(_tex_res_raw.abs(), 0.0, 1.0)
                                            .permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + '_tex_res_abs.png'))
                                save_img_u8(torch.clamp(_tex_res_raw * 2.0 + 0.5, 0.0, 1.0)
                                            .permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + '_tex_res_signed.png'))

                                # untex_sp: untextured half only (kernel forces feat=ReLU(SV),
                                # no residual by design — decompose_mode=None is correct).
                                gaussians._opacity.data.copy_(_saved_op)
                                gaussians._opacity.data[_is_tex] = _OFF
                                _rp = render(viewpoint_cam, gaussians, pipe, current_bg,
                                             ingp=ingp, decompose_mode=None, **_mrk)
                                save_img_u8(torch.clamp(_rp["render"], 0.0, 1.0)
                                            .permute(1, 2, 0).detach().cpu().numpy(),
                                            os.path.join(output_path, str(iteration) + '_untex_sp.png'))
                            finally:
                                gaussians._opacity.data.copy_(_saved_op)
                        else:
                            tqdm.write(f"[DECOMPOSE {iteration}] mixed: _is_textured not "
                                       f"populated (pre-split) — skipping tex/untex split renders")

                # GEStex sort-free JOINT stage: 5-image decomposition —
                #   full | surfels(SV+texture) | surfel SV-base | surfel texture | 3DGS SV.
                # "SH only" here means the per-surfel feature (SV in this run), not literal SH.
                elif getattr(ingp, 'is_gestex_joint', False):
                    # 5-image decomposition via OPACITY-MASKING (works for both the sort-free
                    # AND --ges_no_bake cascade paths): to isolate one set, drive the OTHER
                    # set's opacity to ~0. Surfels carry the rising opacity floor, so hiding
                    # them also needs ges_opac_floor→0; the untextured Gaussians have no floor.
                    from utils.general_utils import inverse_sigmoid as _isig
                    def _gsave(_im, _nm):
                        save_img_u8(torch.clamp(_im, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(),
                                    os.path.join(output_path, f'{iteration}_{_nm}.png'))
                    def _rimg(_dm=None):
                        with torch.no_grad():
                            return render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp, beta=beta,
                                          iteration=iteration, cfg=cfg_model, decompose_mode=_dm, is_training=False)['render']
                    _smask = gaussians._is_textured
                    _has_split = (_smask.numel() == gaussians._opacity.shape[0])
                    _saved_op = gaussians._opacity.data.clone()
                    _saved_floor = float(getattr(gaussians, 'ges_opac_floor', 0.0))
                    _OFF = float(_isig(torch.tensor(1e-4)).item())  # sigmoid(_OFF) ≈ 1e-4 → alpha < 1/255 (culled)
                    try:
                        _full = _rimg(None)
                        _gsave(_full, 'full')
                        save_img_u8(torch.clamp(_full, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy(), img_name)
                        if _has_split:
                            # surfel-only: hide the 3DGS (opacity → ~0). Surfels keep their floor.
                            gaussians._opacity.data.copy_(_saved_op)
                            gaussians._opacity.data[~_smask] = _OFF
                            _gsave(_rimg(None),      'surfel_only')          # surfels: SV + texture
                            _gsave(_rimg('sh_only'), 'surfel_sh_only')       # surfels: SV base only
                            _tex = _rimg('tex_only')
                            _gsave(_tex,             'surfel_texture_only')  # surfels: texture only
                            save_img_u8(torch.clamp(_tex * 2.0 + 0.5, 0.0, 1.0)
                                        .permute(1, 2, 0).detach().cpu().numpy(),
                                        os.path.join(output_path, f'{iteration}_surfel_texture_signed.png'))
                            # 3DGS-only: hide surfels (opacity → ~0 AND floor → 0 so it can't lift them).
                            gaussians._opacity.data.copy_(_saved_op)
                            gaussians._opacity.data[_smask] = _OFF
                            gaussians.ges_opac_floor = 0.0
                            _gsave(_rimg(None),      'gaussian_sh_only')     # 3DGS only
                    finally:
                        gaussians._opacity.data.copy_(_saved_op)
                        gaussians.ges_opac_floor = _saved_floor

                # Save decomposed renders for 3D_SH_res and 3D_SH_cat modes (SH-only and texture-only)
                # All renders done atomically with the same model state (post-optimizer-step)
                elif args.method in ["3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "3D_SH_filmres"] and ingp is not None and not getattr(ingp, 'hashgrid_disabled', False):
                    # Full render (consistent with decomposition renders below)
                    with torch.no_grad():
                        render_pkg_full = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                            beta=beta, iteration=iteration, cfg=cfg_model, decompose_mode=None,
                            is_training=False)
                        full_raw = render_pkg_full["render"]

                    # Overwrite the main image with the consistent full render
                    full_image = torch.clamp(full_raw, 0.0, 1.0)
                    save_img_u8(full_image.permute(1,2,0).detach().cpu().numpy(), img_name)

                    # SH-only render (hashgrid disabled, residual ≈ 0)
                    render_pkg_sh = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                        beta=beta, iteration=iteration, cfg=cfg_model, decompose_mode='sh_only',
                        is_training=False)
                    sh_raw = render_pkg_sh["render"]
                    sh_image = torch.clamp(sh_raw, 0.0, 1.0)
                    sh_name = os.path.join(output_path, str(iteration) + '_sh_only.png')
                    save_img_u8(sh_image.permute(1,2,0).detach().cpu().numpy(), sh_name)

                    # Texture-only render (SH zeroed, only MLP residual)
                    render_pkg_tex = render(viewpoint_cam, gaussians, pipe, current_bg, ingp=ingp,
                        beta=beta, iteration=iteration, cfg=cfg_model, decompose_mode='tex_only',
                        is_training=False)
                    tex_raw = render_pkg_tex["render"]
                    tex_image = torch.clamp(tex_raw, 0.0, 1.0)
                    tex_name = os.path.join(output_path, str(iteration) + '_tex_only.png')
                    save_img_u8(tex_image.permute(1,2,0).detach().cpu().numpy(), tex_name)

                    # True residual = full - sh_only.
                    # Mode 0 (3D_SH_res): outer ReLU couples SH and residual, so
                    #   the residual contribution can be negative where hash
                    #   subtracts from SH — abs / signed visualizations are useful.
                    # Mode 1 (3D_SH_add): per-Gaussian ReLU(residual+bias) ≥ 0 and
                    #   alpha-blending preserves sign, so abs / signed are
                    #   redundant with `_tex_only` itself — skip both.
                    residual_true = full_raw - sh_raw
                    if getattr(args, '_residual_mode', 0) == 0:
                        # Absolute value (magnitude of residual contribution)
                        tex_abs = torch.clamp(residual_true.abs(), 0.0, 1.0)
                        tex_abs_name = os.path.join(output_path, str(iteration) + '_tex_only_abs.png')
                        save_img_u8(tex_abs.permute(1,2,0).detach().cpu().numpy(), tex_abs_name)
                        # Signed: gray(0.5)=zero, bright=positive residual, dark=negative/subtractive
                        tex_signed = torch.clamp(residual_true * 2.0 + 0.5, 0.0, 1.0)
                        tex_signed_name = os.path.join(output_path, str(iteration) + '_tex_residual_signed.png')
                        save_img_u8(tex_signed.permute(1,2,0).detach().cpu().numpy(), tex_signed_name)

                    # Save SH+residual summed image (should match full render exactly)
                    sum_image = torch.clamp(sh_raw + tex_raw, 0.0, 1.0)
                    sum_name = os.path.join(output_path, str(iteration) + '_sh_plus_tex.png')
                    save_img_u8(sum_image.permute(1,2,0).detach().cpu().numpy(), sum_name)

                    # Debug: decomposition stats
                    if gaussians._features_dc.numel() == 0:
                        continue
                    dc_max = gaussians._features_dc.data.abs().max().item()
                    rest_max = gaussians._features_rest.data.abs().max().item() if gaussians._features_rest.numel() > 0 else 0.0
                    tqdm.write(f"[DECOMPOSE {iteration}] SH DC max={dc_max:.4f}, REST max={rest_max:.4f}, "
                              f"sh_degree={gaussians.active_sh_degree}")
                    tqdm.write(f"[DECOMPOSE {iteration}] full: mean={full_raw.mean():.4f}, "
                              f"sh_only: mean={sh_raw.mean():.4f}, tex_only: mean={tex_raw.mean():.4f}, "
                              f"sh+tex: mean={(sh_raw + tex_raw).mean():.4f}")
                    tqdm.write(f"[DECOMPOSE {iteration}] full_vs_decompose diff: {(full_raw - sh_raw - tex_raw).abs().mean():.6f}")
                    # Compare alpha maps
                    alpha_full = render_pkg_full.get("rend_alpha", None)
                    alpha_sh = render_pkg_sh.get("rend_alpha", None)
                    alpha_tex = render_pkg_tex.get("rend_alpha", None)
                    if alpha_full is not None and alpha_sh is not None and alpha_tex is not None:
                        tqdm.write(f"[DECOMPOSE {iteration}] alpha: full={alpha_full.mean():.4f}, "
                                  f"sh_only={alpha_sh.mean():.4f}, tex_only={alpha_tex.mean():.4f}, "
                                  f"full_vs_sh_diff={((alpha_full - alpha_sh).abs().mean()):.6f}, "
                                  f"full_vs_tex_diff={((alpha_full - alpha_tex).abs().mean()):.6f}")
                    # Save diff image (amplified 10x for visibility)
                    diff_raw = (full_raw - sh_raw - tex_raw).abs() * 10
                    diff_image = torch.clamp(diff_raw, 0.0, 1.0)
                    diff_name = os.path.join(output_path, str(iteration) + '_decompose_diff_10x.png')
                    save_img_u8(diff_image.permute(1,2,0).detach().cpu().numpy(), diff_name)

                # Save FG/BG separation if skybox is active
                if "render_fg" in render_pkg:
                    fg_image = torch.clamp(render_pkg["render_fg"], 0.0, 1.0)
                    bg_image = torch.clamp(render_pkg["render_bg"], 0.0, 1.0)
                    alpha = render_pkg["rend_alpha"]

                    fg_name = os.path.join(output_path, str(iteration) + '_fg.png')
                    save_img_u8(fg_image.permute(1,2,0).detach().cpu().numpy(), fg_name)

                    bg_name = os.path.join(output_path, str(iteration) + '_bg.png')
                    save_img_u8(bg_image.permute(1,2,0).detach().cpu().numpy(), bg_name)

                    alpha_name = os.path.join(output_path, str(iteration) + '_alpha.png')
                    save_img_u8(alpha.repeat(3,1,1).permute(1,2,0).detach().cpu().numpy(), alpha_name)

            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer, ingp = ingp, \
                            beta = beta)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        with torch.no_grad():

            # At initialize iteration, optionally generate gs_alpha masks and save warmup checkpoint
            # Skip in cold start mode since there's no warmup phase
            if iteration == cfg_model.ingp_stage.initialize and not args.cold:
                gs_alpha_masks = {}

                # Optionally generate gs_alpha masks if enabled
                if cfg_model.settings.gs_alpha:
                    print('--- Generating mask by 2DGS.')
                    from utils.image_utils import bilateral_filter_opencv

                    if not os.path.exists(os.path.join(scene.model_path, "gs_alpha")):
                        os.mkdir(os.path.join(scene.model_path, "gs_alpha"))

                    if not os.path.exists(os.path.join(scene.model_path, "gt_alpha")):
                        os.mkdir(os.path.join(scene.model_path, "gt_alpha"))

                    # Collect gs_alpha masks for all cameras
                    train_stack = scene.getTrainCameras()
                    for cam in tqdm(train_stack, desc="Generating masks"):
                        cam_name = cam.image_name + '.png'
                        render_pkg = render(cam, gaussians, pipe, background, ingp = ingp, \
                            beta = beta, iteration = iteration, cfg = cfg_model)
                        alpha_image = render_pkg["rend_alpha"]
                        bila_alpha = bilateral_filter_opencv(alpha_image.detach().cpu())
                        cam.gs_alpha_mask = bila_alpha.cpu().float()
                        gs_alpha_masks[cam.image_name] = bila_alpha.cpu().float()

                        alpha_name = os.path.join(scene.model_path, 'gs_alpha', cam_name)
                        save_img_u8(bila_alpha.permute(1,2,0).expand(-1,-1,3).numpy(), alpha_name)

                        alpha_name = os.path.join(scene.model_path, 'gt_alpha', cam_name)
                        save_img_u8(cam.gt_alpha_mask.permute(1,2,0).expand(-1,-1,3).detach().cpu().numpy(), alpha_name)

                # Save warmup checkpoint (only if training from scratch and if_ingp is enabled)
                if not loaded_from_warmup and cfg_model.settings.if_ingp:
                    print("\n" + "="*70)
                    print("  SAVING 2DGS WARMUP CHECKPOINT")
                    print("="*70)

                    warmup_ckpt = {
                        'iteration': iteration,
                        'n_gaussians': len(gaussians.get_xyz),
                        'active_sh_degree': gaussians.active_sh_degree,
                        'xyz': gaussians._xyz.detach().cpu(),
                        'features_dc': gaussians._features_dc.detach().cpu(),
                        'features_rest': gaussians._features_rest.detach().cpu(),
                        'scaling': gaussians._scaling.detach().cpu(),
                        'rotation': gaussians._rotation.detach().cpu(),
                        'opacity': gaussians._opacity.detach().cpu(),
                        'appearance_level': gaussians._appearance_level.detach().cpu(),
                        'max_radii2D': gaussians.max_radii2D.detach().cpu(),
                        'spatial_lr_scale': gaussians.spatial_lr_scale,
                        'optimizer_state': gaussians.optimizer.state_dict(),
                        'gs_alpha_masks': gs_alpha_masks,
                        # Densification state - needed for identical behavior
                        'xyz_gradient_accum': gaussians.xyz_gradient_accum.detach().cpu(),
                        'feat_gradient_accum': gaussians.feat_gradient_accum.detach().cpu(),
                        'denom': gaussians.denom.detach().cpu(),
                    }

                    torch.save(warmup_ckpt, warmup_checkpoint_path)
                    print(f"  Saved to: {warmup_checkpoint_path}")
                    print(f"  Gaussians: {warmup_ckpt['n_gaussians']}")
                    print(f"  GS Alpha masks: {len(gs_alpha_masks)}")
                    print("  Next run will skip 2DGS phase and resume from here.")
                    print("="*70 + "\n")

            # === Save SHARED-RESUME CHECKPOINT at iter == args.share_ckpt_iter ===
            # Captures the FULL state (Gaussians + their optimizer + densif accumulators
            # + INGP weights + INGP optimizer + gs_alpha_masks) so a sweep over
            # downstream-only knobs can resume from this exact snapshot.
            if (args.share_ckpt_iter > 0 and iteration == args.share_ckpt_iter
                    and not loaded_from_shared and share_ckpt_path is not None):
                # Avoid double-save if the file appeared concurrently.
                if os.path.exists(share_ckpt_path):
                    print(f"\n[SHARED CKPT] {share_ckpt_path} already exists — skipping save.")
                else:
                    print("\n" + "="*70)
                    print(f"  SAVING SHARED-RESUME CHECKPOINT (iter {iteration})")
                    print("="*70)

                    # Use the gs_alpha_masks defined above when iteration ==
                    # cfg_model.ingp_stage.initialize; otherwise fall back to per-camera
                    # masks accumulated on the cam objects (or empty when not used).
                    _share_gs_alpha = locals().get('gs_alpha_masks', None)
                    if _share_gs_alpha is None:
                        _share_gs_alpha = {}
                        for _cam in scene.getTrainCameras():
                            _m = getattr(_cam, 'gs_alpha_mask', None)
                            if _m is not None:
                                _share_gs_alpha[_cam.image_name] = _m.cpu().float()

                    share_ckpt = {
                        'iteration': iteration,
                        'n_gaussians': len(gaussians.get_xyz),
                        'active_sh_degree': gaussians.active_sh_degree,
                        # --- Gaussian parameters (all on CPU) ---
                        'xyz':            gaussians._xyz.detach().cpu(),
                        'features_dc':    gaussians._features_dc.detach().cpu(),
                        'features_rest':  gaussians._features_rest.detach().cpu(),
                        'scaling':        gaussians._scaling.detach().cpu(),
                        'rotation':       gaussians._rotation.detach().cpu(),
                        'opacity':        gaussians._opacity.detach().cpu(),
                        'appearance_level': gaussians._appearance_level.detach().cpu(),
                        'max_radii2D':    gaussians.max_radii2D.detach().cpu(),
                        'spatial_lr_scale': gaussians.spatial_lr_scale,
                        # --- Gaussian optimizer state (Adam moments) ---
                        'gaussians_optimizer_state': gaussians.optimizer.state_dict(),
                        # --- Densification accumulators ---
                        'xyz_gradient_accum':  gaussians.xyz_gradient_accum.detach().cpu(),
                        'feat_gradient_accum': gaussians.feat_gradient_accum.detach().cpu(),
                        'denom':               gaussians.denom.detach().cpu(),
                        # --- gs_alpha masks (per-camera) ---
                        'gs_alpha_masks': _share_gs_alpha,
                    }

                    # Method/kernel-specific params (saved only when present)
                    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                        share_ckpt['shape'] = gaussians._shape.detach().cpu()
                    if hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0:
                        share_ckpt['flex_beta'] = gaussians._flex_beta.detach().cpu()
                    if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features.numel() > 0:
                        share_ckpt['gaussian_features'] = gaussians._gaussian_features.detach().cpu()
                        share_ckpt['gaussian_feat_dim'] = getattr(gaussians, '_gaussian_feat_dim', 0)

                    # INGP model + optimizer (when present — same gate as warmup)
                    if cfg_model.settings.if_ingp and ingp_model is not None:
                        share_ckpt['ingp_state_dict'] = {k: v.detach().cpu() for k, v in ingp_model.state_dict().items()}
                        if hasattr(ingp_model, 'optimizer') and ingp_model.optimizer is not None:
                            share_ckpt['ingp_optimizer_state'] = ingp_model.optimizer.state_dict()

                    torch.save(share_ckpt, share_ckpt_path)
                    print(f"  Saved to: {share_ckpt_path}")
                    print(f"  Gaussians: {share_ckpt['n_gaussians']}")
                    print(f"  GS alpha masks: {len(_share_gs_alpha)}")
                    print(f"  INGP weights+optimizer: {'yes' if 'ingp_state_dict' in share_ckpt else 'no'}")
                    print(f"  Next run with --share_ckpt_iter {args.share_ckpt_iter} will resume from here.")
                    print("="*70 + "\n")

        torch.cuda.empty_cache()

    # Prune dead Gaussians in MCMC mode before final rendering
    if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
        print("\n" + "="*70)
        print("  PRUNING DEAD GAUSSIANS (MCMC MODE)")
        print("="*70)
        n_total_before = len(gaussians.get_xyz)
        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
        n_dead = dead_mask.sum().item()
        n_alive = n_total_before - n_dead
        print(f"  Total Gaussians: {n_total_before}")
        print(f"  Alive (opacity > 0.005): {n_alive}")
        print(f"  Dead (opacity <= 0.005): {n_dead}")

        if n_dead > 0:
            gaussians.prune_points(dead_mask)
            print(f"  Pruned {n_dead} dead Gaussians")
            print(f"  Remaining: {len(gaussians.get_xyz)}")
        else:
            print(f"  No dead Gaussians to prune")

        # Save final PLY (always, since we skipped the in-loop save for MCMC)
        print(f"  Saving final point cloud...")
        scene.save(iteration)
        print("="*70 + "\n")

    # Final test and train rendering with stride 1
    final_ingp = ingp_model if ingp_model is not None else ingp

    # For random_background mode, use black background during evaluation
    eval_background = black_bg if use_random_bg else background
    
    print("\n" + "="*70)
    print(" "*20 + "FINAL TEST RENDERING")
    print("="*70)
    render_final_images(scene, gaussians, pipe, eval_background, final_ingp, beta, iteration, cfg_model, args,
                        cameras=scene.getTestCameras(), output_subdir='final_test_renders', metrics_file='test_metrics.txt',
                        skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid,
                        ppisp_apply=(_ppisp_apply if ppisp is not None else None))

    print("\n" + "="*70)
    print(" "*20 + "FINAL TRAIN RENDERING")
    print("="*70)
    render_final_images(scene, gaussians, pipe, eval_background, final_ingp, beta, iteration, cfg_model, args,
                        cameras=scene.getTrainCameras(), output_subdir='final_train_renders', metrics_file='train_metrics.txt',
                        stride=25, skip_decomposition=True, skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid,
                        ppisp_apply=(_ppisp_apply if ppisp is not None else None))
    
    # Save training log with point count and framerate
    save_training_log(scene, gaussians, final_ingp, pipe, args, cfg_model, iteration, training_start_time)


def save_training_log(scene, gaussians, ingp, pipe, args, cfg_model, iteration, training_start_time=None):
    """Save training statistics to training_log.txt."""
    import time

    log_path = os.path.join(scene.model_path, 'training_log.txt')

    # Count Gaussians
    num_gaussians = len(gaussians.get_xyz)

    # Compute weight distribution for adaptive_zero mode
    weight_below_01 = None
    weight_above_09 = None
    if args.method == "adaptive_zero" and hasattr(gaussians, '_adaptive_zero_weight'):
        with torch.no_grad():
            weights = torch.sigmoid(gaussians._adaptive_zero_weight).squeeze()
            weight_below_01 = (weights < 0.1).sum().item()
            weight_above_09 = (weights > 0.9).sum().item()
            pct_below_01 = 100.0 * weight_below_01 / num_gaussians
            pct_above_09 = 100.0 * weight_above_09 / num_gaussians
            print(f"[LOG] Adaptive weight distribution:")
            print(f"[LOG]   Below 0.1 (Gaussian-only): {weight_below_01:,} ({pct_below_01:.1f}%)")
            print(f"[LOG]   Above 0.9 (Hash-enhanced): {weight_above_09:,} ({pct_above_09:.1f}%)")
            print(f"[LOG]   Middle (0.1-0.9):          {num_gaussians - weight_below_01 - weight_above_09:,} ({100 - pct_below_01 - pct_above_09:.1f}%)")

    # Compute gate distribution for adaptive_gate mode (hard threshold at 0.5)
    gate_below_05 = None
    gate_above_05 = None
    if args.method == "adaptive_gate" and hasattr(gaussians, '_gate_logits'):
        with torch.no_grad():
            gate_prob = torch.sigmoid(gaussians._gate_logits).squeeze()
            gate_below_05 = (gate_prob <= 0.5).sum().item()
            gate_above_05 = (gate_prob > 0.5).sum().item()
            pct_below_05 = 100.0 * gate_below_05 / num_gaussians
            pct_above_05 = 100.0 * gate_above_05 / num_gaussians
            print(f"[LOG] Gate distribution (threshold=0.5):")
            print(f"[LOG]   Gate closed (<=0.5, Gaussian-only): {gate_below_05:,} ({pct_below_05:.1f}%)")
            print(f"[LOG]   Gate open (>0.5, uses hash):        {gate_above_05:,} ({pct_above_05:.1f}%)")
    
    # Estimate framerate by timing a few renders
    print("\n[LOG] Measuring render framerate...")
    test_cameras = scene.getTestCameras()
    if len(test_cameras) > 0:
        # Warm up
        viewpoint = test_cameras[0]
        bg = torch.zeros(3, device="cuda")
        beta = cfg_model.surfel.tg_beta
        
        with torch.no_grad():
            # Warm-up render
            _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, aabb_mode=args.aabb)
            torch.cuda.synchronize()

            # Time multiple renders
            num_timing_iters = 100
            start_time = time.time()
            for _ in range(num_timing_iters):
                _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                          iteration=iteration, cfg=cfg_model, aabb_mode=args.aabb)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            fps = num_timing_iters / elapsed
            ms_per_frame = (elapsed / num_timing_iters) * 1000
    else:
        fps = 0
        ms_per_frame = 0
    
    # Get resolution
    if len(test_cameras) > 0:
        H, W = test_cameras[0].image_height, test_cameras[0].image_width
        resolution = f"{W}x{H}"
    else:
        resolution = "unknown"
    
    # Write log
    with open(log_path, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Method: {args.method}\n")
        if args.method in ["cat", "cat_dropout", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_cat", "3D_SH_32", "3D_SH_filmres"]:
            f.write(f"Hybrid Levels: {args.hybrid_levels}\n")
            if args.method == "cat_dropout":
                f.write(f"Dropout Lambda: {args.dropout_lambda}\n")
        if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
            mode_name = "MCMC Deficit" if args.mcmc_deficit else "MCMC"
            f.write(f"{mode_name} Mode: Enabled\n")
            f.write(f"  - Opacity Reg: {args.opacity_reg}\n")
            f.write(f"  - Scale Reg: {args.scale_reg}\n")
            f.write(f"  - Noise LR: {args.noise_lr}\n")
            f.write(f"  - Cap Max: {args.cap_max}\n")
            if args.mcmc_deficit:
                f.write(f"  - Deficit mode: deletes dead Gaussians until reaching cap_max\n")
            f.write(f"  - Note: Dead Gaussians (opacity ≤ 0.005) pruned before final rendering\n")
        if args.bce:
            f.write(f"BCE Opacity Regularization: Enabled\n")
            f.write(f"  - Lambda: {args.bce_lambda}\n")
            f.write(f"  - Active for last {args.bce_iter} iterations\n")
            if args.bce_solo_adaptive:
                f.write(f"  - Adaptive threshold: median opacity at BCE start (solo)\n")
            elif args.bce_adaptive:
                f.write(f"  - Adaptive threshold: median opacity at BCE start (with reg)\n")
            elif args.bce_solo:
                f.write(f"  - Solo mode: opacity/scale reg disabled during BCE\n")
        f.write(f"Iterations: {iteration}\n")
        f.write(f"Resolution: {resolution}\n\n")

        f.write("Model Statistics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of Gaussians: {num_gaussians:,}\n")
        if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
            f.write(f"  (MCMC: only alive Gaussians counted)\n")
        # `--method mixed`: textured (SV + hash/MLP) vs untextured (SV-only 2DGS) split
        if args.method in ("mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
            _it = getattr(gaussians, '_is_textured', None)
            if _it is not None and _it.numel() == num_gaussians and num_gaussians > 0:
                _tex = int(_it.sum().item())
                _untex = num_gaussians - _tex
                f.write(f"  Textured   (SV + hash/MLP):  {_tex:,} "
                        f"({100.0 * _tex / num_gaussians:.2f}%)\n")
                f.write(f"  Untextured (SV-only 2DGS):   {_untex:,} "
                        f"({100.0 * _untex / num_gaussians:.2f}%)\n")
            else:
                f.write(f"  (pre-split: all {num_gaussians:,} textured)\n")
        # `--method res_3d` / `--method res_3d_paired`: 2D residual-carrier (no SV) vs
        # 3D EWA SV-carrier (no residual) split
        if args.method in ("res_3d", "res_3d_paired", "res_3d_double"):
            _it = getattr(gaussians, '_is_textured', None)
            if _it is not None and _it.numel() == num_gaussians and num_gaussians > 0:
                _tex = int(_it.sum().item())
                _untex = num_gaussians - _tex
                f.write(f"  2D residual-carriers (hash/MLP, no SV):     {_tex:,} "
                        f"({100.0 * _tex / num_gaussians:.2f}%)\n")
                f.write(f"  3D EWA SV-carriers (SV-only, no residual):  {_untex:,} "
                        f"({100.0 * _untex / num_gaussians:.2f}%)\n")
            else:
                f.write(f"  (pre-split: all {num_gaussians:,} unified 2DGS Gaussians)\n")
        f.write("\n")

        # Weight distribution for adaptive_zero mode
        if weight_below_01 is not None:
            pct_below = 100.0 * weight_below_01 / num_gaussians
            pct_above = 100.0 * weight_above_09 / num_gaussians
            pct_middle = 100 - pct_below - pct_above
            f.write("Adaptive Weight Distribution\n")
            f.write("-" * 30 + "\n")
            f.write(f"Below 0.1 (Gaussian-only): {weight_below_01:,} ({pct_below:.1f}%)\n")
            f.write(f"Above 0.9 (Hash-enhanced): {weight_above_09:,} ({pct_above:.1f}%)\n")
            f.write(f"Middle (0.1-0.9):          {num_gaussians - weight_below_01 - weight_above_09:,} ({pct_middle:.1f}%)\n")
            f.write("\n")

        # Gate distribution for adaptive_gate mode
        if gate_below_05 is not None:
            pct_closed = 100.0 * gate_below_05 / num_gaussians
            pct_open = 100.0 * gate_above_05 / num_gaussians
            f.write("Gate Distribution (threshold=0.5)\n")
            f.write("-" * 30 + "\n")
            f.write(f"Gate closed (<=0.5, Gaussian-only): {gate_below_05:,} ({pct_closed:.1f}%)\n")
            f.write(f"Gate open (>0.5, uses hash):        {gate_above_05:,} ({pct_open:.1f}%)\n")
            f.write("\n")

        f.write("Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"Render FPS: {fps:.2f}\n")
        f.write(f"Time per frame: {ms_per_frame:.2f} ms\n")
        if training_start_time is not None:
            total_seconds = time.time() - training_start_time
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            f.write(f"Total training time: {hours}h {minutes}m {seconds}s ({total_seconds:.1f}s)\n")

    print(f"[LOG] Training log saved to: {log_path}")
    print(f"[LOG] Number of Gaussians: {num_gaussians:,}")
    print(f"[LOG] Render FPS: {fps:.2f} ({ms_per_frame:.2f} ms/frame)")
    if training_start_time is not None:
        total_seconds = time.time() - training_start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        print(f"[LOG] Total training time: {hours}h {minutes}m {seconds}s")


def render_final_images(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args,
                        cameras, output_subdir, metrics_file, stride=1, skip_decomposition=False,
                        skybox=None, background_mode="none", bg_hashgrid=None, ppisp_apply=None):
    """Render images and compute metrics.

    Cameras are sorted by image_name (e.g., r_0, r_1, ..., r_99) for consistent ordering
    regardless of how they were shuffled during training.
    """
    # Sort cameras by image_name for consistent ordering
    # Extract numeric part from names like "r_23" for proper numeric sorting
    def get_sort_key(cam):
        name = cam.image_name if hasattr(cam, 'image_name') else ""
        # Try to extract number from name like "r_23"
        import re
        match = re.search(r'(\d+)', name)
        if match:
            return int(match.group(1))
        return name  # Fall back to string sorting

    cameras = sorted(cameras, key=get_sort_key)
    print(f"[FINAL] Sorted {len(cameras)} cameras by image_name")

    final_output_dir = os.path.join(scene.model_path, output_subdir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Create depth output directory
    depth_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'depths'))
    os.makedirs(depth_output_dir, exist_ok=True)

    # Create intersection heatmap output directory
    intersection_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'intersection'))
    os.makedirs(intersection_output_dir, exist_ok=True)

    # Create flex beta heatmap output directory (if using flex kernel)
    is_flex_kernel = hasattr(gaussians, 'kernel_type') and gaussians.kernel_type == "flex" and hasattr(gaussians, '_flex_beta') and gaussians._flex_beta.numel() > 0
    if is_flex_kernel:
        flex_beta_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'flex_beta'))
        os.makedirs(flex_beta_output_dir, exist_ok=True)

    # Check if cat mode decomposition should be done (also for cat_dropout mode)
    is_cat_mode = ((ingp is not None and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode) or
                   (ingp is not None and hasattr(ingp, 'is_cat_dropout_mode') and ingp.is_cat_dropout_mode)) \
                   and hasattr(args, 'hybrid_levels') and args.hybrid_levels > 0
    total_levels = ingp.levels if ingp is not None else 0
    
    # Check if hybrid_SH, hybrid_SH_raw, or hybrid_SH_post mode decomposition should be done
    is_hybrid_sh_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_mode') and ingp.is_hybrid_sh_mode)
    is_hybrid_sh_raw_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_raw_mode') and ingp.is_hybrid_sh_raw_mode)
    is_hybrid_sh_post_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_post_mode') and ingp.is_hybrid_sh_post_mode)
    
    # Skip decomposition if hybrid_levels is 0 or equals total_levels (no meaningful decomposition)
    do_cat_decomposition = is_cat_mode and args.hybrid_levels < total_levels and not skip_decomposition
    do_hybrid_sh_decomposition = (is_hybrid_sh_mode or is_hybrid_sh_raw_mode or is_hybrid_sh_post_mode) and not skip_decomposition

    # Adaptive_cat decomposition: visualize per-Gaussian vs hashgrid contributions
    is_adaptive_cat_mode = (ingp is not None and hasattr(ingp, 'is_adaptive_cat_mode') and ingp.is_adaptive_cat_mode)
    do_adaptive_cat_decomposition = is_adaptive_cat_mode and not skip_decomposition

    # 3D_SH_res / 3D_SH_cat decomposition: SH-only vs texture(hash+MLP)-only
    is_3D_SH_res_mode = (ingp is not None and hasattr(ingp, 'is_3D_SH_res_mode') and ingp.is_3D_SH_res_mode)
    is_3D_SH_cat_mode = (ingp is not None and hasattr(ingp, 'is_3D_SH_cat_mode') and ingp.is_3D_SH_cat_mode)
    is_3D_SH_32_mode = (ingp is not None and hasattr(ingp, 'is_3D_SH_32_mode') and ingp.is_3D_SH_32_mode)
    do_sh_res_decomposition = (is_3D_SH_res_mode or is_3D_SH_cat_mode or is_3D_SH_32_mode) and not skip_decomposition

    # Adaptive_zero decomposition: visualize zeros-only vs hash contributors
    is_adaptive_zero_mode = (ingp is not None and hasattr(ingp, 'is_adaptive_zero_mode') and ingp.is_adaptive_zero_mode)
    do_adaptive_zero_decomposition = is_adaptive_zero_mode and not skip_decomposition

    # Adaptive_gate decomposition: visualize gate-closed vs gate-open Gaussians
    is_adaptive_gate_mode = (ingp is not None and hasattr(ingp, 'is_adaptive_gate_mode') and ingp.is_adaptive_gate_mode)
    do_adaptive_gate_decomposition = is_adaptive_gate_mode and not skip_decomposition

    # BG hashgrid visualization: render BG-only for first few frames
    do_bg_visualization = bg_hashgrid is not None and background_mode in ["hashgrid", "hashgrid_relu", "hashgrid_sep"]
    bg_vis_frames = 10  # Number of frames to visualize
    if do_bg_visualization:
        bg_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'bg_only'))
        os.makedirs(bg_output_dir, exist_ok=True)
        print(f"[FINAL] BG hashgrid visualization enabled: saving BG-only renders for first {bg_vis_frames} frames")

    if do_sh_res_decomposition:
        sh_only_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'sh_only'))
        tex_only_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'tex_only'))
        os.makedirs(sh_only_dir, exist_ok=True)
        os.makedirs(tex_only_dir, exist_ok=True)
        print(f"[FINAL] 3D_SH_res decomposition enabled: saving SH-only and texture-only renders")

    if do_cat_decomposition:
        # Create directories for decomposed renders
        ngp_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'ngp_only'))
        gaussian_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gaussian_only'))
        os.makedirs(ngp_output_dir, exist_ok=True)
        os.makedirs(gaussian_output_dir, exist_ok=True)
        print(f"[FINAL] Cat mode decomposition enabled: saving NGP-only and Gaussian-only renders")
    
    if do_hybrid_sh_decomposition:
        # Create directories for hybrid_SH decomposed renders
        ngp_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'ngp_only'))
        gaussian_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gaussians_only'))
        os.makedirs(ngp_output_dir, exist_ok=True)
        os.makedirs(gaussian_output_dir, exist_ok=True)
        print(f"[FINAL] hybrid_SH mode decomposition enabled: saving NGP-only and Gaussians-only renders")

    if do_adaptive_cat_decomposition:
        # Create directories for adaptive_cat decomposed renders
        # New decomposition: separate by weight threshold (0.5)
        pure_gaussian_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'pure_gaussian'))
        hybrid_gaussian_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'hybrid_gaussian_part'))
        hybrid_hash_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'hybrid_hash_part'))
        os.makedirs(pure_gaussian_dir, exist_ok=True)
        os.makedirs(hybrid_gaussian_dir, exist_ok=True)
        os.makedirs(hybrid_hash_dir, exist_ok=True)
        print(f"[FINAL] Adaptive_cat decomposition: pure_gaussian (w>0.5), hybrid_gaussian_part (w<=0.5, hash=0), hybrid_hash_part (w<=0.5, gauss=0)")

    if do_adaptive_zero_decomposition:
        # Create directories for adaptive_zero decomposed renders
        # gaussian_only: Gaussians with weight < 0.5 (use zeros for fine levels)
        # hybrid_gaussian_only: Gaussians with weight >= 0.5, hashgrid masked out
        # hybrid_hash_only: Gaussians with weight >= 0.5, gaussian features masked out
        # training_mode: render with smooth blending (no hard threshold)
        # force_hash: render with all weights forced to 1 (all Gaussians use hash)
        gaussian_only_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gaussian_only'))
        hybrid_gaussian_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'hybrid_gaussian_only'))
        hybrid_hash_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'hybrid_hash_only'))
        training_mode_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'training_mode'))
        force_hash_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'force_hash'))
        os.makedirs(gaussian_only_dir, exist_ok=True)
        os.makedirs(hybrid_gaussian_dir, exist_ok=True)
        os.makedirs(hybrid_hash_dir, exist_ok=True)
        os.makedirs(training_mode_dir, exist_ok=True)
        os.makedirs(force_hash_dir, exist_ok=True)

    if do_adaptive_gate_decomposition:
        # Create directories for adaptive_gate decomposed renders
        # gate_closed: Gaussians with gate probability <= 0.5 (not using hash)
        # gate_open: Gaussians with gate probability > 0.5 (using hash)
        # gaussian_only: Force all gates closed
        # ngp_only: Force all gates open
        gate_closed_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gate_closed'))
        gate_open_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gate_open'))
        gate_gaussian_only_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gaussian_only'))
        gate_ngp_only_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'ngp_only'))
        os.makedirs(gate_closed_dir, exist_ok=True)
        os.makedirs(gate_open_dir, exist_ok=True)
        os.makedirs(gate_gaussian_only_dir, exist_ok=True)
        os.makedirs(gate_ngp_only_dir, exist_ok=True)
        print(f"[FINAL] Adaptive_gate decomposition: gate_closed (prob<=0.5), gate_open (prob>0.5), gaussian_only, ngp_only")

    if len(cameras) == 0:
        print(f"[FINAL] No cameras available, skipping.")
        return
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    l1_values = []
    rendered_indices = []

    # For adaptive modes, also track training mode (soft) metrics
    training_mode_psnr = []
    training_mode_ssim = []
    training_mode_lpips = []
    training_mode_l1 = []

    # For adaptive_zero mode, use hard gating (inference mode) for final renders
    # This ensures binary decisions at weight threshold 0.1
    old_adaptive_zero_inference = None
    if is_adaptive_zero_mode and ingp is not None:
        old_adaptive_zero_inference = getattr(ingp, 'adaptive_zero_inference', False)
        ingp.adaptive_zero_inference = True
        print(f"[FINAL] Adaptive_zero: Using inference mode (hard gating at w>=0.1)")

    # For adaptive_cat mode, use hard gating (inference mode) for final renders
    # This ensures binary decisions at weight threshold 0.9
    old_adaptive_cat_inference = None
    if is_adaptive_cat_mode and ingp is not None:
        old_adaptive_cat_inference = getattr(ingp, 'adaptive_cat_inference', False)
        ingp.adaptive_cat_inference = True
        print(f"[FINAL] Adaptive_cat: Using inference mode (hard gating at w>=0.9)")

    with torch.no_grad():
        for idx, viewpoint in enumerate(cameras):
            if idx % stride != 0:
                continue

            render_pkg = render(viewpoint, gaussians, pipe, background,
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                              skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)

            # --ppisp: same ISP the training loss saw, so the final metrics are
            # consistent with the periodic ones. Train cams resolve to their own
            # fitted exposure/colour by image_name; test cams fall through to
            # frame_idx=-1 (zero per-frame correction).
            if ppisp_apply is not None:
                rendered = torch.clamp(ppisp_apply(render_pkg["render"], viewpoint), 0.0, 1.0)
            else:
                rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            # Keep the small aux maps reused for depth/heatmap saving below, so the
            # rest of render_pkg (full-res allmap etc.) can be freed before LPIPS.
            _depth_exp_f = render_pkg['depth_expected']
            _depth_med_f = render_pkg['depth_median']
            _gnum_f = render_pkg['gaussian_num']

            # Cheap metrics while render_pkg is live, then FREE the full-res render
            # buffers + reclaim the pool BEFORE the memory-heavy LPIPS-VGG forward
            # (renders all test images on top of the resident state → OOM at 4K).
            psnr_val = psnr(rendered, gt).mean().item()
            ssim_val = ssim(rendered, gt).mean().item()
            l1_val = l1_loss(rendered, gt).mean().item()
            del render_pkg
            torch.cuda.empty_cache()
            lpips_val = lpips(rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)
            l1_values.append(l1_val)
            # Store both sorted index and camera name for proper identification
            cam_name_for_index = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            rendered_indices.append(f"{idx:03d}_{cam_name_for_index}")

            # For adaptive modes, also compute training mode (soft) metrics
            if is_adaptive_zero_mode and ingp is not None:
                # Temporarily switch to training mode
                ingp.adaptive_zero_inference = False
                training_pkg = render(viewpoint, gaussians, pipe, background,
                                      ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                      skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)
                ingp.adaptive_zero_inference = True  # Restore inference mode

                training_rendered = torch.clamp(training_pkg["render"], 0.0, 1.0)
                training_mode_psnr.append(psnr(training_rendered, gt).mean().item())
                training_mode_ssim.append(ssim(training_rendered, gt).mean().item())
                training_mode_lpips.append(lpips(training_rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item())
                training_mode_l1.append(l1_loss(training_rendered, gt).mean().item())

            if is_adaptive_cat_mode and ingp is not None:
                # Temporarily switch to training mode
                ingp.adaptive_cat_inference = False
                training_pkg = render(viewpoint, gaussians, pipe, background,
                                      ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                      skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)
                ingp.adaptive_cat_inference = True  # Restore inference mode

                training_rendered = torch.clamp(training_pkg["render"], 0.0, 1.0)
                training_mode_psnr.append(psnr(training_rendered, gt).mean().item())
                training_mode_ssim.append(ssim(training_rendered, gt).mean().item())
                training_mode_lpips.append(lpips(training_rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item())
                training_mode_l1.append(l1_loss(training_rendered, gt).mean().item())
            
            # Save images with camera name for proper ordering
            # After sorting, idx corresponds to sorted order (0=r_0, 1=r_1, etc.)
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            gt_np = gt.permute(1, 2, 0).cpu().numpy()

            save_img_u8(gt_np, os.path.join(final_output_dir, f"{idx:03d}_{cam_name}_gt.png"))
            save_img_u8(rendered_np, os.path.join(final_output_dir, f"{idx:03d}_{cam_name}_render.png"))

            # Always save depth maps to separate folder (extracted before render_pkg free)
            depth_expected = _depth_exp_f  # (1, H, W)
            depth_median = _depth_med_f  # (1, H, W)

            # Convert to numpy for colormap
            depth_expected_np = depth_expected.squeeze(0).cpu().numpy()
            depth_median_np = depth_median.squeeze(0).cpu().numpy()

            # Save as colormapped images
            depth_expected_color = convert_gray_to_cmap(
                depth_expected_np, map_mode='turbo', revert=False
            )
            depth_median_color = convert_gray_to_cmap(
                depth_median_np, map_mode='turbo', revert=False
            )

            save_img_u8(depth_expected_color, os.path.join(depth_output_dir, f"{idx:03d}_{cam_name}_depth_expected.png"))
            save_img_u8(depth_median_color, os.path.join(depth_output_dir, f"{idx:03d}_{cam_name}_depth_median.png"))

            # Save intersection count heatmap (turbo colormap, max_display=200 for consistency)
            gaussian_num = _gnum_f  # (1, H, W)  (extracted before render_pkg free)
            intersection_heatmap, min_count, max_count = create_intersection_heatmap(gaussian_num, max_display=200)
            histogram_img, stats = create_intersection_histogram(gaussian_num, max_display=200)
            save_img_u8(intersection_heatmap, os.path.join(intersection_output_dir, f"{idx:03d}_{cam_name}_intersection.png"))
            save_img_u8(histogram_img, os.path.join(intersection_output_dir, f"{idx:03d}_{cam_name}_histogram.png"))

            # BG-only visualization for first few frames
            if do_bg_visualization and idx < bg_vis_frames:
                # Render BG hashgrid only (no Gaussians)
                H, W = viewpoint.image_height, viewpoint.image_width
                rays_d, rays_o = cam2rays(viewpoint)
                ray_unit = torch.nn.functional.normalize(rays_d, dim=-1).float()

                # Query BG hashgrid with camera position for position-aware sphere intersection
                ray_origins_bg = rays_o.unsqueeze(0).expand(ray_unit.shape[0], -1)
                bg_features = bg_hashgrid(ray_unit, ray_origins_bg)  # (H*W, F)

                # Decode through MLP
                bg_rgb = ingp.rgb_decode(bg_features, ray_unit)  # (H*W, 3)
                bg_rgb = bg_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
                bg_rgb = torch.clamp(bg_rgb, 0.0, 1.0)

                bg_rgb_np = bg_rgb.permute(1, 2, 0).cpu().numpy()
                save_img_u8(bg_rgb_np, os.path.join(bg_output_dir, f"{idx:03d}_{cam_name}_bg.png"))

            # Flex kernel beta heatmap: render flex_beta as color, then apply coolwarm colormap
            if is_flex_kernel:
                # Get per-Gaussian flex beta and expand to RGB (same value in all 3 channels)
                flex_beta_vals = gaussians.get_flex_beta  # (N, 1)
                flex_beta_color = flex_beta_vals.expand(-1, 3)  # (N, 3)

                # Render with flex_beta as override_color (uses alpha blending from rasterizer)
                flex_beta_render_pkg = render(viewpoint, gaussians, pipe, background,
                                             ingp=None, beta=0.0, iteration=iteration, cfg=cfg_model,
                                             override_color=flex_beta_color)
                flex_beta_map = flex_beta_render_pkg["render"][0:1]  # Take first channel (R=G=B)
                render_alpha = flex_beta_render_pkg["rend_alpha"]  # (1, H, W)

                # Create heatmap visualization
                flex_beta_heatmap, min_beta, max_beta = create_flex_beta_heatmap(
                    flex_beta_map, render_alpha, min_display=0.0, max_display=10.0
                )
                save_img_u8(flex_beta_heatmap, os.path.join(flex_beta_output_dir, f"{idx:03d}_flex_beta.png"))

            # 3D_SH_res decomposition: SH-only and texture(hash+MLP)-only
            if do_sh_res_decomposition:
                # SH-only: disable hashgrid, residual ≈ 0
                sh_pkg = render(viewpoint, gaussians, pipe, background,
                                ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                decompose_mode='sh_only')
                sh_rendered = torch.clamp(sh_pkg["render"], 0.0, 1.0)
                save_img_u8(sh_rendered.permute(1, 2, 0).cpu().numpy(),
                           os.path.join(sh_only_dir, f"{idx:03d}_sh.png"))

                # Texture-only: zero SH, only MLP residual
                tex_pkg = render(viewpoint, gaussians, pipe, background,
                                 ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                 decompose_mode='tex_only')
                tex_rendered = torch.clamp(tex_pkg["render"], 0.0, 1.0)
                save_img_u8(tex_rendered.permute(1, 2, 0).cpu().numpy(),
                           os.path.join(tex_only_dir, f"{idx:03d}_tex.png"))

            # Cat mode decomposition: render with masked features
            if do_cat_decomposition:
                # NGP-only: zero out per-Gaussian features, keep hashgrid
                ngp_render_pkg = render(viewpoint, gaussians, pipe, background,
                                       ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                       decompose_mode='ngp_only')
                ngp_rendered = torch.clamp(ngp_render_pkg["render"], 0.0, 1.0)
                ngp_rendered_np = ngp_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(ngp_rendered_np, os.path.join(ngp_output_dir, f"{idx:03d}_ngp.png"))
                
                # Gaussian-only: zero out hashgrid features, keep per-Gaussian
                gaussian_render_pkg = render(viewpoint, gaussians, pipe, background,
                                            ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                            decompose_mode='gaussian_only')
                gaussian_rendered = torch.clamp(gaussian_render_pkg["render"], 0.0, 1.0)
                gaussian_rendered_np = gaussian_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gaussian_rendered_np, os.path.join(gaussian_output_dir, f"{idx:03d}_gaussian.png"))
            
            # hybrid_SH mode decomposition: render with masked features
            if do_hybrid_sh_decomposition:
                # NGP-only: zero out per-Gaussian SH, keep hashgrid DC residual
                ngp_render_pkg = render(viewpoint, gaussians, pipe, background,
                                       ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                       decompose_mode='ngp_only')
                ngp_rendered = torch.clamp(ngp_render_pkg["render"], 0.0, 1.0)
                ngp_rendered_np = ngp_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(ngp_rendered_np, os.path.join(ngp_output_dir, f"{idx:03d}_ngp.png"))

                # Gaussian-only: zero out hashgrid DC residual, keep per-Gaussian SH
                gaussian_render_pkg = render(viewpoint, gaussians, pipe, background,
                                            ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                            decompose_mode='gaussian_only')
                gaussian_rendered = torch.clamp(gaussian_render_pkg["render"], 0.0, 1.0)
                gaussian_rendered_np = gaussian_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gaussian_rendered_np, os.path.join(gaussian_output_dir, f"{idx:03d}_gaussian.png"))

            # Adaptive_cat mode decomposition: separate by weight threshold
            if do_adaptive_cat_decomposition:
                # Pure Gaussian: only Gaussians with weight > 0.5 (don't use hash)
                pure_gauss_pkg = render(viewpoint, gaussians, pipe, background,
                                       ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                       decompose_mode='pure_gaussian')
                pure_gauss_rendered = torch.clamp(pure_gauss_pkg["render"], 0.0, 1.0)
                pure_gauss_np = pure_gauss_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(pure_gauss_np, os.path.join(pure_gaussian_dir, f"{idx:03d}_pure_gaussian.png"))

                # Hybrid Gaussian part: Gaussians with weight <= 0.5, but hash features zeroed
                hybrid_gauss_pkg = render(viewpoint, gaussians, pipe, background,
                                         ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                         decompose_mode='hybrid_gaussian_part')
                hybrid_gauss_rendered = torch.clamp(hybrid_gauss_pkg["render"], 0.0, 1.0)
                hybrid_gauss_np = hybrid_gauss_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(hybrid_gauss_np, os.path.join(hybrid_gaussian_dir, f"{idx:03d}_hybrid_gaussian.png"))

                # Hybrid Hash part: Gaussians with weight <= 0.5, but Gaussian features zeroed
                hybrid_hash_pkg = render(viewpoint, gaussians, pipe, background,
                                        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                        decompose_mode='hybrid_hash_part')
                hybrid_hash_rendered = torch.clamp(hybrid_hash_pkg["render"], 0.0, 1.0)
                hybrid_hash_np = hybrid_hash_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(hybrid_hash_np, os.path.join(hybrid_hash_dir, f"{idx:03d}_hybrid_hash.png"))

            # Adaptive_zero mode decomposition: separate by weight threshold
            if do_adaptive_zero_decomposition:
                # Training mode: render with smooth blending (no hard threshold)
                # Temporarily disable inference mode
                old_inference = ingp.adaptive_zero_inference if hasattr(ingp, 'adaptive_zero_inference') else False
                ingp.adaptive_zero_inference = False
                training_pkg = render(viewpoint, gaussians, pipe, background,
                                      ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                      decompose_mode=None)  # No decompose, just training mode
                ingp.adaptive_zero_inference = old_inference
                training_rendered = torch.clamp(training_pkg["render"], 0.0, 1.0)
                training_np = training_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(training_np, os.path.join(training_mode_dir, f"{idx:03d}_training.png"))

                # Gaussian only: Gaussians with weight < 0.5 (use zeros for fine levels)
                gauss_pkg = render(viewpoint, gaussians, pipe, background,
                                   ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                   decompose_mode='gaussian_only')
                gauss_rendered = torch.clamp(gauss_pkg["render"], 0.0, 1.0)
                gauss_np = gauss_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gauss_np, os.path.join(gaussian_only_dir, f"{idx:03d}_gaussian_only.png"))

                # Hybrid Gaussian only: Gaussians with weight >= 0.5, hashgrid masked out
                hybrid_gauss_pkg = render(viewpoint, gaussians, pipe, background,
                                          ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                          decompose_mode='hybrid_gaussian_only')
                hybrid_gauss_rendered = torch.clamp(hybrid_gauss_pkg["render"], 0.0, 1.0)
                hybrid_gauss_np = hybrid_gauss_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(hybrid_gauss_np, os.path.join(hybrid_gaussian_dir, f"{idx:03d}_hybrid_gaussian.png"))

                # Hybrid Hash only: Gaussians with weight >= 0.5, gaussian features masked out
                hybrid_hash_pkg = render(viewpoint, gaussians, pipe, background,
                                         ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                         decompose_mode='hybrid_hash_only')
                hybrid_hash_rendered = torch.clamp(hybrid_hash_pkg["render"], 0.0, 1.0)
                hybrid_hash_np = hybrid_hash_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(hybrid_hash_np, os.path.join(hybrid_hash_dir, f"{idx:03d}_hybrid_hash.png"))

                # Force hash: render with all weights set to 1 (all Gaussians use hash)
                # This helps verify if hash features are saturating to compensate for low weights
                old_weights = gaussians._adaptive_zero_weight.data.clone()
                # Set logits to large positive value so sigmoid(logit) ≈ 1
                gaussians._adaptive_zero_weight.data.fill_(10.0)  # sigmoid(10) ≈ 0.99995
                force_hash_pkg = render(viewpoint, gaussians, pipe, background,
                                        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                        decompose_mode=None)
                gaussians._adaptive_zero_weight.data.copy_(old_weights)  # Restore original weights
                force_hash_rendered = torch.clamp(force_hash_pkg["render"], 0.0, 1.0)
                force_hash_np = force_hash_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(force_hash_np, os.path.join(force_hash_dir, f"{idx:03d}_force_hash.png"))

            # Adaptive_gate mode decomposition: separate by gate probability threshold
            if do_adaptive_gate_decomposition:
                # Gate closed: only Gaussians with gate probability <= 0.5 (not using hash)
                gate_closed_pkg = render(viewpoint, gaussians, pipe, background,
                                        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                        decompose_mode='gate_closed')
                gate_closed_rendered = torch.clamp(gate_closed_pkg["render"], 0.0, 1.0)
                gate_closed_np = gate_closed_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gate_closed_np, os.path.join(gate_closed_dir, f"{idx:03d}_gate_closed.png"))

                # Gate open: only Gaussians with gate probability > 0.5 (using hash)
                gate_open_pkg = render(viewpoint, gaussians, pipe, background,
                                      ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                      decompose_mode='gate_open')
                gate_open_rendered = torch.clamp(gate_open_pkg["render"], 0.0, 1.0)
                gate_open_np = gate_open_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gate_open_np, os.path.join(gate_open_dir, f"{idx:03d}_gate_open.png"))

                # Gaussian only: Force all gates closed (all Gaussians use Gaussian-only)
                gauss_only_pkg = render(viewpoint, gaussians, pipe, background,
                                       ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                       decompose_mode='gaussian_only')
                gauss_only_rendered = torch.clamp(gauss_only_pkg["render"], 0.0, 1.0)
                gauss_only_np = gauss_only_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gauss_only_np, os.path.join(gate_gaussian_only_dir, f"{idx:03d}_gaussian_only.png"))

                # NGP only: Force all gates open (all Gaussians use hash)
                ngp_only_pkg = render(viewpoint, gaussians, pipe, background,
                                     ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
                                     decompose_mode='ngp_only')
                ngp_only_rendered = torch.clamp(ngp_only_pkg["render"], 0.0, 1.0)
                ngp_only_np = ngp_only_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(ngp_only_np, os.path.join(gate_ngp_only_dir, f"{idx:03d}_ngp_only.png"))

            # cam_name already defined above for image saving
            print(f"[FINAL] Idx {idx:3d} ({cam_name}): PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

    # Summary
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_l1 = np.mean(l1_values)

    # Training mode metrics for adaptive modes
    avg_train_psnr = np.mean(training_mode_psnr) if training_mode_psnr else None
    avg_train_ssim = np.mean(training_mode_ssim) if training_mode_ssim else None
    avg_train_lpips = np.mean(training_mode_lpips) if training_mode_lpips else None
    avg_train_l1 = np.mean(training_mode_l1) if training_mode_l1 else None

    # Check if we're in any adaptive mode
    is_any_adaptive_mode = is_adaptive_zero_mode or is_adaptive_cat_mode

    print(f"\n[FINAL] ════════════════════════════════════════")
    if is_adaptive_zero_mode:
        print(f"[FINAL] INFERENCE MODE Metrics ({len(psnr_values)} images) [threshold w>=0.1]:")
    elif is_adaptive_cat_mode:
        print(f"[FINAL] INFERENCE MODE Metrics ({len(psnr_values)} images) [threshold w>=0.9]:")
    else:
        print(f"[FINAL] Metrics ({len(psnr_values)} images):")
    print(f"[FINAL]   Average PSNR:  {avg_psnr:.2f} dB")
    print(f"[FINAL]   Average SSIM:  {avg_ssim:.4f}")
    print(f"[FINAL]   Average LPIPS: {avg_lpips:.4f}")
    print(f"[FINAL]   Average L1:    {avg_l1:.6f}")
    if is_any_adaptive_mode and avg_train_psnr is not None:
        print(f"[FINAL] ────────────────────────────────────────")
        print(f"[FINAL] TRAINING MODE Metrics (soft blending):")
        print(f"[FINAL]   Average PSNR:  {avg_train_psnr:.2f} dB")
        print(f"[FINAL]   Average SSIM:  {avg_train_ssim:.4f}")
        print(f"[FINAL]   Average LPIPS: {avg_train_lpips:.4f}")
        print(f"[FINAL]   Average L1:    {avg_train_l1:.6f}")
        print(f"[FINAL] ────────────────────────────────────────")
        print(f"[FINAL] Gap (inference - training):")
        print(f"[FINAL]   PSNR:  {avg_psnr - avg_train_psnr:+.2f} dB")
        print(f"[FINAL]   SSIM:  {avg_ssim - avg_train_ssim:+.4f}")
        print(f"[FINAL]   LPIPS: {avg_lpips - avg_train_lpips:+.4f}")
    print(f"[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Images saved to: {final_output_dir}")
    print(f"[FINAL] Depth maps saved to: {depth_output_dir}")
    print(f"[FINAL] Intersection heatmaps saved to: {intersection_output_dir}")
    if do_cat_decomposition or do_hybrid_sh_decomposition:
        print(f"[FINAL] NGP-only renders saved to: {ngp_output_dir}")
        print(f"[FINAL] Gaussian-only renders saved to: {gaussian_output_dir}")
    if do_adaptive_zero_decomposition:
        print(f"[FINAL] Training mode renders saved to: {training_mode_dir}")
        print(f"[FINAL] Gaussian-only renders saved to: {gaussian_only_dir}")
        print(f"[FINAL] Hybrid-gaussian renders saved to: {hybrid_gaussian_dir}")
        print(f"[FINAL] Hybrid-hash renders saved to: {hybrid_hash_dir}")
        print(f"[FINAL] Force-hash renders saved to: {force_hash_dir}")
    if do_adaptive_gate_decomposition:
        print(f"[FINAL] Gate-closed renders saved to: {gate_closed_dir}")
        print(f"[FINAL] Gate-open renders saved to: {gate_open_dir}")
        print(f"[FINAL] Gaussian-only renders saved to: {gate_gaussian_only_dir}")
        print(f"[FINAL] NGP-only renders saved to: {gate_ngp_only_dir}")

    # Restore original inference states
    if old_adaptive_zero_inference is not None and ingp is not None:
        ingp.adaptive_zero_inference = old_adaptive_zero_inference
    if old_adaptive_cat_inference is not None and ingp is not None:
        ingp.adaptive_cat_inference = old_adaptive_cat_inference

    # Save metrics file in output root. For test_metrics.txt the periodic-eval header
    # was written at training start and rows have been appended through training_report;
    # open in append mode so the final block goes underneath. Other metrics_file values
    # (e.g. train_metrics.txt) are written fresh.
    metrics_path = os.path.join(scene.model_path, metrics_file)
    open_mode = 'a' if metrics_file == 'test_metrics.txt' else 'w'
    with open(metrics_path, open_mode) as f:
        if open_mode == 'a':
            f.write("\n")
        f.write(f"Final Evaluation (stride={stride})\n")
        f.write(f"════════════════════════════════════════\n")
        f.write(f"Images rendered: {len(psnr_values)}\n\n")

        if is_adaptive_zero_mode:
            f.write(f"INFERENCE MODE (hard gating at w>=0.1):\n")
        elif is_adaptive_cat_mode:
            f.write(f"INFERENCE MODE (hard gating at w>=0.9):\n")
        f.write(f"Average PSNR:    {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM:    {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS:   {avg_lpips:.4f}\n")
        f.write(f"Average L1:      {avg_l1:.6f}\n\n")

        # Training mode metrics for adaptive modes
        if is_any_adaptive_mode and avg_train_psnr is not None:
            f.write(f"TRAINING MODE (soft blending):\n")
            f.write(f"Average PSNR:    {avg_train_psnr:.2f} dB\n")
            f.write(f"Average SSIM:    {avg_train_ssim:.4f}\n")
            f.write(f"Average LPIPS:   {avg_train_lpips:.4f}\n")
            f.write(f"Average L1:      {avg_train_l1:.6f}\n\n")
            f.write(f"Gap (inference - training):\n")
            f.write(f"  PSNR:  {avg_psnr - avg_train_psnr:+.2f} dB\n")
            f.write(f"  SSIM:  {avg_ssim - avg_train_ssim:+.4f}\n")
            f.write(f"  LPIPS: {avg_lpips - avg_train_lpips:+.4f}\n\n")

        f.write(f"Per-image results (inference mode):\n")
        f.write(f"{'Index':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'LPIPS':<12} {'L1':<12}\n")
        f.write(f"{'-'*58}\n")

        for i, idx in enumerate(rendered_indices):
            f.write(f"{idx:<10} {psnr_values[i]:>10.2f} {ssim_values[i]:>10.4f} {lpips_values[i]:>10.4f} {l1_values[i]:>12.6f}\n")

        # Per-image training mode metrics
        if is_any_adaptive_mode and training_mode_psnr:
            f.write(f"\nPer-image results (training mode):\n")
            f.write(f"{'Index':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'LPIPS':<12} {'L1':<12}\n")
            f.write(f"{'-'*58}\n")
            for i, idx in enumerate(rendered_indices):
                f.write(f"{idx:<10} {training_mode_psnr[i]:>10.2f} {training_mode_ssim[i]:>10.4f} {training_mode_lpips[i]:>10.4f} {training_mode_l1[i]:>12.6f}\n")

    print(f"[FINAL] Metrics saved to: {metrics_path}")


def reload_cameras_at_resolution(scene, args, new_resolution, new_data_device):
    """Mid-train reload all train + test cameras at a new -r value and device.

    Triggered by the `--start_resolution` progressive curriculum at
    `--freeze_hash_iter`. Uses the cached `scene._cam_infos_train` /
    `scene._cam_infos_test` (CameraInfo dataclasses populated at Scene init)
    to re-call `loadCam(args, id, cam_info, resolution_scale=1.0)` for each
    camera. The OLD camera tensors are freed first to reclaim GPU memory
    before the new (potentially larger) tensors are allocated.

    Returns a brief textual summary.
    """
    import gc
    import torch
    from utils.camera_utils import loadCam
    if not hasattr(scene, '_cam_infos_train') or not hasattr(scene, '_cam_infos_test'):
        return "[PROGRESSIVE-RES] WARNING: Scene._cam_infos_{train,test} not cached — skipping reload."
    # Mutate args (loadCam reads .resolution and .data_device off args).
    old_res = args.resolution
    old_dev = args.data_device
    args.resolution = int(new_resolution)
    args.data_device = str(new_data_device)
    # Walk each resolution_scale (usually just [1.0]) and replace the camera lists.
    _n_train = 0
    _n_test = 0
    for scale in list(scene.train_cameras.keys()):
        # Free the old camera tensors first.
        for cam in scene.train_cameras[scale]:
            if hasattr(cam, 'original_image'):
                cam.original_image = None
            if hasattr(cam, 'gt_alpha_mask') and cam.gt_alpha_mask is not None:
                cam.gt_alpha_mask = None
        for cam in scene.test_cameras[scale]:
            if hasattr(cam, 'original_image'):
                cam.original_image = None
            if hasattr(cam, 'gt_alpha_mask') and cam.gt_alpha_mask is not None:
                cam.gt_alpha_mask = None
        scene.train_cameras[scale] = [loadCam(args, i, ci, scale)
                                       for i, ci in enumerate(scene._cam_infos_train)]
        scene.test_cameras[scale] = [loadCam(args, i, ci, scale)
                                      for i, ci in enumerate(scene._cam_infos_test)]
        _n_train += len(scene.train_cameras[scale])
        _n_test += len(scene.test_cameras[scale])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (f"[PROGRESSIVE-RES] Reloaded {_n_train} train + {_n_test} test cams: "
            f"-r {old_res} ({old_dev}) → -r {new_resolution} ({new_data_device}).")


def prepare_output_and_logger(dataset, scene_name, yaml_file="", args=None):
    # Extract dataset and scene from source_path
    # e.g., /path/to/nerf_synthetic/ficus -> dataset=nerf_synthetic, scene=ficus
    source_parts = dataset.source_path.rstrip('/').split('/')
    scene_from_path = source_parts[-1]  # e.g., "ficus"
    dataset_name = source_parts[-2] if len(source_parts) > 1 else "unknown"  # e.g., "nerf_synthetic"
    
    if not dataset.model_path:
        # If no model_path specified, create default name with timestamp
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        now = datetime.datetime.now()
        time_str = now.strftime("-%m%d-%H%M")
        exp_name = scene_name + time_str
        if yaml_file != "":
            exp_name += '-' + yaml_file
        dataset.model_path = os.path.join("./output/", exp_name)
    else:
        # User specified -m flag: organize as outputs/{dataset}/{scene}/{method}/{name}
        run_name = dataset.model_path
        run_name = run_name.lstrip('./').lstrip('/')
        method = args.method if args and hasattr(args, 'method') else "baseline"
        # For cat mode, append hybrid_levels to the run name
        if method == "cat" and args and hasattr(args, 'hybrid_levels'):
            run_name = f"{run_name}_{args.hybrid_levels}_levels"
        dataset.model_path = os.path.join("outputs", dataset_name, scene_from_path, method, run_name)
    
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Save full training configuration for reproducibility
    if args is not None:
        import shutil
        import pickle

        # 1. Save exact command line
        with open(os.path.join(dataset.model_path, "command_line.txt"), 'w') as f:
            f.write(" ".join(sys.argv))

        # 2. Save full args as JSON (human-readable)
        args_dict = vars(args).copy()
        # Convert non-serializable types
        for k, v in args_dict.items():
            if hasattr(v, '__dict__'):
                args_dict[k] = str(v)
        with open(os.path.join(dataset.model_path, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=2, default=str)

        # 3. Save args as pickle (exact reproduction)
        with open(os.path.join(dataset.model_path, "args.pkl"), 'wb') as f:
            pickle.dump(args, f)

        # 4. Copy the YAML config file
        if hasattr(args, 'yaml') and os.path.exists(args.yaml):
            shutil.copy(args.yaml, os.path.join(dataset.model_path, "config.yaml"))

        print(f"[CONFIG] Saved training configuration to {dataset.model_path}")

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, \
ingp_model, beta, args, cfg_model, test_psnr = None, train_psnr = None, iter_list = None, skybox_model = None, background_mode = "none", bg_hashgrid_model = None, ppisp_apply = None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        # Determine if skybox/bg_hashgrid should be active at this iteration
        active_skybox = skybox_model if (skybox_model is not None and iteration >= cfg_model.ingp_stage.switch_iter) else None
        bg_start_iter = max(args.bg_hashgrid_start_iter, cfg_model.ingp_stage.switch_iter) if hasattr(args, 'bg_hashgrid_start_iter') else cfg_model.ingp_stage.switch_iter
        active_bg_hashgrid = bg_hashgrid_model if (bg_hashgrid_model is not None and iteration >= bg_start_iter) else None

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_legacy_test = 0.0   # buggy [0,1] passthrough — matches 3DGS-ecosystem convention
                lpips_canon_test  = 0.0   # canonical [-1,1]-rescaled LPIPS (Zhang spec)
                # Cap eval at ~25 cameras with a uniform stride — NeRF-Synthetic has
                # 200 test cams, BlendedMVS / TnT have similar; rendering them all
                # every save_interval was bottlenecking long runs.
                _n_cams = len(config['cameras'])
                eval_stride = max(1, _n_cams // 25)
                cameras_evaluated = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if idx % eval_stride != 0:
                        continue

                    # Forward the perf-critical render args that training uses, so test
                    # renders aren't crippled by defaults (esp. aabb_mode="2dgs" which
                    # falls back to loose square tile AABB and balloons per-pixel
                    # contributor counts vs. accutile/snugbox/adr*).
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, ingp = ingp_model, \
                         beta = beta, iteration = iteration, cfg = cfg_model, skybox = active_skybox,
                         background_mode = background_mode, bg_hashgrid = active_bg_hashgrid,
                         aabb_mode = args.aabb,
                         max_intersections_per_pixel = args.max_intersections_per_pixel,
                         aa = args.aa, aa_threshold = args.aa_threshold,
                         detach_hash_grad = args.detach_hash_grad,
                         lowpass = args.lowpass, pixel_center = args.pixel_center,
                         antialiasing = args.antialiasing, sv_metric = args.sv_metric,
                         is_training = False)
                    # --ppisp: apply the ISP before the metric clamp. `ppisp_apply`
                    # resolves the frame index by image_name, so TRAIN cams get
                    # their own fitted exposure/colour while TEST cams fall through
                    # to frame_idx=-1 = zero per-frame correction (canonical
                    # appearance). No test GT is consulted either way, so the
                    # held-out PSNR stays comparable to a non-PPISP run.
                    if ppisp_apply is not None:
                        image = ppisp_apply(render_pkg["render"], viewpoint)
                    else:
                        image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    cameras_evaluated += 1

                    # Cheap metrics (no big transient activations) while render_pkg is live.
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    # Log images for first camera only (needs render_pkg → do BEFORE freeing it).
                    if tb_writer and cameras_evaluated == 1:
                        tb_writer.add_image(f'{config["name"]}/render', image, iteration)
                        tb_writer.add_image(f'{config["name"]}/gt', gt_image, iteration)
                        if "render_fg" in render_pkg:
                            fg_image = torch.clamp(render_pkg["render_fg"], 0.0, 1.0)
                            bg_image = torch.clamp(render_pkg["render_bg"], 0.0, 1.0)
                            alpha = render_pkg["rend_alpha"]
                            tb_writer.add_image(f'{config["name"]}/foreground', fg_image, iteration)
                            tb_writer.add_image(f'{config["name"]}/background', bg_image, iteration)
                            tb_writer.add_image(f'{config["name"]}/alpha', alpha.repeat(3, 1, 1), iteration)

                    # Bound eval peak memory. `render_pkg` holds many full-res tensors
                    # (allmap, depth/normal maps, aux buffers ~1 GB at 4K) and the VGG-LPIPS
                    # forward (run twice/img) allocates large transient activations
                    # (~4 GB per conv at 4K). Rendering all test images on top of the resident
                    # train state then computing LPIPS was OOMing at 25k. FREE the render
                    # buffers + reclaim the pool BEFORE the LPIPS-VGG pass so the VGG has room.
                    del render_pkg
                    torch.cuda.empty_cache()

                    # LPIPS in BOTH conventions:
                    #   legacy: vendored lpipsPyTorch convention (no rescale) — matches Inria
                    #     3DGS / 2DGS / FastGS / GaussianSpa / mini-splatting2 reported values.
                    #   canonical: rescale to [-1,1] before scoring (Zhang's PerceptualSimilarity
                    #     spec). ~+18% higher than legacy in practice on Mip-360.
                    _img_b = image.unsqueeze(0)
                    _gt_b = gt_image.unsqueeze(0)
                    lpips_legacy_test += lpips(_img_b, _gt_b, net_type='vgg').mean().double()
                    lpips_canon_test  += lpips(_img_b * 2.0 - 1.0, _gt_b * 2.0 - 1.0, net_type='vgg').mean().double()

                    del image, gt_image, _img_b, _gt_b

                psnr_test /= cameras_evaluated
                l1_test /= cameras_evaluated
                ssim_test /= cameras_evaluated
                lpips_legacy_test /= cameras_evaluated
                lpips_canon_test  /= cameras_evaluated
                n_points = scene.gaussians.get_xyz.shape[0]
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.4f} SSIM {:.4f} LPIPSᴸ {:.4f} LPIPSᶜ {:.4f} Points {}".format(
                    iteration, config['name'],
                    l1_test.item(), psnr_test.item(), ssim_test.item(),
                    lpips_legacy_test.item(), lpips_canon_test.item(), n_points))

                if config['name'] == 'test':
                    test_psnr.append(psnr_test.item())
                    # Append row to test_metrics.txt (header was written at training start).
                    # Columns: Iter | PSNR(dB) | SSIM | LPIPSᴸegacy | LPIPSᶜanonical | L1 | Points
                    try:
                        with open(os.path.join(scene.model_path, 'test_metrics.txt'), 'a') as f:
                            f.write(f"{iteration:<10}{psnr_test.item():<10.4f}"
                                    f"{ssim_test.item():<9.4f}{lpips_legacy_test.item():<11.4f}"
                                    f"{lpips_canon_test.item():<11.4f}{l1_test.item():<12.6f}"
                                    f"{n_points:<12}\n")
                    except Exception as _e:
                        print(f"[WARN] Failed to append periodic test metric: {_e}")
                elif config['name'] == 'train':
                    train_psnr.append(psnr_test.item())

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def merge_cfg_to_args(args, cfg, cli_args=None):
    """Merge specific sections from config into args

    CLI arguments take precedence over config values.
    cli_args: set of argument names explicitly passed on the command line.
    """
    target_sections = ['training_cfg', 'settings', 'loss']

    for section in target_sections:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    # CLI takes precedence over yaml
                    if cli_args is not None and k in cli_args:
                        continue
                    setattr(args, k, v)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    # --patience: from iter 15k onwards, run test eval every 500 iters and log
    # to test_metrics.txt. Track best test-PSNR; if no improvement for `patience`
    # consecutive evals (~500*N iters), stop early. 0 = disabled (default).
    # NOTE: this peeks at the test set during training. Use only for convergence
    # studies / diagnostics, not for paper numbers.
    parser.add_argument("--patience", type=int, default=0,
                        help="Periodic test-eval + early-stop convergence study. "
                             "0 = off (default). N>0 = eval every 500 iters from "
                             "iter 15k, stop after N consecutive non-improvements.")
    parser.add_argument("--patience_eval_interval", type=int, default=500,
                        help="Iterations between periodic test evals (default 500).")
    parser.add_argument("--patience_start_iter", type=int, default=15000,
                        help="First iteration to begin periodic test eval (default 15000).")
    parser.add_argument("--patience_min_delta", type=float, default=1e-4,
                        help="Minimum PSNR (dB) improvement to reset patience counter.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--random_init", type=int, default=0,
                        help="If >0, bypass the dataset's points3d.ply / COLMAP sparse cloud and "
                             "initialize with N random points uniformly distributed in a sphere. "
                             "Radius auto-sized from cameras_extent unless --random_init_radius is set.")
    parser.add_argument("--random_init_radius", type=float, default=0.0,
                        help="--random_init: sphere radius in world units. 0 = auto = 1.3 × cameras_extent.")
    parser.add_argument("--init_ply", type=str, default=None,
                        help="Initialize Gaussians from an external PLY file instead of dataset point cloud")

    # === --finetune_from: continue training a FINISHED run ===
    # Distinct from --start_checkpoint (Gaussians+Adam only, no hash/MLP) and from
    # --share_ckpt_iter (a sweep-resume snapshot keyed to the source dir). This one
    # takes a model output directory and reloads the three things that define a
    # trained 3D_SH_res scene: surfels+SV+SH (PLY), the hash+MLP residual
    # (ngp_<it>.pth), and the photometric ISP (ppisp.pt) when --ppisp is on.
    # Optimizer state is deliberately NOT restored — fresh Adam, fresh LR schedule.
    parser.add_argument("--finetune_from", type=str, default=None,
                        help="Model output dir of a finished run to finetune. Reloads "
                             "point_cloud/iteration_<N>/point_cloud.ply (surfels+SV+SH), "
                             "ngp_<N>.pth (hash+MLP) and ppisp.pt (ISP, if --ppisp). "
                             "Fresh optimizers — no Adam state carried over.")
    parser.add_argument("--finetune_iter", type=int, default=-1,
                        help="Which saved iteration to finetune from (default -1 = highest "
                             "point_cloud/iteration_* present in --finetune_from).")

    # === --3rgs: camera pose refinement (3R-GS "sfm" core) ===
    # Jointly optimizes a per-camera rigid pose delta (3D translation + 6D
    # rotation, zero-init; 3R-GS's CameraOptModule) alongside the Gaussians, to
    # correct COLMAP pose error during training. The CUDA rasterizers here have
    # no viewmatrix gradient, so the delta's gradient is routed through a
    # differentiable world-frame rigid transform of the Gaussian means+rotations
    # (see scene/camera_pose_opt.py). Orthogonal to densification (--fastgs /
    # --mcmc) and to the mode flips/splits of res_3d_paired (separate optimizer,
    # per-camera, fixed count). Refined poses are exported at every save.
    # === --ppisp: photometric / ISP compensation (NVIDIA PPISP, ../ppisp) ===
    # Learned per-frame exposure + colour and per-camera vignetting + CRF applied
    # to the RENDER before the loss, trained jointly with the Gaussians. Aimed at
    # handheld/phone captures where auto-exposure and white-balance drift between
    # frames force the radiance field to invent floaters. Identity at init, so
    # `--ppisp` off ⇒ byte-identical to before. Method-agnostic (image-space).
    parser.add_argument("--ppisp", action="store_true", default=False,
                        help="Enable PPISP photometric compensation (per-frame exposure + "
                             "colour, per-camera vignetting + CRF) on the rendered image.")
    parser.add_argument("--ppisp_lr", type=float, default=0.002,
                        help="Adam LR for the PPISP parameters (default 0.002, the paper value). "
                             "Uses PPISP's own linear-warmup → exponential-decay schedule.")
    parser.add_argument("--ppisp_crf", action="store_true", default=False,
                        help="Let the per-camera CRF (tone curve) train. OFF by default: our "
                             "captures come from one ISP whose curve is already in the GT and "
                             "which we want the splats to reproduce — training it leaves the "
                             "exported scene in pre-CRF space and it renders wrong in a viewer "
                             "that has no CRF stage.")
    parser.add_argument("--ppisp_no_camera", action="store_true", default=False,
                        help="Disable the per-camera stages (vignetting AND CRF), leaving only "
                             "per-frame exposure + colour. Also removes the kernel's [0,1] clamp, "
                             "which otherwise zeroes the gradient of any pixel rendering above 1 "
                             "— use this if bright regions stop converging.")
    parser.add_argument("--ppisp_controller", action="store_true", default=False,
                        help="Train PPISP's CNN controller (predicts per-frame exposure/colour "
                             "from the rendered image, so held-out views get a fitted correction "
                             "instead of the canonical zero one). Requires freezing the scene at "
                             "80%% of training; OFF by default.")
    parser.add_argument("--ppisp_overflow_w", type=float, default=0.01,
                        help="Weight of the over-range penalty relu(render-1).mean() applied when "
                             "the per-camera path is active, to replace the gradient the kernel's "
                             "[0,1] clamp destroys. 0 = off.")
    parser.add_argument("--3rgs", dest="pose_refine", action="store_true", default=False,
                        help="Enable 3R-GS-style per-camera pose refinement during training.")
    parser.add_argument("--3rgs_lr", dest="pose_refine_lr", type=float, default=1e-5,
                        help="Adam LR for the per-camera 9D pose delta (3R-GS sfm default 1e-5).")
    parser.add_argument("--3rgs_warmup", dest="pose_refine_warmup", type=int, default=500,
                        help="Iterations before pose refinement starts (let geometry settle first; "
                             "important under --cold). The learned correction is applied from this "
                             "iter onward and never reverted.")
    parser.add_argument("--3rgs_until", dest="pose_refine_until", type=int, default=-1,
                        help="Last iteration the pose delta is *stepped* (-1 = total iterations). "
                             "After this the correction is frozen but still applied.")
    parser.add_argument("--3rgs_reg", dest="pose_refine_reg", type=float, default=0.0,
                        help="Weight decay on the pose delta embedding (keeps corrections small).")

    # === --deform: per-surfel time-dependent deformation (Deformable-3DGS style) ===
    # Each surfel carries a learnable latent (_deform_latent, a per-Gauss tensor
    # threaded through clone/split/prune/PLY) decoded with a per-frame time code
    # by a small MLP into (Δpos, Δrot), applied before the hash query/raster (and
    # before --3rgs). Models small subject motion during capture. Validated with
    # --fastgs (no depth reinit); the latent rides clone/split/prune cleanly.
    parser.add_argument("--deform", dest="deform", action="store_true", default=False,
                        help="Enable per-surfel time-dependent deformation from a canonical pose.")
    parser.add_argument("--deform_dim", dest="deform_dim", type=int, default=8,
                        help="Per-surfel deformation latent dimension.")
    parser.add_argument("--deform_latent_lr", dest="deform_latent_lr", type=float, default=1.6e-4,
                        help="Adam LR for the per-surfel deformation latent.")
    parser.add_argument("--deform_mlp_lr", dest="deform_mlp_lr", type=float, default=1e-3,
                        help="Adam LR for the deformation MLP weights.")
    parser.add_argument("--deform_width", dest="deform_width", type=int, default=128,
                        help="Deformation MLP hidden width.")
    parser.add_argument("--deform_depth", dest="deform_depth", type=int, default=4,
                        help="Deformation MLP hidden layers.")
    parser.add_argument("--deform_time_freqs", dest="deform_time_freqs", type=int, default=6,
                        help="Fourier frequencies for the scalar frame-time encoding.")
    parser.add_argument("--deform_warmup", dest="deform_warmup", type=int, default=3000,
                        help="Iterations before deformation starts (let the canonical geometry "
                             "form first; important under --cold). Identity (no deform) before this.")
    parser.add_argument("--deform_reg", dest="deform_reg", type=float, default=0.0,
                        help="L2 penalty on (Δpos, Δrot) to keep deformations small/minimal.")

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')
    
    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")

    # === Unbiased Depth (Peng et al.) ===
    # Replaces 2DGS depth distortion with cumulative-opacity surface + convergence loss.
    # When --unbiased is set, the diff_surfel_3D_sh_res_unbiased rasterizer is used
    # (sys.modules swap at module top), lambda_dist is forced to 0, and after iter
    # --unbiased_iter the convergence-loss term is added with weight --lambda_converge.
    parser.add_argument("--unbiased", action="store_true",
                        help="Enable Unbiased Depth: cumulative-opacity surface + convergence loss "
                             "(routes to diff_surfel_3D_sh_res_unbiased; disables lambda_dist)")
    parser.add_argument("--lambda_converge", type=float, default=7.0,
                        help="Weight for the convergence loss (paper default 7.0). Active only when --unbiased")
    parser.add_argument("--unbiased_iter", type=int, default=10000,
                        help="Iteration after which the convergence loss kicks in (paper default 10000)")

    # Shared-resume checkpoint: save EVERYTHING (Gaussians + their optimizer +
    # densif accumulators + INGP model + INGP optimizer + gs_alpha_masks) at one
    # iteration so a sweep over downstream-only knobs (e.g., --lambda_converge)
    # can resume from a single deterministic snapshot without re-running iters
    # 0 → N. The first run with --share_ckpt_iter > 0 SAVES the snapshot if the
    # path doesn't exist; every subsequent run with the same path LOADS it.
    parser.add_argument("--share_ckpt_iter", type=int, default=0,
                        help="When > 0, save a comprehensive resume checkpoint at this iteration "
                             "(or load from it if it already exists). Disabled (0) by default.")
    parser.add_argument("--share_ckpt_tag", type=str, default="",
                        help="Tag for the shared resume checkpoint filename. Empty = auto-derive from "
                             "method+kernel+hybrid_levels+iter so configs don't collide. Path: "
                             "<source_path>/shared_ckpt_<tag>.pth")
    
    # Method argument - baseline, cat, cat_dropout, adaptive, adaptive_add, adaptive_cat, adaptive_zero, adaptive_gate, diffuse, specular, diffuse_ngp, diffuse_offset, hybrid_SH, hybrid_SH_raw, hybrid_SH_post, or residual_hybrid
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "2dgs", "cat", "cat_dropout", "film", "adaptive", "adaptive_add", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse", "specular", "diffuse_ngp", "diffuse_offset", "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "res_3d_double", "3D_SH_add", "3D_SH_cat", "3D_SH_32", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep", "clip_relight", "3D_SH_filmres", "3D_SH_concat", "GEStex", "proberes"],
                        help="Rendering method: 'baseline' (default NeST), 'cat' (hybrid per-Gaussian + hashgrid), 'cat_dropout' (cat with hash dropout during training - use --dropout_lambda), 'adaptive' (learnable per-Gaussian blend), 'adaptive_add' (weighted sum of per-Gaussian and hashgrid features), 'adaptive_cat' (cat with learnable binary blend weights - trains smooth, infers binary), 'adaptive_zero' (cat with weighted hash vs zeros - w=0 skips hash query), 'adaptive_gate' (VQ-AD style gating: soft→STE→hard, L1 regularization toward zeros), 'diffuse' (SH degree 0, no viewdir), 'specular' (full 2DGS with SH), 'diffuse_ngp' (diffuse SH + hashgrid on unprojected depth), 'diffuse_offset' (diffuse SH as xyz offset for hashgrid query), 'hybrid_SH' (activate separately then add: SH→RGB+0.5+clamp + hashgrid→sigmoid, then add+clamp), 'hybrid_SH_raw' (add raw then activate: SH→raw + hashgrid→raw, then sigmoid), 'hybrid_SH_post' (DEPRECATED), 'residual_hybrid' (per-Gaussian SH RGB + hashgrid MLP residual), '3D' (intersection-based SH rendering), '3D_direct' (intersection-based RGB MLP), or '3D_direct_fused' (fused in-kernel MLP, no intersection buffer)")
    parser.add_argument("--probe_tex_res", type=int, default=2048,
                        help="--method proberes: shared texture image resolution (R x R).")
    parser.add_argument("--probe_patch_px", type=float, default=12.0,
                        help="--method proberes: texels covered by +-3 sigma of a median-sized surfel (init; per-surfel exp(raw) adapts from there). Packing break-even is tex_res/sqrt(N) — above it, probes overlap and the head must separate neighbors.")
    parser.add_argument("--probe_tex_levels", type=int, default=16,
                        help="--method proberes: 2D texture-field hash levels (finest level resolution == probe_tex_res).")
    parser.add_argument("--probe_init_dir", type=str, default=None,
                        help="--method proberes: dir from scripts/probe_uv_field.py bake — initializes the pixel image with teacher content (via phi_inv) and installs FIXED probes (via phi). Requires --densify_until_iter 0.")
    parser.add_argument("--probe_nosh_lambda", type=float, default=0.0,
                        help="--method proberes: Texture-GS-style noSH forcing loss weight — L1+DSSIM between the tex_only render (SV killed) and FULL GT. Makes the texture a primary carrier. Paper uses 2.0; try 0.5-2.")
    parser.add_argument("--probe_distill_dir", type=str, default=None,
                        help="--method proberes: directory holding targets.pt from "
                             "scripts/probe_distill_targets.py — per-train-view SIGNED blended "
                             "teacher residual sum(T_i*a_i*res_i). Enabling this FORCES residual_mode 2 "
                             "(signed residual, per-pixel ReLU in Python) so the blend is linear in the "
                             "atlas texels and the objective has no ReLU dead zones.")
    parser.add_argument("--probe_distill_lambda", type=float, default=0.5,
                        help="weight of the residual-matching term vs the GT photometric loss: "
                             "loss = (1-L)*photometric + L*L1(R_probe, R_teacher). Ignored under "
                             "--probe_distill_only.")
    parser.add_argument("--probe_distill_only", action="store_true",
                        help="optimize ONLY the residual-matching term (no GT photometric loss). "
                             "Caps quality at the teacher by construction, but isolates how well the "
                             "probe/atlas can reproduce the teacher's residual field.")
    parser.add_argument("--probe_learn_lr", type=float, default=0.0,
                        help="--method proberes with --probe_init_dir: make the loaded probes LEARNABLE at this LR (texture px per Adam step; try 0.005-0.02). 0 = fixed. Lets colliding surfels migrate apart in the atlas.")
    parser.add_argument("--probe_no_field", action="store_true",
                        help="--method proberes: drop the 2D hash+MLP field entirely — the texture IS the pixel image (the teacher-baked atlas), finetuned directly. The right setting with --probe_init_dir.")
    parser.add_argument("--probe_bake_interval", type=int, default=1,
                        help="--method proberes: re-evaluate the 2D texture field every N iters (cached in between; pixel image still gets exact grads every iter). 1 = every iter. 10-50 is a large fwd/bwd speedup when the field is a slow corrector on a baked atlas.")
    parser.add_argument("--probe_tex_only", action="store_true",
                        help="--method proberes: freeze ALL Gaussian params (geometry + appearance) — only the texture stack trains. Pair with --probe_freeze_head for pure texture fitting.")
    parser.add_argument("--probe_freeze_head", action="store_true",
                        help="--method proberes: freeze the probe head (no optimizer group, detached from geometry) — probes stay at their analytic/init placement while texture trains.")
    parser.add_argument("--probe_abs_placement", action="store_true",
                        help="--method proberes: NO analytic probe base — position = sigmoid(head raw)*tex_res, rotation = raw angle. Pure 'hash-MLP places probes' ablation; scale keeps the metric base.")
    parser.add_argument("--probe_field_lr_scale", type=float, default=1.0,
                        help="--method proberes: LR multiplier for the 2D texture field (hash + MLP) over feat_lr/mlp_lr. Head and pixel image are not boosted. NOTE: 5.0 (hash 1e-1) empirically collapses the scene — residual thrash blasts geometry via dL/duv, densify starves, prune wins. Keep <= 2-3 if raising.")
    parser.add_argument("--probe_tex_base", type=int, default=8,
                        help="--method proberes: coarsest 2D texture-field level resolution (levels log-spaced from here to probe_tex_res).")
    parser.add_argument("--probe_tex_hidden", type=int, default=64,
                        help="--method proberes: texture-field MLP hidden width.")
    parser.add_argument("--probe_smed_freeze_iter", type=int, default=15000,
                        help="--method proberes: iteration at which the running median-surfel-size calibration of probe scale freezes (track densification before, texture stability after).")
    parser.add_argument("--probe_pixel_lr_scale", type=float, default=0.1,
                        help="--method proberes: pixel-image LR = hash feat_lr x this (kept low: sparse noisy grads + eps=1e-15 Adam random-walk otherwise).")
    parser.add_argument("--probe_pixel_decay", type=float, default=1e-4,
                        help="--method proberes: decoupled per-step decay on the pixel image (0 = off). Pulls unsupervised texels back to 0.")
    parser.add_argument("--probe_no_pixels", action="store_true",
                        help="--method proberes: disable the learnable per-texel pixel image riding on the 2D hash+MLP field (tex = MLP(hash(p)) + pixels).")
    parser.add_argument("--probe_c2f_interval", type=int, default=2000,
                        help="--method proberes: iters per extra texture-field hash level (0 = all levels on from start).")
    parser.add_argument("--wsr", action="store_true",
                        help="--method proberes: sort-free WSR finetune (docs/WSR_DISTILL.md). Renders "
                             "through diff_surfel_3D_sh_res_probe_wsr with the order-independent "
                             "weighted-sum composite out=(1-P)*Sum(a*occ*c)/Sum(a*occ); trains a "
                             "per-surfel occlusion logit _wsr_occ plus opacity/SV/probes/texture with "
                             "geometry (xyz/scaling/rotation) LRs forced to 0. Use with --finetune_from "
                             "a trained proberes run; occ is distill-initialized from a sorted "
                             "record_transmittance dump unless --wsr_no_distill_init.")
    parser.add_argument("--wsr_occ_lr", type=float, default=0.01,
                        help="--wsr: LR for the per-surfel occlusion logit.")
    parser.add_argument("--wsr_no_distill_init", action="store_true",
                        help="--wsr: skip the sorted distill dump; occ starts at its PLY/init value.")
    parser.add_argument("--wsr_composite", action="store_true",
                        help="ht=1-style composite finetune (implies --wsr): exact per-pixel "
                             "frontmost fragment (depth argmin, matching the viewer's z-buffer "
                             "core) + occ-weighted mean of the rest. Kills opaque-occluder "
                             "bleed-through by construction; occ only shapes the tail.")
    parser.add_argument("--wsr_unfreeze_geom", action="store_true",
                        help="--wsr: let xyz/scaling/rotation keep their normal LRs instead of 0.")
    parser.add_argument("--wsr_dgate_margin", type=float, default=0.0,
                        help="--wsr: mean-depth gate (?wsr=3). A SORTED pre-pass computes the "
                             "per-pixel alpha-weighted mean depth D-bar and saturation A; the "
                             "occ weight fades to 0 over the relative window "
                             "[D(1+m/2), D(1+3m/2)] scaled by smoothstep(0.6,0.9,A). The "
                             "saturation depth is an opacity INTEGRAL — robust to the order "
                             "swaps that make sorted color compositing pop — so the sort "
                             "anchors visibility only; color stays order-independent. "
                             "0 = off. Typical 0.15. Mutually exclusive with --wsr_gate_tau.")
    parser.add_argument("--wsr_gate_tau", type=float, default=0.0,
                        help="--wsr: transmittance saturation gate (2-pass). A depth-binned "
                             "pre-pass computes per-pixel T(z); fragments whose bin's "
                             "transmittance-in-front < tau are fully discarded in forward AND "
                             "backward — the per-frame relational cut of the deep background "
                             "tail that baked occ can't express. 0 = off. Typical 0.02-0.1. "
                             "Deployed identically by the viewer's ?wsr=2 2-pass mode.")
    parser.add_argument("--film_gamma_init", type=float, default=1.0,
                        help="--method film: init value for per-surfel gamma (hash scale). Default 1.0 (identity). Use e.g. 0.1 to attenuate the hash so beta does more lifting.")
    parser.add_argument("--film_beta_init", type=float, default=0.0,
                        help="--method film: init value for every per-surfel beta channel (bias). Default 0.0. Use e.g. 1.0 to make beta the dominant term at init.")
    parser.add_argument("--film_latent_lr", type=float, default=-1.0,
                        help="LR for the per-surfel FiLM latent (_film_params: gamma + beta), used by "
                             "--method film / 3D_SH_filmres. Default -1 = share feature_lr (0.0025, the "
                             "current behavior). Set e.g. 6e-4 to slow the latent independently of the "
                             "SH/SV feature LR.")
    parser.add_argument("--film_act", type=str, default="identity",
                        choices=["identity", "gamma_relu", "beta_relu", "gamma_sigmoid",
                                 "beta_sigmoid", "double_relu", "double_sigmoid", "gamma_sigm_split"],
                        help="--method 3D_SH_filmres: independent activation on the per-surfel FiLM gamma/beta (mlp_input = gamma_act(gamma)*H + beta_act(beta)). 'identity' (default, both raw); 'gamma_relu'/'beta_relu' relu one only; 'gamma_sigmoid'/'beta_sigmoid' sigmoid one only; 'double_relu'/'double_sigmoid' apply to BOTH; 'gamma_sigm_split' = gamma_sigmoid with a separate gamma PER HASH LEVEL (mlp_input[i] = sigmoid(gamma_l)*H[i] + beta[i], l = i//4; gamma_0 = _film_params col 0, gamma_1..3 = beta cols 21..23 i.e. _film_params cols 22..24; beta raw).")
    parser.add_argument("--film_freeze_gamma_iter", type=int, default=0,
                        help="film/3D_SH_filmres: hold ALL FiLM gammas open (raw pinned to "
                             "--film_freeze_gamma_raw, sigmoid ~1; grads zeroed) for the first N "
                             "iters, so the hash trains ungated before per-surfel/per-level gating "
                             "starts. Covers col 0 (+cols 22:25 for gamma_sigm_split). 0 = off.")
    parser.add_argument("--film_freeze_gamma_raw", type=float, default=4.0,
                        help="Raw gamma value pinned during --film_freeze_gamma_iter (default 4.0 "
                             "-> sigmoid(4)=0.982 ~ fully open). Gammas train from this value "
                             "after release.")
    parser.add_argument("--film_freeze_beta_iter", type=int, default=0,
                        help="--method film/3D_SH_filmres: freeze the per-surfel FiLM beta (offset) for the first N iters of the run (zeroes its gradient; gamma still trains). Default 0 = off. e.g. 2000.")
    parser.add_argument("--film_beta_active_dims", type=int, default=16,
                        help="--method film/3D_SH_filmres: restrict the 16-D FiLM-beta latent to one frequency end (beta dims map to 4 hash levels x 4D, bottom 4D = coarsest/lowest-freq). N>0: keep the bottom N dims (low-freq) active, lock the rest (high-freq) -> N=4 keeps only the coarsest level. N<0: keep the top |N| dims (high-freq) active, lock the lower-freq levels -> N=-4 keeps only the finest level. |N|>=16 (default 16) = all active. Locked dims pinned to 0 + grad-zeroed every iter.")
    parser.add_argument("--lock_gamma", type=float, default=None,
                        help="--method 3D_SH_filmres: lock the FiLM scale gamma to a constant value (bypasses gamma + its --film_act activation, freezes its gradient), so mlp_input = X*H + beta_act(beta). Pass 1.0 for a pure additive latent on the full-strength hash (X=1 + beta=0 is byte-identical to 3D_SH_res). Default None = off (gamma trains).")
    parser.add_argument("--hybrid_levels", type=int, default=5,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
    parser.add_argument("--hash_levels", type=int, default=-1,
                        help="3D_SH_res only: number of HASH levels (preferred name; default -1 = unset, use --hybrid_levels). "
                             "Internally translates to hybrid_levels = encoding.levels - hash_levels. "
                             "With levels=8 in config, --hash_levels K (K in [0..8]) gives K hash levels.")
    parser.add_argument("--decompose_mode", type=str, default=None,
                        choices=[None, "gaussian_only", "ngp_only"],
                        help="Decomposition mode for hybrid_SH visualization: 'gaussian_only' (only per-Gaussian SH), 'ngp_only' (only hashgrid DC residual), or None (normal combined rendering)")
    parser.add_argument(
        "--disable_c2f",
        type=(lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "on")),
        nargs="?", const=True, default=True,
        help="Disable the coarse-to-fine hash-level ramp. Default True = c2f "
             "OFF (the long-standing behavior for ALL non-baseline methods, "
             "incl. 3D_SH_res). Pass `--disable_c2f false` to ENABLE c2f "
             "(also requires --hybrid_levels<5 so there are multiple hash "
             "levels, and coarse2fine.enabled in the yaml). Bare "
             "`--disable_c2f` (no value) still means True for back-compat.")
    parser.add_argument("--dropout_lambda", type=float, default=0.0,
                        help="Hash dropout rate for cat_dropout mode: fraction of Gaussians that don't query hash during training (0.2 = 20%% dropout)")
    parser.add_argument("--lambda_adaptive", type=float, default=0.001,
                        help="Regularization weight for adaptive mode to encourage per-Gaussian features")
    parser.add_argument("--freeze_mlp", action="store_true",
                        help="Freeze MLP weights (random init or from --freeze_mlp_from). Only hashgrid learns. Skips MLP weight gradient computation in CUDA.")
    parser.add_argument("--ste", action="store_true",
                        help="SIGN-AWARE straight-through estimator on the per-Gauss outer ReLU "
                             "(mode 0 / 3D_SH_res + mixed[_3d]). At clamped pixels (pre ≤ 0), "
                             "gradient is allowed to flow ONLY when dL/dpixel < 0 — i.e. loss "
                             "wants the channel HIGHER, so pushing the residual up will release "
                             "the clamp and reduce loss. When dL/dpixel ≥ 0 at a clamped pixel, "
                             "gradient would just push residual deeper negative (forward stays "
                             "clamped, loss unchanged), so it's gated to zero. This avoids the "
                             "deep-negative-runaway divergence of naive STE. Forward unchanged. "
                             "Default off. Mutually exclusive with --lru.")
    parser.add_argument("--detach_res_shape_grad", action="store_true",
                        help="Backward-only (3D_SH_res / res_* / mixed*). Drive the per-Gauss "
                             "alpha/shape gradient from the SV (SH base) color ONLY — detach the "
                             "MLP residual from surfel-shape gradients. The residual still moves "
                             "surfels through the hash-query xyz path and opacity; only the "
                             "shape (G→ρ→scale/rot) gradient stops seeing the residual. Forward "
                             "image is unchanged. Isolates 'should high-frequency residual reshape "
                             "primitives, or should geometry follow the low-frequency SV?'. Pairs "
                             "with --detach_hash_grad (kills the xyz path) for a 2x2 ablation. "
                             "Default off.")
    parser.add_argument("--res_3d_iter", type=int, default=10_000,
                        help="`--method res_3d` STAGE 2: iteration at which to SPLIT each Gauss into "
                             "a 2D residual-only carrier (textured, --kernel) + a 3D EWA SV-only "
                             "carrier (untextured, --kernel2 or --kernel). Stage 1 (the mode 0 → 2 "
                             "activation flip) fires earlier or at the same iter via --res_switch_iter. "
                             "After stage 2 two parallel cascades run inside the single-pass kernel: "
                             "residual → C_tex (T_tex cascade), SV → C_sv (T_sv cascade), final "
                             "image = LRU(C_sv + C_tex, α). The two cascades don't cross-block: a "
                             "wall of opaque tex surfels doesn't saturate T_sv and vice versa. "
                             "If res_switch_iter == res_3d_iter (both default 10000), stages 1 and 2 "
                             "fire together — original single-event behavior preserved. Default 10000.")
    parser.add_argument("--res_switch_iter", type=int, default=10_000,
                        help="`--method res_switch` STAGE 1 / `--method res_3d` STAGE 1: iteration at "
                             "which to flip the residual activation from mode 0 (3D_SH_res — per-Gauss "
                             "outer ReLU in CUDA) to mode 2 (3D_SH_res_sep — per-pixel ReLU after "
                             "blend in Python). Two-phase curriculum: early iters get the locally-"
                             "clamped supervision of mode 0 (easier multi-view convergence); later "
                             "iters get the richer all-Gauss gradient signal of mode 2 (better fine "
                             "detail). With `--lru > 0` on both phases the transition is a smooth "
                             "knee instead of a discontinuous jump. For --method res_3d this fires "
                             "BEFORE OR AT --res_3d_iter (stage 2, the split). Default 10000.")
    parser.add_argument("--lru", type=float, default=0.0,
                        help="LEAKY-RELU slope α for the per-Gauss outer activation in mode 0 "
                             "(3D_SH_res + mixed[_3d]). α == 0 (default) reduces to standard "
                             "ReLU. α > 0 modifies BOTH forward and backward: forward "
                             "`feat = (pre>0) ? pre : α·pre` (clamped sites contribute a "
                             "scaled-down negative value instead of zero), backward gate at "
                             "clamped sites = α instead of 0 (non-zero feedback to MLP/hash). "
                             "Same call sites as --ste — applied in CUDA for the per-Gauss "
                             "clamp in modes 0/1/cat, and in Python for the per-pixel after-"
                             "blend clamp in mode 2 (sep methods). Typical α = 0.01. Mutually "
                             "exclusive with --ste.")
    parser.add_argument("--sv_lru", type=float, default=0.0,
                        help="LEAKY-RELU slope α for the INNER activation — the ReLU on the "
                             "SV/SH base color, relu(SV+sh_bias). Independent of --lru (which is "
                             "the OUTER activation) and NOT mutually exclusive with it: they gate "
                             "different sites. α == 0 (default) = standard ReLU (byte-identical). "
                             "α > 0 lets the SV base recover when SV+sh_bias goes negative "
                             "(otherwise it dies with zero gradient — --lru does not fix this, it "
                             "only protects the residual). For --feature SV this is applied "
                             "Python-side (renderer, F.leaky_relu(feat+0.5, α)) — no rebuild. "
                             "Typical α = 0.01. (--feature SH would need the analogous CUDA change "
                             "in computeColorFromSH; not yet wired.)")
    # ===================== --method GEStex (GES-style sort-free bi-scale) =====================
    # GEStex adapts the GES paper (When Gaussian Meets Surfel) onto the nest hash+MLP+SV
    # residual pipeline. Phases 0-20k behave like `res_switch` (mode 0->2 flip at
    # --ges_phase1_iter, --lru post-blend) with GES surfel opacity + GES pruning. At
    # --ges_joint_iter the hash/MLP residual is baked into an explicit per-surfel RGB
    # texture atlas (trainable leaf), surfel geometry is frozen, 3D Gaussians are spawned,
    # and rendering switches to the sort-free 2-pass pipeline (surfel z-buffer + additive
    # depth-tested Gaussian pass, composited in Python). See docs/GESTEX_MODE.md.
    parser.add_argument("--ges_phase1_iter", type=int, default=10_000,
                        help="GEStex: iter for the 'harden' transition. Prunes surfels with "
                             "w<--ges_prune_w_thresh (saving their positions for later 3D-Gauss "
                             "spawn), ramps surfel opacity to 30, freezes w, disables FastGS "
                             "densification, and flips residual mode 0->2 (post-blend LRU). Aliases "
                             "to --res_switch_iter internally. Default 10000.")
    parser.add_argument("--ges_occlusion_iter", type=int, default=15_000,
                        help="GEStex: iter for GES occlusion culling. Accumulates per-surfel "
                             "frontmost-pixel counts across ALL train views and prunes surfels seen "
                             "as frontmost at fewer than --ges_occlusion_thresh pixels. Default 15000.")
    parser.add_argument("--ges_occlusion_thresh", type=int, default=16,
                        help="GEStex: n_thr for occlusion culling (paper=16 real / 4 synthetic; "
                             "auto-drops to 4 when 'synthetic' in source path). Default 16.")
    parser.add_argument("--ges_opac60_iter", type=int, default=18_000,
                        help="GEStex: (legacy) iter to ramp surfel opacity to 60. Unused by the TS+ "
                             "rising-floor hardening. Default 18000.")
    parser.add_argument("--ges_opac90_iter", type=int, default=19_000,
                        help="GEStex: (legacy) iter to ramp surfel opacity to 90. Unused by the TS+ "
                             "rising-floor hardening. Default 19000.")
    parser.add_argument("--ges_opac_floor_max", type=float, default=0.99,
                        help="GEStex: TS+-style rising opacity FLOOR target. surfel opacity = O_t + "
                             "(1-O_t)*get_opacity with O_t ramped 0 -> this over [ges_phase1_iter, "
                             "ges_joint_iter), staying <= 1 (no ∝opacity>1 geometry-gradient bloat). "
                             "Default 0.99 (near-opaque flat discs by the joint transition).")
    parser.add_argument("--ges_joint_iter", type=int, default=20_000,
                        help="GEStex: iter for 'bake & splat'. Ramps surfel opacity to 255, freezes "
                             "surfel geometry, bakes hash+MLP -> _tex_atlas (trainable leaf), spawns "
                             "3D Gaussians at saved positions, and switches to the sort-free 2-pass "
                             "renderer. Default 20000.")
    parser.add_argument("--ges_prune_w_thresh", type=float, default=0.2,
                        help="GEStex: TS+-style EARLY hard opacity prune. At ges_phase1_iter + "
                             "--ges_hard_prune_offset (while the opacity floor is still low so "
                             "get_opacity is a meaningful signal), prune surfels with get_opacity below "
                             "this. Default 0.2 (TS+ To). Gentle — only genuine non-contributors go.")
    parser.add_argument("--ges_hard_prune_offset", type=int, default=500,
                        help="GEStex: iters after ges_phase1_iter to fire the one-shot hard opacity "
                             "prune. Default 500 (floor still low, junk already separated).")
    parser.add_argument("--ges_surfel_prune_interval", type=int, default=2500,
                        help="GEStex: interval (in [phase1, joint)) for the periodic all-views "
                             "occlusion (T·o-proxy) cull that removes occluded/redundant surfels as they "
                             "harden. 0 disables. Default 2500.")
    parser.add_argument("--ges_shrink_iter", type=int, default=15_000,
                        help="GEStex: iter at which to shrink surfel scale to --ges_shrink_factor. Gives "
                             "the still-trainable geometry room to re-fit the right size under full "
                             "opacity (SV overshoots + discs sit too large once hardened). Default 15000.")
    parser.add_argument("--ges_shrink_factor", type=float, default=0.75,
                        help="GEStex: multiplicative scale shrink applied to surfels at --ges_shrink_iter "
                             "(log-space += log(factor)). 1.0 disables. Default 0.75.")
    parser.add_argument("--ges_beta_end", type=float, default=0.1,
                        help="GEStex: β-ceiling anneal target for --kernel beta_scaled. β is capped at a "
                             "ceiling annealed from its harden-onset value down to this over "
                             "[phase1, joint), driving surfels to near-flat-top discs (β→~0.1) while the "
                             "optimizer stays free below the ceiling. Default 0.1.")
    parser.add_argument("--ges_atlas_res", type=int, default=8,
                        help="GEStex: per-surfel texture atlas resolution R (R x R texels of unbounded "
                             "RGB). Default 8.")
    parser.add_argument("--ges_no_bake", action="store_true",
                        help="GEStex: skip the atlas bake at --ges_joint_iter. The joint stage keeps "
                             "the LIVE hashgrid+MLP as the textured-surfel residual (routed through the "
                             "diff_surfel_gestex cascade with the atlas OFF) and the INGP keeps training. "
                             "Isolates the surfel/texture/Gaussian interplay from the bake.")
    parser.add_argument("--ges_global_lru_iter", type=int, default=-1,
                        help="GEStex: iteration to flip LOCAL (per-Gauss, mode 0) -> GLOBAL (post-blend, "
                             "mode 2) LRU. Default -1 = 5000 (flip before hardening so the signed "
                             "residual can express before surfels go opaque).")
    parser.add_argument("--ges_local_lru", action="store_true",
                        help="GEStex: NEVER flip to global/mode-2 — keep pure 3D_SH_res semantics "
                             "(per-surfel outer ReLU, or LRU if --lru>0, clamping ReLU(SV+0.5)+residual "
                             "BEFORE the blend) for the whole run. A/B for the hypothesis that signed "
                             "textures passing through hardened-opaque surfels let the top layer avoid "
                             "self-correcting. Also skips the --lru 0.01 auto-default (pure ReLU unless "
                             "--lru passed). NOTE: with the bake path, the 20k+ sort-free composite keeps "
                             "its own post-mix activation; the flag's clean A/B window is 0-20k (or use "
                             "--ges_no_bake for consistent mode-0 semantics through the joint stage).")
    parser.add_argument("--ges_floor_start_iter", type=int, default=-1,
                        help="GEStex: iteration at which the opacity FLOOR ramp starts (reaches "
                             "--ges_opac_floor_max at --ges_joint_iter). Default -1 = "
                             "--ges_phase1_iter (10k). Set 5000 for a longer, gentler harden "
                             "(floor ~0.66 by 15k instead of 0.5; less popping shock). Other "
                             "phase1 events (beta ceiling, prunes, shrink) are NOT moved.")
    parser.add_argument("--ges_lock_beta_iter", type=int, default=-1,
                        help="GEStex: iteration from which ALL beta/beta_scaled shapes are HARD-"
                             "LOCKED to --ges_lock_beta_val (pinned every iter; overrides the "
                             "ceiling anneal and optimizer). Fully-flat-disc A/B. -1 = off.")
    parser.add_argument("--ges_lock_beta_val", type=float, default=0.1,
                        help="GEStex: the locked beta value for --ges_lock_beta_iter (default 0.1 "
                             "= flat-top disc; beta_scaled support edge stays hard at rho=3).")
    parser.add_argument("--ges_first_int_iter", type=int, default=-1,
                        help="GEStex: iteration at which the harden switches to FIRST-INTERSECTION "
                             "(tile-depth) SORTING — each tile's surfels blend in ray-disc-"
                             "intersection-depth order (evaluated at the tile-center ray) instead "
                             "of center-depth order. Exact alpha blending at ANY opacity (valid "
                             "for the translucent early-harden), just correctly ordered — fixes "
                             "tilted surfels smearing over the surfels behind them. Rect-AABB only "
                             "(AccuTile cull dropped while active). Set -1 to disable.")
    parser.add_argument("--ges_frontmost_iter", type=int, default=15_000,
                        help="GEStex: iteration at which to ADD the GES-paper FRONTMOST-FIRST "
                             "promotion on top — per pixel, the frontmost surfel by exact "
                             "intersection depth blends FIRST ('the blending order of the other "
                             "surfels is not adjusted'). Only valid once surfels are NEAR-OPAQUE "
                             "(the paper enables it at w>=30; our opacity floor reaches ~0.8 at "
                             "18k) — enabling it while translucent degrades quality. Converges to "
                             "the joint stage's z-buffer. Set -1 to disable.")
    parser.add_argument("--ges_gs_add_start", type=int, default=23_000,
                        help="GEStex: first iter of the error-map 3D-Gauss spawn window (post-joint). "
                             "Default 23000.")
    parser.add_argument("--ges_gs_add_end", type=int, default=33_000,
                        help="GEStex: last iter of the error-map 3D-Gauss spawn window. Default 33000.")
    parser.add_argument("--ges_gs_add_num", type=int, default=0,
                        help="GEStex: total number of 3D Gaussians to add per spawn event via error-map "
                             "sampling (0 = disable error-map spawn; positions from the <w prune are "
                             "always used at joint start). Default 0.")
    parser.add_argument("--ges_gs_prune_interval", type=int, default=500,
                        help="GEStex: interval (iters) for pruning low-contribution 3D Gaussians in the "
                             "joint stage. Default 500.")
    parser.add_argument("--ges_gs_prune_thresh", type=float, default=0.02,
                        help="GEStex: prune 3D Gaussians whose max per-pixel contribution falls below "
                             "this in the joint stage. Default 0.02.")
    parser.add_argument("--ges_s_weight", type=float, default=1.0,
                        help="GEStex: surfel weight s_w in the sort-free composite "
                             "final = (C_S*s_w + C_G)/(s_w + W_G). Default 1.0.")
    # ==========================================================================================
    parser.add_argument("--freeze_hash_iter", type=int, default=0,
                        help="Start periodic hash+MLP freeze at this iteration. 0 = never "
                             "freeze (default). Once active, the 3D_SH_res backward skips all "
                             "hash/MLP gradient work (weight-grad GEMMs, input-chain backprop, "
                             "query_feature<true>, tile flush) except on every Nth iter where "
                             "N is --freeze_hash_period. Geometry backward always runs.")
    parser.add_argument("--freeze_hash_period", type=int, default=10,
                        help="Period for --freeze_hash_iter. Default 10: train hash+MLP on 1 "
                             "of every 10 iters after the freeze starts. Large value (e.g. "
                             "999999) = permanent freeze.")
    parser.add_argument("--freeze_mlp_from", type=str, default=None,
                        help="Load frozen MLP weights from this model_path (loads ngp checkpoint's mlp_fused weights)")
    parser.add_argument("--res_lr_scale", type=float, default=1.0,
                        help="Scale factor for hash encoding and MLP learning rates in 3D_SH_res mode (e.g. 0.1 = 10x lower LR)")
    parser.add_argument("--hash_lr_scale", type=float, default=1.0,
                        help="Scale factor for hash encoding LR only (stacks with --res_lr_scale). e.g. 100 = 100x hash LR")
    parser.add_argument("--nexelparam", action="store_true",
                        help="Adopt Nexels' hash+MLP optimization: base LR 1e-3 (vs YAML's 2e-2 hash) "
                             "with exponential decay to 1e-5 over training. "
                             "(Adam beta2=0.999 is now the default for the INGP optimizer regardless of this flag.) "
                             "Stacks with --res_lr_scale / --hash_lr_scale (those scale the curve).")
    parser.add_argument("--res_warmup", type=int, default=0,
                        help="Disable hash/MLP residual for this many iterations in 3D_SH_res mode (e.g. 10000 = SH-only for first 10k iters)")
    parser.add_argument("--sh_freeze_iter", type=int, default=0,
                        help="Freeze SH (f_dc and f_rest) LR to 0 for the first N iterations, then unfreeze. Lets hashgrid/MLP fit first.")
    parser.add_argument("--texsplit", type=int, default=-1,
                        help="`--method mixed` only: iteration at which to split the live Gaussian set into a "
                             "diffuse-textured (kept) + specular-untextured (duplicated, low opacity) manifold pair. "
                             "-1 disables the split event entirely.")
    parser.add_argument("--texsplit_views", type=int, default=16,
                        help="`--method mixed` only: number of training views to sample for the depth-based "
                             "textured-half reinit at --texsplit.")
    parser.add_argument("--texsplit_tex_frac", type=float, default=0.5,
                        help="`--method mixed[_3d]` only: opacity scale applied to the TEXTURED half at "
                             "--texsplit. The untextured half gets (1 − this) by default (symmetric "
                             "around 0.5). E.g. 0.8 → textured copy α·0.8, untextured copy α·0.2. "
                             "Default 0.5 preserves the historical symmetric split.")
    parser.add_argument("--texsplit_untex_frac", type=float, default=-1.0,
                        help="`--method mixed[_3d]` only: opacity scale for the UNTEXTURED half. -1 = "
                             "auto = 1 − texsplit_tex_frac. Pass explicitly if you want a non-complementary "
                             "split (e.g. 0.8/0.5 — total opacity may not match the pre-split surfel).")
    parser.add_argument("--reg_after_texsplit", action="store_true",
                        help="`--method mixed[_3d]` only: defer the normal-consistency (--lambda_normal "
                             "/ --w_normal) and depth-distortion (--lambda_dist) regularizers until "
                             "AFTER --texsplit fires. Before the split they're forced to 0 so the "
                             "single pre-split surfel set can settle freely; after, the textured-half "
                             "beta surfels alone receive the geometry constraints (the per-pixel mask "
                             "by _is_textured[max_contrib_idx] is always on for mixed[_3d]). Default off.")
    parser.add_argument("--freeze_prim", type=int, default=0,
                        help="Freeze per-Gaussian APPEARANCE parameters only (SH f_dc/f_rest, SB sb_params, "
                             "SG sg_directions/sg_sharpness/sg_rgb, SV sv_sites/sv_colors) for the first N iters. "
                             "Geometry (xyz, scale, rotation), opacity, kernel shape, per-Gaussian hashgrid "
                             "features, and the hash+MLP path all keep training. Lets the hashgrid residual "
                             "fit the scene alone before directional color lobes engage. Also delays the SH "
                             "degree progression by N. Stacks with --sh_freeze_iter.")
    parser.add_argument("--activation_bias", nargs=2, type=float, default=[0.5, 0.0],
                        help="Activation biases [sh_bias, res_bias] for 3D_SH_res/cat/32. "
                             "color = ReLU(ReLU(SH+sh_bias) + residual+res_bias). Default: [0.5, 0.0]")
    parser.add_argument("--eval_depth", action="store_true",
                        help="Render and save depth maps (expected and median) during final evaluation")
    parser.add_argument("--use_xyz_mode", action="store_true",
                        help="Use rasterized xyz instead of unprojected depth (diffuse_ngp/diffuse_offset only)")
    parser.add_argument("--scout_lambda", type=float, default=0.01,
                        help="Weight for scout loss in diffuse_offset xyz mode (moves Gaussians toward offset target)")
    parser.add_argument("--random_background", action="store_true",
                        help="Use random per-pixel background during training for unbiased opacity learning. Eval uses black background.")
    # Volumetric noise-pressure knob — sibling of --random_background.
    # Waits until iter >= --random_mesh_after (default 5000) so the initial
    # 3DGS/2DGS optimization can settle geometry, then composites a per-iter
    # blocky RGB noise pattern (grid G×G upsampled nearest to full frame)
    # into pixels where the LIVE rend_alpha ≥ 0.5 — i.e., the T=0.5 median
    # crossing has been reached (a pixel-covering opaque region exists in
    # the volume). Noise mask replaces the solid random_bg inside that
    # region; solid bg still fills outside. Same seeded per-iter noise is
    # composited into BOTH the render and the GT, so semi-transparent
    # surfels inside the median-crossing region incur an inescapable |0.5·noise|
    # gradient penalty (see finetune_mesh_cull.py's --random_mesh block for
    # the math). Forces the leading-opacity frontier to fully saturate →
    # opaque manifold at the T=0.5 crossing depth. No pre-computed mesh
    # needed; the median-depth surface is derived per-view from the render.
    parser.add_argument("--random_mesh", action="store_true",
                        help="After --random_mesh_after iters, composite blocky RGB "
                             "noise into pixels where rend_alpha ≥ 0.5 (T=0.5 median "
                             "crossing reached). Forces the leading-opacity frontier "
                             "to opacify — the on-the-fly analog of NGS noise Gaussians. "
                             "Stacks with --random_background (solid RGB outside the "
                             "silhouette, blocky noise inside).")
    parser.add_argument("--random_mesh_after", type=int, default=5000,
                        help="Iter to start injecting the median-crossing noise. "
                             "Default 5000 lets 3DGS/2DGS get past initial geometry "
                             "chaos before the extra pressure kicks in.")
    parser.add_argument("--random_mesh_grid", type=int, default=32,
                        help="Noise-grid resolution (G means G×G blocks upsampled "
                             "nearest to full frame). Larger → finer noise (harder "
                             "to hide with partial opacity). Default 32.")
    # DEPTH-LOCATED noise (NGS-faithful; supersedes --random_mesh for opaque-
    # manifold training). Two-pass per iter after --random_mesh_after: a
    # no_grad pre-render captures this view's median-depth map, which is then
    # installed (+eps) as the CUDA per-pixel occluder for the loss render —
    # fragments behind the wall are excluded fwd+bwd, and the noise is
    # composited on the WALLED (1 − rend_alpha), i.e. weighted by the
    # transmittance AT the median. Deep material cannot silence the noise, so
    # the gradient specifically closes frontier cracks (the --random_mesh
    # back-plate lets any-depth saturation escape → ragged tail). Costs ~1
    # extra forward per iter. Requires the DEFAULT rasterizer build (median
    # slot = T=0.5 crossing; NOT a LAST_DEPTH_MODE / T_CROSSING build).
    parser.add_argument("--random_mesh_depth", action="store_true",
                        help="Depth-located noise wall at each iter's own median "
                             "depth (per-pixel CUDA occluder + noise on walled "
                             "alpha). Stacks with --random_background; uses "
                             "--random_mesh_after / --random_mesh_grid. 3D_SH_res-"
                             "family rasterizer only (needs set_occluder_depth).")
    parser.add_argument("--random_mesh_wall_eps", type=float, default=0.01,
                        help="Depth pushed behind the median before installing the "
                             "wall (metres, kernel-depth units). Keeps the median "
                             "surfel itself in front of its own wall. Default 0.01.")
    parser.add_argument("--random_mesh_wall_anneal", type=int, default=0,
                        help="If >0: anneal the wall depth from --random_mesh_wall_eps_start "
                             "down to --random_mesh_wall_eps linearly over this many iters "
                             "after --random_mesh_after. Softens the depth-wall shock (the "
                             "loss suddenly demanding the front-of-median prefix explain "
                             "all of GT). 0 = fixed eps from the start.")
    parser.add_argument("--random_mesh_wall_eps_start", type=float, default=0.15,
                        help="Starting (deep/mild) wall eps for the anneal (metres). "
                             "Only used when --random_mesh_wall_anneal > 0.")
    parser.add_argument("--noise_debug_interval", type=int, default=1000,
                        help="Every N iters (while --random_mesh/--random_mesh_depth "
                             "noise is active), dump the noise-composited render+GT "
                             "pair the loss actually saw into training_output/ "
                             "({iter}_noised_render.png / _noised_gt.png). 0 = off.")
    parser.add_argument("--backface_cull", action="store_true",
                        help="Per-view surfel backface culling during training "
                             "(override_opacity — no CUDA changes). Disc normals "
                             "oriented outward via cloud centroid; surfels facing "
                             "away are zero-opacitied for that view. Closes the "
                             "opposite-shell escape for noise pressure and keeps "
                             "last-fragment depths view-consistent (TSDF holes).")
    parser.add_argument("--backface_cull_after", type=int, default=5000,
                        help="Iter to start backface culling (early normals are noise).")
    parser.add_argument("--backface_cull_cos", type=float, default=0.2,
                        help="FINAL cull threshold: cull when dot(view_dir, "
                             "outward_normal) exceeds this (0.2 ≈ only clearly "
                             "back-facing; 0 = cull at exactly 90°).")
    parser.add_argument("--backface_cull_anneal", type=int, default=0,
                        help="If >0: anneal the cos threshold linearly from "
                             "--backface_cull_cos_start down to --backface_cull_cos "
                             "over this many iters after --backface_cull_after. "
                             "Strictness ramps up gradually → representation "
                             "reorients without a divergence shock. 0 = brute "
                             "constant threshold from the start iter.")
    parser.add_argument("--backface_cull_cos_start", type=float, default=0.9,
                        help="Starting (mild) cos threshold for the anneal — 0.9 "
                             "culls only surfels facing almost directly away "
                             "(~26° cone). Only used when --backface_cull_anneal > 0.")
    parser.add_argument("--cold", action="store_true",
                        help="Cold start: skip 2DGS warmup phase and optimize Nest representation from scratch (no checkpoint loading, hash_in_CUDA always on)")
    parser.add_argument("--scratch", action="store_true",
                        help="Train from scratch: ignore existing warmup checkpoint and train with full warmup phase")

    # Anti-aliasing arguments (Zip-NeRF style distance-based hash attenuation)
    parser.add_argument("--aa", type=float, default=0.0,
                        help="Anti-aliasing scale factor. 0=disabled (default), >0=enabled (recommended: 1.0). Attenuates high-frequency hash levels for distant Gaussians.")
    parser.add_argument("--aa_threshold", type=float, default=0.01,
                        help="Skip hash query when average level weight < threshold (inference optimization)")

    # Adaptive_cat arguments
    parser.add_argument("--lambda_adaptive_cat", type=float, default=0.01,
                        help="Entropy regularization weight for adaptive_cat binarization (pushes weights toward 0 or 1)")
    parser.add_argument("--adaptive_cat_anneal_start", type=int, default=15000,
                        help="Iteration to start annealing adaptive_cat entropy regularization (ramps from 0 to full strength)")
    parser.add_argument("--adaptive_cat_inference", action="store_true",
                        help="Use binary decisions at inference (weight>=threshold uses Gaussian only, skips intersection; weight<threshold uses hashgrid)")
    parser.add_argument("--adaptive_cat_threshold", type=float, default=0.9,
                        help="Inference threshold for adaptive_cat: weight>=threshold uses Gaussian-only (default 0.9, conservative)")

    # Adaptive_zero arguments
    parser.add_argument("--lambda_adaptive_zero", type=float, default=0.0,
                        help="BCE entropy regularization weight for adaptive_zero binarization (pushes hash weights toward 0 or 1)")
    parser.add_argument("--bce_threshold", type=float, default=0.5,
                        help="Threshold for BCE regularization repulsion point (default 0.5, try 0.1 to push weights away from inference threshold)")
    parser.add_argument("--hash_lambda", type=float, default=0.0,
                        help="L1 regularization weight pushing adaptive_zero weights toward 1 (favor hash queries over zeros)")
    parser.add_argument("--adaptive_zero_anneal_start", type=int, default=15000,
                        help="Iteration to start annealing adaptive_zero entropy regularization (ramps from 0 to full strength)")
    parser.add_argument("--relocation", type=str, default="clone", choices=["clone", "reset"],
                        help="How to handle adaptive weights for new/relocated Gaussians: 'clone' copies from source, 'reset' initializes to 0 (sigmoid=0.5)")

    # Adaptive_gate arguments (Gumbel-STE with forced training)
    parser.add_argument("--no_gumbel", action="store_true",
                        help="Disable Gumbel noise in adaptive_gate mode (use deterministic STE instead)")
    parser.add_argument("--hard_switch", action="store_true",
                        help="Use hard switching (no weight multiplication) during training - replicates cat mode behavior")
    parser.add_argument("--lambda_sparsity", type=float, default=0.005,
                        help="Sparsity penalty on gate probability (encourages gates to stay closed)")
    parser.add_argument("--force_ratio", type=float, default=0.2,
                        help="Fraction of Gaussians forced to use hash during training (0.2 = 20%%)")
    parser.add_argument("--gate_init", type=float, default=2.0,
                        help="Initial gate logit value (positive = favor hash, sigmoid(2)≈0.88)")
    parser.add_argument("--gate_bce_lambda", type=float, default=0.0,
                        help="BCE loss weight for gate probabilities (encourages binary 0/1 decisions)")
    parser.add_argument("--adaptive_gate_inference", action="store_true",
                        help="Use hard gating at inference (probability>0.5 uses hash, otherwise zeros)")

    # Temperature annealing (for adaptive_zero and adaptive_gate)
    parser.add_argument("--temp_start", type=float, default=1.0,
                        help="Initial temperature for sigmoid (1.0 = normal sigmoid)")
    parser.add_argument("--temp_end", type=float, default=1.0,
                        help="Final temperature for sigmoid (>1 makes sigmoid sharper, e.g., 10.0)")
    parser.add_argument("--temp_anneal_start", type=int, default=3000,
                        help="Iteration to start temperature annealing")
    parser.add_argument("--temp_anneal_end", type=int, default=25000,
                        help="Iteration to reach final temperature")

    # 3D mode arguments (intersection-based SH rendering, uses --hybrid_levels like cat mode)
    parser.add_argument("--max_intersections_per_pixel", type=int, default=32,
                        help="Maximum intersections per pixel for 3D mode (memory cap, default 32)")
    parser.add_argument("--mlp_3D_hidden", type=int, default=16,
                        help="Hidden dimension for 3D mode MLP (default 32, ≤128 for FullyFusedMLP)")
    parser.add_argument("--mlp_3D_layers", type=int, default=2,
                        help="Number of hidden layers for 3D mode MLP (default 2)")

    # Parabola regularization (additive with BCE)
    parser.add_argument("--lambda_parabola", type=float, default=0.0,
                        help="Weight for w*(1-w) penalty pushing weights away from 0.5 (additive with BCE)")

    # MCMC arguments - based on "3D Gaussian Splatting as Markov Chain Monte Carlo"
    parser.add_argument("--mcmc", action="store_true",
                        help="Enable MCMC-based Gaussian management (replaces traditional densification)")
    parser.add_argument("--mcmc_deficit", action="store_true",
                        help="MCMC deficit mode: delete dead Gaussians until reaching cap_max, then normal MCMC. Use when init points > cap_max.")
    parser.add_argument("--mcmc_fps", action="store_true",
                        help="MCMC mode with farthest point subsampling: subsample init points to cap_max before training using FPS algorithm. Cached for standard cap_max values (40k, 100k, 400k, 1M).")
    parser.add_argument("--mcmc_sample", type=str, default="opacity",
                        choices=["opacity", "gradient"],
                        help="MCMC donor-sampling distribution: 'opacity' (default, sample ∝ α) "
                             "or 'gradient' (Nexels-style, sample ∝ accumulated xyz gradient — "
                             "targets high-error regions).")
    parser.add_argument("--cap_max", type=int, default=-1,
                        help="Maximum number of Gaussians (required for MCMC mode)")
    parser.add_argument("--opacity_reg", type=float, default=0.0,
                        help="L1 regularization weight on opacity (0 = disabled)")
    parser.add_argument("--scale_reg", type=float, default=0.0,
                        help="L1 regularization weight on scale (0 = disabled)")
    parser.add_argument("--noise_lr", type=float, default=5e5,
                        help="SGLD noise learning rate multiplier (MCMC mode)")
    parser.add_argument("--mcmc_depth_reinit", type=int, default=0,
                        help="Iteration for first depth reinitialization in MCMC mode (0 = disabled)")
    parser.add_argument("--reinit_interval", type=int, default=0,
                        help="Repeat depth reinit every N iterations after mcmc_depth_reinit (0 = only once)")
    parser.add_argument("--reinit_end", type=int, default=-1,
                        help="Stop repeating reinit after this iteration (-1 = total_iterations - 15000)")

    # Learnable skybox background for outdoor scenes
    parser.add_argument("--background", type=str, default="none",
                        choices=["none", "skybox_dense", "skybox_sparse", "hashgrid", "hashgrid_relu", "hashgrid_sep"],
                        help="Background mode: 'none' (solid color), 'skybox_*' (learnable texture), 'hashgrid' (composite features before MLP), 'hashgrid_relu' (ReLU on BG features), 'hashgrid_sep' (separate MLP decode, composite RGB)")
    parser.add_argument("--skybox_lr", type=float, default=1e-3,
                        help="Learning rate for skybox texture")
    parser.add_argument("--skybox_res", type=int, default=512,
                        help="Skybox texture resolution (height; width=2*height)")

    # Background hashgrid settings (for --background hashgrid)
    parser.add_argument("--bg_hashgrid_levels", type=int, default=None,
                        help="Number of levels for background hashgrid (default: same as main method's total levels)")
    parser.add_argument("--bg_hashgrid_dim", type=int, default=None,
                        help="Feature dimension per level for background hashgrid (default: same as main method)")
    parser.add_argument("--bg_hashgrid_size", type=int, default=19,
                        help="log2 of hash table size for background hashgrid (default: 19 = 512K entries)")
    parser.add_argument("--bg_hashgrid_res", type=int, default=512,
                        help="Finest resolution for background hashgrid (default: 512)")
    parser.add_argument("--bg_hashgrid_lr", type=float, default=1e-2,
                        help="Learning rate for background hashgrid (default: 1e-2)")
    parser.add_argument("--bg_hashgrid_start_iter", type=int, default=0,
                        help="Iteration to start BG hashgrid training (default: 0, starts with switch_iter). Set higher to let FG train first.")
    parser.add_argument("--bg_hashgrid_radius", type=float, default=500.0,
                        help="Radius of the background sphere for ray intersection (default: 500.0, should be > scene extent)")

    # BCE opacity regularization - reduce semi-transparent foggy Gaussians
    parser.add_argument("--bce", action="store_true",
                        help="Enable BCE regularization on opacity to reduce semi-transparent Gaussians")
    parser.add_argument("--bce_iter", type=int, default=5000,
                        help="Apply BCE regularization for the last N iterations (default: 5000)")
    parser.add_argument("--bce_lambda", type=float, default=0.01,
                        help="BCE regularization weight (default: 0.01)")
    parser.add_argument("--bce_solo", action="store_true",
                        help="Disable MCMC opacity regularization during BCE phase (avoids conflicting gradients)")
    parser.add_argument("--bce_solo_adaptive", action="store_true",
                        help="Like --bce_solo but sets BCE threshold to median opacity at BCE start (pushes weak down, strong up)")
    parser.add_argument("--bce_adaptive", action="store_true",
                        help="Sets BCE threshold to median opacity at BCE start (keeps opacity/scale reg active)")
    parser.add_argument("--bce_adaptive_stat", type=str, default="median", choices=["median", "mean"],
                        help="Statistic for adaptive BCE threshold (default: median)")

    # Beta (Gaussian sharpening) override
    parser.add_argument("--beta", type=float, default=None,
                        help="Override tg_beta from config (higher = sharper Gaussians, more opaque throughout)")

    # --feature beta: view-dependent color function. Default 'sh' = spherical harmonics
    # (current behavior). 'beta' = spherical-beta lobes as in beta-splatting paper.
    parser.add_argument("--feature", type=str, default="sh",
                        choices=["sh", "beta", "sg", "voronoi", "SV"],
                        help="View-dependent color: 'sh' (spherical harmonics, default), 'beta' (spherical-beta), 'sg' (MEGS-2 spherical Gaussians), 'voronoi' (legacy hybrid SH+SV, additive on top of SH-DC), or 'SV' (reference-faithful Spherical Voronoi: SV-only color, pcd-RGB init, no SH-DC, no bias — matches sphericalvoronoi/radiance).")
    parser.add_argument("--sv_l1", type=float, default=0.0,
                        help="L1 regularization on _sv_colors. 0 = off (2dgs-voronoi default). "
                             "Older radiance preset used 1e-5 for blender/tandt/db, 0 for mip-360.")
    parser.add_argument("--sites_lr", type=float, default=2e-3,
                        help="LR for _sv_sites. SV mode: cosine init (decays to sites_lr*0.1, "
                             "with 1k warmup + 3k freeze). voronoi mode: exponential init "
                             "(decays to --sites_lr_final). Default 2e-3 matches 2dgs-voronoi.")
    parser.add_argument("--sites_lr_final", type=float, default=2e-4,
                        help="--feature voronoi final LR for _sv_sites (exponential decay end). "
                             "Unused by --feature SV (cosine scheduler hardcodes 0.1× init).")
    parser.add_argument("--sv_metric", type=str, default="l2",
                        choices=["l2", "cosine"],
                        help="SV logit metric: 'l2' (radiance default, -τ·||s_norm − ω||) or "
                             "'cosine' (paper formulation, s · ω = unconstrained dot product). "
                             "Cosine avoids L2's sqrt gradient blowup near alignment.")
    parser.add_argument("--sv_color_lr", type=float, default=8e-4,
                        help="LR for _sv_colors. Default 8e-4 matches 2dgs-voronoi.")
    parser.add_argument("--sv_dc", action="store_true",
                        help="--feature SV: add an explicit per-Gaussian view-independent "
                             "DC channel [N, 3]. SV becomes a directional residual on top. "
                             "Inits _sv_dc to pcd RGB and _sv_colors to zeros so the Gaussian "
                             "starts at its pcd color. Diverges from the strict reference "
                             "(which has no DC term) — opt-in.")
    parser.add_argument("--sv_dc_lr", type=float, default=2.5e-3,
                        help="--feature SV + --sv_dc: LR for _sv_dc. Default matches the "
                             "reference's sh_lr (0.0025).")
    parser.add_argument("--sv_tau_lr", type=float, default=6e-3,
                        help="--feature SV: LR for _sv_tau (separate sharpness param). "
                             "2dgs-voronoi default 0.006.")
    parser.add_argument("--sv_warmup_iter", type=int, default=1000,
                        help="--feature SV: sites/tau LR is held at 0 for the first N iters "
                             "so colors settle before geometry of the spherical partition moves. "
                             "2dgs-voronoi default 1000.")
    parser.add_argument("--sv_freeze_last", type=int, default=3000,
                        help="--feature SV: sites/tau LR is held at 0 for the last N iters "
                             "so colors fine-tune against fixed cells. 2dgs-voronoi default 3000.")
    parser.add_argument("--sb_number", type=int, default=7,
                        help="Number of directional primitives per Gaussian (K). "
                             "Used by --feature beta/sg/voronoi/SV. Default 7 (2dgs-voronoi). "
                             "Beta-splatting paper uses 2; pass --sb_number 2 to reproduce.")
    parser.add_argument("--sb_params_lr", type=float, default=0.0025,
                        help="--feature beta: LR for sb_params (per-primitive rgb/theta/phi/beta_raw). Default 0.0025")
    parser.add_argument("--sb_beta_lr", type=float, default=0.001,
                        help="--feature beta: LR for shared per-Gaussian sharpness. Default 0.001")

    # Beta kernel arguments
    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "beta", "beta_scaled", "flex", "general", "nexel"],
                        help="Kernel type: 'gaussian' (default exp(-0.5*r²)), 'beta' (pow(1-r², shape) with r∈[0,1]), 'beta_scaled' (same but r∈[0,3] to match 3σ Gaussian extent), 'flex' (Gaussian with learnable per-Gaussian beta), 'general' (Isotropic Generalized Gaussian), or 'nexel' (per-axis learnable gamma exponents, G=exp(-0.5*(s_x^2γx + s_y^2γy)))")
    parser.add_argument("--kernel2", type=str, default=None,
                        choices=["gaussian", "beta", "beta_scaled", "flex", "general", "nexel"],
                        help="`--method mixed_3d` only: kernel for the UNTEXTURED (EWA 3D-ellipsoid) "
                             "primitives, overriding --kernel for that half. Unset → untextured use "
                             "--kernel (current behavior). E.g. `--kernel beta_scaled --kernel2 gaussian` "
                             "→ textured = 2D beta_scaled surfels, untextured = Gaussian EWA ellipsoids.")
    parser.add_argument("--l2", action="store_true",
                        help="`--method mixed_3d[_sep]` only: per-GAUSS photometric-loss routing in the "
                             "rasterizer backward. One forward pass; the kernel returns two image-output "
                             "slots (numerically identical, separate autograd nodes). Python wires "
                             "L1+SSIM(image_tex, gt) to slot 0 and L2(image_untex, gt) to slot 1; the "
                             "backward receives two upstream image gradients and routes them per Gauss "
                             "inside CUDA (textured Gauss accumulate from slot 0's grad, untextured "
                             "from slot 1's grad). No double-render, no detach trickery. Hash weights "
                             "and MLP only flow textured signal (they're never queried for untex). "
                             "Pre-texsplit (no untextured rows) → all-textured kernel path → L2 leg "
                             "contributes nothing → byte-identical to default L1+SSIM loss. "
                             "Mutually exclusive with --l1.")
    parser.add_argument("--l1", action="store_true",
                        help="`--method mixed_3d[_sep]` only: same per-Gauss CUDA routing as --l2, but "
                             "the untex leg uses L1 instead of L2 (textured Gauss still get L1+SSIM). "
                             "Untex Gauss therefore receive a uniformly-scaled error signal rather than "
                             "an error² that amplifies outliers — useful when the untex/EWA half is "
                             "modelling smooth volumetric background where L2 over-penalises edges. "
                             "Mutually exclusive with --l2.")
    parser.add_argument("--freeze_beta", type=float, default=None,
                        help="Freeze beta kernel shape to a fixed value (e.g., 3.0 for semisoft). Disables shape optimization.")
    parser.add_argument("--grads", type=str, default="vanilla",
                        choices=["vanilla", "abs"],
                        help="Densification gradient mode: 'vanilla' (signed gradients, 2DGS default) "
                             "or 'abs' (AbsGS cancellation-free absolute gradients)")
    parser.add_argument("--lowpass", action="store_true",
                        help="Enable low-pass filter gradient in backward (propagate geometry gradients "
                             "through rho2d path to transMat). Off by default (matches reference 2DGS).")
    parser.add_argument("--pixel_center", action="store_true",
                        help="Use pixel-center convention (pixf = pix + 0.5, ndc2pix offset = W/2). "
                             "Off by default = pixel-corner convention (matches reference 2DGS).")
    parser.add_argument("--antialiasing", type=float, default=0.0,
                        help="Nexels-style hash-grid anti-aliasing down-weight factor. "
                             "0 disables (default). Now unit-matched to Nexels' grid_threshold_factor: "
                             "1.0 = Nexels paper default; 0.5 mild; 2.0 aggressive. Must be set "
                             "from iter 1 (model trained without AA can't have AA enabled at render time).")
    parser.add_argument("--aa_2dgs", type=float, default=0.0,
                        help="AA-2DGS Jacobian-based anti-aliasing kernel size (σ) for 3D_SH_res. "
                             "0 disables (default). Typical value 0.1. Replaces the "
                             "min(rho3d, rho2d) heuristic with a mathematically continuous "
                             "object-space mip filter (Σ'_local = I + σ·J·Jᵀ).")
    parser.add_argument("--detach_hash_grad", action="store_true",
                        help="Detach positional gradients from hashgrid in CAT mode (geometry follows per-Gaussian features only)")
    parser.add_argument("--densfix", action="store_true",
                        help="--method 3D_SH_res only: route through the diff_surfel_3D_sh_res_densfix "
                             "clone and EXCLUDE the hashgrid query-point term from the AbsGS "
                             "densification proxy. Surfels still reposition on the full SV+hash "
                             "gradient (transMat/mean3D unchanged); only the densification score drops "
                             "the hash-inflated boost, so densify reflects geometry/reconstruction need "
                             "rather than texture frequency. Off = byte-identical to base 3D_SH_res.")
    parser.add_argument("--trunc", action="store_true",
                        help="--method 3D_SH_res only: route through the diff_surfel_3D_sh_res_trunc "
                             "clone and ramp a POST-blend truncation exit threshold (set_exit_T) from "
                             "1e-4 (off) to --trunc_exit_T over [--trunc_ramp_start, --trunc_ramp_end]. "
                             "The forward walk stops once T drops below the threshold — the crossing "
                             "fragment still blends, so an opaque terminator can drive T→0 and kill the "
                             "(1−rend_alpha)·noise composite (pair with --random_mesh). Trains an "
                             "opacity-cliff 'base plate' behind a translucent textured prefix. "
                             "Off / pre-ramp = byte-identical to base 3D_SH_res.")
    parser.add_argument("--trunc_exit_T", type=float, default=0.5,
                        help="--trunc: final exit threshold. The cliff forms where cumulative alpha "
                             "reaches 1−exit_T (0.5 = median crossing; 0.3 = deeper, gentler).")
    parser.add_argument("--trunc_ramp_start", type=int, default=10000,
                        help="--trunc: iteration where the exit_T ramp begins (before: 1e-4 = off). "
                             "Start at/after densify_until_iter so ADC resets/densification finish first.")
    parser.add_argument("--trunc_ramp_end", type=int, default=15000,
                        help="--trunc: iteration where exit_T reaches --trunc_exit_T (then pinned).")
    parser.add_argument("--gap_noise", action="store_true",
                        help="--method 3D_SH_res only: macro-gap truncation via the "
                             "diff_surfel_3D_sh_res_trunc clone. From --gap_noise_after, each "
                             "ray ends BEFORE blending the first significant fragment lying "
                             "more than --gap_noise_thresh behind the front manifold's deepest "
                             "member; the remaining transmittance carries the --random_mesh "
                             "noise composite (noise fills the inter-manifold void). Faint "
                             "fragments within --gap_noise_margin extend the manifold, so the "
                             "wall never presses on a manifold's trailing fringe. Rays without "
                             "a macro gap are untouched. Pair with --random_background "
                             "--random_mesh.")
    parser.add_argument("--gap_noise_after", type=int, default=5000,
                        help="--gap_noise: iteration at which the gap wall activates.")
    parser.add_argument("--gap_noise_thresh", type=float, default=0.2,
                        help="--gap_noise: minimum inter-manifold gap (scene units) that "
                             "triggers truncation. Chair profiling: intra-manifold structure "
                             "lives at <=0.05, true opposite-face voids at >=0.3 — 0.15-0.2 "
                             "separates them cleanly.")
    parser.add_argument("--gap_noise_margin", type=float, default=0.05,
                        help="--gap_noise: faint fragments within this distance of the "
                             "manifold's deepest member extend the manifold (NGS-style "
                             "erosion buffer; protects the trailing fringe).")
    parser.add_argument("--gap_noise_void_mass", type=float, default=0.1,
                        help="--gap_noise v2: maximum total opacity mass allowed inside a "
                             "void. Dense material in the gap merges the clusters "
                             "(continuous translucent media never truncate; faint bridges "
                             "cannot walk the wall).")
    parser.add_argument("--gap_noise_sat_T", type=float, default=0.05,
                        help="--gap_noise saturation gate: the wall only COMMITS on rays "
                             "whose full (untruncated) march ends with final T below this — "
                             "rays that fully saturate anyway. Rays blending with the "
                             "background (semi-transparent floaters over bg, silhouettes) "
                             "render as if no wall existed and take no noise pressure. "
                             "Implemented as a deferred snapshot commit in the forward "
                             "kernel; <= 0 restores the legacy immediate wall.")
    parser.add_argument("--gap_noise_T_lo", type=float, default=0.05,
                        help="--gap_noise v2: only truncate when leaked transmittance at "
                             "the void exceeds this (already-opaque rays exit the "
                             "mechanism — explicit self-annealing).")
    parser.add_argument("--gap_noise_arm_T", type=float, default=0.6,
                        help="--gap_noise: the wall may only fire when transmittance at the "
                             "candidate fragment is BELOW this (front manifold absorbed >= "
                             "1-arm_T of the ray). Prevents floater amplification: a lone "
                             "low-alpha floater leaves T high and can never truncate the "
                             "scene behind it. 1.0 = guard off; 0.6 default.")
    parser.add_argument("--gap_noise_alpha_min", type=float, default=0.05,
                        help="DEPRECATED (v1 detector). The v2 mass automaton replaces the "
                             "per-fragment significance floor with void opacity-mass "
                             "accounting (--gap_noise_void_mass); this flag is unused.")
    parser.add_argument("--trunc_cliff_reg", type=float, default=0.0,
                        help="--trunc: per-pixel cliff-sharpness penalty "
                             "lambda * mean(T_final) over crossed pixels (rend_alpha >= 0.5), "
                             "active after --trunc_ramp_start. The deterministic, color-free "
                             "alternative/complement to the back-plate noise: drives the "
                             "crossing fragment opaque and widens the margin to the "
                             "inclusion-flip discontinuity. Typical 0.1-0.5.")
    parser.add_argument("--reset_until_iter", type=int, default=-1,
                        help="Override training_cfg.reset_until_iter from the YAML: periodic "
                             "opacity resets (every opacity_reset_interval) stop after this "
                             "iteration. Use e.g. 15000 with --trunc so no reset lands after "
                             "the truncation step (a reset under truncation re-fires the full "
                             "noise penalty and smashes the forming opacity cliff). -1 = YAML value.")
    parser.add_argument("--lambda_shape", type=float, default=0.0,
                        help="L1 regularization weight on beta kernel shape parameter (pushes toward 0 = hard disks). Default 0 = shapes stay at their init.")
    parser.add_argument("--shape_iter", type=int, default=0,
                        help="Apply shape regularization for the last N iterations only (0 = always active, default: 0)")
    parser.add_argument("--lambda_flex_beta", type=float, default=0.0001,
                        help="L1 regularization weight on flex kernel beta parameter (prevents runaway sharpening)")
    parser.add_argument("--l1_hash", type=float, default=0.0,
                        help="L1 regularization on hashgrid embeddings to encourage sparsity (0.0 = disabled)")
    parser.add_argument("--l1_sh_rest", type=float, default=0.0,
                        help="L1 regularization on higher-order SH (features_rest) to keep SH low-frequency (0.0 = disabled)")
    parser.add_argument("--tv_hash", type=float, default=0.0,
                        help="Total variation regularization on hashgrid to penalize uniform grey while preserving edges/detail (0.0 = disabled)")
    parser.add_argument("--genreg", type=str, default="basic",
                        choices=["basic", "decay", "scaled", "scaled_decay"],
                        help="General kernel regularization mode: 'basic' (constant), 'decay' (linear decay over training), 'scaled' (scaled by RGB loss), 'scaled_decay' (both)")
    parser.add_argument("--aabb", type=str, default="2dgs",
                        choices=["2dgs", "adr_only", "rect", "adr", "beta", "accutile", "snugbox"],
                        help="AABB mode: '2dgs' (square, fixed 4σ - default), 'adr_only' (square, AdR cutoff), 'rect' (rectangular, fixed 4σ), 'adr' (rectangular + AdR cutoff), 'beta' (fixed r=1 for beta kernels), 'accutile'/'snugbox' (AdR + rect + AccuTile ellipse cull)")
    parser.add_argument("--warmup", type=str, default=None,
                        help="Warmup checkpoint tag. Creates/loads warmup_checkpoint_{tag}.pth instead of warmup_checkpoint.pth. "
                             "Useful for maintaining separate warmup checkpoints for different configurations (e.g., --warmup beta).")

    # GaussianSpa ADMM sparsification
    parser.add_argument("--gspa", action="store_true",
                        help="Enable GaussianSpa ADMM-based sparsification")
    parser.add_argument("--gspa_rho", type=float, default=0.0005,
                        help="ADMM penalty weight (default: 0.0005)")
    parser.add_argument("--gspa_ratio", type=float, default=0.8,
                        help="ADMM phase: fraction of remaining Gaussians to prune (default: 0.8)")
    parser.add_argument("--gspa_start_iter", type=int, default=15200,
                        help="Iteration to start ADMM sparsification (default: 15200)")
    parser.add_argument("--gspa_stop_iter", type=int, default=25200,
                        help="Iteration to end ADMM and hard prune (default: 25200)")
    parser.add_argument("--gspa_interval", type=int, default=50,
                        help="z/u update frequency during ADMM phase (default: 50)")
    parser.add_argument("--gspa_simp_iter", type=int, default=15000,
                        help="Phase 1: importance-based pre-pruning iteration (default: 15000)")
    parser.add_argument("--gspa_prune_ratio1", type=float, default=0.5,
                        help="Phase 1: fraction of Gaussians to prune by importance (default: 0.5)")
    parser.add_argument("--gspa_imp_metric", type=str, default="indoor",
                        help="Importance metric: 'indoor' (sum weights) or 'outdoor' (weight/area)")
    parser.add_argument("--gspa_target_count", type=int, default=0,
                        help="Upper-limit target for final Gaussian count. When > 0, auto-splits "
                             "the keep ratio evenly across Phase 1 and Phase 2 "
                             "(keep_per_phase = sqrt(target/current)), overriding "
                             "--gspa_prune_ratio1 and --gspa_ratio. 0 = use manual ratios.")

    # MiniSpa: mini v2 aggressive clone + silhouette-aware depth reinit → GSpa ADMM
    parser.add_argument("--minispa", action="store_true",
                        help="MiniSpa: mini v2 aggressive cloning up to --minispa_reinit_iter, one "
                             "silhouette-aware depth reinit at that iter, then GSpa ADMM from "
                             "--minispa_admm_start to --minispa_admm_stop. Skips mini simp1/simp2 "
                             "and GSpa Phase 1 importance-prune. Requires --method 3D_SH_res.")
    parser.add_argument("--minispa_reinit_iter", type=int, default=2000,
                        help="MiniSpa: depth reinit iteration (loose silhouette scale). Default 2000.")
    parser.add_argument("--minispa_admm_start", type=int, default=3000,
                        help="MiniSpa: GSpa ADMM start iter (after reinit settles). Default 3000.")
    parser.add_argument("--minispa_admm_stop", type=int, default=23000,
                        help="MiniSpa: GSpa ADMM stop iter (final hard prune). Default 23000.")
    parser.add_argument("--minispa_reinit_interval", type=int, default=5000,
                        help="MiniSpa: during ADMM phase, fire a fresh depth reinit every N iters. "
                             "Skips the iter coinciding with --minispa_admm_stop (no end-of-ADMM reinit). "
                             "Default 5000. Set 0 to disable periodic reinits.")
    parser.add_argument("--minispa_mesh", action="store_true",
                        help="MiniSpa: TSDF-fuse the max-contributor depth maps into a mesh and "
                             "area-uniformly sample reinit points from it. Removes the view-coverage "
                             "bias of per-pixel reinit. Requires Open3D.")
    parser.add_argument("--minispa_mesh_voxel", type=float, default=0.0,
                        help="MiniSpa mesh reinit: TSDF voxel size in world units. "
                             "0 = scene.cameras_extent / 256 (default).")
    parser.add_argument("--minispa_mesh_poisson", action="store_true",
                        help="MiniSpa mesh reinit: use Poisson-disk (blue-noise) sampling instead "
                             "of uniform area-weighted. Slower (~2x) but more even spacing.")
    parser.add_argument("--minispa_mesh_scale_factor", type=float, default=0.5,
                        help="MiniSpa mesh reinit: multiply per-point NN-distance by this factor "
                             "when setting the initial surfel scale. 1.0 = neighbors overlap at 3σ "
                             "(bloated). 0.5 = neighbors at 2σ (gap-free tiling). Default 0.5.")
    parser.add_argument("--minispa_mesh_opacity", type=float, default=0.3,
                        help="MiniSpa mesh reinit: initial opacity for reinit surfels. "
                             "Default 0.3 (vs 0.8 for pixel reinit) so overlapping surfels don't "
                             "saturate alpha and read as bloat.")

    # FastGS multi-view consistency densification / pruning (Phase 1: Python proxy).
    # Paper: arXiv 2511.04283. Phase-1 uses max_contrib_idx as a per-pixel
    # contributor proxy (undercounts vs FastGS's all-contributor CUDA counter).
    parser.add_argument("--fastgs", action="store_true",
                        help="Enable FastGS multi-view consistent densification + pruning. "
                             "Replaces standard densify_and_prune; requires --method 3D_SH_res.")
    parser.add_argument("--fastgs_num_views", type=int, default=10,
                        help="FastGS: K random views sampled per densify/prune call. Paper: 10.")
    parser.add_argument("--fastgs_loss_thresh", type=float, default=0.1,
                        help="FastGS: per-view normalized L1 threshold flagging high-error pixels.")
    parser.add_argument("--fastgs_lambda_dssim", type=float, default=0.2,
                        help="FastGS: SSIM weight in photometric loss (paper λ=0.2).")
    parser.add_argument("--fastgs_importance_thresh", type=int, default=1,
                        help="FastGS: Gaussians with floor(sum_count / K) > this are eligible "
                             "for densification. Paper uses 5 with the all-contributor counter; "
                             "our max-contributor proxy undercounts so default is 1.")
    parser.add_argument("--fastgs_grad_thresh", type=float, default=0.0002,
                        help="FastGS: gradient threshold for clone qualification.")
    parser.add_argument("--fastgs_grad_abs_thresh", type=float, default=0.0012,
                        help="FastGS: absolute gradient threshold for split qualification (AbsGS).")
    parser.add_argument("--fastgs_dense", type=float, default=0.001,
                        help="FastGS: scale/extent threshold splitting clone vs split region.")
    parser.add_argument("--fastgs_prune_budget_frac", type=float, default=0.5,
                        help="FastGS: fraction of opacity-pruneable Gaussians actually removed per call.")
    parser.add_argument("--fastgs_opacity_clamp", type=float, default=0.8,
                        help="FastGS: cap every Gaussian's opacity at this value at the end of each "
                             "densify step (reference FastGS gaussian_model.py:520, paper 0.8). This is "
                             "the primary count-control lever during densification — prevents opacity "
                             "saturating toward 1.0 so marginal Gaussians stay in the pruneable band for "
                             "VCP. 0 disables (pre-fix behavior: count balloons until the >15k prune).")
    # --blur_split (mini-splatting2 "blur split"): per-pixel max-contributor
    # bincount → flag Gaussians dominating > H*W/blur_thresh pixels in any
    # iteration since the last densify, force-split them next densify event.
    # Targets oversized billboards in outdoor scenes (boost LPIPS for foliage,
    # large surfaces, sky proxies). Reset to zeros after each densify.
    parser.add_argument("--blur_split", action="store_true",
                        help="Enable mini-splatting2-style blur-split: force-split "
                             "Gaussians dominating > H*W/blur_thresh pixels.")
    parser.add_argument("--blur_thresh", type=float, default=5000.0,
                        help="Blur-split threshold: a Gaussian dominating more "
                             "than image_area/blur_thresh pixels is flagged for "
                             "splitting at the next densify. Default 5000 = ~1/5000 "
                             "of pixels (mini-splatting2 setting). HIGHER = smaller "
                             "pixel budget = more aggressive splitting.")
    parser.add_argument("--lpips_w", type=float, default=0.0,
                        help="Weight of an LPIPS perceptual term added to the "
                             "photometric loss (0 = off). Applies to the final "
                             "rendered image in every method incl. both GEStex "
                             "phases. Typical 0.1-0.5.")
    parser.add_argument("--lpips_net", choices=["alex", "vgg"], default="vgg",
                        help="LPIPS backbone for --lpips_w. vgg matches the eval "
                             "metric; alex is ~3x cheaper per iter.")
    parser.add_argument("--lpips_start_iter", type=int, default=0,
                        help="Iteration to enable the LPIPS loss from (early "
                             "geometry churn does not benefit from it; e.g. 15000 "
                             "to apply it only from GEStex hardening onward).")
    parser.add_argument("--fastgs_densify_interval", type=int, default=100,
                        help="FastGS: run VCD+VCP every N iters (paper: 500; default here 100).")
    parser.add_argument("--fastgs_densify_until", type=int, default=15000,
                        help="FastGS: stop densification at this iter (paper: 15000).")
    parser.add_argument("--fastgs_final_prune_interval", type=int, default=3000,
                        help="FastGS: post-15k aggressive pruning cadence (paper: 3000).")
    parser.add_argument("--fastgs_final_prune_until", type=int, default=30000,
                        help="FastGS: stop the post-15k aggressive pruning at this iter.")
    parser.add_argument("--fastgs_final_min_opacity", type=float, default=0.1,
                        help="FastGS: post-15k opacity floor for aggressive prune (paper: 0.1).")
    parser.add_argument("--fastgs_final_score_thresh", type=float, default=0.9,
                        help="FastGS: post-15k pruning score threshold (paper: 0.9).")
    parser.add_argument("--fastgs_mult", type=float, default=1.0,
                        help="FastGS Compact Box Mahalanobis² scale factor for Gaussian "
                             "kernels. Tightens per-Gaussian tile AABB to the ellipse "
                             "contour where opacity·exp(-0.5·maha²·mult) = 1/255. "
                             "DEFAULT 1.0 = OFF (existing AdR cutoff, no change). 0.5 = "
                             "FastGS paper setting (tighter, faster). Applies under --fastgs "
                             "OR standalone (any 3D_SH_res-family method); auto-enables a "
                             "compatible --aabb. KERNEL-DISPATCHED: beta kernels scale the "
                             "use_beta_cutoff radius (~3.3σ baseline, support ends ~0.9), "
                             "gaussian kernels scale the use_adr_cutoff Mahalanobis² — one "
                             "knob for both. Trains the primitives to adapt to the tight box.")

    # Mini-Splatting v2 optimization
    parser.add_argument("--mini", action="store_true",
                        help="Enable Mini-Splatting v2 optimization (contribution-based densification, depth reinit, importance pruning)")
    parser.add_argument("--mini_depth_reinit_iter", type=int, default=2000,
                        help="Iteration for depth reinitialization (default: 2000)")
    parser.add_argument("--mini_simp_iter1", type=int, default=3000,
                        help="First importance-based simplification iteration (default: 3000)")
    parser.add_argument("--mini_simp_iter2", type=int, default=8000,
                        help="Second importance-based simplification iteration (default: 8000)")
    parser.add_argument("--mini_densify_until", type=int, default=3000,
                        help="End densification for MSv2 (default: 10000)")
    parser.add_argument("--mini_clone_interval", type=int, default=250,
                        help="Aggressive clone interval (default: 250)")
    parser.add_argument("--mini_depth_factor", type=float, default=1.0,
                        help="Multiplier for number of depth-reinitialized points (default: 1.0)")
    parser.add_argument("--mini_sampling_factor", type=float, default=0.6,
                        help="Simp1: importance-weighted sampling factor (default: 0.6, keep ~60%% of non-zero importance)")
    parser.add_argument("--mini_imp_metric", type=str, default="indoor", choices=["indoor", "outdoor"],
                        help="Importance metric: 'indoor' (sum weights) or 'outdoor' (weight/area)")
    parser.add_argument("--mini_late_prune_interval", type=int, default=500,
                        help="After mini_simp_iter2, prune Gaussians below opacity threshold every N iters (0=disabled)")
    parser.add_argument("--mini_late_prune_thresh", type=float, default=0.005,
                        help="Opacity threshold for late prune under --mini (default: 0.005)")
    parser.add_argument("--mini_warmup", action="store_true",
                        help="Camera warmup: train at 0.5x resolution during densification phase (until simp1)")

    # Mini-Splatting v1: repeated depth reinit + longer densification, NO aggressive clone
    parser.add_argument("--mini1", action="store_true",
                        help="Enable Mini-Splatting v1 schedule: repeated depth reinit on interval, longer densification window, no aggressive clone, later simplifications")
    parser.add_argument("--mini1_depth_reinit_interval", type=int, default=5000,
                        help="Interval (iters) between repeated depth reinits under --mini1 (default: 5000)")
    parser.add_argument("--mini1_depth_reinit_until", type=int, default=15000,
                        help="Stop repeated depth reinits at this iter under --mini1 (default: 15000)")

    # MiniMC: Zero-Waste RJ-MCMC pipeline (error-driven relocation + pixel-ownership cull)
    # Closed-loop: Mini cull marks non-winners dead → MCMC relocates dead onto high-error alives.
    parser.add_argument("--minimc", action="store_true",
                        help="Enable Zero-Waste RJ-MCMC: closed-loop error-driven relocation + pixel-ownership culling. Requires --mcmc (or --mcmc_fps/--mcmc_deficit) for SGLD noise.")
    parser.add_argument("--minimc_start_iter", type=int, default=500,
                        help="--minimc: do not cull/relocate before this iter (warmup). Default: 500")
    parser.add_argument("--minimc_relocate_interval", type=int, default=100,
                        help="--minimc: Phase-3 error-driven relocate fires every N iters. Default: 100")
    parser.add_argument("--minimc_reinit_interval", type=int, default=0,
                        help="--minimc: depth reinit fires every N iters (uses max-contributor depth from rasterizer, same path as --mini1). 0 = disabled. Default: 0")
    parser.add_argument("--minimc_reinit_until", type=int, default=20000,
                        help="--minimc: stop running depth reinit after this iter. Default: 20000")
    # Deprecated aliases kept for back-compat. If set, they map into the reinit_interval flags.
    parser.add_argument("--minimc_cull_interval", type=int, default=-1,
                        help="(deprecated) Old name for --minimc_reinit_interval. Will be silently remapped.")
    parser.add_argument("--minimc_cull_until", type=int, default=-1,
                        help="(deprecated) Old name for --minimc_reinit_until. Will be silently remapped.")
    parser.add_argument("--minimc_relocate_until", type=int, default=25000,
                        help="--minimc: stop running relocate after this iter (pure SGLD drift after). Default: 25000")
    parser.add_argument("--minimc_max_per_donor", type=int, default=1,
                        help="--minimc: strict per-donor clone cap in a single relocate call. Default: 1 (single foggy probe per donor)")
    parser.add_argument("--minimc_dead_thresh", type=float, default=0.005,
                        help="--minimc: opacity <= this is considered dead. Same as vanilla MCMC. Default: 0.005")
    parser.add_argument("--minimc_no_single_view_cull", action="store_true",
                        help="--minimc: disable culling of single-view Gaussians (count_vis<=1). Default: cull.")
    parser.add_argument("--minimc_low_imp_cdf", type=float, default=0.999,
                        help="--minimc: CDF threshold for low-importance cull. Keeps top X by cumulative importance, drops the rest. Set to 1.0 to disable. Default: 0.999")
    parser.add_argument("--minimc_growth_frac", type=float, default=0.05,
                        help="--minimc: fraction of total tensor cloned into dead slots per sweep (matches vanilla MCMC's 5% add_new_gs growth rate). Actual n_clones = min(growth_frac * N_total, n_candidates, n_dead). Default: 0.05")
    # Deprecated (old 3-stage minimc). Kept to avoid loud breakage if yaml configs still reference them.
    parser.add_argument("--minimc_reinit_iter", type=int, default=0,
                        help="(deprecated, ignored by new --minimc pipeline)")
    parser.add_argument("--minimc_simp_iter1", type=int, default=0,
                        help="(deprecated, ignored by new --minimc pipeline)")
    parser.add_argument("--minimc_simp_iter2", type=int, default=0,
                        help="(deprecated, ignored by new --minimc pipeline)")
    parser.add_argument("--minimc_sampling_factor", type=float, default=0.6,
                        help="(deprecated, ignored by new --minimc pipeline)")

    # Morton z-order sorting for cache locality
    parser.add_argument("--morton_interval", type=int, default=0,
                        help="Apply Morton z-order sorting every N iterations (0 = disabled, recommended: 5000)")
    parser.add_argument("--morton_end_iter", type=int, default=15000,
                        help="Stop periodic Morton sorting after this iteration (default: 15000)")

    # Contribution threshold: skip hash query when effective contribution w = T*alpha is below threshold
    parser.add_argument("--contribution_thresh", type=float, default=0.0,
                        help="Skip hash query when w = T*alpha < threshold, MLP runs with zero features (0.0 = disabled, try 0.01)")

    # Count threshold: skip hash after N contributing Gaussians per pixel
    parser.add_argument("--count_thresh", type=int, default=0,
                        help="Skip hash query after N contributing Gaussians per pixel (0 = disabled)")

    # Opacity threshold: skip hash query when alpha = opa*kernel_val (per-pixel opacity
    # contribution) is below threshold — drops the residual on the soft tails of fuzzy surfels.
    # Occlusion-independent, unlike --contribution_thresh (which uses w = T*alpha). 3D_SH_res only.
    parser.add_argument("--opacity_thresh", type=float, default=0.0,
                        help="Skip hash query when alpha = opa*kernel_val < threshold (0.0 = disabled, try 0.01-0.05). 3D_SH_res only.")

    # Texture-query dropout (training regularization): randomly drop the hash/MLP residual query
    # for a fraction of Gaussians each iteration (per-Gauss, unscaled, off at inference). Bare
    # `--texture_dropout` = 25%; `--texture_dropout 0.4` = 40%. 3D_SH_res only.
    parser.add_argument("--texture_dropout_bw", type=float, nargs='?', const=0.25, default=0.0,
                        help="BACKWARD-ONLY texture dropout: forward renders the full image; the "
                             "dropped Gaussians only skip their hash/MLP GRADIENTS (armed between "
                             "forward and backward; per-Gauss set rotates per iter). Gradient "
                             "sparsification without image perturbation. Mutually exclusive with "
                             "--texture_dropout. 3D_SH_res only. Bare flag = 0.25.")
    parser.add_argument("--texture_dropout", type=float, nargs='?', const=0.25, default=0.0,
                        help="Drop fraction of texture (hash/MLP) queries per iter during training, per-Gauss "
                             "(0.0 = disabled; bare flag = 0.25). SH base learns to render dropped surfels alone. 3D_SH_res only.")

    # Overdraw regularization: penalize per-pixel contributor count
    parser.add_argument("--overdraw_reg", type=float, default=0.0,
                        help="Overdraw regularization lambda. Penalizes soft per-pixel contributor count (0 = disabled)")
    parser.add_argument("--w_overdraw_reg", type=float, default=0.0,
                        help="Weighted overdraw regularization lambda. Error-guided: relaxes penalty where RGB error is high (0 = disabled)")
    parser.add_argument("--w_overdraw_gamma", type=float, default=50.0,
                        help="Gamma for weighted overdraw: w(r) = exp(-gamma * MSE(r)). Higher = more aggressive relaxation.")
    parser.add_argument("--weight_reg", type=float, default=0.0,
                        help="Weight-squared regularization: penalize (1 - sum(w_i^2)) per pixel. Consolidates to single opaque surface (0 = disabled)")
    parser.add_argument("--w_weight_reg", type=float, default=0.0,
                        help="Error-guided weight-squared regularization. Relaxes where RGB error is high (0 = disabled)")
    parser.add_argument("--w_weight_gamma", type=float, default=50.0,
                        help="Gamma for weighted weight_reg: w(r) = exp(-gamma * MSE(r))")
    parser.add_argument("--w_normal", type=float, default=0.0,
                        help="Weighted normal consistency lambda. Error-guided: relaxes where RGB error is high (0 = use lambda_normal instead)")
    parser.add_argument("--w_normal_gamma", type=float, default=50.0,
                        help="Gamma for weighted normal: w(r) = exp(-gamma * MSE(r)). Higher = more aggressive relaxation.")
    parser.add_argument("--w_lambda", type=float, default=0.0,
                        help="Weighted shape regularization lambda (error-guided lambda_shape). "
                             "Effective penalty = lambda · exp(-gamma · mean_MSE) · shape_direction. "
                             "Pushes β toward 0 (flat disk) for beta/beta_scaled, toward 8 (flat box) "
                             "for general. Relaxes where photometric error is high so detail regions "
                             "can keep their soft kernels. 0 = disabled (fall back to static lambda_shape).")
    parser.add_argument("--w_lambda_gamma", type=float, default=50.0,
                        help="Gamma for weighted shape: w = exp(-gamma · MSE). Higher = sharper relaxation.")
    parser.add_argument("--w_lambda_perpix", type=float, default=0.0,
                        help="Per-pixel error-guided shape reg. Penalizes sum_i (w_i * beta_i) at each "
                             "pixel, scaled by per-pixel exp(-gamma * MSE). Unlike --w_lambda (which "
                             "collapses the weight to a scalar), this attributes shape pressure to the "
                             "Gaussians actually contributing to each pixel. CUDA direct grad: only "
                             "dampens beta values, does NOT propagate through alpha. 0 = disabled.")
    parser.add_argument("--w_lambda_perpix_gamma", type=float, default=50.0,
                        help="Gamma for --w_lambda_perpix: w_r = exp(-gamma * MSE(pix)).")

    # Separated depth sort: pre-sort Gaussians by depth, then sort expanded list by tile_id only
    parser.add_argument("--depth_sort", action="store_true",
                        help="Use separated depth sort (pre-sort by depth, then tile-bin). Default: standard full radix sort")

    # Structure-texture decomposition training (--decomp).
    # Pre-computes a guided-filter low-frequency image per train view (cached to
    # {dataset}/decomp_r{R}_eps{EPS}/) and supervises sh_only vs low-freq,
    # tex_only vs high-freq residual, on top of the main photometric loss.
    # Only works with --method 3D_SH_res (the only rasterizer exposing the
    # sh_only/tex_only decompose_mode path).
    parser.add_argument("--decomp", action="store_true",
                        help="Enable structure-texture decomposition training. Requires --method 3D_SH_res.")
    parser.add_argument("--decomp_r", type=int, default=8,
                        help="Guided-filter radius. Larger = more aggressive structure extraction. Default 8.")
    parser.add_argument("--decomp_eps", type=float, default=0.01,
                        help="Guided-filter ε² edge-preservation. Smaller = sharper edges, less detail in residual. Default 0.01.")
    parser.add_argument("--decomp_lambda_sh", type=float, default=1.0,
                        help="Weight on L1(sh_only, gt_low) auxiliary loss. Default 1.0.")
    parser.add_argument("--decomp_lambda_tex", type=float, default=1.0,
                        help="Weight on L1(tex_only, gt_high) auxiliary loss. Default 1.0.")
    parser.add_argument("--decomp_interval", type=int, default=1,
                        help="Apply --decomp losses every N iters. 1 = every step, higher amortizes the 2 extra renders.")
    parser.add_argument("--decomp_comb", action="store_true",
                        help="Change the --decomp structure target from sh_only to the FULL render: "
                             "supervise L1(full, gt_low) instead of L1(sh_only, gt_low). SH and hashgrid "
                             "jointly fit the blur; hashgrid is additionally pulled toward the residual via "
                             "L1(tex_only, gt_high). Saves one render per iter (no sh_only pass). "
                             "Equilibrium (with main L(full, gt) still active): SH ≈ gt_low, tex_only ≈ gt_high.")
    # --blurprog: curriculum alternative to --decomp. No extra renders — just
    # interpolate the target image from gt_low (structure) toward gt (full) over
    # the first N iters. Whole model (SH + hashgrid) fits increasingly sharp images.
    parser.add_argument("--blurprog", action="store_true",
                        help="Progressive-blur curriculum: start training against gt_low and linearly "
                             "ramp to full gt by --blurprog_until. Uses --decomp_r / --decomp_eps for "
                             "the guided-filter cache (pre-computed once, shared with --decomp). "
                             "Same iter cost as normal training.")
    parser.add_argument("--blurprog_until", type=int, default=10000,
                        help="--blurprog: iter at which the target is fully sharp (α=1, pure gt). Default 10000.")
    parser.add_argument("--blurprog_schedule", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="--blurprog: α(t) interpolation shape from 0 (blur) to 1 (sharp). Default 'linear'.")

    # --merge: mode-agnostic geometric primitive consolidation. Every --merge_interval
    # iters (up to --merge_until), pair primitives within τ_dist via a scipy KDTree,
    # run a 4-gate test (normal alignment, coplanarity, spatial overlap, feature
    # similarity), and merge pairs that pass. Designed to stop SH/residual
    # overfitting by collapsing onion-skin layers into a single surface.
    parser.add_argument("--merge", action="store_true",
                        help="Enable geometric primitive consolidation (4-gate merge). Runs "
                             "before opacity reset at each --merge_interval up to --merge_until.")
    parser.add_argument("--merge_interval", type=int, default=3000,
                        help="--merge: run consolidation every N iters. Default 3000 (lines up with "
                             "opacity_reset_interval).")
    parser.add_argument("--merge_until", type=int, default=0,
                        help="--merge: stop running consolidation after this iter. "
                             "0 = auto = densify_until_iter − merge_interval (last merge lands "
                             "one cycle before densification ends).")
    parser.add_argument("--merge_tau_dist", type=float, default=0.0,
                        help="--merge: spatial-overlap radius (Gate 3). 0 = auto = "
                             "0.5 × median(max_scale).")
    parser.add_argument("--merge_tau_normal", type=float, default=0.95,
                        help="--merge: min n_i · n_j for alignment (Gate 1). Default 0.95 "
                             "(~18° cone).")
    parser.add_argument("--merge_tau_planar", type=float, default=0.0,
                        help="--merge: coplanarity threshold (Gate 2). 0 = auto = 0.5 × τ_dist.")
    parser.add_argument("--merge_tau_feat", type=float, default=0.01,
                        help="--merge: max MSE(f_dc_i, f_dc_j) for feature-similarity (Gate 4). "
                             "Default 0.01 (SH DC is small-magnitude; tune per-scene).")

    args = parser.parse_args(sys.argv[1:])

    # Track which args were explicitly set on CLI (so yaml doesn't override them)
    cli_args = set()
    for action in parser._actions:
        for opt in action.option_strings:
            key = opt.lstrip('-').replace('-', '_')
            if key in sys.argv[1:] or opt in sys.argv[1:]:
                cli_args.add(action.dest)
                break
    # Also check positional-style: --iterations 10100 → "iterations" is action.dest
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            dest = arg.lstrip('-').replace('-', '_')
            cli_args.add(dest)

    # --bce_solo / --bce_solo_adaptive / --bce_adaptive implies --bce
    if args.bce_solo or args.bce_solo_adaptive or args.bce_adaptive:
        args.bce = True
        if args.bce_solo_adaptive:
            args.bce_solo = True  # also disable opacity/scale reg during BCE phase

    # MCMC defaults: enable opacity/scale regularization if not explicitly set
    if (args.mcmc or args.mcmc_deficit or args.mcmc_fps):
        if 'opacity_reg' not in cli_args and args.opacity_reg == 0.0:
            args.opacity_reg = 0.01
        if 'scale_reg' not in cli_args and args.scale_reg == 0.0:
            args.scale_reg = 0.01

    # NOTE: --feature SV used to apply a 2dgs-voronoi-style override block here
    # (sb_number=7, sites_lr=2e-3, sv_color_lr=8e-4, sv_tau_lr=6e-3, sv_l1=0,
    # sv_warmup_iter=1000, sv_freeze_last=3000). Those values are now the
    # argparse defaults, so the override block is redundant and was removed.

    # --feature beta: --sb_number default is 7 (matches voronoi/SV) but the
    # beta-splatting paper uses K=2. Special-case: drop to K=2 unless the user
    # explicitly set --sb_number on the CLI.
    if args.feature == "beta" and 'sb_number' not in cli_args:
        args.sb_number = 2
        print(f"[FEATURE beta] Using sb_number=2 (beta-splatting paper). "
              f"Pass --sb_number 7 to match the voronoi/SV K count.")

    # Mini-Splatting v1 (--mini1) is a variant of --mini with a v1-style schedule.
    # It reuses all --mini machinery but enables repeated depth reinit, extends
    # densification, and skips aggressive cloning. Enable --mini automatically.
    if args.mini1:
        args.mini = True
        # Override v2 defaults with v1-style schedule unless user set them explicitly.
        if 'mini_depth_reinit_iter' not in cli_args:
            args.mini_depth_reinit_iter = 5000
        if 'mini_simp_iter1' not in cli_args:
            args.mini_simp_iter1 = 15000
        if 'mini_simp_iter2' not in cli_args:
            args.mini_simp_iter2 = 20000
        if 'mini_densify_until' not in cli_args:
            args.mini_densify_until = 15000
        if 'mini_sampling_factor' not in cli_args:
            args.mini_sampling_factor = 0.5

    # MiniSpa: mini v2 aggressive clone + silhouette reinit → GSpa ADMM.
    # Auto-enables --mini and --gspa and wires the schedule so the two phases
    # chain cleanly (reinit at minispa_reinit_iter, ADMM start after settle,
    # no mini simp1/simp2, no gspa Phase 1).
    # --minispa_mesh implies --minispa (which then implies --mini + --gspa).
    if args.minispa_mesh and not args.minispa:
        args.minispa = True
        print("[MINISPA-MESH] auto-enabled --minispa (mesh reinit requires the minispa pipeline)")

    if args.minispa:
        args.mini = True
        args.gspa = True
        # No depth reinit under minispa — aggressive clone runs straight through
        # until ADMM starts, then ADMM takes over. Park the mini reinit iter past
        # the ADMM start so it never triggers, and keep aggressive clone alive
        # until ADMM begins via mini_simp_iter1.
        if 'mini_depth_reinit_iter' not in cli_args:
            args.mini_depth_reinit_iter = 10**9
        if 'mini_densify_until' not in cli_args:
            args.mini_densify_until = args.minispa_admm_start
        # Disable aggressive cloning: clone window is [500, mini_simp_iter1).
        # Setting mini_simp_iter1 < 500 collapses the window — standard 3DGS
        # densify_and_prune handles growth instead.
        if 'mini_simp_iter1' not in cli_args:
            args.mini_simp_iter1 = 0
        if 'mini_simp_iter2' not in cli_args:
            args.mini_simp_iter2 = 10**9
        if 'gspa_start_iter' not in cli_args:
            args.gspa_start_iter = args.minispa_admm_start
        if 'gspa_stop_iter' not in cli_args:
            args.gspa_stop_iter = args.minispa_admm_stop
        # gspa_simp_iter anchors SH-degree increases. Park at ADMM start so SH
        # begins ramping up when ADMM kicks in. Phase 1 dispatch is gated off.
        if 'gspa_simp_iter' not in cli_args:
            args.gspa_simp_iter = args.minispa_admm_start

    # Mini-Splatting v2: mutual exclusivity check
    if args.mini:
        if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
            raise ValueError("--mini is mutually exclusive with --mcmc/--mcmc_deficit/--mcmc_fps")
        if args.gspa and not args.minispa:
            raise ValueError("--mini is mutually exclusive with --gspa (use --minispa to combine)")

    # --minimc back-compat: old --minimc_cull_interval / --minimc_cull_until flags
    # now map into --minimc_reinit_interval / --minimc_reinit_until.
    if args.minimc_cull_interval != -1:
        print(f"[MINIMC] --minimc_cull_interval is deprecated; remapping to --minimc_reinit_interval={args.minimc_cull_interval}")
        args.minimc_reinit_interval = args.minimc_cull_interval
    if args.minimc_cull_until != -1:
        print(f"[MINIMC] --minimc_cull_until is deprecated; remapping to --minimc_reinit_until={args.minimc_cull_until}")
        args.minimc_reinit_until = args.minimc_cull_until

    # --minimc requires one of the MCMC flags (for SGLD noise + dead-pool semantics).
    # Auto-enable --mcmc to save the user the typing; hard-error if they actively
    # pass --mini / --gspa alongside it.
    if args.minimc:
        if args.mini or args.mini1:
            raise ValueError("--minimc is mutually exclusive with --mini/--mini1")
        if args.gspa:
            raise ValueError("--minimc is mutually exclusive with --gspa")
        if not (args.mcmc or args.mcmc_deficit or args.mcmc_fps):
            args.mcmc = True
            print("[MINIMC] Auto-enabling --mcmc (required for SGLD noise + dead-pool semantics).")
        if args.cap_max is None or args.cap_max <= 0:
            raise ValueError("--minimc requires --cap_max > 0 (same as --mcmc).")

    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")
    if args.method == "cat":
        print(f"Hybrid levels: {args.hybrid_levels} (per-Gaussian features for coarse levels)")
        if args.disable_c2f:
            print(f"C2F disabled: all levels active from start")
    elif args.method == "film":
        print(f"FiLM mode: per-Gauss gamma*H(x) + beta (gamma init 1.0, beta init 0.0); MLP decode in PyTorch")
    elif args.method == "cat_dropout":
        print(f"Cat Dropout mode: cat mode with hash dropout during training")
        print(f"  - Hybrid levels: {args.hybrid_levels} (per-Gaussian features for coarse levels)")
        print(f"  - Dropout lambda: {args.dropout_lambda} ({args.dropout_lambda*100:.0f}% of Gaussians don't query hash)")
        print(f"  - Inference: identical to cat mode (no dropout)")
    elif args.method == "adaptive_cat":
        print(f"Adaptive Cat mode: learnable binary blend weights (smooth training, binary inference)")
        print(f"  - Entropy regularization: {args.lambda_adaptive_cat} (anneals from iter {args.adaptive_cat_anneal_start})")
        print(f"  - Inference mode: {'BINARY (skip intersection or skip Gaussian)' if args.adaptive_cat_inference else 'SMOOTH (blending)'}")
        print(f"  - Single-level hashgrid at finest resolution")
    elif args.method == "diffuse":
        print(f"Diffuse mode: SH degree 0, no viewdir, no hashgrid")
    elif args.method == "specular":
        print(f"Specular mode: full 2DGS with SH (view-dependent), no hashgrid")
    elif args.method == "diffuse_ngp":
        print(f"Diffuse+NGP mode: diffuse SH + hashgrid on unprojected expected depth")
    elif args.method == "diffuse_offset":
        print(f"Diffuse+Offset mode: diffuse SH as xyz offset, hashgrid MLP for final RGB")
    elif args.method == "adaptive_add":
        print(f"Adaptive Add mode: weighted sum of per-Gaussian features and hashgrid features")

    if args.bce:
        mode = "adaptive (median threshold)" if args.bce_solo_adaptive else ("adaptive + reg" if args.bce_adaptive else ("solo" if args.bce_solo else "standard"))
        print(f"BCE opacity regularization: enabled for last {args.bce_iter} iterations (lambda={args.bce_lambda}, mode={mode})")

    # Always print kernel info
    print(f"Kernel type: {args.kernel.upper()}")
    if args.kernel == "beta":
        print(f"  Beta kernel: learnable per-Gaussian shape parameter")
        print(f"  - Formula: alpha = opacity * pow(1 - r², shape), r ∈ [0, 1]")
        print(f"  - Shape range: [0.001, 4.001] (0=hard disk, 4=soft cloud)")
        shape_iter_str = f"last {args.shape_iter} iterations" if args.shape_iter > 0 else "always active"
        print(f"  - Shape regularization: lambda={args.lambda_shape} ({shape_iter_str})")
    elif args.kernel == "beta_scaled":
        print(f"  Beta Scaled kernel: learnable per-Gaussian shape parameter (scaled radius)")
        print(f"  - Formula: alpha = opacity * pow(1 - (r/3)², shape), r ∈ [0, 3]")
        print(f"  - Radius scaled by 3 to match 3σ Gaussian extent")
        print(f"  - Shape range: [0.001, 4.001] (0=hard disk, 4=soft cloud)")
        shape_iter_str = f"last {args.shape_iter} iterations" if args.shape_iter > 0 else "always active"
        print(f"  - Shape regularization: lambda={args.lambda_shape} ({shape_iter_str})")
    elif args.kernel == "flex":
        print(f"  Flex kernel: Gaussian with learnable per-Gaussian beta (sharpening)")
        print(f"  - Formula: G = exp(power); alpha = (1+beta)*G / (1+beta*G)")
        print(f"  - Beta range: [0, inf) via softplus (0=standard Gaussian, higher=sharper)")
        print(f"  - Beta regularization: lambda={args.lambda_flex_beta} (L1 penalty prevents runaway)")
    elif args.kernel == "general":
        print(f"  General kernel: Isotropic Generalized Gaussian")
        print(f"  - Formula: alpha = opacity * exp(-0.5 * (r²)^(β/2))")
        print(f"  - Beta range: [2.0, 8.0] via sigmoid*6+2 (2=Gaussian, 8=super-Gaussian/box)")
        shape_iter_str = f"last {args.shape_iter} iterations" if args.shape_iter > 0 else "always"
        print(f"  - Beta regularization: lambda={args.lambda_shape}, mode={args.genreg}, active={shape_iter_str}")
        if args.genreg == "basic":
            print(f"    (constant regularization throughout training)")
        elif args.genreg == "decay":
            print(f"    (linear decay from lambda to 0 over training)")
        elif args.genreg == "scaled":
            print(f"    (scaled by RGB loss to stay proportional)")
        elif args.genreg == "scaled_decay":
            print(f"    (scaled by RGB loss + linear decay)")
    else:
        print(f"  Gaussian kernel: alpha = opacity * exp(-0.5 * r²)")
        print(f"  (Use --kernel beta for learnable beta kernel, --kernel flex for Gaussian with learnable sharpening)")

    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model, cli_args=cli_args)

    # --reset_until_iter: CLI override for training_cfg.reset_until_iter (the
    # YAML-only gate on periodic opacity resets). E.g. `--reset_until_iter
    # 15000` keeps the 3k-interval resets through 15k and stops them after —
    # used with --trunc so no reset lands after the truncation step. -1 = keep
    # the YAML value.
    if getattr(args, 'reset_until_iter', -1) >= 0:
        cfg_model.training_cfg.reset_until_iter = int(args.reset_until_iter)
        print(f"[CFG OVERRIDE] training_cfg.reset_until_iter = {args.reset_until_iter} "
              f"(opacity resets stop after this iteration)")

    # --hash_levels: 3D_SH_res-friendly knob — K = number of HASH levels (vs --hybrid_levels
    # which is K = number of per-Gauss feature levels, leftover from CAT mode where
    # hybrid_levels + hash_levels = total_levels). For 3D_SH_res the per-surfel feature
    # is SH (no per-Gauss hash split), so users naturally think in terms of "how many
    # hash levels". Translate hash_levels → hybrid_levels for the rest of the pipeline.
    if args.hash_levels >= 0:
        if args.method not in ("3D_SH_res", "3D_SH_res_sep", "res_switch", "res_3d", "res_3d_paired", "3D_SH_cat"):
            raise ValueError(
                f"--hash_levels is only supported with --method 3D_SH_res or 3D_SH_cat (got --method {args.method}). "
                f"For other methods use --hybrid_levels."
            )
        if 'hybrid_levels' in cli_args:
            raise ValueError(
                "Cannot pass both --hash_levels and --hybrid_levels (ambiguous). Pick one."
            )
        _total_levels = cfg_model.encoding.levels
        _level_dim = cfg_model.encoding.hashgrid.dim
        # Per-mode upper bound on hash_levels — driven by what fits the 16D MLP input:
        #   3D_SH_res: [hash | pad] = 16D            → max_hash = 16 // dim
        #   3D_SH_cat: [hash | DC(3) | bias(1) | pad] = 16D → max_hash = (16-3-1) // dim = 3 for dim=4
        if args.method == "3D_SH_cat":
            _max_hash = (16 - 3 - 1) // _level_dim
        else:
            _max_hash = 16 // _level_dim
        _max_hash = min(_max_hash, _total_levels)  # also can't exceed configured total levels
        if not (0 <= args.hash_levels <= _max_hash):
            raise ValueError(
                f"--hash_levels must be in [0, {_max_hash}] for --method {args.method} "
                f"(encoding.levels={_total_levels}, dim={_level_dim} in {args.yaml}); got {args.hash_levels}"
            )
        args.hybrid_levels = _total_levels - args.hash_levels
        print(f"[HASH_LEVELS] hash_levels={args.hash_levels} -> hybrid_levels={_total_levels}-{args.hash_levels}={args.hybrid_levels} (max for {args.method}: {_max_hash})")

    # Mini-Splatting v2: override densify_until_iter and opacity_lr AFTER yaml merge
    if args.mini and 'densify_until_iter' not in cli_args:
        args.densify_until_iter = args.mini_densify_until
    if args.mini and 'opacity_lr' not in cli_args:
        args.opacity_lr = 0.025  # MSv2 halves opacity LR for stability

    # --merge: auto-resolve merge_until to land one cycle before densification ends.
    # Picks the right "densify end" per mode (fastgs / minispa / traditional).
    if args.merge and args.merge_until <= 0:
        if args.fastgs:
            _dend = int(args.fastgs_densify_until)
        elif args.minispa:
            _dend = int(args.minispa_admm_start)  # densify stops when ADMM starts
        else:
            _dend = int(args.densify_until_iter)
        args.merge_until = max(args.merge_interval, _dend - args.merge_interval)
        print(f"[MERGE] Auto merge_until={args.merge_until} "
              f"(densify_end={_dend} − merge_interval={args.merge_interval})")

    # Cold start mode: override config to enable hash_in_CUDA from start
    if args.cold:
        if not cfg_model.settings.if_ingp:
            print("\n[WARNING] --cold flag requires if_ingp=True in config. This will likely fail.")
        print("\n[COLD START] Overriding config:")
        print(f"  ingp_stage.initialize: {cfg_model.ingp_stage.initialize} -> 0")
        print(f"  ingp_stage.switch_iter: {cfg_model.ingp_stage.switch_iter} -> 0")
        cfg_model.ingp_stage.initialize = 0
        cfg_model.ingp_stage.switch_iter = 0
        print("  hash_in_CUDA will be enabled from iteration 1")
        if cfg_model.settings.gs_alpha:
            print("  Note: gs_alpha masks will not be generated in cold mode")
        print()

    print("args: ", args)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Wrap training in try/except to catch GPU errors for SLURM retry
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, \
            args)
        print("\nTraining complete.")
    except Exception as e:
        if is_gpu_error(e):
            # Write to failure file so SLURM worker knows to retry
            write_gpu_failure(args.model_path, str(e))
        # Re-raise to exit with non-zero code
        raise
