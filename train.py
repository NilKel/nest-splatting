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
from utils.loss_utils import l1_loss, ssim
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

    # 3D_SH_add: same architecture as 3D_SH_res, only the outer activation differs.
    # Set the residual-mode flag now so `--decomp` validation accepts it. The
    # actual `args.method = "3D_SH_res"` alias happens AFTER
    # prepare_output_and_logger so the run lands in its own `3D_SH_add/` folder.
    args._residual_mode = 1 if args.method == "3D_SH_add" else 0

    # --decomp: only the diff_surfel_3D_sh_res rasterizer exposes the sh_only /
    # tex_only decompose_mode paths needed to split the supervision.
    if getattr(args, 'decomp', False) and args.method not in ("3D_SH_res", "3D_SH_add"):
        raise RuntimeError(
            f"--decomp requires --method 3D_SH_res or 3D_SH_add; got --method {args.method}. "
            f"Other rasterizers don't expose the sh_only/tex_only decompose path."
        )
    # --blurprog: uses the same gt_low cache as --decomp but only needs main-loop
    # render — no decompose_mode dependency. Gated to 3D_SH_res here only because
    # the cache plumbing (Scene.__init__) is tied to the same code path. Could be
    # relaxed if needed.
    if getattr(args, 'blurprog', False) and args.method not in ("3D_SH_res", "3D_SH_add"):
        raise RuntimeError(
            f"--blurprog requires --method 3D_SH_res or 3D_SH_add; got --method {args.method}."
        )

    # Pass mini flag to OptimizationParams so training_setup can pick SparseGaussianAdam
    opt.mini = getattr(args, 'mini', False)

    testing_iterations += [opt.iterations]
    testing_iterations += [1]  # Also evaluate at first iteration for debugging
    # Periodic eval every 5k from 25k onwards so you can pick a good stop point.
    # The final iter is already covered by line above.
    testing_iterations += list(range(25_000, opt.iterations, 5_000))
    saving_iterations += [opt.iterations]

    test_psnr = []
    train_psnr = []
    iter_list = []
    optimizing_spa = None

    scene_name = args.scene_name
    tb_writer = prepare_output_and_logger(dataset, scene_name, args.yaml, args)
    args.model_path = dataset.model_path

    # 3D_SH_add → alias to 3D_SH_res for the rest of training. We had to wait
    # until after prepare_output_and_logger so the run gets its own
    # `outputs/.../3D_SH_add/<run>` folder; downstream code only knows about
    # 3D_SH_res. The activation switch is plumbed via set_residual_mode(1)
    # further below (gated on args._residual_mode == 1).
    if args.method == "3D_SH_add":
        args.method = "3D_SH_res"
        print("[3D_SH_add] activation = ReLU(SH+sh_bias) + ReLU(residual+res_bias) (separate ReLUs)")

    # Pass method, hybrid_levels, and decompose_mode to dataset for use in Scene/GaussianModel
    dataset.method = args.method
    dataset.hybrid_levels = args.hybrid_levels if hasattr(args, 'hybrid_levels') else 3
    dataset.decompose_mode = args.decompose_mode if hasattr(args, 'decompose_mode') else None

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)

    # Set kernel type for beta kernel support
    gaussians.kernel_type = args.kernel
    # Set densification gradient mode (vanilla = signed, abs = AbsGS)
    gaussians.use_absgs = (args.grads == "abs")

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
            if hasattr(args, 'method') and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
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
                print(f"  [FPS] Skipping FPS - using all {n_loaded} loaded Gaussians as cap_max")

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

        elif args.method == "3D_SH_32":
            # 3D_SH_32: per-Gaussian SH + 32-dim hash MLP residual
            # Same as 3D_SH_res but with 32-dim hidden MLP
            gaussians._gaussian_feat_dim = 0
            gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

            num_levels = cfg_model.encoding.levels
            per_level_dim = cfg_model.encoding.hashgrid.dim
            n_gaussians = len(gaussians.get_xyz)
            print(f"[3D_SH_32 MODE] Initialized {n_gaussians} Gaussians")
            print(f"[3D_SH_32 MODE] Per-Gaussian: standard SH (degree-3, 48 params)")
            print(f"[3D_SH_32 MODE] Hash levels: {num_levels}, {per_level_dim}D per level")
            print(f"[3D_SH_32 MODE] MLP: 32D → 32D → 32D → 3D (RGB residual, identity)")

        elif args.method == "3D_SH_res":
            # 3D_SH_res: per-Gaussian SH + tiny hash MLP residual
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
        skip_methods = ["cat", "adaptive", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_cat", "3D_SH_32"]
        if args.method not in skip_methods and args.kernel == "gaussian":
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
            if hasattr(args, 'method') and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
                gaussians._gaussian_feat_dim = 0
                gaussians._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
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
        f.write("=" * 70 + "\n")
        f.write(f"{'Iter':<10}{'PSNR(dB)':<10}{'SSIM':<9}{'LPIPS':<9}{'L1':<12}{'Points':<12}\n")
        f.write("-" * 70 + "\n")

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

    # FastGS Compact Box: set the Mahalanobis² multiplier + auto-enable AdR+rect AABB.
    if args.fastgs and args.method == "3D_SH_res":
        from diff_surfel_3D_sh_res import set_compact_mult
        set_compact_mult(args.fastgs_mult)
        if args.aabb == "2dgs":
            # Caller didn't pick an AABB mode — switch to adrrect so the compact-box
            # cutoff math actually runs in preprocessCUDA.
            args.aabb = "adrrect"
            print("[FASTGS] Auto-enabled --aabb adrrect for Compact Box.")
        print(f"[FASTGS] Compact Box: mult={args.fastgs_mult} (cutoff = sqrt(2·log(opacity·255)·mult), paper default 0.5)")

    # Set hash query transmittance threshold (skip hash+MLP when T < threshold)
    if args.contribution_thresh > 0.0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_contrib_thresh
        else:
            from diff_surfel_3D_sh_res import set_contrib_thresh
        set_contrib_thresh(args.contribution_thresh)
        print(f"[CONTRIB_THRESH] Skipping hash query when w = T*alpha < {args.contribution_thresh}")

    if args.count_thresh > 0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_count_thresh
        else:
            from diff_surfel_3D_sh_res import set_count_thresh
        set_count_thresh(args.count_thresh)
        print(f"[COUNT_THRESH] Skipping hash query after {args.count_thresh} contributing Gaussians per pixel")

    if args.overdraw_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_overdraw_lambda
        else:
            from diff_surfel_3D_sh_res import set_overdraw_lambda
        set_overdraw_lambda(args.overdraw_reg)
        print(f"[OVERDRAW_REG] Overdraw regularization lambda = {args.overdraw_reg}")

    if args.weight_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_weight_reg_lambda
        else:
            from diff_surfel_3D_sh_res import set_weight_reg_lambda
        set_weight_reg_lambda(args.weight_reg)
        print(f"[WEIGHT_REG] Weight-squared regularization lambda = {args.weight_reg} (CUDA gradient)")

    if args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_activation_bias
        else:
            from diff_surfel_3D_sh_res import set_activation_bias
        _sh_bias, _res_bias = args.activation_bias
        set_activation_bias(sh_bias=_sh_bias, res_bias=_res_bias)
        from gaussian_renderer import set_default_activation_bias
        set_default_activation_bias(_sh_bias, _res_bias)
        print(f"[ACTIVATION_BIAS] SH bias={_sh_bias}, residual bias={_res_bias}")

        # 3D_SH_add: flip outer activation to separate ReLUs (mode 1). Default is 0.
        if getattr(args, '_residual_mode', 0) == 1 and args.method == "3D_SH_res":
            from diff_surfel_3D_sh_res import set_residual_mode
            set_residual_mode(1)
            print("[RESIDUAL_MODE] mode=1 (3D_SH_add: separate outer ReLUs for SH and residual)")

    if args.depth_sort and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
        if args.method == "3D_SH_32":
            from diff_surfel_3D_sh_32 import set_depth_sort
        else:
            from diff_surfel_3D_sh_res import set_depth_sort
        set_depth_sort(True)
        print(f"[DEPTH_SORT] Using separated depth sort")

    # Unbiased Depth: set the per-pair depth-difference cutoff to scene_radius / 4
    # (paper formulation). The hardcoded 1.0 default is fine for DTU/T&T-scale
    # objects but clips most pairs in mip-360-scale outdoor scenes (extent ≈ 5).
    # `set_converge_threshold` resolves to the unbiased fork via the sys.modules
    # swap at the top of this file when --unbiased is on.
    if args.unbiased:
        try:
            from diff_surfel_3D_sh_res import set_converge_threshold
            _converge_thresh = float(scene.cameras_extent) / 4.0
            set_converge_threshold(_converge_thresh)
            print(f"[UNBIASED] CONVERGE_THRESHOLD = scene.cameras_extent / 4 = {_converge_thresh:.4f}")
        except (ImportError, AttributeError) as _e:
            print(f"[UNBIASED] WARNING: set_converge_threshold not available ({_e}); "
                  f"falling back to compiled default 1.0")

    if args.aa_2dgs > 0.0 and args.method == "3D_SH_res":
        from diff_surfel_3D_sh_res import set_aa_kernel_size
        set_aa_kernel_size(args.aa_2dgs)
        print(f"[AA-2DGS] Jacobian mip-filter kernel σ = {args.aa_2dgs} (3D_SH_res standard Gaussian path)")
    elif args.aa_2dgs > 0.0:
        raise RuntimeError(f"--aa_2dgs is only supported for --method 3D_SH_res, got {args.method}")

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

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # LR schedule: reset to "iteration 5000" after GSPA Phase 1 pruning.
        # Under --minispa, Phase 1 is skipped (silhouette reinit serves that role)
        # and the reinit itself calls reset_xyz_lr_schedule — don't double-shift.
        if args.gspa and iteration >= args.gspa_simp_iter and not args.minispa:
            gaussians.update_learning_rate(iteration - args.gspa_simp_iter + 5000)
        else:
            gaussians.update_learning_rate(iteration)

        if args.nexelparam and ingp_model is not None and hasattr(ingp_model, 'update_nexel_lr'):
            ingp_model.update_nexel_lr(iteration)

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
            _FP_HAS_BIAS = args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]
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
                    if args.method == "3D_SH_32":
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
                    if args.method == "3D_SH_32":
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
            if args.res_warmup > 0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"] and iteration < args.res_warmup:
                ingp.hashgrid_disabled = True
                optim_ngp = False
                optim_gaussian = True
            else:
                if args.res_warmup > 0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"] and iteration == args.res_warmup:
                    ingp.hashgrid_disabled = False
                    tqdm.write(f"[3D_SH_RES] Enabling hash/MLP residual at iteration {iteration}")

                active_levels = ingp.set_active_levels(iteration)
                optim_ngp = True
                optim_gaussian = ingp.optim_gaussian

            # Periodic hashgrid+MLP freeze for 3D_SH_res. After `freeze_hash_iter`,
            # train hash+MLP on 1 iter out of every `freeze_hash_period`. On skip
            # iters: CUDA backward skips all hash/MLP-grad work (weight-grad GEMMs,
            # input-chain backprop, query_feature<true>, tile flush) and the
            # optimizer.step() for ingp is skipped. Geometry backward runs unchanged.
            if (args.freeze_hash_iter > 0 and iteration >= args.freeze_hash_iter
                    and args.method == "3D_SH_res"):
                period = max(1, int(args.freeze_hash_period))
                should_train = (iteration % period) == 0
                desired_skip = not should_train
                # Flip the CUDA flag only on state transitions (saves a 1-thread
                # kernel launch every iter).
                current_skip = getattr(ingp, '_skip_mlp_grad', None)
                if current_skip != desired_skip:
                    try:
                        from diff_surfel_3D_sh_res import set_skip_mlp_grad
                        set_skip_mlp_grad(desired_skip)
                    except (ImportError, AttributeError):
                        tqdm.write("[FREEZE_HASH] WARN: set_skip_mlp_grad not available; rebuild?")
                    ingp._skip_mlp_grad = desired_skip
                    if current_skip is None:
                        tqdm.write(f"[FREEZE_HASH] Periodic freeze active at iter {iteration} "
                                   f"(train hash+MLP 1 of every {period} iters)")
                optim_ngp = should_train

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
        if iteration == first_iter + 1 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
            dc_norm = gaussians._features_dc.data.abs().max().item()
            rest_norm = gaussians._features_rest.data.abs().max().item()
            print(f"[DEBUG] First iter SH check: DC max={dc_norm:.6f}, REST max={rest_norm:.6f}, "
                  f"active_sh_degree={gaussians.active_sh_degree}")

        # Timing: forward pass
        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_fwd_start = time.time()

        render_pkg = render(viewpoint_cam, gaussians, pipe, current_bg, ingp = ingp,
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance,
            use_xyz_mode = args.use_xyz_mode, decompose_mode = dataset.decompose_mode,
            temperature = temperature, force_ratio = args.force_ratio, no_gumbel = args.no_gumbel,
            dropout_lambda = args.dropout_lambda, is_training = True, aabb_mode = args.aabb,
            aa = args.aa, aa_threshold = args.aa_threshold, skybox = active_skybox,
            background_mode = background_mode, bg_hashgrid = active_bg_hashgrid,
            detach_hash_grad = args.detach_hash_grad, max_intersections_per_pixel = args.max_intersections_per_pixel,
            lowpass = args.lowpass, pixel_center = args.pixel_center,
            antialiasing = args.antialiasing, sv_metric = args.sv_metric)

        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_fwd_end = time.time()

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # MSv2 SparseGaussianAdam: track visibility for sparse optimizer step
        if args.mini:
            mini_last_visibility = radii > 0

        gt_image = viewpoint_cam.original_image.cuda()
        
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

            # Apply random background to rendered image
            image = image + (1.0 - rend_alpha) * random_bg

            # Apply same random background to GT image
            gt_alpha_for_bg = viewpoint_cam.gt_alpha_mask.cuda().float() if cfg_model.settings.gt_alpha else (gt_image != 0).any(dim=0, keepdim=True).float()
            gt_image = gt_image + (1.0 - gt_alpha_for_bg) * random_bg
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

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

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
                antialiasing=args.antialiasing, sv_metric=args.sv_metric)

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
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        pixels = None
        if record_transmittance:
            pixels = render_pkg["cover_pixels"]
            transmittance_avg = render_pkg["transmittance_avg"]

        scales = gaussians.get_scaling
        alpha = gaussians.get_opacity
        
        mask_error = l1_loss(gt_alpha, rend_alpha).mean()
        mask_loss = lambda_mask * mask_error

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        if args.w_normal > 0.0 and iteration > cfg_model.loss.normal_iter:
            # Weighted normal consistency: relax where RGB error is high
            mse_per_pixel_n = ((image - gt_image) ** 2).mean(dim=0, keepdim=True).detach()
            w_n = torch.exp(-args.w_normal_gamma * mse_per_pixel_n)
            normal_loss = args.w_normal * (w_n * normal_error).mean()
        else:
            normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

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
            if args.kernel in ["beta", "beta_scaled"]:
                # Push β toward 0 (flat disks), same direction as lambda_shape.
                shape_reg_loss = args.w_lambda * w_s * gaussians.get_shape.mean()
            else:  # general kernel: push β toward 8 (flat/super-Gaussian box).
                shape_reg_loss = args.w_lambda * w_s * (8.0 - gaussians.get_shape).mean()
        elif args.kernel in ["beta", "beta_scaled"] and args.lambda_shape > 0 and shape_phase_active and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
            # L1 penalty on shape values - pushes toward 0 (hard disks)
            shape_reg_loss = args.lambda_shape * gaussians.get_shape.mean()

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
        if args.w_weight_reg > 0.0 and args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
            avg_mse = ((image - gt_image) ** 2).mean().detach().item()
            effective_lambda = args.w_weight_reg * float(np.exp(-args.w_weight_gamma * avg_mse))
            if args.method == "3D_SH_32":
                from diff_surfel_3D_sh_32 import set_weight_reg_lambda
            else:
                from diff_surfel_3D_sh_res import set_weight_reg_lambda
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

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_reg_loss + scout_loss + mcmc_opacity_reg + mcmc_scale_reg + adaptive_cat_reg_loss + adaptive_zero_reg_loss + adaptive_gate_reg_loss + bce_opacity_loss + shape_reg_loss + flex_beta_reg_loss + general_beta_reg_loss + l1_hash_loss + l1_sh_rest_loss + gspa_loss + w_overdraw_loss + sv_l1_loss + decomp_sh_loss + decomp_tex_loss + converge_loss

        # --minimc per-step error accumulation: BENCHED.
        # Replaced by the full-view sweep inside `minimc_sweep_and_relocate`,
        # which is dispatched once per `--minimc_relocate_interval`.

        # DEBUG: print loss components before backward
        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_bwd_start = time.time()

        total_loss.backward()

        if iteration % 500 == 0:
            torch.cuda.synchronize()
            _t_bwd_end = time.time()

        # Apply MLP gradients for 3D_direct_fused mode
        # MLP weights are in CUDA constant memory, gradients computed in CUDA backward
        # Skip when freeze_mlp is active (no weight gradients computed)
        if ingp is not None and hasattr(ingp, 'is_3D_direct_fused_mode') and ingp.is_3D_direct_fused_mode and not ingp.freeze_mlp:
            # Import from appropriate library based on mode
            if hasattr(ingp, 'is_3D_SH_32_mode') and ingp.is_3D_SH_32_mode:
                from diff_surfel_3D_sh_32 import get_mlp_grads
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
                if args.mcmc or args.mcmc_deficit or args.mcmc_fps or args.minimc:
                    n_alive = (gaussians.get_opacity > 0.005).sum().item()
                    points_str = f"{int(n_alive)}/{len(gaussians.get_xyz)}"
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
                background_mode = background_mode, bg_hashgrid_model = bg_hashgrid)

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
                        )
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
                if (iteration % 500 == 0 or iteration == first_iter) and ingp is not None and hasattr(ingp, 'hash_encoding') and ingp.hash_encoding is not None:
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

                if args.mini and mini_last_visibility is not None:
                    gaussians.optimizer.step(visibility=mini_last_visibility, N=radii.shape[0])
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if optim_ngp:
                    ingp.current_optimizer.step()
                    ingp.current_optimizer.zero_grad(set_to_none = True)

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

                img_name = os.path.join(output_path,  str(iteration) + '.png')
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), img_name)

                gt_name = os.path.join(output_path,  str(iteration) + '_gt.png')
                save_img_u8(gt_image.permute(1,2,0).detach().cpu().numpy(), gt_name)

                normal_name = os.path.join(output_path,  str(iteration) + '_normal.png')
                save_img_u8(rend_normal.permute(1,2,0).cpu().numpy() * 0.5 + 0.5, normal_name)

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

                # Save contributor heatmap every 5k iterations
                if iteration % 5000 == 0:
                    gs_num = render_pkg.get("gaussian_num", None)
                    if gs_num is not None:
                        heatmap, min_c, max_c = create_intersection_heatmap(gs_num, max_display=100)
                        heatmap_name = os.path.join(output_path, str(iteration) + '_contributors.png')
                        save_img_u8(heatmap, heatmap_name)

                # Save decomposed renders for 3D_SH_res and 3D_SH_cat modes (SH-only and texture-only)
                # All renders done atomically with the same model state (post-optimizer-step)
                if args.method in ["3D_SH_res", "3D_SH_cat", "3D_SH_32"] and ingp is not None and not getattr(ingp, 'hashgrid_disabled', False):
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
                        skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)

    print("\n" + "="*70)
    print(" "*20 + "FINAL TRAIN RENDERING")
    print("="*70)
    render_final_images(scene, gaussians, pipe, eval_background, final_ingp, beta, iteration, cfg_model, args,
                        cameras=scene.getTrainCameras(), output_subdir='final_train_renders', metrics_file='train_metrics.txt',
                        stride=25, skip_decomposition=True, skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)
    
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
        if args.method in ["cat", "cat_dropout", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_cat", "3D_SH_32"]:
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
                        skybox=None, background_mode="none", bg_hashgrid=None):
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

            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            psnr_val = psnr(rendered, gt).mean().item()
            ssim_val = ssim(rendered, gt).mean().item()
            lpips_val = lpips(rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()
            l1_val = l1_loss(rendered, gt).mean().item()

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

            # Always save depth maps to separate folder
            depth_expected = render_pkg['depth_expected']  # (1, H, W)
            depth_median = render_pkg['depth_median']  # (1, H, W)

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
            gaussian_num = render_pkg['gaussian_num']  # (1, H, W)
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
ingp_model, beta, args, cfg_model, test_psnr = None, train_psnr = None, iter_list = None, skybox_model = None, background_mode = "none", bg_hashgrid_model = None):
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
                # Use stride 25 for train cameras to speed up eval, stride 1 for test
                eval_stride = 25 if config['name'] == 'train' else 1
                cameras_evaluated = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if idx % eval_stride != 0:
                        continue

                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, ingp = ingp_model, \
                         beta = beta, iteration = iteration, cfg = cfg_model, skybox = active_skybox,
                         background_mode = background_mode, bg_hashgrid = active_bg_hashgrid)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    cameras_evaluated += 1

                    # Log images for first camera only
                    if tb_writer and cameras_evaluated == 1:
                        tb_writer.add_image(f'{config["name"]}/render', image, iteration)
                        tb_writer.add_image(f'{config["name"]}/gt', gt_image, iteration)

                        # Log FG/BG separation if skybox is active
                        if "render_fg" in render_pkg:
                            fg_image = torch.clamp(render_pkg["render_fg"], 0.0, 1.0)
                            bg_image = torch.clamp(render_pkg["render_bg"], 0.0, 1.0)
                            alpha = render_pkg["rend_alpha"]

                            tb_writer.add_image(f'{config["name"]}/foreground', fg_image, iteration)
                            tb_writer.add_image(f'{config["name"]}/background', bg_image, iteration)
                            tb_writer.add_image(f'{config["name"]}/alpha', alpha.repeat(3, 1, 1), iteration)

                psnr_test /= cameras_evaluated
                l1_test /= cameras_evaluated
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if config['name'] == 'test':
                    test_psnr.append(psnr_test.item())
                    # Append row to test_metrics.txt (header was written at training start).
                    try:
                        with open(os.path.join(scene.model_path, 'test_metrics.txt'), 'a') as f:
                            f.write(f"{iteration:<10}{psnr_test.item():<14.2f}{l1_test.item():<14.6f}\n")
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
                        choices=["baseline", "2dgs", "cat", "cat_dropout", "adaptive", "adaptive_add", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse", "specular", "diffuse_ngp", "diffuse_offset", "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC", "3D_SH_res", "3D_SH_add", "3D_SH_cat", "3D_SH_32"],
                        help="Rendering method: 'baseline' (default NeST), 'cat' (hybrid per-Gaussian + hashgrid), 'cat_dropout' (cat with hash dropout during training - use --dropout_lambda), 'adaptive' (learnable per-Gaussian blend), 'adaptive_add' (weighted sum of per-Gaussian and hashgrid features), 'adaptive_cat' (cat with learnable binary blend weights - trains smooth, infers binary), 'adaptive_zero' (cat with weighted hash vs zeros - w=0 skips hash query), 'adaptive_gate' (VQ-AD style gating: soft→STE→hard, L1 regularization toward zeros), 'diffuse' (SH degree 0, no viewdir), 'specular' (full 2DGS with SH), 'diffuse_ngp' (diffuse SH + hashgrid on unprojected depth), 'diffuse_offset' (diffuse SH as xyz offset for hashgrid query), 'hybrid_SH' (activate separately then add: SH→RGB+0.5+clamp + hashgrid→sigmoid, then add+clamp), 'hybrid_SH_raw' (add raw then activate: SH→raw + hashgrid→raw, then sigmoid), 'hybrid_SH_post' (DEPRECATED), 'residual_hybrid' (per-Gaussian SH RGB + hashgrid MLP residual), '3D' (intersection-based SH rendering), '3D_direct' (intersection-based RGB MLP), or '3D_direct_fused' (fused in-kernel MLP, no intersection buffer)")
    parser.add_argument("--hybrid_levels", type=int, default=5,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
    parser.add_argument("--hash_levels", type=int, default=-1,
                        help="3D_SH_res only: number of HASH levels (preferred name; default -1 = unset, use --hybrid_levels). "
                             "Internally translates to hybrid_levels = encoding.levels - hash_levels. "
                             "With levels=8 in config, --hash_levels K (K in [0..8]) gives K hash levels.")
    parser.add_argument("--decompose_mode", type=str, default=None,
                        choices=[None, "gaussian_only", "ngp_only"],
                        help="Decomposition mode for hybrid_SH visualization: 'gaussian_only' (only per-Gaussian SH), 'ngp_only' (only hashgrid DC residual), or None (normal combined rendering)")
    parser.add_argument("--disable_c2f", action="store_true",
                        help="Disable coarse-to-fine for cat mode (all levels active from start)")
    parser.add_argument("--dropout_lambda", type=float, default=0.0,
                        help="Hash dropout rate for cat_dropout mode: fraction of Gaussians that don't query hash during training (0.2 = 20%% dropout)")
    parser.add_argument("--lambda_adaptive", type=float, default=0.001,
                        help="Regularization weight for adaptive mode to encourage per-Gaussian features")
    parser.add_argument("--freeze_mlp", action="store_true",
                        help="Freeze MLP weights (random init or from --freeze_mlp_from). Only hashgrid learns. Skips MLP weight gradient computation in CUDA.")
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
    parser.add_argument("--sv_l1", type=float, default=1e-5,
                        help="L1 regularization on _sv_colors (sphericalvoronoi default 1e-5 for NeRF-synthetic, 0 for indoor/outdoor).")
    parser.add_argument("--sites_lr", type=float, default=5e-2,
                        help="--feature voronoi initial LR for _sv_sites (reference scheduler init). Decays exponentially to --sites_lr_final.")
    parser.add_argument("--sites_lr_final", type=float, default=1e-4,
                        help="--feature voronoi final LR for _sv_sites at end of training (reference scheduler final).")
    parser.add_argument("--sv_metric", type=str, default="l2",
                        choices=["l2", "cosine"],
                        help="SV logit metric: 'l2' (radiance default, -τ·||s_norm − ω||) or "
                             "'cosine' (paper formulation, s · ω = unconstrained dot product). "
                             "Cosine avoids L2's sqrt gradient blowup near alignment.")
    parser.add_argument("--sv_color_lr", type=float, default=1.25e-4,
                        help="--feature voronoi LR for _sv_colors (reference blender config: 0.000125).")
    parser.add_argument("--sv_dc", action="store_true",
                        help="--feature SV: add an explicit per-Gaussian view-independent "
                             "DC channel [N, 3]. SV becomes a directional residual on top. "
                             "Inits _sv_dc to pcd RGB and _sv_colors to zeros so the Gaussian "
                             "starts at its pcd color. Diverges from the strict reference "
                             "(which has no DC term) — opt-in.")
    parser.add_argument("--sv_dc_lr", type=float, default=2.5e-3,
                        help="--feature SV + --sv_dc: LR for _sv_dc. Default matches the "
                             "reference's sh_lr (0.0025).")
    parser.add_argument("--sb_number", type=int, default=2,
                        help="--feature beta: number of spherical beta primitives per Gaussian (K). Default 2")
    parser.add_argument("--sb_params_lr", type=float, default=0.0025,
                        help="--feature beta: LR for sb_params (per-primitive rgb/theta/phi/beta_raw). Default 0.0025")
    parser.add_argument("--sb_beta_lr", type=float, default=0.001,
                        help="--feature beta: LR for shared per-Gaussian sharpness. Default 0.001")

    # Beta kernel arguments
    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "beta", "beta_scaled", "flex", "general", "nexel"],
                        help="Kernel type: 'gaussian' (default exp(-0.5*r²)), 'beta' (pow(1-r², shape) with r∈[0,1]), 'beta_scaled' (same but r∈[0,3] to match 3σ Gaussian extent), 'flex' (Gaussian with learnable per-Gaussian beta), 'general' (Isotropic Generalized Gaussian), or 'nexel' (per-axis learnable gamma exponents, G=exp(-0.5*(s_x^2γx + s_y^2γy)))")
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
                        choices=["2dgs", "adr_only", "rect", "adr", "beta"],
                        help="AABB mode: '2dgs' (square, fixed 4σ - default), 'adr_only' (square, AdR cutoff), 'rect' (rectangular, fixed 4σ), 'adr' (rectangular + AdR cutoff), 'beta' (fixed r=1 for beta kernels)")
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
    parser.add_argument("--fastgs_mult", type=float, default=0.5,
                        help="FastGS Compact Box Mahalanobis² scale factor. "
                             "Tightens per-Gaussian tile AABB to ellipse contour where "
                             "opacity·exp(-0.5·maha²·mult) = 1/255. "
                             "1.0 = existing AdR cutoff (no change). 0.5 = paper default "
                             "(tighter, faster). Auto-enables --aabb adrrect when --fastgs set.")

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

    # --hash_levels: 3D_SH_res-friendly knob — K = number of HASH levels (vs --hybrid_levels
    # which is K = number of per-Gauss feature levels, leftover from CAT mode where
    # hybrid_levels + hash_levels = total_levels). For 3D_SH_res the per-surfel feature
    # is SH (no per-Gauss hash split), so users naturally think in terms of "how many
    # hash levels". Translate hash_levels → hybrid_levels for the rest of the pipeline.
    if args.hash_levels >= 0:
        if args.method not in ("3D_SH_res", "3D_SH_cat"):
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
