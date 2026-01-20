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
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
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
from utils.point_utils import cam2rays
from utils.render_utils import gsnum_trans_color
import open3d as o3d
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):

    testing_iterations += [opt.iterations]
    saving_iterations += [opt.iterations]

    test_psnr = []
    train_psnr = []
    iter_list = []

    scene_name = args.scene_name
    tb_writer = prepare_output_and_logger(dataset, scene_name, args.yaml, args)
    args.model_path = dataset.model_path
    
    # Pass method, hybrid_levels, and decompose_mode to dataset for use in Scene/GaussianModel
    dataset.method = args.method
    dataset.hybrid_levels = args.hybrid_levels if hasattr(args, 'hybrid_levels') else 3
    dataset.decompose_mode = args.decompose_mode if hasattr(args, 'decompose_mode') else None

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)

    # Set kernel type for beta kernel support
    gaussians.kernel_type = args.kernel

    # Check for warmup checkpoint in data directory
    # If --warmup tag is specified, use warmup_checkpoint_{tag}.pth
    if args.warmup:
        warmup_checkpoint_path = os.path.join(dataset.source_path, f"warmup_checkpoint_{args.warmup}.pth")
    else:
        warmup_checkpoint_path = os.path.join(dataset.source_path, "warmup_checkpoint.pth")
    loaded_from_warmup = False
    
    # Cold start mode: skip all checkpoint loading
    if args.cold:
        print("\n" + "="*70)
        print("  COLD START MODE")
        print("="*70)
        print("  Skipping 2DGS warmup phase and all checkpoint loading")
        print("  Training Nest representation from scratch with hash_in_CUDA=True")
        print("="*70 + "\n")
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
    elif cfg_model.settings.if_ingp and os.path.exists(warmup_checkpoint_path) and not args.scratch:
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

        # Create scene (won't reinitialize Gaussians)
        gaussians._loaded_from_checkpoint = True
        scene = Scene(dataset, gaussians, mcmc_fps=args.mcmc_fps, cap_max=args.cap_max, full_args=args)

        # Initialize per-Gaussian features for cat/cat_dropout mode (trained from scratch after warmup)
        if args.method in ["cat", "cat_dropout"] and args.hybrid_levels > 0:
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

        # Initialize beta kernel shape parameter (if using beta or beta_scaled kernel)
        print(f"[DEBUG] Checking beta kernel init: args.kernel={args.kernel}")
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
        # For cat/adaptive/adaptive_cat/adaptive_zero/adaptive_gate/diffuse mode, new params won't be in saved state - they train from scratch
        # Also skip for beta/flex/general kernels which add new _shape/_flex_beta params not in warmup checkpoint
        if args.method not in ["cat", "adaptive", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse"] and args.kernel == "gaussian":
            gaussians.optimizer.load_state_dict(ckpt['optimizer_state'])
        
        # Move optimizer state to GPU
        for state in gaussians.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        # Restore densification state (critical for identical behavior)
        if 'xyz_gradient_accum' in ckpt:
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
        if args.scratch:
            print(f"\n[INFO] --scratch flag: ignoring warmup checkpoint, training from scratch")
            print(f"[INFO] Will train 2DGS for {cfg_model.ingp_stage.initialize} iterations, then save checkpoint.\n")
        elif cfg_model.settings.if_ingp:
            print(f"\n[INFO] No warmup checkpoint found at {warmup_checkpoint_path}")
            print(f"[INFO] Will train 2DGS for {cfg_model.ingp_stage.initialize} iterations, then save checkpoint.\n")

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

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # For diffuse_ngp/diffuse_offset: prepare alternating backgrounds to prevent RGB hiding
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    use_alternating_bg = args.method in ["diffuse_ngp", "diffuse_offset"]
    
    # Random background mode: use black BG until 10k iters, then random uniform background until 20k, then black again
    use_random_bg = args.random_background
    random_bg_start_iter = 10000
    random_bg_end_iter = 12000
    if use_random_bg:
        print(f"Using black background until iteration {random_bg_start_iter}, then random uniform background until {random_bg_end_iter}, then black background (eval will use black background)")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0
    ema_mcmc_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    ingp_model = None
    if cfg_model.settings.if_ingp:
        ingp_model = INGP(cfg_model, args=args).to('cuda')

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

    for iteration in range(first_iter, opt.iterations + 1):        

        torch.cuda.synchronize()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

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
            active_levels = ingp.set_active_levels(iteration)
            optim_ngp = True
            optim_gaussian = ingp.optim_gaussian
            if iteration % surfel_cfg.update_interval == 0 and optim_gaussian \
                and beta < surfel_cfg.tg_beta and active_levels == cfg_model.encoding.levels:
                
                update_times = (surfel_cfg.update_interations / surfel_cfg.update_interval)
                gaussians.base_opacity += surfel_cfg.tg_base_alpha / update_times
                beta += surfel_cfg.tg_beta / update_times

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
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

        render_pkg = render(viewpoint_cam, gaussians, pipe, current_bg, ingp = ingp,
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance,
            use_xyz_mode = args.use_xyz_mode, decompose_mode = dataset.decompose_mode,
            temperature = temperature, force_ratio = args.force_ratio, no_gumbel = args.no_gumbel,
            dropout_lambda = args.dropout_lambda, is_training = True, aabb_mode = args.aabb,
            aa = args.aa, aa_threshold = args.aa_threshold, skybox = active_skybox,
            background_mode = background_mode, bg_hashgrid = active_bg_hashgrid,
            detach_hash_grad = args.detach_hash_grad)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
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
                # Debug: print mask info on first iteration
                if iteration == first_iter + 1:
                    print(f"[DEBUG gt_alpha] Using gt_alpha_mask for {viewpoint_cam.image_name}")
                    print(f"[DEBUG gt_alpha] Shape: {gt_alpha.shape}, min: {gt_alpha.min():.3f}, max: {gt_alpha.max():.3f}, mean: {gt_alpha.mean():.3f}")
        else:
            gt_alpha = (gt_image != 0).any(dim=0, keepdim=True).float()
            if iteration == first_iter + 1:
                print(f"[DEBUG gt_alpha] Using fallback (non-black pixels) for {viewpoint_cam.image_name}")
        
        try:
            if cfg_model.settings.gs_alpha and ingp is not None:
                gt_alpha = viewpoint_cam.gs_alpha_mask.cuda().float()
        except:
            if not loaded_from_warmup and not args.cold:
                print(f"Error! no gs alpha for {viewpoint_cam.image_name} .")
            pass

        rend_alpha = render_pkg['rend_alpha']
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > cfg_model.loss.normal_iter else 0.0
        lambda_dist = opt.lambda_dist if iteration > cfg_model.loss.dist_iter else 0.0
        lambda_mask = opt.lambda_mask if iteration > cfg_model.loss.mask_iter else 0.0
        
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
        bce_phase_active = args.bce and iteration > (opt.iterations - args.bce_iter)
        if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
            # If --bce_solo, skip MCMC regularization during BCE phase to let BCE work alone
            if not (args.bce_solo and bce_phase_active):
                # Regularize activated opacity (after sigmoid) to encourage low opacity -> sparsity
                mcmc_opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
                # Regularize activated scale (after exp) to encourage small scales -> compact Gaussians
                mcmc_scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling).mean()

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
        if args.bce and iteration > (opt.iterations - args.bce_iter):
            # Get activated opacity (after sigmoid, in [0, 1])
            opacity = gaussians.get_opacity.squeeze()  # (N,)
            eps = 1e-7
            # BCE with target=opacity encourages opacity to be 0 or 1
            # BCE(p, p) = -p*log(p) - (1-p)*log(1-p) = entropy
            # This is minimized when p is 0 or 1
            bce = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps))
            bce_opacity_loss = args.bce_lambda * bce.mean()

        # Beta kernel shape regularization - encourage shapes toward 0 (hard flat disks)
        # Shape in range [0.001, 4.001]: low = hard disk, high = soft Gaussian cloud
        shape_reg_loss = torch.tensor(0.0, device="cuda")
        if args.kernel in ["beta", "beta_scaled"] and args.lambda_shape > 0 and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
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
        general_beta_reg_loss = torch.tensor(0.0, device="cuda")
        if args.kernel == "general" and args.lambda_shape != 0 and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
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

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_reg_loss + scout_loss + mcmc_opacity_reg + mcmc_scale_reg + adaptive_cat_reg_loss + adaptive_zero_reg_loss + adaptive_gate_reg_loss + bce_opacity_loss + shape_reg_loss + flex_beta_reg_loss + general_beta_reg_loss + l1_hash_loss

        total_loss.backward()

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
            
            # Track MCMC regularization losses
            if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
                mcmc_total = mcmc_opacity_reg.item() + mcmc_scale_reg.item()
                ema_mcmc_loss_for_log = 0.4 * mcmc_total + 0.6 * ema_mcmc_loss_for_log

            if iteration % 10 == 0:
                # For MCMC, show alive Gaussians (opacity > 0.005) instead of total
                if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
                    n_alive = (gaussians.get_opacity > 0.005).sum().item()
                    points_str = f"{int(n_alive)}/{len(gaussians.get_xyz)}"
                else:
                    points_str = f"{len(gaussians.get_xyz)}"

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": points_str,
                }
                # Add MCMC loss to progress bar if enabled (hide during BCE solo phase)
                if (args.mcmc or args.mcmc_deficit or args.mcmc_fps) and not (args.bce_solo and bce_phase_active):
                    loss_dict["OpR"] = f"{mcmc_opacity_reg.item():.{5}f}"
                    loss_dict["ScR"] = f"{mcmc_scale_reg.item():.{5}f}"
                # Add BCE phase indicator to progress bar
                if bce_phase_active:
                    loss_dict["BCE"] = "ON"
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
                    loss_dict["BCE"] = f"{bce_opacity_loss.item():.{5}f}"
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
                if args.kernel == "general":
                    loss_dict["GnR"] = f"{general_beta_reg_loss.item():.5f}"
                    if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
                        beta_vals = gaussians.get_shape
                        loss_dict["Gnβ"] = f"{beta_vals.mean().item():.2f}"
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

            # Densification / MCMC Relocation
            if iteration < opt.densify_until_iter and optim_gaussian:
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
                            gaussians.relocate_gs(dead_mask=dead_mask)
                            gaussians.add_new_gs(cap_max=args.cap_max)
                else:
                    # Traditional densification
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels = pixels)
                    
                    prune_tag = (iteration % opacity_reset_interval >= opacity_reset_protect * densification_interval)
                    # Diffuse_offset mode: pause pruning for first 3k iterations after initialize
                    if args.method == "diffuse_offset" and iteration < cfg_model.ingp_stage.initialize + 3000:
                        prune_tag = False
                    if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        gaussians.densify_and_prune(densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, \
                        appearance_update_threshold, active_levels, densify_tag = (iteration < opt.densify_until_iter), prune_tag = prune_tag)
                    
                    if iteration % opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        if iteration <= cfg_model.training_cfg.reset_until_iter:
                            gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if optim_ngp:
                    ingp.current_optimizer.step()
                    ingp.current_optimizer.zero_grad(set_to_none = True)

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
                if args.mcmc or args.mcmc_deficit or args.mcmc_fps:
                    # Get current xyz learning rate
                    xyz_lr = gaussians.optimizer.param_groups[0]['lr']
                    
                    # Build covariance from scale and rotation
                    L = build_scaling_rotation(
                        torch.cat([gaussians.get_scaling, torch.ones_like(gaussians.get_scaling[:, :1])], dim=-1),
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
            if (iteration )  % save_interval == 0:

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
                        skip_decomposition=True, skybox=skybox, background_mode=background_mode, bg_hashgrid=bg_hashgrid)
    
    # Save training log with point count and framerate
    save_training_log(scene, gaussians, final_ingp, pipe, args, cfg_model, iteration)


def save_training_log(scene, gaussians, ingp, pipe, args, cfg_model, iteration):
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
        if args.method in ["cat", "cat_dropout"]:
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
    
    print(f"[LOG] Training log saved to: {log_path}")
    print(f"[LOG] Number of Gaussians: {num_gaussians:,}")
    print(f"[LOG] Render FPS: {fps:.2f} ({ms_per_frame:.2f} ms/frame)")


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

    # Save metrics file in output root
    metrics_path = os.path.join(scene.model_path, metrics_file)
    with open(metrics_path, 'w') as f:
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
                elif config['name'] == 'train':
                    train_psnr.append(psnr_test.item())

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def merge_cfg_to_args(args, cfg):
    """Merge specific sections from config into args
    
    Note: For white_background, CLI flag takes precedence over config
    """
    target_sections = ['training_cfg', 'settings', 'loss']
    
    # Store CLI white_background value before merging
    cli_white_background = args.white_background if hasattr(args, 'white_background') else None
    
    for section in target_sections:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    setattr(args, k, v)
    
    # Restore CLI white_background if it was explicitly set
    if cli_white_background is not None and cli_white_background:
        args.white_background = cli_white_background

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

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')
    
    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")
    
    # Method argument - baseline, cat, cat_dropout, adaptive, adaptive_add, adaptive_cat, adaptive_zero, adaptive_gate, diffuse, specular, diffuse_ngp, diffuse_offset, hybrid_SH, hybrid_SH_raw, hybrid_SH_post, or residual_hybrid
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "cat", "cat_dropout", "adaptive", "adaptive_add", "adaptive_cat", "adaptive_zero", "adaptive_gate", "diffuse", "specular", "diffuse_ngp", "diffuse_offset", "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid"],
                        help="Rendering method: 'baseline' (default NeST), 'cat' (hybrid per-Gaussian + hashgrid), 'cat_dropout' (cat with hash dropout during training - use --dropout_lambda), 'adaptive' (learnable per-Gaussian blend), 'adaptive_add' (weighted sum of per-Gaussian and hashgrid features), 'adaptive_cat' (cat with learnable binary blend weights - trains smooth, infers binary), 'adaptive_zero' (cat with weighted hash vs zeros - w=0 skips hash query), 'adaptive_gate' (VQ-AD style gating: soft→STE→hard, L1 regularization toward zeros), 'diffuse' (SH degree 0, no viewdir), 'specular' (full 2DGS with SH), 'diffuse_ngp' (diffuse SH + hashgrid on unprojected depth), 'diffuse_offset' (diffuse SH as xyz offset for hashgrid query), 'hybrid_SH' (activate separately then add: SH→RGB+0.5+clamp + hashgrid→sigmoid, then add+clamp), 'hybrid_SH_raw' (add raw then activate: SH→raw + hashgrid→raw, then sigmoid), 'hybrid_SH_post' (DEPRECATED), or 'residual_hybrid' (per-Gaussian SH RGB + hashgrid MLP residual)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
    parser.add_argument("--decompose_mode", type=str, default=None,
                        choices=[None, "gaussian_only", "ngp_only"],
                        help="Decomposition mode for hybrid_SH visualization: 'gaussian_only' (only per-Gaussian SH), 'ngp_only' (only hashgrid DC residual), or None (normal combined rendering)")
    parser.add_argument("--disable_c2f", action="store_true",
                        help="Disable coarse-to-fine for cat mode (all levels active from start)")
    parser.add_argument("--dropout_lambda", type=float, default=0.0,
                        help="Hash dropout rate for cat_dropout mode: fraction of Gaussians that don't query hash during training (0.2 = 20%% dropout)")
    parser.add_argument("--lambda_adaptive", type=float, default=0.001,
                        help="Regularization weight for adaptive mode to encourage per-Gaussian features")
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
    parser.add_argument("--cap_max", type=int, default=-1,
                        help="Maximum number of Gaussians (required for MCMC mode)")
    parser.add_argument("--opacity_reg", type=float, default=0.01,
                        help="L1 regularization weight on opacity (MCMC mode)")
    parser.add_argument("--scale_reg", type=float, default=0.01,
                        help="L1 regularization weight on scale (MCMC mode)")
    parser.add_argument("--noise_lr", type=float, default=5e5,
                        help="SGLD noise learning rate multiplier (MCMC mode)")

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

    # Beta (Gaussian sharpening) override
    parser.add_argument("--beta", type=float, default=None,
                        help="Override tg_beta from config (higher = sharper Gaussians, more opaque throughout)")

    # Beta kernel arguments
    parser.add_argument("--kernel", type=str, default="gaussian",
                        choices=["gaussian", "beta", "beta_scaled", "flex", "general"],
                        help="Kernel type: 'gaussian' (default exp(-0.5*r²)), 'beta' (pow(1-r², shape) with r∈[0,1]), 'beta_scaled' (same but r∈[0,3] to match 3σ Gaussian extent), 'flex' (Gaussian with learnable per-Gaussian beta), or 'general' (Isotropic Generalized Gaussian)")
    parser.add_argument("--freeze_beta", type=float, default=None,
                        help="Freeze beta kernel shape to a fixed value (e.g., 3.0 for semisoft). Disables shape optimization.")
    parser.add_argument("--detach_hash_grad", action="store_true",
                        help="Detach positional gradients from hashgrid in CAT mode (geometry follows per-Gaussian features only)")
    parser.add_argument("--lambda_shape", type=float, default=0.001,
                        help="L1 regularization weight on beta kernel shape parameter (pushes toward 0 = hard disks)")
    parser.add_argument("--lambda_flex_beta", type=float, default=0.0001,
                        help="L1 regularization weight on flex kernel beta parameter (prevents runaway sharpening)")
    parser.add_argument("--l1_hash", type=float, default=0.0,
                        help="L1 regularization on hashgrid embeddings to encourage sparsity (0.0 = disabled)")
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

    args = parser.parse_args(sys.argv[1:])

    # --bce_solo implies --bce
    if args.bce_solo:
        args.bce = True

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
        print(f"BCE opacity regularization: enabled for last {args.bce_iter} iterations (lambda={args.bce_lambda})")

    # Always print kernel info
    print(f"Kernel type: {args.kernel.upper()}")
    if args.kernel == "beta":
        print(f"  Beta kernel: learnable per-Gaussian shape parameter")
        print(f"  - Formula: alpha = opacity * pow(1 - r², shape), r ∈ [0, 1]")
        print(f"  - Shape range: [0.001, 4.001] (0=hard disk, 4=soft cloud)")
        print(f"  - Shape regularization: lambda={args.lambda_shape} (L1 penalty pushes toward hard disks)")
    elif args.kernel == "beta_scaled":
        print(f"  Beta Scaled kernel: learnable per-Gaussian shape parameter (scaled radius)")
        print(f"  - Formula: alpha = opacity * pow(1 - (r/3)², shape), r ∈ [0, 3]")
        print(f"  - Radius scaled by 3 to match 3σ Gaussian extent")
        print(f"  - Shape range: [0.001, 4.001] (0=hard disk, 4=soft cloud)")
        print(f"  - Shape regularization: lambda={args.lambda_shape} (L1 penalty pushes toward hard disks)")
    elif args.kernel == "flex":
        print(f"  Flex kernel: Gaussian with learnable per-Gaussian beta (sharpening)")
        print(f"  - Formula: G = exp(power); alpha = (1+beta)*G / (1+beta*G)")
        print(f"  - Beta range: [0, inf) via softplus (0=standard Gaussian, higher=sharper)")
        print(f"  - Beta regularization: lambda={args.lambda_flex_beta} (L1 penalty prevents runaway)")
    elif args.kernel == "general":
        print(f"  General kernel: Isotropic Generalized Gaussian")
        print(f"  - Formula: alpha = opacity * exp(-0.5 * (r²)^(β/2))")
        print(f"  - Beta range: [2.0, 8.0] via sigmoid*6+2 (2=Gaussian, 8=super-Gaussian/box)")
        print(f"  - Beta regularization: lambda={args.lambda_shape}, mode={args.genreg}")
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
    merge_cfg_to_args(args, cfg_model)
    
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
