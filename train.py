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
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_scaling_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8, convert_gray_to_cmap
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
    
    # Check for warmup checkpoint in data directory
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
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
    elif checkpoint:
        # User-specified checkpoint takes priority
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    elif cfg_model.settings.if_ingp and os.path.exists(warmup_checkpoint_path):
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
        scene = Scene(dataset, gaussians)
        
        # Initialize per-Gaussian features for cat mode (trained from scratch after warmup)
        if args.method == "cat" and args.hybrid_levels > 0:
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
        # For cat/adaptive/adaptive_cat/diffuse mode, new params won't be in saved state - they train from scratch
        if args.method not in ["cat", "adaptive", "adaptive_cat", "diffuse"]:
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
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        if cfg_model.settings.if_ingp:
            print(f"\n[INFO] No warmup checkpoint found at {warmup_checkpoint_path}")
            print(f"[INFO] Will train 2DGS for {cfg_model.ingp_stage.initialize} iterations, then save checkpoint.\n")

    surfel_cfg = cfg_model.surfel
    gaussians.base_opacity = surfel_cfg.base_opacity
    beta = surfel_cfg.base_beta
    print(f'base opacity {surfel_cfg.base_opacity}, base beta {beta}')

    if not os.path.exists(os.path.join(scene.model_path, "training_output")):
        os.mkdir(os.path.join(scene.model_path, "training_output"))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # For diffuse_ngp/diffuse_offset: prepare alternating backgrounds to prevent RGB hiding
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    use_alternating_bg = args.method in ["diffuse_ngp", "diffuse_offset"]
    
    # Random background mode: use black BG until 15k iters, then random uniform background
    use_random_bg = args.random_background
    random_bg_start_iter = 15000
    if use_random_bg:
        print(f"Using black background until iteration {random_bg_start_iter}, then random uniform background (eval will use black background)")

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

    opacity_reset_protect = cfg_model.training_cfg.opacity_reset_protect
    if_pixel_densify_enhance = cfg_model.settings.pixel_densify_enhance
    
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
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, current_bg, ingp = ingp,
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance,
            use_xyz_mode = args.use_xyz_mode, decompose_mode = dataset.decompose_mode)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Apply random background for unbiased opacity training
        if use_random_bg and iteration >= random_bg_start_iter:
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
        if args.mcmc:
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

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_reg_loss + scout_loss + mcmc_opacity_reg + mcmc_scale_reg + adaptive_cat_reg_loss

        total_loss.backward()

        iter_end.record()
        
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_for_log
            
            # Track MCMC regularization losses
            if args.mcmc:
                mcmc_total = mcmc_opacity_reg.item() + mcmc_scale_reg.item()
                ema_mcmc_loss_for_log = 0.4 * mcmc_total + 0.6 * ema_mcmc_loss_for_log
            
            if iteration % 10 == 0:
                # For MCMC, show alive Gaussians (opacity > 0.005) instead of total
                if args.mcmc:
                    n_alive = (gaussians.get_opacity > 0.005).sum().item()
                    points_str = f"{int(n_alive)}/{len(gaussians.get_xyz)}"
                else:
                    points_str = f"{len(gaussians.get_xyz)}"
                
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": points_str,
                }
                # Add MCMC loss to progress bar if enabled
                if args.mcmc:
                    loss_dict["OpR"] = f"{mcmc_opacity_reg.item():.{5}f}"
                    loss_dict["ScR"] = f"{mcmc_scale_reg.item():.{5}f}"
                # Add adaptive_cat metrics to progress bar
                if args.method == "adaptive_cat" and hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight.numel() > 0:
                    mean_weight = torch.sigmoid(gaussians._adaptive_cat_weight).mean().item()
                    pct_gaussian = (torch.sigmoid(gaussians._adaptive_cat_weight) > 0.5).float().mean().item() * 100
                    loss_dict["W"] = f"{mean_weight:.2f}"
                    loss_dict["G%"] = f"{pct_gaussian:.0f}"
                    if adaptive_cat_reg_loss.item() > 0:
                        loss_dict["AdR"] = f"{adaptive_cat_reg_loss.item():.{5}f}"
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
                if args.mcmc:
                    tb_writer.add_scalar('train_loss_patches/mcmc_reg_loss', ema_mcmc_loss_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/mcmc_opacity_reg', mcmc_opacity_reg.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/mcmc_scale_reg', mcmc_scale_reg.item(), iteration)
                if args.method == "adaptive_cat" and hasattr(gaussians, '_adaptive_cat_weight') and gaussians._adaptive_cat_weight.numel() > 0:
                    mean_weight = torch.sigmoid(gaussians._adaptive_cat_weight).mean().item()
                    pct_gaussian = (torch.sigmoid(gaussians._adaptive_cat_weight) > 0.5).float().mean().item() * 100
                    tb_writer.add_scalar('adaptive_cat/mean_weight', mean_weight, iteration)
                    tb_writer.add_scalar('adaptive_cat/pct_gaussian', pct_gaussian, iteration)
                    tb_writer.add_scalar('adaptive_cat/reg_loss', adaptive_cat_reg_loss.item(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), \
                ingp_model=ingp, beta = beta, args = args, cfg_model = cfg_model, test_psnr = test_psnr, train_psnr = train_psnr, iter_list = iter_list)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if ingp is not None:
                    ingp.save_model(scene.model_path, iteration)

            # Densification / MCMC Relocation
            if iteration < opt.densify_until_iter and optim_gaussian:
                if args.mcmc:
                    # MCMC mode: relocate dead Gaussians and add new ones
                    if args.cap_max <= 0:
                        raise ValueError("--cap_max must be specified and positive when using --mcmc mode")
                    
                    if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                        # Find dead Gaussians (very low opacity)
                        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
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
                
                # MCMC: SGLD noise injection after optimizer step
                if args.mcmc:
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
    if args.mcmc:
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
        print("="*70 + "\n")

    # Final test and train rendering with stride 1
    final_ingp = ingp_model if ingp_model is not None else ingp

    # For random_background mode, use black background during evaluation
    eval_background = black_bg if use_random_bg else background
    
    print("\n" + "="*70)
    print(" "*20 + "FINAL TEST RENDERING")
    print("="*70)
    render_final_images(scene, gaussians, pipe, eval_background, final_ingp, beta, iteration, cfg_model, args, 
                        cameras=scene.getTestCameras(), output_subdir='final_test_renders', metrics_file='test_metrics.txt')
    
    print("\n" + "="*70)
    print(" "*20 + "FINAL TRAIN RENDERING")
    print("="*70)
    render_final_images(scene, gaussians, pipe, eval_background, final_ingp, beta, iteration, cfg_model, args,
                        cameras=scene.getTrainCameras(), output_subdir='final_train_renders', metrics_file='train_metrics.txt')
    
    # Save training log with point count and framerate
    save_training_log(scene, gaussians, final_ingp, pipe, args, cfg_model, iteration)


def save_training_log(scene, gaussians, ingp, pipe, args, cfg_model, iteration):
    """Save training statistics to training_log.txt."""
    import time
    
    log_path = os.path.join(scene.model_path, 'training_log.txt')
    
    # Count Gaussians
    num_gaussians = len(gaussians.get_xyz)
    
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
                      iteration=iteration, cfg=cfg_model)
            torch.cuda.synchronize()
            
            # Time multiple renders
            num_timing_iters = 100
            start_time = time.time()
            for _ in range(num_timing_iters):
                _ = render(viewpoint, gaussians, pipe, bg, ingp=ingp, beta=beta,
                          iteration=iteration, cfg=cfg_model)
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
        if args.method == "cat":
            f.write(f"Hybrid Levels: {args.hybrid_levels}\n")
        if args.mcmc:
            f.write(f"MCMC Mode: Enabled\n")
            f.write(f"  - Opacity Reg: {args.opacity_reg}\n")
            f.write(f"  - Scale Reg: {args.scale_reg}\n")
            f.write(f"  - Noise LR: {args.noise_lr}\n")
            f.write(f"  - Cap Max: {args.cap_max}\n")
            f.write(f"  - Note: Dead Gaussians (opacity ≤ 0.005) pruned before final rendering\n")
        f.write(f"Iterations: {iteration}\n")
        f.write(f"Resolution: {resolution}\n\n")

        f.write("Model Statistics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of Gaussians: {num_gaussians:,}\n")
        if args.mcmc:
            f.write(f"  (MCMC: only alive Gaussians counted)\n")
        f.write("\n")
        
        f.write("Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"Render FPS: {fps:.2f}\n")
        f.write(f"Time per frame: {ms_per_frame:.2f} ms\n")
    
    print(f"[LOG] Training log saved to: {log_path}")
    print(f"[LOG] Number of Gaussians: {num_gaussians:,}")
    print(f"[LOG] Render FPS: {fps:.2f} ({ms_per_frame:.2f} ms/frame)")


def render_final_images(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args, 
                        cameras, output_subdir, metrics_file, stride=1):
    """Render images and compute metrics."""
    
    final_output_dir = os.path.join(scene.model_path, output_subdir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Create depth output directory
    depth_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'depths'))
    os.makedirs(depth_output_dir, exist_ok=True)
    
    # Check if cat mode decomposition should be done
    is_cat_mode = (ingp is not None and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode 
                   and hasattr(args, 'hybrid_levels') and args.hybrid_levels > 0)
    total_levels = ingp.levels if ingp is not None else 0
    
    # Check if hybrid_SH, hybrid_SH_raw, or hybrid_SH_post mode decomposition should be done
    is_hybrid_sh_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_mode') and ingp.is_hybrid_sh_mode)
    is_hybrid_sh_raw_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_raw_mode') and ingp.is_hybrid_sh_raw_mode)
    is_hybrid_sh_post_mode = (ingp is not None and hasattr(ingp, 'is_hybrid_sh_post_mode') and ingp.is_hybrid_sh_post_mode)
    
    # Skip decomposition if hybrid_levels is 0 or equals total_levels (no meaningful decomposition)
    do_cat_decomposition = is_cat_mode and args.hybrid_levels < total_levels
    do_hybrid_sh_decomposition = is_hybrid_sh_mode or is_hybrid_sh_raw_mode or is_hybrid_sh_post_mode
    
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
    
    if len(cameras) == 0:
        print(f"[FINAL] No cameras available, skipping.")
        return
    
    psnr_values = []
    ssim_values = []
    l1_values = []
    rendered_indices = []
    
    with torch.no_grad():
        for idx, viewpoint in enumerate(cameras):
            if idx % stride != 0:
                continue
            
            render_pkg = render(viewpoint, gaussians, pipe, background, 
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)
            
            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            
            psnr_val = psnr(rendered, gt).mean().item()
            ssim_val = ssim(rendered, gt).mean().item()
            l1_val = l1_loss(rendered, gt).mean().item()
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            l1_values.append(l1_val)
            rendered_indices.append(idx)
            
            # Save images with consistent index naming
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            gt_np = gt.permute(1, 2, 0).cpu().numpy()
            
            save_img_u8(gt_np, os.path.join(final_output_dir, f"{idx:03d}_gt.png"))
            save_img_u8(rendered_np, os.path.join(final_output_dir, f"{idx:03d}_render.png"))
            
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
            
            save_img_u8(depth_expected_color, os.path.join(depth_output_dir, f"{idx:03d}_depth_expected.png"))
            save_img_u8(depth_median_color, os.path.join(depth_output_dir, f"{idx:03d}_depth_median.png"))
            
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
            
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            print(f"[FINAL] Idx {idx:3d} ({cam_name}): PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")
    
    # Summary
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)
    
    print(f"\n[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Metrics ({len(psnr_values)} images):")
    print(f"[FINAL]   Average PSNR: {avg_psnr:.2f} dB")
    print(f"[FINAL]   Average SSIM: {avg_ssim:.4f}")
    print(f"[FINAL]   Average L1:   {avg_l1:.6f}")
    print(f"[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Images saved to: {final_output_dir}")
    print(f"[FINAL] Depth maps saved to: {depth_output_dir}")
    if do_cat_decomposition or do_hybrid_sh_decomposition:
        print(f"[FINAL] NGP-only renders saved to: {ngp_output_dir}")
        print(f"[FINAL] Gaussian-only renders saved to: {gaussian_output_dir}")
    
    # Save metrics file in output root
    metrics_path = os.path.join(scene.model_path, metrics_file)
    with open(metrics_path, 'w') as f:
        f.write(f"Final Evaluation (stride={stride})\n")
        f.write(f"════════════════════════════════════════\n")
        f.write(f"Images rendered: {len(psnr_values)}\n")
        f.write(f"Average PSNR:    {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM:    {avg_ssim:.4f}\n")
        f.write(f"Average L1:      {avg_l1:.6f}\n\n")
        f.write(f"Per-image results:\n")
        f.write(f"{'Index':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'L1':<12}\n")
        f.write(f"{'-'*46}\n")
        
        for i, idx in enumerate(rendered_indices):
            f.write(f"{idx:<10} {psnr_values[i]:>10.2f} {ssim_values[i]:>10.4f} {l1_values[i]:>12.6f}\n")
    
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

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, \
ingp_model, beta, args, cfg_model, test_psnr = None, train_psnr = None, iter_list = None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, ingp = ingp_model, \
                         beta = beta, iteration = iteration, cfg = cfg_model)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
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
    
    # Method argument - baseline, cat, adaptive, adaptive_add, adaptive_cat, diffuse, specular, diffuse_ngp, diffuse_offset, hybrid_SH, hybrid_SH_raw, hybrid_SH_post, or residual_hybrid
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "cat", "adaptive", "adaptive_add", "adaptive_cat", "diffuse", "specular", "diffuse_ngp", "diffuse_offset", "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid"],
                        help="Rendering method: 'baseline' (default NeST), 'cat' (hybrid per-Gaussian + hashgrid), 'adaptive' (learnable per-Gaussian blend), 'adaptive_add' (weighted sum of per-Gaussian and hashgrid features), 'adaptive_cat' (cat with learnable binary blend weights - trains smooth, infers binary), 'diffuse' (SH degree 0, no viewdir), 'specular' (full 2DGS with SH), 'diffuse_ngp' (diffuse SH + hashgrid on unprojected depth), 'diffuse_offset' (diffuse SH as xyz offset for hashgrid query), 'hybrid_SH' (activate separately then add: SH→RGB+0.5+clamp + hashgrid→sigmoid, then add+clamp), 'hybrid_SH_raw' (add raw then activate: SH→raw + hashgrid→raw, then sigmoid), 'hybrid_SH_post' (DEPRECATED), or 'residual_hybrid' (per-Gaussian SH RGB + hashgrid MLP residual)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
    parser.add_argument("--decompose_mode", type=str, default=None,
                        choices=[None, "gaussian_only", "ngp_only"],
                        help="Decomposition mode for hybrid_SH visualization: 'gaussian_only' (only per-Gaussian SH), 'ngp_only' (only hashgrid DC residual), or None (normal combined rendering)")
    parser.add_argument("--disable_c2f", action="store_true",
                        help="Disable coarse-to-fine for cat mode (all levels active from start)")
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
    
    # Adaptive_cat arguments
    parser.add_argument("--lambda_adaptive_cat", type=float, default=0.01,
                        help="Entropy regularization weight for adaptive_cat binarization (pushes weights toward 0 or 1)")
    parser.add_argument("--adaptive_cat_anneal_start", type=int, default=15000,
                        help="Iteration to start annealing adaptive_cat entropy regularization (ramps from 0 to full strength)")
    parser.add_argument("--adaptive_cat_inference", action="store_true",
                        help="Use binary decisions at inference (weight>0.5 uses Gaussian only, skips intersection; weight<=0.5 uses hashgrid only)")
    
    # MCMC arguments - based on "3D Gaussian Splatting as Markov Chain Monte Carlo"
    parser.add_argument("--mcmc", action="store_true",
                        help="Enable MCMC-based Gaussian management (replaces traditional densification)")
    parser.add_argument("--cap_max", type=int, default=-1,
                        help="Maximum number of Gaussians (required for MCMC mode)")
    parser.add_argument("--opacity_reg", type=float, default=0.01,
                        help="L1 regularization weight on opacity (MCMC mode)")
    parser.add_argument("--scale_reg", type=float, default=0.01,
                        help="L1 regularization weight on scale (MCMC mode)")
    parser.add_argument("--noise_lr", type=float, default=5e5,
                        help="SGLD noise learning rate multiplier (MCMC mode)")

    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")
    if args.method == "cat":
        print(f"Hybrid levels: {args.hybrid_levels} (per-Gaussian features for coarse levels)")
        if args.disable_c2f:
            print(f"C2F disabled: all levels active from start")
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, \
        args)

    print("\nTraining complete.")
