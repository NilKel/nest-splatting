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
from utils.general_utils import safe_state
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
    
    # Pass method and hybrid_levels to dataset for use in Scene/GaussianModel
    dataset.method = args.method
    dataset.hybrid_levels = args.hybrid_levels if hasattr(args, 'hybrid_levels') else 3

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Check for warmup checkpoint in data directory
    warmup_checkpoint_path = os.path.join(dataset.source_path, "warmup_checkpoint.pth")
    loaded_from_warmup = False
    
    if checkpoint:
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
        # For cat/adaptive/diffuse mode, new params won't be in saved state - they train from scratch
        if args.method not in ["cat", "adaptive", "diffuse"]:
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
    
    # Random background mode: use random per-pixel background during training for unbiased opacity
    use_random_bg = args.random_background
    if use_random_bg:
        print("Using random per-pixel background during training (eval will use black background)")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0

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
            use_xyz_mode = args.use_xyz_mode)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Apply random per-pixel background for unbiased opacity training
        if use_random_bg:
            H, W = image.shape[1], image.shape[2]
            random_bg = torch.rand(3, H, W, device="cuda")
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
            if not loaded_from_warmup:
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

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss + adaptive_reg_loss + scout_loss

        total_loss.backward()

        iter_end.record()
        
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
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

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), \
                ingp_model=ingp, beta = beta, args = args, cfg_model = cfg_model, test_psnr = test_psnr, train_psnr = train_psnr, iter_list = iter_list)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if ingp is not None:
                    ingp.save_model(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter and optim_gaussian:
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
        
            if iteration == cfg_model.ingp_stage.initialize and cfg_model.settings.gs_alpha:   
                print('--- Generating mask by 2DGS.')
                from utils.image_utils import bilateral_filter_opencv

                if not os.path.exists(os.path.join(scene.model_path, "gs_alpha")):
                    os.mkdir(os.path.join(scene.model_path, "gs_alpha"))

                if not os.path.exists(os.path.join(scene.model_path, "gt_alpha")):
                    os.mkdir(os.path.join(scene.model_path, "gt_alpha"))

                # Collect gs_alpha masks for all cameras
                gs_alpha_masks = {}
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
                
                # Save warmup checkpoint (only if training from scratch)
                if not loaded_from_warmup:
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
    
    # Skip decomposition if hybrid_levels is 0 or equals total_levels (no meaningful decomposition)
    do_decomposition = is_cat_mode and args.hybrid_levels < total_levels
    
    if do_decomposition:
        # Create directories for decomposed renders
        ngp_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'ngp_only'))
        gaussian_output_dir = os.path.join(scene.model_path, output_subdir.replace('renders', 'gaussian_only'))
        os.makedirs(ngp_output_dir, exist_ok=True)
        os.makedirs(gaussian_output_dir, exist_ok=True)
        print(f"[FINAL] Cat mode decomposition enabled: saving NGP-only and Gaussian-only renders")
    
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
            if do_decomposition:
                # Save original gaussian features
                original_gaussian_features = gaussians._gaussian_features.data.clone()
                
                # NGP-only: zero out gaussian features
                gaussians._gaussian_features.data.zero_()
                ngp_render_pkg = render(viewpoint, gaussians, pipe, background,
                                       ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)
                ngp_rendered = torch.clamp(ngp_render_pkg["render"], 0.0, 1.0)
                ngp_rendered_np = ngp_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(ngp_rendered_np, os.path.join(ngp_output_dir, f"{idx:03d}_ngp.png"))
                
                # Restore gaussian features
                gaussians._gaussian_features.data.copy_(original_gaussian_features)
                
                # Gaussian-only: temporarily disable hashgrid by setting active_hashgrid_levels to 0
                original_active_hashgrid = ingp.active_hashgrid_levels
                ingp.active_hashgrid_levels = 0
                gaussian_render_pkg = render(viewpoint, gaussians, pipe, background,
                                            ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)
                gaussian_rendered = torch.clamp(gaussian_render_pkg["render"], 0.0, 1.0)
                gaussian_rendered_np = gaussian_rendered.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gaussian_rendered_np, os.path.join(gaussian_output_dir, f"{idx:03d}_gaussian.png"))
                
                # Restore hashgrid levels
                ingp.active_hashgrid_levels = original_active_hashgrid
            
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
    if do_decomposition:
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
    """Merge specific sections from config into args"""
    target_sections = ['training_cfg', 'settings', 'loss']
    
    for section in target_sections:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
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

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')
    
    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")
    
    # Method argument - baseline, cat, adaptive, diffuse, specular, diffuse_ngp, or diffuse_offset
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "cat", "adaptive", "diffuse", "specular", "diffuse_ngp", "diffuse_offset"],
                        help="Rendering method: 'baseline' (default NeST), 'cat' (hybrid per-Gaussian + hashgrid), 'adaptive' (learnable per-Gaussian blend), 'diffuse' (SH degree 0, no viewdir), 'specular' (full 2DGS with SH), 'diffuse_ngp' (diffuse SH + hashgrid on unprojected depth), or 'diffuse_offset' (diffuse SH as xyz offset for hashgrid query)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of coarse levels to replace with per-Gaussian features (cat mode only)")
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

    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")
    if args.method == "cat":
        print(f"Hybrid levels: {args.hybrid_levels} (per-Gaussian features for coarse levels)")
    elif args.method == "diffuse":
        print(f"Diffuse mode: SH degree 0, no viewdir, no hashgrid")
    elif args.method == "specular":
        print(f"Specular mode: full 2DGS with SH (view-dependent), no hashgrid")
    elif args.method == "diffuse_ngp":
        print(f"Diffuse+NGP mode: diffuse SH + hashgrid on unprojected expected depth")
    elif args.method == "diffuse_offset":
        print(f"Diffuse+Offset mode: diffuse SH as xyz offset, hashgrid MLP for final RGB")

    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

    print("args: ", args)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, \
        args)

    print("\nTraining complete.")
