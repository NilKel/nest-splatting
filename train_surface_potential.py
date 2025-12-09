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
    tb_writer = prepare_output_and_logger(args, scene_name, args.yaml)
    dataset.model_path = args.model_path

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Check for pre-trained 2DGS Gaussians (for INGP training)
    gaussian_checkpoint_path = os.path.join(dataset.source_path, "gaussian_init.pth")
    loaded_pretrained = False
    ingp_model = None  # Initialize to None, will be created during checkpoint loading or in training loop
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif cfg_model.settings.if_ingp and os.path.exists(gaussian_checkpoint_path):
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║              ✓ PRE-TRAINED 2DGS CHECKPOINT FOUND                ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"[2DGS] Checkpoint location: {gaussian_checkpoint_path}")
        # Load to CPU first to avoid any CUDA state issues
        checkpoint_data = torch.load(gaussian_checkpoint_path, weights_only=False, map_location='cpu')
        print(f"[2DGS] Checkpoint was saved at iteration: {checkpoint_data.get('iteration', 'unknown')}")
        
        # Check if this is the new format (dict) or old format (tuple)
        if 'xyz' in checkpoint_data:
            # New simplified format - just load the parameters directly
            gaussians.active_sh_degree = checkpoint_data['active_sh_degree']
            gaussians._xyz = nn.Parameter(checkpoint_data['xyz'].cuda().contiguous().requires_grad_(True))
            gaussians._features_dc = nn.Parameter(checkpoint_data['features_dc'].cuda().contiguous().requires_grad_(True))
            gaussians._features_rest = nn.Parameter(checkpoint_data['features_rest'].cuda().contiguous().requires_grad_(True))
            gaussians._scaling = nn.Parameter(checkpoint_data['scaling'].cuda().contiguous().requires_grad_(True))
            gaussians._rotation = nn.Parameter(checkpoint_data['rotation'].cuda().contiguous().requires_grad_(True))
            gaussians._opacity = nn.Parameter(checkpoint_data['opacity'].cuda().contiguous().requires_grad_(True))
            gaussians.max_radii2D = checkpoint_data['max_radii2D'].cuda()
            gaussians.spatial_lr_scale = checkpoint_data['spatial_lr_scale']
            
            # Handle appearance_level (may not exist in old checkpoints)
            if 'appearance_level' in checkpoint_data:
                gaussians._appearance_level = nn.Parameter(checkpoint_data['appearance_level'].cuda().contiguous().requires_grad_(True))
            else:
                # Create appearance_level with correct size for loaded Gaussians
                init_level = 6
                ap_level = init_level * torch.ones((gaussians.get_xyz.shape[0], 1), device="cuda").float()
                gaussians._appearance_level = nn.Parameter(ap_level.requires_grad_(True))
                print(f"[2DGS] Created appearance_level with shape {gaussians._appearance_level.shape}")
            
            # Handle gaussian_features (for hybrid_features mode)
            # Always create fresh Parameters for training (checkpoint mainly loads point cloud)
            if args.method == 'hybrid_features':
                if args.hybrid_levels > 0:
                    per_gaussian_dim = args.hybrid_levels * cfg_model.encoding.hashgrid.dim
                    # Check if checkpoint has gaussian_features, otherwise initialize to zeros
                    if 'gaussian_features' in checkpoint_data:
                        gaussian_feats = checkpoint_data['gaussian_features'].cuda()
                    else:
                        gaussian_feats = torch.zeros((gaussians.get_xyz.shape[0], per_gaussian_dim), device="cuda")
                    # Always wrap in Parameter with requires_grad=True
                    gaussians._gaussian_features = nn.Parameter(gaussian_feats.float().contiguous().requires_grad_(True))
                    print(f"[2DGS] Created gaussian_features with shape {gaussians._gaussian_features.shape}, requires_grad={gaussians._gaussian_features.requires_grad}")
                else:
                    # hybrid_levels=0: pure baseline mode, no per-Gaussian features
                    gaussians._gaussian_features = None
                    print(f"[2DGS] Skipping gaussian_features (hybrid_levels=0, pure baseline)")
            else:
                # Default for other modes
                if 'gaussian_features' in checkpoint_data:
                    gaussian_feats = checkpoint_data['gaussian_features'].cuda()
                else:
                    gaussian_feats = torch.zeros((gaussians.get_xyz.shape[0], 12), device="cuda")
                gaussians._gaussian_features = nn.Parameter(gaussian_feats.float().contiguous().requires_grad_(True))
                print(f"[2DGS] Created gaussian_features with shape {gaussians._gaussian_features.shape}, requires_grad={gaussians._gaussian_features.requires_grad}")
            
            # Initialize training state (optimizer will be created fresh)
            gaussians.training_setup(opt)
            print(f"[2DGS] ✓ Loaded simplified checkpoint format")
        else:
            # Old format with full capture() tuple - use restore()
            gaussian_data = checkpoint_data['gaussians']
            gaussian_data_cuda = tuple(
                item.detach().clone().contiguous().cuda() if isinstance(item, torch.Tensor) else item 
                for item in gaussian_data
            )
            gaussians.restore(gaussian_data_cuda, opt)
            print(f"[2DGS] ✓ Loaded legacy checkpoint format")
        # When loading from checkpoint, we've already completed 2DGS training
        # So we can initialize INGP immediately and start INGP training
        # Set to initialize-1 so that iteration 'initialize' runs to generate gs_alpha_mask
        first_iter = cfg_model.ingp_stage.initialize - 1  # This allows iteration 10000 to run
        loaded_pretrained = True
        
        # Initialize INGP immediately when loading from checkpoint
        if cfg_model.settings.if_ingp:
            print(f"\n[INGP] Initializing immediately after checkpoint loading...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            ingp_model = INGP(cfg_model, method=args.method, hybrid_levels=args.hybrid_levels).to('cuda')
            print(f"[INGP] ✓ Initialized with method='{args.method}'" + (f", hybrid_levels={args.hybrid_levels}" if args.method == 'hybrid_features' else ""))
            print(f"[INGP] ➤ Ready for INGP training from iteration {first_iter + 1}")
            
            # Set ingp immediately for checkpoint loading
            ingp = ingp_model
        
        # Validate loaded Gaussians
        xyz = gaussians.get_xyz
        print(f"[2DGS] Loaded {len(xyz)} Gaussians")
        
        # Check all Gaussian parameters for NaN/Inf
        params_ok = True
        for name in ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']:
            param = getattr(gaussians, f'get_{name}', getattr(gaussians, f'_{name}', None))
            if param is not None and isinstance(param, torch.Tensor):
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[2DGS] ⚠️ ERROR: {name} contains NaN/Inf values!")
                    params_ok = False
        
        if not params_ok:
            print(f"[2DGS] ⚠️ Please delete {gaussian_checkpoint_path} and retrain from scratch")
            exit(1)
        
        # Clear CUDA cache and synchronize to ensure clean state
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        
        # Force CUDA context reset to clear any corrupted state
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            except:
                pass
        
        print(f"[2DGS] ➤ Skipping 2DGS training phase (iterations 0-{cfg_model.ingp_stage.initialize - 1})")
        print(f"[2DGS] ➤ Will do 1 warmup iteration at {first_iter} (2DGS mode)")
        print(f"[2DGS] ➤ Then switch to INGP training from iteration {first_iter + 1}")
        ingp_iters = opt.iterations - first_iter
        print(f"[2DGS] ➤ Will train for {ingp_iters} iterations with INGP")
        if ingp_iters <= 0:
            print(f"[2DGS] ⚠ WARNING: No INGP training will occur!")
            print(f"[2DGS] ⚠ Requested iterations ({opt.iterations}) <= checkpoint ({cfg_model.ingp_stage.initialize})")
            print(f"[2DGS] ⚠ Use --iterations {cfg_model.ingp_stage.initialize + 1000} or higher")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
    elif cfg_model.settings.if_ingp:
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║           ⚠ NO PRE-TRAINED 2DGS CHECKPOINT FOUND                ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"[2DGS] Checkpoint path: {gaussian_checkpoint_path}")
        print(f"[2DGS] ➤ Will train 2DGS from scratch (iterations 0-{cfg_model.ingp_stage.initialize})")
        print(f"[2DGS] ➤ Checkpoint will be saved at iteration {cfg_model.ingp_stage.initialize}")
        print(f"[2DGS] ➤ Then continue with INGP training (iterations {cfg_model.ingp_stage.initialize + 1}-{opt.iterations})")
        print(f"[2DGS] ➤ Future runs will load this checkpoint and skip 2DGS phase")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Store whether we need to save Gaussians at initialize iteration
    save_gaussian_init = cfg_model.settings.if_ingp and not loaded_pretrained

    surfel_cfg = cfg_model.surfel
    gaussians.base_opacity = surfel_cfg.base_opacity
    beta = surfel_cfg.base_beta
    print(f'base opacity {surfel_cfg.base_opacity}, base beta {beta}')

    if not os.path.exists(os.path.join(scene.model_path, "training_output")):
        os.mkdir(os.path.join(scene.model_path, "training_output"))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    
    # CRITICAL: Don't initialize INGP until we actually need it!
    # This prevents CUDA state corruption when loading from checkpoint
    ingp_needs_init = cfg_model.settings.if_ingp
    
    opacity_reset_protect = cfg_model.training_cfg.opacity_reset_protect
    if_pixel_densify_enhance = cfg_model.settings.pixel_densify_enhance
    
    for iteration in range(first_iter, opt.iterations + 1):        

        torch.cuda.synchronize()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        opacity_reset_interval = opt.opacity_reset_interval
        densification_interval = opt.densification_interval
        # Initialize INGP on first iteration where we need it (only for training from scratch)
        # When loading from checkpoint, INGP is already initialized above
        if ingp_needs_init and ingp_model is None and not loaded_pretrained and iteration > cfg_model.ingp_stage.initialize + 1:
            print(f"\n[INGP] Initializing at iteration {iteration}...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            ingp_model = INGP(cfg_model, method=args.method, hybrid_levels=args.hybrid_levels).to('cuda')
            print(f"[INGP] ✓ Initialized with method='{args.method}'" + (f", hybrid_levels={args.hybrid_levels}" if args.method == 'hybrid_features' else ""))
        
       
        
        if ingp_model is None:
            ingp = None
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif iteration < cfg_model.ingp_stage.initialize:
            # Before initialize iteration: use 2DGS only
            ingp = None
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif iteration == cfg_model.ingp_stage.initialize:
            # AT initialize iteration: 
            if loaded_pretrained:
                # Already have pre-trained Gaussians and alpha masks from checkpoint
                # Use INGP for rendering (warmup phase)
                ingp = ingp_model
            else:
                # Training from scratch: use 2DGS to generate alpha masks
                ingp = None
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
            appearance_update_threshold = 0.0
        elif iteration <= cfg_model.ingp_stage.switch_iter:
            # After initialize, before switch_iter: warmup INGP
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.densify_grad_threshold
        else:
            # After switch_iter: full INGP training
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.ingp_densify_threshold
            densification_interval = cfg_model.training_cfg.ingp_densification_interval
            opacity_reset_interval = cfg_model.training_cfg.ingp_opacity_reset_interval
        optim_gaussian = True
        optim_ngp = False
        active_levels = None
        beta_updated = False  # Track if beta changes this iteration

        if ingp is not None:
            active_levels = ingp.set_active_levels(iteration)
            optim_ngp = True
            optim_gaussian = ingp.optim_gaussian
            if iteration % surfel_cfg.update_interval == 0 and optim_gaussian \
                and beta < surfel_cfg.tg_beta and active_levels == cfg_model.encoding.levels:
                
                update_times = (surfel_cfg.update_interations / surfel_cfg.update_interval)
                gaussians.base_opacity += surfel_cfg.tg_base_alpha / update_times
                beta += surfel_cfg.tg_beta / update_times
                beta_updated = True  # Flag that beta was updated

        # # During warm up process, gaussians are fixed. 
        # for group in gaussians.optimizer.param_groups:
        #     for param in group['params']:
        #         param.requires_grad = optim_gaussian 
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        record_transmittance = if_pixel_densify_enhance & (iteration >= opt.pixel_densify_from_iter) & (iteration < opt.densify_until_iter)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, ingp = ingp, 
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        # Log Gaussian overlap statistics when beta changes
        if beta_updated and 'gaussian_num' in render_pkg:
            gaussian_nums = render_pkg['gaussian_num']
            avg_gaussians = gaussian_nums.mean().item()
            max_gaussians = gaussian_nums.max().item()
            num_gaussians = gaussians.get_xyz.shape[0]
            print(f"\n[Beta Update] Iter {iteration}: Beta={beta:.4f}, Total Gaussians={num_gaussians}, "
                  f"Avg/pixel={avg_gaussians:.1f}, Max/pixel={max_gaussians:.0f}\n")
    
        gt_image = viewpoint_cam.original_image.cuda()

        error_img = torch.abs(gt_image - image)

        if cfg_model.settings.gt_alpha :
            gt_alpha = viewpoint_cam.gt_alpha_mask.cuda().float()
        else:
            gt_alpha = (gt_image != 0).any(dim=0, keepdim=True).float()
        
        try:
            if cfg_model.settings.gs_alpha and ingp is not None:
                gt_alpha = viewpoint_cam.gs_alpha_mask.cuda().float()
        except:
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

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss

        total_loss.backward()
        breakpoint()
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
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    # "normal": f"{ema_normal_for_log:.{5}f}",
                    # "Mask":f"{ema_mask_for_log:.{5}f}",
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
            
            # Save Gaussians at the end of 2DGS phase (before INGP starts)
            if save_gaussian_init and iteration == cfg_model.ingp_stage.initialize:
                print("\n╔══════════════════════════════════════════════════════════════════╗")
                print("║                  💾 SAVING 2DGS CHECKPOINT                       ║")
                print("╚══════════════════════════════════════════════════════════════════╝")
                print(f"[2DGS] Iteration: {iteration}")
                print(f"[2DGS] Number of Gaussians: {len(gaussians.get_xyz)}")
                print(f"[2DGS] Saving to: {gaussian_checkpoint_path}")
                
                # Save only essential Gaussian parameters (no optimizer state to avoid corruption)
                checkpoint_data = {
                    'active_sh_degree': gaussians.active_sh_degree,
                    'xyz': gaussians._xyz.detach().cpu(),
                    'features_dc': gaussians._features_dc.detach().cpu(),
                    'features_rest': gaussians._features_rest.detach().cpu(),
                    'scaling': gaussians._scaling.detach().cpu(),
                    'rotation': gaussians._rotation.detach().cpu(),
                    'opacity': gaussians._opacity.detach().cpu(),
                    'max_radii2D': gaussians.max_radii2D.detach().cpu(),
                    'spatial_lr_scale': gaussians.spatial_lr_scale,
                    'appearance_level': gaussians._appearance_level.detach().cpu(),
                    'iteration': iteration,
                }
                torch.save(checkpoint_data, gaussian_checkpoint_path)
                print(f"[2DGS] ✓ Checkpoint saved successfully!")
                print(f"[2DGS] ➤ Saved parameters: xyz, features, scaling, rotation, opacity")
                print(f"[2DGS] ➤ Future runs will load this checkpoint")
                print(f"[2DGS] ➤ Next: Starting INGP training at iteration {iteration + 1}")
                print("╚══════════════════════════════════════════════════════════════════╝\n")
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if ingp is not None:
                    ingp.save_model(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter and optim_gaussian:
            # if optim_gaussian:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels = pixels)
                
                prune_tag = (iteration % opacity_reset_interval >= opacity_reset_protect * densification_interval)
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
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

        with torch.no_grad():
        
            if iteration == cfg_model.ingp_stage.initialize and cfg_model.settings.gs_alpha:   
                print('--- Generating mask by 2DGS.')
                from utils.image_utils import bilateral_filter_opencv

                if not os.path.exists(os.path.join(scene.model_path, "gs_alpha")):
                    os.mkdir(os.path.join(scene.model_path, "gs_alpha"))

                if not os.path.exists(os.path.join(scene.model_path, "gt_alpha")):
                    os.mkdir(os.path.join(scene.model_path, "gt_alpha"))

                train_stack = scene.getTrainCameras() #.copy()
                for cam in tqdm(train_stack):
                    cam_name = cam.image_name + '.png'
                    render_pkg = render(cam, gaussians, pipe, background, ingp = ingp, \
                        beta = beta, iteration = iteration, cfg = cfg_model)
                    alpha_image = render_pkg["rend_alpha"]
                    bila_alpha = bilateral_filter_opencv(alpha_image.detach().cpu())
                    cam.gs_alpha_mask = bila_alpha.cpu().float()

                    alpha_name = os.path.join(scene.model_path, 'gs_alpha', cam_name)
                    save_img_u8(bila_alpha.permute(1,2,0).expand(-1,-1,3).numpy(), alpha_name)

                    alpha_name = os.path.join(scene.model_path, 'gt_alpha', cam_name)
                    save_img_u8(cam.gt_alpha_mask.permute(1,2,0).expand(-1,-1,3).detach().cpu().numpy(), alpha_name)

        torch.cuda.empty_cache()
    
    # Final test rendering with stride
    # Ensure ingp is set for final rendering (in case last iteration didn't use it)
    final_ingp = ingp_model if ingp_model is not None else ingp
    render_test_images_with_normals(scene, gaussians, pipe, background, final_ingp, beta, iteration, cfg_model, args, stride=args.test_render_stride)

def render_test_images_with_normals(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args, stride=1):
    """
    Render test images at the end of training with GT, rendered image, and normals.
    Also computes and reports PSNR and SSIM metrics.
    
    Args:
        scene: Scene object containing cameras and model path
        stride: Render every Nth test image (default: 25)
    """
    print(f"\n[FINAL] Rendering test images with stride {stride}...")
    
    # Create output directory
    final_output_dir = os.path.join(scene.model_path, 'final_test_renders')
    os.makedirs(final_output_dir, exist_ok=True)
    
    test_cameras = scene.getTestCameras()
    
    if len(test_cameras) == 0:
        print("[FINAL] No test cameras available, skipping final rendering")
        return
    
    # Metrics accumulation
    psnr_values = []
    ssim_values = []
    l1_values = []
    
    with torch.no_grad():
        for idx, viewpoint in enumerate(test_cameras):
            # Only render every 'stride' images
            if idx % stride != 0:
                continue
            
            # Render the image
            render_pkg = render(viewpoint, gaussians, pipe, background, 
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model)
            
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            rendered_normal = render_pkg["rend_normal"]
            
            # Compute metrics
            psnr_value = psnr(rendered_image, gt_image).mean().item()
            ssim_value = ssim(rendered_image, gt_image).mean().item()
            l1_value = l1_loss(rendered_image, gt_image).mean().item()
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            l1_values.append(l1_value)
            
            # Convert to numpy for saving
            rendered_np = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
            gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            normal_np = rendered_normal.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # Map to [0, 1]
            
            # Create concatenated image: GT | Rendered | Normal
            concat_image = np.concatenate([gt_np, rendered_np, normal_np], axis=1)
            
            # Save individual images
            # Use test index for clearer stride visualization, keep original name for reference
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            filename = f"test_{idx:03d}_{cam_name}"  # e.g., "test_000_r_11" or "test_025_r_47"
            
            # Save concatenated
            concat_name = os.path.join(final_output_dir, f"{filename}_concat.png")
            save_img_u8(concat_image, concat_name)
            
            # Also save individual components
            gt_name = os.path.join(final_output_dir, f"{filename}_gt.png")
            save_img_u8(gt_np, gt_name)
            
            render_name = os.path.join(final_output_dir, f"{filename}_render.png")
            save_img_u8(rendered_np, render_name)
            
            normal_name = os.path.join(final_output_dir, f"{filename}_normal.png")
            save_img_u8(normal_np, normal_name)
            
            print(f"[FINAL] Test idx {idx:3d} ({cam_name}): PSNR={psnr_value:.2f} SSIM={ssim_value:.4f}")
    
    # Compute and display average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)
    
    print(f"\n[FINAL] ═══════════════════════════════════════")
    print(f"[FINAL] Test Set Metrics ({len(psnr_values)} images):")
    print(f"[FINAL]   Average PSNR: {avg_psnr:.2f} dB")
    print(f"[FINAL]   Average SSIM: {avg_ssim:.4f}")
    print(f"[FINAL]   Average L1:   {avg_l1:.6f}")
    print(f"[FINAL] ═══════════════════════════════════════")
    print(f"[FINAL] Images saved to: {final_output_dir}\n")
    
    # Save metrics to file
    metrics_file = os.path.join(final_output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Final Test Set Evaluation\n")
        f.write(f"═══════════════════════════════════════\n")
        f.write(f"Number of test images: {len(psnr_values)}\n")
        f.write(f"Stride: {stride}\n")
        f.write(f"Iteration: {iteration}\n\n")
        f.write(f"Average Metrics:\n")
        f.write(f"  PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"  SSIM: {avg_ssim:.4f}\n")
        f.write(f"  L1:   {avg_l1:.6f}\n\n")
        f.write(f"Per-Image Metrics:\n")
        f.write(f"{'Image':<40} {'PSNR':>10} {'SSIM':>10} {'L1':>12}\n")
        f.write(f"{'-'*74}\n")
        
        for i, (idx, viewpoint) in enumerate((idx, cam) for idx, cam in enumerate(test_cameras) if idx % stride == 0):
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            filename = f"test_{idx:03d}_{cam_name}"
            f.write(f"{filename:<40} {psnr_values[i]:>10.2f} {ssim_values[i]:>10.4f} {l1_values[i]:>12.6f}\n")
    
    print(f"[FINAL] Metrics saved to: {metrics_file}")

def prepare_output_and_logger(args, scene_name, yaml_file = ""):    
    # Organize outputs: outputs/method/dataset/scene/name
    # Extract dataset name from source path
    source_parts = args.source_path.rstrip('/').split('/')
    scene_name_from_path = source_parts[-1]  # e.g., "drums"
    dataset_name = source_parts[-2] if len(source_parts) > 1 else "unknown"  # e.g., "nerf_synthetic"
    
    if not args.model_path:
        # If no model_path specified, create default name with timestamp
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        now = datetime.datetime.now()
        time_str = now.strftime("-%m%d-%H%M")
        exp_name = scene_name + time_str 
        if yaml_file != "":
            exp_name += '-' + yaml_file
        args.model_path = os.path.join("./output/", exp_name)
    else:
        # User specified -m flag: treat it as the run name and organize into structure
        run_name = args.model_path
        # Strip any leading ./ or / to get just the name
        run_name = run_name.lstrip('./').lstrip('/')
        # Build organized path: outputs/method/dataset/scene/name
        args.model_path = os.path.join("outputs", args.method, dataset_name, scene_name_from_path, run_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
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

    # Report test and samples of training set
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

def merge_cfg_to_args(args, cfg, cli_args):
    """Merge specific sections from config into args
    Args:
        args: ArgumentParser args
        cfg: Config object from yaml
        cli_args: Raw CLI arguments to check what was explicitly set
    """
    # Only flatten training_cfg, settings and loss sections
    target_sections = ['training_cfg', 'settings', 'loss']
    
    # Check if iterations was explicitly set via CLI
    iterations_set_via_cli = '--iterations' in cli_args
    
    for section in target_sections:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    # Don't override iterations if it was explicitly set via CLI
                    if k == 'iterations' and iterations_set_via_cli:
                        continue
                    setattr(args, k, v)

if __name__ == "__main__":
    # Set up command line argument parser
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
    parser.add_argument("--hybrid_levels", type=int, default=3, help="Number of finest hashgrid levels to use in hybrid_features mode (default: 3)")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')
    
    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "surface", "surface_blend", "surface_depth", "surface_rgb", "baseline_double", "baseline_blend_double", "hybrid_features"],
                        help="Rendering method: 'baseline' (default NeST), 'surface' (surface potential), 'surface_blend' (blend vectors, dot with rendered normals), 'surface_depth' (blend vectors, dot with depth gradient), 'surface_rgb' (surface potential + diffuse RGB), 'baseline_double' (dual 4D hashgrids: xyz + pk), 'baseline_blend_double' (dual 4D hashgrids: blended position + pk), or 'hybrid_features' (12D per-Gaussian + 12D from finest 3 hashgrid levels)")
    parser.add_argument("--disable_coarse_to_fine", action="store_true",
                        help="Disable coarse-to-fine hashgrid level annealing (use all levels from start)")
    parser.add_argument("--test_render_stride", type=int, default=25,
                        help="Stride for final test rendering (render every Nth test image, default: 25)")

    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")

    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model, sys.argv[1:])

    # Override coarse-to-fine setting if CLI flag is provided
    if args.disable_coarse_to_fine:
        cfg_model.encoding.coarse2fine.enabled = False
        print("\n[CLI Override] Coarse-to-fine disabled - using all hashgrid levels from start")

    print("args: ", args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, \
        args)

    # All done
    print("\nTraining complete.")