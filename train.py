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
import json
import yaml

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):

    testing_iterations += [opt.iterations]
    saving_iterations += [opt.iterations]

    test_psnr = []
    train_psnr = []
    iter_list = []

    scene_name = args.scene_name
    tb_writer = prepare_output_and_logger(dataset, scene_name, args.yaml, args)
    args.model_path = dataset.model_path

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)

    # Check for pre-trained 2DGS Gaussians (for INGP training) BEFORE creating Scene
    gaussian_checkpoint_path = os.path.join(dataset.source_path, "gaussian_init.pth")
    loaded_pretrained = False

    if checkpoint:
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif cfg_model.settings.if_ingp and os.path.exists(gaussian_checkpoint_path):
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*15 + "✓ PRE-TRAINED 2DGS CHECKPOINT FOUND" + " "*18 + "║")
        print("╚" + "═"*68 + "╝")
        print(f"[2DGS] Checkpoint location: {gaussian_checkpoint_path}")

        # Load checkpoint to CPU first
        checkpoint_data = torch.load(gaussian_checkpoint_path, weights_only=False, map_location='cpu')
        print(f"[2DGS] Checkpoint was saved at iteration: {checkpoint_data.get('iteration', 'unknown')}")

        # Check if this is the new format (dict) or old format (tuple)
        if 'xyz' in checkpoint_data:
            # New simplified format - load parameters directly
            gaussians.active_sh_degree = checkpoint_data['active_sh_degree']
            gaussians._xyz = nn.Parameter(checkpoint_data['xyz'].cuda().contiguous().requires_grad_(True))
            gaussians._features_dc = nn.Parameter(checkpoint_data['features_dc'].cuda().contiguous().requires_grad_(True))
            gaussians._features_rest = nn.Parameter(checkpoint_data['features_rest'].cuda().contiguous().requires_grad_(True))
            gaussians._scaling = nn.Parameter(checkpoint_data['scaling'].cuda().contiguous().requires_grad_(True))
            gaussians._rotation = nn.Parameter(checkpoint_data['rotation'].cuda().contiguous().requires_grad_(True))
            gaussians._opacity = nn.Parameter(checkpoint_data['opacity'].cuda().contiguous().requires_grad_(True))
            gaussians.max_radii2D = torch.zeros((len(checkpoint_data['xyz']),), device="cuda")
            gaussians.spatial_lr_scale = checkpoint_data['spatial_lr_scale']

            # Initialize appearance_level (required for INGP rendering)
            init_level = 24
            ap_level = init_level * torch.ones((len(checkpoint_data['xyz']), 1), device="cuda").float()
            gaussians._appearance_level = nn.Parameter(ap_level.requires_grad_(True))

            # Initialize per-Gaussian features (for "add" and "cat" modes)
            # In "add" mode: gaussian_feat_dim = total_levels * per_level_dim = 6 * 4 = 24
            # In "cat" mode: gaussian_feat_dim = hybrid_levels * per_level_dim
            if args.method == "cat":
                gaussian_feat_dim = args.hybrid_levels * 4  # per_level_dim = 4
            else:
                gaussian_feat_dim = 24  # 6 levels * 4 dim
            gaussian_feats = torch.zeros((len(checkpoint_data['xyz']), gaussian_feat_dim), device="cuda").float()
            gaussians._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
            
            print(f"[2DGS] Loaded {len(gaussians._xyz)} Gaussians")
            loaded_pretrained = True

            # Now create Scene with loaded Gaussians
            # Pass a special flag to prevent Scene from re-initializing the Gaussians
            # We set gaussians._loaded_from_checkpoint to signal Scene to skip create_from_pcd
            gaussians._loaded_from_checkpoint = True
            scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

            # Setup optimizer with loaded parameters
            gaussians.training_setup(opt)
            
            # Restore optimizer state if available (preserves Adam momentum)
            if 'optimizer_state' in checkpoint_data:
                print(f"[2DGS] Restoring optimizer state (Adam momentum & adaptive LR)")
                gaussians.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            else:
                print(f"[2DGS] Warning: No optimizer state in checkpoint - starting with fresh optimizer")
            
            # Restore densification statistics if available
            if 'xyz_gradient_accum' in checkpoint_data and 'denom' in checkpoint_data:
                print(f"[2DGS] Restoring densification statistics")
                gaussians.xyz_gradient_accum = checkpoint_data['xyz_gradient_accum'].cuda()
                gaussians.denom = checkpoint_data['denom'].cuda()
            else:
                print(f"[2DGS] Note: No densification stats in checkpoint - will start fresh")

            # Skip to INGP training phase
            first_iter = cfg_model.ingp_stage.initialize
            print(f"[2DGS] ✓ Skipping 2DGS training, starting from iteration {first_iter} (INGP phase)")
        else:
            print("[2DGS] Warning: Checkpoint format not recognized, training from scratch")
            scene = Scene(dataset, gaussians)
            gaussians.training_setup(opt)
    else:
        # Normal initialization without checkpoint
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)

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

    # Adjust total iterations if we loaded a checkpoint (already at 10k)
    total_iterations = opt.iterations
    if loaded_pretrained:
        # We're starting at 10k, so we only need to train for (total - 10k) more iterations
        actual_total = total_iterations
        print(f"[Training] Total iterations: {actual_total} (starting from {first_iter})")
    
    progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
    first_iter += 1

    ingp_model = None
    if cfg_model.settings.if_ingp:
        ingp_model = INGP(cfg_model, args).to('cuda')

        # Diagnostic: Verify INGP optimizer is properly set up
        if loaded_pretrained:
            print("\n[INGP] Optimizer diagnostic (after loading checkpoint):")
            print(f"[INGP] - Optimizer type: {type(ingp_model.optimizer)}")
            print(f"[INGP] - Parameter groups: {len(ingp_model.optimizer.param_groups)}")
            for i, pg in enumerate(ingp_model.optimizer.param_groups):
                name = pg.get('name', f'group_{i}')
                lr = pg['lr']
                num_params = len(pg['params'])
                total_params = sum(p.numel() for p in pg['params'])
                print(f"[INGP]   - {name}: lr={lr}, params={num_params}, total_params={total_params:,}")

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

            # Diagnostic: Print optimizer status for first few iterations after loading
            if loaded_pretrained and iteration <= first_iter + 5:
                print(f"[ITER {iteration}] optim_ngp={optim_ngp}, optim_gaussian={optim_gaussian}, active_levels={active_levels}")
            if iteration % surfel_cfg.update_interval == 0 and optim_gaussian \
                and beta < surfel_cfg.tg_beta and active_levels == cfg_model.encoding.levels:
                
                update_times = (surfel_cfg.update_interations / surfel_cfg.update_interval)
                gaussians.base_opacity += surfel_cfg.tg_base_alpha / update_times
                beta += surfel_cfg.tg_beta / update_times

        # During warm up process, gaussians are fixed (except gaussian_features which are part of INGP)
        for group in gaussians.optimizer.param_groups:
            # Keep gaussian_features trainable even when optim_gaussian=False
            # since they're part of the INGP feature learning in cat/add mode
            if group['name'] == 'gaussian_features':
                for param in group['params']:
                    param.requires_grad = True  # Always train gaussian_features
            else:
                for param in group['params']:
                    param.requires_grad = optim_gaussian 
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        record_transmittance = if_pixel_densify_enhance & (iteration >= opt.pixel_densify_from_iter) & (iteration < opt.densify_until_iter)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, ingp = ingp,
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance, render_mode = args.method)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
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
            # Only print warning if not using pretrained checkpoint (gs_alpha masks don't exist when skipping 2DGS training)
            if not loaded_pretrained:
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
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Save per-Gaussian features only at final iteration (30000)
                save_gaussian_features = (iteration == 30000 and args.method in ['add', 'cat'])
                scene.save(iteration, save_gaussian_features=save_gaussian_features)
                if ingp is not None:
                    ingp.save_model(scene.model_path, iteration)
                
                # Save complete config at final iteration for reproducible rendering
                if iteration == 30000:
                    save_complete_checkpoint_config(scene.model_path, args, cfg_model)

            # Densification (skip on last iteration to avoid corrupting final rendering)
            if iteration < opt.densify_until_iter and optim_gaussian and iteration < opt.iterations:
            # if optim_gaussian:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels = pixels)
                
                prune_tag = (iteration % opacity_reset_interval >= opacity_reset_protect * densification_interval)
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gaussians.densify_and_prune(densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, \
                    appearance_update_threshold, active_levels, densify_tag = (iteration < opt.densify_until_iter), prune_tag = prune_tag, iteration=iteration)
                
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

            # Save gaussian_init.pth at the end of 2DGS training phase (before INGP starts)
            if iteration == cfg_model.ingp_stage.initialize and not loaded_pretrained:
                gaussian_save_path = os.path.join(dataset.source_path, "gaussian_init.pth")
                print(f"\n[2DGS] Saving Gaussian checkpoint to: {gaussian_save_path}")
                checkpoint_data = {
                    'iteration': iteration,
                    'active_sh_degree': gaussians.active_sh_degree,
                    'xyz': gaussians._xyz.cpu(),
                    'features_dc': gaussians._features_dc.cpu(),
                    'features_rest': gaussians._features_rest.cpu(),
                    'scaling': gaussians._scaling.cpu(),
                    'rotation': gaussians._rotation.cpu(),
                    'opacity': gaussians._opacity.cpu(),
                    'max_radii2D': gaussians.max_radii2D.cpu(),
                    'spatial_lr_scale': gaussians.spatial_lr_scale,
                    # Save optimizer state to preserve Adam momentum and adaptive learning rates
                    'optimizer_state': gaussians.optimizer.state_dict(),
                    # Save densification statistics for continuing adaptive densification
                    'xyz_gradient_accum': gaussians.xyz_gradient_accum.cpu(),
                    'denom': gaussians.denom.cpu(),
                }
                torch.save(checkpoint_data, gaussian_save_path)
                print(f"[2DGS] ✓ Saved {len(gaussians._xyz)} Gaussians with optimizer state")

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
    # For 30k iteration training, render all images (stride=1) for comprehensive evaluation
    final_stride = 1 if args.iterations == 30000 else args.test_render_stride
    print("\n" + "="*70)
    print(" "*20 + "FINAL TEST RENDERING")
    print("="*70)
    if args.iterations == 30000:
        print(f"[INFO] 30k iteration training detected - rendering ALL test images (stride=1)")
    final_ingp = ingp_model if ingp_model is not None else ingp
    render_test_images_with_normals(scene, gaussians, pipe, background, final_ingp, beta, iteration, cfg_model, args, stride=final_stride)

    # Final train rendering with stride
    print("\n" + "="*70)
    print(" "*20 + "FINAL TRAIN RENDERING")
    print("="*70)
    render_train_images_with_normals(scene, gaussians, pipe, background, final_ingp, beta, iteration, cfg_model, args, stride=final_stride)


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
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model, render_mode=args.method)

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

            # Save individual images with index
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"

            # Save GT
            gt_name = os.path.join(final_output_dir, f"{idx:03d}_gt.png")
            save_img_u8(gt_np, gt_name)

            # Save rendered
            render_name = os.path.join(final_output_dir, f"{idx:03d}_render.png")
            save_img_u8(rendered_np, render_name)

            print(f"[FINAL] Test idx {idx:3d} ({cam_name}): PSNR={psnr_value:.2f} SSIM={ssim_value:.4f}")

    # Compute and display average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)

    print(f"\n[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Test Set Metrics ({len(psnr_values)} images):")
    print(f"[FINAL]   Average PSNR: {avg_psnr:.2f} dB")
    print(f"[FINAL]   Average SSIM: {avg_ssim:.4f}")
    print(f"[FINAL]   Average L1:   {avg_l1:.6f}")
    print(f"[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Images saved to: {final_output_dir}\n")

    # Save metrics to file (in model root, not inside final_test_renders)
    metrics_file = os.path.join(scene.model_path, "test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Final Test Set Evaluation (stride={stride})\n")
        f.write(f"════════════════════════════════════════\n")
        f.write(f"Images rendered: {len(psnr_values)}\n")
        f.write(f"Average PSNR:    {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM:    {avg_ssim:.4f}\n")
        f.write(f"Average L1:      {avg_l1:.6f}\n\n")
        f.write(f"Per-image results:\n")
        f.write(f"{'Index':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'L1':<12}\n")
        f.write(f"{'-'*46}\n")

        test_cameras = scene.getTestCameras()
        result_idx = 0
        for idx in range(len(test_cameras)):
            if idx % stride == 0:
                cam_name = test_cameras[idx].image_name if hasattr(test_cameras[idx], 'image_name') else f"view_{idx:03d}"
                f.write(f"{idx:<10} {psnr_values[result_idx]:>10.2f} {ssim_values[result_idx]:>10.4f} {l1_values[result_idx]:>12.6f}\n")
                result_idx += 1

    print(f"[FINAL] Metrics saved to: {metrics_file}")


def render_train_images_with_normals(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args, stride=1):
    """
    Render train images at the end of training with GT, rendered image, and normals.
    Also computes and reports PSNR and SSIM metrics.

    Args:
        scene: Scene object containing cameras and model path
        stride: Render every Nth train image (default: 25)
    """
    print(f"\n[FINAL] Rendering train images with stride {stride}...")

    # Create output directory
    final_output_dir = os.path.join(scene.model_path, 'final_train_renders')
    os.makedirs(final_output_dir, exist_ok=True)

    train_cameras = scene.getTrainCameras()

    if len(train_cameras) == 0:
        print("[FINAL] No train cameras available, skipping final rendering")
        return

    # Metrics accumulation
    psnr_values = []
    ssim_values = []
    l1_values = []

    with torch.no_grad():
        for idx, viewpoint in enumerate(train_cameras):
            # Only render every 'stride' images
            if idx % stride != 0:
                continue

            # Render the image
            render_pkg = render(viewpoint, gaussians, pipe, background,
                              ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model, render_mode=args.method)

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

            # Save individual images with index
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"

            # Save GT
            gt_name = os.path.join(final_output_dir, f"{idx:03d}_gt.png")
            save_img_u8(gt_np, gt_name)

            # Save rendered
            render_name = os.path.join(final_output_dir, f"{idx:03d}_render.png")
            save_img_u8(rendered_np, render_name)

            print(f"[FINAL] Train idx {idx:3d} ({cam_name}): PSNR={psnr_value:.2f} SSIM={ssim_value:.4f}")

    # Compute and display average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)

    print(f"\n[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Train Set Metrics ({len(psnr_values)} images):")
    print(f"[FINAL]   Average PSNR: {avg_psnr:.2f} dB")
    print(f"[FINAL]   Average SSIM: {avg_ssim:.4f}")
    print(f"[FINAL]   Average L1:   {avg_l1:.6f}")
    print(f"[FINAL] ════════════════════════════════════════")
    print(f"[FINAL] Images saved to: {final_output_dir}\n")

    # Save metrics to file (in model root, not inside final_train_renders)
    metrics_file = os.path.join(scene.model_path, "train_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Final Train Set Evaluation (stride={stride})\n")
        f.write(f"════════════════════════════════════════\n")
        f.write(f"Images rendered: {len(psnr_values)}\n")
        f.write(f"Average PSNR:    {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM:    {avg_ssim:.4f}\n")
        f.write(f"Average L1:      {avg_l1:.6f}\n\n")
        f.write(f"Per-image results:\n")
        f.write(f"{'Index':<10} {'PSNR (dB)':<12} {'SSIM':<12} {'L1':<12}\n")
        f.write(f"{'-'*46}\n")

        train_cameras = scene.getTrainCameras()
        result_idx = 0
        for idx in range(len(train_cameras)):
            if idx % stride == 0:
                cam_name = train_cameras[idx].image_name if hasattr(train_cameras[idx], 'image_name') else f"view_{idx:03d}"
                f.write(f"{idx:<10} {psnr_values[result_idx]:>10.2f} {ssim_values[result_idx]:>10.4f} {l1_values[result_idx]:>12.6f}\n")
                result_idx += 1

    print(f"[FINAL] Metrics saved to: {metrics_file}")


def prepare_output_and_logger(dataset, scene_name, yaml_file="", args=None):
    # Organize outputs: outputs/method/dataset/scene/name
    # Extract dataset name from source path
    source_parts = dataset.source_path.rstrip('/').split('/')
    scene_name_from_path = source_parts[-1]  # e.g., "drums"
    dataset_name = source_parts[-2] if len(source_parts) > 1 else "unknown"  # e.g., "nerf_synthetic"

    if not dataset.model_path:
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
        dataset.model_path = os.path.join("./output/", exp_name)
    else:
        # User specified -m flag: treat it as the run name and organize into structure
        run_name = dataset.model_path
        # Strip any leading ./ or / to get just the name
        run_name = run_name.lstrip('./').lstrip('/')
        # Build organized path: outputs/method/dataset/scene/name
        method = args.method if args and hasattr(args, 'method') else "baseline"
        # For cat mode, append hybrid_levels to run_name
        if method == "cat" and hasattr(args, 'hybrid_levels'):
            run_name = f"{run_name}_{args.hybrid_levels}level"
        dataset.model_path = os.path.join("outputs", method, dataset_name, scene_name_from_path, run_name)

    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create Tensorboard writer
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

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                # Use strided sampling - render the same frame indices each validation
                stride = args.test_render_stride
                camera_indices = list(range(0, len(config['cameras']), stride))
                num_cameras = len(camera_indices)
                
                for cam_idx in camera_indices:
                    viewpoint = config['cameras'][cam_idx]
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, ingp = ingp_model, \
                         beta = beta, iteration = iteration, cfg = cfg_model, render_mode = args.method)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= num_cameras
                l1_test /= num_cameras
                print("\n[ITER {}] Evaluating {} ({} views, stride {}): L1 {} PSNR {}".format(
                    iteration, config['name'], num_cameras, stride, l1_test, psnr_test))

                if config['name'] == 'test':
                    test_psnr.append(psnr_test.item())
                elif config['name'] == 'train':
                    train_psnr.append(psnr_test.item())
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        torch.cuda.empty_cache()

def save_complete_checkpoint_config(model_path, args, cfg_model):
    """
    Save complete configuration at final checkpoint (iteration 30000).
    Saves all CLI arguments, YAML config content, and method-specific parameters
    for exact reproducibility.
    
    Args:
        model_path: Path to model output directory
        args: Complete args from ArgumentParser
        cfg_model: Config object loaded from YAML
    """
    import json
    import yaml
    from datetime import datetime
    
    checkpoint_config = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'iteration': 30000,
        
        # === Complete CLI Arguments ===
        'cli_args': vars(args),  # Convert Namespace to dict with ALL arguments
        
        # === Method-specific parameters ===
        'method': args.method,
        'hybrid_levels': getattr(args, 'hybrid_levels', None),
        'cat_coarse2fine': getattr(args, 'cat_coarse2fine', False),
        
        # === YAML config path ===
        'yaml_config_path': args.yaml,
        
        # === YAML config content (for full reproducibility) ===
        'yaml_config_content': None,  # Will be populated below
        
        # === Model parameters ===
        'sh_degree': args.sh_degree,
        'source_path': args.source_path,
        'iterations': args.iterations,
        'resolution': args.resolution,
        'white_background': args.white_background,
        'eval': args.eval,
        
        # === Training parameters ===
        'densify_from_iter': cfg_model.training_cfg.densify_from_iter,
        'densify_until_iter': cfg_model.training_cfg.densify_until_iter,
        'densify_grad_threshold': cfg_model.training_cfg.densify_grad_threshold,
        'opacity_reset_interval': cfg_model.training_cfg.opacity_reset_interval,
        
        # === INGP parameters ===
        'if_ingp': cfg_model.settings.if_ingp,
        'ingp_levels': cfg_model.encoding.levels,
        'ingp_level_dim': cfg_model.encoding.hashgrid.dim,
        'coarse2fine_enabled': cfg_model.encoding.coarse2fine.enabled,
        'coarse2fine_init_active_level': cfg_model.encoding.coarse2fine.init_active_level,
        'coarse2fine_step': cfg_model.encoding.coarse2fine.step,
        
        # === View encoding ===
        'view_dep': cfg_model.rgb.view_dep,
        'view_encoding_type': cfg_model.rgb.encoding_view.type if cfg_model.rgb.view_dep else None,
        'view_encoding_degree': cfg_model.rgb.encoding_view.degree if cfg_model.rgb.view_dep else None,
    }
    
    # Load YAML config content
    try:
        yaml_path = args.yaml
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(os.path.dirname(__file__), yaml_path)
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                checkpoint_config['yaml_config_content'] = yaml_content
    except Exception as e:
        print(f"[Checkpoint Config] Warning: Could not load YAML content: {e}")
    
    # Save as JSON (human-readable)
    config_json_path = os.path.join(model_path, 'checkpoint_config.json')
    with open(config_json_path, 'w') as f:
        json.dump(checkpoint_config, f, indent=2, default=str)
    
    print(f"[Checkpoint Config] Saved complete config to {config_json_path}")
    
    # Also save as Python Namespace for backward compatibility
    config_namespace_path = os.path.join(model_path, 'checkpoint_args.txt')
    with open(config_namespace_path, 'w') as f:
        f.write(repr(args))
    
    # Save a human-readable summary
    summary_path = os.path.join(model_path, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("  Nest-Splatting Training Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Saved at: {checkpoint_config['saved_at']}\n")
        f.write(f"Final iteration: {checkpoint_config['iteration']}\n\n")
        
        f.write("Method Configuration:\n")
        f.write(f"  Method: {args.method.upper()}\n")
        if args.method == 'cat':
            f.write(f"  Hybrid levels: {args.hybrid_levels}\n")
            f.write(f"  Cat coarse2fine: {args.cat_coarse2fine}\n")
        f.write(f"\n")
        
        f.write("Dataset:\n")
        f.write(f"  Source: {args.source_path}\n")
        f.write(f"  Resolution: {args.resolution}\n")
        f.write(f"  White background: {args.white_background}\n")
        f.write(f"  Eval mode: {args.eval}\n")
        f.write(f"\n")
        
        f.write("INGP Configuration:\n")
        f.write(f"  Enabled: {cfg_model.settings.if_ingp}\n")
        f.write(f"  Levels: {cfg_model.encoding.levels}\n")
        f.write(f"  Level dim: {cfg_model.encoding.hashgrid.dim}\n")
        f.write(f"  Coarse2fine: {cfg_model.encoding.coarse2fine.enabled}\n")
        if cfg_model.encoding.coarse2fine.enabled:
            f.write(f"    Init levels: {cfg_model.encoding.coarse2fine.init_active_level}\n")
            f.write(f"    Step: {cfg_model.encoding.coarse2fine.step} iterations\n")
        f.write(f"\n")
        
        f.write("View Direction Encoding:\n")
        f.write(f"  Enabled: {cfg_model.rgb.view_dep}\n")
        if cfg_model.rgb.view_dep:
            f.write(f"  Type: {cfg_model.rgb.encoding_view.type}\n")
            f.write(f"  Degree: {cfg_model.rgb.encoding_view.degree}\n")
        f.write(f"\n")
        
        f.write("YAML Config:\n")
        f.write(f"  Path: {args.yaml}\n")
        f.write(f"\n")
        
        f.write("Checkpoint Files:\n")
        f.write(f"  Gaussians: point_cloud/iteration_30000/point_cloud.ply\n")
        if args.method in ['add', 'cat']:
            f.write(f"  Per-Gaussian features: point_cloud/iteration_30000/point_cloud_gaussian_features.pth\n")
        f.write(f"  INGP model: ngp_30000.pth\n")
        f.write(f"  Config: checkpoint_config.json\n")
        f.write(f"\n")
        
        f.write("="*70 + "\n")
        
        f.write("\nTo reproduce this training:\n")
        f.write("-"*70 + "\n")
        cmd = f"python train.py -s {args.source_path} -m <output_dir> --yaml {args.yaml}"
        if hasattr(args, 'eval') and args.eval:
            cmd += " --eval"
        if hasattr(args, 'iterations'):
            cmd += f" --iterations {args.iterations}"
        cmd += f" --method {args.method}"
        if args.method == 'cat':
            if hasattr(args, 'hybrid_levels'):
                cmd += f" --hybrid_levels {args.hybrid_levels}"
            if hasattr(args, 'cat_coarse2fine') and args.cat_coarse2fine:
                cmd += " --cat_coarse2fine"
        f.write(cmd + "\n")
        f.write("-"*70 + "\n")
    
    print(f"[Checkpoint Config] Saved training summary to {summary_path}")

def merge_cfg_to_args(args, cfg):
    """Merge specific sections from config into args
    Args:
        args: ArgumentParser args
        cfg: Config object from yaml
    
    Note: CLI arguments take priority over YAML config.
    Only applies YAML values if the CLI argument is still at its default value.
    """
    # Define defaults for arguments that might be overridden by YAML
    cli_defaults = {
        'iterations': 30_000,  # Default from OptimizationParams
    }
    
    # Only flatten training_cfg, settings and loss sections
    target_sections = ['training_cfg', 'settings', 'loss']
    
    for section in target_sections:
        if hasattr(cfg, section):
            section_dict = getattr(cfg, section)
            if isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    # If this argument has a known default and was explicitly set via CLI, don't override
                    if k in cli_defaults:
                        current_value = getattr(args, k, None)
                        if current_value != cli_defaults[k]:
                            # CLI argument was explicitly set, keep it
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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')

    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")

    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "add", "cat"],
                        help="Rendering method: 'baseline' (default NeST), 'add' (per-Gaussian features added to hashgrid), 'cat' (concatenate per-Gaussian and hashgrid features)")
    parser.add_argument("--hybrid_levels", type=int, default=3,
                        help="Number of levels for per-Gaussian features in 'cat' mode (default: 3)")
    parser.add_argument("--cat_coarse2fine", action="store_true",
                        help="Enable coarse-to-fine training for hashgrid in 'cat' mode (default: False)")
    parser.add_argument("--test_render_stride", type=int, default=25,
                        help="Stride for final test rendering (render every Nth test image, default: 25)")

    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")

    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)

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