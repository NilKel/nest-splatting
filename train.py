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
    tb_writer = prepare_output_and_logger(dataset, scene_name, args.yaml)
    args.model_path = dataset.model_path

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

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

    ingp_model = None
    if cfg_model.settings.if_ingp:
        ingp_model = INGP(cfg_model).to('cuda')

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
        else:
            ingp = ingp_model
            densify_grad_threshold = cfg_model.training_cfg.ingp_densify_threshold
            densification_interval = cfg_model.training_cfg.ingp_densification_interval
            opacity_reset_interval = cfg_model.training_cfg.ingp_opacity_reset_interval
        
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

        # During warm up process, gaussians are fixed. 
        for group in gaussians.optimizer.param_groups:
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
            beta = beta, iteration = iteration, cfg = cfg_model, record_transmittance = record_transmittance)

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
    render_test_images_with_normals(dataset, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args, stride=args.test_render_stride)

def render_test_images_with_normals(dataset, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args, stride=25):
    """
    Render test images at the end of training with GT, rendered image, and normals.
    Also computes and reports PSNR and SSIM metrics.
    
    Args:
        stride: Render every Nth test image (default: 25)
    """
    print(f"\n[FINAL] Rendering test images with stride {stride}...")
    
    # Create output directory
    final_output_dir = os.path.join(dataset.model_path, 'final_test_renders')
    os.makedirs(final_output_dir, exist_ok=True)
    
    test_cameras = dataset.getTestCameras()
    
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
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            
            # Save concatenated
            concat_name = os.path.join(final_output_dir, f"{cam_name}_concat.png")
            save_img_u8(concat_image, concat_name)
            
            # Also save individual components
            gt_name = os.path.join(final_output_dir, f"{cam_name}_gt.png")
            save_img_u8(gt_np, gt_name)
            
            render_name = os.path.join(final_output_dir, f"{cam_name}_render.png")
            save_img_u8(rendered_np, render_name)
            
            normal_name = os.path.join(final_output_dir, f"{cam_name}_normal.png")
            save_img_u8(normal_np, normal_name)
            
            print(f"[FINAL] View {idx}/{len(test_cameras)} ({cam_name}): PSNR={psnr_value:.2f} SSIM={ssim_value:.4f}")
    
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
        f.write(f"{'Image':<30} {'PSNR':>10} {'SSIM':>10} {'L1':>12}\n")
        f.write(f"{'-'*64}\n")
        
        for i, (idx, viewpoint) in enumerate((idx, cam) for idx, cam in enumerate(test_cameras) if idx % stride == 0):
            cam_name = viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"view_{idx:03d}"
            f.write(f"{cam_name:<30} {psnr_values[i]:>10.2f} {ssim_values[i]:>10.4f} {l1_values[i]:>12.6f}\n")
    
    print(f"[FINAL] Metrics saved to: {metrics_file}")

def prepare_output_and_logger(args, scene_name, yaml_file = ""):    
    if not args.model_path:
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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--scene_name", type=str, default = None)
    parser.add_argument("--mesh_file", type=str, default = '/xxx/nerf_syn/mesh/')
    
    parser.add_argument("--gaussian_init", action="store_true")
    parser.add_argument("--time_analysis", action="store_true")
    parser.add_argument("--ingp", action="store_true")
    parser.add_argument("--yaml", type=str, default = "tiny")
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "surface"],
                        help="Rendering method: 'baseline' (default NeST) or 'surface' (surface potential)")
    parser.add_argument("--test_render_stride", type=int, default=25,
                        help="Stride for final test rendering (render every Nth test image, default: 25)")

    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    print(f"Method: {args.method.upper()}")

    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model, sys.argv[1:])

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