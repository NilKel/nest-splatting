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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8
from lpipsPyTorch import lpips
from train import merge_cfg_to_args

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--yaml", type=str, default = "tiny")
    args = get_combined_args(parser)

    exp_path = args.model_path

    iteration = args.iteration
    yaml_file = args.yaml

    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    # Default to baseline for eval (can add --method arg if needed)
    ingp_model = INGP(cfg_model, method='baseline').to('cuda')
    ingp_model.load_model(exp_path, iteration)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    resolution = dataset.resolution
    print('test resolution: ', resolution)
    
    args.cfg = cfg_model

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    # print(f'base_a {gaussians.base_opacity}, beta {beta}')

    gaussians.XYZ_TYPE = "UV"
    active_levels = ingp_model.set_active_levels(iteration)


    train_dir = os.path.join(scene.model_path, 'train', "ours_{}".format(iteration))
    test_dir = os.path.join(scene.model_path, 'test', "ours_{}".format(iteration))

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok = True)
        train_renders = os.path.join(train_dir, "renders")
        train_gt = os.path.join(train_dir, "gt")    
        os.makedirs(train_renders, exist_ok = True)
        os.makedirs(train_gt, exist_ok = True)
        
        viewpoint_stack = scene.getTrainCameras().copy()
        with torch.no_grad():
            for cam in tqdm(viewpoint_stack):
                cam_name = cam.image_name + '.png'
                render_pkg = render(cam, gaussians, pipe, background, ingp = ingp_model, \
                    beta = beta, iteration = iteration, cfg = cfg_model)
                image = render_pkg["render"]
                gt = cam.original_image

                img_name = os.path.join(test_renders, cam_name)
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), img_name)
                img_name = os.path.join(test_gt, cam_name)
                save_img_u8(gt.permute(1,2,0).detach().cpu().numpy(), img_name)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok = True)
        test_renders = os.path.join(test_dir, "renders")
        test_gt = os.path.join(test_dir, "gt")    
        os.makedirs(test_renders, exist_ok = True)
        os.makedirs(test_gt, exist_ok = True)
        
        viewpoint_stack = scene.getTestCameras().copy()
        with torch.no_grad():
            for cam in tqdm(viewpoint_stack):
                cam_name = cam.image_name + '.png'
                render_pkg = render(cam, gaussians, pipe, background, ingp = ingp_model, \
                    beta = beta, iteration = iteration, cfg = cfg_model)
            
                image = render_pkg["render"]
                gt = cam.original_image

                img_name = os.path.join(test_renders, cam_name)
                save_img_u8(image.permute(1,2,0).detach().cpu().numpy(), img_name)
                img_name = os.path.join(test_gt, cam_name)
                save_img_u8(gt.permute(1,2,0).detach().cpu().numpy(), img_name)

    if not args.skip_mesh:
        gaussExtractor = GaussianExtractor(render, gaussians, pipe, background, ingp = ingp_model, \
                    beta = beta, iteration = iteration, cfg = cfg_model)
        
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        # gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
