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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], mcmc_fps=False, cap_max=-1, full_args=None):
        """b
        :param path: Path to colmap scene main folder.
        :param full_args: Full training args (for kernel type, method, etc.)
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self._full_args = full_args  # Store full args for create_from_pcd

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # FPS subsampling for mcmc_fps mode
        if mcmc_fps and cap_max > 0:
            from utils.point_cloud_utils import load_or_create_fps_pointcloud

            num_init_points = len(scene_info.point_cloud.points)
            print(f"[FPS] Initial point cloud: {num_init_points} points, cap_max: {cap_max}")

            if num_init_points > cap_max:
                subsampled_pcd = load_or_create_fps_pointcloud(
                    args.source_path, scene_info.point_cloud, cap_max
                )
                scene_info = scene_info._replace(point_cloud=subsampled_pcd)
                print(f"[FPS] Subsampled to {len(scene_info.point_cloud.points)} points")
            else:
                print(f"[FPS] Skipping - init points ({num_init_points}) <= cap_max ({cap_max})")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f'camera extent {self.cameras_extent}')
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args = args)
        elif hasattr(self.gaussians, '_loaded_from_checkpoint') and self.gaussians._loaded_from_checkpoint:
            # Gaussians already loaded from warmup checkpoint, skip create_from_pcd
            print(f"[Scene] Using pre-loaded Gaussians ({len(self.gaussians.get_xyz)} points)")
        else:
            # Use full_args if available (has kernel type, method, etc.), otherwise use ModelParams args
            pcd_args = self._full_args if self._full_args is not None else args
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args = pcd_args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]