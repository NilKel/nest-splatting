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

        # --decomp / --blurprog: pre-compute guided-filter low-freq images for
        # all train views (cached next to the source images). Runs once per
        # (r, eps); reuses on subsequent calls. Both flags share the same cache.
        _needs_decomp_cache = full_args is not None and (
            getattr(full_args, 'decomp', False) or getattr(full_args, 'blurprog', False))
        if _needs_decomp_cache:
            from utils.image_decomp import ensure_decomp_cache
            ensure_decomp_cache(
                scene_info.train_cameras,
                r=int(full_args.decomp_r),
                eps=float(full_args.decomp_eps),
            )

        # --random_init: replace the loaded SfM/PLY point cloud with N uniformly
        # random points inside a sphere. Bypasses COLMAP/points3d.ply entirely.
        # Sphere radius auto-sizes to 1.3×cameras_extent unless set explicitly so
        # points span a bit more than the camera hull.
        _rand_n = int(getattr(full_args, 'random_init', 0) or 0) if full_args is not None else 0
        if _rand_n > 0:
            import numpy as _np
            _extent_proxy = scene_info.nerf_normalization["radius"]
            _r = float(getattr(full_args, 'random_init_radius', 0.0) or 0.0)
            if _r <= 0.0:
                _r = 1.3 * float(_extent_proxy)
            # Uniform in a sphere: sample in cube, reject outside unit sphere, scale.
            _pts = _np.empty((0, 3), dtype=_np.float32)
            while _pts.shape[0] < _rand_n:
                _cand = _np.random.uniform(-1.0, 1.0, size=(_rand_n * 2, 3)).astype(_np.float32)
                _mask = (_cand ** 2).sum(axis=1) <= 1.0
                _pts = _np.concatenate([_pts, _cand[_mask]], axis=0)
            _pts = _pts[:_rand_n] * _r
            _colors = _np.random.uniform(0.0, 1.0, size=(_rand_n, 3)).astype(_np.float32)
            _normals = _np.zeros((_rand_n, 3), dtype=_np.float32)
            from utils.graphics_utils import BasicPointCloud as _BPC
            _rand_pcd = _BPC(points=_pts, colors=_colors, normals=_normals)
            scene_info = scene_info._replace(point_cloud=_rand_pcd)
            print(f"[RANDOM_INIT] Replaced SfM/PLY point cloud with {_rand_n} "
                  f"random points in sphere of radius {_r:.3f} "
                  f"(cameras_extent={_extent_proxy:.3f})")

        # FPS subsampling for mcmc_fps mode.
        #   num_init_points >  cap_max  → FPS down to cap_max (MCMC has no headroom).
        #   num_init_points <= cap_max  → FPS down to num_init_points // 2 so MCMC
        #                                 has density-adaptive headroom up to cap_max.
        # cap_max remains the growth ceiling either way (densification budget).
        if mcmc_fps:
            num_init_points = len(scene_info.point_cloud.points)
            if cap_max > 0 and num_init_points > cap_max:
                from utils.point_cloud_utils import load_or_create_fps_pointcloud
                print(f"[FPS] Initial point cloud: {num_init_points} points, cap_max: {cap_max}")
                subsampled_pcd = load_or_create_fps_pointcloud(
                    args.source_path, scene_info.point_cloud, cap_max
                )
                scene_info = scene_info._replace(point_cloud=subsampled_pcd)
                print(f"[FPS] Subsampled to {len(scene_info.point_cloud.points)} points "
                      f"(MCMC budget cap_max={cap_max})")
            elif cap_max > 0 and num_init_points < cap_max:
                from utils.point_cloud_utils import load_or_create_fps_pointcloud
                target = max(1, num_init_points // 2)
                print(f"[FPS] Initial point cloud: {num_init_points} points ≤ cap_max={cap_max} — "
                      f"FPS-halving to {target} so MCMC can grow back to cap_max.")
                subsampled_pcd = load_or_create_fps_pointcloud(
                    args.source_path, scene_info.point_cloud, target
                )
                scene_info = scene_info._replace(point_cloud=subsampled_pcd)
                print(f"[FPS] Subsampled to {len(scene_info.point_cloud.points)} points "
                      f"(MCMC budget cap_max={cap_max})")
            else:
                print(f"[FPS] Skipping FPS - cap_max not set; using all {num_init_points} init points")

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

        # Cache the per-camera CameraInfo objects (image_path, R, T, fov...) so
        # train.py can mid-train reload at a different `--resolution`/`--data_device`
        # (progressive res schedule). Cheap (small list of dataclasses, no image
        # data — original PIL files stay on disk).
        self._cam_infos_train = list(scene_info.train_cameras)
        self._cam_infos_test = list(scene_info.test_cameras)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # --decomp / --blurprog: attach cached low-freq GT to each train camera as cam.gt_low.
            if _needs_decomp_cache:
                from utils.image_decomp import load_gt_low_for_cameras
                load_gt_low_for_cameras(
                    self.train_cameras[resolution_scale],
                    r=int(full_args.decomp_r),
                    eps=float(full_args.decomp_eps),
                )
        
        if self.loaded_iter:
            # Use full_args if available (has kernel type, method, etc.), otherwise use ModelParams args
            ply_args = self._full_args if self._full_args is not None else args
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args = ply_args)
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