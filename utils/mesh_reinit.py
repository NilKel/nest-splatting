"""TSDF-fused mesh reinitialization for --minispa_mesh.

Integrates all training-view max-contributor depth maps into a TSDF volume
via Open3D, extracts a triangle mesh, and samples points uniformly over it
(area-weighted). This replaces per-view pixel sampling in the minispa reinit
step, removing the bias toward scene regions with more camera coverage.
"""
import numpy as np
import torch


def tsdf_mesh_reinit(scene, render_fn, target_count,
                     voxel_size=None, sdf_trunc_factor=4.0,
                     alpha_mask_thresh=0.5, depth_trunc=1e6,
                     use_poisson_disk=False):
    """Fuse per-view depth + color, extract mesh, sample area-uniformly.

    Args:
        scene: Scene object with getTrainCameras() and cameras_extent
        render_fn: callable(cam) -> dict with 'render', 'rend_alpha',
                   'depth_max_contributor' (or 'depth_median' fallback)
        target_count: number of points to sample from the mesh
        voxel_size: TSDF voxel length in world units. Defaults to
                    scene.cameras_extent / 256
        sdf_trunc_factor: truncation = voxel_size * this
        alpha_mask_thresh: pixels with rend_alpha below this are treated
                           as background and zeroed in the depth map
        depth_trunc: maximum integrated depth (Open3D units)
        use_poisson_disk: if True, use blue-noise Poisson-disk sampling
                          (~2x slower, more uniform spacing)

    Returns:
        dict {xyz, colors, normals} with tensors on CUDA, or None if the
        fused mesh is empty.
    """
    import open3d as o3d

    views = scene.getTrainCameras()
    if len(views) == 0:
        return None

    if voxel_size is None:
        voxel_size = float(scene.cameras_extent) / 256.0
    sdf_trunc = sdf_trunc_factor * voxel_size

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for v in views:
        with torch.no_grad():
            pkg = render_fn(v)
            rgb = pkg['render'].clamp(0.0, 1.0)                    # [3, H, W]
            alpha = pkg['rend_alpha']                               # [1, H, W]
            depth = pkg.get('depth_max_contributor', None)
            if depth is None or depth.numel() == 0:
                depth = pkg.get('depth_median', None)
            if depth is None:
                continue

        H, W = v.image_height, v.image_width
        color_np = (rgb.permute(1, 2, 0).contiguous().cpu().numpy() * 255.0
                    ).astype(np.uint8)
        depth_np = depth.squeeze().detach().cpu().numpy().astype(np.float32)
        alpha_np = alpha.squeeze().detach().cpu().numpy()
        depth_np[alpha_np < alpha_mask_thresh] = 0.0
        depth_np[depth_np < 0] = 0.0

        if hasattr(v, 'focal_x'):
            fx, fy = float(v.focal_x), float(v.focal_y)
        else:
            fx = W / (2.0 * np.tan(v.FoVx / 2.0))
            fy = H / (2.0 * np.tan(v.FoVy / 2.0))
        cx, cy = W / 2.0, H / 2.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_np)),
            o3d.geometry.Image(np.ascontiguousarray(depth_np)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)

        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        # world_view_transform is stored transposed (for column-major CUDA);
        # .T gives the standard row-major world->camera extrinsic Open3D wants.
        extr = v.world_view_transform.T.detach().cpu().numpy().astype(np.float64)

        vol.integrate(rgbd, intr, extr)

    mesh = vol.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None
    mesh.compute_vertex_normals()

    n = max(1, int(target_count))
    if use_poisson_disk:
        pcd = mesh.sample_points_poisson_disk(number_of_points=n,
                                              use_triangle_normal=True)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n,
                                           use_triangle_normal=True)

    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32)
    if pts.shape[0] == 0:
        return None

    return {
        'xyz':     torch.from_numpy(pts).cuda(),
        'normals': torch.from_numpy(nrm).cuda(),
        'colors':  torch.from_numpy(col).cuda(),
    }
