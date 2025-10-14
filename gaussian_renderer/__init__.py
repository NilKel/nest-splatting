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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal, save_points, depths_to_points, cam2rays
import torch.nn.functional as torch_F
import time
# from hash_encoder.FeatureBlend import FeatureBlend
from utils.general_utils import MEM_PRINT

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ingp = None,
    beta = 0, iteration = None, cfg = None, record_transmittance = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    render_start_time = time.time()

    XYZ_TYPE = cfg.ingp_stage.XYZ_TYPE
    assert(XYZ_TYPE == "UV" or XYZ_TYPE == "DEPTH")
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    
    hash_in_CUDA = True
    try:
        if ingp is None:
            hash_in_CUDA = False
        if iteration < cfg.ingp_stage.switch_iter:
            hash_in_CUDA = False
    except:
        pass
    

    if ingp is not None and hash_in_CUDA == False:
        ### warm-up
        override_color = ingp(points_3D = means3D, with_xyz = False).float()
        feat_dim = ingp.active_levels * ingp.level_dim
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    homotrans = None
    features = None
    offsets = None
    gridrange = None 
    levels = base_resolution = interpolation = 0
    per_level_scale = 1
    align_corners = False
    ap_level = None
    contract = False

    if hash_in_CUDA:
        features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
            = ingp.hash_encoding.get_params()
        gridrange = ingp.gridrange

        levels = ingp.active_levels
        
        homotrans = pc.get_homotrans()
        ### ap level
        ap_level = pc.get_appearance_level

        contract = ingp.contract
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        beta=beta,
        if_contract = contract,
        record_transmittance = record_transmittance,
        # pipe.debug
    )

    hashgrid_settings = HashGridSettings(
        L = levels,
        S = math.log2(per_level_scale), 
        H = base_resolution,
        align_corners = align_corners,
        interpolation = interpolation,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    
    rendered_image, radii, allmap, transmittance_avg, num_covered_pixels = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        homotrans = homotrans,
        ap_level = ap_level,
        cov3D_precomp = cov3D_precomp,
        features = features,
        offsets = offsets,
        gridrange = gridrange,
    )
    
    
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    # surf_normal = render_normal

    # get contributed gaussians per pixel 
    render_gs_nums = allmap[7:8]

    if ingp is not None:
        # if hash_in_CUDA:
        rays_d, rays_o = cam2rays(viewpoint_camera)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        ray_unit = torch_F.normalize(rays_d, dim=-1).float().detach()
        normals = render_normal.view(3, -1).permute(1, 0)
        normals_unit = torch_F.normalize(normals, dim=-1).float().detach()

        d_dot_n = torch.sum(ray_unit * normals_unit, dim=-1, keepdim=True)
        ray_out = ray_unit - 2.0 * d_dot_n * normals_unit
        
        ray_out_norm = torch.norm(ray_out, dim = -1)

        feat_dim = rendered_image.shape[0]

        render_mask = (render_alpha > 0)

        rays_dir = ray_unit
        try:
            if cfg.settings.dir_out:
                rays_dir = ray_out
        except:
            pass

        ray_map = rays_dir.view(H, W, -1).permute(2, 0, 1).abs()
        ray_map = ray_map * render_mask

        feature_vis = rendered_image[:3].detach().abs()
        
        rendered_image = ingp.rgb_decode(rendered_image.view(feat_dim, -1).permute(1, 0), rays_dir)

        rendered_image = rendered_image.view(H, W, -1).permute(2, 0, 1)
        rendered_image = rendered_image * render_mask

        vis_appearance_level = allmap[11:14]


    if bg_color.sum() > 0:
        rendered_image = rendered_image + (1.0 - render_alpha) * bg_color.unsqueeze(-1).unsqueeze(-1)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    if record_transmittance:
        rets.update({
            'transmittance_avg': transmittance_avg,
            'cover_pixels': num_covered_pixels,
        })

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'gaussian_num' : render_gs_nums,
    })
    
    return rets
    