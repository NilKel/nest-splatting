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
    beta = 0, iteration = None, cfg = None, record_transmittance = False, use_xyz_mode = False):
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
    
    # Diffuse mode: use 0-degree SH (just DC component = diffuse RGB), no hashgrid
    is_diffuse_mode = ingp is not None and hasattr(ingp, 'is_diffuse_mode') and ingp.is_diffuse_mode
    # Specular mode: full 2DGS with SH (view-dependent), no hashgrid
    is_specular_mode = ingp is not None and hasattr(ingp, 'is_specular_mode') and ingp.is_specular_mode
    # Diffuse_ngp mode: diffuse SH + hashgrid on unprojected depth
    is_diffuse_ngp_mode = ingp is not None and hasattr(ingp, 'is_diffuse_ngp_mode') and ingp.is_diffuse_ngp_mode
    # Diffuse_offset mode: diffuse SH as xyz offset for hashgrid query
    is_diffuse_offset_mode = ingp is not None and hasattr(ingp, 'is_diffuse_offset_mode') and ingp.is_diffuse_offset_mode
    
    hash_in_CUDA = True
    try:
        if ingp is None:
            hash_in_CUDA = False
        if iteration < cfg.ingp_stage.switch_iter:
            hash_in_CUDA = False
        # Diffuse/Specular mode: never use hash_in_CUDA (no hashgrid)
        # Diffuse_ngp/diffuse_offset: also don't use hash_in_CUDA (we query hashgrid in Python on unprojected depth)
        if is_diffuse_mode or is_specular_mode or is_diffuse_ngp_mode or is_diffuse_offset_mode:
            hash_in_CUDA = False
    except:
        pass
    

    if ingp is not None and hash_in_CUDA == False and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode:
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

    # Cat mode detection and setup
    is_cat_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_cat_mode') and ingp.is_cat_mode
    hybrid_levels = ingp.hybrid_levels if is_cat_mode else 0
    
    # Adaptive mode detection
    is_adaptive_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_mode') and ingp.is_adaptive_mode
    
    render_mode = 0  # 0 = baseline, 4 = cat, 6 = adaptive

    if hash_in_CUDA:
        # Check if hashgrid is disabled (cat mode with hybrid_levels == total_levels)
        if hasattr(ingp, 'hashgrid_disabled') and ingp.hashgrid_disabled:
            # No hashgrid - pure per-Gaussian mode
            features = torch.zeros((1, ingp.level_dim), device="cuda")
            offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
            gridrange = ingp.gridrange
            per_level_scale = 1
            base_resolution = 0
            align_corners = False
            interpolation = 0
            # Encode: total_levels in upper bits, 0 hashgrid levels, hybrid_levels in lower bits
            levels = (ingp.levels << 16) | (0 << 8) | ingp.hybrid_levels
        else:
            # Normal hashgrid mode (baseline or cat with hashgrid)
            features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
                = ingp.hash_encoding.get_params()
            gridrange = ingp.gridrange
            levels = ingp.active_levels
        
        homotrans = pc.get_homotrans()
        ap_level = pc.get_appearance_level
        contract = ingp.contract
        
        # Cat mode: only activate if hybrid_levels > 0
        # When hybrid_levels == 0, behave identically to baseline
        if is_cat_mode and hybrid_levels > 0:
            # Set per-Gaussian features as colors_precomp
            colors_precomp = pc.get_gaussian_features
            shs = None
            render_mode = 4
            
            # Encode levels for CUDA case 4: (total << 16) | (active_hashgrid << 8) | hybrid
            # Uses active_hashgrid_levels for C2F (progressively enables hashgrid levels)
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
            
            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            # This is needed because CUDA copies offsets based on max possible levels
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets
        
        # Adaptive mode: blend per-Gaussian and hashgrid features in Python, pass 24D to rasterizer
        elif is_adaptive_mode and pc._adaptive_feat_dim > 0:
            # Query hashgrid features for all Gaussians (24D)
            xyz = pc.get_xyz  # (N, 3)
            hash_features = ingp._encode_3D(xyz)  # (N, 24)
            
            # Get per-Gaussian features (24D)
            adaptive_features = pc.get_adaptive_features  # (N, 24)
            
            # Compute soft mask from gamma (enables autograd for gamma)
            mask = pc.get_adaptive_mask(ingp.level_dim)  # (N, 24)
            
            # Blend in Python: blended = mask * adaptive + (1-mask) * hash
            blended_features = mask * adaptive_features + (1.0 - mask) * hash_features  # (N, 24)
            
            # Pass blended 24D features to rasterizer (same as baseline/cat mode)
            colors_precomp = blended_features
            shs = None
            render_mode = 0  # Standard rendering with precomputed colors
            
            # Disable hashgrid query in CUDA since we already queried in Python
            features = torch.zeros((1, ingp.level_dim), device="cuda")
            offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
            levels = 0
    
    # For diffuse/diffuse_ngp/diffuse_offset mode, use sh_degree=0 (only DC component)
    # For specular mode, use full active_sh_degree
    sh_degree_to_use = 0 if (is_diffuse_mode or is_diffuse_ngp_mode or is_diffuse_offset_mode) else pc.active_sh_degree
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree_to_use,
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
        render_mode = render_mode,
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

    # Diffuse_ngp mode: unproject median depth, query hashgrid, add to diffuse RGB
    gaussian_rgb_diffuse_ngp = None
    ngp_rgb_diffuse_ngp = None
    if is_diffuse_ngp_mode and ingp is not None:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        
        # Store Gaussian RGB (diffuse SH) before adding NGP contribution
        render_mask = (render_alpha > 0)
        gaussian_rgb_diffuse_ngp = rendered_image.clone()
        
        # Unproject median depth to 3D points (detached - no gradient through depth)
        points_3d, rays_d, rays_o = depths_to_points(viewpoint_camera, render_depth_median.detach())
        # points_3d: (H*W, 3), rays_d: (H*W, 3)
        
        # Query hashgrid at unprojected 3D points
        hash_features = ingp(points_3D=points_3d, with_xyz=False).float()  # (H*W, feat_dim)
        
        # Get view direction for MLP
        ray_unit = torch_F.normalize(rays_d, dim=-1).float()
        
        # Decode through MLP to get view-dependent RGB
        ngp_rgb = ingp.rgb_decode(hash_features, ray_unit)  # (H*W, 3)
        ngp_rgb = ngp_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        
        # Apply alpha mask
        ngp_rgb = ngp_rgb * render_mask
        
        # Store NGP RGB separately
        ngp_rgb_diffuse_ngp = ngp_rgb.clone()
        
        # Add NGP contribution to diffuse SH RGB
        rendered_image = rendered_image + ngp_rgb
    
    # Diffuse_offset mode: use rendered diffuse SH as xyz offset, query hashgrid at offset position
    # Implements "Scout and Squad" strategy for clean gradient flow
    scout_loss_data = None
    if is_diffuse_offset_mode and ingp is not None:
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        render_mask = (render_alpha > 0)
        
        # rendered_image contains the diffuse SH output (3, H, W) - this is the offset (Delta_P)
        # SH to RGB conversion is: rgb = sh * 0.28209 + 0.5, so we need to subtract 0.5 to center at 0
        # Reshape to (H*W, 3) for adding to xyz
        offset_3d_raw = rendered_image.permute(1, 2, 0).reshape(-1, 3) - 0.5  # (H*W, 3), centered at 0
        # Clamp offset to [-0.1, 0.1] to prevent large displacements
        offset_3d = torch.clamp(offset_3d_raw, -0.1, 0.1)
        
        # Store the offset for visualization (unclamped, but centered)
        gaussian_rgb_diffuse_ngp = rendered_image.clone() - 0.5
        
        if use_xyz_mode:
            # XYZ MODE: Use rasterized xyz directly from allmap[8:11]
            
            # P_base: rasterized Gaussian positions (alpha-blended)
            render_xyz = allmap[8:11]  # (3, H, W)
            
            # For RGB loss: detach P_base so gradients only flow through offset
            points_3d_base_detached = render_xyz.permute(1, 2, 0).reshape(-1, 3).detach()  # (H*W, 3)
            
            # P_query = P_base.detach() + Delta_P (offset has grads for RGB loss)
            points_3d_query = points_3d_base_detached + offset_3d  # (H*W, 3)
            
            # For scout loss: DON'T detach P_base - we need gradients to move Gaussians
            # scout_loss = MSE(P_base, P_target) where P_target = (P_base + offset).detach()
            # This pulls Gaussians toward where the offset found the surface
            points_3d_base_with_grad = render_xyz.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3), has grads
            points_3d_target = points_3d_query.detach()  # (H*W, 3), detached
            
            scout_loss_data = {
                'points_base': points_3d_base_with_grad,  # Has gradients for Gaussian positions
                'points_target': points_3d_target,  # Detached target
                'render_mask': render_mask,  # Only compute loss where alpha > 0
            }
            
            # Get view direction from camera
            rays_d, rays_o = cam2rays(viewpoint_camera)
        else:
            # DEPTH MODE: Unproject median depth to 3D points
            points_3d, rays_d, rays_o = depths_to_points(viewpoint_camera, render_depth_median.detach())
            # points_3d: (H*W, 3), rays_d: (H*W, 3)
            
            # Add offset to unprojected xyz (offset has gradients, points_3d is detached)
            points_3d_query = points_3d.detach() + offset_3d  # (H*W, 3)
        
        # Query hashgrid at query points
        hash_features = ingp(points_3D=points_3d_query, with_xyz=False).float()  # (H*W, feat_dim)
        
        # Get view direction for MLP
        ray_unit = torch_F.normalize(rays_d, dim=-1).float()
        
        # Decode through MLP to get final RGB
        final_rgb = ingp.rgb_decode(hash_features, ray_unit)  # (H*W, 3)
        final_rgb = final_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        
        # Apply alpha mask
        final_rgb = final_rgb * render_mask
        
        # Store NGP RGB separately (this is the final output in diffuse_offset mode)
        ngp_rgb_diffuse_ngp = final_rgb.clone()
        
        # Final RGB is purely from hashgrid MLP (not additive like diffuse_ngp)
        rendered_image = final_rgb
    
    # Diffuse/Specular mode: no MLP decoding needed, rendered_image is already RGB from SH
    # Baseline mode: decode hashgrid features through MLP
    elif ingp is not None and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode:
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
            'depth_expected': render_depth_expected,
            'depth_median': render_depth_median,
    })
    
    # Add diffuse_ngp mode separate RGB outputs
    if gaussian_rgb_diffuse_ngp is not None:
        rets['gaussian_rgb'] = gaussian_rgb_diffuse_ngp
    if ngp_rgb_diffuse_ngp is not None:
        rets['ngp_rgb'] = ngp_rgb_diffuse_ngp
    
    # Add scout loss data for diffuse_offset xyz mode
    if scout_loss_data is not None:
        rets['scout_loss_data'] = scout_loss_data
    
    # Add adaptive mask for regularization loss (if in adaptive mode)
    if is_adaptive_mode and pc._adaptive_feat_dim > 0:
        rets['adaptive_mask'] = mask  # (N, feat_dim)
    
    return rets
