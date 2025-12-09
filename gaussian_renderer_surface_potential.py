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
from utils.point_utils import depth_to_normal, depth_to_gradient_unnormalized, save_points, depths_to_points, cam2rays
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
            # EXCEPTION: hybrid_features always uses hash_in_CUDA (unless hybrid_levels=total_levels)
            if ingp is not None and ingp.method == 'hybrid_features':
                if ingp.hybrid_levels != ingp.total_levels:
                    # Use hash_in_CUDA for hybrid_levels 0 to total_levels-1
                    hash_in_CUDA = True
                # hybrid_levels=total_levels uses 2DGS mode, doesn't need hashgrid
        # Surface mode now uses hash_in_CUDA for ray-Gaussian intersection!
        # The single hashgrid with 3x features is queried in CUDA
    except:
        pass
    

    if ingp is not None and hash_in_CUDA == False:
        ### warm-up only (surface and surface_rgb modes now use hash_in_CUDA after warmup)
        if ingp.method == 'surface_rgb':
            # Query at Gaussian centers to get both vector potentials and RGB
            full_output = ingp(points_3D = means3D, with_xyz = False).float()  # (N_gaussians, 15*levels)
            # For surface_rgb: 15 features per level = 12 vector potentials + 3 RGB
            # We'll pass all features and split them in the warmup rendering path
            override_color = full_output
            feat_dim = ingp.active_levels * ingp.level_dim  # 15 per level
        else:
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

    # For hybrid_features mode: handle special cases
    # hybrid_levels=0: Use baseline mode (pure hashgrid, NO per-Gaussian features)
    # hybrid_levels=total_levels: Use regular 2DGS mode (pure per-Gaussian features, NO hashgrid)
    # Otherwise: Use TRUE hybrid mode (render_mode=4, both representations)
    feat_dim = 0  # Initialize feat_dim
    if ingp is not None and ingp.method == 'hybrid_features':
        if ingp.hybrid_levels == 0:
            # Pure hashgrid mode: identical to baseline
            # No colors_precomp (will use hashgrid only)
            # render_mode will be set to 0 (baseline), uses hash_in_CUDA
            feat_dim = ingp.total_levels * ingp.level_dim
        elif ingp.hybrid_levels == ingp.total_levels:
            # Pure per-Gaussian mode: identical to 2DGS
            # No hashgrid querying (will use per-Gaussian features only)
            # DON'T use hash_in_CUDA path to avoid hashgrid gradient issues
            hash_in_CUDA = False  # Force standard 2DGS path
            if pc.get_gaussian_features is not None:
                colors_precomp = pc.get_gaussian_features
            shs = None
            feat_dim = ingp.total_levels * ingp.level_dim
        else:
            # TRUE hybrid mode: BOTH per-Gaussian AND hashgrid
            # render_mode will be set to 6 (hybrid)
            if pc.get_gaussian_features is not None:
                colors_precomp = pc.get_gaussian_features
            shs = None
            feat_dim = ingp.total_levels * ingp.level_dim

    homotrans = None
    features = None
    offsets = None
    gridrange = None 
    levels = base_resolution = interpolation = 0
    per_level_scale = 1
    align_corners = False
    ap_level = None
    contract = False
    
    # For surface_rgb mode: separate hashgrids
    features_diffuse = None
    offsets_diffuse = None
    gridrange_diffuse = None
    features_view = None
    offsets_view = None
    gridrange_view = None

    if hash_in_CUDA:
        if ingp.method == 'surface_rgb':
            # Dual hashgrid for surface_rgb
            features_diffuse, offsets_diffuse, levels_diffuse, per_level_scale_diffuse, base_resolution_diffuse, align_corners_diffuse, interpolation_diffuse \
                = ingp.hash_encoding_diffuse.get_params()
            gridrange_diffuse = ingp.gridrange_diffuse
            
            features, offsets, levels_view, per_level_scale, base_resolution, align_corners, interpolation \
                = ingp.hash_encoding_view_features.get_params()
            gridrange = ingp.gridrange_view
            levels = ingp.active_levels
        elif ingp.method == 'baseline_double':
            # Dual 4D hashgrid for baseline_double
            # Hashgrid 1: queried at xyz (main features)
            features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
                = ingp.hash_encoding.get_params()
            gridrange = ingp.gridrange
            
            # Hashgrid 2: queried at pk (per-Gaussian features)
            features_diffuse, offsets_diffuse, levels_diffuse, per_level_scale_diffuse, base_resolution_diffuse, align_corners_diffuse, interpolation_diffuse \
                = ingp.hash_encoding_diffuse.get_params()
            gridrange_diffuse = ingp.gridrange_diffuse
            levels = ingp.active_levels
        elif ingp.method == 'baseline_blend_double':
            # Dual 4D hashgrid for baseline_blend_double
            # Hashgrid 1: queried at blended position (spatial features)
            features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
                = ingp.hash_encoding.get_params()
            gridrange = ingp.gridrange

            # Hashgrid 2: queried at pk (per-Gaussian features, alpha blended)
            features_diffuse, offsets_diffuse, levels_diffuse, per_level_scale_diffuse, base_resolution_diffuse, align_corners_diffuse, interpolation_diffuse \
                = ingp.hash_encoding_diffuse.get_params()
            gridrange_diffuse = ingp.gridrange_diffuse
            levels = ingp.active_levels
        elif ingp.method == 'hybrid_features':
            # Handle special cases for hybrid_features
            if ingp.hybrid_levels == 0:
                # hybrid_levels=0: Pure baseline mode (only hashgrid, no per-Gaussian features)
                # Use standard baseline rendering path
                features, offsets, levels, per_level_scale, base_resolution, align_corners, interpolation \
                    = ingp.hash_encoding.get_params()
                gridrange = ingp.gridrange
                levels = ingp.active_levels
                # render_mode will be determined by other conditions, not forced to 6
            elif ingp.hybrid_levels == ingp.total_levels:
                # hybrid_levels=total_levels: Pure per-Gaussian mode (no hashgrid)
                # Using standard 2DGS path (hash_in_CUDA=False), so no hashgrid params needed
                pass  # No hashgrid setup required
            else:
                # True hybrid mode: combine per-Gaussian + hashgrid
                # Python provides hybrid_levels×D per-Gaussian features via colors_precomp
                # CUDA queries hashgrid (active_levels levels) and concatenates
                features, offsets, _, per_level_scale, base_resolution, align_corners, interpolation \
                    = ingp.hash_encoding.get_params()
                gridrange = ingp.gridrange

                # Encode level parameter for hybrid_features mode
                # Format: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
                # CUDA will extract these to determine per-Gaussian feature dimension and hashgrid query
                total_levels = int(ingp.total_levels)
                active_hashgrid_levels = int(ingp.active_levels)
                hybrid_levels = int(ingp.hybrid_levels)
                levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
        else:
            # Single hashgrid for baseline/surface modes
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
    
    # Determine rendering mode for CUDA rasterizer
    # Render modes: 0=baseline, 1=surface, 2=baseline_double, 3=baseline_blend_double, 4=hybrid_features, 5=surface_rgb
    # surface_blend and surface_depth use baseline rasterization (render_mode=0) and do dot product in Python
    if ingp is not None:
        if ingp.method == 'surface':
            render_mode = 1  # Surface mode: dot product in CUDA per-Gaussian before blending
        elif ingp.method in ['surface_blend', 'surface_depth']:
            render_mode = 0   # Baseline mode: blend 12D features, then dot product in Python
        elif ingp.method == 'surface_rgb':
            # Check if dual hashgrid mode (NEW) or single hashgrid (OLD)
            if hasattr(ingp, 'hash_encoding_diffuse'):
                render_mode = 1  # NEW: Dual hashgrid - use case 1 (surface + baseline)
            else:
                render_mode = 5  # OLD: Single hashgrid - use case 5 (vectors + RGB)
        elif ingp.method == 'baseline_double':
            render_mode = 2   # baseline_double mode: dual 4D hashgrids
        elif ingp.method == 'baseline_blend_double':
            render_mode = 3   # baseline_blend_double mode: dual 4D hashgrids with blended spatial query
        elif ingp.method == 'hybrid_features':
            # Special cases for hybrid_features
            if ingp.hybrid_levels == 0:
                # Pure baseline mode: use render_mode 0
                render_mode = 0
            elif ingp.hybrid_levels == ingp.total_levels:
                # Pure per-Gaussian mode: use standard 2DGS (render_mode=0)
                render_mode = 0
            else:
                # True hybrid mode: combine per-Gaussian + hashgrid
                render_mode = 4
        else:
            render_mode = 0   # Baseline mode
    else:
        render_mode = 0  # Baseline mode
    
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
        # For surface_rgb dual hashgrid mode: baseline features (4D per level)
        features_diffuse = features_diffuse,
        offsets_diffuse = offsets_diffuse,
        gridrange_diffuse = gridrange_diffuse,
        render_mode = render_mode,
    )
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # Transform normals from view space to world space
    render_normal = allmap[2:5]
    
    render_normal = render_normal.contiguous()
    world_view_rot = viewpoint_camera.world_view_transform[:3,:3].T.contiguous()
    H, W = render_normal.shape[1], render_normal.shape[2]
    render_normal_hwc = render_normal.permute(1,2,0).reshape(-1, 3)
    render_normal_transformed = torch.matmul(render_normal_hwc, world_view_rot)
    render_normal = render_normal_transformed.reshape(H, W, 3).permute(2,0,1)
    
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
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    # surf_normal = render_normal

    # get contributed gaussians per pixel 
    render_gs_nums = allmap[7:8]
    
    if ingp is not None:
        # Compute proper camera rays for view-dependent rendering
        rays_d, rays_o = cam2rays(viewpoint_camera)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        
        # rays_d shape: (H*W, 3), rays_o: (3,)
        rays_d = rays_d.reshape(H, W, 3)
        ray_unit = torch_F.normalize(rays_d, dim=-1).float().detach()
        normals = render_normal.view(3, -1).permute(1, 0)
        normals_unit = torch_F.normalize(normals, dim=-1).float().detach()

        # Reshape ray_unit to match normals_unit shape
        ray_unit_flat = ray_unit.view(-1, 3)
        d_dot_n = torch.sum(ray_unit_flat * normals_unit, dim=-1, keepdim=True)
        ray_out = ray_unit_flat - 2.0 * d_dot_n * normals_unit
        ray_out = ray_out.view(H, W, 3)
        
        ray_out_norm = torch.norm(ray_out, dim = -1)

        feat_dim = rendered_image.shape[0]

        render_mask = (render_alpha > 0)

        rays_dir = ray_unit
        try:
            if cfg.settings.dir_out:
                rays_dir = ray_out
        except:
            pass

        # Flatten rays_dir for MLP input (encoder expects (N, 3) not (H, W, 3))
        rays_dir_flat = rays_dir.view(-1, 3)
        
        ray_map = rays_dir.view(H, W, -1).permute(2, 0, 1).abs()
        ray_map = ray_map * render_mask

        feature_vis = rendered_image[:3].detach().abs()
        # Surface potential mode: dot product done in CUDA per-Gaussian, just decode
        if ingp.method == 'surface':
            # if iteration is not None and iteration % 1000 == 0:
            #     print(f"[SURFACE] Iteration {iteration}: Decoding scalar features, feat_dim={feat_dim}")
            
            # rendered_image shape: (feat_dim, H, W) where feat_dim = 4 * levels (e.g., 24 for 6 levels)
            # Features are already scalar (dot product done in CUDA per-Gaussian before alpha-blending)
            surface_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, feat_dim)
            
            # Apply coarse-to-fine padding if needed
            if surface_features.shape[1] < ingp.feat_dim:
                padding = torch.zeros(surface_features.shape[0], ingp.feat_dim - surface_features.shape[1], 
                                     device=surface_features.device, dtype=surface_features.dtype)
                surface_features = torch.cat([surface_features, padding], dim=1)
            
            # Pass through MLP
            rendered_image = ingp.rgb_decode(surface_features, rays_dir_flat)
        elif ingp.method == 'surface_blend':
            # Surface blend mode: baseline rasterization blends 12D features, then dot product in Python
            # rendered_image shape: (feat_dim, H, W) where feat_dim = 12 * active_levels
            # render_normal shape: (3, H, W) - alpha-blended normals from rasterizer

            H, W = rendered_image.shape[1], rendered_image.shape[2]
            feat_dim = rendered_image.shape[0]  # e.g., 72 for 6 levels, 24 for 2 levels

            # Reshape blended features: (feat_dim, H, W) → (H*W, feat_dim/3, 3)
            # feat_dim/3 = levels*4 (e.g., 24 for 6 levels)
            # We have 4 vectors of 3D per level
            blended_vectors = rendered_image.view(feat_dim, H * W).permute(1, 0)  # (H*W, feat_dim)
            blended_vectors = blended_vectors.view(H * W, -1, 3)  # (H*W, feat_dim/3, 3) = (H*W, 24, 3)

            # Get blended normals: (3, H, W) → (H*W, 3)
            blended_normals = render_normal.permute(1, 2, 0).reshape(H * W, 3)  # (H*W, 3)
            # blended_normals = torch_F.normalize(blended_normals, dim=-1).float()  # Normalize

            # Dot product: (H*W, 24, 3) dot (H*W, 3) → (H*W, 24)
            # Each of the 24 vectors gets dotted with the normal
            # blended_vectors: (H*W, 24, 3), blended_normals: (H*W, 3, 1) → (H*W, 24, 1)
            dot_prod = torch.bmm(blended_vectors, blended_normals.unsqueeze(-1)).squeeze(-1)  # (H*W, 24)

            # Negate and apply ReLU
            surface_features = torch.relu(-dot_prod)  # (H*W, 24)

            # Apply coarse-to-fine padding if needed
            if surface_features.shape[1] < ingp.feat_dim:
                padding = torch.zeros(surface_features.shape[0], ingp.feat_dim - surface_features.shape[1],
                                     device=surface_features.device, dtype=surface_features.dtype)
                surface_features = torch.cat([surface_features, padding], dim=1)

            # Pass through MLP
            rendered_image = ingp.rgb_decode(surface_features, rays_dir_flat)

        elif ingp.method == 'surface_depth':
            # Surface depth mode: identical to surface_blend but uses depth gradient normals instead of rendered normals
            # rendered_image shape: (feat_dim, H, W) where feat_dim = 12 * active_levels
            # We'll compute normals from the depth map gradient

            H, W = rendered_image.shape[1], rendered_image.shape[2]
            feat_dim = rendered_image.shape[0]  # e.g., 72 for 6 levels, 24 for 2 levels

            # Reshape blended features: (feat_dim, H, W) → (H*W, feat_dim/3, 3)
            blended_vectors = rendered_image.view(feat_dim, H * W).permute(1, 0)  # (H*W, feat_dim)
            blended_vectors = blended_vectors.view(H * W, -1, 3)  # (H*W, feat_dim/3, 3) = (H*W, 24, 3)

            # Get depth gradient normals: (3, H, W) → (H*W, 3)
            # surf_normal is computed later from depth, so we need to pass it through
            # For now, compute it here from render_depth_expected
            render_depth_expected = allmap[0:1]
            render_depth_median = allmap[1:2]
            surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

            # Use normalized depth gradients
            depth_normals = depth_to_normal(viewpoint_camera, surf_depth)  # (H, W, 3)
            depth_normals = depth_normals.permute(2, 0, 1)  # (3, H, W)
            depth_normals = depth_normals.permute(1, 2, 0).reshape(H * W, 3)  # (H*W, 3)

            # Dot product: (H*W, 24, 3) dot (H*W, 3) → (H*W, 24)
            dot_prod = torch.bmm(blended_vectors, depth_normals.unsqueeze(-1)).squeeze(-1)  # (H*W, 24)

            # Negate and apply ReLU
            surface_features = torch.relu(-dot_prod)  # (H*W, 24)

            # Apply coarse-to-fine padding if needed
            if surface_features.shape[1] < ingp.feat_dim:
                padding = torch.zeros(surface_features.shape[0], ingp.feat_dim - surface_features.shape[1],
                                     device=surface_features.device, dtype=surface_features.dtype)
                surface_features = torch.cat([surface_features, padding], dim=1)

            # Pass through MLP
            rendered_image = ingp.rgb_decode(surface_features, rays_dir_flat)

        elif ingp.method == 'baseline_double':
            # DUAL 4D HASHGRID MODE:
            # - rendered_image contains combined features (feat_xyz + feat_pk)
            # - Already blended: 24D = 6 levels × 4 features
            # - Pass through MLP to get RGB
            
            # Features are already concatenated by CUDA kernel (4 per level × levels)
            combined_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, feat_dim)
            
            # Apply coarse-to-fine padding if needed
            if combined_features.shape[1] < ingp.feat_dim:
                padding = torch.zeros(combined_features.shape[0], ingp.feat_dim - combined_features.shape[1], 
                                     device=combined_features.device, dtype=combined_features.dtype)
                combined_features = torch.cat([combined_features, padding], dim=1)
            
            # Pass through MLP
            rendered_image = ingp.rgb_decode(combined_features, rays_dir_flat)  # (H*W, 3)
        
        elif ingp.method == 'baseline_blend_double':
            # DUAL 4D HASHGRID MODE WITH BLENDED SPATIAL QUERY:
            # - rendered_image contains combined features (blended feat_pk + feat_spatial)
            # - feat_pk was alpha blended, then feat_spatial added at blended position
            # - Already blended: 24D = 6 levels × 4 features
            # - Pass through MLP to get RGB
            
            # Features are already combined by CUDA kernel (4 per level × levels)
            combined_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, feat_dim)
            
            # Apply coarse-to-fine padding if needed
            if combined_features.shape[1] < ingp.feat_dim:
                padding = torch.zeros(combined_features.shape[0], ingp.feat_dim - combined_features.shape[1], 
                                     device=combined_features.device, dtype=combined_features.dtype)
                combined_features = torch.cat([combined_features, padding], dim=1)
            
            # Pass through MLP
            rendered_image = ingp.rgb_decode(combined_features, rays_dir_flat)  # (H*W, 3)
        
        elif ingp.method == 'surface_rgb':
            # Check if dual hashgrid mode by checking if we have baseline hashgrid
            has_baseline = hasattr(ingp, 'hash_encoding_diffuse')
            
            if has_baseline:
                # DUAL HASHGRID MODE:
                # - rendered_image contains combined features (baseline + surface)
                # - Already blended: 24D = 6 levels × 4 features
                # - Pass through MLP to get RGB
                
                # Features are already concatenated by CUDA kernel (4 per level × levels)
                combined_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, feat_dim)
                
                # Apply coarse-to-fine padding if needed
                if combined_features.shape[1] < ingp.feat_dim:
                    padding = torch.zeros(combined_features.shape[0], ingp.feat_dim - combined_features.shape[1], 
                                         device=combined_features.device, dtype=combined_features.dtype)
                    combined_features = torch.cat([combined_features, padding], dim=1)
                
                # Pass through MLP
                rendered_image = ingp.rgb_decode(combined_features, rays_dir_flat)  # (H*W, 3)
            else:
                # OLD SINGLE HASHGRID MODE (case 15):
                # rendered_image shape: (feat_dim, H, W) where feat_dim = 7 * levels (e.g., 42 for 6 levels)
                # Layout per level: [4 scalar features (from dot product), 3 RGB]
                levels = ingp.active_levels
                features_per_level = 7  # 4 scalars + 3 RGB
                scalar_per_level = 4
                rgb_per_level = 3
                
                # Split features
                scalar_features_list = []
                rgb_features_list = []
                
                for lv in range(levels):
                    start_idx = lv * features_per_level
                    scalar_start = start_idx
                    scalar_end = start_idx + scalar_per_level
                    rgb_start = scalar_end
                    rgb_end = rgb_start + rgb_per_level
                    
                    scalar_features_list.append(rendered_image[scalar_start:scalar_end])  # (4, H, W)
                    rgb_features_list.append(rendered_image[rgb_start:rgb_end])  # (3, H, W)
                
                # Concatenate scalar features and sum RGB
                scalar_features = torch.cat(scalar_features_list, dim=0)  # (4*levels, H, W)
                diffuse_rgb = torch.stack(rgb_features_list, dim=0).sum(dim=0)  # (3, H, W)
                
                # Prepare scalar features for MLP
                surface_features = scalar_features.view(scalar_features.shape[0], -1).permute(1, 0)  # (H*W, 4*levels)
                
                # Apply coarse-to-fine padding if needed
                if surface_features.shape[1] < ingp.feat_dim:
                    padding = torch.zeros(surface_features.shape[0], ingp.feat_dim - surface_features.shape[1], 
                                         device=surface_features.device, dtype=surface_features.dtype)
                    surface_features = torch.cat([surface_features, padding], dim=1)
                
                # Pass through MLP for specular component
                specular_rgb = ingp.rgb_decode(surface_features, rays_dir_flat)  # (H*W, 3)
                specular_rgb = specular_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
                
                # Combine diffuse + specular
                rendered_image = diffuse_rgb + specular_rgb
        elif ingp.method == 'hybrid_features':
            # Hybrid mode: 24D features (12D per-Gaussian + 12D hashgrid)
            # rendered_image shape: (24, H, W)
            # Simply pass to MLP decoder
            
            hybrid_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, 24)
            
            rendered_image = ingp.rgb_decode(hybrid_features, rays_dir_flat)
        else:
            # Baseline mode: direct MLP decode
            rendered_image = ingp.rgb_decode(rendered_image.view(feat_dim, -1).permute(1, 0), rays_dir_flat)

        rendered_image = rendered_image.reshape(H, W, -1).permute(2, 0, 1)
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
    