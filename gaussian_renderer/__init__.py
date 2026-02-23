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

# Import rasterizer - use lean library if available, fall back to main
# The lean library (diff_surfel_3D) is a stripped-down version for faster builds
import os
_USE_LEAN_ONLY = os.environ.get('USE_LEAN_RASTERIZER', '0') == '1'

try:
    import diff_surfel_3D as _lean_rasterizer
    from diff_surfel_3D import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
    LEAN_RASTERIZER_AVAILABLE = True
    _main_rasterizer = None
except ImportError:
    _lean_rasterizer = None
    LEAN_RASTERIZER_AVAILABLE = False

# FP16 lean library (diff_surfel_3D_16) — FP16 weights + FP16 GEMM shared memory
try:
    import diff_surfel_3D_16 as _fp16_rasterizer
    FP16_RASTERIZER_AVAILABLE = True
except ImportError:
    _fp16_rasterizer = None
    FP16_RASTERIZER_AVAILABLE = False

# TC lean library (diff_surfel_3D_tc) — Tensor Core WMMA for MLP
try:
    import diff_surfel_3D_tc as _tc_rasterizer
    TC_RASTERIZER_AVAILABLE = True
except ImportError:
    _tc_rasterizer = None
    TC_RASTERIZER_AVAILABLE = False

# SH TC library (diff_surfel_3D_sh) — TC WMMA MLP → SH coefficients
try:
    import diff_surfel_3D_sh as _sh_tc_rasterizer
    SH_TC_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_tc_rasterizer = None
    SH_TC_RASTERIZER_AVAILABLE = False

# SH+residual library (diff_surfel_3D_sh_res) — per-Gaussian SH + tiny hash MLP residual
try:
    import diff_surfel_3D_sh_res as _sh_res_rasterizer
    SH_RES_RASTERIZER_AVAILABLE = True
except ImportError:
    _sh_res_rasterizer = None
    SH_RES_RASTERIZER_AVAILABLE = False

# Import main rasterizer if lean-only mode not set
if not _USE_LEAN_ONLY:
    try:
        import diff_surfel_rasterization as _main_rasterizer
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings
    except ImportError:
        if not LEAN_RASTERIZER_AVAILABLE:
            raise ImportError("Neither diff_surfel_rasterization nor diff_surfel_3D could be imported. Build one of them.")
        _main_rasterizer = None
else:
    _main_rasterizer = None

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal, save_points, depths_to_points, cam2rays
import torch.nn.functional as torch_F
import time
# from hash_encoder.FeatureBlend import FeatureBlend
from utils.general_utils import MEM_PRINT

# One-time verification flags for render modes
_3D_DIRECT_FUSED_VERIFIED = False


class IntersectionOpacityGrad(torch.autograd.Function):
    """
    Custom autograd Function to connect intersection weight gradients back to Gaussian
    opacity, scale, rotation, AND position (through the exact geometry gradient path via transMat).

    This provides a SINGLE clean gradient path for geometry parameters, similar to how
    cat mode gets all geometry gradients through the CUDA rasterizer backward pass.

    CRITICAL DESIGN: This function also computes xyz for all intersections and returns it.
    This allows grad_xyz (from hash/MLP backward) to flow into this function's backward,
    where we compute dL_duv and pass it to the CUDA kernel - matching cat mode's gradient path.

    In 3D/3D_direct mode, the blending weight is: weight = alpha * T = opacity * G * T
    where G is the kernel value, T is transmittance, and alpha = opacity * G.

    IMPORTANT: This function must be called ONCE for ALL intersections (not per-batch)
    to correctly compute the transmittance chain gradient effect.

    The gradient computation uses the unified backward_from_weight_grad kernel that:
    1. Computes transmittance chain gradient: dL_dalpha = (dL_dweight - last_dL_dT) * T
    2. Computes dL_dopacity = G * dL_dalpha
    3. Computes dL_dtransMat via ray-disk intersection backprop with dL_duv from hash/xyz path
    4. Computes dL_dmean2D for position gradients (screen-space position feedback)
    5. NEW: Uses transmat_to_scale_rot_grad CUDA kernel for proper coordinate conversion

    Forward: Returns (weight_values, xyz, scale, rotation_quaternions, screenspace_points)
    Backward: Computes dL/dopacity, dL/dscale, dL/drotation, dL/dmean3D, and dL/dmean2D

    Args:
        opacity: [N] per-Gaussian opacity values
        scale: [N, 2] per-Gaussian scale values
        rotation_quaternions: [N, 4] quaternion rotation parameters
        rotation_matrices: [N, 3, 3] rotation matrices for xyz computation
        screenspace_points: [N, 3] screen-space points (requires_grad=True) for position gradients
        means3D: [N, 3] Gaussian centers
        projmatrix: [4, 4] projection matrix for proper coordinate conversion
        geomBuffer: Raw geometry buffer from CUDA forward (contains transMat)
        weight_values: [M] blending weights (alpha * T) from CUDA
        T_values: [M] transmittance values from CUDA buffer
        G_values: [M] kernel values from CUDA buffer
        alpha_values: [M] alpha values from CUDA buffer
        s_x: [M] intersection s.x coordinates
        s_y: [M] intersection s.y coordinates
        rho_flag: [M] 1.0=disk intersection, 0.0=center intersection
        gaussian_ids: [M] Gaussian indices for each intersection
        pixel_ids: [M] pixel indices for each intersection (must be sorted!)
        W, H: image dimensions
    """
    @staticmethod
    def forward(ctx, opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weight_values,
                T_values, G_values, alpha_values, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H):
        # Compute xyz for ALL intersections (so grad_xyz flows into backward)
        # Using the gathered parameters for the intersections
        b_scales = scale[gaussian_ids]  # [M, 2]
        b_R = rotation_matrices[gaussian_ids]  # [M, 3, 3]
        b_means = means3D[gaussian_ids]  # [M, 3]

        # Disk intersection formula: xyz = s_x * scale_x * R[:,0] + s_y * scale_y * R[:,1] + mean
        xyz_disk = (s_x[:, None] * b_scales[:, 0:1] * b_R[:, :, 0] +
                    s_y[:, None] * b_scales[:, 1:2] * b_R[:, :, 1] +
                    b_means)  # [M, 3]

        # Blend based on rho_flag: disk intersection vs Gaussian center
        xyz = rho_flag[:, None] * xyz_disk + (1.0 - rho_flag[:, None]) * b_means  # [M, 3]

        # Save for backward - note geomBuffer is not a tensor we save directly
        ctx.save_for_backward(opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
                              projmatrix, viewmatrix, gaussian_ids, pixel_ids, T_values, G_values, alpha_values,
                              s_x, s_y, rho_flag)
        ctx.geomBuffer = geomBuffer  # Store as attribute (raw buffer)
        ctx.n_gaussians = opacity.shape[0]
        ctx.W = W
        ctx.H = H

        # Return weights, xyz, scale, rotation_quaternions, and screenspace_points
        # xyz is returned so that grad_xyz flows back here from the hash/MLP chain
        return weight_values, xyz, scale, rotation_quaternions, screenspace_points

    @staticmethod
    def backward(ctx, grad_weight, grad_xyz, grad_scale_passthrough, grad_rotation_passthrough, grad_screenspace_passthrough):
        opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D, \
            projmatrix, viewmatrix, gaussian_ids, pixel_ids, T_values, G_values, alpha_values, \
            s_x, s_y, rho_flag = ctx.saved_tensors

        M = grad_weight.shape[0]
        N = ctx.n_gaussians
        W, H = ctx.W, ctx.H
        device = grad_weight.device
        dtype = grad_weight.dtype

        if M == 0:
            grad_opacity = torch.zeros(N, device=device, dtype=dtype)
            grad_scale = torch.zeros(N, 2, device=device, dtype=dtype)
            grad_rotation = torch.zeros(N, 4, device=device, dtype=dtype)  # quaternion gradients
            grad_screenspace = torch.zeros(N, 3, device=device, dtype=dtype)
            grad_means3D = torch.zeros(N, 3, device=device, dtype=dtype)
            if grad_scale_passthrough is not None:
                grad_scale = grad_scale + grad_scale_passthrough
            if grad_rotation_passthrough is not None:
                grad_rotation = grad_rotation + grad_rotation_passthrough
            if grad_screenspace_passthrough is not None:
                grad_screenspace = grad_screenspace + grad_screenspace_passthrough
            return grad_opacity, grad_scale, grad_rotation, None, grad_screenspace, grad_means3D, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        # =======================================================================
        # Compute dL_duv from grad_xyz (xyz gradient contribution to ray-disk Jacobian)
        #
        # This matches CAT mode backward.cu lines 1240-1241 where dL_duv is added
        # to dL_ds BEFORE the ray-disk Jacobian is applied:
        #   dL_ds.x += dL_duv.x;
        #   dL_ds.y += dL_duv.y;
        #
        # The chain rule for xyz = s_x * scale_x * R[:,0] + s_y * scale_y * R[:,1] + mean:
        #   dL/d(s_x) = dot(dL/d(xyz), scale_x * R[:,0]) = dot(dL_dxyz, SuTu)
        #   dL/d(s_y) = dot(dL/d(xyz), scale_y * R[:,1]) = dot(dL_dxyz, SvTv)
        #
        # This is CRITICAL for matching CAT mode gradients - dL_duv flows through
        # the ray-disk Jacobian to contribute to transMat gradients.
        # =======================================================================
        if grad_xyz is not None:
            # Gather per-intersection parameters
            b_scales = scale[gaussian_ids]  # [M, 2]
            b_R = rotation_matrices[gaussian_ids]  # [M, 3, 3]

            # Only disk intersections contribute (center intersections have s ≈ 0)
            grad_xyz_masked = grad_xyz * rho_flag[:, None]  # [M, 3]

            # SuTu = scale_x * R[:,0], SvTv = scale_y * R[:,1]
            SuTu = b_scales[:, 0:1] * b_R[:, :, 0]  # [M, 3]
            SvTv = b_scales[:, 1:2] * b_R[:, :, 1]  # [M, 3]

            # dL_duv = dot(dL_dxyz, SuTu/SvTv)
            dL_duv_x = (grad_xyz_masked * SuTu).sum(dim=1)  # [M]
            dL_duv_y = (grad_xyz_masked * SvTv).sum(dim=1)  # [M]
        else:
            dL_duv_x = torch.zeros(M, device=device, dtype=dtype)
            dL_duv_y = torch.zeros(M, device=device, dtype=dtype)

        # Compute pixel boundaries for CUDA kernel
        pixel_change = torch.zeros(M + 1, dtype=torch.bool, device=device)
        pixel_change[0] = True
        pixel_change[-1] = True
        if M > 1:
            pixel_change[1:-1] = pixel_ids[1:] != pixel_ids[:-1]
        pixel_starts = torch.nonzero(pixel_change, as_tuple=True)[0].int()

        # Call unified CUDA kernel that reads transMat from geomBuffer
        # Now includes dL_duv from hash/xyz gradient path!
        # Returns: (dL_dopacity [N], dL_dtransMat [N, 9], dL_dmean2D [N, 2])
        if FP16_RASTERIZER_AVAILABLE and _fp16_rasterizer is not None:
            from diff_surfel_3D_16 import backward_from_weight_grad, transmat_to_scale_rot_grad
        elif LEAN_RASTERIZER_AVAILABLE and _main_rasterizer is None:
            from diff_surfel_3D import backward_from_weight_grad, transmat_to_scale_rot_grad
        else:
            from diff_surfel_rasterization import backward_from_weight_grad, transmat_to_scale_rot_grad

        # Compute per-intersection opacity for the kernel
        opacity_per_int = opacity[gaussian_ids]  # [M]

        grad_opacity, dL_dtransMat, dL_dmean2D = backward_from_weight_grad(
            ctx.geomBuffer, N,
            grad_weight.contiguous(),
            gaussian_ids.int().contiguous(),
            pixel_ids.int().contiguous(),
            pixel_starts.contiguous(),
            T_values.contiguous(),
            G_values.contiguous(),
            alpha_values.contiguous(),
            opacity_per_int.contiguous(),
            s_x.contiguous(),
            s_y.contiguous(),
            rho_flag.contiguous(),
            dL_duv_x.contiguous(),  # dL_duv from hash/xyz path
            dL_duv_y.contiguous(),  # dL_duv from hash/xyz path
            W, H
        )

        # =======================================================================
        # PART 1: INDIRECT gradient (via ray-disk Jacobian from CUDA kernel)
        # This converts screen-space dL_dtransMat to world-space scale/rotation gradients
        # using the projection matrix transformation: dL_dM = P * transpose(dL_dT) + dL_dhomoMat
        #
        # CRITICAL: We also include the xyz gradient contribution (dL_dhomoMat) here,
        # matching CAT mode's backward.cu lines 1186-1197 and 1428-1430.
        # This is the missing piece that caused gradient direction issues!
        # =======================================================================

        # Compute dL_dhomoMat from per-intersection grad_xyz and s values
        # Layout: [col0.xyz, col1.xyz, col2.xyz] where:
        #   col0 = sum over intersections of (dL_dxyz * s_x) - for scale_x direction
        #   col1 = sum over intersections of (dL_dxyz * s_y) - for scale_y direction
        #   col2 = sum over intersections of (dL_dxyz) - for mean position
        dL_dhomoMat = torch.zeros(N, 9, device=device, dtype=dtype)
        if grad_xyz is not None:
            # For col0 and col1: only disk intersections contribute (rho_flag=1)
            # because center intersections have s_x ≈ 0, s_y ≈ 0
            grad_xyz_disk = grad_xyz * rho_flag[:, None]  # [M, 3]

            # Column 0: += grad_xyz * s_x (for scale_x direction) - disk only
            dL_dhomoMat[:, 0:3].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz_disk * s_x[:, None]
            )
            # Column 1: += grad_xyz * s_y (for scale_y direction) - disk only
            dL_dhomoMat[:, 3:6].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz_disk * s_y[:, None]
            )
            # Column 2: += grad_xyz (for mean position) - ALL intersections!
            # Center intersections have xyz = mean, so they contribute to mean gradient
            dL_dhomoMat[:, 6:9].scatter_add_(
                0, gaussian_ids[:, None].expand(-1, 3),
                grad_xyz  # No masking for mean gradient!
            )

        # Extract transMat from geomBuffer for the t_vec formula (needed for dL_dmean2D contribution)
        if FP16_RASTERIZER_AVAILABLE and _fp16_rasterizer is not None:
            from diff_surfel_3D_16 import get_transmat_from_geombuffer
        elif LEAN_RASTERIZER_AVAILABLE and _main_rasterizer is None:
            from diff_surfel_3D import get_transmat_from_geombuffer
        else:
            from diff_surfel_rasterization import get_transmat_from_geombuffer
        transMat_precomp = get_transmat_from_geombuffer(ctx.geomBuffer, N)

        # dL_dnormal3D: Normal gradient from depth/normal loss
        # In 3D_direct mode, this is typically zero since normal loss gradients flow through native backward
        # We pass empty tensor for now; can be extended to capture normal gradients if needed
        dL_dnormal3D = torch.empty(0, device=device, dtype=dtype)

        grad_scale_indirect, grad_rotation_indirect, grad_means3D_indirect = transmat_to_scale_rot_grad(
            dL_dtransMat.contiguous(),
            dL_dhomoMat.contiguous(),
            dL_dmean2D.contiguous(),       # 2D mean gradient from backward_from_weight_grad
            dL_dnormal3D,                  # Normal gradient (empty for now)
            means3D.contiguous(),          # World-space positions
            transMat_precomp.contiguous(), # Forward pass transMat for t_vec formula
            scale.contiguous(),
            rotation_quaternions.contiguous(),
            projmatrix.contiguous(),
            viewmatrix.contiguous(),       # View matrix for normal gradient transform
            W, H  # Image dimensions for ndc2pix transformation
        )

        # =======================================================================
        # GRADIENT SCALING FOR 3D MODE
        #
        # With dL_dhomoMat now included in the CUDA kernel, the INDIRECT path
        # computes both:
        #   1. Kernel shape gradient contribution: P * transpose(dL_dT)
        #   2. XYZ gradient contribution: dL_dhomoMat (what DIRECT path computed before)
        #
        # This matches CAT mode's backward.cu where:
        #   dL_dM = P * transpose(dL_dT) + dL_dhomoMat  (line 1428-1430)
        #
        # IMPORTANT: The 3D MLP (xyz -> features) produces larger gradients than
        # CAT mode's 2D hash (uv -> features) because:
        #   1. 3D coordinates have larger spatial extent than 2D screen coords
        #   2. The gradient chains differently through 3D vs 2D networks
        #
        # Based on empirical comparison with CAT mode:
        #   - Scale gradients are ~3-5x larger
        #   - Rotation gradients are ~5-8x larger
        # We use a single scaling factor to bring them in line.
        # =======================================================================

        # NO SCALING FACTORS - gradients should be correct as computed
        # The test shows cosine similarity ~0.99 with autograd
        # If training is unstable, adjust learning rate instead
        grad_scale = grad_scale_indirect

        # =======================================================================
        # Rotation gradients: CUDA kernel returns dL/d(normalized_q)
        # We return this directly - PyTorch will chain through F.normalize
        # (which creates rotation_quaternions from _rotation) automatically.
        # DO NOT apply F.normalize chain rule here - that would double-apply it!
        # =======================================================================
        grad_rotation = grad_rotation_indirect

        # Convert dL_dmean2D [N, 2] to dL_dscreenspace [N, 3] (third component is 0)
        # This provides the screen-space position gradient that tells Gaussians where to move
        grad_screenspace = torch.zeros(N, 3, device=device, dtype=dtype)
        grad_screenspace[:, :2] = dL_dmean2D  # [N, 2] -> [N, 3] with z=0

        # Mean position gradients: from CUDA kernel's dL_dhomoMat column 2 (already computed!)
        # This is sum over intersections of grad_xyz, which equals d(xyz)/d(mean)
        grad_means3D = grad_means3D_indirect

        # Passthrough gradients (should be None/zero in normal operation)
        if grad_scale_passthrough is not None:
            grad_scale = grad_scale + grad_scale_passthrough
        if grad_rotation_passthrough is not None:
            grad_rotation = grad_rotation + grad_rotation_passthrough
        if grad_screenspace_passthrough is not None:
            grad_screenspace = grad_screenspace + grad_screenspace_passthrough

        # Return order: opacity, scale, rotation_quaternions, rotation_matrices, screenspace_points, means3D,
        #               projmatrix, viewmatrix, geomBuffer, weight_values, T_values, G_values, alpha_values,
        #               s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
        return grad_opacity, grad_scale, grad_rotation, None, grad_screenspace, grad_means3D, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def _quat_to_rotmat_vjp_batch(quats, dL_dR0, dL_dR1):
    """
    Convert rotation matrix column gradients to quaternion gradients.
    This is a batch-friendly Python implementation of quat_to_rotmat_vjp.

    IMPORTANT: Unlike the CUDA implementation, this CORRECTLY chains through the
    normalization gradient. The CUDA quat_to_rotmat_vjp computes gradients w.r.t.
    the normalized quaternion but doesn't chain back to the raw quaternion.
    This is technically incorrect but native cat mode works because the INDIRECT
    path (through CUDA) dominates and the gradients are approximately correct.

    For the DIRECT path in 3D_direct mode, we need correct gradients, so we
    include the normalization chain rule here.

    Args:
        quats: [N, 4] quaternions (w, x, y, z ordering as stored in the model)
        dL_dR0: [N, 3] gradient for first column of rotation matrix
        dL_dR1: [N, 3] gradient for second column of rotation matrix

    Returns:
        dL_dquat: [N, 4] quaternion gradients (w.r.t. raw/unnormalized quaternions)
    """
    # Normalize quaternions - same as build_rotation and CUDA quat_to_rotmat
    norm = torch.sqrt((quats ** 2).sum(dim=-1, keepdim=True))
    q = quats / (norm + 1e-8)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build full dL_dR from column gradients (column 2 is zeros)
    # v_R is column-major: v_R[col][row]
    # v_R[0] = dL_dR0, v_R[1] = dL_dR1, v_R[2] = 0
    v_R00, v_R01, v_R02 = dL_dR0[:, 0], dL_dR1[:, 0], torch.zeros_like(w)
    v_R10, v_R11, v_R12 = dL_dR0[:, 1], dL_dR1[:, 1], torch.zeros_like(w)
    v_R20, v_R21, v_R22 = dL_dR0[:, 2], dL_dR1[:, 2], torch.zeros_like(w)

    # quat_to_rotmat_vjp formulas from auxiliary.h
    # These compute dL/d(normalized_q), i.e., gradients w.r.t. normalized quaternion
    # In CUDA, v_R[col][row], so v_R[1][2] maps to Python v_R21, v_R[2][1] to v_R12, etc.

    # v_quat.x (w gradient) - CUDA: x*(v_R[1][2]-v_R[2][1]) + y*(v_R[2][0]-v_R[0][2]) + z*(v_R[0][1]-v_R[1][0])
    v_w = 2.0 * (x * (v_R21 - v_R12) + y * (v_R02 - v_R20) + z * (v_R10 - v_R01))

    # v_quat.y (x gradient) - CUDA: -2x*(v_R[1][1]+v_R[2][2]) + y*(v_R[0][1]+v_R[1][0]) + z*(v_R[0][2]+v_R[2][0]) + w*(v_R[1][2]-v_R[2][1])
    v_x = 2.0 * (-2.0 * x * (v_R11 + v_R22) + y * (v_R10 + v_R01) + z * (v_R20 + v_R02) + w * (v_R21 - v_R12))

    # v_quat.z (y gradient) - CUDA: x*(v_R[0][1]+v_R[1][0]) - 2y*(v_R[0][0]+v_R[2][2]) + z*(v_R[1][2]+v_R[2][1]) + w*(v_R[2][0]-v_R[0][2])
    v_y = 2.0 * (x * (v_R10 + v_R01) - 2.0 * y * (v_R00 + v_R22) + z * (v_R21 + v_R12) + w * (v_R02 - v_R20))

    # v_quat.w (z gradient) - CUDA: x*(v_R[0][2]+v_R[2][0]) + y*(v_R[1][2]+v_R[2][1]) - 2z*(v_R[0][0]+v_R[1][1]) + w*(v_R[0][1]-v_R[1][0])
    v_z = 2.0 * (x * (v_R20 + v_R02) + y * (v_R21 + v_R12) - 2.0 * z * (v_R00 + v_R11) + w * (v_R10 - v_R01))

    # dL/d(normalized_q) - gradients w.r.t. unit quaternion
    dL_dq = torch.stack([v_w, v_x, v_y, v_z], dim=-1)

    # Chain through the normalization: q = quats / ||quats||
    # Jacobian: dq/d(quats) = (I - q q^T) / ||quats||
    # dL/d(quats) = (dL/dq - q * (dL/dq · q)) / ||quats||
    q_dot_grad = (q * dL_dq).sum(dim=-1, keepdim=True)
    dL_dquats = (dL_dq - q * q_dot_grad) / (norm + 1e-8)

    return dL_dquats


class RenderCache:
    """Cache for pre-allocated tensors to avoid repeated allocations during inference."""

    def __init__(self):
        self.screenspace_points = None
        self.rotation_matrices = None
        self.homotrans = None
        self.shape_dims = {}  # Cache shape_dims tensors by key
        self._num_gaussians = 0

    def get_screenspace_points(self, num_gaussians, device='cuda'):
        if self.screenspace_points is None or self.screenspace_points.shape[0] != num_gaussians:
            self.screenspace_points = torch.zeros(num_gaussians, 3, dtype=torch.float32, device=device)
            self._num_gaussians = num_gaussians
        return self.screenspace_points

    def get_shape_dims(self, key, values, device='cuda'):
        """Get or create shape_dims tensor. Key is tuple of (gs_dim, hs_dim, os_dim)."""
        if key not in self.shape_dims:
            self.shape_dims[key] = torch.tensor(values, dtype=torch.int32, device=device)
        return self.shape_dims[key]

    def cache_rotation_matrices(self, rotations_quat):
        """Pre-compute and cache rotation matrices from quaternions."""
        from utils.general_utils import build_rotation
        self.rotation_matrices = build_rotation(rotations_quat)
        return self.rotation_matrices

    def cache_homotrans(self, rotations_quat, scales, xyzs):
        """Pre-compute and cache homogeneous transform matrices from quaternions."""
        from utils.general_utils import build_rotation, build_H
        rots = build_rotation(rotations_quat)
        self.homotrans = build_H(rots, scales, xyzs)
        return self.homotrans


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ingp = None,
    beta = 0, iteration = None, cfg = None, record_transmittance = False, use_xyz_mode = False, decompose_mode = None, max_intersections = 0,
    skip_mlp = False, force_no_hash_cuda = False, temperature = 1.0, force_ratio = 0.2, no_gumbel = False, dropout_lambda = 0.0, is_training = True,
    aabb_mode = "2dgs", aa = 0.0, aa_threshold = 0.01, skybox = None, background_mode = "none", bg_hashgrid = None, detach_hash_grad = False,
    return_raw_features = False, fast_inference = False, cache = None, max_intersections_per_pixel = 32):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    decompose_mode: None (normal), 'gaussian_only' (zero hashgrid features), 'ngp_only' (zero per-Gaussian features)
                   Only used in cat mode for visualization/debugging.
    fast_inference: If True, skip expensive post-processing (normal computation, etc.) for faster inference.
    cache: Optional RenderCache object to reuse pre-allocated tensors and cached computations.
    max_intersections: Max ray-Gaussian intersections per pixel. 0 means no limit.
    skip_mlp: If True, skip MLP decode and return zeros for RGB (for benchmarking).
    force_no_hash_cuda: If True, disable hash_in_CUDA (use plain 2DGS rasterizer, no hash query).
    dropout_lambda: Hash dropout rate for cat_dropout mode (0.2 = 20% of Gaussians don't query hash during training).
    is_training: Whether we're in training mode (affects dropout behavior).
    """

    render_start_time = time.time()

    XYZ_TYPE = cfg.ingp_stage.XYZ_TYPE
    assert(XYZ_TYPE == "UV" or XYZ_TYPE == "DEPTH")

    # Create zero tensor for screen-space points
    if fast_inference and cache is not None:
        # Use cached pre-allocated tensor (no gradients needed for inference)
        screenspace_points = cache.get_screenspace_points(pc.get_xyz.shape[0])
        screenspace_points.zero_()  # Reset to zeros
    else:
        # Training mode: need gradients
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
    # 3D mode: intersection buffer output, SH blending in PyTorch
    is_3D_mode = ingp is not None and hasattr(ingp, 'is_3D_mode') and ingp.is_3D_mode
    # 3D_direct mode: intersection buffer output, direct RGB via MLP (like cat mode's 2D MLP)
    is_3D_direct_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_mode') and ingp.is_3D_direct_mode
    # 3D_direct_fused mode: fused in-kernel MLP (no intersection buffer, hash+MLP in CUDA)
    is_3D_direct_fused_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_fused_mode') and ingp.is_3D_direct_fused_mode
    # 3D_direct_lean mode: same as 3D_direct_fused but uses lean rasterizer library (faster builds)
    is_3D_direct_lean_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_lean_mode') and ingp.is_3D_direct_lean_mode
    # 3D_direct_fp16 mode: FP16 weights + FP16 GEMM shared memory (diff_surfel_3D_16)
    is_3D_direct_fp16_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_fp16_mode') and ingp.is_3D_direct_fp16_mode
    # 3D_direct_TC mode: Tensor Core WMMA for MLP (diff_surfel_3D_tc)
    is_3D_direct_tc_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_tc_mode') and ingp.is_3D_direct_tc_mode
    # 3D_SH_TC mode: TC WMMA MLP → 48D SH coefs, viewdir eval in kernel (diff_surfel_3D_sh)
    is_3D_direct_sh_tc_mode = ingp is not None and hasattr(ingp, 'is_3D_direct_sh_tc_mode') and ingp.is_3D_direct_sh_tc_mode
    # 3D_SH_res mode: per-Gaussian SH + tiny hash MLP residual (diff_surfel_3D_sh_res)
    is_3D_SH_res_mode = ingp is not None and hasattr(ingp, 'is_3D_SH_res_mode') and ingp.is_3D_SH_res_mode
    # Treat lean/fp16/tc/sh_tc/sh_res mode same as fused mode for rendering logic
    if is_3D_direct_lean_mode or is_3D_direct_fp16_mode or is_3D_direct_tc_mode or is_3D_direct_sh_tc_mode or is_3D_SH_res_mode:
        is_3D_direct_fused_mode = True

    hash_in_CUDA = True
    try:
        if ingp is None:
            hash_in_CUDA = False
        if iteration < cfg.ingp_stage.switch_iter:
            hash_in_CUDA = False
        # Diffuse/Specular mode: never use hash_in_CUDA (no hashgrid)
        # Diffuse_ngp/diffuse_offset: also don't use hash_in_CUDA (we query hashgrid in Python on unprojected depth)
        # 3D/3D_direct mode: hash query happens in PyTorch after getting intersection buffer
        # 3D_direct_fused mode: hash query happens in CUDA kernel (hash_in_CUDA = True)
        if is_diffuse_mode or is_specular_mode or is_diffuse_ngp_mode or is_diffuse_offset_mode or is_3D_mode or is_3D_direct_mode:
            hash_in_CUDA = False
        # Force disable for benchmarking
        if force_no_hash_cuda:
            hash_in_CUDA = False
    except:
        pass


    # 3D/3D_direct mode: configure max_intersections_per_pixel (other setup done after hash_in_CUDA block)
    # 3D_direct_fused mode: no intersection buffer needed (fused in-kernel)
    if not is_3D_mode and not is_3D_direct_mode:
        max_intersections_per_pixel = 0  # Disabled for non-3D modes

    if ingp is not None and hash_in_CUDA == False and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode and not is_3D_mode and not is_3D_direct_mode and not is_3D_direct_fused_mode:
        ### warm-up
        override_color = ingp(points_3D = means3D, with_xyz = False).float()
        feat_dim = ingp.active_levels * ingp.level_dim
        # Set shape_dims for warmup phase (baseline mode, no hashgrid in CUDA)
        output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim
        shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
    
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

    # Adaptive_zero mode detection (cat-like features + weighted hash, zeros when weight=0)
    is_adaptive_zero_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_zero_mode') and ingp.is_adaptive_zero_mode

    # Adaptive_gate mode detection (VQ-AD style gating: soft→STE→hard)
    is_adaptive_gate_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_adaptive_gate_mode') and ingp.is_adaptive_gate_mode

    # Cat_dropout mode detection (cat mode with hash dropout during training)
    is_cat_dropout_mode = hash_in_CUDA and ingp is not None and hasattr(ingp, 'is_cat_dropout_mode') and ingp.is_cat_dropout_mode

    hybrid_levels = ingp.hybrid_levels if (is_cat_mode or is_adaptive_zero_mode or is_adaptive_gate_mode or is_cat_dropout_mode or is_3D_direct_fused_mode) else 0
    
    render_mode = 0  # 0 = baseline, 1 = cat, 2 = adaptive_zero, 3 = 3D
    viewdirs_enc = None  # Pre-encoded view directions for 3D_direct_fused mode
    # NOTE: 3D mode (render_mode=3) is set after hash_in_CUDA block to avoid being overwritten

    # Initialize shape_dims tensor [GS, HS, OS] - will be updated per mode
    # GS = Gaussian shape, HS = Hash shape, OS = Output shape
    # Default for SH rendering (no hashgrid): GS=0, HS=0, OS=3 (RGB)
    # This will be overwritten for warmup mode (line 111) or hash_in_CUDA modes (below)
    shape_dims = torch.tensor([0, 0, 3], dtype=torch.int32, device="cuda")

    if hash_in_CUDA:
        # Initialize shape_dims for hash_in_CUDA modes
        output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim
        shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
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

            # FP16 hash features: convert embeddings to half precision
            # Saves per-frame FP32→FP16 conversion in rasterize_points.cu
            # Autograd tracks .half() so gradients flow back to the FP32 parameter
            if is_3D_direct_fp16_mode or is_3D_direct_tc_mode or is_3D_direct_sh_tc_mode:
                features = features.half()

        # Use cached homotrans if available (for fast inference)
        if fast_inference and cache is not None and cache.homotrans is not None:
            homotrans = cache.homotrans
        else:
            homotrans = pc.get_homotrans()
        ap_level = pc.get_appearance_level
        contract = ingp.contract

        # Baseline mode: set shape_dims for hashgrid-only rendering
        # NOTE: We keep SH for preprocessing (it evaluates to 3-channel RGB in geomState.rgb)
        # but the rendering kernel will ignore it when level > 0 and query hashgrid instead
        # shape_dims tells CUDA to allocate 24-channel output buffer for hashgrid features
        if not is_cat_mode:
            # Baseline with hashgrid: GS=0 (no per-Gaussian), HS=output_dim, OS=output_dim
            # Use total levels (not active_levels) - buffer is always full size, C2F masks unused channels
            output_dim = ingp.levels * ingp.level_dim  # Total levels * per_level_dim (e.g., 6*4 = 24D)
            shape_dims = torch.tensor([0, output_dim, output_dim], dtype=torch.int32, device="cuda")
        
        # Cat mode: only activate if hybrid_levels > 0
        # When hybrid_levels == 0, behave identically to baseline
        if is_cat_mode and hybrid_levels > 0:
            # Set per-Gaussian features as colors_precomp
            gaussian_features = pc.get_gaussian_features
            shs = None

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
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

            colors_precomp = gaussian_features
            render_mode = 1

            # Cat mode: G + H = O
            gaussian_dim = hybrid_levels * ingp.level_dim  # e.g., 5*4 = 20
            hash_dim = active_hashgrid_levels * ingp.level_dim  # e.g., 1*4 = 4
            output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24
            if fast_inference and cache is not None:
                shape_dims = cache.get_shape_dims((gaussian_dim, hash_dim, output_dim), [gaussian_dim, hash_dim, output_dim])
            else:
                shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_SH_res mode: per-Gaussian SH + tiny hash MLP residual
        # SH handles per-Gaussian view-dependent appearance (evaluated in CUDA preprocessing)
        # Hash MLP adds view-independent spatial correction per-intersection
        # No per-Gaussian features needed (hybrid_levels=0)
        elif is_3D_SH_res_mode:
            from diff_surfel_3D_sh_res import set_mlp_weights

            # Use standard SH coefficients (NOT per-Gaussian features)
            shs = pc.get_features
            colors_precomp = None

            # Hash grid setup (all levels are hash, no hybrid)
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.hashgrid_levels if not ingp.hashgrid_disabled else 0

            # Encode levels: (total << 16) | (active_hashgrid << 8) | hybrid=0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | 0

            # Pad offsets
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Upload tiny MLP weights [16,16] each
            mlp_weights = ingp.get_fused_mlp_weights()
            if mlp_weights is not None:
                W1, W2, W3 = mlp_weights
                set_mlp_weights(W1, W2, W3)

            render_mode = 5  # Fused in-kernel MLP

            # One-time verification
            global _3D_DIRECT_FUSED_VERIFIED
            if not _3D_DIRECT_FUSED_VERIFIED:
                hash_dim = active_hashgrid_levels * ingp.level_dim
                print(f"[3D_SH_RES] render_mode={render_mode}, "
                      f"SH=degree-3 (48 params), "
                      f"hash={active_hashgrid_levels}×{ingp.level_dim}={hash_dim}D, "
                      f"MLP=16→16→16→3 residual")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Dimensions: no per-Gaussian features, hash only
            gaussian_dim = 0
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = 3  # RGB output
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # 3D_direct_fused mode: fused in-kernel MLP (hash+MLP in CUDA)
        # Like cat mode but MLP runs in-kernel and outputs RGB directly
        # hybrid_levels are per-Gaussian (coarse), remaining levels are hashgrid (fine)
        elif is_3D_direct_fused_mode and ingp.hybrid_levels > 0:
            # Import set_mlp_weights from the appropriate rasterizer
            if is_3D_direct_sh_tc_mode and SH_TC_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_sh import set_mlp_weights
            elif is_3D_direct_tc_mode and TC_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_tc import set_mlp_weights
            elif is_3D_direct_fp16_mode and FP16_RASTERIZER_AVAILABLE:
                from diff_surfel_3D_16 import set_mlp_weights
            elif is_3D_direct_lean_mode and LEAN_RASTERIZER_AVAILABLE:
                from diff_surfel_3D import set_mlp_weights
            else:
                from diff_surfel_rasterization import set_mlp_weights

            # Same setup as cat mode: per-Gaussian features + hashgrid
            gaussian_features = pc.get_gaussian_features
            shs = None

            # Level split: hybrid_levels are Gaussian, rest are hashgrid
            # E.g., total=6, hybrid=3 means 3 Gaussian levels + 3 hashgrid levels
            total_levels = ingp.levels
            hybrid_levels_fused = ingp.hybrid_levels
            hashgrid_levels = total_levels - hybrid_levels_fused  # Fine levels from hash

            # 3D_direct_fused has no C2F - MLP expects fixed input dimensions
            # Use all hashgrid levels (not active_hashgrid_levels which may be limited by C2F)
            active_hashgrid_levels = hashgrid_levels if not ingp.hashgrid_disabled else 0

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_fused

            # Pad offsets (same as cat mode)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            colors_precomp = gaussian_features

            # Upload MLP weights to CUDA constant memory
            # NOTE: MLP gradients are computed in CUDA but not yet wired back to PyTorch
            # The mlp_fused parameters won't receive gradients until that's implemented
            mlp_weights = ingp.get_fused_mlp_weights()
            if mlp_weights is not None:
                W1, W2, W3 = mlp_weights
                set_mlp_weights(W1, W2, W3, is_sh_mode=is_3D_direct_sh_tc_mode)

            render_mode = 5  # 3D_direct_fused: fused in-kernel MLP

            # One-time verification that we're using the fused mode
            if not _3D_DIRECT_FUSED_VERIFIED:
                print(f"[3D_DIRECT_FUSED] render_mode={render_mode}, "
                      f"gauss={hybrid_levels_fused}×{ingp.level_dim}={hybrid_levels_fused * ingp.level_dim}D, "
                      f"hash={active_hashgrid_levels}×{ingp.level_dim}={active_hashgrid_levels * ingp.level_dim}D")
                _3D_DIRECT_FUSED_VERIFIED = True

            # Dimensions for CUDA kernel
            gaussian_dim = hybrid_levels_fused * ingp.level_dim  # Coarse: e.g., 5*4 = 20
            hash_dim = active_hashgrid_levels * ingp.level_dim   # Fine: e.g., 1*4 = 4
            output_dim = 3  # RGB output from fused MLP
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

            # View direction handling depends on mode:
            # - SH_TC: no view encoding needed (CUDA kernel computes raw viewdir per-intersection)
            # - Other fused modes: pre-encode view directions for MLP input
            if not is_3D_direct_sh_tc_mode:
                rays_d, _ = cam2rays(viewpoint_camera)  # [H*W, 3]
                ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # [H*W, 3] normalized
                viewdirs_enc = ingp._encode_view(ray_unit).float().contiguous()  # [H*W, 16]

        # Cat_dropout mode: cat mode with hash dropout during training
        # Uses mode 14 (adaptive_zero kernel) with hardcoded weights
        # Training: weight=1 for (1-dropout_lambda)% of Gaussians, weight=0 for dropout_lambda%
        # Inference: weight=1 for all (identical to cat mode)
        elif is_cat_dropout_mode and hybrid_levels > 0:
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)
            shs = None

            # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            N = gaussian_features.shape[0]

            if is_training and dropout_lambda > 0:
                # Training with dropout: randomly mask some Gaussians to not use hash
                # weight=1 means use hash, weight=0 means use zeros for fine levels
                dropout_mask = (torch.rand(N, 1, device="cuda") >= dropout_lambda).float()
            else:
                # Inference or no dropout: all weights = 1 (use hash for all)
                dropout_mask = torch.ones(N, 1, device="cuda")

            # Concatenate: [coarse_features | weight]
            colors_precomp = torch.cat([gaussian_features, dropout_mask], dim=1)

            # Use mode 14 (adaptive_zero kernel) - training mode (no inference flag)
            render_mode = 2

            # Shape dims: GS = coarse, HS = fine hash, OS = total
            gaussian_dim = hybrid_levels * ingp.level_dim
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = total_levels * ingp.level_dim
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # Adaptive_zero mode: cat-like features + weighted hash (zeros when weight=0)
        elif is_adaptive_zero_mode and hybrid_levels > 0:
            # Same feature layout as cat mode: per-Gaussian for coarse, hash for fine
            # But with a weight that controls whether to query hash or use zeros
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)

            # Apply temperature scaling to sigmoid (higher temp = sharper sigmoid)
            blend_weight = torch.sigmoid(pc._adaptive_zero_weight * temperature)  # (N, 1)

            # Determine if we're in inference mode
            use_inference_mode = (hasattr(ingp, 'adaptive_zero_inference') and ingp.adaptive_zero_inference) or (decompose_mode is not None)

            # Decompose mode: override features/opacity for visualization
            opacity_override = None
            disable_hash_query = False

            if decompose_mode == 'gaussian_only':
                # Show only Gaussians with weight < 0.5 (they use zeros for fine levels)
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) >= 0.5] = 0.0
            elif decompose_mode == 'hybrid_gaussian_only':
                # Show Gaussians with weight >= 0.5, but mask out hashgrid
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) < 0.5] = 0.0
                disable_hash_query = True  # Will set active_hashgrid_levels = 0
            elif decompose_mode == 'hybrid_hash_only':
                # Show Gaussians with weight >= 0.5, but mask out gaussian features
                opacity_override = opacity.clone()
                opacity_override[blend_weight.squeeze(-1) < 0.5] = 0.0
                gaussian_features = torch.zeros_like(gaussian_features)

            if opacity_override is not None:
                opacity = opacity_override

            # Concatenate: [coarse_features | weight]
            colors_precomp = torch.cat([gaussian_features, blend_weight], dim=1)
            shs = None

            # Use mode 14 for adaptive_zero
            inference_flag = 1 if use_inference_mode else 0
            render_mode = 2 | (inference_flag << 8)

            # Level encoding same as cat: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            if disable_hash_query:
                active_hashgrid_levels = 0  # Disable hash query for hybrid_gaussian_only mode
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements (CUDA code expects up to 16 levels + 1)
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Shape dims: GS = coarse features + weight, HS = fine hash, OS = total output
            gaussian_dim = hybrid_levels * ingp.level_dim  # e.g., 5*4 = 20D (coarse only)
            hash_dim = active_hashgrid_levels * ingp.level_dim  # e.g., 1*4 = 4D
            output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24D
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

        # Adaptive_gate mode: Gumbel-STE with forced training for binary hash selection
        # Always binary masking (0 or 1) to prevent scale compensation artifacts
        # Uses same CUDA kernel as adaptive_zero (mode 14)
        elif is_adaptive_gate_mode and hybrid_levels > 0:
            gaussian_features = pc.get_gaussian_features  # (N, hybrid_levels * D)
            gate_logits = pc._gate_logits  # (N, 1)

            use_inference_mode = hasattr(ingp, 'adaptive_gate_inference') and ingp.adaptive_gate_inference

            if use_inference_mode or (decompose_mode is not None):
                # INFERENCE: Hard threshold on probability (prob > 0.5 → use hash)
                effective_mask = (torch.sigmoid(gate_logits) > 0.5).float()
            else:
                # TRAINING: STE with optional Gumbel noise + Forced Training

                if no_gumbel:
                    # Deterministic STE: no Gumbel noise, just hard threshold with soft gradients
                    soft_gate = torch.sigmoid(gate_logits / temperature)
                    hard_gate = (soft_gate > 0.5).float() - soft_gate.detach() + soft_gate
                else:
                    # Gumbel-STE: stochastic exploration
                    # 1. Gumbel noise for stochastic exploration
                    uniform = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
                    gumbel_noise = -torch.log(-torch.log(uniform))
                    noisy_logits = (gate_logits + gumbel_noise) / temperature

                    # 2. STE: hard forward, soft backward
                    soft_gate = torch.sigmoid(noisy_logits)
                    hard_gate = (soft_gate > 0.5).float() - soft_gate.detach() + soft_gate

                # 3. Forced training: force some Gaussians to use hash (detached, not learnable)
                force_mask = (torch.rand_like(gate_logits) < force_ratio).float().detach()

                # 4. Combine: max ensures binary output (0 or 1)
                effective_mask = torch.max(hard_gate, force_mask)

            # Handle decompose modes for visualization
            opacity_override = None
            disable_hash_query = False

            if decompose_mode == 'gaussian_only':
                # Force all gates closed (Gaussian-only rendering)
                effective_mask = torch.zeros_like(effective_mask)
            elif decompose_mode == 'ngp_only':
                # Force all gates open (hash for all)
                effective_mask = torch.ones_like(effective_mask)
            elif decompose_mode == 'gate_closed':
                # Show only Gaussians with gate closed (not using hash)
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) > 0.5] = 0.0  # Hide gate-open Gaussians
                effective_mask = torch.zeros_like(effective_mask)  # All use Gaussian-only
            elif decompose_mode == 'gate_open':
                # Show only Gaussians with gate open (using hash)
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0  # Hide gate-closed Gaussians
                effective_mask = torch.ones_like(effective_mask)  # All use hash
            elif decompose_mode == 'hybrid_gaussian_only':
                # Show Gaussians with gate open, but mask out hashgrid
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0
                disable_hash_query = True
            elif decompose_mode == 'hybrid_hash_only':
                # Show Gaussians with gate open, but mask out gaussian features
                gate_prob = torch.sigmoid(gate_logits)
                opacity_override = opacity.clone()
                opacity_override[gate_prob.squeeze(-1) <= 0.5] = 0.0
                gaussian_features = torch.zeros_like(gaussian_features)

            if opacity_override is not None:
                opacity = opacity_override

            # Concatenate: [coarse_features | mask]
            colors_precomp = torch.cat([gaussian_features, effective_mask], dim=1)
            shs = None

            # Use mode 14 (same CUDA kernel as adaptive_zero)
            inference_flag = 1 if (use_inference_mode or decompose_mode is not None) else 0
            render_mode = 2 | (inference_flag << 8)

            # Level encoding same as cat: (total << 16) | (active_hashgrid << 8) | hybrid
            total_levels = ingp.levels
            active_hashgrid_levels = ingp.active_hashgrid_levels if not ingp.hashgrid_disabled else 0
            if disable_hash_query:
                active_hashgrid_levels = 0
            levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

            # Pad offsets to 17 elements
            if offsets.shape[0] < 17:
                padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
                padded_offsets[:offsets.shape[0]] = offsets
                offsets = padded_offsets

            # Shape dims: GS = coarse features + mask, HS = fine hash, OS = total output
            gaussian_dim = hybrid_levels * ingp.level_dim
            hash_dim = active_hashgrid_levels * ingp.level_dim
            output_dim = total_levels * ingp.level_dim
            shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

    # 3D mode: Configure AFTER the reset block (like cat mode configures inside hash_in_CUDA)
    # This follows cat mode's pattern exactly: set shs=None, colors_precomp=gaussian_features
    if is_3D_mode and ingp is not None:
        # Per-Gaussian features as colors_precomp (like cat mode at line 550-566)
        gaussian_features = pc.get_gaussian_features
        shs = None  # Like cat mode: no SH, use colors_precomp instead
        colors_precomp = gaussian_features  # Like cat mode

        # Homotrans for intersection computation (like cat mode at line 532)
        homotrans = pc.get_homotrans()

        # Encode levels for CUDA: (total << 16) | (active_hashgrid << 8) | hybrid
        # IMPORTANT: levels must be non-zero to enter the render_mode switch in CUDA
        total_levels = ingp.levels
        active_hashgrid_levels = 0  # 3D mode: hashgrid encoding in PyTorch, not CUDA
        hybrid_levels_3D = ingp.hybrid_levels
        levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_3D

        # Pad offsets to 17 elements (like cat mode at line 561-564)
        offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
        if offsets.shape[0] < 17:
            padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
            padded_offsets[:offsets.shape[0]] = offsets
            offsets = padded_offsets

        # Hashgrid params (dummy for 3D mode - hash query happens in PyTorch)
        features = torch.zeros((1, 4), device="cuda")  # Dummy hash features
        gridrange = ingp.gridrange if hasattr(ingp, 'gridrange') else torch.tensor([-2.0, 2.0], device="cuda")
        per_level_scale = ingp.growth_rate if hasattr(ingp, 'growth_rate') else 1.0
        base_resolution = ingp.resolutions[0] if hasattr(ingp, 'resolutions') and len(ingp.resolutions) > 0 else 16
        align_corners = False
        interpolation = 0
        contract = ingp.contract if hasattr(ingp, 'contract') else False

        # render_mode = 3 for intersection buffer output
        render_mode = 3

        # shape_dims like cat mode: G + H = O (line 570-576)
        gaussian_dim = hybrid_levels_3D * ingp.level_dim  # e.g., 5*4 = 20
        hash_dim = active_hashgrid_levels * ingp.level_dim  # 0 for 3D mode
        output_dim = total_levels * ingp.level_dim  # e.g., 6*4 = 24
        shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

    # 3D_direct mode: Same rasterizer setup as 3D mode, but uses direct RGB output instead of SH
    if is_3D_direct_mode and ingp is not None:
        gaussian_features = pc.get_gaussian_features
        shs = None
        colors_precomp = gaussian_features

        homotrans = pc.get_homotrans()

        total_levels = ingp.levels
        active_hashgrid_levels = 0  # Hash query in PyTorch
        hybrid_levels_3D = ingp.hybrid_levels
        levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels_3D

        offsets = torch.zeros((1,), dtype=torch.int32, device="cuda")
        if offsets.shape[0] < 17:
            padded_offsets = torch.zeros(17, dtype=offsets.dtype, device=offsets.device)
            padded_offsets[:offsets.shape[0]] = offsets
            offsets = padded_offsets

        features = torch.zeros((1, 4), device="cuda")
        gridrange = ingp.gridrange if hasattr(ingp, 'gridrange') else torch.tensor([-2.0, 2.0], device="cuda")
        per_level_scale = ingp.growth_rate if hasattr(ingp, 'growth_rate') else 1.0
        base_resolution = ingp.resolutions[0] if hasattr(ingp, 'resolutions') and len(ingp.resolutions) > 0 else 16
        align_corners = False
        interpolation = 0
        contract = ingp.contract if hasattr(ingp, 'contract') else False

        render_mode = 3  # Same rasterizer as 3D mode (intersection buffer output)

        gaussian_dim = hybrid_levels_3D * ingp.level_dim
        hash_dim = active_hashgrid_levels * ingp.level_dim
        output_dim = total_levels * ingp.level_dim
        shape_dims = torch.tensor([gaussian_dim, hash_dim, output_dim], dtype=torch.int32, device="cuda")

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
        max_intersections = max_intersections,
        detach_hash_grad = detach_hash_grad,
        max_intersections_per_pixel = max_intersections_per_pixel,
        # pipe.debug
    )

    hashgrid_settings = HashGridSettings(
        L = levels,
        S = math.log2(per_level_scale),
        H = base_resolution,
        align_corners = align_corners,
        interpolation = interpolation,
        shape_dims = shape_dims,
        aa = aa,
        aa_threshold = aa_threshold
    )

    # Use SH_RES, SH_TC, TC, FP16, or lean rasterizer for fused modes
    if is_3D_SH_res_mode and SH_RES_RASTERIZER_AVAILABLE:
        rasterizer = _sh_res_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_sh_tc_mode and SH_TC_RASTERIZER_AVAILABLE:
        rasterizer = _sh_tc_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_tc_mode and TC_RASTERIZER_AVAILABLE:
        rasterizer = _tc_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_fp16_mode and FP16_RASTERIZER_AVAILABLE:
        rasterizer = _fp16_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    elif is_3D_direct_lean_mode and LEAN_RASTERIZER_AVAILABLE:
        rasterizer = _lean_rasterizer.GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

    # Get shape parameter for beta/general kernel (if using beta or general kernel)
    # Get flex_beta parameter for flex kernel (if using flex kernel)
    shapes = None
    kernel_type = 0  # 0=gaussian, 1=beta, 2=flex, 3=general, 4=beta_scaled
    if hasattr(pc, 'kernel_type') and pc.kernel_type == "beta" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        shapes = pc.get_shape
        kernel_type = 1
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "beta_scaled" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        # Beta kernel scaled to match 3σ Gaussian extent (r ∈ [0,3] instead of [0,1])
        shapes = pc.get_shape
        kernel_type = 4
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "flex" and hasattr(pc, '_flex_beta') and pc._flex_beta.numel() > 0:
        # For flex kernel, pass per-Gaussian beta via the shapes parameter
        # The CUDA kernel will interpret this as beta instead of shape based on kernel_type=2
        shapes = pc.get_flex_beta
        kernel_type = 2
    elif hasattr(pc, 'kernel_type') and pc.kernel_type == "general" and hasattr(pc, '_shape') and pc._shape.numel() > 0:
        # For general kernel (Isotropic Generalized Gaussian), pass beta via shapes parameter
        # Beta in range [2.0, 8.0]: 2.0=standard Gaussian, 8.0=super-Gaussian (box)
        shapes = pc.get_shape
        kernel_type = 3

    # Convert aabb_mode string to int:
    # 0 = square AABB, fixed 4σ cutoff (2DGS default)
    # 1 = square AABB, AdR cutoff (adaptive) - use "adr_only" for this
    # 2 = rectangular AABB, fixed 4σ cutoff
    # 3 = rectangular AABB, AdR cutoff (full optimization) - use "adr" for this
    # 4 = beta kernel: fixed r=1 cutoff (compact support)
    if isinstance(aabb_mode, int):
        aabb_mode_int = aabb_mode
    elif aabb_mode == "adr":
        aabb_mode_int = 3  # Full optimization: AdR + rectangular AABB
    elif aabb_mode == "adr_only":
        aabb_mode_int = 1  # AdR cutoff only (square AABB)
    elif aabb_mode == "rect":
        aabb_mode_int = 2
    elif aabb_mode == "adr_rect":
        aabb_mode_int = 3  # Same as "adr"
    elif aabb_mode == "beta":
        aabb_mode_int = 4  # Beta kernel: fixed r=1 cutoff
    else:
        aabb_mode_int = 0  # "2dgs" or default

    # Build rasterizer kwargs
    rasterizer_kwargs = dict(
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
        shapes = shapes,
        kernel_type = kernel_type,
        aabb_mode = aabb_mode_int,
    )
    # 3D_SH_res rasterizer doesn't accept viewdirs_enc (no view encoding needed)
    if not is_3D_SH_res_mode:
        rasterizer_kwargs['viewdirs_enc'] = viewdirs_enc

    rendered_image, radii, allmap, transmittance_avg, num_covered_pixels, intersection_buffer, intersection_count, geomBuffer = rasterizer(**rasterizer_kwargs)
    
    # 3D mode: Process intersection buffer through PyTorch pipeline
    # Recompute xyz from s_x,s_y → hash encode → gather features → MLP → SH → blend → eval
    if is_3D_mode and ingp is not None and intersection_buffer is not None:
        from utils.general_utils import build_rotation
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

        # intersection_buffer: [H*W * max_per_pixel, 9]
        # Format: (gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G)
        # T and G are stored directly from CUDA for correct gradient computation
        # intersection_count: [H*W] - number of valid intersections per pixel

        # Create mask for valid intersections (non-padded entries)
        counts = intersection_count  # [H*W]
        total_slots = intersection_buffer.shape[0]  # H*W * max_per_pixel

        # Build valid mask by checking if each slot index < count for its pixel
        slot_indices = torch.arange(total_slots, device="cuda")
        local_indices = slot_indices % max_intersections_per_pixel
        pixel_ids_from_slot = slot_indices // max_intersections_per_pixel
        valid_mask = local_indices < counts[pixel_ids_from_slot]

        # Extract valid intersections
        valid_buffer = intersection_buffer[valid_mask]  # [M, 9] where M = total valid
        total_M = valid_buffer.shape[0]

        if total_M > 0:
            # Unpack intersection data (12-field layout)
            # [gaussian_id, weight, pixel_id, xyz.x, xyz.y, xyz.z, s_x, s_y, rho_flag, alpha, T, G]
            gaussian_ids = valid_buffer[:, 0].view(torch.int32).long()  # [M]
            weights_raw = valid_buffer[:, 1].contiguous().float()       # [M] weight = alpha * T
            pixel_ids = valid_buffer[:, 2].view(torch.int32).long()     # [M]
            xyz_x = valid_buffer[:, 3].contiguous().float()             # [M] xyz.x from CUDA
            xyz_y = valid_buffer[:, 4].contiguous().float()             # [M] xyz.y from CUDA
            xyz_z = valid_buffer[:, 5].contiguous().float()             # [M] xyz.z from CUDA
            s_x = valid_buffer[:, 6].contiguous().float()               # [M] disk coord for backward
            s_y = valid_buffer[:, 7].contiguous().float()               # [M] disk coord for backward
            rho_flag = valid_buffer[:, 8].contiguous().float()          # [M] 1.0=disk, 0.0=center
            alpha_all = valid_buffer[:, 9].contiguous().float()         # [M] alpha = opacity * G
            T_all = valid_buffer[:, 10].contiguous().float()            # [M] transmittance (from buffer)
            G_all = valid_buffer[:, 11].contiguous().float()            # [M] kernel value (from buffer)

            # Get Gaussian parameters (needed for xyz recomputation)
            means3D = pc.get_xyz           # [N, 3]
            scales = pc.get_scaling        # [N, 2] - activated
            # CRITICAL: Must use get_rotation (normalized), not _rotation (raw)!
            # The CUDA transMat_to_scale_rot_grad kernel expects normalized quaternions
            # to match native CAT mode's preprocessCUDA which normalizes before computing R.
            rotations = pc.get_rotation    # [N, 4] - normalized quaternions
            opacity_activated = pc.get_opacity.squeeze(-1)  # [N]
            gaussian_features = pc.get_gaussian_features    # [N, gauss_dim]

            # Precompute rotation matrices for all Gaussians
            rotation_matrices = build_rotation(rotations)  # [N, 3, 3]

            # Get camera center for view direction computation
            camera_center = viewpoint_camera.camera_center  # [3]

            # IMPORTANT: Connect weights to opacity, scale, rotation, AND position gradient ONCE for ALL intersections
            # This ensures the transmittance chain effect is computed correctly across all pixels
            # Uses unified backward_from_weight_grad kernel that reads transMat from geomBuffer
            # NOW ALSO computes xyz so that grad_xyz flows back from hash/MLP chain (matching cat mode's dL_duv path)
            # Returns (weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad)
            # rotation_with_grad is now quaternion gradients [N, 4] for proper coordinate conversion
            projmatrix = viewpoint_camera.full_proj_transform  # [4, 4] for proper coordinate conversion
            viewmatrix = viewpoint_camera.world_view_transform  # [4, 4] for normal gradient transform
            weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad = IntersectionOpacityGrad.apply(
                opacity_activated, scales, rotations, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weights_raw, T_all, G_all, alpha_all, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
            )

            # Initialize accumulators for batched processing
            # NOTE: Blend RGB, not SH! SH evaluation must happen per-intersection.
            blended_rgb = torch.zeros(H * W, 3, device="cuda", dtype=torch.float32)
            total_weight = torch.zeros(H * W, device="cuda", dtype=torch.float32)

            # Process MLP/hash in batches to avoid OOM
            # The gradient connection is already set up above for ALL intersections
            batch_size = 5000000  # Tune based on VRAM (matched to 3D_direct)
            for start_idx in range(0, total_M, batch_size):
                end_idx = min(start_idx + batch_size, total_M)

                # Slice batch
                b_gids = gaussian_ids[start_idx:end_idx]
                b_weights = weights_with_grad[start_idx:end_idx]  # Use weights with grad from single call
                b_pids = pixel_ids[start_idx:end_idx]

                # Use xyz computed in IntersectionOpacityGrad (with gradients for hash/MLP chain)
                # This allows grad_xyz to flow back through the hash → MLP → SH → RGB → loss path
                # IntersectionOpacityGrad.backward will capture grad_xyz and compute dL_duv for the CUDA kernel
                b_xyz = xyz_with_grad[start_idx:end_idx]  # [B, 3] - has gradient for hash/MLP chain

                # Hash encoding (batched tcnn query)
                # If detach_hash_grad is set, detach xyz before hash query to prevent
                # hash gradients from flowing to position (matches CUDA behavior in cat mode)
                if ingp.hash_encoding is not None:
                    b_xyz_for_hash = b_xyz.detach() if detach_hash_grad else b_xyz
                    b_hash = ingp._encode_3D(b_xyz_for_hash).float()  # [B, hash_dim]
                else:
                    b_hash = torch.zeros(end_idx - start_idx, 0, device="cuda", dtype=torch.float32)

                # Per-Gaussian features
                b_gauss = gaussian_features[b_gids].float()  # [B, gauss_dim]

                # Apply decompose_mode for debugging visualization (3D mode)
                if decompose_mode == 'gaussian_only':
                    b_hash = torch.zeros_like(b_hash)  # Zero out hash features
                elif decompose_mode == 'ngp_only':
                    b_gauss = torch.zeros_like(b_gauss)  # Zero out Gaussian features

                # Concatenate and MLP → SH coefficients
                if b_hash.shape[1] > 0:
                    b_combined = torch.cat([b_gauss, b_hash], dim=-1)
                else:
                    b_combined = b_gauss
                b_sh = ingp.mlp_3D(b_combined).float().view(-1, 3, 16)  # [B, 3, 16]

                # Compute view direction: from xyz to camera (matches CUDA convention)
                b_viewdir = torch_F.normalize(b_xyz - camera_center[None, :], dim=-1)  # [B, 3]

                # Evaluate SH per intersection to get RGB (correct order: eval first, blend after)
                b_rgb = eval_sh(3, b_sh, b_viewdir)  # [B, 3]

                # Add SH DC offset and clamp per-intersection (matches CUDA behavior)
                b_rgb = torch.clamp(b_rgb + 0.5, 0.0, None)  # [B, 3]

                # Weighted accumulation of RGB (not SH!)
                b_weights_f32 = b_weights.float()
                b_weighted_rgb = b_rgb * b_weights_f32[:, None]
                blended_rgb.scatter_add_(0, b_pids[:, None].expand_as(b_weighted_rgb), b_weighted_rgb)
                total_weight.scatter_add_(0, b_pids, b_weights_f32)

            # Alpha compositing with background (no +0.5 needed since we added it per-intersection)
            T_final = 1.0 - total_weight
            rgb = blended_rgb + T_final[:, None] * bg_color[None, :].float()
            rgb = torch.clamp(rgb, 0.0, 1.0)

            rendered_image = rgb.view(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        else:
            # No valid intersections - return background
            rendered_image = bg_color.unsqueeze(-1).unsqueeze(-1).expand(3, H, W)

    # 3D_direct mode: Same intersection buffer processing, but direct RGB output (like cat mode's 2D MLP)
    # Uses view encoding instead of SH for view-dependent effects
    if is_3D_direct_mode and ingp is not None and intersection_buffer is not None:
        from utils.general_utils import build_rotation
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

        # intersection_buffer: [H*W * max_per_pixel, 9]
        # Format: (gaussian_id, weight, pixel_id, s_x, s_y, rho_flag, alpha, T, G)
        # T and G are stored directly from CUDA for correct gradient computation
        # intersection_count: [H*W] - number of valid intersections per pixel

        # Create mask for valid intersections (non-padded entries)
        counts = intersection_count  # [H*W]
        total_slots = intersection_buffer.shape[0]  # H*W * max_per_pixel

        # Build valid mask by checking if each slot index < count for its pixel
        slot_indices = torch.arange(total_slots, device="cuda")
        local_indices = slot_indices % max_intersections_per_pixel
        pixel_ids_from_slot = slot_indices // max_intersections_per_pixel
        valid_mask = local_indices < counts[pixel_ids_from_slot]

        # Extract valid intersections
        valid_buffer = intersection_buffer[valid_mask]  # [M, 9] where M = total valid
        total_M = valid_buffer.shape[0]

        if total_M > 0:
            # Unpack intersection data (12-field layout)
            # [gaussian_id, weight, pixel_id, xyz.x, xyz.y, xyz.z, s_x, s_y, rho_flag, alpha, T, G]
            gaussian_ids = valid_buffer[:, 0].view(torch.int32).long()  # [M]
            weights_raw = valid_buffer[:, 1].contiguous().float()       # [M] weight = alpha * T
            pixel_ids = valid_buffer[:, 2].view(torch.int32).long()     # [M]
            xyz_x = valid_buffer[:, 3].contiguous().float()             # [M] xyz.x from CUDA
            xyz_y = valid_buffer[:, 4].contiguous().float()             # [M] xyz.y from CUDA
            xyz_z = valid_buffer[:, 5].contiguous().float()             # [M] xyz.z from CUDA
            s_x = valid_buffer[:, 6].contiguous().float()               # [M] disk coord for backward
            s_y = valid_buffer[:, 7].contiguous().float()               # [M] disk coord for backward
            rho_flag = valid_buffer[:, 8].contiguous().float()          # [M] 1.0=disk, 0.0=center
            alpha_all = valid_buffer[:, 9].contiguous().float()         # [M] alpha = opacity * G
            T_all = valid_buffer[:, 10].contiguous().float()            # [M] transmittance (from buffer)
            G_all = valid_buffer[:, 11].contiguous().float()            # [M] kernel value (from buffer)

            # Get Gaussian parameters (needed for xyz recomputation)
            means3D = pc.get_xyz           # [N, 3]
            scales = pc.get_scaling        # [N, 2] - activated
            # CRITICAL: Must use get_rotation (normalized), not _rotation (raw)!
            # The CUDA transMat_to_scale_rot_grad kernel expects normalized quaternions
            # to match native CAT mode's preprocessCUDA which normalizes before computing R.
            rotations = pc.get_rotation    # [N, 4] - normalized quaternions
            opacity_activated = pc.get_opacity.squeeze(-1)  # [N]
            gaussian_features = pc.get_gaussian_features    # [N, gauss_dim]

            # Precompute rotation matrices for all Gaussians
            rotation_matrices = build_rotation(rotations)  # [N, 3, 3]

            # Get camera center for view direction computation
            camera_center = viewpoint_camera.camera_center  # [3]

            # IMPORTANT: Connect weights to opacity, scale, rotation, AND position gradient ONCE for ALL intersections
            # This ensures the transmittance chain effect is computed correctly across all pixels
            # Uses unified backward_from_weight_grad kernel that reads transMat from geomBuffer
            # NOW ALSO computes xyz so that grad_xyz flows back from hash/MLP chain (matching cat mode's dL_duv path)
            # Returns (weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad)
            # rotation_with_grad is now quaternion gradients [N, 4] for proper coordinate conversion
            projmatrix = viewpoint_camera.full_proj_transform  # [4, 4] for proper coordinate conversion
            viewmatrix = viewpoint_camera.world_view_transform  # [4, 4] for normal gradient transform
            weights_with_grad, xyz_with_grad, scales_with_grad, rotation_with_grad, screenspace_with_grad = IntersectionOpacityGrad.apply(
                opacity_activated, scales, rotations, rotation_matrices, screenspace_points, means3D,
                projmatrix, viewmatrix, geomBuffer, weights_raw, T_all, G_all, alpha_all, s_x, s_y, rho_flag, gaussian_ids, pixel_ids, W, H
            )

            # Initialize accumulators for batched processing
            blended_rgb = torch.zeros(H * W, 3, device="cuda", dtype=torch.float32)
            total_weight = torch.zeros(H * W, device="cuda", dtype=torch.float32)

            # Compute per-pixel ray directions ONCE (same as CAT mode)
            # This ensures view encoding is identical to CAT mode: per-pixel, no xyz gradient
            rays_d, _ = cam2rays(viewpoint_camera)  # [H*W, 3]
            ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # [H*W, 3] normalized

            # Process MLP/hash in batches to avoid OOM
            # The gradient connection is already set up above for ALL intersections
            batch_size = 5000000  # Tune based on VRAM
            for start_idx in range(0, total_M, batch_size):
                end_idx = min(start_idx + batch_size, total_M)

                # Slice batch
                b_gids = gaussian_ids[start_idx:end_idx]
                b_weights = weights_with_grad[start_idx:end_idx]  # Use weights with grad from single call
                b_pids = pixel_ids[start_idx:end_idx]

                # Use xyz computed in IntersectionOpacityGrad (with gradients for hash/MLP chain)
                # This allows grad_xyz to flow back through the hash → MLP → RGB → loss path
                # IntersectionOpacityGrad.backward will capture grad_xyz and compute dL_duv for the CUDA kernel
                b_xyz = xyz_with_grad[start_idx:end_idx]  # [B, 3] - has gradient for hash/MLP chain

                # Hash encoding (batched tcnn query)
                # If detach_hash_grad is set, detach xyz before hash query to prevent
                # hash gradients from flowing to position (matches CUDA behavior in cat mode)
                if ingp.hash_encoding is not None:
                    b_xyz_for_hash = b_xyz.detach() if detach_hash_grad else b_xyz
                    b_hash = ingp._encode_3D(b_xyz_for_hash).float()  # [B, hash_dim]
                else:
                    b_hash = torch.zeros(end_idx - start_idx, 0, device="cuda", dtype=torch.float32)

                # Per-Gaussian features
                b_gauss = gaussian_features[b_gids].float()  # [B, gauss_dim]

                # View direction: use per-pixel ray_unit (same as CAT mode)
                # Index by pixel_id to get the ray direction for each intersection
                # This is per-PIXEL (same for all intersections at a pixel), no gradient to xyz
                b_viewdir = ray_unit[b_pids]  # [B, 3]

                # Encode view direction (same as cat mode's 2D MLP)
                b_viewdir_enc = ingp._encode_view(b_viewdir)  # [B, view_enc_dim]

                # Apply decompose_mode for debugging visualization (3D_direct mode)
                if decompose_mode == 'gaussian_only':
                    b_hash = torch.zeros_like(b_hash)  # Zero out hash features
                elif decompose_mode == 'ngp_only':
                    b_gauss = torch.zeros_like(b_gauss)  # Zero out Gaussian features

                # Concatenate: [per-Gaussian features | hash features | view encoding]
                if b_hash.shape[1] > 0:
                    b_combined = torch.cat([b_gauss, b_hash, b_viewdir_enc], dim=-1)
                else:
                    b_combined = torch.cat([b_gauss, b_viewdir_enc], dim=-1)

                # Query MLP for RGB directly (no SH step)
                b_rgb = torch.sigmoid(ingp.mlp_3D_direct(b_combined).float())  # [B, 3]

                # Weighted accumulation of RGB
                b_weights_f32 = b_weights.float()
                b_weighted_rgb = b_rgb * b_weights_f32[:, None]
                blended_rgb.scatter_add_(0, b_pids[:, None].expand_as(b_weighted_rgb), b_weighted_rgb)
                total_weight.scatter_add_(0, b_pids, b_weights_f32)

            # Alpha compositing with background
            T_final = 1.0 - total_weight
            rgb = blended_rgb + T_final[:, None] * bg_color[None, :].float()
            rgb = torch.clamp(rgb, 0.0, 1.0)

            rendered_image = rgb.view(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        else:
            # No valid intersections - return background
            rendered_image = bg_color.unsqueeze(-1).unsqueeze(-1).expand(3, H, W)

    # additional regularizations
    render_alpha = allmap[1:2]

    # Fast inference mode: skip expensive post-processing (normals, etc.)
    # But still do MLP decode if needed (cat mode returns features, not RGB)
    if fast_inference:
        # Check if MLP decode is needed (rendered_image has >3 channels = features)
        if rendered_image.shape[0] > 3 and ingp is not None:
            H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
            feat_dim = rendered_image.shape[0]
            rays_d, _ = cam2rays(viewpoint_camera)
            ray_unit = torch_F.normalize(rays_d, dim=-1).float()  # Normalize ray directions
            fg_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)
            rendered_image = ingp.rgb_decode(fg_features, ray_unit)  # (H*W, 3)
            rendered_image = rendered_image.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
            # Apply render_mask to zero out pixels with no Gaussian coverage
            render_mask = (render_alpha > 0)
            rendered_image = rendered_image * render_mask

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "render_alpha": render_alpha,
            "render_depth": allmap[0:1],  # expected depth (unnormalized)
        }

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
    
    # Save raw features before MLP decode (if requested)
    raw_features_saved = None
    if return_raw_features and hash_in_CUDA:
        raw_features_saved = rendered_image.clone()  # (feat_dim, H, W)

    # Diffuse/Specular mode: no MLP decoding needed, rendered_image is already RGB from SH
    # Baseline mode: decode hashgrid features through MLP
    # ONLY when hash_in_CUDA is True (not during warmup SH rendering)
    elif ingp is not None and hash_in_CUDA and not is_diffuse_mode and not is_specular_mode and not is_diffuse_ngp_mode and not is_diffuse_offset_mode and not is_3D_direct_fused_mode:
        # Hashgrid rendering active:
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
        
        # Cat mode decomposition: mask out features for visualization
        if decompose_mode is not None and is_cat_mode and hybrid_levels > 0:
            per_level_dim = ingp.level_dim
            gaussian_feat_dim = hybrid_levels * per_level_dim
            hashgrid_feat_dim = (ingp.levels - hybrid_levels) * per_level_dim
            
            if decompose_mode == 'gaussian_only':
                # Zero out hashgrid features (last hashgrid_feat_dim channels)
                rendered_image[gaussian_feat_dim:, :, :] = 0
            elif decompose_mode == 'ngp_only':
                # Zero out per-Gaussian features (first gaussian_feat_dim channels)
                rendered_image[:gaussian_feat_dim, :, :] = 0
        
        if skip_mlp:
            # Skip MLP decode, just return zeros (for benchmarking rasterizer only)
            rendered_image = torch.zeros(3, H, W, device="cuda")
        elif background_mode == "skybox_sparse":
            # SPARSE MLP: Only decode pixels with Gaussian coverage
            # This avoids running MLP on sky/background pixels that will be replaced by skybox
            valid_mask = render_mask.squeeze()  # (H, W) bool
            n_valid = valid_mask.sum().item()

            if n_valid > 0:
                # Gather valid features and directions
                features_flat = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)
                valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]  # (D,) indices
                valid_features = features_flat[valid_indices]  # (D, F)
                valid_dirs = rays_dir[valid_indices]  # (D, 3)

                # MLP decode only valid pixels
                valid_rgb = ingp.rgb_decode(valid_features, valid_dirs)  # (D, 3)

                # Scatter back to full image
                rendered_image = torch.zeros(H * W, 3, device="cuda")
                rendered_image[valid_indices] = valid_rgb
                rendered_image = rendered_image.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)
            else:
                # No valid pixels, return zeros
                rendered_image = torch.zeros(3, H, W, device="cuda")
        else:
            # DENSE MLP: Decode all pixels (standard path)
            fg_features = rendered_image.view(feat_dim, -1).permute(1, 0)  # (H*W, F)

            # Background hashgrid compositing
            if bg_hashgrid is not None:
                # Query background hashgrid at ray directions with camera position
                ray_unit_bg = torch_F.normalize(rays_dir, dim=-1).float()  # (H*W, 3)
                # rays_o is (3,) camera position, expand to (H*W, 3)
                ray_origins_bg = rays_o.unsqueeze(0).expand(ray_unit_bg.shape[0], -1)
                bg_features = bg_hashgrid(ray_unit_bg, ray_origins_bg)  # (H*W, F)

                # Get alpha for compositing (H*W, 1)
                alpha_flat = render_alpha.view(-1, 1)  # (H*W, 1)

                if background_mode == "hashgrid_sep":
                    # SEPARATE DECODE: Decode FG and BG separately, composite in RGB space
                    # This prevents BG from learning subtractive features
                    fg_rgb = ingp.rgb_decode(fg_features, rays_dir)  # (H*W, 3)
                    bg_rgb = ingp.rgb_decode(bg_features, rays_dir)  # (H*W, 3)
                    # Composite in RGB space: FG is already alpha-premultiplied from rasterizer
                    # Use alpha for compositing, NOT render_mask (BG fills empty regions)
                    rendered_image = fg_rgb + (1.0 - alpha_flat) * bg_rgb
                elif background_mode == "hashgrid_relu":
                    # RELU: Apply ReLU to BG features to prevent subtractive features
                    bg_features = torch_F.relu(bg_features)
                    composite_features = fg_features + (1.0 - alpha_flat) * bg_features
                    rendered_image = ingp.rgb_decode(composite_features, rays_dir)
                else:
                    # STANDARD: Composite features before MLP decode (original behavior)
                    composite_features = fg_features + (1.0 - alpha_flat) * bg_features
                    rendered_image = ingp.rgb_decode(composite_features, rays_dir)

                # BG hashgrid handles empty regions via (1-alpha) compositing
                # Do NOT apply render_mask here - it would zero out BG contribution
                rendered_image = rendered_image.view(H, W, -1).permute(2, 0, 1)
            else:
                # Standard path without background hashgrid
                rendered_image = ingp.rgb_decode(fg_features, rays_dir)
                rendered_image = rendered_image.view(H, W, -1).permute(2, 0, 1)
                rendered_image = rendered_image * render_mask

        vis_appearance_level = allmap[11:14]

    # Background compositing: skybox or solid color (legacy path for skybox texture)
    # Initialize FG/BG outputs for visualization
    rendered_image_fg = None
    skybox_rgb = None

    # Skip background compositing for 3D/3D_direct modes - they already added background in Python
    if not is_3D_mode and not is_3D_direct_mode:
        if skybox is not None:
            # Compute ray directions if not already available
            H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
            rays_d, rays_o = cam2rays(viewpoint_camera)
            ray_unit_skybox = torch_F.normalize(rays_d, dim=-1).float()

            # Query skybox for background colors
            skybox_rgb = skybox(ray_unit_skybox)  # (H*W, 3)
            skybox_rgb = skybox_rgb.view(H, W, 3).permute(2, 0, 1)  # (3, H, W)

            # Save foreground before compositing (for visualization)
            rendered_image_fg = rendered_image.clone()

            # Composite: foreground + (1-alpha) * skybox
            rendered_image = rendered_image + (1.0 - render_alpha) * skybox_rgb
        elif bg_color.sum() > 0:
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

    # Add skybox FG/BG outputs for visualization
    if rendered_image_fg is not None:
        rets['render_fg'] = rendered_image_fg
    if skybox_rgb is not None:
        rets['render_bg'] = skybox_rgb

    # Add raw features (before MLP decode) if requested
    if raw_features_saved is not None:
        rets['raw_features'] = raw_features_saved

    return rets
