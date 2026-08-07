/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "rgb_type.h"  // rgb_t typedef (FP16/FP32 via FP16_RGB)

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		int* radii_x,  // Separate X radius for rectangular AABB
		int* radii_y,  // Separate Y radius for rectangular AABB
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		rgb_t* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		float4* conic_t,
		bool prefiltered,
		const float* shapes = nullptr,
		const int kernel_type = 0,
		const int aabb_mode = 0,
		const int render_mode = 0);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float beta,
		int W, int H,
		uint32_t c_dim, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
		bool align_corners, uint32_t interp,
		const bool if_contract, const bool record_transmittance,
		float focal_x, float focal_y,
		const glm::vec2* scales,
		const float* means3D,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float* homotrans,
		const float* ap_level,
		const __half* hash_features,
		const int* level_offsets,
		const float* gridrange,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others,
		int* out_index,
		float* cover_pixels,
		float* trans_avg,
		const glm::vec3* cam_pos,
		const uint32_t D_diffuse = 0,
		const float* hash_features_diffuse = nullptr,
		const int* level_offsets_diffuse = nullptr,
		const float* gridrange_diffuse = nullptr,
		const int render_mode = 0,
		const uint32_t max_intersections = 0,
		const float* shapes = nullptr,
		const int kernel_type = 0,
		const float aa = 0.0f,
		const float aa_threshold = 0.01f,
		// 3D mode intersection buffer outputs
		float* intersection_buffer = nullptr,      // [H*W * max_intersections_per_pixel, 6]
		uint32_t* intersection_count = nullptr,    // [H*W] actual count per pixel
		uint32_t max_intersections_per_pixel = 0,  // Cap for memory management
		// Pre-encoded view directions for 3D_direct_fused (H*W, 16)
		const float* viewdirs_enc = nullptr,
		// Separate SH RGB pointer for 3D_SH_cat (render_mode=6).
		// FP16 (rgb_t) when config.h::FP16_RGB=1 — see geomState.rgb.
		const rgb_t* rgb_override = nullptr,
		// FastGS VCD/VCP: per-pixel high-error mask [H*W] int32, per-Gaussian
		// counter [P] int32. Both nullptr => feature disabled.
		const int* metric_map = nullptr,
		int* metric_counts = nullptr);

	// Set contribution threshold for hash query skip: w = T*alpha (0 = disabled)
	void setContribThresh(float val);

	// Set count threshold: skip hash after N contributing Gaussians per pixel (0 = disabled)
	void setCountThresh(int val);

	// Set opacity threshold: skip hash when alpha = opa*kernel_val < val (0 = disabled)
	void setOpacityThresh(float val);

	// Set texture-query dropout rate + per-iteration seed (0 rate = disabled, training only)
	void setDropout(float rate, unsigned int seed);

	// Set overdraw regularization lambda (0 = disabled)
	void setOverdrawLambda(float val);

	// Set weight-squared regularization lambda (0 = disabled)
	void setWeightRegLambda(float val);

	// Set activation biases for SH and residual.
	void setActivationBias(float sh_bias, float res_bias);
	// 0 = 3D_SH_res (stacked: ReLU(ReLU(SH+bias)+residual+bias), default).
	// 1 = 3D_SH_add (separate: ReLU(SH+bias) + ReLU(residual+bias)).
	void setResidualMode(int mode);

	// `--ste`: straight-through estimator on the per-Gauss outer ReLU
	// (mode 0 only). When v=1, backward bypasses the clamp gate so the
	// MLP/hashgrid keeps receiving gradient at clamped activations. Default 0.
	void setSteRelu(int v);

	// `--lru`: leaky-ReLU slope α for the outer per-Gauss activation (mode 0).
	// α == 0 (default) reduces to standard ReLU. α > 0 lets negative
	// activations through scaled by α in BOTH forward and backward.
	void setLruSlope(float v);

	// Set Nexels-style anti-aliasing params for hash-grid down-weighting.
	// factor=0 disables AA. Typical factor=1.0, focal=max(fx,fy).
	void setAntiAlias(float factor, float focal);

	// FastGS Compact Box: Mahalanobis² scale factor for AdR cutoff.
	// val=1.0 → matches existing AdR (our current default). val=0.5 → FastGS paper default (tighter tile AABB).
	void setCompactMult(float val);
	// snugbox (mode 5) beta-kernel footprint multiplier (scales the use_beta_cutoff radius).
	void setBetaMult(float val);

	// Set AA-2DGS mip-filter kernel size σ (0 disables, typical 0.1).
	// When >0, replaces the rho3d/rho2d heuristic with the Jacobian-based
	// object-space mip filter in the scalar mode 5/6 Gaussian path.
	void setAaKernelSize(float val);

	// WSR (sort-free weighted-sum) mode. mode=1 enables; occ = [N] activated
	// per-surfel occlusion; aux = [4*H*W] per-render scratch (C̄ + den) shared
	// with the backward. Pointers must be re-installed before every render.
	void setWsr(int mode, const float* occ, float* aux);

	// WSR transmittance gate (2-pass): tau > 0 arms a pre-pass that bins
	// per-pixel Σlog(1−α) by log-depth (zmin/zmax range, ≤16 bins) into tbin
	// [bins*H*W] as an exclusive-prefix T; the render then discards fragments
	// whose bin transmittance < tau. tau ≤ 0 or tbin == nullptr disables.
	void setWsrGate(float tau, int bins, float zmin, float zmax, float* tbin);

	// WSR mean-depth gate (?wsr=3): margin > 0 arms a SORTED pre-pass that
	// writes (D̄ = ΣαTd/ΣαT, A = ΣαT) to dbuf [2*H*W]; the render fades the
	// occ weight behind D̄·(1+margin/2 .. +3·margin/2), scaled by g(A).
	void setWsrDGate(float margin, float* dbuf);

	// Copy MLP weights to global device memory (bias-free, all [16×16])
	void setMlpWeights(
		const float* W1,   // [16×16]
		const float* W2,   // [16×16]
		const float* W3);  // [16×16] (only first 3 rows = RGB residual)

	// Get MLP weight device pointers for passing to backward kernel (bias-free, FP16)
	void getMlpWeightPointers(
		__half** W1,
		__half** W2,
		__half** W3);
}


#endif
