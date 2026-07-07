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
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
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
		// Separate SH RGB pointer for 3D_SH_cat (render_mode=6)
		const float* rgb_override = nullptr,
		// 3D_SH_concat: per-Gauss 16D surfel latent (cols 0..15 of _film_params beta)
		const float* film_beta = nullptr);

	// Set contribution threshold for hash query skip: w = T*alpha (0 = disabled)
	void setContribThresh(float val);

	// Set count threshold: skip hash after N contributing Gaussians per pixel (0 = disabled)
	void setCountThresh(int val);

	// Set overdraw regularization lambda (0 = disabled)
	void setOverdrawLambda(float val);

	// Set activation biases: color = ReLU(SH + sh_bias) + ReLU(residual + res_bias)
	void setActivationBias(float sh_bias, float res_bias);

	// Leaky slope for the outer activation (0 = ReLU, 1 = identity → signed residual for decomp viz).
	void setLruSlope(float v);

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
