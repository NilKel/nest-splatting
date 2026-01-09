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
		const int aabb_mode = 0);

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
		const float* hash_features,
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
		const uint32_t D_diffuse = 0,
		const float* hash_features_diffuse = nullptr,
		const int* level_offsets_diffuse = nullptr,
		const float* gridrange_diffuse = nullptr,
		const int render_mode = 0,
		const uint32_t max_intersections = 0,
		const float* shapes = nullptr,
		const int kernel_type = 0,
		const float aa = 0.0f,
		const float aa_threshold = 0.01f);
}


#endif
