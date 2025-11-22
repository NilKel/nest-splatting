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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float beta,
		int W, int H,
		uint32_t c_dim, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
		bool align_corners, uint32_t interp, 
		const bool if_contract,
		float focal_x, float focal_y,
		const glm::vec2* scales,
		const float* other_maps,
		const int* out_index,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const float* colors,
		const float* transMats,
		const float* homotrans,
		const float* ap_level,
		const float* hash_features,
		const int* level_offsets,
		const float* gridrange,
		const float* gaussian_features,
		const int render_mode,
		const int hybrid_levels,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_depths,
		float* dL_dfeatures,
		float * dL_dtransMat,
		float * dL_dhomoMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dgaussian_features,
		float* dL_gradsum);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dhomoMat,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
