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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			uint32_t c_dim, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
			bool align_corners, uint32_t interp,
			const bool if_contract, const bool record_transmittance,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* homotrans,
			const float* ap_level,
			const float* hash_features,
			const int* level_offsets,
			const float* gridrange,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			int* out_index,
			int* radii = nullptr,
			float* cover_pixels = nullptr,
			float* trans_avg = nullptr,
			bool debug = false,
			const float beta = 0.0,
			const uint32_t D_diffuse = 0,
			const float* hash_features_diffuse = nullptr,
			const int* level_offsets_diffuse = nullptr,
			const float* gridrange_diffuse = nullptr,
			const int render_mode = 0,
			const uint32_t max_intersections = 0,
			const float* shapes = nullptr,
			const int kernel_type = 0);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			uint32_t c_dim, uint32_t level, uint32_t l_dim, float l_scale, uint32_t Base,
			bool align_corners, uint32_t interp,
			const bool if_contract,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* homotrans,
			const float* ap_level,
			const float* hash_features,
			const int* level_offsets,
			const float* gridrange,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const float* other_maps,
			const int* out_index,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dfeatures,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_dhomoMat,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_gradsum,
			bool debug,
			const float beta,
			const uint32_t D_diffuse = 0,
			const float* hash_features_diffuse = nullptr,
			const int* level_offsets_diffuse = nullptr,
			const float* gridrange_diffuse = nullptr,
			float* dL_dfeatures_diffuse = nullptr,
			const int render_mode = 0,
			const float* shapes = nullptr,
			const int kernel_type = 0,
			float* dL_dshapes = nullptr);
	};
};

#endif
