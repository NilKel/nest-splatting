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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include "rgb_type.h"  // rgb_t typedef (FP16/FP32 via FP16_RGB) — does not pull in cub-incompatible config.h
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		int* radii_x;  // Separate X radius for rectangular AABB
		int* radii_y;  // Separate Y radius for rectangular AABB
		float2* means2D;
		float* transMat;
		float4* normal_opacity;
		rgb_t* rgb;  // FP16 or FP32 per config.h::FP16_RGB
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		// SnugBox+AccuTile (aabb_mode==5). conic_t[idx] = (A, B, E, t); w (==t)
		// > 0 signals "use AccuTile in emit"; w == 0 signals "use rect AABB
		// fallback" (degenerate conic, or aabb_mode != 5).
		float4* conic_t;

		// `--method mixed_3d`: per-Gauss EWA 2D conic for untextured 3D-ellipsoid
		// surfels. ewa_conic[idx] = (a, b, c, opacity); written by preprocessCUDA
		// only when scaling_z != nullptr and is_textured[idx]==false (else unused).
		float4* ewa_conic;

		// Separated depth sort: pre-sort Gaussians by depth, then sort expanded list by tile_id only
		uint32_t* depth_order;          // [P] maps sorted position → original Gaussian index
		uint32_t* depth_sort_buf1;      // [P] temp: depth keys unsorted / tiles_touched gathered
		uint32_t* depth_sort_buf2;      // [P] temp: depth keys sorted
		uint32_t* depth_sort_buf3;      // [P] temp: identity values
		size_t depth_sort_size;
		char* depth_sort_workspace;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};