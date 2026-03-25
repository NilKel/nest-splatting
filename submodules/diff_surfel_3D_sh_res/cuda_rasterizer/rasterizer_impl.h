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
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

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