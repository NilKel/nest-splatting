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
#include <cuda_fp16.h>

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
		int* internal_radii;
		int* radii_x;
		int* radii_y;
		float2* means2D;
		float* transMat;
		float4* normal_opacity;
		__half* rgb;
		// Per-Gaussian Spherical-Beta contribution. Eval'd once per Gaussian
		// in preprocessCUDA (view_dir is per-Gaussian, identical across pixels)
		// and added in renderBakedCUDA's inner loop. Was per-pixel before —
		// each Gaussian's ~K*4 transcendentals were redundantly re-computed
		// for every pixel it touched (~100×). When sb_number == 0, this
		// buffer is allocated but never written/read.
		__half* sb_rgb;
		float4* conic_t;        // SnugBox+AccuTile conic (A, B, E, t) — used when aabb_mode==2/5
		float4* ewa_conic;      // `--method mixed_3d` per-Gauss EWA conic (a,b,c,opacity) for untextured rows
		// LEAN_CONIC Option A + exact rational correction.  Precomputed in preprocessCUDA
		// at AABB-center pixel.  Layout per Gauss (8 floats = 32 B):
		//   [0..1] (u₀, v₀)         disc coord at AABB center pixel
		//   [2..5] J⁻¹ (2×2)        inverse Jacobian at (u₀, v₀)
		//   [6..7] (Tw.x/w_c, Tw.y/w_c)   for exact rational correction:
		//     u = u₀ + (J⁻¹·Δpix).x / (1 + (dwdxr, dwdyr)·Δpix)  — mathematically exact
		//     (v same).  Both u and v share the same denominator correction factor.
		float* conic_uv;        // [P*8] fp32 per-Gauss conic cache — 32 B/Gauss

		// Legacy single-sort path (sort_mode == 0).
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		// FastGS two-stage sort path (sort_mode == 1).
		// preprocessCUDA atomicAdds visible-Gaussian entries into the *_compact
		// arrays, sized N (only n_visible entries used). Per-Gaussian metadata
		// above (means2D, depths, transMat, ...) stays indexed by the original
		// primitive id; `prim_idx_compact[v]` is the indirection.
		uint32_t* depth_keys_compact;       // [n_visible] float-bit depth (32-bit sort key)
		uint32_t* prim_idx_compact;         // [n_visible] original primitive idx (sort value, unsorted)
		uint32_t* prim_idx_compact_sorted;  // [n_visible] depth-sorted output of SortPairs
		uint32_t* offset_compact;           // [n_visible] exclusive prefix sum of reordered tile counts
		uint32_t* n_visible_atomic;         // single-uint atomic counter
		uint32_t* n_instances_atomic;       // single-uint atomic counter
		// Note: per-primitive tile count is stored in the legacy `tiles_touched`
		// array above (indexed by primitive_idx). apply_depth_ordering reads it
		// indirectly via prim_idx_compact_sorted[v] to populate offset_compact.

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		// Legacy 64-bit composite keys (sort_mode == 0).
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		// FastGS 32-bit tile-only keys (sort_mode == 1) — second of two sorts.
		size_t sorting_size_tile;
		uint32_t* tile_keys_unsorted;
		uint32_t* tile_keys;
		// Shared between both modes: per-instance primitive-index list.
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;
		// Workspace + buffers for the depth-sort phase (sort_mode == 1):
		size_t sorting_size_depth;
		char* depth_sorting_space;

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