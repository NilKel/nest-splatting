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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Depth sort toggle (defined in forward.cu)
// Host-side depth sort toggle (set via SetDepthSortCUDA in rasterize_points.cu)
bool g_depth_sort = false;

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	int* radii_x,  // Separate X radius for rectangular AABB
	int* radii_y,  // Separate Y radius for rectangular AABB
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// Use separate X/Y radii for rectangular AABB bounds
		getRectXY(points_xy[idx], radii_x[idx], radii_y[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void duplicateKeysWithTileDepth(
	int P,
	const float2* points_xy,
	const float* depths,
	const float* transMats,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	int* radii_x,  // Separate X radius for rectangular AABB
	int* radii_y,  // Separate Y radius for rectangular AABB
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// Use separate X/Y radii for rectangular AABB bounds
		getRectXY(points_xy[idx], radii_x[idx], radii_y[idx], rect_min, rect_max, grid);

		// Tile based depth sort
		// code from renderCUDA part.
		const float2 xy =  points_xy[idx];
		const float3 Tu = {transMats[9 * idx+0], transMats[9 * idx+1], transMats[9 * idx+2]};
		const float3 Tv = {transMats[9 * idx+3], transMats[9 * idx+4], transMats[9 * idx+5]};
		const float3 Tw = {transMats[9 * idx+6], transMats[9 * idx+7], transMats[9 * idx+8]};

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// pixel center of this tile
				const float pixx = BLOCK_X * x + BLOCK_X / 2;
				const float pixy = BLOCK_Y * y + BLOCK_Y / 2;
				
				float3 k = pixx * Tw - Tu;
				float3 l = pixy * Tw - Tv;
				float3 p = cross(k, l);
				if (p.z == 0.0) continue;
				float2 s = {p.x / p.z, p.y / p.z};
				float tile_depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;

				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&tile_depth);
				// key |= *((uint32_t*)&depths[idx]);

				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Initialize depth sort: keys = depth-as-uint32, values = identity [0..P-1]
__global__ void initDepthSortKeys(int P, const float* depths, uint32_t* keys, uint32_t* values)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	keys[idx] = *((uint32_t*)&depths[idx]);
	values[idx] = idx;
}

// Gather tiles_touched in depth-sorted order
__global__ void gatherTilesTouched(int P, const uint32_t* tiles_touched, const uint32_t* depth_order, uint32_t* tiles_touched_gathered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	tiles_touched_gathered[idx] = tiles_touched[depth_order[idx]];
}

// Like duplicateWithKeys but iterates Gaussians in depth-sorted order.
// Keys contain only tile_id (no depth), relying on insertion order for within-tile depth sort.
__global__ void duplicateWithKeysSorted(
	int P,
	const float2* points_xy,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	const int* radii_x,
	const int* radii_y,
	const uint32_t* depth_order,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Map sorted position to original Gaussian index
	const uint32_t orig_idx = depth_order[idx];

	if (radii[orig_idx] > 0)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		getRectXY(points_xy[orig_idx], radii_x[orig_idx], radii_y[orig_idx], rect_min, rect_max, grid);

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				// Lower 32 bits left as 0 — within-tile depth order
				// is guaranteed by iterating in depth-sorted Gaussian order
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = orig_idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.radii_x, P, 128);  // Separate X radius for rectangular AABB
	obtain(chunk, geom.radii_y, P, 128);  // Separate Y radius for rectangular AABB
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);

	// Depth sort buffers
	obtain(chunk, geom.depth_order, P, 128);
	obtain(chunk, geom.depth_sort_buf1, P, 128);
	obtain(chunk, geom.depth_sort_buf2, P, 128);
	obtain(chunk, geom.depth_sort_buf3, P, 128);
	cub::DeviceRadixSort::SortPairs(nullptr, geom.depth_sort_size,
		geom.depth_sort_buf1, geom.depth_sort_buf2,
		geom.depth_sort_buf3, geom.depth_order, P);
	obtain(chunk, geom.depth_sort_workspace, geom.depth_sort_size, 128);

	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	const __half* hash_features,
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
	int* radii,
	float* cover_pixels,
	float* trans_avg,
	bool debug,
	const float beta,
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	const int render_mode,
	const uint32_t max_intersections,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	const float aa,
	const float aa_threshold,
	float* intersection_buffer,
	uint32_t* intersection_count,
	uint32_t max_intersections_per_pixel,
	const int* metric_map,
	int* metric_counts)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.radii_x,
		geomState.radii_y,
		geomState.means2D,
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode,
		render_mode
	), debug)

	bool use_depth_sort = g_depth_sort;

	int num_rendered;

	if (use_depth_sort) {
		// === Separated depth sort: sort Gaussians by depth first, then tile-bin ===

		// Step 1: Sort P Gaussians by depth (32-bit sort on P entries — much smaller than num_rendered)
		initDepthSortKeys << <(P + 255) / 256, 256 >> > (
			P, geomState.depths, geomState.depth_sort_buf1, geomState.depth_sort_buf3);
		CHECK_CUDA(, debug)

		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			geomState.depth_sort_workspace, geomState.depth_sort_size,
			geomState.depth_sort_buf1, geomState.depth_sort_buf2,
			geomState.depth_sort_buf3, geomState.depth_order,
			P, 0, 32), debug)

		// Step 2: Gather tiles_touched in depth-sorted order, then prefix sum
		gatherTilesTouched << <(P + 255) / 256, 256 >> > (
			P, geomState.tiles_touched, geomState.depth_order, geomState.depth_sort_buf1);
		CHECK_CUDA(, debug)

		// Prefix sum on gathered (depth-sorted) tiles_touched
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
			geomState.depth_sort_buf1, geomState.point_offsets, P), debug)
	} else {
		// === Standard sort: prefix sum on tiles_touched directly ===
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
			geomState.tiles_touched, geomState.point_offsets, P), debug)
	}

	// Retrieve total number of Gaussian instances
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	if (use_depth_sort) {
		// Step 3: Duplicate in depth-sorted order — entries within each tile are depth-ordered
		// Keys contain tile_id only (upper 32 bits), no depth in lower bits
		duplicateWithKeysSorted << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			geomState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			radii,
			geomState.radii_x,
			geomState.radii_y,
			geomState.depth_order,
			tile_grid)
		CHECK_CUDA(, debug)

		int bit = getHigherMsb(tile_grid.x * tile_grid.y);

		// Step 4: Sort by tile_id only (bits [32, 32+bit]) — stable sort preserves within-tile depth order
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 32, 32 + bit), debug)
	} else {
		// Standard: duplicateWithKeys encodes tile_id|depth, full radix sort
		duplicateWithKeys << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			geomState.depths,
			geomState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			radii,
			geomState.radii_x,
			geomState.radii_y,
			tile_grid)
		CHECK_CUDA(, debug)

		int bit = getHigherMsb(tile_grid.x * tile_grid.y);

		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 0, 32 + bit), debug)
	}

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		beta,
		width, height,
		c_dim, level, l_dim, l_scale, Base, align_corners, interp,
		if_contract, record_transmittance,
		focal_x, focal_y,
		(glm::vec2*)scales,
		means3D,
		geomState.means2D,
		feature_ptr,
		transMat_ptr,
		homotrans,
		ap_level,
		hash_features,
		level_offsets,
		gridrange,
		geomState.depths,
		geomState.normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_others,
		out_index,
		cover_pixels,
		trans_avg,
		(glm::vec3*)cam_pos,
		D_diffuse,
		hash_features_diffuse,
		level_offsets_diffuse,
		gridrange_diffuse,
		render_mode,
		max_intersections,
		shapes,
		kernel_type,
		aa,
		aa_threshold,
		intersection_buffer,
		intersection_count,
		max_intersections_per_pixel,
		nullptr,  // viewdirs_enc
		((render_mode & 0xFF) == 6) ? geomState.rgb : nullptr,  // rgb_override: mode 6 needs SH colors separately from features (which has DC SH)
		metric_map,
		metric_counts
		), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	const __half* hash_features,
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
	char* img_buffer,
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
	const uint32_t D_diffuse,
	const float* hash_features_diffuse,
	const int* level_offsets_diffuse,
	const float* gridrange_diffuse,
	float* dL_dfeatures_diffuse,
	const int render_mode,
	const float* shapes,
	const int kernel_type,
	float* dL_dshapes,
	const bool detach_hash_grad,
	// MLP gradient outputs for 3D_SH_res (render_mode=5, bias-free, all [16×16])
	float* dL_dmlp_W1,
	float* dL_dmlp_W2,
	float* dL_dmlp_W3)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// render_mode 6 (3D_SH_cat): colors_precomp has DC SH, but backward needs full SH eval from rgb
	const float* color_ptr = (colors_precomp != nullptr && (render_mode & 0xFF) != 6) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		beta,
		width, height,
		c_dim, level, l_dim, l_scale, Base, align_corners, interp, if_contract,
		focal_x, focal_y,
		(glm::vec2*)scales,
		other_maps,
		out_index,
		background,
		geomState.means2D,
		geomState.normal_opacity,
		color_ptr,
		transMat_ptr,
		homotrans,
		ap_level,
		hash_features,
		level_offsets,
		gridrange,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_depths,
		dL_dfeatures,
		dL_dtransMat,
		dL_dhomoMat,
		(float4*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor,
		dL_gradsum,
		(glm::vec3*)campos,
		D_diffuse,
		hash_features_diffuse,
		level_offsets_diffuse,
		gridrange_diffuse,
		dL_dfeatures_diffuse,
		render_mode,
		shapes,
		kernel_type,
		dL_dshapes,
		detach_hash_grad,
		dL_dmlp_W1, dL_dmlp_W2, dL_dmlp_W3,
		((render_mode & 0xFF) == 6) ? colors_precomp : nullptr), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float4*)dL_dmean2D, // gradient inputs
		dL_dnormal,		     // gradient inputs
		dL_dtransMat,
		dL_dhomoMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot,
		(render_mode & 0x800) != 0), debug)
}
