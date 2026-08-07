/*
 * Baked rendering submodule — forward-only rasterizer implementation.
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
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

// Helper function to find the next-highest bit of the MSB on the CPU.
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

// Tile-Gaussian key emission. Two paths, dispatched per-Gaussian via conic_t:
//   conic_t.w > 0  → SnugBox+AccuTile: ellipse-tight scan-line emission
//   conic_t.w == 0 → rect AABB: enumerate every tile in [radii_x, radii_y]
// preprocessCUDA sets conic_t.w (= t) to 0 when the SnugBox conic was
// degenerate or aabb_mode != 5; this acts as both a "use rect" flag and
// preserves the t value when the SnugBox path succeeded.
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* conic_t,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	int* radii_x,
	int* radii_y,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] <= 0)
		return;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	const float4 ct = conic_t[idx];

	if (ct.w > 0.0f)
	{
		// SnugBox+AccuTile path. duplicateToTilesTouched returns the same
		// tile count as the count-only call in preprocessCUDA, so the offsets
		// from the prefix sum are exact.
		duplicateToTilesTouched(
			ct.x, ct.y, ct.z, ct.w, points_xy[idx], grid,
			(uint32_t)idx, off, depths[idx],
			gaussian_keys_unsorted, gaussian_values_unsorted);
	}
	else
	{
		// Fallback: rect AABB.
		uint2 rect_min, rect_max;
		getRectXY(points_xy[idx], radii_x[idx], radii_y[idx], rect_min, rect_max, grid);

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

// Check keys to see if it is at the start/end of one tile's range.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

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

// =============================================================================
// FastGS two-stage sort kernels (sort_mode == 1)
// =============================================================================

// After depth-sort: write tile counts in depth-sorted order. Feeds the
// prefix-sum that produces per-visible-primitive write offsets for create_instances.
__global__ void apply_depth_ordering(
	const uint32_t* __restrict__ prim_idx_compact_sorted, // depth-sorted prim ids
	const uint32_t* __restrict__ tiles_touched_perid,     // per-primitive tile count (size P)
	uint32_t* __restrict__ offset_compact,                // [n_visible] reordered counts
	const uint32_t n_visible)
{
	const uint32_t v = cg::this_grid().thread_rank();
	if (v >= n_visible) return;
	offset_compact[v] = tiles_touched_perid[prim_idx_compact_sorted[v]];
}

// Per-visible-primitive (in depth-sorted order) tile-instance emission.
// Mirrors the dispatch in legacy duplicateWithKeys but writes 32-bit tile keys
// into a separate buffer (instance_keys) and primitive indices into another
// (instance_prim_indices). The second sort then sorts by tile key with stable
// ordering, preserving depth order within each tile.
__global__ void create_instances(
	const uint32_t* __restrict__ prim_idx_compact_sorted,
	const uint32_t* __restrict__ offset_compact,         // exclusive prefix sum of reordered tile counts
	const float2* __restrict__ points_xy,
	const float4* __restrict__ conic_t,
	const int* __restrict__ radii,
	const int* __restrict__ radii_x,
	const int* __restrict__ radii_y,
	uint32_t* __restrict__ tile_keys_unsorted,
	uint32_t* __restrict__ prim_indices_unsorted,
	const dim3 grid,
	const uint32_t n_visible)
{
	const uint32_t v = cg::this_grid().thread_rank();
	if (v >= n_visible) return;

	const uint32_t idx = prim_idx_compact_sorted[v];
	if (radii[idx] <= 0) return;

	uint32_t off = offset_compact[v];
	const float4 ct = conic_t[idx];

	if (ct.w > 0.0f) {
		// SnugBox+AccuTile path. Emits 32-bit tile-only keys (no depth).
		// Depth ordering is implicit because v is in depth-sorted order and
		// the subsequent radix sort on tile_keys is stable.
		duplicateToTilesTouched(
			ct.x, ct.y, ct.z, ct.w, points_xy[idx], grid,
			(uint32_t)idx, off, 0.0f,
			nullptr, nullptr,
			tile_keys_unsorted, prim_indices_unsorted);
	} else {
		// Rect AABB fallback.
		uint2 rect_min, rect_max;
		getRectXY(points_xy[idx], radii_x[idx], radii_y[idx], rect_min, rect_max, grid);
		for (int y = rect_min.y; y < rect_max.y; y++) {
			for (int x = rect_min.x; x < rect_max.x; x++) {
				tile_keys_unsorted[off] = (uint32_t)(y * grid.x + x);
				prim_indices_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// 32-bit tile-key version of identifyTileRanges (FastGS path).
__global__ void identifyTileRanges32(int L, const uint32_t* tile_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (uint32_t)L) return;
	const uint32_t currtile = tile_keys[idx];
	if (idx == 0) ranges[currtile].x = 0;
	else {
		const uint32_t prevtile = tile_keys[idx - 1];
		if (currtile != prevtile) {
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == (uint32_t)L - 1) ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P, means3D, viewmatrix, projmatrix, present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.radii_x, P, 128);
	obtain(chunk, geom.radii_y, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);   // FP16 — halves inner-loop fetch bandwidth
	obtain(chunk, geom.sb_rgb, P * 3, 128); // SB per-Gaussian color (preprocessCUDA writes, renderBakedCUDA reads when sb_number>0)
	obtain(chunk, geom.conic_t, P, 128);   // SnugBox conic — used when aabb_mode==2/5
	obtain(chunk, geom.ewa_conic, P, 128); // mixed_3d untextured EWA conic (a,b,c,opacity)
	obtain(chunk, geom.conic_uv, P * 8, 128); // LEAN_CONIC: per-Gauss (u0,v0,J⁻¹,dwdxr,dwdyr) for exact correction
	// Legacy single-sort buffers.
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	// FastGS two-stage sort buffers.
	obtain(chunk, geom.depth_keys_compact, P, 128);
	obtain(chunk, geom.prim_idx_compact, P, 128);
	obtain(chunk, geom.prim_idx_compact_sorted, P, 128);
	obtain(chunk, geom.offset_compact, P, 128);
	obtain(chunk, geom.n_visible_atomic, 1, 128);
	obtain(chunk, geom.n_instances_atomic, 1, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	// Shared per-instance values (sorted-by-tile primitive indices).
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	// Legacy single-sort 64-bit composite keys.
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	// FastGS path: 32-bit tile keys (second sort) + workspace for the depth sort.
	obtain(chunk, binning.tile_keys, P, 128);
	obtain(chunk, binning.tile_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size_tile,
		binning.tile_keys_unsorted, binning.tile_keys,
		binning.point_list_unsorted, binning.point_list, P);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size_depth,
		(uint32_t*)nullptr, (uint32_t*)nullptr,
		(uint32_t*)nullptr, (uint32_t*)nullptr, P);
	obtain(chunk, binning.depth_sorting_space, binning.sorting_size_depth, 128);
	return binning;
}

// Forward rendering — forward-only, no backward
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug,
	const float beta,
	const float* shapes,
	const int kernel_type,
	const __half* atlas_texture,
	const float* atlas_rects,
	const int atlas_width,
	const int aabb_mode,
	const float* sb_params,
	const int sb_number,
	cudaTextureObject_t atlas_tex_obj,
	float atlas_offset,
	float atlas_scale,
	const int sort_mode,
	const float* voronoi_sites,
	const float* voronoi_tau,
	const float* voronoi_colors,
	const int voronoi_K,
	const bool* is_textured,
	const float* scaling_z)
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

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Zero atomic counters before preprocess (FastGS path uses these).
	if (sort_mode == 1) {
		CHECK_CUDA(cudaMemsetAsync(geomState.n_visible_atomic, 0, sizeof(uint32_t)), debug);
		CHECK_CUDA(cudaMemsetAsync(geomState.n_instances_atomic, 0, sizeof(uint32_t)), debug);
	}

	// Preprocess: SH eval, transmat, AABB / SnugBox-AccuTile tile-touched count.
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
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
		geomState.conic_t,
		tile_grid,
		geomState.tiles_touched,
		geomState.depth_keys_compact,
		geomState.prim_idx_compact,
		geomState.n_visible_atomic,
		geomState.n_instances_atomic,
		sort_mode,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode,
		voronoi_sites,
		voronoi_tau,
		voronoi_colors,
		voronoi_K,
		// SB fused: preprocess writes per-Gauss RGB to geomState.sb_rgb; render reads it.
		sb_params,
		sb_number,
		geomState.sb_rgb,
		is_textured,
		scaling_z,
		(scaling_z != nullptr) ? geomState.ewa_conic : nullptr,
		geomState.conic_uv
	), debug)

	int num_rendered = 0;
	BinningState binningState;

	if (sort_mode == 0) {
		// ---- Legacy single 64-bit composite sort path. ----
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

		size_t binning_chunk_size = required<BinningState>(num_rendered);
		char* binning_chunkptr = binningBuffer(binning_chunk_size);
		binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

		duplicateWithKeys << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			geomState.conic_t,
			geomState.depths,
			geomState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			radii,
			geomState.radii_x,
			geomState.radii_y,
			tile_grid);
		CHECK_CUDA(, debug)

		int bit = getHigherMsb(tile_grid.x * tile_grid.y);
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 0, 32 + bit), debug)

		CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
		if (num_rendered > 0)
			identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
				num_rendered,
				binningState.point_list_keys,
				imgState.ranges);
		CHECK_CUDA(, debug)
	} else {
		// ---- FastGS two-stage sort path. ----
		// Read atomic counters back to host.
		uint32_t n_visible_h = 0, n_instances_h = 0;
		CHECK_CUDA(cudaMemcpy(&n_visible_h, geomState.n_visible_atomic, sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(&n_instances_h, geomState.n_instances_atomic, sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
		num_rendered = (int)n_instances_h;

		size_t binning_chunk_size = required<BinningState>(num_rendered);
		char* binning_chunkptr = binningBuffer(binning_chunk_size);
		binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

		if (n_visible_h > 0 && num_rendered > 0) {
			// Stage 1: depth-sort 32-bit keys on n_visible entries.
			// CUB SortPairs requires non-aliased buffers. The sorted depth keys
			// themselves are not needed afterward, so we land them in
			// `offset_compact` as scratch (it gets overwritten by
			// apply_depth_ordering immediately after).
			CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				binningState.depth_sorting_space,
				binningState.sorting_size_depth,
				geomState.depth_keys_compact, geomState.offset_compact,  // keys: in -> scratch
				geomState.prim_idx_compact, geomState.prim_idx_compact_sorted,
				n_visible_h, 0, 32), debug)

			// Stage 2: reorder per-primitive tile counts into depth-sorted order.
			apply_depth_ordering<<<(n_visible_h + 255) / 256, 256>>>(
				geomState.prim_idx_compact_sorted,
				geomState.tiles_touched,
				geomState.offset_compact,
				n_visible_h);
			CHECK_CUDA(, debug)

			// Stage 3: exclusive prefix sum on offset_compact (n_visible entries).
			CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
				geomState.scanning_space, geomState.scan_size,
				geomState.offset_compact, geomState.offset_compact, n_visible_h), debug)

			// Stage 4: emit (tile_key, prim_idx) pairs in depth order.
			create_instances<<<(n_visible_h + 255) / 256, 256>>>(
				geomState.prim_idx_compact_sorted,
				geomState.offset_compact,
				geomState.means2D,
				geomState.conic_t,
				radii,
				geomState.radii_x,
				geomState.radii_y,
				binningState.tile_keys_unsorted,
				binningState.point_list_unsorted,
				tile_grid,
				n_visible_h);
			CHECK_CUDA(, debug)

			// Stage 5: tile-sort 32-bit keys on n_instances entries (stable →
			// preserves depth order within each tile).
			int end_bit = getHigherMsb(tile_grid.x * tile_grid.y);
			if (end_bit < 1) end_bit = 1;
			CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				binningState.list_sorting_space,
				binningState.sorting_size_tile,
				binningState.tile_keys_unsorted, binningState.tile_keys,
				binningState.point_list_unsorted, binningState.point_list,
				num_rendered, 0, end_bit), debug)

			// Stage 6: identify per-tile ranges in the sorted list.
			CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
			identifyTileRanges32<<<(num_rendered + 255) / 256, 256>>>(
				num_rendered,
				binningState.tile_keys,
				imgState.ranges);
			CHECK_CUDA(, debug)
		} else {
			CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
		}
	}

	// Render: SH base color + atlas residual. preprocessCUDA already wrote
	// the SH-evaluated (or precomputed) base color into geomState.rgb (FP16).
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		beta,
		width, height,
		geomState.means2D,
		geomState.rgb,
		geomState.transMat,
		geomState.depths,
		geomState.normal_opacity,
		background,
		out_color,
		shapes,
		kernel_type,
		means3D,
		cam_pos,
		atlas_texture,
		atlas_rects,
		atlas_width,
		sb_params,
		sb_number,
		geomState.sb_rgb,    // SB per-Gauss precomputed colors (filled by preprocess above)
		atlas_tex_obj,
		atlas_offset,
		atlas_scale,
		is_textured,
		(scaling_z != nullptr) ? geomState.ewa_conic : nullptr,
		geomState.conic_uv), debug)

	return num_rendered;
}
