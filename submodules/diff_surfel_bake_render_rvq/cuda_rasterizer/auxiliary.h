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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define PI 3.14159
#define TIGHTBBOX 0
#define RENDER_AXUTILITY 0
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
#define NUM_OFFSET 7
#define POS_OFFSET 8
#define VIS_OFFSET 11
#define DIFFUSE_RGB_OFFSET 14
// #define MEDIAN_WEIGHT_OFFSET 7

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
// #define NEAR_PLANE 0.2
// #define FAR_PLANE 100.0
#define DETACH_WEIGHT 0

__device__ const float near_n = 0.2;
__device__ const float far_n = 100.0;
__device__ const float FilterSize = 0.707106; // sqrt(2) / 2
__device__ const float FilterInvSquare = 2.0f;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// Rectangular AABB variant with separate X and Y radii
__forceinline__ __device__ void getRectXY(const float2 p, int radius_x, int radius_y, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - radius_x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - radius_y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + radius_x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + radius_y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// =============================================================================
// SnugBox / AccuTile (FastGS / Speedy-Splat ports)
//
// 2DGS's `transMat T` is the homogeneous mapping pixel → surfel:
//   (u_h, v_h, w_h) = T * (px, py, 1)   with   surfel s = (u_h/w_h, v_h/w_h)
// (this is what `compute_aabb` uses directly, no inverse).
//
// The surfel-space disk `s.x² + s.y² <= cutoff²` pulls back to the inequality
//   u_h² + v_h² − cutoff²·w_h² <= 0
// in homogeneous pixel coords, expanding to a quadratic form in (px, py):
//   Q(px, py) = A·px² + 2B·px·py + E·py² + 2D·px + 2F·py + G  <=  0
//
// Centering at the gradient-zero point p:
//   A·dx² + 2B·dx·dy + E·dy² <= t,   t = -(D·p.x + F·p.y + G)
//
// FastGS's processTiles / computeEllipseIntersection consume exactly this
// (A, B, E, t, p, disc=B²-AE) form.
// =============================================================================

// Compute conic form (A, B, E, t) and screen-space center p from transMat T
// and a surfel-space cutoff radius. Returns false on degenerate projections.
//
// 2DGS uses transMat T to recover surfel coords from a pixel via the
// cross-product trick (in render kernel):
//   k = px·Tw − Tu,  l = py·Tw − Tv,  s = cross(k,l) / cross(k,l).z
// Expanding cross(k,l) componentwise (the px·py terms cancel):
//   cross(k,l) = px·(Tv×Tw) + py·(Tw×Tu) + (Tu×Tv)
// So with n0 = Tv×Tw, n1 = Tw×Tu, n2 = Tu×Tv:
//   cross(k,l).x = n0.x·px + n1.x·py + n2.x    (linear in px, py)
//   cross(k,l).y = n0.y·px + n1.y·py + n2.y
//   cross(k,l).z = n0.z·px + n1.z·py + n2.z
//
// The disk `s.x² + s.y² ≤ k²` becomes `cross.x² + cross.y² − k²·cross.z² ≤ 0`,
// a quadratic in (px, py). Coefficients are extracted by squaring each
// cross-component (linear in px, py) and combining with the appropriate sign:
//   A (px²) = n0.x² + n0.y² − k²·n0.z²
// (NOT n0.x² + n1.x² − k²·n2.x² — that confuses cross-components with vectors!)
__forceinline__ __device__ bool compute_conic_from_transmat(
	const glm::mat3& T, const float cutoff,
	float& A, float& B, float& E, float& t, float2& p)
{
	const float k_sq = cutoff * cutoff;
	const glm::vec3 Tu = T[0];
	const glm::vec3 Tv = T[1];
	const glm::vec3 Tw = T[2];

	// Coefficient vectors of (px, py, 1) in cross(k,l) — see derivation above.
	const glm::vec3 n0 = glm::cross(Tv, Tw);   // coef of px in (cross.x, cross.y, cross.z)
	const glm::vec3 n1 = glm::cross(Tw, Tu);   // coef of py
	const glm::vec3 n2 = glm::cross(Tu, Tv);   // constant

	// Q(px,py) = cross.x² + cross.y² − k²·cross.z².
	// Quadratic coefficients group by px/py power, summed across cross-components.
	const float A_ = n0.x*n0.x + n0.y*n0.y - k_sq * n0.z*n0.z;         // px²
	const float B_ = n0.x*n1.x + n0.y*n1.y - k_sq * n0.z*n1.z;         // px·py (half)
	const float E_ = n1.x*n1.x + n1.y*n1.y - k_sq * n1.z*n1.z;         // py²
	const float D_ = n0.x*n2.x + n0.y*n2.y - k_sq * n0.z*n2.z;         // px linear (half)
	const float F_ = n1.x*n2.x + n1.y*n2.y - k_sq * n1.z*n2.z;         // py linear (half)
	// G is the constant term but we never need it explicitly — see t below.

	const float det = A_*E_ - B_*B_;
	if (!(det > 0.0f) || !(A_ > 0.0f) || !(E_ > 0.0f)) return false;

	// Center: ∇Q = 0 solved by Cramer (A·p.x + B·p.y = -D, B·p.x + E·p.y = -F).
	p.x = (B_*F_ - E_*D_) / det;
	p.y = (B_*D_ - A_*F_) / det;

	// t = -Q(p). The polynomial form (D·p.x + F·p.y + G) involves canceling
	// O(1e9) magnitudes — float32 loses ~7 digits and can return values 2×
	// off, blowing up bbox extents (verified in Python — see
	// scripts/test_snugbox_math.py). Evaluate Q at p directly via the
	// cross-product form (each component O(1) after gradient cancellation):
	//     cross_p = p.x·n0 + p.y·n1 + n2
	//     Q(p)   = cross_p.x² + cross_p.y² − k²·cross_p.z²
	const float cx_p = p.x*n0.x + p.y*n1.x + n2.x;
	const float cy_p = p.x*n0.y + p.y*n1.y + n2.y;
	const float cz_p = p.x*n0.z + p.y*n1.z + n2.z;
	const float t_   = -(cx_p*cx_p + cy_p*cy_p - k_sq * cz_p*cz_p);
	if (!(t_ > 0.0f)) return false;

	A = A_; B = B_; E = E_; t = t_;
	return true;
}

// No-FMA explicit form: see processTiles comment for why this matters.
// The two call sites (count in preprocessCUDA, emit in duplicateWithKeys)
// must produce bit-identical output regardless of NVCC's FMA-fusion choice.
__forceinline__ __device__ float2 computeEllipseIntersection(
	const float A, const float B, const float E,
	const float disc, const float t, const float2 p,
	const bool isY, const float coord)
{
	const float p_u = isY ? p.y : p.x;
	const float p_v = isY ? p.x : p.y;
	const float coeff = isY ? A : E;
	const float h = __fadd_rn(coord, -p_u);
	const float disc_h2 = __fmul_rn(__fmul_rn(disc, h), h);
	const float t_coeff = __fmul_rn(t, coeff);
	const float radicand = __fadd_rn(disc_h2, t_coeff);
	const float sqrt_term = sqrtf(fmaxf(radicand, 0.0f));
	const float neg_Bh = __fmul_rn(-B, h);
	return {
		__fadd_rn(__fdiv_rn(__fadd_rn(neg_Bh, -sqrt_term), coeff), p_v),
		__fadd_rn(__fdiv_rn(__fadd_rn(neg_Bh,  sqrt_term), coeff), p_v)
	};
}

// Scan-line walk along the ellipse boundary. Returns total tile count;
// when key/value buffers are non-null, also emits a (depth-keyed) entry per tile.
//
// IMPORTANT: This function is inlined into BOTH the count phase
// (preprocessCUDA in forward.cu, nullptr buffers) AND the emit phase
// (duplicateWithKeys in rasterizer_impl.cu, real buffers). NVCC compiles
// the two specializations independently; under --use_fast_math, the
// FMA-fusion choice for `disc*h*h + t*coeff` (in computeEllipseIntersection)
// and `ellipse_min/BLOCK_V` (here) can shift `min_tile_v`/`max_tile_v` by 1.
// That makes count != emit → emit overshoots its prefix-sum slot →
// cudaErrorIllegalAddress / InvalidAddressSpace. Verified on bicycle 0w0g,
// room 0w0g, treehill 005w25.
//
// Fix: use round-to-nearest no-fusion intrinsics (__fmul_rn, __fadd_rn) on
// every FP op that feeds into the integer cast. This forces both call sites
// to compile to identical (non-fused) PTX.
__device__ inline uint32_t processTiles(
	const float A, const float B, const float E,
	const float disc, const float t, const float2 p,
	float2 bbox_min, float2 bbox_max,
	float2 bbox_argmin, float2 bbox_argmax,
	int2 rect_min, int2 rect_max,
	const dim3 grid, const bool isY,
	uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted,        // legacy 64-bit composite (sort_mode==0); set null for FastGS
	uint32_t* gaussian_values_unsorted,
	uint32_t* tile_keys_unsorted = nullptr,  // FastGS 32-bit tile-only (sort_mode==1); set null for legacy
	uint32_t* tile_prim_indices = nullptr)
{
	const float BLOCK_U = isY ? (float)BLOCK_Y : (float)BLOCK_X;
	const float BLOCK_V = isY ? (float)BLOCK_X : (float)BLOCK_Y;

	if (isY) {
		rect_min = {rect_min.y, rect_min.x};
		rect_max = {rect_max.y, rect_max.x};
		bbox_min = {bbox_min.y, bbox_min.x};
		bbox_max = {bbox_max.y, bbox_max.x};
		bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
		bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
	}

	uint32_t tiles_count = 0;
	float2 intersect_min_line, intersect_max_line;
	float ellipse_min, ellipse_max;
	float min_line, max_line;

	intersect_max_line = {bbox_max.y, bbox_min.y};
	min_line = __fmul_rn((float)rect_min.x, BLOCK_U);
	if (bbox_min.x <= min_line) {
		intersect_min_line = computeEllipseIntersection(
			A, B, E, disc, t, p, isY, min_line);
	} else {
		intersect_min_line = intersect_max_line;
	}

	for (int u = rect_min.x; u < rect_max.x; ++u)
	{
		max_line = __fadd_rn(min_line, BLOCK_U);
		if (max_line <= bbox_max.x) {
			intersect_max_line = computeEllipseIntersection(
				A, B, E, disc, t, p, isY, max_line);
		}

		if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
			ellipse_min = bbox_min.y;
		} else {
			ellipse_min = fminf(intersect_min_line.x, intersect_max_line.x);
		}

		if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
			ellipse_max = bbox_max.y;
		} else {
			ellipse_max = fmaxf(intersect_min_line.y, intersect_max_line.y);
		}

		// No-FMA scaling of the integer-cast bounds — see comment above the
		// function. __fdiv_rn ensures bit-equal results across both call sites.
		const int min_tile_v = max(rect_min.y, min(rect_max.y, (int)__fdiv_rn(ellipse_min, BLOCK_V)));
		const int max_tile_v = min(rect_max.y, max(rect_min.y, (int)__fadd_rn(__fdiv_rn(ellipse_max, BLOCK_V), 1.0f)));
		tiles_count += (uint32_t)max(0, max_tile_v - min_tile_v);

		if (gaussian_keys_unsorted != nullptr) {
			// Legacy 64-bit composite key (tile<<32 | depth_bits).
			for (int v = min_tile_v; v < max_tile_v; v++) {
				uint64_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
				key <<= 32;
				key |= *((uint32_t*)&depth);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		} else if (tile_keys_unsorted != nullptr) {
			// FastGS 32-bit tile-only key (depth ordering preserved by stable
			// sort, since caller iterates in depth-sorted order).
			for (int v = min_tile_v; v < max_tile_v; v++) {
				const uint32_t tile_idx = isY ? (u * grid.x + v) : (v * grid.x + u);
				tile_keys_unsorted[off] = tile_idx;
				tile_prim_indices[off] = idx;
				off++;
			}
		}

		intersect_min_line = intersect_max_line;
		min_line = max_line;
	}
	return tiles_count;
}

// Two-mode driver: compute screen-space ellipse extents from (A, B, E, t, p)
// and either count touched tiles (all key buffers null) or emit (key, value)
// pairs. Pass either the 64-bit composite buffer (legacy) OR the 32-bit
// tile-only buffer (FastGS), not both.
__device__ inline uint32_t duplicateToTilesTouched(
	const float A, const float B, const float E,
	const float t, const float2 p, const dim3 grid,
	uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	uint32_t* tile_keys_unsorted = nullptr,
	uint32_t* tile_prim_indices = nullptr)
{
	const float disc = __fadd_rn(__fmul_rn(B, B), -__fmul_rn(A, E));
	if (A <= 0.0f || E <= 0.0f || disc >= 0.0f || t <= 0.0f) return 0;

	// Ellipse extreme points (where ∂Q/∂x = 0 and ∂Q/∂y = 0 respectively).
	const float B2t = __fmul_rn(__fmul_rn(B, B), t);
	const float x_term_sq = -__fdiv_rn(B2t, __fmul_rn(disc, A));
	const float y_term_sq = -__fdiv_rn(B2t, __fmul_rn(disc, E));
	if (x_term_sq < 0.0f || y_term_sq < 0.0f) return 0;
	float x_term = sqrtf(x_term_sq);
	float y_term = sqrtf(y_term_sq);
	x_term = (B < 0.0f) ? x_term : -x_term;
	y_term = (B < 0.0f) ? y_term : -y_term;

	const float2 bbox_argmin = { p.y - y_term, p.x - x_term };
	const float2 bbox_argmax = { p.y + y_term, p.x + x_term };
	const float2 bbox_min = {
		computeEllipseIntersection(A, B, E, disc, t, p, true, bbox_argmin.x).x,
		computeEllipseIntersection(A, B, E, disc, t, p, false, bbox_argmin.y).x
	};
	const float2 bbox_max = {
		computeEllipseIntersection(A, B, E, disc, t, p, true, bbox_argmax.x).y,
		computeEllipseIntersection(A, B, E, disc, t, p, false, bbox_argmax.y).y
	};

	const int2 rect_min = {
		max(0, min((int)grid.x, (int)__fdiv_rn(bbox_min.x, (float)BLOCK_X))),
		max(0, min((int)grid.y, (int)__fdiv_rn(bbox_min.y, (float)BLOCK_Y)))
	};
	const int2 rect_max = {
		max(0, min((int)grid.x, (int)__fadd_rn(__fdiv_rn(bbox_max.x, (float)BLOCK_X), 1.0f))),
		max(0, min((int)grid.y, (int)__fadd_rn(__fdiv_rn(bbox_max.y, (float)BLOCK_Y), 1.0f)))
	};

	const int y_span = rect_max.y - rect_min.y;
	const int x_span = rect_max.x - rect_min.x;
	if (y_span * x_span == 0) return 0;

	const bool isY = y_span < x_span;
	return processTiles(
		A, B, E, disc, t, p,
		bbox_min, bbox_max, bbox_argmin, bbox_argmax,
		rect_min, rect_max, grid, isY,
		idx, off, depth,
		gaussian_keys_unsorted, gaussian_values_unsorted,
		tile_keys_unsorted, tile_prim_indices);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 cross(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}

__forceinline__ __device__ float3 operator*(float3 a, float3 b){return make_float3(a.x * b.x, a.y * b.y, a.z*b.z);}

__forceinline__ __device__ float2 operator*(float2 a, float2 b){return make_float2(a.x * b.x, a.y * b.y);}

__forceinline__ __device__ float3 operator*(float f, float3 a){return make_float3(f * a.x, f * a.y, f * a.z);}

__forceinline__ __device__ float2 operator*(float f, float2 a){return make_float2(f * a.x, f * a.y);}

__forceinline__ __device__ float3 operator-(float3 a, float3 b){return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}

__forceinline__ __device__ float2 operator-(float2 a, float2 b){return make_float2(a.x - b.x, a.y - b.y);}

__forceinline__ __device__ float sumf3(float3 a){return a.x + a.y + a.z;}

__forceinline__ __device__ float sumf2(float2 a){return a.x + a.y;}

__forceinline__ __device__ float3 sqrtf3(float3 a){return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));}

__forceinline__ __device__ float2 sqrtf2(float2 a){return make_float2(sqrtf(a.x), sqrtf(a.y));}

__forceinline__ __device__ float3 minf3(float f, float3 a){return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}

__forceinline__ __device__ float2 minf2(float f, float2 a){return make_float2(min(f, a.x), min(f, a.y));}

__forceinline__ __device__ float3 maxf3(float f, float3 a){return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}

__forceinline__ __device__ float2 maxf2(float f, float2 a){return make_float2(max(f, a.x), max(f, a.y));}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y),
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x),
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}


inline __device__ glm::vec4
quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}


inline __device__ glm::mat3
scale_to_mat(const glm::vec2 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	// S[2][2] = glob_scale * scale.z;
	return S;
}



#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif