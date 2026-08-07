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

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define PI 3.14159
#define TIGHTBBOX 0
#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
#define NUM_OFFSET 7
#define POS_OFFSET 8
#define VIS_OFFSET 11
#define OVERDRAW_OFFSET 14
#define MAXDEPTH_OFFSET 15  // Depth of max-contributing Gaussian per pixel
#define WSQUARE_OFFSET 16   // Sum of squared weights: sum(w_i^2) for weight_reg
#define BETA_SUM_OFFSET 17  // Sum of w_i * beta_i per pixel (--w_lambda_perpix shape reg)
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
// SnugBox / AccuTile (FastGS / Speedy-Splat port) — see auxiliary.h in
// diff_surfel_bake_render for the full derivation. Summary:
//   transMat T pulls screen px → surfel (u,v) via cross-product trick.
//   surfel disk u²+v² ≤ k² becomes a quadratic Q(px,py) ≤ 0 in pixel coords,
//   centered at ∇Q=0 we have A·dx² + 2B·dx·dy + E·dy² ≤ t.
// processTiles walks the ellipse row-by-row, emitting only tiles it crosses.
// computeEllipseIntersection uses round-to-nearest no-fusion intrinsics so
// the count phase (in preprocessCUDA) and the emit phase (in duplicateWithKeys)
// produce bit-identical tile counts regardless of NVCC's FMA-fusion choice.
// =============================================================================

__forceinline__ __device__ bool compute_conic_from_transmat(
	const glm::mat3& T, const float cutoff,
	float& A, float& B, float& E, float& t, float2& p)
{
	const float k_sq = cutoff * cutoff;
	const glm::vec3 Tu = T[0];
	const glm::vec3 Tv = T[1];
	const glm::vec3 Tw = T[2];

	const glm::vec3 n0 = glm::cross(Tv, Tw);
	const glm::vec3 n1 = glm::cross(Tw, Tu);
	const glm::vec3 n2 = glm::cross(Tu, Tv);

	const float A_ = n0.x*n0.x + n0.y*n0.y - k_sq * n0.z*n0.z;
	const float B_ = n0.x*n1.x + n0.y*n1.y - k_sq * n0.z*n1.z;
	const float E_ = n1.x*n1.x + n1.y*n1.y - k_sq * n1.z*n1.z;
	const float D_ = n0.x*n2.x + n0.y*n2.y - k_sq * n0.z*n2.z;
	const float F_ = n1.x*n2.x + n1.y*n2.y - k_sq * n1.z*n2.z;

	const float det = A_*E_ - B_*B_;
	if (!(det > 0.0f) || !(A_ > 0.0f) || !(E_ > 0.0f)) return false;

	p.x = (B_*F_ - E_*D_) / det;
	p.y = (B_*D_ - A_*F_) / det;

	const float cx_p = p.x*n0.x + p.y*n1.x + n2.x;
	const float cy_p = p.x*n0.y + p.y*n1.y + n2.y;
	const float cz_p = p.x*n0.z + p.y*n1.z + n2.z;
	const float t_   = -(cx_p*cx_p + cy_p*cy_p - k_sq * cz_p*cz_p);
	if (!(t_ > 0.0f)) return false;

	A = A_; B = B_; E = E_; t = t_;
	return true;
}

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

// Scan-line walk along the ellipse boundary. Returns total tile count.
// When gaussian_keys_unsorted/gaussian_values_unsorted are non-null, also emits
// (tile_id<<32 | depth_bits) keys per touched tile — call sites in
// duplicateWithKeys / duplicateWithKeysSorted (see rasterizer_impl.cu).
// Both the count phase (in preprocessCUDA, nullptr buffers) and the emit phase
// MUST produce bit-identical tile counts — see comment in computeEllipseIntersection.
__device__ inline uint32_t processTiles(
	const float A, const float B, const float E,
	const float disc, const float t, const float2 p,
	float2 bbox_min, float2 bbox_max,
	float2 bbox_argmin, float2 bbox_argmax,
	int2 rect_min, int2 rect_max,
	const dim3 grid, const bool isY,
	uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	bool emit_tile_only = false)
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

		const int min_tile_v = max(rect_min.y, min(rect_max.y, (int)__fdiv_rn(ellipse_min, BLOCK_V)));
		const int max_tile_v = min(rect_max.y, max(rect_min.y, (int)__fadd_rn(__fdiv_rn(ellipse_max, BLOCK_V), 1.0f)));
		tiles_count += (uint32_t)max(0, max_tile_v - min_tile_v);

		if (gaussian_keys_unsorted != nullptr) {
			for (int v = min_tile_v; v < max_tile_v; v++) {
				uint64_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
				key <<= 32;
				// emit_tile_only=true → leave low 32 bits zero (within-tile depth
				// order guaranteed by caller iterating in depth-sorted order +
				// stable sort). emit_tile_only=false → pack depth bits.
				if (!emit_tile_only) key |= *((uint32_t*)&depth);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}

		intersect_min_line = intersect_max_line;
		min_line = max_line;
	}
	return tiles_count;
}

// Two-mode driver: compute ellipse bbox from (A, B, E, t, p) and either
// count touched tiles (keys=nullptr) or emit (key, value) pairs.
__device__ inline uint32_t duplicateToTilesTouched(
	const float A, const float B, const float E,
	const float t, const float2 p, const dim3 grid,
	uint32_t idx, uint32_t off, float depth,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	bool emit_tile_only = false)
{
	const float disc = __fadd_rn(__fmul_rn(B, B), -__fmul_rn(A, E));
	if (A <= 0.0f || E <= 0.0f || disc >= 0.0f || t <= 0.0f) return 0;

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
		emit_tile_only);
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



// ============================================================================
// PROBERES (render_mode flag 0x1000): probe-mapped shared-texture residual.
// The residual for a ray-surfel intersection at surfel-uv (s.x, s.y) is a
// bilinear fetch from ONE shared RGB texture image at the probe-affine-mapped
// coordinate:  tc = A * uv + t,  probe = [a11, a12, a21, a22, tx, ty]  (per
// surfel, [N,6], carried in the `features_diffuse` kernel slot; the texture
// [Ht, Wt, 3] rides in `gridrange_diffuse`, dims {Ht, Wt} in
// `offsets_diffuse`). Texel centers at integer+0.5, border clamp.
// ============================================================================

// Bilinear sample of tex [th, tw, 3] at continuous pixel coords (tx, ty).
// Returns the 4 corner indices/weights so forward and backward share one
// convention. Out-of-range coords clamp to the border (gradient w.r.t. the
// clamped axis is then naturally 0 because both corners coincide).
struct ProbeTexSample {
	int x0, x1, y0, y1;
	float fx, fy;
};

__device__ inline ProbeTexSample probe_tex_setup(float tx, float ty, int th, int tw)
{
	ProbeTexSample smp;
	const float x = tx - 0.5f;
	const float y = ty - 0.5f;
	const float xf = floorf(x);
	const float yf = floorf(y);
	smp.fx = x - xf;
	smp.fy = y - yf;
	const int x0 = (int)xf;
	const int y0 = (int)yf;
	smp.x0 = min(max(x0, 0), tw - 1);
	smp.x1 = min(max(x0 + 1, 0), tw - 1);
	smp.y0 = min(max(y0, 0), th - 1);
	smp.y1 = min(max(y0 + 1, 0), th - 1);
	return smp;
}

__device__ inline void probe_tex_sample(const float* __restrict__ tex, int th, int tw,
                                        float tx, float ty, float out[3])
{
	const ProbeTexSample smp = probe_tex_setup(tx, ty, th, tw);
	const float w00 = (1.0f - smp.fx) * (1.0f - smp.fy);
	const float w10 = smp.fx * (1.0f - smp.fy);
	const float w01 = (1.0f - smp.fx) * smp.fy;
	const float w11 = smp.fx * smp.fy;
	const float* t00 = tex + (smp.y0 * tw + smp.x0) * 3;
	const float* t10 = tex + (smp.y0 * tw + smp.x1) * 3;
	const float* t01 = tex + (smp.y1 * tw + smp.x0) * 3;
	const float* t11 = tex + (smp.y1 * tw + smp.x1) * 3;
	for (int c = 0; c < 3; c++)
		out[c] = w00 * t00[c] + w10 * t10[c] + w01 * t01[c] + w11 * t11[c];
}

// Backward of probe_tex_sample: scatters dL_dout into the 4 texels of
// dL_dtex (atomic) and returns dL/d(tx, ty). Uses the SAME clamped corner
// indices as the forward, so border texels accumulate the clamped mass and
// the positional gradient vanishes where both corners collapse.
__device__ inline float2 probe_tex_backward(const float* __restrict__ tex,
                                            float* __restrict__ dL_dtex,
                                            int th, int tw, float tx, float ty,
                                            const float dL_dout[3])
{
	const ProbeTexSample smp = probe_tex_setup(tx, ty, th, tw);
	const float w00 = (1.0f - smp.fx) * (1.0f - smp.fy);
	const float w10 = smp.fx * (1.0f - smp.fy);
	const float w01 = (1.0f - smp.fx) * smp.fy;
	const float w11 = smp.fx * smp.fy;
	const int i00 = (smp.y0 * tw + smp.x0) * 3;
	const int i10 = (smp.y0 * tw + smp.x1) * 3;
	const int i01 = (smp.y1 * tw + smp.x0) * 3;
	const int i11 = (smp.y1 * tw + smp.x1) * 3;
	float dL_dtx = 0.0f, dL_dty = 0.0f;
	for (int c = 0; c < 3; c++) {
		const float g = dL_dout[c];
		if (dL_dtex != nullptr) {
			atomicAdd(&dL_dtex[i00 + c], w00 * g);
			atomicAdd(&dL_dtex[i10 + c], w10 * g);
			atomicAdd(&dL_dtex[i01 + c], w01 * g);
			atomicAdd(&dL_dtex[i11 + c], w11 * g);
		}
		const float t00 = tex[i00 + c], t10 = tex[i10 + c];
		const float t01 = tex[i01 + c], t11 = tex[i11 + c];
		// d(out)/d(fx) and d(out)/d(fy); dfx/dtx = 1, dfy/dty = 1.
		dL_dtx += g * ((1.0f - smp.fy) * (t10 - t00) + smp.fy * (t11 - t01));
		dL_dty += g * ((1.0f - smp.fx) * (t01 - t00) + smp.fx * (t11 - t10));
	}
	return {dL_dtx, dL_dty};
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