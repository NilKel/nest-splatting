/*
 * Baked rendering submodule — forward-only.
 * Supports standard SH rendering + residual texture lookup (baked mode).
 * No hashgrid, no MLP, no backward pass.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Mirrors the training-time activation constants in diff_surfel_3D_sh_res.
// Set via FORWARD::setActivationBias / FORWARD::setCompactMult from Python.
// Defaults match the training defaults (sh_bias=0.5, res_bias=0.0 under 3D_SH_res;
// we use 0.5 here because the legacy bake pipeline always applied +0.5 to SH).
__device__ float d_sh_bias = 0.5f;
__device__ float d_res_bias = 0.0f;
__device__ float d_compact_mult = 1.0f;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color. Forward-only baked
// path: no `clamped` tracking (no backward pass that needs it).
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += d_sh_bias;
	return glm::max(result, 0.0f);
}

// Spherical-Beta (SB) evaluation — matches eval_sb in Python (gaussian_renderer/__init__.py).
// sb_params: [K, 6] per-Gaussian block: (r, g, b, theta, phi, beta_raw).
// Returns summed RGB lobe contribution for one Gaussian under view_dir.
__device__ glm::vec3 eval_sb(const float* sb_params, int K, const glm::vec3 view_dir)
{
	const float softplus_scale = 10.0f * 0.693147f;  // 10 * ln(2) — matches Python.
	glm::vec3 rgb_sum = glm::vec3(0.0f);
	for (int k = 0; k < K; ++k) {
		const float* p = sb_params + k * 6;
		float r = p[0], g = p[1], b = p[2];
		float theta = p[3], phi = p[4], beta_raw = p[5];
		// Per-primitive beta = 4 * exp(beta_raw). Matches reference.
		float beta = 4.0f * __expf(beta_raw);
		// Steep softplus activation on RGB.
		float sr = __logf(1.0f + __expf(softplus_scale * r)) / softplus_scale;
		float sg = __logf(1.0f + __expf(softplus_scale * g)) / softplus_scale;
		float sb_ = __logf(1.0f + __expf(softplus_scale * b)) / softplus_scale;
		// Direction from (theta, phi).
		float st = __sinf(theta), ct = __cosf(theta);
		float sp = __sinf(phi),   cp = __cosf(phi);
		glm::vec3 mu = glm::vec3(st * cp, st * sp, ct);
		float dot = glm::dot(mu, view_dir);
		if (dot > 0.0f) {
			float w = __powf(dot, beta);
			rgb_sum += glm::vec3(sr * w, sg * w, sb_ * w);
		}
	}
	return rgb_sum;
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into an image plane
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H,
	glm::mat3 &T,
	float3 &normal
) {
	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

	// Normalize surface normals
	float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	if(normal_len > 1e-7f) {
		normal.x /= normal_len;
		normal.y /= normal_len;
		normal.z /= normal_len;
	}
}

// Computing the bounding box of the 2D Gaussian and its center
__device__ bool compute_aabb(
	glm::mat3 T,
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p -
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Preprocessing kernel: frustum culling, SH eval, transmat computation, tile binning
// aabb_mode: 0 = square (2DGS default), 1 = square + AdR, 2 = rect, 3 = rect + AdR
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	int* radii_x,
	int* radii_y,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	__half* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Frustum culling
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Compute transmat
	glm::mat3 T;
	float3 normal;
	compute_transmat(p_orig, scales[idx], scale_modifier, rotations[idx],
		projmatrix, viewmatrix, W, H, T, normal);

	// Flip normal if backfacing
#if BACKFACE_CULL
	float3 ray_dir = {p_view.x, p_view.y, p_view.z};
	float ndotray = normal.x * ray_dir.x + normal.y * ray_dir.y + normal.z * ray_dir.z;
	if (ndotray > 0) {
		normal.x = -normal.x;
		normal.y = -normal.y;
		normal.z = -normal.z;
	}
#if DUAL_VISIABLE
#else
	if (ndotray > 0) return;
#endif
#endif

	// Store transmat
	transMats[idx * 9 + 0] = T[0].x;
	transMats[idx * 9 + 1] = T[0].y;
	transMats[idx * 9 + 2] = T[0].z;
	transMats[idx * 9 + 3] = T[1].x;
	transMats[idx * 9 + 4] = T[1].y;
	transMats[idx * 9 + 5] = T[1].z;
	transMats[idx * 9 + 6] = T[2].x;
	transMats[idx * 9 + 7] = T[2].y;
	transMats[idx * 9 + 8] = T[2].z;

	// Compute AABB cutoff
	// aabb_mode: 0 = square, 1 = square + AdR, 2 = rect, 3 = rect + AdR
	bool use_adr = (aabb_mode == 1 || aabb_mode == 3);
	bool use_rect = (aabb_mode >= 2);

	float cutoff;
	bool is_beta_kernel = (kernel_type >= 1 && kernel_type <= 4);

	// Cutoff selection mirrors diff_surfel_3D_sh_res/forward.cu ~L600 branches.
	// Matching training exactly is critical: even subtle cutoff differences cause
	// multiplicative dimming (we've observed ~16% at mode=2 + beta_scaled).
	bool use_beta_fixed = (aabb_mode == 4);  // aabb=beta: deliberate tight support
	if (use_adr && is_beta_kernel && shapes != nullptr) {
		// Mode 1/3 + beta: AdR for beta kernel with Gaussian-low-pass max-pool.
		float k = (kernel_type == 4) ? 3.0f : 1.0f;

		float opacity_val = opacities[idx];
		float shape = shapes[idx];

		if (opacity_val < (1.0f / 255.0f)) return;

		float ratio = 1.0f / (255.0f * opacity_val);
		float r_beta = 0.0f;
		float threshold = powf(ratio, 1.0f / shape);
		if (threshold < 1.0f) r_beta = k * sqrtf(1.0f - threshold);

		// IMPORTANT: training's beta-AdR branch leaves r_lp unmodified.
		// d_compact_mult is only applied to the pure-Gaussian branch below.
		float r_lp = 0.0f;
		float log_term = logf(255.0f * opacity_val);
		if (log_term > 0.0f) r_lp = sqrtf(2.0f * log_term);

		cutoff = fmaxf(r_beta, r_lp);
		cutoff = fminf(cutoff, k + 2.0f);
	} else if (use_adr) {
		// Mode 1/3 + Gaussian (no shapes): FastGS Compact Box — only branch
		// that uses d_compact_mult.
		float opacity_val = opacities[idx];
		if (opacity_val < (1.0f / 255.0f)) return;

		float log_term = logf(255.0f * opacity_val);
		cutoff = (log_term > 0.0f)
			? sqrtf(2.0f * log_term * d_compact_mult)
			: 0.1f;
		cutoff = fminf(cutoff, 4.0f);
	} else if (use_beta_fixed) {
		// Mode 4 (aabb=beta) only: deliberate tight cutoff for max-pool compact support.
		float k = (kernel_type == 4) ? 3.0f : 1.0f;
		float r_lp_typical = sqrtf(2.0f * logf(127.5f));
		cutoff = fmaxf(k * 1.1f, r_lp_typical);
	} else {
		// Mode 0 (square) / mode 2 (rect) with ANY kernel: fixed 4σ (2DGS default).
		// The prior "is_beta_kernel → 3.3" fallback was wrong here — training uses 4.0
		// for these modes regardless of kernel.
		cutoff = 4.0f;
	}

	// Project the surfel disk to a screen-space ellipse and take its rectangular
	// AABB. SnugBox/AccuTile would be tighter here, but the conic-from-T
	// derivation isn't right for our 2DGS T (see docs/SNUGBOX_ATTEMPT.md).
	float2 point_image;
	float2 extent;
	bool ok = compute_aabb(T, cutoff, point_image, extent);
	if (!ok) return;

	float filter_r = cutoff * FilterSize;
	int rx, ry;
	uint2 rect_min, rect_max;
	bool use_rect_aabb = (aabb_mode >= 2);
	if (use_rect_aabb) {
		rx = (int)ceilf(fmaxf(extent.x, filter_r));
		ry = (int)ceilf(fmaxf(extent.y, filter_r));
		getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
	} else {
		float radius = ceilf(fmaxf(fmaxf(extent.x, extent.y), filter_r));
		rx = (int)radius;
		ry = (int)radius;
		getRect(point_image, (int)radius, rect_min, rect_max, grid);
	}

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// FP16 SH-color storage: range is [0, ~5] post-bias+clamp — well within
	// FP16 precision and halves the global-mem bandwidth of the inner-loop fetch
	// in renderBakedCUDA.
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs);
		rgb[idx * C + 0] = __float2half(result.x);
		rgb[idx * C + 1] = __float2half(result.y);
		rgb[idx * C + 2] = __float2half(result.z);
	} else {
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = __float2half(colors_precomp[idx * C + i]);
		}
	}

	depths[idx] = p_view.z;
	radii[idx] = max(rx, ry);  // Non-zero signals visible
	radii_x[idx] = rx;
	radii_y[idx] = ry;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// =============================================================================
// Render kernel for baked mode: SH base color + bilinear residual texture lookup
// =============================================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBakedCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float beta,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const __half* __restrict__ features,    // SH base RGB from preprocessing [N, 3] FP16
	const float* __restrict__ transMats,   // [N, 9]
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ shapes,       // Beta kernel shape [N] (nullable)
	const int kernel_type,
	const float* __restrict__ means3D,             // [N, 3] Gaussian centers (for SB viewdir, nullable when sb_number==0)
	const float* __restrict__ cam_pos,             // [3] camera position (for SB viewdir, nullable when sb_number==0)
	const __half* __restrict__ atlas_texture,      // [atlas_h * atlas_width * 3] FP16 — used by SW fallback when atlas_tex_obj==0
	const float* __restrict__ atlas_rects,         // [N, 4] (u0_px, v0_px, w_px, h_px) (required for atlas mode)
	const int atlas_width,                         // atlas dimension (e.g. 4096)
	const float* __restrict__ sb_params,           // [N, K, 6] SB params (nullable)
	const int sb_number,                           // K lobes per Gaussian (0 = SB disabled)
	cudaTextureObject_t atlas_tex_obj,             // hardware texture object for atlas (required)
	float atlas_offset,                            // dequantization offset (uint8 atlas)
	float atlas_scale)                             // dequantization scale  (uint8 atlas)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Shared memory for batch loading
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float collected_shapes[BLOCK_SIZE];

	float T = 1.0f;
	float C[3] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Fetch per-Gaussian data
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			if (shapes != nullptr) {
				collected_shapes[block.thread_rank()] = shapes[coll_id];
			}
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// Kernel evaluation (Gaussian or Beta)
			float alpha;
			if ((kernel_type == 1 || kernel_type == 4) && shapes != nullptr) {
				// Beta kernel with max-pool handoff to Gaussian low-pass
				// kernel_type 1: k²=1 (unit disk), kernel_type 4: k²=9 (3σ scaled)
				float k_sq = (kernel_type == 4) ? 9.0f : 1.0f;
				float shape = collected_shapes[j];

				if (rho3d >= k_sq + 1e-6f) continue;

				float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
				float alpha_beta = powf(base, shape);
				float alpha_lp = expf(-rho2d / 2.0f);
				float kernel_val = fmaxf(alpha_beta, alpha_lp);
				alpha = fminf(0.99f, opa * kernel_val);
			} else if (kernel_type == 2) {
				// Flex kernel
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				float G = expf(power);
				float per_gaussian_beta = collected_shapes[j];
				if (per_gaussian_beta > 0.0f)
					G = (1.0f + per_gaussian_beta) * G / (1.0f + per_gaussian_beta * G);
				alpha = min(0.99f, opa * G);
			} else if (kernel_type == 3 && shapes != nullptr) {
				// General kernel: isotropic generalized Gaussian
				float beta_param = collected_shapes[j];
				float power = -0.5f * powf(rho, beta_param * 0.5f);
				if (power > 0.0f) continue;
				alpha = min(0.99f, opa * expf(power));
			} else {
				// Standard Gaussian kernel
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				alpha = min(0.99f, opa * expf(power));
			}

			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;

#if RENDER_AXUTILITY
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				median_contributor = contributor;
			}
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// SH base color (from preprocessing). FP16 storage halves the
			// global-memory bandwidth of this hot read (called 256× per tile).
			int gauss_id = collected_id[j];
			float feat[3] = {
				__half2float(features[gauss_id * 3 + 0]),
				__half2float(features[gauss_id * 3 + 1]),
				__half2float(features[gauss_id * 3 + 2]),
			};

			// Residual texture lookup (atlas mode). Skip when no atlas was
			// bound (SH-only render path) — atlas_rects is nullptr and
			// atlas_tex_obj is 0; dereferencing either is illegal.
			if (atlas_rects != nullptr) {
				float u0_px  = atlas_rects[gauss_id * 4 + 0];
				float v0_px  = atlas_rects[gauss_id * 4 + 1];
				float u_span = atlas_rects[gauss_id * 4 + 2];
				float v_span = atlas_rects[gauss_id * 4 + 3];

				// Surfel s → atlas pixel coords (texel-center convention).
				// Bake kernel places sample i at s = (i+0.5)*step - E
				// Inverse: tex_coord = (s+E)/(2E) * G - 0.5
				float au = u0_px + (s.x + UV_EXTENT) / (2.0f * UV_EXTENT) * u_span - 0.5f;
				float av = v0_px + (s.y + UV_EXTENT) / (2.0f * UV_EXTENT) * v_span - 0.5f;
				au = fmaxf(u0_px, fminf(u0_px + u_span - 1.001f, au));
				av = fmaxf(v0_px, fminf(v0_px + v_span - 1.001f, av));

				if (atlas_tex_obj != 0) {
					// Hardware bilinear via texture unit. cudaReadModeNormalizedFloat
					// returns uint8 as [0, 1]; dequantize via `*scale + offset`.
					// +0.5 is the pixel-center offset for cudaFilterModeLinear.
					float4 rgba = tex2D<float4>(atlas_tex_obj, au + 0.5f, av + 0.5f);
					feat[0] += rgba.x * atlas_scale + atlas_offset;
					feat[1] += rgba.y * atlas_scale + atlas_offset;
					feat[2] += rgba.z * atlas_scale + atlas_offset;
				} else if (atlas_texture != nullptr) {
					// Software bilinear fallback for atlases that exceed the
					// 65,536 cudaArray 2D dimension limit (very tall packed atlases).
					int au0 = (int)au, av0 = (int)av;
					float fu = au - au0, fv = av - av0;
					int au1 = min(au0 + 1, (int)(u0_px + u_span - 1));
					int av1 = min(av0 + 1, (int)(v0_px + v_span - 1));
					long long idx00 = ((long long)av0 * atlas_width + au0) * 3;
					long long idx10 = ((long long)av0 * atlas_width + au1) * 3;
					long long idx01 = ((long long)av1 * atlas_width + au0) * 3;
					long long idx11 = ((long long)av1 * atlas_width + au1) * 3;
					for (int ch = 0; ch < 3; ch++) {
						float c00 = __half2float(atlas_texture[idx00 + ch]);
						float c10 = __half2float(atlas_texture[idx10 + ch]);
						float c01 = __half2float(atlas_texture[idx01 + ch]);
						float c11 = __half2float(atlas_texture[idx11 + ch]);
						feat[ch] += (1-fu)*(1-fv)*c00 + fu*(1-fv)*c10
						          + (1-fu)*fv*c01 + fu*fv*c11;
					}
				}
			}

			// Spherical-Beta (SB) additive contribution if provided.
			if (sb_params != nullptr && sb_number > 0 && means3D != nullptr && cam_pos != nullptr) {
				const float* sbp = sb_params + gauss_id * sb_number * 6;
				// View dir: (cam_pos → Gaussian center). matches training's eval_sb.
				glm::vec3 gc = glm::vec3(means3D[gauss_id * 3 + 0],
				                         means3D[gauss_id * 3 + 1],
				                         means3D[gauss_id * 3 + 2]);
				glm::vec3 cp = glm::vec3(cam_pos[0], cam_pos[1], cam_pos[2]);
				glm::vec3 vd = gc - cp;
				vd = vd / (glm::length(vd) + 1e-8f);
				glm::vec3 sb_rgb = eval_sb(sbp, sb_number, vd);
				feat[0] += sb_rgb.x;
				feat[1] += sb_rgb.y;
				feat[2] += sb_rgb.z;
			}

			// Final activation matches training's line 1455:
			//   color = ReLU(SH_clamped + residual + d_res_bias)
			// SH_clamped is already in feat[] (precomputed via computeColorFromSH
			// which uses d_sh_bias). We apply res_bias + final ReLU once here.
			for (int ch = 0; ch < 3; ch++)
				feat[ch] = fmaxf(0.0f, feat[ch] + d_res_bias);

			// Alpha compositing
			for (int ch = 0; ch < 3; ch++)
				C[ch] += feat[ch] * w;
			T = test_T;
		}
	}

	if (inside)
	{
		for (int ch = 0; ch < 3; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// =============================================================================
// Host dispatch functions
// =============================================================================

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float beta,
	int W, int H,
	const float2* points_xy_image,
	const __half* features,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	const float* bg_color,
	float* out_color,
	const float* shapes,
	const int kernel_type,
	const float* means3D,
	const float* cam_pos,
	const __half* atlas_texture,
	const float* atlas_rects,
	const int atlas_width,
	const float* sb_params,
	const int sb_number,
	cudaTextureObject_t atlas_tex_obj,
	float atlas_offset,
	float atlas_scale)
{
	renderBakedCUDA<<<grid, block>>>(
		ranges, point_list, beta, W, H,
		points_xy_image, features, transMats, depths, normal_opacity,
		bg_color, out_color,
		shapes, kernel_type,
		means3D, cam_pos,
		atlas_texture, atlas_rects, atlas_width,
		sb_params, sb_number, atlas_tex_obj,
		atlas_offset, atlas_scale);
}

// Device-global setters (mirror training-time setters in diff_surfel_3D_sh_res).
__global__ void setBakeActivationBiasKernel(float sh, float res) {
	d_sh_bias = sh;
	d_res_bias = res;
}
void FORWARD::setActivationBias(float sh_bias, float res_bias) {
	setBakeActivationBiasKernel<<<1, 1>>>(sh_bias, res_bias);
}

__global__ void setBakeCompactMultKernel(float val) { d_compact_mult = val; }
void FORWARD::setCompactMult(float val) {
	setBakeCompactMultKernel<<<1, 1>>>(val);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	int* radii_x,
	int* radii_y,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	__half* colors,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		radii_x,
		radii_y,
		points_xy_image,
		depths,
		transMats,
		colors,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode
	);
}
