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
// EXPERIMENT (beta_scaled mult + lowpass sweep): scales the beta/non-AdR footprint
// cutoff (default 1.0 => cutoff=4σ baseline). d_drop_lowpass removes the Gaussian
// low-pass (alpha max-pool in the render kernel + the filter_r screen extension).
// Kept SEPARATE from d_compact_mult so existing beta bakes (which carry compact_mult
// in bake_meta) render byte-identically.
__device__ float d_beta_mult = 1.0f;
__device__ bool  d_drop_lowpass = false;
// Mode-5/0/2 beta footprint: OPACITY-AWARE cutoff = max(r_beta, r_lp) (the 1/255 iso)
// so AccuTile traces a tighter — but lossless — ellipse. DEFAULT true (verified ~1.34–1.35×
// FPS at unchanged PSNR/SSIM/LPIPS on db drjohnson/playroom). set_opacity_aware_beta(False)
// reverts to the old fixed 4σ (×d_beta_mult) for A/B comparison.
__device__ bool  d_opacity_aware_beta = true;
// d_residual_mode mirrors the training-time global in diff_surfel_3D_sh_res /
// diff_surfel_mixed:
//   0 (3D_SH_res): color = ReLU(SH_clamped + residual + d_res_bias)
//   1 (3D_SH_add): color = SH_clamped + ReLU(residual + d_res_bias)
//   2 (mixed):     color = SH_clamped + residual + d_res_bias  (signed; the
//                  per-pixel ReLU on the blended image is applied in Python).
//                  Untextured surfels carry a zero atlas rect → residual==0 →
//                  reduce to the simple-2DGS SV baseline.
__device__ int d_residual_mode = 0;
// `--method mixed_3d --kernel2`: kernel for the UNTEXTURED EWA half at bake
// time. -1 ⇒ unset → untextured use the run's --kernel (kernel_type). Set
// from bake_meta["kernel2"] via set_untex_kernel(), mirroring d_residual_mode.
__device__ int d_untex_kernel = -1;

// ----------------------------------------------------------------------------
// RVQ atlas decode globals — set from Python via SetAtlasRVQCUDA(). When
// `d_rvq_codebooks` is non-null, the atlas-sample step in renderBakedCUDA
// does an L-stage codebook lookup instead of a tex2D / FP16-gather. Indices
// are stored surfel-major so block_id = surfel_offsets[g] + bv·(w/B) + bu.
// ----------------------------------------------------------------------------
__device__ const __half*  d_rvq_codebooks      = nullptr;   // [L, K, B*B*3] FP16
__device__ const uint8_t* d_rvq_indices        = nullptr;   // [L, N_used] uint8 (K ≤ 256)
__device__ const int64_t* d_rvq_surfel_offsets = nullptr;   // [N_gauss + 1] cumulative used-block count
__device__ int  d_rvq_L = 0;
__device__ int  d_rvq_K = 0;
__device__ int  d_rvq_B = 4;
__device__ unsigned long long d_rvq_N_used = 0;
__device__ float d_rvq_scale  = 1.0f;     // codebook is in atlas-residual float space; no extra scale
__device__ float d_rvq_offset = 0.0f;
__device__ int  d_rvq_bilinear = 1;       // 1 = 4-tap bilinear, 0 = nearest (4× fewer reads)
__device__ int  d_rvq_use_shared_cb = 0;  // 1 = load codebook to dynamic __shared__ per block
__device__ cudaTextureObject_t d_rvq_cb_tex  = 0;  // 1-channel FP16 texture, length L*K*B*B*3
__device__ cudaTextureObject_t d_rvq_idx_tex = 0;  // 1-channel uint8 texture, length L*N_used
__device__ int  d_rvq_use_tex_cb  = 0;
__device__ int  d_rvq_use_tex_idx = 0;
__device__ float d_rvq_cb_dq_scale  = 1.0f;
__device__ float d_rvq_cb_dq_offset = 0.0f;

// Host-side dynamic shared size set by SetAtlasRVQUseSharedCBCUDA. Used at
// kernel-launch time in FORWARD::render. Defined later in this TU.
extern size_t g_rvq_shared_bytes;

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

// Spherical Voronoi (SV) per-Gaussian softmax-mix evaluation — fused into
// preprocessCUDA so the rasterizer never sees an MLP / autograd graph.
// Mirrors the reference implementation in 2dgs-voronoi
// (`computeColorFromVoronoi` in submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu)
// and `eval_voronoi_sv` in nest's gaussian_renderer/__init__.py:
//
//     dir = (mean - cam_pos) / |mean - cam_pos|     (per-Gaussian view dir)
//     dist[k] = |sites[k] - dir|                    (L2 chord distance)
//     logit[k] = -tau[k] * dist[k]
//     W = softmax(logit)                             (max-subtract for stability)
//     feat = sum_k W[k] * colors[k]
//     rgb = max(0, feat + sh_bias)                  (ReLU; sh_bias from d_sh_bias)
//
// Inputs assume PRE-ACTIVATED tensors:
//   `sites_all`  : [P, K, 3]  unit vectors (normalized on the Python side).
//   `taus_all`   : [P, K]     post-exp scalars.
//   `colors_all` : [P, K, 3]  raw colors (no activation; ReLU applied here).
// Result is the same fp32 RGB the rasterizer would have gotten via
// computeColorFromSH(fake_shs) — sh_bias additive identical, max(.,0) identical.
__device__ glm::vec3 computeColorFromVoronoi(
	int idx,
	int K,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* sites_all,    // [P, K, 3]
	const float* taus_all,     // [P, K]
	const float* colors_all)   // [P, K, 3]
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir * (1.0f / glm::length(dir));

	const glm::vec3* sites  = ((const glm::vec3*)sites_all)  + idx * K;
	const float*     taus   = taus_all                       + idx * K;
	const glm::vec3* colors = ((const glm::vec3*)colors_all) + idx * K;

	// Pass 1: max(logits) for numerical-stability subtract.
	float max_logit = -1e30f;
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		if (logit > max_logit) max_logit = logit;
	}

	// Pass 2: sum exp(logits - max).
	float sum_exp = 0.0f;
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		sum_exp += __expf(logit - max_logit);
	}
	float inv_sum = 1.0f / sum_exp;

	// Pass 3: weighted color sum.
	glm::vec3 feat(0.0f);
	for (int k = 0; k < K; ++k) {
		glm::vec3 d = sites[k] - dir;
		float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
		float logit = -taus[k] * dist;
		float w = __expf(logit - max_logit) * inv_sum;
		feat += w * colors[k];
	}

	// d_sh_bias is set by set_activation_bias() — same hook the SH path uses
	// (default 0.5). max(., 0) reproduces nest's relu(feat + 0.5).
	feat += d_sh_bias;
	return glm::max(feat, 0.0f);
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

// `--method mixed_3d` — EWA 3D-ellipsoid path for UNTEXTURED surfels.
//
// computeCov3D / computeCov2D are ported VERBATIM from FastGS
// (diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu, the stock
// Inria EWA splatting math) so that, fed identical Gaussians, this submodule's
// untextured output matches FastGS bit-for-bit. Quaternion layout (r,x,y,z),
// no quaternion normalization, eps2d=0.3 low-pass — all kept identical to
// FastGS on purpose. The textured half is untouched (compute_transmat).
// ============================================================================
__device__ void ewa_computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ float3 ewa_computeCov2D(const float3& mean, float focal_x, float focal_y,
	float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Returns false if the Gaussian is culled (degenerate conic).
// On success: conic = inverse 2D covariance (a,b,c), point_image = pixel-space
// center, my_radius = FastGS ceil(3·sqrt(max λ)).
__device__ bool compute_ewa_conic(
	const float3& p_orig,
	const glm::vec3 scale3,
	float mod,
	const glm::vec4 rot,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const int W, const int H,
	float3& conic,
	float2& point_image,
	float& my_radius)
{
	float cov3D[6];
	ewa_computeCov3D(scale3, mod, rot, cov3D);

	float3 cov = ewa_computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return false;
	float det_inv = 1.f / det;
	conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	return true;
}

// Preprocessing kernel: frustum culling, SH eval, transmat computation, tile binning
// aabb_mode: 0 = square (2DGS default), 1 = square + AdR, 2 = rect, 3 = rect + AdR,
//            5 = SnugBox bbox + AccuTile ellipse-tight tile emission
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
	float4* conic_t,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* depth_keys_compact,
	uint32_t* prim_idx_compact,
	uint32_t* n_visible_atomic,
	uint32_t* n_instances_atomic,
	const int sort_mode,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	// Spherical Voronoi (--feature SV) per-Gaussian color path. When
	// voronoi_K > 0 AND colors_precomp == nullptr, the SH evaluation is
	// replaced by `computeColorFromVoronoi(...)`. Empty ⇒ legacy SH path.
	const float* __restrict__ voronoi_sites,    // [P, K, 3]
	const float* __restrict__ voronoi_tau,      // [P, K]
	const float* __restrict__ voronoi_colors,   // [P, K, 3]
	const int voronoi_K,
	// Spherical-Beta (--feature beta). Was evaluated per-pixel in
	// renderBakedCUDA's inner loop, but view_dir is per-Gaussian, so the
	// same RGB was redundantly recomputed for every pixel. Move to here
	// (per-Gauss, fused into preprocess), write fp16 to `sb_rgb_out`,
	// renderBakedCUDA just adds it to feat[]. Same chain math, ~10-25%
	// FPS lift on SB scenes. When sb_number == 0, no work happens.
	const float* __restrict__ sb_params,        // [P, K, 6]
	const int sb_number,
	__half* __restrict__ sb_rgb_out,            // [P, 3]
	// `--method mixed_3d`: untextured surfels render as EWA 3D ellipsoids
	// (FastGS-verbatim conic). nullptr scaling_z → pure 2DGS bake (unchanged).
	const bool* __restrict__ is_textured = nullptr,
	const float* __restrict__ scaling_z = nullptr,
	float4* __restrict__ ewa_conic = nullptr,
	// LEAN_CONIC Option A: per-Gauss conic cache [P*6]: u0, v0, J⁻¹[0,0], J⁻¹[0,1], J⁻¹[1,0], J⁻¹[1,1]
	// Populated at the end of preprocessCUDA for visible Gauss so the render kernel
	// can skip the fetch-time ray-splat.  nullptr → skip (backwards compat).
	float* __restrict__ conic_uv = nullptr)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize. tiles_touched is needed by both modes (legacy uses it for
	// inclusive prefix sum; FastGS uses it as per-id tile-count lookup table
	// for apply_depth_ordering).
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	if (ewa_conic != nullptr)
		ewa_conic[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// Frustum culling
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// `--method mixed_3d`: UNTEXTURED surfels use EWA 3D-ellipsoid geometry
	// (FastGS conic) instead of the 2DGS transmat. Color path is unchanged
	// (per-Gauss SV/SH baseline; untextured carry a zero atlas rect → no
	// residual). Geometry vars (point_image, rx/ry, rect_*, normal) are set
	// here; the shared color + store code below is reused verbatim.
	const bool untex_ewa = (scaling_z != nullptr) && (is_textured != nullptr)
	                       && (ewa_conic != nullptr) && (!is_textured[idx]);

	// Compute transmat
	glm::mat3 T;
	float3 normal;
	if (!untex_ewa)
	compute_transmat(p_orig, scales[idx], scale_modifier, rotations[idx],
		projmatrix, viewmatrix, W, H, T, normal);

	// Geometry outputs (set by the 2DGS path OR the mixed_3d EWA path below);
	// consumed by the shared color + store code that follows.
	float2 point_image;
	float2 extent = {0.0f, 0.0f};
	int rx = 0, ry = 0;
	uint2 rect_min, rect_max;
	uint32_t rect_tiles = 0;
	bool snugbox_succeeded = false;
	float A_c = 0.0f, B_c = 0.0f, E_c = 0.0f, t_c = 0.0f;
	float2 p_c;
	uint32_t n_tiles_sb = 0;
	if (!untex_ewa) {
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
		// aabb_mode: 0 = square, 1 = square + AdR, 2 = rect, 3 = rect + AdR,
		//            5 = SnugBox+AccuTile (uses the same fixed 4σ cutoff as mode 2,
		//                which is what `--aabb rect` produces in training)
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
		} else if (!is_beta_kernel) {
			// Gaussian kernel + any binning mode (2/5 in practice): opacity-aware
			// cutoff = √(2·log(255·α)). The visible iso-line for α·exp(-ρ²/2) at
			// the 1/255 floor. For α=1 this is ~3.33; α=0.5 ~3.0; α=0.1 ~2.5.
			// Replaces the prior fixed 4.0 fallback which was binning ~50% more
			// tiles than necessary for Gaussian bakes (room_gauss was the
			// triggering case — 169 k all-textured Gausses at fixed 4σ).
			float opacity_val = opacities[idx];
			if (opacity_val < (1.0f / 255.0f)) return;
			float log_term = logf(255.0f * opacity_val);
			cutoff = (log_term > 0.0f) ? sqrtf(2.0f * log_term) : 0.1f;
			cutoff = fminf(cutoff, 4.0f);
		} else if (d_opacity_aware_beta) {
			// EXPERIMENT: opacity-aware beta cutoff = max(r_beta, r_lp) — the 1/255 iso.
			// Tighter than fixed 4σ for faint/sharp surfels (lossless; clips only <1/255),
			// so AccuTile traces a smaller ellipse → fewer Gaussian-tile pairs.
			float k = (kernel_type == 4) ? 3.0f : 1.0f;
			float opacity_val = fmaxf(opacities[idx], 1.0f / 255.0f);
			float shape = shapes[idx];
			float ratio = 1.0f / (255.0f * opacity_val);
			float threshold = powf(ratio, 1.0f / shape);
			float r_beta = (threshold < 1.0f) ? k * sqrtf(1.0f - threshold) : 0.0f;
			float log_term = logf(255.0f * opacity_val);
			float r_lp = (log_term > 0.0f) ? sqrtf(2.0f * log_term) : 0.0f;
			cutoff = fminf(fmaxf(r_beta, r_lp), k + 2.0f) * d_beta_mult;
		} else {
			// Mode 0 (square) / mode 2 (rect) + beta_scaled: fixed 4σ (2DGS
			// default). Training uses 4.0 for these modes; the bake-render
			// must match to avoid multiplicative dimming (~16 % at mode=2
			// + beta_scaled was observed when this was 3.3).
			cutoff = 4.0f * d_beta_mult;  // EXPERIMENT: beta footprint mult (1.0 => 4σ baseline)
		}

		// Project the surfel disk to a screen-space ellipse and take its bbox.
		// Two paths:
		//   aabb_mode 0..3 → rect AABB from compute_aabb (existing, byte-identical).
		//   aabb_mode 5    → SnugBox bbox + AccuTile tile count (ellipse-tight).
		// The SnugBox path falls back to rect AABB on degenerate conic (rare,
		// e.g. near-edge-on surfels) so we never lose coverage.

		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;

		float filter_r = d_drop_lowpass ? 0.0f : cutoff * FilterSize;  // EXPERIMENT: drop low-pass screen extension

		// Always compute the rect AABB. We need its tile count as an upper bound
		// for SnugBox+AccuTile (a numerically-sound ellipse can never touch more
		// tiles than its bounding rect; if AccuTile reports more, the conic is
		// near-degenerate and its bbox blew up — fall through to rect).
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
		rect_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		if (rect_tiles == 0) return;

		// SnugBox+AccuTile is enabled for ALL rect modes (2, 3, 5).
		// Composes cleanly with mode-3's AdR: the AdR cutoff just feeds into
		// compute_conic_from_transmat() the same way the fixed 4σ cutoff does.
		// AdR shrinks the per-Gauss ellipse for low-α surfels; AccuTile then
		// enumerates only the tiles where the shrunken ellipse actually
		// intersects — the two tightening effects compound. The rect-AABB code
		// path below is still reached as a fallback for numerically-degenerate
		// conics (disc → 0⁻).
		bool use_snugbox = (aabb_mode == 2 || aabb_mode == 3 || aabb_mode == 5);

		if (use_snugbox) {
			if (compute_conic_from_transmat(T, cutoff, A_c, B_c, E_c, t_c, p_c)) {
				n_tiles_sb = duplicateToTilesTouched(
					A_c, B_c, E_c, t_c, p_c, grid,
					0, 0, 0.0f, nullptr, nullptr);
				// Hard upper bound: AccuTile's count must be ≤ rect's count.
				// If it isn't, the conic was near-degenerate (disc → 0⁻) and the
				// scan inflated outside the rect — fall through to rect.
				if (n_tiles_sb > 0 && n_tiles_sb <= rect_tiles) {
					snugbox_succeeded = true;
					point_image = p_c;  // conic center == compute_aabb's center
				}
			}
		}

	}
	else {
		// `--method mixed_3d` UNTEXTURED: FastGS EWA 3D-ellipsoid geometry.
		// AdR + AccuTile parity with the textured 2DGS+SnugBox path:
		//   * Mahalanobis cutoff is OPACITY-AWARE — t_cut = 2·log(255·α) so the
		//     binning iso-line is exactly where α·exp(-m/2) hits the per-Gauss
		//     1/255 alpha-floor. Anything outside would be culled in the inner
		//     loop's `alpha_e < 1/255` check anyway, so the tighter bound
		//     loses zero renderable contribution.
		//   * Tile count is the **ellipse-tight** AccuTile emission via
		//     duplicateToTilesTouched — same helper the textured SnugBox path
		//     uses. The conic format (A, B, E, t, p) is exactly the EWA path's
		//     (inverse-cov, Mahalanobis-sq cutoff, projected center), so no
		//     adapter / separate implementation is needed; the downstream
		//     conic_t / create_instances dispatch (lines 684+) handles both
		//     producers transparently.
		//   * Rect fallback (only when AccuTile reports degenerate) is per-axis
		//     tight from Σ = (Σ⁻¹)⁻¹  ⇒  Σ.xx = E/det, Σ.yy = A/det — NOT the
		//     legacy 3σ-of-maxλ square that was overshoooting rotated /
		//     elongated ellipsoids by up to 1.5×.
		glm::vec3 scale3 = glm::vec3(scales[idx].x, scales[idx].y, scaling_z[idx]);
		float3 conic_e; float my_radius_legacy;
		if (!compute_ewa_conic(p_orig, scale3, scale_modifier, rotations[idx],
				viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
				W, H, conic_e, point_image, my_radius_legacy))
			return;

		// Opacity-aware Mahalanobis-squared cutoff (≈ FastGS Compact Box for
		// the textured path, but in the EWA conic's natural form).
		float opa = opacities[idx];
		if (opa < (1.0f / 255.0f)) return;
		float t_cut = fmaxf(0.5f, 2.0f * logf(255.0f * opa));

		// Per-axis tight rect AABB (fallback + AccuTile upper-bound).
		// Σ = K⁻¹ where K = (a,b,c) is the inverse cov; det(K) = ac − b².
		// Σ.xx = c / det, Σ.yy = a / det. Half-widths at Mahalanobis = √t_cut.
		float det = conic_e.x * conic_e.z - conic_e.y * conic_e.y;
		if (det <= 0.0f) return;  // degenerate (near-edge-on or numerically dead)
		float r_cut = sqrtf(t_cut);
		rx = (int)ceilf(r_cut * sqrtf(fmaxf(0.0f, conic_e.z / det)));
		ry = (int)ceilf(r_cut * sqrtf(fmaxf(0.0f, conic_e.x / det)));
		getRectXY(point_image, rx, ry, rect_min, rect_max, grid);
		rect_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		if (rect_tiles == 0) return;

		// AccuTile ellipse-tight tile count. EWA conic IS the (A, B, E)
		// quadratic-form coefficients; t = Mahalanobis² cutoff; p = center.
		A_c = conic_e.x; B_c = conic_e.y; E_c = conic_e.z; t_c = t_cut;
		p_c = point_image;
		n_tiles_sb = duplicateToTilesTouched(
			A_c, B_c, E_c, t_c, p_c, grid,
			0, 0, 0.0f, nullptr, nullptr);
		// Same upper-bound sanity check the textured path uses: AccuTile's
		// count must be ≤ rect's count, else the conic was near-degenerate
		// (disc → 0⁻) and the scan inflated — fall through to rect.
		if (n_tiles_sb > 0 && n_tiles_sb <= rect_tiles) {
			snugbox_succeeded = true;
			// EWA's center == conic center == point_image already; nothing
			// to fix up (unlike the 2DGS path where compute_aabb's center
			// drifts and p_c is the corrected one).
		}

		ewa_conic[idx] = make_float4(conic_e.x, conic_e.y, conic_e.z, opa);
		normal = make_float3(0.0f, 0.0f, 1.0f);  // untextured: no surfel normal
	}

	// FP16 SH-color storage: range is [0, ~5] post-bias+clamp — well within
	// FP16 precision and halves the global-mem bandwidth of the inner-loop fetch
	// in renderBakedCUDA.
	if (colors_precomp == nullptr) {
		glm::vec3 result;
		if (voronoi_K > 0 && voronoi_sites != nullptr
		    && voronoi_tau != nullptr && voronoi_colors != nullptr) {
			// --feature SV: per-Gaussian softmax-mix in pre-activated form.
			// Bit-equivalent to nest's torch eval_voronoi_sv_feat → fake-SH-DC
			// → computeColorFromSH path, but skips the autograd graph + fake-SH
			// memcpy roundtrip (~1 ms/frame on a 5090 saved at this scale).
			result = computeColorFromVoronoi(
				idx, voronoi_K,
				(const glm::vec3*)orig_points, *cam_pos,
				voronoi_sites, voronoi_tau, voronoi_colors);
		} else {
			result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs);
		}
		rgb[idx * C + 0] = __float2half(result.x);
		rgb[idx * C + 1] = __float2half(result.y);
		rgb[idx * C + 2] = __float2half(result.z);
	} else {
		for(int i = 0; i < C; i++){
			rgb[idx * C + i] = __float2half(colors_precomp[idx * C + i]);
		}
	}

	// SB per-Gaussian color, computed once. Was per-pixel in renderBakedCUDA
	// but view_dir only depends on the Gaussian's mean → recomputing inside
	// the inner loop wasted ~K*4 transcendentals per touched pixel. The
	// activation chain stays correct because SB is added linearly to feat[]
	// before the outer ReLU; pre-evaluating it here is associative.
	if (sb_number > 0 && sb_params != nullptr && sb_rgb_out != nullptr) {
		const float* sbp = sb_params + idx * sb_number * 6;
		glm::vec3 gc = glm::vec3(orig_points[idx * 3 + 0],
		                         orig_points[idx * 3 + 1],
		                         orig_points[idx * 3 + 2]);
		glm::vec3 cp = *cam_pos;
		glm::vec3 vd = gc - cp;
		vd = vd / (glm::length(vd) + 1e-8f);
		glm::vec3 sb = eval_sb(sbp, sb_number, vd);
		sb_rgb_out[idx * 3 + 0] = __float2half(sb.x);
		sb_rgb_out[idx * 3 + 1] = __float2half(sb.y);
		sb_rgb_out[idx * 3 + 2] = __float2half(sb.z);
	}

	depths[idx] = p_view.z;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};

	// LEAN_CONIC: precompute (u₀, v₀) + J⁻¹ + (Tw.x/w_c, Tw.y/w_c) at AABB-center pixel.
	// The (Tw.x/w_c, Tw.y/w_c) pair enables an EXACT rational-function reconstruction
	// per pixel: u = u₀ + (J⁻¹·Δpix).x / (1 + (dwdxr, dwdyr)·Δpix). No linearization
	// drift because u = p.x/p.z with p.x, p.z BOTH linear in pix → u is exactly a
	// rational function of pixel, which this formula reproduces bit-perfectly.
	if (conic_uv != nullptr) {
		float Tux = T[0].x, Tuy = T[0].y, Tuz = T[0].z;
		float Tvx = T[1].x, Tvy = T[1].y, Tvz = T[1].z;
		float Twx = T[2].x, Twy = T[2].y, Twz = T[2].z;
		// Ray-splat at AABB center → (u₀, v₀).
		float3 k_c = { point_image.x * Twx - Tux, point_image.x * Twy - Tuy, point_image.x * Twz - Tuz };
		float3 l_c = { point_image.y * Twx - Tvx, point_image.y * Twy - Tvy, point_image.y * Twz - Tvz };
		float3 p_c = cross(k_c, l_c);
		// Default to "always cull" values (u₀ huge → rho3d always > cutoff)
		// so degenerate Gauss don't smear.  Fixes the residual drift bug on
		// bicycle/bonsai where near-camera Gauss with |p_c.z|<1e-12 previously
		// wrote (u₀=v₀=0) → rho3d=0 → alpha≈1 at every pixel → color smear.
		float u0 = 1e10f, v0 = 1e10f;
		float Jinv00 = 0.0f, Jinv01 = 0.0f, Jinv10 = 0.0f, Jinv11 = 0.0f;
		float dwdxr = 0.0f, dwdyr = 0.0f;
		if (fabsf(p_c.z) > 1e-12f) {
			u0 = p_c.x / p_c.z;
			v0 = p_c.y / p_c.z;
			// M = -[[k_c.x, k_c.y], [l_c.x, l_c.y]] / w_c, J⁻¹ = M⁻¹.
			float w_c = Twx * u0 + Twy * v0 + Twz;
			float det_kl = k_c.x * l_c.y - k_c.y * l_c.x;
			if (fabsf(det_kl) > 1e-12f && fabsf(w_c) > 1e-8f) {
				float scale = w_c / det_kl;
				Jinv00 = -l_c.y * scale;
				Jinv01 =  k_c.y * scale;
				Jinv10 =  l_c.x * scale;
				Jinv11 = -k_c.x * scale;
				// Exact rational correction: u = u₀ + Δu_lin / (1 + Δp.z/p_c.z).
				// p.z = k.x·l.y - k.y·l.x IS LINEAR in pix (the pix.x·pix.y terms in
				// the cross product cancel).  Its gradient is constant:
				//   ∂p.z/∂pix.x = Tw.y·Tv.x - Tw.x·Tv.y
				//   ∂p.z/∂pix.y = Tu.y·Tw.x - Tu.x·Tw.y
				// (Confirmed by expanding cross(k, l) and collecting pix terms.)
				// Store these normalized by p_c.z so per-fragment we just compute
				// (dwdxr·Δpx + dwdyr·Δpy) → the fractional change to divide by.
				float dpz_dpx = Twy * Tvx - Twx * Tvy;
				float dpz_dpy = Tuy * Twx - Tux * Twy;
				dwdxr = dpz_dpx / p_c.z;
				dwdyr = dpz_dpy / p_c.z;
			} else {
				// Inner guard failed → force cull (values stay at 1e10/0 defaults).
				u0 = 1e10f; v0 = 1e10f;
				Jinv00 = 0.0f; Jinv01 = 0.0f; Jinv10 = 0.0f; Jinv11 = 0.0f;
				dwdxr = 0.0f; dwdyr = 0.0f;
			}
		}
		conic_uv[idx * 8 + 0] = u0;
		conic_uv[idx * 8 + 1] = v0;
		conic_uv[idx * 8 + 2] = Jinv00;
		conic_uv[idx * 8 + 3] = Jinv01;
		conic_uv[idx * 8 + 4] = Jinv10;
		conic_uv[idx * 8 + 5] = Jinv11;
		conic_uv[idx * 8 + 6] = dwdxr;
		conic_uv[idx * 8 + 7] = dwdyr;
	}

	uint32_t my_tile_count;
	if (snugbox_succeeded) {
		// Cache conic for create_instances' AccuTile re-walk.
		conic_t[idx] = make_float4(A_c, B_c, E_c, t_c);
		radii[idx] = 1;        // visible flag (SnugBox path doesn't produce a single radius)
		radii_x[idx] = 0;      // unused on SnugBox path
		radii_y[idx] = 0;
		my_tile_count = n_tiles_sb;
	} else {
		conic_t[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // mark as "use rect AABB"
		radii[idx] = max(rx, ry);
		radii_x[idx] = rx;
		radii_y[idx] = ry;
		my_tile_count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}

	tiles_touched[idx] = my_tile_count;  // per-id (used by both modes)
	if (sort_mode == 1 && my_tile_count > 0) {
		// FastGS compaction: push visible Gaussians into the depth-sort input.
		const uint32_t v_idx = atomicAdd(n_visible_atomic, 1u);
		depth_keys_compact[v_idx] = __float_as_uint(p_view.z);  // 32-bit float-bit key
		prim_idx_compact[v_idx] = (uint32_t)idx;
		atomicAdd(n_instances_atomic, my_tile_count);
	}
}

// =============================================================================
// Render kernel for baked mode: SH base color + bilinear residual texture lookup
// =============================================================================
// LEAN_CTG: compile-time-gate kernel_type and has_atlas.  When set, LEAN_KT
// selects the kernel_type (default 4 = beta_scaled).  atlas branch is enabled.
// The compiler then eliminates the OTHER runtime branches.
#ifdef LEAN_CTG
  #ifndef LEAN_KT
    #define LEAN_KT 4
  #endif
  #define KT_EFF LEAN_KT
  #define HAS_ATLAS_EFF true
#else
  #define KT_EFF kernel_type
  #define HAS_ATLAS_EFF (atlas_rects != nullptr)
#endif

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
	const float* __restrict__ sb_params,           // [N, K, 6] SB params (legacy; unused — sb_rgb_in is per-Gauss)
	const int sb_number,                           // K lobes per Gaussian (0 = SB disabled)
	const __half* __restrict__ sb_rgb_in,          // [N, 3] SB RGB precomputed in preprocessCUDA (read when sb_number > 0)
	cudaTextureObject_t atlas_tex_obj,             // hardware texture object for atlas (required)
	float atlas_offset,                            // dequantization offset (uint8 atlas)
	float atlas_scale,                             // dequantization scale  (uint8 atlas)
	// `--method mixed_3d`: per-Gauss textured flag + EWA conic. nullptr
	// ewa_conic → pure 2DGS (unchanged). Untextured rows use the EWA conic
	// falloff; the per-Gauss SV/SH color path below is shared (untextured
	// carry a zero atlas rect → SV/SH-only, same as `--method mixed`).
	const bool* __restrict__ is_textured = nullptr,
	const float4* __restrict__ ewa_conic = nullptr,
	const float* __restrict__ ewa_depths = nullptr,  // geomState.depths (camera z)
	// LEAN_CONIC Option A: [P*6] per-Gauss (u0,v0,J⁻¹) cache from preprocessCUDA.
	const float* __restrict__ conic_uv = nullptr)
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

	// LEAN Tier 1: no RVQ, no SB, no mixed_3d.
	// -----------------------------------------------------------------------
	// LEAN_T2  → T-matrix, normal_opacity, shapes stored as fp16 in shared.
	//   T2 OFF: 68 B/Gauss (id 4 + xy 8 + normal_opa 16 + T 36 + shape 4)
	//   T2 ON:  42 B/Gauss (id 4 + xy 8 + normal_opa 8  + T 18 + shape 2  + 2 pad)
	// LEAN_CTG → template render kernel on <kernel_type, has_atlas>
	// LEAN_CONIC → replace T-matrix ray-splat with precomputed 2D screen conic
	// -----------------------------------------------------------------------
	__shared__ int    collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
#ifdef LEAN_CONIC
	// Option A + exact correction + SAFE FP16 PACK + PRECOMPUTED ATLAS UV.
	// Per-Gauss shared: previous 36 B + atlas UV precomp.
	//   id 4  |  xy 8 (fp32)  |  opa 2 (fp16)  |  J⁻¹ 8 (half4)
	//   uv₀ 4 (half2)  |  dw 8 (fp32)  |  shape 2 (fp16)
	// + Atlas UV precompute (skip fetching atlas_rects per fragment + eliminate 2 divs):
	//   auv_base 4 (half2 = au_base, av_base — mapped from pixel coords to fp16
	//                by subtracting a per-Gauss offset then packing)... too fragile.
	//   Use fp32 for atlas UV to keep pixel precision:
	//   auv_base 8 (fp32 float2)  |  auv_scale 8 (fp32 float2)  |  auv_span 8 (fp32 float2)
	// Total: 4+8+2+8+4+8+2 + 8+8+8 = 60 B/Gauss (was 36 → +24 B for atlas UV precomp).
	__shared__ __half collected_opa[BLOCK_SIZE];
	__shared__ __half collected_J_h[BLOCK_SIZE * 4];
	__shared__ __half collected_uv0_h[BLOCK_SIZE * 2];
	__shared__ float2 collected_dwdpxy[BLOCK_SIZE];
	__shared__ __half collected_shapes_h[BLOCK_SIZE];
	// Atlas UV precompute — per-fragment reduces to:
	//   au = clamp(auv_base.x + auv_scale.x * s.x, auv_min.x, auv_max.x)
	// Sentinel: auv_scale.x == 0 → zero-area atlas rect, skip the fetch.
	__shared__ float2 collected_auv_base[BLOCK_SIZE];     // (au_base, av_base)  ← u0_px - 0.5 + u_span/2
	__shared__ float2 collected_auv_scale[BLOCK_SIZE];    // (au_scale, av_scale)  ← span/(2·UV_EXTENT)
	__shared__ float2 collected_auv_min[BLOCK_SIZE];      // (u0_px, v0_px)
	__shared__ float2 collected_auv_max[BLOCK_SIZE];      // (u0_px+u_span-1.001, v0…)
#else
	#ifdef LEAN_T2
	__shared__ __half collected_no_h[BLOCK_SIZE * 4];   // 8 B/Gauss  (normal + opacity)
	__shared__ __half collected_T_h[BLOCK_SIZE * 9];    // 18 B/Gauss (Tu, Tv, Tw)
	__shared__ __half collected_shapes_h[BLOCK_SIZE];   // 2 B/Gauss
	#else
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float  collected_shapes[BLOCK_SIZE];
	#endif
#endif

	float T = 1.0f;
	float C[3] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Fetch per-Gaussian data — no is_textured, no ewa_conic.
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int t = block.thread_rank();
			int coll_id = point_list[range.x + progress];
			collected_id[t] = coll_id;
			float2 xy_c = points_xy_image[coll_id];
			// Under LEAN_CONIC, collected_xy will be overwritten with the disc
			// origin below; leave the AABB-center default here for the non-CONIC path.
			collected_xy[t] = xy_c;
			float4 no = normal_opacity[coll_id];
			float shape_v = (shapes != nullptr) ? shapes[coll_id] : 0.0f;
#ifdef LEAN_CONIC
			// Load precomputed (u₀, v₀, J⁻¹, dwdxr, dwdyr) from GLOBAL into SHARED.
			// preprocessCUDA did the ray-splat + Jacobian + dw/dpix.  Just relay 8 fp32.
			const float* cuv_g = conic_uv + coll_id * 8;
			// Pack: opa/uv0/J⁻¹/shapes as fp16, dwdpxy stays fp32 for correction stability.
			collected_opa[t]        = __float2half(no.w);              // drop normal.xyz
			collected_uv0_h[t*2+0]  = __float2half(cuv_g[0]);
			collected_uv0_h[t*2+1]  = __float2half(cuv_g[1]);
			collected_J_h[t*4+0]    = __float2half(cuv_g[2]);
			collected_J_h[t*4+1]    = __float2half(cuv_g[3]);
			collected_J_h[t*4+2]    = __float2half(cuv_g[4]);
			collected_J_h[t*4+3]    = __float2half(cuv_g[5]);
			collected_dwdpxy[t]     = make_float2(cuv_g[6], cuv_g[7]); // FP32
			collected_shapes_h[t]   = __float2half(shape_v);
			// Atlas UV precomp (read atlas_rects once per Gauss at fetch, not per fragment).
			// UV_EXTENT is a global constant (=4.0).  Sentinel: auv_scale.x == 0 means
			// zero-area rect → skip the atlas fetch in the fragment.
			if (atlas_rects != nullptr) {
				float u0_px  = atlas_rects[coll_id * 4 + 0];
				float v0_px  = atlas_rects[coll_id * 4 + 1];
				float u_span = atlas_rects[coll_id * 4 + 2];
				float v_span = atlas_rects[coll_id * 4 + 3];
				const float inv_2E = 1.0f / (2.0f * UV_EXTENT);
				float au_scale = (u_span > 0.0f) ? (u_span * inv_2E) : 0.0f;
				float av_scale = (v_span > 0.0f) ? (v_span * inv_2E) : 0.0f;
				collected_auv_scale[t] = make_float2(au_scale, av_scale);
				collected_auv_base[t]  = make_float2(u0_px - 0.5f + u_span * 0.5f,
				                                     v0_px - 0.5f + v_span * 0.5f);
				collected_auv_min[t]   = make_float2(u0_px, v0_px);
				collected_auv_max[t]   = make_float2(u0_px + u_span - 1.001f,
				                                     v0_px + v_span - 1.001f);
			} else {
				collected_auv_scale[t] = make_float2(0.0f, 0.0f);  // sentinel: skip atlas
			}
#else
	#ifdef LEAN_T2
			collected_no_h[t*4+0] = __float2half(no.x);
			collected_no_h[t*4+1] = __float2half(no.y);
			collected_no_h[t*4+2] = __float2half(no.z);
			collected_no_h[t*4+3] = __float2half(no.w);
			#pragma unroll
			for (int c = 0; c < 9; c++)
				collected_T_h[t*9 + c] = __float2half(transMats[9 * coll_id + c]);
			collected_shapes_h[t] = __float2half(shape_v);
	#else
			collected_normal_opacity[t] = no;
			collected_Tu[t] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[t] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[t] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			collected_shapes[t] = shape_v;
	#endif
#endif
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// LEAN: mixed_3d EWA branch REMOVED.
			// LEAN: kernel_type=1/2/3/4 branches REMOVED (hardcoded Gaussian).
			const float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float2 s;
			float rho3d;
			float depth;
#ifdef LEAN_CONIC
			// Exact rational reconstruction: u = u₀ + Δu_linear / (1 + δw_ratio).
			// Δu_linear = J⁻¹·Δpix, δw_ratio = (dwdxr, dwdyr)·Δpix, both linear in Δpix.
			// Formula is EXACT (not linearized) because u = p.x/p.z with p.x, p.z
			// each linear in pix → u is exactly a rational function of pix.
			// FP16 unpack — u₀, v₀, J⁻¹ stored as half; dw stays fp32.
			float2 uv0 = make_float2(__half2float(collected_uv0_h[j*2+0]),
			                          __half2float(collected_uv0_h[j*2+1]));
			float4 J = make_float4(__half2float(collected_J_h[j*4+0]),
			                        __half2float(collected_J_h[j*4+1]),
			                        __half2float(collected_J_h[j*4+2]),
			                        __half2float(collected_J_h[j*4+3]));
			float2 dw = collected_dwdpxy[j];
			float dx = (float)pix.x - xy.x;
			float dy = (float)pix.y - xy.y;
			float du_lin = J.x * dx + J.y * dy;
			float dv_lin = J.z * dx + J.w * dy;
			float denom  = 1.0f + dw.x * dx + dw.y * dy;
			// Correction denominator crosses zero at the linearization's validity
			// boundary. Past that boundary, the formula's sign flips and it renders
			// wrong colors (source of the "foggy tile discontinuity" artifact on
			// large near-camera Gauss).  Cull those pixels — they're outside the
			// Gauss's actual footprint anyway.
			if (denom < 0.1f) continue;
			float inv_d  = 1.0f / denom;
			float u = uv0.x + du_lin * inv_d;
			float v = uv0.y + dv_lin * inv_d;
			s.x = u;
			s.y = v;
			rho3d = u * u + v * v;
			depth = 1.0f;                // depth cull happens in preprocess
#else
	#ifdef LEAN_T2
			const float3 Tu = {__half2float(collected_T_h[j*9+0]), __half2float(collected_T_h[j*9+1]), __half2float(collected_T_h[j*9+2])};
			const float3 Tv = {__half2float(collected_T_h[j*9+3]), __half2float(collected_T_h[j*9+4]), __half2float(collected_T_h[j*9+5])};
			const float3 Tw = {__half2float(collected_T_h[j*9+6]), __half2float(collected_T_h[j*9+7]), __half2float(collected_T_h[j*9+8])};
	#else
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
	#endif
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			s.x = p.x / p.z;
			s.y = p.y / p.z;
			rho3d = s.x * s.x + s.y * s.y;
			depth = (rho3d <= FilterInvSquare * (d.x*d.x + d.y*d.y))
			        ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
#endif
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
			float rho = min(rho3d, rho2d);
#ifndef LEAN_CONIC
			// Under LEAN_CONIC we don't reconstruct per-fragment depth (it isn't
			// needed for color-only output), and every Gauss survives preprocessCUDA
			// with p_view.z >= near — so this per-fragment check is a no-op.
			if (depth < near_n) continue;
#endif
#ifdef LEAN_CONIC
			float opa = __half2float(collected_opa[j]);
#else
	#ifdef LEAN_T2
			float opa = __half2float(collected_no_h[j*4+3]);
	#else
			float4 nor_o = collected_normal_opacity[j];
			float opa = nor_o.w;
	#endif
#endif

			// LEAN kernel dispatch: Gaussian (kernel_type=0) or beta_scaled (=4).
			// Beta_scaled path is required for RD-family bakes (compact 3σ support
			// + Gaussian low-pass max-pool). Everything else routes to Gaussian.
			// Under LEAN_CTG, KT_EFF is a compile-time constant → dead branch elided.
			float alpha;
			if (KT_EFF == 4 && shapes != nullptr) {
				// beta_scaled: k²=9 compact support + fp16 low-pass max-pool.
				constexpr float k_sq = 9.0f;
				if (rho3d >= k_sq + 1e-6f) continue;
#ifdef LEAN_LOCK_SHAPE
				// Perf-ceiling A/B: shape locked to 1 (linear, fattest single-mul
				// kernel), no shape load, no powf. Skips the biggest per-fragment
				// arithmetic cost (~30-50 cycles on mobile, ~4 on desktop) to see
				// how much of the frame is really the pow fallback.
				float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
				float alpha_beta = base;                    // == powf(base, 1.0f)
				float alpha_lp   = expf(-rho2d / 2.0f);
				float kernel_val = fmaxf(alpha_beta, alpha_lp);
				alpha = fminf(0.99f, opa * kernel_val);
#else
#if defined(LEAN_CONIC) || defined(LEAN_T2)
				float shape = __half2float(collected_shapes_h[j]);
#else
				float shape = collected_shapes[j];
#endif
				float base = fmaxf(0.0f, 1.0f - rho3d / k_sq);
				float alpha_beta = powf(base, shape);
				float alpha_lp   = expf(-rho2d / 2.0f);
				float kernel_val = fmaxf(alpha_beta, alpha_lp);
				alpha = fminf(0.99f, opa * kernel_val);
#endif
			} else {
				float power = -0.5f * rho;
				if (power > 0.0f) continue;
				alpha = fminf(0.99f, opa * expf(power));
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

			// LEAN: SH base color from preprocess. No SH_add snapshot (mode 1
			// unsupported here); no residual texture; no SB add.
			int gauss_id = collected_id[j];
			float feat[3] = {
				__half2float(features[gauss_id * 3 + 0]),
				__half2float(features[gauss_id * 3 + 1]),
				__half2float(features[gauss_id * 3 + 2]),
			};

			// LEAN atlas sample — HW bilinear texture ONLY (no RVQ, no SB,
			// no SW fallback). Under LEAN_CTG, HAS_ATLAS_EFF is compile-time
			// true — no runtime null-check on atlas_rects.
#ifdef LEAN_CONIC
			if (HAS_ATLAS_EFF && atlas_tex_obj != 0) {
				// PRECOMPUTED atlas UV — atlas_rects fetch, divisions, and base
				// math all moved to fetch block.  Per-fragment: 2 fmadd + 4 clamp.
				float2 auv_scale = collected_auv_scale[j];
				if (auv_scale.x > 0.0f) {                       // sentinel: >0 → valid
					float2 auv_base = collected_auv_base[j];
					float2 auv_min  = collected_auv_min[j];
					float2 auv_max  = collected_auv_max[j];
					float au = fmaxf(auv_min.x, fminf(auv_max.x, auv_base.x + auv_scale.x * s.x));
					float av = fmaxf(auv_min.y, fminf(auv_max.y, auv_base.y + auv_scale.y * s.y));
					float4 rgba = tex2D<float4>(atlas_tex_obj, au + 0.5f, av + 0.5f);
					feat[0] += rgba.x * atlas_scale + atlas_offset;
					feat[1] += rgba.y * atlas_scale + atlas_offset;
					feat[2] += rgba.z * atlas_scale + atlas_offset;
				}
			}
#else
			if (HAS_ATLAS_EFF && atlas_tex_obj != 0) {
				float u0_px  = atlas_rects[gauss_id * 4 + 0];
				float v0_px  = atlas_rects[gauss_id * 4 + 1];
				float u_span = atlas_rects[gauss_id * 4 + 2];
				float v_span = atlas_rects[gauss_id * 4 + 3];
				if (u_span > 0.0f && v_span > 0.0f) {
					float au = u0_px + (s.x + UV_EXTENT) / (2.0f * UV_EXTENT) * u_span - 0.5f;
					float av = v0_px + (s.y + UV_EXTENT) / (2.0f * UV_EXTENT) * v_span - 0.5f;
					au = fmaxf(u0_px, fminf(u0_px + u_span - 1.001f, au));
					av = fmaxf(v0_px, fminf(v0_px + v_span - 1.001f, av));
					float4 rgba = tex2D<float4>(atlas_tex_obj, au + 0.5f, av + 0.5f);
					feat[0] += rgba.x * atlas_scale + atlas_offset;
					feat[1] += rgba.y * atlas_scale + atlas_offset;
					feat[2] += rgba.z * atlas_scale + atlas_offset;
				}
			}
#endif

			// LEAN activation: mode 0 (3D_SH_res) only — ReLU(feat + res_bias).
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
	const __half* sb_rgb_in,
	cudaTextureObject_t atlas_tex_obj,
	float atlas_offset,
	float atlas_scale,
	const bool* is_textured,
	const float4* ewa_conic,
	const float* conic_uv)
{
	// Optional dynamic shared for RVQ codebook (set via setRVQSharedBytes
	// helper before this launch). 0 = no dyn shared. `::` forces global
	// scope so we don't accidentally resolve to FORWARD::g_rvq_shared_bytes.
	renderBakedCUDA<<<grid, block, ::g_rvq_shared_bytes>>>(
		ranges, point_list, beta, W, H,
		points_xy_image, features, transMats, depths, normal_opacity,
		bg_color, out_color,
		shapes, kernel_type,
		means3D, cam_pos,
		atlas_texture, atlas_rects, atlas_width,
		sb_params, sb_number, sb_rgb_in, atlas_tex_obj,
		atlas_offset, atlas_scale,
		is_textured, ewa_conic, depths,
		conic_uv);
}

// Host-side cached RVQ shared-codebook byte count. Set by SetAtlasRVQCUDA in
// rasterize_points.cu. Used by FORWARD::render to size the dynamic shared
// memory at kernel launch.
size_t g_rvq_shared_bytes = 0;

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

__global__ void setBakeBetaMultKernel(float val) { d_beta_mult = val; }
void FORWARD::setBetaMult(float val) {
	setBakeBetaMultKernel<<<1, 1>>>(val);
}

__global__ void setBakeOpacityAwareBetaKernel(bool val) { d_opacity_aware_beta = val; }
void FORWARD::setOpacityAwareBeta(bool val) {
	setBakeOpacityAwareBetaKernel<<<1, 1>>>(val);
}

__global__ void setBakeDropLowpassKernel(bool val) { d_drop_lowpass = val; }
void FORWARD::setDropLowpass(bool val) {
	setBakeDropLowpassKernel<<<1, 1>>>(val);
}

__global__ void setBakeResidualModeKernel(int mode) { d_residual_mode = mode; }
void FORWARD::setResidualMode(int mode) {
	setBakeResidualModeKernel<<<1, 1>>>(mode);
}

__global__ void setBakeUntexKernelKernel(int v) { d_untex_kernel = v; }
void FORWARD::setUntexKernel(int v) {
	setBakeUntexKernelKernel<<<1, 1>>>(v);
}

__global__ void setAtlasRVQKernel(
	const __half* cb, const uint8_t* idx, const int64_t* off,
	int L, int K, int B, unsigned long long N_used
) {
	d_rvq_codebooks      = cb;
	d_rvq_indices        = idx;
	d_rvq_surfel_offsets = off;
	d_rvq_L = L; d_rvq_K = K; d_rvq_B = B;
	d_rvq_N_used = N_used;
}
void FORWARD::setAtlasRVQ(
	const __half* codebooks, const uint8_t* indices, const int64_t* surfel_offsets,
	int L, int K, int B, unsigned long long N_used
) {
	setAtlasRVQKernel<<<1, 1>>>(codebooks, indices, surfel_offsets, L, K, B, N_used);
}

__global__ void clearAtlasRVQKernel() {
	d_rvq_codebooks      = nullptr;
	d_rvq_indices        = nullptr;
	d_rvq_surfel_offsets = nullptr;
	d_rvq_L = 0; d_rvq_K = 0; d_rvq_N_used = 0;
}
void FORWARD::clearAtlasRVQ() { clearAtlasRVQKernel<<<1, 1>>>(); }

__global__ void setAtlasRVQBilinearKernel(int v) { d_rvq_bilinear = v; }
void FORWARD::setAtlasRVQBilinear(int v) { setAtlasRVQBilinearKernel<<<1, 1>>>(v); }

__global__ void setRVQUseSharedCBKernel(int v) { d_rvq_use_shared_cb = v; }
void FORWARD::setAtlasRVQUseSharedCB(int v) { setRVQUseSharedCBKernel<<<1, 1>>>(v); }

__global__ void setRVQTexKernel(cudaTextureObject_t cb, cudaTextureObject_t idx,
                                 int use_cb, int use_idx) {
	d_rvq_cb_tex  = cb;
	d_rvq_idx_tex = idx;
	d_rvq_use_tex_cb  = use_cb;
	d_rvq_use_tex_idx = use_idx;
}
void FORWARD::setAtlasRVQTex(cudaTextureObject_t cb, cudaTextureObject_t idx,
                              int use_cb, int use_idx) {
	setRVQTexKernel<<<1, 1>>>(cb, idx, use_cb, use_idx);
}

__global__ void setRVQCBDequantKernel(float s, float o) {
	d_rvq_cb_dq_scale  = s;
	d_rvq_cb_dq_offset = o;
}
void FORWARD::setAtlasRVQCBDequant(float scale, float offset) {
	setRVQCBDequantKernel<<<1, 1>>>(scale, offset);
}

bool FORWARD::optInRVQShared(int bytes) {
	// First check the device limit. Returns the maximum opt-in shared/block.
	int dev = 0; cudaGetDevice(&dev);
	int max_opt_in = 0;
	cudaDeviceGetAttribute(&max_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	if (bytes > max_opt_in) {
		printf("[BAKE_RENDER] requested %d B shared > device max %d B per block; "
		       "cannot opt in\n", bytes, max_opt_in);
		return false;
	}
	// Pass the kernel by name (CUDA-standard); the function-template-instance
	// is uniquely identified by its symbol.
	cudaError_t err = cudaFuncSetAttribute(
		renderBakedCUDA,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		bytes);
	if (err != cudaSuccess) {
		printf("[BAKE_RENDER] cudaFuncSetAttribute MaxDynamicSharedMemorySize=%d "
		       "failed: %s (device max opt-in = %d B)\n",
		       bytes, cudaGetErrorString(err), max_opt_in);
		cudaGetLastError();        // clear the sticky error
		return false;
	}
	printf("[BAKE_RENDER] opt-in MaxDynamicSharedMemorySize=%d B (device max=%d B)\n",
	       bytes, max_opt_in);
	return true;
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
	float4* conic_t,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* depth_keys_compact,
	uint32_t* prim_idx_compact,
	uint32_t* n_visible_atomic,
	uint32_t* n_instances_atomic,
	const int sort_mode,
	bool prefiltered,
	const float* shapes,
	const int kernel_type,
	const int aabb_mode,
	const float* voronoi_sites,
	const float* voronoi_tau,
	const float* voronoi_colors,
	const int voronoi_K,
	const float* sb_params,
	const int sb_number,
	__half* sb_rgb_out,
	const bool* is_textured,
	const float* scaling_z,
	float4* ewa_conic,
	float* conic_uv)
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
		conic_t,
		grid,
		tiles_touched,
		depth_keys_compact,
		prim_idx_compact,
		n_visible_atomic,
		n_instances_atomic,
		sort_mode,
		prefiltered,
		shapes,
		kernel_type,
		aabb_mode,
		voronoi_sites,
		voronoi_tau,
		voronoi_colors,
		voronoi_K,
		sb_params,
		sb_number,
		sb_rgb_out,
		is_textured,
		scaling_z,
		ewa_conic,
		conic_uv
	);
}
