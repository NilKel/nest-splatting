// CUDA kernel for per-fragment RVQ atlas decode — the work a WGSL fragment
// shader would do. Used by scripts/bench_rvq_shader_decode_cuda.py.
//
// Decodes:
//   for each fragment t:
//     g = surfel_ids[t]
//     (u, v) = atlas_uvs[t]  // atlas pixel coordinates (float)
//     bid = surfel_offsets[g] + ((v - v0_g)/B) * (w_g/B) + ((u - u0_g)/B)
//     intra_idx = (intra_v) * B + intra_u
//     rgb = Σ_{l=0..L-1} codebooks[l, indices[l, bid], intra_idx]
//
// Two entry points:
//   rvq_decode_nearest_cuda  — 1 block lookup per fragment
//   rvq_decode_bilinear_cuda — 4 block lookups per fragment (true bilinear)
//
// Codebook stored FP16 to halve memory bandwidth (fits in L1 at 96 KB).
// Indices stored uint8 (K ≤ 256).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

template <bool BILINEAR>
__global__ void rvq_decode_kernel(
    const __half* __restrict__ codebooks,       // [L, K, 48]
    const uint8_t* __restrict__ indices,         // [L, N_used] surfel-major
    const int64_t* __restrict__ surfel_offsets,  // [M+1]
    const int32_t* __restrict__ rects,           // [M, 4] (u0, v0, w, h)
    const int64_t* __restrict__ surfel_ids,      // [n_frag]
    const float* __restrict__ atlas_uvs,         // [n_frag, 2]
    float* __restrict__ out_rgb,                 // [n_frag, 3]
    int n_frag, int L, int K, int N_used, int B
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_frag) return;

    int64_t g = surfel_ids[t];
    float u_atlas = atlas_uvs[t * 2 + 0];
    float v_atlas = atlas_uvs[t * 2 + 1];
    int u0 = rects[g * 4 + 0];
    int v0 = rects[g * 4 + 1];
    int w  = rects[g * 4 + 2];
    int h  = rects[g * 4 + 3];
    int bw = w / B;          // blocks per row in this surfel
    int bh = h / B;
    int64_t surfel_base = surfel_offsets[g];

    auto sample_at = [&](float u_a, float v_a, float* rgb) {
        int local_u = (int)floorf(u_a) - u0;
        int local_v = (int)floorf(v_a) - v0;
        local_u = max(0, min(local_u, w - 1));
        local_v = max(0, min(local_v, h - 1));
        int bu_local = local_u / B;
        int bv_local = local_v / B;
        int intra_u  = local_u - bu_local * B;
        int intra_v  = local_v - bv_local * B;
        int64_t bid  = surfel_base + (int64_t)bv_local * bw + bu_local;
        int     intra_idx = intra_v * B + intra_u;   // 0..15

        float r = 0.f, g_ = 0.f, b = 0.f;
        #pragma unroll 4
        for (int l = 0; l < L; ++l) {
            int code = (int)indices[(int64_t)l * N_used + bid];
            const __half* cw = codebooks + ((int64_t)l * K + code) * (B*B*3) + intra_idx * 3;
            r  += __half2float(cw[0]);
            g_ += __half2float(cw[1]);
            b  += __half2float(cw[2]);
        }
        rgb[0] = r; rgb[1] = g_; rgb[2] = b;
    };

    if constexpr (!BILINEAR) {
        float rgb[3];
        sample_at(u_atlas, v_atlas, rgb);
        out_rgb[t*3 + 0] = rgb[0];
        out_rgb[t*3 + 1] = rgb[1];
        out_rgb[t*3 + 2] = rgb[2];
    } else {
        float au = u_atlas - 0.5f;
        float av = v_atlas - 0.5f;
        float au0_f = floorf(au); float av0_f = floorf(av);
        float fu = au - au0_f;    float fv = av - av0_f;
        float c00[3], c01[3], c10[3], c11[3];
        sample_at(au0_f,       av0_f,       c00);
        sample_at(au0_f + 1.f, av0_f,       c01);
        sample_at(au0_f,       av0_f + 1.f, c10);
        sample_at(au0_f + 1.f, av0_f + 1.f, c11);
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float top = c00[c] * (1 - fu) + c01[c] * fu;
            float bot = c10[c] * (1 - fu) + c11[c] * fu;
            out_rgb[t*3 + c] = top * (1 - fv) + bot * fv;
        }
    }
}

template <bool BILINEAR>
torch::Tensor rvq_decode_impl(
    torch::Tensor codebooks,
    torch::Tensor indices,
    torch::Tensor surfel_offsets,
    torch::Tensor rects,
    torch::Tensor surfel_ids,
    torch::Tensor atlas_uvs,
    int B
) {
    TORCH_CHECK(codebooks.dtype() == torch::kFloat16, "codebooks must be FP16");
    TORCH_CHECK(indices.dtype() == torch::kUInt8, "indices must be uint8");
    TORCH_CHECK(surfel_offsets.dtype() == torch::kInt64, "surfel_offsets must be int64");
    TORCH_CHECK(rects.dtype() == torch::kInt32, "rects must be int32");
    TORCH_CHECK(surfel_ids.dtype() == torch::kInt64, "surfel_ids must be int64");
    TORCH_CHECK(atlas_uvs.dtype() == torch::kFloat32, "atlas_uvs must be float32");

    int n_frag = surfel_ids.size(0);
    int L = codebooks.size(0);
    int K = codebooks.size(1);
    int N_used = indices.size(1);

    auto opts = atlas_uvs.options();
    auto out = torch::empty({n_frag, 3}, opts);

    int threads = 256;
    int blocks = (n_frag + threads - 1) / threads;
    rvq_decode_kernel<BILINEAR><<<blocks, threads>>>(
        reinterpret_cast<const __half*>(codebooks.data_ptr<at::Half>()),
        indices.data_ptr<uint8_t>(),
        surfel_offsets.data_ptr<int64_t>(),
        rects.data_ptr<int32_t>(),
        surfel_ids.data_ptr<int64_t>(),
        atlas_uvs.data_ptr<float>(),
        out.data_ptr<float>(),
        n_frag, L, K, N_used, B
    );
    return out;
}

}  // namespace

torch::Tensor rvq_decode_nearest_cuda(
    torch::Tensor codebooks, torch::Tensor indices,
    torch::Tensor surfel_offsets, torch::Tensor rects,
    torch::Tensor surfel_ids, torch::Tensor atlas_uvs, int B
) {
    return rvq_decode_impl<false>(codebooks, indices, surfel_offsets,
                                  rects, surfel_ids, atlas_uvs, B);
}

torch::Tensor rvq_decode_bilinear_cuda(
    torch::Tensor codebooks, torch::Tensor indices,
    torch::Tensor surfel_offsets, torch::Tensor rects,
    torch::Tensor surfel_ids, torch::Tensor atlas_uvs, int B
) {
    return rvq_decode_impl<true>(codebooks, indices, surfel_offsets,
                                 rects, surfel_ids, atlas_uvs, B);
}


// --- BC7-baseline reference: bilinear sample of a dequantized FP16 atlas. ---
// Approximates what tex2D<float4> + atlas_scale/offset does in the production
// kernel (without the hardware texture-cache assist — so this is the *upper
// bound* on the baseline cost).

__global__ void baseline_bilinear_kernel(
    const __half* __restrict__ atlas,   // [H, W, 3] FP16
    const float* __restrict__ atlas_uvs,
    float* __restrict__ out_rgb,
    int n_frag, int H, int W
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_frag) return;
    float au = atlas_uvs[t*2 + 0] - 0.5f;
    float av = atlas_uvs[t*2 + 1] - 0.5f;
    int au0 = max(0, min((int)floorf(au), W - 1));
    int au1 = max(0, min(au0 + 1,        W - 1));
    int av0 = max(0, min((int)floorf(av), H - 1));
    int av1 = max(0, min(av0 + 1,        H - 1));
    float fu = au - floorf(au); float fv = av - floorf(av);
    const __half* p00 = atlas + (av0 * W + au0) * 3;
    const __half* p01 = atlas + (av0 * W + au1) * 3;
    const __half* p10 = atlas + (av1 * W + au0) * 3;
    const __half* p11 = atlas + (av1 * W + au1) * 3;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v00 = __half2float(p00[c]);
        float v01 = __half2float(p01[c]);
        float v10 = __half2float(p10[c]);
        float v11 = __half2float(p11[c]);
        float top = v00 * (1 - fu) + v01 * fu;
        float bot = v10 * (1 - fu) + v11 * fu;
        out_rgb[t*3 + c] = top * (1 - fv) + bot * fv;
    }
}

torch::Tensor baseline_bilinear_cuda(
    torch::Tensor atlas_fp16, torch::Tensor atlas_uvs
) {
    int n_frag = atlas_uvs.size(0);
    int H = atlas_fp16.size(0); int W = atlas_fp16.size(1);
    auto out = torch::empty({n_frag, 3}, atlas_uvs.options());
    int threads = 256;
    int blocks = (n_frag + threads - 1) / threads;
    baseline_bilinear_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(atlas_fp16.data_ptr<at::Half>()),
        atlas_uvs.data_ptr<float>(),
        out.data_ptr<float>(),
        n_frag, H, W
    );
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rvq_decode_nearest",  &rvq_decode_nearest_cuda);
    m.def("rvq_decode_bilinear", &rvq_decode_bilinear_cuda);
    m.def("baseline_bilinear",   &baseline_bilinear_cuda);
}
