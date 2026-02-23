/*
 * Mode 5: 3D_direct_fused - In-kernel MLP with WMMA Tensor Core GEMM
 *
 * This mode runs hash lookup + MLP inside the rasterizer kernel, following the
 * per-intersection paradigm: sum(w_i * MLP(f_i))
 *
 * Key optimizations:
 * - WMMA Tensor Core GEMM for backward weight gradients (replaces scalar reduce)
 *   Each mma_sync computes 16×16×16 = 4096 FP16 FMAs in one instruction
 * - "Zeroes Matrix" strategy: non-participating pixels contribute 0
 * - FP16 weights in shared memory, FP32 accumulators for weight gradients
 * - Staged interleaved backward: layer-by-layer backward interleaved with GEMM
 * - Bias-free MLP: L1 uses input padding (input[40]=1.0, W1[32×48]),
 *   L2/L3 have no bias. Guarantees W@0=0 for zeroes matrix correctness.
 *
 * WMMA-aligned MLP dimensions (all multiples of 16):
 *   Input:  48D (41D real + 7D padding zeros)
 *   Hidden: 32D (unchanged)
 *   Output: 16D (3D RGB + 13D padding zeros)
 *   W1: [32, 48], W2: [32, 32], W3: [16, 32]
 */

#ifndef MODE_3D_DIRECT_FUSED_CU_INCLUDED
#define MODE_3D_DIRECT_FUSED_CU_INCLUDED

// When included from backward.cu, all declarations already exist
#ifndef BACKWARD_CU_INCLUDES_MODE
#include "modes.h"
#include "../auxiliary.h"
#include "../config.h"
#include "../hashgrid.h"
#include "../mma_utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Forward declaration of encode_view_direction from forward.cu
__device__ void encode_view_direction(const float3& dir, float* enc);
__device__ void encode_view_direction_bw(const float3& dir, float* enc);
#endif // BACKWARD_CU_INCLUDES_MODE

// ============================================================================
// MLP weight pointers struct (bias-free, WMMA-padded dimensions)
// L1: W1[32×48] (col 40 = implicit bias via input[40]=1.0, cols 41-47 = zero)
// L2: W2[32×32] (no bias, already WMMA-aligned)
// L3: W3[16×32] (rows 0-2 = RGB output, rows 3-15 = zero padding)
// ============================================================================
struct MlpWeights {
    const __half* W1;      // [32 * 48] FP16 (WMMA-padded from 32*41)
    const __half* W2;      // [32 * 32] FP16
    const __half* W3_rgb;  // [16 * 32] FP16 (WMMA-padded from 3*32)
};

// ============================================================================
// MLP Forward Pass (bias-free, WMMA-padded weights, scalar per-pixel)
// Input must have input[40] = 1.0f set by caller for implicit L1 bias
// Input positions 41-47 must be 0.0f (WMMA padding)
// ============================================================================
__device__ __forceinline__ void mlp_forward_inline(
    const float* input,    // [TC_INPUT_DIM=48] (input[40] = 1.0, [41..47] = 0)
    float* output,         // [ORIG_OUTPUT_DIM=3]
    float* h1,             // [TC_HIDDEN_DIM=32] hidden1 post-ReLU (for backward)
    float* h2,             // [TC_HIDDEN_DIM=32] hidden2 post-ReLU (for backward)
    bool apply_sigmoid,
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Layer 1: input[48] -> h1[32] with ReLU
    // Positions 41-47 of input are 0, so those weight columns don't contribute.
    // We iterate over all 48 to keep loop bounds WMMA-aligned.
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < TC_INPUT_DIM; i++) {
            acc += input[i] * __half2float(mlp.W1[h * TC_INPUT_DIM + i]);
        }
        h1[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 2: h1[32] -> h2[32] with ReLU
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < TC_HIDDEN_DIM; i++) {
            acc += h1[i] * __half2float(mlp.W2[h * TC_HIDDEN_DIM + i]);
        }
        h2[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 3: h2[32] -> output[3] with optional sigmoid
    // Only compute first 3 outputs (rows 3-15 of W3 are zero padding)
    #pragma unroll
    for (int o = 0; o < ORIG_OUTPUT_DIM; o++) {
        float acc = 0;
        #pragma unroll
        for (int h = 0; h < TC_HIDDEN_DIM; h++) {
            acc += h2[h] * __half2float(mlp.W3_rgb[o * TC_HIDDEN_DIM + h]);
        }
        if (apply_sigmoid) {
            output[o] = 1.0f / (1.0f + expf(-acc));
        } else {
            output[o] = acc;
        }
    }
}

// ============================================================================
// WMMA Tensor Core GEMM for MLP weight gradients
//
// Replaces scalar collaborative GEMM with hardware-accelerated mma_sync.
// Each warp computes one 16×16 output tile of dL_dW, iterating over 16 chunks
// of 16 pixels (K=256 total).
//
// Memory layout:
//   dL_dz stored as [256, M] row-major in smem → matrix_a col_major, stride=M
//   activation stored as [256, N] row-major in smem → matrix_b row_major, stride=N
//   tile_dL_dW[M, N] row-major in shared memory → accumulator load/store
//
// "Zeroes Matrix" strategy: non-participating pixels store zeros to smem,
// so their WMMA contributions are zero (W@0=0 for bias-free MLP).
// ============================================================================

// Layer 3: dL_dW3[16,32] = dL_dz3^T[16,256] @ h2[256,32]
// Output tiles: 1×2 = 2 (warps 0-1 active)
static __device__ void wmma_gemm_layer3(
    const float* my_dL_dz3,    // [TC_OUTPUT_DIM=16] per-pixel (FP32 registers, positions 3-15 = 0)
    const float* my_h2,        // [TC_HIDDEN_DIM=32] per-pixel (FP32 registers)
    float* tile_dL_dW3,        // [W3_SIZE=512] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 16]
    __half* smem_h2 = smem_buffer + TC_BATCH * TC_OUTPUT_DIM;   // [256, 32]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_OUTPUT_DIM; i++) {
        smem_dz[tid * TC_OUTPUT_DIM + i] = __float2half(my_dL_dz3[i]);
    }
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_h2[tid * TC_HIDDEN_DIM + i] = __float2half(my_h2[i]);
    }
    __syncthreads();

    // WMMA compute: 2 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L3_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L3_N_TILES;
        int n_tile = tile_idx % BW_L3_N_TILES;

        // Load existing accumulator (to accumulate across Gaussians)
        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW3 + m_tile * WMMA_M * W3_COLS + n_tile * WMMA_N,
            W3_COLS, wmma::mem_row_major);

        // K-loop: 16 chunks of 16 pixels
        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_OUTPUT_DIM + m_tile * WMMA_M,
                TC_OUTPUT_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_h2 + k * WMMA_K * TC_HIDDEN_DIM + n_tile * WMMA_N,
                TC_HIDDEN_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW3 + m_tile * WMMA_M * W3_COLS + n_tile * WMMA_N,
            frag_c, W3_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Layer 2: dL_dW2[32,32] = dL_dz2^T[32,256] @ h1[256,32]
// Output tiles: 2×2 = 4 (warps 0-3 active)
static __device__ void wmma_gemm_layer2(
    const float* my_dL_dz2,    // [TC_HIDDEN_DIM=32] per-pixel
    const float* my_h1,        // [TC_HIDDEN_DIM=32] per-pixel
    float* tile_dL_dW2,        // [W2_SIZE=1024] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 32]
    __half* smem_h1 = smem_buffer + TC_BATCH * TC_HIDDEN_DIM;   // [256, 32]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_dz[tid * TC_HIDDEN_DIM + i] = __float2half(my_dL_dz2[i]);
        smem_h1[tid * TC_HIDDEN_DIM + i] = __float2half(my_h1[i]);
    }
    __syncthreads();

    // WMMA compute: 4 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L2_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L2_N_TILES;
        int n_tile = tile_idx % BW_L2_N_TILES;

        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW2 + m_tile * WMMA_M * W2_COLS + n_tile * WMMA_N,
            W2_COLS, wmma::mem_row_major);

        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_HIDDEN_DIM + m_tile * WMMA_M,
                TC_HIDDEN_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_h1 + k * WMMA_K * TC_HIDDEN_DIM + n_tile * WMMA_N,
                TC_HIDDEN_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW2 + m_tile * WMMA_M * W2_COLS + n_tile * WMMA_N,
            frag_c, W2_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Layer 1: dL_dW1[32,48] = dL_dz1^T[32,256] @ input[256,48]
// Output tiles: 2×3 = 6 (warps 0-5 active)
// Input positions 41-47 are 0 (WMMA padding), so cols 41-47 of dL_dW1 = 0
static __device__ void wmma_gemm_layer1(
    const float* my_dL_dz1,    // [TC_HIDDEN_DIM=32] per-pixel
    const float* my_input,     // [TC_INPUT_DIM=48] per-pixel (pos 41-47 = 0)
    float* tile_dL_dW1,        // [W1_SIZE=1536] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz    = smem_buffer;                            // [256, 32]
    __half* smem_input = smem_buffer + TC_BATCH * TC_HIDDEN_DIM; // [256, 48]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_dz[tid * TC_HIDDEN_DIM + i] = __float2half(my_dL_dz1[i]);
    }
    #pragma unroll
    for (int i = 0; i < TC_INPUT_DIM; i++) {
        smem_input[tid * TC_INPUT_DIM + i] = __float2half(my_input[i]);
    }
    __syncthreads();

    // WMMA compute: 6 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L1_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L1_N_TILES;
        int n_tile = tile_idx % BW_L1_N_TILES;

        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW1 + m_tile * WMMA_M * W1_COLS + n_tile * WMMA_N,
            W1_COLS, wmma::mem_row_major);

        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_HIDDEN_DIM + m_tile * WMMA_M,
                TC_HIDDEN_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_input + k * WMMA_K * TC_INPUT_DIM + n_tile * WMMA_N,
                TC_INPUT_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW1 + m_tile * WMMA_M * W1_COLS + n_tile * WMMA_N,
            frag_c, W1_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Required dynamic shared memory size for WMMA backward
// Layer 1 is the largest: 256*(32+48)*sizeof(half) = 40,960 bytes
constexpr int COLLABORATIVE_SMEM_SIZE = TC_COLLABORATIVE_SMEM_SIZE;

// Required dynamic shared memory size for WMMA forward
constexpr int FW_SMEM_HALF_BYTES = TC_FORWARD_SMEM_HALF;
constexpr int FW_SMEM_FLOAT_BYTES = TC_FORWARD_SMEM_FLOAT;
constexpr int FW_SMEM_TOTAL_BYTES = TC_FORWARD_SMEM_SIZE;

// ============================================================================
// WMMA forward layer: Out[256, N] = In[256, K] @ W^T[K, N]
// W stored as [N, K] row-major = W^T[K, N] col-major (no transpose needed)
// ============================================================================
template <int K, int N>
__device__ void wmma_forward_layer(
    const __half* smem_input,   // [TC_BATCH, K] row-major half
    const __half* smem_W,       // [N, K] row-major half
    float* smem_output          // [TC_BATCH, N] row-major float
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;

    constexpr int M_TILES = TC_BATCH / WMMA_M;
    constexpr int N_TILES = N / WMMA_N;
    constexpr int K_CHUNKS = K / WMMA_K;
    constexpr int TOTAL_TILES = M_TILES * N_TILES;

    for (int tile_idx = warp_id; tile_idx < TOTAL_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        frag_acc frag_c;
        wmma::fill_fragment(frag_c, 0.0f);

        for (int kk = 0; kk < K_CHUNKS; kk++) {
            fw_frag_a frag_a;
            fw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_input + m_tile * WMMA_M * K + kk * WMMA_K, K);
            wmma::load_matrix_sync(frag_b,
                smem_W + n_tile * WMMA_N * K + kk * WMMA_K, K);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            smem_output + m_tile * WMMA_M * N + n_tile * WMMA_N,
            frag_c, N, wmma::mem_row_major);
    }
}

// ============================================================================
// WMMA forward: full 3-layer MLP for all 256 pixels simultaneously
//
// Input:  smem_fw_half[256 * TC_INPUT_DIM] — each thread's MLP input as half
//         (non-participating threads store zeros; bias-free MLP ⇒ W@0=0)
// Output: smem_fw_float[256 * TC_OUTPUT_DIM] — raw pre-sigmoid MLP output
//         Caller reads first ORIG_OUTPUT_DIM=3 channels and applies sigmoid
//
// Shared memory reuse:
//   L1: reads smem_fw_half[256×48], writes smem_fw_float[256×32]
//        → ReLU+half → smem_fw_half[256×32] (reuse, smaller footprint)
//   L2: reads smem_fw_half[256×32], writes smem_fw_float[256×32]
//        → ReLU+half → smem_fw_half[256×32]
//   L3: reads smem_fw_half[256×32], writes smem_fw_float[256×16]
// ============================================================================
static __device__ void wmma_forward_all(
    __half* smem_fw_half,       // [TC_BATCH * TC_INPUT_DIM] input, reused for intermediates
    float*  smem_fw_float,      // [TC_BATCH * TC_HIDDEN_DIM] accumulator output
    const __half* smem_W1,      // [W1_SIZE] = [32, 48] row-major half
    const __half* smem_W2,      // [W2_SIZE] = [32, 32] row-major half
    const __half* smem_W3       // [W3_SIZE] = [16, 32] row-major half
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Layer 1: H1[256,32] = Input[256,48] × W1^T[48,32]
    wmma_forward_layer<TC_INPUT_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W1, smem_fw_float);
    __syncthreads();

    // ReLU + float→half: reuse smem_fw_half as [256×32]
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 2: H2[256,32] = H1_relu[256,32] × W2^T[32,32]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W2, smem_fw_float);
    __syncthreads();

    // ReLU + float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 3: Out[256,16] = H2_relu[256,32] × W3^T[32,16]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_OUTPUT_DIM>(smem_fw_half, smem_W3, smem_fw_float);
    __syncthreads();

    // Output: smem_fw_float[tid * TC_OUTPUT_DIM + ch] (raw pre-sigmoid)
}

namespace MODES {

// ============================================================================
// Flush tile-local MLP gradients to global memory (WMMA-padded dimensions)
// Called once per tile after processing all Gaussians
// ============================================================================
static __device__ void flush_tile_mlp_grads(
    const float* tile_dL_dW1,
    const float* tile_dL_dW2,
    const float* tile_dL_dW3,
    float* global_dL_dW1,
    float* global_dL_dW2,
    float* global_dL_dW3,
    const int thread_id
) {
    // W1: 32×48 = 1536 floats
    for (int idx = thread_id; idx < W1_SIZE; idx += 256) {
        if (tile_dL_dW1[idx] != 0.0f)
            atomicAdd(&global_dL_dW1[idx], tile_dL_dW1[idx]);
    }
    // W2: 32×32 = 1024 floats
    for (int idx = thread_id; idx < W2_SIZE; idx += 256) {
        if (tile_dL_dW2[idx] != 0.0f)
            atomicAdd(&global_dL_dW2[idx], tile_dL_dW2[idx]);
    }
    // W3: 16×32 = 512 floats
    for (int idx = thread_id; idx < W3_SIZE; idx += 256) {
        if (tile_dL_dW3[idx] != 0.0f)
            atomicAdd(&global_dL_dW3[idx], tile_dL_dW3[idx]);
    }
}

} // namespace MODES

#endif // MODE_3D_DIRECT_FUSED_CU_INCLUDED
