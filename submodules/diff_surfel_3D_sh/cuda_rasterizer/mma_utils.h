/*
 * mma_utils.h - WMMA Tensor Core utilities for MLP weight gradient GEMM
 *
 * SH mode: MLP outputs 48D SH coefficients (degree 3, 16 per channel × 3 RGB)
 * Input: [gauss(20) | hash(4) | bias(1)] = 25D, padded to 32D
 * Hidden: 32D, Output: 48D (already WMMA-aligned)
 */

#ifndef MMA_UTILS_H_INCLUDED
#define MMA_UTILS_H_INCLUDED

#include <mma.h>
using namespace nvcuda;

// ============================================================================
// WMMA tile dimensions (fixed by hardware)
// ============================================================================
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// Padded MLP dimensions for WMMA alignment (SH mode)
// ============================================================================
constexpr int TC_INPUT_DIM  = 32;   // 25 -> 32 (pad 7 zeros after bias at position 24)
constexpr int TC_HIDDEN_DIM = 32;   // already aligned
constexpr int TC_OUTPUT_DIM = 48;   // 48 SH coefficients (16 per channel × 3 RGB)
constexpr int TC_BATCH      = 256;  // pixels per tile (16x16 block)

// Original (unpadded) dimensions for masking/trimming
constexpr int ORIG_INPUT_DIM  = 25; // [gauss(20) | hash(4) | bias(1)]
constexpr int ORIG_OUTPUT_DIM = 48; // all 48 SH coefficients used

// ============================================================================
// Weight matrix dimensions: [rows, cols]
// ============================================================================
constexpr int W1_ROWS = TC_HIDDEN_DIM;  // 32
constexpr int W1_COLS = TC_INPUT_DIM;   // 32
constexpr int W2_ROWS = TC_HIDDEN_DIM;  // 32
constexpr int W2_COLS = TC_HIDDEN_DIM;  // 32
constexpr int W3_ROWS = TC_OUTPUT_DIM;  // 48
constexpr int W3_COLS = TC_HIDDEN_DIM;  // 32

// Total weight elements per layer
constexpr int W1_SIZE = W1_ROWS * W1_COLS;  // 32*32 = 1024
constexpr int W2_SIZE = W2_ROWS * W2_COLS;  // 32*32 = 1024
constexpr int W3_SIZE = W3_ROWS * W3_COLS;  // 48*32 = 1536

// ============================================================================
// WMMA tile counts for backward weight GEMM
// dL_dW[M,N] = dL_dz^T[M,K] @ activation[K,N], where K = TC_BATCH = 256 pixels
// ============================================================================

// Layer 1: dL_dW1[32, 32] = dL_dz1^T[32, 256] @ input[256, 32]
constexpr int BW_L1_M_TILES = W1_ROWS / WMMA_M;   // 2
constexpr int BW_L1_N_TILES = W1_COLS / WMMA_N;    // 2
constexpr int BW_L1_TILES   = BW_L1_M_TILES * BW_L1_N_TILES;  // 4

// Layer 2: dL_dW2[32, 32] = dL_dz2^T[32, 256] @ h1[256, 32]
constexpr int BW_L2_M_TILES = W2_ROWS / WMMA_M;   // 2
constexpr int BW_L2_N_TILES = W2_COLS / WMMA_N;    // 2
constexpr int BW_L2_TILES   = BW_L2_M_TILES * BW_L2_N_TILES;  // 4

// Layer 3: dL_dW3[48, 32] = dL_dz3^T[48, 256] @ h2[256, 32]
constexpr int BW_L3_M_TILES = W3_ROWS / WMMA_M;   // 3
constexpr int BW_L3_N_TILES = W3_COLS / WMMA_N;    // 2
constexpr int BW_L3_TILES   = BW_L3_M_TILES * BW_L3_N_TILES;  // 6

// K dimension chunks (256 pixels / 16 = 16 chunks)
constexpr int BW_K_CHUNKS = TC_BATCH / WMMA_K;     // 16

// ============================================================================
// WMMA tile counts for forward batched MLP (currently disabled, scalar faster)
// H[M,N] = Input[M,K] @ W^T[K,N], where M = TC_BATCH = 256 pixels
// ============================================================================

// Layer 1: H1[256, 32] = Input[256, 32] @ W1^T[32, 32]
constexpr int FW_L1_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L1_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 2
constexpr int FW_L1_K_CHUNKS = TC_INPUT_DIM / WMMA_K;  // 2
constexpr int FW_L1_TILES = FW_L1_M_TILES * FW_L1_N_TILES;  // 32

// Layer 2: H2[256, 32] = H1[256, 32] @ W2^T[32, 32]
constexpr int FW_L2_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L2_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 2
constexpr int FW_L2_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 2
constexpr int FW_L2_TILES = FW_L2_M_TILES * FW_L2_N_TILES;  // 32

// Layer 3: Out[256, 48] = H2[256, 32] @ W3^T[32, 48]
constexpr int FW_L3_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L3_N_TILES = TC_OUTPUT_DIM / WMMA_N;  // 3
constexpr int FW_L3_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 2
constexpr int FW_L3_TILES = FW_L3_M_TILES * FW_L3_N_TILES;  // 48

// Total warps per block: 256 threads / 32 threads-per-warp = 8 warps
constexpr int WARPS_PER_BLOCK = TC_BATCH / 32;  // 8

// ============================================================================
// Fragment type aliases
// ============================================================================

// Backward GEMM: C[M,N] = A^T[M,K] @ B[K,N]
//   dL_dz stored as [K,M] row-major -> matrix_a col_major gives transposed view
//   activation stored as [K,N] row-major -> matrix_b row_major
using bw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
using bw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using frag_acc  = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Forward GEMM: H[M,N] = Input[M,K] @ W^T[K,N]
//   Input stored as [M,K] row-major -> matrix_a row_major
//   W stored as [N,K] row-major = W^T[K,N] col-major -> matrix_b col_major
using fw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using fw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;

// ============================================================================
// Shared memory size constants
// ============================================================================

// Backward GEMM: largest layer is L3 = 256*(48+32)*sizeof(half) = 40,960 bytes
constexpr int TC_COLLABORATIVE_SMEM_SIZE = TC_BATCH * (TC_OUTPUT_DIM + TC_HIDDEN_DIM) * sizeof(half);

// Forward WMMA (disabled): half input buffer + float accumulator output
constexpr int TC_FORWARD_SMEM_HALF = TC_BATCH * TC_INPUT_DIM * sizeof(half);
constexpr int TC_FORWARD_SMEM_FLOAT = TC_BATCH * TC_HIDDEN_DIM * sizeof(float);
constexpr int TC_FORWARD_SMEM_SIZE = TC_FORWARD_SMEM_HALF + TC_FORWARD_SMEM_FLOAT;

#endif // MMA_UTILS_H_INCLUDED
