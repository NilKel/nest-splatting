// Standalone CUDA MLP test for debugging gradient computation
// Build: nvcc -o mlp_test mlp_test.cu -lcudart
// Run: ./mlp_test

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define SUB_TILE_SIZE 128

// MLP dimensions
#define IN_DIM 40
#define HIDDEN_DIM 32
#define OUT_DIM 3

// Forward pass: input[40] -> h1[32] -> h2[32] -> output[3]
__device__ void mlp_forward(
    const float* input,      // [40]
    float* h1_pre,           // [32] pre-activation
    float* h1_post,          // [32] post-ReLU
    float* h2_pre,           // [32] pre-activation
    float* h2_post,          // [32] post-ReLU
    float* output,           // [3]
    const float* W1,         // [32, 40]
    const float* b1,         // [32]
    const float* W2,         // [32, 32]
    const float* b2,         // [32]
    const float* W3,         // [3, 32]
    const float* b3          // [3]
) {
    // Layer 1: h1 = ReLU(W1 @ input + b1)
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float acc = b1[h];
        for (int i = 0; i < IN_DIM; i++) {
            acc += W1[h * IN_DIM + i] * input[i];
        }
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 2: h2 = ReLU(W2 @ h1 + b2)
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float acc = b2[h];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            acc += W2[h * HIDDEN_DIM + i] * h1_post[i];
        }
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 3: output = sigmoid(W3 @ h2 + b3)
    for (int o = 0; o < OUT_DIM; o++) {
        float acc = b3[o];
        for (int h = 0; h < HIDDEN_DIM; h++) {
            acc += W3[o * HIDDEN_DIM + h] * h2_post[h];
        }
        output[o] = 1.0f / (1.0f + expf(-acc));
    }
}

// Compute dL_dz for all layers (for collaborative GEMM)
__device__ void compute_dL_dz_all(
    const float* h1_pre,
    const float* h2_pre,
    const float* h2_post,
    const float* output,
    const float* dL_dout,    // [3] scaled by weight
    float* dL_dz3,           // [3] output
    float* dL_dz2,           // [32] output
    float* dL_dz1,           // [32] output
    const float* W2,         // [32, 32]
    const float* W3          // [3, 32]
) {
    // Gradient through sigmoid: dL_dz3 = dL_dout * sig * (1-sig)
    for (int o = 0; o < OUT_DIM; o++) {
        float sig = output[o];
        dL_dz3[o] = dL_dout[o] * sig * (1.0f - sig);
    }

    // Layer 3 backward -> dL_dh2
    float dL_dh2[HIDDEN_DIM] = {0};
    for (int o = 0; o < OUT_DIM; o++) {
        for (int h = 0; h < HIDDEN_DIM; h++) {
            dL_dh2[h] += dL_dz3[o] * W3[o * HIDDEN_DIM + h];
        }
    }

    // ReLU backward -> dL_dz2
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz2[h] = (h2_pre[h] > 0) ? dL_dh2[h] : 0;
    }

    // Layer 2 backward -> dL_dh1
    float dL_dh1[HIDDEN_DIM] = {0};
    for (int h = 0; h < HIDDEN_DIM; h++) {
        for (int i = 0; i < HIDDEN_DIM; i++) {
            dL_dh1[i] += dL_dz2[h] * W2[h * HIDDEN_DIM + i];
        }
    }

    // ReLU backward -> dL_dz1
    for (int h = 0; h < HIDDEN_DIM; h++) {
        dL_dz1[h] = (h1_pre[h] > 0) ? dL_dh1[h] : 0;
    }
}

// Collaborative GEMM for MLP backward - exactly as in mode_3d_direct_fused.cu
__device__ void collaborative_mlp_backward_all(
    const float* my_input,
    const float* my_h1,
    const float* my_h2,
    const float* my_dL_dz1,
    const float* my_dL_dz2,
    const float* my_dL_dz3,
    float* tile_dL_dW1,
    float* tile_dL_db1,
    float* tile_dL_dW2,
    float* tile_dL_db2,
    float* tile_dL_dW3,
    float* tile_dL_db3,
    float* smem_buffer
) {
    const int tid = threadIdx.x;

    // Layer 3: dL_dW3[3,32] = dL_dz3^T @ h2
    {
        float* smem_dL_dz3 = smem_buffer;              // [128][3]
        float* smem_h2 = smem_buffer + SUB_TILE_SIZE * 3;  // [128][32]

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
                smem_dL_dz3[local_idx * 3 + 0] = my_dL_dz3[0];
                smem_dL_dz3[local_idx * 3 + 1] = my_dL_dz3[1];
                smem_dL_dz3[local_idx * 3 + 2] = my_dL_dz3[2];
                for (int i = 0; i < 32; i++) {
                    smem_h2[local_idx * 32 + i] = my_h2[i];
                }
            }
            __syncthreads();

            // W3 gradient: threads 0-95
            if (tid < 96) {
                int o = tid / 32;
                int h = tid % 32;
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz3[p * 3 + o] * smem_h2[p * 32 + h];
                }
                tile_dL_dW3[tid] += sum;
            }

            // b3 gradient: threads 0-2
            if (tid < 3) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz3[p * 3 + tid];
                }
                tile_dL_db3[tid] += sum;
            }
            __syncthreads();
        }
    }

    // Layer 2: dL_dW2[32,32] = dL_dz2^T @ h1
    {
        float* smem_dL_dz2 = smem_buffer;                  // [128][32]
        float* smem_h1 = smem_buffer + SUB_TILE_SIZE * 32; // [128][32]

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
                for (int i = 0; i < 32; i++) {
                    smem_dL_dz2[local_idx * 32 + i] = my_dL_dz2[i];
                    smem_h1[local_idx * 32 + i] = my_h1[i];
                }
            }
            __syncthreads();

            // W2 gradient: 4 passes for 1024 elements
            for (int pass = 0; pass < 4; pass++) {
                int out_idx = pass * 256 + tid;
                if (out_idx < 1024) {
                    int h = out_idx / 32;
                    int i = out_idx % 32;
                    float sum = 0;
                    for (int p = 0; p < SUB_TILE_SIZE; p++) {
                        sum += smem_dL_dz2[p * 32 + h] * smem_h1[p * 32 + i];
                    }
                    tile_dL_dW2[out_idx] += sum;
                }
            }

            // b2 gradient: threads 0-31
            if (tid < 32) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz2[p * 32 + tid];
                }
                tile_dL_db2[tid] += sum;
            }
            __syncthreads();
        }
    }

    // Layer 1: dL_dW1[32,40] = dL_dz1^T @ input
    {
        float* smem_dL_dz1 = smem_buffer;                  // [128][32]
        float* smem_input = smem_buffer + SUB_TILE_SIZE * 32; // [128][40]

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
                for (int i = 0; i < 32; i++) {
                    smem_dL_dz1[local_idx * 32 + i] = my_dL_dz1[i];
                }
                for (int i = 0; i < 40; i++) {
                    smem_input[local_idx * 40 + i] = my_input[i];
                }
            }
            __syncthreads();

            // W1 gradient: 5 passes for 1280 elements
            for (int pass = 0; pass < 5; pass++) {
                int out_idx = pass * 256 + tid;
                if (out_idx < 1280) {
                    int h = out_idx / 40;
                    int i = out_idx % 40;
                    float sum = 0;
                    for (int p = 0; p < SUB_TILE_SIZE; p++) {
                        sum += smem_dL_dz1[p * 32 + h] * smem_input[p * 40 + i];
                    }
                    tile_dL_dW1[out_idx] += sum;
                }
            }

            // b1 gradient: threads 0-31
            if (tid < 32) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz1[p * 32 + tid];
                }
                tile_dL_db1[tid] += sum;
            }
            __syncthreads();
        }
    }
}

// Test kernel: each thread processes one "pixel"
__global__ void test_mlp_backward_kernel(
    const float* inputs,     // [N, 40]
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float* dL_dW1, float* dL_db1,
    float* dL_dW2, float* dL_db2,
    float* dL_dW3, float* dL_db3,
    int N
) {
    // Shared memory for tile accumulators
    __shared__ float tile_dL_dW1[1280];
    __shared__ float tile_dL_db1[32];
    __shared__ float tile_dL_dW2[1024];
    __shared__ float tile_dL_db2[32];
    __shared__ float tile_dL_dW3[96];
    __shared__ float tile_dL_db3[3];

    // Initialize to zero
    for (int i = threadIdx.x; i < 1280; i += blockDim.x) tile_dL_dW1[i] = 0;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) tile_dL_dW2[i] = 0;
    for (int i = threadIdx.x; i < 96; i += blockDim.x) tile_dL_dW3[i] = 0;
    if (threadIdx.x < 32) { tile_dL_db1[threadIdx.x] = 0; tile_dL_db2[threadIdx.x] = 0; }
    if (threadIdx.x < 3) tile_dL_db3[threadIdx.x] = 0;
    __syncthreads();

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread arrays
    float my_input[IN_DIM] = {0};
    float my_h1_pre[HIDDEN_DIM] = {0}, my_h1_post[HIDDEN_DIM] = {0};
    float my_h2_pre[HIDDEN_DIM] = {0}, my_h2_post[HIDDEN_DIM] = {0};
    float my_output[OUT_DIM] = {0};
    float my_dL_dout[OUT_DIM] = {0};
    float my_dL_dz1[HIDDEN_DIM] = {0}, my_dL_dz2[HIDDEN_DIM] = {0}, my_dL_dz3[OUT_DIM] = {0};

    bool participates = (global_idx < N);

    if (participates) {
        // Load input
        for (int i = 0; i < IN_DIM; i++) {
            my_input[i] = inputs[global_idx * IN_DIM + i];
        }

        // Forward pass
        mlp_forward(my_input, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, my_output,
                    W1, b1, W2, b2, W3, b3);

        // Assume dL_dout = [1, 1, 1] (gradient from loss = output.sum())
        for (int c = 0; c < OUT_DIM; c++) {
            my_dL_dout[c] = 1.0f;
        }

        // Compute dL_dz for all layers
        compute_dL_dz_all(my_h1_pre, my_h2_pre, my_h2_post, my_output, my_dL_dout,
                          my_dL_dz3, my_dL_dz2, my_dL_dz1, W2, W3);
    }

    __syncthreads();

    // Collaborative GEMM
    extern __shared__ float dynamic_smem[];
    collaborative_mlp_backward_all(
        my_input, my_h1_post, my_h2_post,
        my_dL_dz1, my_dL_dz2, my_dL_dz3,
        tile_dL_dW1, tile_dL_db1,
        tile_dL_dW2, tile_dL_db2,
        tile_dL_dW3, tile_dL_db3,
        dynamic_smem
    );

    __syncthreads();

    // Flush to global (atomics)
    for (int i = threadIdx.x; i < 1280; i += blockDim.x) {
        if (tile_dL_dW1[i] != 0) atomicAdd(&dL_dW1[i], tile_dL_dW1[i]);
    }
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        if (tile_dL_db1[i] != 0) atomicAdd(&dL_db1[i], tile_dL_db1[i]);
    }
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        if (tile_dL_dW2[i] != 0) atomicAdd(&dL_dW2[i], tile_dL_dW2[i]);
    }
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        if (tile_dL_db2[i] != 0) atomicAdd(&dL_db2[i], tile_dL_db2[i]);
    }
    for (int i = threadIdx.x; i < 96; i += blockDim.x) {
        if (tile_dL_dW3[i] != 0) atomicAdd(&dL_dW3[i], tile_dL_dW3[i]);
    }
    for (int i = threadIdx.x; i < 3; i += blockDim.x) {
        if (tile_dL_db3[i] != 0) atomicAdd(&dL_db3[i], tile_dL_db3[i]);
    }
}

void set_unit_weights(float* W1, float* b1, float* W2, float* b2, float* W3, float* b3) {
    // W1: 32x40, diagonal for first 32
    memset(W1, 0, 32 * 40 * sizeof(float));
    for (int i = 0; i < 32; i++) W1[i * 40 + i] = 1.0f;
    memset(b1, 0, 32 * sizeof(float));

    // W2: 32x32 identity
    memset(W2, 0, 32 * 32 * sizeof(float));
    for (int i = 0; i < 32; i++) W2[i * 32 + i] = 1.0f;
    memset(b2, 0, 32 * sizeof(float));

    // W3: 3x32, diagonal for first 3
    memset(W3, 0, 3 * 32 * sizeof(float));
    for (int i = 0; i < 3; i++) W3[i * 32 + i] = 1.0f;
    memset(b3, 0, 3 * sizeof(float));
}

int main() {
    printf("=== Standalone MLP Backward Test ===\n\n");

    // Allocate host memory
    float *h_W1, *h_b1, *h_W2, *h_b2, *h_W3, *h_b3;
    float *h_dL_dW1, *h_dL_db1, *h_dL_dW2, *h_dL_db2, *h_dL_dW3, *h_dL_db3;
    float *h_inputs;

    h_W1 = (float*)malloc(32 * 40 * sizeof(float));
    h_b1 = (float*)malloc(32 * sizeof(float));
    h_W2 = (float*)malloc(32 * 32 * sizeof(float));
    h_b2 = (float*)malloc(32 * sizeof(float));
    h_W3 = (float*)malloc(3 * 32 * sizeof(float));
    h_b3 = (float*)malloc(3 * sizeof(float));

    h_dL_dW1 = (float*)malloc(32 * 40 * sizeof(float));
    h_dL_db1 = (float*)malloc(32 * sizeof(float));
    h_dL_dW2 = (float*)malloc(32 * 32 * sizeof(float));
    h_dL_db2 = (float*)malloc(32 * sizeof(float));
    h_dL_dW3 = (float*)malloc(3 * 32 * sizeof(float));
    h_dL_db3 = (float*)malloc(3 * sizeof(float));

    // Test with 256 "pixels" (one block)
    int N = 256;
    h_inputs = (float*)malloc(N * 40 * sizeof(float));

    // Set unit weights
    set_unit_weights(h_W1, h_b1, h_W2, h_b2, h_W3, h_b3);

    // Set inputs to all ones
    for (int i = 0; i < N * 40; i++) h_inputs[i] = 1.0f;

    // Initialize gradients to zero
    memset(h_dL_dW1, 0, 32 * 40 * sizeof(float));
    memset(h_dL_db1, 0, 32 * sizeof(float));
    memset(h_dL_dW2, 0, 32 * 32 * sizeof(float));
    memset(h_dL_db2, 0, 32 * sizeof(float));
    memset(h_dL_dW3, 0, 3 * 32 * sizeof(float));
    memset(h_dL_db3, 0, 3 * sizeof(float));

    // Allocate device memory
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    float *d_dL_dW1, *d_dL_db1, *d_dL_dW2, *d_dL_db2, *d_dL_dW3, *d_dL_db3;
    float *d_inputs;

    cudaMalloc(&d_W1, 32 * 40 * sizeof(float));
    cudaMalloc(&d_b1, 32 * sizeof(float));
    cudaMalloc(&d_W2, 32 * 32 * sizeof(float));
    cudaMalloc(&d_b2, 32 * sizeof(float));
    cudaMalloc(&d_W3, 3 * 32 * sizeof(float));
    cudaMalloc(&d_b3, 3 * sizeof(float));

    cudaMalloc(&d_dL_dW1, 32 * 40 * sizeof(float));
    cudaMalloc(&d_dL_db1, 32 * sizeof(float));
    cudaMalloc(&d_dL_dW2, 32 * 32 * sizeof(float));
    cudaMalloc(&d_dL_db2, 32 * sizeof(float));
    cudaMalloc(&d_dL_dW3, 3 * 32 * sizeof(float));
    cudaMalloc(&d_dL_db3, 3 * sizeof(float));

    cudaMalloc(&d_inputs, N * 40 * sizeof(float));

    // Copy to device
    cudaMemcpy(d_W1, h_W1, 32 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, 3 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, N * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dW1, h_dL_dW1, 32 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_db1, h_dL_db1, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dW2, h_dL_dW2, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_db2, h_dL_db2, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dW3, h_dL_dW3, 3 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_db3, h_dL_db3, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t smem_size = SUB_TILE_SIZE * 40 * sizeof(float) + SUB_TILE_SIZE * 32 * sizeof(float);
    test_mlp_backward_kernel<<<1, 256, smem_size>>>(
        d_inputs, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
        d_dL_dW1, d_dL_db1, d_dL_dW2, d_dL_db2, d_dL_dW3, d_dL_db3, N
    );
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_dL_db1, d_dL_db1, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dL_db2, d_dL_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dL_db3, d_dL_db3, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("With UNIT weights and input=1, dL_dout=[1,1,1]:\n\n");

    printf("dL_db3 nonzeros: ");
    int nz3 = 0;
    for (int i = 0; i < 3; i++) if (fabsf(h_dL_db3[i]) > 1e-10) nz3++;
    printf("%d/3\n", nz3);

    printf("dL_db2 nonzeros: ");
    int nz2 = 0;
    for (int i = 0; i < 32; i++) if (fabsf(h_dL_db2[i]) > 1e-10) nz2++;
    printf("%d/32 at indices: ", nz2);
    for (int i = 0; i < 32; i++) if (fabsf(h_dL_db2[i]) > 1e-10) printf("%d ", i);
    printf("\n");

    printf("dL_db1 nonzeros: ");
    int nz1 = 0;
    for (int i = 0; i < 32; i++) if (fabsf(h_dL_db1[i]) > 1e-10) nz1++;
    printf("%d/32 at indices: ", nz1);
    for (int i = 0; i < 32; i++) if (fabsf(h_dL_db1[i]) > 1e-10) printf("%d ", i);
    printf("\n");

    printf("\n=== Expected with unit weights ===\n");
    printf("dL_db3: 3/3 at [0,1,2]\n");
    printf("dL_db2: 3/32 at [0,1,2]\n");
    printf("dL_db1: 3/32 at [0,1,2]\n");

    // Verify
    bool pass = (nz3 == 3) && (nz2 == 3) && (nz1 == 3);
    printf("\n=== Result: %s ===\n", pass ? "PASS" : "FAIL");

    // Cleanup
    free(h_W1); free(h_b1); free(h_W2); free(h_b2); free(h_W3); free(h_b3);
    free(h_dL_dW1); free(h_dL_db1); free(h_dL_dW2); free(h_dL_db2); free(h_dL_dW3); free(h_dL_db3);
    free(h_inputs);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_dL_dW1); cudaFree(d_dL_db1); cudaFree(d_dL_dW2); cudaFree(d_dL_db2); cudaFree(d_dL_dW3); cudaFree(d_dL_db3);
    cudaFree(d_inputs);

    return pass ? 0 : 1;
}
