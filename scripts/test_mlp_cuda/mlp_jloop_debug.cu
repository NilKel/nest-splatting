/**
 * Debug version of j-loop test
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define SUB_TILE_SIZE 128

__device__ void mlp_forward_debug(
    const float* input,
    float* h1_pre, float* h1_post,
    float* h2_pre, float* h2_post,
    float* output,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    int tid
) {
    for (int h = 0; h < 32; h++) {
        float acc = b1[h];
        for (int i = 0; i < 40; i++) acc += W1[h * 40 + i] * input[i];
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }
    for (int h = 0; h < 32; h++) {
        float acc = b2[h];
        for (int i = 0; i < 32; i++) acc += W2[h * 32 + i] * h1_post[i];
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }
    for (int o = 0; o < 3; o++) {
        float acc = b3[o];
        for (int h = 0; h < 32; h++) acc += W3[o * 32 + h] * h2_post[h];
        output[o] = 1.0f / (1.0f + expf(-acc));
    }

    if (tid == 0) {
        printf("Thread 0 forward:\n");
        printf("  W1[0,0]=%.1f W1[1,1]=%.1f W1[2,2]=%.1f\n", W1[0], W1[41], W1[82]);
        printf("  input[0:3]=%.1f %.1f %.1f\n", input[0], input[1], input[2]);
        printf("  h1_post[0:3]=%.3f %.3f %.3f\n", h1_post[0], h1_post[1], h1_post[2]);
        printf("  h2_post[0:3]=%.3f %.3f %.3f\n", h2_post[0], h2_post[1], h2_post[2]);
        printf("  output[0:3]=%.3f %.3f %.3f\n", output[0], output[1], output[2]);
    }
}

__device__ void compute_dL_dz_all_debug(
    const float* h1_pre, const float* h2_pre, const float* h2_post,
    const float* output, const float* dL_dout,
    float* dL_dz3, float* dL_dz2, float* dL_dz1,
    const float* W2, const float* W3,
    int tid
) {
    for (int o = 0; o < 3; o++) {
        float sig = output[o];
        dL_dz3[o] = dL_dout[o] * sig * (1.0f - sig);
    }

    float dL_dh2[32] = {0};
    for (int o = 0; o < 3; o++) {
        for (int h = 0; h < 32; h++) {
            dL_dh2[h] += dL_dz3[o] * W3[o * 32 + h];
        }
    }

    for (int h = 0; h < 32; h++) {
        dL_dz2[h] = (h2_pre[h] > 0) ? dL_dh2[h] : 0;
    }

    float dL_dh1[32] = {0};
    for (int h = 0; h < 32; h++) {
        for (int i = 0; i < 32; i++) {
            dL_dh1[i] += dL_dz2[h] * W2[h * 32 + i];
        }
    }

    for (int h = 0; h < 32; h++) {
        dL_dz1[h] = (h1_pre[h] > 0) ? dL_dh1[h] : 0;
    }

    if (tid == 0) {
        printf("Thread 0 backward:\n");
        printf("  dL_dz3[0:3]=%.6f %.6f %.6f\n", dL_dz3[0], dL_dz3[1], dL_dz3[2]);
        printf("  dL_dz2[0:5]=%.6f %.6f %.6f %.6f %.6f\n", dL_dz2[0], dL_dz2[1], dL_dz2[2], dL_dz2[3], dL_dz2[4]);
    }
}

__device__ void collaborative_mlp_backward_debug(
    const float* my_input, const float* my_h1, const float* my_h2,
    const float* my_dL_dz1, const float* my_dL_dz2, const float* my_dL_dz3,
    float* tile_dL_db2,
    float* smem_buffer
) {
    const int tid = threadIdx.x;

    // Layer 2 only for simplicity
    float* smem_dL_dz2 = smem_buffer;
    float* smem_h1 = smem_buffer + SUB_TILE_SIZE * 32;

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

        // Debug: print what thread 0 sees in smem for batch 0
        if (tid == 0 && batch == 0) {
            printf("Batch 0 smem_dL_dz2[0][0:5]: %.6f %.6f %.6f %.6f %.6f\n",
                   smem_dL_dz2[0], smem_dL_dz2[1], smem_dL_dz2[2], smem_dL_dz2[3], smem_dL_dz2[4]);
        }

        // Bias computation
        if (tid < 32) {
            float sum = 0;
            for (int p = 0; p < SUB_TILE_SIZE; p++) {
                sum += smem_dL_dz2[p * 32 + tid];
            }
            tile_dL_db2[tid] += sum;

            if (tid < 5) {
                printf("After batch %d: tile_dL_db2[%d] = %.4f\n", batch, tid, tile_dL_db2[tid]);
            }
        }
        __syncthreads();
    }
}

__global__ void test_debug(
    const float* inputs,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float* dL_db2
) {
    __shared__ float tile_dL_db2[32];

    int tid = threadIdx.x;
    if (tid < 32) tile_dL_db2[tid] = 0;
    __syncthreads();

    float my_input[40], my_h1_pre[32], my_h1_post[32], my_h2_pre[32], my_h2_post[32];
    float my_output[3], my_dL_dout[3] = {1, 1, 1};
    float my_dL_dz1[32], my_dL_dz2[32], my_dL_dz3[3];

    for (int i = 0; i < 40; i++) my_input[i] = inputs[tid * 40 + i];
    mlp_forward_debug(my_input, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, my_output, W1, b1, W2, b2, W3, b3, tid);
    compute_dL_dz_all_debug(my_h1_pre, my_h2_pre, my_h2_post, my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, W2, W3, tid);

    __syncthreads();

    extern __shared__ float smem[];
    collaborative_mlp_backward_debug(my_input, my_h1_post, my_h2_post, my_dL_dz1, my_dL_dz2, my_dL_dz3, tile_dL_db2, smem);

    __syncthreads();

    if (tid < 32) dL_db2[tid] = tile_dL_db2[tid];
}

void set_unit_weights(float* W1, float* W2, float* W3) {
    memset(W1, 0, 32 * 40 * sizeof(float));
    for (int i = 0; i < 32; i++) W1[i * 40 + i] = 1.0f;

    memset(W2, 0, 32 * 32 * sizeof(float));
    for (int i = 0; i < 32; i++) W2[i * 32 + i] = 1.0f;

    memset(W3, 0, 3 * 32 * sizeof(float));
    for (int i = 0; i < 3; i++) W3[i * 32 + i] = 1.0f;
}

int main() {
    printf("=== Debug Test ===\n\n");

    float *h_W1 = (float*)calloc(32 * 40, sizeof(float));
    float *h_W2 = (float*)calloc(32 * 32, sizeof(float));
    float *h_W3 = (float*)calloc(3 * 32, sizeof(float));
    float *h_b1 = (float*)calloc(32, sizeof(float));
    float *h_b2 = (float*)calloc(32, sizeof(float));
    float *h_b3 = (float*)calloc(3, sizeof(float));
    float *h_inputs = (float*)malloc(256 * 40 * sizeof(float));
    float *h_db2 = (float*)malloc(32 * sizeof(float));

    set_unit_weights(h_W1, h_W2, h_W3);
    for (int i = 0; i < 256 * 40; i++) h_inputs[i] = 1.0f;

    printf("Host weights:\n");
    printf("  W1[0,0]=%.1f W1[1,1]=%.1f W1[2,2]=%.1f\n", h_W1[0], h_W1[41], h_W1[82]);

    float *d_W1, *d_W2, *d_W3, *d_b1, *d_b2, *d_b3, *d_inputs, *d_db2;
    cudaMalloc(&d_W1, 32 * 40 * sizeof(float));
    cudaMalloc(&d_W2, 32 * 32 * sizeof(float));
    cudaMalloc(&d_W3, 3 * 32 * sizeof(float));
    cudaMalloc(&d_b1, 32 * sizeof(float));
    cudaMalloc(&d_b2, 32 * sizeof(float));
    cudaMalloc(&d_b3, 3 * sizeof(float));
    cudaMalloc(&d_inputs, 256 * 40 * sizeof(float));
    cudaMalloc(&d_db2, 32 * sizeof(float));

    cudaMemcpy(d_W1, h_W1, 32 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, 3 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, 256 * 40 * sizeof(float), cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(test_debug, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    test_debug<<<1, 256, 65536>>>(d_inputs, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_db2);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_db2, d_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nFinal dL_db2 nonzeros: ");
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) printf("%d(%.2f) ", i, h_db2[i]);
    }
    printf("\n");

    return 0;
}
