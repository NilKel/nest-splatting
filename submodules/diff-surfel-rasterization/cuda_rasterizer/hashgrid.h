
#ifndef _CUDA_HASHGRID
#define _CUDA_HASHGRID

#include "stdio.h"

template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
    
    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}

template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const bool align_corners, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        stride *= align_corners ? resolution: (resolution + 1);
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (index % hashmap_size) * C + ch;
}

template <typename T>
__device__ inline T smoothstep(T val) {
	return val*val*(3.0f - 2.0f * val);
}

template <typename T>
__device__ inline T smoothstep_derivative(T val) {
	return 6*val*(1.0f - val);
}
// hash_features, level, l_scale, Base, align_corners, interp);

template <bool BW, uint32_t C, uint32_t LD>
__device__ void query_feature(float* feat, float3 xyz, float vmin, float vmax, 
int* offsets, const uint32_t appearance_level, const float* __restrict__ hash_features,
uint32_t L, float S, uint32_t H, bool align_corners, uint32_t interp, bool contract, bool debug,
float* __restrict__ grad_feat = nullptr,
float * __restrict__ dL_dfeatures = nullptr, float * __restrict__ dL_dxyz = nullptr)
{
    if(debug){
        printf("BW tag: %d \n", BW);
    }
    float inputs[3] = {0.0};
    bool flag_oob = false;
	const uint32_t D = 3;
    float grad_scale = 0.0;

    if(!contract){
        // warp the pos to [0, 1]
        float inv_vsize = 1.0 / (vmax - vmin);
        inputs[0] = (xyz.x-vmin)*inv_vsize;
        inputs[1] = (xyz.y-vmin)*inv_vsize;
        inputs[2] = (xyz.z-vmin)*inv_vsize;
        grad_scale = inv_vsize;
        flag_oob = (inputs[0] < 0 || inputs[0] > 1 || inputs[1] < 0 || inputs[1] > 1 || inputs[2] < 0 || inputs[2] > 1);
    }
    else{
        // warp the center region to [-1, 1]
        float vmid = (vmax + vmin) * 0.5;
        float inv_vsize_ = 1.0 / ((vmax - vmin) * 0.5);
        inputs[0] = (xyz.x-vmid)*inv_vsize_;
        inputs[1] = (xyz.y-vmid)*inv_vsize_;
        inputs[2] = (xyz.z-vmid)*inv_vsize_;
        // warp the outside region to [-2, 2], then warp to [0, 1]
        float norm = sqrtf(inputs[0]*inputs[0] + inputs[1]*inputs[1] + inputs[2]*inputs[2]);
        float inv_norm = 1.0f / norm;
        float scale_trans = (norm <= 1.0f) ? 1.0f : (2.0f - inv_norm) * inv_norm;
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) 
        {
            // warp to range [-2, 2]
            inputs[d] = scale_trans * inputs[d] ;
            // norm to [0, 1]
            inputs[d] = (inputs[d] + 2.0) * 0.25f;
        } 
        grad_scale = inv_vsize_ * scale_trans * 0.25f;
    }

    uint32_t max_level = min(appearance_level, L);
    
	#pragma unroll
	for (uint32_t ch = 0; ch < C; ch++) {
		feat[ch] = 0; 
	}

    // for backward, initialize dy_dx
    float dy_dx[BW ? (D * C) : 1] = {0};
    if constexpr (BW && C > LD) {
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) dL_dxyz[d] = 0;
    } 

    // out of range, no features, no grad update.
	if (flag_oob) return;

    const uint32_t size = (C > LD) ? C / LD * D : 1;
    float level_pos[size];
    float level_pos_deriv[size]; 
    uint32_t level_pos_grid[size];
    
    #pragma unroll
	for(uint32_t level = 0; level < max_level; level++){

		float pos[D];
        float pos_deriv[D];
    	uint32_t pos_grid[D];
        
        const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
        const float scale = exp2f(level * S) * H - 1.0f;
        const uint32_t resolution = (uint32_t)ceil(scale) + 1;
        const float* grid = hash_features + (uint32_t)offsets[level] * LD;

        float* dL_dgrid = nullptr;
        float* grad_level_feat = nullptr;
        if constexpr (BW && C > LD) {
            dL_dgrid = dL_dfeatures + (uint32_t)offsets[level] * LD;
            grad_level_feat = grad_feat + level * LD;
        }

		#pragma unroll
		for (uint32_t d = 0; d < D; d++) {
			pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
			pos_grid[d] = floorf(pos[d]);
			pos[d] -= (float)pos_grid[d];
            // smoothstep instead of linear
            if (interp == 1) {
                pos_deriv[d] = smoothstep_derivative(pos[d]);
                pos[d] = smoothstep(pos[d]);
            } else {
                pos_deriv[d] = 1.0f; // linear deriv is default to 1
            }
		}

		float results[LD] = {0};

		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << D); idx++) {
			float w = 1;
			uint32_t pos_grid_local[D];

			#pragma unroll
			for (uint32_t d = 0; d < D; d++) {
				if ((idx & (1 << d)) == 0) {
					w *= 1 - pos[d];
					pos_grid_local[d] = pos_grid[d];
				} else {
					w *= pos[d];
					pos_grid_local[d] = pos_grid[d] + 1;
				}
			}

            uint32_t index = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);

			// writing to register (fast)
			#pragma unroll
			for (uint32_t ch = 0; ch < LD; ch++) {
				results[ch] += w * grid[index + ch];
			}
            
            // update gradiendt
            if constexpr (BW && C > LD) {
                #pragma unroll
                for (uint32_t ch = 0; ch < LD; ch++) {
                    atomicAdd(&dL_dgrid[index + ch], w * grad_level_feat[ch]);
                }
            }

		}    

		// writing to L*LD features
		#pragma unroll
		for (uint32_t ch = 0; ch < LD; ch++) {
			feat[level * LD + ch] = results[ch]; 
		}

        if constexpr (BW && C > LD) {
            // save for dy_dx calculate
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                level_pos[level * D + d] = pos[d]; 
                level_pos_deriv[level * D + d] = pos_deriv[d]; 
                level_pos_grid[level * D + d] = pos_grid[d]; 
            }
        }

        if(debug && BW == 0){
            printf("fw level %d scale %.4f res %d \npos: ", level, scale, resolution);
            for(int d = 0; d < D; d++)printf("%d ", pos[d]);
            printf("\n");
        }

	}

    if constexpr (BW && C > LD) {
        // B L D C
        // D * F (F = L * LD)
        #pragma unroll
    	for(uint32_t level = 0; level < max_level; level++){
            
            float pos[D];
            float pos_deriv[D];
            uint32_t pos_grid[D];
            
            for (uint32_t d = 0; d < D; d++) {
                pos[d] = level_pos[level * D + d]; 
                pos_deriv[d] = level_pos_deriv[level * D + d]; 
                pos_grid[d] = level_pos_grid[level * D + d]; 
            }

            const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
            const float scale = exp2f(level * S) * H - 1.0f;
            const uint32_t resolution = (uint32_t)ceil(scale) + 1;
            const float* grid = hash_features + (uint32_t)offsets[level] * LD;

            if(debug){
                printf("bw level %d scale %.4f res %d \npos: ", level, scale, resolution);
                for(int d = 0; d < D; d++)printf("%d ", pos[d]);
                printf("\n");
            }

            #pragma unroll
            for (uint32_t gd = 0; gd < D; gd++) { 

                float results_grad[LD] = {0};

                #pragma unroll
                for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                    // float w = scale;
                    float w = scale * grad_scale;
                    uint32_t pos_grid_local[D];

                    #pragma unroll
                    for (uint32_t nd = 0; nd < D - 1; nd++) {
                        const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                        if ((idx & (1 << nd)) == 0) {
                            w *= 1 - pos[d];
                            pos_grid_local[d] = pos_grid[d];
                        } else {
                            w *= pos[d];
                            pos_grid_local[d] = pos_grid[d] + 1;
                        }
                    }

                    pos_grid_local[gd] = pos_grid[gd];
                    uint32_t index_left = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);
                    pos_grid_local[gd] = pos_grid[gd] + 1;
                    uint32_t index_right = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);

                    #pragma unroll
                    for (uint32_t ch = 0; ch < LD; ch++) {
                        results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_deriv[gd];
                    }
                }

                // update to dy_dx (D * F & D * L * LD), 
                #pragma unroll
                for (uint32_t ch = 0; ch < LD; ch++) {
                    dy_dx[gd * C + level * LD + ch] = results_grad[ch];
                }
            }
        }

        # pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            float result = 0;
            # pragma unroll
            for(int level = 0; level < max_level; level++) {
                float* grad_level_feat = grad_feat + level * LD;
                # pragma unroll
                for (int ch = 0; ch < LD; ch++) {
                    // grad_feat (C & L * LD), dy_dx (D * F & D * L * LD), 
                    result += grad_level_feat[ch] * dy_dx[d * C + level * LD + ch];
                }
            }

            dL_dxyz[d] = result;
        }
    }
}



template <bool BW, uint32_t C, uint32_t LD>
__device__ void query_compact_feature(float* feat, float3 xyz, float vmin, float vmax, 
int* offsets, const uint32_t appearance_level, const float* __restrict__ hash_features,
uint32_t L, float S, uint32_t H, bool align_corners, uint32_t interp, bool contract, bool debug,
float* __restrict__ grad_feat = nullptr,
float * __restrict__ dL_dfeatures = nullptr, float * __restrict__ dL_dxyz = nullptr)
{
    if(debug){
        printf("BW tag: %d \n", BW);
    }
    float inputs[3] = {0.0};
    bool flag_oob = false;
	const uint32_t D = 3;
    float grad_scale = 0.0;

    if(!contract){
        // warp the pos to [0, 1]
        float inv_vsize = 1.0 / (vmax - vmin);
        inputs[0] = (xyz.x-vmin)*inv_vsize;
        inputs[1] = (xyz.y-vmin)*inv_vsize;
        inputs[2] = (xyz.z-vmin)*inv_vsize;
        grad_scale = inv_vsize;
        flag_oob = (inputs[0] < 0 || inputs[0] > 1 || inputs[1] < 0 || inputs[1] > 1 || inputs[2] < 0 || inputs[2] > 1);
    }
    else{
        // warp the center region to [-1, 1]
        float vmid = (vmax + vmin) * 0.5;
        float inv_vsize_ = 1.0 / ((vmax - vmin) * 0.5);
        inputs[0] = (xyz.x-vmid)*inv_vsize_;
        inputs[1] = (xyz.y-vmid)*inv_vsize_;
        inputs[2] = (xyz.z-vmid)*inv_vsize_;
        // warp the outside region to [-2, 2], then warp to [0, 1]
        float norm = sqrtf(inputs[0]*inputs[0] + inputs[1]*inputs[1] + inputs[2]*inputs[2]);
        float inv_norm = 1.0f / norm;
        float scale_trans = (norm <= 1.0f) ? 1.0f : (2.0f - inv_norm) * inv_norm;
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) 
        {
            // warp to range [-2, 2]
            inputs[d] = scale_trans * inputs[d] ;
            // norm to [0, 1]
            inputs[d] = (inputs[d] + 2.0) * 0.25f;
        } 
        grad_scale = inv_vsize_ * scale_trans * 0.25f;
    }

    uint32_t max_level = min(appearance_level, L);
    
	#pragma unroll
	for (uint32_t ch = 0; ch < C; ch++) {
		feat[ch] = 0; 
	}

    // for backward, initialize dy_dx
    float dy_dx[BW ? (D * C) : 1] = {0};
    if constexpr (BW && C > LD) {
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) dL_dxyz[d] = 0;
    } 

    // out of range, no features, no grad update.
	if (flag_oob) return;

    const uint32_t size = (C > LD) ? C / LD * D : 1;
    float level_pos[size];
    float level_pos_deriv[size]; 
    uint32_t level_pos_grid[size];
    
    #pragma unroll
	for(uint32_t level = 0; level < max_level; level++){

		float pos[D];
        float pos_deriv[D];
    	uint32_t pos_grid[D];
        
        const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
        const float scale = exp2f(level * S) * H - 1.0f;
        const uint32_t resolution = (uint32_t)ceil(scale) + 1;
        const float* grid = hash_features + (uint32_t)offsets[level] * LD;

        float* dL_dgrid = nullptr;
        float* grad_level_feat = nullptr;
        if constexpr (BW && C > LD) {
            dL_dgrid = dL_dfeatures + (uint32_t)offsets[level] * LD;
            grad_level_feat = grad_feat + level * LD;
        }

		#pragma unroll
		for (uint32_t d = 0; d < D; d++) {
			pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
			pos_grid[d] = floorf(pos[d]);
			pos[d] -= (float)pos_grid[d];
            // smoothstep instead of linear
            if (interp == 1) {
                pos_deriv[d] = smoothstep_derivative(pos[d]);
                pos[d] = smoothstep(pos[d]);
            } else {
                pos_deriv[d] = 1.0f; // linear deriv is default to 1
            }
		}

		float results[LD] = {0};

		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << D); idx++) {
			float w = 1;
			uint32_t pos_grid_local[D];

			#pragma unroll
			for (uint32_t d = 0; d < D; d++) {
				if ((idx & (1 << d)) == 0) {
					w *= 1 - pos[d];
					pos_grid_local[d] = pos_grid[d];
				} else {
					w *= pos[d];
					pos_grid_local[d] = pos_grid[d] + 1;
				}
			}

            uint32_t index = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);

			// writing to register (fast)
			#pragma unroll
			for (uint32_t ch = 0; ch < LD; ch++) {
				results[ch] += w * grid[index + ch];
			}
            
            // update gradiendt
            if constexpr (BW && C > LD) {
                #pragma unroll
                for (uint32_t ch = 0; ch < LD; ch++) {
                    atomicAdd(&dL_dgrid[index + ch], w * grad_level_feat[ch]);
                }
            }

		}    

		// writing to L*LD features
		#pragma unroll
		for (uint32_t ch = 0; ch < LD; ch++) {
			feat[level * LD + ch] = results[ch]; 
		}

        if constexpr (BW && C > LD) {
            // save for dy_dx calculate
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                level_pos[level * D + d] = pos[d]; 
                level_pos_deriv[level * D + d] = pos_deriv[d]; 
                level_pos_grid[level * D + d] = pos_grid[d]; 
            }
        }

        if(debug && BW == 0){
            printf("fw level %d scale %.4f res %d \npos: ", level, scale, resolution);
            for(int d = 0; d < D; d++)printf("%d ", pos[d]);
            printf("\n");
        }

	}

    if constexpr (BW && C > LD) {
        // B L D C
        // D * F (F = L * LD)
        #pragma unroll
    	for(uint32_t level = 0; level < max_level; level++){
            
            float pos[D];
            float pos_deriv[D];
            uint32_t pos_grid[D];
            
            for (uint32_t d = 0; d < D; d++) {
                pos[d] = level_pos[level * D + d]; 
                pos_deriv[d] = level_pos_deriv[level * D + d]; 
                pos_grid[d] = level_pos_grid[level * D + d]; 
            }

            const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
            const float scale = exp2f(level * S) * H - 1.0f;
            const uint32_t resolution = (uint32_t)ceil(scale) + 1;
            const float* grid = hash_features + (uint32_t)offsets[level] * LD;

            if(debug){
                printf("bw level %d scale %.4f res %d \npos: ", level, scale, resolution);
                for(int d = 0; d < D; d++)printf("%d ", pos[d]);
                printf("\n");
            }

            #pragma unroll
            for (uint32_t gd = 0; gd < D; gd++) { 

                float results_grad[LD] = {0};

                #pragma unroll
                for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                    // float w = scale;
                    float w = scale * grad_scale;
                    uint32_t pos_grid_local[D];

                    #pragma unroll
                    for (uint32_t nd = 0; nd < D - 1; nd++) {
                        const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                        if ((idx & (1 << nd)) == 0) {
                            w *= 1 - pos[d];
                            pos_grid_local[d] = pos_grid[d];
                        } else {
                            w *= pos[d];
                            pos_grid_local[d] = pos_grid[d] + 1;
                        }
                    }

                    pos_grid_local[gd] = pos_grid[gd];
                    uint32_t index_left = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);
                    pos_grid_local[gd] = pos_grid[gd] + 1;
                    uint32_t index_right = get_grid_index<D, LD>(align_corners, 0, hashmap_size, resolution, pos_grid_local);

                    #pragma unroll
                    for (uint32_t ch = 0; ch < LD; ch++) {
                        results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_deriv[gd];
                    }
                }

                // update to dy_dx (D * F & D * L * LD), 
                #pragma unroll
                for (uint32_t ch = 0; ch < LD; ch++) {
                    dy_dx[gd * C + level * LD + ch] = results_grad[ch];
                }
            }
        }

        # pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            float result = 0;
            # pragma unroll
            for(int level = 0; level < max_level; level++) {
                float* grad_level_feat = grad_feat + level * LD;
                # pragma unroll
                for (int ch = 0; ch < LD; ch++) {
                    // grad_feat (C & L * LD), dy_dx (D * F & D * L * LD), 
                    result += grad_level_feat[ch] * dy_dx[d * C + level * LD + ch];
                }
            }

            dL_dxyz[d] = result;
        }
    }
}

#endif