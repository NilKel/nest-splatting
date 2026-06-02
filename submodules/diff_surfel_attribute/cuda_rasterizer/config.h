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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_FEATURES 16 // 27 dim
#define BLOCK_X 16
#define BLOCK_Y 16

// FP16 per-Gaussian SH baseline color storage. See rgb_type.h for the
// rgb_t typedef. (Defined in a separate header to avoid leaking NUM_CHANNELS
// above into cub's includes via rasterizer_impl.h.)

#endif