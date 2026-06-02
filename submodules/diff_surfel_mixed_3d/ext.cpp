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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("compute_relocation", &compute_relocation_tensor);
  m.def("compute_opacity_gradient_3D", &ComputeOpacityGradient3DCUDA);
  m.def("compute_geometry_gradient_3D", &ComputeGeometryGradient3DCUDA);
  m.def("backward_from_weight_grad", &BackwardFromWeightGradCUDA);
  m.def("transmat_to_scale_rot_grad", &TransMatToScaleRotGradCUDA);
  m.def("get_transmat_from_geombuffer", &GetTransMatFromGeomBufferCUDA);
  m.def("set_mlp_weights", &SetMlpWeightsCUDA);
  m.def("set_contrib_thresh", &SetContribThreshCUDA);
  m.def("set_count_thresh", &SetCountThreshCUDA);
  m.def("set_overdraw_lambda", &SetOverdrawLambdaCUDA);
  m.def("set_weight_reg_lambda", &SetWeightRegLambdaCUDA);
  m.def("set_activation_bias", &SetActivationBiasCUDA);
  m.def("set_residual_mode", &SetResidualModeCUDA,
        "0 = 3D_SH_res outer ReLU (default). 1 = 3D_SH_add separate ReLUs.");
  m.def("set_ste_relu", &SetSteReluCUDA,
        "`--ste`: straight-through estimator on the per-Gauss outer ReLU "
        "(mode 0). 1 = backward identity even at clamped activations. "
        "0 = exact gradient (default).");
  m.def("set_anti_alias", &SetAntiAliasCUDA);
  m.def("set_compact_mult", &SetCompactMultCUDA);
  m.def("set_aa_kernel_size", &SetAaKernelSizeCUDA);
  m.def("set_skip_mlp_grad", &SetSkipMlpGradCUDA);
  m.def("set_depth_sort", &SetDepthSortCUDA);
  m.def("reset_backward_profile", &ResetBackwardProfileCUDA);
  m.def("read_backward_profile", &ReadBackwardProfileCUDA);
}