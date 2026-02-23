/*
 * Baking submodule: MLP + hashgrid → SH coefficients at UV grid points
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_mlp_weights", &SetMlpWeightsCUDA);
  m.def("bake_gaussians", &BakeGaussiansCUDA);
  // Keep rasterizer for potential debug/comparison
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("compute_relocation", &compute_relocation_tensor);
  m.def("compute_opacity_gradient_3D", &ComputeOpacityGradient3DCUDA);
  m.def("compute_geometry_gradient_3D", &ComputeGeometryGradient3DCUDA);
  m.def("backward_from_weight_grad", &BackwardFromWeightGradCUDA);
  m.def("transmat_to_scale_rot_grad", &TransMatToScaleRotGradCUDA);
  m.def("get_transmat_from_geombuffer", &GetTransMatFromGeomBufferCUDA);
  m.def("reset_backward_profile", &ResetBackwardProfileCUDA);
  m.def("read_backward_profile", &ReadBackwardProfileCUDA);
}
