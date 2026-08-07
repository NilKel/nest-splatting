from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))

# Get conda environment include path
conda_include = os.path.join(sys.prefix, 'include')
# Get glm include path from the main rasterizer (sibling directory)
glm_include = os.path.join(this_dir, '..', 'diff-surfel-rasterization', 'third_party', 'glm')

# NO_EARLY_EXIT: disable T-saturation early exit in the render kernel so that
# `n_contrib` = number of surfels that survive the alpha < 1/255 cull for a
# pixel (matches what the WebGPU renderer pays, which has no per-pixel early
# termination). Only affects timing / n_contrib output, not final color much.
# T_CROSSING: override the T threshold in the median_depth update. Default is
# 0.5 (T=0.5 crossing = "half-opacity" depth). Set to e.g. 0.3 to record depth
# at 70% cumulative opacity — a bit deeper than the default and useful for
# testing where the proxy-mesh Z-cull frontier sits along the accumulation.
_t_crossing = os.environ.get("T_CROSSING")
_no_early_exit = os.environ.get("NO_EARLY_EXIT", "0") == "1"
# LAST_DEPTH_MODE: repurpose median_depth to output the depth of the DEEPEST
# surfel that survived the alpha cull (last contributor before T-saturation),
# not the T=0.5 crossing depth. Used to build proxy occlusion meshes for
# the WebGPU renderer. Downstream (mesh_utils.py, gaussian_renderer)
# doesn't need edits — the same allmap[5:6] slot now carries the tail.
_last_depth = os.environ.get("LAST_DEPTH_MODE", "0") == "1"
_extra_defs = []
if _no_early_exit:
    _extra_defs.append("-DDISABLE_EARLY_EXIT")
    print("[build] NO_EARLY_EXIT=1 → -DDISABLE_EARLY_EXIT", flush=True)
if _last_depth:
    _extra_defs.append("-DLAST_DEPTH_MODE")
    print("[build] LAST_DEPTH_MODE=1 → -DLAST_DEPTH_MODE (median_depth = deepest)", flush=True)
if _t_crossing is not None:
    val = float(_t_crossing)
    _extra_defs.append(f"-DT_CROSSING={val}f")
    print(f"[build] T_CROSSING={val} → -DT_CROSSING={val}f (median_depth updated while T > {val})", flush=True)

setup(
    name="diff_surfel_3D_sh_res_trunc",
    packages=["diff_surfel_3D_sh_res_trunc"],
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_3D_sh_res_trunc._C",
            sources=[
                "ext.cpp",
                "rasterize_points.cu",
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/utils.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + conda_include,
                    "-I" + glm_include,
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ] + _extra_defs,
                "cxx": ["-O3", "-std=c++17"] + _extra_defs,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
