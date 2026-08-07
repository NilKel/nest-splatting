from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

_src_path = os.path.dirname(os.path.abspath(__file__))
conda_include = os.path.join(sys.prefix, 'include')
glm_include = os.path.join(_src_path, 'third_party', 'glm')

# LEAN_FLAGS env var: comma-sep subset of {T2, CTG, CONIC}.  Each adds a
# preprocessor -D at build time.  Example: LEAN_FLAGS=T2,CTG.
_lean_flags = [f.strip().upper() for f in os.environ.get("LEAN_FLAGS", "").split(",") if f.strip()]
_lean_defs  = [f"-DLEAN_{f}" for f in _lean_flags if f in ("T2", "CTG", "CONIC", "DEBUG", "FP16_UVJ", "FP16_OPASHAPE", "LOCK_SHAPE")]
if _lean_defs:
    print(f"[lean build] Applying flags: {_lean_defs}", flush=True)

setup(
    name="diff_surfel_bake_render_lean",
    packages=['diff_surfel_bake_render_lean'],
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_bake_render_lean._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + conda_include,
                    "-I" + glm_include,
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ] + _lean_defs,
                "cxx": ["-O3", "-std=c++17"] + _lean_defs,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
