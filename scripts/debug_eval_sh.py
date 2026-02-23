#!/usr/bin/env python3
"""
Debug: evaluate 48D SH residual in Python and render with pre-evaluated 3D textures.
Compares Python SH evaluation (reference) vs CUDA kernel SH evaluation.
"""
import os, sys, json, math, pickle, torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]


def eval_sh_basis(viewdir):
    """Compute 16 SH basis values for each direction. viewdir: [N, 3] normalized."""
    x, y, z = viewdir[:, 0], viewdir[:, 1], viewdir[:, 2]
    xx, yy, zz = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z

    basis = torch.zeros(viewdir.shape[0], 16, device=viewdir.device)
    basis[:, 0]  = SH_C0
    basis[:, 1]  = -SH_C1 * y
    basis[:, 2]  = SH_C1 * z
    basis[:, 3]  = -SH_C1 * x
    basis[:, 4]  = SH_C2[0] * xy
    basis[:, 5]  = SH_C2[1] * yz
    basis[:, 6]  = SH_C2[2] * (2*zz - xx - yy)
    basis[:, 7]  = SH_C2[3] * xz
    basis[:, 8]  = SH_C2[4] * (xx - yy)
    basis[:, 9]  = SH_C3[0] * y * (3*xx - yy)
    basis[:, 10] = SH_C3[1] * xy * z
    basis[:, 11] = SH_C3[2] * y * (4*zz - xx - yy)
    basis[:, 12] = SH_C3[3] * z * (2*zz - 3*xx - 3*yy)
    basis[:, 13] = SH_C3[4] * x * (4*zz - xx - yy)
    basis[:, 14] = SH_C3[5] * z * (xx - yy)
    basis[:, 15] = SH_C3[6] * x * (xx - 3*yy)
    return basis


def preeval_48d_to_3d(residual_48d, means3D, campos):
    """
    Pre-evaluate 48D SH residual at per-Gaussian viewdir → 3D per-texel residual.

    residual_48d: [N, 8, 8, 48] float
    means3D: [N, 3]
    campos: [3]

    Returns: [N, 8, 8, 3] float (evaluated RGB residual per texel)
    """
    N = residual_48d.shape[0]
    # Compute per-Gaussian viewdir
    viewdir = means3D - campos.unsqueeze(0)  # [N, 3]
    viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-8)

    # Compute SH basis
    basis = eval_sh_basis(viewdir)  # [N, 16]

    # Evaluate per-texel per-channel
    evaluated = torch.zeros(N, 8, 8, 3, device=residual_48d.device)
    for ch in range(3):
        coeffs = residual_48d[:, :, :, ch*16:(ch+1)*16]  # [N, 8, 8, 16]
        evaluated[:, :, :, ch] = (coeffs * basis[:, None, None, :]).sum(dim=-1)

    return evaluated


def render_baked(viewpoint_camera, gaussians, bg_color, residual_textures=None,
                 beta=0.0, kernel_type=0):
    """Render using diff_surfel_bake_render with 3D DC textures."""
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False, beta=beta,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    result = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
        opacities=gaussians.get_opacity,
        shs=gaussians.get_features,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shapes=gaussians.get_shape if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0 else None,
        kernel_type=kernel_type,
        residual_textures=residual_textures,
    )
    return result[0]


def main():
    model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
    baked_dir = os.path.join(model_path, "baked")

    # Load training config
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    cfg_model = Config(os.path.join(model_path, "config.yaml"))

    # Load baked Gaussians
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    dataset = model_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(baked_dir, "baked.ply"))
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    kernel_name = getattr(args, 'kernel', 'gaussian')
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = kernel_name
    kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
    kernel_type = kernel_map.get(kernel_name, 0)

    N = len(gaussians.get_xyz)
    beta = cfg_model.surfel.tg_beta
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Load 48D residual
    residual_48d = torch.load(os.path.join(baked_dir, "residual_textures.pt")).cuda().float()
    print(f"48D residual: {list(residual_48d.shape)}")

    # Also load the CUDA 48D path texture
    cuda_48d_tex = residual_48d.half().view(N, -1).contiguous()

    # Load test cameras
    import glob
    ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
    iteration = max([int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files])
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cameras = scene.getTestCameras()

    # Test on first 10 cameras
    psnr_sh, psnr_cuda48, psnr_py48 = [], [], []

    with torch.no_grad():
        for i, cam in enumerate(test_cameras[:10]):
            gt = cam.original_image[:3].cuda()

            # 1. SH only (no residual)
            render_sh = render_baked(cam, gaussians, bg_color, None, beta, kernel_type)

            # 2. CUDA 48D path (what the kernel does)
            render_cuda48 = render_baked(cam, gaussians, bg_color, cuda_48d_tex, beta, kernel_type)

            # 3. Python-evaluated 3D residual (reference)
            campos = cam.camera_center  # [3]
            means3D = gaussians.get_xyz[:N]  # [N, 3] (match baked count)
            eval_3d = preeval_48d_to_3d(residual_48d, means3D, campos)  # [N, 8, 8, 3]
            py_3d_tex = eval_3d.half().view(N, -1).contiguous()  # [N, 192]
            render_py48 = render_baked(cam, gaussians, bg_color, py_3d_tex, beta, kernel_type)

            p_sh = psnr(render_sh, gt).mean().item()
            p_cuda48 = psnr(render_cuda48, gt).mean().item()
            p_py48 = psnr(render_py48, gt).mean().item()

            psnr_sh.append(p_sh)
            psnr_cuda48.append(p_cuda48)
            psnr_py48.append(p_py48)

            if i < 3:
                # Per-pixel diff between CUDA 48D and Python 48D
                diff = (render_cuda48 - render_py48).abs()
                print(f"  View {i}: SH={p_sh:.2f}  CUDA48={p_cuda48:.2f}  PY48={p_py48:.2f}  "
                      f"CUDA-PY diff: mean={diff.mean():.6f} max={diff.max():.4f}")
            else:
                print(f"  View {i}: SH={p_sh:.2f}  CUDA48={p_cuda48:.2f}  PY48={p_py48:.2f}")

    print(f"\nAvg PSNR (10 views):")
    print(f"  SH only:    {np.mean(psnr_sh):.2f} dB")
    print(f"  CUDA 48D:   {np.mean(psnr_cuda48):.2f} dB")
    print(f"  Python 48D: {np.mean(psnr_py48):.2f} dB")


if __name__ == "__main__":
    main()
