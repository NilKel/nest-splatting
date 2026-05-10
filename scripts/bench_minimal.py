"""Minimal-deps FPS / PSNR / SSIM / LPIPS bench for a single bundle.

Designed to run on a fresh box (e.g. a 4090) with NO nest-splatting source
checkout — only torch, plyfile, Pillow, pytorch_msssim, lpips, and the
diff_surfel_bake_render CUDA submodule.

Usage:
    python bench_minimal.py --bundle <bundle_dir> [--num_warmup 10] [--num_benchmark 200]

Bundle layout (see build_bench_bundle.py):
    gaussian_state.pt   atlas_texture.bc7   atlas_rects.pt
    bake_meta.json      cameras.pt          images/*.png
    sb_params.pt        (optional, --feature beta only)
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr_t(a, b):
    mse = ((a - b) ** 2).mean()
    if mse <= 0:
        return torch.tensor(99.0, device=a.device)
    return -10.0 * torch.log10(mse)


def _ssim_factory(device):
    from pytorch_msssim import ssim as _ssim
    def _go(a, b):
        # both [3, H, W] in [0,1]
        return _ssim(a.unsqueeze(0), b.unsqueeze(0), data_range=1.0, size_average=True)
    return _go


def _lpips_factory(device):
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    @torch.no_grad()
    def _go(a, b):
        # lpips expects [-1, 1]
        return net(a.unsqueeze(0) * 2 - 1, b.unsqueeze(0) * 2 - 1).item()
    return _go


# ---------------------------------------------------------------------------
# Atlas + rasterizer setup (one call per scene)
# ---------------------------------------------------------------------------
def install_atlas(bundle_dir, bake_meta):
    from diff_surfel_bake_render import (
        set_atlas_use_uint8, set_use_atlas_tex_object,
        clear_atlas_cache, set_atlas_bc7, clear_atlas_bc7,
        set_activation_bias, set_compact_mult, set_residual_mode,
    )
    set_use_atlas_tex_object(True)
    set_atlas_use_uint8(True)
    clear_atlas_cache()
    clear_atlas_bc7()

    bc7_file = bake_meta.get("atlas_bc7_file")
    if bc7_file is None:
        raise RuntimeError("Bundle missing BC7 atlas; only BC7 atlases supported.")
    with open(os.path.join(bundle_dir, bc7_file), "rb") as f:
        bc7 = f.read()
    bc7_t = torch.frombuffer(bytearray(bc7), dtype=torch.uint8).cuda()
    W = int(bake_meta["atlas_bc7_padded_w"])
    H = int(bake_meta["atlas_bc7_padded_h"])
    offset = float(bake_meta.get("atlas_offset", 0.0))
    scale  = float(bake_meta.get("atlas_scale",  1.0))
    set_atlas_bc7(bc7_t, W, H, offset, scale)

    set_activation_bias(float(bake_meta.get("sh_bias", 0.5)),
                        float(bake_meta.get("res_bias", 0.0)))
    set_compact_mult(float(bake_meta.get("compact_mult", 1.0)))
    set_residual_mode(int(bake_meta.get("residual_mode", 0)))
    return bc7_t, W, H


def render_one(cam, state, atlas_tex_dummy, atlas_rects, atlas_width,
               sb_params, sb_number, sv_state, bg, sh_degree, beta, aabb_mode, sort_mode):
    from diff_surfel_bake_render import get_rasterizer
    tanfovx = math.tan(cam["FoVx"] * 0.5)
    tanfovy = math.tan(cam["FoVy"] * 0.5)
    rasterizer = get_rasterizer(
        image_height=int(cam["image_height"]),
        image_width=int(cam["image_width"]),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg,
        viewmatrix=cam["world_view_transform"],
        projmatrix=cam["full_proj_transform"],
        campos=cam["camera_center"],
        sh_degree=sh_degree, beta=beta,
        aabb_mode=aabb_mode, sort_mode=sort_mode,
    )
    if sv_state is not None:
        v_sites, v_tau, v_colors, v_K = (sv_state["sites"], sv_state["tau"],
                                          sv_state["colors"], sv_state["K"])
    else:
        v_sites = v_tau = v_colors = None
        v_K = 0
    color, _ = rasterizer(
        means3D=state["means3D"],
        opacities=state["opacities"],
        shs=state["shs"],
        scales=state["scales"],
        rotations=state["rotations"],
        shapes=state["shapes"],
        kernel_type=state["kernel_type"],
        atlas_texture=atlas_tex_dummy,
        atlas_rects=atlas_rects,
        atlas_width=atlas_width,
        sb_params=sb_params, sb_number=sb_number,
        voronoi_sites=v_sites, voronoi_tau=v_tau,
        voronoi_colors=v_colors, voronoi_K=v_K,
    )
    return color


def run_bench(bundle_dir, num_warmup=10, num_benchmark=200, beta=0.0,
              aabb_mode=5, sort_mode=0):
    device = torch.device("cuda")
    bundle = Path(bundle_dir)

    # --- Load Gaussian state, push to GPU ---
    state_cpu = torch.load(bundle / "gaussian_state.pt", map_location="cpu", weights_only=False)
    state = {
        k: (v.contiguous().to(device) if torch.is_tensor(v) else v)
        for k, v in state_cpu.items() if k not in ("sv_state",)
    }
    sv_state = None
    if state_cpu.get("sv_state") is not None:
        sv_state = {
            "sites":  state_cpu["sv_state"]["sites"].contiguous().to(device),
            "tau":    state_cpu["sv_state"]["tau"].contiguous().to(device),
            "colors": state_cpu["sv_state"]["colors"].contiguous().to(device),
            "K":      state_cpu["sv_state"]["K"],
        }
    sh_degree = state.get("active_sh_degree", 3)

    # --- Bake meta + atlas install ---
    with open(bundle / "bake_meta.json") as f:
        bake_meta = json.load(f)
    bc7_tensor, atlas_w, atlas_h = install_atlas(str(bundle), bake_meta)
    # Placeholder FP16 tensor for atlas_texture (CUDA expects at::Half pointer
    # but the BC7 fast path doesn't read it).
    atlas_tex_dummy = torch.zeros(1, 1, 3, dtype=torch.float16, device=device)
    atlas_rects = torch.load(bundle / "atlas_rects.pt").contiguous().to(device)

    sb_params = None
    sb_number = int(bake_meta.get("sb_number", 0))
    if sb_number > 0 and (bundle / "sb_params.pt").exists():
        sb_t = torch.load(bundle / "sb_params.pt").float().to(device)
        sb_params = sb_t.reshape(-1).contiguous()

    # --- Cameras ---
    cams = torch.load(bundle / "cameras.pt", weights_only=False)
    cams = [{**c,
             "world_view_transform": c["world_view_transform"].to(device),
             "full_proj_transform":  c["full_proj_transform"].to(device),
             "camera_center":        c["camera_center"].to(device)}
            for c in cams]

    bg = torch.zeros(3, dtype=torch.float32, device=device)

    # --- Quality pass (PSNR/SSIM/LPIPS) ---
    ssim_fn  = _ssim_factory(device)
    lpips_fn = _lpips_factory(device)
    psnrs, ssims, lps = [], [], []
    with torch.no_grad():
        for c in cams:
            img = Image.open(bundle / "images" / f"{c['image_name']}.png").convert("RGB")
            gt = (torch.from_numpy(np.asarray(img)).float() / 255.0).permute(2, 0, 1).to(device)
            rendered = render_one(c, state, atlas_tex_dummy, atlas_rects, atlas_w,
                                  sb_params, sb_number, sv_state, bg,
                                  sh_degree, beta, aabb_mode, sort_mode).clamp(0, 1)
            psnrs.append(psnr_t(rendered, gt).item())
            ssims.append(ssim_fn(rendered, gt).item())
            lps.append(lpips_fn(rendered, gt))

    # --- FPS pass (CUDA events, --num_benchmark frames cycling cameras) ---
    with torch.no_grad():
        for i in range(num_warmup):
            _ = render_one(cams[i % len(cams)], state, atlas_tex_dummy, atlas_rects,
                           atlas_w, sb_params, sb_number, sv_state, bg,
                           sh_degree, beta, aabb_mode, sort_mode)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmark)]
        for i in range(num_benchmark):
            starts[i].record()
            _ = render_one(cams[i % len(cams)], state, atlas_tex_dummy, atlas_rects,
                           atlas_w, sb_params, sb_number, sv_state, bg,
                           sh_degree, beta, aabb_mode, sort_mode)
            ends[i].record()
        torch.cuda.synchronize()
        ms_each = [starts[i].elapsed_time(ends[i]) for i in range(num_benchmark)]

    mean_ms = float(np.mean(ms_each))
    fps = 1000.0 / mean_ms

    return {
        "n_gaussians":  int(state["means3D"].shape[0]),
        "n_cameras":    len(cams),
        "resolution":   f"{cams[0]['image_width']}x{cams[0]['image_height']}",
        "psnr":  float(np.mean(psnrs)),
        "ssim":  float(np.mean(ssims)),
        "lpips": float(np.mean(lps)),
        "fps":   float(fps),
        "ms_per_frame": mean_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--num_warmup", type=int, default=10)
    ap.add_argument("--num_benchmark", type=int, default=200)
    ap.add_argument("--aabb_mode", type=int, default=5)
    ap.add_argument("--sort_mode", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--out", type=str, default=None,
                    help="Write metrics JSON here (default: <bundle>/bench_results.json)")
    args = ap.parse_args()

    res = run_bench(args.bundle,
                    num_warmup=args.num_warmup,
                    num_benchmark=args.num_benchmark,
                    aabb_mode=args.aabb_mode,
                    sort_mode=args.sort_mode,
                    beta=args.beta)

    out = args.out or os.path.join(args.bundle, "bench_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
