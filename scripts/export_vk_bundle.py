#!/usr/bin/env python
"""Export a baked (atlas) checkpoint + its test cameras into a flat binary bundle
for the standalone Vulkan hardware rasterizer in `vk_raster/`.

Everything the CUDA CONIC bake-renderer (`diff_surfel_bake_render_lean`) is fed
is reproduced here so both renderers see byte-identical inputs:

  * activated per-Gauss params from the baked PLY, exactly as
    `prepare_gaussian_inputs` snapshots them (get_xyz / get_scaling /
    get_rotation / get_opacity incl. base_opacity / get_shape)
  * pre-activated SV state exactly as `benchmark_baked._make_sv_state` builds it
  * per-Gauss atlas-UV precompute using the same formulas as the CUDA fetch
    block, with the BC7 atlas split into a 2D-array texture whose layer cuts
    fall on shelf boundaries (Vulkan maxImageDimension2D = 32768 < our
    ~54k-row atlases; a rect never straddles a cut, so sampling is unchanged)
  * test cameras with view/proj matrices in the SAME memory order the CUDA
    kernel indexes (`matrix[0..15]` of the contiguous torch tensor), and the GT
    images for PSNR.

Usage:
  conda run -n nest_splatting python scripts/export_vk_bundle.py \
      --model_path outputs/mip_360/room/3D_SH_res/aftp_shres
"""
import os, sys, json, pickle, glob, struct, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.config import Config
from arguments import ModelParams

KERNEL_MAP = {"gaussian": 0, "beta": 1, "flex": 2, "general": 3, "beta_scaled": 4}
UV_EXTENT = 4.0


def make_sv_state(gaussians):
    """Verbatim mirror of benchmark_baked._make_sv_state (pre-activated SV)."""
    sv_sites = gaussians._sv_sites
    if sv_sites.numel() == 0:
        return None
    K = sv_sites.shape[1]
    sites_n = torch.nn.functional.normalize(sv_sites, dim=-1)
    sv_tau_raw = getattr(gaussians, '_sv_tau', None)
    if sv_tau_raw is not None and sv_tau_raw.numel() > 0:
        tau = torch.exp(sv_tau_raw)
    else:
        tau = torch.norm(sv_sites, dim=-1)
    sv_mask = getattr(gaussians, '_sv_mask', None)
    apply_mask = (sv_mask is not None
                  and (not getattr(gaussians, '_sv_training_flag', True))
                  and sv_mask.shape[0] == sv_sites.shape[0])
    if apply_mask:
        far = torch.full_like(sites_n, 1e8)
        sites_n = torch.where(sv_mask.unsqueeze(-1), sites_n, far)
    sv_dc_param = getattr(gaussians, '_sv_dc', None)
    if sv_dc_param is not None and sv_dc_param.numel() > 0:
        colors = (gaussians._sv_colors + sv_dc_param.unsqueeze(1)).contiguous()
    else:
        colors = gaussians._sv_colors.contiguous()
    return {'sites': sites_n.contiguous(), 'tau': tau.contiguous(),
            'colors': colors, 'K': int(K)}


def f32(t):
    return t.detach().float().contiguous().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out', default=None, help='default: <model_path>/vk_bundle')
    ap.add_argument('--layer_max', type=int, default=16384,
                    help='max rows per atlas array layer (must be <= maxImageDimension2D)')
    a = ap.parse_args()

    model_path = a.model_path.rstrip('/')
    out = a.out or os.path.join(model_path, 'vk_bundle')
    os.makedirs(out, exist_ok=True)
    baked = os.path.join(model_path, 'baked_atlas')

    # ---- args / cfg / iteration exactly as benchmark_baked does ----
    with open(os.path.join(model_path, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    cfg_path = os.path.join(model_path, 'config.yaml')
    cfg = Config(cfg_path) if os.path.exists(cfg_path) else Config(args.yaml)
    its = [int(os.path.basename(d).split('_')[1])
           for d in glob.glob(os.path.join(model_path, 'point_cloud/iteration_*'))]
    iteration = max(its)
    meta = json.load(open(os.path.join(baked, 'bake_meta.json')))
    kernel = meta.get('kernel', getattr(args, 'kernel', 'gaussian'))
    kernel_type = KERNEL_MAP[kernel]
    assert meta.get('feature_mode', 'SV') == 'SV', "exporter supports --feature SV only"
    assert int(meta.get('residual_mode', 0)) == 0, "exporter supports residual_mode 0 only"

    # ---- scene + baked PLY (skip_bake path of benchmark_baked) ----
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    test_cams = scene.getTestCameras()
    gaussians.load_ply(os.path.join(baked, 'baked.ply'))
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.kernel_type = kernel

    N = int(gaussians.get_xyz.shape[0])
    sv = make_sv_state(gaussians)
    assert sv is not None, "baked PLY has no SV state"
    K = sv['K']
    means = f32(gaussians.get_xyz)                      # [N,3]
    scales = f32(gaussians.get_scaling)                 # [N,2]
    rots = f32(gaussians.get_rotation)                  # [N,4]
    opac = f32(gaussians.get_opacity).reshape(N)        # [N]
    if kernel_type > 0 and hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
        shapes = f32(gaussians.get_shape).reshape(N)
    else:
        shapes = np.zeros(N, np.float32)
    assert scales.shape == (N, 2), f"expected 2DGS scales [N,2], got {scales.shape}"

    # ---- atlas rects -> per-Gauss atlas UV precompute (CUDA fetch-block formulas) ----
    rects = torch.load(os.path.join(baked, 'atlas_rects.pt'), map_location='cpu').float().numpy()
    assert rects.shape == (N, 4), f"rects {rects.shape} vs N={N}"
    Wp = int(meta['atlas_bc7_padded_w']); Hp = int(meta['atlas_bc7_padded_h'])
    bc7_path = os.path.join(baked, meta['atlas_bc7_file'])
    bc7_size = os.path.getsize(bc7_path)
    assert bc7_size == (Wp // 4) * (Hp // 4) * 16, f"bc7 size {bc7_size} != {Wp}x{Hp} blocks"

    # layer cuts on rows no rect straddles (rows are multiples of 4: BC7 blocks)
    u0 = rects[:, 0]; v0 = rects[:, 1]; wr = rects[:, 2]; hr = rects[:, 3]
    inside = np.zeros(Hp + 1, dtype=bool)          # inside[r] = some rect covers rows r-1 and r
    for i in np.nonzero(hr > 0)[0]:
        a0, a1 = int(v0[i]) + 1, int(v0[i] + hr[i])
        if a1 > a0:
            inside[a0:a1] = True
    cuts = [0]
    while cuts[-1] < Hp:
        lo = cuts[-1]; hi = min(lo + a.layer_max, Hp)
        r = hi - (hi % 4)
        while r > lo and inside[r]:
            r -= 4
        assert r > lo, f"no valid layer cut in ({lo}, {hi}] — shelf taller than layer_max?"
        cuts.append(r)
    nLayers = len(cuts) - 1
    layerH = max(cuts[i + 1] - cuts[i] for i in range(nLayers))
    cut_arr = np.array(cuts[:-1], dtype=np.int64)
    layer = np.searchsorted(cut_arr, v0.astype(np.int64), side='right') - 1
    layer = np.clip(layer, 0, nLayers - 1)
    lv0 = v0 - cut_arr[layer]                        # layer-local v0

    inv_2E = 1.0 / (2.0 * UV_EXTENT)
    au_scale = np.where(wr > 0, wr * inv_2E, 0.0).astype(np.float32)
    av_scale = np.where(hr > 0, hr * inv_2E, 0.0).astype(np.float32)
    ap_ = np.zeros((N, 12), np.float32)
    ap_[:, 0] = u0 - 0.5 + wr * 0.5;  ap_[:, 1] = lv0 - 0.5 + hr * 0.5
    ap_[:, 2] = au_scale;             ap_[:, 3] = av_scale
    ap_[:, 4] = u0;                   ap_[:, 5] = lv0
    ap_[:, 6] = u0 + wr - 1.001;      ap_[:, 7] = lv0 + hr - 1.001
    ap_[:, 8] = layer.astype(np.float32)

    # ---- write arrays ----
    def dump(name, arr):
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(out, name))
    dump('means.f32', means); dump('scales.f32', scales); dump('rots.f32', rots)
    dump('opac.f32', opac);   dump('shapes.f32', shapes)
    dump('sv_sites.f32', f32(sv['sites']).reshape(N * K * 3))
    dump('sv_tau.f32', f32(sv['tau']).reshape(N * K))
    dump('sv_colors.f32', f32(sv['colors']).reshape(N * K * 3))
    dump('atlas_params.f32', ap_)
    _lnk = os.path.join(out, "atlas.bc7"); (os.path.islink(_lnk) or os.path.exists(_lnk)) and os.remove(_lnk); os.symlink(os.path.abspath(bc7_path), _lnk)   # symlink: no 200-650 MB copy

    # ---- cameras + GT ----
    n_cam = len(test_cams)
    W0, H0 = int(test_cams[0].image_width), int(test_cams[0].image_height)
    gt_is_u8 = True
    with open(os.path.join(out, 'cams.bin'), 'wb') as f:
        f.write(struct.pack('<I', n_cam))
        for cam in test_cams:
            W, H = int(cam.image_width), int(cam.image_height)
            view = f32(cam.world_view_transform).reshape(16)   # memory order == CUDA matrix[i]
            proj = f32(cam.full_proj_transform).reshape(16)
            cpos = f32(cam.camera_center).reshape(3)
            tanfx = float(np.tan(cam.FoVx * 0.5)); tanfy = float(np.tan(cam.FoVy * 0.5))
            f.write(struct.pack('<II', W, H))
            f.write(view.tobytes()); f.write(proj.tobytes()); f.write(cpos.tobytes())
            f.write(struct.pack('<ff', tanfx, tanfy))
            gt = cam.original_image[:3].detach().float().cpu()        # [3,H,W] in [0,1]
            q = gt * 255.0
            if (q - q.round()).abs().max().item() > 1e-3:
                gt_is_u8 = False
            f.write(q.round().clamp(0, 255).to(torch.uint8).contiguous().numpy().tobytes())

    with open(os.path.join(out, 'meta.txt'), 'w') as f:
        kv = dict(N=N, K=K, kernel_type=kernel_type, kernel=kernel,
                  atlas_w=Wp, atlas_h=Hp, n_layers=nLayers, layer_h=layerH,
                  cuts=','.join(str(c) for c in cuts),
                  atlas_scale=float(meta['atlas_scale']), atlas_offset=float(meta['atlas_offset']),
                  sh_bias=float(meta.get('sh_bias', 0.5)), res_bias=float(meta.get('res_bias', 0.0)),
                  compact_mult=float(meta.get('compact_mult', 1.0)),
                  n_cams=n_cam, W=W0, H=H0, gt_u8=int(gt_is_u8))
        for k, v in kv.items():
            f.write(f'{k}={v}\n')

    print(f"[VKB] N={N:,} K={K} kernel={kernel}({kernel_type}) atlas {Wp}x{Hp} -> "
          f"{nLayers} layers x {layerH} rows (cuts {cuts}) | {n_cam} test cams {W0}x{H0} "
          f"| GT exact u8: {gt_is_u8} | -> {out}")


if __name__ == '__main__':
    main()
