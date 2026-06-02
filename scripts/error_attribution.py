"""Per-Gaussian error attribution diagnostic.

Renders every training view, compares to GT, attributes per-pixel L1 error to the
contributing Gaussian via `max_contrib_idx`, accumulates a per-Gauss error vector
across all views, tags the top X% as "candidates for deformation", and re-renders
a handful of views with the tagged Gaussians' pixels highlighted in red.

Mirrors train.py's render() call site (the same one used at `iteration` 35000 in
the final-test block at train.py:3925-3934). We deliberately reuse train.py's
config loader + GaussianModel + Scene + INGP plumbing to avoid drift between
"how training rendered" and "what this script renders".

Usage:
    conda run -n nest_splatting python scripts/error_attribution.py \
        --model_path /home/nilkel/Projects/nest-splatting/outputs/personal/Max/mixed_3d/perGS2_SV_30thr_005w25gLP4lev_FRP5k10_N2F_Jac_5k_099sp_hresnoev \
        --top_pct 5.0 --num_overlays 12
"""
import os, sys, json, glob, argparse, pickle
import numpy as np
import torch
from PIL import Image

# Run from project root: hash_encoder/modules.py does `sys.path += ["./", "../"]`,
# which makes `./` mean CWD. If CWD ≠ project root, subsequent
# `import diff_surfel_3D_sh_res` finds the submodule build dir (which has no
# __init__.py) as a namespace package and silently strips all setters.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJ_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.general_utils import safe_state


def load_args_and_cfg(model_path: str):
    args_path = os.path.join(model_path, 'args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    cfg_path = os.path.join(model_path, 'config.yaml')
    cfg = Config(cfg_path)
    return args, cfg


def attribute_alpha(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args,
                    top_pct: float, num_overlays: int, out_dir: str, rank_by: str = 'peak',
                    error_metric: str = 'l1'):
    """Alpha-blend-weighted error attribution via forward-only CUDA side output.

    Routes the render through `diff_surfel_attribute` (a clone of
    `diff_surfel_mixed_3d` with one extra forward-only hook). On every Gauss-pixel
    hit inside the render kernel:

        atomicAdd(err_accum[gauss_id], T·α · error_image[pixel])

    where `error_image` is a per-pixel L1 |render − gt| computed from the prior
    frame's output (one-frame lag, harmless for a static-geometry diagnostic).
    Two-pass: pass-1 renders + computes error_image, pass-2 same view with the
    device pointer installed accumulates the score. No autograd, no INGP grads.

    Across views we track {n, sum, sum², max}. Then:
      mean  = sum / n
      std   = sqrt(sum²/n − mean²)
      peak  = max − mean       ("how much worse is this Gauss in its worst view")
      cv    = std / mean       (coefficient of variation; unitless)
      total = sum              (legacy "total error")

    The 'peak' / 'cv' metrics isolate dynamics: a Gauss that's perfect in most
    views but bad in a few will have low mean / high max → ranks high on peak.
    """
    # Route render() through the attribute submodule. Mirror every existing
    # diff_surfel_3D_sh_res setter so its device globals stay in sync.
    import gaussian_renderer as _gr
    import diff_surfel_attribute as _ds_attr
    import diff_surfel_3D_sh_res as _ds_orig
    _MIRROR = ('set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
               'set_overdraw_lambda', 'set_weight_reg_lambda',
               'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
               'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
               'set_depth_sort', 'set_ste_relu')
    for _nm in _MIRROR:
        if not (hasattr(_ds_orig, _nm) and hasattr(_ds_attr, _nm)): continue
        _of = getattr(_ds_orig, _nm); _mf = getattr(_ds_attr, _nm)
        def _mk(of, mf):
            def _w(*a, **k):
                of(*a, **k); mf(*a, **k)
            return _w
        setattr(_ds_orig, _nm, _mk(_of, _mf))
    # Re-apply activation_bias / compact_mult through the now-mirrored setters
    # so diff_surfel_attribute's globals match.
    _sh_b, _res_b = getattr(args, 'activation_bias', [0.5, 0.0])
    _ds_orig.set_activation_bias(_sh_b, _res_b)
    if getattr(args, 'fastgs', False):
        _ds_orig.set_compact_mult(getattr(args, 'fastgs_mult', 0.5))
    # Hijack the gaussian_renderer dispatch so mixed_3d → diff_surfel_attribute.
    _saved_rmod = _gr._mixed_3d_rasterizer
    _saved_avail = _gr.MIXED_3D_RASTERIZER_AVAILABLE
    _gr._mixed_3d_rasterizer = _ds_attr
    _gr.MIXED_3D_RASTERIZER_AVAILABLE = True
    print(f"[ATTR-ALPHA] routed render() through diff_surfel_attribute")
    os.makedirs(out_dir, exist_ok=True)
    render_kw_base = dict(
        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
        skybox=None, background_mode=args.background, bg_hashgrid=None,
        aabb_mode=getattr(args, 'aabb', '2dgs'),
        lowpass=getattr(args, 'lowpass', False),
        pixel_center=getattr(args, 'pixel_center', False),
        antialiasing=getattr(args, 'antialiasing', 0.0),
        aa=getattr(args, 'aa', 0.0),
        max_intersections_per_pixel=getattr(args, 'max_intersections_per_pixel', 32),
    )
    cams = scene.getTrainCameras().copy()
    import re
    def _sk(c):
        m = re.search(r'(\d+)', c.image_name)
        return int(m.group(1)) if m else c.image_name
    cams.sort(key=_sk)

    N = gaussians.get_xyz.shape[0]
    print(f"[ATTR-ALPHA] {len(cams)} train cameras, {N} Gaussians, "
          f"rank_by={rank_by}, error_metric={error_metric}")

    # LPIPS spatial map: per-pixel perceptual error (AlexNet feature diffs upsampled).
    # Returns (1,1,H,W) in spatial mode → squeeze to (H,W). Inputs need to be
    # (1,3,H,W) in [-1, 1].
    lpips_fn = None
    if error_metric == 'lpips':
        import lpips as _lpips_pkg
        lpips_fn = _lpips_pkg.LPIPS(net='alex', spatial=True, verbose=False).to('cuda').eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print(f"[ATTR-ALPHA] LPIPS(alex, spatial=True) initialised")

    err_accum_gpu = torch.zeros(N, device='cuda', dtype=torch.float32)
    err_n     = np.zeros(N, dtype=np.int64)
    err_sum   = np.zeros(N, dtype=np.float64)
    err_sum2  = np.zeros(N, dtype=np.float64)
    err_max   = np.zeros(N, dtype=np.float64)

    # Two-pass per view: pass 1 = vanilla render → e_pix; pass 2 = same render
    # with the device pointer installed → atomic accumulation into err_accum.
    for vi, cam in enumerate(cams):
        # Pass 1: no attribution (pointers cleared).
        _empty = torch.empty(0, device='cuda', dtype=torch.float32)
        _ds_attr._C.set_error_attribution(_empty, _empty)
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, is_training=False, **render_kw_base)
        C = pkg['render'].clamp(0, 1)
        gt = cam.original_image.to('cuda').clamp(0, 1)
        if gt.shape[0] == 4: gt = gt[:3]
        if error_metric == 'lpips':
            # LPIPS expects (B,3,H,W) in [-1,1]. Spatial=True → (1,1,H,W) of
            # per-pixel perceptual error (broader support than L1; smoother
            # gradients across patches). Detach since we already have no grad.
            with torch.no_grad():
                _c = (C.unsqueeze(0) * 2 - 1)
                _g = (gt.unsqueeze(0) * 2 - 1)
                _lp = lpips_fn(_c, _g)
            e_pix = _lp.squeeze(0).squeeze(0).contiguous()       # (H, W) float32
        else:
            e_pix = (C - gt).abs().mean(dim=0).contiguous()       # (H, W) float32

        # Pass 2: install pointers, re-render, harvest err_accum.
        err_accum_gpu.zero_()
        _ds_attr._C.set_error_attribution(err_accum_gpu, e_pix)
        with torch.no_grad():
            _ = render(cam, gaussians, pipe, background, is_training=False, **render_kw_base)
        _ds_attr._C.set_error_attribution(_empty, _empty)
        torch.cuda.synchronize()

        score = err_accum_gpu.detach().cpu().numpy().astype(np.float64)
        active = score > 0
        err_n[active]    += 1
        err_sum[active]  += score[active]
        err_sum2[active] += score[active] ** 2
        err_max = np.maximum(err_max, score)
        if (vi + 1) % 25 == 0 or vi == len(cams) - 1:
            print(f"[ATTR-ALPHA] view {vi+1}/{len(cams)}  active={int(active.sum())}  "
                  f"L1_img={float(e_pix.mean()):.4f}  max_score={score.max():.4g}")
        del pkg, C, e_pix
        torch.cuda.empty_cache()

    n_eff = np.maximum(err_n, 1)
    mean = err_sum / n_eff
    var  = np.maximum(err_sum2 / n_eff - mean ** 2, 0.0)
    std  = np.sqrt(var)
    peak = err_max - mean                       # worst-view excess over mean
    cv   = std / np.maximum(mean, 1e-12)        # coefficient of variation
    total = err_sum

    metrics = {'mean': mean, 'std': std, 'peak': peak, 'cv': cv, 'total': total,
               'max': err_max, 'n': err_n}
    score = metrics[rank_by]
    # Only tag Gausses seen in ≥ a few views, otherwise spurious peaks dominate.
    min_n = max(3, len(cams) // 50)
    valid = err_n >= min_n
    score_v = np.where(valid, score, -np.inf)
    thr = np.percentile(score_v[valid], 100.0 - top_pct) if valid.any() else 0.0
    tagged = (score_v >= thr) & valid
    print(f"[ATTR-ALPHA] tagged {int(tagged.sum())}/{N} ({100*tagged.sum()/N:.2f}%) "
          f"by '{rank_by}' (min_n={min_n}, thr={thr:.4g})")
    # Save (metric-tagged so multiple runs coexist).
    _npz = f'attribution_alpha_{error_metric}.npz'
    np.savez(os.path.join(out_dir, _npz),
             err_n=err_n, err_sum=err_sum, err_sum2=err_sum2, err_max=err_max,
             mean=mean, std=std, peak=peak, cv=cv, total=total,
             tagged=tagged, rank_by=np.array(rank_by),
             error_metric=np.array(error_metric))
    print(f"[ATTR-ALPHA] saved {_npz}")
    torch.cuda.empty_cache()

    # Phase 2 — overlays. We still need max_contrib_idx for *spatial* localisation of
    # which pixels each Gauss owns. The score itself is per-Gauss; we paint pixels
    # where their max-contributor is tagged, coloured by the Gauss's score (inferno).
    if num_overlays <= 0: return
    step = max(1, len(cams) // num_overlays)
    sel = cams[::step][:num_overlays]
    print(f"[OVERLAY] rendering {len(sel)} overlay frames (continuous heatmap by '{rank_by}')")
    # Normalize score for colour mapping (log-scale tail).
    s_pos = np.where(tagged, score, 0.0)
    s_pos = np.log1p(np.maximum(s_pos, 0.0))
    s_norm = s_pos / max(s_pos.max(), 1e-12)
    from matplotlib import cm
    for cam in sel:
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, is_training=False, **render_kw_base)
        img = pkg['render'].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        mci = pkg['max_contrib_idx'].detach().cpu().numpy().astype(np.int64)
        H, W = mci.shape
        valid_pix = (mci >= 0) & (mci < N)
        # Per-pixel score = score[max_contrib_idx], 0 where invalid or untagged.
        pix_score = np.zeros((H, W), dtype=np.float64)
        pix_score[valid_pix] = s_norm[mci[valid_pix]]
        # Heatmap overlay: blend inferno(pix_score) onto img where pix_score > 0.
        heat = cm.inferno(pix_score.clip(0, 1))[:, :, :3].astype(np.float32)
        mask = (pix_score > 0).astype(np.float32)[:, :, None]
        out = img * (1.0 - 0.6 * mask) + heat * (0.6 * mask)
        u8 = (out * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(u8).save(os.path.join(out_dir, f'{cam.image_name}_alpha_{error_metric}_{rank_by}.png'))
        # Per-pixel L1 error map for cross-reference.
        gt = cam.original_image.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        if gt.shape[2] == 4: gt = gt[:, :, :3]
        epp = np.mean(np.abs(img - gt), axis=2)
        epp_fix = (epp / 0.05).clip(0, 1)
        Image.fromarray((cm.inferno(epp_fix)[:, :, :3] * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f'{cam.image_name}_error_{error_metric}_p05.png'))
    print(f"[OVERLAY] wrote {len(sel)} alpha-heatmap + error frames → {out_dir}")
    # Restore original dispatch (in case the function is re-entered).
    _gr._mixed_3d_rasterizer = _saved_rmod
    _gr.MIXED_3D_RASTERIZER_AVAILABLE = _saved_avail


def attribute_errors(scene, gaussians, pipe, background, ingp, beta, iteration, cfg_model, args,
                     top_pct: float, num_overlays: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Render kwargs that match training (read off args.pkl): without these the
    # render() defaults (aabb_mode="2dgs", lowpass=False) silently mismatch the
    # trained configuration, producing wrong colours / tile-boundary error
    # patterns. fastgs is wired via set_compact_mult above.
    render_kw = dict(
        ingp=ingp, beta=beta, iteration=iteration, cfg=cfg_model,
        skybox=None, background_mode=args.background, bg_hashgrid=None,
        aabb_mode=getattr(args, 'aabb', '2dgs'),
        lowpass=getattr(args, 'lowpass', False),
        pixel_center=getattr(args, 'pixel_center', False),
        antialiasing=getattr(args, 'antialiasing', 0.0),
        aa=getattr(args, 'aa', 0.0),
        max_intersections_per_pixel=getattr(args, 'max_intersections_per_pixel', 32),
        is_training=False,
    )
    cams = scene.getTrainCameras().copy()
    # Stable order across runs.
    def _sk(c):
        import re
        m = re.search(r'(\d+)', c.image_name)
        return int(m.group(1)) if m else c.image_name
    cams.sort(key=_sk)

    N = gaussians.get_xyz.shape[0]
    err_sum = np.zeros(N, dtype=np.float64)
    err_count = np.zeros(N, dtype=np.int64)
    print(f"[ATTR] {len(cams)} train cameras, {N} Gaussians")

    # Phase 1 — render every view, attribute per-pixel L1 → contributing Gauss.
    for vi, cam in enumerate(cams):
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, **render_kw)
        img = pkg['render'].clamp(0, 1).detach().cpu().numpy()           # (3, H, W)
        mci = pkg.get('max_contrib_idx', None)
        if mci is None:
            print(f"[ATTR] no max_contrib_idx in render_pkg — bailing"); return
        mci_np = mci.detach().cpu().numpy().astype(np.int64)             # (H, W)
        gt = cam.original_image.clamp(0, 1).detach().cpu().numpy()
        if gt.shape[0] == 4: gt = gt[:3]
        err_per_pix = np.mean(np.abs(img - gt), axis=0)                  # (H, W)
        # Attribute. Negative ids = no contributor; skip them.
        valid = (mci_np >= 0) & (mci_np < N)
        ids = mci_np[valid].ravel()
        errs = err_per_pix[valid].ravel()
        np.add.at(err_sum, ids, errs)
        np.add.at(err_count, ids, 1)
        if (vi + 1) % 25 == 0 or vi == len(cams) - 1:
            print(f"[ATTR] view {vi+1}/{len(cams)}")

    # Per-Gauss aggregated error.
    err_mean = err_sum / np.maximum(err_count, 1)
    # Total contribution (sum) is more useful for tagging — it weights by how
    # often a Gauss is the max-contributor, capturing both how-wrong AND how-
    # often-on-screen.
    err_total = err_sum
    print(f"[ATTR] err_total: nonzero={int((err_count>0).sum())}/{N}")
    print(f"       p50={np.percentile(err_total[err_count>0], 50):.4f}")
    print(f"       p90={np.percentile(err_total[err_count>0], 90):.4f}")
    print(f"       p99={np.percentile(err_total[err_count>0], 99):.4f}")
    thr = np.percentile(err_total, 100.0 - top_pct)
    tagged = (err_total >= thr) & (err_count > 0)
    print(f"[ATTR] tagged {int(tagged.sum())}/{N} ({100*tagged.sum()/N:.2f}%) "
          f"as high-error (top {top_pct}% by Σerr).")

    # Save raw attribution.
    np.savez(os.path.join(out_dir, 'attribution.npz'),
             err_sum=err_sum, err_count=err_count, tagged=tagged)
    print(f"[ATTR] saved attribution.npz")

    # Phase 2 — render `num_overlays` evenly-spaced views with tagged-pixel overlay.
    if num_overlays <= 0: return
    step = max(1, len(cams) // num_overlays)
    sel = cams[::step][:num_overlays]
    print(f"[OVERLAY] rendering {len(sel)} overlay frames")
    for cam in sel:
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, **render_kw)
        img = pkg['render'].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        mci = pkg['max_contrib_idx'].detach().cpu().numpy().astype(np.int64)
        H, W = mci.shape
        mask = np.zeros((H, W), dtype=bool)
        valid = (mci >= 0) & (mci < N)
        mask[valid] = tagged[mci[valid]]
        # Pure red overlay, 60% opacity.
        red = np.array([1.0, 0.1, 0.1], dtype=np.float32)
        out = img.copy()
        out[mask] = 0.4 * img[mask] + 0.6 * red
        u8 = (out * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(u8).save(os.path.join(out_dir, f'{cam.image_name}_tagged.png'))
        # Also save the per-pixel error map (jet) for context.
        gt = cam.original_image.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        if gt.shape[2] == 4: gt = gt[:, :, :3]
        epp = np.mean(np.abs(img - gt), axis=2)
        from matplotlib import cm
        epp_norm = (epp / max(epp.max(), 1e-6)).clip(0, 1)
        heat = (cm.inferno(epp_norm)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(heat).save(os.path.join(out_dir, f'{cam.image_name}_error.png'))
    print(f"[OVERLAY] wrote {len(sel)} tagged + error frames → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--iter', type=int, default=35000)
    p.add_argument('--top_pct', type=float, default=5.0)
    p.add_argument('--num_overlays', type=int, default=12)
    p.add_argument('--out_subdir', type=str, default='error_attribution')
    p.add_argument('--mode', choices=['maxcontrib', 'alpha'], default='alpha',
                   help='attribution scheme: maxcontrib = per-pixel max_contrib_idx; '
                        'alpha = grad-based alpha-blend weighted (default)')
    p.add_argument('--rank_by', choices=['peak', 'cv', 'std', 'mean', 'total', 'max'],
                   default='peak', help='per-Gauss metric to rank by (alpha mode only)')
    p.add_argument('--error_metric', choices=['l1', 'lpips'], default='l1',
                   help='per-pixel error map: l1=|render-gt| mean over channels; '
                        'lpips=AlexNet feature-diff spatial map (smoother, perceptual)')
    a = p.parse_args()

    safe_state(silent=False)
    torch.set_grad_enabled(False)

    args, cfg = load_args_and_cfg(a.model_path)
    # Minimal model + pipe wrap to satisfy render(): mirror train.py's flow.
    parser = ArgumentParser()
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    # Synthesize a Namespace by copying needed attrs from `args`.
    ns = Namespace(**vars(args))
    ns.model_path = a.model_path
    # Some configs use these on the Namespace:
    for k in ['white_background', 'eval', 'data_device', 'images', 'resolution']:
        if not hasattr(ns, k):
            setattr(ns, k, getattr(args, k, None))

    gaussians = GaussianModel(args.sh_degree if hasattr(args, 'sh_degree') else 3)
    # Echo training kernel settings.
    gaussians.kernel_type = getattr(args, 'kernel', 'beta_scaled')
    gaussians.kernel_type2 = getattr(args, 'kernel2', None)

    scene = Scene(ns, gaussians, load_iteration=a.iter, shuffle=False)

    # Build the INGP exactly as training did, then load weights.
    ingp = INGP(cfg_model=cfg, args=args).cuda() if args.method != '2dgs' else None
    if ingp is not None:
        ngp_files = sorted(glob.glob(os.path.join(a.model_path, 'ngp_*.pth')))
        if ngp_files:
            ckpt = torch.load(ngp_files[-1], map_location='cuda', weights_only=True)
            ingp.load_state_dict(ckpt, strict=False)
            print(f"[INGP] loaded {os.path.basename(ngp_files[-1])}")
        else:
            print(f"[INGP] WARNING: no ngp_*.pth found")
        # Force full hashgrid for inference (no coarse-to-fine ramp).
        ingp.set_active_levels(current_iter=a.iter)

    # CRITICAL: push the training-time activation_bias + residual_mode into the
    # CUDA device globals. CUDA defaults are sh_bias=0.5, res_bias=0.5, but
    # nest-splatting overrides res_bias to 0.0 (and similar for residual_mode
    # per-method). Without these explicit setter calls, the renderer's color
    # path is off by a constant 0.5 in the residual and the activation chain
    # collapses, producing visibly wrong colors that LOOK like tiling
    # artefacts (per-Gauss color discontinuities surface where they'd
    # normally be hidden by the trained activation bias).
    sh_bias = getattr(args, 'activation_bias', [0.5, 0.0])[0]
    res_bias = getattr(args, 'activation_bias', [0.5, 0.0])[1]
    _rm = getattr(args, '_residual_mode', 0)
    if args.method in ("3D_SH_res", "3D_SH_res_sep", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
        from diff_surfel_3D_sh_res import set_activation_bias as _sab
        _sab(sh_bias, res_bias)
        if _rm in (1, 2):
            from diff_surfel_3D_sh_res import set_residual_mode as _srm
            _srm(_rm)
        print(f"[CUDA] set_activation_bias({sh_bias}, {res_bias}); residual_mode={_rm}")
    # mixed/mixed_3d also need the setter mirror to propagate to the
    # diff_surfel_mixed[_3d] device globals (see train.py:163+ monkey-patch).
    if args.method in ("mixed_3d", "mixed_3d_sep"):
        import diff_surfel_mixed_3d as _ds_m3d
        _ds_m3d.set_activation_bias(sh_bias, res_bias)
        if _rm in (1, 2):
            _ds_m3d.set_residual_mode(_rm)
        print(f"[CUDA] mirror → diff_surfel_mixed_3d")
    elif args.method in ("mixed", "mixed_sep"):
        import diff_surfel_mixed as _ds_m
        _ds_m.set_activation_bias(sh_bias, res_bias)
        if _rm in (1, 2):
            _ds_m.set_residual_mode(_rm)
        print(f"[CUDA] mirror → diff_surfel_mixed")

    # FastGS Compact-Box AABB multiplier — training calls this when --fastgs.
    # Mirror onto the mixed_3d/mixed CUDA module too so device-globals match.
    if getattr(args, 'fastgs', False):
        _fm = getattr(args, 'fastgs_mult', 0.5)
        if args.method in ("3D_SH_res", "3D_SH_res_sep", "mixed", "mixed_3d", "mixed_sep", "mixed_3d_sep"):
            from diff_surfel_3D_sh_res import set_compact_mult as _scm
            _scm(_fm)
        if args.method in ("mixed_3d", "mixed_3d_sep"):
            import diff_surfel_mixed_3d as _ds_m3d
            _ds_m3d.set_compact_mult(_fm)
        elif args.method in ("mixed", "mixed_sep"):
            import diff_surfel_mixed as _ds_m
            _ds_m.set_compact_mult(_fm)
        print(f"[CUDA] set_compact_mult({_fm}) (fastgs)")
    # PipelineParams uses argparse → build from the existing args namespace so
    # all defaults (depth_ratio, etc.) land on `pipe`.
    _pp_parser = ArgumentParser()
    pipe = PipelineParams(_pp_parser).extract(_pp_parser.parse_args([]))
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    pipe.debug = False
    pipe.depth_ratio = getattr(pipe, 'depth_ratio', 0.0)

    beta = getattr(cfg.surfel, 'tg_beta', 1.0)
    background = torch.zeros(3, device='cuda')
    out_dir = os.path.join(a.model_path, a.out_subdir)
    if a.mode == 'alpha':
        attribute_alpha(scene, gaussians, pipe, background, ingp, beta, a.iter, cfg,
                        args, top_pct=a.top_pct, num_overlays=a.num_overlays,
                        out_dir=out_dir, rank_by=a.rank_by, error_metric=a.error_metric)
    else:
        attribute_errors(scene, gaussians, pipe, background, ingp, beta, a.iter, cfg,
                         args, top_pct=a.top_pct, num_overlays=a.num_overlays,
                         out_dir=out_dir)


if __name__ == '__main__':
    main()
