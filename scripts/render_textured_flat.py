#!/usr/bin/env python3
"""
Render all views from a `--method mixed_3d` checkpoint with ONLY the textured
surfels visible, each shown as a flat color (per-Gauss stable hash) so the
result reads like a mesh of crisp 2DGS surfels. Untextured EWA surfels are
hidden by zeroing their opacity for the duration of this script — non-
destructive, in-memory only.

Two outputs per view:
  - {out}/flat/<split>/<NNN>.png     : hashed-id flat-color visualization
                                       (the per-pixel max-contributor id map
                                       coloured with the same routine as
                                       train.py:_colorize_max_contrib_idx)
  - {out}/rgb/<split>/<NNN>.png      : trained colors of the textured layer
                                       only (sanity reference; uses the same
                                       opacity mask so cross-set occlusion is
                                       gone)

Usage:
  conda run -n nest_splatting python scripts/render_textured_flat.py \
    -m outputs/mip_360/bonsai/mixed_3d/<expname> [--iteration -1]
"""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams, PipelineParams, get_combined_args
from train import _colorize_max_contrib_idx, merge_cfg_to_args


def main():
    parser = ArgumentParser(description="Flat-coloured textured-surfel visualization for mixed_3d")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--splits", default="train,test", type=str,
                        help="Comma-separated subset of {train,test}")
    parser.add_argument("--out_subdir", default="flat_textured", type=str,
                        help="Subdirectory under <model_path> for outputs")
    parser.add_argument("--save_rgb", action="store_true",
                        help="Also save the trained-color render of the textured layer only")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Render at most N views per split (sanity check). -1 = all.")
    parser.add_argument("--no_slice", action="store_true",
                        help="Debug: render the full model (no untex slice) to verify the "
                             "render path is correct; expected to match the training eval output.")
    parser.add_argument("--decompose", action="store_true",
                        help="Also render `sh_only` (SV/SH baseline, zero MLP) and `tex_only` "
                             "(MLP residual alone, SH/SV killed) variants alongside the full "
                             "render. Mirrors render_final_images' decomposition outputs.")
    args = get_combined_args(parser)
    exp_path = args.model_path
    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(exp_path, "point_cloud"))

    # Fold the training-time args (args.pkl) onto our argparse Namespace so the
    # method/kernel/kernel2/yaml/etc. survive — get_combined_args reads cfg_args
    # but the mixed_3d-era flags live in args.pkl.
    import pickle
    pkl_path = os.path.join(exp_path, "args.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            train_args = pickle.load(f)
        for k, v in vars(train_args).items():
            # Don't clobber CLI-provided overrides (e.g. user passed --iteration).
            if not hasattr(args, k) or getattr(args, k, None) is None:
                setattr(args, k, v)
            # The iteration/splits/save_rgb fields are render-time CLI knobs.
        # Force a couple of training-side fields we always want from the checkpoint.
        for k in ("method", "kernel", "kernel2", "feature", "hybrid_levels",
                  "disable_c2f", "aabb", "yaml", "lowpass", "texsplit"):
            if hasattr(train_args, k):
                setattr(args, k, getattr(train_args, k))
    print(f"[FLAT] Model:      {exp_path}")
    print(f"[FLAT] Iteration:  {iteration}")
    print(f"[FLAT] Method:     {getattr(args, 'method', '?')} (expects mixed_3d)")
    print(f"[FLAT] Kernel:     {getattr(args, 'kernel', '?')} / kernel2={getattr(args, 'kernel2', None)}")

    # --method mixed[_3d]: mirror the diff_surfel_3D_sh_res setters onto the
    # mixed submodule's device globals. train.py installs the same monkey-patch
    # at startup so that the renderer's `from diff_surfel_3D_sh_res import
    # set_mlp_weights` etc. also writes to diff_surfel_mixed_3d. Without it the
    # kernel that actually runs (mixed_3d) sees stale device globals → garbage
    # residuals / colors. MUST run before the first render() call.
    if getattr(args, "method", None) in ("mixed", "mixed_3d"):
        try:
            import diff_surfel_3D_sh_res as _ds_orig
            if args.method == "mixed_3d":
                import diff_surfel_mixed_3d as _ds_mirror
            else:
                import diff_surfel_mixed as _ds_mirror
            _MIRRORED_SETTERS = (
                'set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                'set_overdraw_lambda', 'set_weight_reg_lambda',
                'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                'set_depth_sort',
            )
            for _name in _MIRRORED_SETTERS:
                if not hasattr(_ds_orig, _name) or not hasattr(_ds_mirror, _name):
                    continue
                _of = getattr(_ds_orig, _name)
                _mf = getattr(_ds_mirror, _name)
                def _make_mirror(of, mf):
                    def _wrapped(*a, **k):
                        of(*a, **k); mf(*a, **k)
                    return _wrapped
                setattr(_ds_orig, _name, _make_mirror(_of, _mf))
            print(f"[FLAT] mirrored diff_surfel_3D_sh_res setters → diff_surfel_{'mixed_3d' if args.method=='mixed_3d' else 'mixed'}")
        except ImportError as _e:
            print(f"[FLAT] WARNING: could not install setter mirror: {_e}")

    # Load config (yaml saved in training args) + merge so cfg fields are honoured.
    yaml_file = getattr(args, "yaml", None) or "tiny"
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    # INGP for the hash+MLP path. The class infers `is_mixed_mode` /
    # `is_mixed_3d_mode` from `args.method` at __init__ — we already folded
    # args.pkl into `args` above, so this picks up `method=mixed_3d` and the
    # renderer dispatch downstream applies the per-pixel ReLU + kernel2 bits.
    ingp = INGP(cfg_model, args=args).to("cuda")
    ingp.load_model(exp_path, iteration)
    ingp.set_active_levels(iteration)
    if getattr(ingp, "is_mixed_3d_mode", False):
        print(f"[FLAT] INGP mode: mixed_3d (per-pixel ReLU + kernel2 active)")

    # Device-global setup — mirror train.py's startup so the kernel matches the
    # trained pipeline. Without these, residual_mode defaults to 0 (per-Gauss
    # outer ReLU) → signed residuals can't subtract → channels pile up → the
    # "oversaturated weird colors" symptom on the previous run.
    if args.method in ("mixed", "mixed_3d"):
        if args.method == "mixed_3d":
            from diff_surfel_mixed_3d import (set_residual_mode, set_activation_bias,
                                              set_compact_mult)
        else:
            from diff_surfel_mixed import (set_residual_mode, set_activation_bias,
                                           set_compact_mult)
        _sh_bias, _res_bias = getattr(args, "activation_bias", [0.5, 0.0])
        set_activation_bias(sh_bias=float(_sh_bias), res_bias=float(_res_bias))
        set_residual_mode(2)  # mixed: signed residual + Python per-pixel ReLU
        _cm = float(getattr(args, "fastgs_mult", 0.5))
        set_compact_mult(_cm)
        # Mirror onto diff_surfel_3D_sh_res too — gaussian_renderer/__init__.py's
        # mixed_3d dispatch path also imports setters from there for the textured
        # half's MLP backward (no harm at inference; matches train.py).
        from diff_surfel_3D_sh_res import (set_activation_bias as _sab_res,
                                           set_residual_mode as _srm_res)
        _sab_res(sh_bias=float(_sh_bias), res_bias=float(_res_bias))
        _srm_res(2)
        print(f"[FLAT] CUDA globals: sh_bias={_sh_bias} res_bias={_res_bias} "
              f"residual_mode=2 compact_mult={_cm}")

    # Load Gaussians.
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    # `--feature SV/voronoi/beta/sg`: load_ply doesn't restore feature_mode (only
    # create_from_pcd does, at training time), so it defaults to "sh" → the
    # renderer's SV dispatch is skipped → real (unoptimized, near-zero)
    # _features_dc is used → washed-out output. Set it from args here.
    _feat_mode = getattr(args, "feature", "sh")
    gaussians.feature_mode = _feat_mode
    gaussians._sv_training_flag = False  # inference: apply _sv_mask if populated
    print(f"[FLAT] feature_mode = {gaussians.feature_mode}")

    N = gaussians.get_xyz.shape[0]
    if not hasattr(gaussians, "_is_textured") or gaussians._is_textured.numel() != N:
        raise RuntimeError(
            f"Checkpoint has no per-Gauss _is_textured tensor (or mismatched size: "
            f"{getattr(gaussians, '_is_textured', None)}). This script requires "
            f"--method mixed/mixed_3d with --texsplit having fired.")

    mask_t = gaussians._is_textured  # bool [N]
    n_tex = int(mask_t.sum().item())
    n_untex = N - n_tex
    print(f"[FLAT] Surfels:    {N:,}  (textured: {n_tex:,}  untextured: {n_untex:,})")
    if n_tex == 0:
        raise RuntimeError("No textured surfels in the model — nothing to visualize.")

    # Hide untextured surfels by clobbering their _opacity. sigmoid(-1e3) ≈ 0,
    # so get_opacity becomes ~base_opacity·0 + base_opacity ≈ base. But because
    # the alpha gate `α<1/255` culls in CUDA, a tiny α (≤ ~0.004) still escapes
    # → push opacity to a very small fixed value. Using -1e3 → sigmoid ~ 0,
    # then get_opacity ≈ base_opacity (~0.05 by default). That's still > 1/255.
    # Better: zero `_opacity` *raw value*, AND set base_opacity to 0 just for
    # the untextured rows — but base_opacity is a global scalar. So the most
    # surgical fix is to literally drop those rows out of the model for this
    # rendering pass and restore at the end.
    saved = {}
    if args.no_slice:
        print(f"[FLAT] --no_slice: rendering full model (textured + untextured) for parity check.")
    else:
        print(f"[FLAT] Hiding untextured surfels (in-memory slice)…")
        # Generic slice: every tensor attribute whose first dim matches N gets
        # sliced by mask_t. This avoids the hardcoded-name bug class.
        for f in sorted(vars(gaussians).keys()):
            t = getattr(gaussians, f)
            if not torch.is_tensor(t):
                continue
            if t.numel() == 0 or t.shape[0] != N:
                continue
            saved[f] = t  # original (Parameter or Tensor)
            sliced = t[mask_t].detach().clone()
            if isinstance(t, torch.nn.Parameter):
                setattr(gaussians, f, torch.nn.Parameter(sliced.requires_grad_(t.requires_grad)))
            else:
                setattr(gaussians, f, sliced)
        print(f"[FLAT] Sliced tensors: {sorted(saved.keys())}")
        print(f"[FLAT] After slice: {gaussians.get_xyz.shape[0]:,} surfels (all textured)")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    out_root = os.path.join(exp_path, args.out_subdir)
    print(f"[FLAT] Output:     {out_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        cams = scene.getTrainCameras() if split == "train" else scene.getTestCameras()
        if len(cams) == 0:
            print(f"[FLAT] split={split}: no cameras, skip")
            continue
        flat_dir = os.path.join(out_root, "flat", split); os.makedirs(flat_dir, exist_ok=True)
        rgb_dir  = os.path.join(out_root, "rgb",  split); os.makedirs(rgb_dir,  exist_ok=True) if args.save_rgb else None
        if args.decompose:
            rgb_sh_dir  = os.path.join(out_root, "rgb_sv_only", split); os.makedirs(rgb_sh_dir,  exist_ok=True)
            rgb_tex_dir = os.path.join(out_root, "rgb_tex_only", split); os.makedirs(rgb_tex_dir, exist_ok=True)

        if args.limit > 0:
            cams = cams[: args.limit]
        n_invalid_total = 0; n_pixels_total = 0
        with torch.no_grad():
            for idx, view in enumerate(tqdm(cams, desc=f"flat::{split}")):
                # Match `render_final_images` (train.py:4228) verbatim — it
                # produced `final_test_renders/*.png` (our visual ground truth).
                # Extra render kwargs (lowpass / aa / pixel_center / aabb_mode)
                # are deliberately omitted; render_final_images doesn't pass
                # them either and gets the right colors.
                pkg = render(view, gaussians, pipe, bg,
                             ingp=ingp, beta=beta_cfg, iteration=iteration,
                             cfg=cfg_model,
                             skybox=None, background_mode="none", bg_hashgrid=None)
                mci = pkg.get("max_contrib_idx", None)
                if mci is None:
                    raise RuntimeError("Renderer did not return max_contrib_idx; "
                                       "this script requires the mixed_3d / 3D_SH_res rasterizer.")
                flat = _colorize_max_contrib_idx(mci)  # [H, W, 3] float in [0,1]
                save_img_u8(flat, os.path.join(flat_dir, f"{idx:04d}.png"))

                # Track invalid (no-contributor) pixel rate for sanity logging.
                mci_arr = mci.squeeze().detach().cpu().numpy().astype(np.int64)
                n_invalid_total += int((mci_arr < 0).sum())
                n_pixels_total  += int(mci_arr.size)

                if args.save_rgb:
                    rgb = torch.clamp(pkg["render"], 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    save_img_u8(rgb, os.path.join(rgb_dir, f"{idx:04d}.png"))

                if args.decompose:
                    # decompose_mode='sh_only': zero MLP weights → SV/SH baseline only.
                    # decompose_mode='tex_only': sh_bias=-999 → SV/SH killed, residual only.
                    # The renderer applies the bias overrides then restores afterward
                    # (see gaussian_renderer/__init__.py:2526). Per-view sequential
                    # calls — `set_X` is wrapped by our mirror so both submodules stay
                    # in sync.
                    pkg_sv = render(view, gaussians, pipe, bg,
                                    ingp=ingp, beta=beta_cfg, iteration=iteration,
                                    cfg=cfg_model,
                                    skybox=None, background_mode="none", bg_hashgrid=None,
                                    decompose_mode='sh_only')
                    rgb_sv = torch.clamp(pkg_sv["render"], 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    save_img_u8(rgb_sv, os.path.join(rgb_sh_dir, f"{idx:04d}.png"))

                    pkg_tx = render(view, gaussians, pipe, bg,
                                    ingp=ingp, beta=beta_cfg, iteration=iteration,
                                    cfg=cfg_model,
                                    skybox=None, background_mode="none", bg_hashgrid=None,
                                    decompose_mode='tex_only')
                    rgb_tx = torch.clamp(pkg_tx["render"], 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    save_img_u8(rgb_tx, os.path.join(rgb_tex_dir, f"{idx:04d}.png"))

        pct_invalid = 100.0 * n_invalid_total / max(1, n_pixels_total)
        print(f"[FLAT] split={split}: {len(cams)} views → {flat_dir}  "
              f"(invalid pixels {pct_invalid:.1f}%)")

    # Restore (in case some upstream caller reuses `gaussians`).
    for f, t in saved.items():
        setattr(gaussians, f, t)

    print("[FLAT] Done.")


if __name__ == "__main__":
    main()
