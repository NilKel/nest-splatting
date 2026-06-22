#!/usr/bin/env python3
"""Debug-render the first test view of a checkpoint, using the model's TRAINED
activation_bias verbatim. Variants written depend on the checkpoint shape:

  res_switch / 3D_SH_res / pre-split (ALL surfels textured, with SV):
    1_sv_only.png        — `decompose_mode='sh_only'` (zeros MLP → leaves
                            `ReLU(SV + sh_bias)·T·α` per pixel = pure SV image)
    2_residual_only.png  — `decompose_mode='tex_only'` (sh_bias→-999 + zero
                            SH/SV → leaves `residual·T·α` per pixel)
    3_full.png           — standard rasterizer output

  res_3d post-split (tex carriers with SV=0 + residual; untex EWA with SV):
    1_tex_carriers_full.png      — only textured carriers contribute
                                    (untex opacity → 0)
    2_tex_carriers_sv_only.png   — tex carriers + decompose 'sh_only'.
                                    Since tex SV is zeroed at split, this
                                    is the +0.5 SH-bias baseline ONLY.
    3_untex_carriers_full.png    — only untextured EWA carriers contribute
                                    (tex opacity → 0)
    4_full.png                   — standard rasterizer output
"""
import os
import sys
import torch
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams, PipelineParams, get_combined_args
from train import merge_cfg_to_args


def main():
    parser = ArgumentParser(description="Decomposition debug render")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--view_idx", default=0, type=int)
    parser.add_argument("--out_subdir", default="debug", type=str)
    args = get_combined_args(parser)
    exp_path = args.model_path
    iteration = args.iteration
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(exp_path, "point_cloud"))

    pkl_path = os.path.join(exp_path, "args.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            train_args = pickle.load(f)
        for k, v in vars(train_args).items():
            if not hasattr(args, k) or getattr(args, k, None) is None:
                setattr(args, k, v)
        for k in ("method", "kernel", "kernel2", "feature", "hybrid_levels",
                  "disable_c2f", "aabb", "yaml", "lowpass", "texsplit",
                  "activation_bias", "lru"):
            if hasattr(train_args, k):
                setattr(args, k, getattr(train_args, k))

    print(f"[DBG] Model:      {exp_path}")
    print(f"[DBG] Iteration:  {iteration}")
    print(f"[DBG] Method:     {getattr(args, 'method', '?')}")
    print(f"[DBG] Feature:    {getattr(args, 'feature', '?')}")

    # Setter mirror.
    try:
        import diff_surfel_3D_sh_res as _ds_orig
        import diff_surfel_mixed_3d as _ds_m3d
        try:
            import diff_surfel_res_3d as _ds_r3d
        except ImportError:
            _ds_r3d = None
        _MIRRORED = ('set_mlp_weights', 'set_contrib_thresh', 'set_count_thresh',
                     'set_overdraw_lambda', 'set_weight_reg_lambda',
                     'set_activation_bias', 'set_residual_mode', 'set_anti_alias',
                     'set_compact_mult', 'set_aa_kernel_size', 'set_skip_mlp_grad',
                     'set_depth_sort', 'set_ste_relu', 'set_lru_slope')
        def _mk(o, m, m2):
            def w(*a, **k):
                o(*a, **k); m(*a, **k)
                if m2 is not None: m2(*a, **k)
            return w
        for n in _MIRRORED:
            if not hasattr(_ds_orig, n) or not hasattr(_ds_m3d, n):
                continue
            m2 = getattr(_ds_r3d, n, None) if _ds_r3d is not None else None
            setattr(_ds_orig, n, _mk(getattr(_ds_orig, n), getattr(_ds_m3d, n), m2))
    except ImportError as e:
        print(f"[DBG] WARNING: setter mirror install failed: {e}")

    yaml_file = getattr(args, "yaml", None) or "tiny"
    cfg_model = Config(yaml_file)
    merge_cfg_to_args(args, cfg_model)

    ingp = INGP(cfg_model, args=args).to("cuda")
    ingp.load_model(exp_path, iteration)
    ingp.set_active_levels(iteration)

    from diff_surfel_3D_sh_res import (set_activation_bias, set_residual_mode,
                                        set_compact_mult, set_lru_slope)
    _sh_bias, _res_bias = getattr(args, "activation_bias", [0.5, 0.0])
    _sh_bias = float(_sh_bias); _res_bias = float(_res_bias)
    set_activation_bias(sh_bias=_sh_bias, res_bias=_res_bias)
    _method = getattr(args, "method", None)
    if _method in ("res_3d", "res_switch", "mixed_sep", "mixed_3d_sep", "3D_SH_res_sep"):
        set_residual_mode(2); _residual_mode = 2
    else:
        set_residual_mode(0); _residual_mode = 0
    set_compact_mult(float(getattr(args, "fastgs_mult", 0.5)))
    _lru = float(getattr(args, "lru", 0.0))
    if _lru != 0.0:
        set_lru_slope(_lru)
    if ingp is not None:
        ingp.is_res_3d_mode = (_method == "res_3d")
        ingp.is_res_3d_post_split = (_method == "res_3d")
        ingp.is_mixed_deferred_relu_mode = (_residual_mode == 2)
        ingp.lru_slope = _lru
    print(f"[DBG] CUDA globals: sh_bias={_sh_bias} res_bias={_res_bias} "
          f"residual_mode={_residual_mode} lru={_lru}")

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.feature_mode = getattr(args, "feature", "sh")   # set BEFORE load_ply
    scene = Scene(dataset, gaussians, load_iteration=iteration,
                   shuffle=False, full_args=args)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.kernel_type = getattr(args, "kernel", "gaussian")
    gaussians.kernel_type2 = getattr(args, "kernel2", None)
    gaussians.feature_mode = getattr(args, "feature", "sh")
    gaussians._sv_training_flag = False
    print(f"[DBG] feature_mode = {gaussians.feature_mode}")

    N = gaussians.get_xyz.shape[0]
    has_split = (hasattr(gaussians, "_is_textured")
                  and gaussians._is_textured.numel() == N
                  and int((~gaussians._is_textured.bool()).sum().item()) > 0)
    if has_split:
        mask_t = gaussians._is_textured.bool()
        n_tex = int(mask_t.sum().item()); n_untex = N - n_tex
        print(f"[DBG] Surfels: {N:,} (tex {n_tex:,} / untex {n_untex:,}) — SPLIT model")
    else:
        mask_t = torch.ones(N, dtype=torch.bool, device=gaussians.get_xyz.device)
        print(f"[DBG] Surfels: {N:,} (all textured) — NO SPLIT")

    cams = scene.getTestCameras()
    if len(cams) == 0:
        raise RuntimeError("No test cameras.")
    view = cams[args.view_idx]
    print(f"[DBG] View: idx={args.view_idx} name={getattr(view, 'image_name', '?')}")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    beta_cfg = cfg_model.surfel.tg_beta

    out_dir = os.path.join(exp_path, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[DBG] Output dir: {out_dir}")

    def _save(name, img):
        arr = torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        save_img_u8(arr, os.path.join(out_dir, f"{name}.png"))
        print(f"[DBG] wrote {name}.png  (raw range {img.min().item():.3f}..{img.max().item():.3f})")

    orig_opacity = gaussians._opacity.detach().clone()
    def _restore_opacity():
        with torch.no_grad():
            gaussians._opacity.data.copy_(orig_opacity)

    def _render(decompose_mode=None):
        return render(view, gaussians, pipe, bg,
                       ingp=ingp, beta=beta_cfg, iteration=iteration,
                       cfg=cfg_model, skybox=None,
                       background_mode="none", bg_hashgrid=None,
                       decompose_mode=decompose_mode)

    with torch.no_grad():
        if not has_split:
            # res_switch / 3D_SH_res / pre-split: ALL surfels are textured-with-SV.
            # Decompose by zeroing MLP (→ SV-only) or killing SH (→ residual-only).
            img_sv  = _render(decompose_mode='sh_only')["render"].detach()
            _save("1_sv_only", img_sv)
            img_res = _render(decompose_mode='tex_only')["render"].detach()
            _save("2_residual_only", img_res)
            img_full = _render()["render"].detach()
            _save("3_full", img_full)
        else:
            # res_3d post-split: tex carriers (SV=0, residual only) + untex EWA SV.
            # 1) tex carriers full contribution (residual + sh_bias baseline)
            gaussians._opacity.data[~mask_t] = -1e10
            img_tex = _render()["render"].detach()
            _save("1_tex_carriers_full", img_tex)
            # 2) tex carriers WITHOUT residual (decompose='sh_only') → just the
            #    bias baseline (SV is zeroed on tex rows)
            img_tex_svonly = _render(decompose_mode='sh_only')["render"].detach()
            _save("2_tex_carriers_sv_baseline_only", img_tex_svonly)
            _restore_opacity()
            # 3) untex carriers full contribution (= their SV; no residual path)
            gaussians._opacity.data[mask_t] = -1e10
            img_untex = _render()["render"].detach()
            _save("3_untex_carriers_full", img_untex)
            _restore_opacity()
            # 4) standard full single-pass render
            img_full = _render()["render"].detach()
            _save("4_full", img_full)

    print("[DBG] Done.")


if __name__ == "__main__":
    main()
