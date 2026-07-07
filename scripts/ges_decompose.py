"""GEStex sort-free DECOMPOSITION diagnostic.

Loads a trained GEStex checkpoint and, for a few test views, dumps the render
COMPONENTS side by side so we can verify (independent of final quality):
  1. Decomposition works      — SV-base vs atlas-residual are actually different.
  2. Textures work            — the baked atlas contributes (atlas-only is not black,
                                full != SV-only).
  3. Tex/untex separation     — textured surfels (C_S) vs spawned untextured 3D
                                Gaussians (C_G) render as separate, sensible things.

Montage columns (per test view, one row each):
  [ GT | full | C_S = surfels(SV+atlas) | SV-only | atlas-only(norm) | C_G = untex Gauss | W_G | surfel-cov ]

Calls gaussian_renderer._render_gestex_joint directly (no ingp/pipe needed).

Usage:
  conda run -n nest_splatting python scripts/ges_decompose.py \
      --ckpt outputs/nerf_synthetic/chair/GEStex/chair_GEStex_35k2 [--iter 35000] [--n 4]
"""
import os, sys, pickle, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from scene.gaussian_model import GaussianModel
import gaussian_renderer as GR
from gaussian_renderer import _render_gestex_joint


def _norm01(x):
    """Per-image min-max normalize to [0,1] for visualizing signed maps."""
    x = x.detach().float()
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint model dir')
    ap.add_argument('--iter', type=int, default=35000)
    ap.add_argument('--n', type=int, default=4, help='number of test views to dump')
    ap.add_argument('--lru', type=float, default=0.01)
    args = ap.parse_args()
    ck = os.path.abspath(args.ckpt)

    # ---- reconstruct the training ModelParams from args.pkl ----
    margs = pickle.load(open(os.path.join(ck, 'args.pkl'), 'rb'))
    margs.model_path = ck                      # point Scene at this checkpoint
    print(f"[decompose] source={margs.source_path}  white_bg={getattr(margs,'white_background',True)}  "
          f"feature={getattr(margs,'feature','?')}")

    # ---- load gaussians + cameras. feature_mode must be 'SV' BEFORE load_ply. ----
    gaussians = GaussianModel(margs.sh_degree)
    gaussians.feature_mode = 'SV'
    scene = Scene(margs, gaussians, load_iteration=args.iter, shuffle=False, full_args=margs)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    # gestex runtime state (not saved in the PLY)
    gaussians.is_gestex = True
    gaussians.surfel_opac = 255.0
    if not hasattr(gaussians, 'ges_s_weight'):
        gaussians.ges_s_weight = 1.0

    sm = gaussians._is_textured
    n_tex = int(sm.sum().item()); n_untex = int((~sm).sum().item())
    at = gaussians._tex_atlas[sm]
    print(f"[decompose] loaded: {gaussians.get_xyz.shape[0]} prims  "
          f"({n_tex} textured surfels / {n_untex} untextured Gaussians)  "
          f"atlas R={gaussians.ges_atlas_res}")
    print(f"[decompose] atlas stats (surfel rows): mean={at.mean():.4f} std={at.std():.4f} "
          f"min={at.min():.4f} max={at.max():.4f} |v|>0.01={(at.abs()>0.01).float().mean()*100:.1f}%")

    white = bool(getattr(margs, 'white_background', True))
    bg = torch.tensor([1., 1., 1.] if white else [0., 0., 0.], device='cuda')

    cams = scene.getTestCameras() or scene.getTrainCameras()
    cams = cams[:args.n]
    outdir = os.path.join(ck, 'ges_decompose'); os.makedirs(outdir, exist_ok=True)
    from torchvision.utils import save_image

    for i, cam in enumerate(cams):
        with torch.no_grad():
            full = _render_gestex_joint(cam, gaussians, bg, lru_slope=args.lru, decompose_mode=None)
            svp  = _render_gestex_joint(cam, gaussians, bg, lru_slope=args.lru, decompose_mode='sh_only')
            texp = _render_gestex_joint(cam, gaussians, bg, lru_slope=args.lru, decompose_mode='tex_only')

        gt      = cam.original_image.cuda().clamp(0, 1)
        full_im = full['render'].clamp(0, 1)
        C_S     = full['surfel_render'].clamp(0, 1)                 # surfels: SV + atlas
        sv_only = svp['surfel_render'].clamp(0, 1)                  # surfels: SV base
        tex_raw = texp['surfel_render']                            # surfels: atlas residual (signed)
        tex_vis = _norm01(tex_raw)
        C_G     = full['gaussian_render'].clamp(0, 1)              # untextured 3D Gaussians
        W_G     = full['gaussian_weight'].clamp(0, 1).repeat(3, 1, 1)
        cov     = full['surfel_coverage'].clamp(0, 1).repeat(3, 1, 1)

        row = torch.cat([gt, full_im, C_S, sv_only, tex_vis, C_G, W_G, cov], dim=2)  # concat width
        save_image(row, os.path.join(outdir, f'view{i:02d}_decompose.png'))
        # individual components (same output folder) for zoom-in inspection
        save_image(gt,      os.path.join(outdir, f'view{i:02d}_0_gt.png'))
        save_image(full_im, os.path.join(outdir, f'view{i:02d}_1_full.png'))
        save_image(C_S,     os.path.join(outdir, f'view{i:02d}_2_surfels_SVplusAtlas.png'))
        save_image(sv_only, os.path.join(outdir, f'view{i:02d}_3_SV_only.png'))
        save_image(tex_vis, os.path.join(outdir, f'view{i:02d}_4_atlas_only_norm.png'))
        save_image(C_G,     os.path.join(outdir, f'view{i:02d}_5_untex_gaussians.png'))
        save_image(W_G,     os.path.join(outdir, f'view{i:02d}_6_gaussian_weight.png'))
        save_image(cov,     os.path.join(outdir, f'view{i:02d}_7_surfel_cov.png'))

        # Labelled contact sheet (titles burned in) for the FIRST view only.
        if i == 0:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            panels = [
                ('GT', gt), ('FULL render', full_im),
                ('surfels: SV + atlas (C_S)', C_S), ('surfels: SV base only', sv_only),
                ('surfels: atlas residual only (norm)', tex_vis),
                ('untextured 3D Gaussians (C_G)', C_G),
                ('Gaussian weight W_G', W_G), ('surfel coverage', cov),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            for ax, (title, im) in zip(axes.ravel(), panels):
                ax.imshow(im.permute(1, 2, 0).cpu().numpy())
                ax.set_title(title, fontsize=15); ax.axis('off')
            fig.suptitle(f'GEStex decomposition — view {i}  '
                         f'({n_tex} textured surfels / {n_untex} untex Gaussians, atlas R={gaussians.ges_atlas_res})',
                         fontsize=17)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(outdir, f'view{i:02d}_LABELLED.png'), dpi=90)
            plt.close(fig)
            print(f"[decompose] labelled panel → {outdir}/view{i:02d}_LABELLED.png")

        # quantitative confirmation of each question
        atlas_contrib = (C_S - sv_only).abs().mean().item()       # how much the atlas moves C_S
        tex_energy    = tex_raw.abs().mean().item()               # atlas-only magnitude
        cg_energy     = C_G.abs().mean().item()                   # untex-Gaussian magnitude
        cg_cover      = (full['gaussian_weight'] > 0.01).float().mean().item() * 100
        cs_cover      = full['surfel_coverage'].mean().item() * 100
        print(f"[view {i:02d}] atlas_contrib|C_S-SV|={atlas_contrib:.4f}  atlas_only|res|={tex_energy:.4f}  "
              f"C_G|rgb|={cg_energy:.4f}  surfel_cov={cs_cover:.1f}%  gauss_cov={cg_cover:.1f}%")

    print(f"\n[decompose] wrote {len(cams)} montages → {outdir}")
    print("  columns: [GT | full | surfels(SV+atlas) | SV-only | atlas-only(norm) | untex-Gaussians | W_G | surfel-cov]")


if __name__ == '__main__':
    main()
