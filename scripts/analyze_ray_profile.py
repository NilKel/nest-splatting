"""Per-ray depth-structure analysis: cluster statistics + T-crossing ladder.

Uses the diff_surfel_3D_sh_res_trunc clone's analysis device-globals (no
truncation active; exit_T stays 1e-4):

  CLUSTER STATS (set_cluster_mode, median slot repurposed):
    mode 1 -> per-pixel count of DISJOINT surfel clusters along the ray
              (1D single-linkage: new cluster when the next fragment with
              alpha >= alpha_min lies more than `gap` behind the current
              cluster's deepest member)
    mode 2 -> depth gap between cluster 1 and cluster 2 (first-void size;
              0 if fewer than 2 clusters)
    Run for a SWEEP of gap thresholds -> cluster-count vs merge-distance
    curve (the 1D dendrogram summary).

  T-LADDER (set_T_crossing): median slot = depth(T=tau) for a ladder of tau
    -> falloff thickness d(T=tau_lo)-d(T=tau_hi) and inter-level jumps
    (energy-weighted view of the same structure).

Statistics aggregate over ALL selected views; example heatmap images are
written only for --example_views (default "0,100").

Usage:
  python scripts/analyze_ray_profile.py --model_path <run_dir> \
      [--iteration N] [--split train] [--max_views -1] \
      [--cluster_gaps 0.02,0.05,0.10] [--cluster_alpha_min 0.05] \
      [--taus 0.9,0.7,0.5,0.3,0.1,0.05] [--example_views 0,100] \
      --out_dir <dir>
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import imageio.v2 as imageio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from speed_comparison.render_all_views_cull import load_model, depth_to_viridis, to_u8
from gaussian_renderer import render
import diff_surfel_3D_sh_res_trunc as TR

MAX_CL_BIN = 6   # cluster-count histogram bins: 1..5, 6+


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--max_views", type=int, default=-1, help="-1 = all views")
    p.add_argument("--example_views", default="0,100",
                   help="view indices that get heatmap images (stats use all)")
    p.add_argument("--cluster_gaps", default="0.02,0.05,0.10",
                   help="gap thresholds (scene units) for the cluster sweep; "
                        "gap12 maps/stats use the FIRST value")
    p.add_argument("--cluster_alpha_min", type=float, default=0.05)
    p.add_argument("--taus", default="0.9,0.7,0.5,0.3,0.1,0.05")
    p.add_argument("--no_ladder", action="store_true")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    gaps = [float(g) for g in args.cluster_gaps.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    assert all(taus[i] > taus[i + 1] for i in range(len(taus) - 1))
    example_views = {int(v) for v in args.example_views.split(",") if v != ""}

    if args.iteration < 0:
        import glob
        ngps = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        args.iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                             for f in ngps)

    train_args, cfg, ingp, gaussians, scene, pipe, beta_kern = \
        load_model(args.model_path, args.iteration)

    # Mirror device-global installs into the trunc clone (module-local globals).
    ab = getattr(train_args, "activation_bias", [0.5, 0.0])
    TR.set_activation_bias(sh_bias=float(ab[0]), res_bias=float(ab[1]))
    TR.set_residual_mode(int(getattr(train_args, "_residual_mode", 0)))
    TR.set_exit_T(1e-4)
    TR.set_cluster_mode(0)
    ingp.is_trunc_mode = True

    cams = list(scene.getTrainCameras() if args.split == "train"
                else scene.getTestCameras())
    if args.max_views > 0:
        cams = cams[:args.max_views]
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[profile] {len(cams)} {args.split} views · gaps={gaps} · "
          f"alpha_min={args.cluster_alpha_min} · taus={taus} · "
          f"examples={sorted(example_views)}")

    def _render():
        with torch.no_grad():
            return render(cams[idx], gaussians, pipe, bg, beta=beta_kern,
                          iteration=args.iteration, cfg=cfg, ingp=ingp,
                          is_training=False, lowpass=True)

    # accumulators
    cl_hist = {g: torch.zeros(MAX_CL_BIN + 1, dtype=torch.long) for g in gaps}
    frac2_sum = {g: 0.0 for g in gaps}
    gap12_all = []
    thickness_all, ladder_gap_all, ladder_lvl_all = [], [], []
    per_view = []
    nan = torch.tensor(float('nan'), device='cuda')

    for idx in range(len(cams)):
        cam = cams[idx]
        is_example = idx in example_views
        row = {"view": cam.image_name, "idx": idx}

        # ---------- cluster sweep ----------
        alpha = None
        ncl_maps = {}
        for g in gaps:
            TR.set_cluster_mode(1, args.cluster_alpha_min, g)
            pkg = _render()
            ncl = pkg["depth_median"].squeeze().round().long()
            if alpha is None:
                alpha = pkg["rend_alpha"].squeeze().clone()
            fg = alpha > 0.5
            ncl_maps[g] = ncl
            vals = ncl[fg].clamp(0, MAX_CL_BIN)
            cl_hist[g] += torch.bincount(vals.cpu(), minlength=MAX_CL_BIN + 1)
            frac2 = float((ncl[fg] >= 2).float().mean()) if fg.any() else 0.0
            frac2_sum[g] += frac2
            row[f"frac>=2cl@gap{g}"] = frac2

        # gap12 at the primary gap threshold
        TR.set_cluster_mode(2, args.cluster_alpha_min, gaps[0])
        pkg = _render()
        gap12 = pkg["depth_median"].squeeze()
        TR.set_cluster_mode(0)
        fg = alpha > 0.5
        g12 = gap12[fg & (gap12 > 0)]
        gap12_all.append(g12.cpu())
        row["gap12_mean"] = float(g12.mean()) if g12.numel() else 0.0
        row["gap12_p90"] = float(g12.quantile(0.9)) if g12.numel() else 0.0

        # ---------- T ladder ----------
        if not args.no_ladder:
            depths, valids = [], []
            for tau in taus:
                TR.set_T_crossing(tau)
                lpkg = _render()
                depths.append(lpkg["depth_median"].squeeze().clone())
                valids.append(alpha > (1.0 - tau))
            TR.set_T_crossing(0.5)
            D = torch.stack(depths)
            V = torch.stack(valids)
            v_all = V.all(dim=0)
            thickness = (D[-1] - D[0]).where(v_all, nan)
            jumps = (D[1:] - D[:-1]).where(V[1:] & V[:-1], nan)
            lj, lv = torch.nan_to_num(jumps, nan=-1.0).max(dim=0)
            lj = lj.where(lj >= 0, nan)
            th = thickness[~thickness.isnan()]
            l_ = lj[~lj.isnan()]
            thickness_all.append(th.cpu())
            ladder_gap_all.append(l_.cpu())
            ladder_lvl_all.append(lv[~lj.isnan()].cpu())
            row["thickness_mean"] = float(th.mean()) if th.numel() else 0.0
            row["ladder_gap_p90"] = float(l_.quantile(0.9)) if l_.numel() else 0.0

        # ---------- example heatmaps ----------
        if is_example:
            imageio.imwrite(str(out / f"{idx:03d}_color.png"),
                            to_u8(pkg["render"].clamp(0, 1)))
            for g in gaps:
                nm = ncl_maps[g].float().clamp(0, MAX_CL_BIN) / MAX_CL_BIN
                imageio.imwrite(str(out / f"{idx:03d}_nclusters_gap{g}.png"),
                                depth_to_viridis(nm * 4.0))  # spread the colormap
            imageio.imwrite(str(out / f"{idx:03d}_gap12.png"),
                            depth_to_viridis(gap12 * (gap12 > 0)))
            if not args.no_ladder:
                imageio.imwrite(str(out / f"{idx:03d}_thickness.png"),
                                depth_to_viridis(torch.nan_to_num(thickness, nan=0.0)))
                imageio.imwrite(str(out / f"{idx:03d}_ladder_gap.png"),
                                depth_to_viridis(torch.nan_to_num(lj, nan=0.0)))
                for li, tau in enumerate(taus):
                    dm = D[li].where(V[li], torch.tensor(0.0, device='cuda'))
                    imageio.imwrite(str(out / f"{idx:03d}_depth_T{tau:.2f}.png"),
                                    depth_to_viridis(dm))

        per_view.append(row)
        if idx % 20 == 0 or is_example:
            print(f"  [{idx:03d}/{len(cams)}] {cam.image_name:<26s} "
                  f"frac>=2cl@{gaps[0]}={row[f'frac>=2cl@gap{gaps[0]}']:.3f} "
                  f"gap12_p90={row['gap12_p90']:.4f}"
                  + (" [example imgs]" if is_example else ""))

    # ---------- aggregates ----------
    g12 = torch.cat(gap12_all) if gap12_all else torch.empty(0)
    fig, axes = plt.subplots(1, 3 if args.no_ladder else 4,
                             figsize=(5 * (3 if args.no_ladder else 4), 4))
    # cluster count distribution per gap threshold
    w = 0.8 / len(gaps)
    for gi, g in enumerate(gaps):
        h = cl_hist[g].float()
        h = h / h.sum().clamp_min(1)
        axes[0].bar(np.arange(MAX_CL_BIN + 1) + gi * w, h.numpy(), width=w,
                    label=f"gap={g}")
    axes[0].set_xticks(range(MAX_CL_BIN + 1),
                       [str(i) for i in range(MAX_CL_BIN)] + [f"{MAX_CL_BIN}+"])
    axes[0].set_title("clusters per foreground pixel")
    axes[0].legend()
    if g12.numel():
        axes[1].hist(g12.numpy(), bins=120)
        axes[1].set_yscale('log')
        axes[1].set_title(f"cluster1->2 distance (gap={gaps[0]})")
    axes[2].bar(range(len(gaps)),
                [frac2_sum[g] / max(1, len(cams)) for g in gaps])
    axes[2].set_xticks(range(len(gaps)), [str(g) for g in gaps])
    axes[2].set_title("mean frac pixels with >=2 clusters")
    if not args.no_ladder and thickness_all:
        th = torch.cat(thickness_all)
        axes[3].hist(th.numpy(), bins=120)
        axes[3].set_yscale('log')
        axes[3].set_title(f"falloff thickness d(T={taus[-1]})-d(T={taus[0]})")
    fig.tight_layout()
    fig.savefig(str(out / "aggregate_histograms.png"), dpi=110)

    agg = {
        "n_views": len(cams), "gaps": gaps, "alpha_min": args.cluster_alpha_min,
        "taus": taus,
        "cluster_hist": {str(g): cl_hist[g].tolist() for g in gaps},
        "mean_frac>=2clusters": {str(g): frac2_sum[g] / max(1, len(cams)) for g in gaps},
        "gap12_mean": float(g12.mean()) if g12.numel() else 0.0,
        "gap12_median": float(g12.median()) if g12.numel() else 0.0,
        "gap12_p90": float(g12.quantile(0.9)) if g12.numel() else 0.0,
    }
    if not args.no_ladder and thickness_all:
        th = torch.cat(thickness_all)
        lg = torch.cat(ladder_gap_all)
        agg["thickness_mean"] = float(th.mean()) if th.numel() else 0.0
        agg["thickness_p90"] = float(th.quantile(0.9)) if th.numel() else 0.0
        agg["ladder_gap_p90"] = float(lg.quantile(0.9)) if lg.numel() else 0.0
    with open(out / "profile_stats.json", "w") as f:
        json.dump({"aggregate": agg, "views": per_view}, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    print(f"[analyze_ray_profile] wrote {out}/")


if __name__ == "__main__":
    main()
