#!/usr/bin/env python3
"""Compile the 9-scene overdraw comparison figure: nest neural vs FastGS.

Reads per-scene intersection outputs already produced by
`render_intersection_all.py` (nest) and `intersection_maps.py` (FastGS),
then writes:

  <out>/comparison_grid.png
      9-scene grid: view 0 heatmap (nest) | view 0 heatmap (FastGS) |
      per-view mean contributor histogram (both overlaid).

  <out>/stats_table.md
      Per-scene N Gauss, resolution, global mean / max / p95, and the
      nest-vs-FastGS ratios.

Usage:
    conda run -n nest_splatting python speed_comparison/build_intersection_comparison.py \\
        --nest_root speed_comparison/mip360_nest_intersection \\
        --fastgs_root /home/nilkel/Projects/FastGS/output \\
        --out speed_comparison/intersection_comparison
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden",
          "kitchen", "room", "stump", "treehill"]


def load_nest(root: Path, scene: str):
    d = root / scene
    if not (d / "intersection_summary.json").exists():
        return None
    summary = json.load(open(d / "intersection_summary.json"))
    view0 = np.load(d / "intersection_raw" / "00000.npy")
    view0_png = d / "intersection_maps" / "00000.png"
    return {"summary": summary, "view0_raw": view0, "view0_png": view0_png}


def load_fastgs(root: Path, scene: str):
    # FastGS writes into <model>/test/ours_30000/intersection_...
    d = root / scene / "test" / "ours_30000"
    if not (d / "intersection_summary.json").exists():
        return None
    summary = json.load(open(d / "intersection_summary.json"))
    view0 = np.load(d / "intersection_raw" / "00000.npy")
    view0_png = d / "intersection_maps" / "00000.png"
    return {"summary": summary, "view0_raw": view0, "view0_png": view0_png}


def per_view_means(summary):
    return [v["mean"] for v in summary["per_view"]]


def per_view_p99(summary):
    return [v["p99"] for v in summary["per_view"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nest_root", type=Path,
                    default=Path("/home/nilkel/Projects/nest-splatting/speed_comparison/mip360_nest_intersection"))
    ap.add_argument("--fastgs_root", type=Path,
                    default=Path("/home/nilkel/Projects/FastGS/output"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/nilkel/Projects/nest-splatting/speed_comparison/intersection_comparison"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = []
    for s in SCENES:
        n = load_nest(args.nest_root, s)
        f = load_fastgs(args.fastgs_root, s)
        if n is None or f is None:
            print(f"[skip] {s}: nest={n is not None} fastgs={f is not None}")
            continue
        scenes.append((s, n, f))
    if not scenes:
        raise SystemExit("no scenes with both nest + fastgs data")

    # ----------------------------------------------------------------
    # Figure: 9 rows × 3 cols. Each row = nest view0 map, fastgs view0
    # map, per-view mean-contributor histogram (both overlaid).
    # ----------------------------------------------------------------
    n_rows = len(scenes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    if n_rows == 1:
        axes = axes[None, :]

    for i, (s, n, f) in enumerate(scenes):
        ax_n, ax_f, ax_h = axes[i]

        # Nest view 0 heatmap PNG.
        img_n = Image.open(n["view0_png"])
        ax_n.imshow(np.asarray(img_n))
        ax_n.set_title(f"nest neural · view 0", fontsize=10)
        ax_n.axis("off")

        # FastGS view 0 heatmap PNG.
        img_f = Image.open(f["view0_png"])
        ax_f.imshow(np.asarray(img_f))
        ax_f.set_title(f"FastGS · view 0", fontsize=10)
        ax_f.axis("off")

        # Per-view mean-contributor histogram, both overlaid.
        means_n = per_view_means(n["summary"])
        means_f = per_view_means(f["summary"])
        bins = np.linspace(0, max(max(means_n), max(means_f)) * 1.05, 24)
        ax_h.hist(means_n, bins=bins, alpha=0.65, color="#3b6db8",
                  label=f"nest  μ={np.mean(means_n):.1f}", edgecolor="black", linewidth=0.4)
        ax_h.hist(means_f, bins=bins, alpha=0.65, color="#c94a4a",
                  label=f"FastGS μ={np.mean(means_f):.1f}",  edgecolor="black", linewidth=0.4)
        ax_h.set_xlabel("mean contributors / pixel", fontsize=9)
        ax_h.set_ylabel("test views", fontsize=9)
        ax_h.legend(fontsize=8, loc="upper right", frameon=False)
        ax_h.tick_params(labelsize=8)

        # Left annotation: scene name + counts.
        n_res = n["summary"]["resolution"]
        n_ng = n["summary"]["num_points"]
        f_ng = f["summary"]["num_points"]
        ax_n.text(-0.06, 0.5, f"{s}\n{n_res}\nnest {n_ng//1000}k\nFastGS {f_ng//1000}k",
                  transform=ax_n.transAxes, rotation=90, va="center", ha="right",
                  fontsize=10, family="monospace")

    plt.suptitle("Per-pixel overdraw (intersection counts) — nest neural vs FastGS on mip-360",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])
    grid_path = args.out / "comparison_grid.png"
    plt.savefig(grid_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[write] {grid_path}")

    # ----------------------------------------------------------------
    # Stats table markdown.
    # ----------------------------------------------------------------
    lines = ["# Overdraw comparison — nest neural vs FastGS (mip-360, 9 scenes)",
             "",
             "Values are aggregated across all test views of each scene."
             " `mean`/`p95`/`max` are the global summary values reported by"
             " `intersection_summary.json` (per-pixel contributor count).",
             "",
             "| scene | resolution | N Gauss (nest) | N Gauss (FastGS) | ratio | nest mean | FastGS mean | ratio | nest p95 avg | FastGS p95 avg | nest global max | FastGS global max |",
             "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, n, f in scenes:
        nS = n["summary"]; fS = f["summary"]
        n_p95 = np.mean([v["p95"] for v in nS["per_view"]])
        f_p95 = np.mean([v["p95"] for v in fS["per_view"]])
        lines.append(
            f"| {s} | {nS['resolution']} | {nS['num_points']:,} | {fS['num_points']:,} "
            f"| {fS['num_points'] / nS['num_points']:.2f}× "
            f"| {nS['global_mean']:.2f} | {fS['global_mean']:.2f} "
            f"| {fS['global_mean'] / nS['global_mean']:.2f}× "
            f"| {n_p95:.1f} | {f_p95:.1f} "
            f"| {int(nS['global_max'])} | {int(fS['global_max'])} |"
        )
    # Aggregate row.
    n_means = [n["summary"]["global_mean"] for _, n, _ in scenes]
    f_means = [f["summary"]["global_mean"] for _, _, f in scenes]
    n_ng    = [n["summary"]["num_points"]  for _, n, _ in scenes]
    f_ng    = [f["summary"]["num_points"]  for _, _, f in scenes]
    n_p95s  = [np.mean([v["p95"] for v in n["summary"]["per_view"]]) for _, n, _ in scenes]
    f_p95s  = [np.mean([v["p95"] for v in f["summary"]["per_view"]]) for _, _, f in scenes]
    lines.append(
        f"| **mean** | — "
        f"| **{int(np.mean(n_ng)):,}** | **{int(np.mean(f_ng)):,}** "
        f"| **{np.mean(f_ng) / np.mean(n_ng):.2f}×** "
        f"| **{np.mean(n_means):.2f}** | **{np.mean(f_means):.2f}** "
        f"| **{np.mean(f_means) / np.mean(n_means):.2f}×** "
        f"| **{np.mean(n_p95s):.1f}** | **{np.mean(f_p95s):.1f}** "
        f"| — | — |"
    )
    lines.append("")
    lines.append(
        f"**Read:** nest neural averages **{np.mean(n_means):.1f}** contributors/pixel across the 9 scenes, "
        f"vs FastGS's **{np.mean(f_means):.1f}** — FastGS renders roughly "
        f"**{np.mean(f_means) / np.mean(n_means):.1f}×** more overdraw despite carrying "
        f"**{np.mean(f_ng) / np.mean(n_ng):.1f}×** more Gauss."
        f" The two effects mostly cancel wall-clock — nest bakes carry fewer Gauss but each surfel"
        f" is textured and touches the atlas fetch, while FastGS's smaller-primitive per-pixel cost is amortized"
        f" over many more contributions per pixel.")
    lines.append("")
    lines.append("See `comparison_grid.png` for view-0 side-by-side heatmaps and"
                 " per-view mean-contributor histograms.")
    table_path = args.out / "stats_table.md"
    table_path.write_text("\n".join(lines))
    print(f"[write] {table_path}")


if __name__ == "__main__":
    main()
