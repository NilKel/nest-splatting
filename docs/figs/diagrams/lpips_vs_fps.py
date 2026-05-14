#!/usr/bin/env python3
"""LPIPS vs FPS scatter plot for the paper's quantitative table.

Each method gets its own marker shape; the method name is printed to the
right of the marker (no separate legend). Two-word names break across
two lines.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# (label, marker, color, lpips_tnt, fps_tnt, lpips_m360, fps_m360, role)
# role: "ours"  -> red, large; "fastgs" -> purple; others -> dark slate.
DATA = [
    ("3DGS",            "o", "#2F3A4A", 0.169, 154,  0.214, 134),
    ("2DGS",            "s", "#2F3A4A", 0.212, 122,  0.252, 64),
    ("DBS",             "^", "#2F3A4A", 0.144, 150,  0.192, 123),
    ("BBSplat",         "p", "#2F3A4A", 0.147, 66,   0.231, 25),
    ("Nexels",          "h", "#2F3A4A", 0.155, 70,   0.201, 50),
    ("FastGS",          "X", "#8172B2", 0.210, 1173, 0.261, 942),
    ("Ours",            "*", "#C44E52", 0.157, 648,  0.244, 319),
    ("Ours\n(overdraw)","P", "#C44E52", 0.164, 1094, 0.253, 664),
]

# Per-method, per-dataset annotation offsets (in points) and alignment.
# Default = (8, 0, "left", "center")
OFFSETS_M360 = {
    "3DGS":            (8,  3, "left",  "bottom"),
    "2DGS":            (8,  0, "left",  "center"),
    "DBS":             (8, -4, "left",  "top"),
    "BBSplat":         (8,  0, "left",  "center"),
    "Nexels":          (8, -2, "left",  "center"),
    "FastGS":          (-8, 0, "right", "center"),
    "Ours":            (8,  0, "left",  "center"),
    "Ours\n(overdraw)":(-8, 0, "right", "center"),
}
OFFSETS_TNT = {
    "3DGS":            (8,  4, "left",  "bottom"),
    "2DGS":            (8,  0, "left",  "center"),
    "DBS":             (8, -4, "left",  "top"),
    "BBSplat":         (8,  0, "left",  "center"),
    "Nexels":          (8,  0, "left",  "center"),
    "FastGS":          (-8, 0, "right", "center"),
    "Ours":            (8,  0, "left",  "center"),
    "Ours\n(overdraw)":(-8, 0, "right", "center"),
}


def plot_subplot(ax, *, lpips_key, fps_key, title, offsets):
    for name, marker, color, ltnt, ftnt, l360, f360 in DATA:
        lpips = {"tnt": ltnt, "m360": l360}[lpips_key]
        fps   = {"tnt": ftnt, "m360": f360}[fps_key]

        is_ours = color == "#C44E52"
        is_fastgs = color == "#8172B2"
        size = 220 if is_ours else (130 if is_fastgs else 90)
        edge = "black"
        linewidth = 0.7

        ax.scatter(fps, lpips, marker=marker, c=color, s=size,
                   edgecolors=edge, linewidths=linewidth, zorder=3)

        dx, dy, ha, va = offsets[name]
        weight = "bold" if is_ours or is_fastgs else "normal"
        ax.annotate(name, (fps, lpips), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8, ha=ha, va=va, weight=weight, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("FPS (log scale, RTX 4090)  →")
    ax.set_ylabel("LPIPS  ←")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)

plot_subplot(axes[0], lpips_key="m360", fps_key="m360",
             title="Mip-NeRF 360", offsets=OFFSETS_M360)
plot_subplot(axes[1], lpips_key="tnt", fps_key="tnt",
             title="Tanks and Temples", offsets=OFFSETS_TNT)

# Pad x-axis so right-anchored labels (FastGS, Ours (overdraw)) don't clip.
for ax in axes:
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo * 0.7, xhi * 1.6)
    ylo, yhi = ax.get_ylim()
    pad = 0.012
    ax.set_ylim(ylo - pad, yhi + pad)

plt.tight_layout()
out_pdf = "/home/nilkel/Projects/nest-splatting/docs/figs/diagrams/lpips_vs_fps.pdf"
out_png = "/home/nilkel/Projects/nest-splatting/docs/figs/diagrams/lpips_vs_fps.png"
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
