"""FastGS multi-view consistency scoring (Phase 1 — Python-only).

Reference: FastGS (arXiv 2511.04283). Paper says K=10 views, λ=0.2 dssim weight,
densification every 500 iters until 15k, pruning every 500 before 15k and every
3000 after 15k.

Phase-1 proxy: FastGS's real CUDA kernel atomicAdd's per-Gaussian for every
contributor at a high-error pixel (all Gaussians with alpha > 1/255). We
approximate it with our existing ``max_contrib_idx`` map, which records only
the single dominant Gaussian per pixel. This strictly undercounts, so the
FastGS threshold ``importance_score > 5`` does not carry over — we expose a
knob ``fastgs_importance_thresh`` that defaults to ``1`` (any high-error pixel
at which this Gaussian dominates). Phase 2 would add a CUDA counter to get
exact FastGS semantics.
"""
import random

import torch

from utils.loss_utils import l1_loss, ssim


def sampling_cameras(viewpoint_stack, num_cams=10):
    """Pop ``num_cams`` random cameras off a mutable list copy."""
    cams = []
    for _ in range(min(num_cams, len(viewpoint_stack))):
        loc = random.randint(0, len(viewpoint_stack) - 1)
        cams.append(viewpoint_stack.pop(loc))
    return cams


def _per_pixel_l1_normalized(rendered, gt):
    """[3,H,W] → [H,W] L1, normalized to [0,1] across pixels."""
    l1 = (rendered - gt).abs().mean(dim=0).detach()
    lo, hi = l1.min(), l1.max()
    denom = (hi - lo).clamp_min(1e-8)
    return (l1 - lo) / denom


def _photometric_loss(rendered, gt, lambda_dssim=0.2):
    Ll1 = l1_loss(rendered, gt)
    Lssim = 1.0 - ssim(rendered.unsqueeze(0), gt.unsqueeze(0))
    return ((1.0 - lambda_dssim) * Ll1 + lambda_dssim * Lssim).detach()


def compute_gaussian_score_fastgs(camlist, gaussians, render_fn,
                                  loss_thresh=0.1, lambda_dssim=0.2,
                                  densify=False):
    """Per-Gaussian multi-view consistency scores (Phase-1 proxy).

    Args:
        camlist: list of viewpoint cameras.
        gaussians: GaussianModel.
        render_fn: callable(cam) -> render_pkg with keys 'render' and
                   'max_contrib_idx' (our rasterizer exposes the latter from
                   out_index).
        loss_thresh: per-pixel L1 threshold (normalized to [0,1] per view)
                     above which a pixel is flagged high-error.
        lambda_dssim: SSIM weight in photometric loss (paper: 0.2).
        densify: if True, also return ``importance_score`` for densification.

    Returns:
        (importance_score, pruning_score): each [N] on CUDA. importance_score
        is floor-averaged contributor count (None if not densify);
        pruning_score is min-max normalized (photometric_loss * count) sum.
    """
    N = gaussians.get_xyz.shape[0]
    full_metric_counts = torch.zeros(N, dtype=torch.float32, device="cuda")
    full_metric_score = torch.zeros(N, dtype=torch.float32, device="cuda")

    for cam in camlist:
        with torch.no_grad():
            pkg = render_fn(cam)
        rendered = pkg["render"].clamp(0.0, 1.0)
        gt = cam.original_image.cuda()

        # Per-pixel L1, normalized per-view; threshold into a binary mask.
        l1_norm = _per_pixel_l1_normalized(rendered, gt)         # [H, W]
        metric_map = (l1_norm > loss_thresh)                      # [H, W] bool

        photometric_loss = _photometric_loss(rendered, gt, lambda_dssim)

        # Proxy: count high-error pixels where THIS Gaussian is the max
        # contributor. FastGS counts all contributors; we only count winners.
        max_idx = pkg.get("max_contrib_idx", None)
        if max_idx is None or max_idx.numel() == 0:
            raise RuntimeError(
                "--fastgs requires a rasterizer that exposes max_contrib_idx "
                "(currently only diff_surfel_3D_sh_res / --method 3D_SH_res)."
            )
        flat_idx = max_idx.long().reshape(-1)
        flat_mask = metric_map.reshape(-1)
        valid = (flat_idx >= 0) & flat_mask
        winners = flat_idx[valid]
        counts_view = torch.zeros(N, dtype=torch.float32, device="cuda")
        if winners.numel() > 0:
            counts_view.scatter_add_(
                0, winners,
                torch.ones_like(winners, dtype=torch.float32))

        if densify:
            full_metric_counts += counts_view
        full_metric_score += float(photometric_loss.item()) * counts_view

    lo, hi = full_metric_score.min(), full_metric_score.max()
    pruning_score = (full_metric_score - lo) / (hi - lo).clamp_min(1e-8)

    importance_score = None
    if densify:
        # floor((sum_views count) / K)  — average contributor count per view.
        importance_score = torch.div(
            full_metric_counts, len(camlist), rounding_mode="floor")

    return importance_score, pruning_score
