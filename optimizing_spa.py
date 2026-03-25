import torch
import numpy as np


class OptimizingSpa:
    """ADMM-based sparsification for Gaussian splatting (GaussianSpa, CVPR 2025).

    Two-phase pipeline:
      Phase 1 (simp_iteration): Importance-based pre-pruning — render all training
        views, accumulate per-Gaussian importance scores, sample top Gaussians weighted
        by importance, reinitialize parameters, reset optimizer.
      Phase 2 (ADMM): Drive opacity toward zero for a fraction of remaining Gaussians
        using alternating z-update (hard thresholding) and u-update (dual variable),
        then hard prune at stop iteration.
    """

    def __init__(self, gaussians, rho=0.0005, prune_ratio=0.5, device="cuda"):
        self.gaussians = gaussians
        self.rho = rho
        self.prune_ratio = prune_ratio
        self.device = device

        opacity = gaussians.get_opacity.detach()
        self.u = torch.zeros_like(opacity, device=device)
        self.z = opacity.clone()

    def update_z_u(self, update_u=True):
        """Update auxiliary variable z (thresholding) and optionally dual variable u."""
        with torch.no_grad():
            opacity = self.gaussians.get_opacity.detach()
            z_candidate = opacity + self.u

            # Threshold: keep top (1 - prune_ratio) by value, zero the rest
            n = len(z_candidate)
            index = int(self.prune_ratio * n)
            if index <= 0:
                self.z = z_candidate.clone()
            else:
                z_flat = z_candidate.squeeze()
                z_sorted, _ = torch.sort(z_flat, dim=0)
                threshold = z_sorted[index - 1]
                mask = (z_flat > threshold).unsqueeze(-1)
                self.z = mask.float() * z_candidate

            # Dual variable update: u = u + (opacity - z)
            if update_u:
                self.u = self.u + opacity - self.z

    def compute_spa_loss(self):
        """Compute ADMM augmented Lagrangian penalty term."""
        opacity = self.gaussians.get_opacity
        residual = opacity - self.z + self.u
        return 0.5 * self.rho * torch.sum(residual ** 2)

    def prune(self):
        """Hard prune: remove bottom prune_ratio fraction by opacity."""
        with torch.no_grad():
            opacity = self.gaussians.get_opacity.squeeze()
            n = len(opacity)
            n_keep = max(1, int((1.0 - self.prune_ratio) * n))

            _, indices = torch.topk(opacity, k=n_keep, largest=True)
            prune_mask = torch.ones(n, dtype=torch.bool, device=opacity.device)
            prune_mask[indices] = False

            n_pruned = prune_mask.sum().item()
            print(f"[GSPA] Hard pruning: {n_pruned}/{n} Gaussians removed "
                  f"({n_pruned/n*100:.1f}%), remaining: {n - n_pruned}")

            self.gaussians.prune_points(prune_mask)

    def handle_densification_change(self):
        """Resize u and z after Gaussian count changes (densification/MCMC)."""
        n_current = len(self.gaussians.get_opacity)
        n_stored = len(self.u)

        if n_current == n_stored:
            return

        if n_current > n_stored:
            n_new = n_current - n_stored
            self.u = torch.cat([self.u, torch.zeros(n_new, 1, device=self.device)], dim=0)
            new_opacity = self.gaussians.get_opacity[n_stored:].detach()
            self.z = torch.cat([self.z, new_opacity], dim=0)
        else:
            self.u = self.u[:n_current]
            self.z = self.z[:n_current]

    @staticmethod
    def compute_importance_scores(gaussians, scene, render_fn, pipe, background, imp_metric='indoor'):
        """Compute per-Gaussian importance scores by rendering all training views.

        Uses actual alpha-blending weights (sum of alpha*T per Gaussian) from the
        rasterizer, matching GaussianSpa's importance scoring:
        - Indoor: sum of blending weights across all views
        - Outdoor: blending_weight / pixel_count ratio (penalizes large low-contribution Gaussians)

        Requires record_transmittance=True in render call (transmittance_avg stores
        accum_weights = sum(alpha*T) per Gaussian).
        """
        imp_score = torch.zeros(gaussians._xyz.shape[0], device='cuda')
        accum_area = torch.zeros(gaussians._xyz.shape[0], device='cuda')
        views = scene.getTrainCameras()

        for view in views:
            render_pkg = render_fn(view)
            # accum_weights: sum of alpha*T per Gaussian (from CUDA atomicAdd)
            accum_weights = render_pkg["transmittance_avg"].squeeze()
            cover_pixels = render_pkg["cover_pixels"].squeeze()

            if imp_metric == 'outdoor':
                # Outdoor: weight/area ratio (penalizes large low-contribution Gaussians)
                mask = cover_pixels != 0
                temp = imp_score + accum_weights / cover_pixels.clamp(min=1)
                imp_score[mask] = temp[mask]
            else:
                # Indoor: sum of blending weights
                imp_score += accum_weights

            accum_area += cover_pixels

        # Zero out Gaussians that were never visible
        imp_score[accum_area == 0] = 0
        return imp_score

    @staticmethod
    def importance_prune(gaussians, imp_score, prune_ratio, scene=None):
        """Phase 1: Importance-based pre-pruning (MiniSplatting-style).

        Samples (1 - prune_ratio) fraction of Gaussians weighted by importance score.
        Uses probabilistic sampling (not hard thresholding) to preserve diversity.

        Returns the number of pruned Gaussians.
        """
        from utils.sh_utils import SH2RGB

        n = gaussians._xyz.shape[0]
        n_keep = int(n * (1.0 - prune_ratio))

        # Convert importance to sampling probability
        prob = (imp_score + 1) / (imp_score + 1).sum()
        prob = prob.cpu().numpy()

        # Weighted random sampling (matches reference exactly)
        indices = np.random.choice(n, size=n_keep, p=prob, replace=False)

        # Create prune mask
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        prune_mask = torch.tensor(~mask, device='cuda')

        n_pruned = prune_mask.sum().item()
        print(f"[GSPA] Phase 1 importance pruning: {n_pruned}/{n} Gaussians removed "
              f"({n_pruned/n*100:.1f}%), remaining: {n - n_pruned}")

        gaussians.prune_points(prune_mask)

        # Reinitialize kept Gaussians (matches reference: reinitial_pts)
        gaussians.reinitial_pts(
            gaussians._xyz.detach(),
            SH2RGB(gaussians._features_dc.detach() + 0)[:, 0]
        )

        return n_pruned
