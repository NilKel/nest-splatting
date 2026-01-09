import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


class LearnableSkybox(nn.Module):
    """
    Learnable equirectangular skybox for background modeling.

    Maps ray directions to RGB colors via a learnable texture.
    Uses spherical coordinates (phi, theta) to sample an equirectangular map.

    This is useful for outdoor scenes where the background (sky, distant scenery)
    is better modeled separately from the Gaussians to prevent MCMC from wasting
    capacity on infinite-distance content.
    """

    def __init__(self, resolution_h=512, resolution_w=1024):
        """
        Args:
            resolution_h: Height of the equirectangular texture (default 512)
            resolution_w: Width of the equirectangular texture (default 1024, 2:1 aspect)
        """
        super().__init__()
        # Initialize to 5 so sigmoid(5)≈0.993 (white) - matches typical overexposed sky in outdoor scenes
        # Shape: [1, 3, H, W] for grid_sample compatibility
        self.texture = nn.Parameter(torch.full((1, 3, resolution_h, resolution_w), 5.0))
        self.resolution_h = resolution_h
        self.resolution_w = resolution_w
        self.optimizer = None

    def training_setup(self, lr=1e-3):
        """Set up optimizer for training."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-15)

    def forward(self, view_dirs):
        """
        Query skybox colors for given ray directions.

        Args:
            view_dirs: [N, 3] normalized ray directions in world space

        Returns:
            [N, 3] RGB colors
        """
        # Convert Cartesian (x, y, z) to Spherical (phi, theta)
        # Assuming Y-up coordinate system (common in graphics/COLMAP)
        # phi (longitude/azimuth) in [-pi, pi], measured in XZ plane
        phi = torch.atan2(view_dirs[:, 0], view_dirs[:, 2])
        # theta (latitude/elevation) in [0, pi], where 0 = north pole (Y+), pi = south pole (Y-)
        theta = torch.acos(view_dirs[:, 1].clamp(-1.0, 1.0))

        # Convert to UV coordinates in range [-1, 1] for grid_sample
        # u maps phi: [-pi, pi] -> [-1, 1]
        u = phi / math.pi
        # v maps theta: [0, pi] -> [-1, 1]
        v = (theta / math.pi) * 2.0 - 1.0

        # grid_sample expects [B, H_out, W_out, 2] with (x, y) coordinates
        # We have N points, reshape to [1, 1, N, 2]
        grid = torch.stack([u, v], dim=-1).reshape(1, 1, -1, 2)

        # Sample texture with bilinear interpolation
        # padding_mode='border' handles edge cases gracefully
        rgb = F.grid_sample(
            self.texture,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Output shape: [1, 3, 1, N] -> [N, 3]
        # Apply sigmoid to keep RGB in [0, 1] range
        return torch.sigmoid(rgb.reshape(3, -1).permute(1, 0))

    def save_model(self, exp_path, iteration):
        """Save skybox checkpoint."""
        state = {'model_state_dict': self.state_dict()}
        save_path = os.path.join(exp_path, f'skybox_{iteration}.pth')
        torch.save(state, save_path)
        print(f"[SKYBOX] Saved checkpoint to {save_path}")

    def load_model(self, exp_path, iteration):
        """Load skybox checkpoint."""
        load_path = os.path.join(exp_path, f'skybox_{iteration}.pth')
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"[SKYBOX] Loaded checkpoint from {load_path}")
            return True
        return False

    def export_texture(self, save_path):
        """Export the learned texture as an image for visualization."""
        from torchvision.utils import save_image
        # texture is [1, 3, H, W], clamp to valid range
        img = torch.clamp(self.texture[0], 0.0, 1.0)
        save_image(img, save_path)
        print(f"[SKYBOX] Exported texture to {save_path}")
