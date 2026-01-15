import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from types import SimpleNamespace

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridencoder import GridEncoder


class SphereHashGridBackground(nn.Module):
    """
    Background model using a 3D hashgrid queried on an enclosing sphere.

    Intersects rays with a sphere at a given radius, then queries a 3D hashgrid
    at those intersection positions. This makes the background position-aware,
    allowing different backgrounds for different camera positions.

    The sphere radius should be set larger than the scene extent to ensure
    the background is outside all foreground content.
    """

    def __init__(
        self,
        num_levels: int = 6,
        level_dim: int = 4,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        desired_resolution: int = 512,
        sphere_radius: float = 500.0,
    ):
        """
        Args:
            num_levels: Number of hashgrid levels (default 6 to match main method)
            level_dim: Feature dimension per level (default 4)
            log2_hashmap_size: Hash table size = 2^log2_hashmap_size (default 19)
            base_resolution: Coarsest resolution (default 16)
            desired_resolution: Finest resolution (default 512, lower than main for efficiency)
            sphere_radius: Radius of the background sphere (default 500, should be > scene extent)
        """
        super().__init__()

        self.num_levels = num_levels
        self.level_dim = level_dim
        self.output_dim = num_levels * level_dim
        self.sphere_radius = sphere_radius

        # Calculate per_level_scale from base to desired resolution
        if num_levels > 1:
            per_level_scale = np.exp(np.log(desired_resolution / base_resolution) / (num_levels - 1))
        else:
            per_level_scale = 1.0

        # Create the hashgrid encoder
        # Input is 3D position on sphere, normalized to [0,1]
        self.hash_encoding = GridEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=level_dim,
            per_level_scale=per_level_scale,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )

        self.optimizer = None

        print(f"[BG_HASHGRID] Initialized SphereHashGridBackground:")
        print(f"[BG_HASHGRID]   Levels: {num_levels}, Dim per level: {level_dim}")
        print(f"[BG_HASHGRID]   Output dim: {self.output_dim}")
        print(f"[BG_HASHGRID]   Resolution: {base_resolution} -> {desired_resolution}")
        print(f"[BG_HASHGRID]   Hash table size: 2^{log2_hashmap_size} = {2**log2_hashmap_size}")
        print(f"[BG_HASHGRID]   Sphere radius: {sphere_radius}")

    def training_setup(self, lr: float = 1e-2):
        """Set up optimizer for training."""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-15
        )
        print(f"[BG_HASHGRID] Training setup complete (lr={lr})")

    def forward(self, view_dirs: torch.Tensor, ray_origins: torch.Tensor = None) -> torch.Tensor:
        """
        Query background features by intersecting rays with sphere.

        Args:
            view_dirs: [N, 3] normalized ray directions in world space
            ray_origins: [N, 3] ray origins in world space (camera positions)
                        If None, assumes origin at [0, 0, 0]

        Returns:
            [N, output_dim] features (same shape as foreground features)
        """
        if ray_origins is None:
            ray_origins = torch.zeros_like(view_dirs)

        # Ray-sphere intersection
        # Ray: P = O + t*D
        # Sphere: |P|^2 = R^2
        # Solve: |O + t*D|^2 = R^2
        # => t^2*(D·D) + 2t*(O·D) + (O·O - R^2) = 0
        # For normalized directions, D·D = 1

        # Coefficients of quadratic equation: at^2 + bt + c = 0
        # a = 1 (since view_dirs are normalized)
        b = 2.0 * (ray_origins * view_dirs).sum(dim=-1, keepdim=True)  # (N, 1)
        c = (ray_origins * ray_origins).sum(dim=-1, keepdim=True) - self.sphere_radius ** 2  # (N, 1)

        # Discriminant
        discriminant = b * b - 4.0 * c  # (N, 1)

        # For rays inside sphere, we want the far intersection (positive t)
        # t = (-b + sqrt(discriminant)) / 2
        # For rays outside sphere pointing at it, we want the near intersection
        # But typically cameras are inside the sphere, so we use the far intersection
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        t = (-b + sqrt_disc) / 2.0  # (N, 1)

        # Compute intersection points on sphere
        intersection_points = ray_origins + t * view_dirs  # (N, 3)

        # Normalize intersection points to [0, 1] for hashgrid
        # Points are on sphere surface, so they're in range [-R, R]
        # Map to [0, 1]: (x + R) / (2R)
        coords = (intersection_points + self.sphere_radius) / (2.0 * self.sphere_radius)  # (N, 3)

        # Clamp to [0, 1] for safety (should already be in range)
        coords = torch.clamp(coords, 0.0, 1.0)

        # Query hashgrid
        features = self.hash_encoding(coords)  # (N, output_dim)

        return features

    def save_model(self, exp_path: str, iteration: int):
        """Save background hashgrid checkpoint."""
        state = {'model_state_dict': self.state_dict()}
        save_path = os.path.join(exp_path, f'bg_hashgrid_{iteration}.pth')
        torch.save(state, save_path)
        print(f"[BG_HASHGRID] Saved checkpoint to {save_path}")

    def load_model(self, exp_path: str, iteration: int) -> bool:
        """Load background hashgrid checkpoint."""
        load_path = os.path.join(exp_path, f'bg_hashgrid_{iteration}.pth')
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"[BG_HASHGRID] Loaded checkpoint from {load_path}")
            return True
        return False


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
