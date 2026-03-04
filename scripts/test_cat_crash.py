"""Minimal test to isolate cat mode crash location."""
import torch
import sys
sys.path.insert(0, '.')

# Import the rasterizer
import diff_surfel_rasterization as _C
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, HashGridSettings

def test_cat_mode():
    device = 'cuda'
    P = 141698  # Same as training
    H, W = 800, 800  # Match nerf synthetic resolution

    # Basic Gaussian parameters
    means3D = torch.randn(P, 3, device=device) * 0.5
    scales = torch.rand(P, 2, device=device) * 0.01
    rotations = torch.randn(P, 4, device=device)
    rotations = rotations / rotations.norm(dim=1, keepdim=True)
    opacities = torch.sigmoid(torch.randn(P, 1, device=device))

    # Per-Gaussian features for cat mode (hybrid_levels=5, l_dim=4 -> 20D)
    colors_precomp = torch.randn(P, 20, device=device, requires_grad=True)

    # Screen-space points
    means2D = torch.zeros(P, 3, device=device, requires_grad=True)

    # TransMat precomp (9 values per Gaussian) - None means use scales/rotations
    cov3D_precomp = None

    # Homotrans
    homotrans = torch.zeros(P, 9, device=device)

    # Appearance level
    ap_level = torch.zeros(P, dtype=torch.float32, device=device)

    # Hash features (1 level, 4D, table size 2^19)
    table_size = 2**19
    features = torch.randn(table_size, 4, device=device, requires_grad=True)
    offsets = torch.tensor([0, table_size], dtype=torch.int32, device=device)
    gridrange = torch.tensor([-1.5, 1.5], dtype=torch.float32, device=device)

    # SH (empty for cat mode)
    shs = torch.Tensor([]).cuda()

    # Shapes (beta kernel) - requires_grad=True to match training
    shapes = torch.ones(P, 1, device=device, requires_grad=True) * 2.0

    # Camera
    viewmatrix = torch.eye(4, device=device).flatten()
    projmatrix = torch.eye(4, device=device).flatten()
    campos = torch.zeros(3, device=device)
    bg = torch.zeros(3, device=device)

    # Level encoding for cat mode: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
    total_levels = 6
    active_hashgrid_levels = 1
    hybrid_levels = 5
    levels = (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels

    # Shape dims: [GS=20, HS=4, OS=24]
    shape_dims = torch.tensor([20, 4, 24], dtype=torch.int32, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        beta=0.0,
        if_contract=False,
        record_transmittance=True,
        max_intersections=0,
        detach_hash_grad=False,
        max_intersections_per_pixel=0,
    )

    hashgrid_settings = HashGridSettings(
        L=levels,
        S=0.0,  # log2(per_level_scale)
        H=512,  # base_resolution
        align_corners=False,
        interpolation=1,
        shape_dims=shape_dims,
        aa=0.0,
        aa_threshold=0.01,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings, hashgrid_settings=hashgrid_settings)

    print(f"P={P}, H={H}, W={W}")
    print(f"colors_precomp shape: {colors_precomp.shape}")
    print(f"features shape: {features.shape}")
    print(f"offsets: {offsets}")
    print(f"levels encoded: {levels} (total={total_levels}, hash={active_hashgrid_levels}, hybrid={hybrid_levels})")
    print(f"shape_dims: {shape_dims}")
    print(f"render_mode: 1 (cat)")

    # Empty tensors for unused params
    features_diffuse = torch.Tensor([]).cuda()
    offsets_diffuse = torch.Tensor([]).int().cuda()
    gridrange_diffuse = torch.Tensor([]).cuda()

    print("\n--- Forward pass ---")
    torch.cuda.synchronize()
    print("  Before forward: OK")

    try:
        result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            shs=None,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            homotrans=homotrans,
            ap_level=ap_level,
            features=features,
            offsets=offsets,
            gridrange=gridrange,
            features_diffuse=features_diffuse,
            offsets_diffuse=offsets_diffuse,
            gridrange_diffuse=gridrange_diffuse,
            render_mode=1,
            shapes=shapes,
            kernel_type=1,  # beta kernel
            aabb_mode=0,
        )
        torch.cuda.synchronize()
        print("  After forward: OK")

        color, radii, allmap, transmittance_avg, pixels, intersection_buffer, intersection_count, geomBuffer = result
        print(f"  color shape: {color.shape}")
        print(f"  radii nonzero: {(radii > 0).sum().item()}")

    except Exception as e:
        print(f"  FORWARD FAILED: {e}")
        return

    print("\n--- Backward pass ---")
    # RGB loss (on MLP output, but we test raw features)
    target = torch.randn_like(color[:3])
    rgb_loss = (color[:3] - target).abs().mean()

    # Mask loss - this uses allmap alpha channel like training does
    render_alpha = allmap[1:2]  # Alpha channel from auxiliary output
    gt_alpha = torch.ones_like(render_alpha)
    mask_loss = 0.1 * (render_alpha - gt_alpha).abs().mean()

    # Shape regularization
    shape_reg = 0.001 * shapes.mean()

    loss = rgb_loss + mask_loss + shape_reg
    print(f"  loss: {loss.item():.6f} (rgb={rgb_loss.item():.6f}, mask={mask_loss.item():.6f})")

    torch.cuda.synchronize()
    print("  Before backward: OK")

    try:
        loss.backward()
        torch.cuda.synchronize()
        print("  After backward: OK")
    except Exception as e:
        print(f"  BACKWARD FAILED: {e}")
        return

    print("\n--- Gradient check ---")
    print(f"  colors_precomp.grad: {colors_precomp.grad is not None}, shape={colors_precomp.grad.shape if colors_precomp.grad is not None else 'N/A'}")
    print(f"  features.grad: {features.grad is not None}")
    print(f"  means2D.grad: {means2D.grad is not None}")

    print("\nAll OK!")

if __name__ == '__main__':
    test_cat_mode()
