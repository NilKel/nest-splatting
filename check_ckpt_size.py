import torch

ckpt = torch.load("/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/ficus/gaussian_init.pth", 
                  map_location='cpu', weights_only=False)

print(f"Checkpoint iteration: {ckpt['iteration']}")
print(f"Number of Gaussians: {len(ckpt['xyz'])}")
print(f"xyz_gradient_accum shape: {ckpt['xyz_gradient_accum'].shape}")
print(f"denom shape: {ckpt['denom'].shape}")
print(f"max_radii2D shape: {ckpt['max_radii2D'].shape}")

# Check if densification stats are non-zero
print(f"\nxyz_gradient_accum sum: {ckpt['xyz_gradient_accum'].sum().item()}")
print(f"denom sum: {ckpt['denom'].sum().item()}")
print(f"max_radii2D sum: {ckpt['max_radii2D'].sum().item()}")


