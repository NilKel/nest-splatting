import torch

ckpt = torch.load("/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/ficus/gaussian_init.pth", 
                  map_location='cpu', weights_only=False)

print(f"All checkpoint keys: {list(ckpt.keys())}")
print(f"\nChecking for appearance_level or similar...")
for key in ckpt.keys():
    if 'appear' in key.lower() or 'level' in key.lower():
        print(f"  Found: {key}")


