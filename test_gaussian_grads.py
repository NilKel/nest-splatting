import torch
import sys
from scene.gaussian_model import GaussianModel
from argparse import Namespace

# Load a checkpoint and check if gaussian_features has proper optimizer state
checkpoint = torch.load('/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums/gaussian_init.pth')

# Create training args
training_args = Namespace(
    position_lr_init=0.00016,
    position_lr_final=1.6e-06, 
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30000,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01
)

# Create model args with hybrid_levels=5
model_args = Namespace(hybrid_levels=5)

print("Creating GaussianModel...")
gaussians = GaussianModel(3)
gaussians.create_from_pcd(checkpoint['points3D_xyz'], checkpoint['points3D_rgb'], 1.0, args=model_args)

print(f"gaussian_features shape: {gaussians._gaussian_features.shape}")
print(f"gaussian_features requires_grad: {gaussians._gaussian_features.requires_grad}")
print(f"gaussian_features is nn.Parameter: {isinstance(gaussians._gaussian_features, torch.nn.Parameter)}")

# Setup training (creates optimizer)
gaussians.training_setup(training_args)

# Check optimizer state
print("\nOptimizer param_groups:")
for i, pg in enumerate(gaussians.optimizer.param_groups):
    print(f"  {i}: {pg['name']}, lr={pg['lr']}, params={len(pg['params'])}")
    if pg['name'] == 'gaussian_features':
        param = pg['params'][0]
        print(f"    param in optimizer.state: {param in gaussians.optimizer.state}")
        if param in gaussians.optimizer.state:
            print(f"    state keys: {gaussians.optimizer.state[param].keys()}")

# Now restore from checkpoint (this is where the bug might be)
print("\nRestoring from checkpoint...")
model_params = (
    checkpoint['active_sh_degree'],
    checkpoint['xyz'],
    checkpoint['f_dc'],
    checkpoint['f_rest'],
    checkpoint['scaling'],
    checkpoint['rotation'],
    checkpoint['opacity'],
    checkpoint['max_radii2D'],
    checkpoint['xyz_gradient_accum'],
    checkpoint['denom'],
    checkpoint['optimizer_state_dict'],
    checkpoint['spatial_lr_scale'],
    gaussians._gaussian_features  # This is the new parameter
)

gaussians.restore(model_params, training_args)

print("\nAfter restore:")
print(f"gaussian_features requires_grad: {gaussians._gaussian_features.requires_grad}")
print(f"gaussian_features is nn.Parameter: {isinstance(gaussians._gaussian_features, torch.nn.Parameter)}")

# Check optimizer state again
print("\nOptimizer param_groups after restore:")
for i, pg in enumerate(gaussians.optimizer.param_groups):
    print(f"  {i}: {pg['name']}, lr={pg['lr']}, params={len(pg['params'])}")
    if pg['name'] == 'gaussian_features':
        param = pg['params'][0]
        print(f"    param in optimizer.state: {param in gaussians.optimizer.state}")
        if param in gaussians.optimizer.state:
            print(f"    state keys: {gaussians.optimizer.state[param].keys()}")
        else:
            print(f"    WARNING: gaussian_features not in optimizer state!")

print("\nDone!")
