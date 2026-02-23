#!/usr/bin/env python3
"""
Debug 48D SH residual: verify that CUDA SH evaluation matches PyTorch.

Loads baked textures, picks a test view, evaluates the 48D SH residual
in PyTorch to check expected improvement.
"""
import os, sys, json, math, pickle, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
baked_dir = os.path.join(model_path, "baked")

# Load residual textures
tex = torch.load(os.path.join(baked_dir, "residual_textures.pt")).cuda()  # [N, 8, 8, 48] half
N = tex.shape[0]
print(f"Residual textures: {list(tex.shape)}, dtype={tex.dtype}")
print(f"  mean={tex.float().mean():.6f}, std={tex.float().std():.6f}")
print(f"  min={tex.float().min():.6f}, max={tex.float().max():.6f}")

# Convert to float for analysis
tex_f = tex.float()  # [N, 8, 8, 48]

# Check per-coefficient statistics
# Layout: channel-first [R0..R15, G0..G15, B0..B15]
for ch_name, ch_start in [("R", 0), ("G", 16), ("B", 32)]:
    for k in range(16):
        coeff = tex_f[:, :, :, ch_start + k]
        if coeff.abs().mean() > 0.001:
            print(f"  {ch_name}[{k}]: mean={coeff.mean():.6f}, std={coeff.std():.6f}")

# Evaluate SH residual at a sample viewdir
print("\n--- SH evaluation test ---")
# Sample viewdir: looking along z axis
dir_z = torch.tensor([0.0, 0.0, 1.0], device='cuda')

# SH basis values for this direction
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]

x, y, z = 0.0, 0.0, 1.0
xx, yy, zz = x*x, y*y, z*z
xy, yz, xz = x*y, y*z, x*z

basis = torch.zeros(16, device='cuda')
basis[0]  = SH_C0
basis[1]  = -SH_C1 * y
basis[2]  = SH_C1 * z
basis[3]  = -SH_C1 * x
basis[4]  = SH_C2[0] * xy
basis[5]  = SH_C2[1] * yz
basis[6]  = SH_C2[2] * (2*zz - xx - yy)
basis[7]  = SH_C2[3] * xz
basis[8]  = SH_C2[4] * (xx - yy)
basis[9]  = SH_C3[0] * y * (3*xx - yy)
basis[10] = SH_C3[1] * xy * z
basis[11] = SH_C3[2] * y * (4*zz - xx - yy)
basis[12] = SH_C3[3] * z * (2*zz - 3*xx - 3*yy)
basis[13] = SH_C3[4] * x * (4*zz - xx - yy)
basis[14] = SH_C3[5] * z * (xx - yy)
basis[15] = SH_C3[6] * x * (xx - 3*yy)

print(f"SH basis for dir=(0,0,1): {basis.cpu().numpy()}")

# Evaluate residual at center texel (4,4) for all Gaussians
center_residual = tex_f[:, 4, 4, :]  # [N, 48]

# Per-channel evaluation
for ch, ch_name in enumerate(["R", "G", "B"]):
    coeffs = center_residual[:, ch*16:(ch+1)*16]  # [N, 16]
    evaluated = (coeffs * basis.unsqueeze(0)).sum(dim=1)  # [N]
    print(f"  {ch_name} channel: eval mean={evaluated.mean():.6f}, std={evaluated.std():.6f}, "
          f"abs_mean={evaluated.abs().mean():.6f}")

# Overall evaluated residual magnitude
all_eval = torch.zeros(N, 3, device='cuda')
for ch in range(3):
    coeffs = center_residual[:, ch*16:(ch+1)*16]
    all_eval[:, ch] = (coeffs * basis.unsqueeze(0)).sum(dim=1)

print(f"\n  Overall evaluated residual (center texel, z-dir):")
print(f"  per-Gaussian mean: {all_eval.mean():.6f}")
print(f"  per-Gaussian std:  {all_eval.std():.6f}")
print(f"  per-Gaussian abs mean: {all_eval.abs().mean():.6f}")
print(f"  min: {all_eval.min():.6f}, max: {all_eval.max():.6f}")

# Compare to DC-only residual
dc_residual = center_residual[:, 0:1] * SH_C0 + center_residual[:, 16:17] * SH_C0 + center_residual[:, 32:33] * SH_C0
print(f"\n  DC-only residual (SH_C0 * coeff_0 for each channel):")
dc_eval = torch.zeros(N, 3, device='cuda')
for ch in range(3):
    dc_eval[:, ch] = center_residual[:, ch*16] * SH_C0
print(f"  mean: {dc_eval.mean():.6f}, std: {dc_eval.std():.6f}")

# Check FP16 quantization error
tex_fp32 = tex.float()
tex_fp16_back = tex_fp32.half().float()
quant_err = (tex_fp32 - tex_fp16_back).abs()
print(f"\n  FP16 quantization error: mean={quant_err.mean():.8f}, max={quant_err.max():.8f}")
# But tex is already FP16, so quant_err should be zero
# Let me check the original baked values
print(f"  (Note: textures were saved as FP16, so this checks round-trip error)")

# Check: what's the non-DC energy in the residual?
total_energy = tex_f.pow(2).sum()
dc_energy = tex_f[:, :, :, 0].pow(2).sum() + tex_f[:, :, :, 16].pow(2).sum() + tex_f[:, :, :, 32].pow(2).sum()
higher_energy = total_energy - dc_energy
print(f"\n  Energy breakdown:")
print(f"  DC energy: {dc_energy:.2f} ({100*dc_energy/total_energy:.1f}%)")
print(f"  Higher-order energy: {higher_energy:.2f} ({100*higher_energy/total_energy:.1f}%)")
