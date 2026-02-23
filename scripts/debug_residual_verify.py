#!/usr/bin/env python3
"""
Quick test: does storing FULL per-texel SH as the texture (with zero mean)
produce better PSNR than the mean+residual split?

Also checks if the full per-texel SH texture has enough information
to reproduce the training quality.
"""
import os, sys, math, glob, pickle, torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
baked_dir = os.path.join(model_path, "baked")

# Load residual textures and baked PLY SH
residual_tex = torch.load(os.path.join(baked_dir, "residual_textures.pt")).cuda().float()
# [N, 8, 8, 48] residual = per_texel - mean

# Load mean SH from baked PLY
from scene import GaussianModel
from arguments import ModelParams
from argparse import ArgumentParser

with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True

temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
gaussians.load_ply(os.path.join(baked_dir, "baked.ply"))

N = len(gaussians.get_xyz)
print(f"N={N}, residual shape: {list(residual_tex.shape)}")

# Reconstruct mean SH from PLY in channel-first format [N, 48]
# PLY stores: features_dc[N, 1, 3] + features_rest[N, 15, 3]
dc = gaussians._features_dc.data  # [N, 1, 3]
rest = gaussians._features_rest.data  # [N, 15, 3]
mean_sh_ply = torch.cat([dc, rest], dim=1)  # [N, 16, 3]
# Convert from interleaved [N, 16, 3] to channel-first [N, 48]
mean_sh = mean_sh_ply.permute(0, 2, 1).reshape(N, 48)  # [N, 3, 16] → [N, 48]

print(f"mean_sh: {list(mean_sh.shape)}, mean={mean_sh.mean():.6f}")

# Reconstruct full per-texel SH
full_texel_sh = residual_tex.view(N, 64, 48) + mean_sh.unsqueeze(1)  # [N, 64, 48]

# Verify: mean of full should equal mean_sh
recomputed_mean = full_texel_sh.mean(dim=1)  # [N, 48]
mean_diff = (recomputed_mean - mean_sh).abs()
print(f"Verify mean roundtrip: max_diff={mean_diff.max():.6f}, mean_diff={mean_diff.mean():.6f}")

# Check statistics of different decompositions
print(f"\nFull texel SH: mean={full_texel_sh.mean():.6f}, std={full_texel_sh.std():.6f}")
print(f"Mean SH:       mean={mean_sh.mean():.6f}, std={mean_sh.std():.6f}")
print(f"Residual:      mean={residual_tex.float().mean():.6f}, std={residual_tex.float().std():.6f}")

# Evaluate SH at a sample viewdir (0, 0, 1)
SH_C0 = 0.28209479177387814
basis = torch.zeros(16, device='cuda')
basis[0] = SH_C0
basis[2] = 0.4886025119029199  # SH_C1 * z for z=1

# Evaluate at center texel for all gaussians
center = full_texel_sh[:, 32, :]  # texel(4,0) roughly center of 8x8
mean_eval = torch.zeros(N, 3, device='cuda')
texel_eval = torch.zeros(N, 3, device='cuda')
for ch in range(3):
    mean_eval[:, ch] = (mean_sh[:, ch*16:(ch+1)*16] * basis.unsqueeze(0)).sum(dim=1) + 0.5
    texel_eval[:, ch] = (center[:, ch*16:(ch+1)*16] * basis.unsqueeze(0)).sum(dim=1) + 0.5

mean_eval = mean_eval.clamp(min=0)
texel_eval = texel_eval.clamp(min=0)

print(f"\nEvaluated at z-direction:")
print(f"  Mean SH color: mean={mean_eval.mean():.4f}, std={mean_eval.std():.4f}")
print(f"  Texel SH color: mean={texel_eval.mean():.4f}, std={texel_eval.std():.4f}")
color_diff = (texel_eval - mean_eval).abs()
print(f"  Color diff: mean={color_diff.mean():.6f}, std={color_diff.std():.6f}")

# Check per-texel variation magnitude
# How much does the SH actually vary across texels?
per_texel_std = full_texel_sh.std(dim=1)  # [N, 48] std across 64 texels
print(f"\nPer-texel SH std (across 8x8 grid):")
print(f"  mean of std: {per_texel_std.mean():.6f}")
print(f"  max of std:  {per_texel_std.max():.6f}")
print(f"  Fraction of Gaussians with significant variation (std > 0.1): "
      f"{(per_texel_std.max(dim=1)[0] > 0.1).sum().item() / N * 100:.1f}%")
print(f"  Fraction of Gaussians with significant variation (std > 0.01): "
      f"{(per_texel_std.max(dim=1)[0] > 0.01).sum().item() / N * 100:.1f}%")

# FP16 quantization error in residual
residual_fp32 = residual_tex.float()
residual_fp16_rt = residual_fp32.half().float()
quant_err = (residual_fp32 - residual_fp16_rt).abs()
print(f"\nFP16 quantization of residual:")
print(f"  Error: mean={quant_err.mean():.8f}, max={quant_err.max():.8f}")
print(f"  Signal-to-quantization ratio: {residual_fp32.abs().mean():.6f} / {quant_err.mean():.8f}")
