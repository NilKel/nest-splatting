#!/usr/bin/env python3
"""Direct comparison of tcnn MLP vs our CUDA MLP with same input."""
import torch
import tinycudann as tcnn
import os
import sys
import pickle

sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

from hash_encoder.modules import INGP
from hash_encoder.config import Config
from argparse import Namespace

model_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"

with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True

cfg_model = Config(os.path.join(model_path, "config.yaml"))

# Load both modes
args_direct = Namespace(**vars(args))
args_direct.method = "3D_direct"
ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
ingp_direct.load_model(model_path, 30000)

args_lean = Namespace(**vars(args))
args_lean.method = "3D_direct_lean"
ingp_lean = INGP(cfg_model, args=args_lean).to('cuda')
ingp_lean.load_model(model_path, 30000)

print("="*60)
print("MLP COMPARISON TEST")
print("="*60)

# Create test input (40D: 20D gaussian + 4D hash + 16D view)
torch.manual_seed(42)
test_input = torch.randn(100, 40, device='cuda')

# tcnn MLP (3D_direct mode uses this)
tcnn_mlp = ingp_direct.mlp_3D_direct

# PyTorch MLP (3D_direct_lean uploads to CUDA)
pytorch_mlp = ingp_lean.mlp_fused

print(f"\ntcnn MLP params shape: {tcnn_mlp.params.shape}")
print(f"PyTorch MLP structure: {pytorch_mlp}")

# Forward pass comparison
with torch.no_grad():
    # tcnn needs half precision input
    out_tcnn = tcnn_mlp(test_input.half()).float()
    out_tcnn_sigmoid = torch.sigmoid(out_tcnn)
    
    # PyTorch MLP (float32)
    out_pytorch = pytorch_mlp(test_input)  # No sigmoid in mlp_fused
    out_pytorch_sigmoid = torch.sigmoid(out_pytorch)

print(f"\n--- Before sigmoid ---")
print(f"tcnn output [0]: {out_tcnn[0].tolist()}")
print(f"PyTorch output [0]: {out_pytorch[0].tolist()}")
diff_pre = (out_tcnn - out_pytorch).abs()
print(f"MAE: {diff_pre.mean():.6f}, Max: {diff_pre.max():.6f}")

print(f"\n--- After sigmoid ---")
print(f"tcnn output [0]: {out_tcnn_sigmoid[0].tolist()}")
print(f"PyTorch output [0]: {out_pytorch_sigmoid[0].tolist()}")
diff_post = (out_tcnn_sigmoid - out_pytorch_sigmoid).abs()
print(f"MAE: {diff_post.mean():.6f}, Max: {diff_post.max():.6f}")

# Check weights match
W1_tcnn = ingp_direct.mlp_3D_direct.params[:48*32].view(32, 48)[:, :40]
W1_pytorch = ingp_lean.mlp_fused[0].weight.data
print(f"\n--- Weight comparison ---")
print(f"W1 tcnn first row: {W1_tcnn[0, :4].tolist()}")
print(f"W1 pytorch first row: {W1_pytorch[0, :4].tolist()}")
weight_diff = (W1_tcnn.cuda() - W1_pytorch).abs().max()
print(f"W1 max diff: {weight_diff:.8f}")

# Check bias
print(f"\n--- Bias ---")
print(f"PyTorch bias[0]: {ingp_lean.mlp_fused[0].bias.data[:4].tolist()}")
print(f"(tcnn has no explicit bias)")

# Test if precision matters
print(f"\n--- Precision test (PyTorch fp16) ---")
with torch.no_grad():
    out_pytorch_half = pytorch_mlp.half()(test_input.half()).float()
    diff_half = (out_tcnn - out_pytorch_half).abs()
    print(f"MAE with fp16 PyTorch: {diff_half.mean():.6f}, Max: {diff_half.max():.6f}")
