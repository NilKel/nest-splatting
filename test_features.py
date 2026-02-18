#!/usr/bin/env python3
"""Check if Gaussian features and hash lookup are correct."""
import torch
import sys
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')

from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import Namespace
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
import pickle
import os

model_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"

# Load args
with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True

cfg_model = Config(os.path.join(model_path, "config.yaml"))

# Load for 3D_direct mode
args_direct = Namespace(**vars(args))
args_direct.method = "3D_direct"
ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
ingp_direct.load_model(model_path, 30000)

# Load for 3D_direct_lean mode
args_lean = Namespace(**vars(args))
args_lean.method = "3D_direct_lean"
ingp_lean = INGP(cfg_model, args=args_lean).to('cuda')
ingp_lean.load_model(model_path, 30000)

# Check hash encoding table
print("=== Hash encoding comparison ===")
print(f"3D_direct hash_encoding.embeddings shape: {ingp_direct.hash_encoding.embeddings.shape}")
print(f"3D_direct_lean hash_encoding.embeddings shape: {ingp_lean.hash_encoding.embeddings.shape}")

# Check if embeddings are identical
diff = (ingp_direct.hash_encoding.embeddings - ingp_lean.hash_encoding.embeddings).abs().max()
print(f"Hash embeddings max diff: {diff.item()}")

# Check MLP weights
print("\n=== MLP comparison ===")
tcnn_params = ingp_direct.mlp_3D_direct.params.data
print(f"tcnn mlp_3D_direct params shape: {tcnn_params.shape}")
print(f"tcnn mlp_3D_direct first 8: {tcnn_params[:8].tolist()}")

pytorch_W1 = ingp_lean.mlp_fused[0].weight.data
print(f"PyTorch mlp_fused W1 shape: {pytorch_W1.shape}")
print(f"PyTorch mlp_fused W1 first row (8 vals): {pytorch_W1[0, :8].tolist()}")

# Extract tcnn W1 correctly and compare
in_dim, hidden_dim = 40, 32
in_dim_padded = 48
W1_tcnn = tcnn_params[:in_dim_padded * hidden_dim].view(hidden_dim, in_dim_padded)[:, :in_dim]
print(f"\nExtracted tcnn W1 first row (8 vals): {W1_tcnn[0, :8].tolist()}")
print(f"W1 match: {torch.allclose(W1_tcnn.cuda(), pytorch_W1, atol=1e-5)}")

# Test encoding on a point
print("\n=== Hash encoding test ===")
test_xyz = torch.tensor([[0.5, 0.5, 0.5]], device='cuda')
with torch.no_grad():
    hash_direct = ingp_direct._encode_3D(test_xyz)
    hash_lean = ingp_lean._encode_3D(test_xyz)
print(f"3D_direct hash at (0.5,0.5,0.5): {hash_direct[0, :4].tolist()}")
print(f"3D_direct_lean hash at (0.5,0.5,0.5): {hash_lean[0, :4].tolist()}")
print(f"Hash match: {torch.allclose(hash_direct, hash_lean, atol=1e-5)}")

# Test view encoding
print("\n=== View encoding test ===")
test_dir = torch.tensor([[0.577, 0.577, 0.577]], device='cuda')  # normalized
with torch.no_grad():
    view_direct = ingp_direct._encode_view(test_dir)
    view_lean = ingp_lean._encode_view(test_dir)
print(f"3D_direct view enc: {view_direct[0, :4].tolist()}")
print(f"3D_direct_lean view enc: {view_lean[0, :4].tolist()}")
print(f"View enc match: {torch.allclose(view_direct, view_lean, atol=1e-5)}")
