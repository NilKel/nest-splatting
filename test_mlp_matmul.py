#!/usr/bin/env python3
"""Test if tcnn uses different matmul order."""
import torch
import tinycudann as tcnn

model_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"
ckpt = torch.load(f"{model_path}/ngp_30000.pth")
params = [v for k, v in ckpt['model_state_dict'].items() if 'mlp_3D_direct.params' in k][0].cuda()

tcnn_mlp = tcnn.Network(
    n_input_dims=40, n_output_dims=3,
    network_config={"otype": "MLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 32, "n_hidden_layers": 2}
)
tcnn_mlp.params.data.copy_(params)

torch.manual_seed(42)
x = torch.randn(1, 40, device='cuda')

with torch.no_grad():
    out_tcnn = tcnn_mlp(x.half()).float()

print(f"tcnn output: {out_tcnn[0].tolist()}")

in_dim, hidden_dim, out_dim = 40, 32, 3
in_pad, out_pad = 48, 16

# Current extraction: [out, in] view
offset = 0
W1 = params[offset:offset + in_pad * hidden_dim].view(hidden_dim, in_pad)[:, :in_dim]  # [32, 40]
offset += in_pad * hidden_dim
W2 = params[offset:offset + hidden_dim * hidden_dim].view(hidden_dim, hidden_dim)  # [32, 32]
offset += hidden_dim * hidden_dim
W3 = params[offset:offset + hidden_dim * out_pad].view(out_pad, hidden_dim)[:out_dim, :]  # [3, 32]

# V1: PyTorch Linear style (x @ W.T)
h1 = torch.relu(x @ W1.T)
h2 = torch.relu(h1 @ W2.T)
out_v1 = h2 @ W3.T
print(f"\nV1 (x @ W.T): {out_v1[0].tolist()}")
print(f"  MAE vs tcnn: {(out_tcnn - out_v1).abs().mean():.6f}")

# Alternative: [in, out] view
W1_alt = params[:in_pad * hidden_dim].view(in_pad, hidden_dim)[:in_dim, :]  # [40, 32]
W2_alt = params[in_pad * hidden_dim:in_pad * hidden_dim + hidden_dim * hidden_dim].view(hidden_dim, hidden_dim)  # [32, 32]
W3_alt = params[in_pad * hidden_dim + hidden_dim * hidden_dim:].view(hidden_dim, out_pad)[:, :out_dim]  # [32, 3]

# V2: x @ W (no transpose)
h1 = torch.relu(x @ W1_alt)  # [1,40] @ [40,32] = [1,32]
h2 = torch.relu(h1 @ W2_alt)  # [1,32] @ [32,32] = [1,32]
out_v2 = h2 @ W3_alt  # [1,32] @ [32,3] = [1,3]
print(f"\nV2 (x @ W, [in,out] view): {out_v2[0].tolist()}")
print(f"  MAE vs tcnn: {(out_tcnn - out_v2).abs().mean():.6f}")

# V3: Test with half precision
h1 = torch.relu(x.half() @ W1.T.half())
h2 = torch.relu(h1 @ W2.T.half())
out_v3 = (h2 @ W3.T.half()).float()
print(f"\nV3 (fp16 x @ W.T): {out_v3[0].tolist()}")
print(f"  MAE vs tcnn: {(out_tcnn - out_v3).abs().mean():.6f}")

# V4: Test [in,out] view with half precision  
h1 = torch.relu(x.half() @ W1_alt.half())
h2 = torch.relu(h1 @ W2_alt.half())
out_v4 = (h2 @ W3_alt.half()).float()
print(f"\nV4 (fp16 x @ W, [in,out] view): {out_v4[0].tolist()}")
print(f"  MAE vs tcnn: {(out_tcnn - out_v4).abs().mean():.6f}")

print("\n>>> CONCLUSION: tcnn uses [in, out] layout with x @ W (no transpose)")
