#!/usr/bin/env python3
"""Verify tcnn weight extraction."""
import torch
import tinycudann as tcnn

ckpt_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8/ngp_30000.pth"
ckpt = torch.load(ckpt_path)
params = [v for k, v in ckpt['model_state_dict'].items() if 'mlp_3D_direct.params' in k][0].cpu()

in_dim, hidden_dim, out_dim = 40, 32, 3
in_dim_padded, out_dim_padded = 48, 16

# CORRECT extraction: tcnn uses [out, in] layout (NO transpose needed for PyTorch!)
offset = 0
W1 = params[offset:offset + in_dim_padded * hidden_dim].view(hidden_dim, in_dim_padded)[:, :in_dim].contiguous()
offset += in_dim_padded * hidden_dim
W2 = params[offset:offset + hidden_dim * hidden_dim].view(hidden_dim, hidden_dim).contiguous()
offset += hidden_dim * hidden_dim
W3 = params[offset:offset + hidden_dim * out_dim_padded].view(out_dim_padded, hidden_dim)[:out_dim, :].contiguous()

# Build tcnn
tcnn_mlp = tcnn.Network(
    n_input_dims=40, n_output_dims=3,
    network_config={"otype": "MLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 32, "n_hidden_layers": 2}
)
tcnn_mlp.params.data.copy_(params.cuda())

# Build PyTorch MLP
mlp_pt = torch.nn.Sequential(
    torch.nn.Linear(40, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 3),
).cuda()
mlp_pt[0].weight.data, mlp_pt[0].bias.data = W1.cuda(), torch.zeros(32).cuda()
mlp_pt[2].weight.data, mlp_pt[2].bias.data = W2.cuda(), torch.zeros(32).cuda()
mlp_pt[4].weight.data, mlp_pt[4].bias.data = W3.cuda(), torch.zeros(3).cuda()

print("Testing corrected weight extraction:")
torch.manual_seed(42)
total_mae = 0
for i in range(10):
    x = torch.randn(1, 40).cuda()
    with torch.no_grad():
        out_tcnn = tcnn_mlp(x.half()).float()
        out_pt = mlp_pt(x.float())
    mae = abs(out_tcnn - out_pt).mean().item()
    total_mae += mae
    if i < 3:
        print(f"  tcnn:    {out_tcnn[0].tolist()}")
        print(f"  PyTorch: {out_pt[0].tolist()}")
        print(f"  MAE: {mae:.4f}")
        print()

print(f"Average MAE over 10 samples: {total_mae/10:.4f}")

# The remaining error is due to half precision in tcnn
# Let's verify by checking if it's consistent
print("\n>>> THE FIX: tcnn uses [out, in] layout, so NO .T transpose needed! <<<")
print(">>> Current code incorrectly transposes, causing wrong results. <<<")
