#!/usr/bin/env python3
"""
Diagnostic script to check if INGP optimizer contains hash_encoding parameters
"""
import torch
import sys
sys.path.append("./")
from hash_encoder.modules import INGP
from hash_encoder.config import Config
# Load config using the Config class (it expects a filename, not a dict)
cfg_model = Config("./configs/nerfsyn.yaml")

# Create INGP model (this is what happens at line 136 in train.py)
print("Creating INGP model...")
ingp_model = INGP(cfg_model).to('cuda')

print("\n" + "="*70)
print("CHECKING INGP OPTIMIZER STATE")
print("="*70)

# Check if optimizer exists
print(f"\n1. Optimizer created: {hasattr(ingp_model, 'optimizer')}")
if hasattr(ingp_model, 'optimizer'):
    print(f"   Optimizer type: {type(ingp_model.optimizer)}")

# Check optimizer parameter groups
print(f"\n2. Number of parameter groups: {len(ingp_model.optimizer.param_groups)}")
for i, param_group in enumerate(ingp_model.optimizer.param_groups):
    print(f"\n   Group {i}: {param_group.get('name', 'unnamed')}")
    print(f"   - Learning rate: {param_group['lr']}")
    print(f"   - Number of parameters: {len(param_group['params'])}")
    total_params = sum(p.numel() for p in param_group['params'])
    print(f"   - Total parameter count: {total_params:,}")

    # Check if params require grad
    requires_grad = [p.requires_grad for p in param_group['params']]
    print(f"   - All require gradients: {all(requires_grad)}")

# Check hash_encoding parameters specifically
print(f"\n3. Hash encoding module:")
print(f"   - Type: {type(ingp_model.hash_encoding)}")
print(f"   - Has parameters: {len(list(ingp_model.hash_encoding.parameters())) > 0}")

if len(list(ingp_model.hash_encoding.parameters())) > 0:
    for name, param in ingp_model.hash_encoding.named_parameters():
        print(f"   - {name}: shape={param.shape}, requires_grad={param.requires_grad}, numel={param.numel():,}")

# Verify embeddings are in optimizer
print(f"\n4. Verifying hash encoding embeddings are in optimizer...")
hash_param_ids = {id(p) for p in ingp_model.hash_encoding.parameters()}
optimizer_param_ids = {id(p) for group in ingp_model.optimizer.param_groups for p in group['params']}

embeddings_in_optimizer = hash_param_ids.issubset(optimizer_param_ids)
print(f"   - All hash encoding params in optimizer: {embeddings_in_optimizer}")

if not embeddings_in_optimizer:
    print("   ⚠️  WARNING: Some hash encoding parameters are NOT in the optimizer!")
    missing = hash_param_ids - optimizer_param_ids
    print(f"   - Missing {len(missing)} parameter(s)")
else:
    print("   ✓ All hash encoding parameters are properly registered in the optimizer")

# Test gradient flow
print(f"\n5. Testing gradient flow...")
dummy_input = torch.randn(100, 3, device='cuda', requires_grad=True)
output = ingp_model(dummy_input)
print(f"   - Input shape: {dummy_input.shape}")
print(f"   - Output shape: {output.shape}")
print(f"   - Output requires_grad: {output.requires_grad}")

# Backward pass
loss = output.sum()
loss.backward()

# Check if gradients exist
hash_grads_exist = all(p.grad is not None for p in ingp_model.hash_encoding.parameters())
print(f"   - Hash encoding gradients exist after backward: {hash_grads_exist}")

if hash_grads_exist:
    for name, param in ingp_model.hash_encoding.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"   - {name} gradient norm: {grad_norm:.6f}")
    print("   ✓ Gradient flow is working correctly")
else:
    print("   ⚠️  WARNING: No gradients on hash encoding parameters!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
