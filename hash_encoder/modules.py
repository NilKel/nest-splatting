
import torch
import numpy as np
import tinycudann as tcnn
import sys, os
from tqdm import tqdm
sys.path += ["./", "../"]
from hash_encoder.config import Config
from utils.nerf_utils import MLPwithSkipConnection, get_activation
from torch import nn
import torch.nn.init as torch_init
from utils.general_utils import get_expon_lr_func
from gridencoder import GridEncoder
from types import SimpleNamespace

class LRScheduler:
    def __init__(self, optimizer, decay_iter = 50):
        self.optimizer = optimizer
        self.decay_iter = decay_iter

        self.initial_lrs = {id(param_group): param_group['lr'] for param_group in optimizer.param_groups}
        self.last_reset = -decay_iter
    
    def update_last_reset(self, current_iter):
        print(f'Reset alpha at {current_iter}, decay ingp lr.')
        self.last_reset = current_iter

    def update_lr(self, current_iter):
        decay_factor = 1.
        # decay_factor = 1.0 + max(0.0, self.decay_iter - (current_iter - self.last_reset) )
        # if decay_factor > 1.0 and current_iter % 10 == 0:
        #     print("decay_factor: ", decay_factor, current_iter)
        for param_group in self.optimizer.param_groups:
            original_lr = self.initial_lrs[id(param_group)]
            # print("original_lr: ", original_lr)
            param_group['lr'] = original_lr * decay_factor

def register_GridEncoder(cfg_encoding):
    print(f'register Grid Encoder.')
    return GridEncoder(
        num_levels = cfg_encoding.n_levels,
        level_dim = cfg_encoding.n_features_per_level,
        per_level_scale = cfg_encoding.per_level_scale,
        base_resolution = cfg_encoding.base_resolution,
        log2_hashmap_size = cfg_encoding.log2_hashmap_size)

class INGP(nn.Module):

    def __init__(self, cfg_model, args=None):
        super().__init__()  

        self.view_dep = cfg_model.rgb.view_dep
        if self.view_dep:
            self.build_view_enc(cfg_model.rgb.encoding_view)
        
        view_enc_dir = 0 if not self.view_dep else self.encoder_dir.n_output_dims

        # Store args for cat mode configuration
        self.args = args
        self.is_cat_mode = args is not None and hasattr(args, 'method') and args.method == "cat"
        # Store args for cat_dropout mode configuration (cat with hash dropout during training)
        self.is_cat_dropout_mode = args is not None and hasattr(args, 'method') and args.method == "cat_dropout"

        # Store args for adaptive mode configuration
        self.is_adaptive_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive"
        # Store args for adaptive_add mode configuration (weighted sum of per-Gaussian and hashgrid)
        self.is_adaptive_add_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_add"
        # Store args for adaptive_cat mode configuration (cat with learnable binary blend weights)
        self.is_adaptive_cat_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_cat"
        self.adaptive_cat_inference = args is not None and hasattr(args, 'adaptive_cat_inference') and args.adaptive_cat_inference
        # Store args for adaptive_zero mode configuration (cat + weighted hash vs zeros)
        self.is_adaptive_zero_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_zero"
        self.adaptive_zero_inference = args is not None and hasattr(args, 'adaptive_zero_inference') and args.adaptive_zero_inference
        # Store args for adaptive_gate mode configuration (VQ-AD style gating: soft -> STE -> hard)
        self.is_adaptive_gate_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_gate"
        self.adaptive_gate_inference = args is not None and hasattr(args, 'adaptive_gate_inference') and args.adaptive_gate_inference
        # Store args for 3D mode (intersection buffer + SH blending in PyTorch) - defined early for hybrid_levels check
        self.is_3D_mode = args is not None and hasattr(args, 'method') and args.method == "3D"
        # Store args for 3D_direct mode (intersection buffer + direct RGB like cat mode's 2D MLP)
        self.is_3D_direct_mode = args is not None and hasattr(args, 'method') and args.method == "3D_direct"
        # Store args for 3D_direct_fused mode (fused in-kernel MLP, no intersection buffer)
        self.is_3D_direct_fused_mode = args is not None and hasattr(args, 'method') and args.method == "3D_direct_fused"
        # Store args for 3D_direct_lean mode (same as fused but uses lean rasterizer library for faster builds)
        self.is_3D_direct_lean_mode = args is not None and hasattr(args, 'method') and args.method == "3D_direct_lean"
        # Store args for 3D_direct_fp16 mode (FP16 weights + FP16 GEMM shared memory, diff_surfel_3D_16)
        self.is_3D_direct_fp16_mode = args is not None and hasattr(args, 'method') and args.method == "3D_direct_fp16"
        # Store args for 3D_direct_TC mode (Tensor Core WMMA for MLP, diff_surfel_3D_tc)
        self.is_3D_direct_tc_mode = args is not None and hasattr(args, 'method') and args.method == "3D_direct_TC"
        # Store args for 3D_SH_TC mode (TC WMMA MLP → 48D SH coefs, diff_surfel_3D_sh)
        self.is_3D_direct_sh_tc_mode = args is not None and hasattr(args, 'method') and args.method == "3D_SH_TC"
        # Store args for 3D_SH_res mode (per-Gaussian SH + tiny hash MLP residual, diff_surfel_3D_sh_res)
        self.is_3D_SH_res_mode = args is not None and hasattr(args, 'method') and args.method == "3D_SH_res"
        # Store args for 3D_SH_cat mode (per-Gaussian SH + hash+DC MLP residual, diff_surfel_3D_sh_res)
        self.is_3D_SH_cat_mode = args is not None and hasattr(args, 'method') and args.method == "3D_SH_cat"
        self.freeze_mlp = args is not None and hasattr(args, 'freeze_mlp') and args.freeze_mlp
        # Treat lean/fp16/tc/sh_tc/sh_res/sh_cat mode same as fused mode for MLP/rendering logic
        if self.is_3D_direct_lean_mode or self.is_3D_direct_fp16_mode or self.is_3D_direct_tc_mode or self.is_3D_direct_sh_tc_mode or self.is_3D_SH_res_mode or self.is_3D_SH_cat_mode:
            self.is_3D_direct_fused_mode = True

        # hybrid_levels is used by cat, cat_dropout, adaptive_cat, adaptive_zero, adaptive_gate, 3D, 3D_direct, and 3D_direct_fused modes
        self.hybrid_levels = args.hybrid_levels if (self.is_cat_mode or self.is_cat_dropout_mode or self.is_adaptive_cat_mode or self.is_adaptive_zero_mode or self.is_adaptive_gate_mode or self.is_3D_mode or self.is_3D_direct_mode or self.is_3D_direct_fused_mode) and hasattr(args, 'hybrid_levels') else 0

        # Determine method - baseline uses C2F, all other methods disable it
        self.method = args.method if args is not None and hasattr(args, 'method') else "baseline"
        self.is_baseline_mode = (self.method == "baseline")

        # Auto-enable disable_c2f for all methods except baseline
        # Baseline is the only method that benefits from coarse-to-fine scheduling
        explicit_disable_c2f = args is not None and hasattr(args, 'disable_c2f') and args.disable_c2f
        self.disable_c2f = explicit_disable_c2f or (not self.is_baseline_mode)
        
        # Store args for diffuse mode configuration (per-Gaussian RGB, no viewdir, no hashgrid)
        self.is_diffuse_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse"
        # Store args for specular mode configuration (full 2DGS with SH, no hashgrid)
        self.is_specular_mode = args is not None and hasattr(args, 'method') and args.method == "specular"
        # Store args for diffuse_ngp mode (diffuse SH + hashgrid on unprojected depth)
        self.is_diffuse_ngp_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse_ngp"
        # Store args for diffuse_offset mode (diffuse SH as xyz offset for hashgrid query)
        self.is_diffuse_offset_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse_offset"
        # Store args for hybrid_SH mode (activate separately then add)
        self.is_hybrid_sh_mode = args is not None and hasattr(args, 'method') and args.method == "hybrid_SH"
        # Store args for hybrid_SH_raw mode (add raw then activate)
        self.is_hybrid_sh_raw_mode = args is not None and hasattr(args, 'method') and args.method == "hybrid_SH_raw"
        # Store args for hybrid_SH_post mode (DEPRECATED)
        self.is_hybrid_sh_post_mode = args is not None and hasattr(args, 'method') and args.method == "hybrid_SH_post"
        # Store args for residual_hybrid mode (SH RGB + hashgrid residual via MLP)
        self.is_residual_hybrid_mode = args is not None and hasattr(args, 'method') and args.method == "residual_hybrid"

        self.build_encoding(cfg_model.encoding)
        
        # Diffuse/Specular/hybrid_SH/hybrid_SH_raw/hybrid_SH_post mode: no MLP needed, just SH-based RGB
        if self.is_diffuse_mode or self.is_specular_mode or self.is_hybrid_sh_mode or self.is_hybrid_sh_raw_mode or self.is_hybrid_sh_post_mode:
            self.feat_dim = 3  # RGB
            self.mlp_rgb = None
            self.view_dep = False  # View dependency handled by SH, not MLP
        elif self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode:
            # Diffuse_ngp/diffuse_offset mode: need MLP for hashgrid decoding
            # MLP input is hashgrid features + view encoding
            self.feat_dim = cfg_model.encoding.levels * cfg_model.encoding.hashgrid.dim
            self.mlp_rgb = self.build_mlp(cfg_model.rgb, input_dim=self.feat_dim + view_enc_dir, output_dim=3)
        elif self.is_residual_hybrid_mode:
            # Residual_hybrid mode: MLP decodes hashgrid residual only
            # MLP input is (total_levels - hybrid_levels) * per_level_dim + view_enc_dir
            hybrid_levels = args.hybrid_levels if hasattr(args, 'hybrid_levels') else 0
            hashgrid_levels = cfg_model.encoding.levels - hybrid_levels
            self.feat_dim = hashgrid_levels * cfg_model.encoding.hashgrid.dim
            mlp_input_dim = self.feat_dim + view_enc_dir
            print(f'[RESIDUAL_HYBRID MODE] MLP input: {hashgrid_levels} levels × {cfg_model.encoding.hashgrid.dim}D hashgrid + {view_enc_dir}D viewdir = {mlp_input_dim}D')
            self.mlp_rgb = self.build_mlp(cfg_model.rgb, input_dim=mlp_input_dim, output_dim=3)
        else:
            # MLP input is always total_levels * dim (buffer size is constant)
            self.feat_dim = cfg_model.encoding.levels * cfg_model.encoding.hashgrid.dim # + 3
            self.mlp_rgb = self.build_mlp(cfg_model.rgb, input_dim=self.feat_dim + view_enc_dir, output_dim = 3)

        # 3D mode: Additional MLP for intersection-based SH rendering
        # Takes hash_features + per-Gaussian features, outputs SH coefficients
        # Uses cat-style split: hybrid_levels for per-Gaussian (coarse), rest for hashgrid (fine)
        self.mlp_3D = None
        if self.is_3D_mode:
            # Cat-style dimensions: coarse (per-Gaussian) + fine (hashgrid) = total
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            hash_dim = (total_levels - self.hybrid_levels) * level_dim  # Fine levels from hashgrid
            gauss_feat_dim = self.hybrid_levels * level_dim  # Coarse levels as per-Gaussian features
            mlp_3D_input = total_levels * level_dim  # Total: hash + Gaussian = all levels
            mlp_3D_output = 48  # 16 SH coefficients × 3 RGB channels
            mlp_3D_hidden = getattr(args, 'mlp_3D_hidden', 32)  # Default 32 (same as 3D_direct)
            mlp_3D_layers = getattr(args, 'mlp_3D_layers', 2)  # Default 2 hidden layers

            # Store dimensions for use in forward pass
            self.mlp_3D_hash_dim = hash_dim
            self.mlp_3D_gauss_dim = gauss_feat_dim

            print(f'[3D MODE] Building mlp_3D (cat-style):')
            print(f'  Coarse (per-Gaussian): {self.hybrid_levels} levels × {level_dim}D = {gauss_feat_dim}D')
            print(f'  Fine (hashgrid): {total_levels - self.hybrid_levels} levels × {level_dim}D = {hash_dim}D')
            print(f'  Total input: {mlp_3D_input}D')
            print(f'  Hidden: {mlp_3D_hidden} neurons × {mlp_3D_layers} layers')
            print(f'  Output: {mlp_3D_output}D (16 SH coeffs × 3 RGB)')

            # Use FullyFusedMLP for speed (requires hidden_dim ≤ 128)
            self.mlp_3D = tcnn.Network(
                n_input_dims=mlp_3D_input,
                n_output_dims=mlp_3D_output,
                network_config={
                    "otype": "FullyFusedMLP" if mlp_3D_hidden <= 128 else "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": mlp_3D_hidden,
                    "n_hidden_layers": mlp_3D_layers,
                },
            )

        # 3D_direct mode: Like 3D mode but outputs RGB directly (like cat mode's 2D MLP)
        # Takes (hash_features + per-Gaussian features + view_encoding) → RGB
        # Uses PyTorch MLP (float32) for simple gradient comparison with 3D_lean CUDA MLP
        self.mlp_3D_direct = None
        if self.is_3D_direct_mode:
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            hash_dim = (total_levels - self.hybrid_levels) * level_dim
            gauss_feat_dim = self.hybrid_levels * level_dim
            mlp_input = total_levels * level_dim + view_enc_dir  # features + view encoding
            mlp_output = 3  # RGB directly
            mlp_hidden = 32  # Match CUDA MLP hidden dim

            self.mlp_3D_direct_hash_dim = hash_dim
            self.mlp_3D_direct_gauss_dim = gauss_feat_dim

            print(f'[3D_DIRECT MODE] Building mlp_3D_direct (PyTorch MLP, float32):')
            print(f'  Coarse (per-Gaussian): {self.hybrid_levels} levels × {level_dim}D = {gauss_feat_dim}D')
            print(f'  Fine (hashgrid): {total_levels - self.hybrid_levels} levels × {level_dim}D = {hash_dim}D')
            print(f'  View encoding: {view_enc_dir}D')
            print(f'  Total input: {mlp_input}D')
            print(f'  Hidden: {mlp_hidden} neurons × 2 layers')
            print(f'  Output: {mlp_output}D (RGB)')

            # Use PyTorch MLP (float32) instead of tcnn for simpler gradient comparison
            # Architecture matches 3D_lean CUDA MLP: 40D → 32D (ReLU) → 32D (ReLU) → 3D
            self.mlp_3D_direct = nn.Sequential(
                nn.Linear(mlp_input, mlp_hidden),  # W1: [40, 32], b1: [32]
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden),  # W2: [32, 32], b2: [32]
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_output),  # W3: [32, 3], b3: [3]
                # Note: sigmoid applied after in forward pass
            ).cuda()

        # 3D_direct_fused mode: PyTorch MLP for in-kernel evaluation
        # Uses explicit weight matrices (not tcnn) so we can upload to CUDA constant memory
        self.mlp_fused = None
        if self.is_3D_SH_cat_mode:
            # 3D_SH_cat: MLP input = [hash(4) | DC_SH(3) | bias(1) | pad(8)] = 16D
            # Same architecture as 3D_SH_res but with extra DC SH identity input
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            hash_dim = (total_levels - self.hybrid_levels) * level_dim  # All non-hybrid levels
            dc_dim = 3  # DC SH (3 RGB channels)
            mlp_input_dim = hash_dim + dc_dim  # e.g., 4 + 3 = 7D
            # Pad to 16D for WMMA alignment: [hash(4) | dc(3) | bias(1) | pad(8)] = 16
            mlp_input_padded = ((mlp_input_dim + 1 + 15) // 16) * 16  # +1 for bias, round up to 16
            hidden_dim = mlp_input_padded  # 16

            print(f'[3D_SH_CAT MODE] Building bias-free residual MLP for CUDA:')
            print(f'  Hash features: {hash_dim}D ({total_levels - self.hybrid_levels} levels × {level_dim}D)')
            print(f'  DC SH input: {dc_dim}D (per-Gaussian identity)')
            print(f'  MLP input: {hash_dim}D hash + {dc_dim}D DC_SH + 1D bias + {mlp_input_padded - mlp_input_dim - 1}D pad = {mlp_input_padded}D')
            print(f'  Architecture: {mlp_input_padded}D → {hidden_dim}D (ReLU) → {hidden_dim}D (ReLU) → {hidden_dim}D (identity, first 3 = RGB residual)')
            print(f'  Bias-free: L1 uses input padding (col {mlp_input_dim} acts as bias), L2/L3 no bias')
            print(f'  Per-Gaussian SH: standard degree-3 (16 coefficients × 3 channels)')

            self.mlp_fused = nn.Sequential(
                nn.Linear(mlp_input_padded, hidden_dim, bias=False),  # W1: [16, 16]
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),         # W2: [16, 16]
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),         # W3: [16, 16] (only first 3 = RGB residual)
            ).cuda()

            self.mlp_fused_input_dim = mlp_input_dim
            self.mlp_fused_hash_dim = hash_dim
            self.mlp_fused_gauss_dim = 0  # No per-Gaussian features (SH handles it)

            if self.freeze_mlp:
                freeze_from = getattr(args, 'freeze_mlp_from', None)
                if freeze_from:
                    import glob as _glob
                    ngp_files = _glob.glob(os.path.join(freeze_from, "ngp_*.pth"))
                    if ngp_files:
                        iters = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
                        ckpt_path = os.path.join(freeze_from, f"ngp_{max(iters)}.pth")
                        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
                        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                        mlp_sd = {k.replace('mlp_fused.', ''): v for k, v in sd.items() if 'mlp_fused' in k}
                        self.mlp_fused.load_state_dict(mlp_sd)
                        print(f'  [FREEZE_MLP] Loaded MLP weights from {ckpt_path}')
                    else:
                        print(f'  [FREEZE_MLP] WARNING: no ngp checkpoint found in {freeze_from}, using random init')
                else:
                    print(f'  [FREEZE_MLP] MLP weights frozen at random init')
                for p in self.mlp_fused.parameters():
                    p.requires_grad_(False)

        elif self.is_3D_SH_res_mode:
            # 3D_SH_res: Tiny view-independent residual MLP (hash features → RGB residual)
            # Per-Gaussian SH handles view-dependent base color (evaluated in CUDA preprocessing)
            # MLP only adds a small spatial correction from hash grid
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            hash_dim = (total_levels - self.hybrid_levels) * level_dim  # All non-hybrid levels
            mlp_input_dim = hash_dim  # e.g., 4D for 1 hash level
            # Pad to 16D for WMMA alignment: [hash(4) | bias(1) | pad(11)] = 16
            mlp_input_padded = ((mlp_input_dim + 1 + 15) // 16) * 16  # +1 for bias, round up to 16
            hidden_dim = mlp_input_padded  # 16

            print(f'[3D_SH_RES MODE] Building bias-free residual MLP for CUDA:')
            print(f'  Hash features: {hash_dim}D ({total_levels - self.hybrid_levels} levels × {level_dim}D)')
            print(f'  MLP input: {mlp_input_dim}D hash + 1D bias + {mlp_input_padded - mlp_input_dim - 1}D pad = {mlp_input_padded}D')
            print(f'  Architecture: {mlp_input_padded}D → {hidden_dim}D (ReLU) → {hidden_dim}D (ReLU) → {hidden_dim}D (identity, first 3 = RGB residual)')
            print(f'  Bias-free: L1 uses input padding (col {mlp_input_dim} acts as bias), L2/L3 no bias')
            print(f'  Per-Gaussian SH: standard degree-3 (16 coefficients × 3 channels)')

            self.mlp_fused = nn.Sequential(
                nn.Linear(mlp_input_padded, hidden_dim, bias=False),  # W1: [16, 16]
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),         # W2: [16, 16]
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),         # W3: [16, 16] (only first 3 = RGB residual)
            ).cuda()

            self.mlp_fused_input_dim = mlp_input_dim
            self.mlp_fused_hash_dim = hash_dim
            self.mlp_fused_gauss_dim = 0  # No per-Gaussian features (SH handles it)

        elif self.is_3D_direct_sh_tc_mode:
            # 3D_SH_TC: MLP outputs 48D SH coefficients, NO view direction input
            # SH evaluation happens in CUDA kernel with per-intersection viewdir
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            feat_dim = total_levels * level_dim  # 24D (6 levels × 4D)
            mlp_input_dim = feat_dim  # 24D (no view encoding!)

            print(f'[3D_SH_TC MODE] Building bias-free PyTorch MLP for CUDA SH mode:')
            print(f'  Features: {feat_dim}D (Gaussian: {self.hybrid_levels * level_dim}D + Hash: {(total_levels - self.hybrid_levels) * level_dim}D)')
            print(f'  Total input: {mlp_input_dim}D (+ 1D bias = {mlp_input_dim + 1}D, no view encoding)')
            print(f'  Architecture: {mlp_input_dim + 1}D → 32D (ReLU) → 32D (ReLU) → 48D SH (identity)')
            print(f'  Bias-free: L1 uses input padding (col 24 acts as bias), L2/L3 no bias')

            self.mlp_fused = nn.Sequential(
                nn.Linear(mlp_input_dim + 1, 32, bias=False),  # W1: [32, 25] (col 24 = implicit bias)
                nn.ReLU(),
                nn.Linear(32, 32, bias=False),                  # W2: [32, 32]
                nn.ReLU(),
                nn.Linear(32, 48, bias=False),                  # W3: [48, 32] (16 SH coefs × 3 RGB)
            ).cuda()

            self.mlp_fused_input_dim = mlp_input_dim
            self.mlp_fused_hash_dim = (total_levels - self.hybrid_levels) * level_dim
            self.mlp_fused_gauss_dim = self.hybrid_levels * level_dim
        elif self.is_3D_direct_fused_mode:
            # 3D_direct_fused/lean/fp16/TC: MLP outputs 3D RGB with view direction as input
            # Architecture: 40D input → 32D hidden (ReLU) × 2 → 3D RGB (sigmoid in CUDA)
            total_levels = cfg_model.encoding.levels
            level_dim = cfg_model.encoding.hashgrid.dim
            feat_dim = total_levels * level_dim  # 24D (6 levels × 4D)
            view_enc_dim = 16  # Positional encoding for view direction
            mlp_input_dim = feat_dim + view_enc_dim  # 40D

            print(f'[3D_DIRECT_FUSED MODE] Building bias-free PyTorch MLP for CUDA global memory:')
            print(f'  Features: {feat_dim}D (Gaussian: {self.hybrid_levels * level_dim}D + Hash: {(total_levels - self.hybrid_levels) * level_dim}D)')
            print(f'  View encoding: {view_enc_dim}D')
            print(f'  Total input: {mlp_input_dim}D (+ 1D padding = {mlp_input_dim + 1}D)')
            print(f'  Architecture: {mlp_input_dim + 1}D → 32D (ReLU) → 32D (ReLU) → 3D (sigmoid)')
            print(f'  Bias-free: L1 uses input padding (col 41 acts as bias), L2/L3 no bias')

            self.mlp_fused = nn.Sequential(
                nn.Linear(mlp_input_dim + 1, 32, bias=False),  # W1: [32, 41] (col 41 = implicit bias)
                nn.ReLU(),
                nn.Linear(32, 32, bias=False),                  # W2: [32, 32]
                nn.ReLU(),
                nn.Linear(32, 3, bias=False),                   # W3: [3, 32]
                # Note: sigmoid is applied in CUDA kernel, not here
            ).cuda()

            self.mlp_fused_input_dim = mlp_input_dim
            self.mlp_fused_hash_dim = (total_levels - self.hybrid_levels) * level_dim
            self.mlp_fused_gauss_dim = self.hybrid_levels * level_dim

        self.training_setup(cfg_model.optim)

        self.pre_level = None
        
        self.warm_up = cfg_model.encoding.warm_up
        self.switch_iter = cfg_model.ingp_stage.switch_iter
        self.keep_geometry = cfg_model.ingp_stage.keep_geometry
        self.initialize = cfg_model.ingp_stage.initialize

        self.contract = cfg_model.settings.contract

    def training_setup(self, training_args):
        lr_encoding = training_args.params.feat_lr
        lr_mlp_rgb = training_args.params.mlp_lr
        lr_enc_view = training_args.params.view_lr

        lr_spec = training_args.params.spec_lr

        # Apply LR scaling for 3D_SH_res mode
        if self.args is not None and hasattr(self.args, 'res_lr_scale') and self.args.res_lr_scale != 1.0:
            if self.args.method in ["3D_SH_res", "3D_SH_cat"]:
                print(f"[3D_SH_RES] Scaling hash/MLP LR by {self.args.res_lr_scale}: encoding {lr_encoding} -> {lr_encoding * self.args.res_lr_scale}, mlp {lr_mlp_rgb} -> {lr_mlp_rgb * self.args.res_lr_scale}")
                lr_encoding *= self.args.res_lr_scale
                lr_mlp_rgb *= self.args.res_lr_scale

        l = []
        # Only add hash_encoding if it exists (not disabled in cat mode or diffuse mode)
        if self.hash_encoding is not None:
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})
        # Only add MLP if it exists (not diffuse mode)
        if self.mlp_rgb is not None:
            l.append({'params': self.mlp_rgb.parameters(), 'lr': lr_mlp_rgb, "name": "rgb_mlp"})
        # Add mlp_3D for 3D mode
        if hasattr(self, 'mlp_3D') and self.mlp_3D is not None:
            l.append({'params': self.mlp_3D.parameters(), 'lr': lr_mlp_rgb, "name": "mlp_3D"})
        # Add mlp_3D_direct for 3D_direct mode
        # Use same LR as CAT mode's mlp_rgb - gradients are now correct after rend_alpha detach fix
        if hasattr(self, 'mlp_3D_direct') and self.mlp_3D_direct is not None:
            l.append({'params': self.mlp_3D_direct.parameters(), 'lr': lr_mlp_rgb, "name": "mlp_3D_direct"})
        # Add mlp_fused for 3D_direct_fused mode (PyTorch MLP for CUDA constant memory)
        # Skip when freeze_mlp: weights stay at init, no optimizer state needed
        if hasattr(self, 'mlp_fused') and self.mlp_fused is not None and not self.freeze_mlp:
            l.append({'params': self.mlp_fused.parameters(), 'lr': lr_mlp_rgb, "name": "mlp_fused"})

        # For diffuse mode, create a dummy optimizer (no INGP params to optimize)
        if len(l) == 0:
            # Create a dummy parameter for the optimizer
            self._dummy_param = nn.Parameter(torch.zeros(1, device="cuda"))
            l.append({'params': [self._dummy_param], 'lr': 0.0, "name": "dummy"})
        
        self.optimizer = torch.optim.Adam(l, betas=(0.9, 0.99), eps=1e-15)

    def build_encoding(self, cfg_encoding):
        assert(cfg_encoding.type == "hashgrid")
        self.voxel_range = cfg_encoding.hashgrid.range
        self.gridrange = torch.tensor(self.voxel_range).cuda().float()

        l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels_total = cfg_encoding.levels
        self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels_total - 1))
        
        # Calculate all resolutions for baseline (needed for cat mode to select correct ones)
        all_resolutions = []
        for lv in range(0, num_levels_total):
            size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
            all_resolutions.append(size)
        
        self.level_dim = cfg_encoding.hashgrid.dim
        self.levels = num_levels_total  # Total levels (for MLP input size)
        
        # Diffuse mode: no hashgrid at all, just per-Gaussian RGB (SH degree 0)
        if self.is_diffuse_mode:
            print(f'[DIFFUSE MODE] No hashgrid - using SH degree 0')
            print(f'[DIFFUSE MODE] hash_in_cuda = False')
            self.hash_encoding = None
            self.hashgrid_disabled = True
            self.hashgrid_levels = 0
            self.resolutions = []
            self.active_hashgrid_levels = 0
            self.active_levels = 0
            # Set coarse-to-fine params (not used but needed to avoid AttributeError)
            self.level_mask = False
            self.init_active_level = 0
            self.step = 1000  # Dummy value, not used in diffuse mode
            # For diffuse mode, feat_dim is 3 (RGB)
            return 3
        
        # Specular mode: no hashgrid, full 2DGS with SH (view-dependent)
        if self.is_specular_mode:
            print(f'[SPECULAR MODE] No hashgrid - using full SH (2DGS style)')
            print(f'[SPECULAR MODE] hash_in_cuda = False')
            self.hash_encoding = None
            self.hashgrid_disabled = True
            self.hashgrid_levels = 0
            self.resolutions = []
            self.active_hashgrid_levels = 0
            self.active_levels = 0
            # Set coarse-to-fine params (not used but needed to avoid AttributeError)
            self.level_mask = False
            self.init_active_level = 0
            self.step = 1000  # Dummy value, not used in specular mode
            # For specular mode, feat_dim is 3 (RGB from SH)
            return 3
        
        # For cat mode: create hashgrid with only the finest (total - hybrid) levels
        elif self.is_cat_mode and self.hybrid_levels > 0:
            self.hashgrid_levels = num_levels_total - self.hybrid_levels
            
            if self.hashgrid_levels <= 0:
                # Edge case: all levels are per-Gaussian features, no hashgrid
                print(f'[CAT MODE] hybrid_levels={self.hybrid_levels} >= total_levels={num_levels_total}')
                print(f'[CAT MODE] Using only per-Gaussian features (no hashgrid)')
                self.hash_encoding = None
                self.hashgrid_disabled = True
                self.resolutions = []
            else:
                # Normal cat mode: use finest levels starting from hybrid_levels
                self.hashgrid_disabled = False
                selected_resolutions = all_resolutions[self.hybrid_levels:]
                base_res = selected_resolutions[0]
                finest_res = selected_resolutions[-1]
                
                # Calculate growth rate for the shrunken hashgrid
                if self.hashgrid_levels > 1:
                    hash_growth_rate = np.exp((np.log(finest_res) - np.log(base_res)) / (self.hashgrid_levels - 1))
                else:
                    hash_growth_rate = 1.0
                
                print(f'[CAT MODE] Total levels: {num_levels_total}, Hybrid levels: {self.hybrid_levels}')
                print(f'[CAT MODE] Per-Gaussian features: {self.hybrid_levels} levels × {self.level_dim} dim = {self.hybrid_levels * self.level_dim}D')
                print(f'[CAT MODE] Hashgrid: {self.hashgrid_levels} levels × {self.level_dim} dim = {self.hashgrid_levels * self.level_dim}D')
                print(f'[CAT MODE] Hashgrid resolutions: {selected_resolutions}')
                
                config = SimpleNamespace(
                    device="cuda",
                    otype="HashGrid",
                    n_levels=self.hashgrid_levels,
                    n_features_per_level=cfg_encoding.hashgrid.dim,
                    log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                    base_resolution=base_res,
                    finest_resolution=finest_res,
                    init_mode='uniform',
                    per_level_scale=hash_growth_rate,
                    range=self.voxel_range,
                )
                
                print('hash config:', config)
                self.hash_encoding = register_GridEncoder(config)
                self.resolutions = selected_resolutions

        # hybrid_SH/hybrid_SH_raw/hybrid_SH_post mode: single finest-resolution level, 3D features (DC residuals)
        elif self.is_hybrid_sh_mode or self.is_hybrid_sh_raw_mode or self.is_hybrid_sh_post_mode:
            # Override level_dim to 3 for DC residuals (RGB channels)
            self.level_dim = 3

            # Calculate finest resolution from standard progression
            finest_resolution = all_resolutions[-1]

            # Create single-level hashgrid
            self.hashgrid_levels = 1
            self.hashgrid_disabled = False

            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=1,                          # Single level
                n_features_per_level=3,              # 3D features (RGB DC residual)
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=finest_resolution,   # Use finest as base
                finest_resolution=finest_resolution, # Same as base
                init_mode='uniform',
                per_level_scale=1.0,                 # No growth (single level)
                range=self.voxel_range,
            )

            print('[HYBRID_SH MODE] Single-level hashgrid configuration:')
            print(f'  Resolution: {finest_resolution}')
            print(f'  Features per level: 3 (DC residual)')
            print(f'  Hashgrid size: 2^{cfg_encoding.hashgrid.dict_size}')

            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = [finest_resolution]

            # Set active levels
            self.active_hashgrid_levels = 1

        # adaptive_cat mode: multi-level hashgrid for blending with per-Gaussian features
        # Same structure as cat mode but with learned blend weights
        elif self.is_adaptive_cat_mode:
            # Use finest (total - hybrid) levels for hashgrid, just like cat mode
            self.hashgrid_levels = num_levels_total - self.hybrid_levels
            self.hashgrid_disabled = False

            # Extract the finest hashgrid_levels resolutions
            hashgrid_resolutions = all_resolutions[-self.hashgrid_levels:]
            base_resolution = hashgrid_resolutions[0]
            finest_resolution = hashgrid_resolutions[-1]

            # Calculate growth rate for the hashgrid
            if self.hashgrid_levels > 1:
                hash_growth_rate = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (self.hashgrid_levels - 1))
            else:
                hash_growth_rate = 1.0

            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=self.hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                init_mode='uniform',
                per_level_scale=hash_growth_rate,
                range=self.voxel_range,
            )

            print('[ADAPTIVE_CAT MODE] Multi-level hashgrid configuration:')
            print(f'  Total levels: {num_levels_total}')
            print(f'  Hybrid (per-Gaussian) levels: {self.hybrid_levels}')
            print(f'  Hashgrid levels: {self.hashgrid_levels}')
            print(f'  Resolutions: {base_resolution} -> {finest_resolution}')
            print(f'  Features per level: {cfg_encoding.hashgrid.dim}D')
            print(f'  Hashgrid size: 2^{cfg_encoding.hashgrid.dict_size}')
            print(f'  Per-Gaussian features: {num_levels_total}×{cfg_encoding.hashgrid.dim} = {num_levels_total * cfg_encoding.hashgrid.dim}D')
            print(f'  Blend: Learned per-Gaussian weights (training=smooth, inference=binary)')

            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = hashgrid_resolutions

            # Set active levels (all hashgrid levels active, no C2F for adaptive_cat)
            self.active_hashgrid_levels = self.hashgrid_levels

        # adaptive_zero mode: same hashgrid structure as cat/adaptive_cat
        # Uses per-Gaussian features for coarse levels, hashgrid for fine levels
        # Weight controls whether to query hash (1) or use zeros (0)
        elif self.is_adaptive_zero_mode:
            # Use finest (total - hybrid) levels for hashgrid, just like cat mode
            self.hashgrid_levels = num_levels_total - self.hybrid_levels
            self.hashgrid_disabled = False

            # Extract the finest hashgrid_levels resolutions
            hashgrid_resolutions = all_resolutions[-self.hashgrid_levels:]
            base_resolution = hashgrid_resolutions[0]
            finest_resolution = hashgrid_resolutions[-1]

            # Calculate growth rate for the hashgrid
            if self.hashgrid_levels > 1:
                hash_growth_rate = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (self.hashgrid_levels - 1))
            else:
                hash_growth_rate = 1.0

            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=self.hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                init_mode='uniform',
                per_level_scale=hash_growth_rate,
                range=self.voxel_range,
            )

            print('[ADAPTIVE_ZERO MODE] Multi-level hashgrid configuration:')
            print(f'  Total levels: {num_levels_total}')
            print(f'  Hybrid (per-Gaussian) levels: {self.hybrid_levels}')
            print(f'  Hashgrid levels: {self.hashgrid_levels}')
            print(f'  Resolutions: {base_resolution} -> {finest_resolution}')
            print(f'  Features per level: {cfg_encoding.hashgrid.dim}D')
            print(f'  Hashgrid size: 2^{cfg_encoding.hashgrid.dict_size}')
            print(f'  Per-Gaussian features: {self.hybrid_levels}×{cfg_encoding.hashgrid.dim} = {self.hybrid_levels * cfg_encoding.hashgrid.dim}D')
            print(f'  Weight: 0 = zeros (no hash query), 1 = query hash')

            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = hashgrid_resolutions

            # Set active levels (all hashgrid levels active, no C2F for adaptive_zero)
            self.active_hashgrid_levels = self.hashgrid_levels

        # adaptive_gate mode: VQ-AD style gating with three-phase training
        # Same hashgrid structure as adaptive_zero, but different gating logic in Python
        elif self.is_adaptive_gate_mode:
            # Use finest (total - hybrid) levels for hashgrid, just like cat mode
            self.hashgrid_levels = num_levels_total - self.hybrid_levels
            self.hashgrid_disabled = False

            # Extract the finest hashgrid_levels resolutions
            hashgrid_resolutions = all_resolutions[-self.hashgrid_levels:]
            base_resolution = hashgrid_resolutions[0]
            finest_resolution = hashgrid_resolutions[-1]

            # Calculate growth rate for the hashgrid
            if self.hashgrid_levels > 1:
                hash_growth_rate = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (self.hashgrid_levels - 1))
            else:
                hash_growth_rate = 1.0

            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=self.hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                init_mode='uniform',
                per_level_scale=hash_growth_rate,
                range=self.voxel_range,
            )

            print('[ADAPTIVE_GATE MODE] VQ-AD style gating configuration:')
            print(f'  Total levels: {num_levels_total}')
            print(f'  Hybrid (per-Gaussian) levels: {self.hybrid_levels}')
            print(f'  Hashgrid levels: {self.hashgrid_levels}')
            print(f'  Resolutions: {base_resolution} -> {finest_resolution}')
            print(f'  Features per level: {cfg_encoding.hashgrid.dim}D')
            print(f'  Hashgrid size: 2^{cfg_encoding.hashgrid.dict_size}')
            print(f'  Per-Gaussian features: {self.hybrid_levels}×{cfg_encoding.hashgrid.dim} = {self.hybrid_levels * cfg_encoding.hashgrid.dim}D')
            print(f'  Three-phase gating: soft -> STE -> hard')

            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = hashgrid_resolutions

            # Set active levels (all hashgrid levels active, no C2F for adaptive_gate)
            self.active_hashgrid_levels = self.hashgrid_levels

        # For residual_hybrid mode: hashgrid uses only finest (total - hybrid) levels
        # Similar to cat mode but outputs both SH RGB and hash features
        elif self.is_residual_hybrid_mode and hasattr(self.args, 'hybrid_levels') and self.args.hybrid_levels > 0:
            hybrid_levels = self.args.hybrid_levels
            self.hybrid_levels = hybrid_levels
            self.hashgrid_levels = num_levels_total - hybrid_levels
            
            if self.hashgrid_levels <= 0:
                raise ValueError(f'[RESIDUAL_HYBRID] residual_hybrid requires hashgrid_levels > 0, got {self.hashgrid_levels} (total={num_levels_total}, hybrid={hybrid_levels})')
            
            # Use finest levels: [hybrid_levels:total_levels]
            selected_resolutions = all_resolutions[hybrid_levels:]
            base_res = selected_resolutions[0]
            finest_res = selected_resolutions[-1]
            
            # Calculate growth rate for the shrunken hashgrid
            if self.hashgrid_levels > 1:
                hash_growth_rate = np.exp((np.log(finest_res) - np.log(base_res)) / (self.hashgrid_levels - 1))
            else:
                hash_growth_rate = 1.0
            
            print(f'[RESIDUAL_HYBRID MODE] Total levels: {num_levels_total}, Hybrid levels: {hybrid_levels}')
            print(f'[RESIDUAL_HYBRID MODE] Per-Gaussian: Full SH (degree 0-3) → RGB')
            print(f'[RESIDUAL_HYBRID MODE] Hashgrid: {self.hashgrid_levels} levels × {self.level_dim}D = {self.hashgrid_levels * self.level_dim}D')
            print(f'[RESIDUAL_HYBRID MODE] Hashgrid resolutions: {selected_resolutions}')
            
            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=self.hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_res,
                finest_resolution=finest_res,
                init_mode='uniform',
                per_level_scale=hash_growth_rate,
                range=self.voxel_range,
            )
            
            print('[RESIDUAL_HYBRID MODE] Hash config:', config)
            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = selected_resolutions
            self.hashgrid_disabled = False
            
            # Set active levels (no C2F)
            self.active_hashgrid_levels = self.hashgrid_levels

        # 3D/3D_direct/3D_direct_fused mode: cat-style split with hybrid_levels for per-Gaussian, rest for hashgrid
        # Hash features → PyTorch pipeline → SH coefficients (3D) or direct RGB (3D_direct/3D_direct_fused)
        elif (self.is_3D_mode or self.is_3D_direct_mode or self.is_3D_direct_fused_mode) and self.hybrid_levels > 0:
            # Use finest (total - hybrid) levels for hashgrid, just like cat mode
            self.hashgrid_levels = num_levels_total - self.hybrid_levels
            self.hashgrid_disabled = False

            if self.hashgrid_levels <= 0:
                # Edge case: all levels are per-Gaussian features, no hashgrid
                mode_name = "3D_DIRECT_FUSED" if self.is_3D_direct_fused_mode else ("3D_DIRECT" if self.is_3D_direct_mode else "3D")
                print(f'[{mode_name} MODE] hybrid_levels={self.hybrid_levels} >= total_levels={num_levels_total}')
                print(f'[{mode_name} MODE] Using only per-Gaussian features (no hashgrid)')
                self.hash_encoding = None
                self.hashgrid_disabled = True
                self.resolutions = []
                self.active_hashgrid_levels = 0
            else:
                # Normal 3D mode: use finest levels for hashgrid
                hashgrid_resolutions = all_resolutions[-self.hashgrid_levels:]
                base_resolution = hashgrid_resolutions[0]
                finest_resolution = hashgrid_resolutions[-1]

                # Calculate growth rate for the hashgrid
                if self.hashgrid_levels > 1:
                    hash_growth_rate = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (self.hashgrid_levels - 1))
                else:
                    hash_growth_rate = 1.0

                config = SimpleNamespace(
                    device="cuda",
                    otype="HashGrid",
                    n_levels=self.hashgrid_levels,
                    n_features_per_level=cfg_encoding.hashgrid.dim,
                    log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                    base_resolution=base_resolution,
                    finest_resolution=finest_resolution,
                    init_mode='uniform',
                    per_level_scale=hash_growth_rate,
                    range=self.voxel_range,
                )

                mode_name = "3D_DIRECT_FUSED" if self.is_3D_direct_fused_mode else ("3D_DIRECT" if self.is_3D_direct_mode else "3D")
                print(f'[{mode_name} MODE] Cat-style hashgrid configuration:')
                print(f'  Total levels: {num_levels_total}')
                print(f'  Hybrid (per-Gaussian) levels: {self.hybrid_levels}')
                print(f'  Hashgrid levels: {self.hashgrid_levels}')
                print(f'  Resolutions: {base_resolution} -> {finest_resolution}')
                print(f'  Features per level: {cfg_encoding.hashgrid.dim}D')
                print(f'  Hashgrid size: 2^{cfg_encoding.hashgrid.dict_size}')
                print(f'  Per-Gaussian features: {self.hybrid_levels}×{cfg_encoding.hashgrid.dim} = {self.hybrid_levels * cfg_encoding.hashgrid.dim}D')
                print(f'  Hash features: {self.hashgrid_levels}×{cfg_encoding.hashgrid.dim} = {self.hashgrid_levels * cfg_encoding.hashgrid.dim}D')
                print(f'  MLP input: {num_levels_total * cfg_encoding.hashgrid.dim}D (cat-style concat)')

                self.hash_encoding = register_GridEncoder(config)
                self.resolutions = hashgrid_resolutions

                # Set active levels (all hashgrid levels active, no C2F for 3D)
                self.active_hashgrid_levels = self.hashgrid_levels

        else:
            # Baseline mode: use all levels
            self.hashgrid_levels = num_levels_total
            self.hashgrid_disabled = False
            
            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=cfg_encoding.levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=2**cfg_encoding.hashgrid.min_logres,
                finest_resolution=2**cfg_encoding.hashgrid.max_logres,
                init_mode='uniform',
                per_level_scale=self.growth_rate,
                range=self.voxel_range,
            )
            
            print('hash config:', config)
            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = all_resolutions
        
        print(f'hash resolution : {self.resolutions}')
        print(f'init activate level {cfg_encoding.coarse2fine.init_active_level}')

        # encoding_dim calculation
        if self.is_hybrid_sh_mode or self.is_hybrid_sh_raw_mode or self.is_hybrid_sh_post_mode:
            encoding_dim = 3  # 3D features for DC residuals
        elif self.is_residual_hybrid_mode:
            encoding_dim = self.hashgrid_levels * self.level_dim  # Reduced input (only hashgrid levels)
        else:
            encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels

        self.level_mask = cfg_encoding.coarse2fine.enabled
        # Override C2F for non-baseline methods (baseline always uses C2F from config)
        # disable_c2f is auto-set for all non-baseline methods in __init__
        if self.is_baseline_mode:
            # Baseline mode: always respect config, ignore disable_c2f flag
            print(f'If coarse2fine : {self.level_mask} (baseline mode)')
            if self.level_mask:
                self.init_active_level = cfg_encoding.coarse2fine.init_active_level
                self.step = cfg_encoding.coarse2fine.step
        elif self.disable_c2f:
            # Non-baseline modes: C2F is disabled
            print(f'If coarse2fine : False (disabled for {self.method} mode)')
            self.level_mask = False
            self.init_active_level = 1
            self.step = 1000  # Dummy value
        else:
            print(f'If coarse2fine : {self.level_mask}')
            if self.level_mask:
                self.init_active_level = cfg_encoding.coarse2fine.init_active_level
                self.step = cfg_encoding.coarse2fine.step

        # Initialize active_hashgrid_levels (will be updated by set_active_levels)
        if not hasattr(self, 'active_hashgrid_levels'):
            self.active_hashgrid_levels = self.hashgrid_levels

        return encoding_dim

    def build_view_enc(self, cfg_view_enc):
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": cfg_view_enc.type,
                "degree": cfg_view_enc.degree,
            },
        )

    def build_mlp(self, cfg_rgb, input_dim, output_dim = 3):
        cfg_mlp = cfg_rgb.mlp
        return tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": "MLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg_mlp.hidden_dim,
                "n_hidden_layers": cfg_mlp.num_layers,
            },
        )

    def forward(self, points_3D, with_xyz = False):
        # hash table features
        feat = self._encode_3D(points_3D)
        ### using coarse to fine ingp train
        if self.level_mask:
            mask = self._get_coarse2fine_mask(feat)
            feat = feat * mask
        
        if with_xyz:
            feat = torch.cat([feat, points_3D], dim=-1)  # [B,R,N,LD+3]

        return feat.float()

    def get_color(self, feat, ray_unit=None):
        if self.encoder_dir :
            enc_dir = self._encode_view(ray_unit)
            feat = torch.cat([feat, enc_dir], dim=-1)

        h = self.mlp_rgb(feat).float()
        rgb = torch.sigmoid(h)[:, :3]
        return rgb
    
    def rgb_decode(self, features, ray_unit):
        # Diffuse/Specular mode: features are already RGB from SH, just return them
        if self.is_diffuse_mode or self.is_specular_mode:
            return features
        # Diffuse_ngp and baseline: decode features through MLP
        rgb = self.get_color(features, ray_unit)
        return rgb  
        
    def set_active_levels(self, current_iter=None):
        self.current_optimizer = self.optimizer

        anneal_levels = max((current_iter - self.initialize - self.warm_up) // self.step, 0)
        
        # Diffuse/Specular mode: no hashgrid levels, just SH
        if self.is_diffuse_mode or self.is_specular_mode:
            self.active_levels = 0
            self.active_hashgrid_levels = 0
            self.optim_gaussian = True
            return 0

        # 3D/3D_direct mode: all levels active (no C2F), hash encoding is for intersection positions
        if self.is_3D_mode or self.is_3D_direct_mode:
            self.active_levels = self.levels
            self.active_hashgrid_levels = self.levels
            self.optim_gaussian = True
            return self.active_levels
        
        # Diffuse_ngp/diffuse_offset mode: disable C2F, use all levels immediately
        if self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode:
            self.active_levels = self.levels
            self.active_hashgrid_levels = self.levels
            # Skip C2F annealing - all levels active from start
        elif self.is_adaptive_cat_mode:
            # Adaptive_cat mode: All hashgrid levels active (no C2F)
            self.active_levels = self.levels  # Total levels (for MLP input size)
            self.active_hashgrid_levels = self.hashgrid_levels  # All hashgrid levels active
            self.optim_gaussian = True  # Train Gaussians throughout
        elif self.is_adaptive_zero_mode:
            # Adaptive_zero mode: All hashgrid levels active (no C2F)
            self.active_levels = self.levels  # Total levels (for MLP input size)
            self.active_hashgrid_levels = self.hashgrid_levels  # All hashgrid levels active
            self.optim_gaussian = True  # Train Gaussians throughout
        elif self.is_adaptive_gate_mode:
            # Adaptive_gate mode: VQ-AD gating, all hashgrid levels active (no C2F)
            self.active_levels = self.levels  # Total levels (for MLP input size)
            self.active_hashgrid_levels = self.hashgrid_levels  # All hashgrid levels active
            self.optim_gaussian = True  # Train Gaussians throughout
        elif self.is_3D_direct_fused_mode:
            # 3D_direct_fused mode: Fused in-kernel MLP, all hashgrid levels active (no C2F)
            # MLP is built expecting fixed input dimensions, can't use C2F
            self.active_levels = self.levels  # Total levels (for MLP input size)
            self.active_hashgrid_levels = self.hashgrid_levels  # All hashgrid levels active
            self.optim_gaussian = True  # Train Gaussians throughout
        elif self.is_residual_hybrid_mode:
            # Residual_hybrid mode: No C2F, all hashgrid levels active from start
            self.active_levels = self.hashgrid_levels
            self.active_hashgrid_levels = self.hashgrid_levels
            self.optim_gaussian = True  # Train Gaussians throughout
        elif self.is_cat_mode and self.hybrid_levels > 0:
            # Cat mode C2F:
            # - Per-Gaussian features: Always fully active (all hybrid_levels)
            # - Hashgrid: Start with 0 levels if hybrid_levels >= init_active_level, then C2F to hashgrid_levels
            # This ensures Gaussians train alone first, then hashgrid is gradually added
            if self.disable_c2f:
                # C2F disabled: all hashgrid levels active from start
                self.active_hashgrid_levels = self.hashgrid_levels
            elif self.hybrid_levels >= self.init_active_level:
                # Gaussian features cover the "coarse" levels, start hashgrid from 0
                init_hashgrid_levels = 0
                self.active_hashgrid_levels = min(self.hashgrid_levels, anneal_levels + init_hashgrid_levels)
            else:
                # hybrid_levels < init_active_level: start with some hashgrid levels
                init_hashgrid_levels = min(self.init_active_level - self.hybrid_levels, self.hashgrid_levels)
                self.active_hashgrid_levels = min(self.hashgrid_levels, anneal_levels + init_hashgrid_levels)
            # active_levels represents total active output levels (for MLP input size tracking)
            self.active_levels = self.hybrid_levels + self.active_hashgrid_levels
        else:
            # Baseline mode: C2F from init_active_level → total_levels
            self.active_levels = min(self.levels, anneal_levels + self.init_active_level)
            self.active_hashgrid_levels = self.active_levels  # Same as active_levels in baseline

        self.pre_level = self.active_levels
        
        if current_iter >= self.switch_iter and current_iter < self.switch_iter + self.keep_geometry:
            self.optim_gaussian = False
        elif current_iter >= self.initialize and current_iter < self.initialize + self.warm_up:
            # Cat mode: never freeze Gaussians during warm-up
            # Per-Gaussian features need to train from the start since they handle coarse levels
            if self.is_cat_mode:
                self.optim_gaussian = True
            else:
                self.optim_gaussian = False
        elif self.is_diffuse_ngp_mode and current_iter < self.initialize + 2000:
            # Diffuse_ngp mode: freeze gaussians for first 2k iterations, only train hashgrid
            self.optim_gaussian = False
        elif self.is_diffuse_offset_mode and current_iter < self.initialize + 1000:
            # Diffuse_offset mode: freeze gaussians for first 1k iterations, only train hashgrid
            self.optim_gaussian = False
        else:
            self.optim_gaussian = True
        
        return self.active_levels
        
    def set_epsilon(self):
        
        epsilon_res = self.resolutions[self.active_levels - 1]
        self.level_eps = 1. / epsilon_res

    @torch.no_grad()
    def _get_coarse2fine_mask(self, points_enc):
        mask = torch.zeros_like(points_enc)
        mask[..., :(self.active_levels * self.level_dim)] = 1
        return mask

    def _encode_3D(self, points_3D):
        # Tri-linear interpolate the corresponding embeddings from the dictionary.
        vol_min, vol_max = self.voxel_range

        if self.contract == False:
            points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
        else:
            ### with contract function
            ### this part should be the same with "query_feature" function in CUDA
            vmid = (vol_min + vol_max) * 0.5
            vsize_ = (vol_max - vol_min) * 0.5
            points_3D_normalized = (points_3D - vmid) / vsize_

            norm = points_3D_normalized.norm(dim=-1, keepdim=True)  # Compute the norm of points_3D
            inv_norm = 1.0 / norm
            scale_trans = torch.where(norm <= 1.0, torch.ones_like(norm), (2.0 - inv_norm) * inv_norm)

            # Apply the scaling transformation
            points_3D_normalized = points_3D_normalized * scale_trans  # Warp to range [-2, 2] for outside region

            # Normalize to [0, 1]
            points_3D_normalized = (points_3D_normalized + 2.0) * 0.25

        xyz_input = points_3D_normalized.view(-1, 3)

        feat_output = self.hash_encoding(xyz_input)#, eps = self.level_eps)

        points_enc = feat_output.view(*points_3D_normalized.shape[:-1], feat_output.shape[-1])
        return points_enc

    def _encode_view(self, d):
        d = (d+1) / 2
        d = self.encoder_dir(d)
        return d

    def get_fused_mlp_weights(self):
        """
        Extract MLP weights for CUDA global memory upload (bias-free).
        Returns (W1, W2, W3) in the format expected by set_mlp_weights.

        The CUDA kernel uses: mlp_W1[h * 41 + i]
        - h = hidden neuron index (row), i = input index (column)
        - This is row-major access of a [HIDDEN_DIM, IN_DIM] = [32, 41] matrix
        - PyTorch Linear(41, 32, bias=False) stores weight as [32, 41]
        - So NO transpose needed - PyTorch format matches CUDA row-major layout
        - Layer 1 column 41 acts as implicit bias (input padded with 1.0 in CUDA)
        """
        if self.mlp_fused is None:
            return None

        # mlp_fused is nn.Sequential with layers: Linear, ReLU, Linear, ReLU, Linear
        # Indices: [0]=Linear1, [1]=ReLU, [2]=Linear2, [3]=ReLU, [4]=Linear3
        W1 = self.mlp_fused[0].weight.data  # [32, 41] - col 41 = implicit bias
        W2 = self.mlp_fused[2].weight.data  # [32, 32]
        W3 = self.mlp_fused[4].weight.data  # [3, 32]

        # NO transpose - PyTorch [out, in] is already row-major [out][in]
        # which matches CUDA's W[h * in_dim + i] access pattern

        # Pad weights for WMMA alignment (Tensor Core modes)
        if self.is_3D_SH_res_mode or self.is_3D_SH_cat_mode:
            # 3D_SH_res / 3D_SH_cat: All weights are [16, 16] — already WMMA-aligned
            # But W1 might be smaller if input_dim+1 < 16, so pad
            import torch
            actual_input_cols = W1.shape[1]  # e.g., mlp_input_padded (should be 16)
            if actual_input_cols < 16:
                W1_padded = torch.zeros(16, 16, device=W1.device, dtype=W1.dtype)
                W1_padded[:W1.shape[0], :W1.shape[1]] = W1
                W1 = W1_padded
            # W3 output: [16, 16] — only first 3 rows used as RGB residual
            actual_output_rows = W3.shape[0]
            if actual_output_rows < 16:
                W3_padded = torch.zeros(16, 16, device=W3.device, dtype=W3.dtype)
                W3_padded[:W3.shape[0], :W3.shape[1]] = W3
                W3 = W3_padded
            return W1.contiguous(), W2.contiguous(), W3.contiguous()
        elif self.is_3D_direct_sh_tc_mode:
            import torch
            # SH mode: W1[32,25] → pad to [32,32], W3[48,32] → already aligned
            W1_padded = torch.zeros(32, 32, device=W1.device, dtype=W1.dtype)
            W1_padded[:, :25] = W1
            # W3 is [48, 32] — already WMMA-aligned, no padding needed
            return W1_padded.contiguous(), W2.contiguous(), W3.contiguous()
        elif self.is_3D_direct_tc_mode:
            import torch
            W1_padded = torch.zeros(32, 48, device=W1.device, dtype=W1.dtype)
            W1_padded[:, :41] = W1
            W3_padded = torch.zeros(16, 32, device=W3.device, dtype=W3.dtype)
            W3_padded[:3, :] = W3
            return W1_padded.contiguous(), W2.contiguous(), W3_padded.contiguous()

        return W1.contiguous(), W2.contiguous(), W3.contiguous()

    def _copy_tcnn_to_pytorch_mlp(self, checkpoint_state_dict=None):
        """
        Copy weights from tcnn MLP (mlp_rgb or mlp_3D_direct) to PyTorch MLP (mlp_fused).

        tcnn stores weights with padding (dimensions padded to multiples of 16):
        - Input: 40 → 48 (padded with 1s)
        - Hidden: 32 (no padding needed)
        - Output: 3 → 16 (padded)

        tcnn layout (NO BIASES, weights only):
        - W1: [32, 48] = 1536 floats
        - W2: [32, 32] = 1024 floats
        - W3: [16, 32] = 512 floats (we use [3, 32] portion)
        Total: 3072 floats

        Our bias-free architecture:
        - W1: [32, 41] — columns 0-39 from tcnn W1[:, 0:40], column 40 = sum(tcnn W1[:, 40:48])
        - W2: [32, 32] — direct copy
        - W3: [3, 32] — slice from tcnn's [16, 32]

        Args:
            checkpoint_state_dict: Optional state_dict from checkpoint. If provided and
                contains mlp_3D_direct.params, extract weights from there directly.
        """
        if self.mlp_fused is None:
            print("[WARN] mlp_fused is None, cannot copy weights")
            return

        # First priority: extract directly from checkpoint if mlp_3D_direct is there
        params = None
        source_name = None

        if checkpoint_state_dict is not None:
            # Check for mlp_3D_direct.params in checkpoint
            for key, value in checkpoint_state_dict.items():
                if 'mlp_3D_direct.params' in key:
                    params = value.cpu()
                    source_name = f"checkpoint[{key}]"
                    break

        # Fallback to existing tcnn MLPs in model
        if params is None:
            if hasattr(self, 'mlp_3D_direct') and self.mlp_3D_direct is not None:
                params = self.mlp_3D_direct.params.data.cpu()
                source_name = "mlp_3D_direct"
            elif self.mlp_rgb is not None:
                params = self.mlp_rgb.params.data.cpu()
                source_name = "mlp_rgb"
            else:
                print("[WARN] No source tcnn MLP found, mlp_fused will use random init")
                return

        print(f"[COPY MLP] Source: {source_name}, params shape: {params.shape}")

        # Actual dimensions
        in_dim = 40
        hidden_dim = 32
        out_dim = 3

        # tcnn padded dimensions (multiples of 16)
        in_dim_padded = 48  # 40 → 48
        out_dim_padded = 16  # 3 → 16

        offset = 0

        # Layer 1: tcnn stores as [hidden, in_padded] = [32, 48]
        # CRITICAL: tcnn pads input with 1s (not zeros)!
        # W1_tcnn[:, 40:48] @ [1,1,...,1] acts as implicit bias.
        # Our bias-free format: W1[32, 41] where col 40 = sum(tcnn padding columns)
        w1_size = in_dim_padded * hidden_dim
        W1_full = params[offset:offset+w1_size].view(hidden_dim, in_dim_padded)
        W1_weights = W1_full[:, :in_dim]  # [32, 40] actual weights
        b1_implicit = W1_full[:, in_dim:].sum(dim=1, keepdim=True)  # [32, 1] implicit bias
        W1 = torch.cat([W1_weights, b1_implicit], dim=1).contiguous()  # [32, 41]
        offset += w1_size

        # Layer 2: tcnn stores as [hidden, hidden] = [32, 32]
        w2_size = hidden_dim * hidden_dim
        W2 = params[offset:offset+w2_size].view(hidden_dim, hidden_dim).contiguous()
        offset += w2_size

        # Layer 3: tcnn stores as [out_padded, hidden] = [16, 32], slice to [3, 32]
        w3_size = hidden_dim * out_dim_padded
        W3 = params[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :].contiguous()

        print(f"[COPY MLP] Extracted weights: W1{list(W1.shape)}, W2{list(W2.shape)}, W3{list(W3.shape)}")
        print(f"[COPY MLP] Implicit bias (W1[:, 40]): mean={b1_implicit.mean():.4f}")

        # Copy to mlp_fused (indices: 0=Linear1, 1=ReLU, 2=Linear2, 3=ReLU, 4=Linear3)
        with torch.no_grad():
            self.mlp_fused[0].weight.data.copy_(W1.cuda())  # [32, 41] — col 40 = implicit bias
            self.mlp_fused[2].weight.data.copy_(W2.cuda())
            self.mlp_fused[4].weight.data.copy_(W3.cuda())

        print(f"[COPY MLP] Successfully copied {source_name} weights to mlp_fused (bias-free, implicit bias in col 40)")

    def _copy_tcnn_to_pytorch_mlp_3D_direct(self, checkpoint_state_dict=None):
        """
        Copy weights from tcnn MLP (mlp_3D_direct or mlp_rgb) to PyTorch MLP (mlp_3D_direct).

        tcnn stores weights with padding (dimensions padded to multiples of 16):
        - Input: 40 → 48 (padded)
        - Hidden: 32 (no padding needed)
        - Output: 3 → 16 (padded)

        tcnn layout (NO BIASES, weights only):
        - W1: [48, 32] = 1536 floats (we use [40, 32] portion)
        - W2: [32, 32] = 1024 floats
        - W3: [32, 16] = 512 floats (we use [32, 3] portion)
        Total: 3072 floats

        CRITICAL: tcnn pads input with 1s (not zeros)! So W1[:, 40:48] @ [1,...] acts as bias.
        We extract: b1 = sum(W1[:, 40:48], dim=1)
        """
        if self.mlp_3D_direct is None:
            print("[WARN] mlp_3D_direct is None, cannot copy weights")
            return

        # First priority: extract directly from checkpoint if mlp_3D_direct.params is there
        params = None
        source_name = None

        if checkpoint_state_dict is not None:
            # Check for mlp_3D_direct.params (old tcnn format)
            for key, value in checkpoint_state_dict.items():
                if 'mlp_3D_direct.params' in key:
                    params = value.cpu()
                    source_name = f"checkpoint[{key}]"
                    break

            # Fallback: try mlp_rgb.params (CAT checkpoint)
            if params is None:
                for key, value in checkpoint_state_dict.items():
                    if 'mlp_rgb.params' in key:
                        params = value.cpu()
                        source_name = f"checkpoint[{key}]"
                        break

        # Fallback to existing tcnn MLP in model (shouldn't happen with new code)
        if params is None:
            if self.mlp_rgb is not None and hasattr(self.mlp_rgb, 'params'):
                params = self.mlp_rgb.params.data.cpu()
                source_name = "mlp_rgb.params"
            else:
                print("[WARN] No source tcnn MLP found, mlp_3D_direct will use random init")
                return

        print(f"[COPY MLP] Source: {source_name}, params shape: {params.shape}")

        # Actual dimensions
        in_dim = 40
        hidden_dim = 32
        out_dim = 3

        # tcnn padded dimensions (multiples of 16)
        in_dim_padded = 48  # 40 → 48
        out_dim_padded = 16  # 3 → 16

        # tcnn stores weights as [out, in] - same as PyTorch Linear format!
        offset = 0

        # Layer 1: tcnn stores as [hidden, in_padded] = [32, 48]
        w1_size = in_dim_padded * hidden_dim
        W1_full = params[offset:offset+w1_size].view(hidden_dim, in_dim_padded)
        W1 = W1_full[:, :in_dim].contiguous()
        b1 = W1_full[:, in_dim:].sum(dim=1)  # Implicit bias from tcnn padding
        offset += w1_size

        # Layer 2: tcnn stores as [hidden, hidden] = [32, 32]
        w2_size = hidden_dim * hidden_dim
        W2 = params[offset:offset+w2_size].view(hidden_dim, hidden_dim).contiguous()
        offset += w2_size

        # Layer 3: tcnn stores as [out_padded, hidden] = [16, 32], slice to [3, 32]
        w3_size = hidden_dim * out_dim_padded
        W3 = params[offset:offset+w3_size].view(out_dim_padded, hidden_dim)[:out_dim, :].contiguous()

        print(f"[COPY MLP] Extracted weights: W1{list(W1.shape)}, W2{list(W2.shape)}, W3{list(W3.shape)}")
        print(f"[COPY MLP] Implicit bias b1 from tcnn padding: mean={b1.mean():.4f}")

        # Copy to mlp_3D_direct (indices: 0=Linear1, 1=ReLU, 2=Linear2, 3=ReLU, 4=Linear3)
        with torch.no_grad():
            self.mlp_3D_direct[0].weight.data.copy_(W1.cuda())
            self.mlp_3D_direct[0].bias.data.copy_(b1.cuda())
            self.mlp_3D_direct[2].weight.data.copy_(W2.cuda())
            self.mlp_3D_direct[2].bias.data.zero_()  # No bias for hidden layer in tcnn
            self.mlp_3D_direct[4].weight.data.copy_(W3.cuda())
            self.mlp_3D_direct[4].bias.data.zero_()  # No bias for output layer

        print(f"[COPY MLP] Successfully copied {source_name} to mlp_3D_direct (PyTorch)")

    def initialize_weights(self):
        
        for m in self.mlp_rgb.linears:
            if isinstance(m, torch.nn.Linear):
                torch_init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch_init.zeros_(m.bias)
    
    def save_model(self, exp_path, iteration):

        state = {
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict()
        }
        save_path = os.path.join(exp_path, f'ngp_{iteration}.pth')
        torch.save(state, save_path)
        print(f"save ingp model at {save_path}")

    def load_model(self, exp_path, iteration):

        checkpoint = torch.load(os.path.join(exp_path, f'ngp_{iteration}.pth'))

        # Check if we're loading a CAT model into 3D_direct mode
        # In this case, mlp_3D_direct exists but won't be in the checkpoint
        has_mlp_3D_direct = self.mlp_3D_direct is not None
        checkpoint_has_mlp_3D_direct = any('mlp_3D_direct' in k for k in checkpoint['model_state_dict'].keys())

        # Check if we're loading a CAT model into 3D_direct_fused mode
        has_mlp_fused = self.mlp_fused is not None
        checkpoint_has_mlp_fused = any('mlp_fused' in k for k in checkpoint['model_state_dict'].keys())

        # Check if checkpoint has old tcnn format for mlp_3D_direct
        checkpoint_has_tcnn_mlp_3D_direct = any('mlp_3D_direct.params' in k for k in checkpoint['model_state_dict'].keys())
        # Check if checkpoint has new PyTorch format for mlp_3D_direct
        checkpoint_has_pytorch_mlp_3D_direct = any('mlp_3D_direct.0.weight' in k for k in checkpoint['model_state_dict'].keys())

        # Load with strict=False to allow missing keys
        if has_mlp_3D_direct and not checkpoint_has_mlp_3D_direct:
            # Loading CAT model into 3D_direct mode - need to convert mlp_rgb weights
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[3D_DIRECT] Loaded CAT model - initializing mlp_3D_direct from mlp_rgb weights")

            # Copy mlp_rgb weights (tcnn) to mlp_3D_direct (PyTorch)
            if self.mlp_rgb is not None:
                self._copy_tcnn_to_pytorch_mlp_3D_direct(checkpoint['model_state_dict'])
        elif has_mlp_3D_direct and checkpoint_has_tcnn_mlp_3D_direct:
            # Loading old tcnn 3D_direct checkpoint into new PyTorch 3D_direct mode
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[3D_DIRECT] Loading old tcnn checkpoint - converting to PyTorch MLP")
            self._copy_tcnn_to_pytorch_mlp_3D_direct(checkpoint['model_state_dict'])
        elif has_mlp_fused and not checkpoint_has_mlp_fused:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[3D_DIRECT_FUSED] Loaded CAT/3D_direct model - copying MLP weights to mlp_fused")
            # Copy weights from checkpoint's mlp_3D_direct or mlp_rgb (tcnn) to mlp_fused (PyTorch)
            # Pass checkpoint_state_dict so we can extract mlp_3D_direct.params directly
            # (since mlp_3D_direct doesn't exist in fused mode)
            self._copy_tcnn_to_pytorch_mlp(checkpoint_state_dict=checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint['model_state_dict'])

        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"loading ingp model from {os.path.join(exp_path, f'ngp_{iteration}.pth')}")
        
