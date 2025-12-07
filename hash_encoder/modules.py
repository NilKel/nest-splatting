
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

        # Store args for cat mode
        self.args = args
        self.is_cat_mode = args is not None and hasattr(args, 'method') and args.method == "cat"
        self.hybrid_levels = args.hybrid_levels if self.is_cat_mode and hasattr(args, 'hybrid_levels') else 0
        self.cat_coarse2fine = args.cat_coarse2fine if self.is_cat_mode and hasattr(args, 'cat_coarse2fine') else False
        
        self.build_encoding(cfg_model.encoding)
        # In cat mode, MLP input is still total_levels * dim (to match baseline architecture)
        # But hashgrid only outputs (total_levels - hybrid_levels) * dim
        self.feat_dim = cfg_model.encoding.levels * cfg_model.encoding.hashgrid.dim # + 3

        self.mlp_rgb = self.build_mlp(cfg_model.rgb, input_dim=self.feat_dim + view_enc_dir, output_dim = 3)
        
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

        l = []
        
        # Only add hash_encoding parameters if it exists (not disabled)
        if self.hash_encoding is not None:
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})
        
        l.append({'params': self.mlp_rgb.parameters(), 'lr': lr_mlp_rgb, "name": "rgb_mlp"})

        self.optimizer = torch.optim.Adam(l, betas=(0.9, 0.99), eps=1e-15)

    def build_encoding(self, cfg_encoding):
        assert(cfg_encoding.type == "hashgrid")
        self.voxel_range = cfg_encoding.hashgrid.range
        self.gridrange = torch.tensor(self.voxel_range).cuda().float()

        l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels_total = cfg_encoding.levels
        self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels_total - 1))
        
        # Calculate all resolutions first (for both baseline and cat mode)
        all_resolutions = []
        for lv in range(0, num_levels_total):
            size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
            all_resolutions.append(size)
        
        print(f'Baseline hash resolution : {all_resolutions}')
        
        # In cat mode, use only higher-frequency levels (skip first hybrid_levels)
        if self.is_cat_mode:
            num_hashgrid_levels = num_levels_total - self.hybrid_levels
            
            # Edge case: hybrid_levels = 0 means all hashgrid (like baseline)
            if self.hybrid_levels == 0:
                print('[CAT MODE] hybrid_levels=0: Using all hashgrid levels (no per-Gaussian features)')
                num_hashgrid_levels = num_levels_total
                selected_resolutions = all_resolutions
                base_res = 2 ** cfg_encoding.hashgrid.min_logres
                finest_res = selected_resolutions[-1]
                hash_growth_rate = self.growth_rate
            # Edge case: hybrid_levels = total_levels means all Gaussian features (no hashgrid)
            elif self.hybrid_levels >= num_levels_total:
                print(f'[CAT MODE] hybrid_levels={self.hybrid_levels}: Using all per-Gaussian features (no hashgrid)')
                # No hashgrid needed - will skip creation below
                num_hashgrid_levels = 0
                selected_resolutions = []
                base_res = 0
                finest_res = 0
                hash_growth_rate = 1.0
            else:
                # Normal case: split between Gaussian and hashgrid
                # Use resolutions from [hybrid_levels:] (e.g., if hybrid_levels=2, use levels 2,3,4,5)
                selected_resolutions = all_resolutions[self.hybrid_levels:]
                base_res = selected_resolutions[0]
                finest_res = selected_resolutions[-1]
                # Recalculate growth rate for the selected levels
                hash_growth_rate = np.exp((np.log(finest_res) - np.log(base_res)) / (num_hashgrid_levels - 1)) if num_hashgrid_levels > 1 else 1.0
            
            print(f'[CAT MODE] Using {num_hashgrid_levels} hashgrid levels (skipping first {self.hybrid_levels})')
            print(f'[CAT MODE] Per-Gaussian features: {self.hybrid_levels} levels × 4 dim = {self.hybrid_levels * 4}D')
            print(f'[CAT MODE] Hashgrid features: {num_hashgrid_levels} levels × 4 dim = {num_hashgrid_levels * 4}D')
            print(f'[CAT MODE] Selected hash resolutions: {selected_resolutions}')
        else:
            # Baseline/add mode: use all levels
            num_hashgrid_levels = num_levels_total
            base_res = 2 ** cfg_encoding.hashgrid.min_logres
            finest_res = all_resolutions[-1]
            hash_growth_rate = self.growth_rate
            selected_resolutions = all_resolutions
        
        self.level_dim = cfg_encoding.hashgrid.dim
        self.levels = cfg_encoding.levels  # Keep original total levels for compatibility
        self.actual_hash_levels = num_hashgrid_levels  # Actual hashgrid levels

        # Track if hashgrid should be ignored (when hybrid_levels >= total_levels)
        self.hashgrid_disabled = self.is_cat_mode and self.hybrid_levels >= cfg_encoding.levels

        # Only create hashgrid if we actually need it
        if num_hashgrid_levels > 0:
            tcnn_config = dict(
                otype="HashGrid",
                n_levels=num_hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_res,
                per_level_scale=hash_growth_rate,
            )

            config = SimpleNamespace(
                device="cuda",
                otype="HashGrid",
                n_levels=num_hashgrid_levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=base_res,
                finest_resolution=finest_res,
                init_mode='uniform',
                per_level_scale=hash_growth_rate,
                range=self.voxel_range,
            )

            print('hash config:', config)
            print(f'init activate level {cfg_encoding.coarse2fine.init_active_level}')

            self.hash_encoding = register_GridEncoder(config)
            self.resolutions = selected_resolutions
        else:
            # No hashgrid - create a dummy None encoder
            print('[CAT MODE] Skipping hashgrid creation - using only per-Gaussian features')
            self.hash_encoding = None
            self.resolutions = []
        
        print(f'Active hash resolution : {self.resolutions}')
        
        encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels

        self.level_mask = cfg_encoding.coarse2fine.enabled
        print(f'If coarse2fine : {self.level_mask}')
        if self.level_mask:
            self.init_active_level = cfg_encoding.coarse2fine.init_active_level
            self.step = cfg_encoding.coarse2fine.step
            
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
        # Apply coarse-to-fine masking for cat mode
        if self.is_cat_mode and self.cat_coarse2fine:
            mask = self._get_cat_coarse2fine_mask(features)
            features = features * mask

        rgb = self.get_color(features, ray_unit)
        return rgb  
        
    def set_active_levels(self, current_iter=None):
        self.current_optimizer = self.optimizer

        if self.is_cat_mode:
            if self.cat_coarse2fine:
                # Cat mode with coarse-to-fine:
                # - Always use ALL Gaussian features (hybrid_levels)
                # - Start hashgrid at min(2, effective_hash_levels) and progressively unlock
                # Example: hybrid_levels=3, total_levels=6, effective_hash_levels=3
                # - Iter 0: 3 Gaussian + min(2,3)=2 Hash = 5 total levels (20D)
                # - Iter 3000: 3 Gaussian + 3 Hash = 6 total levels (24D)
                effective_hash_levels = self.levels - self.hybrid_levels
                init_hash_levels = min(self.init_active_level, effective_hash_levels)
                
                anneal_levels = max((current_iter - self.initialize - self.warm_up) // self.step, 0)
                active_hash_levels = min(effective_hash_levels, anneal_levels + init_hash_levels)
                
                # Total active levels = all Gaussian + active hashgrid
                self.active_levels = self.hybrid_levels + active_hash_levels

                # Print when a new level is enabled
                if self.active_levels != self.pre_level:
                    gaussian_dim = self.hybrid_levels * self.level_dim
                    hash_dim = active_hash_levels * self.level_dim
                    total_dim = self.active_levels * self.level_dim
                    print(f"[CAT C2F] Iter {current_iter}: Enabled level {self.active_levels}/{self.levels} - "
                          f"{self.hybrid_levels}/{self.hybrid_levels} Gaussian ({gaussian_dim}D) + {active_hash_levels}/{effective_hash_levels} Hash ({hash_dim}D) = {total_dim}D total")
            else:
                # Cat mode without coarse-to-fine: use all levels from start
                self.active_levels = self.levels  # All levels (6)
        else:
            # Baseline/add mode: use coarse-to-fine training
            anneal_levels = max((current_iter - self.initialize - self.warm_up) // self.step, 0)
            self.active_levels = min(self.levels, anneal_levels + self.init_active_level)

            # Print when a new level is enabled
            if self.active_levels != self.pre_level:
                mode_label = "Add" if hasattr(self.args, 'method') and self.args.method == "add" else "Baseline"
                active_dim = self.active_levels * self.level_dim
                print(f"[{mode_label} C2F] Iter {current_iter}: Enabled level {self.active_levels}/{self.levels} - "
                      f"Hashgrid & Gaussian features: {active_dim}D")

        self.pre_level = self.active_levels
        
        if current_iter >= self.switch_iter and current_iter < self.switch_iter + self.keep_geometry:
            self.optim_gaussian = False
        elif current_iter >= self.initialize and current_iter < self.initialize + self.warm_up:
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

    @torch.no_grad()
    def _get_cat_coarse2fine_mask(self, points_enc):
        """
        Create mask for cat mode coarse-to-fine training.
        Always uses ALL Gaussian features, progressively unlocks hashgrid.
        
        Since concatenation is [Gaussian features | Hashgrid features],
        active_levels = hybrid_levels + active_hash_levels
        
        Example with hybrid_levels=3, total_levels=6, effective_hash=3:
        - Iter 0: active_levels=5 (3 Gaussian + 2 Hash) → unmask 20D
        - Iter 3000: active_levels=6 (3 Gaussian + 3 Hash) → unmask 24D
        
        Edge case: hybrid_levels=6, total=6, effective_hash=0:
        - active_levels always 6 (all Gaussian, no hash) → unmask 24D from start
        """
        mask = torch.zeros_like(points_enc)
        active_dim = self.active_levels * self.level_dim
        mask[..., :active_dim] = 1
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

        if self.hash_encoding is not None:
            feat_output = self.hash_encoding(xyz_input)#, eps = self.level_eps)
            points_enc = feat_output.view(*points_3D_normalized.shape[:-1], feat_output.shape[-1])
        else:
            # No hashgrid - return empty features
            # This case is for hybrid_levels >= total_levels (all Gaussian features)
            points_enc = torch.zeros(*points_3D_normalized.shape[:-1], 0, device=xyz_input.device, dtype=xyz_input.dtype)
        
        return points_enc

    def _encode_view(self, d):
        d = (d+1) / 2 
        d = self.encoder_dir(d)
        return d
    
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
        self.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"loading ingp model from {os.path.join(exp_path, f'ngp_{iteration}.pth')}")
        
