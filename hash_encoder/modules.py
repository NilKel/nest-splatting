
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
        self.hybrid_levels = args.hybrid_levels if self.is_cat_mode and hasattr(args, 'hybrid_levels') else 0
        
        # Store args for adaptive mode configuration
        self.is_adaptive_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive"
        # Store args for adaptive_add mode configuration (weighted sum of per-Gaussian and hashgrid)
        self.is_adaptive_add_mode = args is not None and hasattr(args, 'method') and args.method == "adaptive_add"
        
        # Store args for diffuse mode configuration (per-Gaussian RGB, no viewdir, no hashgrid)
        self.is_diffuse_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse"
        # Store args for specular mode configuration (full 2DGS with SH, no hashgrid)
        self.is_specular_mode = args is not None and hasattr(args, 'method') and args.method == "specular"
        # Store args for diffuse_ngp mode (diffuse SH + hashgrid on unprojected depth)
        self.is_diffuse_ngp_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse_ngp"
        # Store args for diffuse_offset mode (diffuse SH as xyz offset for hashgrid query)
        self.is_diffuse_offset_mode = args is not None and hasattr(args, 'method') and args.method == "diffuse_offset"
        
        self.build_encoding(cfg_model.encoding)
        
        # Diffuse/Specular mode: no MLP needed, just SH-based RGB
        if self.is_diffuse_mode or self.is_specular_mode:
            self.feat_dim = 3  # RGB
            self.mlp_rgb = None
            self.view_dep = False  # View dependency handled by SH, not MLP
        elif self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode:
            # Diffuse_ngp/diffuse_offset mode: need MLP for hashgrid decoding
            # MLP input is hashgrid features + view encoding
            self.feat_dim = cfg_model.encoding.levels * cfg_model.encoding.hashgrid.dim
            self.mlp_rgb = self.build_mlp(cfg_model.rgb, input_dim=self.feat_dim + view_enc_dir, output_dim=3)
        else:
            # MLP input is always total_levels * dim (buffer size is constant)
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
        # Only add hash_encoding if it exists (not disabled in cat mode or diffuse mode)
        if self.hash_encoding is not None:
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})
        # Only add MLP if it exists (not diffuse mode)
        if self.mlp_rgb is not None:
            l.append({'params': self.mlp_rgb.parameters(), 'lr': lr_mlp_rgb, "name": "rgb_mlp"})

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
        
        encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels

        self.level_mask = cfg_encoding.coarse2fine.enabled
        # Diffuse_ngp/diffuse_offset: override C2F to disabled
        if self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode:
            print(f'If coarse2fine : False (disabled for diffuse_ngp/diffuse_offset mode)')
        else:
            print(f'If coarse2fine : {self.level_mask}')
        if self.level_mask:
            self.init_active_level = cfg_encoding.coarse2fine.init_active_level
            self.step = cfg_encoding.coarse2fine.step
        
        # Initialize active_hashgrid_levels (will be updated by set_active_levels)
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
        
        # Diffuse_ngp/diffuse_offset mode: disable C2F, use all levels immediately
        if self.is_diffuse_ngp_mode or self.is_diffuse_offset_mode:
            self.active_levels = self.levels
            self.active_hashgrid_levels = self.levels
            # Skip C2F annealing - all levels active from start
        elif self.is_cat_mode and self.hybrid_levels > 0:
            # Cat mode C2F:
            # - Per-Gaussian features: Always fully active (all hybrid_levels)
            # - Hashgrid: Start with 0 levels if hybrid_levels >= init_active_level, then C2F to hashgrid_levels
            # This ensures Gaussians train alone first, then hashgrid is gradually added
            if self.hybrid_levels >= self.init_active_level:
                # Gaussian features cover the "coarse" levels, start hashgrid from 0
                init_hashgrid_levels = 0
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
        
