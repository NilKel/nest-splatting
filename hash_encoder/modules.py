
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

    def __init__(self, cfg_model, method='baseline', hybrid_levels=3):
        super().__init__()  

        self.method = method  # 'baseline', 'surface', 'surface_rgb', 'hybrid_features', etc.
        self.hybrid_levels = hybrid_levels  # Number of finest hashgrid levels for hybrid_features mode
        self.view_dep = cfg_model.rgb.view_dep
        if self.view_dep:
            self.build_view_enc(cfg_model.rgb.encoding_view)
        
        view_enc_dir = 0 if not self.view_dep else self.encoder_dir.n_output_dims

        # Surface mode uses 3x features for vector potentials, baseline uses original
        original_dim = cfg_model.encoding.hashgrid.dim  # e.g., 4
        
        if method == 'surface':
            # Query 12 features (4 base × 3 for vectors), dot product reduces back to 4 per level
            cfg_model.encoding.hashgrid.dim = original_dim * 3  # 12 features per level in hashgrid
            print(f"[INGP] Surface mode: Hashgrid has {cfg_model.encoding.hashgrid.dim} features per level (will dot product to {original_dim})")
            self.build_encoding(cfg_model.encoding)
            self.level_dim = original_dim  # 4 per level (after dot product)
            self.feat_dim = cfg_model.encoding.levels * self.level_dim
            
        elif method == 'surface_blend':
            # Surface blend mode: use baseline 12D features, then dot product with blended normals in Python
            # Use dim=12 but ensure CUDA treats it as baseline (no dot product in CUDA)
            cfg_model.encoding.hashgrid.dim = original_dim * 3  # 12 features per level
            print(f"[INGP] Surface blend mode: Hashgrid has {cfg_model.encoding.hashgrid.dim} features per level")
            print(f"  - Features alpha-blended in CUDA (level × 12 dims)")
            print(f"  - Dot product with blended normals in Python → {original_dim} per level")
            self.build_encoding(cfg_model.encoding)
            self.level_dim = original_dim  # 4 per level (after Python dot product)
            self.feat_dim = cfg_model.encoding.levels * self.level_dim

        elif method == 'surface_depth':
            # Surface depth mode: identical to surface_blend but uses depth gradient instead of rendered normals
            cfg_model.encoding.hashgrid.dim = original_dim * 3  # 12 features per level
            print(f"[INGP] Surface depth mode: Hashgrid has {cfg_model.encoding.hashgrid.dim} features per level")
            print(f"  - Features alpha-blended in CUDA (level × 12 dims)")
            print(f"  - Dot product with depth gradient normals in Python → {original_dim} per level")
            self.build_encoding(cfg_model.encoding)
            self.level_dim = original_dim  # 4 per level (after Python dot product)
            self.feat_dim = cfg_model.encoding.levels * self.level_dim

        elif method == 'surface_rgb':
            # Two separate hashgrids: baseline features (4D) + surface features (12D→4D via dot)
            print(f"[INGP] Surface RGB mode: Creating two separate hashgrids")
            print(f"  - Baseline hashgrid: {original_dim} features per level (queried at Gaussian center, CONCATENATED → {original_dim * cfg_model.encoding.levels}D)")
            print(f"  - Surface hashgrid: {original_dim * 3} features per level (queried at intersection, dot product to {original_dim}, CONCATENATED → {original_dim * cfg_model.encoding.levels}D)")
            print(f"  - Combined: baseline + surface (element-wise) → {original_dim}D per level → MLP")
            
            # Build baseline hashgrid (4 features per level, concatenated)
            cfg_encoding_diffuse = cfg_model.encoding
            cfg_encoding_diffuse.hashgrid.dim = original_dim  # Changed from 3 to 4
            self.build_encoding_diffuse(cfg_encoding_diffuse)
            
            # Build surface hashgrid (12 features per level, dot product to 4, concatenated to 24)
            cfg_encoding_view = cfg_model.encoding
            cfg_encoding_view.hashgrid.dim = original_dim * 3  # 12 features per level
            self.build_encoding_view_features(cfg_encoding_view)
            
            self.level_dim = original_dim  # 4 per level (after combination)
            self.feat_dim = cfg_model.encoding.levels * self.level_dim  # MLP input: 24D for 6 levels × 4
            
        elif method == 'baseline_double':
            # Two separate 4D hashgrids: one at xyz, one at pk
            print(f"[INGP] baseline_double mode: Creating two 4D hashgrids")
            print(f"  - Hashgrid 1: {original_dim} features per level (queried at xyz intersection, CONCATENATED)")
            print(f"  - Hashgrid 2: {original_dim} features per level (queried at pk Gaussian center, CONCATENATED)")
            print(f"  - Combined: feat1 + feat2 (element-wise), then MLP")
            
            # Build main hashgrid (queried at xyz)
            cfg_model.encoding.hashgrid.dim = original_dim
            self.build_encoding(cfg_model.encoding)
            
            # Build second hashgrid (queried at pk)
            cfg_encoding_diffuse = cfg_model.encoding
            cfg_encoding_diffuse.hashgrid.dim = original_dim  # Same 4D as main
            self.build_encoding_diffuse(cfg_encoding_diffuse)
            
            self.level_dim = original_dim  # 4 per level
            self.feat_dim = cfg_model.encoding.levels * self.level_dim  # 24D for 6 levels × 4
            
        elif method == 'baseline_blend_double':
            # Two separate 4D hashgrids: spatial at blended position, per-Gaussian at pk
            print(f"[INGP] baseline_blend_double mode: Creating two 4D hashgrids")
            print(f"  - Hashgrid 1: {original_dim} features per level (queried at BLENDED 3D position, AFTER alpha blending)")
            print(f"  - Hashgrid 2: {original_dim} features per level (queried at pk Gaussian center, alpha blended)")
            print(f"  - Combined: blended feat_pk + feat_spatial, then MLP")
            print(f"  - Advantage: Only 1 spatial query per pixel (not N queries)")
            
            # Build spatial hashgrid (queried at blended position)
            cfg_model.encoding.hashgrid.dim = original_dim
            self.build_encoding(cfg_model.encoding)
            
            # Build per-Gaussian hashgrid (queried at pk, alpha blended)
            cfg_encoding_diffuse = cfg_model.encoding
            cfg_encoding_diffuse.hashgrid.dim = original_dim  # Same 4D as main
            self.build_encoding_diffuse(cfg_encoding_diffuse)
            
            self.level_dim = original_dim  # 4 per level
            self.feat_dim = cfg_model.encoding.levels * self.level_dim  # 24D for 6 levels × 4

        elif method == 'hybrid_features':
            # Hybrid mode: Split between per-Gaussian and hashgrid features
            # Total dimension is always (total_levels × D)
            # Per-Gaussian: hybrid_levels × D
            # Hashgrid: (total_levels - hybrid_levels) × D (from finest levels)
            
            cfg_encoding_hybrid = cfg_model.encoding
            original_levels = cfg_encoding_hybrid.levels  # e.g., 6 from config
            
            per_gaussian_dim = hybrid_levels * original_dim
            hashgrid_levels = original_levels - hybrid_levels  # Remaining levels for hashgrid
            hashgrid_dim = hashgrid_levels * original_dim
            total_dim = original_levels * original_dim  # Always constant (e.g., 24D)
            
            print(f"[INGP] Hybrid features mode: Splitting {original_levels} levels between per-Gaussian and hashgrid")
            print(f"  - Total levels: {original_levels} (from config)")
            print(f"  - Per-Gaussian features: {hybrid_levels} levels × {original_dim} = {per_gaussian_dim}D")
            print(f"  - Hashgrid features: {hashgrid_levels} finest levels × {original_dim} = {hashgrid_dim}D")
            print(f"  - Combined: {per_gaussian_dim}D + {hashgrid_dim}D = {total_dim}D total")
            
            # Store total levels for dimension calculations
            self.total_levels = original_levels
            self.hashgrid_levels = hashgrid_levels
            
            # Build hashgrid with (total_levels - hybrid_levels) levels
            # Special case: hybrid_levels=total_levels means pure 2DGS, no hashgrid needed
            # These are the finest (total_levels - hybrid_levels) levels of original grid
            if hashgrid_levels > 0:
                cfg_encoding_hybrid.levels = hashgrid_levels
                cfg_encoding_hybrid.hashgrid.dim = original_dim

                # Calculate exact resolutions to match the original grid's finest levels
                # Original grid: base_res * scale^level for level in [0, original_levels-1]
                # Hashgrid: should use levels [hybrid_levels, original_levels-1]
                original_base = 2 ** cfg_encoding_hybrid.hashgrid.min_logres  # 128
                original_finest = 2 ** cfg_encoding_hybrid.hashgrid.max_logres  # 512
                
                # Calculate the exact starting resolution for level=hybrid_levels
                start_level = hybrid_levels
                original_scale = (original_finest / original_base) ** (1 / (original_levels - 1))
                start_res = original_base * (original_scale ** start_level)
                
                # Instead of using log2 (which rounds), directly set base_resolution
                # and calculate the per_level_scale for the hashgrid
                # Hashgrid has hashgrid_levels, from start_res to original_finest
                hashgrid_scale = (original_finest / start_res) ** (1 / (hashgrid_levels - 1)) if hashgrid_levels > 1 else 1.0
                
                # Store these for hashgrid init
                cfg_encoding_hybrid.hashgrid.base_resolution = int(np.round(start_res))
                cfg_encoding_hybrid.hashgrid.per_level_scale = float(hashgrid_scale)
                cfg_encoding_hybrid.hashgrid.finest_resolution = original_finest
                
                # Don't use min/max_logres anymore for hybrid mode
                # These will be overridden by base_resolution and per_level_scale in build_encoding
                
                print(f"  - Hashgrid config: {hashgrid_levels} levels, base={int(start_res)}, scale={hashgrid_scale:.4f}, finest={original_finest}")

                self.build_encoding(cfg_encoding_hybrid)
                # Don't override active_levels here - let set_active_levels() handle coarse-to-fine
                
                # Store hashgrid_levels for dynamic step calculation in hybrid mode
                self.hashgrid_levels_for_annealing = hashgrid_levels
            else:
                # Pure 2DGS mode: no hashgrid
                print(f"  - No hashgrid (hybrid_levels={hybrid_levels} equals total_levels={original_levels})")
                self.build_encoding(cfg_encoding_hybrid)  # Build dummy encoding
                self.active_levels = 0
            
            self.level_dim = original_dim  # 4 per level
            self.feat_dim = total_dim  # Always total_levels × D (e.g., 24D)

            # Coarse-to-fine is now ENABLED for hybrid_features (uses config value from build_encoding)
            # self.level_mask is set by build_encoding() from cfg_encoding.coarse2fine.enabled
            if hasattr(self, 'level_mask') and self.level_mask:
                print(f"  - Coarse-to-fine ENABLED: will anneal from {self.init_active_level} to {self.levels} hashgrid levels")
            else:
                print(f"  - Coarse-to-fine disabled: using all {self.levels} hashgrid levels")

        else:
            # Baseline: use original dimension directly
            print(f"[INGP] Baseline mode: {original_dim} features per level")
            cfg_model.encoding.hashgrid.dim = original_dim
            self.build_encoding(cfg_model.encoding)
            self.level_dim = original_dim  # 4 per level (direct from hashgrid)
            self.feat_dim = cfg_model.encoding.levels * self.level_dim
        
        # MLP processes view-dependent features only (diffuse RGB is added after)
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

        l = [
            {'params': self.mlp_rgb.parameters(), 'lr': lr_mlp_rgb, "name": "rgb_mlp"}, # Separate LR for MLP
        ]
        
        # Add hash encoding parameters (different for surface_rgb, baseline_double, baseline_blend_double with two hashgrids)
        if self.method == 'surface_rgb':
            l.append({'params': self.hash_encoding_diffuse.parameters(), 'lr': lr_encoding, "name": "hash_encoding_diffuse"})
            l.append({'params': self.hash_encoding_view_features.parameters(), 'lr': lr_encoding, "name": "hash_encoding_view_features"})
        elif self.method == 'baseline_double':
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})  # xyz hashgrid
            l.append({'params': self.hash_encoding_diffuse.parameters(), 'lr': lr_encoding, "name": "hash_encoding_diffuse"})  # pk hashgrid
        elif self.method == 'baseline_blend_double':
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})  # spatial hashgrid (blended position)
            l.append({'params': self.hash_encoding_diffuse.parameters(), 'lr': lr_encoding, "name": "hash_encoding_diffuse"})  # per-Gaussian hashgrid (pk)
        else:
            l.append({'params': self.hash_encoding.parameters(), 'lr': lr_encoding, "name": "hash_encoding"})

        self.optimizer = torch.optim.Adam(l, betas=(0.9, 0.99), eps=1e-15)

    def build_encoding(self, cfg_encoding, name_suffix=''):
        assert(cfg_encoding.type == "hashgrid")
        self.voxel_range = cfg_encoding.hashgrid.range
        self.gridrange = torch.tensor(self.voxel_range).cuda().float()

        # Check if explicit base_resolution and per_level_scale are provided (hybrid mode)
        if hasattr(cfg_encoding.hashgrid, 'base_resolution') and hasattr(cfg_encoding.hashgrid, 'per_level_scale'):
            # Use explicit values (for hybrid_features mode)
            r_min = cfg_encoding.hashgrid.base_resolution
            r_max = cfg_encoding.hashgrid.finest_resolution
            self.growth_rate = cfg_encoding.hashgrid.per_level_scale
        else:
            # Calculate from log2 resolutions (default behavior)
            l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
            r_min, r_max = 2 ** l_min, 2 ** l_max
            num_levels = cfg_encoding.levels
            self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
        
        num_levels = cfg_encoding.levels
        features_per_level = cfg_encoding.hashgrid.dim
        
        tcnn_config = dict(
            otype="HashGrid",
            n_levels=cfg_encoding.levels,
            n_features_per_level=features_per_level,
            log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
            base_resolution=int(r_min),
            per_level_scale=float(self.growth_rate),
        )

        config = SimpleNamespace(
            device="cuda",
            otype="HashGrid",
            n_levels=cfg_encoding.levels,
            n_features_per_level=features_per_level,
            log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
            base_resolution=int(r_min),
            finest_resolution=int(r_max),
            init_mode='uniform',
            per_level_scale=float(self.growth_rate),
            range=self.voxel_range,
        )
        
        if not name_suffix:  # Only print once
            print('hash config:', config)
            print(f'init activate level {cfg_encoding.coarse2fine.init_active_level}')

        # Don't overwrite level_dim here - it was already set correctly in __init__ for surface modes
        # self.level_dim is the OUTPUT dimension per level (after dot product in surface mode)
        # features_per_level is the HASHGRID dimension per level (before dot product)
        self.hashgrid_level_dim = features_per_level  # Store the hashgrid features per level
        self.base_level_dim = cfg_encoding.hashgrid.dim  # Store the base dimension
        self.levels = cfg_encoding.levels

        self.hash_encoding = register_GridEncoder(config)

        self.resolutions = []
        for lv in range(0, num_levels):
            size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
            self.resolutions.append(size)
        
        print(f'hash resolution : {self.resolutions}')
        
        encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels

        self.level_mask = cfg_encoding.coarse2fine.enabled
        print(f'If coarse2fine : {self.level_mask}')
        if self.level_mask:
            self.init_active_level = cfg_encoding.coarse2fine.init_active_level
            self.step = cfg_encoding.coarse2fine.step
            
        return encoding_dim

    def build_encoding_diffuse(self, cfg_encoding):
        """Build baseline feature hashgrid (4 features per level, queried at Gaussian center, CONCATENATED)"""
        print('[INGP] Building baseline feature hashgrid...')
        print('  Note: 4 features per level, queried at Gaussian center, CONCATENATED across levels')
        
        assert(cfg_encoding.type == "hashgrid")
        self.voxel_range_diffuse = cfg_encoding.hashgrid.range
        self.gridrange_diffuse = torch.tensor(self.voxel_range_diffuse).cuda().float()

        l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels = cfg_encoding.levels
        growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
        
        config = SimpleNamespace(
            device="cuda",
            otype="HashGrid",
            n_levels=cfg_encoding.levels,
            n_features_per_level=4,  # 4 baseline features per level
            log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
            base_resolution=2**cfg_encoding.hashgrid.min_logres,
            finest_resolution=2**cfg_encoding.hashgrid.max_logres,
            init_mode='uniform',
            per_level_scale=growth_rate,
            range=self.voxel_range_diffuse,
        )
        
        print('  Baseline feature hash config:', config)
        self.hash_encoding_diffuse = register_GridEncoder(config)
        self.levels_diffuse = num_levels
        self.baseline_feat_dim = 4 * num_levels  # 4 features × num_levels (concatenated)

    def build_encoding_view_features(self, cfg_encoding):
        """Build view-dependent feature hashgrid (12 features per level, queried at intersection, CONCATENATED after dot product)"""
        print('[INGP] Building view-dependent feature hashgrid...')
        print('  Note: Features from all levels will be CONCATENATED after dot product → 24D features (6 levels × 4)')
        
        assert(cfg_encoding.type == "hashgrid")
        self.voxel_range_view = cfg_encoding.hashgrid.range
        self.gridrange_view = torch.tensor(self.voxel_range_view).cuda().float()

        l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels = cfg_encoding.levels
        growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
        
        config = SimpleNamespace(
            device="cuda",
            otype="HashGrid",
            n_levels=cfg_encoding.levels,
            n_features_per_level=cfg_encoding.hashgrid.dim,  # 12 features per level
            log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
            base_resolution=2**cfg_encoding.hashgrid.min_logres,
            finest_resolution=2**cfg_encoding.hashgrid.max_logres,
            init_mode='uniform',
            per_level_scale=growth_rate,
            range=self.voxel_range_view,
        )
        
        print('  View-dependent hash config:', config)
        self.hash_encoding_view_features = register_GridEncoder(config)
        self.levels_view = num_levels
        self.levels = num_levels  # Also set self.levels for set_active_levels compatibility
        self.hashgrid_level_dim = cfg_encoding.hashgrid.dim  # Store the hashgrid features per level (12)
        self.view_feat_dim = num_levels * (cfg_encoding.hashgrid.dim // 3)  # 6 levels × 4 = 24D after dot product
        
        # Set resolutions (needed by set_epsilon)
        self.resolutions = []
        for lv in range(0, num_levels):
            size = np.floor(r_min * growth_rate ** lv).astype(int) + 1
            self.resolutions.append(size)
        print(f'  Hash resolution : {self.resolutions}')
        
        # Set coarse-to-fine attributes (needed by set_active_levels)
        self.level_mask = cfg_encoding.coarse2fine.enabled
        print(f'  If coarse2fine : {self.level_mask}')
        if self.level_mask:
            self.init_active_level = cfg_encoding.coarse2fine.init_active_level
            self.step = cfg_encoding.coarse2fine.step
            print(f'  Init active level: {self.init_active_level}, step: {self.step}')

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
        rgb = self.get_color(features, ray_unit)
        return rgb  
        
    def set_active_levels(self, current_iter=None):
        self.current_optimizer = self.optimizer

        # Only use coarse-to-fine logic if it's enabled
        if self.level_mask:
            # For hybrid_features mode, calculate dynamic step size
            if self.method == 'hybrid_features' and hasattr(self, 'hashgrid_levels_for_annealing'):
                # Target: all hashgrid levels active by iteration 16k
                target_iter = 16000
                hashgrid_levels = self.hashgrid_levels_for_annealing
                
                # Start annealing after warmup: initialize (10k) + warm_up (1k) = 11k
                start_iter = self.initialize + self.warm_up
                
                # If hybrid_levels >= 2, start with 0 hashgrid levels (per-Gaussian features act as coarse levels)
                # Otherwise, start with 1 hashgrid level
                if self.hybrid_levels >= 2:
                    hybrid_init_level = 0  # Per-Gaussian features are enough for initial coarse representation
                else:
                    hybrid_init_level = 1  # Need at least 1 hashgrid level
                
                # Calculate dynamic step size to reach all levels by target_iter
                # Number of level increases needed: hashgrid_levels - hybrid_init_level
                # Available iterations: target_iter - start_iter (5000 iterations)
                if hashgrid_levels > hybrid_init_level:
                    dynamic_step = (target_iter - start_iter) / (hashgrid_levels - hybrid_init_level)
                else:
                    dynamic_step = self.step  # Fallback if already at max
                
                anneal_levels = max((current_iter - start_iter) // dynamic_step, 0)
                self.active_levels = min(hashgrid_levels, anneal_levels + hybrid_init_level)
            else:
                # Standard coarse-to-fine for baseline and other modes
                anneal_levels = max((current_iter - self.initialize - self.warm_up) // self.step, 0)
                self.active_levels = min(self.levels, anneal_levels + self.init_active_level)
        else:
            # If coarse-to-fine is disabled, use all levels
            self.active_levels = self.levels

        # if self.active_levels != self.pre_level:
        #     print(f"Now ingp model level : {self.active_levels}")

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

        # Query hash grid (outputs 3x features in surface mode)
        feat_output = self.hash_encoding(xyz_input)  # (N, feat_dim) or (N, feat_dim*3) for surface

        points_enc = feat_output.view(*points_3D_normalized.shape[:-1], -1)
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
        
