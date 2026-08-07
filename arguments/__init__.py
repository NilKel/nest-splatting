#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        self.load_allres = False

        # Progressive resolution curriculum (`--start_resolution N`): train at
        # this lower-res `-r` value initially, then mid-train reload at the
        # full `--resolution` value at `--freeze_hash_iter`. Pre-switch the
        # hash+MLP train every iter and need cheap iters (low-res + GPU GT).
        # Post-switch the hash trains 1-in-`--freeze_hash_period` iters → can
        # afford slow 4K iters with CPU-staged GT. Default 0 = curriculum off.
        self.start_resolution = 0
        # Optional separate `--data_device` during the start-resolution phase.
        # Default "cuda" — the low-res GT fits comfortably on-GPU even with
        # 200+ images. After the mid-train reload, `--data_device` (the host
        # field above) takes effect.
        self.start_data_device = "cuda"
        
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        # Skip the per-call Python-side `depth_to_normal` + render-normal
        # rotate + normal_error computation when no normal-consistency or
        # depth-distortion regularizer is active. At 4K image resolution
        # this avoids ~600 MB of intermediate tensors (dx, dy, cross product,
        # normal error map) per render call. Auto-enabled by train.py when
        # lambda_normal == w_normal == lambda_dist == 0; user can also pass
        # --skip_aux_normal_dist to force-enable.
        self.skip_aux_normal_dist = False
        # Opt-out of the above auto-enable: pass --keep_aux_normal_dist to force
        # normals/depth-distortion aux rendering ON even when no reg consumes them
        # (e.g. to populate the training_output normal/decomposition views).
        self.keep_aux_normal_dist = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.lambda_floater = 0.0
        self.lambda_mask = 0.0
        # SSIM mix for the alpha/mask loss (Niedermayr et al. 2024 use L1+SSIM
        # on the alpha channel): mask_error = (1-mask_dssim)*L1 + mask_dssim*(1-SSIM).
        # 0.0 keeps the historical pure-L1 behavior.
        self.mask_dssim = 0.0
        
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.pixel_densify_from_iter = 3000
        self.freeze_sh = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
