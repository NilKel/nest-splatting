#!/usr/bin/env python3
"""Render one frame with LEAN CONIC and PROD, save both + a diff image."""
import sys, os, glob, math, pickle, json
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting/scripts')

# Import bake_render modules FIRST — before benchmark_baked which does
# sys.modules aliasing for the res_3d_paired mode.
import diff_surfel_bake_render as prod
import diff_surfel_bake_render_lean as lean
# Snapshot the API funcs we need so any downstream sys.modules alias doesn't
# steal them from us.
_prod_api = {'prep': prod.prepare_gaussian_inputs, 'get_r': prod.get_rasterizer,
             'set_ab': prod.set_activation_bias, 'set_cm': prod.set_compact_mult,
             'set_bm': prod.set_beta_mult, 'set_rm': prod.set_residual_mode,
             'set_bc7': prod.set_atlas_bc7}
_lean_api = {'prep': lean.prepare_gaussian_inputs, 'get_r': lean.get_rasterizer,
             'set_ab': lean.set_activation_bias, 'set_cm': lean.set_compact_mult,
             'set_bm': lean.set_beta_mult, 'set_rm': lean.set_residual_mode,
             'set_bc7': lean.set_atlas_bc7}

from argparse import ArgumentParser
from pathlib import Path
import torch, numpy as np
from PIL import Image

model_path = Path('/home/nilkel/Projects/nest-splatting/outputs/mip_360/garden/3D_SH_res/RD_SV_30thr_005w25gLP_N2f_frz5k10')
bake_dir = model_path / 'baked_atlas'

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from hash_encoder.config import Config
from benchmark_baked import _make_sv_state

train_args = pickle.load(open(model_path/'args.pkl','rb'))
train_args.model_path = str(model_path); train_args.eval = True
cfg_model = Config(str(model_path/'config.yaml'))
gaussians = GaussianModel(train_args.sh_degree)
gaussians.load_ply(str(bake_dir/'baked.ply'), args=train_args)
gaussians.kernel_type = train_args.kernel
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
gaussians.XYZ_TYPE = 'UV'; gaussians._sv_training_flag = False; gaussians.update_sites_mask()

parser = ArgumentParser()
mp = ModelParams(parser, sentinel=True); PipelineParams(parser)
dataset = mp.extract(train_args)
scene = Scene(dataset, GaussianModel(train_args.sh_degree), load_iteration=35000, shuffle=False)
cam = scene.getTestCameras()[0]

atlas_rects = torch.load(bake_dir/'atlas_rects.pt').cuda()
bc7 = torch.from_numpy(np.fromfile(bake_dir/'atlas_texture.bc7', dtype=np.uint8)).cuda()
bake_meta = json.load(open(bake_dir/'bake_meta.json'))
atlas_W = int(bake_meta['atlas_width']); atlas_H = int(bake_meta['atlas_height'])
atlas_offset = float(bake_meta.get('atlas_offset', 0.0)); atlas_scale = float(bake_meta.get('atlas_scale', 1.0))
sv_state = _make_sv_state(gaussians)
bg = torch.zeros(3, dtype=torch.float32, device='cuda')
beta = float(cfg_model.surfel.tg_beta)
kt = {'gaussian': 0, 'beta_scaled': 4}.get(train_args.kernel, 0)

def render(api):
    pkg = api['prep'](gaussians, sh_degree=train_args.sh_degree, kernel_type=kt)
    api['set_ab'](0.5, 0.0)
    api['set_cm'](1.0); api['set_bm'](1.0); api['set_rm'](0)
    api['set_bc7'](bc7, atlas_W, atlas_H, atlas_offset, atlas_scale)
    r = api['get_r'](image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx*0.5), tanfovy=math.tan(cam.FoVy*0.5),
        bg=bg, viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        campos=cam.camera_center, sh_degree=train_args.sh_degree, beta=beta, aabb_mode=5, sort_mode=0)
    color, _ = r(means3D=pkg['means3D'], opacities=pkg['opacities'], shs=pkg['shs'],
        scales=pkg['scales'], rotations=pkg['rotations'], shapes=pkg.get('shapes', None),
        kernel_type=pkg['kernel_type'],
        atlas_texture=torch.zeros(3, dtype=torch.float16, device='cuda'),
        atlas_rects=atlas_rects, atlas_width=atlas_W,
        voronoi_sites=sv_state['sites'], voronoi_tau=sv_state['tau'],
        voronoi_colors=sv_state['colors'], voronoi_K=sv_state['K'])
    return color.clamp(0, 1)

img_prod = render(_prod_api)
img_lean = render(_lean_api)
gt = cam.original_image[:3].cuda().clamp(0, 1)

def to_img(t):
    return (t.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

outdir = Path('/home/nilkel/Projects/nest-splatting/speed_comparison/renders_conic_debug')
outdir.mkdir(parents=True, exist_ok=True)
Image.fromarray(to_img(img_prod)).save(outdir/'render_prod.png')
Image.fromarray(to_img(img_lean)).save(outdir/'render_lean_conic.png')
Image.fromarray(to_img(gt)).save(outdir/'render_gt.png')
diff = (img_lean - img_prod).abs()
Image.fromarray(to_img((diff * 10).clamp(0, 1))).save(outdir/'render_diff10x.png')
Image.fromarray(to_img(diff.clamp(0, 1))).save(outdir/'render_diff1x.png')

print(f"Prod   mean={img_prod.mean():.4f} std={img_prod.std():.4f} min={img_prod.min():.4f} max={img_prod.max():.4f}")
print(f"Lean   mean={img_lean.mean():.4f} std={img_lean.std():.4f} min={img_lean.min():.4f} max={img_lean.max():.4f}")
print(f"|Diff| mean={diff.mean():.4f} std={diff.std():.4f} min={diff.min():.4f} max={diff.max():.4f}")
print(f"images saved to {outdir}/render_*.png")
