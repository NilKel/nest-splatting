#!/usr/bin/env python3
"""Compare 3D_direct vs 3D_direct_lean with various masks."""
import os
import sys
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8


def main():
    model_path = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair/3D_direct/newmod1sccenin_FIXED8"
    iteration = 30000
    
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    
    cfg_model = Config(os.path.join(model_path, "config.yaml"))
    
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    
    args_direct = Namespace(**vars(args))
    args_direct.method = "3D_direct"
    ingp_direct = INGP(cfg_model, args=args_direct).to('cuda')
    ingp_direct.load_model(model_path, iteration)
    
    args_lean = Namespace(**vars(args))
    args_lean.method = "3D_direct_lean"
    ingp_lean = INGP(cfg_model, args=args_lean).to('cuda')
    ingp_lean.load_model(model_path, iteration)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_direct.set_active_levels(iteration)
    ingp_lean.set_active_levels(iteration)
    
    cam = scene.getTestCameras()[0]
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta
    
    output_dir = os.path.join(model_path, "mode_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save originals
    orig_gaussian_features = gaussians._gaussian_features.data.clone()
    orig_hash_embeddings = ingp_direct.hash_encoding.embeddings.data.clone()
    
    def compute_psnr(img1, img2):
        mse = ((img1 - img2) ** 2).mean()
        return 10 * torch.log10(1.0 / mse).item() if mse > 0 else float('inf')
    
    def save_and_compare(img_direct, img_lean, name):
        diff = (img_direct - img_lean).abs()
        psnr = compute_psnr(img_direct, img_lean)
        save_img_u8(img_direct.permute(1,2,0).cpu().numpy(), os.path.join(output_dir, f"{name}_direct.png"))
        save_img_u8(img_lean.permute(1,2,0).cpu().numpy(), os.path.join(output_dir, f"{name}_lean.png"))
        save_img_u8(np.clip(diff.permute(1,2,0).cpu().numpy() * 10, 0, 1), os.path.join(output_dir, f"{name}_diff10x.png"))
        print(f"  {name}: PSNR={psnr:.2f} dB, mean_diff={diff.mean():.6f}")
        return psnr
    
    print("="*60)
    print("MASKED COMPARISON (direct manipulation)")
    print("="*60)
    
    with torch.no_grad():
        # 1. Normal render
        img_d = render(cam, gaussians, pipe, background, ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        img_l = render(cam, gaussians, pipe, background, ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        save_and_compare(img_d, img_l, "normal")
        
        # 2. Zero Gaussian features
        gaussians._gaussian_features.data.zero_()
        img_d = render(cam, gaussians, pipe, background, ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        img_l = render(cam, gaussians, pipe, background, ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        save_and_compare(img_d, img_l, "no_gaussian_feat")
        gaussians._gaussian_features.data.copy_(orig_gaussian_features)  # Restore
        
        # 3. Zero hash embeddings
        ingp_direct.hash_encoding.embeddings.data.zero_()
        ingp_lean.hash_encoding.embeddings.data.zero_()
        img_d = render(cam, gaussians, pipe, background, ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        img_l = render(cam, gaussians, pipe, background, ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        save_and_compare(img_d, img_l, "no_hash_feat")
        ingp_direct.hash_encoding.embeddings.data.copy_(orig_hash_embeddings)  # Restore
        ingp_lean.hash_encoding.embeddings.data.copy_(orig_hash_embeddings)
        
        # 4. Zero view encoding
        orig_encode_view_d = ingp_direct._encode_view
        orig_encode_view_l = ingp_lean._encode_view
        ingp_direct._encode_view = lambda d: torch.zeros(d.shape[0], 16, device=d.device, dtype=torch.float32)
        ingp_lean._encode_view = lambda d: torch.zeros(d.shape[0], 16, device=d.device, dtype=torch.float32)
        img_d = render(cam, gaussians, pipe, background, ingp=ingp_direct, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        img_l = render(cam, gaussians, pipe, background, ingp=ingp_lean, beta=beta, iteration=iteration, cfg=cfg_model)["render"]
        save_and_compare(img_d, img_l, "no_viewdir")
        ingp_direct._encode_view = orig_encode_view_d
        ingp_lean._encode_view = orig_encode_view_l
    
    print(f"\nImages saved to: {output_dir}")

if __name__ == "__main__":
    main()
