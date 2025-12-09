#!/usr/bin/env python3
"""
Fast metrics computation - processes images one at a time.
"""

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def evaluate(model_path, skip_lpips=False):
    print(f"\nEvaluating: {model_path}")
    
    test_dir = Path(model_path) / "test"
    
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return
    
    methods = os.listdir(test_dir)
    print(f"Found methods: {methods}")
    
    full_dict = {}
    per_view_dict = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        
        if not renders_dir.exists() or not gt_dir.exists():
            print(f"Skipping {method} - missing renders or gt directory")
            continue
        
        file_list = sorted(os.listdir(renders_dir))
        print(f"Found {len(file_list)} images")
        
        ssims = []
        psnrs = []
        lpipss = []
        image_names = []
        
        # Process one image at a time
        desc = "Computing PSNR/SSIM" + ("" if skip_lpips else "/LPIPS")
        for fname in tqdm(file_list, desc=desc):
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            
            render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
            gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
            
            ssims.append(ssim(render_tensor, gt_tensor))
            psnrs.append(psnr(render_tensor, gt_tensor))
            if not skip_lpips:
                lpipss.append(lpips(render_tensor, gt_tensor, net_type='vgg'))
            image_names.append(fname)
            
            # Free memory
            del render_tensor, gt_tensor
            torch.cuda.empty_cache()
        
        mean_ssim = torch.tensor(ssims).mean().item()
        mean_psnr = torch.tensor(psnrs).mean().item()
        
        print(f"\n{'='*60}")
        print(f"Results for {method} ({len(file_list)} images):")
        print(f"{'='*60}")
        print(f"  SSIM : {mean_ssim:.7f}")
        print(f"  PSNR : {mean_psnr:.7f} dB")
        if not skip_lpips:
            mean_lpips = torch.tensor(lpipss).mean().item()
            print(f"  LPIPS: {mean_lpips:.7f}")
        print(f"{'='*60}\n")
        
        result_dict = {
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
        }
        per_view_result = {
            "SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        }
        
        if not skip_lpips:
            mean_lpips = torch.tensor(lpipss).mean().item()
            result_dict["LPIPS"] = mean_lpips
            per_view_result["LPIPS"] = {name: l for l, name in zip(torch.tensor(lpipss).tolist(), image_names)}
        
        full_dict[method] = result_dict
        per_view_dict[method] = per_view_result
    
    # Save results
    results_file = Path(model_path) / "results.json"
    per_view_file = Path(model_path) / "per_view.json"
    
    with open(results_file, 'w') as fp:
        json.dump(full_dict, fp, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    with open(per_view_file, 'w') as fp:
        json.dump(per_view_dict, fp, indent=2)
    print(f"✓ Saved per-view results to: {per_view_file}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    parser = ArgumentParser(description="Compute metrics for rendered images")
    parser.add_argument('--model_path', '-m', required=True, type=str)
    parser.add_argument('--skip_lpips', action='store_true', help='Skip LPIPS computation (faster)')
    args = parser.parse_args()
    
    evaluate(args.model_path, skip_lpips=args.skip_lpips)

