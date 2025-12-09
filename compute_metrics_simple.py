#!/usr/bin/env python3
"""
Simple metrics computation script for rendered images.
Computes PSNR, SSIM, and LPIPS for test set renders.
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

def readImages(renders_dir, gt_dir):
    print(f"Reading images from {renders_dir} and {gt_dir}...")
    renders = []
    gts = []
    image_names = []
    
    file_list = sorted(os.listdir(renders_dir))
    print(f"Found {len(file_list)} images")
    
    for i, fname in enumerate(tqdm(file_list, desc="Loading images")):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    
    print(f"Loaded {len(renders)} image pairs")
    return renders, gts, image_names

def evaluate(model_path):
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
        
        renders, gts, image_names = readImages(renders_dir, gt_dir)
        
        ssims = []
        psnrs = []
        lpipss = []
        
        print("Computing metrics...")
        for idx in tqdm(range(len(renders)), desc="Computing PSNR/SSIM/LPIPS"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
        
        mean_ssim = torch.tensor(ssims).mean().item()
        mean_psnr = torch.tensor(psnrs).mean().item()
        mean_lpips = torch.tensor(lpipss).mean().item()
        
        print(f"\n{'='*60}")
        print(f"Results for {method}:")
        print(f"{'='*60}")
        print(f"  SSIM : {mean_ssim:.7f}")
        print(f"  PSNR : {mean_psnr:.7f} dB")
        print(f"  LPIPS: {mean_lpips:.7f}")
        print(f"{'='*60}\n")
        
        full_dict[method] = {
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
            "LPIPS": mean_lpips
        }
        per_view_dict[method] = {
            "SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS": {name: l for l, name in zip(torch.tensor(lpipss).tolist(), image_names)}
        }
    
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
    args = parser.parse_args()
    
    evaluate(args.model_path)

