#!/usr/bin/env python
"""Score Vulkan HW-raster renders (vk_raster --dumpall) with EXACTLY the metric code
benchmark_baked.py uses: utils.image_utils.psnr (per-channel mean, no clamp),
utils.loss_utils.ssim, lpipsPyTorch VGG on clamped images.

  vk_raster/vk_raster <bundle> --lp 1 --pad 0.1 --byid --fp16 --bench 0 --warmup 0 --dumpall /tmp/vkr
  conda run -n nest_splatting python scripts/eval_vk_renders.py --bundle <bundle> --renders /tmp/vkr
"""
import os, sys, struct, argparse, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss
from lpipsPyTorch import lpips

ap = argparse.ArgumentParser(); ap.add_argument('--bundle', required=True); ap.add_argument('--renders', required=True)
a = ap.parse_args()
buf = open(os.path.join(a.bundle, 'cams.bin'), 'rb').read(); off = 0
n, = struct.unpack_from('<I', buf, off); off += 4
ps, ss, ls, l1s = [], [], [], []
for i in range(n):
    W, H = struct.unpack_from('<II', buf, off); off += 8 + 64 + 64 + 12 + 8
    gt = torch.from_numpy(np.frombuffer(buf, np.uint8, 3 * W * H, off).reshape(3, H, W).copy()).float().cuda() / 255.0
    off += 3 * W * H
    r = np.fromfile(os.path.join(a.renders, f'{i:03d}.f32'), np.float32).reshape(H, W, 3)
    rendered = torch.from_numpy(r).permute(2, 0, 1).contiguous().cuda()
    ps.append(psnr(rendered, gt).mean().item()); l1s.append(l1_loss(rendered, gt).item()); ss.append(ssim(rendered, gt).item())
    ls.append(lpips(rendered.clamp(0, 1).unsqueeze(0), gt.clamp(0, 1).unsqueeze(0), net_type='vgg').item())
print(f"[VK-EVAL] {n} views  PSNR {np.mean(ps):.2f}  SSIM {np.mean(ss):.4f}  LPIPS {np.mean(ls):.4f}  L1 {np.mean(l1s):.4f}")
