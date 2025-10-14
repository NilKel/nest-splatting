import sys,os,cv2 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as tf
sys.path += ["./", "../"]
from glob import glob 
import argparse
import torch 
import numpy as np 
from plyfile import PlyData, PlyElement

from utils.loss_utils import ssim
from utils.image_utils import psnr
import matplotlib.pyplot as plt
from lpipsPyTorch import lpips
# from metrics import readImages

def cal_psnr(pth, with_lpips, quiet = False):

    if quiet == False:
        print(f'Eval at path {pth} ')
    
    gt_path = os.path.join(pth , 'gt')
    pred_path = os.path.join(pth , 'renders')
    
    gt_list = glob(gt_path + '/*.png')
    # gt_list.sort(key = lambda x: int(os.path.basename(x).split('.')[0][2:]))
    gt_list.sort()
    
    psnrs = []
    ssims = []
    lpips_list = []
    for gt_file in gt_list:
        img_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_path, img_name)
        gt = Image.open(gt_file)
        pred = Image.open(pred_file)
        
        pred = tf.to_tensor(pred)[:3, :, :].cuda()
        gt = tf.to_tensor(gt)[:3, :, :].cuda()

        psnr_val = psnr(gt,pred).mean()
        psnrs.append(psnr_val)
        ssim_val = ssim(gt, pred)
        ssims.append(ssim_val)
        if with_lpips:
            lpips_eval = lpips(gt, pred, net_type='vgg')
            lpips_list.append(lpips_eval)
            
    if quiet == False:
        print("SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))

    return torch.tensor(psnrs).mean(), torch.tensor(ssims).mean(), torch.tensor(lpips_list).mean()
    
exp_folder = sys.argv[1]
iteration = sys.argv[2]

exp_path = glob(exp_folder + '/*')

print(f'eval {exp_path} at {iteration} iter')

with_lpips = False

AVE_PSNR = 0.0
AVE_SSIM = 0.0
AVE_LPIPS = 0.0
AVE_PTS = 0

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return positions

for exp in exp_path:
    exp_name = os.path.basename(exp)
    load_path = os.path.join(exp, 'test', f'ours_{iteration}')
    # load_path = os.path.join(exp, f'{iteration}_iter', 'test')
    psnr_eval, ssim_eval, lpips_eval = cal_psnr(load_path, with_lpips, quiet = True)
    AVE_PSNR += psnr_eval
    AVE_SSIM += ssim_eval
    AVE_LPIPS += lpips_eval

    load_path = os.path.join(exp, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
    pts = fetchPly(load_path)
    pts_num = pts.shape[0]
    AVE_PTS += pts_num

    print(f"{exp_name} PSNR {psnr_eval:.4f} SSIM {ssim_eval:.4f} LPIPS {lpips_eval:.4f} {(pts_num // 1000)}k gaussians")

print(f'AVE PSNR {(AVE_PSNR / len(exp_path)):.4f}')
print(f'AVE SSIM {(AVE_SSIM / len(exp_path)):.4f}')
print(f'AVE LPIPS {(AVE_LPIPS / len(exp_path)):.4f}')
print(f'AVE NUM {(AVE_PTS / len(exp_path))//1000}k gaussians')
