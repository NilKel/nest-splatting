#!/usr/bin/env python3
"""
Train baseline and surface potential methods on all NeRF synthetic scenes.
Uses 2DGS checkpoint system - first method trains 2DGS, second reuses it.
"""
import os
from argparse import ArgumentParser

scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
dataset_dir = "/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic"

parser = ArgumentParser(description="Full comparison script for baseline vs surface")
parser.add_argument("--yaml", default="./configs/nerfsyn.yaml", help="Config file")
parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
parser.add_argument("--methods", nargs='+', default=["baseline", "surface"], 
                    help="Methods to compare (default: baseline surface)")
parser.add_argument("--scenes", nargs='+', default=scenes, 
                    help="Scenes to train (default: all)")
parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")
args = parser.parse_args()

yaml_file = args.yaml
iterations = args.iterations
methods_to_train = args.methods
scenes_to_train = args.scenes

print("╔══════════════════════════════════════════════════════════════════╗")
print("║           NERF SYNTHETIC MULTI-SCENE COMPARISON                  ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"Config: {yaml_file}")
print(f"Iterations: {iterations}")
print(f"Methods: {methods_to_train}")
print(f"Scenes: {scenes_to_train}")
print("")

# Train all methods for all scenes
for method in methods_to_train:
    print("\n" + "="*70)
    print(f"TRAINING METHOD: {method.upper()}")
    print("="*70 + "\n")
    
    for scene in scenes_to_train:
        source = f'{dataset_dir}/{scene}'
        save_dir = f'./output/{scene}_{method}_{iterations}'
        
        cmd = (f'python train.py '
               f'-s {source} '
               f'-m {save_dir} '
               f'--yaml {yaml_file} '
               f'--method {method} '
               f'--eval '
               f'--iterations {iterations} '
               f'--scene_name {scene}')
        
        print(f"\n[{scene.upper()}] Training {method}...")
        print(f"Command: {cmd}\n")
        
        ret = os.system(cmd)
        if ret != 0:
            print(f"✗ Failed to train {scene} with {method}")
        else:
            print(f"✓ Completed {scene} with {method}")

# Collect and display results
if not args.skip_metrics:
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")
    
    for scene in scenes_to_train:
        print(f"\n{scene.upper()}:")
        print("-" * 40)
        
        for method in methods_to_train:
            save_dir = f'./output/{scene}_{method}_{iterations}'
            metrics_file = f'{save_dir}/final_test_renders/metrics.txt'
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'PSNR:' in line or 'SSIM:' in line:
                            print(f"  {method:10s}: {line.strip()}")
            else:
                print(f"  {method:10s}: No metrics found")

print("\n" + "="*70)
print("ALL SCENES COMPLETE")
print("="*70)
print(f"\n💾 2DGS checkpoints saved in:")
for scene in scenes_to_train:
    ckpt = f"{dataset_dir}/{scene}/gaussian_init.pth"
    if os.path.exists(ckpt):
        print(f"  ✓ {scene}: {ckpt}")
    else:
        print(f"  ✗ {scene}: Not found")
print("")

