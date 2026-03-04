"""Collect per-scene results.json files and print a combined DTU metrics table."""

import json
import os
from argparse import ArgumentParser

def collect(output_dir):
    scenes = sorted([d for d in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("scan")])

    results = {}
    for scene in scenes:
        rpath = os.path.join(output_dir, scene, "results.json")
        if not os.path.exists(rpath):
            print(f"WARNING: {rpath} not found, skipping")
            continue
        with open(rpath) as f:
            data = json.load(f)
        # results.json has one key per method (e.g. "ours_30000")
        method = list(data.keys())[0]
        results[scene] = data[method]

    if not results:
        print("No results found!")
        return

    # Print table
    header = f"{'Scene':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}"
    print(header)
    print("-" * len(header))

    psnrs, ssims, lpipss = [], [], []
    for scene in sorted(results):
        m = results[scene]
        print(f"{scene:<12} {m['PSNR']:8.2f} {m['SSIM']:8.4f} {m['LPIPS']:8.4f}")
        psnrs.append(m["PSNR"])
        ssims.append(m["SSIM"])
        lpipss.append(m["LPIPS"])

    print("-" * len(header))
    print(f"{'Average':<12} {sum(psnrs)/len(psnrs):8.2f} {sum(ssims)/len(ssims):8.4f} {sum(lpipss)/len(lpipss):8.4f}")
    print(f"\n({len(results)} scenes)")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_dir", help="Path to experiment output dir containing scan* subdirs")
    args = parser.parse_args()
    collect(args.output_dir)
