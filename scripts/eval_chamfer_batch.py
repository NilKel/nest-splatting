#!/usr/bin/env python
"""
Batch Chamfer Distance evaluation for DTU experiments.

Usage:
    python scripts/eval_chamfer_batch.py \
        --exp_dir /path/to/outputs/DTU \
        --exp_name b01o001s0n1e4BS_cat5_5_levels \
        --dtu_source /path/to/dtu/2DGS_data/DTU \
        --dtu_official /path/to/dtu

This will:
1. Find all experiments matching: {exp_dir}/scan*/cat/{exp_name}
2. Extract mesh from each (if not already done)
3. Run chamfer distance evaluation
4. Print results table and mean
"""

import os
import sys
import argparse
import re
import json
import subprocess
from pathlib import Path
from glob import glob


DTU_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def parse_training_log(exp_path):
    """Parse training_log.txt to extract method and hybrid_levels."""
    log_path = os.path.join(exp_path, "training_log.txt")
    method = None
    hybrid_levels = None

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()

        method_match = re.search(r'Method:\s*(\w+)', content)
        if method_match:
            method = method_match.group(1)

        hybrid_match = re.search(r'Hybrid Levels:\s*(\d+)', content)
        if hybrid_match:
            hybrid_levels = int(hybrid_match.group(1))

    return method, hybrid_levels


def find_latest_iteration(exp_path):
    """Find the latest checkpoint iteration."""
    pc_dir = os.path.join(exp_path, 'point_cloud')
    if not os.path.exists(pc_dir):
        return 30000  # default

    iterations = []
    for name in os.listdir(pc_dir):
        if name.startswith('iteration_'):
            try:
                iterations.append(int(name.split('_')[1]))
            except ValueError:
                pass

    return max(iterations) if iterations else 30000


def find_existing_mesh(exp_path, iteration):
    """Look for existing mesh."""
    possible_paths = [
        os.path.join(exp_path, 'train', f'ours_{iteration}', 'fuse_post.ply'),
        os.path.join(exp_path, 'train', f'ours_{iteration}', 'fuse.ply'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def extract_mesh(exp_path, source_path, iteration, method, hybrid_levels):
    """Extract mesh using eval_render.py."""
    existing = find_existing_mesh(exp_path, iteration)
    if existing:
        print(f"  Mesh exists: {existing}")
        return existing

    print(f"  Extracting mesh...")
    cmd = [
        sys.executable, 'eval_render.py',
        '--iteration', str(iteration),
        '-s', source_path,
        '-m', exp_path,
        '--yaml', './configs/dtu.yaml',
        '--method', method,
        '--hybrid_levels', str(hybrid_levels),
        '--skip_train',
        '--skip_test',
        '--depth_ratio', '1.0',
        '--num_cluster', '1',
        '--voxel_size', '0.004',
        '--sdf_trunc', '0.016',
        '--depth_trunc', '3.0',
    ]

    print(f"  Running: {' '.join(cmd)}")
    # Pass current environment to subprocess (needed for LD_LIBRARY_PATH, CUDA, etc.)
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        print(f"  ERROR: Mesh extraction failed (exit code {result.returncode})")
        if result.stdout:
            print(f"  STDOUT: {result.stdout[-1000:]}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-1000:]}")
        return None

    mesh = find_existing_mesh(exp_path, iteration)
    if not mesh:
        print(f"  ERROR: Mesh extraction ran but no mesh found")
        if result.stdout:
            print(f"  STDOUT: {result.stdout[-1000:]}")
    return mesh


def run_chamfer_eval(mesh_file, scan_id, dtu_source, dtu_official, output_dir):
    """Run chamfer distance evaluation."""
    script_path = os.path.join(os.path.dirname(__file__), 'eval_dtu', 'evaluate_single_scene.py')

    if not os.path.exists(script_path):
        print(f"  ERROR: Eval script not found: {script_path}")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, script_path,
        '--input_mesh', mesh_file,
        '--scan_id', str(scan_id),
        '--output_dir', output_dir,
        '--mask_dir', dtu_source,
        '--DTU', dtu_official,
    ]

    print(f"  CMD: {' '.join(cmd)}")
    # Pass current environment to subprocess (needed for LD_LIBRARY_PATH, CUDA, etc.)
    env = os.environ.copy()
    # Ensure CUDA is visible
    if 'CUDA_VISIBLE_DEVICES' not in env:
        env['CUDA_VISIBLE_DEVICES'] = '0'
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Always print output for debugging
    if result.stdout:
        print(f"  STDOUT: {result.stdout[-2000:]}")
    if result.stderr:
        print(f"  STDERR: {result.stderr[-2000:]}")

    if result.returncode != 0:
        print(f"  ERROR: Chamfer eval failed (exit code {result.returncode})")
        return None

    results_file = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    else:
        print(f"  ERROR: results.json not created at {results_file}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Batch Chamfer Distance evaluation for DTU')
    parser.add_argument('--exp_dir', required=True,
                        help='Root directory (e.g., /path/to/outputs/DTU)')
    parser.add_argument('--exp_name', required=True,
                        help='Experiment folder name to evaluate (e.g., b01o001s0n1e4BS_cat5_5_levels)')
    parser.add_argument('--method_dir', default='cat',
                        help='Method subdirectory name (default: cat)')
    parser.add_argument('--dtu_source', required=True,
                        help='Path to preprocessed DTU data (with cameras.npz, masks)')
    parser.add_argument('--dtu_official', required=True,
                        help='Path to official DTU data (with Points/ and ObsMask/)')
    parser.add_argument('--scans', nargs='+', type=int, default=None,
                        help=f'Specific scans to evaluate (default: all 15 DTU scans)')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Iteration to evaluate (default: auto-detect latest)')
    parser.add_argument('--skip_mesh', action='store_true',
                        help='Skip mesh extraction, use existing meshes only')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip scans that already have results.json')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just print what would be evaluated')
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.exp_dir):
        print(f"ERROR: exp_dir not found: {args.exp_dir}")
        sys.exit(1)

    obsmask_path = os.path.join(args.dtu_official, 'ObsMask')
    if not os.path.exists(obsmask_path):
        print(f"ERROR: ObsMask not found at {obsmask_path}")
        print("Download SampleSet.zip from DTU and extract ObsMask folder")
        sys.exit(1)

    scans = args.scans if args.scans else DTU_SCANS

    # Find experiments
    print(f"\nSearching for: {args.exp_dir}/scan*/{args.method_dir}/{args.exp_name}")
    print("=" * 70)

    experiments = []
    for scan_id in scans:
        exp_path = os.path.join(args.exp_dir, f'scan{scan_id}', args.method_dir, args.exp_name)
        if os.path.exists(exp_path):
            experiments.append((scan_id, exp_path))
        else:
            print(f"  scan{scan_id}: NOT FOUND")

    if not experiments:
        print("\nNo experiments found!")
        sys.exit(1)

    print(f"\nFound {len(experiments)}/{len(scans)} experiments:")
    for scan_id, exp_path in experiments:
        print(f"  scan{scan_id}: {exp_path}")

    if args.dry_run:
        print("\n[DRY RUN] Would evaluate the above experiments.")
        return

    # Evaluate each
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print("=" * 70)

    results = []
    failed = []

    for i, (scan_id, exp_path) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] scan{scan_id}")
        print("-" * 50)

        # Check for existing results
        chamfer_dir = os.path.join(exp_path, 'chamfer_eval')
        results_file = os.path.join(chamfer_dir, 'results.json')

        if args.skip_existing and os.path.exists(results_file):
            print(f"  Skipping (results exist)")
            with open(results_file) as f:
                r = json.load(f)
            r['scan_id'] = scan_id
            results.append(r)
            continue

        # Get method info
        method, hybrid_levels = parse_training_log(exp_path)
        method = method or 'cat'
        hybrid_levels = hybrid_levels or 5

        # Get iteration
        iteration = args.iteration or find_latest_iteration(exp_path)
        print(f"  Method: {method}, Levels: {hybrid_levels}, Iter: {iteration}")

        # Source path
        source_path = os.path.join(args.dtu_source, f'scan{scan_id}')
        if not os.path.exists(source_path):
            print(f"  ERROR: Source not found: {source_path}")
            failed.append(scan_id)
            continue

        # Extract mesh
        if args.skip_mesh:
            mesh_file = find_existing_mesh(exp_path, iteration)
            if not mesh_file:
                print(f"  ERROR: No mesh found (use without --skip_mesh)")
                failed.append(scan_id)
                continue
        else:
            mesh_file = extract_mesh(exp_path, source_path, iteration, method, hybrid_levels)
            if not mesh_file:
                failed.append(scan_id)
                continue

        # Run chamfer eval
        print(f"  Running chamfer eval on: {mesh_file}")
        r = run_chamfer_eval(mesh_file, scan_id, args.dtu_source, args.dtu_official, chamfer_dir)
        if r:
            r['scan_id'] = scan_id
            results.append(r)
            print(f"  Chamfer: {r['overall']:.4f} (d2s: {r['mean_d2s']:.4f}, s2d: {r['mean_s2d']:.4f})")
        else:
            failed.append(scan_id)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Experiment: {args.exp_name}")
    print(f"Evaluated: {len(results)}/{len(experiments)} scans")
    if failed:
        print(f"Failed: {failed}")
    print()

    if results:
        print(f"{'Scan':<10} {'Accuracy (d2s)':<16} {'Complete (s2d)':<16} {'Overall':<12}")
        print("-" * 54)

        total_d2s = 0
        total_s2d = 0
        total_overall = 0

        for r in sorted(results, key=lambda x: x['scan_id']):
            print(f"scan{r['scan_id']:<6} {r['mean_d2s']:<16.4f} {r['mean_s2d']:<16.4f} {r['overall']:<12.4f}")
            total_d2s += r['mean_d2s']
            total_s2d += r['mean_s2d']
            total_overall += r['overall']

        n = len(results)
        print("-" * 54)
        print(f"{'MEAN':<10} {total_d2s/n:<16.4f} {total_s2d/n:<16.4f} {total_overall/n:<12.4f}")

        # Save aggregate
        aggregate_file = os.path.join(args.exp_dir, f'chamfer_{args.exp_name}.json')
        aggregate = {
            'exp_name': args.exp_name,
            'num_scans': n,
            'mean_d2s': total_d2s / n,
            'mean_s2d': total_s2d / n,
            'mean_overall': total_overall / n,
            'per_scan': results,
        }
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate, f, indent=2)
        print(f"\nResults saved to: {aggregate_file}")


if __name__ == '__main__':
    main()
