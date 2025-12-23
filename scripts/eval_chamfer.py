#!/usr/bin/env python
"""
Evaluate Chamfer Distance for DTU experiments.

Usage:
    # Single experiment (auto-detects method from training_log.txt)
    python scripts/eval_chamfer.py --experiment outputs/DTU/scan24/baseline/vanilla_baseline

    # Single experiment with explicit method
    python scripts/eval_chamfer.py --experiment outputs/DTU/scan24/cat/exp1 --method cat --hybrid_levels 5

    # Multiple scenes for a specific method
    python scripts/eval_chamfer.py --exp_dir outputs/DTU --method baseline --scenes 24 37 40

    # All scenes in a directory (auto-detect method per experiment)
    python scripts/eval_chamfer.py --exp_dir outputs/DTU

    # Skip already evaluated experiments (useful for incremental runs)
    python scripts/eval_chamfer.py --exp_dir outputs/DTU --skip_existing

This script:
1. Extracts mesh from the trained model (if not already done)
2. Runs chamfer distance evaluation against GT point cloud
3. Supports batch evaluation across multiple scenes
"""

import os
import sys
import argparse
import re
from pathlib import Path
import subprocess
import json


def parse_training_log(exp_path):
    """Parse training_log.txt to extract method and hybrid_levels."""
    log_path = os.path.join(exp_path, "training_log.txt")
    method = None
    hybrid_levels = None

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract method
        method_match = re.search(r'Method:\s*(\w+)', content)
        if method_match:
            method = method_match.group(1)

        # Extract hybrid levels
        hybrid_match = re.search(r'Hybrid Levels:\s*(\d+)', content)
        if hybrid_match:
            hybrid_levels = int(hybrid_match.group(1))

    return method, hybrid_levels


def get_scan_id(experiment_path):
    """Extract scan ID from experiment path like outputs/DTU/scan24/..."""
    parts = Path(experiment_path).parts
    for part in parts:
        if part.startswith('scan'):
            return part[4:]  # Remove 'scan' prefix
    return None


def find_existing_mesh(experiment_path, iteration=30000):
    """Look for existing mesh in various locations."""
    possible_paths = [
        os.path.join(experiment_path, 'train', f'ours_{iteration}', 'fuse_post.ply'),
        os.path.join(experiment_path, 'train', f'ours_{iteration}', 'fuse.ply'),
        os.path.join(experiment_path, f'fuse_post.ply'),
        os.path.join(experiment_path, f'fuse.ply'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def find_experiment_folders(root_dir, method_filter=None, scenes_filter=None):
    """Find all experiment folders containing trained models.

    Args:
        root_dir: Root directory to search (e.g., outputs/DTU)
        method_filter: Only include experiments with this method (e.g., 'baseline', 'cat')
        scenes_filter: Only include these scan IDs (e.g., ['24', '37', '40'])

    Returns:
        List of experiment paths
    """
    experiments = []
    root_path = Path(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirpath = Path(dirpath)

        # Check for point_cloud directory or ngp checkpoint
        has_point_cloud = 'point_cloud' in dirnames
        has_ngp_checkpoint = any(f.startswith('ngp_') and f.endswith('.pth') for f in filenames)

        if has_point_cloud or has_ngp_checkpoint:
            exp_path = str(dirpath)

            # Filter by scene if specified
            if scenes_filter:
                scan_id = get_scan_id(exp_path)
                if scan_id not in scenes_filter:
                    continue

            # Filter by method if specified
            if method_filter:
                log_method, _ = parse_training_log(exp_path)
                if log_method != method_filter:
                    continue

            experiments.append(exp_path)

    return sorted(experiments)


def run_mesh_extraction(experiment_path, source_path, iteration=30000, method='baseline', hybrid_levels=3):
    """Run eval_render.py to extract mesh if not already done."""
    # Check for existing mesh
    existing_mesh = find_existing_mesh(experiment_path, iteration)
    if existing_mesh:
        print(f"Mesh already exists: {existing_mesh}")
        return existing_mesh

    # Create output directory
    train_dir = os.path.join(experiment_path, 'train', f'ours_{iteration}')
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    print("Extracting mesh using eval_render.py...")
    cmd = [
        sys.executable, 'eval_render.py',
        '--iteration', str(iteration),
        '-s', source_path,
        '-m', experiment_path,
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
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Find the extracted mesh
    mesh_file = find_existing_mesh(experiment_path, iteration)
    if mesh_file:
        return mesh_file

    raise RuntimeError(f"Mesh extraction failed - no mesh found in {experiment_path}")


def run_chamfer_evaluation(mesh_file, scan_id, dtu_source, dtu_official, output_dir):
    """Run chamfer distance evaluation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, 'eval_dtu', 'evaluate_single_scene.py')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, eval_script,
        '--input_mesh', mesh_file,
        '--scan_id', scan_id,
        '--output_dir', output_dir,
        '--mask_dir', dtu_source,
        '--DTU', dtu_official,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read results
    results_file = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"\n=== Chamfer Distance Results for scan{scan_id} ===")
        print(f"Accuracy (d2s):    {results['mean_d2s']:.4f}")
        print(f"Completeness (s2d): {results['mean_s2d']:.4f}")
        print(f"Overall:           {results['overall']:.4f}")
        return results
    return None


def evaluate_single_experiment(experiment_path, args):
    """Evaluate a single experiment."""
    scan_id = get_scan_id(experiment_path)

    if scan_id is None:
        print(f"Error: Could not extract scan ID from path: {experiment_path}")
        return None

    # Auto-detect method and hybrid_levels from training_log.txt
    log_method, log_hybrid_levels = parse_training_log(experiment_path)

    # Use CLI args if specified, otherwise use auto-detected values, otherwise use defaults
    if args.method is not None:
        method = args.method
    elif log_method is not None:
        method = log_method
        print(f"[AUTO-DETECT] Method: {log_method} (from training_log.txt)")
    else:
        method = "baseline"
        print(f"[AUTO-DETECT] Method: baseline (default)")

    if args.hybrid_levels is not None:
        hybrid_levels = args.hybrid_levels
    elif log_hybrid_levels is not None:
        hybrid_levels = log_hybrid_levels
        print(f"[AUTO-DETECT] Hybrid levels: {log_hybrid_levels} (from training_log.txt)")
    else:
        hybrid_levels = 3

    print(f"\nEvaluating scan{scan_id} from {experiment_path}")
    print(f"Method: {method}, Hybrid levels: {hybrid_levels}")

    # Source path for the scan
    source_path = os.path.join(args.dtu_source, f'scan{scan_id}')
    if not os.path.exists(source_path):
        print(f"Error: DTU source not found: {source_path}")
        return None

    # Output directory
    output_dir = os.path.join(experiment_path, 'chamfer_eval')

    # Check if already evaluated
    results_file = os.path.join(output_dir, 'results.json')
    if args.skip_existing and os.path.exists(results_file):
        print(f"[SKIP] Already evaluated: {results_file}")
        # Load and return existing results
        try:
            with open(results_file) as f:
                results = json.load(f)
            results['scan_id'] = scan_id
            results['experiment'] = experiment_path
            results['method'] = method
            return results
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
            # Continue with re-evaluation

    # Step 1: Extract mesh
    try:
        if not args.skip_mesh:
            mesh_file = run_mesh_extraction(experiment_path, source_path, args.iteration, method, hybrid_levels)
        else:
            mesh_file = find_existing_mesh(experiment_path, args.iteration)
            if not mesh_file:
                print(f"Error: No mesh found in {experiment_path}")
                print("Run without --skip_mesh to extract mesh first")
                return None
    except Exception as e:
        print(f"Error extracting mesh: {e}")
        return None

    print(f"Using mesh: {mesh_file}")

    # Step 2: Run chamfer evaluation
    try:
        results = run_chamfer_evaluation(
            mesh_file, scan_id, args.dtu_source, args.dtu_official, output_dir
        )
        if results:
            results['scan_id'] = scan_id
            results['experiment'] = experiment_path
            results['method'] = method
        return results
    except Exception as e:
        print(f"Error in chamfer evaluation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chamfer Distance for DTU')

    # Single experiment mode
    parser.add_argument('--experiment', '-e', default=None,
                        help='Path to single experiment (e.g., outputs/DTU/scan24/baseline/vanilla_baseline)')

    # Batch mode
    parser.add_argument('--exp_dir', default=None,
                        help='Root directory containing experiment folders (e.g., outputs/DTU)')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Specific scan IDs to evaluate (e.g., 24 37 40)')

    # Method specification
    parser.add_argument('--method', type=str, default=None,
                        choices=["baseline", "cat", "adaptive", "adaptive_add", "adaptive_cat",
                                "diffuse", "specular", "diffuse_ngp", "diffuse_offset",
                                "hybrid_SH", "hybrid_SH_raw", "hybrid_SH_post", "residual_hybrid"],
                        help='Rendering method (auto-detected from training_log.txt if not specified)')
    parser.add_argument('--hybrid_levels', type=int, default=None,
                        help='Number of hybrid levels (auto-detected from training_log.txt if not specified)')

    # Data paths
    parser.add_argument('--dtu_source', default='data/dtu/2DGS_data/DTU',
                        help='Path to preprocessed DTU data (with cameras.npz, masks)')
    parser.add_argument('--dtu_official', default='data/dtu',
                        help='Path to official DTU data (with Points/ and ObsMask/)')

    # Other options
    parser.add_argument('--iteration', type=int, default=30000,
                        help='Training iteration to evaluate')
    parser.add_argument('--skip_mesh', action='store_true',
                        help='Skip mesh extraction (use existing mesh)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments that already have chamfer_eval/results.json')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiments that would be processed without running')

    args = parser.parse_args()

    # Validate arguments
    if args.experiment is None and args.exp_dir is None:
        parser.error("Either --experiment or --exp_dir must be specified")

    if args.experiment is not None and args.exp_dir is not None:
        parser.error("Cannot specify both --experiment and --exp_dir")

    # Check for ObsMask
    obsmask_path = os.path.join(args.dtu_official, 'ObsMask')
    if not os.path.exists(obsmask_path):
        print(f"Error: ObsMask not found at {obsmask_path}")
        print("Please download SampleSet.zip from DTU and extract ObsMask folder")
        sys.exit(1)

    # Single experiment mode
    if args.experiment is not None:
        experiment_path = os.path.abspath(args.experiment)
        results = evaluate_single_experiment(experiment_path, args)
        return results

    # Batch mode
    print(f"\n{'='*60}")
    print(f"Searching for experiments in: {args.exp_dir}")
    if args.method:
        print(f"Filtering by method: {args.method}")
    if args.scenes:
        print(f"Filtering by scenes: {args.scenes}")
    print(f"{'='*60}\n")

    # Convert scenes to strings if provided
    scenes_filter = [str(s) for s in args.scenes] if args.scenes else None

    experiments = find_experiment_folders(args.exp_dir, args.method, scenes_filter)

    if not experiments:
        print(f"No experiment folders found in {args.exp_dir}")
        if args.method:
            print(f"(with method filter: {args.method})")
        if args.scenes:
            print(f"(with scene filter: {args.scenes})")
        return

    print(f"Found {len(experiments)} experiment(s):\n")
    for exp in experiments:
        print(f"  - {exp}")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above experiments.")
        return

    print(f"\n{'='*60}")
    print("Starting batch evaluation...")
    print(f"{'='*60}\n")

    all_results = []
    successful = []
    failed = []

    for i, exp_path in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Processing: {exp_path}")
        print("-" * 60)

        results = evaluate_single_experiment(exp_path, args)
        if results:
            all_results.append(results)
            successful.append(exp_path)
        else:
            failed.append(exp_path)

    # Summary
    print(f"\n{'='*60}")
    print("Batch Evaluation Complete")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)}/{len(experiments)}")
    print(f"  Failed:     {len(failed)}/{len(experiments)}")

    if failed:
        print(f"\nFailed experiments:")
        for exp_path in failed:
            print(f"  - {exp_path}")

    if all_results:
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}")
        print(f"{'Scan':<8} {'Method':<12} {'Accuracy':<10} {'Complete':<10} {'Overall':<10}")
        print("-" * 60)

        total_d2s = 0
        total_s2d = 0
        total_overall = 0

        for r in all_results:
            print(f"scan{r['scan_id']:<4} {r['method']:<12} {r['mean_d2s']:<10.4f} {r['mean_s2d']:<10.4f} {r['overall']:<10.4f}")
            total_d2s += r['mean_d2s']
            total_s2d += r['mean_s2d']
            total_overall += r['overall']

        n = len(all_results)
        print("-" * 60)
        print(f"{'Average':<8} {'':<12} {total_d2s/n:<10.4f} {total_s2d/n:<10.4f} {total_overall/n:<10.4f}")

        # Save aggregate results
        aggregate_file = os.path.join(args.exp_dir, 'chamfer_aggregate.json')
        aggregate_data = {
            'experiments': all_results,
            'summary': {
                'num_scenes': n,
                'avg_d2s': total_d2s / n,
                'avg_s2d': total_s2d / n,
                'avg_overall': total_overall / n,
            }
        }
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate_data, f, indent=2)
        print(f"\nAggregate results saved to: {aggregate_file}")

    return all_results


if __name__ == '__main__':
    main()
