#!/usr/bin/env python3
"""
Collect and organize metrics from all completed experiments.

A completed experiment has both test_metrics.txt and train_metrics.txt.
Outputs one file per scene + one combined file per dataset.
Each dataset gets its own metrics_reports subdirectory.

Usage:
    # Process all datasets in outputs/
    python collect_metrics.py

    # Process a specific dataset
    python collect_metrics.py --dataset nerf_synthetic

    # Custom outputs and save directories
    python collect_metrics.py --outputs_base_dir /path/to/outputs --save_base_dir my_reports
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_metrics_file(filepath):
    """Parse a metrics file and extract summary + per-image data."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract summary metrics
        psnr_match = re.search(r'Average PSNR:\s+([\d.]+)\s+dB', content)
        ssim_match = re.search(r'Average SSIM:\s+([\d.]+)', content)
        l1_match = re.search(r'Average L1:\s+([\d.]+)', content)
        images_match = re.search(r'Images rendered:\s+(\d+)', content)
        
        if psnr_match and ssim_match and l1_match:
            return {
                'psnr': float(psnr_match.group(1)),
                'ssim': float(ssim_match.group(1)),
                'l1': float(l1_match.group(1)),
                'num_images': int(images_match.group(1)) if images_match else 0,
                'raw_content': content
            }
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
    
    return None


def scan_completed_experiments(output_dir):
    """
    Scan for completed experiments (have both test and train metrics).
    
    Path structure: {output_dir}/{scene}/{method}/{name}/
    
    Returns:
        dict: {scene: [(method, name, test_metrics, train_metrics), ...]}
    """
    experiments = defaultdict(list)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory does not exist: {output_dir}")
        return experiments
    
    # Scan: {scene}/{method}/{name}/
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene = scene_dir.name
        
        for method_dir in scene_dir.iterdir():
            if not method_dir.is_dir():
                continue
            
            method = method_dir.name
            
            for exp_dir in method_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                name = exp_dir.name
                
                test_path = exp_dir / 'test_metrics.txt'
                train_path = exp_dir / 'train_metrics.txt'
                
                # Only include if BOTH exist (completed experiment)
                if test_path.exists() and train_path.exists():
                    test_metrics = parse_metrics_file(test_path)
                    train_metrics = parse_metrics_file(train_path)
                    
                    if test_metrics and train_metrics:
                        experiments[scene].append((method, name, test_metrics, train_metrics))
    
    return experiments


def format_metrics_table(data, split_name):
    """Format metrics as a text table."""
    if not data:
        return f"No {split_name} data available.\n"
    
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"  {split_name.upper()} METRICS")
    lines.append(f"{'='*80}")
    lines.append(f"{'Method':<12} {'Name':<35} {'PSNR':>8} {'SSIM':>8} {'L1':>10} {'#Img':>6}")
    lines.append(f"{'-'*80}")
    
    # Sort by method, then name
    for method, name, metrics in sorted(data, key=lambda x: (x[0], x[1])):
        lines.append(
            f"{method:<12} {name:<35} {metrics['psnr']:>8.2f} {metrics['ssim']:>8.4f} "
            f"{metrics['l1']:>10.6f} {metrics['num_images']:>6}"
        )
    
    lines.append(f"{'='*80}")
    lines.append("")
    return "\n".join(lines)


def create_scene_report(scene, experiments, save_dir):
    """Create a report file for a single scene."""
    lines = []
    lines.append(f"{'#'*80}")
    lines.append(f"#  SCENE: {scene.upper()}")
    lines.append(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"#  Completed experiments: {len(experiments)}")
    lines.append(f"{'#'*80}")
    lines.append("")
    
    # Prepare test and train data
    test_data = [(m, n, t) for m, n, t, _ in experiments]
    train_data = [(m, n, tr) for m, n, _, tr in experiments]
    
    # Test metrics
    lines.append(format_metrics_table(test_data, "TEST"))
    
    # Train metrics
    lines.append(format_metrics_table(train_data, "TRAIN"))
    
    # Write to file
    filepath = os.path.join(save_dir, f"{scene}_metrics.txt")
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    return filepath


def create_combined_report(all_experiments, save_dir):
    """Create a combined report with all scenes."""
    lines = []
    lines.append(f"{'#'*80}")
    lines.append(f"#  COMBINED METRICS REPORT")
    lines.append(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"#  Scenes: {len(all_experiments)}")
    total_exp = sum(len(v) for v in all_experiments.values())
    lines.append(f"#  Total completed experiments: {total_exp}")
    lines.append(f"{'#'*80}")
    lines.append("")
    
    # Summary table across all scenes
    lines.append(f"{'='*80}")
    lines.append("  SUMMARY: TEST PSNR BY SCENE AND METHOD")
    lines.append(f"{'='*80}")
    
    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _ in scene_exps:
            all_methods.add((method, name))
    
    # Sort scenes
    scenes = sorted(all_experiments.keys())
    
    # Header
    header = f"{'Method':<12} {'Name':<30}"
    for scene in scenes:
        header += f" {scene[:8]:>8}"
    header += f" {'Avg':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for method, name in sorted(all_methods):
        row = f"{method:<12} {name:<30}"
        psnr_values = []
        
        for scene in scenes:
            # Find this experiment in this scene
            found = False
            for m, n, test, _ in all_experiments.get(scene, []):
                if m == method and n == name:
                    row += f" {test['psnr']:>8.2f}"
                    psnr_values.append(test['psnr'])
                    found = True
                    break
            if not found:
                row += f" {'-':>8}"
        
        # Average
        if psnr_values:
            avg = sum(psnr_values) / len(psnr_values)
            row += f" {avg:>8.2f}"
        else:
            row += f" {'-':>8}"
        
        lines.append(row)
    
    lines.append(f"{'='*80}")
    lines.append("")
    
    # Per-scene details
    for scene in scenes:
        lines.append("")
        lines.append(f"{'#'*80}")
        lines.append(f"#  SCENE: {scene.upper()}")
        lines.append(f"{'#'*80}")
        lines.append("")
        
        experiments = all_experiments[scene]
        test_data = [(m, n, t) for m, n, t, _ in experiments]
        train_data = [(m, n, tr) for m, n, _, tr in experiments]
        
        lines.append(format_metrics_table(test_data, "TEST"))
        lines.append(format_metrics_table(train_data, "TRAIN"))
    
    # Write to file
    filepath = os.path.join(save_dir, "all_metrics.txt")
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    return filepath


def create_csv_summary(all_experiments, save_dir):
    """Create CSV files for easy import to spreadsheets."""
    import csv
    
    scenes = sorted(all_experiments.keys())
    
    # Collect all unique (method, name) pairs
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _ in scene_exps:
            all_methods.add((method, name))
    
    # Test CSV
    test_csv_path = os.path.join(save_dir, "all_test_metrics.csv")
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Method', 'Name']
        for scene in scenes:
            header.extend([f'{scene}_PSNR', f'{scene}_SSIM', f'{scene}_L1'])
        header.extend(['Avg_PSNR', 'Avg_SSIM', 'Avg_L1'])
        writer.writerow(header)
        
        # Data
        for method, name in sorted(all_methods):
            row = [method, name]
            psnr_vals, ssim_vals, l1_vals = [], [], []
            
            for scene in scenes:
                found = False
                for m, n, test, _ in all_experiments.get(scene, []):
                    if m == method and n == name:
                        row.extend([f"{test['psnr']:.2f}", f"{test['ssim']:.4f}", f"{test['l1']:.6f}"])
                        psnr_vals.append(test['psnr'])
                        ssim_vals.append(test['ssim'])
                        l1_vals.append(test['l1'])
                        found = True
                        break
                if not found:
                    row.extend(['-', '-', '-'])
            
            # Averages
            if psnr_vals:
                row.extend([
                    f"{sum(psnr_vals)/len(psnr_vals):.2f}",
                    f"{sum(ssim_vals)/len(ssim_vals):.4f}",
                    f"{sum(l1_vals)/len(l1_vals):.6f}"
                ])
            else:
                row.extend(['-', '-', '-'])
            
            writer.writerow(row)
    
    # Train CSV
    train_csv_path = os.path.join(save_dir, "all_train_metrics.csv")
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Method', 'Name']
        for scene in scenes:
            header.extend([f'{scene}_PSNR', f'{scene}_SSIM', f'{scene}_L1'])
        header.extend(['Avg_PSNR', 'Avg_SSIM', 'Avg_L1'])
        writer.writerow(header)
        
        # Data
        for method, name in sorted(all_methods):
            row = [method, name]
            psnr_vals, ssim_vals, l1_vals = [], [], []
            
            for scene in scenes:
                found = False
                for m, n, _, train in all_experiments.get(scene, []):
                    if m == method and n == name:
                        row.extend([f"{train['psnr']:.2f}", f"{train['ssim']:.4f}", f"{train['l1']:.6f}"])
                        psnr_vals.append(train['psnr'])
                        ssim_vals.append(train['ssim'])
                        l1_vals.append(train['l1'])
                        found = True
                        break
                if not found:
                    row.extend(['-', '-', '-'])
            
            # Averages
            if psnr_vals:
                row.extend([
                    f"{sum(psnr_vals)/len(psnr_vals):.2f}",
                    f"{sum(ssim_vals)/len(ssim_vals):.4f}",
                    f"{sum(l1_vals)/len(l1_vals):.6f}"
                ])
            else:
                row.extend(['-', '-', '-'])
            
            writer.writerow(row)
    
    return test_csv_path, train_csv_path


def process_dataset(dataset_name, dataset_path, base_save_dir):
    """Process a single dataset and create its reports."""
    save_dir = os.path.join(base_save_dir, dataset_name)

    print("=" * 70)
    print(f"  Processing Dataset: {dataset_name}")
    print("=" * 70)
    print(f"Scanning: {dataset_path}")
    print(f"Save to:  {save_dir}")
    print("=" * 70)
    print()

    # Scan experiments
    experiments = scan_completed_experiments(dataset_path)

    if not experiments:
        print(f"No completed experiments found in {dataset_name}!")
        print("(Completed = has both test_metrics.txt and train_metrics.txt)")
        print()
        return False

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Report findings
    total = sum(len(v) for v in experiments.values())
    print(f"Found {len(experiments)} scene(s) with {total} completed experiment(s):")
    for scene in sorted(experiments.keys()):
        print(f"  - {scene}: {len(experiments[scene])} experiments")
    print()

    # Create per-scene reports
    print("Creating per-scene reports...")
    for scene in sorted(experiments.keys()):
        filepath = create_scene_report(scene, experiments[scene], save_dir)
        print(f"  {filepath}")

    # Create combined report
    print("\nCreating combined report...")
    combined_path = create_combined_report(experiments, save_dir)
    print(f"  {combined_path}")

    # Create CSV files
    print("\nCreating CSV files...")
    test_csv, train_csv = create_csv_summary(experiments, save_dir)
    print(f"  {test_csv}")
    print(f"  {train_csv}")

    print("\n" + "=" * 70)
    print(f"  {dataset_name} COMPLETE!")
    print("=" * 70)
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description="Collect metrics from completed experiments")
    parser.add_argument(
        '--outputs_base_dir',
        type=str,
        default='/home/nilkel/Projects/nest-splatting/outputs',
        help='Base outputs directory containing dataset subdirectories'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Specific dataset to process (e.g., nerf_synthetic, DTU). If not provided, processes all datasets.'
    )
    parser.add_argument(
        '--save_base_dir',
        type=str,
        default='metrics_reports',
        help='Base directory to save reports (each dataset gets its own subfolder)'
    )

    args = parser.parse_args()

    outputs_path = Path(args.outputs_base_dir)

    if not outputs_path.exists():
        print(f"Error: Outputs directory does not exist: {args.outputs_base_dir}")
        return

    # Find all dataset directories
    if args.dataset:
        # Process specific dataset
        dataset_path = outputs_path / args.dataset
        if not dataset_path.exists():
            print(f"Error: Dataset directory does not exist: {dataset_path}")
            return
        datasets = [(args.dataset, dataset_path)]
    else:
        # Process all datasets
        datasets = [(d.name, d) for d in outputs_path.iterdir() if d.is_dir()]

    if not datasets:
        print("No dataset directories found!")
        return

    print("=" * 70)
    print("  Collecting Metrics from Completed Experiments")
    print("=" * 70)
    print(f"Outputs base: {args.outputs_base_dir}")
    print(f"Reports base: {args.save_base_dir}")
    print(f"Datasets to process: {len(datasets)}")
    for name, _ in datasets:
        print(f"  - {name}")
    print("=" * 70)
    print()

    # Process each dataset
    processed_count = 0
    for dataset_name, dataset_path in sorted(datasets):
        if process_dataset(dataset_name, str(dataset_path), args.save_base_dir):
            processed_count += 1

    print("\n" + "=" * 70)
    print("  ALL DATASETS COMPLETE!")
    print("=" * 70)
    print(f"Processed {processed_count}/{len(datasets)} dataset(s)")
    print(f"\nReports saved to: {args.save_base_dir}/<dataset_name>/")
    print(f"  - Per-scene: <scene>_metrics.txt")
    print(f"  - Combined:  all_metrics.txt")
    print(f"  - CSV:       all_test_metrics.csv, all_train_metrics.csv")
    print()


if __name__ == "__main__":
    main()

