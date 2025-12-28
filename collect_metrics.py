#!/usr/bin/env python3
"""
Collect and organize metrics from all completed experiments.

A completed experiment has both test_metrics.txt and train_metrics.txt.
Outputs one file per scene + one combined file per dataset.
Each dataset gets its own metrics_reports subdirectory.

Includes:
- PSNR, SSIM, LPIPS tables (test set)
- FPS table
- Point count table
- Averages across scenes

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
        lpips_match = re.search(r'Average LPIPS:\s+([\d.]+)', content)
        images_match = re.search(r'Images rendered:\s+(\d+)', content)

        if psnr_match and ssim_match and l1_match:
            return {
                'psnr': float(psnr_match.group(1)),
                'ssim': float(ssim_match.group(1)),
                'l1': float(l1_match.group(1)),
                'lpips': float(lpips_match.group(1)) if lpips_match else None,
                'num_images': int(images_match.group(1)) if images_match else 0,
                'raw_content': content
            }
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")

    return None


def parse_training_log(filepath):
    """Parse training_log.txt for FPS and other stats."""
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        fps_match = re.search(r'Render FPS:\s+([\d.]+)', content)
        gaussians_match = re.search(r'Number of Gaussians:\s+([\d,]+)', content)

        result = {}
        if fps_match:
            result['fps'] = float(fps_match.group(1))
        if gaussians_match:
            result['gaussians'] = int(gaussians_match.group(1).replace(',', ''))

        return result if result else None
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")

    return None


def count_ply_points(ply_path):
    """Count the number of points in a PLY file by reading the header."""
    try:
        with open(ply_path, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('element vertex'):
                    return int(line.split()[-1])
                if line == 'end_header':
                    break
        return None
    except Exception:
        return None


def find_point_cloud(exp_dir):
    """Find the final point cloud PLY file for an experiment."""
    exp_path = Path(exp_dir)

    # Look for point_cloud/iteration_*/point_cloud.ply
    pc_dir = exp_path / 'point_cloud'
    if pc_dir.exists():
        # Find the highest iteration
        iterations = []
        for d in pc_dir.iterdir():
            if d.is_dir() and d.name.startswith('iteration_'):
                try:
                    iter_num = int(d.name.replace('iteration_', ''))
                    iterations.append((iter_num, d))
                except ValueError:
                    pass

        if iterations:
            iterations.sort(key=lambda x: x[0], reverse=True)
            ply_path = iterations[0][1] / 'point_cloud.ply'
            if ply_path.exists():
                return count_ply_points(ply_path)

    return None


def scan_completed_experiments(output_dir):
    """
    Scan for completed experiments (have both test and train metrics).

    Path structure: {output_dir}/{scene}/{method}/{name}/

    Returns:
        dict: {scene: [(method, name, test_metrics, train_metrics, training_log, point_count, exp_dir), ...]}
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
                training_log_path = exp_dir / 'training_log.txt'

                # Only include if BOTH exist (completed experiment)
                if test_path.exists() and train_path.exists():
                    test_metrics = parse_metrics_file(test_path)
                    train_metrics = parse_metrics_file(train_path)
                    training_log = parse_training_log(training_log_path)

                    # Get point count from PLY file, fallback to training_log
                    point_count = find_point_cloud(exp_dir)
                    if point_count is None and training_log and 'gaussians' in training_log:
                        point_count = training_log['gaussians']

                    if test_metrics and train_metrics:
                        experiments[scene].append((method, name, test_metrics, train_metrics, training_log, point_count, str(exp_dir)))

    return experiments


def format_summary_table(all_experiments, metric_key, metric_name, scenes):
    """Create a summary table for a specific metric across all scenes."""
    lines = []
    lines.append(f"{'='*120}")
    lines.append(f"  TEST {metric_name.upper()} BY SCENE")
    lines.append(f"{'='*120}")

    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _, _, _, _ in scene_exps:
            all_methods.add((method, name))

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
        values = []

        for scene in scenes:
            found = False
            for m, n, test, _, _, _, _ in all_experiments.get(scene, []):
                if m == method and n == name:
                    val = test.get(metric_key)
                    if val is not None:
                        row += f" {val:>8.4f}" if metric_key in ['ssim', 'lpips'] else f" {val:>8.2f}"
                        values.append(val)
                    else:
                        row += f" {'-':>8}"
                    found = True
                    break
            if not found:
                row += f" {'-':>8}"

        # Average
        if values:
            avg = sum(values) / len(values)
            row += f" {avg:>8.4f}" if metric_key in ['ssim', 'lpips'] else f" {avg:>8.2f}"
        else:
            row += f" {'-':>8}"

        lines.append(row)

    lines.append(f"{'='*120}")
    lines.append("")
    return "\n".join(lines)


def format_fps_table(all_experiments, scenes):
    """Create a FPS summary table across all scenes."""
    lines = []
    lines.append(f"{'='*120}")
    lines.append(f"  RENDER FPS BY SCENE")
    lines.append(f"{'='*120}")

    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _, _, _, _ in scene_exps:
            all_methods.add((method, name))

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
        fps_values = []

        for scene in scenes:
            found = False
            for m, n, _, _, training_log, _, _ in all_experiments.get(scene, []):
                if m == method and n == name:
                    if training_log and 'fps' in training_log:
                        fps = training_log['fps']
                        row += f" {fps:>8.1f}"
                        fps_values.append(fps)
                    else:
                        row += f" {'-':>8}"
                    found = True
                    break
            if not found:
                row += f" {'-':>8}"

        # Average
        if fps_values:
            avg = sum(fps_values) / len(fps_values)
            row += f" {avg:>8.1f}"
        else:
            row += f" {'-':>8}"

        lines.append(row)

    lines.append(f"{'='*120}")
    lines.append("")
    return "\n".join(lines)


def format_point_count_table(all_experiments, scenes):
    """Create a point count summary table across all scenes."""
    lines = []
    lines.append(f"{'='*140}")
    lines.append(f"  POINT COUNT BY SCENE (in thousands)")
    lines.append(f"{'='*140}")

    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _, _, _, _ in scene_exps:
            all_methods.add((method, name))

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
        point_values = []

        for scene in scenes:
            found = False
            for m, n, _, _, _, point_count, _ in all_experiments.get(scene, []):
                if m == method and n == name:
                    if point_count is not None:
                        # Display in thousands
                        row += f" {point_count/1000:>8.1f}"
                        point_values.append(point_count)
                    else:
                        row += f" {'-':>8}"
                    found = True
                    break
            if not found:
                row += f" {'-':>8}"

        # Average
        if point_values:
            avg = sum(point_values) / len(point_values)
            row += f" {avg/1000:>8.1f}"
        else:
            row += f" {'-':>8}"

        lines.append(row)

    lines.append(f"{'='*140}")
    lines.append("")
    return "\n".join(lines)


def format_combined_quality_table(all_experiments, scenes):
    """Create a combined quality table with PSNR, SSIM, LPIPS and avg point count."""
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"  COMBINED TEST METRICS (Averages across scenes)")
    lines.append(f"{'='*100}")

    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _, _, _, _, _ in scene_exps:
            all_methods.add((method, name))

    # Header
    header = f"{'Method':<12} {'Name':<35} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FPS':>8} {'#Pts(K)':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for method, name in sorted(all_methods):
        psnr_vals, ssim_vals, lpips_vals, fps_vals, point_vals = [], [], [], [], []

        for scene in scenes:
            for m, n, test, _, training_log, point_count, _ in all_experiments.get(scene, []):
                if m == method and n == name:
                    psnr_vals.append(test['psnr'])
                    ssim_vals.append(test['ssim'])
                    if test.get('lpips') is not None:
                        lpips_vals.append(test['lpips'])
                    if training_log and 'fps' in training_log:
                        fps_vals.append(training_log['fps'])
                    if point_count is not None:
                        point_vals.append(point_count)
                    break

        row = f"{method:<12} {name:<35}"

        # PSNR
        if psnr_vals:
            row += f" {sum(psnr_vals)/len(psnr_vals):>8.2f}"
        else:
            row += f" {'-':>8}"

        # SSIM
        if ssim_vals:
            row += f" {sum(ssim_vals)/len(ssim_vals):>8.4f}"
        else:
            row += f" {'-':>8}"

        # LPIPS
        if lpips_vals:
            row += f" {sum(lpips_vals)/len(lpips_vals):>8.4f}"
        else:
            row += f" {'-':>8}"

        # FPS
        if fps_vals:
            row += f" {sum(fps_vals)/len(fps_vals):>8.1f}"
        else:
            row += f" {'-':>8}"

        # Point count (in thousands)
        if point_vals:
            row += f" {sum(point_vals)/len(point_vals)/1000:>8.1f}"
        else:
            row += f" {'-':>8}"

        lines.append(row)

    lines.append(f"{'='*100}")
    lines.append("")
    return "\n".join(lines)


def format_metrics_table(data, split_name):
    """Format metrics as a text table."""
    if not data:
        return f"No {split_name} data available.\n"

    lines = []
    lines.append(f"{'='*90}")
    lines.append(f"  {split_name.upper()} METRICS")
    lines.append(f"{'='*90}")

    # Check if we have LPIPS data
    has_lpips = any(m.get('lpips') is not None for _, _, m in data)

    if has_lpips:
        lines.append(f"{'Method':<12} {'Name':<35} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'L1':>10}")
    else:
        lines.append(f"{'Method':<12} {'Name':<35} {'PSNR':>8} {'SSIM':>8} {'L1':>10}")
    lines.append(f"{'-'*90}")

    # Sort by method, then name
    for method, name, metrics in sorted(data, key=lambda x: (x[0], x[1])):
        if has_lpips:
            lpips_str = f"{metrics['lpips']:>8.4f}" if metrics.get('lpips') is not None else f"{'-':>8}"
            lines.append(
                f"{method:<12} {name:<35} {metrics['psnr']:>8.2f} {metrics['ssim']:>8.4f} "
                f"{lpips_str} {metrics['l1']:>10.6f}"
            )
        else:
            lines.append(
                f"{method:<12} {name:<35} {metrics['psnr']:>8.2f} {metrics['ssim']:>8.4f} "
                f"{metrics['l1']:>10.6f}"
            )

    lines.append(f"{'='*90}")
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
    test_data = [(m, n, t) for m, n, t, _, _, _, _ in experiments]
    train_data = [(m, n, tr) for m, n, _, tr, _, _, _ in experiments]

    # Test metrics
    lines.append(format_metrics_table(test_data, "TEST"))

    # Train metrics
    lines.append(format_metrics_table(train_data, "TRAIN"))

    # FPS and Point counts
    lines.append(f"{'='*80}")
    lines.append(f"  PERFORMANCE & SIZE")
    lines.append(f"{'='*80}")
    lines.append(f"{'Method':<12} {'Name':<35} {'FPS':>10} {'Points':>12}")
    lines.append(f"{'-'*80}")

    for method, name, _, _, training_log, point_count, _ in sorted(experiments, key=lambda x: (x[0], x[1])):
        fps_str = f"{training_log['fps']:>10.1f}" if training_log and 'fps' in training_log else f"{'-':>10}"
        pts_str = f"{point_count:>12,}" if point_count else f"{'-':>12}"
        lines.append(f"{method:<12} {name:<35} {fps_str} {pts_str}")

    lines.append(f"{'='*80}")
    lines.append("")

    # Write to file
    filepath = os.path.join(save_dir, f"{scene}_metrics.txt")
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

    return filepath


def create_combined_report(all_experiments, save_dir):
    """Create a combined report with all scenes."""
    scenes = sorted(all_experiments.keys())

    lines = []
    lines.append(f"{'#'*80}")
    lines.append(f"#  COMBINED METRICS REPORT")
    lines.append(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"#  Scenes: {len(all_experiments)}")
    total_exp = sum(len(v) for v in all_experiments.values())
    lines.append(f"#  Total completed experiments: {total_exp}")
    lines.append(f"{'#'*80}")
    lines.append("")

    # Combined quality table (averages)
    lines.append(format_combined_quality_table(all_experiments, scenes))

    # PSNR table
    lines.append(format_summary_table(all_experiments, 'psnr', 'PSNR', scenes))

    # SSIM table
    lines.append(format_summary_table(all_experiments, 'ssim', 'SSIM', scenes))

    # LPIPS table (if available)
    has_lpips = False
    for scene_exps in all_experiments.values():
        for _, _, test, _, _, _, _ in scene_exps:
            if test.get('lpips') is not None:
                has_lpips = True
                break
        if has_lpips:
            break

    if has_lpips:
        lines.append(format_summary_table(all_experiments, 'lpips', 'LPIPS', scenes))

    # FPS table
    lines.append(format_fps_table(all_experiments, scenes))

    # Point count table
    lines.append(format_point_count_table(all_experiments, scenes))

    # Per-scene details
    for scene in scenes:
        lines.append("")
        lines.append(f"{'#'*80}")
        lines.append(f"#  SCENE: {scene.upper()}")
        lines.append(f"{'#'*80}")
        lines.append("")

        experiments = all_experiments[scene]
        test_data = [(m, n, t) for m, n, t, _, _, _, _ in experiments]
        train_data = [(m, n, tr) for m, n, _, tr, _, _, _ in experiments]

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
        for method, name, _, _, _, _, _ in scene_exps:
            all_methods.add((method, name))

    # Test CSV
    test_csv_path = os.path.join(save_dir, "all_test_metrics.csv")
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Method', 'Name']
        for scene in scenes:
            header.extend([f'{scene}_PSNR', f'{scene}_SSIM', f'{scene}_LPIPS', f'{scene}_L1', f'{scene}_FPS', f'{scene}_Points'])
        header.extend(['Avg_PSNR', 'Avg_SSIM', 'Avg_LPIPS', 'Avg_L1', 'Avg_FPS', 'Avg_Points'])
        writer.writerow(header)

        # Data
        for method, name in sorted(all_methods):
            row = [method, name]
            psnr_vals, ssim_vals, lpips_vals, l1_vals, fps_vals, point_vals = [], [], [], [], [], []

            for scene in scenes:
                found = False
                for m, n, test, _, training_log, point_count, _ in all_experiments.get(scene, []):
                    if m == method and n == name:
                        row.append(f"{test['psnr']:.2f}")
                        row.append(f"{test['ssim']:.4f}")
                        row.append(f"{test['lpips']:.4f}" if test.get('lpips') is not None else '-')
                        row.append(f"{test['l1']:.6f}")
                        row.append(f"{training_log['fps']:.1f}" if training_log and 'fps' in training_log else '-')
                        row.append(f"{point_count}" if point_count else '-')

                        psnr_vals.append(test['psnr'])
                        ssim_vals.append(test['ssim'])
                        if test.get('lpips') is not None:
                            lpips_vals.append(test['lpips'])
                        l1_vals.append(test['l1'])
                        if training_log and 'fps' in training_log:
                            fps_vals.append(training_log['fps'])
                        if point_count:
                            point_vals.append(point_count)
                        found = True
                        break
                if not found:
                    row.extend(['-', '-', '-', '-', '-', '-'])

            # Averages
            row.append(f"{sum(psnr_vals)/len(psnr_vals):.2f}" if psnr_vals else '-')
            row.append(f"{sum(ssim_vals)/len(ssim_vals):.4f}" if ssim_vals else '-')
            row.append(f"{sum(lpips_vals)/len(lpips_vals):.4f}" if lpips_vals else '-')
            row.append(f"{sum(l1_vals)/len(l1_vals):.6f}" if l1_vals else '-')
            row.append(f"{sum(fps_vals)/len(fps_vals):.1f}" if fps_vals else '-')
            row.append(f"{sum(point_vals)/len(point_vals):.0f}" if point_vals else '-')

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
                for m, n, _, train, _, _, _ in all_experiments.get(scene, []):
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
