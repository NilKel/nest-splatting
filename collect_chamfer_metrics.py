#!/usr/bin/env python3
"""
Collect and organize Chamfer Distance metrics from DTU experiments.

Scans for chamfer_eval/results.json files in experiment directories
and creates summary reports.

Usage:
    # Process DTU dataset (default)
    python collect_chamfer_metrics.py

    # Custom paths
    python collect_chamfer_metrics.py --outputs_dir outputs/DTU --save_dir metrics_reports/DTU
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_chamfer_results(filepath):
    """Parse a chamfer results.json file."""
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        return {
            'accuracy': data.get('mean_d2s', 0),  # d2s = distance to surface (accuracy)
            'completeness': data.get('mean_s2d', 0),  # s2d = surface to distance (completeness)
            'overall': data.get('overall', 0),
        }
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
        return None


def scan_chamfer_results(output_dir):
    """
    Scan for chamfer evaluation results.

    Path structure: {output_dir}/{scene}/{method}/{name}/chamfer_eval/results.json

    Returns:
        dict: {scene: [(method, name, chamfer_metrics), ...]}
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

                chamfer_path = exp_dir / 'chamfer_eval' / 'results.json'

                if chamfer_path.exists():
                    chamfer_metrics = parse_chamfer_results(chamfer_path)

                    if chamfer_metrics:
                        experiments[scene].append((method, name, chamfer_metrics))

    return experiments


def format_chamfer_table(data):
    """Format chamfer metrics as a text table."""
    if not data:
        return "No Chamfer distance data available.\n"

    lines = []
    lines.append(f"{'Method':<12} {'Name':<35} {'Accuracy':>10} {'Complete':>10} {'Overall':>10}")
    lines.append(f"{'-'*80}")

    # Sort by method, then name
    for method, name, metrics in sorted(data, key=lambda x: (x[0], x[1])):
        lines.append(
            f"{method:<12} {name:<35} {metrics['accuracy']:>10.4f} {metrics['completeness']:>10.4f} "
            f"{metrics['overall']:>10.4f}"
        )

    return "\n".join(lines)


def create_chamfer_report(all_experiments, save_dir):
    """Create Chamfer distance report."""
    lines = []
    lines.append(f"{'#'*80}")
    lines.append(f"#  CHAMFER DISTANCE METRICS REPORT (DTU)")
    lines.append(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"#  Scenes: {len(all_experiments)}")
    total_exp = sum(len(v) for v in all_experiments.values())
    lines.append(f"#  Total experiments with Chamfer eval: {total_exp}")
    lines.append(f"#")
    lines.append(f"#  Metrics:")
    lines.append(f"#    Accuracy (d2s):    Mean distance from reconstruction to GT (lower is better)")
    lines.append(f"#    Completeness (s2d): Mean distance from GT to reconstruction (lower is better)")
    lines.append(f"#    Overall:           Average of Accuracy and Completeness (lower is better)")
    lines.append(f"{'#'*80}")
    lines.append("")

    # Summary table across all scenes
    lines.append(f"{'='*100}")
    lines.append("  SUMMARY: CHAMFER DISTANCE (OVERALL) BY SCENE AND METHOD")
    lines.append(f"{'='*100}")

    # Collect all methods
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _ in scene_exps:
            all_methods.add((method, name))

    # Sort scenes (extract numeric part for proper sorting)
    def scene_sort_key(s):
        # Extract number from 'scan24' -> 24
        import re
        match = re.search(r'\d+', s)
        return int(match.group()) if match else s

    scenes = sorted(all_experiments.keys(), key=scene_sort_key)

    # Header
    header = f"{'Method':<12} {'Name':<25}"
    for scene in scenes:
        # Shorten scene name (scan24 -> s24)
        short_scene = scene.replace('scan', 's')
        header += f" {short_scene:>6}"
    header += f" {'Avg':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for method, name in sorted(all_methods):
        row = f"{method:<12} {name:<25}"
        values = []

        for scene in scenes:
            # Find this experiment in this scene
            found = False
            for m, n, metrics in all_experiments.get(scene, []):
                if m == method and n == name:
                    row += f" {metrics['overall']:>6.3f}"
                    values.append(metrics['overall'])
                    found = True
                    break
            if not found:
                row += f" {'-':>6}"

        # Average
        if values:
            avg = sum(values) / len(values)
            row += f" {avg:>8.4f}"
        else:
            row += f" {'-':>8}"

        lines.append(row)

    lines.append(f"{'='*100}")
    lines.append("")

    # Detailed tables per scene
    for scene in scenes:
        lines.append("")
        lines.append(f"{'='*80}")
        lines.append(f"  {scene.upper()}")
        lines.append(f"{'='*80}")

        experiments = all_experiments.get(scene, [])
        lines.append(format_chamfer_table(experiments))
        lines.append("")

    # Write to file
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "chamfer_metrics.txt")
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))

    return filepath


def create_chamfer_csv(all_experiments, save_dir):
    """Create CSV file for Chamfer metrics."""
    import csv

    def scene_sort_key(s):
        import re
        match = re.search(r'\d+', s)
        return int(match.group()) if match else s

    scenes = sorted(all_experiments.keys(), key=scene_sort_key)

    # Collect all unique (method, name) pairs
    all_methods = set()
    for scene_exps in all_experiments.values():
        for method, name, _ in scene_exps:
            all_methods.add((method, name))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "chamfer_metrics.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Method', 'Name']
        for scene in scenes:
            header.extend([f'{scene}_Acc', f'{scene}_Comp', f'{scene}_Overall'])
        header.extend(['Avg_Acc', 'Avg_Comp', 'Avg_Overall'])
        writer.writerow(header)

        # Data
        for method, name in sorted(all_methods):
            row = [method, name]
            acc_vals, comp_vals, overall_vals = [], [], []

            for scene in scenes:
                found = False
                for m, n, metrics in all_experiments.get(scene, []):
                    if m == method and n == name:
                        row.extend([
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['completeness']:.4f}",
                            f"{metrics['overall']:.4f}"
                        ])
                        acc_vals.append(metrics['accuracy'])
                        comp_vals.append(metrics['completeness'])
                        overall_vals.append(metrics['overall'])
                        found = True
                        break
                if not found:
                    row.extend(['-', '-', '-'])

            # Averages
            if acc_vals:
                row.extend([
                    f"{sum(acc_vals)/len(acc_vals):.4f}",
                    f"{sum(comp_vals)/len(comp_vals):.4f}",
                    f"{sum(overall_vals)/len(overall_vals):.4f}"
                ])
            else:
                row.extend(['-', '-', '-'])

            writer.writerow(row)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Collect Chamfer Distance metrics from DTU experiments")
    parser.add_argument(
        '--outputs_dir',
        type=str,
        default='/home/nilkel/Projects/nest-splatting/outputs/DTU',
        help='DTU outputs directory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='metrics_reports/DTU',
        help='Directory to save reports'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Collecting Chamfer Distance Metrics")
    print("=" * 70)
    print(f"Scanning: {args.outputs_dir}")
    print(f"Save to:  {args.save_dir}")
    print("=" * 70)
    print()

    # Scan experiments
    experiments = scan_chamfer_results(args.outputs_dir)

    if not experiments:
        print("No Chamfer evaluation results found!")
        print("Run eval_chamfer.py first to generate chamfer_eval/results.json files.")
        return

    # Report findings
    total = sum(len(v) for v in experiments.values())
    print(f"Found {len(experiments)} scene(s) with {total} Chamfer evaluation(s):")

    def scene_sort_key(s):
        import re
        match = re.search(r'\d+', s)
        return int(match.group()) if match else s

    for scene in sorted(experiments.keys(), key=scene_sort_key):
        print(f"  - {scene}: {len(experiments[scene])} experiments")
    print()

    # Create report
    print("Creating Chamfer distance report...")
    report_path = create_chamfer_report(experiments, args.save_dir)
    print(f"  {report_path}")

    # Create CSV
    print("\nCreating CSV file...")
    csv_path = create_chamfer_csv(experiments, args.save_dir)
    print(f"  {csv_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"Reports saved to: {args.save_dir}/")
    print(f"  - chamfer_metrics.txt")
    print(f"  - chamfer_metrics.csv")
    print()


if __name__ == "__main__":
    main()
