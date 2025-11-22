#!/usr/bin/env python3
"""
Script to parse experiment metrics and create tables organized by scene.
Scans outputs/method/dataset/scene/name/ for test_metrics.txt and train_metrics.txt files.
Creates train and test tables for each scene with method/name as row identifiers.

Usage:
    python create_metrics_tables.py [--output_dir outputs] [--format csv|markdown|both]
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import colorsys


def parse_metrics_file(filepath):
    """
    Parse a metrics file and extract average PSNR, SSIM, and L1.
    
    Returns:
        dict: {'psnr': float, 'ssim': float, 'l1': float, 'num_images': int}
        None if file doesn't exist or parsing fails
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract metrics using regex
        psnr_match = re.search(r'Average PSNR:\s+([\d.]+)\s+dB', content)
        ssim_match = re.search(r'Average SSIM:\s+([\d.]+)', content)
        l1_match = re.search(r'Average L1:\s+([\d.]+)', content)
        images_match = re.search(r'Images rendered:\s+(\d+)', content)
        
        if psnr_match and ssim_match and l1_match:
            return {
                'psnr': float(psnr_match.group(1)),
                'ssim': float(ssim_match.group(1)),
                'l1': float(l1_match.group(1)),
                'num_images': int(images_match.group(1)) if images_match else 0
            }
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
    
    return None


def scan_experiments(output_dir):
    """
    Scan the outputs directory for experiments with metrics files.
    
    Returns:
        dict: {
            scene_name: {
                'test': [(method, name, metrics_dict), ...],
                'train': [(method, name, metrics_dict), ...]
            }
        }
    """
    experiments = defaultdict(lambda: {'test': [], 'train': []})
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return experiments
    
    # Scan outputs/method/dataset/scene/name/
    for method_dir in output_path.iterdir():
        if not method_dir.is_dir():
            continue
        
        method = method_dir.name
        
        for dataset_dir in method_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name
            
            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                
                scene = scene_dir.name
                
                for name_dir in scene_dir.iterdir():
                    if not name_dir.is_dir():
                        continue
                    
                    name = name_dir.name
                    
                    # Check for metrics files at the top level
                    test_metrics_path = name_dir / "test_metrics.txt"
                    train_metrics_path = name_dir / "train_metrics.txt"
                    
                    has_test = test_metrics_path.exists()
                    has_train = train_metrics_path.exists()
                    
                    if not (has_test or has_train):
                        continue
                    
                    # Parse test metrics
                    if has_test:
                        test_metrics = parse_metrics_file(test_metrics_path)
                        if test_metrics:
                            experiments[scene]['test'].append((method, name, test_metrics))
                    
                    # Parse train metrics
                    if has_train:
                        train_metrics = parse_metrics_file(train_metrics_path)
                        if train_metrics:
                            experiments[scene]['train'].append((method, name, train_metrics))
    
    return experiments


def create_markdown_table(scene, split, data):
    """
    Create a markdown table for a specific scene and split (test/train).
    
    Args:
        scene: Scene name
        split: 'test' or 'train'
        data: List of (method, name, metrics_dict) tuples
    
    Returns:
        str: Markdown table
    """
    if not data:
        return f"No {split} data available for {scene}\n"
    
    # Sort by method, then name
    data = sorted(data, key=lambda x: (x[0], x[1]))
    
    lines = []
    lines.append(f"## {scene.upper()} - {split.upper()} Metrics\n")
    lines.append("| Method | Experiment Name | PSNR (dB) | SSIM | L1 | # Images |")
    lines.append("|--------|-----------------|-----------|------|-----|----------|")
    
    for method, name, metrics in data:
        lines.append(
            f"| {method} | {name} | {metrics['psnr']:.2f} | "
            f"{metrics['ssim']:.4f} | {metrics['l1']:.6f} | {metrics['num_images']} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def get_color_for_value(value, min_val, max_val, ascending=True):
    """
    Get RGB color for a value based on its position between min and max.
    
    Args:
        value: Current value
        min_val: Minimum value in dataset
        max_val: Maximum value in dataset
        ascending: If True, higher values are better (green). If False, lower values are better.
    
    Returns:
        str: RGB color string like "rgb(255, 0, 0)"
    """
    if max_val == min_val:
        # All values are the same
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # If descending metric (like L1), flip the normalization
    if not ascending:
        normalized = 1.0 - normalized
    
    # Red (0) -> Yellow (0.5) -> Green (1.0)
    # Using HSV color space: Hue goes from 0 (red) to 120 (green)
    hue = normalized * 120 / 360  # Convert to 0-1 range for colorsys
    saturation = 0.7
    value_brightness = 0.9
    
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value_brightness)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


def create_html_table(scene, split, data):
    """
    Create an HTML table with color-coded cells for a specific scene and split.
    
    Args:
        scene: Scene name
        split: 'test' or 'train'
        data: List of (method, name, metrics_dict) tuples
    
    Returns:
        str: HTML table
    """
    if not data:
        return f"<p>No {split} data available for {scene}</p>\n"
    
    # Sort by method, then name
    data = sorted(data, key=lambda x: (x[0], x[1]))
    
    # Calculate min/max for each metric for color scaling
    psnr_values = [m[2]['psnr'] for m in data]
    ssim_values = [m[2]['ssim'] for m in data]
    l1_values = [m[2]['l1'] for m in data]
    
    psnr_min, psnr_max = min(psnr_values), max(psnr_values)
    ssim_min, ssim_max = min(ssim_values), max(ssim_values)
    l1_min, l1_max = min(l1_values), max(l1_values)
    
    lines = []
    lines.append(f'<h2>{scene.upper()} - {split.upper()} Metrics</h2>')
    lines.append('<table class="metrics-table">')
    lines.append('<thead>')
    lines.append('<tr>')
    lines.append('<th>Method</th>')
    lines.append('<th>Experiment Name</th>')
    lines.append('<th>PSNR (dB) ↑</th>')
    lines.append('<th>SSIM ↑</th>')
    lines.append('<th>L1 ↓</th>')
    lines.append('<th># Images</th>')
    lines.append('</tr>')
    lines.append('</thead>')
    lines.append('<tbody>')
    
    for method, name, metrics in data:
        psnr_color = get_color_for_value(metrics['psnr'], psnr_min, psnr_max, ascending=True)
        ssim_color = get_color_for_value(metrics['ssim'], ssim_min, ssim_max, ascending=True)
        l1_color = get_color_for_value(metrics['l1'], l1_min, l1_max, ascending=False)
        
        lines.append('<tr>')
        lines.append(f'<td class="method">{method}</td>')
        lines.append(f'<td class="name">{name}</td>')
        lines.append(f'<td class="metric" style="background-color: {psnr_color};">{metrics["psnr"]:.2f}</td>')
        lines.append(f'<td class="metric" style="background-color: {ssim_color};">{metrics["ssim"]:.4f}</td>')
        lines.append(f'<td class="metric" style="background-color: {l1_color};">{metrics["l1"]:.6f}</td>')
        lines.append(f'<td class="count">{metrics["num_images"]}</td>')
        lines.append('</tr>')
    
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('')
    
    return '\n'.join(lines)


def create_html_document(scene, test_data, train_data):
    """
    Create a complete HTML document with both test and train tables for a scene.
    
    Args:
        scene: Scene name
        test_data: List of (method, name, metrics_dict) tuples for test
        train_data: List of (method, name, metrics_dict) tuples for train
    
    Returns:
        str: Complete HTML document
    """
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="en">')
    html.append('<head>')
    html.append('<meta charset="UTF-8">')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html.append(f'<title>{scene.upper()} - Experiment Metrics</title>')
    html.append('<style>')
    html.append("""
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1400px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 40px;
        }
        .metrics-table thead {
            background-color: #333;
            color: white;
        }
        .metrics-table th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        .metrics-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        .metrics-table tbody tr:hover {
            background-color: #f9f9f9;
        }
        .metrics-table .method {
            font-weight: 600;
            color: #2196F3;
        }
        .metrics-table .name {
            color: #666;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .metrics-table .metric {
            text-align: center;
            font-weight: 600;
            font-family: 'Courier New', monospace;
            color: #000;
            text-shadow: 0 0 2px rgba(255,255,255,0.8);
        }
        .metrics-table .count {
            text-align: center;
            color: #999;
        }
        .info {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin-bottom: 30px;
            border-radius: 4px;
        }
        .legend {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #666;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .color-sample {
            width: 60px;
            height: 20px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
    """)
    html.append('</style>')
    html.append('</head>')
    html.append('<body>')
    html.append(f'<h1>{scene.upper()} - Experiment Metrics</h1>')
    html.append('<div class="info">')
    html.append('<strong>Color Coding:</strong> Cells are color-coded from red (worst) to green (best) relative to other experiments.')
    html.append('</div>')
    html.append('<div class="legend">')
    html.append('<div class="legend-item">')
    html.append('<span><strong>↑</strong> Higher is better (PSNR, SSIM)</span>')
    html.append('</div>')
    html.append('<div class="legend-item">')
    html.append('<span><strong>↓</strong> Lower is better (L1)</span>')
    html.append('</div>')
    html.append('<div class="legend-item">')
    html.append('<div class="color-sample" style="background: linear-gradient(to right, rgb(230, 57, 57), rgb(230, 230, 57), rgb(57, 230, 57));"></div>')
    html.append('<span>Worst → Best</span>')
    html.append('</div>')
    html.append('</div>')
    
    # Add test table
    if test_data:
        html.append(create_html_table(scene, 'test', test_data))
    
    # Add train table
    if train_data:
        html.append(create_html_table(scene, 'train', train_data))
    
    html.append('</body>')
    html.append('</html>')
    
    return '\n'.join(html)


def create_csv_table(scene, split, data, output_file):
    """
    Create a CSV table for a specific scene and split (test/train).
    
    Args:
        scene: Scene name
        split: 'test' or 'train'
        data: List of (method, name, metrics_dict) tuples
        output_file: Path to output CSV file
    """
    if not data:
        print(f"No {split} data available for {scene}")
        return
    
    # Sort by method, then name
    data = sorted(data, key=lambda x: (x[0], x[1]))
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Experiment_Name', 'PSNR_dB', 'SSIM', 'L1', 'Num_Images'])
        
        for method, name, metrics in data:
            writer.writerow([
                method,
                name,
                f"{metrics['psnr']:.2f}",
                f"{metrics['ssim']:.4f}",
                f"{metrics['l1']:.6f}",
                metrics['num_images']
            ])
    
    print(f"  Saved CSV: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create metrics tables from experiment results"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Base output directory to scan (default: outputs)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='metrics_tables',
        help='Directory to save tables (default: metrics_tables)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Nest-Splatting Metrics Table Generator")
    print("="*70)
    print(f"Scanning: {args.output_dir}")
    print(f"Format:   {args.format}")
    print(f"Save to:  {args.save_dir}")
    print("="*70)
    print()
    
    # Scan experiments
    experiments = scan_experiments(args.output_dir)
    
    if not experiments:
        print("No experiments found with top-level test_metrics.txt or train_metrics.txt")
        return
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Process each scene
    scenes = sorted(experiments.keys())
    print(f"Found {len(scenes)} scene(s): {', '.join(scenes)}\n")
    
    all_markdown = []
    all_markdown.append("# Nest-Splatting Experiment Metrics\n")
    all_markdown.append(f"Generated from: {args.output_dir}\n")
    all_markdown.append("---\n")
    
    for scene in scenes:
        print(f"\nProcessing scene: {scene}")
        
        # Create per-scene markdown with both train and test
        scene_markdown = []
        scene_markdown.append(f"# {scene.upper()} - Experiment Metrics\n")
        scene_markdown.append(f"Generated from: {args.output_dir}\n")
        scene_markdown.append("---\n")
        
        # Test metrics
        if experiments[scene]['test']:
            print(f"  Found {len(experiments[scene]['test'])} test experiments")
            
            test_md_table = create_markdown_table(scene, 'test', experiments[scene]['test'])
            
            if args.format in ['markdown', 'both']:
                all_markdown.append(test_md_table)
                scene_markdown.append(test_md_table)
            
            if args.format in ['csv', 'both']:
                csv_file = os.path.join(args.save_dir, f"{scene}_test_metrics.csv")
                create_csv_table(scene, 'test', experiments[scene]['test'], csv_file)
        
        # Train metrics
        if experiments[scene]['train']:
            print(f"  Found {len(experiments[scene]['train'])} train experiments")
            
            train_md_table = create_markdown_table(scene, 'train', experiments[scene]['train'])
            
            if args.format in ['markdown', 'both']:
                all_markdown.append(train_md_table)
                scene_markdown.append(train_md_table)
            
            if args.format in ['csv', 'both']:
                csv_file = os.path.join(args.save_dir, f"{scene}_train_metrics.csv")
                create_csv_table(scene, 'train', experiments[scene]['train'], csv_file)
        
        # Save per-scene markdown file
        if args.format in ['markdown', 'both'] and (experiments[scene]['test'] or experiments[scene]['train']):
            scene_md_file = os.path.join(args.save_dir, f"{scene}_metrics.md")
            with open(scene_md_file, 'w') as f:
                f.write("\n".join(scene_markdown))
            print(f"  Saved scene Markdown: {scene_md_file}")
        
        # Save per-scene HTML file with color-coded tables
        if args.format in ['markdown', 'both'] and (experiments[scene]['test'] or experiments[scene]['train']):
            scene_html_file = os.path.join(args.save_dir, f"{scene}_metrics.html")
            html_content = create_html_document(
                scene,
                experiments[scene]['test'] if experiments[scene]['test'] else [],
                experiments[scene]['train'] if experiments[scene]['train'] else []
            )
            with open(scene_html_file, 'w') as f:
                f.write(html_content)
            print(f"  Saved scene HTML: {scene_html_file}")
    
    # Save combined markdown file
    if args.format in ['markdown', 'both']:
        md_file = os.path.join(args.save_dir, "all_metrics.md")
        with open(md_file, 'w') as f:
            f.write("\n".join(all_markdown))
        print(f"\n  Saved combined Markdown: {md_file}")
    
    print("\n" + "="*70)
    print("  COMPLETE!")
    print("="*70)
    print(f"\nTables saved to: {args.save_dir}/")
    if args.format in ['markdown', 'both']:
        print(f"\n  Per-scene markdown: {args.save_dir}/<scene>_metrics.md")
        print(f"  Per-scene HTML (color-coded): {args.save_dir}/<scene>_metrics.html")
        print(f"  Combined markdown: {args.save_dir}/all_metrics.md")
    if args.format in ['csv', 'both']:
        print(f"\n  CSV files: {args.save_dir}/<scene>_test_metrics.csv, <scene>_train_metrics.csv")
    print()


if __name__ == "__main__":
    main()

