#!/usr/bin/env python3
"""
Nest-Splatting Results Table Generator for Baseline vs Cat Experiments
=======================================================================

Generates comparison tables for baseline and cat (hybrid_levels 0-6) experiments.

Usage:
    python create_results_table.py --base_name exp1
    python create_results_table.py --base_name exp1 --scenes drums,mic,lego
    python create_results_table.py --base_name exp1 --output_dir outputs --save_dir metrics_tables
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import csv


def parse_metrics_file(filepath):
    """Parse a metrics file and extract PSNR, SSIM, L1."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
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


def scan_experiments(output_dir, base_name, scenes=None):
    """
    Scan for experiments matching base_name pattern.
    
    Path structure: outputs/nerf_synthetic/{scene}/{method}/{name}
    For cat mode, name is: {base_name}_cat{hl}_{hl}_levels
    
    Returns:
        dict: {scene: {'test': [(method_label, metrics)], 'train': [...]}}
    """
    experiments = defaultdict(lambda: {'test': [], 'train': []})
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return experiments
    
    # Methods to look for: (method_dir, exp_name, label)
    methods_to_scan = [
        ('baseline', f'{base_name}_baseline', 'baseline'),
    ]
    # Add cat0-6 (train.py appends _{hl}_levels to name)
    for hl in range(7):
        methods_to_scan.append(('cat', f'{base_name}_cat{hl}_{hl}_levels', f'cat{hl}'))
    
    # Path: outputs/nerf_synthetic/{scene}/{method}/{name}
    nerf_syn_dir = output_path / 'nerf_synthetic'
    if not nerf_syn_dir.exists():
        print(f"Warning: nerf_synthetic directory not found in {output_dir}")
        return experiments
    
    # Scan each scene
    for scene_dir in nerf_syn_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene = scene_dir.name
        
        # Filter by scenes if specified
        if scenes and scene not in scenes:
            continue
        
        # Check each method
        for method, exp_name, label in methods_to_scan:
            method_dir = scene_dir / method
            if not method_dir.exists():
                continue
            
            exp_dir = method_dir / exp_name
            if not exp_dir.exists():
                continue
            
            # Parse metrics
            test_metrics = parse_metrics_file(exp_dir / 'test_metrics.txt')
            train_metrics = parse_metrics_file(exp_dir / 'train_metrics.txt')
            
            if test_metrics:
                experiments[scene]['test'].append((label, test_metrics))
            if train_metrics:
                experiments[scene]['train'].append((label, train_metrics))
    
    return experiments


def get_color_for_value(value, min_val, max_val, ascending=True):
    """Get RGB color for value (red=worst, green=best)."""
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    if not ascending:
        normalized = 1.0 - normalized
    
    r = int(230 + (57 - 230) * normalized)
    g = int(57 + (230 - 57) * normalized)
    b = 57
    
    return f"rgb({r}, {g}, {b})"


def create_html_table(scene, split, data):
    """Create HTML table for a scene/split."""
    if not data:
        return f"<p>No {split} data for {scene}</p>\n"
    
    # Sort by method label (baseline first, then cat0-6)
    def sort_key(x):
        label = x[0]
        if label == 'baseline':
            return (0, 0)
        elif label.startswith('cat'):
            return (1, int(label[3:]))
        return (2, label)
    
    data = sorted(data, key=sort_key)
    
    # Calculate min/max for coloring
    psnr_values = [m[1]['psnr'] for m in data]
    ssim_values = [m[1]['ssim'] for m in data]
    l1_values = [m[1]['l1'] for m in data]
    
    psnr_min, psnr_max = min(psnr_values), max(psnr_values)
    ssim_min, ssim_max = min(ssim_values), max(ssim_values)
    l1_min, l1_max = min(l1_values), max(l1_values)
    
    lines = []
    lines.append(f'<h2>{scene.upper()} - {split.upper()}</h2>')
    lines.append('<table class="metrics-table">')
    lines.append('<thead><tr>')
    lines.append('<th>Method</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th><th>L1 ↓</th>')
    lines.append('</tr></thead>')
    lines.append('<tbody>')
    
    for label, metrics in data:
        psnr_color = get_color_for_value(metrics['psnr'], psnr_min, psnr_max, True)
        ssim_color = get_color_for_value(metrics['ssim'], ssim_min, ssim_max, True)
        l1_color = get_color_for_value(metrics['l1'], l1_min, l1_max, False)
        
        lines.append('<tr>')
        lines.append(f'<td class="method">{label}</td>')
        lines.append(f'<td class="metric" style="background-color: {psnr_color};">{metrics["psnr"]:.2f}</td>')
        lines.append(f'<td class="metric" style="background-color: {ssim_color};">{metrics["ssim"]:.4f}</td>')
        lines.append(f'<td class="metric" style="background-color: {l1_color};">{metrics["l1"]:.6f}</td>')
        lines.append('</tr>')
    
    lines.append('</tbody></table>')
    return '\n'.join(lines)


def create_summary_table(experiments, split):
    """Create a summary table with all scenes as columns and methods as rows."""
    if not experiments:
        return "<p>No data available</p>"
    
    scenes = sorted(experiments.keys())
    methods = ['baseline'] + [f'cat{i}' for i in range(7)]
    
    # Collect all data
    data = {}
    for scene in scenes:
        data[scene] = {}
        for label, metrics in experiments[scene].get(split, []):
            data[scene][label] = metrics
    
    lines = []
    lines.append(f'<h2>Summary - {split.upper()} PSNR (dB)</h2>')
    lines.append('<table class="metrics-table">')
    lines.append('<thead><tr>')
    lines.append('<th>Method</th>')
    for scene in scenes:
        lines.append(f'<th>{scene}</th>')
    lines.append('<th>Average</th>')
    lines.append('</tr></thead>')
    lines.append('<tbody>')
    
    # Calculate column min/max for coloring
    col_stats = {}
    for scene in scenes:
        values = [data[scene].get(m, {}).get('psnr', 0) for m in methods if data[scene].get(m)]
        if values:
            col_stats[scene] = (min(values), max(values))
        else:
            col_stats[scene] = (0, 1)
    
    for method in methods:
        lines.append('<tr>')
        lines.append(f'<td class="method">{method}</td>')
        
        row_values = []
        for scene in scenes:
            if method in data[scene]:
                psnr = data[scene][method]['psnr']
                row_values.append(psnr)
                color = get_color_for_value(psnr, col_stats[scene][0], col_stats[scene][1], True)
                lines.append(f'<td class="metric" style="background-color: {color};">{psnr:.2f}</td>')
            else:
                lines.append('<td class="metric">-</td>')
        
        # Average
        if row_values:
            avg = sum(row_values) / len(row_values)
            lines.append(f'<td class="metric" style="font-weight: bold;">{avg:.2f}</td>')
        else:
            lines.append('<td class="metric">-</td>')
        
        lines.append('</tr>')
    
    lines.append('</tbody></table>')
    return '\n'.join(lines)


def get_html_styles():
    """CSS styles for HTML output."""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            max-width: 1400px;
            margin: 40px auto;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
        }
        h1 {
            color: #00d9ff;
            border-bottom: 3px solid #00d9ff;
            padding-bottom: 10px;
        }
        h2 {
            color: #ff6b6b;
            margin-top: 40px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #16213e;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 40px;
        }
        .metrics-table thead {
            background-color: #0f3460;
        }
        .metrics-table th {
            padding: 12px;
            text-align: center;
            font-weight: 600;
            color: #00d9ff;
        }
        .metrics-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #0f3460;
            text-align: center;
        }
        .metrics-table tbody tr:hover {
            background-color: #1a1a40;
        }
        .metrics-table .method {
            font-weight: 600;
            color: #ff6b6b;
            text-align: left;
        }
        .metrics-table .metric {
            font-weight: 600;
            font-family: 'Courier New', monospace;
            color: #000;
        }
        .info {
            background-color: #0f3460;
            padding: 15px;
            border-left: 4px solid #00d9ff;
            margin-bottom: 30px;
            border-radius: 4px;
        }
        .legend {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        .color-sample {
            width: 60px;
            height: 20px;
            border: 1px solid #333;
            border-radius: 3px;
        }
        .scene-section {
            margin-bottom: 60px;
            padding-bottom: 40px;
            border-bottom: 2px solid #0f3460;
        }
    """


def create_full_html(experiments, base_name):
    """Create complete HTML document."""
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="en">')
    html.append('<head>')
    html.append('<meta charset="UTF-8">')
    html.append(f'<title>{base_name} - Results</title>')
    html.append(f'<style>{get_html_styles()}</style>')
    html.append('</head>')
    html.append('<body>')
    html.append(f'<h1>{base_name} - Baseline vs Cat Results</h1>')
    html.append('<div class="info">')
    html.append('<strong>Methods:</strong> baseline, cat0 (0 Gaussian levels), cat1-6 (1-6 Gaussian levels)<br>')
    html.append('<strong>Color:</strong> Red = worst, Green = best (per column)')
    html.append('</div>')
    html.append('<div class="legend">')
    html.append('<span><strong>↑</strong> Higher is better</span>')
    html.append('<span><strong>↓</strong> Lower is better</span>')
    html.append('<div class="color-sample" style="background: linear-gradient(to right, rgb(230, 57, 57), rgb(57, 230, 57));"></div>')
    html.append('<span>Worst → Best</span>')
    html.append('</div>')
    
    # Summary tables
    html.append('<div class="scene-section">')
    html.append('<h1>Summary Tables</h1>')
    html.append(create_summary_table(experiments, 'test'))
    html.append(create_summary_table(experiments, 'train'))
    html.append('</div>')
    
    # Per-scene tables
    for scene in sorted(experiments.keys()):
        html.append('<div class="scene-section">')
        html.append(f'<h1>{scene.upper()}</h1>')
        html.append(create_html_table(scene, 'test', experiments[scene]['test']))
        html.append(create_html_table(scene, 'train', experiments[scene]['train']))
        html.append('</div>')
    
    html.append('</body></html>')
    return '\n'.join(html)


def create_csv(experiments, base_name, save_dir, split):
    """Create CSV file for a split."""
    scenes = sorted(experiments.keys())
    methods = ['baseline'] + [f'cat{i}' for i in range(7)]
    
    filepath = os.path.join(save_dir, f'{base_name}_{split}_results.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Method']
        for scene in scenes:
            header.extend([f'{scene}_PSNR', f'{scene}_SSIM', f'{scene}_L1'])
        header.extend(['Avg_PSNR', 'Avg_SSIM', 'Avg_L1'])
        writer.writerow(header)
        
        # Data rows
        for method in methods:
            row = [method]
            psnr_vals, ssim_vals, l1_vals = [], [], []
            
            for scene in scenes:
                scene_data = {label: m for label, m in experiments[scene].get(split, [])}
                if method in scene_data:
                    m = scene_data[method]
                    row.extend([f"{m['psnr']:.2f}", f"{m['ssim']:.4f}", f"{m['l1']:.6f}"])
                    psnr_vals.append(m['psnr'])
                    ssim_vals.append(m['ssim'])
                    l1_vals.append(m['l1'])
                else:
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
    
    print(f"  Saved: {filepath}")


def create_markdown(experiments, base_name, save_dir):
    """Create markdown summary."""
    scenes = sorted(experiments.keys())
    methods = ['baseline'] + [f'cat{i}' for i in range(7)]
    
    lines = []
    lines.append(f'# {base_name} - Baseline vs Cat Results\n')
    lines.append('## Test PSNR Summary\n')
    
    # Header
    header = '| Method |'
    sep = '|--------|'
    for scene in scenes:
        header += f' {scene} |'
        sep += '--------|'
    header += ' Avg |'
    sep += '-----|'
    lines.append(header)
    lines.append(sep)
    
    # Data
    for method in methods:
        row = f'| {method} |'
        psnr_vals = []
        for scene in scenes:
            scene_data = {label: m for label, m in experiments[scene].get('test', [])}
            if method in scene_data:
                psnr = scene_data[method]['psnr']
                row += f' {psnr:.2f} |'
                psnr_vals.append(psnr)
            else:
                row += ' - |'
        
        if psnr_vals:
            row += f' **{sum(psnr_vals)/len(psnr_vals):.2f}** |'
        else:
            row += ' - |'
        lines.append(row)
    
    lines.append('')
    
    filepath = os.path.join(save_dir, f'{base_name}_results.md')
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate results tables for baseline vs cat experiments")
    parser.add_argument('--base_name', type=str, required=True, help='Base experiment name')
    parser.add_argument('--scenes', type=str, default=None, help='Comma-separated scene names (default: all found)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory to scan')
    parser.add_argument('--save_dir', type=str, default='metrics_tables', help='Directory to save tables')
    
    args = parser.parse_args()
    
    scenes = args.scenes.replace(' ', ',').split(',') if args.scenes else None
    
    print("="*70)
    print("  Nest-Splatting Results Table Generator")
    print("="*70)
    print(f"Base name:  {args.base_name}")
    print(f"Scanning:   {args.output_dir}")
    print(f"Scenes:     {scenes if scenes else 'all'}")
    print("="*70)
    
    experiments = scan_experiments(args.output_dir, args.base_name, scenes)
    
    if not experiments:
        print("\nNo experiments found!")
        return
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nFound {len(experiments)} scene(s): {', '.join(sorted(experiments.keys()))}")
    print(f"\nGenerating tables...")
    
    # HTML
    html_path = os.path.join(args.save_dir, f'{args.base_name}_results.html')
    with open(html_path, 'w') as f:
        f.write(create_full_html(experiments, args.base_name))
    print(f"  Saved: {html_path}")
    
    # CSV
    create_csv(experiments, args.base_name, args.save_dir, 'test')
    create_csv(experiments, args.base_name, args.save_dir, 'train')
    
    # Markdown
    create_markdown(experiments, args.base_name, args.save_dir)
    
    print("\n" + "="*70)
    print("  COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.save_dir}/")
    print(f"  - {args.base_name}_results.html  (visual comparison)")
    print(f"  - {args.base_name}_results.md    (markdown table)")
    print(f"  - {args.base_name}_test_results.csv")
    print(f"  - {args.base_name}_train_results.csv")
    print()


if __name__ == "__main__":
    main()


