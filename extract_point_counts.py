#!/usr/bin/env python3
"""
Script to extract point counts from all experiment PLY files.
"""

import os
import sys
from pathlib import Path

def count_ply_points(ply_path):
    """Count the number of points in a PLY file by reading the header."""
    try:
        with open(ply_path, 'rb') as f:
            # Read header to find vertex count
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('element vertex'):
                    return int(line.split()[-1])
                if line == 'end_header':
                    break
        return None
    except Exception as e:
        print(f"Error reading {ply_path}: {e}")
        return None


def find_experiments(base_dir):
    """Find all experiments with point_cloud.ply files."""
    experiments = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'point_cloud.ply' in files:
            ply_path = os.path.join(root, 'point_cloud.ply')
            # Extract experiment info from path
            # e.g., .../chair/cat/exp1_cat5_5_levels/point_cloud/iteration_30000/point_cloud.ply
            parts = root.split(os.sep)
            
            # Find the experiment name (parent of point_cloud folder)
            try:
                pc_idx = parts.index('point_cloud')
                exp_name = parts[pc_idx - 1]
                method = parts[pc_idx - 2] if pc_idx >= 2 else 'unknown'
                scene = parts[pc_idx - 3] if pc_idx >= 3 else 'unknown'
                iteration = parts[pc_idx + 1].replace('iteration_', '') if pc_idx + 1 < len(parts) else 'unknown'
            except (ValueError, IndexError):
                exp_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                method = 'unknown'
                scene = 'unknown'
                iteration = 'unknown'
            
            experiments.append({
                'scene': scene,
                'method': method,
                'experiment': exp_name,
                'iteration': iteration,
                'ply_path': ply_path
            })
    
    return experiments


def main():
    # Default to chair experiments
    base_dir = "/home/nilkel/Projects/nest-splatting/outputs/nerf_synthetic/chair"
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    print(f"Scanning: {base_dir}")
    print()
    
    experiments = find_experiments(base_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Sort by method, then experiment name
    experiments.sort(key=lambda x: (x['method'], x['experiment']))
    
    # Print results
    print(f"{'Method':<20} {'Experiment':<35} {'Iteration':<12} {'Points':>12}")
    print("=" * 85)
    
    results = []
    for exp in experiments:
        num_points = count_ply_points(exp['ply_path'])
        if num_points is not None:
            print(f"{exp['method']:<20} {exp['experiment']:<35} {exp['iteration']:<12} {num_points:>12,}")
            results.append({**exp, 'num_points': num_points})
    
    # Save to file
    output_file = os.path.join(base_dir, 'point_counts.txt')
    with open(output_file, 'w') as f:
        f.write(f"Point Counts for {base_dir}\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Method':<20} {'Experiment':<35} {'Iteration':<12} {'Points':>12}\n")
        f.write("-" * 85 + "\n")
        
        for exp in results:
            f.write(f"{exp['method']:<20} {exp['experiment']:<35} {exp['iteration']:<12} {exp['num_points']:>12,}\n")
        
        f.write("\n")
        f.write(f"Total experiments: {len(results)}\n")
        
        # Summary by method
        f.write("\nSummary by method:\n")
        f.write("-" * 40 + "\n")
        method_counts = {}
        for exp in results:
            method = exp['method']
            if method not in method_counts:
                method_counts[method] = []
            method_counts[method].append(exp['num_points'])
        
        for method, counts in sorted(method_counts.items()):
            avg = sum(counts) / len(counts)
            f.write(f"{method:<20} Avg: {avg:>12,.0f} ({len(counts)} experiments)\n")
    
    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

