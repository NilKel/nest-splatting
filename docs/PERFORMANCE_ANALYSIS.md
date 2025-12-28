# Performance Analysis: Rasterizer and Hash Grid Benchmarks

## Overview

This document summarizes the performance analysis conducted on the NeST-Splatting renderer, specifically examining the bottlenecks in the hash grid query rasterizer and potential optimization strategies.

## Test Configuration

- **Model**: `mic` scene from nerf_synthetic dataset (adaptive_cat mode)
- **Resolution**: 800x800
- **Hardware**: CUDA GPU
- **Benchmark methodology**: CUDA events for accurate timing, warmup iterations excluded

## Key Findings

### 1. Intersection Cap Analysis

Testing the impact of limiting per-pixel intersections (max_intersections parameter):

| Max Intersections | FPS    | Frame Time |
|-------------------|--------|------------|
| Unlimited         | 147    | 6.80 ms    |
| 100               | ~150   | ~6.67 ms   |
| 50                | ~155   | ~6.45 ms   |
| 20                | ~160   | ~6.25 ms   |
| 1                 | ~165   | ~6.06 ms   |

**Finding**: Capping intersections to 1 only provides ~11% speedup. Intersection processing is NOT the main bottleneck.

### 2. Rasterizer vs MLP Timing Separation

| Component        | FPS  | Time   | % of Total |
|------------------|------|--------|------------|
| Full Render      | 147  | 6.80ms | 100%       |
| Rasterizer Only  | 201  | 4.97ms | 73%        |
| MLP Only         | 584  | 1.71ms | 27%        |

**Finding**: The rasterizer dominates render time at 73%, while MLP inference is relatively fast.

### 3. Hash Grid Query Overhead (Critical Finding)

Comparing `hash_in_cuda=True` (in-rasterizer hash query) vs `hash_in_cuda=False` (plain 2DGS):

| Configuration                  | FPS  | Time    |
|--------------------------------|------|---------|
| Hash Rasterizer (unlimited)    | 189  | 5.27 ms |
| Hash Rasterizer (max_ints=1)   | 220  | 4.55 ms |
| Plain 2DGS (unlimited)         | 430  | 2.33 ms |
| Plain 2DGS (max_ints=1)        | 452  | 2.21 ms |

**Hash Grid Overhead**: 2.34 - 2.95 ms even with only 1 intersection!

This overhead is present regardless of the number of intersections, suggesting the cost is per-pixel rather than per-intersection.

### 4. Hash Query Method Comparison

| Method                           | Time     | Relative Speed |
|----------------------------------|----------|----------------|
| In-CUDA rasterizer hash query    | 2.34 ms  | 1x (baseline)  |
| tiny-cuda-nn hash query (Python) | 0.16 ms  | ~14x faster    |

**Finding**: The tiny-cuda-nn library is significantly more optimized for hash grid queries. The in-CUDA implementation has substantial overhead.

## Analysis

### Why is the in-CUDA hash query slow?

1. **Thread divergence**: Each pixel may query different hash grid cells, causing warp divergence
2. **Memory access patterns**: Non-coalesced memory access when querying the hash table
3. **Per-pixel overhead**: The hash query code path executes for every pixel, not just visible intersections
4. **Interpolation overhead**: Trilinear interpolation requires 8 lookups per query

### The 2-Pass Alternative

A potential optimization is to split rendering into two passes:
1. **Pass 1**: Rasterize to get 3D intersection points (fast, like plain 2DGS)
2. **Pass 2**: Query hash grid in batch using tiny-cuda-nn (optimized)
3. **Pass 3**: Alpha-blend features with Gaussian features

This approach could potentially reduce hash query time from 2.34ms to 0.16ms.

## Optimization: adaptive_cat_fast (Mode 13)

For the adaptive_cat architecture, during inference:
- Gaussians with weight > 0.5 use only per-Gaussian features (no hash query needed)
- Gaussians with weight ≤ 0.5 need hash grid queries

**Mode 13 Implementation**:
Checks the blend weight BEFORE computing the 3D intersection, allowing Gaussian-only primitives to skip the expensive intersection computation entirely during inference.

```
if (weight > 0.5):
    # FAST PATH: Skip 3D intersection, use Gaussian features directly
    feat = gauss_feat
else:
    # SLOW PATH: Compute intersection, query hash grid
    xyz = compute_3d_intersection()
    hash_feat = query_hashgrid(xyz)
    feat = blend(gauss_feat, hash_feat)
```

## Recommendations

1. **For maximum inference speed**: Use adaptive_cat_fast (mode 13) which skips hash queries for Gaussian-dominated primitives

2. **For research/prototyping**: Consider the 2-pass method to leverage tiny-cuda-nn's optimized hash queries

3. **For training**: The current implementation is acceptable as training is not real-time sensitive

4. **Future work**:
   - Profile hash query implementation for optimization opportunities
   - Consider caching hash queries for static viewpoints
   - Investigate lower-resolution hash feature maps with upsampling

## Conclusion

The hash grid query overhead (~2.3-2.9ms) is the dominant factor in render time, accounting for nearly half of total frame time. The plain 2DGS rasterizer without hash queries achieves 430 FPS (2.33ms), demonstrating the inherent speed of the Gaussian splatting approach. Optimization efforts should focus on reducing hash query overhead, either through algorithmic improvements or by leveraging the adaptive weight to skip unnecessary queries.
