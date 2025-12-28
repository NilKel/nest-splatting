# Adaptive Modes Reference

This document describes the adaptive rendering modes and their associated CLI flags for NeST-Splatting. These modes allow per-Gaussian decisions about whether to use hash-grid lookups or rely solely on Gaussian features.

## Overview

The adaptive modes implement a learnable gating mechanism where each Gaussian has a weight parameter that controls the blend between:
- **Gaussian-only rendering**: Uses only per-Gaussian SH features (fast, no hash lookup)
- **Hash-enhanced rendering**: Combines Gaussian features with hash-grid MLP output (higher quality)

The goal is to learn which Gaussians benefit from hash enhancement and which can render well with just their own features, enabling faster inference with minimal quality loss.

---

## Methods

### `--method adaptive_zero`

The primary adaptive mode. Each Gaussian has a learnable weight `w` that controls:
- `w → 0`: Use zeros for hash features (Gaussian-only)
- `w → 1`: Use full hash-grid features (Hash-enhanced)

During training, the blend is smooth: `output = w * hash_features + (1-w) * zeros`

During inference, a hard threshold is applied: skip hash lookup entirely if `w < 0.1`

### `--method adaptive_cat`

Additive blending mode where weight controls Gaussian vs hash contribution:
- `w → 1`: Use Gaussian-only features (skip hash query)
- `w → 0`: Use hash-grid features for fine levels

During training: `coarse = gaussian * w`, `fine = gaussian * w + hash * (1-w)`

During inference: hard threshold at `w >= 0.9` skips hash lookup (conservative)

### `--method adaptive_gate`

**Gumbel-STE with Forced Training** for binary hash selection:
- Each Gaussian has a learnable gate logit (initialized sparse with `--gate_init`)
- During training: Gumbel noise + STE for stochastic binary gating
- Forced training: 20% of Gaussians forced to use hash (ensures hash features learn)
- During inference: Hard threshold at probability > 0.5

Key features:
- **Always binary masking**: Mask is strictly 0 or 1 (no soft blending)
- **Sparse initialization**: Gate logits start negative (most gates closed)
- **Sparsity loss**: Penalizes gate probability to encourage closed gates

---

## Core Flags

### Method Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--method` | str | `baseline` | Rendering method. Use `adaptive_zero`, `adaptive_cat`, or `adaptive_gate` |

### Adaptive Cat Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lambda_adaptive_cat` | float | `0.01` | Entropy regularization weight (pushes weights toward 0 or 1) |
| `--adaptive_cat_anneal_start` | int | `15000` | Iteration to start annealing regularization |
| `--adaptive_cat_threshold` | float | `0.9` | Inference threshold: w>=threshold uses Gaussian-only (conservative) |
| `--adaptive_cat_inference` | flag | `False` | Force hard gating at inference (auto-enabled for final eval) |

### Adaptive Zero Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lambda_adaptive_zero` | float | `0.0` | BCE regularization weight (pushes weights away from threshold) |
| `--bce_threshold` | float | `0.5` | Repulsion point for BCE loss (0.5 = symmetric, 0.1 = match inference threshold) |
| `--hash_lambda` | float | `0.0` | L1 penalty pushing weights toward 1 (favor hash over Gaussian-only) |
| `--adaptive_zero_anneal_start` | int | `15000` | Iteration to start annealing regularization (ramps from 0 to full) |

### Adaptive Gate Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lambda_sparsity` | float | `0.005` | Sparsity penalty on gate probability (encourages gates to stay closed) |
| `--force_ratio` | float | `0.2` | Fraction of Gaussians forced to use hash during training (0.2 = 20%) |
| `--gate_init` | float | `-2.0` | Initial gate logit (negative = sparse start, sigmoid(-2) ≈ 0.12) |
| `--adaptive_gate_inference` | flag | `False` | Force hard gating at inference (probability > 0.5 uses hash) |

---

## Temperature Annealing

Temperature annealing affects the Gumbel-softmax temperature in `adaptive_gate` and the sigmoid sharpness in `adaptive_zero`. Lower temperature = sharper decisions.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--temp_start` | float | `1.0` | Initial temperature (1.0 = normal) |
| `--temp_end` | float | `1.0` | Final temperature (e.g., 0.5 for sharper gating) |
| `--temp_anneal_start` | int | `3000` | Iteration to start temperature ramp |
| `--temp_anneal_end` | int | `25000` | Iteration to reach final temperature |

**For adaptive_gate**: Lower temperature makes Gumbel-STE more deterministic.

**For adaptive_zero**: Higher temperature makes sigmoid steeper: `sigmoid(logit * temperature)`.

---

## Parabola Regularization

An additional regularization term `w*(1-w)` that penalizes weights near 0.5 (maximum penalty) and has zero penalty at 0 or 1. This is additive with other regularization.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lambda_parabola` | float | `0.0` | Weight for parabola penalty (0 = disabled) |

---

## Densification Behavior

Controls how adaptive weights are handled when Gaussians are created during densification or MCMC relocation.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--relocation` | str | `clone` | `clone`: copy weight from source Gaussian; `reset`: initialize to 0 (sigmoid=0.5) |

---

## Inference Settings

### Adaptive Zero Inference Threshold

During inference, the CUDA kernel uses a threshold to skip hash lookups:
- **Threshold**: `0.1` (conservative - only skip hash if weight is very low)
- Gaussians with `sigmoid(weight) >= 0.1` perform hash lookups
- Gaussians with `sigmoid(weight) < 0.1` use zeros (Gaussian-only)

### Adaptive Gate Inference Threshold

During inference, Gaussians use hash if `sigmoid(gate_logit) > 0.5`:
- **Threshold**: `0.5` (probability > 50% → use hash)
- Gaussians with probability ≤ 0.5 use zeros (Gaussian-only)

### Metrics Recorded

The final evaluation records metrics for both modes:
- **Inference mode**: Hard gating with threshold
- **Training mode**: Soft blending (for comparison, adaptive_zero only)

Metrics include: PSNR, SSIM, LPIPS, L1

---

## Weight Distribution Logging

At the end of training, the weight distribution is logged to `training_log.txt`:
- **Below 0.1**: Gaussians using Gaussian-only (skip hash)
- **Above 0.9**: Gaussians using full hash enhancement
- **Middle (0.1-0.9)**: Gaussians with mixed contribution

For adaptive_gate, the logging shows:
- **Gate%**: Percentage of gates open (probability > 0.5)
- **AvgP**: Average gate probability

---

## Example Commands

### Basic Adaptive Zero
```bash
python train.py -s /path/to/scene -m run_name \
    --method adaptive_zero \
    --lambda_adaptive_zero 0.01 \
    --adaptive_zero_anneal_start 15000
```

### Adaptive Zero with Temperature Annealing
```bash
python train.py -s /path/to/scene -m run_name \
    --method adaptive_zero \
    --lambda_adaptive_zero 0.01 \
    --temp_end 10.0 \
    --temp_anneal_start 3000 \
    --temp_anneal_end 25000
```

### Adaptive Gate (Gumbel-STE)
```bash
python train.py -s /path/to/scene -m run_name \
    --method adaptive_gate \
    --lambda_sparsity 0.005 \
    --force_ratio 0.2 \
    --gate_init -2.0
```

### Adaptive Gate with Custom Settings
```bash
python train.py -s /path/to/scene -m run_name \
    --method adaptive_gate \
    --lambda_sparsity 0.01 \
    --force_ratio 0.3 \
    --gate_init -3.0 \
    --temp_end 0.5 \
    --temp_anneal_start 5000 \
    --temp_anneal_end 20000 \
    --relocation reset \
    --mcmc --cap_max 500000
```

---

## Training Visualization

### Adaptive Zero Timeline
```
Iteration:  0 -------- 15000 -------- 30000
            |          |              |
            v          v              v
Regularization: 0 → ramps up → full strength
Temperature:    1.0 → anneals → temp_end
```

### Adaptive Gate (Gumbel-STE)
```
Iteration:  0 ----------------------------- 30000
            |                               |
            v                               v
Gumbel-STE: Always binary (stochastic during training)
Force Mask: 20% forced hash (ensures hash learns)
Sparsity:   Penalty on gate probability
```

---

## Output Files

| File | Description |
|------|-------------|
| `test_metrics.txt` | PSNR, SSIM, LPIPS, L1 for inference and training modes |
| `train_metrics.txt` | Same metrics for training set |
| `training_log.txt` | Gaussian count, weight distribution, FPS |
| `test_renders/` | Rendered test images (inference mode) |
| `test_training_mode/` | Rendered test images (training/soft mode) |

---

## Tips

1. **Start with defaults**: The default regularization values work well for most scenes
2. **Sparse initialization**: For adaptive_gate, `--gate_init -2.0` starts with ~12% open gates
3. **Force ratio**: 20% forced hash ensures the hash network learns valid features
4. **Monitor gate distribution**: Check progress bar for `Gate%` and `AvgP` during training
5. **MCMC mode**: Combine with `--mcmc --cap_max N` for adaptive Gaussian count management

---

## How Adaptive Gate Works

### Gumbel-STE Mechanism

During training, each Gaussian's gate decision is made stochastically:

1. **Gumbel noise**: Add Gumbel(0,1) noise to gate logits for exploration
2. **Temperature scaling**: Divide by temperature for sharper/softer decisions
3. **STE (Straight-Through Estimator)**:
   - Forward: Hard threshold (0 or 1)
   - Backward: Gradient flows through soft sigmoid
4. **Force mask**: Randomly force 20% to use hash (detached, not learnable)
5. **Combine**: `effective_mask = max(hard_gate, force_mask)` ensures binary output

### Why Forced Training?

The forced training mechanism ensures:
- Hash features learn valid outputs even when gates are closed
- No "dead" hash features that never receive gradients
- Smooth transition when gates open during training

### Sparsity Loss

The sparsity loss `lambda_sparsity * mean(sigmoid(gate_logits))` encourages:
- Gates to stay closed by default
- Only open gates where hash genuinely improves quality
- Faster inference with fewer hash lookups
