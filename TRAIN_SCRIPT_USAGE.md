# train_baseline_cat.sh - Usage Guide

## Overview
Automated training script for Nest-Splatting that runs baseline and cat modes (hybrid_levels 0-6) across multiple scenes.

## Syntax
```bash
./train_baseline_cat.sh <dataset> <base_name> [scene_names] [iterations] [extra_args]
```

## Arguments

1. **dataset** (Required)
   - `nerf_synthetic` - Blender synthetic scenes (8 scenes)
   - `DTU` - DTU real-world scans (15 scenes)
   - `mip_360` - Mip-NeRF 360 dataset (7 scenes)

2. **base_name** (Required)
   - Base name for experiments (e.g., `exp1`, `test`, `final`)

3. **scene_names** (Optional, default: `all`)
   - Comma-separated scene names (e.g., `chair,drums`)
   - `all` - Run all scenes in the dataset

4. **iterations** (Optional, default: `30000`)
   - Number of training iterations

5. **extra_args** (Optional)
   - Additional arguments passed to `train.py` (e.g., `"--disable_c2f"`, `"--mcmc --cap_max 300000"`)

## Dataset Configurations

### nerf_synthetic
- **Data dir**: `/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic`
- **Config**: `./configs/nerfsyn.yaml`
- **Scenes**: chair, drums, ficus, hotdog, lego, materials, mic, ship
- **Resolution**: 1 (default)
- **Output**: `outputs/nerf_synthetic/{scene}/{method}/{name}`

### DTU
- **Data dir**: `/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU`
- **Config**: `./configs/dtu.yaml`
- **Scenes**: scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122
- **Resolution**: 2 (automatically set)
- **Output**: `outputs/DTU/{scene}/{method}/{name}`

### mip_360
- **Data dir**: `/home/nilkel/Projects/data/mip360`
- **Config**: `./configs/360_outdoor.yaml`
- **Scenes**: bicycle, bonsai, counter, garden, kitchen, room, stump
- **Resolution**: 2 (automatically set)
- **Output**: `outputs/mip_360/{scene}/{method}/{name}`

## Methods Trained

For each scene, the script trains:
1. **baseline** - Standard 2DGS
2. **cat0** - Cat mode with hybrid_levels=0
3. **cat1** - Cat mode with hybrid_levels=1
4. **cat2** - Cat mode with hybrid_levels=2
5. **cat3** - Cat mode with hybrid_levels=3
6. **cat4** - Cat mode with hybrid_levels=4
7. **cat5** - Cat mode with hybrid_levels=5
8. **cat6** - Cat mode with hybrid_levels=6

**Total**: 8 methods per scene

## Examples

### Basic Usage

Run all nerf_synthetic scenes:
```bash
./train_baseline_cat.sh nerf_synthetic exp1
```

Run all DTU scenes:
```bash
./train_baseline_cat.sh DTU exp1
```

Run all mip_360 scenes:
```bash
./train_baseline_cat.sh mip_360 exp1
```

### Specific Scenes

Train only chair and drums from nerf_synthetic:
```bash
./train_baseline_cat.sh nerf_synthetic exp1 chair,drums
```

Train only scan24 and scan37 from DTU:
```bash
./train_baseline_cat.sh DTU exp1 scan24,scan37
```

### Custom Iterations

Run with 40k iterations:
```bash
./train_baseline_cat.sh nerf_synthetic exp1 all 40000
```

### With Extra Arguments

Disable coarse-to-fine:
```bash
./train_baseline_cat.sh nerf_synthetic exp1 all 30000 "--disable_c2f"
```

Enable MCMC mode:
```bash
./train_baseline_cat.sh DTU exp1 scan24 30000 "--mcmc --cap_max 300000"
```

Use depth ratio for DTU:
```bash
./train_baseline_cat.sh DTU exp1 all 30000 "--depth_ratio 1"
```

Multiple extra arguments:
```bash
./train_baseline_cat.sh nerf_synthetic exp1 all 30000 "--disable_c2f --random_background"
```

## Features

### Automatic Skip
- Script checks for existing `test_metrics.txt`
- Skips already completed experiments
- Safe to re-run after interruption

### Progress Tracking
- Shows `[CURRENT/TOTAL]` for each experiment
- Displays completion status
- Logs all output to timestamped log file

### Summary Statistics
- Total runs
- Completed count
- Skipped count
- Failed count

### Automatic Results Table
- Generates results table after completion
- Saved to `metrics_tables/`

## Output Structure

```
outputs/{dataset}/{scene}/{method}/{experiment_name}/
├── test_metrics.txt          # Final test metrics
├── train_metrics.txt          # Final train metrics
├── training_log.txt           # Training progress log
├── cameras/                   # Camera visualizations
├── point_cloud/               # Saved point clouds
└── renders/                   # Rendered images
```

For cat mode, output includes hybrid_levels suffix:
```
outputs/nerf_synthetic/chair/cat/exp1_cat5_5_levels/
```

## Log Files

Global log file: `logs/train_{base_name}_{timestamp}.log`

Contains full output from all training runs.

## Workflow Example

```bash
# 1. Train all nerf_synthetic scenes
./train_baseline_cat.sh nerf_synthetic exp1

# 2. Train specific DTU scenes with depth
./train_baseline_cat.sh DTU exp1 scan24,scan37,scan40 30000 "--depth_ratio 1"

# 3. Train mip_360 with higher iterations
./train_baseline_cat.sh mip_360 exp1 all 40000

# 4. Check results
ls metrics_tables/

# 5. If interrupted, re-run (will skip completed)
./train_baseline_cat.sh nerf_synthetic exp1  # Resumes from last incomplete
```

## Tips

1. **Start small**: Test with a single scene first
   ```bash
   ./train_baseline_cat.sh nerf_synthetic test chair
   ```

2. **Monitor progress**: Watch the log file in real-time
   ```bash
   tail -f logs/train_exp1_*.log
   ```

3. **Batch processing**: Run different datasets in parallel terminals
   ```bash
   # Terminal 1
   ./train_baseline_cat.sh nerf_synthetic exp1
   
   # Terminal 2
   ./train_baseline_cat.sh DTU exp1
   ```

4. **Resume after failure**: Script automatically skips completed experiments
   ```bash
   # After a failure or interruption, just re-run
   ./train_baseline_cat.sh nerf_synthetic exp1
   ```

## Troubleshooting

### Data directory not found
Check that the dataset paths are correct:
- nerf_synthetic: `/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic`
- DTU: `/home/nilkel/Projects/nest-splatting/data/dtu/2DGS_data/DTU`
- mip_360: `/home/nilkel/Projects/data/mip360`

### YAML config not found
Ensure config files exist:
- `./configs/nerfsyn.yaml`
- `./configs/dtu.yaml`
- `./configs/360_outdoor.yaml`

### Experiment fails
Check the log file for details:
```bash
grep -A 10 "FAILED" logs/train_exp1_*.log
```
