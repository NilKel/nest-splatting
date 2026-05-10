# nest-splatting bench bundle (4090)

Self-contained FPS / PSNR / SSIM / LPIPS bench for baked nest-splatting scenes.
No nest-splatting source needed — only the BC7 atlas, a snapshot of the
pre-activated Gaussian state, and the test cameras.

## Layout

    nest-bench/
      bench_4090/
        setup_4090.sh         install miniconda + env + build CUDA submodule
        run_all.sh            iterate bundles/ and emit a markdown table
        bench_minimal.py      single-bundle bench script
        README.md             this file
      diff_surfel_bake_render/    CUDA rasterizer source (built locally)
      bundles/
        truck/
          gaussian_state.pt   pre-activated tensors snapshot
          atlas_texture.bc7   BC7-compressed RGB residual atlas
          atlas_rects.pt      [N, 4] per-Gaussian atlas rects
          bake_meta.json      activation/dequant/feature config
          cameras.pt          test cameras only
          images/<name>.png   ground-truth test images at eval resolution
          sb_params.pt        (only when --feature beta was used)
        train/
        ...

## One-time setup

```bash
cd ~/nest-bench/bench_4090
./setup_4090.sh                # ~10 min first time, idempotent on re-run
```

What it installs:
- Miniforge → `~/miniforge3` (skipped if present; from GitHub releases since
  campus Zscaler blocks repo.anaconda.com)
- Conda env `bench`: python 3.10, torch 2.4.1+cu121, plyfile, Pillow,
  pytorch_msssim, lpips
- Builds `diff_surfel_bake_render` against `TORCH_CUDA_ARCH_LIST=8.9` (Ada
  / RTX 4090). Uses torch's bundled CUDA toolchain — no system nvcc needed.

## Running

```bash
cd ~/nest-bench/bench_4090
./run_all.sh                   # all scenes, 10 warmup + 200 benchmark frames

# Or override frame counts:
NUM_WARMUP=20 NUM_BENCHMARK=500 ./run_all.sh
```

Output:
- `~/nest-bench/bench_results/<scene>.json` per scene (raw metrics)
- A markdown table printed to stdout

## Single scene

```bash
conda activate bench
python bench_minimal.py --bundle ../bundles/truck \
    --num_warmup 10 --num_benchmark 200
```

## Notes

- The bench uses CUDA events (`torch.cuda.Event(enable_timing=True)`) so
  measurements exclude Python wall-clock noise.
- For stable FPS, make sure no other GPU jobs are running. Check with
  `nvidia-smi`.
- LPIPS uses the standalone `lpips` PyPI package. Numbers may differ by
  ~0.02 from the `lpipsPyTorch` impl in nest-splatting (same VGG backbone,
  slightly different normalization).
