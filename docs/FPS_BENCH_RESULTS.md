# FPS Benchmark — RTX 5090 (sherlock checkpoints)

End-to-end render-time FPS for trained checkpoints pulled from sherlock.
All measurements use `torch.cuda.Event` timing (GPU-throughput, not paper-style
`time.time`). Test cameras only, 1/4-resolution images (`images_4`) where
relevant.

- Machine: nilkel-Workstation (10.176.128.124), RTX 5090, torch 2.11/cu128.
- Env (nest-splatting): `nest_splatting` conda env.
- Env (nexels): `nexels` conda env (clone of `nest_splatting` + fpsample +
  fused-ssim + diff-nexel-rasterization).
- Bench script (nest): [`benchmark_render.py`](../benchmark_render.py)
  (`--warmup 20 --bench_iters 200 --skip_save`).
- Bench script (nexels): [`render.py`](../../nexels/render.py) (its own
  cuda.Event timing, writes `info.json`).
- Checkpoint source: `sherlock01:/home/z0051beu/userdir/Projects/{nest-splatting/outputs/mip_360/,nexels_runs/tnt_cap400000/}`
- Local cache: [`data/sherlock_ckpts/`](../data/sherlock_ckpts/).
- Run date: 2026-05-12.

## Mip-NeRF 360 — baseline vs cat (5-level hybrid)

`vanilla_baseline` (baseline) vs `def10HKb01o01s0n1e3BS_10kbce_cat5_5_levels` (cat).
Both trained on sherlock, 35k iters, `images_4` (1/4 res).

| Scene    | N test cams |  baseline FPS  |  cat FPS  |  cat / baseline |
|----------|:---:|:----:|:-----:|:----:|
| **bicycle**  | 25 | 14.62 | 41.03 | 2.81× |
| **garden**   | 24 | 16.99 | 55.91 | 3.29× |
| **stump**    | 16 | 20.95 | 40.60 | 1.94× |
| **flowers**  | 22 | 15.78 | 39.37 | 2.49× |
| **treehill** | 18 | 15.95 | 37.97 | 2.38× |
| **counter**  | 30 | 31.07 | 31.75 | 1.02× |
| **room**     | 39 | 33.54 | 39.68 | 1.18× |
| **kitchen**  | 35 | 32.72 | 39.83 | 1.22× |
| **bonsai**   | 37 | 31.41 | 34.39 | 1.10× |
| **mean (outdoor)** | — | **16.86** | **42.98** | **2.55×** |
| **mean (indoor)**  | — | **32.18** | **36.41** | **1.13×** |
| **mean (all)**     | — | **23.67** | **40.06** | **1.93×** |

**Read:** cat consistently wins, but the gap is scene-dependent. Outdoor
scenes (bicycle/garden/stump/flowers/treehill) get a **~2.5× boost** —
baseline has many more surfels there because there's no per-pixel
consolidation, and cat's hashgrid+MLP halves alpha-blending work. Indoor
scenes (counter/room/kitchen/bonsai) see only **~10-20%** — the surfel
counts are already similar.

## Nexels — Tanks-and-Temples

`tnt_cap400000` checkpoints (iter 30000), capped at 400k points.

| Scene  | N points  |  FPS   |
|--------|----:|--------:|
| **train** | 399,745 | 130.16 |
| **truck** | 399,789 | 126.96 |

**Read:** nexels caps surfel count to 400k, so per-scene FPS is dominated
by the fixed cap rather than scene complexity. Both TnT scenes land in
the same ballpark (~127-130 FPS).

## FastGS — RTX 4090

Trained locally on the 5090 (`/home/nilkel/Projects/FastGS/`) using the
authors' per-scene `train_base.sh` recipes (30k iters, default args
including per-scene `--grad_abs_thresh`, `--dense`, `--highfeature_lr`).
Ckpts + datasets then rsynced to `neel@10.176.128.69:~/fastgs/`. FPS
measured with FastGS's own `render_eval.py` (`--num_warmup 10
--num_benchmark 200`, cuda.Event timing). Matches the methodology of
`bench_minimal.py` exactly — cycle test cameras, warmup, time 200
frames, mean ms/frame, FPS = 1000/mean.

Run date: 2026-05-13.

### Mip-NeRF 360

| Scene    | N points | resolution | FPS   |
|----------|---------:|:----------:|------:|
| counter  | 208,171  | 1600×1066  | 915.25 |
| room     | 207,167  | 1600×1066  | **1063.69** |
| kitchen  | 379,671  | 1600×1067  | 772.83 |
| bonsai   | 275,719  | 1600×1066  | 992.38 |
| stump    | 392,348  | 1600×1060  | 976.06 |
| treehill | 394,335  | 1600×1050  | 962.31 |
| flowers  | 489,951  | 1600×1054  | 934.88 |
| bicycle  | 539,624  | 1600×1063  | 925.30 |
| garden   | 660,711  | 1297×840   | 937.51 |
| **mean (outdoor)** | — | — | **947.21** |
| **mean (indoor)**  | — | — | **936.04** |
| **mean (all 9)**   | — | — | **942.25** |

### Tanks and Temples

| Scene  | N points | resolution | FPS |
|--------|---------:|:----------:|----:|
| truck  | 252,868  | 979×546    | **1268.83** |
| train  | 231,398  | 980×545    | 1076.69 |
| **mean** | — | — | **1172.76** |

Note: `--mult 0.7` was passed for both TnT scenes (matches the authors'
`render.py` invocations in `train_base.sh`). PSNR/SSIM/LPIPS are 0.000
in the raw JSON because `render_eval.py` parses metrics off a path
key our aggregator didn't lift — only the FPS numbers above are
authoritative from this run.

Per-scene logs: on the 4090 at `~/fastgs/bench_logs/<scene>.log`. Raw TSV:
`~/fastgs/bench_logs/fps_results.tsv`. Trained ckpts:
`/home/nilkel/Projects/FastGS/output/<scene>/` (locally on 5090) and
`~/fastgs/output/<scene>/` (mirrored on 4090).

## Reproducing

### Mip-360 (one scene)

```bash
conda run -n nest_splatting python benchmark_render.py \
  -m data/sherlock_ckpts/mip_360/<scene>/baseline/vanilla_baseline \
  -s data/mip_360/<scene> \
  --yaml ./configs/360_outdoor.yaml \      # or 360_indoor.yaml
  --method baseline \                       # or `--method cat --hybrid_levels 5`
  --warmup 20 --bench_iters 200 --skip_save
```

Per-scene logs at `/tmp/nest_bench/<scene>_<method>.log`; aggregate TSV at
`/tmp/nest_bench/fps_results.tsv`.

### Nexels (one scene)

```bash
cd /home/nilkel/Projects/nexels
conda run -n nexels python render.py \
  -m /home/nilkel/Projects/nest-splatting/data/sherlock_ckpts/nexels/tnt_cap400000/<scene> \
  -s /home/nilkel/Projects/nest-splatting/data/tnt/<scene> \
  --iteration 30000 --skip_train --skip_ellipse --quiet
# FPS written to: <model_path>/test/ours_30000/info.json
```

## Notes / gotchas

- **`benchmark_render.py` arg name**: training-config `iterations` (in yaml's
  `training_cfg`) collides with the bench's `--iterations`. The script was
  renamed to `--bench_iters` to dodge this. If you re-touch it, don't
  revert.
- **`--iteration` auto-detect**: bench picks the latest `ngp_*.pth` if
  `--iteration` is left at the default `-1`.
- **Path rewriting**: every saved `cfg_args` from sherlock has a sherlock
  source_path. Pass `-m <local_ckpt>` and `-s <local_data>` to override.
- **Indoor vs outdoor yaml** matters at preprocess: `360_indoor.yaml` and
  `360_outdoor.yaml` differ in distortion/normal lambdas. For FPS-only this
  doesn't change the renderer math, but pick the correct one so the scene
  contraction matches.
