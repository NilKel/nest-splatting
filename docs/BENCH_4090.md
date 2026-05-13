# Benchmarking on the 4090 box

Self-contained FPS / PSNR / SSIM / LPIPS bench for **baked** nest-splatting
scenes (BC7 atlas), running on `neel@10.176.128.69`. The 4090 box has no
nest-splatting source — only `diff_surfel_bake_render` is built there, and
each scene is shipped as a self-contained "bench bundle" snapshot.

## SSH

Passwordless SSH key is installed. Direct `ssh neel@10.176.128.69` works.
(See `/home/nilkel/.claude/projects/-home-nilkel-Projects-nest-splatting/memory/reference_4090_box.md`.)

If you need to re-install the key from a different machine:
```bash
ssh neel@10.176.128.69  # password: himalaya
# inside:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<your ssh-ed25519 pubkey>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Layout on the 4090

```
~/nest-bench/
  bench_4090/                   # scripts (mirrored from scripts/bench_4090/)
    setup_4090.sh
    run_all.sh
    bench_minimal.py
    README.md
  diff_surfel_bake_render/      # CUDA src + built .so for SM 8.9
  bundles/<scene>/              # one dir per scene we've baked
    gaussian_state.pt           # pre-activated tensors snapshot
    atlas_texture.bc7
    atlas_rects.pt
    bake_meta.json
    cameras.pt                  # test split only
    images/<name>.png           # GT at eval resolution
    sb_params.pt                # only for --feature beta runs
  bench_results/<scene>.json    # per-scene metrics output
  lpipsPyTorch/                 # vendored from nest-splatting for LPIPS parity
```

## Procedure

### 1. Build the bundle locally (once per model)

Bundles are built on a host that has the full nest-splatting source +
`baked_atlas/atlas_texture.bc7` already present. The model dir must contain
`args.json`/`args.pkl`, `config.yaml`, `point_cloud/iteration_*/point_cloud.ply`,
and `baked_atlas/`.

```bash
cd /home/nilkel/Projects/nest-splatting

MODEL=/home/nilkel/Projects/nest-splatting/outputs/mip_360/bicycle/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_JBT3_rbgact16dist100_3k
NAME=mip_360_bicycle_jbt3dist100

conda run -n nest_splatting python scripts/build_bench_bundle.py \
    --model_path "$MODEL" \
    --out_dir /tmp/bench_bundle_$NAME/$NAME
```

The bundle is ~700MB-1GB depending on N and atlas size. Output goes under
`/tmp/bench_bundle_$NAME/$NAME/` (the trailing `$NAME` is the bundle slug
that the 4090 will use to identify it).

### 2. Rsync to the 4090

```bash
rsync -a --info=progress2 \
    /tmp/bench_bundle_$NAME/$NAME/ \
    neel@10.176.128.69:~/nest-bench/bundles/$NAME/
```

LAN throughput ~110 MB/s → roughly 6-10 s per bundle.

### 3. Run the bench

The `python` is in the remote conda env `bench`. Because `ssh <host> "..."`
runs via `sh -c`, not bash, use `bash -c '...'` to keep `conda activate`
working:

```bash
ssh neel@10.176.128.69 "bash -c '
. ~/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd ~/nest-bench/bench_4090
python bench_minimal.py \
    --bundle ~/nest-bench/bundles/$NAME \
    --num_warmup 10 --num_benchmark 200 \
    --out ~/nest-bench/bench_results/$NAME.json
'"
```

Defaults: 10 warmup frames + 200 benchmark frames, `cuda.Event` timing.

### 4. Read the results

```bash
ssh neel@10.176.128.69 "cat ~/nest-bench/bench_results/$NAME.json"
```

JSON keys: `n_gaussians, n_cameras, resolution, psnr, ssim, lpips, fps, ms_per_frame`.

## Running all scenes at once

```bash
ssh neel@10.176.128.69 "bash -c '
. ~/miniforge3/etc/profile.d/conda.sh && conda activate bench
cd ~/nest-bench/bench_4090
./run_all.sh
'"
```

Iterates `~/nest-bench/bundles/*/` and emits a markdown table to stdout.

## Bench methodology

- **Timing**: `torch.cuda.Event(enable_timing=True)` with explicit
  `cudaDeviceSynchronize`. Excludes Python wall-clock noise. This is the
  "GPU-throughput" convention (matches gsplat/our 5090 internal numbers).
  Not paper-style `time.time` (no-warmup) — those numbers are typically
  20-40% lower; see `scripts/bench_4090/bench_paperstyle.py` if needed.
- **Resolution**: whatever the bundle was baked at (typically eval-res:
  1237×822 for bicycle, 779×519 for bonsai/kitchen, etc.).
- **LPIPS backend**: vendored `lpipsPyTorch` (VGG) so numbers match the
  5090-side reference exactly.

## Updating the bench scripts themselves

Bench scripts live in `scripts/bench_4090/` in the nest-splatting tree and
are mirrored to the 4090 with:

```bash
scp scripts/bench_4090/setup_4090.sh \
    scripts/bench_4090/run_all.sh \
    scripts/bench_4090/bench_minimal.py \
    scripts/bench_4090/README.md \
    neel@10.176.128.69:~/nest-bench/bench_4090/
```

(These specific `scp` invocations are already allow-listed in
`.claude/settings.json` so they won't prompt.)

## Common pitfalls

- **`source: not found`** when running through ssh: remote `sh` is dash.
  Use `bash -c '. ~/miniforge3/etc/profile.d/conda.sh && ...'` (note `.`,
  not `source`) or wrap your command in `bash -lc`.
- **`No such file or directory: gaussian_state.pt`** at bench time: the
  bundle wasn't rsynced under the right name. Bundle dir on the 4090 must
  match the `--bundle` arg, not the `baked_atlas` directory layout.
- **Atlas size mismatch warnings**: `bench_minimal.py` reads
  `atlas_bc7_padded_w/h` from `bake_meta.json`; make sure the bundle's
  `bake_meta.json` was generated by the same `build_bench_bundle.py`
  version (it injects the BC7 padding fields).
- **PSNR ~3 dB lower than expected**: the bundle was baked at a different
  resolution than the GT images shipped inside it. Re-run `build_bench_bundle`
  with the same eval-resolution flag the original `benchmark_baked.py` used.
