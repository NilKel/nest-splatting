# Video → 3DGS training dataset

End-to-end pipeline: any `.mp4` (or other ffmpeg-readable video) → a folder
the nest-splatting COLMAP loader can train on directly (i.e. a 3DGS
`--source` layout with `images/`, `sparse/0/`, and pre-built `images_2/` /
`images_4/` mips).

The wrapped tooling lives in
[`/home/nilkel/Projects/video-3d-reconstruction-gsplat`](https://github.com/nannigalaxy/video-3d-reconstruction-gsplat)
(submodule-ish: cloned separately from nest-splatting). We only call its
`colmap_undistorted_sfm_export.sh` — the speedy-splat training step it ships
with is irrelevant since we train with nest-splatting.

## One-shot command

```bash
scripts/video_to_dataset.sh <input_video> <out_dir> [fps]
```

Defaults: `fps=10`. Re-runs are idempotent (each stage skips if its output
already exists).

Example — what produced `data/personal/Max/`:

```bash
scripts/video_to_dataset.sh \
    /home/nilkel/Projects/nest-splatting/data/personal/Max/Max.mp4 \
    /home/nilkel/Projects/nest-splatting/data/personal/Max \
    10
```

After it finishes:

```bash
python train.py \
    -s /home/nilkel/Projects/nest-splatting/data/personal/Max \
    -m outputs/personal/Max/<run_name> \
    --yaml configs/<scene>.yaml \
    --method 3D_SH_res ...
```

(`360_indoor.yaml` is usually a sane starting point for handheld-video
scenes — unbounded background, MERF contract on, mask reg off.)

## Pipeline (6 stages)

| # | stage | tool | output |
|---|---|---|---|
| 1 | extract frames | `ffmpeg -vf "fps=$FPS"` | `images_raw/frame_%04d.png` |
| 2 | SfM + undistort | COLMAP via `colmap_undistorted_sfm_export.sh` | `sparse/`, `undistorted/{images,sparse/0}` |
| 3 | restructure | `mv` | promote `undistorted/{images,sparse}` to top level; rename pre-undistort sparse → `sparse_distorted/`; drop `database.db`, `images_raw/`, `undistorted/` |
| 4 | image mips | PIL LANCZOS | `images_2/`, `images_4/` |
| 5 | PLY export | `colmap model_converter` | `sparse/0/points3D.ply` |
| 6 | summary | log | counts + final tree |

COLMAP runs feature_extractor → sequential_matcher → mapper → image_undistorter
(the wrapper script does all four). Sequential matching is the right choice
for video (consecutive frames overlap heavily); switch to `--exhaustive` in
the wrapper only for unordered photo sets.

## Final layout (nest-splatting `--source`)

```
<out>/
├── Max.mp4                  (if you copied the video in)
├── pipeline.log             (full run log)
├── images/                  (undistorted, full-res)
├── images_2/                (half — for `-r 2`)
├── images_4/                (quarter — for `-r 4`)
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       ├── points3D.bin
│       └── points3D.ply     (diagnostic; nest-splatting loads from .bin)
└── sparse_distorted/        (pre-undistort recon, kept for debug — safe to delete)
```

The nest-splatting COLMAP loader (`scene/dataset_readers.py:readColmapSceneInfo`)
keys on `<source>/sparse/` existing, so this layout drops straight in.

## Headless gotchas (baked into the script)

* **ffmpeg not on PATH**: training boxes often only have ffmpeg inside the
  conda env. The script does `conda run -n nest_splatting which ffmpeg` to
  resolve the binary before calling it.
* **COLMAP Qt abort over SSH**: COLMAP links Qt5 and aborts at startup if it
  can't open an X display (`qt.qpa.xcb: could not connect to display` →
  `Aborted (core dumped)`). Fix: `export QT_QPA_PLATFORM=offscreen`. The
  script sets this before invoking `colmap_undistorted_sfm_export.sh`.
* **COLMAP GPU SIFT needs an OpenGL context**: even with `QT_QPA_PLATFORM=offscreen`,
  the GPU SIFT extractor / matcher fails on a headless box with
  `Check failed: context_.create()` (opengl_utils.cc:54) because it tries to
  create a real GL context. Default fix: pass `--disable_gpu` (CPU SIFT —
  ~3–5× slower, but rock-solid). Override with `USE_GPU=1` env var if you
  have `xvfb` or an X session — e.g. `USE_GPU=1 scripts/video_to_dataset.sh ...`
  inside `xvfb-run -a` or with `$DISPLAY` set.

## Tuning notes

| knob | when to change | how |
|---|---|---|
| `fps` | longer / shorter video, denser parallax | last arg; ≈ `desired_frames / video_duration_s`. 300–500 frames usually a good target. |
| `--exhaustive` | photo set, not video | edit script: replace `--enable_gpu` with `--exhaustive --enable_gpu` in stage 2 |
| matcher type | unordered sequence | same as above |
| video too long | >60 s | trim first, or drop `fps` so total frames stay ≤ ~500 — COLMAP scaling gets painful past that |
| sparse recon empty | low-texture / fast motion | bump `fps`; verify ≥ 80 % frames register in COLMAP log (`Registering image #...`) |

## Reproducing `data/personal/jhabbu/` (legacy `Max.mp4` run)

The May 2026 run used `fps≈4` (113 frames). The current default (`fps=10`)
gives ~295 frames on the same 29.6 s video — denser parallax, more reliable
matcher convergence. Both are valid; pick `fps` based on motion speed.

## Where everything lives

* Script: [`scripts/video_to_dataset.sh`](../scripts/video_to_dataset.sh)
* Upstream COLMAP wrapper: `/home/nilkel/Projects/video-3d-reconstruction-gsplat/colmap_undistorted_sfm_export.sh`
* Training entry point: [`train.py`](../train.py) (nest-splatting)
* COLMAP loader: [`scene/dataset_readers.py`](../scene/dataset_readers.py) — `readColmapSceneInfo`
