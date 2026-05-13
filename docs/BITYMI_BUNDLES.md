# BITYMI Bundles — bake → package → upload pipeline

End-to-end docs for taking a trained `3D_SH_res` checkpoint, baking its
neural residual into a textured atlas, packing it into a single-file
`.bitymi` viewer bundle, and publishing to the `Nilkel/bitymi-demos`
dataset on Hugging Face so it shows up in the
[live WebGPU demo page](https://nilkel.github.io/bitymi-demos/).

> The viewer-side bundle format and JS unpacker live in
> `Halloumi-WS` and `Halloumi-web-splat`. This doc only covers the
> nest-splatting producer pipeline.

---

## Pipeline overview

```
trained checkpoint            (point_cloud.ply + ngp_*.pth + args.pkl + config.yaml)
        │
        │  scripts/benchmark_baked.py  (bake + bench in one call)
        ▼
baked_atlas/                  baked.ply + atlas_texture.bc7 + atlas_rects.pt + bake_meta.json
        │
        │  scripts/export_textures_bin.py        (BC7  path)
        │  scripts/encode_astc.py                (ASTC path)
        ▼
scene.nat2 / scene_astc.nat2   single binary header + atlas bytes
        │
        │  scripts/pack_bitymi.py
        ▼
<scene>[_sb][_astc][_lite].bitymi
        │
        │  hf upload Nilkel/bitymi-demos ...
        ▼
   HF dataset → viewer fetches at run time
```

Three independent variations are produced from one checkpoint:

| axis | options | meaning |
|---|---|---|
| **detail** | `HD` / `lite` | `max_res=64` vs `max_res=32` at bake time |
| **GPU format** | `BC7` / `ASTC 4×4` | desktop+Apple vs Android phones |
| **feature mode** | `SV` (default) / `_sb` | which directional model the checkpoint used |

This gives up to 8 bundles per scene. In practice the project ships:

- `<scene>.bitymi`            ← SV HD, BC7
- `<scene>_lite.bitymi`       ← SV lite, BC7
- `<scene>_astc.bitymi`       ← SV HD, ASTC
- `<scene>_astc_lite.bitymi`  ← SV lite, ASTC
- `<scene>_sb.bitymi`         ← SB HD, BC7
- `<scene>_sb_astc.bitymi`    ← SB HD, ASTC

The viewer page picks the right bundle per device.

---

## Stage 1 — Bake the atlas

`scripts/benchmark_baked.py` does both bake and bench in one call. The
bake step produces a packed BC7 atlas plus per-Gaussian rects and a
metadata JSON.

```bash
conda run -n nest_splatting python scripts/benchmark_baked.py \
    --model_path $MP \
    --max_res 64                  # HD; use 32 for lite
    --atlas_budget_mb 8192        # safe ceiling; auto-shrinks if needed
    --aabb_mode 5 --sort_mode 0   # SnugBox + AccuTile, single-pass sort
    --bake_dtype bc7              # uint8 atlas → BC7 in one go
    --num_warmup 10 --num_benchmark 200
```

Use `--output_dir $MP/baked_atlas_lite` to place a lite bake alongside
the HD `baked_atlas/`.

Outputs in `<output_dir>` (default `<model_path>/baked_atlas/`):

| file | what it is |
|---|---|
| `atlas_texture.pt` | uint8 RGBA `[H, W, 4]` post-quantization |
| `atlas_texture.bc7` | BC7-compressed atlas as raw byte stream |
| `atlas_rects.pt`   | `[N, 4]` per-surfel `(u0_px, v0_px, w_px, h_px)` |
| `baked.ply`        | Gaussian state with SH + SV/SB params |
| `bake_meta.json`   | atlas dims, `scale`/`offset` for dequant, kernel type, feature mode, SV/SB metadata |
| `benchmark_results.json` | PSNR/SSIM/LPIPS/FPS at the bake site |

Pitfalls:
- **`Scene()` overwrites baked.ply** during render-side load. Tests
  that reload after `Scene()` ctor are correct; ad-hoc rendering paths
  must re-call `gaussians.load_ply(baked_ply)` after the constructor.
- **Default `--atlas_budget_mb 2048`** silently clamps `max_res 64→16`
  on dense scenes. Use 8192 for full-res HD bakes.
- The bench portion (`--num_warmup`/`--num_benchmark`) is just a
  sanity check; rendered PSNR/SSIM/LPIPS+FPS land in
  `baked_atlas/benchmark_results.json`. Skip with `--num_benchmark 0`
  if you only want the bake artifacts.

---

## Stage 2 — Convert atlas to a NAT2 container

The atlas needs to be packed into a single binary the viewer can
mmap. `scripts/export_textures_bin.py` writes `scene.nat2`
(BC7-flavored); `scripts/encode_astc.py` writes `scene_astc.nat2`
(ASTC 4×4-flavored). Both use the same NAT2 v2 container format —
see the docstring in `export_textures_bin.py` for the byte layout.

### BC7 (desktop, iOS Safari, modern Apple)

```bash
python scripts/export_textures_bin.py <baked_atlas_dir> <baked_atlas_dir>/scene.nat2
```

Re-reads `atlas_texture.bc7` directly. Splits the atlas into layers
(8192-row chunks) so it fits Android-Adreno-style WebGPU
`max_texture_dimension_2d=8192` limits, since the same BC7 bundle
might be served to a multi-platform viewer.

### ASTC 4×4 (Android — Adreno/Mali devices lack BC7)

```bash
python scripts/encode_astc.py <baked_atlas_dir> --quality medium
```

Re-encodes the uint8 atlas via `astc-encoder-py` (`pip install
astc-encoder-py`; no system astcenc install needed). Same 8192-row
layer split. Output: `<baked_atlas_dir>/scene_astc.nat2`.

ASTC 4×4 has 1 byte/texel like BC7 but slightly worse quality at the
same bitrate — the trade-off for being mobile-deployable.

Both NAT2 files contain:
- Header: atlas dims, kernel type, dequant `scale`/`offset`, feature
  metadata (sb_number, etc.)
- Per-layer cut points (so loader splits into a `texture_2d_array`)
- Per-Gaussian rects table `[N, 4]`
- The raw block-compressed atlas bytes (concatenated per layer)
- Optional SB params block at the tail

---

## Stage 3 — Pack the .bitymi bundle

`scripts/pack_bitymi.py` concatenates three pieces into a single
viewer-fetchable blob:

```bash
python scripts/pack_bitymi.py <output>.bitymi --bake-dir <baked_atlas_dir>
```

`--bake-dir` auto-discovers:
- `baked.ply`         (in the bake dir)
- `cameras.json`      (walks up to grandparent — typically lives in
                       the model dir, written by the training Scene class)
- `scene.nat2` or `scene_astc.nat2`  (whichever exists in the bake dir)

You can override any of those with `--ply`, `--cameras`, `--atlas`.

If your model dir doesn't have `cameras.json` (eg. brand-new
checkpoints), copy from a sibling config or the previous training run:

```bash
cp $NEST/outputs/mip_360/<scene>/3D_SH_res/<sibling_config>/cameras.json $MP/cameras.json
```

The `.bitymi` is a TLV-style container with a 4-byte magic +
chunk-table header, then raw payloads. JS-side unpacker lives in the
Halloumi-WS / Halloumi-web-splat viewer.

### FP16 PLY round-trip (optional but recommended)

`build_bc7_bundles_fp16.py` / `build_astc_bundles_fp16.py` first run an
in-place round-trip of the SV/SB color params (`f_dc_*`, `f_rest_*`,
`sv_*`, `sb_*`) through FP16 → FP32, producing `baked_fp16.ply` next to
`baked.ply`. This halves the per-Gaussian color storage with bit-identical
rendering (the underlying CUDA path reads via `__half`). Use the
`_fp16` PLY when packing for production bundles to shrink the
bundle by ~30%.

```python
# from scripts/build_bc7_bundles_fp16.py
fp16_roundtrip_ply(baked / "baked.ply", baked / "baked_fp16.ply")
pack(bundle, fp16_ply, cams, bc7_nat2)
```

---

## Stage 4 — Upload to Hugging Face

The viewer fetches bundles from `https://huggingface.co/datasets/Nilkel/bitymi-demos`.
GitHub Pages can't hold >100 MB blobs and GitHub Release CDN doesn't
send CORS headers cross-origin; HF dataset URLs work because their CDN
sends `Access-Control-Allow-Origin: https://nilkel.github.io` on both
the 302 redirect and the final blob.

### Single-file upload

```bash
conda run -n nest_splatting hf upload \
    Nilkel/bitymi-demos \
    /path/to/<scene>.bitymi \
    <scene>.bitymi \
    --repo-type=dataset \
    --commit-message="Add <scene> (...variant)"
```

`hf` ships with the `huggingface_hub` Python package (installed in the
`nest_splatting` env). Auth via `hf auth login` once per machine
(reads from `~/.huggingface/token`).

### Folder upload (batch sweep)

```bash
conda run -n nest_splatting hf upload-large-folder \
    Nilkel/bitymi-demos \
    /path/to/scenes/ \
    --repo-type=dataset \
    --include "*.bitymi"
```

Used by `sv_lite_bake_upload.sh` and friends. Faster than per-file
calls for >5 bundles because it parallelizes shard uploads.

---

## Naming conventions

```
<scene>[_<feature>][_<format>][_<tier>].bitymi
```

| part | required | example | notes |
|---|---|---|---|
| `<scene>` | yes | `bicycle` | mip-360 scene name; *or* a variant tag like `bicycle_opaque` |
| `<feature>` | no | `sb` | omit for SV (default); set `sb` for `--feature beta` checkpoints |
| `<format>` | no | `astc` | omit for BC7 (default); set `astc` for ASTC 4×4 |
| `<tier>` | no | `lite` | omit for HD; set `lite` for `max_res=32` |

Examples:
- `bicycle.bitymi`            → SV HD BC7
- `bicycle_lite.bitymi`       → SV lite BC7
- `bicycle_astc.bitymi`       → SV HD ASTC
- `bicycle_astc_lite.bitymi`  → SV lite ASTC
- `bicycle_sb.bitymi`         → SB HD BC7
- `bicycle_sb_astc.bitymi`    → SB HD ASTC
- `bicycle_opaque.bitymi`     → custom variant (different training config); independent name

---

## Batch helpers

| script | what it does |
|---|---|
| `scripts/build_bc7_bundles_fp16.py` | Sweep the 9 mip-360 scenes × {SV HD, SB HD} → 18 BC7 bundles. FP16-roundtrips PLYs, ensures `scene.nat2`, packs, uploads. |
| `scripts/build_astc_bundles_fp16.py` | Same for ASTC: 9 scenes × {SV HD, SV lite, SB HD} → 27 ASTC bundles. |

Both are idempotent: skip stages whose outputs exist. To rerun
everything cleanly: delete `baked_fp16.ply`, `scene.nat2`,
`scene_astc.nat2`, and `/tmp/{bc7,astc}_bundles_fp16/*` first.

```bash
# Just SV bundles for bicycle + garden
python scripts/build_bc7_bundles_fp16.py --scenes bicycle garden

# Same but dry-run (skip upload, keep local bundles)
python scripts/build_bc7_bundles_fp16.py --dry-run
```

Both scripts hardcode:
- `BAKE_ROOT = /mnt/nilkel_hdd/outputs/mip_360`
- `SV_DIR    = {scene}/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac`
- `SB_DIR    = {scene}/3D_SH_res/SB_30thr_005w25gLP4lev_FRP5k10_c2f_Jac`

If you trained with a different config name, **copy the script and edit
the templates** rather than passing flags — the templates are baked in
on purpose to keep the official sweep reproducible.

---

## Single-scene template (one-off variants)

For ad-hoc variants (e.g. `bicycle_opaque`), the batch script is the
wrong tool. Use the template at `/tmp/bicycle_opaque_pipeline.sh` (or
the `sv_lite_bake_upload.sh` precursor):

```bash
#!/usr/bin/env bash
set -u
NEST=/home/nilkel/Projects/nest-splatting
DEMO_SCENES=/home/nilkel/Projects/bitymi-demos/scenes
MP=$NEST/outputs/mip_360/<scene>/3D_SH_res/<your_config>
BAKE=$MP/baked_atlas                       # use baked_atlas_lite for lite
BUNDLE_NAME=<scene>_<variant>.bitymi       # e.g. bicycle_opaque.bitymi
PY="conda run -n nest_splatting python"

mkdir -p "$DEMO_SCENES"

# 1) Bake (HD = max_res 64; lite = max_res 32 + --output_dir $BAKE_LITE)
$PY scripts/benchmark_baked.py --model_path "$MP" \
    --max_res 64 --atlas_budget_mb 8192 \
    --aabb_mode 5 --sort_mode 0 --bake_dtype bc7 \
    --num_warmup 10 --num_benchmark 200

# 2) NAT2 (BC7 — swap for encode_astc.py for ASTC build)
$PY scripts/export_textures_bin.py "$BAKE" "$BAKE/scene.nat2"

# 3) Pack
$PY scripts/pack_bitymi.py "$DEMO_SCENES/$BUNDLE_NAME" --bake-dir "$BAKE"

# 4) Upload
conda run -n nest_splatting hf upload \
    Nilkel/bitymi-demos \
    "$DEMO_SCENES/$BUNDLE_NAME" \
    "$BUNDLE_NAME" \
    --repo-type=dataset \
    --commit-message="Add $BUNDLE_NAME"
```

For lite + ASTC variants, copy and swap the relevant paths /
encoder. To chain HD + lite without GPU contention, gate the lite
pipeline on `tail --pid=$HD_PID -f /dev/null` so the lite bake only
fires once the HD bake exits.

---

## HD vs lite — size and quality

| tier | `max_res` | mean atlas size (mip-360) | mean PSNR drop vs HD |
|---|---:|---:|---:|
| HD | 64 | ~640 MB BC7 / ~600 MB ASTC | reference |
| lite | 32 | ~160 MB BC7 / ~150 MB ASTC | −0.4 to −0.8 dB |

Lite is the right default for mobile / slow-connection users. The
viewer's quality toggle (`Quality: HD / Lite` button) lives in
`bitymi-demos/index.html` and just swaps the bundle URL.

---

## BC7 vs ASTC — which to ship

| platform | BC7 (`.bitymi`) | ASTC 4×4 (`_astc.bitymi`) |
|---|---|---|
| desktop Chrome / Edge | ✓ | ✓ |
| desktop Firefox | ✓ | ✓ |
| iOS Safari 18.2+ | ✓ | ✓ (but BC7 preferred) |
| Android (Adreno/Mali) | ✗ — no HW support | ✓ — required |
| Chrome on WebGPU emulation | ✓ | ✓ |

The viewer probes `navigator.gpu.adapter.features` for
`texture-compression-bc` vs `texture-compression-astc` and picks the
matching bundle URL. Always upload **both** formats for every scene
you want phone users to load.

---

## Common pitfalls

| symptom | cause | fix |
|---|---|---|
| HF download blocked by CORS on `nilkel.github.io` | uploading to GitHub Release CDN instead of HF | use HF dataset; HF sends `Access-Control-Allow-Origin: https://nilkel.github.io` |
| Bundle loads but renders all-black | viewer fetched ASTC bundle but adapter only has BC7 (or vice-versa) | match adapter feature → bundle suffix |
| `cameras.json not found` during pack | new model dir doesn't have it (rare with recent training) | copy from a sibling config |
| `scene.nat2` not in baked dir | `export_textures_bin.py` not yet run | run it; check `bake_meta.json` exists |
| BC7 encode crashes with `bc7encoder not installed` | submodule not built | `cd submodules/bc7enc_lib && pip install -e .` |
| ASTC encode crashes with `from astc_encoder import ...` | pip package missing | `pip install astc-encoder-py` |
| HF push 401 | not authenticated | `hf auth login` (paste a write-scoped token) |
| `--atlas_budget_mb` clamped `max_res 64 → 16` | default 2048 budget too small for large scenes | always pass `--atlas_budget_mb 8192` |
| Upload silently overwrites existing | same `path_in_repo` | use a distinct filename for variants (e.g. `bicycle_opaque.bitymi`, not `bicycle.bitymi`) |

---

## Where the files live (quick reference)

- **Scripts (producer side)**: `nest-splatting/scripts/`
- **Default bake root** (for batch helpers): `/mnt/nilkel_hdd/outputs/mip_360/`
- **Demo scenes staging dir**: `bitymi-demos/scenes/`
- **HF dataset**: [`Nilkel/bitymi-demos`](https://huggingface.co/datasets/Nilkel/bitymi-demos)
- **Viewer page**: [`bitymi-demos`](https://nilkel.github.io/bitymi-demos/) (page index in `bitymi-demos/index.html`)
- **Viewer source**: `Halloumi-WS` / `Halloumi-web-splat` (separate repos)
