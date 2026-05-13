# Deploy a scene to bitymi-demos — quick procedure

Operational summary for taking a trained `3D_SH_res` checkpoint and
getting it live on [bitymi-demos](https://nilkel.github.io/bitymi-demos/).
For the deep reference (NAT2 format, FP16 round-trip, ASTC vs BC7
trade-offs, batch helpers), see [BITYMI_BUNDLES.md](BITYMI_BUNDLES.md).

---

## Which viewer is live?

**Rust `Halloumi-web-splat` (WASM)**. Cards in
[`bitymi-demos/index.html`](../../bitymi-demos/index.html) all link to
`viewer/index.html?bundle=...`, which serves
`viewer/web_splats_bg.wasm` + `viewer/web_splats.js`.

The TypeScript `Halloumi-WS` is staged in `bitymi-demos/viewer-ws/`
but **not** wired into any demo card. See
[HALLOUMI_WS_VIEWER.md § 12](HALLOUMI_WS_VIEWER.md) for the swap
procedure if/when we promote it.

---

## What gets uploaded per scene

Each scene typically ships 4 bundles to `Nilkel/bitymi-demos`:

| filename | tier | format | purpose |
|---|---|---|---|
| `<slug>.bitymi` | HD | BC7 | desktop / iOS default |
| `<slug>_lite.bitymi` | lite | BC7 | low-bandwidth / older desktop |
| `<slug>_astc.bitymi` | HD | ASTC 4×4 | Android (Adreno/Mali) |
| `<slug>_astc_lite.bitymi` | lite | ASTC | Android slow connection |

The viewer probes `navigator.gpu.adapter.features` and picks the
matching bundle. SB variants (`<slug>_sb*`) only if the checkpoint
used `--feature beta`.

---

## Four-stage pipeline

```
checkpoint  →  baked_atlas/  →  scene.nat2  →  <slug>.bitymi  →  HF dataset
   (1)            (2)              (3)             (4)
```

| stage | script | what it does |
|---|---|---|
| 1. Bake | `scripts/benchmark_baked.py` | run MLP residual to 8×8 SH atlas per Gauss, pack to BC7 |
| 2. NAT2 | `scripts/export_textures_bin.py` (BC7) / `scripts/encode_astc.py` (ASTC) | wrap atlas in single binary container with header + rects |
| 3. Pack | `scripts/pack_bitymi.py` | concat PLY + cameras.json + scene.nat2 into one `.bitymi` |
| 4. Upload | `hf upload Nilkel/bitymi-demos ... --repo-type=dataset` | push to HF (CORS works because HF CDN sends `Access-Control-Allow-Origin` for `nilkel.github.io`) |

---

## Single-scene template

Copy [`/tmp/bicycle_random_dist_pipeline.sh`](/tmp/bicycle_random_dist_pipeline.sh) (or any
recent `*_pipeline.sh`) and edit `MP=`, `SLUG=`. The script bakes HD +
lite, exports BC7 + ASTC, packs, and uploads all four bundles. Skips
stages whose outputs exist, so safe to re-run.

Skeleton:

```bash
NEST=/home/nilkel/Projects/nest-splatting
DEMO_SCENES=/home/nilkel/Projects/bitymi-demos/scenes
MP=$NEST/outputs/mip_360/<scene>/3D_SH_res/<your_config>
SLUG=<scene_or_variant>
BAKE_HD=$MP/baked_atlas
BAKE_LITE=$MP/baked_atlas_lite
PY="conda run -n nest_splatting python"

# HD bake (max_res 64)
$PY scripts/benchmark_baked.py --model_path "$MP" --output_dir "$BAKE_HD" \
    --max_res 64 --atlas_budget_mb 8192 \
    --aabb_mode 5 --sort_mode 0 --bake_dtype bc7 \
    --num_warmup 5 --num_benchmark 50

# lite bake (max_res 32)
$PY scripts/benchmark_baked.py --model_path "$MP" --output_dir "$BAKE_LITE" \
    --max_res 32 --atlas_budget_mb 8192 \
    --aabb_mode 5 --sort_mode 0 --bake_dtype bc7

# BC7 nat2 + pack + upload (HD)
$PY scripts/export_textures_bin.py "$BAKE_HD" "$BAKE_HD/scene.nat2"
$PY scripts/pack_bitymi.py "$DEMO_SCENES/${SLUG}.bitymi" --bake-dir "$BAKE_HD"
conda run -n nest_splatting hf upload Nilkel/bitymi-demos \
    "$DEMO_SCENES/${SLUG}.bitymi" "${SLUG}.bitymi" --repo-type=dataset \
    --commit-message="Add ${SLUG}"

# ASTC nat2 + pack + upload (HD)
$PY scripts/encode_astc.py "$BAKE_HD" --quality medium
$PY scripts/pack_bitymi.py "$DEMO_SCENES/${SLUG}_astc.bitymi" \
    --ply "$BAKE_HD/baked.ply" --cameras "$MP/cameras.json" \
    --atlas "$BAKE_HD/scene_astc.nat2"
conda run -n nest_splatting hf upload Nilkel/bitymi-demos \
    "$DEMO_SCENES/${SLUG}_astc.bitymi" "${SLUG}_astc.bitymi" \
    --repo-type=dataset --commit-message="Add ${SLUG}_astc"

# repeat the ASTC + BC7 pack/upload blocks for lite with BAKE_LITE
```

---

## Add the card to bitymi-demos

After upload, append a `<a class="demo">` entry to
[`bitymi-demos/index.html`](../../bitymi-demos/index.html) (find the
`<section class="grid">` of demo cards). The HD size is what shows in
the meta line; the `data-astc-*` attributes let the viewer swap to
ASTC on Android.

```html
<a class="demo" data-scene="<slug>" data-hd-mb="<HD MB rounded>"
   data-astc-hd="<slug>_astc.bitymi"
   data-astc-lite="<slug>_astc_lite.bitymi"
   href="viewer/index.html?bundle=https://huggingface.co/datasets/Nilkel/bitymi-demos/resolve/main/<slug>.bitymi">
    <img class="thumb" src="static/thumbs/<scene>.jpg" alt="<scene> preview" loading="lazy">
    <div class="text">
        <div class="title"><Pretty Name></div>
        <div class="meta"><span class="size"><HD MB> MB</span> · mip-360</div>
    </div>
</a>
```

Commit + push from `bitymi-demos/`:

```bash
cd /home/nilkel/Projects/bitymi-demos
git add index.html
git commit -m "Add <slug> demo card"
git push
```

GitHub Pages rebuilds in ~30 s.

---

## Batch sweep (official mip-360 18 pairs)

For the canonical SV + SB run across all 9 mip-360 scenes, use the
batch helpers — see [BITYMI_BUNDLES.md § Batch helpers](BITYMI_BUNDLES.md#batch-helpers).

```bash
python scripts/build_bc7_bundles_fp16.py    # 9 scenes × {SV HD, SB HD} = 18 BC7 bundles
python scripts/build_astc_bundles_fp16.py   # 9 scenes × {SV HD, SV lite, SB HD} = 27 ASTC bundles
```

Both expect `BAKE_ROOT=/mnt/nilkel_hdd/outputs/mip_360` and the
hardcoded SV/SB config names — copy + edit if your training config
differs.

---

## Top pitfalls (cheat sheet)

| symptom | fix |
|---|---|
| Bundle renders black on phone | adapter has ASTC but viewer fetched BC7 — re-check `data-astc-hd`/`data-astc-lite` attrs on the card |
| `max_res` silently clamped 64 → 16 | pass `--atlas_budget_mb 8192` (default 2048 is too small for dense scenes) |
| `cameras.json not found` during pack | copy from a sibling config: `cp <sibling>/cameras.json $MP/` |
| HF upload 401 | `hf auth login` (write-scoped token) |
| Site CORS fails | check the bundle is on HF dataset, NOT GitHub Release (HF CDN sends the right `Access-Control-Allow-Origin`) |

Full pitfall table in [BITYMI_BUNDLES.md § Common pitfalls](BITYMI_BUNDLES.md#common-pitfalls).

---

## Where things live

- Producer scripts → `nest-splatting/scripts/`
- Bake artifacts → `<model_path>/baked_atlas/` + `<model_path>/baked_atlas_lite/`
- Local bundle staging → `bitymi-demos/scenes/`
- HF dataset → [`Nilkel/bitymi-demos`](https://huggingface.co/datasets/Nilkel/bitymi-demos)
- Live page → [`bitymi-demos`](https://nilkel.github.io/bitymi-demos/)
- Live viewer source → `Halloumi-web-splat` (Rust + wgpu → WASM)
- Future viewer source → `Halloumi-WS` (TS + WebGPU); see [HALLOUMI_WS_VIEWER.md](HALLOUMI_WS_VIEWER.md)
