# NEST-Splatting Project Rules

## Environment
- **Conda environment**: `nest_splatting`
- Always use `conda run -n nest_splatting` for Python commands

## Build Commands

**CRITICAL: ALWAYS use `run_in_background: true` for ALL CUDA builds** — they take 2-5 minutes.

| Submodule | Path | Purpose |
|---|---|---|
| `diff-surfel-rasterization` | `submodules/diff-surfel-rasterization` | Original 2DGS rasterizer (modes 0/1) |
| `diff_surfel_3D_sh_res` | `submodules/diff_surfel_3D_sh_res` | **Main training rasterizer** for 3D_SH_res |
| `diff_surfel_mixed` | `submodules/diff_surfel_mixed` | `--method mixed` rasterizer (fork of `diff_surfel_3D_sh_res`) with the per-Gauss textured/untextured CUDA branch (untextured = 2DGS ray-splat, SV-only). |
| `diff_surfel_mixed_3d` | `submodules/diff_surfel_mixed_3d` | `--method mixed_3d` rasterizer (fork of `diff_surfel_mixed`). Untextured surfels render as **EWA 3D ellipsoids** (FastGS-verbatim `computeCov3D`/`computeCov2D`/conic, eps2d=0.3) instead of 2DGS ray-splats. Textured half untouched. |
| `diff_surfel_bake` | `submodules/diff_surfel_bake` | Bakes MLP residual into per-Gaussian SH atlas |
| `diff_surfel_bake_render` | `submodules/diff_surfel_bake_render` | Forward-only renderer for baked atlas |
| `gridencoder` | `gridencoder` | Python-side hash encoding (not used at inference) |

Build any of them with:
```bash
cd <submodule-path> && conda run -n nest_splatting python -m pip install -e . --no-build-isolation
```

**Rebuild triggers**: `.cu`, `.cuh`, `.h` → rebuild. `.py` → no rebuild (editable install).

**Troubleshooting**:
- "externally-managed-environment" → use `python -m pip` (not `pip`)
- "No module named 'torch'" during build → add `--no-build-isolation`

## Architecture: 3D_SH_res

The primary rendering mode. **Color formula**:
```
color = ReLU( ReLU(SH + sh_bias) + MLP_residual + res_bias )      # 3D_SH_res (default, stacked outer ReLU)
color = ReLU(SH + sh_bias) + ReLU(MLP_residual + res_bias)        # 3D_SH_add (separate ReLUs)
```
Defaults: `sh_bias=0.5`, `res_bias=0.0` (configurable at runtime via `set_activation_bias(sh, res)` — no rebuild needed). SH is unbounded; the residual rides on top to add high-frequency detail. Both components are evaluated in CUDA inside `diff_surfel_3D_sh_res`.

**Activation mode** (`d_residual_mode`, set via `set_residual_mode(0|1)` in both training and bake_render extensions):
- `--method 3D_SH_res` (mode 0, default): outer ReLU gates SH and residual together. Residual can subtract from SH up to where the sum hits zero.
- `--method 3D_SH_add` (mode 1): separate ReLUs. Residual can only ADD to SH (since `ReLU(x) ≥ 0`); it cannot subtract. Same architecture as 3D_SH_res — only the activation differs. Stored in `bake_meta.json` as `residual_mode` and re-applied automatically at bake-render time.

**Components**:
- **SH** (Lagrangian, attached to Gaussians): handles low-frequency color + view-dependence
- **Hashgrid + MLP residual** (Eulerian, fixed grid): handles high-frequency spatial detail
- View encoding: 16D (SH degree 4, from tcnn `_encode_view`)

**Hybrid levels** (`--hybrid_levels N`): split between per-Gaussian features and hash levels, with 6 total levels at 4D each:
- `hybrid_levels=5` → 5 per-Gauss (20D) + 1 hash (4D, finest)
- `hybrid_levels=2` → 2 per-Gauss (8D) + 4 hash (16D)

Hash levels are selected **finest-first** from `[128, 203, 322, 512]`. Hash table: 2^19 entries × 4D, voxel range `[-1.5, 1.5]`.

**Coarse-to-fine** (multi-level hash only): coarsest hash level always on, one finer added every 2000 iters. Single-level (hybrid=5) is always fully active.

## 3D_SH_cat (alternative)

`MLP(sum(w_i * f_i))` — accumulate features then evaluate MLP once per pixel. Render mode 1 in `diff-surfel-rasterization`. Same hybrid_levels semantics. Less expressive than 3D_SH_res; kept for comparison.

## Baked Rendering Pipeline

Goal: bake the MLP residual into static per-Gaussian SH textures for fast inference.

**Full reference**: see [`docs/BAKED_RENDERING.md`](../../docs/BAKED_RENDERING.md) — includes BC7 atlas compression, AABB modes (SnugBox+AccuTile), sort modes, importance-based pruning/skip-texture, atlas-width auto-grow, FP16/uint8/BC7 dtype trade-offs, and the 18-pair mip-360 results.

**Deploy a single scene (quick procedure)**: see [`docs/DEPLOY_DEMO.md`](../../docs/DEPLOY_DEMO.md) — short operational summary of the 4-stage bake → nat2 → pack → upload pipeline plus the `index.html` card snippet. Cross-references BITYMI_BUNDLES for depth. **Live viewer is the Rust `Halloumi-web-splat` WASM build** (not the TS Halloumi-WS).

**Deployment / WebGPU viewer (full reference)**: see [`docs/BITYMI_BUNDLES.md`](../../docs/BITYMI_BUNDLES.md) — full bake → `scene.nat2` → `.bitymi` → HF upload pipeline, including BC7 vs. ASTC, HD vs. lite, naming conventions, batch helpers (`build_bc7_bundles_fp16.py`, `build_astc_bundles_fp16.py`), and the single-scene variant template.

**Halloumi-WS viewer (TS/WebGPU "WebSplatter")**: see [`docs/HALLOUMI_WS_VIEWER.md`](../../docs/HALLOUMI_WS_VIEWER.md) — the TypeScript viewer at `/home/nilkel/Projects/Halloumi-WS` (build with `npm run build`, dev with `npm run dev`). Covers the surfel buffer layout (32 B/Gauss), shader pipeline (surfel_cull → preprocess_2dgs → radix sort → tile_raster → display), bundle loader (BITYMI chunks), orbit-pivot logic (ray-disk intersection in `pickGaussAt`), and modifications vs. upstream WebSplatter (2DGS-only, BC7+ASTC, SV/SB color paths). Currently *not* deployed to bitymi-demos — the live viewer is still the Rust `Halloumi-web-splat` build; swap procedure in § 12 of the doc.

**4090 benchmarking**: see [`docs/BENCH_4090.md`](../../docs/BENCH_4090.md) — full procedure to bench a baked model on `neel@10.176.128.69` (SSH key installed). Pipeline: `build_bench_bundle.py` locally → `rsync` to `~/nest-bench/bundles/<name>/` → `ssh ... bash -c '. miniforge3/.../conda.sh && conda activate bench && python bench_minimal.py ...'`. Returns PSNR/SSIM/LPIPS/FPS JSON. Uses cuda.Event timing (GPU-throughput).

**Video → 3DGS training dataset**: see [`docs/VIDEO_TO_SPLAT_PIPELINE.md`](../../docs/VIDEO_TO_SPLAT_PIPELINE.md) — one-shot wrapper `scripts/video_to_dataset.sh <video> <out> [fps]` that runs ffmpeg + COLMAP (via `/home/nilkel/Projects/video-3d-reconstruction-gsplat/colmap_undistorted_sfm_export.sh`) and restructures output into the nest-splatting `--source` layout (`images/`, `images_2/`, `images_4/`, `sparse/0/`). Defaults to CPU SIFT (`--disable_gpu`) since GPU SIFT needs an OpenGL context — set `USE_GPU=1` under `xvfb-run` or with `$DISPLAY` to opt in. Idempotent (each stage skips if its output exists). Sequential matcher (right for video; switch to `--exhaustive` inside the wrapper for unordered photo sets).

**Sherlock cluster (paper-scale A100 runs)**: see [`docs/SHERLOCK_CLUSTER.md`](../../docs/SHERLOCK_CLUSTER.md) — `z0051beu@sherlock01.ainet.local` (institution-internal — VPN required). SLURM scheduler with job arrays; partition `a100-4gpu-40gb`, account `rctcd82061`. Filesystem convention: bulk lives in `~/userdir/` (conda + projects + data), with `~/userdir/Projects/<repo>` mirroring local layout and `~/userdir/Projects/data/<dataset>/` for inputs. Modules: `gcc/13.2.0`, `cuda12.1/toolkit/12.1.0` (A100 sm_80 — do NOT clone 5090's cu128 env, rebuild CUDA extensions against the cluster's CUDA 12.1). Canonical SLURM template: `../beta-splatting/slurm_benchmark_mip360.sh`. Common auth failure mode (VSCode → password prompt loop) and recovery procedure documented in § 1.

**Pipeline**:
1. **Bake** (`diff_surfel_bake`): evaluate MLP at 8×8 UV grid per Gaussian → 48D SH residual per texel
2. **Render** (`diff_surfel_bake_render`): forward-only 2DGS rasterizer, samples atlas via texture lookup
3. **Format**: mean SH stored in PLY; per-Gaussian residual textures (uint8 or BC7) in a packed atlas

**Scripts**:
- `scripts/benchmark_baked.py` — neural vs baked-SH-only vs baked-SH+atlas timings + PSNR/SSIM/LPIPS
- `scripts/render_baked.py` — save baked renders to disk (no benchmarking)

**Production defaults** (mip-360 18-pair mean, max_res=64, BC7 atlas):
| | PSNR | SSIM | LPIPS | FPS | Atlas |
|---|---|---|---|---|---|
| Neural renderer | 26.81 | 0.7811 | 0.2307 | 67.2 | — |
| Baked BC7 | 26.76 | 0.7762 | 0.2487 | 472.8 | 638 MB |
| Δ vs neural | −0.05 | −0.005 | +0.018 | **7.04×** | — |

**Reference quality** (chair scene, legacy 48D residual format):
| Mode | PSNR | SSIM |
|---|---|---|
| Neural renderer (training) | 34.66 | 0.9844 |
| SH + 48D residual (baked) | 34.01 | 0.9814 |
| SH only (baked mean) | 33.23 | 0.9757 |

**Pitfalls** (all bitten in past sessions):
- **Scene() overwrites baked PLY**: `Scene()` constructor loads training PLY. Always reload baked PLY after `Scene()` init.
- **UV conventions**: bake kernel must use texel-center `(i+0.5)*step - extent`, NOT endpoint-inclusive `/(N-1)`.
- **SH layout**: MLP outputs channel-first `[R0..R15, G0..G15, B0..B15]`; PLY stores interleaved `[N, 16, 3]`.
- **Default budget clamps resolution**: `--atlas_budget_mb 2048` (default) silently clamps `max_res` from 128 down to 16 on large scenes. Bump to 8192 for full-res bakes (`--max_res 64 --atlas_budget_mb 8192` is the path-2 standard).

## Key Files

- `gaussian_renderer/__init__.py` — `render()` dispatch
- `hash_encoder/modules.py` — INGP class (hash + MLPs), tcnn weight loading
- `scene/gaussian_model.py` — `GaussianModel`
- `train.py` — training loop
- `submodules/diff_surfel_3D_sh_res/cuda_rasterizer/{forward,backward}.cu` — main CUDA path
- `submodules/diff_surfel_bake_render/cuda_rasterizer/forward.cu` — bake render forward

## CLI Flags

### Activation & Bias
- `--activation_bias SH RES` — defaults `0.5 0.0`
- `--freeze_sh` — permanently zero SH LR
- `--sh_freeze_iter N` — freeze SH LR for first N iters
- `--freeze_mlp` — freeze MLP weights

### Hash/MLP LRs
- `--res_lr_scale X` — scale hash + MLP LR
- `--hash_lr_scale X` — scale hash LR only (stacks)
- `--res_warmup N` — disable hash/MLP residual for first N iters

### Regularization
- `--overdraw_reg λ` — sigmoid-based overdraw penalty (CUDA grad)
- `--w_overdraw_reg λ` (`--w_overdraw_gamma γ`) — error-guided overdraw
- `--weight_reg λ` — weight-squared consolidation, penalizes `1−sum(w²)` (CUDA grad)
- `--w_weight_reg λ` (`--w_weight_gamma γ`) — error-guided weight-squared
- `--w_normal λ` (`--w_normal_gamma γ`) — error-guided normal consistency
- `--contribution_thresh T` — skip hash query when `α·T < T`
- `--count_thresh N` — skip hash query past N contributors per pixel

**Error-guided pattern**: `w(r) = exp(-γ * MSE(r).detach())`. Critical: `.detach()` the error, otherwise the optimizer sabotages rendering quality to lower reg loss. Pick γ so `exp(-γ * avg_MSE) ≈ 0.5` mid-training.

### MCMC
- `--mcmc_fps`, `--cap_max N`, `--noise_lr X` (default 5e5)
- `--mcmc_depth_reinit N`, `--reinit_interval N`, `--reinit_end N`
- `--opacity_reg λ` (auto-set to 0.01 under `--mcmc`), `--scale_reg λ`

### Training schedules
- `--mini` — Mini-Splatting v2: aggressive clone (every 250) + densification (every 100), depth reinit at 2000, simp1 at 3000 (keep 60%), simp2 at 8000 (keep top 99%).
- `--mini1` — Mini-Splatting v1: repeated depth reinit (5k/10k/15k), longer densification (500→15000), simp1 at 15000 (keep 50%), simp2 at 20000. Auto-enables `--mini`.
- `--minimc` — Sweep-based RJ-MCMC. Requires `--method 3D_SH_res`. Two events on separate intervals: depth reinit (every `--minimc_reinit_interval`, in-place via `reinitial_alive_inplace`) and sweep+cull+clone (every `--minimc_relocate_interval`). Tensor size is fixed; alive/dead split moves inside `cap_max`. See `gaussian_model.py` for `minimc_sweep_and_relocate`.
- `--minispa` — chains mini v2 aggressive clone + silhouette-aware depth reinit at 2000 → GSpa Phase-2 ADMM (3000→25000) → hard prune.

### `--method mixed` (textured/untextured split)
- `--method mixed` — textured/untextured manifold split. Routes through the
  **`diff_surfel_mixed`** rasterizer (a fork of `diff_surfel_3D_sh_res` with the per-Gauss
  branch). Per-Gauss `_is_textured` bool tensor lives on `GaussianModel`; pre-split it's
  all-True (or absent → nullptr in CUDA) and behavior is bit-identical to `--method 3D_SH_res`.
  - **Textured** surfels: full 3D_SH_res — `ReLU(SV) + residual` (hash query + MLP), run's
    chosen `--kernel`, full geometry + shape gradients.
  - **Untextured** surfels: identical 2DGS ray-splat geometry + kernel, but the color path
    is SV-only (`feat = ReLU(SV)`, no hash query, no MLP, no world-xyz reconstruction, no
    MODE-5 GEMMs). Matches the reference `../2d-gaussian-splatting` lean path.
  - Residual activation is **mode 2** (`args._residual_mode = 2`, `set_residual_mode(2)`):
    `feat = ReLU(SV) + residual` (signed residual, NO per-Gauss outer ReLU); the per-pixel
    ReLU on the final blended image is applied in Python (`torch.relu`, autograd-gated).
- `--texsplit N` — iteration at which to split (default `-1` = disabled). At `N`,
  `split_at_texsplit()` **duplicates** the live surfel set (no depth reinit — that shocked
  a converged scene): textured copy (`_is_textured=True`) + untextured copy
  (`_is_textured=False`), all per-Gauss tensors identical. Opacity = each surfel's trained
  alpha scaled by `opacity_scale=0.5` (both copies); SV color untouched (`halve_baseline=False`).
  Adam rebuilt via `training_setup(opt)`. PLY round-trip preserves `_is_textured` (column
  only written when ≥1 untextured row, so older readers stay happy).
- One-way inheritance via densification: textured parents → textured children; untextured
  parents → untextured children. Mass conservation.
- `--freeze_hash_iter`/`--freeze_hash_period` only throttle the **textured** hash/MLP path;
  untextured surfels never touch hash/MLP and keep training every iter (decoupled).
- tqdm shows `Points=<N>/<pct>%tex`; `training_log.txt` records the textured/untextured split.
- **Baked pipeline support** (`benchmark_baked.py` / `render_baked.py` / `diff_surfel_bake_render`):
  untextured surfels are folded into the **skip-texture** set in `bake_atlas` → zero atlas
  rect → the bake-render kernel renders them SV-only (no atlas lookup, no residual) at no
  atlas cost. `bake_meta.json` carries `residual_mode=2` (+ `mixed_textured`/`mixed_untextured`
  counts); the kernel's `d_residual_mode==2` branch leaves the per-Gauss color signed and
  the Python side re-applies the per-pixel `torch.relu` (`final_relu`). `_is_textured` is
  kept aligned through bake-time pruning. No `is_textured` plumbing needed in the bake-render
  CUDA kernel — the zero-rect geometry carries the signal.

### `--method mixed_3d` (untextured = EWA 3D ellipsoids)
- `--method mixed_3d` — same textured/untextured split as `--method mixed`, but
  **untextured surfels render as 3D ellipsoids via EWA splatting** instead of
  flat 2DGS ray-splats. Routes through the **`diff_surfel_mixed_3d`** submodule
  (a fork of `diff_surfel_mixed`). `--method mixed` is unchanged and separate.
  - **Textured** half: byte-identical to `--method mixed` (compute_transmat,
    hash+MLP, residual_mode 2).
  - **Untextured** half: a learnable 3rd scale axis `_scaling_z` (model-side,
    `get_scaling_z = exp(_scaling_z)`, flattened init `log(0.05)+min(log sx,log sy)`
    at split). The CUDA preprocess builds the world covariance from
    `(sx, sy, sz)`+quat and projects it with **FastGS-verbatim** `computeCov3D`
    / `computeCov2D` (quat layout (r,x,y,z), no quat normalization, eps2d=0.3
    low-pass), inverts to a per-Gauss conic stored in `GeometryState.ewa_conic`,
    radius `ceil(3·√maxλ)`. The render kernel computes the Mahalanobis
    `m = a·dx² + 2b·dx·dy + c·dy²` and applies the run's `--kernel` (just like
    the 2DGS path): `beta`/`beta_scaled` (kernel_type 1/4) → restricted-beta
    `α = min(.99, opa·max(0,1−m/k²)^β)` (k²=9 for beta_scaled, compact support,
    β = activated `get_shape` ∈ [0,5]); any other kernel → FastGS Gaussian
    `α = min(.99, opa·exp(−0.5·m))`. SV/SH baseline color (no hash/MLP/Tu/Tv/Tw).
    Verified vs analytic fields to FP16-rgb noise: Gaussian rel ~0.02%,
    beta_scaled max|Δ|~1.7e-4 with compact support confirmed (0 beyond 3σ).
- **`--l2`** / **`--l1`** (mutually exclusive) — `mixed_3d[_sep]` only:
  **per-Gauss** photometric-loss routing in the rasterizer backward. *Not*
  per-pixel attribution. The rasterizer Function returns two image-output
  slots (numerically identical, separate autograd nodes); Python wires
  `(1-λ_dssim)·L1(slot_0, gt) + λ_dssim·(1-SSIM(slot_0, gt))` to slot 0
  (`image_tex`) and the untex loss to slot 1 (`image_untex`):
  `--l2` → `mean((slot_1 - gt)²)`, `--l1` → `mean(|slot_1 - gt|)`. `--l1`
  is the right choice when the untex/EWA half is modelling smooth
  volumetric background where L2 over-penalises edges.
  The Function's backward receives two upstream image gradients and the
  CUDA kernel routes them per Gauss inside `renderCUDAsurfelBackward`:
  textured Gauss accumulate `dL/dxyz`, `dL/dscale`, `dL/dopacity`, `dL/dfeat`
  using `dL_dpixels` (slot 0's grad); untextured Gauss accumulate using
  `dL_dpixels_untex` (slot 1's grad). One forward, one backward — no
  double-render, no detach trickery, no per-pixel `max_contrib_idx`
  attribution. Mechanism (CUDA, in-place mode):
  - Kernel signature gains an optional `const float* dL_dpixels_untex`
    threaded through `BACKWARD::render` → `Rasterizer::backward` →
    `RasterizeGaussiansBackwardCUDA`. Empty tensor at the binding → nullptr
    in CUDA → kernel reverts to single-loss behavior (byte-identical to
    pre-flag).
  - At the per-pixel load, both `dL_dpixel_tex[C]` and `dL_dpixel_untex[C]`
    are populated (untex mirrors tex when the second array is null).
  - After `tex_j`/`tex_bw` is read per Gauss in the inner loop (both std and
    MODE-5 collab paths), a local `const float* const dL_dpixel = tex_j ? dL_dpixel_tex : dL_dpixel_untex;`
    shadows the outer arrays. All downstream `dL_dpixel[ch]` references
    resolve to the routed pointer without any further code changes.
  - Python detects the no-L2 case by checking `grad_out_color_untex.abs().sum() == 0`
    (PyTorch passes a zeros tensor for unused autograd outputs, not None),
    and passes an empty tensor → nullptr to bypass routing.
  - Hash weights and MLP only flow textured signal (they're never queried
    for untextured Gauss in the existing kernel branches). Geometry that
    affects the alpha cascade (xyz/scale/rot/opacity) receives the routed
    per-Gauss image gradient at every pixel where that Gauss contributed.
  Pre-`--texsplit` (no untextured rows) → kernel's all-textured fast path →
  L2 leg contributes nothing → byte-identical to default L1+SSIM loss.
  Fails loudly at setup if paired with a non-mixed_3d method.

- **`--kernel2 <kernel>`** — `mixed_3d` only: overrides `--kernel` for the
  UNTEXTURED (EWA) half *only*; the textured half always uses `--kernel`.
  Unset ⇒ untextured use `--kernel` (byte-identical to before). Canonical use:
  `--kernel beta_scaled --kernel2 gaussian` → textured = 2D beta_scaled
  surfels, untextured = Gaussian EWA ellipsoids. **Mechanism (no signature
  plumbing):** the renderer packs `(kernel_type2+1)` into **`render_mode` bits
  [16..19]** (mixed_3d-gated; zero nibble = unset); the 3 untextured-EWA CUDA
  branches (fwd-render, bwd-std, bwd-MODE-5) decode
  `ut_kt = nibble ? nibble−1 : kernel_type` and switch beta↔Gaussian on it
  (textured stays on `kernel_type`). Bake path has no `render_mode`, so it
  mirrors the `set_residual_mode` device-global idiom: `set_untex_kernel(int)`
  → `d_untex_kernel` (-1=unset); `benchmark_baked.py` records
  `bake_meta["kernel2"]` at bake and calls `set_untex_kernel` at render so
  baked geometry matches training. **Verified:** forward decode bit-identical
  both directions (`kt=4 +k2=gaussian == kt=0`; reverse `kt=0 +k2=beta == kt=4`;
  unset == `kernel_type`); backward gradcheck exact with the k2 bit (std +
  MODE-5 GEMM): means3D/opac/scales/scaling_z/rot/sh relmax ≤1e-2, cos=1.0.
  Constraint: `--kernel2 beta*` needs `--kernel` also beta-family (per-Gauss
  `_shape` is only passed when `pc.kernel_type` is beta-family); the documented
  `beta_scaled`/`gaussian` combo is unaffected.
- Plumbing: `scaling_z` (activated `pc.get_scaling_z`) flows
  renderer → wrapper → binding → `Rasterizer::forward` → `FORWARD::preprocess`.
  The render kernel only takes the EWA branch when **`scaling_z != nullptr`**
  (rasterizer_impl gates `ewa_conic` to nullptr otherwise). So pre-`--texsplit`
  (scaling_z empty) `mixed_3d` is byte-identical to `--method mixed`/`3D_SH_res`.
- Split: `split_at_texsplit(make_scaling_z=True)` creates `_scaling_z` (it's
  built only for `mixed_3d`; `mixed` leaves it empty).
- **Backward: implemented + gradcheck-verified.** Untextured EWA VJP is the
  FastGS-verbatim port (`ewa_backward_vjp` in `backward.cu`: conic→cov2D→cov3D
  →scale/`scaling_z`/quat + proj→mean3D) wired through a self-contained branch
  in the std render-backward (mirrors the per-pixel color/depth/alpha/normal/
  dist/reg recurrence so cross-set occlusion grads stay correct), then
  `preprocessCUDA`'s untextured-EWA branch. `dL_dscaling_z` is a new grad
  output plumbed binding→`Rasterizer::backward`→`__init__.py` (the `scaling_z`
  autograd slot). Numerical gradcheck (analytic vs central FD) is **exact**
  (relmax ≤ 1e-2, cos = 1.0) for means3D, opacity, scales(sx,sy), `scaling_z`,
  rotations, sh, and β-shape — both `--kernel` Gaussian (kt 0) and restricted
  beta_scaled (kt 4). Implementation notes / gotchas:
  - The conic-grad carrier reuses `dL_dtransMat[gid*9+0..2]` (untextured rows
    never use transMat). Its **off-diagonal (b) channel drops the factor 2**
    (`dL_dm·dx·dy`, not `2·dx·dy`) to match FastGS `computeCov2DCUDA`'s half
    convention; the `dL_dmean2D` screen-grad keeps the full `2·dx·dy`.
  - **Collaborative GEMM is ENABLED** (same as `mixed`/`3D_SH_res`). The
    MODE-5 path has a block-uniform untextured-EWA branch (mirrors the std
    EWA branch; `tex_j`+`ewa_conic` are per-Gauss uniform across all 256
    threads → the whole block handles the untextured Gaussian and `continue`s,
    uniformly skipping the GEMM — untextured has no MLP). Per-thread gating
    uses `ok` flags, NOT `continue`, so no thread diverges before the
    block-uniform `continue` (the next j iterates a `__syncthreads_count`
    barrier). Verified: MODE-5 backward == std backward to float precision
    (max|Δ|~1e-6, cos=1.0) on a mixed textured+untextured scene with the GEMM
    exercised (nonzero MLP+hash) → cross-set recurrence handoff is correct and
    textured surfels keep the tensor-core MLP backward (no speed regression).
  - `preprocessCUDA` skips the 2DGS transMat VJP + the densify depth-hack for
    untextured-EWA rows (keeps `computeColorFromSH` → dL_dsh); the EWA
    `dL_dmean2D` from the render backward is the densification proxy.
  - Depth-map / normal-map gradients are not backpropagated to untextured
    geometry (simple-splat layer, normal≡0); the recurrence STATE is still
    advanced so textured neighbours' depth/dist/normal grads stay correct.
  - `beta_scaled` central-FD gradcheck degrades near the hard compact-support
    edge (`m≥k²` cull flips edge pixels) — a FD-probe artifact, not a VJP
    error (cos stays ≥0.98; β=2 + adequately-sized splats → relmax ≤1e-4).
- **Baked pipeline: EWA-aware (forward-only port).** `diff_surfel_bake_render`
  now renders untextured surfels as EWA 3D ellipsoids: the verified
  `compute_ewa_conic` device fns + a conditional-geometry `preprocessCUDA`
  (`if(!untex_ewa)` 2DGS / `else` EWA conic → rect-AABB binning, new
  `GeometryState.ewa_conic`) + a self-contained untextured-EWA branch in
  `renderBakedCUDA` (conic falloff + the **shared** SV/SH-baseline colour path
  — untextured carry a zero atlas rect so colour is byte-identical to
  `--method mixed` skip-texture; only geometry differs). `is_textured`/
  `scaling_z` plumbed through forward.h/rasterizer.h/rasterizer_impl/
  rasterize_points/`__init__.py`/`prepare_gaussian_inputs`; `benchmark_baked.py`
  passes them and keeps `_scaling_z` in **both** bake-time prune lists (next to
  `_is_textured`) so the PLY round-trip stays aligned. `diff_surfel_bake`
  (atlas baker) unchanged — untextured fold into skip-texture (zero atlas
  tiles); the residual is an xyz→INGP-MLP function, geometry-independent.
  **`load_ply` fix**: the `scale_` filter excluded `scale_z` (it shared the
  prefix with 2DGS `scale_0/1` → `int('z')` crash; latent because `--cold`
  never reloads a PLY). **Verified**: garden `mixed_3d` (35k) baked BC7 =
  **26.34 dB / 0.776 / 0.208 @ 1157 FPS vs neural 26.61 / 0.819 / 0.160 @
  111 FPS — 10.4× speedup, −0.27 dB** (same bake-quant band as `--method
  mixed`: −0.32 dB), confirming the EWA geometry bakes correctly.

### Other
- `--init_ply PATH` — initialize Gaussians from external PLY
- `--cold` — skip 2DGS warmup, train from scratch with hash-in-CUDA from iter 1
- `--bce_solo_adaptive --bce_iter N` — opacity binarization in last N iters

## Hybrid Architecture Insights

**Lagrangian vs Eulerian clash**: any operation that destroys/recreates Gaussians (depth reinit, aggressive prune) breaks the hash's spatial correspondence, while SH just relearns locally. Mini-Splatting depth reinit is designed for pure 3DGS; under hybrid, the hash loses learned spatial features. Use higher reinit opacity (0.5 vs 0.1) to keep hash gradients alive during structural changes.

**Hash gradient magnitude**: `dL/dfeat ∝ T_i · α_i`. Tiny by default (~1e-5 norm for 2M params). Adam normalizes per-parameter so updates are still ~lr-sized. With 1 hash level (4D), the hash has limited capacity — MLP does most of the work.

**MCMC noise on 2DGS surfels**: standard MCMC builds 3D covariance with `[scale_x, scale_y, 1]`. For tiny surfels (`scale=0.01`), normal noise=1 is 100× larger than in-plane noise → blasts surfels off the surface. Use `max(scale_x, scale_y)` for normal axis, or 0 for pure in-plane jitter.

**SH +0.5 bias note**: `diff_surfel_3D_sh_res` keeps the +0.5 bias (configurable at runtime). Removing it forces SH to start at black, giving hash equal footing — but typically converges slower without explicit upside.

## `out_index` / `max_contrib_idx`

Per-pixel max-contributor Gaussian id, written by `diff_surfel_3D_sh_res` only.
- CUDA: in `forward.cu`, tracked alongside `depth_max_contributor` via `if (w > max_w)`. Both WMMA and scalar paths populate.
- Python: `_RasterizeGaussians.forward` returns a 9-tuple, exposed as `render_pkg['max_contrib_idx']`. Other rasterizers return `None`.
- Consumers: `mini_depth_reinit` SH transfer (looks up `src_features_dc/_rest[max_idx]` per sampled pixel; falls back to `RGB2SH(GT_pixel)` when invalid), debug visualization (`_colorize_max_contrib_idx`).

## Known Issues & Pitfalls

- **`_appearance_level = 0` silently disables hash**: `hashgrid.h` does `max_level = min(ap_level, L)`. `ap_level=0` → 0 hash levels queried, hash features all zero, hash grads exactly zero. `create_from_pcd` correctly sets `ap_level=24` (sentinel for "all levels active"). Any code that creates/reinitializes Gaussians (`reinitial_from_depth`, prune+realloc) MUST set `_appearance_level=24`.

- **MODE 5 vs standard backward**: `diff_surfel_3D_sh_res` has separate code for the collaborative-GEMM path (mode 5) and the standard backward. When adding kernel features (e.g. beta kernel), patch BOTH — alpha computation, `dL_dshapes`, `dG_factor` (rho3d), `dG_factor_2d` (rho2d). Past silent bug: zero gradients for beta kernel under mode 5.

- **tcnn weight layout**: tcnn pads input dim to multiples of 16 (`40→48`); padding columns are filled with 1s, giving an implicit bias `b1 = sum(W1[:, 40:48], dim=1)`. tcnn has no explicit biases → zero `b2`, `b3`. Handled by `_copy_tcnn_to_pytorch_mlp()` in `hash_encoder/modules.py`.

- **Renderer level encoding**: `gaussian_renderer/__init__.py` uses `ingp.hashgrid_levels` (not `ingp.levels`) for `(total << 16) | (active << 8) | hybrid`. Wrong field → CUDA kernel queries non-existent levels → zeros/garbage.

- **Don't reset opacity under `--mini`/`--mini1`**: MSv2 never resets opacity; doing so destabilizes the importance-prune pipeline.

## Config

`configs/nerfsyn.yaml`: `lambda_normal=0`, `lambda_dist=0`, `lambda_mask=0.1`, `mask_iter=10k`. `tg_beta`, `tg_base_alpha` for surfel parameters.
