# `--ppisp` — Photometric (ISP) compensation

Wraps NVIDIA's **PPISP** (Deutsch et al. 2026, [arXiv:2601.18336](https://arxiv.org/abs/2601.18336),
clone at `../ppisp`) as a differentiable ISP layer applied to the **rendered image**
before the photometric loss, trained jointly with the Gaussians.

It is **not** a data preprocessor and **not** a post-process on a finished model.
Nothing about the training images or the trained splats is modified — the ISP is a
separate ~2 KB parameter set that *absorbs* the photometric nuisances a real capture
carries, so the radiance field no longer has to explain them with floaters and
view-dependent SH abuse.

```
render() → image [3,H,W] ──► PPISP ──► random-bg composite ──► L1 / SSIM / … vs GT
                               ▲
                     per-frame + per-camera params (own Adam, own LR schedule)
```

## The chain

Applied in this order; **identity at init** (verified to 1e-5 against the repo's own
`tests/torch_reference.py`), so `--ppisp` off is byte-identical to before.

| Stage | Scope | Params | What it models |
|---|---|---|---|
| Exposure | per **frame** | 1 (`2^e`) | auto-exposure drift between shots |
| Vignetting | per **camera**, per channel | 3×5 (radial poly, 3 α terms + learnable optical centre) | lens falloff |
| Colour | per **frame** | 8 (ZCA latents → chromaticity homography) | white-balance / colour-cast drift |
| CRF | per **camera**, per channel | 3×4 (toe / shoulder / gamma / centre) | the camera's fixed tone curve |

The colour homography works in (R, G, Intensity) space and renormalises intensity
afterwards, so it **cannot** change brightness — that is exposure's job. The two are
deliberately decoupled.

For a 300-image scene: `300 + 15 + 2400 + 12` ≈ **2.7 K floats**.

## Mapping onto our data

One physical lens per scene ⇒ `num_cameras = 1` (vignetting and CRF are scene-global),
`num_frames = len(scene.getTrainCameras())`, keyed by `image_name → contiguous index`
(scale-agnostic — a downsampled camera carries the same identity). Same idiom as
`--3rgs`.

## Two deliberate deviations from the library defaults

Both are about **what the exported scene should look like**, since our end product is a
baked `.bitymi` bundle rendered by a viewer that has no ISP stage.

1. **CRF frozen at identity** unless `--ppisp_crf`. Our captures come from a single ISP
   whose tone curve is *already in the GT* and which we *want* the splats to reproduce.
   Letting it train leaves the splats in pre-CRF space, and they render wrong in
   Halloumi-WS. Per-camera CRF only earns its keep on multi-camera rigs whose ISPs differ.
2. **Controller off** unless `--ppisp_controller`. The controller is a per-camera CNN that
   predicts per-frame exposure/colour *from the rendered radiance*, so held-out views get a
   fitted correction. It requires freezing the scene at 80 % of training. With it off,
   novel views get zero per-frame correction = the canonical appearance, which is both the
   fair evaluation and exactly what we want to bake.

## The `[0,1]` clamp trap

The kernel does `rgb.clamp(0,1)` before the CRF, **gated on `camera_idx != -1`** — so it
fires whenever the per-camera path is on, even with the CRF frozen at identity. Our SH is
unbounded (`ReLU(ReLU(SH+0.5) + residual)`), and measured directly:

```
cam=0    (vig+CRF, clamp)   |grad| above 1.0 = 1.0e-10   below = 2.2e-04
cam=None (exposure+colour)  |grad| above 1.0 = 6.6e-04   below = 2.1e-04
```

A pixel that overshoots past 1 receives **exactly zero gradient** and can never come back
down. Two mitigations, both wired:

- `--ppisp_overflow_w` (default **0.01**) adds `w · relu(render − 1).mean()` on the
  **pre**-ISP render, restoring an explicit downward push. Set 0 to disable.
- `--ppisp_no_camera` drops the per-camera stages entirely (no vignetting, no CRF, **no
  clamp**), leaving exposure + colour only. Use it if bright regions still stall.

## Ordering inside the training loop

PPISP is applied immediately after `render_pkg["render"]` is unpacked, **before** the
`--random_background` composite. That ordering matters: the composite adds the same *raw*
background to both `image` and `gt_image`, so keeping the ISP upstream means it never sees
— and never tries to explain — the synthetic background. Everything downstream
(`error_img`, L1/SSIM/LPIPS, the error-guided reg weights, the debug dumps) then operates
on the ISP-corrected image, which is what GT is compared against.

## Evaluation

`training_report` applies the ISP through the same helper, which resolves the frame index
by `image_name`. Train cams get their own fitted exposure/colour; **test cams fall through
to `frame_idx = -1` = zero per-frame correction**. No test GT is consulted either way, so
held-out PSNR stays comparable to a non-PPISP run.

Caveat worth stating in any comparison: if a scene has genuine exposure drift, held-out
PSNR is *penalised* under this protocol, because the held-out GT carries an exposure the
canonical render deliberately does not reproduce. The win shows up in the reconstruction
(fewer floaters, cleaner corners, fewer Gaussians), not necessarily in the test number.
Turn on `--ppisp_controller` if you want the fitted-correction protocol instead.

## Output

`point_cloud/iteration_<N>/ppisp.pt` — `state_dict` + `name_to_idx` + the two mode flags.
Kept **out** of the PLY on purpose: the exported splats are the ISP-free canonical scene.
The file is only needed to reproduce a specific training frame's appearance.

Progress is logged every 500 iters:

```
[PPISP iter=5000] exposure(stops): mean=+0.0021 std=0.0847 min=-0.213 max=+0.198 | colour |c|max=0.0412 | vig alpha=[...] centre=(+0.0021,-0.0009)
```

`std` of the exposure is the useful number — it is how many stops of drift the capture
actually had. Near-zero ⇒ the capture was photometrically clean and PPISP has nothing to do.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--ppisp` | off | enable |
| `--ppisp_lr` | 0.002 | Adam LR (paper value); uses PPISP's linear-warmup → exp-decay schedule |
| `--ppisp_crf` | off | let the per-camera tone curve train |
| `--ppisp_no_camera` | off | exposure + colour only (drops vignetting, CRF, and the clamp) |
| `--ppisp_controller` | off | train the CNN controller for held-out per-frame corrections |
| `--ppisp_overflow_w` | 0.01 | over-range penalty weight; 0 = off |

## Build

```bash
cd ../ppisp && conda run -n nest_splatting python -m pip install . --no-build-isolation
```

Self-contained CUDA extension (`ppisp_cuda`), builds in ~1 min. Compiled with
`--use_fast_math`, hence the ~1e-5 agreement with the torch reference rather than exact.

## Expected gain

- **mip-360 / benchmark scenes**: small. Exposure is near-locked; most of the gain would be
  vignetting on the outdoor scenes.
- **Handheld / phone video captures** (`scripts/video_to_dataset.sh`, bitymi demo scenes):
  this is the target. Continuous auto-exposure and WB drift is exactly what 3DGS explains
  with semi-transparent floaters.
- **Micro-CT / HIMALAYA volumes**: irrelevant, no camera ISP.

## Finetuning a PPISP-trained run (`--finetune_from`)

**The ISP must come along, or the finetune actively destroys what the PPISP run bought.**
The trained Gaussians are the *canonical* scene: they satisfy `PPISP(render) ≈ GT`, not
`render ≈ GT`. Resume with an identity ISP and the loss compares the canonical render
against raw GT, so the Gaussians re-bake the vignetting into the corners and the SH
re-absorbs per-frame exposure as view-dependent junk. Those are large, low-frequency,
trivially-fittable errors, so it happens within a few hundred iterations — and it looks
like the finetune is *working*, because training loss falls the whole time.

`--finetune_from <MODEL_DIR>` (+ optional `--finetune_iter N`, default = highest saved)
reloads the three things that define a trained `3D_SH_res` scene:

| Artifact | Path | Restores |
|---|---|---|
| PLY | `point_cloud/iteration_<N>/point_cloud.ply` | surfels, SV (`sv_col/sv_site/sv_tau`), SH, opacity, scale, rot, shape |
| INGP | `ngp_<N>.pth` | hash table + residual MLP |
| ISP | `point_cloud/iteration_<N>/ppisp.pt` | exposure, colour, vignetting, CRF |

Optimizer state is deliberately **not** restored — fresh Adam, fresh LR schedule.

Three things it handles that a naive reload would get wrong:

1. **The ISP is re-keyed by `image_name`, not index.** A different `--eval` split, `-i`
   resolution or image subset shifts frame ordering. Frames the source run never saw keep
   their identity init (= canonical), which is the correct default. The log reports
   `<hit>/<total> frames matched`; a 0 there means the datasets don't correspond.
2. **`--ppisp_crf` / `--ppisp_no_camera` must match the source run** — they change the chain
   itself, so a mismatch means the reloaded Gaussians are in the wrong space. Asserted, fatal.
   (`--ppisp_controller` is *not* asserted; toggling it off after a controller run will fail
   loudly inside `load_state_dict` on unexpected keys.)
3. A missing `ngp_<N>.pth` is loud but non-fatal — without it the residual restarts random
   while the reloaded SH base is left explaining detail it was never fit for.

**Watch the position LR.** `first_iter` is 0 on a finetune, so the three scheduled LRs
(`xyz`, `sv_sites`, `sv_tau`) restart at the top of their schedules. `position_lr_init` is
1.6e-4 decaying to 1.6e-6 over 30 k, so a run finetuned from iteration 35 000 restarts with
a **100× higher** position LR than it ended with, and the geometry gets kicked. Pass
`--position_lr_init 0.0000016` to pin it. This also matters for correctness when pairing
with `--3rgs` or `--deform`: free positions compete with the pose delta / deformation field
to explain the same per-frame misalignment, which is the degenerate coupling Höllein et al.
avoid by freezing positions.

## Not wired

- The baked pipeline (`benchmark_baked.py` / `diff_surfel_bake_render`) has no PPISP stage,
  by design — bake the canonical scene. If `--ppisp_crf` was used, the bake will not match
  the training renders.
- `--start_checkpoint` still does not reload `ppisp.pt` (or the hash/MLP — [train.py:1365]
  is the only INGP restore and it is gated on the shared-ckpt path). Use `--finetune_from`.
