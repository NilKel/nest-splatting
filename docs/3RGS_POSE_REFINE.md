# `--3rgs` — Camera Pose Refinement (3R-GS "sfm" core)

Jointly optimizes the camera **extrinsics** alongside the Gaussians, correcting
COLMAP pose error during training. This is the directly-transferable core of
**3R-GS** (Huang et al. 2025, [arXiv:2504.04294](https://arxiv.org/abs/2504.04294),
local clone `../3rgs`): the per-camera learnable pose delta (`CameraOptModule`,
*sfm* mode). The MLP-pose variant and the global epipolar loss are **not** ported —
they need MASt3R-SfM correspondences we don't have for COLMAP scenes.

## Why it isn't a direct copy

3R-GS is built on **gsplat**, whose CUDA backward returns a gradient w.r.t. the
camera `viewmats`. Every rasterizer in this repo (`diff_surfel_3D_sh_res`,
`diff_surfel_mixed_3d`, …) treats the camera (`viewmatrix`/`projmatrix`/`campos`)
as a **constant** — the autograd `backward` emits grads for
`means3D / means2D / sh / scales / rotations / opacities` but **nothing for the
pose** (confirmed at [`diff_surfel_3D_sh_res/__init__.py`](../submodules/diff_surfel_3D_sh_res/diff_surfel_3D_sh_res/__init__.py) `backward`).

So we route the pose gradient through the geometry the kernel *does*
differentiate. For a per-camera rigid pose delta `Td` right-multiplied onto
camera-to-world (exactly 3R-GS: `C2W' = C2W @ Td`), rendering the scene through
the **original** camera `W2V` but with every Gaussian rigidly transformed by

```
M    = C2W · inv(Td) · W2V          # world-frame rigid transform
x'   = M_rot · x + M_t              # Gaussian centers
q'   = quat(M_rot) ⊗ q              # surfel orientations
```

is mathematically identical to moving the camera by `Td` (derivation: we want
`W2V·x' = W2V'·x = inv(Td)·W2V·x  ⇒  x' = C2W·inv(Td)·W2V·x`). `Td` is
**zero-init** ⇒ `M = I` ⇒ the iter-0 render is byte-identical to no pose opt.
The gradient flows to the per-camera 9D delta (3 translation + 6D rotation) via
`dL/dmeans3D` + `dL/drotations`. **No CUDA rebuild** — pure Python; works for
**every method** (`3D_SH_res`, `res_3d_paired`, `mixed`, `mixed_3d`, …).

The Gaussians stay in the canonical world frame; only the camera moves, so they
converge to true multi-view-consistent geometry while the delta absorbs the pose
error. Both receive gradient from the same forward (joint optimization), exactly
as 3R-GS.

## Files

- `scene/camera_pose_opt.py` — `CameraPoseOpt` (per-camera 9D embedding,
  zero-init), `rotation_6d_to_matrix` / `matrix_to_quaternion` /
  `quaternion_multiply` (differentiable, `[w,x,y,z]` matching `build_rotation`),
  `correction()` (returns `M_rot, M_t, q_M`), `refined_world2cam()`,
  `export_refined_poses()`.
- `gaussian_renderer/__init__.py` — new `pose_correction=(M_rot, M_t, q_M)` arg.
  Applied to `means3D` right after `means3D = pc.get_xyz` (so the SV/beta/hash
  view-dir prep and the rasterizer are all consistent) and to `rotations` right
  after `rotations = pc.get_rotation`. `None` ⇒ identity (no-op).
- `train.py` — flags, the separate Adam optimizer, the `_pose_correction_for(cam)`
  closure (returns `None` before warmup), the gated optimizer step, and the
  refined-pose export at every save.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--3rgs` | off | Enable per-camera pose refinement (`dest=pose_refine`). |
| `--3rgs_lr` | `1e-5` | Adam LR for the 9D pose delta (3R-GS sfm default). |
| `--3rgs_warmup` | `500` | Iters before pose opt starts (let geometry settle — important under `--cold`). The learned correction is applied from this iter onward and **never reverted**. |
| `--3rgs_until` | `-1` (= total iters) | Last iter the delta is **stepped**; after this it's frozen but still applied. |
| `--3rgs_reg` | `0.0` | Weight decay on the delta embedding (keeps corrections small). |

Notes:
- **Orthogonal to densification** (`--fastgs` / `--mcmc`) and to the
  `res_3d_paired` mode flips/splits: the pose params are per-camera (fixed count)
  in a **separate** optimizer, untouched by Gaussian tensor rebuilds.
- The delta is keyed by `image_name` → contiguous index, so it's resolution-scale
  agnostic (a downsampled camera carries the same extrinsics).
- Surfel rotations are composed via `matrix_to_quaternion(M_rot)`; `M_rot ≈ I`
  throughout (deltas are tiny), so the trace branch is selected and it's stable.

## Usage

Append `--3rgs` to a normal training command:

```bash
python train.py -s data/personal/Max -m <out> --yaml ./configs/max.yaml \
  --iterations 35000 --method res_3d_paired --res_switch_iter 10000 --res_3d_iter 15000 \
  --hybrid_levels 2 --disable_c2f --aabb rect --kernel beta_scaled -i images_2 --cold --fastgs \
  --feature SV --lowpass \
  --3rgs --3rgs_warmup 1000          # ← the only addition
```

Tuning:
- If poses drift unstably early (common under `--cold`), **raise `--3rgs_warmup`**
  (1000–2000) so geometry is roughly right before perturbing cameras.
- If COLMAP is badly off and corrections look too small at the end, **raise
  `--3rgs_lr`** (e.g. `3e-5`). Adam normalizes per-param, so the per-step move is
  ≈ LR regardless of scene size; over 35k iters `1e-5` allows ~0.1–0.3 world-unit
  translation drift if monotonic.
- Add `--3rgs_reg 1e-6` to discourage runaway corrections.

## Outputs

At every `--save_iterations` (and the final iter), next to the PLY:

```
point_cloud/iteration_<N>/refined_poses/
  refined_poses.npz     # authoritative: names, R_wc[K,3,3], t[K,3], c2w[K,4,4], delta[K,9]
  images_refined.txt    # best-effort COLMAP images.txt (reuse the original cameras.txt;
                        #   intrinsics are unchanged — only extrinsics move)
```

`R_wc`/`t` are COLMAP convention (`X_cam = R_wc·X_world + t`). The **trained PLY is
itself the improved reconstruction** — it was optimized under the corrected poses.
The export lets you re-feed the corrected cameras into a downstream COLMAP/MVS
pipeline. Per-iter progress prints `[3RGS iter=N] pose delta | trans(...) rot6d(...)`.

## Verified

- Math: zero-init ⇒ `M = I`, `q = [1,0,0,0]`; per-camera gradient isolation
  (only the rendered camera's delta gets grad); `refined_world2cam` round-trips to
  base COLMAP extrinsics at zero delta.
- End-to-end on `data/personal/Max` (296 cams, `3D_SH_res` + `--feature SV` +
  `--fastgs` + `--cold`, 600 iters): no crash/NaN, deltas grow from 0, all 296
  refined extrinsics exported with `det(R_wc) = 1.0`.

## Not ported (would need correspondence data)

- **MLP pose variant** (`--pose_opt_type mlp`): a global MLP reparametrization of
  the per-camera delta. Easy to add (mirror `CameraOptModuleMLP`) but without the
  epipolar loss it's just a different parametrization.
- **Global epipolar loss** (`--use_corres_epipolar_loss`): needs precomputed
  per-pair correspondences (MASt3R-SfM, or extracted from the COLMAP DB). This is
  the 3R-GS "best practice" that lifts accuracy, but it's a separate pipeline + env.
- **Test-time pose optimization** (`optim_camtoworlds`): refines held-out test
  poses at eval (refined train poses live in a slightly different frame than raw
  COLMAP test poses). Add this if you need meaningful held-out PSNR under `--eval`.
