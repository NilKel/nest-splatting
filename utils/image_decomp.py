"""Structure-texture decomposition helpers for --decomp training.

Applies a guided filter (He & Sun 2010) to every training image ONCE, caching
the low-frequency result next to the source images. During training, the cached
low image is loaded onto each Camera as ``cam.gt_low`` and the high-frequency
residual is derived on-the-fly as ``gt - gt_low``.

Pipeline:
  1. At dataset load time, ``ensure_decomp_cache`` iterates CameraInfos, runs
     the guided filter on each, and saves ``<image_name>_low.png`` into
     ``<dataset_root>/decomp_r{R}_eps{EPS}/``. Skips if the file already exists.
  2. ``load_gt_low_for_cameras`` attaches the cached tensor to each Camera so
     the train loop can read ``cam.gt_low`` without I/O per step.
"""
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _box_filter(img, r):
    k = 2 * r + 1
    return cv2.boxFilter(img, -1, (k, k), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """He & Sun 2010 guided filter; per-channel for RGB. Inputs float32 in [0,1]."""
    if src.ndim == 3:
        out = np.empty_like(src)
        for c in range(src.shape[2]):
            out[..., c] = guided_filter(guide[..., c], src[..., c], r, eps)
        return out
    mean_I = _box_filter(guide, r)
    mean_p = _box_filter(src, r)
    mean_Ip = _box_filter(guide * src, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _box_filter(guide * guide, r)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)
    return mean_a * guide + mean_b


def _eps_tag(eps: float) -> str:
    """Format eps for directory naming: 0.01 -> '1e-2'."""
    return f"{eps:.0e}".replace("0e", "e").replace("+", "").replace("-0", "-")


def decomp_cache_dir(image_path: str, r: int, eps: float) -> str:
    """Sibling directory to the image folder (one level up from the image itself)."""
    img_dir = os.path.dirname(image_path)
    dataset_root = os.path.dirname(img_dir)  # parent of e.g. images_2/
    return os.path.join(dataset_root, f"decomp_r{r}_eps{_eps_tag(eps)}")


def _image_cache_path(cache_dir: str, image_name: str) -> str:
    return os.path.join(cache_dir, f"{image_name}_low.png")


def ensure_decomp_cache(cam_infos, r: int, eps: float, verbose: bool = True) -> str:
    """Build (or reuse) the guided-filter cache for a list of CameraInfo objects.

    Args:
        cam_infos: iterable of CameraInfo (must have ``image_path`` and ``image_name``).
        r: guided-filter radius.
        eps: guided-filter regularizer.

    Returns:
        Path to the cache directory.
    """
    if len(cam_infos) == 0:
        return ""
    first_cache = decomp_cache_dir(cam_infos[0].image_path, r, eps)
    os.makedirs(first_cache, exist_ok=True)

    # Figure out which views are missing so we only decompose those.
    todo = []
    for ci in cam_infos:
        cache_path = _image_cache_path(
            decomp_cache_dir(ci.image_path, r, eps), ci.image_name)
        if not os.path.exists(cache_path):
            todo.append(ci)

    if len(todo) == 0:
        if verbose:
            print(f"[DECOMP] Cache hit for all {len(cam_infos)} views at "
                  f"{first_cache}; skipping guided-filter pass.")
        return first_cache

    if verbose:
        print(f"[DECOMP] Building cache in {first_cache} "
              f"(r={r}, eps={eps}) — {len(todo)}/{len(cam_infos)} views missing.")

    it = tqdm(todo, desc="[DECOMP]") if verbose else todo
    for ci in it:
        cache_dir = decomp_cache_dir(ci.image_path, r, eps)
        os.makedirs(cache_dir, exist_ok=True)
        out_path = _image_cache_path(cache_dir, ci.image_name)
        bgr = cv2.imread(ci.image_path)
        if bgr is None:
            raise RuntimeError(f"[DECOMP] Failed to read {ci.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        low = guided_filter(rgb, rgb, r=r, eps=eps)
        low_u8 = np.clip(low * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, cv2.cvtColor(low_u8, cv2.COLOR_RGB2BGR))

    return first_cache


def load_gt_low_for_cameras(cameras, r: int, eps: float, device: str = "cuda",
                            verbose: bool = True) -> int:
    """Attach cached low-freq GT to each Camera as ``cam.gt_low``.

    Resizes the cached full-res low image to each camera's rendered resolution
    (handled by the --resolution flag at Camera construction time), matching
    the layout of ``cam.original_image``: [C, H, W] float32 in [0, 1].

    Returns the number of cameras successfully populated.
    """
    populated = 0
    for cam in cameras:
        img_path = getattr(cam, "image_path", None)
        image_name = getattr(cam, "image_name", None)
        if img_path is None or image_name is None:
            continue
        cache_path = _image_cache_path(decomp_cache_dir(img_path, r, eps), image_name)
        if not os.path.exists(cache_path):
            raise RuntimeError(
                f"[DECOMP] Missing cached low image at {cache_path}. "
                f"Call ensure_decomp_cache() first.")
        bgr = cv2.imread(cache_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # [H0, W0, 3]

        # Match the render resolution used by the main path. cam.original_image
        # has shape [C, H, W] at the downscaled resolution, so resize to match.
        H, W = int(cam.image_height), int(cam.image_width)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().to(device)
        cam.gt_low = t
        populated += 1

    if verbose:
        print(f"[DECOMP] Attached gt_low to {populated}/{len(cameras)} cameras.")
    return populated
