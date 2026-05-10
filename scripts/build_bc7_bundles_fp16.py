#!/usr/bin/env python3
"""Re-build HD BC7 .bitymi bundles with FP16-quantized SV/SB params.

Reuses the cached `baked_fp16.ply` produced by build_astc_bundles_fp16.py.
For the BC7 NAT2, runs `export_textures_bin.py` (idempotent — skips if
scene.nat2 already exists). Packs `<scene>.bitymi` (SV HD) and
`<scene>_sb.bitymi` (SB HD) and uploads, replacing the older FP32-PLY
versions on HF.

Usage:
    python scripts/build_bc7_bundles_fp16.py [--dry-run] [--scenes ...]
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

SCRIPTS = Path(__file__).parent
EXPORT_SCRIPT = SCRIPTS / "export_textures_bin.py"
PACK_SCRIPT   = SCRIPTS / "pack_bitymi.py"

PYTHON = "/home/nilkel/miniconda3/envs/nest_splatting/bin/python"

BAKE_ROOT = Path("/mnt/nilkel_hdd/outputs/mip_360")
SCENES = ["counter", "room", "kitchen", "bonsai",
          "stump", "garden", "bicycle", "flowers", "treehill"]
SV_DIR = "{scene}/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac"
SB_DIR = "{scene}/3D_SH_res/SB_30thr_005w25gLP4lev_FRP5k10_c2f_Jac"

TMP_BUNDLE = Path("/tmp/bc7_bundles_fp16")
TMP_BUNDLE.mkdir(exist_ok=True)
HF_REPO = "Nilkel/bitymi-demos"


def fp16_roundtrip_ply(src: Path, dst: Path) -> int:
    data = src.read_bytes()
    hdr_end = data.find(b'end_header\n') + len(b'end_header\n')
    header = data[:hdr_end].decode('ascii')

    lines = header.splitlines()
    N = next(int(l.split()[2]) for l in lines if l.startswith('element vertex'))
    fields = [l.split()[2] for l in lines if l.startswith('property ')]

    color_prefixes = ('f_dc_', 'f_rest_',
                      'sv_site_', 'sv_col_', 'sv_tau_', 'sv_dc_',
                      'sb_')
    target_cols = [i for i, n in enumerate(fields)
                   if any(n.startswith(p) for p in color_prefixes)]
    if not target_cols:
        dst.write_bytes(data)
        return 0

    body = np.frombuffer(data, dtype=np.float32, offset=hdr_end).copy()
    body = body.reshape(N, len(fields))
    cols = np.array(target_cols, dtype=np.int64)
    body[:, cols] = body[:, cols].astype(np.float16).astype(np.float32)

    with open(dst, 'wb') as f:
        f.write(data[:hdr_end])
        f.write(body.tobytes())
    return len(target_cols)


def ensure_bc7_nat2(baked: Path) -> Path:
    out = baked / "scene.nat2"
    if out.exists():
        return out
    cmd = [PYTHON, str(EXPORT_SCRIPT), str(baked), str(out)]
    subprocess.check_call(cmd)
    return out


def pack(out_bundle: Path, ply: Path, cams: Path, atlas: Path):
    cmd = [PYTHON, str(PACK_SCRIPT), str(out_bundle),
           "--ply", str(ply), "--cameras", str(cams), "--atlas", str(atlas)]
    subprocess.check_call(cmd)


def upload(local_path: Path, repo_filename: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_filename,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"Update {repo_filename} (FP16 SV/SB params + BC7 atlas)",
    )


def find_cameras(baked: Path) -> Path | None:
    p = baked
    for _ in range(3):
        p = p.parent
        cand = p / "cameras.json"
        if cand.exists():
            return cand
    return None


def process_bake(scene: str, mode: str, baked: Path, out_name: str, dry_run: bool):
    if not baked.exists():
        print(f"  ⚠️  no bake at {baked}, skipping")
        return False
    cams = find_cameras(baked)
    if cams is None:
        print(f"  ⚠️  no cameras.json near {baked}, skipping")
        return False

    src_ply = baked / "baked.ply"
    fp16_ply = baked / "baked_fp16.ply"
    if not fp16_ply.exists():
        t0 = time.time()
        n = fp16_roundtrip_ply(src_ply, fp16_ply)
        print(f"  · FP16-roundtripped {n} f32 cols ({time.time()-t0:.1f}s)")
    else:
        print(f"  · baked_fp16.ply cached")

    print(f"  · ensuring scene.nat2…")
    t0 = time.time()
    bc7_nat2 = ensure_bc7_nat2(baked)
    print(f"    ({bc7_nat2.stat().st_size/1e6:.1f} MB, {time.time()-t0:.1f}s)")

    bundle = TMP_BUNDLE / out_name
    print(f"  · packing {out_name}…")
    pack(bundle, fp16_ply, cams, bc7_nat2)
    print(f"    ({bundle.stat().st_size/1e6:.1f} MB)")

    if dry_run:
        print(f"  · DRY RUN: skipping upload")
        bundle.unlink()
        return True

    print(f"  · uploading…")
    t0 = time.time()
    upload(bundle, out_name)
    print(f"    ({time.time()-t0:.1f}s)")
    bundle.unlink()
    print(f"  ✓ done: {out_name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    print(f"[BATCH] FP16 + BC7 HD bundle re-upload")
    print(f"[BATCH] source: {BAKE_ROOT}")
    print(f"[BATCH] HF:     {HF_REPO}")
    print(f"[BATCH] scenes: {args.scenes}\n")

    todo = []
    for scene in args.scenes:
        for mode, dir_template in (("SV", SV_DIR), ("SB", SB_DIR)):
            mode_dir = BAKE_ROOT / dir_template.format(scene=scene)
            baked = mode_dir / "baked_atlas"  # HD only

            # Naming: <scene>.bitymi (SV HD) / <scene>_sb.bitymi (SB HD).
            # Matches the existing on-HF naming for kitchen.bitymi, kitchen_sb.bitymi.
            parts = [scene]
            if mode == "SB":
                parts.append("sb")
            out_name = "_".join(parts) + ".bitymi"
            todo.append((scene, mode, baked, out_name))

    print(f"[BATCH] queued {len(todo)} bundles\n")
    n_done = n_failed = 0
    t_start = time.time()
    for i, (scene, mode, baked, out_name) in enumerate(todo, 1):
        elapsed = time.time() - t_start
        eta = (elapsed / max(i-1, 1)) * (len(todo) - i + 1) if i > 1 else 0
        print(f"[{i}/{len(todo)}] [{scene}][{mode}] → {out_name}  "
              f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")
        try:
            if process_bake(scene, mode, baked, out_name, args.dry_run):
                n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
        print()

    total = time.time() - t_start
    print(f"[BATCH] complete: {n_done}/{len(todo)} done  ({n_failed} failed)  in {total/60:.1f} min")


if __name__ == "__main__":
    main()
