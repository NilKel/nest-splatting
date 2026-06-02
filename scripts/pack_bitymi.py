#!/usr/bin/env python3
"""Pack a Halloumi viewer bundle into a single `.bitymi` file.

Layout
------
    [8 B] magic = "BITYMI01"
    [4 B] num_chunks (u32 LE)
    per chunk:
        [4 B] kind   (u32 LE)
        [8 B] offset (u64 LE, from start of file)
        [8 B] size   (u64 LE)
    [concatenated payload bytes]

Header is 12 + 20 * num_chunks bytes. With 3 chunks (PLY + cameras + atlas)
total header overhead is 72 B — negligible vs the multi-MB payloads.

Kinds
-----
    0 = ply               (binary little-endian PLY)
    1 = npz               (compressed-3DGS .npz)
    2 = cameras_json      (utf-8 JSON, no trailing null)
    3 = nat2              (Halloumi atlas, NAT2 v2; see export_textures_bin.py)
    4 = natl              (legacy NATL v1 atlas — FP16 RGB only)

The JS loader on the viewer side parses the TOC and slices the buffer into
(pc_data, scene_data, atlas_data), then calls `run_wasm(...)` exactly as if
the user had passed the three URL params separately. So the wasm binary
doesn't need to know anything about BITYMI.

Usage
-----
    pack_bitymi.py <out.bitymi> --ply <baked.ply> [--cameras cams.json] [--atlas scene.nat2]
or  pack_bitymi.py <out.bitymi> --bake-dir <dir>     (auto-discovers files)
"""
import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"BITYMI01"
KIND_PLY     = 0
KIND_NPZ     = 1
KIND_CAMERAS = 2
KIND_NAT2    = 3
KIND_NATL    = 4
KIND_BPLY    = 5   # 8-bit Min-Max compressed PLY (see scripts/compress_baked_ply.py)
KIND_NAMES = {KIND_PLY: "ply", KIND_NPZ: "npz", KIND_CAMERAS: "cameras",
              KIND_NAT2: "nat2", KIND_NATL: "natl", KIND_BPLY: "bply"}

HEADER_FIXED = 8 + 4   # magic + num_chunks
ENTRY_SIZE   = 4 + 8 + 8  # kind + offset + size


def detect_atlas_kind(path: Path) -> int:
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"NAT2":
        return KIND_NAT2
    if head == b"NATL":
        return KIND_NATL
    raise SystemExit(f"unrecognized atlas magic in {path}: {head!r}")


def detect_pc_kind(path: Path, data: bytes) -> int:
    if path.suffix.lower() == ".npz":
        return KIND_NPZ
    if data[:4] == b"BPLY" or path.suffix.lower() == ".bply":
        return KIND_BPLY
    if data[:3] == b"ply":
        return KIND_PLY
    if path.suffix.lower() == ".ply":
        # Trust the extension if magic is missing — old PLY writers sometimes
        # skip the leading "ply\n" if they emit a comment block first.
        return KIND_PLY
    raise SystemExit(f"can't tell point cloud format of {path} (head={data[:8]!r})")


def auto_discover(bake_dir: Path):
    """Find baked.ply / cameras.json / scene.nat2 (or .natl) under a bake dir.
    Cameras are typically one level up (the parent training dir), so we walk
    upward up to two levels if not present in `bake_dir` itself.
    """
    ply = bake_dir / "baked.ply"
    if not ply.exists():
        # Fall back to any .ply / .npz in the dir.
        cands = list(bake_dir.glob("*.ply")) + list(bake_dir.glob("*.npz"))
        if not cands:
            raise SystemExit(f"--bake-dir {bake_dir} has no baked.ply / *.ply / *.npz")
        ply = cands[0]

    cameras = None
    for candidate in (bake_dir / "cameras.json",
                      bake_dir.parent / "cameras.json",
                      bake_dir.parent.parent / "cameras.json"):
        if candidate.exists():
            cameras = candidate
            break

    atlas = None
    for name in ("scene.nat2", "scene.natl", "atlas.nat2", "atlas.natl"):
        candidate = bake_dir / name
        if candidate.exists():
            atlas = candidate
            break

    return ply, cameras, atlas


def main():
    ap = argparse.ArgumentParser(description="Pack viewer bundle (.bitymi)")
    ap.add_argument("output", type=Path, help="path to write .bitymi")
    ap.add_argument("--ply",     type=Path, default=None, help="point cloud (.ply or .npz)")
    ap.add_argument("--cameras", type=Path, default=None, help="cameras.json (optional)")
    ap.add_argument("--atlas",   type=Path, default=None, help="scene.nat2 or scene.natl (optional)")
    ap.add_argument("--bake-dir", type=Path, default=None,
                    help="auto-discover baked.ply, cameras.json, scene.nat2 in this dir "
                         "(cameras.json walked up to grandparent)")
    args = ap.parse_args()

    if args.bake_dir is not None:
        ply_p, cam_p, atl_p = auto_discover(args.bake_dir)
        if args.ply is None:     args.ply = ply_p
        if args.cameras is None: args.cameras = cam_p
        if args.atlas is None:   args.atlas = atl_p

    if not args.ply:
        raise SystemExit("need --ply or --bake-dir")
    if not args.ply.exists():
        raise SystemExit(f"point cloud not found: {args.ply}")

    chunks = []  # list of (kind, bytes, source_label)

    pc_bytes = args.ply.read_bytes()
    chunks.append((detect_pc_kind(args.ply, pc_bytes), pc_bytes, str(args.ply)))

    if args.cameras is not None:
        if not args.cameras.exists():
            raise SystemExit(f"cameras file not found: {args.cameras}")
        chunks.append((KIND_CAMERAS, args.cameras.read_bytes(), str(args.cameras)))

    if args.atlas is not None:
        if not args.atlas.exists():
            raise SystemExit(f"atlas not found: {args.atlas}")
        chunks.append((detect_atlas_kind(args.atlas), args.atlas.read_bytes(), str(args.atlas)))

    n = len(chunks)
    header_size = HEADER_FIXED + ENTRY_SIZE * n
    cursor = header_size
    offsets = []
    for _, data, _ in chunks:
        offsets.append(cursor)
        cursor += len(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", n))
        for (kind, data, _), off in zip(chunks, offsets):
            f.write(struct.pack("<IQQ", kind, off, len(data)))
        for _, data, _ in chunks:
            f.write(data)

    total = args.output.stat().st_size
    print(f"Wrote {args.output} ({total / (1024 * 1024):.1f} MB) with {n} chunks:")
    for (kind, data, source), off in zip(chunks, offsets):
        print(f"  [{KIND_NAMES.get(kind, '???'):>8}] off={off:>11}  size={len(data) / (1024 * 1024):>7.1f} MB  ← {source}")


if __name__ == "__main__":
    main()
