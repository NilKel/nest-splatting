"""Parse a Cinematic Renderer .key animation file and dump keyframe values as JSON.

The format is a `Sequence(duration, mode) { -  <track>(<type>, <interp>){ {t: v...}, ... } ... }`
tree, one track per renderer parameter. Tracks are scalar, vector (3 floats) or
rotation (4 floats, quaternion).

Only keyframe values are extracted -- no interpolation is attempted, since the
renderer's own spline evaluation is the authority for intermediate times.
"""
import argparse
import json
import re

TRACK_RE = re.compile(r"^-\s+(\w+)\((\w+),\s*(\w+)\)\{")
KEY_RE = re.compile(r"\{\s*([-\d.eE]+)\s*:\s*([^}]*)\}")


def parse(path):
    tracks = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            m = TRACK_RE.match(s)
            if m:
                cur = {"name": m.group(1), "type": m.group(2), "interp": m.group(3), "keys": {}}
                tracks[cur["name"]] = cur
                continue
            if cur is None:
                continue
            if s.startswith("}"):
                cur = None
                continue
            m = KEY_RE.search(s)
            if m:
                t = float(m.group(1))
                vals = [float(x) for x in m.group(2).split(",") if x.strip()]
                cur["keys"][t] = vals[0] if len(vals) == 1 else vals
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keyfile")
    ap.add_argument("--at", type=float, default=0.0, help="keyframe time to extract")
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()

    tracks = parse(args.keyfile)
    times = sorted({t for tr in tracks.values() for t in tr["keys"]})
    print(f"{len(tracks)} tracks, {len(times)} keyframes at t = {times}\n")

    snap, animated = {}, []
    for name, tr in sorted(tracks.items()):
        if not tr["keys"]:
            continue
        if args.at in tr["keys"]:
            snap[name] = tr["keys"][args.at]
        vals = [json.dumps(v) for v in tr["keys"].values()]
        if len(set(vals)) > 1:
            animated.append(name)

    print(f"--- keyframe t={args.at} ---")
    for k, v in snap.items():
        mark = "  <-- ANIMATED" if k in animated else ""
        print(f"  {k:18s} = {v}{mark}")
    print(f"\nanimated tracks (differ across keyframes): {animated or 'none'}")
    print(f"static tracks: {len(snap) - len(animated)}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"at": args.at, "values": snap, "animated_tracks": animated,
                       "keyframe_times": times}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
