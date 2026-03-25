#!/usr/bin/env python3
"""
Render final circular trajectory videos with optional decomposition.

Renders many frames (250 by default) and encodes them as MP4 videos.
Use `preview.py` first to find the right camera angle, then run this
with the same --radius_scale / --tilt / --elevation parameters.

Usage:
    # Full render only:
    python scripts/circle/render_video.py --model_path outputs/.../model --radius_scale 1.3 --tilt 0.15

    # With decomposition (full + gaussian_only + ngp_only):
    python scripts/circle/render_video.py --model_path outputs/.../model --radius_scale 1.3 --tilt 0.15 --decompose

    # Custom frame count and playback speed:
    python scripts/circle/render_video.py --model_path outputs/.../model --num_views 300 --fps 24

Output:
    <model_path>/circle_renders/circle.mp4              — full render
    <model_path>/circle_renders_gaussian_only/circle.mp4 — per-gaussian features only (hash masked)
    <model_path>/circle_renders_ngp_only/circle.mp4      — hash features only (per-gaussian masked)

    Individual frames are also saved as frame_XXXX.png in each directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser
from scripts.circle.circle_cam import (
    load_model, compute_orbit_params, render_frames, save_frames_to_disk, save_videos,
)


def main():
    parser = ArgumentParser(description="Render circular trajectory videos with decomposition")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--source_path", type=str, default=None,
                       help="Override source data path (if training data moved)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Checkpoint iteration (-1 = latest)")
    parser.add_argument("--num_views", type=int, default=250,
                       help="Number of frames around the circle (default: 250)")
    parser.add_argument("--radius_scale", type=float, default=1.0,
                       help="Orbit radius multiplier (>1 = zoom out)")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Vertical offset for camera orbit (units of scene radius)")
    parser.add_argument("--tilt", type=float, default=0.0,
                       help="Tilt look target (units of scene radius, sign = scene-dependent)")
    parser.add_argument("--decompose", action="store_true",
                       help="Also render gaussian_only and ngp_only decompositions")
    parser.add_argument("--fps", type=int, default=20,
                       help="Video playback framerate (default: 20)")
    parser.add_argument("--no_frames", action="store_true",
                       help="Skip saving individual PNG frames (video only)")
    args = parser.parse_args()

    model = load_model(args.model_path, source_path=args.source_path, iteration=args.iteration)
    orbit = compute_orbit_params(model['scene'],
                                 radius_scale=args.radius_scale,
                                 elevation=args.elevation,
                                 tilt=args.tilt)

    all_frames, render_fps = render_frames(model, orbit,
                                           num_views=args.num_views,
                                           decompose=args.decompose)

    if not args.no_frames:
        save_frames_to_disk(all_frames, args.model_path)

    save_videos(all_frames, args.model_path, fps=args.fps)

    duration = args.num_views / args.fps
    print(f"\n[DONE] {args.num_views} frames, {args.fps} FPS ({duration:.1f}s video), render speed: {render_fps:.1f} FPS")


if __name__ == "__main__":
    main()
