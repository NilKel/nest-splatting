#!/usr/bin/env python3
"""
Preview circular camera trajectory — quickly render a few frames to dial in the POV.

Renders 10 frames by default (fast) so you can iterate on camera parameters.
Once you're happy with the angle, use `render_video.py` with the same parameters.

Usage:
    # Default orbit (matches training camera distance):
    python scripts/circle/preview.py --model_path outputs/.../model

    # Zoom out 30%, tilt camera to look downward:
    python scripts/circle/preview.py --model_path outputs/.../model --radius_scale 1.3 --tilt 0.3

    # If source data is in a different location than where training ran:
    python scripts/circle/preview.py --model_path outputs/.../model --source_path data/mip_360/bicycle

Camera parameters:
    --radius_scale  Zoom: >1 = further out, <1 = closer in (default: 1.0)
    --elevation     Move cameras up/down along scene up vector (default: 0.0)
    --tilt          Angle cameras to look up/down. Sign depends on scene orientation —
                    try positive first, flip if it goes the wrong way (default: 0.0)
    --num_views     Number of preview frames around the circle (default: 10)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser
from scripts.circle.circle_cam import (
    load_model, compute_orbit_params, render_frames, save_frames_to_disk,
)


def main():
    parser = ArgumentParser(description="Preview circular camera trajectory (quick POV adjustment)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--source_path", type=str, default=None,
                       help="Override source data path (if training data moved)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Checkpoint iteration (-1 = latest)")
    parser.add_argument("--num_views", type=int, default=10,
                       help="Number of preview frames (default: 10)")
    parser.add_argument("--radius_scale", type=float, default=1.0,
                       help="Orbit radius multiplier (>1 = zoom out)")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Vertical offset for camera orbit (units of scene radius)")
    parser.add_argument("--tilt", type=float, default=0.0,
                       help="Tilt look target (units of scene radius, sign = scene-dependent)")
    args = parser.parse_args()

    model = load_model(args.model_path, source_path=args.source_path, iteration=args.iteration)
    orbit = compute_orbit_params(model['scene'],
                                 radius_scale=args.radius_scale,
                                 elevation=args.elevation,
                                 tilt=args.tilt)

    all_frames, fps = render_frames(model, orbit, num_views=args.num_views)
    save_frames_to_disk(all_frames, args.model_path)

    print(f"\nPreview frames saved to {args.model_path}/circle_renders/")
    print(f"Once happy with the angle, run:")
    print(f"  python scripts/circle/render_video.py \\")
    print(f"    --model_path {args.model_path} \\")
    if args.source_path:
        print(f"    --source_path {args.source_path} \\")
    print(f"    --radius_scale {args.radius_scale} --tilt {args.tilt} --elevation {args.elevation}")


if __name__ == "__main__":
    main()
