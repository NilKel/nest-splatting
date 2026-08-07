#!/usr/bin/env bash
# Re-measure FastGS per-pixel overdraw at the CORRECT mip-360 resolutions,
# using FastGS's own `mult` default (0.5).
#
# The original numbers in intersection_comparison/stats_table.md were measured
# with intersection_maps.py's *script* default mult=1.0 at the 1600-capped
# training resolution. Both differ from FastGS's actual configuration
# (arguments/__init__.py:100 sets mult=0.5, used by train.py) and from NeST's
# evaluation resolution. On treehill the two corrections together moved the
# mean from 49.66 -> 38.42.
#
# NeST's own overdraw maps (speed_comparison/mip360_nest_intersection/) are
# already at images_2/images_4, so only the FastGS side needs redoing.
set -uo pipefail

FASTGS=/home/nilkel/Projects/FastGS
OUT=/home/nilkel/Projects/nest-splatting/speed_comparison/fastgs_overdraw_correct_res
mkdir -p "$OUT"

declare -A TIER=(
  [bicycle]=images_4  [flowers]=images_4  [garden]=images_4
  [stump]=images_4    [treehill]=images_4
  [bonsai]=images_2   [counter]=images_2  [kitchen]=images_2  [room]=images_2
)

cd "$FASTGS"
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
  imgs=${TIER[$scene]}
  echo "=== $scene ($imgs) ==="
  conda run -n fastgs --no-capture-output python intersection_maps.py \
      -m "output/$scene" -i "$imgs" --mult 0.5 2>&1 | tail -4
  src="output/$scene/test/ours_30000/intersection_summary.json"
  [ -f "$src" ] && cp "$src" "$OUT/${scene}.json"
done

echo "=== overdraw sweep complete -> $OUT ==="
