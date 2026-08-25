#!/bin/bash
set -e
cd "$(dirname "$0")"
GLSLC=/home/nilkel/miniconda3/envs/nest_splatting/bin/glslc
mkdir -p spv
$GLSLC -O --target-env=vulkan1.3 -I shaders shaders/preprocess.comp -o spv/preprocess.comp.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders shaders/radix_scan.comp -o spv/radix_scan.comp.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders shaders/sort_args.comp -o spv/sort_args.comp.spv
for E in 1 2 4 8; do
  $GLSLC -O --target-env=vulkan1.3 -I shaders -DELEMS_OVERRIDE=${E}u shaders/radix_hist.comp -o spv/radix_hist.comp_e$E.spv
  $GLSLC -O --target-env=vulkan1.3 -I shaders -DELEMS_OVERRIDE=${E}u shaders/radix_scatter.comp -o spv/radix_scatter.comp_e$E.spv
done
$GLSLC -O --target-env=vulkan1.3 -I shaders shaders/splat.vert -o spv/splat.vert.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders -DFETCH_BY_ID shaders/splat.vert -o spv/splat.vert_id.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders shaders/splat.frag -o spv/splat.frag.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders -DSTATS shaders/splat.frag -o spv/splat.frag_stats.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders -DFETCH_BY_ID shaders/splat.frag -o spv/splat.frag_id.spv
$GLSLC -O --target-env=vulkan1.3 -I shaders -DFETCH_BY_ID -DSTATS shaders/splat.frag -o spv/splat.frag_id_stats.spv
g++ -O2 -std=c++17 main.cpp -o vk_raster -lvulkan
echo "build OK"
