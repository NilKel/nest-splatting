#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
layout(push_constant) uniform PC { uint shift; } pcs;
layout(std430, set = 0, binding = 5) readonly buffer ARGS { uint visCount; uint M; uint numBlocks; uint pad0; uint dispatchX, dispatchY, dispatchZ, pad1; uint drawVerts, drawInst, drawFirstV, drawFirstI; } args;
#define PC_SHIFT pcs.shift
#define PC_NB args.numBlocks
#define PC_N args.M
#ifndef ELEMS_OVERRIDE
#define ELEMS 4u
#else
#define ELEMS ELEMS_OVERRIDE
#endif
#define BLOCK (256u * ELEMS)
