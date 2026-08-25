// Shared uniform layout (std140 vec4 array, indexed as flat floats).
layout(std140, set = 0, binding = 0) uniform U { vec4 v[16]; } u;
#define UF(i) u.v[(i) >> 2][(i) & 3]
#define U_VIEW(i)   UF(i)          // 0..15   viewmatrix, CUDA memory order
#define U_PROJ(i)   UF(16 + (i))   // 16..31  projmatrix, CUDA memory order
#define U_CAMPOS    vec3(UF(32), UF(33), UF(34))
#define U_W         UF(35)
#define U_H         UF(36)
#define U_N         int(UF(39))
#define U_K         int(UF(40))
#define U_KT        int(UF(41))
#define U_SHBIAS    UF(42)
#define U_RESBIAS   UF(43)
#define U_COMPACT   UF(44)
#define U_OABETA    (UF(45) > 0.5)
#define U_BETAMULT  UF(46)
#define U_DROPLP    (UF(47) > 0.5)
#define U_SCALEMOD  UF(48)
#define U_ASCALE    UF(49)
#define U_AOFFSET   UF(50)
#define U_ATLASW    UF(51)
#define U_LAYERH    UF(52)
const float FilterSize = 0.707106;     // sqrt(2)/2  (auxiliary.h)
const float FilterInvSquare = 2.0;
#define U_PAD       UF(54)             // octagon pad in px
#define U_LPMODE    UF(55)             // 0: SW-binning filter_r ; 1: exact low-pass radius sqrt(ln(255*opa))
