#version 450
#include "common.glsl"
layout(std430, set = 0, binding = 1) readonly buffer B1 { vec4 prep[]; };
layout(std430, set = 0, binding = 2) readonly buffer B2 { uint sorted_vals[]; };
layout(std430, set = 0, binding = 3) readonly buffer B3 { vec4 aparams[]; };
#ifdef FETCH_BY_ID
layout(location = 0) flat out uint gidOut;
#else
layout(location = 0) flat out vec4 f0;
layout(location = 1) flat out vec4 f1;
layout(location = 2) flat out vec4 f2;
layout(location = 3) flat out vec4 f3;
layout(location = 4) flat out vec4 f4;
layout(location = 5) flat out vec4 f5;
layout(location = 6) flat out float layer;
#endif
float hsup(vec4 c, vec2 n) {            // support of ellipse {d: d^T M d <= t} in direction n
    float det = c.x*c.z - c.y*c.y;
    return sqrt(max(c.w * (c.z*n.x*n.x - 2.0*c.y*n.x*n.y + c.x*n.y*n.y) / det, 0.0));
}
// Octagon circumscribing the support (8 half-planes at 45deg), then Sutherland-Hodgman
// clipped by the rational's validity half-plane (-dw).d <= 0.9 (+pad). Up to 9 vertices;
// emitted as a TRIANGLE_FAN of 10 vertices (unused slots repeat the last vertex).
void main() {
    uint gid = sorted_vals[gl_InstanceIndex];
    vec4 p3 = prep[gid*8 + 3];
    if (gid == 0xFFFFFFFFu || p3.w == 0.0 || (U_LPMODE > 0.5 && prep[gid*8 + 7].x < 0.5)) { gl_Position = vec4(-4.0, -4.0, 0.0, 1.0); return; }
    vec4 p0 = prep[gid*8 + 0], p2 = prep[gid*8 + 2], p4 = prep[gid*8 + 4], p5 = prep[gid*8 + 5], p6 = prep[gid*8 + 6];
#ifdef FETCH_BY_ID
    gidOut = gid;
#else
    f0 = p0; f1 = prep[gid*8 + 1]; f2 = p2; f3 = p3;
    f4 = aparams[gid*3 + 0]; f5 = aparams[gid*3 + 1]; layer = aparams[gid*3 + 2].x;
#endif
    bool tight = U_LPMODE > 0.5;
    float PAD = U_PAD;
    // 8 supports
    float h[8];
    for (int j = 0; j < 8; j++) {
        float th = float(j) * 0.78539816339;
        vec2 n = vec2(cos(th), sin(th));
        float hj;
        if (!tight)          hj = max(hsup(p4, n), p5.x);
        else if (U_KT == 4)  hj = min(hsup(p6, n), max(hsup(p4, n), p5.w));
        else                 hj = max(hsup(p4, n), p5.w);
        h[j] = hj + PAD;
    }
    // 8 corners (intersection of adjacent edge lines), CCW
    vec2 c[8];
    const float D = 0.70710678118;      // sin(45deg)
    for (int j = 0; j < 8; j++) {
        float th0 = float(j) * 0.78539816339, th1 = float(j + 1) * 0.78539816339;
        vec2 n0 = vec2(cos(th0), sin(th0)), n1 = vec2(cos(th1), sin(th1));
        c[j] = vec2(h[j]*n1.y - n0.y*h[(j+1)&7], n0.x*h[(j+1)&7] - h[j]*n1.x) / D;
    }
    // clip against (-dw).d <= 0.9 + pad*|dw|
    vec2 dw = p2.xy; float dwl = length(dw);
    vec2 poly[10]; int m = 0;
    if (tight && dwl > 1e-7) {
        vec2 nh = -dw / dwl; float hc = 0.9 / dwl + PAD;
        for (int j = 0; j < 8; j++) {
            vec2 a = c[j], b = c[(j+1)&7];
            float da = dot(nh, a) - hc, db = dot(nh, b) - hc;
            if (da <= 0.0) poly[m++] = a;
            if ((da <= 0.0) != (db <= 0.0)) poly[m++] = mix(a, b, da / (da - db));
        }
    } else {
        for (int j = 0; j < 8; j++) poly[m++] = c[j];
    }
    if (m == 0) { gl_Position = vec4(-4.0, -4.0, 0.0, 1.0); return; }
    int k = min(gl_VertexIndex, m - 1);
    vec2 px = p0.xy + poly[k];
    gl_Position = vec4((px.x + 0.5) / U_W * 2.0 - 1.0, (px.y + 0.5) / U_H * 2.0 - 1.0, 0.0, 1.0);
}
