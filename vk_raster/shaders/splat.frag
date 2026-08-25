#version 450
#include "common.glsl"
layout(set = 0, binding = 4) uniform sampler2DArray atlas;
#ifdef FETCH_BY_ID
layout(std430, set = 0, binding = 1) readonly buffer B1 { vec4 prep[]; };
layout(std430, set = 0, binding = 3) readonly buffer B3 { vec4 aparams[]; };
layout(location = 0) flat in uint gid;
#else
layout(location = 0) flat in vec4 f0;
layout(location = 1) flat in vec4 f1;
layout(location = 2) flat in vec4 f2;
layout(location = 3) flat in vec4 f3;
layout(location = 4) flat in vec4 f4;
layout(location = 5) flat in vec4 f5;
layout(location = 6) flat in float layer;
#endif
#ifdef STATS
layout(std430, set = 0, binding = 5) buffer S { uint survivors; uint nonhelper; uint f_denom; uint f_rho9; uint f_alpha; uint f_power; uint fb_frags; uint fb_rho9; };
#endif
layout(location = 0) out vec4 outColor;
void main() {
#ifdef FETCH_BY_ID
    vec4 f0 = prep[gid*8 + 0], f1 = prep[gid*8 + 1], f2 = prep[gid*8 + 2], f3 = prep[gid*8 + 3];
#endif
#ifdef STATS
    bool H_ = gl_HelperInvocation; if (!H_) atomicAdd(nonhelper, 1u);
#ifdef FETCH_BY_ID
    bool FB_ = prep[gid*8 + 7].x < 1.5; if (!H_ && FB_) atomicAdd(fb_frags, 1u);
#else
    bool FB_ = false;
#endif
#endif
    vec2 pix = floor(gl_FragCoord.xy);
    vec2 xy = f0.xy;
    vec2 d = xy - pix;
    float dx = pix.x - xy.x, dy = pix.y - xy.y;
    float du_lin = f1.x*dx + f1.y*dy;
    float dv_lin = f1.z*dx + f1.w*dy;
    float denom  = 1.0 + f2.x*dx + f2.y*dy;
    if (denom < 0.1) {
#ifdef STATS
        if (!H_) atomicAdd(f_denom, 1u);
#endif
        discard; }
    float inv_d = 1.0 / denom;
    float uu = f0.z + du_lin * inv_d;
    float vv = f0.w + dv_lin * inv_d;
    float rho3d = uu*uu + vv*vv;
    float rho2d = FilterInvSquare * (d.x*d.x + d.y*d.y);
    float rho = min(rho3d, rho2d);
    float opa = f2.z, shape = f2.w;
    float alpha;
    if (U_KT == 4) {
        const float k_sq = 9.0;
        if (rho3d >= k_sq + 1e-6) {
#ifdef STATS
            if (!H_) atomicAdd(f_rho9, 1u); if (!H_ && FB_) atomicAdd(fb_rho9, 1u);
#endif
            discard; }
        float base = max(0.0, 1.0 - rho3d / k_sq);
        float alpha_beta = pow(base, shape);
        float alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else {
        float power = -0.5 * rho;
        if (power > 0.0) {
#ifdef STATS
            if (!H_) atomicAdd(f_power, 1u);
#endif
            discard; }
        alpha = min(0.99, opa * exp(power));
    }
    if (alpha < 1.0/255.0) {
#ifdef STATS
        if (!H_) atomicAdd(f_alpha, 1u);
#endif
        discard; }
#ifdef STATS
    atomicAdd(survivors, 1u);
#endif
#ifdef FETCH_BY_ID
    vec4 f4 = aparams[gid*3 + 0], f5 = aparams[gid*3 + 1]; float layer = aparams[gid*3 + 2].x;
#endif
    vec3 feat = f3.xyz;
    if (f4.z > 0.0) {
        float au = max(f5.x, min(f5.z, f4.x + f4.z * uu));
        float av = max(f5.y, min(f5.w, f4.y + f4.w * vv));
        vec3 tex = texture(atlas, vec3((au + 0.5) / U_ATLASW, (av + 0.5) / U_LAYERH, layer)).rgb;
        feat += tex * U_ASCALE + U_AOFFSET;
    }
    feat = max(vec3(0.0), feat + U_RESBIAS);
    outColor = vec4(feat * alpha, alpha);
}
