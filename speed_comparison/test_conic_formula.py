#!/usr/bin/env python3
"""Test CONIC linearization against production ray-splat for a real Gauss.

Uses the (Tu, Tv, Tw) values extracted from the CUDA debug print of Gauss 1000
in the garden RD_SV bake.  Sweeps a grid of test pixels and computes rho3d via
both methods, reporting divergence.
"""
import numpy as np

# Real values captured from CUDA debug print of Gauss 1000 (garden RD_SV bake).
Tu = np.array([-22.804861, -17.815971, 2048.293457], dtype=np.float64)
Tv = np.array([ 55.461285, -14.544003, 1075.938843], dtype=np.float64)
Tw = np.array([ -0.010340,  -0.024043,   14.498360], dtype=np.float64)

# Production ray-splat: given a pixel (px, py), compute (s.x, s.y) via
#   k = px·Tw - Tu, l = py·Tw - Tv, p = k×l, s = p.xy / p.z
def rho3d_prod(px, py):
    k = px*Tw - Tu
    l = py*Tw - Tv
    p = np.cross(k, l)
    if abs(p[2]) < 1e-30:
        return np.nan, np.nan, np.nan
    sx, sy = p[0]/p[2], p[1]/p[2]
    return sx, sy, sx*sx + sy*sy


# CONIC linearization: forward Jacobian at (u=v=0).
#   Screen center = (Tu.z/Tw.z, Tv.z/Tw.z).
#   M = ∂screen/∂(u,v) at (0,0):
#     M[0,0] = (Tu.x·Tw.z − Tu.z·Tw.x) / Tw.z²   etc.
#   J⁻¹ = M⁻¹.
Twz = Tw[2]
cx_disc = Tu[2] / Twz
cy_disc = Tv[2] / Twz
a = Tu[0]*Twz - Tu[2]*Tw[0]
b = Tu[1]*Twz - Tu[2]*Tw[1]
c = Tv[0]*Twz - Tv[2]*Tw[0]
d = Tv[1]*Twz - Tv[2]*Tw[1]
det = a*d - b*c
scale = Twz*Twz / det
J = np.array([[ d*scale, -b*scale],
              [-c*scale,  a*scale]])
print(f"disc_center = ({cx_disc:.6f}, {cy_disc:.6f})")
print(f"J⁻¹ =")
print(J)
print(f"det(a·d - b·c) = {det:.6e}, scale = {scale:.6e}\n")

def rho3d_conic(px, py):
    d = np.array([px - cx_disc, py - cy_disc], dtype=np.float64)
    s = J @ d
    return s[0], s[1], s[0]*s[0] + s[1]*s[1]


# Sweep pixels in a grid centered on the surfel.
print(f"{'pixel':<15} {'prod s.x':>10} {'prod s.y':>10} {'prod rho3d':>12} "
      f"{'conic s.x':>10} {'conic s.y':>10} {'conic rho3d':>12} {'Δrho3d %':>10}")
print("-" * 110)

for offset in [0, 1, 2, 3, 5, 10, 20]:
    for direction in ['+x', '-x', '+y', '-y', 'diag']:
        if direction == '+x':
            px = cx_disc + offset
            py = cy_disc
        elif direction == '-x':
            px = cx_disc - offset
            py = cy_disc
        elif direction == '+y':
            px = cx_disc
            py = cy_disc + offset
        elif direction == '-y':
            px = cx_disc
            py = cy_disc - offset
        else:
            px = cx_disc + offset
            py = cy_disc + offset
        sxp, syp, rp = rho3d_prod(px, py)
        sxc, syc, rc = rho3d_conic(px, py)
        pct = 100 * (rc - rp) / rp if rp > 1e-10 else 0
        print(f"({px:>6.2f},{py:>6.2f}) {sxp:>10.4f} {syp:>10.4f} {rp:>12.4e} "
              f"{sxc:>10.4f} {syc:>10.4f} {rc:>12.4e} {pct:>+9.2f}%")

# The critical range: where beta_scaled compact support end (rho3d = 9).
# Find pixel where rho3d_prod ~= 9 along +x direction.
print("\n--- Sweep across compact-support boundary (rho3d ≈ 9) ---")
for offset_x in np.arange(0, 15, 1):
    px = cx_disc + offset_x
    py = cy_disc
    sxp, syp, rp = rho3d_prod(px, py)
    sxc, syc, rc = rho3d_conic(px, py)
    marker = " <-- boundary (prod)" if 8 < rp < 10 else ""
    marker += " <== boundary (conic)" if 8 < rc < 10 else ""
    print(f"  Δx=+{offset_x:.1f}: prod={rp:>10.4f}  conic={rc:>10.4f}  Δ={100*(rc-rp)/rp if rp>0 else 0:+.1f}%{marker}")
