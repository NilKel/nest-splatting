#
# Camera pose refinement (`--3rgs`) — the "sfm" core of 3R-GS
# (Huang et al. 2025, arXiv:2504.04294), ported onto this repo's 2DGS/INRIA
# rasterizers.
#
# WHY THE PORT IS NOT A DIRECT COPY
# ---------------------------------
# 3R-GS is built on gsplat, whose CUDA backward returns a gradient w.r.t. the
# camera `viewmats`. The rasterizers in this repo (diff_surfel_3D_sh_res,
# diff_surfel_mixed_3d, ...) treat the camera (viewmatrix / projmatrix / campos)
# as CONSTANTS — their autograd `backward` emits grads for means3D, means2D,
# sh, scales, rotations, opacities, but NOTHING for the pose. So we cannot make
# `world_view_transform` a learnable leaf and expect a gradient.
#
# Instead we route the pose gradient through the geometry the kernel DOES
# differentiate (means3D + rotations). For a per-camera rigid pose delta `Td`
# right-multiplied onto camera-to-world (exactly 3R-GS's `CameraOptModule`:
# C2W' = C2W @ Td), rendering the scene through the ORIGINAL camera `W2V` but
# with every Gaussian rigidly transformed by
#
#       M = C2W @ inv(Td) @ W2V          (world-frame rigid transform)
#       x'  = M_rot @ x + M_t
#       q'  = quat(M_rot) ⊗ q            (surfel orientation)
#
# is mathematically identical to moving the camera by Td (derivation: we want
# W2V·x' = W2V'·x = inv(Td)·W2V·x  ⇒  x' = C2W·inv(Td)·W2V·x). Td is zero-init,
# so M = I and the iter-0 render is byte-identical to no pose opt. Gradients
# flow to Td (hence the per-camera 9D embedding) via dL/dmeans3D + dL/drotations.
# No CUDA rebuild; works for every method (3D_SH_res, res_3d_paired, mixed, ...).
#
# Conventions in this repo (INRIA / 2DGS):
#   camera.world_view_transform == getWorld2View2(R, T).T   (i.e. W2V transposed,
#       row-vector convention).  So the column-vector W2V = world_view_transform.T
#   camera.camera_center        == inv(world_view_transform)[3, :3] == C2W[:3, 3]
#   COLMAP images.txt extrinsics: X_cam = R_wc·X_world + t,  R_wc = W2V[:3,:3]

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Differentiable rotation helpers (Hamilton quaternions, [w, x, y, z] layout,
# matching utils.general_utils.build_rotation).
# --------------------------------------------------------------------------- #
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D continuous rotation (Zhou et al. 2019) -> 3x3, via Gram-Schmidt.
    Verbatim from 3R-GS / pytorch3d."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """3x3 rotation -> [w, x, y, z] unit quaternion. Differentiable (pytorch3d).
    M_rot stays ≈ I during pose refinement, so the trace branch is selected and
    this is numerically stable."""
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(*batch_dim, 4)
    return out.squeeze(0) if squeeze else out


def quaternion_multiply(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product q ⊗ r ([w,x,y,z]). Rotation composition R(q⊗r) = R(q)·R(r).
    q is (4,) (the per-camera M_rot quaternion), r is (N, 4) (surfel quats)."""
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return torch.stack([w, x, y, z], dim=-1)


# --------------------------------------------------------------------------- #
# The per-camera pose-delta module (3R-GS "sfm" parametrization).
# --------------------------------------------------------------------------- #
class CameraPoseOpt(nn.Module):
    """Per-camera learnable rigid pose delta: 3D translation + 6D rotation, zero
    init. Identical parametrization to 3R-GS's `CameraOptModule` (sfm mode)."""

    def __init__(self, num_cameras: int):
        super().__init__()
        self.num_cameras = num_cameras
        # cols 0:3 = translation dx (camera frame), 3:9 = 6D rotation delta.
        self.embeds = nn.Embedding(num_cameras, 9)
        nn.init.zeros_(self.embeds.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def _idx(self, idx, device):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, device=device, dtype=torch.long)
        return idx.to(device)

    def delta_matrix(self, idx) -> torch.Tensor:
        """Td (4x4): C2W' = C2W @ Td. Zero-init ⇒ Td = I."""
        p = self.embeds(self._idx(idx, self.embeds.weight.device))  # (9,)
        dx, drot = p[:3], p[3:]
        Rd = rotation_6d_to_matrix(drot + self.identity)            # (3,3)
        Td = torch.eye(4, device=p.device, dtype=p.dtype).clone()
        Td[:3, :3] = Rd
        Td[:3, 3] = dx
        return Td

    def correction(self, world_view_transform: torch.Tensor, idx):
        """Differentiable world-frame rigid transform that emulates moving camera
        `idx` by its current pose delta. Returns (M_rot[3,3], M_t[3], q_M[4])."""
        W2V = world_view_transform.transpose(0, 1)   # col-vector world->cam (const)
        C2W = torch.inverse(W2V)                      # const
        Td = self.delta_matrix(idx)                   # carries grad to embeds
        M = C2W @ torch.inverse(Td) @ W2V             # world-frame rigid
        M_rot = M[:3, :3]
        M_t = M[:3, 3]
        q_M = matrix_to_quaternion(M_rot)
        return M_rot, M_t, q_M

    @torch.no_grad()
    def refined_world2cam(self, world_view_transform: torch.Tensor, idx):
        """Refined COLMAP-convention extrinsics (R_wc, t) after applying delta:
        X_cam = R_wc·X_world + t."""
        W2V = world_view_transform.transpose(0, 1)
        C2W = torch.inverse(W2V)
        Td = self.delta_matrix(idx)
        C2W_ref = C2W @ Td
        W2V_ref = torch.inverse(C2W_ref)
        return W2V_ref[:3, :3], W2V_ref[:3, 3]


def _rotmat2qvec(R):
    """[w,x,y,z] quaternion from a 3x3 rotation (numpy), COLMAP convention."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


@torch.no_grad()
def export_refined_poses(pose_opt: CameraPoseOpt, cameras, name_to_idx, out_dir):
    """Write the refined extrinsics for every train camera:
       <out_dir>/refined_poses.npz   (authoritative: names, R_wc, t, C2W, delta)
       <out_dir>/images_refined.txt  (best-effort COLMAP images.txt; intrinsics
                                       unchanged so reuse the original cameras.txt)
    The trained PLY is itself the *improved reconstruction* — it was optimized
    under these corrected poses. This export lets you re-feed the corrected
    cameras into a downstream COLMAP/MVS pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    # One Camera object per unique image_name (any resolution scale carries the
    # same extrinsics).
    seen, rows = {}, []
    for cam in cameras:
        nm = cam.image_name
        if nm in seen or nm not in name_to_idx:
            continue
        seen[nm] = True
        idx = name_to_idx[nm]
        R_wc, t = pose_opt.refined_world2cam(cam.world_view_transform, idx)
        R_wc = R_wc.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        C2W = np.linalg.inv(np.block([[R_wc, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        delta = pose_opt.embeds(torch.tensor(idx, device=pose_opt.embeds.weight.device)).detach().cpu().numpy()
        cam_id = int(getattr(cam, "colmap_id", idx + 1) or (idx + 1))
        rows.append((nm, idx, cam_id, R_wc, t, C2W, delta))

    rows.sort(key=lambda r: r[1])
    names = np.array([r[0] for r in rows], dtype=object)
    R_all = np.stack([r[3] for r in rows]) if rows else np.zeros((0, 3, 3))
    t_all = np.stack([r[4] for r in rows]) if rows else np.zeros((0, 3))
    c2w_all = np.stack([r[5] for r in rows]) if rows else np.zeros((0, 4, 4))
    delta_all = np.stack([r[6] for r in rows]) if rows else np.zeros((0, 9))
    np.savez(os.path.join(out_dir, "refined_poses.npz"),
             names=names, R_wc=R_all, t=t_all, c2w=c2w_all, delta=delta_all)

    lines = ["# Refined COLMAP images.txt (poses optimized by --3rgs)",
             "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "#   (empty POINTS2D line)"]
    for i, (nm, idx, cam_id, R_wc, t, _c2w, _d) in enumerate(rows):
        q = _rotmat2qvec(R_wc)
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam_id} {nm}")
        lines.append("")
    with open(os.path.join(out_dir, "images_refined.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Magnitude summary (handy sanity print).
    if rows:
        dt = np.linalg.norm(delta_all[:, :3], axis=1)
        drot = np.linalg.norm(delta_all[:, 3:], axis=1)
        return dict(n=len(rows),
                    trans_mean=float(dt.mean()), trans_max=float(dt.max()),
                    rot6d_mean=float(drot.mean()), rot6d_max=float(drot.max()),
                    out_dir=out_dir)
    return dict(n=0, out_dir=out_dir)
