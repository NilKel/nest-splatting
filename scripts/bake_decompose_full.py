"""
Decompose an existing single-bake atlas into 4 per-level atlases
{A_0, A_1, A_2, A_3} such that  Σ_k A_k = T  to FP precision, where T is
the dequantised single-bake atlas. Per-bucket joint-finetune (Adam), saves
each per-level atlas as a torch dict keyed by Gaussian indices.

Outputs land in `<bake_dir>/decomposed/`:
  bucket_<rx>x<ry>.pt    — { 'A': [L, N_b, h, w, 3] FP16, 'gauss_idx': [N_b] }
  summary.json           — per-bucket recon PSNR + global PSNR vs atlas

Usage:
  python scripts/bake_decompose_full.py --bake_dir <path> [--max_iters 800]
                                        [--min_bucket_n 50] [--target_psnr 80]
"""
import argparse, glob, json, os, pickle, sys, time
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hash_encoder.modules import INGP
from hash_encoder.config  import Config
from arguments  import ModelParams
from scene      import Scene
from scene.gaussian_model import GaussianModel


def quat_to_rotcols(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    norm = (w*w + x*x + y*y + z*z + 1e-8).rsqrt()
    w, x, y, z = w*norm, x*norm, y*norm, z*norm
    r00 = 1 - 2*(y*y + z*z); r10 = 2*(x*y + w*z); r20 = 2*(x*z - w*y)
    r01 = 2*(x*y - w*z); r11 = 1 - 2*(x*x + z*z); r21 = 2*(y*z + w*x)
    return torch.stack([r00, r10, r20], dim=-1), torch.stack([r01, r11, r21], dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True,
                   help="Path to an existing baked_atlas/ directory.")
    p.add_argument("--model_path", default=None,
                   help="Parent of bake_dir, holds args.pkl/cfg/ngp_*.pth. "
                        "Default: dirname(bake_dir).")
    p.add_argument("--max_iters", type=int, default=800)
    p.add_argument("--target_psnr", type=float, default=80.0,
                   help="Stop a bucket's finetune once it clears this PSNR.")
    p.add_argument("--min_bucket_n", type=int, default=50)
    p.add_argument("--uv_extent",  type=float, default=4.0)
    args = p.parse_args()

    bake_dir = args.bake_dir
    model_path = args.model_path or os.path.dirname(os.path.normpath(bake_dir))
    out_dir = os.path.join(bake_dir, "decomposed")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Load model + atlas ----
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        train_args = pickle.load(f)
    train_args.model_path = model_path
    train_args.eval = True
    cfg_path = os.path.join(model_path, "config.yaml")
    cfg = Config(cfg_path) if os.path.exists(cfg_path) else Config(train_args.yaml)
    iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                    for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))
    print(f"[LOAD] model={model_path}  iter={iteration}  method={train_args.method}")

    ingp = INGP(cfg, args=train_args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)
    pm = ArgumentParser()
    dataset = ModelParams(pm, sentinel=True).extract(train_args)
    gaussians = GaussianModel(dataset.sh_degree)
    Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=train_args)
    if hasattr(train_args, 'kernel'): gaussians.kernel_type = train_args.kernel

    # Same prune used by bake_atlas — keeps Gauss indices aligned with rects.
    dead = (gaussians.get_opacity <= 0.005).squeeze(-1)
    keep = ~dead
    for a in ['_xyz', '_scaling', '_rotation', '_opacity']:
        t = getattr(gaussians, a, None)
        if t is not None and t.numel() > 0 and t.shape[0] == keep.shape[0]:
            setattr(gaussians, a, t[keep.to(t.device)])
    N = len(gaussians.get_xyz)
    print(f"[LOAD] {N:,} live Gausses (post-prune)")

    meta = json.load(open(os.path.join(bake_dir, "bake_meta.json")))
    aw, ah = int(meta["atlas_width"]), int(meta["atlas_height"])
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    uv_extent = float(meta.get("uv_extent", args.uv_extent))
    print(f"[LOAD] atlas {aw}×{ah}  scale={a_scale:.4f}  offset={a_off:.4f}  uv_extent={uv_extent}")

    atlas_u8 = torch.load(os.path.join(bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    rects = torch.load(os.path.join(bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    if rects.shape[1] >= 5 and (rects[:, 4] != 0).any():
        raise SystemExit("Multi-layer atlas not supported here.")
    u0 = rects[:, 0].long(); v0 = rects[:, 1].long()
    w_ = rects[:, 2].long(); h_ = rects[:, 3].long()
    assert rects.shape[0] == N, f"rects rows ({rects.shape[0]}) != live Gausses ({N})"

    # Bucket by (w, h) and skip skip-texture entries.
    key = (w_ * 100000 + h_).long()
    uniq, inv = key.unique(return_inverse=True)
    buckets = []
    for b in range(uniq.shape[0]):
        idx = (inv == b).nonzero(as_tuple=True)[0]
        ww, hh = int(w_[idx[0]].item()), int(h_[idx[0]].item())
        if ww == 0 or hh == 0:
            continue
        if idx.shape[0] < args.min_bucket_n:
            continue
        buckets.append((ww, hh, idx))
    buckets.sort(key=lambda t: -(t[0] * t[1] * t[2].shape[0]))
    print(f"[BUCKETS] {len(buckets)} usable buckets (≥ {args.min_bucket_n} Gausses)")

    # MLP / hashgrid plumbing — match benchmark_baked.py.
    mlp = ingp.mlp_fused.half().eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_in_pad = mlp[0].weight.shape[1]
    L = hash_dim // 4
    print(f"[MLP] hash_dim={hash_dim} = {L} levels × 4D, pad={mlp_in_pad}")

    centers = gaussians.get_xyz
    quats   = gaussians.get_rotation
    scales  = gaussians.get_scaling
    R0, R1  = quat_to_rotcols(quats)

    @torch.no_grad()
    def mlp_eval(hash_feat):
        mlp_in = torch.zeros(hash_feat.shape[0], mlp_in_pad, device='cuda', dtype=torch.float16)
        mlp_in[:, :hash_dim] = hash_feat[:, :hash_dim].to(torch.float16)
        return mlp(mlp_in)[:, :3].float()

    summary = {"buckets": [], "global_mse": 0.0, "global_count": 0}
    print(f"\n{'bucket':>10}  {'N':>7}  {'init PSNR':>10}  {'final PSNR':>10}  {'time s':>7}")
    t0 = time.time()
    # Memory cap per chunk: hash_full alone for 4000 Gausses × 64² = 16.4M
    # points × 16 features × 4 B ≈ 1 GB; plus the L copies of MLP output, the
    # joint-finetune A tensor + grad + 2 Adam moments. Cap at ~4 k for safety
    # on the biggest bucket.
    MAX_POINTS_PER_CHUNK = 4_000 * 64 * 64   # ≈ 16.4 M points
    for ww, hh, idx in buckets:
        n_b = idx.shape[0]
        npts = ww * hh
        chunk_size = max(1, min(n_b, MAX_POINTS_PER_CHUNK // max(npts, 1)))

        step_x = 2.0 * uv_extent / ww
        step_y = 2.0 * uv_extent / hh
        uu = (torch.arange(ww, dtype=torch.float32, device='cuda') + 0.5) * step_x - uv_extent
        vv = (torch.arange(hh, dtype=torch.float32, device='cuda') + 0.5) * step_y - uv_extent
        u_grid, v_grid = torch.meshgrid(uu, vv, indexing='ij')
        u_flat = u_grid.reshape(-1)
        v_flat = v_grid.reshape(-1)

        # Per-chunk: extract atlas patches, init A from MLP-masked, joint
        # finetune, append. Saves are concatenated per-bucket at the end.
        A_chunks = []
        gauss_idx_chunks = []
        total_se = 0.0; total_n = 0
        init_se = 0.0
        t_buck = time.time()
        for cs in range(0, n_b, chunk_size):
            ce = min(cs + chunk_size, n_b)
            idx_c = idx[cs:ce]
            n_c = idx_c.shape[0]

            # Extract atlas patches.
            u0i = u0[idx_c]; v0i = v0[idx_c]
            T = torch.empty(n_c, hh, ww, 3, device='cuda', dtype=torch.float32)
            for r in range(n_c):
                T[r] = atlas_u8[v0i[r]:v0i[r]+hh, u0i[r]:u0i[r]+ww].float()
            T = (T / 255.0 * a_scale + a_off)            # [n_c, hh, ww, 3]

            # xyz per (gauss, texel).
            c  = centers[idx_c]
            sx = scales[idx_c, 0:1]
            sy = scales[idx_c, 1:2]
            r0 = R0[idx_c]
            r1 = R1[idx_c]
            xyz = (c.unsqueeze(1)
                   + u_flat.unsqueeze(0).unsqueeze(-1) * (sx.unsqueeze(1) * r0.unsqueeze(1))
                   + v_flat.unsqueeze(0).unsqueeze(-1) * (sy.unsqueeze(1) * r1.unsqueeze(1)))
            xyz_flat = xyz.reshape(-1, 3)

            with torch.no_grad():
                hash_full = ingp._encode_3D(xyz_flat)
                A_init = torch.empty(L, n_c, hh, ww, 3, device='cuda', dtype=torch.float32)
                for k in range(L):
                    mask = torch.zeros_like(hash_full)
                    mask[:, k*4:(k+1)*4] = 1.0
                    T_k = (mlp_eval(hash_full * mask).reshape(n_c, ww, hh, 3)
                                                       .permute(0, 2, 1, 3).contiguous())
                    A_init[k] = T_k
                del hash_full, xyz, xyz_flat

            init_se += (A_init.sum(0) - T).pow(2).sum().item()
            A = A_init.clone().requires_grad_(True)
            opt = torch.optim.Adam([A], lr=5e-3)
            del A_init
            for it in range(args.max_iters):
                opt.zero_grad()
                loss = (A.sum(0) - T).pow(2).mean()
                loss.backward()
                opt.step()
                if (it + 1) % 50 == 0:
                    psnr = -10.0 * torch.log10(loss.detach().clamp_min(1e-20)).item()
                    if psnr >= args.target_psnr:
                        break
            chunk_se = (A.detach().sum(0) - T).pow(2).sum().item()
            total_se += chunk_se
            total_n  += n_c * hh * ww * 3

            A_chunks.append(A.detach().half().cpu())
            gauss_idx_chunks.append(idx_c.cpu())
            del A, T
            torch.cuda.empty_cache()

        A_full = torch.cat(A_chunks, dim=1)              # [L, n_b, hh, ww, 3]
        gauss_idx = torch.cat(gauss_idx_chunks, dim=0)
        init_psnr  = -10.0 * torch.log10(torch.tensor(init_se / total_n).clamp_min(1e-20)).item()
        final_psnr = -10.0 * torch.log10(torch.tensor(total_se / total_n).clamp_min(1e-20)).item()
        dt = time.time() - t_buck

        torch.save(
            {'A': A_full, 'gauss_idx': gauss_idx,
             'rect_wh': (ww, hh),
             'atlas_scale': a_scale, 'atlas_offset': a_off,
             'init_psnr_vs_atlas': float(init_psnr),
             'final_psnr_vs_atlas': float(final_psnr)},
            os.path.join(out_dir, f"bucket_{ww}x{hh}.pt"),
        )
        summary["buckets"].append({
            "rx": ww, "ry": hh, "n": n_b,
            "init_psnr_vs_atlas": float(init_psnr),
            "final_psnr_vs_atlas": float(final_psnr),
            "time_s": float(dt),
        })
        summary["global_mse"]  += total_se
        summary["global_count"] += total_n
        print(f"  {ww:>3}×{hh:<3}  {n_b:>7,}  {init_psnr:>10.2f}  {final_psnr:>10.2f}  {dt:>7.1f}")
        del A_full, gauss_idx
        torch.cuda.empty_cache()

    g_psnr = -10.0 * torch.log10(torch.tensor(summary["global_mse"] /
                                              max(summary["global_count"], 1)).clamp_min(1e-20)).item()
    summary["global_psnr_vs_atlas"] = float(g_psnr)
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {len(buckets)} buckets, total {time.time()-t0:.1f}s")
    print(f"   GLOBAL Σ A_k vs atlas PSNR = {g_psnr:.2f} dB  (target ≥ {args.target_psnr})")
    print(f"   saved to {out_dir}/")


if __name__ == "__main__":
    main()
