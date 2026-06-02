"""
Per-level decomposition experiment for the baked atlas pipeline.

For each sampled Gaussian, evaluates the residual MLP four ways:
  T_full     = MLP(z₀ ⊕ z₁ ⊕ z₂ ⊕ z₃)       # the current ground-truth bake
  T_k        = MLP(0  ⊕ … ⊕ z_k ⊕ … ⊕ 0)    # only level k unmasked
  T_naive    = Σ_k T_k                       # additive approximation
  T_jointFT  = Σ_k A_k after joint finetune  # per-level atlases jointly trained
                                              # to recover the true T_full

Reports PSNR(naive vs T_full), PSNR(jointFT vs T_full), and per-level
constancy (what fraction of Gausses have spatially-flat T_k — those
collapse to a single per-Gauss RGB instead of a rect).

Usage:
  python scripts/bake_decomposed_test.py --model_path <path> [--n_sample 2000]
                                         [--fixed_res 16] [--finetune_iters 500]
"""
import argparse, glob, os, pickle, sys, time
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hash_encoder.modules import INGP
from hash_encoder.config  import Config
from arguments  import ModelParams, get_combined_args
from argparse   import ArgumentParser, Namespace
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration",   type=int, default=-1)
    parser.add_argument("--n_sample",    type=int, default=2000,
                        help="Random subset of Gausses to evaluate on.")
    parser.add_argument("--fixed_res",   type=int, default=16,
                        help="UV grid resolution per Gauss (square).")
    parser.add_argument("--uv_extent",   type=float, default=4.0)
    parser.add_argument("--finetune_iters", type=int, default=500,
                        help="Adam steps for joint per-level atlas finetune.")
    parser.add_argument("--const_eps",   type=float, default=1.0/255,
                        help="Threshold for declaring T_k spatially flat "
                             "(max abs deviation from per-Gauss mean).")
    parser.add_argument("--variable_res", action="store_true",
                        help="After full-res finetune, also try Nyquist-sized "
                             "per-level resolutions (R, R/2, R/4, R/8 ...) "
                             "and re-finetune; report PSNR + storage budget.")
    parser.add_argument("--cluster_ks",  type=int, nargs="+",
                        default=[8, 32, 128, 512, 2048],
                        help="K values to try when clustering per-level "
                             "atlases across Gausses (K-means).")
    parser.add_argument("--cluster_iters", type=int, default=30,
                        help="Lloyd K-means iterations per K.")
    bargs = parser.parse_args()

    model_path = bargs.model_path
    # Load training args + config exactly like benchmark_baked.py does.
    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    args.model_path = model_path
    args.eval = True
    cfg_path = os.path.join(model_path, "config.yaml")
    cfg = Config(cfg_path) if os.path.exists(cfg_path) else Config(args.yaml)

    iteration = bargs.iteration
    if iteration == -1:
        files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in files)
    print(f"[LOAD] iter={iteration}, method={args.method}")

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(model_path, iteration)
    ingp.set_active_levels(iteration)

    parser_m = ArgumentParser()
    dataset = ModelParams(parser_m, sentinel=True).extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, full_args=args)
    if hasattr(args, 'kernel'): gaussians.kernel_type = args.kernel

    # Prune dead Gausses (same as bake script).
    dead = (gaussians.get_opacity <= 0.005).squeeze(-1)
    keep = ~dead
    for a in ['_xyz', '_scaling', '_rotation', '_opacity']:
        t = getattr(gaussians, a, None)
        if t is not None and t.numel() > 0 and t.shape[0] == keep.shape[0]:
            setattr(gaussians, a, t[keep.to(t.device)])
    N = len(gaussians.get_xyz)
    print(f"[LOAD] {N:,} live Gausses")

    # Random subset.
    g = torch.Generator(device='cuda').manual_seed(0)
    sel = torch.randperm(N, generator=g, device='cuda')[:bargs.n_sample]
    centers = gaussians.get_xyz[sel]
    quats   = gaussians.get_rotation[sel]
    scales  = gaussians.get_scaling[sel]
    R0, R1  = quat_to_rotcols(quats)
    Nsub = sel.shape[0]

    # MLP setup — mirror bake_atlas (FP16 weights, zero-pad input).
    mlp = ingp.mlp_fused.half().eval()
    hash_dim = ingp.mlp_fused_hash_dim
    mlp_in_pad = mlp[0].weight.shape[1]
    assert hash_dim % 4 == 0, f"hash_dim={hash_dim} not a multiple of 4"
    L = hash_dim // 4         # number of hash levels (4D each)
    print(f"[LOAD] hash_dim={hash_dim} = {L} levels × 4D ; MLP pad={mlp_in_pad}")

    # UV grid (uniform, fixed_res square).
    R = bargs.fixed_res
    step = 2.0 * bargs.uv_extent / R
    coords = (torch.arange(R, dtype=torch.float32, device='cuda') + 0.5) * step - bargs.uv_extent
    uu, vv = torch.meshgrid(coords, coords, indexing='ij')
    uvw = torch.stack([uu.reshape(-1), vv.reshape(-1)], 1)   # [R*R, 2]
    npts = R * R

    # xyz per (gauss, texel).
    sx = scales[:, 0:1]; sy = scales[:, 1:2]
    xyz = (centers.unsqueeze(1)
           + uvw[:, 0:1].T.unsqueeze(-1).unsqueeze(0) * (sx.unsqueeze(1) * R0.unsqueeze(1))
           + uvw[:, 1:2].T.unsqueeze(-1).unsqueeze(0) * (sy.unsqueeze(1) * R1.unsqueeze(1)))
    xyz = xyz.reshape(-1, 3)      # [Nsub * npts, 3]

    @torch.no_grad()
    def mlp_eval(hash_feat_16):
        mlp_in = torch.zeros(hash_feat_16.shape[0], mlp_in_pad,
                             device='cuda', dtype=torch.float16)
        mlp_in[:, :hash_dim] = hash_feat_16[:, :hash_dim].to(torch.float16)
        # MLP outputs ≥3D; only the first 3 dims are the RGB residual
        # (matches benchmark_baked.py: `rgb_residual = mlp_out[:, :3]`).
        return mlp(mlp_in)[:, :3].float()

    with torch.no_grad():
        hash_full = ingp._encode_3D(xyz)        # [Nsub*npts, hash_dim]
        T_full = mlp_eval(hash_full).reshape(Nsub, R, R, 3)

        # Per-level masked eval.
        T_per_level = []
        for k in range(L):
            mask = torch.zeros_like(hash_full)
            mask[:, k*4:(k+1)*4] = 1.0
            T_k = mlp_eval(hash_full * mask).reshape(Nsub, R, R, 3)
            T_per_level.append(T_k)
        T_stack = torch.stack(T_per_level, 0)   # [L, Nsub, R, R, 3]

    # --- Naive additive baseline ---
    T_naive = T_stack.sum(0)
    err_naive = (T_naive - T_full)
    mse_naive = (err_naive ** 2).mean().item()
    psnr_naive = -10.0 * torch.log10(torch.tensor(max(mse_naive, 1e-20))).item()

    # --- Per-level constancy stats (storage-saving signal) ---
    print(f"\n--- Per-level structure on {Nsub:,} Gausses @ {R}×{R} (uv_extent={bargs.uv_extent}) ---")
    print(f"{'level':>5} {'finest-first':>12} {'mean |T_k|':>12} {'flat-Gauss %':>14}  {'max axis range':>16}")
    for k, T_k in enumerate(T_per_level):
        per_gauss_mean   = T_k.mean(dim=(1, 2), keepdim=True)                 # [Nsub,1,1,3]
        per_gauss_devmax = (T_k - per_gauss_mean).abs().amax(dim=(1, 2, 3))   # [Nsub]
        flat_frac = (per_gauss_devmax < bargs.const_eps).float().mean().item()
        meanmag = T_k.abs().mean().item()
        print(f"{k:>5}  level k={k:<6}  {meanmag:>12.5f}  {100*flat_frac:>12.1f} %  {per_gauss_devmax.max().item():>14.4f}")

    print(f"\n--- Naive additive decomposition ---")
    print(f"   PSNR(Σ T_k vs T_full) = {psnr_naive:6.2f} dB  (MSE={mse_naive:.4e})")

    # --- Joint finetune ---
    if bargs.finetune_iters > 0:
        print(f"\n--- Joint finetune: optimize per-level atlases A_k so Σ A_k ≈ T_full ---")
        A = torch.stack(T_per_level, 0).clone().contiguous().requires_grad_(True)   # [L, Nsub, R, R, 3]
        opt = torch.optim.Adam([A], lr=5e-3)
        t0 = time.time()
        for it in range(bargs.finetune_iters):
            opt.zero_grad()
            recon = A.sum(0)
            loss = (recon - T_full).pow(2).mean()
            loss.backward()
            opt.step()
            if it == 0 or it == bargs.finetune_iters - 1 or (it + 1) % max(1, bargs.finetune_iters // 5) == 0:
                psnr = -10.0 * torch.log10(loss.detach().clamp_min(1e-20)).item()
                print(f"   iter {it+1:>4}/{bargs.finetune_iters}  loss={loss.item():.4e}  PSNR={psnr:6.2f} dB")
        psnr_ft = -10.0 * torch.log10(loss.detach().clamp_min(1e-20)).item()
        print(f"   final PSNR(jointFT vs T_full) = {psnr_ft:6.2f} dB   ({time.time()-t0:.1f}s)")

        # Constancy after finetune.
        print(f"\n--- Per-level structure AFTER finetune (does A_k stay flat?) ---")
        with torch.no_grad():
            for k in range(L):
                A_k = A[k]
                m = A_k.mean(dim=(1, 2), keepdim=True)
                dev = (A_k - m).abs().amax(dim=(1, 2, 3))
                flat_frac = (dev < bargs.const_eps).float().mean().item()
                meanmag = A_k.abs().mean().item()
                print(f"   level k={k}:  mean|A_k|={meanmag:.5f}  flat-Gauss={100*flat_frac:.1f}%  "
                      f"max axis dev={dev.max().item():.4f}")

        # -------- Variable-res lite test (Nyquist-sized per level) --------
        if bargs.variable_res:
            print(f"\n--- Variable-resolution per-level finetune ---")
            print(f"   Hashgrid voxel side decreases finest→coarsest, so coarse levels")
            print(f"   carry strictly band-limited signal — they can be sampled sparser")
            print(f"   without information loss.")
            # Powers-of-two grid steps. Each step k halves resolution.
            step_pow2_options = [
                ("R / 2^k     ", [max(2, R >> k) for k in range(L)]),         # 16, 8, 4, 4
                ("R / (k+1)≈  ", [R, max(4, R // 2), max(4, R // 3), max(4, R // 4)]),
                ("R, R, R/2, R/4", [R, R, max(4, R // 2), max(4, R // 4)]),
            ]
            single_texels = R * R
            for label, res_per_level in step_pow2_options:
                # Downsample current A to per-level resolutions via avg-pool, then
                # joint-finetune to recover T_full.
                A_down_list = []
                for k in range(L):
                    rk = res_per_level[k]
                    if rk == R:
                        A_down_list.append(A[k].detach().clone())
                    else:
                        # mean-pool along the two spatial dims.
                        factor = R // rk
                        Ak = A[k].detach()                                  # [Nsub, R, R, 3]
                        Ak = Ak.unfold(1, factor, factor).unfold(2, factor, factor)
                        Ak = Ak.mean(dim=(-1, -2))                          # [Nsub, rk, rk, 3]
                        A_down_list.append(Ak.contiguous())
                # Joint finetune at per-level reduced res.
                Ap = [a.clone().requires_grad_(True) for a in A_down_list]
                optp = torch.optim.Adam(Ap, lr=5e-3)
                for it in range(bargs.finetune_iters):
                    optp.zero_grad()
                    # Upsample each level to R via nearest, then sum.
                    recon = 0
                    for k, a in enumerate(Ap):
                        rk = res_per_level[k]
                        if rk == R:
                            up = a
                        else:
                            up = torch.nn.functional.interpolate(
                                a.permute(0, 3, 1, 2), size=(R, R),
                                mode='bilinear', align_corners=False
                            ).permute(0, 2, 3, 1)
                        recon = recon + up
                    loss = (recon - T_full).pow(2).mean()
                    loss.backward()
                    optp.step()
                psnr_lite = -10.0 * torch.log10(loss.detach().clamp_min(1e-20)).item()
                multi_texels = sum(r * r for r in res_per_level)
                ratio = multi_texels / single_texels
                print(f"   res {res_per_level}  →  PSNR={psnr_lite:6.2f} dB   "
                      f"texels/Gauss: {multi_texels} vs single-bake {single_texels}  ({ratio:.2f}×)")

        # ============================================================
        # CROSS-GAUSS CLUSTERING (the user's question 3 — the real win)
        # ============================================================
        # For each level, run K-means over the Nsub per-Gauss patches and
        # report reconstruction PSNR if every Gauss stores only its
        # cluster id (codebook lookup). If a level clusters well, that
        # level's storage shrinks from `Nsub × R²·3` to `K × R²·3` (the
        # codebook) plus `Nsub × log2(K)` bits of indices — typically
        # < 1/100th the per-Gauss cost.
        print(f"\n--- Cross-Gauss clustering per level (K-means, "
              f"{bargs.cluster_iters} iters) ---")
        D = R * R * 3
        with torch.no_grad():
            single_bake_bytes_per_gauss = D * 1   # single BC7 byte per texel × 3 unioned to ~1 per channel; treat as floor
            print(f"   Reference per-Gauss single-bake: R²·3 = {D} BC7 bytes (no clustering)\n")
            print(f"   {'level':>5} {'K':>5} {'recon PSNR (T_k vs Ã_k)':>26} "
                  f"{'codebook (KB)':>14} {'idx bits':>9} "
                  f"{'effective bytes/Gauss':>22}")
            # Also the joint reconstruction PSNR (Σ_k Ã_k vs T_full) per K.
            full_recon_per_K = {}
            for K_try in bargs.cluster_ks:
                if K_try > Nsub:
                    continue
                recons_per_level = []
                for k in range(L):
                    X = A[k].detach().reshape(Nsub, D).contiguous()    # [Nsub, D]
                    # K-means: init = random Nsub indices.
                    perm = torch.randperm(Nsub, device='cuda')[:K_try]
                    centroids = X[perm].clone()
                    for _ in range(bargs.cluster_iters):
                        # Assign: argmin over centroids.
                        # |x - c|² = |x|² - 2 x·c + |c|² → for argmin, only -2x·c + |c|² matters.
                        d = -2.0 * X @ centroids.T + (centroids * centroids).sum(1)
                        ass = d.argmin(1)                              # [Nsub]
                        # Update: per-cluster mean.
                        new_c = torch.zeros_like(centroids)
                        cnt   = torch.zeros(K_try, device='cuda')
                        new_c.index_add_(0, ass, X)
                        cnt.index_add_(0, ass, torch.ones(Nsub, device='cuda'))
                        non_empty = cnt > 0
                        new_c[non_empty] = new_c[non_empty] / cnt[non_empty].unsqueeze(1)
                        # Re-seed empty clusters from random samples.
                        if (~non_empty).any():
                            new_c[~non_empty] = X[torch.randperm(Nsub, device='cuda')[:(~non_empty).sum()]]
                        centroids = new_c
                    # Final assignment + reconstruction.
                    d = -2.0 * X @ centroids.T + (centroids * centroids).sum(1)
                    ass = d.argmin(1)
                    recon = centroids[ass]                              # [Nsub, D]
                    mse_k = (recon - X).pow(2).mean().item()
                    psnr_k = -10.0 * torch.log10(torch.tensor(max(mse_k, 1e-20))).item()
                    codebook_kb = K_try * D * 4 / 1024.0  # FP32 codebook (BC7 → ÷8)
                    idx_bits = int(torch.tensor(K_try).float().log2().ceil().item())
                    bytes_per = (K_try * D / Nsub) + (idx_bits / 8.0)
                    print(f"   {k:>5} {K_try:>5} {psnr_k:>20.2f} dB  "
                          f"{codebook_kb/8:>14.2f}* {idx_bits:>9} {bytes_per:>22.1f}")
                    recons_per_level.append(recon.reshape(Nsub, R, R, 3))
                # Combined reconstruction.
                recon_full = sum(recons_per_level)
                mse_full = (recon_full - T_full).pow(2).mean().item()
                psnr_full = -10.0 * torch.log10(torch.tensor(max(mse_full, 1e-20))).item()
                full_recon_per_K[K_try] = psnr_full
                print(f"   K={K_try:<5}  Σ Ã_k vs T_full  PSNR = {psnr_full:.2f} dB\n")
            print(f"   * codebook size shown is BC7-equivalent (FP32 size ÷ 8). "
                  f"R²·3 BC7 bytes/Gauss = {D} single-bake reference.")

if __name__ == "__main__":
    main()
