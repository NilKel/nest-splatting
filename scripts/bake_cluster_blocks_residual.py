"""
Residual VQ (L stages of greedy K-means) on the single-bake atlas
blocks. Same recurrence as Compact3DGS / RVQ-Gaussians, but post-hoc:

  stage l (1 ≤ l ≤ L):
    R_l       = X − Σ_{j<l} cents_j[ass_j]
    cents_l,_ = KMeans(R_l, K_l)
    ass_l     = assign(R_l, cents_l)
  recon      = Σ_l cents_l[ass_l]

Storage:
  codebooks:  Σ_l K_l × 48          bytes (uint8 BC7-equiv)
  indices  :  N × Σ_l ceil(log2 K_l) bits

Effective codebook size = Π_l K_l, far above single-stage practical K
at the same byte budget. Greedy nesting underperforms joint training
(the paper's stop-gradient loss) by typically 0.5–1 dB at matched L,
but doesn't require re-baking.

`--K_stages K1 K2 ...` controls L and the per-stage codebook sizes.
Default sweep below covers L=1..4 at several total-byte targets.

Reference single-stage (for the sweep table):
  K=65 536  → 39.11 dB @ 12.94 %  (29.1 MB)
  K=131 072 → 39.75 dB @ 14.17 %  (31.9 MB)
  K=262 144 → 40.17 dB @ 15.84 %  (35.6 MB)
  K=524 288 → 40.60 dB @ 18.40 %  (41.4 MB)
"""
import argparse, json, math, os, time
import torch


def kmeans_chunked(X, K, iters=15, seed=0, weights=None, dist_chunk=None):
    if dist_chunk is None:
        dist_chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    N, D = X.shape
    if K >= N:
        return X.clone(), torch.arange(N, device=X.device)
    g = torch.Generator(device=X.device).manual_seed(seed)
    cents = X[torch.randperm(N, generator=g, device=X.device)[:K]].clone()
    if weights is not None:
        weights = weights.float()
    ass = torch.empty(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        cn2 = (cents * cents).sum(1)
        for s in range(0, N, dist_chunk):
            e = min(s + dist_chunk, N)
            d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
            ass[s:e] = d.argmin(1)
            del d
        new = torch.zeros_like(cents)
        cnt = torch.zeros(K, device=X.device)
        if weights is not None:
            new.index_add_(0, ass, X * weights.unsqueeze(1))
            cnt.index_add_(0, ass, weights)
        else:
            new.index_add_(0, ass, X)
            cnt.index_add_(0, ass, torch.ones(N, device=X.device))
        ne = cnt > 0
        new[ne] = new[ne] / cnt[ne].unsqueeze(1)
        if (~ne).any():
            rs = torch.randperm(N, generator=g, device=X.device)[:(~ne).sum()]
            new[~ne] = X[rs]
        cents = new
    cn2 = (cents * cents).sum(1)
    for s in range(0, N, dist_chunk):
        e = min(s + dist_chunk, N)
        d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return cents, ass


def assign_full(X, cents, n_used):
    cn2 = (cents * cents).sum(1)
    K = cents.shape[0]
    ass = torch.empty(n_used, dtype=torch.long, device=X.device)
    chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    for s in range(0, n_used, chunk):
        e = min(s + chunk, n_used)
        d = -2.0 * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return ass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--configs", type=str, nargs="+",
                   default=[
                       # Effective K @ ~comparable storage
                       "64,16384",            # L=2, eff K = 1M
                       "256,1024",            # L=2, eff K = 262k
                       "256,256",             # L=2, eff K = 65k
                       "128,128,128",         # L=3, eff K = 2M
                       "256,256,256",         # L=3, eff K = 16M
                       "64,64,64,64",         # L=4, eff K = 16M
                       "128,128,128,128",     # L=4, eff K = 268M
                       "256,256,256,256",     # L=4, eff K = 4.3B (paper config)
                       "64,64,64,64,64",      # L=5, eff K = 1B
                   ],
                   help="Comma-separated K per stage, one per config to sweep.")
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_500_000)
    args = p.parse_args()
    B = args.block

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    H, W, _ = atlas_u8.shape
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    if rects.shape[1] >= 5 and (rects[:, 4] != 0).any():
        raise SystemExit("Multi-layer atlas not supported.")
    u0, v0 = rects[:, 0].long(), rects[:, 1].long()
    w_, h_ = rects[:, 2].long(), rects[:, 3].long()
    used = torch.zeros((H // B, W // B), dtype=torch.bool, device='cuda')
    for i in range(rects.shape[0]):
        ww = int(w_[i].item()); hh = int(h_[i].item())
        if ww == 0 or hh == 0: continue
        bu = int(u0[i].item()) // B
        bv = int(v0[i].item()) // B
        used[bv:bv + hh // B, bu:bu + ww // B] = True
    used_idx = used.nonzero(as_tuple=False)
    n_used = used_idx.shape[0]
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off
    atlas_blocks = atlas_f.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
    X = atlas_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used, -1).contiguous()
    del atlas_blocks, atlas_f
    D = X.shape[1]
    print(f"[BLOCKS] {n_used:,} used 4×4 blocks; X = {tuple(X.shape)} "
          f"({X.element_size()*X.numel()/1024/1024:.0f} MB)")

    single_bc7_mb = n_used * B * B / 1024 / 1024
    print(f"[REF] single-bake BC7: {single_bc7_mb:.2f} MB\n")

    # Sub-sample for K-means fit. Same indices used for both stages so
    # the residual sub-sample matches the stage-1 fit set.
    if n_used > args.max_blocks_for_kmeans:
        perm = torch.randperm(n_used, device='cuda')[:args.max_blocks_for_kmeans]
        X_fit = X[perm]
    else:
        perm = None
        X_fit = X

    print(f"   {'config (K_l per stage)':>28}  {'L':>2} {'eff K':>14}  "
          f"{'codebook KB':>11}  {'idx MB':>8}  {'total MB':>9}  "
          f"{'vs single':>10}  {'recon PSNR':>11}")

    for cfg in args.configs:
        Ks = [int(x) for x in cfg.split(",")]
        L = len(Ks)
        t0 = time.time()
        # Greedy L-stage residual K-means.
        recon_running = torch.zeros_like(X)               # Σ_{j<l} cents_j[ass_j]
        cents_list = []
        ass_list = []
        for l, K_l in enumerate(Ks):
            # Residual at this stage = X − recon_running.
            R = X - recon_running
            R_fit = R[perm] if perm is not None else R
            cents, _ = kmeans_chunked(R_fit, K_l, iters=args.iters)
            ass = assign_full(R, cents, n_used)
            cents_list.append(cents)
            ass_list.append(ass)
            # Update recon_running in chunks (don't allocate another N×D tensor).
            chunk = max(2_000, min(2_000_000, 500_000_000 // (K_l * 4)))
            for s in range(0, n_used, chunk):
                e = min(s + chunk, n_used)
                recon_running[s:e] = recon_running[s:e] + cents[ass[s:e]]
            del R, R_fit

        # Final PSNR vs X.
        se = 0.0
        for s in range(0, n_used, 500_000):
            e = min(s + 500_000, n_used)
            se += (recon_running[s:e] - X[s:e]).pow(2).sum().item()
        mse = se / (n_used * D)
        psnr = -10.0 * math.log10(max(mse, 1e-20))

        codebook_bytes = sum(K_l for K_l in Ks) * B * B
        bits_per = sum(math.ceil(math.log2(max(K_l, 2))) for K_l in Ks)
        idx_bytes = (n_used * bits_per + 7) // 8
        total_mb = (codebook_bytes + idx_bytes) / 1024 / 1024
        eff_K = 1
        for K_l in Ks:
            eff_K *= K_l
        dt = time.time() - t0
        cfg_str = "×".join(str(K_l) for K_l in Ks)
        print(f"   {cfg_str:>28}  {L:>2} {eff_K:>14,}  "
              f"{codebook_bytes/1024:>9.1f}KB  {idx_bytes/1024/1024:>6.2f}MB  "
              f"{total_mb:>9.3f}  {total_mb/single_bc7_mb*100:>8.2f}%  "
              f"{psnr:>9.2f} dB   ({dt:.1f}s)")
        del recon_running, cents_list, ass_list
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
