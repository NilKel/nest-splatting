"""
Product Quantization (PQ) on the single-bake atlas blocks.

Split each 48-D block into M contiguous sub-vectors of (48/M) dims each,
cluster each sub-vector independently with K codewords, recon = concat
of per-sub-vector recons.

Default M=4 → 4 sub-vectors of 12-D each. With the block flattened as
[pixel-y outermost, pixel-x, RGB innermost], each 12-D chunk is one
row of the 4×4 block × 3 channels (RGB triplets kept together).

Storage:
  codebook: M × K × (48/M) bytes        (M independent codebooks)
  indices : M × N × ceil(log2 K) bits   (one index per sub-vector per block)
Effective codebook = K^M, much higher than single VQ at matched bytes.

Sweep: --M (default 4) × --Ks (default several K).
"""
import argparse, json, math, os, time
import torch


def kmeans_chunked(X, K, iters=15, seed=0, dist_chunk=None):
    if dist_chunk is None:
        dist_chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    N, D = X.shape
    if K >= N:
        return X.clone(), torch.arange(N, device=X.device)
    g = torch.Generator(device=X.device).manual_seed(seed)
    cents = X[torch.randperm(N, generator=g, device=X.device)[:K]].clone()
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
    p.add_argument("--M", type=int, default=4,
                   help="Number of PQ sub-vectors (each is 48/M dim).")
    p.add_argument("--Ks", type=int, nargs="+",
                   default=[64, 256, 1024, 4096, 16384, 65536, 262144])
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_500_000)
    args = p.parse_args()
    B = args.block
    M = args.M
    D = B * B * 3
    if D % M != 0:
        raise SystemExit(f"48 must be divisible by M; got M={M}")
    D_sub = D // M

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    H, W, _ = atlas_u8.shape
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
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
    X = atlas_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used, D).contiguous()
    del atlas_blocks, atlas_f
    print(f"[BLOCKS] {n_used:,} used 4×4 blocks; X={tuple(X.shape)}; "
          f"PQ M={M} → {D_sub}-D × {M} sub-vectors")

    single_bc7_mb = n_used * B * B / 1024 / 1024
    print(f"[REF] single-bake BC7: {single_bc7_mb:.2f} MB\n")

    # Build M contiguous views into X. (No copy.)
    X_M = X.view(n_used, M, D_sub)
    if n_used > args.max_blocks_for_kmeans:
        perm = torch.randperm(n_used, device='cuda')[:args.max_blocks_for_kmeans]
    else:
        perm = None

    print(f"   {'K':>8}  {'sub-cbook':>10}  {'cbook total':>11}  "
          f"{'idx MB':>8}  {'total MB':>9}  {'vs single':>10}  "
          f"{'eff K':>14}  {'recon PSNR':>11}")
    for K in args.Ks:
        t0 = time.time()
        recon = torch.empty_like(X)
        recon_M = recon.view(n_used, M, D_sub)
        for m in range(M):
            Xm = X_M[:, m].contiguous()                              # [N, D_sub]
            Xm_fit = Xm[perm] if perm is not None else Xm
            cents_m, _ = kmeans_chunked(Xm_fit, K, iters=args.iters)
            ass_m = assign_full(Xm, cents_m, n_used)
            recon_M[:, m] = cents_m[ass_m]
            del Xm, Xm_fit, cents_m, ass_m
            torch.cuda.empty_cache()

        # PSNR vs X.
        se = 0.0
        for s in range(0, n_used, 500_000):
            e = min(s + 500_000, n_used)
            se += (recon[s:e] - X[s:e]).pow(2).sum().item()
        mse = se / (n_used * D)
        psnr = -10.0 * math.log10(max(mse, 1e-20))

        sub_cbook_bytes = K * D_sub                                  # 1 codebook, uint8 BC7-equiv
        cbook_total_bytes = M * sub_cbook_bytes
        bits_per = M * math.ceil(math.log2(max(K, 2)))
        idx_bytes = (n_used * bits_per + 7) // 8
        total_mb = (cbook_total_bytes + idx_bytes) / 1024 / 1024
        eff_K = K ** M
        dt = time.time() - t0
        print(f"   {K:>8,}  {sub_cbook_bytes/1024:>8.2f}KB  "
              f"{cbook_total_bytes/1024:>9.2f}KB  "
              f"{idx_bytes/1024/1024:>6.2f}MB  "
              f"{total_mb:>9.3f}  {total_mb/single_bc7_mb*100:>8.2f}%  "
              f"{eff_K:>14,}  {psnr:>9.2f} dB   ({dt:.1f}s)")
        del recon, recon_M


if __name__ == "__main__":
    main()
