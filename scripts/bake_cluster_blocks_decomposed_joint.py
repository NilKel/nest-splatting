"""
Block-level (4×4) clustering on the DECOMPOSED atlases with a SINGLE
SHARED CODEBOOK across all 4 levels (vs the per-level codebooks in
`bake_cluster_blocks_decomposed.py`).

Setup:
  - Build per-level used-block tensor X_L[k] [14.7 M, 48] for k = 0..3
    (same indexing as `bake_cluster_blocks.py` → joint recon alignment).
  - Sample a balanced sub-set across all 4 levels (~max_blocks_for_kmeans
    total) → fit one K-codeword codebook on the concat.
  - For each level k, assign all 14.7 M blocks against the shared
    codebook → get per-level ass_k.
  - Per-level PSNR: recon_k = cents[ass_k]  vs  X_L[k] unclustered.
  - Joint PSNR : Σ_k recon_k                vs  deployed atlas T.

Storage:
  codebook   K × 48 bytes  (single shared)
  indices    4 × N_blocks × ceil(log2 K) bits
At K=65 536 with N=14.7 M: 1 MB codebook + 4 × 28 MB ≈ 113 MB total
(vs 116 MB for the 4-codebook version → codebook itself is rounding
error at high K).
"""
import argparse, glob, json, math, os, time
import torch


def kmeans_chunked(X, K, iters=20, seed=0, dist_chunk=None):
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


def build_level_atlas(H, W, bucket_files, level_k, rects):
    u0 = rects[:, 0].long(); v0 = rects[:, 1].long()
    lvl = torch.zeros((H, W, 3), dtype=torch.float32, device='cuda')
    for bf in bucket_files:
        blob = torch.load(bf, map_location='cpu', weights_only=False)
        A_k = blob['A'][level_k]
        gidx = blob['gauss_idx']
        hh, ww = int(A_k.shape[1]), int(A_k.shape[2])
        n_b = gidx.shape[0]
        chunk = 2000
        for s in range(0, n_b, chunk):
            e = min(s + chunk, n_b)
            A_chunk = A_k[s:e].to('cuda', non_blocking=True).float()
            u0c = u0[gidx[s:e].to('cuda')]
            v0c = v0[gidx[s:e].to('cuda')]
            for r in range(e - s):
                lvl[v0c[r]:v0c[r] + hh, u0c[r]:u0c[r] + ww] = A_chunk[r]
        del A_k, blob
    return lvl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--Ks", type=int, nargs="+",
                   default=[64, 256, 1024, 4096, 16384, 65536])
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_600_000,
                   help="Total fit-sample budget split equally across levels.")
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
    u0i, v0i = rects[:, 0].long(), rects[:, 1].long()
    w_, h_ = rects[:, 2].long(), rects[:, 3].long()

    used = torch.zeros((H // B, W // B), dtype=torch.bool, device='cuda')
    for i in range(rects.shape[0]):
        ww = int(w_[i].item()); hh = int(h_[i].item())
        if ww == 0 or hh == 0: continue
        bu = int(u0i[i].item()) // B
        bv = int(v0i[i].item()) // B
        used[bv:bv + hh // B, bu:bu + ww // B] = True
    used_idx = used.nonzero(as_tuple=False)
    n_used_blocks = used_idx.shape[0]
    print(f"[BLOCKS] {n_used_blocks:,} used 4×4 blocks")

    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off
    atlas_blocks = atlas_f.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
    X_atlas = atlas_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used_blocks, -1).contiguous()
    del atlas_blocks, atlas_f
    D = X_atlas.shape[1]
    print(f"[ATLAS] X_atlas {tuple(X_atlas.shape)} "
          f"({X_atlas.element_size()*X_atlas.numel()/1024/1024:.0f} MB)")

    decomp_dir = os.path.join(args.bake_dir, "decomposed")
    bucket_files = sorted(glob.glob(os.path.join(decomp_dir, "bucket_*.pt")))
    if not bucket_files:
        raise SystemExit(f"No decomposed atlases in {decomp_dir}.")
    sample = torch.load(bucket_files[0], map_location='cpu', weights_only=False)
    L = int(sample['A'].shape[0])
    del sample
    print(f"[DECOMP] {L} levels, {len(bucket_files)} buckets")

    single_bc7_mb = n_used_blocks * B * B / 1024 / 1024
    print(f"\n[REF] single-bake BC7: {single_bc7_mb:.2f} MB\n")

    X_L = []
    for k in range(L):
        t0 = time.time()
        lvl = build_level_atlas(H, W, bucket_files, k, rects)
        lvl_blocks = lvl.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
        Xk = lvl_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used_blocks, -1).contiguous()
        del lvl, lvl_blocks
        torch.cuda.empty_cache()
        X_L.append(Xk.half())
        print(f"  level {k} blocks built in {time.time()-t0:.1f}s "
              f"(|X|={tuple(Xk.shape)})")

    # Σ A_k vs T sanity (same caveat as before — FP16 cache drops sanity
    # to ~53 dB).
    se = 0.0
    rchunk = 500_000
    for s in range(0, n_used_blocks, rchunk):
        e = min(s + rchunk, n_used_blocks)
        rec = torch.zeros(e - s, D, device='cuda')
        for k in range(L):
            rec = rec + X_L[k][s:e].float()
        se += (rec - X_atlas[s:e]).pow(2).sum().item()
        del rec
    print(f"  Σ A_k vs T (sanity): "
          f"{-10.0 * math.log10(max(se / (n_used_blocks * D), 1e-20)):.2f} dB\n")

    # Build the joint fit sample: equal slice from each level (subsample).
    per_lvl = args.max_blocks_for_kmeans // L
    g = torch.Generator(device='cuda').manual_seed(0)
    fit_pieces = []
    for k in range(L):
        perm = torch.randperm(n_used_blocks, generator=g, device='cuda')[:per_lvl]
        fit_pieces.append(X_L[k][perm].float())
    X_fit = torch.cat(fit_pieces, dim=0)
    del fit_pieces
    print(f"[FIT] joint sample {X_fit.shape} (≈{per_lvl:,}/level × {L} levels)\n")

    print(f"   {'K':>10}  {'L0 PSNR':>9}  {'L1 PSNR':>9}  {'L2 PSNR':>9}  {'L3 PSNR':>9}  "
          f"{'joint vs T':>11}  {'total MB':>9}  {'vs single':>10}")

    for K in args.Ks:
        K_eff = min(K, X_fit.shape[0])
        t0 = time.time()
        cents, _ = kmeans_chunked(X_fit, K_eff, iters=args.iters)
        cn2 = (cents * cents).sum(1)
        chunk = max(2_000, min(2_000_000, 500_000_000 // (K_eff * 4)))

        # Per-level assign + per-level PSNR.
        ass_all = []
        psnr_per = []
        for k in range(L):
            ass_k = torch.empty(n_used_blocks, dtype=torch.long, device='cuda')
            X_full = X_L[k].float()
            for s in range(0, n_used_blocks, chunk):
                e = min(s + chunk, n_used_blocks)
                d = -2.0 * (X_full[s:e] @ cents.T) + cn2.unsqueeze(0)
                ass_k[s:e] = d.argmin(1)
                del d
            se_k = 0.0
            for s in range(0, n_used_blocks, chunk):
                e = min(s + chunk, n_used_blocks)
                rec = cents[ass_k[s:e]]
                se_k += (rec - X_full[s:e]).pow(2).sum().item()
                del rec
            psnr_per.append(-10.0 * math.log10(max(se_k / (n_used_blocks * D), 1e-20)))
            ass_all.append(ass_k)
            del X_full
            torch.cuda.empty_cache()

        # Joint recon vs T.
        joint_se = 0.0
        for s in range(0, n_used_blocks, rchunk):
            e = min(s + rchunk, n_used_blocks)
            rec = torch.zeros(e - s, D, device='cuda')
            for k in range(L):
                rec = rec + cents[ass_all[k][s:e]]
            joint_se += (rec - X_atlas[s:e]).pow(2).sum().item()
            del rec
        joint_psnr = -10.0 * math.log10(max(joint_se / (n_used_blocks * D), 1e-20))

        codebook_bytes = K_eff * B * B
        idx_bits_per = math.ceil(math.log2(max(K_eff, 2)))
        idx_bytes = (L * n_used_blocks * idx_bits_per + 7) // 8
        total_mb = (codebook_bytes + idx_bytes) / 1024 / 1024
        dt = time.time() - t0
        ps = "  ".join(f"{p:>7.2f} dB" for p in psnr_per)
        print(f"   {K:>10}  {ps}  "
              f"{joint_psnr:>8.2f} dB  {total_mb:>7.2f} MB  "
              f"{total_mb/single_bc7_mb*100:>8.2f}%   ({dt:.0f}s)")
        del ass_all, cents
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
