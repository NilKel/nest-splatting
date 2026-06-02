"""
Block-level (4×4) clustering on the DECOMPOSED atlases (one per hashgrid
level). For each requested codebook size K:
  - Cluster each level's used 4×4 blocks INDEPENDENTLY with K_k = K
    (or per --per_level_Ks).
  - Report per-level PSNR(recon_k  vs  A_k unclustered).
  - Report joint PSNR(Σ_k recon_k  vs  deployed single-bake atlas T).
  - Report total storage = Σ_k (codebook_k + indices_k).

Alignment vs the single-atlas case: we build a full pixel atlas for level
k from the per-Gauss patches and extract used blocks with the SAME
used-mask indexing as `bake_cluster_blocks.py`. By construction the block
at position i in level k corresponds to the same atlas-block position i
in X_atlas → summing across levels is consistent.

Requires `<bake_dir>/decomposed/bucket_*.pt` from
`bake_decompose_full.py`.
"""
import argparse, glob, json, math, os, time
import torch


def kmeans_chunked(X, K, iters=20, seed=0, dist_chunk=None):
    """Memory-disciplined Lloyd's. X: [N, D] float32."""
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
    """Paint per-Gauss patches for `level_k` into a [H, W, 3] atlas."""
    u0 = rects[:, 0].long()
    v0 = rects[:, 1].long()
    lvl = torch.zeros((H, W, 3), dtype=torch.float32, device='cuda')
    for bf in bucket_files:
        blob = torch.load(bf, map_location='cpu', weights_only=False)
        A_k = blob['A'][level_k]                       # [N_b, hh, ww, 3] FP16 CPU
        gidx = blob['gauss_idx']                       # CPU long
        hh, ww = int(A_k.shape[1]), int(A_k.shape[2])
        n_b = gidx.shape[0]
        # Move to GPU in chunks to bound the staging buffer.
        chunk = 2000
        for s in range(0, n_b, chunk):
            e = min(s + chunk, n_b)
            A_chunk = A_k[s:e].to('cuda', non_blocking=True).float()  # [m, hh, ww, 3]
            u0c = u0[gidx[s:e].to('cuda')]
            v0c = v0[gidx[s:e].to('cuda')]
            # Per-Gauss writes — Python loop is fine, m is ≤2000.
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
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_500_000)
    p.add_argument("--per_level_Ks", type=str, default=None,
                   help="Comma-separated K_0..K_3 (overrides --Ks).")
    p.add_argument("--with_dc", action="store_true",
                   help="Subtract per-Gauss per-level patch mean before "
                        "clustering; store DC tables separately (4 × N × 3 B).")
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

    # Used 4×4 block mask + gauss-id map.
    used = torch.zeros((H // B, W // B), dtype=torch.bool, device='cuda')
    gauss_at = torch.full((H // B, W // B), -1, dtype=torch.long, device='cuda')
    for i in range(rects.shape[0]):
        ww = int(w_[i].item()); hh = int(h_[i].item())
        if ww == 0 or hh == 0: continue
        bu = int(u0i[i].item()) // B
        bv = int(v0i[i].item()) // B
        used[bv:bv + hh // B, bu:bu + ww // B] = True
        gauss_at[bv:bv + hh // B, bu:bu + ww // B] = i
    used_idx = used.nonzero(as_tuple=False)
    n_used_blocks = used_idx.shape[0]
    block_gauss = gauss_at[used_idx[:, 0], used_idx[:, 1]]                # [n_used]
    n_gauss = rects.shape[0]
    print(f"[BLOCKS] {n_used_blocks:,} used 4×4 blocks")

    # Deployed atlas blocks (the GT for joint PSNR).
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off
    atlas_blocks = atlas_f.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
    X_atlas = atlas_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used_blocks, -1).contiguous()
    del atlas_blocks, atlas_f
    D = X_atlas.shape[1]
    print(f"[ATLAS] X_atlas {tuple(X_atlas.shape)} ({X_atlas.element_size()*X_atlas.numel()/1024/1024:.0f} MB)")

    decomp_dir = os.path.join(args.bake_dir, "decomposed")
    bucket_files = sorted(glob.glob(os.path.join(decomp_dir, "bucket_*.pt")))
    if not bucket_files:
        raise SystemExit(f"No decomposed atlases in {decomp_dir}.")
    sample = torch.load(bucket_files[0], map_location='cpu', weights_only=False)
    L = int(sample['A'].shape[0])
    del sample
    print(f"[DECOMP] {L} levels, {len(bucket_files)} buckets")

    Ks = args.Ks
    per_level_Ks = None
    if args.per_level_Ks:
        per_level_Ks = [int(x) for x in args.per_level_Ks.split(",")]
        assert len(per_level_Ks) == L
        Ks = [0]                                           # one row
    single_bc7_mb = n_used_blocks * B * B / 1024 / 1024
    print(f"\n[REF] single-bake BC7: {single_bc7_mb:.2f} MB\n")

    # Pre-extract per-level block tensors ONCE and cache to /tmp to avoid
    # repaint per K. ~2.7 GB FP32 per level → keep as FP16 (1.35 GB ea).
    X_L = []
    DC_per_block_L = []                                       # [L][n_used, 48] FP32 on GPU
    dc_total_bytes = 0
    for k in range(L):
        t_paint = time.time()
        lvl = build_level_atlas(H, W, bucket_files, k, rects)
        lvl_blocks = lvl.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
        Xk = lvl_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used_blocks, -1).contiguous()
        del lvl, lvl_blocks
        torch.cuda.empty_cache()

        if args.with_dc:
            block_mean_rgb = Xk.reshape(n_used_blocks, B*B, 3).mean(dim=1)  # [n_used, 3]
            dc_sum = torch.zeros(n_gauss, 3, device='cuda')
            dc_cnt = torch.zeros(n_gauss, device='cuda')
            dc_sum.index_add_(0, block_gauss, block_mean_rgb)
            dc_cnt.index_add_(0, block_gauss, torch.ones(n_used_blocks, device='cuda'))
            valid = dc_cnt > 0
            DC_k = torch.zeros(n_gauss, 3, device='cuda')
            DC_k[valid] = dc_sum[valid] / dc_cnt[valid].unsqueeze(1)
            DC_pb = DC_k[block_gauss].repeat_interleave(B*B, 1).reshape(n_used_blocks, B*B*3)
            Xk = Xk - DC_pb                                                # centered
            DC_per_block_L.append(DC_pb.half())                            # keep FP16 for memory
            dc_total_bytes += int(valid.sum().item()) * 3
        else:
            DC_per_block_L.append(None)

        X_L.append(Xk.half())
        print(f"  level {k} blocks built in {time.time()-t_paint:.1f}s "
              f"(|X|={tuple(Xk.shape)}{', −DC' if args.with_dc else ''})")
    if args.with_dc:
        print(f"[DC] total DC tables: {dc_total_bytes/1024/1024:.2f} MB "
              f"(4 levels × ≤{n_gauss:,} × 3 B)")

    # Sanity check: Σ (X_L_k + DC_k) ≈ X_atlas. (DC adds to recon at PSNR
    # time, so include it here for consistency.)
    se = 0.0
    rchunk = 500_000
    for s in range(0, n_used_blocks, rchunk):
        e = min(s + rchunk, n_used_blocks)
        rec = torch.zeros(e - s, D, device='cuda')
        for k in range(L):
            rec = rec + X_L[k][s:e].float()
            if args.with_dc:
                rec = rec + DC_per_block_L[k][s:e].float()
        se += (rec - X_atlas[s:e]).pow(2).sum().item()
        del rec
    sanity_psnr = -10.0 * math.log10(max(se / (n_used_blocks * D), 1e-20))
    print(f"  Σ A_k vs T (sanity): {sanity_psnr:.2f} dB\n")

    hdr_K = "K"
    if per_level_Ks: hdr_K = "K0/K1/K2/K3"
    print(f"   {hdr_K:>14}  {'L0 PSNR':>9}  {'L1 PSNR':>9}  {'L2 PSNR':>9}  {'L3 PSNR':>9}  "
          f"{'joint vs T':>11}  {'total MB':>9}  {'vs single':>10}")

    for K in Ks:
        Ks_lvl = per_level_Ks if per_level_Ks else [K] * L
        cents_all = []
        ass_all = []
        psnrs_per_level = []
        codebook_bytes = 0
        idx_bytes = 0
        t0 = time.time()
        for k in range(L):
            K_k = Ks_lvl[k]
            X_full = X_L[k].float()                          # [N, 48] FP32 from FP16 cache

            if n_used_blocks > args.max_blocks_for_kmeans:
                perm = torch.randperm(n_used_blocks, device='cuda')[:args.max_blocks_for_kmeans]
                X_fit = X_full[perm]
            else:
                X_fit = X_full
            K_k_eff = min(K_k, X_fit.shape[0])
            cents, _ = kmeans_chunked(X_fit, K_k_eff, iters=args.iters)

            cn2 = (cents * cents).sum(1)
            ass = torch.empty(n_used_blocks, dtype=torch.long, device='cuda')
            chunk = max(2_000, min(2_000_000, 500_000_000 // (K_k_eff * 4)))
            for s in range(0, n_used_blocks, chunk):
                e = min(s + chunk, n_used_blocks)
                d = -2.0 * (X_full[s:e] @ cents.T) + cn2.unsqueeze(0)
                ass[s:e] = d.argmin(1)
                del d

            se_k = 0.0
            for s in range(0, n_used_blocks, chunk):
                e = min(s + chunk, n_used_blocks)
                rec = cents[ass[s:e]]
                se_k += (rec - X_full[s:e]).pow(2).sum().item()
                del rec
            psnr_k = -10.0 * math.log10(max(se_k / (n_used_blocks * D), 1e-20))
            psnrs_per_level.append(psnr_k)
            cents_all.append(cents)
            ass_all.append(ass)
            codebook_bytes += K_k_eff * B * B
            idx_bytes += (n_used_blocks * math.ceil(math.log2(max(K_k_eff, 2))) + 7) // 8
            del X_full, X_fit
            torch.cuda.empty_cache()

        # Joint recon vs X_atlas.
        joint_se = 0.0
        for s in range(0, n_used_blocks, rchunk):
            e = min(s + rchunk, n_used_blocks)
            rec = torch.zeros(e - s, D, device='cuda')
            for k in range(L):
                rec = rec + cents_all[k][ass_all[k][s:e]]
                if args.with_dc:
                    rec = rec + DC_per_block_L[k][s:e].float()
            joint_se += (rec - X_atlas[s:e]).pow(2).sum().item()
            del rec
        joint_psnr = -10.0 * math.log10(max(joint_se / (n_used_blocks * D), 1e-20))

        total_mb = (codebook_bytes + idx_bytes + dc_total_bytes) / 1024 / 1024
        dt = time.time() - t0
        label = "/".join(str(x) for x in Ks_lvl) if per_level_Ks else f"{K}"
        ps = "  ".join(f"{p:>7.2f} dB" for p in psnrs_per_level)
        print(f"   {label:>14}  {ps}  "
              f"{joint_psnr:>8.2f} dB  {total_mb:>7.2f} MB  "
              f"{total_mb/single_bc7_mb*100:>8.2f}%   ({dt:.0f}s)")
        del cents_all, ass_all
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
