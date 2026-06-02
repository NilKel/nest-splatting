"""
Block-level (4×4) clustering on the original single-bake atlas.
Replaces BC7's per-block scalar quantisation with vector-quantisation:
each 4×4 RGB block becomes one index into a global codebook of 4×4 patches.

Why this is fundamentally different from §2/§3 patch clustering:
- Patch clustering had 75 207 vectors of dim 12 288 (one per Gauss).
- Block clustering has ~14 M vectors of dim 48 (one per 4×4 atlas block,
  intersecting per-Gauss rects).
- Far more vectors at far lower dimension → far more cross-vector
  redundancy. Flat-ish regions of the atlas (the bulk of it) collapse to
  one codebook entry.

Storage:
  codebook   K × 4·4·3  bytes (uint8)  ≈ 48·K
  indices    N_blocks × ceil(log2(K))  bits
For K=256, 14M blocks → ~14 MB total. Compare 225 MB single-bake BC7.

Reports PSNR vs the dequantised atlas at the same byte-cost points as §3.
"""
import argparse, json, math, os, time
import torch


def kmeans_chunked(X, K, iters=30, seed=0, weights=None, dist_chunk=None):
    """Memory-disciplined Lloyd's. X: [N, D] float32. dist_chunk is
    auto-sized to keep peak distance-matrix below ~2 GB if not given."""
    if dist_chunk is None:
        # 500 MB cap on (chunk × K × 4 bytes). Floor 2k to keep huge-K
        # passes from going negative; ceiling 2M to stay fast for small K.
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
    # Final assign.
    cn2 = (cents * cents).sum(1)
    for s in range(0, N, dist_chunk):
        e = min(s + dist_chunk, N)
        d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return cents, ass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--Ks", type=int, nargs="+",
                   default=[64, 256, 1024, 4096, 16384, 65536],
                   help="Codebook sizes to sweep.")
    p.add_argument("--block", type=int, default=4,
                   help="Block side in pixels (BC7-native = 4).")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=2_000_000,
                   help="Sub-sample to this many blocks for K-means fit; "
                       "always assign over the full set.")
    p.add_argument("--weight", choices=["none", "variance"], default="variance",
                   help="Importance proxy = per-block stddev (high-variance "
                        "blocks matter more — flat blocks are easy to share).")
    p.add_argument("--with_dc", action="store_true",
                   help="Subtract per-Gauss patch mean before clustering; "
                        "store DC table separately (N × 3 B).")
    args = p.parse_args()
    B = args.block

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    print(f"[LOAD] atlas scale={a_scale:.4f} offset={a_off:.4f}")
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    H, W, _ = atlas_u8.shape
    print(f"[LOAD] atlas {W}×{H} uint8")

    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    if rects.shape[1] >= 5 and (rects[:, 4] != 0).any():
        raise SystemExit("Multi-layer atlas not supported.")
    u0, v0 = rects[:, 0].long(), rects[:, 1].long()
    w_, h_ = rects[:, 2].long(), rects[:, 3].long()

    # Build the "used-block" mask: rows v in [v0, v0+h), cols u in [u0, u0+w)
    # for every per-Gauss rect, snapped to B-aligned blocks. Bake rect sizes
    # are always multiples of 4 (min_res=4 + powers of 2), so no fractional
    # blocks straddle a rect boundary.
    used = torch.zeros((H // B, W // B), dtype=torch.bool, device='cuda')
    # Also build (H/B, W/B) → Gauss-id map for per-Gauss DC lookup.
    gauss_at = torch.full((H // B, W // B), -1, dtype=torch.long, device='cuda')
    for i in range(rects.shape[0]):
        ww = int(w_[i].item()); hh = int(h_[i].item())
        if ww == 0 or hh == 0: continue
        bu = int(u0[i].item()) // B
        bv = int(v0[i].item()) // B
        nu = ww // B; nv = hh // B
        used[bv:bv+nv, bu:bu+nu] = True
        gauss_at[bv:bv+nv, bu:bu+nu] = i
    n_used_blocks = int(used.sum().item())
    print(f"[BLOCKS] {n_used_blocks:,} used 4×4 blocks  "
          f"(atlas has {(H//B)*(W//B):,} total)")

    # Extract every used 4×4 block as a [n_used_blocks, B²·3] FP32 vector
    # (atlas-residual units). Build by gathering via the used-mask indices.
    print(f"[EXTRACT] tiling atlas into {B}×{B} blocks …")
    t0 = time.time()
    # Use unfold: atlas → [H/B, W/B, B, B, 3] (each block contiguous).
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off            # [H, W, 3]
    atlas_blocks = atlas_f.unfold(0, B, B).unfold(1, B, B)          # [H/B, W/B, 3, B, B]
    atlas_blocks = atlas_blocks.permute(0, 1, 3, 4, 2).contiguous()  # [H/B, W/B, B, B, 3]
    used_idx = used.nonzero(as_tuple=False)                          # [n_used, 2]
    X_blocks = atlas_blocks[used_idx[:, 0], used_idx[:, 1]]          # [n_used, B, B, 3]
    X = X_blocks.reshape(n_used_blocks, -1)                          # [n_used, 48]
    D = X.shape[1]
    print(f"[EXTRACT] {X.shape} float32 ({X.element_size()*X.numel()/1024/1024:.1f} MB) "
          f"in {time.time()-t0:.1f}s")
    del atlas_blocks

    # X is the atlas with DC (raw). Build X_cluster = X − DC_per_block
    # if --with_dc is set; else X_cluster = X. PSNR is always reported
    # against X (the raw deployed atlas), adding DC back to recon.
    X_orig = X
    block_gauss = gauss_at[used_idx[:, 0], used_idx[:, 1]]                 # [n_used]
    n_gauss = rects.shape[0]
    if args.with_dc:
        block_mean_rgb = X.reshape(n_used_blocks, B*B, 3).mean(dim=1)       # [n_used, 3]
        dc_sum = torch.zeros(n_gauss, 3, device='cuda')
        dc_cnt = torch.zeros(n_gauss, device='cuda')
        dc_sum.index_add_(0, block_gauss, block_mean_rgb)
        dc_cnt.index_add_(0, block_gauss, torch.ones(n_used_blocks, device='cuda'))
        valid = dc_cnt > 0
        DC = torch.zeros(n_gauss, 3, device='cuda')
        DC[valid] = dc_sum[valid] / dc_cnt[valid].unsqueeze(1)
        DC_per_block_3 = DC[block_gauss]                                    # [n_used, 3]
        DC_per_block = DC_per_block_3.repeat_interleave(B*B, 1).reshape(n_used_blocks, B*B*3)
        X = X_orig - DC_per_block                                           # cluster on centered signal
        n_valid_gauss = int(valid.sum().item())
        dc_table_bytes = n_valid_gauss * 3
        print(f"[DC] subtracting per-Gauss patch mean; DC table = "
              f"{n_valid_gauss:,} × 3 B = {dc_table_bytes/1024/1024:.2f} MB")
    else:
        DC_per_block = None
        dc_table_bytes = 0

    # Weights: per-block stddev (high-variance blocks pull centroids).
    if args.weight == "variance":
        W_full = X.std(dim=1) + 1e-6
        W_full = W_full / W_full.mean()
    else:
        W_full = None

    # Sub-sample for K-means fit if too large.
    if n_used_blocks > args.max_blocks_for_kmeans:
        perm = torch.randperm(n_used_blocks, device='cuda')[:args.max_blocks_for_kmeans]
        X_fit = X[perm]
        W_fit = W_full[perm] if W_full is not None else None
        print(f"[FIT] sub-sampling {args.max_blocks_for_kmeans:,} blocks for K-means; "
              f"will assign over the full {n_used_blocks:,}")
    else:
        X_fit = X
        W_fit = W_full

    # Reference single-bake BC7 size = 1 byte/texel = used blocks × B²:
    single_bc7_mb = n_used_blocks * B * B / 1024 / 1024
    print(f"\n[REF] single-bake BC7 (used blocks only): {single_bc7_mb:.2f} MB\n")

    print(f"   {'K':>6}  {'codebook B':>11}  {'index B':>9}  {'total MB':>9}  "
          f"{'vs single':>10}  {'recon PSNR':>11}")
    for K in args.Ks:
        K = min(K, X_fit.shape[0])
        t0 = time.time()
        cents, _ = kmeans_chunked(X_fit, K, iters=args.iters, weights=W_fit)
        # Assign over the full block set, chunked. Same auto-sizing as
        # kmeans_chunked's dist_chunk to stay under ~2 GB.
        cn2 = (cents * cents).sum(1)
        ass = torch.empty(n_used_blocks, dtype=torch.long, device='cuda')
        chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
        for s in range(0, n_used_blocks, chunk):
            e = min(s + chunk, n_used_blocks)
            d = -2.0 * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
            ass[s:e] = d.argmin(1)
            del d
        se = 0.0
        for s in range(0, n_used_blocks, chunk):
            e = min(s + chunk, n_used_blocks)
            recon = cents[ass[s:e]]
            if DC_per_block is not None:
                recon = recon + DC_per_block[s:e]
            se += (recon - X_orig[s:e]).pow(2).sum().item()
            del recon
        mse = se / (n_used_blocks * D)
        psnr = -10.0 * math.log10(max(mse, 1e-20))
        codebook_bytes = K * B * B           # 1 byte/texel = uint8 codebook
        idx_bits_per   = math.ceil(math.log2(max(K, 2)))
        idx_bytes      = (n_used_blocks * idx_bits_per + 7) // 8
        total_bytes    = codebook_bytes + idx_bytes + dc_table_bytes
        total_mb       = total_bytes / 1024 / 1024
        dt = time.time() - t0
        print(f"   {K:>6}  {codebook_bytes:>11,}  {idx_bytes/1024/1024:>7.2f}MB  "
              f"{total_mb:>9.3f}  {total_mb/single_bc7_mb*100:>8.2f}%  "
              f"{psnr:>9.2f} dB   ({dt:.1f}s)")


if __name__ == "__main__":
    main()
