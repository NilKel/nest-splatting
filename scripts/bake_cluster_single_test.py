"""
Cluster the per-Gauss patches of an existing single-bake atlas and report
storage savings vs reconstruction PSNR. No re-bake, no level decomposition
— just K-means on what's already in `<bake_dir>/atlas_texture.pt`.

Each Gauss has a variable per-axis resolution rect (rx, ry). We bucket by
(rx, ry) and cluster within each bucket separately (you can't K-means
mixed-shape vectors). The total storage = Σ_bucket (K_b × rect_bytes + N_b ×
log2(K_b) bits); the single-bake reference is Σ_bucket N_b × rect_bytes.

Usage:
  python scripts/bake_cluster_single_test.py --bake_dir <path> \
      [--ratios 2 4 8 16 32 64] [--cluster_iters 30] [--min_bucket_n 50]
"""
import argparse, json, os, time, sys, zlib
import numpy as np
import torch
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


def kmeans_torch(X: torch.Tensor, K: int, iters: int = 30, seed: int = 0,
                 weights: torch.Tensor = None):
    """Lloyd's K-means on GPU. X: [N, D] float32.
    If `weights` given (shape [N]), uses importance-weighted centroid updates
    (c3dgs-style): c_k = Σ w_i · x_i / Σ w_i over cluster k. Assignments still
    minimise plain L2 (we want centroids close to high-w points; assignment
    of an individual point doesn't depend on its own weight since w_i scales
    all candidate distances equally for that point)."""
    N, D = X.shape
    if K >= N:
        return X.clone(), torch.arange(N, device=X.device), 0.0
    g = torch.Generator(device=X.device).manual_seed(seed)
    centroids = X[torch.randperm(N, generator=g, device=X.device)[:K]].clone()
    use_w = weights is not None
    if use_w:
        weights = weights.float()
    for _ in range(iters):
        d = -2.0 * (X @ centroids.T) + (centroids * centroids).sum(1)
        ass = d.argmin(1)
        new = torch.zeros_like(centroids)
        cnt = torch.zeros(K, device=X.device)
        if use_w:
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
        centroids = new
    d = -2.0 * (X @ centroids.T) + (centroids * centroids).sum(1)
    ass = d.argmin(1)
    mse = (X - centroids[ass]).pow(2).mean().item()
    return centroids, ass, mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--ratios", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64, 128],
                   help="K_b = max(min_K, N_b // r). Smaller r = more clusters.")
    p.add_argument("--cluster_iters", type=int, default=30)
    p.add_argument("--min_K", type=int, default=4)
    p.add_argument("--min_bucket_n", type=int, default=50,
                   help="Skip buckets smaller than this (keep them un-clustered).")
    p.add_argument("--max_bucket_n", type=int, default=200000,
                   help="Sub-sample large buckets for K-means speed (the report "
                        "weights by FULL bucket size).")
    p.add_argument("--importance", choices=["none", "patch_norm"], default="patch_norm",
                   help="Per-Gauss importance weight for c3dgs-style centroid "
                        "update (high-importance patches pull centroids harder).")
    p.add_argument("--keep_top_pct", type=float, default=0.0,
                   help="Keep this fraction of highest-importance Gausses "
                        "uncompressed (c3dgs `importance_include` trick).")
    p.add_argument("--morton_zlib", action="store_true", default=True,
                   help="Sort indices in Morton order along Gauss centers + "
                        "zlib(DEFLATE)-compress for the final byte count.")
    args = p.parse_args()

    bake_dir = args.bake_dir
    meta = json.load(open(os.path.join(bake_dir, "bake_meta.json")))
    aw, ah = int(meta["atlas_width"]), int(meta["atlas_height"])
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    print(f"[LOAD] atlas {aw}×{ah}  scale={a_scale:.4f}  offset={a_off:.4f}")

    atlas = torch.load(os.path.join(bake_dir, "atlas_texture.pt"),
                       map_location='cuda', weights_only=False)  # uint8 HxWx3 (or 4)
    if atlas.dim() == 3 and atlas.shape[2] == 4:
        atlas = atlas[..., :3]
    print(f"[LOAD] atlas tensor {tuple(atlas.shape)} dtype={atlas.dtype}")
    rects = torch.load(os.path.join(bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    print(f"[LOAD] atlas_rects {tuple(rects.shape)}  ({rects.shape[0]:,} Gausses)")

    # rects shape varies (4 or 5 cols depending on bake version). Treat first 4
    # as (u0, v0, w, h). 5th if present is layer (multi-layer atlas) which
    # would need extra handling — fail loud if encountered.
    if rects.shape[1] == 5 and (rects[:, 4] != 0).any():
        raise SystemExit("Multi-layer atlas not supported in this single-page test.")
    u0 = rects[:, 0].long()
    v0 = rects[:, 1].long()
    w_ = rects[:, 2].long()
    h_ = rects[:, 3].long()
    N = rects.shape[0]
    # Skip-texture entries (w=h=0) are kept out of clustering — they contribute
    # 0 bytes anyway.
    keep = (w_ > 0) & (h_ > 0)
    drop = (~keep).sum().item()
    if drop:
        print(f"[INFO] dropping {drop:,} skip-texture rects (w=h=0)")
    u0, v0, w_, h_ = u0[keep], v0[keep], w_[keep], h_[keep]
    N = w_.shape[0]

    # Bucket by (w, h).
    key = (w_ * 100000 + h_).long()
    uniq, inv = key.unique(return_inverse=True)
    buckets = []
    for b, k in enumerate(uniq.tolist()):
        idx = (inv == b).nonzero(as_tuple=True)[0]
        ww = int(w_[idx[0]].item()); hh = int(h_[idx[0]].item())
        buckets.append((ww, hh, idx))
    buckets.sort(key=lambda t: -(t[0] * t[1] * t[2].shape[0]))
    print(f"\n[BUCKETS] {len(buckets)} unique (rx, ry) shapes; "
          f"top 5 by total-byte-share:")
    total_bytes_single = 0
    for ww, hh, idx in buckets[:5]:
        n = idx.shape[0]
        share = ww * hh * 3 * n
        total_bytes_single_part = ww * hh * 3 * n
        print(f"   {ww:>3}×{hh:<3}  N={n:>7,}  "
              f"single-bake bytes (BC7 ≈ 1B/texel·3): {share/1024/1024:>7.1f} MB")
    total_bytes_single_full = sum(ww * hh * 3 * idx.shape[0] for ww, hh, idx in buckets)
    print(f"   ALL bytes (uint8 reference): {total_bytes_single_full/1024/1024:.1f} MB")

    # Morton order over Gauss centers for index entropy-coding. Loads baked.ply
    # to read xyz; if absent we fall back to unsorted indices and warn.
    morton_order = None
    if args.morton_zlib and PlyData is not None:
        ply_path = os.path.join(bake_dir, "baked.ply")
        if os.path.exists(ply_path):
            print(f"[MORTON] loading {ply_path} for spatial sort …")
            pd = PlyData.read(ply_path)
            v = pd['vertex']
            xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], 1)
            # Quantize each axis to 21-bit; interleave to a 63-bit Morton code.
            mn = xyz.min(0); mx = xyz.max(0)
            q = np.clip(((xyz - mn) / (mx - mn + 1e-12)) * (2**21 - 1), 0, 2**21 - 1).astype(np.uint64)
            def morton3(a, b, c, n=21):
                def spread(x):
                    x = x & 0x1fffff
                    x = (x | (x << 32)) & 0x1f00000000ffff
                    x = (x | (x << 16)) & 0x1f0000ff0000ff
                    x = (x | (x << 8))  & 0x100f00f00f00f00f
                    x = (x | (x << 4))  & 0x10c30c30c30c30c3
                    x = (x | (x << 2))  & 0x1249249249249249
                    return x
                return spread(a) | (spread(b) << 1) | (spread(c) << 2)
            codes = morton3(q[:, 0], q[:, 1], q[:, 2])
            morton_order = np.argsort(codes)   # global Gauss permutation
            print(f"[MORTON] computed {len(codes):,}-element ordering")
        else:
            print(f"[MORTON] no baked.ply → skipping spatial sort (zlib runs on natural order)")
    elif args.morton_zlib:
        print(f"[MORTON] plyfile not installed → skipping spatial sort")

    # Extract + dequantize all patches up-front, per-bucket.
    print(f"\n[EXTRACT] patches → float32 …")
    bucket_X = []   # list of (ww, hh, N, X[N, D])
    t0 = time.time()
    for ww, hh, idx in buckets:
        if idx.shape[0] < args.min_bucket_n:
            continue
        # Cap bucket size for K-means perf; report weights with full N still.
        idx_full = idx
        if idx.shape[0] > args.max_bucket_n:
            perm = torch.randperm(idx.shape[0], device='cuda')[:args.max_bucket_n]
            idx = idx[perm]
        rows = idx.shape[0]
        # Build X by sliced indexing — atlas_uv = atlas[v0+y, u0+x].
        X = torch.empty(rows, hh, ww, 3, device='cuda', dtype=torch.float32)
        u0i = u0[idx]; v0i = v0[idx]
        for r in range(rows):
            patch = atlas[v0i[r]:v0i[r]+hh, u0i[r]:u0i[r]+ww].float()
            X[r] = patch
        X = (X / 255.0 * a_scale + a_off).reshape(rows, hh * ww * 3)
        bucket_X.append((ww, hh, idx_full.shape[0], X))
    print(f"[EXTRACT] done in {time.time()-t0:.1f}s; "
          f"clustering {len(bucket_X)} buckets ≥ {args.min_bucket_n} Gausses")

    # Single-bake quantization noise itself is the floor: re-quantize the
    # extracted floats to uint8 and back to estimate that baseline.
    with torch.no_grad():
        total_se_quant = 0.0; total_count = 0
        for ww, hh, n_full, X in bucket_X:
            q = ((X - a_off) / a_scale * 255.0).clamp(0, 255).round() / 255.0 * a_scale + a_off
            total_se_quant += (q - X).pow(2).sum().item() * (n_full / X.shape[0])
            total_count += X.numel() * (n_full / X.shape[0])
    quant_psnr = -10.0 * (torch.tensor(total_se_quant / total_count)).log10().item()
    print(f"[BASELINE] uint8-quant PSNR vs FP atlas (the bake's own quantization "
          f"noise): {quant_psnr:.2f} dB  (this is the ceiling for any clustering)")

    # Per-bucket importance weights = L2 norm of each patch (proxy for c3dgs
    # render-grad sensitivity; high-energy patches matter more to preserve).
    # Computed once; reused across all ratio sweeps.
    bucket_W = []
    for (ww, hh, n_full, X) in bucket_X:
        if args.importance == "patch_norm":
            w = X.pow(2).sum(1).sqrt() + 1e-6
            w = w / w.mean()       # normalize to mean=1 (keeps K-means scale stable)
        else:
            w = torch.ones(X.shape[0], device='cuda')
        bucket_W.append(w)

    # Sweep K_b = max(min_K, N_b // r).
    keep_str = f"keep_top {args.keep_top_pct:.1%} uncompressed" if args.keep_top_pct > 0 else "all clustered"
    print(f"\n--- K-means per-bucket sweep (importance={args.importance}, {keep_str}) ---")
    print(f"   {'ratio':>5}  {'codebook MB':>14}  {'kept MB':>9}  {'idx MB':>9}  "
          f"{'comp MB':>12}  {'vs single':>10}  {'recon PSNR':>11}")
    single_mb = total_bytes_single_full / 1024 / 1024
    single_bc7_mb = single_mb / 3.0   # uint8 → BC7 is /3
    for r_ratio in args.ratios:
        total_se = 0.0
        total_codebook_bytes = 0
        total_kept_bytes = 0
        total_idx_bits = 0
        total_n = 0
        for (ww, hh, n_full, X), W in zip(bucket_X, bucket_W):
            n_sub = X.shape[0]
            D = hh * ww * 3
            # keep_top_pct: protect highest-importance patches from clustering.
            keep_mask = torch.zeros(n_sub, dtype=torch.bool, device='cuda')
            if args.keep_top_pct > 0:
                k_keep = max(1, int(n_sub * args.keep_top_pct))
                top = W.topk(k_keep).indices
                keep_mask[top] = True
            cluster_X = X[~keep_mask]
            cluster_W = W[~keep_mask]
            n_cluster = cluster_X.shape[0]
            K = max(args.min_K, n_cluster // r_ratio) if n_cluster > 0 else 0
            K = min(K, n_cluster)
            if K > 0:
                cents, _, _ = kmeans_torch(cluster_X, K, iters=args.cluster_iters,
                                            seed=0, weights=cluster_W)
                d = -2.0 * (cluster_X @ cents.T) + (cents * cents).sum(1)
                ass = d.argmin(1)
                # Reconstruction error — weighted by patch importance.
                err = (cluster_X - cents[ass]).pow(2)
                # Plain MSE (we'll report PSNR in unweighted units to match
                # what render-time perception sees).
                mse_cluster = err.mean().item()
            else:
                mse_cluster = 0.0
            # Kept patches contribute zero error. Full-bucket weighted mean error
            # = (n_cluster / n_sub) · mse_cluster (kept patches are exact).
            mse_sub = mse_cluster * (n_cluster / max(n_sub, 1))
            total_se += mse_sub * D * n_full
            total_n  += D * n_full
            # Codebook = K patches × h·w bytes BC7. Plus N_kept × h·w bytes for
            # the protected patches stored verbatim BC7.
            kept_frac = keep_mask.float().mean().item() if n_sub > 0 else 0.0
            total_codebook_bytes += K * D // 3
            total_kept_bytes     += int(round(kept_frac * n_full)) * D // 3
            n_clustered_full = n_full - int(round(kept_frac * n_full))
            total_idx_bits       += n_clustered_full * int(torch.tensor(float(max(K, 2))).log2().ceil().item())
        psnr = -10.0 * (torch.tensor(total_se / total_n).clamp_min(1e-20)).log10().item()
        cb_mb = total_codebook_bytes / 1024 / 1024
        kept_mb = total_kept_bytes / 1024 / 1024
        idx_mb = total_idx_bits / 8 / 1024 / 1024
        comp_mb = cb_mb + kept_mb + idx_mb
        print(f"   {r_ratio:>5}  {cb_mb:>14.2f}  {kept_mb:>9.2f}  {idx_mb:>9.2f}  {comp_mb:>12.2f}  "
              f"{comp_mb/single_bc7_mb*100:>9.1f}%  {psnr:>9.2f} dB")
    print(f"\n   single-bake BC7 reference: {single_bc7_mb:.2f} MB "
          f"(uint8 src = {single_mb:.2f} MB ÷ 3)")


if __name__ == "__main__":
    main()
