"""
Side-by-side clustering benchmark on a baked atlas. Ground truth = the
dequantised single-bake atlas (T). Streaming, memory-disciplined:
processes one bucket at a time, frees before the next; chunks the K-means
distance matmul so 48 k × 24 k cluster cases don't OOM.

Four schemes (all evaluated vs T):
  A. single-bake, K-means on raw patches
  B. single-bake, K-means on (patch − per-Gauss DC), DC stored separately
  C. per-level decomposed, K-means per level on raw A_k patches
  D. per-level decomposed, K-means per level on (A_k − per-Gauss DC_k)

Requires `<bake_dir>/decomposed/bucket_<rx>x<ry>.pt` for C/D (produced by
`bake_decompose_full.py`).
"""
import argparse, json, os, glob, time
import torch


def kmeans_chunked(X, K, iters=30, seed=0, weights=None, dist_chunk=8000):
    """Lloyd's K-means with chunked distance matmul (caps peak memory)."""
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
    # Final assignment.
    cn2 = (cents * cents).sum(1)
    for s in range(0, N, dist_chunk):
        e = min(s + dist_chunk, N)
        d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return cents, ass


def bc7_bytes(n_patches, D):
    return n_patches * D // 3


def patch_weights(X):
    """Importance proxy = per-patch L2 norm (normalized to mean=1)."""
    w = X.pow(2).sum(1).sqrt() + 1e-6
    return w / w.mean()


def cluster_eval(X, T, K, iters, dist_chunk, subtract_dc=False, dc_extra_bytes=0,
                 hh=None, ww=None):
    """K-means → reconstruct → return (MSE vs T, codebook_bytes, idx_bits, dc_bytes)."""
    n_sub, D = X.shape
    if subtract_dc:
        dc = X.reshape(n_sub, hh, ww, 3).mean(dim=(1, 2))     # [n_sub, 3]
        X_centered = X - dc.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
        dc_bytes = n_sub * 3                                   # uint8 BC7-equiv
    else:
        X_centered = X
        dc_bytes = 0
    W = patch_weights(X_centered)
    K = min(n_sub, max(4, K))
    cents, ass = kmeans_chunked(X_centered, K, iters=iters, weights=W, dist_chunk=dist_chunk)
    recon = cents[ass]
    if subtract_dc:
        recon = recon + dc.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
    se = (recon - T).pow(2).sum().item()
    return se, bc7_bytes(K, D), n_sub * int(torch.tensor(float(max(K, 2))).log2().ceil().item()), dc_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--ratios", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--cluster_iters", type=int, default=30)
    p.add_argument("--min_bucket_n", type=int, default=100)
    p.add_argument("--max_bucket_n", type=int, default=200000,
                   help="Sub-sample large buckets for K-means speed; reported "
                        "PSNR is on the sub-sample, weighted to the full bucket.")
    p.add_argument("--dist_chunk", type=int, default=4000)
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    if rects.shape[1] >= 5 and (rects[:, 4] != 0).any():
        raise SystemExit("Multi-layer atlas not supported.")
    u0, v0 = rects[:, 0].long(), rects[:, 1].long()
    w_, h_ = rects[:, 2].long(), rects[:, 3].long()

    decomp_dir = os.path.join(args.bake_dir, "decomposed")
    have_decomp = os.path.isdir(decomp_dir) and bool(glob.glob(os.path.join(decomp_dir, "bucket_*.pt")))
    if have_decomp:
        try:
            g_psnr = json.load(open(os.path.join(decomp_dir, "summary.json")))["global_psnr_vs_atlas"]
            print(f"[LOAD] decomposition global Σ A_k vs atlas: {g_psnr:.2f} dB")
        except Exception:
            pass
    print(f"[LOAD] atlas scale={a_scale:.4f} offset={a_off:.4f}; "
          f"decomposed atlases {'FOUND' if have_decomp else 'MISSING'}")

    # Bucketize.
    key = (w_ * 100000 + h_).long()
    uniq, inv = key.unique(return_inverse=True)
    buckets = []
    for b in range(uniq.shape[0]):
        idx = (inv == b).nonzero(as_tuple=True)[0]
        ww, hh = int(w_[idx[0]].item()), int(h_[idx[0]].item())
        if ww == 0 or hh == 0: continue
        if idx.shape[0] < args.min_bucket_n: continue
        buckets.append((ww, hh, idx))
    buckets.sort(key=lambda t: -(t[0] * t[1] * t[2].shape[0]))

    # Reference single-bake size (BC7).
    single_bc7_mb = sum(bc7_bytes(idx.shape[0], ww*hh*3) for ww, hh, idx in buckets) / 1024 / 1024
    print(f"[REF] single-bake BC7 atlas reference: {single_bc7_mb:.2f} MB\n")

    def stream_bucket(ww, hh, idx):
        """Yield (T_atlas [n_sub, D], n_full, sub_idx) for one bucket."""
        n_full = idx.shape[0]
        sub_idx = idx
        if n_full > args.max_bucket_n:
            perm = torch.randperm(n_full, device='cuda')[:args.max_bucket_n]
            sub_idx = idx[perm]
        n_sub = sub_idx.shape[0]
        D = hh * ww * 3
        T = torch.empty(n_sub, hh, ww, 3, device='cuda', dtype=torch.float32)
        u0i = u0[sub_idx]; v0i = v0[sub_idx]
        for r in range(n_sub):
            T[r] = atlas_u8[v0i[r]:v0i[r]+hh, u0i[r]:u0i[r]+ww].float()
        return (T / 255.0 * a_scale + a_off).reshape(n_sub, D), n_full, sub_idx

    def stream_levels(ww, hh, sub_idx):
        """Yield A_k [n_sub, D] one level at a time for memory."""
        bpath = os.path.join(decomp_dir, f"bucket_{ww}x{hh}.pt")
        if not os.path.exists(bpath):
            return None
        blob = torch.load(bpath, map_location='cpu', weights_only=False)
        A_full = blob['A']   # [L, n_full_bucket, hh, ww, 3] FP16 on CPU
        gidx = blob['gauss_idx'].to('cuda')
        pos_map = torch.full((rects.shape[0],), -1, dtype=torch.long, device='cuda')
        pos_map[gidx] = torch.arange(gidx.shape[0], device='cuda')
        pos = pos_map[sub_idx].cpu()
        L = A_full.shape[0]
        for k in range(L):
            A_k = A_full[k][pos].to('cuda').float().reshape(pos.shape[0], -1)
            yield k, A_k, L
            del A_k

    def run_experiment(name, decomposed: bool, subtract_dc: bool):
        print(f"\n--- {name} ---")
        print(f"   {'ratio':>5}  {'cb MB':>8}  {'idx MB':>7}  {'dc MB':>6}  "
              f"{'total MB':>9}  {'vs single':>10}  {'recon PSNR':>11}")
        for r_ratio in args.ratios:
            tot_se = 0.0; tot_n = 0
            tot_cb = 0; tot_idx_bits = 0; tot_dc_bytes = 0
            for ww, hh, idx in buckets:
                T, n_full, sub_idx = stream_bucket(ww, hh, idx)
                n_sub, D = T.shape
                K = max(4, n_sub // r_ratio)
                scale = n_full / max(n_sub, 1)
                if not decomposed:
                    se, cb_b, idx_b, dc_b = cluster_eval(
                        T, T, K, args.cluster_iters, args.dist_chunk,
                        subtract_dc=subtract_dc, hh=hh, ww=ww,
                    )
                    tot_se += se * scale
                    tot_n  += n_sub * D * scale
                    tot_cb += cb_b
                    tot_idx_bits += int(round(idx_b * scale))
                    tot_dc_bytes += int(round(dc_b * scale))
                else:
                    if not have_decomp: break
                    recon_sum = torch.zeros(n_sub, D, device='cuda')
                    for k, A_k, L in stream_levels(ww, hh, sub_idx):
                        if subtract_dc:
                            dc = A_k.reshape(n_sub, hh, ww, 3).mean(dim=(1, 2))
                            A_k_c = A_k - dc.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
                            tot_dc_bytes += int(round(n_sub * 3 * scale))
                        else:
                            A_k_c = A_k
                        W = patch_weights(A_k_c)
                        cents, ass = kmeans_chunked(A_k_c, K, iters=args.cluster_iters,
                                                     weights=W, dist_chunk=args.dist_chunk)
                        recon_k = cents[ass]
                        if subtract_dc:
                            recon_k = recon_k + dc.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
                        recon_sum = recon_sum + recon_k
                        tot_cb += bc7_bytes(min(K, n_sub), D)
                        tot_idx_bits += int(round(n_sub * scale * int(torch.tensor(float(max(min(K, n_sub), 2))).log2().ceil().item())))
                        del A_k, A_k_c, cents, ass, recon_k
                        torch.cuda.empty_cache()
                    se = (recon_sum - T).pow(2).sum().item()
                    tot_se += se * scale
                    tot_n  += n_sub * D * scale
                    del recon_sum
                del T
                torch.cuda.empty_cache()
            psnr = -10.0 * (torch.tensor(tot_se / max(tot_n, 1)).clamp_min(1e-20)).log10().item()
            cb_mb = tot_cb / 1024 / 1024
            idx_mb = tot_idx_bits / 8 / 1024 / 1024
            dc_mb = tot_dc_bytes / 1024 / 1024
            total_mb = cb_mb + idx_mb + dc_mb
            print(f"   {r_ratio:>5}  {cb_mb:>8.2f}  {idx_mb:>7.2f}  {dc_mb:>6.2f}  "
                  f"{total_mb:>9.2f}  {total_mb/single_bc7_mb*100:>8.1f}%  "
                  f"{psnr:>9.2f} dB")

    run_experiment("A. single-bake, K-means on raw patches",          decomposed=False, subtract_dc=False)
    run_experiment("B. single-bake, K-means on (patch − DC)",         decomposed=False, subtract_dc=True)
    if have_decomp:
        run_experiment("C. decomposed (4 levels), K-means per level",  decomposed=True,  subtract_dc=False)
        run_experiment("D. decomposed (4 levels), K-means on (A_k − DC_k)", decomposed=True, subtract_dc=True)
    print(f"\n   single-bake BC7 reference: {single_bc7_mb:.2f} MB")


if __name__ == "__main__":
    main()
