"""
Quadtree + RVQ — adaptive block-size VQ with L-stage residual codebooks
at each leaf size (16×16, 8×8, 4×4).

For each 16×16 atlas superblock, choose the largest leaf size whose
L-stage RVQ reconstruction beats a per-region error budget.

Storage per 16×16 superblock at L=4 K=256:
  16-leaf:  1 split bit  + L×8 = 33 bits     (1 RVQ index per stage)
  8-leaf  : 1 split + 4×(1 + L×8) = 133 bits (4 sub-blocks, each L×8)
  4-leaf  : 1 + 4×(1 + 4×(1 + L×8)) = 533 bits

Compare flat-4 RVQ: 16 × L×8 = 512 bits/16×16 area.

→ 16-leaf is 15× cheaper than flat-4 RVQ. Worst case (all-4) is 4 % worse
   (split-bit overhead). 50/50 mix is ~50 % cheaper than flat-4.

Three RVQ towers, one per block size:
  cents_16:  L codebooks of K codewords × (16·16·3)=768 D each
  cents_8 :  L × K × 192-D
  cents_4 :  L × K × 48-D
Codebook totals at K=256 L=4: 4·256·(768+192+48) bytes = ~1.04 MB.
Negligible vs indices.
"""
import argparse, json, math, os, time
import torch
import numpy as np


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
    return cents


def train_rvq(X, K, L, iters=15, max_fit=1_500_000):
    """L-stage RVQ on [N, D]. Returns list of L codebooks ([K, D] each)."""
    if X.shape[0] > max_fit:
        perm = torch.randperm(X.shape[0], device=X.device)[:max_fit]
        X_fit = X[perm]
    else:
        X_fit = X
    codebooks = []
    R = X_fit.clone()
    for l in range(L):
        cb = kmeans_chunked(R, K, iters=iters)
        codebooks.append(cb)
        # subtract greedy assignment of this fit set
        cn2 = (cb * cb).sum(1)
        chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
        for s in range(0, R.shape[0], chunk):
            e = min(s + chunk, R.shape[0])
            d = -2.0 * (R[s:e] @ cb.T) + cn2.unsqueeze(0)
            R[s:e] = R[s:e] - cb[d.argmin(1)]
            del d
    return codebooks


def rvq_assign_recon(X, codebooks):
    """L-stage RVQ greedy assign over the FULL set. Returns
    (recon [N, D], per_block_mse [N], asses [list of L [N]])."""
    N, D = X.shape
    K = codebooks[0].shape[0]
    L = len(codebooks)
    chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    R = X.clone()
    recon = torch.zeros_like(X)
    asses = []
    for cb in codebooks:
        cn2 = (cb * cb).sum(1)
        ass = torch.empty(N, dtype=torch.long, device=X.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = -2.0 * (R[s:e] @ cb.T) + cn2.unsqueeze(0)
            ass[s:e] = d.argmin(1)
            del d
        recon = recon + cb[ass]
        R = X - recon                                                # residual = X - running recon
        asses.append(ass)
    # Per-block MSE (chunked)
    mse = torch.empty(N, device=X.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        mse[s:e] = (recon[s:e] - X[s:e]).pow(2).mean(dim=1)
    return recon, mse, asses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--K", type=int, default=256, help="K per RVQ stage (default for all 3 block sizes).")
    p.add_argument("--K16", type=int, default=-1, help="Override K for 16×16 codebooks (-1 = use --K).")
    p.add_argument("--K8",  type=int, default=-1, help="Override K for 8×8 codebooks.")
    p.add_argument("--K4",  type=int, default=-1, help="Override K for 4×4 codebooks.")
    p.add_argument("--L", type=int, default=4, help="RVQ stages per block size.")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_500_000)
    p.add_argument("--thresholds", type=str, default="auto")
    args = p.parse_args()
    L = args.L
    K16 = args.K16 if args.K16 > 0 else args.K
    K8  = args.K8  if args.K8  > 0 else args.K
    K4  = args.K4  if args.K4  > 0 else args.K

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    H, W, _ = atlas_u8.shape
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off
    del atlas_u8
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    u0 = rects[:, 0].long(); v0 = rects[:, 1].long()
    w_ = rects[:, 2].long(); h_ = rects[:, 3].long()
    print(f"[LOAD] atlas {H}×{W}, {rects.shape[0]} surfels")

    # ---- Per-surfel top-block-size decision --------------------------------
    sizes = []
    for i in range(rects.shape[0]):
        ww = int(w_[i]); hh = int(h_[i])
        if ww == 0 or hh == 0: sizes.append(0); continue
        if ww >= 16 and hh >= 16 and ww % 16 == 0 and hh % 16 == 0:
            sizes.append(16)
        elif ww >= 8 and hh >= 8 and ww % 8 == 0 and hh % 8 == 0:
            sizes.append(8)
        else:
            sizes.append(4)
    sizes = torch.tensor(sizes, device='cuda')
    n16 = int((sizes == 16).sum()); n8 = int((sizes == 8).sum()); n4 = int((sizes == 4).sum())
    print(f"[TILE] surfels by top-block-size: 16: {n16:,}  8: {n8:,}  4: {n4:,}")

    # ---- Gather per-block tensors at three sizes ---------------------------
    def gather(B, surf_mask):
        view = atlas_f.unfold(0, B, B).unfold(1, B, B).permute(0,1,3,4,2).contiguous()
        out = []
        for i in surf_mask.nonzero(as_tuple=False).flatten().tolist():
            ww = int(w_[i]); hh = int(h_[i])
            bu = int(u0[i]) // B; bv = int(v0[i]) // B
            patch = view[bv:bv + hh//B, bu:bu + ww//B].reshape(-1, B*B*3)
            out.append(patch)
        if not out:
            return torch.empty(0, B*B*3, device='cuda')
        return torch.cat(out, 0)

    t0 = time.time()
    X16 = gather(16, sizes == 16)
    X8_top = gather(8,  sizes == 8)
    X4_top = gather(4,  sizes == 4)
    del atlas_f; torch.cuda.empty_cache()
    print(f"[TILE] gathered: 16: {X16.shape[0]:,} ({X16.element_size()*X16.numel()/1024/1024:.0f} MB) "
          f"| 8: {X8_top.shape[0]:,} | 4: {X4_top.shape[0]:,}  ({time.time()-t0:.1f}s)")

    # Subdivisions of 16-blocks (for evaluating 8/4 leaf alternatives).
    X16_sub8 = X16.reshape(-1, 2, 8, 2, 8, 3).permute(0,1,3,2,4,5).reshape(-1, 8*8*3)
    X16_sub4 = X16.reshape(-1, 4, 4, 4, 4, 3).permute(0,1,3,2,4,5).reshape(-1, 4*4*3)
    X8_sub4 = X8_top.reshape(-1, 2, 4, 2, 4, 3).permute(0,1,3,2,4,5).reshape(-1, 4*4*3)

    # ---- Train 3 RVQ towers ------------------------------------------------
    print(f"[RVQ] training 3 RVQ towers: K16={K16}, K8={K8}, K4={K4}, L={L} …")
    t0 = time.time()
    rvq_16 = train_rvq(X16, K16, L, iters=args.iters, max_fit=args.max_blocks_for_kmeans)
    print(f"  rvq_16 trained ({time.time()-t0:.1f}s, {len(rvq_16)} codebooks)")
    t0 = time.time()
    pool8 = torch.cat([X8_top, X16_sub8], 0)
    rvq_8 = train_rvq(pool8, K8, L, iters=args.iters, max_fit=args.max_blocks_for_kmeans)
    del pool8; torch.cuda.empty_cache()
    print(f"  rvq_8 trained ({time.time()-t0:.1f}s)")
    t0 = time.time()
    pool4 = torch.cat([X4_top, X8_sub4, X16_sub4], 0)
    rvq_4 = train_rvq(pool4, K4, L, iters=args.iters, max_fit=args.max_blocks_for_kmeans)
    del pool4; torch.cuda.empty_cache()
    print(f"  rvq_4 trained ({time.time()-t0:.1f}s)")

    # ---- Greedy assign + per-block MSE at each scale -----------------------
    print(f"[ASSIGN] L-stage RVQ assignments at each scale …")
    t0 = time.time()
    _, mse_16_at16, _ = rvq_assign_recon(X16,     rvq_16)
    _, mse_16_at8_sub, _ = rvq_assign_recon(X16_sub8, rvq_8)            # [N16*4]
    mse_16_at8 = mse_16_at8_sub.reshape(-1, 4).mean(1)                  # [N16]
    _, mse_16_at4_sub, _ = rvq_assign_recon(X16_sub4, rvq_4)            # [N16*16]
    mse_16_at4 = mse_16_at4_sub.reshape(-1, 16).mean(1)
    _, mse_8_at8, _    = rvq_assign_recon(X8_top, rvq_8)
    _, mse_8_at4_sub, _ = rvq_assign_recon(X8_sub4, rvq_4)
    mse_8_at4 = mse_8_at4_sub.reshape(-1, 4).mean(1)
    _, mse_4_at4, _    = rvq_assign_recon(X4_top, rvq_4)
    print(f"[ASSIGN] done ({time.time()-t0:.1f}s)")

    # ---- Threshold sweep --------------------------------------------------
    cb_bytes = L * (K16 * 16*16 + K8 * 8*8 + K4 * 4*4) * 3   # 3 towers × L codebooks × K each
    bits_per_idx_16 = math.ceil(math.log2(max(K16, 2)))
    bits_per_idx_8  = math.ceil(math.log2(max(K8,  2)))
    bits_per_idx_4  = math.ceil(math.log2(max(K4,  2)))
    bits_per_leaf_16 = L * bits_per_idx_16
    bits_per_leaf_8  = L * bits_per_idx_8
    bits_per_leaf_4  = L * bits_per_idx_4

    def eval_thresholds(T16, T8):
        # 16 superblocks
        m16_keep = (mse_16_at16 <= T16)
        m8_in16  = (~m16_keep) & (mse_16_at8 <= T8)
        m4_in16  = (~m16_keep) & (~m8_in16)
        # 8 top-level
        m8_keep8 = (mse_8_at8 <= T8)
        m4_in8   = ~m8_keep8

        se = 0.0; n_texels = 0.0
        n_leaf_16 = 0; n_leaf_8 = 0; n_leaf_4 = 0
        n_split_bits = 0
        # From top-level 16-superblocks
        if m16_keep.any():
            se = se + (mse_16_at16[m16_keep] * 16*16*3).sum().item()
            n_texels += int(m16_keep.sum()) * 16*16
            n_leaf_16 += int(m16_keep.sum())
            n_split_bits += int(m16_keep.sum())                        # 1 "keep" flag
        if m8_in16.any():
            se = se + (mse_16_at8[m8_in16] * 4 * 8*8*3).sum().item()
            n_texels += int(m8_in16.sum()) * 16*16
            n_leaf_8 += int(m8_in16.sum()) * 4
            n_split_bits += int(m8_in16.sum()) * (1 + 4)               # split16 + 4 keep-flags
        if m4_in16.any():
            se = se + (mse_16_at4[m4_in16] * 16 * 4*4*3).sum().item()
            n_texels += int(m4_in16.sum()) * 16*16
            n_leaf_4 += int(m4_in16.sum()) * 16
            n_split_bits += int(m4_in16.sum()) * (1 + 4 + 16)          # full subdivide
        # Top-level 8-surfels
        if m8_keep8.any():
            se = se + (mse_8_at8[m8_keep8] * 8*8*3).sum().item()
            n_texels += int(m8_keep8.sum()) * 64
            n_leaf_8 += int(m8_keep8.sum())
            n_split_bits += int(m8_keep8.sum())
        if m4_in8.any():
            se = se + (mse_8_at4[m4_in8] * 4 * 4*4*3).sum().item()
            n_texels += int(m4_in8.sum()) * 64
            n_leaf_4 += int(m4_in8.sum()) * 4
            n_split_bits += int(m4_in8.sum()) * (1 + 4)
        # Top-level 4-surfels
        se = se + (mse_4_at4 * 4*4*3).sum().item()
        n_texels += X4_top.shape[0] * 16
        n_leaf_4 += X4_top.shape[0]

        mse = se / (n_texels * 3)
        psnr = -10.0 * math.log10(max(mse, 1e-20))
        bits_idx = (n_leaf_16 * bits_per_leaf_16
                    + n_leaf_8  * bits_per_leaf_8
                    + n_leaf_4  * bits_per_leaf_4)
        idx_bytes = (bits_idx + n_split_bits + 7) // 8
        total_mb = (idx_bytes + cb_bytes) / 1024 / 1024
        return {"T16": T16, "T8": T8,
                "n_leaf_16": int(m16_keep.sum()),
                "n_leaf_8": n_leaf_8,
                "n_leaf_4": n_leaf_4,
                "psnr": psnr, "total_mb": total_mb}

    # Baselines.
    flat = eval_thresholds(T16=-1, T8=-1)
    all16 = eval_thresholds(T16=1e9, T8=1e9)
    print(f"\n[REF] flat-4 RVQ L={L} K4={K4}:      "
          f"PSNR={flat['psnr']:.2f} dB  {flat['total_mb']:.2f} MB")
    print(f"[REF] all-16 (no subdivide) RVQ:    "
          f"PSNR={all16['psnr']:.2f} dB  {all16['total_mb']:.2f} MB  "
          f"(n_16={all16['n_leaf_16']:,}, n_8={all16['n_leaf_8']:,}, n_4={all16['n_leaf_4']:,})")

    if args.thresholds == "auto":
        e16_np = mse_16_at16.cpu().numpy()
        e8_pool = torch.cat([mse_16_at8.flatten(), mse_8_at8.flatten()]).cpu().numpy()
        configs = []
        for q16 in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            for q8 in [0.10, 0.25, 0.50, 0.75, 0.90]:
                configs.append((np.quantile(e16_np, q16),
                                np.quantile(e8_pool, q8),
                                f"q16={q16:.2f}  q8={q8:.2f}"))
    else:
        T16, T8 = [float(x) for x in args.thresholds.split(",")]
        configs = [(T16, T8, "manual")]

    print(f"\n{'config':>22}  {'n_16':>9}  {'n_8':>9}  {'n_4':>10}  "
          f"{'PSNR':>7}  {'MB':>7}  {'vs flat':>8}")
    rows = []
    for T16, T8, label in configs:
        r = eval_thresholds(T16, T8); r["label"] = label
        share = r["total_mb"] / flat["total_mb"] * 100
        rows.append((r, share))
        print(f"   {label:>20}  {r['n_leaf_16']:>9,}  {r['n_leaf_8']:>9,}  "
              f"{r['n_leaf_4']:>10,}  {r['psnr']:>5.2f}dB  {r['total_mb']:>6.2f}MB  "
              f"{share:>6.2f}%")


if __name__ == "__main__":
    main()
