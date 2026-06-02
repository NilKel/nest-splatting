"""
Adaptive quadtree block-size VQ on the atlas blocks.

Each 16×16 atlas region picks the LARGEST block-size that hits a
per-region reconstruction-error budget:

  16×16 → 8×8 (×4) → 4×4 (×16)

Storage per 16×16 region:
  9 bits  if 16×16 leaf       (1 split flag + 1 byte index)
  37 bits if all 4 sub-blocks at 8×8
  133 bits if all 16 sub-sub-blocks at 4×4 (worst case)

vs flat 4×4: 16 × 8 = 128 bits/16×16 area.

Three independent codebooks:
  K_16: 256 codewords of (16·16·3)=768 B each → 192 KB at K=256
  K_8:  256 codewords of 192 B    → 48 KB
  K_4:  256 codewords of 48 B     → 12 KB
Total codebook bytes ≈ 252 KB at K=256 (rounding error vs indices).

Surfels too small for 16×16 (rect dim < 16) start at the largest fitting
block size. The quadtree split decisions are deterministic given the
per-block error and the (T_16, T_8) thresholds — no full search.
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
    cn2 = (cents * cents).sum(1)
    for s in range(0, N, dist_chunk):
        e = min(s + dist_chunk, N)
        d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return cents, ass


def assign_full(X, cents):
    K = cents.shape[0]
    cn2 = (cents * cents).sum(1)
    N = X.shape[0]
    ass = torch.empty(N, dtype=torch.long, device=X.device)
    chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = -2.0 * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return ass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--K", type=int, default=256,
                   help="Codebook size for ALL three block-size codebooks.")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--max_blocks_for_kmeans", type=int, default=1_500_000)
    p.add_argument("--thresholds", type=str, default="auto",
                   help="Either 'auto' (sweeps a range) or '<T16>,<T8>' "
                        "for a single (16→8 and 8→4) error threshold pair.")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    H, W, _ = atlas_u8.shape
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off                 # [H, W, 3]
    del atlas_u8
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    u0 = rects[:, 0].long(); v0 = rects[:, 1].long()
    w_ = rects[:, 2].long(); h_ = rects[:, 3].long()
    print(f"[LOAD] atlas {H}×{W}, {rects.shape[0]} surfels")

    # ---- Build 3 tilings ----------------------------------------------------
    # For each surfel, decide max block size B_top ∈ {16, 8, 4} that divides
    # both rect_h and rect_w. Then build:
    #   blocks_16[N_16, 768]  — 16×16 RGB tiles from surfels with B_top=16
    #   blocks_8 [N_8,  192]  — 8×8 RGB tiles from surfels with B_top=8
    #   blocks_4 [N_4,   48]  — 4×4 RGB tiles from surfels with B_top=4
    # PLUS we'll also build the recursive children:
    #   children_8 (4 sub-blocks per 16×16 superblock that needs subdivide)
    #   children_4 (4 sub-blocks per 8×8 block that needs subdivide)
    # All blocks come from the same atlas → reconstruction is a tiled write.

    # First: full atlas-block tiling at 4×4 (we'll group up for 8/16).
    B4 = 4
    blocks_4_all = atlas_f.unfold(0, B4, B4).unfold(1, B4, B4).permute(0, 1, 3, 4, 2).contiguous()  # [H/4, W/4, 4, 4, 3]
    # 8×8 grouping: [H/8, W/8, 2, 2, 4, 4, 3] → reshape to [H/8, W/8, 8, 8, 3]
    blocks_8_all = atlas_f.unfold(0, 8, 8).unfold(1, 8, 8).permute(0, 1, 3, 4, 2).contiguous()      # [H/8, W/8, 8, 8, 3]
    # 16×16
    blocks_16_all = atlas_f.unfold(0, 16, 16).unfold(1, 16, 16).permute(0, 1, 3, 4, 2).contiguous() # [H/16, W/16, 16, 16, 3]

    # Per-surfel max-block-size decision.
    sizes = []
    for i in range(rects.shape[0]):
        ww = int(w_[i]); hh = int(h_[i])
        if ww == 0 or hh == 0:
            sizes.append(0); continue
        if ww >= 16 and hh >= 16 and ww % 16 == 0 and hh % 16 == 0:
            sizes.append(16)
        elif ww >= 8 and hh >= 8 and ww % 8 == 0 and hh % 8 == 0:
            sizes.append(8)
        else:
            sizes.append(4)
    sizes = torch.tensor(sizes, device='cuda')
    n16 = int((sizes == 16).sum())
    n8  = int((sizes == 8).sum())
    n4  = int((sizes == 4).sum())
    print(f"[TILE] surfels by top-block-size: 16×16: {n16:,}  8×8: {n8:,}  4×4: {n4:,}")

    def gather_blocks_for_surfels(blocks_all, B, surf_idx_mask):
        """For each surfel in surf_idx_mask, gather all its B×B blocks.
        Returns flat tensor [N, B*B*3] and a [N] surfel-id tensor."""
        out_blocks = []
        out_surfel = []
        out_u0 = []                # block top-left in atlas (in B-aligned cells)
        out_v0 = []
        idxs = surf_idx_mask.nonzero(as_tuple=False).flatten()
        for i in idxs.tolist():
            ww = int(w_[i]); hh = int(h_[i])
            bu = int(u0[i]) // B
            bv = int(v0[i]) // B
            nu = ww // B; nv = hh // B
            # Slice
            patch = blocks_all[bv:bv+nv, bu:bu+nu]                       # [nv, nu, B, B, 3]
            patch_flat = patch.reshape(-1, B*B*3)
            out_blocks.append(patch_flat)
            out_surfel.append(torch.full((nv * nu,), i, dtype=torch.long, device='cuda'))
            # absolute B-cell coords
            grid_v = bv + torch.arange(nv, device='cuda').unsqueeze(1).expand(nv, nu)
            grid_u = bu + torch.arange(nu, device='cuda').unsqueeze(0).expand(nv, nu)
            out_v0.append(grid_v.reshape(-1))
            out_u0.append(grid_u.reshape(-1))
        if not out_blocks:
            empty_b = torch.empty(0, B*B*3, device='cuda')
            empty_l = torch.empty(0, dtype=torch.long, device='cuda')
            return empty_b, empty_l, empty_l, empty_l
        return (torch.cat(out_blocks, 0),
                torch.cat(out_surfel, 0),
                torch.cat(out_v0, 0),
                torch.cat(out_u0, 0))

    t0 = time.time()
    X16, surfel_16, v0_16, u0_16 = gather_blocks_for_surfels(blocks_16_all, 16, sizes == 16)
    del blocks_16_all; torch.cuda.empty_cache()
    X8_top, surfel_8, v0_8, u0_8 = gather_blocks_for_surfels(blocks_8_all, 8, sizes == 8)
    del blocks_8_all; torch.cuda.empty_cache()
    X4_top, surfel_4, v0_4, u0_4 = gather_blocks_for_surfels(blocks_4_all, 4, sizes == 4)
    del blocks_4_all, atlas_f; torch.cuda.empty_cache()
    print(f"[TILE] gathered top-level blocks: 16: {X16.shape[0]:,} | 8: {X8_top.shape[0]:,} | 4: {X4_top.shape[0]:,}  ({time.time()-t0:.1f}s)")

    # ---- Train 3 codebooks (one per block size) ----------------------------
    print(f"[CLUSTER] training 3 codebooks at K={args.K} …")
    t0 = time.time()
    # 16×16 codebook from top-level 16 blocks
    fit16 = X16[torch.randperm(X16.shape[0], device='cuda')[:args.max_blocks_for_kmeans]] if X16.shape[0] > args.max_blocks_for_kmeans else X16
    cents_16, _ = kmeans_chunked(fit16, args.K, iters=args.iters)
    # 8×8 codebook: need to TRAIN on what an 8×8 block could be. Sample from:
    #   (a) the 4 sub-blocks of every 16×16 (top-level) for downstream subdivision
    #   (b) plus the surfels that natively start at 8×8.
    # Just (b) is fine for codebook coverage; let's use top-level 8s + sub-8s of top-level 16s.
    X16_sub8 = X16.reshape(-1, 2, 8, 2, 8, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 8*8*3)  # 4 8×8 children per 16×16
    pool8 = torch.cat([X8_top, X16_sub8], 0)
    fit8 = pool8[torch.randperm(pool8.shape[0], device='cuda')[:args.max_blocks_for_kmeans]] if pool8.shape[0] > args.max_blocks_for_kmeans else pool8
    cents_8, _ = kmeans_chunked(fit8, args.K, iters=args.iters)
    # 4×4 codebook: every 4×4 block from the atlas (top-level 4s, sub-4s of 8s and 16s).
    X8_sub4 = pool8.reshape(-1, 2, 4, 2, 4, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4*4*3)
    pool4 = torch.cat([X4_top, X8_sub4], 0)
    fit4 = pool4[torch.randperm(pool4.shape[0], device='cuda')[:args.max_blocks_for_kmeans]] if pool4.shape[0] > args.max_blocks_for_kmeans else pool4
    cents_4, _ = kmeans_chunked(fit4, args.K, iters=args.iters)
    print(f"[CLUSTER] codebooks trained ({time.time()-t0:.1f}s); "
          f"shapes: 16x16={tuple(cents_16.shape)}, 8x8={tuple(cents_8.shape)}, 4x4={tuple(cents_4.shape)}")

    # ---- Assign + compute per-block error at each scale ---------------------
    # For each 16×16 superblock, we need:
    #   E_16 — error if kept as 16×16 leaf.
    #   E_8  — error if all 4 sub-blocks become 8×8 leaves.
    #   E_4  — error if all 16 sub-sub-blocks become 4×4 leaves.
    # Compare to the user's pair of thresholds.

    # Helper: per-block MSE = ||X − cents[idx]||² / D. Chunked to avoid
    # materialising N×D recon tensors at large N.
    def per_block_mse(X, cents, ass, chunk=300_000):
        out = torch.empty(X.shape[0], device=X.device)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            recon = cents[ass[s:e]]
            out[s:e] = (recon - X[s:e]).pow(2).mean(dim=1)
            del recon
        return out

    print(f"[ASSIGN] computing assignments + per-block errors …")
    t0 = time.time()
    # Top-level 16 blocks
    ass_16   = assign_full(X16,        cents_16)
    e16_at16 = per_block_mse(X16,      cents_16, ass_16)                 # [N_16]
    # 16's sub-8 blocks → assign to cents_8
    X16_sub8 = X16.reshape(-1, 2, 8, 2, 8, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 8*8*3)
    ass_8_in16 = assign_full(X16_sub8, cents_8)
    e16_at8  = per_block_mse(X16_sub8, cents_8, ass_8_in16).reshape(-1, 4).mean(1)
    # 16's sub-4 blocks → assign to cents_4
    X16_sub4 = X16.reshape(-1, 4, 4, 4, 4, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4*4*3)
    ass_4_in16 = assign_full(X16_sub4, cents_4)
    e16_at4  = per_block_mse(X16_sub4, cents_4, ass_4_in16).reshape(-1, 16).mean(1)
    # Top-level 8 blocks (surfels too small for 16): always at 8 or descend to 4
    ass_8_top = assign_full(X8_top, cents_8)
    e8_at8  = per_block_mse(X8_top, cents_8, ass_8_top)
    X8_sub4 = X8_top.reshape(-1, 2, 4, 2, 4, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4*4*3)
    ass_4_in8 = assign_full(X8_sub4, cents_4)
    e8_at4  = per_block_mse(X8_sub4, cents_4, ass_4_in8).reshape(-1, 4).mean(1)
    # Top-level 4 blocks (surfels too small for 8)
    ass_4_top = assign_full(X4_top, cents_4)
    e4_at4  = per_block_mse(X4_top, cents_4, ass_4_top)
    print(f"[ASSIGN] done ({time.time()-t0:.1f}s)")

    # ---- Total reconstructable variance for PSNR conversion ----------------
    # PSNR(X) = -10 log10(MSE / dynamic_range²). Use dynamic_range²=1 (matches
    # our atlas-fidelity convention used elsewhere in §4.5).
    total_n_texels = (X16.numel() + X8_top.numel() + X4_top.numel()) / 3   # texels (not channels)
    total_n_blocks_4equiv = (X16.shape[0] * 16 + X8_top.shape[0] * 4 + X4_top.shape[0]) if False else 0  # not used

    # ---- Sweep / pick thresholds -------------------------------------------
    # We sweep error-quantile thresholds for a clean storage / PSNR curve.
    # leaf_size_decision per top-level 16 superblock: pick smallest B such
    # that subdividing further wouldn't help much (error already low). The
    # canonical rule:
    #   if e16_at16 <= T16: leaf = 16
    #   elif e16_at8  <= T8:  leaf = 8
    #   else:                leaf = 4
    # Top-level 8: if e8_at8 <= T8: leaf=8, else: leaf=4.
    # Top-level 4: always leaf=4.

    def eval_thresholds(T16, T8):
        # 16 superblocks: choose leaf size 16 / 8 / 4
        m16_keep = (e16_at16 <= T16)
        m8_in16  = (~m16_keep) & (e16_at8 <= T8)
        m4_in16  = (~m16_keep) & (~m8_in16)
        # 8 top-level: keep 8 vs descend to 4
        m8_keep8  = (e8_at8 <= T8)
        m4_in8    = ~m8_keep8

        # Per-block weighted MSE — every leaf contributes its texels.
        # Sum (per_block_mse × texel_count) / total_texels.
        se = 0.0
        n_texels = 0.0
        n_idx_16 = 0; n_idx_8 = 0; n_idx_4 = 0
        n_split_bits = 0
        # 16-leaves
        if m16_keep.any():
            se = se + (e16_at16[m16_keep] * 16*16*3).sum().item()
            n_texels += int(m16_keep.sum()) * 16*16
            n_idx_16 += int(m16_keep.sum())
            n_split_bits += int(m16_keep.sum())                # 1 bit "no split" per superblock
        # 8-leaves under 16
        if m8_in16.any():
            # mean over 4 sub-blocks already; multiply by 4 × 64*3 texels-of-error
            se = se + (e16_at8[m8_in16] * 4 * 8*8*3).sum().item()
            n_texels += int(m8_in16.sum()) * 16*16
            n_idx_8 += int(m8_in16.sum()) * 4
            n_split_bits += int(m8_in16.sum()) * (1 + 4)       # 1 split bit + 4 leaf flags
        # 4-leaves under 16
        if m4_in16.any():
            se = se + (e16_at4[m4_in16] * 16 * 4*4*3).sum().item()
            n_texels += int(m4_in16.sum()) * 16*16
            n_idx_4 += int(m4_in16.sum()) * 16
            n_split_bits += int(m4_in16.sum()) * (1 + 4 + 16)  # full subdivision
        # Top-level 8 keep 8
        if m8_keep8.any():
            se = se + (e8_at8[m8_keep8] * 8*8*3).sum().item()
            n_texels += int(m8_keep8.sum()) * 64
            n_idx_8 += int(m8_keep8.sum())
            n_split_bits += int(m8_keep8.sum())
        # Top-level 8 descend to 4
        if m4_in8.any():
            se = se + (e8_at4[m4_in8] * 4 * 4*4*3).sum().item()
            n_texels += int(m4_in8.sum()) * 64
            n_idx_4 += int(m4_in8.sum()) * 4
            n_split_bits += int(m4_in8.sum()) * (1 + 4)
        # Top-level 4
        se = se + (e4_at4 * 4*4*3).sum().item()
        n_texels += X4_top.shape[0] * 16
        n_idx_4 += X4_top.shape[0]
        # No split bits for top-level 4 (no choice).

        mse = se / (n_texels * 3)                              # /3 because se sums over RGB
        psnr = -10.0 * math.log10(max(mse, 1e-20))
        bits_idx = (n_idx_16 + n_idx_8 + n_idx_4) * 8          # K=256 → 8 bits/idx
        bytes_idx = (bits_idx + n_split_bits + 7) // 8
        cb_bytes = args.K * (16*16 + 8*8 + 4*4) * 3            # uint8 BC7-equiv
        total_mb = (bytes_idx + cb_bytes) / 1024 / 1024
        return {
            "T16": T16, "T8": T8,
            "n_leaf_16": int(m16_keep.sum()),
            "n_leaf_8":  n_idx_8,
            "n_leaf_4":  n_idx_4,
            "psnr": psnr,
            "total_mb": total_mb,
            "split_bits": n_split_bits,
        }

    # Reference: all-4 leaves (flat 4×4 single-stage VQ).
    flat = eval_thresholds(T16=-1, T8=-1)
    # Reference: all-16 leaves where possible (no subdivide).
    all16 = eval_thresholds(T16=1e9, T8=1e9)
    print(f"\n[REF] flat 4×4 single-stage VQ:    "
          f"PSNR={flat['psnr']:.2f} dB  {flat['total_mb']:.2f} MB  "
          f"(idx_4={flat['n_leaf_4']:,})")
    print(f"[REF] all-16 (no subdivide, lossy): "
          f"PSNR={all16['psnr']:.2f} dB  {all16['total_mb']:.2f} MB  "
          f"(idx_16={all16['n_leaf_16']:,}, idx_8={all16['n_leaf_8']:,}, idx_4={all16['n_leaf_4']:,})")

    # Sweep thresholds.
    if args.thresholds == "auto":
        # Pick T16 / T8 percentiles of the per-16 and per-8 error distributions
        # so we get a clean trade-off curve.
        e16_at16_np = e16_at16.cpu().numpy()
        e8_pool_np  = torch.cat([e16_at8.flatten(), e8_at8.flatten()]).cpu().numpy()
        configs = []
        for q16 in [0.10, 0.25, 0.50, 0.75, 0.90]:
            for q8 in [0.10, 0.25, 0.50, 0.75, 0.90]:
                configs.append((np.quantile(e16_at16_np, q16),
                                np.quantile(e8_pool_np, q8),
                                f"q16={q16:.2f}  q8={q8:.2f}"))
    else:
        T16, T8 = [float(x) for x in args.thresholds.split(",")]
        configs = [(T16, T8, "manual")]

    print(f"\n{'config':>22}  {'leaf 16':>9}  {'leaf 8':>9}  {'leaf 4':>10}  "
          f"{'PSNR':>7}  {'MB':>7}  {'vs flat 4':>9}")
    for T16, T8, label in configs:
        r = eval_thresholds(T16, T8)
        share = r["total_mb"] / flat["total_mb"] * 100
        print(f"   {label:>20}  {r['n_leaf_16']:>9,}  {r['n_leaf_8']:>9,}  "
              f"{r['n_leaf_4']:>10,}  {r['psnr']:>5.2f}dB  {r['total_mb']:>6.2f}MB  "
              f"{share:>7.2f}%")


if __name__ == "__main__":
    main()
