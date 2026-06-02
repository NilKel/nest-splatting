"""
End-to-end VQ-bake (Phase 1: atlas-fidelity-driven RVQ training).

Replaces the post-hoc K-means RVQ in `bake_cluster_blocks_residual.py`
with a JOINTLY-TRAINED variant: codebooks and a continuous "soft"
atlas are both learnable, optimised with Adam under a VQ-VAE-style
loss with straight-through gradient through the argmin.

Why this beats post-hoc K-means RVQ:
- K-means freezes the source signal and only moves codewords toward
  assigned points. Each stage of post-hoc RVQ is a greedy fit to a
  static residual.
- VQ-VAE training also moves the SOURCE (the per-Gauss atlas patches)
  toward codebook-friendly positions via the commitment loss and the
  STE gradient. The atlas can "trade" some self-fidelity for much
  better codebook approximation. Typical gain: +0.5 to +1 dB at
  matched (L, C) over post-hoc RVQ.

What's optimised:
- A_soft : the per-Gauss atlas patches, initialised from the baked
           float32 atlas. FP32 learnable.
- C_1..L: L codebooks of K codewords each (each codeword = 48-D
           = one 4×4 RGB tile). Initialised from K-means on residuals.
What's frozen:
- The trained MLP, hashgrid, and Gaussian set (xyz, scale, rot, op, SH).
- The atlas rect packing (per-Gauss u0, v0, w, h).

Phase 1 loss (atlas-fidelity):
  L_recon  = ||VQ_recon  −  baked_atlas_target||²
  L_commit = β · ||A_soft − sg[VQ_recon]||²
  Total    = L_recon + L_commit

Phase 2 (render-loss) will replace L_recon with rendered-image MSE
once `diff_surfel_bake_render` has a backward pass. See docs/VQ_BAKE.md.

Output artifacts written to `<bake_dir>/vq/`:
  codebooks.pt    [(L, K, 48)] FP16 — the L codebooks
  indices.pt      [(L, N_blocks)] uint16/uint32 — per-stage block indices
  block_meta.pt   atlas dims, block grid, used-block ordering
  vq_meta.json    L, K, training config, final PSNR
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F


def kmeans_chunked(X, K, iters=15, seed=0, dist_chunk=None):
    """K-means used for codebook init."""
    if dist_chunk is None:
        dist_chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
    N, D = X.shape
    if K >= N:
        return X.clone()
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


def rvq_forward(A_blocks, codebooks):
    """L-stage RVQ over [N, D] blocks with JOINT codebook gradients.

    Greedy hard assignment (no-grad) per stage gives the index list.
    The reconstruction is then `Σ_l codebooks[l][ass_l]`, which is
    differentiable w.r.t. every codebook via index-select (gradient
    sums at each used codeword). STE puts identity gradient on A_blocks.

    Returns (recon, asses) where recon is the STE-wrapped sum-recon.
    """
    asses = []
    with torch.no_grad():
        R = A_blocks
        for cb in codebooks:
            K = cb.shape[0]
            dist_chunk = max(2_000, min(2_000_000, 500_000_000 // (K * 4)))
            N = R.shape[0]
            ass = torch.empty(N, dtype=torch.long, device=R.device)
            cn2 = (cb * cb).sum(1)
            for s in range(0, N, dist_chunk):
                e = min(s + dist_chunk, N)
                d = -2.0 * (R[s:e] @ cb.T) + cn2.unsqueeze(0)
                ass[s:e] = d.argmin(1)
                del d
            asses.append(ass)
            R = R - cb[ass]                                        # next-stage residual
    # Differentiable reconstruction: gradient flows to ALL codebooks via
    # index-select. STE on A_blocks (identity backward).
    hard_recon = sum(codebooks[l][asses[l]] for l in range(len(codebooks)))
    recon = A_blocks + (hard_recon - A_blocks).detach()
    return recon, asses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--L", type=int, default=4, help="RVQ stages.")
    p.add_argument("--K", type=int, default=256, help="Codewords per stage.")
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=500_000,
                   help="Block batch per iter (random subset of used blocks).")
    p.add_argument("--lr_codebook", type=float, default=1e-3)
    p.add_argument("--lr_atlas", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=0.25,
                   help="Commitment-loss weight (VQ-VAE β).")
    p.add_argument("--init_kmeans_iters", type=int, default=15)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", type=str, default=None,
                   help="Override output dir (default <bake_dir>/vq/).")
    args = p.parse_args()
    B = args.block

    out_dir = args.out_dir or os.path.join(args.bake_dir, "vq")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load baked atlas ---------------------------------------------------
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
    # Build A_target with the absolute minimum simultaneously-alive tensors.
    # Treehill (14976×54528 atlas, 47M used blocks) needs every byte: atlas_u8
    # is 2.4 GB, atlas_f is 9.6 GB, atlas_blocks (unfold().contiguous()) is
    # another 9.6 GB, and A_target is 9.1 GB → 30+ GB peak if all four are
    # alive at once. Free each tensor the moment its successor is done.
    atlas_f = atlas_u8.float() / 255.0 * a_scale + a_off
    del atlas_u8
    atlas_blocks = atlas_f.unfold(0, B, B).unfold(1, B, B).permute(0, 1, 3, 4, 2).contiguous()
    del atlas_f
    torch.cuda.empty_cache()
    A_target = atlas_blocks[used_idx[:, 0], used_idx[:, 1]].reshape(n_used, -1).contiguous()
    del atlas_blocks
    torch.cuda.empty_cache()
    D = A_target.shape[1]
    print(f"[LOAD] {n_used:,} used 4×4 blocks; D={D}; atlas {H}×{W}")
    print(f"[VQ-BAKE] L={args.L} stages, K={args.K} codewords, β={args.beta}\n")

    # ---- Initialize codebooks via post-hoc K-means RVQ ---------------------
    # Same recurrence as bake_cluster_blocks_residual.py — gives a strong
    # starting point so Adam can refine rather than search from scratch.
    print("[INIT] running L-stage K-means RVQ on baked atlas as warm-start…")
    perm_init = torch.randperm(n_used, device='cuda')[:1_500_000]
    codebooks_init = []
    recon_running = torch.zeros_like(A_target)
    # treehill (50.9M used blocks, 14976×54528 atlas) cannot afford 3×N×D
    # residual tensors (30 GB on a 31 GB 5090). Compute R = A_target -
    # recon_running on the fly per chunk — only the K-means fit subset
    # (perm_init, 1.5M rows = ~290 MB) and the per-chunk residual are kept
    # alive. Net peak: A_target + recon_running + R_fit + chunks = 2× N×D
    # instead of 3×.
    for l in range(args.L):
        t0 = time.time()
        # Build R_fit = (A_target - recon_running)[perm_init] in one shot —
        # it's only 290 MB so chunking isn't needed; the gather is the
        # memory peak (perm_init.numel() × D × 4 bytes).
        R_fit = A_target[perm_init] - recon_running[perm_init]
        cb = kmeans_chunked(R_fit, args.K, iters=args.init_kmeans_iters)
        del R_fit
        codebooks_init.append(cb)
        # Hard-assign over the full block set + accumulate into recon_running.
        # R_chunk is freed inside the loop; total peak adds < 1 GB beyond
        # A_target + recon_running.
        with torch.no_grad():
            cn2 = (cb * cb).sum(1)
            chunk = max(2_000, min(2_000_000, 500_000_000 // (args.K * 4)))
            for s in range(0, n_used, chunk):
                e = min(s + chunk, n_used)
                R_chunk = A_target[s:e] - recon_running[s:e]
                d = -2.0 * (R_chunk @ cb.T) + cn2.unsqueeze(0)
                ass = d.argmin(1)
                recon_running[s:e] = recon_running[s:e] + cb[ass]
                del d, ass, R_chunk
            # Chunked SE accumulator — `(recon - target).pow(2).sum()` over the
            # whole tensor allocates another N×D temporary which OOMs on
            # ≥30 M-block atlases.
            se = 0.0
            for s in range(0, n_used, chunk):
                e = min(s + chunk, n_used)
                se += (recon_running[s:e] - A_target[s:e]).pow(2).sum().item()
        psnr0 = -10.0 * math.log10(max(se / (n_used * D), 1e-20))
        print(f"  stage {l+1}/{args.L}: K-means + assign ({time.time()-t0:.1f}s), "
              f"cumulative PSNR = {psnr0:.2f} dB")
    print(f"[INIT] K-means RVQ baseline = {psnr0:.2f} dB\n")

    # ---- Setup learnable parameters -----------------------------------------
    #
    # `--iters 0` skips the Adam refine loop entirely (no A_soft, no Adam
    # state, no commitment loss). Useful when (a) the K-means warm-start is
    # already good enough (large scenes saturate around 40 dB at L=4/K=256;
    # the Adam refine adds < 1 dB) or (b) the model is too big to hold
    # A_soft + Adam moments in GPU memory (bonsai-class atlases at 20M+
    # blocks blow 31 GB on the Adam step's foreach_sqrt scratch).
    if args.iters > 0:
        A_soft = torch.nn.Parameter(A_target.clone())              # [N, D] FP32
        codebooks = torch.nn.ParameterList([
            torch.nn.Parameter(cb.clone()) for cb in codebooks_init
        ])
        opt = torch.optim.Adam([
            {"params": [A_soft], "lr": args.lr_atlas},
            {"params": list(codebooks), "lr": args.lr_codebook},
        ])
    else:
        A_soft = A_target                                          # no parameter copy
        codebooks = codebooks_init                                  # plain tensors
        opt = None
        print("[INFO] --iters 0: skipping Adam refine (using K-means warm-start directly)")

    # ---- Training loop ------------------------------------------------------
    t_train = time.time()
    log_lines = []
    best_psnr = psnr0
    for it in range(args.iters):
        # Random batch of blocks (full set per epoch ≈ n_used // batch iters).
        idx = torch.randint(0, n_used, (args.batch,), device='cuda')
        A_batch_soft = A_soft[idx]
        A_batch_target = A_target[idx]
        recon, asses = rvq_forward(A_batch_soft, list(codebooks))

        # Losses.
        L_recon  = (recon - A_batch_target).pow(2).mean()
        # Commitment: encourage A_soft to stay close to the running recon
        # of codebook entries (stop-grad on the codebook side so it doesn't
        # collapse to A_soft).
        L_commit = args.beta * (A_batch_soft - recon.detach()).pow(2).mean()
        L = L_recon + L_commit

        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()

        if (it + 1) % args.log_every == 0 or it == 0:
            with torch.no_grad():
                # Full-set PSNR for honest comparison.
                se = 0.0
                rchunk = 500_000
                for s in range(0, n_used, rchunk):
                    e = min(s + rchunk, n_used)
                    r, _ = rvq_forward(A_soft[s:e], list(codebooks))
                    se += (r - A_target[s:e]).pow(2).sum().item()
                full_psnr = -10.0 * math.log10(max(se / (n_used * D), 1e-20))
                if full_psnr > best_psnr:
                    best_psnr = full_psnr
                line = (f"  iter {it+1:>5}/{args.iters}  "
                        f"L_recon={L_recon.item():.4e}  L_commit={L_commit.item():.4e}  "
                        f"full PSNR={full_psnr:.2f} dB  (best {best_psnr:.2f}, "
                        f"K-means baseline {psnr0:.2f})")
                print(line)
                log_lines.append(line)

    train_time = time.time() - t_train
    print(f"\n[DONE] training {train_time:.1f}s; "
          f"K-means init {psnr0:.2f} dB → trained {best_psnr:.2f} dB "
          f"(Δ = +{best_psnr-psnr0:.2f} dB)")

    # ---- Save artifacts -----------------------------------------------------
    # Free the warmup-loop's recon_running BEFORE the save block re-allocates
    # its own. Without this, garden/treehill-class atlases (38–50M blocks ≈
    # 7-10 GB per N×D tensor) keep 4× N×D alive (warmup R + warmup
    # recon_running + A_target + save recon_running) and OOM the 31 GB 5090.
    # (warmup's R was removed in favor of on-the-fly per-chunk residuals.)
    del recon_running
    torch.cuda.empty_cache()
    with torch.no_grad():
        # Final assignment over the full block set. On-the-fly residual
        # computation (no persistent R tensor) — saves 9.8 GB on treehill.
        recon_running = torch.zeros_like(A_target)
        all_ass = []
        A_src = A_soft.detach()        # alias of A_target when iters=0
        for cb in codebooks:
            chunk = max(2_000, min(2_000_000, 500_000_000 // (args.K * 4)))
            ass_full = torch.empty(n_used, dtype=torch.long, device='cuda')
            cn2 = (cb * cb).sum(1)
            # Per-chunk: build R_chunk = A_src - recon_running on the fly,
            # compute distances, accumulate. Memory peak per stage is one
            # chunk × D ≈ 500 MB.
            for s in range(0, n_used, chunk):
                e = min(s + chunk, n_used)
                R_chunk = A_src[s:e] - recon_running[s:e]
                d = -2.0 * (R_chunk @ cb.T) + cn2.unsqueeze(0)
                ass_full[s:e] = d.argmin(1)
                recon_running[s:e] = recon_running[s:e] + cb[ass_full[s:e]]
                del d, R_chunk
            all_ass.append(ass_full)
        # Chunked SE accumulator (matches warmup).
        se = 0.0
        for s in range(0, n_used, chunk):
            e = min(s + chunk, n_used)
            se += (recon_running[s:e] - A_target[s:e]).pow(2).sum().item()
        final_psnr = -10.0 * math.log10(max(se / (n_used * D), 1e-20))

    cb_tensor = torch.stack([cb.detach().half() for cb in codebooks])     # [L, K, D]
    # Pack indices into the smallest int dtype that fits.
    if args.K <= 256:
        idx_dtype = torch.uint8
    elif args.K <= 65536:
        idx_dtype = torch.int16                          # uint16 not supported by torch.save round-trip; int16 = same 2 B
    else:
        idx_dtype = torch.int32
    idx_tensor = torch.stack([a.to(idx_dtype) for a in all_ass])          # [L, N]
    torch.save(cb_tensor, os.path.join(out_dir, "codebooks.pt"))
    torch.save(idx_tensor, os.path.join(out_dir, "indices.pt"))
    # `used_idx` is a deterministic function of atlas_rects.pt (in <bake_dir>);
    # don't save it — re-derive at load time. Block_meta now stores only the
    # 4 scalars needed to recompute it.
    torch.save({
        "atlas_HW": (int(H), int(W)),
        "block": int(B),
        "n_used_blocks": int(n_used),
        "atlas_scale": a_scale,
        "atlas_offset": a_off,
    }, os.path.join(out_dir, "block_meta.pt"))
    cfg = {
        "L": args.L, "K": args.K, "block": args.block, "D": int(D),
        "iters": args.iters, "batch": args.batch,
        "lr_codebook": args.lr_codebook, "lr_atlas": args.lr_atlas,
        "beta": args.beta, "init_kmeans_iters": args.init_kmeans_iters,
        "kmeans_baseline_psnr": float(psnr0),
        "final_psnr": float(final_psnr),
        "train_time_sec": float(train_time),
        "n_used_blocks": int(n_used),
        "codebook_bytes": int(args.L * args.K * D),
        "index_bytes": int((n_used * args.L * math.ceil(math.log2(max(args.K, 2))) + 7) // 8),
    }
    cfg["total_mb"] = (cfg["codebook_bytes"] + cfg["index_bytes"]) / 1024 / 1024
    cfg["vs_single_bc7_pct"] = 100.0 * cfg["total_mb"] / (n_used * B * B / 1024 / 1024)
    with open(os.path.join(out_dir, "vq_meta.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[SAVE] codebooks {tuple(cb_tensor.shape)} + indices {tuple(idx_tensor.shape)} "
          f"→ {out_dir}")
    print(f"       total = {cfg['total_mb']:.2f} MB "
          f"({cfg['vs_single_bc7_pct']:.2f} % vs single-bake BC7), "
          f"final PSNR = {final_psnr:.2f} dB")


if __name__ == "__main__":
    main()
