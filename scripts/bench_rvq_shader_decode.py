"""
Micro-benchmark: per-fragment RVQ atlas decode (the work a WGSL fragment
shader would do at sample-time, simulated in CUDA via PyTorch ops).

Compares throughput of:
  A. UINT8 atlas sample      — single bilinear texel-fetch (baseline,
                                proxy for the current BC7 path)
  B. RVQ shader decode       — L codebook lookups + sum per fragment
                                (4-tap variant for true bilinear is also
                                measured)

Inputs are loaded from <bake_dir>/vq/{codebooks,indices,block_meta}.pt
and <bake_dir>/atlas_rects.pt — same artifacts the production decoder
would have.

Methodology:
- Synthetic fragments: sample N_FRAG random (surfel_id, local_uv) pairs.
  This matches what a typical frame would generate after the
  2DGS rasterizer figures out which surfels each pixel hits.
- Per fragment: compute its atlas (u, v), look up the residual.
- Both A and B compute on the SAME fragments and return RGB.
- Sanity check: A and B should agree (within VQ-quant noise).
- Wall-clock via cuda.Event; warm-up N_WARMUP, then average N_TRIAL.

Output a fragments-per-second number for each, plus the
peak-bandwidth-bound theoretical ceiling for context.
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--n_frag", type=int, default=2_000_000,
                   help="Fragments per trial. 2M ≈ a full 1920x1080 frame.")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_trial", type=int, default=20)
    p.add_argument("--no_bilinear", action="store_true",
                   help="Use nearest-neighbour instead of bilinear (faster, "
                        "lower quality — matches a WGSL `filter: nearest` sampler).")
    args = p.parse_args()

    device = 'cuda'

    # ------------ Load RVQ artifacts + rects --------------------------------
    print(f"[LOAD] {args.bake_dir}")
    codebooks = torch.load(os.path.join(args.bake_dir, "vq/codebooks.pt"),
                           map_location=device, weights_only=False).float()  # [L, K, 48]
    indices = torch.load(os.path.join(args.bake_dir, "vq/indices.pt"),
                         map_location=device, weights_only=False).long()     # [L, N_blocks]
    block_meta = torch.load(os.path.join(args.bake_dir, "vq/block_meta.pt"),
                            map_location=device, weights_only=False)
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location=device, weights_only=False).long()       # [M, 4] (u0,v0,w,h)
    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    H = int(block_meta["atlas_HW"][0]); W = int(block_meta["atlas_HW"][1])
    B = int(block_meta["block"])
    L, K, D = codebooks.shape
    print(f"  L={L} K={K} D={D} (block={B}x{B} RGB)")
    print(f"  atlas {H}x{W}, {rects.shape[0]:,} surfels, {indices.shape[1]:,} used blocks")
    print(f"  codebook = {codebooks.element_size()*codebooks.numel()/1024:.1f} KB FP32 in mem")
    print(f"  indices  = {indices.shape[1]*L:,} bytes uint8 ({indices.shape[1]*L/1024/1024:.1f} MB)")

    # ------------ Per-surfel block-offset table -----------------------------
    # surfel_block_offset[g] = number of USED blocks in surfels [0..g-1], i.e.
    # the first global block index belonging to surfel g. (Surfels with w*h=0
    # contribute 0.)
    bw = rects[:, 2] // B
    bh = rects[:, 3] // B
    n_blocks_per_surfel = bw * bh
    surfel_offsets = torch.zeros(rects.shape[0] + 1, dtype=torch.long, device=device)
    surfel_offsets[1:] = torch.cumsum(n_blocks_per_surfel, dim=0)
    total_used = int(surfel_offsets[-1].item())
    assert total_used == indices.shape[1], \
           f"surfel-block sum {total_used} != indices N {indices.shape[1]}"
    print(f"  surfel_offsets table: {surfel_offsets.numel()*8/1024:.1f} KB")

    # ---- Reorder indices into SURFEL-major order ---------------------------
    # The producer writes indices in atlas-row-major order (the K-means
    # `used.nonzero()` iteration). For a shader-side decoder to compute
    # block_id = surfel_offsets[g] + bv_local*(w/B) + bu_local at zero
    # cost, we need indices in surfel-major order. Do that once at load
    # time. Production producer should write directly in surfel-major
    # order to avoid this remap; for the bench we do it on-the-fly.
    print(f"[REMAP] reordering indices from row-major to surfel-major …")
    t0 = time.time()
    # Build the row-major → surfel-major permutation.
    # For each surfel, get its (bv0, bu0, nv, nu); for each (bvl, bul) in
    # that surfel: row-major id = used_mask_cumsum[bv, bu] — i.e. count of
    # used blocks strictly before (bv, bu) in row-major.
    used_mask = torch.zeros((H // B, W // B), dtype=torch.bool, device=device)
    for i in range(rects.shape[0]):
        ww = int(rects[i, 2].item()); hh = int(rects[i, 3].item())
        if ww == 0 or hh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        used_mask[bv0:bv0 + hh//B, bu0:bu0 + ww//B] = True
    # Row-major rank for every grid cell — block_id at (bv, bu) for used cells.
    used_flat = used_mask.reshape(-1)
    row_major_id_flat = torch.cumsum(used_flat.long(), dim=0) - 1     # [(H/B)*(W/B)]
    row_major_id_flat[~used_flat] = -1
    row_major_id = row_major_id_flat.reshape(H // B, W // B)
    # Build the perm: for each surfel-major slot, what was its row-major id?
    perm = torch.empty(total_used, dtype=torch.long, device=device)
    for i in range(rects.shape[0]):
        nw = int(bw[i].item()); nh = int(bh[i].item())
        if nw == 0 or nh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        off = int(surfel_offsets[i].item())
        sub = row_major_id[bv0:bv0+nh, bu0:bu0+nw].reshape(-1)
        perm[off:off+nw*nh] = sub
    indices = indices[:, perm].contiguous()                          # surfel-major now
    print(f"  done in {time.time()-t0:.2f}s")
    del used_mask, used_flat, row_major_id_flat, row_major_id, perm

    # ------------ Build a uint8 atlas (baseline reference) ------------------
    # We need this for the baseline benchmark. In production this would be the
    # dequantized atlas in GPU memory (944 MB at uint8 RGBA — too big — but
    # for the bench we accept the memory; the comparison is per-fragment work).
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location=device, weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    if atlas_u8.dtype != torch.uint8:
        raise SystemExit(f"need uint8 atlas, got {atlas_u8.dtype}")
    atlas_f16 = ((atlas_u8.float() / 255.0) * a_scale + a_off).half().contiguous()
    print(f"  atlas FP16 in GPU mem: {atlas_f16.element_size()*atlas_f16.numel()/1024/1024:.1f} MB "
          f"(baseline only — not what RVQ would ship to GPU)")
    del atlas_u8; torch.cuda.empty_cache()

    # ------------ Synthetic fragments --------------------------------------
    # Pick random (surfel_id, local_uv) pairs.  Weight by surfel area so
    # large surfels (most of the frame) dominate, matching real fragment
    # distribution.
    g = torch.Generator(device=device).manual_seed(0)
    weights = (rects[:, 2] * rects[:, 3]).float()
    sample_surfels = torch.multinomial(weights, args.n_frag, replacement=True,
                                       generator=g)
    # Local UV in [0, w), [0, h)  (atlas-pixel space).
    rand_u = torch.rand(args.n_frag, generator=g, device=device)
    rand_v = torch.rand(args.n_frag, generator=g, device=device)
    rects_s = rects[sample_surfels]                                       # [N_frag, 4]
    u0 = rects_s[:, 0]; v0 = rects_s[:, 1]
    w_ = rects_s[:, 2]; h_ = rects_s[:, 3]
    # Pixel coords within surfel rect, real-valued for bilinear.
    pix_u = rand_u * (w_.float() - 1.0)
    pix_v = rand_v * (h_.float() - 1.0)
    atlas_u = u0.float() + pix_u                                          # [N_frag]
    atlas_v = v0.float() + pix_v
    # Block-local coords within surfel (which 4×4 the fragment lies in).
    bu_local = torch.clamp((pix_u / B).long(), max=(w_ // B - 1))         # [N_frag]
    bv_local = torch.clamp((pix_v / B).long(), max=(h_ // B - 1))
    intra_u = (pix_u.long() % B).long()                                   # [N_frag] 0..3
    intra_v = (pix_v.long() % B).long()
    # Global used-block index.
    bid = surfel_offsets[sample_surfels] + bv_local * (w_ // B) + bu_local  # [N_frag]
    intra_idx = intra_v * B + intra_u                                     # 0..15 within the 4×4

    # ------------ Baseline A: uint8 atlas sample (nearest only) -------------
    # The actual production path uses bilinear via tex2D; here we benchmark
    # the cheapest possible "look up the atlas at one pixel" which is the
    # *lower bound* on the baseline cost. Real BC7 path is bilinear so ~2-4×
    # more bandwidth.
    def baseline_nearest_kernel():
        au = atlas_u.long().clamp(0, W-1)
        av = atlas_v.long().clamp(0, H-1)
        rgb = atlas_f16[av, au]                                           # [N_frag, 3]
        return rgb

    def baseline_bilinear_kernel():
        # 4-tap bilinear sample, what tex2D does.
        au = atlas_u - 0.5; av = atlas_v - 0.5
        au0 = au.floor().long().clamp(0, W-1); au1 = (au0 + 1).clamp(0, W-1)
        av0 = av.floor().long().clamp(0, H-1); av1 = (av0 + 1).clamp(0, H-1)
        fu = (au - au0.float()).unsqueeze(1); fv = (av - av0.float()).unsqueeze(1)
        c00 = atlas_f16[av0, au0]; c01 = atlas_f16[av0, au1]
        c10 = atlas_f16[av1, au0]; c11 = atlas_f16[av1, au1]
        top = c00 * (1 - fu) + c01 * fu
        bot = c10 * (1 - fu) + c11 * fu
        return (top * (1 - fv) + bot * fv).half()

    # ------------ Method B: RVQ shader decode (per-fragment) -----------------
    # 1 fragment → 1 block ID → L codeword indices → L lookups of (block-pixel)
    # entries → sum.
    def rvq_nearest_kernel():
        # Gather L codeword indices for this fragment's block.
        # indices: [L, N_used] long. Use advanced indexing.
        codes = indices[:, bid]                                            # [L, N_frag] long
        # Lookup codeword pixel values: codebook[l, codes[l, t], intra_idx[t]*3:intra_idx[t]*3+3]
        # codebooks: [L, K, D]. We want codebook[l, codes[l], intra_idx*3 : intra_idx*3+3] per fragment.
        # Easiest: gather flat (l, k) → [3]
        # Reshape codebook to [L, K, 16, 3]
        cb = codebooks.view(L, K, 16, 3)                                   # [L, K, 16, 3]
        # Per (l, codes[l, t], intra_idx[t]) → [3]. Use advanced indexing.
        # cb[l, codes[l, t], intra_idx[t]] = [3 floats]
        l_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, args.n_frag)  # [L, N_frag]
        contrib = cb[l_idx, codes, intra_idx.unsqueeze(0).expand(L, args.n_frag)]   # [L, N_frag, 3]
        return contrib.sum(dim=0)                                          # [N_frag, 3]

    def rvq_bilinear_kernel():
        # True 4-tap bilinear: each tap requires a separate block lookup
        # (taps can fall in different blocks). 4× the codebook bandwidth of
        # the nearest variant.
        au = atlas_u - 0.5; av = atlas_v - 0.5
        au0_f = au.floor(); av0_f = av.floor()
        fu = (au - au0_f).unsqueeze(1); fv = (av - av0_f).unsqueeze(1)

        def sample_at(du, dv):
            # Fragment at atlas (au0+du, av0+dv).
            af_u = (au0_f + du).long().clamp(0, W-1)
            af_v = (av0_f + dv).long().clamp(0, H-1)
            # Block decomposition: which block (within the same surfel) and
            # which pixel inside it. For a true cross-rect bilinear we'd
            # have to pick the rect each tap lies in; here we assume the
            # fragment lies inside one surfel's rect (the rasterizer
            # guarantees that for the SAME fragment, BUT not across
            # sub-pixel taps near a rect boundary). Approximation: use the
            # current fragment's surfel_id for all 4 taps — gives correct
            # bilinear in the interior, mild ringing at rect borders. (BC7
            # has the same issue; not worse.)
            local_u = torch.clamp(af_u - u0, min=0); local_u = torch.minimum(local_u, w_ - 1)
            local_v = torch.clamp(af_v - v0, min=0); local_v = torch.minimum(local_v, h_ - 1)
            bu_l = torch.minimum(local_u // B, (w_ // B - 1))
            bv_l = torch.minimum(local_v // B, (h_ // B - 1))
            bid_t = surfel_offsets[sample_surfels] + bv_l * (w_ // B) + bu_l
            intra_t = (local_v % B) * B + (local_u % B)
            cb = codebooks.view(L, K, 16, 3)
            codes = indices[:, bid_t]
            l_idx = torch.arange(L, device=device).unsqueeze(1).expand(L, args.n_frag)
            contrib = cb[l_idx, codes, intra_t.unsqueeze(0).expand(L, args.n_frag)]
            return contrib.sum(dim=0)                                      # [N_frag, 3]

        c00 = sample_at(0, 0); c01 = sample_at(1, 0)
        c10 = sample_at(0, 1); c11 = sample_at(1, 1)
        top = c00 * (1 - fu) + c01 * fu
        bot = c10 * (1 - fu) + c11 * fu
        return (top * (1 - fv) + bot * fv)

    # ------------ Sanity (RVQ-nearest should ≈ uint8-nearest) --------------
    with torch.no_grad():
        ref = baseline_nearest_kernel().float()
        a_rvq = rvq_nearest_kernel().float()
        mse = (a_rvq - ref).pow(2).mean().item()
        psnr = -10.0 * math.log10(max(mse / (a_scale**2), 1e-20))
        print(f"\n[SANITY] RVQ-nearest vs uint8-nearest (full-set PSNR): {psnr:.2f} dB "
              f"(mse={mse:.4e})  ← expect ≈ RVQ atlas-fidelity (42 dB)\n")

    # ------------ Benchmark loop -------------------------------------------
    def bench(fn, label):
        for _ in range(args.n_warmup): fn(); torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_trial)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_trial)]
        for t in range(args.n_trial):
            starts[t].record(); fn(); ends[t].record()
        torch.cuda.synchronize()
        times = sorted(starts[i].elapsed_time(ends[i]) for i in range(args.n_trial))
        med_ms = times[len(times)//2]
        frag_per_sec = args.n_frag / (med_ms / 1000.0)
        print(f"  {label:<28}  median {med_ms:>7.2f} ms / "
              f"{args.n_frag:,} frags  →  {frag_per_sec/1e9:.3f} G frag/s")
        return med_ms

    print(f"[BENCH] {args.n_frag:,} fragments per trial × {args.n_trial} trials\n")
    t_a_n = bench(baseline_nearest_kernel, "baseline uint8 nearest")
    t_a_b = bench(baseline_bilinear_kernel, "baseline uint8 bilinear")
    t_b_n = bench(rvq_nearest_kernel, f"RVQ nearest (L={L} taps)")
    t_b_b = bench(rvq_bilinear_kernel, f"RVQ bilinear (4·L={4*L} taps)")

    print(f"\n[SUMMARY] @ 1920×1080 = 2.07 M fragments / frame:")
    print(f"  baseline uint8 nearest  → {2_073_600/(t_a_n*1e-3)/1e6:>6.1f} M frag/s "
          f"⇒ {1000.0/t_a_n*args.n_frag/2_073_600:>6.1f} fps headroom")
    print(f"  baseline uint8 bilinear → {2_073_600/(t_a_b*1e-3)/1e6:>6.1f} M frag/s "
          f"⇒ {1000.0/t_a_b*args.n_frag/2_073_600:>6.1f} fps headroom")
    print(f"  RVQ nearest             → {2_073_600/(t_b_n*1e-3)/1e6:>6.1f} M frag/s "
          f"⇒ {1000.0/t_b_n*args.n_frag/2_073_600:>6.1f} fps headroom")
    print(f"  RVQ bilinear            → {2_073_600/(t_b_b*1e-3)/1e6:>6.1f} M frag/s "
          f"⇒ {1000.0/t_b_b*args.n_frag/2_073_600:>6.1f} fps headroom")
    print(f"\n[VERDICT]")
    print(f"  RVQ-nearest  vs uint8-nearest:  {t_b_n/t_a_n:.2f}× slower per frag")
    print(f"  RVQ-bilinear vs uint8-bilinear: {t_b_b/t_a_b:.2f}× slower per frag")


if __name__ == "__main__":
    main()
