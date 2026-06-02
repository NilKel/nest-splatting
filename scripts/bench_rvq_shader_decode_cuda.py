"""
Benchmark per-fragment RVQ atlas decode against the BC7 baseline.

JIT-compiles a small CUDA kernel (scripts/_rvq_decode_cuda.cu) that mirrors
exactly what a WGSL fragment shader would do — one block-id computation,
L codebook lookups, sum. This is the apples-to-apples version of
bench_rvq_shader_decode.py (which used PyTorch indexing with its
own kernel-launch + dtype-coercion overhead).

Two configurations measured:
  - nearest sample  (1 block lookup / fragment, L codebook reads)
  - bilinear sample (4 block lookups / fragment, 4·L codebook reads)

Comparison: an equivalent CUDA bilinear kernel sampling a pre-dequantized
FP16 atlas (proxy for the hardware BC7 texture-sample path — the latter
benefits from the texture cache, so this gives an *upper bound* on its
cost).
"""
import argparse, json, math, os, time
import torch
from torch.utils.cpp_extension import load


def jit_compile():
    cu_path = os.path.join(os.path.dirname(__file__), "_rvq_decode_cuda.cu")
    return load(
        name="rvq_decode_cuda",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--n_frag", type=int, default=2_073_600,                # 1920×1080
                   help="Fragments per trial. Default = one Full-HD frame.")
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_trial", type=int, default=50)
    args = p.parse_args()
    device = 'cuda'

    print("[JIT] compiling _rvq_decode_cuda.cu …")
    t0 = time.time()
    M = jit_compile()
    print(f"  compiled in {time.time()-t0:.1f}s")

    # ------------ Load RVQ artifacts ---------------------------------------
    print(f"[LOAD] {args.bake_dir}")
    codebooks = torch.load(os.path.join(args.bake_dir, "vq/codebooks.pt"),
                           map_location=device, weights_only=False).to(torch.float16).contiguous()
    indices = torch.load(os.path.join(args.bake_dir, "vq/indices.pt"),
                         map_location=device, weights_only=False).to(torch.uint8).contiguous()
    block_meta = torch.load(os.path.join(args.bake_dir, "vq/block_meta.pt"),
                            map_location=device, weights_only=False)
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location=device, weights_only=False).to(torch.int32).contiguous()
    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale = float(meta["atlas_scale"]); a_off = float(meta["atlas_offset"])
    H = int(block_meta["atlas_HW"][0]); W = int(block_meta["atlas_HW"][1])
    B = int(block_meta["block"])
    L, K, D = codebooks.shape
    M_surf = rects.shape[0]
    N_used = indices.shape[1]
    print(f"  L={L} K={K} D={D}, atlas {H}x{W}, {M_surf:,} surfels, {N_used:,} used blocks")

    # ------------ Build surfel offsets + reorder indices --------------------
    bw = (rects[:, 2] // B).long()
    bh = (rects[:, 3] // B).long()
    surfel_offsets = torch.zeros(M_surf + 1, dtype=torch.int64, device=device)
    surfel_offsets[1:] = torch.cumsum(bw * bh, dim=0)
    assert int(surfel_offsets[-1].item()) == N_used

    # Row-major → surfel-major remap (one-time at load).
    print("[REMAP] building row-major→surfel-major perm …")
    t0 = time.time()
    used_mask = torch.zeros((H // B, W // B), dtype=torch.bool, device=device)
    for i in range(M_surf):
        ww = int(rects[i, 2].item()); hh = int(rects[i, 3].item())
        if ww == 0 or hh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        used_mask[bv0:bv0 + hh//B, bu0:bu0 + ww//B] = True
    used_flat = used_mask.reshape(-1)
    rm_id_flat = torch.cumsum(used_flat.long(), dim=0) - 1
    rm_id_flat[~used_flat] = -1
    rm_id = rm_id_flat.reshape(H // B, W // B)
    perm = torch.empty(N_used, dtype=torch.int64, device=device)
    for i in range(M_surf):
        nw = int(bw[i].item()); nh = int(bh[i].item())
        if nw == 0 or nh == 0: continue
        bu0 = int(rects[i, 0].item()) // B
        bv0 = int(rects[i, 1].item()) // B
        off = int(surfel_offsets[i].item())
        sub = rm_id[bv0:bv0+nh, bu0:bu0+nw].reshape(-1)
        perm[off:off+nw*nh] = sub
    indices_sm = indices[:, perm].contiguous()
    print(f"  done in {time.time()-t0:.2f}s")
    del used_mask, used_flat, rm_id_flat, rm_id, perm; torch.cuda.empty_cache()

    # ------------ Baseline reference atlas (FP16) ---------------------------
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location=device, weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    atlas_f16 = ((atlas_u8.float() / 255.0) * a_scale + a_off).half().contiguous()
    print(f"[LOAD] baseline FP16 atlas in GPU mem: {atlas_f16.numel()*2/1024/1024:.1f} MB")
    del atlas_u8; torch.cuda.empty_cache()

    # ------------ Synthetic fragments ---------------------------------------
    # Weight by surfel area so most fragments hit the dominant rect sizes.
    g = torch.Generator(device=device).manual_seed(0)
    weights = (rects[:, 2].float() * rects[:, 3].float())
    sample_surfels = torch.multinomial(weights, args.n_frag, replacement=True, generator=g).to(torch.int64)
    rand_u = torch.rand(args.n_frag, generator=g, device=device)
    rand_v = torch.rand(args.n_frag, generator=g, device=device)
    rs = rects[sample_surfels]
    atlas_uvs = torch.stack([
        rs[:, 0].float() + rand_u * (rs[:, 2].float() - 1.0),
        rs[:, 1].float() + rand_v * (rs[:, 3].float() - 1.0),
    ], dim=1).contiguous()
    print(f"[FRAG] {args.n_frag:,} synthetic fragments")

    # ------------ Sanity check ---------------------------------------------
    with torch.no_grad():
        a_base = M.baseline_bilinear(atlas_f16, atlas_uvs)
        a_rvq_b = M.rvq_decode_bilinear(codebooks, indices_sm, surfel_offsets,
                                        rects, sample_surfels, atlas_uvs, B)
        mse = (a_rvq_b.float() - a_base.float()).pow(2).mean().item()
        psnr = -10.0 * math.log10(max(mse / (a_scale ** 2), 1e-20))
    print(f"\n[SANITY] RVQ-bilinear vs baseline-bilinear PSNR: {psnr:.2f} dB "
          f"(expect ≈ RVQ atlas-fidelity ~42 dB)\n")

    # ------------ Benchmark loop -------------------------------------------
    def bench(fn, label):
        for _ in range(args.n_warmup):
            fn(); torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_trial)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_trial)]
        for t in range(args.n_trial):
            starts[t].record(); fn(); ends[t].record()
        torch.cuda.synchronize()
        times = sorted(starts[i].elapsed_time(ends[i]) for i in range(args.n_trial))
        med_ms = times[len(times)//2]
        fps = 1000.0 / med_ms
        print(f"  {label:<32}  median {med_ms:>7.3f} ms / "
              f"{args.n_frag:,} frags  →  {fps:>6.1f} fps")
        return med_ms

    print(f"[BENCH] {args.n_frag:,} frag/frame × {args.n_trial} trials (warmup {args.n_warmup})\n")
    t_base = bench(lambda: M.baseline_bilinear(atlas_f16, atlas_uvs),
                   "baseline FP16 atlas bilinear")
    t_rvq_n = bench(lambda: M.rvq_decode_nearest(codebooks, indices_sm,
                                                  surfel_offsets, rects,
                                                  sample_surfels, atlas_uvs, B),
                     "RVQ nearest (L=4 lookups)")
    t_rvq_b = bench(lambda: M.rvq_decode_bilinear(codebooks, indices_sm,
                                                   surfel_offsets, rects,
                                                   sample_surfels, atlas_uvs, B),
                     "RVQ bilinear (4·L=16 lookups)")

    print(f"\n[VERDICT]")
    print(f"  RVQ nearest  vs baseline bilinear : {t_rvq_n / t_base:>5.2f}× slower")
    print(f"  RVQ bilinear vs baseline bilinear : {t_rvq_b / t_base:>5.2f}× slower")
    print(f"\n  Note: baseline uses pre-dequantized FP16 atlas (1.35 GB) — this is")
    print(f"  the *upper bound* on real BC7-tex2D cost. Real hardware BC7 path")
    print(f"  benefits from the texture cache, so the real BC7 baseline is faster.")
    print(f"\n  GPU memory at decode time:")
    print(f"    BC7 atlas             : 225 MB (deployed)")
    print(f"    RVQ codebooks+indices : {(codebooks.numel()*2 + indices_sm.numel() + surfel_offsets.numel()*8 + rects.numel()*4)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
