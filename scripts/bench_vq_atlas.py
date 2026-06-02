"""
Render-PSNR comparison: baked atlas (uint8) vs VQ-reconstructed atlas.

Reconstructs the atlas from `<bake_dir>/vq/{codebooks,indices,block_meta}.pt`,
quantises it back to uint8 with the original scale/offset, then runs
`scripts/benchmark_baked.py --skip_bake` twice (once with the original
atlas, once with the VQ-reconstructed one) and prints a side-by-side
PSNR/SSIM/LPIPS/FPS comparison on the test set.

Usage:
  python scripts/bench_vq_atlas.py \\
      --bake_dir outputs/.../baked_atlas \\
      --model_path outputs/<scene>/<config>          # passed through
      [...other benchmark_baked.py flags...]
"""
import argparse, json, os, shutil, subprocess, sys, time
import torch


def reconstruct_vq_atlas(bake_dir, vq_subdir="vq"):
    """Build a uint8 atlas tensor from saved VQ artifacts.

    Returns (atlas_u8 [H, W, 4 or 3] uint8, atlas_scale, atlas_offset).
    Texels NOT covered by any used block are copied from the original
    atlas (these were already padding bytes; the VQ pipeline only ever
    touched used blocks).
    """
    vq_dir = os.path.join(bake_dir, vq_subdir)
    codebooks = torch.load(os.path.join(vq_dir, "codebooks.pt"),
                           map_location='cuda', weights_only=False).float()      # [L, K, 48]
    indices = torch.load(os.path.join(vq_dir, "indices.pt"),
                         map_location='cuda', weights_only=False).long()         # [L, N]
    meta = torch.load(os.path.join(vq_dir, "block_meta.pt"),
                      map_location='cuda', weights_only=False)
    L = codebooks.shape[0]
    n_used = int(meta["n_used_blocks"])
    B = int(meta["block"])
    H, W = meta["atlas_HW"]
    a_scale = float(meta["atlas_scale"])
    a_off = float(meta["atlas_offset"])
    D = codebooks.shape[2]

    # Re-derive used_idx from atlas_rects.pt (deterministic — only depends
    # on the rect packing, which is fixed by the bake).
    if "used_idx" in meta:                                                       # legacy artifacts
        used_idx = meta["used_idx"].to('cuda')
    else:
        rects = torch.load(os.path.join(bake_dir, "atlas_rects.pt"),
                           map_location='cuda', weights_only=False)
        u0 = rects[:, 0].long(); v0 = rects[:, 1].long()
        w_ = rects[:, 2].long(); h_ = rects[:, 3].long()
        used = torch.zeros((H // B, W // B), dtype=torch.bool, device='cuda')
        for i in range(rects.shape[0]):
            ww = int(w_[i].item()); hh = int(h_[i].item())
            if ww == 0 or hh == 0: continue
            bu = int(u0[i].item()) // B; bv = int(v0[i].item()) // B
            used[bv:bv + hh // B, bu:bu + ww // B] = True
        used_idx = used.nonzero(as_tuple=False)
        assert used_idx.shape[0] == n_used, \
               f"re-derived n_used={used_idx.shape[0]} != saved {n_used}"

    # Σ_l codebooks[l, indices[l, n]] for every used block n.
    recon = torch.zeros(n_used, D, device='cuda', dtype=torch.float32)
    for l in range(L):
        recon = recon + codebooks[l][indices[l]]
    recon_blocks = recon.reshape(n_used, B, B, 3)                                # [n_used, B, B, 3]

    # Load original atlas to seed unused texels (kept as-is — they're padding).
    atlas_orig = torch.load(os.path.join(bake_dir, "atlas_texture.pt"),
                            map_location='cuda', weights_only=False)
    has_alpha = (atlas_orig.dim() == 3 and atlas_orig.shape[2] == 4)
    if atlas_orig.dtype == torch.uint8:
        atlas_u8 = atlas_orig.clone()
    else:                                                                        # legacy fp16 path
        rgb = atlas_orig[..., :3] if has_alpha else atlas_orig
        a_u8 = ((rgb.float() - a_off) / a_scale * 255.0).clamp(0, 255).round().byte()
        if has_alpha:
            atlas_u8 = torch.cat([a_u8, torch.full_like(a_u8[..., :1], 255)], dim=-1)
        else:
            atlas_u8 = a_u8

    # Quantise the VQ float reconstruction back to uint8.
    recon_u8 = ((recon_blocks - a_off) / a_scale * 255.0).clamp(0, 255).round().byte()
    # Scatter into the atlas at every used (bv, bu) position.
    bv = used_idx[:, 0]; bu = used_idx[:, 1]
    pix_row = bv.unsqueeze(1) * B + torch.arange(B, device='cuda').unsqueeze(0)  # [n_used, B]
    pix_col = bu.unsqueeze(1) * B + torch.arange(B, device='cuda').unsqueeze(0)  # [n_used, B]
    # Use a per-block double-broadcast write.
    rows = pix_row.unsqueeze(2).expand(n_used, B, B)                             # [n_used, B, B]
    cols = pix_col.unsqueeze(1).expand(n_used, B, B)
    if has_alpha:
        atlas_u8[rows, cols, :3] = recon_u8
    else:
        atlas_u8[rows, cols] = recon_u8

    return atlas_u8.cpu(), a_scale, a_off, codebooks, indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bake_dir", required=True,
                   help="Path to <bake_dir> with vq/ subdir and atlas_texture.pt.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--vq_subdir", default="vq",
                   help="Subdir of bake_dir holding vq artifacts.")
    p.add_argument("--skip_orig", action="store_true",
                   help="Skip the baked-atlas baseline render (useful if you already have it).")
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--num_benchmark", type=int, default=50)
    p.add_argument("--extra_args", nargs=argparse.REMAINDER,
                   help="Additional flags passed through to benchmark_baked.py "
                        "(must come last on the command line).")
    args = p.parse_args()

    # ---- 1. Reconstruct VQ atlas in memory ---------------------------------
    print(f"[VQ-BENCH] Reconstructing VQ atlas from {args.bake_dir}/{args.vq_subdir}/ …")
    atlas_u8_vq, a_scale, a_off, codebooks, indices = reconstruct_vq_atlas(
        args.bake_dir, args.vq_subdir)
    L = codebooks.shape[0]; K = codebooks.shape[1]
    n_used = indices.shape[1]
    print(f"  L={L} stages, K={K} codewords; "
          f"recon atlas shape {tuple(atlas_u8_vq.shape)} dtype {atlas_u8_vq.dtype}")

    # ---- 2. Stash original, drop VQ in place -------------------------------
    orig_path = os.path.join(args.bake_dir, "atlas_texture.pt")
    backup_path = orig_path + ".orig"
    if not os.path.exists(backup_path):
        shutil.copy(orig_path, backup_path)
        print(f"  backed-up original → {backup_path}")
    else:
        print(f"  backup already exists at {backup_path}")

    # benchmark_baked.py prefers atlas_texture.bc7 over atlas_texture.pt
    # when bake_meta.json mentions `atlas_bc7_file`. Strip those fields so
    # the renderer falls back to the (swappable) atlas_texture.pt path.
    meta_path = os.path.join(args.bake_dir, "bake_meta.json")
    meta_backup = meta_path + ".orig"
    if not os.path.exists(meta_backup):
        shutil.copy(meta_path, meta_backup)
    with open(meta_path) as f:
        meta = json.load(f)
    meta_stripped = {k: v for k, v in meta.items()
                     if not (isinstance(k, str) and k.startswith("atlas_bc7"))}
    with open(meta_path, "w") as f:
        json.dump(meta_stripped, f, indent=2)
    print(f"  stripped atlas_bc7_* keys from bake_meta.json "
          f"(restored from {meta_backup} on cleanup)")

    # ---- 3. Run benchmark_baked.py for each atlas --------------------------
    def run_bench(label, atlas_tensor_cpu, renders_subdir):
        # Write the atlas in place.
        torch.save(atlas_tensor_cpu, orig_path)
        cmd = [sys.executable, "scripts/benchmark_baked.py",
               "--model_path", args.model_path,
               "--output_dir", args.bake_dir,
               "--skip_bake",
               "--num_warmup", str(args.num_warmup),
               "--num_benchmark", str(args.num_benchmark)]
        if args.extra_args:
            cmd += [a for a in args.extra_args if a not in ("--",)]
        print(f"\n=== Running benchmark with {label} atlas ===")
        print("    " + " ".join(cmd))
        t0 = time.time()
        subprocess.run(cmd, check=True)
        # Move the just-written renders to a per-pass subdir under vq/ so
        # the next run doesn't overwrite them. benchmark_baked.py writes
        # into <output_dir>/renders/{sh_only,sh_atlas}/.
        renders_dir = os.path.join(args.bake_dir, "renders")
        dest = os.path.join(args.bake_dir, "vq", renders_subdir)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        if os.path.exists(renders_dir):
            shutil.move(renders_dir, dest)
            print(f"  renders moved → {dest}")
        # Also stash benchmark_results.json so the next run doesn't overwrite.
        results_src = os.path.join(args.bake_dir, "benchmark_results.json")
        if os.path.exists(results_src):
            results_dst = os.path.join(dest, "benchmark_results.json")
            shutil.move(results_src, results_dst)
        print(f"  done in {time.time()-t0:.0f}s")

    try:
        orig_u8_cpu = torch.load(backup_path, map_location='cpu',
                                 weights_only=False)
        if not args.skip_orig:
            run_bench("ORIGINAL baked", orig_u8_cpu, "renders_orig")
        run_bench("VQ-reconstructed", atlas_u8_vq, "renders_vq")
    finally:
        # Always restore the original atlas + bake_meta.json.
        shutil.copy(backup_path, orig_path)
        shutil.copy(meta_backup, meta_path)
        print(f"\n[CLEANUP] Restored original atlas at {orig_path}")
        print(f"[CLEANUP] Restored bake_meta.json at {meta_path}")


if __name__ == "__main__":
    main()
