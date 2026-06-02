"""
Per-level K-means with NON-UNIFORM K per level. For each test config (a
4-tuple K_0..K_3) reports:
  - Per-level PSNR(recon_k  vs  unclustered A_k).
  - Joint PSNR(Σ_k recon_k  vs  the deployed single-bake atlas T).
  - Storage breakdown.

Optional `--with_dc` subtracts per-Gauss DC_k before clustering and adds
it back at reconstruction (DC table stored separately, ~0.21 MB/level).

Requires `<bake_dir>/decomposed/bucket_*.pt` from
`bake_decompose_full.py`.
"""
import argparse, json, math, os, glob, time
import torch


def kmeans_chunked(X, K, iters=20, seed=0, weights=None, dist_chunk=1000):
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
    cn2 = (cents * cents).sum(1)
    for s in range(0, N, dist_chunk):
        e = min(s + dist_chunk, N)
        d = (-2.0) * (X[s:e] @ cents.T) + cn2.unsqueeze(0)
        ass[s:e] = d.argmin(1)
        del d
    return cents, ass


def bc7_bytes(n, D): return n * D // 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bake_dir", required=True)
    p.add_argument("--min_bucket_n", type=int, default=100)
    p.add_argument("--max_bucket_n", type=int, default=20000)
    p.add_argument("--cluster_iters", type=int, default=20)
    p.add_argument("--with_dc",  action="store_true",
                   help="Subtract per-Gauss per-level DC before clustering.")
    p.add_argument("--configs",  type=str, nargs="+",
                   default=[
                       "uniform/4",          # K_k = N/4   (baseline = §3 D)
                       "uniform/8",
                       "uniform/16",
                       "geom:4-8-16-32",     # finest-rich
                       "geom:8-16-32-64",
                       "geom:4-16-32-64",    # very finest-rich
                       "geom:16-16-16-16",   # uniform 1/16 (sanity check)
                       "geom:2-8-32-128",    # extreme finest-rich
                       "geom:8-8-8-8",
                   ],
                   help="Either 'uniform/R' (all levels K=N/R) or "
                        "'geom:R0-R1-R2-R3' (K_k = N/Rk).")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.bake_dir, "bake_meta.json")))
    a_scale, a_off = float(meta["atlas_scale"]), float(meta["atlas_offset"])
    atlas_u8 = torch.load(os.path.join(args.bake_dir, "atlas_texture.pt"),
                          map_location='cuda', weights_only=False)
    if atlas_u8.dim() == 3 and atlas_u8.shape[2] == 4:
        atlas_u8 = atlas_u8[..., :3]
    rects = torch.load(os.path.join(args.bake_dir, "atlas_rects.pt"),
                       map_location='cuda', weights_only=False)
    u0, v0 = rects[:, 0].long(), rects[:, 1].long()
    w_, h_ = rects[:, 2].long(), rects[:, 3].long()

    decomp_dir = os.path.join(args.bake_dir, "decomposed")
    if not glob.glob(os.path.join(decomp_dir, "bucket_*.pt")):
        raise SystemExit("No decomposed atlases — run bake_decompose_full.py first.")

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

    single_bc7_mb = sum(bc7_bytes(idx.shape[0], ww*hh*3) for ww, hh, idx in buckets) / 1024 / 1024
    print(f"[REF] single-bake BC7 = {single_bc7_mb:.2f} MB ; DC mode = {args.with_dc}")

    # Parse configs into {label: [R_0, R_1, R_2, R_3]}.
    config_specs = []
    for c in args.configs:
        if c.startswith("uniform/"):
            R = int(c.split("/", 1)[1])
            config_specs.append((c, [R, R, R, R]))
        elif c.startswith("geom:"):
            rs = [int(x) for x in c.split(":", 1)[1].split("-")]
            assert len(rs) == 4
            config_specs.append((c, rs))
        else:
            raise SystemExit(f"unknown config '{c}'")

    # Per-config table.
    print(f"\n{'config':>22}  "
          f"{'L0 PSNR':>9} {'L1 PSNR':>9} {'L2 PSNR':>9} {'L3 PSNR':>9}  "
          f"{'joint vs T':>11}  {'total MB':>10}  {'vs single':>10}")
    for cname, Rs in config_specs:
        # Streaming accumulation across buckets.
        per_lvl_se   = [0.0, 0.0, 0.0, 0.0]
        per_lvl_n    = [0, 0, 0, 0]
        joint_se     = 0.0
        joint_n      = 0
        total_cb     = 0
        total_idx_b  = 0
        total_dc_b   = 0
        t0 = time.time()
        for ww, hh, idx in buckets:
            n_full = idx.shape[0]
            sub_idx = idx
            if n_full > args.max_bucket_n:
                perm = torch.randperm(n_full, device='cuda')[:args.max_bucket_n]
                sub_idx = idx[perm]
            n_sub = sub_idx.shape[0]
            D = hh * ww * 3
            scale = n_full / max(n_sub, 1)

            # Load T (deployed atlas, dequantised).
            T = torch.empty(n_sub, hh, ww, 3, device='cuda', dtype=torch.float32)
            u0i = u0[sub_idx]; v0i = v0[sub_idx]
            for r in range(n_sub):
                T[r] = atlas_u8[v0i[r]:v0i[r]+hh, u0i[r]:u0i[r]+ww].float()
            T = (T / 255.0 * a_scale + a_off).reshape(n_sub, D)

            # Load decomposed bucket A.
            bpath = os.path.join(decomp_dir, f"bucket_{ww}x{hh}.pt")
            blob = torch.load(bpath, map_location='cpu', weights_only=False)
            A_full = blob['A']                                       # FP16 CPU
            gidx = blob['gauss_idx'].to('cuda')
            pos_map = torch.full((rects.shape[0],), -1, dtype=torch.long, device='cuda')
            pos_map[gidx] = torch.arange(gidx.shape[0], device='cuda')
            pos = pos_map[sub_idx].cpu()
            L = A_full.shape[0]

            recon_sum = torch.zeros(n_sub, D, device='cuda')
            for k in range(L):
                A_k = A_full[k][pos].to('cuda').float().reshape(n_sub, D)
                if args.with_dc:
                    dc_k = A_k.reshape(n_sub, hh, ww, 3).mean(dim=(1, 2))
                    A_k_c = A_k - dc_k.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
                    total_dc_b += int(round(n_sub * 3 * scale))
                else:
                    A_k_c = A_k

                K_k = max(4, n_sub // Rs[k])
                K_k = min(K_k, n_sub)
                cents, ass = kmeans_chunked(A_k_c, K_k,
                                            iters=args.cluster_iters,
                                            dist_chunk=1000)
                recon_k = cents[ass]
                if args.with_dc:
                    recon_k = recon_k + dc_k.repeat_interleave(hh * ww, 1).reshape(n_sub, D)
                # Per-level error vs UNCLUSTERED A_k.
                lvl_se = (recon_k - A_k).pow(2).sum().item()
                per_lvl_se[k] += lvl_se * scale
                per_lvl_n[k]  += n_sub * D * scale
                recon_sum = recon_sum + recon_k
                total_cb    += bc7_bytes(K_k, D)
                total_idx_b += int(round(n_sub * scale * math.ceil(math.log2(max(K_k, 2)))))
                del A_k, A_k_c, cents, ass, recon_k

            joint_se += (recon_sum - T).pow(2).sum().item() * scale
            joint_n  += n_sub * D * scale
            del recon_sum, T, A_full
            torch.cuda.empty_cache()

        lvl_psnrs = [-10.0 * math.log10(max(per_lvl_se[k] / max(per_lvl_n[k], 1), 1e-20)) for k in range(4)]
        joint_psnr = -10.0 * math.log10(max(joint_se / max(joint_n, 1), 1e-20))
        cb_mb = total_cb / 1024 / 1024
        idx_mb = total_idx_b / 8 / 1024 / 1024
        dc_mb = total_dc_b / 1024 / 1024
        total_mb = cb_mb + idx_mb + dc_mb
        print(f"   {cname:>20}  "
              f"{lvl_psnrs[0]:>7.2f} dB  {lvl_psnrs[1]:>7.2f} dB  "
              f"{lvl_psnrs[2]:>7.2f} dB  {lvl_psnrs[3]:>7.2f} dB  "
              f"{joint_psnr:>8.2f} dB  {total_mb:>8.2f} MB  "
              f"{total_mb/single_bc7_mb*100:>8.2f}%   ({time.time()-t0:.0f}s)")

    print(f"\n   single-bake BC7 reference: {single_bc7_mb:.2f} MB")


if __name__ == "__main__":
    main()
