"""Fair off-lattice comparison: classic per-surfel RxR rect atlas vs the neural
probe-atlas variants, all on DENSE RANDOM uv (not the 16^2 bake lattice, which
self-favors lattice-aligned representations), same kernel-weighted PSNR.

Classic-R student: teacher snapshotted at RxR texel centers, bilinear at query.
Neural student: field(probe(uv)) from a saved train_ckpt.pt.
"""
import sys, os, math, argparse
sys.path.insert(0, '/home/nilkel/Projects/nest-splatting')
import torch
import scripts.probe_atlas_distill as P

DEV = 'cuda'
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model_path', required=True)
ap.add_argument('--neural_tags', nargs='*', default=['v_neural_8k_t21', 'v8k_t21_noz'])
ap.add_argument('--classic_res', nargs='*', type=int, default=[8, 16, 32, 64])
ap.add_argument('--n_eval', type=int, default=2048)
ap.add_argument('--uv_per', type=int, default=512)
args = ap.parse_args()

run = P.Run(args.model_path)
g = torch.Generator().manual_seed(7)
sids = torch.arange(run.N)[torch.randperm(run.N, generator=g)[:args.n_eval]].to(DEV)
gu = torch.Generator(device=DEV).manual_seed(99)
E = P.UV_EXTENT

def wpsnr(student_fn, chunk=256):
    se_sum, w_sum = 0.0, 0.0
    gu.manual_seed(99)  # same uv for every contender
    for i in range(0, len(sids), chunk):
        sid = sids[i:i + chunk]
        uv = (torch.rand(len(sid), args.uv_per, 2, device=DEV, generator=gu) * 2 - 1) * E
        t = run.teacher_uv(sid, uv)
        s = student_fn(sid, uv)
        w = run.kernel_weight(sid, uv)
        se = ((t - s) ** 2).mean(-1) * w
        se_sum += se.sum().item(); w_sum += w.sum().item()
    return 10.0 * math.log10(1.0 / max(se_sum / max(w_sum, 1e-8), 1e-12))

print(f'Eval: {args.n_eval} surfels x {args.uv_per} random uv, kernel-weighted, off-lattice')

# --- classic per-surfel RxR atlases (teacher texel-center snapshot + bilinear) ---
for R in args.classic_res:
    tc = ((torch.arange(R, device=DEV) + 0.5) / R) * 2 * E - E
    gy, gx = torch.meshgrid(tc, tc, indexing='ij')
    lat = torch.stack([gx, gy], -1).view(1, -1, 2)                    # [1,R*R,2] (u=x fast)
    def classic(sid, uv, R=R, lat=lat):
        tex = run.teacher_uv(sid, lat.expand(len(sid), -1, -1))       # [B,R*R,3]
        tex = tex.view(len(sid), R, R, 3).permute(0, 3, 1, 2)         # [B,3,R,R] (rows=v)
        gridc = (uv / E).flip(-1) if False else (uv / E)              # grid_sample: x=u,y=v
        s = torch.nn.functional.grid_sample(tex, gridc.view(len(sid), -1, 1, 2),
                mode='bilinear', align_corners=False, padding_mode='border')
        return s.view(len(sid), 3, -1).permute(0, 2, 1)
    texels = run.N * R * R
    print(f'classic {R:2d}x{R:2d}: {wpsnr(classic):6.2f} dB   '
          f'({texels/1e6:7.1f}M texels = {texels*3/1e6:7.0f} MB u8)')

# --- neural variants from checkpoints ---
for tag in args.neural_tags:
    ck_path = os.path.join(args.model_path, 'probe_atlas', tag, 'train_ckpt.pt')
    if not os.path.exists(ck_path):
        print(f'{tag}: (no ckpt)'); continue
    ck = torch.load(ck_path, map_location=DEV)
    img_res = ck['img_res']
    field = P.NeuralTexture2D(img_res=img_res, levels=ck['levels'], feat=ck['feat'],
                              log2_table=ck['hash2d_log2'], base=ck['tex_base']).to(DEV)
    field.load_state_dict(ck['field']); field.eval()
    centers, cell = P.grid_probe_centers(run.N, img_res)
    if ck.get('head') is not None:
        gd = ck['head'].get('anchor').shape[1] > 32
        zd = ck['head']['z'].shape[1] if 'z' in ck['head'] else 0
        head = P.ProbeHead(run, geom_inputs=gd, latent_dim=zd,
                           free_residual=('resid' in ck['head'])).to(DEV)
        head.load_state_dict(ck['head'])
        with torch.no_grad(): probes_all = head.probes().detach()
    else:
        probes_all = ck['probes'].to(DEV)
    if ck.get('student') == 'image':
        px = ck.get('pixels')
        px = px.to(DEV).float() if px is not None else None
        sampler = P.image_bilinear_sampler(field, img_res, ck['levels'], px)
    else:
        sampler = P.field_sampler(field, ck['levels'])
    def neural(sid, uv, probes_all=probes_all, img_res=img_res, cell=cell, sampler=sampler):
        gcoord = P.probe_grid_coords(probes_all[sid], centers[sid], uv, img_res, cell)
        with torch.no_grad():
            return sampler(gcoord)
    nb = sum(p.numel() for p in field.parameters()) * 4
    print(f'{tag:18s}: {wpsnr(neural):6.2f} dB   (field {nb/1e6:.0f} MB fp32 / {nb/2e6:.0f} MB fp16)')
