"""Value-level verification of a probe_uv_field bake.

By construction, the atlas fetched at each surfel's probe CENTER must equal the
teacher residual at that surfel's 3D center:
    T[phi(c)] = teacher(phi_inv(phi(c))) ~= teacher(c)   (up to 3d-cycle error)
If these match (high cosine, |err| << content std), the residual VALUES in the
atlas are provably correct — signed, unbounded, correctly scaled — and any
render problem is placement (phi quality / probe congestion), not values.

  conda run -n nest_splatting python scripts/verify_uv_bake.py -m <ckpt_dir>
"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.probe_uv_field import Ckpt

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model_path', required=True)
ap.add_argument('--out_tag', type=str, default='uv_field')
a = ap.parse_args()

ck = Ckpt(a.model_path)
d = os.path.join(a.model_path, a.out_tag)
T = torch.load(os.path.join(d, 'tex_init.pt'), map_location='cuda')
pr = torch.load(os.path.join(d, 'probes.pt'), map_location='cuda')
pr = pr['probes'] if isinstance(pr, dict) else pr
R = T.shape[0]

with torch.no_grad():
    teach = ck.residual(ck.center)
    x = (pr[:, 4] - 0.5).clamp(0, R - 2)
    y = (pr[:, 5] - 0.5).clamp(0, R - 2)
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = (x - x0.float()).unsqueeze(-1), (y - y0.float()).unsqueeze(-1)
    atlas = ((1 - fx) * (1 - fy) * T[y0, x0] + fx * (1 - fy) * T[y0, x0 + 1]
             + (1 - fx) * fy * T[y0 + 1, x0] + fx * fy * T[y0 + 1, x0 + 1])
    err = (atlas - teach).norm(dim=-1)
    print('teacher@centers: mean=%+.3f std=%.3f range[%+.2f,%+.2f]'
          % (teach.mean(), teach.std(), teach.min(), teach.max()))
    print('atlas@probes:    mean=%+.3f std=%.3f range[%+.2f,%+.2f]'
          % (atlas.mean(), atlas.std(), atlas.min(), atlas.max()))
    print('|err|: p50=%.3f p90=%.3f  (teacher content std %.3f)'
          % (err.quantile(0.5), err.quantile(0.9), teach.std()))
    print('cosine(atlas, teacher) = %.4f'
          % torch.nn.functional.cosine_similarity(atlas.flatten(), teach.flatten(), dim=0))
    print('\nPASS criteria: cosine > 0.9 and |err| p50 well under the content std.')
    print('Match  -> values are correct; render issues are PLACEMENT (phi quality).')
    print('Mismatch -> genuine value bug in the bake chain.')
