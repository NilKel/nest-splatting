#!/usr/bin/env python
"""Fragment-level probe: measure inner-loop iterations (= shared-memory reads of a
staged Gauss) per FRAME, so the shared-bandwidth arithmetic rests on measurement
rather than on an assumed amplification factor.

Pins the instrumented t8 clone in as 'diff_surfel_bake_render_lean' so
benchmark_baked.py's normal CONIC aliasing picks it up, then wraps the rasterizer
call to count invocations -> per-frame = total / calls.
"""
import os, sys, runpy

REPO = "/home/nilkel/Projects/nest-splatting"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import diff_surfel_bake_render_lean_t8 as t8
# benchmark_baked.py does __import__('diff_surfel_bake_render_lean'); seed the cache
# so it resolves to the instrumented clone without touching the committed script.
sys.modules['diff_surfel_bake_render_lean'] = t8

_C = t8._C
_calls = {'n': 0}
_orig = _C.rasterize_gaussians


def _counted(*a, **k):
    _calls['n'] += 1
    return _orig(*a, **k)


_C.rasterize_gaussians = _counted

_C.reset_sat_counters()
print("[FRAG] counters zeroed", flush=True)

import atexit


@atexit.register
def _report():
    try:
        vals = _C.read_sat_counters()
    except Exception as e:
        print(f"[FRAG] read failed: {e}")
        return
    run, need, evals, blended = vals[0], vals[1], vals[2], vals[3]
    w_iters = vals[4] if len(vals) > 4 else 0
    w_live  = vals[5] if len(vals) > 5 else 0
    w_skip  = vals[6] if len(vals) > 6 else 0
    n = max(1, _calls['n'])
    try:
        inst = _C.get_last_num_rendered()
    except Exception:
        inst = -1
    print(f"[FRAG] render_calls={n}")
    print(f"[FRAG] TOTAL   rounds_run={run:,} needed={need:,} "
          f"evals={evals:,} blended={blended:,}")
    print(f"[FRAG] PERFRAME rounds_run={run/n:,.0f} evals={evals/n:,.0f} "
          f"blended={blended/n:,.0f}  (last-frame instances={inst:,})")
    if run:
        print(f"[FRAG] amplification evals/rounds_run = {evals/run:.2f} "
              f"(Gauss evaluated per thread per staging round)")
    if evals:
        print(f"[FRAG] blend survival = {100.0*blended/evals:.1f}%  "
              f"(culled {100.0*(evals-blended)/evals:.1f}%)")
    if w_iters:
        ideal = w_iters - w_live
        print(f"[STRIP] warp_iters={w_iters:,} live={w_live:,} skipped={w_skip:,}")
        print(f"[STRIP] realized skip = {100.0*w_skip/w_iters:.1f}%  "
              f"ideal ceiling = {100.0*ideal/w_iters:.1f}%  "
              f"mask efficiency = {100.0*w_skip/max(1,ideal):.1f}%")


sys.argv = ["benchmark_baked.py"] + sys.argv[1:]
runpy.run_path(os.path.join(REPO, "scripts", "benchmark_baked.py"),
               run_name="__main__")
