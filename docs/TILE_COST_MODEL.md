# What actually costs time in the baked tile rasterizer

Written 2026-08-10 after the treehill blur-split investigation. The short
version: **frame time tracks (primitive × tile) pairs, not primitives and not
overdraw.** Both of the metrics we habitually quote are poor predictors, and
one of them (overdraw) can move the *wrong way* while time gets worse.

---

## 1. The three candidate cost models

| model | what it counts | correlation with FPS advantage (9-scene study) |
|---|---|---|
| primitive count | N Gaussians | r = **+0.72** |
| surfel bloat | fraction with axis > 0.2 | r = **−0.63** |
| overdraw | contributors per pixel | r = +0.22 |

Overdraw is the metric everyone reaches for and the weakest of the three.
See [MIP360_BOTTLENECK_ANALYSIS.md](../speed_comparison/MIP360_BOTTLENECK_ANALYSIS.md).

Neither of the first two is the real quantity though — they are proxies that
happen to correlate. The controlled comparisons below separate them.

## 2. Two controlled comparisons that isolate each proxy

**bonsai vs room (3D_SH_res, CONIC)** — near-identical resolution
(1559×1039 vs 1557×1038) and near-identical surfel size distributions
(mean axis 0.069 vs 0.055), so footprint is controlled and only count varies:

| | bonsai | room | ratio |
|---|---:|---:|---:|
| Gaussians | 92,489 | 63,164 | 1.464× |
| frame time | 0.793 ms | 0.558 ms | 1.421× |
| overdraw | 17.26 | 15.37 | 1.125× |
| **cost per Gaussian** | 7.73 ns | 7.87 ns | **0.971×** |

Frame time tracks the count ratio to within 3%; the fragment ratio (1.125)
mispredicts by 26%. bonsai is if anything marginally *more* efficient per
primitive. There is no anomaly here — bonsai just carries 46% more Gauss.

**garden vs the other outdoor scenes** — counts vary but footprints vary far
more, and now count stops predicting: garden carries 55% MORE Gauss than
stump yet renders 39% FASTER.

| scene | Mpx | mean axis | p95 axis | >0.5 | ns/Gauss |
|---|---:|---:|---:|---:|---:|
| garden | 1.089 | 0.057 | 0.157 | 0.61% | **3.46** |
| flowers | 1.040 | 0.142 | 0.410 | 3.80% | 4.19 |
| bicycle | 1.017 | 0.096 | 0.294 | 1.74% | 4.49 |
| treehill | 1.054 | 0.134 | 0.406 | 3.70% | 5.16 |
| stump | 1.027 | 0.211 | 0.877 | 8.58% | **7.44** |

corr(ns/Gauss, p95 axis) = **+0.97**; mean axis +0.91; frac>0.5 +0.95.

Garden has *indoor-scene* surfel statistics (mean 0.057 ≈ room's 0.055) in an
outdoor scene, and renders the most pixels of the group — so resolution works
against it and it still wins. Garden is the existence proof that compact
outdoor surfels are reachable.

## 3. Screen footprint of the worst splats (measured)

3σ compact support (beta_scaled has hard-zero alpha beyond ρ=3, so this is a
bound not an estimate), AABB clipped to screen, maximised over all training
cameras:

| scene | screen px | p50 | p95 | p99.9 | max | >1% screen |
|---|---:|---:|---:|---:|---:|---:|
| garden | 1,089,480 | 456 | 3,928 | 46,442 | **246 K** | 1.28% |
| bicycle | 1,016,814 | 667 | 5,610 | 97,951 | 970 K | 2.30% |
| flowers | 1,039,968 | 757 | 8,871 | 126,530 | 1.04 M | 4.09% |
| treehill | 1,054,144 | 661 | 9,798 | 97,221 | 1.05 M | 4.58% |
| stump | 1,027,125 | 1,238 | 10,700 | 170,687 | 1.03 M | 5.26% |

The largest splats cover **the entire screen** on four of five outdoor scenes.
Garden's worst covers 23% of it. Median is 456–1,238 px (a ~21×21 to ~35×35
splat) — so the distribution is heavy-tailed by a factor of 100–370× in area,
which is exactly why a **mean** scale regularizer cannot touch it and why the
p95 statistic correlates better than the mean.

**Calibration note for `--blur_thresh`:** the default 5000 flags anything
dominating more than `H·W/5000` ≈ **205 px** — *below the median splat* on
every outdoor scene. To target genuine bloat the value wants to be ~10–100
(flagging >10 K–100 K px), not 5000.

## 4. The real cost model: (primitive × tile) pairs

Every pixel in a tile evaluates **every** Gaussian binned to that tile,
whether or not it contributes. So:

```
evaluations = 256 threads × Σ_tiles(list_length) = 256 × instances
```

`instances` = (Gaussian, tile) pairs = what `duplicateWithKeys` emits, what
the radix sort orders, and what `identifyTileRanges` walks.

This is NOT `contributors` (what overdraw measures) and NOT `N`.

### treehill blur-split A/B — the case that breaks both proxies

`RD_SV_30thr_005w25gLP_N2f_frz5k10` (no blur_split) vs
`2D_SV_30thr_005w25gLP_N2F_Jac_BS3k` (`--blur_split --blur_thresh 3000`):

| | RD | BS3k | ratio |
|---|---:|---:|---:|
| primitives | 175,459 | 442,643 | 2.52× |
| **overdraw mean** | 27.72 | **23.69** | **0.855×** |
| overdraw p95 | 53.8 | 48.9 | |
| mean surfel axis | 0.1339 | 0.0978 | 0.73× |
| p99 axis | 1.350 | 0.738 | 0.55× |
| footprint p99.9 | 97,221 px | 26,409 px | 0.27× |
| footprint >1% screen | 4.58% | 0.70% | 0.15× |
| visible Gauss/view | 45,225 | 131,968 | 2.92× |
| tiles per visible Gauss | 11.29 | 5.74 | 0.51× |
| **instances/view** | 510,807 | **757,094** | **1.48×** |
| evaluations (256×inst) | 130.8 M | 193.8 M | 1.48× |
| contributions (overdraw×px) | 29.2 M | 25.0 M | 0.855× |
| **useful fraction** | **22.3%** | **12.9%** | **0.58×** |
| prod FPS | 651 | 354 | 0.54× |
| **CONIC FPS** | **995** | **728** | **0.73×** |

blur_split did exactly what it was asked to: **bloat is genuinely fixed** —
p99 axis halved, p99.9 footprint down 73%, screen-hogging splats down 6.5×,
and overdraw *dropped* 15%. And the result is **1.84× slower**.

Because: each splat got 49% smaller in tile terms, but there are 2.52× as many,
so **instances still rose 48%** — and each instance costs a full 256-thread
tile walk. Efficiency fell 1.73× (22.3% → 12.9% of evaluated pairs producing a
contribution).

### CONIC confirms the model

Re-benched both through `diff_surfel_bake_render_lean` `LEAN_FLAGS=CONIC`
(50 warmup / 400 timed, idle GPU):

| | prod FPS | CONIC FPS | CONIC gain | frame ms | baked PSNR |
|---|---:|---:|---:|---:|---:|
| RD | 651 | **995** | +52.9% | 1.005 | 22.29 |
| BS3k | 354 | **728** | +105.6% | 1.374 | 21.95 |

**BS3k gains twice as much from CONIC (+106% vs +53%)** — exactly what the
model predicts. CONIC makes each *evaluation* cheaper (precomputed rational
reconstruction replaces the per-fragment T-matrix cross-product ray-splat).
BS3k performs 1.48× more evaluations, so a per-evaluation saving is worth
proportionally more to it.

And the ratio converges on the instance ratio:

| ratio BS3k/RD | value |
|---|---:|
| primitives | 2.525× |
| **instances (est)** | **1.482×** |
| frame time, prod | 1.839× |
| **frame time, CONIC** | **1.367×** |

Under prod, expensive per-fragment work *amplified* the evaluation-count gap
(1.84 > 1.48). Under CONIC the per-fragment term shrinks and the measured
ratio lands within 8% of the pure instance ratio. That is the cost model
falsifiable-and-confirmed: **once per-fragment cost is minimised, time tracks
(primitive × tile) pairs almost exactly.**

Note this also means CONIC disproportionately rescues over-split scenes — but
BS3k is still 1.37× slower AND 0.34 dB worse than RD. CONIC narrows the
penalty; it does not make blur_split free.

### Why sub-tile splats are pathological

A tile is 16×16 = **256 pixels**. A splat whose real footprint is ~400 px is
binned into ~4 tiles (31% of BS3k's splats touch 3–4), so **1,024 threads
evaluate it to shade ~400** — ≥60% waste by construction, monotonically worse
as splats shrink toward and below tile size.

Splitting a large splat into four small ones does **not** quarter the work.
Each child pays a full tile tax wherever it lands, and straddling means it
usually lands in more than one. BS3k's tile-count histogram shows 58% of
visible splats touching ≤4 tiles — near the quantisation floor, where further
splitting is pure cost.

Per-Gaussian *preprocess* is not the driver, contrary to a first guess:
~290 B/Gaussian × 441 K = 128 MB/frame ≈ **0.07 ms** at 1.8 TB/s, against a
2.83 ms frame. ~2%.

## 5. Consequences

- **Stop quoting overdraw as the perf metric.** It can improve while time
  regresses, as BS3k demonstrates.
- **`--blur_split` at default thresholds is a perf regression**, even though it
  fixes the geometry problem it was built for. It also cost 0.45 dB
  (21.93 vs 22.38 neural).
- **The wanted operating point is garden's**: few, compact, ~tile-sized
  splats. Both "few large" (RD) and "many small" (BS3k) lose to it.

## 6. Open experiments

1. **Real instance counts.** All `instances` figures above are AABB-derived
   upper bounds; the binner uses AccuTile (ellipse-tight), so true values are
   lower. `Rasterizer::forward` already returns `num_rendered` but
   `rasterize_points.cu` discards it — exposing it is a small binding change
   and would firm up every ratio here.
2. **8×8 tiles.** Cuts the per-splat tax 4× for sub-tile splats at the cost of
   more instances and more sort work. Given how small our splats have become
   this is the single most promising renderer-side change. Must be done in a
   lean clone (CUDA isolation rule).
3. **Size-floor in the split path.** Refuse to split when the projected
   footprint is already < ~2 tiles. Would have prevented most of BS3k's
   regression at zero quality cost.
4. **Size-gated blur_split.** `split_mask | (esm & split_qualifiers)` — keeps
   blur_split's gradient-independence (its one unique property: it is the only
   mechanism in the pipeline not gated on photometric error, so it is the only
   one that can reach bloated *well-reconstructed* background surfels) while
   restricting it to genuinely world-large primitives.
5. **Tail-targeted scale reg.** The existing `--scale_reg` is
   `w·|get_scaling|.mean()` — a mean, dominated by the small bulk, so it cannot
   reach the tail without uniformly shrinking everything.
   `w·relu(max_axis − τ)²` with τ ≈ `dense·extent` is the right shape.

---

## 7. Experiment: 8×8 tiles — decisively WORSE (−54%)

Clone `diff_surfel_bake_render_lean_t8` (`BLOCK_X/Y = 8`, `LEAN_FLAGS=CONIC`),
50/400 frames, idle GPU, PSNR bit-identical (22.29 / 21.95 unchanged):

| | 16×16 | 8×8 | Δ |
|---|---:|---:|---:|
| RD | 995 | **459** | **−53.9%** |
| BS3k | 728 | **332** | **−54.4%** |

Both lose ~54%, so this is structural, not scene-specific.

**Why, and it inverts §4's conclusion.** `BLOCK_SIZE = BLOCK_X·BLOCK_Y`, so 8×8
also drops the block from **256 threads to 64**. That costs on two axes at once:

- **Amortization**: the cooperative staging loads one Gaussian per thread per
  batch, then every thread evaluates all of them. At 256 threads that's 256
  global loads serving 256×256 = 65,536 evaluations. At 64 threads it is 64
  loads serving 64×64 = 4,096 — **4× fewer evaluations per load**.
- **Occupancy**: 64 threads is 2 warps per block; far less latency hiding, and
  4× more blocks with their attendant launch and tile-range overhead.

The 4× reduction in per-splat tile tax is real but is swamped by both.

**So the "waste" in §4 is not a defect — it is the price of the amortization
that makes this renderer fast.** The 12.9–22.3% useful-fraction figure reads
like inefficiency, but buying it back by shrinking tiles costs more than it
saves. Tile size is already at/near its optimum for this kernel.

**Consequence for where to spend effort:** there is no renderer-side tiling
lever here. The only way to move FPS is to change *what is drawn* — keep
primitive counts down and footprints near tile-sized, i.e. reach garden's
regime (§2). That makes the training-side items in §6 the whole remaining
opportunity, not a secondary one.

Untested opposite direction: 32×32 would be 1024 threads/block — exactly the
CUDA maximum — and would quadruple the shared-memory staging footprint, very
likely blowing the 48 KB static cap given the CONIC path already stages ~60 B
per entry. Not obviously viable, and the amortization curve is already
flattening at 256.

## 8. Real instance counts (measured, replaces the AABB estimates)

`Rasterizer::forward` already computes `num_rendered` and discarded it. Exposed
via a host global + `get_last_num_rendered()` binding (no signature change, no
device code) in the scratch clone; shipping `diff_surfel_bake_render_lean` is
untouched.

| | RD | BS3k | ratio |
|---|---:|---:|---:|
| **real instances/frame** | **2,290,384** | **3,373,473** | **1.473×** |
| §4 AABB estimate | 510,807 | 757,094 | 1.482× |
| frame-time ratio (CONIC) | — | — | 1.367× |
| **cost per instance** | **0.452 ns** | **0.425 ns** | 0.94× |

Two things to take from this:

**The ratio held; the absolutes did not.** The AABB estimate got the ratio right
to within 0.6% (1.482 vs 1.473) — so every *relative* claim in §4 stands. But it
under-counted absolute instances by **4.5×** on both scenes, so the "evaluations"
figures in §4 (130.8 M / 193.8 M) are ~4.5× too low and the "useful fraction"
(22.3% / 12.9%) is correspondingly ~4.5× too *high*. True useful fraction is
closer to **5%** and **2.9%**. The direction of every argument is unchanged and
the waste is in fact larger than stated — which, given §7, is still the price of
amortization rather than a defect.

The under-count is expected in sign but not magnitude: the analytic model used a
per-camera max over training views and a face-on 3σ footprint, whereas the real
binner emits per rendered frame across all tiles a Gauss touches. It is a proxy
for *relative* footprint, not an absolute instance count. Do not reuse it for
absolute work estimates.

**Cost per instance is nearly constant across the two scenes** (0.452 vs
0.425 ns, within 6%) despite a 2.5× difference in primitive count and a 1.4×
difference in mean surfel size. That is the strongest single piece of evidence
for the model: **instances is the unit of work.** BS3k is marginally cheaper per
instance because its smaller splats terminate the alpha cascade sooner within
each tile.

Residual 7.7% gap between the instance ratio (1.473×) and the frame-time ratio
(1.367×) is the part that does *not* scale with instances — per-Gaussian
preprocess (≈2%, §4) plus sort and cull terms that scale with N and with
instances at different rates.

## 9. Late saturation — measured, only 3–4% (not a lever)

Question: the block only retires when **all** 256 pixels have saturated
(`__syncthreads_count(done) == BLOCK_SIZE`). Does one stubborn pixel — say one
seeing through foliage into distant sky — hold the whole tile hostage?

Counters added to the scratch clone (`reset_sat_counters` /
`read_sat_counters`):

* `thread_rounds_run` — Σ over threads of rounds the BLOCK executed
* `thread_rounds_needed` — Σ over threads of rounds until THAT thread saturated

| | thread-rounds run | needed | **waste** |
|---|---:|---:|---:|
| RD | 133,705,216 | 129,301,450 | **3.3%** |
| BS3k | 178,950,656 | 171,798,402 | **4.0%** |

**Only 3–4%, so per-block exit is a non-issue.** The reason it is this small is
the inner-loop guard:

```cuda
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
```

A saturated thread **exits its inner loop immediately** — it does not
re-evaluate the tile's surfels. It costs only its share of the shared-memory
staging (`if (range.x + progress < range.y) { ...stage... }`, which is NOT
gated on `done`) plus barrier participation. So the cost is a few extra global
loads, not 256 wasted evaluations.

BS3k's waste is marginally higher (4.0% vs 3.3%) — consistent with its many
weak splats decaying `T` more slowly, so pixels saturate later and blocks run
slightly longer past the point of usefulness. But the effect is ~0.7 pp, far
too small to explain a 1.37× frame-time gap.

**Conclusion: the tile walk itself is near-optimal.** Between this, the 8×8
result (§7) and the near-constant cost per instance (§8), all three
renderer-side hypotheses are now closed. Gating the staging on `!done`, or a
per-warp exit, would be chasing ~3–4% at best.

### Correction to an earlier claim in this doc

Section 4's "1,024 threads evaluate it to shade ~400" overstates the case: it
is true of the *binning*, but saturated threads skip the evaluation. The waste
is memory traffic for staging, not compute. §7's conclusion (waste is the price
of 256-wide amortization) is unaffected and is if anything strengthened — the
kernel is even tighter than §4 implied.

---

## 10. Fragment counters: the bottleneck is sub-tile coverage

§7–§9 closed every renderer-side hypothesis without ever saying what the frame
time *is* spent on. Counting instances answered "how many units of work"; it did
not answer "what does a unit cost, and how much of it is useful". So the inner
loop was instrumented directly:

- `d_frag_evals`   — inner-loop iterations entered = a thread reading a staged
  Gauss out of shared and reconstructing its fragment.
- `d_frag_blended` — iterations surviving every cull (denom, `rho3d >= k²`,
  `alpha < 1/255`) to reach the atlas fetch + blend recurrence.

Both are per-thread locals atomically folded once at kernel exit. The driver
(`/tmp/frag_probe.py`) wraps `_C.rasterize_gaussians` to count invocations, so
these are **per-frame** figures — unlike §9's absolutes, which were cumulative
over the whole benchmark (that ratio was still valid; the absolutes were not).

treehill, CONIC lean, `--aabb_mode 3`, 256 render calls:

| per frame | RD | BS3k |
|---|---:|---:|
| instances (Gauss × tile) | 1,519,354 | 2,286,897 |
| fragment evaluations | 386,610,493 | 549,914,425 |
| fragments reaching a blend | 29,289,117 | 24,849,121 |
| **blend survival** | **7.6%** | **4.5%** |

### Shared-memory bandwidth is NOT the wall

The prior hypothesis — that the 256× read amplification on staged data saturates
shared bandwidth — is refuted. The always-read prefix before the first cull is
~28 B (xy 8 + u₀v₀ 4 + J⁻¹ 8 + dw 8); the wider fields (`auv_*`, `collected_id`,
32+ B) are read only by the ~5–8% that blend. That gives

    386.6M × 28 B = 10.8 GB/frame ÷ 1.537 ms ≈ 7.0 TB/s

against a ~39–52 TB/s aggregate shared ceiling: **~15–18% utilised**. Both
factors in the earlier 90% estimate were wrong (60 B assumed vs 28 B real,
582M evals assumed vs 387M real).

### What the numbers do say

Normalising per instance:

| per instance (Gauss × tile) | RD | BS3k |
|---|---:|---:|
| threads evaluating it | 254.5 | 240.5 |
| pixels actually blending | 19.3 | 10.9 |
| **tile coverage** | **7.5%** | **4.2%** |

254.5 ≈ 256 confirms there is effectively **no intra-round early termination**:
every thread evaluates every staged surfel. And of those 256 evaluations, ~19
produce a pixel. **92.4% of all per-fragment work is discarded** (95.5% on BS3k).

This is *with* ellipse-tight binning already on — `--aabb_mode 3` is
rect + AdR + AccuTile SnugBox, the tightest available (mode 5 is an alias for the
weaker mode 2). AccuTile guarantees the ellipse *intersects* the tile; it cannot
constrain how much of the tile the ellipse *fills*. The waste lives strictly
below tile granularity.

### This resolves the BS3k paradox

BS3k genuinely has **lower overdraw** — 24.8M blends vs RD's 29.3M — and is still
slower, because overdraw counts *survivors* while frame time is paid by
*candidates*. BS3k burns 42% more fragment evaluations (550M vs 387M) to produce
15% fewer blends. Its survival rate is 1.7× worse.

blur_split moves the two in opposite directions: splitting makes each splat
weaker (fewer fragments clear `alpha ≥ 1/255`) while spreading it across more
tiles it merely grazes (more fragments evaluated). Coverage drops 7.5% → 4.2%.
"Lower overdraw" was never evidence of a cheaper frame.

### Consequence

The lever is **coverage per instance**, not count. A splat is cheap when it fills
the tiles it touches. This is the quantitative case against indiscriminate
blur_split, and the metric any training-side fix should be scored on — coverage,
not primitive count and not overdraw.

Renderer-side, the remaining idea is sub-tile rejection (e.g. a per-warp or
per-quad bound test before the full reconstruct), which §7's 8×8 result suggests
must not come at the cost of staging amortization.

### Not measured

What the surviving ~208 instruction-slots per evaluation actually stall on —
MUFU throughput (`powf` + `expf` + the `1/denom` reciprocal are up to 4 SFU ops
per fragment, and SFU runs at 1/32 the FP32 lane rate), LSU latency, or branch
divergence — is **not** separable by this arithmetic. That needs `ncu`.

Instrumentation cost: the counted build reads 650/582 FPS vs 783/621
uninstrumented, compressing the gap to 1.12× from 1.26×. Count ratios are
unaffected; do not read frame times off this build.

---

## 11. ncu + nsys: issue-bound on the LSU pipe; frame-level split

Nsight Compute (`--set full`, 6 launches of `renderBakedCUDA` per checkpoint,
clean CONIC lean build, treehill) plus an Nsight Systems timeline. Reports:
`/tmp/ncu_{rd,bs3k}.ncu-rep`, `/tmp/nsys_{rd,bs3k}.nsys-rep`.

### Inside the render kernel: instruction-issue-bound, LSU on top

| metric | RD | BS3k |
|---|---:|---:|
| kernel duration (mean) | 494 µs | 701 µs |
| issue slots busy | 79.6% | ~79% |
| executed IPC (of 4.0 max) | 3.18 | ~3.1 |
| **LSU pipe** (load/store instr.) | **65.6%** | **~66%** |
| FMA pipe (fp32+int, incl. fp16 21.3%) | 41.0% | ~41% |
| ALU pipe | 18.1% | ~18% |
| **XU pipe (MUFU: powf/expf/rcp)** | **9.6%** | **8.1%** |
| TEX pipe (atlas fetch) | 0.0% | 0.0% |
| DRAM | 1.2% | ~1% |
| L2 hit | 94.7% | ~95% |
| shared wavefronts (of peak) | 28.4% | — |
| shared bank conflicts | 0.1% | — |
| executed warp-instructions / launch | 474.4 M | 697.2 M |

The stall picture confirms throughput-bound, not latency-bound: the largest
"stall" is **not-selected** (4.0 of 11.7 cycles between issues, 34%) — warps
are eligible and waiting for the scheduler, which is saturated. Long-scoreboard
(global-latency) is only 0.73 cycles; occupancy is 78% with the top stall
saying more warps would not help.

Three hypotheses die here:

- **MUFU/SFU (special function unit) is NOT the wall** — XU at 9.6%,
  math-pipe-throttle stall 2.5%. `powf`/`expf` micro-opts (LOCK_SHAPE) are
  not worth it on desktop.
- **Shared/L1 bandwidth is NOT the wall** (§10's arithmetic confirmed in
  hardware: 28% of peak wavefronts, negligible bank conflicts). The LSU number
  is *instruction slots*, not bytes — many small LDS ops, each a cheap load of
  a wide free bus.
- **Atlas texture fetches are FREE** — TEX pipe 0.0%. The HW-bilinear path
  costs nothing measurable; only 4.5–7.6% of fragments reach it.

What remains is arithmetic identity: instructions scale 1.47× RD→BS3k, evals
1.42×, kernel duration 1.42×. Per inner-loop iteration the kernel executes a
measured **~36 warp-instructions** (474.4 M / (386.6 M evals ÷ 29.1 active
threads/warp)) — §10's "~208 instruction-slots" estimate was ~6× high; the
culled path is ~36 instructions of which ~6 are LDS, and the machine retires
them at a fixed ~3.2 IPC. **Frame kernel time = iterations × 36 / issue rate.**
The only lever with real range is *fewer iterations* — i.e. the §10 coverage
lever — with instruction-count reduction per iteration (fewer LDS via packing)
a bounded second.

### Frame-level split (nsys per-launch means × clean frame time)

Clean untraced re-bench on idle GPU: RD **1265.5 FPS** (790 µs), BS3k
**927.4 FPS** (1078 µs). (nsys-traced FPS was within 2% — 1243.7/921.0. The
782.6/621.0 previously stored in `benchmark_results.json` were depressed ~1.6×:
they were written by the counter-instrumented t8 runs of §9. Rewritten clean.
All same-build ratios in §7–§10 are unaffected; absolutes from instrumented
builds are not benchmarks.)

| per frame | RD (µs) | RD % | BS3k (µs) | BS3k % | Δ (µs) |
|---|---:|---:|---:|---:|---:|
| renderBakedCUDA | 413.5 | 52% | 589.3 | 55% | **+175.8** |
| duplicateWithKeys | 117.1 | 15% | 139.8 | 13% | +22.7 |
| radix sort (6 onesweep + hist) | 110.4 | 14% | 143.7 | 13% | +33.3 |
| preprocessCUDA | 27.8 | 3.5% | 78.5 | 7% | +50.7 |
| identifyTileRanges | ~5 | 0.6% | 5.9 | 0.5% | +0.9 |
| residual (launch gaps, misc ops) | ~116 | 15% | ~121 | 11% | +5 |
| **frame** | **790** | | **1078** | | **+288** |

So the BS3k slowdown is 61% render kernel, 18% preprocess (linear in the 2.5×
primitive count), 20% binning+sort (linear in the 1.5× instance count). And
structurally: the render kernel is **~52–55%** of the frame; **binning+sort is
~28%** — which upgrades GS-TG-style sort-granularity ideas from "probably
irrelevant" to a real (if bounded) second target.

### Consequences, ranked by measured headroom

1. **Warp-strip mask** (GS-TG §IV bitmask, repurposed sub-tile): at binning,
   emit an 8-bit mask per instance — one bit per 16×2-pixel warp strip the
   ellipse overlaps. Inner loop: warp-uniform 1-bit test replaces the
   ~36-instruction eval for empty strips. Attacks the issue bound exactly
   where §10 localized the waste. Ceiling measurable first: count strips with
   nonzero coverage per instance in the t8 clone.
2. **LDS packing**: `uv0_h(4B) + J_h(8B) + opa(2B) + shape(2B) = 16 B` — one
   LDS.128 instead of four loads; `dwdpxy` a fifth as LDS.64. Cuts the top
   pipe (LSU 65.6%) roughly in half; bounded by FMA at 41% becoming the next
   ceiling.
3. **Binning+sort** (~28% of frame): fewer keys (coarser sort granularity à la
   GS-TG, or instance-count reduction from training-side coverage fixes) —
   every instance removed also saves its dup+sort+render cost.
4. NOT worth it: powf/expf tricks (XU 9.6%), occupancy tuning (not-selected
   dominant), atlas fetch optimization (TEX 0%), shared-bandwidth reduction
   (28%).
