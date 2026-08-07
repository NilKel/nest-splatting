# mip-360 render bottleneck analysis — nest baked vs FastGS

**Purpose:** figure out what actually limits the CONIC baked renderer's
FPS on the RTX 5090, and where the deltas vs FastGS come from.

**Setup:**
- Nest: `--method 3D_SH_res`, config
  `RD_SV_30thr_005w25gLP_N2f_frz5k10`, HD BC7 atlas at `max_res 64`,
  rendered through `diff_surfel_bake_render_lean` built with
  `LEAN_FLAGS="CONIC"` (CONIC linearization + fp16 pack + atlas UV
  precompute). See [BENCH_5090_MIP360.md](../docs/BENCH_5090_MIP360.md).
- FastGS: `iter_30000` checkpoints trained locally on the 5090; timing
  via `FastGS/bench_fps.py --num_warmup 10 --num_benchmark 200`.
- Overdraw: per-pixel contributor counts from
  `speed_comparison/render_intersection_all.py` (nest neural) and
  `FastGS/intersection_maps.py` (FastGS's own tool).
- Date: 2026-07-21.

---

## 1. Baseline table — FPS + overdraw + Gauss counts (all 9 scenes)

| scene | resolution | nest Gauss | FastGS Gauss | Gauss ratio (F/N) | nest overdraw | FastGS overdraw | Overdraw ratio (F/N) | nest FPS | FastGS FPS | FPS ratio (N/F) |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle  | 1237×822  | 161k | 540k | 3.36 | 20.2 | 42.4 | 2.10 | 1385 | 1172 | 1.18 |
| bonsai   | 1559×1039 |  92k | 276k | 3.00 | 17.3 | 46.9 | 2.71 | 1261 | 1236 | 1.02 |
| counter  | 1558×1038 |  68k | 208k | 3.06 | 21.6 | 54.0 | 2.50 | 1442 | 1165 | 1.24 |
| flowers  | 1256×828  | 185k | 490k | 2.65 | 22.1 | 44.9 | 2.03 | 1291 | 1147 | 1.13 |
| garden   | 1297×840  | 149k | 661k | 4.44 | 20.1 | 42.0 | 2.09 | 1943 | 1138 | **1.71** |
| kitchen  | 1558×1039 | 120k | 380k | 3.17 | 25.1 | 57.2 | 2.28 | 1495 | 1046 | 1.43 |
| room     | 1557×1038 |  63k | 207k | 3.29 | 15.4 | 51.0 | 3.31 | 1792 | 1294 | 1.38 |
| stump    | 1245×825  |  96k | 392k | 4.08 | 19.1 | 37.6 | 1.97 | 1394 | 1225 | 1.14 |
| treehill | 1267×832  | 175k | 394k | 2.25 | 27.7 | 49.7 | 1.79 | 1105 | 1221 | **0.90** |
| **mean** | — | **123k** | **394k** | **3.20** | **20.96** | **47.28** | **2.26** | **1456** | **1183** | **1.23** |

**Read at a glance:**
- Nest carries **3.2× fewer Gauss** on average and each pixel is touched
  by **2.3× fewer contributors**.
- Nest wins by **1.23× mean FPS**. Best scene garden (1.71×). Worst
  scene treehill (0.90× — the only scene where FastGS wins on FPS).

---

## 2. What actually correlates with the FPS advantage

Cross-scene Pearson correlations between scene metrics and FPS ratio:

| predictor | r vs FPS ratio | direction | independence check |
|---|---:|---|---|
| **Gauss-count ratio (F/N)** | **+0.72** | more Gauss on their side ⇒ more FPS win | r=0.002 vs overdraw ratio (independent) |
| **Frac fat surfels (axis > 0.2)** | **−0.63** | more fat surfels on our side ⇒ less FPS win | r=0.14 vs Gauss ratio (independent) |
| Nest mean surfel size | −0.53 | same | |
| Nest p95 surfel size | −0.48 | same | |
| Nest p99 surfel size | −0.47 | same | |
| Frac surfels axis > 0.5 | −0.45 | same | |
| Overdraw ratio (F/N) | **+0.22** | weak positive | r=−0.48 vs size (weakly negatively correlated) |
| Nest gauss count (raw) | −0.16 | noise | |
| Nest overdraw (raw) | −0.25 | noise | |

**Conclusion:** three independent effects, ranked by strength:
1. **Gauss count** (largest lever — hits sort/cull/preprocess cost)
2. **Surfel size / bloat** (second lever — hits fragment count via AABB)
3. **Overdraw ratio** (weakest — surprising, given it's the most-cited metric)

The Gauss ratio and size predictor are largely independent (r=0.14),
so they explain different parts of the variance. A regression on
{Gauss ratio, frac_>0.2_axis} would probably reach R² ~0.75–0.85;
adding overdraw ratio contributes almost nothing.

---

## 3. Per-scene surfel scale distributions

`get_scaling.max(axis=-1)` = biggest activated in-plane axis per surfel.
Fraction of surfels with that axis exceeding 0.1, 0.2, 0.5.

| scene | tag | N | mean | p50 | p95 | p99 | max | >0.1% | >0.2% | >0.5% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden   | winner (1.71×) | 149k | 0.057 | 0.036 | 0.157 | 0.384 | 6.4 | 10.9 | 3.2 | 0.61 |
| room     | winner (1.38×) |  63k | 0.055 | 0.035 | 0.170 | 0.356 | 6.5 | 12.5 | 3.5 | 0.38 |
| kitchen  | winner (1.43×) | 119k | 0.039 | 0.026 | 0.142 | 0.420 | — | — | 3.2 | 0.73 |
| counter  | winner (1.24×) |  68k | 0.046 | — | 0.169 | 0.370 | — | — | 3.8 | 0.39 |
| bicycle  | winner (1.18×) | 161k | 0.096 | — | 0.294 | 0.674 | — | — | 9.9 | 1.74 |
| bonsai   | tied (1.02×)   |  92k | 0.069 | 0.027 | 0.263 | 0.518 | 25.5 | 20.9 | 8.3 | 1.12 |
| flowers  | winner (1.13×) | 185k | 0.141 | — | 0.410 | 1.587 | — | — | 14.0 | 3.80 |
| **stump**    | winner (1.14×) |  96k | **0.211** | 0.046 | **0.877** | **2.728** | 34.2 | 34.2 | 21.7 | 8.58 |
| **treehill** | **LOSER (0.90×)** | 175k | **0.134** | 0.063 | **0.406** | **1.350** | 14.0 | 30.3 | **13.3** | **3.70** |

**Treehill's bloat is real** — 2.4× the mean surfel size of garden/room,
2.6× the p95, 3.5× the p99, **6× more surfels with axis > 0.5** (3.7%
vs 0.4-0.6%). This directly produces treehill's outlier overdraw of
27.7 contributors/pixel (the highest of all 9 scenes) despite it being
an outdoor low-res scene.

**Interesting counter — stump has EVEN bigger surfels** (mean 0.21,
p99 2.7, 8.6% with axis > 0.5) but still wins by 1.14× because its
Gauss-count advantage (4.08×) offsets the fragment penalty from the
bloat. **Bloat + few Gauss = OK. Bloat + not-so-few Gauss = losing
(treehill).**

---

## 4. Experiment: `LEAN_FLAGS="CONIC,LOCK_SHAPE"` — kill `powf` entirely

Hypothesis: `powf(base, shape)` is the biggest single per-fragment op,
so hardcoding `alpha_beta = base` (equivalent to shape=1, fattest
single-mul kernel) should reveal the pow ceiling.

Full 9-scene sweep, LEAN builds:

| scene | CONIC SH+atlas | CONIC+LOCK_SHAPE SH+atlas | Δ | CONIC SH-only | CONIC+LOCK_SHAPE SH-only | Δ |
|---|---:|---:|---:|---:|---:|---:|
| bicycle  | 1385 | 1392 | +0.5% | 1598 | 1630 | +2.0% |
| bonsai   | 1261 | 1273 | +1.0% | 1398 | 1418 | +1.4% |
| counter  | 1442 | 1419 | −1.6% | 1652 | 1648 | −0.2% |
| flowers  | 1291 | 1303 | +0.9% | 1401 | 1427 | +1.9% |
| garden   | 1943 | 1977 | +1.7% | 2337 | 2387 | +2.1% |
| kitchen  | 1495 | 1466 | −1.9% | 1786 | 1776 | −0.6% |
| room     | 1792 | 1801 | +0.5% | 2012 | 2034 | +1.1% |
| stump    | 1394 | 1364 | −2.2% | 1606 | 1601 | −0.3% |
| treehill | 1105 | 1100 | −0.5% | 1319 | 1336 | +1.3% |
| **mean** | **1456** | **1477** | **+1.4%** | **1679** | **1721** | **+2.5%** |

**PSNR cost** (SH+atlas): −0.03 to −1.42 dB per scene (worst on bonsai,
best on counter). Bonsai's PSNR drops most because it has a large
fraction of low-shape (fat) surfels whose falloff shape is far from 1.

**Read:**
- 5090 gains **~1.4% SH+atlas, ~2.5% SH-only** mean. Well below what
  the ops-accounting would suggest.
- On Blackwell + `--use_fast_math`, `__powf` is ~4 cycles per call.
  In a ~50-cycle fragment that's roughly 8%, but the observed win is
  smaller because the perf saving is partly cancelled by an overdraw
  side-effect (see next section).
- On a mobile GPU where `pow()` runs 30-50 cycles, the same change
  should give an order-of-magnitude larger relative speedup.

**Best remaining lever that isn't pow:** forcing `kernel_type=0`
(Gaussian branch) — skips compact-support discard, base compute,
kernel dispatch chain, and `max(alpha_beta, alpha_lp)` selection.
Estimated ~15-25% on 5090 but comes with a bigger PSNR hit than
LOCK_SHAPE.

---

## 5. Experiment: overdraw under `--lock_shape 1.0`

Ran the neural intersection tool with each Gauss's shape forced to
1.0 (activated). Hypothesis was that overdraw would DROP because a
sharper falloff means more fragments fail the `alpha < 1/255` cull.

Actual result on 4 scenes:

| scene | trained-shape overdraw | shape=1 overdraw | Δ |
|---|---:|---:|---:|
| garden   | 20.06 | 20.67 | **+3.0%** |
| room     | 15.37 | 16.74 | **+8.9%** |
| stump    | 19.15 | 21.40 | **+11.7%** |
| treehill | 27.72 | 29.26 | **+5.5%** |

**Overdraw went UP, not down.** Why:

For beta_scaled kernel `alpha_beta = (1 − ρ3d/9)^β`, at fixed ρ3d < 9,
`base < 1`, so:
- β > 1 ⇒ `base^β < base`   → sharper falloff than shape=1
- β < 1 ⇒ `base^β > base`   → gentler falloff than shape=1
- β = 1 ⇒ linear

Trained shape distribution has **mean ~0.3** but a meaningful **tail
extends past 1.0** (max ~2.8). Locking to 1 means:
- Most Gauss (β < 1, majority) get SHARPER — small contributor drop
- Tail Gauss (β > 1, minority ~10-15%) get GENTLER — big contributor gain
- Gentler-tail Gauss dominate because each contributes to *many*
  extra pixels

**Corollary — LOCK_SHAPE perf gain decomposed:**
- `pow` removal alone: ~7-8% per fragment on 5090 (matches ~4 cycles
  in a ~50-cycle frag).
- Overdraw side-effect: +5-9% more work per frame.
- Net: **+1.4% mean FPS** — the pow saving barely covered the overdraw
  hit.

**Better move for a real speedup:** per-Gauss shape quantization to
the CLOSEST fast-path value in {0.5, 1, 2, 4}. Each Gauss keeps a
close-to-trained falloff (overdraw stable), `pow()` vanishes for all
of them. Would probably be **~+2-5% on 5090** and much more on mobile.
The docs mention `scripts/quantize_shape.py` for exactly this
purpose.

---

## 6. Experiment (in progress): filter treehill by surfel size

Question: if bloat is really what breaks treehill, does simply
dropping the fat surfels lift FPS above FastGS's 1221?

Patched `bench_lean_vs_prod.py` with `--max_axis` (see the change);
running:
- `--max_axis 1000` (identity, no filter)
- `--max_axis 0.5`  (drop biggest 3.7% = 6,492 surfels)
- `--max_axis 0.2`  (drop biggest 13.3% = 23,393 surfels)
- `--max_axis 0.1`  (drop biggest 30.3% = 53,167 surfels)
- `--max_axis 0.05` (drop biggest ~50%, aggressive)

Results will land in this doc as they come in. Expected: mid-tier
filter (~0.2) lifts treehill above FastGS while remaining visually
reasonable; aggressive filter (~0.05) shows the "no-bloat ceiling"
but is destructive to quality.

**Results (2026-07-21 sweep, LEAN CONIC, `--num_warmup 30 --num_benchmark 300`):**

| threshold | Gauss kept | LEAN SH+atlas FPS | ΔFPS vs no-filter | vs FastGS 1221 | PSNR SH+atlas | Δ dB |
|---|---:|---:|---:|---:|---:|---:|
| ∞ (no filter) | 174,744 (100%) | **1100** | — | 0.90× | 22.32 | — |
| axis ≤ 0.5 | 168,276 (96.3%) | **1247** | +13.4% | 1.02× | 17.27 | **−5.05** |
| axis ≤ 0.2 | 151,438 (86.7%) | **1591** | +44.7% | **1.30×** | 15.16 | −7.16 |
| axis ≤ 0.1 | 121,800 (69.7%) | **2067** | +87.9% | 1.69× | 13.72 | −8.60 |
| axis ≤ 0.05 | 67,245 (38.5%) | **3797** | +245%  | 3.11× | 11.69 | −10.63 |

SH-only side (atlas fetch bypassed) shows the same trend, more starkly:
1319 → 1560 (+18%) → 2077 (+57%) → 2667 (+102%) → 4640 (+252%).

**Confirmed and quantified:**
- **Dropping just the 3.7% biggest surfels lifts FPS by 13%** (1100 →
  1247), *just* clearing FastGS's 1221.
- **Dropping the top 13.3% (axis > 0.2) hits 1591 FPS = 1.30× FastGS.**
  So the bloat alone is the difference between winning and losing on
  treehill.
- **Dropping 30% (axis > 0.1) reaches 2067 FPS = 1.69× FastGS**
  — matches our winning-scene FPS ratios, so this is the ceiling
  treehill *would* achieve with garden/room-like surfel size stats.

**But the quality cost is brutal — even the smallest filter drops
PSNR by 5 dB.** Those big surfels aren't noise; they're covering large
background regions (sky, distant trees, understorey) that a single
huge surfel represents efficiently. Dropping them creates literal
holes.

**So this experiment is a ceiling test, not a fix.** The realistic
way to convert the perf potential into a shipping bake is to
**retrain treehill with tighter regularization** — force the big
surfels to be replaced by many smaller ones covering the same
region:

- `--scale_reg` (MCMC path) or higher `opacity_reg` — penalize large
  scale directly.
- Aggressive densification (`--mini` / `--minimc` / more clone events)
  — split fat surfels into smaller ones during training.
- Depth-reinit with more sampling in high-scale regions.

Prediction: a retrained treehill with the same PSNR (~22.3 dB) but
scale distribution matched to garden's (mean 0.06 instead of 0.13)
would land somewhere around 1500-1800 FPS — winning vs FastGS while
matching or exceeding current quality.

**Also of note:** the filter experiment shows the frame time is
*extremely* sensitive to the biggest surfels. At axis ≤ 0.5 (drop 3.7%
of surfels) the atlas-fetch cost also drops disproportionately —
LEAN SH+atlas moved 1100 → 1247 (+147 FPS) while LEAN SH-only moved
1319 → 1560 (+241 FPS). So the biggest surfels touch atlas the
hardest too (their big AABBs multiply the atlas-sample count).

---

## 7. Working conclusions

- **Overdraw ratio is a distraction on this data.** It correlates
  barely (r=0.22) with our FPS advantage. Most speed comes from
  having ~3× fewer Gauss (setup + sort dominate frame time at these
  frame rates).
- **Surfel bloat is a real and independent second factor** (r=−0.63
  for fraction > 0.2 axis). Treehill is the clearest case where bloat
  costs us the win: **dropping the top 13% biggest surfels alone
  lifts FPS from 1100 → 1591, moving from 0.90× FastGS to 1.30×**
  (§6). The bloat is *literally* the difference between our only
  losing scene and a comfortable win.
- **`pow(base, shape)` isn't a 5090 bottleneck** (~1.4% total; ~4
  cycles per call under `__powf`). Same lever should be much bigger
  on mobile.
- **Locking shape to 1 doesn't help — it slightly hurts overdraw**
  because the trained shape has a fat tail past 1. Per-Gauss shape
  quantization to the CLOSEST fast-path value is the correct low-risk
  move.
- **The realistic desktop headroom** looks like:
  - ~15-30% from bloat cleanup on the outdoor scenes (retrain with
    tighter scale reg — treehill and flowers benefit most)
  - ~10-25% from kernel simplification (Gaussian-kernel force, or
    shape quantization killing pow)
  - Approaching FastGS's per-Gauss fragment cost isn't on the table
    without changing the render formula significantly.

## 7b. WebGPU-equivalent overdraw (no CUDA early exit)

The intersection counts in §1 are measured with the CUDA renderer's
`if (T < 0.0001) done = true; break;` early exit engaged. WebGPU has no
per-pixel persistent state during rasterization → no such exit. To
measure what the WebGPU renderer *actually* pays, patched
`diff_surfel_3D_sh_res/cuda_rasterizer/forward.cu` with a
`#ifndef DISABLE_EARLY_EXIT` guard around the three T-saturation branches
and rebuilt with `NO_EARLY_EXIT=1`.

Garden (24 test views):

|  | CUDA baseline (T-exit) | no early exit (≈ WebGPU) | Δ |
|---|---:|---:|---:|
| **global mean contributors/pixel** | 20.06 | **23.01** | **1.15×** |
| global max | 84 | 163 | 1.94× |
| per-view mean | 20.06 | 23.01 | 1.15× |
| per-view p95  | 39.9 | 54.3 | 1.36× |
| per-view p99  | 48.5 | 75.7 | 1.56× |
| per-view max  | 75.2 | 127.8 | 1.70× |

**Reads:**
- Mean overhead is only **+15%** — for most pixels early exit doesn't
  skip a whole lot on garden.
- Tail is much worse: **p99 pixels do 56% more work, max pixels 70%
  more.** Those are exactly the "deep into a densely-populated tile"
  pixels where T saturates fast on the CUDA side and everything past
  that point gets skipped.
- The 1.94× ratio in global max says some pixels have 2× as much
  dead work behind them as get counted in the CUDA baseline.

**Implication for a proxy-mesh depth pre-pass on WebGPU:** the "extra"
contributors are exactly the ones a depth pre-pass would cull (they're
BEHIND the opaque frontier and would fail a `depth < proxy_depth`
test). On mobile TBDR GPUs the p99 pixel drives the tile cost, so
addressing the tail matters much more than the mean would suggest.

Rough expected win from a Hi-Z-style pre-pass:
- Desktop: recover the 15% mean overhead → +15% render, marginal in
  frame time when render is already sub-ms.
- Mobile: recover a bigger fraction of the 56% p99 gap (tile-level
  early-Z kicks in before fragment shading), plus TBDR's per-tile
  serialization is dominated by the worst-case pixel — could easily
  be **20-35% render savings**.

## 8. Open questions / next levers

- **Retrain treehill with tighter `--scale_reg` or `opacity_reg`**
  — the filter-bench (§6) proved bloat costs us ~45% FPS; a retrain
  should convert most of that into a real win without the PSNR
  destruction the filter caused. Prediction: 1500-1800 FPS at the same
  22.3 dB.
- Same treatment for flowers (13% > 0.2 axis) — currently winning
  1.13× but its FPS is the lowest of the winning scenes (1291); a
  bloat cleanup would probably move it to 1.4-1.5×.
- Per-Gauss shape quantization on a bundle — real desktop AND
  mobile perf test. Should sidestep the LOCK_SHAPE=1 overdraw side-
  effect while eliminating pow.
- Compile-time `kernel_type=0` variant of the lean CUDA kernel —
  measure how big the Gaussian-branch ceiling is on desktop.
- Sanity-check on stump: massive surfel bloat (21.7% > 0.2 axis)
  but still winning 1.14× because of the 4.08× Gauss ratio. Would
  a bloat cleanup change this dynamic?
