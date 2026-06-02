# Atlas decomposition + clustering experiments

Running notebook for the multi-level atlas + cross-Gauss clustering line of
work. All experiments on the **room** scene, production config
`SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac` (75 207 live Gausses, 4 hashgrid
levels × 4D = 16D concatenated → view-independent MLP → 3D RGB residual,
baked at `max_res 64`).

Reference numbers:

|  | size | PSNR (atlas-fidelity) |
|---|---:|---:|
| `baked.ply` | 32 MB | — |
| `atlas_texture.pt` (uint8 source) | 675 MB | — |
| `atlas_texture.bc7` (deployed) | **225 MB** | reference (uint8→BC7 is the bake's own quant noise) |

All "compressed MB" figures below are BC7-equivalent. PSNR is atlas-fidelity
(reconstructed patches vs the dequantised single-bake atlas), NOT
rendered-image PSNR.

---

## 1. Per-level decomposition (the approach we're using)

The MLP residual at each (Gauss, UV) is `f(z₀ ⊕ z₁ ⊕ z₂ ⊕ z₃)`. The
decomposition factorises this into 4 per-level atlases `A_0…A_3` with
`Σ_k A_k = T` (the deployed atlas) to FP precision.

A naive `Σ_k MLP(only z_k)` decomposition is **~25 dB off** the target
because of MLP nonlinearity (ReLU cross-coupling). Doesn't matter — we
finetune. **Joint Adam over per-bucket `A_k` patches converges to >95 dB
per bucket and 98.84 dB globally** in 48 s on a 5090, blowing past the
atlas's own uint8 quant noise floor. Treat the decomposed atlas as an
exact reproduction of the deployed atlas, factorised into 4 additive
parts.

Script: `scripts/bake_decompose_full.py`. Saves
`<bake_dir>/decomposed/bucket_<rx>×<ry>.pt`.

### 1.1 Per-bucket finetune results (full 75 207 Gausses)

| bucket | N | init PSNR (naive Σ T_k) | final PSNR (after finetune) | time |
|---|---:|---:|---:|---:|
| 64×64 | 48 637 | 26.3 | **99.5** | 33 s |
| 32×64 | 6 743 | 24.0 | 96.8 | 2.7 s |
| 64×32 | 6 437 | 24.1 | 96.7 | 2.5 s |
| 32×32 | 2 561 | 23.0 | 96.3 | 0.6 s |
| 16×64 | 2 391 | 23.1 | 96.2 | 0.5 s |
| 64×16 | 2 135 | 23.1 | 96.2 | 0.5 s |
| 16×32 | 1 391 | 21.8 | 94.2 | 0.2 s |
| 32×16 | 1 277 | 21.8 | 94.1 | 0.2 s |
| 16×16 | 826 | 21.6 | 93.7 | 0.2 s |
| ⋯ small buckets | ⋯ | 20–23 | 92–95 | ≈ 0.1 s |
| 8×8 | 146 | 20.0 | 91.7 | 0.1 s |

Global `Σ_k A_k` vs deployed atlas: **98.84 dB**. Per-level magnitudes
(after finetune) decrease finest → coarsest: L0 = 0.053, L1 = 0.022,
L2 = 0.018, L3 = 0.015 (in atlas-residual units; atlas_scale ≈ 1.13).

### 1.2 Per-level structure observations
- No `A_k` is spatially flat across a Gauss's UV grid at ε = 1/255 —
  the "coarse-level patches collapse to one RGB" intuition doesn't
  fire at `uv_extent = 4`. Even the coarsest hashgrid voxel (~0.023
  world units) is crossed 3–4 times within a typical Gauss's UV grid.
  → §3 below tests whether subtracting the per-Gauss DC helps anyway.

---

## 2. Cross-Gauss clustering — single-bake atlas (baseline)

Per-resolution-bucket K-means on the deployed atlas patches. Variable
K_b = max(4, N_b // r) per bucket. Importance proxy = per-patch L2 norm
(c3dgs-style — high-energy patches pull centroids harder).

Script: `scripts/bake_cluster_single_test.py`.

### 2.1 Unweighted vs importance-weighted vs +keep-top-5 %

| compression | (A) unweighted | (B) patch-norm weighted | (C) weighted + keep top 5 % |
|---:|---:|---:|---:|
| 50.0 % storage | 30.36 dB | **30.56 dB** | 30.91 dB (52.5 %) |
| 25.0 % | 27.99 dB | 28.09 dB | 28.47 dB (28.8 %) |
| 12.5 % | 26.94 dB | 27.00 dB | 27.40 dB (16.9 %) |
| 6.3 % | 26.35 dB | — | — |
| 3.2 % | 25.97 dB | — | — |

- **Importance weighting alone: +0.1–0.2 dB.** Real but small —
  per-Gauss residual patches don't have the fat-tailed importance
  distribution that c3dgs's Gaussian color SHs do.
- **Keep-top-5 % uncompressed: +0.4–0.5 dB at a fixed +11 MB storage
  premium** (5 % × 75 k Gausses × 64²×3 bytes ≈ 11 MB regardless of K).
  Useful at moderate ratios; the premium dominates at aggressive
  compression. (At K = N/8 it bumps storage from 12.5 % to 17 %, eating
  most of the codebook reduction.)

### 2.2 Morton + DEFLATE on index stream — deferred
Index stream is < 0.15 MB at every K above. Even halving via DEFLATE
saves ~75 kB, a rounding error vs the codebook. Worth revisiting if we
move to product quantisation where index volume grows.

---

## 3. Cross-Gauss clustering — decomposed atlases (§1 finetuned)

Uses the saved `<bake_dir>/decomposed/bucket_*.pt` artifacts. Each level
clustered independently per bucket; recon = Σ_k cent_k[ass_k]. Joint
PSNR is recon vs the deployed atlas T (so the §1 finetune residual
(~99 dB) is the noise floor we can't beat).

Script: `scripts/bake_cluster_compare.py`. Four schemes:

- **A.** single-bake, K-means on raw patches  *(= §2)*
- **B.** single-bake, K-means on (patch − per-Gauss DC), DC stored separately
- **C.** decomposed, K-means per level on raw `A_k` patches
- **D.** decomposed, K-means per level on (`A_k` − per-Gauss DC_k)

The DC variant tests the "if there's a redundant DC component we
can bake it into the surfels" hypothesis. Per-Gauss DC table costs
`N × 3 B` (BC7-equiv) ≈ 0.21 MB single, 0.86 MB per-level — trivial.

### 3.1 Results (full 75 207 Gausses, max_bucket_n = 20 000)

| K (per-bucket) | A: single raw | B: single − DC | C: decomp 4-level raw | D: decomp − DC |
|---:|---:|---:|---:|---:|
| N/4 | 12.6 % / 27.97 dB | **12.7 % / 28.51 dB** | 50.4 % / 28.59 dB | 50.8 % / 29.13 dB |
| N/8 | 6.3 % / 26.83 | **6.4 % / 27.39** | 25.3 % / 27.44 | 25.7 % / 27.89 |
| N/16 | 3.2 % / 26.17 | **3.3 % / 26.76** | 12.7 % / 26.68 | 13.1 % / 27.27 |
| N/32 | 1.6 % / 25.77 | 1.7 % / 26.39 | 6.4 % / 26.23 | 6.8 % / 26.80 |
| N/64 | 0.8 % / 25.44 | 0.9 % / 26.03 | 3.3 % / 25.89 | 3.6 % / 26.43 |

All percentages are against the 224.89 MB single-bake BC7 reference.

### 3.2 Findings

**Per-Gauss DC subtraction is a clean +0.5–0.6 dB lift at every ratio
for both representations** (B − A and D − C deltas are essentially
identical). The DC table costs ~0.1 % of the deployed atlas. The
per-Gauss residual patches carry a substantial DC offset that the
codebook was wasting capacity on. **Folding this DC back into the
per-Gauss SH/SV baseline is essentially free and recovers ~0.5 dB
everywhere** — this should be productised.

**Single-bake beats per-level decomposition at every matched storage
point**, with the gap shrinking at aggressive compression:

| matched storage | single (B) | decomp (D) | gap |
|---:|---:|---:|---:|
| ~13 % | 28.51 dB | 27.27 dB | **−1.24 dB** |
| ~6 % | 26.83 dB | 26.23 dB | −0.60 dB |
| ~3 % | 26.17 dB | 25.89 dB | −0.28 dB |

The 4-level codebook overhead (`4 × K_per_level` codewords vs single's
`K`) never pays back. The decomposition is mathematically clean — Σ A_k
matches the deployed atlas to 99 dB — but the byte budget is spent
worse than just clustering T directly.

**At matched PSNR, decomposed needs ~2× the bytes.** To hit ~28 dB:
B (single + DC) uses 12.7 %; D (decomposed + DC) uses 25.7 %.

### 3.3 Best operating point on patch-level clustering

**B (single + DC, K_b = N_b / 4) → 28.51 dB at 12.7 % (28.6 MB)** vs the
225 MB deployed BC7. **~8× smaller atlas for ~0.5 dB atlas-fidelity
loss.** The corresponding rendered-image PSNR drop is typically 30–50 %
of the atlas-fidelity drop, so ~0.15–0.25 dB at the output.

### 3.4 Non-uniform K per level for the decomposed atlases

Hypothesis: per-level magnitudes drop 0.053 → 0.011 finest → coarsest,
so each level needs different cluster counts. Allocate more codebook to
the high-energy finest level, less to the smooth coarse levels.

Script: `scripts/bake_cluster_per_level.py`. K_k = N // R_k per level.

| config (K_0,K_1,K_2,K_3) | L0 PSNR | L1 | L2 | L3 | joint vs T | MB | % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `geom:8-16-32-64` (N/8,N/16,N/32,N/64) | 33.14 | 34.72 | 34.67 | 35.57 | 26.72 | 26.81 | 11.9 |
| `uniform/16` (N/16 × 4) | 32.39 | 34.76 | 34.97 | 35.92 | 26.65 | 28.60 | 12.7 |
| `geom:4-16-32-64` (N/4,N/16,N/32,N/64) | 34.37 | 34.73 | 34.66 | 35.56 | 27.14 | 40.95 | 18.2 |
| `geom:4-8-16-32` (N/4,N/8,N/16,N/32) | 34.37 | 35.27 | 34.99 | 35.75 | **27.49** | 53.34 | **23.7** |
| `uniform/8` (N/8 × 4) | 33.09 | 35.25 | 35.38 | 36.28 | 27.25 | 56.88 | 25.3 |
| `geom:2-8-32-128` | 37.11 | 35.25 | 34.67 | 35.46 | 27.97 | 75.40 | 33.5 |
| `uniform/4` (N/4 × 4) | 34.46 | 36.30 | 36.27 | 37.07 | **28.44** | 113.44 | 50.4 |

(Per-level PSNRs measured vs the unclustered A_k from the §1 finetune;
joint PSNR measured vs the deployed atlas T.)

**Findings:**
- **Coarser levels cluster ~3 dB easier per-level** at every config
  (L0 ≈ L3 − 3 dB), tracking the level-magnitude ratio.
- **Non-uniform K helps the joint by ~0.1–0.25 dB at matched storage.**
  Real but modest — allocating budget toward L0 lifts its PSNR by
  ~1.3 dB but the other levels lose fidelity faster than L0 gains.
- **Extreme L0-rich (geom:2-8-32-128) is suboptimal.** Joint sits between
  uniform/4 and uniform/8 in storage but is dominated on bytes by
  geom:4-8-16-32.
- **The decomposition + non-uniform K pulls roughly even with
  single-bake + DC at matched storage**, e.g. at ~13 %:
  geom:8-16-32-64 = 26.72 dB vs single-bake B @ N/16 = 26.76 dB. The
  earlier 1.3 dB gap closes to ~0 — but doesn't reverse meaningfully.

**Conclusion on the decomposition path**: the per-level decomposition is
mathematically clean (~99 dB Σ A_k = T) and non-uniform K narrows the
clustering gap, but it never compresses better than single-bake + DC
at matched bytes. Decomposition is **not** the storage lever it looked
like it could be. Single-bake + DC subtraction remains the
Pareto-best whole-patch approach.

---

## 4. Block-level clustering — 4×4 BC7-native blocks

Whole-patch K-means has only 75 207 vectors. **Block clustering** tiles
the atlas into 4×4 BC7-native blocks and clusters those 48-D vectors
instead — ~14.7 M vectors per atlas, two orders of magnitude more
cross-block redundancy to exploit. Codebook is a list of 4×4 RGB tiles
(K × 48 bytes uint8); each used block stores one index of `ceil(log₂K)`
bits.

Used 4×4 blocks: **14 741 890** (rect dims are powers of 2 ≥ 4, so each
per-Gauss rect tiles cleanly). Reference single-bake BC7 = 224.94 MB at
1 byte/texel.

K-means: fit on a 1.5 M sub-sample, assign over the full 14.7 M set.
Scripts use a 500 MB cap on the (chunk × K × 4 bytes) distance matrix —
chunk size auto-scales inversely with K.

### 4.1 Single-bake atlas

Script: `scripts/bake_cluster_blocks.py`.

| K | codebook | indices | total | vs single | recon PSNR |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.0 KB | 10.54 MB | 10.55 MB | 4.69 % | 33.43 dB |
| 256 | 4.1 KB | 14.06 MB | 14.06 MB | 6.25 % | 34.99 dB |
| 1 024 | 16.4 KB | 17.57 MB | 17.59 MB | 7.82 % | 36.29 dB |
| 4 096 | 65.5 KB | 21.09 MB | 21.15 MB | 9.40 % | 37.43 dB |
| 16 384 | 262 KB | 24.60 MB | 24.85 MB | 11.05 % | 38.37 dB |
| 65 536 | 1.0 MB | 28.12 MB | 29.12 MB | 12.94 % | **39.11 dB** |

**~+9 dB vs whole-patch clustering at matched bytes.** Patch
clustering (§2/§3) topped out at 28.5 dB at 12.7 %; block clustering
reaches 39 dB at the same byte budget. The 200×-larger sample count
+ 250×-lower dimensionality is decisive — flat-ish regions of the
atlas collapse to a handful of codewords.

### 4.2 Decomposed atlases (§1 finetuned) — block clustering per level

Script: `scripts/bake_cluster_blocks_decomposed.py`. Each level's
14.7 M blocks clustered independently with the same K. Joint recon =
Σ_k cent_k[ass_k] vs deployed atlas T. (Per-level PSNRs measured vs
the unclustered FP16 level cache, which carries ~53 dB of float→FP16
quant noise — sanity Σ A_k = T at 53 dB.)

| K (per level) | L0 | L1 | L2 | L3 | joint vs T | total | vs single |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 40.53 | 42.26 | 41.40 | 40.91 | 34.06 dB | 42.2 MB | 18.75 % |
| 256 | 42.82 | 43.71 | 42.76 | 42.05 | 35.64 dB | 56.3 MB | 25.01 % |
| 1 024 | 44.34 | 44.94 | 43.97 | 43.08 | 36.86 dB | 70.4 MB | 31.28 % |
| 4 096 | 45.66 | 46.08 | 44.99 | 43.99 | 37.94 dB | 84.6 MB | 37.61 % |
| 16 384 | 46.63 | 46.99 | 45.87 | 44.79 | 38.84 dB | 99.4 MB | 44.19 % |

**Decomposition loses decisively at block granularity.** Per-level
PSNRs are excellent (40–47 dB) but the joint vs T comes apart for
two reasons:
- **4× codebook + index overhead.** Each of the 4 levels carries its
  own 14.7 M-entry index stream → storage is 4× the single-atlas
  curve at matched K.
- **Quant errors add incoherently across levels.** Joint MSE ≈ Σ_k
  (per-level MSE), so joint PSNR ≈ per-level PSNR − 6 dB.

Matched-storage comparison vs single-atlas blocks:

| matched storage | single (§4.1) | decomposed (§4.2) | gap |
|---:|---:|---:|---:|
| ~13 % | 39.11 dB (K=65 536) | (need K≈4 at level ⇒ infeasible) | n/a |
| ~19 % | extrapolated 39.5+ dB | 34.06 dB (K=64) | **−5.5 dB** |
| ~44 % | extrapolated 40+ dB | 38.84 dB (K=16 384) | **−1.5+ dB** |

The decomposition is mathematically clean (Σ A_k = T at >98 dB after
finetune), but for **storage** there is nothing to harvest at the
block level: the 4 levels' blocks are not redundant enough to gain
from independent codebooks.

### 4.3 Shared codebook across all 4 decomposed levels

Hypothesis: per-level codebooks may waste capacity if many codewords
recur across levels. Test: one shared K-codeword codebook fit on a
balanced sample from all 4 levels; each level's 14.7 M blocks assign
against the shared codebook (4 × N index streams, but only 1 codebook).

Script: `scripts/bake_cluster_blocks_decomposed_joint.py`.

| K | L0 | L1 | L2 | L3 | joint vs T | total | vs single |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 38.56 | 42.23 | 41.23 | 40.78 | 33.53 dB | 42.2 MB | 18.75 % |
| 256 | 41.20 | 43.65 | 42.52 | 41.90 | 35.23 dB | 56.2 MB | 25.00 % |
| 1 024 | 43.04 | 44.89 | 43.65 | 42.89 | 36.54 dB | 70.3 MB | 31.26 % |
| 4 096 | 44.81 | 45.94 | 44.70 | 43.78 | 37.69 dB | 84.4 MB | 37.53 % |
| 16 384 | 46.00 | 46.84 | 45.58 | 44.55 | 38.61 dB | 98.7 MB | 43.86 % |
| 65 536 | 46.87 | 47.58 | 46.28 | 45.19 | **39.35 dB** | 113.5 MB | 50.44 % |

**Joint codebook is strictly worse than per-level at every K** by
0.2–0.5 dB at near-identical storage:

| K | per-level (§4.2) | joint (§4.3) | Δ |
|---:|---:|---:|---:|
| 64 | 34.06 | 33.53 | −0.53 |
| 256 | 35.64 | 35.23 | −0.41 |
| 1 024 | 36.86 | 36.54 | −0.32 |
| 4 096 | 37.94 | 37.69 | −0.25 |
| 16 384 | 38.84 | 38.61 | −0.23 |

**Why L0 pays the price**: shared codewords are pulled by the
3:1 numerical majority of L1/L2/L3 (smoother, lower magnitude). L0
PSNR drops 1.9 dB at K=64 (40.53 → 38.56) while L1–L3 lose < 0.1 dB.
The shared codebook starves L0's high-frequency content. Per-level
codebooks specialise correctly to each level's distribution.

Codebook bytes are rounding error at every meaningful K: at K=65 536
the shared codebook saves 3 MB vs 4 × 1 MB per-level, while the index
stream is 112 MB. The hoped-for compression gain has nowhere to land.

### 4.4 Per-surfel DC subtraction at block scale

Test: subtract each Gauss's patch mean from every one of its 4×4 blocks
before clustering, store the per-Gauss DC tables separately. (§3 saw
+0.5 dB at every ratio for whole-patch clustering with this trick.)

**Single-atlas, +DC table 0.22 MB (75 207 × 3 B):**

| K | no DC (§4.1) | with DC | Δ |
|---:|---:|---:|---:|
| 64 | 33.43 | 33.58 | +0.15 |
| 256 | 34.99 | 35.10 | +0.11 |
| 1 024 | 36.29 | 36.33 | +0.04 |
| 4 096 | 37.43 | 37.42 | −0.01 |
| 16 384 | 38.37 | 38.31 | −0.06 |
| 65 536 | 39.11 | 39.06 | −0.05 |

**Decomposed (per-level codebooks), +DC tables 0.86 MB (4 × 75 207 × 3 B):**

| K | no DC (§4.2) | with DC | Δ |
|---:|---:|---:|---:|
| 64 | 34.06 | 34.27 | +0.21 |
| 256 | 35.64 | 35.70 | +0.06 |
| 1 024 | 36.86 | 36.87 | +0.01 |
| 4 096 | 37.94 | 37.92 | −0.02 |
| 16 384 | 38.84 | 38.81 | −0.03 |

**Per-surfel DC is a whole-patch trick, not a block trick.** Tiny
+0.1–0.2 dB only at the smallest K, neutral or slightly negative for
K ≥ 4 096. At block granularity each codeword pools ~225 blocks
across many Gausses (at K = 65 k) → the codebook naturally clusters
by DC level without an explicit DC table. The DC tables cost more
than they save above K = 1 024.

**Skip the DC table for block-level clustering.** Keep it only for the
whole-patch path (§3 best operating point).

### 4.5 Residual VQ (L-stage nested K-means)

Same recurrence as Compact3DGS / RVQ-Gaussians, applied post-hoc to
the baked atlas blocks (no re-bake):

```
stage l (1 ≤ l ≤ L):
  R_l        = X − Σ_{j<l} cents_j[ass_j]
  cents_l, _ = KMeans(R_l, K_l)
  ass_l      = assign(R_l, cents_l)
recon        = Σ_l cents_l[ass_l]
```

Storage: Σ K_l × 48 bytes codebook + N × Σ ceil(log₂ K_l) bits indices.
Effective codebook = Π K_l, far above any single-stage K we can hold.

Script: `scripts/bake_cluster_blocks_residual.py`.

| config | L | eff K | codebook | idx | total | vs single | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | 2 | 65 536 | 8 KB | 28.1 MB | 28.1 MB | 12.50 % | 38.49 dB |
| 256 × 1 024 | 2 | 262 144 | 20 KB | 31.6 MB | 31.7 MB | 14.07 % | 39.32 dB |
| 64 × 16 384 | 2 | 1 048 576 | 257 KB | 35.2 MB | 35.4 MB | 15.74 % | 40.11 dB |
| 128 × 128 × 128 | 3 | 2 097 152 | 6 KB | 36.9 MB | 36.9 MB | 16.41 % | 39.71 dB |
| 256 × 256 × 256 | 3 | 16 777 216 | 12 KB | 42.2 MB | 42.2 MB | 18.76 % | 40.71 dB |
| 64⁴ | 4 | 16 777 216 | 4 KB | 42.2 MB | 42.2 MB | 18.75 % | 40.15 dB |
| 128⁴ | 4 | 268 435 456 | 8 KB | 49.2 MB | 49.2 MB | 21.88 % | 41.37 dB |
| **256⁴** | 4 | 4 294 967 296 | 16 KB | 56.2 MB | 56.3 MB | **25.01 %** | **42.49 dB** |
| 64⁵ | 5 | 1 073 741 824 | 5 KB | 52.7 MB | 52.7 MB | 23.44 % | 41.43 dB |

**RVQ crosses over single-stage VQ at ~18 % storage** and pulls ahead
increasingly above that:

| storage | single-stage best | RVQ best | Δ |
|---:|---:|---:|---:|
| ~13 % | 39.11 (K=65 k) | 38.49 (256²) | −0.62 |
| ~16 % | 40.17 (K=262 k) | 40.11 (64 × 16 384) | −0.06 |
| ~19 % | 40.60 (K=524 k) | **40.71** (256³) | **+0.11** |
| ~25 % | ~41.2 (extrap K=1 M) | **42.49** (256⁴) | **+1.3** |

**Codebook bytes are negligible at high L** — 256⁴ stores just 16 KB
of codebook vs 56 MB of indices. Each added stage costs ~14 MB of
index (one extra 8-bit lookup per block) for +0.5–1 dB. Effective
K=4.3 billion at L=4 is physically unreachable for single-stage
(codebook would be 196 GB).

**Comparison to the paper (Compact3DGS / RVQ-Gaussians):** they train
codebooks end-to-end during Gaussian fitting with a stop-gradient loss
`L_r = (1/NC) Σ_l ||sg[r_n − r̂_n^{l−1}] − Z^l[i_n^l]||²`. Joint
training typically adds **0.5–1 dB** vs greedy post-hoc K-means at
matched (L, C). Our numbers above are an *under*-estimate of what
RVQ-during-bake could achieve.

### 4.6 Conclusion

**Block clustering on the single-bake atlas is the clear winner.**
Single-atlas K=1 024 hits **36.29 dB at 7.82 %** (17.6 MB vs 225 MB
deployed) — a **13× smaller atlas** at a quality bar that whole-patch
clustering needed ~50 % storage to reach. K=16 384 gives 38.37 dB at
11 % (25 MB) for higher fidelity.

The decomposition path (§1 + §3 + §4.2) is preserved for completeness
and for the **DC-subtraction win** in §3 (free +0.5 dB at every ratio
by folding the per-Gauss DC into the SH baseline), but should not be
read as a storage-reduction direction in its own right.

---

## Open questions

- [ ] **Rendered-image PSNR**, not just atlas PSNR. Atlas-PSNR drop
  typically translates to ~30–50 % of that on rendered output.
- [ ] **Product quantisation on blocks** — split each 48-D block into
  M sub-vectors (e.g. 4 × 12-D channels-of-2×2-pixels or 3 × 16-D
  per-channel tiles), codebook each sub-vector. Potentially big wins
  vs the K=65 536 block result.
- [ ] **BC7-the-codebook** — fold the BC7 quant of the codebook itself
  into the PSNR (currently we compute MSE in float and only count BC7
  bytes for storage).
- [ ] **Entropy-code the index stream.** At K=1 024 the index stream
  is 14.7 M × 10 bits = 17.6 MB — by far the dominant cost. Spatially
  coherent blocks (≥ 4×4 neighbour reuse) should DEFLATE significantly.

---

## Files

- `scripts/bake_decompose_full.py` — full-N per-bucket decomposition +
  Adam joint-finetune, saves per-bucket per-level FP16 atlases.
- `scripts/bake_cluster_compare.py` — streamed 4-way whole-patch
  clustering comparison (single ± DC × decomposed ± DC).
- `scripts/bake_cluster_single_test.py` — single-bake atlas whole-patch
  K-means sweep with importance weighting + keep-top-X %.
- `scripts/bake_cluster_per_level.py` — non-uniform K per level for
  decomposed whole-patch clustering.
- **`scripts/bake_cluster_blocks.py`** — block-level (4×4) clustering
  on the single-bake atlas. *Current best storage line.*
- **`scripts/bake_cluster_blocks_decomposed.py`** — block-level
  clustering on each decomposed level with **per-level codebooks**.
- **`scripts/bake_cluster_blocks_decomposed_joint.py`** — same, but
  with a **single shared codebook** across all 4 levels (§4.3).
- **`scripts/bake_cluster_blocks_residual.py`** — L-stage residual VQ
  on the single-bake atlas blocks (§4.5).
- `outputs/mip_360/room/3D_SH_res/SV_30thr_005w25gLP4lev_FRP5k10_c2f_Jac/baked_atlas/`
  — reference bake.
- `outputs/.../baked_atlas/decomposed/bucket_<rx>×<ry>.pt` — saved
  per-bucket per-level decomposed atlases (FP16).
