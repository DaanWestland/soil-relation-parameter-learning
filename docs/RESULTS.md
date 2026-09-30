# Results overview

Own preliminary pilot on public WoSIS data (conterminous USA, 0-30 cm; 28,192 profiles, N at 7,556,
CEC7 at 19,217). Not peer-reviewed. "Points" = difference in CRPS skill x 100 (skill = 1 - CRPS / CRPS
of the independent QRF on the same held-out sites). Intervals: fold-level 95% t-intervals unless stated.
Every number below is copied from a published report in `reports/`; the reports are the reference.

## Main result: `gpu_full`

The main run is `gpu_full`: QRF, TabM, TabICLv2 in context and fine-tuned TabICLv2, 5 spatial folds,
125 configurations (why this run: docs/DESIGN.md, "Which run is the main result"). Read, in
`reports/gpu_full/REPORT.md`:

* section 0: the headline numbers per learner (formula-only contrast, chaining alone, Figure-1 check,
  joint scores);
* section 1: the pre-specified primary contrast (structured vs chained at 3% of the sparse labels,
  removed by region, averaged over QRF, TabM and TabICLv2);
* `bootstrap_structured_ya_vs_chained.json` and `bootstrap_structured_vs_chained.json`: the region
  (300 km) cluster bootstrap for TabICLv2.

## Supporting runs

| Run | What it adds | Report |
|---|---|---|
| `gpu_quick` | first GPU run: QRF, TabM, TabICLv2; fewer replicates, k_chain = 20 | `reports/gpu_quick` |
| `cpu_qrf_w1` | QRF only, clay weights (the design of the GPU runs) | `reports/cpu_qrf_w1` |
| `cpu_qrf` | QRF only, clay^2 weights (the pre-specified design) | `reports/cpu_qrf` |
| `cpu_qrf_w1_s2` | `cpu_qrf_w1` with a second fold layout (seed 2027) | `reports/cpu_qrf_w1_s2` |
| `cpu_qrf_oldfolds` | an earlier block-order fold layout (clay^2; older code, not reproducible) | `reports/cpu_qrf_oldfolds` |
| `validation` | synthetic worlds with a known answer (QRF and a small CPU TabM) | `reports/validation` |

## CEC weighting: clay vs clay^2 (QRF, 3% of sparse labels kept, removed by region)

| Quantity | clay^2 weights (pre-specified, `cpu_qrf`) | clay weights (GPU design, `cpu_qrf_w1`) |
|---|---|---|
| Formula only (structured_ya vs chained), mean of N and CEC7 | +2.7 [+1.2, +4.1] | **+2.7 [+1.4, +4.0]** |
| ... region bootstrap (109 regions) | +2.5 [+1.0, +4.1] | +2.5 [+1.0, +4.1] |
| ... N / CEC7 | +2.9 [+0.8, +5.0] / +2.4 [+0.4, +4.4] | +2.9 [+0.8, +5.0] / +2.4 [+0.5, +4.4] |
| ... random thinning (same counts) | +1.5 [+0.5, +2.5] | +1.6 [+0.7, +2.6] |
| ... with all labels | -0.5 [-0.8, -0.2] | -0.3 [-0.5, -0.1] |
| Pre-specified main contrast (parameter from x only) | +1.7 [-0.2, +3.6] | +1.7 [-0.1, +3.5] |
| Chaining alone (chained vs independent) | +11.2 [+9.2, +13.2] | +11.2 [+9.2, +13.2] |
| CEC7-clay Spearman at held-out sites | observed 0.74; independent 0.02; chained 0.47; learned 0.64; fixed PTF 0.85 | learned 0.63 (others identical) |
| Joint variogram-score skill | learned 0.245 vs chained 0.188 | learned 0.243 vs chained 0.188 |

The result does not depend on the CEC weighting.

**Fold-layout robustness.** A second hashed layout with the final design (`reports/cpu_qrf_w1_s2`,
only the fold seed changes) gives a formula-only gain of +3.7 [+1.1, +6.2] (random thinning +1.3, all
labels -0.3; CEC7-clay Spearman independent 0.01, learned 0.66, observed 0.75). An earlier layout
(block order instead of hashed blocks, `reports/cpu_qrf_oldfolds`, clay^2) gave +4.0 [+2.5, +5.4].
So across three layouts the gain is +2.7 to +4.0 points: always positive, size depending on which
regions end up in which fold; the coherence result is the same in every layout.

## How to read the CPU runs

* **The formula helps where labels are missing by region**, by a few points of CRPS skill on top of a
  strong chained baseline that already uses clay, carbon and pH (which itself is worth about 11 points).
  It helps less when the same number of labels is removed at random (+1.6), and costs about 0.3
  points with all labels. That points to an extrapolation gain; on the same thinned sites the
  group-minus-random difference is positive for both targets (clay weights: N +1.3, CEC7 +1.0) but
  its interval still includes zero at 3%; at 10% it excludes zero for N (+1.0 [+0.4, +1.5]) and just
  touches it for CEC7 (+1.5 [-0.1, +3.0]).
* **Coherence is the larger effect**: independent draws lose the CEC-clay rank correlation almost
  completely (0.02 vs 0.74 observed), the chained model keeps part of it (0.47), learned parameters
  keep most of it (0.63), and a fixed pedotransfer function exaggerates it (0.85).
* **The synthetic controls mostly behave**: the formula wins when the relation holds (QRF and TabM);
  with unrelated drivers QRF loses and the small CPU TabM is inconclusive (+0.020 [-0.285, +0.325]);
  22/23 checks pass (reports/validation).

## Caveats

* One region (CONUS), one depth (0-30 cm), WoSIS (mostly NCSS lab data), coarse covariates.
* Fold intervals share training data; the region bootstrap agrees for this contrast, but it keeps
  the fitted models fixed, so it does not cover that dependence (several fold layouts would).
* The level is a share of profiles with any sparse label; N keeps about 4% of its labels at "3%".
* CEC weights: clay^2 was pre-specified; clay is the CRPS-consistent choice (docs/DESIGN.md).
