# Pilot design

## Question

Does predicting sparse soil chemistry through the **bounded parameters of pedological relations**
help the sparse properties borrow strength from abundant ones, and is any gain due to the
**formula** (not just to using predicted clay/OC/pH), across learner families? And do the joint
draws keep the relations that independent models lose (the rank-correlation "Figure-1 check" below)?

| Relation | Parameter (bounded) | Case weights (two-stage) |
|---|---|---|
| N = OC / r | r = C:N in [5, 60] (fitted on bounded-logit scale) | none (multiplicative: log scale) |
| CEC7 = a * clay + 0.35 * OC | a = clay activity in [0.02, 1.5] | clay (from 28 Sep; pre-specified: clay^2, see below) |

Units: clay %, OC and N g/kg, CEC7 cmolc/kg; 0.35 per g/kg OC = 3.5 per % OC.

**Deviation from the pre-specified design (28 September, before any GPU run).** The CEC parameter fit
was pre-specified with clay^2 case weights, which make the two-stage fit match an end-to-end
*least-squares* loss. All learners here minimise a pinball (quantile) or CRPS-type loss, and those scale
linearly: an error e in the clay activity gives an error clay * e in CEC7, and CRPS(c F, c y) =
c CRPS(F, y). The matching weight is therefore clay, not clay^2. clay^2 also halves the effective
sample size of the CEC fit (6,204 vs 12,057 of 16,657 non-calcareous rows), which hurts most where
labels are scarce and for TabICL, whose context emulates weights by resampling. The change was made on
these grounds before comparing results; `cec_weight_power` in the configs records it (2 = pre-specified,
1 = used for the GPU runs), and the CPU run `cpu_qrf` (clay^2) is kept as a sensitivity check next to
`cpu_qrf_w1` (clay).

## Data

WoSIS "latest" (ISRIC) layers for the conterminous USA, aggregated to 0-30 cm per profile
(thickness-weighted, at least 80% coverage), mineral soils (OC < 12%): 28,192 profiles with clay,
OC and pH; 7,556 with total N; 19,217 with CEC7; 3,311 calcareous (0-30 cm mean CaCO3
equivalent >= 1 g/kg, i.e. 0.1%: WoSIS `tceq` is in g/kg; a missing carbonate value is read as
non-calcareous, following the NCSS practice of analysing carbonate only when the sample fizzes); 815 blocks of 100 km, 112 regions of 300 km. Covariates:
WorldClim 2.1 (19 bioclimatic variables, 5 arc-min), elevation, slope and 5x5 relief (2.5 arc-min).
The exact table is tracked in `data_snapshot/` (sha256 checked at load and by `check`).

## Design

* **Spatial folds**: 5 folds of whole 100 km blocks (EPSG:5070); the fold of a block is a hash of
  (seed, block id), so it does not change when blocks are added or removed.
* **Thinning** of the sparse labels in the training fold to 30 / 10 / 3%:
  * `group`: whole 300 km regions are kept (mimics data missing by survey / state);
  * `random`: the same number of profiles per label pattern (N only / CEC only / both), chosen at
    random: the matched-count control that separates "fewer labels" from "labels from other places".
    Random replicate r is scored against the regions of group replicate r, so the thinned /
    retained strata are the same sites in both schemes (difference-in-differences).
* **Shared abundant model**: one multi-output random forest over (logit clay, log OC, pH); draws
  are whole training rows chosen with QRF weights, so clay, OC and pH are drawn **jointly**.
  All arms use the same abundant draws, so arms differ only in the sparse part.
* **Regime gate** (CEC7 only): calcareous horizons are exempt from the CEC relation (NH4OAc
  artefacts). A classifier p(calcareous | x), fitted per configuration on the surviving CEC rows,
  sets the regime per draw; calcareous draws fall back to the chained model at the same y_A draw.
  `structured_og` uses the observed regime instead.
* **Residual**: log-scale residual of the observation around the bounded relation, drawn from the
  training residuals of the same clay (CEC7) or OC (N) decile, so the structured prediction keeps
  support at real observations.
* **Common random numbers**: every site uses the same uniforms in every arm.

## Arms

For each learner L in {QRF, TabM, TabICLv2 (in context), TabICLv2 fine-tuned}:

| Arm | What | Controls for |
|---|---|---|
| `L:free` | log target from covariates | current practice (SoilGrids design for QRF) |
| `L:chained` | log target from covariates + observed clay/OC/pH; at test from the joint draws | any use of the abundant properties, without the formula |
| `L:structured` | bounded parameter from covariates, pushed through the relation with the draws | the knowledge (pre-specified primary contrast vs chained) |
| `L:structured_ya` | bounded parameter from covariates **and** clay/OC/pH, paired with the same draws as chained | the formula only: identical inputs to chained |
| `L:structured_og` | structured with the observed (oracle) regime | cost of predicting the regime |
| `L:oracle_ya` | structured parameter with the measured clay/OC | cost of uncertain abundant properties (upper bound) |
| `L:clip` | free draws clipped to the envelope | feasibility only |
| `ptf` | constant (regime-specific weighted median) parameter | "a classic pedotransfer function is enough" |
| `ptf_marg` | parameter drawn from its training distribution, independent of x | "the parameter varies, but not with x" |

Reading: structured_ya > chained means the formula helps beyond a learned coupling with the same
information; structured > ptf / ptf_marg means learning theta(x) helps; clip matching structured
means only feasibility matters; gains concentrated in thinned regions mean extrapolation.

## Scoring

Every arm is represented by exactly m = 100 draws per site. Fair CRPS (Ferro 2014) per site, also
on the log scale; mid-PIT; 90% coverage; skill = 1 - CRPS(arm)/CRPS(free QRF) on the same sites.
Chained-type arms with k_chain < m generate their draws in groups that share one abundant draw;
the fair CRPS, energy and variogram scores then use only between-group pairs (otherwise the
within-group dependence biases the fair estimator). Joint: energy and variogram scores (p = 0.5)
of (logit clay, log OC, pH, log N, log CEC7), standardised with training-fold moments, at sites
where both sparse targets are observed. **Figure-1 check**: Spearman of (clay, CEC7) and (OC, N)
at held-out sites, observed vs pooled joint draws vs the per-site median "map".
Binding: share of chained draws outside the envelope of their own abundant draw, and of free draws
vs the observed clay/OC, next to the violation rate of the observed data.

Intervals: replicates are averaged within fold, then 95% t-intervals over the 5 folds; only
complete folds enter the report. Contrasts per target and for the mean of N and CEC7. Primary
(pre-specified in this document before the runs; not a public registration): structured vs chained at 3%, group thinning, averaged over QRF, TabM, TabICLv2;
also a mixed model (target as a sum-coded fixed effect, fold random intercept, replicate variance
component).

## Validation on synthetic worlds (known answer)

`python -m pedopilot validate` runs the full pipeline on three synthetic tables with the real
design (22 covariates, region-wise N missingness, 100 km blocks): *relation* (the relations hold up
to 5% noise; calcareous rows break the CEC relation) as a positive control, *independent* (N and
CEC driven by factors orthogonal to clay, OC and pH) as a negative control, and *shared* (N and CEC
share drivers with OC and clay, no exact relation) as a diagnostic. With QRF alone (the default,
`validate`) all 13 checks pass. The published validation (`reports/validation`, QRF and a small CPU
TabM: `validate --learners qrf,tabm --device cpu`) passes 22 of 23: the one miss is the small TabM in
the negative control, whose interval includes zero; its VALIDATION.md explains it.

## Learners

* **QRF**: random forest with QRF-weighted draws (Meinshausen 2006), case weights supported.
* **TabM** (Gorishniy et al., ICLR 2025): BatchEnsemble MLP (k = 32), periodic embeddings,
  monotone 99-quantile head (normal-spacing initialisation), case-weighted pinball loss, early
  stopping on held-out blocks.
* **TabICLv2** in context: no training; case weights emulated by systematic weight-proportional
  context resampling (ESS logged); 99 quantiles -> draws; chunked prediction.
* **TabICLv2 fine-tuned**: tabicl's `FinetunedTabICLRegressor` on distinct rows, early stopping on
  the pinball loss of held-out blocks, then the final context (all rows, with weights) on the
  fine-tuned weights; the fine-tuning gain is logged (measured on the same validation blocks that
  select the stopping epoch, so it is optimistic; the test-fold CRPS is the fair comparison).

## Runtime safety (GPU)

One fitted model at a time (GPU memory is freed between models); TabICL prediction in chunks that
halve on CUDA out-of-memory; KV cache falls back kv -> repr -> none; a failing configuration is
logged to `errors.log` and the run continues; runs are resumable and refuse to mix configurations.

## Which run is the main result

`gpu_full` (`reports/gpu_full`) is the main result: the full design (5 group and 3 random replicates,
full chain pairing k_chain = m = 100, 500 trees, and the fine-tuned TabICLv2 arm at 10% and 3%). The
rule was fixed on 28 September 2026, before any `gpu_full` result at 3% of the sparse labels existed:
`gpu_full` is reported if all 5 folds complete, whatever the direction of its results; otherwise
`gpu_quick`. No learner, setting or analysis was changed after seeing `gpu_quick` or `gpu_full`
results. `gpu_quick` and the CPU QRF runs (`cpu_qrf`, `cpu_qrf_w1`, `cpu_qrf_w1_s2`) are supporting
runs and sensitivity checks.

## What the pilot does NOT do (limits of the pilot)

* WoSIS, not the KSSL lab database: mixed labs and methods (more noise), no raw exchangeable
  bases, so **no base-saturation relation** yet; 0-30 cm only; no horizon-level modelling.
* Groups are 300 km regions, not survey projects (WoSIS US data are one dataset, US-NCSS).
* Coarse covariates (WorldClim 5', no lithology or land cover yet).
* One fixed hyperparameter set per learner (no tuning); no power analysis yet.
* End-to-end fine-tuning of TabICL *through* the relations is not implemented.
* Cross-validation on a purposive sample compares models; it is not an unbiased map accuracy.

Present results only as *"own preliminary pilot on public data"*: they show the pipeline and the
design work; they are not the study.
