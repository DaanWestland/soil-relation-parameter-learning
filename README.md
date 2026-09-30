# Soil relation parameter learning

**Predict rarely measured soil properties (total nitrogen, CEC at pH 7) by learning the bounded parameters of
known soil relations. A probabilistic, spatially validated benchmark with quantile regression forests, TabM and
the tabular foundation model TabICLv2, on public WoSIS data.**

*Keywords: digital soil mapping, pedometrics, pedotransfer functions, parameter learning, hybrid and
knowledge-guided machine learning, probabilistic prediction, CRPS, spatial cross-validation, sparse labels,
tabular foundation models (TabICL), WoSIS, cation exchange capacity, total nitrogen, C:N ratio.*

> **Status: a preliminary pilot on public data. Not peer-reviewed.** One region (conterminous USA), one depth
> (0-30 cm). The numbers in `reports/` show that the pipeline and the design work; they are not the results of a
> full study.

## 1. The idea

Total N and CEC7 are measured at far fewer places than clay, organic carbon (OC) and pH, and often
missing for whole regions. Pedology links them: N follows OC through the C:N ratio, and CEC7 follows clay and
OC, which carry the exchange sites. Instead of predicting N or CEC7 directly, a learner predicts the **one
bounded parameter of the relation**, and joint draws of clay, OC and pH are pushed through it:

    N    = OC / r                      r in [5, 60]      (r = C:N ratio)
    CEC7 = a * clay + 0.35 * OC        a in [0.02, 1.5]  (a = clay activity, cmolc per % clay)

(units: clay %, OC and N g/kg, CEC7 cmolc/kg). Every predicted soil then agrees with its own clay and OC, the
bounds stop the prediction from drifting where labels are missing, and any learner can be used, including a
foundation model in context. Calcareous soils, where the CEC7 relation does not hold, are handled by a
separate regime.

The pilot asks: (1) when N and CEC7 labels are scarce and missing by **region**, does learning the
parameter beat a model with the same inputs that predicts N or CEC7 directly ("chained"), so that any gain
is due to the formula and not to using predicted clay, OC and pH; (2) do the joint draws keep the relations
(for example the CEC7-clay rank correlation) that independent per-property models lose; (3) does this hold
across learner families? Labels of the sparse properties are thinned to 30, 10 and 3% by 300 km region (and
at random with the same counts) and everything is scored at spatially held-out sites (5 folds of 100 km
blocks) with fair CRPS, joint energy and variogram scores, and a rank-correlation check.

## 2. Headline numbers (main run `gpu_full`)

Copied from `reports/gpu_full/REPORT.md`, section 0. Points = difference in CRPS skill x 100 (skill = 1 - CRPS /
CRPS of the independent QRF at the same sites); 95% t-intervals over the five spatial folds. 3% of the sparse
labels kept.

| Learner | Formula only: parameter vs chained, removed by region | same, removed at random | same, all labels | Chaining alone: chained vs independent |
|---|---|---|---|---|
| QRF | +2.8 [+1.6, +4.1] | +1.7 [+0.7, +2.8] | -0.3 [-0.8, +0.2] | +10.9 [+9.9, +11.8] |
| TabM | +2.3 [+0.9, +3.7] | +0.3 [-0.8, +1.3] | -0.0 [-0.6, +0.6] | +17.7 [+13.3, +22.1] |
| TabICLv2 | +3.4 [-5.3, +12.2] | -0.4 [-0.8, +0.0] | -0.2 [-0.5, +0.1] | +9.9 [-0.7, +20.5] |

The gain from the formula is small, positive when labels are missing by region, and about zero with all labels.
The pre-specified primary contrast (parameter from covariates only vs chained, averaged over N, CEC7 and the
three learners) is +1.7 points with an interval that includes zero (section 1 of the report). The larger and
clearer effects are (a) using predicted clay, OC and pH at all (chaining), and (b) **coherence**: at held-out
sites the CEC7-clay Spearman correlation is 0.74 in the data, 0.01 for independent models, 0.44 (QRF) for
chained models and 0.62 (QRF) for the learned parameter; a fixed textbook parameter overshoots (0.85).
`docs/RESULTS.md` gives an overview of all runs and their limits.

## 3. What is in this repository

| Path | What |
|---|---|
| `src/pedopilot/` | the pipeline (Python package `pedopilot`; section 10 maps modules to steps) |
| `configs/` | one YAML file per run; `gpu_full.yaml` is the main result |
| `reports/` | published, analysed outputs of every run (`REPORT.md`, CSV tables, figures) |
| `docs/` | `DESIGN.md` (design, pre-specification, limits), `RESULTS.md`, `REFACTOR_NOTES.md` |
| `tests/` | pytest suite (unit, runtime, TabICLv2 code paths, interface contracts); GPU-free |
| `data_snapshot/` | how to get the pilot's data table (the table itself is not distributed) |
| `NOTICE`, `LICENSE`, `CITATION.cff` | data and model licences, code licence (MIT), how to cite |

## 4. Method at a glance

| Relation | Parameter (bounded) | Case weights of the parameter fit |
|---|---|---|
| N = OC / r | r = C:N in [5, 60] (fitted on the bounded-logit scale) | none (multiplicative: log scale) |
| CEC7 = a * clay + 0.35 * OC | a = clay activity in [0.02, 1.5] | clay (`cec_weight_power: 1.0`; pre-specified clay^2, see DESIGN.md) |

Units: clay %, OC and N g/kg, CEC7 cmolc/kg.

For each learner L in {`qrf`, `tabm`, `tabicl`, `tabicl_ft`}:

| Arm | What | Controls for |
|---|---|---|
| `L:free` | log target from covariates | current practice |
| `L:chained` | log target from covariates + observed clay/OC/pH; at test from the joint draws | any use of the abundant properties, without the formula |
| `L:structured` | bounded parameter from covariates, pushed through the relation with the draws | the knowledge (pre-specified primary contrast vs chained) |
| `L:structured_ya` | bounded parameter from covariates **and** clay/OC/pH, paired with the same draws as chained | the formula only: identical inputs to chained |
| `L:structured_og` | structured with the observed (oracle) calcareous regime (CEC7) | cost of predicting the regime |
| `L:oracle_ya` | structured parameter with the measured clay/OC | cost of uncertain abundant properties (upper bound) |
| `L:clip` | free draws clipped to the relation's envelope | feasibility only |
| `ptf` | constant (regime-specific weighted median) parameter | "a classic pedotransfer function is enough" |
| `ptf_marg` | parameter drawn from its training distribution, independent of x | "the parameter varies, but not with x" |

Glossary

| Term | Meaning |
|---|---|
| y_A | the abundant properties clay, OC and pH, drawn jointly from one multi-output forest at every test site |
| theta | the parameter of a relation (C:N ratio, clay activity), bounded to a plausible range |
| arm | one way of predicting a sparse target; every arm gives m draws per site |
| level | share of the training profiles with a sparse label that keep it (1.0, 0.3, 0.1, 0.03) |
| scheme | `all` (level 1.0), `group` (whole 300 km regions kept) or `random` (the same counts, at random) |
| rep | replicate of a thinning draw; random rep r is matched to group rep r |
| configuration | one (fold, level, scheme, rep): the unit of work, of resuming and of the result files |
| unit | fold x rep: skill is computed per unit and target, replicates are averaged within a fold |
| m | draws per site of every arm (100) |
| k_chain (K) | y_A draws a chained-type arm conditions on (m/K draws share one y_A draw) |
| CRN | common random numbers: every arm draws at the same uniforms per site |

Names in the accompanying manuscript: `free` = Independent, `structured_ya` = Hybrid, `structured` = Hybrid-x,
`ptf` = Fixed rule.

Skill = 1 - CRPS(arm) / CRPS(`qrf:free`) on the same held-out sites; a contrast "a vs b" is
skill(a) - skill(b) (> 0: a is better). Intervals are 95% t-intervals over the 5 spatial folds after
averaging replicates within a fold, plus a region (300 km) cluster bootstrap.

## 5. Installation

Python 3.10 or newer (tested with Python 3.12 on Windows 11). For the GPU runs install a CUDA build of
PyTorch **before** the package, otherwise pip may install a CPU build. `curl` must be on the PATH for
`download` (it is on Windows 10+, macOS and most Linux systems).

bash (Linux, macOS, Git Bash):

```bash
git clone https://github.com/DaanWestland/soil-relation-parameter-learning.git
cd soil-relation-parameter-learning
python -m venv .venv
source .venv/bin/activate          # Git Bash on Windows: source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU runs only (pick your CUDA version)
pip install -e ".[finetune,stats,dev]"
python -m pedopilot download --tabicl-only                              # TabICLv2 checkpoint, about 110 MB
python -m pedopilot check --cpu-ok
```

PowerShell (Windows):

```powershell
git clone https://github.com/DaanWestland/soil-relation-parameter-learning.git
cd soil-relation-parameter-learning
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1         # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[finetune,stats,dev]"
python -m pedopilot download --tabicl-only
python -m pedopilot check --cpu-ok
```

`check` lists the packages, whether torch sees a GPU, whether the data table is identical to the
pilot snapshot (if you have it) and whether the TabICLv2 checkpoint loads; it ends with `check: OK` once a data table is available (section 6; without one it says `data table: MISSING`). Drop `--cpu-ok` on a
GPU machine to make a missing GPU an error. Extras: `finetune` (transformers, for `tabicl_ft`), `stats`
(statsmodels, for the exploratory mixed model), `dev` (pytest).

The pinned versions `tabm==0.0.3` and `tabicl==2.2.0` are load-bearing: the wrappers in `learners.py`
use internals of both packages, and other versions may fail or behave differently.

## 6. Data

Every published run used one frozen table, `data_snapshot/conus_0_30.parquet` (28,192 profiles; sha256
`7a890cb6be3d858894c48cd48c9c28c45aedeec42f335a278c9d42a4473ee624`, checked at load and by `check`). It was
built by `python -m pedopilot download` and `python -m pedopilot prep` (`src/pedopilot/data.py`): WoSIS
"latest" layers (clay, organic carbon, pH in water, Kjeldahl N, CEC at pH 7, CaCO3 equivalent) restricted to the
CONUS box and mineral layers, thickness-weighted 0-30 cm means with at least 24 cm covered, plausibility
filters, a calcareous flag (0-30 cm mean CaCO3 equivalent >= 1 g/kg, i.e. 0.1%; WoSIS `tceq` is in g/kg;
missing = non-calcareous), EPSG:5070 ids of 100 km blocks and 300 km regions, and WorldClim 2.1 covariates (19
bioclimatic variables, elevation, slope and 5x5 relief) at the profile locations. Counts are in
`docs/DESIGN.md`, section Data.

**The table is not distributed in this repository** (WoSIS source datasets carry their own licences and
WorldClim 2.1 does not allow redistribution without permission; `NOTICE`, `data_snapshot/README.md`). You can
(a) rebuild a table with `download` + `prep` (needs `rasterio`; WoSIS "latest" is updated over time, so the
table differs somewhat from the pilot's and `check` says so), or (b) ask the author for the exact table and put
it in `data_snapshot/`. A rebuilt table is written to `data/conus_0_30.parquet` and is used instead of the
snapshot while it exists.

The TabICLv2 checkpoint is downloaded by `python -m pedopilot download --tabicl-only` into `models/`
(or by tabicl itself at first use). Licences of the data and weights: section 15 and `NOTICE`.

## 7. Quick start

```bash
pytest -q                              # under a minute; GPU-free, needs no data
python -m pedopilot validate           # synthetic positive/negative controls, QRF only: about 3 min on a CPU
# the following steps need a data table (section 6)
python -m pedopilot run smoke          # a few minutes on a CPU: QRF and a small TabM, one fold, a subsample
python -m pedopilot analyze smoke      # results/smoke/REPORT.md
```

`validate` prints `Overall: **PASS** (13/13 checks)` and exits with code 0; it is bit-reproducible. The
smoke run is not (it includes TabM, see section 13).

## 8. Reproducing the published runs

To reproduce the published numbers exactly you need the pilot's table (section 6; the QRF and rule arms are then
bit-reproducible). Every run is `python -m pedopilot run <run>`, then `python -m pedopilot analyze <run>` (writes
`results/<run>/`), then the region bootstrap. GPU times were measured on one Windows 11 PC with an
RTX 3080 (10 GB); CPU times are rough.

| Run | Config | Learners | Bootstrap | Hardware and time | Bit-reproducible | Published in |
|---|---|---|---|---|---|---|
| `gpu_full` (**main result**) | `configs/gpu_full.yaml` | qrf, tabm, tabicl, tabicl_ft (10%, 3%) | `bootstrap gpu_full --learner tabicl` | 1 CUDA GPU; 125 configurations, about 15 h on an RTX 3080 capped at 280 W | QRF and PTF arms: yes; TabM: no; TabICL: not verified | `reports/gpu_full` |
| `gpu_quick` | `configs/gpu_quick.yaml` | qrf, tabm, tabicl | `bootstrap gpu_quick --learner tabicl` | 1 CUDA GPU; 50 configurations, about 1.5 h | as gpu_full | `reports/gpu_quick` |
| `cpu_qrf_w1` | `configs/cpu_qrf_w1.yaml` | qrf | `bootstrap cpu_qrf_w1 --learner qrf` | CPU; 95 configurations, roughly 30-60 min | yes | `reports/cpu_qrf_w1` |
| `cpu_qrf` | `configs/cpu_qrf.yaml` | qrf, pre-specified clay^2 weights | `bootstrap cpu_qrf --learner qrf` | CPU, as above | yes | `reports/cpu_qrf` |
| `cpu_qrf_w1_s2` | `configs/cpu_qrf_w1_s2.yaml` | qrf, second fold layout (seed 2027) | `bootstrap cpu_qrf_w1_s2 --learner qrf` | CPU, as above | yes | `reports/cpu_qrf_w1_s2` |
| `validation` | built in (`validate.py`) | qrf and a small TabM on CPU | none | CPU; QRF alone about 3 min, much longer with TabM | QRF checks: yes; TabM checks: no | `reports/validation` |
| `cpu_qrf_oldfolds` | none | qrf | none | | **not reproducible** with this code | `reports/cpu_qrf_oldfolds` |

For example (the same commands in bash and PowerShell):

```bash
python -m pedopilot run gpu_full
python -m pedopilot analyze gpu_full
python -m pedopilot bootstrap gpu_full --learner tabicl
```

and for the synthetic validation: `python -m pedopilot validate --learners qrf,tabm --device cpu`
(writes `results/validation/`). This is expected to end with `Overall: **CHECK FAILED** (22/23 checks)`
and exit code 1, as the published validation did: the one miss is the small CPU TabM in the negative
control, and the TabM values differ from run to run (docs/DESIGN.md, "Validation on synthetic worlds").

Notes:

* Two more configs in `configs/` have no published results: `gpu_smoke` is an optional GPU pre-flight
  (every learner on the real table, one fold, 3 configurations, about 15-30 min on an RTX 3080) that
  catches install and memory problems before a long run; `gpu_ft` is an optional follow-up to
  `gpu_quick` (in-context vs fine-tuned TabICLv2 at 3% on the same folds, 20 configurations; needs the
  `finetune` extra). `configs/extra/` holds configs never used for a published run.
* `bootstrap NAME` computes structured_ya vs chained and structured vs chained at 3%, group thinning
  (`--level`, `--scheme` change that); `python -m pedopilot.bootstrap NAME` is the same command.
* To compare a rerun with the published copy, compare `results/<run>/` with `reports/<run>/` (for
  example `diff results/cpu_qrf_w1/contrasts.csv reports/cpu_qrf_w1/contrasts.csv` in bash, or
  `Compare-Object (Get-Content results\cpu_qrf_w1\contrasts.csv) (Get-Content reports\cpu_qrf_w1\contrasts.csv)`
  in PowerShell, which prints nothing when the files are identical). Do not run `publish` for this: it
  replaces `reports/<run>/`.
* The `bootstrap_*.json` files of the CPU runs were written by an earlier `bootstrap.py` without the
  `folds` field; a rerun adds it. `reports/cpu_qrf_oldfolds` also lacks `rank_correlation.csv` and
  `runtime.csv`.
* `cpu_qrf_oldfolds` used folds assigned in block order (before the hashed,
  stable folds) and clay^2 weights; its `config.yaml` says `name: cpu_qrf`. It is kept as a record of
  a third fold layout only.
* The raw per-site results (`results/<run>/`, for example 441 MB for `gpu_full`) are not in the
  repository; `analyze` and `bootstrap` need them.

**Which run is the main result.** `gpu_full`, by a rule fixed before any of its results at 3% existed:
report `gpu_full` if all 5 folds complete, whatever the direction of the results, otherwise
`gpu_quick` (docs/DESIGN.md, "Which run is the main result"). `docs/RESULTS.md` gives an overview.

## 9. Outputs

`run` writes to `results/<run>/`:

| File | Content |
|---|---|
| `meta_<tag>.json` | per configuration: sizes, realised label shares, learner diagnostics, seconds, binding shares; written LAST (completion marker) |
| `sites_<tag>.parquet` | per configuration, target, arm and test site: CRPS, CRPS of the log, PIT, 90% coverage, retained region, calcareous |
| `joint_<tag>.parquet` | per arm and site with both targets: energy and variogram scores of (clay, OC, pH, N, CEC7) |
| `rank_<tag>.parquet` | per target and arm: Spearman of CEC7-clay / N-OC (observed, joint draws, median map) |
| `config.yaml`, `code_version.txt`, `observed_violation.json` | the configuration, the pipeline commit per start, the observed envelope-violation rates |
| `errors.log` | tracebacks of configurations that failed (only if any failed) |

`<tag>` is `f{fold}_{scheme}_{level}_r{rep}`, e.g. `f0_group_0.03_r1`.

`analyze` adds: `summary_skill.csv`, `contrasts.csv`, `contrasts_log.csv`, `strata_skill.csv`,
`strata_contrasts.csv`, `did_group_vs_random.csv`, `joint_skill.csv`, `joint_contrasts.csv`,
`coverage90.csv`, `binding.csv`, `rank_correlation.csv`, `runtime.csv`, `primary.json`,
`mixed_model.json`, four figures and `REPORT.md` (the module docstring of `analysis.py` describes each
file). `bootstrap` adds `bootstrap_structured_ya_vs_chained.json` and `bootstrap_structured_vs_chained.json`.

REPORT.md sections:

| Section | Content |
|---|---|
| 0 | key numbers per learner (points = skill difference x 100) |
| 1 | the pre-specified primary contrast: structured vs chained at the lowest level, group thinning, averaged over QRF, TabM and TabICLv2 |
| 2 | every contrast per learner at the lowest level |
| 3 | structured vs chained at every level and scheme |
| 4 | formula only: structured_ya vs chained |
| 5 | thinned vs retained regions, and group minus random on thinned sites |
| 6 | joint energy and variogram scores |
| 7 | CRPS skill of every arm |
| 8 | 90% interval coverage |
| 9 | binding: draws outside the relation's envelope |
| 10 | exploratory mixed model |
| 11 | runtime and learner diagnostics |
| 12 | Figure-1 check: rank correlations of the joint draws |

`python -m pedopilot publish <run>` copies the reports (md, csv, json, png, yaml, log; not the raw
parquet and meta files) to `reports/<run>/`, replacing that folder.

## 10. Code structure

In pipeline order (`src/pedopilot/`):

| Module | Role |
|---|---|
| `paths.py` | where everything lives; `PEDOPILOT_ROOT` |
| `data.py` | download WoSIS and WorldClim, build (`prep`) and load the profile table |
| `relations.py` | the relations, bounded parameters, y_A transforms, envelopes and residuals |
| `sampling.py`, `thinning.py` | quantiles to draws with common random numbers; group and matched random label thinning |
| `design.py` | `Config`, spatial folds, configurations, file tags, arm naming (no torch import) |
| `learners.py` | QRF, TabM, TabICLv2 and fine-tuned TabICLv2 behind one `fit` / `sample` interface |
| `experiment.py` | the arms of one target and configuration, their scores and files; `run` |
| `scoring.py` | fair CRPS, energy and variogram scores (cluster-aware), PIT, coverage |
| `analysis.py`, `bootstrap.py` | skill, contrasts, intervals, figures, REPORT.md; the region bootstrap |
| `synthetic.py`, `validate.py` | synthetic worlds with a known answer and the checks on them |
| `cli.py` | the command line (`python -m pedopilot --help`) |

`analysis` and `bootstrap` do not import torch, so a finished run can be analysed without a
deep-learning stack.

## 11. Tests

`pytest -q` runs GPU-free in about a minute:

* `test_core.py`: relations and their inverses, transforms, quantile draws, common random numbers,
  thinning, fair and clustered scores, PIT and coverage (the parts where a silent bug would bias every
  result);
* `test_runtime.py`: out-of-memory fallbacks and chunked prediction, stable folds, the reuse of chained
  draws when k_chain = m, the resume guard, per-configuration error handling, the rank check and the
  bootstrap on small synthetic runs;
* `test_tabicl_paths.py`: every TabICLv2 code path (in context and fine-tuned) with a tiny randomly
  initialised checkpoint written by the test, so no GPU and no download are needed;
* `test_contracts.py`: names, signatures, file names, column and key orders and report sections that
  other tools rely on.

## 12. Compute

* **CPU or GPU.** QRF runs and the validation run on a CPU. TabM, TabICLv2 and fine-tuned TabICLv2 need
  a CUDA GPU (a config can set `allow_cpu: true` for small smoke tests). Measured model time per
  configuration in `gpu_full` (both targets, the four models of a learner; `reports/gpu_full/runtime.csv`):
  QRF 6-17 s, TabM 17-79 s, TabICLv2 150-240 s, fine-tuned TabICLv2 270-310 s.
* **GPUs with about 10 GB** (tested: RTX 3080). One fitted model is on the GPU at a time. TabICLv2
  predicts in row chunks that are halved on CUDA out of memory, and its KV cache falls back
  kv -> repr -> none; fine-tuning episodes are capped at 10k rows. On Windows, set *CUDA - Sysmem Fallback
  Policy* to *Prefer No Sysmem Fallback* in the NVIDIA Control Panel: silent spill-over into system memory
  is much slower than the handled out-of-memory fallbacks. `configs/extra/gpu_quick_lowmem.yaml` has
  smaller TabICLv2 memory settings if a GPU keeps running out of memory.
* **Resumable.** Stop a run at any time (Ctrl+C, crash, reboot) and run the same command again:
  finished configurations are skipped. A configuration that raises is logged to `errors.log`, the run
  continues, and the command exits with code 1: run it again to retry the failed ones. A results folder
  refuses a changed configuration (use a new `name:`). `analyze` works at any time; incomplete folds are
  left out and listed in the report header. Do not change the code during a run: `code_version.txt`
  records the commit and the run warns if it changes. `--folds 0,1` restricts a run to some folds.
* **Unattended runs with retries:**

  bash:
  ```bash
  for i in 1 2 3; do python -m pedopilot run gpu_full && break; done
  ```

  PowerShell:
  ```powershell
  for ($i = 0; $i -lt 3; $i++) { python -m pedopilot run gpu_full; if ($LASTEXITCODE -eq 0) { break } }
  ```

  `scripts/windows/run_with_retry.ps1` does the same as a separate process with a log file
  (`scripts/windows/README.md`).
* **Another root folder.** `PEDOPILOT_ROOT` moves `configs/`, `data/` (downloads and a locally rebuilt
  table, which is used instead of the snapshot while it exists), `data_snapshot/`, `results/` and
  `models/` to another folder (bash: `PEDOPILOT_ROOT=/data/pilot python -m pedopilot run smoke`;
  PowerShell: `$env:PEDOPILOT_ROOT = "D:\pilot"; python -m pedopilot run smoke`).

## 13. Determinism

Every random stream is seeded locally: site uniforms from (profile id, salt); thinning from a CRC32
hash of (scheme, fold, level, rep); each learner model from `fold * 1000 + rep * 10`; the abundant
forest and the regime gate from the fold. The config's `seed` only sets the fold layout (and
`subsample`). So the QRF arms of a run do not depend on which other learners are in the config.

* **Bit-identical** on a rerun: the QRF and PTF arms, the abundant model, the gate, thinning, folds,
  the analysis and the bootstrap (checked on one machine by repeated runs, and before and after the
  refactor in `docs/REFACTOR_NOTES.md`). Other versions of numpy or scikit-learn, or another platform,
  may change the forests slightly.
* **Not bit-identical:** TabM training, even on a CPU (`torch.manual_seed` is set, but no deterministic
  algorithms or thread settings are enforced). TabM arms, and any learner average that includes TabM
  (`primary.json`, REPORT.md section 1), vary slightly on a rerun.
* **Not verified:** TabICLv2 and fine-tuned TabICLv2 on a GPU.

## 14. Limitations

* WoSIS, not the KSSL lab database: mixed labs and methods (more noise), no raw exchangeable bases,
  so no base-saturation relation yet; 0-30 cm only; no horizon-level modelling.
* Groups are 300 km regions, not survey projects (WoSIS US data are one dataset, US-NCSS).
* Coarse covariates (WorldClim 5', no lithology or land cover yet).
* One fixed hyperparameter set per learner (no tuning); no power analysis yet.
* End-to-end fine-tuning of TabICL *through* the relations is not implemented.
* Cross-validation on a purposive sample compares models; it is not an unbiased map accuracy.
* The fold intervals are somewhat optimistic (the folds share training data); the region bootstrap
  keeps the fitted models fixed.

## 15. Licences

* **Code:** MIT (`LICENSE`).
* **Data:** the pilot table is not distributed. It is derived from WoSIS (ISRIC; each source dataset keeps its
  own licence, mostly CC BY or CC BY-NC; 27,878 of 28,192 profiles are from the US National Cooperative Soil
  Survey compilation) and WorldClim 2.1 (academic and non-commercial use; redistribution needs permission).
  `NOTICE` lists the details. If you rebuild the table, cite WoSIS and the original data providers and respect
  their terms.
* **Models:** TabICLv2 (Inria SODA): the `tabicl` code is BSD-3-Clause (package metadata), and the
  Hugging Face model repository `jingang/TabICL` carries the BSD-3-Clause tag (checked 2026-09-30); the
  checkpoint is downloaded, not included. TabM (Yandex Research) is used as the `tabm` package,
  Apache-2.0 (package metadata).

## 16. Citation

Please cite this software with `CITATION.cff` (GitHub's "Cite this repository" button) and the data and
methods it builds on. A manuscript describing the pilot is in preparation; this section will point to it.

* Batjes, N. H., Calisto, L. and de Sousa, L. M. (2024). Providing quality-assessed and standardised
  soil data to support global mapping and modelling (WoSIS snapshot 2023). *Earth System Science Data*
  16, 4735-4765.
* Fick, S. E. and Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for
  global land areas. *International Journal of Climatology* 37, 4302-4315.
* Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research* 7, 983-999.
* Gorishniy, Y., Kotelnikov, A. and Babenko, A. (2025). TabM: advancing tabular deep learning with
  parameter-efficient ensembling. *ICLR 2025*.
* Qu, J., Holzmüller, D., Varoquaux, G. and Le Morvan, M. (2025). TabICL: a tabular foundation model for
  in-context learning on large data. *ICML 2025* (TabICLv2 regressor checkpoint from the `tabicl`
  package, version 2.2.0).
* Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal
  Meteorological Society* 140, 1917-1923.
* Scheuerer, M. and Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic
  forecasts of multivariate quantities. *Monthly Weather Review* 143, 1321-1334.

## 17. Contact and contributions

Daan Westland. Questions, problems and requests for the exact pilot data table: please open an issue at
https://github.com/DaanWestland/soil-relation-parameter-learning/issues. Contributions are welcome (`CONTRIBUTING.md`).

---

Further documents: `docs/DESIGN.md` (the design, the deviation from the pre-specified design, and what
the pilot does not show), `docs/RESULTS.md` (overview of the published runs), `CHANGELOG.md` and
`docs/REFACTOR_NOTES.md` (what changed in the code and how equivalence was checked).
