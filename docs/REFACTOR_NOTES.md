# Refactor for sharing (2026-09-30): what changed and how equivalence was verified

Starting point: version 0.2.0 (development commit 0fb7318; the public history starts at 0.3.0). Goal: code, documentation and layout that other researchers can read and
reproduce, **without changing any result**.

## 1. What changed

Code (`src/pedopilot/`):

* `experiment.py`: `run_target` split into `_theta_targets`, `_residual_sampler`, `_ptf_arms`,
  `_learner_params`, `_chain_inputs`, `_fit_free`, `_fit_chained` (with `_chained_at` and a
  `_GateFallback` record), `_fit_theta`, `_fit_theta_ya`, `_combine_arms` and `_binding_meta`;
  `run_config` into `_thinning_masks`, `_regime_draws`, `_site_score_frame`, `_joint_scores`,
  `_rank_rows`, `_config_meta` and `_write_config_outputs`; `run` into `_prepare_table`,
  `_print_design`, `_record_code_version`, `_observed_violation`, `_fold_inputs` and `_run_one_config`.
  Named constants `N_RESIDUAL_BINS`, `MIN_CALC_FOR_PTF` and `learner_seed()`; descriptive local names
  (`u_draw`, `u_chain`, `ya_idx_chained`, `theta_draws`, ...). Module docstring: glossary, files per
  configuration, seeding scheme, reproducibility, resuming.
* `design.py` (new, no torch import): `Config`, `_salt`, `assign_folds`, `configurations`, `config_tag`,
  `paired_ya_index`, `LEARNER_ORDER`, `PAIRED_ARMS`/`is_paired`, `grid_id`. All re-exported from
  `experiment`.
* `analysis.py`: `analyze` split into `_compute_tables` (`AnalysisTables`), `_write_tables`,
  `_write_figures` and `_report_lines`; `load` returns the `LoadedResults` NamedTuple (still unpacks as
  the old 7-tuple); `_fold_complete`, `_target_series`, `_concat_or_empty` remove duplicated code; the
  module docstring lists every output file and REPORT.md section. No torch import any more.
* `validate.py`: `validate` split into `_relation_checks`, `_independent_checks`, `_shared_diagnostic`,
  `_coverage_check` and `_validation_markdown`; `ROOT` renamed `VALIDATION_DIR` (alias kept).
* `learners.py`: learner table and the pinned-internals warning in the module docstring; one
  `free_memory()` (old names `_free_cuda` and `experiment._release` kept as aliases); a `_LeafIndex`
  NamedTuple for the per-tree forest index; `TabICLQuantile._oom_ctx` moved to the top of the class.
* `bootstrap.py`: `bootstrap_contrasts()` and `main()`; `REF` imported from `analysis`.
* `data.py`: the preparation chain in the module docstring; `snapshot_status()` (used by `load_table`
  and `check`); grid ids from `design.grid_id`.
* `cli.py`: help texts, epilog, `bootstrap` subcommand, corrected command list.
* Everywhere: docstrings for every public function, `from __future__ import annotations` and type hints
  on public signatures, neutral wording in comments and messages (no machine- or event-specific notes).
  Version string aligned to 0.3.0.

Layout and documents: PowerShell helpers live in `scripts/windows/`, configurations never used for a
published run in `configs/extra/`; new `README.md`, `docs/RESULTS.md`, `CHANGELOG.md`, `LICENSE`,
`CITATION.cff`, `NOTICE` and `tests/test_contracts.py`; a "Which run is the main result" section in
`docs/DESIGN.md`. The calcareous threshold is documented in its real unit (CaCO3 equivalent >= 1 g/kg, i.e.
0.1%; WoSIS `tceq` is in g/kg): the earlier text said 1%. The value (`CALC_THRESHOLD = 1.0`) and the data
are unchanged. `analysis` passes `axis=` by keyword (positional axes are deprecated in pandas).

## 2. External contract kept

Kept name- and byte-identical: every name imported by code outside the package (`experiment.Config`,
`_salt`, `_spearman`, `_code_version`, `assign_folds`, `paired_ya_index`, `run_target`,
`learners.ForestSampler`, `RegimeClassifier`, `relations.*`, `sampling.site_uniforms`,
`thinning.group_thin`/`random_thin`, `data.load_table`, `paths.DATA`); the signature of `run_target`
and the order of its arms and meta keys; the `Config` fields and their order; the three constant lines
of `relations.py` that are read as text; the data counts in `docs/DESIGN.md`; the `config_tag` format;
all file names, CSV/JSON columns and REPORT.md sections 0-12; `python -m pedopilot.bootstrap`;
`experiment.run_config` as a module global and `experiment.LAST_FAILED` as the same list object.
`tests/test_contracts.py` now checks these.

## 3. How equivalence was verified

1. **Regression harness** (outside the repository): with the current code, in a scratch
   `PEDOPILOT_ROOT`, on CPU:
   * `regress` = `configs/smoke.yaml` without TabM (QRF, 1 fold, subsample 0.35, k_chain 20);
   * `regress_k100` (added first, before any code change) = QRF with k_chain = m = 100 (the `gpu_full`
     chain pairing, which takes the reuse branch of the gated fallback), `cec_weight_power: 1.0` and a
     per-learner `levels:` filter (QRF skips level 0.3), as used by `tabicl_ft`;
   * `analyze` of both, `analyze` of the real `gpu_quick` raw results, and the region bootstrap of
     `gpu_quick` (`python -m pedopilot.bootstrap gpu_quick --learner tabicl`).

   Every CSV, parquet, JSON, YAML and markdown file is compared with the baseline made from commit
   0fb7318, ignoring only timing fields, the runtime section of REPORT.md, `code_version.txt`,
   `errors.log` and PNG bytes. The baselines are deterministic (two runs of the old code identical: 248
   and 285 files). Result after each step of the refactor (docs and wording; duplicated code, CLI and
   type hints; the `run_target`/`run_config`/`run` split; `design.py`, `analyze` and `validate` split;
   final): **IDENTICAL**, 248 files against the original baseline and 285 against the widened one,
   every time.
2. **Synthetic validation** (`python -m pedopilot validate`, QRF): all 234 output files, including
   `validation_checks.csv`, `VALIDATION.md` and `validation.json`, identical to the output of the old
   code (13/13 checks pass), after the `validate` split and at the end. The old code gave identical
   output on two runs.
3. **Tests**: the 35 existing tests pass after every step; with the 7 new contract tests, 42 passed in the development repository; this repository has 41 tests (40 pass, one is skipped outside a git checkout).
4. **Other checks**: the scripts outside the package that import pedopilot still import; their text
   parsers still read the constants of `relations.py` and the counts in `docs/DESIGN.md`; `grid_id`
   reproduces the `block` and `region` columns of the data snapshot for all 28,192 profiles (so the
   `data.prep` call site is equivalent, although `prep` itself cannot be rerun: WoSIS "latest" has
   changed); `analysis` and `bootstrap` import without torch; a tiny QRF + CPU TabM run with a level
   filter goes through `run`, `analyze` and `bootstrap` (TabM numbers are not reproducible, so this is a
   code-path check only).

Not covered: GPU runs of TabM, TabICLv2 and fine-tuned TabICLv2 (the experiment code they go through is
the same as QRF's; the TabICL wrappers are covered by the tiny-checkpoint tests); figure bytes (the
figure code is unchanged); `download`, `prep` and `publish` (not run: `publish` replaces `reports/<run>`).

## 4. Release

The public repository starts at version 0.3.0 with a fresh history (the development history of 0.2.0 and
earlier is not public). Before reuse: the licence of the code is MIT (`LICENSE`); the data table is not
distributed (`data_snapshot/README.md`, `NOTICE`); the raw per-site results of `gpu_full` (441 MB) are not in
the repository and could be archived on Zenodo so that others can rerun `analyze` and the bootstrap without
GPUs.

## 5. Follow-ups kept out of this refactor (they would change behaviour)

* `publish` skips `.txt` files, so `code_version.txt` never reaches `reports/`; add it for provenance.
* Learner constructors silently ignore unknown settings (`**_`): warn on them.
* Move `rasterio` to a `prep` extra (only `prep` needs it); `check` would then treat it as optional.
* TabM determinism (deterministic algorithms, thread settings): would change TabM numbers.
* A `model` property alias for the TabICL wrappers' `.m` (confusable with `cfg.m`): skipped, low value.
