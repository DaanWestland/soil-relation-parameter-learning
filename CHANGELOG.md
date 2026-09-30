# Changelog

## 0.3.0 (2026-09-30): first public version

The code of the pilot (version 0.2.0, which produced the published results) refactored for sharing.
**Results unchanged.** A QRF end-to-end run (also with k_chain = m and a per-learner level filter), the
re-analysis of the real `gpu_quick` results, the region bootstrap and the synthetic validation are
byte-identical before and after (ignoring timings); see `docs/REFACTOR_NOTES.md`.

* **Readable code.** `run_target`, `run_config`, `run`, `analyze` and `validate` split into named helpers;
  named constants for magic numbers; module, class and function docstrings with a glossary, the seeding
  scheme and the files each step writes; type hints on public signatures; duplicated code merged.
* **New `design.py`** (no torch import): `Config`, folds, configurations, tags and arm naming; still
  importable from `experiment`. `analysis` and `bootstrap` no longer import torch.
* **CLI.** Help for every command and option, an epilog with the reproduce commands, and a
  `pedopilot bootstrap` subcommand (same as `python -m pedopilot.bootstrap`).
* **Documentation.** `README.md` (installation, data, reproducing every published run, outputs, compute,
  determinism, licences, citation), `docs/RESULTS.md`, `docs/DESIGN.md` (which run is the main result),
  `LICENSE` (MIT), `CITATION.cff`, `NOTICE` (data and model licences).
* **Correction.** The calcareous threshold is documented in its real unit: CaCO3 equivalent >= 1 g/kg
  (0.1%) in the 0-30 cm mean (WoSIS `tceq` is in g/kg); earlier text said 1%. The value
  (`CALC_THRESHOLD = 1.0`) and the data are unchanged.
* **Tests.** `tests/test_contracts.py` guards the names, signatures, file names, column and key orders and
  report sections other tools rely on.
* **Data.** The frozen profile table of the pilot is not part of this repository (data licences); see
  `data_snapshot/README.md`.
