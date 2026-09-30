# Contributing

Issues and pull requests are welcome, in particular: other regions or depths, other relations (for example a
base-saturation relation), other learners, and fixes to the documentation.

* Set up: `pip install -e ".[finetune,stats,dev]"`, then `pytest -q` (GPU-free, needs no data, under a minute).
* Keep results reproducible: a change that alters any published number must say so in `CHANGELOG.md`.
  `tests/test_contracts.py` guards the names, signatures and file layouts that other tools rely on.
* The pinned versions `tabm==0.0.3` and `tabicl==2.2.0` are load-bearing (the wrappers use their internals).
* Data: do not commit data tables (see `NOTICE` and `data_snapshot/README.md`).
* Questions: open an issue at https://github.com/DaanWestland/soil-relation-parameter-learning/issues.
