"""Pipeline validation on synthetic data with a known answer (positive and negative controls).

    python -m pedopilot validate                       # QRF only, a few minutes on a CPU
    python -m pedopilot validate --learners qrf,tabm,tabicl --device cuda    # on a CUDA GPU

Writes results/validation/VALIDATION.md with explicit pass/fail checks on three synthetic worlds
(synthetic.py) that share covariates, missingness and design with the real table:
  relation    (positive control): the structured arms must beat free and chained when the sparse
              labels are thinned, the formula-only contrast (structured_ya vs chained) must be positive,
              the oracle arm must bound the structured arm, and the formula must not hurt at 100%;
  shared      (diagnostic): no relation, but shared drivers; reports how much a structured target gains
              by borrowing strength alone (no pass/fail: this is what the formula-only contrast controls);
  independent (negative control): drivers orthogonal to clay/OC/pH; the formula must NOT beat chained.
Every run starts from scratch (the synthetic results folders are deleted first).
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from .analysis import analyze, load, unit_skill, contrast_table
from .experiment import Config, run
from .paths import RESULTS
from .synthetic import make_table

VALIDATION_DIR = RESULTS / "validation"
ROOT = VALIDATION_DIR                     # former name (not the repository root: that is paths.ROOT)
SCENARIOS = ("relation", "shared", "independent")


def _config(name, learners):
    """The validation design: 3 of 5 folds, levels 100/10/3%, one group and one random replicate."""
    return Config(name=name, n_folds=5, folds=[0, 1, 2], levels=[1.0, 0.1, 0.03], group_reps=1, random_reps=1,
                  m=100, k_chain=20, residual=True, gate=True, min_train=30, seed=7,
                  abundant={"n_estimators": 200}, gate_model={"n_estimators": 200}, learners=learners,
                  allow_cpu=True, cec_weight_power=1.0)


def _learners(names, device):
    """Learner settings for the validation (smaller than the GPU runs; a small TabM on a CPU)."""
    from .learners import resolve_device
    cpu = resolve_device(device) == "cpu"
    tabm = ({"k": 8, "d_block": 128, "n_blocks": 2, "max_epochs": 60, "patience": 8} if cpu      # CPU: small model, code-path check only
            else {"k": 16, "d_block": 256, "n_blocks": 2, "max_epochs": 120, "patience": 12})
    spec = {"qrf": {"n_estimators": 200},
            "tabm": {"device": device, **tabm},
            "tabicl": {"device": device, "n_estimators": 4, "max_ctx": 20000},
            "tabicl_ft": {"device": device, "epochs": 10, "patience": 4, "n_estimators_inference": 4,
                          "levels": [0.03]}}
    return {k: spec[k] for k in names}


def _get(con, learner, contrast, level, target="mean"):
    """(diff, lo, hi) of one contrast at one level (group thinning; scheme 'all' at 100%), NaN if absent."""
    scheme = "all" if level >= 1 else "group"
    r = con[(con.learner == learner) & (con.contrast == contrast) & (con.level == level) & (con.scheme == scheme)
            & (con.target == target)]
    return (float(r["diff"].iloc[0]), float(r["lo"].iloc[0]), float(r["hi"].iloc[0])) if len(r) else (np.nan,) * 3


def _relation_checks(con, sites, L):
    """Positive control (the relation holds): the formula must help when labels are thinned, and the oracle
    arm must bound the structured arm."""
    checks = []
    for contrast, lv, rule, thr in (("structured vs free", 0.03, ">", 0.0),
                                    ("structured vs chained", 0.03, ">", 0.0),
                                    ("structured_ya vs chained", 0.03, ">", 0.0),
                                    ("chained vs free", 0.03, ">", 0.0),
                                    ("structured vs chained", 1.0, ">", -0.02)):
        d = _get(con, L, contrast, lv)
        checks.append({"scenario": "positive (relation holds)", "learner": L, "check": f"{contrast} at {lv:.0%}",
                       "expect": f"{rule} {thr}", "value": d[0], "ci": f"[{d[1]:+.3f}, {d[2]:+.3f}]",
                       "pass": bool(d[0] > thr) if np.isfinite(d[0]) else False})
    sk = unit_skill(sites)
    g = sk[(sk.level == 0.03) & (sk.scheme == "group")]
    if f"{L}:oracle_ya" in g:
        o, s = g[f"{L}:oracle_ya"].mean(), g[f"{L}:structured"].mean()
        checks.append({"scenario": "positive (relation holds)", "learner": L,
                       "check": "oracle_ya skill >= structured skill at 3%", "expect": ">= 0",
                       "value": o - s, "ci": "", "pass": bool(o >= s - 1e-9)})
    return checks


def _independent_checks(con, L):
    """Negative control (drivers orthogonal to clay, OC and pH): the formula must not win."""
    checks = []
    for contrast, lv in (("structured_ya vs chained", 0.03), ("structured vs chained", 0.03),
                         ("structured vs chained", 0.1)):
        d = _get(con, L, contrast, lv)
        checks.append({"scenario": "negative (independent drivers)", "learner": L,
                       "check": f"{contrast} at {lv:.0%}", "expect": "<= 0.01", "value": d[0],
                       "ci": f"[{d[1]:+.3f}, {d[2]:+.3f}]",
                       "pass": bool(d[0] <= 0.01) if np.isfinite(d[0]) else False})
    return checks


def _shared_diagnostic(con, L):
    """Shared drivers, no relation: how much a structured target gains by borrowing alone (no pass/fail)."""
    d = _get(con, L, "structured vs chained", 0.03)
    return [{"scenario": "diagnostic (shared drivers, no relation)", "learner": L,
             "check": "structured vs chained at 3% (borrowing without a relation)",
             "expect": "report", "value": d[0], "ci": f"[{d[1]:+.3f}, {d[2]:+.3f}]", "pass": True}]


def _coverage_check(scen, sites):
    """Calibration of the reference arm: free QRF 90% intervals must cover 80-97% of the test sites."""
    cov = sites[(sites.arm == "qrf:free")]["cov90"].mean()
    return {"scenario": scen, "learner": "qrf", "check": "free QRF 90% coverage", "expect": "0.80-0.97",
            "value": cov, "ci": "", "pass": bool(0.80 <= cov <= 0.97)}


def _validation_markdown(ch, ok, learners, tables):
    """Lines of VALIDATION.md: overall verdict, the checks table, notes on failed checks, contrast tables."""
    lines = ["# Pipeline validation on synthetic data (known answer)", "",
             f"Overall: **{'PASS' if ok else 'CHECK FAILED'}** ({int(ch['pass'].sum())}/{len(ch)} checks)", "",
             "Three synthetic worlds with the real design (22 covariates, region-wise N missingness, spatial "
             "blocks). *relation* (positive control): N = OC / r(x) and CEC = a(x) clay + 0.35 OC hold up to "
             "5% lab noise; calcareous profiles break the CEC relation. *shared* (diagnostic): no relation, "
             "but N and CEC share drivers with OC and clay. *independent* (negative control): N and CEC depend "
             "on drivers orthogonal to those of clay, OC and pH. Contrasts are CRPS skill differences "
             "(mean of N and CEC; positive = first arm better) under group thinning (scheme 'all' at 100%), "
             "3 spatial folds, fold-level 95% t-intervals.", "",
             "Learner settings: " + "; ".join(f"{k} {v}" for k, v in learners.items()), "",
             ch.assign(value=ch.value.map(lambda v: f"{v:+.3f}" if pd.notna(v) else "")).to_markdown(index=False), ""]
    failed = ch[~ch["pass"].astype(bool)]
    if len(failed):                       # the pass rule stays as pre-specified; say what a failure means
        lines += ["**Checks that did not pass** (rule unchanged; interpretation added):", ""]
        for _, r in failed.iterrows():
            lo_hi = [float(x) for x in str(r.ci).strip("[]").split(",")] if str(r.ci).startswith("[") else None
            note = ("the interval includes zero, so there is no evidence of a spurious gain; with 3 folds and this "
                    "learner the check is too noisy to pass on the point estimate"
                    if lo_hi and lo_hi[0] <= 0 <= lo_hi[1] else "a real failure: investigate before relying on it")
            lines.append(f"- {r.learner}, {r.scenario}, {r.check}: {r.value:+.3f} {r.ci}: {note}.")
        lines.append("")
    for scen, con in tables.items():
        c = con[(con.scheme.isin(["group", "all"])) & (con.target == "mean")]
        lines += [f"## {scen}: skill differences (mean of N and CEC)", "",
                  c.pivot_table(index=["learner", "contrast"], columns="level", values="diff").round(3).to_markdown(), ""]
    return lines


def validate(learner_names=("qrf",), device: str = "auto", n: int = 6000) -> bool:
    """Run, analyse and check the three synthetic worlds; write validation_checks.csv, VALIDATION.md and
    validation.json to results/validation/. Returns True if every check passes."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    learners = _learners(learner_names, device)
    checks, tables = [], {}
    for scen in SCENARIOS:
        name = f"synthetic_{scen}"
        shutil.rmtree(VALIDATION_DIR / name, ignore_errors=True)            # never score stale results
        run(_config(name, learners), df=make_table(n=n, scenario=scen, seed=11), results_root=VALIDATION_DIR)
        analyze(name, results_root=VALIDATION_DIR)
        _, sites, _, _, _, _, _ = load(VALIDATION_DIR / name)
        con = contrast_table(unit_skill(sites), list(learners))
        tables[scen] = con
        for L in learners:
            if scen == "relation":
                checks += _relation_checks(con, sites, L)
            elif scen == "independent":
                checks += _independent_checks(con, L)
            else:                                           # shared drivers: diagnostic only, no pass/fail
                checks += _shared_diagnostic(con, L)
        checks.append(_coverage_check(scen, sites))
    ch = pd.DataFrame(checks)
    ch.to_csv(VALIDATION_DIR / "validation_checks.csv", index=False)
    ok = bool(ch["pass"].all())
    lines = _validation_markdown(ch, ok, learners, tables)
    (VALIDATION_DIR / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    (VALIDATION_DIR / "validation.json").write_text(json.dumps({"pass": ok, "n_checks": len(ch)}, indent=1))
    print("\n".join(lines[:8]))
    return ok
