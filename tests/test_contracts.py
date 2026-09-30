"""The external contract of the package: names, signatures, text and file layouts that code outside
src/pedopilot (figure scripts, report readers) and earlier results rely on. A refactor must keep every
one of these; a failure here means a silent break somewhere else, not a wrong number."""
import inspect
import json
import re
from dataclasses import fields
from pathlib import Path

import pandas as pd

from pedopilot import experiment
from pedopilot.experiment import Config, run
from pedopilot.synthetic import make_table

ROOT = Path(__file__).resolve().parents[1]


def test_public_names_are_importable():
    from pedopilot.data import load_table  # noqa: F401
    from pedopilot.experiment import (CONFIGS, LAST_FAILED, Config, _code_version, _salt,  # noqa: F401
                                      _spearman, assign_folds, config_tag, configurations, paired_ya_index,
                                      run, run_config, run_target)
    from pedopilot.learners import ForestSampler, RegimeClassifier, TabICLQuantile  # noqa: F401
    from pedopilot.paths import DATA  # noqa: F401
    from pedopilot.relations import (A_HI, A_LO, B, COVS, R_HI, R_LO, RELATIONS, blogit,  # noqa: F401
                                     inv_blogit, ya_inverse, ya_transform)
    from pedopilot.sampling import site_uniforms  # noqa: F401
    from pedopilot.thinning import group_thin, random_thin  # noqa: F401
    assert isinstance(TabICLQuantile.__dict__["_oom_ctx"], dict)            # class attribute, patched in tests
    assert isinstance(LAST_FAILED, list)


def test_signatures():
    assert list(inspect.signature(experiment.run_target).parameters) == [
        "target", "cfg", "tr", "te", "ya_te", "calc_draw", "fold", "rep", "level"]
    assert list(inspect.signature(experiment.run_config).parameters) == [
        "cfg", "fold", "level", "scheme", "rep", "shared", "out_dir"]


def test_config_fields_and_order():
    """config.yaml is dumped in field order and compared on resume: never add, rename or reorder."""
    assert [f.name for f in fields(Config)] == [
        "name", "n_folds", "folds", "levels", "group_reps", "random_reps", "m", "k_chain", "residual", "gate",
        "min_train", "seed", "abundant", "gate_model", "learners", "subsample", "allow_cpu", "cec_weight_power"]


def test_config_tag_format():
    assert experiment.config_tag(0, 0.03, "group", 2) == "f0_group_0.03_r2"
    assert experiment.config_tag(4, 1.0, "all", 0) == "f4_all_1_r0"


def test_relation_constants_parse_as_text():
    """Report builders read the rule bounds from relations.py as text (nothing is imported)."""
    from pedopilot import relations
    txt = (ROOT / "src" / "pedopilot" / "relations.py").read_text(encoding="utf-8")
    assert float(re.search(r"^B\s*=\s*([\d.]+)", txt, re.M).group(1)) == relations.B
    m = re.search(r"^R_LO,\s*R_HI\s*=\s*([\d.]+),\s*([\d.]+)", txt, re.M)
    assert (float(m.group(1)), float(m.group(2))) == (relations.R_LO, relations.R_HI)
    m = re.search(r"^A_LO,\s*A_HI\s*=\s*([\d.]+),\s*([\d.]+)", txt, re.M)
    assert (float(m.group(1)), float(m.group(2))) == (relations.A_LO, relations.A_HI)


def test_design_doc_data_counts():
    """The data counts are read from docs/DESIGN.md by report builders (first match wins)."""
    txt = (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    want = {r"([\d,]+) profiles with clay": 28192, r"([\d,]+) with total N": 7556, r"([\d,]+) with CEC7": 19217,
            r"([\d,]+) calcareous": 3311, r"([\d,]+) blocks of 100 km": 815, r"([\d,]+) regions of 300 km": 112}
    for pat, n in want.items():
        assert int(re.search(pat, txt).group(1).replace(",", "")) == n, pat


def _tiny(name, **kw):
    base = dict(name=name, n_folds=3, folds=[0], levels=[1.0, 0.3], group_reps=1, random_reps=1, m=20,
                k_chain=20, min_train=30, abundant={"n_estimators": 20}, gate_model={"n_estimators": 20},
                learners={"qrf": {"n_estimators": 20}})
    base.update(kw)
    return Config(**base)


def test_output_layout(tmp_path):
    """Arm order (row order of sites_/rank_ parquet), meta key order (column order of binding.csv), file
    names and REPORT.md section numbers 0-12."""
    from pedopilot.analysis import analyze
    out = run(_tiny("layout"), df=make_table(n=1200, scenario="relation", seed=4), results_root=tmp_path)
    tag = "f0_group_0.3_r0"
    for prefix in ("meta_", "sites_", "joint_", "rank_"):
        assert (out / f"{prefix}{tag}{'.json' if prefix == 'meta_' else '.parquet'}").exists(), prefix
    s = pd.read_parquet(out / f"sites_{tag}.parquet")
    tail = ["qrf:free", "qrf:chained", "qrf:structured", "qrf:structured_ya", "qrf:oracle_ya", "qrf:clip"]
    assert list(pd.unique(s[s.target == "n"].arm)) == ["ptf", "ptf_marg", *tail]
    assert list(pd.unique(s[s.target == "cec"].arm)) == ["ptf", "ptf_marg", "qrf:structured_og", *tail]
    assert list(s.columns) == ["fold", "level", "scheme", "rep", "target", "arm", "site", "block", "crps", "crps_log",
                               "pit", "cov90", "retained", "calc"]
    meta = json.loads((out / f"meta_{tag}.json").read_text())
    keys = [k for k in meta if k.startswith("cec:")]
    assert keys == ["cec:n_train", "cec:n_theta_fit", "cec:theta_outside_bounds", "cec:residual_nonzero",
                    "cec:qrf_ess", "cec:qrf_epochs", "cec:qrf_ft_gain", "cec:qrf_seconds",
                    "cec:binding_obs[qrf:free]", "cec:binding[qrf:chained]", "cec:binding[qrf:structured_ya]",
                    "cec:frac_kept"]
    assert list(meta)[:9] == ["fold", "level", "scheme", "rep", "n_kept_profiles", "n_kept_pattern1",
                              "n_kept_pattern2", "n_kept_pattern3", "calc_draw_share"]
    analyze("layout", results_root=tmp_path)
    for f in ("summary_skill.csv", "contrasts.csv", "contrasts_log.csv", "strata_skill.csv", "strata_contrasts.csv",
              "did_group_vs_random.csv", "joint_skill.csv", "joint_contrasts.csv", "coverage90.csv", "binding.csv",
              "rank_correlation.csv", "runtime.csv", "primary.json", "mixed_model.json", "REPORT.md",
              "skill_vs_thinning.png", "contrast_structured_vs_chained.png", "contrast_structured_ya_vs_chained.png",
              "pit.png"):
        assert (out / f).exists(), f
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert [int(n) for n in re.findall(r"^## (\d+)\. ", report, re.M)] == list(range(13))
    assert re.search(r"complete folds used: \[0\]; incomplete folds excluded: \[\]", report)
    assert re.search(r"sites scored: \d+; configurations: 3; learners: \['qrf'\]", report)
