"""Runtime safety of the GPU run: out-of-memory fallbacks, chunked prediction, stable folds, the reuse of
chained draws when k_chain == m, the resume guard and per-configuration error handling."""
import numpy as np
import pandas as pd
import pytest
import torch

from pedopilot import experiment
from pedopilot.experiment import Config, _check_resume, assign_folds, run
from pedopilot.learners import ForestSampler, _q_chunked
from pedopilot.sampling import LEVELS
from pedopilot.synthetic import make_table


def _fake_predict(limit):
    calls = []

    def predict(Z, output_type=None, alphas=None):
        calls.append(len(Z))
        if len(Z) > limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return np.repeat(Z[:, :1], len(alphas), 1) + np.asarray(alphas)[None, :]
    return predict, calls


def test_chunked_prediction_halves_on_oom_and_matches():
    X = np.random.default_rng(0).normal(size=(1000, 3))
    predict, calls = _fake_predict(limit=300)
    q = _q_chunked(predict, X, chunk=1024)
    ref = np.repeat(X[:, :1].astype(np.float32), len(LEVELS), 1) + LEVELS[None, :]
    assert np.allclose(q, ref) and q.shape == (1000, len(LEVELS))
    assert max(c for c in calls if c <= 300) <= 256          # 1024 -> 512 -> 256


def test_chunked_prediction_reraises_other_errors():
    def bad(Z, **_):
        raise RuntimeError("shape mismatch")
    with pytest.raises(RuntimeError, match="shape"):
        _q_chunked(bad, np.zeros((10, 2)))


def test_tabicl_kv_cache_falls_back_on_oom(monkeypatch):
    pytest.importorskip("tabicl")
    from pedopilot.learners import TabICLQuantile

    class FakeReg:
        def __init__(self):
            self.kv_cache, self.modes = True, []

        def set_params(self, **kw):
            self.__dict__.update(kw)

        def fit(self, X, y):
            self.modes.append(self.kv_cache)
            if self.kv_cache is True:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return self

    monkeypatch.setattr(TabICLQuantile, "_oom_ctx", {})
    m = TabICLQuantile.__new__(TabICLQuantile)
    m.seed, m.max_ctx, m.pred_chunk, m.m = 0, 1000, 4096, FakeReg()
    m.fit(np.zeros((20, 3)), np.arange(20.0))
    assert m.m.modes == [True, "repr"] and m.kv_mode_ == "repr"
    m2 = TabICLQuantile.__new__(TabICLQuantile)                        # a larger context: skip the kv attempt
    m2.seed, m2.max_ctx, m2.pred_chunk, m2.m = 0, 1000, 4096, FakeReg()
    m2.fit(np.zeros((30, 3)), np.arange(30.0))
    assert m2.m.modes == ["repr"]
    m3 = TabICLQuantile.__new__(TabICLQuantile)                        # a smaller context: kv is tried again
    m3.seed, m3.max_ctx, m3.pred_chunk, m3.m = 0, 1000, 4096, FakeReg()
    m3.fit(np.zeros((10, 3)), np.arange(10.0))
    assert m3.m.modes == [True, "repr"]


def test_folds_are_stable_when_blocks_are_added():
    df = pd.DataFrame({"block": [f"{i}_{j}" for i in range(20) for j in range(20)]})
    f_all = pd.Series(assign_folds(df, 5, 2026), index=df.block)
    sub = df.sample(frac=0.5, random_state=1)
    f_sub = pd.Series(assign_folds(sub, 5, 2026), index=sub.block)
    assert (f_all.loc[f_sub.index].values == f_sub.values).all()
    assert set(f_all.unique()) == set(range(5)) and f_all.value_counts().min() > 40


def test_chained_reuse_when_k_equals_m():
    """With k_chain == m the chained arm already holds the draw at every (site, y_A draw) cell, so the
    calcareous fallback may reuse it: check that it equals sampling that cell on its own."""
    rng = np.random.default_rng(0)
    n, m, p = 40, 10, 4
    X = rng.normal(size=(400, p))
    y = X[:, 0] + 0.2 * rng.normal(size=400)
    mc = ForestSampler(seed=0, n_estimators=50, n_jobs=1).fit(X, y)
    Xte, ya = rng.normal(size=(n, p - 2)), rng.normal(size=(n, m, 2))
    U = rng.random((n, m))
    Xc = np.hstack([np.repeat(Xte, m, axis=0), ya.reshape(-1, 2)])
    chained = mc.sample(Xc, U.reshape(n * m, 1)).reshape(n, m)
    ii, jj = np.nonzero(rng.random((n, m)) < 0.2)
    alone = mc.sample(np.hstack([Xte[ii], ya[ii, jj]]), U[ii, jj][:, None])[:, 0]
    assert np.allclose(chained[ii, jj], alone)


def _cfg(tmp_name, **kw):
    base = dict(name=tmp_name, n_folds=3, folds=[0], levels=[1.0, 0.3], group_reps=1, random_reps=0, m=20,
                k_chain=20, min_train=30, abundant={"n_estimators": 20}, gate_model={"n_estimators": 20},
                learners={"qrf": {"n_estimators": 20}})
    base.update(kw)
    return Config(**base)


def test_resume_guard_refuses_a_changed_config(tmp_path):
    cfg = _cfg("g")
    out = tmp_path / "g"
    out.mkdir()
    import yaml
    from dataclasses import asdict
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (out / "meta_f0_all_1_r0.json").write_text("{}")
    _check_resume(cfg, out)                                    # same config: fine
    old = yaml.safe_load((out / "config.yaml").read_text())    # a folder written before a field existed
    old.pop("cec_weight_power")
    (out / "config.yaml").write_text(yaml.safe_dump(old, sort_keys=False))
    _check_resume(cfg, out)                                    # the missing field takes its default: fine
    with pytest.raises(RuntimeError, match="k_chain"):
        _check_resume(_cfg("g", k_chain=10), out)


def test_failing_configuration_is_logged_and_skipped(tmp_path, monkeypatch):
    df = make_table(n=900, scenario="relation", seed=3)
    real = experiment.run_config

    def flaky(cfg, fold, level, scheme, rep, shared, out_dir):
        if level < 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(cfg, fold, level, scheme, rep, shared, out_dir)

    monkeypatch.setattr(experiment, "run_config", flaky)
    out = run(_cfg("flaky"), df=df, results_root=tmp_path)
    assert (out / "meta_f0_all_1_r0.json").exists()
    assert not (out / "meta_f0_group_0.3_r0.json").exists()
    assert "simulated" in (out / "errors.log").read_text()
    assert experiment.LAST_FAILED == ["f0_group_0.3_r0"]               # the CLI exits 1 on this
    assert (out / "code_version.txt").read_text().strip()               # code version recorded per run


def test_spearman_matches_scipy_and_handles_constants():
    from scipy.stats import spearmanr
    from pedopilot.experiment import _spearman
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    b = a + b
    assert abs(_spearman(a, b) - spearmanr(a, b).statistic) < 1e-12
    assert np.isnan(_spearman(a, np.ones(500)))


def test_rank_check_separates_coherent_from_independent_draws(tmp_path):
    """On the synthetic relation world the structured draws must keep the CEC-clay rank correlation
    much better than the independent (free) draws: the Figure-1 check works end to end."""
    from pedopilot.analysis import rank_table
    out = run(_cfg("rank", levels=[1.0], learners={"qrf": {"n_estimators": 30}}),
              df=make_table(n=1500, scenario="relation", seed=5), results_root=tmp_path)
    rk = rank_table(out)
    g = rk[rk.target == "cec"].set_index("arm")
    obs = g.rho_obs.mean()
    assert abs(g.loc["qrf:structured_ya", "rho_draws"] - obs) < abs(g.loc["qrf:free", "rho_draws"] - obs)
    assert g.loc["qrf:free", "rho_draws"] < g.loc["qrf:structured_ya", "rho_draws"]


def test_region_bootstrap_runs_and_brackets_the_estimate(tmp_path):
    from pedopilot.bootstrap import bootstrap, region_of
    assert region_of("7_-2") == "2_-1" and region_of("-1_4") == "-1_1"            # floor division, as in prep
    run(_cfg("boot", levels=[1.0, 0.3], group_reps=1, learners={"qrf": {"n_estimators": 20}}),
        df=make_table(n=1200, scenario="relation", seed=9), results_root=tmp_path)
    b = bootstrap("boot", level=0.3, n_boot=200, results_root=tmp_path)
    assert b["lo"] <= b["estimate"] <= b["hi"] and b["n_regions"] > 3
    assert (tmp_path / "boot" / "bootstrap_structured_ya_vs_chained.json").exists()


def test_cec_weight_power():
    from pedopilot.relations import RELATIONS
    clay = np.array([5.0, 20.0, 40.0])
    assert RELATIONS["n"].weights(clay, 1.0) is None
    assert np.allclose(RELATIONS["cec"].weights(clay), clay ** 2)            # pre-specified default
    assert np.allclose(RELATIONS["cec"].weights(clay, 1.0), clay)
    assert RELATIONS["cec"].weights(clay, 0) is None
    import yaml
    for name in ("gpu_smoke", "gpu_quick", "gpu_ft", "gpu_full", "cpu_qrf_w1"):
        cfg = yaml.safe_load((experiment.CONFIGS / f"{name}.yaml").read_text())
        assert cfg["cec_weight_power"] == 1.0, name                           # the clay^1 design (docs/DESIGN.md)
        Config(**cfg)                                                         # every config loads


def test_neural_learners_need_cuda_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run(_cfg("gpu", learners={"tabm": {}}), df=make_table(n=300, seed=1), results_root=tmp_path)


def test_code_version_tracks_pipeline_code_only():
    """code_version.txt follows the last commit of src/pedopilot, so committing results or docs while a
    run is in progress does not trigger the 'mixed code versions' warning."""
    import subprocess
    from pathlib import Path
    pkg = Path(experiment.__file__).resolve().parent
    git = subprocess.run(["git", "log", "-1", "--format=%h", "--", "."], capture_output=True, text=True,
                         cwd=pkg)
    if git.returncode != 0 or not git.stdout.strip():
        pytest.skip("not a git checkout")
    assert experiment._code_version().split("+")[0] == git.stdout.strip()


def test_region_bootstrap_uses_complete_folds_only(tmp_path):
    """Like analyze, the region bootstrap leaves out a fold with a missing configuration (a stopped run)."""
    from pedopilot.bootstrap import bootstrap
    run(_cfg("bootp", folds=[0, 1], levels=[1.0, 0.3], group_reps=2, learners={"qrf": {"n_estimators": 20}}),
        df=make_table(n=1200, scenario="relation", seed=9), results_root=tmp_path)
    assert bootstrap("bootp", level=0.3, n_boot=50, results_root=tmp_path)["folds"] == [0, 1]
    for p in (tmp_path / "bootp").glob("*f1_group_0.3_r1*"):
        p.unlink()
    assert bootstrap("bootp", level=0.3, n_boot=50, results_root=tmp_path)["folds"] == [0]
