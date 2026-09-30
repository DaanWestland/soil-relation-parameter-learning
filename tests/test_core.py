"""Unit tests for the parts where a silent bug would bias every result."""
import numpy as np
import pandas as pd
import pytest

from pedopilot.learners import ForestSampler, group_val_split
from pedopilot.relations import (A_HI, A_LO, B, R_HI, R_LO, RELATIONS, blogit, inv_blogit, weighted_median,
                                 ya_inverse, ya_transform)
from pedopilot.sampling import LEVELS, draws_from_quantiles, site_uniforms
from pedopilot.scoring import crps_fair, pit
from pedopilot.thinning import group_thin, random_thin


def test_blogit_roundtrip():
    v = np.linspace(R_LO + 0.1, R_HI - 0.1, 50)
    assert np.allclose(inv_blogit(blogit(v, R_LO, R_HI), R_LO, R_HI), v)
    assert np.all(inv_blogit(np.array([-50.0, 50.0]), A_LO, A_HI) >= A_LO)


def test_ya_roundtrip():
    clay, oc, ph = np.array([5.0, 30.0, 70.0]), np.array([2.0, 15.0, 90.0]), np.array([4.5, 6.2, 8.1])
    c2, o2, p2 = ya_inverse(ya_transform(clay, oc, ph))
    assert np.allclose(c2, clay) and np.allclose(o2, oc) and np.allclose(p2, ph)


@pytest.mark.parametrize("target", ["n", "cec"])
def test_relation_inverts_and_residual(target):
    rel = RELATIONS[target]
    rng = np.random.default_rng(0)
    clay, oc = rng.uniform(5, 60, 200), rng.uniform(2, 60, 200)
    theta = rng.uniform(rel.lo, rel.hi, 200)
    y = rel.forward(theta, clay, oc)
    assert np.allclose(rel.implied(y, clay, oc), theta)
    assert np.allclose(rel.residual(y, clay, oc), 0, atol=1e-9)          # inside bounds: no residual
    # outside the bounds, forward(clip(theta)) + residual recovers y
    y_out = rel.forward(np.full(200, rel.hi * 1.5), clay, oc)
    r = rel.residual(y_out, clay, oc)
    back = rel.add_residual(rel.forward(np.full(200, rel.hi), clay, oc), r)
    assert np.allclose(back, y_out)


def test_cec_constants():
    # 3.5 cmolc/kg per % OC == 0.35 per g/kg
    assert B == pytest.approx(0.35)
    lo, hi = RELATIONS["cec"].envelope(30.0, 20.0)                        # 30% clay, 2% OC
    assert lo == pytest.approx(0.02 * 30 + 7.0) and hi == pytest.approx(1.5 * 30 + 7.0)


def test_weighted_median():
    assert weighted_median([1, 2, 3], [1, 1, 10]) == 3
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2


def test_quantile_draws_recover_normal():
    from scipy.stats import norm
    q = np.tile(norm.ppf(LEVELS), (4, 1))
    u = np.random.default_rng(1).random((4, 20000))
    d = draws_from_quantiles(q, LEVELS, u)
    assert abs(d.mean()) < 0.03 and abs(d.std() - 1) < 0.05


def test_site_uniforms_are_common_and_distinct():
    a = site_uniforms([10, 11], 50, 7)
    b = site_uniforms([11, 10], 50, 7)
    assert np.allclose(a[0], b[1]) and not np.allclose(a[0], a[1])
    assert not np.allclose(site_uniforms([10], 50, 8), a[:1])


def test_forest_sampler_weighted_leaf_draws():
    # one feature, two clean groups; within group 1 the weights favour the value 10
    X = np.r_[np.zeros(50), np.ones(50)][:, None]
    y = np.r_[np.zeros(50), np.where(np.arange(50) < 25, 10.0, 20.0)]
    w = np.r_[np.ones(50), np.where(np.arange(50) < 25, 9.0, 1.0)]
    fs = ForestSampler(seed=0, n_estimators=50, min_samples_leaf=5, max_features=1.0, n_jobs=1).fit(X, y, w=w)
    d = fs.sample(np.array([[1.0]]), np.random.default_rng(0).random((1, 5000)))
    assert set(np.unique(d)) <= {10.0, 20.0}
    assert 0.85 < np.mean(d == 10.0) < 0.95                               # expected 0.9
    d0 = fs.sample(np.array([[0.0]]), np.random.default_rng(0).random((1, 500)))
    assert np.all(d0 == 0.0)


def test_forest_sampler_joint_rows():
    rng = np.random.default_rng(0)
    X = rng.random((300, 2))
    Y = np.column_stack([X[:, 0], 2 * X[:, 0] + 1])                     # perfectly dependent columns
    fs = ForestSampler(seed=0, n_estimators=30, n_jobs=1).fit(X, Y)
    J = fs.sample_joint(X[:5], rng.random((5, 40)))
    assert np.allclose(J[..., 1], 2 * J[..., 0] + 1)                      # whole rows are drawn


def test_fair_crps_and_pit():
    y = np.array([0.0, 0.0])
    d = np.random.default_rng(0).normal(size=(2, 4000))
    assert np.allclose(crps_fair(y, d), 0.2337, atol=0.02)               # N(0,1) at 0: (sqrt2-1)/sqrt(pi)
    # mid-rank on the (m + 1) grid: 1 below, 1 tie, m = 4 -> (1 + (1 + 1) / 2) / 5
    assert np.allclose(pit(np.array([0.0]), np.array([[-1.0, 1.0, 0.0, 2.0]])), [0.4])


def test_pit_and_coverage_are_calibrated_for_exchangeable_draws():
    from pedopilot.scoring import coverage
    rng = np.random.default_rng(1)
    n, m = 40000, 100
    y, d = rng.normal(size=n), rng.normal(size=(n, m))
    assert abs(coverage(y, d, 0.9).mean() - 91 / 101) < 0.006          # np.quantile intervals give ~0.88
    h = np.histogram(pit(y, d), bins=10, range=(0, 1))[0] / n
    assert np.abs(h - 0.1).max() < 0.012                                 # flat: no binning artefact
    # clustered draws (5 draws share each of 20 abundant draws): one draw per cluster keeps calibration
    z = rng.normal(size=(n, 20))
    dc = np.repeat(z, 5, axis=1) * np.sqrt(0.8) + rng.normal(size=(n, m)) * np.sqrt(0.2)
    cl = np.repeat(np.arange(20), 5)
    assert abs(coverage(y, dc, 0.9, cl).mean() - 19 / 21) < 0.008


def _toy_train(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": rng.integers(0, 40, n).astype(str),
                         "n": np.where(rng.random(n) < 0.5, 1.0, np.nan),
                         "cec": np.where(rng.random(n) < 0.7, 10.0, np.nan)})


def test_group_thin_fraction_and_regions():
    tr = _toy_train()
    has = (tr.n.notna() | tr.cec.notna()).values
    keep, kept = group_thin(tr, 0.1, seed=3)
    assert keep.sum() <= has.sum()
    assert abs(keep.sum() / has.sum() - 0.1) < 0.05
    assert set(tr.loc[keep, "region"]) <= kept
    assert np.all(~keep | has)


def test_random_thin_matches_counts_per_target():
    tr = _toy_train()
    g, _ = group_thin(tr, 0.1, seed=3)
    keep = random_thin(tr, g, seed=1)
    for c in ("n", "cec"):
        assert (keep & tr[c].notna().values).sum() == (g & tr[c].notna().values).sum()
    assert keep.sum() == g.sum()


def test_clustered_scores_reduce_to_fair_and_remove_bias():
    import scoringrules as sr
    from pedopilot.scoring import crps_fair, energy_score, variogram_score
    rng = np.random.default_rng(0)
    m, K = 100, 20
    e = rng.normal(size=(500, m))
    y = rng.normal(size=500)
    assert np.allclose(crps_fair(y, e), crps_fair(y, e, np.arange(m)))
    # grouped draws: 20 groups of 5 sharing a component; clustered estimator ~ iid estimate
    n = 20000
    ya = rng.normal(size=(n, K))
    grouped = np.repeat(ya, m // K, 1) + 0.5 * rng.normal(size=(n, m))
    iid = rng.normal(size=(n, m)) * np.sqrt(1.25)
    yy = rng.normal(size=n) * np.sqrt(1.25)
    cl = (np.arange(m) * K) // m
    true = crps_fair(yy, iid).mean()
    assert abs(crps_fair(yy, grouped, cl).mean() / true - 1) < 0.01
    assert crps_fair(yy, grouped).mean() / true - 1 > 0.015          # the naive estimator is biased
    o, f = rng.normal(size=(30, 5)), rng.normal(size=(30, 40, 5))
    assert np.allclose(energy_score(o, f), energy_score(o, f, np.arange(40)))
    assert np.allclose(variogram_score(o, f), np.asarray(sr.vs_ensemble(o, f, p=0.5, estimator="fair")))


def test_tabm_head_starts_as_standard_normal():
    import torch
    from scipy.stats import norm
    from pedopilot.learners import TabMQuantile
    q = TabMQuantile(device="cpu")._monotone(torch.zeros(1, 1, len(LEVELS))).numpy()[0, 0]
    assert np.allclose(np.diff(q), np.diff(norm.ppf(LEVELS)), atol=1e-5)


def test_clamp_to_range():
    from pedopilot.learners import clamp_to_range
    q = np.array([[-100.0, 0.0, 100.0]])
    assert np.allclose(clamp_to_range(q, 0.0, 10.0), [[-5.0, 0.0, 15.0]])


def test_group_val_split_whole_blocks():
    groups = np.repeat(np.arange(30), 20)
    va = group_val_split(groups, 0.2, 0, len(groups))
    for g in np.unique(groups):
        assert va[groups == g].all() or (~va[groups == g]).all()
