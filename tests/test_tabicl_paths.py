"""Exercise the TabICLv2 wrappers end to end with a TINY randomly initialised TabICL checkpoint.
The predictions are meaningless; the point is that every code path (context resampling, quantile
output, clamping, draws, fine-tuning with a spatial validation split, final-context refit) runs.
Uses the real checkpoint format of tabicl (dict with 'config' and 'state_dict')."""
import numpy as np
import pytest
import torch

tabicl = pytest.importorskip("tabicl")

TINY = dict(max_classes=0, num_quantiles=999, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
            row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2, ff_factor=1)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    from tabicl._model.tabicl import TabICL
    torch.manual_seed(0)
    try:
        model = TabICL(**TINY)
    except TypeError:                                    # config keys differ across tabicl versions
        cfg = dict(TINY)
        cfg.pop("max_classes")
        model = TabICL(**cfg)
        TINY.clear()
        TINY.update(cfg)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.ckpt"
    torch.save({"config": TINY, "state_dict": model.state_dict()}, path)
    return path


def _data(n=300, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.3 * rng.normal(size=n)
    groups = rng.integers(0, 20, size=n).astype(str)
    return X, y, groups


def test_tabicl_in_context_quantiles_and_draws(tiny_ckpt):
    from pedopilot.learners import TabICLQuantile
    from pedopilot.sampling import LEVELS
    X, y, g = _data()
    m = TabICLQuantile(seed=0, device="cpu", n_estimators=2, model_path=tiny_ckpt, kv_cache=False)
    m.fit(X[:250], y[:250], w=np.abs(X[:250, 1]) + 0.1, groups=g[:250])
    q = m.quantiles(X[250:])
    assert q.shape == (50, len(LEVELS)) and np.all(np.diff(q, axis=1) >= 0)
    lo, hi = y[:250].min(), y[:250].max()
    assert q.min() >= lo - 0.5 * (hi - lo) - 1e-9 and q.max() <= hi + 0.5 * (hi - lo) + 1e-9
    d = m.sample(X[250:], np.random.default_rng(1).random((50, 7)))
    assert d.shape == (50, 7) and np.isfinite(d).all()
    assert 0 < m.ess_ <= 250


def test_tabicl_finetuned_runs_and_refits_final_context(tiny_ckpt):
    pytest.importorskip("transformers")
    from pedopilot.learners import TabICLFinetuned
    X, y, g = _data(n=260)
    m = TabICLFinetuned(seed=0, device="cpu", epochs=1, patience=1, n_estimators_inference=2, model_path=tiny_ckpt)
    m.fit(X[:220], y[:220], w=np.abs(X[:220, 1]) + 0.1, groups=g[:220])
    q = m.quantiles(X[220:])
    assert q.shape[0] == 40 and np.isfinite(q).all()
    d = m.sample(X[220:], np.random.default_rng(2).random((40, 5)))
    assert d.shape == (40, 5)
