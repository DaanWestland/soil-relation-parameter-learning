"""Learners. Every learner has the same interface:

    fit(X, y, w=None, groups=None, X_val=None, y_val=None) -> self
    sample(X, u) -> draws of shape (n, m)          # u: (n, m) uniforms (common random numbers)

so every arm is represented by exactly m draws per site, produced from shared uniforms.
`w` are case weights, `groups` the spatial block of each row (for grouped validation splits).
`X_val`/`y_val` are an unused interface slot (no caller passes them; the neural learners make their own
spatially grouped split). Constructors accept `**_` and silently IGNORE unknown keyword arguments, so a
misspelt option in a config has no effect: check the spelling against the signatures below.

  config name  class            method                                          case weights enter      device
  qrf          ForestSampler    random forest with QRF-weighted draws           forest fit, leaf draws  CPU
                                (Meinshausen 2006)
  tabm         TabMQuantile     TabM MLP ensemble (Gorishniy et al. 2025),      weighted pinball loss   GPU (CPU ok
                                99-quantile head, pinball loss                                          for tests)
  tabicl       TabICLQuantile   TabICLv2 in context (no training), 99 quantiles resampled context       GPU
  tabicl_ft    TabICLFinetuned  TabICLv2 fine-tuned, pinball-validated          resampled final context GPU

Pinned versions are load-bearing: `tabm==0.0.3` and `tabicl==2.2.0` (pyproject.toml). The wrappers use
internals of these packages: TabM.make with rtdl_num_embeddings.PeriodicEmbeddings (TabMQuantile._net),
the private tabicl._finetune module (TabICLFinetuned.__init__) and the fitted _final_estimator_,
_transform_X_for_inference and _load_model of FinetunedTabICLRegressor (TabICLFinetuned.fit/quantiles).
Other versions may fail, or silently behave differently.
"""
from __future__ import annotations

import gc
from typing import NamedTuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .design import LEARNER_ORDER
from .paths import TABICL_CKPT
from .sampling import LEVELS, draws_from_quantiles


def resolve_device(device="auto"):
    """'auto' -> 'cuda' if torch sees a GPU, else 'cpu'; any other value is returned unchanged."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def free_memory():
    """Free memory between fitted models: collect garbage and empty the CUDA cache, so that only one
    fitted model (one TabICL KV cache) is alive on the GPU at a time. Harmless on a CPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_free_cuda = free_memory                                 # former name


def clamp_to_range(q, lo, hi, margin=0.5):
    """Safety net for neural learners on a transformed (log / logit) target scale: keep predicted
    quantiles within the training range widened by `margin` x range on each side, so that a badly
    extrapolating network cannot produce exp()-exploded draws. QRF cannot leave the range at all."""
    span = hi - lo
    return np.clip(q, lo - margin * span, hi + margin * span)


def group_val_split(groups, frac, seed, n):
    """Validation mask made of whole spatial blocks (falls back to random rows if too few blocks)."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        ug = np.unique(groups)
        if len(ug) >= 5:
            val_g = rng.choice(ug, size=max(1, int(round(frac * len(ug)))), replace=False)
            va = np.isin(groups, val_g)
            if 10 <= va.sum() <= n - 20:
                return va
    return rng.random(n) < frac


# ----------------------------------------------------------------------------- forests
class _LeafIndex(NamedTuple):
    """Per tree: training rows sorted by leaf (`order`), the sorted leaf ids, the first and one-past-last
    position of each leaf in `order`, the cumulative case weights along `order`, and the cumulative
    weight just before each leaf's first row and at its last row."""
    order: np.ndarray
    leaf_ids: np.ndarray
    start: np.ndarray
    end: np.ndarray
    cumw: np.ndarray
    cumw_start: np.ndarray
    cumw_end: np.ndarray


class ForestSampler:
    """Random forest whose predictive distribution is the (case-weighted) QRF distribution
    (Meinshausen 2006): a draw picks a tree and then a training row in the test point's leaf with
    probability proportional to its case weight. With a 2-D target this yields joint draws of whole
    rows (a DRF-style joint draw from a multi-output forest)."""

    def __init__(self, seed=0, n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=-1, **_):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, n_jobs=n_jobs, random_state=seed)

    def fit(self, X, y, w=None, groups=None, X_val=None, y_val=None):
        """Fit the forest on X (n, p) and y (n,) or (n, d), with optional case weights w (used in the
        forest fit and in the leaf draws). groups, X_val and y_val are ignored (interface slot).

        Builds per tree an index of the training rows sorted by leaf, so that a draw can pick a row of a
        leaf with probability proportional to its case weight by a binary search in cumulative weights."""
        Y = np.asarray(y, float)
        self.Y = Y if Y.ndim == 2 else Y[:, None]
        self.w = np.ones(len(X)) if w is None else np.asarray(w, float)
        self.rf.fit(X, self.Y if self.Y.shape[1] > 1 else self.Y.ravel(), sample_weight=w)
        leaves = self.rf.apply(X)
        self.index = []
        for t in range(leaves.shape[1]):
            order = np.argsort(leaves[:, t], kind="stable")
            lv = leaves[order, t]
            uniq, start = np.unique(lv, return_index=True)
            end = np.r_[start[1:], len(lv)]
            cw = np.cumsum(self.w[order])
            cw0 = np.r_[0.0, cw][start]
            self.index.append(_LeafIndex(order, uniq, start, end, cw, cw0, cw[end - 1]))
        return self

    def sample_rows(self, X, u):
        """Indices of training rows drawn for each (site, draw); u has shape (n, m). Leaves are looked up
        per tree, only for the (site, draw) cells that chose that tree: identical to rf.apply, but memory
        is O(n m) instead of O(n T) (rf.apply on 600k chained rows x 500 trees needs about 5 GB)."""
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        n, T = len(X32), len(self.rf.estimators_)
        m = u.shape[1]
        # split each uniform into a tree choice and a within-leaf position (keeps CRN structure)
        tsel = np.minimum((u * T).astype(np.int64), T - 1)
        v = u * T - tsel
        out = np.empty((n, m), dtype=np.int64)
        for t in range(T):
            mask = tsel == t
            if not mask.any():
                continue
            rows = np.nonzero(mask)[0]
            order, uniq, start, end, cw, cw0, cwe = self.index[t]
            pos = np.searchsorted(uniq, self.rf.estimators_[t].apply(X32[rows]))
            target = cw0[pos] + v[mask] * (cwe[pos] - cw0[pos])
            k = np.clip(np.searchsorted(cw, target, side="right"), start[pos], end[pos] - 1)
            out[mask] = order[k]
        return out

    def sample(self, X, u):
        """Draws of the (first) target: (n, m) for uniforms u of shape (n, m)."""
        return self.Y[self.sample_rows(X, u)][..., 0]

    def sample_joint(self, X, u):
        """Joint draws of every target column (whole training rows): (n, m, d)."""
        return self.Y[self.sample_rows(X, u)]            # (n, m, d)


class RegimeClassifier:
    """p(regime | covariates) for the prediction-time regime gate."""

    def __init__(self, seed=0, n_estimators=300, min_samples_leaf=5, n_jobs=-1, **_):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                         n_jobs=n_jobs, random_state=seed)

    def fit(self, X, flag):
        """Fit p(flag | X); a constant probability if the flag has one value only."""
        flag = np.asarray(flag, bool)
        self.const = None
        if flag.all() or (~flag).all():
            self.const = float(flag.mean())
        else:
            self.rf.fit(X, flag)
        return self

    def proba(self, X):
        """p(flag = True | X), shape (n,)."""
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.rf.predict_proba(X)[:, list(self.rf.classes_).index(True)]


# ----------------------------------------------------------------------------- TabM
class TabMQuantile:
    """TabM (BatchEnsemble MLP with periodic numerical embeddings; Gorishniy et al. 2025) with a
    monotone 99-quantile head, trained with a case-weighted pinball loss (a CRPS approximation)
    and early stopping on a spatially grouped validation split."""

    def __init__(self, seed=0, device="auto", k=32, d_block=512, n_blocks=2, d_embedding=16,
                 lr=1e-3, wd=3e-4, max_epochs=200, patience=16, batch=256, val_frac=0.15, **_):
        self.seed, self.device = seed, resolve_device(device)
        self.k, self.d_block, self.n_blocks, self.d_embedding = k, d_block, n_blocks, d_embedding
        self.lr, self.wd, self.max_epochs, self.patience = lr, wd, max_epochs, patience
        self.batch, self.val_frac = batch, val_frac

    def _net(self, p):
        from rtdl_num_embeddings import PeriodicEmbeddings
        from tabm import TabM
        return TabM.make(n_num_features=p, d_out=len(LEVELS), k=self.k, n_blocks=self.n_blocks,
                         d_block=self.d_block,
                         num_embeddings=PeriodicEmbeddings(p, self.d_embedding, lite=False)).to(self.device)

    def _monotone(self, raw):                            # (b, k, Q) -> non-decreasing quantiles
        # increments = softplus(raw + bias) with bias chosen so that raw = 0 gives the spacings of
        # a standard normal (the targets are standardised). Without this offset an untrained head
        # starts with a spread of about 68 standard deviations (softplus(0) x 98 increments).
        if not hasattr(self, "_bias") or self._bias.device != raw.device:
            from scipy.stats import norm
            gaps = np.diff(norm.ppf(LEVELS))
            self._bias = torch.as_tensor(np.log(np.expm1(gaps)), dtype=raw.dtype, device=raw.device)
        inc = torch.nn.functional.softplus(raw[..., 1:] + self._bias)
        return torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(inc, -1)], -1)

    def _loss(self, q, y, w):
        """Case-weighted pinball loss averaged over the k members and the 99 levels (a CRPS approximation).
        q (b, k, Q), y (b,), w (b,). Its numpy twin (tabicl_ft validation) is in TabICLFinetuned."""
        if getattr(self, "_alpha", None) is None or self._alpha.device != q.device or self._alpha.dtype != q.dtype:
            self._alpha = torch.as_tensor(LEVELS, dtype=q.dtype, device=q.device)   # cached: no sync per step
        a = self._alpha
        d = y[:, None, None] - q
        per_row = torch.maximum(a * d, (a - 1) * d).mean(dim=(1, 2))
        return (per_row * w).sum() / w.sum()

    def fit(self, X, y, w=None, groups=None, X_val=None, y_val=None):
        """Train on standardised X and y with AdamW and early stopping on whole spatial blocks
        (group_val_split); the best-validation weights are restored. X_val/y_val are ignored.
        Sets epochs_ (epochs run) and val_loss_ (best validation loss; a diagnostic, not used elsewhere).
        Not bit-reproducible run to run, even on CPU: torch.manual_seed is set, but no deterministic
        algorithms or thread settings are enforced."""
        torch.manual_seed(self.seed)
        X, y = np.asarray(X, float), np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = y.mean(), y.std() + 1e-8
        va = group_val_split(groups, self.val_frac, self.seed, len(y))
        dev = self.device

        def t(a):
            return torch.as_tensor(np.asarray(a, np.float32), device=dev)

        Xs, ys = (X - self.xm) / self.xs, (y - self.ym) / self.ys
        self.y_lo, self.y_hi = float(ys.min()), float(ys.max())
        Xt, yt, wt = t(Xs[~va]), t(ys[~va]), t(w[~va] / w[~va].mean())
        Xv, yv, wv = t(Xs[va]), t(ys[va]), t(w[va] / w[va].mean())
        net = self._net(X.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        # batch size scales with n so that small (thinned) training sets still get enough steps
        batch = int(np.clip(len(yt) // 16, 32, self.batch))
        best, bad, ep = np.inf, 0, 0
        best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}   # safe if val loss is NaN
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(yt), device=dev)
            for i in range(0, len(yt), batch):
                b = perm[i:i + batch]
                opt.zero_grad(set_to_none=True)
                self._loss(self._monotone(net(Xt[b])), yt[b], wt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = self._loss(self._monotone(net(Xv)), yv, wv).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net, self.epochs_, self.val_loss_ = net.eval(), ep + 1, best
        return self

    def quantiles(self, X):
        """99 quantiles (n, 99) on the original target scale: members averaged, sorted, clamped."""
        Xs = torch.as_tensor(((np.asarray(X, float) - self.xm) / self.xs).astype(np.float32), device=self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xs), 8192):
                # average the k members' quantiles (quantile averaging), then re-sort
                out.append(self._monotone(self.net(Xs[i:i + 8192])).mean(1).float().cpu().numpy())
        q = np.sort(np.concatenate(out), axis=1)
        return clamp_to_range(q, self.y_lo, self.y_hi) * self.ys + self.ym

    def sample(self, X, u):
        """Inverse-CDF draws (n, m) from the 99 quantiles at uniforms u."""
        return draws_from_quantiles(self.quantiles(X), LEVELS, u)


# ----------------------------------------------------------------------------- TabICLv2
def _resample_context(X, y, w, max_ctx, seed):
    """TabICL takes no case weights: emulate them by weight-proportional SYSTEMATIC resampling of the
    context (each row appears floor or ceil of size*p_i times: the weights are reproduced with almost
    no Monte Carlo noise), and cap the context size. Returns X, y and the effective sample size."""
    n = len(y)
    rng = np.random.default_rng(seed)
    ess = float(n) if w is None else float(np.sum(w) ** 2 / np.sum(np.square(w)))
    if w is None and n <= max_ctx:
        return X, y, ess
    size = min(n, max_ctx)
    if w is None:
        idx = rng.choice(n, size=size, replace=False)
    else:
        p = np.asarray(w, float) / np.sum(w)
        idx = np.minimum(np.searchsorted(np.cumsum(p), (rng.random() + np.arange(size)) / size), n - 1)
    return X[idx], y[idx], ess


def _checkpoint_kwargs(model_path=None):
    """Use an explicit checkpoint, else the local models/ copy, else let tabicl download it."""
    from pathlib import Path
    p = Path(model_path) if model_path else (TABICL_CKPT if TABICL_CKPT.exists() else None)
    return {"model_path": p, "allow_auto_download": False} if p is not None else {}


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _q_chunked(predict, X, chunk=4096, min_chunk=128):
    """TabICL predicts all test rows in one pass (it splits only over ensemble members), so predict in
    row chunks. Identical results: test rows attend only to the context (embed_with_test=False).
    On a CUDA out-of-memory error the chunk is halved and the same rows are retried (GPUs with about
    10 GB; tested: RTX 3080)."""
    X = np.asarray(X, np.float32)
    out, i = [], 0
    while i < len(X):
        try:
            out.append(np.asarray(predict(X[i:i + chunk], output_type="quantiles", alphas=list(LEVELS)), float))
            i += chunk
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if not _is_oom(e) or chunk <= min_chunk:
                raise
            free_memory()
            chunk //= 2
            print(f"    [tabicl] CUDA out of memory: prediction chunk -> {chunk} rows", flush=True)
    return np.concatenate(out)


class TabICLQuantile:
    """TabICLv2 used in context (no training). Knowledge enters only through the target.
    The wrapped tabicl regressor is `self.m` (not to be confused with the number of draws, cfg.m)."""

    _oom_ctx: dict = {}                       # kv_cache mode -> smallest context size that ran out of memory

    def __init__(self, seed=0, device="auto", n_estimators=8, max_ctx=48_000, kv_cache=True, batch_size=8,
                 model_path=None, pred_chunk=4096, **_):
        from tabicl import TabICLRegressor
        self.seed, self.max_ctx, self.pred_chunk = seed, max_ctx, pred_chunk
        kw = dict(n_estimators=n_estimators, device=resolve_device(device), random_state=seed,
                  kv_cache=kv_cache, batch_size=batch_size)
        kw.update(_checkpoint_kwargs(model_path))
        self.m = TabICLRegressor(**kw)

    def fit(self, X, y, w=None, groups=None, X_val=None, y_val=None):
        """'Fit' = store the context: case weights are emulated by resampling (ess_ = effective sample
        size), the context is capped at max_ctx rows, and the KV cache is built. groups, X_val and y_val
        are ignored. On CUDA out of memory the cache mode falls back kv -> repr -> none (kv_mode_)."""
        self.y_lo, self.y_hi = float(np.min(y)), float(np.max(y))
        X, y, self.ess_ = _resample_context(np.asarray(X, np.float32), np.asarray(y, float), w, self.max_ctx, self.seed)
        # the KV cache of a large context may not fit a GPU with about 10 GB (tested: RTX 3080): fall back to the ~24x smaller "repr"
        # cache, then to no cache (same predictions, slower). A mode that ran out of memory at some context
        # size is skipped for contexts at least as large in later fits (no repeated failed attempts).
        modes = list(dict.fromkeys([self.m.kv_cache, "repr", False]))
        n = len(y)
        todo = [md for md in modes if n < TabICLQuantile._oom_ctx.get(md, np.inf)] or modes[-1:]
        for mode in todo:
            try:
                self.m.set_params(kv_cache=mode)
                self.m.fit(X, y)
                break
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if not _is_oom(e) or mode is todo[-1]:
                    raise
                TabICLQuantile._oom_ctx[mode] = min(n, TabICLQuantile._oom_ctx.get(mode, np.inf))
                self.m.__dict__.pop("model_kv_cache_", None)
                free_memory()
                print(f"    [tabicl] CUDA out of memory building the kv_cache={mode!r} cache ({n} rows): "
                      "falling back", flush=True)
        self.kv_mode_ = self.m.kv_cache
        return self

    def quantiles(self, X):
        """99 quantiles (n, 99), predicted in row chunks, sorted and clamped to the widened target range."""
        q = _q_chunked(self.m.predict, X, self.pred_chunk)
        return clamp_to_range(np.sort(q, axis=1), self.y_lo, self.y_hi)

    def sample(self, X, u):
        """Inverse-CDF draws (n, m) from the 99 quantiles at uniforms u."""
        return draws_from_quantiles(self.quantiles(X), LEVELS, u)


class TabICLFinetuned(TabICLQuantile):
    """TabICLv2 fully fine-tuned (tabicl's FinetunedTabICLRegressor: AdamW, cosine warm-up, gradient
    clipping, early stopping with best-weight restore). The training loss is the pinball loss over the
    model's quantile head (a CRPS approximation); early stopping uses the mean pinball loss on a
    spatially grouped validation split instead of the default MSE. Needs `pip install transformers`."""

    def __init__(self, seed=0, device="auto", epochs=30, learning_rate=1e-5, patience=8,
                 n_estimators_inference=8, max_ctx=48_000, val_frac=0.15, model_path=None,
                 steps_per_epoch=1, time_limit=None, pred_chunk=4096, **_):
        from tabicl._finetune.base import ValidationMetrics
        from tabicl._finetune.regressor import FinetunedTabICLRegressor

        class _PinballValidated(FinetunedTabICLRegressor):
            def _run_validation(self, inner, X_train, y_train, X_val, y_val):
                try:
                    inner.fit(X_train, y_train)
                    q = np.sort(np.asarray(inner.predict(X_val, output_type="quantiles", alphas=list(LEVELS)), float), 1)
                except (ValueError, RuntimeError):
                    self.__dict__.setdefault("hist_", []).append(float("nan"))
                    return ValidationMetrics(primary=float("nan"))
                d = np.asarray(y_val, float)[:, None] - q      # pinball loss: numpy twin of TabMQuantile._loss
                pin = float(np.mean(np.maximum(LEVELS * d, (LEVELS - 1) * d)))
                self.__dict__.setdefault("hist_", []).append(pin)     # validation history (epoch 0 = baseline)
                return ValidationMetrics(primary=-pin, secondary={"pinball": pin})

        self.seed, self.max_ctx, self.val_frac, self.steps_per_epoch = seed, max_ctx, val_frac, steps_per_epoch
        self.pred_chunk = pred_chunk
        kw = dict(epochs=epochs, learning_rate=learning_rate, patience=patience,
                  n_estimators_inference=n_estimators_inference, device=resolve_device(device),
                  random_state=seed, verbose=False, time_limit=time_limit)
        kw.update(_checkpoint_kwargs(model_path))
        self.m = _PinballValidated(**kw)

    def fit(self, X, y, w=None, groups=None, X_val=None, y_val=None):
        """Fine-tune on the rows outside a spatially grouped validation split (distinct rows, no weights),
        with early stopping on the validation pinball loss, then build the final prediction context from
        ALL rows with the case weights on the fine-tuned weights. X_val/y_val are ignored.
        Sets epochs_, ft_gain_ (validation pinball gain over the pretrained model) and ess_."""
        X, y = np.asarray(X, np.float32), np.asarray(y, float)
        self.y_lo, self.y_hi = float(np.min(y)), float(np.max(y))
        va = group_val_split(groups, self.val_frac, self.seed, len(y))
        # fine-tune on distinct, unweighted rows: resampled duplicates could otherwise sit in both the
        # context and the query of a fine-tuning episode and teach the model to copy its twin
        Xtr, ytr, _ = _resample_context(X[~va], y[~va], None, self.max_ctx, self.seed)
        # tabicl makes one optimiser step per chunk of max_data_size rows per epoch. The default (1) keeps each
        # episode at the size of the prediction context (the ctx/query split is redrawn every epoch); more
        # steps come from more epochs. steps_per_epoch > 1 gives smaller episodes than at prediction time.
        # capped at tabicl's default (10k rows): back-propagating through larger episodes needs more than
        # the about 10 GB of the GPUs this was tested on (RTX 3080)
        self.m.set_params(max_data_size=min(10_000, max(128, -(-len(ytr) // self.steps_per_epoch))))
        self.m.fit(Xtr, ytr, X_val=X[va], y_val=y[va])
        h = np.asarray(self.m.__dict__.get("hist_", [np.nan]), float)
        self.epochs_ = len(h) - 1
        self.ft_gain_ = float(h[0] - np.nanmin(h)) if np.isfinite(h).any() else float("nan")   # 0: pretrained kept
        # final prediction context: ALL training rows (incl. the validation blocks, so the context is as
        # large as the in-context arm's) with the case weights, on the fine-tuned weights
        Xc, yc, self.ess_ = _resample_context(X, y, w, self.max_ctx, self.seed)
        fe = self.m._final_estimator_
        fe._load_model = lambda: None                    # keep the fine-tuned model_, do not reload
        fe.fit(self.m._transform_X_for_inference(Xc), yc)
        fe.__dict__.pop("_load_model", None)
        return self

    def quantiles(self, X):
        """99 quantiles (n, 99) from the fine-tuned final estimator, chunked, sorted and clamped."""
        fe = self.m._final_estimator_
        q = _q_chunked(lambda Z, **k: fe.predict(self.m._transform_X_for_inference(Z), **k), X, self.pred_chunk)
        return clamp_to_range(np.sort(q, axis=1), self.y_lo, self.y_hi)


LEARNERS = {"qrf": ForestSampler, "tabm": TabMQuantile, "tabicl": TabICLQuantile, "tabicl_ft": TabICLFinetuned}
assert tuple(LEARNERS) == LEARNER_ORDER
