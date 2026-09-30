"""The pilot experiment.

For every spatial fold x thinning level x scheme x replicate ("configuration"), and for each sparse
target (total N, CEC7), the arms below are fitted on the training fold and scored at the test fold.

Shared by all arms (so arms differ only in how the sparse target is predicted):
  * abundant model: one multi-output forest -> joint (clay, OC, pH) draws at test sites (per fold)
  * regime gate (CEC only): p(calcareous | x) fitted per configuration on the surviving CEC rows;
    per-draw regime from common uniforms
  * common random numbers: the same uniforms per site in every arm

Arms per learner L (qrf, tabm, tabicl, tabicl_ft):
  L:free           log target from covariates x
  L:chained        log target from (x, observed y_A); at test from (x, y_A draws)       [chain rule]
  L:structured     bounded parameter theta from x (clay^2 case weights for CEC), pushed through the
                   relation with the y_A draws, plus a residual drawn conditionally on clay (CEC) or
                   OC (N); calcareous draws (predicted gate) fall back to the chained model at the same
                   y_A draw
  L:structured_ya  theta from (x, y_A), paired with the same y_A draws as L:chained: differs from
                   L:chained ONLY in the formula (same conditioning information)
  L:structured_og  L:structured with the oracle (observed) regime                         [CEC only]
  L:oracle_ya      structured theta pushed through the OBSERVED clay/OC at the test site    [upper bound]
  L:clip           free draws clipped to the pedological envelope (same calcareous fallback)
Learner-free:
  ptf              one constant (regime-specific weighted median) theta          [classic point PTF]
  ptf_marg         theta drawn from its training distribution, independent of x  [theta varies, not with x]

Glossary
  configuration  one (fold, level, scheme, rep): the unit of work, of resuming and of the output files
  level          share of the training profiles with any sparse label that keep their labels (1.0, 0.3, ...)
  scheme         "all" (level 1.0), "group" (whole 300 km regions kept) or "random" (matched counts at
                 random); see thinning.py
  rep            replicate of a thinning draw; random rep r is matched to group rep r (same counts, and
                 scored against the same retained regions)
  unit           fold x rep (x level x scheme x target): the level at which skill is computed
  m              number of draws per site of every arm (100)
  k_chain (K)    number of y_A draws a chained-type arm conditions on; its m draws come in K groups of m/K
                 draws that share one y_A draw (K = m: every draw has its own y_A draw)
  CRN            common random numbers: every arm draws at the same uniforms per site (sampling.py)
  paired arms    chained and structured_ya: they use the same K y_A draws in the same order, so they
                 differ only in the formula (paired_ya_index maps each draw to its y_A draw)

Files per configuration (tag = config_tag(fold, level, scheme, rep), e.g. f0_group_0.03_r1) in
results/<name>/: rank_<tag>.parquet (Figure-1 rank correlations), joint_<tag>.parquet (energy and
variogram scores), sites_<tag>.parquet (per-site CRPS, log CRPS, PIT, coverage), and meta_<tag>.json
(sizes, diagnostics, binding shares and seconds), written in this order and each atomically: the meta
file is the completion marker. Per run: config.yaml, code_version.txt, observed_violation.json and, if a
configuration failed, errors.log.

Seeding (every random stream is seeded locally, never from global state):
  * site uniforms: sampling.site_uniforms(profile id, salt), salts from _salt("target"|"eps"|"ya"|"gate",...)
  * thinning: numpy Generators seeded with _salt("group"|"random", fold, level, rep)
  * learner models: seed = fold * 1000 + rep * 10 (learner_seed), the same for every level and scheme
  * the abundant forest and the regime gate: seed = fold
  * cfg.seed: only the fold layout (assign_folds) and the `subsample` draw
So the QRF arms of a run do not depend on which other learners are in the config, and splitting the code
into functions cannot change a result unless it changes the order of draws from one Generator.
Reproducibility: QRF and PTF arms, the abundant model, the gate, thinning, folds, the analysis and the
bootstrap are bit-identical run to run. TabM training is not (even on CPU: torch.manual_seed is set, but
no deterministic algorithms or thread settings), so TabM arms and any average that includes TabM vary
slightly on a rerun. TabICL / fine-tuned TabICL determinism on GPU has not been checked.

Resuming: a configuration whose meta file exists is skipped, so an interrupted or partly failed run is
continued by running the same command again; a configuration that raises is logged to errors.log and
retried on the next run (the CLI exits 1 while any failed). A results folder refuses a changed config.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from .data import load_table
# the design lives in design.py (no torch import); re-exported here, where it was defined before
from .design import (LEARNER_ORDER, PAIRED_ARMS, Config, _salt, assign_folds, config_tag,  # noqa: F401
                     configurations, is_paired, paired_ya_index)
from .learners import LEARNERS, ForestSampler, RegimeClassifier, free_memory
from .paths import CONFIGS, RESULTS  # noqa: F401  (CONFIGS: re-exported)
from .relations import COVS, RELATIONS, TARGETS, blogit, inv_blogit, weighted_median, ya_inverse, ya_transform
from .sampling import site_uniforms
from .scoring import coverage, crps_fair, energy_score, pit, variogram_score
from .thinning import group_thin, label_pattern, random_thin


def _wquant(v, w, u):
    """Weighted empirical quantile function evaluated at uniforms u (any shape).
    Deliberately not merged with relations.weighted_median: this one normalises the cumulative weights
    before the search, that one compares with half the unnormalised total, and at ties the two can pick
    different elements, so merging them would change results in the last bits."""
    o = np.argsort(v)
    cw = np.cumsum(np.asarray(w, float)[o])
    cw /= cw[-1]
    return np.asarray(v)[o][np.clip(np.searchsorted(cw, u), 0, len(v) - 1)]


def _spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties); NaN for fewer than 3 values or a constant input."""
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 and ra.std() > 0 and rb.std() > 0 else float("nan")


def _replace(tmp, path, tries=6):
    """os.replace with retries: on Windows an antivirus or indexer can briefly lock a new file."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1.0)


def _atomic_parquet(df, path):
    """Write a parquet file atomically (temporary file, then rename): no half-written result files."""
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    _replace(tmp, path)


def _atomic_json(obj, path):
    """Write a JSON file atomically (temporary file, then rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    _replace(tmp, path)


_release = free_memory                     # free (GPU) memory between models (learners.free_memory)

NEURAL = {L for L in LEARNER_ORDER if L != "qrf"}  # learners that need a CUDA GPU unless allow_cpu


# ----------------------------------------------------------------------------- one target
N_RESIDUAL_BINS = 10            # residuals are drawn per decile of the driver (clay for CEC7, OC for N)
MIN_CALC_FOR_PTF = 10           # calcareous training rows needed for the calcareous-regime PTF parameter

# Names used in run_target and its helpers (n test sites, m draws, K = k_chain):
#   u_draw          (n, m) site uniforms of every arm (common random numbers)
#   u_chain         the same uniforms as (n K, m / K): the draws of chained-type models, K y_A draws per site
#   u_eps           (n, m) site uniforms of the residual draws
#   clay_d, oc_d    (n, m) clay and OC of the joint y_A draws at the test sites
#   ya_idx_chained  (m,) y_A draw of each draw of a chained-type arm (paired_ya_index)
#   theta_draws     (n, m) structured parameter from x;  theta_ya_draws: from (x, y_A), paired like chained
#   eps_d, eps_c, eps_o  residuals at the y_A draws, at the paired y_A draws, at the observed clay / OC


def learner_seed(fold, rep):
    """Seed of every learner model of a configuration: the same at every level, scheme and target."""
    return fold * 1000 + rep * 10


def _push(rel, theta, clay, oc, eps):
    """Target draws from parameter draws: the relation, then the log-scale residual."""
    return rel.add_residual(rel.forward(theta, clay, oc), eps)


class _ThetaTargets(NamedTuple):
    theta: np.ndarray               # parameter implied by each training row
    fit_rows: np.ndarray            # rows the parameter models are fitted on (non-calcareous when gated)
    z: np.ndarray                   # blogit(theta): the learning target of the structured models
    w_fit: np.ndarray | None        # case weights of the fit rows (None: unweighted)
    theta_clipped: np.ndarray       # theta clipped to its bounds (PTF baselines)
    w_all: np.ndarray               # case weights of every row (ones when unweighted)


def _theta_targets(rel, tr, y, cfg, gated):
    """Parameter targets and weights. Calcareous rows are exempt from the CEC relation (NH4OAc artefacts)
    when gated, unless fewer than min_train rows would remain."""
    clay_tr, oc_tr = tr.clay.values, tr.oc.values
    theta = rel.implied(y, clay_tr, oc_tr)
    fit_rows = ~tr["calc"].values if gated else np.ones(len(tr), bool)
    if fit_rows.sum() < cfg.min_train:
        fit_rows = np.ones(len(tr), bool)
    z = blogit(theta, rel.lo, rel.hi)
    w = rel.weights(clay_tr, cfg.cec_weight_power)
    wf = None if w is None else w[fit_rows]
    thc = np.clip(theta, rel.lo, rel.hi)
    wt = np.ones(len(tr)) if w is None else w
    return _ThetaTargets(theta, fit_rows, z, wf, thc, wt)


def _residual_sampler(rel, target, y, clay_tr, oc_tr, fit_rows, u_eps, enabled):
    """Residual draws: a draw with driver value v (clay for CEC7, OC for N) gets the u_eps quantile of the
    training residuals (fit rows) in the same driver decile as v. Returns (eps_for(driver), residuals);
    eps_for returns zeros when the residual is disabled."""
    res = rel.residual(y, clay_tr, oc_tr)[fit_rows]
    drv_tr = (clay_tr if target == "cec" else oc_tr)[fit_rows]
    edges = np.quantile(drv_tr, np.linspace(0, 1, N_RESIDUAL_BINS + 1)[1:-1])

    def eps_for(driver):
        if not enabled:
            return np.zeros(driver.shape)
        bt, bd = np.searchsorted(edges, drv_tr), np.searchsorted(edges, driver)
        e = np.zeros(driver.shape)
        for k in range(N_RESIDUAL_BINS):
            rk, mk = res[bt == k], bd == k
            if len(rk) and mk.any():
                e[mk] = np.quantile(rk, u_eps[mk])
        return e
    return eps_for, res


def _ptf_arms(rel, tt, calc_tr, gated, calc_draw, u_draw, clay_d, oc_d, eps_d):
    """Learner-free baselines (regime-specific): ptf = one weighted-median parameter, ptf_marg = the
    parameter drawn from its weighted training distribution at u_draw. With the gate, predicted
    calcareous draws use the calcareous rows' parameter (without residual)."""
    fr, thc, wt = tt.fit_rows, tt.theta_clipped, tt.w_all
    ptf = _push(rel, weighted_median(thc[fr], wt[fr]), clay_d, oc_d, eps_d)
    ptf_marg = _push(rel, _wquant(thc[fr], wt[fr], u_draw), clay_d, oc_d, eps_d)
    if gated and calc_tr.sum() >= MIN_CALC_FOR_PTF:
        cm = calc_tr
        ptf = np.where(calc_draw, rel.forward(weighted_median(thc[cm], wt[cm]), clay_d, oc_d), ptf)
        ptf_marg = np.where(calc_draw, rel.forward(_wquant(thc[cm], wt[cm], u_draw), clay_d, oc_d), ptf_marg)
    return {"ptf": ptf, "ptf_marg": ptf_marg}


def _learner_params(params, level):
    """(active at this level, constructor settings): a settings key `levels: [...]` restricts a learner to
    those levels; it is removed from the settings passed to the learner."""
    params = dict(params or {})
    lv_only = params.pop("levels", None)
    return lv_only is None or any(abs(level - v) < 1e-9 for v in lv_only), params


def _chain_inputs(X_te, ya_te, K):
    """Inputs of the chained-type models at the test sites: every site repeated K times with its first K
    y_A draws, shape (n K, p + 3)."""
    return np.hstack([np.repeat(X_te, K, axis=0), ya_te[:, :K, :].reshape(-1, 3)])


def _fit_free(L, seed, params, X_tr, y, groups, X_te, u_draw):
    """L:free draws (n, m): log target from the covariates only."""
    return np.exp(L(seed=seed, **params).fit(X_tr, np.log(y), groups=groups).sample(X_te, u_draw))


@dataclass
class _GateFallback:
    """Chained draws for the cells the CEC7 regime gate sends to the direct (chained) model."""
    ii: np.ndarray                  # (site, draw) cells with a predicted calcareous regime ...
    jj: np.ndarray
    fallback_pred: np.ndarray       # ... and their chained draws (same y_A draw and uniform as that draw)
    io: np.ndarray                  # every draw of the observed calcareous sites (structured_og) ...
    jo: np.ndarray
    fallback_obs: np.ndarray        # ... and their chained draws
    ic: np.ndarray                  # observed calcareous sites (oracle_ya) ...
    oracle_calc: np.ndarray | None  # ... chained draws at their observed y_A (None if there are none)


def _chained_at(chained_model, chained, X_te, ya_te, u_draw, ii, jj, K, m):
    """Chained draws at (site, draw) cells, each at the y_A draw and uniform of that cell."""
    if len(ii) == 0:
        return np.empty(0)
    if K == m:                                    # then the chained arm already holds exactly these
        return chained[ii, jj]
    return np.exp(chained_model.sample(np.hstack([X_te[ii], ya_te[ii, jj]]), u_draw[ii, jj][:, None])[:, 0])


def _fit_chained(L, seed, params, X_tr, ya_tr_t, y, groups, X_te, X_chain, ya_te, ya_obs_t, u_draw, u_chain,
                 calc_draw, calc_obs, gated, K):
    """L:chained draws (n, m), and the gate's fallback draws (None if not gated). Every use of the chained
    model (incl. the calcareous fallbacks) happens here, so the model is released when this returns."""
    n, m = u_draw.shape
    chained_model = L(seed=seed, **params).fit(np.hstack([X_tr, ya_tr_t]), np.log(y), groups=groups)
    chained = np.exp(chained_model.sample(X_chain, u_chain).reshape(n, m))
    if not gated:
        return chained, None
    # chained draws at every cell the predicted (calc_draw) or observed (calc_obs) regime sends to the
    # direct model; the union is predicted once (each cell's draw is deterministic)
    fu = np.full((n, m), np.nan)
    iu, ju = np.nonzero(calc_draw | calc_obs[:, None])
    fu[iu, ju] = _chained_at(chained_model, chained, X_te, ya_te, u_draw, iu, ju, K, m)
    ii, jj = np.nonzero(calc_draw)
    io, jo = np.nonzero(np.repeat(calc_obs[:, None], m, 1))
    ic = np.nonzero(calc_obs)[0]                  # oracle at calcareous sites: chained at observed y_A
    oracle_calc = (np.exp(chained_model.sample(np.hstack([X_te[ic], ya_obs_t[ic]]), u_draw[ic])) if len(ic)
                   else None)
    return chained, _GateFallback(ii, jj, fu[ii, jj], io, jo, fu[io, jo], ic, oracle_calc)


def _fit_theta(L, seed, params, rel, X_tr, tt, groups, X_te, u_draw):
    """L:structured parameter model (theta from x on the fit rows, case-weighted): theta draws (n, m) and
    the model's diagnostics {ess, epochs, ft_gain} (NaN where a learner has none)."""
    fr = tt.fit_rows
    theta_model = L(seed=seed, **params).fit(X_tr[fr], tt.z[fr], w=tt.w_fit, groups=groups[fr])
    theta_draws = inv_blogit(theta_model.sample(X_te, u_draw), rel.lo, rel.hi)
    diag = {k: float(getattr(theta_model, a, np.nan)) for k, a in (("ess", "ess_"), ("epochs", "epochs_"),
                                                                   ("ft_gain", "ft_gain_"))}
    return theta_draws, diag


def _fit_theta_ya(L, seed, params, rel, X_tr, ya_tr_t, tt, groups, X_chain, u_chain, n, m):
    """L:structured_ya parameter model (theta from x and y_A): theta draws (n, m), paired like chained."""
    fr = tt.fit_rows
    theta_ya_model = L(seed=seed, **params).fit(np.hstack([X_tr, ya_tr_t])[fr], tt.z[fr], w=tt.w_fit,
                                                groups=groups[fr])
    return inv_blogit(theta_ya_model.sample(X_chain, u_chain).reshape(n, m), rel.lo, rel.hi)


def _combine_arms(rel, te, free, chained, theta_draws, theta_ya_draws, clay_d, oc_d, ya_idx_chained, eps_d, eps_c,
                  eps_o, envelope, calc_draw, fallback):
    """The arms of one learner from its four models' draws, in output order: [structured_og], free,
    chained, structured, structured_ya, oracle_ya, clip. `fallback` (gated CEC7 only) puts the chained draw
    at calcareous cells."""
    struct_rel = _push(rel, theta_draws, clay_d, oc_d, eps_d)
    struct_ya = _push(rel, theta_ya_draws, clay_d[:, ya_idx_chained], oc_d[:, ya_idx_chained], eps_c)
    oracle = _push(rel, theta_draws, te.clay.values[:, None], te.oc.values[:, None], eps_o)
    clip = np.clip(free, *envelope)
    struct = struct_rel.copy()
    arms = {}
    if fallback is not None:
        fb = fallback
        struct[fb.ii, fb.jj] = fb.fallback_pred
        clip[fb.ii, fb.jj] = fb.fallback_pred
        struct_ya = np.where(calc_draw, chained, struct_ya)    # identical pairing (ya_idx_chained)
        og = struct_rel.copy()
        og[fb.io, fb.jo] = fb.fallback_obs
        arms["structured_og"] = og
        if fb.oracle_calc is not None:
            oracle[fb.ic] = fb.oracle_calc
    arms.update({"free": free, "chained": chained, "structured": struct, "structured_ya": struct_ya,
                 "oracle_ya": oracle, "clip": clip})
    return arms


def _binding_meta(meta, out, rel, te, clay_d, oc_d, ya_idx_chained, calc_draw, calc_obs, gated):
    """Binding shares, added to meta in arm order: paired arms against the envelope of their own y_A draws
    (draws of the predicted non-calcareous regime), free against the OBSERVED clay/OC (diagnostic)."""
    n, m = calc_draw.shape
    ok = ~calc_draw if gated else np.ones((n, m), bool)
    lo_obs, hi_obs = rel.envelope(te.clay.values[:, None], te.oc.values[:, None])
    for arm, d in out.items():
        if is_paired(arm):
            lo_a, hi_a = rel.envelope(clay_d[:, ya_idx_chained], oc_d[:, ya_idx_chained])
            meta[f"binding[{arm}]"] = float(np.mean(((d < lo_a) | (d > hi_a))[ok]))
        elif arm.endswith(":free"):
            keep = ~calc_obs if gated else np.ones(n, bool)
            meta[f"binding_obs[{arm}]"] = float(np.mean(((d < lo_obs) | (d > hi_obs))[keep]))


def run_target(target, cfg, tr, te, ya_te, calc_draw, fold, rep, level):
    """Fit every arm for one sparse target in one configuration and draw m values per test site.

    target     "n" or "cec"
    tr, te     training rows (thinned, with this target observed) and test rows (target observed)
    ya_te      (n_te, m, 3) joint y_A draws at the test sites (ya_transform scale)
    calc_draw  (n_te, m) bool: predicted calcareous regime per draw (all False without the gate)
    fold, rep, level  configuration (seeds and the per-learner level filter)
    Returns (out, meta): out = {arm: (n_te, m) draws} in a fixed order (ptf, ptf_marg, then per learner
    [structured_og], free, chained, structured, structured_ya, oracle_ya, clip), meta = sizes, learner
    diagnostics, seconds and binding shares, in a fixed key order."""
    rel = RELATIONS[target]
    m, K = cfg.m, cfg.k_chain
    assert m % K == 0, "k_chain must divide m"
    n = len(te)
    X_tr, X_te = tr[COVS].values, te[COVS].values
    y = tr[target].values
    clay_tr, oc_tr = tr.clay.values, tr.oc.values
    ya_tr_t = ya_transform(clay_tr, oc_tr, tr.ph.values)
    ya_obs_t = ya_transform(te.clay.values, te.oc.values, te.ph.values)
    clay_d, oc_d, _ = ya_inverse(ya_te)                                   # (n, m)
    groups = tr["block"].values
    u_draw = site_uniforms(te["profile_id"].values, m, _salt("target", target))
    u_eps = site_uniforms(te["profile_id"].values, m, _salt("eps", target))
    u_chain = u_draw.reshape(n, K, m // K).reshape(n * K, m // K)
    ya_idx_chained = paired_ya_index(":chained", m, K)
    calc_obs = te["calc"].values.astype(bool)
    gated = cfg.gate and target == "cec"

    tt = _theta_targets(rel, tr, y, cfg, gated)
    eps_for, res = _residual_sampler(rel, target, y, clay_tr, oc_tr, tt.fit_rows, u_eps, cfg.residual)
    drv = clay_d if target == "cec" else oc_d
    eps_d = eps_for(drv)
    eps_c = eps_for(drv[:, ya_idx_chained])
    eps_o = eps_for(np.repeat((te.clay.values if target == "cec" else te.oc.values)[:, None], m, 1))

    meta = {"n_train": int(len(tr)), "n_theta_fit": int(tt.fit_rows.sum()),
            "theta_outside_bounds": float(np.mean((tt.theta < rel.lo) | (tt.theta > rel.hi))),
            "residual_nonzero": float(np.mean(np.abs(res) > 1e-9))}
    out = _ptf_arms(rel, tt, tr["calc"].values, gated, calc_draw, u_draw, clay_d, oc_d, eps_d)

    envelope = rel.envelope(clay_d, oc_d)
    for name, params in cfg.learners.items():
        active, params = _learner_params(params, level)
        if not active:
            continue
        L, seed = LEARNERS[name], learner_seed(fold, rep)
        t0 = time.time()
        # models are fitted, used and freed one at a time, in this order (GPU memory: one TabICL context at a
        # time). Each helper returns draws only, so its model is unreferenced when it returns: release then.
        free = _fit_free(L, seed, params, X_tr, y, groups, X_te, u_draw)
        _release()
        X_chain = _chain_inputs(X_te, ya_te, K)
        chained, fallback = _fit_chained(L, seed, params, X_tr, ya_tr_t, y, groups, X_te, X_chain, ya_te, ya_obs_t,
                                         u_draw, u_chain, calc_draw, calc_obs, gated, K)
        _release()
        theta_draws, diag = _fit_theta(L, seed, params, rel, X_tr, tt, groups, X_te, u_draw)
        for k, v in diag.items():
            meta[f"{name}_{k}"] = v
        _release()
        theta_ya_draws = _fit_theta_ya(L, seed, params, rel, X_tr, ya_tr_t, tt, groups, X_chain, u_chain, n, m)
        _release()
        arms = _combine_arms(rel, te, free, chained, theta_draws, theta_ya_draws, clay_d, oc_d, ya_idx_chained,
                             eps_d, eps_c, eps_o, envelope, calc_draw, fallback)
        out.update({f"{name}:{kind}": d for kind, d in arms.items()})
        meta[f"{name}_seconds"] = round(time.time() - t0, 1)

    _binding_meta(meta, out, rel, te, clay_d, oc_d, ya_idx_chained, calc_draw, calc_obs, gated)
    return out, meta


# ----------------------------------------------------------------------------- one configuration
def _thinning_masks(train, fold, level, scheme, rep):
    """(keep, lab): the training profiles that keep their sparse labels, and the group mask whose regions
    define the retained / thinned strata (random rep r is scored against the regions of group rep r)."""
    if scheme == "all":
        keep, _ = group_thin(train, 1.0, 0)
        return keep, keep
    g_keep, _ = group_thin(train, level, _salt("group", fold, level, rep))
    keep = g_keep if scheme == "group" else random_thin(train, g_keep, _salt("random", fold, level, rep))
    return keep, g_keep


def _regime_draws(cfg, train, test, keep, fold, u_gate):
    """(n_test, m) predicted calcareous regime per draw: p(calcareous | x) from a classifier fitted on the
    surviving CEC rows of this configuration, compared with the gate uniforms. All False without the gate
    or with fewer than min_train such rows."""
    calc_draw = np.zeros((len(test), cfg.m), bool)
    if cfg.gate:
        gm = keep & train["cec"].notna().values
        if gm.sum() >= cfg.min_train:
            p = RegimeClassifier(seed=fold, **cfg.gate_model).fit(train.loc[gm, COVS].values,
                                                                  train.loc[gm, "calc"].values).proba(test[COVS].values)
            calc_draw = u_gate < p[:, None]
    return calc_draw


def _site_score_frame(cfg, fold, level, scheme, rep, target, arm, te, d, retained):
    """Per-site scores of one arm (the rows of sites_<tag>.parquet, fixed column order)."""
    y = te[target].values
    cl = paired_ya_index(arm, cfg.m, cfg.k_chain)
    cl_rank = cl if len(np.unique(cl)) < cfg.m else None     # PIT/coverage: one draw per y_A draw
    return pd.DataFrame({
        "fold": fold, "level": level, "scheme": scheme, "rep": rep, "target": target, "arm": arm,
        "site": te["profile_id"].values, "block": te["block"].values,
        "crps": crps_fair(y, d, cl),
        "crps_log": crps_fair(np.log(y), np.log(np.clip(d, 1e-9, None)), cl),
        "pit": pit(y, d, cl_rank), "cov90": coverage(y, d, 0.9, cl_rank), "retained": retained,
        "calc": te["calc"].values})


def _joint_scores(cfg, fold, level, scheme, rep, train, test, ya_t, draws_by_target):
    """Energy and variogram scores of the vector (clay, OC, pH, N, CEC7) where both sparse targets are
    observed (at least 10 sites), standardised with training-fold moments; one frame per arm."""
    joint = []
    if len(draws_by_target) == 2:
        (i_n, d_n), (i_c, d_c) = draws_by_target["n"], draws_by_target["cec"]
        both = np.intersect1d(i_n, i_c)
        if len(both) >= 10:
            pn, pc = np.searchsorted(i_n, both), np.searchsorted(i_c, both)
            tb = test.iloc[both]
            obs = np.column_stack([ya_transform(tb.clay, tb.oc, tb.ph), np.log(tb.n), np.log(tb.cec)])
            ref = train.dropna(subset=["n", "cec"])        # fixed score scale within a fold (all configs)
            ref_t = np.column_stack([ya_transform(ref.clay, ref.oc, ref.ph), np.log(ref.n), np.log(ref.cec)])
            mu, sd = ref_t.mean(0), ref_t.std(0) + 1e-9
            for arm in sorted(set(d_n) & set(d_c)):
                if arm.endswith(":oracle_ya"):
                    continue
                idx = paired_ya_index(arm, cfg.m, cfg.k_chain)
                fct = np.concatenate([ya_t[both][:, idx, :], np.log(d_n[arm][pn])[..., None],
                                      np.log(d_c[arm][pc])[..., None]], -1)
                o, f = (obs - mu) / sd, (fct - mu) / sd
                joint.append(pd.DataFrame({"fold": fold, "level": level, "scheme": scheme, "rep": rep, "arm": arm,
                                           "site": tb["profile_id"].values, "block": tb["block"].values,
                                           "es": energy_score(o, f, idx), "vs": variogram_score(o, f, 0.5, idx)}))
    return joint


def _rank_rows(cfg, fold, level, scheme, rep, test, ya_t, draws_by_target):
    """Figure-1 check (the project's motivating figure): rank correlation of each sparse target with its
    abundant driver (CEC7-clay, N-OC) at the held-out sites: observed pairs, pooled joint draws (each draw
    with the y_A draw it was made with) and the median "map" (per-site medians); one row per arm."""
    rank = []
    for target, (i_t, d_t) in draws_by_target.items():
        j = 0 if target == "cec" else 1                    # logit clay / log OC: monotone, same ranks
        te_t = test.iloc[i_t]
        rho_obs = _spearman(te_t["clay" if target == "cec" else "oc"].values, te_t[target].values)
        for arm, d in d_t.items():
            if arm.endswith(":oracle_ya"):
                continue
            drv = ya_t[i_t][:, paired_ya_index(arm, cfg.m, cfg.k_chain), j]
            rank.append({"fold": fold, "level": level, "scheme": scheme, "rep": rep, "target": target, "arm": arm,
                         "n_sites": len(i_t), "rho_obs": rho_obs, "rho_draws": _spearman(drv.ravel(), d.ravel()),
                         "rho_median": _spearman(np.median(drv, 1), np.median(d, 1))})
    return rank


def _config_meta(fold, level, scheme, rep, keep, pat, calc_draw, meta_all):
    """The meta_<tag>.json record: design, kept-label counts, gate share, then each target's meta."""
    return {"fold": fold, "level": level, "scheme": scheme, "rep": rep,
            "n_kept_profiles": int(keep.sum()),
            **{f"n_kept_pattern{p}": int((keep & (pat == p)).sum()) for p in (1, 2, 3)},
            "calc_draw_share": float(calc_draw.mean()),
            **{f"{t}:{k}": v for t, mm in meta_all.items() for k, v in mm.items()}}


def _write_config_outputs(out_dir, tag, rank, joint, rows, meta_record):
    """Write rank_, joint_ and sites_<tag>.parquet (when not empty), then meta_<tag>.json LAST: the meta
    file is the completion marker, so a configuration interrupted before it is rerun."""
    if rank:
        _atomic_parquet(pd.DataFrame(rank), out_dir / f"rank_{tag}.parquet")
    if joint:
        _atomic_parquet(pd.concat(joint), out_dir / f"joint_{tag}.parquet")
    if rows:
        _atomic_parquet(pd.concat(rows), out_dir / f"sites_{tag}.parquet")
    _atomic_json(meta_record, out_dir / f"meta_{tag}.json")


def run_config(cfg, fold, level, scheme, rep, shared, out_dir):
    """Thin the training labels, fit the regime gate, run both targets and score every arm for one
    configuration; write rank_, joint_, sites_ and finally meta_<tag> to out_dir. `shared` =
    (train, test, ya_t, U_gate) of the fold. Returns the tag, or "skip" if the meta file already exists."""
    tag = config_tag(fold, level, scheme, rep)
    if (out_dir / f"meta_{tag}.json").exists():            # meta is written last: completion marker
        return "skip"
    train, test, ya_t, U_gate = shared
    keep, lab = _thinning_masks(train, fold, level, scheme, rep)
    calc_draw = _regime_draws(cfg, train, test, keep, fold, U_gate)

    rows, meta_all, draws_by_target = [], {}, {}
    pat = label_pattern(train)
    for target in TARGETS:
        tr = train[keep & train[target].notna().values]
        te_mask = test[target].notna().values
        te = test[te_mask].reset_index(drop=True)
        if len(tr) < cfg.min_train or len(te) < 10:
            continue
        draws, meta = run_target(target, cfg, tr, te, ya_t[te_mask], calc_draw[te_mask], fold, rep, level)
        # realised share of this target's training labels (the level is a share of profiles with ANY sparse
        # label; N is missing by region, so its realised share varies between replicates)
        meta["frac_kept"] = float(len(tr) / max(1, train[target].notna().sum()))
        meta_all[target] = meta
        draws_by_target[target] = (np.nonzero(te_mask)[0], draws)
        kept_regions = set(train.loc[lab & train[target].notna().values, "region"])
        retained = te["region"].isin(kept_regions).values if scheme != "all" else np.ones(len(te), bool)
        for arm, d in draws.items():
            rows.append(_site_score_frame(cfg, fold, level, scheme, rep, target, arm, te, d, retained))

    joint = _joint_scores(cfg, fold, level, scheme, rep, train, test, ya_t, draws_by_target)
    rank = _rank_rows(cfg, fold, level, scheme, rep, test, ya_t, draws_by_target)
    _write_config_outputs(out_dir, tag, rank, joint, rows,
                          _config_meta(fold, level, scheme, rep, keep, pat, calc_draw, meta_all))
    return tag


def _check_device(cfg):
    """Refuse neural learners without a CUDA GPU unless cfg.allow_cpu."""
    neural = [k for k in cfg.learners if k in NEURAL]
    if not neural or cfg.allow_cpu:
        return
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if not cuda:
        raise RuntimeError(f"{cfg.name}: learners {neural} need a CUDA GPU (torch sees none). Install a CUDA "
                           "build of torch (README, Installation) or set allow_cpu: true for a CPU smoke test.")


def _check_resume(cfg, out_dir):
    """Refuse to resume a results folder that was produced with a different configuration."""
    p = out_dir / "config.yaml"
    if not p.exists() or not any(out_dir.glob("meta_*.json")):
        return
    # through the dataclass, so fields added later take their defaults on both sides
    old = yaml.safe_load(yaml.safe_dump(asdict(Config.load(p)), sort_keys=False))
    new = yaml.safe_load(yaml.safe_dump(asdict(cfg), sort_keys=False))
    diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if diff:
        raise RuntimeError(f"{out_dir} holds results of a different configuration (changed: {diff}). "
                           "Use a new config name, or delete the folder to start over.")


LAST_FAILED: list = []                     # configurations that failed in the last run() (the CLI exits 1)


def _code_version():
    """Last commit that changed the pipeline code (src/pedopilot), so commits of results or docs during a
    run do not look like a code change; '+local-changes' if the package has uncommitted edits."""
    try:
        import subprocess
        r = subprocess.run(["git", "log", "-1", "--format=%h", "--", "."], capture_output=True, text=True,
                           timeout=10, cwd=Path(__file__).resolve().parent)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                               timeout=10, cwd=Path(__file__).resolve().parent).stdout.strip()
        return (r.stdout.strip() or "unknown") + ("+local-changes" if dirty else "")
    except Exception:                                   # noqa: BLE001  (no git: not fatal)
        return "unknown"


def _prepare_table(cfg, df=None):
    """The profile table (the snapshot, or `df` for synthetic runs), optionally subsampled (seeded with
    cfg.seed), with the `fold` column added."""
    df = load_table() if df is None else df.copy()
    if cfg.subsample:
        df = df.sample(frac=cfg.subsample, random_state=cfg.seed).reset_index(drop=True)
    df["fold"] = assign_folds(df, cfg.n_folds, cfg.seed)
    return df


def _print_design(cfg, df, folds, confs):
    """Print the size of the design: profiles, folds, configurations, learners and the test fold sizes."""
    print(f"{cfg.name}: {len(df)} profiles, {len(folds)} folds x {len(confs)} configurations, "
          f"learners {list(cfg.learners)}")
    for f in folds:
        te = df[df.fold == f]
        print(f"  fold {f}: test {len(te)} profiles ({te.n.notna().sum()} N, {te.cec.notna().sum()} CEC7), "
              f"{te.block.nunique()} blocks")


def _record_code_version(out_dir):
    """Append the pipeline code version to code_version.txt; warn if it changed since the run started."""
    ver, vfile = _code_version(), out_dir / "code_version.txt"
    old_ver = vfile.read_text().split()[-1] if vfile.exists() and vfile.read_text().strip() else None
    if old_ver and old_ver != ver:
        print(f"WARNING: {out_dir.name} was started with pipeline code {old_ver}, now {ver} (git log -- src/pedopilot): "
              f"results may mix code versions (use a new config name after changing the code)", flush=True)
    with open(vfile, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {ver}\n")


def _observed_violation(cfg, df):
    """Share of observed N / CEC7 values outside the envelope of their own clay and OC (non-calcareous
    CEC7 rows when gated): the reference for the binding shares."""
    obs = {}
    for t in TARGETS:
        d = df.dropna(subset=[t])
        d = d[~d["calc"]] if t == "cec" and cfg.gate else d
        lo, hi = RELATIONS[t].envelope(d.clay.values, d.oc.values)
        obs[t] = float(np.mean((d[t].values < lo) | (d[t].values > hi)))
    return obs


def _fold_inputs(cfg, df, fold):
    """Shared inputs of every configuration of a fold: (train, test, ya_t, U_gate). ya_t = (n_test, m, 3)
    joint y_A draws of the abundant forest (seed = fold); U_gate = the site uniforms of the regime gate."""
    train = df[df.fold != fold].reset_index(drop=True)
    test = df[df.fold == fold].reset_index(drop=True)
    ya_model = ForestSampler(seed=fold, **cfg.abundant).fit(
        train[COVS].values, ya_transform(train.clay.values, train.oc.values, train.ph.values))
    ya_t = ya_model.sample_joint(test[COVS].values, site_uniforms(test["profile_id"].values, cfg.m, _salt("ya")))
    U_gate = site_uniforms(test["profile_id"].values, cfg.m, _salt("gate"))
    return train, test, ya_t, U_gate


def _run_one_config(cfg, fold, level, scheme, rep, shared, out_dir, t_start, failed):
    """run_config with the per-configuration error handling: an exception (e.g. CUDA out of memory) is
    logged to errors.log, the tag is added to `failed`, and the run continues with the next configuration."""
    t0, tag = time.time(), config_tag(fold, level, scheme, rep)
    try:
        if run_config(cfg, fold, level, scheme, rep, shared, out_dir) != "skip":
            print(f"[{time.time() - t_start:7.0f}s] {tag} done in {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:                        # e.g. CUDA out of memory: log, free, continue
        failed.append(tag)
        with open(out_dir / "errors.log", "a", encoding="utf-8") as fh:
            fh.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {tag}\n{traceback.format_exc()}\n")
        print(f"[{time.time() - t_start:7.0f}s] {tag} FAILED ({type(e).__name__}: {str(e)[:200]}); "
              f"see errors.log", flush=True)


def run(cfg: Config, only_folds=None, dry_run=False, df=None, results_root=None) -> Path | None:
    """Run all configurations (resumable). `df` and `results_root` allow synthetic validation runs.
    A configuration that raises is logged to errors.log and skipped; rerunning retries it.
    only_folds overrides cfg.folds; dry_run prints the design and returns None. Returns the results
    folder; the tags of failed configurations are in LAST_FAILED."""
    df = _prepare_table(cfg, df)
    confs = configurations(cfg)
    folds = only_folds if only_folds is not None else cfg.folds
    _print_design(cfg, df, folds, confs)
    if dry_run:
        return
    _check_device(cfg)
    out_dir = Path(results_root or RESULTS) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_resume(cfg, out_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    _record_code_version(out_dir)
    (out_dir / "observed_violation.json").write_text(json.dumps(_observed_violation(cfg, df), indent=1))

    t_start, failed = time.time(), []
    for fold in folds:
        todo = [c for c in confs if not (out_dir / f"meta_{config_tag(fold, *c)}.json").exists()]
        if not todo:
            continue
        shared = _fold_inputs(cfg, df, fold)
        for level, scheme, rep in todo:
            _run_one_config(cfg, fold, level, scheme, rep, shared, out_dir, t_start, failed)
            _release()
    print(f"finished {cfg.name} in {time.time() - t_start:.0f}s"
          + (f"; {len(failed)} FAILED (rerun the same command to retry): {failed}" if failed else ""))
    LAST_FAILED[:] = failed
    return out_dir
