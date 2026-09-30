"""Scores for m-member draw ensembles.

The fair CRPS / energy score (Ferro 2014) assume independent members. Chained arms with
k_chain < m generate their m draws in k_chain groups that share one abundant (clay, OC, pH) draw;
pairs within a group are closer, which would inflate their score. `cluster` gives each member's
group; the spread term is then estimated from pairs in DIFFERENT groups only, which is unbiased for
grouped draws and identical to the ordinary fair estimator when all members are independent.

Calibration: `pit` is the mid-rank probability integral transform of the observation among the draws
(uniform for a calibrated ensemble), and `coverage` says whether the observation falls in the central
interval, computed from that rank. Both keep one draw per cluster, because dependent draws would make
the rank non-uniform even for a calibrated forecast.

All scores are per site (lower is better for CRPS, energy and variogram scores); the analysis turns
them into skill against the free QRF arm."""
from __future__ import annotations

import numpy as np
import scoringrules as sr


def _between(cluster, m: int) -> np.ndarray:
    """(m, m) mask of member pairs that belong to different clusters (all off-diagonal pairs if None)."""
    c = np.arange(m) if cluster is None else np.asarray(cluster)
    return c[:, None] != c[None, :]


def crps_fair(y, draws, cluster=None) -> np.ndarray:
    """Fair CRPS of an ensemble per site: y (n,), draws (n, m), cluster (m,) group of each member or None.
    Returns (n,). With clusters, the spread term E|X - X'| uses between-cluster pairs only."""
    y, d = np.asarray(y, float), np.asarray(draws, float)
    m = d.shape[1]
    if cluster is None or len(np.unique(cluster)) == m:
        return np.asarray(sr.crps_ensemble(y, d, estimator="fair"))
    b = _between(cluster, m)
    out = []
    for i in range(0, len(d), 256):
        c = d[i:i + 256]
        out.append(np.abs(c - y[i:i + 256, None]).mean(1) - 0.5 * np.abs(c[:, :, None] - c[:, None, :])[:, b].mean(1))
    return np.concatenate(out)


def energy_score(obs, fct, cluster=None) -> np.ndarray:
    """Fair energy score per site: obs (n, d), fct (n, m, d), cluster as in crps_fair. Returns (n,)."""
    obs, fct = np.asarray(obs, float), np.asarray(fct, float)
    m = fct.shape[1]
    if cluster is None or len(np.unique(cluster)) == m:
        return np.asarray(sr.es_ensemble(obs, fct, estimator="fair"))
    b = _between(cluster, m)
    out = []
    for i in range(0, len(fct), 64):
        c = fct[i:i + 64]
        out.append(np.linalg.norm(c - obs[i:i + 64, None], axis=-1).mean(1)
                   - 0.5 * np.linalg.norm(c[:, :, None] - c[:, None], axis=-1)[:, b].mean(1))
    return np.concatenate(out)


def variogram_score(obs, fct, p: float = 0.5, cluster=None) -> np.ndarray:
    """Fair variogram score (Scheuerer & Hamill 2015), sum over all ordered component pairs:
    VS = sum_{i,j} (|o_i - o_j|^p - E|X_i - X_j|^p)^2, with the squared expectation estimated from
    pairs of members in different groups (unbiased; equals scoringrules' 'fair' estimator when all
    members are independent). O(n m d^2) memory instead of O(n m^2 d^2)."""
    obs, fct = np.asarray(obs, float), np.asarray(fct, float)
    n, m, d = fct.shape
    iu, ju = np.triu_indices(d, 1)
    c = np.arange(m) if cluster is None else np.asarray(cluster)
    labels, c = np.unique(c, return_inverse=True)
    sizes = np.bincount(c)
    out = np.empty(n)
    for s in range(0, n, 512):
        o, f = obs[s:s + 512], fct[s:s + 512]
        a = np.abs(o[:, iu] - o[:, ju])[:, None, :] ** p - np.abs(f[:, :, iu] - f[:, :, ju]) ** p   # (b, m, P)
        tot = a.sum(1)
        A = np.zeros((len(o), len(labels), a.shape[2]))
        np.add.at(A, (slice(None), c), a)
        num = tot ** 2 - (A ** 2).sum(1)
        out[s:s + 512] = 2 * (num / (m * m - np.sum(sizes ** 2))).sum(-1)
    return out


def _one_per_cluster(draws: np.ndarray, cluster) -> np.ndarray:
    """Draws that share an abundant draw are dependent: keep one draw per cluster, so the rank of the
    observation among the kept draws is uniform under calibration."""
    if cluster is None:
        return draws
    _, first = np.unique(np.asarray(cluster), return_index=True)
    return draws[:, np.sort(first)]


def pit(y, draws, cluster=None) -> np.ndarray:
    """Mid-rank PIT on the (m + 1) grid: (#below + (#ties + 1) / 2) / (m + 1). For a calibrated
    ensemble of m exchangeable draws the rank of y among the m + 1 values is uniform, so the PIT is
    uniform on the grid {0.5, ..., m + 0.5} / (m + 1); coverage from it is unbiased. For histograms, jitter
    within the grid cell (analysis.figure_pit): equal bins over a coarse grid show binning artefacts."""
    d = _one_per_cluster(np.asarray(draws, float), cluster)
    y = np.asarray(y, float)[:, None]
    return (np.sum(d < y, 1) + 0.5 * (np.sum(d == y, 1) + 1)) / (d.shape[1] + 1)


def coverage(y, draws, level: float = 0.9, cluster=None) -> np.ndarray:
    """Central-interval coverage from the rank of y (unbiased for m exchangeable draws: 91/101 = 0.901
    at level 0.9 and m = 100). np.quantile intervals of 100 draws would cover only about 0.88."""
    p = pit(y, draws, cluster)
    a = (1 - level) / 2
    return (p >= a) & (p <= 1 - a)
