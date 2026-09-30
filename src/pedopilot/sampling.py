"""Turning quantile predictions into draws, with common random numbers.

Every arm is represented by m draws per site. Draws are made by inverse-CDF sampling at uniforms that
depend only on (site id, salt): the same site gets the same uniforms in every arm, fold and run
("common random numbers", CRN), so arm differences are not blurred by Monte Carlo noise.
"""
from __future__ import annotations

import numpy as np

LEVELS = np.linspace(0.01, 0.99, 99)       # the 99 quantile levels every neural learner predicts
U_LO, U_HI = 0.005, 0.995                  # uniforms are clipped to this range (tails extrapolated linearly)


def draws_from_quantiles(q: np.ndarray, levels: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draws from a (n, Q) quantile grid for given uniforms u of shape (n, m).

    Linear interpolation between levels, linear extrapolation of the two outermost segments
    into [U_LO, U_HI]; uniforms outside that range are clipped to it. Returns (n, m)."""
    q = np.sort(np.asarray(q, float), axis=1)
    levels = np.asarray(levels, float)
    u = np.clip(u, U_LO, U_HI)
    lo = q[:, :1] - (q[:, 1:2] - q[:, :1]) * (levels[0] - U_LO) / (levels[1] - levels[0])
    hi = q[:, -1:] + (q[:, -1:] - q[:, -2:-1]) * (U_HI - levels[-1]) / (levels[-1] - levels[-2])
    qq = np.hstack([lo, q, hi])
    lv = np.r_[U_LO, levels, U_HI]
    j = np.clip(np.searchsorted(lv, u), 1, len(lv) - 1)
    rows = np.arange(q.shape[0])[:, None]
    q0, q1 = qq[rows, j - 1], qq[rows, j]
    return q0 + (u - lv[j - 1]) / (lv[j] - lv[j - 1]) * (q1 - q0)


def site_uniforms(site_ids, m: int, salt: int) -> np.ndarray:
    """Common random numbers: the same uniforms for a site in every arm (seeded by site id and salt).

    site_ids must be integers (profile ids; they are reduced modulo 2**31). Returns (len(site_ids), m).
    Different salts (experiment._salt) give independent streams for different uses of the same site."""
    out = np.empty((len(site_ids), m))
    for i, s in enumerate(site_ids):
        out[i] = np.random.default_rng([int(s) % (2**31), int(salt)]).random(m)
    return out
