"""Pedological relations, bounded parameters and transforms.

    N   = OC / r                 r = C:N ratio,       bounded to [R_LO, R_HI]
    CEC = a * clay + B * OC      a = clay activity,   bounded to [A_LO, A_HI]

Units: clay in %, OC and N in g/kg, CEC7 in cmolc/kg.  B = 0.35 cmolc/kg per g/kg OC
(= 3.5 per % OC).  a is on the Soil Taxonomy CEC7/clay scale (0.24 = 24 cmolc per kg clay).

Glossary
  y_A        the ABUNDANT properties, measured almost everywhere: clay, OC and pH. They are modelled
             jointly on an unbounded scale, ya_transform = (logit(clay / 100), log OC, pH).
  target     a SPARSE property predicted through a relation: total N ("n") or CEC7 ("cec").
  theta      the relation's parameter implied by one observation (C:N ratio r for N, clay activity a for
             CEC7): Relation.implied. The structured arms predict theta, not the target.
  bounds     every theta is restricted to a pedologically plausible [lo, hi]. Learners fit the bounded
             logit blogit(theta) = log(p / (1 - p)), p = (theta - lo) / (hi - lo), which maps [lo, hi] to
             the real line, so a prediction pushed back with inv_blogit can never leave the bounds.
  envelope   the feasible range of the target given clay and OC: forward(lo) to forward(hi).
  residual   the part of an observation the bounded relation cannot express (non-zero only where the
             implied theta lies outside its bounds). It is multiplicative, log y - log forward(clip(theta)),
             for both targets: added back as y * exp(eps) it can never make a draw zero or negative, and
             it scales with the size of the prediction, as lab and method errors of N and CEC7 do.

The three constant lines below are also read as text by report builders: keep each on one line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-3
B = 0.35
R_LO, R_HI = 5.0, 60.0
A_LO, A_HI = 0.02, 1.5
COVS = [f"bio{i}" for i in range(1, 20)] + ["elev", "slope", "relief"]   # WorldClim bio1-19, elevation, slope, relief
TARGETS = ("n", "cec")


def blogit(v: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """Bounded logit: map [lo, hi] to the real line (values are clipped into the open interval)."""
    p = (np.clip(v, lo, hi) - lo) / (hi - lo)
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def inv_blogit(z: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """Inverse of blogit: the real line back to the open interval (lo, hi)."""
    return lo + (hi - lo) / (1 + np.exp(-np.asarray(z, float)))


def ya_transform(clay, oc, ph) -> np.ndarray:
    """Abundant properties to an unbounded scale: logit(clay/100), log(OC), pH."""
    c = np.clip(np.asarray(clay, float) / 100, EPS, 1 - EPS)
    return np.stack([np.log(c / (1 - c)), np.log(np.asarray(oc, float)), np.asarray(ph, float)], axis=-1)


def ya_inverse(t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ya_transform: (..., 3) array -> (clay %, OC g/kg, pH), each of shape (...)."""
    t = np.asarray(t, float)
    return 100 / (1 + np.exp(-t[..., 0])), np.exp(t[..., 1]), t[..., 2]


def weighted_median(v, w) -> float:
    """Weighted median: the smallest value whose cumulative weight reaches half the total weight."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][np.searchsorted(cw, 0.5 * cw[-1])]


@dataclass(frozen=True)
class Relation:
    """One structured target: how to compute the implied parameter from observations, its bounds,
    the case weights of the two-stage fit, and how to push parameter draws through the relation."""
    target: str
    lo: float
    hi: float

    def implied(self, y, clay, oc) -> np.ndarray:
        """Parameter implied by an observed target y: r = OC / N, or a = (CEC7 - B * OC) / clay."""
        if self.target == "n":
            return np.asarray(oc, float) / np.asarray(y, float)
        return (np.asarray(y, float) - B * np.asarray(oc, float)) / np.asarray(clay, float)

    def forward(self, theta, clay, oc) -> np.ndarray:
        """Target from a parameter (the relation itself): N = OC / r, or CEC7 = a * clay + B * OC."""
        if self.target == "n":
            return np.asarray(oc, float) / theta
        return theta * np.asarray(clay, float) + B * np.asarray(oc, float)

    def weights(self, clay, power=2.0):
        """Case weights for the parameter fit, so that an error in the parameter counts as much as the
        error it causes in the target: none for the multiplicative relation (fitted on the log scale);
        clay**power for CEC7 = a * clay + b * OC. power = 2 matches a squared-error (least-squares) target
        loss; power = 1 matches the CRPS and pinball losses of the learners used here, which scale
        linearly (CRPS(c F, c y) = c CRPS(F, y)), and keeps about twice the effective sample size."""
        if self.target == "n" or power == 0:
            return None
        return np.asarray(clay, float) ** power

    def envelope(self, clay, oc) -> tuple[np.ndarray, np.ndarray]:
        """Feasible [low, high] range of the target given clay and OC."""
        if self.target == "n":
            return np.asarray(oc, float) / self.hi, np.asarray(oc, float) / self.lo
        return self.forward(self.lo, clay, oc), self.forward(self.hi, clay, oc)

    def residual(self, y, clay, oc) -> np.ndarray:
        """Part of the observation the bounded relation cannot express (non-zero only where the
        implied parameter is outside its bounds). Multiplicative (log scale) for both targets, so a
        residual can never make a draw non-positive."""
        th = np.clip(self.implied(y, clay, oc), self.lo, self.hi)
        return np.log(np.asarray(y, float)) - np.log(self.forward(th, clay, oc))

    def add_residual(self, y, eps) -> np.ndarray:
        """Apply a (log-scale) residual to a prediction: y * exp(eps)."""
        return y * np.exp(eps)


RELATIONS = {"n": Relation("n", R_LO, R_HI), "cec": Relation("cec", A_LO, A_HI)}
