"""Synthetic soil tables with a KNOWN answer, in the same format as data/conus_0_30.parquet.

scenario="relation"  (positive control): N = OC / r(x) and CEC = a(x) * clay + 0.35 * OC hold exactly
                      up to small lab noise, with smooth spatial parameters r(x), a(x). Calcareous profiles
                      break the CEC relation (an NH4OAc-type artefact). The structured arms should win,
                      most clearly when the sparse labels are thinned.
scenario="shared"     (borrowing without a relation): N and CEC are NOT related to clay/OC given the
                      covariates, but share covariate drivers with them. A structured target can still gain
                      here (dividing by a well-predicted abundant correlate stabilises it): a gain is then
                      borrowing strength, not evidence for the pedological formula.
scenario="independent" (true negative control): N and CEC depend on drivers that are uncorrelated with the
                      drivers of clay and OC. The formula must NOT beat the chained-free arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import grid_id
from .relations import B, COVS


def _sig(v):
    return 1 / (1 + np.exp(-v))


def make_table(n: int = 6000, scenario: str = "relation", seed: int = 0, p_n: float = 0.35,
               p_cec: float = 0.65) -> pd.DataFrame:
    """A synthetic profile table with the columns of the real one. n profiles on a 3000 x 1500 km plane;
    p_n and p_cec are the shares of profiles with an N label (80% of the regions have N at all, the rest
    of the missingness is random) and with a CEC7 label. All draws come from one numpy Generator seeded
    with `seed`, in a fixed order. Profile ids are offset by 10,000,000 per scenario: the site uniforms
    (sampling.site_uniforms) are seeded by profile id, so the scenarios do not share random numbers."""
    rng = np.random.default_rng(seed)
    x_km, y_km = rng.uniform(0, 3000, n), rng.uniform(0, 1500, n)
    # 22 smooth covariates: random sinusoids of location plus noise (climate/terrain-like fields)
    cov = np.empty((n, len(COVS)))
    for j in range(len(COVS)):
        f = rng.uniform(0.5, 3, 2) / 1000
        ph = rng.uniform(0, 2 * np.pi, 2)
        cov[:, j] = np.sin(f[0] * x_km + ph[0]) + np.cos(f[1] * y_km + ph[1]) + 0.3 * rng.normal(size=n)
    cov = (cov - cov.mean(0)) / cov.std(0)
    W = rng.normal(size=(len(COVS), 6)) / np.sqrt(len(COVS))
    z = cov @ W                                                   # 6 latent soil-forming drivers
    z, _ = np.linalg.qr(z - z.mean(0))                            # mutually uncorrelated drivers
    z = z / z.std(0)
    clay = 100 * _sig(-1.2 + 0.9 * z[:, 0] + 0.35 * rng.normal(size=n))
    oc = np.clip(np.exp(2.5 + 0.5 * z[:, 1] + 0.35 * rng.normal(size=n)), 0.3, 110)
    ph = np.clip(6.2 + 0.8 * z[:, 2] + 0.3 * rng.normal(size=n), 3.5, 9.5)
    calc = rng.random(n) < _sig(1.6 * z[:, 2] - 2.2)
    if scenario == "relation":
        r = 5 + 55 * _sig(-1.6 + 0.6 * z[:, 3] + 0.15 * rng.normal(size=n))        # C:N about 9-16
        a = 0.02 + 1.48 * _sig(-1.0 + 0.8 * z[:, 4] + 0.2 * rng.normal(size=n))    # clay activity
        n_ = oc / r * np.exp(0.05 * rng.normal(size=n))
        cec = (a * clay + B * oc) * np.exp(0.05 * rng.normal(size=n))
        cec = np.where(calc, cec * np.exp(0.4 + 0.2 * rng.normal(size=n)), cec)   # lab artefact
    elif scenario == "shared":
        # no relation given the covariates, but the same drivers as OC (z1) and clay (z0)
        n_ = np.exp(np.log(12 / 11) + 0.5 * z[:, 1] + 0.3 * z[:, 3] + 0.45 * rng.normal(size=n))
        cec = np.exp(np.log(16) + 0.35 * z[:, 0] + 0.3 * z[:, 4] + 0.45 * rng.normal(size=n))
    elif scenario == "independent":
        # drivers uncorrelated with those of clay (z0), OC (z1) and pH (z2)
        n_ = np.exp(np.log(12 / 11) + 0.5 * z[:, 5] + 0.3 * z[:, 3] + 0.45 * rng.normal(size=n))
        cec = np.exp(np.log(16) + 0.45 * z[:, 4] + 0.3 * z[:, 5] + 0.45 * rng.normal(size=n))
    else:
        raise ValueError(scenario)
    # missingness: partly by region (whole surveys lack an analysis), partly at random
    region = grid_id(x_km, y_km, 300)
    reg_n = {r_: rng.random() < 0.8 for r_ in np.unique(region)}
    has_n = np.array([reg_n[r_] for r_ in region]) & (rng.random(n) < p_n / 0.8)
    has_c = rng.random(n) < p_cec
    df = pd.DataFrame(cov, columns=COVS)
    df.insert(0, "profile_id", np.arange(1, n + 1) + 10_000_000 * ["relation", "shared", "independent"].index(scenario))
    df["clay"], df["oc"], df["ph"] = clay, oc, ph
    df["n"] = np.where(has_n, n_, np.nan)
    df["cec"] = np.where(has_c, cec, np.nan)
    df["calc"] = calc
    df["x_km"], df["y_km"] = x_km, y_km
    df["block"] = grid_id(x_km, y_km, 100)
    df["region"] = region
    df["dataset_id"], df["year"], df["X"], df["Y"] = f"SYN-{scenario}", 2000.0, x_km / 100, y_km / 100
    return df
