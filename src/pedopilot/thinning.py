"""Controlled label sparsity: keep the sparse labels (N, CEC7) of only a fraction of training profiles.

group  : keep whole 300 km regions until the target fraction is reached (mimics missing-by-source)
random : keep randomly chosen profiles with exactly the same numbers of N-only, CEC-only and
         both-labelled profiles as the group draw of the same replicate, so the N count, the CEC
         count and the union count all match (the matched-count control that separates
         "fewer labels" from "labels from other places")

Both return a boolean mask over the rows of the training table: True = the profile keeps its sparse
labels. Abundant properties (clay, OC, pH) and covariates are never thinned. The level is a share of the
profiles with ANY sparse label, so the realised share of one target varies (N is missing by region).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sparse_labelled(train: pd.DataFrame) -> np.ndarray:
    """Mask of profiles with at least one sparse label (N or CEC7)."""
    return (train["n"].notna() | train["cec"].notna()).values


def label_pattern(train: pd.DataFrame) -> np.ndarray:
    """1 = N only, 2 = CEC only, 3 = both, 0 = neither."""
    return train["n"].notna().values * 1 + train["cec"].notna().values * 2


def group_thin(train: pd.DataFrame, frac: float, seed: int) -> tuple[np.ndarray, set]:
    """Keep the sparse labels of whole regions until about `frac` of the labelled profiles are kept.

    Regions are visited in a random order (numpy Generator seeded with `seed`) and added while the kept
    count is below the target; a region that would overshoot the target by more than the remaining gap
    is skipped (unless nothing is kept yet). frac >= 1 keeps every labelled profile.
    Returns (mask of kept labelled profiles, set of kept region ids)."""
    has = sparse_labelled(train)
    if frac >= 1:
        return has, set(train["region"].unique())
    rng = np.random.default_rng(seed)
    target = int(round(frac * has.sum()))
    counts = train.loc[has, "region"].value_counts()
    kept, total = [], 0
    # add regions in random order; skip a region that would overshoot by more than the remaining gap
    for r in rng.permutation(counts.index.values):
        if total >= target:
            break
        if total + counts[r] - target > target - total and kept:
            continue
        kept.append(r)
        total += counts[r]
    return has & train["region"].isin(kept).values, set(kept)


def random_thin(train: pd.DataFrame, g_keep: np.ndarray, seed: int) -> np.ndarray:
    """Matched-count control for a group draw: for each label pattern (N only, CEC only, both) keep as
    many randomly chosen profiles, from anywhere in the training fold, as the group mask `g_keep` keeps.
    Returns the mask of kept profiles."""
    rng = np.random.default_rng(seed)
    pat = label_pattern(train)
    keep = np.zeros(len(train), bool)
    for p in (1, 2, 3):
        pool = np.nonzero(pat == p)[0]
        k = int((np.asarray(g_keep) & (pat == p)).sum())
        if k:
            keep[rng.choice(pool, size=k, replace=False)] = True
    return keep
