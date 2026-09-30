"""Region-level cluster bootstrap for a skill contrast (a check on the fold-level t-intervals, which
are somewhat optimistic because the folds share training data).

    python -m pedopilot.bootstrap cpu_qrf_w1 [--learner qrf] [--level 0.03] [--scheme group]
    python -m pedopilot bootstrap cpu_qrf_w1 [--learner qrf] [--level 0.03] [--scheme group]   (the same)

Held-out sites are grouped by 300 km region (the thinning unit); regions are resampled with
replacement, and the skill difference 1 - CRPS(a)/CRPS(ref) - (1 - CRPS(b)/CRPS(ref)) is recomputed per
(replicate, target) from CRPS pooled over folds (a ratio of sums, so the point estimate can differ slightly
from the report's fold-averaged value), then averaged. The fitted models stay fixed: the interval reflects
the spatial sampling of the test sites, not the dependence between folds that share training data.
Writes bootstrap_<a>_vs_<b>.json next to the results and prints the interval. Only complete folds are
used, as in `analyze`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import REF, complete_folds
from .paths import RESULTS

BOOTSTRAP_ARMS = ("structured_ya", "structured")      # each against chained, in this order


def region_of(block):
    """300 km region id of a 100 km block id "i_j" (floor division by 3, as the grid ids are built)."""
    i, j = (int(v) for v in str(block).split("_"))
    return f"{i // 3}_{j // 3}"


def bootstrap(name, learner="qrf", level=0.03, scheme="group", a="structured_ya", b="chained", n_boot=2000, seed=0,
              results_root=None) -> dict:
    """Region cluster bootstrap of skill(learner:a) - skill(learner:b) at one level and scheme (b = chained
    by default). n_boot multinomial region weights from a numpy Generator seeded with `seed`. Returns and
    writes {run, learner, level, scheme, contrast, estimate, lo, hi, n_regions, n_boot, folds, note}."""
    res = Path(results_root or RESULTS) / name
    files = sorted(res.glob("sites_*.parquet"))
    s = pd.concat([pd.read_parquet(f, columns=["fold", "level", "scheme", "rep", "target", "arm", "site", "block",
                                                "crps"]) for f in files], ignore_index=True)
    done = complete_folds(res)                   # as in `analyze`: leave out a stopped or failed fold
    if done:
        s = s[s.fold.isin(done)]
    arms = [f"{learner}:{a}", f"{learner}:{b}", REF]
    s = s[(s.level == level) & (s.scheme == scheme) & s.arm.isin(arms)]
    s["region"] = s.block.map(region_of)
    # per (rep, target, region): CRPS sums per arm and site counts (sites are scored once per rep)
    g = s.pivot_table(index=["rep", "target", "region"], columns="arm", values="crps", aggfunc="sum").fillna(0.0)
    cnt = s[s.arm == REF].groupby(["rep", "target", "region"]).size()
    g = g.join(cnt.rename("n"))
    regions = np.array(sorted(s.region.unique()))
    ridx = {r: k for k, r in enumerate(regions)}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(regions), np.ones(len(regions)) / len(regions), size=n_boot).astype(float)   # (B, R)

    def contrast(weights):                                   # weights: (R,) or (B, R)
        vals = []
        for (rep, t), gg in g.groupby(level=["rep", "target"]):
            ri = np.array([ridx[r] for r in gg.index.get_level_values("region")])
            w = weights[..., ri]
            A = (w * gg[arms[0]].to_numpy()).sum(-1)
            Bv = (w * gg[arms[1]].to_numpy()).sum(-1)
            R = (w * gg[REF].to_numpy()).sum(-1)
            vals.append((Bv - A) / R)                        # skill(a) - skill(b); site counts cancel
        return np.mean(vals, axis=0)

    est = float(contrast(np.ones(len(regions))))
    boot = contrast(W)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {"run": name, "learner": learner, "level": level, "scheme": scheme, "contrast": f"{a} vs {b}",
           "estimate": est, "lo": float(lo), "hi": float(hi), "n_regions": int(len(regions)), "n_boot": n_boot,
           "folds": sorted(int(f) for f in s.fold.unique()),
           "note": "region (300 km) cluster bootstrap of held-out sites; replicates and targets averaged"}
    (res / f"bootstrap_{a}_vs_{b}.json").write_text(json.dumps(out, indent=1))
    print(f"{name} {learner} {a} vs {b} at {level:.0%} {scheme}: {100 * est:+.1f} points, region bootstrap 95% "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] over {len(regions)} regions")
    return out


def bootstrap_contrasts(name, learner="qrf", level=0.03, scheme="group", results_root=None) -> list[dict]:
    """The published pair: structured_ya vs chained, then structured vs chained (one JSON file each)."""
    return [bootstrap(name, learner, level, scheme, a=arm, results_root=results_root) for arm in BOOTSTRAP_ARMS]


def main(argv=None):
    """python -m pedopilot.bootstrap NAME [--learner] [--level] [--scheme]"""
    ap = argparse.ArgumentParser(prog="python -m pedopilot.bootstrap",
                                 description="Region-level cluster bootstrap of structured_ya and structured vs chained.")
    ap.add_argument("name", help="results folder name (results/NAME)")
    ap.add_argument("--learner", default="qrf", help="learner of the contrast (default: qrf)")
    ap.add_argument("--level", type=float, default=0.03, help="label level (default: 0.03)")
    ap.add_argument("--scheme", default="group", help="thinning scheme (default: group)")
    ar = ap.parse_args(argv)
    bootstrap_contrasts(ar.name, ar.learner, ar.level, ar.scheme)


if __name__ == "__main__":
    main()
