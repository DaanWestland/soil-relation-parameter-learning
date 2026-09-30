"""Summaries, contrasts, figures and a markdown report for one results directory.

Intervals: replicates are averaged within each spatial fold first (replicates share the same test
sites), then a t-interval over folds (df = n_folds - 1). Only folds with ALL configurations complete
are used, so every level is averaged over the same folds.

Skill = 1 - mean CRPS(arm) / mean CRPS(REF) over the same sites of one unit (fold x replicate x level x
scheme x target); REF is the free QRF arm. A contrast "a vs b" is skill(a) - skill(b) per unit (> 0: a is
better), also averaged over the two targets ("mean", units with both targets).

`analyze(name)` writes, next to the raw files in results/<name>/:
  summary_skill.csv        skill of every arm per (level, scheme, target), fold t-interval
  contrasts.csv            the CONTRASTS per learner, (level, scheme) and target incl. "mean"
  contrasts_log.csv        the same on the CRPS of log N and log CEC7
  strata_skill.csv         skill on retained vs thinned regions (column `stratum`), per scheme
  strata_contrasts.csv     contrasts on retained vs thinned regions
  did_group_vs_random.csv  difference-in-differences on thinned sites: [arm - chained] group - random
  joint_skill.csv          energy ("es") and variogram ("vs") skill of the joint (clay, OC, pH, N, CEC7) draws
  joint_contrasts.csv      contrasts on the joint scores
  coverage90.csv           mean 90% central-interval coverage per (level, scheme, target) and arm
  binding.csv              share of paired draws outside the envelope, and of free draws vs observed clay/OC
  rank_correlation.csv     Figure-1 check: Spearman of CEC7-clay and N-OC (observed, draws, median map)
  runtime.csv              seconds per configuration and learner, n_train, realised label shares, ESS,
                           epochs and fine-tuning gain
  primary.json             the pre-specified primary contrast (CRPS and log-CRPS)
  mixed_model.json         exploratory mixed model per learner
  skill_vs_thinning.png, contrast_structured_vs_chained.png, contrast_structured_ya_vs_chained.png, pit.png
  REPORT.md                sections 0 key numbers, 1 primary result, 2 per learner at the lowest level,
                           3 structured vs chained, 4 formula only (structured_ya vs chained), 5 thinned vs
                           retained regions and difference-in-differences, 6 joint scores, 7 skill of every
                           arm, 8 coverage, 9 binding, 10 mixed model, 11 runtime, 12 Figure-1 check

matplotlib is switched to the non-interactive Agg backend at import (figures are only written to files).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .design import LEARNER_ORDER, Config, config_tag, configurations, is_paired
from .paths import RESULTS

REF = "qrf:free"
UNIT = ["fold", "rep"]
KEYS = ["level", "scheme", "target"]
PRIMARY_LEARNERS = ("qrf", "tabm", "tabicl")          # the learner-averaged primary contrast (pre-specified in docs/DESIGN.md)
COLORS = {"qrf": "#2b6ca3", "tabm": "#c0392b", "tabicl": "#27ae60", "tabicl_ft": "#8e44ad", "ptf": "#333333",
          "ptf_marg": "#888888", "mean": "#000000"}
STYLE = {"free": (0, (4, 2)), "chained": "--", "structured": "-", "structured_ya": "-.", "ptf": ":", "ptf_marg": ":"}
CONTRASTS = [("structured", "chained"), ("structured_ya", "chained"), ("structured", "ptf_marg"),
             ("structured", "ptf"), ("structured", "clip"), ("structured", "free"), ("chained", "free")]


# ----------------------------------------------------------------------------- loading
def _fold_complete(res_dir: Path, cfg: Config, fold: int) -> bool:
    """True if every configuration of `fold` has its meta file (the completion marker)."""
    return all((res_dir / f"meta_{config_tag(fold, lv, s, r)}.json").exists() for lv, s, r in configurations(cfg))


def complete_folds(res_dir) -> list[int]:
    """Folds of cfg.folds whose configurations all have a meta file: the folds `load` (and so `analyze`)
    uses when the run used cfg.folds."""
    res_dir = Path(res_dir)
    cfg = Config.load(res_dir / "config.yaml")
    return [f for f in cfg.folds if _fold_complete(res_dir, cfg, f)]


class LoadedResults(NamedTuple):
    """What `load` returns; unpacks as the 7-tuple (cfg, sites, joint, metas, obs, done, partial)."""
    cfg: Config
    sites: pd.DataFrame                     # per-site scores (sites_*.parquet)
    joint: pd.DataFrame                     # joint scores (joint_*.parquet; may be empty)
    metas: pd.DataFrame                     # one row per configuration (meta_*.json)
    obs: dict                               # observed envelope-violation rates (observed_violation.json)
    done: list                              # complete folds
    partial: list                           # folds of cfg.folds that are not complete


def load(res_dir, include_partial: bool = False) -> LoadedResults:
    """Read one results folder. Returns (cfg, sites, joint, metas, obs, done, partial): the Config, the
    per-site scores (all sites_*.parquet), the joint scores (joint_*.parquet, may be empty), one row per
    configuration (meta_*.json), the observed envelope-violation rates, the complete folds and the folds
    of cfg.folds that are not complete. Unless include_partial, sites, joint and metas are restricted to
    the complete folds (when there is at least one)."""
    res_dir = Path(res_dir)
    cfg = Config.load(res_dir / "config.yaml")
    metas = pd.DataFrame([json.loads(p.read_text()) for p in sorted(res_dir.glob("meta_*.json"))])
    present = sorted(set(metas.fold)) if len(metas) else []
    # folds present in the meta files (not cfg.folds): `run --folds` may have run folds outside cfg.folds
    done = [f for f in present if _fold_complete(res_dir, cfg, f)]
    partial = [f for f in cfg.folds if f not in done]           # includes folds that never started
    sites = pd.concat([pd.read_parquet(p) for p in sorted(res_dir.glob("sites_*.parquet"))], ignore_index=True)
    jf = sorted(res_dir.glob("joint_*.parquet"))
    joint = pd.concat([pd.read_parquet(p) for p in jf], ignore_index=True) if jf else pd.DataFrame()
    if done and not include_partial:
        sites, metas = sites[sites.fold.isin(done)], metas[metas.fold.isin(done)]
        joint = joint[joint.fold.isin(done)] if len(joint) else joint
    obs = json.loads((res_dir / "observed_violation.json").read_text())
    return LoadedResults(cfg, sites, joint, metas, obs, done, partial)


# ----------------------------------------------------------------------------- statistics
def fold_ci(frame: pd.DataFrame, col: str) -> tuple[float, float, float, int]:
    """Mean over replicates within fold, then mean and t-interval over folds."""
    f = frame.dropna(subset=[col]).groupby("fold")[col].mean()
    k = len(f)
    if k == 0:
        return np.nan, np.nan, np.nan, 0
    mu = float(f.mean())
    if k < 2:
        return mu, np.nan, np.nan, 1
    h = float(stats.t.ppf(0.975, k - 1) * f.std(ddof=1) / np.sqrt(k))
    return mu, mu - h, mu + h, k


def unit_skill(df: pd.DataFrame, score: str = "crps", keys=KEYS) -> pd.DataFrame:
    """Per (keys, fold, rep): skill = 1 - mean score(arm) / mean score(REF) on the same sites."""
    m = df.groupby(list(keys) + UNIT + ["arm"])[score].mean().unstack("arm")
    if REF not in m:
        return pd.DataFrame()
    return (1 - m.div(m[REF], axis=0)).reset_index()


def summarise(sk: pd.DataFrame, keys=KEYS) -> pd.DataFrame:
    """Skill of every arm per key (level, scheme, target) with the fold t-interval; one row per key and arm."""
    arms = [c for c in sk.columns if c not in list(keys) + UNIT]
    rows = []
    for key, g in sk.groupby(list(keys)):
        key = key if isinstance(key, tuple) else (key,)
        for a in arms:
            mu, lo, hi, k = fold_ci(g, a)
            rows.append({**dict(zip(keys, key)), "arm": a, "skill": mu, "lo": lo, "hi": hi, "n_folds": k,
                         "units": int(g[a].notna().sum())})
    return pd.DataFrame(rows)


def _pairs(learners):
    """(learner, contrast name, arm a, arm b) for every CONTRASTS entry; PTF arms are learner-free."""
    out = []
    for L in learners:
        for a, b in CONTRASTS:
            aa = f"{L}:{a}"
            bb = b if b.startswith("ptf") else f"{L}:{b}"
            out.append((L, f"{a} vs {b}", aa, bb))
    return out


def _target_series(d: pd.DataFrame) -> dict:
    """{target: per-unit differences} from a unit x target table, plus "mean" over targets (units with every
    target) when there is more than one target."""
    series = {t: d[t] for t in d.columns}
    if d.shape[1] > 1:
        series["mean"] = d.dropna().mean(axis=1)
    return series


def contrast_table(sk: pd.DataFrame, learners, by=("level", "scheme")) -> pd.DataFrame:
    """Skill difference per unit, per target and as the mean over targets (units with both targets)."""
    rows = []
    for key, g in sk.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for L, name, a, b in _pairs(learners):
            if a not in g or b not in g:
                continue
            d = g.assign(d=g[a] - g[b]).pivot_table(index=UNIT, columns="target", values="d")
            for t, s in _target_series(d).items():
                mu, lo, hi, k = fold_ci(s.rename("d").reset_index(), "d")
                rows.append({**dict(zip(by, key)), "learner": L, "contrast": name, "target": t,
                             "diff": mu, "lo": lo, "hi": hi, "n_folds": k, "units": int(s.notna().sum())})
    return pd.DataFrame(rows)


def primary(sk, cfg, learners):
    """Pre-specified (docs/DESIGN.md, before the runs): structured vs chained-free at the lowest level, group
    thinning, averaged over the learners QRF, TabM and TabICLv2, per target and as the mean over targets.
    Also the formula-only version (structured_ya vs chained)."""
    lv = min(cfg.levels)
    g = sk[(sk.level == lv) & (sk.scheme == "group")]
    if g.empty:
        return {"status": f"not yet available (no complete fold with level {lv} group)"}
    common = [L for L in PRIMARY_LEARNERS if f"{L}:structured" in g and f"{L}:chained" in g]
    out = {"level": lv, "scheme": "group", "learners_averaged": common, "results": []}
    for a, b in (("structured", "chained"), ("structured_ya", "chained")):
        have = [L for L in common if f"{L}:{a}" in g]
        if not have:
            continue
        diff = sum(g[f"{L}:{a}"] - g[f"{L}:{b}"] for L in have) / len(have)
        d = g.assign(d=diff).pivot_table(index=UNIT, columns="target", values="d")
        for t, s in _target_series(d).items():
            fr = s.rename("d").reset_index()
            mu, lo, hi, k = fold_ci(fr, "d")
            fm = fr.dropna(subset=["d"]).groupby("fold")["d"].mean()
            p = float(2 * stats.t.sf(abs(mu) / (fm.std(ddof=1) / np.sqrt(k)), k - 1)) if k > 1 and fm.std(ddof=1) > 0 \
                else float("nan")
            out["results"].append({"contrast": f"{a} vs {b}", "target": t, "learners": "average", "diff": mu,
                                   "lo": lo, "hi": hi, "n_folds": k, "p_fold_t": p})
    return out


def mixed_model(sites, level, scheme, learner):
    """Equal weight over targets; replicates nested in folds (statsmodels, optional)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"note": "statsmodels not installed (pip install -e .[stats])"}
    s = sites[(sites.level == level) & (sites.scheme == scheme)]
    a, b = f"{learner}:structured", f"{learner}:chained"
    p = s[s.arm.isin([a, b, REF])].pivot_table(index=["fold", "rep", "target", "site"], columns="arm", values="crps")
    if not {a, b, REF} <= set(p.columns):
        return None
    p = p.dropna()
    scale = p.groupby(level=["fold", "rep", "target"])[REF].transform("mean")
    d = pd.DataFrame({"y": (p[b] - p[a]) / scale}).reset_index()
    n_sites = int(d["site"].nunique())
    # unit means (fold x replicate x target): site-level fits often stop at a boundary optimum
    d = d.groupby(["fold", "rep", "target"], as_index=False)["y"].mean()
    out = {"per_target": d.groupby(["target", "fold"])["y"].mean().groupby("target").mean().to_dict(),
           "n_sites": n_sites, "n_units": len(d),
           "note": "exploratory; the primary inference is the fold-level t-interval (section 1)"}
    d["rep_s"] = d["rep"].astype(str)
    k = d["fold"].nunique()
    try:
        import warnings
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            # fold random intercept (re_formula="1": without it statsmodels fits NO group effect when a
            # vc_formula is given) plus replicate-within-fold variance; t test with n_folds - 1 df
            fit = smf.mixedlm("y ~ C(target, Sum)", d, groups=d["fold"], re_formula="1",
                              vc_formula={"rep": "0 + C(rep_s)"}).fit(reml=True)
        est, se = float(fit.params["Intercept"]), float(fit.bse["Intercept"])
        msgs = sorted({str(w.message).split(".")[0][:80] for w in wlog})
        bad = any(m.startswith("MixedLM optimization failed") or "not positive definite" in m for m in msgs)
        out.update({"estimate_equal_weight": est, "se": se, "df": k - 1,
                    "p_t": float(2 * stats.t.sf(abs(est / se), k - 1)) if k > 1 and se > 0 else float("nan"),
                    "converged": bool(fit.converged) and not bad,
                    "boundary": any("boundary" in m for m in msgs), "warnings": msgs})
    except Exception as e:                                    # noqa: BLE001
        out["error"] = str(e)
    return out


def strata_tables(sites, learners):
    """Retained vs thinned regions for both schemes (random rep r carries the regions of group rep r,
    so the same sites are compared), and the difference-in-differences on thinned sites, for the
    parameter-from-x and the formula-only contrasts: [arm - chained](group) - [arm - chained](random)."""
    summ, cons, did = [], [], []
    for sch in ("group", "random"):
        for flag, lab in ((True, "retained"), (False, "thinned")):
            g = sites[(sites.scheme == sch) & (sites.retained == flag)]
            if g.empty:
                continue
            sk = unit_skill(g)
            summ.append(summarise(sk).assign(stratum=lab))
            cons.append(contrast_table(sk, learners).assign(stratum=lab))
    thin = sites[(sites.retained == False) & sites.scheme.isin(["group", "random"])]   # noqa: E712
    if len(thin):
        sk = unit_skill(thin)
        for L in learners:
            for arm in ("structured", "structured_ya"):
                a, b = f"{L}:{arm}", f"{L}:chained"
                if a not in sk or b not in sk:
                    continue
                w = sk.assign(d=sk[a] - sk[b]).pivot_table(index=["level", "target", "fold", "rep"], columns="scheme",
                                                           values="d")
                if not {"group", "random"} <= set(w.columns):
                    continue
                w = (w["group"] - w["random"]).rename("d").reset_index()
                for (lv, t), gg in w.groupby(["level", "target"]):
                    mu, lo, hi, k = fold_ci(gg, "d")
                    did.append({"level": lv, "target": t, "learner": L, "contrast": f"{arm} vs chained", "did": mu,
                                "lo": lo, "hi": hi, "n_folds": k})
    return _concat_or_empty(summ), _concat_or_empty(cons), pd.DataFrame(did)


def _concat_or_empty(frames):
    """pd.concat of a list of frames, or an empty frame for an empty list."""
    return pd.concat(frames) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------- figures
def figure_skill(summ, out, schemes):
    """Skill vs level of every arm, one row per thinning scheme, N and CEC7 side by side."""
    fig, axes = plt.subplots(len(schemes), 2, figsize=(12, 4.3 * len(schemes)), squeeze=False)
    for r, scheme in enumerate(schemes):
        for c, (target, title) in enumerate((("n", "Total N"), ("cec", "CEC7"))):
            ax = axes[r, c]
            g = summ[(summ.target == target) & (summ.scheme.isin([scheme, "all"]))]
            for a in sorted(g.arm.unique()):
                L, kind = a.split(":") if ":" in a else (a, a)
                if a == REF or kind not in STYLE:
                    continue
                h = g[g.arm == a].sort_values("level", ascending=False)
                x = 100 * h.level.values
                ax.plot(x, h.skill, ls=STYLE[kind], lw=2.6 if kind.startswith("structured") else 1.3,
                        color=COLORS.get(L, "#777"), marker="o", ms=3, label=a)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks([3, 10, 30, 100])
            ax.set_xticklabels(["3%", "10%", "30%", "100%"])
            ax.invert_xaxis()
            ax.set_title(f"{title} - thinning: {scheme}")
            ax.set_xlabel("sparse labels kept in training")
            ax.set_ylabel("CRPS skill vs free QRF")
    axes[0, 1].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Own preliminary pilot on public WoSIS data (CONUS, 0-30 cm, 100 km block CV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_contrast(con, out, contrast="structured vs chained"):
    """One contrast vs level per learner and scheme with fold t-intervals: N, CEC7 and their mean."""
    c = con[(con.contrast == contrast) & con.scheme.isin(["group", "random", "all"])]
    if c.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, t, title in zip(axes, ("n", "cec", "mean"), ("Total N", "CEC7", "mean of N and CEC7")):
        for L in sorted(c.learner.unique()):
            for sch, mk, off in (("group", "o", -0.04), ("random", "^", 0.04)):
                h = c[(c.learner == L) & (c.target == t) & c.scheme.isin([sch, "all"])].sort_values("level")
                if h.empty:
                    continue
                x = np.log10(100 * h.level.values) + off + 0.02 * list(COLORS).index(L)
                ax.errorbar(x, h["diff"], yerr=[h["diff"] - h.lo, h.hi - h["diff"]], fmt=mk, ms=4, capsize=2,
                            color=COLORS.get(L, "#777"), label=f"{L} ({sch})" if t == "n" else None, alpha=0.9)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(np.log10([3, 10, 30, 100]))
        ax.set_xticklabels(["3%", "10%", "30%", "100%"])
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel("sparse labels kept")
    axes[0].set_ylabel(f"skill difference: {contrast}\n(> 0: the formula helps)")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Own preliminary pilot: 95% t-intervals over spatial folds (replicates averaged within fold)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_pit(sites, out, level, cfg=None):
    """PIT histograms at one level (group thinning) for the free, chained and structured arms."""
    s = sites[(sites.level == level) & (sites.scheme.isin(["group", "all"]))]
    arms = [a for a in sorted(s.arm.unique()) if a.endswith((":free", ":chained", ":structured")) or a == "ptf_marg"]
    if not arms:
        return
    fig, axes = plt.subplots(2, len(arms), figsize=(2.1 * len(arms), 4.4), squeeze=False)
    for c, a in enumerate(arms):
        for r, t in enumerate(("n", "cec")):
            v = s[(s.arm == a) & (s.target == t)]["pit"].values
            # the stored PIT lives on a grid of g = draws + 1 points (k_chain + 1 for chained-type arms):
            # jitter within the grid cell so that 10 equal bins show no binning artefact
            g = ((cfg.k_chain if is_paired(a) and cfg.k_chain < cfg.m else cfg.m) + 1
                 if cfg is not None else 101)
            v = np.clip(v + (np.random.default_rng(0).random(len(v)) - 0.5) / g, 0, 1)
            axes[r, c].hist(v, bins=10, range=(0, 1), color=COLORS.get(a.split(":")[0], "#555"), alpha=0.8)
            axes[r, c].axhline(len(v) / 10, color="k", lw=0.7)
            axes[r, c].set_title(f"{a}\n{t}", fontsize=7)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(f"PIT histograms at {level:.0%} (flat = calibrated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def _ci(r, col="diff"):
    """'+0.012 [-0.003, +0.027]' (or the value alone without an interval) for one table row."""
    if pd.isna(r[col]):
        return ""
    if pd.isna(r["lo"]):
        return f"{r[col]:+.3f}"
    return f"{r[col]:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _md(df, index, columns, col="diff"):
    """Markdown pivot table of _ci cells."""
    if df is None or df.empty:
        return "_(none)_\n"
    d = df.copy()
    d["cell"] = d.apply(lambda r: _ci(r, col), axis=1)
    p = d.pivot_table(index=index, columns=columns, values="cell", aggfunc="first").fillna("")
    return p.to_markdown(disable_numparse=True) + "\n"


def rank_table(res_dir, folds=None):
    """Figure-1 check: Spearman of CEC7-clay and N-OC at held-out sites, observed vs joint draws vs the
    median map, per arm; fold-level means with t-intervals for the gaps to the observed correlation."""
    files = sorted(Path(res_dir).glob("rank_*.parquet"))
    if not files:
        return pd.DataFrame()
    r = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if folds:
        r = r[r.fold.isin(folds)]
    r = r.assign(gap_draws=r.rho_draws - r.rho_obs, gap_median=r.rho_median - r.rho_obs)
    rows = []
    for (lv, sc, t, arm), g in r.groupby(["level", "scheme", "target", "arm"]):
        row = {"level": lv, "scheme": sc, "target": t, "arm": arm}
        for c in ("rho_obs", "rho_draws", "rho_median", "gap_draws", "gap_median"):
            mu, lo, hi, k = fold_ci(g, c)
            row[c], row[f"{c}_lo"], row[f"{c}_hi"], row["folds"] = mu, lo, hi, k
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_md(rk):
    """Markdown tables of the Figure-1 check: Spearman on joint draws (median map in brackets) per target."""
    if not len(rk):
        return "_(none)_\n"
    out = []
    for t, g in rk.groupby("target"):
        d = g.assign(cell=g.apply(lambda r: f"{r.rho_draws:+.2f} ({r.rho_median:+.2f})", axis=1))
        p = d.pivot_table(index=["scheme", "level"], columns="arm", values="cell", aggfunc="first").fillna("")
        p.insert(0, "observed", g.groupby(["scheme", "level"])["rho_obs"].mean().map(lambda v: f"{v:+.2f}"))
        out += [f"**{'CEC7 vs clay' if t == 'cec' else 'N vs OC'}**: Spearman on joint draws (median map in brackets)", "",
                p.to_markdown(disable_numparse=True), ""]
    return "\n".join(out) + "\n"


def _runtime(metas):
    """Seconds, TabICL context ESS, epochs and fine-tuning gain per (target, learner) and level."""
    rows = []
    for t in ("n", "cec"):
        for L in LEARNER_ORDER:
            c = f"{t}:{L}_seconds"
            if c not in metas:
                continue
            g = metas.dropna(subset=[c]).groupby("level")
            for lv, d in g:
                fk = d.get(f"{t}:frac_kept", pd.Series(dtype=float))
                r = {"target": t, "learner": L, "level": lv, "configs": len(d), "seconds": d[c].mean(),
                     "n_train": d.get(f"{t}:n_train", pd.Series(dtype=float)).mean(),
                     "labels_kept_mean": fk.mean(), "labels_kept_min": fk.min(), "labels_kept_max": fk.max()}
                for k in ("ess", "epochs", "ft_gain"):
                    if f"{t}:{L}_{k}" in d and d[f"{t}:{L}_{k}"].notna().any():
                        r[k] = d[f"{t}:{L}_{k}"].mean()
                rows.append(r)
    return pd.DataFrame(rows).set_index(["target", "learner", "level"]).round(3) if rows else pd.DataFrame()


def _key_numbers(con, rk, jsum, learners, lv):
    """A few plain-language lines with the headline numbers."""
    def p(r):
        return f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"

    def get(contrast, level, scheme, target, L):
        c = con[(con.contrast == contrast) & (con.level == level) & (con.scheme == scheme) & (con.target == target)
                & (con.learner == L)]
        return p(c.iloc[0]) if len(c) else "n/a"
    out = []
    for L in learners:
        f = "structured_ya vs chained"
        out.append(f"- **{L}**: formula only ({f}) at {lv:.0%} by region: mean {get(f, lv, 'group', 'mean', L)} "
                   f"(N {get(f, lv, 'group', 'n', L)}, CEC7 {get(f, lv, 'group', 'cec', L)}); at {lv:.0%} at random "
                   f"{get(f, lv, 'random', 'mean', L)}; with all labels {get(f, 1.0, 'all', 'mean', L)}; chaining "
                   f"alone (chained vs free) at {lv:.0%}: {get('chained vs free', lv, 'group', 'mean', L)}")
    if rk is not None and len(rk):
        r = rk[(rk.level == lv) & (rk.scheme == "group") & (rk.target == "cec")].set_index("arm")
        for L in learners:
            arms = [a for a in (f"{L}:free", f"{L}:chained", f"{L}:structured_ya", "ptf") if a in r.index]
            if arms:
                out.append(f"- CEC7-clay Spearman at held-out sites ({L}, {lv:.0%}): observed {r.rho_obs.mean():.2f}; "
                           + "; ".join(f"{a.split(':')[-1]} {r.loc[a, 'rho_draws']:.2f}" for a in arms))
    if len(jsum):
        j = jsum[(jsum.score == "vs") & (jsum.scheme == "group") & (jsum.level == lv)].set_index("arm")
        for L in learners:
            if f"{L}:structured_ya" in j.index and f"{L}:chained" in j.index:
                out.append(f"- joint variogram-score skill ({L}, {lv:.0%}): structured_ya "
                           f"{j.loc[f'{L}:structured_ya', 'skill']:+.3f} vs chained {j.loc[f'{L}:chained', 'skill']:+.3f}")
    return out or ["_(not available yet)_"]


@dataclass
class AnalysisTables:
    """Every table `analyze` computes (see the module docstring for the files they become)."""
    learners: list                  # learners present, in LEARNER_ORDER
    summ: pd.DataFrame              # summary skill
    con: pd.DataFrame               # contrasts (CRPS)
    con_log: pd.DataFrame           # contrasts (CRPS of the log target)
    prim: dict                      # primary contrast (CRPS)
    prim_log: dict                  # primary contrast (log CRPS)
    st_summ: pd.DataFrame           # strata (retained vs thinned) skill
    st_con: pd.DataFrame            # strata contrasts
    did: pd.DataFrame               # difference-in-differences
    jsum: pd.DataFrame              # joint-score skill
    jcon: pd.DataFrame              # joint-score contrasts
    cov: pd.DataFrame               # 90% coverage
    bind: pd.DataFrame              # binding shares
    mm: dict                        # mixed models per learner
    rk: pd.DataFrame                # rank correlations (Figure-1 check)
    rt: pd.DataFrame                # runtime and learner diagnostics


def _learners_in(sites):
    """Learners with at least one arm in the per-site scores, in LEARNER_ORDER."""
    return [L for L in LEARNER_ORDER if any(a.startswith(L + ":") for a in sites.arm.unique())]


def _compute_tables(res_dir, r: LoadedResults, include_partial=False) -> AnalysisTables:
    """All tables of one results folder (sk = unit skill: one row per unit, one column per arm)."""
    cfg, sites, joint, metas = r.cfg, r.sites, r.joint, r.metas
    learners = _learners_in(sites)
    sk = unit_skill(sites)
    summ = summarise(sk)
    con = contrast_table(sk, learners)
    con_log = contrast_table(unit_skill(sites, "crps_log"), learners)
    prim = primary(sk, cfg, learners)
    prim_log = primary(unit_skill(sites, "crps_log"), cfg, learners)
    st_summ, st_con, did = strata_tables(sites, learners)
    jsum, jcon = pd.DataFrame(), pd.DataFrame()
    if len(joint):
        jparts, cparts = [], []
        for sc in ("es", "vs"):
            js = unit_skill(joint.assign(target="joint"), sc)
            jparts.append(summarise(js).assign(score=sc))
            cparts.append(contrast_table(js, learners).assign(score=sc))
        jsum, jcon = pd.concat(jparts), pd.concat(cparts)
    cov = sites.groupby(["level", "scheme", "target", "arm"])["cov90"].mean().unstack("arm")
    bcols = [c for c in metas.columns if c.startswith(("n:binding", "cec:binding"))]
    bind = metas.groupby(["level", "scheme"])[bcols].mean() if bcols else pd.DataFrame()
    mm = {L: mixed_model(sites, min(cfg.levels), "group", L) for L in learners}
    rk = rank_table(res_dir, None if include_partial else r.done)
    rt = _runtime(metas)
    return AnalysisTables(learners, summ, con, con_log, prim, prim_log, st_summ, st_con, did, jsum, jcon, cov, bind,
                          mm, rk, rt)


def _write_tables(res_dir, t: AnalysisTables):
    """The CSV and JSON outputs (names and columns are read by other tools: keep them)."""
    t.rk.to_csv(res_dir / "rank_correlation.csv", index=False)
    t.rt.to_csv(res_dir / "runtime.csv")
    for df_, fn in ((t.summ, "summary_skill"), (t.con, "contrasts"), (t.con_log, "contrasts_log"),
                    (t.st_summ, "strata_skill"), (t.st_con, "strata_contrasts"), (t.did, "did_group_vs_random"),
                    (t.jsum, "joint_skill"), (t.jcon, "joint_contrasts")):
        df_.to_csv(res_dir / f"{fn}.csv", index=False)
    t.cov.to_csv(res_dir / "coverage90.csv")
    t.bind.to_csv(res_dir / "binding.csv")
    (res_dir / "primary.json").write_text(json.dumps({"crps": t.prim, "crps_log": t.prim_log}, indent=1, default=float))
    (res_dir / "mixed_model.json").write_text(json.dumps(t.mm, indent=1, default=float))


def _write_figures(res_dir, r: LoadedResults, t: AnalysisTables):
    """The four PNG figures."""
    schemes = [s for s in ("group", "random") if s in set(r.sites.scheme)] or ["all"]
    figure_skill(t.summ, res_dir / "skill_vs_thinning.png", schemes)
    figure_contrast(t.con, res_dir / "contrast_structured_vs_chained.png")
    figure_contrast(t.con, res_dir / "contrast_structured_ya_vs_chained.png", "structured_ya vs chained")
    figure_pit(r.sites, res_dir / "pit.png", min(r.cfg.levels), r.cfg)


def _report_lines(name, r: LoadedResults, t: AnalysisTables) -> list[str]:
    """The lines of REPORT.md. Section numbers and titles 0-12 and the header lines are parsed by other
    tools: keep them."""
    cfg, sites, metas, obs, done, partial = r.cfg, r.sites, r.metas, r.obs, r.done, r.partial
    learners, summ, con, prim, prim_log = t.learners, t.summ, t.con, t.prim, t.prim_log
    st_con, did, jsum, cov, bind, mm, rk, rt = t.st_con, t.did, t.jsum, t.cov, t.bind, t.mm, t.rk, t.rt
    pr = pd.DataFrame(prim.get("results", []))
    pr_log = pd.DataFrame(prim_log.get("results", []))
    lv = min(cfg.levels)
    main = con[(con.level == lv) & (con.scheme == "group")]
    return [
        f"# Results: `{name}`", "",
        "Own preliminary pilot on public WoSIS data (conterminous USA, 0-30 cm). Skill = 1 - CRPS(arm) / CRPS(free QRF) "
        "on the same held-out sites. Differences are in skill units (> 0: first arm better). Intervals: approximate 95% "
        "t-intervals over spatial folds (the folds share training data, so they are somewhat optimistic), replicates averaged "
        "within fold.", "",
        f"- complete folds used: {done}; incomplete folds excluded: {partial}"
        + ("" if done else " (NO fold complete yet: levels are averaged over different folds)"),
        f"- sites scored: {sites.site.nunique()}; configurations: {len(metas)}; learners: {learners}",
        f"- observed envelope-violation rate in the data: N {obs.get('n', float('nan')):.3f}, CEC7 {obs.get('cec', float('nan')):.3f}",
        "", "## 0. Key numbers (read this first; points = skill difference x 100)", "",
        *_key_numbers(con, rk, jsum, learners, lv), "",
        f"## 1. Primary result (pre-specified in docs/DESIGN.md): {lv:.0%} of sparse labels, group thinning, averaged over {prim.get('learners_averaged', [])}", "",
        _md(pr, ["contrast"], "target") if len(pr) else f"_{prim.get('status', 'n/a')}_\n",
        "Same on the log scale (CRPS of log N, log CEC7):", "",
        _md(pr_log, ["contrast"], "target") if len(pr_log) else "_(n/a)_\n",
        f"## 2. Per learner at {lv:.0%} (group thinning)", "",
        _md(main, ["learner", "contrast"], "target"),
        "## 3. Structured vs chained-free at every level (group and random thinning)", "",
        _md(con[con.contrast == "structured vs chained"], ["scheme", "level", "learner"], "target"),
        "## 4. Formula only (theta from x AND abundant properties) vs chained-free", "",
        _md(con[con.contrast == "structured_ya vs chained"], ["scheme", "level", "learner"], "target"),
        "## 5. Extrapolation: structured vs chained-free on thinned vs retained regions, and group - random on thinned sites", "",
        _md(st_con[st_con.contrast == "structured vs chained"] if len(st_con) else st_con,
            ["scheme", "stratum", "level", "learner"], "target"),
        "Difference-in-differences on thinned sites ([arm - chained] group minus random, same sites):", "",
        _md(did.rename(columns={"did": "diff"}) if len(did) else did, ["contrast", "level", "learner"], "target"),
        "## 6. Joint scores (vector clay, OC, pH, N, CEC7): skill vs free QRF", "",
        _md(jsum.rename(columns={"skill": "diff"}) if len(jsum) else jsum, ["score", "scheme", "level"], "arm"),
        "## 7. CRPS skill of every arm vs free QRF", "",
        _md(summ.rename(columns={"skill": "diff"}), ["scheme", "level", "target"], "arm"),
        "## 8. 90% interval coverage", "", cov.round(3).to_markdown() + "\n",
        "## 9. Binding: share of paired chained draws outside the envelope, and of free draws vs observed clay/OC", "",
        bind.round(3).to_markdown() + "\n" if len(bind) else "_(none)_\n",
        "## 10. Mixed model (exploratory: fold-x-replicate-x-target means, fold random intercept, replicates within fold; the primary inference is section 1)", "",
        "```", json.dumps(mm, indent=1, default=float), "```", "",
        "## 11. Runtime and learner diagnostics (mean per configuration; seconds cover the 4 models of a learner)", "",
        "labels_kept = realised share of the target's training labels (the level is a share of profiles with any "
        "sparse label; N is missing by region, so its share varies more).", "",
        rt.to_markdown() + "\n" if len(rt) else "_(none)_\n",
        "## 12. Figure-1 check: rank correlation with the abundant driver at held-out sites", "",
        "Spearman of (clay, CEC7) and (OC, N) pooled over sites and draws, each sparse draw paired with the y_A "
        "draw it was made with; in brackets the per-site median 'map'. A coherent joint predictive reproduces "
        "the observed correlation; independent draws (free) attenuate it. rank_correlation.csv has the gaps "
        "to the observed value with fold-level 95% intervals.", "",
        _rank_md(rk),
        "![skill](skill_vs_thinning.png)", "", "![contrast](contrast_structured_vs_chained.png)", "",
        "![formula only](contrast_structured_ya_vs_chained.png)", "", "![pit](pit.png)", ""]


def analyze(name, include_partial=False, results_root=None):
    """Write every table, figure and REPORT.md for results/<name> (see the module docstring); returns the
    folder. Table names (AnalysisTables): summ = summary skill; con / con_log = contrasts on CRPS / log
    CRPS; prim = primary contrast; st_summ / st_con = strata (retained vs thinned) skill / contrasts;
    did = difference-in-differences; jsum / jcon = joint-score skill / contrasts; cov = coverage;
    bind = binding; mm = mixed models; rk = rank correlations; rt = runtime."""
    res_dir = Path(results_root or RESULTS) / name
    r = load(res_dir, include_partial)
    t = _compute_tables(res_dir, r, include_partial)
    _write_tables(res_dir, t)
    _write_figures(res_dir, r, t)
    (res_dir / "REPORT.md").write_text("\n".join(_report_lines(name, r, t)), encoding="utf-8")
    return res_dir
