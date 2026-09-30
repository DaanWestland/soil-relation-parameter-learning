"""The experimental design, shared by the pipeline and the analysis: the Config, spatial folds, the
configurations of a fold, their file tags, arm-name conventions and grid-cell ids.

Imports numpy and yaml only (no torch), so the analysis and the bootstrap work without a deep-learning
stack. Every name here is also importable from pedopilot.experiment, as before.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .paths import CONFIGS

# learners in report order (config names; learners.LEARNERS maps them to classes)
LEARNER_ORDER = ("qrf", "tabm", "tabicl", "tabicl_ft")

# arm-name suffixes of the chained-type arms: their m draws come in k_chain groups that share one y_A draw.
# Arm names ("<learner>:<kind>", "ptf", "ptf_marg") are output data: never rename them.
PAIRED_ARMS = (":chained", ":structured_ya")


@dataclass
class Config:
    """One experiment (configs/<name>.yaml). Fields (do not add, rename or reorder: config.yaml is written
    in this order and compared when a run is resumed):

      name            results folder (results/<name>)
      n_folds, folds  number of spatial folds of 100 km blocks, and which of them to run
      levels          label levels; 1.0 always runs as scheme "all", lower levels as "group" and "random"
      group_reps, random_reps  replicates per level of each thinning scheme
      m, k_chain      draws per site, and y_A draws per chained-type arm (k_chain must divide m)
      residual        draw the log-scale residual around the bounded relation (structured and PTF arms)
      gate            CEC7 regime gate (calcareous rows exempt from the relation, fallback to chained)
      min_train       fewest training rows (per target) for a configuration to be fitted
      seed            fold layout (and `subsample`) only; thinning and learner seeds do not use it
      abundant, gate_model  ForestSampler / RegimeClassifier settings of the shared abundant model and gate
      learners        {name: settings} with names from learners.LEARNERS; a settings key `levels: [...]`
                      restricts that learner to those levels (e.g. tabicl_ft at 10% and 3% only)
      subsample       optional fraction of profiles (smoke tests)
      allow_cpu       allow neural learners without CUDA
      cec_weight_power  CEC7 parameter-fit weights clay**power (2 pre-specified, 1 used by the GPU runs)
    """
    name: str = "smoke"
    n_folds: int = 5
    folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    levels: list[float] = field(default_factory=lambda: [1.0, 0.3, 0.1, 0.03])
    group_reps: int = 3
    random_reps: int = 1
    m: int = 100
    k_chain: int = 20
    residual: bool = True
    gate: bool = True
    min_train: int = 30
    seed: int = 2026
    abundant: dict = field(default_factory=lambda: {"n_estimators": 300})
    gate_model: dict = field(default_factory=lambda: {"n_estimators": 300})
    learners: dict[str, dict] = field(default_factory=lambda: {"qrf": {}})
    subsample: float | None = None          # optional: use a fraction of profiles (for smoke tests)
    allow_cpu: bool = False                 # allow neural learners without CUDA (CPU smoke tests only)
    cec_weight_power: float = 2.0           # CEC parameter fit weighted by clay**power: 2 = pre-specified
                                            # (least squares; cpu_qrf, cpu_qrf_oldfolds, smoke), 1 =
                                            # CRPS/pinball-consistent (gpu_full, gpu_quick, cpu_qrf_w1,
                                            # cpu_qrf_w1_s2, gpu_smoke, gpu_ft and the synthetic
                                            # validation; docs/DESIGN.md, Deviation)

    @classmethod
    def load(cls, path) -> "Config":
        """From a YAML file path, or a config name looked up as configs/<name>.yaml."""
        p = Path(path)
        if not p.exists():
            p = CONFIGS / f"{path}.yaml"
        return cls(**yaml.safe_load(p.read_text()))


def _salt(*parts) -> int:
    """Deterministic 31-bit seed from any parts (CRC32 of their text). Never Python's hash(), which is
    randomised per process for strings."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)


def assign_folds(df, n_folds, seed):
    """Fold of each 100 km block from a hash of (seed, block id). A block keeps its fold when other
    blocks are added or dropped (a newer WoSIS snapshot, a subsample), so folds are reproducible."""
    fold_of = {b: _salt("fold", seed, b) % n_folds for b in df["block"].unique()}
    return df["block"].map(fold_of).values.astype(int)


def configurations(cfg) -> list[tuple[float, str, int]]:
    """(level, scheme, rep) of every configuration of one fold: (1.0, "all", 0) first, then per level < 1
    the group replicates and the random replicates."""
    out = [(1.0, "all", 0)]
    for lv in cfg.levels:
        if lv >= 1:
            continue
        out += [(lv, "group", r) for r in range(cfg.group_reps)]
        out += [(lv, "random", r) for r in range(cfg.random_reps)]
    return out


def config_tag(fold, level, scheme, rep) -> str:
    """File tag of a configuration, e.g. f0_group_0.03_r1 (f{fold}_{scheme}_{level:g}_r{rep})."""
    return f"f{fold}_{scheme}_{level:g}_r{rep}"


def is_paired(arm: str) -> bool:
    """True for chained-type arms (L:chained, L:structured_ya), whose draws are paired with k_chain y_A draws."""
    return arm.endswith(PAIRED_ARMS)


def paired_ya_index(arm, m, k_chain):
    """Which y_A draw each of the m draws of an arm was generated with. Chained-type arms generate
    their draws in k_chain groups of m/k_chain that share one y_A draw."""
    if is_paired(arm):
        return (np.arange(m) * k_chain) // m
    return np.arange(m)


def grid_id(x_km, y_km, size_km):
    """Id "i_j" of the size_km grid cell of each point (floor division of the EPSG:5070 km coordinates):
    100 km blocks (cross-validation unit) and 300 km regions (thinning unit). Works on numpy arrays and
    pandas Series alike. Block "i_j" lies in region "(i // 3)_(j // 3)" (bootstrap.region_of)."""
    return np.floor(x_km / size_km).astype(int).astype(str) + "_" + np.floor(y_km / size_km).astype(int).astype(str)
