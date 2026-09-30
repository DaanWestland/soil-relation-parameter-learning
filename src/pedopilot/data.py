"""Download public data and build the CONUS 0-30 cm profile table.

Sources
  WoSIS 'latest' layers (ISRIC WFS): clay, orgc, phaq, nitkjd, cecph7, tceq (carbonate equivalent)
  WorldClim 2.1: 19 bioclimatic variables (5 arc-min) and elevation (2.5 arc-min)

Preparation chain (`prep`, one row per profile):
  1. layers of each WoSIS property, restricted to the USA inside the CONUS box (lon -125 to -66,
     lat 24 to 50) and to mineral layers (organic_surface == 0)                          [_read_layers]
  2. thickness-weighted mean over 0-30 cm per profile; a profile needs at least MIN_COVER = 24 cm
     (80%) of the 0-30 cm interval covered, else the value is missing                    [_aggregate]
  3. profiles need clay, OC and pH; plausibility filters 0 < clay < 100 %, 0 < OC < 120 g/kg,
     2 < pH < 11; non-positive N or CEC7 is set to missing
  4. calcareous flag: CaCO3 equivalent >= CALC_THRESHOLD (1 g/kg = 0.1 %; WoSIS tceq is in g/kg) in
     the 0-30 cm mean; a missing carbonate value is read as non-calcareous (NCSS analyses carbonate
     only when a sample fizzes)
  5. EPSG:5070 (CONUS Albers) coordinates in km and grid-cell ids "i_j" of 100 km blocks (the
     cross-validation unit) and 300 km regions (the thinning unit)
  6. covariates sampled at the profile location: WorldClim bio1-19 (5'), elevation (2.5'), slope
     from central differences and relief = max - min elevation in the 5 x 5 cell window (2.5')
  7. profiles with any missing covariate are dropped; written to data/conus_0_30.parquet

`load_table` reads data/conus_0_30.parquet when it exists (a local rebuild), else the snapshot in
data_snapshot/ (the exact table of every published run, checked by sha256; it is not distributed with the
code, see data_snapshot/README.md). WoSIS 'latest' is updated
over time, so `prep` run today gives a different table and will NOT reproduce the snapshot's sha256;
use the snapshot to reproduce published results.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import zipfile

from pathlib import Path

import numpy as np
import pandas as pd

from .design import grid_id
from .paths import DATA, MODELS, SNAPSHOT, SNAPSHOT_SHA256, TABICL_CKPT, TABLE
from .relations import COVS

WFS = ("https://maps.isric.org/mapserv?map=/map/wosis_latest.map&SERVICE=WFS&VERSION=2.0.0"
       "&REQUEST=GetFeature&TYPENAMES=ms:wosis_latest_{layer}&OUTPUTFORMAT=csv")
WORLDCLIM = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/{name}"
TABICL_URL = "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt"
LAYERS = {"clay": "clay", "orgc": "oc", "phaq": "ph", "nitkjd": "n", "cecph7": "cec", "tceq": "caco3"}
TOP, BOT, MIN_COVER = 0.0, 30.0, 24.0   # depth interval (cm) and the minimum covered thickness (80%)
CALC_THRESHOLD = 1.0          # g/kg (0.1 %) CaCO3 equivalent: a profile whose 0-30 cm mean reaches it is calcareous


def _fetch(url, dest):
    """Download with curl (uses the OS certificate store; more robust behind proxies than urllib)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    print(f"  downloading {dest.name}")
    part = dest.with_name(dest.name + ".part")           # an interrupted download never looks complete
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "--max-time", "3600", "-o", str(part), url], check=True)
    os.replace(part, dest)


def download(with_tabicl=True, tabicl_only=False):
    """WoSIS + WorldClim (only needed to REBUILD the table)
    and the TabICLv2 regressor checkpoint (about 110 MB, from Hugging Face)."""
    for layer in ([] if tabicl_only else LAYERS):
        _fetch(WFS.format(layer=layer), DATA / f"wosis_{layer}.zip")
    for name in ([] if tabicl_only else ("wc2.1_5m_bio.zip", "wc2.1_2.5m_elev.zip")):
        _fetch(WORLDCLIM.format(name=name), DATA / name)
        with zipfile.ZipFile(DATA / name) as z:
            z.extractall(DATA / "worldclim")
    if with_tabicl or tabicl_only:
        try:
            _fetch(TABICL_URL, TABICL_CKPT)
        except subprocess.CalledProcessError:
            print("  could not download the TabICL checkpoint (huggingface.co blocked?). tabicl will try "
                  "its own download at first use; or place the file in", MODELS)


def _read_layers(name):
    """Rows (layers) of one WoSIS property zip: CONUS box, United States, mineral layers only."""
    with zipfile.ZipFile(DATA / f"wosis_{name}.zip") as z:
        csv = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(csv), low_memory=False)
    df = df[df["country_name"] == "United States of America"]
    df = df[(df.X > -125) & (df.X < -66) & (df.Y > 24) & (df.Y < 50)]
    return df[df["organic_surface"].fillna(0).astype(int) == 0]


def _aggregate(df, col):
    """Thickness-weighted 0-30 cm mean per profile (a Series named `col`, indexed by profile_id);
    missing where less than MIN_COVER cm of the interval has a value."""
    up = df["upper_depth"].clip(lower=TOP)
    lo = df["lower_depth"].clip(upper=BOT)
    w = (lo - up).clip(lower=0)
    d = pd.DataFrame({"profile_id": df["profile_id"], "w": w, "wv": w * df["value_avg"]})
    g = d[d["w"] > 0].groupby("profile_id").agg(w=("w", "sum"), wv=("wv", "sum"))
    return (g["wv"] / g["w"]).where(g["w"] >= MIN_COVER).rename(col)


def _year(s):
    """Year from a WoSIS date string ('1987-06-01', '1987'), else NaN."""
    m = re.match(r"(\d{4})", str(s))
    return float(m.group(1)) if m else np.nan


def prep():
    """Build data/conus_0_30.parquet from the downloaded WoSIS and WorldClim files (see the module
    docstring for the steps). Returns the table. Needs rasterio."""
    import rasterio
    from rasterio.warp import transform

    cols, meta = [], []
    for name, col in LAYERS.items():
        path = DATA / f"wosis_{name}.zip"
        if not path.exists():
            print(f"  missing {path.name}; skipping {col}")
            continue
        df = _read_layers(name)
        cols.append(_aggregate(df, col))
        meta.append(df[["profile_id", "X", "Y", "dataset_id", "date"]])
        print(f"  {name}: {df.profile_id.nunique()} CONUS profiles")
    tab = pd.concat(cols, axis=1).join(pd.concat(meta).drop_duplicates("profile_id").set_index("profile_id"))
    tab["year"] = tab.pop("date").map(_year)

    tab = tab.dropna(subset=["clay", "oc", "ph"])
    tab = tab[(tab.clay > 0) & (tab.clay < 100) & (tab.oc > 0) & (tab.oc < 120) & (tab.ph > 2) & (tab.ph < 11)]
    for c in ("n", "cec"):
        tab.loc[~(tab[c] > 0), c] = np.nan
    # calcareous flag: NCSS analyses carbonate only when the sample fizzes, so a missing value is read
    # as non-calcareous (documented assumption); the flag gates the CEC relation
    if "caco3" in tab:
        tab["calc"] = (tab["caco3"].fillna(0) >= CALC_THRESHOLD)
    else:
        tab["calc"] = False

    xs, ys = transform("EPSG:4326", "EPSG:5070", tab.X.tolist(), tab.Y.tolist())
    tab["x_km"], tab["y_km"] = np.array(xs) / 1000, np.array(ys) / 1000
    for size, name in ((100, "block"), (300, "region")):
        tab[name] = grid_id(tab.x_km, tab.y_km, size)

    pts = list(zip(tab.X, tab.Y))
    wc = DATA / "worldclim"
    for i in range(1, 20):
        with rasterio.open(wc / f"wc2.1_5m_bio_{i}.tif") as r:
            v = np.array([s[0] for s in r.sample(pts)], float)
            v[v == r.nodata] = np.nan
            tab[f"bio{i}"] = v
    with rasterio.open(wc / "wc2.1_2.5m_elev.tif") as r:
        elev = r.read(1).astype(float)
        elev[elev == r.nodata] = np.nan
        rows, cols_ = (np.asarray(a) for a in rasterio.transform.rowcol(r.transform, tab.X.values, tab.Y.values))
        res = r.res[0]
    tab["elev"] = elev[rows, cols_]
    dy = res * 111_320.0
    dx = dy * np.cos(np.deg2rad(tab.Y.values))
    dzdx = (elev[rows, cols_ + 1] - elev[rows, cols_ - 1]) / (2 * dx)
    dzdy = (elev[rows - 1, cols_] - elev[rows + 1, cols_]) / (2 * dy)
    tab["slope"] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    win = np.stack([elev[rows + a, cols_ + b] for a in range(-2, 3) for b in range(-2, 3)])
    tab["relief"] = np.nanmax(win, 0) - np.nanmin(win, 0)

    tab = tab.dropna(subset=COVS)
    tab.index.name = "profile_id"
    tab.reset_index().to_parquet(TABLE, index=False)
    print(f"\n  profiles: {len(tab)}  with N: {tab.n.notna().sum()}  with CEC7: {tab.cec.notna().sum()}  "
          f"calcareous: {int(tab.calc.sum())}  blocks: {tab.block.nunique()}  regions: {tab.region.nunique()}")
    return tab


def sha256(path) -> str:
    """Hex sha256 of a file (read in 1 MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_path() -> Path:
    """data/conus_0_30.parquet if it was rebuilt locally, else the pilot snapshot, if present."""
    return TABLE if TABLE.exists() else SNAPSHOT


def snapshot_status(path=None) -> tuple[Path, bool]:
    """(path of the table, whether it is byte-identical to the pilot snapshot). path defaults to
    table_path(); a missing file counts as not identical."""
    path = table_path() if path is None else Path(path)
    return path, path.exists() and sha256(path) == SNAPSHOT_SHA256


def load_table() -> pd.DataFrame:
    """The profile table every run uses: a local rebuild if present, else the pilot snapshot. Prints
    whether it is byte-identical to the pilot snapshot (a rebuilt table usually is not)."""
    path = table_path()
    if not path.exists():
        raise FileNotFoundError(f"no data table: neither {TABLE} nor the snapshot {SNAPSHOT} exists "
                                "(run: python -m pedopilot download && python -m pedopilot prep, or see data_snapshot/README.md)")
    _, same = snapshot_status(path)
    print(f"data: {path.relative_to(path.parents[1])} ({'identical to' if same else 'DIFFERS from'} the pilot snapshot)")
    return pd.read_parquet(path)
