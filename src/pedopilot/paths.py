"""Project paths.

Every path hangs off ROOT, the repository root by default. Set the environment variable PEDOPILOT_ROOT to
use another folder (it must contain configs/ and data_snapshot/; results/ is created): the regression and
validation scripts use this to run in a scratch copy without touching the repository's results/.

    ROOT/data/             downloads and a locally rebuilt table (`download`, `prep`; not in git)
    ROOT/models/           the TabICLv2 checkpoint (`download --tabicl-only`; not in git)
    ROOT/results/<name>/   raw and analysed results of one run (not in git; `publish` copies the reports)
    ROOT/configs/          experiment configurations (<name>.yaml)
    ROOT/data_snapshot/    the exact table every published run used (not distributed; sha256 below)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PEDOPILOT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
TABLE = DATA / "conus_0_30.parquet"            # written by `prep`; used instead of the snapshot when present
# the exact table the pilot results were produced with (tracked in git; WoSIS 'latest' changes over time)
SNAPSHOT = ROOT / "data_snapshot" / "conus_0_30.parquet"
SNAPSHOT_SHA256 = "7a890cb6be3d858894c48cd48c9c28c45aedeec42f335a278c9d42a4473ee624"
TABICL_CKPT = MODELS / "tabicl-regressor-v2-20260212.ckpt"
