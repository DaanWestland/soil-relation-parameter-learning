"""Command line: python -m pedopilot <command>   (or the `pedopilot` script installed with the package)

  check [--cpu-ok]      environment check (GPU, packages, data table, TabICL checkpoint);
                        --cpu-ok: do not fail on a machine without a CUDA GPU
  download              download WoSIS layers, WorldClim and the TabICL checkpoint (--tabicl-only: the
                        checkpoint only)
  prep                  rebuild data/conus_0_30.parquet from the downloads (not needed for published runs)
  run CONFIG            run an experiment (resumable; CONFIG = configs/<name>.yaml or a path)
  analyze NAME          summarise results/<NAME> (tables, figures, REPORT.md)
  bootstrap NAME        region-level cluster bootstrap of the formula contrasts (as python -m pedopilot.bootstrap)
  validate              positive/negative controls on synthetic data (known answer)
  publish NAME          copy the reports of results/<NAME> (md, csv, json, png, yaml, log; not the raw
                        parquet or meta files) to the tracked reports/<NAME>, replacing that folder
  all CONFIG            download + prep only if no data table exists (the snapshot counts), then run + analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

from . import paths

EPILOG = """examples (reproduce a published run; see README.md):
  python -m pedopilot run gpu_full && python -m pedopilot analyze gpu_full
  python -m pedopilot bootstrap gpu_full --learner tabicl
  python -m pedopilot run cpu_qrf_w1 && python -m pedopilot analyze cpu_qrf_w1
  python -m pedopilot validate --learners qrf,tabm --device cpu
      (expected: "CHECK FAILED (22/23 checks)" and exit code 1, as published: the small CPU TabM
      misses the negative control, and TabM values vary run to run; QRF alone passes 13/13)

environment:
  PEDOPILOT_ROOT   folder holding configs/, data/, data_snapshot/, results/ and models/
                   (default: the repository)
"""


def check(cpu_ok=False):
    """Print the environment status; returns False if something required is missing or broken."""
    import torch
    ok = True
    cuda = torch.cuda.is_available()
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda: {cuda}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)" if cuda else ""))
    if not cuda:
        print("  " + ("note" if cpu_ok else "FAIL") + ": torch sees no CUDA GPU. For the GPU runs install a CUDA "
              "build of torch (README, Installation); --cpu-ok accepts a CPU-only machine.")
        ok &= cpu_ok
    for mod in ("sklearn", "scipy", "scoringrules", "tabm", "rtdl_num_embeddings", "tabicl", "rasterio", "yaml", "tabulate",
                "transformers", "statsmodels"):
        found = importlib.util.find_spec(mod) is not None
        req = mod not in ("transformers", "statsmodels")
        print(f"  {'ok ' if found else ('MISSING' if req else 'optional-missing')}  {mod}"
              + ("" if found or req else "  (needed for tabicl_ft / the mixed model)"))
        ok &= found or not req
    from .data import snapshot_status
    tp, same = snapshot_status()
    if tp.exists():
        print(f"  data table: ok  {tp.relative_to(paths.ROOT)}  "
              + ("(identical to the pilot snapshot)" if same else "(DIFFERS from the pilot snapshot: rebuilt data)"))
    else:
        ok = False
        print("  data table: MISSING (build it: python -m pedopilot download && python -m pedopilot prep; or put the pilot snapshot into data_snapshot/, see data_snapshot/README.md)")
    if paths.TABICL_CKPT.exists():
        try:
            ck = torch.load(paths.TABICL_CKPT, map_location="cpu", weights_only=True)
            good = isinstance(ck, dict) and {"config", "state_dict"} <= set(ck)
        except Exception:                      # truncated file or an HTML error page saved as .ckpt
            good = False
        ok &= good
        print(f"  TabICL checkpoint: {'ok' if good else 'CORRUPT (delete models/*.ckpt and download again)'}")
    else:
        print("  TabICL checkpoint: not local (tabicl downloads it from Hugging Face at first use, "
              "or: python -m pedopilot download --tabicl-only)")
    print("check:", "OK" if ok else "FAILED")
    return ok


def publish(name):
    """Copy the human-readable outputs of results/<name> (ignored by git) to reports/<name> (tracked).
    Deletes an existing reports/<name> first."""
    import shutil
    src, dst = paths.RESULTS / name, paths.ROOT / "reports" / name
    if not src.exists():
        sys.exit(f"no results/{name}")
    shutil.rmtree(dst, ignore_errors=True)
    n = 0
    for f in src.rglob("*"):
        if f.is_file() and f.suffix in {".md", ".csv", ".json", ".png", ".yaml", ".log"} \
                and not f.name.startswith("meta_") and len(f.relative_to(src).parts) <= 2:
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    print(f"published {n} files to {dst.relative_to(paths.ROOT)} (git add reports/{name} to keep them)")


def _parser():
    ap = argparse.ArgumentParser(prog="pedopilot", description=__doc__, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="environment check (GPU, packages, data table, TabICL checkpoint)")
    c.add_argument("--cpu-ok", action="store_true", help="do not fail on a machine without a CUDA GPU")
    d = sub.add_parser("download", help="download WoSIS, WorldClim and the TabICLv2 checkpoint")
    d.add_argument("--no-tabicl", action="store_true", help="skip the TabICLv2 checkpoint")
    d.add_argument("--tabicl-only", action="store_true", help="only the TabICLv2 checkpoint (no data)")
    sub.add_parser("prep", help="rebuild data/conus_0_30.parquet from the downloads (differs from the snapshot)")
    for name, hlp in (("run", "run an experiment (resumable: rerun the same command to continue or retry)"),
                      ("all", "download + prep if no data table exists, then run + analyze")):
        r = sub.add_parser(name, help=hlp)
        r.add_argument("config", help="config name (configs/<name>.yaml) or a path to a YAML file")
        r.add_argument("--folds", default=None, help="comma-separated subset of folds (overrides the config's folds)")
        r.add_argument("--dry-run", action="store_true", help="print the design (profiles and sites per fold) and stop")
    a = sub.add_parser("analyze", help="tables, figures and REPORT.md for results/NAME")
    a.add_argument("name", help="results folder name (the config's name:)")
    a.add_argument("--include-partial", action="store_true", help="also use folds that are not complete")
    b = sub.add_parser("bootstrap", help="region-level cluster bootstrap of structured_ya and structured vs chained")
    b.add_argument("name", help="results folder name")
    b.add_argument("--learner", default="qrf", help="learner of the contrast (default: qrf)")
    b.add_argument("--level", type=float, default=0.03, help="label level (default: 0.03)")
    b.add_argument("--scheme", default="group", help="thinning scheme (default: group)")
    pb = sub.add_parser("publish", help="copy results/NAME reports to the tracked reports/NAME folder")
    pb.add_argument("name", help="results folder name")
    v = sub.add_parser("validate", help="positive/negative controls on synthetic data with a known answer")
    v.add_argument("--learners", default="qrf", help="comma-separated learners (default: qrf)")
    v.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    return ap


def main(argv=None):
    """Entry point of `python -m pedopilot` and the `pedopilot` script. Exits 1 if a check, a
    configuration of `run`/`all` or a validation check failed."""
    args = _parser().parse_args(argv)

    if args.cmd == "check":
        sys.exit(0 if check(args.cpu_ok) else 1)
    if args.cmd == "download":
        from .data import download
        download(with_tabicl=not args.no_tabicl, tabicl_only=args.tabicl_only)
    elif args.cmd == "prep":
        from .data import prep
        prep()
    elif args.cmd in ("run", "all"):
        from .experiment import Config, run
        cfg = Config.load(args.config)
        folds = [int(f) for f in args.folds.split(",")] if args.folds else None
        from .data import table_path
        if args.cmd == "all" and not table_path().exists():
            from .data import download, prep
            download()
            prep()
        run(cfg, only_folds=folds, dry_run=args.dry_run)
        if args.cmd == "all" and not args.dry_run:
            from .analysis import analyze
            print("report:", analyze(cfg.name) / "REPORT.md")
        from .experiment import LAST_FAILED
        if LAST_FAILED:
            sys.exit(1)                               # some configurations failed: rerun the same command
    elif args.cmd == "validate":
        from .validate import validate
        sys.exit(0 if validate(tuple(args.learners.split(",")), args.device) else 1)
    elif args.cmd == "publish":
        publish(args.name)
    elif args.cmd == "analyze":
        from .analysis import analyze
        print("report:", analyze(args.name, include_partial=args.include_partial) / "REPORT.md")
    elif args.cmd == "bootstrap":
        from .bootstrap import bootstrap_contrasts
        bootstrap_contrasts(args.name, args.learner, args.level, args.scheme)


if __name__ == "__main__":
    main()
