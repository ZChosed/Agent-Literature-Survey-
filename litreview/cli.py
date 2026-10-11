"""litreview command line: one subcommand per stage, all reading project.toml."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from . import build, calibrate, config, eligibility, extract, fetch, report, screen
from .common import VERSION, latest_run, read_json

TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")


def init(args) -> int:
    dest = args.dir
    if os.path.exists(os.path.join(dest, "project.toml")):
        sys.exit(f"{dest}/project.toml already exists")
    shutil.copytree(TEMPLATES, dest, dirs_exist_ok=True)
    print(f"[init] new review in {dest}/. Next:\n"
          f"  1. edit project.toml (name, contact_email, roles)\n"
          f"  2. list your base papers in seeds.csv\n"
          f"  3. write criteria.md and your extraction schema(s) in schemas/\n"
          f"  4. cd {dest} && litreview build")
    return 0


def status(cfg, args) -> int:
    prov = read_json(cfg.provenance, {})
    steps = [
        ("build", os.path.exists(cfg.candidates), "litreview build"),
        ("eligibility", os.path.exists(cfg.eligibility), "litreview eligibility"),
        ("calibration sample", os.path.exists(cfg.path("screening", "calibration_sample.csv")), "litreview calibrate draw"),
        ("human labels", os.path.exists(cfg.path("screening", "labels.csv")), "fill screening/labels_TODO.csv -> labels.csv"),
        ("sample screened", bool(latest_run(cfg.path("screening", "runs"), "sample_")), "litreview screen --sample"),
        ("calibration scored", "calibration" in prov, "litreview calibrate score"),
        ("corpus screened", bool(latest_run(cfg.path("screening", "runs"), "corpus_")), "litreview screen"),
        ("documents fetched", "fetch" in prov, "litreview fetch"),
        ("extracted", "extraction" in prov, "litreview extract"),
        ("report", os.path.exists(cfg.path("report.md")), "litreview report"),
    ]
    nxt = None
    for name, done, cmd in steps:
        print(f"  [{'x' if done else ' '}] {name}")
        if not done and nxt is None:
            nxt = cmd
    if "calibration" in prov:
        c = prov["calibration"]
        print(f"  calibration: kappa {c['kappa_binary']}, recall {c['include_recall']} -> "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    if "fetch" in prov and prov["fetch"].get("missing"):
        print(f"  {prov['fetch']['missing']} document(s) to retrieve by hand: manual_pdfs/RETRIEVE.md")
    print(f"next: {nxt}" if nxt else "all stages done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="litreview", description=f"Literature review pipeline v{VERSION}")
    ap.add_argument("--project", default=".", help="project directory (default: current)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="scaffold a new review")
    p.add_argument("dir")
    sub.add_parser("status", help="what exists, what is next")
    sub.add_parser("build", help="seeds -> candidates")
    p = sub.add_parser("eligibility", help="open-access rule -> data/eligibility.csv")
    p.add_argument("--offline", action="store_true", help="replay the call log only")
    p = sub.add_parser("calibrate", help="draw a blind sample, or score a sample run")
    p.add_argument("action", choices=("draw", "score"))
    p.add_argument("--run", help="sample run directory (default: latest)")
    for name, h in (("screen", "screen papers"), ("extract", "extract from full texts")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--sync", action="store_true", help="no batching (smoke tests)")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--only", help="comma-separated paper ids")
        p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    sub.choices["screen"].add_argument("--sample", action="store_true", help="the calibration sample")
    sub.choices["extract"].add_argument("--recheck", action="store_true", help="re-run evidence checks, no API")
    p = sub.add_parser("fetch", help="document list + full texts")
    p.add_argument("--run", help="corpus screening run (default: latest)")
    p.add_argument("--missing-abstracts", action="store_true")
    p.add_argument("--manual-only", action="store_true")
    p.add_argument("--retry", action="store_true", help="retry papers that failed before")
    p = sub.add_parser("report", help="tables + report.md")
    p.add_argument("--scaffold", action="store_true")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "init":
        return init(args)
    cfg = config.load(args.project)
    if args.cmd == "calibrate":
        return (calibrate.draw if args.action == "draw" else calibrate.score)(cfg, args)
    return {"status": status, "build": build.run, "eligibility": eligibility.run,
            "screen": screen.run, "fetch": fetch.run, "extract": extract.run,
            "report": report.run}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
