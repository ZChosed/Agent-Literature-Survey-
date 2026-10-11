"""
calibrate -- does the screening model agree with a human, before it screens the corpus?

  litreview calibrate draw    stratified sample + blind labelling sheet
  litreview calibrate score   kappa / recall / precision of the latest sample run

draw: from eligible candidates WITH abstract text, `calibration_per_stratum` from
the top third of the ranked list and as many from the bottom two-thirds, fixed
seed, shuffled. The head is oversampled so includes are numerous enough to
estimate recall, so kappa here is a kappa ON THE STRATIFIED SAMPLE. The blind
sheet (screening/labels_TODO.csv) carries paper_id, title, abstract and nothing
else: a rater who sees rank or seed links is no longer independent. Fill it and
save it as screening/labels.csv. An existing sample is never overwritten; it is
moved to screening/archive/<timestamp>/ first.

score: binary kappa (uncertain counts as include: both keep the paper) is primary;
include recall weighs most, because a false exclude is never seen again. A
partly filled sheet is refused. Labelled papers the run excluded at eligibility
are dropped from the denominator. The run passes only if kappa >= kappa_floor AND
recall >= recall_floor. If the labels were revised after model output was seen,
the result is no longer an independent calibration -- record that.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

from . import eligibility as elig
from .common import latest_run, provenance, read_csv, read_jsonl, read_json, write_csv, write_json

DECISIONS = ["include", "exclude", "uncertain"]


def kappa(a: list[str], b: list[str], labels: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return None if abs(1 - pe) < 1e-12 else round((po - pe) / (1 - pe), 4)


def binary(d: str) -> str:
    return "include" if d == "uncertain" else d


def draw(cfg, args) -> int:
    sc = cfg["screening"]
    rows = read_csv(cfg.candidates)
    eg = elig.load(cfg, rows)
    pool = [r for r in rows if eg[r["openalex_id"]]["eligible"]
            and r.get("abstract_source") in ("openalex", "semantic_scholar") and r["abstract"].strip()]
    n = int(sc["calibration_per_stratum"])
    cut = len(pool) // 3
    top, bottom = pool[:cut], pool[cut:]
    if len(top) < n or len(bottom) < n:
        sys.exit(f"pool too small: top third {len(top)}, bottom {len(bottom)}, need {n} each")
    rng = random.Random(int(sc["calibration_seed"]))
    sample = ([dict(r, stratum="top_third") for r in rng.sample(sorted(top, key=lambda r: r["openalex_id"]), n)]
              + [dict(r, stratum="bottom_two_thirds") for r in rng.sample(sorted(bottom, key=lambda r: r["openalex_id"]), n)])
    rng.shuffle(sample)

    sdir = cfg.path("screening")
    old = [f for f in ("calibration_sample.csv", "labels_TODO.csv", "labels.csv") if os.path.exists(os.path.join(sdir, f))]
    if old:
        arch = os.path.join(sdir, "archive", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        os.makedirs(arch)
        for f in old:
            shutil.move(os.path.join(sdir, f), os.path.join(arch, f))
        print(f"[calibrate] previous sample archived to {arch}")
    for r in sample:
        r["paper_id"] = r["openalex_id"]
    write_csv(os.path.join(sdir, "calibration_sample.csv"), sample,
              ["paper_id", "doi", "title", "abstract", "year", "venue", "abstract_source", "stratum"])
    write_csv(os.path.join(sdir, "labels_TODO.csv"),
              [{**r, "human_decision": "", "human_criterion_id": "", "human_notes": ""} for r in sample],
              ["paper_id", "title", "abstract", "human_decision", "human_criterion_id", "human_notes"])
    provenance(cfg, "calibration_sample", {
        "seed": sc["calibration_seed"], "per_stratum": n, "pool": len(pool),
        "pool_rule": "eligible AND abstract_source in openalex|semantic_scholar AND abstract non-empty",
        "strata": {"top_third": [1, cut], "bottom_two_thirds": [cut + 1, len(pool)]},
        "paper_ids": sorted(r["paper_id"] for r in sample)})
    print(f"[calibrate] {2 * n} papers drawn from a pool of {len(pool)} (seed {sc['calibration_seed']})")
    print(f"  -> {os.path.join(sdir, 'labels_TODO.csv')}  fill human_decision "
          f"(include/exclude/uncertain) + human_criterion_id, save as screening/labels.csv")
    return 0


def score(cfg, args) -> int:
    sc = cfg["screening"]
    run = args.run or latest_run(cfg.path("screening", "runs"), "sample_")
    if not run:
        sys.exit("no sample screening run -- run `litreview screen --sample`")
    labels_path = cfg.path("screening", "labels.csv")
    if not os.path.exists(labels_path):
        sys.exit("screening/labels.csv not found -- fill labels_TODO.csv and save it under that name")
    labels = read_csv(labels_path)
    bad = [r["paper_id"] for r in labels if r["human_decision"].strip().lower() not in DECISIONS]
    if bad:
        sys.exit(f"REFUSING TO SCORE: {len(bad)} row(s) without a valid human_decision "
                 f"(include/exclude/uncertain), e.g. {bad[:5]}. A kappa over a partly filled "
                 f"sheet is a kappa over the easiest papers.")
    manifest = read_json(os.path.join(run, "manifest.json"), {})
    inel = set((manifest.get("excluded_at_eligibility") or {}).get("paper_ids") or {})
    calls = {c["paper_id"]: c for c in read_jsonl(os.path.join(run, "calls.jsonl"))}
    scored = [(r, calls[r["paper_id"]]) for r in labels if r["paper_id"] not in inel
              and r["paper_id"] in calls and calls[r["paper_id"]]["validation_status"] in ("valid", "forced")]
    h = [r["human_decision"].strip().lower() for r, _ in scored]
    m = [c["parsed"]["decision"] for _, c in scored]
    hb, mb = [binary(x) for x in h], [binary(x) for x in m]
    tp = sum(x == y == "include" for x, y in zip(hb, mb))
    fn = sum(x == "include" and y == "exclude" for x, y in zip(hb, mb))
    fp = sum(x == "exclude" and y == "include" for x, y in zip(hb, mb))
    recall = round(tp / (tp + fn), 4) if tp + fn else None
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    kb = kappa(hb, mb, ["include", "exclude"])
    passed = bool(kb is not None and recall is not None
                  and kb >= sc["kappa_floor"] and recall >= sc["recall_floor"])
    dis = [{"paper_id": r["paper_id"], "title": r.get("title", ""), "human": r["human_decision"],
            "human_criterion": r.get("human_criterion_id", ""), "model": c["parsed"]["decision"],
            "model_criterion": c["parsed"].get("criterion_id", ""), "model_reason": c["parsed"].get("reason", ""),
            "direction": ("FALSE_EXCLUDE" if binary(r["human_decision"].lower()) == "include"
                          and binary(c["parsed"]["decision"]) == "exclude" else
                          "false_include" if binary(r["human_decision"].lower()) == "exclude"
                          and binary(c["parsed"]["decision"]) == "include" else "uncertain_boundary")}
           for r, c in scored if r["human_decision"].strip().lower() != c["parsed"]["decision"]]
    write_csv(os.path.join(run, "disagreements.csv"), sorted(dis, key=lambda d: d["direction"]),
              ["direction", "paper_id", "title", "human", "human_criterion", "model",
               "model_criterion", "model_reason"])
    result = {"run": os.path.basename(run), "labels": len(labels), "scored": len(scored),
              "excluded_at_eligibility": len(set(r["paper_id"] for r in labels) & inel),
              "invalid_responses": sum(1 for r in labels if r["paper_id"] in calls
                                       and calls[r["paper_id"]]["validation_status"] == "invalid"),
              "kappa_binary": kb, "kappa_3class": kappa(h, m, DECISIONS),
              "include_recall": recall, "include_precision": precision,
              "true_include": tp, "false_exclude": fn, "false_include": fp,
              "confusion": {f"{x}->{y}": n for (x, y), n in Counter(zip(h, m)).items()},
              "floors": {"kappa": sc["kappa_floor"], "recall": sc["recall_floor"]}, "passed": passed,
              "cost_usd": manifest.get("actual_cost_usd")}
    write_json(os.path.join(run, "calibration.json"), result)
    provenance(cfg, "calibration", result)
    print(f"[calibrate] {len(scored)}/{len(labels)} scored: binary kappa {kb}, include recall {recall} "
          f"({tp}/{tp + fn}), precision {precision}; {len(dis)} disagreements")
    print(f"[calibrate] {'PASS' if passed else 'FAIL'} (floors: kappa >= {sc['kappa_floor']}, "
          f"recall >= {sc['recall_floor']})" + ("" if passed else
          " -- revise criteria.md using disagreements.csv and rerun; do not lower a floor after seeing results"))
    print(f"  -> {os.path.join(run, 'disagreements.csv')}")
    return 0
