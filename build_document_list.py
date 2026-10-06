#!/usr/bin/env python3
"""
build_document_list.py -- turn a full-corpus screening run into the document list.

Python counts (proposal S3.3). Every number here is computed from files on disk:
candidates.csv, eligibility.csv, and the run's calls.jsonl and manifest.json. No
model is consulted.

Every candidate gets exactly ONE outcome
----------------------------------------
    included                 Opus answered `include`, response valid
    needs_human_review       Opus answered `uncertain`; OR the row was title-only
                             and forced to `uncertain` without a model call; OR
                             the response failed validation
    excluded_at_screening    Opus answered `exclude`, response valid
    excluded_at_eligibility  no OA copy (eligibility.csv); never screened

A failed response is routed to a human, NEVER treated as an exclude. Screening is
the one stage that removes papers on model output, and a false exclude is not
recoverable; a response that could not be read is not evidence for exclusion.

Outputs (in --out-dir, default screening/final/)
------------------------------------------------
    screening_outcomes.csv     all candidates, rank order, one row each
    document_list.csv          `included` only -- the papers going forward
    needs_human_review.csv     the rows a human must decide before the list
                               is final
    document_list_summary.json the counts, the run's identity, and the
                               calibration status of the model that screened

The document list is not final until needs_human_review.csv has been decided.

Usage
-----
    python build_document_list.py --run screening/runs/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

from eligibility import load_eligibility

SCRIPT_VERSION = "1.0.0"

OUTCOMES = ("included", "needs_human_review", "excluded_at_screening",
            "excluded_at_eligibility")

OUT_COLUMNS = ["rank", "paper_id", "doi", "title", "year", "venue",
               "outcome", "outcome_basis", "decision", "criterion_id", "reason",
               "text_kind", "validation_status", "validation_errors",
               "exclusion_reason", "linked_to_framework_core", "seed_weight"]

csv.field_size_limit(10_000_000)


def classify(call: dict | None, eligible: bool) -> tuple[str, str]:
    """(outcome, basis) for one candidate. See the module docstring."""
    if not eligible:
        return "excluded_at_eligibility", "no OA copy (eligibility.csv)"
    if call is None:
        return "needs_human_review", "eligible but no screening call logged"
    status = call.get("validation_status")
    decision = ((call.get("parsed") or {}).get("decision") or "").strip().lower()
    if status == "forced":
        return "needs_human_review", "title only: forced uncertain, no model call"
    if status != "valid":
        return "needs_human_review", "model response failed validation"
    if decision == "include":
        return "included", "model include"
    if decision == "uncertain":
        return "needs_human_review", "model uncertain"
    if decision == "exclude":
        return "excluded_at_screening", "model exclude"
    return "needs_human_review", f"unrecognised decision {decision!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the document list from a "
                                             "full-corpus screening run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--candidates", default="corpus/candidates.csv")
    ap.add_argument("--eligibility", default="corpus/eligibility.csv")
    ap.add_argument("--out-dir", default="screening/final")
    args = ap.parse_args()

    with open(args.candidates, newline="", encoding="utf-8") as f:
        cands = list(csv.DictReader(f))
    elig = load_eligibility(args.eligibility, cands)   # exits if missing/stale

    with open(os.path.join(args.run, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    calls: dict[str, dict] = {}
    with open(os.path.join(args.run, "calls.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                if c.get("repeat", 1) == 1:
                    if c["paper_id"] in calls:
                        sys.exit(f"two repeat-1 calls for {c['paper_id']}; "
                                 f"the run is not a single screening pass")
                    calls[c["paper_id"]] = c
    models = sorted({c["model"] for c in calls.values()})
    if len(models) != 1:
        sys.exit(f"expected one model in the run, found {models}")

    inel_logged = set((manifest.get("excluded_at_eligibility") or {})
                      .get("paper_ids") or {})
    leaked = sorted(inel_logged & set(calls))
    if leaked:
        sys.exit(f"ineligible papers have screening calls: {leaked[:10]}")
    unknown = sorted(set(calls) - {r["openalex_id"] for r in cands})
    if unknown:
        sys.exit(f"calls for papers not in candidates.csv: {unknown[:10]}")

    rows = []
    for rank, r in enumerate(cands, 1):
        pid = r["openalex_id"]
        e = elig[pid]
        call = calls.get(pid)
        outcome, basis = classify(call, e["eligible"])
        p = (call or {}).get("parsed") or {}
        rows.append({
            "rank": rank, "paper_id": pid, "doi": r.get("doi", ""),
            "title": r.get("title", ""), "year": r.get("year", ""),
            "venue": r.get("venue", ""),
            "outcome": outcome, "outcome_basis": basis,
            "decision": p.get("decision", ""),
            "criterion_id": p.get("criterion_id", ""),
            "reason": p.get("reason", ""),
            "text_kind": (call or {}).get("text_kind", ""),
            "validation_status": (call or {}).get("validation_status", ""),
            "validation_errors": "|".join((call or {}).get("validation_errors") or []),
            "exclusion_reason": e["exclusion_reason"],
            "linked_to_framework_core": r.get("linked_to_framework_core", ""),
            "seed_weight": r.get("seed_weight", ""),
        })

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name: str, subset: list[dict]) -> str:
        path = os.path.join(args.out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            w.writeheader()
            w.writerows(subset)
        return path

    p_all = write("screening_outcomes.csv", rows)
    p_inc = write("document_list.csv", [r for r in rows if r["outcome"] == "included"])
    p_rev = write("needs_human_review.csv",
                  [r for r in rows if r["outcome"] == "needs_human_review"])

    by_outcome = Counter(r["outcome"] for r in rows)
    included = [r for r in rows if r["outcome"] == "included"]
    review = [r for r in rows if r["outcome"] == "needs_human_review"]
    result = manifest.get("result") or {}
    summary = {
        "script": "build_document_list.py",
        "script_version": SCRIPT_VERSION,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "run": args.run,
        "run_id": manifest.get("run_id"),
        "model": models[0],
        "criteria_sha256": manifest.get("criteria_sha256"),
        "system_prompt_sha256": manifest.get("system_prompt_sha256"),
        "candidates_total": len(rows),
        "counts_by_outcome": {k: by_outcome[k] for k in OUTCOMES},
        "included_by_criterion": dict(sorted(Counter(
            r["criterion_id"] for r in included).items())),
        "needs_human_review_by_basis": dict(sorted(Counter(
            r["outcome_basis"] for r in review).items())),
        "excluded_at_screening_by_criterion": dict(sorted(Counter(
            r["criterion_id"] for r in rows
            if r["outcome"] == "excluded_at_screening").items())),
        "framework_core_linked": {
            k: sum(1 for r in rows if r["outcome"] == k
                   and str(r["linked_to_framework_core"]).strip() == "1")
            for k in OUTCOMES},
        "validation_warning_kinds": result.get("validation_warning_kinds", {}),
        "actual_cost_usd": result.get("actual_cost_usd"),
        "list_is_final": False,
        "list_is_final_basis": (
            "document_list.csv holds model includes only. It is final once a "
            "human has decided every row of needs_human_review.csv."),
        "screening_model_calibration_status": (
            "NOT independently calibrated. On the only labels made before any "
            "model output was seen (v1, calib_v3_20261006) Opus 5.5 scored "
            "binary kappa 0.59 and include recall 0.67, below the "
            "pre-registered floors (0.70 / 0.95). Its 0.93 / 1.00 on label v3 "
            "is against labels revised toward its own answers. See README "
            "'Rescoring against revised labels'. Expect some false excludes "
            "in excluded_at_screening."),
        "outputs": {"all": p_all, "document_list": p_inc, "needs_human_review": p_rev},
    }
    sp = os.path.join(args.out_dir, "document_list_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"[list] {len(rows)} candidates, screened by {models[0]} "
          f"(run {manifest.get('run_id')})")
    for k in OUTCOMES:
        print(f"[list]   {k:24} {by_outcome[k]}")
    print(f"  -> {p_inc}  ({len(included)} papers)")
    print(f"  -> {p_rev}  ({len(review)} papers: decide these before the list "
          f"is final)")
    print(f"  -> {p_all}")
    print(f"  -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
