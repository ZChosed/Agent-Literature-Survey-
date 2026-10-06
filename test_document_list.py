#!/usr/bin/env python3
"""
test_document_list.py -- offline test for build_document_list.py.

Every outcome path is exercised on a fixture run, and the one rule that matters
most is pinned: a response that failed validation goes to human review, never
to `excluded_at_screening`.

Run:  python test_document_list.py
Exit 0 = all assertions passed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import eligibility as el

FAILS: list[str] = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, label):
    print(f"  {'pass' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


# (id, eligible, call) -- call is (validation_status, decision) or None
FIXTURE = [
    ("W1", True, ("valid", "include")),
    ("W2", True, ("valid", "exclude")),
    ("W3", True, ("valid", "uncertain")),
    ("W4", True, ("forced", "uncertain")),
    ("W5", True, ("invalid", "exclude")),     # failed validation: NOT an exclude
    ("W6", False, None),                      # no OA copy
]
EXPECT = {"W1": "included", "W2": "excluded_at_screening",
          "W3": "needs_human_review", "W4": "needs_human_review",
          "W5": "needs_human_review", "W6": "excluded_at_eligibility"}


def build(d: str, extra_call: str | None = None):
    os.makedirs(d, exist_ok=True)
    cand = os.path.join(d, "candidates.csv")
    rows = [{"openalex_id": i, "doi": f"10.0000/{i}", "title": f"T {i}",
             "year": "2024", "venue": "V", "linked_to_framework_core": "1",
             "seed_weight": "1.0", "eligible": "True" if e else "False",
             "exclusion_reason": "none" if e else "no_oa",
             "eligibility_rule": "fixture"} for i, e, _ in FIXTURE]
    with open(cand, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["openalex_id", "doi", "title", "year",
                                          "venue", "linked_to_framework_core",
                                          "seed_weight"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    elig = os.path.join(d, "eligibility.csv")
    el.write_eligibility_csv(elig, rows, el.candidates_key(rows))

    run = os.path.join(d, "run")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": "fixture", "excluded_at_eligibility":
                   {"n": 1, "paper_ids": {"W6": "no_oa"}},
                   "result": {"actual_cost_usd": 0.01}}, f)
    with open(os.path.join(run, "calls.jsonl"), "w", encoding="utf-8") as f:
        for i, _, call in FIXTURE:
            if call is None:
                continue
            status, decision = call
            f.write(json.dumps({
                "model": "claude-opus-5-5", "paper_id": i, "repeat": 1,
                "validation_status": status,
                "validation_errors": ["reason_too_long:60w"] if status == "invalid" else [],
                "parsed": {"decision": decision, "criterion_id": "I1",
                           "reason": "r"}, "text_kind": "abstract"}) + "\n")
        if extra_call:
            f.write(extra_call + "\n")
    out = os.path.join(d, "final")
    proc = subprocess.run(
        [sys.executable, "build_document_list.py", "--run", run,
         "--candidates", cand, "--eligibility", elig, "--out-dir", out],
        capture_output=True, text=True, cwd=HERE)
    return proc, out


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="doclist_test_")
    try:
        print("\n[outcomes] one outcome per candidate, from Python")
        proc, out = build(os.path.join(tmp, "a"))
        check(proc.returncode == 0, f"builds (rc={proc.returncode})\n{proc.stderr}")
        allr = list(csv.DictReader(open(os.path.join(out, "screening_outcomes.csv"),
                                        encoding="utf-8")))
        got = {r["paper_id"]: r["outcome"] for r in allr}
        check(got == EXPECT, f"every outcome as expected ({got})")
        check(len(allr) == len(FIXTURE), "one row per candidate, none dropped")
        check(got["W5"] == "needs_human_review",
              "a failed-validation response goes to a human, NOT to excluded")
        inc = [r["paper_id"] for r in csv.DictReader(
            open(os.path.join(out, "document_list.csv"), encoding="utf-8"))]
        rev = [r["paper_id"] for r in csv.DictReader(
            open(os.path.join(out, "needs_human_review.csv"), encoding="utf-8"))]
        check(inc == ["W1"], f"document_list.csv holds the include only ({inc})")
        check(rev == ["W3", "W4", "W5"], f"review list ({rev})")
        s = json.load(open(os.path.join(out, "document_list_summary.json"),
                           encoding="utf-8"))
        check(s["counts_by_outcome"] == {"included": 1, "needs_human_review": 3,
                                         "excluded_at_screening": 1,
                                         "excluded_at_eligibility": 1},
              f"summary counts ({s['counts_by_outcome']})")
        check(s["list_is_final"] is False and "NOT independently calibrated"
              in s["screening_model_calibration_status"],
              "the summary says the list is not final and states the model's "
              "calibration status")

        print("\n[guards] inconsistent runs are refused")
        bad = json.dumps({"model": "claude-opus-5-5", "paper_id": "W6", "repeat": 1,
                          "validation_status": "valid",
                          "parsed": {"decision": "include"}})
        proc, _ = build(os.path.join(tmp, "b"), extra_call=bad)
        check(proc.returncode != 0 and "ineligible" in proc.stderr,
              "a call for an ineligible paper aborts the build")
        dup = json.dumps({"model": "claude-opus-5-5", "paper_id": "W1", "repeat": 1,
                          "validation_status": "valid",
                          "parsed": {"decision": "exclude"}})
        proc, _ = build(os.path.join(tmp, "c"), extra_call=dup)
        check(proc.returncode != 0 and "two repeat-1 calls" in proc.stderr,
              "two answers for one paper abort the build")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
