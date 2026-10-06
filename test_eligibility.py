#!/usr/bin/env python3
"""
test_eligibility.py -- offline test for eligibility.py.

Nothing here touches the network. The rule is tested as a pure function, and
the script end to end is run with `--offline` against a call log written by the
test, so every OpenAlex and Unpaywall "response" is a fixture.

Covers:

  * a 10.1016 DOI is ineligible EVEN WHEN OpenAlex and Unpaywall both say OA,
    and the prefix match is exact (10.10160 is not Elsevier)
  * rules 2 and 3 on every path, including no-DOI rows and an Unpaywall 404
  * rows are never deleted, reordered or altered: candidates.csv is not
    written at all, and eligibility.csv holds exactly one row per candidate,
    in order, stamped with the candidates key
  * the rule-1-removed counterfactual, against hand counts, with its
    OpenAlex-only lower-bound rows counted
  * provenance counts by reason, by reason x rank stratum, top venues and the
    framework_core fraction, checked against hand counts of the fixture
  * all-or-nothing: an undecidable row or a cache miss writes nothing
  * overrides apply only to rows the rules cannot decide

Run:  python test_eligibility.py
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
    if cond:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILS.append(label)


UP_OA = {"outcome": "ok", "n_oa_locations": 2, "is_oa": True}
UP_CLOSED = {"outcome": "ok", "n_oa_locations": 0, "is_oa": False}
UP_404 = {"outcome": "not_found", "http_status": 404}


def test_rule():
    print("\n[rule] the classification, as a pure function")

    # Rule 1 fires on the DOI alone, whatever the OA evidence says.
    for oa, up in ((True, UP_OA), (False, UP_OA), (True, None), (None, None)):
        got = el.classify("10.1016/j.jmsy.2020.01.001", oa, up)
        check(got == (False, "elsevier", "doi_prefix_10.1016"),
              f"10.1016 is ineligible with openalex_is_oa={oa}, "
              f"unpaywall={'OA' if up else None} (got {got})")
    check(el.classify("10.1016/J.CIRP.2019.1", True, UP_OA)[1] == "elsevier",
          "the prefix match is case-insensitive")
    check(el.classify("10.10160/x.1", True, None)[0] is True,
          "10.10160 is a different registrant and is not caught by rule 1")
    check(el.classify("10.1109/10.1016.x", True, None)[0] is True,
          "10.1016 appearing later in the DOI suffix does not fire rule 1")

    # Rule 3 via OpenAlex, rule 3 via Unpaywall, rule 2 both ways.
    check(el.classify("10.1109/x", True, None) == (True, "none", "openalex_is_oa"),
          "OpenAlex is_oa true -> eligible without consulting Unpaywall")
    check(el.classify("10.1109/x", False, UP_OA)
          == (True, "none", "unpaywall_oa_location"),
          "OpenAlex false but an Unpaywall OA location -> eligible")
    check(el.classify("10.1109/x", False, UP_CLOSED)
          == (False, "no_oa", "no_oa:unpaywall_no_location"),
          "OpenAlex false and no Unpaywall location -> no_oa")
    check(el.classify("10.1109/x", False, UP_404)
          == (False, "no_oa", "no_oa:unpaywall_doi_unknown"),
          "OpenAlex false and Unpaywall 404 -> no_oa, on its own rule label")

    # No DOI: OpenAlex alone.
    check(el.classify("", True, None) == (True, "none", "openalex_is_oa_no_doi"),
          "no DOI, OpenAlex OA -> eligible")
    check(el.classify("", False, None)
          == (False, "no_oa", "no_oa:no_doi_openalex_only"),
          "no DOI, OpenAlex not OA -> no_oa, without Unpaywall")

    # Missing inputs are never guessed around.
    for args, why in ((("10.1109/x", False, None), "no Unpaywall answer"),
                      (("10.1109/x", False, {"outcome": "http_error"}),
                       "an Unpaywall error"),
                      (("10.1109/x", None, UP_CLOSED),
                       "no OpenAlex record and no Unpaywall location"),
                      (("", None, None), "no DOI and no OpenAlex record")):
        try:
            el.classify(*args)
            check(False, f"{why} must raise Undecided")
        except el.Undecided:
            check(True, f"{why} -> Undecided, not a guess")
    check(el.classify("10.48550/arxiv.1", None, UP_OA)
          == (True, "none", "unpaywall_oa_location:openalex_record_missing"),
          "a deleted OpenAlex record is still decided if Unpaywall finds a "
          "location -- rule 2 is a conjunction")

    for bad in ("true", "1", "", "yes"):
        try:
            el.parse_eligible(bad)
            check(False, f"parse_eligible({bad!r}) must raise")
        except ValueError:
            check(True, f"parse_eligible rejects {bad!r}")


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------

# (id, doi, venue, framework_core, openalex is_oa, unpaywall record or None)
FIXTURE = [
    ("W1", "10.1016/j.a.1", "JMSy",      "1", True,  None),       # elsevier, OA anyway
    ("W2", "10.1016/j.a.2", "JMSy",      "0", False, None),       # elsevier
    ("W3", "10.1109/b.3",   "TII",       "1", True,  None),       # eligible
    ("W4", "10.1109/b.4",   "TII",       "0", False, UP_OA),      # eligible via Unpaywall
    ("W5", "10.1007/c.5",   "IJAMT",     "1", False, UP_CLOSED),  # no_oa
    ("W6", "10.1007/c.6",   "IJAMT",     "0", False, UP_404),     # no_oa
    ("W7", "",              "",          "0", False, None),       # no_oa, no DOI
    ("W8", "",              "arXiv",     "0", True,  None),       # eligible, no DOI
    ("W9", "10.1016/j.a.9", "CIRP",      "0", True,  None),       # elsevier, OA anyway
]
# Hand counts. n=9, top third = ranks 1-3 (W1-W3), bottom = W4-W9.
EXPECT_REASON = {"W1": "elsevier", "W2": "elsevier", "W3": "none",
                 "W4": "none", "W5": "no_oa", "W6": "no_oa", "W7": "no_oa",
                 "W8": "none", "W9": "elsevier"}

COLS = ["openalex_id", "doi", "title", "venue", "linked_to_framework_core",
        "abstract", "abstract_source", "oa_pdf_url"]


def write_fixture(d: str, rows=FIXTURE) -> tuple[str, str]:
    cand = os.path.join(d, "candidates.csv")
    with open(cand, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for wid, doi, venue, fc, _, _ in rows:
            w.writerow({"openalex_id": wid, "doi": doi, "title": f"Title {wid}",
                        "venue": venue, "linked_to_framework_core": fc,
                        "abstract": f"Abstract, with a comma, {wid}.",
                        "abstract_source": "openalex", "oa_pdf_url": ""})
    log = os.path.join(d, "calls.jsonl")
    with open(log, "w", encoding="utf-8") as f:
        for wid, doi, _, _, is_oa, up in rows:
            f.write(json.dumps({"source": "openalex", "openalex_id": wid,
                                "outcome": "ok", "is_oa": is_oa}) + "\n")
            if up is not None:
                f.write(json.dumps({"source": "unpaywall", "doi": doi, **up}) + "\n")
    return cand, log


def run(d, cand, log, *extra):
    return subprocess.run(
        [sys.executable, "eligibility.py", "--offline", "--quiet",
         "--candidates", cand, "--log", log,
         "--out", os.path.join(d, "eligibility.csv"),
         "--provenance", os.path.join(d, "prov.json"),
         "--overrides", os.path.join(d, "overrides.csv"), *extra],
        capture_output=True, text=True, cwd=HERE)


def test_end_to_end(tmpdir: str):
    print("\n[e2e] offline run: rows kept, columns added, provenance counted")
    d = os.path.join(tmpdir, "e2e")
    os.makedirs(d)
    cand, log = write_fixture(d)
    before_bytes = open(cand, "rb").read()
    before = list(csv.DictReader(open(cand, encoding="utf-8")))

    proc = run(d, cand, log)
    check(proc.returncode == 0, f"runs offline (rc={proc.returncode})\n"
                                f"{proc.stdout}{proc.stderr}")
    check(open(cand, "rb").read() == before_bytes,
          "candidates.csv is not modified at all")
    epath = os.path.join(d, "eligibility.csv")
    after = list(csv.DictReader(open(epath, encoding="utf-8")))

    # --- rows are never deleted -------------------------------------------
    check(len(after) == len(before) == 9,
          f"no row deleted: {len(before)} candidates, {len(after)} eligibility rows")
    check([r["openalex_id"] for r in after] == [r["openalex_id"] for r in before],
          "one row per candidate, in candidates.csv order")
    check(list(after[0].keys()) == el.ELIGIBILITY_COLUMNS,
          f"eligibility.csv columns (got {list(after[0].keys())})")
    check({r[el.KEY_COLUMN] for r in after} == {el.candidates_key(before)},
          "every row is stamped with the current candidates key")
    check(el.load_eligibility(epath, before)["W1"]["eligible"] is False,
          "and load_eligibility accepts it against the same candidates")
    inel = [r for r in after if r["eligible"] == "False"]
    check(len(inel) == 6, f"all 6 ineligible rows are still present (got {len(inel)})")

    got = {r["openalex_id"]: r["exclusion_reason"] for r in after}
    check(got == EXPECT_REASON, f"every row classified as hand-computed ({got})")
    by = {r["openalex_id"]: r for r in after}
    check(by["W1"]["eligible"] == "False" and by["W1"]["eligibility_rule"]
          == "doi_prefix_10.1016",
          "W1: a 10.1016 DOI that OpenAlex says is OA is still ineligible")
    check(all((r["eligible"] == "True") == (r["exclusion_reason"] == "none")
              for r in after),
          "eligible and exclusion_reason never disagree")

    prov = json.load(open(os.path.join(d, "prov.json"), encoding="utf-8"))["eligibility"]
    check(prov["counts_by_exclusion_reason"] == {"elsevier": 3, "no_oa": 3, "none": 3},
          f"counts by reason (got {prov['counts_by_exclusion_reason']})")
    x = prov["counts_by_exclusion_reason_x_rank_stratum"]
    check((x["top_third"]["elsevier"], x["top_third"]["no_oa"], x["top_third"]["none"])
          == (2, 0, 1) and x["top_third"]["total"] == 3,
          f"top third W1-W3: elsevier 2, no_oa 0, eligible 1 (got {x['top_third']})")
    check((x["bottom_two_thirds"]["elsevier"], x["bottom_two_thirds"]["no_oa"],
           x["bottom_two_thirds"]["none"]) == (1, 3, 2),
          f"bottom W4-W9: elsevier 1, no_oa 3, eligible 2 "
          f"(got {x['bottom_two_thirds']})")
    v = [(t["venue"], t["excluded"]) for t in prov["top_excluded_venues"]]
    check(v == [("IJAMT", 2), ("JMSy", 2), ("(no venue)", 1), ("CIRP", 1)],
          f"venues by count, ties alphabetical, blank named (got {v})")
    fc = prov["framework_core_linked"]
    check(fc["total"] == 3 and fc["excluded"] == 2
          and abs(fc["fraction_excluded"] - 0.6667) < 1e-4,
          f"framework_core: W1, W3, W5 linked, W1 and W5 excluded -> 2/3 "
          f"(got {fc})")
    check(prov["elsevier_rows_openalex_is_oa_true"] == 2,
          "the cost of rule 1 is measured: W1 and W9 were OA and excluded anyway")
    check(prov["candidates_modified"] is False and prov["rows_written"] == 9,
          "provenance states candidates.csv was not modified, 9 rows written")

    # --- the counterfactual, rule 1 removed --------------------------------
    # W1, W9: 10.1016 and OpenAlex OA -> eligible. W2: 10.1016, OpenAlex
    # closed, no Unpaywall answer -> no_oa on OpenAlex alone (lower bound).
    # Others as in the real run. Eligible: W1 W3 W4 W8 W9 = 5; no_oa: W2 W5
    # W6 W7 = 4. Top third W1-W3: W2 excluded (1/3). Bottom W4-W9: W5 W6 W7
    # (3/6). framework_core W1 W3 W5: W5 excluded -> 1/3.
    full = json.load(open(os.path.join(d, "prov.json"), encoding="utf-8"))
    cf = full["eligibility_counterfactual_no_rule1"]
    check(cf["eligible"] == 5 and cf["counts_by_exclusion_reason"]
          == {"elsevier": 0, "no_oa": 4, "none": 5},
          f"counterfactual eligible 5, no_oa 4, elsevier 0 "
          f"(got {cf['counts_by_exclusion_reason']})")
    cx = cf["counts_by_exclusion_reason_x_rank_stratum"]
    check(cx["top_third"]["excluded"] == 1 and cx["bottom_two_thirds"]["excluded"] == 3,
          f"counterfactual excluded by stratum: top 1, bottom 3 "
          f"(got {cx['top_third']['excluded']}, {cx['bottom_two_thirds']['excluded']})")
    check(abs(cf["framework_core_linked"]["fraction_excluded"] - 0.3333) < 1e-4,
          f"counterfactual framework_core fraction 1/3 "
          f"(got {cf['framework_core_linked']['fraction_excluded']})")
    check([(t["venue"], t["excluded"]) for t in cf["top_excluded_venues"]]
          == [("IJAMT", 2), ("(no venue)", 1), ("JMSy", 1)],
          f"counterfactual venues (got {cf['top_excluded_venues']})")
    cov = cf["unpaywall_coverage_10_1016"]
    check(cov["rows_excluded_on_openalex_alone"] == 1 and cf["exact"] is False,
          "W2 has no Unpaywall answer: counted as excluded on OpenAlex alone and "
          "the block is marked NOT exact -- the caveat is measured, not stated")
    check("LOWER bound" in cov["basis"], "and its basis says lower bound")
    check(proc.returncode == 0,
          "a missing counterfactual-only lookup does not block --offline")
    check(cf["delta_vs_actual"]["eligible"] == 2,
          "delta vs actual: +2 eligible")
    check("COUNTERFACTUAL" in cf["status"], "the block says it decides nothing")
    real = {r["openalex_id"]: r["exclusion_reason"]
            for r in csv.DictReader(open(epath, encoding="utf-8"))}
    check(real == EXPECT_REASON,
          "and the real decisions in eligibility.csv are unchanged by it "
          "(W1, W9 still elsevier)")

    # With W2's Unpaywall answer in the log the counterfactual becomes exact,
    # and the REAL decision for W2 does not move (rule 1 still fires first).
    log2 = os.path.join(d, "calls_cf.jsonl")
    shutil.copy(log, log2)
    with open(log2, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "unpaywall", "doi": "10.1016/j.a.2",
                            **UP_OA}) + "\n")
    d2 = os.path.join(d, "cf_exact")
    os.makedirs(d2)
    run(d2, cand, log2)
    cf2 = json.load(open(os.path.join(d2, "prov.json"),
                         encoding="utf-8"))["eligibility_counterfactual_no_rule1"]
    check(cf2["exact"] is True
          and cf2["unpaywall_coverage_10_1016"]["rows_excluded_on_openalex_alone"] == 0,
          "with every 10.1016 closed row answered, the block is marked exact")
    check(cf2["eligible"] == 6 and "EXACT" in cf2["unpaywall_coverage_10_1016"]["basis"],
          f"and W2's OA location moves it to eligible in the counterfactual "
          f"(6, got {cf2['eligible']})")
    real2 = {r["openalex_id"]: r["exclusion_reason"] for r in
             csv.DictReader(open(os.path.join(d2, "eligibility.csv"), encoding="utf-8"))}
    check(real2 == EXPECT_REASON,
          "while the real decision for W2 stays elsevier -- the lookup feeds "
          "the counterfactual only")

    # Idempotent: a second offline run changes nothing.
    snap = open(epath, "rb").read()
    run(d, cand, log)
    check(open(epath, "rb").read() == snap, "a rerun is byte-identical")
    check(open(cand, "rb").read() == before_bytes, "and candidates.csv still untouched")

    # A candidates.csv that still carries the old inline columns is refused.
    legacy = os.path.join(d, "legacy.csv")
    with open(legacy, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["eligible"])
        w.writeheader()
        for r in before:
            w.writerow({**r, "eligible": "True"})
    proc = run(d, legacy, log)
    check(proc.returncode != 0 and "eligibility columns" in proc.stderr,
          "a candidates.csv carrying eligibility columns is refused")


def test_all_or_nothing(tmpdir: str):
    print("\n[abort] undecidable rows and cache misses write nothing")
    d = os.path.join(tmpdir, "abort")
    os.makedirs(d)
    rows = list(FIXTURE)
    # W5 loses its Unpaywall answer: OpenAlex said not-OA, nothing else known.
    rows[4] = ("W5", "10.1007/c.5", "IJAMT", "1", False, None)
    cand, log = write_fixture(d, rows)
    before = open(cand, "rb").read()

    proc = run(d, cand, log)
    check(proc.returncode != 0, f"a cache miss under --offline aborts "
                                f"(rc={proc.returncode})")
    check(open(cand, "rb").read() == before, "and candidates.csv is untouched")
    check(not os.path.exists(os.path.join(d, "eligibility.csv")),
          "and no eligibility.csv is written")
    check(not os.path.exists(os.path.join(d, "prov.json")),
          "and no provenance is written")

    # An Unpaywall error is in the cache but is not an answer.
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "unpaywall", "doi": "10.1007/c.5",
                            "outcome": "http_error", "http_status": 503}) + "\n")
    proc = run(d, cand, log)
    check(proc.returncode == 5 and "could not be decided" in proc.stdout,
          f"a row resting on an HTTP error is undecided (rc={proc.returncode})")
    check(not os.path.exists(os.path.join(d, "eligibility.csv")),
          "and still nothing is written")


def test_overrides(tmpdir: str):
    print("\n[override] judgments only where the rules cannot decide")
    d = os.path.join(tmpdir, "ov")
    os.makedirs(d)
    rows = list(FIXTURE)
    # W8 becomes a deleted OpenAlex record with no DOI: undecidable.
    rows[7] = ("W8", "", "arXiv", "0", None, None)
    cand, log = write_fixture(d, rows)
    with open(log, "w", encoding="utf-8") as f:
        for wid, doi, _, _, is_oa, up in rows:
            rec = ({"source": "openalex", "openalex_id": wid, "outcome": "not_found"}
                   if is_oa is None else
                   {"source": "openalex", "openalex_id": wid, "outcome": "ok",
                    "is_oa": is_oa})
            f.write(json.dumps(rec) + "\n")
            if up is not None:
                f.write(json.dumps({"source": "unpaywall", "doi": doi, **up}) + "\n")

    proc = run(d, cand, log)
    check(proc.returncode == 5, f"without an override the run aborts "
                                f"(rc={proc.returncode})")

    ov = os.path.join(d, "overrides.csv")

    def write_ov(*lines):
        with open(ov, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(el.OVERRIDE_COLUMNS)
            for ln in lines:
                w.writerow(ln)

    write_ov(["W8", "True", "none", "arXiv PDF on record", "Tester", "2026-10-05"])
    proc = run(d, cand, log)
    check(proc.returncode == 0, f"with an override it completes (rc={proc.returncode})")
    epath = os.path.join(d, "eligibility.csv")
    by = {r["openalex_id"]: r for r in csv.DictReader(open(epath, encoding="utf-8"))}
    check(by["W8"]["eligible"] == "True"
          and by["W8"]["eligibility_rule"] == "manual_override",
          "the overridden row is labelled manual_override, not a rule")
    prov = json.load(open(os.path.join(d, "prov.json"), encoding="utf-8"))["eligibility"]
    check(len(prov["manual_overrides"]) == 1
          and prov["manual_overrides"][0]["reason"] == "arXiv PDF on record",
          "provenance lists the override with its stated reason")

    snap = open(epath, "rb").read()
    write_ov(["W8", "True", "none", "arXiv PDF on record", "Tester", "2026-10-05"],
             ["W2", "True", "none", "trying to rescue an Elsevier paper",
              "Tester", "2026-10-05"])
    proc = run(d, cand, log)
    check(proc.returncode != 0 and "rules decide it" in proc.stderr,
          "an override on a row the rules DO decide aborts -- it cannot "
          "replace a rule")
    check(open(epath, "rb").read() == snap, "and writes nothing")


def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="eligibility_test_")
    try:
        test_rule()
        test_end_to_end(tmpdir)
        test_all_or_nothing(tmpdir)
        test_overrides(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
