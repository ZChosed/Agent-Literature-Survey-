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
  * rows are never deleted, reordered or altered -- only the three eligibility
    columns are added
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
    after = list(csv.DictReader(open(cand, encoding="utf-8")))

    # --- rows are never deleted -------------------------------------------
    check(len(after) == len(before) == 9,
          f"no row deleted: {len(before)} in, {len(after)} out")
    check([r["openalex_id"] for r in after] == [r["openalex_id"] for r in before],
          "and none reordered")
    check(all({k: a[k] for k in COLS} == b for a, b in zip(after, before)),
          "every original column is unchanged on every row")
    check(list(after[0].keys()) == COLS + el.OUT_COLUMNS,
          f"only the three eligibility columns are added, at the end "
          f"(got {list(after[0].keys())[len(COLS):]})")
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
    check(prov["rows_deleted"] == 0, "provenance states that no row was deleted")

    # Idempotent: a second offline run changes nothing.
    snap = open(cand, "rb").read()
    run(d, cand, log)
    check(open(cand, "rb").read() == snap, "a rerun is byte-identical")
    check(before_bytes != snap, "(and the first run did write)")


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
    check(not os.path.exists(os.path.join(d, "prov.json")),
          "and no provenance is written")

    # An Unpaywall error is in the cache but is not an answer.
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "unpaywall", "doi": "10.1007/c.5",
                            "outcome": "http_error", "http_status": 503}) + "\n")
    proc = run(d, cand, log)
    check(proc.returncode == 5 and "could not be decided" in proc.stdout,
          f"a row resting on an HTTP error is undecided (rc={proc.returncode})")
    check(open(cand, "rb").read() == before, "and still nothing is written")


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
    by = {r["openalex_id"]: r for r in csv.DictReader(open(cand, encoding="utf-8"))}
    check(by["W8"]["eligible"] == "True"
          and by["W8"]["eligibility_rule"] == "manual_override",
          "the overridden row is labelled manual_override, not a rule")
    prov = json.load(open(os.path.join(d, "prov.json"), encoding="utf-8"))["eligibility"]
    check(len(prov["manual_overrides"]) == 1
          and prov["manual_overrides"][0]["reason"] == "arXiv PDF on record",
          "provenance lists the override with its stated reason")

    snap = open(cand, "rb").read()
    write_ov(["W8", "True", "none", "arXiv PDF on record", "Tester", "2026-10-05"],
             ["W2", "True", "none", "trying to rescue an Elsevier paper",
              "Tester", "2026-10-05"])
    proc = run(d, cand, log)
    check(proc.returncode != 0 and "rules decide it" in proc.stderr,
          "an override on a row the rules DO decide aborts -- it cannot "
          "replace a rule")
    check(open(cand, "rb").read() == snap, "and writes nothing")


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
