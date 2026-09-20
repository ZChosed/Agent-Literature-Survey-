#!/usr/bin/env python3
"""
test_corpus_build.py -- offline smoke test for corpus_build.py.

The cloud container this was developed in cannot reach api.openalex.org (proxy
allowlist), so this harness substitutes a fake OpenAlex and exercises the logic
that is actually easy to get wrong:

  * seed CSV loading and role/expand parsing
  * DOI-first then title-search resolution, including a deliberate near-miss
  * the min_similarity floor and the RESOLVED_LOW_CONFIDENCE band
  * backward + forward expansion and per-seed caps
  * title-level deduplication of a preprint/published pair
  * year / type / language filtering
  * seed-link scoring and the framework_core-first sort order
  * survey_baseline coverage check arithmetic
  * CSV and provenance output shape

Run:  python test_corpus_build.py
Exit 0 = all assertions passed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

import corpus_build as cb


# ---------------------------------------------------------------------------
# Fake OpenAlex
# ---------------------------------------------------------------------------

def W(wid, title, year=2022, wtype="article", refs=None, cited=10,
      doi=None, lang="en", abstract="alpha beta gamma"):
    """Build a minimal OpenAlex-shaped work record."""
    inv = {}
    for i, word in enumerate(abstract.split()):
        inv.setdefault(word, []).append(i)
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "display_name": title,
        "publication_year": year,
        "type": wtype,
        "language": lang,
        "primary_location": {"source": {"display_name": "Test Journal"},
                             "pdf_url": None},
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "cited_by_count": cited,
        "referenced_works": [f"https://openalex.org/{r}" for r in (refs or [])],
        "abstract_inverted_index": inv,
        "open_access": {"oa_url": None},
    }


# The fake universe.
SEED_10 = W("W10", "The model-based product agent: A control oriented architecture "
                   "for intelligent products in multi-agent manufacturing systems",
            2019, refs=["W901", "W902"], cited=300,
            doi="10.1016/j.conengprac.2019.03.009")
SEED_24 = W("W24", "Agent-based manufacturing - review and expert evaluation",
            2023, refs=["W901", "W903", "W904"], cited=90)
SEED_20 = W("W20", "A requirements driven digital twin framework: Specification "
                   "and opportunities", 2020, refs=["W905"], cited=400)

# Backward targets
REF_901 = W("W901", "Holonic manufacturing control architecture", 2016, cited=55)
REF_902 = W("W902", "Contract net protocol revisited for shop floor control", 2017, cited=40)
REF_903 = W("W903", "Ein deutschsprachiger Beitrag", 2018, lang="de", cited=5)   # filtered: language
REF_904 = W("W904", "Ancient agent work", 2009, cited=500)                        # filtered: year
REF_905 = W("W905", "Digital twin dataset description", 2021, wtype="dataset")    # filtered: type

# Forward targets. F1 cites both W10 and W24 -> two seed links, one framework_core.
FWD_1 = W("W801", "LLM-driven product agents that discover new resource capabilities",
          2025, cited=12)
FWD_2 = W("W802", "Reinforcement learning for multi-agent job shop scheduling", 2024, cited=30)
# A preprint duplicate of FWD_2 under a different ID -> must dedupe by title.
FWD_2_PRE = W("W803", "Reinforcement Learning for Multi-Agent Job-Shop Scheduling!",
              2024, wtype="preprint", cited=1)

ALL = {w["id"].rsplit("/", 1)[-1]: w
       for w in [SEED_10, SEED_24, SEED_20, REF_901, REF_902, REF_903, REF_904,
                 REF_905, FWD_1, FWD_2, FWD_2_PRE]}

CITES = {
    "W10": [FWD_1, FWD_2, FWD_2_PRE],
    "W24": [FWD_1],
    "W20": [],
}

TITLE_INDEX = [SEED_10, SEED_24, SEED_20, FWD_1, FWD_2]


class FakeOpenAlex:
    """Drop-in stand-in for cb.OpenAlex. Mirrors the filter grammar used."""

    def __init__(self, *_a, **_kw):
        self.n_requests = 0

    def get(self, path, params):
        self.n_requests += 1
        flt = params.get("filter", "")

        if flt.startswith("doi:"):
            want = flt.split("doi:", 1)[1]
            hits = [w for w in ALL.values()
                    if cb.clean_doi(w.get("doi")) == want.lower()]
            return {"results": hits, "meta": {}}

        if flt.startswith("openalex_id:"):
            ids = flt.split("openalex_id:", 1)[1].split("|")
            return {"results": [ALL[i] for i in ids if i in ALL], "meta": {}}

        if flt.startswith("title.search:"):
            q = flt.split("title.search:", 1)[1]
            scored = sorted(TITLE_INDEX,
                            key=lambda w: -cb.title_similarity(q, w["display_name"]))
            return {"results": scored[:25], "meta": {}}

        if flt.startswith("cites:"):
            sid = flt.split("cites:", 1)[1].split(",")[0]
            return {"results": CITES.get(sid, []), "meta": {"next_cursor": None}}

        return {"results": [], "meta": {}}

    def paged(self, path, params, cap=None):
        data = self.get(path, params)
        for i, w in enumerate(data["results"]):
            if cap is not None and i >= cap:
                return
            yield w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FAILS: list[str] = []


def check(cond, label):
    if cond:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILS.append(label)


def test_seed_loading(real_seeds: str):
    print("\n[seed CSV] loading the real 35-row seed list")
    seeds = cb.load_seeds(real_seeds)
    check(len(seeds) == 35, f"35 seed rows loaded (got {len(seeds)})")

    roles = {}
    for s in seeds:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
    check(roles.get("framework_core") == 3,
          f"3 framework_core seeds [7,8,10] (got {roles.get('framework_core')})")
    check(roles.get("survey_baseline") == 4,
          f"4 survey_baseline seeds [6,23,24,27] (got {roles.get('survey_baseline')})")

    refs = {s["ref"] for s in seeds}
    # Proposal refs [1] (ADI website) and [32] (NSF IUCRC website) are URLs, not
    # papers, so they are intentionally absent and must not be treated as a gap.
    check("1" not in refs and "32" not in refs,
          "website-only proposal refs [1] and [32] correctly excluded")
    check(refs >= {"7", "8", "10", "20", "26"},
          "key refs present (7, 8, 10, 20, 26)")
    check(all(isinstance(s["expand"], bool) for s in seeds),
          "expand column parsed to bool on every row")
    # Refs 4, 16, 26, 30, 31 are expand=no: lab prior work not yet indexed
    # (MSEC 2026 papers), NIST technical reports, and the internal ME 590
    # report [26]. Expanding from those yields nothing useful.
    no_expand = sorted(s["ref"] for s in seeds if not s["expand"])
    check(no_expand == ["16", "26", "30", "31", "4"],
          f"the 5 non-expandable seeds are the expected ones (got {no_expand})")


def test_resolution():
    print("\n[resolve] DOI-first, then title search, with a similarity floor")
    api = FakeOpenAlex()

    by_doi = cb.resolve_seed(api, {
        "ref": "10", "role": "framework_core", "expand": True, "year": "2019",
        "first_author": "Kovalenko",
        "title": "The model-based product agent",
        "doi": "10.1016/j.conengprac.2019.03.009",
    }, 0.72)
    check(by_doi["status"] == "RESOLVED_BY_DOI", "exact DOI resolves by DOI")
    check(by_doi["openalex_id"] == "W10", "DOI resolution returns the right work")
    check(by_doi["n_references"] == 2, "reference count carried through")

    by_title = cb.resolve_seed(api, {
        "ref": "24", "role": "survey_baseline", "expand": True, "year": "2023",
        "first_author": "Pulikottil",
        "title": "Agent-based manufacturing - review and expert evaluation",
        "doi": "",
    }, 0.72)
    check(by_title["status"] == "RESOLVED_BY_TITLE",
          f"near-exact title resolves at high confidence ({by_title['status']})")
    check(by_title["openalex_id"] == "W24", "title resolution returns the right work")

    junk = cb.resolve_seed(api, {
        "ref": "99", "role": "agent_related", "expand": True, "year": "2020",
        "first_author": "Nobody",
        "title": "Zzzz qqqq xxxx completely unrelated nonexistent paper", "doi": "",
    }, 0.72)
    check(junk["openalex_id"] == "",
          f"unmatched title is NOT silently accepted ({junk['status']})")
    check(junk["status"] == "REJECTED_LOW_SIMILARITY",
          "unmatched title is flagged REJECTED_LOW_SIMILARITY for hand review")


def test_expansion_and_assembly():
    print("\n[expand + assemble] dedupe, filters, scoring, sort order")
    api = FakeOpenAlex()
    resolved = [
        cb.resolve_seed(api, {"ref": "10", "role": "framework_core", "expand": True,
                              "year": "2019", "first_author": "Kovalenko",
                              "title": "The model-based product agent",
                              "doi": "10.1016/j.conengprac.2019.03.009"}, 0.72),
        cb.resolve_seed(api, {"ref": "24", "role": "survey_baseline", "expand": True,
                              "year": "2023", "first_author": "Pulikottil",
                              "title": "Agent-based manufacturing - review and expert evaluation",
                              "doi": ""}, 0.72),
    ]
    check(all(r["openalex_id"] for r in resolved), "both test seeds resolved")

    bw, bl, bd = cb.expand_backward(api, resolved, 250, verbose=False)
    check("W901" in bw, "backward expansion retrieved a cited work")
    check(bl["W901"] == {"10", "24"},
          f"shared reference linked to BOTH seeds (got {bl['W901']})")

    fw, fl, fd = cb.expand_forward(api, resolved, None, 2015, verbose=False)
    check("W801" in fw, "forward expansion retrieved a citing work")
    check(fl["W801"] == {"10", "24"}, "citing work linked to both seeds")

    all_works = {**bw, **fw}
    links = {k: set(v) for k, v in bl.items()}
    for k, v in fl.items():
        links.setdefault(k, set()).update(v)
    directions = {k: set(v) for k, v in bd.items()}
    for k, v in fd.items():
        directions.setdefault(k, set()).update(v)

    seed_ids = {r["openalex_id"] for r in resolved}
    seed_roles = {r["ref"]: r["role"] for r in resolved}
    cands = cb.build_candidates(all_works, links, directions, seed_roles, seed_ids, 2015)
    ids = [c["openalex_id"] for c in cands]

    check(not (seed_ids & set(ids)), "seeds excluded from the candidate set")
    check("W904" not in ids, "pre-2015 work filtered out by min_year")
    check("W903" not in ids, "non-English work filtered out")
    check("W905" not in ids, "dataset-type work filtered out")
    check(("W802" in ids) != ("W803" in ids),
          "preprint/published title duplicate collapsed to exactly one row")

    top = cands[0]
    # W801 (2025, 12 cites -> ~12/yr) and W901 (2016, 55 cites -> ~5.5/yr) both
    # carry 2 seed links and a framework_core link, so the tiebreak decides.
    # Under raw citation count W901 won; under velocity the recent LLM-agent
    # paper wins, which is the behavior this survey needs (proposal S2 calls
    # these "fast-moving areas").
    check(top["openalex_id"] == "W801",
          f"velocity tiebreak puts the recent paper first (got {top['openalex_id']})")
    check(top["linked_to_framework_core"] == 1, "framework_core flag set on it")
    check(top["n_seed_links"] == 2, "seed-link count correct")
    check(top["seed_roles"] == "framework_core|survey_baseline",
          f"roles recorded (got {top['seed_roles']})")
    check(top["abstract"] != "", "abstract reconstructed from inverted index")
    check(set(top) >= {"screen_decision", "screen_reason", "screen_model", "human_label"},
          "empty screening columns present for the next stage")

    w901 = next(c for c in cands if c["openalex_id"] == "W901")
    check(w901["cited_by_count"] > top["cited_by_count"],
          "the demoted paper does have more raw citations (tiebreak really is velocity)")
    check(w901["citations_per_year"] < top["citations_per_year"],
          "and a lower citation velocity")

    order_ok = all(
        (-cands[i]["linked_to_framework_core"], -cands[i]["n_seed_links"],
         -cands[i]["citations_per_year"]) <=
        (-cands[i + 1]["linked_to_framework_core"], -cands[i + 1]["n_seed_links"],
         -cands[i + 1]["citations_per_year"])
        for i in range(len(cands) - 1)
    )
    check(order_ok, "full sort order holds: framework_core, seed breadth, velocity")
    return resolved, cands, seed_ids, api


def test_coverage(resolved, cands, seed_ids, api, tmpdir):
    print("\n[coverage] survey_baseline reference-list coverage arithmetic")
    corpus_ids = {c["openalex_id"] for c in cands} | seed_ids
    out = os.path.join(tmpdir, "coverage_report.csv")
    rep = cb.coverage_check(api, resolved, corpus_ids, out, verbose=False)
    check(len(rep) == 1, f"only the survey_baseline seed reported (got {len(rep)})")
    row = rep[0]
    # W24 cites W901, W903, W904. Only W901 survives the filters.
    check(row["n_references"] == 3, f"3 references counted (got {row['n_references']})")
    check(row["n_in_corpus"] == 1, f"1 of them in corpus (got {row['n_in_corpus']})")
    check(abs(row["pct_covered"] - 33.3) < 0.1,
          f"percentage computed correctly (got {row['pct_covered']})")
    check(os.path.exists(out), "coverage_report.csv written")


def test_csv_roundtrip(cands, tmpdir):
    print("\n[output] CSV round-trips with embedded commas and quotes")
    path = os.path.join(tmpdir, "candidates.csv")
    rows = list(cands)
    rows[0] = dict(rows[0])
    rows[0]["title"] = 'A title with, a comma and "quotes" in it'
    cb.write_csv(path, rows, list(rows[0].keys()))
    with open(path, newline="", encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    check(len(back) == len(rows), f"row count preserved ({len(back)}/{len(rows)})")
    check(back[0]["title"] == 'A title with, a comma and "quotes" in it',
          "comma/quote title survives the round trip")
    check(back[0]["n_seed_links"] == str(rows[0]["n_seed_links"]),
          "numeric column preserved")


def main() -> int:
    seeds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.csv")
    if not os.path.exists(seeds_path):
        print(f"seeds.csv not found at {seeds_path}")
        return 1

    tmpdir = tempfile.mkdtemp(prefix="corpus_test_")
    try:
        test_seed_loading(seeds_path)
        test_resolution()
        resolved, cands, seed_ids, api = test_expansion_and_assembly()
        test_coverage(resolved, cands, seed_ids, api, tmpdir)
        test_csv_roundtrip(cands, tmpdir)
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
