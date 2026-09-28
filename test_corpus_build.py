#!/usr/bin/env python3
"""
test_corpus_build.py -- offline smoke test for corpus_build.py.

The cloud container this was developed in cannot reach api.openalex.org (proxy
allowlist), so this harness substitutes a fake OpenAlex and exercises the logic
that is actually easy to get wrong:

  * seed CSV loading, role parsing, and per-direction expand flags
    (including the pre-1.2.0 single-`expand` fallback)
  * DOI-first then title-search resolution, including a deliberate near-miss
  * the min_similarity floor and the RESOLVED_LOW_CONFIDENCE band
  * backward + forward expansion, per-seed direction gating, and the
    per-seed caps -- including that a cap which bites is REPORTED
  * the target domain T, p(s) measurement, and its two guards
  * the pre-registered backward gate: that BOTH conditions are necessary,
    that it is off by default, and that every decision records its inputs
  * precision-weighted scoring, and that weighting demotes without removing
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

# Topic ids used by the fake universe. T210 / T211 stand for the agent-
# manufacturing fingerprint carried by the framework seeds; T990 is off-target.
AGENT_TOPICS = [("T210", 0.85), ("T211", 0.40)]
OFFTOPIC_TOPICS = [("T990", 0.90)]
WEAK_TOPICS = [("T210", 0.12)]      # below the 0.3 score floor -> must NOT count


def W(wid, title, year=2022, wtype="article", refs=None, cited=10,
      doi=None, lang="en", abstract="alpha beta gamma", topics=None):
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
        "topics": [{"id": f"https://openalex.org/{t}", "score": s}
                   for t, s in (topics if topics is not None else AGENT_TOPICS)],
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
# Off-target references, reachable only from the dilute seed W35 below.
REF_906 = W("W906", "Transformer scaling laws for vision", 2023, cited=80,
            topics=OFFTOPIC_TOPICS)
REF_907 = W("W907", "Self-supervised pretraining on web text", 2022, cited=70,
            topics=OFFTOPIC_TOPICS)
REF_908 = W("W908", "A weakly related manufacturing note", 2021, cited=3,
            topics=WEAK_TOPICS)
REF_903 = W("W903", "Ein deutschsprachiger Beitrag", 2018, lang="de", cited=5)   # filtered: language
REF_904 = W("W904", "Ancient agent work", 2009, cited=500)                        # filtered: year
REF_905 = W("W905", "Digital twin dataset description", 2021, wtype="dataset")    # filtered: type

# Forward targets. F1 cites both W10 and W24 -> two seed links, one framework_core.
# Their `referenced_works` must name the seeds they cite: that is what forward
# uniqueness reads, so a fixture without it would silently make every sampled
# citing work look unique.
FWD_1 = W("W801", "LLM-driven product agents that discover new resource capabilities",
          2025, cited=12, refs=["W10", "W24"])
FWD_2 = W("W802", "Reinforcement learning for multi-agent job shop scheduling", 2024,
          cited=30, refs=["W10"])
# A preprint duplicate of FWD_2 under a different ID -> must dedupe by title.
FWD_2_PRE = W("W803", "Reinforcement Learning for Multi-Agent Job-Shop Scheduling!",
              2024, wtype="preprint", cited=1, refs=["W10"])

# A dilute seed: three of its four unique references sit outside the target
# domain. This is the shape the old asserted [35] rule claimed to detect.
SEED_35 = W("W35", "Large scale foundation models for intelligent manufacturing "
                   "applications: a survey",
            2025, refs=["W901", "W906", "W907", "W908"], cited=33,
            topics=OFFTOPIC_TOPICS)

ALL = {w["id"].rsplit("/", 1)[-1]: w
       for w in [SEED_10, SEED_24, SEED_20, SEED_35, REF_901, REF_902, REF_903,
                 REF_904, REF_905, REF_906, REF_907, REF_908,
                 FWD_1, FWD_2, FWD_2_PRE]}

CITES = {
    "W10": [FWD_1, FWD_2, FWD_2_PRE],
    "W24": [FWD_1],
    "W20": [],
}

TITLE_INDEX = [SEED_10, SEED_24, SEED_20, SEED_35, FWD_1, FWD_2]


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

    def count(self, path, params):
        return len(self.get(path, params)["results"])

    def sample(self, path, params, n, seed):
        """Deterministic stand-in for OpenAlex `sample`+`seed`: rotate, then take n."""
        rows = list(self.get(path, params)["results"])
        if not rows:
            return []
        return (rows[seed % len(rows):] + rows[:seed % len(rows)])[:n]

    def paged(self, path, params, cap=None, meta_out=None):
        data = self.get(path, params)
        results = data["results"]
        if meta_out is not None:
            # Real OpenAlex reports the unpaginated total here; that is what lets
            # a capped pull be reported as "kept N of M".
            meta_out.update({**data.get("meta", {}), "count": len(results)})
        for i, w in enumerate(results):
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
    check(roles.get("framework_core") == 5,
          f"5 framework_core seeds [2,7,8,9,10] (got {roles.get('framework_core')})")
    check(roles.get("survey_baseline") == 4,
          f"4 survey_baseline seeds [6,23,24,27] (got {roles.get('survey_baseline')})")

    refs = {s["ref"] for s in seeds}
    # Proposal refs [1] (ADI website) and [32] (NSF IUCRC website) are URLs, not
    # papers, so they are intentionally absent and must not be treated as a gap.
    check("1" not in refs and "32" not in refs,
          "website-only proposal refs [1] and [32] correctly excluded")
    check(refs >= {"7", "8", "10", "20", "26"},
          "key refs present (7, 8, 10, 20, 26)")
    check(all(isinstance(s["expand_backward"], bool)
              and isinstance(s["expand_forward"], bool) for s in seeds),
          "both expand columns parsed to bool on every row")

    by_ref = {s["ref"]: s for s in seeds}

    # Two kinds of flag live in seeds.csv at v1.5.0, and they must not be
    # confused with each other.
    #
    # DATA AVAILABILITY: the four seeds OpenAlex has no record of cannot expand
    # in either direction. Nothing topical is being claimed about them.
    #
    # SCOPE: the six dt_survey seeds expand backward only. This is a declared
    # judgment from proposal S2.1 -- the DT half of the review is a ~15-paper
    # comparison subset, so the field citing a generic DT survey is out of
    # scope -- and NOT a claim about what those citing works contain. v1.2.0
    # made the content claim; v1.4.0 measured it and it did not hold. The rule
    # is back, the assertion is not.
    #
    # What is gone for good is the flag set from a recollection of what a
    # bibliography contains: v1.2.0's `standards` and [35] Zhang rules. Every
    # seed OpenAlex can resolve now expands BACKWARD, without exception.
    dead_refs = ["4", "26", "30", "31"]
    dead = sorted((r for r, s in by_ref.items()
                   if not s["expand_backward"] and not s["expand_forward"]),
                  key=lambda x: (len(x), x))
    check(dead == dead_refs,
          f"only the unresolvable seeds are fully off (got {dead})")

    live = [s for s in seeds if s["ref"] not in set(dead_refs)]
    check(len(live) == 31, f"31 resolvable seeds (got {len(live)})")
    check(all(s["expand_backward"] for s in live),
          "every resolvable seed expands BACKWARD, no exceptions -- retrieval "
          "is not gated for precision")

    dt_refs = sorted((s["ref"] for s in seeds if s["role"] == "dt_survey"),
                     key=lambda x: (len(x), x))
    check(dt_refs == ["14", "15", "19", "22", "34", "36"],
          f"the 6 dt_survey seeds are [14,15,19,22,34,36] (got {dt_refs})")
    check(all(by_ref[r]["expand_backward"] and not by_ref[r]["expand_forward"]
              for r in dt_refs),
          "dt_survey expands backward only -- S2.1 scope, not measured precision")

    fwd_off = sorted((s["ref"] for s in live if not s["expand_forward"]),
                     key=lambda x: (len(x), x))
    check(fwd_off == dt_refs,
          f"dt_survey is the ONLY role with forward off (got {fwd_off})")

    check(by_ref["35"]["expand_backward"] and by_ref["35"]["expand_forward"],
          "[35] Zhang expands both ways; the measured rule decides its weight")

    fc = sorted((s["ref"] for s in seeds if s["role"] == "framework_core"),
                key=lambda x: (len(x), x))
    check(fc == ["2", "7", "8", "9", "10"],
          f"framework_core is [2,7,8,9,10] -- [2] Bi and [9] Kovalenko retagged "
          f"at v1.5.0 (got {fc})")


def test_legacy_seed_column(tmpdir: str):
    print("\n[seed CSV] pre-1.2.0 single `expand` column still loads")
    path = os.path.join(tmpdir, "legacy_seeds.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("ref,role,expand,year,first_author,title,doi\n")
        f.write("7,framework_core,yes,2022,Kovalenko,Cooperative product agents,\n")
        f.write("4,lab_prior_work,no,2026,Chen,A coordinated planning framework,\n")
    seeds = cb.load_seeds(path)
    check(len(seeds) == 2, f"legacy file loaded (got {len(seeds)} rows)")
    check(seeds[0]["expand_backward"] and seeds[0]["expand_forward"],
          "legacy expand=yes applies to BOTH directions")
    check(not seeds[1]["expand_backward"] and not seeds[1]["expand_forward"],
          "legacy expand=no applies to BOTH directions")


def test_resolution():
    print("\n[resolve] DOI-first, then title search, with a similarity floor")
    api = FakeOpenAlex()

    by_doi = cb.resolve_seed(api, {
        "ref": "10", "role": "framework_core", "expand_backward": True,
        "expand_forward": True, "year": "2019",
        "first_author": "Kovalenko",
        "title": "The model-based product agent",
        "doi": "10.1016/j.conengprac.2019.03.009",
    }, 0.72)
    check(by_doi["status"] == "RESOLVED_BY_DOI", "exact DOI resolves by DOI")
    check(by_doi["openalex_id"] == "W10", "DOI resolution returns the right work")
    check(by_doi["n_references"] == 2, "reference count carried through")

    by_title = cb.resolve_seed(api, {
        "ref": "24", "role": "survey_baseline", "expand_backward": True,
        "expand_forward": True, "year": "2023",
        "first_author": "Pulikottil",
        "title": "Agent-based manufacturing - review and expert evaluation",
        "doi": "",
    }, 0.72)
    check(by_title["status"] == "RESOLVED_BY_TITLE",
          f"near-exact title resolves at high confidence ({by_title['status']})")
    check(by_title["openalex_id"] == "W24", "title resolution returns the right work")

    junk = cb.resolve_seed(api, {
        "ref": "99", "role": "agent_related", "expand_backward": True,
        "expand_forward": True, "year": "2020",
        "first_author": "Nobody",
        "title": "Zzzz qqqq xxxx completely unrelated nonexistent paper", "doi": "",
    }, 0.72)
    check(junk["openalex_id"] == "",
          f"unmatched title is NOT silently accepted ({junk['status']})")
    check(junk["status"] == "REJECTED_LOW_SIMILARITY",
          "unmatched title is flagged REJECTED_LOW_SIMILARITY for hand review")


SEED_STUB = {"first_author": "X", "year": "2020", "doi": ""}


def measure_bwd(resolved, seed_refs, ref_works, T, policy):
    """Build precision rows and fill in the backward half."""
    rows = [cb._precision_row(r) for r in resolved]
    cb.measure_backward_precision(rows, seed_refs, ref_works, T, policy, verbose=False)
    return rows


def measure_fwd(resolved, samples, seed_ids, T, policy, rows=None):
    """Build precision rows and fill in the forward half."""
    rows = rows if rows is not None else [cb._precision_row(r) for r in resolved]
    cb.measure_forward_precision(rows, samples, seed_ids, T, policy, verbose=False)
    return rows


def _resolve(api, ref, role, title, doi="", **kw):
    seed = {"ref": ref, "role": role, "title": title, "doi": doi,
            "expand_backward": True, "expand_forward": True, **SEED_STUB, **kw}
    return cb.resolve_seed(api, seed, 0.72)


def test_precision_measurement():
    print("\n[precision] T from the framework seeds, then p(s) per seed")
    api = FakeOpenAlex()
    policy = cb.load_policy(None)
    policy["target_domain"]["seed_refs"] = ["10"]

    resolved = [
        _resolve(api, "10", "framework_core", SEED_10["display_name"],
                 doi="10.1016/j.conengprac.2019.03.009"),
        _resolve(api, "24", "survey_baseline",
                 "Agent-based manufacturing - review and expert evaluation"),
        _resolve(api, "35", "foundation_models",
                 "Large scale foundation models for intelligent manufacturing "
                 "applications: a survey"),
    ]
    check(all(r["openalex_id"] for r in resolved), "three test seeds resolved")

    T, missing = cb.target_domain(resolved, policy, verbose=False)
    check(T == {"T210", "T211"}, f"T is the framework seed's topic set (got {sorted(T)})")
    check(missing == [], "no target seed failed to contribute topics")
    check("T990" not in T, "an off-target topic is not in T")

    # The 0.3 score floor must actually bite: W908 carries T210 at 0.12.
    check(cb.topic_ids(ALL["W908"], 0.3) == set(),
          "a topic below the score floor is discarded")
    check(cb.topic_ids(ALL["W908"], 0.1) == {"T210"},
          "...and kept when the floor is lowered, so the floor is what filtered it")

    seed_refs = cb.collect_backward_refs(resolved)
    ref_works = cb.fetch_reference_works(api, seed_refs, verbose=False)
    policy["precision"]["min_sample"] = 2
    rows = measure_bwd(resolved, seed_refs, ref_works, T, policy)
    by = {r["ref"]: r for r in rows}

    # W10 cites W901 + W902, both on target.
    check(by["10"]["bwd_precision_all"] == 1.0,
          f"on-target seed measures p=1.0 (got {by['10']['bwd_precision_all']})")
    # W35 cites W901 (on target, shared) + W906/W907/W908 (off target, unique).
    check(by["35"]["bwd_precision_all"] == 0.25,
          f"dilute seed measures p=0.25 (got {by['35']['bwd_precision_all']})")
    check(by["35"]["bwd_precision_unique"] == 0.0,
          f"...and p_unique=0.0 once the corroborated reference is excluded "
          f"(got {by['35']['bwd_precision_unique']})")
    check(by["35"]["n_unique_references"] == 3,
          f"U(35) counts only references no other seed reached (got "
          f"{by['35']['n_unique_references']})")
    check(by["10"]["n_unique_references"] == 1,
          f"W901 is shared, so U(10) excludes it (got {by['10']['n_unique_references']})")
    check(all(r["bwd_basis"] == "measured" for r in rows),
          "all three seeds cleared the sample floor, so all are 'measured'")


def test_precision_guards():
    print("\n[precision] guards: under-sampled seeds and empty reference lists")
    api = FakeOpenAlex()
    policy = cb.load_policy(None)
    policy["target_domain"]["seed_refs"] = ["10"]
    policy["precision"]["min_sample"] = 3      # W10 has 2 refs, W35 has 4

    resolved = [
        _resolve(api, "10", "framework_core", SEED_10["display_name"],
                 doi="10.1016/j.conengprac.2019.03.009"),
        _resolve(api, "35", "foundation_models",
                 "Large scale foundation models for intelligent manufacturing "
                 "applications: a survey"),
        _resolve(api, "20", "dt_core", SEED_20["display_name"]),
    ]
    T, _ = cb.target_domain(resolved, policy, verbose=False)
    seed_refs = cb.collect_backward_refs(resolved)
    ref_works = cb.fetch_reference_works(api, seed_refs, verbose=False)
    rows = measure_bwd(resolved, seed_refs, ref_works, T, policy)
    by = {r["ref"]: r for r in rows}

    check(by["10"]["bwd_precision_all"] == 1.0,
          "the raw estimate is still reported for an under-sampled seed")
    check(by["10"]["bwd_basis"].startswith("under_sampled(n=2<3)"),
          f"...but flagged under_sampled (got {by['10']['bwd_basis']})")
    check(by["10"]["bwd_weight"] != by["10"]["bwd_precision_all"] or
          "default" in by["10"]["bwd_basis"],
          "an under-sampled seed's WEIGHT comes from a default, not its own estimate")
    check("global_default" in by["10"]["bwd_basis"],
          f"no other framework_core seed clears the bar, so the global mean is used "
          f"(got {by['10']['bwd_basis']})")

    # W20 cites only W905; that is a dataset, but it still resolves, so W20 is
    # under-sampled rather than reference-less. W905's own topics decide p.
    check(by["20"]["bwd_basis"].startswith("under_sampled"),
          f"a one-reference seed is under-sampled (got {by['20']['bwd_basis']})")

    # A seed OpenAlex has no reference list for is a data gap, coded distinctly.
    empty = _resolve(api, "21", "dt_core", SEED_20["display_name"])
    empty["_work"] = dict(empty["_work"], referenced_works=[])
    empty["n_references"] = 0
    rows2 = measure_bwd([empty], {"21": []}, {}, T, policy)
    check(rows2[0]["bwd_basis"].startswith("no_references"),
          f"empty reference list is coded no_references, not low precision "
          f"(got {rows2[0]['bwd_basis']})")
    check(rows2[0]["bwd_precision_all"] is None,
          "no precision value is invented for a seed with no references")
    # Degenerate case: nothing measurable anywhere. Falling back to a global mean
    # of 0.0 would zero every weight and destroy the ranking -- "could not
    # measure" must not silently become "counts for nothing".
    check(rows2[0]["bwd_weight"] == 1.0,
          f"with no measurable seed at all, weighting goes UNIFORM at 1.0 "
          f"(got {rows2[0]['bwd_weight']})")
    check("uniform:no_measurable_seeds" in rows2[0]["bwd_basis"],
          f"...and says so in precision_basis (got {rows2[0]['bwd_basis']})")


def test_policy_file(tmpdir):
    print("\n[policy] thresholds come from a committed file, not from argparse")
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "expansion_policy.json")
    check(os.path.exists(real), "expansion_policy.json is committed alongside the script")
    pol = cb.load_policy(real)
    check(pol["target_domain"]["seed_refs"] == ["7", "8", "9", "10"],
          f"T is defined from [7,8,9,10] (got {pol['target_domain']['seed_refs']})")
    check(pol["precision"]["min_sample"] == 30,
          f"min_sample is 30 (got {pol['precision']['min_sample']})")
    check(all("basis" in pol[k] for k in
              ("target_domain", "precision", "scoring", "forward_precision")),
          "every policy section carries a written basis for its value")

    # v1.6.0: the policy file governs RANKING only. Retrieval is decided
    # exclusively by the expand_backward/expand_forward columns of seeds.csv,
    # so nothing gate-shaped may survive anywhere -- not in the committed file,
    # not in the built-in defaults, not as a function, not as a CLI flag.
    check("backward_gate" not in pol and "forward_gate" not in pol,
          f"no gate section in the committed policy (got {sorted(pol)})")
    check("min_discrimination_ratio" not in pol["target_domain"],
          "no discrimination threshold in the committed policy")
    check("backward_gate" not in cb.DEFAULT_POLICY
          and "forward_gate" not in cb.DEFAULT_POLICY,
          "no gate section in the built-in defaults either")
    check(not hasattr(cb, "apply_gate"),
          "apply_gate() is gone from the module")
    check(not hasattr(cb, "domain_discrimination"),
          "domain_discrimination() is gone from the module")

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "corpus_build.py"), encoding="utf-8").read()
    for flag in ("--no-forward-gate", "--backward-gate"):
        check(f'"{flag}"' not in src, f"the {flag} CLI flag is gone")
    check(cb.SCRIPT_VERSION == "1.6.0",
          f"script is v1.6.0 (got {cb.SCRIPT_VERSION})")

    partial = os.path.join(tmpdir, "partial_policy.json")
    with open(partial, "w", encoding="utf-8") as f:
        json.dump({"precision": {"min_sample": 5}}, f)
    merged = cb.load_policy(partial)
    check(merged["precision"]["min_sample"] == 5, "a partial policy file overrides")
    check(merged["target_domain"]["topic_score_min"] == 0.3,
          "...and unspecified sections keep their defaults rather than vanishing")

    try:
        cb.load_policy(os.path.join(tmpdir, "nope.json"))
        check(False, "a missing policy file is refused rather than silently defaulted")
    except SystemExit:
        check(True, "a missing policy file is refused rather than silently defaulted")


def test_forward_precision():
    print("\n[q(s)] forward precision from a deterministic sample")
    api = FakeOpenAlex()
    policy = cb.load_policy(None)
    policy["target_domain"]["seed_refs"] = ["10"]
    policy["forward_precision"]["min_sample"] = 2

    resolved = [
        _resolve(api, "10", "framework_core", SEED_10["display_name"],
                 doi="10.1016/j.conengprac.2019.03.009"),
        _resolve(api, "24", "survey_baseline",
                 "Agent-based manufacturing - review and expert evaluation"),
    ]
    T, _ = cb.target_domain(resolved, policy, verbose=False)
    seed_ids = {r["ref"]: r["openalex_id"] for r in resolved}

    samples = cb.sample_forward_citations(api, resolved, 2015, policy, verbose=False)
    check(set(samples) == {"10", "24"}, f"sampled both forward seeds (got {set(samples)})")
    check(samples["10"]["total"] == 3,
          f"the window TOTAL is counted separately from the sample "
          f"(got {samples['10']['total']})")

    # The sample must be reproducible: same seed, same works, every run.
    again = cb.sample_forward_citations(api, resolved, 2015, policy, verbose=False)
    check([w["id"] for w in samples["10"]["works"]] ==
          [w["id"] for w in again["10"]["works"]],
          "the same sample seed returns the same works -- the estimate is reproducible")

    rows = measure_fwd(resolved, samples, seed_ids, T, policy)
    by = {r["ref"]: r for r in rows}
    check(by["10"]["fwd_precision_all"] == 1.0,
          f"citing works all carry a topic in T (got {by['10']['fwd_precision_all']})")
    check(by["10"]["n_sampled"] == 3,
          f"sample size recorded (got {by['10']['n_sampled']})")

    # Uniqueness is EXACT, not estimated: a sampled citing work's own
    # referenced_works says which other seeds it cites. W801 cites both W10 and
    # W24, so it is not unique to either; W802/W803 cite only W10.
    check(by["10"]["n_unique_sampled"] == 2,
          f"W801 cites both seeds so is excluded from U(10) (got "
          f"{by['10']['n_unique_sampled']})")
    check(by["24"]["n_unique_sampled"] == 0,
          f"W24's only citing work is shared, so U(24)=0 (got "
          f"{by['24']['n_unique_sampled']})")
    check(by["24"]["fwd_precision_unique"] is None,
          "no q_unique is invented when nothing is unique")
    check(by["10"]["est_unique_citations"] == 2,
          f"est_unique scales window total by the sampled unique fraction "
          f"(3 * 2/3 = 2, got {by['10']['est_unique_citations']})")

    # A seed nobody cites is a data gap, coded like the backward equivalent.
    solo = _resolve(api, "20", "dt_core", SEED_20["display_name"])
    rows2 = measure_fwd([solo], {"20": {"total": 0, "works": []}},
                        {"20": solo["openalex_id"]}, T, policy)
    check(rows2[0]["fwd_basis"].startswith("no_citations"),
          f"an uncited seed is coded no_citations (got {rows2[0]['fwd_basis']})")
    check(rows2[0]["fwd_precision_all"] is None,
          "no forward precision is invented for an uncited seed")


def test_directional_weighting():
    print("\n[weight] each edge is weighted by the direction it was followed")
    works = {"C1": W("C1", "Reached backward from a good seed, forward from a bad one",
                     2023, cited=5),
             "C2": W("C2", "Reached forward only, from the bad seed", 2023, cited=5)}
    links = {"C1": {"10", "99"}, "C2": {"99"}}
    dirs = {"C1": {"10": {"backward"}, "99": {"forward"}},
            "C2": {"99": {"forward"}}}
    roles = {"10": "framework_core", "99": "dt_survey"}
    weights = {"10": {"backward": 0.80, "forward": 0.10},
               "99": {"backward": 0.60, "forward": 0.05}}
    cands = cb.build_candidates(works, links, dirs, roles, set(), 2015, weights)
    by = {c["openalex_id"]: c for c in cands}
    check(abs(by["C1"]["seed_weight"] - 0.85) < 1e-9,
          f"backward edge takes p(s)=0.80, forward edge takes q(s)=0.05 "
          f"(got {by['C1']['seed_weight']})")
    check(by["C1"]["seed_link_directions"] == "backward|forward",
          f"the direction of each edge is recorded (got {by['C1']['seed_link_directions']})")
    check(abs(by["C2"]["seed_weight"] - 0.05) < 1e-9,
          f"a forward-only link takes the forward weight, not the backward one "
          f"(got {by['C2']['seed_weight']})")

    # One seed reaching a candidate BOTH ways counts once, at its higher weight.
    both = cb.build_candidates(
        {"C3": W("C3", "Reached both ways from one seed", 2023, cited=5)},
        {"C3": {"10"}}, {"C3": {"10": {"backward", "forward"}}},
        roles, set(), 2015, weights)
    check(abs(both[0]["seed_weight"] - 0.80) < 1e-9,
          f"a seed linking both ways contributes max(p,q)=0.80 once, not 0.90 "
          f"(got {both[0]['seed_weight']})")
    check(both[0]["n_seed_links"] == 1,
          "...and still counts as a single seed link")


def test_expansion_and_assembly():
    print("\n[expand + assemble] dedupe, filters, scoring, sort order")
    api = FakeOpenAlex()
    resolved = [
        cb.resolve_seed(api, {"ref": "10", "role": "framework_core",
                              "expand_backward": True, "expand_forward": True,
                              "year": "2019", "first_author": "Kovalenko",
                              "title": "The model-based product agent",
                              "doi": "10.1016/j.conengprac.2019.03.009"}, 0.72),
        cb.resolve_seed(api, {"ref": "24", "role": "survey_baseline",
                              "expand_backward": True, "expand_forward": True,
                              "year": "2023", "first_author": "Pulikottil",
                              "title": "Agent-based manufacturing - review and expert evaluation",
                              "doi": ""}, 0.72),
    ]
    check(all(r["openalex_id"] for r in resolved), "both test seeds resolved")

    seed_refs = cb.collect_backward_refs(resolved)
    ref_works = cb.fetch_reference_works(api, seed_refs, verbose=False)
    bw, bl, bd, btrunc = cb.expand_backward(resolved, seed_refs, ref_works,
                                            600, verbose=False)
    check("W901" in bw, "backward expansion retrieved a cited work")
    check(bl["W901"] == {"10", "24"},
          f"shared reference linked to BOTH seeds (got {bl['W901']})")
    check(btrunc == [], "no truncation reported when the cap does not bite")

    fw, fl, fd, ftrunc = cb.expand_forward(api, resolved, None, 2015, verbose=False)
    check("W801" in fw, "forward expansion retrieved a citing work")
    check(fl["W801"] == {"10", "24"}, "citing work linked to both seeds")
    check(ftrunc == [], "uncapped forward pull reports no truncation")

    # Direction is per seed: a backward-only seed must contribute no forward
    # edges, and vice versa. This is the whole point of the v1.2.0 split.
    back_only = [dict(resolved[0], expand_forward=False),
                 dict(resolved[1], expand_forward=False)]
    fw2, fl2, _, _ = cb.expand_forward(api, back_only, None, 2015, verbose=False)
    check(fw2 == {} and fl2 == {},
          f"expand_forward=no suppresses ALL forward edges (got {len(fw2)} works)")

    fwd_only = [dict(resolved[0], expand_backward=False),
                dict(resolved[1], expand_backward=False)]
    check(cb.collect_backward_refs(fwd_only) == {},
          "expand_backward=no suppresses ALL backward edges")

    # A cap that bites must be reported, not silently applied: truncation
    # substitutes OpenAlex pagination order for an inclusion criterion.
    one = {"10": seed_refs["10"]}
    _, bl3, _, btrunc3 = cb.expand_backward(resolved, one, ref_works, 1,
                                            verbose=False)
    check(len(bl3) == 1, f"backward cap of 1 kept exactly 1 reference (got {len(bl3)})")
    check(len(btrunc3) == 1, f"the biting backward cap produced a record (got {btrunc3})")
    if btrunc3:
        t = btrunc3[0]
        check(t["ref"] == "10" and t["direction"] == "backward",
              "truncation record identifies the seed and direction")
        check(t["available"] == 2 and t["retrieved"] == 1 and t["dropped"] == 1,
              f"truncation arithmetic: kept 1 of 2, dropped 1 (got {t})")

    _, _, _, ftrunc3 = cb.expand_forward(api, resolved[:1], 1, 2015, verbose=False)
    check(len(ftrunc3) == 1 and ftrunc3[0]["dropped"] >= 1,
          f"a biting forward cap is reported too (got {ftrunc3})")

    all_works = {**bw, **fw}
    links = {k: set(v) for k, v in bl.items()}
    for k, v in fl.items():
        links.setdefault(k, set()).update(v)
    # Per-EDGE directions: candidate -> seed ref -> {directions}.
    directions: dict = {}
    for src_map in (bd, fd):
        for wid, per_ref in src_map.items():
            for ref, dirs in per_ref.items():
                directions.setdefault(wid, {}).setdefault(ref, set()).update(dirs)

    seed_ids = {r["openalex_id"] for r in resolved}
    seed_roles = {r["ref"]: r["role"] for r in resolved}
    cands = cb.build_candidates(all_works, links, directions, seed_roles, seed_ids, 2015)
    ids = [c["openalex_id"] for c in cands]

    # Unweighted call must behave exactly as before: every seed counts 1.0.
    top_unweighted = cands[0]
    check(top_unweighted["seed_weight"] == float(top_unweighted["n_seed_links"]),
          "with no weights supplied, seed_weight degrades to the raw link count")

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

    # Weighting changes the ranking term, and low weight demotes rather than removes.
    weighted = cb.build_candidates(
        all_works, links, directions, seed_roles, seed_ids, 2015,
        seed_weights={"10": {"backward": 0.70, "forward": 0.70},
                      "24": {"backward": 0.05, "forward": 0.05}})
    wby = {c["openalex_id"]: c for c in weighted}
    check(len(weighted) == len(cands),
          f"weighting removes no rows ({len(weighted)} vs {len(cands)})")
    check(abs(wby["W801"]["seed_weight"] - 0.75) < 1e-9,
          f"a candidate reached from both seeds sums their weights "
          f"(got {wby['W801']['seed_weight']})")
    check(wby["W801"]["seed_weight"] > wby["W902"]["seed_weight"],
          f"reached-from-both ({wby['W801']['seed_weight']}) outranks "
          f"reached-from-one ({wby['W902']['seed_weight']}) under weighting")
    check(wby["W902"]["seed_weight"] == 0.7 and wby["W902"]["n_seed_links"] == 1,
          "a candidate reached only from the high-precision seed carries its weight")
    check(wby["W801"]["seed_link_weights"] == "0.700|0.050",
          f"per-link weights recorded for audit (got {wby['W801']['seed_link_weights']})")
    check(wby["W901"]["n_seed_links"] == 2,
          "the unweighted link count stays in the row as the audit trail")
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
        test_legacy_seed_column(tmpdir)
        test_policy_file(tmpdir)
        test_resolution()
        test_precision_measurement()
        test_precision_guards()
        test_forward_precision()
        test_directional_weighting()
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
