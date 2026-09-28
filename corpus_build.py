#!/usr/bin/env python3
"""
corpus_build.py -- deterministic corpus construction for the ROB 590 / NIST MSR
literature review (Phase II: AI-Agents for manufacturing).

Why this exists as a script rather than a prompt
------------------------------------------------
The proposal's own background section (S3.3, summarizing [26]) records that
foundation models "exhibit inconsistencies in numerical accuracy" and that their
outputs are "probabilistic and sensitive to input formulation, requiring
validation through deterministic tools or human oversight." S2.4 repeats it.
So: retrieval, deduplication, scoring and counting are done here, in Python,
with a reproducible provenance record. LLMs are used downstream for screening
and extraction only -- never for producing reference lists.

Method
------
1. Resolve the proposal's own bibliography (seeds.csv) to OpenAlex work IDs.
2. Expand one hop backward (works the seeds cite) and one hop forward
   (works that cite the seeds), each direction enabled PER SEED.
3. Deduplicate, then score each candidate by how many DISTINCT seeds link to it.
4. Emit a screening CSV plus a provenance JSON recording every parameter.

Seed roles carry the proposal's intent into the corpus:
  framework_core  [2,7,8,9,10] -- the comparison axis (S2.2 "features that are
                                 common with our definition ... as well as
                                 those that differ"). Forward citations of these
                                 are the highest-yield channel. [2] Bi and [9]
                                 Kovalenko joined the role at v1.5.0; note that
                                 expansion_policy.json -> target_domain.seed_refs
                                 is PRE-REGISTERED at [7,8,9,10] and does NOT
                                 follow the retag. Nothing else depends on the
                                 role: the gates that once did were removed in
                                 v1.6.0.
  survey_baseline   [6,23,24,27] -- the surveys S2.2 explicitly declines to
                                 duplicate. Used as COVERAGE CHECKS: their
                                 reference lists should land largely inside the
                                 corpus. Run with --coverage-check.
  agent_related, standards, dt_core, dt_survey, foundation_models, lab_prior_work

Direction is per seed, not per run (v1.2.0)
-------------------------------------------
A single expand flag cannot express what the seed set actually needs. The six
dt_survey seeds carry 74% of all forward edges -- [14] Liu (1,877 citations) and
[19] Mihai (1,034) alone are 58% of the forward pull -- and what cites a generic
digital-twin survey is the digital-twin field, which S2.1 explicitly declines to
re-survey. Their REFERENCE lists, however, are exactly what the ~15-paper DT
comparison subset needs. So: dt_survey expands backward only.

That forward rule was withdrawn in v1.4.0 and REINSTATED in v1.5.0 on different
grounds. v1.2.0 justified it by claiming the citing works are topically off
-- an assertion about content, which v1.4.0 measured and did not support (see
below and README, "What the measurement actually said"). v1.5.0 does not make
that claim. dt_survey forward expansion is off because S2.1 defines the DT half
of this review as a ~15-paper comparison subset, so the field citing a generic
DT survey is OUT OF SCOPE by the proposal's own scoping -- a declared judgment
about what this review is for, not a measurement of what those papers contain.
p(s)/q(s) neither support nor contradict it, and are not cited for it.

Seed precision is measured, not asserted (v1.3.0)
-------------------------------------------------
An earlier version of this file disabled backward expansion for [35] Zhang on
the stated ground that its 463 references were "nearly all general machine
learning". That was recall, not measurement: not reproducible, not auditable,
and an n-of-1 rule written as though it were policy. It has been removed, along
with the same style of rule on `standards`.

What replaces it is a quantity computed from metadata already in flight:

  T    = union of OpenAlex topics (score >= 0.3) carried by the seeds named in
         expansion_policy.json -> target_domain.seed_refs, which are the
         framework seeds [7,8,10] plus [9]. S2.2 defines the whole review as a
         comparison against that framework, so the framework's own topical
         fingerprint is the natural target rather than a hand-written topic list.
  p(s) = fraction of seed s's references carrying at least one topic in T.

p(s) is then a WEIGHT on scoring, not a gate on retrieval:

  score(c) = sum of p(s) over every seed s that links to candidate c

replacing the raw n_seed_links integer as the ranking term. Two reasons to
prefer a weight here. Retrieval is not the scarce resource -- reference lists
arrive attached to the seed record, and resolving them costs ~48 batched
requests for the whole seed set -- so a gate protects no budget; it only
shortens the screening queue. And exclusion is unauditable: a paper never
retrieved leaves no trace, so "why isn't Smith 2023 in here?" has no answer. A
paper retrieved, ranked low and screened out carries its reason in the record.

The forward mirror, q(s), arrived in v1.4.0 (README has the full treatment).
Citing works do not ride free on the seed record, so q(s) is estimated from a
DETERMINISTIC RANDOM sample -- OpenAlex sample=N&seed=S -- which is an unbiased
estimator that happens to be reproducible, unlike a pagination-order cap. Each
edge is weighted by its seed's precision in the direction that edge was followed.

No gates (v1.6.0)
-----------------
Earlier versions carried pre-registered hard gates in both directions, plus a
warn-only "does T actually discriminate?" ratio. All three are GONE, along with
their thresholds, their provenance fields and their tests.

THE ONLY RETRIEVAL POLICY IS THE PER-SEED expand_backward / expand_forward
COLUMNS OF seeds.csv. Nothing computed at run time decides what is retrieved.

Two reasons this is the right shape. The gates never bit: at the v1.5.0
measurement the forward gate fired on zero seeds, because the only seeds with
enough forward volume to trip it were the dt_survey six, which seeds.csv
already turns off by hand. And the discrimination ratio they leaned on was
measured at 1.05-1.3 against a 1.3 floor -- T separates framework_core from
dt_survey barely or not at all -- so a gate that did not fire was ABSTAINING,
not clearing. A rule that cannot act and cannot justify acting is not a policy;
it is machinery that makes the record harder to read.

p(s) and q(s) survive as RANKING WEIGHTS only. They still describe the seed a
candidate was reached from, which is a real signal for ordering a screening
queue, and a low-weighted paper is still in candidates.csv with its weight
attached. What they no longer do is remove anything.

Everything the weights consume and produce -- T, p(s), q(s), U(s), |R(s)| --
is written to corpus/seed_precision.csv and provenance.json, so a reader can
re-derive every score without rerunning anything.

Usage
-----
    pip install requests
    python corpus_build.py --mailto you@umich.edu

    # smaller/faster first pass
    python corpus_build.py --mailto you@umich.edu --max-forward-per-seed 200

Per-seed direction is set by the `expand_backward` / `expand_forward` columns of
seeds.csv. A legacy single `expand` column is still accepted and applied to both.

    # verify the survey_baseline coverage claim after building
    python corpus_build.py --mailto you@umich.edu --coverage-check

Outputs (into --out-dir, default ./corpus/)
    seeds_resolved.csv   -- VERIFY THIS BY HAND before trusting anything else.
    candidates.csv       -- the screening set, ranked.
    coverage_report.csv  -- only with --coverage-check.
    provenance.json      -- run parameters, counts, timings, API version.

No API key required. OpenAlex asks for a mailto for its polite pool; supply one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

API = "https://api.openalex.org"
SCRIPT_VERSION = "1.6.0"
CURRENT_YEAR = datetime.now(timezone.utc).year

# Fields requested from OpenAlex. Keeping this tight materially reduces payload
# size; abstract_inverted_index dominates, but screening needs abstracts.
SELECT = ",".join([
    "id", "doi", "display_name", "publication_year", "type",
    "primary_location", "authorships", "cited_by_count",
    "referenced_works", "abstract_inverted_index", "open_access",
    "language", "topics",
])

# Work types worth screening. Excludes datasets, editorials, errata, peer reviews.
KEEP_TYPES = {"article", "preprint", "book-chapter", "book", "review", "proceedings-article"}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class OpenAlex:
    def __init__(self, mailto: str, sleep: float = 0.12, retries: int = 4):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"rob590-corpus-build/{SCRIPT_VERSION} (mailto:{mailto})"
        })
        self.mailto = mailto
        self.sleep = sleep
        self.retries = retries
        self.n_requests = 0

    def get(self, path: str, params: dict) -> dict:
        params = dict(params)
        params["mailto"] = self.mailto
        url = f"{API}{path}"
        backoff = 1.0
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=60)
                self.n_requests += 1
                if r.status_code == 200:
                    time.sleep(self.sleep)
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"OpenAlex request failed after {self.retries} attempts: {url}")

    def count(self, path: str, params: dict) -> int:
        """Total results for a filter, without pulling them."""
        data = self.get(path, {**params, "per-page": 1, "select": "id"})
        return int((data.get("meta") or {}).get("count") or 0)

    def sample(self, path: str, params: dict, n: int, seed: int) -> list[dict]:
        """
        A deterministic RANDOM sample, via OpenAlex `sample` + `seed`.

        This is the difference between an estimator and a hidden inclusion
        criterion. A pagination-order cap takes whatever the API returns first,
        which is why every truncation in this script is reported as a defect. A
        seeded random sample returns the same works on every run and is unbiased
        with respect to the thing being measured, so it can carry a measurement.

        `sample` does not work with cursor pagination, so this is one page.
        Note that under `sample`, meta.count reports the SAMPLE size, not the
        population -- use count() for the denominator.
        """
        data = self.get(path, {**params, "sample": n, "seed": seed,
                               "per-page": min(n, 200)})
        return data.get("results", []) or []

    def paged(self, path: str, params: dict, cap: int | None = None,
              meta_out: dict | None = None):
        """
        Cursor-paginate a /works listing. Yields individual work dicts.

        `meta_out`, if given, is filled from the first page's `meta` block. That
        carries the true total for the filter, which is what makes a capped pull
        reportable as "N of M" rather than just "hit the cap".
        """
        params = dict(params)
        params["per-page"] = 200
        params["cursor"] = "*"
        seen = 0
        first = True
        while True:
            data = self.get(path, params)
            if first and meta_out is not None:
                meta_out.update(data.get("meta") or {})
                first = False
            results = data.get("results", [])
            for w in results:
                yield w
                seen += 1
                if cap is not None and seen >= cap:
                    return
            nxt = data.get("meta", {}).get("next_cursor")
            if not nxt or not results:
                return
            params["cursor"] = nxt


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def norm_title(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def short_id(openalex_id: str | None) -> str:
    if not openalex_id:
        return ""
    return openalex_id.rsplit("/", 1)[-1]


def reconstruct_abstract(inv: dict | None, max_words: int = 400) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild linear text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return ""
    positions.sort()
    words = [w for _, w in positions[:max_words]]
    return " ".join(words).replace("\n", " ").strip()


def venue_of(work: dict) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def authors_of(work: dict, limit: int = 6) -> str:
    names = []
    for a in (work.get("authorships") or [])[:limit]:
        author = a.get("author") or {}
        n = author.get("display_name")
        if n:
            names.append(n)
    return "; ".join(names)


def oa_pdf_of(work: dict) -> str:
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    loc = work.get("primary_location") or {}
    return loc.get("pdf_url") or ""


def clean_doi(doi: str | None) -> str:
    if not doi:
        return ""
    return doi.replace("https://doi.org/", "").strip().lower()


def citation_velocity(cited_by_count: int, year: int | None,
                      current_year: int = CURRENT_YEAR) -> float:
    """
    Citations per year since publication, used as the screening-order tiebreak
    instead of raw citation count.

    Rationale: raw counts systematically demote recent work, and proposal S2
    describes these as "emerging, exciting, and fast-moving areas" whose recent
    literature is precisely what the survey must cover. A 2016 paper with 55
    citations should not outrank a 2025 paper with 12 when the question is which
    to read first. Raw cited_by_count is still written to the CSV, so nothing is
    lost -- this only sets reading order.

    Age floor of 1.0 year keeps same-year papers from dividing by zero and from
    being scored as if they had a full year to accumulate citations.
    """
    if not year:
        return 0.0
    age = max(1.0, float(current_year - year))
    return cited_by_count / age


# ----------------------------------------------------------------------------
# Seed resolution
# ----------------------------------------------------------------------------

def _truthy(v, default: str = "yes") -> bool:
    return ((v if v is not None else default) or default).strip().lower() in ("yes", "true", "1")


def load_seeds(path: str) -> list[dict]:
    """
    Read seeds.csv. Direction is per seed: `expand_backward` and `expand_forward`
    are independent columns, because the roles that are worth following backward
    are not the roles worth following forward (see module docstring).

    A file carrying only the pre-1.2.0 single `expand` column still loads; its
    value is applied to both directions, reproducing the old behaviour exactly.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or ())
        rows = [r for r in reader if (r.get("title") or "").strip()]

    split = "expand_backward" in cols or "expand_forward" in cols
    for r in rows:
        if split:
            r["expand_backward"] = _truthy(r.get("expand_backward"))
            r["expand_forward"] = _truthy(r.get("expand_forward"))
        else:
            both = _truthy(r.get("expand"))
            r["expand_backward"] = r["expand_forward"] = both
        r["ref"] = (r.get("ref") or "").strip()
        r["role"] = (r.get("role") or "unassigned").strip()
    return rows


def resolve_seed(api: OpenAlex, seed: dict, min_similarity: float) -> dict:
    """Resolve one seed to an OpenAlex work. DOI first, then title search."""
    result = {
        **{k: seed.get(k, "") for k in ("ref", "role", "year", "first_author", "title")},
        "expand_backward": seed["expand_backward"],
        "expand_forward": seed["expand_forward"],
        "openalex_id": "", "matched_title": "", "matched_year": "",
        "matched_doi": "", "similarity": 0.0, "n_references": 0,
        "cited_by_count": 0, "status": "UNRESOLVED",
    }

    doi = clean_doi(seed.get("doi"))
    work = None
    if doi:
        data = api.get("/works", {"filter": f"doi:{doi}", "select": SELECT})
        hits = data.get("results", [])
        if hits:
            work = hits[0]
            result["status"] = "RESOLVED_BY_DOI"
            result["similarity"] = 1.0

    if work is None:
        title = seed["title"]
        # Quoting keeps multi-word titles together; OpenAlex strips most punctuation.
        query = re.sub(r'["\\]', " ", title)
        data = api.get("/works", {
            "filter": f'title.search:{query}',
            "select": SELECT,
            "per-page": 25,
        })
        best, best_score = None, 0.0
        for cand in data.get("results", []):
            score = title_similarity(title, cand.get("display_name") or "")
            # Year agreement breaks ties between reprints/preprints.
            try:
                if seed.get("year") and cand.get("publication_year") == int(seed["year"]):
                    score += 0.05
            except (TypeError, ValueError):
                pass
            if score > best_score:
                best, best_score = cand, score
        if best is not None and best_score >= min_similarity:
            work = best
            result["similarity"] = round(min(best_score, 1.0), 3)
            result["status"] = "RESOLVED_BY_TITLE" if best_score >= 0.90 else "RESOLVED_LOW_CONFIDENCE"
        elif best is not None:
            result["similarity"] = round(best_score, 3)
            result["matched_title"] = best.get("display_name") or ""
            result["status"] = "REJECTED_LOW_SIMILARITY"

    if work is not None:
        result["openalex_id"] = short_id(work.get("id"))
        result["matched_title"] = work.get("display_name") or ""
        result["matched_year"] = work.get("publication_year") or ""
        result["matched_doi"] = clean_doi(work.get("doi"))
        result["n_references"] = len(work.get("referenced_works") or [])
        result["cited_by_count"] = work.get("cited_by_count") or 0
        result["_work"] = work

    return result


# ----------------------------------------------------------------------------
# Expansion
# ----------------------------------------------------------------------------

def fetch_works_by_ids(api: OpenAlex, ids: list[str], batch: int = 50):
    """Batch-fetch works by OpenAlex short ID (OR filter caps at 50 values)."""
    ids = [i for i in ids if i]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        data = api.get("/works", {
            "filter": "openalex_id:" + "|".join(chunk),
            "select": SELECT,
            "per-page": batch,
        })
        for w in data.get("results", []):
            yield w


# ----------------------------------------------------------------------------
# Seed precision: the measured basis for weighting (and, optionally, gating)
# ----------------------------------------------------------------------------

DEFAULT_POLICY = {
    "policy_version": "built-in-default",
    "target_domain": {"seed_refs": ["7", "8", "9", "10"], "topic_score_min": 0.3},
    "precision": {"min_sample": 30},
    "forward_precision": {"sample_size": 200, "sample_seed": 42, "min_sample": 30},
    "scoring": {"weight_by_precision": True},
}


def load_policy(path: str | None) -> dict:
    """
    Read the pre-registered scoring policy.

    Since v1.6.0 this file governs RANKING only. It sets the target domain T and
    the sample sizes behind p(s)/q(s), which weight the seed-link score. It
    does not decide what is retrieved: that is the expand_backward /
    expand_forward columns of seeds.csv and nothing else.

    The parameters still live in a committed JSON file rather than in argparse
    defaults so that git history shows they were fixed BEFORE the run they
    govern -- the difference between a pre-registered rule and a post-hoc one.
    """
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        for section, value in loaded.items():
            if isinstance(value, dict) and isinstance(policy.get(section), dict):
                policy[section].update(value)
            else:
                policy[section] = value
    elif path:
        raise SystemExit(f"policy file not found: {path}  (pass --policy '' to use built-in defaults)")
    return policy


def topic_ids(work: dict, min_score: float) -> set[str]:
    """OpenAlex `topics`, not the deprecated `concepts`. Score-filtered."""
    out = set()
    for t in (work.get("topics") or []):
        try:
            if float(t.get("score") or 0.0) < min_score:
                continue
        except (TypeError, ValueError):
            continue
        tid = short_id(t.get("id"))
        if tid:
            out.add(tid)
    return out


def target_domain(resolved: list[dict], policy: dict, verbose: bool) -> tuple[set[str], list[str]]:
    """
    T -- the topical fingerprint of the framework this review is a comparison
    against. Derived from the seeds named in the policy file, not hand-written.
    """
    cfg = policy["target_domain"]
    want = set(cfg["seed_refs"])
    min_score = float(cfg["topic_score_min"])
    T, used = set(), []
    for s in resolved:
        if s["ref"] in want and s.get("_work"):
            got = topic_ids(s["_work"], min_score)
            if got:
                T |= got
                used.append(s["ref"])
    missing = sorted(want - set(used), key=lambda x: (len(x), x))
    if verbose:
        print(f"  [domain]   T = {len(T)} topics (score >= {min_score}) from seeds "
              f"{sorted(used, key=lambda x: (len(x), x))}", flush=True)
        if missing:
            print(f"  [domain]   *** target seeds contributed no topics: {missing}", flush=True)
    return T, missing


def _fill_weights(rows: list[dict], p_key: str, n_key: str, declared_key: str,
                  weight_key: str, basis_key: str, min_sample: int,
                  empty_label: str) -> None:
    """
    Turn raw precision estimates into usable weights, applying the two guards.

    Shared by both directions because the guards are the same in both: an
    estimate on too small a sample is noise and must be replaced by a default
    (same-role mean, then global mean) with the substitution recorded, and a
    seed with no data at all is a gap in OpenAlex rather than a finding about
    the paper.

    If NOTHING is measurable the fallback is 1.0, i.e. uniform weighting, which
    reproduces the unweighted ranking. A global mean of 0.0 would zero every
    weight and silently destroy the sort order, turning "could not measure this"
    into "none of it counts".
    """
    measured = {r["ref"]: r[p_key] for r in rows
                if r[p_key] is not None and r[n_key] >= min_sample}
    by_role: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["ref"] in measured:
            by_role[r["role"]].append(measured[r["ref"]])

    if measured:
        global_mean = round(sum(measured.values()) / len(measured), 4)
        global_label = "global_default"
    else:
        global_mean = 1.0
        global_label = "global_default(uniform:no_measurable_seeds)"

    for r in rows:
        if r["ref"] in measured:
            r[basis_key] = "measured"
            r[weight_key] = r[p_key]
            continue
        if not r[declared_key]:
            r[basis_key] = empty_label
        else:
            r[basis_key] = f"under_sampled(n={r[n_key]}<{min_sample})"
        pool = by_role.get(r["role"], [])
        if pool:
            r[weight_key] = round(sum(pool) / len(pool), 4)
            r[basis_key] += f"|role_default:{r['role']}"
        else:
            r[weight_key] = global_mean
            r[basis_key] += f"|{global_label}"


def _precision_row(s: dict) -> dict:
    """One row per seed, carrying both directions. Written to seed_precision.csv."""
    return {
        "ref": s["ref"], "role": s["role"], "first_author": s.get("first_author", ""),
        "expand_backward": s["expand_backward"], "expand_forward": s["expand_forward"],
        # backward
        "n_references_declared": s.get("n_references", 0),
        "n_references_resolved": 0, "n_unique_references": 0,
        "n_refs_on_target": 0, "n_unique_refs_on_target": 0,
        "bwd_precision_all": None, "bwd_precision_unique": None,
        "bwd_basis": "not_expanded", "bwd_weight": 1.0,
        # forward
        "n_citations_window": 0, "n_sampled": 0, "n_unique_sampled": 0,
        "n_cites_on_target": 0, "n_unique_cites_on_target": 0,
        "est_unique_citations": 0,
        "fwd_precision_all": None, "fwd_precision_unique": None,
        "fwd_basis": "not_expanded", "fwd_weight": 1.0,
    }


def measure_backward_precision(rows: list[dict], seed_refs: dict[str, list[str]],
                               ref_works: dict, T: set[str], policy: dict,
                               verbose: bool) -> None:
    """
    p(s) -- fraction of seed s's references carrying at least one topic in T.

    Reported over two denominators:
      bwd_precision_all    -- over all of s's resolvable references. This is the
                              scoring weight: it describes the seed, not its margin.
      bwd_precision_unique -- over U(s), references reached from s and no other
                              seed. Retained as a diagnostic only; it fed the
                              backward gate, which was removed in v1.6.0.
    """
    min_sample = int(policy["precision"]["min_sample"])
    min_score = float(policy["target_domain"]["topic_score_min"])

    on_target = {wid: bool(topic_ids(w, min_score) & T) for wid, w in ref_works.items()}
    owners: dict[str, set[str]] = defaultdict(set)
    for ref, ids in seed_refs.items():
        for rid in ids:
            owners[rid].add(ref)

    for r in rows:
        ids = seed_refs.get(r["ref"])
        if ids is None:
            continue
        known = [i for i in ids if i in on_target]
        unique = [i for i in known if len(owners[i]) == 1]
        r["n_references_resolved"] = len(known)
        r["n_unique_references"] = len(unique)
        r["n_refs_on_target"] = sum(1 for i in known if on_target[i])
        r["n_unique_refs_on_target"] = sum(1 for i in unique if on_target[i])
        r["bwd_precision_all"] = round(r["n_refs_on_target"] / len(known), 4) if known else None
        r["bwd_precision_unique"] = (round(r["n_unique_refs_on_target"] / len(unique), 4)
                                     if unique else None)

    eligible = [r for r in rows if seed_refs.get(r["ref"]) is not None]
    _fill_weights(eligible, "bwd_precision_all", "n_references_resolved",
                  "n_references_declared", "bwd_weight", "bwd_basis",
                  min_sample, "no_references")
    if verbose:
        for r in sorted(eligible, key=lambda r: (len(r["ref"]), r["ref"])):
            pa = "  n/a" if r["bwd_precision_all"] is None else f"{r['bwd_precision_all']:.3f}"
            print(f"  [p(s)]     [{r['ref']:>2}] {r['role']:17} p={pa} w={r['bwd_weight']:.3f}"
                  f"  R={r['n_references_resolved']:>3} U={r['n_unique_references']:>3}"
                  f"  {r['bwd_basis']}", flush=True)


def sample_forward_citations(api: OpenAlex, resolved: list[dict], min_year: int,
                             policy: dict, verbose: bool) -> dict[str, dict]:
    """
    A deterministic random sample of each seed's citing works, for q(s).

    Backward references arrive attached to the seed record; citing works do not,
    so forward precision has to be estimated rather than counted. Two requests
    per seed -- one count over the min-year window for the denominator, one
    seeded sample -- which is what makes it affordable to measure the forward
    direction at all instead of asserting it.
    """
    cfg = policy["forward_precision"]
    n = int(cfg["sample_size"])
    seed_val = int(cfg["sample_seed"])
    out: dict[str, dict] = {}
    for s in resolved:
        if not s.get("openalex_id") or not s["expand_forward"]:
            continue
        params = {"filter": f"cites:{s['openalex_id']},"
                            f"from_publication_date:{min_year}-01-01",
                  "select": SELECT}
        total = api.count("/works", params)
        works = api.sample("/works", params, min(n, max(total, 1)), seed_val) if total else []
        out[s["ref"]] = {"total": total, "works": works}
        if verbose:
            print(f"  [sample]   seed [{s['ref']:>2}] {len(works)} of {total} citing works "
                  f"(sample={n}, seed={seed_val})", flush=True)
    return out


def measure_forward_precision(rows: list[dict], samples: dict[str, dict],
                              seed_ids: dict[str, str], T: set[str], policy: dict,
                              verbose: bool) -> None:
    """
    q(s) -- the forward mirror of p(s), over a sample of works CITING seed s.

    Uniqueness is exact rather than estimated: a sampled citing work's own
    referenced_works says which other seeds it cites, so "cites s and no other
    seed" is decidable per sampled work. The unique fraction of the sample then
    scales the window total into est_unique_citations, retained as a diagnostic
    only; it was the volume term of the forward gate, removed in v1.6.0.
    """
    cfg = policy["forward_precision"]
    min_sample = int(cfg["min_sample"])
    min_score = float(policy["target_domain"]["topic_score_min"])
    id_to_ref = {wid: ref for ref, wid in seed_ids.items()}

    for r in rows:
        got = samples.get(r["ref"])
        if got is None:
            continue
        works = got["works"]
        r["n_citations_window"] = got["total"]
        r["n_sampled"] = len(works)
        unique, n_hit, u_hit = 0, 0, 0
        for w in works:
            hit = bool(topic_ids(w, min_score) & T)
            n_hit += hit
            cited_seeds = {id_to_ref[short_id(x)] for x in (w.get("referenced_works") or [])
                           if short_id(x) in id_to_ref}
            if cited_seeds <= {r["ref"]}:
                unique += 1
                u_hit += hit
        r["n_unique_sampled"] = unique
        r["n_cites_on_target"] = n_hit
        r["n_unique_cites_on_target"] = u_hit
        r["fwd_precision_all"] = round(n_hit / len(works), 4) if works else None
        r["fwd_precision_unique"] = round(u_hit / unique, 4) if unique else None
        r["est_unique_citations"] = (round(got["total"] * unique / len(works))
                                     if works else 0)

    eligible = [r for r in rows if r["ref"] in samples]
    _fill_weights(eligible, "fwd_precision_all", "n_sampled",
                  "n_citations_window", "fwd_weight", "fwd_basis",
                  min_sample, "no_citations")
    if verbose:
        for r in sorted(eligible, key=lambda r: (len(r["ref"]), r["ref"])):
            qa = "  n/a" if r["fwd_precision_all"] is None else f"{r['fwd_precision_all']:.3f}"
            print(f"  [q(s)]     [{r['ref']:>2}] {r['role']:17} q={qa} w={r['fwd_weight']:.3f}"
                  f"  n={r['n_sampled']:>3}/{r['n_citations_window']:<5}"
                  f" estU={r['est_unique_citations']:>4}  {r['fwd_basis']}", flush=True)


def _truncation(s: dict, direction: str, available: int, cap: int) -> dict:
    """
    Record a cap that actually bit.

    Truncation is not a neutral subsample: OpenAlex returns references and citing
    works in its own order, so a cap silently substitutes API pagination order for
    an inclusion criterion. That is not defensible in a systematic review, so every
    instance is printed loudly and carried into provenance.json for the report.
    """
    return {
        "ref": s["ref"],
        "first_author": s.get("first_author", ""),
        "role": s.get("role", ""),
        "direction": direction,
        "available": available,
        "retrieved": cap,
        "dropped": available - cap,
    }


def collect_backward_refs(resolved: list[dict]) -> dict[str, list[str]]:
    """
    Reference IDs per backward-eligible seed, uncapped.

    Free: OpenAlex ships `referenced_works` on the seed record itself, so this
    costs no requests. "Eligible" means expand_backward=yes in seeds.csv, which
    since v1.6.0 is the whole of the retrieval policy.
    """
    out: dict[str, list[str]] = {}
    for s in resolved:
        if not s.get("_work") or not s["expand_backward"]:
            continue
        out[s["ref"]] = [short_id(r) for r in (s["_work"].get("referenced_works") or [])]
    return out


def fetch_reference_works(api: OpenAlex, seed_refs: dict[str, list[str]],
                          verbose: bool) -> dict:
    """
    Resolve every collected reference once, deduplicated across seeds.

    This is the whole marginal cost of backward expansion: ~50 works per request.
    """
    ids = sorted({rid for ids in seed_refs.values() for rid in ids})
    if verbose:
        print(f"  [backward] fetching {len(ids)} unique cited works "
              f"(~{(len(ids) + 49) // 50} requests)", flush=True)
    works = {}
    for w in fetch_works_by_ids(api, ids):
        works[short_id(w.get("id"))] = w
    return works


def expand_backward(resolved: list[dict], seed_refs: dict[str, list[str]],
                    ref_works: dict,
                    cap_per_seed: int, verbose: bool):
    """
    Turn the collected references into candidate links. Takes no api:
    everything was fetched already.

    Which seeds expand backward is decided solely by the `expand_backward`
    column of seeds.csv (v1.6.0). There is no precision gate in either
    direction any more.
    """
    wanted: dict[str, set[str]] = defaultdict(set)   # candidate id -> seed refs
    # candidate id -> seed ref -> directions. Per EDGE, not per candidate: the
    # two directions carry different weights, so which seed reached a candidate
    # which way has to survive into scoring.
    edge_dirs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    truncated: list[dict] = []
    by_ref = {s["ref"]: s for s in resolved}

    for ref, ids in seed_refs.items():
        s = by_ref[ref]
        refs = list(ids)
        available = len(refs)
        if cap_per_seed and available > cap_per_seed:
            refs = refs[:cap_per_seed]
            truncated.append(_truncation(s, "backward", available, cap_per_seed))
        for rid in refs:
            wanted[rid].add(ref)
            edge_dirs[rid][ref].add("backward")
        if verbose:
            note = ""
            if cap_per_seed and available > cap_per_seed:
                note = (f"  *** TRUNCATED: {available - cap_per_seed} of {available} "
                        f"references dropped by --max-backward-per-seed={cap_per_seed}")
            print(f"  [backward] seed [{ref}] -> {len(refs)} references{note}", flush=True)

    works = {wid: w for wid, w in ref_works.items() if wid in wanted}
    return works, wanted, edge_dirs, truncated


def expand_forward(api: OpenAlex, resolved: list[dict], cap_per_seed: int,
                   min_year: int, verbose: bool):
    """
    Works that CITE the seeds. Highest-yield channel for framework_core.

    Which seeds expand forward is decided solely by the `expand_forward` column
    of seeds.csv (v1.6.0). There is no precision gate here any more.
    """
    works, wanted = {}, defaultdict(set)
    edge_dirs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    truncated: list[dict] = []
    for s in resolved:
        if not s.get("openalex_id") or not s["expand_forward"]:
            continue
        params = {
            "filter": f"cites:{s['openalex_id']},from_publication_date:{min_year}-01-01",
            "select": SELECT,
        }
        meta: dict = {}
        n = 0
        for w in api.paged("/works", params, cap=cap_per_seed, meta_out=meta):
            wid = short_id(w.get("id"))
            works[wid] = w
            wanted[wid].add(s["ref"])
            edge_dirs[wid][s["ref"]].add("forward")
            n += 1
        available = meta.get("count", n) or n
        hit_cap = bool(cap_per_seed) and n >= cap_per_seed and available > n
        if hit_cap:
            truncated.append(_truncation(s, "forward", available, n))
        if verbose:
            note = ""
            if hit_cap:
                note = (f"  *** TRUNCATED: {available - n} of {available} citing works "
                        f"dropped by --max-forward-per-seed={cap_per_seed}")
            print(f"  [forward]  seed [{s['ref']}] <- {n} citing works{note}", flush=True)
    return works, wanted, edge_dirs, truncated


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def build_candidates(all_works: dict, links: dict, edge_dirs: dict,
                     seed_roles: dict, seed_ids: set, min_year: int,
                     seed_weights: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """
    Assemble, filter, score, rank.

    Ranking term is `seed_weight` = sum over the seeds linking to the candidate
    of that seed's precision IN THE DIRECTION the link was followed -- p(s)
    backward, q(s) forward. A work reached only from a low-precision survey
    should not tie with one reached from the framework the review is built
    around, and a work reached from both should outrank either.

    A seed that reaches the same candidate BOTH ways contributes once, at the
    higher of its two weights. Summing both would let a single seed outweigh two
    genuinely independent ones, which is the opposite of what seed breadth is
    supposed to measure. `n_seed_links` stays in the CSV as the unweighted audit
    trail.
    """
    def edge_weight(ref: str, dirs: set[str]) -> float:
        if not seed_weights:
            return 1.0
        w = seed_weights.get(ref) or {}
        return max((w.get(d, 1.0) for d in dirs), default=1.0)

    rows = []
    seen_titles: dict[str, str] = {}

    for wid, work in all_works.items():
        if wid in seed_ids:
            continue
        year = work.get("publication_year")
        if not year or year < min_year:
            continue
        if (work.get("type") or "") not in KEEP_TYPES:
            continue
        if (work.get("language") or "en") != "en":
            continue

        title = work.get("display_name") or ""
        if not title:
            continue

        # Title-level dedupe catches preprint/published pairs OpenAlex keeps apart.
        nt = norm_title(title)
        if nt in seen_titles:
            continue
        seen_titles[nt] = wid

        refs = sorted(links.get(wid, set()), key=lambda r: (len(r), r))
        roles = sorted({seed_roles.get(r, "unassigned") for r in refs})
        per_edge = edge_dirs.get(wid, {})
        edge_ws = [edge_weight(r, per_edge.get(r) or {"backward"}) for r in refs]
        weight = sum(edge_ws)
        dirs = sorted({d for r in refs for d in (per_edge.get(r) or set())})

        rows.append({
            "openalex_id": wid,
            "doi": clean_doi(work.get("doi")),
            "title": title,
            "year": year,
            "venue": venue_of(work),
            "authors": authors_of(work),
            "type": work.get("type") or "",
            "cited_by_count": work.get("cited_by_count") or 0,
            "citations_per_year": round(
                citation_velocity(work.get("cited_by_count") or 0, year), 2),
            "n_seed_links": len(refs),
            "seed_weight": round(weight, 4),
            "seed_link_weights": "|".join(f"{w:.3f}" for w in edge_ws),
            "seed_link_directions": "|".join(
                ",".join(sorted(per_edge.get(r) or [])) or "?" for r in refs),
            "seed_refs": "|".join(refs),
            "seed_roles": "|".join(roles),
            "linked_to_framework_core": int("framework_core" in roles),
            "direction": "|".join(dirs),
            "oa_pdf_url": oa_pdf_of(work),
            "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
            # Filled downstream by the screening step.
            "screen_decision": "",
            "screen_reason": "",
            "screen_model": "",
            "human_label": "",
        })

    # Priority for screening: framework_core linkage, then PRECISION-WEIGHTED
    # seed linkage (v1.3.0 -- was the raw link count), then citation VELOCITY
    # (not raw count -- see citation_velocity for why), then newest first.
    # Fully deterministic; no model involved. Nothing here removes a row: a
    # low-weight candidate is still in the file, with its weight attached, and
    # is excluded only by a screening decision that carries a reason.
    rows.sort(key=lambda r: (-r["linked_to_framework_core"], -r["seed_weight"],
                             -r["n_seed_links"], -r["citations_per_year"], -r["year"]))
    return rows


def coverage_check(api: OpenAlex, resolved: list[dict], corpus_ids: set,
                   out_path: str, verbose: bool) -> list[dict]:
    """
    Do the four survey_baseline papers' own reference lists land inside the corpus?

    S2.2 declines to write "yet another survey ... [6,23,24,27]". Those surveys
    therefore serve as an external check on seed-set breadth: if a large share of
    what they cite is missing here, the seeds are too narrow. This is a
    deterministic test, not a judgment call.
    """
    report = []
    for s in resolved:
        if s["role"] != "survey_baseline" or not s.get("_work"):
            continue
        refs = {short_id(r) for r in (s["_work"].get("referenced_works") or [])}
        if not refs:
            continue
        covered = refs & corpus_ids
        pct = 100.0 * len(covered) / len(refs)
        report.append({
            "ref": s["ref"],
            "first_author": s["first_author"],
            "year": s["matched_year"],
            "n_references": len(refs),
            "n_in_corpus": len(covered),
            "pct_covered": round(pct, 1),
        })
        if verbose:
            print(f"  [coverage] survey [{s['ref']}] {s['first_author']}: "
                  f"{len(covered)}/{len(refs)} ({pct:.1f}%) of its references are in the corpus",
                  flush=True)
    if report:
        write_csv(out_path, report, list(report[0].keys()))
    return report


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mailto", required=True,
                   help="Your email. OpenAlex polite pool; no key needed.")
    p.add_argument("--seeds", default="seeds.csv")
    p.add_argument("--out-dir", default="corpus")
    p.add_argument("--min-year", type=int, default=2015,
                   help="Candidate cutoff. Seeds themselves are exempt. Default 2015.")
    p.add_argument("--max-forward-per-seed", type=int, default=600,
                   help="Cap on citing works pulled per seed. 0 = uncapped.")
    p.add_argument("--max-backward-per-seed", type=int, default=600,
                   help="Cap on references pulled per seed. 0 = uncapped. Default 600 "
                        "clears the longest bibliography in the seed set ([13] Leitao, "
                        "515 refs); at 250 three seeds silently lost 580 references "
                        "chosen by API pagination order.")
    p.add_argument("--policy", default="expansion_policy.json",
                   help="Pre-registered policy for the seed-precision WEIGHTS "
                        "(target domain, sample sizes). It no longer gates "
                        "retrieval: as of v1.6.0 the only retrieval policy is "
                        "the expand_backward/expand_forward columns of "
                        "seeds.csv. Pass '' to use built-in defaults.")
    p.add_argument("--min-similarity", type=float, default=0.72,
                   help="Title-match floor for seed resolution.")
    p.add_argument("--skip-backward", action="store_true")
    p.add_argument("--skip-forward", action="store_true")
    p.add_argument("--coverage-check", action="store_true",
                   help="Report how much of each survey_baseline reference list is covered.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    verbose = not args.quiet
    os.makedirs(args.out_dir, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.time()

    policy = load_policy(args.policy or None)

    api = OpenAlex(args.mailto)
    seeds = load_seeds(args.seeds)
    if verbose:
        print(f"Loaded {len(seeds)} seeds from {args.seeds}", flush=True)

    # --- 1. resolve -------------------------------------------------------
    if verbose:
        print("\n[1/4] Resolving seeds against OpenAlex ...", flush=True)
    resolved = []
    for s in seeds:
        r = resolve_seed(api, s, args.min_similarity)
        resolved.append(r)
        if verbose:
            flag = "ok " if r["status"].startswith("RESOLVED") else "MISS"
            if r["status"] == "RESOLVED_LOW_CONFIDENCE":
                flag = "??  "
            print(f"  {flag} [{r['ref']:>2}] {r['status']:<24} sim={r['similarity']:<5} "
                  f"{(r['matched_title'] or r['title'])[:66]}", flush=True)

    write_csv(os.path.join(args.out_dir, "seeds_resolved.csv"), resolved, [
        "ref", "role", "expand_backward", "expand_forward", "status",
        "similarity", "year", "matched_year",
        "first_author", "title", "matched_title", "matched_doi", "openalex_id",
        "n_references", "cited_by_count",
    ])

    n_unresolved = sum(1 for r in resolved if not r["openalex_id"])
    n_low = sum(1 for r in resolved if r["status"] == "RESOLVED_LOW_CONFIDENCE")
    if verbose:
        print(f"\n  resolved {len(resolved) - n_unresolved}/{len(resolved)}"
              f"  ({n_low} low-confidence)", flush=True)
        print("  >>> Open seeds_resolved.csv and verify every match by hand before", flush=True)
        print("  >>> screening. A wrong seed silently biases the whole corpus.", flush=True)

    seed_ids = {r["openalex_id"] for r in resolved if r["openalex_id"]}
    seed_roles = {r["ref"]: r["role"] for r in resolved}

    # --- 2. expand --------------------------------------------------------
    all_works: dict = {}
    links: dict = defaultdict(set)
    edge_dirs: dict = defaultdict(lambda: defaultdict(set))
    truncations: list[dict] = []

    def merge(works, wanted, edges, truncated):
        all_works.update(works)
        for k, v in wanted.items():
            links[k] |= v
        for wid, per_ref in edges.items():
            for ref, dirs in per_ref.items():
                edge_dirs[wid][ref] |= dirs
        truncations.extend(truncated)

    precision_rows: list[dict] = [_precision_row(r) for r in resolved]
    by_ref = {r["ref"]: r for r in precision_rows}
    seed_id_map = {r["ref"]: r["openalex_id"] for r in resolved if r["openalex_id"]}
    domain_missing: list[str] = []

    # T is needed by both directions, so it is computed once up front. It costs
    # nothing: the topics ride on the seed records already resolved in step 1.
    T, domain_missing = target_domain(resolved, policy, verbose)

    if not args.skip_backward:
        if verbose:
            print("\n[2/4] Backward expansion (works the seeds cite) ...", flush=True)
        seed_refs = collect_backward_refs(resolved)
        ref_works = fetch_reference_works(api, seed_refs, verbose)
        measure_backward_precision(precision_rows, seed_refs, ref_works, T, policy, verbose)
        merge(*expand_backward(resolved, seed_refs, ref_works,
                               args.max_backward_per_seed, verbose))
    elif verbose:
        print("\n[2/4] Backward expansion SKIPPED; p(s) cannot be measured, so every "
              "backward weight falls back to 1.0", flush=True)

    if not args.skip_forward:
        if verbose:
            print("\n[3/4] Forward expansion (works that cite the seeds) ...", flush=True)
        samples = sample_forward_citations(api, resolved, args.min_year, policy, verbose)
        measure_forward_precision(precision_rows, samples, seed_id_map, T, policy, verbose)
        merge(*expand_forward(api, resolved, args.max_forward_per_seed or None,
                              args.min_year, verbose))
    elif verbose:
        print("\n[3/4] Forward expansion SKIPPED; q(s) cannot be measured, so every "
              "forward weight falls back to 1.0", flush=True)

    # Each edge is weighted by its seed's precision IN THAT DIRECTION.
    seed_weights = {r["ref"]: {"backward": r["bwd_weight"], "forward": r["fwd_weight"]}
                    for r in precision_rows}

    # --- 3. assemble ------------------------------------------------------
    if verbose:
        print("\n[4/4] Deduplicating, filtering, scoring ...", flush=True)
    candidates = build_candidates(all_works, links, edge_dirs,
                                  seed_roles, seed_ids, args.min_year,
                                  seed_weights if policy["scoring"]["weight_by_precision"]
                                  else None)

    # Per-seed audit table: every input to every weighting and gating decision,
    # so a reader can re-derive the ranking without rerunning the pipeline.
    prec_path = os.path.join(args.out_dir, "seed_precision.csv")
    if precision_rows:
        prec_out = [dict(r) for r in precision_rows]
        write_csv(prec_path, prec_out, list(prec_out[0].keys()))

    cand_path = os.path.join(args.out_dir, "candidates.csv")
    write_csv(cand_path, candidates, list(candidates[0].keys()) if candidates else ["openalex_id"])

    # --- 4. coverage ------------------------------------------------------
    coverage = []
    if args.coverage_check:
        if verbose:
            print("\n[+] Coverage check against survey_baseline reference lists ...", flush=True)
        corpus_ids = {c["openalex_id"] for c in candidates} | seed_ids
        coverage = coverage_check(api, resolved, corpus_ids,
                                  os.path.join(args.out_dir, "coverage_report.csv"), verbose)

    # --- provenance -------------------------------------------------------
    by_links = defaultdict(int)
    for c in candidates:
        by_links[min(c["n_seed_links"], 5)] += 1

    provenance = {
        "script": "corpus_build.py",
        "script_version": SCRIPT_VERSION,
        "run_started_utc": started.isoformat(),
        "run_seconds": round(time.time() - t0, 1),
        "api": "OpenAlex",
        "api_requests": api.n_requests,
        "parameters": vars(args) | {"mailto": "<redacted>"},
        "counts": {
            "seeds_loaded": len(seeds),
            "seeds_resolved": len(seeds) - n_unresolved,
            "seeds_low_confidence": n_low,
            "unique_works_retrieved": len(all_works),
            "candidates_after_filtering": len(candidates),
            "candidates_by_n_seed_links": {str(k): v for k, v in sorted(by_links.items())},
            "candidates_linked_to_framework_core":
                sum(c["linked_to_framework_core"] for c in candidates),
        },
        "policy": policy,
        "seed_precision": {
            "target_domain_topics": sorted(T),
            "target_domain_seeds_without_topics": domain_missing,
            "per_seed": [
                dict(r) for r in precision_rows
            ],
            "note": ("p(s) and q(s) are measured from OpenAlex `topics`, itself a "
                     "classifier with its own error rate; the raw counts are reported "
                     "here so a reader can judge rather than taking the threshold as "
                     "objective. q(s) additionally rests on a seeded random sample of "
                     "citing works -- reproducible, but an estimate with sampling error "
                     "of roughly 0.5/sqrt(n_sampled)."),
        },
        "seed_expansion": {
            "backward_enabled": sorted(
                (r["ref"] for r in resolved if r["expand_backward"]), key=lambda x: (len(x), x)),
            "forward_enabled": sorted(
                (r["ref"] for r in resolved if r["expand_forward"]), key=lambda x: (len(x), x)),
        },
        # Empty list = no cap bit, i.e. every enabled edge was actually followed.
        # A non-empty list means retrieval was decided partly by OpenAlex
        # pagination order; say so in the report rather than quietly reporting N.
        "truncations": truncations,
        "coverage_check": coverage,
        "note": ("Retrieval, deduplication and counting performed deterministically. "
                 "No language model contributed to this reference list. See proposal "
                 "S3.3 and [26] on FM numerical reliability."),
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    if verbose:
        print(f"\nDone in {provenance['run_seconds']}s, {api.n_requests} API requests.")
        print(f"  {len(all_works)} unique works retrieved")
        print(f"  {len(candidates)} candidates after year/type/language/dedupe filters")
        print(f"  {provenance['counts']['candidates_linked_to_framework_core']} linked to "
              f"framework_core [7,8,10]")
        if truncations:
            total_dropped = sum(t["dropped"] for t in truncations)
            print(f"\n  *** {len(truncations)} seed/direction pair(s) hit a cap; "
                  f"{total_dropped} edges dropped by pagination order, not by criterion.")
            for t in truncations:
                print(f"  ***   [{t['ref']}] {t['first_author']} {t['direction']}: "
                      f"kept {t['retrieved']} of {t['available']}")
            print("  *** Raise --max-{backward,forward}-per-seed or justify this in the report.")
        print(f"\n  -> {cand_path}")
        if precision_rows:
            print(f"  -> {prec_path}  (per-seed p(s), q(s), U(s))")
        print(f"  -> {os.path.join(args.out_dir, 'seeds_resolved.csv')}  (verify by hand)")
        print(f"  -> {os.path.join(args.out_dir, 'provenance.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
