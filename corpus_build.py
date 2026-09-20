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
   (works that cite the seeds).
3. Deduplicate, then score each candidate by how many DISTINCT seeds link to it.
4. Emit a screening CSV plus a provenance JSON recording every parameter.

Seed roles carry the proposal's intent into the corpus:
  framework_core    [7,8,10]  -- the comparison axis (S2.2 "features that are
                                 common with our definition ... as well as
                                 those that differ"). Forward citations of these
                                 are the highest-yield channel.
  survey_baseline   [6,23,24,27] -- the surveys S2.2 explicitly declines to
                                 duplicate. Used as COVERAGE CHECKS: their
                                 reference lists should land largely inside the
                                 corpus. Run with --coverage-check.
  agent_related, standards, dt_core, dt_survey, foundation_models, lab_prior_work

Usage
-----
    pip install requests
    python corpus_build.py --mailto you@umich.edu

    # smaller/faster first pass
    python corpus_build.py --mailto you@umich.edu --max-forward-per-seed 200

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
SCRIPT_VERSION = "1.1.0"
CURRENT_YEAR = datetime.now(timezone.utc).year

# Fields requested from OpenAlex. Keeping this tight materially reduces payload
# size; abstract_inverted_index dominates, but screening needs abstracts.
SELECT = ",".join([
    "id", "doi", "display_name", "publication_year", "type",
    "primary_location", "authorships", "cited_by_count",
    "referenced_works", "abstract_inverted_index", "open_access",
    "language",
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

    def paged(self, path: str, params: dict, cap: int | None = None):
        """Cursor-paginate a /works listing. Yields individual work dicts."""
        params = dict(params)
        params["per-page"] = 200
        params["cursor"] = "*"
        seen = 0
        while True:
            data = self.get(path, params)
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

def load_seeds(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("title") or "").strip()]
    for r in rows:
        r["expand"] = (r.get("expand", "yes") or "yes").strip().lower() in ("yes", "true", "1")
        r["ref"] = (r.get("ref") or "").strip()
        r["role"] = (r.get("role") or "unassigned").strip()
    return rows


def resolve_seed(api: OpenAlex, seed: dict, min_similarity: float) -> dict:
    """Resolve one seed to an OpenAlex work. DOI first, then title search."""
    result = {
        **{k: seed.get(k, "") for k in ("ref", "role", "year", "first_author", "title")},
        "expand": seed["expand"],
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


def expand_backward(api: OpenAlex, resolved: list[dict], cap_per_seed: int, verbose: bool):
    """Works cited BY the seeds. Cheap: reference lists come with the seed record."""
    wanted: dict[str, set[str]] = defaultdict(set)   # candidate id -> seed refs
    direction: dict[str, set[str]] = defaultdict(set)
    for s in resolved:
        if not s.get("_work") or not s["expand"]:
            continue
        refs = [short_id(r) for r in (s["_work"].get("referenced_works") or [])]
        if cap_per_seed and len(refs) > cap_per_seed:
            refs = refs[:cap_per_seed]
        for rid in refs:
            wanted[rid].add(s["ref"])
            direction[rid].add("backward")
        if verbose:
            print(f"  [backward] seed [{s['ref']}] -> {len(refs)} references", flush=True)

    works = {}
    ids = list(wanted)
    if verbose:
        print(f"  [backward] fetching {len(ids)} unique cited works", flush=True)
    for w in fetch_works_by_ids(api, ids):
        works[short_id(w.get("id"))] = w
    return works, wanted, direction


def expand_forward(api: OpenAlex, resolved: list[dict], cap_per_seed: int,
                   min_year: int, verbose: bool):
    """Works that CITE the seeds. Highest-yield channel for framework_core."""
    works, wanted, direction = {}, defaultdict(set), defaultdict(set)
    for s in resolved:
        if not s.get("openalex_id") or not s["expand"]:
            continue
        params = {
            "filter": f"cites:{s['openalex_id']},from_publication_date:{min_year}-01-01",
            "select": SELECT,
        }
        n = 0
        for w in api.paged("/works", params, cap=cap_per_seed):
            wid = short_id(w.get("id"))
            works[wid] = w
            wanted[wid].add(s["ref"])
            direction[wid].add("forward")
            n += 1
        if verbose:
            capped = " (CAPPED)" if cap_per_seed and n >= cap_per_seed else ""
            print(f"  [forward]  seed [{s['ref']}] <- {n} citing works{capped}", flush=True)
    return works, wanted, direction


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def build_candidates(all_works: dict, links: dict, directions: dict,
                     seed_roles: dict, seed_ids: set, min_year: int) -> list[dict]:
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
            "seed_refs": "|".join(refs),
            "seed_roles": "|".join(roles),
            "linked_to_framework_core": int("framework_core" in roles),
            "direction": "|".join(sorted(directions.get(wid, set()))),
            "oa_pdf_url": oa_pdf_of(work),
            "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
            # Filled downstream by the screening step.
            "screen_decision": "",
            "screen_reason": "",
            "screen_model": "",
            "human_label": "",
        })

    # Priority for screening: framework_core linkage, then breadth of seed
    # linkage, then citation VELOCITY (not raw count -- see citation_velocity
    # for why), then newest first. Fully deterministic; no model involved.
    rows.sort(key=lambda r: (-r["linked_to_framework_core"], -r["n_seed_links"],
                             -r["citations_per_year"], -r["year"]))
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
    p.add_argument("--max-backward-per-seed", type=int, default=250,
                   help="Cap on references pulled per seed; surveys cite hundreds. 0 = uncapped.")
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
        "ref", "role", "expand", "status", "similarity", "year", "matched_year",
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
    directions: dict = defaultdict(set)

    def merge(works, wanted, direction):
        all_works.update(works)
        for k, v in wanted.items():
            links[k] |= v
        for k, v in direction.items():
            directions[k] |= v

    if not args.skip_backward:
        if verbose:
            print("\n[2/4] Backward expansion (works the seeds cite) ...", flush=True)
        merge(*expand_backward(api, resolved, args.max_backward_per_seed, verbose))

    if not args.skip_forward:
        if verbose:
            print("\n[3/4] Forward expansion (works that cite the seeds) ...", flush=True)
        merge(*expand_forward(api, resolved,
                              args.max_forward_per_seed or None,
                              args.min_year, verbose))

    # --- 3. assemble ------------------------------------------------------
    if verbose:
        print("\n[4/4] Deduplicating, filtering, scoring ...", flush=True)
    candidates = build_candidates(all_works, links, directions,
                                  seed_roles, seed_ids, args.min_year)

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
        print(f"\n  -> {cand_path}")
        print(f"  -> {os.path.join(args.out_dir, 'seeds_resolved.csv')}  (verify by hand)")
        print(f"  -> {os.path.join(args.out_dir, 'provenance.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
