#!/usr/bin/env python3
"""
abstract_backfill.py -- fill the OpenAlex abstract gap from Semantic Scholar.

Why this exists
---------------
OpenAlex removed Elsevier-sourced `abstract_inverted_index` values in 2022 under
publisher pressure. The gap is therefore not random: it tracks a publisher, and
Elsevier publishes a large share of the manufacturing-systems venues this review
cares about (JMSy, CIRP, Procedia CIRP, Robotics and CIM). Screening on title
alone over a publisher-correlated subset would push exactly one slice of the
literature toward `uncertain`, so the gap is closed before the calibration
sample is drawn rather than after.

Screening input is title + abstract only (no PDFs at this stage), which makes
the abstract the entire evidence base for a decision. A row with no abstract is
therefore not a row that screens badly -- it is a row that cannot be screened on
the stated inputs at all. Those rows are excluded from the CALIBRATION sample,
because a kappa computed partly over title-only decisions measures two different
tasks at once. They are NOT excluded from the corpus: in production they are
screened on title and forced to `uncertain`, which routes them to a human.

What it adds
------------
An `abstract_source` column on candidates.csv, one of:

    openalex          -- abstract came back on the OpenAlex work record
    semantic_scholar  -- backfilled here, from the S2 Graph API by DOI
    none              -- still empty after this run

`none` is deliberately not subdivided in the CSV, because downstream code should
treat every unscreenable row identically. The REASON each row is `none` is a
different question -- "S2 has no record of this DOI" and "S2 timed out" have the
same effect today but different fixes -- so it is recorded per row in the call
log, and the split is summarised into provenance.json. A backfill left at 40%
`none` because the rate limiter was angry should look different in the record
from one left at 40% because the papers genuinely are not indexed.

Idempotent. Every API response is cached to the log file, and a rerun replays
the cache instead of re-requesting, so the 1 req/sec walk is paid once. Use
--refresh-errors to retry only the rows that failed transiently.

Usage
-----
    python abstract_backfill.py --candidates corpus/candidates.csv

    # retry rows that errored last time, leave confirmed misses alone
    python abstract_backfill.py --refresh-errors

    # an S2 API key lifts the rate limit; without one, 1 req/sec
    python abstract_backfill.py --api-key $S2_API_KEY --rate 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

S2_API = "https://api.semanticscholar.org/graph/v1"
SCRIPT_VERSION = "1.0.0"

# Outcome codes written to the call log. The first two are successes; the rest
# all land as abstract_source=none but are kept apart so the failure mode is
# legible in provenance.
OK = "ok"                        # S2 returned a non-empty abstract
NO_ABSTRACT = "no_abstract"      # S2 knows the paper, has no abstract for it
NOT_FOUND = "not_found"          # S2 has no record of this DOI (404)
NO_DOI = "no_doi"                # nothing to look up with
HTTP_ERROR = "http_error"        # 4xx/5xx that is not a 404
NETWORK_ERROR = "network_error"  # timeout, connection reset, etc.

TRANSIENT = {HTTP_ERROR, NETWORK_ERROR}

csv.field_size_limit(10_000_000)


class SemanticScholar:
    """Minimal S2 Graph API client with polite rate limiting and 429 backoff."""

    def __init__(self, rate: float, api_key: str | None, timeout: float = 30.0):
        self.rate = rate
        self.timeout = timeout
        self.session = requests.Session()
        headers = {"User-Agent": f"rob590-abstract-backfill/{SCRIPT_VERSION}"}
        if api_key:
            headers["x-api-key"] = api_key
        self.session.headers.update(headers)
        self._last = 0.0
        self.requests_made = 0

    def _wait(self) -> None:
        gap = self.rate - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def abstract_by_doi(self, doi: str, max_retries: int = 3) -> tuple[str, str, int | None]:
        """
        Return (outcome, abstract, http_status).

        A 429 is retried with exponential backoff and does NOT consume an
        outcome: S2's unauthenticated pool is shared, so a burst of 429s says
        something about other clients, not about this DOI.
        """
        url = f"{S2_API}/paper/DOI:{doi}"
        delay = 5.0
        for attempt in range(max_retries + 1):
            self._wait()
            try:
                r = self.session.get(url, params={"fields": "abstract"},
                                     timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return NETWORK_ERROR, f"{type(exc).__name__}", None
            self.requests_made += 1

            if r.status_code == 429:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return HTTP_ERROR, "429 rate limited after retries", 429
            if r.status_code == 404:
                return NOT_FOUND, "", 404
            if r.status_code >= 400:
                return HTTP_ERROR, f"{r.status_code}", r.status_code
            try:
                payload = r.json()
            except ValueError:
                return HTTP_ERROR, "non-JSON response", r.status_code
            abstract = (payload or {}).get("abstract") or ""
            abstract = " ".join(abstract.split())
            if abstract:
                return OK, abstract, r.status_code
            return NO_ABSTRACT, "", r.status_code
        return NETWORK_ERROR, "retries exhausted", None


def load_cache(path: str) -> dict[str, dict]:
    """Replay the call log so a rerun costs nothing for rows already resolved."""
    cache: dict[str, dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("doi"):
                cache[rec["doi"].lower()] = rec
    return cache


def update_provenance(path: str, block: dict) -> None:
    """
    Merge the audit block into provenance.json under `abstract_backfill`.

    corpus_build.py rewrites provenance.json wholesale on every run, so this
    block is written AFTER the build and will be lost by the next one. That is
    the correct coupling -- a backfill describes one specific candidates.csv --
    but it means the build must be rerun before the backfill, never after.
    """
    prov = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prov = json.load(f)
    prov["abstract_backfill"] = block
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill missing abstracts from Semantic Scholar by DOI.")
    ap.add_argument("--candidates", default="corpus/candidates.csv")
    ap.add_argument("--provenance", default="corpus/provenance.json")
    ap.add_argument("--log", default="corpus/abstract_backfill_calls.jsonl",
                    help="append-only call log; doubles as the response cache")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="seconds between requests (default 1.0 = the "
                         "unauthenticated S2 limit)")
    ap.add_argument("--api-key", default=os.environ.get("S2_API_KEY"),
                    help="S2 API key; lifts the rate limit")
    ap.add_argument("--refresh-errors", action="store_true",
                    help="retry cached rows whose outcome was transient "
                         "(http_error/network_error); leave confirmed "
                         "not_found/no_abstract alone")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N live requests (debugging)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet

    with open(args.candidates, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        sys.exit(f"{args.candidates} has no rows")

    # --- before ------------------------------------------------------------
    def has_abstract(r: dict) -> bool:
        return bool((r.get("abstract") or "").strip())

    before_with = sum(1 for r in rows if has_abstract(r))
    before_without = len(rows) - before_with
    if verbose:
        print(f"[audit] {len(rows)} candidates")
        print(f"[audit] abstract present : {before_with} "
              f"({100 * before_with / len(rows):.1f}%)")
        print(f"[audit] abstract MISSING : {before_without} "
              f"({100 * before_without / len(rows):.1f}%)")

    targets = [r for r in rows if not has_abstract(r)]
    cache = load_cache(args.log)
    if verbose and cache:
        print(f"[cache] {len(cache)} prior responses in {args.log}")

    # The CALL LOG is the authority for `abstract_source`, not the CSV column
    # and not "does this row have an abstract". Deriving the source from the
    # presence of an abstract is right exactly once -- on the first pass, where
    # a present abstract can only have come from OpenAlex. On any later pass the
    # rows this script already filled have abstracts too, and re-deriving would
    # relabel every semantic_scholar row as openalex, destroying the provenance
    # the column exists to carry. Trusting the CSV's prior value instead would
    # fix the rerun but not repair a column already corrupted that way.
    #
    # The log is append-only and records which DOIs S2 actually answered, so
    # deriving from it is both correct on a rerun and self-healing on a file
    # that was mislabelled earlier. --refresh-errors is the normal way to finish
    # a rate-limited run, so this rerun is expected, not exotic.
    s2_filled = {doi for doi, rec in cache.items() if rec.get("outcome") == OK}

    outcomes: Counter[str] = Counter()
    from_cache = 0
    live = 0
    api = SemanticScholar(rate=args.rate, api_key=args.api_key)
    run_started = datetime.now(timezone.utc)

    log = open(args.log, "a", encoding="utf-8")
    try:
        for i, row in enumerate(targets, 1):
            doi = (row.get("doi") or "").strip().lower()
            if not doi:
                outcomes[NO_DOI] += 1
                log.write(json.dumps({
                    "openalex_id": row.get("openalex_id"), "doi": "",
                    "outcome": NO_DOI, "detail": "no DOI on the OpenAlex record",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                continue

            rec = cache.get(doi)
            replayable = rec is not None and not (
                args.refresh_errors and rec.get("outcome") in TRANSIENT)

            if replayable:
                outcome = rec["outcome"]
                abstract = rec.get("abstract") or ""
                from_cache += 1
            else:
                if args.limit and live >= args.limit:
                    outcomes["skipped_limit"] += 1
                    continue
                outcome, detail, status = api.abstract_by_doi(doi)
                abstract = detail if outcome == OK else ""
                live += 1
                entry = {
                    "openalex_id": row.get("openalex_id"), "doi": doi,
                    "outcome": outcome, "http_status": status,
                    "abstract": abstract,
                    "detail": "" if outcome == OK else detail,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                cache[doi] = entry

            outcomes[outcome] += 1
            if outcome == OK and abstract:
                row["abstract"] = abstract
                row["abstract_source"] = "semantic_scholar"

            if verbose and i % 50 == 0:
                print(f"  [s2] {i}/{len(targets)}  live={live} cached={from_cache} "
                      f"filled={outcomes[OK]}", flush=True)
    finally:
        log.close()

    # --- after -------------------------------------------------------------
    for r in rows:
        if not has_abstract(r):
            r["abstract_source"] = "none"
        elif (r.get("doi") or "").strip().lower() in s2_filled:
            r["abstract_source"] = "semantic_scholar"
        else:
            r["abstract_source"] = "openalex"

    after_with = sum(1 for r in rows if has_abstract(r))
    by_source = Counter(r["abstract_source"] for r in rows)
    still_missing = by_source["none"]

    if "abstract_source" not in fieldnames:
        fieldnames.insert(fieldnames.index("abstract") + 1, "abstract_source")

    tmp = args.candidates + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.candidates)

    # Transient failures are separated out because they are the only bucket a
    # rerun can fix. A reviewer reading provenance should be able to tell
    # "these papers are not indexed" from "the network had a bad afternoon".
    recoverable = sum(outcomes[o] for o in TRANSIENT)

    block = {
        "script": "abstract_backfill.py",
        "script_version": SCRIPT_VERSION,
        "run_started_utc": run_started.isoformat(),
        "run_seconds": round((datetime.now(timezone.utc) - run_started).total_seconds(), 1),
        "source_api": "Semantic Scholar Graph API /paper/DOI:{doi}?fields=abstract",
        "rate_limit_seconds_per_request": args.rate,
        "api_key_used": bool(args.api_key),
        "api_requests": api.requests_made,
        "responses_replayed_from_cache": from_cache,
        "candidates_total": len(rows),
        # before/after describe the BACKFILL, not this invocation. They are
        # derived from abstract_source -- which is itself derived from the call
        # log -- so a rerun reports the same figures instead of reporting "0
        # missing before, 0 after" once the work is already done. The delta of
        # the current invocation is `filled_this_invocation`.
        "before": {
            "with_abstract": by_source["openalex"],
            "without_abstract": len(rows) - by_source["openalex"],
            "pct_missing": round(
                100 * (len(rows) - by_source["openalex"]) / len(rows), 1),
            "basis": "rows OpenAlex supplied an abstract for",
        },
        "after": {
            "with_abstract": by_source["openalex"] + by_source["semantic_scholar"],
            "without_abstract": still_missing,
            "pct_missing": round(100 * still_missing / len(rows), 1),
            "basis": "after Semantic Scholar backfill by DOI",
        },
        "filled_by_backfill": by_source["semantic_scholar"],
        "filled_this_invocation": after_with - before_with,
        "abstract_source_counts": dict(by_source),
        "abstract_source_basis": (
            "Derived from the append-only call log, not from the CSV column "
            "and not from abstract presence: a rerun must not relabel rows S2 "
            "filled as openalex."),
        "outcome_counts": dict(outcomes),
        "unrecovered_transient_failures": recoverable,
        "note": (
            "abstract_source=none rows are excluded from the calibration sample "
            "(a kappa mixing title-only and title+abstract decisions measures two "
            "tasks) and are forced to `uncertain` in production screening. "
            "unrecovered_transient_failures counts rows that a rerun with "
            "--refresh-errors could still fix; if it is non-zero the `none` "
            "count is an upper bound, not a finding."
        ),
    }
    update_provenance(args.provenance, block)

    if verbose:
        print()
        print(f"[done] {api.requests_made} live requests, {from_cache} replayed")
        print(f"[done] filled {after_with - before_with} abstracts")
        print(f"[done] abstract_source: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_source.items())))
        print(f"[done] outcomes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(outcomes.items())))
        print(f"[done] missing {block['before']['without_abstract']} "
              f"({block['before']['pct_missing']}%) "
              f"-> {still_missing} ({block['after']['pct_missing']}%)")
        if recoverable:
            print(f"  *** {recoverable} transient failure(s) unrecovered; "
                  f"rerun with --refresh-errors before trusting the `none` count")
        print(f"  -> {args.candidates}  (abstract_source column added)")
        print(f"  -> {args.provenance}  (abstract_backfill block)")
        print(f"  -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
