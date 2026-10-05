#!/usr/bin/env python3
"""
fulltext_head.py -- recover a screenable text head for rows with no abstract.

Why
---
After the Semantic Scholar backfill, 431 of 1,941 candidates (22.2%) still had
no abstract. Those rows cannot be screened on the stated inputs at all: the
abstract IS the evidence base. Left alone an ELIGIBLE one is forced to
`uncertain` and lands on a human desk, so the size of that pile is the size of
the manual screening burden.

Scope: eligible rows only (v2.0.0)
----------------------------------
Since the 2026-10-05 scope change, only rows eligibility.py marked
`eligible=True` are worked on. An ineligible row is out of scope, so recovering
text for it would buy nothing; its existing columns are left exactly as they
were.

This script recovers the FIRST 5,000 CHARACTERS of the full text where it can
be obtained legitimately, giving those rows something to screen on. It is not
an abstract and is not stored as one.

Sources, in order, stopping at the first success
------------------------------------------------
1. Open-access PDF: `best_oa_location.pdf_url` as OpenAlex already recorded it
   in candidates.csv, then Unpaywall by DOI.
2. Nothing else. No publisher HTML scraping, no shadow libraries, and -- since
   v2.0.0 -- no Elsevier Article Retrieval API: 10.1016 DOIs are out of scope
   under the access rule, so that path and its ELSEVIER_API_KEY handling are
   gone. An eligible row no OA source covers keeps abstract_source=none and is
   screened on title alone, forced to `uncertain`.

Extraction is deterministic
---------------------------
PyMuPDF for PDFs. Truncation is to the first 5,000
CHARACTERS -- a character count, not a token estimate, and no model is involved
at any point. Roughly 1,200 tokens.

Cleaning is deliberately timid. Running headers and footers are removed by a
rule that can be stated in one line -- a line that repeats on three or more
pages is furniture, not content -- and bare page numbers go with them.
Affiliation lines are removed only in the first 40 lines and only when a line
both names an institution word and carries an email or a leading affiliation
marker. Anything more aggressive risks eating the first paragraph of the
introduction, which is the part actually worth screening. What is left may
still contain title-page matter; that is the intended trade.

Outputs
-------
`fulltext_head` -- a new column holding the extracted text. The `abstract`
column is NOT touched: an abstract and a truncated full-text head are different
kinds of evidence, and conflating them would hide that difference from the
screening prompt and from the calibration analysis.

`abstract_source` becomes `fulltext_head_oa`.

Downloaded files are cached under corpus/fulltext/ (gitignored -- the PDFs are
not ours to redistribute). Every attempt is appended to a JSONL log with the
DOI, source, HTTP status and characters extracted, as with the S2 backfill.

Usage
-----
    python fulltext_head.py --mailto zchosed@umich.edu
    python fulltext_head.py --refresh-errors     # retry transient failures only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

from eligibility import parse_eligible

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

SCRIPT_VERSION = "2.0.0"

UNPAYWALL = "https://api.unpaywall.org/v2"

HEAD_CHARS = 5000
MIN_USEFUL_CHARS = 200     # below this the extraction failed in substance

# Outcome codes. `ok_*` are successes; the rest all leave abstract_source=none
# but are kept apart so a rerun can target only what a rerun can fix.
OK_OA = "ok_oa"
NO_SOURCE = "no_source"             # no OA pdf, no Unpaywall hit
NO_DOI = "no_doi"
NOT_PDF = "not_pdf"                 # URL returned HTML/landing page, not a PDF
EXTRACT_EMPTY = "extract_empty"     # PDF parsed but yielded almost no text
FORBIDDEN = "forbidden"             # 403 from an OA host: paywall or bot block
HTTP_ERROR = "http_error"           # other 4xx/5xx, plausibly transient
NETWORK_ERROR = "network_error"

TRANSIENT = {HTTP_ERROR, NETWORK_ERROR}
SUCCESSES = {OK_OA}

SOURCE_OF_OUTCOME = {OK_OA: "fulltext_head_oa"}

csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, mailto: str, rate: float, timeout: float = 60.0):
        self.mailto = mailto
        self.rate = rate
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"rob590-fulltext-head/{SCRIPT_VERSION} "
                          f"(mailto:{mailto})"})
        self._last = 0.0
        self.requests_made = 0

    def _wait(self) -> None:
        gap = self.rate - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def get(self, url: str, **kw):
        self._wait()
        self.requests_made += 1
        return self.session.get(url, timeout=self.timeout, **kw)

    def unpaywall_pdf_url(self, doi: str) -> tuple[str, int | None]:
        """Best OA pdf URL Unpaywall knows for this DOI, or ('', status)."""
        try:
            r = self.get(f"{UNPAYWALL}/{doi}", params={"email": self.mailto})
        except requests.RequestException:
            return "", None
        if r.status_code != 200:
            return "", r.status_code
        try:
            data = r.json()
        except ValueError:
            return "", r.status_code
        best = (data or {}).get("best_oa_location") or {}
        url = best.get("url_for_pdf") or ""
        if not url:
            for loc in (data or {}).get("oa_locations") or []:
                if loc.get("url_for_pdf"):
                    url = loc["url_for_pdf"]
                    break
        return url, r.status_code


def download_pdf(fetcher: Fetcher, url: str, path: str) -> tuple[str, int | None]:
    """
    Fetch a URL and keep it only if it really is a PDF.

    Many `pdf_url` values in the wild are landing pages. Publisher HTML is
    explicitly out of scope, so a non-PDF response is recorded and abandoned
    rather than parsed.
    """
    try:
        r = fetcher.get(url, allow_redirects=True)
    except requests.RequestException as exc:
        return f"{NETWORK_ERROR}:{type(exc).__name__}", None
    if r.status_code == 403:
        # A 403 from an open-access host is a paywall or a bot block, not a
        # blip. Kept apart from http_error so it is not counted as something a
        # rerun could fix -- measured over 2-4 attempts each, none recovered.
        return FORBIDDEN, r.status_code
    if r.status_code != 200:
        return HTTP_ERROR, r.status_code
    ctype = (r.headers.get("content-type") or "").lower()
    body = r.content
    if not body.startswith(b"%PDF") and "pdf" not in ctype:
        return NOT_PDF, r.status_code
    with open(path, "wb") as f:
        f.write(body)
    return "", r.status_code



# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

PAGE_NUM = re.compile(r"^\s*(page\s+)?[ivxlcdm\d]{1,6}\s*(/\s*\d+)?\s*$", re.I)
INSTITUTION = re.compile(
    r"\b(univ|universit|department|dept\.|institute|instituto|laborator|"
    r"academy|faculty|school of|college|gmbh|inc\.|ltd\.)", re.I)
EMAILISH = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
AFFIL_MARKER = re.compile(r"^\s*[\d\*†‡§¶a-d]\s*[\.,)]?\s*[A-Z]")
DEHYPHEN = re.compile(r"(\w)-\n(\w)")


def _strip_running_furniture(pages: list[list[str]]) -> list[list[str]]:
    """
    Drop lines that repeat across three or more pages.

    That one rule catches running headers, running footers and journal
    watermarks without needing to know which is which, and it cannot fire on a
    sentence of body text, which does not recur verbatim on three pages.
    """
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for lines in pages:
        for ln in set(x.strip() for x in lines if x.strip()):
            counts[ln] += 1
    furniture = {ln for ln, n in counts.items() if n >= 3 and len(ln) < 200}
    return [[ln for ln in lines if ln.strip() not in furniture] for lines in pages]


def _strip_affiliations(lines: list[str], window: int = 40) -> list[str]:
    """
    Remove affiliation lines, conservatively, and only near the top.

    A line must BOTH look institutional AND carry an email or an affiliation
    marker. Requiring both is what keeps this from eating a sentence in the
    introduction that happens to mention a university.
    """
    out = []
    for i, ln in enumerate(lines):
        if i < window and INSTITUTION.search(ln) and (
                EMAILISH.search(ln) or AFFIL_MARKER.match(ln)):
            continue
        out.append(ln)
    return out


def extract_pdf_head(path: str, head_chars: int,
                     max_pages: int = 12) -> str:
    """Deterministic text extraction. No model, no heuristic beyond the above."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required:  pip install PyMuPDF")
    pages: list[list[str]] = []
    with fitz.open(path) as doc:
        for pno in range(min(len(doc), max_pages)):
            text = doc[pno].get_text("text") or ""
            text = DEHYPHEN.sub(r"\1\2", text)
            pages.append(text.splitlines())

    pages = _strip_running_furniture(pages)
    lines: list[str] = []
    for p in pages:
        lines.extend(p)
    lines = [ln for ln in lines if not PAGE_NUM.match(ln)]
    lines = _strip_affiliations(lines)

    text = "\n".join(ln.strip() for ln in lines if ln.strip())
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text[:head_chars].strip()



# ---------------------------------------------------------------------------

def safe_name(row: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", row.get("openalex_id") or "unknown")


def load_cache(path: str) -> dict[str, dict]:
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
            if rec.get("openalex_id"):
                cache[rec["openalex_id"]] = rec
    return cache


def update_provenance(path: str, block: dict) -> None:
    prov = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prov = json.load(f)
    prov["fulltext_head"] = block
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recover a truncated full-text head for abstract-less rows.")
    ap.add_argument("--candidates", default="corpus/candidates.csv")
    ap.add_argument("--provenance", default="corpus/provenance.json")
    ap.add_argument("--cache-dir", default="corpus/fulltext")
    ap.add_argument("--log", default="corpus/fulltext_head_calls.jsonl")
    ap.add_argument("--mailto", required=True, help="for the Unpaywall polite pool")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--head-chars", type=int, default=HEAD_CHARS)
    ap.add_argument("--refresh-errors", action="store_true",
                    help="retry rows whose last outcome was transient; leave "
                         "confirmed misses alone")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    if fitz is None:
        sys.exit("PyMuPDF is required:  pip install PyMuPDF")

    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.candidates, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "abstract_source" not in fieldnames:
        sys.exit(f"{args.candidates} has no abstract_source column -- run "
                 f"abstract_backfill.py first")
    if "eligible" not in fieldnames:
        sys.exit(f"{args.candidates} has no eligible column -- run "
                 f"eligibility.py first")

    for r in rows:
        r.setdefault("fulltext_head", "")

    eligible = [r for r in rows if parse_eligible(r.get("eligible", ""))]
    # Eligible rows lacking an abstract. Rows already carrying a head are
    # re-derived from the cache below rather than re-fetched, so a rerun is
    # free and idempotent. Ineligible rows are not targets and are not touched.
    targets = [r for r in eligible
               if r.get("abstract_source") in ("none", "")
               or r.get("abstract_source", "").startswith("fulltext_head")]
    still_none = [r for r in eligible if r.get("abstract_source") in ("none", "")]

    before_none = len(still_none)
    if verbose:
        print(f"[fulltext] {len(rows)} candidates, {len(eligible)} eligible, "
              f"{before_none} eligible with no abstract to work on")

    cache = load_cache(args.log)
    fetcher = Fetcher(args.mailto, args.rate)
    outcomes: Counter[str] = Counter()
    from_cache = 0
    live = 0
    run_started = datetime.now(timezone.utc)

    log = open(args.log, "a", encoding="utf-8")
    try:
        for i, row in enumerate(targets, 1):
            wid = row.get("openalex_id") or ""
            doi = (row.get("doi") or "").strip()
            rec = cache.get(wid)
            # Outcomes carry a detail suffix (`network_error:ConnectionError`),
            # so match on the prefix -- comparing the whole string silently
            # excluded every network error from --refresh-errors, which is the
            # one class of failure a rerun exists to fix.
            prior = (rec or {}).get("outcome", "").split(":")[0]
            replayable = rec is not None and not (
                args.refresh_errors and prior in TRANSIENT)

            if replayable:
                outcome = rec["outcome"]
                head = rec.get("head") or ""
                from_cache += 1
            else:
                if args.limit and live >= args.limit:
                    outcomes["skipped_limit"] += 1
                    continue
                live += 1
                outcome, head, status, url_used = "", "", None, ""

                # --- 1. open-access PDF ---------------------------------
                pdf_path = os.path.join(args.cache_dir, safe_name(row) + ".pdf")
                candidates_urls = []
                if (row.get("oa_pdf_url") or "").strip():
                    candidates_urls.append(("openalex", row["oa_pdf_url"].strip()))
                if doi:
                    up_url, up_status = fetcher.unpaywall_pdf_url(doi)
                    if up_url:
                        candidates_urls.append(("unpaywall", up_url))

                for origin, url in candidates_urls:
                    if os.path.exists(pdf_path):
                        err, status = "", 200
                    else:
                        err, status = download_pdf(fetcher, url, pdf_path)
                    if err:
                        outcome = err
                        continue
                    try:
                        head = extract_pdf_head(pdf_path, args.head_chars)
                    except Exception as exc:              # noqa: BLE001
                        outcome = f"{EXTRACT_EMPTY}:{type(exc).__name__}"
                        head = ""
                    if len(head) >= MIN_USEFUL_CHARS:
                        outcome, url_used = OK_OA, f"{origin}:{url}"
                        break
                    outcome = EXTRACT_EMPTY

                if not outcome:
                    outcome = NO_DOI if not doi else NO_SOURCE

                entry = {
                    "openalex_id": wid, "doi": doi, "outcome": outcome,
                    "source_url": url_used, "http_status": status,
                    "chars_extracted": len(head),
                    "head": head if outcome in SUCCESSES else "",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                cache[wid] = entry

            outcomes[outcome] += 1
            if outcome in SUCCESSES and head:
                row["fulltext_head"] = head
                row["abstract_source"] = SOURCE_OF_OUTCOME[outcome]

            if verbose and i % 25 == 0:
                got = outcomes[OK_OA]
                print(f"  [ft] {i}/{len(targets)}  live={live} cached={from_cache} "
                      f"recovered={got}", flush=True)
    finally:
        log.close()

    # --- source is derived from the append-only log, as in the S2 backfill ---
    by_outcome = {wid: rec["outcome"] for wid, rec in cache.items()}
    for r in targets:
        oc = by_outcome.get(r.get("openalex_id"))
        if oc in SUCCESSES and (r.get("fulltext_head") or "").strip():
            r["abstract_source"] = SOURCE_OF_OUTCOME[oc]
        elif r.get("abstract_source", "").startswith("fulltext_head") \
                and not (r.get("fulltext_head") or "").strip():
            r["abstract_source"] = "none"

    if "fulltext_head" not in fieldnames:
        fieldnames.insert(fieldnames.index("abstract_source") + 1, "fulltext_head")

    tmp = args.candidates + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.candidates)

    by_source = Counter(r["abstract_source"] for r in eligible)
    recoverable = sum(n for o, n in outcomes.items()
                      if o.split(":")[0] in TRANSIENT)
    n_oa = by_source.get("fulltext_head_oa", 0)

    block = {
        "script": "fulltext_head.py",
        "script_version": SCRIPT_VERSION,
        "run_started_utc": run_started.isoformat(),
        "run_seconds": round(
            (datetime.now(timezone.utc) - run_started).total_seconds(), 1),
        "head_chars": args.head_chars,
        "truncation": "first N CHARACTERS of the extracted text; a character "
                      "count, not a token estimate. No model is involved.",
        "extractors": {"pdf": "PyMuPDF (fitz) get_text('text')"},
        "cleaning": ("lines repeating on >=3 pages dropped as running "
                     "headers/footers; bare page numbers dropped; affiliation "
                     "lines dropped only in the first 40 lines and only when a "
                     "line both names an institution and carries an email or an "
                     "affiliation marker. Deliberately timid -- title-page "
                     "matter may remain."),
        "sources_tried_in_order": [
            "OpenAlex best_oa_location.pdf_url (from candidates.csv)",
            "Unpaywall by DOI",
        ],
        "elsevier_path": "removed in v2.0.0 with the 2026-10-05 access rule; "
                         "10.1016 DOIs are ineligible and never targeted",
        "publisher_html_scraped": False,
        "shadow_libraries_used": False,
        "scope": "eligible rows (eligibility.py) lacking an abstract; "
                 "ineligible rows are not targeted and not modified",
        "api_requests": fetcher.requests_made,
        "responses_replayed_from_cache": from_cache,
        "candidates_total": len(rows),
        "eligible_total": len(eligible),
        "targets": len(targets),
        "before": {"no_text": before_none,
                   "pct_of_eligible": round(100 * before_none / max(len(eligible), 1), 1),
                   "basis": "eligible rows with abstract_source=none at the "
                            "start of this run"},
        "after": {"no_text": by_source.get("none", 0),
                  "pct_of_eligible": round(
                      100 * by_source.get("none", 0) / max(len(eligible), 1), 1),
                  "basis": "eligible rows still with no abstract and no "
                           "full-text head"},
        "recovered_by_source": {"fulltext_head_oa": n_oa, "total": n_oa},
        "abstract_source_counts_eligible": dict(by_source),
        "outcome_counts": dict(outcomes),
        "unrecovered_transient_failures": recoverable,
        "unrecoverable_forbidden": outcomes.get(FORBIDDEN, 0),
        "note": (
            "A full-text head is NOT an abstract: it is the opening of the "
            "paper, truncated by character count, and it is stored in its own "
            "column so the screening prompt can label it as such. Because it is "
            "a different input distribution, these rows are calibrated on their "
            "own 20-paper sample rather than mixed into the 50-paper abstract "
            "sample. Eligible rows still at abstract_source=none are screened "
            "on title alone and forced to `uncertain`; ineligible rows are not "
            "screened at all."),
    }
    update_provenance(args.provenance, block)

    if verbose:
        print()
        print(f"[done] {fetcher.requests_made} live requests, {from_cache} replayed")
        print(f"[done] eligible rows carrying an OA head: {n_oa}")
        print(f"[done] abstract_source (eligible rows): " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_source.items())))
        print(f"[done] outcomes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(outcomes.items())))
        print(f"[done] eligible with no text: {before_none} "
              f"({block['before']['pct_of_eligible']}%) -> "
              f"{by_source.get('none', 0)} ({block['after']['pct_of_eligible']}%)")
        if outcomes.get(FORBIDDEN):
            print(f"  *** {outcomes[FORBIDDEN]} host(s) returned 403 -- paywalled "
                  f"or bot-blocked. Not recoverable by rerunning.")
        if recoverable:
            print(f"  *** {recoverable} transient failure(s); rerun with "
                  f"--refresh-errors before trusting the `none` count")
        print(f"  -> {args.candidates}  (fulltext_head column)")
        print(f"  -> {args.provenance}  (fulltext_head block)")
        print(f"  -> {args.log}")
        print(f"  -> {args.cache_dir}/  (gitignored; PDFs are not committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
