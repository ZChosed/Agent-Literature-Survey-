#!/usr/bin/env python3
"""
eligibility.py -- mark which candidates are in scope under the 10/05 access rule.

Why
---
Scope change from the 2026-10-05 advisor meeting: the review is restricted to
papers the project can actually read. This is a SCOPE decision, made before any
model sees a paper, so it is made here, in Python, from measured inputs -- and
it is recorded per row rather than applied by deleting rows. An out-of-scope
paper keeps a row in corpus/eligibility.csv with the reason it is out, so
"why isn't Smith 2023 in the review?" always has an answer on file.

The rule (v2.0.0)
-----------------
    no OA copy: OpenAlex open_access.is_oa is
    false AND Unpaywall has no OA location        -> ineligible, `no_oa`
    otherwise                                     -> eligible,   `none`

That is the only rule. v1 also had a rule 1 -- DOI prefix 10.1016 ->
ineligible `elsevier`, even when OA -- which was removed on 2026-10-05
(eligibility v2.0.0). Publisher plays no part in the
decision now; a 10.1016 paper is in scope exactly when it is open access. The
v1 decisions and provenance are archived under
corpus/archive/2026-10-05_eligibility_v1/, and the effect of the removal is
measured in provenance (`doi_10_1016_rows`).

Rows with no DOI cannot be looked up in Unpaywall, so for them the rule is
decided on OpenAlex is_oa alone. The path taken is recorded in
`eligibility_rule`:

    openalex_is_oa                    eligible: OpenAlex says OA
    unpaywall_oa_location             eligible: OpenAlex said not OA, Unpaywall found a location
    no_oa:unpaywall_no_location       ineligible: Unpaywall knows the DOI, lists no OA location
    no_oa:unpaywall_doi_unknown       ineligible: Unpaywall 404 for this DOI
    no_oa:no_doi_openalex_only        ineligible: no DOI, OpenAlex says not OA
    openalex_is_oa_no_doi             eligible: no DOI, OpenAlex says OA
    unpaywall_oa_location:openalex_record_missing
                                      eligible: OpenAlex no longer has the
                                      work, Unpaywall found an OA location

The last one exists because OpenAlex deletes records. Rule 2 is a conjunction,
so an OA location in Unpaywall defeats it whatever OpenAlex would have said; a
missing OpenAlex record therefore still decides IF Unpaywall finds a location.
If Unpaywall finds none, the row is undecided and the run aborts -- "OpenAlex
has no record" is not evidence that is_oa is false.

The measured inputs
-------------------
candidates.csv carries `oa_pdf_url` but NOT is_oa, and an empty pdf URL is not
the same claim as is_oa=false (OpenAlex can know a work is OA without holding a
PDF link). So is_oa is fetched fresh from OpenAlex, batched 50 IDs a request,
and Unpaywall is queried by DOI only for rows the first two checks have not
already decided -- a 10.1016 row is never sent to Unpaywall, because nothing it
could return would change the outcome.

Every response is appended to corpus/eligibility_calls.jsonl, which doubles as
the cache: a rerun replays it and requests nothing already answered. `--offline`
replays only and aborts on a cache miss, which is how the decision is
reproduced without the network.

All or nothing
--------------
If any row cannot be decided -- a network error, a 5xx, an OpenAlex ID that no
longer resolves -- NOTHING is written. A half-filled `eligible` column would be
read as a complete one. Rerun with --refresh-errors.

Overrides are judgments, and are labelled as such
-------------------------------------------------
A row that NEITHER source can answer for (OpenAlex deleted the record and
Unpaywall does not index the DOI) is not a transient failure, and no rerun will
fix it. Such a row is decided by a human, in corpus/eligibility_overrides.csv:

    openalex_id, eligible, exclusion_reason, reason, decided_by, decided_on

The rule for that row becomes `manual_override` and provenance lists every
override with its stated reason. An override is applied ONLY to a row the rules
cannot decide: one naming a row the rules can decide aborts the run, so an
override can never quietly replace a rule.

Outputs
-------
corpus/eligibility.csv -- one row per candidate, in candidates.csv order,
keyed by openalex_id: `eligible`, `exclusion_reason`, `eligibility_rule`,
`candidates_key_sha256`. candidates.csv itself is NOT modified (v1.1.0): a
corpus rebuild rewrites that file wholesale, and decisions stored in it would
silently vanish with it.

Staleness. Every consumer (screen.py, draw_calibration_sample.py,
fulltext_head.py) loads eligibility through `load_eligibility()`, which fails
loudly if the file is missing or if its `candidates_key_sha256` does not match
the current candidates.csv. The key is the SHA-256 of the ordered
(openalex_id, doi) pairs -- exactly the inputs the decision and the rank
strata depend on -- NOT of the whole file. A whole-file hash would go stale
every time fulltext_head.py writes its own column, which it must do AFTER
eligibility (it only targets eligible rows), so it would fail on every normal
run without detecting anything about eligibility. Any rebuild that adds, drops,
reorders or re-DOIs a row changes the key. The whole-file hash at decision
time is still recorded in provenance, for information.

An `eligibility` block in provenance.json carries every count the Report 1
limitation statement cites: by reason, by reason x rank stratum, the top 15
excluded venues, and the share of framework_core-linked candidates lost.

Usage
-----
    python eligibility.py --mailto zchosed@umich.edu
    python eligibility.py --mailto zchosed@umich.edu --refresh-errors
    python eligibility.py --offline          # replay the call log only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

SCRIPT_VERSION = "2.0.0"

OPENALEX = "https://api.openalex.org"
UNPAYWALL = "https://api.unpaywall.org/v2"
ELSEVIER_PREFIX = "10.1016"

OPENALEX_BATCH = 50
TOP_VENUES = 15

REASONS = ("no_oa", "none")
STRATA = ("top_third", "bottom_two_thirds")

OUT_COLUMNS = ["eligible", "exclusion_reason", "eligibility_rule"]
KEY_COLUMN = "candidates_key_sha256"
ELIGIBILITY_COLUMNS = ["openalex_id"] + OUT_COLUMNS + [KEY_COLUMN]
OVERRIDE_COLUMNS = ["openalex_id", "eligible", "exclusion_reason", "reason",
                    "decided_by", "decided_on"]

# Call-log outcomes. `ok` and `not_found` are answers; the rest are not, and a
# row resting on one of them cannot be decided.
OK = "ok"
NOT_FOUND = "not_found"
HTTP_ERROR = "http_error"
NETWORK_ERROR = "network_error"
TRANSIENT = {HTTP_ERROR, NETWORK_ERROR}

csv.field_size_limit(10_000_000)


class Undecided(Exception):
    """A row whose measured inputs are incomplete. Never guessed around."""


# ---------------------------------------------------------------------------
# The rule. Pure, so it can be tested without a network or a file.
# ---------------------------------------------------------------------------

def is_elsevier_doi(doi: str) -> bool:
    """Registrant prefix exactly 10.1016 -- not 10.10160, not a substring."""
    return (doi or "").strip().lower().split("/", 1)[0] == ELSEVIER_PREFIX


def classify(doi: str, openalex_is_oa: bool | None,
             unpaywall: dict | None) -> tuple[bool, str, str]:
    """
    Return (eligible, exclusion_reason, eligibility_rule).

    `openalex_is_oa` is None when OpenAlex has not answered for this row.
    `unpaywall` is the cached Unpaywall record for the DOI, or None when it was
    not queried. Missing inputs that the rule actually needs raise Undecided.
    The DOI prefix plays no part (rule 1 was removed in v2.0.0).
    """
    doi = (doi or "").strip()

    if openalex_is_oa is None:
        if (doi and unpaywall is not None and unpaywall.get("outcome") == OK
                and unpaywall.get("n_oa_locations", 0) > 0):
            return True, "none", "unpaywall_oa_location:openalex_record_missing"
        raise Undecided("no OpenAlex is_oa")
    if openalex_is_oa:
        return True, "none", "openalex_is_oa" if doi else "openalex_is_oa_no_doi"

    if not doi:
        return False, "no_oa", "no_oa:no_doi_openalex_only"

    if unpaywall is None or unpaywall.get("outcome") not in (OK, NOT_FOUND):
        raise Undecided("no Unpaywall answer")
    if unpaywall["outcome"] == NOT_FOUND:
        return False, "no_oa", "no_oa:unpaywall_doi_unknown"
    if unpaywall.get("n_oa_locations", 0) > 0:
        return True, "none", "unpaywall_oa_location"
    return False, "no_oa", "no_oa:unpaywall_no_location"


def candidates_key(rows: list[dict]) -> str:
    """SHA-256 of the ordered (openalex_id, doi) pairs. See the docstring."""
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['openalex_id']}\t{(r.get('doi') or '').strip().lower()}\n"
                 .encode("utf-8"))
    return h.hexdigest()


def write_eligibility_csv(path: str, rows: list[dict], key: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ELIGIBILITY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{c: r[c] for c in ["openalex_id"] + OUT_COLUMNS},
                        KEY_COLUMN: key})
    os.replace(tmp, path)


def load_eligibility(path: str, candidate_rows: list[dict]) -> dict[str, dict]:
    """
    openalex_id -> {eligible: bool, exclusion_reason, eligibility_rule}.

    The one way any script reads eligibility. Exits loudly -- never falls back,
    never treats a row as eligible by default -- if the file is missing, has
    the wrong columns, or was computed against a different candidates.csv.
    """
    if not os.path.exists(path):
        sys.exit(f"\n  *** {path} is missing. Eligibility is not decided for "
                 f"this corpus;\n      run eligibility.py (--offline replays "
                 f"the call log).\n")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ELIGIBILITY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path} is missing column(s) {missing}")
        erows = list(reader)
    want = candidates_key(candidate_rows)
    keys = {r[KEY_COLUMN] for r in erows}
    if keys != {want}:
        sys.exit(f"\n  *** STALE: {path} was computed against a different "
                 f"candidates.csv.\n      recorded key {sorted(keys)[0][:16]}... "
                 f"!= current {want[:16]}...\n      The corpus changed since "
                 f"eligibility was decided. Rerun eligibility.py.\n")
    out = {r["openalex_id"]: {"eligible": parse_eligible(r["eligible"]),
                              "exclusion_reason": r["exclusion_reason"],
                              "eligibility_rule": r["eligibility_rule"]}
           for r in erows}
    if len(out) != len(erows) or set(out) != {r["openalex_id"] for r in candidate_rows}:
        sys.exit(f"{path} does not hold exactly one row per candidate")
    return out


def attach_eligibility(path: str, candidate_rows: list[dict]) -> None:
    """Load, check, and copy the three fields onto the rows IN MEMORY only."""
    elig = load_eligibility(path, candidate_rows)
    for r in candidate_rows:
        e = elig[r["openalex_id"]]
        r["eligible"] = "True" if e["eligible"] else "False"
        r["exclusion_reason"] = e["exclusion_reason"]
        r["eligibility_rule"] = e["eligibility_rule"]


def parse_eligible(value: str) -> bool:
    """The one place the CSV string is read back. Anything else is an error."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"eligible must be 'True' or 'False', got {value!r} -- "
                     f"run eligibility.py")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, mailto: str, rate: float, timeout: float = 60.0):
        import requests  # only needed when actually fetching
        self.requests = requests
        self.mailto = mailto
        self.rate = rate
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"rob590-eligibility/{SCRIPT_VERSION} (mailto:{mailto})"})
        self._last = 0.0
        self.n_openalex = 0
        self.n_unpaywall = 0

    def _get(self, url: str, params: dict, retries: int = 4):
        backoff = 2.0
        for attempt in range(retries):
            gap = self.rate - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except self.requests.RequestException as exc:
                if attempt == retries - 1:
                    return None, f"{NETWORK_ERROR}:{type(exc).__name__}"
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return r, ""
        return None, f"{NETWORK_ERROR}:retries_exhausted"

    def openalex_batch(self, ids: list[str]) -> tuple[dict[str, dict], str]:
        """is_oa for up to 50 work IDs in one request. Returns ({id: rec}, err)."""
        self.n_openalex += 1
        r, err = self._get(f"{OPENALEX}/works", {
            "filter": "openalex:" + "|".join(ids),
            "select": "id,doi,open_access",
            "per-page": len(ids), "mailto": self.mailto})
        if r is None:
            return {}, err
        if r.status_code != 200:
            return {}, f"{HTTP_ERROR}:{r.status_code}"
        out = {}
        for w in r.json().get("results") or []:
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            out[wid] = w
        return out, ""

    def openalex_one(self, wid: str) -> tuple[dict | None, str]:
        """
        Single-work lookup for an ID the batch filter did not return. The
        /works/{id} route follows OpenAlex's merge redirects; the filter does
        not, so a merged work shows up here and not in the batch.
        """
        self.n_openalex += 1
        r, err = self._get(f"{OPENALEX}/works/{wid}", {
            "select": "id,doi,open_access", "mailto": self.mailto})
        if r is None:
            return None, err
        if r.status_code == 404:
            return None, NOT_FOUND
        if r.status_code != 200:
            return None, f"{HTTP_ERROR}:{r.status_code}"
        return r.json(), ""

    def unpaywall(self, doi: str) -> dict:
        self.n_unpaywall += 1
        r, err = self._get(f"{UNPAYWALL}/{doi}", {"email": self.mailto})
        if r is None:
            return {"outcome": err.split(":")[0], "detail": err, "http_status": None}
        if r.status_code == 404:
            return {"outcome": NOT_FOUND, "http_status": 404}
        if r.status_code != 200:
            return {"outcome": HTTP_ERROR, "http_status": r.status_code}
        try:
            data = r.json() or {}
        except ValueError:
            return {"outcome": HTTP_ERROR, "detail": "non-JSON", "http_status": 200}
        locs = data.get("oa_locations") or []
        best = data.get("best_oa_location") or {}
        return {"outcome": OK, "http_status": 200,
                "is_oa": bool(data.get("is_oa")),
                "n_oa_locations": len(locs),
                "best_oa_host_type": best.get("host_type"),
                "best_oa_url": best.get("url")}


def openalex_record(wid: str, work: dict | None, outcome: str) -> dict:
    oa = (work or {}).get("open_access") or {}
    rec = {"source": "openalex", "openalex_id": wid, "outcome": outcome,
           "ts": datetime.now(timezone.utc).isoformat()}
    if outcome == OK:
        rec.update({"is_oa": bool(oa.get("is_oa")),
                    "oa_status": oa.get("oa_status"),
                    "oa_url": oa.get("oa_url"),
                    "resolved_id": (work.get("id") or "").rsplit("/", 1)[-1]})
    return rec


def load_cache(path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Replay the append-only log. Later lines win, so a retry supersedes."""
    oa, up = {}, {}
    if not os.path.exists(path):
        return oa, up
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") == "openalex" and rec.get("openalex_id"):
                oa[rec["openalex_id"]] = rec
            elif rec.get("source") == "unpaywall" and rec.get("doi"):
                up[rec["doi"].lower()] = rec
    return oa, up


def load_overrides(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in OVERRIDE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path} is missing column(s) {missing}")
        out = {}
        for r in reader:
            wid = r["openalex_id"].strip()
            elig = parse_eligible(r["eligible"].strip())
            reason = r["exclusion_reason"].strip()
            if reason not in REASONS or (reason == "none") != elig:
                sys.exit(f"{path}: {wid} has eligible={elig} with "
                         f"exclusion_reason={reason!r}")
            if not r["reason"].strip() or not r["decided_by"].strip():
                sys.exit(f"{path}: {wid} needs a reason and a decided_by")
            out[wid] = {k: (v or "").strip() for k, v in r.items()}
        return out


def _is_answer(rec: dict | None, refresh: bool) -> bool:
    """A cached record is reused unless it is a transient failure being retried."""
    if rec is None:
        return False
    return not (refresh and rec.get("outcome", "").split(":")[0] in TRANSIENT)


# ---------------------------------------------------------------------------
# Provenance arithmetic. Every number in the limitation statement comes from
# here, not from a model and not from a hand count.
# ---------------------------------------------------------------------------

def rank_stratum(rank: int, n: int) -> str:
    """Same cut as draw_calibration_sample.py, applied to the FULL ranked list."""
    return "top_third" if rank <= n // 3 else "bottom_two_thirds"


def summarise(rows: list[dict], oa_cache: dict[str, dict]) -> dict:
    n = len(rows)
    by_reason = Counter(r["exclusion_reason"] for r in rows)
    by_rule = Counter(r["eligibility_rule"] for r in rows)

    cross = {s: {k: 0 for k in REASONS} for s in STRATA}
    stratum_size = Counter()
    for i, r in enumerate(rows, 1):
        s = rank_stratum(i, n)
        cross[s][r["exclusion_reason"]] += 1
        stratum_size[s] += 1
    cross_out = {}
    for s in STRATA:
        excl = cross[s]["no_oa"]
        cross_out[s] = {**cross[s], "total": stratum_size[s],
                        "excluded": excl,
                        "pct_excluded": round(100 * excl / max(stratum_size[s], 1), 1)}

    excluded = [r for r in rows if r["exclusion_reason"] != "none"]
    venue_counts: Counter[str] = Counter()
    venue_reason: dict[str, Counter] = {}
    for r in excluded:
        v = (r.get("venue") or "").strip() or "(no venue)"
        venue_counts[v] += 1
        venue_reason.setdefault(v, Counter())[r["exclusion_reason"]] += 1
    top = sorted(venue_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_VENUES]
    top_venues = [{"venue": v, "excluded": c} for v, c in top]

    fc = [r for r in rows if str(r.get("linked_to_framework_core", "")).strip() == "1"]
    fc_reason = Counter(r["exclusion_reason"] for r in fc)
    fc_excl = fc_reason["no_oa"]

    # What removing rule 1 did, measured on this run: every 10.1016 row is now
    # decided by the OA rule like any other.
    els = [r for r in rows if is_elsevier_doi(r.get("doi") or "")]
    els_rule = Counter(r["eligibility_rule"] for r in els)
    els_fc = [r for r in els if str(r.get("linked_to_framework_core", "")).strip() == "1"]

    return {
        "candidates_total": n,
        "eligible": by_reason["none"],
        "excluded": n - by_reason["none"],
        "pct_excluded": round(100 * (n - by_reason["none"]) / max(n, 1), 1),
        "counts_by_exclusion_reason": {k: by_reason[k] for k in REASONS},
        "counts_by_eligibility_rule": dict(sorted(by_rule.items())),
        "counts_by_exclusion_reason_x_rank_stratum": cross_out,
        "rank_stratum_definition": (
            "position in the FULL ranked candidates.csv (corpus_build.py rank "
            "order); top_third = ranks 1..n//3, bottom_two_thirds = the rest. "
            "Same cut as draw_calibration_sample.py but on the full list, since "
            "eligibility is decided before any filtering."),
        "top_excluded_venues": top_venues,
        "top_excluded_venues_basis": (
            f"top {TOP_VENUES} venues by count of ineligible rows (either "
            f"reason), ties broken alphabetically; blank venue reported as "
            f"'(no venue)'. {len(venue_counts)} distinct venues have at least "
            f"one excluded row."),
        "framework_core_linked": {
            "total": len(fc),
            "excluded": fc_excl,
            "fraction_excluded": round(fc_excl / len(fc), 4) if fc else None,
            "by_exclusion_reason": {k: fc_reason[k] for k in REASONS},
            "basis": "rows with linked_to_framework_core == 1",
        },
        "doi_10_1016_rows": {
            "total": len(els),
            "eligible": sum(1 for r in els if r["eligible"] == "True"),
            "no_oa": sum(1 for r in els if r["eligible"] == "False"),
            "by_eligibility_rule": dict(sorted(els_rule.items())),
            "framework_core_linked": len(els_fc),
            "framework_core_linked_eligible": sum(
                1 for r in els_fc if r["eligible"] == "True"),
            "note": ("Rule 1 (10.1016 -> ineligible regardless of OA) was "
                     "removed in v2.0.0; these rows are decided by the OA rule "
                     "alone. Every one of them was ineligible under v1 "
                     "(corpus/archive/2026-10-05_eligibility_v1/), so "
                     "`eligible` here is exactly the number of rows the "
                     "removal brought into scope."),
        },
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mark candidates eligible/ineligible under the access rule.")
    ap.add_argument("--candidates", default="corpus/candidates.csv",
                    help="read only; never modified")
    ap.add_argument("--out", default="corpus/eligibility.csv")
    ap.add_argument("--provenance", default="corpus/provenance.json")
    ap.add_argument("--log", default="corpus/eligibility_calls.jsonl",
                    help="append-only call log; doubles as the response cache")
    ap.add_argument("--overrides", default="corpus/eligibility_overrides.csv",
                    help="human decisions for rows neither source can answer")
    ap.add_argument("--mailto", default=None,
                    help="for the OpenAlex and Unpaywall polite pools; required "
                         "unless --offline")
    ap.add_argument("--rate", type=float, default=0.2,
                    help="seconds between requests")
    ap.add_argument("--offline", action="store_true",
                    help="replay the call log only; abort on any cache miss")
    ap.add_argument("--refresh-errors", action="store_true",
                    help="retry cached transient failures")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    if not args.offline and not args.mailto:
        sys.exit("--mailto is required unless --offline")

    with open(args.candidates, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        sys.exit(f"{args.candidates} has no rows")
    for need in ("openalex_id", "doi", "venue", "linked_to_framework_core"):
        if need not in fieldnames:
            sys.exit(f"{args.candidates} has no {need!r} column")

    if any(c in fieldnames for c in OUT_COLUMNS):
        sys.exit(f"{args.candidates} carries eligibility columns; since v1.1.0 "
                 f"they live in {args.out} only. Strip them from candidates.csv.")
    with open(args.candidates, "rb") as f:
        candidates_file_sha = hashlib.sha256(f.read()).hexdigest()
    key = candidates_key(rows)

    oa_cache, up_cache = load_cache(args.log)
    fetcher = None if args.offline else Fetcher(args.mailto, args.rate)
    run_started = datetime.now(timezone.utc)
    replayed = {"openalex": 0, "unpaywall": 0}
    misses: list[str] = []

    log = None if args.offline else open(args.log, "a", encoding="utf-8")

    def write(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()

    try:
        # --- 1. OpenAlex is_oa, every row ---------------------------------
        ids = [r["openalex_id"] for r in rows]
        need = [w for w in ids if not _is_answer(oa_cache.get(w), args.refresh_errors)]
        replayed["openalex"] = len(ids) - len(need)
        if args.offline:
            misses += [f"openalex:{w}" for w in need]
        else:
            if verbose and need:
                print(f"[elig] OpenAlex: {len(need)} IDs to fetch, "
                      f"{replayed['openalex']} replayed")
            for i in range(0, len(need), OPENALEX_BATCH):
                chunk = need[i:i + OPENALEX_BATCH]
                got, err = fetcher.openalex_batch(chunk)
                for w in chunk:
                    if w in got:
                        rec = openalex_record(w, got[w], OK)
                    elif err:
                        rec = openalex_record(w, None, err)
                        rec["detail"] = err
                    else:
                        work, err1 = fetcher.openalex_one(w)
                        rec = openalex_record(w, work, OK if work else err1)
                    write(rec)
                    oa_cache[w] = rec

        # --- 2. Unpaywall, only where rules 1 and the OpenAlex check leave
        #        the row undecided -----------------------------------------
        up_need = []
        for r in rows:
            doi = (r.get("doi") or "").strip().lower()
            if not doi:
                continue
            oa = oa_cache.get(r["openalex_id"]) or {}
            # Queried when OpenAlex says not-OA, and also when OpenAlex has
            # no record at all (see the docstring); not when OpenAlex says OA.
            if oa.get("outcome") == OK and oa.get("is_oa"):
                continue
            if oa.get("outcome") not in (OK, NOT_FOUND):
                continue
            if _is_answer(up_cache.get(doi), args.refresh_errors):
                replayed["unpaywall"] += 1
            else:
                up_need.append(doi)
        up_need = sorted(set(up_need))
        if args.offline:
            misses += [f"unpaywall:{d}" for d in up_need]
        else:
            if verbose and up_need:
                print(f"[elig] Unpaywall: {len(up_need)} DOIs to query, "
                      f"{replayed['unpaywall']} replayed")
            for j, doi in enumerate(up_need, 1):
                rec = {"source": "unpaywall", "doi": doi, **fetcher.unpaywall(doi),
                       "ts": datetime.now(timezone.utc).isoformat()}
                write(rec)
                up_cache[doi] = rec
                if verbose and j % 100 == 0:
                    print(f"  [up] {j}/{len(up_need)}", flush=True)
    finally:
        if log:
            log.close()

    if misses:
        print(f"\n  *** ABORT (--offline): {len(misses)} lookup(s) not in "
              f"{args.log}:")
        for m in misses[:10]:
            print(f"      {m}")
        print("  Nothing was written.")
        return 4

    # --- 3. classify ------------------------------------------------------
    overrides = load_overrides(args.overrides)
    applied_overrides = []
    undecided = []
    # Decisions are held in these dicts in memory and written to args.out;
    # the candidate rows are never written back.
    decided = [dict(r) for r in rows]
    for r in decided:
        doi = (r.get("doi") or "").strip().lower()
        oa = oa_cache.get(r["openalex_id"]) or {}
        oa_is_oa = oa.get("is_oa") if oa.get("outcome") == OK else None
        ov = overrides.get(r["openalex_id"])
        try:
            elig, reason, rule = classify(doi, oa_is_oa, up_cache.get(doi))
        except Undecided as exc:
            if ov:
                elig = parse_eligible(ov["eligible"])
                reason, rule = ov["exclusion_reason"], "manual_override"
                applied_overrides.append({
                    **ov, "why_rules_could_not_decide": str(exc),
                    "openalex_outcome": oa.get("outcome"),
                    "unpaywall_outcome": (up_cache.get(doi) or {}).get("outcome")})
            else:
                undecided.append((r["openalex_id"], doi, str(exc),
                                  oa.get("outcome")
                                  or (up_cache.get(doi) or {}).get("outcome")))
                continue
        else:
            if ov:
                sys.exit(f"\n  *** ABORT: {args.overrides} overrides "
                         f"{r['openalex_id']}, but the rules decide it "
                         f"({rule}). Overrides apply only to rows the rules "
                         f"cannot decide; remove the stale line. Nothing was "
                         f"written.")
        r["eligible"] = "True" if elig else "False"
        r["exclusion_reason"] = reason
        r["eligibility_rule"] = rule

    unknown_ov = set(overrides) - {r["openalex_id"] for r in rows}
    if unknown_ov:
        sys.exit(f"{args.overrides} names ID(s) not in candidates: "
                 f"{sorted(unknown_ov)}")

    if undecided:
        print(f"\n  *** ABORT: {len(undecided)} row(s) could not be decided. "
              f"Nothing was written --\n      a partly filled `eligible` "
              f"column would be read as a complete one.")
        for wid, doi, why, oc in undecided[:15]:
            print(f"      {wid} {doi or '(no doi)'}: {why} (last outcome: {oc})")
        print("  Rerun with --refresh-errors. A row neither source can answer "
              "for needs a line\n  in " + args.overrides + " (a recorded "
              "human judgment).")
        return 5

    with open(args.candidates, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != candidates_file_sha:
            raise AssertionError("candidates.csv changed during the run")

    write_eligibility_csv(args.out, decided, key)

    summary = summarise(decided, oa_cache)
    up_outcomes = Counter((up_cache.get(d) or {}).get("outcome")
                          for d in {(r.get("doi") or "").strip().lower() for r in decided
                                    if r.get("eligibility_rule", "").startswith(
                                        ("unpaywall", "no_oa:unpaywall"))})

    block = {
        "script": "eligibility.py",
        "script_version": SCRIPT_VERSION,
        "run_started_utc": run_started.isoformat(),
        "decided_by": "advisor meeting 2026-10-05: candidates restricted to "
                      "accessible papers",
        "rules_in_order": [
            "OpenAlex open_access.is_oa false AND no Unpaywall OA location "
            "(oa_locations empty, or DOI unknown to Unpaywall) -> ineligible, "
            "no_oa. No DOI: OpenAlex is_oa alone.",
            "otherwise -> eligible, none",
        ],
        "rule_history": (
            "v1 (commit 1a41053) also had rule 1: DOI prefix 10.1016 -> "
            "ineligible `elsevier`, even when OA. Removed 2026-10-05 in v2.0.0. "
            "v1 decisions and provenance: "
            "corpus/archive/2026-10-05_eligibility_v1/. Effect of the removal: "
            "see doi_10_1016_rows."),
        "inputs": {
            "openalex": "GET /works?filter=openalex:<ids>&select=id,doi,"
                        "open_access, 50 IDs per request; IDs missing from the "
                        "batch retried singly via /works/<id> (follows merges)",
            "unpaywall": "GET /v2/<doi>; queried for every row with a DOI "
                         "where OpenAlex is_oa is false or the OpenAlex record "
                         "is missing (404)",
            "call_log": args.log,
        },
        "api_requests": {
            "openalex": fetcher.n_openalex if fetcher else 0,
            "unpaywall": fetcher.n_unpaywall if fetcher else 0,
        },
        "responses_replayed_from_cache": replayed,
        "offline_replay": args.offline,
        "unpaywall_outcomes_for_rows_decided_by_it": dict(up_outcomes),
        "manual_overrides": applied_overrides,
        "manual_overrides_note": (
            "Human judgments, not rule outputs: rows that neither OpenAlex nor "
            "Unpaywall could answer for. Applied only where the rules could "
            "not decide. Source: " + args.overrides),
        "output": args.out,
        "candidates_modified": False,
        "candidates_key_sha256": key,
        "candidates_key_basis": "SHA-256 of ordered (openalex_id, lowercased "
                                "doi) pairs; the staleness check",
        "candidates_file_sha256_at_decision": candidates_file_sha,
        "rows_written": len(decided),
        "columns_written": ELIGIBILITY_COLUMNS,
        **summary,
        "note": (
            "Eligibility is a scope decision, not a screening decision. "
            "Ineligible rows keep their row in eligibility.csv, are never sent to a "
            "screening model, and are never forced to `uncertain`: they are "
            "out of scope, not undecided. Counts here are the 'excluded at "
            "eligibility' stage; 'excluded at screening' is reported "
            "separately by benchmark.py."),
    }
    prov = {}
    if os.path.exists(args.provenance):
        with open(args.provenance, encoding="utf-8") as f:
            prov = json.load(f)
    prov["eligibility"] = block
    # The v1 counterfactual describes a rule that no longer exists; its
    # figures are now the actual ones, and the block itself is archived.
    prov.pop("eligibility_counterfactual_no_rule1", None)
    with open(args.provenance, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    if verbose:
        c = summary["counts_by_exclusion_reason"]
        fc = summary["framework_core_linked"]
        print()
        print(f"[elig] {summary['candidates_total']} candidates: eligible="
              f"{c['none']}, no_oa={c['no_oa']} "
              f"({summary['pct_excluded']}% excluded)")
        for s, d in summary["counts_by_exclusion_reason_x_rank_stratum"].items():
            print(f"[elig]   {s}: {d['excluded']}/{d['total']} excluded "
                  f"({d['pct_excluded']}%)")
        print(f"[elig] framework_core-linked: {fc['excluded']}/{fc['total']} "
              f"excluded ({fc['fraction_excluded']})")
        e = summary["doi_10_1016_rows"]
        print(f"[elig] 10.1016 rows (no longer a rule): {e['eligible']}/"
              f"{e['total']} eligible on OA status")
        print(f"[elig] rules: " + ", ".join(
            f"{k}={v}" for k, v in summary["counts_by_eligibility_rule"].items()))
        print(f"  -> {args.out}  (candidates.csv not modified)")
        print(f"  -> {args.provenance}  (eligibility block)")
        if log is not None:
            print(f"  -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
