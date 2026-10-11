"""
eligibility -- which candidates are in scope, written to data/eligibility.csv.

Rules, first match wins:
  1. DOI prefix in [eligibility] exclude_doi_prefixes      -> ineligible, doi_prefix
  2. require_open_access and OpenAlex is_oa is false and
     Unpaywall lists no OA location (no DOI: OpenAlex alone) -> ineligible, no_oa
  3. otherwise                                               -> eligible, none
A row neither source can answer for (record deleted, DOI unknown) needs a human
decision in data/eligibility_overrides.csv (openalex_id,eligible,exclusion_reason,
reason,decided_by,decided_on); an override is applied ONLY where the rules cannot
decide. If any row stays undecided, nothing is written.

candidates.csv is never modified. Each row of eligibility.csv carries a key: the
SHA-256 of the ordered (openalex_id, doi) pairs of candidates.csv. Every later
stage loads eligibility through load(), which exits if the file is missing or the
key no longer matches -- i.e. after any rebuild that changed the corpus.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter

from .common import Http, JsonlLog, UNPAYWALL, provenance, read_csv, read_jsonl, short_id, write_csv

COLUMNS = ["openalex_id", "eligible", "exclusion_reason", "eligibility_rule", "candidates_key"]


class Undecided(Exception):
    pass


def classify(doi: str, is_oa: bool | None, up: dict | None, cfg_e: dict) -> tuple[bool, str, str]:
    doi = (doi or "").strip().lower()
    for prefix in cfg_e["exclude_doi_prefixes"]:
        if doi and doi.split("/", 1)[0] == prefix.lower():
            return False, "doi_prefix", f"doi_prefix_{prefix}"
    if not cfg_e["require_open_access"]:
        return True, "none", "no_oa_requirement"
    up_oa = bool(up and up.get("outcome") == "ok" and up.get("n_oa_locations", 0) > 0)
    if is_oa is None:
        if up_oa:
            return True, "none", "unpaywall_oa_location:openalex_record_missing"
        raise Undecided("no OpenAlex is_oa")
    if is_oa:
        return True, "none", "openalex_is_oa"
    if not doi:
        return False, "no_oa", "no_oa:no_doi_openalex_only"
    if up is None or up.get("outcome") not in ("ok", "not_found"):
        raise Undecided("no Unpaywall answer")
    if up["outcome"] == "not_found":
        return False, "no_oa", "no_oa:unpaywall_doi_unknown"
    return (True, "none", "unpaywall_oa_location") if up_oa else (False, "no_oa", "no_oa:unpaywall_no_location")


def key(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['openalex_id']}\t{(r.get('doi') or '').strip().lower()}\n".encode())
    return h.hexdigest()


def load(cfg, rows: list[dict] | None = None) -> dict[str, dict]:
    """openalex_id -> {eligible: bool, exclusion_reason, eligibility_rule}; exits if missing or stale."""
    if not os.path.exists(cfg.eligibility):
        sys.exit(f"{cfg.eligibility} is missing -- run `litreview eligibility`")
    rows = rows if rows is not None else read_csv(cfg.candidates)
    erows = read_csv(cfg.eligibility)
    if {r.get("candidates_key") for r in erows} != {key(rows)}:
        sys.exit(f"STALE: {cfg.eligibility} was computed for a different candidates.csv "
                 f"(the corpus changed). Rerun `litreview eligibility` (--offline replays the log).")
    return {r["openalex_id"]: {"eligible": r["eligible"] == "True",
                               "exclusion_reason": r["exclusion_reason"],
                               "eligibility_rule": r["eligibility_rule"]} for r in erows}


def summarise(rows: list[dict], dec: dict) -> dict:
    n = len(rows)
    cut = n // 3
    reasons = Counter(dec[r["openalex_id"]][1] for r in rows)
    strata = {}
    for name, part in (("top_third", rows[:cut]), ("bottom_two_thirds", rows[cut:])):
        c = Counter(dec[r["openalex_id"]][1] for r in part)
        strata[name] = {**c, "total": len(part), "pct_excluded": round(100 * (len(part) - c["none"]) / max(len(part), 1), 1)}
    venues = Counter((r.get("venue") or "(no venue)") for r in rows if dec[r["openalex_id"]][1] != "none")
    core = [r for r in rows if str(r.get("linked_to_core")) == "1"]
    core_ex = sum(dec[r["openalex_id"]][1] != "none" for r in core)
    return {"candidates": n, "eligible": reasons["none"], "by_reason": dict(reasons),
            "by_rank_stratum": strata,
            "top_excluded_venues": [{"venue": v, "excluded": k} for v, k in
                                    sorted(venues.items(), key=lambda x: (-x[1], x[0]))[:15]],
            "core_linked": {"total": len(core), "excluded": core_ex,
                            "fraction_excluded": round(core_ex / len(core), 4) if core else None},
            "by_rule": dict(Counter(dec[r["openalex_id"]][2] for r in rows))}


def run(cfg, args) -> int:
    ce = cfg["eligibility"]
    rows = read_csv(cfg.candidates)
    log_path = cfg.path("data", "eligibility_calls.jsonl")
    oa, up = {}, {}
    for rec in read_jsonl(log_path):
        if rec.get("source") == "openalex":
            oa[rec["openalex_id"]] = rec
        elif rec.get("source") == "unpaywall":
            up[rec["doi"]] = rec

    def answered(rec):
        return rec is not None and rec.get("outcome") in ("ok", "not_found")

    need_oa = [r["openalex_id"] for r in rows if not answered(oa.get(r["openalex_id"]))]
    if ce["require_open_access"] and need_oa:
        if args.offline:
            sys.exit(f"--offline: {len(need_oa)} OpenAlex lookups not in the log; nothing written")
        http, log = Http(cfg.mailto, rate=0.2), JsonlLog(log_path)
        for i in range(0, len(need_oa), 50):
            chunk = need_oa[i:i + 50]
            d = http.json("https://api.openalex.org/works", params={
                "filter": "openalex:" + "|".join(chunk), "select": "id,open_access",
                "per-page": 50, "mailto": cfg.mailto}) or {}
            got = {short_id(w["id"]): w for w in d.get("results") or []}
            for wid in chunk:
                w = got.get(wid)
                if w is None and d:
                    w = http.json(f"https://api.openalex.org/works/{wid}",
                                  params={"select": "id,open_access", "mailto": cfg.mailto})
                rec = ({"source": "openalex", "openalex_id": wid, "outcome": "ok",
                        "is_oa": bool((w.get("open_access") or {}).get("is_oa"))} if w else
                       {"source": "openalex", "openalex_id": wid, "outcome": "not_found" if d else "error"})
                oa[wid] = rec
                log.write(rec)
        log.close()

    need_up = sorted({r["doi"].lower() for r in rows if r.get("doi")
                      and ce["require_open_access"]
                      and not (oa.get(r["openalex_id"]) or {}).get("is_oa")
                      and not answered(up.get(r["doi"].lower()))})
    if need_up:
        if args.offline:
            sys.exit(f"--offline: {len(need_up)} Unpaywall lookups not in the log; nothing written")
        http, log = Http(cfg.mailto, rate=0.2), JsonlLog(log_path)
        print(f"[eligibility] Unpaywall: {len(need_up)} DOIs")
        for doi in need_up:
            r = http.get(f"{UNPAYWALL}/{doi}", params={"email": cfg.mailto})
            if r is None or r.status_code not in (200, 404):
                rec = {"outcome": "error", "http_status": getattr(r, "status_code", None)}
            elif r.status_code == 404:
                rec = {"outcome": "not_found"}
            else:
                rec = {"outcome": "ok", "n_oa_locations": len((r.json() or {}).get("oa_locations") or [])}
            up[doi] = {"source": "unpaywall", "doi": doi, **rec}
            log.write(up[doi])
        log.close()

    ov_path = cfg.path("data", "eligibility_overrides.csv")
    overrides = {r["openalex_id"]: r for r in read_csv(ov_path)} if os.path.exists(ov_path) else {}
    for o in overrides.values():
        if not (o.get("reason") or "").strip() or not (o.get("decided_by") or "").strip():
            sys.exit(f"{ov_path}: {o['openalex_id']} needs a reason and decided_by")
    dec, undecided, used = {}, [], []
    for r in rows:
        wid, doi = r["openalex_id"], (r.get("doi") or "").lower()
        rec = oa.get(wid) or {}
        is_oa = rec.get("is_oa") if rec.get("outcome") == "ok" else None
        try:
            dec[wid] = classify(doi, is_oa, up.get(doi), ce)
            if wid in overrides:
                sys.exit(f"override for {wid} but the rules decide it ({dec[wid][2]}); remove it")
        except Undecided as e:
            if wid in overrides:
                o = overrides[wid]
                dec[wid] = (o["eligible"] == "True", o["exclusion_reason"], "manual_override")
                used.append({**o, "why_rules_could_not_decide": str(e)})
            else:
                undecided.append(f"{wid} {doi or '(no doi)'}: {e}")
    if undecided:
        print(f"*** {len(undecided)} row(s) undecided; nothing written. Rerun (transient errors are "
              f"retried) or add a recorded human decision to {ov_path}:")
        for u in undecided[:15]:
            print("    " + u)
        return 5

    k = key(rows)
    write_csv(cfg.eligibility, [{"openalex_id": r["openalex_id"], "eligible": str(dec[r["openalex_id"]][0]),
                                 "exclusion_reason": dec[r["openalex_id"]][1],
                                 "eligibility_rule": dec[r["openalex_id"]][2], "candidates_key": k}
                                for r in rows], COLUMNS)
    s = summarise(rows, dec)
    provenance(cfg, "eligibility", {"rules": ce, "candidates_key": k, **s, "manual_overrides": used,
                                    "candidates_modified": False})
    print(f"[eligibility] {s['eligible']}/{s['candidates']} eligible; by reason {s['by_reason']}; "
          f"core-linked excluded {s['core_linked']['excluded']}/{s['core_linked']['total']}")
    print(f"  -> {cfg.eligibility}")
    return 0
