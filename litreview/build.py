"""
build -- seeds -> OpenAlex seed-and-expand -> ranked data/candidates.csv.

1. Resolve each seed (DOI first, then title match). Verify data/seeds_resolved.csv
   by hand: a wrong seed silently biases the whole corpus.
2. Expand backward (works a seed cites; free, on the seed record) and forward
   (works citing a seed since min_year), per the seed's expand_backward /
   expand_forward columns. Those columns are the ONLY retrieval policy.
3. Weight each seed by topical precision against a target domain T (the OpenAlex
   topics of the target seeds): p(s) = share of its references in T, q(s) = share
   of a seeded random sample of its citing works in T. A candidate's score is the
   sum of its linking seeds' weights, in the direction each link was followed.
   Weights order the list; they never remove anything.
4. Filter (year, type, language, title dedupe), rank, and backfill missing
   abstracts from Semantic Scholar by DOI (cached).

Every cap that bites is recorded as a truncation: a pagination-order cap is a
hidden inclusion criterion, so it is reported, not absorbed. No model is involved.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone

from .common import (Http, JsonlLog, S2, clean_doi, provenance, read_csv, read_jsonl,
                     short_id, title_similarity, norm_title, truthy, write_csv)

SELECT = ("id,doi,display_name,publication_year,type,primary_location,authorships,"
          "cited_by_count,referenced_works,abstract_inverted_index,open_access,"
          "language,topics")


# --- seeds ------------------------------------------------------------------

def load_seeds(path: str) -> list[dict]:
    rows = [r for r in read_csv(path) if (r.get("title") or "").strip()]
    for r in rows:
        r["ref"] = (r.get("ref") or "").strip()
        r["role"] = (r.get("role") or "unassigned").strip()
        r["expand_backward"] = truthy(r.get("expand_backward"))
        r["expand_forward"] = truthy(r.get("expand_forward"))
    if len({r["ref"] for r in rows}) != len(rows):
        raise SystemExit(f"{path}: every seed needs a unique `ref`")
    return rows


def resolve(http: Http, seed: dict, min_sim: float) -> dict:
    out = {k: seed.get(k, "") for k in ("ref", "role", "year", "first_author", "title")}
    out.update(expand_backward=seed["expand_backward"], expand_forward=seed["expand_forward"],
               openalex_id="", matched_title="", matched_year="", matched_doi="",
               similarity=0.0, status="UNRESOLVED", _work=None)
    work = None
    if clean_doi(seed.get("doi")):
        hits = http.openalex("/works", filter=f"doi:{clean_doi(seed['doi'])}",
                             select=SELECT).get("results") or []
        if hits:
            work, out["status"], out["similarity"] = hits[0], "RESOLVED_BY_DOI", 1.0
    if work is None:
        q = re.sub(r'["\\,]', " ", seed["title"])
        best, score = None, 0.0
        for c in http.openalex("/works", filter=f"title.search:{q}", select=SELECT,
                               **{"per-page": 25}).get("results") or []:
            s = title_similarity(seed["title"], c.get("display_name") or "")
            if str(c.get("publication_year")) == str(seed.get("year") or ""):
                s += 0.05
            if s > score:
                best, score = c, s
        if best is not None and score >= min_sim:
            work = best
            out["similarity"] = round(min(score, 1.0), 3)
            out["status"] = "RESOLVED_BY_TITLE" if score >= 0.9 else "RESOLVED_LOW_CONFIDENCE"
        elif best is not None:
            out.update(similarity=round(score, 3), matched_title=best.get("display_name") or "",
                       status="REJECTED_LOW_SIMILARITY")
    if work is not None:
        out.update(openalex_id=short_id(work.get("id")), matched_title=work.get("display_name") or "",
                   matched_year=work.get("publication_year") or "",
                   matched_doi=clean_doi(work.get("doi")), _work=work)
    return out


# --- precision weights --------------------------------------------------------

def topics(work: dict, min_score: float) -> set[str]:
    return {short_id(t.get("id")) for t in (work.get("topics") or [])
            if float(t.get("score") or 0) >= min_score}


def fill_weights(rows: list[dict], p: str, n: str, w: str, basis: str, min_n: int, empty: str):
    """Small samples fall back to the same-role mean, then the global mean, then 1.0."""
    measured = {r["ref"]: r[p] for r in rows if r[p] is not None and r[n] >= min_n}
    by_role = defaultdict(list)
    for r in rows:
        if r["ref"] in measured:
            by_role[r["role"]].append(measured[r["ref"]])
    glob = round(sum(measured.values()) / len(measured), 4) if measured else 1.0
    for r in rows:
        if r["ref"] in measured:
            r[w], r[basis] = r[p], "measured"
            continue
        r[basis] = f"under_sampled(n={r[n]}<{min_n})" if r[n] else empty
        pool = by_role.get(r["role"])
        r[w] = round(sum(pool) / len(pool), 4) if pool else glob
        r[basis] += f"|role_default:{r['role']}" if pool else ("|global_default" if measured else "|uniform")


# --- candidates ---------------------------------------------------------------

def abstract(inv: dict | None) -> str:
    pos = sorted((i, w) for w, idx in (inv or {}).items() for i in idx)
    return " ".join(w for _, w in pos[:400]).replace("\n", " ").strip()


def candidate_row(w: dict, refs: list[str], dirs: dict, roles: dict, weights: dict,
                  core_roles: set[str], year_now: int) -> dict:
    ew = [max((weights.get(r, {}).get(d, 1.0) for d in (dirs.get(r) or {"backward"})), default=1.0)
          for r in refs]
    seed_roles = sorted({roles.get(r, "unassigned") for r in refs})
    year = w.get("publication_year")
    oa = w.get("open_access") or {}
    loc = w.get("primary_location") or {}
    return {
        "openalex_id": short_id(w.get("id")), "doi": clean_doi(w.get("doi")),
        "title": w.get("display_name") or "", "year": year,
        "venue": ((loc.get("source") or {}).get("display_name")) or "",
        "authors": "; ".join((a.get("author") or {}).get("display_name") or ""
                             for a in (w.get("authorships") or [])[:6]),
        "type": w.get("type") or "", "cited_by_count": w.get("cited_by_count") or 0,
        "citations_per_year": round((w.get("cited_by_count") or 0) / max(1.0, year_now - year), 2),
        "n_seed_links": len(refs), "seed_weight": round(sum(ew), 4),
        "seed_refs": "|".join(refs),
        "seed_link_directions": "|".join(",".join(sorted(dirs.get(r) or [])) for r in refs),
        "seed_roles": "|".join(seed_roles),
        "linked_to_core": int(bool(core_roles & set(seed_roles))),
        "oa_pdf_url": oa.get("oa_url") or loc.get("pdf_url") or "",
        "abstract": abstract(w.get("abstract_inverted_index")),
        "abstract_source": "openalex" if w.get("abstract_inverted_index") else "none",
    }


def backfill_abstracts(cfg, http: Http, rows: list[dict]) -> dict:
    """Fill missing abstracts from Semantic Scholar by DOI. Cached; transient errors retried next run."""
    log_path = cfg.path("data", "abstract_backfill.jsonl")
    cache = {r["doi"]: r for r in read_jsonl(log_path) if r.get("outcome") in ("ok", "no_abstract", "not_found")}
    log, counts = JsonlLog(log_path), defaultdict(int)
    try:
        for r in rows:
            if r["abstract"] or not r["doi"]:
                continue
            rec = cache.get(r["doi"])
            if rec is None:
                resp = http.get(f"{S2}/DOI:{r['doi']}", params={"fields": "abstract"})
                if resp is None or resp.status_code not in (200, 404):
                    rec = {"doi": r["doi"], "outcome": "error", "status": getattr(resp, "status_code", None)}
                elif resp.status_code == 404:
                    rec = {"doi": r["doi"], "outcome": "not_found"}
                else:
                    a = " ".join(((resp.json() or {}).get("abstract") or "").split())
                    rec = {"doi": r["doi"], "outcome": "ok" if a else "no_abstract", "abstract": a}
                log.write(rec)
            counts[rec["outcome"]] += 1
            if rec["outcome"] == "ok":
                r["abstract"], r["abstract_source"] = rec["abstract"], "semantic_scholar"
    finally:
        log.close()
    return dict(counts)


# --- main ---------------------------------------------------------------------

def run(cfg, args) -> int:
    c = cfg["corpus"]
    http = Http(cfg.mailto)
    year_now = datetime.now(timezone.utc).year
    seeds = load_seeds(cfg.path(c["seeds"]))
    core_roles = set(c["core_roles"])
    target = set(c["target_refs"]) or {s["ref"] for s in seeds if s["role"] in core_roles}
    print(f"[build] {len(seeds)} seeds; core roles {sorted(core_roles) or '(none)'}; "
          f"target seeds {sorted(target) or '(none: uniform weights)'}")

    resolved = [resolve(http, s, c["min_title_similarity"]) for s in seeds]
    for r in resolved:
        flag = {"RESOLVED_LOW_CONFIDENCE": "??", "UNRESOLVED": "--",
                "REJECTED_LOW_SIMILARITY": "--"}.get(r["status"], "ok")
        print(f"  {flag} [{r['ref']}] {r['status']:<24} {(r['matched_title'] or r['title'])[:70]}")
    write_csv(cfg.path("data", "seeds_resolved.csv"), resolved,
              ["ref", "role", "expand_backward", "expand_forward", "status", "similarity",
               "year", "matched_year", "first_author", "title", "matched_title",
               "matched_doi", "openalex_id"])
    ok = [r for r in resolved if r["_work"]]
    seed_ids = {r["openalex_id"]: r["ref"] for r in ok}
    roles = {r["ref"]: r["role"] for r in resolved}

    tmin = float(c["topic_score_min"])
    T = set().union(*[topics(r["_work"], tmin) for r in ok if r["ref"] in target]) if target else set()
    prec = {r["ref"]: {"ref": r["ref"], "role": r["role"], "p": None, "p_n": 0, "q": None, "q_n": 0,
                       "q_population": 0, "bwd_weight": 1.0, "fwd_weight": 1.0,
                       "bwd_basis": "not_expanded", "fwd_basis": "not_expanded"} for r in resolved}
    works, links = {}, defaultdict(set)
    dirs = defaultdict(lambda: defaultdict(set))
    truncations = []

    # backward
    refs_of = {r["ref"]: [short_id(x) for x in r["_work"].get("referenced_works") or []]
               for r in ok if r["expand_backward"]}
    ids = sorted({i for v in refs_of.values() for i in v})
    print(f"[build] backward: {len(ids)} cited works from {len(refs_of)} seeds")
    ref_works = {}
    for i in range(0, len(ids), 50):
        for w in http.openalex("/works", filter="openalex_id:" + "|".join(ids[i:i + 50]),
                               select=SELECT, **{"per-page": 50}).get("results") or []:
            ref_works[short_id(w["id"])] = w
    for ref, rids in refs_of.items():
        known = [i for i in rids if i in ref_works]
        prec[ref]["p_n"] = len(known)
        if known and T:
            prec[ref]["p"] = round(sum(bool(topics(ref_works[i], tmin) & T) for i in known) / len(known), 4)
        cap = c["max_backward_per_seed"]
        if cap and len(rids) > cap:
            truncations.append({"ref": ref, "direction": "backward", "available": len(rids), "kept": cap})
            rids = rids[:cap]
        for i in rids:
            if i in ref_works:
                works[i] = ref_works[i]
                links[i].add(ref)
                dirs[i][ref].add("backward")

    # forward
    fwd = [r for r in ok if r["expand_forward"]]
    print(f"[build] forward: citing works of {len(fwd)} seeds since {c['min_year']}")
    for r in fwd:
        flt = f"cites:{r['openalex_id']},from_publication_date:{c['min_year']}-01-01"
        if T:
            total = int((http.openalex("/works", filter=flt, select="id",
                                       **{"per-page": 1}).get("meta") or {}).get("count") or 0)
            n = min(int(c["forward_sample_size"]), total)
            sample = (http.openalex("/works", filter=flt, select="id,topics", sample=n,
                                    seed=int(c["forward_sample_seed"]), **{"per-page": min(n, 200)})
                      .get("results") or []) if n else []
            prec[r["ref"]].update(q_population=total, q_n=len(sample))
            if sample:
                prec[r["ref"]]["q"] = round(sum(bool(topics(w, tmin) & T) for w in sample) / len(sample), 4)
        cap, got, cursor, available = c["max_forward_per_seed"], 0, "*", None
        while cursor:
            d = http.openalex("/works", filter=flt, select=SELECT, cursor=cursor, **{"per-page": 200})
            available = available if available is not None else (d.get("meta") or {}).get("count", 0)
            for w in d.get("results") or []:
                if cap and got >= cap:
                    break
                wid = short_id(w["id"])
                works[wid] = w
                links[wid].add(r["ref"])
                dirs[wid][r["ref"]].add("forward")
                got += 1
            cursor = (d.get("meta") or {}).get("next_cursor") if (d.get("results") and not (cap and got >= cap)) else None
        if cap and available and available > got:
            truncations.append({"ref": r["ref"], "direction": "forward", "available": available, "kept": got})

    rows = list(prec.values())
    min_n = int(c["precision_min_sample"])
    fill_weights([x for x in rows if x["ref"] in refs_of], "p", "p_n", "bwd_weight", "bwd_basis", min_n, "no_references")
    fill_weights([x for x in rows if x["ref"] in {r["ref"] for r in fwd}], "q", "q_n", "fwd_weight", "fwd_basis", min_n, "no_citations")
    weights = ({x["ref"]: {"backward": x["bwd_weight"], "forward": x["fwd_weight"]} for x in rows}
               if c["weight_by_precision"] and T else {})
    write_csv(cfg.path("data", "seed_precision.csv"), rows)

    # assemble
    cands, seen = [], set()
    for wid, w in works.items():
        if wid in seed_ids or not w.get("publication_year") or w["publication_year"] < c["min_year"]:
            continue
        if (w.get("type") or "") not in c["keep_types"] or (w.get("language") or c["language"]) != c["language"]:
            continue
        t = norm_title(w.get("display_name") or "")
        if not t or t in seen:
            continue
        seen.add(t)
        refs = sorted(links[wid], key=lambda x: (len(x), x))
        cands.append(candidate_row(w, refs, dirs[wid], roles, weights, core_roles, year_now))
    cands.sort(key=lambda r: (-r["linked_to_core"], -r["seed_weight"], -r["n_seed_links"],
                              -r["citations_per_year"], -r["year"]))
    backfill = backfill_abstracts(cfg, http, cands) if c["backfill_abstracts"] else {}
    write_csv(cfg.candidates, cands)

    coverage = []
    ids_in = {x["openalex_id"] for x in cands} | set(seed_ids)
    for r in ok:
        if r["role"] in set(c["coverage_roles"]):
            rr = {short_id(x) for x in r["_work"].get("referenced_works") or []}
            if rr:
                coverage.append({"ref": r["ref"], "references": len(rr), "in_corpus": len(rr & ids_in),
                                 "pct": round(100 * len(rr & ids_in) / len(rr), 1)})

    provenance(cfg, "build", {
        "config": c, "api_requests": http.n,
        "seeds": {"loaded": len(seeds), "resolved": len(ok),
                  "low_confidence": sum(r["status"] == "RESOLVED_LOW_CONFIDENCE" for r in resolved)},
        "target_domain_topics": sorted(T), "seed_precision": rows,
        "works_retrieved": len(works), "candidates": len(cands),
        "candidates_linked_to_core": sum(x["linked_to_core"] for x in cands),
        "abstract_sources": {k: sum(x["abstract_source"] == k for x in cands)
                             for k in ("openalex", "semantic_scholar", "none")},
        "abstract_backfill": backfill, "truncations": truncations, "coverage": coverage,
    })
    print(f"[build] {len(ok)}/{len(seeds)} seeds resolved -> {len(works)} works -> "
          f"{len(cands)} candidates ({sum(x['linked_to_core'] for x in cands)} linked to core); "
          f"{http.n} requests")
    for t in truncations:
        print(f"  *** cap hit: seed [{t['ref']}] {t['direction']} kept {t['kept']} of {t['available']}")
    for cv in coverage:
        print(f"  coverage: seed [{cv['ref']}] {cv['in_corpus']}/{cv['references']} ({cv['pct']}%)")
    print(f"  -> {cfg.candidates}\n  -> {cfg.path('data', 'seeds_resolved.csv')}  (verify every match by hand)")
    return 0
