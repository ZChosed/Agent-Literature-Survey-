"""
fetch -- document list -> extraction targets -> validated full text.

  litreview fetch                      targets from the latest corpus screening run
  litreview fetch --missing-abstracts  eligible papers with no abstract, so screening
                                       can use a full-text head instead of the title
  litreview fetch --manual-only        no network: just ingest manual_pdfs/

1. Every candidate gets ONE outcome (screening/outcomes.csv): included,
   needs_human_review (model uncertain, title-only, or a failed response -- a
   failure is never treated as an exclude), excluded_at_screening, or
   excluded_at_eligibility. Decide review rows in screening/review_decisions.csv
   (paper_id, decision=include|exclude, criterion_id, decided_by).
2. Included papers map to extraction schemas by criterion_id ([extraction.schemas.*]
   `criteria`); `max_papers` keeps the top N by corpus rank -- rank only, so the
   subset is never shaped by which PDFs happened to download.
3. Each target's PDF is taken from manual_pdfs/ (any filename, matched by the title
   on its first pages) or fetched from legitimate open-access sources only: OpenAlex,
   Unpaywall, Semantic Scholar, Zenodo, arXiv. No publisher HTML is scraped and bot
   blocks are not evaded; those become manual retrievals. A PDF is accepted only if
   it yields >= 3,000 characters and >= 60% of its title words appear on its first
   two pages. Text goes to documents/<id>.txt with a [[page N]] marker per page.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections import Counter
from urllib.parse import quote_plus

from . import eligibility as elig
from .common import (Http, JsonlLog, OPENALEX, S2, UNPAYWALL, latest_run, provenance, read_csv,
                     read_jsonl, write_csv)

MIN_CHARS, TITLE_MIN = 3000, 0.6
OUTCOMES = ("included", "needs_human_review", "excluded_at_screening", "excluded_at_eligibility")


# --- PDF text -----------------------------------------------------------------

def pdf_text(path: str) -> tuple[str, int, str]:
    import pymupdf
    parts, first2 = [], []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", page.get_text("text") or "")
            parts.append(f"[[page {i + 1}]]\n{t.strip()}\n")
            if i < 2:
                first2.append(t)
        n = len(doc)
    return "\n".join(parts), n, "\n".join(first2)


def title_words_match(title: str, text: str) -> float:
    words = set(re.findall(r"[a-z]{4,}", title.lower()))
    hay = re.sub(r"\s+", " ", text.lower())
    return sum(w in hay for w in words) / len(words) if words else 1.0


def title_pairs_match(title: str, text: str) -> float:
    """Share of the title's consecutive word pairs present; discriminates similar titles."""
    tw = re.findall(r"[a-z0-9]+", title.lower())
    xw = re.findall(r"[a-z0-9]+", text.lower())
    pairs, have = set(zip(tw, tw[1:])), set(zip(xw, xw[1:]))
    return sum(p in have for p in pairs) / len(pairs) if pairs else 0.0


def accept(path: str, title: str) -> tuple[bool, str, dict]:
    try:
        text, pages, first2 = pdf_text(path)
    except Exception as e:                                      # noqa: BLE001
        return False, f"unparseable:{type(e).__name__}", {}
    chars = len(re.sub(r"\[\[page \d+\]\]", "", text).strip())
    tm = round(title_words_match(title, first2), 3)
    info = {"text": text, "pages": pages, "chars": chars, "title_match": tm}
    if chars < MIN_CHARS:
        return False, f"too_little_text:{chars}", info
    return (True, "", info) if tm >= TITLE_MIN else (False, f"title_mismatch:{tm}", info)


# --- sources ------------------------------------------------------------------

def urls_for(http: Http, row: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def add(origin, url):
        if url and url not in [u for _, u in out]:
            out.append((origin, url))

    add("openalex", row.get("oa_pdf_url"))
    w = http.json(f"{OPENALEX}/works/{row['openalex_id']}", params={"select": "locations", "mailto": http.mailto}) or {}
    for loc in w.get("locations") or []:
        add("openalex_location", loc.get("pdf_url"))
    landings = [loc.get("landing_page_url") or "" for loc in w.get("locations") or []]
    if row.get("doi"):
        for loc in (http.json(f"{UNPAYWALL}/{row['doi']}", params={"email": http.mailto}) or {}).get("oa_locations") or []:
            add("unpaywall", loc.get("url_for_pdf"))
            landings.append(loc.get("url") or "")
        s2 = http.json(f"{S2}/DOI:{row['doi']}", params={"fields": "openAccessPdf,externalIds"}) or {}
        add("semantic_scholar", (s2.get("openAccessPdf") or {}).get("url"))
        if (s2.get("externalIds") or {}).get("ArXiv"):
            add("arxiv", f"https://arxiv.org/pdf/{s2['externalIds']['ArXiv']}")
    for u in landings + [u for _, u in out]:
        if m := re.search(r"arxiv\.org/abs/([^\s?#]+)", u or ""):
            add("arxiv", f"https://arxiv.org/pdf/{m.group(1)}")
        if m := re.search(r"zenodo\.org/records?/(\d+)", u or ""):
            for f in (http.json(f"https://zenodo.org/api/records/{m.group(1)}") or {}).get("files") or []:
                if (f.get("key") or "").lower().endswith(".pdf"):
                    add("zenodo", (f.get("links") or {}).get("self"))
    if not any(o == "arxiv" for o, _ in out):
        words = re.findall(r"[a-z]{4,}", row["title"].lower())[:8]
        r = http.get("http://export.arxiv.org/api/query",
                     params={"search_query": " AND ".join(f"ti:{x}" for x in words), "max_results": 3}) if len(words) >= 3 else None
        for entry in re.findall(r"<entry>(.*?)</entry>", r.text if r is not None and r.status_code == 200 else "", re.S):
            t, i = re.search(r"<title>(.*?)</title>", entry, re.S), re.search(r"arxiv\.org/abs/([^<]+)</id>", entry)
            if t and i and title_words_match(row["title"], t.group(1)) >= 0.8:
                add("arxiv", f"https://arxiv.org/pdf/{i.group(1).strip()}")
    return out


def download(http: Http, url: str, path: str) -> str:
    r = http.get(url, allow_redirects=True)
    if r is None:
        return "network_error"
    if r.status_code == 403:
        return "forbidden"
    if r.status_code != 200:
        return f"http_{r.status_code}"
    if not r.content.startswith(b"%PDF") and "pdf" not in (r.headers.get("content-type") or "").lower():
        return "not_pdf"
    with open(path, "wb") as f:
        f.write(r.content)
    return ""


def ingest_manual(drop: str, docs: str, missing: dict[str, str]) -> None:
    """Match PDFs in the drop folder to missing papers: by <paper_id>.pdf, else by title pairs."""
    for name in sorted(os.listdir(drop)) if os.path.isdir(drop) else []:
        if not name.lower().endswith(".pdf") or not missing:
            continue
        stem = os.path.splitext(name)[0]
        pid = stem if stem in missing else None
        if pid is None:
            try:
                first2 = pdf_text(os.path.join(drop, name))[2]
            except Exception:                                   # noqa: BLE001
                print(f"  [manual] {name}: unreadable")
                continue
            ranked = sorted(((title_pairs_match(t, first2), p) for p, t in missing.items()), reverse=True)
            if ranked[0][0] < TITLE_MIN or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.2):
                print(f"  [manual] {name}: no unambiguous match -- rename it to the paper_id")
                continue
            pid = ranked[0][1]
        shutil.copyfile(os.path.join(drop, name), os.path.join(docs, f"{pid}.pdf.manual"))
        print(f"  [manual] {name} -> {pid}")
        del missing[pid]


# --- stages -------------------------------------------------------------------

def outcomes(cfg, run: str, rows: list[dict]) -> list[dict]:
    eg = elig.load(cfg, rows)
    calls = {c["paper_id"]: c for c in read_jsonl(os.path.join(run, "calls.jsonl"))}
    dpath = cfg.path("screening", "review_decisions.csv")
    human = {d["paper_id"]: d for d in read_csv(dpath)} if os.path.exists(dpath) else {}
    out = []
    for rank, r in enumerate(rows, 1):
        pid, c = r["openalex_id"], calls.get(r["openalex_id"])
        p = (c or {}).get("parsed") or {}
        d = p.get("decision")
        if not eg[pid]["eligible"]:
            o, basis = "excluded_at_eligibility", eg[pid]["exclusion_reason"]
        elif c is None:
            o, basis = "needs_human_review", "not screened in this run"
        elif c["validation_status"] == "forced":
            o, basis = "needs_human_review", "title only"
        elif c["validation_status"] != "valid":
            o, basis = "needs_human_review", "failed response"
        else:
            o, basis = {"include": ("included", "model"), "exclude": ("excluded_at_screening", "model")}.get(d, ("needs_human_review", "model uncertain"))
        crit = p.get("criterion_id", "")
        if o == "needs_human_review" and pid in human:
            hd = human[pid]
            o = "included" if hd["decision"].strip().lower() == "include" else "excluded_at_screening"
            basis, crit = f"human:{hd.get('decided_by', '')}", hd.get("criterion_id", "") or crit
        out.append({"rank": rank, "paper_id": pid, "doi": r.get("doi", ""), "title": r["title"],
                    "year": r.get("year", ""), "venue": r.get("venue", ""), "outcome": o, "basis": basis,
                    "decision": d or "", "criterion_id": crit, "reason": p.get("reason", ""),
                    "linked_to_core": r.get("linked_to_core", ""), "oa_pdf_url": r.get("oa_pdf_url", "")})
    return out


def run(cfg, args) -> int:
    rows = read_csv(cfg.candidates)
    by_id = {r["openalex_id"]: r for r in rows}
    if args.missing_abstracts:
        eg = elig.load(cfg, rows)
        targets = [{"paper_id": r["openalex_id"], "title": r["title"], "doi": r.get("doi", ""),
                    "oa_pdf_url": r.get("oa_pdf_url", ""), "schema": ""}
                   for r in rows if eg[r["openalex_id"]]["eligible"] and not r["abstract"].strip()]
    else:
        srun = args.run or latest_run(cfg.path("screening", "runs"), "corpus_")
        if not srun:
            sys.exit("no corpus screening run -- run `litreview screen`")
        oc = outcomes(cfg, srun, rows)
        write_csv(cfg.path("screening", "outcomes.csv"), oc)
        write_csv(cfg.path("screening", "document_list.csv"), [o for o in oc if o["outcome"] == "included"])
        write_csv(cfg.path("screening", "needs_human_review.csv"), [o for o in oc if o["outcome"] == "needs_human_review"])
        counts = Counter(o["outcome"] for o in oc)
        targets, unmapped = [], []
        for name, sch in cfg["extraction"]["schemas"].items():
            hits = [o for o in oc if o["outcome"] == "included" and o["criterion_id"] in sch.get("criteria", [])]
            cap = int(sch.get("max_papers") or 0)
            targets += [{**o, "schema": name} for o in (hits[:cap] if cap else hits)]
        mapped = {c for s in cfg["extraction"]["schemas"].values() for c in s.get("criteria", [])}
        unmapped = [o for o in oc if o["outcome"] == "included" and o["criterion_id"] not in mapped]
        write_csv(cfg.path("extraction", "targets.csv"), targets)
        provenance(cfg, "document_list", {"screening_run": os.path.basename(srun),
                                          "outcomes": {k: counts[k] for k in OUTCOMES},
                                          "extraction_targets": dict(Counter(t["schema"] for t in targets)),
                                          "included_without_schema": len(unmapped)})
        print(f"[fetch] outcomes {dict(counts)}; extraction targets "
              f"{dict(Counter(t['schema'] for t in targets))}"
              + (f"; {len(unmapped)} included papers map to no schema" if unmapped else ""))

    docs = cfg.path("documents")
    os.makedirs(docs, exist_ok=True)
    drop = cfg.path("manual_pdfs")
    os.makedirs(drop, exist_ok=True)
    have = lambda pid: os.path.exists(os.path.join(docs, f"{pid}.txt"))
    tried = {r["paper_id"]: r for r in read_jsonl(cfg.path("data", "fetch_calls.jsonl"))}
    ingest_manual(drop, docs, {t["paper_id"]: t["title"] for t in targets if not have(t["paper_id"])})

    http = None if args.manual_only else Http(cfg.mailto, rate=1.0)
    log = JsonlLog(cfg.path("data", "fetch_calls.jsonl"))
    status = []
    try:
        for i, t in enumerate(targets, 1):
            pid = t["paper_id"]
            if have(pid):
                status.append({**t, "status": "ok", "source": "existing"})
                continue
            manual = os.path.join(docs, f"{pid}.pdf.manual")
            srcs = [("manual", manual)] if os.path.exists(manual) else []
            if http and (args.retry or pid not in tried):
                srcs += urls_for(http, {**by_id.get(pid, {}), **t, "openalex_id": pid})
            attempts, tmp = [], os.path.join(docs, f"{pid}.part")
            for origin, src in srcs:
                if origin == "manual":
                    shutil.copyfile(src, tmp)
                    err = ""
                else:
                    err = download(http, src, tmp)
                ok, why, info = accept(tmp, t["title"]) if not err else (False, err, {})
                attempts.append({"origin": origin, "url": src, "result": "ok" if ok else why})
                if ok:
                    os.replace(tmp, os.path.join(docs, f"{pid}.pdf"))
                    with open(os.path.join(docs, f"{pid}.txt"), "w", encoding="utf-8") as f:
                        f.write(info["text"])
                    break
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(manual):
                os.remove(manual)
            ok = have(pid)
            if srcs:
                log.write({"paper_id": pid, "ok": ok, "attempts": attempts})
            prior = tried.get(pid, {}).get("attempts", [])
            status.append({**t, "status": "ok" if ok else "missing",
                           "source": next((a["origin"] for a in attempts if a["result"] == "ok"), ""),
                           "why": "; ".join(f"{a['origin']}:{a['result']}" for a in (attempts or prior))})
            if i % 10 == 0:
                print(f"  [fetch] {i}/{len(targets)}", flush=True)
    finally:
        log.close()

    missing = [s for s in status if s["status"] != "ok"]
    write_csv(cfg.path("documents", "status.csv"), status,
              ["paper_id", "schema", "title", "doi", "status", "source", "why"])
    lines = ["# Papers to retrieve by hand", "",
             f"{len(missing)} paper(s). Download each PDF into this folder under any filename, then run "
             "`litreview fetch --manual-only`. Each PDF is matched to its paper by the title on its first "
             "pages and validated; rename a file to its paper_id to force a match.", "",
             "| | Title | DOI | Search | paper_id |", "|---|---|---|---|---|"]
    lines += [f"| ☐ | {s['title'].replace('|', '/')} | "
              + (f"[{s['doi']}](https://doi.org/{s['doi']})" if s.get("doi") else "")
              + f" | [Scholar](https://scholar.google.com/scholar?q={quote_plus(chr(34) + s['title'] + chr(34))})"
              f" | `{s['paper_id']}` |" for s in missing]
    with open(os.path.join(drop, "RETRIEVE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    provenance(cfg, "fetch_missing_abstracts" if args.missing_abstracts else "fetch",
               {"targets": len(status), "ok": len(status) - len(missing), "missing": len(missing),
                "sources": dict(Counter(s["source"] for s in status if s["status"] == "ok")),
                "requests": http.n if http else 0,
                "policy": "open-access sources only; no publisher HTML scraping; no bot-block evasion"})
    print(f"[fetch] {len(status) - len(missing)}/{len(status)} documents in hand; {len(missing)} to "
          f"retrieve by hand -> {os.path.join(drop, 'RETRIEVE.md')}")
    return 0
