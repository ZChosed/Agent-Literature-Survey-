#!/usr/bin/env python3
"""
fetch_documents.py -- retrieve the complete full text of every document-list paper.

Step one of extraction, and deliberately model-free: no LLM is called here. The
extraction stage reads only what this script assembles, so the corpus is fixed
and inspectable before any model sees it.

Targets
-------
The `included` rows of screening/final/document_list.csv: the agent papers
(I1-I4) and ALL D1 (digital twin) papers. Which D1 papers enter the report is a
separate selection decision; fetching every one costs a few requests and keeps
that decision independent of what happened to download.

Sources, legitimate open access only, tried in order until one validates
-----------------------------------------------------------------------
    1. a PDF already cached by fulltext_head.py (corpus/fulltext/)
    2. a PDF the human dropped in corpus/documents/<paper_id>.pdf (manual)
    3. OpenAlex: oa_pdf_url from candidates.csv, then every location's pdf_url
    4. Unpaywall: every oa_location's url_for_pdf
    5. Semantic Scholar Graph API by DOI: `openAccessPdf`, and the arXiv ID in
       `externalIds`
    6. Zenodo records API, for a zenodo.org/records/<id> landing URL: the
       record's listed PDF files
    7. arXiv: an arxiv.org/abs/ URL from any of the above, rewritten to /pdf/;
       failing that, an arXiv API title search (hit still title-validated)
No publisher HTML is scraped, no shadow library is used, and the user agent is
this script's own (with a mailto) -- a host that blocks it is recorded as a
manual-retrieval case, not worked around.

Validation -- a downloaded PDF is accepted only if
    * it parses with PyMuPDF and yields >= MIN_CHARS characters of text, and
    * >= TITLE_MATCH_MIN of the paper's title words (4+ letters) appear in its
      first two pages.
The second check catches the common failure where an "OA PDF" link returns a
different paper, a table of contents or a cover sheet. A rejected file is
discarded and the next source tried.

Outputs
-------
    corpus/documents/<paper_id>.pdf   the accepted PDF (gitignored: not ours to
                                      redistribute)
    corpus/documents/<paper_id>.txt   the full text, every page, each preceded
                                      by a `[[page N]]` marker for evidence
                                      citations. Deterministic: PyMuPDF only.
    corpus/document_corpus.csv        one row per target: status, source, URL,
                                      pages, characters, title match, PDF sha256
    corpus/manual_retrieval_needed.csv  every target that failed, the URLs
                                      tried, why each failed, and the exact
                                      filename to save a manual PDF as
    corpus/document_fetch_calls.jsonl append-only log of every attempt

Manual retrieval: drop the PDF into manual_pdfs/ under ANY filename and run
`python fetch_documents.py --manual-only`. Each dropped PDF is matched to the
missing paper whose title it carries on its first two pages (it must be a clear,
unambiguous best match), copied to corpus/documents/<paper_id>.pdf, and then
goes through the same validation as a download, recorded as `source=manual`.
A file named <paper_id>.pdf is matched by name instead. Files that match
nothing, or match ambiguously, are reported and left alone.

Usage
-----
    python fetch_documents.py --mailto zchosed@umich.edu
    python fetch_documents.py --mailto zchosed@umich.edu --retry-failed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

try:
    import pymupdf as fitz
except ImportError:
    sys.exit("PyMuPDF is required:  pip install PyMuPDF")

from fulltext_head import DEHYPHEN, Fetcher, download_pdf

SCRIPT_VERSION = "1.2.0"
OPENALEX = "https://api.openalex.org"
UNPAYWALL = "https://api.unpaywall.org/v2"
S2 = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_API = "http://export.arxiv.org/api/query"

MIN_CHARS = 3000
TITLE_MATCH_MIN = 0.6

DOC_COLUMNS = ["paper_id", "doi", "title", "role", "criterion_id", "rank",
               "status", "source", "url", "pages", "chars", "title_match",
               "pdf_sha256", "failure_reason"]
MANUAL_COLUMNS = ["paper_id", "doi", "doi_url", "title", "year", "venue", "role",
                  "save_as", "urls_tried", "why_failed"]

csv.field_size_limit(10_000_000)


def title_words(title: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", title.lower())]


def title_match(title: str, text: str) -> float:
    words = set(title_words(title))
    if not words:
        return 1.0
    hay = re.sub(r"\s+", " ", text.lower())
    return sum(1 for w in words if w in hay) / len(words)


def title_phrase_match(title: str, text: str) -> float:
    """
    Fraction of the title's consecutive word PAIRS that appear in `text`.

    Used to pick which missing paper a dropped PDF is. Single words are too
    common across this corpus ("digital", "twin", "manufacturing", "agent") to
    tell 60 candidate titles apart; word pairs from the actual title rarely
    occur in another paper's first pages.
    """
    tw = re.findall(r"[a-z0-9]+", title.lower())
    pairs = {(tw[i], tw[i + 1]) for i in range(len(tw) - 1)}
    if not pairs:
        return 0.0
    xw = re.findall(r"[a-z0-9]+", text.lower())
    have = {(xw[i], xw[i + 1]) for i in range(len(xw) - 1)}
    return sum(1 for p in pairs if p in have) / len(pairs)


def extract(path: str) -> tuple[str, int, str]:
    """(full text with [[page N]] markers, page count, first-two-pages text)."""
    parts, first2 = [], []
    with fitz.open(path) as doc:
        n = len(doc)
        for i in range(n):
            t = DEHYPHEN.sub(r"\1\2", doc[i].get_text("text") or "")
            parts.append(f"[[page {i + 1}]]\n{t.strip()}\n")
            if i < 2:
                first2.append(t)
    return "\n".join(parts), n, "\n".join(first2)


def validate_pdf(path: str, title: str) -> tuple[bool, str, dict]:
    try:
        text, pages, first2 = extract(path)
    except Exception as exc:                       # noqa: BLE001
        return False, f"unparseable:{type(exc).__name__}", {}
    body = re.sub(r"\[\[page \d+\]\]", "", text)
    chars = len(body.strip())
    tm = round(title_match(title, first2), 3)
    info = {"text": text, "pages": pages, "chars": chars, "title_match": tm}
    if chars < MIN_CHARS:
        return False, f"too_little_text:{chars}", info
    if tm < TITLE_MATCH_MIN:
        return False, f"title_mismatch:{tm}", info
    return True, "", info


def arxiv_pdf(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/([^\s?#]+)", url or "")
    return f"https://arxiv.org/pdf/{m.group(1)}" if m else None


def openalex_locations(fetcher: Fetcher, ids: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        try:
            r = fetcher.get(f"{OPENALEX}/works", params={
                "filter": "openalex:" + "|".join(chunk),
                "select": "id,locations", "per-page": len(chunk),
                "mailto": fetcher.mailto})
            data = r.json() if r.status_code == 200 else {}
        except Exception:                              # noqa: BLE001
            data = {}
        for w in data.get("results") or []:
            out[(w.get("id") or "").rsplit("/", 1)[-1]] = w.get("locations") or []
    return out


def unpaywall_locations(fetcher: Fetcher, doi: str) -> list[dict]:
    try:
        r = fetcher.get(f"{UNPAYWALL}/{doi}", params={"email": fetcher.mailto})
        if r.status_code != 200:
            return []
        return (r.json() or {}).get("oa_locations") or []
    except Exception:                                  # noqa: BLE001
        return []


def _get_json(fetcher: Fetcher, url: str, **kw):
    """GET -> parsed JSON or None; one polite retry on 429."""
    import time
    for attempt in range(2):
        try:
            r = fetcher.get(url, **kw)
        except Exception:                              # noqa: BLE001
            return None
        if r.status_code == 429 and attempt == 0:
            time.sleep(5)
            continue
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def s2_urls(fetcher: Fetcher, doi: str) -> list[tuple[str, str]]:
    if not doi:
        return []
    d = _get_json(fetcher, f"{S2}/DOI:{doi}",
                  params={"fields": "openAccessPdf,externalIds"}) or {}
    out = []
    pdf = (d.get("openAccessPdf") or {}).get("url")
    if pdf:
        out.append(("semantic_scholar", pdf))
    ax = (d.get("externalIds") or {}).get("ArXiv")
    if ax:
        out.append(("arxiv", f"https://arxiv.org/pdf/{ax}"))
    return out


def zenodo_urls(fetcher: Fetcher, landing_urls: list[str]) -> list[tuple[str, str]]:
    out = []
    for u in landing_urls:
        m = re.search(r"zenodo\.org/(?:records?|record)/(\d+)", u or "")
        if not m:
            continue
        d = _get_json(fetcher, f"https://zenodo.org/api/records/{m.group(1)}") or {}
        for fobj in d.get("files") or []:
            key = fobj.get("key") or ""
            link = (fobj.get("links") or {}).get("self") or ""
            if key.lower().endswith(".pdf") and link:
                out.append(("zenodo", link))
    return out


def arxiv_title_search(fetcher: Fetcher, title: str) -> list[tuple[str, str]]:
    words = title_words(title)
    if len(words) < 3:
        return []
    q = " AND ".join(f"ti:{w}" for w in words[:8])
    try:
        r = fetcher.get(ARXIV_API, params={"search_query": q, "max_results": 3})
        if r.status_code != 200:
            return []
    except Exception:                                  # noqa: BLE001
        return []
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
        t = re.search(r"<title>(.*?)</title>", entry, re.S)
        i = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", entry)
        # Keep only hits whose own title closely matches; the PDF is then
        # title-validated again before acceptance.
        if t and i and title_match(title, t.group(1)) >= 0.8:
            out.append(("arxiv_search", f"https://arxiv.org/pdf/{i.group(1).strip()}"))
    return out


def candidate_urls(cand: dict, oa_locs: list[dict], up_locs: list[dict]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []

    def add(origin, url):
        if url and url not in [u for _, u in urls]:
            urls.append((origin, url))

    add("openalex_best", (cand.get("oa_pdf_url") or "").strip())
    for loc in oa_locs:
        add("openalex_location", loc.get("pdf_url"))
    for loc in up_locs:
        add("unpaywall", loc.get("url_for_pdf"))
    for origin, url in list(urls):
        add("arxiv", arxiv_pdf(url))
    for loc in oa_locs:
        add("arxiv", arxiv_pdf(loc.get("landing_page_url") or ""))
    for loc in up_locs:
        add("arxiv", arxiv_pdf(loc.get("url") or ""))
    return urls


def ingest_manual_dir(manual_dir: str, targets: list[dict], out_dir: str,
                      done: set[str]) -> list[tuple[str, str, str]]:
    """
    Match PDFs in manual_dir to the targets still missing a document, and copy
    each match to out_dir/<paper_id>.pdf. Returns (file, paper_id or '', note).

    By name first (<paper_id>.pdf). Otherwise by title phrase (word pairs, see
    title_phrase_match): the best-scoring missing paper must reach
    TITLE_MATCH_MIN and beat the runner-up by 0.2, so a file
    that could be either of two similar titles is not guessed at.
    """
    if not os.path.isdir(manual_dir):
        return []
    missing = {t["paper_id"]: t for t in targets if t["paper_id"] not in done}
    report = []
    for name in sorted(os.listdir(manual_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(manual_dir, name)
        stem = os.path.splitext(name)[0]
        if stem in missing:
            pid, note = stem, "matched by filename"
        else:
            try:
                _, _, first2 = extract(path)
            except Exception as exc:                   # noqa: BLE001
                report.append((name, "", f"unreadable PDF ({type(exc).__name__})"))
                continue
            scores = sorted(((title_phrase_match(t["title"], first2), p)
                             for p, t in missing.items()), reverse=True)
            if not scores or scores[0][0] < TITLE_MATCH_MIN:
                report.append((name, "", "matches no missing paper's title"))
                continue
            if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.2:
                report.append((name, "", f"ambiguous: {scores[0][1]} vs {scores[1][1]}"))
                continue
            pid, note = scores[0][1], f"matched by title ({scores[0][0]:.2f})"
        shutil.copyfile(path, os.path.join(out_dir, pid + ".pdf"))
        report.append((name, pid, note))
        del missing[pid]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch full texts for the document list.")
    ap.add_argument("--document-list", default="screening/final/document_list.csv")
    ap.add_argument("--candidates", default="corpus/candidates.csv")
    ap.add_argument("--out-dir", default="corpus/documents")
    ap.add_argument("--cache-dir", default="corpus/fulltext")
    ap.add_argument("--corpus-csv", default="corpus/document_corpus.csv")
    ap.add_argument("--manual-csv", default="corpus/manual_retrieval_needed.csv")
    ap.add_argument("--log", default="corpus/document_fetch_calls.jsonl")
    ap.add_argument("--provenance", default="corpus/provenance.json")
    ap.add_argument("--mailto", default=None,
                    help="required unless --manual-only")
    ap.add_argument("--manual-dir", default="manual_pdfs",
                    help="drop folder for manually retrieved PDFs, any filename")
    ap.add_argument("--manual-only", action="store_true",
                    help="no network: only ingest and validate manual PDFs")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt targets that failed last time (an "
                         "accepted document is never re-downloaded)")
    args = ap.parse_args()
    if not args.manual_only and not args.mailto:
        sys.exit("--mailto is required unless --manual-only")

    targets = [r for r in csv.DictReader(open(args.document_list, encoding="utf-8"))]
    cands = {r["openalex_id"]: r for r in csv.DictReader(open(args.candidates, encoding="utf-8"))}
    os.makedirs(args.out_dir, exist_ok=True)

    prior = {}
    if os.path.exists(args.corpus_csv):
        prior = {r["paper_id"]: r for r in csv.DictReader(open(args.corpus_csv, encoding="utf-8"))}

    done = {pid for pid, r in prior.items() if r.get("status") == "ok"
            and os.path.exists(os.path.join(args.out_dir, pid + ".txt"))}
    ingested = ingest_manual_dir(args.manual_dir, targets, args.out_dir, done)
    for name, pid, note in ingested:
        print(f"  [manual] {name} -> {pid or '(not used)'}: {note}")

    fetcher = Fetcher(args.mailto or "manual-only@localhost", args.rate)
    started = datetime.now(timezone.utc)
    need_net = [] if args.manual_only else [
        t for t in targets
        if not os.path.exists(os.path.join(args.out_dir, t["paper_id"] + ".txt"))]
    oa_locs = openalex_locations(fetcher, [t["paper_id"] for t in need_net]) if need_net else {}

    rows, manual = [], []
    log = open(args.log, "a", encoding="utf-8")
    try:
        for i, t in enumerate(targets, 1):
            pid, title = t["paper_id"], t["title"]
            role = "dt" if t["criterion_id"] == "D1" else "agent"
            final_pdf = os.path.join(args.out_dir, pid + ".pdf")
            final_txt = os.path.join(args.out_dir, pid + ".txt")
            attempts: list[dict] = []
            accepted = None

            # Already accepted on an earlier run: keep it, do not re-download.
            if os.path.exists(final_txt) and os.path.exists(final_pdf) \
                    and (prior.get(pid) or {}).get("status") == "ok":
                p = prior[pid]
                rows.append({**{k: p.get(k, "") for k in DOC_COLUMNS}})
                continue
            if (prior.get(pid) or {}).get("status") == "failed" \
                    and (not args.retry_failed or args.manual_only) \
                    and not os.path.exists(final_pdf):
                rows.append({**{k: prior[pid].get(k, "") for k in DOC_COLUMNS}})
                manual.append(None)    # rebuilt below from prior
                continue

            sources: list[tuple[str, str]] = []
            if os.path.exists(final_pdf):
                sources.append(("manual", final_pdf))
            cached = os.path.join(args.cache_dir, pid + ".pdf")
            if os.path.exists(cached):
                sources.append(("fulltext_head_cache", cached))
            net: list[tuple[str, str]] = []
            if not args.manual_only:
                up = unpaywall_locations(fetcher, t["doi"]) if t["doi"] else []
                net = candidate_urls(cands.get(pid, {}), oa_locs.get(pid, []), up)
                landings = ([loc.get("landing_page_url") or "" for loc in oa_locs.get(pid, [])]
                            + [loc.get("url") or "" for loc in up] + [u for _, u in net])
                more = s2_urls(fetcher, t["doi"]) + zenodo_urls(fetcher, landings)
                for o, u in more:
                    if u not in [x for _, x in net]:
                        net.append((o, u))
                if not any(o.startswith("arxiv") for o, _ in net):
                    net += arxiv_title_search(fetcher, title)
            sources += net

            tmp = final_pdf + ".part"
            for origin, src in sources:
                if origin in ("manual", "fulltext_head_cache"):
                    shutil.copyfile(src, tmp)
                    err, status = "", None
                else:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    err, status = download_pdf(fetcher, src, tmp)
                if err:
                    attempts.append({"origin": origin, "url": src, "result": err,
                                     "http_status": status})
                    continue
                ok, why, info = validate_pdf(tmp, title)
                attempts.append({"origin": origin, "url": src,
                                 "result": "ok" if ok else why, "http_status": status})
                if ok:
                    accepted = (origin, src, info)
                    break
            if accepted:
                origin, src, info = accepted
                os.replace(tmp, final_pdf)
                with open(final_txt, "w", encoding="utf-8") as f:
                    f.write(info["text"])
                sha = hashlib.sha256(open(final_pdf, "rb").read()).hexdigest()
                rows.append({"paper_id": pid, "doi": t["doi"], "title": title,
                             "role": role, "criterion_id": t["criterion_id"],
                             "rank": t["rank"], "status": "ok", "source": origin,
                             "url": src if origin != "manual" else "", "pages": info["pages"],
                             "chars": info["chars"], "title_match": info["title_match"],
                             "pdf_sha256": sha, "failure_reason": ""})
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if os.path.exists(final_pdf) and sources and sources[0][0] == "manual":
                    os.replace(final_pdf, final_pdf + ".rejected")
                why = ("no open-access PDF URL known" if not sources else
                       "; ".join(sorted({a["result"].split(":")[0] for a in attempts})))
                rows.append({"paper_id": pid, "doi": t["doi"], "title": title,
                             "role": role, "criterion_id": t["criterion_id"],
                             "rank": t["rank"], "status": "failed", "source": "",
                             "url": "", "pages": "", "chars": "", "title_match": "",
                             "pdf_sha256": "", "failure_reason": why})
                manual.append({"paper_id": pid, "doi": t["doi"],
                               "doi_url": f"https://doi.org/{t['doi']}" if t["doi"] else "",
                               "title": title, "year": t["year"], "venue": t["venue"],
                               "role": role, "save_as": final_pdf,
                               "urls_tried": " | ".join(a["url"] for a in attempts),
                               "why_failed": " | ".join(
                                   f"{a['origin']}:{a['result']}" for a in attempts)
                               or "no open-access PDF URL known"})
            log.write(json.dumps({"paper_id": pid, "doi": t["doi"],
                                  "status": rows[-1]["status"], "attempts": attempts,
                                  "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
            log.flush()
            if i % 10 == 0:
                print(f"  [fetch] {i}/{len(targets)}  ok="
                      f"{sum(r['status'] == 'ok' for r in rows)}", flush=True)
    finally:
        log.close()

    # Rebuild the manual list from scratch so it always matches the corpus CSV.
    prior_manual = {}
    if os.path.exists(args.manual_csv):
        prior_manual = {r["paper_id"]: r for r in csv.DictReader(open(args.manual_csv, encoding="utf-8"))}
    manual_rows = []
    fresh = {m["paper_id"]: m for m in manual if m}
    for r in rows:
        if r["status"] == "failed":
            manual_rows.append(fresh.get(r["paper_id"]) or prior_manual.get(r["paper_id"])
                               or {"paper_id": r["paper_id"], "doi": r["doi"],
                                   "title": r["title"], "role": r["role"],
                                   "save_as": os.path.join(args.out_dir, r["paper_id"] + ".pdf")})

    with open(args.corpus_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DOC_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(args.manual_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANUAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(manual_rows)

    by = Counter((r["role"], r["status"]) for r in rows)
    block = {
        "script": "fetch_documents.py", "script_version": SCRIPT_VERSION,
        "run_started_utc": started.isoformat(),
        "targets": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "by_role_status": {f"{k[0]}:{k[1]}": v for k, v in sorted(by.items())},
        "ok_by_source": dict(Counter(r["source"] for r in rows if r["status"] == "ok")),
        "validation": {"min_chars": MIN_CHARS, "title_match_min": TITLE_MATCH_MIN},
        "requests": fetcher.requests_made,
        "publisher_html_scraped": False, "shadow_libraries_used": False,
        "model_calls": 0,
    }
    prov = json.load(open(args.provenance, encoding="utf-8")) if os.path.exists(args.provenance) else {}
    prov["document_fetch"] = block
    with open(args.provenance, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    print(f"\n[done] {block['ok']}/{block['targets']} documents retrieved and "
          f"validated, {block['failed']} need manual retrieval")
    print("[done] " + ", ".join(f"{k}={v}" for k, v in block["by_role_status"].items()))
    print(f"[done] sources: {block['ok_by_source']}  ({fetcher.requests_made} requests)")
    print(f"  -> {args.corpus_csv}\n  -> {args.manual_csv}\n  -> {args.out_dir}/ (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
