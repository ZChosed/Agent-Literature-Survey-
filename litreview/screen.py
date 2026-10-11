"""
screen -- title/abstract screening against criteria.md, one paper per request.

  litreview screen --sample     the calibration sample (screening/calibration_sample.csv)
  litreview screen              every eligible candidate

The model sees a title plus ONE of: the abstract; the first 5,000 characters of
the full text (labelled as such) if `litreview fetch --missing-abstracts` found
one; or nothing more. Title-only papers get NO model call: they are logged as a
forced `uncertain` and routed to a human. Ineligible papers are never sent and
never forced -- they are out of scope, and are listed in the run manifest.

Each response is a tool call {decision, criterion_id, reason}. Validation:
decision in include/exclude/uncertain; criterion_id one of the IDs parsed from
criteria.md (enforced when >= 2 are found); reason over `reason_words` is a
logged warning, at `reason_fail_words` or more a failure. Failures are never
retried: a retry would make the malformed-output rate unmeasurable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

from . import eligibility as elig
from .common import JsonlLog, now, read_csv, run_id, write_json
from .llm import LLM, confirm, tool_input, usage

TOOL = "screen_paper"
DECISIONS = ("include", "exclude", "uncertain")
HEAD_CHARS = 5000
ID_PATTERNS = [re.compile(p, re.M) for p in (
    r"^#{1,6}\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\b",
    r"^\s*[-*+]\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\b",
    r"^\s*\|\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\**\s*\|",
    r"^\s*\*\*([A-Z]{1,3}[-_]?\d{1,3})\*\*",
    r"^\s*([A-Z]{1,3}[-_]?\d{1,3})\s*[—–-]\s+")]
FORBIDDEN = ("seed_refs", "seed_roles", "seed_weight", "linked_to_core", "stratum")

SYSTEM = """\
You are screening papers for a structured literature review{topic}. You decide \
whether each paper should be included.

You are given a title plus ONE of: an abstract, or the first part of the paper's \
full text, or nothing beyond the title. The user message says which. Judge only \
from the text you are given; do not use outside knowledge of the paper, its \
authors, its venue or its field.

Apply the criteria below exactly as written, including their decision procedure \
and their rules about `uncertain`. The criteria are the only authority on what \
each decision means.

Answer by calling the `{tool}` tool exactly once; a text reply is discarded. Return:
  - decision: include, exclude or uncertain
  - criterion_id: the ID, from the criteria, of the single criterion that most \
drove the decision
  - reason: at most {words} words referring to the text you were given

--- BEGIN CRITERIA ---
{criteria}
--- END CRITERIA ---
"""
LABELS = {"abstract": "Abstract",
          "fulltext_head": "Full-text head (the first 5,000 characters of the paper, truncated "
                           "mid-sentence; NOT an abstract -- it typically holds title-page matter "
                           "and the start of the introduction, so absence of a finding here is "
                           "not evidence the paper lacks it)"}


def criterion_ids(text: str) -> list[str]:
    out = []
    for p in ID_PATTERNS:
        for m in p.finditer(text):
            if m.group(1) not in out:
                out.append(m.group(1))
    return out


def screening_text(cfg, row: dict) -> tuple[str, str]:
    if (row.get("abstract") or "").strip():
        return row["abstract"].strip(), "abstract"
    txt = cfg.path("documents", f"{row['openalex_id']}.txt")
    if os.path.exists(txt):
        head = re.sub(r"\[\[page \d+\]\]\n?", "", open(txt, encoding="utf-8").read())[:HEAD_CHARS]
        if head.strip():
            return head.strip(), "fulltext_head"
    return "", "title_only"


def user_text(title: str, text: str, kind: str) -> str:
    return f"Title: {title.strip()}\n\n{LABELS[kind]}:\n{text}"


def validate(parsed: dict | None, failure: str | None, ids: set | None, sc: dict) -> tuple[str, list, list]:
    if failure or parsed is None:
        return "invalid", [failure or "no_tool_use"], []
    errs, warns = [], []
    if parsed.get("decision") not in DECISIONS:
        errs.append(f"bad_decision:{parsed.get('decision')!r}")
    cid = parsed.get("criterion_id")
    if not isinstance(cid, str) or not cid.strip():
        errs.append("empty_criterion_id")
    elif ids is not None and cid.strip() not in ids:
        errs.append(f"unknown_criterion_id:{cid!r}")
    reason = parsed.get("reason")
    n = len(reason.split()) if isinstance(reason, str) else 0
    if not n:
        errs.append("empty_reason")
    elif n >= sc["reason_fail_words"]:
        errs.append(f"reason_too_long:{n}w")
    elif n > sc["reason_words"]:
        warns.append(f"reason_overrun:{n}w")
    if set(parsed) - {"decision", "criterion_id", "reason"}:
        errs.append("unexpected_keys")
    return ("invalid" if errs else "valid"), errs, warns


def run(cfg, args) -> int:
    sc = cfg["screening"]
    rows = read_csv(cfg.candidates)
    by_id = {r["openalex_id"]: r for r in rows}
    eg = elig.load(cfg, rows)
    if args.sample:
        path = cfg.path("screening", "calibration_sample.csv")
        if not os.path.exists(path):
            sys.exit("no calibration sample -- run `litreview calibrate draw`")
        ids = [r["paper_id"] for r in read_csv(path)]
    else:
        ids = [r["openalex_id"] for r in rows]
    if args.only:
        ids = [i for i in ids if i in set(args.only.split(","))]
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        sys.exit(f"{len(unknown)} paper id(s) not in candidates.csv: {unknown[:5]}")
    ineligible = {i: eg[i]["exclusion_reason"] for i in ids if not eg[i]["eligible"]}
    papers = [by_id[i] for i in ids if eg[i]["eligible"]]
    if args.limit:
        papers = papers[:args.limit]

    criteria = open(cfg.path(sc["criteria"]), encoding="utf-8").read()
    ids_known = criterion_ids(criteria)
    enforce = set(ids_known) if len(ids_known) >= 2 else None
    topic = f" on {sc['topic']}" if sc.get("topic") else ""
    system = SYSTEM.format(topic=topic, tool=TOOL, words=sc["reason_words"], criteria=criteria.strip())
    tool = {"name": TOOL, "strict": True, "description": "Record the screening decision. Call exactly once.",
            "input_schema": {"type": "object", "additionalProperties": False,
                             "required": ["decision", "criterion_id", "reason"],
                             "properties": {"decision": {"type": "string", "enum": list(DECISIONS)},
                                            "criterion_id": {"type": "string"},
                                            "reason": {"type": "string"}}}}
    kinds = {p["openalex_id"]: screening_text(cfg, p) for p in papers}
    to_send = [p for p in papers if kinds[p["openalex_id"]][1] != "title_only"]
    forced = [p for p in papers if kinds[p["openalex_id"]][1] == "title_only"]
    print(f"[screen] {len(ids)} papers: {len(ineligible)} ineligible (not sent), "
          f"{len(forced)} title-only (forced uncertain), {len(to_send)} to screen; "
          f"criterion IDs {ids_known} ({'enforced' if enforce else 'NOT enforced: fewer than 2 parsed'})")

    llm = LLM(cfg)
    reqs = {}
    for p in to_send:
        text, kind = kinds[p["openalex_id"]]
        u = user_text(p["title"], text, kind)
        for col in FORBIDDEN:
            v = str(p.get(col) or "").strip()
            if len(v) >= 4 and v in u:
                raise AssertionError(f"isolation: {col} value reached the prompt for {p['openalex_id']}")
        reqs[p["openalex_id"]] = llm.params(system, [tool], u, 4096)
    batch = not args.sync
    est = llm.estimate(reqs, 640, batch) if reqs else 0.0
    if not confirm(len(reqs), est, sc["max_usd"], args.yes):
        return 2

    rid = run_id("sample_" if args.sample else "corpus_")
    rdir = cfg.path("screening", "runs", rid)
    log = JsonlLog(os.path.join(rdir, "calls.jsonl"))
    totals = Counter()
    spent = [0.0]
    for p in forced:
        log.write({"paper_id": p["openalex_id"], "text_kind": "title_only", "validation_status": "forced",
                   "parsed": {"decision": "uncertain", "criterion_id": "", "reason": "title only; forced, no model call"},
                   "validation_errors": [], "validation_warnings": [], "cost_usd": 0.0})
        totals["forced"] += 1

    def on_result(cid, msg, err, is_batch):
        parsed, failure = tool_input(msg, TOOL) if msg is not None else (None, err)
        st, errs, warns = validate(parsed, failure, enforce, sc)
        u = usage(msg) if msg is not None else {}
        c = llm.cost(u, is_batch)
        spent[0] += c
        totals[st] += 1
        totals.update(w.split(":")[0] for w in warns)
        log.write({"paper_id": cid, "model": llm.model, "text_kind": kinds[cid][1], "parsed": parsed,
                   "validation_status": st, "validation_errors": errs, "validation_warnings": warns,
                   **u, "cost_usd": round(c, 8), "raw_response": json.loads(msg.to_json()) if msg is not None else None})

    try:
        llm.run(reqs, batch, on_result)
    finally:
        log.close()
    write_json(os.path.join(rdir, "manifest.json"), {
        "run_id": rid, "mode": "sample" if args.sample else "corpus", "model": llm.model,
        "ended_utc": now(), "api_mode": "batch" if batch else "sync",
        "criteria_sha256": hashlib.sha256(criteria.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "criterion_ids": ids_known, "criterion_ids_enforced": enforce is not None,
        "reason_words": sc["reason_words"], "reason_fail_words": sc["reason_fail_words"],
        "papers": len(ids), "screened": len(reqs), "forced_title_only": len(forced),
        "excluded_at_eligibility": {"n": len(ineligible), "paper_ids": ineligible},
        "estimate_usd": round(est, 4), "actual_cost_usd": round(spent[0], 4), "counts": dict(totals)})
    print(f"[screen] {dict(totals)}; actual ${spent[0]:.4f} (estimated ${est:.4f})")
    print(f"  -> {rdir}")
    return 0
