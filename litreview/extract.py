"""
extract -- schema-driven full-text extraction with verbatim-evidence checking.

  litreview extract              every fetched target, batched
  litreview extract --sync --limit 2     smoke test
  litreview extract --recheck    no API: re-run the evidence checks on stored records

A schema is JSON with `blocks` -> `fields`; every field typed `evidence_object`
is extracted as {value, evidence_quote, page, confidence}, its value constrained
by `value_type` (enum | multi_enum | string | object with `sub_fields`) and
`allowed`. Its `prompt`/`note` text is shown to the model. Other fields are ignored.
One paper per request; the model never counts or compares across papers.

Every field is then checked in Python against the paper's text:
  ok / ok_loose        quote (<= quote_words) found verbatim on the cited page
                       (ok_loose: only after stripping punctuation and spacing)
  page_mismatch        found, but not on the cited page
  not_reported         value not_reported and no quote
  absence_unquoted     an absence answer (see [extraction] absence_values) with no
                       quote -- absence cannot be quoted; counted apart, neither
                       verified nor failed: sample these first when verifying
  quote_not_found / quote_too_long / missing_quote / bad_value   failures
Nothing is retried. aggregate (litreview report) counts only passing fields.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

from .common import JsonlLog, now, provenance, read_csv, read_json, run_id, write_json
from .llm import LLM, confirm, tool_input, usage

TOOL = "record_extraction"
ABSENCE = ["no", "none", "unclear", "not_applicable", "neither", "none_reported",
           "no_explicit_definition"]
FIX = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
       "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-",
       "—": "-", " ": " ", "−": "-", "­": ""}


def fields_of(schema: dict) -> list[tuple[str, dict]]:
    return [(f"{b}__{f}", spec) for b, blk in schema["blocks"].items()
            for f, spec in blk["fields"].items() if spec.get("type") == "evidence_object"]


def value_schema(spec: dict) -> dict:
    allowed = list(spec.get("allowed") or []) + (["not_reported"] if "not_reported" not in (spec.get("allowed") or []) else [])
    vt = spec.get("value_type")
    if vt == "enum":
        return {"type": "string", "enum": allowed}
    if vt == "multi_enum":
        return {"type": "array", "items": {"type": "string", "enum": allowed}}
    if vt == "object":
        subs = list(spec.get("sub_fields") or {})
        return {"type": "object", "required": subs, "properties": {k: {"type": "integer"} for k in subs}}
    return {"type": "string"}


def tool_for(fields) -> dict:
    # Not strict: real schemas exceed the strict-grammar size limit, so value
    # validity is enforced per field by check() instead.
    props = {k: {"type": "object", "required": ["value", "evidence_quote", "page", "confidence"],
                 "properties": {"value": value_schema(s), "evidence_quote": {"type": "string"},
                                "page": {"type": "integer"},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}
             for k, s in fields}
    return {"name": TOOL, "description": "Record the extraction for this one paper. Call exactly once.",
            "input_schema": {"type": "object", "required": list(props), "properties": props}}


def system_for(schema: dict, fields, quote_words: int) -> str:
    lines = [
        "You extract structured evidence from ONE research paper for a systematic comparison. "
        "You are given its full text with a [[page N]] marker before each page.",
        f"Purpose: {schema.get('purpose', '')}", "",
        f"- Quotes: a VERBATIM span of at most {quote_words} words, exactly as it appears in the "
        "text (spelling, symbols and line breaks included). Every quote is checked by a program; "
        "a quote not found counts as a failed field.",
        "- page: the N of the [[page N]] marker the quote appears under.",
        "- No supporting text: value not_reported (a list: [\"not_reported\"]; object counts: -1), "
        "evidence_quote \"\" and page 0.",
        "- Lists: every applicable value; never mix not_reported with others.",
        "- Do not compute totals, percentages or cross-paper comparisons.",
        "- Judge only from this paper's text, not outside knowledge.",
        "- confidence: high = stated explicitly; medium = clearly implied; low = ambiguous.",
        f"Answer by calling `{TOOL}` exactly once.", "", "Fields (key = BLOCK__field):"]
    for k, s in fields:
        desc = [k] + [str(s[x]) for x in ("prompt", "note", "description") if s.get(x)]
        if s.get("allowed"):
            desc.append("values: " + ", ".join(s["allowed"]))
        lines.append("- " + " -- ".join(desc))
    return "\n".join(lines)


# --- evidence checks ------------------------------------------------------------

def norm(s: str) -> str:
    for a, b in FIX.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)).strip().lower()


def loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm(s))


class Paper:
    def __init__(self, text: str):
        pages = dict((int(m.group(1)), m.group(2)) for m in
                     re.finditer(r"\[\[page (\d+)\]\]\n(.*?)(?=\n\[\[page \d+\]\]|\Z)", text, re.S))
        self.n = {p: norm(t) for p, t in pages.items()}
        self.l = {p: loose(t) for p, t in pages.items()}

    def where(self, q: str, table: dict, page: int | None) -> tuple[bool, bool]:
        """(found anywhere, found on/across the cited page)."""
        sep = " " if table is self.n else ""
        found = q in sep.join(table[p] for p in sorted(table))
        if not found or not page:
            return found, False
        a = table.get(page, "")
        return True, (q in a or q in a + sep + table.get(page + 1, "") or q in table.get(page - 1, "") + sep + a)


def unsentinel(o: dict) -> dict:
    o = dict(o)
    o["evidence_quote"] = (o.get("evidence_quote") or "").strip() or None
    o["page"] = o.get("page") or None
    if isinstance(o.get("value"), dict):
        o["value"] = {k: (None if v is None or v < 0 else v) for k, v in o["value"].items()}
    return o


def check(o: dict, paper: Paper, spec: dict, absence: set, quote_words: int) -> str:
    v, q = o.get("value"), o.get("evidence_quote")
    allowed = set(spec.get("allowed") or []) | {"not_reported"}
    vt = spec.get("value_type")
    if ((vt == "enum" and v not in allowed)
            or (vt == "multi_enum" and (not isinstance(v, list) or not v or not set(v) <= allowed
                                        or ("not_reported" in v and len(v) > 1)))
            or (vt == "object" and not isinstance(v, dict))
            or (vt not in ("enum", "multi_enum", "object") and not isinstance(v, str))):
        return "bad_value"
    unreported = v == "not_reported" or v == ["not_reported"] or (isinstance(v, dict) and all(x is None for x in v.values()))
    if unreported:
        return "not_reported" if not q else "bad_value"
    if not q:
        absent = (v in absence) if isinstance(v, str) else (isinstance(v, list) and all(x in absence for x in v))
        return "absence_unquoted" if absent else "missing_quote"
    if len(q.split()) > quote_words:
        return "quote_too_long"
    for table, key, ok in ((paper.n, norm(q), "ok"), (paper.l, loose(q), "ok_loose")):
        found, on_page = paper.where(key, table, o.get("page")) if key else (False, False)
        if found:
            return ok if on_page else "page_mismatch"
    return "quote_not_found"


# --- run ----------------------------------------------------------------------

def load_schemas(cfg) -> dict:
    out = {}
    for name, s in cfg["extraction"]["schemas"].items():
        schema = read_json(cfg.path(s["schema"]))
        if schema is None:
            sys.exit(f"schema file not found: {s['schema']}")
        out[name] = (schema, fields_of(schema))
    if not out:
        sys.exit("no [extraction.schemas.*] in project.toml")
    return out


def recheck(cfg) -> int:
    ce, schemas = cfg["extraction"], load_schemas(cfg)
    absence = set(ce.get("absence_values") or ABSENCE)
    moved = Counter()
    for path in glob.glob(cfg.path("extraction", "records", "*.json")):
        rec = read_json(path)
        specs = dict(schemas[rec["schema"]][1])
        paper = Paper(open(cfg.path("documents", f"{rec['paper_id']}.txt"), encoding="utf-8").read())
        for k, o in rec["fields"].items():
            new = check(o, paper, specs.get(k, {}), absence, ce["quote_words"])
            if new != o["check"]:
                moved[f"{o['check']}->{new}"] += 1
                o["check"] = new
        write_json(path, rec)
    provenance(cfg, "extraction_recheck", {"changed": dict(moved)})
    print(f"[extract] recheck: {sum(moved.values())} field(s) changed {dict(moved)}")
    return 0


def run(cfg, args) -> int:
    if args.recheck:
        return recheck(cfg)
    ce, schemas = cfg["extraction"], load_schemas(cfg)
    absence = set(ce.get("absence_values") or ABSENCE)
    targets = [t for t in read_csv(cfg.path("extraction", "targets.csv"))
               if os.path.exists(cfg.path("documents", f"{t['paper_id']}.txt"))]
    if args.only:
        targets = [t for t in targets if t["paper_id"] in set(args.only.split(","))]
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        sys.exit("nothing to extract -- run `litreview fetch` first")
    llm = LLM(cfg)
    tools = {n: [tool_for(f)] for n, (s, f) in schemas.items()}
    systems = {n: system_for(s, f, ce["quote_words"]) for n, (s, f) in schemas.items()}
    texts = {t["paper_id"]: open(cfg.path("documents", f"{t['paper_id']}.txt"), encoding="utf-8").read() for t in targets}
    by_id = {t["paper_id"]: t for t in targets}
    reqs = {t["paper_id"]: llm.params(systems[t["schema"]], tools[t["schema"]],
                                      f"Paper: {t['title']} ({t.get('year', '')}, {t.get('venue', '')})\n\n"
                                      f"--- BEGIN PAPER TEXT ---\n{texts[t['paper_id']]}\n--- END PAPER TEXT ---",
                                      32000) for t in targets}
    batch = not args.sync
    est = llm.estimate(reqs, int(ce["assumed_output_tokens"]), batch)
    if not confirm(len(reqs), est, ce["max_usd"], args.yes):
        return 2
    rid = run_id("extract_")
    rdir = cfg.path("extraction", "runs", rid)
    log = JsonlLog(os.path.join(rdir, "calls.jsonl"))
    checks, spent = Counter(), [0.0]

    def on_result(pid, msg, err, is_batch):
        t = by_id[pid]
        schema, fields = schemas[t["schema"]]
        parsed, failure = tool_input(msg, TOOL) if msg is not None else (None, err)
        u = usage(msg) if msg is not None else {}
        spent[0] += llm.cost(u, is_batch)
        paper, out = Paper(texts[pid]), {}
        for k, spec in fields if parsed else []:
            o = unsentinel(parsed[k]) if isinstance(parsed.get(k), dict) else {"value": None}
            o["check"] = check(o, paper, spec, absence, ce["quote_words"])
            checks[o["check"]] += 1
            out[k] = o
        log.write({"paper_id": pid, "schema": t["schema"], "failure": failure, **u,
                   "raw_response": json.loads(msg.to_json()) if msg is not None else None})
        write_json(cfg.path("extraction", "records", f"{pid}.json"), {
            "paper_id": pid, "schema": t["schema"], "title": t["title"], "doi": t.get("doi", ""),
            "year": t.get("year", ""), "venue": t.get("venue", ""), "run_id": rid, "model": llm.model,
            "response": "valid" if parsed else "invalid", "failure": failure, "extracted_utc": now(),
            "fields": out, "human_verified": False, "fields_corrected": []})

    try:
        llm.run(reqs, batch, on_result, poll=60)
    finally:
        log.close()
    result = {"run_id": rid, "model": llm.model, "papers": len(reqs), "field_checks": dict(checks),
              "estimate_usd": round(est, 4), "actual_cost_usd": round(spent[0], 4),
              "schemas": {n: s.get("schema_version", "") for n, (s, _) in schemas.items()}}
    write_json(os.path.join(rdir, "manifest.json"), result)
    provenance(cfg, "extraction", result)
    print(f"[extract] {dict(checks)}; actual ${spent[0]:.2f} (estimated ${est:.2f})")
    return 0
