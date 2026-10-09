#!/usr/bin/env python3
"""
extract.py -- schema-driven full-text extraction with verbatim-evidence checking.

Governing rule (proposal S3.3): LLMs extract; Python counts. One paper per
request, no history, so nothing crosses between papers (`one_paper_per_context`).
The model fills evidence objects; it never computes a total, a percentage or a
comparison. aggregate.py does all of that from the records written here.

What the model is asked for
---------------------------
Only the evidence fields. For each schema the tool input schema is generated
from schemas/*_extraction_schema.json: every `evidence_object` field becomes

    {value, evidence_quote, page, confidence}

with `value` constrained by the field's own type -- an enum, a list of enum
members, free text, or (system_scale) an object of nullable integers. Every
enum gains `not_reported`, which the contract requires when the paper has no
supporting text. IDENT and J_EXTRACTION_META are NOT asked of the model: IDENT
is copied from the corpus, the metadata is written by this script, and the DT
`conformance_score` (type `computed`) is computed in Python by aggregate.py.

What is checked, in Python, on every field
------------------------------------------
    not_reported         value is not_reported (or ["not_reported"], or all-null
                         scale) AND the quote is null
    ok                   the quote is <= 25 words AND found verbatim in the
                         paper's text AND on the cited page (or across that page
                         boundary)
    ok_loose             found only after stripping punctuation/spacing -- PDF
                         text extraction mangles these; recorded separately so
                         the rate is visible
    page_mismatch        found verbatim, but not on the cited page
    quote_not_found      not in the paper at all: an invented or paraphrased
                         quote, the failure this contract exists to catch
    quote_too_long       over 25 words
    absence_unquoted     an ABSENCE answer (no / none / unclear / not_applicable /
                         no_dt / neither ...) with no quote. Absence cannot be
                         quoted, so the quote rule cannot apply; these are
                         counted apart -- neither verified nor failed -- and
                         are the first thing a human verification sample
                         should cover
    missing_quote        any other value with no quote
    bad_value            not_reported mixed with real values, etc.

Nothing is retried. A field that fails stays failed and is counted; the report
states the rate.

Usage
-----
    python extract.py --sync --limit 1 --role agent      # smoke test
    python extract.py --max-usd 20                       # batch, all retrieved docs
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
    import anthropic
except ImportError:
    sys.exit("anthropic is required:  pip install anthropic")

import screen as sc

SCRIPT_VERSION = "1.1.0"   # 1.1.0: absence set + soft hyphen; --recheck
MODEL = sc.MODEL_IDS["opus"]
MAX_QUOTE_WORDS = 25
MAX_TOKENS = 32000
ASSUMED_OUTPUT_TOKENS = 9000        # estimate only; actual usage is what is billed
SCHEMAS = {"agent": "schemas/agent_extraction_schema.json",
           "dt": "schemas/dt_extraction_schema.json"}
NOT_EXTRACTED_BLOCKS = {"IDENT", "J_EXTRACTION_META"}
TOOL_NAME = "record_extraction"

csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------
# Schema -> fields -> tool
# --------------------------------------------------------------------------

def evidence_fields(schema: dict) -> list[tuple[str, str, dict]]:
    """(block, field, spec) for every field the model fills."""
    out = []
    for b, blk in schema["blocks"].items():
        if b in NOT_EXTRACTED_BLOCKS:
            continue
        for f, spec in blk["fields"].items():
            if spec.get("type") == "evidence_object":
                out.append((b, f, spec))
    return out


def value_schema(spec: dict) -> dict:
    vt = spec.get("value_type")
    allowed = list(spec.get("allowed") or [])
    if vt in ("enum", "multi_enum") and "not_reported" not in allowed:
        allowed.append("not_reported")
    if vt == "enum":
        return {"type": "string", "enum": allowed}
    if vt == "multi_enum":
        return {"type": "array", "items": {"type": "string", "enum": allowed}}
    if vt == "object":
        subs = spec.get("sub_fields") or {}
        # -1 = not reported (strict tool schemas cap nullable parameters at 16;
        # sentinels are converted back to null by from_sentinels()).
        return {"type": "object", "additionalProperties": False,
                "required": list(subs),
                "properties": {k: {"type": "integer"} for k in subs}}
    return {"type": "string"}


def build_tool(fields: list[tuple[str, str, dict]]) -> dict:
    props = {}
    for b, f, spec in fields:
        props[f"{b}__{f}"] = {
            "type": "object", "additionalProperties": False,
            "required": ["value", "evidence_quote", "page", "confidence"],
            "properties": {
                "value": value_schema(spec),
                "evidence_quote": {"type": "string"},
                "page": {"type": "integer"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            }}
    # NOT strict: these schemas (51 / 32 evidence objects) exceed the API's
    # strict-grammar size limit ("compiled grammar is too large"). Value
    # validity is enforced in Python by check_field() instead.
    return {"name": TOOL_NAME,
            "description": "Record the extraction for this one paper. Call exactly once.",
            "input_schema": {"type": "object", "additionalProperties": False,
                             "required": list(props), "properties": props}}


def build_system(schema: dict, fields) -> str:
    c = schema["extraction_contract"]
    lines = [
        "You extract structured evidence from ONE manufacturing research paper "
        "for a systematic comparison. You are given the paper's full text, "
        "extracted from its PDF, with a [[page N]] marker before each page.",
        "",
        f"Purpose: {schema.get('purpose', '')}",
        "",
        "Rules (the extraction contract):",
        f"- {c.get('evidence_rule', '')}",
        f"- Quotes: copy a VERBATIM span of at most {MAX_QUOTE_WORDS} words, exactly as "
        "it appears in the text you were given (keep its spelling, symbols and "
        "line-broken words as they appear). Do not paraphrase, join separate "
        "sentences, or fix typos. Every quote is checked against the text by a "
        "program; a quote that is not found counts as a failed field.",
        "- page: the N of the [[page N]] marker under which the quote appears.",
        "- If the paper gives no supporting text, value is not_reported "
        "(or [\"not_reported\"] for a list; -1 for each system_scale count "
        "not reported), evidence_quote is the empty string \"\" and page is 0. "
        "(These stand for null; the tool schema cannot express null.)",
        "- Lists: give every applicable value; never mix not_reported with others.",
        f"- {c.get('no_counting_rule', '')}",
        "- Judge only from this paper's text. Do not use outside knowledge of "
        "the paper, its authors or its venue.",
        "- confidence: high = stated explicitly; medium = clearly implied by the "
        "quoted text; low = a reading of ambiguous text.",
        "",
        f"Answer by calling the `{TOOL_NAME}` tool exactly once. A text reply "
        "is discarded.",
        "",
        "Fields (tool key = BLOCK__field):",
    ]
    for b, f, spec in fields:
        blk = schema["blocks"][b]
        desc = [f"{b}__{f}"]
        for k in ("prompt", "note", "description"):
            if spec.get(k):
                desc.append(f"{spec[k]}")
        if spec.get("allowed"):
            desc.append("values: " + ", ".join(spec["allowed"]))
        lines.append(f"- [{blk.get('label', b)}] " + " -- ".join(desc))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verbatim checking
# --------------------------------------------------------------------------

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
             "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"',
             "”": '"', "–": "-", "—": "-", " ": " ",
             "−": "-", "­": ""}   # soft hyphen: PDFs embed it mid-word


def norm(s: str) -> str:
    for a, b in LIGATURES.items():
        s = s.replace(a, b)
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm(s))


def split_pages(text: str) -> dict[int, str]:
    pages = {}
    for m in re.finditer(r"\[\[page (\d+)\]\]\n(.*?)(?=\n\[\[page \d+\]\]|\Z)", text, re.S):
        pages[int(m.group(1))] = m.group(2)
    return pages


class PaperText:
    def __init__(self, text: str):
        self.pages = split_pages(text)
        self.n = {p: norm(t) for p, t in self.pages.items()}
        self.l = {p: loose(t) for p, t in self.pages.items()}
        self.all_n = " ".join(self.n[p] for p in sorted(self.n))
        self.all_l = "".join(self.l[p] for p in sorted(self.l))

    def on_page(self, q: str, page: int | None, table: dict) -> bool:
        if page is None:
            return False
        a = table.get(page, "")
        b = table.get(page + 1, "")
        sep = " " if table is self.n else ""
        return q in a or q in (a + sep + b) or q in (table.get(page - 1, "") + sep + a)


def from_sentinels(obj: dict) -> dict:
    """Map the tool's sentinels back to the contract's nulls."""
    out = dict(obj)
    if not (out.get("evidence_quote") or "").strip():
        out["evidence_quote"] = None
    if not out.get("page"):
        out["page"] = None
    if isinstance(out.get("value"), dict):
        out["value"] = {k: (None if (v is None or v < 0) else v)
                        for k, v in out["value"].items()}
    return out


def is_unreported(value) -> bool:
    if value == "not_reported":
        return True
    if isinstance(value, list) and value == ["not_reported"]:
        return True
    if isinstance(value, dict) and all(v is None for v in value.values()):
        return True
    return False


def value_ok(value, spec: dict) -> bool:
    """The value has the shape and vocabulary the schema allows."""
    vt = spec.get("value_type")
    allowed = set(spec.get("allowed") or []) | {"not_reported"}
    if vt == "enum":
        return isinstance(value, str) and value in allowed
    if vt == "multi_enum":
        return (isinstance(value, list) and value != []
                and all(isinstance(v, str) and v in allowed for v in value))
    if vt == "object":
        subs = set((spec.get("sub_fields") or {}))
        return isinstance(value, dict) and set(value) == subs and all(
            v is None or isinstance(v, int) for v in value.values())
    return isinstance(value, str)


ABSENCE_VALUES = {"no", "none", "unclear", "not_applicable", "no_dt", "no_agent",
                  "neither", "does_not_cite", "none_static", "not_synchronized",
                  "no_static_model", "no_explicit_definition", "none_reported"}


def is_absence(value) -> bool:
    if isinstance(value, str):
        return value in ABSENCE_VALUES
    if isinstance(value, list):
        return bool(value) and all(v in ABSENCE_VALUES for v in value)
    return False


def check_field(obj: dict, paper: PaperText, spec: dict | None = None) -> str:
    value, quote, page = obj.get("value"), obj.get("evidence_quote"), obj.get("page")
    if spec is not None and not value_ok(value, spec):
        return "bad_value"
    if isinstance(value, list) and "not_reported" in value and len(value) > 1:
        return "bad_value"
    if is_unreported(value):
        return "not_reported" if not quote else "bad_value"
    if not quote or not str(quote).strip():
        return "absence_unquoted" if is_absence(value) else "missing_quote"
    if len(str(quote).split()) > MAX_QUOTE_WORDS:
        return "quote_too_long"
    qn, ql = norm(quote), loose(quote)
    if qn and qn in paper.all_n:
        return "ok" if paper.on_page(qn, page, paper.n) else "page_mismatch"
    if ql and ql in paper.all_l:
        return "ok_loose" if paper.on_page(ql, page, paper.l) else "page_mismatch"
    return "quote_not_found"


# --------------------------------------------------------------------------

def load_targets(role_filter: str | None) -> list[dict]:
    rows = [r for r in csv.DictReader(open("corpus/report_corpus.csv", encoding="utf-8"))
            if r["retrieval_status"] == "ok"]
    if role_filter:
        rows = [r for r in rows if r["role"] == role_filter]
    return rows


def user_text(r: dict, text: str) -> str:
    return (f"Paper: {r['title']} ({r['year']}, {r['venue']})\n\n"
            f"--- BEGIN PAPER TEXT ---\n{text}\n--- END PAPER TEXT ---")


def parse(message) -> tuple[dict | None, str | None]:
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        return None, "refusal"
    if stop == "max_tokens":
        return None, "max_tokens"
    blocks = [b for b in (message.content or [])
              if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == TOOL_NAME]
    if len(blocks) != 1:
        return None, "no_tool_use" if not blocks else f"multiple_tool_use:{len(blocks)}"
    return blocks[0].input, None


def recheck(records_dir: str) -> int:
    """Re-check stored answers (values, quotes, pages unchanged) with the current checker."""
    import glob
    schemas = {k: json.load(open(p, encoding="utf-8")) for k, p in SCHEMAS.items()}
    specs = {k: {f"{b}__{f}": sp for b, f, sp in evidence_fields(s)} for k, s in schemas.items()}
    before, after, moved = Counter(), Counter(), Counter()
    for path in sorted(glob.glob(os.path.join(records_dir, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        role, pid = rec["J_EXTRACTION_META"]["schema"], rec["IDENT"]["paper_id"]
        paper = PaperText(open(f"corpus/documents/{pid}.txt", encoding="utf-8").read())
        for k, v in rec["fields"].items():
            old = v.get("check")
            new = check_field({x: v.get(x) for x in ("value", "evidence_quote", "page")},
                              paper, specs[role].get(k))
            before[old] += 1
            after[new] += 1
            if new != old:
                moved[f"{old}->{new}"] += 1
                v["check"] = new
        rec["J_EXTRACTION_META"]["checker_version"] = SCRIPT_VERSION
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
    print(f"[recheck] checker {SCRIPT_VERSION}: {sum(moved.values())} field(s) changed: {dict(moved)}")
    print(f"[recheck] now: {dict(after)}")
    prov = json.load(open("corpus/provenance.json", encoding="utf-8"))
    prov.setdefault("extraction_rechecks", []).append({
        "utc": datetime.now(timezone.utc).isoformat(), "checker_version": SCRIPT_VERSION,
        "records_dir": records_dir, "changed": dict(moved),
        "before": dict(before), "after": dict(after),
        "note": "stored model answers unchanged; only the deterministic evidence "
                "check was re-run"})
    json.dump(prov, open("corpus/provenance.json", "w", encoding="utf-8"), indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-text extraction with Opus 5.5.")
    ap.add_argument("--runs-dir", default="extraction/runs")
    ap.add_argument("--records-dir", default="extraction/records")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--role", choices=("agent", "dt"), default=None)
    ap.add_argument("--only", default=None, help="comma-separated paper_ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--recheck", action="store_true",
                    help="no API: re-run the evidence checks on existing records "
                         "with this version of check_field, and record the change")
    args = ap.parse_args()

    if args.recheck:
        return recheck(args.records_dir)

    targets = load_targets(args.role)
    if args.only:
        keep = set(args.only.split(","))
        targets = [t for t in targets if t["paper_id"] in keep]
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        sys.exit("no targets")

    schemas = {k: json.load(open(p, encoding="utf-8")) for k, p in SCHEMAS.items()}
    fields = {k: evidence_fields(s) for k, s in schemas.items()}
    tools = {k: [build_tool(fields[k])] for k in schemas}
    systems = {k: build_system(schemas[k], fields[k]) for k in schemas}
    sys_blocks = {k: [{"type": "text", "text": systems[k],
                       "cache_control": {"type": "ephemeral"}}] for k in schemas}
    prompt_sha = {k: sc.sha256(systems[k] + json.dumps(tools[k], sort_keys=True))
                  for k in schemas}

    run_id = args.run_id or "extract_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.records_dir, exist_ok=True)

    texts = {t["paper_id"]: open(f"corpus/documents/{t['paper_id']}.txt", encoding="utf-8").read()
             for t in targets}
    print(f"[run] {run_id}: {len(targets)} papers ("
          + ", ".join(f"{k}={v}" for k, v in Counter(t['role'] for t in targets).items())
          + f"), model {MODEL}, {'sync' if args.sync else 'batch'}")
    print(f"[run] evidence fields: agent={len(fields['agent'])}, dt={len(fields['dt'])}")

    client = anthropic.Anthropic()
    sc.validate_model_ids(client, [MODEL])

    # --- estimate: input counted exactly, output assumed ---------------------
    in_tok = 0
    for t in targets:
        ct = client.messages.count_tokens(
            model=MODEL, system=sys_blocks[t["role"]], tools=tools[t["role"]],
            messages=[{"role": "user", "content": user_text(t, texts[t["paper_id"]])}])
        in_tok += ct.input_tokens
    mult = 1.0 if args.sync else sc.BATCH_MULT
    pr = sc.PRICING[MODEL]
    est = mult * (in_tok * pr["input"] + len(targets) * ASSUMED_OUTPUT_TOKENS * pr["output"]) / 1e6
    worst = mult * (in_tok * pr["input"] + len(targets) * MAX_TOKENS * pr["output"]) / 1e6
    print(f"[cost] {in_tok} input tokens (counted), ~{ASSUMED_OUTPUT_TOKENS} output/paper "
          f"(assumed): estimated ${est:.2f}, worst case ${worst:.2f}, cap ${args.max_usd:g}")
    if est > args.max_usd:
        print(f"\n  *** ABORT: estimate ${est:.2f} exceeds --max-usd ${args.max_usd:g}. "
              f"Nothing was submitted.")
        return 2

    manifest = {
        "script": "extract.py", "script_version": SCRIPT_VERSION, "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "request_shape": {**sc.MODEL_REQUEST[MODEL], "sent": sc.model_kwargs(MODEL)},
        "papers": [t["paper_id"] for t in targets],
        "schema_versions": {k: s["schema_version"] for k, s in schemas.items()},
        "prompt_sha256": prompt_sha, "n_evidence_fields": {k: len(v) for k, v in fields.items()},
        "max_tokens": MAX_TOKENS, "max_quote_words": MAX_QUOTE_WORDS,
        "strict_tool": False,
        "strict_tool_basis": ("strict tool use is rejected for these schemas "
                              "(API: 'compiled grammar is too large'); enum "
                              "membership and value shape are checked per field "
                              "in Python (check_field -> bad_value)"),
        "estimate": {"input_tokens_counted": in_tok, "assumed_output_per_paper": ASSUMED_OUTPUT_TOKENS,
                     "estimated_usd": round(est, 4), "worst_case_usd": round(worst, 4),
                     "cap_usd": args.max_usd},
        "api_mode": "sync" if args.sync else "batch",
    }
    json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    if not args.yes and input(f"Submit {len(targets)} requests for ~${est:.2f}? [y/N] ").lower() not in ("y", "yes"):
        print("aborted; nothing submitted")
        return 1

    by_id = {t["paper_id"]: t for t in targets}
    calls_path = os.path.join(run_dir, "calls.jsonl")
    log = open(calls_path, "a", encoding="utf-8")
    total_cost = 0.0
    statuses: Counter[str] = Counter()
    field_status: Counter[str] = Counter()

    def record(pid: str, message, err: str | None, batch: bool):
        nonlocal total_cost
        t = by_id[pid]
        role = t["role"]
        if message is None:
            parsed, failure, usage, raw = None, err, {"input_tokens": 0, "output_tokens": 0,
                                                      "cache_read_input_tokens": 0,
                                                      "cache_creation_input_tokens": 0}, None
        else:
            parsed, failure = parse(message)
            usage = sc.usage_dict(message.usage)
            raw = json.loads(message.to_json())
        cost = sc.cost_usd(MODEL, usage, batch)
        total_cost += cost
        status = "valid" if parsed is not None else "invalid"
        statuses[status] += 1
        paper = PaperText(texts[pid])
        checked = {}
        if parsed:
            for b, f, spec in fields[role]:
                raw_obj = parsed.get(f"{b}__{f}")
                obj = from_sentinels(raw_obj) if isinstance(raw_obj, dict) else {}
                st = check_field(obj, paper, spec) if obj else "missing_field"
                field_status[st] += 1
                checked[f"{b}__{f}"] = {**obj, "check": st}
        log.write(json.dumps({"run_id": run_id, "paper_id": pid, "role": role,
                              "model": MODEL, "api_mode": "batch" if batch else "sync",
                              "status": status, "failure": failure, "raw_response": raw,
                              **usage, "cost_usd": round(cost, 6),
                              "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
        log.flush()
        cand = next((c for c in csv.DictReader(open("corpus/candidates.csv", encoding="utf-8"))
                     if c["openalex_id"] == pid), {})
        rec = {
            "IDENT": {"paper_id": pid, "doi": t["doi"], "title": t["title"],
                      "year": int(t["year"]) if t["year"].isdigit() else t["year"],
                      "venue": t["venue"],
                      "first_author": (cand.get("authors") or "").split(";")[0].strip(),
                      "seed_role_links": [s for s in (cand.get("seed_roles") or "").split("|") if s]},
            "fields": checked,
            "J_EXTRACTION_META": {
                "schema_version": schemas[role]["schema_version"], "schema": role,
                "extraction_model": MODEL, "prompt_hash": prompt_sha[role],
                "extraction_date": datetime.now(timezone.utc).date().isoformat(),
                "source_text_method": "pymupdf_text",
                "run_id": run_id, "response_status": status, "failure": failure,
                "human_verified": False, "verifier": None, "fields_corrected": [],
                "verifier_notes": None},
        }
        with open(os.path.join(args.records_dir, f"{pid}.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)

    try:
        if args.sync:
            for t in targets:
                try:
                    # Streaming is required by the SDK at this max_tokens; the
                    # final message is identical to a non-streamed response.
                    with client.messages.stream(
                            model=MODEL, max_tokens=MAX_TOKENS, system=sys_blocks[t["role"]],
                            tools=tools[t["role"]], tool_choice={"type": "auto"},
                            messages=[{"role": "user", "content": user_text(t, texts[t["paper_id"]])}],
                            **sc.model_kwargs(MODEL)) as stream:
                        msg = stream.get_final_message()
                    record(t["paper_id"], msg, None, False)
                except anthropic.APIError as exc:
                    record(t["paper_id"], None, f"api_error:{type(exc).__name__}:{exc}"[:300], False)
                    print(f"  [err] {t['paper_id']}: {exc}")
        else:
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request
            reqs = [Request(custom_id=t["paper_id"], params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=MAX_TOKENS, system=sys_blocks[t["role"]],
                tools=tools[t["role"]], tool_choice={"type": "auto"},
                messages=[{"role": "user", "content": user_text(t, texts[t["paper_id"]])}],
                **sc.model_kwargs(MODEL))) for t in targets]
            batch = client.messages.batches.create(requests=reqs)
            print(f"[batch] {batch.id}, {len(reqs)} requests")
            manifest["batch_id"] = batch.id
            while True:
                batch = client.messages.batches.retrieve(batch.id)
                if batch.processing_status == "ended":
                    break
                c = batch.request_counts
                print(f"  [batch] {batch.processing_status} processing={c.processing} "
                      f"succeeded={c.succeeded} errored={c.errored}", flush=True)
                time.sleep(args.poll_seconds)
            for res in client.messages.batches.results(batch.id):
                if res.result.type == "succeeded":
                    record(res.custom_id, res.result.message, None, True)
                else:
                    detail = getattr(getattr(res.result, "error", None), "type", res.result.type)
                    record(res.custom_id, None, f"batch_{res.result.type}:{detail}", True)
    finally:
        log.close()

    manifest["result"] = {"ended_utc": datetime.now(timezone.utc).isoformat(),
                          "responses": dict(statuses), "field_checks": dict(field_status),
                          "actual_cost_usd": round(total_cost, 4)}
    json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    print(f"\n[done] responses: {dict(statuses)}")
    print(f"[done] field checks: {dict(field_status)}")
    print(f"[done] actual cost ${total_cost:.4f} (estimated ${est:.2f})")
    print(f"  -> {calls_path}\n  -> {args.records_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
