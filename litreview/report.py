"""
report -- every table, computed in Python, and the report rendered from a template.

Tables (extraction/tables/): one frequency table per extracted field
(<schema>__<BLOCK>__<field>.csv: value, papers, share), every cross-tab listed in
[report] crosstabs as [schema, field_a, field_b], and evidence-check tables. Only
fields whose evidence check passed (or is not_reported / absence_unquoted) are
counted; each table records its n in aggregates.json.

The template (report_template.md; scaffolded on first run) is prose with
placeholders, so no number is ever typed by hand:
  {{agg.<path>}}  {{prov.<path>}}     a value from aggregates.json / data/provenance.json
  {{pct:agg.<path>}}                  ... as a percentage
  {{table:<name>}}  {{table:<name>|10}}           a table as Markdown (first N rows)
  {{cell:<table>|<value>|papers}}  {{pctcell:<table>|<value>}}  one frequency row
  {{xcell:<table>|<row>|<col>}}                   one cross-tab cell
An unresolved placeholder is an error and nothing is written.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter

from .common import read_csv, read_json, write_csv, write_json

COUNTED = {"ok", "ok_loose", "page_mismatch", "not_reported", "absence_unquoted"}
VERIFIED = {"ok", "ok_loose", "page_mismatch"}


def as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def label(v) -> str:
    return v if isinstance(v, str) else ("" if v is None else str(v))


def aggregate(cfg) -> dict:
    recs = [read_json(p) for p in sorted(glob.glob(cfg.path("extraction", "records", "*.json")))]
    if not recs:
        sys.exit("no extraction records -- run `litreview extract`")
    tdir = cfg.path("extraction", "tables")
    os.makedirs(tdir, exist_ok=True)
    out = {"papers": {"total": len(recs), **Counter(r["schema"] for r in recs),
                      "valid_responses": sum(r["response"] == "valid" for r in recs),
                      "human_verified": sum(bool(r.get("human_verified")) for r in recs)},
           "tables": {}}
    checks = Counter(o["check"] for r in recs for o in r["fields"].values())
    answered = sum(n for c, n in checks.items() if c not in ("not_reported", "absence_unquoted"))
    every = ["ok", "ok_loose", "page_mismatch", "not_reported", "absence_unquoted", "quote_not_found",
             "quote_too_long", "missing_quote", "bad_value"]
    out["evidence"] = {"fields": sum(checks.values()),
                       "by_check": {c: checks.get(c, 0) for c in every + sorted(set(checks) - set(every))},
                       "verified_share_of_answered": round(sum(checks[c] for c in VERIFIED) / max(answered, 1), 4),
                       "excluded_from_tables": sum(n for c, n in checks.items() if c not in COUNTED)}

    def values(schema: str, field: str) -> dict[str, object]:
        return {r["paper_id"]: r["fields"][field]["value"] for r in recs
                if r["schema"] == schema and field in r["fields"] and r["fields"][field]["check"] in COUNTED}

    fields = sorted({(r["schema"], k) for r in recs for k in r["fields"]})
    for schema, field in fields:
        vals = values(schema, field)
        if any(isinstance(v, dict) for v in vals.values()):
            continue
        c = Counter(label(x) for v in vals.values() for x in as_list(v))
        name = f"{schema}__{field}"
        write_csv(os.path.join(tdir, f"{name}.csv"),
                  [{"value": k, "papers": n, "share": round(n / len(vals), 3) if vals else ""}
                   for k, n in c.most_common()], ["value", "papers", "share"])
        out["tables"][name] = {"n": len(vals)}
    for schema, fa, fb in cfg["report"]["crosstabs"]:
        a, b = values(schema, fa), values(schema, fb)
        both = sorted(set(a) & set(b))
        cells = Counter((label(x), label(y)) for p in both for x in as_list(a[p]) for y in as_list(b[p]))
        cols = sorted({y for _, y in cells})
        name = f"{schema}__{fa.split('__')[-1]}__x__{fb.split('__')[-1]}"
        write_csv(os.path.join(tdir, f"{name}.csv"),
                  [{"": x, **{y: cells[(x, y)] for y in cols}} for x in sorted({x for x, _ in cells})], [""] + cols)
        out["tables"][name] = {"n": len(both)}
    by_field = Counter((f"{r['schema']}__{k}", o["check"]) for r in recs for k, o in r["fields"].items())
    names = sorted({f for f, _ in by_field})
    cks = sorted({c for _, c in by_field})
    write_csv(os.path.join(tdir, "evidence_by_field.csv"),
              [{"": f, **{c: by_field[(f, c)] for c in cks}} for f in names], [""] + cks)
    write_json(cfg.path("extraction", "aggregates.json"), out)
    return out


def scaffold(cfg, agg: dict) -> str:
    lines = [f"# {cfg['project'].get('name') or 'Literature review'}", "",
             "> Every number below is computed by `litreview report` from the extraction records. "
             "{{agg.papers.human_verified}} of {{agg.papers.total}} records are human-verified.", "",
             "## Method", "",
             "| Stage | n |", "|---|---:|",
             "| Candidates | {{prov.build.candidates}} |",
             "| Eligible | {{prov.eligibility.eligible}} |",
             "| Included at screening | {{prov.document_list.outcomes.included}} |",
             "| Awaiting human review | {{prov.document_list.outcomes.needs_human_review}} |",
             "| Full text extracted | {{agg.papers.total}} |", "",
             "Evidence: {{pct:agg.evidence.verified_share_of_answered}} of answered fields carry a quote "
             "verified verbatim in the paper; {{agg.evidence.by_check.absence_unquoted}} absence answers "
             "carry none; {{agg.evidence.excluded_from_tables}} failed fields are excluded from every table.", ""]
    for name, (schema, flds) in sorted(_schemas(cfg).items()):
        lines += [f"## {name}", ""]
        block = None
        for key, spec in flds:
            b = key.split("__")[0]
            if b != block:
                block = b
                lines += [f"### {schema['blocks'][b].get('label', b)}", ""]
            t = f"{name}__{key}"
            if t in agg["tables"]:
                lines += [f"**{key.split('__')[1]}** (n = {{{{agg.tables.{t}.n}}}})", "", f"{{{{table:{t}}}}}", ""]
    for t in agg["tables"]:
        if "__x__" in t:
            lines += [f"### {t}", "", f"{{{{table:{t}}}}}", ""]
    return "\n".join(lines)


def _schemas(cfg) -> dict:
    from .extract import load_schemas
    return load_schemas(cfg)


def md_table(path: str, nrows: int | None) -> str:
    rows = read_csv(path)[:nrows] if nrows else read_csv(path)
    cols = list(rows[0].keys()) if rows else []
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows])


def render(cfg, agg: dict, template: str) -> tuple[str, list[str]]:
    data = {"agg": agg, "prov": read_json(cfg.provenance, {})}
    tdir = cfg.path("extraction", "tables")
    missing = []

    def look(path: str):
        src, _, rest = path.partition(".")
        cur = data[src]
        for part in rest.split("."):
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        return cur

    def fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.3f}".rstrip("0").rstrip(".")
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, dict):
            return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items())
        return str(v)

    def freq_row(t: str, value: str) -> dict:
        row = next((r for r in read_csv(os.path.join(tdir, f"{t}.csv")) if r["value"] == value), None)
        if row is None:
            raise KeyError(value)
        return row

    def sub(m):
        k = m.group(1).strip()
        try:
            kind, _, rest = k.partition(":")
            if kind == "table":
                t, _, n = rest.partition("|")
                return md_table(os.path.join(tdir, f"{t}.csv"), int(n) if n else None)
            if kind == "cell":
                t, v, col = rest.split("|")
                return freq_row(t, v)[col]
            if kind == "pctcell":
                t, v = rest.split("|")
                return f"{100 * float(freq_row(t, v)['share']):.0f}%"
            if kind == "xcell":
                t, r, c = rest.split("|")
                return next(x for x in read_csv(os.path.join(tdir, f"{t}.csv")) if x[""] == r)[c]
            if kind == "pct":
                return f"{100 * float(look(rest)):.1f}%"
            return fmt(look(k))
        except (KeyError, IndexError, ValueError, StopIteration, FileNotFoundError):
            missing.append(k)
            return m.group(0)

    return re.sub(r"\{\{(.+?)\}\}", sub, template), missing


def run(cfg, args) -> int:
    agg = aggregate(cfg)
    tpath = cfg.path(cfg["report"]["template"])
    if not os.path.exists(tpath) or args.scaffold:
        if os.path.exists(tpath) and not args.force:
            sys.exit(f"{tpath} exists; add --force to overwrite it with a fresh scaffold")
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(scaffold(cfg, agg))
        print(f"[report] scaffolded {tpath} -- edit the prose; keep numbers as placeholders")
    text, missing = render(cfg, agg, open(tpath, encoding="utf-8").read())
    if missing:
        print("*** unresolved placeholders; report NOT written:\n    " + "\n    ".join(sorted(set(missing))))
        return 1
    out = cfg.path("report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] {agg['papers']['total']} papers, {len(agg['tables'])} tables; evidence "
          f"{agg['evidence']['by_check']}")
    print(f"  -> {out}\n  -> {cfg.path('extraction', 'tables')}/")
    return 0
