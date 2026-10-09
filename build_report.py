#!/usr/bin/env python3
"""
build_report.py -- render the report from a template whose numbers all come from Python.

The prose in report/report_template.md is written by hand. Every number in it is
a placeholder, filled here from files the pipeline wrote:

    {{agg.<dotted.path>}}     extraction/aggregate/aggregates.json
    {{prov.<dotted.path>}}    corpus/provenance.json
    {{list.<dotted.path>}}    screening/final/document_list_summary.json
    {{bench.<dotted.path>}}   the calibration benchmark summary
    {{pct:<source>.<path>}}   the same value formatted as a percentage
    {{table:<name>}}          extraction/aggregate/<name>.csv as a Markdown table
    {{table:<name>|<n>}}      ... first n rows only
    {{cell:<table>|<value>|<column>}}  one row of a frequency table (value column)
    {{xcell:<table>|<row>|<col>}}      one cell of a cross-tab
    {{pctcell:<table>|<value>}}        a frequency row's share as a percentage
    {{ext.<dotted.path>}}     the extraction run's manifest
    {{papers:S7_gap}}         the S7 gap papers as a bulleted list

An unresolved placeholder is a hard error, never a blank: a report with a
missing number must not render as though it were complete.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

TEMPLATE = "report/report_template.md"
OUT = "report/report1.md"
SOURCES = {
    "agg": "extraction/aggregate/aggregates.json",
    "prov": "corpus/provenance.json",
    "list": "screening/final/document_list_summary.json",
    "bench": "screening/runs/calib_v3_20261006/benchmark_summary.json",
    "ext": "extraction/runs/extract_20261009/manifest.json",
}
TABLES = "extraction/aggregate"


class Missing(KeyError):
    pass


def lookup(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise Missing(path)
    return cur


def fmt(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items())
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def md_table(name: str, nrows: int | None) -> str:
    path = os.path.join(TABLES, f"{name}.csv")
    if not os.path.exists(path):
        raise Missing(f"table:{name}")
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: ""})
    if nrows:
        df = df.head(nrows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            "" if pd.isna(x) else (f"{x:.3f}".rstrip("0").rstrip(".")
                                   if isinstance(x, float) else str(x))
            for x in r) + " |")
    return "\n".join(lines)


def read_table(name: str) -> pd.DataFrame:
    path = os.path.join(TABLES, f"{name}.csv")
    if not os.path.exists(path):
        raise Missing(f"table:{name}")
    return pd.read_csv(path, index_col=0)


def cell(name: str, value: str, column: str):
    df = read_table(name)
    row = df[df["value"].astype(str) == value]
    if row.empty:
        raise Missing(f"cell:{name}|{value}")
    return row.iloc[0][column]


def xcell(name: str, r: str, c: str):
    df = read_table(name)
    df.index = df.index.astype(str)
    if r not in df.index or c not in df.columns:
        raise Missing(f"xcell:{name}|{r}|{c}")
    return df.loc[r, c]


def main() -> int:
    data = {k: json.load(open(p, encoding="utf-8")) for k, p in SOURCES.items()
            if os.path.exists(p)}
    text = open(TEMPLATE, encoding="utf-8").read()
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        try:
            if key.startswith("table:"):
                name, _, n = key[6:].partition("|")
                return md_table(name, int(n) if n else None)
            if key.startswith(("cell:", "xcell:", "pctcell:")):
                kind, _, rest = key.partition(":")
                parts = rest.split("|")
                if kind == "cell":
                    v = cell(*parts)
                    return fmt(int(v)) if float(v).is_integer() else fmt(float(v))
                if kind == "pctcell":
                    return f"{100 * float(cell(parts[0], parts[1], 'share')):.0f}%"
                return fmt(int(xcell(*parts)))
            if key == "papers:S7_gap":
                ps = lookup(data["agg"], "S7_gap.papers")
                return "\n".join(f"- {p['title']} (`{p['paper_id']}`)" for p in ps) \
                    or "- *(none)*"
            pct = key.startswith("pct:")
            src, _, path = (key[4:] if pct else key).partition(".")
            v = lookup(data[src], path)
            return f"{100 * float(v):.1f}%" if pct else fmt(v)
        except (Missing, KeyError, IndexError, ValueError):
            missing.append(key)
            return "{{" + key + "}}"

    rendered = re.sub(r"\{\{(.+?)\}\}", sub, text)
    if missing:
        print("  *** UNRESOLVED placeholders (report not written):")
        for k in sorted(set(missing)):
            print(f"      {k}")
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(f"[report] {OUT}  ({len(rendered):,} chars, every number from Python)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
