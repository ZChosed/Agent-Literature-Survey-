"""
check -- validate a project's inputs before any stage runs or any money is spent.

  litreview check             every input file, plus the API key and model
  litreview check --offline   skip the API call

Reads only; writes nothing. Each line is `ok`, `WARN` (legal but probably not
what you meant) or `FAIL` (a later stage would abort, or would silently do the
wrong thing). Exit status 1 if anything FAILs.

Checked:
  project.toml   contact email; numeric settings in range; unknown keys (a typo
                 is silently ignored, so it is reported); model priced
  seeds.csv      columns; unique refs; expand values recognised; template rows
                 left in; core_roles / target_refs name real seeds (coverage_roles: WARN)
  criteria.md    at least 2 criterion IDs parsed (else IDs are not enforced);
                 template text left in
  schemas        every [extraction.schemas.*] file loads; every evidence_object
                 field has a valid value_type and its options; every mapped
                 criterion exists in criteria.md; criteria mapped to no schema
  report         every crosstab names a real schema and field
  git            inputs committed (settings fixed before the run they govern)
  API            ANTHROPIC_API_KEY set; the configured model is served (never
                 substituted)
"""

from __future__ import annotations

import json
import os
import subprocess

from . import llm
from .build import load_seeds
from .common import read_csv
from .config import DEFAULTS
from .extract import fields_of
from .screen import criterion_ids

SEED_COLUMNS = ["ref", "role", "expand_backward", "expand_forward", "title"]
YES_NO = {"", "yes", "no", "y", "n", "true", "false", "1", "0"}
VALUE_TYPES = {"enum", "multi_enum", "string", "object"}
# Keys a project may set that have no entry in DEFAULTS.
EXTRA_KEYS = {"screening": {"topic"}, "model": {"pricing", "request"},
              "extraction": {"absence_values"}}
TEMPLATE_MARKS = ("Replace with", "10.0000/replace-me")


class Report:
    def __init__(self):
        self.fails = self.warns = 0

    def section(self, name: str) -> None:
        print(f"\n[check] {name}")

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        print(f"  WARN  {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        print(f"  FAIL  {msg}")


def check_toml(cfg, r: Report) -> None:
    r.section("project.toml")
    if cfg["project"].get("contact_email", "").strip():
        r.ok("contact_email set")
    else:
        r.fail("[project] contact_email is empty (OpenAlex and Unpaywall require one)")

    for section, keys in cfg.data.items():
        if section not in DEFAULTS:
            r.warn(f"unknown section [{section}] is ignored")
            continue
        if not isinstance(keys, dict):
            continue
        known = set(DEFAULTS[section]) | EXTRA_KEYS.get(section, set())
        for k in sorted(set(keys) - known):
            r.warn(f"unknown key [{section}] {k} is ignored (typo?)")

    sc, ex, c = cfg["screening"], cfg["extraction"], cfg["corpus"]
    for name, v in (("[screening] kappa_floor", sc["kappa_floor"]), ("[screening] recall_floor", sc["recall_floor"])):
        if not isinstance(v, (int, float)) or not 0 < v <= 1:
            r.fail(f"{name} = {v!r}: must be a number in (0, 1]")
    for name, v in (("[screening] max_usd", sc["max_usd"]), ("[extraction] max_usd", ex["max_usd"])):
        if not isinstance(v, (int, float)) or v <= 0:
            r.fail(f"{name} = {v!r}: must be a positive number")
    for name, v in (("[screening] calibration_per_stratum", sc["calibration_per_stratum"]),
                    ("[corpus] min_year", c["min_year"]), ("[extraction] quote_words", ex["quote_words"])):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            r.fail(f"{name} = {v!r}: must be a positive integer")
    if not sc.get("topic", "").strip():
        r.warn("[screening] topic is empty; the model is told only 'a structured literature review'")

    model = cfg["model"]["id"]
    price = {**llm.PRICING.get(model, {}), **(cfg["model"].get("pricing") or {})}
    if {"input", "output"} <= set(price):
        r.ok(f"model {model} is priced (${price['input']:g} in / ${price['output']:g} out per Mtok)")
    else:
        r.fail(f"model {model} has no price: add [model] pricing = {{ input = ..., output = ... }} (USD per Mtok)")


def check_seeds(cfg, r: Report) -> None:
    c = cfg["corpus"]
    path = cfg.path(c["seeds"])
    r.section(c["seeds"])
    if not os.path.exists(path):
        r.fail(f"{path} not found")
        return
    raw = read_csv(path)
    cols = list(raw[0]) if raw else []
    missing = [k for k in SEED_COLUMNS if k not in cols]
    if missing:
        r.fail(f"missing column(s) {missing}; expected {SEED_COLUMNS + ['year', 'first_author', 'doi']}")
        return
    untitled = [i + 2 for i, row in enumerate(raw) if not (row.get("title") or "").strip()]
    if untitled:
        r.warn(f"row(s) on line {untitled} have no title and are ignored")
    try:
        seeds = load_seeds(path)
    except SystemExit as e:
        r.fail(str(e.code))
        return
    if not seeds:
        r.fail("no seeds")
        return
    r.ok(f"{len(seeds)} seed(s), refs unique")

    template = [row["ref"] for row in raw if any(m in (row.get("title", "") + row.get("doi", "")) for m in TEMPLATE_MARKS)]
    if template:
        r.fail(f"seed(s) {template} still hold the template's placeholder title/DOI")
    for row in raw:
        for col in ("expand_backward", "expand_forward"):
            v = (row.get(col) or "").strip().lower()
            if v not in YES_NO:
                r.fail(f"seed [{row['ref']}] {col} = {row[col]!r}: use yes or no (anything else reads as no)")
    inert = [s["ref"] for s in seeds if not s["expand_backward"] and not s["expand_forward"]]
    if inert:
        r.warn(f"seed(s) {inert} expand in neither direction and contribute no candidates")
    no_doi = [row["ref"] for row in raw if (row.get("title") or "").strip() and not (row.get("doi") or "").strip()]
    if no_doi:
        r.warn(f"seed(s) {no_doi} have no DOI and will be title-matched: check data/seeds_resolved.csv after build")

    roles, refs = {s["role"] for s in seeds}, {s["ref"] for s in seeds}
    if not c["core_roles"]:
        r.warn("[corpus] core_roles is empty: no candidate is flagged linked_to_core, weights are uniform")
    absent = [x for x in c["core_roles"] if x not in roles]
    if absent:
        r.fail(f"[corpus] core_roles {absent} match no seed role (roles in seeds.csv: {sorted(roles)})")
    absent = [x for x in c["coverage_roles"] if x not in roles]
    if absent:
        r.warn(f"[corpus] coverage_roles {absent} match no seed role (roles in seeds.csv: {sorted(roles)}): "
               f"no coverage check runs for them")
    absent = [str(x) for x in c["target_refs"] if str(x) not in refs]
    if absent:
        r.fail(f"[corpus] target_refs {absent} match no seed ref")


def check_criteria(cfg, r: Report) -> list[str]:
    name = cfg["screening"]["criteria"]
    r.section(name)
    path = cfg.path(name)
    if not os.path.exists(path):
        r.fail(f"{path} not found")
        return []
    text = open(path, encoding="utf-8").read()
    ids = criterion_ids(text)
    if len(ids) >= 2:
        r.ok(f"{len(ids)} criterion IDs parsed and enforced: {ids}")
    else:
        r.fail(f"only {len(ids)} criterion ID(s) parsed ({ids}); with fewer than 2 the model's criterion_id "
               f"is not checked. Start each criterion line with its ID and a dash: 'I1 — ...'")
    if "The paper ..." in text:
        r.fail("template text 'The paper ...' is still in the criteria")
    return ids


def check_schemas(cfg, r: Report, ids: list[str]) -> dict[str, set[str]]:
    r.section("extraction schemas")
    schemas = cfg["extraction"]["schemas"]
    fields: dict[str, set[str]] = {}
    if not schemas:
        r.fail("no [extraction.schemas.<name>] in project.toml")
        return fields
    mapped: set[str] = set()
    for name, s in schemas.items():
        label = f"[extraction.schemas.{name}]"
        path = cfg.path(s.get("schema", ""))
        if not s.get("schema") or not os.path.isfile(path):
            r.fail(f"{label} schema file not found: {s.get('schema')!r}")
            continue
        try:
            schema = json.load(open(path, encoding="utf-8"))
            evidence = fields_of(schema)
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
            r.fail(f"{label} {s['schema']} does not load ({type(e).__name__}: {e}); "
                   f"expected {{\"blocks\": {{BLOCK: {{\"fields\": {{...}}}}}}}}")
            continue
        fields[name] = {k for k, _ in evidence}
        bad = 0
        for key, spec in evidence:
            vt = spec.get("value_type")
            if vt not in VALUE_TYPES:
                r.fail(f"{s['schema']} {key}: value_type {vt!r} is not one of {sorted(VALUE_TYPES)}")
                bad += 1
            elif vt in ("enum", "multi_enum") and not spec.get("allowed"):
                r.fail(f"{s['schema']} {key}: {vt} needs a non-empty `allowed` list")
                bad += 1
            elif vt == "object" and not spec.get("sub_fields"):
                r.fail(f"{s['schema']} {key}: object needs `sub_fields`")
                bad += 1
        if not evidence:
            r.fail(f"{s['schema']} has no field typed evidence_object, so nothing would be extracted")
        elif not bad:
            r.ok(f"{label} {s['schema']}: {len(evidence)} evidence field(s)")
        if any(m in str(schema.get("purpose", "")) for m in TEMPLATE_MARKS):
            r.warn(f"{s['schema']} `purpose` is still the template text (it is shown to the model)")

        crit = s.get("criteria") or []
        if not crit:
            r.fail(f"{label} criteria is empty: no included paper would be extracted with it")
        unknown = [x for x in crit if ids and x not in ids]
        if unknown:
            r.fail(f"{label} criteria {unknown} are not IDs in {cfg['screening']['criteria']} ({ids})")
        mapped |= set(crit)
        cap = s.get("max_papers", 0)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            r.fail(f"{label} max_papers = {cap!r}: must be 0 (all) or a positive integer")
    unmapped = [x for x in ids if x not in mapped]
    if unmapped and ids:
        r.warn(f"criteria {unmapped} map to no schema: an included paper whose deciding criterion is one "
               f"of these is not extracted (fine for exclude/uncertain criteria)")
    return fields


def check_report(cfg, r: Report, fields: dict[str, set[str]]) -> None:
    r.section("report")
    tabs = cfg["report"]["crosstabs"]
    if not tabs:
        r.ok("no crosstabs configured")
        return
    for t in tabs:
        if not (isinstance(t, list) and len(t) == 3):
            r.fail(f"crosstab {t!r}: must be [schema, field_a, field_b]")
            continue
        schema, fa, fb = t
        if schema not in cfg["extraction"]["schemas"]:
            r.fail(f"crosstab {t}: no schema named {schema!r}")
        elif schema in fields:
            absent = [f for f in (fa, fb) if f not in fields[schema]]
            if absent:
                r.fail(f"crosstab {t}: {absent} not evidence fields of {schema!r} "
                       f"(use BLOCK__field, e.g. {sorted(fields[schema])[:2]})")
            else:
                r.ok(f"crosstab {schema}: {fa} x {fb}")


def check_git(cfg, r: Report) -> None:
    r.section("git")
    paths = ["project.toml", cfg["corpus"]["seeds"], cfg["screening"]["criteria"]]
    paths += [s["schema"] for s in cfg["extraction"]["schemas"].values() if s.get("schema")]
    paths = [p for p in paths if os.path.exists(cfg.path(p))]
    try:
        inside = subprocess.run(["git", "-C", cfg.root, "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True)
    except FileNotFoundError:
        r.warn("git not found: settings cannot be shown to have been fixed before a run")
        return
    if inside.returncode != 0:
        r.warn("project is not a git repository: run `git init` and commit the inputs before each run")
        return
    out = subprocess.run(["git", "-C", cfg.root, "status", "--porcelain", "--", *paths],
                         capture_output=True, text=True).stdout
    dirty = sorted({line[3:].strip() for line in out.splitlines() if line.strip()})
    if dirty:
        r.warn(f"uncommitted input(s) {dirty}: commit before the run they govern")
    else:
        r.ok("inputs committed")


def check_api(cfg, r: Report, offline: bool) -> None:
    r.section("Anthropic API")
    if offline:
        print("  --    skipped (--offline)")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        r.fail("ANTHROPIC_API_KEY is not set (needed from `screen` onward; build/eligibility work without it)")
        return
    if llm.anthropic is None:
        r.fail("the anthropic package is not installed: pip install anthropic")
        return
    model = cfg["model"]["id"]
    try:
        served = {m.id for m in llm.anthropic.Anthropic().models.list()}
    except Exception as e:                                      # noqa: BLE001
        r.fail(f"could not list models with this key: {type(e).__name__}: {e}")
        return
    if model in served:
        r.ok(f"key works; {model} is served")
    else:
        r.fail(f"{model} is not served by this key (available: {', '.join(sorted(served))}). "
               f"litreview never substitutes a model; get a key with access or change [model] id")


def run(cfg, args) -> int:
    r = Report()
    check_toml(cfg, r)
    check_seeds(cfg, r)
    ids = check_criteria(cfg, r)
    fields = check_schemas(cfg, r, ids)
    check_report(cfg, r, fields)
    check_git(cfg, r)
    check_api(cfg, r, getattr(args, "offline", False))
    print(f"\n[check] {r.fails} FAIL, {r.warns} WARN"
          + (" -- fix every FAIL before running a stage" if r.fails else " -- ready"))
    return 1 if r.fails else 0
