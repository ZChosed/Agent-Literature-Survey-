"""
Offline tests for litreview: no network, no API key. OpenAlex is faked in-process
and the Anthropic client is stubbed; the stub records every request it receives.

Invariants covered, stage by stage:
  build        seeds never become candidates; year/type/dedupe filters; core first
  eligibility  OA rule + optional DOI-prefix rule; undecidable rows abort and write
               nothing; overrides only where the rules cannot decide; candidates.csv
               never modified; a changed corpus makes eligibility STALE
  calibrate    draw is reproducible and archives (never overwrites) an old sample;
               a partly filled sheet is refused; kappa/recall against a hand count
  screen       ineligible papers never reach the API; title-only papers are forced
               without a call; the budget aborts before submitting; an unserved
               model aborts (never substituted); reason overrun warns, 50+ fails
  fetch        failed responses go to human review, never to exclude; human review
               decisions apply; schema mapping with a rank cap; manual PDFs matched
  extract      every evidence-check outcome
  report       only passing fields are counted; an unresolved placeholder writes nothing

Run:  python tests/test_litreview.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litreview import build, calibrate, cli, config, eligibility, extract, fetch, llm, report, screen  # noqa: E402
from litreview.common import read_csv, read_json, read_jsonl, write_csv  # noqa: E402

FAILS: list[str] = []


def check(cond, label):
    print(f"  {'pass' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


def raises_exit(fn, *a) -> str | None:
    try:
        fn(*a)
    except SystemExit as e:
        return str(e.code)
    return None


def args(**kw):
    base = dict(offline=False, sample=False, sync=True, limit=0, only=None, yes=True, run=None,
                missing_abstracts=False, manual_only=True, retry=False, recheck=False,
                scaffold=False, force=False)
    return types.SimpleNamespace(**{**base, **kw})


def new_project(tmp: str, extra: str = "") -> config.Config:
    d = os.path.join(tmp, "proj")
    cli.main(["init", d])
    toml = open(os.path.join(d, "project.toml")).read().replace('contact_email = ""', 'contact_email = "t@example.org"')
    toml = toml.replace("[report]", extra + "\n[report]") if extra else toml
    open(os.path.join(d, "project.toml"), "w").write(toml)
    return config.load(d)


# --- fake OpenAlex ---------------------------------------------------------------

def work(wid, title, year=2020, refs=(), typ="article", topics=("T1",), abstract=True, doi=None):
    return {"id": f"https://openalex.org/{wid}", "doi": doi, "display_name": title,
            "publication_year": year, "type": typ, "language": "en", "cited_by_count": 10,
            "referenced_works": [f"https://openalex.org/{r}" for r in refs],
            "topics": [{"id": f"https://openalex.org/{t}", "score": 0.9} for t in topics],
            "abstract_inverted_index": {"An": [0], "abstract.": [1]} if abstract else None,
            "primary_location": {"source": {"display_name": "Venue"}}, "authorships": [],
            "open_access": {"is_oa": True}}


WORKS = {w["id"].rsplit("/", 1)[1]: w for w in [
    work("S1", "Seed paper one", refs=("A", "B", "OLD", "BOOK")),
    work("S2", "Seed paper two", refs=("B", "C")),
    work("A", "Candidate alpha"), work("B", "Candidate beta"), work("C", "Candidate gamma", topics=("T9",)),
    work("OLD", "Too old", year=2001), work("BOOK", "A dataset", typ="dataset"),
    work("F1", "Citing work one"), work("F2", "candidate ALPHA")]}  # F2 duplicates A's title
CITES = {"S1": ["F1", "F2"], "S2": []}


class FakeHttp:
    def __init__(self, *a, **k):
        self.n, self.mailto = 0, "t@example.org"

    def openalex(self, path, **p):
        self.n += 1
        f = p.get("filter", "")
        if f.startswith("doi:"):
            wid = {"10.1/s1": "S1", "10.1/s2": "S2"}.get(f[4:])
            return {"results": [WORKS[wid]] if wid else []}
        if f.startswith("openalex_id:"):
            return {"results": [WORKS[i] for i in f[12:].split("|") if i in WORKS]}
        if f.startswith("cites:"):
            ids = CITES[f.split(",")[0][6:]]
            if p.get("select") == "id":
                return {"meta": {"count": len(ids)}}
            return {"results": [WORKS[i] for i in ids], "meta": {"count": len(ids), "next_cursor": None}}
        return {"results": []}

    def get(self, *a, **k):
        return None


def test_build(tmp):
    print("\n[build]")
    cfg = new_project(tmp)
    write_csv(cfg.path("seeds.csv"), [
        {"ref": "1", "role": "core", "expand_backward": "yes", "expand_forward": "yes", "title": "Seed paper one", "doi": "10.1/s1"},
        {"ref": "2", "role": "other", "expand_backward": "yes", "expand_forward": "no", "title": "Seed paper two", "doi": "10.1/s2"}])
    cfg.data["corpus"]["backfill_abstracts"] = False
    build.Http = FakeHttp
    build.run(cfg, args())
    c = read_csv(cfg.candidates)
    ids = [r["openalex_id"] for r in c]
    check(not {"S1", "S2"} & set(ids), "seeds never become candidates")
    check("OLD" not in ids and "BOOK" not in ids, "year and type filters applied")
    check(sorted(ids) == ["A", "B", "C", "F1"], f"title dedupe keeps one of A/F2 ({ids})")
    check(c[0]["linked_to_core"] == "1" and c[-1]["linked_to_core"] == "0", "core-linked candidates rank first")
    check(next(r for r in c if r["openalex_id"] == "B")["seed_refs"] == "1|2", "a candidate keeps every linking seed")
    return cfg


# --- eligibility -------------------------------------------------------------------

def test_eligibility(cfg, tmp):
    print("\n[eligibility]")
    e = cfg["eligibility"]
    check(eligibility.classify("10.1/x", True, None, e)[0] is True, "OpenAlex OA -> eligible")
    check(eligibility.classify("10.1/x", False, {"outcome": "ok", "n_oa_locations": 0}, e)[1] == "no_oa", "closed -> no_oa")
    check(eligibility.classify("10.1016/x", True, None, {**e, "exclude_doi_prefixes": ["10.1016"]})[1] == "doi_prefix",
          "an optional DOI-prefix rule excludes even an OA paper")
    try:
        eligibility.classify("10.1/x", False, None, e)
        check(False, "missing Unpaywall answer must be undecided")
    except eligibility.Undecided:
        check(True, "a missing Unpaywall answer is undecided, never guessed")

    before = open(cfg.candidates, "rb").read()
    log = [{"source": "openalex", "openalex_id": i, "outcome": "ok", "is_oa": i != "C"} for i in ("A", "B", "F1")]
    with open(cfg.path("data", "eligibility_calls.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in log) + "\n")
    check(raises_exit(eligibility.run, cfg, args(offline=True)) is not None,
          "a missing lookup under --offline aborts")
    check(not os.path.exists(cfg.eligibility), "and nothing is written")
    with open(cfg.path("data", "eligibility_calls.jsonl"), "a") as f:
        f.write(json.dumps({"source": "openalex", "openalex_id": "C", "outcome": "not_found"}) + "\n")
    check(eligibility.run(cfg, args(offline=True)) == 5, "an undecidable row (deleted record, no DOI) aborts")
    write_csv(cfg.path("data", "eligibility_overrides.csv"),
              [{"openalex_id": "C", "eligible": "False", "exclusion_reason": "no_oa", "reason": "record deleted",
                "decided_by": "tester", "decided_on": "2026-10-11"}])
    check(eligibility.run(cfg, args(offline=True)) == 0, "an override decides only the undecidable row")
    check(open(cfg.candidates, "rb").read() == before, "candidates.csv is never modified")
    eg = eligibility.load(cfg)
    check(eg["C"]["eligibility_rule"] == "manual_override" and eg["A"]["eligible"], "decisions as expected")
    rows = read_csv(cfg.candidates)
    write_csv(cfg.candidates, rows[::-1])
    check(raises_exit(eligibility.load, cfg) is not None, "a reordered corpus makes eligibility STALE")
    open(cfg.candidates, "wb").write(before)


# --- stub Anthropic ---------------------------------------------------------------

SENT: list[dict] = []


class Msg:
    def __init__(self, inp, words=8):
        self.stop_reason = "tool_use"
        self.content = [types.SimpleNamespace(type="tool_use", name="screen_paper", input=inp)]
        self.usage = types.SimpleNamespace(input_tokens=1000, output_tokens=50,
                                           cache_creation_input_tokens=0, cache_read_input_tokens=0)

    def to_json(self):
        return json.dumps({"content": [c.input for c in self.content]})


class Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def stub_anthropic(served=("claude-opus-5-5",), tokens=1000, reason="Presents a method for the domain."):
    class Messages:
        def count_tokens(self, **kw):
            SENT.append({"surface": "count_tokens", **kw})
            return types.SimpleNamespace(input_tokens=tokens)

        def stream(self, **kw):
            SENT.append({"surface": "stream", **kw})
            return Stream(Msg({"decision": "include", "criterion_id": "I1", "reason": reason}))

    class Client:
        def __init__(self):
            self.messages = Messages()
            self.models = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(id=m) for m in served])

    llm.anthropic = types.SimpleNamespace(Anthropic=Client, APIError=Exception)


def test_calibrate_and_screen(cfg, tmp):
    print("\n[calibrate + screen]")
    rows = read_csv(cfg.candidates)
    big = [dict(rows[0], openalex_id=f"P{i:03d}", title=f"Paper {i}", doi=f"10.2/{i}",
                abstract=f"Abstract {i}", abstract_source="openalex") for i in range(90)]
    big[5].update(abstract="", abstract_source="none")      # title-only
    write_csv(cfg.candidates, big)
    elig_rows = [{"openalex_id": r["openalex_id"], "eligible": str(i % 10 != 3),
                  "exclusion_reason": "none" if i % 10 != 3 else "no_oa", "eligibility_rule": "fixture",
                  "candidates_key": eligibility.key(big)} for i, r in enumerate(big)]
    write_csv(cfg.eligibility, elig_rows)
    cfg.data["screening"]["calibration_per_stratum"] = 5
    calibrate.draw(cfg, args())
    first = read_csv(cfg.path("screening", "calibration_sample.csv"))
    check(all(r["abstract"] for r in first), "the sample holds only eligible papers with abstracts")
    calibrate.draw(cfg, args())
    check([r["paper_id"] for r in read_csv(cfg.path("screening", "calibration_sample.csv"))]
          == [r["paper_id"] for r in first], "the draw is reproducible")
    check(os.path.isdir(cfg.path("screening", "archive")), "a previous sample is archived, never overwritten")
    check(set(read_csv(cfg.path("screening", "labels_TODO.csv"))[0]) ==
          {"paper_id", "title", "abstract", "human_decision", "human_criterion_id", "human_notes"},
          "the blind sheet carries no rank, stratum or seed columns")

    stub_anthropic()
    SENT.clear()
    screen.run(cfg, args(only=",".join(r["openalex_id"] for r in big[:20])))
    texts = [s["messages"][0]["content"] for s in SENT if s.get("messages")]
    check(texts and not any(t.startswith(("Title: Paper 3\n", "Title: Paper 13\n")) for t in texts),
          "ineligible papers never reach any API surface (token counting included)")
    run = sorted(os.listdir(cfg.path("screening", "runs")))[-1]
    calls = read_jsonl(cfg.path("screening", "runs", run, "calls.jsonl"))
    check(not {"P003", "P013"} & {c["paper_id"] for c in calls}, "and get no call-log row or forced decision")
    forced = [c for c in calls if c["paper_id"] == "P005"]
    check(forced and forced[0]["validation_status"] == "forced"
          and not any(t.startswith("Title: Paper 5\n") for t in texts),
          "a title-only paper is forced to uncertain with no model call")
    man = read_json(cfg.path("screening", "runs", run, "manifest.json"))
    check(man["excluded_at_eligibility"]["n"] == 2, "the manifest records eligibility exclusions")

    stub_anthropic(tokens=10_000_000)
    n_runs = len(os.listdir(cfg.path("screening", "runs")))
    check(screen.run(cfg, args(only="P000,P001")) == 2 and len(os.listdir(cfg.path("screening", "runs"))) == n_runs,
          "an estimate over max_usd aborts before submitting")
    stub_anthropic(served=("claude-opus-5",))
    check(raises_exit(screen.run, cfg, args(only="P000")) is not None, "an unserved model ID aborts (never substituted)")

    sc = cfg["screening"]
    ok = {"decision": "include", "criterion_id": "I1"}
    check(screen.validate({**ok, "reason": "w " * 45}, None, {"I1"}, sc)[0:3:2] == ("valid", ["reason_overrun:45w"]),
          "a 45-word reason is valid with a warning")
    check(screen.validate({**ok, "reason": "w " * 50}, None, {"I1"}, sc)[0] == "invalid", "50+ words fails")
    check(screen.validate({**ok, "criterion_id": "Z9", "reason": "r"}, None, {"I1"}, sc)[0] == "invalid",
          "an unknown criterion ID fails")

    stub_anthropic()
    screen.run(cfg, args(sample=True))
    sample = read_csv(cfg.path("screening", "calibration_sample.csv"))
    labels = [{"paper_id": r["paper_id"], "title": r["title"], "abstract": "",
               "human_decision": "include" if i < 6 else "exclude", "human_criterion_id": "I1"}
              for i, r in enumerate(sample)]
    write_csv(cfg.path("screening", "labels.csv"), [dict(labels[0], human_decision="")] + labels[1:])
    check(raises_exit(calibrate.score, cfg, args()) is not None, "a partly filled label sheet is refused")
    write_csv(cfg.path("screening", "labels.csv"), labels)
    calibrate.score(cfg, args())
    res = read_json(cfg.provenance)["calibration"]
    # The stub always answers include: 6 human includes all caught -> recall 1.0;
    # 4 human excludes all false includes -> precision 6/10, binary kappa 0.
    check(res["include_recall"] == 1.0 and res["include_precision"] == 0.6 and res["kappa_binary"] == 0.0,
          f"recall/precision/kappa match the hand count ({res['include_recall']}, {res['include_precision']}, {res['kappa_binary']})")
    check(res["passed"] is False, "and the floors are not met")


# --- fetch / extract / report ------------------------------------------------------

def test_fetch_extract_report(cfg, tmp):
    print("\n[fetch + extract + report]")
    run = cfg.path("screening", "runs", "corpus_fixture")
    os.makedirs(run)
    calls = [{"paper_id": "P000", "validation_status": "valid", "parsed": {"decision": "include", "criterion_id": "I1"}},
             {"paper_id": "P001", "validation_status": "valid", "parsed": {"decision": "include", "criterion_id": "I2"}},
             {"paper_id": "P002", "validation_status": "invalid", "parsed": {"decision": "exclude", "criterion_id": "E1"}},
             {"paper_id": "P004", "validation_status": "valid", "parsed": {"decision": "uncertain", "criterion_id": "U1"}},
             {"paper_id": "P006", "validation_status": "valid", "parsed": {"decision": "exclude", "criterion_id": "E1"}}]
    with open(os.path.join(run, "calls.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(c) for c in calls) + "\n")
    write_csv(cfg.path("screening", "review_decisions.csv"),
              [{"paper_id": "P004", "decision": "include", "criterion_id": "I1", "decided_by": "tester"}])
    cfg.data["extraction"]["schemas"]["main"]["max_papers"] = 1
    fetch.run(cfg, args(run=run))
    oc = {r["paper_id"]: r["outcome"] for r in read_csv(cfg.path("screening", "outcomes.csv"))}
    check(oc["P002"] == "needs_human_review", "a failed response goes to human review, never to exclude")
    check(oc["P004"] == "included", "a human review decision is applied")
    check(oc["P003"] == "excluded_at_eligibility" and oc["P006"] == "excluded_at_screening", "stages kept apart")
    t = read_csv(cfg.path("extraction", "targets.csv"))
    check([x["paper_id"] for x in t] == ["P000"], "schema mapping with max_papers keeps the top by rank")

    import pymupdf
    pdf = os.path.join(tmp, "x.pdf")
    doc = pymupdf.open()
    for k in range(3):
        doc.new_page().insert_text((50, 72), f"Paper 0 study page {k}. " + "Body text of the paper. " * 8)
        doc[-1].insert_textbox(pymupdf.Rect(50, 100, 550, 800), "Lorem ipsum dolor sit amet. " * 60)
    doc.save(pdf)
    shutil.copy(pdf, cfg.path("manual_pdfs", "download (3).pdf"))
    fetch.run(cfg, args(run=run))
    check(os.path.exists(cfg.path("documents", "P000.txt")), "a manual PDF under any filename is matched and validated")

    schema = read_json(cfg.path("schemas", "example_schema.json"))
    flds = dict(extract.fields_of(schema))
    paper = extract.Paper(open(cfg.path("documents", "P000.txt")).read())
    q = "Body text of the paper. Body text of the paper."
    spec = flds["A_CONTRIBUTION__contribution_type"]
    absent = set(extract.ABSENCE)
    cases = {"ok": {"value": "method", "evidence_quote": q, "page": 1},
             "page_mismatch": {"value": "method", "evidence_quote": "Paper 0 study page 2", "page": 1},
             "quote_not_found": {"value": "method", "evidence_quote": "an invented sentence", "page": 1},
             "quote_too_long": {"value": "method", "evidence_quote": "w " * 30, "page": 1},
             "bad_value": {"value": "not_a_member", "evidence_quote": q, "page": 1},
             "not_reported": {"value": "not_reported", "evidence_quote": None, "page": None},
             "missing_quote": {"value": "method", "evidence_quote": None, "page": None}}
    for want, o in cases.items():
        check(extract.check(o, paper, spec, absent, 25) == want, f"evidence check: {want}")
    check(extract.check({"value": ["none_reported"], "evidence_quote": None},
                        paper, flds["B_VALIDATION__metrics_reported"], absent, 25) == "absence_unquoted",
          "evidence check: absence_unquoted")

    os.makedirs(cfg.path("extraction", "records"), exist_ok=True)
    rec = {"paper_id": "P000", "schema": "main", "response": "valid", "fields": {
        "A_CONTRIBUTION__contribution_type": {"value": "method", "check": "ok"},
        "B_VALIDATION__validation_level": {"value": "simulation", "check": "quote_not_found"}}}
    json.dump(rec, open(cfg.path("extraction", "records", "P000.json"), "w"))
    agg = report.aggregate(cfg)
    check(agg["tables"]["main__A_CONTRIBUTION__contribution_type"]["n"] == 1, "a passing field is counted")
    check(agg["tables"]["main__B_VALIDATION__validation_level"]["n"] == 0, "a failed field is not")
    check(report.run(cfg, args()) == 0 and os.path.exists(cfg.path("report.md")), "the scaffolded report renders")
    with open(cfg.path("report_template.md"), "a") as f:
        f.write("\n{{agg.no.such.value}}\n")
    os.remove(cfg.path("report.md"))
    check(report.run(cfg, args()) == 1 and not os.path.exists(cfg.path("report.md")),
          "an unresolved placeholder writes nothing")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="litreview_test_")
    try:
        cfg = test_build(tmp)
        test_eligibility(cfg, tmp)
        test_calibrate_and_screen(cfg, tmp)
        test_fetch_extract_report(cfg, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} FAILURE(S)" if FAILS else "\nALL CHECKS PASSED")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
