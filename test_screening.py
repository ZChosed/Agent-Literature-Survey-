#!/usr/bin/env python3
"""
test_screening.py -- offline test for the screening calibration pipeline.

Nothing here touches the network. The Anthropic client is replaced by a fake
that returns canned Message-shaped objects, so the budget guard, the response
parser and the validator are exercised against the shapes they will actually
meet -- including the malformed ones, which are the point.

Covers, in the order the pipeline runs:

  * draw_calibration_sample.py -- the fixed RNG seed reproduces the draw, both
    strata hold exactly 25, only eligible rows with abstract text can be drawn
    (no ineligible row, no abstract_source=none row, no full-text-head row),
    and the shuffle does not group the strata
  * screen.py's eligibility gate -- an ineligible row never reaches any API
    surface (token counting, batch, sync), gets no forced decision and no
    calls.jsonl row, and is recorded in the manifest instead
  * benchmark.py's two exclusion stages -- eligibility and screening are
    counted apart and an ineligible labelled paper is not scored
  * labels_TODO.csv -- carries no stratum, no rank, no seed linkage: the
    blinding is asserted against a list of banned columns AND against the
    stratum values themselves
  * screen.py's isolation check -- a stratum value reaching the prompt raises
  * screen.py's parser and validator -- a good tool call passes; missing tool
    use, a bad decision, an over-long reason, a hallucinated criterion_id, a
    refusal and a max_tokens truncation each fail, and none of them is retried
  * screen.py's prompt hashes -- both change when criteria.md changes, and the
    system hash changes when only the template around the criteria changes
  * screen.py's budget guard -- aborts above the cap and submits nothing
  * benchmark.py's kappa, recall and precision -- against a hand-computed toy
    example whose arithmetic is written out in the test
  * benchmark.py's refusal -- an unfilled labels.csv row stops the scoring

Run:  python test_screening.py
Exit 0 = all assertions passed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import pandas as pd

import benchmark as bm
import draw_calibration_sample as draw
import screen as sc

FAILS: list[str] = []


def check(cond, label):
    if cond:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILS.append(label)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CRITERIA_A = """\
# Screening criteria (toy)

## I1 Agent architecture
Include if the paper presents a multi-agent architecture for manufacturing.

## I2 Digital twin coupling
Include if it couples a digital twin to a decision-making agent.

## E1 Out of scope
Exclude if the paper is not about manufacturing systems.

## E2 Wrong artifact type
Exclude if the paper is an editorial, erratum or call for papers.

## U1 Insufficient abstract
Uncertain if the abstract does not say enough to apply I1, I2, E1 or E2.
"""

CRITERIA_B = CRITERIA_A.replace("multi-agent architecture",
                                "multi-agent control architecture")


def make_candidates(path: str, n: int = 300, missing_every: int = 5,
                    ineligible_every: int = 7) -> None:
    """
    A ranked candidates.csv. Every `missing_every`-th row has no abstract --
    alternately `none` and a full-text-head row, which is not `none` but has
    no abstract either -- and every `ineligible_every`-th row is a 10.1016 DOI
    marked ineligible. So the pool is not the full set and the strata cannot
    be read off the raw row numbers.
    """
    cols = ["openalex_id", "doi", "title", "year", "venue", "authors", "type",
            "cited_by_count", "citations_per_year", "n_seed_links",
            "seed_weight", "seed_refs", "seed_roles",
            "linked_to_framework_core", "direction", "oa_pdf_url",
            "abstract", "abstract_source", "fulltext_head",
            "eligible", "exclusion_reason", "eligibility_rule"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(1, n + 1):
            has = (i % missing_every) != 0
            inel = (i % ineligible_every) == 0
            src = ("openalex" if has else
                   "fulltext_head_oa" if i % (2 * missing_every) == 0 else "none")
            w.writerow({
                "openalex_id": f"W{100000 + i}",
                "doi": f"10.1016/test.{i}" if inel else f"10.0000/test.{i}",
                "title": f"Paper number {i} on agents and manufacturing",
                "year": 2015 + (i % 11),
                "venue": f"Journal {i % 7}",
                "authors": "A Author; B Author",
                "type": "article",
                "cited_by_count": n - i,
                "citations_per_year": round((n - i) / 3, 2),
                "n_seed_links": 1 + (i % 4),
                "seed_weight": round(1.0 + (n - i) / n, 4),
                "seed_refs": "7|10",
                "seed_roles": "framework_core",
                "linked_to_framework_core": 1 if i < 50 else 0,
                "direction": "backward",
                "oa_pdf_url": "",
                "abstract": (f"This abstract for paper {i} describes an agent "
                             f"architecture and its evaluation.") if has else "",
                "abstract_source": src,
                "fulltext_head": (f"Introduction to paper {i}. " * 5
                                  if src == "fulltext_head_oa" else ""),
                "eligible": "False" if inel else "True",
                "exclusion_reason": "elsevier" if inel else "none",
                "eligibility_rule": "doi_prefix_10.1016" if inel else "openalex_is_oa",
            })


def run_draw(tmpdir: str, rng_seed: int = 20260928, n: int = 25):
    cand = os.path.join(tmpdir, "candidates.csv")
    if not os.path.exists(cand):
        make_candidates(cand)
    outdir = os.path.join(tmpdir, f"screening_{rng_seed}_{n}")
    prov = os.path.join(tmpdir, f"prov_{rng_seed}_{n}.json")
    rc = subprocess.run(
        [sys.executable, "draw_calibration_sample.py",
         "--candidates", cand, "--out-dir", outdir, "--provenance", prov,
         "--rng-seed", str(rng_seed), "--n-per-stratum", str(n), "--quiet"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    if rc.returncode != 0:
        print(rc.stdout, rc.stderr)
        raise SystemExit(f"draw failed: {rc.returncode}")
    sample = list(csv.DictReader(open(os.path.join(outdir, "calibration_sample.csv"),
                                      encoding="utf-8")))
    labels = list(csv.DictReader(open(os.path.join(outdir, "labels_TODO.csv"),
                                      encoding="utf-8")))
    with open(prov, encoding="utf-8") as f:
        block = json.load(f)["calibration_sample"]
    return sample, labels, block, outdir


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

class FakeUsage:
    def __init__(self, i=1200, cr=0, cw=0, o=40):
        self.input_tokens = i
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw
        self.output_tokens = o


class FakeBlock:
    def __init__(self, btype, name=None, inp=None, text=None):
        self.type = btype
        self.name = name
        self.input = inp
        self.text = text


class FakeMessage:
    def __init__(self, blocks, stop_reason="tool_use", usage=None,
                 stop_details=None):
        self.content = blocks
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = usage or FakeUsage()

    def to_json(self):
        return json.dumps({
            "stop_reason": self.stop_reason,
            "content": [{"type": b.type, "name": b.name, "input": b.input}
                        for b in self.content],
        })


def tool_msg(decision="include", cid="I1", reason="Presents a multi-agent architecture."):
    return FakeMessage([FakeBlock("tool_use", sc.TOOL_NAME,
                                  {"decision": decision, "criterion_id": cid,
                                   "reason": reason})])


class FakeCountTokens:
    def __init__(self, n):
        self.input_tokens = n


class FakeMessages:
    def __init__(self, token_count):
        self._tc = token_count

    def count_tokens(self, **kwargs):
        return FakeCountTokens(self._tc)


class FakeClient:
    """Enough of anthropic.Anthropic for the budget guard."""
    def __init__(self, token_count=1500):
        self.messages = FakeMessages(token_count)


STUB_ANTHROPIC = '''\
"""Offline stand-in for the anthropic SDK, used only by test_screening.py.

It implements exactly the surface screen.py touches before it submits anything:
a client whose messages.count_tokens returns a fixed token count, the APIError
type screen.py catches, and the two params modules it imports for batching.
Batch creation deliberately raises -- no test here should ever reach it, and a
stub that silently "succeeded" would hide a test that did.
"""
import json as _json
import os as _os

TOKEN_COUNT = __TOKEN_COUNT__


def _record(surface, kw):
    """Append every request this stub receives to $STUB_RECORD, if set."""
    path = _os.environ.get("STUB_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps({"surface": surface, "kw": kw}, default=str) + "\\n")


class APIError(Exception):
    pass


class _Count:
    def __init__(self, n):
        self.input_tokens = n


class _Batches:
    def create(self, **kw):
        _record("batches.create", kw)
        raise APIError("stub anthropic: batch submission attempted")

    def retrieve(self, *a, **kw):
        raise APIError("stub anthropic: batch retrieve attempted")

    def results(self, *a, **kw):
        raise APIError("stub anthropic: batch results attempted")


class _Messages:
    def __init__(self):
        self.batches = _Batches()

    def count_tokens(self, **kw):
        # The real token-counting endpoint applies the same forced-tool-choice
        # check as Messages, so a harness that still sent tool_choice type
        # "tool" would fail here on the 5.5 models -- before spending anything.
        _record("count_tokens", kw)
        tc = kw.get("tool_choice") or {}
        if tc.get("type") in ("tool", "any"):
            raise APIError(
                'tool_choice: type "tool" and "any" are not supported for '
                'this model.')
        return _Count(TOKEN_COUNT)

    def create(self, **kw):
        _record("messages.create", kw)
        raise APIError("stub anthropic: sync create attempted")


class _Model:
    def __init__(self, mid):
        self.id = mid
        self.display_name = mid
        self.max_input_tokens = 1000000
        self.max_tokens = 128000


SERVED = __SERVED__


class _Models:
    def list(self, **kw):
        return [_Model(m) for m in SERVED]


class Anthropic:
    def __init__(self, *a, **kw):
        self.messages = _Messages()
        self.models = _Models()
'''

STUB_TYPES_MCP = (
    "class MessageCreateParamsNonStreaming(dict):\n"
    "    def __init__(self, **kw):\n"
    "        super().__init__(**kw)\n"
)

STUB_TYPES_BCP = (
    "class Request(dict):\n"
    "    def __init__(self, **kw):\n"
    "        super().__init__(**kw)\n"
)


def _stub_anthropic(tmpdir: str, token_count: int,
                    served=None) -> str:
    """
    Write a stub `anthropic` package and return the directory to put on
    PYTHONPATH. Lets the budget-guard subprocess run with no network and no
    API key, so a pass means the guard fired -- not that the machine was
    offline.
    """
    if served is None:
        served = list(sc.MODEL_IDS.values())
    tag = hashlib.sha256(
        (repr(sorted(served)) + str(token_count)).encode()).hexdigest()[:12]
    root = os.path.join(tmpdir, "stubpath_" + tag)
    pkg = os.path.join(root, "anthropic")
    types_dir = os.path.join(pkg, "types")
    msgs_dir = os.path.join(types_dir, "messages")
    os.makedirs(msgs_dir, exist_ok=True)

    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        # Plain string replacement, not .format(): the stub source contains
        # literal braces (dict literals), which .format() would try to expand.
        f.write(STUB_ANTHROPIC
                .replace('__SERVED__', repr(list(served)))
                .replace('__TOKEN_COUNT__', str(token_count)))
    open(os.path.join(types_dir, "__init__.py"), "w").close()
    with open(os.path.join(types_dir, "message_create_params.py"), "w",
              encoding="utf-8") as f:
        f.write(STUB_TYPES_MCP)
    open(os.path.join(msgs_dir, "__init__.py"), "w").close()
    with open(os.path.join(msgs_dir, "batch_create_params.py"), "w",
              encoding="utf-8") as f:
        f.write(STUB_TYPES_BCP)
    return root


# ---------------------------------------------------------------------------
# Tests: the draw
# ---------------------------------------------------------------------------

def test_sample_draw(tmpdir: str):
    print("\n[sample] stratified draw: reproducibility, strata, eligibility")
    sample, labels, block, _ = run_draw(tmpdir)

    check(len(sample) == 50, f"50 papers drawn (got {len(sample)})")

    strata = {}
    for r in sample:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    check(strata.get("top_third") == 25,
          f"25 from the top third (got {strata.get('top_third')})")
    check(strata.get("bottom_two_thirds") == 25,
          f"25 from the bottom two-thirds (got {strata.get('bottom_two_thirds')})")

    # The whole point of filtering before cutting: nothing out of scope and
    # nothing unscreenable is drawn.
    cand = {r["openalex_id"]: r for r in csv.DictReader(
        open(os.path.join(tmpdir, "candidates.csv"), encoding="utf-8"))}
    check(all(cand[r["paper_id"]]["eligible"] == "True" for r in sample),
          "no ineligible row was drawn")
    check(not any(r["doi"].startswith("10.1016") for r in sample),
          "in particular no 10.1016 row")
    check(all(r["abstract_source"] in ("openalex", "semantic_scholar")
              for r in sample),
          "no abstract_source=none row and no full-text-head row was drawn")
    check(all(r["abstract"].strip() for r in sample),
          "every drawn row actually carries abstract text")

    # Reproducibility: same seed, same 50 papers, same strata.
    sample2, _, block2, _ = run_draw(tmpdir, rng_seed=20260928)
    ids1 = [r["paper_id"] for r in sample]
    ids2 = [r["paper_id"] for r in sample2]
    check(ids1 == ids2, "the fixed RNG seed reproduces the draw exactly, in order")
    check(block["drawn_paper_ids"] == block2["drawn_paper_ids"],
          "and reproduces the per-stratum membership")

    # The redraw is reproducible byte for byte: two independent runs into
    # separate directories write identical sample AND blind sheet files.
    _, _, _, out_a = run_draw(tmpdir, rng_seed=20260928)
    out_b = os.path.join(tmpdir, "redraw_b")
    rc = subprocess.run(
        [sys.executable, "draw_calibration_sample.py",
         "--candidates", os.path.join(tmpdir, "candidates.csv"),
         "--out-dir", out_b, "--provenance", os.path.join(tmpdir, "prov_b.json"),
         "--rng-seed", "20260928", "--n-per-stratum", "25", "--quiet"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    for name in ("calibration_sample.csv", "labels_TODO.csv"):
        a = open(os.path.join(out_a, name), "rb").read()
        b = open(os.path.join(out_b, name), "rb").read()
        check(rc.returncode == 0 and a == b,
              f"redraw is byte-identical across runs: {name}")

    # A different seed must actually move the draw, or "fixed seed" means nothing.
    sample3, _, _, _ = run_draw(tmpdir, rng_seed=1)
    check(set(r["paper_id"] for r in sample3) != set(ids1),
          "a different RNG seed draws a different sample")

    check(block["rng_seed"] == 20260928,
          f"rng_seed recorded in provenance (got {block.get('rng_seed')})")
    check(block["strata"]["top_third"]["drawn"] == 25
          and block["strata"]["bottom_two_thirds"]["drawn"] == 25,
          "provenance records the per-stratum draw counts")
    # Fixture: 300 rows; every 7th ineligible (42); every 5th lacks an
    # abstract (60), of which every 35th is also ineligible (8), so 52
    # eligible rows lack one.
    check(block["excluded_ineligible"] == 42,
          f"provenance records the 42 ineligible exclusions "
          f"(got {block['excluded_ineligible']})")
    check(block["excluded_no_abstract"] == 52,
          f"and the 52 eligible-but-no-abstract exclusions, counted apart "
          f"(got {block['excluded_no_abstract']})")
    check(block["eligible_with_abstract"] == 300 - 42 - 52,
          f"pool = 300 - 42 - 52 = 206 (got {block['eligible_with_abstract']})")
    g = block["pool_exclusion_by_full_list_third"]
    check(g["top_third"]["ineligible"] + g["bottom_two_thirds"]["ineligible"] == 42,
          "the per-third exclusion split adds back up to the total")

    # Shuffled: the strata must be interleaved, not blocked. With 25/25 the
    # probability of the first 25 being one stratum by chance is ~1/1e14.
    first25 = {r["stratum"] for r in sample[:25]}
    check(len(first25) == 2,
          "the 50 are shuffled -- position does not reveal the stratum")

    # The eligible-list cut, not the full-list cut.
    top_rank_hi = block["strata"]["top_third"]["eligible_rank_range"][1]
    check(top_rank_hi == block["eligible_with_abstract"] // 3,
          f"thirds are cut on the ELIGIBLE list "
          f"({top_rank_hi} == {block['eligible_with_abstract']}//3)")


def make_fulltext_candidates(path: str, n: int = 150) -> None:
    """A candidates.csv holding a mix of abstract rows, head rows and neither."""
    cols = ["openalex_id", "doi", "title", "year", "venue", "authors", "type",
            "cited_by_count", "citations_per_year", "n_seed_links",
            "seed_weight", "seed_refs", "seed_roles",
            "linked_to_framework_core", "direction", "oa_pdf_url",
            "abstract", "abstract_source", "fulltext_head",
            "eligible", "exclusion_reason", "eligibility_rule"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(1, n + 1):
            if i % 3 == 0:
                src, abstract, head = "none", "", ""
            elif i % 3 == 1:
                src, abstract, head = "openalex", f"Abstract {i}.", ""
            else:
                src = "fulltext_head_oa"
                abstract, head = "", f"Introduction for paper {i}. " * 5
            w.writerow({
                "openalex_id": f"W{200000 + i}", "doi": f"10.0000/ft.{i}",
                "title": f"Full text paper {i}", "year": 2020 + (i % 5),
                "venue": f"Journal {i % 4}", "authors": "A Author",
                "type": "article", "cited_by_count": n - i,
                "citations_per_year": 1.0, "n_seed_links": 1,
                "seed_weight": 1.0, "seed_refs": "7", "seed_roles": "framework_core",
                "linked_to_framework_core": 0, "direction": "backward",
                "oa_pdf_url": "", "abstract": abstract,
                "abstract_source": src, "fulltext_head": head,
                "eligible": "True", "exclusion_reason": "none",
                "eligibility_rule": "openalex_is_oa",
            })


def test_fulltext_sample(tmpdir: str):
    print("\n[ft-sample] the full-text sample is separate, blind and reproducible")
    cand = os.path.join(tmpdir, "ft_candidates.csv")
    make_fulltext_candidates(cand)
    outdir = os.path.join(tmpdir, "screening_ft")
    prov = os.path.join(tmpdir, "prov_ft.json")

    def run(seed=20260928, n=20, out=outdir, pv=prov):
        rc = subprocess.run(
            [sys.executable, "draw_calibration_sample.py", "--fulltext",
             "--candidates", cand, "--out-dir", out, "--provenance", pv,
             "--rng-seed", str(seed), "--n-fulltext", str(n), "--quiet"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)))
        if rc.returncode != 0:
            print(rc.stdout, rc.stderr)
            raise SystemExit("fulltext draw failed")
        sample = list(csv.DictReader(open(
            os.path.join(out, "calibration_sample_fulltext.csv"), encoding="utf-8")))
        labels = list(csv.DictReader(open(
            os.path.join(out, "labels_fulltext_TODO.csv"), encoding="utf-8")))
        block = json.load(open(pv, encoding="utf-8"))["calibration_sample_fulltext"]
        return sample, labels, block

    sample, labels, block = run()
    check(len(sample) == 20, f"20 papers drawn (got {len(sample)})")
    check(all(r["abstract_source"] == "fulltext_head_oa" for r in sample),
          "every drawn row is a full-text-head row")
    check(all(r["fulltext_head"].strip() for r in sample),
          "and every one actually carries head text")
    check(block["stratified"] is False,
          "the draw is recorded as unstratified")
    check(block["rng_seed"] == 20260928,
          f"seed 20260928 recorded (got {block['rng_seed']})")

    # Reproducible, and the seed actually matters.
    s2, _, _ = run(out=os.path.join(tmpdir, "screening_ft2"),
                   pv=os.path.join(tmpdir, "prov_ft2.json"))
    check([r["paper_id"] for r in sample] == [r["paper_id"] for r in s2],
          "the fixed seed reproduces the draw exactly")
    s3, _, _ = run(seed=7, out=os.path.join(tmpdir, "screening_ft3"),
                   pv=os.path.join(tmpdir, "prov_ft3.json"))
    check(block["eligible_pool"] > 20,
          f"the fixture pool is larger than the draw ({block['eligible_pool']}), "
          f"so the seed has something to choose between")
    check({r["paper_id"] for r in s3} != {r["paper_id"] for r in sample},
          "a different seed draws a different sample")

    # Blind, and the text column is named for what it holds.
    check(list(labels[0].keys()) == ["paper_id", "title", "fulltext_head",
                                     "human_decision", "human_criterion_id",
                                     "human_notes"],
          f"blind sheet columns (got {list(labels[0].keys())})")
    check("abstract" not in labels[0],
          "the sheet never calls a truncated head an abstract -- the rater has "
          "to know a missing result may just be past the cut")
    banned = {"stratum", "rank", "seed_weight", "n_seed_links", "year", "venue",
              "doi", "abstract_source"}
    check(not (banned & set(labels[0].keys())),
          "no ranking, stratum or provenance column leaks into the sheet")
    check(all(not r["human_decision"] and not r["human_criterion_id"]
              and not r["human_notes"] for r in labels),
          "all three human_* columns are empty")

    # Must not overlap the abstract sample -- impossible by construction, but
    # the whole point of the separate draw is that they stay apart.
    abs_sample, _, _, _ = run_draw(tmpdir)
    check(not ({r["paper_id"] for r in sample}
               & {r["paper_id"] for r in abs_sample}),
          "no paper appears in both the abstract and full-text samples")

    check(block["thresholds"]["kappa_binary_min"] == 0.70
          and block["thresholds"]["include_recall_min"] == 0.95,
          "the same floors apply to this sample as to the abstract one")
    check(block["eligible_pool"] >= 20,
          f"the eligible pool is recorded ({block['eligible_pool']}) so the "
          f"denominator behind include-recall is visible")

    # A pool too small to draw from must fail loudly, not draw fewer.
    small = os.path.join(tmpdir, "ft_small.csv")
    make_fulltext_candidates(small, n=9)
    rc = subprocess.run(
        [sys.executable, "draw_calibration_sample.py", "--fulltext",
         "--candidates", small, "--out-dir", os.path.join(tmpdir, "ft_small_out"),
         "--provenance", os.path.join(tmpdir, "prov_small.json"), "--quiet"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(rc.returncode != 0,
          f"too few head rows aborts rather than silently drawing fewer "
          f"(rc={rc.returncode})")
    check("need 20" in rc.stdout + rc.stderr, "and says how many it needed")


def test_labels_sheet_is_blind(tmpdir: str):
    print("\n[sheet] labels_TODO.csv reveals nothing about the ranking")
    sample, labels, _, _ = run_draw(tmpdir)

    check(list(labels[0].keys()) == ["paper_id", "title", "abstract",
                                     "human_decision", "human_criterion_id",
                                     "human_notes"],
          f"exact column set (got {list(labels[0].keys())})")

    banned = {"stratum", "rank", "seed_weight", "n_seed_links", "seed_refs",
              "seed_roles", "linked_to_framework_core", "cited_by_count",
              "citations_per_year", "direction", "year", "venue", "doi"}
    present = banned & set(labels[0].keys())
    check(not present, f"no ranking/stratum column present (found {present})")

    check(all(r["human_decision"] == "" and r["human_criterion_id"] == ""
              and r["human_notes"] == "" for r in labels),
          "all three human_* columns are empty")

    check(len(labels) == 50, f"50 rows to label (got {len(labels)})")

    # Same order as the sample file, so the two can be joined -- but since the
    # sample was shuffled, shared order leaks nothing.
    check([r["paper_id"] for r in labels] == [r["paper_id"] for r in sample],
          "row order matches calibration_sample.csv (both already shuffled)")

    # Belt and braces: no stratum VALUE anywhere in the sheet's bytes.
    blob = json.dumps(labels)
    check("top_third" not in blob and "bottom_two_thirds" not in blob,
          "no stratum value appears anywhere in the sheet")


# ---------------------------------------------------------------------------
# Tests: prompt hashes and isolation
# ---------------------------------------------------------------------------

def test_prompt_hashes():
    print("\n[hash] criteria and system-prompt hashes track criteria.md")
    sys_a = sc.build_system_text(CRITERIA_A, "criteria.md")
    sys_b = sc.build_system_text(CRITERIA_B, "criteria.md")

    crit_a, crit_b = sc.sha256(CRITERIA_A), sc.sha256(CRITERIA_B)
    sha_a, sha_b = sc.sha256(sys_a), sc.sha256(sys_b)

    check(crit_a != crit_b,
          "criteria.md hash changes when the criteria text changes")
    check(sha_a != sha_b,
          "system-prompt hash changes when the criteria text changes")
    check(sc.sha256(sc.build_system_text(CRITERIA_A, "criteria.md")) == sha_a,
          "same criteria text -> same system-prompt hash (cache prefix is stable)")

    # The criteria hash alone cannot detect a change in the instructions WRAPPED
    # around the criteria, which is why both hashes are logged, not just one.
    check(CRITERIA_A in sys_a, "the full criteria text is embedded in the system prompt")
    check(sc.TOOL_NAME in sys_a, "the system prompt names the tool it forces")
    check(str(sc.MAX_REASON_WORDS) in sys_a, "the system prompt states the word cap")

    # Cache minimums are model-dependent, and Haiku's is the one that bites.
    check(sc.CACHE_MIN_PREFIX["claude-haiku-4-5-20251001"] == 4096,
          "Haiku 4.5's 4096-token cache minimum is recorded")
    check(sc.CACHE_MIN_PREFIX["claude-opus-5-5"] == 512,
          "Opus 5.5's 512-token cache minimum is recorded")
    check(sc.CACHE_MIN_PREFIX["claude-sonnet-5-5"] == 512,
          "Sonnet 5.5's 512-token cache minimum is recorded")


def test_model_table():
    print("\n[models] the brief's model IDs are used verbatim, never substituted")
    check(sc.MODEL_IDS == {"haiku": "claude-haiku-4-5-20251001",
                           "sonnet": "claude-sonnet-5-5",
                           "opus": "claude-opus-5-5"},
          f"the three IDs are exactly as the brief specified (got {sc.MODEL_IDS})")
    check(not hasattr(sc, "BRIEF_MODEL_IDS"),
          "the substitution table is gone entirely")

    # Prices, verified against the pricing page. Getting Opus 5.5 wrong by
    # using Opus 5's $5/$25 inflates it 25% against a rule that picks on cost.
    check(sc.PRICING["claude-opus-5-5"]["input"] == 4.00
          and sc.PRICING["claude-opus-5-5"]["output"] == 20.00,
          "Opus 5.5 priced at $4/$20, not Opus 5's $5/$25")
    check(sc.PRICING["claude-sonnet-5-5"]["input"] == 2.00
          and sc.PRICING["claude-sonnet-5-5"]["output"] == 10.00,
          "Sonnet 5.5 priced at $2/$10")
    check(sc.PRICING["claude-haiku-4-5-20251001"]["input"] == 1.00
          and sc.PRICING["claude-haiku-4-5-20251001"]["output"] == 5.00,
          "Haiku 4.5 priced at $1/$5")

    # The cache-read multiplier is NOT global: Opus 5.5 reads at 0.05x.
    check(sc.PRICING["claude-opus-5-5"]["cache_read_mult"] == 0.05,
          "Opus 5.5 cache reads at 0.05x base input")
    check(sc.PRICING["claude-sonnet-5-5"]["cache_read_mult"] == 0.10,
          "Sonnet 5.5 cache reads at the standard 0.10x")
    check(not hasattr(sc, "CACHE_READ_MULT"),
          "no single global cache-read constant survives -- it is per model")

    # A cache-heavy call, priced by hand. 100k cache-read tokens on Opus 5.5:
    # 100_000 * ($4/1e6) * 0.05 = $0.02. At the old global 0.1x it would be
    # $0.04, and at Opus 5's $5 base it would be $0.025.
    c = sc.cost_usd("claude-opus-5-5",
                    {"input_tokens": 0, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 100_000, "output_tokens": 0},
                    batch=False)
    check(abs(c - 0.02) < 1e-12,
          f"100k Opus 5.5 cache-read tokens cost $0.02 (got ${c})")

    # An unpriced model must raise, not silently cost nothing: the selection
    # rule picks the cheapest, so $0.00 would win on a fiction.
    try:
        sc.cost_usd("claude-not-a-model", {"input_tokens": 1000}, batch=False)
        check(False, "an unpriced model must raise rather than cost $0.00")
    except KeyError as exc:
        check("refusing to price" in str(exc),
              "an unpriced model raises rather than costing $0.00")


def test_request_shape_per_model():
    print("\n[request] forced tool use is gone; thinking is minimised per model")

    # Forced tool_choice returns 400 on both 5.5 models, so the harness must
    # not send it to ANY of them -- a per-model request shape would make the
    # benchmark compare request shapes as well as models.
    src = open("screen.py", encoding="utf-8").read()
    check('tool_choice={"type": "auto"}' in src,
          "requests send tool_choice auto")
    check('"type": "tool", "name": TOOL_NAME' not in src,
          "no forced tool_choice is sent anywhere")

    check(sc.build_tool()["strict"] is True,
          "strict:true is on by default -- it replaces forcing as the "
          "schema-validity mechanism")
    check("strict" not in sc.build_tool(False),
          "--no-strict-tool can still drop it")

    # Haiku 4.5: omit thinking entirely, no effort parameter.
    kw = sc.model_kwargs("claude-haiku-4-5-20251001")
    check("thinking" not in kw and "output_config" not in kw,
          f"Haiku 4.5 sends no thinking and no effort (got {kw})")

    # Sonnet 5.5: `disabled` is a 400; between_tools is the lowest setting,
    # and it is only valid at low/medium/high effort.
    kw = sc.model_kwargs("claude-sonnet-5-5")
    check(kw.get("thinking") == {"type": "between_tools"},
          f"Sonnet 5.5 sends thinking=between_tools, not disabled (got {kw})")
    check(kw.get("output_config", {}).get("effort") in ("low", "medium", "high"),
          "and an effort level between_tools actually accepts")

    # Opus 5.5: thinking cannot be disabled at all; effort is the only lever.
    kw = sc.model_kwargs("claude-opus-5-5")
    check("thinking" not in kw,
          f"Opus 5.5 sends NO thinking field -- disabled would 400 (got {kw})")
    check(kw.get("output_config", {}).get("effort") == "low",
          "and effort=low, the only lever on thinking depth it has")
    check(sc.MODEL_REQUEST["claude-opus-5-5"]["thinking_unavoidable"] is True,
          "Opus 5.5 is flagged thinking_unavoidable so the cost comparison "
          "can be read with the asymmetry visible")
    check(sc.MODEL_REQUEST["claude-sonnet-5-5"]["thinking_unavoidable"] is False
          and sc.MODEL_REQUEST["claude-haiku-4-5-20251001"]["thinking_unavoidable"] is False,
          "the other two are not")

    # max_tokens now covers thinking + text, so a 300-token cap would truncate
    # every Opus 5.5 call into a max_tokens validation failure.
    check(sc.MAX_TOKENS >= 2048,
          f"max_tokens leaves room for thinking (got {sc.MAX_TOKENS})")


def test_text_kind_labelling():
    print("\n[input] the prompt says which artefact it is showing the model")

    abs_row = {"paper_id": "W1", "title": "T", "abstract": "An abstract.",
               "fulltext_head": "", "abstract_source": "openalex"}
    text, kind = sc.resolve_screening_text(abs_row)
    check((text, kind) == ("An abstract.", "abstract"),
          f"an openalex row screens on its abstract (got {kind})")

    s2_row = dict(abs_row, abstract_source="semantic_scholar")
    check(sc.resolve_screening_text(s2_row)[1] == "abstract",
          "so does a semantic_scholar row")

    ft_row = {"paper_id": "W2", "title": "T", "abstract": "",
              "fulltext_head": "Introduction. Manufacturing systems ...",
              "abstract_source": "fulltext_head_oa"}
    text, kind = sc.resolve_screening_text(ft_row)
    check(kind == "fulltext_head" and text.startswith("Introduction"),
          f"a fulltext_head_oa row screens on its head (got {kind})")

    none_row = {"paper_id": "W3", "title": "T", "abstract": "",
                "fulltext_head": "", "abstract_source": "none"}
    check(sc.resolve_screening_text(none_row) == ("", "title_only"),
          "a row with neither is title_only")

    # The label must actually reach the prompt, and must distinguish the two.
    p_abs = sc.build_user_text("A title", "An abstract.", "abstract")
    p_ft = sc.build_user_text("A title", "Introduction text.", "fulltext_head")
    check("Abstract:" in p_abs, "the abstract prompt labels it an Abstract")
    check("Full-text head" in p_ft,
          "the full-text prompt labels it a full-text head")
    check("NOT an abstract" in p_ft,
          "and says explicitly that it is not an abstract")
    check("truncated" in p_ft,
          "and that it is truncated")
    check("absence of a finding here is not evidence" in p_ft,
          "and warns that a missing result is not a missing result -- the head "
          "stops before the results section")
    check("Full-text head" not in p_abs and "Abstract:" not in p_ft,
          "the two labels do not bleed into each other")

    # Everything apart from the labelled body must be identical, or the
    # benchmark compares prompts rather than inputs.
    body_abs = p_abs.split("\n\n", 1)[0]
    body_ft = p_ft.split("\n\n", 1)[0]
    check(body_abs == body_ft == "Title: A title",
          f"the title line is identical across kinds ({body_abs!r} vs {body_ft!r})")

    p_none = sc.build_user_text("A title", "", "title_only")
    check("TITLE ONLY" in p_none and "title alone" in p_none,
          "the title-only prompt says there is nothing else")

    # A row whose text is empty must not be labelled as though it had one.
    check(sc.build_user_text("A title", "", "fulltext_head") == p_none,
          "an empty head falls back to the title-only prompt rather than "
          "announcing a head that is not there")


def _write_screen_candidates(path: str, eligible_ids, ineligible=None) -> None:
    """Minimal candidates.csv for screen.py's eligibility join."""
    ineligible = ineligible or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["openalex_id", "eligible",
                                          "exclusion_reason"])
        w.writeheader()
        for pid in eligible_ids:
            w.writerow({"openalex_id": pid, "eligible": "True",
                        "exclusion_reason": "none"})
        for pid, reason in ineligible.items():
            w.writerow({"openalex_id": pid, "eligible": "False",
                        "exclusion_reason": reason})


def test_title_only_forced_uncertain(tmpdir: str):
    print("\n[force] ELIGIBLE title-only papers are forced to uncertain with no "
          "model call")
    stub = _stub_anthropic(tmpdir, token_count=1500)
    crit = os.path.join(tmpdir, "criteria_force.md")
    with open(crit, "w", encoding="utf-8") as f:
        f.write(CRITERIA_A)

    inp = os.path.join(tmpdir, "force_input.csv")
    with open(inp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "abstract",
                                          "fulltext_head", "abstract_source"])
        w.writeheader()
        for i in range(4):
            w.writerow({"paper_id": f"W{i}", "title": f"Paper {i}",
                        "abstract": "", "fulltext_head": "",
                        "abstract_source": "none"})
    # W3 is title-only too, but ineligible: out of scope, so NOT forced.
    cand = os.path.join(tmpdir, "force_candidates.csv")
    _write_screen_candidates(cand, ["W0", "W1", "W2"], {"W3": "no_oa"})

    runs = os.path.join(tmpdir, "runs_force")
    env = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-stub",
               PYTHONPATH=stub + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, "screen.py", "--criteria", crit, "--input", inp,
         "--candidates", cand,
         "--models", "haiku", "--runs-dir", runs, "--run-id", "forcetest",
         "--max-usd", "5.00", "--yes"],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)))

    calls = os.path.join(runs, "forcetest", "calls.jsonl")
    check(os.path.exists(calls), f"a call log was written\n{proc.stdout[-400:]}")
    recs = [json.loads(l) for l in open(calls, encoding="utf-8") if l.strip()]
    check(len(recs) == 3, f"one row per ELIGIBLE title-only paper (got {len(recs)})")
    check("W3" not in {r["paper_id"] for r in recs},
          "the ineligible title-only paper gets no forced decision -- out of "
          "scope, not uncertain")
    check(all(r["parsed"]["decision"] == "uncertain" for r in recs),
          "every one is uncertain")
    check(all(r["api_mode"] == "forced_no_model_call" for r in recs),
          "and none of them was a model call")
    check(all(r["cost_usd"] == 0.0 and r["input_tokens"] == 0
              and r["output_tokens"] == 0 for r in recs),
          "so none of them cost anything")
    check(all(r["validation_status"] == "forced" for r in recs),
          "they are marked `forced`, not `valid` -- a rule, not a judgement")
    check(all(r["text_kind"] == "title_only" for r in recs),
          "and the log records which artefact they had")
    check("forced to" in proc.stdout and "no model call" in proc.stdout,
          "and the run says so on stdout")


def test_isolation():
    print("\n[isolation] the model cannot be handed the stratum or a human label")
    row = {"paper_id": "W1", "title": "An agent architecture",
           "abstract": "We present a multi-agent system.",
           "stratum": "top_third", "human_decision": "include"}
    rendered = sc.build_user_text(row["title"], row["abstract"])
    sc._isolation_check(row, rendered)   # must not raise
    check(True, "a correctly rendered prompt passes the isolation check")
    check("top_third" not in rendered and "include" not in rendered,
          "rendered prompt contains neither the stratum nor the human label")

    leaky = rendered + "\n\nStratum: top_third"
    try:
        sc._isolation_check(row, leaky)
        check(False, "a leaked stratum value must raise")
    except AssertionError as exc:
        check("stratum" in str(exc), f"a leaked stratum value raises ({exc})")

    leaky2 = rendered + "\n\nPrior label: include"
    try:
        sc._isolation_check({"paper_id": "W1", "human_decision": "include"}, leaky2)
        check(False, "a leaked human label must raise")
    except AssertionError as exc:
        check("human_decision" in str(exc), "a leaked human label raises")


def test_ineligible_never_reach_api(tmpdir: str):
    print("\n[eligibility] ineligible rows never reach the API")
    stub = _stub_anthropic(tmpdir, token_count=1500)
    crit = os.path.join(tmpdir, "criteria_elig.md")
    with open(crit, "w", encoding="utf-8") as f:
        f.write(CRITERIA_A)

    # Five eligible, five ineligible (both reasons), every one with a real
    # abstract -- so if the gate failed, they WOULD be sent. Each carries a
    # unique marker string to search the recorded requests for.
    inp = os.path.join(tmpdir, "elig_input.csv")
    inel = {"W5": "elsevier", "W6": "elsevier", "W7": "no_oa", "W8": "no_oa",
            "W9": "elsevier"}
    with open(inp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "abstract",
                                          "abstract_source"])
        w.writeheader()
        for i in range(10):
            tag = "OUTOFSCOPE" if f"W{i}" in inel else "INSCOPE"
            w.writerow({"paper_id": f"W{i}",
                        "title": f"{tag}-MARKER-{i} agent architecture",
                        "abstract": f"{tag}-ABSTRACT-{i} multi-agent system.",
                        "abstract_source": "openalex"})
    cand = os.path.join(tmpdir, "elig_candidates.csv")
    _write_screen_candidates(cand, [f"W{i}" for i in range(5)], inel)

    runs = os.path.join(tmpdir, "runs_elig")
    for mode in ("batch", "sync"):
        record = os.path.join(tmpdir, f"stub_record_{mode}.jsonl")
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-stub",
                   STUB_RECORD=record,
                   PYTHONPATH=stub + os.pathsep + os.environ.get("PYTHONPATH", ""))
        run_id = f"elig_{mode}"
        proc = subprocess.run(
            [sys.executable, "screen.py", "--criteria", crit, "--input", inp,
             "--candidates", cand, "--models", "haiku", "--runs-dir", runs,
             "--run-id", run_id, "--max-usd", "5.00", "--yes"]
            + (["--sync"] if mode == "sync" else []),
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)))

        sent = open(record, encoding="utf-8").read() if os.path.exists(record) else ""
        surfaces = {json.loads(l)["surface"] for l in sent.splitlines() if l}
        expect = {"count_tokens", "batches.create" if mode == "batch"
                  else "messages.create"}
        check(expect <= surfaces,
              f"[{mode}] the stub saw the request surfaces it should "
              f"({sorted(surfaces)}) -- so an absence below means something\n"
              f"{proc.stdout[-300:]}{proc.stderr[-300:]}")
        check(all(f"INSCOPE-MARKER-{i}" in sent for i in range(5)),
              f"[{mode}] every eligible paper was sent")
        check("OUTOFSCOPE" not in sent,
              f"[{mode}] no ineligible title or abstract reached ANY API "
              f"surface (token counting included)")

        calls = os.path.join(runs, run_id, "calls.jsonl")
        logged = ({json.loads(l)["paper_id"]
                   for l in open(calls, encoding="utf-8") if l.strip()}
                  if os.path.exists(calls) else set())
        check(not (logged & set(inel)),
              f"[{mode}] no ineligible paper has a calls.jsonl row -- no model "
              f"call and no forced decision")
        man = json.load(open(os.path.join(runs, run_id, "manifest.json"),
                             encoding="utf-8"))
        ex = man["excluded_at_eligibility"]
        check(ex["n"] == 5 and ex["paper_ids"] == inel,
              f"[{mode}] the manifest records all 5, each with its reason")
        check(ex["by_reason"] == {"elsevier": 3, "no_oa": 2},
              f"[{mode}] counted by reason (got {ex['by_reason']})")
        check(man["papers"] == 5, f"[{mode}] only the 5 eligible are papers to "
                                  f"screen (got {man['papers']})")

    # An input paper_id the candidates file does not know is an unknown scope
    # decision: abort before anything is priced.
    _write_screen_candidates(cand, [f"W{i}" for i in range(4)], inel)
    record = os.path.join(tmpdir, "stub_record_unknown.jsonl")
    env = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-stub", STUB_RECORD=record,
               PYTHONPATH=stub + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, "screen.py", "--criteria", crit, "--input", inp,
         "--candidates", cand, "--models", "haiku", "--runs-dir", runs,
         "--run-id", "elig_unknown", "--max-usd", "5.00", "--yes"],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode != 0 and "eligibility is unknown" in proc.stderr,
          f"a paper_id missing from candidates.csv aborts the run "
          f"(rc={proc.returncode})")
    check(not os.path.exists(record), "and nothing at all was sent")


# ---------------------------------------------------------------------------
# Tests: parser and validator
# ---------------------------------------------------------------------------

def test_criterion_id_extraction():
    print("\n[criteria] criterion IDs are parsed out of criteria.md")
    ids = sc.extract_criterion_ids(CRITERIA_A)
    check(set(ids) == {"I1", "I2", "E1", "E2", "U1"},
          f"all five IDs found from '## ID Title' headings (got {ids})")

    bullets = "- **I1** include if ...\n- **E1** exclude if ...\n"
    check(set(sc.extract_criterion_ids(bullets)) == {"I1", "E1"},
          "bolded bullet form parses too")

    table = "| ID | Rule |\n|---|---|\n| I1 | include |\n| E1 | exclude |\n"
    check(set(sc.extract_criterion_ids(table)) == {"I1", "E1"},
          "markdown table form parses too")

    # Bare definition-list form, which is what the real criteria.md uses. All
    # four of the original patterns missed it, so membership checking silently
    # switched itself off on the only file that matters.
    bare = ("I1 \u2014 Learning or exploratory agents. Agents that learn.\n"
            "D1 \u2014 DT framework (supporting set).\n"
            "E7 - DT without a framework contribution.\n"
            "U3 \u2013 Insufficient text.\n")
    check(set(sc.extract_criterion_ids(bare)) == {"I1", "D1", "E7", "U3"},
          f"bare `ID — text` lines parse, with em dash, en dash or hyphen "
          f"(got {sc.extract_criterion_ids(bare)})")

    # Prose must not be scanned, or every capital-letter-plus-digit in the
    # criteria text would become a criterion ID.
    prose = "This is a paragraph mentioning I1 and E1 in passing.\n"
    check(sc.extract_criterion_ids(prose) == [],
          "IDs mentioned only in prose are not harvested")

    # A decision procedure that names IDs mid-sentence must not yield them
    # either -- otherwise the enforced set would silently include duplicates
    # of whatever the prose happens to reference.
    proc = ("Is it a research paper? If not \u2192 exclude, E1.\n"
            "If unclear \u2192 uncertain, U2.\n")
    check(sc.extract_criterion_ids(proc) == [],
          "IDs referenced mid-sentence in the decision procedure are not "
          "harvested as definitions")

    # Pinned against the real file: if criteria.md is reformatted such that the
    # parser stops seeing its IDs, this fails rather than quietly falling back
    # to not enforcing membership.
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "criteria.md")
    if os.path.exists(real):
        got = sc.extract_criterion_ids(open(real, encoding="utf-8").read())
        expected = ["I1", "I2", "I3", "I4", "D1",
                    "E1", "E2", "E3", "E4", "E5", "E6", "E7",
                    "U1", "U2", "U3"]
        check(sorted(got) == sorted(expected),
              f"the real criteria.md yields exactly its 15 IDs (got {sorted(got)})")
        check(len(got) >= 2,
              "so membership is ENFORCED on the real file, not silently skipped")
    else:
        print("  note  criteria.md not present; the pinned check is skipped")


def test_system_prompt_defers_to_criteria():
    print("\n[prompt] the system prompt adds no decision rules of its own")
    sys_text = sc.build_system_text(CRITERIA_A, "criteria.md")

    check("nothing in these instructions overrides them" in sys_text,
          "it says the criteria are the only authority on the decision")

    # The old hardcoded tie-break must be gone: criteria.md carries its own,
    # and two sets in one prompt make the uncertain rate a property of the
    # prompt rather than of the criteria.
    for gone in ("false exclude cannot be recovered",
                 "torn between exclude and uncertain",
                 "caught at the next stage"):
        check(gone not in sys_text,
              f"the prompt no longer carries its own tie-break ({gone!r})")

    # It must not claim the input is always an abstract -- it may be a head.
    check("title and abstract alone" not in sys_text,
          "it does not claim the input is always an abstract")
    check("The user message says which" in sys_text,
          "it tells the model the input kind is labelled in the user message")

    check(CRITERIA_A.strip() in sys_text,
          "and the criteria text is still embedded verbatim")


def test_parser_and_validator():
    print("\n[parse] malformed tool output is rejected, never retried")
    ids = set(sc.extract_criterion_ids(CRITERIA_A))

    # Good.
    parsed, failure = sc.parse_response(tool_msg())
    status, errors = sc.validate(parsed, failure, ids)
    check(status == "valid" and not errors, f"a well-formed tool call validates ({errors})")

    # No tool_use block at all -- the Opus-with-thinking-disabled failure mode,
    # where the call is written as visible text instead.
    text_only = FakeMessage([FakeBlock("text", text='{"decision": "include"}')],
                            stop_reason="end_turn")
    parsed, failure = sc.parse_response(text_only)
    status, errors = sc.validate(parsed, failure, ids)
    check(status == "invalid" and errors == ["no_tool_use"],
          f"a tool call emitted as plain text is rejected (got {errors})")

    # Wrong tool name.
    wrong = FakeMessage([FakeBlock("tool_use", "some_other_tool", {"decision": "include"})])
    parsed, failure = sc.parse_response(wrong)
    check(sc.validate(parsed, failure, ids)[0] == "invalid",
          "a call to a different tool is rejected")

    # Two tool calls where one was forced.
    two = FakeMessage([FakeBlock("tool_use", sc.TOOL_NAME, {"decision": "include",
                                                            "criterion_id": "I1",
                                                            "reason": "ok"}),
                       FakeBlock("tool_use", sc.TOOL_NAME, {"decision": "exclude",
                                                            "criterion_id": "E1",
                                                            "reason": "no"})])
    parsed, failure = sc.parse_response(two)
    status, errors = sc.validate(parsed, failure, ids)
    check(status == "invalid" and errors[0].startswith("multiple_tool_use"),
          f"two tool calls are rejected rather than silently taking the first "
          f"(got {errors})")

    # Decision outside the enum.
    status, errors = sc.validate(
        {"decision": "maybe", "criterion_id": "I1", "reason": "hm"}, None, ids)
    check(status == "invalid" and any(e.startswith("bad_decision") for e in errors),
          f"a decision outside include/exclude/uncertain is rejected (got {errors})")

    # Hallucinated criterion ID.
    status, errors = sc.validate(
        {"decision": "include", "criterion_id": "I9", "reason": "ok"}, None, ids)
    check(status == "invalid" and any(e.startswith("unknown_criterion_id") for e in errors),
          f"a criterion_id not in criteria.md is rejected (got {errors})")

    # ...but only when the ID set is trusted. valid_ids=None means the parser
    # found too few IDs to enforce, and the model must not be blamed for that.
    status, errors = sc.validate(
        {"decision": "include", "criterion_id": "I9", "reason": "ok"}, None, None)
    check(status == "valid",
          "with an untrusted ID set, membership is not enforced")

    # Reason over the word cap.
    long_reason = " ".join(["word"] * (sc.MAX_REASON_WORDS + 1))
    status, errors = sc.validate(
        {"decision": "include", "criterion_id": "I1", "reason": long_reason}, None, ids)
    check(status == "invalid" and any(e.startswith("reason_too_long") for e in errors),
          f"a reason over {sc.MAX_REASON_WORDS} words is rejected (got {errors})")
    status, errors = sc.validate(
        {"decision": "include", "criterion_id": "I1",
         "reason": " ".join(["word"] * sc.MAX_REASON_WORDS)}, None, ids)
    check(status == "valid", f"exactly {sc.MAX_REASON_WORDS} words is fine")

    # Missing fields.
    status, errors = sc.validate({"decision": "include"}, None, ids)
    check(status == "invalid" and "empty_criterion_id" in errors
          and "empty_reason" in errors,
          f"missing fields are each reported (got {errors})")

    # Extra fields -- a model inventing its own schema keys.
    status, errors = sc.validate(
        {"decision": "include", "criterion_id": "I1", "reason": "ok",
         "confidence": 0.9}, None, ids)
    check(status == "invalid" and any(e.startswith("unexpected_keys") for e in errors),
          f"unexpected keys are reported (got {errors})")

    # Refusal and truncation are distinguishable from a malformed call.
    refusal = FakeMessage([], stop_reason="refusal",
                          stop_details=types.SimpleNamespace(category="cyber"))
    parsed, failure = sc.parse_response(refusal)
    check(failure == "refusal:cyber", f"a refusal is recorded as such (got {failure})")

    trunc = FakeMessage([], stop_reason="max_tokens")
    parsed, failure = sc.parse_response(trunc)
    check(failure == "max_tokens",
          f"a max_tokens truncation is recorded as such (got {failure})")


# ---------------------------------------------------------------------------
# Tests: budget guard
# ---------------------------------------------------------------------------

def test_budget_guard(tmpdir: str):
    print("\n[budget] the cost guard aborts above --max-usd and submits nothing")
    papers = [{"paper_id": f"W{i}", "title": f"Paper {i}",
               "abstract": "An abstract about agents in manufacturing."}
              for i in range(50)]
    sys_blocks = [{"type": "text", "text": sc.build_system_text(CRITERIA_A, "criteria.md"),
                   "cache_control": {"type": "ephemeral"}}]
    tools = [sc.build_tool(False)]

    client = FakeClient(token_count=1500)
    est = sc.estimate_cost(client, papers, ["claude-opus-5-5"], sys_blocks, tools,
                           repeat=1, batch=True, assumed_output_tokens=80,
                           exact_max=200, verbose=False)

    # Hand-computed. 50 calls x 1500 input tokens = 75,000 input tokens.
    # Opus 5.5 input $4/MTok -> $0.300. Opus 5.5 cannot disable thinking, so
    # the assumed output is scaled by THINKING_OUTPUT_FACTOR: 80 x 8 = 640 tok
    # per call, 50 x 640 = 32,000 tok at $20/MTok -> $0.640.
    # Batch halves both: (0.300 + 0.640) / 2 = $0.470.
    got = est["per_model"]["claude-opus-5-5"]["estimated_usd"]
    check(abs(got - 0.470) < 1e-6,
          f"batch cost arithmetic is exact at Opus 5.5 prices: expected "
          f"$0.4700, got ${got}")
    check(est["per_model"]["claude-opus-5-5"]["thinking_unavoidable"] is True,
          "and the estimate records that its thinking could not be disabled")

    nobatch = sc.estimate_cost(client, papers, ["claude-opus-5-5"], sys_blocks,
                               tools, 1, False, 80, 200, verbose=False)
    check(abs(nobatch["total_estimated_usd"] - 0.940) < 1e-6,
          f"without batching the same run is exactly double: "
          f"${nobatch['total_estimated_usd']}")

    # Sonnet 5.5 can suppress thinking, so its output is NOT scaled:
    # 75,000 in x $2/MTok = $0.150; 50 x 80 = 4,000 out x $10/MTok = $0.040.
    # Batch: (0.150 + 0.040) / 2 = $0.095.
    son = sc.estimate_cost(client, papers, ["claude-sonnet-5-5"], sys_blocks,
                           tools, 1, True, 80, 200, verbose=False)
    check(abs(son["total_estimated_usd"] - 0.095) < 1e-6,
          f"Sonnet 5.5's output estimate is unscaled "
          f"(${son['total_estimated_usd']})")

    three = sc.estimate_cost(
        client, papers, list(sc.MODEL_IDS.values()),
        sys_blocks, tools, 1, True, 80, 200, verbose=False)
    check(three["total_estimated_usd"] < 2.0,
          f"a 50-paper calibration run across all three models is under $2 "
          f"(${three['total_estimated_usd']})")

    check(three["total_estimated_usd"] < 10.0,
          f"and therefore under the $10 default cap "
          f"(${three['total_estimated_usd']} < $10.00)")

    # The guard itself, end to end. screen.py is run in a subprocess against a
    # STUB anthropic module (see _stub_anthropic) so the test is fully offline
    # and deterministic -- otherwise a machine without an API key would "pass"
    # by failing at the network before the guard was ever reached, which proves
    # nothing about the guard.
    stub = _stub_anthropic(tmpdir, token_count=1500)
    crit = os.path.join(tmpdir, "criteria_guard.md")
    with open(crit, "w", encoding="utf-8") as f:
        f.write(CRITERIA_A)
    inp = os.path.join(tmpdir, "guard_input.csv")
    with open(inp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "abstract"])
        w.writeheader()
        for p in papers:
            w.writerow(p)
    runs = os.path.join(tmpdir, "runs_guard")
    cand = os.path.join(tmpdir, "guard_candidates.csv")
    _write_screen_candidates(cand, [p["paper_id"] for p in papers])

    env = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-stub",
               PYTHONPATH=stub + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def run_guard(cap, run_id):
        return subprocess.run(
            [sys.executable, "screen.py", "--criteria", crit, "--input", inp,
             "--candidates", cand,
             "--models", "opus", "--runs-dir", runs, "--run-id", run_id,
             "--max-usd", cap, "--yes"],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)))

    over = run_guard("0.0001", "guard_over")
    check(over.returncode == 2,
          f"screen.py exits 2 under a $0.0001 cap (rc={over.returncode})\n"
          f"{over.stdout[-500:]}{over.stderr[-500:]}")
    check("ABORT" in over.stdout and "exceeds --max-usd" in over.stdout,
          "the abort message names the cap that was exceeded")
    check(not os.path.exists(os.path.join(runs, "guard_over", "calls.jsonl")),
          "no calls.jsonl was written -- nothing was submitted")

    # The guard must not be a blanket refusal: the same run under an adequate
    # cap has to get past the estimate. The stub then fails at batch creation,
    # which is fine -- what is being asserted is that the guard let it through.
    under = run_guard("5.00", "guard_under")
    check("ABORT" not in under.stdout,
          f"the same run under a $5.00 cap is NOT aborted by the guard")
    check(under.returncode != 2,
          f"and does not exit with the guard's code 2 (rc={under.returncode})")
    check(os.path.exists(os.path.join(runs, "guard_under", "manifest.json")),
          "a manifest is written once the estimate clears the cap")
    man = json.load(open(os.path.join(runs, "guard_under", "manifest.json"),
                         encoding="utf-8"))
    check(abs(man["cost_estimate"]["total_estimated_usd"] - 0.470) < 1e-6,
          f"the manifest records the pre-submission estimate "
          f"(${man['cost_estimate']['total_estimated_usd']})")
    check(man["criteria_sha256"] == sc.sha256(CRITERIA_A),
          "and the criteria hash, matching the file on disk")
    check(man["models"]["opus"] == "claude-opus-5-5",
          f"the manifest records the requested ID unchanged "
          f"(got {man['models']['opus']})")
    check(man["tool_choice"] == {"type": "auto"},
          "and that tool_choice was auto, not forced")

    # --- never substitute: an unserved ID must abort ----------------------
    # The stub serves only Haiku, so the requested Opus ID is not available.
    stub_partial = _stub_anthropic(tmpdir, token_count=1500,
                                   served=["claude-haiku-4-5-20251001"])
    env2 = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-stub",
                PYTHONPATH=stub_partial + os.pathsep
                + os.environ.get("PYTHONPATH", ""))
    bad = subprocess.run(
        [sys.executable, "screen.py", "--criteria", crit, "--input", inp,
         "--candidates", cand,
         "--models", "opus", "--runs-dir", runs, "--run-id", "guard_badmodel",
         "--max-usd", "5.00", "--yes"],
        capture_output=True, text=True, env=env2,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(bad.returncode != 0,
          f"an unserved model ID aborts the run (rc={bad.returncode})")
    check("not served by this account" in bad.stdout + bad.stderr,
          "and says the ID is not served")
    check("claude-opus-5-5" in bad.stdout + bad.stderr,
          "naming the ID that was requested")
    check("does not\n  substitute" in bad.stdout + bad.stderr
          or "substitute" in bad.stdout + bad.stderr,
          "and states that it will not substitute one")
    check(not os.path.exists(os.path.join(runs, "guard_badmodel", "calls.jsonl")),
          "nothing was submitted")


# ---------------------------------------------------------------------------
# Tests: metrics
# ---------------------------------------------------------------------------

def test_kappa_and_recall():
    print("\n[metrics] kappa, recall and precision on a hand-computed example")

    # --- 3-class kappa, worked by hand -----------------------------------
    # 10 papers. Confusion (rows human, cols model):
    #                 inc  exc  unc
    #   include        3    1    0     (4)
    #   exclude        1    3    0     (4)
    #   uncertain      1    0    1     (2)
    #   model totals   5    4    1
    # po = (3 + 3 + 1) / 10 = 0.70
    # pe = (4/10)(5/10) + (4/10)(4/10) + (2/10)(1/10)
    #    = 0.20 + 0.16 + 0.02 = 0.38
    # k  = (0.70 - 0.38) / (1 - 0.38) = 0.32 / 0.62 = 0.516129...
    human = pd.Series(["include"] * 4 + ["exclude"] * 4 + ["uncertain"] * 2)
    model = pd.Series(["include", "include", "include", "exclude",
                       "include", "exclude", "exclude", "exclude",
                       "include", "uncertain"])
    k3 = bm.cohen_kappa(human, model, bm.DECISIONS)
    check(abs(k3 - 0.32 / 0.62) < 1e-9,
          f"3-class kappa = 0.32/0.62 = 0.516129 (got {k3:.6f})")

    # --- binary kappa, uncertain folded into include ---------------------
    # human:  include x4, exclude x4, include x2  -> inc 6, exc 4
    # model:  inc inc inc exc | inc exc exc exc | inc inc -> inc 6, exc 4
    # agree:  1,1,1,0 | 0,1,1,1 | 1,1 -> 8 of 10
    # po = 0.80
    # pe = (6/10)(6/10) + (4/10)(4/10) = 0.36 + 0.16 = 0.52
    # k  = (0.80 - 0.52) / 0.48 = 0.28 / 0.48 = 0.583333...
    hb, mb = bm.to_binary(human), bm.to_binary(model)
    kb = bm.cohen_kappa(hb, mb, bm.BINARY)
    check(abs(kb - 0.28 / 0.48) < 1e-9,
          f"binary kappa = 0.28/0.48 = 0.583333 (got {kb:.6f})")
    check(kb > k3,
          "collapsing uncertain into include raises kappa here, as it must -- "
          "the uncertain/include split stops being a disagreement")

    # --- recall and precision --------------------------------------------
    # positives = binary include. human inc = 6; model called 5 of them inc
    #   (rows 1,2,3 include; row 9 uncertain->inc; row 10 uncertain->inc) = 5
    #   -> recall 5/6
    # model inc total = 6, of which 5 were truly inc, 1 was human-exclude
    #   -> precision 5/6
    recall, precision, tp, fn, fp = bm.recall_precision(hb, mb)
    check(tp == 5 and fn == 1 and fp == 1,
          f"tp=5 fn=1 fp=1 (got tp={tp} fn={fn} fp={fp})")
    check(abs(recall - 5 / 6) < 1e-9, f"include recall = 5/6 (got {recall:.6f})")
    check(abs(precision - 5 / 6) < 1e-9, f"include precision = 5/6 (got {precision:.6f})")

    # --- perfect and degenerate cases ------------------------------------
    perfect = pd.Series(["include", "exclude", "uncertain", "include"])
    check(abs(bm.cohen_kappa(perfect, perfect.copy(), bm.DECISIONS) - 1.0) < 1e-12,
          "identical raters give kappa 1.0")

    allsame = pd.Series(["include"] * 5)
    k = bm.cohen_kappa(allsame, allsame.copy(), bm.DECISIONS)
    check(k != k,
          "two raters who used one class for everything give kappa NaN, not 1.0 "
          "-- chance agreement is 1.0 and kappa is undefined")

    # A model that excludes everything: recall 0, and kappa must not reward it.
    all_exclude = pd.Series(["exclude"] * 10)
    r, p, *_ = bm.recall_precision(hb, all_exclude)
    check(r == 0.0, f"a model that excludes everything has include recall 0 (got {r})")
    check(p != p, "and undefined precision -- it never predicted include")

    # The floors themselves.
    check(bm.KAPPA_FLOOR == 0.70 and bm.RECALL_FLOOR == 0.95,
          "selection floors are kappa >= 0.70 and include recall >= 0.95")
    check(not (kb >= bm.KAPPA_FLOOR),
          f"the worked example's kappa {kb:.3f} does NOT clear the 0.70 floor")


def _write_labels(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "abstract",
                                          "human_decision", "human_criterion_id",
                                          "human_notes"])
        w.writeheader()
        w.writerows(rows)


def test_benchmark_refuses_partial_labels(tmpdir: str):
    print("\n[refuse] benchmark.py will not score a partly-filled sheet")
    run_dir = os.path.join(tmpdir, "runs", "refusetest")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "calls.jsonl"), "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "run_id": "refusetest", "model_alias": "opus",
                "model": "claude-opus-5-5", "paper_id": f"W{i}", "repeat": 1,
                "parsed": {"decision": "include", "criterion_id": "I1",
                           "reason": "ok"},
                "validation_status": "valid",
                "input_tokens": 1000, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "output_tokens": 40,
                "cost_usd": 0.001,
            }) + "\n")

    labels_path = os.path.join(tmpdir, "labels_partial.csv")
    rows = [{"paper_id": f"W{i}", "title": f"T{i}", "abstract": "a",
             "human_decision": "include" if i < 2 else "",
             "human_criterion_id": "I1" if i < 2 else "", "human_notes": ""}
            for i in range(4)]
    _write_labels(labels_path, rows)

    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 3,
          f"exits 3 on an unfilled sheet (got {proc.returncode})")
    check("REFUSING TO COMPUTE METRICS" in proc.stdout,
          "and says so plainly")
    check("2 of 4 rows have no human_decision" in proc.stdout,
          f"and counts the blanks")
    check(not os.path.exists(os.path.join(run_dir, "benchmark.csv")),
          "no benchmark.csv was written")

    # An out-of-vocabulary decision is also refused, not silently dropped.
    rows[2]["human_decision"] = "include"
    rows[3]["human_decision"] = "maybe"
    _write_labels(labels_path, rows)
    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 3 and "outside" in proc.stdout,
          "a decision outside include/exclude/uncertain is also refused")

    # Fully and validly filled: it scores.
    rows[3]["human_decision"] = "exclude"
    rows[3]["human_criterion_id"] = "E1"
    _write_labels(labels_path, rows)
    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 0, f"a fully filled sheet scores (rc={proc.returncode})")
    check(os.path.exists(os.path.join(run_dir, "benchmark.csv")),
          "benchmark.csv is written")
    check(os.path.exists(os.path.join(run_dir, "disagreements.csv")),
          "disagreements.csv is written")

    b = pd.read_csv(os.path.join(run_dir, "benchmark.csv"))
    check(len(b) == 1 and b.iloc[0]["model"] == "claude-opus-5-5",
          "one row per model")
    # The model said include x4; the human said include x3, exclude x1.
    # Binary: human inc 3 exc 1; model inc 4. tp=3 fn=0 fp=1.
    check(b.iloc[0]["include_recall_binary"] == 1.0,
          f"recall 1.0 (got {b.iloc[0]['include_recall_binary']})")
    check(abs(b.iloc[0]["include_precision_binary"] - 0.75) < 1e-9,
          f"precision 3/4 (got {b.iloc[0]['include_precision_binary']})")
    check(abs(b.iloc[0]["cost_usd"] - 0.004) < 1e-9,
          f"cost summed from the log, not estimated (got {b.iloc[0]['cost_usd']})")
    check(b.iloc[0]["input_tokens"] == 4000 and b.iloc[0]["output_tokens"] == 160,
          "token totals summed from the log")

    d = pd.read_csv(os.path.join(run_dir, "disagreements.csv"))
    check(len(d) == 1 and d.iloc[0]["paper_id"] == "W3",
          f"the single disagreement is recorded (got {len(d)} rows)")
    check(d.iloc[0]["direction"] == "false_include",
          f"and its direction is labelled (got {d.iloc[0]['direction']})")

    # Kappa here is 0 (the model used one class), so nothing qualifies and the
    # selection rule must decline rather than pick the only candidate.
    check("NO MODEL QUALIFIES" in proc.stdout,
          "with no model over both floors, none is selected")


def test_selection_rule(tmpdir: str):
    print("\n[select] the cheapest qualifying model wins; otherwise none does")
    run_dir = os.path.join(tmpdir, "runs", "selecttest")
    os.makedirs(run_dir, exist_ok=True)

    # 20 papers. Human: 10 include, 10 exclude.
    human = ["include"] * 10 + ["exclude"] * 10
    # haiku: cheap, misses two includes -> recall 0.8, fails the recall floor.
    haiku = ["include"] * 8 + ["exclude"] * 2 + ["exclude"] * 10
    # sonnet: catches every include, one false include -> recall 1.0,
    #         binary kappa = (0.95 - 0.505)/0.495 = 0.8990 -> qualifies.
    sonnet = ["include"] * 10 + ["include"] + ["exclude"] * 9
    # opus: perfect, but more expensive -> must NOT be chosen over sonnet.
    opus = list(human)

    per_call_cost = {"claude-haiku-4-5-20251001": 0.0001,
                     "claude-sonnet-5-5": 0.0005,
                     "claude-opus-5-5": 0.0020}
    with open(os.path.join(run_dir, "calls.jsonl"), "w", encoding="utf-8") as f:
        for mid, alias, decisions in (
                ("claude-haiku-4-5-20251001", "haiku", haiku),
                ("claude-sonnet-5-5", "sonnet", sonnet),
                ("claude-opus-5-5", "opus", opus)):
            for i, d in enumerate(decisions):
                f.write(json.dumps({
                    "run_id": "selecttest", "model_alias": alias, "model": mid,
                    "paper_id": f"W{i}", "repeat": 1,
                    "parsed": {"decision": d, "criterion_id":
                               "I1" if d == "include" else "E1", "reason": "r"},
                    "validation_status": "valid",
                    "input_tokens": 1000, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0, "output_tokens": 40,
                    "cost_usd": per_call_cost[mid],
                }) + "\n")

    labels_path = os.path.join(tmpdir, "labels_select.csv")
    _write_labels(labels_path, [
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": "a",
         "human_decision": h,
         "human_criterion_id": "I1" if h == "include" else "E1",
         "human_notes": ""}
        for i, h in enumerate(human)])

    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 0, f"scores cleanly (rc={proc.returncode})")

    summary = json.load(open(os.path.join(run_dir, "benchmark_summary.json"),
                             encoding="utf-8"))
    sel = summary["selection"]
    check(sel["selected"] is True, "a model qualifies")
    check(sel["model_alias"] == "sonnet",
          f"the CHEAPEST qualifying model is chosen, not the best "
          f"(got {sel.get('model_alias')})")

    b = pd.read_csv(os.path.join(run_dir, "benchmark.csv"))
    by = {r["model_alias"]: r for _, r in b.iterrows()}
    check(abs(by["haiku"]["include_recall_binary"] - 0.8) < 1e-9,
          f"haiku recall 0.8 (got {by['haiku']['include_recall_binary']})")
    check(not by["haiku"]["meets_recall_floor"],
          "haiku fails the 0.95 recall floor despite being cheapest")
    check(by["opus"]["include_recall_binary"] == 1.0
          and by["opus"]["kappa_binary_primary"] == 1.0,
          "opus is perfect but costlier, so it is passed over")
    check(list(b["cost_usd"]) == sorted(b["cost_usd"]),
          "benchmark.csv is ordered cheapest first")

    # And when nothing clears the floors, nothing is picked.
    run2 = os.path.join(tmpdir, "runs", "noqualify")
    os.makedirs(run2, exist_ok=True)
    with open(os.path.join(run2, "calls.jsonl"), "w", encoding="utf-8") as f:
        for i, d in enumerate(haiku):
            f.write(json.dumps({
                "run_id": "noqualify", "model_alias": "haiku",
                "model": "claude-haiku-4-5-20251001", "paper_id": f"W{i}", "repeat": 1,
                "parsed": {"decision": d, "criterion_id": "I1", "reason": "r"},
                "validation_status": "valid", "input_tokens": 1000,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                "output_tokens": 40, "cost_usd": 0.0001,
            }) + "\n")
    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run2, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    s2 = json.load(open(os.path.join(run2, "benchmark_summary.json"),
                        encoding="utf-8"))["selection"]
    check(s2["selected"] is False and s2["model"] is None,
          "with only a sub-floor model available, none is selected")
    check("revise" in s2["next_step"],
          "and the recorded next step is to revise the criteria, not to relax "
          "the floor")


def test_thinking_asymmetry_is_surfaced(tmpdir: str):
    print("\n[thinking] benchmark.py flags the model that could not stop thinking")
    run_dir = os.path.join(tmpdir, "runs", "thinktest")
    os.makedirs(run_dir, exist_ok=True)

    human = ["include"] * 5 + ["exclude"] * 5
    with open(os.path.join(run_dir, "calls.jsonl"), "w", encoding="utf-8") as f:
        for mid, alias in (("claude-sonnet-5-5", "sonnet"),
                           ("claude-opus-5-5", "opus")):
            for i, d in enumerate(human):
                f.write(json.dumps({
                    "run_id": "thinktest", "model_alias": alias, "model": mid,
                    "paper_id": f"W{i}", "repeat": 1,
                    "parsed": {"decision": d, "criterion_id": "I1", "reason": "r"},
                    "validation_status": "valid", "input_tokens": 1000,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                    "output_tokens": 40, "cost_usd": 0.001,
                }) + "\n")

    # The manifest is what carries the request-shape fact into scoring.
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"per_model_request_shape": {
            m: {**sc.MODEL_REQUEST[m], "sent": sc.model_kwargs(m)}
            for m in ("claude-sonnet-5-5", "claude-opus-5-5")}}, f)

    labels_path = os.path.join(tmpdir, "labels_think.csv")
    _write_labels(labels_path, [
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": "a",
         "human_decision": h, "human_criterion_id": "I1", "human_notes": ""}
        for i, h in enumerate(human)])

    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 0, f"scores cleanly (rc={proc.returncode})")

    b = pd.read_csv(os.path.join(run_dir, "benchmark.csv"))
    by = {r["model_alias"]: r for _, r in b.iterrows()}
    check(bool(by["opus"]["thinking_unavoidable"]) is True,
          "opus is marked thinking_unavoidable in benchmark.csv")
    check(bool(by["sonnet"]["thinking_unavoidable"]) is False,
          "sonnet is not")
    check("thinking could not be disabled" in proc.stdout,
          "and the report warns that the cost comparison is not like-for-like")

    summary = json.load(open(os.path.join(run_dir, "benchmark_summary.json"),
                             encoding="utf-8"))
    check(summary["models_thinking_unavoidable"] == ["claude-opus-5-5"],
          f"the summary names it (got {summary['models_thinking_unavoidable']})")


def test_benchmark_exclusion_stages(tmpdir: str):
    print("\n[stages] eligibility and screening exclusions are counted apart")
    run_dir = os.path.join(tmpdir, "runs", "stagetest")
    os.makedirs(run_dir, exist_ok=True)
    # W0-W5 eligible and screened; W6, W7 excluded at eligibility.
    decisions = ["include", "include", "exclude", "exclude", "exclude", "include"]
    with open(os.path.join(run_dir, "calls.jsonl"), "w", encoding="utf-8") as f:
        for i, d in enumerate(decisions):
            f.write(json.dumps({
                "run_id": "stagetest", "model_alias": "haiku",
                "model": "claude-haiku-4-5-20251001", "paper_id": f"W{i}",
                "repeat": 1,
                "parsed": {"decision": d, "criterion_id": "I1", "reason": "r"},
                "validation_status": "valid", "input_tokens": 1000,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                "output_tokens": 40, "cost_usd": 0.0001,
            }) + "\n")
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"excluded_at_eligibility": {
            "n": 2, "by_reason": {"elsevier": 1, "no_oa": 1},
            "paper_ids": {"W6": "elsevier", "W7": "no_oa"}}}, f)

    # The sheet labels all 8, including the two out-of-scope papers.
    labels_path = os.path.join(tmpdir, "labels_stages.csv")
    _write_labels(labels_path, [
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": "a",
         "human_decision": h, "human_criterion_id": "I1", "human_notes": ""}
        for i, h in enumerate(decisions + ["include", "include"])])

    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode == 0, f"scores cleanly (rc={proc.returncode})\n"
                                f"{proc.stdout[-400:]}{proc.stderr[-400:]}")
    b = pd.read_csv(os.path.join(run_dir, "benchmark.csv")).iloc[0]
    check(b["n_labels"] == 6 and b["n_scored"] == 6 and b["coverage"] == 1.0,
          f"the 2 ineligible labels leave the denominator: 6 scored of 6, "
          f"coverage 1.0 (got {b['n_scored']}/{b['n_labels']}, {b['coverage']})")
    check(b["n_excluded_at_screening"] == 3,
          f"excluded at screening = the model's 3 excludes "
          f"(got {b['n_excluded_at_screening']})")
    check(b["n_excluded_at_eligibility"] == 2,
          f"excluded at eligibility = 2, in its own column "
          f"(got {b['n_excluded_at_eligibility']})")
    summ = json.load(open(os.path.join(run_dir, "benchmark_summary.json"),
                          encoding="utf-8"))["exclusion_stages"]
    st1 = summ["excluded_at_eligibility"]
    check(st1["n_in_run"] == 2 and st1["by_reason_in_run"] == {"elsevier": 1, "no_oa": 1},
          "the summary carries stage 1 by reason")
    check(st1["n_labelled_dropped_from_scoring"] == 2,
          "and says 2 labelled papers were dropped from scoring for it")
    check(summ["excluded_at_screening_by_model"] == {"claude-haiku-4-5-20251001": 3},
          "stage 2 is reported per model, separately")
    check("excluded at eligibility: 2" in proc.stdout
          and "excluded at screening by haiku: 3" in proc.stdout,
          "and the report prints the two stages on separate lines")

    # A call log holding a paper the manifest says was ineligible is a broken
    # run, not something to score around.
    with open(os.path.join(run_dir, "calls.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": "stagetest", "model_alias": "haiku",
            "model": "claude-haiku-4-5-20251001", "paper_id": "W6", "repeat": 1,
            "parsed": {"decision": "exclude", "criterion_id": "E1", "reason": "r"},
            "validation_status": "valid", "input_tokens": 1,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "output_tokens": 1, "cost_usd": 0.0}) + "\n")
    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    check(proc.returncode != 0 and "excluded at eligibility" in proc.stderr,
          "a call row for an ineligible paper makes the run unscoreable")


def test_invalid_responses_shrink_coverage(tmpdir: str):
    print("\n[coverage] invalid responses shrink n_scored but not cost")
    run_dir = os.path.join(tmpdir, "runs", "covtest")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "calls.jsonl"), "w", encoding="utf-8") as f:
        for i in range(4):
            ok = i < 3
            f.write(json.dumps({
                "run_id": "covtest", "model_alias": "opus",
                "model": "claude-opus-5-5", "paper_id": f"W{i}", "repeat": 1,
                "parsed": ({"decision": "include", "criterion_id": "I1",
                            "reason": "ok"} if ok else None),
                "validation_status": "valid" if ok else "invalid",
                "validation_errors": [] if ok else ["no_tool_use"],
                "input_tokens": 1000, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "output_tokens": 40,
                "cost_usd": 0.001,
            }) + "\n")

    labels_path = os.path.join(tmpdir, "labels_cov.csv")
    _write_labels(labels_path, [
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": "a",
         "human_decision": "include", "human_criterion_id": "I1",
         "human_notes": ""} for i in range(4)])

    proc = subprocess.run(
        [sys.executable, "benchmark.py", "--run", run_dir, "--labels", labels_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    b = pd.read_csv(os.path.join(run_dir, "benchmark.csv"))
    r = b.iloc[0]
    check(r["n_scored"] == 3 and r["n_invalid_responses"] == 1,
          f"3 of 4 scored, 1 invalid (got {r['n_scored']}, {r['n_invalid_responses']})")
    check(abs(r["coverage"] - 0.75) < 1e-9, f"coverage 0.75 (got {r['coverage']})")
    check(abs(r["cost_usd"] - 0.004) < 1e-9,
          f"cost still counts all 4 calls (got {r['cost_usd']}) -- an unusable "
          f"answer was still paid for")
    check("failed validation" in proc.stdout,
          "and the partial coverage is called out in the report")


# ---------------------------------------------------------------------------

def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="screening_test_")
    try:
        test_sample_draw(tmpdir)
        test_labels_sheet_is_blind(tmpdir)
        test_fulltext_sample(tmpdir)
        test_prompt_hashes()
        test_text_kind_labelling()
        test_title_only_forced_uncertain(tmpdir)
        test_ineligible_never_reach_api(tmpdir)
        test_model_table()
        test_request_shape_per_model()
        test_isolation()
        test_criterion_id_extraction()
        test_system_prompt_defers_to_criteria()
        test_parser_and_validator()
        test_budget_guard(tmpdir)
        test_kappa_and_recall()
        test_benchmark_refuses_partial_labels(tmpdir)
        test_selection_rule(tmpdir)
        test_thinking_asymmetry_is_surfaced(tmpdir)
        test_invalid_responses_shrink_coverage(tmpdir)
        test_benchmark_exclusion_stages(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
