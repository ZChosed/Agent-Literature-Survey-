#!/usr/bin/env python3
"""
screen.py -- LLM title/abstract screening harness with full call provenance.

Governing rule (proposal S3.3): LLMs classify; Python counts. Every model call
made here produces one row in calls.jsonl carrying the model string, the hash of
the criteria it was screening against, the hash of the exact system prompt sent,
the raw response, the parsed output, whether that output validated, the measured
token usage and the cost computed from it. benchmark.py then does the counting.
No aggregate in this pipeline is ever produced by a model.

Eligible rows only (v2.0.0)
---------------------------
Since the 2026-10-05 scope change, a paper is screened only if eligibility.py
marked it `eligible=True` in corpus/eligibility.csv (`--eligibility`). It is
joined by paper_id, never trusted from the input file; the file must have been
computed against the current `--candidates` (else STALE, abort), and a
paper_id it does not know aborts the run before anything is priced.

An ineligible paper gets NO model call and NO forced decision -- it is not
written to calls.jsonl at all. It is out of scope, not uncertain: forcing it to
`uncertain` would route it to a human desk for a decision that has already
been made, and would make it look like a screening outcome. It is recorded in
the run manifest under `excluded_at_eligibility`, with its reason, so
benchmark.py can report "excluded at eligibility" and "excluded at screening"
as two different stages.

The title-only forcing below still applies, but only to ELIGIBLE rows with no
abstract and no full-text head.

What the model sees, and what it must not
-----------------------------------------
Title and abstract. Nothing else. Not the stratum, not the rank or seed
linkage, not a human label, not another model's answer. Each call is a fresh
request with no conversation history, so cross-contamination between papers is
structurally impossible rather than merely intended. The stratum and human-label
exclusions are additionally CHECKED at render time (see `_isolation_check`):
calibration_sample.csv carries a stratum column, so "we only read three columns"
is an invariant worth enforcing rather than trusting.

Structured output, and why it is not a forced tool call
-------------------------------------------------------
A tool call returning {decision, criterion_id, reason}, with
`tool_choice: {"type": "auto"}` and `strict: true`.

The brief asked for a FORCED tool call. That is no longer possible on two of
the three models: `tool_choice` `{"type": "any"}` and `{"type": "tool", ...}`
return

    400 tool_choice: type "tool" and "any" are not supported for this model.

on `claude-opus-5-5` and `claude-sonnet-5-5`, and the same check applies to the
token counting endpoint, so even the budget estimate would fail. The documented
replacement is `auto` plus strict tool use, and that is what this harness
sends -- to ALL THREE models, including Haiku 4.5, which would still accept
forcing. Using a different request shape per model would mean the benchmark
compares request shapes as well as models, and the whole point is to compare
models.

The cost of `auto` is that a model may answer in prose instead of calling the
tool. That is not hidden: the system prompt says the tool must be called for
every paper, and a prose answer is logged as a `no_tool_use` validation
failure. On a run where that rate is non-trivial it is a finding about the
model, which is exactly what Report 2 is for.

`strict: true` is therefore ON by default (`--no-strict-tool` drops it). With
forcing gone it is the mechanism that keeps the tool INPUT schema-valid, and
the client-side validator is not made redundant by it: two of the checks --
criterion_id membership and the 25-word reason cap -- are not expressible in
JSON Schema, and `strict` says nothing about whether the tool was called at
all.

Validation failures are logged and NOT retried. A silent retry would turn a
measured malformed-output rate into an unmeasurable one.

Thinking is minimised per model, and on one model that is impossible
---------------------------------------------------------------------
The brief asked for no extended thinking. Achievable on two of three:

    claude-haiku-4-5-20251001  omit `thinking`          -> none
    claude-sonnet-5-5          `between_tools`          -> no up-front thinking
    claude-opus-5-5            NOT POSSIBLE             -> effort=low only

Opus 5.5 runs adaptive thinking always; both `disabled` and `enabled` return
400. Its thinking tokens are billed as output tokens, so it pays for something
the other two do not, and the selection rule picks the cheapest qualifying
model. `benchmark.py` carries `thinking_unavoidable` per model so that cost
comparison is read with the asymmetry visible rather than silently folded in.

criterion_id membership
-----------------------
The valid IDs live in criteria.md, which this script does not author and cannot
assume the shape of. It extracts them with a conservative pattern and then
states what it did rather than pretending to certainty:

  - 2 or more IDs extracted -> membership is ENFORCED; an ID outside the set is
    a validation failure (it is a hallucinated criterion, which is exactly the
    failure mode worth catching).
  - fewer than 2 -> membership is RECORDED but not enforced, and provenance
    says so, because a parser that found nothing is evidence about the parser,
    not about the model.

`--criterion-ids I1,I2,E1` overrides the parse; `--criterion-id-check` forces
the mode. The extracted set is printed at startup so a bad parse is visible
before any money is spent.

Cost
----
Message Batches: 50% off, results are not time-sensitive. `--sync` bypasses
batching for debugging 1-3 papers. Cost is ESTIMATED before submission against
--max-usd and then RECOMPUTED from the usage figures the API actually returned;
benchmark.py reports only the latter.

Usage
-----
    python screen.py --criteria screening/criteria.md \\
                     --input screening/calibration_sample.csv \\
                     --candidates corpus/candidates.csv \\
                     --models haiku,sonnet,opus

    python screen.py --criteria screening/criteria.md \\
                     --input screening/calibration_sample.csv \\
                     --models opus --sync --limit 3

    # self-consistency: same request twice per paper
    python screen.py ... --repeat 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import eligibility as el

try:
    import anthropic
except ImportError:
    sys.exit("anthropic is required:  pip install anthropic")

SCRIPT_VERSION = "2.0.0"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# These are the IDs the task brief specified, used verbatim. THE HARNESS NEVER
# SUBSTITUTES A MODEL ID. An earlier version of this file silently swapped
# `claude-opus-5-5` for `claude-opus-5` on the strength of a stale internal
# table; that was wrong three times over -- it changes which models the
# benchmark is about, it changes the cost basis the selection rule picks on
# (Opus 5.5 is $4/$20 against Opus 5's $5/$25), and a logged substitution is
# still an unrequested change to the method.
#
# Instead, every requested ID is checked against GET /v1/models at startup and
# the run aborts on a mismatch. A model this table does not know is a reason to
# stop, not a reason to guess.
#
# Verified against platform.claude.com/docs/en/about-claude/pricing and
# .../models/overview on 2026-09-28. Haiku's dated form is the pinned snapshot,
# which is what reproducibility wants.
MODEL_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_CHECKED_UTC = "2026-09-28"

# USD per million tokens, base (non-batch) rates, plus the per-model cache-read
# multiplier. That multiplier is NOT a global constant: Opus 5.5 reads cache at
# 0.05x base input ($0.20/MTok) where the others read at 0.10x. Hardcoding 0.10
# for everything overstates Opus 5.5's cache cost, and the selection rule picks
# on cost.
PRICING = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00, "output": 5.00, "cache_read_mult": 0.10},
    "claude-haiku-4-5": {
        "input": 1.00, "output": 5.00, "cache_read_mult": 0.10},
    "claude-sonnet-5-5": {
        "input": 2.00, "output": 10.00, "cache_read_mult": 0.10},
    "claude-opus-5-5": {
        "input": 4.00, "output": 20.00, "cache_read_mult": 0.05},
}

CACHE_WRITE_MULT = 1.25   # 5-minute TTL, all current models
BATCH_MULT = 0.50

# Minimum cacheable prefix, in tokens. Below this the API silently does not
# cache -- no error, cache_creation_input_tokens just comes back 0. Haiku 4.5's
# 4096 is the one that bites: a criteria.md short enough to be readable is very
# likely under it, so the "marked for prompt caching" system block may simply
# not cache on the cheapest model. Both 5.5 models dropped to 512.
CACHE_MIN_PREFIX = {
    "claude-haiku-4-5-20251001": 4096,
    "claude-haiku-4-5": 4096,
    "claude-sonnet-5-5": 512,
    "claude-opus-5-5": 512,
}

# Per-model request shape. The three models do NOT agree on how to minimise
# thinking, and on two of them forced tool use no longer exists at all.
#
#   thinking   -- what to send to get the least thinking the model allows.
#                 None means send no `thinking` field.
#   effort     -- output_config.effort, or None where the model has no effort
#                 parameter (Haiku 4.5).
#   min_think  -- False where the model can genuinely run without thinking.
#                 True where thinking is unavoidable and its tokens are billed
#                 as output. This asymmetry is a real confound in a cost
#                 comparison and is surfaced, not buried.
MODEL_REQUEST = {
    "claude-haiku-4-5-20251001": {
        "thinking": None, "effort": None, "thinking_unavoidable": False,
        "basis": "Haiku 4.5 uses manual extended thinking; omitting the "
                 "`thinking` field runs it with none. No effort parameter."},
    "claude-haiku-4-5": {
        "thinking": None, "effort": None, "thinking_unavoidable": False,
        "basis": "as claude-haiku-4-5-20251001"},
    "claude-sonnet-5-5": {
        "thinking": {"type": "between_tools"}, "effort": "low",
        "thinking_unavoidable": False,
        "basis": "Sonnet 5.5 rejects thinking.type=disabled with a 400 naming "
                 "`between_tools`, which is its lowest thinking setting. Since "
                 "screening makes one tool call and stops, there is no "
                 "between-tool moment, so this is effectively no thinking. "
                 "`between_tools` is accepted at low/medium/high effort only."},
    "claude-opus-5-5": {
        "thinking": None, "effort": "low", "thinking_unavoidable": True,
        "basis": "Opus 5.5 runs adaptive thinking ALWAYS; thinking.type "
                 "disabled and enabled both return 400. The `thinking` field "
                 "is omitted (equivalent to adaptive) and effort is set to "
                 "`low`, the only lever on thinking depth. Its thinking tokens "
                 "bill as output tokens, so its cost is not comparable "
                 "like-for-like with the other two."},
}

TOOL_NAME = "screen_paper"
DECISIONS = ("include", "exclude", "uncertain")
MAX_REASON_WORDS = 25

# max_tokens is a hard cap on TOTAL output -- thinking plus response text -- on
# every current model. The tool call itself is ~60 tokens, but Opus 5.5 always
# thinks and Sonnet 5.5 may, so a tight cap would truncate those calls into
# max_tokens validation failures and measure the cap rather than the model.
# Output is billed on tokens actually produced, so headroom is close to free.
MAX_TOKENS = 4096

# Rough multiplier applied to the assumed output tokens of a model that cannot
# run without thinking. An assumption, flagged as one, used only in the
# pre-submission estimate.
THINKING_OUTPUT_FACTOR = 8

csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

# The system prompt deliberately contains NO decision guidance of its own.
#
# It once carried its own tie-break ("a false exclude cannot be recovered; when
# torn between exclude and uncertain, answer uncertain"). That was written
# before criteria.md existed. The criteria now carry a complete, ordered
# decision procedure including their own tie-breaks -- and theirs point the
# other way in one place, telling the model NOT to reach for `uncertain` merely
# because a paper looks marginal. Two sets of tie-break rules in one prompt
# would make the uncertain rate a property of the prompt rather than of the
# criteria, and the uncertain rate is a reported metric that feeds the model
# comparison. So the criteria govern the decision; this template governs only
# the task frame and the output mechanics.
SYSTEM_TEMPLATE = """\
You are screening papers for a structured literature review on agent-based \
frameworks for manufacturing systems. You decide whether a paper should be \
included in the review.

You are given a title plus ONE of: an abstract, or the first part of the paper's \
full text, or nothing beyond the title. The user message says which. Judge only \
from the text you are given. Do not use outside knowledge of the paper, its \
authors, its venue or its field.

Apply the criteria below exactly as written, including their decision procedure \
and their rules about when to answer `uncertain`. The criteria are the only \
authority on what each decision means; nothing in these instructions overrides \
them.

You must answer by calling the `{tool_name}` tool. Call it exactly once, for \
every paper, without exception -- there is no paper for which a plain text \
reply is correct, and a text reply is discarded as an invalid response. Return:
  - decision: one of include, exclude, uncertain
  - criterion_id: the ID, from the criteria below, of the single criterion that \
most drove the decision
  - reason: at most {max_words} words referring to the text you were given

Do not include internal or system XML tags in your response.

--- BEGIN CRITERIA ({criteria_basename}) ---
{criteria_text}
--- END CRITERIA ---
"""


def build_system_text(criteria_text: str, criteria_basename: str) -> str:
    return SYSTEM_TEMPLATE.format(
        tool_name=TOOL_NAME,
        max_words=MAX_REASON_WORDS,
        criteria_basename=criteria_basename,
        criteria_text=criteria_text.strip(),
    )


# The three input kinds a paper can arrive with, and the label the model sees.
# The label is not decoration: a truncated full-text head opens with title-page
# matter and an introduction, not with a structured summary of contributions,
# so a model told it is reading an abstract would treat an absent finding as a
# finding that is absent. Telling it which artefact it has lets the criteria be
# applied to what is actually there.
TEXT_KINDS = {
    "abstract": "Abstract",
    "fulltext_head": (
        "Full-text head (the first 5,000 characters of the paper, truncated "
        "mid-sentence; this is NOT an abstract -- it typically contains title "
        "page matter and the start of the introduction, and says nothing about "
        "results or conclusions, so absence of a finding here is not evidence "
        "the paper lacks it)"),
    "title_only": (
        "TITLE ONLY -- no abstract and no full text could be obtained for this "
        "paper"),
}

ABSTRACT_SOURCES = ("openalex", "semantic_scholar")
FULLTEXT_SOURCES = ("fulltext_head_oa",)


def resolve_screening_text(row: dict) -> tuple[str, str]:
    """
    Pick the text this paper is screened on, and say which kind it is.

    Driven by `abstract_source` rather than by which column happens to be
    non-empty, so the label can never drift from the provenance record. Order
    matters only in that a row carrying both an abstract and a head is screened
    on the abstract -- the better artefact.
    """
    src = (row.get("abstract_source") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    head = (row.get("fulltext_head") or "").strip()

    if src in ABSTRACT_SOURCES and abstract:
        return abstract, "abstract"
    if src in FULLTEXT_SOURCES and head:
        return head, "fulltext_head"
    # Fall back on content when abstract_source is missing entirely (e.g. a
    # hand-made input CSV), still preferring the abstract.
    if not src:
        if abstract:
            return abstract, "abstract"
        if head:
            return head, "fulltext_head"
    return "", "title_only"


def build_user_text(title: str, text: str, kind: str = "abstract") -> str:
    """
    The whole user message. Everything except the labelled body is identical
    across input kinds, so the benchmark compares inputs and not prompts.
    """
    label = TEXT_KINDS.get(kind, TEXT_KINDS["abstract"])
    if kind == "title_only" or not text.strip():
        return (f"Title: {title.strip()}\n\n{TEXT_KINDS['title_only']}.\n\n"
                f"Apply the criteria to the title alone.")
    return f"Title: {title.strip()}\n\n{label}:\n{text.strip()}"


def build_tool(strict: bool = True) -> dict:
    tool = {
        "name": TOOL_NAME,
        "description": (
            "Record the screening decision for one paper. Call exactly once."),
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": list(DECISIONS),
                    "description": "include, exclude, or uncertain",
                },
                "criterion_id": {
                    "type": "string",
                    "description": (
                        "The ID, exactly as written in the criteria, of the "
                        "single criterion that most drove this decision."),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        f"At most {MAX_REASON_WORDS} words referring to the "
                        f"text you were given. For an `uncertain` decision, "
                        f"name the specific condition that could not be "
                        f"evaluated."),
                },
            },
            "required": ["decision", "criterion_id", "reason"],
            "additionalProperties": False,
        },
    }
    if strict:
        tool["strict"] = True
    return tool


def model_kwargs(model_id: str) -> dict:
    """
    Per-model request options for "as little thinking as the model permits".

    The brief asked for no extended thinking. That is achievable on two of the
    three models and IMPOSSIBLE on the third, so it is implemented per model
    and the gap is reported rather than papered over:

      claude-haiku-4-5-20251001  omit `thinking`            -> no thinking
      claude-sonnet-5-5          thinking=between_tools     -> no up-front
                                 (`disabled` returns 400)      thinking
      claude-opus-5-5            thinking always on         -> effort=low is
                                 (`disabled` returns 400)      the only lever

    Opus 5.5's thinking tokens are billed as output tokens. Its cost in the
    benchmark therefore includes a component the other two do not pay, which
    matters because the selection rule picks the cheapest qualifying model.
    `benchmark.py` reports `thinking_unavoidable` per model so the comparison
    is read with that in mind.
    """
    spec = MODEL_REQUEST.get(model_id)
    if spec is None:
        # Unknown model: send the most conservative shape that is valid on
        # every current model rather than guessing a thinking parameter that
        # might 400. Startup validation should already have stopped this.
        return {}
    kwargs: dict = {}
    if spec["thinking"] is not None:
        kwargs["thinking"] = dict(spec["thinking"])
    if spec["effort"] is not None:
        kwargs["output_config"] = {"effort": spec["effort"]}
    return kwargs


def validate_model_ids(client, model_ids: list[str]) -> dict:
    """
    Check every requested model ID against GET /v1/models and abort on a
    mismatch. NEVER substitute.

    A wrong model ID must stop the run, not be quietly corrected: the
    substitution changes which models the benchmark is about and changes the
    cost basis the selection rule uses. The Models API is the only authority
    that is current by construction -- any table compiled into this file can go
    stale, and one did.
    """
    try:
        served = {m.id: m for m in client.models.list()}
    except Exception as exc:                      # noqa: BLE001
        sys.exit(
            f"could not reach GET /v1/models to validate model IDs: "
            f"{type(exc).__name__}: {exc}\n"
            f"The run is aborted rather than proceeding on unverified IDs.")

    missing = [m for m in model_ids if m not in served]
    if missing:
        available = ", ".join(sorted(served)[:40]) or "(none returned)"
        sys.exit(
            "\n  *** ABORT: model ID(s) not served by this account:\n"
            + "".join(f"      {m}\n" for m in missing)
            + f"\n  Models the API reports as available:\n      {available}\n"
            + "\n  Nothing was submitted. Fix --models / --model-id; this "
              "harness does not\n  substitute a different model for the one "
              "you asked for.\n")

    unpriced = [m for m in model_ids if m not in PRICING]
    if unpriced:
        sys.exit(
            "\n  *** ABORT: no price on file for:\n"
            + "".join(f"      {m}\n" for m in unpriced)
            + f"\n  Prices were last verified {PRICING_CHECKED_UTC} from\n"
              f"  {PRICING_SOURCE}\n"
            + "\n  An unpriced model would cost $0.00 in the estimate and in\n"
              "  benchmark.csv, and the selection rule picks the CHEAPEST\n"
              "  qualifying model -- so it would win on a fiction. Add its\n"
              "  input/output/cache_read_mult to PRICING and rerun.\n")

    return {
        m: {
            "served": True,
            "max_input_tokens": getattr(served[m], "max_input_tokens", None),
            "max_output_tokens": getattr(served[m], "max_tokens", None),
            "display_name": getattr(served[m], "display_name", None),
        }
        for m in model_ids
    }


# --------------------------------------------------------------------------
# criteria.md
# --------------------------------------------------------------------------

# Conservative: an ID is a short uppercase-led token in a leading position -- a
# heading, a list bullet, a table cell or a bolded lead-in. Deliberately does
# not scan prose, where "I2" could be anything.
ID_PATTERNS = [
    re.compile(r"^#{1,6}\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\b", re.M),
    re.compile(r"^\s*[-*+]\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\b", re.M),
    re.compile(r"^\s*\|\s*\**\s*([A-Z]{1,3}[-_]?\d{1,3})\**\s*\|", re.M),
    re.compile(r"^\s*\*\*([A-Z]{1,3}[-_]?\d{1,3})\*\*", re.M),
    # Bare definition-list form: an ID at the start of a line, followed by a
    # dash. This is the shape criteria.md actually uses ("I1 — Learning or
    # exploratory agents."), and the four patterns above all missed it, which
    # silently disabled membership checking on the real file. Requiring both
    # line start AND a dash separator keeps it off prose: a sentence that
    # merely mentions I1 does not begin with it and is not followed by a dash.
    re.compile(r"^\s*([A-Z]{1,3}[-_]?\d{1,3})\s*[\u2014\u2013-]\s+", re.M),
]


def extract_criterion_ids(text: str) -> list[str]:
    found: list[str] = []
    for pat in ID_PATTERNS:
        for m in pat.finditer(text):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(parsed: dict | None, failure: str | None,
             valid_ids: set[str] | None) -> tuple[str, list[str]]:
    """
    Return (status, errors). status is 'valid' or 'invalid'.

    `valid_ids` of None means membership is recorded elsewhere but not enforced.
    """
    errors: list[str] = []
    if failure:
        return "invalid", [failure]
    if parsed is None:
        return "invalid", ["no_tool_use"]

    decision = parsed.get("decision")
    if decision not in DECISIONS:
        errors.append(f"bad_decision:{decision!r}")

    cid = parsed.get("criterion_id")
    if not isinstance(cid, str) or not cid.strip():
        errors.append("empty_criterion_id")
    elif valid_ids is not None and cid.strip() not in valid_ids:
        errors.append(f"unknown_criterion_id:{cid!r}")

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("empty_reason")
    elif len(reason.split()) > MAX_REASON_WORDS:
        errors.append(f"reason_too_long:{len(reason.split())}w")

    extra = set(parsed) - {"decision", "criterion_id", "reason"}
    if extra:
        errors.append(f"unexpected_keys:{sorted(extra)}")

    return ("invalid" if errors else "valid"), errors


def parse_response(message) -> tuple[dict | None, str | None]:
    """Pull the forced tool call out of a Message. Returns (parsed, failure)."""
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        cat = getattr(details, "category", None) if details else None
        return None, f"refusal:{cat}"
    if stop == "max_tokens":
        return None, "max_tokens"

    blocks = [b for b in (message.content or [])
              if getattr(b, "type", None) == "tool_use"
              and getattr(b, "name", None) == TOOL_NAME]
    if not blocks:
        return None, "no_tool_use"
    if len(blocks) > 1:
        return None, f"multiple_tool_use:{len(blocks)}"

    raw = blocks[0].input
    if not isinstance(raw, dict):
        return None, f"tool_input_not_object:{type(raw).__name__}"
    return raw, None


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def cost_usd(model_id: str, usage: dict, batch: bool) -> float:
    """
    Price one call from its measured usage.

    The cache-read multiplier is per model (0.05x on Opus 5.5, 0.10x
    elsewhere). An unpriced model raises rather than returning 0.0: a silent
    zero would make that model the cheapest, and the selection rule picks on
    cost.
    """
    p = PRICING.get(model_id)
    if not p:
        raise KeyError(
            f"no price on file for {model_id!r}; refusing to price its calls "
            f"at $0.00 (see PRICING, last verified {PRICING_CHECKED_UTC})")
    mult = BATCH_MULT if batch else 1.0
    inp = p["input"] / 1_000_000
    out = p["output"] / 1_000_000
    return mult * (
        usage.get("input_tokens", 0) * inp
        + usage.get("cache_creation_input_tokens", 0) * inp * CACHE_WRITE_MULT
        + usage.get("cache_read_input_tokens", 0) * inp * p["cache_read_mult"]
        + usage.get("output_tokens", 0) * out
    )


def usage_dict(usage) -> dict:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(
            usage, "cache_read_input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def estimate_cost(client, papers: list[dict], model_ids: list[str],
                  system_blocks, tools, repeat: int, batch: bool,
                  assumed_output_tokens: int, exact_max: int,
                  verbose: bool) -> dict:
    """
    Price the run before committing to it.

    Input tokens are COUNTED with messages.count_tokens (free, exact) rather
    than guessed from character length. Output tokens cannot be counted ahead of
    generation, so `assumed_output_tokens` is an explicit, recorded assumption
    -- the tool output is a fixed shape, so this is a narrow guess, and the
    worst case at max_tokens is reported alongside it.

    The estimate prices every call at full input rate. Prompt caching only makes
    the real number smaller, so the guard is conservative in the direction that
    protects the budget.
    """
    exact = len(papers) <= exact_max
    sample = papers
    if not exact:
        sample = random.Random(0).sample(papers, min(50, len(papers)))

    per_model = {}
    for mid in model_ids:
        total_in = 0
        for p in sample:
            ct = client.messages.count_tokens(
                model=mid,
                system=system_blocks,
                tools=tools,
                messages=[{"role": "user",
                           "content": build_user_text(
                               p["title"], *resolve_screening_text(p))}],
            )
            total_in += ct.input_tokens
        mean_in = total_in / max(len(sample), 1)
        calls = len(papers) * repeat
        est_in = mean_in * calls
        pr = PRICING[mid]
        mult = BATCH_MULT if batch else 1.0

        # Output tokens cannot be counted before generation. On a model that
        # cannot switch thinking off, the billed output is the tool call PLUS
        # however much the model thought -- so one flat assumption across
        # models would understate exactly the expensive one. The multiplier is
        # deliberately crude and is recorded as an assumption; the actual
        # figures in calls.jsonl are what benchmark.py reports.
        spec = MODEL_REQUEST.get(mid, {})
        thinks = bool(spec.get("thinking_unavoidable"))
        out_tok = assumed_output_tokens * (THINKING_OUTPUT_FACTOR if thinks else 1)

        est = mult * (est_in * pr["input"] / 1e6
                      + out_tok * calls * pr["output"] / 1e6)
        worst = mult * (est_in * pr["input"] / 1e6
                        + MAX_TOKENS * calls * pr["output"] / 1e6)
        per_model[mid] = {
            "calls": calls,
            "mean_input_tokens": round(mean_in, 1),
            "estimated_input_tokens": int(est_in),
            "assumed_output_tokens_per_call": out_tok,
            "thinking_unavoidable": thinks,
            "estimated_usd": round(est, 4),
            "worst_case_usd_at_max_tokens": round(worst, 4),
            "system_prompt_caches": (
                mean_in >= CACHE_MIN_PREFIX.get(mid, 0)),
            "cache_min_prefix_tokens": CACHE_MIN_PREFIX.get(mid),
        }
        if verbose:
            flags = []
            if not per_model[mid]["system_prompt_caches"]:
                flags.append(
                    f"prompt under the {CACHE_MIN_PREFIX.get(mid)}-token cache "
                    f"minimum; it will NOT cache")
            if thinks:
                flags.append(
                    f"thinking cannot be disabled; output estimate scaled "
                    f"x{THINKING_OUTPUT_FACTOR} and is the least reliable here")
            print(f"  [est] {mid}: {calls} calls, ~{mean_in:.0f} in-tok each, "
                  f"${est:.4f}")
            for f in flags:
                print(f"        *** {f}")

    return {
        "method": "messages.count_tokens" + ("" if exact else " on a 50-paper sample, extrapolated"),
        "exact": exact,
        "batch_pricing_applied": batch,
        "per_model": per_model,
        "total_estimated_usd": round(
            sum(m["estimated_usd"] for m in per_model.values()), 4),
        "total_worst_case_usd": round(
            sum(m["worst_case_usd_at_max_tokens"] for m in per_model.values()), 4),
    }


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------

FORBIDDEN_COLUMNS = ("stratum", "human_decision", "human_criterion_id",
                     "human_notes", "seed_refs", "seed_roles", "seed_weight",
                     "n_seed_links", "linked_to_framework_core",
                     "screen_decision", "screen_model")


def _isolation_check(row: dict, rendered: str) -> None:
    """
    Assert that nothing the model must not see made it into the prompt.

    calibration_sample.csv legitimately carries `stratum`, so this is not a
    hypothetical: it is the one column most likely to leak if the render code is
    ever changed carelessly. Values shorter than 4 characters are skipped --
    a year or a one-digit flag would collide with abstract text and the check
    would fire on nothing.
    """
    for col in FORBIDDEN_COLUMNS:
        val = (row.get(col) or "").strip()
        if len(val) >= 4 and val in rendered:
            raise AssertionError(
                f"isolation violation: value of column {col!r} ({val!r}) "
                f"appears in the rendered prompt for {row.get('paper_id')}")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def load_eligibility(candidates_path: str, eligibility_path: str
                     ) -> dict[str, tuple[bool, str]]:
    """
    paper_id -> (eligible, exclusion_reason), from corpus/eligibility.csv.

    Checked against the CURRENT candidates.csv by eligibility.load_eligibility:
    a missing file, or one computed for a different corpus, stops the run
    before anything is priced. screen.py never decides eligibility itself.
    """
    if not os.path.exists(candidates_path):
        sys.exit(f"candidates file not found: {candidates_path}\n"
                 f"It is needed to check that eligibility.csv is current.")
    with open(candidates_path, newline="", encoding="utf-8") as f:
        crow = list(csv.DictReader(f))
    elig = el.load_eligibility(eligibility_path, crow)
    return {k: (v["eligible"], v["exclusion_reason"]) for k, v in elig.items()}


def split_eligible(rows: list[dict], elig: dict[str, tuple[bool, str]]
                   ) -> tuple[list[dict], list[dict]]:
    """
    (eligible rows, ineligible rows). Every row must be known to candidates.csv:
    an unknown paper_id is an unknown scope decision, and the run stops rather
    than guess which side it falls on.
    """
    unknown = [r["paper_id"] for r in rows if r["paper_id"] not in elig]
    if unknown:
        sys.exit(f"{len(unknown)} paper_id(s) are not in the candidates file, "
                 f"so their eligibility is unknown: {unknown[:10]}")
    keep = [r for r in rows if elig[r["paper_id"]][0]]
    drop = [r for r in rows if not elig[r["paper_id"]][0]]
    return keep, drop


def load_papers(path: str, limit: int, elig: dict[str, tuple[bool, str]]
                ) -> tuple[list[dict], list[dict]]:
    """Return (eligible papers to screen, ineligible papers recorded only)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows")
    for need in ("paper_id", "title", "abstract"):
        if need not in rows[0]:
            sys.exit(f"{path} has no {need!r} column")
    rows, ineligible = split_eligible(rows, elig)
    if ineligible:
        by_reason = Counter(elig[r["paper_id"]][1] for r in ineligible)
        print(f"[input] {len(ineligible)} row(s) are INELIGIBLE ("
              + ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
              + "): out of scope, not screened, no model call, no forced "
                "decision")
    kinds = Counter(resolve_screening_text(r)[1] for r in rows)
    print(f"[input] {len(rows)} eligible rows: " + ", ".join(
        f"{k}={v}" for k, v in sorted(kinds.items())))
    if kinds.get("title_only"):
        print(f"[input] {kinds['title_only']} eligible row(s) have neither an "
              f"abstract nor a full-text head. They are screened on TITLE ONLY "
              f"and forced to `uncertain` -- the model is not asked to guess.")
    papers = rows[:limit] if limit else rows
    return papers, ineligible


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Screen papers on title+abstract with one or more Claude models.")
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--candidates", default="corpus/candidates.csv",
                    help="the corpus eligibility.csv must have been computed "
                         "against (staleness check)")
    ap.add_argument("--eligibility", default="corpus/eligibility.csv",
                    help="eligible/exclusion_reason per paper, joined on "
                         "paper_id = openalex_id")
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--model-id", action="append", default=[],
                    metavar="ALIAS=ID", help="override a model ID")
    ap.add_argument("--runs-dir", default="screening/runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--sync", action="store_true",
                    help="bypass the Batches API (debugging 1-3 papers)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="calls per paper per model; >1 measures self-consistency")
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--assumed-output-tokens", type=int, default=80)
    ap.add_argument("--estimate-exact-max", type=int, default=200,
                    help="count tokens exactly up to this many papers; above "
                         "it, extrapolate from a 50-paper sample")
    ap.add_argument("--no-strict-tool", action="store_true",
                    help="drop strict:true from the tool definition. On by "
                         "default: with forced tool_choice unavailable on the "
                         "5.5 models, strict tool use is the documented "
                         "replacement for guaranteeing schema-valid input.")
    ap.add_argument("--criterion-ids", default=None,
                    help="comma-separated override for the IDs parsed from criteria.md")
    ap.add_argument("--criterion-id-check", choices=("auto", "strict", "off"),
                    default="auto")
    ap.add_argument("--screen-title-only", action="store_true",
                    help="send title-only papers to the model instead of "
                         "forcing them to `uncertain` in Python. Off by "
                         "default -- the forced answer is the only correct one "
                         "and a model call would just charge for it.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--yes", action="store_true",
                    help="skip the pre-submission confirmation prompt")
    args = ap.parse_args()

    for spec in args.model_id:
        if "=" not in spec:
            sys.exit(f"--model-id wants ALIAS=ID, got {spec!r}")
        alias, mid = spec.split("=", 1)
        MODEL_IDS[alias.strip()] = mid.strip()

    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    unknown = [a for a in aliases if a not in MODEL_IDS]
    if unknown:
        sys.exit(f"unknown model alias(es): {unknown}; known: {sorted(MODEL_IDS)}")
    model_ids = [MODEL_IDS[a] for a in aliases]

    if not os.path.exists(args.criteria):
        sys.exit(f"criteria file not found: {args.criteria}\n"
                 f"screening/criteria.md is written by Zachary; this harness "
                 f"does not author it.")
    with open(args.criteria, encoding="utf-8") as f:
        criteria_text = f.read()
    if not criteria_text.strip():
        sys.exit(f"{args.criteria} is empty")

    system_text = build_system_text(criteria_text, os.path.basename(args.criteria))
    criteria_sha = sha256(criteria_text)
    system_sha = sha256(system_text)

    # criterion_id set
    if args.criterion_ids:
        ids = [i.strip() for i in args.criterion_ids.split(",") if i.strip()]
        id_source = "--criterion-ids"
    else:
        ids = extract_criterion_ids(criteria_text)
        id_source = "parsed from criteria.md"

    if args.criterion_id_check == "off":
        enforce_ids, enforce_why = False, "disabled by --criterion-id-check off"
    elif args.criterion_id_check == "strict":
        if not ids:
            sys.exit("--criterion-id-check strict but no criterion IDs are known")
        enforce_ids, enforce_why = True, "forced by --criterion-id-check strict"
    else:
        enforce_ids = len(ids) >= 2
        enforce_why = (f"{len(ids)} ID(s) {id_source}; "
                       + ("enforced" if enforce_ids else
                          "NOT enforced -- fewer than 2 IDs found, which is "
                          "evidence about the parser, not about the model"))
    valid_ids = set(ids) if enforce_ids else None

    elig = load_eligibility(args.candidates, args.eligibility)
    papers, ineligible = load_papers(args.input, args.limit, elig)
    if not papers:
        sys.exit("no eligible papers to screen")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    calls_path = os.path.join(run_dir, "calls.jsonl")

    system_blocks = [{"type": "text", "text": system_text,
                      "cache_control": {"type": "ephemeral"}}]
    strict_tool = not args.no_strict_tool
    tools = [build_tool(strict_tool)]

    print(f"[run] run_id      : {run_id}")
    print(f"[run] models      : " + ", ".join(
        f"{a}={MODEL_IDS[a]}" for a in aliases))
    print(f"[run] papers      : {len(papers)} eligible  x{args.repeat} "
          f"repeat(s); {len(ineligible)} excluded at eligibility")
    print(f"[run] criteria    : {args.criteria}  sha256={criteria_sha[:16]}...")
    print(f"[run] system sha  : {system_sha[:16]}...")
    print(f"[run] criterion IDs: {ids if ids else '(none found)'}")
    print(f"[run] ID check    : {enforce_why}")
    print(f"[run] mode        : {'sync' if args.sync else 'batch (50% discount)'}")

    client = anthropic.Anthropic()

    # Before anything is priced or submitted: are these IDs real? Never
    # substitute -- abort and let the operator decide.
    print("[models] validating requested IDs against GET /v1/models ...")
    served = validate_model_ids(client, model_ids)
    for mid, info in served.items():
        req = MODEL_REQUEST.get(mid, {})
        note = ""
        if req.get("thinking_unavoidable"):
            note = "  *** thinking cannot be disabled on this model"
        print(f"  [ok] {mid}"
              + (f"  ({info['display_name']})" if info.get("display_name") else "")
              + note)

    print("[cost] estimating ...")
    est = estimate_cost(client, papers, model_ids, system_blocks, tools,
                        args.repeat, not args.sync, args.assumed_output_tokens,
                        args.estimate_exact_max, verbose=True)
    print(f"[cost] estimated total ${est['total_estimated_usd']:.4f} "
          f"(worst case ${est['total_worst_case_usd']:.4f}), "
          f"cap ${args.max_usd:g}")

    if est["total_estimated_usd"] > args.max_usd:
        print(f"\n  *** ABORT: estimated ${est['total_estimated_usd']:.4f} "
              f"exceeds --max-usd ${args.max_usd:g}. Nothing was submitted.")
        return 2

    manifest = {
        "script": "screen.py",
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "criteria_path": args.criteria,
        "criteria_sha256": criteria_sha,
        "system_prompt_sha256": system_sha,
        "system_prompt_chars": len(system_text),
        "models": {a: MODEL_IDS[a] for a in aliases},
        "models_validated_against": "GET /v1/models",
        "models_served": served,
        "model_id_substitutions": "none -- this harness never substitutes a "
                                  "model ID; a mismatch aborts the run",
        "per_model_request_shape": {
            m: {**MODEL_REQUEST.get(m, {}), "sent": model_kwargs(m)}
            for m in model_ids},
        "papers": len(papers),
        "candidates_path": args.candidates,
        "eligibility_path": args.eligibility,
        "excluded_at_eligibility": {
            "n": len(ineligible),
            "by_reason": dict(Counter(elig[r["paper_id"]][1] for r in ineligible)),
            "paper_ids": {r["paper_id"]: elig[r["paper_id"]][1]
                          for r in ineligible},
            "basis": ("eligible=False in eligibility.csv (eligibility.py). No "
                      "model call, no forced decision, no calls.jsonl row: "
                      "out of scope, not uncertain."),
        },
        "repeat": args.repeat,
        "api_mode": "sync" if args.sync else "batch",
        "thinking": (
            "minimised per model, not uniform: Haiku 4.5 omits the field, "
            "Sonnet 5.5 uses between_tools (disabled returns 400), Opus 5.5 "
            "CANNOT disable thinking and runs at effort=low. Opus 5.5's "
            "thinking tokens bill as output."),
        "tool_choice": {"type": "auto"},
        "tool_choice_basis": (
            "Forced tool_choice ({'type':'tool'}) returns 400 on "
            "claude-opus-5-5 and claude-sonnet-5-5, including on the token "
            "counting endpoint. The documented replacement is tool_choice auto "
            "plus strict tool use, applied uniformly to all three models so "
            "the benchmark compares models rather than request shapes."),
        "strict_tool": strict_tool,
        "max_tokens": MAX_TOKENS,
        "criterion_ids_known": ids,
        "criterion_ids_source": id_source,
        "criterion_id_enforced": enforce_ids,
        "criterion_id_basis": enforce_why,
        "cost_estimate": est,
        "max_usd": args.max_usd,
        "pricing_usd_per_mtok": {m: PRICING[m] for m in model_ids},
        "pricing_source": PRICING_SOURCE,
        "pricing_checked_utc": PRICING_CHECKED_UTC,
        "note": (
            "Costs in calls.jsonl are computed from the usage figures the API "
            "returned, not from this estimate. benchmark.py reports the former."
        ),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if not args.yes:
        reply = input(f"\nSubmit {sum(m['calls'] for m in est['per_model'].values())} "
                      f"calls for ~${est['total_estimated_usd']:.4f}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted by user; nothing was submitted")
            return 1

    # paper_id -> which artefact it was screened on, so the call log says what
    # the model actually saw without re-deriving it later.
    text_kind_of = {p["paper_id"]: resolve_screening_text(p)[1] for p in papers}

    log = open(calls_path, "a", encoding="utf-8")
    n_valid = 0
    n_invalid = 0
    total_cost = 0.0
    by_model_cost: dict[str, float] = defaultdict(float)
    err_counter: Counter[str] = Counter()

    def record(alias, mid, paper, rep, message, err, batch):
        nonlocal n_valid, n_invalid, total_cost
        if message is None:
            parsed, failure = None, err
            usage = {"input_tokens": 0, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0, "output_tokens": 0}
            raw = None
        else:
            parsed, failure = parse_response(message)
            usage = usage_dict(message.usage)
            raw = json.loads(message.to_json())
        status, errors = validate(parsed, failure, valid_ids)
        c = cost_usd(mid, usage, batch)
        total_cost += c
        by_model_cost[mid] += c
        if status == "valid":
            n_valid += 1
        else:
            n_invalid += 1
            for e in errors:
                err_counter[e.split(":")[0]] += 1

        cid = (parsed or {}).get("criterion_id")
        log.write(json.dumps({
            "run_id": run_id,
            "model_alias": alias,
            "model": mid,
            "criteria_sha256": criteria_sha,
            "system_prompt_sha256": system_sha,
            "paper_id": paper["paper_id"],
            "repeat": rep,
            "api_mode": "batch" if batch else "sync",
            "text_kind": text_kind_of.get(paper["paper_id"], "abstract"),
            "raw_response": raw,
            "parsed": parsed,
            "validation_status": status,
            "validation_errors": errors,
            "criterion_id_in_criteria": (
                None if not ids else (isinstance(cid, str) and cid.strip() in set(ids))),
            "input_tokens": usage["input_tokens"],
            "cache_read_input_tokens": usage["cache_read_input_tokens"],
            "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd": round(c, 8),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        log.flush()

    def force_uncertain(alias, mid, paper, rep):
        """
        An ELIGIBLE row with neither an abstract nor a full-text head is forced
        to `uncertain` without a model call. Ineligible rows never get here.

        `uncertain` routes to a human, and that is already the correct and only
        available outcome: there is no evidence for the criteria to be applied
        to. Paying a model to read a bare title and return a predetermined
        answer would buy nothing, would put a fabricated judgement in the
        record, and would charge for it. Python makes this decision, which is
        the right place for a rule (S3.3); the model is not consulted and is
        not blamed for the result. `--screen-title-only` sends them anyway if
        the rate is ever worth measuring for the report.
        """
        log.write(json.dumps({
            "run_id": run_id, "model_alias": alias, "model": mid,
            "criteria_sha256": criteria_sha, "system_prompt_sha256": system_sha,
            "paper_id": paper["paper_id"], "repeat": rep,
            "api_mode": "forced_no_model_call",
            "text_kind": "title_only",
            "raw_response": None,
            "parsed": {"decision": "uncertain",
                       "criterion_id": "",
                       "reason": "No abstract or full text available; forced to "
                                 "uncertain without a model call."},
            "validation_status": "forced",
            "validation_errors": [],
            "criterion_id_in_criteria": None,
            "input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        log.flush()

    forced = 0
    if not args.screen_title_only:
        title_only = [p for p in papers
                      if resolve_screening_text(p)[1] == "title_only"]
        papers = [p for p in papers
                  if resolve_screening_text(p)[1] != "title_only"]
        for alias, mid in zip(aliases, model_ids):
            for rep in range(1, args.repeat + 1):
                for p in title_only:
                    force_uncertain(alias, mid, p, rep)
                    forced += 1
        if title_only:
            print(f"[force] {len(title_only)} title-only paper(s) forced to "
                  f"`uncertain` with no model call ({forced} log rows)")

    try:
        if args.sync:
            for alias, mid in zip(aliases, model_ids):
                for rep in range(1, args.repeat + 1):
                    for p in papers:
                        text, kind = resolve_screening_text(p)
                        user_text = build_user_text(p["title"], text, kind)
                        _isolation_check(p, user_text)
                        try:
                            msg = client.messages.create(
                                model=mid, max_tokens=MAX_TOKENS,
                                system=system_blocks, tools=tools,
                                tool_choice={"type": "auto"},
                                messages=[{"role": "user", "content": user_text}],
                                **model_kwargs(mid),
                            )
                            record(alias, mid, p, rep, msg, None, batch=False)
                        except anthropic.APIError as exc:
                            record(alias, mid, p, rep, None,
                                   f"api_error:{type(exc).__name__}", batch=False)
                            print(f"  [err] {mid} {p['paper_id']}: {exc}")
        else:
            from anthropic.types.message_create_params import (
                MessageCreateParamsNonStreaming)
            from anthropic.types.messages.batch_create_params import Request

            by_id = {p["paper_id"]: p for p in papers}
            for alias, mid in zip(aliases, model_ids):
                requests = []
                for rep in range(1, args.repeat + 1):
                    for p in papers:
                        text, kind = resolve_screening_text(p)
                        user_text = build_user_text(p["title"], text, kind)
                        _isolation_check(p, user_text)
                        requests.append(Request(
                            custom_id=f"{p['paper_id']}__r{rep}",
                            params=MessageCreateParamsNonStreaming(
                                model=mid, max_tokens=MAX_TOKENS,
                                system=system_blocks, tools=tools,
                                tool_choice={"type": "auto"},
                                messages=[{"role": "user", "content": user_text}],
                                **model_kwargs(mid),
                            ),
                        ))
                batch = client.messages.batches.create(requests=requests)
                print(f"[batch] {alias} ({mid}): {batch.id}, "
                      f"{len(requests)} requests")
                while True:
                    batch = client.messages.batches.retrieve(batch.id)
                    if batch.processing_status == "ended":
                        break
                    counts = batch.request_counts
                    print(f"  [batch] {alias}: {batch.processing_status} "
                          f"processing={counts.processing} "
                          f"succeeded={counts.succeeded} errored={counts.errored}",
                          flush=True)
                    time.sleep(args.poll_seconds)

                # Results arrive in any order -- key by custom_id, never position.
                for result in client.messages.batches.results(batch.id):
                    pid, _, rep_s = result.custom_id.partition("__r")
                    rep = int(rep_s or 1)
                    paper = by_id.get(pid, {"paper_id": pid})
                    rtype = result.result.type
                    if rtype == "succeeded":
                        record(alias, mid, paper, rep, result.result.message,
                               None, batch=True)
                    else:
                        detail = getattr(getattr(result.result, "error", None),
                                         "type", rtype)
                        record(alias, mid, paper, rep, None,
                               f"batch_{rtype}:{detail}", batch=True)
    finally:
        log.close()

    summary = {
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "calls_logged": n_valid + n_invalid,
        "validated": n_valid,
        "failed_validation": n_invalid,
        "validation_failure_kinds": dict(err_counter),
        "actual_cost_usd": round(total_cost, 6),
        "actual_cost_usd_by_model": {k: round(v, 6) for k, v in by_model_cost.items()},
        "estimate_vs_actual": {
            "estimated_usd": est["total_estimated_usd"],
            "actual_usd": round(total_cost, 6),
        },
    }
    manifest["result"] = summary
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print()
    print(f"[done] {n_valid} valid, {n_invalid} failed validation")
    if err_counter:
        print(f"[done] failure kinds: " + ", ".join(
            f"{k}={v}" for k, v in err_counter.most_common()))
        print("       (not retried -- a silent retry would make the malformed-"
              "output rate unmeasurable)")
    print(f"[done] actual cost ${total_cost:.4f} "
          f"(estimated ${est['total_estimated_usd']:.4f})")
    print(f"  -> {calls_path}")
    print(f"  -> {os.path.join(run_dir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
