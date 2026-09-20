# Literature review pipeline — Stage 1 files

ROB 590, Phase II (AI-Agents). NIST MSR grant, Tilbury/Barton.
Built 2026-09-20. Proposal reference: `NIST_AI_DT_2026update`; bracketed
numbers below are its reference-list numbers.

## Files

| File | What it is |
|---|---|
| `agent_extraction_schema.json` | 11 blocks, 67 fields. Every block traced to the proposal clause that motivates it (22 `proposal_basis` annotations). |
| `dt_extraction_schema.json` | 8 blocks, 49 fields. Deliberately shallow — supports the Phase II argument, does not replace the Phase I student's review. |
| `seeds.csv` | The proposal's own bibliography, 35 rows, tagged by role. |
| `corpus_build.py` | Deterministic OpenAlex seed-and-expand. v1.1.0. |
| `validate_schemas.py` | Structural checks on both schemas. Run after any schema edit. |
| `test_corpus_build.py` | Offline smoke test, 33 assertions, no network needed. |

## Run order

```bash
pip install requests

python validate_schemas.py         # must print PASS
python test_corpus_build.py        # must print ALL CHECKS PASSED

python corpus_build.py --mailto zchosed@umich.edu --coverage-check
```

**This has not been run against the live OpenAlex API.** The cloud container it
was built in cannot reach `api.openalex.org` (proxy allowlist denies CONNECT).
The logic is verified against a mock; the first live run happens on your
machine. Expect ~600–900 candidates with default caps.

A first pass with `--max-forward-per-seed 200` finishes much faster and is
enough to sanity-check seed resolution before committing to the full pull.

### After the first live run

1. **Open `corpus/seeds_resolved.csv` and verify every row by hand.** Sort by
   `similarity`. Anything marked `RESOLVED_LOW_CONFIDENCE` or
   `REJECTED_LOW_SIMILARITY` needs a DOI pasted into `seeds.csv`, then re-run.
   A wrong seed silently biases the entire corpus, and this is the one step
   that cannot be automated away.
2. Check `corpus/coverage_report.csv`. If a `survey_baseline` paper
   [6, 23, 24, 27] has a low share of its references in the corpus, the seed
   set is too narrow — add seeds rather than loosening filters.
3. `corpus/provenance.json` records every parameter, count and timing. It is
   Report 2 material (proposal §2.2: "summarize how the AI tools were
   successful (or not)"), not a log file to discard.

## Design decisions worth knowing

**Schema before search.** The schema is derived top-down from proposal text,
not bottom-up from what papers happen to report. §2.2 declines to write "yet
another survey of agent-based approaches to manufacturing [6,23,24,27]" and
asks instead for "features that are common with our definition, as well as
those that differ" — a structured comparison against [7,8,10], not a narrative.

**Evidence quotes are mandatory.** Every substantive field returns
`{value, evidence_quote, page, confidence}`, max 25 verbatim words. This is the
primary hallucination control and it answers §3.3 directly: foundation models
"require validation through deterministic tools or human oversight" [26].

**No LLM ever counts anything.** Retrieval, deduplication, scoring and all
aggregation happen in Python. §3.3 records that foundation models "exhibit
inconsistencies in numerical accuracy"; §2.4 repeats it. Designing a pipeline
that ignores the project's own background section would be indefensible in the
report.

**`explores_beyond_predefined_env` is the load-bearing field.** §2.2 and §3.2
both frame the research gap in exactly those terms — prior agents "were not
able to explore beyond their pre-defined environment." Expect a strong majority
of `no`. That distribution is what justifies the Phase II case study.

**Citation velocity, not raw count, sets reading order.** Raw counts
systematically demote recent work; §2 calls these "emerging, exciting, and
fast-moving areas." Raw `cited_by_count` is still in the CSV — velocity only
decides what gets read first. (This changed in v1.1.0 after the smoke test
surfaced a 2016 paper outranking a 2025 LLM-agent paper.)

**Shared join keys with the Phase I schema.** `validation_level`,
`metrics_reported` and `application_domain` use identical enums across both
schemas, enforced by `validate_schemas.py`. §2.3 Phase III compares DT and
agent features on common axes; if the two reviews use different dimensions,
that comparison is unrecoverable without re-extraction. **Agree this spine with
the Phase I student before either of you extracts anything.**

## Open items for the weekly meeting

- **Timeline divergence.** Figure 6 schedules the DT literature survey for Fall
  2026 and the manufacturing-agents survey for Winter 2027. Running the agent
  survey now, in parallel, is probably better — but it means both schemas are
  being designed simultaneously rather than sequentially. Shared spine, agreed
  this week, is the cheap fix.
- Confirm [2] and IEEE 10973115 are correctly placed in `seeds.csv`. Ref [2]
  (Bi 2021) is currently tagged `agent_related`; the professors named it as core
  reading alongside [10], so it may belong in `framework_core` — which would
  change the sort order of the whole candidate set.
- Proposal refs [1] and [32] are websites (ADI, NSF IUCRC), not papers, so they
  are intentionally absent from `seeds.csv`.

## Next stage (not yet built)

Screening criteria + the 50-paper hand-label that produces the Cohen's κ, then
the extraction harness. κ and the per-field human-correction rates are the only
quantitative evidence Report 2 will have, so they get built as instrumentation,
not as an afterthought.
