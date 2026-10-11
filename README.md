# litreview

A literature review pipeline that starts from a handful of **base papers** and ends
at a **report**, with every step recorded. OpenAlex supplies the corpus; Claude
screens and extracts. Every count in the output is computed in Python, never by a
model.

```
base papers ─► build ─► eligibility ─► calibrate ─► screen ─► fetch ─► extract ─► report
 seeds.csv   candidates   open access   κ / recall   include?  full text  evidence   tables
```

## Install

```bash
pip install -e .                       # Python 3.11+; installs the `litreview` command
export ANTHROPIC_API_KEY=...           # needed from `screen` onward
```

## Quickstart

```bash
litreview init my-review && cd my-review
# edit project.toml, seeds.csv, criteria.md and schemas/ (see below)
litreview build                 # seeds → ranked candidates      (OpenAlex; free)
litreview eligibility           # open-access rule               (OpenAlex + Unpaywall; free)
litreview calibrate draw        # blind sample → screening/labels_TODO.csv
#   label it yourself, save as screening/labels.csv
litreview screen --sample       # model screens the same sample
litreview calibrate score       # κ, include recall vs your labels → PASS / FAIL
litreview screen                # screen every eligible candidate
litreview fetch                 # document list → open-access PDFs → full text
#   fetch PDFs listed in manual_pdfs/RETRIEVE.md by hand; then: litreview fetch --manual-only
litreview extract               # schema-driven extraction with verbatim-quote checks
litreview report                # tables + report.md (template scaffolded on first run)
litreview status                # where you are; what to run next
```

Model stages print a cost estimate and ask before submitting. They abort without
sending anything if the estimate exceeds `max_usd`. Smoke-test first with
`--sync --limit 2`.

## What you provide

| File | What it is |
|---|---|
| `project.toml` | Every setting, documented inline. Commit it **before** the run it governs, so git history shows the settings were fixed in advance. |
| `seeds.csv` | Your base papers: `ref, role, expand_backward, expand_forward, year, first_author, title, doi`. A DOI resolves exactly; otherwise the title is matched. `role` is free text. Roles listed in `core_roles` mark the core of the review and rank first. |
| `criteria.md` | The screening criteria, for the model and the human rater alike. IDs at line start followed by a dash (`I1 — …`) are parsed and enforced. |
| `schemas/*.json` | What to extract. Every field typed `evidence_object` (enum, multi_enum, string, or object) is extracted with a verbatim quote and page. Map criteria to schemas under `[extraction.schemas.<name>]`; use `max_papers` to cap one by corpus rank. |
| `report_template.md` | Optional. Scaffolded on the first `litreview report`, then yours to edit. Prose, with numbers as placeholders such as `{{agg.papers.total}}` and `{{table:main__B_VALIDATION__validation_level}}`. |

## What each stage guarantees

- **build.** Seeds are expanded backward and forward, per their columns, and nothing else decides retrieval. Seeds are weighted by topical precision against the target seeds' topics: share of references on-target (p) and of a seeded random sample of citing works (q). Weights only order the list. Every cap that bites is reported as a truncation. Verify `data/seeds_resolved.csv` by hand.
- **eligibility.** Writes `data/eligibility.csv`; `candidates.csv` is never modified. If any row can't be decided, nothing is written. A row neither source can answer for needs a recorded human decision in `data/eligibility_overrides.csv`. A rebuilt corpus makes eligibility **STALE**, and every later stage refuses to run until it's recomputed. `--offline` replays the call log.
- **calibrate.** Stratified draw at a fixed seed. The blind sheet shows no rank or seed links. An existing sample is archived, never overwritten. A partly filled sheet is refused. Binary κ is primary; include recall weighs most, because a false exclude is never seen again. **Labels revised after seeing model output no longer give an independent calibration.**
- **screen.** Ineligible papers are never sent. Title-only papers are forced to `uncertain` with no model call. The model ID is checked against the API and never substituted. Malformed responses are logged and never retried. Reasons over the word limit are warnings; at 50+ words they fail.
- **fetch.** Each candidate gets one outcome: included, needs human review, excluded at screening, or excluded at eligibility. A failed response goes to review, never to exclude. Decide review rows in `screening/review_decisions.csv`. PDFs come from open-access sources only, with no publisher-HTML scraping and no bot-block evasion. Each PDF must have enough text and its title on its first pages. Manual PDFs can have any filename; they're matched by title.
- **extract.** Every quote is checked: verbatim, on the cited page, within the word limit. "No/none" answers can't be quoted and are counted apart. Failed fields are excluded from every table. `--recheck` re-runs the checks offline.
- **report.** Only passing fields are counted, and each table records its n. An unresolved placeholder writes nothing.

Every stage writes its block of `data/provenance.json`, and every model call is
logged in its run folder.

## Project layout

```
project.toml  seeds.csv  criteria.md  schemas/  report_template.md   ← yours
data/         candidates, eligibility, provenance, call logs (caches)
screening/    calibration sample, labels, runs/<run>/, outcomes, document list
documents/    PDFs + full text (gitignored)          manual_pdfs/  drop folder + RETRIEVE.md
extraction/   targets, runs/, records/, tables/      report.md
```

## Tests

```bash
python tests/test_litreview.py      # offline: no network, no API key
```

## History

This repository began as the NIST MSR Phase II review of agent-based
manufacturing frameworks (Tilbury/Barton). That project's complete state is the
git tag `nist-agents-review-2026-10`: corpus, calibration runs, labels,
extraction records and the Report 1 draft. Its methods rationale and decision
log are in [docs/methods-log-nist-agents.md](docs/methods-log-nist-agents.md).
