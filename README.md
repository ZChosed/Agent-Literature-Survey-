# litreview

`litreview` takes a handful of **base papers** you already trust. It builds the
literature around them from citation links, has Claude screen and extract from
that literature, and produces a **report whose every number is computed in
Python**. Every decision is written to disk, so someone else can audit how the
result was reached.

Use it to test a suspected research gap. If you think nobody has done X, the
pipeline gives you a defined corpus, a calibrated screen, and counts with
verbatim evidence for how often X does appear. Read
[What the results can and cannot tell you](#what-the-results-can-and-cannot-tell-you-about-a-gap)
before you write "no prior work".

```
base papers ─► build ─► eligibility ─► calibrate ─► screen ─► fetch ─► extract ─► report
 seeds.csv   candidates   open access   κ / recall   include?  full text  evidence   tables
```

---

## Before you start

| You need | Notes |
|---|---|
| Python 3.11 or newer | `python3 --version` to check |
| An Anthropic API key with access to `claude-opus-5-5` | Needed from the screening step onward. Ask the lab for a key, and don't commit it. |
| An email address | OpenAlex and Unpaywall ask for a contact address. It goes in `project.toml`. |
| 3–10 base papers | The papers your gap is defined against. **They decide what the corpus contains, so choose them carefully.** |
| A few hours of your own time | Mostly writing criteria, labelling 50 papers, and checking results. |

**Cost.** Corpus building and the open-access check are free. Screening and
extraction call Claude through the Batches API at half price. For reference,
the original NIST review spent $0.23 calibrating on 50 papers and $6.28
screening 1,091 papers. Extraction reads full texts, so each paper costs more.
Every model step prints an estimate and asks before it sends anything. It
refuses to run if the estimate is over your `max_usd` cap.

---

## Step 1: Install

```bash
git clone git@github.com:ZChosed/Agent-Literature-Survey-.git litreview-tool
cd litreview-tool
python3 -m venv .venv
source .venv/bin/activate              # run this again in every new terminal
pip install -e .                       # installs the `litreview` command
python tests/test_litreview.py         # offline self-test; should end "ALL CHECKS PASSED"
```

Set your API key in the shell, or put the line in `~/.bashrc` / `~/.zshrc`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Step 2: Create your review project

Each review gets its own folder. Put it **outside** the tool's folder and make
it its own git repository. Git history is how you show that settings were
fixed before results were seen.

```bash
cd ~/research
litreview init my-gap-review
cd my-gap-review
git init
```

This creates `project.toml`, `seeds.csv`, `criteria.md`, `schemas/example_schema.json`
and a `.gitignore` that keeps downloaded PDFs out of git.

Run `litreview status` at any time to see what's done and what to run next.

## Step 3: Fill in the four input files

### 3a. `project.toml`: settings

Open it and set at least:

```toml
[project]
name = "Digital twins for SME machining: gap check"
contact_email = "you@university.edu"          # required

[corpus]
min_year = 2015                               # forward citations before this are dropped
core_roles = ["core"]                         # seed roles that define the review's core
coverage_roles = ["survey"]                   # seeds whose reference lists check coverage

[screening]
topic = "digital twins for machining in small manufacturers"
max_usd = 20.0                                # hard cap for one screening run
```

Leave `kappa_floor = 0.70` and `recall_floor = 0.95` unless you have a reason to
change them, and if you do, change them now. Changing a floor after seeing a
result defeats the purpose of having one. Every setting is commented in the
file.

### 3b. `seeds.csv`: your base papers

One row per paper:

```csv
ref,role,expand_backward,expand_forward,year,first_author,title,doi
1,core,yes,yes,2021,Smith,A digital twin framework for CNC machining,10.1016/j.jmsy.2021.01.001
2,core,yes,yes,2022,Lee,Edge-deployed twins for small job shops,
3,survey,yes,no,2023,Garcia,Digital twins in manufacturing: a review,10.1109/xxxx
```

- **`doi`** resolves exactly. Without one, the title is matched and you must check the match in Step 4.
- **`role`** is free text. Seeds whose role is in `core_roles` mark the centre of your review. Papers linked to them rank first.
- **`expand_backward`** = include the papers this seed cites. **`expand_forward`** = include papers that cite this seed. These two columns are the **only** retrieval rule; there is no keyword search. For broad surveys, `yes,no` is usually right, because their citing papers are a large, unfocused set.

### 3c. `criteria.md`: what counts as relevant

This is the most important file. The model and the person labelling the
calibration sample both apply it word for word. Keep the template's structure:

```markdown
## Include
I1 — The paper proposes or evaluates a digital twin of a machining process.
I2 — ...
## Exclude
E1 — Not a research paper (editorial, erratum, call for papers).
E3 — Outside the review's domain.
## Uncertain
U1 — The abstract is too short to evaluate I1.
```

Each criterion ID must start a line and be followed by a dash (`I1 — `). The
tool parses these IDs and rejects any model answer that cites an unknown one.
**Write criteria that define your gap.** If your gap is "twins validated on real
shop-floor hardware", the include criteria should capture the whole topic, and
the extraction schema (next) should record which validation level each paper
reached. Don't screen the gap out at this stage; you want to count it.

### 3d. `schemas/*.json`: what to record from each included paper

Start from `schemas/example_schema.json`. Each field typed `evidence_object` is
extracted with a **verbatim quote and page number**, and the quote is then
checked against the PDF text:

```json
"validation_level": {"type": "evidence_object", "value_type": "enum",
                     "allowed": ["conceptual_only", "simulation", "lab_hardware", "field_deployment"],
                     "prompt": "What is the strongest validation the paper reports?"}
```

`value_type` is one of `enum`, `multi_enum`, `string`, or `object` (with
`sub_fields`). Use `enum` and `multi_enum` for anything you plan to count:
they become frequency tables. In `project.toml`, map include criteria to
schemas:

```toml
[extraction.schemas.main]
schema = "schemas/my_schema.json"
criteria = ["I1", "I2"]       # included papers whose deciding criterion is listed go here
max_papers = 0                # 0 = all; N = the top N by corpus rank
```

### 3e. Check your inputs

```bash
litreview check
```

This reads every input file and tests your API key. It changes nothing and
costs nothing. Each line is `ok`, `WARN` (allowed, but probably not what you
meant), or `FAIL` (a later step would stop, or would quietly do the wrong
thing). Fix every `FAIL` before going on. It catches:

- template text still in `seeds.csv`, `criteria.md` or the schema
- a missing `contact_email`, or floors and caps outside their valid range
- misspelled settings in `project.toml`, which would otherwise be ignored without a word
- `expand_backward` / `expand_forward` values other than yes/no (anything else reads as no)
- `core_roles` or `target_refs` that match none of your seeds
- fewer than 2 criterion IDs parsed from `criteria.md`
- a schema that doesn't load, or a field missing its `allowed` options
- a schema mapped to a criterion ID that isn't in `criteria.md`
- crosstabs naming fields that don't exist
- inputs not yet committed to git
- an API key that is missing, or doesn't have access to the configured model

Use `litreview check --offline` to skip the API test. Run the check again
whenever you edit an input.

### 3f. Commit

```bash
git add -A && git commit -m "Inputs fixed before any run"
```

Commit again whenever you change an input, **before** the run it affects.

## Step 4: Build the corpus (free)

```bash
litreview build
```

This resolves your seeds in OpenAlex, follows their citation links, filters by
year, type and language, and writes a ranked `data/candidates.csv`.

**Then check `data/seeds_resolved.csv` by hand.** Each seed is flagged `ok`,
`??` (low-confidence title match), or `--` (not found). A wrong match quietly
skews the whole corpus. Fix the cause (add the DOI, correct the title), then run
`build` again. Also read the printed lines:

- `*** cap hit`: a seed had more than 600 links in one direction and only the first 600 were kept. Raise `max_backward_per_seed` / `max_forward_per_seed` or accept it; the truncation is recorded either way.
- `coverage: seed [3] 85/120 (70.8%)`: share of a survey's references that ended up in your corpus. Low coverage means your seeds miss part of the field.

## Step 5: Apply the open-access rule (free)

```bash
litreview eligibility
```

This marks each candidate eligible or not in `data/eligibility.csv`. By default
only open-access papers are eligible, because the pipeline can only extract from
PDFs it can legally fetch. If some rows can't be decided, it writes nothing and
lists them. Run it again (network errors are retried); if a row still can't be
decided, record your call in `data/eligibility_overrides.csv`:

```csv
openalex_id,eligible,exclusion_reason,reason,decided_by,decided_on
W1234567890,True,none,Author PDF on lab site,ZC,2026-11-23
```

Rebuilding the corpus makes this file **STALE**, and later steps will refuse to
run until you rerun `eligibility`.

## Step 6: Calibrate the screener against yourself

Before the model screens hundreds of papers, check that it agrees with a human
on a sample.

**6a. Draw a blind sample.**

```bash
litreview calibrate draw
```

This writes `screening/labels_TODO.csv` with 50 papers: 25 from the top third
of the ranking and 25 from the rest. Each row has a title and abstract and
nothing else.

**6b. Label it yourself, before seeing any model output.** For every row, fill:

| column | value |
|---|---|
| `human_decision` | `include`, `exclude`, or `uncertain` |
| `human_criterion_id` | the criterion ID from `criteria.md` that decided it |
| `human_notes` | optional |

Save the file as **`screening/labels.csv`**. Every row must be filled, or
scoring refuses to run. Ideally the labeller is someone other than the person who
wrote the criteria.

**6c. Screen the same sample with the model, and score it.**

```bash
litreview screen --sample --sync --limit 2    # smoke test: 2 papers, a few cents
litreview screen --sample                     # the full 50
litreview calibrate score
```

The result is **PASS** if binary κ ≥ 0.70 **and** include recall ≥ 0.95.
Recall matters most here: a relevant paper the model wrongly excludes is never
seen again, and every one of those makes a gap look bigger than it is.

**6d. If it FAILS:** read `screening/runs/sample_<time>/disagreements.csv`,
starting with the `FALSE_EXCLUDE` rows. Clarify the criterion the model
misread, commit `criteria.md`, and repeat 6c. Don't change your labels to match
the model. If you do have to revise a label, record that, because the
calibration is no longer independent. `litreview calibrate draw` again gives a
fresh sample; the old one is archived, not deleted.

## Step 7: Screen the whole corpus

```bash
litreview screen
```

Only eligible papers are sent. Papers with no abstract aren't sent at all; they
go to human review. To give those papers a chance at model screening, run
`litreview fetch --missing-abstracts` **before** this step. The model then sees
the first part of each one's full text instead.

Results go to `screening/runs/corpus_<time>/`.

## Step 8: Settle the borderline papers and fetch full texts

```bash
litreview fetch
```

This gives every candidate exactly one outcome in `screening/outcomes.csv`:
`included`, `needs_human_review`, `excluded_at_screening`, or
`excluded_at_eligibility`. It also writes `screening/document_list.csv` (the
included papers) and downloads their open-access PDFs.

**8a. Decide the review pile.** Open `screening/needs_human_review.csv` (model
said uncertain, title only, or a failed response). Record each decision in
`screening/review_decisions.csv`:

```csv
paper_id,decision,criterion_id,decided_by
W1234567890,include,I1,ZC
W2345678901,exclude,E3,ZC
```

Then run `litreview fetch` again so the newly included papers are fetched.

**8b. Get the PDFs that couldn't be downloaded.** `manual_pdfs/RETRIEVE.md`
lists them with DOIs. Download each through the library, drop it into
`manual_pdfs/` with any filename, and run:

```bash
litreview fetch --manual-only
```

PDFs are matched to papers by the title on their first pages. If a match is
ambiguous, rename the file to `<paper_id>.pdf`. `litreview fetch --retry`
retries automatic downloads that failed before.

## Step 9: Extract evidence

```bash
litreview extract --sync --limit 2     # smoke test on 2 papers
litreview extract                      # every fetched paper
```

Each paper is read against your schema. Each answer is checked in Python: the
quote must appear verbatim, on the cited page, within 25 words. Answers that
fail the check are **left out of every table**. "No / none" answers can't be
quoted, so they're counted separately as `absence_unquoted`. If you edit the
checks or the texts, `litreview extract --recheck` re-runs the checks with no
API calls.

**Check a sample yourself.** Open some records in `extraction/records/*.json`,
starting with `absence_unquoted` fields. Those are the answers a gap claim rests
on. When you've checked a record, set `"human_verified": true` in it. The report
states how many records were checked.

## Step 10: Generate the report

```bash
litreview report
```

The first run creates `report_template.md`, a draft whose numbers are
placeholders such as `{{agg.papers.total}}` and
`{{table:main__B_VALIDATION__validation_level}}`. Edit the prose as you like,
but keep the numbers as placeholders so none is typed by hand. Run
`litreview report` again to write `report.md`. If any placeholder doesn't
resolve, nothing is written.

Frequency tables for every field are in `extraction/tables/`. For a cross-tab,
add one to `project.toml`:

```toml
[report]
crosstabs = [["main", "B_VALIDATION__validation_level", "A_CONTRIBUTION__contribution_type"]]
```

Commit everything except `documents/` and `manual_pdfs/*.pdf` (already
gitignored). The repository is now the full record of your review.

---

## What the results can and cannot tell you about a gap

A low count in your tables means **"rare in this corpus"**. It doesn't mean
"absent from the literature". Before you claim a gap, check and report each of
these:

1. **The corpus comes from your seeds.** Papers with no citation link to them
   aren't in it. Check the `coverage` lines from Step 4. If coverage is low, or
   an expert names a relevant paper that's missing, add seeds and rebuild.
2. **Only open-access papers are read.** `data/provenance.json` →
   `eligibility.by_reason` shows how many were excluded as `no_oa`, and
   `core_linked.fraction_excluded` shows how many of those were close to your
   core. A gap that might sit behind paywalls is a weaker claim.
3. **Screening recall is below 100%.** State the calibration κ and recall
   (`litreview status` prints them).
4. **Failed and absence answers are excluded or counted separately.** Each
   table records its n. Report n, not just percentages.
5. **Check the papers that do mention your gap.** A count of 2 out of 300 is
   only meaningful once you've read those 2.

A strong gap claim reads like: *"Of N open-access papers linked to these seeds,
screened with κ = …, recall = …, k reported field deployment (quotes verified
on the page)."* Every number in that sentence comes from the report.

---

## Troubleshooting

| Message | What to do |
|---|---|
| Anything going wrong before or during a run | Run `litreview check` first; it names most input mistakes directly |
| `ModuleNotFoundError: No module named 'requests'` | Activate the venv: `source <tool folder>/.venv/bin/activate` |
| `no project.toml in …` | `cd` into your project folder, or pass `litreview --project <dir> …` |
| `set [project] contact_email` | Fill it in `project.toml` |
| `STALE: … eligibility.csv was computed for a different candidates.csv` | You rebuilt; run `litreview eligibility` (`--offline` replays the saved lookups) |
| `ABORT: model … is not served by this account` | Your API key doesn't have that model. Get a key that does. The tool never switches models for you. |
| `*** ABORT: estimate $… exceeds max_usd` | Nothing was sent. Raise `max_usd` (and commit), or test with `--limit` |
| `REFUSING TO SCORE: … without a valid human_decision` | Finish labelling every row in `screening/labels.csv` |
| `pool too small` at `calibrate draw` | Fewer than 75 eligible papers with abstracts. Add seeds or lower `calibration_per_stratum` |
| `criterion IDs … NOT enforced` | `criteria.md` IDs aren't at line start followed by a dash |
| `[manual] x.pdf: no unambiguous match` | Rename it to `<paper_id>.pdf` |
| `unresolved placeholders; report NOT written` | Fix or remove the listed placeholders in `report_template.md` |

---

## Reference

### Commands

| Command | Does | Costs money |
|---|---|---|
| `litreview init <dir>` | Create a new project from templates | no |
| `litreview status` | Show finished stages and the next command | no |
| `litreview check [--offline]` | Validate every input and the API key; changes nothing | no |
| `litreview build` | Seeds → `data/candidates.csv` | no |
| `litreview eligibility [--offline]` | Open-access rule → `data/eligibility.csv` | no |
| `litreview calibrate draw` | Blind 50-paper sample → `screening/labels_TODO.csv` | no |
| `litreview screen --sample` | Model screens the calibration sample | yes |
| `litreview calibrate score [--run DIR]` | κ / recall vs your labels → PASS/FAIL | no |
| `litreview screen` | Model screens every eligible candidate | yes |
| `litreview fetch [--missing-abstracts \| --manual-only \| --retry \| --run DIR]` | Outcomes, document list, PDFs → text | no |
| `litreview extract [--recheck]` | Schema-driven extraction with quote checks | yes (not `--recheck`) |
| `litreview report [--scaffold] [--force]` | Tables and `report.md` | no |

`screen` and `extract` also take `--sync` (no batching, for quick tests),
`--limit N`, `--only id1,id2`, and `--yes` (skip the cost prompt).

### Project layout

```
project.toml  seeds.csv  criteria.md  schemas/  report_template.md   ← you write these
data/         candidates, seeds_resolved, eligibility, provenance.json, call logs
screening/    labels, runs/<run>/, outcomes, document_list, review_decisions
documents/    PDFs + extracted text (gitignored)   manual_pdfs/  drop folder + RETRIEVE.md
extraction/   targets, runs/, records/, tables/    report.md
```

Each stage records its settings and counts in `data/provenance.json`, and every
model call is logged in its run folder.

### What each stage guarantees

- **build.** Only the seeds' expand columns decide retrieval. Seed weights (topical precision of each seed's references and citing works against the core seeds' topics) only order the list; they never remove anything. Every cap that cuts results is reported.
- **eligibility.** `candidates.csv` is never modified. Nothing is written while any row is undecided. A rebuilt corpus makes the result STALE, and every later stage then refuses to run.
- **calibrate.** Fixed-seed stratified draw. The sheet shows no rank or seed information. An existing sample is archived, never overwritten. A partly filled sheet is refused.
- **screen.** Ineligible papers are never sent. Title-only papers are forced to `uncertain`. The model ID is checked against the API and never substituted. Malformed responses are logged and never retried. Reasons over 40 words are warnings; 50 or more fail.
- **fetch.** A failed model response goes to human review, never to exclude. PDFs come from open-access sources only (OpenAlex, Unpaywall, Semantic Scholar, Zenodo, arXiv), with no publisher scraping. A PDF is accepted only if it has ≥ 3,000 characters and its title on the first two pages.
- **extract.** Every quote is checked: verbatim, on the cited page, within the word limit. Failed fields are excluded from every table.
- **report.** Only passing fields are counted, and each table records its n. An unresolved placeholder writes nothing.

## History

This repository began as the NIST MSR Phase II review of agent-based
manufacturing frameworks (Tilbury/Barton). That project's complete state is the
git tag `nist-agents-review-2026-10`: corpus, calibration runs, labels,
extraction records and the Report 1 draft. Its methods rationale and decision
log are in [docs/methods-log-nist-agents.md](docs/methods-log-nist-agents.md).
It's a worked example of every step above.
