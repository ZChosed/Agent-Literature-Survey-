#!/usr/bin/env python3
"""
draw_calibration_sample.py -- stratified 50-paper calibration draw + blind sheet.

Why stratified rather than random
---------------------------------
A uniform 50 from 1,941 ranked candidates would be dominated by single-seed-link
papers, which screen `exclude` almost without exception. Cohen's kappa on a
sample with two or three positives is dominated by its own sampling noise, and
include-recall -- the number this whole exercise exists to measure, because a
false exclude is unrecoverable -- would be estimated from a denominator of two
or three. Neither number would mean anything.

So the draw is stratified: 25 from the top third of the pool's ranked list and
25 from the bottom two-thirds. This oversamples the head, where the includes
are, and the resulting kappa is therefore a kappa ON THE STRATIFIED SAMPLE, not
an estimate of kappa over the corpus. That distinction is recorded in provenance
rather than left for a reader to infer, and stratum is carried on every row so
any later analysis can reweight.

The pool, and the order of operations
-------------------------------------
v2.0.0 (2026-10-05 scope change): the pool is rows that are BOTH eligible under
the access rule (`eligible=True` in corpus/eligibility.csv, joined on
openalex_id; missing or stale stops the draw) AND carry abstract text
(`abstract_source` openalex or semantic_scholar, non-empty `abstract`). The
second condition is stated on the abstract itself, not as
`abstract_source != none`: a full-text-head row is not `none` either, and it
has no abstract to put on the sheet.

Both filters are applied BEFORE the thirds are cut, not after. The reasoning was that the OpenAlex abstract gap tracks Elsevier,
Elsevier tracks certain venues, and venue plausibly correlates with rank -- so
cutting thirds first and filtering second would leave the two strata holding
different numbers of papers for a reason having nothing to do with rank.

That was an assumption, so the script MEASURES it rather than asserting it:
per-third pool-exclusion rates, split by cause (ineligible / eligible but no
abstract), go into provenance every run, alongside both sets of rank
boundaries.

On the 2026-09-28 corpus the assumption DID NOT HOLD for the abstract gap --
22.7% missing in the full-list top third against 21.9% in the bottom
two-thirds, which is flat. The ordering is kept because it is the one that
stays correct if a corpus does skew; the eligibility filter is a new and
different cause, so read the provenance block, not this docstring, for what the
current corpus does.

The v1 draw (abstract-only pool, before the access rule) is archived under
screening/archive/v1/ and is not overwritten.

Blinding
--------
`screening/labels_TODO.csv` carries paper_id, title and abstract and nothing
else. No stratum, no rank, no seed links, no score, no framework_core flag --
every one of those is a signal about how the pipeline already ranked the paper,
and a human who can see it is no longer an independent rater. The kappa would
measure agreement with the ranking rather than agreement about the criteria.
The 50 rows are shuffled before writing so that position carries no information
either.

Usage
-----
    python draw_calibration_sample.py
    python draw_calibration_sample.py --rng-seed 20260928 --n-per-stratum 25
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

from eligibility import attach_eligibility, parse_eligible

SCRIPT_VERSION = "2.0.0"

SAMPLE_COLUMNS = ["paper_id", "doi", "title", "abstract", "year", "venue",
                  "abstract_source", "stratum"]

# The full-text-head sample. Its text column is named for what it holds, not
# `abstract`, in both the sample and the blind sheet. A rater who thinks they
# are reading an abstract will read an absent result as a result the paper does
# not report; a rater who knows they are reading the first 5,000 characters
# will not. The same reasoning applies to the model (see screen.py TEXT_KINDS),
# and the two have to agree or the kappa measures the mismatch.
FULLTEXT_SAMPLE_COLUMNS = ["paper_id", "doi", "title", "fulltext_head", "year",
                           "venue", "abstract_source"]
FULLTEXT_LABEL_COLUMNS = ["paper_id", "title", "fulltext_head",
                          "human_decision", "human_criterion_id", "human_notes"]

FULLTEXT_SOURCES = ("fulltext_head_oa",)
ABSTRACT_SOURCES = ("openalex", "semantic_scholar")

# The blind sheet. `human_criterion_id` takes an ID from screening/criteria.md,
# which Zachary writes -- this script deliberately does not know what the valid
# IDs are, because inventing them here would be authoring the criteria.
LABEL_COLUMNS = ["paper_id", "title", "abstract",
                 "human_decision", "human_criterion_id", "human_notes"]

DECISIONS = ("include", "exclude", "uncertain")

csv.field_size_limit(10_000_000)


def read_candidates(path: str, eligibility_path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows")
    if "abstract_source" not in rows[0]:
        sys.exit(f"{path} has no abstract_source column -- run abstract_backfill.py first")
    # corpus/eligibility.csv, joined on openalex_id and checked against this
    # candidates.csv; exits loudly if missing or stale.
    attach_eligibility(eligibility_path, rows)
    return rows


def has_abstract_text(r: dict) -> bool:
    return ((r.get("abstract_source") or "") in ABSTRACT_SOURCES
            and bool((r.get("abstract") or "").strip()))


def draw_fulltext(rows: list[dict], args) -> int:
    """
    The separate full-text-head calibration sample.

    Kept apart from the 50-paper abstract sample on purpose. A truncated
    full-text head is a different input distribution: it opens with title-page
    matter and an introduction rather than a structured summary, it says
    nothing about results, and it is cut off mid-sentence. Pooling the two
    would produce one kappa describing neither, and would hide the possibility
    that a model is reliable on abstracts and not on heads -- which is exactly
    the thing worth knowing before the head pipeline is trusted.

    Unstratified, per the brief: a plain random 20 at the fixed seed. Worth
    saying plainly that 20 is a small denominator -- if the includes among them
    number two or three, include-recall is being estimated from that, and the
    same objection that motivated stratifying the 50 applies here with less
    room. The script reports the drawn size and the eligible pool so the
    denominator is visible rather than implied; whether to stratify or to widen
    is a judgement for the run that has the labels in hand.
    """
    # Eligible rows only, as for the abstract draw. The 2026-09-28 full-text
    # sample predates the access rule and is left as it was drawn; this filter
    # governs any future redraw.
    eligible = [r for r in rows
                if parse_eligible(r.get("eligible", ""))
                and (r.get("abstract_source") or "") in FULLTEXT_SOURCES
                and (r.get("fulltext_head") or "").strip()]

    if not args.quiet:
        print(f"[draw-ft] {len(rows)} candidates, {len(eligible)} with a "
              f"full-text head")
    if len(eligible) < args.n_fulltext:
        sys.exit(f"only {len(eligible)} rows carry a full-text head; need "
                 f"{args.n_fulltext}. Run fulltext_head.py first, or lower "
                 f"--n-fulltext.")

    rng = random.Random(args.rng_seed)
    pool = sorted(eligible, key=lambda r: r["openalex_id"])
    sample = rng.sample(pool, args.n_fulltext)
    rng.shuffle(sample)

    # If the draw takes the whole pool, no selection happened. Saying "random
    # sample, seed 20260928" would imply one did, and would leave a reader
    # thinking the 20 generalise to some larger set of head rows. They are the
    # set.
    census = len(sample) == len(eligible)

    os.makedirs(args.out_dir, exist_ok=True)
    sample_path = os.path.join(args.out_dir, "calibration_sample_fulltext.csv")
    labels_path = os.path.join(args.out_dir, "labels_fulltext_TODO.csv")

    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FULLTEXT_SAMPLE_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({
                "paper_id": r["openalex_id"], "doi": r.get("doi", ""),
                "title": r.get("title", ""),
                "fulltext_head": r.get("fulltext_head", ""),
                "year": r.get("year", ""), "venue": r.get("venue", ""),
                "abstract_source": r.get("abstract_source", ""),
            })

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FULLTEXT_LABEL_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({
                "paper_id": r["openalex_id"], "title": r.get("title", ""),
                "fulltext_head": r.get("fulltext_head", ""),
                "human_decision": "", "human_criterion_id": "", "human_notes": "",
            })

    by_source = Counter(r["abstract_source"] for r in sample)
    block = {
        "script": "draw_calibration_sample.py --fulltext",
        "script_version": SCRIPT_VERSION,
        "drawn_utc": datetime.now(timezone.utc).isoformat(),
        "rng_seed": args.rng_seed,
        "rng": "python random.Random(rng_seed); pool sorted by paper_id, then "
               "random.sample(), then shuffle()",
        "stratified": False,
        "census": census,
        "census_note": (
            "The draw took the ENTIRE eligible pool, so no sampling occurred "
            "and rng_seed selected nothing -- these 20 are every full-text-head "
            "row that exists, not a sample of them. Metrics computed here "
            "describe this set exactly and generalise to nothing wider."
            if census else
            "A random subset of the eligible pool at the fixed seed."),
        "source_file": args.candidates,
        "eligible_pool": len(eligible),
        "sample_size": len(sample),
        "by_source": dict(by_source),
        "drawn_paper_ids": sorted(r["openalex_id"] for r in sample),
        "outputs": {"sample": sample_path, "labels_todo": labels_path},
        "thresholds": {"kappa_binary_min": 0.70, "include_recall_min": 0.95,
                       "basis": "same floors as the abstract sample; the input "
                                "differs, the bar does not"},
        "note": (
            "Scored SEPARATELY from the 50-paper abstract sample and never "
            "pooled with it: a truncated full-text head is a different input "
            "distribution, and one kappa over both would describe neither. The "
            "two samples cannot overlap by construction, since a row carries "
            "either an abstract or a head. Caveat on size: 20 is a small "
            "denominator for include-recall, and unlike the 50-paper draw this "
            "one is unstratified, so if includes are rare in the pool the "
            "recall estimate will be noisy. eligible_pool is recorded so that "
            "can be judged rather than assumed."),
    }

    prov = {}
    if os.path.exists(args.provenance):
        with open(args.provenance, encoding="utf-8") as f:
            prov = json.load(f)
    prov["calibration_sample_fulltext"] = block
    with open(args.provenance, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    if not args.quiet:
        print(f"[draw-ft] rng_seed={args.rng_seed}, unstratified")
        print(f"[draw-ft] drawn {len(sample)} from a pool of {len(eligible)}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
        if census:
            print(f"  *** CENSUS, NOT A SAMPLE: the draw took all "
                  f"{len(eligible)} eligible rows, so the seed selected "
                  f"nothing.\n      Metrics on these 20 describe exactly these "
                  f"20 and generalise no further.")
        print(f"  -> {sample_path}")
        print(f"  -> {labels_path}  (blind; text column named `fulltext_head`, "
              f"not `abstract`)")
        print(f"  -> {args.provenance}  (calibration_sample_fulltext block)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Draw the screening calibration samples (abstract, and "
                    "separately full-text head).")
    ap.add_argument("--candidates", default="corpus/candidates.csv")
    ap.add_argument("--eligibility", default="corpus/eligibility.csv")
    ap.add_argument("--out-dir", default="screening")
    ap.add_argument("--provenance", default="corpus/provenance.json")
    ap.add_argument("--rng-seed", type=int, default=20260928,
                    help="fixed RNG seed; recorded in provenance so the draw "
                         "is reproducible")
    ap.add_argument("--n-per-stratum", type=int, default=25)
    ap.add_argument("--fulltext", action="store_true",
                    help="draw the separate full-text-head sample instead of "
                         "the stratified abstract sample")
    ap.add_argument("--n-fulltext", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    rows = read_candidates(args.candidates, args.eligibility)

    if args.fulltext:
        return draw_fulltext(rows, args)

    # candidates.csv is written in rank order by corpus_build.py, so position is
    # rank. Rank is captured here as an explicit 1-based column rather than
    # relied on implicitly, because it has to survive the filter below and still
    # mean "position in the FULL ranked list" in the provenance record.
    for i, r in enumerate(rows, 1):
        r["_full_rank"] = i

    in_scope = [r for r in rows if parse_eligible(r.get("eligible", ""))]
    eligible = [r for r in in_scope if has_abstract_text(r)]
    excluded_ineligible = len(rows) - len(in_scope)
    excluded = len(in_scope) - len(eligible)

    if verbose:
        print(f"[draw] {len(rows)} candidates, {len(in_scope)} eligible, "
              f"{len(eligible)} eligible with abstract text; "
              f"{excluded_ineligible} out of scope, {excluded} eligible but "
              f"no abstract")

    need = 2 * args.n_per_stratum
    if len(eligible) < need:
        sys.exit(f"only {len(eligible)} eligible rows; need at least {need}")

    # Thirds are cut on the POOL (eligible + abstract) -- see the docstring.
    cut = len(eligible) // 3
    top = eligible[:cut]
    bottom = eligible[cut:]

    if len(top) < args.n_per_stratum or len(bottom) < args.n_per_stratum:
        sys.exit(f"stratum too small: top={len(top)} bottom={len(bottom)}, "
                 f"need {args.n_per_stratum} each")

    rng = random.Random(args.rng_seed)
    # Two independent draws from one seeded Random. Sorting each stratum by
    # paper_id first makes the draw independent of any incidental ordering
    # within a stratum, so the seed alone reproduces it.
    top_sorted = sorted(top, key=lambda r: r["openalex_id"])
    bottom_sorted = sorted(bottom, key=lambda r: r["openalex_id"])
    drawn_top = rng.sample(top_sorted, args.n_per_stratum)
    drawn_bottom = rng.sample(bottom_sorted, args.n_per_stratum)

    for r in drawn_top:
        r["stratum"] = "top_third"
    for r in drawn_bottom:
        r["stratum"] = "bottom_two_thirds"

    sample = drawn_top + drawn_bottom
    rng.shuffle(sample)  # order must not leak the stratum

    os.makedirs(args.out_dir, exist_ok=True)
    sample_path = os.path.join(args.out_dir, "calibration_sample.csv")
    labels_path = os.path.join(args.out_dir, "labels_TODO.csv")

    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({
                "paper_id": r["openalex_id"],
                "doi": r.get("doi", ""),
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
                "year": r.get("year", ""),
                "venue": r.get("venue", ""),
                "abstract_source": r.get("abstract_source", ""),
                "stratum": r["stratum"],
            })

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({
                "paper_id": r["openalex_id"],
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
                "human_decision": "",
                "human_criterion_id": "",
                "human_notes": "",
            })

    # --- what the filter-then-cut ordering actually cost -------------------
    # Measured, not assumed: if pool exclusion is flat across rank these rates
    # match and the ordering was immaterial. Split by cause, because the access
    # rule and the abstract gap are different mechanisms.
    full_cut = len(rows) // 3

    def gap(part: list[dict]) -> dict:
        inel = sum(1 for r in part if not parse_eligible(r["eligible"]))
        noab = sum(1 for r in part
                   if parse_eligible(r["eligible"]) and not has_abstract_text(r))
        return {"size": len(part), "ineligible": inel,
                "eligible_no_abstract": noab, "not_in_pool": inel + noab,
                "pct_not_in_pool": round(100 * (inel + noab) / max(len(part), 1), 1)}

    block = {
        "script": "draw_calibration_sample.py",
        "script_version": SCRIPT_VERSION,
        "drawn_utc": datetime.now(timezone.utc).isoformat(),
        "rng_seed": args.rng_seed,
        "rng": "python random.Random(rng_seed); strata sorted by paper_id, "
               "then random.sample() per stratum, then shuffle() of the 50",
        "source_file": args.candidates,
        "candidates_total": len(rows),
        "eligibility_source": args.eligibility,
        "pool_definition": ("eligible == True (eligibility.csv) AND "
                            "abstract_source in openalex|semantic_scholar AND "
                            "abstract non-empty"),
        "eligible_total": len(in_scope),
        "excluded_ineligible": excluded_ineligible,
        "eligible_with_abstract": len(eligible),
        "excluded_no_abstract": excluded,
        "excluded_no_abstract_basis": "eligible rows only; ineligible rows are "
                                      "counted under excluded_ineligible",
        "n_per_stratum": args.n_per_stratum,
        "sample_size": len(sample),
        "strata": {
            "top_third": {
                "eligible_rank_range": [1, cut],
                "eligible_size": len(top),
                "drawn": args.n_per_stratum,
            },
            "bottom_two_thirds": {
                "eligible_rank_range": [cut + 1, len(eligible)],
                "eligible_size": len(bottom),
                "drawn": args.n_per_stratum,
            },
        },
        "pool_exclusion_by_full_list_third": {
            "full_list_cut_at_rank": full_cut,
            "top_third": gap(rows[:full_cut]),
            "bottom_two_thirds": gap(rows[full_cut:]),
            "basis": (
                "Thirds are cut on the POOL, so these rates say how much that "
                "choice mattered. Equal rates mean cutting before or after the "
                "filters was the same decision; divergent rates mean pool "
                "exclusion is rank-correlated and the pool cut sits at a "
                "different paper than the full-list cut would."
            ),
        },
        "drawn_paper_ids": {
            "top_third": sorted(r["openalex_id"] for r in drawn_top),
            "bottom_two_thirds": sorted(r["openalex_id"] for r in drawn_bottom),
        },
        "outputs": {"sample": sample_path, "labels_todo": labels_path},
        "note": (
            "kappa computed on this sample is a kappa ON A STRATIFIED SAMPLE. It "
            "is not an estimate of kappa over the full ranked corpus, because the "
            "top third is deliberately oversampled 2:1 to put enough includes in "
            "the denominator for include-recall to mean anything. `stratum` is "
            "carried on every sample row so a later analysis can reweight. "
            "labels_TODO.csv omits stratum, rank and every seed-linkage column: a "
            "rater who can see how the pipeline ranked a paper is not an "
            "independent rater, and the resulting kappa would measure agreement "
            "with the ranking instead of agreement about the criteria."
        ),
    }

    prov = {}
    if os.path.exists(args.provenance):
        with open(args.provenance, encoding="utf-8") as f:
            prov = json.load(f)
    prov["calibration_sample"] = block
    with open(args.provenance, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    if verbose:
        by_stratum = Counter(r["stratum"] for r in sample)
        by_source = Counter(r["abstract_source"] for r in sample)
        print(f"[draw] rng_seed={args.rng_seed}")
        print(f"[draw] pool thirds: top=ranks 1-{cut} ({len(top)}), "
              f"bottom=ranks {cut + 1}-{len(eligible)} ({len(bottom)})")
        print(f"[draw] drawn: " + ", ".join(f"{k}={v}" for k, v in sorted(by_stratum.items())))
        print(f"[draw] abstract_source in sample: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
        g = block["pool_exclusion_by_full_list_third"]
        print(f"[draw] not in pool by full-list third: "
              f"top={g['top_third']['pct_not_in_pool']}%, "
              f"bottom={g['bottom_two_thirds']['pct_not_in_pool']}%")
        print(f"  -> {sample_path}")
        print(f"  -> {labels_path}  (blind: no stratum, no rank, no seed links)")
        print(f"  -> {args.provenance}  (calibration_sample block)")
        print()
        print("Next: write screening/criteria.md, then fill in the three human_* "
              "columns of labels_TODO.csv and save it as screening/labels.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
