#!/usr/bin/env python3
"""
benchmark.py -- score screening models against the human calibration labels.

Python counts (proposal S3.3). Every number here is computed with pandas from
two files on disk: the human labels, and the call log screen.py wrote. No model
is asked to evaluate anything, including itself.

The numbers, and why these ones
-------------------------------
**Binary kappa is primary.** `uncertain` routes a paper to a human, which is
operationally the same outcome as `include`: the paper survives to the next
stage. So the binary collapse folds uncertain into include on BOTH sides and
asks the only question the pipeline cares about -- does this paper survive
screening. The 3-class kappa is reported alongside it because it says something
different and useful: whether the model uses `uncertain` the way the criteria
say to, rather than as a hedge.

**Include recall is the number to look at first.** Screening is the only stage
that removes papers on model output, and a false exclude is never seen again --
there is no later step at which a wrongly excluded paper can be recovered. A
false include costs one extraction. The two errors are not symmetric and the
metrics should not pretend they are.

**Criterion-ID agreement is scored only where the decisions already agree.**
Comparing criterion IDs across a disagreement would conflate two failures: the
model reached a different conclusion, and it cited a different rule. Restricting
to agreeing rows isolates "right answer, right reason" from "right answer,
wrong reason" -- the second is the one that says the criteria are ambiguous.

Refusal to score a partial sheet
--------------------------------
If any row of labels.csv is missing a decision, this script exits without
computing anything. A kappa over whichever rows happened to be filled in is a
kappa over a sample selected by how easy the papers were to label, and easy
papers agree. That is not a weaker number, it is a misleading one.

Selection rule
--------------
Cheapest model with binary kappa >= 0.7 AND include recall >= 0.95.

One caveat the rule cannot express: cost is not like-for-like across models
when one of them cannot be run without thinking. Claude Opus 5.5 always thinks
and bills those tokens as output, so it carries a cost component Haiku 4.5 and
Sonnet 5.5 do not. `thinking_unavoidable` is carried per model from the run
manifest and printed with the table; the rule is still applied exactly as
specified, but the comparison should be read with that in view. If none
qualifies, that is reported and NO model is chosen -- the fix is to revise
criteria.md and rerun, and disagreements.csv is the input for that revision.

Usage
-----
    python benchmark.py --run screening/runs/<run_id> --labels screening/labels.csv
    python benchmark.py --run ... --labels ... --repeat-consistency
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from collections import Counter

SCRIPT_VERSION = "1.0.0"

DECISIONS = ["include", "exclude", "uncertain"]
BINARY = ["include", "exclude"]

KAPPA_FLOOR = 0.70
RECALL_FLOOR = 0.95


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def cohen_kappa(a: pd.Series, b: pd.Series, labels: list[str]) -> float:
    """
    Cohen's kappa, computed from the confusion matrix rather than imported.

    Returns NaN when chance agreement is 1.0 -- which happens when both raters
    used exactly one class for everything. Kappa is genuinely undefined there
    (the denominator is zero); returning 0.0 would read as "no better than
    chance" when the truth is "this sample cannot distinguish the two".
    """
    if len(a) == 0:
        return float("nan")
    cm = confusion(a, b, labels)
    n = cm.to_numpy().sum()
    if n == 0:
        return float("nan")
    po = sum(cm.loc[c, c] for c in labels) / n
    pe = sum((cm.loc[c, :].sum() / n) * (cm.loc[:, c].sum() / n) for c in labels)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def confusion(human: pd.Series, model: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Rows = human, columns = model."""
    cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for h, m in zip(human, model):
        if h in labels and m in labels:
            cm.loc[h, m] += 1
    return cm


def to_binary(s: pd.Series) -> pd.Series:
    """uncertain collapses to include: both keep the paper in the pipeline."""
    return s.replace({"uncertain": "include"})


def recall_precision(human_b: pd.Series, model_b: pd.Series) -> tuple[float, float, int, int, int]:
    tp = int(((human_b == "include") & (model_b == "include")).sum())
    fn = int(((human_b == "include") & (model_b == "exclude")).sum())
    fp = int(((human_b == "exclude") & (model_b == "include")).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return recall, precision, tp, fn, fp


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"labels file not found: {path}\n"
                 f"Fill in screening/labels_TODO.csv and save it as {path}.")
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("paper_id", "human_decision"):
        if col not in df.columns:
            sys.exit(f"{path} has no {col!r} column")

    df["human_decision"] = df["human_decision"].str.strip().str.lower()

    blank = df[df["human_decision"] == ""]
    bad = df[(df["human_decision"] != "")
             & (~df["human_decision"].isin(DECISIONS))]

    if len(blank) or len(bad):
        print(f"\n  *** REFUSING TO COMPUTE METRICS.\n")
        if len(blank):
            print(f"  {len(blank)} of {len(df)} rows have no human_decision:")
            for pid in blank["paper_id"].head(10):
                print(f"    {pid}")
            if len(blank) > 10:
                print(f"    ... and {len(blank) - 10} more")
        if len(bad):
            print(f"  {len(bad)} row(s) have a value outside "
                  f"{DECISIONS}:")
            for _, r in bad.head(10).iterrows():
                print(f"    {r['paper_id']}: {r['human_decision']!r}")
        print(f"\n  A kappa over a partially-filled sheet is a kappa over the "
              f"papers that were\n  easy enough to label first, and easy papers "
              f"agree. Fill every row.\n")
        sys.exit(3)

    if "human_criterion_id" not in df.columns:
        df["human_criterion_id"] = ""
    df["human_criterion_id"] = df["human_criterion_id"].str.strip()
    return df


def load_manifest(run_dir: str) -> dict:
    """
    The run manifest, if screen.py wrote one. Used only to carry forward facts
    about the REQUEST SHAPE that the call log does not hold -- chiefly which
    models could not have thinking switched off.
    """
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_calls(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "calls.jsonl")
    if not os.path.exists(path):
        sys.exit(f"call log not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit(f"{path} is empty")
    df = pd.DataFrame(rows)
    parsed = df["parsed"].apply(lambda d: d if isinstance(d, dict) else {})
    df["model_decision"] = parsed.apply(
        lambda d: str(d.get("decision", "")).strip().lower())
    df["model_criterion_id"] = parsed.apply(
        lambda d: str(d.get("criterion_id", "")).strip())
    df["model_reason"] = parsed.apply(lambda d: str(d.get("reason", "")).strip())
    if "repeat" not in df.columns:
        df["repeat"] = 1
    if "text_kind" not in df.columns:
        df["text_kind"] = "abstract"
    df["text_kind"] = df["text_kind"].fillna("abstract")
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score screening models against the human calibration labels.")
    ap.add_argument("--run", required=True, help="screening/runs/<run_id>")
    ap.add_argument("--labels", default="screening/labels.csv")
    ap.add_argument("--score-repeat", type=int, default=1,
                    help="which repeat supplies the headline metrics (default 1)")
    ap.add_argument("--repeat-consistency", action="store_true",
                    help="also report each model's agreement with itself across "
                         "repeats")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    calls = load_calls(args.run)
    manifest = load_manifest(args.run)
    shapes = manifest.get("per_model_request_shape", {}) or {}

    run_id = str(calls["run_id"].iloc[0]) if "run_id" in calls else os.path.basename(args.run)
    print(f"[bench] run {run_id}")
    print(f"[bench] {len(labels)} human labels, all filled")
    print(f"[bench] {len(calls)} logged calls across "
          f"{calls['model'].nunique()} model(s)")

    # Cost and tokens come from EVERY logged call, including the invalid ones.
    # An invalid response was still paid for, and a model that is cheap because
    # a third of its answers were unusable is not cheap.
    spend = calls.groupby("model").agg(
        calls_logged=("paper_id", "size"),
        input_tokens=("input_tokens", "sum"),
        cache_read_tokens=("cache_read_input_tokens", "sum"),
        cache_write_tokens=("cache_creation_input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
    )

    scored = calls[calls["repeat"] == args.score_repeat]
    rows = []
    disagreements = []

    for model, g in scored.groupby("model"):
        alias = g["model_alias"].iloc[0] if "model_alias" in g else model
        # `forced` rows are title-only papers that screen.py set to
        # `uncertain` in Python without a model call. They are scored: they are
        # decisions the pipeline actually makes, and excluding them would
        # measure a pipeline that does not exist. They are also counted
        # separately, because a kappa propped up by forced agreement is not
        # evidence about the model.
        valid = g[g["validation_status"].isin(["valid", "forced"])]
        merged = labels.merge(
            valid[["paper_id", "model_decision", "model_criterion_id",
                   "model_reason"]],
            on="paper_id", how="inner")

        n_scored = len(merged)
        coverage = n_scored / len(labels) if len(labels) else float("nan")

        h3 = merged["human_decision"]
        m3 = merged["model_decision"]
        hb, mb = to_binary(h3), to_binary(m3)

        k3 = cohen_kappa(h3, m3, DECISIONS)
        kb = cohen_kappa(hb, mb, BINARY)
        recall, precision, tp, fn, fp = recall_precision(hb, mb)

        cm3 = confusion(h3, m3, DECISIONS)

        agree = merged[h3 == m3]
        has_h_cid = agree[agree["human_criterion_id"] != ""]
        cid_agree = (
            (has_h_cid["human_criterion_id"] == has_h_cid["model_criterion_id"]).mean()
            if len(has_h_cid) else float("nan"))

        s = spend.loc[model] if model in spend.index else None
        cost = float(s["cost_usd"]) if s is not None else float("nan")

        rows.append({
            "model_alias": alias,
            "model": model,
            "n_labels": len(labels),
            "n_scored": n_scored,
            "n_invalid_responses": int(
                (~g["validation_status"].isin(["valid", "forced"])).sum()),
            "n_forced_uncertain": int((g["validation_status"] == "forced").sum()),
            "text_kinds": "|".join(
                f"{k}:{v}" for k, v in sorted(Counter(g["text_kind"]).items())),
            "coverage": round(coverage, 4),
            "kappa_binary_primary": round(kb, 4) if kb == kb else None,
            "kappa_3class": round(k3, 4) if k3 == k3 else None,
            "include_recall_binary": round(recall, 4) if recall == recall else None,
            "include_precision_binary": round(precision, 4) if precision == precision else None,
            "uncertain_rate_model": round((m3 == "uncertain").mean(), 4) if n_scored else None,
            "uncertain_rate_human": round((h3 == "uncertain").mean(), 4) if n_scored else None,
            "true_include": tp, "false_exclude": fn, "false_include": fp,
            "criterion_id_agreement_on_agreeing_rows": (
                round(float(cid_agree), 4) if cid_agree == cid_agree else None),
            "criterion_id_agreement_n": len(has_h_cid),
            **{f"cm_human_{h}_model_{m}": int(cm3.loc[h, m])
               for h in DECISIONS for m in DECISIONS},
            "input_tokens": int(s["input_tokens"]) if s is not None else 0,
            "cache_read_tokens": int(s["cache_read_tokens"]) if s is not None else 0,
            "cache_write_tokens": int(s["cache_write_tokens"]) if s is not None else 0,
            "output_tokens": int(s["output_tokens"]) if s is not None else 0,
            "cost_usd": round(cost, 6),
            "meets_kappa_floor": bool(kb == kb and kb >= KAPPA_FLOOR),
            "meets_recall_floor": bool(recall == recall and recall >= RECALL_FLOOR),
            # Carried from the run manifest, not inferred. A model that cannot
            # run without thinking pays for thinking tokens as output tokens,
            # so its cost is not like-for-like with a model that can.
            "thinking_unavoidable": bool(
                (shapes.get(model) or {}).get("thinking_unavoidable", False)),
        })

        for _, r in merged[h3 != m3].iterrows():
            disagreements.append({
                "model_alias": alias, "model": model,
                "paper_id": r["paper_id"],
                "title": r.get("title", ""),
                "human_decision": r["human_decision"],
                "model_decision": r["model_decision"],
                "human_criterion_id": r["human_criterion_id"],
                "model_criterion_id": r["model_criterion_id"],
                "model_reason": r["model_reason"],
                "direction": (
                    "FALSE_EXCLUDE" if to_binary(pd.Series([r["human_decision"]]))[0] == "include"
                    and to_binary(pd.Series([r["model_decision"]]))[0] == "exclude"
                    else "false_include" if to_binary(pd.Series([r["human_decision"]]))[0] == "exclude"
                    and to_binary(pd.Series([r["model_decision"]]))[0] == "include"
                    else "uncertain_boundary"),
            })

    # --- metrics broken out by input kind ---------------------------------
    # An abstract and a truncated full-text head are different inputs, and the
    # brief calls for them to be reported separately. Normally each sample gets
    # its own run, in which case this table has one row per model and simply
    # confirms that; if a run ever mixes them, this is where the difference
    # shows up rather than averaging away.
    by_kind_rows = []
    for (model, kind), g in scored.groupby(["model", "text_kind"]):
        valid = g[g["validation_status"].isin(["valid", "forced"])]
        m = labels.merge(valid[["paper_id", "model_decision"]],
                         on="paper_id", how="inner")
        if not len(m):
            continue
        h3, m3 = m["human_decision"], m["model_decision"]
        hb, mb = to_binary(h3), to_binary(m3)
        kb = cohen_kappa(hb, mb, BINARY)
        rec, prec, tp, fn, fp = recall_precision(hb, mb)
        by_kind_rows.append({
            "model": model,
            "model_alias": g["model_alias"].iloc[0] if "model_alias" in g else model,
            "text_kind": kind,
            "n_scored": len(m),
            "kappa_binary": round(kb, 4) if kb == kb else None,
            "include_recall_binary": round(rec, 4) if rec == rec else None,
            "include_precision_binary": round(prec, 4) if prec == prec else None,
            "uncertain_rate_model": round((m3 == "uncertain").mean(), 4),
            "true_include": tp, "false_exclude": fn, "false_include": fp,
            "meets_kappa_floor": bool(kb == kb and kb >= KAPPA_FLOOR),
            "meets_recall_floor": bool(rec == rec and rec >= RECALL_FLOOR),
        })
    by_kind = pd.DataFrame(by_kind_rows)

    bench = pd.DataFrame(rows).sort_values("cost_usd").reset_index(drop=True)
    dis = pd.DataFrame(disagreements)

    if len(by_kind):
        by_kind.to_csv(os.path.join(args.run, "benchmark_by_text_kind.csv"),
                       index=False)

    bench_path = os.path.join(args.run, "benchmark.csv")
    dis_path = os.path.join(args.run, "disagreements.csv")
    bench.to_csv(bench_path, index=False)
    if len(dis):
        dis = dis.sort_values(["direction", "model_alias", "paper_id"])
    dis.to_csv(dis_path, index=False)

    # --- self-consistency --------------------------------------------------
    consistency = {}
    if args.repeat_consistency:
        reps = sorted(calls["repeat"].unique())
        if len(reps) < 2:
            print("[bench] --repeat-consistency asked for, but only one repeat "
                  "is logged; skipping")
        else:
            a_rep, b_rep = reps[0], reps[1]
            for model, g in calls.groupby("model"):
                va = g[(g["repeat"] == a_rep) & (g["validation_status"] == "valid")]
                vb = g[(g["repeat"] == b_rep) & (g["validation_status"] == "valid")]
                j = va.merge(vb, on="paper_id", suffixes=("_a", "_b"))
                if not len(j):
                    continue
                consistency[model] = {
                    "n_paired": len(j),
                    "exact_decision_agreement": round(
                        float((j["model_decision_a"] == j["model_decision_b"]).mean()), 4),
                    "self_kappa_3class": round(
                        cohen_kappa(j["model_decision_a"], j["model_decision_b"],
                                    DECISIONS), 4),
                    "criterion_id_agreement": round(
                        float((j["model_criterion_id_a"]
                               == j["model_criterion_id_b"]).mean()), 4),
                    "basis": (
                        f"repeat {a_rep} vs {b_rep}, identical requests. "
                        f"temperature/top_p are not settable on these models, so "
                        f"this measures residual nondeterminism, not sampling "
                        f"spread -- a low number here means the decision was "
                        f"marginal, not that the harness varied anything."),
                }

    # --- selection ---------------------------------------------------------
    qualified = bench[bench["meets_kappa_floor"] & bench["meets_recall_floor"]]
    selection = {
        "rule": f"cheapest model with binary kappa >= {KAPPA_FLOOR} "
                f"AND include recall >= {RECALL_FLOOR}",
        "kappa_floor": KAPPA_FLOOR,
        "recall_floor": RECALL_FLOOR,
    }
    if len(qualified):
        pick = qualified.iloc[0]
        selection.update({
            "selected": True,
            "model": pick["model"],
            "model_alias": pick["model_alias"],
            "kappa_binary": pick["kappa_binary_primary"],
            "include_recall": pick["include_recall_binary"],
            "cost_usd": pick["cost_usd"],
        })
    else:
        selection.update({
            "selected": False,
            "model": None,
            "why": "no model cleared both floors",
            "next_step": "revise screening/criteria.md using disagreements.csv, "
                         "then rerun screen.py and benchmark.py",
        })

    summary = {
        "script": "benchmark.py",
        "script_version": SCRIPT_VERSION,
        "run": args.run,
        "run_id": run_id,
        "labels": args.labels,
        "n_labels": len(labels),
        "score_repeat": args.score_repeat,
        "selection": selection,
        "models_thinking_unavoidable": sorted(
            bench[bench["thinking_unavoidable"]]["model"].tolist()),
        "cost_comparability_note": (
            "cost_usd is computed from measured usage. Where "
            "thinking_unavoidable is true the model's thinking tokens are "
            "billed as output tokens, so cost across models is not "
            "like-for-like; the selection rule nevertheless picks on cost, as "
            "specified."),
        "per_text_kind": by_kind.to_dict(orient="records") if len(by_kind) else [],
        "self_consistency": consistency,
        "per_model": bench.to_dict(orient="records"),
    }
    with open(os.path.join(args.run, "benchmark_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # --- report ------------------------------------------------------------
    print()
    cols = ["model_alias", "cost_usd", "kappa_binary_primary", "kappa_3class",
            "include_recall_binary", "include_precision_binary",
            "uncertain_rate_model", "coverage", "n_invalid_responses"]
    print(bench[cols].to_string(index=False))

    if len(by_kind) and by_kind["text_kind"].nunique() > 1:
        print("\n[by input kind] an abstract and a full-text head are different "
              "inputs; they are not pooled")
        print(by_kind[["model_alias", "text_kind", "n_scored", "kappa_binary",
                       "include_recall_binary", "uncertain_rate_model"]]
              .to_string(index=False))

    forced = bench[bench["n_forced_uncertain"] > 0]
    if len(forced):
        n = int(forced["n_forced_uncertain"].iloc[0])
        print(f"\n  *** {n} row(s) per model were title-only and forced to "
              f"`uncertain` in Python\n      with no model call. They are "
              f"included above because they are decisions the\n      pipeline "
              f"really makes -- but agreement on them is agreement with a rule, "
              f"not\n      with the model.")

    thinkers = bench[bench["thinking_unavoidable"]]
    if len(thinkers):
        names = ", ".join(thinkers["model_alias"])
        print(f"\n  *** {names}: thinking could not be disabled on this run. "
              f"Thinking tokens bill\n      as output tokens, so its cost_usd "
              f"includes a component the other models\n      do not pay. The "
              f"selection rule picks on cost -- read the comparison with\n"
              f"      that asymmetry in view, not as like-for-like.")

    for _, r in bench.iterrows():
        if r["coverage"] < 1.0:
            print(f"\n  *** {r['model_alias']}: {r['n_invalid_responses']} "
                  f"response(s) failed validation, so its kappa is computed over "
                  f"{r['n_scored']}/{r['n_labels']} papers. Its cost is over all "
                  f"calls. Compare it to the others with that in mind.")

    if consistency:
        print("\n[self-consistency]")
        for m, c in consistency.items():
            print(f"  {m}: {c['exact_decision_agreement']:.3f} exact, "
                  f"self-kappa {c['self_kappa_3class']}, n={c['n_paired']}")

    print()
    if selection["selected"]:
        print(f"[select] {selection['model_alias']} ({selection['model']}): "
              f"binary kappa {selection['kappa_binary']}, "
              f"include recall {selection['include_recall']}, "
              f"${selection['cost_usd']:.4f}")
    else:
        print(f"[select] NO MODEL QUALIFIES.")
        print(f"         Floors: binary kappa >= {KAPPA_FLOOR}, "
              f"include recall >= {RECALL_FLOOR}.")
        print(f"         Not picking the closest one. The fix is to revise "
              f"screening/criteria.md")
        print(f"         using {dis_path} ({len(dis)} disagreement rows), then "
              f"rerun.")

    print()
    print(f"  -> {bench_path}")
    print(f"  -> {dis_path}")
    print(f"  -> {os.path.join(args.run, 'benchmark_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
