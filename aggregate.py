#!/usr/bin/env python3
"""
aggregate.py -- every number in the report, computed in pandas from the records.

No model is involved (proposal S3.3). Input: extraction/records/<paper_id>.json,
written by extract.py. Output: extraction/aggregate/ -- one CSV per table and
aggregates.json, which the report cites.

Which extracted values are counted
----------------------------------
A field enters a table only if its evidence check passed or is a legitimate
non-answer:
    counted   ok, ok_loose, page_mismatch (the quote IS in the paper, on another
              page), not_reported, absence_unquoted
    excluded  quote_not_found, quote_too_long, missing_quote, bad_value,
              missing_field -- the value is not backed by checkable evidence,
              so it is treated as missing and counted in the evidence table
Every table carries the n it was computed over.

DT conformance score (0-7): one point per clause of the reference definition
[20] (proposal S3.1) that the extraction shows satisfied:
    replica           conf_replica_of_physical_twin == yes
    synchronized      conf_synchronized in {continuous_realtime, periodic_batch}
    purpose           conf_purpose_stated in {yes_specific, yes_generic}
    models            conf_incorporates_models has any value but not_reported
    expertise         conf_subject_matter_expertise == yes_explicit
    historical data   conf_data_historical == yes
    real-time data    conf_data_realtime == yes
`manual_or_on_demand` sync and `implicit` expertise are deliberately NOT counted
(the strict reading); an excluded field scores 0 for its clause and is flagged.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter

import pandas as pd

RECORDS = "extraction/records"
OUT = "extraction/aggregate"

COUNTED = {"ok", "ok_loose", "page_mismatch", "not_reported", "absence_unquoted"}
VERIFIED = {"ok", "ok_loose", "page_mismatch"}

CONFORMANCE = {
    "replica": ("DT_A_DEFINITION_CONFORMANCE__conf_replica_of_physical_twin", {"yes"}),
    "synchronized": ("DT_A_DEFINITION_CONFORMANCE__conf_synchronized",
                     {"continuous_realtime", "periodic_batch"}),
    "purpose": ("DT_A_DEFINITION_CONFORMANCE__conf_purpose_stated",
                {"yes_specific", "yes_generic"}),
    "models": ("DT_A_DEFINITION_CONFORMANCE__conf_incorporates_models", "ANY"),
    "expertise": ("DT_A_DEFINITION_CONFORMANCE__conf_subject_matter_expertise", {"yes_explicit"}),
    "historical_data": ("DT_A_DEFINITION_CONFORMANCE__conf_data_historical", {"yes"}),
    "realtime_data": ("DT_A_DEFINITION_CONFORMANCE__conf_data_realtime", {"yes"}),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(papers, fields): one row per paper; one row per (paper, field)."""
    papers, fields = [], []
    for f in sorted(glob.glob(os.path.join(RECORDS, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        meta, ident = r["J_EXTRACTION_META"], r["IDENT"]
        papers.append({**ident, "schema": meta["schema"],
                       "response_status": meta["response_status"],
                       "failure": meta.get("failure"),
                       "seed_role_links": "|".join(ident.get("seed_role_links") or [])})
        for k, v in (r.get("fields") or {}).items():
            fields.append({"paper_id": ident["paper_id"], "schema": meta["schema"],
                           "field": k, "block": k.split("__")[0],
                           "value": v.get("value"), "check": v.get("check"),
                           "quote": v.get("evidence_quote"), "page": v.get("page"),
                           "confidence": v.get("confidence")})
    return pd.DataFrame(papers), pd.DataFrame(fields)


def val(F: pd.DataFrame, field: str) -> pd.Series:
    """paper_id -> value for one field, counted rows only."""
    s = F[(F["field"] == field) & F["check"].isin(COUNTED)]
    return s.set_index("paper_id")["value"]


def as_list(v) -> list:
    if isinstance(v, list):
        return v
    return [v] if v is not None else []


def freq(series: pd.Series, multi: bool = False) -> pd.DataFrame:
    vals = Counter()
    for v in series:
        for x in (as_list(v) if multi else [v]):
            vals[x] += 1
    n = len(series)
    df = pd.DataFrame([{"value": k, "papers": c, "share": round(c / n, 3) if n else None}
                       for k, c in vals.most_common()])
    df.attrs["n"] = n
    return df


def crosstab(F, a: str, b: str, multi_a=False, multi_b=False) -> pd.DataFrame:
    A, B = val(F, a), val(F, b)
    rows = []
    for pid in A.index.intersection(B.index):
        for x in (as_list(A[pid]) if multi_a else [A[pid]]):
            for y in (as_list(B[pid]) if multi_b else [B[pid]]):
                rows.append({a.split("__")[1]: x, b.split("__")[1]: y})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ct = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])
    ct.attrs["n"] = len(A.index.intersection(B.index))
    return ct


def main() -> int:
    P, F = load()
    os.makedirs(OUT, exist_ok=True)
    out: dict = {"source": RECORDS, "counted_checks": sorted(COUNTED),
                 "papers": {"total": len(P), **{k: int(v) for k, v in
                            P.groupby("schema").size().items()},
                            "valid_responses": int((P["response_status"] == "valid").sum())}}
    tables: dict[str, pd.DataFrame] = {}

    # --- evidence quality (Report 2 material) --------------------------------
    ev = F.groupby(["schema", "check"]).size().unstack(fill_value=0)
    tables["evidence_checks_by_schema"] = ev
    per_field = F.groupby(["schema", "field", "check"]).size().unstack(fill_value=0)
    tables["evidence_checks_by_field"] = per_field
    tot = len(F)
    out["evidence"] = {
        "fields_total": tot,
        "by_check": {k: int(v) for k, v in F["check"].value_counts().items()},
        "verified_quote_share_of_answered": round(
            F["check"].isin(VERIFIED).sum()
            / max((~F["check"].isin({"not_reported", "absence_unquoted"})).sum(), 1), 4),
        "quote_not_found": int((F["check"] == "quote_not_found").sum()),
        "excluded_from_tables": int((~F["check"].isin(COUNTED)).sum()),
        "human_verified_papers": 0,
        "human_verification_note": "no record has been human-verified yet; "
                                   "the per-field human-correction rate is not measured",
    }

    A = F[F["schema"] == "agent"]
    D = F[F["schema"] == "dt"]
    n_agent = int((P["schema"] == "agent").sum())

    # --- R1 S1 definitional landscape ---------------------------------------
    tables["S1_term_x_definition_source"] = crosstab(A, "A_DEFINITION__agent_term_used",
                                                      "A_DEFINITION__definition_source")
    tables["S1_definition_source"] = freq(val(A, "A_DEFINITION__definition_source"))
    tables["S1_agent_entity_types"] = freq(val(A, "A_DEFINITION__agent_entity_types"), multi=True)

    # --- R1 S2 necessary vs optional components -----------------------------
    for f in ("B_ARCHITECTURE__has_knowledge_base", "B_ARCHITECTURE__has_communication_manager"):
        tables[f"S2_{f.split('__')[1]}_x_validation"] = crosstab(
            A, f, "G_VALIDATION_METRICS__validation_level")
    tables["S2_decision_mechanism"] = freq(val(A, "B_ARCHITECTURE__decision_mechanism"), multi=True)
    tables["S2_decision_mechanism_x_validation"] = crosstab(
        A, "B_ARCHITECTURE__decision_mechanism", "G_VALIDATION_METRICS__validation_level", multi_a=True)
    tables["S2_kb_representation"] = freq(val(A, "B_ARCHITECTURE__kb_representation"))
    tables["S2_implementation_platform"] = freq(val(A, "B_ARCHITECTURE__implementation_platform"))

    # --- R1 S3 memory and communication -------------------------------------
    tables["S3_memory_historical_x_realtime"] = crosstab(A, "C_MEMORY__memory_historical",
                                                         "C_MEMORY__memory_realtime")
    tables["S3_memory_requirements_stated"] = freq(val(A, "C_MEMORY__memory_requirements_stated"))
    tables["S3_protocol_x_negotiation"] = crosstab(A, "D_COMMUNICATION__comm_protocol",
                                                   "D_COMMUNICATION__negotiation_mechanism",
                                                   multi_a=True)
    tables["S3_comm_cost_analyzed"] = freq(val(A, "D_COMMUNICATION__comm_cost_analyzed"))
    tables["S3_comm_topology"] = freq(val(A, "D_COMMUNICATION__comm_topology"))

    # --- R1 S4 learning and exploration -------------------------------------
    tables["S4_explores_x_validation"] = crosstab(A, "E_LEARNING__explores_beyond_predefined_env",
                                                  "G_VALIDATION_METRICS__validation_level")
    tables["S4_learning_type"] = freq(val(A, "E_LEARNING__learning_type"), multi=True)
    tables["S4_learning_quality_measured"] = freq(val(A, "E_LEARNING__learning_quality_measured"))
    tables["S4_uses_foundation_model"] = freq(val(A, "E_LEARNING__uses_foundation_model"))
    tables["S4_explores"] = freq(val(A, "E_LEARNING__explores_beyond_predefined_env"))

    # --- R1 S5 metrics and maturity -----------------------------------------
    tables["S5_validation_level"] = freq(val(A, "G_VALIDATION_METRICS__validation_level"))
    tables["S5_metrics_x_validation"] = crosstab(A, "G_VALIDATION_METRICS__metrics_reported",
                                                 "G_VALIDATION_METRICS__validation_level",
                                                 multi_a=True)
    tables["S5_metrics_reported"] = freq(val(A, "G_VALIDATION_METRICS__metrics_reported"), multi=True)
    tables["S5_baseline"] = freq(val(A, "G_VALIDATION_METRICS__baseline_compared_against"))
    tables["S5_disruptions"] = freq(val(A, "G_VALIDATION_METRICS__disruption_types_handled"), multi=True)

    # --- R1 S6 delta vs reference --------------------------------------------
    tables["S6_features_common"] = freq(val(A, "I_DELTA_VS_REFERENCE__features_common_with_reference"),
                                        multi=True)
    tables["S6_cites_reference"] = freq(val(A, "I_DELTA_VS_REFERENCE__cites_reference_framework"))
    tables["S6_structure_x_explores"] = crosstab(A, "I_DELTA_VS_REFERENCE__static_vs_dynamic_structure",
                                                 "E_LEARNING__explores_beyond_predefined_env")
    tables["S6_agent_dt_coupling"] = freq(val(A, "F_DT_COUPLING__agent_dt_coupling"))

    # --- R1 S7 the gap -------------------------------------------------------
    ex = val(A, "E_LEARNING__explores_beyond_predefined_env")
    vl = val(A, "G_VALIDATION_METRICS__validation_level")
    both = ex.index.intersection(vl.index)
    gap = [pid for pid in both if ex[pid] == "yes" and vl[pid] in ("lab_hardware", "factory_floor")]
    titles = P.set_index("paper_id")["title"]
    out["S7_gap"] = {
        "rule": "explores_beyond_predefined_env == yes AND validation_level in "
                "{lab_hardware, factory_floor}",
        "n_assessable": len(both), "n_agent_papers": n_agent,
        "explores_yes": int((ex == "yes").sum()),
        "papers": [{"paper_id": p, "title": titles.get(p, "")} for p in gap],
    }

    # --- DT ------------------------------------------------------------------
    dt_ids = P[P["schema"] == "dt"]["paper_id"].tolist()
    conf_rows = []
    for pid in dt_ids:
        row, flags = {"paper_id": pid}, []
        for clause, (field, ok) in CONFORMANCE.items():
            r = D[(D["paper_id"] == pid) & (D["field"] == field)]
            if r.empty or r.iloc[0]["check"] not in COUNTED:
                row[clause] = 0
                flags.append(clause)
                continue
            v = r.iloc[0]["value"]
            sat = (bool([x for x in as_list(v) if x != "not_reported"]) if ok == "ANY"
                   else v in ok)
            row[clause] = int(sat)
        row["conformance_score"] = sum(row[c] for c in CONFORMANCE)
        row["excluded_clauses"] = "|".join(flags)
        conf_rows.append(row)
    conf = pd.DataFrame(conf_rows)
    tables["DT_conformance"] = conf
    if len(conf):
        out["DT_conformance"] = {
            "rule": {k: (v[0].split("__")[1], sorted(v[1]) if v[1] != "ANY" else "any value")
                     for k, v in CONFORMANCE.items()},
            "distribution": {int(k): int(v) for k, v in conf["conformance_score"].value_counts().sort_index().items()},
            "clause_rates": {c: round(conf[c].mean(), 3) for c in CONFORMANCE},
            "n": len(conf),
        }
    tables["DT_uq_x_validation"] = crosstab(D, "DT_B_OUTPUT_AND_UQ__uncertainty_quantified",
                                            "G_VALIDATION_METRICS__validation_level")
    tables["DT_uncertainty_quantified"] = freq(val(D, "DT_B_OUTPUT_AND_UQ__uncertainty_quantified"))
    tables["DT_construction_burden"] = crosstab(D, "DT_C_CONSTRUCTION_AND_UPDATE__sme_involvement_level",
                                                "DT_C_CONSTRUCTION_AND_UPDATE__model_updates_over_time")
    tables["DT_agent_coupling"] = freq(val(D, "DT_D_AGENT_RELATIONSHIP__dt_agent_coupling"))
    tables["DT_makes_decisions"] = freq(val(D, "DT_D_AGENT_RELATIONSHIP__dt_makes_decisions"))
    tables["DT_phase_iii_feasibility"] = crosstab(
        D, "DT_B_OUTPUT_AND_UQ__model_extractable_by_third_party",
        "DT_B_OUTPUT_AND_UQ__simulatable_forward_in_time")
    tables["DT_validation_level"] = freq(val(D, "G_VALIDATION_METRICS__validation_level"))

    # --- write ---------------------------------------------------------------
    written = {}
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        path = os.path.join(OUT, f"{name}.csv")
        df.to_csv(path)
        written[name] = {"path": path, "n": df.attrs.get("n")}
    out["tables"] = written
    with open(os.path.join(OUT, "aggregates.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
        fh.write("\n")
    P.to_csv(os.path.join(OUT, "papers.csv"), index=False)
    F.assign(value=F["value"].apply(lambda v: json.dumps(v) if not isinstance(v, str) else v)) \
        .to_csv(os.path.join(OUT, "fields_long.csv"), index=False)
    print(f"[agg] {len(P)} papers ({out['papers']}), {len(F)} fields")
    print(f"[agg] evidence: {out['evidence']['by_check']}")
    print(f"[agg] {len(written)} tables -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
