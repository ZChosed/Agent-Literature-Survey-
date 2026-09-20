#!/usr/bin/env python3
"""
validate_schemas.py -- structural checks on the two extraction schemas.

Run this after ANY schema edit, and before an extraction batch. It catches the
class of error that silently corrupts a review: a field whose enum drifted, a
"shared spine" that stopped being shared, or a substantive field that lost its
evidence requirement.

Checks
------
 1. Both files parse as JSON.
 2. Every block declares a proposal_basis (traceability to the proposal is the
    whole point of the schema design; an untraceable field is scope creep).
 3. Every field in an evidence block is type 'evidence_object' or a declared
    exception, so the {value, evidence_quote, page, confidence} contract holds.
 4. Enum fields declare 'allowed'; multi_enum fields do too.
 5. The join keys the two schemas must share are byte-identical:
    IDENT, H_STANDARDS, J_EXTRACTION_META (whole blocks) and
    validation_level / application_domain (fields inside G_VALIDATION_METRICS).
    Basis: proposal S2.3 Phase III compares DT and agent features on common
    axes; these fields are the axes.
 6. No duplicate enum values, no empty allowed-lists.

Exit code 0 = clean, 1 = at least one ERROR. Warnings do not fail the run.

    python validate_schemas.py
    python validate_schemas.py --agent path/to/agent.json --dt path/to/dt.json
"""

from __future__ import annotations

import argparse
import json
import sys

# Blocks whose fields are metadata, not extracted claims, so they are exempt
# from the evidence_object requirement.
NON_EVIDENCE_BLOCKS = {"IDENT", "J_EXTRACTION_META"}

# Blocks required to be field-identical across the two schemas.
IDENTICAL_BLOCKS = ["IDENT", "H_STANDARDS", "J_EXTRACTION_META"]

# Fields inside G_VALIDATION_METRICS required to match exactly (the join keys).
SHARED_G_FIELDS = ["validation_level", "application_domain"]


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_schema(name: str, schema: dict, errors: list, warnings: list) -> None:
    blocks = schema.get("blocks")
    if not blocks:
        errors.append(f"{name}: no 'blocks' key")
        return

    for bname, block in blocks.items():
        if "proposal_basis" not in block and bname not in ("IDENT",):
            warnings.append(f"{name}.{bname}: no proposal_basis on the block")

        fields = block.get("fields", {})
        if not fields:
            errors.append(f"{name}.{bname}: block has no fields")
            continue

        exempt = bname in NON_EVIDENCE_BLOCKS or block.get("not_evidence_objects") is True

        for fname, spec in fields.items():
            if not isinstance(spec, dict):
                errors.append(f"{name}.{bname}.{fname}: field spec is not an object")
                continue

            ftype = spec.get("type")
            if ftype is None:
                errors.append(f"{name}.{bname}.{fname}: no 'type'")
                continue

            if not exempt and ftype not in ("evidence_object", "computed"):
                errors.append(
                    f"{name}.{bname}.{fname}: type={ftype!r}, expected 'evidence_object' "
                    f"(substantive fields must carry evidence_quote)"
                )

            vtype = spec.get("value_type")
            if vtype in ("enum", "multi_enum"):
                allowed = spec.get("allowed")
                if not allowed:
                    errors.append(f"{name}.{bname}.{fname}: {vtype} without 'allowed'")
                elif len(allowed) != len(set(allowed)):
                    dupes = sorted({v for v in allowed if allowed.count(v) > 1})
                    errors.append(f"{name}.{bname}.{fname}: duplicate enum values {dupes}")
            elif ftype == "evidence_object" and vtype is None and "sub_fields" not in spec:
                warnings.append(f"{name}.{bname}.{fname}: evidence_object without value_type")


def check_shared(agent: dict, dt: dict, errors: list, warnings: list) -> None:
    ab, tb = agent["blocks"], dt["blocks"]

    for blk in IDENTICAL_BLOCKS:
        if blk not in ab or blk not in tb:
            errors.append(f"shared spine: block {blk} missing from one schema")
            continue
        fa, ft = ab[blk].get("fields", {}), tb[blk].get("fields", {})
        if set(fa) != set(ft):
            errors.append(
                f"shared spine {blk}: field sets differ. "
                f"agent-only={sorted(set(fa) - set(ft))} dt-only={sorted(set(ft) - set(fa))}"
            )
            continue
        for fn in fa:
            a_allowed = fa[fn].get("allowed")
            t_allowed = ft[fn].get("allowed")
            if a_allowed != t_allowed:
                errors.append(f"shared spine {blk}.{fn}: 'allowed' lists differ")

    ga, gt = ab.get("G_VALIDATION_METRICS", {}), tb.get("G_VALIDATION_METRICS", {})
    gaf, gtf = ga.get("fields", {}), gt.get("fields", {})
    for fn in SHARED_G_FIELDS:
        if fn not in gaf or fn not in gtf:
            errors.append(f"join key G_VALIDATION_METRICS.{fn} missing from one schema")
            continue
        if gaf[fn].get("allowed") != gtf[fn].get("allowed"):
            errors.append(
                f"join key G_VALIDATION_METRICS.{fn}: enums differ -- cross-tabulation "
                f"between the agent and DT corpora would be invalid (proposal S2.3)"
            )

    # metrics_reported is allowed to differ by exactly the DT-specific addition.
    if "metrics_reported" in gaf and "metrics_reported" in gtf:
        a_set = set(gaf["metrics_reported"].get("allowed", []))
        t_set = set(gtf["metrics_reported"].get("allowed", []))
        extra_dt = t_set - a_set
        extra_agent = a_set - t_set
        if extra_dt - {"prediction_accuracy"}:
            errors.append(f"metrics_reported: unexpected DT-only values {sorted(extra_dt)}")
        if extra_agent - {"tardiness_or_due_date"}:
            errors.append(f"metrics_reported: unexpected agent-only values {sorted(extra_agent)}")

    # Cross-schema coupling enum must mirror, or the boundary finding can't be pooled.
    a_coup = ab.get("F_DT_COUPLING", {}).get("fields", {}).get("agent_dt_coupling", {})
    t_coup = tb.get("DT_D_AGENT_RELATIONSHIP", {}).get("fields", {}).get("dt_agent_coupling", {})
    if a_coup and t_coup:
        a_vals = set(a_coup.get("allowed", []))
        t_vals = set(t_coup.get("allowed", []))
        # These two enums describe the same relation from opposite seats; they are
        # expected to overlap heavily but not be identical.
        overlap = a_vals & t_vals
        if len(overlap) < 3:
            warnings.append(
                f"coupling enums share only {len(overlap)} values "
                f"({sorted(overlap)}); pooling the boundary finding will need a mapping table"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="agent_extraction_schema.json")
    p.add_argument("--dt", default="dt_extraction_schema.json")
    args = p.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    try:
        agent = load(args.agent)
        dt = load(args.dt)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR  could not load schemas: {e}")
        return 1

    check_schema("agent", agent, errors, warnings)
    check_schema("dt", dt, errors, warnings)
    check_shared(agent, dt, errors, warnings)

    for s, label in ((agent, "agent"), (dt, "dt")):
        nb = len(s["blocks"])
        nf = sum(len(b.get("fields", {})) for b in s["blocks"].values())
        print(f"{label:6s} v{s['schema_version']}  blocks={nb:2d}  fields={nf:3d}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARN   {w}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  ERROR  {e}")
        print("\nFAIL")
        return 1

    print("\nPASS -- schemas are structurally sound and the join keys agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
