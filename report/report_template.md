# Agent-Based Frameworks for Manufacturing: A Structured Comparison Against the Reference Framework

**Report 1 — Phase II (AI-Agents), NIST MSR project (Tilbury / Barton).** Draft, 2026-10-09.

> **Status of this draft.** Every number in this report is computed by `aggregate.py` from the extraction records and inserted by `build_report.py`. None was typed by hand. A language model extracted the per-paper values, under verbatim-evidence checks, but no count, share or comparison here was computed by one. Three limits apply to all of it, and are set out in §10:
> 1. **No extraction record has been human-verified yet.** The per-field human-correction rate, the extraction counterpart of κ, has not been measured.
> 2. **The screening model was not independently calibrated.** On the only blind labels, Opus 5.5 scored binary κ {{bench.per_model.2.kappa_binary_primary}} and include recall {{bench.per_model.2.include_recall_binary}}, below the pre-registered floors of 0.70 and 0.95.
> 3. **The corpus is {{agg.papers.total}} of 84 selected papers**: four agent papers could not be retrieved. In addition, {{list.counts_by_outcome.needs_human_review}} screening decisions are still awaiting human review.

---

## Summary

This report compares **{{agg.papers.agent}} agent-based manufacturing frameworks** against the reference framework of Kovalenko, Tilbury and Barton [7, 8, 10], and examines a supporting set of **{{agg.papers.dt}} digital-twin (DT) frameworks**. The proposal (§2.2) declines to write "yet another survey of agent-based approaches to manufacturing". Following it, the comparison is structured: each paper is extracted against a fixed schema, every extracted value carries a verbatim quote that a program checks against the paper's text, and every count is computed deterministically.

The findings that bear on the Phase II case:

- **Agents that explore beyond a predefined environment are almost absent.** Only {{cell:S4_explores|yes|papers}} of {{agg.tables.S4_explores.n}} agent frameworks ({{pctcell:S4_explores|yes}}) can acquire capabilities, processes or environment structure they were not given. {{cell:S4_explores|no|papers}} cannot. This is the research gap proposal §2.2 and §3.2 describe, now measured rather than asserted.
- **The gap the Phase II case study targets is nearly empty.** Both exploring frameworks ({{agg.S7_gap.papers.0.paper_id}}, {{agg.S7_gap.papers.1.paper_id}}) are lab-hardware demonstrations, and neither reaches a factory floor (§8).
- **Most frameworks never define "agent".** {{cell:S1_definition_source|no_explicit_definition|papers}} of {{agg.tables.S1_definition_source.n}} give no explicit definition ({{pctcell:S1_definition_source|no_explicit_definition}}). Only {{cell:S1_definition_source|own_novel|papers}} propose their own.
- **Validation is mostly simulation or the lab.** {{cell:S5_validation_level|simulation|papers}} of {{agg.tables.S5_validation_level.n}} frameworks are validated in simulation and {{cell:S5_validation_level|lab_hardware|papers}} on lab hardware; {{cell:S5_validation_level|factory_floor|papers}} reach a factory floor.
- **Manufacturing metrics are often missing.** {{cell:S5_metrics_reported|none_reported|papers}} of {{agg.tables.S5_metrics_reported.n}} frameworks report no manufacturing metric. {{cell:S5_baseline|none|papers}} of {{agg.tables.S5_baseline.n}} compare against no baseline.
- **The agent/DT boundary that proposal §2.2 draws is not the field's norm.** In {{cell:DT_agent_coupling|dt_embeds_decision_making|papers}} of the {{agg.papers.dt}} DT frameworks, the DT itself embeds decision-making. On the agent side, {{cell:S6_agent_dt_coupling|no_dt|papers}} of {{agg.tables.S6_agent_dt_coupling.n}} frameworks use no DT at all, and only {{cell:S6_agent_dt_coupling|agent_queries_external_dt|papers}} query a DT as an external component, the arrangement the reference framework prescribes.
- **DT outputs rarely carry uncertainty.** {{cell:DT_uncertainty_quantified|yes_quantified|papers}} of {{agg.papers.dt}} DT frameworks quantify the uncertainty of their output, which the adopted DT definition [20] requires.

---

## 1. Method

### 1.1 Corpus

| Stage | Papers |
|---|---:|
| Candidates retrieved (OpenAlex seed-and-expand from the proposal bibliography) | {{prov.eligibility.candidates_total}} |
| Excluded at eligibility: no open-access copy | {{prov.eligibility.counts_by_exclusion_reason.no_oa}} |
| Eligible, screened by Opus 5.5 on title and abstract | {{prov.eligibility.eligible}} |
| Excluded at screening | {{list.counts_by_outcome.excluded_at_screening}} |
| Routed to human review (not yet decided; not in this report) | {{list.counts_by_outcome.needs_human_review}} |
| Included | {{list.counts_by_outcome.included}} |
| Selected for this report: all agent includes + top 15 DT includes by corpus rank | 84 |
| Full text retrieved and validated | {{agg.papers.total}} |

Eligibility requires an open-access copy (OpenAlex `is_oa`, or an Unpaywall OA location). Screening applied `criteria.md` with Opus 5.5 to title and abstract. Only 15 of the DT includes were taken, because the proposal scopes the DT side as a roughly 15-paper comparison subset. The 15 are the highest-ranked by the corpus's seed-weighted score, so retrieval status played no part in the choice.

### 1.2 Extraction

Each paper's complete text was extracted from its PDF with PyMuPDF and given to Opus 5.5, one paper per request with no shared context. The agent schema asks for {{ext.n_evidence_fields.agent}} evidence fields per paper and the DT schema for {{ext.n_evidence_fields.dt}}. Each field is returned as `{value, evidence_quote, page, confidence}`. A program then checks every quote against the paper's text: it must be verbatim, at most 25 words, and on the cited page.

| Evidence check, all {{agg.evidence.fields_total}} fields | n |
|---|---:|
| quote found verbatim on the cited page | {{agg.evidence.by_check.ok}} |
| quote found once punctuation and spacing are normalised | {{agg.evidence.by_check.ok_loose}} |
| an absence answer ("no", "none", "unclear"), which cannot be quoted | {{agg.evidence.by_check.absence_unquoted}} |
| `not_reported` | {{agg.evidence.by_check.not_reported}} |
| quote over 25 words | {{agg.evidence.by_check.quote_too_long}} |
| quote not found in the paper | {{agg.evidence.by_check.quote_not_found}} |
| other failures (value outside its schema, or no quote) | {{agg.evidence.by_check.bad_value}} + {{agg.evidence.by_check.missing_quote}} |

Of the fields where the paper was quoted in support of an answer, **{{pct:agg.evidence.verified_quote_share_of_answered}}** carry a quote the program verified in the text. **{{agg.evidence.quote_not_found}}** quotes could not be found at all: invented, paraphrased, or garbled by PDF extraction. Fields that fail a check ({{agg.evidence.excluded_from_tables}} in total) are left out of every table below and treated as missing. Each table states the number of papers it covers.

**Absence answers are not verified by quote.** A "no" means the model found nothing in the text, which no quote can show. These {{agg.evidence.by_check.absence_unquoted}} fields are the most important target for the human verification sample (§10).

Extraction cost **${{ext.result.actual_cost_usd}}** for {{agg.papers.total}} papers (batch, measured usage).

---

## 2. What is an agent? (definitional landscape)

{{table:S1_definition_source}}

*n = {{agg.tables.S1_definition_source.n}} agent frameworks.*

Most frameworks do not say what they mean by "agent". Of those that do, most borrow a definition from a cited work rather than from a canonical source such as Wooldridge and Jennings ({{cell:S1_definition_source|adopted_wooldridge_jennings|papers}}) or Leitão's industrial agents ({{cell:S1_definition_source|adopted_industrial_agents_leitao|papers}}). Proposal §2.4 asks for recommendations on clarifying and aligning terminology. Here the problem is less that definitions conflict than that **in most of these papers there is no definition to align.**

What the agents represent:

{{table:S1_agent_entity_types}}

Resource agents ({{cell:S1_agent_entity_types|machine_or_resource|papers}}) are more common than part or product agents ({{cell:S1_agent_entity_types|part_or_product|papers}}). Only {{cell:S1_agent_entity_types|human_operator|papers}} frameworks represent human operators as agents.

## 3. Necessary vs optional components

The reference framework gives each agent a knowledge base, a decision maker and a communication manager. How often each is present, by how far the framework was validated:

**Knowledge base:**

{{table:S2_has_knowledge_base_x_validation}}

**Communication manager:**

{{table:S2_has_communication_manager_x_validation}}

**Decision mechanism** (a framework may use several):

{{table:S2_decision_mechanism}}

Rule-based reasoning ({{cell:S2_decision_mechanism|rule_based_reasoning|papers}}) and market or auction mechanisms ({{cell:S2_decision_mechanism|auction_or_market|papers}}) lead. Reinforcement learning ({{cell:S2_decision_mechanism|reinforcement_learning|papers}}) and LLM or foundation-model decision-making ({{cell:S2_decision_mechanism|LLM_or_foundation_model|papers}}) are less common. Many papers are "unclear" on whether a component exists; the components are often implied by the architecture without being named as such.

Implementation platforms:

{{table:S2_implementation_platform}}

## 4. Memory and communication

**Memory.** What agents remember historically (rows) against what they read in real time (columns):

{{table:S3_memory_historical_x_realtime}}

The most common pairing is static configuration with real-time queries to other agents ({{xcell:S3_memory_historical_x_realtime|static_configuration_only|query_to_other_agents}} frameworks): agents that know only what they were configured with and learn the rest by asking each other. Only {{cell:S3_memory_requirements_stated|yes_explicit|papers}} of {{agg.tables.S3_memory_requirements_stated.n}} frameworks argue explicitly what an agent *must* remember.

**Communication cost:**

{{table:S3_comm_cost_analyzed}}

{{cell:S3_comm_cost_analyzed|no|papers}} of {{agg.tables.S3_comm_cost_analyzed.n}} frameworks do not analyse message volume, bandwidth or latency at all. {{cell:S3_comm_cost_analyzed|yes_quantified|papers}} quantify it.

## 5. Learning and exploration

{{table:S4_learning_type}}

*n = {{agg.tables.S4_learning_type.n}}; a framework may use several learning types.*

{{cell:S4_learning_type|none|papers}} frameworks use no learning at all. Reinforcement learning is the most common learning method ({{cell:S4_learning_type|reinforcement_learning|papers}}). Foundation models appear in {{cell:S4_uses_foundation_model|yes|papers}} of {{agg.tables.S4_uses_foundation_model.n}}.

**Exploration beyond the predefined environment**, the property proposal §2.2 and §3.2 identify as the gap, against validation level:

{{table:S4_explores_x_validation}}

The expectation recorded when the schema was designed was "a strong majority of `no`". The measured result is stronger than that: **{{cell:S4_explores|no|papers}} of {{agg.tables.S4_explores.n}}**.

Whether learning quality is measured at all:

{{table:S4_learning_quality_measured}}

## 6. Manufacturing metrics and maturity

{{table:S5_validation_level}}

{{table:S5_metrics_reported}}

*n = {{agg.tables.S5_metrics_reported.n}}; a framework may report several metrics.*

Makespan or cycle time is the most reported metric ({{cell:S5_metrics_reported|makespan_or_cycle_time|papers}}). Throughput, quality, reliability and downtime, the metrics the proposal names, are each reported by a minority. Downtime is reported by {{cell:S5_metrics_reported|downtime|papers}}.

What the frameworks are compared against:

{{table:S5_baseline}}

Disruptions the frameworks are shown to handle:

{{table:S5_disruptions}}

## 7. Delta against the reference framework

Features shared with the Kovalenko/Tilbury/Barton framework:

{{table:S6_features_common}}

*n = {{agg.tables.S6_features_common.n}}.*

Decentralized decision-making is the most commonly shared feature ({{cell:S6_features_common|decentralized_decision_making|papers}} of {{agg.tables.S6_features_common.n}}). The features that most distinguish the reference framework are its formal model-based representation ({{cell:S6_features_common|model_based_formal_representation|papers}}) and plan generation from discovered capabilities ({{cell:S6_features_common|plan_generation_from_discovered_capabilities|papers}}). Both are rare.

How the frameworks relate to the reference work:

{{table:S6_cites_reference}}

**Static vs dynamic structure, against exploration.** Proposal §3.2 notes that even the reference framework's prior work [10] assumes a static structure.

{{table:S6_structure_x_explores}}

{{xcell:S6_structure_x_explores|dynamic_structure|no}} frameworks have a dynamic structure, with agents that join, leave or reconfigure, yet do not explore. The two frameworks that explore both run on a static structure. Being able to reconfigure is not the same as being able to explore, and none of these frameworks does both.

## 8. The gap

The schema marks the intersection below as the evidence for the Phase II case study: agents that explore beyond their predefined environment **and** are validated on real hardware.

*Rule:* `{{agg.S7_gap.rule}}`. Of {{agg.S7_gap.n_assessable}} assessable agent frameworks, {{agg.S7_gap.explores_yes}} explore, and these are validated on hardware:

{{papers:S7_gap}}

Both are lab demonstrations, not factory deployments. One gains new skills through meta-learning over primitive skills learned by reinforcement learning. The other has resource agents retrieve manufacturing documentation through an LLM to discover capabilities at run time. The second also reports how its LLM fails: it compares numbers by their text rather than their value, it retrieves passages that are textually similar but irrelevant, and it degrades with long prompts. That is direct input to the Report 2 failure taxonomy. Neither combines exploration with a dynamic system structure, or with a digital twin as the simulation substrate the proposal envisions.

## 9. The digital-twin supporting set ({{agg.papers.dt}} papers)

### 9.1 Conformance to the reference DT definition

Each DT is scored 0–7: one point for each clause of the adopted definition [20] (proposal §3.1) that its extraction shows satisfied. Clauses are judged strictly: on-demand synchronisation and implicit subject-matter expertise do not count.

Distribution of scores (score: papers): **{{agg.DT_conformance.distribution}}**. Rate at which each clause is satisfied:

| Clause | Rate |
|---|---:|
| replica of the physical twin | {{agg.DT_conformance.clause_rates.replica}} |
| synchronized | {{agg.DT_conformance.clause_rates.synchronized}} |
| purpose stated | {{agg.DT_conformance.clause_rates.purpose}} |
| incorporates models | {{agg.DT_conformance.clause_rates.models}} |
| explicit subject-matter expertise | {{agg.DT_conformance.clause_rates.expertise}} |
| historical data | {{agg.DT_conformance.clause_rates.historical_data}} |
| real-time data | {{agg.DT_conformance.clause_rates.realtime_data}} |

Explicit subject-matter expertise is the clause least often met.

### 9.2 Uncertainty

{{table:DT_uq_x_validation}}

The definition requires a DT's output to be "accompanied by a level of confidence, or uncertainty". {{cell:DT_uncertainty_quantified|no|papers}} of {{agg.papers.dt}} frameworks do not address it, consistent with the §3.1 claim that uncertainty quantification is under-reported.

### 9.3 The agent/DT boundary

{{table:DT_agent_coupling}}

{{table:DT_makes_decisions}}

Proposal §2.2 separates the two roles: the agent decides, and the DT estimates and predicts. In {{cell:DT_agent_coupling|dt_embeds_decision_making|papers}} of {{agg.papers.dt}} DT frameworks the DT embeds decision-making itself, and only {{cell:DT_makes_decisions|no|papers}} clearly make no decisions. On the agent side the separation is also uncommon (§7: {{cell:S6_agent_dt_coupling|agent_queries_external_dt|papers}} frameworks query an external DT, and {{cell:S6_agent_dt_coupling|agent_and_dt_conflated|papers}} conflate the two). **The separation the reference framework prescribes is a minority position in both literatures.** That makes it a substantive design commitment to defend, not a convention to assume.

### 9.4 Construction burden and Phase III feasibility

{{table:DT_construction_burden}}

Can a third party extract a DT's model, and simulate it forward in time? Both are preconditions for the combined-simulation idea in proposal §2.2.

{{table:DT_phase_iii_feasibility}}

---

## 10. Limitations

1. **Extraction is not yet human-verified.** No record has been checked by a person, so the per-field human-correction rate is unmeasured. The verbatim check shows that a quote *exists*; it does not show that the quote *supports* the value. The {{agg.evidence.by_check.absence_unquoted}} absence answers are not backed by any quote. A stratified human check, weighted toward absence answers and toward the gap-defining fields (`explores_beyond_predefined_env`, `validation_level`), should come before any of these figures goes beyond this draft.
2. **Screening was not independently calibrated.** Opus 5.5's only blind calibration (labels v1) gave binary κ {{bench.per_model.2.kappa_binary_primary}} and include recall {{bench.per_model.2.include_recall_binary}}, below the pre-registered floors. The later 0.93 was measured against labels revised toward the model, and is not an independent figure. Some relevant papers are likely among the {{list.counts_by_outcome.excluded_at_screening}} screening excludes, and {{list.counts_by_outcome.needs_human_review}} papers awaiting human review are not here.
3. **Open-access only.** {{prov.eligibility.counts_by_exclusion_reason.no_oa}} of {{prov.eligibility.candidates_total}} candidates were excluded for having no open-access copy, including {{prov.eligibility.framework_core_linked.excluded}} of the {{prov.eligibility.framework_core_linked.total}} candidates linked to the reference framework's own seeds.
4. **Four selected agent papers could not be retrieved** and are absent.
5. **A single extracting model.** Values are Opus 5.5's reading of each paper, checked for quote presence but not cross-checked by a second model or rater.
6. **The DT subset is 15 of the 44 screened DT papers**, chosen by rank. Conclusions about DTs describe this subset.

## 11. Reproducing this report

```bash
python aggregate.py        # all tables, from extraction/records/
python build_report.py     # this file, from report/report_template.md
```

Records: `extraction/records/`. Tables: `extraction/aggregate/`. Extraction run: `extraction/runs/extract_20261009/`. Every number above can be traced to a CSV or JSON in those folders.
