# Agent-Based Frameworks for Manufacturing: A Structured Comparison Against the Reference Framework

**Report 1 — Phase II (AI-Agents), NIST MSR project (Tilbury / Barton).** Draft, 2026-10-09.

> **Status of this draft.** Every number in this report is computed by `aggregate.py` from the extraction records and inserted by `build_report.py`. None was typed by hand. A language model extracted the per-paper values, under verbatim-evidence checks, but no count, share or comparison here was computed by one. Three limits apply to all of it, and are set out in §10:
> 1. **No extraction record has been human-verified yet.** The per-field human-correction rate, the extraction counterpart of κ, has not been measured.
> 2. **The screening model was not independently calibrated.** On the only blind labels, Opus 5.5 scored binary κ 0.588 and include recall 0.667, below the pre-registered floors of 0.70 and 0.95.
> 3. **The corpus is 80 of 84 selected papers**: four agent papers could not be retrieved. In addition, 130 screening decisions are still awaiting human review.

---

## Summary

This report compares **65 agent-based manufacturing frameworks** against the reference framework of Kovalenko, Tilbury and Barton [7, 8, 10], and examines a supporting set of **15 digital-twin (DT) frameworks**. The proposal (§2.2) declines to write "yet another survey of agent-based approaches to manufacturing". Following it, the comparison is structured: each paper is extracted against a fixed schema, every extracted value carries a verbatim quote that a program checks against the paper's text, and every count is computed deterministically.

The findings that bear on the Phase II case:

- **Agents that explore beyond a predefined environment are almost absent.** Only 2 of 65 agent frameworks (3%) can acquire capabilities, processes or environment structure they were not given. 62 cannot. This is the research gap proposal §2.2 and §3.2 describe, now measured rather than asserted.
- **The gap the Phase II case study targets is nearly empty.** Both exploring frameworks (W3101443789, W7126169356) are lab-hardware demonstrations, and neither reaches a factory floor (§8).
- **Most frameworks never define "agent".** 33 of 64 give no explicit definition (52%). Only 4 propose their own.
- **Validation is mostly simulation or the lab.** 31 of 65 frameworks are validated in simulation and 23 on lab hardware; 3 reach a factory floor.
- **Manufacturing metrics are often missing.** 21 of 65 frameworks report no manufacturing metric. 29 of 64 compare against no baseline.
- **The agent/DT boundary that proposal §2.2 draws is not the field's norm.** In 7 of the 15 DT frameworks, the DT itself embeds decision-making. On the agent side, 49 of 64 frameworks use no DT at all, and only 7 query a DT as an external component, the arrangement the reference framework prescribes.
- **DT outputs rarely carry uncertainty.** 2 of 15 DT frameworks quantify the uncertainty of their output, which the adopted DT definition [20] requires.

---

## 1. Method

### 1.1 Corpus

| Stage | Papers |
|---|---:|
| Candidates retrieved (OpenAlex seed-and-expand from the proposal bibliography) | 1,941 |
| Excluded at eligibility: no open-access copy | 794 |
| Eligible, screened by Opus 5.5 on title and abstract | 1,147 |
| Excluded at screening | 904 |
| Routed to human review (not yet decided; not in this report) | 130 |
| Included | 113 |
| Selected for this report: all agent includes + top 15 DT includes by corpus rank | 84 |
| Full text retrieved and validated | 80 |

Eligibility requires an open-access copy (OpenAlex `is_oa`, or an Unpaywall OA location). Screening applied `criteria.md` with Opus 5.5 to title and abstract. Only 15 of the DT includes were taken, because the proposal scopes the DT side as a roughly 15-paper comparison subset. The 15 are the highest-ranked by the corpus's seed-weighted score, so retrieval status played no part in the choice.

### 1.2 Extraction

Each paper's complete text was extracted from its PDF with PyMuPDF and given to Opus 5.5, one paper per request with no shared context. The agent schema asks for 51 evidence fields per paper and the DT schema for 32. Each field is returned as `{value, evidence_quote, page, confidence}`. A program then checks every quote against the paper's text: it must be verbatim, at most 25 words, and on the cited page.

| Evidence check, all 3,795 fields | n |
|---|---:|
| quote found verbatim on the cited page | 2,282 |
| quote found once punctuation and spacing are normalised | 37 |
| an absence answer ("no", "none", "unclear"), which cannot be quoted | 1,006 |
| `not_reported` | 414 |
| quote over 25 words | 33 |
| quote not found in the paper | 7 |
| other failures (value outside its schema, or no quote) | 7 + 9 |

Of the fields where the paper was quoted in support of an answer, **97.6%** carry a quote the program verified in the text. **7** quotes could not be found at all: invented, paraphrased, or garbled by PDF extraction. Fields that fail a check (56 in total) are left out of every table below and treated as missing. Each table states the number of papers it covers.

**Absence answers are not verified by quote.** A "no" means the model found nothing in the text, which no quote can show. These 1,006 fields are the most important target for the human verification sample (§10).

Extraction cost **$7.52** for 80 papers (batch, measured usage).

---

## 2. What is an agent? (definitional landscape)

|  | value | papers | share |
|---|---|---|---|
| 0 | no_explicit_definition | 33 | 0.516 |
| 1 | adopted_other_cited | 19 | 0.297 |
| 2 | adopted_wooldridge_jennings | 4 | 0.062 |
| 3 | own_novel | 4 | 0.062 |
| 4 | adopted_industrial_agents_leitao | 3 | 0.047 |
| 5 | adopted_holonic_HMS | 1 | 0.016 |

*n = 64 agent frameworks.*

Most frameworks do not say what they mean by "agent". Of those that do, most borrow a definition from a cited work rather than from a canonical source such as Wooldridge and Jennings (4) or Leitão's industrial agents (3). Proposal §2.4 asks for recommendations on clarifying and aligning terminology. Here the problem is less that definitions conflict than that **in most of these papers there is no definition to align.**

What the agents represent:

|  | value | papers | share |
|---|---|---|---|
| 0 | machine_or_resource | 48 | 0.75 |
| 1 | part_or_product | 35 | 0.547 |
| 2 | supervisory_or_system | 31 | 0.484 |
| 3 | transport_or_AGV | 20 | 0.312 |
| 4 | other | 16 | 0.25 |
| 5 | robot_or_manipulator | 15 | 0.234 |
| 6 | batch_or_order | 10 | 0.156 |
| 7 | process | 9 | 0.141 |
| 8 | human_operator | 8 | 0.125 |

Resource agents (48) are more common than part or product agents (35). Only 8 frameworks represent human operators as agents.

## 3. Necessary vs optional components

The reference framework gives each agent a knowledge base, a decision maker and a communication manager. How often each is present, by how far the framework was validated:

**Knowledge base:**

| has_knowledge_base | conceptual_only | emulation_or_HIL | factory_floor | lab_hardware | simulation |
|---|---|---|---|---|---|
| no | 0 | 0 | 0 | 0 | 5 |
| not_reported | 0 | 0 | 0 | 0 | 2 |
| unclear | 2 | 1 | 1 | 7 | 8 |
| yes | 4 | 1 | 2 | 16 | 16 |

**Communication manager:**

| has_communication_manager | conceptual_only | emulation_or_HIL | factory_floor | lab_hardware | simulation |
|---|---|---|---|---|---|
| no | 1 | 0 | 0 | 0 | 4 |
| not_reported | 1 | 0 | 0 | 3 | 10 |
| unclear | 2 | 2 | 1 | 8 | 11 |
| yes | 2 | 0 | 2 | 12 | 6 |

**Decision mechanism** (a framework may use several):

|  | value | papers | share |
|---|---|---|---|
| 0 | rule_based_reasoning | 26 | 0.448 |
| 1 | auction_or_market | 16 | 0.276 |
| 2 | planning_or_search | 13 | 0.224 |
| 3 | hybrid | 13 | 0.224 |
| 4 | optimization_MILP_or_CP | 13 | 0.224 |
| 5 | reinforcement_learning | 12 | 0.207 |
| 6 | heuristic_dispatching | 10 | 0.172 |
| 7 | supervised_ML | 5 | 0.086 |
| 8 | LLM_or_foundation_model | 4 | 0.069 |
| 9 | not_reported | 2 | 0.034 |

Rule-based reasoning (26) and market or auction mechanisms (16) lead. Reinforcement learning (12) and LLM or foundation-model decision-making (4) are less common. Many papers are "unclear" on whether a component exists; the components are often implied by the architecture without being named as such.

Implementation platforms:

|  | value | papers | share |
|---|---|---|---|
| 0 | not_reported | 16 | 0.246 |
| 1 | custom_python | 14 | 0.215 |
| 2 | JADE | 10 | 0.154 |
| 3 | commercial_simulation_tool | 10 | 0.154 |
| 4 | custom_java_or_cpp | 5 | 0.077 |
| 5 | none_conceptual_only | 4 | 0.062 |
| 6 | PLC_or_IEC_61131 | 3 | 0.046 |
| 7 | ROS_or_ROS2 | 3 | 0.046 |

## 4. Memory and communication

**Memory.** What agents remember historically (rows) against what they read in real time (columns):

| memory_historical | event_driven_stream | not_reported | polled_sensor_reads | query_to_digital_twin | query_to_other_agents | shared_blackboard |
|---|---|---|---|---|---|---|
| case_library_of_past_episodes | 1 | 0 | 2 | 0 | 1 | 0 |
| full_event_log | 0 | 0 | 0 | 1 | 1 | 0 |
| learned_model_parameters | 1 | 1 | 6 | 0 | 3 | 1 |
| logged_asset_history | 2 | 1 | 3 | 1 | 5 | 1 |
| not_reported | 1 | 0 | 1 | 1 | 0 | 0 |
| static_configuration_only | 5 | 1 | 3 | 2 | 19 | 1 |

The most common pairing is static configuration with real-time queries to other agents (19 frameworks): agents that know only what they were configured with and learn the rest by asking each other. Only 13 of 61 frameworks argue explicitly what an agent *must* remember.

**Communication cost:**

|  | value | papers | share |
|---|---|---|---|
| 0 | no | 42 | 0.646 |
| 1 | yes_quantified | 14 | 0.215 |
| 2 | yes_qualitative | 9 | 0.138 |

42 of 65 frameworks do not analyse message volume, bandwidth or latency at all. 14 quantify it.

## 5. Learning and exploration

|  | value | papers | share |
|---|---|---|---|
| 0 | none | 39 | 0.6 |
| 1 | reinforcement_learning | 12 | 0.185 |
| 2 | supervised_learning | 7 | 0.108 |
| 3 | parameter_tuning | 7 | 0.108 |
| 4 | evolutionary_or_metaheuristic | 3 | 0.046 |
| 5 | LLM_query_or_in_context | 3 | 0.046 |
| 6 | kb_expansion_via_communication | 2 | 0.031 |
| 7 | simulation_based_exploration | 2 | 0.031 |
| 8 | case_based_reasoning | 2 | 0.031 |

*n = 65; a framework may use several learning types.*

39 frameworks use no learning at all. Reinforcement learning is the most common learning method (12). Foundation models appear in 4 of 65.

**Exploration beyond the predefined environment**, the property proposal §2.2 and §3.2 identify as the gap, against validation level:

| explores_beyond_predefined_env | conceptual_only | emulation_or_HIL | factory_floor | lab_hardware | simulation |
|---|---|---|---|---|---|
| no | 6 | 2 | 3 | 21 | 30 |
| unclear | 0 | 0 | 0 | 0 | 1 |
| yes | 0 | 0 | 0 | 2 | 0 |

The expectation recorded when the schema was designed was "a strong majority of `no`". The measured result is stronger than that: **62 of 65**.

Whether learning quality is measured at all:

|  | value | papers | share |
|---|---|---|---|
| 0 | no | 51 | 0.785 |
| 1 | yes_quantified | 14 | 0.215 |

## 6. Manufacturing metrics and maturity

|  | value | papers | share |
|---|---|---|---|
| 0 | simulation | 31 | 0.477 |
| 1 | lab_hardware | 23 | 0.354 |
| 2 | conceptual_only | 6 | 0.092 |
| 3 | factory_floor | 3 | 0.046 |
| 4 | emulation_or_HIL | 2 | 0.031 |

|  | value | papers | share |
|---|---|---|---|
| 0 | none_reported | 21 | 0.323 |
| 1 | makespan_or_cycle_time | 21 | 0.323 |
| 2 | cost | 10 | 0.154 |
| 3 | tardiness_or_due_date | 9 | 0.138 |
| 4 | throughput | 9 | 0.138 |
| 5 | flexibility_or_reconfiguration_time | 8 | 0.123 |
| 6 | quality_scrap_or_yield | 7 | 0.108 |
| 7 | energy | 6 | 0.092 |
| 8 | resource_utilization | 5 | 0.077 |
| 9 | reliability_or_availability | 5 | 0.077 |
| 10 | downtime | 1 | 0.015 |

*n = 65; a framework may report several metrics.*

Makespan or cycle time is the most reported metric (21). Throughput, quality, reliability and downtime, the metrics the proposal names, are each reported by a minority. Downtime is reported by 1.

What the frameworks are compared against:

|  | value | papers | share |
|---|---|---|---|
| 0 | none | 29 | 0.453 |
| 1 | no_agent_baseline | 12 | 0.188 |
| 2 | alternative_MAS | 10 | 0.156 |
| 3 | centralized_optimal | 8 | 0.125 |
| 4 | centralized_heuristic | 5 | 0.078 |

Disruptions the frameworks are shown to handle:

|  | value | papers | share |
|---|---|---|---|
| 0 | machine_failure_or_breakdown | 24 | 0.393 |
| 1 | demand_or_order_change | 18 | 0.295 |
| 2 | none | 17 | 0.279 |
| 3 | resource_added_or_removed | 14 | 0.23 |
| 4 | new_part_type_introduced | 7 | 0.115 |
| 5 | tool_wear_or_degradation | 6 | 0.098 |
| 6 | quality_defect | 6 | 0.098 |
| 7 | transport_delay | 5 | 0.082 |

## 7. Delta against the reference framework

Features shared with the Kovalenko/Tilbury/Barton framework:

|  | value | papers | share |
|---|---|---|---|
| 0 | decentralized_decision_making | 48 | 0.774 |
| 1 | part_or_product_centric_agent | 32 | 0.516 |
| 2 | explicit_separation_of_agent_and_physical_asset | 32 | 0.516 |
| 3 | action_request_and_negotiation | 29 | 0.468 |
| 4 | reactive_replanning_on_disruption | 27 | 0.435 |
| 5 | resource_capability_discovery | 23 | 0.371 |
| 6 | model_based_formal_representation | 15 | 0.242 |
| 7 | control_oriented_architecture | 13 | 0.21 |
| 8 | plan_generation_from_discovered_capabilities | 10 | 0.161 |

*n = 62.*

Decentralized decision-making is the most commonly shared feature (48 of 62). The features that most distinguish the reference framework are its formal model-based representation (15) and plan generation from discovered capabilities (10). Both are rare.

How the frameworks relate to the reference work:

|  | value | papers | share |
|---|---|---|---|
| 0 | does_not_cite | 39 | 0.6 |
| 1 | cites_as_related_work | 16 | 0.246 |
| 2 | cites_and_builds_on | 10 | 0.154 |

**Static vs dynamic structure, against exploration.** Proposal §3.2 notes that even the reference framework's prior work [10] assumes a static structure.

| static_vs_dynamic_structure | no | unclear | yes |
|---|---|---|---|
| dynamic_structure | 36 | 1 | 0 |
| static_structure | 22 | 0 | 2 |
| unclear | 1 | 0 | 0 |

36 frameworks have a dynamic structure, with agents that join, leave or reconfigure, yet do not explore. The two frameworks that explore both run on a static structure. Being able to reconfigure is not the same as being able to explore, and none of these frameworks does both.

## 8. The gap

The schema marks the intersection below as the evidence for the Phase II case study: agents that explore beyond their predefined environment **and** are validated on real hardware.

*Rule:* `explores_beyond_predefined_env == yes AND validation_level in {lab_hardware, factory_floor}`. Of 65 assessable agent frameworks, 2 explore, and these are validated on hardware:

- Multi-Agent Modeling of Cyber-Physical Systems for IEC 61499 Based Distributed Automation (`W3101443789`)
- Adaptive task planning and coordination in multi-agent manufacturing systems using large language models (`W7126169356`)

Both are lab demonstrations, not factory deployments. One gains new skills through meta-learning over primitive skills learned by reinforcement learning. The other has resource agents retrieve manufacturing documentation through an LLM to discover capabilities at run time. The second also reports how its LLM fails: it compares numbers by their text rather than their value, it retrieves passages that are textually similar but irrelevant, and it degrades with long prompts. That is direct input to the Report 2 failure taxonomy. Neither combines exploration with a dynamic system structure, or with a digital twin as the simulation substrate the proposal envisions.

## 9. The digital-twin supporting set (15 papers)

### 9.1 Conformance to the reference DT definition

Each DT is scored 0–7: one point for each clause of the adopted definition [20] (proposal §3.1) that its extraction shows satisfied. Clauses are judged strictly: on-demand synchronisation and implicit subject-matter expertise do not count.

Distribution of scores (score: papers): **3: 1, 4: 1, 5: 6, 6: 5, 7: 2**. Rate at which each clause is satisfied:

| Clause | Rate |
|---|---:|
| replica of the physical twin | 0.933 |
| synchronized | 0.733 |
| purpose stated | 1.00 |
| incorporates models | 0.933 |
| explicit subject-matter expertise | 0.4 |
| historical data | 0.667 |
| real-time data | 0.733 |

Explicit subject-matter expertise is the clause least often met.

### 9.2 Uncertainty

| uncertainty_quantified | conceptual_only | emulation_or_HIL | lab_hardware | simulation |
|---|---|---|---|---|
| mentioned_not_quantified | 2 | 0 | 0 | 0 |
| no | 5 | 0 | 4 | 2 |
| yes_quantified | 0 | 1 | 0 | 1 |

The definition requires a DT's output to be "accompanied by a level of confidence, or uncertainty". 11 of 15 frameworks do not address it, consistent with the §3.1 claim that uncertainty quantification is under-reported.

### 9.3 The agent/DT boundary

|  | value | papers | share |
|---|---|---|---|
| 0 | no_agent | 8 | 0.533 |
| 1 | dt_embeds_decision_making | 7 | 0.467 |

|  | value | papers | share |
|---|---|---|---|
| 0 | unclear | 6 | 0.4 |
| 1 | no | 5 | 0.333 |
| 2 | yes | 4 | 0.267 |

Proposal §2.2 separates the two roles: the agent decides, and the DT estimates and predicts. In 7 of 15 DT frameworks the DT embeds decision-making itself, and only 5 clearly make no decisions. On the agent side the separation is also uncommon (§7: 7 frameworks query an external DT, and 3 conflate the two). **The separation the reference framework prescribes is a minority position in both literatures.** That makes it a substantive design commitment to defend, not a convention to assume.

### 9.4 Construction burden and Phase III feasibility

| sme_involvement_level | automatic_online_update | manual_recalibration | not_reported |
|---|---|---|---|
| heavy_manual | 0 | 1 | 2 |
| largely_automated | 1 | 0 | 0 |
| moderate | 1 | 1 | 1 |
| not_reported | 5 | 0 | 1 |

Can a third party extract a DT's model, and simulate it forward in time? Both are preconditions for the combined-simulation idea in proposal §2.2.

| model_extractable_by_third_party | no | unclear | yes |
|---|---|---|---|
| implied | 0 | 1 | 3 |
| not_reported | 0 | 1 | 3 |
| yes_explicit_interface | 2 | 1 | 2 |

---

## 10. Limitations

1. **Extraction is not yet human-verified.** No record has been checked by a person, so the per-field human-correction rate is unmeasured. The verbatim check shows that a quote *exists*; it does not show that the quote *supports* the value. The 1,006 absence answers are not backed by any quote. A stratified human check, weighted toward absence answers and toward the gap-defining fields (`explores_beyond_predefined_env`, `validation_level`), should come before any of these figures goes beyond this draft.
2. **Screening was not independently calibrated.** Opus 5.5's only blind calibration (labels v1) gave binary κ 0.588 and include recall 0.667, below the pre-registered floors. The later 0.93 was measured against labels revised toward the model, and is not an independent figure. Some relevant papers are likely among the 904 screening excludes, and 130 papers awaiting human review are not here.
3. **Open-access only.** 794 of 1,941 candidates were excluded for having no open-access copy, including 52 of the 154 candidates linked to the reference framework's own seeds.
4. **Four selected agent papers could not be retrieved** and are absent.
5. **A single extracting model.** Values are Opus 5.5's reading of each paper, checked for quote presence but not cross-checked by a second model or rater.
6. **The DT subset is 15 of the 44 screened DT papers**, chosen by rank. Conclusions about DTs describe this subset.

## 11. Reproducing this report

```bash
python aggregate.py        # all tables, from extraction/records/
python build_report.py     # this file, from report/report_template.md
```

Records: `extraction/records/`. Tables: `extraction/aggregate/`. Extraction run: `extraction/runs/extract_20261009/`. Every number above can be traced to a CSV or JSON in those folders.
