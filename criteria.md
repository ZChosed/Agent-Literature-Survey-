Screening Criteria — Phase II Literature Review (v0.1, 2026-09-28)
Purpose

You are screening candidate papers for a structured review of agent-based frameworks for manufacturing systems, with a small supporting set of digital twin (DT) framework papers. The review compares existing frameworks against a reference architecture in which:

every resource (machine, robot, conveyor, human) has an agent that decides which processes it executes;
every part or product has an agent that requests processes from resources to reach its goal;
agents communicate and negotiate with each other, keep a knowledge base / memory, and make decisions and take actions on behalf of their physical counterpart;
DTs are separate: a DT estimates or predicts the state of its physical twin with an uncertainty; an agent may query DTs but is not part of them.

The project is especially interested in agents that learn or explore beyond predefined capabilities (e.g., through simulation, reinforcement learning, or queries to foundation models), and in the manufacturing metrics (throughput, quality, reliability, downtime) that frameworks demonstrate.

You see only a title plus either an abstract or the first part of the full text. Judge only from that text. Do not use outside knowledge of the paper.

Decision procedure

Apply the steps in order and stop at the first one that decides.

Is it a research paper? If not → exclude, E1.
Is it a review, survey, or bibliometric study? If yes → exclude, E2.
Is the domain manufacturing system operations? If clearly not → exclude, E3. If unclear → uncertain, U2.
Does the paper contribute an agent-based approach, as defined below?
If yes → include with the highest-priority matching ID from I1–I4.
If the paper uses the word "agent" in an excluded sense → exclude, E4, E5 or E6.
If unclear → uncertain, U1.
If it has no agents, does it contribute a DT framework, as defined below?
If yes → include, D1.
If it is a DT paper without a framework contribution → exclude, E7.
Otherwise → exclude, E4.
If the text is too short or truncated to evaluate a step that matters → uncertain, U3.

When unsure between include and exclude, choose uncertain, not include. uncertain means a specific condition above cannot be evaluated from the text; your reason must name that condition. Do not use uncertain because the paper seems only moderately relevant; relevance ranking happens later.

What counts as "agent-based" (step 4)

All three must hold:

Autonomous decision-making entities. Software entities that decide and act, not merely compute or predict.
They represent or control manufacturing entities. Examples: machines, robots, workstations, transporters/AGVs, parts, products, orders, or production units, including holons.
They make operational decisions for the manufacturing system. Examples: scheduling, routing, resource allocation, task negotiation, reconfiguration, disruption or failure handling, dispatching, production control.

Multi-agent systems, holonic architectures, product/resource agents, BDI agents, multi-agent reinforcement learning with one agent per machine or part, and LLM-based agents all qualify if the three conditions hold.

Include IDs (use the highest-priority match)
I1 — Learning or exploratory agents. Agents that learn, adapt their own knowledge or structure, or explore beyond predefined capabilities. Examples: RL, simulation-based exploration, queries to foundation or LLM models, online learning.
I2 — Agent–DT integration. Agents that use, query, orchestrate, or are generated from digital twins or DT models.
I3 — LLM/foundation-model agents for manufacturing operations. Language or foundation models acting as, or inside, decision-making agents for the operational decisions above.
I4 — Agent architecture or mechanism. Multi-agent or holonic frameworks, product/resource agent architectures, negotiation, communication protocols, agent memory or knowledge representation, and agent interoperability standards (e.g., Asset Administration Shell–based agents), applied to manufacturing control. Conceptual, simulated, lab and factory validation all qualify.
D1 — DT framework (supporting set). Applies only when there are no agents. The paper proposes or formalizes a DT framework, architecture, definition or specification for manufacturing assets or systems, and it addresses at least one of the following:
required or optional DT characteristics;
DT outputs with uncertainty quantification;
alignment with standards (ISA-95, Asset Administration Shell, etc.);
how DTs interface with decision-making components.
Exclude IDs
E1 — Not a research paper. Editorial, masthead, table of contents, call for papers, erratum, advertisement, book front matter, proceedings preface.
E2 — Review, survey, bibliometric analysis, or position overview without a framework of its own. These are kept separately; do not include them.
E3 — Outside manufacturing system operations. Examples:
vehicles or automotive driving;
power grids or energy markets;
healthcare, buildings, agriculture, finance, transportation networks;
inter-company supply chains without a shop-floor component;
general-purpose robotics with no manufacturing system context.
E4 — Manufacturing, but no agent-based approach and no DT framework. Examples:
optimization or scheduling algorithms without agents;
a single centralized RL or ML controller that does not represent individual entities;
process monitoring or prediction models;
robot motion, grasping or manipulation learning at the task level;
quality inspection or vision models.
E5 — "Agent" only as a simulation modeling technique. Agent-based modeling or simulation used to analyze or predict system behavior, where the agents do not make decisions controlling a real or emulated manufacturing system.
E6 — AI or LLM for non-operational tasks. Design, CAD, generative part design, code or PLC generation, documentation, question answering, training or education, when no agent makes operational decisions in a running manufacturing system.
E7 — DT without a framework contribution. A DT built for one specific asset, prediction or visualization, with no general framework, definition or specification and no agents.
Uncertain IDs
U1 — Agent status unclear. It cannot be determined whether "agent" means an autonomous decision-making entity representing manufacturing entities (step 4).
U2 — Domain unclear. It cannot be determined whether the application is manufacturing system operations (step 3).
U3 — Insufficient text. Title only, or the text is truncated before the contribution is described.
Not your concern
Publication year, language, and venue: already filtered.
Validation quality, novelty, and citation count.
How relevant an included paper is. Do not downgrade a qualifying paper to uncertain or exclude because it seems minor.
Output

Return exactly one decision (include / exclude / uncertain), one criterion ID from the lists above, and a reason of at most 25 words that refers to the text.