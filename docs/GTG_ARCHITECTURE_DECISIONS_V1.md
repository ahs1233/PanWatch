# GTG Architecture Decisions v1

## ADR-001 — Separate architecture from implementation

Decision:
Architecture lives on branch architecture/gtg-v1 and contains documentation only.

Reason:
Prevent prototypes from becoming accidental architecture.

## ADR-002 — Framework-independent GTG Domain Core

Decision:
GTG logic is a pure domain layer.

Reason:
Testing, deterministic replay, portability, and reduced vendor/runtime lock-in.

## ADR-003 — RuntimePort over runtime vendor

Decision:
Freeze a runtime capability contract (RuntimePort), not a specific vendor/version.

NautilusTrader is Candidate A.
LEAN is Candidate B and independent verifier.
A custom runtime is a last resort only after both mature candidates fail documented critical acceptance requirements.

Why:
NautilusTrader is architecturally strong, but current 2.x releases are prerelease/RC and current inspected distributions require Python 3.12+. GTG must not inherit vendor lifecycle risk.

Constraint:
GTG Domain Core remains framework-independent and contains no Nautilus/LEAN-specific objects.

## ADR-004 — vectorbt is research-only authority

Decision:
Use vectorbt for fast research and screening.

Do not use it as the sole promotion authority for event-sensitive logic.

## ADR-005 — LEAN is independent verifier

Decision:
Frozen candidates are independently checked against LEAN where feasible.

A material mismatch blocks promotion until explained.

## ADR-006 — Parquet + DuckDB for historical analytics

Decision:
Use immutable Parquet as canonical analytical storage and DuckDB for efficient local analytical queries.

Avoid loading all historical bars into PostgreSQL.

## ADR-007 — PostgreSQL for operational evidence/state

Decision:
Use PostgreSQL for predictions, outcomes, manifests, experiment metadata references, and operational state.

## ADR-008 — MLflow outside the critical path

Decision:
Use MLflow for experiment/model lineage.

GTG live/shadow inference must continue if MLflow is unavailable.

## ADR-009 — No Feast deployment initially

Decision:
Adopt point-in-time correctness semantics without deploying a feature store.

Revisit only when measured online/offline feature-serving complexity justifies it.

## ADR-010 — No automatic online learning in the champion

Decision:
Champion remains frozen during a forward campaign.

Adaptive models may run only as challengers until separately validated.

## ADR-011 — No silent proxies

Decision:
A source may only support capabilities explicitly declared in its source contract.

Proxy evidence remains labeled as proxy evidence.

## ADR-012 — Abstention is success when evidence is unsafe

Decision:
No-decision is an expected, measured system outcome.

## ADR-013 — Complexity must earn its place

Decision:
No new infrastructure component enters architecture without:
- a concrete failure it solves,
- a measurable requirement,
- a simpler alternative analysis,
- an exit/removal plan.


## ADR-014 — Runtime qualification is a golden-test gate

Decision:
A runtime candidate is accepted only after deterministic replay, time-boundary, duplicate, late-event, restart/recovery, custom-data, catalog round-trip and Domain-Core isolation tests pass.

Reason:
A mature framework reduces infrastructure risk but does not automatically prove semantic compatibility with GTG.

## ADR-015 — Forward stopping rules must be preregistered

Decision:
Forward campaigns use either a fixed preregistered information horizon or an explicitly anytime-valid sequential inference method.

Ordinary repeated confidence intervals cannot justify early promotion.

Reason:
Continuous peeking can create false confidence even with fresh data.

## ADR-016 — Path quality is an independent promotion gate

Decision:
If GTG publishes MFE/MAE/path forecasts, path performance must beat a locked baseline with uncertainty bounds, not just point estimates.

Reason:
Directional classification can improve while path prediction remains worse than a simple baseline.

## ADR-017 — No fixed universal event-count promotion threshold

Decision:
Required forward sample/information size is derived per campaign from effect size, precision/power, dependence and regime/session coverage.

Reason:
A fixed event count such as 60 can be either grossly insufficient or unnecessarily large depending on the process.
