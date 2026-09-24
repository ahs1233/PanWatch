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

## ADR-003 — Prefer NautilusTrader as runtime spine candidate

Decision:
Do not build a custom clock/event/replay runtime unless compatibility review rejects NautilusTrader.

Why:
It already provides event-driven architecture, clock abstraction, shared backtest/live semantics, data adapters/catalog concepts, deterministic simulation work, and recovery/reconciliation patterns.

Constraint:
GTG Domain Core must remain independent so NautilusTrader can be replaced if required.

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
