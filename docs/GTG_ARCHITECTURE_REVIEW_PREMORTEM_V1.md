# GTG Architecture Review + Pre-Mortem v1

Status: DRAFT
Purpose: attack the architecture before implementation.

## A. Data Contract Review

### A1. Pass conditions

The data architecture passes only if it can answer, for every value used in a decision:

1. What source produced it?
2. What instrument/venue does it actually describe?
3. When did the underlying event occur?
4. When was the value available to GTG?
5. Was it revised later?
6. Is it direct evidence or a proxy?
7. Is the source complete enough for the claimed capability?
8. Can the exact historical state be reproduced?

### A2. Critical decisions

- XAUT data cannot be relabeled as XAU order flow.
- Spot XAU cannot claim centralized exchange volume.
- Futures order flow requires feed capabilities that actually include trades/aggressor/book state.
- Macro data must be vintage-aware.
- Missing data is explicit, never silently neutral.
- Late/out-of-order events are handled by runtime ordering/watermark policy.
- Duplicate events require idempotency keys or deterministic deduplication.
- Source disagreement is retained as information.

### A3. Result

PASS WITH CONDITIONS.

Conditions:
- source capability matrix must be completed before implementation,
- exact provider choices remain separate from the domain contract,
- unavailable premium capabilities must degrade safely rather than be faked by proxies.

## B. Decision Contract Review

### B1. Inputs

GTG Core accepts immutable, versioned snapshots only.

No direct network/database/clock access is allowed inside inference.

### B2. Outputs

Direction and action are separated.

GTG may output directional hypothesis without authorizing a trade.

Required decision states:
- QUALIFIED
- ABSTAIN
- DATA_UNCERTAIN
- CONFLICT
- OOD

### B3. Evidence fusion

Rejected:
- naive indicator voting
- naive score addition
- double counting correlated features
- hard-coded confidence labels without calibration

Required:
- evidence family grouping
- provenance
- reliability/freshness
- conflict preservation
- calibrated fusion selected by validation, not intuition alone

### B4. Regime

Regime is context with uncertainty, not an infallible switch.

### B5. Result

PASS.

Open model-selection question is intentionally deferred to research because architecture should define the contract, not overfit an algorithm in advance.

## C. Validation Architecture Review

### C1. Primary failure to prevent

Researcher overfitting caused by:
- repeated parameter search,
- repeated model search,
- repeated event-definition changes,
- reuse of OOS after observing it,
- leakage through overlapping labels,
- revised macro data,
- live/backtest semantic mismatch.

### C2. Required pipeline

Development
-> purged/embargoed temporal validation
-> walk-forward
-> untouched OOS
-> frozen candidate
-> independent event-driven replay
-> fresh forward shadow

### C3. Experiment accounting

All attempts count.

Experiment Registry must include failures so multiple-testing risk can be measured.

### C4. Metrics

Classification:
- ROC-AUC / PR-AUC where meaningful
- MCC / balanced accuracy
- calibration/Brier/ECE
- selective precision and coverage

Path:
- candidate error vs locked baseline
- MFE/MAE and timing errors
- regime/month/session stability

Strategy-level:
- realistic costs/slippage
- drawdown
- Sharpe/Sortino only with uncertainty
- DSR/PBO or equivalent when trial multiplicity matters

### C5. Forward evidence

No arbitrary "60 events / 30 days" gate.

Campaign minimums derive from:
- desired confidence interval width,
- base rate,
- effect size,
- independence/effective sample size,
- regime coverage.

### C6. Result

PASS.

## D. Complexity Review

### D1. Keep

- GTG pure domain core
- NautilusTrader runtime spine candidate
- Parquet
- DuckDB
- PostgreSQL
- SQLAlchemy
- MLflow outside live critical path
- vectorbt for fast research
- LEAN independent verification
- OpenTelemetry
- Pydantic/Pandera
- selected technical-analysis libraries

### D2. Do not deploy now

- Feast
- Kafka
- Kubernetes
- Redis cluster
- generic workflow orchestration platform
- separate feature microservice
- online auto-retraining
- autonomous agents in the hot path

### D3. Why Feast is excluded

The architecture needs point-in-time correctness, but not necessarily a full feature-store service.
We adopt the semantic rule now and keep the option to introduce Feast later if online/offline feature serving complexity proves it necessary.

### D4. Why APScheduler is not the market loop

Market decisions are event-driven.
Schedulers are only for periodic non-market tasks:
- maintenance
- reporting
- macro polling where appropriate
- housekeeping

### D5. Why vectorbt is not authoritative

Vectorized research is valuable for speed, but event ordering, bar timing, latency, and state transitions require an event-driven authority for final validation.

### D6. Result

PASS.

## E. Pre-Mortem

Assume GTG failed badly after one year. What likely caused it?

### Failure 1: Backtest looked great because event definitions were tuned after seeing OOS
Prevention:
- event definition versioning
- experiment registry
- OOS consumption rule
- fresh forward campaign

### Failure 2: "Order flow" was actually a weak proxy
Prevention:
- source capability contracts
- instrument/venue provenance
- explicit proxy labels
- abstain when required capability missing

### Failure 3: Live features did not match historical features
Prevention:
- same feature definitions
- point-in-time snapshots
- schema hashes
- replay parity tests

### Failure 4: Historical macro data used revised values
Prevention:
- vintage-aware macro store
- available_at semantics

### Failure 5: Confidence was high but uncalibrated
Prevention:
- calibration metrics
- reliability curves
- selective prediction gates
- OOD checks

### Failure 6: Correlated indicators were counted as independent evidence
Prevention:
- evidence families
- correlation-aware fusion
- ablation

### Failure 7: Hard regime routing sent events to the wrong expert
Prevention:
- probabilistic regime context
- uncertainty preserved
- avoid hard routing unless validated

### Failure 8: Data outages produced plausible-looking values
Prevention:
- fail closed
- stale/gap state
- no silent fill
- data-quality gate

### Failure 9: Restart duplicated or lost predictions
Prevention:
- append-only lifecycle
- transactional/idempotent transitions
- deterministic IDs
- replay recovery

### Failure 10: Library update changed behavior
Prevention:
- lockfiles
- dependency manifest hash
- container/image digest
- reproducibility checks

### Failure 11: Forward model changed during the campaign
Prevention:
- frozen candidate manifest
- immutable model artifact hash
- new campaign for any model change

### Failure 12: Research engine and live runtime had different semantics
Prevention:
- same GTG domain core
- event-driven authoritative runtime
- parity tests
- independent LEAN audit

### Failure 13: Too many components caused operational failure
Prevention:
- modular monolith
- no premature microservices
- no unnecessary Kafka/Kubernetes/Feast

### Failure 14: Ahmed Toolbox outage disabled GTG
Prevention:
- Toolbox outside critical path
- explicit degradation/abstention policy

### Failure 15: Market regime changed permanently
Prevention:
- drift monitoring
- regime/OOD monitoring
- champion/challenger
- no automatic promotion

### Failure 16: Forward sample was too small but looked impressive
Prevention:
- predeclared statistical power/precision target
- effective sample size
- regime/session coverage
- confidence intervals

### Failure 17: Cross-source disagreement was averaged away
Prevention:
- disagreement retained as evidence
- source quality state
- no blind averaging

### Failure 18: Same evidence entered through multiple engineered features
Prevention:
- evidence lineage
- family grouping
- feature correlation audit
- ablation

### Failure 19: Timing assumptions worked in backtest but not live
Prevention:
- injected clock
- event-time/available-at semantics
- no exact-wall-clock polling assumptions
- authoritative event-driven replay

### Failure 20: We kept adding complexity to fix weak signal
Prevention:
- complexity budget
- every component must prove incremental value
- ablation before promotion
- remove components that do not improve evidence

## F. Architecture-level conclusion

The system should be optimized for truth, not for producing trades.

The preferred architecture is:

Data contracts
-> mature event runtime
-> immutable point-in-time snapshot
-> causal features
-> probabilistic regime context
-> selective event detector
-> provenance-aware evidence
-> pure GTG core
-> explicit abstention
-> immutable forward evidence

Research speed and production truth are different concerns.

Fast research:
vectorbt + DuckDB

Authoritative replay/live-shadow semantics:
NautilusTrader candidate

Independent audit:
LEAN

Model/experiment governance:
MLflow

Historical data:
Parquet

Operational metadata/evidence:
PostgreSQL

## G. Remaining blockers before Architecture Freeze

1. Source Capability Matrix for each intended gold/macro feed.
2. Exact GTG DecisionArtifact schema.
3. Exact promotion-statistics specification.
4. NautilusTrader compatibility spike definition (not implementation yet).
5. Exit strategy if NautilusTrader is rejected after compatibility review.

Until these are resolved:
ARCHITECTURE STATUS = NOT FROZEN.
