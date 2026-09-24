# GTG Architecture v1 — Architecture Only

Status: READY FOR FINAL FREEZE REVIEW
Implementation: FORBIDDEN until architecture is frozen
Base commit: 6086611d47a476a677e0161036bcbf6647fbfe6c

## 1. Core principle

GTG is not a monolithic trading model.

GTG is a causal, event-driven decision system with five non-negotiable properties:

1. Point-in-time correctness.
2. Deterministic replay.
3. Explicit uncertainty and abstention.
4. Immutable evidence.
5. Same decision core across historical and forward environments.

The architecture follows:

Reuse first -> Integrate second -> Extend third -> Build from scratch last.

## 2. Architectural planes

### A. Data Plane
Responsibilities:
- ingest raw market and macro data,
- preserve source identity,
- normalize into canonical event contracts,
- detect gaps, duplicates, stale data and revisions,
- persist immutable historical data.

No trading intelligence is allowed here.

### B. Runtime Plane
Architecture boundary: RuntimePort.

The architecture freezes required runtime capabilities, not a specific vendor/version.

Responsibilities delegated to a mature runtime:
- event loop,
- injected clock/time frontier,
- event ordering,
- historical replay,
- live/shadow data adapters,
- data catalog,
- deterministic simulation semantics,
- message routing,
- restart/recovery primitives.

Runtime candidates:
- Candidate A: NautilusTrader
- Candidate B: LEAN
- Custom runtime: last resort only after mature candidates fail documented critical requirements.

GTG Domain Core must not contain runtime-specific objects.
Changing runtime must not require rewriting GTG Brain.

### C. GTG Domain Core
Framework-independent pure decision domain.

The core receives immutable snapshots and returns a decision artifact.
It must not:
- access network,
- query databases,
- call brokers,
- mutate historical evidence,
- read wall clock directly,
- depend on Ahmed Toolbox availability.

Input:
- MarketSnapshot
- FeatureSnapshot
- RegimeSnapshot
- LiquidityBehaviorSnapshot
- EvidenceSnapshot
- ModelManifest
- DataQualitySnapshot

Output:
- DecisionArtifact

### D. Research Plane
Purpose:
- hypothesis development,
- fast experiments,
- feature studies,
- ablations,
- candidate generation.

Tools:
- Parquet as immutable analytical storage,
- DuckDB for analytical queries,
- vectorbt for fast screening,
- MLflow Tracking/Registry for experiment and model lineage,
- Pandera/Pydantic for contracts.

Research results are not promotion evidence by themselves.

### E. Validation Plane
Purpose:
- authoritative event-driven replay,
- walk-forward evaluation,
- independent verification.

Authoritative replay:
- whichever RuntimePort candidate passes the GTG Runtime Acceptance Suite.

Independent audit:
- a second mature engine where feasible (LEAN is the default independent verifier when Nautilus is the selected runtime, and vice versa where practical).

vectorbt is a research accelerator, not the final authority when event sequencing matters.

### F. Forward Shadow Plane
Purpose:
- fresh, forward-only evidence,
- zero order execution,
- frozen candidate/model/features,
- immutable prediction/outcome journal.

The exact same GTG Domain Core is used as in historical validation.

### G. Future Execution Plane
Out of scope for v1.

No execution API belongs in GTG v1.
A future execution layer, if approved, must be isolated behind a separate policy/risk boundary.

## 3. Data model

Every raw/canonical market event must carry at least:

- source
- venue
- canonical_instrument_id
- native_instrument_id
- data_type
- event_time
- received_time
- available_at
- sequence or source ordering token when available
- revision/vintage when applicable
- quality flags
- payload hash
- source_event_id/idempotency key

Missing is not neutral.

A missing value must remain explicitly missing and may trigger degraded quality or abstention.

## 4. Source capability contracts

A source must declare what it can prove.

Examples:

XAU spot source:
- spot price: yes
- decentralized global traded volume: no
- exchange order book: no
- true footprint: no

GC futures source:
- exchange price: yes
- futures volume: yes
- aggressor/order-flow fields: only if feed supports them
- order book: only if feed supports required depth

XAUT venue:
- tokenized gold price: yes
- venue-specific volume/order book: yes if available
- XAU spot footprint: no

Macro source:
- observation value
- release time
- availability time
- vintage/revision

No proxy may silently impersonate a different capability.

## 5. Time model

Three concepts are distinct:

- event_time: when the market/economic event occurred
- received_time: when GTG infrastructure received it
- available_at: earliest time the information was legally/causally usable

Bars become usable only after their closing frontier.

Multi-timeframe features must use closed bars only unless a feature is explicitly defined as an intrabar feature.

Macro backtests must use the value/vintage available at the historical decision time, not today's revised value.

## 6. Data quality gate

Each decision context has a DataQualitySnapshot.

Possible states:
- HEALTHY
- DEGRADED
- STALE
- GAP_DETECTED
- SOURCE_DISAGREEMENT
- CAPABILITY_MISSING
- SCHEMA_MISMATCH
- UNKNOWN

Policy:
- UNKNOWN is not HEALTHY.
- Critical uncertainty -> abstain.
- Noncritical degradation may reduce confidence only if predeclared.

No silent interpolation in decision-critical streams.

Historical synthetic fills, if ever used for analytics, must be marked synthetic and excluded from decision evidence unless explicitly approved.

## 7. Feature architecture

Generic math should come from mature libraries.

Candidate reuse:
- pandas / numpy
- pandas-ta-classic
- MarketProfile
- smartmoneyconcepts
- scikit-learn
- River for streaming metrics/drift

Custom GTG feature semantics include:
- MA slope
- MA curvature
- MA spacing/compression
- distance-to-MA
- RSI slope/curvature/state
- multi-timeframe alignment
- source agreement
- GTG-specific event context

Each feature definition must have:
- feature_id
- semantic_version
- source requirements
- lookback
- available_at rule
- missing-data rule
- deterministic implementation
- schema hash

## 8. Regime engine

Regime is context, not a brittle hard switch.

Output should be a probability/confidence vector when possible, e.g.:
- trend
- range
- compression
- expansion
- shock
- transition
- uncertain

The GTG Brain consumes regime context and uncertainty.

Initial architecture forbids:
- "if regime=A use model X, otherwise model Y" without validation,
because hard routers can amplify classification mistakes.

## 9. Strategic Liquidity & Market-Behavior Layer

Purpose:
- model observable liquidity-provider / dealer / aggressor behavior from broad context to local microstructure,
- connect strategic risk environment, structural liquidity, auction/session state, local liquidity and executed flow,
- produce probabilistic behavior hypotheses rather than claims about private intent.

Top-down levels:
1. strategic risk environment,
2. structural liquidity map,
3. auction/session state,
4. local liquidity state,
5. aggressor flow + price response,
6. final LiquidityBehaviorSnapshot.

Core rules:
- behavior, not mind reading,
- no "manipulation/trap" story labels without rigorous preregistered definitions,
- depth alone is not liquidity,
- lower-capability sources cannot fabricate higher-capability claims,
- instrument/venue provenance survives to DecisionArtifact,
- correlated derivatives of the same raw flow are not independent votes,
- NO_RELIABLE_INFERENCE is a valid state.

This layer feeds:
- Event Detector with meaningful behavior/state transitions,
- Evidence Engine with structured behavior evidence.

It never authorizes an order.

Detailed contract and failure review:
docs/GTG_STRATEGIC_LIQUIDITY_BEHAVIOR_LAYER_V1.md

## 10. Event detector

GTG is selective.

The system does not ask for a directional answer every minute.

Flow:
market update -> state update -> setup detector -> event or nothing.

Every event definition is versioned and preregistered before OOS/forward evaluation.

Event identity includes:
- event_definition_version
- instrument
- event_time
- direction/setup family
- deterministic event_id

Changing event logic creates a new event definition version and invalidates direct comparison unless explicitly controlled.

## 11. Evidence engine

Evidence is grouped by families to prevent double counting.

Families:
- structure
- trend
- momentum
- location
- volatility
- volume
- liquidity
- order flow
- strategic_liquidity_behavior
- macro
- cross-market/source agreement

Each evidence item includes:
- family
- source
- direction
- strength
- reliability
- freshness
- regime relevance
- provenance
- availability time

Correlated evidence must not be treated as independent votes.

The architecture rejects simple naive summation as the final fusion rule.

## 12. GTG Brain

GTG Brain is a pure decision component.

It may use:
- calibrated ML models,
- probabilistic fusion,
- expert models,
- analog/path models,
- deterministic safety rules.

But the interface is fixed before model selection.

Required outputs:
- directional hypothesis: UP / DOWN / NONE
- decision state: QUALIFIED / ABSTAIN / DATA_UNCERTAIN / CONFLICT / OOD
- calibrated probability where applicable
- expected path metrics (e.g. MFE/MAE) if model is validated
- uncertainty
- evidence summary
- conflict summary
- model/version references
- snapshot references

No direct order action is produced.

## 13. Abstention as a first-class output

GTG must be allowed to say no.

Reasons include:
- insufficient data
- poor data quality
- regime uncertainty
- source disagreement
- model out-of-distribution
- evidence conflict
- low calibrated confidence
- missing required capability

Abstention rate is measured and reported.

## 14. Market snapshot

The GTG Brain never reads a mutable live dataframe directly.

For a decision event, the runtime constructs an immutable MarketSnapshot containing only information available at that time frontier.

The same snapshot contract is reconstructed during historical replay.

A later data arrival cannot retroactively change a committed prediction.

## 15. Storage architecture

### Market history
Primary format:
- Parquet

Query layer:
- DuckDB

Replay/catalog layer:
- RuntimePort-compatible catalog/adapter over the same immutable historical data.

### Operational state / evidence
- PostgreSQL
- SQLAlchemy abstraction

### Experiment/model lineage
- MLflow Tracking + Model Registry, outside the live critical path.

### Forward evidence
Append-only journal semantics:
- PredictionCreated
- OutcomeObserved
- EvaluationCompleted

No in-place mutation of committed historical predictions.

## 16. Experiment governance

Every experiment must be registered, including failures.

Required metadata:
- experiment_id
- code commit
- dataset manifest/hash
- feature schema hash
- event definition version
- parameters
- seed
- model family
- metrics
- result status

The number of attempted variants is part of the evidence.

Cherry-picking only successful trials is prohibited.

## 17. Model manifest

A frozen candidate includes:

- model_id
- model_artifact_hash
- training_dataset_hash
- feature_schema_hash
- event_definition_version
- code_commit
- dependency lock hash
- container/image digest where used
- training window
- calibration method
- validation report id

Forward campaigns pin exact immutable versions.
Mutable aliases such as "champion" are not accepted as campaign identity.

## 18. Validation architecture

Stages:

1. Development data
2. Purged/embargoed temporal validation
3. Walk-forward evaluation
4. Untouched OOS
5. Frozen-candidate independent replay
6. Fresh Forward Shadow

Rules:
- no random K-fold for temporal market prediction
- label overlap must be purged
- embargo length derives from information/label horizon
- once OOS is inspected and used to alter the design, it becomes development evidence
- new claims require new untouched/fresh evidence

Secondary robustness:
- CPCV or equivalent may be used when justified
- regime/month/session stability
- bootstrap confidence intervals
- calibration metrics
- selective risk/coverage
- cost/slippage sensitivity
- multiple-testing adjustment / DSR when strategy-level returns are evaluated

Forward minimum sample size is not hard-coded arbitrarily.
It must come from a power/precision requirement defined before the campaign.

## 19. Research vs authoritative engines

vectorbt:
- fast research
- parameter exploration
- ablation
- candidate generation

Runtime Candidate A — NautilusTrader:
- event-driven runtime candidate
- clock/event semantics
- replay/catalog/custom data
- version must pass compatibility and golden tests

Runtime Candidate B — LEAN:
- independent verifier
- fallback authoritative runtime candidate
- must pass the same Runtime Acceptance Suite

Disagreement between authoritative engines is a defect to investigate, not a result to average.

## 20. Recovery and idempotency

Event lifecycle:

DETECTED
-> SNAPSHOT_COMMITTED
-> PREDICTION_COMMITTED
-> WAITING_OUTCOME
-> OUTCOME_COMMITTED
-> EVALUATED

Every transition must be idempotent.

Restart after any step must recover without:
- duplicate prediction,
- lost prediction,
- duplicate outcome,
- model/version substitution,
- retroactive changes.

## 21. Observability

Use existing OpenTelemetry-compatible infrastructure.

Every decision should correlate:
- trace_id
- event_id
- snapshot_id
- prediction_id
- model_id
- source ids

Goal:
a strange final decision must be traceable back to raw source events.

## 22. Ahmed Toolbox boundary

Ahmed Toolbox may provide:
- research
- web/news retrieval
- macro context
- evidence gathering
- analyst/agent workflows

Ahmed Toolbox must not be:
- the system clock
- the market event runtime
- required for core numerical inference
- a single point of failure

If Toolbox is unavailable:
- GTG continues with available local/market context,
- confidence may degrade,
- required missing context may cause abstention.

## 23. Deliberate exclusions

Do not add now:
- Kafka
- Kubernetes
- Feast deployment
- Redis cluster
- microservice decomposition
- graph database
- autonomous online model promotion
- automatic online retraining
- direct broker execution
- custom scheduler for market events

Add complexity only after measured need.

## 24. Complexity target

Preferred topology:
- modular monolith for GTG domain,
- mature external runtime spine,
- isolated research services,
- PostgreSQL,
- Parquet/DuckDB,
- MLflow outside critical runtime,
- OpenTelemetry.

Fewer moving parts are a reliability feature.

## 25. Freeze criteria

Architecture may be marked FROZEN only after:

- Data Contract Review passed
- Decision Contract Review passed
- Validation Architecture Review passed
- Complexity Review passed
- Pre-Mortem Review passed
- unresolved critical risks = 0
- all deliberate assumptions documented
- all external dependencies have fallback/exit strategy
- Source Capability Matrix passed
- exact DecisionArtifact contract passed
- Promotion Statistics specification passed
- RuntimePort acceptance specification passed
- runtime exit strategy passed
- Strategic Liquidity & Market-Behavior Layer review passed
- behavior-layer golden architecture tests defined
- no unsupported market-maker-intent claims in machine vocabulary

Current gate status:
- critical unresolved architecture blockers: 0
- implementation qualification remains required for providers/runtime candidates/campaign parameters
- architecture is not FROZEN until final human review approves the freeze
