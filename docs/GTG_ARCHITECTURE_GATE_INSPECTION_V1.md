# GTG Architecture Gate Inspection v1

Status: PASSED — READY FOR FINAL FREEZE REVIEW
Scope: architecture only; no implementation authorization.

This document closes the five architecture blockers identified in the pre-mortem.

---

## Gate 1 — Source Capability Matrix

Rule: source identity and capability are part of the data contract.
No source may silently impersonate a stronger source.

### Existing / currently evidenced sources

| Source | Instrument / venue | Direct price | Bid/ask | Executed trade volume | Aggressor side | Order book | Open interest | True XAU OTC footprint | Role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dukascopy BI5 | XAUUSD | Yes | Yes | No | No | No | No | No | Historical price/tick/liquidity activity |
| Biquote MT5 reference | XAUUSD | Yes | Yes | No guaranteed centralized volume | No | No | No | No | Indicative spot reference |
| GoldPriceDev reference | XAU spot reference | Yes | provider-dependent | No | No | No | No | No | Secondary indicative spot |
| OKX | XAU-USDT-SWAP | Yes | Yes | Yes, venue-specific | Public trade side available; semantics must be adapter-tested | Yes, venue-specific | Yes | No | Centralized gold-derivative microstructure proxy |
| OKX | XAUT-USDT | Yes | Yes | Yes, venue-specific | Public trade side available; semantics must be adapter-tested | Yes, venue-specific | N/A | No | Tokenized-gold microstructure proxy |
| Bitfinex | XAUT/USD | Yes | Yes | Yes, venue-specific | Yes from signed public trade amount | Yes, raw R0 book | N/A | No | Tokenized-gold microstructure proxy |

### Desired but not currently present as a PanWatch provider

| Source class | Capability | Architectural status |
|---|---|---|
| CME COMEX GC trade / MDP-quality feed | Centralized futures trades, exchange volume, aggressor/order-flow fields when feed supplies them, book depth when entitled | OPTIONAL HIGH-VALUE CAPABILITY; not assumed until acquired and verified |
| Vintage-aware macro source (ALFRED/FRED class) | Historical vintages/revisions | REQUIRED for revision-sensitive historical macro features |
| Timestamped economic-release source | Exact release/availability timestamp and surprise values | REQUIRED only for intraday event-surprise features; cannot be inferred from revised daily macro series |

### Capability rules

1. Dukascopy quoted bid/ask sizes are liquidity/activity proxies, not executed global gold volume.
2. XAUT venue volume is XAUT venue volume, not XAU OTC volume.
3. OKX XAU-USDT-SWAP is centralized derivative evidence, not global OTC XAUUSD order flow.
4. A venue footprint is labeled with venue + instrument.
5. "Global XAU footprint" is forbidden unless a future data source can actually justify that claim.
6. Missing CME-quality data does not get synthetically replaced by XAUT or MT5 tick volume.
7. Missing critical capability => CAPABILITY_MISSING and either abstention or a predeclared reduced-feature model.
8. Macro values must carry release_time, available_at and vintage/revision identity.

### Data source acceptance contract

A new provider is accepted only after:
- schema validation,
- timestamp semantics documented,
- duplicate identity documented,
- gap detection/recovery defined,
- source capabilities explicitly declared,
- historical/live semantics compared,
- unit/contract conversion tested,
- no silent fallback to a semantically different source.

Gate 1 result: PASS.

---

## Gate 2 — Exact DecisionArtifact Contract

GTG v1 separates hypothesis from action.
It never emits an executable order.

### Identity

- schema_version
- decision_id
- event_id
- event_definition_version
- instrument_id
- decision_time
- time_frontier
- campaign_id: nullable outside forward campaign

### Immutable references

- market_snapshot_id
- market_snapshot_hash
- feature_snapshot_id
- feature_schema_hash
- regime_snapshot_id
- evidence_snapshot_id
- data_quality_snapshot_id
- model_manifest_id
- model_artifact_hash
- code_commit
- dependency_manifest_hash

### Decision state

directional_hypothesis:
- UP
- DOWN
- NONE

decision_state:
- QUALIFIED
- ABSTAIN
- DATA_UNCERTAIN
- CONFLICT
- OOD

### Calibrated inference

Fields are nullable unless the corresponding model is validated:

- p_success
- p_up
- p_down
- calibration_version
- uncertainty_interval
- ood_score
- tradeability_probability

Important:
- arbitrary uncalibrated "confidence 81%" is forbidden.
- a score may only be called probability if calibrated and validated as such.

### Path forecast

Prefer distributional/quantile output over a single point:

- mfe_atr_p10 / p50 / p90
- mae_atr_p10 / p50 / p90
- target_time_minutes_p50 / p90
- path_model_id

If path model is not validated:
- all path fields are null,
- no fabricated fallback path estimate is shown as a prediction.

### Evidence summary

For each evidence family:
- family
- direction
- strength
- reliability
- freshness
- regime_relevance
- provenance_refs
- missing flag

Also:
- conflicts[]
- missing_capabilities[]
- gate_reasons[]
- abstention_reasons[]

### Quality

- data_quality_state
- feature_completeness
- source_disagreement_state
- regime_uncertainty
- ood_state

### Governance

Hardcoded in GTG v1:
- execution_authorized = false
- promotion_authorized = false

Promotion is a separate governance process, never an inference-field side effect.

### Explanation

A human-readable explanation may be generated from structured fields.
Free-form LLM text is not allowed to change the decision.

### Integrity

- artifact_hash over canonical serialized artifact
- append-only once committed

Gate 2 result: PASS.

---

## Gate 3 — Promotion Statistics Specification

Promotion is evidence-based and preregistered.
No single attractive metric can promote a model.

### 3.1 Campaign preregistration

Before observing final evaluation outcomes, freeze:

- candidate manifest
- event definition
- feature schema
- baseline
- primary endpoints
- secondary endpoints
- economic / operational minimum effect
- stopping rule
- required market-regime/session coverage
- exclusion rules
- cost/slippage assumptions if P&L is evaluated

Any change creates a new campaign.

### 3.2 Historical evidence sequence

1. Development
2. Purged/embargoed temporal validation
3. Walk-forward
4. Untouched OOS
5. Frozen-candidate authoritative event-driven replay
6. Independent engine parity/audit where feasible
7. Fresh Forward Shadow

If OOS influenced a design change, it is no longer untouched evidence.

### 3.3 Directional primary evidence

For a selective directional model, report at minimum:

- event base rate
- selected coverage
- precision / recall
- MCC
- balanced accuracy
- Brier score / Brier skill
- calibration error / reliability
- confidence interval or sequential-valid interval

Promotion requires:
- useful directional discrimination,
- positive calibration skill against locked baseline,
- no result driven solely by vanishing coverage,
- preregistered lower-bound improvement over the baseline for the primary selective metric.

The minimum meaningful uplift is a campaign parameter derived before the test, not chosen after seeing results.

### 3.4 Path primary evidence

Prior GTG versions showed that classification can improve while path quality fails.
Therefore path is a separate primary gate when path forecasts are used.

For MFE and MAE:
- compare candidate error to a locked baseline using paired events,
- report candidate/baseline error ratio,
- promotion requires the upper uncertainty bound of the ratio to be below 1.0 for each required path endpoint, or below a stricter preregistered margin.

Point estimate < 1 alone is insufficient.

If target-time is operationally used, it receives its own preregistered test.

### 3.5 Calibration

A directional probability must demonstrate:
- Brier skill > 0 against locked baseline,
- acceptable reliability/calibration,
- no severe degradation in required regimes.

An uncalibrated score cannot be promoted as a probability.

### 3.6 Coverage

No universal arbitrary 10% coverage rule.

Minimum coverage is defined from:
- intended use,
- opportunity frequency,
- effective sample size,
- precision target,
- regime/session coverage.

A model with excellent precision on trivial coverage must be reported as such and cannot be promoted unless that coverage satisfies the preregistered operational objective.

### 3.7 Forward stopping rule

Default policy:
- preregistered fixed-horizon / fixed-information campaign,
- no early promotion because a dashboard temporarily looks good.

Alternative:
- an explicitly designed anytime-valid confidence-sequence/e-value procedure may allow continuous monitoring.

Ordinary repeated 95% confidence intervals are not used for optional stopping.

### 3.8 Sample size

No arbitrary "60 events / 30 days" rule.

Before campaign start compute the requirement from:
- expected base rate,
- smallest meaningful effect,
- target interval precision or power,
- dependence/effective sample size,
- required regimes and sessions.

Calendar duration is a coverage constraint, not a substitute for statistical information.

### 3.9 Stability

Report by:
- regime
- session
- month/time block
- volatility bucket
- source-quality state

No requirement that every tiny slice be positive.
But catastrophic failure in a required operational slice blocks promotion.

### 3.10 Multiple testing

Experiment registry counts all attempted variants.

When strategy/P&L selection is evaluated:
- apply multiple-testing-aware analysis such as DSR/PBO where appropriate,
- report number of tried variants,
- never present the winner as if it were the only experiment.

### 3.11 P&L

GTG v1 is a decision engine, not an execution system.
P&L is secondary research evidence unless a later execution architecture is approved.

Any P&L evaluation must include realistic:
- spread
- fees
- slippage
- latency assumptions
- session liquidity constraints

### 3.12 Final promotion rule

A candidate is not promotable unless:

HISTORICAL VALIDATION
AND
AUTHORITATIVE REPLAY
AND
INDEPENDENT AUDIT / EXPLAINED PARITY
AND
FRESH FORWARD EVIDENCE

are mutually compatible.

Material disagreement => NO PROMOTION.

Gate 3 result: PASS.

---

## Gate 4 — Runtime Compatibility and Acceptance

### Finding

NautilusTrader is architecturally attractive, but it must not be hardwired into GTG architecture.

Current release-state finding:
- current non-prerelease GitHub release is v1.231.0 and is labeled Beta,
- current 2.x line is release-candidate / prerelease,
- current distributed wheels inspected are Python 3.12+,
- PanWatch currently spans Python 3.11 and 3.12.

Therefore:
GTG architecture does not mandate a Nautilus version.

### RuntimePort

GTG architecture freezes a capability contract, not a vendor.

A conforming RuntimePort must provide:

1. injected clock/time frontier,
2. deterministic historical replay,
3. ordered market/custom events,
4. closed-bar semantics,
5. custom data types for macro/proxy/evidence streams,
6. immutable/canonical snapshot construction,
7. duplicate/idempotency handling,
8. restart/recovery behavior,
9. data catalog or adapter to immutable historical data,
10. same GTG Domain Core in historical and forward modes,
11. observability hooks,
12. no forced broker/execution coupling.

### Runtime acceptance test suite

A runtime candidate passes only if all critical tests pass:

R1. Replay same dataset >=3 times -> identical event/decision hashes.
R2. Historical and forward adapter produce identical feature/snapshot semantics for an equivalent fixture.
R3. Shuffled raw arrival fixture respects declared event/available-at ordering policy.
R4. Duplicate input does not duplicate committed decision.
R5. Multi-timeframe boundary tests prevent incomplete H1/H4/D1 leakage.
R6. Late event behavior is explicit and reproducible.
R7. Restart after SNAPSHOT_COMMITTED recovers without duplicate/lost prediction.
R8. Restart after PREDICTION_COMMITTED recovers without mutation.
R9. Custom streams can represent OKX/XAUT/macro without lying about instrument identity.
R10. Dataset/catalog round-trip preserves timestamps, units and provenance.
R11. Runtime failure cannot silently substitute another model/source.
R12. No runtime-specific object is allowed inside the GTG Domain Core contract.

Any failure in R1-R8 or R12 is critical and rejects the candidate.

### Candidate A — NautilusTrader

Strengths:
- event-driven architecture,
- backtest/live conceptual alignment,
- clock abstraction,
- catalog/custom data/adapters,
- mature trading-engine design.

Constraints:
- version line must be chosen only after compatibility test,
- 2.x RC is not assumed production-stable,
- Python 3.12 isolation may be required,
- GTG cannot depend directly on unstable version-specific APIs.

Recommended evaluation mode:
isolated Python 3.12 runtime adapter / sidecar during implementation spike, with GTG Domain Core remaining portable.

### Candidate B — LEAN

Role:
- independent verifier today,
- fallback authoritative runtime if Nautilus fails acceptance.

Strengths:
- mature event-driven engine,
- explicit time frontier,
- backtest/live design,
- custom data,
- current Python 3.11 support.

Cost:
- heavier integration boundary,
- .NET engine and adapter work.

### Candidate C — custom runtime

Not a normal option.

Allowed only if both mature candidates fail documented critical requirements.
If needed, build the thinnest possible adapter/runtime for the missing capability, not a new full trading engine.

Gate 4 result: PASS AT ARCHITECTURE LEVEL.
Implementation-time candidate qualification is still required, but no architecture redesign is needed if a candidate fails.

---

## Gate 5 — Runtime Exit Strategy

The GTG Domain Core and data contracts remain independent of runtime vendor.

### Portability boundary

Runtime-specific code may:
- receive external events,
- order/replay events,
- build canonical snapshots,
- call GTG Domain Core,
- persist lifecycle events.

Runtime-specific code may not define:
- GTG features semantically,
- GTG evidence meaning,
- event labels,
- model logic,
- decision schema,
- promotion rules.

### Replacement path

If Nautilus is rejected:
1. retain GTG Domain Core unchanged,
2. retain source adapters/canonical contracts unchanged,
3. implement LEAN RuntimePort adapter,
4. rerun Runtime Acceptance Suite,
5. compare golden decision hashes.

If LEAN also fails:
1. document exact missing requirements,
2. seek another maintained runtime/library,
3. only then consider a narrow custom runtime component.

No rewrite of GTG Brain is permitted merely because runtime vendor changes.

Gate 5 result: PASS.

---

# Final Gate Summary

| Gate | Result |
|---|---|
| Source Capability Matrix | PASS |
| DecisionArtifact Contract | PASS |
| Promotion Statistics | PASS |
| Runtime Compatibility Architecture | PASS |
| Runtime Exit Strategy | PASS |

Critical unresolved architecture blockers: 0.

Important implementation qualification remaining:
- provider adapters must pass source-specific acceptance tests,
- runtime candidate must pass Runtime Acceptance Suite,
- campaign-specific statistical effect/precision values must be preregistered before each evaluation.

These are implementation/experiment qualifications, not architecture holes.

Recommendation:
ARCHITECTURE STATUS -> READY FOR FINAL FREEZE REVIEW.
