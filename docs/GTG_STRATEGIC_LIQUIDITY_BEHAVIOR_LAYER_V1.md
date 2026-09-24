# GTG Strategic Liquidity & Market-Behavior Layer v1

Status: ARCHITECTURE-ONLY — REVIEWED
Implementation: FORBIDDEN until GTG Architecture v1 is frozen

## 1. Purpose

This layer adds a top-down analytical model for the behavior of liquidity providers,
dealers, market makers and aggressive participants without pretending to know any
participant's private intent.

It starts from the broad market environment and descends toward local execution
microstructure.

The goal is not to tell a story about "what the market maker wants".

The goal is to answer:

- What liquidity/risk environment exists?
- Where is risk being transferred?
- Is liquidity being provided, withdrawn, replenished or consumed?
- Is aggressive flow producing price progress?
- Is the market accepting or rejecting price?
- Is the evidence strong enough to infer a behavior state at all?

## 2. Epistemic rule — behavior, not mind reading

Allowed:
- infer observable behavior,
- estimate a probabilistic behavior state,
- label a hypothesis as possible/probable,
- preserve uncertainty,
- say NO_RELIABLE_INFERENCE.

Forbidden:
- claim knowledge of a dealer's private inventory,
- claim a single actor controls the market,
- label "manipulation", "trap", "stop hunt", "accumulation" or "distribution"
  from chart shape alone,
- convert XAUT/crypto venue behavior into an unlabeled XAU OTC claim,
- infer spoofing without order-event data sufficient to support the definition.

All outputs require:
- explicit evidence,
- source provenance,
- capability checks,
- time-causal availability,
- uncertainty.

## 3. Why this is a separate layer

This is not another indicator family.

It is a cross-horizon contextual model that connects:

Strategic environment
-> structural liquidity
-> auction/session state
-> local liquidity state
-> executed flow
-> price response
-> behavior hypothesis

It feeds both:
- Event Detector: "is something important happening?"
- Evidence Engine: "what does the observed interaction mean?"

It must not directly authorize a trade.

## 4. Top-down hierarchy

### Level 0 — Strategic Risk Environment

Question:
What broad risk-bearing environment are liquidity providers operating inside?

Candidate inputs, only when causally available:
- gold volatility regime,
- USD/rates context,
- major macro-event risk,
- futures basis / cross-market dislocation,
- futures open interest,
- funding/hedging-market stress proxies,
- cross-venue disagreement,
- options/volatility information when a verified source is later available.

Outputs:
- normal_risk_bearing
- elevated_risk_bearing_cost
- event_risk
- cross_market_dislocation
- uncertain

This level is context, not an entry signal.

### Level 1 — Structural Liquidity Map

Question:
Where has the market historically accepted or rejected price, and where may
future liquidity interaction matter?

Candidate features:
- prior day/week/month extremes,
- session extremes,
- rolling volume profile from eligible sources,
- POC / value-area migration,
- high/low-volume nodes,
- repeated rejection / acceptance zones,
- major swing structure,
- gaps / thin-auction zones where definition is causal,
- futures open-interest changes when available.

Rules:
- every profile is source/instrument labeled,
- a profile cannot use future session volume,
- rolling/session value must be reconstructed point-in-time,
- XAUT profile is never labeled global XAU profile.

Outputs:
- structural_liquidity_zones[]
- accepted_value_zones[]
- rejected_price_zones[]
- thin_liquidity_zones[]
- structural_context_quality

### Level 2 — Auction / Session State

Question:
How is price migrating through the trading day and across sessions?

Candidate concepts:
- balance vs directional auction,
- value migrating higher/lower,
- acceptance above/below a prior reference,
- failed auction hypothesis,
- session handoff,
- overnight positioning proxy,
- opening expansion / rejection,
- compression before expansion.

Sessions are calendar/market constructs, not hardcoded naive clock labels.
DST and exchange calendars must be handled explicitly.

Outputs:
- auction_state
- value_migration
- acceptance_state
- session_context
- auction_uncertainty

### Level 3 — Local Liquidity State

Question:
What is happening to immediately available liquidity near price?

Only when the feed capability supports it.

Candidate measurements:
- bid/ask spread,
- depth by distance from mid,
- depth asymmetry,
- quote/update rate,
- replenishment rate,
- depletion rate,
- cancellation/addition rates only if order-event semantics support them,
- price impact per unit traded,
- fill-quality proxy,
- resilience after aggressive flow,
- book recovery after sweep.

Critical rule:
Depth alone is not liquidity.

Low displayed depth with rapid replenishment may be materially different from
low depth with poor replenishment and high price impact.

Outputs:
- provision_normal
- provision_high_refresh
- liquidity_withdrawal
- asymmetric_liquidity
- fragile_liquidity
- local_state_uncertain

### Level 4 — Aggressor Flow and Price Response

Question:
Who is demanding liquidity and what is price doing in response?

Candidate inputs:
- aggressor-signed executed trades,
- trade intensity,
- delta / CVD,
- volume-at-price,
- spread,
- book response,
- price response per flow unit,
- flow persistence,
- post-sweep response.

Hypotheses:
- AGGRESSIVE_BUY_PRESSURE
- AGGRESSIVE_SELL_PRESSURE
- PASSIVE_BUY_ABSORPTION_POSSIBLE
- PASSIVE_SELL_ABSORPTION_POSSIBLE
- BUY_EXHAUSTION_POSSIBLE
- SELL_EXHAUSTION_POSSIBLE
- PRICE_DISCOVERY_UP
- PRICE_DISCOVERY_DOWN
- NO_RELIABLE_INFERENCE

"Absorption" requires effort-vs-result evidence.
Large delta alone is not absorption.

### Level 5 — Strategic Liquidity Behavior State

This level fuses Levels 0-4 while preserving disagreement.

Allowed states include:
- LIQUIDITY_PROVISION_NORMAL
- LIQUIDITY_PROVISION_HIGH_REFRESH
- LIQUIDITY_WITHDRAWAL
- LIQUIDITY_FRAGILE
- AGGRESSIVE_BUY_PRESSURE
- AGGRESSIVE_SELL_PRESSURE
- PASSIVE_BUY_ABSORPTION_POSSIBLE
- PASSIVE_SELL_ABSORPTION_POSSIBLE
- BUY_EXHAUSTION_POSSIBLE
- SELL_EXHAUSTION_POSSIBLE
- PRICE_DISCOVERY_UP
- PRICE_DISCOVERY_DOWN
- AUCTION_ACCEPTANCE_UP
- AUCTION_ACCEPTANCE_DOWN
- FAILED_AUCTION_UP_POSSIBLE
- FAILED_AUCTION_DOWN_POSSIBLE
- CROSS_MARKET_DISLOCATION
- CONFLICT
- NO_RELIABLE_INFERENCE

No state is automatically bullish or bearish in all regimes.

## 5. No timeframe voting

This layer is not:

Weekly vote
+ Daily vote
+ H4 vote
+ H1 vote
+ M5 vote

Instead it is a causal state cascade.

Broad context constrains interpretation of local evidence.
Local evidence can update but not rewrite the historical broad context.

Example:

Strategic: event-risk elevated
Structural: price at prior weekly accepted-value edge
Auction: London accepted above prior-day high
Local: ask depth low but replenishment fast
Flow: aggressive buying strong with low price impact

Possible interpretation:
- liquidity is still being replenished,
- low displayed depth alone does not imply withdrawal,
- breakout thesis is not proven by depth alone.

## 6. LiquidityBehaviorSnapshot contract

Identity:
- schema_version
- snapshot_id
- event_time_frontier
- instrument_scope
- source_scope
- behavior_model_version

References:
- market_snapshot_id
- data_quality_snapshot_id
- structural_snapshot_id
- auction_snapshot_id
- microstructure_snapshot_ids[]
- macro_context_snapshot_id: nullable

Level outputs:
- strategic_risk_state
- structural_liquidity_state
- auction_state
- local_liquidity_state
- aggressor_flow_state

Final hypothesis:
- primary_behavior_state
- alternative_behavior_states[]
- state_probabilities or calibrated scores when validated
- uncertainty
- conflict_state

Evidence:
- supporting_evidence[]
- contradicting_evidence[]
- missing_capabilities[]
- provenance_refs[]
- freshness_by_source
- capability_quality

Integrity:
- feature_lineage_hash
- artifact_hash

## 7. Integration with GTG

Updated flow:

Market / Macro / Cross-market Data
-> Data Quality + Point-in-Time State
-> Causal Features
-> Regime Context
-> Strategic Liquidity & Market-Behavior Layer
-> Event Detector
-> Evidence Engine
-> GTG Domain Core

The layer has two outputs:

A. event_context
Used by Event Detector to detect meaningful state changes/interactions.

B. behavior_evidence
Used by Evidence Engine as structured evidence.

The same underlying raw feature cannot be counted twice as independent evidence.

## 8. Anti-double-counting rule

A feature-lineage graph is mandatory.

Example:
OKX aggressor trades
-> delta
-> CVD
-> absorption hypothesis

The Evidence Engine cannot count:
- delta,
- CVD,
- absorption

as three independent votes if they arise primarily from the same source/flow.

Each evidence family carries lineage IDs.
Fusion logic must understand shared ancestry.

## 9. Capability tiers

### Tier A — Price/Bar only
Can support:
- structure,
- acceptance/rejection proxies,
- volatility,
- causal profiles only if volume semantics are valid.

Cannot support:
- true aggressor flow,
- order-book behavior,
- replenishment,
- venue footprint.

### Tier B — Executed Trades
Can add:
- trade intensity,
- aggressor delta when side semantics are valid,
- venue CVD,
- venue volume-at-price,
- effort-vs-result hypotheses.

### Tier C — Book Snapshots
Can add:
- displayed depth,
- asymmetry,
- spread,
- snapshot-level liquidity state.

Cannot reliably infer:
- cancellation/addition dynamics,
- queue behavior,
- true replenishment rates

unless sequential event integrity is sufficient.

### Tier D — Sequenced Order/Book Events
Can add:
- additions/cancellations,
- quote refresh,
- replenishment/depletion,
- resilience,
- queue/order-flow dynamics,
- stronger spoofing-related research definitions if rigorously specified.

### Tier E — Cross-market institutional context
Can add:
- futures OI,
- basis,
- options/volatility,
- verified macro releases/vintages,
- related hedging-market stress.

Missing higher tier data never gets fabricated from a lower tier.

## 10. Source-specific truth rules

Dukascopy XAUUSD:
- price/tick and quoted-size activity,
- no global executed gold volume claim,
- no centralized book claim.

Biquote / indicative spot:
- price reference,
- no market-maker microstructure claim.

OKX XAU-USDT-SWAP:
- venue-specific derivative microstructure,
- never global OTC XAUUSD order flow.

OKX XAUT-USDT:
- tokenized-gold venue behavior only.

Bitfinex XAUT/USD:
- tokenized-gold executed trades/raw-book behavior only.

Future CME/COMEX-quality feed:
- exchange-specific GC microstructure,
- still not identical to decentralized OTC spot gold.

## 11. Hypothesis definitions must be preregistered

Examples:

### Liquidity withdrawal hypothesis
Must specify measurable conditions involving more than low depth alone, such as:
- worsening price impact,
- weak replenishment/resilience,
- wider spread,
- reduced displayed depth,
- source quality healthy.

### Passive absorption hypothesis
Must specify:
- meaningful aggressive flow,
- limited directional price progress,
- opposing liquidity persistence/replenishment where observable,
- causal observation window,
- source coverage complete enough for the claim.

### Price discovery hypothesis
Must specify:
- persistent flow,
- directional price response,
- acceptance beyond structural reference,
- no evidence of immediate rejection,
- source/market context.

### Failed auction hypothesis
Must specify:
- excursion beyond reference,
- failure to achieve acceptance,
- return into prior value/structure,
- causal volume/flow evidence if used.

No visual hindsight relabeling is allowed after outcome is known.

## 12. Failure-mode review

### F1 — Narrative hindsight
Risk:
Any chart can be explained after the move.

Control:
Predefine state rules, timestamps and labels before outcome.

### F2 — Low depth misread as low liquidity
Risk:
Fast quote replenishment can make shallow displayed depth resilient.

Control:
Use depth + refresh/resilience + price impact + spread, not depth alone.

### F3 — Partial-book false absorption
Risk:
Limited book/trade history may look like absorption.

Control:
Coverage state; incomplete windows cannot produce decision-eligible absorption.

### F4 — Proxy contamination
Risk:
XAUT behavior gets presented as XAU behavior.

Control:
Instrument/venue-specific states; cross-market evidence remains proxy-labeled.

### F5 — Sequence gap
Risk:
Book state becomes fictitious after missing updates.

Control:
Gap => invalidate local dynamic state until resynchronization.

### F6 — Spoofing overclaim
Risk:
Large visible orders disappear and are called manipulation.

Control:
No spoofing label without event-level order evidence and a preregistered definition.

### F7 — Session/DST error
Risk:
Wrong session boundaries change auction context.

Control:
Exchange/session calendars, timezone-aware timestamps, DST tests.

### F8 — Look-ahead profile
Risk:
Historical session POC/value area calculated using later session volume.

Control:
Point-in-time rolling reconstruction only.

### F9 — Double counting
Risk:
Delta/CVD/footprint/absorption multiply the same evidence.

Control:
Feature lineage + family-aware fusion + ablation.

### F10 — Cross-market lag illusion
Risk:
A proxy venue appears predictive due to timing/clock mismatch.

Control:
Clock alignment, available_at semantics, lead/lag tests with strict causality.

### F11 — Overfit state taxonomy
Risk:
Too many attractive labels create sparse, untestable states.

Control:
Start with minimal state set; promote new labels only after incremental validation.

### F12 — False inventory inference
Risk:
Observed quoting behavior is interpreted as known dealer inventory.

Control:
Inventory remains latent/unobserved unless a valid direct source exists.
Use "inventory-rebalancing-consistent behavior" only as a research hypothesis if ever added.

Result: PASS WITH ARCHITECTURAL CONTROLS.

## 13. Validation plan

The whole layer must prove incremental value.

Tests:
1. baseline GTG without behavior layer,
2. GTG + strategic/structural context,
3. + auction/session context,
4. + local liquidity,
5. + aggressor flow,
6. full behavior layer.

Use:
- ablation,
- walk-forward,
- untouched OOS,
- forward shadow,
- calibration,
- selective risk/coverage,
- regime/session stability.

A sub-layer that does not improve validated evidence does not survive merely because
it sounds sophisticated.

## 14. Golden architecture tests

L1. Same historical event replay -> identical LiquidityBehaviorSnapshot hash.
L2. No future profile/session data enters past snapshot.
L3. Missing Tier C/D data cannot yield book-dynamic claims.
L4. XAUT evidence remains XAUT-labeled through final DecisionArtifact.
L5. Sequence gap invalidates affected dynamic state.
L6. Low depth + high replenishment does not automatically map to withdrawal.
L7. Large delta + large price progress does not map to absorption.
L8. Large delta + no price progress is only an absorption candidate until required opposing-liquidity evidence is satisfied.
L9. Same raw lineage cannot count as independent votes.
L10. Human-readable explanation cannot introduce a state absent from structured evidence.
L11. No "manipulation" / "trap" claim exists in machine decision vocabulary.
L12. Removing the behavior layer returns GTG to a valid baseline architecture.

## 15. Architecture impact

New GTG Domain Core input:
- LiquidityBehaviorSnapshot

New evidence family:
- strategic_liquidity_behavior

New Event Detector context:
- behavior state transitions,
- structural-zone interaction,
- auction acceptance/rejection,
- liquidity deterioration/replenishment,
- flow-vs-price-response divergence.

No new infrastructure service is required by this architecture addition.

It is a domain layer inside the modular GTG system.

## 16. Freeze conclusion

This layer does not justify Architecture Freeze by itself.

It has been integrated and failure-reviewed.
GTG Architecture v1 remains:

READY FOR FINAL FREEZE REVIEW

not FROZEN.
