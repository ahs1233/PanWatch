# GTG Event v3 — Fresh Forward Shadow Contract

Status: **research / evidence collection only**

This layer exists to create the first genuinely fresh evidence after the original GTG v3 OOS year became known.

## Locked candidate

Only one path candidate is admitted:

- Side: DOWN
- Event: `bearish_14_50_to_200`
- Candidate: `knn_embedding_median_k64`
- Candidate version: `gtg-event-v3-down-path-forward-v1`
- Feature schema: `gtg-event-v3-feature-v1`

UP is intentionally excluded because no UP path candidate beat both path baselines in the diagnostic.

## Forward boundary

Fresh evidence starts at:

`2026-09-24T00:00:00Z`

Historical/backfilled events before this timestamp are rejected.

A prediction must be persisted no later than five minutes after its event timestamp. This prevents recreating predictions after the future path is already visible.

## Outcome boundary

The path horizon is four hours.

An outcome cannot be written until the complete four-hour horizon has elapsed. Predictions and outcomes are append-only:

- duplicate prediction event IDs are rejected;
- outcomes without a previously persisted prediction are rejected;
- premature outcomes are rejected;
- replacing an existing outcome is rejected.

## Evidence ledger

Implementation:

`src/modules/xau/gtg_event_v3_forward.py`

Storage:

SQLite, with separate immutable prediction and outcome tables.

Each prediction stores:

- event ID and UTC timestamp;
- record timestamp;
- label-not-before timestamp;
- expected MFE/MAE;
- locked baseline MFE/MAE predictions;
- source commit;
- candidate/version;
- feature-schema version;
- metadata;
- SHA-256 prediction hash.

## Review gate

A fresh-forward sample can become **review-ready** only when all of the following are true:

- at least 60 completed events;
- at least 30 calendar days from first to last completed event;
- candidate MFE MAE beats the locked baseline;
- candidate MAE MAE beats the locked baseline.

This is not an automatic promotion gate.

The ledger always returns:

`promotion_authorized = false`

Any later promotion requires a separate explicit governance decision after reviewing directional, calibration, path, stability, and deterministic trading-ablation evidence.

## Production boundary

This module exposes no order, sizing, stop, target-routing, position-management, or execution API.

It is not wired into Railway production by this change.
