# GTG Microstructure Forward Collection v0.1

Status: ACTIVE RESEARCH COLLECTION PROTOCOL
Registered: 2026-10-04
Branch: codex/gtg-microstructure-forward-2026-10-04
Base: feat/ahmed-toolbox-xau @ 6086611d47a476a677e0161036bcbf6647fbfe6c

## Purpose

Create a durable, append-only forward dataset of gold microstructure evidence that is materially independent from GTG's price-only research.

This collector is research-only. It has no trading authority and does not alter PanWatch Production behavior.

## Sources

Reuse existing PanWatch public providers only:

1. OKX XAU-USDT-SWAP
   - executed trades with aggressor side
   - bid/ask/last
   - order book
   - open interest when available
   - 24h venue volume when available

2. OKX XAUT-USDT spot
   - executed trades with aggressor side
   - bid/ask/last
   - order book
   - 24h venue volume when available

3. Bitfinex XAUT/USD
   - executed trades with taker direction from signed amount
   - bid/ask/last
   - raw R0 book

These are centralized proxy markets. They are never described as global OTC XAUUSD order flow.

## Storage isolation

The research database is separate from Production XAU paper/replay storage:

- default: DATA_DIR/gtg_microstructure_forward.sqlite3

It must not replace, reconcile with, or become primary for:
- paper_store
- gold_market_tape.sqlite3
- Production local_primary storage

No Production storage boundary is modified.

## Persistence contract

The dataset is append-only.

Tables:
- capture_batches
- venue_snapshots
- venue_trades

No pruning is allowed in this research store.

Executed trades are deduplicated by:
- venue + trade_id

Each venue snapshot stores:
- capture ID
- source venue/instrument
- observed timestamp
- bid / ask / last
- OI / 24h volume when supplied by provider
- compressed normalized order-book payload
- normalized snapshot SHA256
- count of trades seen
- count of book rows
- overlap count with previously stored trade IDs
- continuity status

## Book sampling

Default collector settings:
- polling interval: 30 seconds when loop mode is explicitly used
- OKX book depth: 100
- Bitfinex raw book length: 100

The collector does not run automatically merely because PanWatch starts.

Default CLI behavior is one capture only.

## Trade bootstrap

Optional OKX historical warm start is allowed only through the existing public
history-trades endpoint, with a finite user-supplied page count.

Default:
- no historical bootstrap

A bootstrap is metadata-labeled and does not imply full historical coverage.

Bitfinex REST snapshot history is limited to the provider's returned recent trades.
No claim of full historical coverage is allowed.

## Continuity metadata

For each venue capture:

- BOOTSTRAP: no earlier trade exists in research store
- OVERLAP_CONFIRMED: at least one trade ID in this capture already existed
- NO_OVERLAP_GAP_RISK: prior trades exist but this capture contains no previously stored trade ID
- NO_TRADES: provider snapshot has no usable trades

NO_OVERLAP_GAP_RISK does not prove a gap; it forbids claiming complete tape coverage for that interval.

## Integrity

For every venue snapshot compute SHA256 over the canonical normalized payload:
- quote
- venue metadata
- trades in the provider snapshot
- order-book rows
- OI / venue volume when available

The DB stores the SHA256 and the compressed book payload.

The collector may later export manifests without decoding future GTG price OOS.

## Research separation

This microstructure dataset starts forward from collector activation.

It must not be retroactively aligned to hidden GTG Pristine Forward price outcomes until a separate analysis protocol is registered.

The existing GTG Pristine Forward BID/ASK dataset remains sealed.

Historical Holdout remains locked.

## Maturation gate before strategy research

Do not fit or select a microstructure trading rule until BOTH are true:

1. at least 30 calendar days of collection exist, and
2. at least 30 independent GTG transition/resumption decision events can be joined causally to collected microstructure snapshots.

Before that point, only:
- source health
- coverage
- continuity
- storage integrity
- schema validation

may be evaluated.

## Safety / Production

- No live order routing.
- No Production deployment.
- No automatic scheduler integration in v0.1.
- No modification to production/panwatch-stable.
- No change to existing local-primary storage.
- External source outage must fail soft per capture and must not affect PanWatch startup.

## Acceptance for collector v0.1

Collector implementation is accepted when:

1. unit tests pass for append-only storage, deduplication, SHA stability, and continuity states;
2. one real smoke capture succeeds for at least two venues;
3. persisted DB can be reopened and coverage read back;
4. a second real smoke capture does not duplicate existing trades;
5. no Production files/branches are changed.

This acceptance says only that collection works. It says nothing about trading edge.
