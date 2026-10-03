# GTG v1.0 — First One-Month Historical-Core Result

Status: research-only. This is the first fixed one-month architecture/hypothesis test, not proof of edge.

## Frozen test

- Strategy revision: `f1929bf49716f2682300078de6a19bec4fde3f35`
- Test window: 2026-08-23 <= t < 2026-09-23 UTC
- Historical warm-up: 2025-11-23 <= t < 2026-08-23 UTC
- Source: frozen Dukascopy XAUUSD native M1 BID/ASK midpoint dataset
- M1 bars loaded: 437,580
- Data integrity: PASS
- Lookahead in trade decision: false
- Outcome tuning: false

## Primary GTG price-ladder hypothesis

Definition: PRICE becomes closed beyond both MA14 and MA50, with MA200 ahead in the same
direction.

| Timeframe | Events | MA200 hit rate | Median time to hit |
|---|---:|---:|---:|
| 1m | 272 | 73.53% | 7 min |
| 5m | 258 | 65.50% | 24 min |
| 15m | 86 | 82.56% | 92 min |
| 1h | 27 | 66.67% | 240 min |
| 4h | 5 | 60.00% | 2280 min |

The user's core observation is directionally supported in this month, especially on 15m and
1m, but the data does not support a universal 90% rule. Higher-timeframe events are deduplicated
so one closed source-timeframe bar counts once.

## Secondary MA14/MA50 cross diagnostic

This is explicitly separate from price breaking the MA14/MA50 pair.

| Timeframe | Events | MA200 hit rate | Median time to hit |
|---|---:|---:|---:|
| 1m | 57 | 71.93% | 17 min |
| 5m | 58 | 70.69% | 20 min |
| 15m | 23 | 91.30% | 75 min |
| 1h | 5 | 80.00% | 126.5 min |
| 4h | 1 | 0.00% | n/a |

The 4h zero comes from only one independent event and is not promoted as a general conclusion;
it requires broader validation and inspection of target geometry/horizon assumptions.

## Accepted MA200 -> MA1000

Definition: two consecutive closes beyond MA200 after a prior close on the opposite side,
with MA1000 ahead.

| Timeframe | Events | MA1000 hit rate | Median time to hit |
|---|---:|---:|---:|
| 1m | 71 | 63.38% | 66 min |
| 5m | 86 | 55.81% | 82.5 min |
| 15m | 38 | 63.16% | 221 min |
| 1h | 7 | 71.43% | 829 min |
| 4h | 3 | 33.33% | 290 min |

## MA barrier behavior (M5)

- MA200 rejection -> one ATR before re-cross, 120m:
  - events: 263
  - hit rate: 34.60%
  - median hit time: 6 min
- MA1000 rejection -> one ATR before re-cross, 120m:
  - events: 82
  - hit rate: 34.15%
  - median hit time: 6.5 min

A raw touch/rejection is therefore not a sufficient GTG trigger. It needs confirmation from
regime, structure, RSI, volume/location and live microstructure.

## First fixed GTG execution test

- Market scans: 8,917
- Qualified setups: 39
- Executed trades: 37
- Win rate: 54.05%
- Target rate: 54.05%
- Profit factor: 1.608
- Mean net result: +5.105 bps/trade
- Median net result: +14.676 bps/trade
- Cumulative net result: +188.900 bps
- Max drawdown: -72.660 bps
- Median duration: 22 min

### By setup

- Price 14/50 -> 200:
  - 8 trades
  - 25.0% wins
  - PF 0.419
  - -56.810 cumulative bps
- Accepted 200 -> 1000:
  - 2 trades
  - 100% wins
  - +47.119 cumulative bps
  - sample far too small for inference
- MA barrier rejection with GTG confluence:
  - 27 trades
  - 59.26% wins
  - PF 1.933
  - +198.591 cumulative bps

The event-level MA ladder observation and the execution setup are different questions:
price often reaches MA200 after the 14/50 break, but entering immediately on that break was
not profitable in this one-month execution sample. GTG therefore must distinguish TARGET
probability from ENTRY quality.

### By direction

- Long: 19 trades, 63.16% wins, PF 2.157, +153.151 bps
- Short: 18 trades, 44.44% wins, PF 1.200, +35.748 bps

### By session

- Asia: 12 trades, 58.33% wins, PF 1.987
- London: 5 trades, 20.0% wins, PF 0.431
- London/New York overlap: 7 trades, 71.43% wins, PF 2.826
- New York: 10 trades, 40.0% wins, PF 1.179
- Off-session: 3 trades, 100% wins; sample too small for inference

## Historical layer coverage

Covered:
- RSI
- MA dynamics
- deterministic price structure / supply-demand / order-block proxy
- session model
- volume profile using Dukascopy activity proxy
- daily macro proxies (DXY, US10Y, VIX, SPX, oil)

Unavailable and NOT fabricated:
- historical heatmap/order book
- centralized footprint
- delta/CVD from centralized tape
- historical centralized cash/order flow
- exact historical event-calendar surprise/reaction pipeline

## Current conclusion

GTG v1.0 passed its first architecture test and produced a positive one-month execution result,
but `edge_proven=false` remains correct.

The strongest finding is not the PnL. It is that the MA ladder must be treated as a
timeframe-dependent target-probability model, while entry requires separate confirmation.
The next scientifically valid step is out-of-sample expansion across the remaining frozen
months without changing these rules based on this month's outcomes.
