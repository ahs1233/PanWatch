# GTG v1.0 — Gen1 Trad Gold Method

Status: research methodology, fixed before the first GTG monthly test.

## Core principle

GTG reads gold as a layered state machine. No single indicator is a signal by itself.
A trade requires agreement between location, direction, movement dynamics, liquidity/volume
context, execution microstructure, macro context, and risk.

## 1. RSI

- RSI(14) is a momentum/pressure confirmation, not a standalone overbought/oversold trigger.
- Read RSI by timeframe and by regime.
- In trend continuation, RSI holding the trend-side of 50 matters more than static 70/30.
- Divergence is contextual evidence only.

## 2. MA Dynamics

Primary moving averages: 14, 22, 50, 200, 1000.

The same MA number has different meaning on each timeframe. GTG therefore records for every
available timeframe:

- level
- slope / direction
- normalized slope / speed
- slope change / acceleration
- spacing between averages / momentum expansion
- compression and expansion
- price distance from each MA
- crosses
- acceptance and rejection
- next MA target in the direction of travel

### MA ladder hypothesis

The primary GTG price-ladder hypothesis is:

1. When PRICE crosses/reclaims the MA14 and MA50 pair and becomes established on their new
   side, MA200 can become the next major target when MA200 lies ahead in that direction and
   the higher-timeframe regime does not contradict the move.
2. After PRICE breaks MA200 and is accepted beyond it, MA1000 can become the next structural
   target when MA1000 lies ahead in the same direction.
3. Any MA can simultaneously be a target and a dynamic barrier. A target that rejects price
   becomes support/resistance; a target that is broken and accepted can become a transition
   or launch level.
4. MA14 crossing MA50 is recorded separately as a secondary MA-dynamics event. It is NOT the
   same event as price breaking/reclaiming the 14/50 pair.

Historical GTG defines a 14/50 price break as the first closed bar whose close is beyond BOTH
MA14 and MA50 after the previous closed bar was not beyond both in that direction. A wick
alone does not count.

"Acceptance" beyond MA200 is stronger than a wick: two consecutive closed bars on the new side
after a prior close on the opposite side.

## 3. Price Structure

GTG maps:

- important swing highs/lows
- support/resistance zones
- supply/demand zones
- break of structure
- liquidity sweeps
- algorithmic order blocks

Historical order blocks must be deterministic: the last opposite candle before an ATR-qualified
displacement that breaks recent structure. No hindsight drawing is allowed.

## 4. Liquidity and Volume Map

Live GTG uses:

- heatmap
- volume profile
- POC
- VAH / VAL
- HVN / LVN
- visible liquidity concentrations and voids

Historical Dukascopy volume is only quoted/tick activity proxy. It must never be labeled as
centralized global gold traded volume.

## 5. Execution Microstructure

Live GTG uses centralized/venue-qualified evidence where available:

- footprint
- delta / CVD
- bid/ask imbalance
- absorption
- aggressor flow
- order-book liquidity
- cash/order flow

For gold, venue provenance is mandatory. Spot XAUUSD is OTC; a broker feed is not "the whole
gold market". Centralized futures/venue data should be identified explicitly.

Historical monthly tests must mark this layer UNAVAILABLE when raw historical book/tape is not
present. It is forbidden to synthesize fake footprint or order flow from OHLC.

## 6. Macro / Economic Layer

GTG considers:

- CPI / PCE
- NFP, unemployment, wages
- FOMC / Fed communication and policy expectations
- policy rates
- US 2Y / 10Y yields and real-yield context
- DXY
- GDP / PMI / retail sales and other material releases
- major geopolitical safe-haven catalysts

The framework evaluates expectation -> release -> cross-asset reaction -> gold reaction.
Historical proxy series may be used only if labeled as proxies; they do not replace an event
calendar.

## 7. Market Regime and Multi-Timeframe Engine

Each decision starts by classifying the environment:

- trend
- range
- compression
- expansion
- reversal / transition

GTG reads bottom-up execution frames together with top-down context. The MA ladder is never
interpreted without its timeframe.

Initial research frames:
- execution: 1m / 5m
- confirmation: 15m
- regime: 1h / 4h

Daily context can be used, but MA1000 on Daily cannot be validated by the current two-year
dataset because it needs more than 1000 daily bars.

## 8. Session / Time Model

Record Asia, London, New York, overlap, and off-session behavior. A setup must be evaluated
by session because intraday gold liquidity and speed are not stationary through the day.

## 9. Decision Engine

GTG follows:

Macro -> Regime -> MA Dynamics -> Price/Zone Location -> Liquidity/Volume Map ->
Microstructure Trigger -> Entry -> Invalidation -> Risk -> Management -> Feedback.

Live execution requires a current trigger. Context alone is not an entry.

Research setup families for the first fixed test:

1. MA price-ladder continuation: price breaks/reclaims the 14/50 pair toward 200.
2. MA price-ladder continuation: accepted price break of 200 toward 1000.
3. MA barrier rejection: 200 or 1000 rejects price with contextual confirmation.

The first monthly test must not optimize thresholds from outcomes.

## 10. Risk and Trade Management

- One position at a time in the first research benchmark.
- Stop is derived from structure and ATR, with explicit minimum and maximum distance.
- Position sizing is defined from risk budget / stop distance, never a fixed lot assumption.
- Same-minute target+stop ambiguity is scored conservatively as stop.
- Every trade has a maximum holding time.
- No trade is valid without explicit invalidation.

## 11. Validation

Every hypothesis is measured separately from trading PnL.

Minimum outputs:
- event count
- target hit rate
- time to target
- false-break / rejection behavior
- trade count
- win rate
- mean / median net bps
- profit factor
- max drawdown
- result by setup
- result by direction
- result by session
- coverage of every GTG layer
- explicit missing-data flags

No result from one month proves an edge. The one-month run is an architecture and hypothesis
test before extending to the full frozen two-year dataset.

## First test period

Evaluation month: 2026-08-23 <= t < 2026-09-23 (frozen shard m24).

Warm-up uses earlier frozen shards only and is excluded from performance. This supplies enough
history to calculate MA1000 through the 4h regime frame without future leakage.

Dataset source: frozen Dukascopy XAUUSD BID/ASK M1 research dataset already published by
PanWatch under tag `gen1-xau-2y-dataset-v2-20240923-20260923`.
