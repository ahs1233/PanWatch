# GTG Event-Conditioned Directional Experience v3

Status: **research / shadow only**

GTG v3 is a new experience layer beside the frozen deterministic GTG core,
GTG Deep Experience v1, and GTG Directional Experience v2. It does not replace
or delete any earlier version.

## Research question

v2 showed that generic UP/DOWN/NEUTRAL prediction did not generalize. v1 showed
that the narrower question "does MA200 become a destination from this state?"
was materially more learnable.

v3 therefore asks a conditional question: given a specific GTG event, what is
the independently learned probability that the bullish or bearish destination
succeeds, what adverse/favorable path occurs first, and when should the model
abstain?

Initial events:

- bullish_14_50_to_200
- bearish_14_50_to_200

The transition is a causal price move through the MA14/MA50 cluster toward a
fixed MA200 destination. The already-established v1 MA200 distance support
(0.5 to 6 ATR) is reused. MA slope, acceleration, compression, RSI, volatility
and timeframe relationships are features, not newly hand-tuned event
thresholds.

## Causal feature contract

Decision grid: M5.

Context is built from M1, M5, M15 and H1.

Every source feature is aligned by **bar availability time**, not bar-open time.
For example, a 10:00 H1 candle is first available at 11:00. It cannot enter a
10:30 decision.

The initial v3 benchmark defers H4 so that the first experiment isolates
event-conditioning without adding a much longer warm-up and compute surface.
H4 can be added as a separate ablation after the initial event experts are
measured.

Feature families include price returns/candle geometry, RSI level/slope/
acceleration, MA14/22/50/200 geometry on every timeframe, MA1000 context on M5,
MA slope/acceleration, MA spacing/compression, realized volatility, observed
Dukascopy volume activity, and UTC session cyclic encoding.

Normalization is fitted on the training period only.

## Labels

For each event, MA200 is frozen at the event timestamp.

Within the next 48 M5 bars (4 hours), v3 learns:

- MA200 success probability;
- directional MFE in ATR;
- directional MAE in ATR;
- whether a 1 ATR research adverse barrier is reached before MA200;
- time-to-MA200;
- clean path = target reached without the research adverse barrier first;
- signed trend strength.

If MA200 and the research adverse barrier first touch inside the same M5 candle,
the outcome is ambiguous and is masked.

The 1 ATR adverse barrier is a **research path label**, not a stop loss and not
an execution rule.

## Independent experts

Bullish and bearish experts are completely separate model instances.

It is valid to observe P(up success)=0.31 and P(down success)=0.28. There is no
identity such as P_down = 1 - P_up.

Architectures compared on the same features and labels:

- GRU;
- causal TCN;
- patch Transformer.

Champion selection uses validation only.

## Independent abstention engine

A separate GRU predicts only whether the event path is tradeable/clean versus
ABSTAIN. It receives event direction as a context channel but does not choose
direction.

The selective gate combines validation-calibrated event success probability
with validation-calibrated tradeability probability. Coverage is reported
explicitly. The gate cannot claim improvement by silently reducing the sample
to a trivial number.

## Event-conditioned analog memory

Generic v1 analog memory remains rejected.

v3 analog retrieval is restricted to the same event and same direction because
UP and DOWN experts have separate banks. It is then bucketed by UTC session and
a training-derived volatility tercile.

The benchmark compares neural only, analog only, and a validation-selected
neural + analog blend. The analog weight can remain zero.

## Frozen evaluation protocol

Dataset: gen1-xau-2y-dataset-v2-20240923-20260923

Expected: 24 Dukascopy XAUUSD shards covering 730 calendar days.

Temporal split:

- Train: 2024-09-23 -> 2025-07-23
- Validation/calibration: 2025-07-23 -> 2025-09-23
- Untouched OOS: 2025-09-23 -> 2026-09-23

Forbidden: random split, OOS architecture selection, OOS threshold selection,
OOS normalization, best-seed selection, and deletion of bad months.

## Metrics

Each independent expert reports accuracy, balanced accuracy, precision/recall/
F1, MCC, ROC-AUC, PR-AUC, Brier/Brier skill, ECE, selective precision,
coverage/no-trade rate, MFE/MAE absolute error, time-to-target error, and
month/session/volatility stability.

## Initial promotion gate

Passing this gate does **not** grant decision authority. It only makes an expert
eligible for a deterministic GTG ablation.

Initial criteria:

- positive untouched-OOS Brier skill;
- OOS MCC >= 0.05;
- validation-selected selective precision at least 3 percentage points above
  the training event base rate;
- OOS selective coverage >= 10%;
- a reasonable number of OOS months with both positive MCC and positive Brier
  skill;
- MFE and MAE prediction both beat a training-median path baseline.

Only an eligible expert proceeds to frozen deterministic GTG ablation and then
walk-forward/rolling robustness.

## Bundle/runtime safety

Every expert bundle carries model/schema/label versions, dataset ID, split
ranges, git commit, seed, normalization, calibration, architecture and
threshold policy. Runtime rejects a feature-schema mismatch.

Missing bundle, failed load, malformed bundle, NaN input, or schema mismatch is
a v3 failure and must fall back to deterministic GTG. v3 has no API for entry,
position sizing, stops, risk, or order routing.

## Microstructure boundary

The frozen Dukascopy history does not contain centralized CME L2/L3 order book,
true aggressor-side footprint, or heatmap history. v3 therefore does not invent
footprint, delta, CVD, or depth from OHLC.

A future Microstructure Expert should consume venue-qualified GC/MGC data
(trades/aggressor side and depth/MBO/MBP when licensed/available) and be tested
forward separately before any fusion with the historical v3 benchmark.

## Economic-data boundary

PanWatch already contains economic evidence semantics, but the frozen two-year
dataset does not bundle a causally timestamped historical release feed for the
same period. Initial Event A/B therefore excludes macro-event actuals rather
than backfilling them with leakage.

When added, actual values are valid only after their publication timestamp.

## Self-supervised learning

TS2Vec-like contrastive/masked temporal pretraining remains a planned ablation,
not a default dependency. The first v3 run isolates whether event-conditioning
itself improves OOS behavior using a bounded native PyTorch surface. If an
expert establishes OOS signal, self-supervised pretraining can then be measured
against the same frozen protocol rather than assumed to help.
