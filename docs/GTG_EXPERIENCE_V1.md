# GTG Deep Experience Layer v1

Status: shadow research candidate. It does not alter frozen GTG v1.0 decisions yet.

## Purpose

The Experience Layer gives GTG learned pattern recognition in addition to deterministic rules.
It is designed to approximate "experience" as measurable statistical memory:

1. **Neural temporal encoder**: reads a causal sequence of market states rather than one candle.
2. **Multi-task learning**: learns several market questions at once instead of a single price forecast.
3. **Latent pattern embedding**: compresses each market state into a learned representation.
4. **Analog memory**: retrieves historically similar learned states and summarizes what followed.
5. **Uncertainty**: probabilities near 50% are treated as uncertain, not forced into a trade.
6. **Shadow-first promotion**: the learned layer cannot change risk or execution until OOS evidence proves incremental value.

## Data split

No random train/test split is allowed.

- Training: 2024-09-23 -> 2025-07-23
- Calibration: 2025-07-23 -> 2025-09-23
- Untouched OOS test: 2025-09-23 -> 2026-09-23

The second year that was used for the GTG annual benchmark is never used to fit neural weights.

## Input sequence

Default context: 96 x 5-minute closed bars (8 hours).

Causal features include:

- 1/3/12/24 bar log returns
- candle body/range/wicks normalized by ATR
- volume-activity z-score
- RSI14
- price distance from MA14/22/50/200/1000
- slope/speed of MA14/22/50/200/1000
- MA spacing: 14-50, 50-200, 200-1000
- cyclical time-of-day representation

Future features are forbidden.

## Learned tasks

The first model is multi-task:

- next_move_up: which 1-ATR barrier is reached first
- expansion: whether the next two hours produce >=1.5 ATR excursion
- ma200_touch: probability the MA200-at-event is reached within four hours
- ma1000_touch: probability the MA1000-at-event is reached within eight hours
- ma200_rejection: probability an MA200 interaction moves one ATR away before recross
- ma1000_rejection: same for MA1000

These outputs are **context probabilities**, not direct orders.

## Architecture

GTG Experience v1 uses a causal two-layer GRU encoder, a compact latent embedding, and separate
binary heads for each task. The embedding is also used for historical analog retrieval.

Why a sequence encoder:
- GTG's core hypothesis depends on speed, momentum, compression/expansion and MA dynamics.
- Those are path-dependent; a snapshot loses part of the information.
- A causal recurrent encoder can learn temporal shape without seeing future bars.

## Analog memory

The learned embedding represents the current market state. GTG can compare this vector against a
bank of training-period embeddings and retrieve nearest historical states.

The analog layer can answer:
- How often did similar states move up first?
- How often did similar states reach MA200?
- Did similar states expand or fail?
- What was the dispersion of outcomes?

Analog memory is advisory and is never allowed to retrieve examples from the future relative to
the evaluation timestamp.

## Promotion gates

The neural layer remains shadow-only until all of the following are satisfied:

1. Positive OOS Brier skill versus unconditional base rate on useful heads.
2. Stable walk-forward results rather than one favorable regime.
3. Calibration does not collapse between train/validation/OOS.
4. An ablation shows GTG + Experience improves execution quality versus frozen GTG alone.
5. Profit factor/drawdown improve after costs without reducing sample size to a trivial subset.
6. Live forward shadow test confirms no data-source mismatch.

Deep learning is a pattern-recognition layer, not a substitute for:
- risk control
- invalidation
- venue-qualified footprint/order flow
- macro event context
- deterministic safety gates
