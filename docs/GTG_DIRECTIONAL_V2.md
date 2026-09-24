# GTG Directional Experience v2

Status: **shadow research only**

## Objective

GTG v2 learns the shape of upward and downward gold moves directly rather than
treating direction as a secondary binary head.

The design is intentionally separated into:

1. `gtg_directional_data.py` — causal feature and label semantics.
2. `gtg_directional_v2.py` — model registry and calibrated inference only.
3. `benchmarks/gtg_directional_v2/run.py` — temporal split, training, model
   selection, OOS evaluation, artifacts.
4. dedicated workflow — reproducible training on the frozen two-year XAU dataset.

Execution/risk code does not import training code.

## Research basis

The implementation takes ideas from, but does not vendor or directly depend on:

- DeepLOB — UP / STATIONARY / DOWN market movement classification:
  https://arxiv.org/abs/1808.03668
- Triple-barrier labeling — volatility-aware upper/lower/vertical outcomes:
  https://mlfinpy.readthedocs.io/en/latest/Labelling.html
- Trend-scanning — direction plus trend-strength t-statistic:
  https://mlfinpy.readthedocs.io/en/latest/_modules/mlfinpy/labeling/trend_scanning.html
- TimesNet — adaptive multi-period temporal variation:
  https://openreview.net/forum?id=ju_Uqw384Oq
- PatchTST — patch-based temporal Transformer representation:
  https://arxiv.org/abs/2211.14730
- TS2Vec — hierarchical contrastive time-series representations:
  https://arxiv.org/abs/2106.10466

We implement small native models instead of importing research repositories into
production.  This bounds dependency and supply-chain risk.

## Direction semantics

Each M5 state is labeled independently at:

- 30 minutes: +/- 0.75 ATR
- 60 minutes: +/- 1.00 ATR
- 120 minutes: +/- 1.40 ATR
- 240 minutes: +/- 2.00 ATR

Class:

- DOWN: lower barrier touched first.
- UP: upper barrier touched first.
- NEUTRAL: neither touched before the vertical horizon.
- AMBIGUOUS: both touched in the same M5 bar before order can be known; masked
  from training rather than guessed.

Barrier width grows with horizon to avoid the long horizons degenerating into an
almost-always-active label.

## Path learning

Direction alone is insufficient.  v2 also learns over the four-hour path:

- maximum upward excursion in ATR;
- maximum downward excursion in ATR;
- future trend slope t-statistic, scaled and clipped;
- normalized first-touch time for directional barriers.

This explicitly teaches the difference between:

- a clean rise;
- a rise after a deep adverse pullback;
- a noisy range that finishes higher;
- a strong downside cascade.

## Features

Causal-only sequence: 96 x M5 bars = 8 hours.

Features include:

- log returns at 1/3/12/24 bars;
- ATR-normalized candle body/range/wicks;
- close location inside candle;
- quoted/tick-activity z-score;
- RSI14;
- distance to MA14/22/50/200/1000;
- MA slopes;
- MA acceleration for 14/50/200;
- MA spacing;
- realized volatility;
- cyclic time-of-day.

Normalization is fitted only on training data.

## Architectures

All architectures use the exact same labels and features:

1. `gru` — causal recurrent baseline.
2. `patch_transformer` — temporal patches + Transformer encoder.
3. `timesnet_lite` — adaptive FFT-selected periods + 2D temporal convolution.

The architecture winner is selected only on validation data.

## Split

- Train: 2024-09-23 -> 2025-07-23
- Validation/calibration: 2025-07-23 -> 2025-09-23
- Untouched OOS: 2025-09-23 -> 2026-09-23

No random split.  Every sample is guarded so its future label ends before the
split boundary.

## Metrics

Per horizon:

- balanced accuracy;
- macro F1;
- multiclass MCC;
- multiclass Brier score and Brier skill versus training prior;
- expected calibration error;
- UP/DOWN/NEUTRAL precision and recall;
- selective high-confidence coverage and precision.

Path:

- UP MFE MAE in ATR;
- DOWN MFE MAE in ATR;
- trend-strength MAE.

A validation-selected confidence threshold is applied unchanged to OOS.

## Promotion

Even if v2 passes OOS, it remains shadow evidence.

The next gate is an ablation against frozen deterministic GTG:

`GTG alone` versus `GTG + directional shadow evidence`

using identical risk, stop, target and costs.

Only incremental OOS improvement in profit factor/drawdown can justify decision
weighting.  The neural layer never receives independent order-routing authority.
