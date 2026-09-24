# GTG Event-Conditioned Directional Experience v3 — Results

Run date: 2026-09-24
Status: **research / shadow only**
Benchmark commit: `fd86ee7f41812811c6704b10d666e7b5188e110b`

## Verification

GitHub Actions:

- Run: `35987140840`
- Job: `107592362920`
- Result: SUCCESS
- Focused regression tests: 6 passed
- Frozen dataset shards: 24/24
- Artifact: `10802936037`
- Artifact SHA256:
  `efac55866e8d26fd0b9984ed25997edb4c05929bb705cc9d9d5e850f60d74887`

The first candidate run `35986812593` stopped before expert training because
NumPy 2.5 removed `np.trapz`. The metric implementation was changed to
`np.trapezoid` with backward-compatible fallback and a regression test. Event
definitions, split policy, thresholds, and OOS labels were not changed in
response to OOS results.

## Data integrity

Dataset ID:

`gen1-xau-2y-dataset-v2-20240923-20260923`

Source:

`dukascopy:datafeed:XAUUSD:native-m1-bid-ask-mid`

Observed integrity:

- requested calendar days: 730
- data days: 729
- no-data days: 1
- M1 bars: 1,049,520
- first bar: 2024-09-23 00:00 UTC
- last bar: 2026-09-22 23:59 UTC
- crossed rows discarded: 0
- dataset SHA256:
  `21101f709d0a1eedeec98b143d5755fb5e61af0746630a00b7f0a9616561aa20`

Frozen temporal split:

- train: 2024-09-23 -> 2025-07-23
- validation/calibration: 2025-07-23 -> 2025-09-23
- untouched OOS: 2025-09-23 -> 2026-09-23

Event observations:

| Split | All | UP | DOWN |
|---|---:|---:|---:|
| Train | 761 | 305 | 456 |
| Validation | 128 | 48 | 80 |
| OOS | 993 | 446 | 547 |

Observations use a 12-M5-bar same-direction cooldown. Distinct/opposite event
horizons can still overlap, so these counts are event observations rather than
claims of statistically independent trades.

## Feature contract

Feature schema: `gtg-event-v3-feature-v1`

- 97 features
- decision grid: M5
- context: M1, M5, M15, H1
- sequence: 96 M5 decisions / 8 hours
- HTF data: closed-bar only
- normalization: fitted on train period only
- MA14 / MA22 / MA50 / MA200 across timeframes
- MA1000 context on M5
- MA distance, slope, acceleration, spacing/compression
- RSI level, slope, acceleration
- returns, candle geometry, realized volatility, observed volume activity
- session cyclic features

H4 was intentionally deferred from the first v3 benchmark.

## Architecture comparison

Parameters per event expert:

- GRU: 65,864
- Patch Transformer: 90,888
- causal TCN: 115,592

All candidate models used the same feature/label protocol. Model selection was
made from validation only. OOS was opened after the champion was frozen.

### UP architecture transparency

| Architecture | Validation selection score | Validation AUC | Validation Brier Skill | Validation MCC | OOS AUC | OOS Brier Skill | OOS MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | 1.0213 | 0.8494 | +0.04441 | 0.4213 | 0.6152 | +0.00445 | 0.2092 |
| Patch Transformer | 0.9490 | 0.7695 | +0.04181 | 0.4670 | 0.6494 | +0.00109 | 0.1279 |
| TCN | **1.0318** | 0.8203 | +0.05493 | 0.5165 | 0.6437 | +0.01415 | 0.1881 |

Validation selected TCN. The fact that another architecture has a different OOS
metric does not change the frozen selection.

### DOWN architecture transparency

| Architecture | Validation selection score | Validation AUC | Validation Brier Skill | Validation MCC | OOS AUC | OOS Brier Skill | OOS MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | **1.0938** | 0.8453 | +0.08129 | 0.5060 | 0.6867 | +0.00376 | 0.2517 |
| Patch Transformer | 1.0350 | 0.8080 | +0.06633 | 0.5101 | 0.6760 | -0.01401 | 0.2792 |
| TCN | 1.0626 | 0.8480 | +0.06853 | 0.4472 | 0.7080 | +0.00177 | 0.2640 |

Validation selected GRU. OOS was not used to switch to TCN.

## UP expert

Event: `bullish_14_50_to_200`

Champion: TCN, 115,592 parameters, 5 epochs.

Training success base rate: 58.69%.

Validation:

- Accuracy: 77.08%
- Balanced Accuracy: 72.87%
- Precision: 75.00%
- Recall: 93.10%
- F1: 83.08%
- MCC: 0.5165
- ROC-AUC: 0.8203
- PR-AUC: 0.8807
- Brier Skill: +0.05493
- ECE: 0.1581

Validation selected a 50% neural / 50% event-conditioned analog blend.

Untouched OOS final blend:

- N: 446
- event base rate: 56.50%
- Accuracy: 60.76%
- Balanced Accuracy: 60.53%
- Precision: 66.24%
- Recall: 62.30%
- F1: 64.21%
- MCC: 0.2093
- ROC-AUC: 0.6374
- PR-AUC: 0.6885
- Brier: 0.23463
- baseline Brier: 0.24625
- Brier Skill: **+0.01162**
- ECE: 0.0603

Neural-only OOS:

- ROC-AUC: 0.6437
- MCC: 0.1881
- Brier Skill: +0.01415

Analog-only OOS:

- ROC-AUC: 0.6287
- MCC: 0.1758
- Brier Skill: +0.00391

The validation-selected blend was retained even though neural-only happened to
have slightly higher OOS Brier Skill. Retuning the blend after OOS would violate
the protocol.

### UP abstention

Validation-selected gate:

- success probability >= 0.55
- tradeability probability >= 0.40
- validation coverage: 43.75%
- validation precision: 85.71%

OOS:

- selected: 44 / 446
- coverage: **9.87%**
- no-trade rate: 90.13%
- selective precision: **75.00%**
- precision lift vs train event base: +16.31 percentage points

The initial promotion policy required at least 10% OOS coverage. UP narrowly
missed this condition.

### UP path heads

- MFE prediction MAE: 1.0808 ATR
- training-median MFE baseline MAE: 0.9968 ATR
- MAE prediction MAE: 2.6279 ATR
- training-median MAE baseline MAE: 2.5135 ATR
- time-to-target MAE: 60.70 minutes
- trend-strength MAE: 0.7568

Both learned MFE and MAE heads were worse than the simple training-median path
baseline.

### UP stability

9 of 12 reported OOS months had both MCC > 0 and Brier Skill > 0.

Weak months included February 2026 and August 2026. September 2026 retained
positive MCC but slightly negative Brier Skill.

Session view:

- Asia: negative calibration/discrimination in this sample
- London: small positive Brier Skill and MCC
- New York: strongest meaningful large-sample session partition
- off-hours: too few observations for strong interpretation

Volatility view:

- high-volatility bucket: positive Brier Skill and MCC
- middle bucket: slightly negative Brier Skill
- low bucket: very small sample

## DOWN expert

Event: `bearish_14_50_to_200`

Champion: GRU, 65,864 parameters, 5 epochs.

Training success base rate: 57.68%.

Validation:

- Accuracy: 77.50%
- Balanced Accuracy: 73.33%
- Precision: 77.59%
- Recall: 90.00%
- F1: 83.33%
- MCC: 0.5060
- ROC-AUC: 0.8453
- PR-AUC: 0.8879
- Brier Skill: +0.08129
- ECE: 0.0978

Validation rejected analog fusion: selected analog weight = 0.

Untouched OOS:

- N: 547
- event base rate: 60.33%
- Accuracy: 60.15%
- Balanced Accuracy: 62.63%
- Precision: 75.23%
- Recall: 50.61%
- F1: 60.51%
- MCC: **0.2517**
- ROC-AUC: **0.6867**
- PR-AUC: **0.7806**
- Brier: 0.23628
- baseline Brier: 0.24004
- Brier Skill: **+0.00376**
- ECE: 0.1275

Analog-only OOS had MCC 0.2721 but Brier Skill -0.01421 and worse ECE. Because
validation selected analog weight zero, analog remained disabled.

### DOWN abstention

Validation-selected gate:

- success probability >= 0.70
- tradeability probability >= 0.40
- validation coverage: 23.75%
- validation precision: 94.74%

OOS:

- selected: 57 / 547
- coverage: **10.42%**
- no-trade rate: 89.58%
- selective precision: **85.96%**
- precision lift vs train event base: +28.29 percentage points

This is the strongest selective classification signal in the initial v3 run.

### DOWN path heads

- MFE prediction MAE: 1.5154 ATR
- training-median MFE baseline MAE: 1.3581 ATR
- MAE prediction MAE: 2.3357 ATR
- training-median MAE baseline MAE: 2.2435 ATR
- time-to-target MAE: 57.61 minutes
- trend-strength MAE: 0.7299

The path heads again failed to beat the simple training-median baselines.

### DOWN stability

7 of 13 reported OOS months had both MCC > 0 and Brier Skill > 0.

The late OOS period improved materially in July-August-September 2026, while
November 2025, February-March 2026, and June 2026 exposed calibration/regime
weakness.

Session view:

- New York: positive Brier Skill and MCC on the largest sample
- London: positive MCC but slightly negative Brier Skill
- Asia: promising but very small sample
- off-hours: poor calibration and too few cases

Volatility view:

- low and mid buckets: positive calibration and MCC
- high-volatility bucket: positive MCC but slightly negative Brier Skill

## Tradeability / abstention engine

Independent GRU:

- parameters: 61,589
- epochs: 5
- target: clean MA200 path without the 1 ATR research adverse barrier first

Validation:

- MCC: 0.2578
- ROC-AUC: 0.6548
- Brier Skill: +0.02243
- ECE: 0.0461

OOS:

- N: 993
- MCC: 0.1021
- ROC-AUC: 0.5748
- PR-AUC: 0.3957
- Brier Skill: +0.00390
- ECE: 0.0211

Recall is low, which is acceptable for a rejector only if the combined gate
retains useful coverage. It has no buy/sell authority.

## Adverse-first heads

The adverse-first auxiliary task was not equally successful.

UP OOS adverse-first:

- MCC: 0.0000
- ROC-AUC: 0.5561
- Brier Skill: -0.00188

DOWN OOS adverse-first:

- MCC: 0.0333
- ROC-AUC: 0.6364
- Brier Skill: +0.00582

These heads remain research outputs and are not suitable for stop-loss or risk
control.

## Comparison with earlier GTG experience

### GTG Destination Experience v1

MA200 destination remained a meaningful reference:

- OOS AUC: ~0.6292
- OOS Brier Skill: +0.00590
- precision at probability >= 70%: ~71.99%
- coverage at probability >= 70%: ~5.53%

v1 remains preserved as shadow evidence.

### GTG Directional Experience v2

Generic 60-minute direction remained not promoted:

- Balanced Accuracy: ~44.46%
- MCC: ~0.0489
- Brier Skill: ~-0.02403
- selective directional accuracy: ~49.34%
- months with both MCC > 0 and Brier Skill > 0: 0 / 13

### GTG Event Experience v3

The narrower event-conditioned task produced substantially healthier OOS
classification/calibration than the generic v2 benchmark:

- UP: AUC 0.6374, MCC 0.2093, Brier Skill +0.01162
- DOWN: AUC 0.6867, MCC 0.2517, Brier Skill +0.00376

These tasks have different labels and base rates, so this is evidence that
event-conditioning is a more promising formulation, not proof that v3 is a
universal replacement for every directional task.

## Promotion decision

Neither expert is promoted.

Reasons:

UP:

- positive OOS calibration skill: yes
- meaningful MCC: yes
- selective precision lift: yes
- monthly stability: materially better than v2
- OOS selective coverage >= 10%: **no, 9.87%**
- MFE/MAE path model beats baseline: **no**

DOWN:

- positive OOS calibration skill: yes
- meaningful MCC: yes
- selective precision lift: yes
- OOS selective coverage >= 10%: yes
- monthly stability: moderate
- MFE/MAE path model beats baseline: **no**

Therefore:

- decision authority: **false**
- deterministic GTG remains authoritative
- GTG v3 remains shadow evidence
- trading ablation: **not run**
- walk-forward retraining: **not run**
- MA200 -> MA1000 expansion: **not started as a promoted stage**

No PF, drawdown, expectancy, or trading-PnL claim is reported because the gate
that authorizes that ablation was not passed.

## What was learned

1. Event-conditioned directional success is meaningfully more promising than the
   previous generic direction formulation.
2. DOWN event classification is the strongest initial v3 result, especially
   after abstention.
3. UP also has real classification/calibration evidence, but selective coverage
   is just below the predefined promotion floor.
4. Event-conditioned analog memory is not universally useful:
   - UP validation selected a 50% analog blend.
   - DOWN validation correctly disabled analog.
5. The current path-regression formulation is the main blocker. Both MFE and
   MAE heads failed simple training-median baselines.
6. Adverse-first prediction is too weak to influence stops/risk.
7. The next research step should improve path/event semantics and test that
   separately, rather than granting v3 execution authority or tuning against
   this OOS year.
