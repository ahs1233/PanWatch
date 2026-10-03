# GTG Directional Experience v2 — OOS Results

Status: **RESEARCH COMPLETE — NOT PROMOTED**

GitHub Actions validation run:
- Run ID: `35982961791`
- Job ID: `107578959269`
- Artifact ID: `10800958460`
- Artifact SHA256: `cb80b1df5a21cb58dff389ac787e553e210a13a8b6858c3ece42269799da5e15`

## Integrity

- Frozen XAUUSD dataset: 24 Dukascopy shards / 730 days.
- Train: 2024-09-23 -> 2025-07-23.
- Validation/calibration: 2025-07-23 -> 2025-09-23.
- Untouched OOS: 2025-09-23 -> 2026-09-23.
- Model selection used validation only.
- Confidence-threshold selection used validation only.
- Random split: disabled.
- Future leakage: disabled.
- Regression suite: 3/3 passed before training.

Samples:
- train: 13,331
- validation: 5,602
- OOS: 32,834

## Direction task

Classes are UP / DOWN / NEUTRAL at 30m / 60m / 120m / 240m using ATR-scaled symmetric barriers.
A same-M5-bar double touch is masked as ambiguous rather than guessed.

60-minute class distribution:
- train: DOWN 6,152 / NEUTRAL 653 / UP 6,526
- OOS: DOWN 15,563 / NEUTRAL 1,738 / UP 15,533

The neutral class is only about 5% and is therefore too small for a robust abstention regime.
This is a label-design issue to fix before increasing model complexity.

## Architecture comparison

Champion selection was frozen on validation before OOS metrics were read.

| Model | Validation selection score | OOS accuracy | OOS balanced acc. | OOS macro F1 | OOS MCC | OOS Brier skill |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 0.6064 | 46.43% | 42.76% | 41.33% | 0.0361 | -0.00471 |
| Patch Transformer | **0.6175** | 45.75% | **44.46%** | 40.44% | **0.0489** | -0.02403 |
| TimesNetLite | 0.6019 | **46.68%** | 43.62% | **41.37%** | 0.0437 | -0.00968 |

Validation-only champion: **Patch Transformer**.

None of the models produced a robust calibrated OOS directional edge.

## Champion — 60 minute OOS

- Accuracy: 45.75%
- Balanced accuracy: 44.46%
- Macro F1: 40.44%
- MCC: 0.0489
- ECE: 0.0386
- Brier: 0.57321
- baseline Brier: 0.54917
- Brier skill: **-0.02403**

Per class:

DOWN:
- precision 49.97%
- recall 56.58%
- F1 53.07%

NEUTRAL:
- precision 21.20%
- recall 41.43%
- F1 28.05%

UP:
- precision 46.52%
- recall 35.38%
- F1 40.19%

The model is asymmetric and materially better at recognizing DOWN than UP, but neither side reaches promotion quality.

## Validation-selected confidence gate

Threshold chosen without OOS knowledge: 0.45.

Validation:
- coverage: 63.80%
- directional accuracy: 51.29%
- opposite-direction rate: 46.70%

Untouched OOS:
- selected events: 20,027
- directional coverage: 60.99%
- directional accuracy: **49.34%**
- opposite-direction rate: **48.62%**
- DOWN calls: 15,385
- UP calls: 4,642

The confidence gate did not generalize and showed a strong DOWN-call bias.

The reported event-R sum (+143 before costs) is a barrier-event diagnostic over overlapping observations,
not executable trading PnL, and must not be presented as strategy profit.

## OOS by primary horizon

Patch Transformer:

- 30m: balanced accuracy 47.59%, MCC 0.1002, negative Brier skill; extreme DOWN-recall / weak UP-recall asymmetry.
- 60m: balanced accuracy 44.46%, MCC 0.0489, negative Brier skill.
- 120m: balanced accuracy 41.02%, MCC 0.0487, negative Brier skill.
- 240m: balanced accuracy 41.17%, MCC 0.0367, negative Brier skill.

No horizon passed the promotion standard.

## Path-learning result

Champion absolute errors:

- UP MFE MAE: 2.30 ATR
- DOWN MFE MAE: 2.39 ATR
- trend t-stat scaled MAE: 1.15

These errors are too large to use path prediction for stop/target decisions.

## Stability

OOS month slices: 13.

Months with both positive MCC and positive Brier skill: **0 / 13**.

March 2026 showed locally better selective direction behavior, but the year-wide and month-by-month
calibration test did not confirm a stable directional edge.

## Promotion decision

- decision authority: **FALSE**
- eligible for deterministic GTG ablation: **FALSE**
- live sizing/entry/risk authority: **FORBIDDEN**

The v2 neural layer remains research-only.

## What the negative result teaches us

The earlier GTG Deep Experience v1 MA200-destination task was materially more learnable than generic direction:
MA200 touch OOS AUC was about 0.629 with positive Brier skill, whereas direct UP/DOWN/NEUTRAL prediction here
failed calibration and stability gates.

Therefore the next version must not simply make the network larger.

### Recommended v3 architecture

1. **Event-conditioned learning**
   Train only around GTG grammar:
   - 14/50 transition -> MA200
   - accepted MA200 -> MA1000
   - MA barrier interaction
   - supply/demand/order-block interaction
   - liquidity sweep / reclaim

2. **Separate experts**
   - UP expert learns only bullish opportunity quality.
   - DOWN expert learns only bearish opportunity quality.
   - Neither probability is defined as 1 minus the other.

3. **Independent abstention/regime gate**
   Learn when neither expert should be trusted.
   Rebalance the neutral/no-trade state instead of forcing almost every observation into UP/DOWN.

4. **Destination + path conditional model**
   Given a valid GTG event, learn:
   - target probability
   - probability of adverse excursion before target
   - MFE / MAE
   - time to target
   - acceptance vs rejection

5. **Microstructure upgrade when data exists**
   Add venue-qualified GC/MGC futures footprint, delta/CVD, depth/heatmap and aggressor flow.
   Never fabricate these from Dukascopy OHLC/tick-activity data.

6. **Keep v1 MA200 destination head**
   It remains a useful shadow feature and should be compared against v3 rather than discarded.

This preserves GTG's deterministic market grammar and lets deep learning acquire experience only where the
historical data demonstrates that the task is learnable.
