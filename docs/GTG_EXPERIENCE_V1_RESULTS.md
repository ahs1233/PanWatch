# GTG Deep Experience v1 — OOS Results

Status: **shadow research; not decision authority**

Run: GitHub Actions `35976023371`  
Artifact: `gtg-experience-v1-ci-5ec9e877e1224d0caeebee50d474c5ef2e6d681b`  
Artifact ID: `10798735022`

## Temporal split

- Train: 2024-09-23 -> 2025-07-23
- Calibration: 2025-07-23 -> 2025-09-23
- Untouched OOS: 2025-09-23 -> 2026-09-23
- Random split: forbidden
- Future leakage: forbidden

Samples:
- Train sequences: 13,887
- Calibration sequences: 5,803
- OOS sequences: 34,156

## Architecture

- 96 x 5-minute causal sequence (8 hours)
- 25 causal market features
- 2-layer GRU
- hidden size 64
- learned embedding size 32
- 55,448 parameters
- multi-task heads
- temperature calibration fitted only on calibration period

## OOS findings

### MA200 touch — PROMISING SHADOW HEAD

- count: 23,779
- base rate: 52.33%
- AUC: 0.6292
- Brier: 0.24355
- baseline Brier: 0.24946
- Brier skill: **+0.00590**
- p >= 0.60: 18.74% coverage, 68.10% precision
- p >= 0.65: 11.51% coverage, 69.33% precision
- p >= 0.70: 5.53% coverage, 71.99% precision
- p >= 0.75: 1.55% coverage, 72.63% precision

Interpretation: the learned state contains useful OOS information about whether MA200 becomes a destination.
This is the only v1 head currently eligible for downstream shadow ablation.

### MA1000 touch — RANKING SIGNAL ONLY

- count: 19,501
- AUC: 0.6153
- Brier skill: -0.00130
- p >= 0.65 precision: 66.78% at 3.10% coverage

Interpretation: ranking information exists, but probability calibration/generalization is not yet good enough for GTG weighting.

### Next move direction — REJECTED

- count: 32,724
- AUC: 0.5100
- Brier skill: +0.000048
- accuracy: 50.09%

Interpretation: no meaningful OOS directional edge. Do not use this head for trade direction.

### MA200 rejection — REJECTED

- count: 1,384
- AUC: 0.5448
- Brier skill: -0.00098

### MA1000 rejection — REJECTED

- count: 658
- AUC: 0.5209
- Brier skill: -0.01167

### Expansion — NOT ACTIONABLE

- base rate: 93.90%
- AUC: 0.7292
- Brier skill: -0.00031

The label is too common to be useful as a decision gate in its current form despite ranking ability.

## Analog memory v1

Nearest-neighbor memory over the generic latent embedding did **not** improve the directional task:

- neural-only Brier: 0.24981
- analog Brier: 0.28410
- blended Brier: 0.25282

Decision: analog memory stays disabled for decision weighting. Future analog memory must be event-conditioned
(e.g. MA200 destination states compared only with similar MA200 destination events).

## Promotion policy

GTG v1 deterministic logic remains authoritative.

Deep Experience v1:
- may emit shadow probability for MA200 destination confidence;
- may record embeddings for research;
- must not alter position size, stop, direction, or entry;
- must not use rejected heads;
- must not use generic analog memory.

Next research version should be **event-conditioned**, trained specifically on:
1. price break/reclaim 14/50 -> MA200 outcome;
2. accepted MA200 break -> MA1000 outcome;
3. MA interaction -> rejection vs acceptance;
4. execution timing conditional on a destination hypothesis.

This converts deep learning from generic prediction into learned experience around GTG's actual setup grammar.
