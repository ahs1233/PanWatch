# GTG Event-Conditioned Directional Experience v3 — Measured Results

Date: 2026-09-24  
Mode: **shadow research only**  
Repository: `ahs1233/PanWatch`  
Research branch: `codex/gtg-event-v3`  
Successful benchmark commit/tree: `fd86ee7f41812811c6704b10d666e7b5188e110b`  
Current branch commit: `70846062eead98f2d76d567726b3250bde58b9be`  
Note: `70846062...` is a no-tree-change child of `fd86ee7...`; both point to the same tree for the benchmark code.  
GitHub Actions run: `35987140840`  
Job: `107592362920`  
Artifact: `10802936037`  
Artifact SHA-256: `efac55866e8d26fd0b9984ed25997edb4c05929bb705cc9d9d5e850f60d74887`

## Dataset and protocol

- Dataset ID: `gen1-xau-2y-dataset-v2-20240923-20260923`
- Source: Dukascopy XAUUSD native M1 bid/ask midpoint candles.
- Requested days: 730; observed data days: 729; one no-data day.
- M1 bars: 1,049,520.
- Train: 2024-09-23 → 2025-07-23.
- Validation/calibration: 2025-07-23 → 2025-09-23.
- OOS: 2025-09-23 → 2026-09-23.
- Architecture selection, normalization, calibration, and threshold selection did **not** use OOS.
- Event observations: train 761, validation 128, OOS 993.
- No order, sizing, stop, or execution authority is granted by this benchmark.

## UP expert — bullish 14/50 → MA200

Champion selected on validation: **TCN**

Samples:
- Train: 305
- Validation: 48
- OOS: 446
- Train success base rate: 58.69%

Validation success:
- Accuracy: 77.08%
- Balanced accuracy: 72.87%
- MCC: 0.5165
- ROC-AUC: 0.8203
- PR-AUC: 0.8807
- Brier skill: +0.05493

OOS success:
- Accuracy: 60.09%
- Balanced accuracy: 59.40%
- Precision: 64.68%
- Recall: 64.68%
- MCC: **0.1881**
- ROC-AUC: **0.6437**
- PR-AUC: **0.6876**
- Brier skill: **+0.01415**
- ECE: 0.0411

Validation-frozen abstention gate:
- Success threshold: 0.55
- Tradeability threshold: 0.40
- Validation coverage: 43.75%
- Validation precision: 85.71%

OOS selective result:
- Coverage: **9.87%**
- Selected events: 44
- Precision: **75.00%**
- Precision lift vs train base: **+16.31 pp**
- No-trade rate: 90.13%

Path:
- MFE MAE: 1.0808 ATR vs train-median baseline 0.9968 ATR
- MAE MAE: 2.6279 ATR vs train-median baseline 2.5135 ATR
- Time-to-target MAE: 60.70 minutes
- Positive MCC + positive Brier-skill months: 9 / 12

Analog memory:
- Validation selected weight: 0.50
- OOS blend MCC: 0.2093
- OOS blend Brier skill: +0.01162
- Neural-only Brier skill remained better: +0.01415

Promotion status: **NOT ELIGIBLE**.  
Reason: the path head did not beat the frozen train-median baseline on MFE/MAE, and selective OOS coverage is slightly below the 10% promotion floor.

## DOWN expert — bearish 14/50 → MA200

Champion selected on validation: **GRU**

Samples:
- Train: 456
- Validation: 80
- OOS: 547
- Train success base rate: 57.68%

Validation success:
- Accuracy: 77.50%
- Balanced accuracy: 73.33%
- MCC: 0.5060
- ROC-AUC: 0.8453
- PR-AUC: 0.8879
- Brier skill: +0.08129

OOS success:
- Accuracy: 60.15%
- Balanced accuracy: 62.63%
- Precision: **75.23%**
- Recall: 50.61%
- MCC: **0.2517**
- ROC-AUC: **0.6867**
- PR-AUC: **0.7806**
- Brier skill: **+0.00376**
- ECE: 0.1275

Validation-frozen abstention gate:
- Success threshold: 0.70
- Tradeability threshold: 0.40
- Validation coverage: 23.75%
- Validation precision: 94.74%

OOS selective result:
- Coverage: **10.42%**
- Selected events: 57
- Precision: **85.96%**
- Precision lift vs train base: **+28.29 pp**
- No-trade rate: 89.58%

Path:
- MFE MAE: 1.5154 ATR vs train-median baseline 1.3581 ATR
- MAE MAE: 2.3357 ATR vs train-median baseline 2.2435 ATR
- Time-to-target MAE: 57.61 minutes
- Positive MCC + positive Brier-skill months: 7 / 13

Analog memory:
- Validation selected weight: 0.00
- OOS neural-only MCC: 0.2517
- OOS neural-only Brier skill: +0.00376
- Analog-only Brier skill: -0.01421

Promotion status: **NOT ELIGIBLE**.  
Reason: despite strong selective precision and adequate coverage, the path head failed the frozen baseline test.

## Independent tradeability engine

Architecture: GRU

OOS:
- Accuracy: 68.88%
- Balanced accuracy: 51.91%
- MCC: 0.1021
- ROC-AUC: 0.5748
- PR-AUC: 0.3957
- Brier skill: +0.00390

This gate is useful only as a conservative abstention filter; it is not a directional model.

## Comparison with earlier GTG generations

v1 MA200-touch probability model:
- OOS AUC: 0.6292
- Brier skill: +0.0059
- p≥0.70 coverage: 5.53%
- p≥0.70 precision: 71.99%

v2 generic directional model:
- OOS balanced accuracy: 44.46%
- MCC: 0.0489
- Brier skill: -0.02403
- Stability: 0/13 months with both positive MCC and positive Brier skill

v3 event-conditioned experts:
- UP: MCC 0.1881, Brier skill +0.01415
- DOWN: MCC 0.2517, Brier skill +0.00376
- DOWN selective: 85.96% precision at 10.42% coverage
- Both experts fail the path-quality promotion gate

Interpretation: event conditioning materially improved directional discrimination versus v2, while the multi-task path head did not yet demonstrate OOS value beyond a simple train-median path baseline.

## Boundaries kept intact

Historical v3 intentionally excludes:
- synthetic footprint/CVD/order-flow labels from Dukascopy OHLC,
- CME L2/L3/aggressor claims without venue-qualified data,
- backfilled economic-release values without publication timestamps,
- self-supervised pretraining in the first event-conditioning benchmark.

These remain separate future ablations.

## Decision

**Keep GTG Event v3 in shadow/research mode. Do not promote to deterministic trading authority.**

The next engineering target is the path model, not the event grammar or directional heads. Any new path-method experiment must be labeled as a post-OOS diagnostic unless validated on a fresh forward holdout, because these v3 OOS results are now known.
