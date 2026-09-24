# GTG Event v3 — Path Diagnostic Results

Run date: 2026-09-24  
Status: **post-OOS diagnostic only — not promotion evidence**  
Branch: `codex/gtg-event-v3-path-diagnostic`  
Workflow run: `35992585531`  
Job: `107609934262`  
Benchmark commit: `aa5f8f7d1ae166159e911223e23c320492c9ee30`  
Artifact: `10804678407`  
Artifact SHA-256: `b1ae9860a19fc39726674e02aa995b15fafb441deb4a7e3b21048d8f403a08a6`

## Protocol integrity

This diagnostic does **not** alter the frozen GTG v3 event grammar or directional champions.

- Dataset: `gen1-xau-2y-dataset-v2-20240923-20260923`
- Original v3 champions are reproduced first.
- UP architecture remains TCN.
- DOWN architecture remains GRU.
- Candidate path method selection uses validation only.
- The already observed 2025-09-23 → 2026-09-23 OOS period is reported only as diagnostic reuse.
- No candidate may be promoted from this report.
- A fresh forward holdout beginning after the original OOS cutoff is required.

Candidates:
- native neural path head
- ridge on frozen champion embeddings, alpha 0.1 / 1 / 10 / 100
- ridge with log-positive MFE/MAE targets
- cosine embedding KNN median, k 8 / 16 / 32 / 64
- UTC session × train-volatility-tercile median

Selection objective:
minimize the worse of the MFE and MAE error ratios relative to the train-median baseline.

## DOWN path

Event: `bearish_14_50_to_200`  
Frozen architecture: GRU  
Samples: train 456 / validation 80 / diagnostic reuse 547

Validation-selected candidate:

**`knn_embedding_median_k64`**

Validation:

| Metric | Candidate | Baseline | Ratio |
|---|---:|---:|---:|
| MFE MAE (ATR) | 0.86796 | 0.94269 | 0.92073 |
| MAE MAE (ATR) | 2.38411 | 2.54232 | 0.93777 |

The candidate beats both path baselines on validation.

Diagnostic reuse of the already-known OOS year:

| Metric | Candidate | Baseline | Ratio |
|---|---:|---:|---:|
| MFE MAE (ATR) | 1.34747 | 1.35806 | 0.99220 |
| MAE MAE (ATR) | 2.16379 | 2.24355 | 0.96445 |

It also beats both baselines in the reused OOS data, but the MFE improvement is narrow and this period is no longer an untouched holdout.

For comparison, the original neural path head on the reused OOS had:
- MFE ratio to baseline: 1.11587
- MAE ratio to baseline: 1.04106

### DOWN conclusion

Embedding-neighbor path retrieval is the first path formulation in GTG v3 that passes the two-path-baseline condition on validation and remains directionally consistent on the reused OOS sample.

It is a **forward-test candidate**, not a promoted model.

## UP path

Event: `bullish_14_50_to_200`  
Frozen architecture: TCN  
Samples: train 305 / validation 48 / diagnostic reuse 446

Validation-selected candidate:

**`session_vol_bucket_median`**

Validation:

| Metric | Candidate | Baseline | Ratio |
|---|---:|---:|---:|
| MFE MAE (ATR) | 0.88674 | 0.89100 | 0.99522 |
| MAE MAE (ATR) | 2.37425 | 2.25807 | 1.05145 |

Diagnostic reuse of the known OOS year:

| Metric | Candidate | Baseline | Ratio |
|---|---:|---:|---:|
| MFE MAE (ATR) | 1.01110 | 0.99682 | 1.01432 |
| MAE MAE (ATR) | 2.49566 | 2.51350 | 0.99290 |

The UP candidate does not beat both baselines on validation or on the reused OOS sample.

For comparison, the original neural path head on reused OOS had:
- MFE ratio to baseline: 1.08424
- MAE ratio to baseline: 1.04552

### UP conclusion

UP path estimation remains unsolved. It stays shadow-only and should not be forwarded into any deterministic trading ablation.

## Decision

- GTG v3 directional classification remains shadow evidence.
- DOWN `knn_embedding_median_k64` becomes the only path candidate eligible for **fresh forward evaluation**.
- UP path has no qualifying candidate.
- No trading PnL / PF / drawdown ablation is authorized by this diagnostic.
- No production deployment is authorized.
- Fresh forward events must begin strictly after the original untouched-OOS cutoff, with predictions persisted before labels become observable.
