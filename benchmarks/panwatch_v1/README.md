# PanWatch Benchmark v1.1

PanWatch Benchmark is the reproducible measurement layer for the project. Its purpose is to stop architectural progress from being judged by intuition alone.

## Track 1 — XAU decision core

The original deterministic track exercises the current XAU decision core across 15 failure-sensitive cases: event-risk vetoes, stale-data rejection, confidence calibration, adversarial counter-evidence, probabilistic regimes, sensor disagreement, research-memory guards, conflicting priors, evidence hierarchy, fill isolation, adaptive thresholds, source-family discrimination, state similarity, and permanent live-execution disablement.

The score is weighted to 100 points and uses a **95% regression gate**. This is a regression score for already implemented behavior, not a product-completeness score.

## Track 2 — Economic evidence integrity

v1.1 adds an automated 10-case track that exercises the generic Evidence Foundation rather than XAU-specific logic.

It checks:

- actual vs forecast separation;
- revisions vs original releases;
- syndicated-copy source independence;
- stale macro data;
- conflicting numeric claims;
- event chronology reconstruction;
- policy statements vs market pricing;
- delayed transmission / temporal lag preservation;
- forecast-to-realization updates;
- confidence changes after new contradictory evidence.

This track uses a **90% regression gate** and frozen deterministic fixtures. It tests evidence semantics, not economic forecasting skill or live-source retrieval quality.

## Evidence Foundation introduced in v1.1

PanWatch now has domain-neutral primitives for:

- canonical source provenance;
- stable source and evidence IDs;
- append-only evidence records;
- Actual / Forecast / Revision / Estimate / Guidance / Policy Statement / Market Pricing semantics;
- source-family and independence keys;
- freshness guards;
- numeric conflict detection;
- linked revision precedence;
- event chronology and lag;
- confidence updates without double-counting syndicated copies;
- SQLite persistence through `research_sources` and `research_evidence`.

## Track 3 — Company / sector research

The company/sector track is still specified but not automated.

## What v1.1 does not claim

A 100% score in either automated track does **not** mean PanWatch is 100% complete or superior to ChatGPT, Claude, or a professional analyst. External comparative research quality still requires frozen/live source packets, blind labels, identical tasks, and human or model baselines.

Claim Graph, generic falsification, full inline citation generation, and persistent cross-session belief-state integration remain future work.

## Run locally

```bash
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.economic_runner
python -m pytest -q \
  ci_tests/test_panwatch_benchmark_v1.py \
  ci_tests/test_research_evidence_foundation.py \
  ci_tests/test_panwatch_economic_benchmark_v1.py
```
