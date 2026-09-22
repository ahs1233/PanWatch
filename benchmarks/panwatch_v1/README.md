# PanWatch Benchmark v1.3

PanWatch Benchmark is the reproducible measurement layer for the project. It separates implemented research integrity from future capabilities and prevents architectural progress from being judged by intuition alone.

## Automated track 1 — XAU decision core

15 deterministic failure-sensitive cases covering event-risk vetoes, stale-data rejection, calibration, adversarial counter-evidence, probabilistic regimes, sensor disagreement, research-memory integrity, evidence hierarchy, adaptive thresholds, source-family discrimination, state similarity and permanent live-execution disablement.

Regression gate: **95%**.

## Automated track 2 — Economic evidence integrity

10 frozen-fixture cases covering Actual vs Forecast, revisions, source independence, freshness, numeric conflicts, chronology, semantic separation, delayed transmission, forecast-to-realization updates and confidence changes after contradictory evidence.

Regression gate: **90%**.

## Automated track 3 — Claim Graph + Falsification

Domain-neutral reasoning integrity across multi-hop support, required dependencies, cycle rejection, hard/soft falsifiers, counterclaims, numeric conflicts, and research probes for untestable falsification rules.

Regression gate: **90%**.

## Automated track 4 — Persistent Belief State

v1.3 makes PanWatch remember what it believed across runs instead of reconstructing every conclusion from scratch.

The layer adds:

- immutable belief snapshots for every evaluated claim;
- append-only belief change events;
- explicit status transitions;
- confidence increase/decrease events;
- evidence-change events;
- falsification-state change events;
- dependency-failure events;
- falsified → recovered history;
- no-noise behavior when nothing material changed;
- historical as-of evaluation that blocks future revisions from rewriting past beliefs;
- propagation of required-dependency falsification into downstream beliefs;
- durable PanWatch evaluation cycles with summary counts and research probes.

Persistent tables:

- research_belief_cycles
- research_belief_snapshots
- research_belief_events

Regression gate: **90%**.

## Specified track — Company / sector research

The company/sector research track remains specified but not automated.

## What v1.3 does not claim

A deterministic 100% score does not prove that PanWatch is generally more intelligent than ChatGPT, Claude, or a professional researcher.

Current remaining boundaries include:

- automatic claim extraction from arbitrary documents;
- automatic dispatch of falsification probes to live research tools;
- external comparative benchmarks with blind human labels.

## Run locally

```bash
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.economic_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.reasoning_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.belief_runner

python -m pytest -q \
  ci_tests/test_panwatch_benchmark_v1.py \
  ci_tests/test_research_evidence_foundation.py \
  ci_tests/test_research_claim_graph_falsification.py \
  ci_tests/test_research_persistent_belief_state.py \
  ci_tests/test_panwatch_economic_benchmark_v1.py \
  ci_tests/test_panwatch_reasoning_benchmark_v1.py \
  ci_tests/test_panwatch_belief_benchmark_v1.py
```
