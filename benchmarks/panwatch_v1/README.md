# PanWatch Benchmark v1.2

PanWatch Benchmark is the reproducible measurement layer for the project. It separates implemented research integrity from future capabilities and prevents architectural progress from being judged by intuition alone.

## Automated track 1 — XAU decision core

15 deterministic failure-sensitive cases covering event-risk vetoes, stale-data rejection, calibration, adversarial counter-evidence, probabilistic regimes, sensor disagreement, research-memory integrity, evidence hierarchy, adaptive thresholds, source-family discrimination, state similarity and permanent live-execution disablement.

Regression gate: **95%**.

## Automated track 2 — Economic evidence integrity

10 frozen-fixture cases covering Actual vs Forecast, revisions, source independence, freshness, numeric conflicts, chronology, semantic separation, delayed transmission, forecast-to-realization updates and confidence changes after contradictory evidence.

Regression gate: **90%**.

## Automated track 3 — Claim Graph + Falsification

v1.2 adds a domain-neutral reasoning layer:

- explicit claim nodes;
- logical and causal claim edges;
- explicit evidence-to-claim links;
- multi-hop confidence propagation;
- required dependency failures;
- downstream impact sets;
- cycle rejection for reasoning dependencies;
- explicit falsification rules;
- hard and soft failure conditions;
- numeric-threshold falsifiers;
- counterclaim-confidence falsifiers;
- freshness and source-independence falsifiers;
- generated research probes when a falsification test cannot yet be executed;
- guardrail preventing hypotheses/conclusions from finishing as supported when no falsification condition exists.

The graph and rules persist through:

- research_claims
- research_claim_edges
- research_claim_evidence_links
- research_falsification_rules

Regression gate: **90%**.

## Specified track — Company / sector research

The company/sector track remains specified but not automated.

## What v1.2 does not claim

A 100% deterministic score does not prove that PanWatch is generally more intelligent than ChatGPT, Claude, or a professional researcher.

The current falsification layer can **generate explicit counter-research probes**, but it does not yet automatically dispatch those probes to live web/research tools. Persistent cross-session belief-state evolution is also not part of v1.2.

Those are the next integration steps.

## Run locally

```bash
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.economic_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.reasoning_runner

python -m pytest -q \
  ci_tests/test_panwatch_benchmark_v1.py \
  ci_tests/test_research_evidence_foundation.py \
  ci_tests/test_research_claim_graph_falsification.py \
  ci_tests/test_panwatch_economic_benchmark_v1.py \
  ci_tests/test_panwatch_reasoning_benchmark_v1.py
```
