# PanWatch Benchmark v1.5

PanWatch Benchmark is the reproducible measurement layer for the project. It separates implemented research integrity from future capabilities and prevents architectural progress from being judged by intuition alone.

## Automated track 1 — XAU decision core

15 deterministic failure-sensitive cases covering event-risk vetoes, stale-data rejection, calibration, adversarial counter-evidence, probabilistic regimes, sensor disagreement, research-memory integrity, evidence hierarchy, adaptive thresholds, source-family discrimination, state similarity and permanent live-execution disablement.

Regression gate: **95%**.

## Automated track 2 — Economic evidence integrity

10 frozen-fixture cases covering Actual vs Forecast, revisions, source independence, freshness, numeric conflicts, chronology, semantic separation, delayed transmission, forecast-to-realization updates and confidence changes after contradictory evidence.

Regression gate: **90%**.

## Automated track 3 — Claim Graph + Falsification

Domain-neutral reasoning integrity across multi-hop support, required dependencies, cycle rejection, hard/soft falsifiers, counterclaims, numeric conflicts, and research probes.

Regression gate: **90%**.

## Automated track 4 — Persistent Belief State

PanWatch stores immutable historical belief snapshots and append-only material change events so a claim can be reconstructed across restarts and redeploys.

The track covers restart continuity, unchanged-cycle noise suppression, confidence changes, falsification/recovery, historical as-of integrity, dependency propagation and idempotent persistence.

Regression gate: **90%**.

## Automated track 5 — Automatic Research Loop

v1.4 closes the falsification loop:

```
Claim
  → Falsification Probe
  → Ahmed ToolBox discovery/search/read
  → Quote-grounded extraction
  → Evidence Ledger
  → Claim Graph link
  → Belief re-evaluation
  → Persistent change event
```

Safety/integrity constraints:

- exact source quote required before LLM-extracted evidence is accepted;
- hallucinated/unverifiable quotes are discarded;
- per-cycle probe/source/tool-call budgets;
- persisted cooldown for repeated probes;
- no-claim cycles make zero research calls;
- one source failure cannot kill the whole cycle;
- live execution remains unrelated and disabled;
- external provider failure does not block PanWatch startup.

Production loop persistence adds:

- research_loop_runs
- research_probe_attempts

The production XAU adapter seeds a real time-sensitive macro claim into the otherwise domain-neutral research engine. Other domains can register their own claims without changing the loop core.

Regression gate: **90%**.

## Automated track 6 — General Claim Acquisition

v1.5 adds domain-neutral knowledge intake before the reasoning loop:

```
Document / transcript / web result
  → exact-quote Claim Candidate
  → admission audit
  → exact deduplication / conservative versioning
  → Evidence Ledger
  → Claim Graph
  → generated falsification rules
  → initial Belief State
```

Admission rules:

- every admitted claim must be grounded in an exact source quote;
- non-testable or question-like candidates are rejected;
- the same statement from another independent source strengthens one claim instead of creating a duplicate;
- a new version supersedes history only when the source contains an explicit revision/update cue;
- company, science, policy, macro, XAU, and future domains use the same core;
- every candidate receives an auditable accepted / duplicate / rejected / superseded decision.

Regression gate: **90%**.

## Specified track — Company / sector research

The company/sector research benchmark remains specified but not automated. The generic engine can accept such claims, but the frozen company/sector corpus and comparative scoring harness still need implementation.

## What v1.4 does not claim

A deterministic 100% score does not prove general research superiority over ChatGPT, Claude, or a professional researcher.

Current boundaries:

- claim extraction is now generalized, but recall/semantic clustering over arbitrary long-form corpora is not yet proven;
- external live search quality is provider-dependent and verified separately from deterministic CI;
- blind human-scored external comparative benchmarks are not yet implemented.

## Run locally

```bash
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.economic_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.reasoning_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.belief_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.automatic_research_runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.claim_acquisition_runner

python -m pytest -q \
  ci_tests/test_panwatch_benchmark_v1.py \
  ci_tests/test_research_evidence_foundation.py \
  ci_tests/test_research_claim_graph_falsification.py \
  ci_tests/test_research_persistent_belief_state.py \
  ci_tests/test_research_external_store.py \
  ci_tests/test_research_automatic_loop.py \
  ci_tests/test_research_claim_acquisition.py \
  ci_tests/test_panwatch_economic_benchmark_v1.py \
  ci_tests/test_panwatch_reasoning_benchmark_v1.py \
  ci_tests/test_panwatch_belief_benchmark_v1.py \
  ci_tests/test_panwatch_automatic_research_benchmark_v1.py \
  ci_tests/test_panwatch_claim_acquisition_benchmark_v1.py
```
