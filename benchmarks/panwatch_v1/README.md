# PanWatch Benchmark v1

PanWatch Benchmark v1 is the first reproducible measurement layer for the project. Its job is to stop architectural progress from being judged by intuition alone.

## What v1 measures

The automated track exercises the current XAU decision core across 15 failure-sensitive cases:

- event-risk vetoes;
- stale/blocked-data vetoes;
- confidence response to realized history;
- adversarial counter-evidence;
- probabilistic regime consistency;
- sensor-disagreement penalties;
- sample-size and integrity guards on research memory;
- damping of conflicting research priors;
- precedence of realized trade evidence over weaker research priors;
- isolation of fill availability from analytical quality;
- adaptive confidence thresholds under poor calibration;
- source-family discrimination;
- market-state similarity discrimination;
- permanent live-execution disablement.

The score is weighted to 100 points. CI uses a **95% regression gate**. The gate is deliberately a regression guard for the already-implemented core, not a claim that PanWatch is 95% complete.

## What v1 does not claim

Two additional benchmark tracks are defined but not yet automated:

1. economic research;
2. company/sector research.

The manifest contains ten tasks for each. They will become the basis for direct comparison between PanWatch, a general web-enabled LLM, and a human researcher.

The following future architecture is also explicitly outside the v1 core score: Source Provenance, Evidence Ledger, inline claim citations, actual/forecast guardrails, source independence, numerical conflict resolution, Claim Graph, generic falsification, and full persistent-research state integration.

## Run locally

```bash
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner
PYTHONPATH=. python -m benchmarks.panwatch_v1.runner --json benchmark-result.json
python -m pytest -q ci_tests/test_panwatch_benchmark_v1.py
```

## Why there are two kinds of truth here

A passing core score means the current deterministic decision safeguards still behave as specified. It does **not** prove that PanWatch produces better research than ChatGPT, Claude, or a professional analyst. That second claim requires the two cross-domain tracks, frozen source packets, blind labels, and external baselines. The benchmark is designed so those can be added without rewriting the current core suite.
