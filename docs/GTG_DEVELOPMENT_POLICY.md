# GTG Development Policy

Status: ACTIVE
Applies to: all GTG development for PanWatch

## 1. Branch ownership

- `production/panwatch-stable` is the production release branch.
- Treat `production/panwatch-stable` as read-only during normal GTG work.
- GTG development continues only on `feat/ahmed-toolbox-xau` (or a child feature branch created from it).
- Never push experimental GTG work directly to the production branch.

## 2. Production deployment rule

Railway Production is sourced only from:
`production/panwatch-stable`

A normal GTG commit must never trigger a production deployment.

Promotion to Production is a separate release operation after all gates below pass.

## 3. Required gates before promotion

A GTG change is promotable only when all relevant checks for the exact candidate SHA are green:

1. Ahmed ToolBox + XAU CI.
2. XAU Real Dataset Smoke.
3. XAU Native vs LEAN Parity when touched by the change.
4. XAU Dukascopy Deep History Smoke when historical data paths are touched.
5. GEN1 Gold 2Y Real Backtest when decision logic/model behavior changes.
6. Focused regression tests for the changed module.
7. No startup regression in PanWatch.
8. No regression in Ahmed Toolbox connectivity/auth lifecycle.
9. No regression in storage health or persistence.
10. A benchmark/result artifact exists for material GTG model/strategy changes.

Do not promote a SHA with queued, failed, cancelled, or skipped required gates.

## 4. Stable infrastructure boundaries

GTG work must not modify these stable boundaries unless the task explicitly requires it:

- Ahmed Toolbox durable refresh-token lifecycle.
- PanWatch volume mount at `/app/data`.
- XAU paper/replay single-primary storage policy.
- `local_primary` as the authoritative XAU paper/replay store in Production.
- One-way external replica/archive sync.
- Railway healthcheck and production service wiring.

If one of these boundaries must change, isolate that work, add dedicated regression tests, and require a separate production acceptance.

## 5. Storage safety rules

Production XAU storage architecture is:

`local SQLite on /app/data (PRIMARY) -> optional external replica/archive`

Rules:

- Never auto-promote Neon/external PostgreSQL to primary.
- Never use bidirectional automatic reconciliation.
- Never delete external rows during replica sync.
- Replica outage/quota must not block PanWatch startup or paper-trading state.
- Sync must remain idempotent.
- Any detected target conflict must surface as degraded/diagnostic state, not be silently resolved.

## 6. GTG experimentation rules

- New GTG models, labels, features, thresholds, experience layers, or decision logic start as research/shadow behavior.
- Do not grant a new model decision authority because an in-sample metric improved.
- Use temporal OOS evaluation; random train/test splitting is not acceptable for time-series claims.
- Preserve anti-lookahead rules.
- Record exact dataset window, source, commit SHA, configuration, and artifact/run ID for material benchmarks.
- A benchmark failure must not be hidden by weakening the gate.

## 7. Release process

When a GTG candidate is ready:

1. Freeze the candidate SHA on the GTG branch.
2. Run/verify all required gates for that exact SHA.
3. Review benchmark artifacts and regressions.
4. Promote that exact SHA to `production/panwatch-stable`.
5. Deploy Railway Production.
6. Verify:
   - deployment SUCCESS,
   - `Application startup complete`,
   - volume mounted,
   - `primary=local_sqlite persistent=true`,
   - Ahmed Toolbox connectivity healthy,
   - no new storage/auth/startup errors.
7. Only then mark the release complete.

## 8. No moving-target acceptance

Do not chase the newest development HEAD when validating a completed production concern.

For Production acceptance, always report:

- production branch,
- exact production SHA,
- CI status for that candidate,
- Railway deployment ID/status,
- production health evidence.

Development may continue independently after that.

## 9. Failure policy

If a GTG change fails CI or a benchmark:

- keep Production unchanged,
- diagnose on the GTG branch,
- fix and retest there,
- do not bypass the gate,
- do not change stable infrastructure merely to make a GTG test pass.

This policy exists to let GTG evolve quickly without turning Production into an experiment.
