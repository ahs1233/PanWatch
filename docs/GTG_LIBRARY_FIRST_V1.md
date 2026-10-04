# GTG Library-First v1

This is a clean-room architectural restart for GTG inside PanWatch.

It does not delete or modify the current production strategy. The old system remains a reference until this implementation proves itself.

## Core rule

Reuse first -> Integrate second -> Extend third -> Build from scratch last.

Custom code is allowed only where it represents GTG-specific domain knowledge or where no mature maintained component exists.

## Responsibility map

| Capability | Reused component | What remains custom |
|---|---|---|
| EMA / RSI | pandas-ta-classic | GTG slope, curvature, spacing semantics |
| tabular / resampling | pandas | causal timeframe policy |
| analogue path model | scikit-learn NearestNeighbors + StandardScaler | event feature selection and path target semantics |
| forward metrics | River | GTG acceptance gates |
| research backtesting | vectorbt | GTG entries/exits/event definitions |
| independent verification | LEAN | adapter/parity contract only |
| storage | SQLAlchemy + PostgreSQL | prediction/outcome schema |
| scheduled shadow jobs | APScheduler | GTG job definitions |
| volume profile | MarketProfile | GTG interpretation |
| SMC primitives | smartmoneyconcepts | GTG confirmation rules |
| contracts | Pydantic | GTG event models |

## Architectural boundaries

1. gtg_next has no broker/order execution API.
2. Research/backtest and live-forward evidence remain separate.
3. Production is untouched until promotion criteria are met.
4. A third-party library is wrapped behind a small adapter only when GTG semantics need it.
5. No custom implementation of generic KNN, indicators, schedulers, ORMs, online metrics, or generic backtest engines.

## First implemented slice

- Library capability registry.
- Immutable GTG domain contracts.
- Feature composition using pandas-ta-classic.
- KNN path analogue using scikit-learn.
- Forward streaming metrics using River.
- Dedicated isolated CI.

## Next slices

- SQLAlchemy/PostgreSQL forward evidence repository.
- APScheduler shadow worker.
- vectorbt GTG event backtest adapter.
- LEAN parity adapter.
- MarketProfile / SMC adapters only after their contribution is measured by ablation.

Every new component must answer this question first: Is there a maintained library or existing PanWatch component that already solves the generic part?

If yes, integrate it. If no, then custom code may be justified.
