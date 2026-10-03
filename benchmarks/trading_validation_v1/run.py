"""Executable Validation Framework v1 benchmark.

It proves the framework can distinguish a stable synthetic fixture from a
textbook IS winner / OOS loser while enforcing leakage guards.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from src.modules.strategy.validation import (
    DatasetManifest,
    ExperimentSpec,
    ParameterPoint,
    TradeRecord,
    ValidationPolicy,
    WalkForwardFoldResult,
    validate_experiment,
)
from src.modules.strategy.validation.splits import walk_forward_splits


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def trade(i: int, pnl: float) -> TradeRecord:
    opened = START + timedelta(hours=i * 6)
    return TradeRecord(
        trade_id=f"b{i:04d}",
        opened_at=opened,
        closed_at=opened + timedelta(minutes=30),
        pnl=pnl,
        mae=-abs(pnl) * 0.4,
        mfe=max(pnl, 0.0) * 1.4 + 0.2,
        session=("asia", "london", "new_york")[i % 3],
        regime=("trend", "range")[i % 2],
        news_window=(i % 7 == 0),
    )


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="trading-validation-v1-benchmark",
        strategy_fingerprint="benchmark-strategy-v1",
        dataset=DatasetManifest(
            dataset_id="synthetic-validation-v1",
            fingerprint="sha256:synthetic-validation-v1",
            symbol="XAUUSD",
            start=START,
            end=START + timedelta(days=365),
            source="deterministic synthetic benchmark",
            bar_count=100_000,
        ),
        walk_forward_train=60,
        walk_forward_test=20,
        walk_forward_step=20,
        monte_carlo_iterations=500,
        monte_carlo_seed=20260923,
        strategy_frozen_before_oos=True,
        policy=ValidationPolicy(
            min_oos_trades=30,
            min_oos_expectancy=0.0,
            min_oos_profit_factor=1.0,
            max_oos_drawdown=3.0,
            min_walk_forward_positive_fraction=0.75,
            min_parameter_stability_score=0.5,
            max_monte_carlo_loss_probability=0.10,
        ),
    )


def surface() -> list[ParameterPoint]:
    return [
        ParameterPoint({"ema_fast": 8, "ema_slow": 20}, 1.04, 180),
        ParameterPoint({"ema_fast": 9, "ema_slow": 20}, 1.08, 190),
        ParameterPoint({"ema_fast": 9, "ema_slow": 21}, 1.10, 200),
        ParameterPoint({"ema_fast": 10, "ema_slow": 21}, 1.07, 194),
        ParameterPoint({"ema_fast": 10, "ema_slow": 22}, 1.02, 184),
        ParameterPoint({"ema_fast": 11, "ema_slow": 22}, 0.98, 178),
    ]


def walk_forward_evidence(trades: list[TradeRecord]) -> list[WalkForwardFoldResult]:
    results: list[WalkForwardFoldResult] = []
    for fold in walk_forward_splits(trades, train_size=60, test_size=20, step=20):
        train = list(fold.train)
        test = list(fold.test)
        results.append(
            WalkForwardFoldResult(
                fold_index=fold.index,
                train_start=train[0].opened_at,
                train_end=train[-1].closed_at,
                selected_at=train[-1].closed_at,
                test_start=test[0].opened_at,
                test_end=test[-1].closed_at,
                selected_strategy_fingerprint=f"wf:{fold.index}",
                selected_parameters={"ema_fast": 9, "ema_slow": 21},
                test_trades=tuple(test),
            )
        )
    return results


def validate(trades: list[TradeRecord]):
    return validate_experiment(
        spec=spec(),
        trades=trades,
        parameter_surface=surface(),
        parameter_surface_is_in_sample_only=True,
        walk_forward_results=walk_forward_evidence(trades),
    )


def main() -> None:
    pattern = (1.2, 0.9, 0.7, -0.55, 0.8, 1.0, -0.45, 0.65)
    stable = [trade(i, pattern[i % len(pattern)]) for i in range(200)]
    overfit = [trade(i, 1.0 if i < 120 else -0.8) for i in range(200)]

    stable_report = validate(stable)
    overfit_report = validate(overfit)

    result = {
        "benchmark": "trading-validation-v1",
        "stable_fixture": {
            "verdict": stable_report.verdict.value,
            "oos": asdict(stable_report.out_of_sample),
            "walk_forward_positive_fraction": stable_report.walk_forward["positive_test_fraction"],
            "walk_forward_provenance": stable_report.walk_forward["provenance"],
            "monte_carlo_loss_probability": stable_report.monte_carlo["loss_probability"],
            "parameter_stability_score": stable_report.parameter_stability["stability_score"],
        },
        "overfit_fixture": {
            "verdict": overfit_report.verdict.value,
            "is_expectancy": overfit_report.in_sample.expectancy,
            "oos_expectancy": overfit_report.out_of_sample.expectancy,
            "failed_gates": [g.name for g in overfit_report.gates if g.passed is False],
        },
        "passed": (
            stable_report.verdict.value == "pass"
            and stable_report.walk_forward["provenance"] == "train_select_future_test"
            and overfit_report.verdict.value == "fail"
            and overfit_report.in_sample.expectancy > 0
            and overfit_report.out_of_sample.expectancy < 0
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
