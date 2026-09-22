from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.modules.strategy.validation import (
    DatasetManifest,
    ExperimentSpec,
    ParameterPoint,
    TradeRecord,
    ValidationPolicy,
    ValidationVerdict,
    WalkForwardFoldResult,
    validate_experiment,
)
from src.modules.strategy.validation.metrics import compute_performance
from src.modules.strategy.validation.monte_carlo import bootstrap_trade_sequences
from src.modules.strategy.validation.splits import chronological_split, walk_forward_splits
from src.modules.strategy.validation.stability import analyze_parameter_stability
from src.modules.strategy.validation.stress import cost_stress


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _trade(i: int, pnl: float) -> TradeRecord:
    opened = START + timedelta(hours=i * 6)
    return TradeRecord(
        trade_id=f"t{i:04d}",
        opened_at=opened,
        closed_at=opened + timedelta(minutes=20 + (i % 40)),
        pnl=pnl,
        mae=-abs(pnl) * 0.4,
        mfe=max(0.0, pnl) * 1.5 + 0.2,
        session=("asia", "london", "new_york")[i % 3],
        regime=("trend", "range")[i % 2],
        news_window=(i % 5 == 0),
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="xau-validation-fixture-v1",
        fingerprint="sha256:test-fixture",
        symbol="XAUUSD",
        start=START,
        end=START + timedelta(days=365),
        source="synthetic-ci-fixture",
        bar_count=100_000,
    )


def _spec(**kwargs) -> ExperimentSpec:
    defaults = dict(
        experiment_id="validation-ci-v1",
        strategy_fingerprint="strategy:test",
        dataset=_manifest(),
        in_sample_fraction=0.6,
        out_of_sample_fraction=0.4,
        walk_forward_train=60,
        walk_forward_test=20,
        walk_forward_step=20,
        monte_carlo_iterations=500,
        monte_carlo_seed=123,
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
    defaults.update(kwargs)
    return ExperimentSpec(**defaults)


def _stable_trades(n: int = 200) -> list[TradeRecord]:
    pattern = (1.2, 0.9, 0.7, -0.55, 0.8, 1.0, -0.45, 0.65)
    return [_trade(i, pattern[i % len(pattern)]) for i in range(n)]


def _stable_surface() -> list[ParameterPoint]:
    return [
        ParameterPoint({"fast": 8, "slow": 20}, 1.04, 180),
        ParameterPoint({"fast": 9, "slow": 20}, 1.08, 190),
        ParameterPoint({"fast": 9, "slow": 21}, 1.10, 200),
        ParameterPoint({"fast": 10, "slow": 21}, 1.07, 194),
        ParameterPoint({"fast": 10, "slow": 22}, 1.02, 184),
        ParameterPoint({"fast": 11, "slow": 22}, 0.98, 178),
    ]


def _wf_results(trades: list[TradeRecord]) -> list[WalkForwardFoldResult]:
    folds = walk_forward_splits(trades, train_size=60, test_size=20, step=20)
    results: list[WalkForwardFoldResult] = []
    for fold in folds:
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
                selected_strategy_fingerprint=f"strategy:fold:{fold.index}",
                selected_parameters={"fast": 9, "slow": 21},
                test_trades=tuple(test),
            )
        )
    return results


def _validate(trades: list[TradeRecord], **kwargs):
    defaults = dict(
        spec=_spec(),
        trades=trades,
        parameter_surface=_stable_surface(),
        parameter_surface_is_in_sample_only=True,
        walk_forward_results=_wf_results(trades),
    )
    defaults.update(kwargs)
    return validate_experiment(**defaults)


def test_performance_includes_expectancy_drawdown_mae_mfe_and_holding_time():
    metrics = compute_performance([_trade(0, 2.0), _trade(1, -1.0), _trade(2, 1.0)])
    assert metrics.trade_count == 3
    assert metrics.net_pnl == pytest.approx(2.0)
    assert metrics.expectancy == pytest.approx(2 / 3)
    assert metrics.profit_factor == pytest.approx(3.0)
    assert metrics.max_drawdown == pytest.approx(1.0)
    assert metrics.avg_mae is not None
    assert metrics.avg_mfe is not None
    assert metrics.avg_holding_seconds is not None


def test_chronological_split_never_randomizes_oos():
    trades = list(reversed(_stable_trades(10)))
    ins, oos = chronological_split(trades, 0.6)
    assert len(ins) == 6
    assert len(oos) == 4
    assert max(t.closed_at for t in ins) < min(t.closed_at for t in oos)


def test_walk_forward_planner_uses_strict_train_then_future_test_windows():
    folds = walk_forward_splits(_stable_trades(120), train_size=60, test_size=20, step=20)
    assert len(folds) == 3
    for fold in folds:
        assert max(t.closed_at for t in fold.train) < min(t.closed_at for t in fold.test)


def test_walk_forward_evidence_rejects_bad_chronology():
    trades = _stable_trades(80)
    with pytest.raises(ValueError, match="train -> select -> future test"):
        WalkForwardFoldResult(
            fold_index=0,
            train_start=trades[0].opened_at,
            train_end=trades[59].closed_at,
            selected_at=trades[70].opened_at,
            test_start=trades[60].opened_at,
            test_end=trades[79].closed_at,
            selected_strategy_fingerprint="leaky",
            selected_parameters={"x": 1},
            test_trades=tuple(trades[60:80]),
        )


def test_monte_carlo_is_reproducible_with_fixed_seed():
    trades = _stable_trades(80)
    a = bootstrap_trade_sequences(trades, iterations=300, seed=99)
    b = bootstrap_trade_sequences(trades, iterations=300, seed=99)
    assert a == b
    assert a["loss_probability"] == 0.0
    assert a["max_drawdown_p95"] is not None


def test_parameter_surface_rewards_plateau_not_single_lucky_point():
    stable = analyze_parameter_stability(_stable_surface())
    brittle = analyze_parameter_stability(
        [
            ParameterPoint({"x": 1}, 10.0),
            ParameterPoint({"x": 2}, -1.0),
            ParameterPoint({"x": 3}, -1.0),
            ParameterPoint({"x": 4}, -1.0),
            ParameterPoint({"x": 5}, -1.0),
        ]
    )
    assert stable["stability_score"] > 0.5
    assert brittle["stability_score"] < stable["stability_score"]


def test_cost_stress_exposes_expectancy_sensitivity():
    stressed = cost_stress(_stable_trades(80), (0.0, 0.25, 0.75, 1.25))
    scenarios = stressed["scenarios"]
    assert scenarios[0]["expectancy"] > scenarios[-1]["expectancy"]
    assert stressed["expectancy_break_even_cost"] is not None


def test_robust_fixture_passes_all_evidence_gates():
    report = _validate(
        _stable_trades(),
        cost_per_trade_levels=(0.0, 0.10, 0.25, 0.50),
    )
    assert report.verdict is ValidationVerdict.PASS
    assert report.out_of_sample.trade_count == 80
    assert report.walk_forward["fold_count"] > 0
    assert report.walk_forward["provenance"] == "train_select_future_test"
    assert report.walk_forward["positive_test_fraction"] == 1.0
    assert report.parameter_stability["stability_score"] > 0.5
    assert report.monte_carlo["loss_probability"] <= 0.10
    assert report.segmentation["session"]
    assert report.segmentation["regime"]
    assert report.segmentation["news"]
    assert all(g.passed is True for g in report.gates)


def test_overfit_fixture_wins_in_sample_but_fails_out_of_sample():
    pnls = [1.0] * 120 + [-0.8] * 80
    trades = [_trade(i, pnl) for i, pnl in enumerate(pnls)]
    report = _validate(trades)
    assert report.in_sample.expectancy > 0
    assert report.out_of_sample.expectancy < 0
    assert report.verdict is ValidationVerdict.FAIL
    failed = {g.name for g in report.gates if g.passed is False}
    assert "oos_expectancy" in failed
    assert "oos_profit_factor" in failed


def test_missing_real_walk_forward_evidence_is_not_silently_inferred():
    trades = _stable_trades()
    report = validate_experiment(
        spec=_spec(),
        trades=trades,
        parameter_surface=_stable_surface(),
        parameter_surface_is_in_sample_only=True,
        walk_forward_results=None,
    )
    assert report.verdict is ValidationVerdict.INSUFFICIENT_EVIDENCE
    gate = next(g for g in report.gates if g.name == "walk_forward_positive_fraction")
    assert gate.passed is None


def test_missing_parameter_surface_is_not_silently_treated_as_stable():
    trades = _stable_trades()
    report = validate_experiment(
        spec=_spec(),
        trades=trades,
        parameter_surface=None,
        walk_forward_results=_wf_results(trades),
    )
    assert report.verdict is ValidationVerdict.INSUFFICIENT_EVIDENCE
    stability_gate = next(g for g in report.gates if g.name == "parameter_stability")
    assert stability_gate.passed is None
    assert any("unproven" in item for item in report.limitations)


def test_parameter_surface_that_touched_oos_is_rejected():
    trades = _stable_trades()
    report = validate_experiment(
        spec=_spec(),
        trades=trades,
        parameter_surface=_stable_surface(),
        parameter_surface_is_in_sample_only=False,
        walk_forward_results=_wf_results(trades),
    )
    assert report.verdict is ValidationVerdict.FAIL
    gate = next(g for g in report.gates if g.name == "parameter_stability")
    assert gate.passed is False
    assert "contaminated" in gate.detail


def test_strategy_not_frozen_before_oos_is_rejected():
    trades = _stable_trades()
    report = validate_experiment(
        spec=_spec(strategy_frozen_before_oos=False),
        trades=trades,
        parameter_surface=_stable_surface(),
        parameter_surface_is_in_sample_only=True,
        walk_forward_results=_wf_results(trades),
    )
    assert report.verdict is ValidationVerdict.FAIL
    gate = next(g for g in report.gates if g.name == "strategy_frozen_before_oos")
    assert gate.passed is False


def test_trade_contract_rejects_invalid_mae_and_mfe_signs():
    opened = START
    with pytest.raises(ValueError, match="mae"):
        TradeRecord("x", opened, opened + timedelta(minutes=1), 1.0, mae=0.1)
    with pytest.raises(ValueError, match="mfe"):
        TradeRecord("x", opened, opened + timedelta(minutes=1), 1.0, mfe=-0.1)
