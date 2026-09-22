from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import sin

from src.modules.strategy.validation import (
    DatasetManifest,
    ExperimentSpec,
    ValidationPolicy,
    ValidationVerdict,
    validate_experiment,
)
from src.modules.strategy.xau_v2_replay import (
    XAUV2HistoricalDataset,
    ablation_summary,
    bar_available_at,
    build_variant_trade_ledger,
    causal_slice,
    evaluate_xau_v2_at,
    run_xau_v2_ablation,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


UTC = timezone.utc
INTRADAY_START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)


def _bars(
    timeframe: XAUTimeframe,
    start: datetime,
    count: int,
    delta: timedelta,
    *,
    base: float = 4400.0,
    drift: float = 0.04,
    amplitude: float = 0.15,
    source: str = "synthetic-xau",
    symbol: str = "XAUUSD",
) -> tuple[XAUBar, ...]:
    rows: list[XAUBar] = []
    previous = base
    for index in range(count):
        timestamp = start + index * delta
        close = base + drift * index + amplitude * sin(index / 9.0)
        open_price = previous
        high = max(open_price, close) + 0.35
        low = min(open_price, close) - 0.35
        rows.append(
            XAUBar(
                timestamp=timestamp,
                timeframe=timeframe,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + (index % 17) * 23.0,
                source=source,
                symbol=symbol,
                execution_eligible=False,
            )
        )
        previous = close
    return tuple(rows)


def _dataset(*, future_drop: bool = False) -> XAUV2HistoricalDataset:
    daily_start = INTRADAY_START - timedelta(days=1200)
    hourly_start = INTRADAY_START - timedelta(hours=1450)

    m1 = list(
        _bars(
            XAUTimeframe.M1,
            INTRADAY_START,
            720,
            timedelta(minutes=1),
            base=4490.0,
            drift=0.018,
            amplitude=0.22,
        )
    )
    if future_drop:
        mutation_cut = INTRADAY_START + timedelta(minutes=430)
        mutated: list[XAUBar] = []
        for row in m1:
            if bar_available_at(row) > mutation_cut:
                shifted = row.close - 80.0
                mutated.append(
                    replace(
                        row,
                        open=shifted + 0.1,
                        high=shifted + 0.4,
                        low=shifted - 0.4,
                        close=shifted,
                        source="synthetic-future-mutated",
                    )
                )
            else:
                mutated.append(row)
        m1 = mutated

    bars = {
        XAUTimeframe.M1: tuple(m1),
        XAUTimeframe.M5: _bars(
            XAUTimeframe.M5,
            INTRADAY_START,
            160,
            timedelta(minutes=5),
            base=4490.0,
            drift=0.09,
            amplitude=0.32,
        ),
        XAUTimeframe.M15: _bars(
            XAUTimeframe.M15,
            INTRADAY_START,
            60,
            timedelta(minutes=15),
            base=4490.0,
            drift=0.27,
            amplitude=0.45,
        ),
        XAUTimeframe.H1: _bars(
            XAUTimeframe.H1,
            hourly_start,
            1500,
            timedelta(hours=1),
            base=4200.0,
            drift=0.18,
            amplitude=1.1,
        ),
        XAUTimeframe.D1: _bars(
            XAUTimeframe.D1,
            daily_start,
            1220,
            timedelta(days=1),
            base=3000.0,
            drift=1.20,
            amplitude=4.0,
        ),
    }
    futures = _bars(
        XAUTimeframe.H1,
        hourly_start,
        1500,
        timedelta(hours=1),
        base=4205.0,
        drift=0.18,
        amplitude=1.0,
        source="synthetic-GC-futures",
        symbol="GC=F",
    )
    return XAUV2HistoricalDataset(
        bars_by_timeframe=bars,
        futures_hourly=futures,
        source="synthetic-causal-ci",
        dataset_id="xau-v2-causal-ci",
    )


def test_causal_slice_excludes_partial_and_future_bars():
    dataset = _dataset()
    observed_at = INTRADAY_START + timedelta(minutes=10, seconds=30)
    visible = causal_slice(
        dataset.bars_by_timeframe[XAUTimeframe.M5],
        XAUTimeframe.M5,
        observed_at,
    )
    assert visible
    assert all(bar_available_at(row) <= observed_at for row in visible)
    assert visible[-1].timestamp == INTRADAY_START + timedelta(minutes=5)


def test_future_mutation_cannot_change_past_strategy_decision():
    original = _dataset(future_drop=False)
    mutated = _dataset(future_drop=True)
    observed_at = INTRADAY_START + timedelta(minutes=430)

    before = evaluate_xau_v2_at(original, observed_at)
    after = evaluate_xau_v2_at(mutated, observed_at)

    assert original.fingerprint != mutated.fingerprint
    assert before.decision_fingerprint == after.decision_fingerprint
    assert (
        before.canonical_decision_payload()["assessments"]
        == after.canonical_decision_payload()["assessments"]
    )


def test_dataset_fingerprint_changes_when_future_data_changes():
    assert _dataset(False).fingerprint != _dataset(True).fingerprint


def test_ablation_replay_produces_normalized_non_overlapping_trade_ledgers():
    dataset = _dataset()
    report = run_xau_v2_ablation(
        dataset,
        start=INTRADAY_START + timedelta(minutes=330),
        end=INTRADAY_START + timedelta(minutes=600),
        horizon_minutes=45,
        step_minutes=15,
        round_trip_cost_bps=0.4,
    )

    assert report.research_only is True
    assert report.live_execution_allowed is False
    assert report.realistic_fill_model is False
    assert len(report.variants) == 5

    by_name = {item.variant: item for item in report.variants}
    assert set(by_name) == {
        "baseline_intraday",
        "plus_htf_momentum",
        "plus_ema_ladder",
        "plus_smart_money",
        "plus_flow_and_profile_context",
    }
    assert any(item.signal_count > 0 for item in report.variants)

    for variant, trades in report.trades_by_variant.items():
        ordered = sorted(trades, key=lambda item: item.opened_at)
        for previous, current in zip(ordered, ordered[1:]):
            assert current.opened_at >= previous.closed_at
        for trade in trades:
            assert trade.metadata["variant"] == variant
            assert trade.metadata["execution_model"] == "research_forward_mid_to_mid_bps"
            assert trade.metadata["realistic_fill_model"] is False
            assert trade.mae is not None and trade.mae <= 0
            assert trade.mfe is not None and trade.mfe >= 0


def test_repeated_signals_are_not_counted_as_independent_overlapping_trades():
    dataset = _dataset()
    report = run_xau_v2_ablation(
        dataset,
        start=INTRADAY_START + timedelta(minutes=330),
        end=INTRADAY_START + timedelta(minutes=600),
        horizon_minutes=60,
        step_minutes=5,
    )

    candidate_variant = max(report.variants, key=lambda item: item.signal_count)
    assert candidate_variant.signal_count >= candidate_variant.trade_count
    assert candidate_variant.skipped_overlapping_signals >= 0


def test_research_cost_assumption_reduces_expectancy():
    dataset = _dataset()
    common = dict(
        dataset=dataset,
        start=INTRADAY_START + timedelta(minutes=330),
        end=INTRADAY_START + timedelta(minutes=600),
        horizon_minutes=45,
        step_minutes=15,
    )
    zero = run_xau_v2_ablation(**common, round_trip_cost_bps=0.0)
    costly = run_xau_v2_ablation(**common, round_trip_cost_bps=2.0)

    zero_map = {item.variant: item for item in zero.variants}
    costly_map = {item.variant: item for item in costly.variants}
    compared = 0
    for name, item in zero_map.items():
        if item.trade_count and costly_map[name].trade_count:
            compared += 1
            assert costly_map[name].metrics.expectancy is not None
            assert item.metrics.expectancy is not None
            assert costly_map[name].metrics.expectancy < item.metrics.expectancy
    assert compared > 0


def test_ablation_summary_cannot_be_mistaken_for_realistic_execution_backtest():
    report = run_xau_v2_ablation(
        _dataset(),
        start=INTRADAY_START + timedelta(minutes=330),
        end=INTRADAY_START + timedelta(minutes=480),
        horizon_minutes=30,
        step_minutes=15,
    )
    summary = ablation_summary(report)
    assert summary["research_only"] is True
    assert summary["live_execution_allowed"] is False
    assert summary["realistic_fill_model"] is False
    assert summary["execution_model"] == "research_forward_mid_to_mid_bps"


def test_replay_trade_ledger_is_directly_accepted_by_validation_framework():
    dataset = _dataset()
    report = run_xau_v2_ablation(
        dataset,
        start=INTRADAY_START + timedelta(minutes=300),
        end=INTRADAY_START + timedelta(minutes=650),
        horizon_minutes=30,
        step_minutes=10,
    )
    variant = max(report.variants, key=lambda item: item.trade_count)
    trades = list(report.trades_by_variant[variant.variant])
    assert trades

    observed_start = min(item.opened_at for item in trades)
    observed_end = max(item.closed_at for item in trades) + timedelta(minutes=1)
    validation = validate_experiment(
        spec=ExperimentSpec(
            experiment_id="xau-v2-replay-validation-contract",
            strategy_fingerprint=variant.strategy_fingerprint,
            dataset=DatasetManifest(
                dataset_id=dataset.dataset_id,
                fingerprint=dataset.fingerprint,
                symbol="XAUUSD",
                start=observed_start,
                end=observed_end,
                source=dataset.source,
            ),
            strategy_frozen_before_oos=True,
            monte_carlo_iterations=200,
            policy=ValidationPolicy(
                min_oos_trades=1,
                min_oos_expectancy=-10_000,
                min_oos_profit_factor=0.0,
                min_walk_forward_positive_fraction=0.0,
                min_parameter_stability_score=0.0,
                max_monte_carlo_loss_probability=1.0,
            ),
        ),
        trades=trades,
        parameter_surface=None,
        walk_forward_results=None,
    )

    # Replay gives us a normalized ledger, but it deliberately cannot earn a
    # scientific PASS without true walk-forward and parameter-stability evidence.
    assert validation.verdict is ValidationVerdict.INSUFFICIENT_EVIDENCE
    assert any("walk_forward" in item for item in validation.limitations)


def test_manual_trade_builder_preserves_non_overlap_guard():
    dataset = _dataset()
    report = run_xau_v2_ablation(
        dataset,
        start=INTRADAY_START + timedelta(minutes=330),
        end=INTRADAY_START + timedelta(minutes=500),
        horizon_minutes=45,
        step_minutes=5,
    )
    variant = max(report.variants, key=lambda item: item.signal_count)
    trades, skipped = build_variant_trade_ledger(
        report.points,
        variant=variant.variant,
        dataset=dataset,
        horizon_minutes=45,
        allow_overlapping=False,
    )
    assert len(trades) == variant.trade_count
    assert skipped == variant.skipped_overlapping_signals
