"""Executable causal-ablation benchmark for XAU Strategy v2.

Synthetic prices are used only to test research infrastructure invariants.
This benchmark MUST NOT be interpreted as evidence that the strategy has a
profitable real-world edge.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import sin

from src.modules.strategy.xau_v2_replay import (
    XAUV2HistoricalDataset,
    ablation_summary,
    bar_available_at,
    evaluate_xau_v2_at,
    run_xau_v2_ablation,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


UTC = timezone.utc
START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)


def bars(
    timeframe: XAUTimeframe,
    start: datetime,
    count: int,
    delta: timedelta,
    *,
    base: float,
    drift: float,
    amplitude: float,
    source: str = "synthetic-xau",
    symbol: str = "XAUUSD",
) -> tuple[XAUBar, ...]:
    out: list[XAUBar] = []
    previous = base
    for index in range(count):
        timestamp = start + index * delta
        close = base + drift * index + amplitude * sin(index / 11.0)
        out.append(
            XAUBar(
                timestamp=timestamp,
                timeframe=timeframe,
                open=previous,
                high=max(previous, close) + 0.35,
                low=min(previous, close) - 0.35,
                close=close,
                volume=900.0 + (index % 19) * 31.0,
                source=source,
                symbol=symbol,
                execution_eligible=False,
            )
        )
        previous = close
    return tuple(out)


def dataset(*, mutate_future_after: datetime | None = None) -> XAUV2HistoricalDataset:
    hourly_start = START - timedelta(hours=1300)
    daily_start = START - timedelta(days=1120)

    m1 = list(
        bars(
            XAUTimeframe.M1,
            START,
            620,
            timedelta(minutes=1),
            base=4490.0,
            drift=0.02,
            amplitude=0.22,
        )
    )
    if mutate_future_after is not None:
        changed: list[XAUBar] = []
        for row in m1:
            if bar_available_at(row) > mutate_future_after:
                close = row.close - 100.0
                changed.append(
                    replace(
                        row,
                        open=close + 0.1,
                        high=close + 0.5,
                        low=close - 0.5,
                        close=close,
                        source="future-mutated",
                    )
                )
            else:
                changed.append(row)
        m1 = changed

    return XAUV2HistoricalDataset(
        bars_by_timeframe={
            XAUTimeframe.M1: tuple(m1),
            XAUTimeframe.M5: bars(
                XAUTimeframe.M5,
                START,
                130,
                timedelta(minutes=5),
                base=4490.0,
                drift=0.10,
                amplitude=0.30,
            ),
            XAUTimeframe.M15: bars(
                XAUTimeframe.M15,
                START,
                45,
                timedelta(minutes=15),
                base=4490.0,
                drift=0.30,
                amplitude=0.45,
            ),
            XAUTimeframe.H1: bars(
                XAUTimeframe.H1,
                hourly_start,
                1350,
                timedelta(hours=1),
                base=4200.0,
                drift=0.20,
                amplitude=1.1,
            ),
            XAUTimeframe.D1: bars(
                XAUTimeframe.D1,
                daily_start,
                1140,
                timedelta(days=1),
                base=3000.0,
                drift=1.30,
                amplitude=4.0,
            ),
        },
        futures_hourly=bars(
            XAUTimeframe.H1,
            hourly_start,
            1350,
            timedelta(hours=1),
            base=4204.0,
            drift=0.20,
            amplitude=1.0,
            source="synthetic-gc",
            symbol="GC=F",
        ),
        source="synthetic-xau-v2-causality-benchmark",
        dataset_id="xau-v2-ablation-v1",
    )


def main() -> None:
    observed_at = START + timedelta(minutes=390)
    original = dataset()
    mutated = dataset(mutate_future_after=observed_at)
    original_point = evaluate_xau_v2_at(original, observed_at)
    mutated_point = evaluate_xau_v2_at(mutated, observed_at)

    report = run_xau_v2_ablation(
        original,
        start=START + timedelta(minutes=300),
        end=START + timedelta(minutes=540),
        horizon_minutes=45,
        step_minutes=15,
        round_trip_cost_bps=0.5,
    )
    summary = ablation_summary(report)

    non_overlap = True
    for trades in report.trades_by_variant.values():
        ordered = sorted(trades, key=lambda item: item.opened_at)
        for previous, current in zip(ordered, ordered[1:]):
            if current.opened_at < previous.closed_at:
                non_overlap = False

    variant_count = len(summary["variants"])
    variants_with_signals = sum(
        1 for item in summary["variants"] if item["signal_count"] > 0
    )
    causality_passed = (
        original.fingerprint != mutated.fingerprint
        and original_point.decision_fingerprint == mutated_point.decision_fingerprint
    )

    result = {
        "benchmark": "xau-v2-causal-ablation-v1",
        "synthetic_data_warning": (
            "Synthetic benchmark validates causality and plumbing only; "
            "it is not evidence of real trading edge."
        ),
        "causality": {
            "future_mutation_changes_dataset": original.fingerprint != mutated.fingerprint,
            "past_decision_unchanged": (
                original_point.decision_fingerprint
                == mutated_point.decision_fingerprint
            ),
            "passed": causality_passed,
        },
        "ablation": summary,
        "invariants": {
            "variant_count": variant_count,
            "variants_with_signals": variants_with_signals,
            "non_overlapping_trade_ledgers": non_overlap,
            "research_only": summary["research_only"],
            "live_execution_allowed": summary["live_execution_allowed"],
            "realistic_fill_model": summary["realistic_fill_model"],
        },
    }
    result["passed"] = bool(
        causality_passed
        and variant_count == 5
        and variants_with_signals > 0
        and non_overlap
        and summary["research_only"] is True
        and summary["live_execution_allowed"] is False
        and summary["realistic_fill_model"] is False
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
