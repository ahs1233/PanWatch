"""Causal historical replay and ablation harness for XAU Strategy v2.

This is a research layer, not a fill simulator and not a live trading path.

It answers a narrower question than LEAN:
    "Did adding this intelligence layer improve directional signal quality?"

Rules:
- a bar is visible only after its candle duration has elapsed;
- Market Context is rebuilt only from causally available bars;
- future bars are inaccessible while the signal is produced;
- future bars are used only by the separate outcome-labeling step;
- generated TradeRecord PnL is forward mid-to-mid return in basis points,
  optionally net of an explicit research cost assumption;
- this ledger is suitable for feature ablation and Validation Framework input,
  but it is NOT evidence of realistic broker fills. LEAN remains the execution
  and fill-model kernel for that stage.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from hashlib import sha256
from typing import Any, Iterable

from src.modules.strategy.validation.metrics import compute_performance
from src.modules.strategy.validation.models import PerformanceMetrics, TradeRecord
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.strategy.xau_v2 import (
    XAUStrategyV2Assessment,
    XAUStrategyV2Spec,
    assess_xau_strategy_v2,
    xau_v2_ablation_specs,
)
from src.modules.xau.market_context import build_market_context
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


_TIMEFRAME_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.M30: timedelta(minutes=30),
    XAUTimeframe.H1: timedelta(hours=1),
    XAUTimeframe.H4: timedelta(hours=4),
    XAUTimeframe.D1: timedelta(days=1),
}


@dataclass(frozen=True)
class XAUV2HistoricalDataset:
    """One immutable identity for a replay dataset.

    Weekly/monthly context is derived causally from daily bars by market_context;
    direct W1/MN1 bars are deliberately not required here because calendar-close
    semantics are less portable across providers.
    """

    bars_by_timeframe: dict[XAUTimeframe, tuple[XAUBar, ...]]
    futures_hourly: tuple[XAUBar, ...] = ()
    source: str = "historical"
    dataset_id: str = "xau-v2-historical"

    def __post_init__(self) -> None:
        required = {
            XAUTimeframe.M1,
            XAUTimeframe.M5,
            XAUTimeframe.M15,
            XAUTimeframe.H1,
            XAUTimeframe.D1,
        }
        missing = [
            timeframe.value
            for timeframe in required
            if not self.bars_by_timeframe.get(timeframe)
        ]
        if missing:
            raise ValueError(
                "XAU v2 historical dataset missing required timeframes: "
                + ", ".join(sorted(missing))
            )

    @cached_property
    def fingerprint(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "source": self.source,
            "bars": {
                timeframe.value: [
                    _bar_identity(row)
                    for row in sorted(rows, key=lambda item: _utc(item.timestamp))
                ]
                for timeframe, rows in sorted(
                    self.bars_by_timeframe.items(),
                    key=lambda item: item[0].value,
                )
            },
            "futures_hourly": [
                _bar_identity(row)
                for row in sorted(
                    self.futures_hourly,
                    key=lambda item: _utc(item.timestamp),
                )
            ],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class XAUV2ReplayPoint:
    observed_at: datetime
    entry_reference: float | None
    dataset_fingerprint: str
    assessments: dict[str, XAUStrategyV2Assessment]
    causal_bar_counts: dict[str, int]
    source: str

    def canonical_decision_payload(self) -> dict[str, Any]:
        """Only information available at decision time.

        This is intentionally usable in causality tests: mutating bars that
        become available after observed_at must not change this payload.
        """

        return {
            "observed_at": _utc(self.observed_at).isoformat(),
            "entry_reference": self.entry_reference,
            "dataset_fingerprint": self.dataset_fingerprint,
            "causal_bar_counts": dict(self.causal_bar_counts),
            "assessments": {
                name: {
                    "strategy_fingerprint": result.strategy_fingerprint,
                    "status": result.status,
                    "candidate": result.candidate,
                    "score": result.score,
                    "directional_feature_count": result.directional_feature_count,
                    "block_reasons": list(result.block_reasons),
                    "warnings": list(result.warnings),
                    "observations": {
                        key: {
                            "available": obs.available,
                            "score": obs.score,
                            "contributing": obs.contributing,
                            "provenance_group": obs.provenance_group,
                            "detail": obs.detail,
                        }
                        for key, obs in result.observations.items()
                    },
                    "context": result.context,
                }
                for name, result in sorted(self.assessments.items())
            },
        }

    @property
    def decision_fingerprint(self) -> str:
        payload = self.canonical_decision_payload().copy()
        # Dataset fingerprint includes future rows by design. It identifies the
        # experiment but must not make a past decision depend on future data.
        payload.pop("dataset_fingerprint", None)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class XAUV2AblationVariantResult:
    variant: str
    strategy_fingerprint: str
    evaluation_count: int
    signal_count: int
    trade_count: int
    skipped_overlapping_signals: int
    metrics: PerformanceMetrics
    execution_model: str = "research_forward_mid_to_mid_bps"
    realistic_fill_model: bool = False


@dataclass(frozen=True)
class XAUV2AblationReport:
    dataset_id: str
    dataset_fingerprint: str
    source: str
    horizon_minutes: int
    step_minutes: int
    round_trip_cost_bps: float
    points: tuple[XAUV2ReplayPoint, ...]
    trades_by_variant: dict[str, tuple[TradeRecord, ...]]
    variants: tuple[XAUV2AblationVariantResult, ...]
    research_only: bool = True
    live_execution_allowed: bool = False
    realistic_fill_model: bool = False


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _bar_identity(row: XAUBar) -> dict[str, Any]:
    return {
        "timestamp": _utc(row.timestamp).isoformat(),
        "timeframe": row.timeframe.value,
        "open": round(float(row.open), 8),
        "high": round(float(row.high), 8),
        "low": round(float(row.low), 8),
        "close": round(float(row.close), 8),
        "volume": None if row.volume is None else round(float(row.volume), 8),
        "source": row.source,
        "symbol": row.symbol,
    }


def bar_available_at(bar: XAUBar) -> datetime:
    try:
        duration = _TIMEFRAME_DURATION[bar.timeframe]
    except KeyError as exc:
        raise ValueError(
            f"direct replay availability is unsupported for {bar.timeframe.value}; "
            "derive weekly/monthly state from causally available daily bars"
        ) from exc
    return _utc(bar.timestamp) + duration


def causal_slice(
    rows: Iterable[XAUBar],
    timeframe: XAUTimeframe,
    observed_at: datetime,
    *,
    max_bars: int | None = None,
) -> list[XAUBar]:
    """Return only bars fully closed at observed_at."""

    ordered = sorted(rows, key=lambda item: _utc(item.timestamp))
    if not ordered:
        return []
    observed_at = _utc(observed_at)
    availability = [bar_available_at(row) for row in ordered]
    end = bisect_right(availability, observed_at)
    start = 0 if max_bars is None else max(0, end - max(1, int(max_bars)))
    visible = ordered[start:end]

    # Fail closed if a future/partial candle slips through.
    if any(bar_available_at(row) > observed_at for row in visible):
        raise RuntimeError("causal_slice admitted a future bar")
    if any(row.timeframe is not timeframe for row in visible):
        raise ValueError("causal_slice received mixed timeframes")
    return visible


def _causal_inputs(
    dataset: XAUV2HistoricalDataset,
    observed_at: datetime,
) -> tuple[dict[XAUTimeframe, list[XAUBar]], list[XAUBar], dict[str, int]]:
    observed_at = _utc(observed_at)
    limits = {
        XAUTimeframe.M1: 1500,
        XAUTimeframe.M5: 800,
        XAUTimeframe.M15: 600,
        XAUTimeframe.H1: 1600,
        XAUTimeframe.H4: 1200,
        XAUTimeframe.D1: 1600,
    }
    visible: dict[XAUTimeframe, list[XAUBar]] = {}
    for timeframe, limit in limits.items():
        rows = dataset.bars_by_timeframe.get(timeframe) or ()
        if rows:
            visible[timeframe] = causal_slice(
                rows,
                timeframe,
                observed_at,
                max_bars=limit,
            )

    futures = causal_slice(
        dataset.futures_hourly,
        XAUTimeframe.H1,
        observed_at,
        max_bars=400,
    ) if dataset.futures_hourly else []

    counts = {
        timeframe.value: len(rows)
        for timeframe, rows in visible.items()
    }
    counts["gc_futures_1h"] = len(futures)
    return visible, futures, counts


def evaluate_xau_v2_at(
    dataset: XAUV2HistoricalDataset,
    observed_at: datetime,
    *,
    variants: tuple[tuple[str, XAUStrategyV2Spec], ...] | None = None,
) -> XAUV2ReplayPoint:
    """Evaluate all Strategy v2 variants using past-and-present closed bars only."""

    observed_at = _utc(observed_at)
    visible, futures, counts = _causal_inputs(dataset, observed_at)

    intraday_bars = {
        timeframe: visible.get(timeframe, [])
        for timeframe in (
            XAUTimeframe.M1,
            XAUTimeframe.M5,
            XAUTimeframe.M15,
        )
    }
    intraday = XAUIntradayEngine(require_execution_data=False).analyze(
        intraday_bars,
        now=observed_at,
        event_risk=False,
        macro_bias=0,
    )

    hourly = visible.get(XAUTimeframe.H1, [])
    h4 = visible.get(XAUTimeframe.H4, [])
    daily = visible.get(XAUTimeframe.D1, [])

    # Market context itself contains explicit proxy disclosures. All inputs here
    # are causal slices; future outcomes are not passed into this function.
    market_context = build_market_context(
        hourly=hourly,
        h4=h4,
        daily=daily,
        futures_hourly=futures,
        deep_hourly=hourly,
        deep_daily=daily,
    )

    configured = variants or xau_v2_ablation_specs()
    assessments = {
        name: assess_xau_strategy_v2(
            spec=spec,
            intraday=intraday,
            market_context=market_context,
        )
        for name, spec in configured
    }

    m1 = visible.get(XAUTimeframe.M1, [])
    entry_reference = float(m1[-1].close) if m1 else None
    return XAUV2ReplayPoint(
        observed_at=observed_at,
        entry_reference=entry_reference,
        dataset_fingerprint=dataset.fingerprint,
        assessments=assessments,
        causal_bar_counts=counts,
        source=dataset.source,
    )


def replay_evaluation_times(
    dataset: XAUV2HistoricalDataset,
    *,
    step_minutes: int = 15,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[datetime]:
    """Generate evaluation times from fully closed M1 bars."""

    step = timedelta(minutes=max(1, int(step_minutes)))
    rows = sorted(
        dataset.bars_by_timeframe[XAUTimeframe.M1],
        key=lambda item: _utc(item.timestamp),
    )
    start_utc = _utc(start) if start is not None else None
    end_utc = _utc(end) if end is not None else None
    out: list[datetime] = []
    last: datetime | None = None

    for row in rows:
        observed_at = bar_available_at(row)
        if start_utc is not None and observed_at < start_utc:
            continue
        if end_utc is not None and observed_at > end_utc:
            continue
        if last is None or observed_at - last >= step:
            out.append(observed_at)
            last = observed_at
    return out


def _future_outcome(
    m1: tuple[XAUBar, ...],
    observed_at: datetime,
    *,
    horizon_minutes: int,
    tolerance_minutes: int = 2,
) -> tuple[XAUBar, datetime] | None:
    ordered = sorted(m1, key=lambda item: _utc(item.timestamp))
    availability = [bar_available_at(row) for row in ordered]
    target = _utc(observed_at) + timedelta(minutes=max(1, int(horizon_minutes)))
    index = bisect_left(availability, target)
    if index >= len(ordered):
        return None
    outcome_at = availability[index]
    if outcome_at - target > timedelta(minutes=max(0, int(tolerance_minutes))):
        return None
    return ordered[index], outcome_at


def _path_bars(
    m1: tuple[XAUBar, ...],
    observed_at: datetime,
    outcome_at: datetime,
) -> list[XAUBar]:
    ordered = sorted(m1, key=lambda item: _utc(item.timestamp))
    return [
        row
        for row in ordered
        if _utc(observed_at) < bar_available_at(row) <= _utc(outcome_at)
    ]


def _session(observed_at: datetime) -> str:
    hour = _utc(observed_at).hour
    if hour < 7:
        return "asia"
    if hour < 12:
        return "london"
    if hour < 16:
        return "london_ny_overlap"
    if hour < 21:
        return "new_york"
    return "late_us"


def _regime(point: XAUV2ReplayPoint, variant: str) -> str:
    assessment = point.assessments[variant]
    htf = assessment.observations.get("htf_momentum")
    ema = assessment.observations.get("ema_ladder")
    htf_score = htf.score if htf and htf.score is not None else 0.0
    ema_score = ema.score if ema and ema.score is not None else 0.0
    if abs(htf_score) >= 0.35 and abs(ema_score) >= 0.35 and htf_score * ema_score > 0:
        return "trend"
    if htf_score * ema_score < -0.05:
        return "transition"
    return "range_or_mixed"


def _trade_from_signal(
    *,
    point: XAUV2ReplayPoint,
    variant: str,
    dataset: XAUV2HistoricalDataset,
    horizon_minutes: int,
    round_trip_cost_bps: float,
) -> TradeRecord | None:
    assessment = point.assessments[variant]
    candidate = assessment.candidate
    if candidate not in {"long_setup", "short_setup"}:
        return None
    if point.entry_reference is None or point.entry_reference <= 0:
        return None

    future = _future_outcome(
        dataset.bars_by_timeframe[XAUTimeframe.M1],
        point.observed_at,
        horizon_minutes=horizon_minutes,
    )
    if future is None:
        return None
    outcome_bar, outcome_at = future

    entry = float(point.entry_reference)
    exit_price = float(outcome_bar.close)
    direction = 1.0 if candidate == "long_setup" else -1.0
    gross_bps = ((exit_price - entry) / entry) * 10_000.0 * direction
    net_bps = gross_bps - max(0.0, float(round_trip_cost_bps))

    path = _path_bars(
        dataset.bars_by_timeframe[XAUTimeframe.M1],
        point.observed_at,
        outcome_at,
    )
    if path:
        if direction > 0:
            adverse = min((float(row.low) - entry) / entry * 10_000.0 for row in path)
            favorable = max((float(row.high) - entry) / entry * 10_000.0 for row in path)
        else:
            adverse = min((entry - float(row.high)) / entry * 10_000.0 for row in path)
            favorable = max((entry - float(row.low)) / entry * 10_000.0 for row in path)
        mae = min(0.0, adverse)
        mfe = max(0.0, favorable)
    else:
        mae = 0.0
        mfe = 0.0

    material = (
        f"{variant}|{assessment.strategy_fingerprint}|"
        f"{_utc(point.observed_at).isoformat()}|{horizon_minutes}"
    )
    trade_id = sha256(material.encode("utf-8")).hexdigest()[:24]
    return TradeRecord(
        trade_id=trade_id,
        opened_at=_utc(point.observed_at),
        closed_at=_utc(outcome_at),
        pnl=round(net_bps, 8),
        mae=round(mae, 8),
        mfe=round(mfe, 8),
        session=_session(point.observed_at),
        regime=_regime(point, variant),
        news_window=None,
        metadata={
            "variant": variant,
            "strategy_fingerprint": assessment.strategy_fingerprint,
            "candidate": candidate,
            "score": assessment.score,
            "entry_reference": entry,
            "outcome_close": exit_price,
            "gross_forward_return_bps": round(gross_bps, 8),
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "execution_model": "research_forward_mid_to_mid_bps",
            "realistic_fill_model": False,
            "research_only": True,
        },
    )


def build_variant_trade_ledger(
    points: Iterable[XAUV2ReplayPoint],
    *,
    variant: str,
    dataset: XAUV2HistoricalDataset,
    horizon_minutes: int,
    round_trip_cost_bps: float = 0.0,
    allow_overlapping: bool = False,
) -> tuple[list[TradeRecord], int]:
    """Convert signal events into a normalized research ledger.

    By default a new signal is ignored until the prior forward-label horizon is
    finished, preventing repeated overlapping observations from masquerading as
    independent trades.
    """

    trades: list[TradeRecord] = []
    skipped_overlap = 0
    unavailable_until: datetime | None = None

    for point in sorted(points, key=lambda item: _utc(item.observed_at)):
        assessment = point.assessments.get(variant)
        if assessment is None or assessment.candidate not in {"long_setup", "short_setup"}:
            continue
        if (
            not allow_overlapping
            and unavailable_until is not None
            and _utc(point.observed_at) < unavailable_until
        ):
            skipped_overlap += 1
            continue

        trade = _trade_from_signal(
            point=point,
            variant=variant,
            dataset=dataset,
            horizon_minutes=horizon_minutes,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        if trade is None:
            continue
        trades.append(trade)
        unavailable_until = trade.closed_at

    return trades, skipped_overlap


def run_xau_v2_ablation(
    dataset: XAUV2HistoricalDataset,
    *,
    horizon_minutes: int = 60,
    step_minutes: int = 15,
    round_trip_cost_bps: float = 0.0,
    start: datetime | None = None,
    end: datetime | None = None,
    variants: tuple[tuple[str, XAUStrategyV2Spec], ...] | None = None,
) -> XAUV2AblationReport:
    configured = variants or xau_v2_ablation_specs()
    times = replay_evaluation_times(
        dataset,
        step_minutes=step_minutes,
        start=start,
        end=end,
    )
    points = tuple(
        evaluate_xau_v2_at(dataset, observed_at, variants=configured)
        for observed_at in times
    )

    trades_by_variant: dict[str, tuple[TradeRecord, ...]] = {}
    results: list[XAUV2AblationVariantResult] = []
    for name, spec in configured:
        trades, skipped = build_variant_trade_ledger(
            points,
            variant=name,
            dataset=dataset,
            horizon_minutes=horizon_minutes,
            round_trip_cost_bps=round_trip_cost_bps,
            allow_overlapping=False,
        )
        trades_by_variant[name] = tuple(trades)
        signal_count = sum(
            1
            for point in points
            if point.assessments[name].candidate in {"long_setup", "short_setup"}
        )
        results.append(
            XAUV2AblationVariantResult(
                variant=name,
                strategy_fingerprint=spec.fingerprint,
                evaluation_count=len(points),
                signal_count=signal_count,
                trade_count=len(trades),
                skipped_overlapping_signals=skipped,
                metrics=compute_performance(trades),
            )
        )

    return XAUV2AblationReport(
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.fingerprint,
        source=dataset.source,
        horizon_minutes=max(1, int(horizon_minutes)),
        step_minutes=max(1, int(step_minutes)),
        round_trip_cost_bps=max(0.0, float(round_trip_cost_bps)),
        points=points,
        trades_by_variant=trades_by_variant,
        variants=tuple(results),
    )


def ablation_summary(report: XAUV2AblationReport) -> dict[str, Any]:
    return {
        "dataset_id": report.dataset_id,
        "dataset_fingerprint": report.dataset_fingerprint,
        "source": report.source,
        "horizon_minutes": report.horizon_minutes,
        "step_minutes": report.step_minutes,
        "round_trip_cost_bps": report.round_trip_cost_bps,
        "research_only": report.research_only,
        "live_execution_allowed": report.live_execution_allowed,
        "execution_model": "research_forward_mid_to_mid_bps",
        "realistic_fill_model": False,
        "variants": [
            {
                "variant": item.variant,
                "strategy_fingerprint": item.strategy_fingerprint,
                "evaluation_count": item.evaluation_count,
                "signal_count": item.signal_count,
                "trade_count": item.trade_count,
                "skipped_overlapping_signals": item.skipped_overlapping_signals,
                "metrics": asdict(item.metrics),
            }
            for item in report.variants
        ],
    }
