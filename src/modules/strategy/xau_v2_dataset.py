"""Real XAUUSD historical dataset acquisition and audit for Strategy v2.

The builder is deliberately strict:
- all spot timeframes come from one XAUUSD provider family;
- there is no silent fallback from XAUUSD to GC=F;
- every experiment receives a deterministic dataset fingerprint;
- short intraday history may be wiring-valid but is never promoted to
  long-horizon edge evidence.

Biquote/MT5 can provide deep H1/H4/D1 history efficiently. Minute history is
more expensive, so the audit separates infrastructure readiness from evidence
sufficiency instead of pretending a short smoke window proves durable edge.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from src.modules.strategy.xau_v2_replay import XAUV2HistoricalDataset
from src.platform.marketdata.xau_biquote import BiquoteXAUOHLCProvider
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


_REQUIRED = (
    XAUTimeframe.M1,
    XAUTimeframe.M5,
    XAUTimeframe.M15,
    XAUTimeframe.H1,
    XAUTimeframe.H4,
    XAUTimeframe.D1,
)


class XAUV2DatasetReadiness(str, Enum):
    INVALID = "invalid"
    WIRING_VALID = "wiring_valid"
    RESEARCH_CANDIDATE = "research_candidate"


class XAUHistoricalProvider(Protocol):
    def bars_range(
        self,
        timeframe: XAUTimeframe,
        *,
        start: datetime,
        end: datetime,
        timeout_seconds: float = 12.0,
        max_bars_per_request: int = 900,
        max_chunks: int = 64,
    ) -> list[XAUBar]: ...


@dataclass(frozen=True)
class XAUV2DatasetPlan:
    end: datetime
    lookback_days: dict[XAUTimeframe, float] = field(
        default_factory=lambda: {
            XAUTimeframe.M1: 10.0,
            XAUTimeframe.M5: 30.0,
            XAUTimeframe.M15: 90.0,
            XAUTimeframe.H1: 450.0,
            XAUTimeframe.H4: 900.0,
            XAUTimeframe.D1: 3650.0,
        }
    )
    minimum_bars: dict[XAUTimeframe, int] = field(
        default_factory=lambda: {
            XAUTimeframe.M1: 1000,
            XAUTimeframe.M5: 1000,
            XAUTimeframe.M15: 1000,
            XAUTimeframe.H1: 1000,
            XAUTimeframe.H4: 1000,
            XAUTimeframe.D1: 1000,
        }
    )
    minimum_intraday_research_days: float = 365.0
    provider_timeout_seconds: float = 20.0
    max_chunks_per_timeframe: int = 64
    dataset_id: str = "xau-v2-biquote"
    profile: str = "research_wiring"

    def __post_init__(self) -> None:
        end = self.end
        if end.tzinfo is None:
            object.__setattr__(self, "end", end.replace(tzinfo=timezone.utc))
        else:
            object.__setattr__(self, "end", end.astimezone(timezone.utc))
        for timeframe in _REQUIRED:
            if float(self.lookback_days.get(timeframe, 0.0)) <= 0:
                raise ValueError(f"lookback_days missing/invalid for {timeframe.value}")
            if int(self.minimum_bars.get(timeframe, 0)) < 1:
                raise ValueError(f"minimum_bars missing/invalid for {timeframe.value}")
        if self.minimum_intraday_research_days <= 0:
            raise ValueError("minimum_intraday_research_days must be positive")
        if self.max_chunks_per_timeframe < 1:
            raise ValueError("max_chunks_per_timeframe must be positive")

    @classmethod
    def real_smoke(cls, end: datetime) -> "XAUV2DatasetPlan":
        """Small real-network plan used to verify Biquote plumbing in CI.

        It intentionally cannot satisfy the long-horizon M1 evidence gate.
        """

        return cls(
            end=end,
            lookback_days={
                XAUTimeframe.M1: 0.5,
                XAUTimeframe.M5: 3.0,
                XAUTimeframe.M15: 10.0,
                XAUTimeframe.H1: 90.0,
                XAUTimeframe.H4: 300.0,
                XAUTimeframe.D1: 1800.0,
            },
            minimum_bars={
                XAUTimeframe.M1: 100,
                XAUTimeframe.M5: 300,
                XAUTimeframe.M15: 400,
                XAUTimeframe.H1: 1000,
                XAUTimeframe.H4: 1000,
                XAUTimeframe.D1: 1000,
            },
            minimum_intraday_research_days=365.0,
            max_chunks_per_timeframe=32,
            dataset_id="xau-v2-biquote-real-smoke",
            profile="real_network_smoke",
        )


@dataclass(frozen=True)
class XAUV2FrameAudit:
    timeframe: str
    bar_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    coverage_days: float
    duplicate_timestamps: int
    minimum_bars: int
    minimum_met: bool
    ema1000_supported: bool
    symbol_set: tuple[str, ...]
    source_set: tuple[str, ...]


@dataclass(frozen=True)
class XAUV2DatasetAudit:
    readiness: XAUV2DatasetReadiness
    valid: bool
    wiring_ready: bool
    edge_claim_ready: bool
    dataset_id: str
    dataset_fingerprint: str
    profile: str
    instrument: str
    provider_family: str
    frames: dict[str, XAUV2FrameAudit]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    minimum_intraday_research_days: float
    observed_intraday_coverage_days: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness": self.readiness.value,
            "valid": self.valid,
            "wiring_ready": self.wiring_ready,
            "edge_claim_ready": self.edge_claim_ready,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "profile": self.profile,
            "instrument": self.instrument,
            "provider_family": self.provider_family,
            "frames": {
                key: {
                    "timeframe": item.timeframe,
                    "bar_count": item.bar_count,
                    "first_timestamp": item.first_timestamp,
                    "last_timestamp": item.last_timestamp,
                    "coverage_days": item.coverage_days,
                    "duplicate_timestamps": item.duplicate_timestamps,
                    "minimum_bars": item.minimum_bars,
                    "minimum_met": item.minimum_met,
                    "ema1000_supported": item.ema1000_supported,
                    "symbol_set": list(item.symbol_set),
                    "source_set": list(item.source_set),
                }
                for key, item in self.frames.items()
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "minimum_intraday_research_days": self.minimum_intraday_research_days,
            "observed_intraday_coverage_days": self.observed_intraday_coverage_days,
        }


@dataclass(frozen=True)
class XAUV2DatasetBuildResult:
    dataset: XAUV2HistoricalDataset
    audit: XAUV2DatasetAudit


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _frame_audit(
    timeframe: XAUTimeframe,
    rows: tuple[XAUBar, ...],
    minimum_bars: int,
) -> XAUV2FrameAudit:
    ordered = sorted(rows, key=lambda item: _utc(item.timestamp))
    timestamps = [_utc(row.timestamp) for row in ordered]
    unique = len(set(timestamps))
    duplicate_count = len(timestamps) - unique
    first = timestamps[0] if timestamps else None
    last = timestamps[-1] if timestamps else None
    coverage_days = (
        max(0.0, (last - first).total_seconds() / 86400.0)
        if first is not None and last is not None
        else 0.0
    )
    symbols = tuple(sorted({str(row.symbol) for row in ordered}))
    sources = tuple(sorted({str(row.source) for row in ordered}))
    return XAUV2FrameAudit(
        timeframe=timeframe.value,
        bar_count=len(ordered),
        first_timestamp=first.isoformat() if first else None,
        last_timestamp=last.isoformat() if last else None,
        coverage_days=round(coverage_days, 6),
        duplicate_timestamps=duplicate_count,
        minimum_bars=int(minimum_bars),
        minimum_met=len(ordered) >= int(minimum_bars),
        ema1000_supported=len(ordered) >= 1000,
        symbol_set=symbols,
        source_set=sources,
    )


def audit_xau_v2_dataset(
    dataset: XAUV2HistoricalDataset,
    plan: XAUV2DatasetPlan,
) -> XAUV2DatasetAudit:
    errors: list[str] = []
    warnings: list[str] = []
    frame_audits: dict[str, XAUV2FrameAudit] = {}

    for timeframe in _REQUIRED:
        rows = tuple(dataset.bars_by_timeframe.get(timeframe) or ())
        audit = _frame_audit(
            timeframe,
            rows,
            plan.minimum_bars[timeframe],
        )
        frame_audits[timeframe.value] = audit

        if not rows:
            errors.append(f"missing_{timeframe.value}_bars")
            continue
        if audit.duplicate_timestamps:
            errors.append(f"duplicate_{timeframe.value}_timestamps")
        if any(symbol != "XAUUSD" for symbol in audit.symbol_set):
            errors.append(f"instrument_identity_mismatch_{timeframe.value}")
        if not audit.minimum_met:
            errors.append(f"minimum_{timeframe.value}_bars_not_met")
        if any(
            not source.lower().startswith("biquote.io:mt5-ohlc")
            for source in audit.source_set
        ):
            errors.append(f"provider_family_mismatch_{timeframe.value}")

    m1 = frame_audits.get(XAUTimeframe.M1.value)
    intraday_coverage = m1.coverage_days if m1 is not None else 0.0

    deep_ema_frames = (
        XAUTimeframe.H1.value,
        XAUTimeframe.H4.value,
        XAUTimeframe.D1.value,
    )
    missing_ema1000 = [
        name
        for name in deep_ema_frames
        if name not in frame_audits or not frame_audits[name].ema1000_supported
    ]
    if missing_ema1000:
        warnings.append(
            "ema1000_not_supported_on:" + ",".join(missing_ema1000)
        )

    if intraday_coverage < plan.minimum_intraday_research_days:
        warnings.append(
            "intraday_history_too_short_for_long_horizon_edge_claim:"
            f"{intraday_coverage:.2f}d<"
            f"{plan.minimum_intraday_research_days:.2f}d"
        )

    valid = not errors
    wiring_ready = valid and all(
        item.minimum_met for item in frame_audits.values()
    )
    edge_claim_ready = bool(
        wiring_ready
        and intraday_coverage >= plan.minimum_intraday_research_days
        and not missing_ema1000
    )
    readiness = (
        XAUV2DatasetReadiness.INVALID
        if not valid
        else XAUV2DatasetReadiness.RESEARCH_CANDIDATE
        if edge_claim_ready
        else XAUV2DatasetReadiness.WIRING_VALID
    )

    return XAUV2DatasetAudit(
        readiness=readiness,
        valid=valid,
        wiring_ready=wiring_ready,
        edge_claim_ready=edge_claim_ready,
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.fingerprint,
        profile=plan.profile,
        instrument="XAUUSD",
        provider_family="biquote.io:MT5-ohlc",
        frames=frame_audits,
        errors=tuple(errors),
        warnings=tuple(warnings),
        minimum_intraday_research_days=float(
            plan.minimum_intraday_research_days
        ),
        observed_intraday_coverage_days=intraday_coverage,
    )


class BiquoteXAUV2DatasetBuilder:
    """Fetch all required XAUUSD frames without cross-instrument fallback."""

    def __init__(
        self,
        provider: XAUHistoricalProvider | None = None,
    ) -> None:
        self.provider = provider or BiquoteXAUOHLCProvider()

    def _fetch_one(
        self,
        timeframe: XAUTimeframe,
        plan: XAUV2DatasetPlan,
    ) -> tuple[XAUTimeframe, tuple[XAUBar, ...]]:
        end = _utc(plan.end)
        start = end - timedelta(
            days=float(plan.lookback_days[timeframe])
        )
        rows = self.provider.bars_range(
            timeframe,
            start=start,
            end=end,
            timeout_seconds=float(plan.provider_timeout_seconds),
            max_bars_per_request=900,
            max_chunks=int(plan.max_chunks_per_timeframe),
        )
        ordered = tuple(
            sorted(rows, key=lambda item: _utc(item.timestamp))
        )
        return timeframe, ordered

    def build(
        self,
        plan: XAUV2DatasetPlan,
        *,
        workers: int = 6,
    ) -> XAUV2DatasetBuildResult:
        fetched: dict[XAUTimeframe, tuple[XAUBar, ...]] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 6))) as pool:
            futures = {
                pool.submit(self._fetch_one, timeframe, plan): timeframe
                for timeframe in _REQUIRED
            }
            for future in as_completed(futures):
                timeframe, rows = future.result()
                fetched[timeframe] = rows

        # Fail before constructing XAUV2HistoricalDataset if a provider returned
        # an instrument other than XAUUSD. A proxy belongs in a separate field,
        # never inside the spot timeframe map.
        for timeframe, rows in fetched.items():
            wrong = sorted(
                {
                    str(row.symbol)
                    for row in rows
                    if str(row.symbol) != "XAUUSD"
                }
            )
            if wrong:
                raise ValueError(
                    f"{timeframe.value} returned non-XAUUSD symbols: {wrong}"
                )

        dataset = XAUV2HistoricalDataset(
            bars_by_timeframe=fetched,
            futures_hourly=(),
            source="biquote.io:MT5-ohlc",
            dataset_id=plan.dataset_id,
        )
        audit = audit_xau_v2_dataset(dataset, plan)
        if not audit.valid:
            raise ValueError(
                "real XAU dataset audit failed: "
                + ", ".join(audit.errors)
            )
        return XAUV2DatasetBuildResult(
            dataset=dataset,
            audit=audit,
        )
