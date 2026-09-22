"""Real-network Dukascopy deep-history smoke for XAUUSD.

The smoke proves that:
- public BI5 data can be decoded as XAUUSD with sane price scale;
- an old historical hour and a recent hour are both accessible;
- old data predates Biquote's observed retention by years;
- emitted bars are research-only and preserve Dukascopy provenance.

It deliberately does NOT download a multi-year dataset in CI.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.platform.marketdata.xau_dukascopy import DukascopyXAUHistoryProvider
from src.platform.marketdata.xau_models import XAUTimeframe


UTC = timezone.utc


def _find_data_hour(provider, candidates):
    attempts = []
    for hour in candidates:
        fetched = provider.fetch_hour(hour)
        attempts.append(
            {
                "hour": hour.isoformat(),
                "status": fetched.status,
                "ticks": len(fetched.ticks),
                "attempts": fetched.attempts,
            }
        )
        if fetched.status == "data" and fetched.ticks:
            return fetched, attempts
    return None, attempts


def _modern_candidates():
    # Fixed known trading session avoids conflating deep-history access with
    # Dukascopy's separate latest-file publication lag.
    start = datetime(2026, 7, 3, 12, tzinfo=UTC)
    return [start + timedelta(hours=i) for i in range(0, 6)]


def _old_candidates():
    start = datetime(2020, 1, 6, 12, tzinfo=UTC)
    return [start + timedelta(hours=i) for i in range(0, 8)]


def main() -> None:
    provider = DukascopyXAUHistoryProvider(
        timeout_seconds=8.0,
        retries=1,
        retry_backoff_seconds=0.25,
    )

    old, old_attempts = _find_data_hour(provider, _old_candidates())
    recent, recent_attempts = _find_data_hour(provider, _modern_candidates())

    result = {
        "benchmark": "xau-dukascopy-deep-history-smoke-v1",
        "provider": provider.source,
        "symbol": provider.symbol,
        "price_basis": provider.price_basis,
        "volume_semantics": provider.volume_semantics,
        "centralized_order_flow": provider.centralized_order_flow,
        "execution_eligible": provider.execution_eligible,
        "old_search": old_attempts,
        "modern_search": recent_attempts,
        "old_hour": None,
        "modern_hour": None,
        "deep_history_gap_days": None,
        "passed": False,
    }

    if old is not None:
        mids = [tick.mid for tick in old.ticks]
        old_bars, old_diag = provider.bars_range(
            XAUTimeframe.M1,
            start=old.hour,
            end=old.hour + timedelta(hours=1),
            max_hours=1,
        )
        result["old_hour"] = {
            "hour": old.hour.isoformat(),
            "tick_count": len(old.ticks),
            "min_mid": min(mids),
            "max_mid": max(mids),
            "m1_bars": len(old_bars),
            "failed_hours": old_diag.failed_hours,
            "all_research_only": all(not bar.execution_eligible for bar in old_bars),
            "source_set": sorted({bar.source for bar in old_bars}),
        }

    if recent is not None:
        mids = [tick.mid for tick in recent.ticks]
        recent_bars, recent_diag = provider.bars_range(
            XAUTimeframe.M1,
            start=recent.hour,
            end=recent.hour + timedelta(hours=1),
            max_hours=1,
        )
        result["modern_hour"] = {
            "hour": recent.hour.isoformat(),
            "tick_count": len(recent.ticks),
            "min_mid": min(mids),
            "max_mid": max(mids),
            "m1_bars": len(recent_bars),
            "failed_hours": recent_diag.failed_hours,
            "all_research_only": all(not bar.execution_eligible for bar in recent_bars),
            "source_set": sorted({bar.source for bar in recent_bars}),
        }

    if old is not None and recent is not None:
        gap_days = (recent.hour - old.hour).total_seconds() / 86400.0
        result["deep_history_gap_days"] = gap_days

        old_info = result["old_hour"]
        recent_info = result["modern_hour"]
        sane_old = (
            old_info["tick_count"] > 0
            and old_info["m1_bars"] > 0
            and 100.0 < old_info["min_mid"] < 10_000.0
            and old_info["max_mid"] < 10_000.0
        )
        sane_recent = (
            recent_info["tick_count"] > 0
            and recent_info["m1_bars"] > 0
            and 100.0 < recent_info["min_mid"] < 10_000.0
            and recent_info["max_mid"] < 10_000.0
        )
        provenance_ok = (
            old_info["source_set"] == [provider.source]
            and recent_info["source_set"] == [provider.source]
            and old_info["all_research_only"]
            and recent_info["all_research_only"]
        )
        result["passed"] = bool(
            sane_old
            and sane_recent
            and provenance_ok
            and gap_days > 365 * 5
            and provider.centralized_order_flow is False
            and provider.execution_eligible is False
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
