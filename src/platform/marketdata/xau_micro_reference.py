"""Live indicative XAU spot micro-series from XAUS.

XAUS records its upstream live gold quote every roughly two minutes and exposes
up to 48 hours without authentication. This series is useful for near-term
momentum confirmation, but remains indicative and is never execution eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


@dataclass(frozen=True)
class XAUMicroPoint:
    timestamp: datetime
    price: float


@dataclass(frozen=True)
class XAUMicroSeries:
    points: tuple[XAUMicroPoint, ...]
    observed_at: datetime
    source: str
    age_seconds: float | None
    coverage_seconds: float | None
    is_stale: bool
    execution_eligible: bool = False
    indicative: bool = True


def _timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)

    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    try:
        raw = float(text)
        if raw > 10_000_000_000:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    except ValueError:
        pass

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class XAUSIntradayReferenceProvider:
    url = "https://xaus.com/api/v1/intraday"

    def fetch(self, *, hours: int = 2, timeout_seconds: float = 12.0) -> XAUMicroSeries:
        hours = max(1, min(int(hours), 48))
        response = httpx.get(
            self.url,
            params={"symbol": "xau", "hours": hours},
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("xaus.com intraday returned malformed response")

        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise RuntimeError("xaus.com intraday returned no points")

        points: list[XAUMicroPoint] = []
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            price = _positive(item.get("p"))
            if price is None:
                continue
            points.append(
                XAUMicroPoint(
                    timestamp=_timestamp(item.get("t")),
                    price=price,
                )
            )
        points.sort(key=lambda item: item.timestamp)
        if not points:
            raise RuntimeError("xaus.com intraday returned no usable XAU points")

        state = payload.get("data_state")
        state = state if isinstance(state, dict) else {}
        status = str(state.get("status") or "").strip().lower()

        age_seconds = _positive(state.get("age_seconds"))
        if age_seconds is None:
            age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - points[-1].timestamp).total_seconds(),
            )

        coverage_seconds = _positive(
            payload.get("coverage_seconds") or state.get("coverage_seconds")
        )
        observed_at = _timestamp(state.get("as_of") or points[-1].timestamp.isoformat())

        return XAUMicroSeries(
            points=tuple(points),
            observed_at=observed_at,
            source="xaus.com:intraday",
            age_seconds=age_seconds,
            coverage_seconds=coverage_seconds,
            is_stale=status not in {"", "fresh"} or age_seconds > 240,
        )


def sampled_spot_bars(
    series: XAUMicroSeries,
    timeframe: XAUTimeframe,
) -> list[XAUBar]:
    """Aggregate the roughly-two-minute indicative tape into sampled OHLC bars.

    High/low reflect observed samples, not every market tick. Bars remain
    research-only and execution_eligible=False.
    """

    interval_minutes = {
        XAUTimeframe.M5: 5,
        XAUTimeframe.M15: 15,
    }.get(timeframe)
    if interval_minutes is None:
        raise ValueError("sampled spot bars support only 5m and 15m")

    buckets: dict[int, list[XAUMicroPoint]] = {}
    seconds = interval_minutes * 60
    for point in series.points:
        bucket = int(point.timestamp.timestamp()) // seconds
        buckets.setdefault(bucket, []).append(point)

    bars: list[XAUBar] = []
    for bucket in sorted(buckets):
        points = sorted(buckets[bucket], key=lambda item: item.timestamp)
        if not points:
            continue
        prices = [item.price for item in points]
        bars.append(
            XAUBar(
                timestamp=points[-1].timestamp,
                timeframe=timeframe,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=None,
                source="xaus.com:intraday-sampled",
                symbol="XAUUSD",
                execution_eligible=False,
            )
        )
    return bars
