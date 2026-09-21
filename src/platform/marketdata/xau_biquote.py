"""Biquote XAUUSD market-data adapters.

Biquote provides public MT5-backed XAUUSD ticks, historical OHLC bars and an
economic calendar. All data remains research/paper-only inside PanWatch and is
never marked execution eligible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _utc_param(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class BiquoteXAUOHLCProvider:
    """MT5-backed OHLC provider for XAUUSD research bars."""

    url = "https://biquote.io/api/XAUUSD/ohlc"

    def bars(
        self,
        timeframe: XAUTimeframe,
        *,
        limit: int = 240,
        timeout_seconds: float = 12.0,
    ) -> list[XAUBar]:
        interval = {
            XAUTimeframe.M1: "1m",
            XAUTimeframe.M5: "5m",
            XAUTimeframe.M15: "15m",
        }.get(timeframe)
        if interval is None:
            raise ValueError(f"unsupported Biquote XAU timeframe: {timeframe}")

        response = httpx.get(
            self.url,
            params={
                "interval": interval,
                "limit": max(30, min(int(limit), 1000)),
            },
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.5"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("biquote.io OHLC returned malformed response")
        raw_bars = payload.get("bars")
        if not isinstance(raw_bars, list):
            raise RuntimeError("biquote.io OHLC returned no bars")

        out: list[XAUBar] = []
        for item in raw_bars:
            if not isinstance(item, dict):
                continue
            timestamp = _timestamp(item.get("openTime"))
            open_price = _number(item.get("open"))
            high = _number(item.get("high"))
            low = _number(item.get("low"))
            close = _number(item.get("close"))
            if None in {timestamp, open_price, high, low, close}:
                continue
            try:
                volume_raw = item.get("tickVolume")
                if volume_raw is None:
                    volume_raw = item.get("volume")
                volume = float(volume_raw) if volume_raw is not None else None
            except (TypeError, ValueError):
                volume = None

            try:
                out.append(
                    XAUBar(
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=float(open_price),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=volume,
                        source="biquote.io:MT5-ohlc",
                        symbol="XAUUSD",
                        execution_eligible=False,
                    )
                )
            except ValueError:
                continue

        out.sort(key=lambda bar: bar.timestamp)
        if not out:
            raise RuntimeError("biquote.io OHLC returned no usable XAU bars")
        return out


class BiquoteEconomicCalendarProvider:
    """Deterministic high-impact USD event window from Biquote calendar."""

    url = "https://biquote.io/api/calendar"

    def active_usd_event(
        self,
        *,
        now: datetime | None = None,
        before_minutes: int = 90,
        after_minutes: int = 30,
        timeout_seconds: float = 12.0,
    ) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)

        start = current - timedelta(minutes=max(0, int(after_minutes)))
        end = current + timedelta(minutes=max(0, int(before_minutes)))
        response = httpx.get(
            self.url,
            params={
                "from": _utc_param(start),
                "to": _utc_param(end),
                "countries": "US",
                "importance": "high",
                "limit": 100,
            },
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.5"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("biquote.io calendar returned malformed response")

        candidates: list[tuple[float, dict[str, Any]]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("importance") or "").lower() != "high":
                continue
            if str(item.get("timeMode") or "").lower() != "exact":
                continue
            currency = str(item.get("currency") or "").upper()
            country = str(item.get("countryCode") or "").upper()
            if currency not in {"", "USD"} and country != "US":
                continue
            event_time = _timestamp(item.get("time"))
            if event_time is None:
                continue
            delta_minutes = (event_time - current).total_seconds() / 60.0
            if not (-float(after_minutes) <= delta_minutes <= float(before_minutes)):
                continue
            candidates.append((abs(delta_minutes), {
                "event_risk": True,
                "event_kind": "scheduled",
                "event_name": str(item.get("name") or "High-impact USD event").strip(),
                "event_time_utc": event_time.isoformat(),
                "event_age_minutes": max(0.0, -delta_minutes) if delta_minutes < 0 else None,
                "event_confidence": 1.0,
                "event_validation": "calendar_high_impact_window",
                "event_source_url": item.get("sourceUrl"),
                "event_id": item.get("id"),
            }))

        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]
