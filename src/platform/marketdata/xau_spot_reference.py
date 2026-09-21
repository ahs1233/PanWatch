"""Indicative XAU/USD spot reference feeds.

These feeds improve situational awareness but are deliberately NOT marked as
execution eligible. A broker/exchange quote tied to the eventual execution
venue is still required before live order routing can be enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass(frozen=True)
class XAUIndicativeSpot:
    price: float
    bid: float | None
    ask: float | None
    observed_at: datetime
    source: str
    is_stale: bool
    execution_eligible: bool = False
    indicative: bool = True

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return max(0.0, self.ask - self.bid)

    @property
    def spread_bps(self) -> float | None:
        if self.spread is None or self.price <= 0:
            return None
        return (self.spread / self.price) * 10_000.0


def _parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class GoldPriceDevSpotReference:
    """Keyless spot reference with bid/ask and explicit stale metadata."""

    url = "https://api.goldprice.dev/v1/prices?symbol=XAU-USD-SPOT"

    def fetch(self, timeout_seconds: float = 10.0) -> XAUIndicativeSpot:
        response = httpx.get(
            self.url,
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("goldprice.dev returned no XAU spot row")
        row = rows[0]
        if not isinstance(row, dict):
            raise RuntimeError("goldprice.dev returned malformed XAU spot row")

        price = _number(row.get("price"))
        if price is None:
            raise RuntimeError("goldprice.dev returned no spot price")

        return XAUIndicativeSpot(
            price=price,
            bid=_number(row.get("bid")),
            ask=_number(row.get("ask")),
            observed_at=_parse_timestamp(row.get("computed_at")),
            source="goldprice.dev",
            is_stale=bool(row.get("is_stale", False)),
        )


class XAUSSpotReference:
    """Fallback indicative mid-market spot reference with freshness contract."""

    url = "https://xaus.com/api/v1/spot?compact=1"

    def fetch(self, timeout_seconds: float = 10.0) -> XAUIndicativeSpot:
        response = httpx.get(
            self.url,
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("xaus.com returned malformed response")

        price = _number(payload.get("spot_usd_oz"))
        if price is None and isinstance(payload.get("xau"), dict):
            price = _number(payload["xau"].get("price"))
        if price is None:
            raise RuntimeError("xaus.com returned no spot price")

        state = payload.get("data_state")
        state = state if isinstance(state, dict) else {}
        status = str(state.get("status") or "").lower()
        observed_at = state.get("as_of") or payload.get("updated_at")

        return XAUIndicativeSpot(
            price=price,
            bid=None,
            ask=None,
            observed_at=_parse_timestamp(observed_at),
            source="xaus.com",
            is_stale=status not in {"", "fresh"},
        )


class CompositeXAUIndicativeSpotProvider:
    """Try the bid/ask source first, then fail soft to a mid-market source."""

    def __init__(self) -> None:
        self.providers = (
            GoldPriceDevSpotReference(),
            XAUSSpotReference(),
        )

    def fetch(self, timeout_seconds: float = 10.0) -> XAUIndicativeSpot:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.fetch(timeout_seconds=timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - fail over to next reference
                errors.append(f"{type(provider).__name__}:{type(exc).__name__}")
        raise RuntimeError("all XAU spot references failed: " + ",".join(errors))
