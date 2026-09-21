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


class BiquoteXAUIndicativeSpotReference:
    """Public MT5-backed indicative XAUUSD bid/ask reference.

    This feed is suitable for paper-fill friction and quote-side simulation only.
    It is never execution eligible because it is not tied to the user's broker.
    """

    url = "https://biquote.io/api/XAUUSD"

    def fetch(self, timeout_seconds: float = 10.0) -> XAUIndicativeSpot:
        response = httpx.get(
            self.url,
            params={"allowStale": "false"},
            timeout=timeout_seconds,
            headers={"User-Agent": "PanWatch-XAU/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("biquote.io returned malformed XAUUSD response")

        bid = _number(payload.get("bid"))
        ask = _number(payload.get("ask"))
        if bid is None or ask is None:
            raise RuntimeError("biquote.io returned no usable bid/ask")
        if ask < bid:
            raise RuntimeError("biquote.io returned crossed XAUUSD quote")

        price = _number(payload.get("mid")) or ((bid + ask) / 2.0)
        observed_at = _parse_timestamp(
            payload.get("timestamp") or payload.get("lastQuoteAt")
        )

        try:
            quote_age_seconds = float(payload.get("quoteAgeSeconds") or 0.0)
        except (TypeError, ValueError):
            quote_age_seconds = 0.0

        market_state = str(payload.get("marketState") or "").strip().lower()
        stale = bool(payload.get("stale", False))
        is_stale = stale or quote_age_seconds > 300.0
        if market_state and market_state != "open":
            is_stale = True

        raw_source = str(payload.get("source") or "MT5").strip()
        source = f"biquote.io:{raw_source}" if raw_source else "biquote.io:MT5"

        return XAUIndicativeSpot(
            price=price,
            bid=bid,
            ask=ask,
            observed_at=observed_at,
            source=source,
            is_stale=is_stale,
        )


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
    """Prefer a fresh bid/ask quote, then fail soft to fresh mid references."""

    def __init__(self) -> None:
        self.providers = (
            BiquoteXAUIndicativeSpotReference(),
            GoldPriceDevSpotReference(),
            XAUSSpotReference(),
        )

    def fetch(self, timeout_seconds: float = 10.0) -> XAUIndicativeSpot:
        errors: list[str] = []
        fallback: XAUIndicativeSpot | None = None

        for provider in self.providers:
            try:
                quote = provider.fetch(timeout_seconds=timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - fail over to next reference
                errors.append(f"{type(provider).__name__}:{type(exc).__name__}")
                continue

            if not quote.is_stale and quote.bid is not None and quote.ask is not None:
                return quote

            if fallback is None:
                fallback = quote
            elif fallback.is_stale and not quote.is_stale:
                fallback = quote

        if fallback is not None:
            return fallback
        raise RuntimeError("all XAU spot references failed: " + ",".join(errors))
