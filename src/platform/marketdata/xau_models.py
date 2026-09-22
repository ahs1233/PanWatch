"""Provider-neutral XAUUSD models.

XAUUSD spot and GC=F are intentionally represented as different identities.
GC=F may be used as a research proxy; it must never silently become the
execution quote for XAUUSD.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class XAUTimeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1mo"


@dataclass(frozen=True)
class XAUInstrument:
    symbol: str = "XAUUSD"
    name: str = "Gold / U.S. Dollar"
    asset_class: str = "commodity"
    quote_currency: str = "USD"
    research_symbol: str = "GC=F"


XAUUSD = XAUInstrument()


@dataclass(frozen=True)
class XAUQuote:
    bid: float
    ask: float
    observed_at: datetime
    source: str
    execution_eligible: bool = True
    symbol: str = "XAUUSD"

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("XAU quote prices must be positive")
        if self.ask < self.bid:
            raise ValueError("XAU quote ask cannot be below bid")
        if self.observed_at.tzinfo is None:
            object.__setattr__(self, "observed_at", self.observed_at.replace(tzinfo=timezone.utc))

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid) * 10_000 if self.mid else 0.0


@dataclass(frozen=True)
class XAUBar:
    timestamp: datetime
    timeframe: XAUTimeframe
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    source: str = ""
    symbol: str = "XAUUSD"
    execution_eligible: bool = True

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("XAU OHLC values must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("XAU bar high is inconsistent")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("XAU bar low is inconsistent")
