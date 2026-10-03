"""Public Binance USDⓈ-M gold microstructure feed for PanWatch research.

Venue:
- XAUUSDT perpetual

This is a centralized proxy market and MUST NOT be represented as global OTC
XAU/USD order flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass(frozen=True)
class BinanceGoldTrade:
    trade_id: str
    timestamp: datetime
    price: float
    size_xau: float
    aggressor_side: str

    @property
    def signed_size_xau(self) -> float:
        return self.size_xau if self.aggressor_side == "buy" else -self.size_xau


@dataclass(frozen=True)
class BinanceBookLevel:
    price: float
    size_xau: float
    side: str
    order_count: int = 0


@dataclass(frozen=True)
class BinanceGoldSnapshot:
    instrument_id: str
    market_kind: str
    bid: float
    ask: float
    last: float
    observed_at: datetime
    trades: tuple[BinanceGoldTrade, ...]
    book: tuple[BinanceBookLevel, ...]
    open_interest_xau: float | None = None
    open_interest_usd: float | None = None
    volume_24h_xau: float | None = None
    source: str = "binance:XAUUSDT"
    execution_eligible: bool = False
    centralized_proxy_market: bool = True

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class BinanceGoldPublicProvider:
    base_url = "https://fapi.binance.com"

    def __init__(self, instrument_id: str = "XAUUSDT") -> None:
        self.instrument_id = instrument_id
        self.market_kind = "perpetual"

    def _get(self, client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
        response = client.get(f"{self.base_url}{path}", params=params or {})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_trade(row: dict[str, Any]) -> BinanceGoldTrade:
        price = float(row["price"])
        qty = float(row["qty"])
        if price <= 0 or qty <= 0:
            raise ValueError("invalid Binance trade")
        # isBuyerMaker=True means the aggressive taker was the seller.
        side = "sell" if bool(row.get("isBuyerMaker")) else "buy"
        ts = datetime.fromtimestamp(float(row["time"]) / 1000.0, tz=timezone.utc)
        return BinanceGoldTrade(
            trade_id=str(row["id"]),
            timestamp=ts,
            price=price,
            size_xau=qty,
            aggressor_side=side,
        )

    @staticmethod
    def _parse_book_side(rows: Any, side: str) -> list[BinanceBookLevel]:
        out: list[BinanceBookLevel] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                price = float(row[0])
                qty = float(row[1])
            except (TypeError, ValueError):
                continue
            if price > 0 and qty > 0:
                out.append(BinanceBookLevel(price, qty, side, 0))
        return out

    def fetch_snapshot(
        self,
        *,
        trade_limit: int = 500,
        book_depth: int = 500,
        timeout_seconds: float = 8.0,
    ) -> BinanceGoldSnapshot:
        trade_limit = max(1, min(int(trade_limit), 1000))
        allowed_depths = [5, 10, 20, 50, 100, 500, 1000]
        book_depth = min(allowed_depths, key=lambda v: abs(v - max(5, int(book_depth))))
        headers = {"User-Agent": "PanWatch-XAU/0.4"}
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            ticker = self._get(client, "/fapi/v1/ticker/bookTicker", {"symbol": self.instrument_id})
            trades_raw = self._get(client, "/fapi/v1/trades", {"symbol": self.instrument_id, "limit": trade_limit})
            depth = self._get(client, "/fapi/v1/depth", {"symbol": self.instrument_id, "limit": book_depth})
            premium = self._get(client, "/fapi/v1/premiumIndex", {"symbol": self.instrument_id})
            oi = None
            try:
                oi = self._get(client, "/fapi/v1/openInterest", {"symbol": self.instrument_id})
            except Exception:
                oi = None
            ticker24 = None
            try:
                ticker24 = self._get(client, "/fapi/v1/ticker/24hr", {"symbol": self.instrument_id})
            except Exception:
                ticker24 = None

        bid = float(ticker["bidPrice"])
        ask = float(ticker["askPrice"])
        last = float(premium.get("markPrice") or premium.get("indexPrice") or (bid + ask) / 2.0)
        if min(bid, ask, last) <= 0 or ask < bid:
            raise RuntimeError("Binance ticker returned invalid prices")

        trades: list[BinanceGoldTrade] = []
        for row in trades_raw if isinstance(trades_raw, list) else []:
            try:
                trades.append(self._parse_trade(row))
            except (KeyError, TypeError, ValueError):
                continue
        trades.sort(key=lambda t: t.timestamp)
        if not trades:
            raise RuntimeError("Binance returned no usable trades")

        book = self._parse_book_side(depth.get("bids"), "bid")
        book.extend(self._parse_book_side(depth.get("asks"), "ask"))
        if not book:
            raise RuntimeError("Binance returned no usable order book")

        oi_xau = None
        oi_usd = None
        if isinstance(oi, dict):
            try:
                oi_xau = float(oi.get("openInterest"))
                oi_usd = oi_xau * last if oi_xau is not None else None
            except (TypeError, ValueError):
                oi_xau = oi_usd = None

        vol24 = None
        if isinstance(ticker24, dict):
            try:
                vol24 = float(ticker24.get("volume"))
            except (TypeError, ValueError):
                vol24 = None

        return BinanceGoldSnapshot(
            instrument_id=self.instrument_id,
            market_kind=self.market_kind,
            bid=bid,
            ask=ask,
            last=last,
            observed_at=datetime.now(timezone.utc),
            trades=tuple(trades),
            book=tuple(book),
            open_interest_xau=oi_xau,
            open_interest_usd=oi_usd,
            volume_24h_xau=vol24,
        )

    def fetch_history_trades(
        self,
        *,
        max_pages: int = 10,
        page_size: int = 1000,
        timeout_seconds: float = 8.0,
    ) -> list[BinanceGoldTrade]:
        """Bounded public history warm-start using aggregate trades."""
        max_pages = max(1, min(int(max_pages), 100))
        page_size = max(1, min(int(page_size), 1000))
        headers = {"User-Agent": "PanWatch-XAU/0.4"}
        rows: list[BinanceGoldTrade] = []
        seen: set[str] = set()
        end_time: int | None = None
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            for _ in range(max_pages):
                params: dict[str, Any] = {"symbol": self.instrument_id, "limit": page_size}
                if end_time is not None:
                    params["endTime"] = end_time
                raw = self._get(client, "/fapi/v1/aggTrades", params)
                if not isinstance(raw, list) or not raw:
                    break
                page: list[BinanceGoldTrade] = []
                oldest_ms = None
                for item in raw:
                    try:
                        trade_id = str(item["a"])
                        price = float(item["p"])
                        qty = float(item["q"])
                        ts_ms = int(item["T"])
                        side = "sell" if bool(item.get("m")) else "buy"
                    except (KeyError, TypeError, ValueError):
                        continue
                    if trade_id in seen or price <= 0 or qty <= 0:
                        continue
                    seen.add(trade_id)
                    page.append(BinanceGoldTrade(
                        trade_id=trade_id,
                        timestamp=datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                        price=price,
                        size_xau=qty,
                        aggressor_side=side,
                    ))
                    oldest_ms = ts_ms if oldest_ms is None else min(oldest_ms, ts_ms)
                if not page or oldest_ms is None:
                    break
                rows.extend(page)
                end_time = oldest_ms - 1
                if len(raw) < page_size:
                    break
        rows.sort(key=lambda t: t.timestamp)
        return rows
