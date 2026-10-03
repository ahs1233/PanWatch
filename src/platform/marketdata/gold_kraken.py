"""Public Kraken PAXG/USD gold microstructure feed for PanWatch research.

PAXG/USD is a centralized tokenized-gold proxy market. It MUST NOT be
represented as global OTC XAU/USD order flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass(frozen=True)
class KrakenGoldTrade:
    trade_id: str
    timestamp: datetime
    price: float
    size_xau: float
    aggressor_side: str

    @property
    def signed_size_xau(self) -> float:
        return self.size_xau if self.aggressor_side == "buy" else -self.size_xau


@dataclass(frozen=True)
class KrakenBookLevel:
    price: float
    size_xau: float
    side: str
    order_count: int = 0


@dataclass(frozen=True)
class KrakenGoldSnapshot:
    instrument_id: str
    market_kind: str
    bid: float
    ask: float
    last: float
    observed_at: datetime
    trades: tuple[KrakenGoldTrade, ...]
    book: tuple[KrakenBookLevel, ...]
    open_interest_xau: float | None = None
    open_interest_usd: float | None = None
    volume_24h_xau: float | None = None
    source: str = "kraken:PAXGUSD"
    execution_eligible: bool = False
    centralized_proxy_market: bool = True

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class KrakenGoldPublicProvider:
    base_url = "https://api.kraken.com/0/public"

    def __init__(self, instrument_id: str = "PAXGUSD") -> None:
        self.instrument_id = instrument_id
        self.market_kind = "tokenized_spot"

    def _get(self, client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = client.get(f"{self.base_url}{path}", params=params or {})
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("error") or []
        if errors:
            raise RuntimeError(f"Kraken error: {errors}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Kraken returned invalid result")
        return result

    @staticmethod
    def _first_market(result: dict[str, Any], *, exclude_last: bool = False) -> Any:
        for key, value in result.items():
            if exclude_last and key == "last":
                continue
            return value
        raise RuntimeError("Kraken returned no market payload")

    @staticmethod
    def _parse_trade(row: Any) -> KrakenGoldTrade:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            raise ValueError("invalid Kraken trade row")
        price = float(row[0])
        qty = float(row[1])
        ts = float(row[2])
        side_raw = str(row[3]).lower()
        if side_raw not in {"b", "s"} or price <= 0 or qty <= 0:
            raise ValueError("invalid Kraken trade")
        trade_id = str(row[6]) if len(row) > 6 else f"{int(ts*1_000_000)}-{price}-{qty}"
        return KrakenGoldTrade(
            trade_id=trade_id,
            timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
            price=price,
            size_xau=qty,
            aggressor_side="buy" if side_raw == "b" else "sell",
        )

    @staticmethod
    def _parse_book_side(rows: Any, side: str) -> list[KrakenBookLevel]:
        out: list[KrakenBookLevel] = []
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
                out.append(KrakenBookLevel(price, qty, side, 0))
        return out

    def fetch_snapshot(
        self,
        *,
        trade_limit: int = 500,
        book_depth: int = 400,
        timeout_seconds: float = 8.0,
    ) -> KrakenGoldSnapshot:
        trade_limit = max(1, min(int(trade_limit), 1000))
        book_depth = max(1, min(int(book_depth), 500))
        headers = {"User-Agent": "PanWatch-XAU/0.5"}
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            ticker_res = self._get(client, "/Ticker", {"pair": self.instrument_id})
            trades_res = self._get(client, "/Trades", {"pair": self.instrument_id, "count": trade_limit})
            depth_res = self._get(client, "/Depth", {"pair": self.instrument_id, "count": book_depth})

        ticker = self._first_market(ticker_res)
        trades_raw = self._first_market(trades_res, exclude_last=True)
        depth = self._first_market(depth_res)

        ask = float(ticker["a"][0])
        bid = float(ticker["b"][0])
        last = float(ticker["c"][0])
        if min(bid, ask, last) <= 0 or ask < bid:
            raise RuntimeError("Kraken ticker returned invalid prices")

        trades: list[KrakenGoldTrade] = []
        for row in trades_raw if isinstance(trades_raw, list) else []:
            try:
                trades.append(self._parse_trade(row))
            except (TypeError, ValueError):
                continue
        trades.sort(key=lambda t: t.timestamp)
        if not trades:
            raise RuntimeError("Kraken returned no usable trades")

        book = self._parse_book_side(depth.get("bids"), "bid")
        book.extend(self._parse_book_side(depth.get("asks"), "ask"))
        if not book:
            raise RuntimeError("Kraken returned no usable order book")

        vol24 = None
        try:
            vol24 = float(ticker["v"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            pass

        return KrakenGoldSnapshot(
            instrument_id=self.instrument_id,
            market_kind=self.market_kind,
            bid=bid,
            ask=ask,
            last=last,
            observed_at=datetime.now(timezone.utc),
            trades=tuple(trades),
            book=tuple(book),
            volume_24h_xau=vol24,
        )

    def fetch_history_trades(
        self,
        *,
        max_pages: int = 10,
        page_size: int = 1000,
        timeout_seconds: float = 8.0,
    ) -> list[KrakenGoldTrade]:
        # Kraken public Trades returns the recent tape plus a cursor. For a bounded
        # warm-start we use the largest recent snapshot rather than claiming
        # complete historical paging backwards.
        page_size = max(1, min(int(page_size), 1000))
        headers = {"User-Agent": "PanWatch-XAU/0.5"}
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            result = self._get(client, "/Trades", {"pair": self.instrument_id, "count": page_size})
        raw = self._first_market(result, exclude_last=True)
        out: list[KrakenGoldTrade] = []
        for row in raw if isinstance(raw, list) else []:
            try:
                out.append(self._parse_trade(row))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda t: t.timestamp)
        return out
