"""Free public OKX gold microstructure feeds for PanWatch research.

The feeds are centralized venues and MUST NOT be represented as global OTC
XAU/USD order flow.  They are research sensors only.

Primary venue:
- XAU-USDT-SWAP (linear perpetual; ctVal=0.001 XAU per contract)

Secondary venue:
- XAUT-USDT spot (tokenized gold)

Public OKX trade rows include aggressor side (buy/sell), which lets PanWatch
build real venue-level delta/CVD/footprint from executed trades rather than
candle direction or MT5 tick volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass(frozen=True)
class OKXGoldTrade:
    trade_id: str
    timestamp: datetime
    price: float
    size_xau: float
    aggressor_side: str

    @property
    def signed_size_xau(self) -> float:
        return self.size_xau if self.aggressor_side == "buy" else -self.size_xau


@dataclass(frozen=True)
class OKXBookLevel:
    price: float
    size_xau: float
    side: str
    order_count: int = 0


@dataclass(frozen=True)
class OKXGoldSnapshot:
    instrument_id: str
    market_kind: str
    bid: float
    ask: float
    last: float
    observed_at: datetime
    trades: tuple[OKXGoldTrade, ...]
    book: tuple[OKXBookLevel, ...]
    open_interest_xau: float | None = None
    open_interest_usd: float | None = None
    volume_24h_xau: float | None = None
    source: str = "okx"
    execution_eligible: bool = False
    centralized_proxy_market: bool = True

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class OKXGoldPublicProvider:
    """Keyless REST provider for OKX centralized gold markets."""

    base_url = "https://www.okx.com/api/v5"

    def __init__(
        self,
        instrument_id: str,
        *,
        market_kind: str,
        contract_xau: float = 1.0,
    ) -> None:
        self.instrument_id = instrument_id
        self.market_kind = market_kind
        self.contract_xau = float(contract_xau)
        if self.contract_xau <= 0:
            raise ValueError("contract_xau must be positive")

    @classmethod
    def xau_swap(cls) -> "OKXGoldPublicProvider":
        return cls("XAU-USDT-SWAP", market_kind="perpetual", contract_xau=0.001)

    @classmethod
    def xaut_spot(cls) -> "OKXGoldPublicProvider":
        return cls("XAUT-USDT", market_kind="tokenized_spot", contract_xau=1.0)

    def _get(self, client: httpx.Client, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX error code={payload.get('code')} msg={payload.get('msg')}")
        return payload

    def _size_to_xau(self, raw_size: Any) -> float:
        return float(raw_size) * self.contract_xau

    def _parse_trade(self, row: dict[str, Any]) -> OKXGoldTrade:
        price = float(row["px"])
        size_xau = self._size_to_xau(row["sz"])
        side = str(row.get("side") or "").lower()
        if side not in {"buy", "sell"}:
            raise ValueError("invalid OKX aggressor side")
        if price <= 0 or size_xau <= 0:
            raise ValueError("invalid OKX trade")
        ts = datetime.fromtimestamp(float(row["ts"]) / 1000.0, tz=timezone.utc)
        return OKXGoldTrade(
            trade_id=str(row.get("tradeId") or row.get("id") or ""),
            timestamp=ts,
            price=price,
            size_xau=size_xau,
            aggressor_side=side,
        )

    def _parse_book_side(self, rows: Any, side: str) -> list[OKXBookLevel]:
        out: list[OKXBookLevel] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                price = float(row[0])
                size_xau = self._size_to_xau(row[1])
                order_count = int(row[3]) if len(row) > 3 else 0
            except (TypeError, ValueError):
                continue
            if price <= 0 or size_xau <= 0:
                continue
            out.append(OKXBookLevel(price, size_xau, side, order_count))
        return out

    def fetch_snapshot(
        self,
        *,
        trade_limit: int = 100,
        book_depth: int = 400,
        timeout_seconds: float = 8.0,
    ) -> OKXGoldSnapshot:
        trade_limit = max(1, min(int(trade_limit), 500))
        book_depth = max(1, min(int(book_depth), 400))
        headers = {"User-Agent": "PanWatch-XAU/0.3"}
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            ticker = self._get(
                client,
                "/market/ticker",
                {"instId": self.instrument_id},
            )
            trades = self._get(
                client,
                "/market/trades",
                {"instId": self.instrument_id, "limit": trade_limit},
            )
            book = self._get(
                client,
                "/market/books",
                {"instId": self.instrument_id, "sz": book_depth},
            )
            oi_payload: dict[str, Any] | None = None
            if self.market_kind == "perpetual":
                try:
                    oi_payload = self._get(
                        client,
                        "/public/open-interest",
                        {"instType": "SWAP", "instId": self.instrument_id},
                    )
                except Exception:
                    oi_payload = None

        ticker_rows = ticker.get("data") or []
        if not ticker_rows:
            raise RuntimeError("OKX ticker returned no data")
        tick = ticker_rows[0]
        bid = float(tick["bidPx"])
        ask = float(tick["askPx"])
        last = float(tick["last"])
        if min(bid, ask, last) <= 0 or ask < bid:
            raise RuntimeError("OKX ticker returned invalid prices")

        parsed_trades: list[OKXGoldTrade] = []
        for row in trades.get("data") or []:
            try:
                parsed_trades.append(self._parse_trade(row))
            except (KeyError, TypeError, ValueError):
                continue
        parsed_trades.sort(key=lambda t: t.timestamp)
        if not parsed_trades:
            raise RuntimeError("OKX returned no usable trades")

        book_rows = book.get("data") or []
        if not book_rows:
            raise RuntimeError("OKX returned no usable order book")
        first_book = book_rows[0]
        parsed_book = self._parse_book_side(first_book.get("bids"), "bid")
        parsed_book.extend(self._parse_book_side(first_book.get("asks"), "ask"))
        if not parsed_book:
            raise RuntimeError("OKX returned no usable book levels")

        oi_xau = None
        oi_usd = None
        if oi_payload and (oi_payload.get("data") or []):
            oi = oi_payload["data"][0]
            try:
                # OKX provides oiCcy in underlying units for this contract.
                oi_xau = float(oi.get("oiCcy"))
            except (TypeError, ValueError):
                try:
                    oi_xau = self._size_to_xau(oi.get("oi"))
                except (TypeError, ValueError):
                    oi_xau = None
            try:
                oi_usd = float(oi.get("oiUsd"))
            except (TypeError, ValueError):
                oi_usd = None

        volume_24h_xau = None
        try:
            if self.market_kind == "perpetual":
                # Derivatives vol24h is contracts; convert by ctVal.
                volume_24h_xau = self._size_to_xau(tick.get("vol24h"))
            else:
                # Spot vol24h is base currency (XAUT); volCcy24h is quote USDT.
                volume_24h_xau = float(tick.get("vol24h"))
        except (TypeError, ValueError):
            volume_24h_xau = None

        return OKXGoldSnapshot(
            instrument_id=self.instrument_id,
            market_kind=self.market_kind,
            bid=bid,
            ask=ask,
            last=last,
            observed_at=datetime.now(timezone.utc),
            trades=tuple(parsed_trades),
            book=tuple(parsed_book),
            open_interest_xau=oi_xau,
            open_interest_usd=oi_usd,
            volume_24h_xau=volume_24h_xau,
            source=f"okx:{self.instrument_id}",
        )

    def fetch_history_trades(
        self,
        *,
        max_pages: int = 10,
        page_size: int = 100,
        timeout_seconds: float = 8.0,
    ) -> list[OKXGoldTrade]:
        """Fetch older public trades for warm-starting the local tape.

        This deliberately has a finite page cap.  PanWatch must not imply a
        complete daily/weekly footprint unless the local collector has complete
        coverage for that window.
        """
        max_pages = max(1, min(int(max_pages), 100))
        page_size = max(1, min(int(page_size), 100))
        headers = {"User-Agent": "PanWatch-XAU/0.3"}
        rows: list[OKXGoldTrade] = []
        seen: set[str] = set()
        after: str | None = None
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            for _ in range(max_pages):
                params: dict[str, Any] = {"instId": self.instrument_id, "limit": page_size}
                if after:
                    params["after"] = after
                payload = self._get(client, "/market/history-trades", params)
                raw_rows = payload.get("data") or []
                if not raw_rows:
                    break
                page: list[OKXGoldTrade] = []
                for raw in raw_rows:
                    try:
                        trade = self._parse_trade(raw)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if trade.trade_id in seen:
                        continue
                    seen.add(trade.trade_id)
                    page.append(trade)
                if not page:
                    break
                rows.extend(page)
                oldest = min(page, key=lambda t: t.timestamp)
                after = oldest.trade_id
                if len(raw_rows) < page_size:
                    break
        rows.sort(key=lambda t: t.timestamp)
        return rows
