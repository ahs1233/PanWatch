"""Free XAUT/USD microstructure feed from Bitfinex public APIs.

This module is intentionally research-only for XAUUSD. XAUT/USD is a centralized
exchange market for tokenized gold and provides real executed trade direction
and a raw per-order book. It is useful as a gold microstructure sensor, but it
is NOT the execution venue for OTC XAUUSD and must never silently become the
XAUUSD execution quote.

Bitfinex public trade semantics:
- positive AMOUNT => taker buy
- negative AMOUNT => taker sell

Bitfinex raw book (R0) semantics:
- positive AMOUNT => bid order
- negative AMOUNT => ask order
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable

import httpx


@dataclass(frozen=True)
class XAUTTrade:
    trade_id: int
    timestamp: datetime
    amount: float
    price: float

    @property
    def size(self) -> float:
        return abs(self.amount)

    @property
    def aggressor_side(self) -> str:
        return "buy" if self.amount > 0 else "sell"


@dataclass(frozen=True)
class XAUTBookOrder:
    order_id: int
    price: float
    amount: float

    @property
    def size(self) -> float:
        return abs(self.amount)

    @property
    def side(self) -> str:
        return "bid" if self.amount > 0 else "ask"


@dataclass(frozen=True)
class XAUTMicrostructureSnapshot:
    bid: float
    ask: float
    last: float
    observed_at: datetime
    trades: tuple[XAUTTrade, ...]
    raw_book: tuple[XAUTBookOrder, ...]
    source: str = "bitfinex:XAUTUSD"
    symbol: str = "XAUTUSD"
    reference_symbol: str = "XAUUSD"
    execution_eligible: bool = False
    centralized_proxy_market: bool = True

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def _utc_from_ms(value: Any) -> datetime:
    raw = float(value)
    return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)


def parse_trade(row: Any) -> XAUTTrade:
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        raise ValueError("invalid Bitfinex public trade row")
    trade_id, mts, amount, price = row[:4]
    amount_f = float(amount)
    price_f = float(price)
    if amount_f == 0 or price_f <= 0:
        raise ValueError("invalid Bitfinex trade amount/price")
    return XAUTTrade(
        trade_id=int(trade_id),
        timestamp=_utc_from_ms(mts),
        amount=amount_f,
        price=price_f,
    )


def parse_raw_book_order(row: Any) -> XAUTBookOrder:
    if not isinstance(row, (list, tuple)) or len(row) < 3:
        raise ValueError("invalid Bitfinex raw-book row")
    order_id, price, amount = row[:3]
    price_f = float(price)
    amount_f = float(amount)
    if price_f <= 0 or amount_f == 0:
        raise ValueError("invalid live raw-book order")
    return XAUTBookOrder(order_id=int(order_id), price=price_f, amount=amount_f)


class BitfinexXAUTPublicProvider:
    """Keyless REST snapshot provider for XAUT/USD.

    REST is used for bootstrap/recovery. The websocket stream below should be
    preferred for continuous scalping-state updates so the trades endpoint rate
    limit is not abused.
    """

    base_url = "https://api-pub.bitfinex.com/v2"
    symbol = "tXAUT:USD"

    def fetch_snapshot(
        self,
        *,
        trade_limit: int = 1000,
        book_len: int = 100,
        timeout_seconds: float = 8.0,
    ) -> XAUTMicrostructureSnapshot:
        trade_limit = max(25, min(int(trade_limit), 10_000))
        book_len = 100 if int(book_len) >= 100 else 25
        headers = {"User-Agent": "PanWatch-XAU/0.2"}
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            ticker_res = client.get(f"{self.base_url}/ticker/{self.symbol}")
            trades_res = client.get(
                f"{self.base_url}/trades/{self.symbol}/hist",
                params={"limit": trade_limit, "sort": -1},
            )
            book_res = client.get(
                f"{self.base_url}/book/{self.symbol}/R0",
                params={"len": book_len},
            )
            ticker_res.raise_for_status()
            trades_res.raise_for_status()
            book_res.raise_for_status()

        ticker = ticker_res.json()
        if not isinstance(ticker, list) or len(ticker) < 7:
            raise RuntimeError("Bitfinex XAUT ticker returned malformed response")
        bid, ask, last = float(ticker[0]), float(ticker[2]), float(ticker[6])
        if min(bid, ask, last) <= 0 or ask < bid:
            raise RuntimeError("Bitfinex XAUT ticker returned invalid prices")

        trades: list[XAUTTrade] = []
        for row in trades_res.json() if isinstance(trades_res.json(), list) else []:
            try:
                trades.append(parse_trade(row))
            except (TypeError, ValueError):
                continue
        trades.sort(key=lambda item: item.timestamp)

        raw_book: list[XAUTBookOrder] = []
        for row in book_res.json() if isinstance(book_res.json(), list) else []:
            try:
                raw_book.append(parse_raw_book_order(row))
            except (TypeError, ValueError):
                continue

        if not trades:
            raise RuntimeError("Bitfinex XAUT returned no usable public trades")
        if not raw_book:
            raise RuntimeError("Bitfinex XAUT returned no usable raw-book orders")

        return XAUTMicrostructureSnapshot(
            bid=bid,
            ask=ask,
            last=last,
            observed_at=datetime.now(timezone.utc),
            trades=tuple(trades),
            raw_book=tuple(raw_book),
        )


class BitfinexXAUTStream:
    """Live public WebSocket stream for XAUT/USD trades + raw R0 book.

    The stream maintains a local raw book keyed by Bitfinex ORDER_ID and emits
    immutable snapshots after each meaningful event. Trade updates use only
    `te` (trade executed) messages; `tu` is ignored to avoid duplicate CVD.
    """

    url = "wss://api-pub.bitfinex.com/ws/2"
    symbol = "tXAUT:USD"

    def __init__(self, *, book_len: int = 100) -> None:
        self.book_len = 100 if int(book_len) >= 100 else 25
        self._channels: dict[int, str] = {}
        self._book: dict[int, XAUTBookOrder] = {}
        self._trades: list[XAUTTrade] = []
        self._last: float | None = None
        self._bid: float | None = None
        self._ask: float | None = None

    def apply_book_payload(self, payload: Any) -> None:
        rows: Iterable[Any]
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            rows = payload
        else:
            rows = [payload]
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            order_id, price, amount = row[:3]
            try:
                oid = int(order_id)
                price_f = float(price)
                amount_f = float(amount)
            except (TypeError, ValueError):
                continue
            if price_f == 0:
                self._book.pop(oid, None)
                continue
            if amount_f == 0 or price_f < 0:
                continue
            self._book[oid] = XAUTBookOrder(oid, price_f, amount_f)
        bids = [x.price for x in self._book.values() if x.amount > 0]
        asks = [x.price for x in self._book.values() if x.amount < 0]
        self._bid = max(bids) if bids else self._bid
        self._ask = min(asks) if asks else self._ask

    def apply_trade_payload(self, payload: Any) -> XAUTTrade | None:
        try:
            trade = parse_trade(payload)
        except (TypeError, ValueError):
            return None
        if self._trades and self._trades[-1].trade_id == trade.trade_id:
            return None
        self._trades.append(trade)
        if len(self._trades) > 10_000:
            del self._trades[:-10_000]
        self._last = trade.price
        return trade

    def snapshot(self) -> XAUTMicrostructureSnapshot | None:
        if self._bid is None or self._ask is None or self._last is None:
            return None
        if not self._trades or not self._book:
            return None
        return XAUTMicrostructureSnapshot(
            bid=self._bid,
            ask=self._ask,
            last=self._last,
            observed_at=datetime.now(timezone.utc),
            trades=tuple(self._trades),
            raw_book=tuple(self._book.values()),
            source="bitfinex:XAUTUSD:websocket",
        )

    async def snapshots(self) -> AsyncIterator[XAUTMicrostructureSnapshot]:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("websockets package is required for live XAUT stream") from exc

        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_queue=4096,
        ) as ws:
            await ws.send(json.dumps({
                "event": "subscribe",
                "channel": "trades",
                "symbol": self.symbol,
            }))
            await ws.send(json.dumps({
                "event": "subscribe",
                "channel": "book",
                "symbol": self.symbol,
                "prec": "R0",
                "freq": "F0",
                "len": str(self.book_len),
            }))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(msg, dict):
                    if msg.get("event") == "subscribed" and msg.get("chanId") is not None:
                        self._channels[int(msg["chanId"])] = str(msg.get("channel") or "")
                    continue
                if not isinstance(msg, list) or len(msg) < 2:
                    continue
                if msg[1] == "hb":
                    continue
                chan = self._channels.get(int(msg[0]), "")
                changed = False
                if chan == "book":
                    self.apply_book_payload(msg[1])
                    changed = True
                elif chan == "trades":
                    if isinstance(msg[1], list):
                        # Initial snapshot.
                        for row in msg[1]:
                            changed = bool(self.apply_trade_payload(row)) or changed
                    elif msg[1] == "te" and len(msg) >= 3:
                        changed = self.apply_trade_payload(msg[2]) is not None
                    # Ignore `tu`: same execution, later canonical update.
                if changed:
                    snap = self.snapshot()
                    if snap is not None:
                        yield snap


async def first_live_snapshot(timeout_seconds: float = 12.0) -> XAUTMicrostructureSnapshot:
    stream = BitfinexXAUTStream()
    async def _next() -> XAUTMicrostructureSnapshot:
        async for snapshot in stream.snapshots():
            return snapshot
        raise RuntimeError("Bitfinex XAUT stream closed before first snapshot")
    return await asyncio.wait_for(_next(), timeout=timeout_seconds)
