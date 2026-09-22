"""Fast cached runtime for the free XAUT microstructure sensor.

WebSocket is primary. REST is bootstrap/fallback only and is throttled so the
public trades endpoint stays below its documented request ceiling.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.modules.xau.xaut_order_flow import analyze_xaut_microstructure
from src.platform.marketdata.xaut_bitfinex import (
    BitfinexXAUTPublicProvider,
    BitfinexXAUTStream,
    XAUTMicrostructureSnapshot,
)

logger = logging.getLogger(__name__)

_LIVE_STALE_SECONDS = 5.0
_REST_TTL_SECONDS = 5.2  # < 12 calls/min; trades endpoint allows 15 req/min.
_RECONNECT_DELAY_SECONDS = 2.0

_live_snapshot: XAUTMicrostructureSnapshot | None = None
_stream_task: asyncio.Task | None = None
_rest_cache: tuple[float, XAUTMicrostructureSnapshot] | None = None
_rest_lock = asyncio.Lock()
_task_lock = asyncio.Lock()


def _age_seconds(snapshot: XAUTMicrostructureSnapshot) -> float:
    return max(0.0, (datetime.now(timezone.utc) - snapshot.observed_at).total_seconds())


async def _stream_forever() -> None:
    global _live_snapshot
    while True:
        stream = BitfinexXAUTStream(book_len=100)
        try:
            async for snapshot in stream.snapshots():
                _live_snapshot = snapshot
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - self-healing sensor loop
            logger.warning("XAUT Bitfinex websocket failed: %s", type(exc).__name__)
            await asyncio.sleep(_RECONNECT_DELAY_SECONDS)


async def _ensure_stream_started() -> None:
    global _stream_task
    if _stream_task is not None and not _stream_task.done():
        return
    async with _task_lock:
        if _stream_task is None or _stream_task.done():
            _stream_task = asyncio.create_task(
                _stream_forever(),
                name="xaut-bitfinex-public-stream",
            )


async def _rest_snapshot(force: bool = False) -> XAUTMicrostructureSnapshot:
    global _rest_cache
    now = time.monotonic()
    if not force and _rest_cache and now - _rest_cache[0] < _REST_TTL_SECONDS:
        return _rest_cache[1]
    async with _rest_lock:
        now = time.monotonic()
        if not force and _rest_cache and now - _rest_cache[0] < _REST_TTL_SECONDS:
            return _rest_cache[1]
        provider = BitfinexXAUTPublicProvider()
        snapshot = await asyncio.to_thread(
            provider.fetch_snapshot,
            trade_limit=1000,
            book_len=100,
            timeout_seconds=8.0,
        )
        _rest_cache = (time.monotonic(), snapshot)
        return snapshot


def _attach_basis(data: dict[str, Any], xau_spot_price: float | None) -> dict[str, Any]:
    if xau_spot_price is None or xau_spot_price <= 0:
        return data
    out = dict(data)
    quote = dict(out.get("quote") or {})
    xaut_mid = float(quote.get("mid") or 0.0)
    if xaut_mid <= 0:
        return out
    basis = float(xau_spot_price) - xaut_mid
    out["basis"] = {
        "available": True,
        "xau_minus_xaut": round(basis, 6),
        "basis_bps": round((basis / float(xau_spot_price)) * 10_000.0, 4),
        "mapping": "instantaneous_offset_only",
    }
    out["forward_range_map"] = {
        f"pm{distance}": {
            "xau_up": round(float(xau_spot_price) + distance, 4),
            "xau_down": round(float(xau_spot_price) - distance, 4),
            "xaut_equivalent_up": round(xaut_mid + distance, 4),
            "xaut_equivalent_down": round(xaut_mid - distance, 4),
        }
        for distance in (10, 20, 30)
    }
    return out


async def get_xaut_order_flow(
    *,
    xau_spot_price: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return current free gold microstructure context for Gen1.

    Live WebSocket data is preferred. REST is used only when the stream hasn't
    produced a recent complete snapshot yet.
    """
    await _ensure_stream_started()
    snapshot = _live_snapshot
    transport = "websocket"
    if snapshot is None or _age_seconds(snapshot) > _LIVE_STALE_SECONDS:
        snapshot = await _rest_snapshot(force=force)
        transport = "rest_bootstrap"
    data = analyze_xaut_microstructure(snapshot, xau_spot_price=xau_spot_price)
    data["transport"] = transport
    data["snapshot_age_seconds"] = round(_age_seconds(snapshot), 3)
    return _attach_basis(data, xau_spot_price)


async def stop_xaut_stream() -> None:
    global _stream_task
    task = _stream_task
    _stream_task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
