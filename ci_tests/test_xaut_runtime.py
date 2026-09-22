from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.modules.xau import xaut_runtime
from src.platform.marketdata.xaut_bitfinex import XAUTBookOrder, XAUTMicrostructureSnapshot, XAUTTrade


NOW = datetime.now(timezone.utc)


def _snapshot() -> XAUTMicrostructureSnapshot:
    trades = (
        XAUTTrade(1, NOW - timedelta(seconds=20), 0.3, 4360.0),
        XAUTTrade(2, NOW - timedelta(seconds=5), -0.2, 4360.5),
    )
    book = (
        XAUTBookOrder(1, 4359.5, 5.0),
        XAUTBookOrder(2, 4361.0, -2.0),
    )
    return XAUTMicrostructureSnapshot(
        bid=4360.0,
        ask=4361.0,
        last=4360.5,
        observed_at=NOW,
        trades=trades,
        raw_book=book,
    )


def test_attach_basis_generates_forward_map():
    base = {"quote": {"mid": 4360.5}, "basis": {"available": False}, "forward_range_map": None}
    out = xaut_runtime._attach_basis(base, 4363.5)
    assert out["basis"]["xau_minus_xaut"] == 3.0
    assert out["forward_range_map"]["pm10"]["xau_up"] == 4373.5
    assert out["forward_range_map"]["pm30"]["xau_down"] == 4333.5


def test_live_snapshot_is_preferred_without_rest(monkeypatch):
    snap = _snapshot()
    monkeypatch.setattr(xaut_runtime, "_live_snapshot", snap)

    async def no_start():
        return None

    async def should_not_run(force=False):
        raise AssertionError("REST fallback should not run for fresh live data")

    monkeypatch.setattr(xaut_runtime, "_ensure_stream_started", no_start)
    monkeypatch.setattr(xaut_runtime, "_rest_snapshot", should_not_run)
    result = asyncio.run(xaut_runtime.get_xaut_order_flow(xau_spot_price=4363.5))
    assert result["transport"] == "websocket"
    assert result["basis"]["xau_minus_xaut"] == 3.0
