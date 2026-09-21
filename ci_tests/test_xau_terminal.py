from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.modules.xau import service
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _bars(timeframe: XAUTimeframe, count: int = 40) -> list[XAUBar]:
    now = datetime.now(timezone.utc)
    minutes = {XAUTimeframe.M1: 1, XAUTimeframe.M5: 5, XAUTimeframe.M15: 15}[timeframe]
    rows = []
    for i in range(count):
        close = 2600.0 + i * 0.4
        rows.append(
            XAUBar(
                timestamp=now - timedelta(minutes=(count - 1 - i) * minutes),
                timeframe=timeframe,
                open=close - 0.2,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=1000 + i,
                source="test",
                execution_eligible=False,
            )
        )
    return rows


def test_snapshot_is_research_only(monkeypatch):
    async def fake_bars(force: bool = False):
        return {
            XAUTimeframe.M1: _bars(XAUTimeframe.M1),
            XAUTimeframe.M5: _bars(XAUTimeframe.M5),
            XAUTimeframe.M15: _bars(XAUTimeframe.M15),
        }

    monkeypatch.setattr(service, "get_research_bars", fake_bars)
    result = asyncio.run(service.get_xau_snapshot())

    assert result["instrument"] == "XAUUSD"
    assert result["research_proxy"] == "GC=F"
    assert result["research_only"] is True
    assert result["execution_feed_connected"] is False
    assert result["execution_status"] == "LOCKED_NO_SPOT_FEED"
    assert set(result["frames"]) == {"1m", "5m", "15m"}


def test_macro_json_parser_accepts_plain_object():
    parsed = service._parse_json(
        '{"bias":1,"confidence":0.7,"event_risk":false,"summary":"x","drivers":[]}'
    )
    assert parsed["bias"] == 1
    assert parsed["confidence"] == 0.7
