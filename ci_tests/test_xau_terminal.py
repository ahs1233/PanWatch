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

    async def fake_spot(force: bool = False):
        return {
            "price": 2620.0,
            "bid": 2619.8,
            "ask": 2620.2,
            "spread": 0.4,
            "spread_bps": 1.53,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "age_seconds": 1.0,
            "source": "test-spot",
            "is_stale": False,
            "indicative": True,
            "execution_eligible": False,
        }

    monkeypatch.setattr(service, "get_research_bars", fake_bars)
    monkeypatch.setattr(service, "get_indicative_spot", fake_spot)
    result = asyncio.run(service.get_xau_snapshot())

    assert result["instrument"] == "XAUUSD"
    assert result["research_proxy"] == "GC=F"
    assert result["research_only"] is True
    assert result["execution_feed_connected"] is False
    assert result["execution_status"] == "LOCKED_NO_TRADABLE_SPOT_FEED"
    assert result["indicative_spot"]["source"] == "test-spot"
    assert result["indicative_spot"]["execution_eligible"] is False
    assert set(result["frames"]) == {"1m", "5m", "15m"}


def test_macro_json_parser_accepts_plain_object():
    parsed = service._parse_json(
        '{"bias":1,"confidence":0.7,"event_risk":false,"summary":"x","drivers":[]}'
    )
    assert parsed["bias"] == 1
    assert parsed["confidence"] == 0.7
