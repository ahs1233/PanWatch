from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.modules.xau import service
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata.xau_micro_reference import XAUMicroPoint, XAUMicroSeries, sampled_spot_bars


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


async def _fake_micro(force: bool = False):
    now = datetime.now(timezone.utc)
    return {
        "status": "ready",
        "direction": "bullish",
        "price": 2620.0,
        "ema_fast": 2619.5,
        "ema_slow": 2618.8,
        "return_10m_pct": 0.08,
        "return_30m_pct": 0.16,
        "recent_high": 2620.2,
        "recent_low": 2617.5,
        "point_count": 40,
        "observed_at": now.isoformat(),
        "last_point_at": now.isoformat(),
        "age_seconds": 10.0,
        "coverage_seconds": 7200.0,
        "source": "test-micro",
        "is_stale": False,
        "indicative": True,
        "execution_eligible": False,
    }


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
    monkeypatch.setattr(service, "get_micro_context", _fake_micro)
    result = asyncio.run(service.get_xau_snapshot())

    assert result["instrument"] == "XAUUSD"
    assert result["research_proxy"] == "GC=F"
    assert result["research_only"] is True
    assert result["execution_feed_connected"] is False
    assert result["execution_status"] == "LOCKED_NO_TRADABLE_SPOT_FEED"
    assert result["indicative_spot"]["source"] == "test-spot"
    assert result["indicative_spot"]["execution_eligible"] is False
    assert set(result["frames"]) == {"1m", "5m", "15m"}


def test_live_micro_replaces_only_stale_gc_1m_gate(monkeypatch):
    async def stale_1m_bars(force: bool = False):
        data = {
            XAUTimeframe.M1: _bars(XAUTimeframe.M1),
            XAUTimeframe.M5: _bars(XAUTimeframe.M5),
            XAUTimeframe.M15: _bars(XAUTimeframe.M15),
        }
        stale = []
        for bar in data[XAUTimeframe.M1]:
            stale.append(
                XAUBar(
                    timestamp=bar.timestamp - timedelta(minutes=20),
                    timeframe=bar.timeframe,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    source=bar.source,
                    execution_eligible=False,
                )
            )
        data[XAUTimeframe.M1] = stale
        return data

    async def fake_spot(force: bool = False):
        now = datetime.now(timezone.utc)
        return {
            "price": 2620.0,
            "bid": 2619.8,
            "ask": 2620.2,
            "spread": 0.4,
            "spread_bps": 1.53,
            "observed_at": now.isoformat(),
            "age_seconds": 1.0,
            "source": "test-spot",
            "is_stale": False,
            "indicative": True,
            "execution_eligible": False,
        }

    monkeypatch.setattr(service, "get_research_bars", stale_1m_bars)
    monkeypatch.setattr(service, "get_indicative_spot", fake_spot)
    monkeypatch.setattr(service, "get_micro_context", _fake_micro)

    result = asyncio.run(service.get_xau_snapshot())

    assert result["technical_mode"] == "spot_micro_plus_spot_5m_15m"
    assert result["status"] == "ready_with_spot_micro"
    assert result["blocked"] is False
    assert "stale_1m_bars" in result["raw_proxy_block_reasons"]
    assert "stale_1m_bars" not in result["block_reasons"]
    assert "gc_1m_stale_replaced_by_live_spot_micro" in result["warnings"]
    assert result["micro"]["source"] == "test-micro"


def test_decision_fusion_support_conflict_and_event_gate():
    technical = {
        "candidate": "long_setup",
        "blocked": False,
        "status": "ready_with_spot_micro",
        "technical_mode": "spot_micro_plus_spot_5m_15m",
        "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
        "block_reasons": [],
    }

    support = service.build_decision_fusion(
        technical,
        {
            "bias": 1,
            "bias_label": "bullish",
            "confidence": 0.8,
            "event_risk": False,
        },
    )
    assert support["state"] == "setup_macro_support"
    assert support["research_ready"] is True
    assert support["execution_allowed"] is False

    conflict = service.build_decision_fusion(
        technical,
        {
            "bias": -1,
            "bias_label": "bearish",
            "confidence": 0.9,
            "event_risk": False,
        },
    )
    assert conflict["state"] == "setup_macro_conflict"
    assert conflict["macro_relation"] == "conflict"

    event = service.build_decision_fusion(
        technical,
        {
            "bias": 1,
            "bias_label": "bullish",
            "confidence": 0.9,
            "event_risk": True,
        },
    )
    assert event["state"] == "event_gate"
    assert event["research_ready"] is False
    assert "high_impact_macro_event" in event["reasons"]


def test_macro_json_parser_accepts_plain_object():
    parsed = service._parse_json(
        '{"bias":1,"confidence":0.7,"event_risk":false,"summary":"x","drivers":[]}'
    )
    assert parsed["bias"] == 1
    assert parsed["confidence"] == 0.7



def test_sampled_spot_bars_use_stable_bucket_timestamp():
    base = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    series = XAUMicroSeries(
        points=(
            XAUMicroPoint(timestamp=base + timedelta(minutes=1), price=4300.0),
            XAUMicroPoint(timestamp=base + timedelta(minutes=3), price=4301.0),
            XAUMicroPoint(timestamp=base + timedelta(minutes=4), price=4302.0),
            XAUMicroPoint(timestamp=base + timedelta(minutes=6), price=4303.0),
        ),
        observed_at=base + timedelta(minutes=6),
        source="test",
        age_seconds=0.0,
        coverage_seconds=360.0,
        is_stale=False,
    )

    bars = sampled_spot_bars(series, XAUTimeframe.M5)

    assert len(bars) == 2
    assert bars[0].timestamp == base
    assert bars[1].timestamp == base + timedelta(minutes=5)
    assert bars[0].close == 4302.0
    assert bars[1].close == 4303.0
