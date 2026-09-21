from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.modules.xau import service
from src.modules.xau.service import _resolve_event_gate, _validated_event_gate
from src.platform.marketdata.xau_biquote import (
    BiquoteEconomicCalendarProvider,
    BiquoteXAUOHLCProvider,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata import xau_spot_reference
from src.platform.marketdata.xau_spot_reference import (
    BiquoteXAUIndicativeSpotReference,
    CompositeXAUIndicativeSpotProvider,
    XAUIndicativeSpot,
)
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


async def _fake_consensus(force: bool = False):
    return {
        "source_count": 2,
        "usable_count": 2,
        "reference_median": 2620.0,
        "references": [
            {"source": "alt-a", "price": 2620.1, "is_stale": False},
            {"source": "alt-b", "price": 2619.9, "is_stale": False},
        ],
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
    monkeypatch.setattr(service, "get_spot_consensus", _fake_consensus)
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
    monkeypatch.setattr(service, "get_spot_consensus", _fake_consensus)

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
            "event_kind": "scheduled",
            "event_name": "US CPI",
            "event_time_utc": "2026-09-21T15:45:00+00:00",
            "event_age_minutes": None,
            "event_confidence": 0.9,
            "event_validation": "scheduled_event_window",
        },
    )
    assert event["state"] == "event_gate"
    assert event["research_ready"] is False
    assert event["event_kind"] == "scheduled"
    assert event["event_name"] == "US CPI"
    assert event["event_validation"] == "scheduled_event_window"
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



def test_biquote_parses_fresh_bid_ask_quote(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "symbol": "XAUUSD",
                "bid": 4350.10,
                "ask": 4350.40,
                "mid": 4350.25,
                "timestamp": "2026-09-21T14:20:00Z",
                "source": "MetaTrader 5 (Broker 1)",
                "marketState": "open",
                "stale": False,
                "quoteAgeSeconds": 2,
            }

    monkeypatch.setattr(
        xau_spot_reference.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    quote = BiquoteXAUIndicativeSpotReference().fetch()

    assert quote.bid == 4350.10
    assert quote.ask == 4350.40
    assert quote.price == 4350.25
    assert quote.is_stale is False
    assert quote.execution_eligible is False
    assert quote.source.startswith("biquote.io:")


def test_biquote_marks_closed_market_as_stale(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "bid": 4350.10,
                "ask": 4350.40,
                "mid": 4350.25,
                "timestamp": "2026-09-21T14:20:00Z",
                "source": "MetaTrader 5 (Broker 1)",
                "marketState": "closed",
                "stale": False,
                "quoteAgeSeconds": 2,
            }

    monkeypatch.setattr(
        xau_spot_reference.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    quote = BiquoteXAUIndicativeSpotReference().fetch()
    assert quote.is_stale is True
    assert quote.market_state == "closed"


def test_composite_prefers_fresh_bid_ask_over_mid_only():
    fresh_mid = XAUIndicativeSpot(
        price=4350.0,
        bid=None,
        ask=None,
        observed_at=datetime.now(timezone.utc),
        source="mid-only",
        is_stale=False,
    )
    fresh_quote = XAUIndicativeSpot(
        price=4350.25,
        bid=4350.10,
        ask=4350.40,
        observed_at=datetime.now(timezone.utc),
        source="bid-ask",
        is_stale=False,
    )

    class Provider:
        def __init__(self, quote):
            self.quote = quote

        def fetch(self, timeout_seconds=10.0):
            return self.quote

    composite = CompositeXAUIndicativeSpotProvider()
    composite.providers = (Provider(fresh_mid), Provider(fresh_quote))

    result = composite.fetch()
    assert result.source == "bid-ask"
    assert result.bid == 4350.10
    assert result.ask == 4350.40



def test_event_gate_requires_scheduled_window_or_fresh_breaking_event():
    now = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)

    scheduled_active = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "scheduled",
            "event_name": "US CPI",
            "event_time_utc": "2026-09-21T15:45:00Z",
            "event_confidence": 0.9,
        },
        now=now,
    )
    assert scheduled_active["event_risk"] is True
    assert scheduled_active["event_validation"] == "scheduled_event_window"

    scheduled_too_far = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "scheduled",
            "event_name": "FOMC",
            "event_time_utc": "2026-09-21T18:00:00Z",
            "event_confidence": 0.95,
        },
        now=now,
    )
    assert scheduled_too_far["event_risk"] is False
    assert scheduled_too_far["event_validation"] == "scheduled_event_outside_window"

    scheduled_old = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "scheduled",
            "event_name": "US payrolls",
            "event_time_utc": "2026-09-21T14:00:00Z",
            "event_confidence": 0.95,
        },
        now=now,
    )
    assert scheduled_old["event_risk"] is False

    breaking_active = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "breaking",
            "event_name": "Unexpected central-bank announcement",
            "event_age_minutes": 12,
            "event_confidence": 0.9,
        },
        now=now,
    )
    assert breaking_active["event_risk"] is True
    assert breaking_active["event_validation"] == "breaking_event_window"

    breaking_old = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "breaking",
            "event_name": "Older shock",
            "event_age_minutes": 75,
            "event_confidence": 0.9,
        },
        now=now,
    )
    assert breaking_old["event_risk"] is False
    assert breaking_old["event_validation"] == "breaking_event_too_old"

    low_confidence = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "breaking",
            "event_name": "Unverified headline",
            "event_age_minutes": 5,
            "event_confidence": 0.4,
        },
        now=now,
    )
    assert low_confidence["event_risk"] is False
    assert low_confidence["event_validation"] == "event_confidence_too_low"


def test_event_gate_rejects_vague_or_untimed_claims():
    now = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)

    vague = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "none",
            "event_name": "Ongoing geopolitical tensions",
            "event_confidence": 0.9,
        },
        now=now,
    )
    assert vague["event_risk"] is False
    assert vague["event_validation"] == "unsupported_event_kind"

    untimed = _validated_event_gate(
        {
            "event_risk": True,
            "event_kind": "scheduled",
            "event_name": "Fed remarks",
            "event_time_utc": None,
            "event_confidence": 0.9,
        },
        now=now,
    )
    assert untimed["event_risk"] is False
    assert untimed["event_validation"] == "missing_or_invalid_event_time"



def test_biquote_ohlc_parses_and_sorts_mt5_bars(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "symbol": "XAUUSD",
                "interval": "1m",
                "bars": [
                    {
                        "openTime": "2026-09-21T15:01:00Z",
                        "open": 4351.0,
                        "high": 4352.0,
                        "low": 4350.5,
                        "close": 4351.5,
                        "volume": 0,
                        "tickVolume": 120,
                        "isOpen": True,
                    },
                    {
                        "openTime": "2026-09-21T15:00:00Z",
                        "open": 4350.0,
                        "high": 4351.2,
                        "low": 4349.8,
                        "close": 4351.0,
                        "volume": 0,
                        "tickVolume": 100,
                        "isOpen": False,
                    },
                ],
            }

    from src.platform.marketdata import xau_biquote
    monkeypatch.setattr(xau_biquote.httpx, "get", lambda *args, **kwargs: FakeResponse())

    bars = BiquoteXAUOHLCProvider().bars(XAUTimeframe.M1, limit=100)

    assert len(bars) == 2
    assert bars[0].timestamp.isoformat() == "2026-09-21T15:00:00+00:00"
    assert bars[1].timestamp.isoformat() == "2026-09-21T15:01:00+00:00"
    assert bars[1].close == 4351.5
    assert bars[1].source == "biquote.io:MT5-ohlc"
    assert bars[1].execution_eligible is False


def test_biquote_calendar_selects_high_impact_exact_usd_window(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "id": "mql5:1",
                    "time": "2026-09-21T15:45:00Z",
                    "countryCode": "US",
                    "currency": "USD",
                    "name": "US CPI",
                    "importance": "high",
                    "timeMode": "exact",
                    "sourceUrl": "https://example.test/cpi",
                },
                {
                    "id": "mql5:2",
                    "time": "2026-09-21T15:20:00Z",
                    "countryCode": "US",
                    "currency": "USD",
                    "name": "Low impact",
                    "importance": "low",
                    "timeMode": "exact",
                },
            ]

    from src.platform.marketdata import xau_biquote
    monkeypatch.setattr(xau_biquote.httpx, "get", lambda *args, **kwargs: FakeResponse())

    event = BiquoteEconomicCalendarProvider().active_usd_event(
        now=datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc),
        before_minutes=90,
        after_minutes=30,
    )

    assert event is not None
    assert event["event_risk"] is True
    assert event["event_name"] == "US CPI"
    assert event["event_validation"] == "calendar_high_impact_window"


def test_structured_calendar_overrides_ai_scheduled_claim_and_preserves_breaking():
    ai_scheduled = {
        "event_risk": True,
        "event_kind": "scheduled",
        "event_name": "Unverified scheduled claim",
        "event_time_utc": "2026-09-21T15:30:00+00:00",
        "event_age_minutes": None,
        "event_confidence": 0.9,
        "event_validation": "scheduled_event_window",
    }
    no_calendar = _resolve_event_gate(ai_scheduled, None)
    assert no_calendar["event_risk"] is False
    assert no_calendar["event_validation"] == "scheduled_event_requires_calendar"

    breaking = {
        "event_risk": True,
        "event_kind": "breaking",
        "event_name": "Breaking shock",
        "event_time_utc": None,
        "event_age_minutes": 5.0,
        "event_confidence": 0.9,
        "event_validation": "breaking_event_window",
    }
    resolved_breaking = _resolve_event_gate(breaking, None)
    assert resolved_breaking["event_risk"] is True

    calendar = {
        "event_risk": True,
        "event_kind": "scheduled",
        "event_name": "US CPI",
        "event_time_utc": "2026-09-21T15:45:00+00:00",
        "event_age_minutes": None,
        "event_confidence": 1.0,
        "event_validation": "calendar_high_impact_window",
        "event_source_url": "https://example.test/cpi",
    }
    resolved_calendar = _resolve_event_gate(breaking, calendar)
    assert resolved_calendar["event_risk"] is True
    assert resolved_calendar["event_name"] == "US CPI"
    assert resolved_calendar["event_validation"] == "calendar_high_impact_window"



def test_stale_5m_biquote_is_replaced_by_fresh_sampled_spot(monkeypatch):
    fresh_1m = _bars(XAUTimeframe.M1)
    stale_5m = [
        XAUBar(
            timestamp=bar.timestamp - timedelta(minutes=40),
            timeframe=bar.timeframe,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            source="biquote.io:MT5-ohlc",
            execution_eligible=False,
        )
        for bar in _bars(XAUTimeframe.M5)
    ]
    fresh_15m = _bars(XAUTimeframe.M15)

    def fake_biquote(self, timeframe, *, limit=240, timeout_seconds=12.0):
        if timeframe == XAUTimeframe.M1:
            return fresh_1m
        if timeframe == XAUTimeframe.M5:
            return stale_5m
        return fresh_15m

    async def fake_series(force: bool = False):
        now = datetime.now(timezone.utc)
        points = [
            XAUMicroPoint(
                timestamp=now - timedelta(minutes=(119 - i)),
                price=2600.0 + i * 0.1,
            )
            for i in range(120)
        ]
        return XAUMicroSeries(
            points=points,
            observed_at=now,
            source="test-live-spot",
            is_stale=False,
            age_seconds=1.0,
            coverage_seconds=7140.0,
        )

    monkeypatch.setattr(BiquoteXAUOHLCProvider, "bars", fake_biquote)
    monkeypatch.setattr(service, "get_micro_series", fake_series)
    service._bars_cache = None

    result = asyncio.run(service.get_research_bars(force=True))

    assert result[XAUTimeframe.M5]
    assert result[XAUTimeframe.M5][-1].source == "xaus.com:intraday-sampled"
    assert datetime.now(timezone.utc) - result[XAUTimeframe.M5][-1].timestamp <= timedelta(minutes=12)



def test_macro_cold_start_returns_immediately_and_refreshes_in_background(monkeypatch):
    async def scenario():
        gate = asyncio.Event()

        async def slow_refresh(force: bool = False):
            await gate.wait()
            return {
                "bias": -1,
                "bias_label": "bearish",
                "confidence": 0.8,
                "event_risk": False,
            }

        service._macro_cache = None
        service._macro_refresh_task = None
        monkeypatch.setattr(service, "_refresh_macro_context", slow_refresh)

        result = await service.get_macro_context(force=False)
        assert result["bias"] == 0
        assert result["cache_stale"] is True
        assert result["refresh_pending"] is True
        assert service._macro_refresh_task is not None
        assert service._macro_refresh_task.done() is False

        service._macro_refresh_task.cancel()
        try:
            await service._macro_refresh_task
        except asyncio.CancelledError:
            pass
        service._macro_refresh_task = None

    asyncio.run(scenario())


def test_macro_force_waits_for_full_refresh(monkeypatch):
    async def scenario():
        async def refresh(force: bool = False):
            assert force is True
            return {
                "bias": 1,
                "bias_label": "bullish",
                "confidence": 0.75,
                "event_risk": False,
                "cache_stale": False,
                "refresh_pending": False,
            }

        service._macro_cache = None
        service._macro_refresh_task = None
        monkeypatch.setattr(service, "_refresh_macro_context", refresh)

        result = await service.get_macro_context(force=True)
        assert result["bias"] == 1
        assert result["confidence"] == 0.75

    asyncio.run(scenario())



def test_indicative_spot_marks_quotes_older_than_three_minutes_stale(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(seconds=181)

    def fake_fetch(self, timeout_seconds: float = 10.0):
        quote = XAUIndicativeSpot(
            price=2620.0,
            bid=2619.8,
            ask=2620.2,
            observed_at=old,
            source="test-old-spot",
            is_stale=False,
        )
        return quote, [{
            "provider": "test",
            "status": "ok",
            "source": quote.source,
            "is_stale": True,
            "selected": True,
        }]

    monkeypatch.setattr(CompositeXAUIndicativeSpotProvider, "fetch_with_diagnostics", fake_fetch)
    service._spot_cache = None
    result = asyncio.run(service.get_indicative_spot(force=True))
    assert result["age_seconds"] >= 180.0
    assert result["is_stale"] is True



def test_composite_spot_provider_reports_fallback_health():
    now = datetime.now(timezone.utc)

    class Broken:
        def fetch(self, timeout_seconds: float = 10.0):
            raise RuntimeError("boom")

    class MidOnly:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4340.0,
                bid=None,
                ask=None,
                observed_at=now,
                source="mid-only",
                is_stale=False,
            )

    provider = CompositeXAUIndicativeSpotProvider()
    provider.providers = (Broken(), MidOnly())
    quote, health = provider.fetch_with_diagnostics()

    assert quote.source == "mid-only"
    assert health[0]["status"] == "error"
    assert health[0]["error_type"] == "RuntimeError"
    assert health[1]["status"] == "ok"
    assert health[1]["has_bid_ask"] is False
    assert health[1]["selected"] is True


def test_composite_spot_provider_prefers_fresh_bid_ask():
    now = datetime.now(timezone.utc)

    class MidOnly:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4340.0,
                bid=None,
                ask=None,
                observed_at=now,
                source="mid-only",
                is_stale=False,
            )

    class FreshBidAsk:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4340.1,
                bid=4340.0,
                ask=4340.2,
                observed_at=now,
                source="fresh-bidask",
                is_stale=False,
            )

    provider = CompositeXAUIndicativeSpotProvider()
    provider.providers = (MidOnly(), FreshBidAsk())
    quote, health = provider.fetch_with_diagnostics()

    assert quote.source == "fresh-bidask"
    assert quote.bid == 4340.0
    assert quote.ask == 4340.2
    assert health[-1]["selected"] is True



def test_composite_spot_provider_prefers_newer_fresh_mid_fallback():
    now = datetime.now(timezone.utc)

    class OlderMid:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4340.0,
                bid=None,
                ask=None,
                observed_at=now - timedelta(seconds=120),
                source="older-mid",
                is_stale=False,
            )

    class NewerMid:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4341.0,
                bid=None,
                ask=None,
                observed_at=now - timedelta(seconds=10),
                source="newer-mid",
                is_stale=False,
            )

    provider = CompositeXAUIndicativeSpotProvider()
    provider.providers = (OlderMid(), NewerMid())
    quote, health = provider.fetch_with_diagnostics()

    assert quote.source == "newer-mid"
    selected = [row for row in health if row.get("selected")]
    assert len(selected) == 1
    assert selected[0]["source"] == "newer-mid"
    assert selected[0]["selection_reason"] == "fresh_context_fallback"


def test_composite_spot_provider_marks_overage_quote_stale_before_selection():
    now = datetime.now(timezone.utc)

    class OldBidAsk:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4340.0,
                bid=4339.9,
                ask=4340.1,
                observed_at=now - timedelta(seconds=240),
                source="old-bidask",
                is_stale=False,
            )

    class FreshMid:
        def fetch(self, timeout_seconds: float = 10.0):
            return XAUIndicativeSpot(
                price=4341.0,
                bid=None,
                ask=None,
                observed_at=now - timedelta(seconds=10),
                source="fresh-mid",
                is_stale=False,
            )

    provider = CompositeXAUIndicativeSpotProvider()
    provider.providers = (OldBidAsk(), FreshMid())
    quote, health = provider.fetch_with_diagnostics()

    assert quote.source == "fresh-mid"
    old = next(row for row in health if row.get("source") == "old-bidask")
    assert old["provider_stale"] is False
    assert old["is_stale"] is True



def test_biquote_strict_404_retries_as_stale_context(monkeypatch):
    calls = []

    class Strict404:
        status_code = 404
        def raise_for_status(self):
            raise AssertionError("strict 404 should be retried before raise_for_status")
        def json(self):
            return {}

    class ContextResponse:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {
                "symbol": "XAUUSD",
                "bid": 4340.10,
                "ask": 4340.40,
                "mid": 4340.25,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "MetaTrader 5 (Broker 1)",
                "marketState": "open",
                "stale": False,
                "quoteAgeSeconds": 301,
            }

    def fake_get(*args, **kwargs):
        calls.append(kwargs.get("params"))
        return Strict404() if len(calls) == 1 else ContextResponse()

    monkeypatch.setattr(xau_spot_reference.httpx, "get", fake_get)

    quote = BiquoteXAUIndicativeSpotReference().fetch()

    assert len(calls) == 2
    assert calls[0] == {"allowStale": "false"}
    assert calls[1] is None
    assert quote.bid == 4340.10
    assert quote.ask == 4340.40
    assert quote.is_stale is True
    assert quote.provider_quote_age_seconds == 301
    assert quote.execution_eligible is False
