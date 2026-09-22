from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta, timezone

import pytest

from src.platform.marketdata.xau_dukascopy import (
    DukascopyHourFetch,
    DukascopyXAUHistoryProvider,
    decode_dukascopy_xau_bi5,
    dukascopy_hour_url,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata.xau_provider_audit import audit_xau_provider_overlap


UTC = timezone.utc
HOUR = datetime(2020, 1, 6, 12, tzinfo=UTC)
RECORD = struct.Struct(">IIIff")


def _payload(records):
    raw = b"".join(RECORD.pack(*row) for row in records)
    return lzma.compress(raw)


def test_dukascopy_url_uses_zero_indexed_month():
    url = dukascopy_hour_url(HOUR)
    assert url.endswith("/XAUUSD/2020/00/06/12h_ticks.bi5")


def test_decoder_parses_xau_divider_and_bid_ask_mid():
    payload = _payload(
        [
            (1000, 1550500, 1550300, 2.0, 3.0),
            (2500, 1550700, 1550500, 4.0, 5.0),
        ]
    )
    ticks = decode_dukascopy_xau_bi5(payload, HOUR)

    assert len(ticks) == 2
    assert ticks[0].timestamp == HOUR + timedelta(seconds=1)
    assert ticks[0].ask == pytest.approx(1550.5)
    assert ticks[0].bid == pytest.approx(1550.3)
    assert ticks[0].mid == pytest.approx(1550.4)
    assert ticks[0].quoted_volume == pytest.approx(5.0)


def test_decoder_drops_zero_volume_ticks():
    payload = _payload(
        [
            (1000, 1550500, 1550300, 0.0, 0.0),
            (2000, 1550600, 1550400, 1.0, 1.0),
        ]
    )
    ticks = decode_dukascopy_xau_bi5(payload, HOUR)
    assert len(ticks) == 1


def test_decoder_rejects_non_chronological_hour():
    payload = _payload(
        [
            (2000, 1550500, 1550300, 1.0, 1.0),
            (1000, 1550600, 1550400, 1.0, 1.0),
        ]
    )
    with pytest.raises(ValueError, match="not chronological"):
        decode_dukascopy_xau_bi5(payload, HOUR)


class FixtureProvider(DukascopyXAUHistoryProvider):
    def fetch_hour(self, hour):
        hour = hour.astimezone(UTC)
        payload = _payload(
            [
                (0, 2000100, 1999900, 1.0, 2.0),
                (30_000, 2000300, 2000100, 2.0, 3.0),
                (70_000, 2000500, 2000300, 3.0, 4.0),
                (310_000, 2000700, 2000500, 4.0, 5.0),
            ]
        )
        ticks = decode_dukascopy_xau_bi5(payload, hour)
        return DukascopyHourFetch(
            hour=hour,
            status="data",
            ticks=ticks,
            url=dukascopy_hour_url(hour),
            attempts=1,
        )


def test_provider_aggregates_mid_bars_and_preserves_proxy_semantics():
    provider = FixtureProvider()
    bars, diagnostics = provider.bars_range(
        XAUTimeframe.M1,
        start=HOUR,
        end=HOUR + timedelta(hours=1),
    )
    assert len(bars) == 3
    assert bars[0].open == pytest.approx(2000.0)
    assert bars[0].close == pytest.approx(2000.2)
    assert bars[0].high >= bars[0].close
    assert bars[0].source == "dukascopy:datafeed:XAUUSD:mid"
    assert bars[0].symbol == "XAUUSD"
    assert bars[0].execution_eligible is False
    assert diagnostics.centralized_order_flow is False
    assert diagnostics.volume_semantics == "quoted_bid_plus_ask_activity_proxy"


def test_provider_range_safety_cap_requires_incremental_acquisition():
    provider = FixtureProvider()
    with pytest.raises(ValueError, match="safety cap"):
        provider.ticks_range(
            start=HOUR,
            end=HOUR + timedelta(hours=25),
            max_hours=24,
        )


def _bar(ts, close, source):
    return XAUBar(
        timestamp=ts,
        timeframe=XAUTimeframe.M5,
        open=close,
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        volume=100.0,
        source=source,
        symbol="XAUUSD",
        execution_eligible=False,
    )


def test_cross_provider_audit_characterizes_difference_but_never_auto_merges():
    left = [
        _bar(HOUR + timedelta(minutes=5 * i), 2000.0 + i, "dukascopy:datafeed:XAUUSD:mid")
        for i in range(10)
    ]
    right = [
        _bar(HOUR + timedelta(minutes=5 * i), 2000.1 + i, "biquote.io:MT5-ohlc")
        for i in range(10)
    ]
    audit = audit_xau_provider_overlap(left, right)

    assert audit.overlap_count == 10
    assert audit.same_instrument is True
    assert audit.same_timeframe is True
    assert audit.median_close_deviation_bps is not None
    assert audit.p95_return_deviation_bps is not None
    assert audit.auto_merge_allowed is False


def test_cross_provider_audit_rejects_timeframe_mismatch():
    left = [_bar(HOUR, 2000.0, "dukascopy")]
    right = [
        XAUBar(
            timestamp=HOUR,
            timeframe=XAUTimeframe.M15,
            open=2000.0,
            high=2000.2,
            low=1999.8,
            close=2000.0,
            source="biquote",
            symbol="XAUUSD",
            execution_eligible=False,
        )
    ]
    with pytest.raises(ValueError, match="timeframes must match"):
        audit_xau_provider_overlap(left, right)
