from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.modules.strategy.xau_kvo import assess_kvo_gold, kvo_reading
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _bars(direction: int = 1, n: int = 120):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 2600.0
    for i in range(n):
        wave = ((i % 11) - 5) * 0.08
        move = direction * 0.35 + wave
        open_ = price
        close = max(1.0, open_ + move)
        high = max(open_, close) + 0.8
        low = min(open_, close) - 0.7
        rows.append(XAUBar(
            timestamp=start + timedelta(minutes=15 * i),
            timeframe=XAUTimeframe.M15,
            open=open_, high=high, low=low, close=close,
            volume=100 + (i % 17) * 7,
            source="synthetic-tick-volume",
            execution_eligible=False,
        ))
        price = close
    return rows


def test_kvo_is_available_with_standard_periods():
    value = kvo_reading(_bars())
    assert value.available is True
    assert value.kvo is not None
    assert value.signal is not None
    assert value.histogram is not None


def test_kvo_rejects_missing_volume():
    rows = [
        XAUBar(
            timestamp=b.timestamp,
            timeframe=b.timeframe,
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=0.0, source=b.source, execution_eligible=False,
        )
        for b in _bars()
    ]
    assert kvo_reading(rows).available is False


def test_gold_assessment_is_confirmation_not_standalone_execution():
    m15 = _bars(1, 140)
    h1 = [
        XAUBar(
            timestamp=b.timestamp,
            timeframe=XAUTimeframe.H1,
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=b.volume, source=b.source, execution_eligible=False,
        )
        for b in _bars(1, 140)
    ]
    result = assess_kvo_gold(
        m15, h1, candidate="long_setup", setup_type="breakout"
    )
    assert result.direction == "LONG"
    assert result.setup_type == "breakout"
    assert result.m15.available and result.h1.available
