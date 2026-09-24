"""Klinger Volume Oscillator research layer for XAUUSD.

Uses closed OHLCV bars only. For spot XAUUSD, volume is a tick/activity proxy,
not centralized global traded volume. The KVO layer is therefore confirmation
evidence, never an execution quote or standalone directional oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.platform.marketdata.xau_models import XAUBar


@dataclass(frozen=True)
class KVOReading:
    available: bool
    kvo: float | None = None
    signal: float | None = None
    histogram: float | None = None
    previous_histogram: float | None = None
    histogram_delta: float | None = None
    cross: str = "none"
    recent_cross: str = "none"
    bar_count: int = 0
    volume_source: str = "tick_volume_proxy"


@dataclass(frozen=True)
class KVOGoldAssessment:
    allowed: bool
    setup_type: str
    direction: str
    reason: str
    m15: KVOReading
    h1: KVOReading


def _ema(values: Sequence[float], period: int) -> list[float | None]:
    if period < 1:
        raise ValueError("period must be >= 1")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    current = seed
    for index in range(period, len(values)):
        current = (values[index] - current) * alpha + current
        out[index] = current
    return out


def kvo_reading(
    bars: Sequence[XAUBar],
    *,
    fast: int = 34,
    slow: int = 55,
    signal_period: int = 13,
    recent_cross_bars: int = 4,
) -> KVOReading:
    rows = list(bars)
    minimum = slow + signal_period + 2
    if len(rows) < minimum:
        return KVOReading(available=False, bar_count=len(rows))
    if not any(float(row.volume or 0.0) > 0.0 for row in rows[-minimum:]):
        return KVOReading(available=False, bar_count=len(rows))

    dm = [max(0.0, float(row.high) - float(row.low)) for row in rows]
    trend = [0] * len(rows)
    cm = [0.0] * len(rows)
    vf = [0.0] * len(rows)

    trend[0] = 1
    cm[0] = dm[0]
    for i in range(1, len(rows)):
        current_hlc = float(rows[i].high) + float(rows[i].low) + float(rows[i].close)
        previous_hlc = (
            float(rows[i - 1].high)
            + float(rows[i - 1].low)
            + float(rows[i - 1].close)
        )
        trend[i] = 1 if current_hlc > previous_hlc else -1
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i]
        volume = max(0.0, float(rows[i].volume or 0.0))
        if cm[i] > 0.0:
            force = abs(2.0 * ((dm[i] / cm[i]) - 1.0))
            vf[i] = volume * force * trend[i] * 100.0

    fast_ema = _ema(vf, fast)
    slow_ema = _ema(vf, slow)
    kvo_values: list[float | None] = [None] * len(rows)
    valid_kvo: list[float] = []
    valid_indices: list[int] = []
    for i, (a, b) in enumerate(zip(fast_ema, slow_ema)):
        if a is None or b is None:
            continue
        value = float(a) - float(b)
        kvo_values[i] = value
        valid_kvo.append(value)
        valid_indices.append(i)

    signal_valid = _ema(valid_kvo, signal_period)
    signal_values: list[float | None] = [None] * len(rows)
    for j, index in enumerate(valid_indices):
        signal_values[index] = signal_valid[j]

    hist: list[float | None] = [None] * len(rows)
    for i, (k, s) in enumerate(zip(kvo_values, signal_values)):
        if k is not None and s is not None:
            hist[i] = float(k) - float(s)

    valid_hist_indices = [i for i, value in enumerate(hist) if value is not None]
    if len(valid_hist_indices) < 2:
        return KVOReading(available=False, bar_count=len(rows))

    i = valid_hist_indices[-1]
    p = valid_hist_indices[-2]
    current_hist = float(hist[i])
    previous_hist = float(hist[p])
    cross = (
        "bullish"
        if previous_hist <= 0.0 < current_hist
        else "bearish"
        if previous_hist >= 0.0 > current_hist
        else "none"
    )

    recent_cross = "none"
    recent_indices = valid_hist_indices[-max(2, recent_cross_bars + 1):]
    for left, right in zip(recent_indices, recent_indices[1:]):
        a = float(hist[left])
        b = float(hist[right])
        if a <= 0.0 < b:
            recent_cross = "bullish"
        elif a >= 0.0 > b:
            recent_cross = "bearish"

    return KVOReading(
        available=True,
        kvo=float(kvo_values[i]),
        signal=float(signal_values[i]),
        histogram=current_hist,
        previous_histogram=previous_hist,
        histogram_delta=current_hist - previous_hist,
        cross=cross,
        recent_cross=recent_cross,
        bar_count=len(rows),
    )


def assess_kvo_gold(
    m15_bars: Sequence[XAUBar],
    h1_bars: Sequence[XAUBar],
    *,
    candidate: str,
    setup_type: str,
) -> KVOGoldAssessment:
    side = 1 if candidate == "long_setup" else -1
    direction = "LONG" if side > 0 else "SHORT"
    wanted = "bullish" if side > 0 else "bearish"
    opposite = "bearish" if side > 0 else "bullish"
    m15 = kvo_reading(m15_bars)
    h1 = kvo_reading(h1_bars)

    if not m15.available or not h1.available:
        return KVOGoldAssessment(
            False, setup_type, direction, "kvo_unavailable", m15, h1
        )

    m15_hist = float(m15.histogram or 0.0) * side
    m15_delta = float(m15.histogram_delta or 0.0) * side
    h1_hist = float(h1.histogram or 0.0) * side
    h1_delta = float(h1.histogram_delta or 0.0) * side

    if h1_hist < 0.0 and h1_delta < 0.0:
        return KVOGoldAssessment(
            False, setup_type, direction, "h1_kvo_opposes", m15, h1
        )

    if setup_type == "trend_pullback":
        allowed = m15.recent_cross == wanted and m15_delta > 0.0
        reason = "pullback_kvo_reentry" if allowed else "pullback_no_kvo_reentry"
    elif setup_type == "breakout":
        allowed = m15_hist > 0.0 and m15_delta > 0.0 and h1_delta >= 0.0
        reason = "breakout_kvo_expansion" if allowed else "breakout_no_kvo_expansion"
    elif setup_type == "sweep_reversal":
        allowed = (
            m15.recent_cross == wanted
            and m15.cross != opposite
            and m15_delta > 0.0
        )
        reason = "sweep_kvo_turn" if allowed else "sweep_no_kvo_turn"
    else:
        allowed = False
        reason = "unsupported_setup"

    return KVOGoldAssessment(allowed, setup_type, direction, reason, m15, h1)
