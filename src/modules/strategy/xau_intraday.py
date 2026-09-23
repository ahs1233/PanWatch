"""Deterministic XAU intraday state engine for 1m/5m/15m research.

The engine does not call an LLM and does not place trades. It produces a
candidate setup plus explicit gates/invalidation references that a paper-trade
or alert layer can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Iterable

from src.platform.marketdata.xau_models import (
    XAUBar,
    XAUQuote,
    XAUTimeframe,
)


@dataclass(frozen=True)
class XAUFrameState:
    timeframe: XAUTimeframe
    close: float
    ema_fast: float
    ema_slow: float
    rsi14: float
    atr14: float
    atr_pct: float
    breakout: str
    direction: str
    recent_swing_high: float
    recent_swing_low: float
    observed_at: datetime


@dataclass(frozen=True)
class XAUIntradayAssessment:
    status: str
    candidate: str
    blocked: bool
    block_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    frame_states: dict[str, XAUFrameState]
    macro_bias: int
    event_risk: bool
    spread_bps: float | None
    atr_reference: float | None
    swing_high_reference: float | None
    swing_low_reference: float | None


class XAUIntradayEngine:
    """Small deterministic engine suitable for frequent evaluation."""

    _MAX_AGE = {
        XAUTimeframe.M1: timedelta(minutes=3),
        XAUTimeframe.M5: timedelta(minutes=12),
        XAUTimeframe.M15: timedelta(minutes=35),
    }

    def __init__(
        self,
        *,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14,
        breakout_lookback: int = 20,
        max_spread_bps: float | None = None,
        require_execution_data: bool = False,
    ) -> None:
        if fast_ema < 2 or slow_ema <= fast_ema:
            raise ValueError("EMA periods must satisfy 2 <= fast < slow")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.max_spread_bps = max_spread_bps
        self.require_execution_data = require_execution_data

    def analyze(
        self,
        bars_by_timeframe: dict[XAUTimeframe, list[XAUBar]],
        *,
        quote: XAUQuote | None = None,
        event_risk: bool = False,
        macro_bias: int = 0,
        now: datetime | None = None,
        assume_sorted: bool = False,
    ) -> XAUIntradayAssessment:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        macro_bias = max(-1, min(1, int(macro_bias)))

        states: dict[str, XAUFrameState] = {}
        reasons: list[str] = []
        warnings: list[str] = []

        for timeframe in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15):
            source_bars = bars_by_timeframe.get(timeframe) or []
            bars = (
                source_bars
                if assume_sorted
                else sorted(source_bars, key=lambda item: item.timestamp)
            )
            minimum = max(
                self.slow_ema + 2,
                self.rsi_period + 2,
                self.atr_period + 2,
                self.breakout_lookback + 2,
            )
            if len(bars) < minimum:
                reasons.append(f"insufficient_{timeframe.value}_bars")
                continue
            latest = bars[-1]
            if now - latest.timestamp > self._MAX_AGE[timeframe]:
                reasons.append(f"stale_{timeframe.value}_bars")
                continue
            if self.require_execution_data and not latest.execution_eligible:
                reasons.append(f"execution_{timeframe.value}_bars_required")
            states[timeframe.value] = self._frame_state(bars, timeframe)

        if self.require_execution_data and quote is None:
            reasons.append("execution_quote_missing")
        if quote is not None:
            if not quote.execution_eligible:
                reasons.append("quote_not_execution_eligible")
            if now - quote.observed_at > timedelta(seconds=30):
                reasons.append("stale_quote")
            if (
                self.max_spread_bps is not None
                and quote.spread_bps > self.max_spread_bps
            ):
                reasons.append("spread_too_wide")

        if event_risk:
            reasons.append("high_impact_event_gate")

        required = {"1m", "5m", "15m"}
        if not required.issubset(states):
            return XAUIntradayAssessment(
                status="blocked",
                candidate="none",
                blocked=True,
                block_reasons=tuple(dict.fromkeys(reasons)),
                warnings=tuple(dict.fromkeys(warnings)),
                frame_states=states,
                macro_bias=macro_bias,
                event_risk=event_risk,
                spread_bps=quote.spread_bps if quote else None,
                atr_reference=states.get("5m").atr14 if states.get("5m") else None,
                swing_high_reference=states.get("5m").recent_swing_high if states.get("5m") else None,
                swing_low_reference=states.get("5m").recent_swing_low if states.get("5m") else None,
            )

        one = states["1m"]
        five = states["5m"]
        fifteen = states["15m"]

        candidate = "none"
        if (
            five.direction == "bullish"
            and fifteen.direction == "bullish"
            and one.direction != "bearish"
        ):
            candidate = "long_setup"
        elif (
            five.direction == "bearish"
            and fifteen.direction == "bearish"
            and one.direction != "bullish"
        ):
            candidate = "short_setup"

        if candidate == "long_setup" and macro_bias < 0:
            warnings.append("macro_bias_conflicts_long")
        elif candidate == "short_setup" and macro_bias > 0:
            warnings.append("macro_bias_conflicts_short")

        blocked = any(
            reason
            in {
                "high_impact_event_gate",
                "stale_quote",
                "quote_not_execution_eligible",
                "spread_too_wide",
                "execution_quote_missing",
            }
            or reason.startswith("stale_")
            or reason.startswith("insufficient_")
            or reason.startswith("execution_")
            for reason in reasons
        )

        return XAUIntradayAssessment(
            status="blocked" if blocked else "ready",
            candidate="none" if blocked else candidate,
            blocked=blocked,
            block_reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(dict.fromkeys(warnings)),
            frame_states=states,
            macro_bias=macro_bias,
            event_risk=event_risk,
            spread_bps=quote.spread_bps if quote else None,
            atr_reference=five.atr14,
            swing_high_reference=five.recent_swing_high,
            swing_low_reference=five.recent_swing_low,
        )

    def _frame_state(
        self,
        bars: list[XAUBar],
        timeframe: XAUTimeframe,
    ) -> XAUFrameState:
        closes = [bar.close for bar in bars]
        ema_fast = _ema(closes, self.fast_ema)
        ema_slow = _ema(closes, self.slow_ema)
        rsi = _rsi(closes, self.rsi_period)
        atr = _atr(bars, self.atr_period)
        latest = bars[-1]

        prior = bars[-(self.breakout_lookback + 1):-1]
        prior_high = max(bar.high for bar in prior)
        prior_low = min(bar.low for bar in prior)
        breakout = (
            "up"
            if latest.close > prior_high
            else "down"
            if latest.close < prior_low
            else "none"
        )

        score = 0
        if latest.close > ema_fast > ema_slow:
            score += 1
        elif latest.close < ema_fast < ema_slow:
            score -= 1
        if rsi >= 55:
            score += 1
        elif rsi <= 45:
            score -= 1
        if breakout == "up":
            score += 1
        elif breakout == "down":
            score -= 1

        direction = "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral"
        recent = bars[-self.breakout_lookback:]

        return XAUFrameState(
            timeframe=timeframe,
            close=latest.close,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi14=rsi,
            atr14=atr,
            atr_pct=(atr / latest.close) * 100 if latest.close else 0.0,
            breakout=breakout,
            direction=direction,
            recent_swing_high=max(bar.high for bar in recent),
            recent_swing_low=min(bar.low for bar in recent),
            observed_at=latest.timestamp,
        )


def _ema(values: Iterable[float], period: int) -> float:
    data = list(values)
    seed = fmean(data[:period])
    multiplier = 2 / (period + 1)
    value = seed
    for item in data[period:]:
        value = (item - value) * multiplier + value
    return value


def _rsi(values: list[float], period: int) -> float:
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    window = deltas[-period:]
    gains = [max(delta, 0.0) for delta in window]
    losses = [max(-delta, 0.0) for delta in window]
    avg_gain = fmean(gains)
    avg_loss = fmean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(bars: list[XAUBar], period: int) -> float:
    ranges: list[float] = []
    for index in range(1, len(bars)):
        current = bars[index]
        previous = bars[index - 1]
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return fmean(ranges[-period:])
