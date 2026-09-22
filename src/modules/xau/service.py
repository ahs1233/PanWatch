"""XAU/USD terminal service.

GC=F is a research proxy only. It never satisfies an execution-data gate.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.market_structure import (
    aggregate_bars,
    ema_stack,
    fair_value_gaps,
    flow_proxy,
    liquidity_map,
    top_down_bias,
    volume_profile,
)
from src.modules.xau.market_context import build_market_context
from src.platform.ai.ai_client import AIClient
from src.platform.external_tools.ahmed_toolbox import AhmedToolboxClient
from src.platform.marketdata.xau_biquote import (
    BiquoteEconomicCalendarProvider,
    BiquoteXAUOHLCProvider,
)
from src.platform.marketdata.xau_models import XAUTimeframe
from src.platform.marketdata.xau_micro_reference import (
    XAUSIntradayReferenceProvider,
    sampled_spot_bars,
)
from src.platform.marketdata.xau_research_provider import YahooGoldResearchProvider
from src.platform.marketdata.xau_spot_reference import (
    CompositeXAUIndicativeSpotProvider,
    GoldPriceDevSpotReference,
    XAUSSpotReference,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

MACRO_EVENT_POLICY_VERSION = "calendar-v1+breaking-v1"

_BARS_TTL = 25.0
_SPOT_TTL = 8.0
_MICRO_TTL = 15.0
_SERIES_TTL = 15.0
_MACRO_TTL = 180.0
_CONSENSUS_TTL = 60.0
_CONTEXT_TTL = 300.0
_MARKET_CONTEXT_TTL = 120.0
_bars_cache = None
_spot_cache = None
_consensus_cache = None
_micro_cache = None
_series_cache = None
_macro_cache = None
_macro_last_good = None
_macro_refresh_task = None
_context_cache = None
_market_context_cache = None
_bars_lock = asyncio.Lock()
_spot_lock = asyncio.Lock()
_consensus_lock = asyncio.Lock()
_micro_lock = asyncio.Lock()
_series_lock = asyncio.Lock()
_macro_lock = asyncio.Lock()
_context_lock = asyncio.Lock()
_market_context_lock = asyncio.Lock()


async def get_micro_series(force: bool = False):
    global _series_cache
    now = time.monotonic()
    if not force and _series_cache and now - _series_cache[0] < _SERIES_TTL:
        return _series_cache[1]

    async with _series_lock:
        now = time.monotonic()
        if not force and _series_cache and now - _series_cache[0] < _SERIES_TTL:
            return _series_cache[1]

        provider = XAUSIntradayReferenceProvider()
        series = await asyncio.to_thread(provider.fetch, hours=12)
        _series_cache = (time.monotonic(), series)
        return series


async def get_research_bars(force: bool = False):
    global _bars_cache
    now = time.monotonic()
    if not force and _bars_cache and now - _bars_cache[0] < _BARS_TTL:
        return _bars_cache[1]

    async with _bars_lock:
        now = time.monotonic()
        if not force and _bars_cache and now - _bars_cache[0] < _BARS_TTL:
            return _bars_cache[1]

        biquote = BiquoteXAUOHLCProvider()
        yahoo = YahooGoldResearchProvider()

        async def biquote_bars(timeframe):
            try:
                return await asyncio.to_thread(
                    biquote.bars,
                    timeframe,
                    limit=1000,
                )
            except Exception:
                return []

        async def yahoo_bars(timeframe):
            try:
                return await asyncio.to_thread(yahoo.bars, timeframe)
            except Exception:
                return []

        m1, m5, m15 = await asyncio.gather(
            biquote_bars(XAUTimeframe.M1),
            biquote_bars(XAUTimeframe.M5),
            biquote_bars(XAUTimeframe.M15),
        )

        # Fail soft per timeframe. A stale primary 5m/15m series is treated
        # exactly like an unavailable one: prefer a fresher sampled XAU spot
        # structure before falling back to GC=F research.
        now_utc = datetime.now(timezone.utc)

        def fresh_enough(rows, max_age: timedelta) -> bool:
            if not rows:
                return False
            try:
                latest = max(rows, key=lambda item: item.timestamp)
                return now_utc - latest.timestamp <= max_age
            except Exception:
                return False

        m5_fresh = fresh_enough(m5, timedelta(minutes=12))
        m15_fresh = fresh_enough(m15, timedelta(minutes=35))

        sampled_5 = []
        sampled_15 = []
        if not m5_fresh or not m15_fresh:
            try:
                series = await get_micro_series(force=force)
                sampled_5 = sampled_spot_bars(series, XAUTimeframe.M5)
                sampled_15 = sampled_spot_bars(series, XAUTimeframe.M15)
            except Exception:
                sampled_5 = []
                sampled_15 = []

        if not m1:
            m1 = await yahoo_bars(XAUTimeframe.M1)

        if not m5_fresh:
            if fresh_enough(sampled_5, timedelta(minutes=12)):
                m5 = sampled_5
            else:
                m5 = await yahoo_bars(XAUTimeframe.M5)

        if not m15_fresh:
            if fresh_enough(sampled_15, timedelta(minutes=35)):
                m15 = sampled_15
            else:
                m15 = await yahoo_bars(XAUTimeframe.M15)

        data = {
            XAUTimeframe.M1: m1,
            XAUTimeframe.M5: m5,
            XAUTimeframe.M15: m15,
        }
        _bars_cache = (time.monotonic(), data)
        return data





async def get_market_context(force: bool = False) -> dict[str, Any]:
    """Higher-timeframe XAU context: 1h/4h/daily → weekly/monthly bias.

    The source is MT5-backed XAUUSD research data. Tick volume is deliberately
    labelled as a participation proxy because spot gold has no centralized
    global volume tape.
    """
    global _market_context_cache
    now = time.monotonic()
    if not force and _market_context_cache and now - _market_context_cache[0] < _MARKET_CONTEXT_TTL:
        return _market_context_cache[1]

    async with _market_context_lock:
        now = time.monotonic()
        if not force and _market_context_cache and now - _market_context_cache[0] < _MARKET_CONTEXT_TTL:
            return _market_context_cache[1]

        provider = BiquoteXAUOHLCProvider()

        async def fetch(tf: XAUTimeframe):
            return await asyncio.to_thread(
                provider.bars,
                tf,
                limit=1000,
                timeout_seconds=14.0,
            )

        h1, h4, daily = await asyncio.gather(
            fetch(XAUTimeframe.H1),
            fetch(XAUTimeframe.H4),
            fetch(XAUTimeframe.D1),
        )
        data = build_market_context(h1, h4, daily)
        data["sources"] = {
            "1h": h1[-1].source if h1 else None,
            "4h": h4[-1].source if h4 else None,
            "1d": daily[-1].source if daily else None,
        }
        data["observed_at"] = max(
            [row.timestamp for rows in (h1, h4, daily) for row in rows[-1:]],
            default=datetime.now(timezone.utc),
        ).isoformat()
        _market_context_cache = (time.monotonic(), data)
        return data

async def get_chart_series(
    timeframe: str = "5m",
    *,
    limit: int = 160,
    force: bool = False,
) -> dict[str, Any]:
    """Return deep-history OHLC with EMA 9/21/50/200/1000.

    Higher-timeframe charting is part of the same research architecture as the
    decision engine. Weekly/monthly bars are aggregated from daily XAUUSD data.
    """
    direct_map = {
        "1m": XAUTimeframe.M1,
        "5m": XAUTimeframe.M5,
        "15m": XAUTimeframe.M15,
        "30m": XAUTimeframe.M30,
        "1h": XAUTimeframe.H1,
        "4h": XAUTimeframe.H4,
        "1d": XAUTimeframe.D1,
    }
    aggregate_map = {
        "1w": XAUTimeframe.W1,
        "1mo": XAUTimeframe.MN1,
    }
    key = str(timeframe or "5m").strip().lower()
    if key not in direct_map and key not in aggregate_map:
        key = "5m"
    limit = max(30, min(int(limit), 240))

    provider = BiquoteXAUOHLCProvider()
    rows = []
    if key in aggregate_map:
        try:
            daily = await asyncio.to_thread(
                provider.bars,
                XAUTimeframe.D1,
                limit=1000,
                timeout_seconds=14.0,
            )
            rows = aggregate_bars(daily, aggregate_map[key])
        except Exception:
            yahoo = YahooGoldResearchProvider()
            try:
                rows = await asyncio.to_thread(yahoo.bars, aggregate_map[key])
            except Exception:
                rows = []
    else:
        tf = direct_map[key]
        try:
            rows = await asyncio.to_thread(
                provider.bars,
                tf,
                limit=1000,
                timeout_seconds=14.0,
            )
        except Exception:
            if tf in {XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15}:
                bars = await get_research_bars(force=force)
                rows = list(bars.get(tf) or [])
            elif tf in {XAUTimeframe.H1, XAUTimeframe.D1}:
                yahoo = YahooGoldResearchProvider()
                try:
                    rows = await asyncio.to_thread(yahoo.bars, tf)
                except Exception:
                    rows = []
            else:
                rows = []

    rows = sorted(rows, key=lambda row: row.timestamp)

    def ema_series(values: list[float], period: int) -> list[float | None]:
        out: list[float | None] = [None] * len(values)
        if len(values) < period:
            return out
        seed = sum(values[:period]) / period
        alpha = 2.0 / (period + 1.0)
        current = seed
        out[period - 1] = current
        for i in range(period, len(values)):
            current = (values[i] - current) * alpha + current
            out[i] = current
        return out

    closes = [float(row.close) for row in rows]
    ema_map = {
        period: ema_series(closes, period)
        for period in (9, 21, 50, 200, 1000)
    }
    start_index = max(0, len(rows) - limit)
    payload = []
    for i, row in enumerate(rows[start_index:], start=start_index):
        item = {
            "time": row.timestamp.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(getattr(row, "volume", 0.0) or 0.0),
            "source": str(getattr(row, "source", "") or ""),
        }
        for period in (9, 21, 50, 200, 1000):
            value = ema_map[period][i]
            item[f"ema{period}"] = round(value, 6) if value is not None else None
        payload.append(item)

    return {
        "instrument": "XAUUSD",
        "timeframe": key,
        "count": len(payload),
        "history_count": len(rows),
        "bars": payload,
        "source": payload[-1]["source"] if payload else None,
        "observed_at": payload[-1]["time"] if payload else None,
        "ema_periods": [9, 21, 50, 200, 1000],
        "ema_coverage": {
            str(period): len(rows) >= period
            for period in (9, 21, 50, 200, 1000)
        },
        "research_only": True,
    }


def _spot_fill_state(provider_health: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify paper-fill availability independently from analysis context."""
    healthy_bidask = [
        row
        for row in provider_health
        if row.get("status") == "ok"
        and row.get("has_bid_ask")
        and not row.get("is_stale")
    ]
    if healthy_bidask:
        best = min(
            healthy_bidask,
            key=lambda row: float(row.get("age_seconds") or 1e9),
        )
        return {
            "state": "ready",
            "source": best.get("source"),
            "age_seconds": best.get("age_seconds"),
        }

    closed = [
        row
        for row in provider_health
        if row.get("status") == "ok"
        and row.get("has_bid_ask")
        and str(row.get("market_state") or "").lower() == "closed"
    ]
    if closed:
        best = min(
            closed,
            key=lambda row: float(row.get("age_seconds") or 1e9),
        )
        return {
            "state": "market_closed_or_rollover",
            "source": best.get("source"),
            "age_seconds": best.get("age_seconds"),
        }

    stale_bidask = [
        row
        for row in provider_health
        if row.get("status") == "ok" and row.get("has_bid_ask")
    ]
    if stale_bidask:
        best = min(
            stale_bidask,
            key=lambda row: float(row.get("age_seconds") or 1e9),
        )
        return {
            "state": "stale_bid_ask",
            "source": best.get("source"),
            "age_seconds": best.get("age_seconds"),
        }

    return {
        "state": "unavailable",
        "source": None,
        "age_seconds": None,
    }


async def get_indicative_spot(force: bool = False) -> dict[str, Any]:
    global _spot_cache
    now = time.monotonic()
    if not force and _spot_cache and now - _spot_cache[0] < _SPOT_TTL:
        return _spot_cache[1]

    async with _spot_lock:
        now = time.monotonic()
        if not force and _spot_cache and now - _spot_cache[0] < _SPOT_TTL:
            return _spot_cache[1]

        provider = CompositeXAUIndicativeSpotProvider()
        quote, provider_health = await asyncio.to_thread(provider.fetch_with_diagnostics)
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - quote.observed_at).total_seconds(),
        )
        fill = _spot_fill_state(provider_health)
        data = {
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread": quote.spread,
            "spread_bps": quote.spread_bps,
            "observed_at": quote.observed_at.isoformat(),
            "age_seconds": age_seconds,
            "source": quote.source,
            "market_state": quote.market_state,
            "provider_quote_age_seconds": quote.provider_quote_age_seconds,
            "is_stale": bool(quote.is_stale or age_seconds > 180.0),
            "provider_health": provider_health,
            "fill_state": fill.get("state"),
            "fill_source": fill.get("source"),
            "fill_age_seconds": fill.get("age_seconds"),
            "indicative": True,
            "execution_eligible": False,
        }
        _spot_cache = (time.monotonic(), data)
        return data


async def get_spot_consensus(force: bool = False) -> dict[str, Any]:
    """Low-frequency cross-source validation for the primary indicative spot.

    This is a sensor-consistency layer only. It never upgrades any quote to
    execution-eligible and is deliberately cached longer than the fast loop.
    """
    global _consensus_cache
    now = time.monotonic()
    if not force and _consensus_cache and now - _consensus_cache[0] < _CONSENSUS_TTL:
        return _consensus_cache[1]

    async with _consensus_lock:
        now = time.monotonic()
        if not force and _consensus_cache and now - _consensus_cache[0] < _CONSENSUS_TTL:
            return _consensus_cache[1]

        providers = (
            GoldPriceDevSpotReference(),
            XAUSSpotReference(),
        )

        async def fetch_one(provider):
            try:
                quote = await asyncio.to_thread(provider.fetch)
                age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - quote.observed_at).total_seconds(),
                )
                panwatch_stale = bool(quote.is_stale) or age_seconds > 300.0
                return {
                    "source": quote.source,
                    "price": quote.price,
                    "observed_at": quote.observed_at.isoformat(),
                    "age_seconds": round(age_seconds, 3),
                    "provider_stale": bool(quote.is_stale),
                    "is_stale": panwatch_stale,
                    "freshness_policy_seconds": 300.0,
                }
            except Exception as exc:
                return {
                    "source": type(provider).__name__,
                    "error": type(exc).__name__,
                }

        rows = await asyncio.gather(*(fetch_one(provider) for provider in providers))
        usable = [
            row for row in rows
            if row.get("price") and not row.get("is_stale") and not row.get("error")
        ]
        prices = sorted(float(row["price"]) for row in usable)
        median = None
        if prices:
            middle = len(prices) // 2
            median = (
                prices[middle]
                if len(prices) % 2
                else (prices[middle - 1] + prices[middle]) / 2.0
            )

        data = {
            "source_count": len(rows),
            "usable_count": len(usable),
            "reference_median": median,
            "references": rows,
            "execution_eligible": False,
        }
        _consensus_cache = (time.monotonic(), data)
        return data


def _ema_values(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    period = max(1, min(period, len(values)))
    seed = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1.0)
    current = seed
    for value in values[period:]:
        current = (value - current) * multiplier + current
    return current


def _return_pct(points, minutes: int) -> float | None:
    if len(points) < 2:
        return None
    latest = points[-1]
    target = latest.timestamp.timestamp() - minutes * 60
    base = points[0]
    for point in points:
        if point.timestamp.timestamp() <= target:
            base = point
        else:
            break
    if base.price <= 0:
        return None
    return ((latest.price - base.price) / base.price) * 100.0


async def get_micro_context(force: bool = False) -> dict[str, Any]:
    global _micro_cache
    now = time.monotonic()
    if not force and _micro_cache and now - _micro_cache[0] < _MICRO_TTL:
        return _micro_cache[1]

    async with _micro_lock:
        now = time.monotonic()
        if not force and _micro_cache and now - _micro_cache[0] < _MICRO_TTL:
            return _micro_cache[1]

        series = await get_micro_series(force=force)
        prices = [point.price for point in series.points]
        latest = series.points[-1]

        fast = _ema_values(prices, 5)
        slow = _ema_values(prices, 10)
        ret10 = _return_pct(series.points, 10)
        ret30 = _return_pct(series.points, 30)

        direction = "neutral"
        if len(prices) >= 10 and ret10 is not None:
            if latest.price > fast > slow and ret10 > 0:
                direction = "bullish"
            elif latest.price < fast < slow and ret10 < 0:
                direction = "bearish"

        recent = prices[-15:] if len(prices) >= 15 else prices
        status = "ready" if len(prices) >= 10 and not series.is_stale else "blocked"
        data = {
            "status": status,
            "direction": direction,
            "price": latest.price,
            "ema_fast": fast,
            "ema_slow": slow,
            "return_10m_pct": ret10,
            "return_30m_pct": ret30,
            "recent_high": max(recent) if recent else None,
            "recent_low": min(recent) if recent else None,
            "point_count": len(prices),
            "observed_at": series.observed_at.isoformat(),
            "last_point_at": latest.timestamp.isoformat(),
            "age_seconds": series.age_seconds,
            "coverage_seconds": series.coverage_seconds,
            "source": series.source,
            "is_stale": series.is_stale,
            "indicative": True,
            "execution_eligible": False,
        }
        _micro_cache = (time.monotonic(), data)
        return data


async def get_xau_snapshot(force: bool = False) -> dict[str, Any]:
    bars_task = asyncio.create_task(get_research_bars(force=force))
    spot_task = asyncio.create_task(get_indicative_spot(force=force))
    micro_task = asyncio.create_task(get_micro_context(force=force))
    consensus_task = asyncio.create_task(get_spot_consensus(force=force))
    market_context_task = asyncio.create_task(get_market_context(force=force))

    bars = await bars_task
    spot = None
    spot_error = None
    try:
        spot = await spot_task
    except Exception as exc:
        spot_error = type(exc).__name__

    micro = None
    micro_error = None
    try:
        micro = await micro_task
    except Exception as exc:
        micro_error = type(exc).__name__

    consensus = None
    consensus_error = None
    try:
        consensus = await consensus_task
    except Exception as exc:
        consensus_error = type(exc).__name__

    market_context = None
    market_context_error = None
    try:
        market_context = await market_context_task
    except Exception as exc:
        market_context_error = type(exc).__name__
        logger.warning("XAU higher-timeframe context unavailable: %s", market_context_error)

    assessment = XAUIntradayEngine(require_execution_data=False).analyze(
        bars,
        now=datetime.now(timezone.utc),
    )

    frames = {}
    for name, state in assessment.frame_states.items():
        frame_bars = bars.get(state.timeframe) or []
        frame_source = frame_bars[-1].source if frame_bars else ""
        frames[name] = {
            "timeframe": state.timeframe.value,
            "source": frame_source,
            "close": state.close,
            "ema_fast": state.ema_fast,
            "ema_slow": state.ema_slow,
            "rsi14": state.rsi14,
            "atr14": state.atr14,
            "atr_pct": state.atr_pct,
            "breakout": state.breakout,
            "direction": state.direction,
            "recent_swing_high": state.recent_swing_high,
            "recent_swing_low": state.recent_swing_low,
            "observed_at": state.observed_at.isoformat(),
        }

    latest = frames.get("1m") or frames.get("5m") or frames.get("15m") or {}
    m1 = bars.get(XAUTimeframe.M1) or []
    change_pct = None
    if len(m1) >= 2 and m1[-2].close:
        change_pct = ((m1[-1].close - m1[-2].close) / m1[-2].close) * 100.0

    raw_block_reasons = list(assessment.block_reasons)
    one_minute_proxy_reasons = {
        "stale_1m_bars",
        "insufficient_1m_bars",
    }
    remaining_reasons = [
        reason for reason in raw_block_reasons
        if reason not in one_minute_proxy_reasons
    ]
    micro_can_replace_1m = bool(
        micro
        and micro.get("status") == "ready"
        and not micro.get("is_stale")
        and any(reason in one_minute_proxy_reasons for reason in raw_block_reasons)
    )

    terminal_blocked = assessment.blocked
    terminal_status = assessment.status
    terminal_candidate = assessment.candidate
    frame_sources = {
        name: str(frame.get("source") or "")
        for name, frame in frames.items()
    }
    all_biquote = all(
        frame_sources.get(name, "").startswith("biquote.io:")
        for name in ("1m", "5m", "15m")
    )
    technical_mode = (
        "biquote_mt5_1m_5m_15m"
        if all_biquote
        else "mixed_research_fallback_1m_5m_15m"
    )

    if micro_can_replace_1m:
        terminal_blocked = bool(remaining_reasons)
        terminal_status = "blocked" if terminal_blocked else "ready_with_spot_micro"
        technical_mode = "spot_micro_plus_spot_5m_15m"
        terminal_candidate = "none"

        five = frames.get("5m")
        fifteen = frames.get("15m")
        micro_direction = str(micro.get("direction") or "neutral")
        if not terminal_blocked and five and fifteen:
            if (
                five.get("direction") == "bullish"
                and fifteen.get("direction") == "bullish"
                and micro_direction != "bearish"
            ):
                terminal_candidate = "long_setup"
            elif (
                five.get("direction") == "bearish"
                and fifteen.get("direction") == "bearish"
                and micro_direction != "bullish"
            ):
                terminal_candidate = "short_setup"

    if micro_can_replace_1m:
        directions = [
            str(micro.get("direction") or "neutral"),
            str((frames.get("5m") or {}).get("direction") or "neutral"),
            str((frames.get("15m") or {}).get("direction") or "neutral"),
        ]
    else:
        directions = [item.get("direction") for item in frames.values()]

    bullish = sum(1 for item in directions if item == "bullish")
    bearish = sum(1 for item in directions if item == "bearish")
    alignment = "bullish" if bullish >= 2 else "bearish" if bearish >= 2 else "mixed"

    proxy_price = (frames.get("1m") or {}).get("close")
    basis = (
        (spot["price"] - proxy_price)
        if spot and proxy_price is not None
        else None
    )
    basis_bps = (
        (basis / spot["price"]) * 10_000.0
        if basis is not None and spot and spot["price"]
        else None
    )

    consensus_delta_bps = None
    if spot and consensus and consensus.get("reference_median"):
        reference_median = float(consensus["reference_median"])
        if reference_median > 0:
            consensus_delta_bps = (
                (float(spot["price"]) - reference_median) / reference_median
            ) * 10_000.0
        consensus["primary_delta_bps"] = consensus_delta_bps

    # Analytical reference is intentionally separate from paper/live fill data.
    # If the primary indicative quote is stale, a fresh micro tape can still
    # support regime/perception analysis, but never execution.
    analysis_reference = None
    if spot and not bool(spot.get("is_stale")) and spot.get("price") is not None:
        analysis_reference = {
            "price": spot.get("price"),
            "source": spot.get("source"),
            "observed_at": spot.get("observed_at"),
            "age_seconds": spot.get("age_seconds"),
            "is_stale": False,
            "kind": "indicative_spot",
            "execution_eligible": False,
        }
    elif (
        micro
        and micro.get("status") == "ready"
        and not bool(micro.get("is_stale"))
        and micro.get("price") is not None
    ):
        analysis_reference = {
            "price": micro.get("price"),
            "source": micro.get("source"),
            "observed_at": micro.get("last_point_at") or micro.get("observed_at"),
            "age_seconds": micro.get("age_seconds"),
            "is_stale": False,
            "kind": "micro_fallback",
            "execution_eligible": False,
        }
    elif proxy_price is not None:
        proxy_observed_at = latest.get("observed_at")
        proxy_age_seconds = None
        try:
            proxy_dt = _event_timestamp(proxy_observed_at)
            if proxy_dt is not None:
                proxy_age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - proxy_dt).total_seconds(),
                )
        except Exception:
            proxy_age_seconds = None
        analysis_reference = {
            "price": proxy_price,
            "source": latest.get("source"),
            "observed_at": proxy_observed_at,
            "age_seconds": proxy_age_seconds,
            "is_stale": bool(terminal_blocked),
            "kind": "structural_proxy",
            "execution_eligible": False,
        }

    if analysis_reference and consensus and consensus.get("reference_median"):
        try:
            ref = float(consensus["reference_median"])
            price = float(analysis_reference["price"])
            if ref > 0:
                analysis_reference["consensus_delta_bps"] = (
                    (price - ref) / ref
                ) * 10_000.0
        except (TypeError, ValueError):
            pass

    warnings = list(assessment.warnings)
    if not all_biquote:
        warnings.append("biquote_ohlc_fallback_active")
    if micro_can_replace_1m:
        warnings.append("gc_1m_stale_replaced_by_live_spot_micro")
    elif micro_error:
        warnings.append("spot_micro_unavailable")
    elif micro and micro.get("is_stale"):
        warnings.append("spot_micro_stale")
    if spot_error:
        warnings.append("indicative_spot_unavailable")
    elif spot and spot.get("is_stale"):
        warnings.append("indicative_spot_stale")
    if consensus_error:
        warnings.append("spot_consensus_unavailable")
    elif consensus and int(consensus.get("usable_count") or 0) == 0:
        warnings.append("spot_consensus_no_fresh_reference")
    elif consensus_delta_bps is not None and abs(consensus_delta_bps) >= 8.0:
        warnings.append("spot_consensus_disagreement")

    return {
        "instrument": "XAUUSD",
        "name": "Gold / U.S. Dollar",
        "research_proxy": "GC=F",
        "research_source": "Biquote MT5 OHLC primary; XAUS/Yahoo research fallbacks",
        "primary_intraday_source": "biquote.io:MT5-ohlc",
        "research_only": True,
        "execution_feed_connected": False,
        "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
        "price": proxy_price,
        "indicative_spot": spot,
        "indicative_spot_error": spot_error,
        "analysis_reference": analysis_reference,
        "spot_consensus": consensus,
        "spot_consensus_error": consensus_error,
        "micro": micro,
        "micro_error": micro_error,
        "market_context": market_context,
        "market_context_error": market_context_error,
        "technical_mode": technical_mode,
        "spot_minus_proxy": basis,
        "spot_minus_proxy_bps": basis_bps,
        "change_pct_1m": change_pct,
        "observed_at": (
            (spot or {}).get("observed_at")
            or (micro or {}).get("observed_at")
            or latest.get("observed_at")
        ),
        "status": terminal_status,
        "candidate": terminal_candidate,
        "blocked": terminal_blocked,
        "block_reasons": remaining_reasons if micro_can_replace_1m else raw_block_reasons,
        "raw_proxy_block_reasons": raw_block_reasons,
        "warnings": warnings,
        "alignment": alignment,
        "atr_reference": assessment.atr_reference,
        "swing_high_reference": assessment.swing_high_reference,
        "swing_low_reference": assessment.swing_low_reference,
        "frames": frames,
        "disclaimer": (
            "The live spot reference is indicative and GC=F is a delayed research proxy. "
            "Neither is a broker execution quote. Live entry, stop-loss and take-profit "
            "automation remains locked until a tradable venue-specific bid/ask feed is connected."
        ),
    }


def macro_entry_readiness(macro: dict[str, Any]) -> tuple[bool, str]:
    """Unknown, failed or expired research is not a neutral market assessment."""
    if macro.get("cache_stale") or macro.get("refresh_pending"):
        return False, "macro_refresh_pending"
    if not all(macro.get(key) is True for key in ("calendar_ok", "search_ok", "synthesis_ok")):
        return False, "macro_inputs_unavailable"
    try:
        observed = datetime.fromisoformat(str(macro.get("observed_at") or "").replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        if not 0 <= age <= _MACRO_TTL:
            return False, "macro_expired"
    except (TypeError, ValueError):
        return False, "macro_timestamp_unavailable"
    return True, "ready"


def build_decision_fusion(
    technical: dict[str, Any],
    macro: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Fuse technical, macro and cognitive layers into one inspectable state."""

    candidate = str(technical.get("candidate") or "none")
    technical_blocked = bool(technical.get("blocked"))
    event_risk = bool(macro.get("event_risk"))
    macro_ready, macro_readiness_reason = macro_entry_readiness(macro)
    try:
        macro_bias = max(-1, min(1, int(macro.get("bias", 0))))
    except (TypeError, ValueError):
        macro_bias = 0

    setup_direction = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    if setup_direction == 0:
        macro_relation = "not_applicable"
    elif not macro_ready:
        macro_relation = "unknown"
    elif macro_bias == 0:
        macro_relation = "neutral"
    elif macro_bias == setup_direction:
        macro_relation = "support"
    else:
        macro_relation = "conflict"

    settings = Settings()
    threshold = (
        float(min_confidence)
        if min_confidence is not None
        else float(settings.xau_cognition_min_confidence)
    )
    cognition = build_cognitive_state(
        technical,
        macro,
        memory=memory,
        min_confidence=threshold,
    )
    cognition_active = bool(
        settings.xau_cognition_enabled
        and technical.get("frames")
        and technical.get("indicative_spot")
    )
    meta_decision = str(
        (cognition.get("meta_controller") or {}).get("decision") or "observe"
    )

    reasons: list[str] = []
    if technical_blocked:
        base_state = "data_gate"
        reasons.extend(str(x) for x in technical.get("block_reasons") or [])
    elif event_risk:
        base_state = "event_gate"
        reasons.append("high_impact_macro_event")
    elif candidate == "none":
        base_state = "no_setup"
        reasons.append("no_aligned_technical_setup")
    elif not macro_ready:
        base_state = "macro_unavailable"
        reasons.append(macro_readiness_reason)
    elif macro_relation == "conflict":
        base_state = "setup_macro_conflict"
        reasons.append("macro_bias_conflicts_with_technical_setup")
    elif macro_relation == "support":
        base_state = "setup_macro_support"
        reasons.append("macro_bias_supports_technical_setup")
    else:
        base_state = "setup_macro_neutral"
        reasons.append("macro_bias_is_neutral_or_mixed")

    state = base_state
    edge = cognition.get("directional_edge") or {}
    edge_direction = str(edge.get("direction") or "neutral")
    edge_strength = float(edge.get("strength") or 0.0)
    if (
        cognition_active
        and base_state == "no_setup"
        and edge_direction in {"bullish", "bearish"}
        and edge_strength >= 0.24
        and not technical_blocked
        and not event_risk
    ):
        state = "directional_bias_wait"
        reasons.append("directional_edge_without_entry_trigger")
    elif cognition_active and base_state in {"setup_macro_support", "setup_macro_neutral"}:
        if meta_decision == "veto":
            state = "cognitive_veto"
            reasons.append("adversarial_or_quality_veto")
        elif meta_decision == "wait":
            state = "cognitive_wait"
            reasons.append("execution_timing_wait")
        elif meta_decision != "eligible":
            state = "cognitive_observe"
            reasons.append("meta_controller_not_eligible")

    research_ready = base_state in {
        "setup_macro_support",
        "setup_macro_neutral",
        "setup_macro_conflict",
    } or state == "directional_bias_wait"

    return {
        "state": state,
        "base_state": base_state,
        "technical_candidate": candidate,
        "technical_status": technical.get("status"),
        "technical_mode": technical.get("technical_mode"),
        "macro_bias": macro_bias,
        "macro_bias_label": macro.get("bias_label"),
        "macro_confidence": macro.get("confidence"),
        "macro_relation": macro_relation,
        "macro_ready": macro_ready,
        "macro_readiness_reason": macro_readiness_reason,
        "event_risk": event_risk,
        "event_kind": macro.get("event_kind"),
        "event_name": macro.get("event_name"),
        "event_time_utc": macro.get("event_time_utc"),
        "event_age_minutes": macro.get("event_age_minutes"),
        "event_confidence": macro.get("event_confidence"),
        "event_validation": macro.get("event_validation"),
        "event_policy_version": macro.get("event_policy_version"),
        "event_source_url": macro.get("event_source_url"),
        "research_ready": research_ready,
        "cognition": cognition,
        "cognition_active": cognition_active,
        "regime": (cognition.get("regime") or {}).get("label"),
        "cognitive_confidence": (
            (cognition.get("confidence") or {}).get("calibrated_confidence")
        ),
        "meta_decision": meta_decision,
        "paper_entry_allowed": (
            bool((cognition.get("meta_controller") or {}).get("paper_entry_allowed"))
            if cognition_active
            else base_state in {"setup_macro_support", "setup_macro_neutral"}
        )
        and base_state in {"setup_macro_support", "setup_macro_neutral"},
        "execution_allowed": False,
        "execution_status": technical.get(
            "execution_status",
            "LOCKED_NO_TRADABLE_SPOT_FEED",
        ),
        "reasons": list(dict.fromkeys(reasons)),
    }

def _tool_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _event_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validated_event_gate(
    parsed: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    claimed = bool(parsed.get("event_risk", False))
    kind = str(parsed.get("event_kind") or "none").strip().lower()
    name = str(parsed.get("event_name") or "").strip()

    try:
        event_confidence = max(
            0.0,
            min(1.0, float(parsed.get("event_confidence", 0.0))),
        )
    except (TypeError, ValueError):
        event_confidence = 0.0

    event_time = _event_timestamp(parsed.get("event_time_utc"))
    try:
        event_age_minutes = float(parsed.get("event_age_minutes"))
    except (TypeError, ValueError):
        event_age_minutes = None

    active = False
    reason = "not_claimed"

    if claimed and not name:
        reason = "missing_event_name"
    elif claimed and kind == "scheduled":
        if event_time is None:
            reason = "missing_or_invalid_event_time"
        elif event_confidence < 0.65:
            reason = "event_confidence_too_low"
        else:
            minutes_until = (event_time - current).total_seconds() / 60.0
            active = -30.0 <= minutes_until <= 90.0
            reason = "scheduled_event_window" if active else "scheduled_event_outside_window"
    elif claimed and kind == "breaking":
        if event_age_minutes is None:
            reason = "missing_event_age"
        elif event_confidence < 0.70:
            reason = "event_confidence_too_low"
        else:
            active = 0.0 <= event_age_minutes <= 30.0
            reason = "breaking_event_window" if active else "breaking_event_too_old"
    elif claimed:
        reason = "unsupported_event_kind"

    return {
        "event_risk": active,
        "event_kind": kind if kind in {"scheduled", "breaking"} else "none",
        "event_name": name or None,
        "event_time_utc": event_time.isoformat() if event_time else None,
        "event_age_minutes": event_age_minutes,
        "event_confidence": event_confidence,
        "event_validation": reason,
        "event_policy_version": MACRO_EVENT_POLICY_VERSION,
    }


def _resolve_event_gate(
    ai_event_gate: dict[str, Any],
    calendar_event: dict[str, Any] | None,
) -> dict[str, Any]:
    if calendar_event:
        resolved = dict(calendar_event)
        resolved["event_policy_version"] = MACRO_EVENT_POLICY_VERSION
        return resolved

    resolved = dict(ai_event_gate)
    resolved["event_policy_version"] = MACRO_EVENT_POLICY_VERSION
    # Scheduled events must come from the structured calendar, not the LLM.
    if resolved.get("event_kind") == "scheduled":
        resolved["event_risk"] = False
        resolved["event_validation"] = "scheduled_event_requires_calendar"
    return resolved


def _parse_json(value: str) -> dict[str, Any]:
    value = str(value or "").strip()
    # Providers commonly wrap otherwise-valid JSON in Markdown fences.
    # Accept both backtick and tilde fences while keeping the parser strict.
    value = re.sub(r"^(?:```|~~~)(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*(?:```|~~~)\s*$", "", value)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", value)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, SyntaxError):
                return {}


async def _refresh_macro_context(force: bool = False) -> dict[str, Any]:
    global _macro_cache, _macro_last_good
    now = time.monotonic()
    if not force and _macro_cache and now - _macro_cache[0] < _MACRO_TTL:
        return _macro_cache[1]

    async with _macro_lock:
        now = time.monotonic()
        if not force and _macro_cache and now - _macro_cache[0] < _MACRO_TTL:
            return _macro_cache[1]

        settings = Settings()
        macro_now = datetime.now(timezone.utc)
        today = macro_now.strftime("%Y-%m-%d")

        calendar_event = None
        calendar_error = None
        try:
            calendar_event = await asyncio.wait_for(
                asyncio.to_thread(
                    BiquoteEconomicCalendarProvider().active_usd_event,
                    now=macro_now,
                    before_minutes=90,
                    after_minutes=30,
                ),
                timeout=15,
            )
        except Exception as exc:
            calendar_error = type(exc).__name__
        query = (
            "gold XAUUSD latest macro drivers " + today +
            " Federal Reserve US dollar DXY Treasury yields inflation "
            "economic data central banks geopolitics"
        )

        raw = ""
        search_error = None
        try:
            toolbox = AhmedToolboxClient(
                settings.ahmed_toolbox_url,
                token=settings.ahmed_toolbox_token,
                timeout_seconds=max(10.0, settings.ahmed_toolbox_timeout_seconds),
            )
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    toolbox.call_tool,
                    "reach_web_search",
                    {"query": query, "num_results": 7},
                ),
                timeout=55,
            )
            if result.get("isError"):
                search_error = _tool_text(result)[:500]
            else:
                raw = _tool_text(result)
        except Exception as exc:
            search_error = type(exc).__name__

        data = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "bias": 0,
            "bias_label": "neutral",
            "confidence": 0.0,
            "event_risk": False,
            "event_kind": "none",
            "event_name": None,
            "event_time_utc": None,
            "event_age_minutes": None,
            "event_confidence": 0.0,
            "event_validation": "not_claimed",
            "event_policy_version": MACRO_EVENT_POLICY_VERSION,
            "event_source_url": None,
            "calendar_ok": calendar_error is None,
            "calendar_error": calendar_error,
            "summary": "Macro research unavailable.",
            "drivers": [],
            "search_ok": bool(raw),
            "search_error": search_error,
            "synthesis_ok": False,
        }

        ai_event_gate = _validated_event_gate({}, now=macro_now)

        answer = ""
        if raw and settings.ai_api_key:
            try:
                ai = AIClient(
                    base_url=settings.ai_base_url,
                    api_key=settings.ai_api_key,
                    model=settings.ai_model,
                )
                prompt = (
                    "Using ONLY the evidence below, synthesize the current macro context for gold. "
                    "Return compact JSON only with keys bias, confidence, event_risk, event_kind, "
                    "event_name, event_time_utc, event_age_minutes, event_confidence, summary, drivers. "
                    "bias is -1, 0, or 1; confidence values are 0..1. "
                    "event_kind is breaking or none. Scheduled-event gating is handled separately. "
                    "Only set event_risk=true for an explicitly supported market-moving breaking shock "
                    "from the last 30 minutes. Never invent facts, prices, dates, or times. "
                    "If evidence conflicts or is insufficient, bias=0. Keep summary under 50 words and "
                    "drivers to at most three short factual bullets. Do NOT quote or infer the current gold spot "
                    "price from web material; live price comes from the market-data layer and may have moved. "
                    "Describe macro forces only, and avoid presenting stale article prices as current.\n\n"
                    + raw[:3000]
                )
                answer = await asyncio.wait_for(
                    ai.chat_multi(
                        [
                            {
                                "role": "system",
                                "content": "You are a conservative gold macro parser. Output strict JSON only.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                    ),
                    timeout=60,
                )
                parsed = _parse_json(answer)
                if not parsed:
                    raise ValueError("invalid_macro_synthesis")

                summary_value = (
                    parsed.get("summary")
                    or parsed.get("macro_summary")
                    or parsed.get("analysis")
                    or parsed.get("context")
                    or ""
                )
                drivers_value = parsed.get("drivers")
                if not isinstance(drivers_value, list):
                    drivers_value = parsed.get("key_drivers")
                if not isinstance(drivers_value, list):
                    drivers_value = parsed.get("factors")
                drivers = drivers_value if isinstance(drivers_value, list) else []
                if not isinstance(summary_value, str) or not summary_value.strip():
                    if drivers:
                        summary_value = "; ".join(str(item).strip() for item in drivers[:2] if str(item).strip())
                    else:
                        raise ValueError("invalid_macro_synthesis")

                raw_bias = parsed.get("bias", parsed.get("direction", 0))
                if isinstance(raw_bias, str):
                    normalized = raw_bias.strip().lower()
                    label_map = {"bullish": 1, "positive": 1, "bearish": -1, "negative": -1, "neutral": 0, "mixed": 0}
                    if normalized in label_map:
                        bias = label_map[normalized]
                    else:
                        try:
                            bias = int(float(normalized))
                        except ValueError as exc:
                            raise ValueError("invalid_macro_bias") from exc
                else:
                    try:
                        bias = int(raw_bias)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("invalid_macro_bias") from exc
                if bias not in (-1, 0, 1):
                    raise ValueError("invalid_macro_bias")
                try:
                    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                ai_event_gate = _validated_event_gate(
                    parsed,
                    now=macro_now,
                )
                data.update({
                    "synthesis_ok": True,
                    "bias": bias,
                    "bias_label": "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral",
                    "confidence": confidence,
                    "summary": summary_value.strip() or data["summary"],
                    "drivers": [str(item).strip() for item in drivers[:3] if str(item).strip()],
                })
                _macro_last_good = dict(data)
                logger.info(
                    "XAU macro synthesis ok bias=%s confidence=%.2f drivers=%s",
                    bias,
                    confidence,
                    len(data["drivers"]),
                )
            except Exception as exc:
                preview: list[str] = []
                seen: set[str] = set()
                for line in raw.splitlines():
                    cleaned = re.sub(r"^[\\s#>*\\-\\d.]+", "", line).strip()
                    if not cleaned or cleaned.startswith(("http://", "https://")) or len(cleaned) < 24:
                        continue
                    cleaned = re.sub(r"\\s+", " ", cleaned)[:220]
                    key = cleaned.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    preview.append(cleaned)
                    if len(preview) >= 4:
                        break
                data["summary"] = (
                    "Macro evidence was collected, but model synthesis is temporarily unavailable. "
                    "No directional macro bias is being asserted until synthesis recovers."
                )
                data["drivers"] = preview
                data["synthesis_error"] = type(exc).__name__
                if _macro_last_good:
                    last_time = _event_timestamp(_macro_last_good.get("observed_at"))
                    fallback_age = (
                        max(0.0, (datetime.now(timezone.utc) - last_time).total_seconds())
                        if last_time else None
                    )
                    decay = (
                        max(0.20, 1.0 - fallback_age / 1800.0)
                        if fallback_age is not None else 0.20
                    )
                    data["bias"] = int(_macro_last_good.get("bias", 0) or 0)
                    data["bias_label"] = str(_macro_last_good.get("bias_label") or "neutral")
                    data["confidence"] = round(
                        float(_macro_last_good.get("confidence", 0.0) or 0.0) * decay,
                        4,
                    )
                    data["summary"] = (
                        "Last successful macro synthesis retained as decaying context: "
                        + str(_macro_last_good.get("summary") or "")
                    ).strip()
                    data["drivers"] = list(_macro_last_good.get("drivers") or [])[:3]
                    data["observed_at"] = _macro_last_good.get("observed_at") or data["observed_at"]
                    data["fallback_used"] = True
                    data["fallback_age_seconds"] = fallback_age
                logger.warning(
                    "XAU macro synthesis degraded error=%s evidence_chars=%s answer_preview=%r",
                    type(exc).__name__,
                    len(raw),
                    answer[:800],
                )

        data.update(_resolve_event_gate(ai_event_gate, calendar_event))

        data["cache_stale"] = False
        data["refresh_pending"] = False
        _macro_cache = (time.monotonic(), data)
        return data


def _neutral_macro_context() -> dict[str, Any]:
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "query": "",
        "bias": 0,
        "bias_label": "neutral",
        "confidence": 0.0,
        "event_risk": False,
        "event_kind": "none",
        "event_name": None,
        "event_time_utc": None,
        "event_age_minutes": None,
        "event_confidence": 0.0,
        "event_validation": "macro_refresh_pending",
        "event_policy_version": MACRO_EVENT_POLICY_VERSION,
        "event_source_url": None,
        "calendar_ok": False,
        "calendar_error": None,
        "summary": "Macro context is refreshing in the background.",
        "drivers": [],
        "search_ok": False,
        "search_error": None,
        "cache_stale": True,
        "refresh_pending": True,
    }


def _consume_macro_refresh_result(task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("XAU macro background refresh failed: %s", type(exc).__name__)


async def get_macro_context(force: bool = False) -> dict[str, Any]:
    """Return macro context without blocking the fast/paper path on cold research.

    Normal callers get a fresh cached snapshot immediately. When the cache is
    stale or absent, a background refresh is scheduled and the latest cached
    snapshot (or a neutral bootstrap) is returned. Explicit force=True is the
    only mode that waits for the expensive calendar/search/AI refresh.
    """
    global _macro_refresh_task

    now = time.monotonic()
    if _macro_cache and now - _macro_cache[0] < _MACRO_TTL:
        data = dict(_macro_cache[1])
        data["cache_stale"] = False
        data["refresh_pending"] = bool(
            _macro_refresh_task and not _macro_refresh_task.done()
        )
        return data

    if force:
        if _macro_refresh_task and not _macro_refresh_task.done():
            return await _macro_refresh_task
        return await _refresh_macro_context(force=True)

    if _macro_refresh_task is None or _macro_refresh_task.done():
        _macro_refresh_task = asyncio.create_task(
            _refresh_macro_context(force=True),
            name="xau_macro_background_refresh",
        )
        _macro_refresh_task.add_done_callback(_consume_macro_refresh_result)

    if _macro_cache:
        data = dict(_macro_cache[1])
        data["cache_stale"] = True
        data["refresh_pending"] = True
        return data

    return _neutral_macro_context()
