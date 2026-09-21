"""XAU/USD terminal service.

GC=F is a research proxy only. It never satisfies an execution-data gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.platform.ai.ai_client import AIClient
from src.platform.external_tools.ahmed_toolbox import AhmedToolboxClient
from src.platform.marketdata.xau_models import XAUTimeframe
from src.platform.marketdata.xau_micro_reference import XAUSIntradayReferenceProvider
from src.platform.marketdata.xau_research_provider import YahooGoldResearchProvider
from src.platform.marketdata.xau_spot_reference import CompositeXAUIndicativeSpotProvider
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

_BARS_TTL = 45.0
_SPOT_TTL = 55.0
_MICRO_TTL = 55.0
_MACRO_TTL = 300.0
_bars_cache = None
_spot_cache = None
_micro_cache = None
_macro_cache = None
_bars_lock = asyncio.Lock()
_spot_lock = asyncio.Lock()
_micro_lock = asyncio.Lock()
_macro_lock = asyncio.Lock()


async def get_research_bars(force: bool = False):
    global _bars_cache
    now = time.monotonic()
    if not force and _bars_cache and now - _bars_cache[0] < _BARS_TTL:
        return _bars_cache[1]

    async with _bars_lock:
        now = time.monotonic()
        if not force and _bars_cache and now - _bars_cache[0] < _BARS_TTL:
            return _bars_cache[1]

        provider = YahooGoldResearchProvider()

        async def load(timeframe):
            return await asyncio.to_thread(provider.bars, timeframe)

        m1, m5, m15 = await asyncio.gather(
            load(XAUTimeframe.M1),
            load(XAUTimeframe.M5),
            load(XAUTimeframe.M15),
        )
        data = {
            XAUTimeframe.M1: m1,
            XAUTimeframe.M5: m5,
            XAUTimeframe.M15: m15,
        }
        _bars_cache = (time.monotonic(), data)
        return data


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
        quote = await asyncio.to_thread(provider.fetch)
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - quote.observed_at).total_seconds(),
        )
        data = {
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread": quote.spread,
            "spread_bps": quote.spread_bps,
            "observed_at": quote.observed_at.isoformat(),
            "age_seconds": age_seconds,
            "source": quote.source,
            "is_stale": quote.is_stale,
            "indicative": True,
            "execution_eligible": False,
        }
        _spot_cache = (time.monotonic(), data)
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

        provider = XAUSIntradayReferenceProvider()
        series = await asyncio.to_thread(provider.fetch, hours=2)
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

    assessment = XAUIntradayEngine(require_execution_data=False).analyze(
        bars,
        now=datetime.now(timezone.utc),
    )

    frames = {}
    for name, state in assessment.frame_states.items():
        frames[name] = {
            "timeframe": state.timeframe.value,
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
    technical_mode = "gc_proxy_1m_5m_15m"

    if micro_can_replace_1m:
        terminal_blocked = bool(remaining_reasons)
        terminal_status = "blocked" if terminal_blocked else "ready_with_spot_micro"
        technical_mode = "spot_micro_plus_gc_5m_15m"
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

    proxy_price = latest.get("close")
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

    warnings = list(assessment.warnings)
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

    return {
        "instrument": "XAUUSD",
        "name": "Gold / U.S. Dollar",
        "research_proxy": "GC=F",
        "research_source": "Yahoo Finance via yfinance",
        "research_only": True,
        "execution_feed_connected": False,
        "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
        "price": proxy_price,
        "indicative_spot": spot,
        "indicative_spot_error": spot_error,
        "micro": micro,
        "micro_error": micro_error,
        "technical_mode": technical_mode,
        "spot_minus_proxy": basis,
        "spot_minus_proxy_bps": basis_bps,
        "change_pct_1m": change_pct,
        "observed_at": latest.get("observed_at"),
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


def _tool_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _parse_json(value: str) -> dict[str, Any]:
    value = value.strip()
    value = re.sub(r"^~~~(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*~~~$", "", value)
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
            return {}


async def get_macro_context(force: bool = False) -> dict[str, Any]:
    global _macro_cache
    now = time.monotonic()
    if not force and _macro_cache and now - _macro_cache[0] < _MACRO_TTL:
        return _macro_cache[1]

    async with _macro_lock:
        now = time.monotonic()
        if not force and _macro_cache and now - _macro_cache[0] < _MACRO_TTL:
            return _macro_cache[1]

        settings = Settings()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            "summary": "Macro research unavailable.",
            "drivers": [],
            "search_ok": bool(raw),
            "search_error": search_error,
        }

        if raw and settings.ai_api_key:
            try:
                ai = AIClient(
                    base_url=settings.ai_base_url,
                    api_key=settings.ai_api_key,
                    model=settings.ai_model,
                )
                prompt = (
                    "Using ONLY the search material below, summarize the current macro context for gold. "
                    "Return JSON only with keys bias, confidence, event_risk, summary, drivers. "
                    "bias must be -1, 0, or 1. confidence must be 0..1. "
                    "drivers must contain at most five short factual bullets. "
                    "Do not invent facts, prices, or dates. If evidence conflicts, use bias 0.\n\n"
                    + raw[:24000]
                )
                answer = await asyncio.wait_for(
                    ai.chat(
                        "You are a conservative macro research parser. Output strict JSON only.",
                        prompt,
                        temperature=0.1,
                    ),
                    timeout=35,
                )
                parsed = _parse_json(answer)
                try:
                    bias = max(-1, min(1, int(parsed.get("bias", 0))))
                except (TypeError, ValueError):
                    bias = 0
                try:
                    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                drivers = parsed.get("drivers") if isinstance(parsed.get("drivers"), list) else []
                data.update({
                    "bias": bias,
                    "bias_label": "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral",
                    "confidence": confidence,
                    "event_risk": bool(parsed.get("event_risk", False)),
                    "summary": str(parsed.get("summary") or "").strip() or data["summary"],
                    "drivers": [str(item).strip() for item in drivers[:5] if str(item).strip()],
                })
            except Exception as exc:
                data["summary"] = "Web research succeeded, but macro synthesis failed."
                data["synthesis_error"] = type(exc).__name__

        _macro_cache = (time.monotonic(), data)
        return data
