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
from src.modules.xau.cognition import build_cognitive_state
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
from src.platform.marketdata.xau_spot_reference import CompositeXAUIndicativeSpotProvider
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

MACRO_EVENT_POLICY_VERSION = "calendar-v1+breaking-v1"

_BARS_TTL = 25.0
_SPOT_TTL = 8.0
_MICRO_TTL = 15.0
_SERIES_TTL = 15.0
_MACRO_TTL = 180.0
_bars_cache = None
_spot_cache = None
_micro_cache = None
_series_cache = None
_macro_cache = None
_bars_lock = asyncio.Lock()
_spot_lock = asyncio.Lock()
_micro_lock = asyncio.Lock()
_series_lock = asyncio.Lock()
_macro_lock = asyncio.Lock()


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
                    limit=240,
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

        # Fail soft per timeframe. XAUS sampled spot is preferred over GC=F for
        # 5m/15m; Yahoo GC=F remains the last-resort research proxy.
        sampled_5 = []
        sampled_15 = []
        if not m5 or not m15:
            try:
                series = await get_micro_series(force=force)
                sampled_5 = sampled_spot_bars(series, XAUTimeframe.M5)
                sampled_15 = sampled_spot_bars(series, XAUTimeframe.M15)
            except Exception:
                sampled_5 = []
                sampled_15 = []

        if not m1:
            m1 = await yahoo_bars(XAUTimeframe.M1)
        if not m5:
            m5 = sampled_5 or await yahoo_bars(XAUTimeframe.M5)
        if not m15:
            m15 = sampled_15 or await yahoo_bars(XAUTimeframe.M15)

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
        "micro": micro,
        "micro_error": micro_error,
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
    try:
        macro_bias = max(-1, min(1, int(macro.get("bias", 0))))
    except (TypeError, ValueError):
        macro_bias = 0

    setup_direction = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    if setup_direction == 0:
        macro_relation = "not_applicable"
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
    if cognition_active and base_state in {"setup_macro_support", "setup_macro_neutral"}:
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
    }

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
        }

        ai_event_gate = _validated_event_gate({}, now=macro_now)

        if raw and settings.ai_api_key:
            try:
                ai = AIClient(
                    base_url=settings.ai_base_url,
                    api_key=settings.ai_api_key,
                    model=settings.ai_model,
                )
                prompt = (
                    "Using ONLY the search material below, summarize the current macro context for gold. "
                    "Return JSON only with keys bias, confidence, event_risk, event_kind, "
                    "event_name, event_time_utc, event_age_minutes, event_confidence, summary, drivers. "
                    "bias must be -1, 0, or 1. confidence and event_confidence must be 0..1. "
                    "event_kind must be breaking or none. Scheduled-event gating is handled separately "
                    "by a structured economic calendar, so NEVER set event_risk=true for scheduled releases. "
                    "Set event_risk=true only for a breaking market-moving shock explicitly supported by "
                    "the search material and occurring/published within the last 30 minutes. "
                    "For breaking events provide event_age_minutes. "
                    "Do NOT flag general ongoing geopolitics, old news, earlier-day events outside the window, "
                    "generic volatility, or events with uncertain timing. If uncertain, event_risk=false. "
                    "drivers must contain at most five short factual bullets. "
                    "Do not invent facts, prices, dates, or event times. If evidence conflicts, use bias 0.\n\n"
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
                ai_event_gate = _validated_event_gate(
                    parsed,
                    now=macro_now,
                )
                data.update({
                    "bias": bias,
                    "bias_label": "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral",
                    "confidence": confidence,
                    "summary": str(parsed.get("summary") or "").strip() or data["summary"],
                    "drivers": [str(item).strip() for item in drivers[:5] if str(item).strip()],
                })
            except Exception as exc:
                data["summary"] = "Web research succeeded, but macro synthesis failed."
                data["synthesis_error"] = type(exc).__name__

        data.update(_resolve_event_gate(ai_event_gate, calendar_event))

        _macro_cache = (time.monotonic(), data)
        return data
