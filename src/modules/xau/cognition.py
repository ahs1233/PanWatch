"""Fast, inspectable multi-layer cognition for XAU/USD research and paper trading.

The hot path is deterministic and LLM-free.  Slow macro/AI research is consumed
as cached context.  The engine builds a market-state vector, probabilistic
regimes, competing hypotheses, adversarial review, empirically-shrunk
confidence, execution timing, and a meta-controller.

Nothing in this module can place a live order or make indicative prices
execution-eligible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import exp
from typing import Any

COGNITION_VERSION = "5.4.0"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _direction_value(value: Any) -> int:
    text = str(value or "").lower()
    return 1 if text == "bullish" else -1 if text == "bearish" else 0


def _softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temperature = max(0.20, float(temperature))
    ceiling = max(scores.values())
    transformed = {
        key: exp((value - ceiling) / temperature)
        for key, value in scores.items()
    }
    total = sum(transformed.values()) or 1.0
    return {key: round(value / total, 4) for key, value in transformed.items()}


def _parse_time(value: Any) -> datetime | None:
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


def _market_session(observed_at: Any) -> str:
    value = _parse_time(observed_at) or datetime.now(timezone.utc)
    hour = value.hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london_open"
    if 12 <= hour < 16:
        return "london_ny_overlap"
    if 16 <= hour < 21:
        return "new_york"
    return "late_us"


def _fill_market_state(spot: dict[str, Any]) -> dict[str, Any]:
    """Classify paper-fill availability from all provider diagnostics."""
    explicit = str(spot.get("fill_state") or "").lower()
    explicit_map = {
        "ready": ("open", "fresh_bid_ask_available"),
        "market_closed_or_rollover": ("closed", "provider_market_closed"),
        "stale_bid_ask": ("unavailable", "stale_bid_ask"),
        "unavailable": ("unavailable", "no_fresh_bid_ask"),
    }
    if explicit in explicit_map:
        state, reason = explicit_map[explicit]
        return {
            "state": state,
            "reason": reason,
            "provider": spot.get("fill_source"),
            "market_state": spot.get("market_state"),
            "age_seconds": spot.get("fill_age_seconds"),
        }

    health = spot.get("provider_health") or []
    bid_ask_rows = [
        row for row in health
        if isinstance(row, dict) and bool(row.get("has_bid_ask"))
    ]
    fresh_bid_ask = next(
        (
            row for row in bid_ask_rows
            if row.get("status") == "ok" and not bool(row.get("is_stale"))
        ),
        None,
    )
    if fresh_bid_ask is not None:
        return {
            "state": "open",
            "reason": "fresh_bid_ask_available",
            "provider": fresh_bid_ask.get("provider"),
            "market_state": fresh_bid_ask.get("market_state"),
        }

    closed_rows = [
        row for row in bid_ask_rows
        if str(row.get("market_state") or "").lower()
        in {"closed", "market_closed", "maintenance", "rollover"}
    ]
    if closed_rows:
        row = closed_rows[0]
        return {
            "state": "closed",
            "reason": "provider_market_closed",
            "provider": row.get("provider"),
            "market_state": row.get("market_state"),
        }

    error_rows = [
        row for row in health
        if isinstance(row, dict) and row.get("status") == "error"
    ]
    if error_rows and not bid_ask_rows:
        return {
            "state": "feed_unavailable",
            "reason": "bid_ask_provider_error",
            "provider": error_rows[0].get("provider"),
            "market_state": None,
        }

    return {
        "state": "unavailable",
        "reason": "no_fresh_bid_ask",
        "provider": bid_ask_rows[0].get("provider") if bid_ask_rows else None,
        "market_state": (
            bid_ask_rows[0].get("market_state") if bid_ask_rows else None
        ),
    }


def _data_quality(technical: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    """Score analytical sensing independently from paper/live fill readiness."""
    score = 1.0
    issues: list[str] = []
    spot = technical.get("indicative_spot") or {}
    analysis_reference = technical.get("analysis_reference") or {}
    micro = technical.get("micro") or {}
    frames = technical.get("frames") or {}

    # Analytical perception should use the freshest valid context reference.
    # A stale/mid-only fill quote must not make the market "unknowable" when
    # a fresh micro/structural reference still exists.
    analysis_age_raw = analysis_reference.get("age_seconds")
    analysis_age = (
        _number(analysis_age_raw, 0.0)
        if analysis_age_raw is not None
        else None
    )
    analysis_stale = bool(analysis_reference.get("is_stale"))
    analysis_kind = str(analysis_reference.get("kind") or "none")
    analysis_price = analysis_reference.get("price")

    spot_age_raw = spot.get("age_seconds")
    spot_age = _number(spot_age_raw, 0.0) if spot_age_raw is not None else None
    micro_age_raw = micro.get("age_seconds")
    micro_age = _number(micro_age_raw, 0.0) if micro_age_raw is not None else None
    spread_bps = _number(spot.get("spread_bps"), 0.0)
    basis_bps = abs(_number(technical.get("spot_minus_proxy_bps"), 0.0))
    consensus = technical.get("spot_consensus") or {}
    consensus_delta_bps = abs(_number(consensus.get("primary_delta_bps"), 0.0))
    consensus_usable = int(_number(consensus.get("usable_count"), 0.0))

    if technical.get("blocked"):
        score -= 0.35
        issues.append("technical_data_gate")

    if analysis_price is None:
        score -= 0.40
        issues.append("analysis_reference_missing")
    else:
        if analysis_stale:
            score -= 0.30
            issues.append("analysis_reference_stale")
        elif analysis_age is not None and analysis_age > 180:
            score -= 0.25
            issues.append("analysis_reference_very_old")
        elif analysis_age is not None and analysis_age > 90:
            score -= 0.12
            issues.append("analysis_reference_aging")
        elif analysis_age is not None and analysis_age > 30:
            score -= 0.05
            issues.append("analysis_reference_slightly_aged")

    if not micro or micro.get("status") == "blocked" or micro.get("is_stale"):
        score -= 0.18
        issues.append("micro_unreliable")
    elif micro_age is not None and micro_age > 120:
        score -= 0.12
        issues.append("micro_aging")

    micro_substitutes_1m = bool(
        technical.get("technical_mode") == "spot_micro_plus_spot_5m_15m"
        and micro
        and micro.get("status") == "ready"
        and not micro.get("is_stale")
    )
    required_frames = ("5m", "15m") if micro_substitutes_1m else ("1m", "5m", "15m")
    missing_frames = [name for name in required_frames if name not in frames]
    if missing_frames:
        score -= 0.09 * len(missing_frames)
        issues.extend(f"{name}_missing" for name in missing_frames)

    if micro_substitutes_1m:
        issues.append("micro_substitutes_1m")
    elif "1m" not in frames:
        score -= 0.05

    if any("fallback" in str(item) for item in technical.get("warnings") or []):
        score -= 0.05
        issues.append("fallback_source_active")

    # Basis is only meaningful when both sides are actually available.
    if technical.get("spot_minus_proxy_bps") is not None:
        if basis_bps >= 8.0:
            score -= 0.22
            issues.append("spot_structure_disagreement")
        elif basis_bps >= 4.0:
            score -= 0.08
            issues.append("spot_structure_basis_elevated")

    # Consensus validates the analytical reference, not whether a fill quote
    # contains bid/ask. Missing consensus is only a small uncertainty penalty.
    if consensus_usable == 0:
        score -= 0.05
        issues.append("spot_consensus_unavailable")
    elif consensus_delta_bps >= 8.0:
        score -= 0.24
        issues.append("cross_source_spot_disagreement")
    elif consensus_delta_bps >= 4.0:
        score -= 0.10
        issues.append("cross_source_spot_basis_elevated")

    fill_market = _fill_market_state(spot)
    fill_issues: list[str] = []
    fill_score = 1.0
    if fill_market["state"] == "closed":
        fill_score = 0.0
        fill_issues.append("fill_market_closed")
    elif fill_market["state"] == "feed_unavailable":
        fill_score = 0.0
        fill_issues.append("fill_feed_unavailable")
    elif fill_market.get("reason") == "stale_bid_ask":
        fill_score = 0.0
        fill_issues.append("fill_stale_bid_ask")
    elif not spot:
        fill_score = 0.0
        fill_issues.append("fill_quote_missing")
    else:
        if bool(spot.get("is_stale")):
            fill_score -= 0.55
            fill_issues.append("fill_quote_stale")
        if spot.get("bid") is None or spot.get("ask") is None:
            fill_score -= 0.45
            fill_issues.append("fill_bid_ask_missing")
        if spread_bps > 3.0:
            fill_score -= min(0.35, (spread_bps - 3.0) * 0.04)
            fill_issues.append("fill_spread_elevated")

    sensors = {
        "analysis_reference_kind": analysis_kind,
        "analysis_reference_source": analysis_reference.get("source"),
        "analysis_reference_age_seconds": (
            round(analysis_age, 2) if analysis_age is not None else None
        ),
        "analysis_reference_stale": analysis_stale,
        "spot_age_seconds": round(spot_age, 2) if spot_age is not None else None,
        "micro_age_seconds": round(micro_age, 2) if micro_age is not None else None,
        "spread_bps": round(spread_bps, 4),
        "spot_proxy_basis_bps": round(basis_bps, 4),
        "spot_consensus_delta_bps": round(consensus_delta_bps, 4),
        "spot_consensus_usable": consensus_usable,
        "frame_count": len(frames),
        "technical_mode": technical.get("technical_mode"),
        "fill_readiness": {
            "score": round(_clip(fill_score), 4),
            "issues": list(dict.fromkeys(fill_issues)),
            "ready_for_paper_fill": _clip(fill_score) >= 0.99,
            "market": fill_market,
        },
    }
    return round(_clip(score), 4), list(dict.fromkeys(issues)), sensors

def _perception(technical: dict[str, Any]) -> dict[str, Any]:
    frames = technical.get("frames") or {}
    micro = technical.get("micro") or {}
    five = frames.get("5m") or {}
    fifteen = frames.get("15m") or {}
    one = frames.get("1m") or {}

    directions = [
        _direction_value(one.get("direction")),
        _direction_value(five.get("direction")),
        _direction_value(fifteen.get("direction")),
        _direction_value(micro.get("direction")),
    ]
    directional_pressure = sum(directions) / max(1, len(directions))
    ret10 = _number(micro.get("return_10m_pct"), 0.0)
    ret30 = _number(micro.get("return_30m_pct"), 0.0)
    baseline_10m = ret30 / 3.0 if ret30 else 0.0
    acceleration = ret10 - baseline_10m

    atr5 = _number(five.get("atr_pct"), 0.0)
    atr15 = _number(fifteen.get("atr_pct"), 0.0)
    volatility_pct = max(atr5, atr15)
    breakout = str(five.get("breakout") or "none")
    if breakout == "none":
        breakout = str(one.get("breakout") or "none")

    rsi5 = _number(five.get("rsi14"), 50.0)
    rsi15 = _number(fifteen.get("rsi14"), 50.0)
    momentum = "neutral"
    if directional_pressure >= 0.5 and ret10 >= 0:
        momentum = "bullish"
    elif directional_pressure <= -0.5 and ret10 <= 0:
        momentum = "bearish"

    return {
        "directional_pressure": round(directional_pressure, 4),
        "momentum": momentum,
        "return_10m_pct": round(ret10, 5),
        "return_30m_pct": round(ret30, 5),
        "acceleration": round(acceleration, 5),
        "volatility_pct": round(volatility_pct, 5),
        "breakout": breakout,
        "rsi_5m": round(rsi5, 2),
        "rsi_15m": round(rsi15, 2),
    }


def _directional_edge(
    technical: dict[str, Any],
    macro: dict[str, Any],
    perception: dict[str, Any],
    data_quality: float,
) -> dict[str, Any]:
    """Continuous multi-horizon lean.

    Higher-timeframe context is deliberately dominant: monthly/weekly/daily
    structure defines the strategic bias; 4h/1h and intraday state determine
    whether today's tape is aligned with it.
    """
    frames = technical.get("frames") or {}
    weights = {"1m": 0.15, "5m": 0.35, "15m": 0.50}
    trend = 0.0
    ema_structure = 0.0
    active_weight = 0.0
    for name, weight in weights.items():
        frame = frames.get(name) or {}
        if not frame:
            continue
        active_weight += weight
        trend += weight * _direction_value(frame.get("direction"))
        atr = max(1e-9, abs(_number(frame.get("atr14"), 0.0)))
        gap = (_number(frame.get("ema_fast")) - _number(frame.get("ema_slow"))) / atr
        ema_structure += weight * _clip(gap / 1.5, -1.0, 1.0)
    if active_weight > 0:
        trend /= active_weight
        ema_structure /= active_weight

    long_ema_structure = 0.0
    long_ema_weight = 0.0
    for name, frame_weight in (("5m", 0.40), ("15m", 0.60)):
        frame = frames.get(name) or {}
        if not frame:
            continue
        close = _number(frame.get("close"), 0.0)
        e50 = frame.get("ema50")
        e200 = frame.get("ema200")
        e1000 = frame.get("ema1000")
        frame_score = 0.0
        frame_den = 0.0
        for value, weight in ((e50, 0.30), (e200, 0.40), (e1000, 0.30)):
            if value is None or not close:
                continue
            ema_value = _number(value, 0.0)
            if not ema_value:
                continue
            frame_score += weight * (1.0 if close > ema_value else -1.0 if close < ema_value else 0.0)
            frame_den += weight
        if e50 is not None and e200 is not None:
            frame_score += 0.18 * (1.0 if _number(e50) > _number(e200) else -1.0 if _number(e50) < _number(e200) else 0.0)
            frame_den += 0.18
        if e200 is not None and e1000 is not None:
            frame_score += 0.14 * (1.0 if _number(e200) > _number(e1000) else -1.0 if _number(e200) < _number(e1000) else 0.0)
            frame_den += 0.14
        if frame_den > 0:
            long_ema_structure += frame_weight * (frame_score / frame_den)
            long_ema_weight += frame_weight
    if long_ema_weight > 0:
        long_ema_structure /= long_ema_weight
    long_ema_structure = _clip(long_ema_structure, -1.0, 1.0)

    micro = technical.get("micro") or {}
    ret10 = _number(perception.get("return_10m_pct"), 0.0)
    ret30 = _number(perception.get("return_30m_pct"), 0.0)
    momentum = (
        0.55 * _clip(ret10 / 0.18, -1.0, 1.0)
        + 0.30 * _clip(ret30 / 0.40, -1.0, 1.0)
        + 0.15 * _direction_value(micro.get("direction"))
    )

    rsi5 = _number(perception.get("rsi_5m"), 50.0)
    rsi15 = _number(perception.get("rsi_15m"), 50.0)
    rsi_impulse = (
        0.60 * _clip((rsi5 - 50.0) / 28.0, -1.0, 1.0)
        + 0.40 * _clip((rsi15 - 50.0) / 28.0, -1.0, 1.0)
    )
    breakout = str(perception.get("breakout") or "none").lower()
    breakout_score = 1.0 if breakout == "up" else -1.0 if breakout == "down" else 0.0

    context = technical.get("market_context") or {}
    bias_context = context.get("bias") or {}
    htf_score = _clip(
        _number(
            bias_context.get("today_score"),
            _number(bias_context.get("composite_score"), 0.0),
        ),
        -1.0,
        1.0,
    )
    smart = context.get("smart_money") or {}
    raw_smart_score = _clip(_number(smart.get("score"), 0.0), -1.0, 1.0)
    validation = context.get("cross_validation") or {}
    library_direction = str(validation.get("library_direction") or "neutral")
    library_agreement = _clip(_number(validation.get("library_agreement"), 0.0))
    smc_concordance = str(validation.get("smc_concordance") or "unresolved")
    # Multiple SMC libraries read the same bars, so agreement is used as a
    # reliability gate rather than counted as extra market evidence.
    if smc_concordance == "conflict" and library_agreement >= 0.67:
        smart_reliability = 0.45
    elif smc_concordance == "aligned":
        smart_reliability = 0.75 + 0.25 * library_agreement
    else:
        smart_reliability = 0.80
    smart_score = _clip(raw_smart_score * smart_reliability, -1.0, 1.0)
    flow = context.get("cash_flow") or {}
    flow_score = _clip(_number(flow.get("score"), 0.0), -1.0, 1.0)

    macro_bias = max(-1.0, min(1.0, _number(macro.get("bias"), 0.0)))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    market_time = _parse_time(technical.get("observed_at"))
    macro_time = _parse_time(macro.get("observed_at"))
    macro_age = None
    macro_freshness = 0.0
    if market_time and macro_time:
        macro_age = max(0.0, (market_time - macro_time).total_seconds())
        if macro_age <= 180:
            macro_freshness = 1.0
        elif macro_age <= 600:
            macro_freshness = 0.55
        elif macro_age <= 1800:
            macro_freshness = 0.25
    macro_component = macro_bias * macro_conf * macro_freshness

    candidate = str(technical.get("candidate") or "none")
    candidate_component = 1.0 if candidate == "long_setup" else -1.0 if candidate == "short_setup" else 0.0

    score = (
        0.28 * htf_score
        + 0.16 * smart_score
        + 0.09 * flow_score
        + 0.12 * trend
        + 0.08 * ema_structure
        + 0.12 * long_ema_structure
        + 0.07 * momentum
        + 0.03 * rsi_impulse
        + 0.02 * breakout_score
        + 0.02 * macro_component
        + 0.01 * candidate_component
    )
    score *= 0.55 + 0.45 * _clip(data_quality)
    score = _clip(score, -1.0, 1.0)
    strength = abs(score)

    if score >= 0.14:
        direction = "bullish"
    elif score <= -0.14:
        direction = "bearish"
    else:
        direction = "neutral"

    if strength >= 0.58:
        band = "strong"
    elif strength >= 0.36:
        band = "moderate"
    elif strength >= 0.14:
        band = "weak"
    else:
        band = "none"

    intraday_direction = 1 if trend >= 0.20 else -1 if trend <= -0.20 else 0
    htf_direction = 1 if htf_score >= 0.15 else -1 if htf_score <= -0.15 else 0
    htf_conflict = bool(intraday_direction and htf_direction and intraday_direction != htf_direction)

    return {
        "score": round(score, 4),
        "direction": direction,
        "strength": round(strength, 4),
        "band": band,
        "higher_timeframe_score": round(htf_score, 4),
        "higher_timeframe_direction": str(bias_context.get("today_direction") or bias_context.get("composite_direction") or "neutral"),
        "higher_timeframe_conflict": htf_conflict,
        "smart_money_score": round(smart_score, 4),
        "smart_money_raw_score": round(raw_smart_score, 4),
        "smart_money_reliability": round(smart_reliability, 4),
        "library_direction": library_direction,
        "library_agreement": round(library_agreement, 4),
        "smc_concordance": smc_concordance,
        "cash_flow_score": round(flow_score, 4),
        "macro_age_seconds": round(macro_age, 1) if macro_age is not None else None,
        "macro_freshness": round(macro_freshness, 3),
        "context_available": bool(context),
        "components": {
            "higher_timeframe": round(htf_score, 4),
            "smart_money": round(smart_score, 4),
            "smart_money_raw": round(raw_smart_score, 4),
            "smc_validation_reliability": round(smart_reliability, 4),
            "cash_flow": round(flow_score, 4),
            "intraday_trend": round(trend, 4),
            "intraday_ema_structure": round(ema_structure, 4),
            "long_ema_structure": round(long_ema_structure, 4),
            "momentum": round(momentum, 4),
            "rsi_impulse": round(rsi_impulse, 4),
            "breakout": round(breakout_score, 4),
            "macro": round(macro_component, 4),
            "candidate": round(candidate_component, 4),
        },
    }


def _regime(
    perception: dict[str, Any],
    technical: dict[str, Any],
    macro: dict[str, Any],
    data_quality: float,
) -> dict[str, Any]:
    pressure = _number(perception.get("directional_pressure"))
    vol = _number(perception.get("volatility_pct"))
    acceleration = _number(perception.get("acceleration"))
    breakout = str(perception.get("breakout") or "none")
    alignment = str(technical.get("alignment") or "mixed")
    event_risk = bool(macro.get("event_risk"))

    scores = {
        "trend_bull": 0.15 + max(0.0, pressure) * 2.0,
        "trend_bear": 0.15 + max(0.0, -pressure) * 2.0,
        "breakout_expansion": 0.10,
        "compression_range": 0.10,
        "range_rotation": 0.15,
        "transition": 0.15,
        "shock_repricing": 0.05,
        "data_uncertain": 0.05,
    }

    if alignment == "bullish":
        scores["trend_bull"] += 0.75
    elif alignment == "bearish":
        scores["trend_bear"] += 0.75
    else:
        scores["transition"] += 0.80
        scores["range_rotation"] += 0.30

    if breakout in {"up", "down"}:
        scores["breakout_expansion"] += 1.20
        if breakout == "up":
            scores["trend_bull"] += 0.30
        else:
            scores["trend_bear"] += 0.30
    else:
        scores["range_rotation"] += 0.35

    if abs(pressure) <= 0.25:
        scores["range_rotation"] += 0.75
        scores["compression_range"] += 0.50
    if vol <= 0.12:
        scores["compression_range"] += 0.85
    elif vol >= 0.24:
        scores["breakout_expansion"] += 0.45
        scores["shock_repricing"] += 0.30

    if pressure * acceleration < 0 and abs(acceleration) >= 0.02:
        scores["transition"] += 0.65
    if event_risk:
        scores["shock_repricing"] += 2.50
    if technical.get("blocked"):
        # A hard technical gate means directional inference is secondary to
        # sensor/data uncertainty. Keep probabilities, but make uncertainty
        # decisively dominant rather than allowing trend evidence to overrule it.
        scores["data_uncertain"] += 6.00
    elif data_quality < 0.60:
        scores["data_uncertain"] += 3.50

    probabilities = _softmax(scores, temperature=0.75)
    label, confidence = max(probabilities.items(), key=lambda item: item[1])
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "volatility_pct": round(vol, 5),
        "alignment": alignment,
    }


def _technical_source_family(technical: dict[str, Any]) -> str:
    """Classify structural market-data family for memory isolation.

    Spot/MT5/XAUS-derived structure and GC=F futures research are useful, but
    they are not interchangeable historical sensors.  Mixed/unknown families
    remain explicit rather than silently collapsing into one memory pool.
    """
    frames = technical.get("frames") or {}
    sources = [
        str((frames.get(name) or {}).get("source") or "").lower()
        for name in ("1m", "5m", "15m")
    ]
    nonempty = [value for value in sources if value]
    mode = str(technical.get("technical_mode") or "").lower()

    if nonempty and all(
        ("biquote" in value or "mt5" in value or "xaus.com" in value)
        for value in nonempty
    ):
        return "xau_spot_structure"
    if any("gc=f" in value or "yfinance" in value for value in nonempty):
        if all("gc=f" in value or "yfinance" in value for value in nonempty):
            return "gc_futures_proxy"
        return "mixed_research"

    if mode in {"biquote_mt5_1m_5m_15m", "spot_micro_plus_spot_5m_15m"}:
        return "xau_spot_structure"
    if "mixed_research" in mode:
        return "mixed_research"

    reference_source = str(
        (technical.get("analysis_reference") or {}).get("source") or ""
    ).lower()
    if "biquote" in reference_source or "xaus.com" in reference_source:
        return "xau_spot_structure"
    if "gc=f" in reference_source or "yfinance" in reference_source:
        return "gc_futures_proxy"
    return "unknown"


def build_market_state_vector(
    technical: dict[str, Any],
    macro: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact state vector suitable for episodic similarity search."""

    perception = _perception(technical)
    quality, quality_issues, sensors = _data_quality(technical)
    regime = _regime(perception, technical, macro, quality)
    spot = technical.get("indicative_spot") or {}
    observed_at = (
        spot.get("observed_at")
        or technical.get("observed_at")
        or (technical.get("micro") or {}).get("observed_at")
    )
    candidate = str(technical.get("candidate") or "none")
    session = _market_session(observed_at)

    return {
        "version": 2,
        "source_family": _technical_source_family(technical),
        "candidate": candidate,
        "alignment": str(technical.get("alignment") or "mixed"),
        "session": session,
        "regime": regime.get("label"),
        "directional_pressure": _number(perception.get("directional_pressure")),
        "return_10m_pct": _number(perception.get("return_10m_pct")),
        "return_30m_pct": _number(perception.get("return_30m_pct")),
        "acceleration": _number(perception.get("acceleration")),
        "volatility_pct": _number(perception.get("volatility_pct")),
        "rsi_5m_norm": (_number(perception.get("rsi_5m"), 50.0) - 50.0) / 50.0,
        "rsi_15m_norm": (_number(perception.get("rsi_15m"), 50.0) - 50.0) / 50.0,
        "breakout": str(perception.get("breakout") or "none"),
        "macro_bias": max(-1.0, min(1.0, _number(macro.get("bias"), 0.0))),
        "macro_confidence": _clip(_number(macro.get("confidence"), 0.0)),
        "event_risk": 1.0 if macro.get("event_risk") else 0.0,
        "spread_bps": _number(sensors.get("spread_bps"), 0.0),
        "spot_age_seconds": _number(sensors.get("spot_age_seconds"), 60.0),
        "micro_age_seconds": _number(sensors.get("micro_age_seconds"), 120.0),
        "spot_proxy_basis_bps": _number(sensors.get("spot_proxy_basis_bps"), 0.0),
        "spot_consensus_delta_bps": _number(sensors.get("spot_consensus_delta_bps"), 0.0),
        "data_quality": quality,
        "quality_issues": quality_issues,
    }


def state_vector_similarity(current: dict[str, Any], historical: dict[str, Any]) -> float:
    """Return 0..1 similarity for two XAU market-state vectors.

    Scales are intentionally explicit and conservative so one noisy feature
    cannot dominate the retrieval.
    """

    if not current or not historical:
        return 0.0

    numeric = (
        ("directional_pressure", 2.0, 2.0),
        ("return_10m_pct", 0.50, 1.2),
        ("return_30m_pct", 1.00, 0.8),
        ("acceleration", 0.40, 1.0),
        ("volatility_pct", 0.35, 0.8),
        ("rsi_5m_norm", 1.0, 0.8),
        ("rsi_15m_norm", 1.0, 0.5),
        ("macro_bias", 2.0, 0.8),
        ("macro_confidence", 1.0, 0.4),
        ("spread_bps", 5.0, 0.5),
        ("spot_proxy_basis_bps", 10.0, 0.5),
        ("spot_consensus_delta_bps", 10.0, 0.6),
        ("data_quality", 1.0, 0.8),
    )
    categorical = (
        ("candidate", 1.6),
        ("alignment", 1.0),
        ("session", 0.55),
        ("regime", 1.3),
        ("breakout", 0.65),
    )

    weighted_similarity = 0.0
    total_weight = 0.0
    for key, scale, weight in numeric:
        a = _number(current.get(key), 0.0)
        b = _number(historical.get(key), 0.0)
        local = 1.0 - min(1.0, abs(a - b) / max(scale, 1e-9))
        weighted_similarity += local * weight
        total_weight += weight

    for key, weight in categorical:
        a = str(current.get(key) or "")
        b = str(historical.get(key) or "")
        local = 1.0 if a and a == b else 0.0
        weighted_similarity += local * weight
        total_weight += weight

    return round(_clip(weighted_similarity / max(total_weight, 1e-9)), 4)


def _hypotheses(
    technical: dict[str, Any],
    macro: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
    edge: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate = str(technical.get("candidate") or "none")
    setup_dir = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    if setup_dir == 0:
        edge_direction = str(edge.get("direction") or "neutral")
        if edge_direction == "bullish":
            setup_dir = 1
        elif edge_direction == "bearish":
            setup_dir = -1
    edge_strength = _number(edge.get("strength"), 0.0)
    context = technical.get("market_context") or {}
    bias_context = context.get("bias") or {}
    htf_score = _clip(
        _number(
            bias_context.get("today_score"),
            _number(bias_context.get("composite_score"), 0.0),
        ),
        -1.0,
        1.0,
    )
    if setup_dir == 0:
        if htf_score >= 0.18:
            setup_dir = 1
        elif htf_score <= -0.18:
            setup_dir = -1
    smart_money = context.get("smart_money") or {}
    smart_score = _clip(_number(smart_money.get("score"), 0.0), -1.0, 1.0)
    sweep_state = str(smart_money.get("liquidity_sweep") or "none")
    displacement = str(smart_money.get("displacement") or "none")
    profile = context.get("volume_profile") or {}
    profile_location = str(profile.get("location") or "unknown")
    flow_context = context.get("cash_flow") or {}
    flow_score = _clip(_number(flow_context.get("score"), 0.0), -1.0, 1.0)
    macro_bias = int(max(-1, min(1, _number(macro.get("bias"), 0.0))))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    pressure = _number(perception.get("directional_pressure"))
    acceleration = _number(perception.get("acceleration"))
    breakout = str(perception.get("breakout") or "none")
    rsi5 = _number(perception.get("rsi_5m"), 50.0)
    probs = regime.get("probabilities") or {}
    session = _market_session(technical.get("observed_at"))

    htf_alignment = htf_score * setup_dir if setup_dir else 0.0
    smart_alignment = smart_score * setup_dir if setup_dir else 0.0
    flow_alignment = flow_score * setup_dir if setup_dir else 0.0

    continuation = (
        0.25
        + (0.55 if setup_dir else -0.30)
        + 0.75 * edge_strength
        + 0.65 * abs(pressure)
        + 0.85 * max(0.0, htf_alignment)
        - 0.65 * max(0.0, -htf_alignment)
        + 0.40 * max(0.0, smart_alignment)
        + 0.25 * max(0.0, flow_alignment)
        + 1.00 * _number(probs.get("trend_bull" if setup_dir > 0 else "trend_bear"))
        + 0.65 * _number(probs.get("breakout_expansion"))
    )
    if setup_dir and macro_bias == setup_dir:
        continuation += 0.50 * max(0.30, macro_conf)
    elif setup_dir and macro_bias == -setup_dir:
        continuation -= 0.70 * max(0.30, macro_conf)

    mean_reversion = (
        0.15
        + 1.10 * _number(probs.get("range_rotation"))
        + 0.80 * _number(probs.get("compression_range"))
        + (0.70 if rsi5 >= 70 or rsi5 <= 30 else 0.0)
        + (0.45 if profile_location in {"above_value", "below_value"} else 0.0)
        + (0.35 if setup_dir and acceleration * setup_dir < 0 else 0.0)
        + (0.30 if setup_dir and flow_alignment < -0.12 else 0.0)
    )

    liquidity_sweep = (
        0.10
        + (0.90 if sweep_state in {"buy_side_sweep", "sell_side_sweep"} else 0.0)
        + (0.50 if breakout in {"up", "down"} else 0.0)
        + 0.60 * _number(probs.get("transition"))
        + (0.35 if rsi5 >= 75 or rsi5 <= 25 else 0.0)
    )

    failed_breakout = (
        0.05
        + (0.65 if breakout in {"up", "down"} else 0.0)
        + (0.45 if sweep_state in {"buy_side_sweep", "sell_side_sweep"} else 0.0)
        + 0.70 * _number(probs.get("transition"))
        + (0.45 if setup_dir and acceleration * setup_dir < 0 else 0.0)
    )

    volatility_expansion = (
        0.05
        + 1.35 * _number(probs.get("breakout_expansion"))
        + (0.45 if displacement in {"bullish", "bearish"} else 0.0)
        + (0.30 if breakout in {"up", "down"} else 0.0)
    )

    news_repricing = (
        0.02
        + 1.80 * _number(probs.get("shock_repricing"))
        + (0.50 * macro_conf if macro_bias else 0.0)
    )

    exhaustion = (
        0.05
        + (0.80 if rsi5 >= 78 or rsi5 <= 22 else 0.0)
        + (0.50 if setup_dir and acceleration * setup_dir < 0 else 0.0)
    )

    session_reversal = (
        0.05
        + (0.30 if session in {"london_open", "london_ny_overlap"} else 0.0)
        + 0.50 * _number(probs.get("transition"))
    )

    scores = {
        "trend_continuation": continuation,
        "mean_reversion": mean_reversion,
        "liquidity_sweep": liquidity_sweep,
        "failed_breakout": failed_breakout,
        "volatility_expansion": volatility_expansion,
        "news_repricing": news_repricing,
        "exhaustion": exhaustion,
        "session_reversal": session_reversal,
    }
    weights = _softmax(scores, temperature=0.75)

    direction = "long" if setup_dir > 0 else "short" if setup_dir < 0 else "none"
    opposite = "short" if direction == "long" else "long" if direction == "short" else "none"
    macro_direction = "long" if macro_bias > 0 else "short" if macro_bias < 0 else direction

    direction_map = {
        "trend_continuation": direction,
        "volatility_expansion": direction,
        "news_repricing": macro_direction,
        "mean_reversion": opposite,
        "liquidity_sweep": opposite,
        "failed_breakout": opposite,
        "exhaustion": opposite,
        "session_reversal": opposite,
    }
    evidence_map = {
        "trend_continuation": ["timeframe_alignment", "directional_pressure"],
        "volatility_expansion": ["breakout_probability", "volatility_regime"],
        "news_repricing": ["macro_context", "shock_probability"],
        "mean_reversion": ["range_probability", "rsi_extremity"],
        "liquidity_sweep": ["breakout_state", "transition_probability"],
        "failed_breakout": ["breakout_state", "momentum_deceleration"],
        "exhaustion": ["rsi_extremity", "momentum_deceleration"],
        "session_reversal": ["session_context", "transition_probability"],
    }

    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]
    return [
        {
            "name": name,
            "weight": weight,
            "direction": direction_map.get(name, "none"),
            "evidence": evidence_map.get(name, []),
        }
        for name, weight in ordered
    ]


def _memory_adjustment(memory: dict[str, Any] | None) -> dict[str, Any]:
    memory = memory or {}
    trade_count = int(_number(memory.get("trade_count"), 0.0))
    similar_samples = int(_number(memory.get("similar_samples"), trade_count))
    expectancy = _number(memory.get("expectancy_r"), 0.0)
    profit_factor = memory.get("profit_factor")
    posterior = memory.get("posterior_win_probability")
    empirical = memory.get("empirical_win_rate")
    calibration_samples = int(_number(memory.get("calibration_sample_count"), 0.0))
    brier = memory.get("brier_score")
    ece = memory.get("expected_calibration_error")

    sample_basis = max(similar_samples, calibration_samples)
    strength = _clip(sample_basis / 40.0)
    edge = _clip(expectancy / 1.5, -1.0, 1.0)
    profit_factor_value = _number(profit_factor, 1.0) if profit_factor is not None else 1.0
    pf_edge = _clip((profit_factor_value - 1.0) / 2.0, -1.0, 1.0)
    trade_adjustment = (0.035 * edge + 0.025 * pf_edge) * strength

    # Shadow memory is research-only. It never replaces trade calibration and
    # contributes at most +/-1.5 confidence points after enough similar samples.
    shadow = memory.get("shadow_memory") or {}
    shadow_samples = int(_number(shadow.get("sample_count"), 0.0))
    shadow_similarity = _number(shadow.get("average_similarity"), 0.0)
    shadow_positive_rate = shadow.get("positive_rate")
    shadow_adjustment = 0.0
    if (
        shadow_samples >= 8
        and shadow_similarity >= 0.68
        and shadow_positive_rate is not None
        and bool(shadow.get("decision_filtered", False))
        and bool(shadow.get("research_only", False))
        and bool(shadow.get("lookahead_protected", False))
        and bool(shadow.get("temporally_decorrelated", False))
    ):
        shadow_strength = _clip(shadow_samples / 40.0) * _clip(shadow_similarity)
        positive_edge = _clip(
            (float(shadow_positive_rate) - 0.5) * 2.0,
            -1.0,
            1.0,
        )
        # Forward shadow outcomes are heterogeneous across 15m/30m/60m.
        # Their return magnitude is diagnostic only; confidence receives only
        # a small directional prior, capped at +/-0.8 percentage points.
        shadow_adjustment = 0.008 * positive_edge * shadow_strength

    # Walk-forward replay is also research-only. It requires a larger sample
    # than live shadow memory and is bounded to +/-1 confidence point.
    replay = memory.get("replay_memory") or {}
    replay_samples = int(_number(replay.get("sample_count"), 0.0))
    replay_similarity = _number(replay.get("average_similarity"), 0.0)
    replay_positive_rate = replay.get("positive_rate")
    replay_weighted_bps = replay.get("similarity_weighted_return_bps")
    replay_adjustment = 0.0
    if (
        replay_samples >= 15
        and replay_similarity >= 0.70
        and replay_positive_rate is not None
        and replay_weighted_bps is not None
        and bool(replay.get("research_only", False))
        and bool(replay.get("lookahead_protected", False))
        and bool(replay.get("temporally_decorrelated", False))
    ):
        replay_strength = _clip(replay_samples / 80.0) * _clip(replay_similarity)
        replay_positive_edge = _clip(
            (float(replay_positive_rate) - 0.5) * 2.0,
            -1.0,
            1.0,
        )
        replay_return_edge = _clip(
            float(replay_weighted_bps) / 25.0,
            -1.0,
            1.0,
        )
        replay_adjustment = (
            0.006 * replay_positive_edge + 0.004 * replay_return_edge
        ) * replay_strength

    research_raw = shadow_adjustment + replay_adjustment
    research_conflict = bool(
        shadow_adjustment
        and replay_adjustment
        and shadow_adjustment * replay_adjustment < 0
    )
    if research_conflict:
        # Independent research priors disagree: reduce their joint influence
        # instead of letting one source dominate by magnitude.
        research_raw *= 0.25

    trade_research_conflict = bool(
        sample_basis >= 15
        and trade_adjustment
        and research_raw
        and trade_adjustment * research_raw < 0
    )
    if trade_research_conflict:
        # Executed, calibrated evidence outranks research-only priors.  Replay
        # and shadow memory may challenge the live record, but must not cancel
        # a sufficiently established realized edge.
        research_raw *= 0.20

    research_adjustment = _clip(
        research_raw,
        -0.02,
        0.02,
    )
    confidence_adjustment = trade_adjustment + research_adjustment

    return {
        "trade_count": trade_count,
        "similar_samples": similar_samples,
        "expectancy_r": round(expectancy, 4),
        "profit_factor": profit_factor,
        "empirical_win_rate": empirical,
        "posterior_win_probability": posterior,
        "calibration_sample_count": calibration_samples,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "average_similarity": memory.get("average_similarity"),
        "memory_source": memory.get("source"),
        "learning_strength": round(strength, 4),
        "confidence_adjustment": round(confidence_adjustment, 4),
        "trade_confidence_adjustment": round(trade_adjustment, 4),
        "shadow_confidence_adjustment": round(shadow_adjustment, 4),
        "replay_confidence_adjustment": round(replay_adjustment, 4),
        "research_confidence_adjustment": round(research_adjustment, 4),
        "research_prior_conflict": research_conflict,
        "trade_research_conflict": trade_research_conflict,
        "shadow_memory": shadow,
        "replay_memory": replay,
        "autopsy_counts": memory.get("autopsy_counts") or {},
    }


def _adaptive_entry_threshold(
    base_threshold: float,
    memory_state: dict[str, Any],
) -> dict[str, Any]:
    """Conservative online adaptation: adjust only the paper-entry threshold."""
    base = _clip(float(base_threshold), 0.50, 0.90)
    ece = memory_state.get("expected_calibration_error")
    ece_value = _number(ece, 0.0) if ece is not None else 0.0
    samples = int(_number(memory_state.get("calibration_sample_count"), 0.0))
    expectancy = _number(memory_state.get("expectancy_r"), 0.0)
    autopsy_counts = memory_state.get("autopsy_counts") or {}
    high_conf_errors = int(_number(autopsy_counts.get("high_confidence_error"), 0.0))
    # Autopsy primary counts may not include secondary high-confidence labels,
    # so ECE remains the main calibration signal.
    delta = 0.0
    reasons: list[str] = []

    if samples >= 8 and ece_value >= 0.20:
        delta += min(0.06, 0.02 + (ece_value - 0.20) * 0.20)
        reasons.append("calibration_error_raise_threshold")
    if high_conf_errors >= 3:
        delta += 0.02
        reasons.append("repeated_high_confidence_errors")
    if samples >= 25 and ece_value <= 0.08 and expectancy >= 0.25:
        delta -= 0.015
        reasons.append("well_calibrated_positive_history_small_relaxation")

    effective = _clip(base + delta, 0.50, 0.90)
    return {
        "base": round(base, 4),
        "effective": round(effective, 4),
        "delta": round(effective - base, 4),
        "reasons": reasons,
    }


def _adversarial_review(
    technical: dict[str, Any],
    macro: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    data_quality: float,
) -> dict[str, Any]:
    candidate = str(technical.get("candidate") or "none")
    setup_dir = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    macro_bias = int(max(-1, min(1, _number(macro.get("bias"), 0.0))))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    context = technical.get("market_context") or {}
    bias_context = context.get("bias") or {}
    htf_score = _clip(
        _number(
            bias_context.get("today_score"),
            _number(bias_context.get("composite_score"), 0.0),
        ),
        -1.0,
        1.0,
    )
    flow_score = _clip(_number((context.get("cash_flow") or {}).get("score"), 0.0), -1.0, 1.0)
    smart_score = _clip(_number((context.get("smart_money") or {}).get("score"), 0.0), -1.0, 1.0)
    validation = context.get("cross_validation") or {}
    library_direction = str(validation.get("library_direction") or "neutral")
    library_agreement = _clip(_number(validation.get("library_agreement"), 0.0))
    smc_concordance = str(validation.get("smc_concordance") or "unresolved")
    primary = hypotheses[0] if hypotheses else {"name": "none", "weight": 0.0, "direction": "none"}
    rsi5 = _number(perception.get("rsi_5m"), 50.0)

    counter_evidence: list[str] = []
    hard_veto = False

    if data_quality < 0.58:
        counter_evidence.append("insufficient_data_quality")
        hard_veto = True
    if bool(macro.get("event_risk")):
        counter_evidence.append("active_high_impact_event")
        hard_veto = True
    if setup_dir and macro_bias == -setup_dir and macro_conf >= 0.65:
        counter_evidence.append("high_confidence_macro_conflict")
    if setup_dir and htf_score * setup_dir <= -0.35:
        counter_evidence.append("strong_higher_timeframe_conflict")
    if setup_dir and flow_score * setup_dir <= -0.30:
        counter_evidence.append("cash_flow_opposes_setup")
    if setup_dir and smart_score * setup_dir <= -0.35:
        counter_evidence.append("smart_money_structure_opposes_setup")
    if smc_concordance == "conflict" and library_agreement >= 0.67:
        counter_evidence.append("smc_cross_validation_conflict")
    if setup_dir > 0 and library_direction == "bearish" and library_agreement >= 0.67:
        counter_evidence.append("smc_validators_oppose_long")
    elif setup_dir < 0 and library_direction == "bullish" and library_agreement >= 0.67:
        counter_evidence.append("smc_validators_oppose_short")
    if primary.get("direction") not in {
        "long" if setup_dir > 0 else "short" if setup_dir < 0 else "none",
        "none",
    }:
        counter_evidence.append("dominant_hypothesis_opposes_setup")
    if _number(primary.get("weight")) < 0.30:
        counter_evidence.append("weak_primary_hypothesis")
    if setup_dir > 0 and rsi5 >= 74:
        counter_evidence.append("long_setup_overextended_rsi")
    elif setup_dir < 0 and rsi5 <= 26:
        counter_evidence.append("short_setup_overextended_rsi")
    if _number((regime.get("probabilities") or {}).get("transition")) >= 0.30:
        counter_evidence.append("transition_risk")
    if regime.get("label") == "data_uncertain":
        counter_evidence.append("unstable_data_regime")
        hard_veto = True

    veto_score = 0.0
    veto_score += 0.50 if hard_veto else 0.0
    veto_score += min(0.42, 0.085 * len(counter_evidence))
    veto_score = _clip(veto_score)
    veto = hard_veto or veto_score >= 0.64

    return {
        "veto": veto,
        "veto_score": round(veto_score, 4),
        "counter_evidence": list(dict.fromkeys(counter_evidence)),
    }


def _confidence(
    data_quality: float,
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    macro: dict[str, Any],
    adversarial: dict[str, Any],
    memory_state: dict[str, Any],
) -> dict[str, Any]:
    primary_weight = _number((hypotheses[0] if hypotheses else {}).get("weight"), 0.0)
    regime_conf = _number(regime.get("confidence"), 0.0)
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    event_penalty = 0.24 if macro.get("event_risk") else 0.0
    adversarial_penalty = 0.32 * _number(adversarial.get("veto_score"), 0.0)

    raw = (
        0.32 * data_quality
        + 0.28 * primary_weight
        + 0.20 * regime_conf
        + 0.10 * macro_conf
        + 0.10
        - event_penalty
        - adversarial_penalty
    )
    raw = _clip(raw, 0.05, 0.95)

    historical_probability = memory_state.get("posterior_win_probability")
    sample_count = int(_number(memory_state.get("calibration_sample_count"), 0.0))
    similar_samples = int(_number(memory_state.get("similar_samples"), 0.0))
    effective_samples = max(sample_count, similar_samples)
    ece = memory_state.get("expected_calibration_error")
    ece_value = _number(ece, 0.0) if ece is not None else 0.0

    if historical_probability is not None and effective_samples > 0:
        historical = _clip(_number(historical_probability), 0.05, 0.95)
        reliability = min(0.72, effective_samples / 45.0)
        reliability *= max(0.35, 1.0 - min(0.50, ece_value))
        calibrated = raw * (1.0 - reliability) + historical * reliability
        basis = "empirical_bayesian_history"
    else:
        # No history: deliberately shrink toward uncertainty instead of
        # pretending a model score is a measured probability.
        calibrated = raw * 0.76 + 0.50 * 0.24
        basis = "prior_shrunk_no_history"

    calibrated += _number(memory_state.get("confidence_adjustment"), 0.0)
    calibrated = _clip(calibrated, 0.05, 0.95)
    band = "high" if calibrated >= 0.72 else "medium" if calibrated >= 0.58 else "low"

    return {
        "raw_confidence": round(raw, 4),
        "calibrated_confidence": round(calibrated, 4),
        "band": band,
        "calibration_basis": basis,
        "score_kind": "heuristic_with_historical_adjustment",
        "is_validated_win_probability": False,
        "historical_probability": historical_probability,
        "sample_count": effective_samples,
        "brier_score": memory_state.get("brier_score"),
        "expected_calibration_error": memory_state.get("expected_calibration_error"),
    }


def _scenario_paths(
    technical: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    edge: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build scenario geometry from the activation level, never from spot alone.

    Targets are always placed beyond their trigger in the scenario direction.
    This prevents impossible states such as a bullish target below its breakout
    trigger or a bearish target above its breakdown trigger.
    """
    analysis_reference = technical.get("analysis_reference") or {}
    spot = technical.get("indicative_spot") or {}
    price = _number(analysis_reference.get("price"), _number(spot.get("price"), 0.0))
    atr = max(0.0, _number(technical.get("atr_reference"), 0.0))
    swing_high = technical.get("swing_high_reference")
    swing_low = technical.get("swing_low_reference")
    swing_high_n = _number(swing_high, 0.0) if swing_high is not None else 0.0
    swing_low_n = _number(swing_low, 0.0) if swing_low is not None else 0.0

    weights = {str(item.get("name")): _number(item.get("weight")) for item in hypotheses}
    continuation = (
        weights.get("trend_continuation", 0.0)
        + 0.55 * weights.get("volatility_expansion", 0.0)
        + 0.35 * weights.get("news_repricing", 0.0)
    )
    reversal = (
        weights.get("mean_reversion", 0.0)
        + weights.get("failed_breakout", 0.0)
        + weights.get("exhaustion", 0.0)
        + 0.60 * weights.get("session_reversal", 0.0)
    )
    sweep = weights.get("liquidity_sweep", 0.0)
    total = continuation + reversal + sweep or 1.0

    edge_direction = str(edge.get("direction") or "neutral")
    context = technical.get("market_context") or {}
    bias_context = context.get("bias") or {}
    smart_money = context.get("smart_money") or {}
    profile = context.get("volume_profile") or {}
    flow = context.get("cash_flow") or {}
    frames = technical.get("frames") or {}
    five = frames.get("5m") or {}
    fifteen = frames.get("15m") or {}

    top_down_direction = str(
        bias_context.get("today_direction")
        or bias_context.get("composite_direction")
        or "neutral"
    )
    smart_direction = str(smart_money.get("bias") or "neutral")
    continuation_direction = (
        top_down_direction
        if top_down_direction in {"bullish", "bearish"}
        else edge_direction
        if edge_direction in {"bullish", "bearish"}
        else smart_direction
        if smart_direction in {"bullish", "bearish"}
        else "neutral"
    )
    continuation_basis = (
        "top_down_bias"
        if top_down_direction in {"bullish", "bearish"}
        else "directional_edge"
        if edge_direction in {"bullish", "bearish"}
        else "smart_money_structure"
        if smart_direction in {"bullish", "bearish"}
        else "unresolved"
    )

    sweep_signal = str(smart_money.get("liquidity_sweep") or "none")
    flow_direction = str(flow.get("direction") or "balanced")
    profile_location = str(profile.get("location") or "unknown")
    rsi5 = _number(five.get("rsi14"), 50.0)
    rsi15 = _number(fifteen.get("rsi14"), 50.0)

    reversal_direction = "neutral"
    reversal_basis = "unresolved"
    if sweep_signal == "buy_side_sweep":
        reversal_direction = "bearish"
        reversal_basis = "confirmed_buy_side_sweep"
    elif sweep_signal == "sell_side_sweep":
        reversal_direction = "bullish"
        reversal_basis = "confirmed_sell_side_sweep"
    elif profile_location == "above_value" and flow_direction == "outflow":
        reversal_direction = "bearish"
        reversal_basis = "above_value_with_outflow"
    elif profile_location == "below_value" and flow_direction == "inflow":
        reversal_direction = "bullish"
        reversal_basis = "below_value_with_inflow"
    elif rsi5 >= 68 and rsi15 >= 55:
        reversal_direction = "bearish"
        reversal_basis = "intraday_overextension"
    elif rsi5 <= 32 and rsi15 <= 45:
        reversal_direction = "bullish"
        reversal_basis = "intraday_underextension"
    elif continuation_direction == "bullish":
        reversal_direction = "bearish"
        reversal_basis = "countertrend_conditional"
    elif continuation_direction == "bearish":
        reversal_direction = "bullish"
        reversal_basis = "countertrend_conditional"

    sweep_direction = "neutral"
    sweep_basis = "unresolved"
    if sweep_signal == "buy_side_sweep":
        sweep_direction = "bearish"
        sweep_basis = "observed_buy_side_sweep"
    elif sweep_signal == "sell_side_sweep":
        sweep_direction = "bullish"
        sweep_basis = "observed_sell_side_sweep"
    elif reversal_direction in {"bullish", "bearish"}:
        sweep_direction = reversal_direction
        sweep_basis = "conditional_liquidity_path"

    def directional_target(trigger: float | None, direction: str, atr_multiple: float) -> float | None:
        if trigger is None or trigger <= 0 or atr <= 0 or direction == "neutral":
            return None
        distance = max(0.35, atr * atr_multiple)
        sign = 1.0 if direction == "bullish" else -1.0
        return round(trigger + sign * distance, 4)

    if continuation_direction == "bullish":
        continuation_trigger = swing_high_n or (price if price > 0 else None)
        continuation_invalidation = swing_low_n or None
    elif continuation_direction == "bearish":
        continuation_trigger = swing_low_n or (price if price > 0 else None)
        continuation_invalidation = swing_high_n or None
    else:
        continuation_trigger = None
        continuation_invalidation = None

    if reversal_direction == "bearish":
        reversal_trigger = swing_low_n or (price if price > 0 else None)
        reversal_invalidation = swing_high_n or None
    elif reversal_direction == "bullish":
        reversal_trigger = swing_high_n or (price if price > 0 else None)
        reversal_invalidation = swing_low_n or None
    else:
        reversal_trigger = None
        reversal_invalidation = None

    if sweep_direction == "bearish":
        sweep_trigger = swing_high_n or (price if price > 0 else None)
    elif sweep_direction == "bullish":
        sweep_trigger = swing_low_n or (price if price > 0 else None)
    else:
        sweep_trigger = None

    sweep_target = directional_target(sweep_trigger, sweep_direction, 0.80)
    sweep_invalidation = None
    if sweep_trigger and atr > 0 and sweep_direction != "neutral":
        sign = 1.0 if sweep_direction == "bearish" else -1.0
        sweep_invalidation = round(sweep_trigger + sign * atr * 0.30, 4)

    scenarios = [
        {
            "name": "continuation",
            "direction": continuation_direction,
            "weight": round(continuation / total, 4),
            "target": directional_target(continuation_trigger, continuation_direction, 0.85),
            "trigger": round(continuation_trigger, 4) if continuation_trigger else None,
            "trigger_kind": "breakout",
            "direction_basis": continuation_basis,
            "weight_type": "relative_hypothesis_weight",
            "invalidation": round(continuation_invalidation, 4) if continuation_invalidation else None,
        },
        {
            "name": "reversal",
            "direction": reversal_direction,
            "weight": round(reversal / total, 4),
            "target": directional_target(reversal_trigger, reversal_direction, 0.75),
            "trigger": round(reversal_trigger, 4) if reversal_trigger else None,
            "trigger_kind": "structure_break",
            "direction_basis": reversal_basis,
            "weight_type": "relative_hypothesis_weight",
            "invalidation": round(reversal_invalidation, 4) if reversal_invalidation else None,
        },
        {
            "name": "liquidity_sweep",
            "direction": sweep_direction,
            "weight": round(sweep / total, 4),
            "target": sweep_target,
            "trigger": round(sweep_trigger, 4) if sweep_trigger else None,
            "trigger_kind": "sweep_then_reclaim",
            "direction_basis": sweep_basis,
            "weight_type": "relative_hypothesis_weight",
            "invalidation": sweep_invalidation,
        },
    ]

    # Geometry invariant: directional targets must sit beyond their triggers.
    for scenario in scenarios:
        trigger = scenario.get("trigger")
        target = scenario.get("target")
        direction = scenario.get("direction")
        if isinstance(trigger, (int, float)) and isinstance(target, (int, float)):
            if direction == "bullish" and target <= trigger:
                scenario["target"] = round(trigger + max(0.35, atr * 0.50), 4)
            elif direction == "bearish" and target >= trigger:
                scenario["target"] = round(trigger - max(0.35, atr * 0.50), 4)
    return scenarios


def _activation_state(
    side: str | None,
    technical: dict[str, Any],
    perception: dict[str, Any],
    *,
    setup_confirmed: bool,
) -> dict[str, Any]:
    frames = technical.get("frames") or {}
    five = frames.get("5m") or {}
    fifteen = frames.get("15m") or {}
    close5 = _number(five.get("close"), 0.0)
    ema_fast = _number(five.get("ema_fast"), 0.0)
    dir5 = str(five.get("direction") or "neutral")
    dir15 = str(fifteen.get("direction") or "neutral")
    context = technical.get("market_context") or {}
    bias_context = context.get("bias") or {}
    htf_score = _clip(
        _number(
            bias_context.get("today_score"),
            _number(bias_context.get("composite_score"), 0.0),
        ),
        -1.0,
        1.0,
    )
    flow_score = _clip(_number((context.get("cash_flow") or {}).get("score"), 0.0), -1.0, 1.0)
    validation = context.get("cross_validation") or {}
    library_direction = str(validation.get("library_direction") or "neutral")
    library_agreement = _clip(_number(validation.get("library_agreement"), 0.0))
    ret10 = _number(perception.get("return_10m_pct"), 0.0)
    momentum = str(perception.get("momentum") or "neutral")

    conditions: list[dict[str, Any]] = []
    if side == "long":
        conditions = [
            {
                "key": "5m_close_above_fast_ema",
                "label": "5m close above fast EMA",
                "status": "satisfied" if close5 > ema_fast else "pending",
                "current": round(close5, 4) if close5 else None,
                "threshold": round(ema_fast, 4) if ema_fast else None,
            },
            {
                "key": "5m_momentum_positive",
                "label": "5m momentum positive",
                "status": "satisfied" if (dir5 == "bullish" or momentum == "bullish" or ret10 > 0.015) else "failed" if (dir5 == "bearish" and ret10 < -0.015) else "pending",
                "current": round(ret10, 5),
                "threshold": 0.0,
            },
            {
                "key": "15m_not_bearish",
                "label": "15m not bearish",
                "status": "failed" if dir15 == "bearish" else "satisfied" if dir15 in {"bullish", "neutral"} else "pending",
                "current": dir15,
                "threshold": "not_bearish",
            },
            {
                "key": "top_down_bias_not_strongly_bearish",
                "label": "Top-down bias not strongly bearish",
                "status": "failed" if htf_score <= -0.35 else "pending" if htf_score < -0.10 else "satisfied",
                "current": round(htf_score, 4),
                "threshold": -0.35,
            },
            {
                "key": "flow_not_strongly_opposed",
                "label": "Cash flow not strongly opposed",
                "status": "failed" if flow_score <= -0.30 else "pending" if flow_score < -0.10 else "satisfied",
                "current": round(flow_score, 4),
                "threshold": -0.30,
            },
            {
                "key": "smc_cross_validation_not_bearish",
                "label": "SMC cross-validation not bearish",
                "status": (
                    "failed" if library_direction == "bearish" and library_agreement >= 0.67
                    else "satisfied" if library_direction == "bullish" and library_agreement >= 0.50
                    else "pending"
                ),
                "current": f"{library_direction}:{round(library_agreement, 2)}",
                "threshold": "not_bearish_consensus",
            },
        ]
    elif side == "short":
        conditions = [
            {
                "key": "5m_close_below_fast_ema",
                "label": "5m close below fast EMA",
                "status": "satisfied" if close5 < ema_fast else "pending",
                "current": round(close5, 4) if close5 else None,
                "threshold": round(ema_fast, 4) if ema_fast else None,
            },
            {
                "key": "5m_momentum_negative",
                "label": "5m momentum negative",
                "status": "satisfied" if (dir5 == "bearish" or momentum == "bearish" or ret10 < -0.015) else "failed" if (dir5 == "bullish" and ret10 > 0.015) else "pending",
                "current": round(ret10, 5),
                "threshold": 0.0,
            },
            {
                "key": "15m_not_bullish",
                "label": "15m not bullish",
                "status": "failed" if dir15 == "bullish" else "satisfied" if dir15 in {"bearish", "neutral"} else "pending",
                "current": dir15,
                "threshold": "not_bullish",
            },
            {
                "key": "top_down_bias_not_strongly_bullish",
                "label": "Top-down bias not strongly bullish",
                "status": "failed" if htf_score >= 0.35 else "pending" if htf_score > 0.10 else "satisfied",
                "current": round(htf_score, 4),
                "threshold": 0.35,
            },
            {
                "key": "flow_not_strongly_opposed",
                "label": "Cash flow not strongly opposed",
                "status": "failed" if flow_score >= 0.30 else "pending" if flow_score > 0.10 else "satisfied",
                "current": round(flow_score, 4),
                "threshold": 0.30,
            },
            {
                "key": "smc_cross_validation_not_bullish",
                "label": "SMC cross-validation not bullish",
                "status": (
                    "failed" if library_direction == "bullish" and library_agreement >= 0.67
                    else "satisfied" if library_direction == "bearish" and library_agreement >= 0.50
                    else "pending"
                ),
                "current": f"{library_direction}:{round(library_agreement, 2)}",
                "threshold": "not_bullish_consensus",
            },
        ]

    conditions.append({
        "key": "strict_setup_confirmation",
        "label": "Strict setup confirmation",
        "status": "satisfied" if setup_confirmed else "pending",
        "current": "confirmed" if setup_confirmed else "not_confirmed",
        "threshold": "confirmed",
    })

    satisfied = sum(1 for item in conditions if item["status"] == "satisfied")
    failed = sum(1 for item in conditions if item["status"] == "failed")
    pending = len(conditions) - satisfied - failed
    if not conditions:
        state = "inactive"
    elif failed:
        state = "blocked"
    elif pending:
        state = "pending"
    else:
        state = "confirmed"

    return {
        "state": state,
        "conditions": conditions,
        "satisfied": satisfied,
        "pending": pending,
        "failed": failed,
        "total": len(conditions),
    }


def _execution_plan(
    technical: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    confidence: dict[str, Any],
    adversarial: dict[str, Any],
    edge: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    candidate = str(technical.get("candidate") or "none")
    strict_side = "long" if candidate == "long_setup" else "short" if candidate == "short_setup" else None
    edge_direction = str(edge.get("direction") or "neutral")
    lean_side = "long" if edge_direction == "bullish" else "short" if edge_direction == "bearish" else None
    side = strict_side or lean_side
    setup_confirmed = strict_side is not None

    spot = technical.get("indicative_spot") or {}
    analysis_reference = technical.get("analysis_reference") or {}
    price = _number(analysis_reference.get("price"), _number(spot.get("price"), 0.0))
    atr = max(0.0, _number(technical.get("atr_reference"), 0.0))
    frames = technical.get("frames") or {}
    five = frames.get("5m") or {}
    ema_fast = _number(five.get("ema_fast"), price)
    extension_atr = abs(price - ema_fast) / atr if atr > 0 and price > 0 else 0.0
    calibrated = _number(confidence.get("calibrated_confidence"), 0.0)
    primary = hypotheses[0] if hypotheses else {}
    primary_direction = str(primary.get("direction") or "none")
    primary_name = str(primary.get("name") or "none")
    edge_strength = _number(edge.get("strength"), 0.0)

    activation = _activation_state(
        side,
        technical,
        perception,
        setup_confirmed=setup_confirmed,
    )

    dominant_scenario = max(scenarios, key=lambda item: _number(item.get("weight")), default={})
    same_direction_weight = max(
        (
            _number(item.get("weight"))
            for item in scenarios
            if str(item.get("direction")) == edge_direction
        ),
        default=0.0,
    )
    dominant_direction = str(dominant_scenario.get("direction") or "neutral")
    dominant_weight = _number(dominant_scenario.get("weight"), 0.0)
    scenario_conflict = bool(
        side
        and edge_direction in {"bullish", "bearish"}
        and dominant_direction in {"bullish", "bearish"}
        and dominant_direction != edge_direction
        and dominant_weight >= 0.45
        and dominant_weight >= same_direction_weight + 0.08
    )

    reasons: list[str] = []
    action = "STAND_DOWN"
    htf_score = _number(edge.get("higher_timeframe_score"), 0.0)
    htf_conflict = bool(
        side == "long" and htf_score <= -0.25
        or side == "short" and htf_score >= 0.25
        or edge.get("higher_timeframe_conflict")
    )

    if side is None:
        reasons.append("no_directional_edge")
    elif adversarial.get("veto"):
        reasons.append("adversarial_veto")
    elif htf_conflict:
        action = "WAIT_CONFIRMATION"
        reasons.append("higher_timeframe_bias_conflicts_entry")
    elif scenario_conflict:
        action = "WAIT_CONFIRMATION"
        reasons.append("dominant_scenario_opposes_directional_edge")
    elif activation.get("failed", 0) > 0:
        action = "WAIT_CONFIRMATION"
        reasons.append("activation_condition_failed")
    elif not setup_confirmed:
        if edge_strength < 0.24:
            action = "BIAS_ONLY"
            reasons.append("edge_too_weak_for_trigger")
        elif regime.get("label") in {"transition", "range_rotation", "compression_range"}:
            action = "WAIT_TRIGGER"
            reasons.append("directional_lean_needs_regime_confirmation")
        elif calibrated < min_confidence:
            action = "WAIT_TRIGGER"
            reasons.append("directional_lean_below_confidence_threshold")
        else:
            action = "WAIT_TRIGGER"
            reasons.append("directional_edge_present_but_entry_trigger_missing")
    elif primary_direction not in {side, "none"}:
        reasons.append("dominant_hypothesis_opposes_entry")
    elif calibrated < min_confidence:
        action = "WAIT"
        reasons.append("confidence_below_threshold")
    elif regime.get("label") in {"transition", "range_rotation"} and primary_name == "trend_continuation":
        action = "WAIT_CONFIRMATION"
        reasons.append("regime_requires_confirmation")
    elif extension_atr >= 0.80:
        action = "WAIT_PULLBACK"
        reasons.append("price_extended_from_5m_fast_ema")
    else:
        action = "ENTER_NOW"
        reasons.append("cognitive_gates_passed")

    swing_high = _number(technical.get("swing_high_reference"), 0.0)
    swing_low = _number(technical.get("swing_low_reference"), 0.0)
    close5 = _number(five.get("close"), price)

    trigger_level = None
    if side == "long":
        if close5 <= ema_fast and ema_fast > 0:
            trigger_level = round(ema_fast, 4)
        elif not setup_confirmed and swing_high > price:
            trigger_level = round(swing_high, 4)
        invalidation = technical.get("swing_low_reference")
        entry_zone = (
            [round(price - 0.22 * atr, 4), round(price + 0.05 * atr, 4)]
            if price > 0 and atr > 0 else None
        )
    elif side == "short":
        if close5 >= ema_fast and ema_fast > 0:
            trigger_level = round(ema_fast, 4)
        elif not setup_confirmed and swing_low > 0 and swing_low < price:
            trigger_level = round(swing_low, 4)
        invalidation = technical.get("swing_high_reference")
        entry_zone = (
            [round(price - 0.05 * atr, 4), round(price + 0.22 * atr, 4)]
            if price > 0 and atr > 0 else None
        )
    else:
        invalidation = None
        entry_zone = None

    return {
        "action": action,
        "side": side,
        "setup_confirmed": setup_confirmed,
        "source": "strict_setup" if setup_confirmed else "directional_edge" if side else "none",
        "entry_zone": entry_zone,
        "trigger_level": trigger_level,
        "trigger_state": activation.get("state"),
        "activation": activation,
        "activation_conditions": activation.get("conditions", []),
        "invalidation_reference": invalidation,
        "extension_atr": round(extension_atr, 4),
        "primary_hypothesis": primary_name,
        "dominant_scenario": dominant_scenario.get("name"),
        "dominant_scenario_direction": dominant_direction,
        "dominant_scenario_weight": round(dominant_weight, 4),
        "scenario_conflict": scenario_conflict,
        "edge_score": edge.get("score"),
        "edge_strength": edge.get("strength"),
        "reasons": reasons,
        "execution_allowed": False,
    }


def build_cognitive_state(
    technical: dict[str, Any],
    macro: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
    min_confidence: float = 0.58,
) -> dict[str, Any]:
    """Build the complete inspectable XAU cognition stack."""

    min_confidence = _clip(float(min_confidence), 0.50, 0.90)
    data_quality, quality_issues, sensors = _data_quality(technical)
    perception = _perception(technical)
    edge = _directional_edge(technical, macro, perception, data_quality)
    regime = _regime(perception, technical, macro, data_quality)
    hypotheses = _hypotheses(technical, macro, perception, regime, edge)
    market_state = build_market_state_vector(technical, macro)
    memory_state = _memory_adjustment(memory)
    adversarial = _adversarial_review(
        technical,
        macro,
        perception,
        regime,
        hypotheses,
        data_quality,
    )
    confidence = _confidence(
        data_quality,
        regime,
        hypotheses,
        macro,
        adversarial,
        memory_state,
    )
    adaptive_threshold = _adaptive_entry_threshold(
        min_confidence,
        memory_state,
    )
    scenarios = _scenario_paths(technical, hypotheses, edge)
    plan = _execution_plan(
        technical,
        perception,
        regime,
        hypotheses,
        scenarios,
        confidence,
        adversarial,
        edge,
        adaptive_threshold["effective"],
    )

    action = str(plan.get("action") or "STAND_DOWN")
    if action == "ENTER_NOW":
        decision = "eligible"
    elif action in {"WAIT", "WAIT_PULLBACK", "WAIT_CONFIRMATION", "WAIT_TRIGGER", "BIAS_ONLY"}:
        decision = "wait"
    elif technical.get("candidate") in {"long_setup", "short_setup"}:
        decision = "veto"
    else:
        decision = "observe"

    return {
        "version": COGNITION_VERSION,
        "market_state": market_state,
        "data_quality": {
            "score": data_quality,
            "issues": quality_issues,
            "sensors": sensors,
        },
        "perception": perception,
        "directional_edge": edge,
        "regime": regime,
        "hypotheses": hypotheses,
        "scenarios": scenarios,
        "memory": memory_state,
        "adversarial": adversarial,
        "confidence": confidence,
        "execution_plan": plan,
        "meta_controller": {
            "decision": decision,
            "paper_entry_allowed": decision == "eligible",
            "min_confidence": adaptive_threshold["effective"],
            "threshold_adaptation": adaptive_threshold,
            "dominant_hypothesis": plan.get("primary_hypothesis"),
            "live_execution_allowed": False,
        },
    }
