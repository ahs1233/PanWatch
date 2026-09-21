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

COGNITION_VERSION = "2.0.0"


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
) -> list[dict[str, Any]]:
    candidate = str(technical.get("candidate") or "none")
    setup_dir = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    macro_bias = int(max(-1, min(1, _number(macro.get("bias"), 0.0))))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    pressure = _number(perception.get("directional_pressure"))
    acceleration = _number(perception.get("acceleration"))
    breakout = str(perception.get("breakout") or "none")
    rsi5 = _number(perception.get("rsi_5m"), 50.0)
    probs = regime.get("probabilities") or {}
    session = _market_session(technical.get("observed_at"))

    continuation = (
        0.25
        + (1.0 if setup_dir else -0.30)
        + 0.80 * abs(pressure)
        + 1.20 * _number(probs.get("trend_bull" if setup_dir > 0 else "trend_bear"))
        + 0.70 * _number(probs.get("breakout_expansion"))
    )
    if setup_dir and macro_bias == setup_dir:
        continuation += 0.50 * max(0.30, macro_conf)
    elif setup_dir and macro_bias == -setup_dir:
        continuation -= 0.70 * max(0.30, macro_conf)

    mean_reversion = (
        0.15
        + 1.20 * _number(probs.get("range_rotation"))
        + 0.90 * _number(probs.get("compression_range"))
        + (0.70 if rsi5 >= 70 or rsi5 <= 30 else 0.0)
        + (0.35 if setup_dir and acceleration * setup_dir < 0 else 0.0)
    )

    liquidity_sweep = (
        0.10
        + (0.65 if breakout in {"up", "down"} else 0.0)
        + 0.70 * _number(probs.get("transition"))
        + (0.40 if rsi5 >= 75 or rsi5 <= 25 else 0.0)
    )

    failed_breakout = (
        0.05
        + (0.75 if breakout in {"up", "down"} else 0.0)
        + 0.80 * _number(probs.get("transition"))
        + (0.45 if setup_dir and acceleration * setup_dir < 0 else 0.0)
    )

    volatility_expansion = (
        0.05
        + 1.45 * _number(probs.get("breakout_expansion"))
        + (0.35 if breakout in {"up", "down"} else 0.0)
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
    confidence_adjustment = (0.035 * edge + 0.025 * pf_edge) * strength

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
        "historical_probability": historical_probability,
        "sample_count": effective_samples,
        "brier_score": memory_state.get("brier_score"),
        "expected_calibration_error": memory_state.get("expected_calibration_error"),
    }


def _execution_plan(
    technical: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    confidence: dict[str, Any],
    adversarial: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    candidate = str(technical.get("candidate") or "none")
    side = "long" if candidate == "long_setup" else "short" if candidate == "short_setup" else None
    spot = technical.get("indicative_spot") or {}
    analysis_reference = technical.get("analysis_reference") or {}
    price = _number(
        analysis_reference.get("price"),
        _number(spot.get("price"), 0.0),
    )
    atr = max(0.0, _number(technical.get("atr_reference"), 0.0))
    five = (technical.get("frames") or {}).get("5m") or {}
    ema_fast = _number(five.get("ema_fast"), price)
    extension_atr = abs(price - ema_fast) / atr if atr > 0 and price > 0 else 0.0
    calibrated = _number(confidence.get("calibrated_confidence"), 0.0)
    primary = hypotheses[0] if hypotheses else {}
    primary_direction = str(primary.get("direction") or "none")
    primary_name = str(primary.get("name") or "none")

    reasons: list[str] = []
    action = "STAND_DOWN"
    if side is None:
        reasons.append("no_directional_setup")
    elif adversarial.get("veto"):
        reasons.append("adversarial_veto")
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

    if side and price > 0 and atr > 0:
        if side == "long":
            entry_zone = [round(price - 0.22 * atr, 4), round(price + 0.05 * atr, 4)]
            invalidation = technical.get("swing_low_reference")
        else:
            entry_zone = [round(price - 0.05 * atr, 4), round(price + 0.22 * atr, 4)]
            invalidation = technical.get("swing_high_reference")
    else:
        entry_zone = None
        invalidation = None

    return {
        "action": action,
        "side": side,
        "entry_zone": entry_zone,
        "invalidation_reference": invalidation,
        "extension_atr": round(extension_atr, 4),
        "primary_hypothesis": primary_name,
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
    regime = _regime(perception, technical, macro, data_quality)
    hypotheses = _hypotheses(technical, macro, perception, regime)
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
    plan = _execution_plan(
        technical,
        perception,
        regime,
        hypotheses,
        confidence,
        adversarial,
        adaptive_threshold["effective"],
    )

    action = str(plan.get("action") or "STAND_DOWN")
    if action == "ENTER_NOW":
        decision = "eligible"
    elif action in {"WAIT", "WAIT_PULLBACK", "WAIT_CONFIRMATION"}:
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
        "regime": regime,
        "hypotheses": hypotheses,
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
